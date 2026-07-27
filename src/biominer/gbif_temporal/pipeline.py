from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.gbif_temporal.parser import (
    PARSER_VERSION,
    TemporalDerivation,
    derive_temporal_components,
)


SCHEMA_VERSION = "biominer-gbif-media-temporal/v1"
REQUIRED_SOURCE_COLUMNS = ("gbifID", "eventDate", "year", "month", "day")
DERIVED_FIELDS = (
    pa.field("derived_year", pa.int16()),
    pa.field("derived_month", pa.int8()),
    pa.field("derived_day", pa.int8()),
    pa.field("temporal_derivation_method", pa.string()),
)
AUDIT_SCHEMA = pa.schema(
    [
        ("temporal_derivation_id", pa.string()),
        ("source_artifact_sha256", pa.string()),
        ("gbifID", pa.string()),
        ("eventDate", pa.string()),
        ("source_year", pa.string()),
        ("source_month", pa.string()),
        ("source_day", pa.string()),
        ("derived_year", pa.int16()),
        ("derived_month", pa.int8()),
        ("derived_day", pa.int8()),
        ("temporal_derivation_method", pa.string()),
        ("temporal_derivation_status", pa.string()),
        ("temporal_derived_components", pa.string()),
        ("interval_start", pa.string()),
        ("interval_end", pa.string()),
        ("temporal_parser_version", pa.string()),
        ("exclusion_reason", pa.string()),
        ("source_media_rows", pa.int64()),
    ]
)


def publish_temporal_enrichment(
    *,
    source: str | Path,
    source_manifest: str | Path,
    output_directory: str | Path,
    expected_source_sha256: str | None = (
        "c96505f410723da57db4bd11bcffdc4e72be59ee59ecbaad8f4af8677229e57f"
    ),
    expected_source_rows: int | None = 16_612_063,
    expected_derived_year_rows: int | None = 2_360,
    expected_derived_month_rows: int | None = 4_941,
    expected_derived_day_rows: int | None = 18_741,
    expected_pre_1960_excluded_rows: int | None = 2_236,
    batch_rows: int = 50_000,
    duckdb_memory_limit: str = "24GB",
    duckdb_threads: int = 8,
) -> dict[str, Any]:
    """Publish a create-only temporal audit, enriched Parquet, and DuckDB."""

    if batch_rows <= 0 or duckdb_threads <= 0:
        raise ValueError("batch_rows and duckdb_threads must be positive")
    source_path = Path(source).resolve()
    source_manifest_path = Path(source_manifest).resolve()
    destination = Path(output_directory).resolve()
    for path in (source_path, source_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {destination}")

    source_sha = "sha256:" + _sha256(source_path)
    expected_sha = _normalize_sha256(expected_source_sha256)
    if expected_sha is not None and source_sha != expected_sha:
        raise ValueError(
            f"source checksum mismatch: expected {expected_sha}, found {source_sha}"
        )
    source_file = pq.ParquetFile(source_path)
    _require_columns(source_file.schema_arrow)
    source_rows = source_file.metadata.num_rows
    if expected_source_rows is not None and source_rows != expected_source_rows:
        raise ValueError(
            f"source row count mismatch: expected {expected_source_rows}, found {source_rows}"
        )
    collisions = set(source_file.schema_arrow.names) & {
        field.name for field in DERIVED_FIELDS
    }
    if collisions:
        raise ValueError(f"source already contains temporal columns: {sorted(collisions)}")

    audit_by_gbif, scan_counts = _collect_derivations(
        source_file=source_file,
        source_sha=source_sha,
        batch_rows=batch_rows,
    )
    expected_counts = {
        "derived_year_rows": expected_derived_year_rows,
        "derived_month_rows": expected_derived_month_rows,
        "derived_day_rows": expected_derived_day_rows,
        "excluded_pre_1960_rows": expected_pre_1960_excluded_rows,
    }
    for name, expected in expected_counts.items():
        if expected is not None and scan_counts[name] != expected:
            raise ValueError(
                f"{name} mismatch: expected {expected}, found {scan_counts[name]}"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    try:
        return _publish_staging(
            source_path=source_path,
            source_manifest_path=source_manifest_path,
            source_file=source_file,
            source_sha=source_sha,
            source_rows=source_rows,
            destination=destination,
            staging=staging,
            audit_by_gbif=audit_by_gbif,
            scan_counts=scan_counts,
            batch_rows=batch_rows,
            duckdb_memory_limit=duckdb_memory_limit,
            duckdb_threads=duckdb_threads,
        )
    except Exception as exc:
        _write_json(
            staging / "failure.json",
            {
                "schema_version": SCHEMA_VERSION,
                "failed_at": _timestamp(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "source_sha256": source_sha,
                "git_commit": _git_revision(),
            },
        )
        failed = staging.with_suffix(".failed")
        staging.replace(failed)
        raise RuntimeError(f"temporal publication failed; evidence retained at {failed}") from exc


def _collect_derivations(
    *,
    source_file: pq.ParquetFile,
    source_sha: str,
    batch_rows: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    audit_by_gbif: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter(
        {
            "candidate_rows": 0,
            "derived_year_rows": 0,
            "derived_month_rows": 0,
            "derived_day_rows": 0,
            "excluded_pre_1960_rows": 0,
        }
    )
    for batch in source_file.iter_batches(
        batch_size=batch_rows,
        columns=list(REQUIRED_SOURCE_COLUMNS),
        use_threads=True,
    ):
        selected = _candidate_batch(batch)
        if selected.num_rows == 0:
            continue
        values = {
            name: selected.column(index).to_pylist()
            for index, name in enumerate(REQUIRED_SOURCE_COLUMNS)
        }
        for offset in range(selected.num_rows):
            gbif_id = _trimmed(values["gbifID"][offset])
            if gbif_id is None:
                raise ValueError("temporal candidate has a blank gbifID")
            event_date = _optional(values["eventDate"][offset])
            source_year = _optional(values["year"][offset])
            source_month = _optional(values["month"][offset])
            source_day = _optional(values["day"][offset])
            result = derive_temporal_components(
                event_date=event_date,
                year=source_year,
                month=source_month,
                day=source_day,
            )
            counts["candidate_rows"] += 1
            counts["derived_year_rows"] += int(result.derived_year is not None)
            counts["derived_month_rows"] += int(result.derived_month is not None)
            counts["derived_day_rows"] += int(result.derived_day is not None)
            counts["excluded_pre_1960_rows"] += int(
                result.status == "excluded_pre_1960"
            )
            identity_values = (
                event_date,
                source_year,
                source_month,
                source_day,
            )
            existing = audit_by_gbif.get(gbif_id)
            if existing is not None:
                existing_values = (
                    existing["eventDate"],
                    existing["source_year"],
                    existing["source_month"],
                    existing["source_day"],
                )
                if existing_values != identity_values:
                    raise ValueError(
                        f"gbifID {gbif_id} has inconsistent temporal values"
                    )
                existing["source_media_rows"] += 1
                continue
            audit_by_gbif[gbif_id] = _audit_row(
                source_sha=source_sha,
                gbif_id=gbif_id,
                event_date=event_date,
                source_year=source_year,
                source_month=source_month,
                source_day=source_day,
                result=result,
            )
    counts["candidate_occurrences"] = len(audit_by_gbif)
    counts["excluded_pre_1960_occurrences"] = sum(
        row["temporal_derivation_status"] == "excluded_pre_1960"
        for row in audit_by_gbif.values()
    )
    return audit_by_gbif, dict(counts)


def _publish_staging(
    *,
    source_path: Path,
    source_manifest_path: Path,
    source_file: pq.ParquetFile,
    source_sha: str,
    source_rows: int,
    destination: Path,
    staging: Path,
    audit_by_gbif: dict[str, dict[str, Any]],
    scan_counts: dict[str, int],
    batch_rows: int,
    duckdb_memory_limit: str,
    duckdb_threads: int,
) -> dict[str, Any]:
    audit_path = staging / "temporal_derivations.parquet"
    audit_rows = sorted(audit_by_gbif.values(), key=lambda row: row["gbifID"])
    pq.write_table(
        pa.Table.from_pylist(audit_rows, schema=AUDIT_SCHEMA),
        audit_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
        row_group_size=batch_rows,
    )

    parquet_path = staging / "gbif_media_temporal.parquet"
    output_schema = source_file.schema_arrow
    for field in DERIVED_FIELDS:
        output_schema = output_schema.append(field)
    writer = pq.ParquetWriter(
        parquet_path,
        output_schema,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    source_gbif_index = source_file.schema_arrow.get_field_index("gbifID")
    output_rows = 0
    excluded_rows = 0
    try:
        for batch in source_file.iter_batches(batch_size=batch_rows, use_threads=True):
            gbif_ids = batch.column(source_gbif_index).to_pylist()
            keep: list[bool] = []
            years: list[int | None] = []
            months: list[int | None] = []
            days: list[int | None] = []
            methods: list[str | None] = []
            for value in gbif_ids:
                gbif_id = _trimmed(value)
                row = audit_by_gbif.get(gbif_id or "")
                if row is not None and row["temporal_derivation_status"] == "excluded_pre_1960":
                    keep.append(False)
                    excluded_rows += 1
                    continue
                keep.append(True)
                years.append(row["derived_year"] if row is not None else None)
                months.append(row["derived_month"] if row is not None else None)
                days.append(row["derived_day"] if row is not None else None)
                methods.append(
                    row["temporal_derivation_method"] if row is not None else None
                )
            mask = pa.array(keep, type=pa.bool_())
            filtered = pa.RecordBatch.from_arrays(
                [pc.filter(column, mask) for column in batch.columns],
                schema=batch.schema,
            )
            table = pa.Table.from_batches([filtered])
            table = table.append_column(DERIVED_FIELDS[0], pa.array(years, pa.int16()))
            table = table.append_column(DERIVED_FIELDS[1], pa.array(months, pa.int8()))
            table = table.append_column(DERIVED_FIELDS[2], pa.array(days, pa.int8()))
            table = table.append_column(DERIVED_FIELDS[3], pa.array(methods, pa.string()))
            if table.num_rows:
                writer.write_table(table, row_group_size=batch_rows)
            output_rows += table.num_rows
    finally:
        writer.close()

    if excluded_rows != scan_counts["excluded_pre_1960_rows"]:
        raise RuntimeError(
            "pre-1960 exclusion mismatch: "
            f"scan={scan_counts['excluded_pre_1960_rows']}, publish={excluded_rows}"
        )
    if source_rows != output_rows + excluded_rows:
        raise RuntimeError("temporal output row reconciliation failed")

    database_path = staging / "gbif_media_temporal.duckdb"
    database = _build_database(
        database_path=database_path,
        parquet_path=parquet_path,
        audit_path=audit_path,
        memory_limit=duckdb_memory_limit,
        threads=duckdb_threads,
    )
    parquet_inventory = _parquet_inventory(parquet_path)
    audit_inventory = _parquet_inventory(audit_path)
    validation = {
        "source_sha256_recalculated": True,
        "source_row_count_matches_metadata": source_rows
        == source_file.metadata.num_rows,
        "row_count_reconciled": source_rows == output_rows + excluded_rows,
        "database_row_count_matches_parquet": database["media_row_count"]
        == output_rows,
        "database_audit_count_matches_parquet": database["audit_row_count"]
        == len(audit_rows),
        "database_column_names_match_parquet": [
            row["name"] for row in database["media_columns"]
        ]
        == output_schema.names,
        "original_column_names_and_types_unchanged": pa.schema(
            list(output_schema)[: len(source_file.schema_arrow)]
        )
        == source_file.schema_arrow,
        "parquet_row_groups_complete": parquet_inventory["row_groups_complete"],
        "audit_row_groups_complete": audit_inventory["row_groups_complete"],
        "pre_1960_rows_excluded": excluded_rows
        == scan_counts["excluded_pre_1960_rows"],
        "occurrence_temporal_values_consistent": True,
    }
    if not all(validation.values()):
        raise RuntimeError(f"temporal validation failed: {validation}")

    status_counts = dict(
        sorted(Counter(row["temporal_derivation_status"] for row in audit_rows).items())
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "generated_at": _timestamp(),
        "git_commit": _git_revision(),
        "baseline_maturity": "legacy_v3_migration_not_ground_zero_production",
        "policy": {
            "event_date_only": True,
            "interval_policy": "start_boundary",
            "timezone_policy": "preserve_lexical_calendar_date_without_utc_shift",
            "original_temporal_fields_unchanged": True,
            "pre_1960_policy": "exclude_and_retain_in_audit",
            "year_cutoff_inclusive": 1960,
        },
        "input": {
            "source_path": str(source_path),
            "source_sha256": source_sha,
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": "sha256:" + _sha256(source_manifest_path),
            "source_rows": source_rows,
            "source_columns": len(source_file.schema_arrow),
        },
        "counts": {
            "source_rows": source_rows,
            **scan_counts,
            "audit_rows": len(audit_rows),
            "excluded_pre_1960_rows": excluded_rows,
            "output_rows": output_rows,
            "output_columns": len(output_schema),
            "audit_status_counts": status_counts,
        },
        "artifacts": {
            parquet_path.name: parquet_inventory,
            audit_path.name: audit_inventory,
            database_path.name: {
                "path": database_path.name,
                "physical_bytes": database_path.stat().st_size,
                "physical_sha256": "sha256:" + _sha256(database_path),
                **database,
            },
        },
        "validation": validation,
        "manifest_policy": {"written_last": True, "create_only": True},
    }
    _write_json(staging / "manifest.json", manifest)
    staging.replace(destination)
    return manifest


def _candidate_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    indexes = {name: index for index, name in enumerate(REQUIRED_SOURCE_COLUMNS)}
    missing = _blank_mask(batch.column(indexes["year"]))
    missing = pc.or_(missing, _blank_mask(batch.column(indexes["month"])))
    missing = pc.or_(missing, _blank_mask(batch.column(indexes["day"])))
    has_event = pc.invert(_blank_mask(batch.column(indexes["eventDate"])))
    mask = pc.and_(missing, has_event)
    return pa.RecordBatch.from_arrays(
        [pc.filter(column, mask) for column in batch.columns],
        schema=batch.schema,
    )


def _blank_mask(column: pa.Array) -> pa.Array:
    trimmed = pc.utf8_trim_whitespace(column)
    return pc.or_(pc.is_null(column), pc.fill_null(pc.equal(trimmed, ""), False))


def _audit_row(
    *,
    source_sha: str,
    gbif_id: str,
    event_date: str | None,
    source_year: str | None,
    source_month: str | None,
    source_day: str | None,
    result: TemporalDerivation,
) -> dict[str, Any]:
    identity = canonical_semantic_fingerprint(
        {
            "contract": SCHEMA_VERSION,
            "parser_version": PARSER_VERSION,
            "source_artifact_sha256": source_sha,
            "gbifID": gbif_id,
            "eventDate": event_date,
            "year": source_year,
            "month": source_month,
            "day": source_day,
        }
    )
    return {
        "temporal_derivation_id": identity,
        "source_artifact_sha256": source_sha,
        "gbifID": gbif_id,
        "eventDate": event_date,
        "source_year": source_year,
        "source_month": source_month,
        "source_day": source_day,
        "derived_year": result.derived_year,
        "derived_month": result.derived_month,
        "derived_day": result.derived_day,
        "temporal_derivation_method": result.method,
        "temporal_derivation_status": result.status,
        "temporal_derived_components": result.derived_components,
        "interval_start": result.interval_start,
        "interval_end": result.interval_end,
        "temporal_parser_version": PARSER_VERSION,
        "exclusion_reason": result.exclusion_reason,
        "source_media_rows": 1,
    }


def _build_database(
    *,
    database_path: Path,
    parquet_path: Path,
    audit_path: Path,
    memory_limit: str,
    threads: int,
) -> dict[str, Any]:
    temporary = database_path.parent / "duckdb_tmp"
    temporary.mkdir()
    con = duckdb.connect(str(database_path))
    try:
        escaped = str(temporary).replace("'", "''")
        con.execute(f"SET temp_directory='{escaped}'")
        con.execute(f"SET memory_limit='{memory_limit}'")
        con.execute(f"SET threads={threads}")
        con.execute("CREATE TABLE gbif_media AS SELECT * FROM read_parquet(?)", [str(parquet_path)])
        con.execute(
            "CREATE TABLE temporal_derivations AS SELECT * FROM read_parquet(?)",
            [str(audit_path)],
        )
        con.execute("CREATE INDEX idx_gbif_media_gbif_id ON gbif_media(gbifID)")
        if "media_identifier" in pq.ParquetFile(parquet_path).schema_arrow.names:
            con.execute(
                "CREATE INDEX idx_gbif_media_media_identifier ON gbif_media(media_identifier)"
            )
        con.execute("CREATE INDEX idx_gbif_media_derived_year ON gbif_media(derived_year)")
        con.execute(
            "CREATE INDEX idx_temporal_derivations_gbif_id ON temporal_derivations(gbifID)"
        )
        con.execute("ANALYZE gbif_media")
        con.execute("ANALYZE temporal_derivations")
        con.execute("CHECKPOINT")
        media_rows = int(con.execute("SELECT count(*) FROM gbif_media").fetchone()[0])
        audit_rows = int(
            con.execute("SELECT count(*) FROM temporal_derivations").fetchone()[0]
        )
        media_columns = [
            {"name": str(row[0]), "type": str(row[1])}
            for row in con.execute("DESCRIBE gbif_media").fetchall()
        ]
        audit_columns = [
            {"name": str(row[0]), "type": str(row[1])}
            for row in con.execute("DESCRIBE temporal_derivations").fetchall()
        ]
        indexes = [
            {
                "index_name": str(row[0]),
                "table_name": str(row[1]),
                "expressions": str(row[2]),
                "sql": str(row[3]),
            }
            for row in con.execute(
                "SELECT index_name, table_name, expressions, sql "
                "FROM duckdb_indexes() ORDER BY index_name"
            ).fetchall()
        ]
    finally:
        con.close()
        shutil.rmtree(temporary, ignore_errors=True)
    return {
        "media_row_count": media_rows,
        "audit_row_count": audit_rows,
        "media_columns": media_columns,
        "audit_columns": audit_columns,
        "indexes": indexes,
    }


def _parquet_inventory(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    row_groups = [
        parquet.metadata.row_group(index).num_rows
        for index in range(parquet.metadata.num_row_groups)
    ]
    return {
        "path": path.name,
        "physical_bytes": path.stat().st_size,
        "physical_sha256": "sha256:" + _sha256(path),
        "row_count": parquet.metadata.num_rows,
        "column_count": len(parquet.schema_arrow),
        "columns": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in parquet.schema_arrow
        ],
        "row_group_count": parquet.metadata.num_row_groups,
        "row_group_rows": row_groups,
        "row_groups_complete": sum(row_groups) == parquet.metadata.num_rows
        and all(value > 0 for value in row_groups),
    }


def _require_columns(schema: pa.Schema) -> None:
    missing = [name for name in REQUIRED_SOURCE_COLUMNS if name not in schema.names]
    if missing:
        raise ValueError(f"source is missing required columns: {missing}")
    for name in REQUIRED_SOURCE_COLUMNS:
        if not pa.types.is_string(schema.field(name).type):
            raise ValueError(f"source column {name} must be string")


def _normalize_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized.startswith("sha256:"):
        normalized = "sha256:" + normalized
    if len(normalized) != 71:
        raise ValueError("expected_source_sha256 must be a SHA-256 digest")
    try:
        int(normalized[7:], 16)
    except ValueError as exc:
        raise ValueError("expected_source_sha256 must be a SHA-256 digest") from exc
    return normalized


def _trimmed(value: object | None) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _optional(value: object | None) -> str | None:
    return None if value is None else str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "AUDIT_SCHEMA",
    "DERIVED_FIELDS",
    "SCHEMA_VERSION",
    "publish_temporal_enrichment",
]
