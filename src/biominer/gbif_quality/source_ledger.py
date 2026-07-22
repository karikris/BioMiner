from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


SOURCE_LEDGER_VERSION = "biominer-gbif-media-source-ledger/v1"
SOURCE_LEDGER_SCHEMA = pa.schema(
    [
        ("ledger_version", pa.string()),
        ("source_snapshot_id", pa.string()),
        ("source_file", pa.string()),
        ("source_sort_position", pa.int64()),
        ("source_row_id", pa.string()),
        ("gbifID", pa.string()),
        ("media_join_status", pa.string()),
        ("v3_funnel_status", pa.string()),
        ("exclusion_reason", pa.string()),
        ("local_quality_status", pa.string()),
    ]
)


@dataclass(frozen=True, slots=True)
class SourceLedgerResult:
    output_directory: Path
    ledger_path: Path
    manifest: dict[str, object]


def publish_source_media_ledger(
    *,
    joined_parquet: str | Path,
    normalized_parquet: str | Path,
    output_directory: str | Path,
    source_snapshot_id: str,
    expected_counts: dict[str, int],
    code_commit: str,
    memory_limit: str = "4GB",
    temp_directory: str | Path | None = None,
) -> SourceLedgerResult:
    """Assign every raw joined media assertion one deterministic funnel status."""

    joined = Path(joined_parquet).resolve()
    normalized = Path(normalized_parquet).resolve()
    destination = Path(output_directory).resolve()
    for path in (joined, normalized):
        if not path.is_file():
            raise FileNotFoundError(path)
    if destination.exists():
        raise FileExistsError(destination)
    required = {
        "raw_multimedia_rows",
        "pre_1960_media_rows_excluded",
        "legacy_cohort_rows_excluded",
        "explicit_rights_rows_excluded",
        "v3_media_rows",
    }
    if set(expected_counts) < required:
        raise ValueError(f"expected_counts missing: {sorted(required - set(expected_counts))}")
    if not source_snapshot_id or not code_commit:
        raise ValueError("source_snapshot_id and code_commit are required")
    joined_schema = pq.ParquetFile(joined).schema_arrow
    normalized_schema = pq.ParquetFile(normalized).schema_arrow
    for field in ("gbifID", "year"):
        if field not in joined_schema.names:
            raise ValueError(f"joined Parquet lacks {field}")
    for field in (
        "identifiedBy",
        "identificationVerificationStatus",
        "media_identifier",
        "media_license",
    ):
        if field not in normalized_schema.names:
            raise ValueError(f"normalized Parquet lacks {field}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    temporary = (
        Path(temp_directory).resolve()
        if temp_directory is not None
        else staging / "duckdb_tmp"
    )
    temporary.mkdir(parents=True, exist_ok=True)
    output = staging / "source_media_status.parquet"
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 1")
        connection.execute(f"SET memory_limit = {_literal(memory_limit)}")
        connection.execute(f"SET temp_directory = {_literal(str(temporary))}")
        connection.execute("SET preserve_insertion_order = true")
        query = _ledger_query(joined, normalized, source_snapshot_id)
        connection.execute(
            f"COPY ({query}) TO {_literal(str(output))} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"
        )
        counts = _ledger_counts(connection, output)
    except Exception:
        connection.close()
        shutil.rmtree(staging, ignore_errors=True)
        if temp_directory is None:
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        try:
            connection.close()
        except Exception:
            pass
    if temp_directory is None:
        shutil.rmtree(temporary, ignore_errors=True)

    expected = {
        "total_rows": int(expected_counts["raw_multimedia_rows"]),
        "excluded_pre_1960": int(expected_counts["pre_1960_media_rows_excluded"]),
        "excluded_outside_cohort": int(expected_counts["legacy_cohort_rows_excluded"]),
        "excluded_explicit_rights": int(expected_counts["explicit_rights_rows_excluded"]),
        "retained_v3": int(expected_counts["v3_media_rows"]),
        "resolved_occurrence": int(expected_counts["raw_multimedia_rows"]),
    }
    validation = {
        "all_source_rows_have_status": counts["total_rows"]
        == counts["status_rows"]
        == expected["total_rows"],
        "all_source_rows_resolve_to_occurrence": counts["resolved_occurrence"]
        == expected["resolved_occurrence"],
        "pre_1960_count_matches": counts["excluded_pre_1960"]
        == expected["excluded_pre_1960"],
        "cohort_exclusion_count_matches": counts["excluded_outside_cohort"]
        == expected["excluded_outside_cohort"],
        "rights_exclusion_count_matches": counts["excluded_explicit_rights"]
        == expected["excluded_explicit_rights"],
        "retained_v3_count_matches": counts["retained_v3"]
        == expected["retained_v3"],
        "funnel_partition_reconciles": counts["total_rows"]
        == counts["excluded_pre_1960"]
        + counts["excluded_outside_cohort"]
        + counts["excluded_explicit_rights"]
        + counts["retained_v3"],
        "source_row_ids_unique": counts["distinct_source_row_ids"]
        == counts["total_rows"],
        "schema_matches": pq.ParquetFile(output).schema_arrow.equals(
            SOURCE_LEDGER_SCHEMA
        ),
    }
    if not all(validation.values()):
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(
            f"source media ledger validation failed: counts={counts}, validation={validation}"
        )
    artifact = _parquet_artifact(output)
    manifest = {
        "schema_version": SOURCE_LEDGER_VERSION,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "code_commit": code_commit,
        "source_snapshot_id": source_snapshot_id,
        "identity": {
            "source_file": "multimedia.txt",
            "stable_location": "deterministic joined sort position",
            "source_row_id": "sha256(source_snapshot_id|multimedia.txt|source_sort_position)",
            "join_order_evidence": "occurrence_multimedia_join_manifest.json order_by contract",
        },
        "inputs": {
            "joined_parquet": str(joined),
            "normalized_parquet": str(normalized),
        },
        "counts": counts,
        "expected_counts": expected,
        "validation": validation,
        "artifacts": [artifact],
        "manifest_policy": {"written_last": True},
    }
    _write_json(staging / "manifest.json", manifest)
    _verify_artifact(staging, artifact)
    os.replace(staging, destination)
    return SourceLedgerResult(
        output_directory=destination,
        ledger_path=destination / output.name,
        manifest=manifest,
    )


def _ledger_query(joined: Path, normalized: Path, snapshot: str) -> str:
    year_value = "try_cast(nullif(trim(cast(year as varchar)), '') as double)"
    identified = "nullif(trim(cast(n.identifiedBy as varchar)), '') is not null"
    accepted = (
        "lower(trim(cast(n.identificationVerificationStatus as varchar))) = 'accepted'"
    )
    has_image = "nullif(trim(cast(n.media_identifier as varchar)), '') is not null"
    restricted = (
        "coalesce(lower(cast(n.media_license as varchar)) like '%all rights reserved%' "
        "or lower(trim(cast(n.media_license as varchar))) = 'copyright', false)"
    )
    return f"""
        WITH source_ordered AS (
            SELECT row_number() OVER () - 1 AS source_sort_position,
                   gbifID,
                   {year_value} AS parsed_year
            FROM read_parquet({_literal(str(joined))})
        ), source_numbered AS (
            SELECT *,
                   parsed_year IS NULL OR parsed_year >= 1960 AS year_retained,
                   sum(CASE WHEN parsed_year IS NULL OR parsed_year >= 1960 THEN 1 ELSE 0 END)
                     OVER (ORDER BY source_sort_position ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) - 1
                     AS post_year_position
            FROM source_ordered
        ), normalized_ordered AS (
            SELECT row_number() OVER () - 1 AS post_year_position,
                   identifiedBy,
                   identificationVerificationStatus,
                   media_identifier,
                   media_license
            FROM read_parquet({_literal(str(normalized))})
        ), classified AS (
            SELECT s.source_sort_position,
                   s.gbifID,
                   s.year_retained,
                   n.post_year_position IS NOT NULL AS normalized_row_resolved,
                   ({identified}) OR ({accepted}) AS cohort_retained,
                   ({has_image}) AND ({restricted}) AS rights_restricted
            FROM source_numbered s
            LEFT JOIN normalized_ordered n
              ON s.year_retained AND s.post_year_position = n.post_year_position
        )
        SELECT {_literal(SOURCE_LEDGER_VERSION)} AS ledger_version,
               {_literal(snapshot)} AS source_snapshot_id,
               'multimedia.txt' AS source_file,
               source_sort_position,
               sha256({_literal(snapshot)} || '|multimedia.txt|' || cast(source_sort_position AS varchar))
                 AS source_row_id,
               gbifID,
               CASE WHEN normalized_row_resolved OR NOT year_retained
                    THEN 'resolved_occurrence' ELSE 'unresolved_occurrence' END AS media_join_status,
               CASE
                 WHEN NOT year_retained THEN 'EXCLUDED_PRE_1960'
                 WHEN NOT cohort_retained THEN 'EXCLUDED_OUTSIDE_IDENTIFIED_OR_ACCEPTED'
                 WHEN rights_restricted THEN 'EXCLUDED_EXPLICIT_MEDIA_RIGHTS'
                 ELSE 'RETAINED_V3'
               END AS v3_funnel_status,
               CASE
                 WHEN NOT year_retained THEN 'PARSEABLE_YEAR_BEFORE_1960'
                 WHEN NOT cohort_retained THEN 'OUTSIDE_LEGACY_IDENTIFIED_OR_ACCEPTED_COHORT'
                 WHEN rights_restricted THEN 'EXPLICIT_ALL_RIGHTS_RESERVED_OR_COPYRIGHT'
                 ELSE 'NONE'
               END AS exclusion_reason,
               CASE WHEN year_retained AND cohort_retained AND NOT rights_restricted
                    THEN 'NOT_TESTED' ELSE 'NOT_APPLICABLE' END AS local_quality_status
        FROM classified
        ORDER BY source_sort_position
    """


def _ledger_counts(
    connection: duckdb.DuckDBPyConnection, output: Path
) -> dict[str, int]:
    row = connection.execute(
        f"""
        SELECT count(*)::BIGINT AS total_rows,
               count(v3_funnel_status)::BIGINT AS status_rows,
               count(*) FILTER (WHERE media_join_status = 'resolved_occurrence')::BIGINT AS resolved_occurrence,
               count(*) FILTER (WHERE v3_funnel_status = 'EXCLUDED_PRE_1960')::BIGINT AS excluded_pre_1960,
               count(*) FILTER (WHERE v3_funnel_status = 'EXCLUDED_OUTSIDE_IDENTIFIED_OR_ACCEPTED')::BIGINT AS excluded_outside_cohort,
               count(*) FILTER (WHERE v3_funnel_status = 'EXCLUDED_EXPLICIT_MEDIA_RIGHTS')::BIGINT AS excluded_explicit_rights,
               count(*) FILTER (WHERE v3_funnel_status = 'RETAINED_V3')::BIGINT AS retained_v3,
               count(DISTINCT source_row_id)::BIGINT AS distinct_source_row_ids
        FROM read_parquet({_literal(str(output))})
        """
    ).fetchone()
    assert row is not None
    names = [item[0] for item in connection.description]
    return dict(zip(names, map(int, row), strict=True))


def _parquet_artifact(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    return {
        "path": path.name,
        "physical_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "row_count": parquet.metadata.num_rows,
        "column_count": len(parquet.schema_arrow),
        "row_group_count": parquet.metadata.num_row_groups,
    }


def _verify_artifact(root: Path, artifact: dict[str, object]) -> None:
    path = root / str(artifact["path"])
    if _sha256(path) != artifact["sha256"]:
        raise ValueError("source ledger checksum verification failed")
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != artifact["row_count"]:
        raise ValueError("source ledger row-count verification failed")


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


__all__ = [
    "SOURCE_LEDGER_SCHEMA",
    "SOURCE_LEDGER_VERSION",
    "SourceLedgerResult",
    "publish_source_media_ledger",
]
