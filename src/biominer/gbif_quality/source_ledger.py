from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

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
    batch_rows: int = 100_000,
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
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
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
    output = staging / "source_media_status.parquet"
    try:
        counts = _write_ledger_streaming(
            joined=joined,
            normalized=normalized,
            output=output,
            source_snapshot_id=source_snapshot_id,
            batch_rows=batch_rows,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

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
        "runtime": {
            "method": "bounded_arrow_stream_alignment",
            "batch_rows": batch_rows,
            "memory_limit_advisory": memory_limit,
            "temp_directory_advisory": (
                str(Path(temp_directory).resolve())
                if temp_directory is not None
                else None
            ),
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


def _write_ledger_streaming(
    *,
    joined: Path,
    normalized: Path,
    output: Path,
    source_snapshot_id: str,
    batch_rows: int,
) -> dict[str, int]:
    joined_file = pq.ParquetFile(joined)
    normalized_file = pq.ParquetFile(normalized)
    normalized_cursor = _NormalizedCursor(normalized_file, batch_rows=batch_rows)
    writer = pq.ParquetWriter(
        output,
        SOURCE_LEDGER_SCHEMA,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    counts = {
        "total_rows": 0,
        "status_rows": 0,
        "resolved_occurrence": 0,
        "excluded_pre_1960": 0,
        "excluded_outside_cohort": 0,
        "excluded_explicit_rights": 0,
        "retained_v3": 0,
        "distinct_source_row_ids": 0,
    }
    identity_prefix = f"{source_snapshot_id}|multimedia.txt|".encode()
    try:
        for batch in joined_file.iter_batches(
            batch_size=batch_rows, columns=["gbifID", "year"], use_threads=True
        ):
            gbif_ids = batch.column(0).to_pylist()
            years = batch.column(1).to_pylist()
            retained = [_year_retained(value) for value in years]
            normalized_values = normalized_cursor.take(sum(retained))
            normalized_index = 0
            output_rows = {field.name: [] for field in SOURCE_LEDGER_SCHEMA}
            for offset, (gbif_id, year_is_retained) in enumerate(
                zip(gbif_ids, retained, strict=True)
            ):
                position = counts["total_rows"] + offset
                if year_is_retained:
                    identified_by = normalized_values["identifiedBy"][normalized_index]
                    verification = normalized_values[
                        "identificationVerificationStatus"
                    ][normalized_index]
                    identifier = normalized_values["media_identifier"][normalized_index]
                    media_license = normalized_values["media_license"][normalized_index]
                    normalized_index += 1
                    cohort_retained = _nonblank(identified_by) or (
                        str(verification or "").strip().casefold() == "accepted"
                    )
                    rights_restricted = _nonblank(identifier) and _restricted(
                        media_license
                    )
                    if not cohort_retained:
                        funnel_status = "EXCLUDED_OUTSIDE_IDENTIFIED_OR_ACCEPTED"
                        exclusion = "OUTSIDE_LEGACY_IDENTIFIED_OR_ACCEPTED_COHORT"
                        count_key = "excluded_outside_cohort"
                    elif rights_restricted:
                        funnel_status = "EXCLUDED_EXPLICIT_MEDIA_RIGHTS"
                        exclusion = "EXPLICIT_ALL_RIGHTS_RESERVED_OR_COPYRIGHT"
                        count_key = "excluded_explicit_rights"
                    else:
                        funnel_status = "RETAINED_V3"
                        exclusion = "NONE"
                        count_key = "retained_v3"
                else:
                    funnel_status = "EXCLUDED_PRE_1960"
                    exclusion = "PARSEABLE_YEAR_BEFORE_1960"
                    count_key = "excluded_pre_1960"
                output_rows["ledger_version"].append(SOURCE_LEDGER_VERSION)
                output_rows["source_snapshot_id"].append(source_snapshot_id)
                output_rows["source_file"].append("multimedia.txt")
                output_rows["source_sort_position"].append(position)
                output_rows["source_row_id"].append(
                    hashlib.sha256(identity_prefix + str(position).encode()).hexdigest()
                )
                output_rows["gbifID"].append(gbif_id)
                output_rows["media_join_status"].append("resolved_occurrence")
                output_rows["v3_funnel_status"].append(funnel_status)
                output_rows["exclusion_reason"].append(exclusion)
                output_rows["local_quality_status"].append(
                    "NOT_TESTED" if funnel_status == "RETAINED_V3" else "NOT_APPLICABLE"
                )
                counts[count_key] += 1
            if normalized_index != sum(retained):
                raise RuntimeError("normalized cursor did not align with retained rows")
            writer.write_table(
                pa.Table.from_pydict(output_rows, schema=SOURCE_LEDGER_SCHEMA),
                row_group_size=batch_rows,
            )
            counts["total_rows"] += batch.num_rows
            counts["status_rows"] += batch.num_rows
            counts["resolved_occurrence"] += batch.num_rows
            counts["distinct_source_row_ids"] += batch.num_rows
        normalized_cursor.assert_exhausted()
    finally:
        writer.close()
    return counts


class _NormalizedCursor:
    _COLUMNS = (
        "identifiedBy",
        "identificationVerificationStatus",
        "media_identifier",
        "media_license",
    )

    def __init__(self, parquet: pq.ParquetFile, *, batch_rows: int) -> None:
        self._batches = iter(
            parquet.iter_batches(
                batch_size=batch_rows, columns=list(self._COLUMNS), use_threads=True
            )
        )
        self._batch: pa.RecordBatch | None = None
        self._offset = 0

    def take(self, count: int) -> dict[str, list[object]]:
        values: dict[str, list[object]] = {name: [] for name in self._COLUMNS}
        remaining = count
        while remaining:
            self._ensure_batch()
            if self._batch is None:
                raise ValueError("normalized Parquet ended before joined retained rows")
            available = self._batch.num_rows - self._offset
            size = min(available, remaining)
            piece = self._batch.slice(self._offset, size)
            for index, name in enumerate(self._COLUMNS):
                values[name].extend(piece.column(index).to_pylist())
            self._offset += size
            remaining -= size
        return values

    def assert_exhausted(self) -> None:
        self._ensure_batch()
        if self._batch is not None:
            raise ValueError("normalized Parquet has rows beyond joined retained rows")

    def _ensure_batch(self) -> None:
        if self._batch is not None and self._offset < self._batch.num_rows:
            return
        self._batch = next(self._batches, None)
        self._offset = 0


def _year_retained(value: object | None) -> bool:
    if value is None or not str(value).strip():
        return True
    try:
        return float(value) >= 1960
    except (TypeError, ValueError):
        return True


def _nonblank(value: object | None) -> bool:
    return value is not None and bool(str(value).strip())


def _restricted(value: object | None) -> bool:
    normalized = str(value or "").strip().casefold()
    return "all rights reserved" in normalized or normalized == "copyright"


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


__all__ = [
    "SOURCE_LEDGER_SCHEMA",
    "SOURCE_LEDGER_VERSION",
    "SourceLedgerResult",
    "publish_source_media_ledger",
]
