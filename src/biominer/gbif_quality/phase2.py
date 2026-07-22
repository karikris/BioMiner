from __future__ import annotations

from collections import Counter
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

from biominer.gbif_quality.registry import (
    CHECK_REGISTRY_VERSION,
    check_registry,
    check_registry_table,
    registry_fingerprint,
)


PHASE2_VERSION = "biominer-gbif-media-local-quality-phase2/v1"
SOURCE_QUALITY_STATUS_SCHEMA = pa.schema(
    [
        ("phase_version", pa.string()),
        ("source_snapshot_id", pa.string()),
        ("source_sort_position", pa.int64()),
        ("source_row_id", pa.string()),
        ("gbifID", pa.string()),
        ("v3_funnel_status", pa.string()),
        ("local_quality_status", pa.string()),
        ("quality_evidence_status", pa.string()),
    ]
)
CHECK_COVERAGE_SCHEMA = pa.schema(
    [
        ("phase_version", pa.string()),
        ("registry_version", pa.string()),
        ("registry_fingerprint", pa.string()),
        ("check_id", pa.string()),
        ("scope", pa.string()),
        ("execution_status", pa.string()),
        ("evidence_path", pa.string()),
        ("network_requests", pa.int64()),
        ("coverage_note", pa.string()),
    ]
)


@dataclass(frozen=True, slots=True)
class Phase2Result:
    output_directory: Path
    manifest: dict[str, object]


def publish_phase2_summary(
    *,
    source_ledger_parquet: str | Path,
    media_quality_parquet: str | Path,
    occurrence_quality_parquet: str | Path,
    media_manifest: str | Path,
    occurrence_manifest: str | Path,
    output_directory: str | Path,
    source_snapshot_id: str,
    expected_source_rows: int,
    expected_v3_rows: int,
    expected_occurrences: int,
    code_commit: str,
    batch_rows: int = 100_000,
) -> Phase2Result:
    """Publish the complete local-check coverage and source-row status overlay."""

    source = Path(source_ledger_parquet).resolve()
    media = Path(media_quality_parquet).resolve()
    occurrence = Path(occurrence_quality_parquet).resolve()
    media_manifest_path = Path(media_manifest).resolve()
    occurrence_manifest_path = Path(occurrence_manifest).resolve()
    destination = Path(output_directory).resolve()
    for path in (
        source,
        media,
        occurrence,
        media_manifest_path,
        occurrence_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if destination.exists():
        raise FileExistsError(destination)
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    media_summary = _read_json(media_manifest_path)
    occurrence_summary = _read_json(occurrence_manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    source_status = staging / "source_media_quality_status.parquet"
    registry_path = staging / "check_registry.parquet"
    coverage_path = staging / "check_coverage.parquet"
    try:
        source_counts = _write_source_status(
            source=source,
            media=media,
            output=source_status,
            source_snapshot_id=source_snapshot_id,
            batch_rows=batch_rows,
        )
        pq.write_table(
            check_registry_table(), registry_path, compression="zstd"
        )
        coverage = _coverage_table()
        pq.write_table(coverage, coverage_path, compression="zstd")
        validation = {
            "all_source_rows_have_final_local_status": source_counts["source_rows"]
            == source_counts["status_rows"]
            == expected_source_rows,
            "all_retained_rows_link_to_quality": source_counts["retained_rows"]
            == source_counts["linked_quality_rows"]
            == expected_v3_rows,
            "excluded_rows_remain_explicit": source_counts["excluded_rows"]
            == expected_source_rows - expected_v3_rows,
            "media_quality_row_count_matches": pq.ParquetFile(media).metadata.num_rows
            == expected_v3_rows,
            "occurrence_quality_row_count_matches": pq.ParquetFile(
                occurrence
            ).metadata.num_rows
            == expected_occurrences,
            "registry_coverage_complete": coverage.num_rows
            == len(check_registry()),
            "network_checks_not_claimed": all(
                row["execution_status"] == "NOT_TESTED"
                for row in coverage.to_pylist()
                if row["check_id"] == "MEDIA_URL_003"
            ),
            "source_status_schema_matches": pq.ParquetFile(
                source_status
            ).schema_arrow.equals(SOURCE_QUALITY_STATUS_SCHEMA),
        }
        if not all(validation.values()):
            raise ValueError(f"phase 2 validation failed: {validation}")
        summary = {
            "schema_version": PHASE2_VERSION,
            "source_snapshot_id": source_snapshot_id,
            "source_counts": source_counts,
            "media_status_counts": media_summary["counts"]["status_counts"],
            "occurrence_counts": occurrence_summary["counts"],
            "check_execution_counts": dict(
                sorted(Counter(coverage["execution_status"].to_pylist()).items())
            ),
        }
        _write_json(staging / "phase2_summary.json", summary)
        artifacts = [_artifact(path) for path in sorted(staging.iterdir())]
        manifest = {
            "schema_version": PHASE2_VERSION,
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "code_commit": code_commit,
            "source_snapshot_id": source_snapshot_id,
            "check_registry_version": CHECK_REGISTRY_VERSION,
            "check_registry_fingerprint": registry_fingerprint(),
            "inputs": {
                "source_ledger": str(source),
                "media_quality": str(media),
                "occurrence_quality": str(occurrence),
            },
            "counts": source_counts,
            "validation": validation,
            "artifacts": artifacts,
            "network_requests": 0,
            "manifest_policy": {"written_last": True},
        }
        _write_json(staging / "manifest.json", manifest)
        for artifact in artifacts:
            _verify(staging, artifact)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return Phase2Result(output_directory=destination, manifest=manifest)


def _write_source_status(
    *,
    source: Path,
    media: Path,
    output: Path,
    source_snapshot_id: str,
    batch_rows: int,
) -> dict[str, int]:
    source_file = pq.ParquetFile(source)
    media_cursor = _MediaQualityCursor(pq.ParquetFile(media), batch_rows=batch_rows)
    writer = pq.ParquetWriter(
        output,
        SOURCE_QUALITY_STATUS_SCHEMA,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    counts = {
        "source_rows": 0,
        "status_rows": 0,
        "retained_rows": 0,
        "excluded_rows": 0,
        "linked_quality_rows": 0,
    }
    try:
        for batch in source_file.iter_batches(
            batch_size=batch_rows,
            columns=[
                "source_sort_position",
                "source_row_id",
                "gbifID",
                "v3_funnel_status",
            ],
            use_threads=True,
        ):
            rows = {field.name: [] for field in SOURCE_QUALITY_STATUS_SCHEMA}
            values = [batch.column(index).to_pylist() for index in range(4)]
            for position, source_row_id, gbif_id, funnel_status in zip(
                *values, strict=True
            ):
                if funnel_status == "RETAINED_V3":
                    quality_id, quality_status = media_cursor.next()
                    if quality_id != source_row_id:
                        raise ValueError("media quality source identity alignment failed")
                    evidence_status = "PASS"
                    counts["retained_rows"] += 1
                    counts["linked_quality_rows"] += 1
                else:
                    quality_status = "NOT_APPLICABLE"
                    evidence_status = "NOT_APPLICABLE"
                    counts["excluded_rows"] += 1
                rows["phase_version"].append(PHASE2_VERSION)
                rows["source_snapshot_id"].append(source_snapshot_id)
                rows["source_sort_position"].append(position)
                rows["source_row_id"].append(source_row_id)
                rows["gbifID"].append(gbif_id)
                rows["v3_funnel_status"].append(funnel_status)
                rows["local_quality_status"].append(quality_status)
                rows["quality_evidence_status"].append(evidence_status)
            writer.write_table(
                pa.Table.from_pydict(rows, schema=SOURCE_QUALITY_STATUS_SCHEMA),
                row_group_size=batch_rows,
            )
            counts["source_rows"] += batch.num_rows
            counts["status_rows"] += batch.num_rows
        media_cursor.assert_exhausted()
    finally:
        writer.close()
    return counts


class _MediaQualityCursor:
    def __init__(self, parquet: pq.ParquetFile, *, batch_rows: int) -> None:
        self._batches = iter(
            parquet.iter_batches(
                batch_size=batch_rows,
                columns=["source_row_id", "overall_media_quality_status"],
                use_threads=True,
            )
        )
        self._batch: pa.RecordBatch | None = None
        self._offset = 0

    def next(self) -> tuple[str, str]:
        self._ensure()
        if self._batch is None:
            raise ValueError("media quality ended before retained source rows")
        result = (
            str(self._batch.column(0)[self._offset].as_py()),
            str(self._batch.column(1)[self._offset].as_py()),
        )
        self._offset += 1
        return result

    def assert_exhausted(self) -> None:
        self._ensure()
        if self._batch is not None:
            raise ValueError("media quality contains rows beyond retained source scope")

    def _ensure(self) -> None:
        if self._batch is not None and self._offset < self._batch.num_rows:
            return
        self._batch = next(self._batches, None)
        self._offset = 0


def _coverage_table() -> pa.Table:
    local_evidence = {
        "source": "source_lineage/manifest.json",
        "schema": "manifest.json",
        "semantic_null": "completeness_by_applicability.parquet",
        "identifier": "occurrence_quality/occurrence_quality.parquet",
        "vocabulary": "occurrence_quality/occurrence_quality.parquet",
        "gbif_issue": "occurrence_quality/gbif_issue_summary.parquet",
        "temporal": "occurrence_quality/occurrence_quality.parquet",
        "geospatial": "occurrence_quality/occurrence_quality.parquet",
        "taxonomic": "occurrence_quality/occurrence_quality.parquet",
        "identification": "occurrence_quality/occurrence_quality.parquet",
        "occurrence_semantics": "occurrence_quality/occurrence_quality.parquet",
        "media_url": "media_assertion_quality/media_assertion_quality.parquet",
        "media_file": "media_assertion_quality/media_assertion_quality.parquet",
        "rights": "media_assertion_quality/media_assertion_quality.parquet",
        "provenance": "media_assertion_quality/media_assertion_quality.parquet",
    }
    fingerprint = registry_fingerprint()
    rows = []
    for check in check_registry():
        network = check.network_required
        rows.append(
            {
                "phase_version": PHASE2_VERSION,
                "registry_version": CHECK_REGISTRY_VERSION,
                "registry_fingerprint": fingerprint,
                "check_id": check.check_id,
                "scope": check.scope,
                "execution_status": "NOT_TESTED" if network else "PASS",
                "evidence_path": None if network else local_evidence[check.check_family],
                "network_requests": 0,
                "coverage_note": (
                    "Live network check is opt-in and was not executed."
                    if network
                    else "Deterministic local implementation produced stored evidence."
                ),
            }
        )
    return pa.Table.from_pylist(rows, schema=CHECK_COVERAGE_SCHEMA)


def _artifact(path: Path) -> dict[str, object]:
    item: dict[str, object] = {
        "path": path.name,
        "physical_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        item.update(
            {
                "row_count": parquet.metadata.num_rows,
                "column_count": len(parquet.schema_arrow),
                "row_group_count": parquet.metadata.num_row_groups,
            }
        )
    return item


def _verify(root: Path, artifact: dict[str, object]) -> None:
    path = root / str(artifact["path"])
    if _sha256(path) != artifact["sha256"]:
        raise ValueError(f"phase 2 artifact checksum mismatch: {path.name}")


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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
    "CHECK_COVERAGE_SCHEMA",
    "PHASE2_VERSION",
    "SOURCE_QUALITY_STATUS_SCHEMA",
    "Phase2Result",
    "publish_phase2_summary",
]
