from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from urllib.parse import urlparse
from uuid import uuid4

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from biominer.gbif_quality.registry import CHECK_REGISTRY_VERSION
from biominer.references.licensing import canonicalise_creative_commons_licence


MEDIA_QUALITY_VERSION = "biominer-gbif-media-assertion-quality/v1"
MEDIA_QUALITY_SCHEMA = pa.schema(
    [
        ("quality_version", pa.string()),
        ("check_registry_version", pa.string()),
        ("source_snapshot_id", pa.string()),
        ("source_row_id", pa.string()),
        ("source_sort_position", pa.int64()),
        ("media_assertion_id", pa.string()),
        ("gbifID", pa.string()),
        ("direct_media_url_status", pa.string()),
        ("media_reference_url_status", pa.string()),
        ("media_type_format_status", pa.string()),
        ("media_rights_status", pa.string()),
        ("legacy_transformation_provenance_status", pa.string()),
        ("overall_media_quality_status", pa.string()),
    ]
)
_V3_COLUMNS = (
    "gbifID",
    "media_identifier",
    "media_references",
    "media_type",
    "media_format",
    "media_license",
)


@dataclass(frozen=True, slots=True)
class MediaQualityResult:
    output_directory: Path
    quality_path: Path
    manifest: dict[str, object]


def publish_media_assertion_quality(
    *,
    v3_parquet: str | Path,
    source_ledger_parquet: str | Path,
    output_directory: str | Path,
    source_snapshot_id: str,
    expected_rows: int,
    code_commit: str,
    batch_rows: int = 100_000,
) -> MediaQualityResult:
    """Run request-free media assertion checks with exact source identities."""

    v3 = Path(v3_parquet).resolve()
    ledger = Path(source_ledger_parquet).resolve()
    destination = Path(output_directory).resolve()
    for path in (v3, ledger):
        if not path.is_file():
            raise FileNotFoundError(path)
    if destination.exists():
        raise FileExistsError(destination)
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    if pq.ParquetFile(v3).metadata.num_rows != expected_rows:
        raise ValueError("v3 row count differs from expected_rows")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    output = staging / "media_assertion_quality.parquet"
    try:
        counts = _write_quality(
            v3=v3,
            ledger=ledger,
            output=output,
            source_snapshot_id=source_snapshot_id,
            batch_rows=batch_rows,
        )
        parquet = pq.ParquetFile(output)
        validation = {
            "row_count_matches_v3": counts["rows"]
            == parquet.metadata.num_rows
            == expected_rows,
            "one_source_identity_per_row": counts["source_identity_rows"]
            == expected_rows,
            "one_status_per_local_check": counts["complete_status_rows"]
            == expected_rows,
            "schema_matches": parquet.schema_arrow.equals(MEDIA_QUALITY_SCHEMA),
            "source_ledger_exhausted_at_retained_scope": counts[
                "ledger_retained_rows"
            ]
            == expected_rows,
        }
        if not all(validation.values()):
            raise ValueError(
                f"media assertion quality validation failed: {validation}"
            )
        artifact = _artifact(output)
        manifest = {
            "schema_version": MEDIA_QUALITY_VERSION,
            "check_registry_version": CHECK_REGISTRY_VERSION,
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "code_commit": code_commit,
            "source_snapshot_id": source_snapshot_id,
            "inputs": {
                "v3_parquet": str(v3),
                "source_ledger_parquet": str(ledger),
            },
            "counts": counts,
            "validation": validation,
            "artifacts": [artifact],
            "network_requests": 0,
            "manifest_policy": {"written_last": True},
        }
        _write_json(staging / "manifest.json", manifest)
        _verify(staging, artifact)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return MediaQualityResult(
        output_directory=destination,
        quality_path=destination / output.name,
        manifest=manifest,
    )


def _write_quality(
    *,
    v3: Path,
    ledger: Path,
    output: Path,
    source_snapshot_id: str,
    batch_rows: int,
) -> dict[str, object]:
    v3_file = pq.ParquetFile(v3)
    ledger_cursor = _RetainedLedgerCursor(
        pq.ParquetFile(ledger), batch_rows=batch_rows
    )
    writer = pq.ParquetWriter(
        output,
        MEDIA_QUALITY_SCHEMA,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    status_counts: dict[str, Counter[str]] = {
        name: Counter()
        for name in (
            "direct_media_url_status",
            "media_reference_url_status",
            "media_type_format_status",
            "media_rights_status",
            "legacy_transformation_provenance_status",
            "overall_media_quality_status",
        )
    }
    rows = 0
    try:
        for batch in v3_file.iter_batches(
            batch_size=batch_rows, columns=list(_V3_COLUMNS), use_threads=True
        ):
            source = ledger_cursor.take(batch.num_rows)
            values = {
                name: batch.column(index).to_pylist()
                for index, name in enumerate(_V3_COLUMNS)
            }
            output_rows = {field.name: [] for field in MEDIA_QUALITY_SCHEMA}
            for index in range(batch.num_rows):
                direct = _url_status(values["media_identifier"][index], missing="UNKNOWN")
                reference = _url_status(
                    values["media_references"][index], missing="NOT_APPLICABLE"
                )
                media_type = _type_format_status(
                    identifier=values["media_identifier"][index],
                    media_type=values["media_type"][index],
                    media_format=values["media_format"][index],
                )
                rights = _rights_status(values["media_license"][index])
                provenance = "PASS"
                overall = _overall((direct, reference, media_type, rights, provenance))
                source_row_id = source["source_row_id"][index]
                output_rows["quality_version"].append(MEDIA_QUALITY_VERSION)
                output_rows["check_registry_version"].append(CHECK_REGISTRY_VERSION)
                output_rows["source_snapshot_id"].append(source_snapshot_id)
                output_rows["source_row_id"].append(source_row_id)
                output_rows["source_sort_position"].append(
                    source["source_sort_position"][index]
                )
                output_rows["media_assertion_id"].append(
                    "sha256:"
                    + hashlib.sha256(
                        f"{source_snapshot_id}|media_assertion|{source_row_id}".encode()
                    ).hexdigest()
                )
                output_rows["gbifID"].append(values["gbifID"][index])
                for name, status in (
                    ("direct_media_url_status", direct),
                    ("media_reference_url_status", reference),
                    ("media_type_format_status", media_type),
                    ("media_rights_status", rights),
                    ("legacy_transformation_provenance_status", provenance),
                    ("overall_media_quality_status", overall),
                ):
                    output_rows[name].append(status)
                    status_counts[name][status] += 1
            writer.write_table(
                pa.Table.from_pydict(output_rows, schema=MEDIA_QUALITY_SCHEMA),
                row_group_size=batch_rows,
            )
            rows += batch.num_rows
        ledger_cursor.assert_exhausted()
    finally:
        writer.close()
    return {
        "rows": rows,
        "source_identity_rows": rows,
        "complete_status_rows": rows,
        "ledger_retained_rows": ledger_cursor.retained_rows,
        "status_counts": {
            name: dict(sorted(counter.items()))
            for name, counter in status_counts.items()
        },
    }


class _RetainedLedgerCursor:
    def __init__(self, parquet: pq.ParquetFile, *, batch_rows: int) -> None:
        self._batches = iter(
            parquet.iter_batches(
                batch_size=batch_rows,
                columns=["source_row_id", "source_sort_position", "v3_funnel_status"],
                use_threads=True,
            )
        )
        self._batch: pa.RecordBatch | None = None
        self._offset = 0
        self.retained_rows = 0

    def take(self, count: int) -> dict[str, list[object]]:
        output = {"source_row_id": [], "source_sort_position": []}
        while len(output["source_row_id"]) < count:
            self._ensure_batch()
            if self._batch is None:
                raise ValueError("source ledger ended before v3 rows")
            remaining = count - len(output["source_row_id"])
            available = self._batch.num_rows - self._offset
            size = min(remaining, available)
            piece = self._batch.slice(self._offset, size)
            output["source_row_id"].extend(piece.column(0).to_pylist())
            output["source_sort_position"].extend(piece.column(1).to_pylist())
            self._offset += size
        self.retained_rows += count
        return output

    def assert_exhausted(self) -> None:
        self._ensure_batch()
        if self._batch is not None:
            raise ValueError("source ledger contains retained rows beyond v3")

    def _ensure_batch(self) -> None:
        if self._batch is not None and self._offset < self._batch.num_rows:
            return
        for batch in self._batches:
            retained = pc.equal(batch.column(2), "RETAINED_V3")
            filtered = batch.filter(retained).select([0, 1])
            if filtered.num_rows:
                self._batch = filtered
                self._offset = 0
                return
        self._batch = None
        self._offset = 0


def _url_status(value: object | None, *, missing: str) -> str:
    text = str(value or "").strip()
    if not text:
        return missing
    try:
        parsed = urlparse(text)
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            return "FAIL"
        _ = parsed.port
    except ValueError:
        return "FAIL"
    return "PASS"


def _type_format_status(
    *, identifier: object | None, media_type: object | None, media_format: object | None
) -> str:
    if not str(identifier or "").strip():
        return "NOT_APPLICABLE"
    kind = str(media_type or "").strip().casefold()
    media_format_value = str(media_format or "").strip().casefold()
    if not kind and not media_format_value:
        return "UNKNOWN"
    if not media_format_value:
        return "UNKNOWN"
    compatible = (
        (kind in {"", "stillimage"} and media_format_value.startswith("image/"))
        or (kind == "sound" and media_format_value.startswith("audio/"))
        or (kind == "movingimage" and media_format_value.startswith("video/"))
    )
    return "PASS" if compatible else "CONFLICT"


def _rights_status(value: object | None) -> str:
    text = str(value or "").strip()
    if not text:
        return "UNKNOWN"
    lowered = text.casefold()
    if "all rights reserved" in lowered or lowered == "copyright":
        return "FAIL"
    if canonicalise_creative_commons_licence(text) is not None:
        return "PASS"
    if "public domain" in lowered or lowered in {"pdm", "cc0"}:
        return "PASS"
    return "UNKNOWN"


def _overall(statuses: tuple[str, ...]) -> str:
    if "CONFLICT" in statuses:
        return "CONFLICT"
    if "FAIL" in statuses:
        return "FAIL"
    if "UNKNOWN" in statuses:
        return "UNKNOWN"
    return "PASS"


def _artifact(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    return {
        "path": path.name,
        "physical_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "row_count": parquet.metadata.num_rows,
        "column_count": len(parquet.schema_arrow),
        "row_group_count": parquet.metadata.num_row_groups,
    }


def _verify(root: Path, artifact: dict[str, object]) -> None:
    path = root / str(artifact["path"])
    if _sha256(path) != artifact["sha256"]:
        raise ValueError("media quality artifact checksum mismatch")
    if pq.ParquetFile(path).metadata.num_rows != artifact["row_count"]:
        raise ValueError("media quality artifact row count mismatch")


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
    "MEDIA_QUALITY_SCHEMA",
    "MEDIA_QUALITY_VERSION",
    "MediaQualityResult",
    "publish_media_assertion_quality",
]
