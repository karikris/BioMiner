from __future__ import annotations

from contextlib import ExitStack
import csv
from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path
import shutil
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4
import xml.etree.ElementTree as ET
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.gbif_media_resolution.models import (
    ATTEMPT_SCHEMA,
    JOB_NAME,
    RESULT_SCHEMA,
    STAGE,
    ResolutionInput,
    ResolutionResult,
    ResolutionStatus,
)
from biominer.workstore.base import WorkStore


ARCHIVE_CIRCUIT_VERSION = "gbif-media-provider-archive-circuit/v2"
TERMINAL_REASON = (
    "provider_archive_associated_media_reference_only_no_multimedia_extension"
)
_MULTIMEDIA_ROW_TYPE = "http://rs.gbif.org/terms/1.0/multimedia"
_OCCURRENCE_ROW_TYPE = "http://rs.tdwg.org/dwc/terms/occurrence"
_ASSOCIATED_MEDIA_TERM = "http://rs.tdwg.org/dwc/terms/associatedmedia"
_OCCURRENCE_ID_TERM = "http://rs.tdwg.org/dwc/terms/occurrenceid"
_DIRECT_IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
ARCHIVE_BINDING_SCHEMA = pa.schema(
    [
        ("source_row_id", pa.string()),
        ("gbif_id", pa.string()),
        ("provider", pa.string()),
        ("media_references", pa.string()),
        ("reference_host", pa.string()),
        ("archive_occurrence_id", pa.string()),
        ("archive_reference_occurrences", pa.int64()),
        ("archive_sha256", pa.string()),
        ("archive_manifest_sha256", pa.string()),
        ("binding_status", pa.string()),
        ("terminal_status", pa.string()),
        ("terminal_reason", pa.string()),
        ("network_requests", pa.int64()),
    ]
)


def complete_archive_reference_only_rows(
    *,
    workstore: WorkStore,
    run_id: str,
    output_root: str | Path,
    archive_manifest: str | Path,
    provider: str,
    dataset_key: str,
    expected_pending_rows: int | None = None,
) -> dict[str, Any]:
    """Complete a provider cohort from checksum-bound, item-scoped DwCA evidence."""

    provider = str(provider).strip()
    dataset_key = str(dataset_key).strip()
    if not provider or not dataset_key:
        raise ValueError("provider and dataset_key must be nonblank")
    run = workstore.get_run(run_id=run_id)
    if run is None or run["job_name"] != JOB_NAME or run["stage"] != STAGE:
        raise ValueError(f"unknown GBIF URL resolution run: {run_id}")
    root = Path(output_root).resolve()
    if root != Path(str(run["config"].get("output_root", ""))).resolve():
        raise ValueError("archive circuit output root does not match prepared run")

    manifest_path = Path(archive_manifest).resolve()
    manifest_sha256 = _sha256_uri(manifest_path)
    archive_entry, archive_path = _validated_archive_entry(
        manifest_path=manifest_path,
        provider=provider,
        dataset_key=dataset_key,
    )
    archive_sha256 = f"sha256:{archive_entry['sha256']}"
    archive = _read_reference_only_archive(archive_path)

    registry_version = str(run["registry_version"])
    lock_key = (
        f"{JOB_NAME}:archive-circuit:{run_id}:{provider}:{dataset_key}:"
        f"{archive_sha256}"
    )
    with workstore.publication_lock(lock_key):
        work_items = workstore.list_work_items(
            job_name=JOB_NAME,
            stage=STAGE,
            registry_version=registry_version,
        )
        claimed = [item for item in work_items if item["status"] == "claimed"]
        if claimed:
            raise RuntimeError(
                "archive circuit requires zero active resolver claims; "
                f"found {len(claimed)}"
            )
        pending: list[tuple[dict[str, Any], ResolutionInput]] = []
        for work_item in work_items:
            if work_item["status"] != "pending":
                continue
            item = ResolutionInput.from_payload(dict(work_item["payload"]))
            if item.provider == provider:
                pending.append((work_item, item))
        if expected_pending_rows is not None and len(pending) != expected_pending_rows:
            raise ValueError(
                "archive circuit pending row count mismatch: "
                f"expected {expected_pending_rows}, found {len(pending)}"
            )
        if not pending:
            raise ValueError("archive circuit selected no pending rows")

        unmatched = [
            item.source_row_id
            for _, item in pending
            if item.media_references not in archive["references"]
        ]
        direct_images = [
            item.source_row_id
            for _, item in pending
            if _looks_like_direct_image(item.media_references)
        ]
        if unmatched:
            raise ValueError(
                "pending provider rows are not all item-bound to archive "
                f"associatedMedia values: {len(unmatched)} unmatched"
            )
        if direct_images:
            raise ValueError(
                "archive circuit refuses direct-image associatedMedia values: "
                f"{len(direct_images)} rows"
            )

        generated_at = _timestamp()
        result_rows: list[dict[str, Any]] = []
        binding_rows: list[dict[str, Any]] = []
        for _, item in pending:
            archive_match = archive["references"][item.media_references]
            provenance = canonical_semantic_fingerprint(
                {
                    "contract": ARCHIVE_CIRCUIT_VERSION,
                    "source_row_id": item.source_row_id,
                    "source_artifact_sha256": item.source_artifact_sha256,
                    "provider": provider,
                    "dataset_key": dataset_key,
                    "media_references": item.media_references,
                    "archive_sha256": archive_sha256,
                    "archive_manifest_sha256": manifest_sha256,
                    "terminal_status": (
                        ResolutionStatus.UNRESOLVED_ARCHIVE_REFERENCE_ONLY.value
                    ),
                    "terminal_reason": TERMINAL_REASON,
                }
            )
            result_rows.append(
                ResolutionResult(
                    source_row_id=item.source_row_id,
                    source_artifact_sha256=item.source_artifact_sha256,
                    gbif_id=item.gbif_id,
                    media_references=item.media_references,
                    reference_host=item.host,
                    media_type=item.media_type,
                    media_format=item.media_format,
                    media_license=item.media_license,
                    occurrence_license=item.occurrence_license,
                    license_basis=item.license_basis,
                    status=ResolutionStatus.UNRESOLVED_ARCHIVE_REFERENCE_ONLY,
                    method="provider_darwin_core_archive",
                    stable_candidate_url=None,
                    validated_final_url=None,
                    redirect_count=0,
                    declared_content_type=None,
                    detected_content_type=None,
                    bytes_sampled=0,
                    probe_prefix_sha256=None,
                    content_sha256=None,
                    content_hash_status="not_retrieved",
                    adapter_version=ARCHIVE_CIRCUIT_VERSION,
                    attempt_count=0,
                    terminal_reason=TERMINAL_REASON,
                    resolved_at=generated_at,
                    provenance_fingerprint=provenance,
                ).to_row()
            )
            binding_rows.append(
                {
                    "source_row_id": item.source_row_id,
                    "gbif_id": item.gbif_id,
                    "provider": provider,
                    "media_references": item.media_references,
                    "reference_host": item.host,
                    "archive_occurrence_id": archive_match[
                        "first_occurrence_id"
                    ],
                    "archive_reference_occurrences": archive_match["count"],
                    "archive_sha256": archive_sha256,
                    "archive_manifest_sha256": manifest_sha256,
                    "binding_status": "exact_associated_media_match",
                    "terminal_status": (
                        ResolutionStatus.UNRESOLVED_ARCHIVE_REFERENCE_ONLY.value
                    ),
                    "terminal_reason": TERMINAL_REASON,
                    "network_requests": 0,
                }
            )

        token = canonical_semantic_fingerprint(
            {
                "contract": ARCHIVE_CIRCUIT_VERSION,
                "run_id": run_id,
                "provider": provider,
                "dataset_key": dataset_key,
                "archive_sha256": archive_sha256,
                "source_row_ids": sorted(row["source_row_id"] for row in result_rows),
            }
        ).split(":", 1)[1]
        result_path = root / "shards" / "results" / f"archive-{token}.parquet"
        attempt_path = root / "shards" / "attempts" / f"archive-{token}.parquet"
        publication = root / "archive_circuits" / token
        if publication.exists() or result_path.exists() or attempt_path.exists():
            raise FileExistsError(
                "archive circuit content-addressed publication already exists"
            )
        staging = publication.parent / f".{token}.{uuid4().hex}.staging"
        staging.mkdir(parents=True)
        try:
            binding_path = staging / "archive_reference_bindings.parquet"
            _write_parquet(
                result_path,
                pa.Table.from_pylist(result_rows, schema=RESULT_SCHEMA),
            )
            _write_parquet(
                attempt_path,
                pa.Table.from_pylist([], schema=ATTEMPT_SCHEMA),
            )
            _write_parquet(
                binding_path,
                pa.Table.from_pylist(binding_rows, schema=ARCHIVE_BINDING_SCHEMA),
            )
            result_inventory = _parquet_inventory(result_path)
            attempt_inventory = _parquet_inventory(attempt_path)
            binding_inventory = _parquet_inventory(binding_path)
            workstore.register_shard(
                shard_id=f"archive-{token}",
                job_name=JOB_NAME,
                registry_version=registry_version,
                stage=STAGE,
                run_id=run_id,
                worker_id="provider-archive-circuit",
                uri=str(result_path),
                checksum=result_inventory["physical_sha256"],
                row_count=len(result_rows),
                byte_count=result_path.stat().st_size,
                metadata={
                    "attempt_uri": str(attempt_path),
                    "attempt_sha256": attempt_inventory["physical_sha256"],
                    "attempt_rows": 0,
                    "archive_manifest": str(manifest_path),
                    "archive_manifest_sha256": manifest_sha256,
                    "archive_path": str(archive_path),
                    "archive_sha256": archive_sha256,
                    "archive_circuit_version": ARCHIVE_CIRCUIT_VERSION,
                    "network_requests": 0,
                },
            )
            completed = workstore.complete_pending_batch(
                [str(work_item["work_key"]) for work_item, _ in pending],
                output_uri=str(result_path),
                checksum=result_inventory["physical_sha256"],
                row_count=1,
            )
            if completed != {
                str(work_item["work_key"]) for work_item, _ in pending
            }:
                raise RuntimeError(
                    "pending archive cohort changed during atomic completion"
                )
            manifest = {
                "schema_version": ARCHIVE_CIRCUIT_VERSION,
                "generated_at": generated_at,
                "git_commit": _git_revision(),
                "run_id": run_id,
                "source_snapshot_id": registry_version,
                "provider": provider,
                "dataset_key": dataset_key,
                "terminal_status": (
                    ResolutionStatus.UNRESOLVED_ARCHIVE_REFERENCE_ONLY.value
                ),
                "terminal_reason": TERMINAL_REASON,
                "network_requests": 0,
                "input": {
                    "archive_manifest": str(manifest_path),
                    "archive_manifest_sha256": manifest_sha256,
                    "archive_path": str(archive_path),
                    "archive_sha256": archive_sha256,
                    "archive_source_url": archive_entry.get("source_url"),
                    "archive_occurrence_location": archive["occurrence_location"],
                    "archive_occurrence_table_kind": archive[
                        "occurrence_table_kind"
                    ],
                },
                "counts": {
                    "archive_occurrence_rows": archive["occurrence_rows"],
                    "archive_nonblank_associated_media_rows": (
                        archive["nonblank_reference_rows"]
                    ),
                    "archive_unique_associated_media_values": len(
                        archive["references"]
                    ),
                    "pending_rows_selected": len(pending),
                    "exact_archive_bindings": len(binding_rows),
                    "completed_rows": len(completed),
                    "attempt_rows": 0,
                },
                "artifacts": {
                    "result_shard": {
                        **result_inventory,
                        "path": str(result_path),
                    },
                    "attempt_shard": {
                        **attempt_inventory,
                        "path": str(attempt_path),
                    },
                    "archive_reference_bindings": {
                        **binding_inventory,
                        "path": "archive_reference_bindings.parquet",
                    },
                },
                "validation": {
                    "archive_manifest_checksum_recorded": True,
                    "archive_checksum_matches_manifest": True,
                    "archive_zip_crc_passed": True,
                    "archive_has_occurrence_table": True,
                    "archive_has_associated_media_field": True,
                    "archive_has_no_multimedia_extension": True,
                    "all_selected_rows_exactly_bound": len(binding_rows)
                    == len(pending),
                    "no_direct_image_associated_media_values": not direct_images,
                    "zero_active_claims_before_completion": True,
                    "all_pending_rows_completed": len(completed) == len(pending),
                    "zero_network_requests": True,
                    "attempt_shard_empty": attempt_inventory["row_count"] == 0,
                    "all_parquet_row_groups_complete": all(
                        item["row_groups_complete"]
                        for item in (
                            result_inventory,
                            attempt_inventory,
                            binding_inventory,
                        )
                    ),
                },
                "manifest_policy": {"written_last": True, "create_only": True},
            }
            if not all(manifest["validation"].values()):
                raise RuntimeError(
                    f"archive circuit validation failed: {manifest['validation']}"
                )
            _write_json(staging / "manifest.json", manifest)
            staging.replace(publication)
            return manifest
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def _validated_archive_entry(
    *,
    manifest_path: Path,
    provider: str,
    dataset_key: str,
) -> tuple[dict[str, Any], Path]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [
        entry
        for entry in value.get("archives", [])
        if str(entry.get("provider", "")).strip() == provider
        and str(entry.get("dataset_key", "")).strip() == dataset_key
    ]
    if len(matches) != 1:
        raise ValueError(
            "archive manifest must contain exactly one provider/dataset match"
        )
    entry = matches[0]
    if entry.get("intake_status") != "PASS":
        raise ValueError("archive circuit requires a PASS archive intake")
    relative = entry.get("path")
    expected_sha = str(entry.get("sha256", "")).strip().casefold()
    if not relative or len(expected_sha) != 64:
        raise ValueError("archive manifest entry lacks path or SHA-256")
    archive_path = (manifest_path.parent / str(relative)).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if _sha256_hex(archive_path) != expected_sha:
        raise ValueError("archive SHA-256 does not match pinned manifest")
    expected_bytes = entry.get("physical_bytes")
    if expected_bytes is not None and archive_path.stat().st_size != int(
        expected_bytes
    ):
        raise ValueError("archive byte count does not match pinned manifest")
    with zipfile.ZipFile(archive_path) as bundle:
        bad_member = bundle.testzip()
    if bad_member is not None:
        raise ValueError(f"archive ZIP CRC failed for member: {bad_member}")
    return entry, archive_path


def _read_reference_only_archive(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as bundle:
        try:
            meta = ET.fromstring(bundle.read("meta.xml"))
        except KeyError as exc:
            raise ValueError("archive has no meta.xml") from exc
        namespace = {"dwc": "http://rs.tdwg.org/dwc/text/"}
        core = meta.find("dwc:core", namespace)
        if core is None:
            raise ValueError("archive has no Darwin Core core table")
        if str(core.get("rowType", "")).casefold() == _OCCURRENCE_ROW_TYPE:
            occurrence_table = core
            occurrence_table_kind = "core"
        else:
            occurrence_extensions = [
                extension
                for extension in meta.findall("dwc:extension", namespace)
                if str(extension.get("rowType", "")).casefold()
                == _OCCURRENCE_ROW_TYPE
            ]
            if len(occurrence_extensions) != 1:
                raise ValueError(
                    "archive must contain exactly one Darwin Core occurrence table"
                )
            occurrence_table = occurrence_extensions[0]
            occurrence_table_kind = "extension"
        multimedia = [
            extension
            for extension in meta.findall("dwc:extension", namespace)
            if str(extension.get("rowType", "")).casefold()
            == _MULTIMEDIA_ROW_TYPE
        ]
        if multimedia:
            raise ValueError(
                "archive circuit requires absence of a Multimedia extension"
            )
        location = occurrence_table.findtext(
            "dwc:files/dwc:location",
            namespaces=namespace,
        )
        if not location:
            raise ValueError("archive occurrence table has no file location")
        associated_index = None
        occurrence_id_index = None
        for field in occurrence_table.findall("dwc:field", namespace):
            if str(field.get("term", "")).casefold() == _ASSOCIATED_MEDIA_TERM:
                associated_index = int(str(field.get("index")))
            if str(field.get("term", "")).casefold() == _OCCURRENCE_ID_TERM:
                occurrence_id_index = int(str(field.get("index")))
        if associated_index is None:
            raise ValueError("archive occurrence table has no associatedMedia field")
        if occurrence_id_index is None and occurrence_table_kind == "core":
            occurrence_id = occurrence_table.find("dwc:id", namespace)
            occurrence_id_index = (
                int(str(occurrence_id.get("index")))
                if occurrence_id is not None
                else None
            )
        delimiter = _dwca_character(
            occurrence_table.get("fieldsTerminatedBy", "\\t")
        )
        quote = _dwca_character(occurrence_table.get("fieldsEnclosedBy", ""))
        encoding = str(occurrence_table.get("encoding", "UTF-8"))
        ignore_header_lines = int(
            occurrence_table.get("ignoreHeaderLines", "0")
        )
        references: dict[str, dict[str, Any]] = {}
        occurrence_rows = 0
        nonblank_reference_rows = 0
        with ExitStack() as stack:
            raw = stack.enter_context(bundle.open(location))
            text = stack.enter_context(
                io.TextIOWrapper(raw, encoding=encoding, newline="")
            )
            reader = csv.reader(
                text,
                delimiter=delimiter,
                quotechar=quote or '"',
                quoting=csv.QUOTE_MINIMAL if quote else csv.QUOTE_NONE,
            )
            for _ in range(ignore_header_lines):
                next(reader, None)
            for row in reader:
                occurrence_rows += 1
                if associated_index >= len(row):
                    raise ValueError(
                        "archive occurrence row is shorter than associatedMedia index"
                    )
                reference = row[associated_index].strip()
                if not reference:
                    continue
                nonblank_reference_rows += 1
                match = references.setdefault(
                    reference,
                    {
                        "count": 0,
                        "first_occurrence_id": (
                            row[occurrence_id_index].strip()
                            if occurrence_id_index is not None
                            and occurrence_id_index < len(row)
                            else None
                        ),
                    },
                )
                match["count"] += 1
        return {
            "occurrence_location": location,
            "occurrence_table_kind": occurrence_table_kind,
            "occurrence_rows": occurrence_rows,
            "nonblank_reference_rows": nonblank_reference_rows,
            "references": references,
        }


def _dwca_character(value: str) -> str:
    return {
        "": "",
        "\\t": "\t",
        "\\n": "\n",
        "\\r": "\r",
    }.get(value, value)


def _looks_like_direct_image(url: str) -> bool:
    suffix = Path(urlsplit(url).path).suffix.casefold()
    return suffix in _DIRECT_IMAGE_SUFFIXES


def _write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
        row_group_size=50_000,
    )


def _parquet_inventory(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    rows = parquet.metadata.num_rows
    row_group_rows = [
        parquet.metadata.row_group(index).num_rows
        for index in range(parquet.metadata.num_row_groups)
    ]
    return {
        "physical_bytes": path.stat().st_size,
        "physical_sha256": _sha256_uri(path),
        "row_count": rows,
        "row_groups": parquet.metadata.num_row_groups,
        "row_group_rows": row_group_rows,
        "row_groups_complete": sum(row_group_rows) == rows,
        "schema": str(parquet.schema_arrow),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_uri(path: Path) -> str:
    return f"sha256:{_sha256_hex(path)}"


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _git_revision() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


__all__ = [
    "ARCHIVE_BINDING_SCHEMA",
    "ARCHIVE_CIRCUIT_VERSION",
    "TERMINAL_REASON",
    "complete_archive_reference_only_rows",
]
