from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping
from urllib.parse import urlsplit
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_final.pipeline import (
    FINAL_FILENAME,
    FINAL_SCHEMA_VERSION,
    MANIFEST_FILENAME,
)
from biominer.gbif_media_resolution.models import (
    RESULT_SCHEMA,
    SCHEMA_VERSION as RESOLUTION_SCHEMA_VERSION,
    ResolutionStatus,
    license_basis,
    source_row_id,
)


RESOLUTION_ENRICHMENT_VERSION = (
    "gbif-final-resolution-enrichment/v1"
)
RESOLUTION_FIELDS = (
    pa.field("resolved_media_identifier", pa.string()),
    pa.field("effective_media_identifier", pa.string()),
    pa.field("media_identifier_resolution_status", pa.string()),
    pa.field("media_identifier_resolution_id", pa.string()),
    pa.field("media_identifier_license_basis", pa.string()),
)
REQUIRED_SOURCE_COLUMNS = (
    "gbifID",
    "license",
    "media_identifier",
    "media_references",
    "media_license",
)
DEFAULT_EXPECTED_RESOLUTION_ROWS = 130_689
MAX_RESOLUTION_ROWS = 500_000
MAX_RESOLUTION_TABLE_BYTES = 512 * 1024 * 1024


def enrich_final_with_resolutions(
    *,
    base_publication_directory: str | Path,
    resolution_directory: str | Path,
    output_directory: str | Path,
    repository_root: str | Path,
    producer_git_sha: str,
    expected_resolution_rows: int | None = (
        DEFAULT_EXPECTED_RESOLUTION_ROWS
    ),
    batch_rows: int = 50_000,
    row_group_rows: int = 100_000,
) -> dict[str, Any]:
    """Append terminal URL-resolution evidence without dropping source rows."""

    if batch_rows <= 0 or row_group_rows <= 0:
        raise ValueError("batch_rows and row_group_rows must be positive")
    if not producer_git_sha.strip():
        raise ValueError("producer_git_sha must be non-empty")
    repository = Path(repository_root).resolve()
    _require_git_commit(repository, producer_git_sha)

    base = Path(base_publication_directory).resolve()
    resolution = Path(resolution_directory).resolve()
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite final publication: {destination}"
        )
    for dependency in (base, resolution):
        if (
            destination == dependency
            or destination.is_relative_to(dependency)
            or dependency.is_relative_to(destination)
        ):
            raise ValueError(
                "output directory must not overlap an input directory"
            )

    base_path, base_manifest_path, base_manifest = (
        _validate_base_publication(base)
    )
    (
        result_path,
        resolution_manifest_path,
        resolution_manifest,
        result_table,
        results,
        resolution_source_sha256,
    ) = _validate_resolution_publication(
        resolution,
        expected_rows=expected_resolution_rows,
    )

    source = pq.ParquetFile(base_path)
    source_schema = source.schema_arrow
    _require_source_columns(source_schema)
    collisions = set(source_schema.names) & {
        field.name for field in RESOLUTION_FIELDS
    }
    if collisions:
        raise ValueError(
            "base publication already contains resolution fields: "
            + ", ".join(sorted(collisions))
        )
    output_schema = pa.schema(
        [*source_schema, *RESOLUTION_FIELDS],
        metadata=source_schema.metadata,
    )
    indexes = {
        name: source_schema.get_field_index(name)
        for name in REQUIRED_SOURCE_COLUMNS
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = (
        destination.parent
        / f".{destination.name}.{uuid4().hex}.staging"
    )
    staging.mkdir()
    output_path = staging / FINAL_FILENAME
    writer = pq.ParquetWriter(
        output_path,
        output_schema,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    matched_result_ids: set[str] = set()
    output_status_counts: Counter[str] = Counter()
    source_rows = 0
    resolved_urls_added = 0
    missing_reference_rows = 0
    try:
        for batch in source.iter_batches(
            batch_size=batch_rows,
            use_threads=True,
        ):
            extra, batch_matches, batch_statuses = _resolution_arrays(
                batch=batch,
                indexes=indexes,
                results=results,
                resolution_source_sha256=resolution_source_sha256,
            )
            matched_result_ids.update(batch_matches)
            output_status_counts.update(batch_statuses)
            resolved_urls_added += batch_statuses.get(
                ResolutionStatus.RESOLVED.value,
                0,
            )
            missing_reference_rows += batch_statuses.get(
                "missing_reference_not_selected",
                0,
            )
            output_batch = pa.RecordBatch.from_arrays(
                [*batch.columns, *extra],
                schema=output_schema,
            )
            writer.write_batch(
                output_batch,
                row_group_size=row_group_rows,
            )
            source_rows += batch.num_rows
    except BaseException:
        writer.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    writer.close()

    with output_path.open("rb") as stream:
        os.fsync(stream.fileno())
    output_inventory = _parquet_inventory(output_path)
    output_file = pq.ParquetFile(output_path)
    unmatched_result_ids = set(results) - matched_result_ids
    resolution_rows = result_table.num_rows
    matched_status_counts = Counter(
        str(results[identity]["status"])
        for identity in matched_result_ids
    )
    unmatched_status_counts = Counter(
        str(results[identity]["status"])
        for identity in unmatched_result_ids
    )
    acceptance_gate = {
        "row_count_preserved": (
            source_rows == source.metadata.num_rows
            and output_inventory["rows"] == source_rows
        ),
        "source_schema_preserved_as_output_prefix": (
            pa.schema(
                [
                    output_file.schema_arrow.field(index)
                    for index in range(len(source_schema))
                ],
                metadata=output_file.schema_arrow.metadata,
            )
            == source_schema
        ),
        "resolution_fields_appended_exactly_once": (
            output_file.schema_arrow == output_schema
        ),
        "every_eligible_base_reference_has_terminal_result": (
            output_status_counts.get(
                "missing_resolution_result",
                0,
            )
            == 0
        ),
        "resolution_sidecar_reconciled": (
            len(matched_result_ids) + len(unmatched_result_ids)
            == resolution_rows
        ),
        "all_source_rows_retained_including_unresolved": (
            output_inventory["rows"] == source.metadata.num_rows
        ),
        "original_media_fields_retained": all(
            name in output_file.schema_arrow.names
            for name in (
                "media_identifier",
                "media_references",
                "media_license",
            )
        ),
        "row_groups_complete": bool(
            output_inventory["row_groups_complete"]
        ),
        "input_checksums_recalculated": True,
        "producer_commit_addressable": True,
        "manifest_written_last": True,
    }
    if not all(acceptance_gate.values()):
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            f"resolution enrichment validation failed: {acceptance_gate}"
        )

    published_inventory = {
        "path": FINAL_FILENAME,
        "rows": output_inventory["rows"],
        "columns": output_inventory["columns"],
        "row_groups": output_inventory["row_groups"],
        "row_group_rows": output_inventory["row_group_rows"],
        "bytes": output_inventory["bytes"],
        "sha256": output_inventory["sha256"],
    }
    manifest: dict[str, Any] = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "publication_role": (
            "terminal_resolver_integrated_source_of_truth"
        ),
        "enrichment_contract": RESOLUTION_ENRICHMENT_VERSION,
        "ground_zero_production_lineage": False,
        "created_at": _timestamp(),
        "producer_git_sha": producer_git_sha,
        "artifact": published_inventory,
        "inputs": {
            **dict(base_manifest.get("inputs") or {}),
            "base_final_artifact": {
                "path": str(base_path),
                "sha256": _sha256(base_path),
            },
            "base_publication_manifest": {
                "path": str(base_manifest_path),
                "sha256": _sha256(base_manifest_path),
            },
            "resolution_results": {
                "path": str(result_path),
                "sha256": _sha256(result_path),
            },
            "resolution_manifest": {
                "path": str(resolution_manifest_path),
                "sha256": _sha256(resolution_manifest_path),
            },
        },
        "counts": {
            "base_rows": source_rows,
            "output_rows": output_inventory["rows"],
            "resolution_rows": resolution_rows,
            "matched_resolution_rows": len(matched_result_ids),
            "unmatched_resolution_rows": len(unmatched_result_ids),
            "resolved_urls_added": resolved_urls_added,
            "missing_reference_rows": missing_reference_rows,
            "output_resolution_status_counts": dict(
                sorted(output_status_counts.items())
            ),
            "matched_resolution_status_counts": dict(
                sorted(matched_status_counts.items())
            ),
            "unmatched_resolution_status_counts": dict(
                sorted(unmatched_status_counts.items())
            ),
        },
        "resolution_reconciliation": {
            "source_artifact_sha256": resolution_source_sha256,
            "matched_result_ids": len(matched_result_ids),
            "unmatched_result_ids": len(unmatched_result_ids),
            "unmatched_reason": (
                "resolver operated on the pre-temporal-filter source; "
                "unmatched terminal results remain retained in the "
                "checksummed resolution sidecar"
            ),
            "rights_blocked_and_unresolved_policy": (
                "retain matching base rows and record terminal status; "
                "never silently discard"
            ),
        },
        "base_publication": {
            "schema_version": base_manifest.get("schema_version"),
            "producer_git_sha": base_manifest.get("producer_git_sha"),
            "artifact_sha256": _sha256(base_path),
        },
        "terminal_resolution": {
            "schema_version": resolution_manifest.get(
                "schema_version"
            ),
            "run_id": resolution_manifest.get("run_id"),
            "manifest_sha256": _sha256(resolution_manifest_path),
        },
        "configuration": {
            "batch_rows": batch_rows,
            "row_group_rows": row_group_rows,
            "maximum_resolution_rows_in_memory": MAX_RESOLUTION_ROWS,
            "maximum_resolution_bytes_in_memory": (
                MAX_RESOLUTION_TABLE_BYTES
            ),
        },
        "acceptance_gate": acceptance_gate,
        "manifest_policy": {
            "create_only": True,
            "manifest_written_last": True,
            "base_publication_unchanged": True,
            "resolution_publication_unchanged": True,
        },
    }
    _write_json(staging / MANIFEST_FILENAME, manifest)
    if (staging / MANIFEST_FILENAME).stat().st_mtime_ns < (
        output_path.stat().st_mtime_ns
    ):
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError("final manifest was not written last")
    os.replace(staging, destination)
    _fsync_directory(destination.parent)
    return manifest


def validate_resolution_enriched_publication(
    output_directory: str | Path,
    *,
    base_publication_directory: str | Path | None = None,
    resolution_directory: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Independently revalidate a terminal resolver-integrated publication."""

    output = Path(output_directory).resolve()
    final_path = output / FINAL_FILENAME
    manifest_path = output / MANIFEST_FILENAME
    if {
        path.resolve() for path in output.rglob("*") if path.is_file()
    } != {final_path, manifest_path}:
        raise RuntimeError("final publication file inventory is not exact")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FINAL_SCHEMA_VERSION:
        raise RuntimeError("final publication schema version differs")
    if (
        manifest.get("enrichment_contract")
        != RESOLUTION_ENRICHMENT_VERSION
    ):
        raise RuntimeError("resolution enrichment contract differs")
    gate = manifest.get("acceptance_gate")
    if not isinstance(gate, dict) or not gate or not all(gate.values()):
        raise RuntimeError("final publication acceptance gate is not PASS")
    inventory = _parquet_inventory(final_path)
    recorded = manifest.get("artifact") or {}
    _require_legacy_inventory_match(recorded, inventory)
    if manifest_path.stat().st_mtime_ns < final_path.stat().st_mtime_ns:
        raise RuntimeError("final publication manifest was not written last")
    schema = pq.ParquetFile(final_path).schema_arrow
    missing = [
        field.name
        for field in RESOLUTION_FIELDS
        if field.name not in schema.names
    ]
    if missing:
        raise RuntimeError(
            "final publication lacks resolution fields: "
            + ", ".join(missing)
        )
    if repository_root is not None:
        _require_git_commit(
            Path(repository_root).resolve(),
            str(manifest.get("producer_git_sha") or ""),
        )

    bindings = manifest.get("inputs") or {}
    if base_publication_directory is not None:
        base = Path(base_publication_directory).resolve()
        base_path, base_manifest_path, _ = _validate_base_publication(
            base
        )
        _require_binding(
            bindings,
            "base_final_artifact",
            base_path,
        )
        _require_binding(
            bindings,
            "base_publication_manifest",
            base_manifest_path,
        )
    if resolution_directory is not None:
        resolution = Path(resolution_directory).resolve()
        (
            result_path,
            resolution_manifest_path,
            _,
            _,
            _,
            _,
        ) = _validate_resolution_publication(
            resolution,
            expected_rows=int(
                (manifest.get("counts") or {}).get(
                    "resolution_rows",
                    -1,
                )
            ),
        )
        _require_binding(bindings, "resolution_results", result_path)
        _require_binding(
            bindings,
            "resolution_manifest",
            resolution_manifest_path,
        )
    return manifest


def _resolution_arrays(
    *,
    batch: pa.RecordBatch,
    indexes: Mapping[str, int],
    results: Mapping[str, Mapping[str, object]],
    resolution_source_sha256: str,
) -> tuple[list[pa.Array], set[str], Counter[str]]:
    raw_identifiers = batch.column(
        indexes["media_identifier"]
    ).to_pylist()
    references = batch.column(indexes["media_references"]).to_pylist()
    gbif_ids = batch.column(indexes["gbifID"]).to_pylist()
    media_licenses = batch.column(indexes["media_license"]).to_pylist()
    occurrence_licenses = batch.column(indexes["license"]).to_pylist()
    resolved_values: list[str | None] = []
    effective_values: list[str | None] = []
    statuses: list[str] = []
    resolution_ids: list[str | None] = []
    license_bases: list[str] = []
    matched: set[str] = set()
    counts: Counter[str] = Counter()
    for offset in range(batch.num_rows):
        identifier = _trimmed(raw_identifiers[offset])
        reference = _trimmed(references[offset])
        result: Mapping[str, object] | None = None
        identity: str | None = None
        if identifier is None and reference is not None:
            identity = source_row_id(
                resolution_source_sha256,
                _trimmed(gbif_ids[offset]) or "",
                reference,
            )
            result = results.get(identity)
            if result is None:
                counts["missing_resolution_result"] += 1
                raise RuntimeError(
                    "missing terminal resolution result for base row: "
                    f"{identity}"
                )
            if (
                _trimmed(result.get("gbif_id"))
                != (_trimmed(gbif_ids[offset]) or "")
                or _trimmed(result.get("media_references")) != reference
            ):
                raise RuntimeError(
                    f"resolution result source binding differs: {identity}"
                )
            matched.add(identity)

        if identifier is not None:
            resolved = None
            effective = identifier
            status = "source_identifier"
            resolution_id = None
            basis = license_basis(
                media_licenses[offset],
                occurrence_licenses[offset],
            )
        elif result is not None:
            status = str(result["status"])
            resolved = (
                _trimmed(result.get("stable_candidate_url"))
                if status == ResolutionStatus.RESOLVED.value
                else None
            )
            if status == ResolutionStatus.RESOLVED.value:
                if resolved is None or urlsplit(resolved).scheme not in {
                    "http",
                    "https",
                }:
                    raise RuntimeError(
                        f"resolved result has invalid direct URL: {identity}"
                    )
            effective = resolved
            resolution_id = str(result["source_row_id"])
            basis = str(result["license_basis"])
        else:
            resolved = None
            effective = None
            status = "missing_reference_not_selected"
            resolution_id = None
            basis = license_basis(
                media_licenses[offset],
                occurrence_licenses[offset],
            )
        resolved_values.append(resolved)
        effective_values.append(effective)
        statuses.append(status)
        resolution_ids.append(resolution_id)
        license_bases.append(basis)
        counts[status] += 1
    return (
        [
            pa.array(resolved_values, type=pa.string()),
            pa.array(effective_values, type=pa.string()),
            pa.array(statuses, type=pa.string()),
            pa.array(resolution_ids, type=pa.string()),
            pa.array(license_bases, type=pa.string()),
        ],
        matched,
        counts,
    )


def _validate_base_publication(
    base: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    final_path = base / FINAL_FILENAME
    manifest_path = base / MANIFEST_FILENAME
    expected = {final_path.resolve(), manifest_path.resolve()}
    observed = {
        path.resolve() for path in base.rglob("*") if path.is_file()
    }
    if observed != expected:
        raise RuntimeError("base publication file inventory is not exact")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FINAL_SCHEMA_VERSION:
        raise RuntimeError("base publication schema version differs")
    gate = manifest.get("acceptance_gate")
    if not isinstance(gate, dict) or not gate or not all(gate.values()):
        raise RuntimeError("base publication acceptance gate is not PASS")
    inventory = _parquet_inventory(final_path)
    _require_legacy_inventory_match(
        manifest.get("artifact") or {},
        inventory,
    )
    if manifest_path.stat().st_mtime_ns < final_path.stat().st_mtime_ns:
        raise RuntimeError("base publication manifest was not written last")
    return final_path, manifest_path, manifest


def _validate_resolution_publication(
    resolution: Path,
    *,
    expected_rows: int | None,
) -> tuple[
    Path,
    Path,
    dict[str, Any],
    pa.Table,
    dict[str, dict[str, object]],
    str,
]:
    manifest_path = resolution / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != RESOLUTION_SCHEMA_VERSION:
        raise RuntimeError("resolution manifest schema version differs")
    input_value = manifest.get("input") or {}
    if input_value.get("mode") != "full":
        raise RuntimeError("terminal enrichment requires a full resolver run")
    validation = manifest.get("validation")
    if (
        not isinstance(validation, dict)
        or not validation
        or not all(validation.values())
        or validation.get("rights_blocked_zero_attempts") is not True
    ):
        raise RuntimeError("resolution manifest validation is not PASS")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("resolution artifact inventory is absent")
    expected_files = {
        (resolution / name).resolve() for name in artifacts
    } | {manifest_path.resolve()}
    observed_files = {
        path.resolve()
        for path in resolution.rglob("*")
        if path.is_file()
    }
    if observed_files != expected_files:
        raise RuntimeError(
            "resolution publication file inventory is not exact"
        )
    newest_artifact_mtime = 0
    for name, recorded in artifacts.items():
        path = resolution / name
        if not path.is_file():
            raise FileNotFoundError(path)
        inventory = _parquet_inventory(path)
        if (
            recorded.get("physical_sha256") != inventory["sha256"]
            or int(recorded.get("row_count", -1))
            != inventory["rows"]
            or recorded.get("row_groups_complete") is not True
            or not inventory["row_groups_complete"]
        ):
            raise RuntimeError(
                f"resolution artifact inventory differs: {name}"
            )
        newest_artifact_mtime = max(
            newest_artifact_mtime,
            path.stat().st_mtime_ns,
        )
    if manifest_path.stat().st_mtime_ns < newest_artifact_mtime:
        raise RuntimeError("resolution manifest was not written last")

    result_path = resolution / "resolution_results.parquet"
    result_file = pq.ParquetFile(result_path)
    if result_file.schema_arrow != RESULT_SCHEMA:
        raise RuntimeError("resolution results schema differs")
    result_rows = result_file.metadata.num_rows
    counts = manifest.get("counts") or {}
    input_rows = int(
        input_value.get("work_rows", input_value.get("input_rows", -1))
    )
    if (
        int(counts.get("result_rows", -1)) != result_rows
        or input_rows != result_rows
        or (
            expected_rows is not None
            and result_rows != expected_rows
        )
    ):
        raise RuntimeError("resolution result row count differs")
    table = pq.read_table(result_path)
    if table.nbytes > MAX_RESOLUTION_TABLE_BYTES:
        raise MemoryError("resolution sidecar exceeds memory byte bound")
    if table.num_rows > MAX_RESOLUTION_ROWS:
        raise MemoryError("resolution sidecar exceeds memory row bound")
    rows = table.to_pylist()
    by_id = {str(row["source_row_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise RuntimeError("resolution results contain duplicate identities")
    source_sha256 = str(
        input_value.get("source_artifact_sha256") or ""
    )
    if not source_sha256.startswith("sha256:"):
        raise RuntimeError("resolution source checksum is absent")
    valid_statuses = {status.value for status in ResolutionStatus}
    for identity, row in by_id.items():
        if (
            str(row.get("source_artifact_sha256")) != source_sha256
            or str(row.get("status")) not in valid_statuses
            or str(row.get("source_row_id")) != identity
        ):
            raise RuntimeError(
                f"resolution result binding differs: {identity}"
            )
    return (
        result_path,
        manifest_path,
        manifest,
        table,
        by_id,
        source_sha256,
    )


def _parquet_inventory(path: Path) -> dict[str, Any]:
    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise RuntimeError(
            f"cannot inspect Parquet artifact: {path}"
        ) from exc
    row_group_rows = [
        parquet.metadata.row_group(index).num_rows
        for index in range(parquet.metadata.num_row_groups)
    ]
    rows = parquet.metadata.num_rows
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "rows": rows,
        "columns": parquet.metadata.num_columns,
        "row_groups": parquet.metadata.num_row_groups,
        "row_group_rows": row_group_rows,
        "row_groups_complete": (
            sum(row_group_rows) == rows
            and (rows == 0 or all(value > 0 for value in row_group_rows))
        ),
    }


def _require_legacy_inventory_match(
    recorded: Mapping[str, object],
    observed: Mapping[str, object],
) -> None:
    for key in (
        "path",
        "bytes",
        "sha256",
        "rows",
        "columns",
        "row_groups",
        "row_group_rows",
    ):
        if recorded.get(key) != observed.get(key):
            raise RuntimeError(
                f"publication artifact {key} differs"
            )
    if not observed["row_groups_complete"]:
        raise RuntimeError("publication row groups are incomplete")


def _require_source_columns(schema: pa.Schema) -> None:
    missing = [
        name for name in REQUIRED_SOURCE_COLUMNS if name not in schema.names
    ]
    if missing:
        raise ValueError(
            "base publication lacks source columns: "
            + ", ".join(missing)
        )


def _require_binding(
    bindings: Mapping[str, object],
    key: str,
    path: Path,
) -> None:
    binding = bindings.get(key)
    if (
        not isinstance(binding, Mapping)
        or binding.get("path") != str(path)
        or binding.get("sha256") != _sha256(path)
    ):
        raise RuntimeError(f"final publication input binding differs: {key}")


def _require_git_commit(repository: Path, commit: str) -> None:
    if not commit:
        raise ValueError("Git commit must be non-empty")
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(16 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _trimmed(value: object | None) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "DEFAULT_EXPECTED_RESOLUTION_ROWS",
    "RESOLUTION_ENRICHMENT_VERSION",
    "RESOLUTION_FIELDS",
    "enrich_final_with_resolutions",
    "validate_resolution_enriched_publication",
]
