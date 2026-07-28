from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.common.semantic_hash import canonical_semantic_fingerprint


PART_RECEIPT_VERSION = "gbif-final-bounded-part/v1"
ASSEMBLY_MANIFEST_VERSION = "gbif-final-bounded-assembly/v1"
STATE_CLEANUP_RECEIPT_VERSION = "gbif-final-bounded-state-cleanup/v1"
FINAL_FILENAME = "gbif_media_final_enriched.parquet"
MANIFEST_FILENAME = "manifest.json"


def seal_part(
    *,
    table: pa.Table,
    part_path: str | Path,
    source_start_ordinal: int,
    source_stop_ordinal: int,
    dependencies: Mapping[str, object],
    row_group_size: int = 100_000,
) -> dict[str, Any]:
    """Write, reopen, verify, and seal one immutable bounded output part."""

    if row_group_size <= 0:
        raise ValueError("row_group_size must be positive")
    if source_start_ordinal < 0 or source_stop_ordinal <= source_start_ordinal:
        raise ValueError("source ordinal range must be non-empty and increasing")
    expected_rows = source_stop_ordinal - source_start_ordinal
    if table.num_rows != expected_rows:
        raise ValueError(
            "part row count does not match its source ordinal range: "
            f"rows={table.num_rows}, range_rows={expected_rows}"
        )
    return seal_record_batches(
        batches=table.to_batches(max_chunksize=row_group_size),
        schema=table.schema,
        part_path=part_path,
        source_start_ordinal=source_start_ordinal,
        source_stop_ordinal=source_stop_ordinal,
        dependencies=dependencies,
        row_group_size=row_group_size,
    )


def seal_record_batches(
    *,
    batches: Iterable[pa.RecordBatch],
    schema: pa.Schema,
    part_path: str | Path,
    source_start_ordinal: int,
    source_stop_ordinal: int,
    dependencies: Mapping[str, object],
    row_group_size: int = 100_000,
) -> dict[str, Any]:
    """Stream, reopen, verify, and seal one immutable bounded output part."""

    path = Path(part_path).resolve()
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    if path.exists() or receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite sealed part: {path}")
    if row_group_size <= 0:
        raise ValueError("row_group_size must be positive")
    if source_start_ordinal < 0 or source_stop_ordinal <= source_start_ordinal:
        raise ValueError("source ordinal range must be non-empty and increasing")
    expected_rows = source_stop_ordinal - source_start_ordinal

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    rows_written = 0
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(
            temporary,
            schema,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        for batch in batches:
            if batch.schema != schema:
                raise RuntimeError("bounded-part record batch schema changed")
            if not batch.num_rows:
                continue
            rows_written += batch.num_rows
            if rows_written > expected_rows:
                raise ValueError(
                    "record batch stream exceeds its source ordinal range: "
                    f"rows={rows_written}, range_rows={expected_rows}"
                )
            writer.write_batch(batch, row_group_size=row_group_size)
        writer.close()
        writer = None
        if rows_written != expected_rows:
            raise ValueError(
                "record batch stream does not cover its source ordinal range: "
                f"rows={rows_written}, range_rows={expected_rows}"
            )
        _fsync_file(temporary)
        inventory = _parquet_inventory(temporary)
        if inventory["row_count"] != expected_rows:
            raise RuntimeError("sealed part row count changed during write")
        if not inventory["row_groups_complete"]:
            raise RuntimeError("sealed part contains incomplete Parquet row groups")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)

    return _publish_part_receipt(
        path=path,
        receipt_path=receipt_path,
        inventory=inventory,
        source_start_ordinal=source_start_ordinal,
        source_stop_ordinal=source_stop_ordinal,
        dependencies=dependencies,
    )


def _publish_part_receipt(
    *,
    path: Path,
    receipt_path: Path,
    inventory: Mapping[str, Any],
    source_start_ordinal: int,
    source_stop_ordinal: int,
    dependencies: Mapping[str, object],
) -> dict[str, Any]:
    expected_rows = source_stop_ordinal - source_start_ordinal
    dependency_payload = _normalized_dependencies(dependencies)
    body: dict[str, Any] = {
        "schema_version": PART_RECEIPT_VERSION,
        "part_id": canonical_semantic_fingerprint(
            {
                "schema_version": PART_RECEIPT_VERSION,
                "source_start_ordinal": source_start_ordinal,
                "source_stop_ordinal": source_stop_ordinal,
                "dependencies": dependency_payload,
                "schema_fingerprint": inventory["schema_fingerprint"],
                "physical_sha256": inventory["physical_sha256"],
            }
        ),
        "created_at": _timestamp(),
        "source_start_ordinal": source_start_ordinal,
        "source_stop_ordinal": source_stop_ordinal,
        "dependencies": dependency_payload,
        "dependency_fingerprint": canonical_semantic_fingerprint(
            dependency_payload
        ),
        "artifact": {
            **inventory,
            "path": path.name,
        },
        "validation": {
            "source_range_matches_rows": inventory["row_count"] == expected_rows,
            "row_groups_complete": inventory["row_groups_complete"],
            "part_reopened_after_write": True,
            "receipt_written_last": True,
        },
        "manifest_policy": {
            "part_create_only": True,
            "receipt_written_last": True,
        },
    }
    body["receipt_fingerprint"] = _receipt_fingerprint(body)
    try:
        _write_json_atomic(receipt_path, body)
    except BaseException:
        path.unlink(missing_ok=True)
        _fsync_directory(path.parent)
        raise
    return validate_part_receipt(receipt_path, expected_dependencies=dependencies)


def validate_part_receipt(
    receipt_path: str | Path,
    *,
    expected_dependencies: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Independently verify a sealed part and its dependency-bound receipt."""

    receipt = Path(receipt_path).resolve()
    if not receipt.is_file():
        raise FileNotFoundError(receipt)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PART_RECEIPT_VERSION:
        raise RuntimeError(f"unsupported bounded-part receipt: {receipt}")
    if payload.get("receipt_fingerprint") != _receipt_fingerprint(payload):
        raise RuntimeError(f"bounded-part receipt fingerprint mismatch: {receipt}")

    raw_part = str((payload.get("artifact") or {}).get("path") or "")
    part = (receipt.parent / raw_part).resolve()
    if not part.is_relative_to(receipt.parent):
        raise RuntimeError(f"bounded-part artifact escapes receipt directory: {part}")
    if not part.is_file():
        raise FileNotFoundError(part)
    recorded = payload["artifact"]
    observed = _parquet_inventory(part)
    for field in (
        "physical_bytes",
        "physical_sha256",
        "row_count",
        "column_count",
        "row_group_count",
        "row_group_rows",
        "schema_fingerprint",
    ):
        if recorded.get(field) != observed.get(field):
            raise RuntimeError(
                f"bounded-part {field} mismatch for {part}: "
                f"recorded={recorded.get(field)!r}, observed={observed.get(field)!r}"
            )
    if not observed["row_groups_complete"]:
        raise RuntimeError(f"bounded-part row groups are incomplete: {part}")

    start = int(payload["source_start_ordinal"])
    stop = int(payload["source_stop_ordinal"])
    if start < 0 or stop <= start or stop - start != observed["row_count"]:
        raise RuntimeError(f"bounded-part source range does not reconcile: {part}")
    dependencies = _normalized_dependencies(payload.get("dependencies") or {})
    if payload.get("dependency_fingerprint") != canonical_semantic_fingerprint(
        dependencies
    ):
        raise RuntimeError(f"bounded-part dependency fingerprint mismatch: {part}")
    if expected_dependencies is not None and dependencies != _normalized_dependencies(
        expected_dependencies
    ):
        raise RuntimeError(f"bounded-part dependencies are stale: {part}")
    if receipt.stat().st_mtime_ns < part.stat().st_mtime_ns:
        raise RuntimeError(f"bounded-part receipt was not written last: {part}")
    if not payload.get("validation") or not all(payload["validation"].values()):
        raise RuntimeError(f"bounded-part receipt validation is not PASS: {receipt}")
    return payload


def preflight_assembly(
    *,
    part_receipts: Iterable[str | Path],
    output_parent: str | Path,
    expected_rows: int,
    free_space_multiplier: float = 1.25,
    minimum_headroom_bytes: int = 2 * 1024**3,
) -> dict[str, Any]:
    """Validate all parts and calculate the no-spill finalization budget."""

    _validate_preflight_parameters(
        expected_rows=expected_rows,
        free_space_multiplier=free_space_multiplier,
        minimum_headroom_bytes=minimum_headroom_bytes,
    )
    receipts = _validated_receipts(part_receipts, expected_rows=expected_rows)
    projected_output_bytes = sum(
        int(receipt["artifact"]["physical_bytes"]) for receipt in receipts
    )
    return _assembly_preflight(
        receipts=receipts,
        output_parent=Path(output_parent).resolve(),
        expected_rows=expected_rows,
        projected_output_bytes=projected_output_bytes,
        free_space_multiplier=free_space_multiplier,
        minimum_headroom_bytes=minimum_headroom_bytes,
    )


def assemble_parts(
    *,
    part_receipts: Iterable[str | Path],
    output_directory: str | Path,
    expected_rows: int,
    code_commit: str,
    source_scope: Mapping[str, object],
    row_group_size: int = 100_000,
    free_space_multiplier: float = 1.25,
    minimum_headroom_bytes: int = 2 * 1024**3,
) -> dict[str, Any]:
    """Merge verified parts sequentially into one create-only final Parquet."""

    if row_group_size <= 0:
        raise ValueError("row_group_size must be positive")
    _validate_preflight_parameters(
        expected_rows=expected_rows,
        free_space_multiplier=free_space_multiplier,
        minimum_headroom_bytes=minimum_headroom_bytes,
    )
    if not code_commit.strip():
        raise ValueError("code_commit must be non-empty")
    if not source_scope:
        raise ValueError("source_scope must be non-empty")
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite final output: {destination}")
    receipts = _validated_receipts(part_receipts, expected_rows=expected_rows)
    projected_output_bytes = sum(
        int(receipt["artifact"]["physical_bytes"]) for receipt in receipts
    )
    preflight = _assembly_preflight(
        receipts=receipts,
        output_parent=destination.parent,
        expected_rows=expected_rows,
        projected_output_bytes=projected_output_bytes,
        free_space_multiplier=free_space_multiplier,
        minimum_headroom_bytes=minimum_headroom_bytes,
    )
    if preflight["status"] != "PASS":
        raise RuntimeError(
            "insufficient free space for bounded final assembly: "
            f"required={preflight['required_free_bytes']}, "
            f"free={preflight['free_bytes']}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    final_path = staging / FINAL_FILENAME
    expected_schema = pq.ParquetFile(receipts[0]["_part_path"]).schema_arrow
    writer = pq.ParquetWriter(
        final_path,
        expected_schema,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    rows_written = 0
    try:
        for receipt in receipts:
            parquet = pq.ParquetFile(receipt["_part_path"])
            if parquet.schema_arrow != expected_schema:
                raise RuntimeError("bounded final parts have inconsistent schemas")
            for batch in parquet.iter_batches(batch_size=row_group_size):
                writer.write_batch(batch, row_group_size=row_group_size)
                rows_written += batch.num_rows
    except BaseException:
        writer.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    else:
        writer.close()
        _fsync_file(final_path)

    try:
        inventory = _parquet_inventory(final_path)
        if rows_written != expected_rows or inventory["row_count"] != expected_rows:
            raise RuntimeError(
                "bounded final row count mismatch: "
                f"expected={expected_rows}, written={rows_written}, "
                f"observed={inventory['row_count']}"
            )
        if inventory["schema_fingerprint"] != receipts[0]["artifact"][
            "schema_fingerprint"
        ]:
            raise RuntimeError("bounded final schema fingerprint changed")
        if not inventory["row_groups_complete"]:
            raise RuntimeError("bounded final Parquet row groups are incomplete")

        part_evidence = [
            {
                "receipt_path": receipt["_receipt_path"],
                "receipt_sha256": _prefixed_sha256(
                    Path(receipt["_receipt_path"])
                ),
                "part_id": receipt["part_id"],
                "source_start_ordinal": receipt["source_start_ordinal"],
                "source_stop_ordinal": receipt["source_stop_ordinal"],
                "part_sha256": receipt["artifact"]["physical_sha256"],
                "row_count": receipt["artifact"]["row_count"],
                "embedded_receipt": _public_receipt(receipt),
            }
            for receipt in receipts
        ]
        manifest: dict[str, Any] = {
            "schema_version": ASSEMBLY_MANIFEST_VERSION,
            "generated_at": _timestamp(),
            "code_commit": code_commit,
            "source_scope": _normalized_dependencies(source_scope),
            "source_scope_fingerprint": canonical_semantic_fingerprint(
                _normalized_dependencies(source_scope)
            ),
            "counts": {
                "rows": inventory["row_count"],
                "columns": inventory["column_count"],
                "row_groups": inventory["row_group_count"],
                "input_parts": len(receipts),
            },
            "artifacts": [
                {
                    **inventory,
                    "path": FINAL_FILENAME,
                }
            ],
            "part_evidence": part_evidence,
            "preflight": preflight,
            "validation": {
                "source_ranges_contiguous": True,
                "one_output_row_per_source_row": inventory["row_count"]
                == expected_rows,
                "all_part_receipts_reverified": True,
                "schemas_identical": True,
                "row_groups_complete": inventory["row_groups_complete"],
                "final_artifact_reopened": True,
                "manifest_written_last": True,
            },
            "manifest_policy": {
                "create_only": True,
                "manifest_written_last": True,
                "part_receipts_embedded": True,
                "external_parts_required_after_publication": False,
            },
        }
        if not all(manifest["validation"].values()):
            raise RuntimeError("bounded final manifest validation failed")
        manifest["manifest_fingerprint"] = (
            _assembly_manifest_fingerprint(manifest)
        )
        _write_json_atomic(staging / MANIFEST_FILENAME, manifest)
        _fsync_directory(staging)
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_assembled_output(
    output_directory: str | Path,
    *,
    expected_rows: int | None = None,
    expected_code_commit: str | None = None,
    expected_source_scope: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Independently verify a completed bounded final publication."""

    destination = Path(output_directory).resolve()
    manifest_path = destination / MANIFEST_FILENAME
    final_path = destination / FINAL_FILENAME
    if not destination.is_dir():
        raise FileNotFoundError(destination)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not final_path.is_file():
        raise FileNotFoundError(final_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ASSEMBLY_MANIFEST_VERSION:
        raise RuntimeError(
            f"unsupported bounded assembly manifest: {manifest_path}"
        )
    if manifest.get(
        "manifest_fingerprint"
    ) != _assembly_manifest_fingerprint(manifest):
        raise RuntimeError(
            f"bounded assembly manifest fingerprint mismatch: {manifest_path}"
        )
    if expected_code_commit is not None and manifest.get(
        "code_commit"
    ) != expected_code_commit:
        raise RuntimeError("bounded assembly code commit is stale")
    source_scope = _normalized_dependencies(
        manifest.get("source_scope") or {}
    )
    if not source_scope or manifest.get(
        "source_scope_fingerprint"
    ) != canonical_semantic_fingerprint(source_scope):
        raise RuntimeError("bounded assembly source scope is invalid")
    if (
        expected_source_scope is not None
        and source_scope
        != _normalized_dependencies(expected_source_scope)
    ):
        raise RuntimeError("bounded assembly source scope is stale")
    validation = manifest.get("validation")
    if not isinstance(validation, dict) or not validation or not all(
        validation.values()
    ):
        raise RuntimeError("bounded assembly validation is not PASS")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise RuntimeError("bounded assembly artifact inventory is invalid")
    recorded_artifact = artifacts[0]
    if recorded_artifact.get("path") != FINAL_FILENAME:
        raise RuntimeError("bounded assembly artifact path is invalid")
    observed_artifact = _parquet_inventory(final_path)
    for field in (
        "physical_bytes",
        "physical_sha256",
        "row_count",
        "column_count",
        "row_group_count",
        "row_group_rows",
        "row_groups_complete",
        "schema_fingerprint",
        "columns",
    ):
        if recorded_artifact.get(field) != observed_artifact.get(field):
            raise RuntimeError(
                f"bounded assembly artifact {field} mismatch"
            )

    rows = int(observed_artifact["row_count"])
    if expected_rows is not None and rows != expected_rows:
        raise RuntimeError(
            "bounded assembly row count is stale: "
            f"expected={expected_rows}, observed={rows}"
        )
    counts = manifest.get("counts") or {}
    if (
        int(counts.get("rows") or -1) != rows
        or int(counts.get("columns") or -1)
        != int(observed_artifact["column_count"])
        or int(counts.get("row_groups") or -1)
        != int(observed_artifact["row_group_count"])
    ):
        raise RuntimeError("bounded assembly counts do not reconcile")

    part_evidence = manifest.get("part_evidence")
    if not isinstance(part_evidence, list) or not part_evidence:
        raise RuntimeError("bounded assembly part evidence is missing")
    receipt_paths = [
        Path(str(evidence.get("receipt_path") or "")).resolve()
        for evidence in part_evidence
        if isinstance(evidence, dict)
    ]
    if len(receipt_paths) != len(part_evidence):
        raise RuntimeError("bounded assembly part evidence is invalid")
    external_presence = [path.is_file() for path in receipt_paths]
    if any(external_presence) and not all(external_presence):
        raise RuntimeError(
            "bounded assembly external part receipts are only partially retained"
        )
    if all(external_presence):
        receipts = _validated_receipts(receipt_paths, expected_rows=rows)
        external_parts_retained = True
    else:
        policy = manifest.get("manifest_policy") or {}
        if (
            not policy.get("part_receipts_embedded")
            or policy.get("external_parts_required_after_publication")
            is not False
        ):
            raise RuntimeError(
                "bounded assembly external part receipts are missing"
            )
        receipts = _validated_embedded_receipts(
            part_evidence,
            expected_rows=rows,
        )
        external_parts_retained = False
    if len(receipts) != int(counts.get("input_parts") or -1):
        raise RuntimeError("bounded assembly input part count is stale")
    for evidence, receipt in zip(part_evidence, receipts):
        receipt_path = Path(str(evidence["receipt_path"])).resolve()
        expected = {
            "receipt_path": str(receipt_path),
            "receipt_sha256": evidence["receipt_sha256"],
            "part_id": receipt["part_id"],
            "source_start_ordinal": receipt["source_start_ordinal"],
            "source_stop_ordinal": receipt["source_stop_ordinal"],
            "part_sha256": receipt["artifact"]["physical_sha256"],
            "row_count": receipt["artifact"]["row_count"],
            "embedded_receipt": _public_receipt(receipt),
        }
        if external_parts_retained:
            expected["receipt_sha256"] = _prefixed_sha256(receipt_path)
        if evidence != expected:
            raise RuntimeError(
                f"bounded assembly part evidence mismatch: {receipt_path}"
            )
    if (
        receipts[0]["artifact"]["schema_fingerprint"]
        != observed_artifact["schema_fingerprint"]
    ):
        raise RuntimeError("bounded assembly final schema differs from parts")

    observed_files = {
        path.resolve()
        for path in destination.rglob("*")
        if path.is_file()
    }
    if observed_files != {manifest_path, final_path}:
        raise RuntimeError(
            "bounded assembly publication file inventory is invalid"
        )
    newest_input_mtime = final_path.stat().st_mtime_ns
    if external_parts_retained:
        newest_input_mtime = max(
            newest_input_mtime,
            *(
            Path(receipt["_receipt_path"]).stat().st_mtime_ns
            for receipt in receipts
            ),
        )
    if manifest_path.stat().st_mtime_ns < newest_input_mtime:
        raise RuntimeError("bounded assembly manifest was not written last")
    return manifest


def cleanup_bounded_state(
    *,
    output_directory: str | Path,
    state_directory: str | Path,
    cleanup_receipt_path: str | Path,
) -> dict[str, Any]:
    """Delete only verified bounded state after final publication is durable."""

    output = Path(output_directory).resolve()
    state = Path(state_directory).resolve()
    cleanup_receipt = Path(cleanup_receipt_path).resolve()
    if not state.is_dir() or state.is_symlink():
        raise FileNotFoundError(state)
    if (
        state == output
        or state.is_relative_to(output)
        or output.is_relative_to(state)
    ):
        raise ValueError("bounded state and final output must not overlap")
    if cleanup_receipt.exists():
        raise FileExistsError(cleanup_receipt)
    if (
        cleanup_receipt.is_relative_to(state)
        or cleanup_receipt.is_relative_to(output)
    ):
        raise ValueError(
            "cleanup receipt must be outside state and final output"
        )
    required_markers = (
        state / "source_spine" / MANIFEST_FILENAME,
        state / "pipeline" / "checkpoint.json",
    )
    if any(not path.is_file() for path in required_markers):
        raise RuntimeError(
            "bounded state lacks required source-spine or pipeline markers"
        )

    manifest = validate_assembled_output(output)
    part_paths: list[Path] = []
    for evidence in manifest["part_evidence"]:
        receipt_path = Path(evidence["receipt_path"]).resolve()
        if not receipt_path.is_relative_to(state):
            raise RuntimeError(
                "bounded final receipt is outside the requested state tree"
            )
        part_path = Path(
            str(receipt_path).removesuffix(".receipt.json")
        )
        if not receipt_path.is_file() or not part_path.is_file():
            raise RuntimeError(
                "bounded state cleanup requires every physical final part"
            )
        part_paths.extend((receipt_path, part_path))

    state_entries = list(state.rglob("*"))
    symlinks = [path for path in state_entries if path.is_symlink()]
    if symlinks:
        raise RuntimeError(
            f"bounded state cleanup refuses symlinks: {symlinks[0]}"
        )
    state_files = [path for path in state_entries if path.is_file()]
    deleted_bytes = sum(path.stat().st_size for path in state_files)
    deleted_files = len(state_files)
    final_manifest_path = output / MANIFEST_FILENAME
    final_artifact_path = output / FINAL_FILENAME
    evidence = {
        "schema_version": STATE_CLEANUP_RECEIPT_VERSION,
        "generated_at": _timestamp(),
        "state_directory": str(state),
        "deleted_file_count": deleted_files,
        "deleted_physical_bytes": deleted_bytes,
        "embedded_final_part_receipts": len(manifest["part_evidence"]),
        "final_manifest_path": str(final_manifest_path),
        "final_manifest_sha256": _prefixed_sha256(final_manifest_path),
        "final_artifact_path": str(final_artifact_path),
        "final_artifact_sha256": manifest["artifacts"][0][
            "physical_sha256"
        ],
        "final_rows": manifest["counts"]["rows"],
        "validation": {
            "final_valid_before_cleanup": True,
            "all_final_parts_inside_state": all(
                path.is_relative_to(state) for path in part_paths
            ),
            "state_markers_present": True,
            "state_contains_no_symlinks": True,
            "external_part_receipts_embedded": bool(
                manifest["manifest_policy"].get("part_receipts_embedded")
            ),
            "final_valid_after_cleanup": False,
            "cleanup_receipt_written_last": True,
        },
        "cleanup_policy": {
            "explicit_state_tree_only": True,
            "final_output_preserved": True,
            "cleanup_receipt_outside_publication": True,
        },
    }
    if not all(
        value
        for key, value in evidence["validation"].items()
        if key != "final_valid_after_cleanup"
    ):
        raise RuntimeError("bounded state cleanup preflight failed")

    shutil.rmtree(state)
    if state.exists():
        raise RuntimeError("bounded state directory still exists after cleanup")
    validated_after = validate_assembled_output(output)
    if validated_after != manifest:
        raise RuntimeError("final manifest changed during state cleanup")
    evidence["validation"]["final_valid_after_cleanup"] = True
    evidence["receipt_fingerprint"] = canonical_semantic_fingerprint(
        {
            key: value
            for key, value in evidence.items()
            if key != "receipt_fingerprint"
        }
    )
    cleanup_receipt.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(cleanup_receipt, evidence)
    return evidence


def _validated_embedded_receipts(
    part_evidence: list[object],
    *,
    expected_rows: int,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for evidence in part_evidence:
        if not isinstance(evidence, dict):
            raise RuntimeError("bounded assembly part evidence is invalid")
        embedded = evidence.get("embedded_receipt")
        if not isinstance(embedded, dict):
            raise RuntimeError("bounded assembly embedded receipt is missing")
        receipt = _normalized_dependencies(embedded)
        if receipt.get("schema_version") != PART_RECEIPT_VERSION:
            raise RuntimeError("bounded assembly embedded receipt is invalid")
        if receipt.get("receipt_fingerprint") != _receipt_fingerprint(
            receipt
        ):
            raise RuntimeError(
                "bounded assembly embedded receipt fingerprint mismatch"
            )
        dependencies = _normalized_dependencies(
            receipt.get("dependencies") or {}
        )
        if receipt.get(
            "dependency_fingerprint"
        ) != canonical_semantic_fingerprint(dependencies):
            raise RuntimeError(
                "bounded assembly embedded receipt dependencies are invalid"
            )
        artifact = receipt.get("artifact") or {}
        start = int(receipt["source_start_ordinal"])
        stop = int(receipt["source_stop_ordinal"])
        if (
            start < 0
            or stop <= start
            or stop - start != int(artifact.get("row_count") or -1)
            or not artifact.get("row_groups_complete")
            or not receipt.get("validation")
            or not all(receipt["validation"].values())
        ):
            raise RuntimeError(
                "bounded assembly embedded receipt does not reconcile"
            )
        receipts.append(receipt)
    receipts.sort(key=lambda item: int(item["source_start_ordinal"]))
    expected_start = 0
    schema_fingerprint = receipts[0]["artifact"]["schema_fingerprint"]
    for receipt in receipts:
        start = int(receipt["source_start_ordinal"])
        stop = int(receipt["source_stop_ordinal"])
        if start != expected_start:
            raise RuntimeError(
                "embedded bounded-part source ranges are not contiguous"
            )
        if receipt["artifact"]["schema_fingerprint"] != schema_fingerprint:
            raise RuntimeError(
                "embedded bounded-part schemas are inconsistent"
            )
        expected_start = stop
    if expected_start != expected_rows:
        raise RuntimeError(
            "embedded bounded parts do not cover the expected source scope"
        )
    return receipts


def _validated_receipts(
    part_receipts: Iterable[str | Path],
    *,
    expected_rows: int,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for value in part_receipts:
        path = Path(value).resolve()
        receipt = validate_part_receipt(path)
        receipt["_receipt_path"] = str(path)
        receipt["_part_path"] = str(
            (path.parent / str(receipt["artifact"]["path"])).resolve()
        )
        receipts.append(receipt)
    if not receipts:
        raise ValueError("at least one bounded part receipt is required")
    receipts.sort(key=lambda item: int(item["source_start_ordinal"]))
    expected_start = 0
    schema_fingerprint = receipts[0]["artifact"]["schema_fingerprint"]
    for receipt in receipts:
        start = int(receipt["source_start_ordinal"])
        stop = int(receipt["source_stop_ordinal"])
        if start != expected_start:
            raise RuntimeError(
                "bounded-part source ranges are not contiguous: "
                f"expected_start={expected_start}, observed_start={start}"
            )
        if receipt["artifact"]["schema_fingerprint"] != schema_fingerprint:
            raise RuntimeError("bounded parts have inconsistent schemas")
        expected_start = stop
    if expected_start != expected_rows:
        raise RuntimeError(
            "bounded parts do not cover the expected source scope: "
            f"expected_rows={expected_rows}, covered_rows={expected_start}"
        )
    return receipts


def _assembly_preflight(
    *,
    receipts: list[dict[str, Any]],
    output_parent: Path,
    expected_rows: int,
    projected_output_bytes: int,
    free_space_multiplier: float,
    minimum_headroom_bytes: int,
) -> dict[str, Any]:
    required_free_bytes = (
        int(projected_output_bytes * free_space_multiplier)
        + minimum_headroom_bytes
    )
    usage_path = output_parent
    while not usage_path.exists():
        parent = usage_path.parent
        if parent == usage_path:
            raise FileNotFoundError(
                f"no existing ancestor for output path: {output_parent}"
            )
        usage_path = parent
    free_bytes = shutil.disk_usage(usage_path).free
    return {
        "schema_version": "gbif-final-bounded-assembly-preflight/v1",
        "expected_rows": expected_rows,
        "part_count": len(receipts),
        "part_bytes": projected_output_bytes,
        "projected_output_bytes": projected_output_bytes,
        "free_space_multiplier": free_space_multiplier,
        "minimum_headroom_bytes": minimum_headroom_bytes,
        "required_free_bytes": required_free_bytes,
        "free_bytes": free_bytes,
        "disk_usage_path": str(usage_path),
        "status": "PASS" if free_bytes >= required_free_bytes else "FAIL",
    }


def _validate_preflight_parameters(
    *,
    expected_rows: int,
    free_space_multiplier: float,
    minimum_headroom_bytes: int,
) -> None:
    if expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    if free_space_multiplier < 1:
        raise ValueError("free_space_multiplier must be at least one")
    if minimum_headroom_bytes < 0:
        raise ValueError("minimum_headroom_bytes must be non-negative")


def _parquet_inventory(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    row_group_rows = [
        metadata.row_group(index).num_rows
        for index in range(metadata.num_row_groups)
    ]
    return {
        "physical_bytes": path.stat().st_size,
        "physical_sha256": _prefixed_sha256(path),
        "row_count": metadata.num_rows,
        "column_count": len(parquet.schema_arrow),
        "row_group_count": metadata.num_row_groups,
        "row_group_rows": row_group_rows,
        "row_groups_complete": (
            metadata.num_rows > 0
            and metadata.num_row_groups > 0
            and sum(row_group_rows) == metadata.num_rows
            and all(rows > 0 for rows in row_group_rows)
        ),
        "schema_fingerprint": _schema_fingerprint(parquet.schema_arrow),
        "columns": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in parquet.schema_arrow
        ],
    }


def _schema_fingerprint(schema: pa.Schema) -> str:
    return "sha256:" + hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _receipt_fingerprint(payload: Mapping[str, object]) -> str:
    body = {
        key: value
        for key, value in payload.items()
        if key not in {"receipt_fingerprint", "_receipt_path", "_part_path"}
    }
    return canonical_semantic_fingerprint(body)


def _public_receipt(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    return _normalized_dependencies(
        {
            key: value
            for key, value in receipt.items()
            if key not in {"_receipt_path", "_part_path"}
        }
    )


def _assembly_manifest_fingerprint(
    payload: Mapping[str, object],
) -> str:
    body = {
        key: value
        for key, value in payload.items()
        if key != "manifest_fingerprint"
    }
    return canonical_semantic_fingerprint(body)


def _normalized_dependencies(
    value: Mapping[str, object],
) -> dict[str, object]:
    return json.loads(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )


def _prefixed_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _fsync_file(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "ASSEMBLY_MANIFEST_VERSION",
    "FINAL_FILENAME",
    "MANIFEST_FILENAME",
    "PART_RECEIPT_VERSION",
    "STATE_CLEANUP_RECEIPT_VERSION",
    "assemble_parts",
    "cleanup_bounded_state",
    "preflight_assembly",
    "seal_record_batches",
    "seal_part",
    "validate_assembled_output",
    "validate_part_receipt",
]
