from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.gbif_media_resolution.models import (
    ATTEMPT_SCHEMA,
    JOB_NAME,
    RESULT_SCHEMA,
    SCHEMA_VERSION,
    STAGE,
    ResolutionInput,
    ResolutionResult,
    ResolutionStatus,
    is_explicitly_restricted,
    license_basis,
    source_row_id,
)
from biominer.gbif_media_resolution.adapters import (
    DEFAULT_PROVIDER_ADAPTERS,
    DEFAULT_PROVIDER_ADAPTER_VERSIONS,
)
from biominer.gbif_media_resolution.resolver import (
    MediaURLResolver,
    RESOLVER_VERSION,
    ResolverConfig,
)
from biominer.workstore.base import WorkStore


V4_SCHEMA_VERSION = "biominer-gbif-media-url-database/v4"
REQUIRED_SOURCE_COLUMNS = (
    "gbifID",
    "license",
    "media_type",
    "media_format",
    "media_identifier",
    "media_references",
    "media_license",
)
V4_FIELDS = (
    pa.field("resolved_media_identifier", pa.string()),
    pa.field("effective_media_identifier", pa.string()),
    pa.field("media_identifier_resolution_status", pa.string()),
    pa.field("media_identifier_resolution_id", pa.string()),
    pa.field("media_identifier_license_basis", pa.string()),
)
PILOT_CONTEXT_COLUMNS = (
    "publisher",
    "datasetName",
    "media_publisher",
    "taxonRank",
    "countryCode",
)
PILOT_SELECTION_SCHEMA = pa.schema(
    [
        ("source_row_id", pa.string()),
        ("gbifID", pa.string()),
        ("media_references", pa.string()),
        ("media_host", pa.string()),
        ("host_population_rows", pa.int64()),
        ("host_size_band", pa.string()),
        ("provider", pa.string()),
        ("publisher", pa.string()),
        ("dataset_name", pa.string()),
        ("url_pattern", pa.string()),
        ("license_state", pa.string()),
        ("reference_type", pa.string()),
        ("taxon_rank", pa.string()),
        ("country_code", pa.string()),
        ("expected_adapter", pa.string()),
        ("rights_blocked", pa.bool_()),
        ("selection_stratum", pa.string()),
        ("selection_hash", pa.string()),
    ]
)


def prepare_resolution(
    *,
    source: str | Path,
    source_manifest: str | Path,
    output_root: str | Path,
    workstore: WorkStore,
    run_id: str,
    expected_missing_rows: int | None = 130_689,
    enqueue_batch_rows: int = 1_000,
    mode: str = "pilot",
    expected_rights_blocked_rows: int | None = None,
    resolver_config: ResolverConfig | None = None,
    pilot_acceptance_manifest: str | Path | None = None,
) -> dict[str, Any]:
    source_path = Path(source).resolve()
    manifest_path = Path(source_manifest).resolve()
    root = Path(output_root).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if enqueue_batch_rows <= 0:
        raise ValueError("enqueue_batch_rows must be positive")
    if mode not in {"pilot", "full"}:
        raise ValueError("mode must be pilot or full")
    run_id = str(run_id).strip()
    if not run_id:
        raise ValueError("run_id must be nonblank")

    source_file = pq.ParquetFile(source_path)
    _require_columns(source_file.schema_arrow, REQUIRED_SOURCE_COLUMNS)
    source_sha = "sha256:" + _sha256(source_path)
    pilot_acceptance = _validate_pilot_acceptance(
        pilot_acceptance_manifest,
        source_sha256=source_sha,
        required=mode == "full",
    )
    source_manifest_sha = "sha256:" + _sha256(manifest_path)
    effective_resolver_config = resolver_config or ResolverConfig()
    resolver_fingerprint = canonical_semantic_fingerprint(
        {
            "resolver_version": RESOLVER_VERSION,
            "config_fingerprint": effective_resolver_config.fingerprint,
            "provider_adapters": [
                {"adapter_id": adapter_id, "version": version}
                for adapter_id, version in DEFAULT_PROVIDER_ADAPTER_VERSIONS
            ],
        }
    )
    selected = 0
    rights_blocked = 0
    identities: set[str] = set()
    all_inputs: list[ResolutionInput] = []
    scan_columns = list(REQUIRED_SOURCE_COLUMNS) + [
        name for name in PILOT_CONTEXT_COLUMNS if name in source_file.schema_arrow.names
    ]
    for batch in source_file.iter_batches(
        batch_size=max(enqueue_batch_rows, 10_000),
        columns=scan_columns,
        use_threads=True,
    ):
        values = {
            name: batch.column(index).to_pylist()
            for index, name in enumerate(scan_columns)
        }
        for name in PILOT_CONTEXT_COLUMNS:
            values.setdefault(name, [None] * batch.num_rows)
        for offset in range(batch.num_rows):
            identifier = _trimmed(values["media_identifier"][offset])
            reference = _trimmed(values["media_references"][offset])
            if identifier is not None or reference is None:
                continue
            gbif_id = _trimmed(values["gbifID"][offset])
            if gbif_id is None:
                raise ValueError("affected row has a blank gbifID")
            identity = source_row_id(source_sha, gbif_id, reference)
            if identity in identities:
                raise ValueError("affected source identity is not unique")
            identities.add(identity)
            media_license = _optional(values["media_license"][offset])
            item = ResolutionInput(
                source_row_id=identity,
                source_artifact_sha256=source_sha,
                gbif_id=gbif_id,
                media_references=reference,
                media_type=_optional(values["media_type"][offset]),
                media_format=_optional(values["media_format"][offset]),
                media_license=media_license,
                occurrence_license=_optional(values["license"][offset]),
                provider=_optional(values["media_publisher"][offset])
                or _optional(values["publisher"][offset]),
                publisher=_optional(values["publisher"][offset]),
                dataset_name=_optional(values["datasetName"][offset]),
                taxon_rank=_optional(values["taxonRank"][offset]),
                country_code=_optional(values["countryCode"][offset]),
            )
            all_inputs.append(item)
            selected += 1
            rights_blocked += int(is_explicitly_restricted(media_license))
    if expected_missing_rows is not None and selected != expected_missing_rows:
        raise ValueError(
            f"missing media_identifier count mismatch: expected {expected_missing_rows}, found {selected}"
        )
    if (
        expected_rights_blocked_rows is not None
        and rights_blocked != expected_rights_blocked_rows
    ):
        raise ValueError(
            "rights-blocked count mismatch: "
            f"expected {expected_rights_blocked_rows}, found {rights_blocked}"
        )
    work_inputs = select_pilot_inputs(all_inputs) if mode == "pilot" else all_inputs
    work_rights_blocked = sum(item.rights_blocked for item in work_inputs)
    config = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source_path),
        "source_artifact_sha256": source_sha,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": source_manifest_sha,
        "source_rows": source_file.metadata.num_rows,
        "output_root": str(root),
        "expected_missing_rows": expected_missing_rows,
        "expected_rights_blocked_rows": expected_rights_blocked_rows,
        "input_rows": selected,
        "work_rows": len(work_inputs),
        "mode": mode,
        "resolver_version": RESOLVER_VERSION,
        "resolver_config_fingerprint": effective_resolver_config.fingerprint,
        "provider_adapter_versions": [
            {"adapter_id": adapter_id, "version": version}
            for adapter_id, version in DEFAULT_PROVIDER_ADAPTER_VERSIONS
        ],
        "resolver_fingerprint": resolver_fingerprint,
        "baseline_maturity": "legacy_v3_migration_not_ground_zero_production",
        "pilot_acceptance_manifest": (
            str(pilot_acceptance["path"]) if pilot_acceptance is not None else None
        ),
        "pilot_acceptance_manifest_sha256": (
            pilot_acceptance["sha256"] if pilot_acceptance is not None else None
        ),
    }
    run = workstore.get_or_create_run(
        job_name=JOB_NAME,
        stage=STAGE,
        run_id=run_id,
        registry_version=source_sha,
        config=config,
    )
    if run["config"] != config:
        raise ValueError("run_id already exists with incompatible configuration")
    inserted = 0
    for offset in range(0, len(work_inputs), enqueue_batch_rows):
        payloads: list[dict[str, Any]] = []
        for item in work_inputs[offset : offset + enqueue_batch_rows]:
            payload = item.to_payload()
            payload["work_key"] = item.source_row_id
            payloads.append(payload)
        inserted += workstore.enqueue_work(
            JOB_NAME,
            source_sha,
            payloads,
            stage=STAGE,
        )

    root.mkdir(parents=True, exist_ok=True)
    pilot_artifact = None
    if mode == "pilot":
        pilot_path = root / f"pilot-selection-{run_id}.parquet"
        _write_parquet_create_only(
            pilot_path,
            pilot_selection_table(work_inputs, population=all_inputs),
        )
        pilot_artifact = _parquet_inventory(pilot_path)
    receipt = {
        **config,
        "run_id": run_id,
        "input_rows": selected,
        "rights_blocked_rows": rights_blocked,
        "eligible_resolution_rows": selected - rights_blocked,
        "mode": mode,
        "work_rows": len(work_inputs),
        "work_rights_blocked_rows": work_rights_blocked,
        "newly_enqueued_rows": inserted,
        "prepared_at": _timestamp(),
        "git_commit": _git_revision(),
        "pilot_selection_artifact": pilot_artifact,
    }
    _write_json_idempotent(root / f"prepare-{run_id}.json", receipt)
    return receipt


def run_worker(
    *,
    workstore: WorkStore,
    output_root: str | Path,
    run_id: str,
    worker_id: str,
    batch_rows: int = 25,
    stale_after_seconds: int = 900,
    resolver: MediaURLResolver | None = None,
) -> dict[str, Any]:
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    if stale_after_seconds <= 0:
        raise ValueError("stale_after_seconds must be positive")
    run = workstore.get_run(run_id=run_id)
    if run is None or run["job_name"] != JOB_NAME or run["stage"] != STAGE:
        raise ValueError(f"unknown GBIF URL resolution run: {run_id}")
    registry_version = str(run["registry_version"])
    active_resolver = resolver or MediaURLResolver(
        request_guard=lambda host: workstore.publication_lock(
            f"{JOB_NAME}:origin:{host}"
        )
    )
    owns_resolver = resolver is None
    if active_resolver.semantic_fingerprint != run["config"].get(
        "resolver_fingerprint"
    ):
        if owns_resolver:
            active_resolver.close()
        raise ValueError("worker resolver configuration does not match prepared run")
    requeued = workstore.requeue_stale_claims(
        job_name=JOB_NAME,
        stage=STAGE,
        registry_version=registry_version,
        stale_after_seconds=stale_after_seconds,
    )
    claimed = workstore.claim_next_batch(
        worker_id,
        batch_rows,
        job_name=JOB_NAME,
        stage=STAGE,
        registry_version=registry_version,
    )
    if not claimed:
        if owns_resolver:
            active_resolver.close()
        return {
            "run_id": run_id,
            "worker_id": worker_id,
            "claimed_rows": 0,
            "completed_rows": 0,
            "requeued_stale_rows": requeued,
        }

    results: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    try:
        for work_item in claimed:
            item = ResolutionInput.from_payload(dict(work_item["payload"]))
            try:
                result, item_attempts = active_resolver.resolve(item)
            except Exception as exc:  # noqa: BLE001 - preserve a terminal evidence row.
                result = _worker_exception_result(item, exc)
                item_attempts = ()
            results.append(result.to_row())
            attempts.extend(attempt.to_row() for attempt in item_attempts)
    finally:
        if owns_resolver:
            active_resolver.close()

    shard_token = canonical_semantic_fingerprint(
        {
            "run_id": run_id,
            "claims": [
                [str(item["work_key"]), int(item["attempt_count"])] for item in claimed
            ],
        }
    ).split(":", 1)[1]
    root = Path(output_root).resolve()
    result_path = root / "shards" / "results" / f"part-{shard_token}.parquet"
    attempt_path = root / "shards" / "attempts" / f"part-{shard_token}.parquet"
    _write_parquet_create_only(
        result_path,
        pa.Table.from_pylist(results, schema=RESULT_SCHEMA),
    )
    _write_parquet_create_only(
        attempt_path,
        pa.Table.from_pylist(attempts, schema=ATTEMPT_SCHEMA),
    )
    result_sha = "sha256:" + _sha256(result_path)
    attempt_sha = "sha256:" + _sha256(attempt_path)
    workstore.register_shard(
        shard_id=shard_token,
        job_name=JOB_NAME,
        registry_version=registry_version,
        stage=STAGE,
        run_id=run_id,
        worker_id=worker_id,
        uri=str(result_path),
        checksum=result_sha,
        row_count=len(results),
        byte_count=result_path.stat().st_size,
        metadata={
            "attempt_uri": str(attempt_path),
            "attempt_sha256": attempt_sha,
            "attempt_rows": len(attempts),
        },
    )
    for work_item in claimed:
        committed = workstore.complete_claim(
            str(work_item["work_key"]),
            worker_id=worker_id,
            attempt_count=int(work_item["attempt_count"]),
            stale_after_seconds=stale_after_seconds,
            output_uri=str(result_path),
            checksum=result_sha,
            row_count=1,
        )
        if not committed:
            raise RuntimeError(f"claim lease was lost before commit: {work_item['work_key']}")
    return {
        "run_id": run_id,
        "worker_id": worker_id,
        "claimed_rows": len(claimed),
        "completed_rows": len(results),
        "attempt_rows": len(attempts),
        "requeued_stale_rows": requeued,
        "result_shard": str(result_path),
        "result_sha256": result_sha,
        "attempt_shard": str(attempt_path),
        "attempt_sha256": attempt_sha,
    }


def finalize_resolution(
    *,
    workstore: WorkStore,
    run_id: str,
    output_root: str | Path,
    output_directory: str | Path,
    expected_rows: int | None = 130_689,
) -> dict[str, Any]:
    run = workstore.get_run(run_id=run_id)
    if run is None or run["job_name"] != JOB_NAME:
        raise ValueError(f"unknown GBIF URL resolution run: {run_id}")
    registry_version = str(run["registry_version"])
    work_items = workstore.list_work_items(
        job_name=JOB_NAME,
        stage=STAGE,
        registry_version=registry_version,
    )
    effective_expected = (
        int(run["config"]["work_rows"])
        if expected_rows is None
        else expected_rows
    )
    if len(work_items) != effective_expected:
        raise ValueError(
            f"work item count mismatch: expected {effective_expected}, found {len(work_items)}"
        )
    non_completed = Counter(str(item["status"]) for item in work_items if item["status"] != "completed")
    if non_completed:
        raise RuntimeError(f"resolution run is incomplete: {dict(non_completed)}")

    source_ids_by_result_uri: dict[str, set[str]] = {}
    for item in work_items:
        uri = str(item["output_uri"])
        source_ids_by_result_uri.setdefault(uri, set()).add(str(item["work_key"]))
    result_paths = sorted(Path(uri) for uri in source_ids_by_result_uri)
    shard_root = (Path(output_root).resolve() / "shards").resolve()
    if not result_paths and work_items:
        raise RuntimeError("completed work items have no result shards")
    committed = workstore.list_committed_shards(
        job_name=JOB_NAME,
        stage=STAGE,
        registry_version=registry_version,
        run_id=run_id,
    )
    metadata_by_uri = {str(item["uri"]): item["metadata"] for item in committed}
    attempt_selections: list[tuple[Path, set[str]]] = []
    for result_path in result_paths:
        resolved_result_path = result_path.resolve()
        if not resolved_result_path.is_relative_to(shard_root):
            raise RuntimeError(f"result shard escapes the run output root: {result_path}")
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        metadata = metadata_by_uri.get(str(result_path))
        if metadata is None:
            raise RuntimeError(f"result shard is not registered: {result_path}")
        attempt_path = Path(str(metadata["attempt_uri"]))
        if not attempt_path.resolve().is_relative_to(shard_root):
            raise RuntimeError(f"attempt shard escapes the run output root: {attempt_path}")
        if not attempt_path.is_file():
            raise FileNotFoundError(attempt_path)
        if "sha256:" + _sha256(attempt_path) != metadata["attempt_sha256"]:
            raise RuntimeError(f"attempt shard checksum mismatch: {attempt_path}")
        attempt_selections.append(
            (attempt_path, source_ids_by_result_uri[str(result_path)])
        )

    result_table = _read_selected_tables(
        [
            (path, source_ids_by_result_uri[str(path)])
            for path in result_paths
        ],
        RESULT_SCHEMA,
    )
    if result_table.num_rows != len(work_items):
        raise RuntimeError("result row count does not match completed work items")
    source_ids = result_table.column("source_row_id").to_pylist()
    if len(set(source_ids)) != len(source_ids):
        raise RuntimeError("resolution reducer found duplicate source rows")
    result_table = result_table.sort_by([("source_row_id", "ascending")])
    attempt_table = _read_selected_tables(attempt_selections, ATTEMPT_SCHEMA).sort_by(
        [("source_row_id", "ascending"), ("sequence", "ascending")]
    )
    status_values = result_table.column("status").to_pylist()
    status_counts = Counter(str(value) for value in status_values)
    unresolved_table = result_table.filter(
        pc.not_equal(result_table.column("status"), ResolutionStatus.RESOLVED.value)
    )

    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    try:
        artifacts = {
            "resolution_attempts.parquet": attempt_table,
            "resolution_results.parquet": result_table,
            "unresolved_rows.parquet": unresolved_table,
        }
        entries: dict[str, Any] = {}
        for name, table in artifacts.items():
            path = staging / name
            pq.write_table(
                table,
                path,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
                row_group_size=50_000,
            )
            entries[name] = _parquet_inventory(path)
        rights_blocked = status_counts.get(ResolutionStatus.RIGHTS_BLOCKED.value, 0)
        blocked_attempts = attempt_table.filter(
            pc.is_in(
                attempt_table.column("source_row_id"),
                value_set=pa.array(
                    result_table.filter(
                        pc.equal(
                            result_table.column("status"),
                            ResolutionStatus.RIGHTS_BLOCKED.value,
                        )
                    ).column("source_row_id").to_pylist(),
                    type=pa.string(),
                ),
            )
        ).num_rows
        if blocked_attempts:
            raise RuntimeError("rights-blocked rows contain network attempts")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _timestamp(),
            "run_id": run_id,
            "git_commit": _git_revision(),
            "baseline_maturity": "legacy_v3_migration_not_ground_zero_production",
            "input": run["config"],
            "counts": {
                "result_rows": result_table.num_rows,
                "attempt_rows": attempt_table.num_rows,
                "unresolved_rows": unresolved_table.num_rows,
                "resolved_rows": status_counts.get(ResolutionStatus.RESOLVED.value, 0),
                "rights_blocked_rows": rights_blocked,
                "eligible_resolution_rows": result_table.num_rows - rights_blocked,
                "status_counts": dict(sorted(status_counts.items())),
            },
            "artifacts": entries,
            "validation": {
                "one_result_per_input": result_table.num_rows == len(work_items),
                "unique_source_row_ids": len(set(source_ids)) == len(source_ids),
                "every_work_item_completed": not non_completed,
                "rights_blocked_zero_attempts": blocked_attempts == 0,
                "all_parquet_row_groups_complete": all(
                    value["row_groups_complete"] for value in entries.values()
                ),
            },
            "manifest_policy": {"written_last": True},
        }
        if not all(manifest["validation"].values()):
            raise RuntimeError(f"resolution publication validation failed: {manifest['validation']}")
        _write_json(staging / "manifest.json", manifest)
        staging.replace(destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def publish_v4(
    *,
    source: str | Path,
    source_manifest: str | Path,
    resolution_directory: str | Path,
    output_directory: str | Path,
    batch_rows: int = 50_000,
    duckdb_memory_limit: str = "24GB",
    duckdb_threads: int = 8,
) -> dict[str, Any]:
    if batch_rows <= 0 or duckdb_threads <= 0:
        raise ValueError("batch_rows and duckdb_threads must be positive")
    source_path = Path(source).resolve()
    source_manifest_path = Path(source_manifest).resolve()
    resolution_root = Path(resolution_directory).resolve()
    destination = Path(output_directory).resolve()
    for path in (source_path, source_manifest_path, resolution_root / "manifest.json", resolution_root / "resolution_results.parquet"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {destination}")
    source_sha = "sha256:" + _sha256(source_path)
    resolution_manifest = json.loads((resolution_root / "manifest.json").read_text(encoding="utf-8"))
    if resolution_manifest["input"].get("mode") != "full":
        raise ValueError("v4 publication requires a completed full resolution run")
    if int(resolution_manifest["counts"]["result_rows"]) != int(
        resolution_manifest["input"]["input_rows"]
    ):
        raise ValueError("v4 publication requires one result for every affected input row")
    if resolution_manifest["input"]["source_artifact_sha256"] != source_sha:
        raise ValueError("resolution artifact was produced from a different source")
    result_inventory = resolution_manifest["artifacts"]["resolution_results.parquet"]
    if "sha256:" + _sha256(resolution_root / "resolution_results.parquet") != result_inventory["physical_sha256"]:
        raise ValueError("resolution result checksum mismatch")

    source_file = pq.ParquetFile(source_path)
    _require_columns(source_file.schema_arrow, REQUIRED_SOURCE_COLUMNS)
    collisions = set(source_file.schema_arrow.names) & {field.name for field in V4_FIELDS}
    if collisions:
        raise ValueError(f"source already contains v4 columns: {sorted(collisions)}")
    result_rows = pq.read_table(resolution_root / "resolution_results.parquet").to_pylist()
    results = {str(row["source_row_id"]): row for row in result_rows}
    if len(results) != len(result_rows):
        raise ValueError("resolution results contain duplicate source identities")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    parquet_path = staging / "gbif_media.parquet"
    output_schema = source_file.schema_arrow.append(V4_FIELDS[0]).append(V4_FIELDS[1]).append(V4_FIELDS[2]).append(V4_FIELDS[3]).append(V4_FIELDS[4])
    indexes = {name: source_file.schema_arrow.get_field_index(name) for name in REQUIRED_SOURCE_COLUMNS}
    writer = pq.ParquetWriter(
        parquet_path,
        output_schema,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    source_rows = 0
    output_rows = 0
    rights_excluded = 0
    resolved_added = 0
    try:
        for batch in source_file.iter_batches(batch_size=batch_rows, use_threads=True):
            raw_identifier = batch.column(indexes["media_identifier"]).to_pylist()
            references = batch.column(indexes["media_references"]).to_pylist()
            gbif_ids = batch.column(indexes["gbifID"]).to_pylist()
            media_licenses = batch.column(indexes["media_license"]).to_pylist()
            occurrence_licenses = batch.column(indexes["license"]).to_pylist()
            keep: list[bool] = []
            resolved_values: list[str | None] = []
            effective_values: list[str | None] = []
            statuses: list[str] = []
            resolution_ids: list[str | None] = []
            license_bases: list[str] = []
            for offset in range(batch.num_rows):
                identifier = _trimmed(raw_identifier[offset])
                reference = _trimmed(references[offset])
                result: dict[str, Any] | None = None
                if identifier is None and reference is not None:
                    identity = source_row_id(source_sha, _trimmed(gbif_ids[offset]) or "", reference)
                    result = results.get(identity)
                    if result is None:
                        raise RuntimeError(f"missing resolution result for source row: {identity}")
                if result is not None and result["status"] == ResolutionStatus.RIGHTS_BLOCKED.value:
                    keep.append(False)
                    rights_excluded += 1
                    continue
                keep.append(True)
                if identifier is not None:
                    resolved_values.append(None)
                    effective_values.append(identifier)
                    statuses.append("source_identifier")
                    resolution_ids.append(None)
                elif result is not None:
                    resolved = _trimmed(result.get("stable_candidate_url")) if result["status"] == ResolutionStatus.RESOLVED.value else None
                    resolved_values.append(resolved)
                    effective_values.append(resolved)
                    statuses.append(str(result["status"]))
                    resolution_ids.append(str(result["source_row_id"]))
                    resolved_added += int(resolved is not None)
                else:
                    resolved_values.append(None)
                    effective_values.append(None)
                    statuses.append("missing_reference")
                    resolution_ids.append(None)
                license_bases.append(license_basis(media_licenses[offset], occurrence_licenses[offset]))
            source_rows += batch.num_rows
            filtered = pa.RecordBatch.from_arrays(
                [pc.filter(column, pa.array(keep)) for column in batch.columns],
                schema=batch.schema,
            )
            table = pa.Table.from_batches([filtered])
            table = table.append_column(V4_FIELDS[0], pa.array(resolved_values, pa.string()))
            table = table.append_column(V4_FIELDS[1], pa.array(effective_values, pa.string()))
            table = table.append_column(V4_FIELDS[2], pa.array(statuses, pa.string()))
            table = table.append_column(V4_FIELDS[3], pa.array(resolution_ids, pa.string()))
            table = table.append_column(V4_FIELDS[4], pa.array(license_bases, pa.string()))
            if table.num_rows:
                writer.write_table(table, row_group_size=batch_rows)
            output_rows += table.num_rows
    finally:
        writer.close()
    expected_rights = int(resolution_manifest["counts"]["rights_blocked_rows"])
    if rights_excluded != expected_rights:
        raise RuntimeError(f"rights exclusion mismatch: expected {expected_rights}, found {rights_excluded}")
    if output_rows != source_rows - rights_excluded:
        raise RuntimeError("v4 row reconciliation failed")

    database_path = staging / "gbif_media.duckdb"
    temporary = staging / "duckdb_tmp"
    temporary.mkdir()
    con = duckdb.connect(str(database_path))
    try:
        escaped = str(temporary).replace("'", "''")
        con.execute(f"SET temp_directory='{escaped}'")
        con.execute(f"SET memory_limit='{duckdb_memory_limit}'")
        con.execute(f"SET threads={duckdb_threads}")
        con.execute("CREATE TABLE gbif_media AS SELECT * FROM read_parquet(?)", [str(parquet_path)])
        con.execute("CREATE INDEX idx_gbif_media_media_identifier ON gbif_media(media_identifier)")
        con.execute("CREATE INDEX idx_gbif_media_effective_identifier ON gbif_media(effective_media_identifier)")
        con.execute("CREATE INDEX idx_gbif_media_gbif_id ON gbif_media(gbifID)")
        con.execute("ANALYZE gbif_media")
        con.execute("CHECKPOINT")
        db_rows = int(con.execute("SELECT COUNT(*) FROM gbif_media").fetchone()[0])
        db_columns = [(str(row[0]), str(row[1])) for row in con.execute("DESCRIBE gbif_media").fetchall()]
        index_rows = [
            {
                "index_name": str(row[0]),
                "expressions": str(row[1]),
                "sql": str(row[2]),
            }
            for row in con.execute(
                "SELECT index_name, expressions, sql FROM duckdb_indexes() ORDER BY index_name"
            ).fetchall()
        ]
        sample = con.execute(
            "SELECT effective_media_identifier FROM gbif_media WHERE effective_media_identifier IS NOT NULL LIMIT 1"
        ).fetchone()
        benchmark = _benchmark_effective_url(con, str(sample[0])) if sample else None
    finally:
        con.close()
        shutil.rmtree(temporary, ignore_errors=True)

    parquet_inventory = _parquet_inventory(parquet_path)
    expected_columns = output_schema.names
    validation = {
        "source_sha256_recalculated": True,
        "resolution_checksum_matches_manifest": True,
        "row_count_reconciled": output_rows == source_rows - rights_excluded,
        "database_row_count_matches_parquet": db_rows == output_rows,
        "database_column_names_match_parquet": [name for name, _ in db_columns] == expected_columns,
        "parquet_row_groups_complete": parquet_inventory["row_groups_complete"],
        "rights_blocked_rows_excluded": rights_excluded == expected_rights,
    }
    if not all(validation.values()):
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(f"v4 validation failed: {validation}")
    manifest = {
        "schema_version": V4_SCHEMA_VERSION,
        "generated_at": _timestamp(),
        "git_commit": _git_revision(),
        "baseline_maturity": "legacy_v3_migration_not_ground_zero_production",
        "input": {
            "source_path": str(source_path),
            "source_sha256": source_sha,
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": "sha256:" + _sha256(source_manifest_path),
            "resolution_directory": str(resolution_root),
            "resolution_manifest_sha256": "sha256:" + _sha256(resolution_root / "manifest.json"),
        },
        "counts": {
            "source_rows": source_rows,
            "rights_blocked_rows_excluded": rights_excluded,
            "resolved_urls_added": resolved_added,
            "output_rows": output_rows,
            "output_columns": len(expected_columns),
        },
        "parquet": parquet_inventory,
        "database": {
            "path": database_path.name,
            "physical_bytes": database_path.stat().st_size,
            "physical_sha256": "sha256:" + _sha256(database_path),
            "row_count": db_rows,
            "columns": [{"name": name, "type": dtype} for name, dtype in db_columns],
            "indexes": index_rows,
            "effective_url_lookup_benchmark": benchmark,
        },
        "validation": validation,
        "manifest_policy": {"written_last": True},
    }
    _write_json(staging / "manifest.json", manifest)
    staging.replace(destination)
    return manifest


def _worker_exception_result(item: ResolutionInput, exc: Exception) -> ResolutionResult:
    now = _timestamp()
    reason = f"worker_exception:{type(exc).__name__}"
    identity = {
        "contract": RESOLVER_VERSION,
        "source_row_id": item.source_row_id,
        "status": ResolutionStatus.RETRY_EXHAUSTED.value,
        "reason": reason,
    }
    return ResolutionResult(
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
        status=ResolutionStatus.RETRY_EXHAUSTED,
        method="worker_exception",
        stable_candidate_url=None,
        validated_final_url=None,
        redirect_count=0,
        declared_content_type=None,
        detected_content_type=None,
        bytes_sampled=0,
        probe_prefix_sha256=None,
        content_sha256=None,
        content_hash_status="deferred",
        adapter_version=RESOLVER_VERSION,
        attempt_count=0,
        terminal_reason=reason,
        resolved_at=now,
        provenance_fingerprint=canonical_semantic_fingerprint(identity),
    )


def select_pilot_inputs(inputs: list[ResolutionInput]) -> list[ResolutionInput]:
    """Return a deterministic, host-capped, multidimensional pilot workload."""
    by_host: dict[str, list[ResolutionInput]] = {}
    for item in inputs:
        by_host.setdefault(item.host, []).append(item)
    selected: list[ResolutionInput] = []
    for host in sorted(by_host):
        rows = by_host[host]
        if len(rows) >= 1_000:
            limit = 100
        elif len(rows) >= 25:
            limit = 25
        else:
            limit = len(rows)
        strata: dict[tuple[str, ...], list[ResolutionInput]] = {}
        for item in rows:
            strata.setdefault(_pilot_stratum(item), []).append(item)
        for values in strata.values():
            values.sort(key=lambda item: _selection_hash(item.source_row_id))
        ordered_strata = sorted(strata)
        chosen: list[ResolutionInput] = []
        while len(chosen) < limit:
            progressed = False
            for stratum in ordered_strata:
                values = strata[stratum]
                if values and len(chosen) < limit:
                    chosen.append(values.pop(0))
                    progressed = True
            if not progressed:
                break
        selected.extend(chosen)
    return sorted(selected, key=lambda item: item.source_row_id)


def pilot_selection_table(
    selected: list[ResolutionInput], *, population: list[ResolutionInput]
) -> pa.Table:
    host_counts = Counter(item.host for item in population)
    rows = []
    for item in selected:
        host_count = host_counts[item.host]
        stratum = _pilot_stratum(item)
        rows.append(
            {
                "source_row_id": item.source_row_id,
                "gbifID": item.gbif_id,
                "media_references": item.media_references,
                "media_host": item.host or "<MISSING>",
                "host_population_rows": host_count,
                "host_size_band": "large" if host_count >= 1_000 else "medium" if host_count >= 25 else "small",
                "provider": item.provider or "<MISSING>",
                "publisher": item.publisher or "<MISSING>",
                "dataset_name": item.dataset_name or "<MISSING>",
                "url_pattern": _url_pattern(item),
                "license_state": _license_state(item),
                "reference_type": _reference_type(item.media_references),
                "taxon_rank": item.taxon_rank or "<MISSING>",
                "country_code": item.country_code or "<MISSING>",
                "expected_adapter": _expected_adapter(item),
                "rights_blocked": item.rights_blocked,
                "selection_stratum": "|".join(stratum),
                "selection_hash": "sha256:" + _selection_hash(item.source_row_id),
            }
        )
    return pa.Table.from_pylist(rows, schema=PILOT_SELECTION_SCHEMA)


def _pilot_stratum(item: ResolutionInput) -> tuple[str, ...]:
    return (
        item.provider or "<MISSING>",
        item.publisher or "<MISSING>",
        _url_pattern(item),
        _license_state(item),
        _reference_type(item.media_references),
        item.taxon_rank or "<MISSING>",
        item.country_code or "<MISSING>",
        _expected_adapter(item),
    )


def _expected_adapter(item: ResolutionInput) -> str:
    for adapter in DEFAULT_PROVIDER_ADAPTERS:
        if adapter.supports(item):
            return adapter.adapter_id
    return "generic_structured_or_gbif"


def _url_pattern(item: ResolutionInput) -> str:
    from urllib.parse import urlsplit
    parsed = urlsplit(item.media_references)
    path = parsed.path.casefold()
    if any(adapter.supports(item) for adapter in DEFAULT_PROVIDER_ADAPTERS):
        return _expected_adapter(item)
    if "/occurrence/" in path:
        return "occurrence_record"
    suffix = Path(path).suffix
    return f"path_suffix:{suffix}" if suffix else "extensionless_reference"


def _reference_type(url: str) -> str:
    from urllib.parse import urlsplit
    suffix = Path(urlsplit(url).path.casefold()).suffix
    if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif", ".tiff"}:
        return "image_like_reference"
    if suffix in {".json", ".xml"}:
        return "structured_endpoint"
    return "html_or_unknown_reference"


def _license_state(item: ResolutionInput) -> str:
    if item.rights_blocked:
        return "explicitly_restricted"
    return item.license_basis


def _selection_hash(value: str) -> str:
    return hashlib.sha256(f"gbif-media-url-pilot/v2|{value}".encode()).hexdigest()


def _read_selected_tables(
    selections: Iterable[tuple[Path, set[str]]],
    schema: pa.Schema,
) -> pa.Table:
    tables: list[pa.Table] = []
    for path, source_ids in selections:
        table = pq.read_table(path, schema=schema)
        table = table.filter(
            pc.is_in(
                table.column("source_row_id"),
                value_set=pa.array(sorted(source_ids), type=pa.string()),
            )
        )
        tables.append(table)
    return pa.concat_tables(tables) if tables else pa.Table.from_pylist([], schema=schema)


def _write_parquet_create_only(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        if path.exists():
            if _sha256(path) != _sha256(temporary):
                raise FileExistsError(f"existing shard differs: {path}")
            return
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _parquet_inventory(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    row_groups = [parquet.metadata.row_group(index).num_rows for index in range(parquet.metadata.num_row_groups)]
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
        "row_groups_complete": sum(row_groups) == parquet.metadata.num_rows and all(value > 0 for value in row_groups),
    }


def _benchmark_effective_url(con: duckdb.DuckDBPyConnection, value: str) -> dict[str, Any]:
    query = "SELECT gbifID FROM gbif_media WHERE effective_media_identifier = ?"
    con.execute(query, [value]).fetchall()
    timings: list[float] = []
    result_rows = 0
    for _ in range(5):
        started = time.perf_counter()
        result_rows = len(con.execute(query, [value]).fetchall())
        timings.append((time.perf_counter() - started) * 1_000)
    return {
        "runs": len(timings),
        "result_rows": result_rows,
        "minimum_ms": min(timings),
        "median_ms": sorted(timings)[len(timings) // 2],
        "maximum_ms": max(timings),
        "environment_note": "warm-cache local point lookup; not a cross-machine guarantee",
    }


def _require_columns(schema: pa.Schema, columns: Iterable[str]) -> None:
    for column in columns:
        if schema.get_field_index(column) < 0:
            raise ValueError(f"source has no {column} column")


def _validate_pilot_acceptance(
    manifest: str | Path | None,
    *,
    source_sha256: str,
    required: bool,
) -> dict[str, object] | None:
    if manifest is None:
        if required:
            raise ValueError(
                "full resolution mode requires a PASS pilot acceptance manifest"
            )
        return None
    path = Path(manifest).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version")
        != "biominer-gbif-media-url-pilot-execution-audit/v1"
    ):
        raise ValueError("pilot acceptance manifest schema is unsupported")
    if value.get("overall_acceptance_status") != "PASS":
        raise ValueError("pilot acceptance manifest is not PASS")
    if value.get("source_snapshot_id") != source_sha256:
        raise ValueError("pilot acceptance manifest belongs to another source snapshot")
    required_validation = (
        "all_resolved_rows_reviewed",
        "rights_blocked_zero_attempts",
        "unresolved_reasons_complete",
        "manifest_written_last",
    )
    validation = value.get("validation") or {}
    if not all(validation.get(name) is True for name in required_validation):
        raise ValueError("pilot acceptance manifest validation is incomplete")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("pilot acceptance manifest has no artifacts")
    for artifact in artifacts:
        artifact_path = path.parent / str(artifact.get("path", ""))
        expected = str(artifact.get("sha256", ""))
        if not artifact_path.is_file() or _sha256(artifact_path) != expected:
            raise ValueError(
                f"pilot acceptance artifact checksum mismatch: {artifact_path}"
            )
    return {
        "path": path,
        "sha256": "sha256:" + _sha256(path),
    }


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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_idempotent(path: Path, value: dict[str, Any]) -> None:
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = dict(existing)
        comparable.pop("prepared_at", None)
        comparable.pop("git_commit", None)
        expected = dict(value)
        expected.pop("prepared_at", None)
        expected.pop("git_commit", None)
        if comparable != expected:
            raise FileExistsError(f"existing prepare receipt differs: {path}")
        return
    path.write_text(content, encoding="utf-8")
