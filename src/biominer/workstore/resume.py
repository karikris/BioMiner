from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biominer.storage.uri import is_cloud_uri, normalize_local_uri
from biominer.workstore.base import WorkStore
from biominer.workstore.keys import uri_shard_id


@dataclass(frozen=True)
class ResumePlan:
    run_id: str
    job_name: str
    stage: str
    registry_version: str | None
    planned_count: int
    enqueued_count: int
    pending_count: int
    claimed_count: int
    completed_count: int
    failed_count: int
    stale_requeued_count: int
    committed_shard_count: int
    repaired_shard_count: int
    skipped_completed_count: int
    worker_id: str


def prepare_resume_plan(
    *,
    workstore: WorkStore,
    storage: Any,
    job_name: str,
    stage: str,
    run_id: str,
    registry_version: str | None,
    planned_items: list[dict[str, Any]],
    worker_id: str,
    stale_after_seconds: int,
    repair_manifest: bool = False,
    shard_prefix: str | None = None,
    claim_limit: int | None = None,
    config: dict[str, Any] | None = None,
) -> ResumePlan:
    workstore.get_or_create_run(
        job_name=job_name,
        stage=stage,
        run_id=run_id,
        registry_version=registry_version,
        config=config or {},
    )
    stale_requeued_count = workstore.requeue_stale_claims(
        job_name=job_name,
        stage=stage,
        registry_version=registry_version,
        stale_after_seconds=stale_after_seconds,
    )
    completed = workstore.completed_keys(job_name=job_name, stage=stage, registry_version=registry_version)
    missing_items = [dict(item) for item in planned_items if str(item.get("work_key", "")) not in completed]
    enqueued_count = workstore.enqueue_work(
        job_name=job_name,
        stage=stage,
        registry_version=registry_version,
        items=missing_items,
    )

    repaired_shard_count = 0
    if repair_manifest and shard_prefix:
        repaired_shard_count = repair_shard_manifest_from_storage(
            storage=storage,
            workstore=workstore,
            job_name=job_name,
            stage=stage,
            registry_version=registry_version,
            run_id=run_id,
            shard_prefix=shard_prefix,
        )

    effective_claim_limit = claim_limit if claim_limit is not None else max(len(missing_items), 0)
    claimed = workstore.claim_next_batch(
        worker_id=worker_id,
        job_name=job_name,
        stage=stage,
        registry_version=registry_version,
        limit=effective_claim_limit,
    )
    pending_count = len(
        workstore.list_work_items(
            job_name=job_name,
            stage=stage,
            registry_version=registry_version,
            statuses=["pending"],
        )
    )
    failed_count = len(
        workstore.list_work_items(
            job_name=job_name,
            stage=stage,
            registry_version=registry_version,
            statuses=["failed"],
        )
    )
    committed_shard_count = len(
        workstore.list_committed_shards(
            job_name=job_name,
            stage=stage,
            registry_version=registry_version,
            run_id=run_id,
        )
    )

    return ResumePlan(
        run_id=run_id,
        job_name=job_name,
        stage=stage,
        registry_version=registry_version,
        planned_count=len(planned_items),
        enqueued_count=enqueued_count,
        pending_count=pending_count,
        claimed_count=len(claimed),
        completed_count=len(completed),
        failed_count=failed_count,
        stale_requeued_count=stale_requeued_count,
        committed_shard_count=committed_shard_count,
        repaired_shard_count=repaired_shard_count,
        skipped_completed_count=len(planned_items) - len(missing_items),
        worker_id=worker_id,
    )


def repair_shard_manifest_from_storage(
    *,
    storage: Any,
    workstore: WorkStore,
    job_name: str,
    stage: str,
    registry_version: str | None,
    run_id: str | None,
    shard_prefix: str,
) -> int:
    shard_uris = storage.list_shards(shard_prefix)
    existing = workstore.list_committed_shards(
        job_name=job_name,
        stage=stage,
        registry_version=registry_version,
        run_id=run_id,
    )
    existing_uris = {str(shard["uri"]) for shard in existing}
    repaired = 0
    for uri in shard_uris:
        if uri in existing_uris:
            continue
        row_count = _cheap_row_count(storage, uri)
        byte_count = _cheap_byte_count(uri)
        workstore.register_shard(
            shard_id=uri_shard_id(uri),
            job_name=job_name,
            stage=stage,
            run_id=run_id or "",
            registry_version=registry_version,
            uri=uri,
            row_count=row_count,
            checksum=None,
            byte_count=byte_count,
            metadata={"repaired": True},
        )
        repaired += 1
    return repaired


def _cheap_row_count(storage: Any, uri: str) -> int | None:
    if is_cloud_uri(uri):
        return None
    try:
        return int(storage.read_parquet(uri).height)
    except Exception:
        return None


def _cheap_byte_count(uri: str) -> int | None:
    if is_cloud_uri(uri):
        return None
    try:
        path = normalize_local_uri(uri)
    except ValueError:
        return None
    return Path(path).stat().st_size if Path(path).exists() else None
