from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class WorkStore(Protocol):
    def publication_lock(self, key: str) -> AbstractContextManager[None]: ...

    def get_or_create_run(
        self,
        *,
        job_name: str,
        stage: str,
        run_id: str,
        registry_version: str | None,
        config: dict[str, Any],
    ) -> dict[str, Any]: ...

    def get_run(self, *, run_id: str) -> dict[str, Any] | None: ...

    def list_runs(
        self,
        *,
        job_name: str | None = None,
        stage: str | None = None,
        registry_version: str | None = None,
        statuses: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...

    def enqueue_work(
        self,
        job_name: str,
        registry_version: str | None = None,
        items: list[dict[str, Any]] | None = None,
        *,
        stage: str = "default",
    ) -> int: ...

    def claim_next_batch(
        self,
        worker_id: str,
        limit: int | None = None,
        *,
        job_name: str | None = None,
        stage: str | None = None,
        registry_version: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def list_work_items(
        self,
        *,
        job_name: str,
        stage: str,
        registry_version: str | None,
        statuses: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...

    def set_pending_schedule_ranks(
        self,
        schedule: list[tuple[str, int]],
        *,
        job_name: str,
        stage: str,
        registry_version: str | None,
    ) -> int: ...

    def mark_completed(
        self,
        work_key: str,
        output_uri: str | None,
        checksum: str | None,
        row_count: int | None,
    ) -> None: ...

    def complete_pending(
        self,
        work_key: str,
        *,
        output_uri: str | None,
        checksum: str | None,
        row_count: int | None,
    ) -> bool: ...

    def complete_pending_batch(
        self,
        work_keys: list[str],
        *,
        output_uri: str | None,
        checksum: str | None,
        row_count: int | None,
    ) -> set[str]: ...

    def mark_failed(self, work_key: str, error: str) -> None: ...

    def renew_claim(
        self,
        work_key: str,
        *,
        worker_id: str,
        attempt_count: int,
        stale_after_seconds: int,
    ) -> bool: ...

    def complete_claim(
        self,
        work_key: str,
        *,
        worker_id: str,
        attempt_count: int,
        stale_after_seconds: int,
        output_uri: str | None,
        checksum: str | None,
        row_count: int | None,
    ) -> bool: ...

    def fail_claim(
        self,
        work_key: str,
        *,
        worker_id: str,
        attempt_count: int,
        stale_after_seconds: int,
        error: str,
    ) -> bool: ...

    def completed_keys(
        self,
        job_name: str,
        registry_version: str | None = None,
        *,
        stage: str | None = None,
    ) -> set[str]: ...

    def requeue_stale_claims(
        self,
        *,
        job_name: str | None = None,
        stage: str | None = None,
        registry_version: str | None = None,
        stale_after_seconds: int,
    ) -> int: ...

    def stale_claims_to_pending(self, stale_after_seconds: int) -> int: ...

    def register_shard(
        self,
        *,
        shard_id: str | None = None,
        job_name: str,
        registry_version: str | None,
        stage: str,
        run_id: str,
        worker_id: str = "",
        uri: str,
        checksum: str | None,
        row_count: int | None,
        byte_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def list_committed_shards(
        self,
        *,
        job_name: str,
        stage: str,
        registry_version: str | None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def list_candidate_shards(
        self,
        *,
        job_name: str,
        stage: str,
        registry_version: str | None,
        run_id: str | None = None,
        include_compacted: bool = False,
    ) -> list[dict[str, Any]]: ...

    def list_compacted_source_shard_ids(
        self,
        *,
        job_name: str,
        stage: str,
        registry_version: str | None,
    ) -> set[str]: ...

    def register_compaction_output(
        self,
        *,
        compaction_run_id: str,
        output_shard_id: str,
        output_uri: str,
        source_shards: list[dict[str, Any]],
        job_name: str,
        source_stage: str,
        output_stage: str,
        registry_version: str | None,
        row_count: int | None,
        byte_count: int | None,
        checksum: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...


def validate_claim_lease(
    *,
    work_key: str,
    worker_id: str,
    attempt_count: int,
    stale_after_seconds: int,
) -> None:
    if not work_key:
        raise ValueError("work_key must not be empty")
    if not worker_id:
        raise ValueError("worker_id must not be empty")
    if isinstance(attempt_count, bool) or not isinstance(attempt_count, int) or attempt_count <= 0:
        raise ValueError("attempt_count must be a positive integer")
    if isinstance(stale_after_seconds, bool) or not isinstance(stale_after_seconds, int) or stale_after_seconds <= 0:
        raise ValueError("stale_after_seconds must be a positive integer")


def validate_schedule_ranks(schedule: list[tuple[str, int]]) -> None:
    work_keys: set[str] = set()
    ranks: set[int] = set()
    for work_key, rank in schedule:
        if not work_key:
            raise ValueError("scheduled work_key must not be empty")
        if work_key in work_keys:
            raise ValueError(f"duplicate scheduled work_key: {work_key}")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("schedule ranks must be non-negative integers")
        if rank in ranks:
            raise ValueError(f"duplicate schedule rank: {rank}")
        work_keys.add(work_key)
        ranks.add(rank)
