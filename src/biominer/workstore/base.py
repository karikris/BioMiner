from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class WorkStore(Protocol):
    def get_or_create_run(
        self,
        *,
        job_name: str,
        stage: str,
        run_id: str,
        registry_version: str | None,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        ...

    def get_run(self, *, run_id: str) -> dict[str, Any] | None:
        ...

    def list_runs(
        self,
        *,
        job_name: str | None = None,
        stage: str | None = None,
        registry_version: str | None = None,
        statuses: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def enqueue_work(
        self,
        job_name: str,
        registry_version: str | None = None,
        items: list[dict[str, Any]] | None = None,
        *,
        stage: str = "default",
    ) -> int:
        ...

    def claim_next_batch(
        self,
        worker_id: str,
        limit: int | None = None,
        *,
        job_name: str | None = None,
        stage: str | None = None,
        registry_version: str | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def list_work_items(
        self,
        *,
        job_name: str,
        stage: str,
        registry_version: str | None,
        statuses: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def mark_completed(
        self,
        work_key: str,
        output_uri: str | None,
        checksum: str | None,
        row_count: int | None,
    ) -> None:
        ...

    def mark_failed(self, work_key: str, error: str) -> None:
        ...

    def completed_keys(
        self,
        job_name: str,
        registry_version: str | None = None,
        *,
        stage: str | None = None,
    ) -> set[str]:
        ...

    def requeue_stale_claims(
        self,
        *,
        job_name: str | None = None,
        stage: str | None = None,
        registry_version: str | None = None,
        stale_after_seconds: int,
    ) -> int:
        ...

    def stale_claims_to_pending(self, stale_after_seconds: int) -> int:
        ...

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
    ) -> None:
        ...

    def list_committed_shards(
        self,
        *,
        job_name: str,
        stage: str,
        registry_version: str | None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def mark_run_started(self, *, run_id: str) -> None:
        ...

    def mark_run_completed(self, *, run_id: str, summary: dict[str, Any] | None = None) -> None:
        ...

    def mark_run_failed(self, *, run_id: str, error: str) -> None:
        ...
