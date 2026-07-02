from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class WorkStore(Protocol):
    def enqueue_work(self, job_name: str, registry_version: str | None, items: list[dict[str, Any]]) -> int:
        ...

    def claim_next_batch(self, worker_id: str, limit: int) -> list[dict[str, Any]]:
        ...

    def mark_completed(self, work_key: str, output_uri: str | None, checksum: str | None, row_count: int | None) -> None:
        ...

    def mark_failed(self, work_key: str, error: str) -> None:
        ...

    def completed_keys(self, job_name: str, registry_version: str | None) -> set[str]:
        ...

    def stale_claims_to_pending(self, stale_after_seconds: int) -> int:
        ...
