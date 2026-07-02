from __future__ import annotations

from typing import Any

from biominer.workstore.schema import POSTGRES_CLAIM_SQL, POSTGRES_SCHEMA_SQL


class PostgresWorkStore:
    def __init__(self, dsn: str) -> None:
        if not dsn:
            raise ValueError("BIOMINER_WORKSTORE_DSN is required for postgres workstore")
        self.dsn = dsn

    def enqueue_work(self, job_name: str, registry_version: str | None, items: list[dict[str, Any]]) -> int:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore enqueue_work will be implemented in the cloud migration phase")

    def claim_next_batch(self, worker_id: str, limit: int) -> list[dict[str, Any]]:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore claim_next_batch will use FOR UPDATE SKIP LOCKED")

    def mark_completed(self, work_key: str, output_uri: str | None, checksum: str | None, row_count: int | None) -> None:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore mark_completed will be implemented in the cloud migration phase")

    def mark_failed(self, work_key: str, error: str) -> None:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore mark_failed will be implemented in the cloud migration phase")

    def completed_keys(self, job_name: str, registry_version: str | None) -> set[str]:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore completed_keys will be implemented in the cloud migration phase")

    def stale_claims_to_pending(self, stale_after_seconds: int) -> int:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore stale claim requeue will be implemented in the cloud migration phase")

    @staticmethod
    def _require_psycopg() -> None:
        try:
            import psycopg  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("psycopg is required to use PostgresWorkStore; install the optional postgres dependency") from exc
