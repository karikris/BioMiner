from __future__ import annotations

from typing import Any

from biominer.workstore.schema import POSTGRES_CLAIM_SQL, POSTGRES_SCHEMA_SQL


class PostgresWorkStore:
    def __init__(self, dsn: str) -> None:
        if not dsn:
            raise ValueError("BIOMINER_WORKSTORE_DSN is required for postgres workstore")
        self.dsn = dsn

    def get_or_create_run(
        self,
        *,
        job_name: str,
        stage: str,
        run_id: str,
        registry_version: str | None,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore run state will be implemented in the cloud migration phase")

    def get_run(self, *, run_id: str) -> dict[str, Any] | None:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore run lookup will be implemented in the cloud migration phase")

    def list_runs(
        self,
        *,
        job_name: str | None = None,
        stage: str | None = None,
        registry_version: str | None = None,
        statuses: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore run listing will be implemented in the cloud migration phase")

    def enqueue_work(
        self,
        job_name: str,
        registry_version: str | None = None,
        items: list[dict[str, Any]] | None = None,
        *,
        stage: str = "default",
    ) -> int:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore enqueue_work will be implemented in the cloud migration phase")

    def claim_next_batch(
        self,
        worker_id: str,
        limit: int | None = None,
        *,
        job_name: str | None = None,
        stage: str | None = None,
        registry_version: str | None = None,
    ) -> list[dict[str, Any]]:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore claim_next_batch will use FOR UPDATE SKIP LOCKED")

    def list_work_items(
        self,
        *,
        job_name: str,
        stage: str,
        registry_version: str | None,
        statuses: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore work item listing will be implemented in the cloud migration phase")

    def mark_completed(self, work_key: str, output_uri: str | None, checksum: str | None, row_count: int | None) -> None:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore mark_completed will be implemented in the cloud migration phase")

    def mark_failed(self, work_key: str, error: str) -> None:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore mark_failed will be implemented in the cloud migration phase")

    def completed_keys(
        self,
        job_name: str,
        registry_version: str | None = None,
        *,
        stage: str | None = None,
    ) -> set[str]:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore completed_keys will be implemented in the cloud migration phase")

    def requeue_stale_claims(
        self,
        *,
        job_name: str | None = None,
        stage: str | None = None,
        registry_version: str | None = None,
        stale_after_seconds: int,
    ) -> int:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore stale claim requeue will be implemented in the cloud migration phase")

    def stale_claims_to_pending(self, stale_after_seconds: int) -> int:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore stale claim requeue will be implemented in the cloud migration phase")

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
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore shard inventory will be implemented in the cloud migration phase")

    def list_committed_shards(
        self,
        *,
        job_name: str,
        stage: str,
        registry_version: str | None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore shard inventory listing will be implemented in the cloud migration phase")

    def list_candidate_shards(
        self,
        *,
        job_name: str,
        stage: str,
        registry_version: str | None,
        run_id: str | None = None,
        include_compacted: bool = False,
    ) -> list[dict[str, Any]]:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore compaction candidate listing will be implemented in the cloud migration phase")

    def list_compacted_source_shard_ids(
        self,
        *,
        job_name: str,
        stage: str,
        registry_version: str | None,
    ) -> set[str]:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore compaction input listing will be implemented in the cloud migration phase")

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
    ) -> None:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore compaction manifests will be implemented in the cloud migration phase")

    def mark_run_started(self, *, run_id: str) -> None:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore run state will be implemented in the cloud migration phase")

    def mark_run_completed(self, *, run_id: str, summary: dict[str, Any] | None = None) -> None:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore run state will be implemented in the cloud migration phase")

    def mark_run_failed(self, *, run_id: str, error: str) -> None:
        self._require_psycopg()
        raise NotImplementedError("PostgresWorkStore run state will be implemented in the cloud migration phase")

    @staticmethod
    def _require_psycopg() -> None:
        try:
            import psycopg  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("psycopg is required to use PostgresWorkStore; install the optional postgres dependency") from exc
