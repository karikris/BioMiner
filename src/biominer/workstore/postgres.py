from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
import hashlib
import json

from biominer.common.status import (
    CLAIMED,
    COMPLETED,
    FAILED,
    PENDING,
    RUN_PLANNED,
)
from biominer.workstore.base import validate_claim_lease
from biominer.workstore.keys import publication_lock_digest, scoped_work_item_key
from biominer.workstore.schema import POSTGRES_CLAIM_SQL, POSTGRES_SCHEMA_SQL

POSTGRES_PUBLICATION_LOCK_SQL = "SELECT pg_advisory_xact_lock(%s)"


class PostgresWorkStore:
    def __init__(self, dsn: str, *, connect: Callable[[], Any] | None = None) -> None:
        if not dsn:
            raise ValueError("BIOMINER_WORKSTORE_DSN is required for postgres workstore")
        self.dsn = dsn
        self._connect_override = connect

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(POSTGRES_SCHEMA_SQL)

    @contextmanager
    def publication_lock(self, key: str) -> Iterator[None]:
        lock_id = int.from_bytes(
            publication_lock_digest(key)[:8],
            byteorder="big",
            signed=True,
        )
        with self._connect() as conn:
            with conn.transaction():
                conn.execute(POSTGRES_PUBLICATION_LOCK_SQL, (lock_id,))
                yield

    def get_or_create_run(
        self,
        *,
        job_name: str,
        stage: str,
        run_id: str,
        registry_version: str | None,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO biominer_runs (
                    run_id, job_name, stage, registry_version, status, started_at,
                    config_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (run_id) DO NOTHING
                """,
                (
                    run_id,
                    job_name,
                    stage,
                    registry_version,
                    RUN_PLANNED,
                    _timestamp(),
                    _json_dumps(config),
                ),
            )
            row = conn.execute("SELECT * FROM biominer_runs WHERE run_id = %s", (run_id,)).fetchone()
        if row is None:
            raise RuntimeError(f"failed to create run row: {run_id}")
        return _row_to_run(row)

    def get_run(self, *, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM biominer_runs WHERE run_id = %s", (run_id,)).fetchone()
        return _row_to_run(row) if row is not None else None

    def list_runs(
        self,
        *,
        job_name: str | None = None,
        stage: str | None = None,
        registry_version: str | None = None,
        statuses: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        _add_filter_if_not_none(clauses, params, "job_name", job_name)
        _add_filter_if_not_none(clauses, params, "stage", stage)
        _add_filter_if_not_none(clauses, params, "registry_version", registry_version)
        if statuses:
            placeholders = ", ".join("%s" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(statuses)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM biominer_runs
                {where}
                ORDER BY started_at, run_id
                """,
                tuple(params),
            ).fetchall()
        return [_row_to_run(row) for row in rows]

    def enqueue_work(
        self,
        job_name: str,
        registry_version: str | None = None,
        items: list[dict[str, Any]] | None = None,
        *,
        stage: str = "default",
    ) -> int:
        if items is None:
            raise TypeError("items is required")
        inserted = 0
        with self._connect() as conn:
            for item in items:
                payload = dict(item)
                work_key = str(payload.pop("work_key", "") or scoped_work_item_key(job_name, stage, registry_version, payload))
                result = conn.execute(
                    """
                    INSERT INTO biominer_work_items (
                        work_key, job_name, stage, registry_version, status,
                        payload_json, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                    ON CONFLICT (work_key) DO NOTHING
                    """,
                    (
                        work_key,
                        job_name,
                        stage,
                        registry_version,
                        PENDING,
                        _json_dumps(payload),
                        _timestamp(),
                    ),
                )
                inserted += max(int(getattr(result, "rowcount", 0)), 0)
        return inserted

    def claim_next_batch(
        self,
        worker_id: str,
        limit: int | None = None,
        *,
        job_name: str | None = None,
        stage: str | None = None,
        registry_version: str | None = None,
    ) -> list[dict[str, Any]]:
        if limit is None:
            raise TypeError("limit is required")
        if limit <= 0:
            return []
        with self._connect() as conn:
            if job_name is not None and stage is not None:
                rows = conn.execute(
                    POSTGRES_CLAIM_SQL,
                    (job_name, stage, registry_version, limit, worker_id),
                ).fetchall()
            else:
                clauses = ["status = %s"]
                params: list[Any] = [PENDING]
                _add_filter_if_not_none(clauses, params, "job_name", job_name)
                _add_filter_if_not_none(clauses, params, "stage", stage)
                if job_name is not None or stage is not None or registry_version is not None:
                    _add_nullable_filter(clauses, params, "registry_version", registry_version)
                rows = conn.execute(
                    f"""
                    WITH picked AS (
                      SELECT work_key
                      FROM biominer_work_items
                      WHERE {" AND ".join(clauses)}
                      ORDER BY created_at, work_key
                      FOR UPDATE SKIP LOCKED
                      LIMIT %s
                    )
                    UPDATE biominer_work_items w
                    SET status = %s,
                        claimed_by = %s,
                        claimed_at = now(),
                        attempt_count = attempt_count + 1,
                        error = NULL
                    FROM picked
                    WHERE w.work_key = picked.work_key
                    RETURNING w.*
                    """,
                    (*params, limit, CLAIMED, worker_id),
                ).fetchall()
        return [_row_to_work_item(row) for row in rows]

    def list_work_items(
        self,
        *,
        job_name: str,
        stage: str,
        registry_version: str | None,
        statuses: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["job_name = %s", "stage = %s"]
        params: list[Any] = [job_name, stage]
        _add_nullable_filter(clauses, params, "registry_version", registry_version)
        if statuses:
            placeholders = ", ".join("%s" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(statuses)
        limit_clause = "LIMIT %s" if limit is not None else ""
        if limit is not None:
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM biominer_work_items
                WHERE {" AND ".join(clauses)}
                ORDER BY created_at, work_key
                {limit_clause}
                """,
                tuple(params),
            ).fetchall()
        return [_row_to_work_item(row) for row in rows]

    def mark_completed(
        self,
        work_key: str,
        output_uri: str | None,
        checksum: str | None,
        row_count: int | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE biominer_work_items
                SET status = %s,
                    completed_at = %s,
                    output_uri = %s,
                    checksum = %s,
                    row_count = %s,
                    error = %s
                WHERE work_key = %s
                """,
                (
                    COMPLETED,
                    _timestamp(),
                    output_uri,
                    checksum,
                    row_count,
                    None,
                    work_key,
                ),
            )

    def complete_pending(
        self,
        work_key: str,
        *,
        output_uri: str | None,
        checksum: str | None,
        row_count: int | None,
    ) -> bool:
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE biominer_work_items
                SET status = %s,
                    completed_at = now(),
                    output_uri = %s,
                    checksum = %s,
                    row_count = %s,
                    error = NULL,
                    claimed_by = NULL,
                    claimed_at = NULL
                WHERE work_key = %s
                  AND status = %s
                """,
                (
                    COMPLETED,
                    output_uri,
                    checksum,
                    row_count,
                    work_key,
                    PENDING,
                ),
            )
        return int(getattr(result, "rowcount", 0)) == 1

    def mark_failed(self, work_key: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE biominer_work_items
                SET status = %s,
                    completed_at = %s,
                    error = %s
                WHERE work_key = %s
                """,
                (FAILED, _timestamp(), error, work_key),
            )

    def renew_claim(
        self,
        work_key: str,
        *,
        worker_id: str,
        attempt_count: int,
        stale_after_seconds: int,
    ) -> bool:
        validate_claim_lease(
            work_key=work_key,
            worker_id=worker_id,
            attempt_count=attempt_count,
            stale_after_seconds=stale_after_seconds,
        )
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE biominer_work_items
                SET claimed_at = now()
                WHERE work_key = %s
                  AND status = %s
                  AND claimed_by = %s
                  AND attempt_count = %s
                  AND claimed_at IS NOT NULL
                  AND claimed_at >= now() - (%s * INTERVAL '1 second')
                """,
                (work_key, CLAIMED, worker_id, attempt_count, stale_after_seconds),
            )
        return int(getattr(result, "rowcount", 0)) == 1

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
    ) -> bool:
        validate_claim_lease(
            work_key=work_key,
            worker_id=worker_id,
            attempt_count=attempt_count,
            stale_after_seconds=stale_after_seconds,
        )
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE biominer_work_items
                SET status = %s,
                    completed_at = now(),
                    output_uri = %s,
                    checksum = %s,
                    row_count = %s,
                    error = NULL,
                    claimed_by = NULL,
                    claimed_at = NULL
                WHERE work_key = %s
                  AND status = %s
                  AND claimed_by = %s
                  AND attempt_count = %s
                  AND claimed_at IS NOT NULL
                  AND claimed_at >= now() - (%s * INTERVAL '1 second')
                """,
                (
                    COMPLETED,
                    output_uri,
                    checksum,
                    row_count,
                    work_key,
                    CLAIMED,
                    worker_id,
                    attempt_count,
                    stale_after_seconds,
                ),
            )
        return int(getattr(result, "rowcount", 0)) == 1

    def fail_claim(
        self,
        work_key: str,
        *,
        worker_id: str,
        attempt_count: int,
        stale_after_seconds: int,
        error: str,
    ) -> bool:
        validate_claim_lease(
            work_key=work_key,
            worker_id=worker_id,
            attempt_count=attempt_count,
            stale_after_seconds=stale_after_seconds,
        )
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE biominer_work_items
                SET status = %s,
                    completed_at = now(),
                    error = %s,
                    claimed_by = NULL,
                    claimed_at = NULL
                WHERE work_key = %s
                  AND status = %s
                  AND claimed_by = %s
                  AND attempt_count = %s
                  AND claimed_at IS NOT NULL
                  AND claimed_at >= now() - (%s * INTERVAL '1 second')
                """,
                (
                    FAILED,
                    error,
                    work_key,
                    CLAIMED,
                    worker_id,
                    attempt_count,
                    stale_after_seconds,
                ),
            )
        return int(getattr(result, "rowcount", 0)) == 1

    def completed_keys(
        self,
        job_name: str,
        registry_version: str | None = None,
        *,
        stage: str | None = None,
    ) -> set[str]:
        clauses = ["job_name = %s", "status = %s"]
        params: list[Any] = [job_name, COMPLETED]
        if stage is not None:
            clauses.append("stage = %s")
            params.append(stage)
        _add_nullable_filter(clauses, params, "registry_version", registry_version)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT work_key
                FROM biominer_work_items
                WHERE {" AND ".join(clauses)}
                """,
                tuple(params),
            ).fetchall()
        return {str(_row_get(row, "work_key")) for row in rows}

    def requeue_stale_claims(
        self,
        *,
        job_name: str | None = None,
        stage: str | None = None,
        registry_version: str | None = None,
        stale_after_seconds: int,
    ) -> int:
        cutoff = datetime.now(UTC) - timedelta(seconds=stale_after_seconds)
        clauses = ["status = %s", "claimed_at IS NOT NULL", "claimed_at < %s"]
        params: list[Any] = [CLAIMED, cutoff]
        scoped = job_name is not None or stage is not None or registry_version is not None
        _add_filter_if_not_none(clauses, params, "job_name", job_name)
        _add_filter_if_not_none(clauses, params, "stage", stage)
        if scoped:
            _add_nullable_filter(clauses, params, "registry_version", registry_version)
        with self._connect() as conn:
            result = conn.execute(
                f"""
                UPDATE biominer_work_items
                SET status = %s,
                    claimed_by = NULL,
                    claimed_at = NULL
                WHERE {" AND ".join(clauses)}
                """,
                (PENDING, *params),
            )
        return max(int(getattr(result, "rowcount", 0)), 0)

    def stale_claims_to_pending(self, stale_after_seconds: int) -> int:
        return self.requeue_stale_claims(stale_after_seconds=stale_after_seconds)

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
        resolved_shard_id = shard_id or hashlib.sha256(uri.encode("utf-8")).hexdigest()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO biominer_parquet_shards (
                    shard_id, job_name, registry_version, stage, run_id, worker_id,
                    uri, row_count, byte_count, checksum, metadata_json, committed_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT DO NOTHING
                """,
                (
                    resolved_shard_id,
                    job_name,
                    registry_version,
                    stage,
                    run_id,
                    worker_id,
                    uri,
                    row_count,
                    byte_count,
                    checksum,
                    _json_dumps(metadata or {}),
                    _timestamp(),
                ),
            )

    def list_committed_shards(
        self,
        *,
        job_name: str,
        stage: str,
        registry_version: str | None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["job_name = %s", "stage = %s"]
        params: list[Any] = [job_name, stage]
        _add_nullable_filter(clauses, params, "registry_version", registry_version)
        _add_filter_if_not_none(clauses, params, "run_id", run_id)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM biominer_parquet_shards
                WHERE {" AND ".join(clauses)}
                ORDER BY committed_at, uri
                """,
                tuple(params),
            ).fetchall()
        return [_row_to_shard(row) for row in rows]

    def list_candidate_shards(
        self,
        *,
        job_name: str,
        stage: str,
        registry_version: str | None,
        run_id: str | None = None,
        include_compacted: bool = False,
    ) -> list[dict[str, Any]]:
        candidates = self.list_committed_shards(
            job_name=job_name,
            stage=stage,
            registry_version=registry_version,
            run_id=run_id,
        )
        if include_compacted:
            return candidates
        consumed = self.list_compacted_source_shard_ids(job_name=job_name, stage=stage, registry_version=registry_version)
        return [shard for shard in candidates if str(shard["shard_id"]) not in consumed]

    def list_compacted_source_shard_ids(
        self,
        *,
        job_name: str,
        stage: str,
        registry_version: str | None,
    ) -> set[str]:
        clauses = ["job_name = %s", "source_stage = %s"]
        params: list[Any] = [job_name, stage]
        _add_nullable_filter(clauses, params, "registry_version", registry_version)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT source_shard_id
                FROM biominer_compaction_inputs
                WHERE {" AND ".join(clauses)}
                """,
                tuple(params),
            ).fetchall()
        return {str(_row_get(row, "source_shard_id")) for row in rows}

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
        merged_metadata = {
            **(metadata or {}),
            "compaction_run_id": compaction_run_id,
            "source_stage": source_stage,
            "source_shard_count": len(source_shards),
            "source_shard_ids": [str(shard["shard_id"]) for shard in source_shards],
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO biominer_parquet_shards (
                    shard_id, job_name, registry_version, stage, run_id, worker_id,
                    uri, row_count, byte_count, checksum, metadata_json, committed_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT DO NOTHING
                """,
                (
                    output_shard_id,
                    job_name,
                    registry_version,
                    output_stage,
                    compaction_run_id,
                    "compaction",
                    output_uri,
                    row_count,
                    byte_count,
                    checksum,
                    _json_dumps(merged_metadata),
                    _timestamp(),
                ),
            )
            for shard in source_shards:
                conn.execute(
                    """
                    INSERT INTO biominer_compaction_inputs (
                        compaction_run_id, output_shard_id, source_shard_id, source_uri,
                        job_name, source_stage, output_stage, registry_version, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (output_shard_id, source_shard_id) DO NOTHING
                    """,
                    (
                        compaction_run_id,
                        output_shard_id,
                        str(shard["shard_id"]),
                        str(shard["uri"]),
                        job_name,
                        source_stage,
                        output_stage,
                        registry_version,
                        _timestamp(),
                    ),
                )

    def _connect(self):
        if self._connect_override is not None:
            return self._connect_override()
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("psycopg is required to use PostgresWorkStore; install the optional postgres dependency") from exc
        return psycopg.connect(self.dsn, row_factory=dict_row)


def _row_to_run(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": str(_row_get(row, "run_id")),
        "job_name": str(_row_get(row, "job_name")),
        "stage": str(_row_get(row, "stage")),
        "registry_version": _row_get(row, "registry_version"),
        "status": str(_row_get(row, "status")),
        "started_at": str(_row_get(row, "started_at")),
        "ended_at": _row_get(row, "ended_at"),
        "config": _json_loads_dict(_row_get(row, "config_json")),
        "summary": _json_loads_dict(_row_get(row, "summary_json")) if _row_get(row, "summary_json") else None,
    }


def _row_to_work_item(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "work_key": str(_row_get(row, "work_key")),
        "job_name": _row_get(row, "job_name"),
        "stage": _row_get(row, "stage"),
        "registry_version": _row_get(row, "registry_version"),
        "status": _row_get(row, "status"),
        "payload": _json_loads_dict(_row_get(row, "payload_json")),
        "claimed_by": _row_get(row, "claimed_by"),
        "claimed_at": _row_get(row, "claimed_at"),
        "completed_at": _row_get(row, "completed_at"),
        "output_uri": _row_get(row, "output_uri"),
        "checksum": _row_get(row, "checksum"),
        "row_count": _row_get(row, "row_count"),
        "attempt_count": int(_row_get(row, "attempt_count")),
        "error": _row_get(row, "error"),
        "created_at": _row_get(row, "created_at"),
    }


def _row_to_shard(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "shard_id": str(_row_get(row, "shard_id")),
        "job_name": str(_row_get(row, "job_name")),
        "stage": str(_row_get(row, "stage")),
        "run_id": str(_row_get(row, "run_id")),
        "registry_version": _row_get(row, "registry_version"),
        "worker_id": _row_get(row, "worker_id"),
        "uri": str(_row_get(row, "uri")),
        "row_count": _row_get(row, "row_count"),
        "byte_count": _row_get(row, "byte_count"),
        "checksum": _row_get(row, "checksum"),
        "metadata": _json_loads_dict(_row_get(row, "metadata_json")) if _row_get(row, "metadata_json") else {},
        "committed_at": str(_row_get(row, "committed_at")),
    }


def _row_get(row: Mapping[str, Any], key: str) -> Any:
    return row[key]


def _add_filter_if_not_none(clauses: list[str], params: list[Any], column: str, value: str | None) -> None:
    if value is None:
        return
    clauses.append(f"{column} = %s")
    params.append(value)


def _add_nullable_filter(clauses: list[str], params: list[Any], column: str, value: str | None) -> None:
    if value is None:
        clauses.append(f"{column} IS NULL")
        return
    clauses.append(f"{column} = %s")
    params.append(value)


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _json_loads_dict(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    payload = json.loads(str(raw or "{}"))
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object in workstore row")
    return payload


def _timestamp() -> datetime:
    return datetime.now(UTC)
