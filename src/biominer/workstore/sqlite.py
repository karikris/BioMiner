from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
import fcntl
import hashlib
import json
import os
import sqlite3
import stat

from biominer.common.status import (
    CLAIMED,
    COMPLETED,
    FAILED,
    PENDING,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_PLANNED,
    RUN_RUNNING,
)
from biominer.workstore.base import validate_claim_lease
from biominer.workstore.keys import publication_lock_digest, scoped_work_item_key

DEFAULT_STAGE = "default"


class SQLiteWorkStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def publication_lock(self, key: str) -> Iterator[None]:
        key_digest = publication_lock_digest(key).hex()
        database_path = self.path.resolve()
        lock_path = database_path.with_name(
            f".{database_path.name}.publication.{key_digest}.lock"
        )
        flags = (
            os.O_CREAT
            | os.O_RDWR
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        lock_fd = os.open(lock_path, flags, 0o600)
        acquired = False
        try:
            if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
                raise RuntimeError(
                    f"publication lock path is not a regular file: {lock_path}"
                )
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            acquired = True
            yield
        finally:
            try:
                if acquired:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)

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
                INSERT OR IGNORE INTO biominer_runs (
                    run_id, job_name, stage, registry_version, status, started_at,
                    config_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
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
            row = conn.execute(
                "SELECT * FROM biominer_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise RuntimeError(f"failed to create run row: {run_id}")
        return _row_to_run(row)

    def get_run(self, *, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM biominer_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
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
            placeholders = ", ".join("?" for _ in statuses)
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
                params,
            ).fetchall()
        return [_row_to_run(row) for row in rows]

    def enqueue_work(
        self,
        job_name: str,
        registry_version: str | None = None,
        items: list[dict[str, Any]] | None = None,
        *,
        stage: str = DEFAULT_STAGE,
    ) -> int:
        if items is None:
            raise TypeError("items is required")
        inserted = 0
        with self._connect() as conn:
            for item in items:
                payload = dict(item)
                work_key = str(
                    payload.pop("work_key", "")
                    or scoped_work_item_key(job_name, stage, registry_version, payload)
                )
                result = conn.execute(
                    """
                    INSERT OR IGNORE INTO biominer_work_items (
                        work_key, job_name, stage, registry_version, status,
                        payload_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
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
                inserted += max(int(result.rowcount), 0)
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

        clauses = ["status = ?"]
        params: list[Any] = [PENDING]
        scoped = (
            job_name is not None or stage is not None or registry_version is not None
        )
        _add_filter_if_not_none(clauses, params, "job_name", job_name)
        _add_filter_if_not_none(clauses, params, "stage", stage)
        if scoped:
            _add_nullable_filter(clauses, params, "registry_version", registry_version)

        claimed_rows: list[sqlite3.Row] = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                f"""
                SELECT *
                FROM biominer_work_items
                WHERE {" AND ".join(clauses)}
                ORDER BY created_at, work_key
                LIMIT ?
                """,
                (*params, limit),
            ).fetchall()
            now = _timestamp()
            work_keys = [str(row["work_key"]) for row in rows]
            for row in rows:
                conn.execute(
                    """
                    UPDATE biominer_work_items
                    SET status = ?,
                        claimed_by = ?,
                        claimed_at = ?,
                        attempt_count = attempt_count + 1,
                        error = NULL
                    WHERE work_key = ?
                    """,
                    (CLAIMED, worker_id, now, row["work_key"]),
                )
            if work_keys:
                placeholders = ", ".join("?" for _ in work_keys)
                refreshed = conn.execute(
                    f"""
                    SELECT *
                    FROM biominer_work_items
                    WHERE work_key IN ({placeholders})
                    """,
                    work_keys,
                ).fetchall()
                rows_by_key = {str(row["work_key"]): row for row in refreshed}
                claimed_rows = [
                    rows_by_key[key] for key in work_keys if key in rows_by_key
                ]
            conn.execute("COMMIT")
        return [_row_to_work_item(row) for row in claimed_rows]

    def list_work_items(
        self,
        *,
        job_name: str,
        stage: str,
        registry_version: str | None,
        statuses: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["job_name = ?", "stage = ?"]
        params: list[Any] = [job_name, stage]
        _add_nullable_filter(clauses, params, "registry_version", registry_version)
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(statuses)
        limit_clause = "LIMIT ?" if limit is not None else ""
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
                params,
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
                SET status = ?,
                    completed_at = ?,
                    output_uri = ?,
                    checksum = ?,
                    row_count = ?,
                    error = NULL
                WHERE work_key = ?
                """,
                (COMPLETED, _timestamp(), output_uri, checksum, row_count, work_key),
            )

    def mark_failed(self, work_key: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE biominer_work_items
                SET status = ?,
                    completed_at = ?,
                    error = ?
                WHERE work_key = ?
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
        cutoff = (
            datetime.now(UTC) - timedelta(seconds=stale_after_seconds)
        ).isoformat()
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE biominer_work_items
                SET claimed_at = ?
                WHERE work_key = ?
                  AND status = ?
                  AND claimed_by = ?
                  AND attempt_count = ?
                  AND claimed_at IS NOT NULL
                  AND claimed_at >= ?
                """,
                (_timestamp(), work_key, CLAIMED, worker_id, attempt_count, cutoff),
            )
        return int(result.rowcount) == 1

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
        cutoff = (
            datetime.now(UTC) - timedelta(seconds=stale_after_seconds)
        ).isoformat()
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE biominer_work_items
                SET status = ?,
                    completed_at = ?,
                    output_uri = ?,
                    checksum = ?,
                    row_count = ?,
                    error = NULL,
                    claimed_by = NULL,
                    claimed_at = NULL
                WHERE work_key = ?
                  AND status = ?
                  AND claimed_by = ?
                  AND attempt_count = ?
                  AND claimed_at IS NOT NULL
                  AND claimed_at >= ?
                """,
                (
                    COMPLETED,
                    _timestamp(),
                    output_uri,
                    checksum,
                    row_count,
                    work_key,
                    CLAIMED,
                    worker_id,
                    attempt_count,
                    cutoff,
                ),
            )
        return int(result.rowcount) == 1

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
        cutoff = (
            datetime.now(UTC) - timedelta(seconds=stale_after_seconds)
        ).isoformat()
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE biominer_work_items
                SET status = ?,
                    completed_at = ?,
                    error = ?,
                    claimed_by = NULL,
                    claimed_at = NULL
                WHERE work_key = ?
                  AND status = ?
                  AND claimed_by = ?
                  AND attempt_count = ?
                  AND claimed_at IS NOT NULL
                  AND claimed_at >= ?
                """,
                (
                    FAILED,
                    _timestamp(),
                    error,
                    work_key,
                    CLAIMED,
                    worker_id,
                    attempt_count,
                    cutoff,
                ),
            )
        return int(result.rowcount) == 1

    def completed_keys(
        self,
        job_name: str,
        registry_version: str | None = None,
        *,
        stage: str | None = None,
    ) -> set[str]:
        clauses = ["job_name = ?", "status = ?"]
        params: list[Any] = [job_name, COMPLETED]
        if stage is not None:
            clauses.append("stage = ?")
            params.append(stage)
        _add_nullable_filter(clauses, params, "registry_version", registry_version)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT work_key
                FROM biominer_work_items
                WHERE {" AND ".join(clauses)}
                """,
                params,
            ).fetchall()
        return {str(row["work_key"]) for row in rows}

    def requeue_stale_claims(
        self,
        *,
        job_name: str | None = None,
        stage: str | None = None,
        registry_version: str | None = None,
        stale_after_seconds: int,
    ) -> int:
        cutoff = (
            datetime.now(UTC) - timedelta(seconds=stale_after_seconds)
        ).isoformat()
        clauses = ["status = ?", "claimed_at IS NOT NULL", "claimed_at < ?"]
        params: list[Any] = [CLAIMED, cutoff]
        scoped = (
            job_name is not None or stage is not None or registry_version is not None
        )
        _add_filter_if_not_none(clauses, params, "job_name", job_name)
        _add_filter_if_not_none(clauses, params, "stage", stage)
        if scoped:
            _add_nullable_filter(clauses, params, "registry_version", registry_version)
        with self._connect() as conn:
            result = conn.execute(
                f"""
                UPDATE biominer_work_items
                SET status = ?,
                    claimed_by = NULL,
                    claimed_at = NULL
                WHERE {" AND ".join(clauses)}
                """,
                (PENDING, *params),
            )
        return max(int(result.rowcount), 0)

    def stale_claims_to_pending(self, stale_after_seconds: int) -> int:
        return self.requeue_stale_claims(stale_after_seconds=stale_after_seconds)

    def register_shard(
        self,
        *,
        job_name: str,
        registry_version: str | None,
        stage: str,
        run_id: str,
        uri: str,
        checksum: str | None,
        row_count: int | None,
        shard_id: str | None = None,
        worker_id: str = "",
        byte_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        resolved_shard_id = shard_id or hashlib.sha256(uri.encode("utf-8")).hexdigest()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO biominer_parquet_shards (
                    shard_id, job_name, registry_version, stage, run_id, worker_id,
                    uri, row_count, byte_count, checksum, metadata_json, committed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        clauses = ["job_name = ?", "stage = ?"]
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
                params,
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
        consumed = self.list_compacted_source_shard_ids(
            job_name=job_name, stage=stage, registry_version=registry_version
        )
        return [shard for shard in candidates if str(shard["shard_id"]) not in consumed]

    def list_compacted_source_shard_ids(
        self,
        *,
        job_name: str,
        stage: str,
        registry_version: str | None,
    ) -> set[str]:
        clauses = ["job_name = ?", "source_stage = ?"]
        params: list[Any] = [job_name, stage]
        _add_nullable_filter(clauses, params, "registry_version", registry_version)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT source_shard_id
                FROM biominer_compaction_inputs
                WHERE {" AND ".join(clauses)}
                """,
                params,
            ).fetchall()
        return {str(row["source_shard_id"]) for row in rows}

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
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT OR IGNORE INTO biominer_parquet_shards (
                    shard_id, job_name, registry_version, stage, run_id, worker_id,
                    uri, row_count, byte_count, checksum, metadata_json, committed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    INSERT OR IGNORE INTO biominer_compaction_inputs (
                        compaction_run_id, output_shard_id, source_shard_id, source_uri,
                        job_name, source_stage, output_stage, registry_version, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            conn.execute("COMMIT")

    def mark_run_started(self, *, run_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE biominer_runs
                SET status = ?,
                    started_at = ?
                WHERE run_id = ?
                """,
                (RUN_RUNNING, _timestamp(), run_id),
            )

    def mark_run_completed(
        self, *, run_id: str, summary: dict[str, Any] | None = None
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE biominer_runs
                SET status = ?,
                    ended_at = ?,
                    summary_json = ?
                WHERE run_id = ?
                """,
                (RUN_COMPLETED, _timestamp(), _json_dumps(summary or {}), run_id),
            )

    def mark_run_failed(self, *, run_id: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE biominer_runs
                SET status = ?,
                    ended_at = ?,
                    summary_json = ?
                WHERE run_id = ?
                """,
                (RUN_FAILED, _timestamp(), _json_dumps({"error": error}), run_id),
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS biominer_runs (
                    run_id TEXT PRIMARY KEY,
                    job_name TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    registry_version TEXT,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    summary_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS biominer_work_items (
                    work_key TEXT PRIMARY KEY,
                    job_name TEXT NOT NULL,
                    stage TEXT NOT NULL DEFAULT 'default',
                    registry_version TEXT,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    claimed_by TEXT,
                    claimed_at TEXT,
                    completed_at TEXT,
                    output_uri TEXT,
                    checksum TEXT,
                    row_count INTEGER,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS biominer_parquet_shards (
                    shard_id TEXT PRIMARY KEY,
                    job_name TEXT NOT NULL,
                    registry_version TEXT,
                    stage TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    worker_id TEXT NOT NULL DEFAULT '',
                    uri TEXT NOT NULL UNIQUE,
                    row_count INTEGER,
                    byte_count INTEGER,
                    checksum TEXT,
                    metadata_json TEXT,
                    committed_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS biominer_compaction_inputs (
                    compaction_run_id TEXT NOT NULL,
                    output_shard_id TEXT NOT NULL,
                    source_shard_id TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    job_name TEXT NOT NULL,
                    source_stage TEXT NOT NULL,
                    output_stage TEXT NOT NULL,
                    registry_version TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (output_shard_id, source_shard_id)
                )
                """
            )
            _ensure_column(
                conn, "biominer_runs", "stage", "stage TEXT NOT NULL DEFAULT 'default'"
            )
            _ensure_column(conn, "biominer_runs", "summary_json", "summary_json TEXT")
            _ensure_column(
                conn,
                "biominer_work_items",
                "stage",
                "stage TEXT NOT NULL DEFAULT 'default'",
            )
            _ensure_column(
                conn,
                "biominer_parquet_shards",
                "worker_id",
                "worker_id TEXT NOT NULL DEFAULT ''",
            )
            _ensure_column(
                conn, "biominer_parquet_shards", "metadata_json", "metadata_json TEXT"
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_biominer_runs_scope
                ON biominer_runs(job_name, stage, registry_version, status)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_biominer_work_items_pending
                ON biominer_work_items(job_name, stage, registry_version, status, created_at, work_key)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_biominer_work_items_completed
                ON biominer_work_items(job_name, stage, registry_version, status)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_biominer_parquet_shards_stage
                ON biominer_parquet_shards(job_name, registry_version, stage, run_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_biominer_compaction_inputs_source
                ON biominer_compaction_inputs(job_name, source_stage, registry_version, source_shard_id)
                """
            )


def _row_to_run(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "run_id": str(row["run_id"]),
        "job_name": str(row["job_name"]),
        "stage": str(row["stage"]),
        "registry_version": row["registry_version"],
        "status": str(row["status"]),
        "started_at": str(row["started_at"]),
        "ended_at": row["ended_at"],
        "config": _json_loads_dict(row["config_json"]),
        "summary": _json_loads_dict(row["summary_json"])
        if row["summary_json"]
        else None,
    }


def _row_to_work_item(row: sqlite3.Row) -> dict[str, Any]:
    payload = _json_loads_dict(row["payload_json"])
    return {
        "work_key": str(row["work_key"]),
        "job_name": row["job_name"],
        "stage": row["stage"],
        "registry_version": row["registry_version"],
        "status": row["status"],
        "payload": payload,
        "claimed_by": row["claimed_by"],
        "claimed_at": row["claimed_at"],
        "completed_at": row["completed_at"],
        "output_uri": row["output_uri"],
        "checksum": row["checksum"],
        "row_count": row["row_count"],
        "attempt_count": int(row["attempt_count"]),
        "error": row["error"],
        "created_at": row["created_at"],
    }


def _row_to_shard(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "shard_id": str(row["shard_id"]),
        "job_name": str(row["job_name"]),
        "stage": str(row["stage"]),
        "run_id": str(row["run_id"]),
        "registry_version": row["registry_version"],
        "worker_id": row["worker_id"],
        "uri": str(row["uri"]),
        "row_count": row["row_count"],
        "byte_count": row["byte_count"],
        "checksum": row["checksum"],
        "metadata": _json_loads_dict(row["metadata_json"])
        if row["metadata_json"]
        else {},
        "committed_at": str(row["committed_at"]),
    }


def _add_filter_if_not_none(
    clauses: list[str], params: list[Any], column: str, value: str | None
) -> None:
    if value is None:
        return
    clauses.append(f"{column} = ?")
    params.append(value)


def _add_nullable_filter(
    clauses: list[str], params: list[Any], column: str, value: str | None
) -> None:
    if value is None:
        clauses.append(f"{column} IS NULL")
        return
    clauses.append(f"{column} = ?")
    params.append(value)


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    columns = {
        str(row["name"])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _json_loads_dict(raw: Any) -> dict[str, Any]:
    payload = json.loads(str(raw or "{}"))
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object in workstore row")
    return payload


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()
