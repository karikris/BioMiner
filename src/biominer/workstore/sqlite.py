from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
import hashlib
import json
import sqlite3


PENDING = "pending"
CLAIMED = "claimed"
COMPLETED = "completed"
FAILED = "failed"


class SQLiteWorkStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def enqueue_work(self, job_name: str, registry_version: str | None, items: list[dict[str, Any]]) -> int:
        inserted = 0
        with self._connect() as conn:
            for item in items:
                payload = dict(item)
                work_key = str(payload.pop("work_key", "") or _derive_work_key(job_name, registry_version, payload))
                result = conn.execute(
                    """
                    INSERT OR IGNORE INTO biominer_work_items (
                        work_key, job_name, registry_version, status, payload_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        work_key,
                        job_name,
                        registry_version,
                        PENDING,
                        json.dumps(payload, sort_keys=True, ensure_ascii=False),
                        _timestamp(),
                    ),
                )
                inserted += int(result.rowcount)
        return inserted

    def claim_next_batch(self, worker_id: str, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT work_key, job_name, registry_version, payload_json, attempt_count
                FROM biominer_work_items
                WHERE status = ?
                ORDER BY created_at, work_key
                LIMIT ?
                """,
                (PENDING, limit),
            ).fetchall()
            now = _timestamp()
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
            conn.execute("COMMIT")
        return [_row_to_work_item(row) for row in rows]

    def mark_completed(self, work_key: str, output_uri: str | None, checksum: str | None, row_count: int | None) -> None:
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

    def completed_keys(self, job_name: str, registry_version: str | None) -> set[str]:
        with self._connect() as conn:
            if registry_version is None:
                rows = conn.execute(
                    """
                    SELECT work_key
                    FROM biominer_work_items
                    WHERE job_name = ? AND registry_version IS NULL AND status = ?
                    """,
                    (job_name, COMPLETED),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT work_key
                    FROM biominer_work_items
                    WHERE job_name = ? AND registry_version = ? AND status = ?
                    """,
                    (job_name, registry_version, COMPLETED),
                ).fetchall()
        return {str(row["work_key"]) for row in rows}

    def stale_claims_to_pending(self, stale_after_seconds: int) -> int:
        cutoff = (datetime.now(UTC) - timedelta(seconds=stale_after_seconds)).isoformat()
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE biominer_work_items
                SET status = ?,
                    claimed_by = NULL,
                    claimed_at = NULL
                WHERE status = ?
                  AND claimed_at IS NOT NULL
                  AND claimed_at < ?
                """,
                (PENDING, CLAIMED, cutoff),
            )
        return int(result.rowcount)

    def register_shard(
        self,
        *,
        job_name: str,
        registry_version: str | None,
        stage: str,
        run_id: str,
        worker_id: str,
        uri: str,
        checksum: str | None,
        row_count: int,
        byte_count: int | None = None,
    ) -> None:
        shard_id = hashlib.sha256(uri.encode("utf-8")).hexdigest()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO biominer_parquet_shards (
                    shard_id, job_name, registry_version, stage, run_id, worker_id,
                    uri, row_count, byte_count, checksum, committed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    shard_id,
                    job_name,
                    registry_version,
                    stage,
                    run_id,
                    worker_id,
                    uri,
                    row_count,
                    byte_count,
                    checksum,
                    _timestamp(),
                ),
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS biominer_work_items (
                    work_key TEXT PRIMARY KEY,
                    job_name TEXT NOT NULL,
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
                CREATE INDEX IF NOT EXISTS idx_biominer_work_items_pending
                ON biominer_work_items(status, created_at, work_key)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_biominer_work_items_completed
                ON biominer_work_items(job_name, registry_version, status)
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
                    worker_id TEXT NOT NULL,
                    uri TEXT NOT NULL UNIQUE,
                    row_count INTEGER NOT NULL,
                    byte_count INTEGER,
                    checksum TEXT,
                    committed_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_biominer_parquet_shards_stage
                ON biominer_parquet_shards(job_name, registry_version, stage, run_id)
                """
            )


def _derive_work_key(job_name: str, registry_version: str | None, payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"job_name": job_name, "registry_version": registry_version, "payload": payload},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return f"{job_name}:{digest}"


def _row_to_work_item(row: sqlite3.Row) -> dict[str, Any]:
    payload = json.loads(str(row["payload_json"]))
    if not isinstance(payload, dict):
        raise ValueError(f"work item payload is not a JSON object: {row['work_key']}")
    return {
        "work_key": str(row["work_key"]),
        "job_name": row["job_name"],
        "registry_version": row["registry_version"],
        "payload": payload,
        "attempt_count": int(row["attempt_count"]),
    }


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()
