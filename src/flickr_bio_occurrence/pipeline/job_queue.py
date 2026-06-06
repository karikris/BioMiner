from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
from pathlib import Path
import sqlite3


PENDING = "pending"
CLAIMED = "claimed"
COMPLETED = "completed"
FAILED = "failed"
ACTIVE_STATUSES = {PENDING, CLAIMED}


@dataclass(frozen=True)
class ClassificationJob:
    job_id: str
    evidence_parquet_path: Path
    status: str
    model_version: str
    created_at: str
    claimed_at: str | None
    completed_at: str | None
    attempts: int
    error: str | None


class ClassificationJobQueue:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS classification_jobs (
                    job_id TEXT PRIMARY KEY,
                    evidence_parquet_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    claimed_at TEXT,
                    completed_at TEXT,
                    attempts INTEGER NOT NULL,
                    error TEXT,
                    UNIQUE(evidence_parquet_path, model_version)
                )
                """
            )

    def enqueue_evidence_shard(
        self,
        evidence_parquet_path: str | Path,
        *,
        model_version: str,
        job_id: str | None = None,
        now: datetime | None = None,
    ) -> ClassificationJob:
        evidence_path = Path(evidence_parquet_path)
        resolved_job_id = job_id or _job_id(evidence_path, model_version)
        created_at = _timestamp(now)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO classification_jobs (
                    job_id, evidence_parquet_path, status, model_version,
                    created_at, claimed_at, completed_at, attempts, error
                )
                VALUES (?, ?, ?, ?, ?, NULL, NULL, 0, NULL)
                """,
                (resolved_job_id, str(evidence_path), PENDING, model_version, created_at),
            )
        return self.get_job(resolved_job_id)

    def claim_next(self, *, now: datetime | None = None) -> ClassificationJob | None:
        claimed_at = _timestamp(now)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT *
                FROM classification_jobs
                WHERE status = ?
                ORDER BY created_at, job_id
                LIMIT 1
                """,
                (PENDING,),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            conn.execute(
                """
                UPDATE classification_jobs
                SET status = ?, claimed_at = ?, attempts = attempts + 1, error = NULL
                WHERE job_id = ?
                """,
                (CLAIMED, claimed_at, row["job_id"]),
            )
            conn.execute("COMMIT")
        return self.get_job(str(row["job_id"]))

    def mark_complete(self, job_id: str, *, now: datetime | None = None) -> ClassificationJob:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE classification_jobs
                SET status = ?, completed_at = ?, error = NULL
                WHERE job_id = ?
                """,
                (COMPLETED, _timestamp(now), job_id),
            )
        return self.get_job(job_id)

    def mark_failed(self, job_id: str, *, error: str, now: datetime | None = None) -> ClassificationJob:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE classification_jobs
                SET status = ?, completed_at = ?, error = ?
                WHERE job_id = ?
                """,
                (FAILED, _timestamp(now), error, job_id),
            )
        return self.get_job(job_id)

    def retry_stale_claimed(
        self,
        *,
        stale_after: timedelta,
        now: datetime | None = None,
        max_attempts: int = 3,
    ) -> list[ClassificationJob]:
        cutoff = (now or datetime.now(UTC)) - stale_after
        updated: list[str] = []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM classification_jobs
                WHERE status = ? AND claimed_at IS NOT NULL AND claimed_at < ?
                ORDER BY claimed_at, job_id
                """,
                (CLAIMED, _timestamp(cutoff)),
            ).fetchall()
            for row in rows:
                next_status = FAILED if int(row["attempts"]) >= max_attempts else PENDING
                error = "stale_claim_retry_limit_reached" if next_status == FAILED else "stale_claim_requeued"
                conn.execute(
                    """
                    UPDATE classification_jobs
                    SET status = ?, claimed_at = NULL, error = ?
                    WHERE job_id = ?
                    """,
                    (next_status, error, row["job_id"]),
                )
                updated.append(str(row["job_id"]))
        return [self.get_job(job_id) for job_id in updated]

    def get_job(self, job_id: str) -> ClassificationJob:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM classification_jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return _row_to_job(row)

    def list_jobs(self, *, status: str | None = None) -> list[ClassificationJob]:
        with self._connect() as conn:
            if status is None:
                rows = conn.execute("SELECT * FROM classification_jobs ORDER BY created_at, job_id").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM classification_jobs WHERE status = ? ORDER BY created_at, job_id",
                    (status,),
                ).fetchall()
        return [_row_to_job(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn


def _row_to_job(row: sqlite3.Row) -> ClassificationJob:
    return ClassificationJob(
        job_id=str(row["job_id"]),
        evidence_parquet_path=Path(str(row["evidence_parquet_path"])),
        status=str(row["status"]),
        model_version=str(row["model_version"]),
        created_at=str(row["created_at"]),
        claimed_at=row["claimed_at"],
        completed_at=row["completed_at"],
        attempts=int(row["attempts"]),
        error=row["error"],
    )


def _job_id(evidence_parquet_path: Path, model_version: str) -> str:
    key = f"{evidence_parquet_path.resolve()}|{model_version}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"vision-{digest}"


def _timestamp(value: datetime | None = None) -> str:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).isoformat()
