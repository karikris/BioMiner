from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path


DEFAULT_RATE_LIMIT_LEDGER_PATH = Path("data/rate_limits/flickr_global.sqlite")


class RateLimitExceeded(RuntimeError):
    """Raised when an API or photo-record hard cap would be exceeded."""


class FlickrRateLimiter:
    def __init__(
        self,
        ledger_path: str | Path = DEFAULT_RATE_LIMIT_LEDGER_PATH,
        *,
        soft_api_calls_per_hour: int = 3200,
        hard_api_calls_per_hour: int = 3600,
        hard_photo_records_per_hour: int = 3600,
        window_seconds: int = 3600,
    ) -> None:
        self.ledger_path = Path(ledger_path)
        self.soft_api_calls_per_hour = soft_api_calls_per_hour
        self.hard_api_calls_per_hour = hard_api_calls_per_hour
        self.hard_photo_records_per_hour = hard_photo_records_per_hour
        self.window_seconds = window_seconds
        self._lock = threading.RLock()
        self._init_db()

    def acquire_api_token(self, endpoint: str, work_item_id: str) -> None:
        with self._lock, self._connect() as conn:
            now = time.time()
            count = self._api_calls_in_window(conn, now)
            if count >= self.soft_api_calls_per_hour:
                raise RateLimitExceeded("soft API call cap reached")
            if count >= self.hard_api_calls_per_hour:
                raise RateLimitExceeded("hard API call cap reached")
            conn.execute(
                "INSERT INTO api_call_ledger(endpoint, work_item_id, status, created_at) VALUES (?, ?, ?, ?)",
                (endpoint, work_item_id, "reserved", now),
            )

    def reserve_photo_record_slots(self, requested: int) -> int:
        with self._lock, self._connect() as conn:
            remaining = self.hard_photo_records_per_hour - self._photo_records_in_window(conn, time.time())
            return max(0, min(requested, remaining))

    def log_call(self, endpoint: str, work_item_id: str, status: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE api_call_ledger
                SET status = ?
                WHERE id = (
                    SELECT id FROM api_call_ledger
                    WHERE endpoint = ? AND work_item_id = ? AND status = 'reserved'
                    ORDER BY created_at DESC
                    LIMIT 1
                )
                """,
                (status, endpoint, work_item_id),
            )

    def log_photo_records(self, photo_ids: list[str], work_item_id: str) -> None:
        with self._lock, self._connect() as conn:
            allowed = self.reserve_photo_record_slots(len(photo_ids))
            rows = [(photo_id, work_item_id, time.time()) for photo_id in photo_ids[:allowed]]
            conn.executemany(
                "INSERT OR IGNORE INTO photo_record_ledger(photo_id, work_item_id, created_at) VALUES (?, ?, ?)",
                rows,
            )

    def api_calls_in_window(self) -> int:
        with self._lock, self._connect() as conn:
            return self._api_calls_in_window(conn, time.time())

    def photo_records_in_window(self) -> int:
        with self._lock, self._connect() as conn:
            return self._photo_records_in_window(conn, time.time())

    def _init_db(self) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_call_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT NOT NULL,
                    work_item_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS photo_record_ledger (
                    photo_id TEXT PRIMARY KEY,
                    work_item_id TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.ledger_path, timeout=30, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _api_calls_in_window(self, conn: sqlite3.Connection, now: float) -> int:
        cutoff = now - self.window_seconds
        return int(conn.execute("SELECT count(*) FROM api_call_ledger WHERE created_at >= ?", (cutoff,)).fetchone()[0])

    def _photo_records_in_window(self, conn: sqlite3.Connection, now: float) -> int:
        cutoff = now - self.window_seconds
        return int(conn.execute("SELECT count(*) FROM photo_record_ledger WHERE created_at >= ?", (cutoff,)).fetchone()[0])
