from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable

import httpx
import polars as pl

from biominer.filter.extractor import build_evidence_frame
from biominer.flickr_fetch.endpoints import FLICKR_REST_BASE_URL, SEARCH_METHOD
from biominer.flickr_fetch.query_planner import (
    FlickrQuery,
    build_count_probes,
    deduplicate_photo_records,
    flickr_search_params,
    plan_queries_from_count,
    query_date_kind,
    query_hash,
    query_max_date,
    query_min_date,
    split_priority,
)
from biominer.storage.parquet import write_parquet


SOFT_API_CALLS_PER_HOUR = 3500
HARD_API_CALLS_PER_HOUR = 3600
PENDING = "pending"
CLAIMED = "claimed"
COMPLETED = "completed"
FAILED = "failed"
DEFAULT_STALE_CLAIM_SECONDS = 3600
DEFAULT_MAX_RETRIES = 2

FetchMetadata = Callable[[FlickrQuery], dict[str, Any]]
ProgressCallback = Callable[[dict[str, Any]], None]


class FlickrFetchError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False, http_status: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.http_status = http_status


class FlickrFetchFailure(RuntimeError):
    def __init__(self, error: FlickrFetchError, *, attempts: int) -> None:
        super().__init__(str(error))
        self.error = error
        self.attempts = attempts


@dataclass(frozen=True)
class PollOnceResult:
    state_db: Path
    raw_responses_written: int
    evidence_rows_written: int
    evidence_rows_total: int
    source_records_inserted: int
    duplicate_records_skipped: int
    query_hits_inserted: int
    duplicate_query_hits_skipped: int
    image_urls_queued: int
    work_items_claimed: int
    api_calls_made: int
    remaining_soft_budget: int
    remaining_hard_budget: int
    stale_claims_requeued: int


@dataclass(frozen=True)
class PageEnsureResult:
    target_pages: int
    new_pages_enqueued: int
    total_known_work_items: int
    highest_known_page: int
    missing_pages: tuple[int, ...]
    warnings: tuple[dict[str, object], ...]


class MetadataPollState:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def ensure_seed_work_items(self, queries: tuple[FlickrQuery, ...] | None = None) -> int:
        if self.work_item_count() > 0:
            return 0
        seeded = 0
        for query in queries or build_count_probes():
            seeded += self.enqueue_work_item(query)
        return seeded

    def work_item_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT count(*) FROM flickr_work_items").fetchone()[0])

    def enqueue_work_item(self, query: FlickrQuery) -> int:
        with self._connect() as conn:
            result = _insert_work_item(conn, query)
        return int(result.rowcount)

    def ensure_reported_pages(self, query: FlickrQuery, *, response_pages: int, response_perpage: int | None = None) -> PageEnsureResult:
        if query.lane == "count_probe" or response_pages <= 0:
            return PageEnsureResult(
                target_pages=0,
                new_pages_enqueued=0,
                total_known_work_items=0,
                highest_known_page=0,
                missing_pages=(),
                warnings=(),
            )
        effective_perpage = response_perpage or query.per_page
        if effective_perpage <= 0:
            effective_perpage = query.per_page
        warnings: list[dict[str, object]] = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            before = _pagination_rows(conn, query)
            previous_reported_page_counts = [int(row["response_pages"]) for row in before if row["response_pages"] is not None]
            previous_reported_page_count = max(previous_reported_page_counts, default=0)
            highest_before = max((int(row["page"]) for row in before), default=0)
            if previous_reported_page_count and previous_reported_page_count != response_pages:
                warnings.append(
                    {
                        "event": "pagination_metadata_changed",
                        "level": "warning",
                        "previous_response_pages": previous_reported_page_count,
                        "response_pages": response_pages,
                        "min_date": query_min_date(query),
                        "max_date": query_max_date(query),
                    }
                )
            target_pages = max(response_pages, previous_reported_page_count)
            existing_pages = {int(row["page"]) for row in before}
            inserted = 0
            for page in range(1, target_pages + 1):
                if page in existing_pages:
                    continue
                result = _insert_work_item(conn, replace(query, page=page, per_page=effective_perpage))
                inserted += int(result.rowcount)
            after = _pagination_rows(conn, query)
            known_pages = {int(row["page"]) for row in after}
            missing_pages = tuple(page for page in range(1, target_pages + 1) if page not in known_pages)
            highest_after = max(known_pages, default=0)
            if response_pages > highest_before:
                warnings.append(
                    {
                        "event": "pagination_pages_discovered",
                        "level": "warning",
                        "response_pages": response_pages,
                        "highest_queued_page_before_enqueue": highest_before,
                        "min_date": query_min_date(query),
                        "max_date": query_max_date(query),
                    }
                )
            if response_pages > highest_after:
                warnings.append(
                    {
                        "event": "pagination_invariant_failed",
                        "level": "error",
                        "response_pages": response_pages,
                        "highest_queued_page": highest_after,
                        "min_date": query_min_date(query),
                        "max_date": query_max_date(query),
                    }
                )
            if missing_pages:
                warnings.append(
                    {
                        "event": "pagination_missing_pages",
                        "level": "error",
                        "response_pages": target_pages,
                        "missing_pages": list(missing_pages),
                        "min_date": query_min_date(query),
                        "max_date": query_max_date(query),
                    }
                )
            conn.execute("COMMIT")
        return PageEnsureResult(
            target_pages=target_pages,
            new_pages_enqueued=inserted,
            total_known_work_items=len(after),
            highest_known_page=highest_after,
            missing_pages=missing_pages,
            warnings=tuple(warnings),
        )

    def requeue_stale_claims(self, *, stale_after_seconds: int = DEFAULT_STALE_CLAIM_SECONDS, now: datetime | None = None) -> int:
        cutoff = datetime.fromtimestamp((now or datetime.now(UTC)).timestamp() - stale_after_seconds, UTC).isoformat()
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE flickr_work_items
                SET status = ?, claimed_at = NULL, error = NULL
                WHERE status = ?
                  AND claimed_at IS NOT NULL
                  AND claimed_at < ?
                """,
                (PENDING, CLAIMED, cutoff),
            )
        return int(result.rowcount)

    def api_calls_in_window(self, *, now: datetime | None = None) -> int:
        with self._connect() as conn:
            return self._api_calls_in_window(conn, _unix_timestamp(now))

    def remaining_api_budget(self, *, max_api_calls: int, now: datetime | None = None) -> tuple[int, int]:
        used = self.api_calls_in_window(now=now)
        soft_limit = min(max_api_calls, SOFT_API_CALLS_PER_HOUR)
        return max(0, soft_limit - used), max(0, HARD_API_CALLS_PER_HOUR - used)

    def claim_and_reserve_pending(
        self,
        *,
        limit: int,
        max_api_calls: int,
        endpoint: str,
        now: datetime | None = None,
    ) -> list[tuple[str, FlickrQuery]]:
        if limit <= 0:
            return []
        claimed: list[tuple[str, FlickrQuery]] = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            used = self._api_calls_in_window(conn, _unix_timestamp(now))
            soft_limit = min(max_api_calls, SOFT_API_CALLS_PER_HOUR)
            claim_limit = min(limit, max(0, soft_limit - used), max(0, HARD_API_CALLS_PER_HOUR - used))
            if claim_limit <= 0:
                conn.execute("COMMIT")
                return []
            rows = conn.execute(
                """
                SELECT work_item_id, query_json
                FROM flickr_work_items
                WHERE status = ?
                ORDER BY
                    COALESCE(split_depth, 0),
                    COALESCE(split_priority, 99),
                    COALESCE(date_kind, ''),
                    COALESCE(min_date, ''),
                    COALESCE(max_date, ''),
                    COALESCE(slice_index, 999999),
                    COALESCE(bbox_index, 999999),
                    COALESCE(bbox_label, ''),
                    COALESCE(term, ''),
                    CASE lane WHEN 'count_probe' THEN 0 WHEN 'normal_page' THEN 1 WHEN 'bbox_page' THEN 1 ELSE 99 END,
                    page,
                    COALESCE(query_hash, work_item_id)
                LIMIT ?
                """,
                (PENDING, claim_limit),
            ).fetchall()
            timestamp = _timestamp(now)
            unix_timestamp = _unix_timestamp(now)
            for row in rows:
                conn.execute(
                    "UPDATE flickr_work_items SET status = ?, claimed_at = ? WHERE work_item_id = ?",
                    (CLAIMED, timestamp, row["work_item_id"]),
                )
                conn.execute(
                    """
                    INSERT INTO api_call_ledger(endpoint, work_item_id, status, created_at, started_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (endpoint, row["work_item_id"], "reserved", unix_timestamp, unix_timestamp),
                )
                claimed.append((str(row["work_item_id"]), _query_from_json(str(row["query_json"]))))
            conn.execute("COMMIT")
        return claimed

    def reserve_retry_api_call(
        self,
        *,
        work_item_id: str,
        endpoint: str,
        max_api_calls: int,
        now: datetime | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            used = self._api_calls_in_window(conn, _unix_timestamp(now))
            soft_limit = min(max_api_calls, SOFT_API_CALLS_PER_HOUR)
            if used >= soft_limit:
                conn.execute("COMMIT")
                raise FlickrFetchError("soft API call cap reached before retry", retryable=False)
            if used >= HARD_API_CALLS_PER_HOUR:
                conn.execute("COMMIT")
                raise FlickrFetchError("hard API call cap reached before retry", retryable=False)
            conn.execute(
                """
                INSERT INTO api_call_ledger(endpoint, work_item_id, status, created_at, started_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (endpoint, work_item_id, "reserved", _unix_timestamp(now), _unix_timestamp(now)),
            )
            conn.execute("COMMIT")

    def api_budget_summary(self, *, max_api_calls: int = SOFT_API_CALLS_PER_HOUR, now: datetime | None = None) -> dict[str, object]:
        soft_remaining, hard_remaining = self.remaining_api_budget(max_api_calls=max_api_calls, now=now)
        return {
            "state_db": str(self.path),
            "api_calls_in_window": self.api_calls_in_window(now=now),
            "remaining_soft_budget": soft_remaining,
            "remaining_hard_budget": hard_remaining,
            "soft_api_calls_per_hour": SOFT_API_CALLS_PER_HOUR,
            "hard_api_calls_per_hour": HARD_API_CALLS_PER_HOUR,
            "photo_records_in_window": "not_instrumented",
            "hard_photo_records_per_hour": "not_instrumented",
            "window_seconds": 3600,
        }

    def _api_calls_in_window(self, conn: sqlite3.Connection, now: float) -> int:
        cutoff = now - 3600
        return int(conn.execute("SELECT count(*) FROM api_call_ledger WHERE created_at >= ?", (cutoff,)).fetchone()[0])

    def log_api_call(self, *, work_item_id: str, endpoint: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO api_call_ledger(endpoint, work_item_id, status, created_at, started_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (endpoint, work_item_id, status, _unix_timestamp(), _unix_timestamp()),
            )

    def update_api_call_status(
        self,
        *,
        work_item_id: str,
        status: str,
        duration_sec: float | None = None,
        http_status: int | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE api_call_ledger
                SET status = ?,
                    finished_at = ?,
                    duration_sec = ?,
                    http_status = ?
                WHERE id = (
                    SELECT id
                    FROM api_call_ledger
                    WHERE work_item_id = ? AND status = 'reserved'
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                )
                """,
                (status, _unix_timestamp(), duration_sec, http_status, work_item_id),
            )

    def complete_work_item(
        self,
        work_item_id: str,
        *,
        records_returned: int | None = None,
        response_total: int | None = None,
        response_pages: int | None = None,
        response_page: int | None = None,
        response_perpage: int | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE flickr_work_items
                SET status = ?,
                    completed_at = ?,
                    error = NULL,
                    records_returned = ?,
                    response_total = ?,
                    response_pages = ?,
                    response_page = ?,
                    response_perpage = ?
                WHERE work_item_id = ?
                """,
                (
                    COMPLETED,
                    _timestamp(),
                    records_returned,
                    response_total,
                    response_pages,
                    response_page,
                    response_perpage,
                    work_item_id,
                ),
            )

    def fail_work_item(self, work_item_id: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE flickr_work_items SET status = ?, completed_at = ?, error = ? WHERE work_item_id = ?",
                (FAILED, _timestamp(), error, work_item_id),
            )

    def insert_source_records(self, records: list[dict[str, Any]], *, source_query: FlickrQuery) -> tuple[int, int, int, int, int]:
        inserted = 0
        unique_records = deduplicate_photo_records(records)
        skipped = len(records) - len(unique_records)
        queued = 0
        query_hits_inserted = 0
        duplicate_query_hits = 0
        with self._connect() as conn:
            for record in unique_records:
                photo_id = str(record.get("id") or "")
                image_url = str(record.get("url_l") or record.get("url_m") or "")
                if not photo_id or not image_url:
                    skipped += 1
                    continue
                source_record_hash = _source_record_hash(record)
                image_url_kind = "url_l" if record.get("url_l") else "url_m"
                conn.execute(
                    """
                    INSERT OR IGNORE INTO source_record_image_urls (
                        source, flickr_photo_id, image_url, image_url_kind, first_seen_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    ("flickr", photo_id, image_url, image_url_kind, _timestamp()),
                )
                existing = conn.execute(
                    """
                    SELECT image_url
                    FROM source_records
                    WHERE source = ? AND flickr_photo_id = ?
                    ORDER BY created_at
                    LIMIT 1
                    """,
                    ("flickr", photo_id),
                ).fetchone()
                canonical_image_url = str(existing["image_url"]) if existing else image_url
                source_inserted = False
                if existing is None:
                    result = conn.execute(
                        """
                        INSERT OR IGNORE INTO source_records (
                            source, flickr_photo_id, image_url, image_url_kind,
                            source_record_hash, query_term, query_language,
                            query_field, raw_json, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            "flickr",
                            photo_id,
                            image_url,
                            image_url_kind,
                            source_record_hash,
                            source_query.term,
                            source_query.language,
                            source_query.search_field,
                            json.dumps(record, sort_keys=True, ensure_ascii=False),
                            _timestamp(),
                        ),
                    )
                    source_inserted = bool(result.rowcount)
                query_result = conn.execute(
                    """
                    INSERT OR IGNORE INTO source_record_query_hits (
                        source, flickr_photo_id, image_url, query_field,
                        query_term, query_language, query_lane, query_page,
                        first_seen_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "flickr",
                        photo_id,
                        canonical_image_url,
                        source_query.search_field,
                        source_query.term,
                        source_query.language,
                        source_query.lane,
                        source_query.page,
                        _timestamp(),
                    ),
                )
                if query_result.rowcount:
                    query_hits_inserted += 1
                else:
                    duplicate_query_hits += 1
                if source_inserted:
                    inserted += 1
                    queue_result = conn.execute(
                        """
                        INSERT OR IGNORE INTO image_triage_queue (
                            source, flickr_photo_id, image_url, image_url_kind,
                            source_record_hash, status, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            "flickr",
                            photo_id,
                            image_url,
                            "url_l" if record.get("url_l") else "url_m",
                            source_record_hash,
                            PENDING,
                            _timestamp(),
                        ),
                    )
                    queued += int(queue_result.rowcount)
                else:
                    skipped += 1
        return inserted, skipped, queued, query_hits_inserted, duplicate_query_hits

    def source_records_with_query_provenance(self) -> pl.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    source_records.source,
                    source_records.flickr_photo_id,
                    source_records.image_url,
                    source_records.image_url_kind,
                    source_records.query_field AS first_query_field,
                    source_records.query_term AS first_query_term,
                    source_records.query_language AS first_query_language,
                    source_record_query_hits.query_field,
                    source_record_query_hits.query_term
                FROM source_records
                LEFT JOIN source_record_query_hits
                  ON source_record_query_hits.source = source_records.source
                 AND source_record_query_hits.flickr_photo_id = source_records.flickr_photo_id
                 AND source_record_query_hits.image_url = source_records.image_url
                ORDER BY source_records.flickr_photo_id, source_record_query_hits.query_field, source_record_query_hits.query_term
                """
            ).fetchall()
        grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in rows:
            key = (str(row["source"]), str(row["flickr_photo_id"]), str(row["image_url"]))
            item = grouped.setdefault(
                key,
                {
                    "source": row["source"],
                    "flickr_photo_id": row["flickr_photo_id"],
                    "image_url": row["image_url"],
                    "image_url_kind": row["image_url_kind"],
                    "first_query_field": row["first_query_field"],
                    "first_query_term": row["first_query_term"],
                    "first_query_language": row["first_query_language"],
                    "first_query_label": f"{row['first_query_field']}:{row['first_query_term']}",
                    "all_query_labels": [],
                    "all_query_terms": [],
                    "all_query_fields": [],
                    "query_hit_count": 0,
                },
            )
            if row["query_field"] and row["query_term"]:
                item["all_query_labels"].append(f"{row['query_field']}:{row['query_term']}")
                item["all_query_terms"].append(row["query_term"])
                item["all_query_fields"].append(row["query_field"])
                item["query_hit_count"] += 1
        return pl.DataFrame(list(grouped.values())) if grouped else pl.DataFrame()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_call_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT NOT NULL,
                    work_item_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    duration_sec REAL,
                    http_status INTEGER
                )
                """
            )
            self._ensure_api_call_columns(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS flickr_work_items (
                    work_item_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    query_json TEXT NOT NULL,
                    lane TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    per_page INTEGER NOT NULL,
                    split_depth INTEGER,
                    split_priority INTEGER,
                    split_reason TEXT,
                    parent_query_hash TEXT,
                    parent_total INTEGER,
                    date_kind TEXT,
                    min_date TEXT,
                    max_date TEXT,
                    bbox_index INTEGER,
                    slice_index INTEGER,
                    bbox_label TEXT,
                    term TEXT,
                    query_hash TEXT,
                    claimed_at TEXT,
                    completed_at TEXT,
                    error TEXT,
                    records_returned INTEGER,
                    response_total INTEGER,
                    response_pages INTEGER,
                    response_page INTEGER,
                    response_perpage INTEGER,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._ensure_work_item_columns(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS source_records (
                    source TEXT NOT NULL,
                    flickr_photo_id TEXT NOT NULL,
                    image_url TEXT NOT NULL,
                    image_url_kind TEXT NOT NULL,
                    source_record_hash TEXT NOT NULL,
                    query_term TEXT NOT NULL,
                    query_language TEXT NOT NULL,
                    query_field TEXT NOT NULL,
                    raw_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (source, flickr_photo_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS source_record_image_urls (
                    source TEXT NOT NULL,
                    flickr_photo_id TEXT NOT NULL,
                    image_url TEXT NOT NULL,
                    image_url_kind TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    PRIMARY KEY (source, flickr_photo_id, image_url)
                )
                """
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO source_record_image_urls (
                    source, flickr_photo_id, image_url, image_url_kind, first_seen_at
                )
                SELECT source, flickr_photo_id, image_url, image_url_kind, created_at
                FROM source_records
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS source_record_query_hits (
                    source TEXT NOT NULL,
                    flickr_photo_id TEXT NOT NULL,
                    image_url TEXT NOT NULL,
                    query_field TEXT NOT NULL,
                    query_term TEXT NOT NULL,
                    query_language TEXT NOT NULL,
                    query_lane TEXT NOT NULL,
                    query_page INTEGER NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    PRIMARY KEY (
                        source, flickr_photo_id, image_url,
                        query_field, query_term, query_language
                    )
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS image_triage_queue (
                    source TEXT NOT NULL,
                    flickr_photo_id TEXT NOT NULL,
                    image_url TEXT NOT NULL,
                    image_url_kind TEXT NOT NULL,
                    source_record_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (source, flickr_photo_id, image_url)
                )
                """
            )

    def _ensure_work_item_columns(self, conn: sqlite3.Connection) -> None:
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(flickr_work_items)").fetchall()}
        columns = {
            "split_depth": "INTEGER",
            "split_priority": "INTEGER",
            "split_reason": "TEXT",
            "parent_query_hash": "TEXT",
            "parent_total": "INTEGER",
            "date_kind": "TEXT",
            "min_date": "TEXT",
            "max_date": "TEXT",
            "bbox_index": "INTEGER",
            "slice_index": "INTEGER",
            "bbox_label": "TEXT",
            "term": "TEXT",
            "query_hash": "TEXT",
            "records_returned": "INTEGER",
            "response_total": "INTEGER",
            "response_pages": "INTEGER",
            "response_page": "INTEGER",
            "response_perpage": "INTEGER",
        }
        for name, sql_type in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE flickr_work_items ADD COLUMN {name} {sql_type}")

    def _ensure_api_call_columns(self, conn: sqlite3.Connection) -> None:
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(api_call_ledger)").fetchall()}
        columns = {
            "started_at": "REAL",
            "finished_at": "REAL",
            "duration_sec": "REAL",
            "http_status": "INTEGER",
        }
        for name, sql_type in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE api_call_ledger ADD COLUMN {name} {sql_type}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn


def poll_once(
    *,
    state_db: str | Path,
    raw_root: str | Path,
    evidence_output: str | Path,
    max_api_calls: int = SOFT_API_CALLS_PER_HOUR,
    api_key: str | None = None,
    fetch_metadata: FetchMetadata | None = None,
    workers: int = 1,
    stale_claim_seconds: int = DEFAULT_STALE_CLAIM_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff_seconds: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> PollOnceResult:
    state = MetadataPollState(state_db)
    stale_requeued = state.requeue_stale_claims(stale_after_seconds=stale_claim_seconds)
    _progress(progress_callback, {"event": "stale_claims_requeued", "count": stale_requeued})
    state.ensure_seed_work_items()
    soft_remaining, hard_remaining = state.remaining_api_budget(max_api_calls=max_api_calls)
    _progress(
        progress_callback,
        {
            "event": "budget_checked",
            "remaining_soft_budget": soft_remaining,
            "remaining_hard_budget": hard_remaining,
            "max_api_calls": max_api_calls,
        },
    )
    raw_written = 0
    records_inserted = 0
    duplicates = 0
    query_hits_inserted = 0
    duplicate_query_hits = 0
    queued = 0
    work_items_claimed = 0
    api_calls_made = 0
    evidence_rows_written = 0
    fetcher = fetch_metadata or _http_fetcher(api_key=api_key)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        while True:
            soft_remaining, hard_remaining = state.remaining_api_budget(max_api_calls=max_api_calls)
            claim_limit = min(soft_remaining, hard_remaining, max(1, workers))
            if claim_limit <= 0:
                break
            claimed = state.claim_and_reserve_pending(
                limit=claim_limit,
                max_api_calls=max_api_calls,
                endpoint=SEARCH_METHOD,
            )
            _progress(progress_callback, {"event": "work_claimed", "claimed": len(claimed), "claim_limit": claim_limit})
            if not claimed:
                break
            work_items_claimed += len(claimed)
            pending: dict[Future[tuple[dict[str, Any], int]], tuple[str, FlickrQuery]] = {
                pool.submit(
                    _fetch_with_retries,
                    state=state,
                    work_item_id=work_item_id,
                    query=query,
                    fetcher=fetcher,
                    max_api_calls=max_api_calls,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                ): (work_item_id, query)
                for work_item_id, query in claimed
            }
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    work_item_id, query = pending.pop(future)
                    try:
                        payload, attempts = future.result()
                        api_calls_made += attempts
                        raw_written += 1
                        _write_raw_response(raw_root=Path(raw_root), work_item_id=work_item_id, query=query, payload=payload)
                        evidence_rows_written += _write_evidence_shard(
                            evidence_output=evidence_output,
                            work_item_id=work_item_id,
                            payload=payload,
                        )
                        total = _payload_total(payload)
                        response_pages = _payload_pages(payload)
                        response_page = _payload_page(payload)
                        response_perpage = _payload_perpage(payload)
                        if query.lane == "count_probe":
                            for next_query in plan_queries_from_count(query, total=total):
                                state.enqueue_work_item(next_query)
                            records_returned = None
                            _progress(
                                progress_callback,
                                {
                                    "event": "count_probe_completed",
                                    "lane": query.lane,
                                    "page": query.page,
                                    "response_total": total,
                                },
                            )
                        else:
                            records = _payload_photo_records(payload)
                            inserted, skipped, queued_count, query_hits, duplicate_hits = state.insert_source_records(records, source_query=query)
                            records_returned = len(records)
                            records_inserted += inserted
                            duplicates += skipped
                            query_hits_inserted += query_hits
                            duplicate_query_hits += duplicate_hits
                            queued += queued_count
                            page_ensure = state.ensure_reported_pages(
                                query,
                                response_pages=response_pages,
                                response_perpage=response_perpage,
                            )
                            _progress(
                                progress_callback,
                                {
                                    "event": "page_completed",
                                    "lane": query.lane,
                                    "page": query.page,
                                    "per_page": query.per_page,
                                    "min_date": query_min_date(query),
                                    "max_date": query_max_date(query),
                                    "response_total": total,
                                    "response_pages": response_pages,
                                    "response_page": response_page,
                                    "response_perpage": response_perpage,
                                    "records_returned": records_returned,
                                    "records_inserted": inserted,
                                    "duplicates_skipped": skipped,
                                    "remaining_pages_enqueued": page_ensure.new_pages_enqueued,
                                    "known_work_items_for_query": page_ensure.total_known_work_items,
                                    "highest_known_page": page_ensure.highest_known_page,
                                    "missing_pages": list(page_ensure.missing_pages),
                                },
                            )
                            for warning in page_ensure.warnings:
                                _progress(progress_callback, warning)
                            if page_ensure.new_pages_enqueued:
                                _progress(
                                    progress_callback,
                                    {
                                        "event": "remaining_pages_enqueued",
                                        "enqueued": page_ensure.new_pages_enqueued,
                                        "target_pages": page_ensure.target_pages,
                                        "known_work_items_for_query": page_ensure.total_known_work_items,
                                        "missing_pages": list(page_ensure.missing_pages),
                                        "min_date": query_min_date(query),
                                        "max_date": query_max_date(query),
                                    },
                                )
                        state.complete_work_item(
                            work_item_id,
                            records_returned=records_returned,
                            response_total=total,
                            response_pages=response_pages,
                            response_page=response_page,
                            response_perpage=response_perpage,
                        )
                    except FlickrFetchFailure as exc:
                        api_calls_made += exc.attempts
                        state.fail_work_item(work_item_id, str(exc))
                        _progress(
                            progress_callback,
                            {
                                "event": "work_failed",
                                "work_item_id": work_item_id,
                                "lane": query.lane,
                                "page": query.page,
                                "error": str(exc),
                                "attempts": exc.attempts,
                                "retryable": exc.error.retryable,
                                "http_status": exc.error.http_status,
                            },
                        )
                    except Exception as exc:  # noqa: BLE001 - poller records unexpected failure and exits bounded cycle.
                        state.update_api_call_status(work_item_id=work_item_id, status="failed")
                        api_calls_made += 1
                        state.fail_work_item(work_item_id, str(exc))
                        _progress(
                            progress_callback,
                            {
                                "event": "work_failed",
                                "work_item_id": work_item_id,
                                "lane": query.lane,
                                "page": query.page,
                                "error": str(exc),
                                "attempts": 1,
                                "retryable": False,
                                "http_status": None,
                            },
                        )

    evidence_rows_total = _compact_evidence_output(evidence_output)
    soft_after, hard_after = state.remaining_api_budget(max_api_calls=max_api_calls)
    result = PollOnceResult(
        state_db=Path(state_db),
        raw_responses_written=raw_written,
        evidence_rows_written=evidence_rows_written,
        evidence_rows_total=evidence_rows_total,
        source_records_inserted=records_inserted,
        duplicate_records_skipped=duplicates,
        query_hits_inserted=query_hits_inserted,
        duplicate_query_hits_skipped=duplicate_query_hits,
        image_urls_queued=queued,
        work_items_claimed=work_items_claimed,
        api_calls_made=api_calls_made,
        remaining_soft_budget=soft_after,
        remaining_hard_budget=hard_after,
        stale_claims_requeued=stale_requeued,
    )
    _progress(progress_callback, {"event": "poll_completed", **{**result.__dict__, "state_db": str(result.state_db)}})
    return result


def _progress(callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    if callback:
        callback(event)


def _http_fetcher(*, api_key: str | None) -> FetchMetadata:
    if not api_key:
        raise RuntimeError("Flickr API key is required for poll-once")

    def fetch(query: FlickrQuery) -> dict[str, Any]:
        params = {
            "method": SEARCH_METHOD,
            "api_key": api_key,
            **flickr_search_params(query),
            "format": "json",
            "nojsoncallback": 1,
        }
        with httpx.Client(timeout=30) as client:
            response = client.get(FLICKR_REST_BASE_URL, params=params)
        response.raise_for_status()
        return response.json()

    return fetch


def _fetch_with_retries(
    *,
    state: MetadataPollState,
    work_item_id: str,
    query: FlickrQuery,
    fetcher: FetchMetadata,
    max_api_calls: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> tuple[dict[str, Any], int]:
    attempts = 0
    while True:
        attempts += 1
        started = time.perf_counter()
        try:
            payload = fetcher(query)
            _validate_flickr_search_payload(payload)
            state.update_api_call_status(
                work_item_id=work_item_id,
                status="ok",
                duration_sec=time.perf_counter() - started,
            )
            return payload, attempts
        except Exception as exc:  # noqa: BLE001 - classification keeps retry policy explicit.
            error = _classify_fetch_error(exc)
            state.update_api_call_status(
                work_item_id=work_item_id,
                status="failed",
                duration_sec=time.perf_counter() - started,
                http_status=error.http_status,
            )
            if not error.retryable or attempts > max_retries:
                raise FlickrFetchFailure(error, attempts=attempts) from exc
            if retry_backoff_seconds > 0:
                time.sleep(retry_backoff_seconds * (2 ** (attempts - 1)))
            try:
                state.reserve_retry_api_call(
                    work_item_id=work_item_id,
                    endpoint=SEARCH_METHOD,
                    max_api_calls=max_api_calls,
                )
            except FlickrFetchError as budget_error:
                raise FlickrFetchFailure(budget_error, attempts=attempts) from exc


def _classify_fetch_error(exc: Exception) -> FlickrFetchError:
    if isinstance(exc, FlickrFetchError):
        return exc
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return FlickrFetchError(
            f"HTTP {status} from Flickr",
            retryable=status == 429 or 500 <= status <= 599,
            http_status=status,
        )
    if isinstance(exc, httpx.TimeoutException | httpx.TransportError):
        return FlickrFetchError(str(exc) or exc.__class__.__name__, retryable=True)
    return FlickrFetchError(str(exc) or exc.__class__.__name__, retryable=False)


def _validate_flickr_search_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise FlickrFetchError("Flickr response must be a JSON object")
    if payload.get("stat") == "fail":
        code = payload.get("code")
        message = payload.get("message") or "Flickr API error"
        raise FlickrFetchError(f"Flickr API error {code}: {message}")
    photos = payload.get("photos")
    if not isinstance(photos, dict):
        raise FlickrFetchError("Flickr response missing photos object")
    for key in ("total", "pages", "page", "perpage"):
        _response_int(photos.get(key), key=key)
    rows = photos.get("photo", [])
    if rows is None:
        rows = []
    if not isinstance(rows, list):
        raise FlickrFetchError("Flickr response photos.photo must be a list")


def _write_raw_response(*, raw_root: Path, work_item_id: str, query: FlickrQuery, payload: dict[str, Any]) -> Path:
    target_dir = raw_root / "flickr" / "photos_search" / query.search_field / _safe_query_variant(query.term)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{query.lane}-{query.page:05d}-{work_item_id[:12]}.json"
    target.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return target


def _safe_query_variant(term: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in term.casefold()).strip("_")


def _write_evidence_shard(*, evidence_output: str | Path, work_item_id: str, payload: dict[str, Any]) -> int:
    frame = build_evidence_frame([payload], species_query="multilingual_lepidoptera")
    if frame.is_empty():
        return 0
    shard_root = _evidence_shard_root(Path(evidence_output))
    shard_root.mkdir(parents=True, exist_ok=True)
    write_parquet(frame, shard_root / f"{work_item_id}.parquet")
    return frame.height


def _compact_evidence_output(evidence_output: str | Path) -> int:
    output = Path(evidence_output)
    shard_root = _evidence_shard_root(output)
    _ensure_legacy_evidence_shard(output=output, shard_root=shard_root)
    shard_paths = sorted(shard_root.glob("*.parquet")) if shard_root.exists() else []
    if not shard_paths:
        return 0
    frame = pl.read_parquet(shard_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(frame, output)
    return frame.height


def _ensure_legacy_evidence_shard(*, output: Path, shard_root: Path) -> None:
    if not output.exists() or (shard_root.exists() and any(shard_root.glob("*.parquet"))):
        return
    try:
        legacy = pl.read_parquet(output)
    except Exception:
        return
    if legacy.is_empty():
        return
    shard_root.mkdir(parents=True, exist_ok=True)
    write_parquet(legacy, shard_root / "__legacy__.parquet")


def _evidence_shard_root(output: Path) -> Path:
    return output.parent / f"{output.stem}_pages"


def _payload_total(payload: dict[str, Any]) -> int:
    return _response_int(payload.get("photos", {}).get("total"), key="total")


def _payload_pages(payload: dict[str, Any]) -> int:
    return _response_int(payload.get("photos", {}).get("pages"), key="pages")


def _payload_page(payload: dict[str, Any]) -> int:
    return _response_int(payload.get("photos", {}).get("page"), key="page")


def _payload_perpage(payload: dict[str, Any]) -> int:
    return _response_int(payload.get("photos", {}).get("perpage"), key="perpage")


def _response_int(value: object, *, key: str) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise FlickrFetchError(f"Flickr response photos.{key} must be an integer") from exc


def _insert_work_item(conn: sqlite3.Connection, query: FlickrQuery) -> sqlite3.Cursor:
    work_item_id = _work_item_id(query)
    return conn.execute(
        """
        INSERT OR IGNORE INTO flickr_work_items (
            work_item_id, status, query_json, lane, page, per_page,
            split_depth, split_priority, split_reason, parent_query_hash,
            parent_total, date_kind, min_date, max_date, bbox_index,
            slice_index, bbox_label, term, query_hash, claimed_at,
            completed_at, error, records_returned, response_total,
            response_pages, response_page, response_perpage, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?)
        """,
        (
            work_item_id,
            PENDING,
            json.dumps(asdict(query), sort_keys=True, ensure_ascii=False),
            query.lane,
            query.page,
            query.per_page,
            query.split_depth,
            split_priority(query),
            query.split_reason,
            query.parent_query_hash,
            query.parent_total,
            query_date_kind(query),
            query_min_date(query),
            query_max_date(query),
            query.bbox_index,
            query.slice_index,
            query.region or query.bbox,
            query.term,
            query_hash(query),
            _timestamp(),
        ),
    )


def _pagination_rows(conn: sqlite3.Connection, query: FlickrQuery) -> list[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT page, query_json, response_pages
        FROM flickr_work_items
        WHERE lane = ?
          AND term = ?
          AND COALESCE(date_kind, '') = ?
          AND COALESCE(min_date, '') = ?
          AND COALESCE(max_date, '') = ?
          AND COALESCE(bbox_label, '') = ?
        """,
        (
            query.lane,
            query.term,
            query_date_kind(query),
            query_min_date(query),
            query_max_date(query),
            query.region or query.bbox or "",
        ),
    ).fetchall()
    identity = _pagination_identity(query)
    return [row for row in rows if _pagination_identity(_query_from_json(str(row["query_json"]))) == identity]


def _pagination_identity(query: FlickrQuery) -> dict[str, Any]:
    payload = asdict(query)
    payload.pop("page", None)
    payload.pop("per_page", None)
    return payload


def _payload_photo_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    photos = payload.get("photos", {})
    rows = photos.get("photo", []) if isinstance(photos, dict) else []
    return [row for row in rows if isinstance(row, dict)]


def _work_item_id(query: FlickrQuery) -> str:
    payload = json.dumps(asdict(query), sort_keys=True, ensure_ascii=False)
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_record_hash(record: dict[str, Any]) -> str:
    import hashlib

    payload = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _query_from_json(payload: str) -> FlickrQuery:
    data = json.loads(payload)
    return FlickrQuery(**data)


def _timestamp(value: datetime | None = None) -> str:
    return (value or datetime.now(UTC)).isoformat()


def _unix_timestamp(value: datetime | None = None) -> float:
    return (value or datetime.now(UTC)).timestamp()
