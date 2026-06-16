from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
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

FetchMetadata = Callable[[FlickrQuery], dict[str, Any]]
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class PollOnceResult:
    state_db: Path
    raw_responses_written: int
    evidence_rows_written: int
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
        work_item_id = _work_item_id(query)
        with self._connect() as conn:
            result = conn.execute(
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
        return int(result.rowcount)

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
        cutoff = _unix_timestamp(now) - 3600
        with self._connect() as conn:
            return int(conn.execute("SELECT count(*) FROM api_call_ledger WHERE created_at >= ?", (cutoff,)).fetchone()[0])

    def remaining_api_budget(self, *, max_api_calls: int, now: datetime | None = None) -> tuple[int, int]:
        used = self.api_calls_in_window(now=now)
        soft_limit = min(max_api_calls, SOFT_API_CALLS_PER_HOUR)
        return max(0, soft_limit - used), max(0, HARD_API_CALLS_PER_HOUR - used)

    def claim_pending(self, *, limit: int) -> list[tuple[str, FlickrQuery]]:
        if limit <= 0:
            return []
        claimed: list[tuple[str, FlickrQuery]] = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
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
                (PENDING, limit),
            ).fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE flickr_work_items SET status = ?, claimed_at = ? WHERE work_item_id = ?",
                    (CLAIMED, _timestamp(), row["work_item_id"]),
                )
                claimed.append((str(row["work_item_id"]), _query_from_json(str(row["query_json"]))))
            conn.execute("COMMIT")
        return claimed

    def log_api_call(self, *, work_item_id: str, endpoint: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO api_call_ledger(endpoint, work_item_id, status, created_at) VALUES (?, ?, ?, ?)",
                (endpoint, work_item_id, status, _unix_timestamp()),
            )

    def reserve_api_call(self, *, work_item_id: str, endpoint: str) -> None:
        self.log_api_call(work_item_id=work_item_id, endpoint=endpoint, status="reserved")

    def update_api_call_status(self, *, work_item_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE api_call_ledger
                SET status = ?
                WHERE id = (
                    SELECT id
                    FROM api_call_ledger
                    WHERE work_item_id = ? AND status = 'reserved'
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                )
                """,
                (status, work_item_id),
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
                        image_url,
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
                if result.rowcount:
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
                    created_at REAL NOT NULL
                )
                """
            )
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
                    PRIMARY KEY (source, flickr_photo_id, image_url)
                )
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
    claim_limit = min(soft_remaining, hard_remaining)
    claimed = state.claim_pending(limit=claim_limit)
    _progress(progress_callback, {"event": "work_claimed", "claimed": len(claimed), "claim_limit": claim_limit})
    raw_written = 0
    records_inserted = 0
    duplicates = 0
    query_hits_inserted = 0
    duplicate_query_hits = 0
    queued = 0
    payloads: list[dict[str, Any]] = []
    fetcher = fetch_metadata or _http_fetcher(api_key=api_key)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for work_item_id, _query in claimed:
            state.reserve_api_call(work_item_id=work_item_id, endpoint=SEARCH_METHOD)
        pending: dict[Future[dict[str, Any]], tuple[str, FlickrQuery]] = {
            pool.submit(fetcher, query): (work_item_id, query)
            for work_item_id, query in claimed
        }
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                work_item_id, query = pending.pop(future)
                try:
                    payload = future.result()
                    state.update_api_call_status(work_item_id=work_item_id, status="ok")
                    raw_written += 1
                    payloads.append(payload)
                    _write_raw_response(raw_root=Path(raw_root), work_item_id=work_item_id, query=query, payload=payload)
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
                        remaining_queries = _remaining_page_queries(query, pages=response_pages)
                        remaining_inserted = 0
                        for next_query in remaining_queries:
                            remaining_inserted += state.enqueue_work_item(next_query)
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
                                "remaining_pages_enqueued": remaining_inserted,
                            },
                        )
                        if remaining_queries:
                            _progress(
                                progress_callback,
                                {
                                    "event": "remaining_pages_enqueued",
                                    "enqueued": remaining_inserted,
                                    "pages": [item.page for item in remaining_queries],
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
                except Exception as exc:  # noqa: BLE001 - poller records failure and exits bounded cycle.
                    state.update_api_call_status(work_item_id=work_item_id, status="failed")
                    state.fail_work_item(work_item_id, str(exc))
                    _progress(
                        progress_callback,
                        {
                            "event": "work_failed",
                            "work_item_id": work_item_id,
                            "lane": query.lane,
                            "page": query.page,
                            "error": str(exc),
                        },
                    )

    evidence_rows = _write_evidence(evidence_output, payloads)
    soft_after, hard_after = state.remaining_api_budget(max_api_calls=max_api_calls)
    result = PollOnceResult(
        state_db=Path(state_db),
        raw_responses_written=raw_written,
        evidence_rows_written=evidence_rows,
        source_records_inserted=records_inserted,
        duplicate_records_skipped=duplicates,
        query_hits_inserted=query_hits_inserted,
        duplicate_query_hits_skipped=duplicate_query_hits,
        image_urls_queued=queued,
        work_items_claimed=len(claimed),
        api_calls_made=len(claimed),
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


def _write_raw_response(*, raw_root: Path, work_item_id: str, query: FlickrQuery, payload: dict[str, Any]) -> Path:
    target_dir = raw_root / "flickr" / "photos_search" / query.search_field / _safe_query_variant(query.term)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{query.lane}-{query.page:05d}-{work_item_id[:12]}.json"
    target.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return target


def _safe_query_variant(term: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in term.casefold()).strip("_")


def _write_evidence(evidence_output: str | Path, payloads: list[dict[str, Any]]) -> int:
    if not payloads:
        return 0
    frame = build_evidence_frame(payloads, species_query="multilingual_lepidoptera")
    output = Path(evidence_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = build_evidence_frame([], species_query="multilingual_lepidoptera")
        try:
            import polars as pl

            existing = pl.read_parquet(output)
            frame = pl.concat([existing, frame], how="diagonal_relaxed")
        except Exception:
            pass
    write_parquet(frame, output)
    return frame.height


def _payload_total(payload: dict[str, Any]) -> int:
    return int(payload.get("photos", {}).get("total") or 0)


def _payload_pages(payload: dict[str, Any]) -> int:
    return int(payload.get("photos", {}).get("pages") or 0)


def _payload_page(payload: dict[str, Any]) -> int:
    return int(payload.get("photos", {}).get("page") or 0)


def _payload_perpage(payload: dict[str, Any]) -> int:
    return int(payload.get("photos", {}).get("perpage") or 0)


def _remaining_page_queries(query: FlickrQuery, *, pages: int) -> tuple[FlickrQuery, ...]:
    if query.lane != "normal_page" or query.page != 1:
        return ()
    if query.split_reason != "upload_date" or not query.min_upload_date or not query.max_upload_date:
        return ()
    last_page = min(max(0, pages), 8)
    if last_page <= 1:
        return ()
    return tuple(replace(query, page=page) for page in range(2, last_page + 1))


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


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _unix_timestamp(value: datetime | None = None) -> float:
    return (value or datetime.now(UTC)).timestamp()
