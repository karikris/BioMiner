from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable

import httpx

from biominer.filter.extractor import build_evidence_frame
from biominer.flickr_fetch.endpoints import FLICKR_REST_BASE_URL, SEARCH_METHOD
from biominer.flickr_fetch.query_planner import (
    FlickrQuery,
    build_count_probes,
    deduplicate_photo_records,
    flickr_search_params,
    plan_queries_from_count,
)


SOFT_API_CALLS_PER_HOUR = 3450
HARD_API_CALLS_PER_HOUR = 3600
PENDING = "pending"
CLAIMED = "claimed"
COMPLETED = "completed"
FAILED = "failed"

FetchMetadata = Callable[[FlickrQuery], dict[str, Any]]


@dataclass(frozen=True)
class PollOnceResult:
    state_db: Path
    raw_responses_written: int
    evidence_rows_written: int
    source_records_inserted: int
    duplicate_records_skipped: int
    image_urls_queued: int
    work_items_claimed: int
    api_calls_made: int
    remaining_soft_budget: int
    remaining_hard_budget: int


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
                    claimed_at, completed_at, error, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?)
                """,
                (
                    work_item_id,
                    PENDING,
                    json.dumps(asdict(query), sort_keys=True, ensure_ascii=False),
                    query.lane,
                    query.page,
                    query.per_page,
                    _timestamp(),
                ),
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
                ORDER BY created_at, work_item_id
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

    def complete_work_item(self, work_item_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE flickr_work_items SET status = ?, completed_at = ?, error = NULL WHERE work_item_id = ?",
                (COMPLETED, _timestamp(), work_item_id),
            )

    def fail_work_item(self, work_item_id: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE flickr_work_items SET status = ?, completed_at = ?, error = ? WHERE work_item_id = ?",
                (FAILED, _timestamp(), error, work_item_id),
            )

    def insert_source_records(self, records: list[dict[str, Any]], *, source_query: FlickrQuery) -> tuple[int, int, int]:
        inserted = 0
        unique_records = deduplicate_photo_records(records)
        skipped = len(records) - len(unique_records)
        queued = 0
        with self._connect() as conn:
            for record in unique_records:
                photo_id = str(record.get("id") or "")
                image_url = str(record.get("url_l") or record.get("url_m") or "")
                if not photo_id or not image_url:
                    skipped += 1
                    continue
                source_record_hash = _source_record_hash(record)
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
                        "url_l" if record.get("url_l") else "url_m",
                        source_record_hash,
                        source_query.term,
                        source_query.language,
                        source_query.search_field,
                        json.dumps(record, sort_keys=True, ensure_ascii=False),
                        _timestamp(),
                    ),
                )
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
        return inserted, skipped, queued

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
                    claimed_at TEXT,
                    completed_at TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
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
) -> PollOnceResult:
    state = MetadataPollState(state_db)
    state.ensure_seed_work_items()
    soft_remaining, hard_remaining = state.remaining_api_budget(max_api_calls=max_api_calls)
    claim_limit = min(soft_remaining, hard_remaining)
    claimed = state.claim_pending(limit=claim_limit)
    raw_written = 0
    records_inserted = 0
    duplicates = 0
    queued = 0
    payloads: list[dict[str, Any]] = []
    fetcher = fetch_metadata or _http_fetcher(api_key=api_key)

    for work_item_id, query in claimed:
        try:
            payload = fetcher(query)
            state.log_api_call(work_item_id=work_item_id, endpoint=SEARCH_METHOD, status="ok")
            raw_written += 1
            payloads.append(payload)
            _write_raw_response(raw_root=Path(raw_root), work_item_id=work_item_id, query=query, payload=payload)
            total = _payload_total(payload)
            if query.lane == "count_probe":
                for next_query in plan_queries_from_count(query, total=total):
                    state.enqueue_work_item(next_query)
            else:
                records = _payload_photo_records(payload)
                inserted, skipped, queued_count = state.insert_source_records(records, source_query=query)
                records_inserted += inserted
                duplicates += skipped
                queued += queued_count
            state.complete_work_item(work_item_id)
        except Exception as exc:  # noqa: BLE001 - poller records failure and exits bounded cycle.
            state.log_api_call(work_item_id=work_item_id, endpoint=SEARCH_METHOD, status="failed")
            state.fail_work_item(work_item_id, str(exc))

    evidence_rows = _write_evidence(evidence_output, payloads)
    soft_after, hard_after = state.remaining_api_budget(max_api_calls=max_api_calls)
    return PollOnceResult(
        state_db=Path(state_db),
        raw_responses_written=raw_written,
        evidence_rows_written=evidence_rows,
        source_records_inserted=records_inserted,
        duplicate_records_skipped=duplicates,
        image_urls_queued=queued,
        work_items_claimed=len(claimed),
        api_calls_made=len(claimed),
        remaining_soft_budget=soft_after,
        remaining_hard_budget=hard_after,
    )


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
    frame.write_parquet(output)
    return frame.height


def _payload_total(payload: dict[str, Any]) -> int:
    return int(payload.get("photos", {}).get("total") or 0)


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
