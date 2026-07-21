from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable

import httpx

from biominer.common.status import CLAIMED, COMPLETED, FAILED, PENDING
from biominer.flickr_fetch.endpoints import FLICKR_REST_BASE_URL
from biominer.flickr_fetch.metadata_poller import (
    HARD_API_CALLS_PER_HOUR,
    SOFT_API_CALLS_PER_HOUR,
    FlickrFetchError,
    MetadataPollState,
)
from biominer.storage.sqlite_connection import connect_closing


GET_INFO_METHOD = "flickr.photos.getInfo"
DEFAULT_STALE_CLAIM_SECONDS = 3600

FetchInfo = Callable[[str], dict[str, Any]]
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class EnrichmentWorkItem:
    source: str
    flickr_photo_id: str
    endpoint: str
    attempt_count: int

    @property
    def work_item_id(self) -> str:
        return f"enrichment:{self.endpoint}:{self.source}:{self.flickr_photo_id}"


@dataclass(frozen=True)
class FlickrEnrichmentResult:
    state_db: Path
    endpoint: str
    source_records_seen: int
    work_items_enqueued: int
    stale_claims_requeued: int
    work_items_claimed: int
    records_enriched: int
    records_deferred: int
    records_failed: int
    api_calls_made: int
    pending_records: int
    completed_records: int
    failed_records: int
    remaining_soft_budget: int
    remaining_hard_budget: int


class EnrichmentFetchFailure(RuntimeError):
    def __init__(self, error: FlickrFetchError, *, attempts: int) -> None:
        super().__init__(str(error))
        self.error = error
        self.attempts = attempts


class FlickrEnrichmentState:
    """Durable Flickr endpoint work sharing the discovery API-call ledger."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        MetadataPollState(self.path)
        self._init_db()

    def enqueue_source_records(self, *, endpoint: str = GET_INFO_METHOD) -> tuple[int, int]:
        now = time.time()
        with self._connect() as conn:
            seen = int(
                conn.execute(
                    "SELECT count(*) FROM source_records WHERE source = 'flickr'"
                ).fetchone()[0]
            )
            result = conn.execute(
                """
                INSERT OR IGNORE INTO flickr_enrichment_work_items (
                    source, flickr_photo_id, endpoint, status, attempt_count,
                    next_attempt_at, created_at, updated_at
                )
                SELECT source, flickr_photo_id, ?, ?, 0, 0, ?, ?
                FROM source_records
                WHERE source = 'flickr'
                """,
                (endpoint, PENDING, now, now),
            )
        return seen, int(result.rowcount)

    def requeue_stale_claims(
        self,
        *,
        stale_after_seconds: int = DEFAULT_STALE_CLAIM_SECONDS,
        now: datetime | None = None,
    ) -> int:
        timestamp = _unix_timestamp(now)
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE flickr_enrichment_work_items
                SET status = ?, claimed_at = NULL, updated_at = ?
                WHERE status = ? AND claimed_at IS NOT NULL AND claimed_at < ?
                """,
                (PENDING, timestamp, CLAIMED, timestamp - stale_after_seconds),
            )
        return int(result.rowcount)

    def claim_and_reserve(
        self,
        *,
        limit: int,
        max_api_calls: int,
        endpoint: str = GET_INFO_METHOD,
        now: datetime | None = None,
    ) -> list[EnrichmentWorkItem]:
        if limit <= 0:
            return []
        timestamp = _unix_timestamp(now)
        claimed: list[EnrichmentWorkItem] = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            used = _api_calls_in_window(conn, timestamp)
            soft_limit = min(max_api_calls, SOFT_API_CALLS_PER_HOUR)
            claim_limit = min(
                limit,
                max(0, soft_limit - used),
                max(0, HARD_API_CALLS_PER_HOUR - used),
            )
            rows = conn.execute(
                """
                SELECT source, flickr_photo_id, endpoint, attempt_count
                FROM flickr_enrichment_work_items
                WHERE status = ? AND endpoint = ? AND next_attempt_at <= ?
                ORDER BY next_attempt_at, created_at, flickr_photo_id
                LIMIT ?
                """,
                (PENDING, endpoint, timestamp, claim_limit),
            ).fetchall()
            for row in rows:
                item = EnrichmentWorkItem(
                    source=str(row["source"]),
                    flickr_photo_id=str(row["flickr_photo_id"]),
                    endpoint=str(row["endpoint"]),
                    attempt_count=int(row["attempt_count"]),
                )
                conn.execute(
                    """
                    UPDATE flickr_enrichment_work_items
                    SET status = ?, claimed_at = ?, updated_at = ?
                    WHERE source = ? AND flickr_photo_id = ? AND endpoint = ?
                    """,
                    (
                        CLAIMED,
                        timestamp,
                        timestamp,
                        item.source,
                        item.flickr_photo_id,
                        item.endpoint,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO api_call_ledger (
                        endpoint, work_item_id, status, created_at, started_at
                    ) VALUES (?, ?, 'reserved', ?, ?)
                    """,
                    (item.endpoint, item.work_item_id, timestamp, timestamp),
                )
                claimed.append(item)
            conn.execute("COMMIT")
        return claimed

    def complete(
        self,
        item: EnrichmentWorkItem,
        *,
        payload: dict[str, Any],
        attempts: int,
    ) -> None:
        payload_json = _canonical_json(payload)
        timestamp = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO flickr_enrichment_results (
                    source, flickr_photo_id, endpoint, payload_json,
                    payload_sha256, retrieved_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, flickr_photo_id, endpoint) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    payload_sha256 = excluded.payload_sha256,
                    retrieved_at = excluded.retrieved_at
                """,
                (
                    item.source,
                    item.flickr_photo_id,
                    item.endpoint,
                    payload_json,
                    hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
                    timestamp,
                ),
            )
            conn.execute(
                """
                UPDATE flickr_enrichment_work_items
                SET status = ?, attempt_count = attempt_count + ?,
                    claimed_at = NULL, completed_at = ?, error = NULL,
                    next_attempt_at = 0, updated_at = ?
                WHERE source = ? AND flickr_photo_id = ? AND endpoint = ?
                """,
                (
                    COMPLETED,
                    attempts,
                    timestamp,
                    timestamp,
                    item.source,
                    item.flickr_photo_id,
                    item.endpoint,
                ),
            )
            conn.execute("COMMIT")

    def defer(
        self,
        item: EnrichmentWorkItem,
        *,
        error: str,
        attempts: int,
        retry_backoff_seconds: float,
    ) -> None:
        attempt_count = item.attempt_count + attempts
        delay = retry_backoff_seconds * (2 ** min(attempt_count - 1, 10))
        timestamp = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE flickr_enrichment_work_items
                SET status = ?, attempt_count = ?, claimed_at = NULL,
                    next_attempt_at = ?, error = ?, updated_at = ?
                WHERE source = ? AND flickr_photo_id = ? AND endpoint = ?
                """,
                (
                    PENDING,
                    attempt_count,
                    timestamp + min(3600.0, max(retry_backoff_seconds, delay)),
                    error,
                    timestamp,
                    item.source,
                    item.flickr_photo_id,
                    item.endpoint,
                ),
            )

    def fail(
        self,
        item: EnrichmentWorkItem,
        *,
        error: str,
        attempts: int,
    ) -> None:
        timestamp = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE flickr_enrichment_work_items
                SET status = ?, attempt_count = attempt_count + ?,
                    claimed_at = NULL, completed_at = ?, error = ?, updated_at = ?
                WHERE source = ? AND flickr_photo_id = ? AND endpoint = ?
                """,
                (
                    FAILED,
                    attempts,
                    timestamp,
                    error,
                    timestamp,
                    item.source,
                    item.flickr_photo_id,
                    item.endpoint,
                ),
            )

    def summary(self, *, endpoint: str = GET_INFO_METHOD) -> dict[str, int]:
        with self._connect() as conn:
            counts = {
                str(row["status"]): int(row["row_count"])
                for row in conn.execute(
                    """
                    SELECT status, count(*) AS row_count
                    FROM flickr_enrichment_work_items
                    WHERE endpoint = ?
                    GROUP BY status
                    """,
                    (endpoint,),
                ).fetchall()
            }
        return {
            "pending": counts.get(PENDING, 0),
            "claimed": counts.get(CLAIMED, 0),
            "completed": counts.get(COMPLETED, 0),
            "failed": counts.get(FAILED, 0),
        }

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS flickr_enrichment_work_items (
                    source TEXT NOT NULL,
                    flickr_photo_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL DEFAULT 0,
                    claimed_at REAL,
                    completed_at REAL,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(source, flickr_photo_id, endpoint)
                );
                CREATE INDEX IF NOT EXISTS flickr_enrichment_work_claim_idx
                    ON flickr_enrichment_work_items(
                        endpoint, status, next_attempt_at, created_at, flickr_photo_id
                    );
                CREATE TABLE IF NOT EXISTS flickr_enrichment_results (
                    source TEXT NOT NULL,
                    flickr_photo_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    retrieved_at REAL NOT NULL,
                    PRIMARY KEY(source, flickr_photo_id, endpoint)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = connect_closing(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn


def enrich_once(
    *,
    state_db: str | Path,
    api_key: str | None = None,
    fetch_info: FetchInfo | None = None,
    max_api_calls: int = SOFT_API_CALLS_PER_HOUR,
    workers: int = 1,
    max_retries: int = 2,
    retry_backoff_seconds: float = 2.0,
    min_call_interval_seconds: float = 0.0,
    stale_claim_seconds: int = DEFAULT_STALE_CLAIM_SECONDS,
    progress_callback: ProgressCallback | None = None,
) -> FlickrEnrichmentResult:
    state = FlickrEnrichmentState(state_db)
    source_records_seen, enqueued = state.enqueue_source_records()
    stale_requeued = state.requeue_stale_claims(
        stale_after_seconds=stale_claim_seconds
    )
    poll_state = MetadataPollState(state_db)
    fetcher = fetch_info or _http_info_fetcher(api_key=api_key)
    claimed_total = 0
    completed = 0
    deferred = 0
    failed = 0
    api_calls = 0
    last_dispatch_at = 0.0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        while True:
            soft_remaining, hard_remaining = poll_state.remaining_api_budget(
                max_api_calls=max_api_calls
            )
            claim_limit = min(soft_remaining, hard_remaining, max(1, workers))
            claimed = state.claim_and_reserve(
                limit=claim_limit,
                max_api_calls=max_api_calls,
            )
            if not claimed:
                break
            claimed_total += len(claimed)
            futures: dict[Future[tuple[dict[str, Any], int]], EnrichmentWorkItem] = {}
            for item in claimed:
                if min_call_interval_seconds > 0:
                    delay = min_call_interval_seconds - (
                        time.monotonic() - last_dispatch_at
                    )
                    if delay > 0:
                        time.sleep(delay)
                futures[
                    pool.submit(
                        _fetch_with_retries,
                        poll_state=poll_state,
                        item=item,
                        fetcher=fetcher,
                        max_api_calls=max_api_calls,
                        max_retries=max_retries,
                        retry_backoff_seconds=retry_backoff_seconds,
                    )
                ] = item
                last_dispatch_at = time.monotonic()
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    item = futures.pop(future)
                    try:
                        payload, attempts = future.result()
                    except EnrichmentFetchFailure as exc:
                        api_calls += exc.attempts
                        if exc.error.retryable:
                            state.defer(
                                item,
                                error=str(exc.error),
                                attempts=exc.attempts,
                                retry_backoff_seconds=retry_backoff_seconds,
                            )
                            deferred += 1
                        else:
                            state.fail(
                                item,
                                error=str(exc.error),
                                attempts=exc.attempts,
                            )
                            failed += 1
                    else:
                        api_calls += attempts
                        state.complete(item, payload=payload, attempts=attempts)
                        completed += 1
                    _progress(
                        progress_callback,
                        {
                            "event": "flickr_enrichment_progress",
                            "claimed": claimed_total,
                            "completed": completed,
                            "deferred": deferred,
                            "failed": failed,
                            "api_calls_made": api_calls,
                        },
                    )

    summary = state.summary()
    soft_remaining, hard_remaining = poll_state.remaining_api_budget(
        max_api_calls=max_api_calls
    )
    return FlickrEnrichmentResult(
        state_db=Path(state_db),
        endpoint=GET_INFO_METHOD,
        source_records_seen=source_records_seen,
        work_items_enqueued=enqueued,
        stale_claims_requeued=stale_requeued,
        work_items_claimed=claimed_total,
        records_enriched=completed,
        records_deferred=deferred,
        records_failed=failed,
        api_calls_made=api_calls,
        pending_records=summary["pending"] + summary["claimed"],
        completed_records=summary["completed"],
        failed_records=summary["failed"],
        remaining_soft_budget=soft_remaining,
        remaining_hard_budget=hard_remaining,
    )


def result_payload(result: FlickrEnrichmentResult) -> dict[str, Any]:
    return {**asdict(result), "state_db": str(result.state_db)}


def _http_info_fetcher(*, api_key: str | None) -> FetchInfo:
    if not api_key:
        raise RuntimeError("Flickr API key is required for enrichment")

    def fetch(photo_id: str) -> dict[str, Any]:
        params = {
            "method": GET_INFO_METHOD,
            "api_key": api_key,
            "photo_id": photo_id,
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
    poll_state: MetadataPollState,
    item: EnrichmentWorkItem,
    fetcher: FetchInfo,
    max_api_calls: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> tuple[dict[str, Any], int]:
    attempts = 0
    while True:
        attempts += 1
        started = time.perf_counter()
        try:
            payload = fetcher(item.flickr_photo_id)
            _validate_info_payload(payload, expected_photo_id=item.flickr_photo_id)
            poll_state.update_api_call_status(
                work_item_id=item.work_item_id,
                status="ok",
                duration_sec=time.perf_counter() - started,
            )
            return payload, attempts
        except Exception as exc:  # noqa: BLE001 - HTTP failures need explicit classification.
            error = _classify_fetch_error(exc)
            poll_state.update_api_call_status(
                work_item_id=item.work_item_id,
                status="failed",
                duration_sec=time.perf_counter() - started,
                http_status=error.http_status,
            )
            if not error.retryable or attempts > max_retries:
                raise EnrichmentFetchFailure(error, attempts=attempts) from exc
            if retry_backoff_seconds > 0:
                time.sleep(retry_backoff_seconds * (2 ** (attempts - 1)))
            try:
                poll_state.reserve_retry_api_call(
                    work_item_id=item.work_item_id,
                    endpoint=item.endpoint,
                    max_api_calls=max_api_calls,
                )
            except FlickrFetchError as budget_error:
                raise EnrichmentFetchFailure(
                    FlickrFetchError(str(budget_error), retryable=True),
                    attempts=attempts,
                ) from exc


def _validate_info_payload(
    payload: dict[str, Any],
    *,
    expected_photo_id: str,
) -> None:
    if not isinstance(payload, dict):
        raise FlickrFetchError("Flickr response must be a JSON object")
    if payload.get("stat") == "fail":
        code = payload.get("code")
        message = payload.get("message") or "Flickr API error"
        raise FlickrFetchError(f"Flickr API error {code}: {message}")
    photo = payload.get("photo")
    if not isinstance(photo, dict):
        raise FlickrFetchError("Flickr getInfo response missing photo object")
    photo_id = str(photo.get("id") or "")
    if photo_id != expected_photo_id:
        raise FlickrFetchError(
            f"Flickr getInfo response photo id {photo_id!r} does not match {expected_photo_id!r}"
        )


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
        return FlickrFetchError(
            str(exc) or exc.__class__.__name__,
            retryable=True,
        )
    return FlickrFetchError(str(exc) or exc.__class__.__name__, retryable=False)


def _api_calls_in_window(conn: sqlite3.Connection, now: float) -> int:
    return int(
        conn.execute(
            "SELECT count(*) FROM api_call_ledger WHERE created_at >= ?",
            (now - 3600,),
        ).fetchone()[0]
    )


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _unix_timestamp(value: datetime | None = None) -> float:
    return (value or datetime.now(UTC)).timestamp()


def _progress(callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    if callback:
        callback(event)
