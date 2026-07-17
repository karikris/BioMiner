from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable

import httpx
import polars as pl

from biominer.common.status import CLAIMED, COMPLETED, FAILED, PENDING
from biominer.filter.extractor import build_evidence_frame, extract_photo_evidence
from biominer.flickr_fetch.endpoints import FLICKR_REST_BASE_URL, SEARCH_METHOD
from biominer.flickr_fetch.query_planner import (
    FLICKR_SEARCH_RESULT_WINDOW,
    FlickrQuery,
    deduplicate_photo_records,
    flickr_search_params,
    plan_queries_from_count,
    query_date_kind,
    query_hash,
    query_max_date,
    query_min_date,
    split_priority,
)
from biominer.registry.normalize import normalize_name_key
from biominer.registry.unified import stable_identity
from biominer.storage.parquet import write_parquet
from biominer.storage.cloud import CloudStorage
from biominer.storage.sqlite_connection import connect_closing
from biominer.config import StorageConfig, create_storage_backend, load_storage_config_from_env
from biominer.storage.local import LocalStorageBackend
from biominer.storage.paths import (
    build_evidence_shard_uri,
    build_raw_flickr_response_uri,
)
from biominer.storage.uri import is_cloud_uri, normalize_local_uri
from biominer.workstore.base import WorkStore


SOFT_API_CALLS_PER_HOUR = 3500
HARD_API_CALLS_PER_HOUR = 3600
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
    reported_pages: int
    target_pages: int
    accessible_pages: int
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

    def enqueue_initial_work_items(self, queries: tuple[FlickrQuery, ...]) -> int:
        inserted = 0
        for query in queries:
            inserted += self.enqueue_work_item(query)
        return inserted

    def register_registry(self, registry_dir: str | Path) -> dict[str, int]:
        registry = Path(registry_dir)
        names = pl.read_parquet(registry / "names.parquet")
        definitions = pl.read_parquet(registry / "flickr_query_definitions.parquet")
        return self.register_query_definitions(definitions, keyword_associations=names)

    def register_query_definitions(
        self,
        definitions: pl.DataFrame,
        *,
        keyword_associations: pl.DataFrame | None = None,
    ) -> dict[str, int]:
        """Upsert registry associations without making Flickr request identity versioned."""

        canonical_inserted = 0
        associations_inserted = 0
        logical_inserted = 0
        backfilled = 0
        association_frame = keyword_associations if keyword_associations is not None else definitions
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for row in association_frame.iter_rows(named=True):
                normalized = str(row.get("normalized_match_key") or row.get("normalized_query_term") or "")
                if not normalized:
                    continue
                canonical_id = str(row.get("canonical_keyword_id") or "") or stable_identity("canonical-keyword", normalized)
                keyword_id = str(row.get("keyword_id") or row.get("name_id") or "") or stable_identity(
                    "keyword-association",
                    row.get("accepted_taxon_key"),
                    normalized,
                    row.get("source"),
                    row.get("source_record_id"),
                )
                effective_tier = str(row.get("effective_trust_tier") or row.get("trust_tier") or "T4")
                original_tier = str(row.get("original_trust_tier") or row.get("trust_tier") or effective_tier)
                now = _timestamp()
                result = conn.execute(
                    """
                    INSERT OR IGNORE INTO canonical_keywords (
                        canonical_keyword_id, normalized_term, canonical_keyword_id_source,
                        canonical_term, effective_trust_tier, first_seen_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        canonical_id,
                        normalized,
                        keyword_id,
                        str(row.get("display_name") or row.get("source_term") or normalized),
                        effective_tier,
                        now,
                        now,
                    ),
                )
                canonical_inserted += int(result.rowcount)
                conn.execute(
                    """
                    UPDATE canonical_keywords
                    SET effective_trust_tier = CASE
                            WHEN CAST(substr(effective_trust_tier, 2) AS INTEGER) <= CAST(substr(?, 2) AS INTEGER)
                            THEN effective_trust_tier ELSE ? END,
                        canonical_keyword_id_source = CASE WHEN ? THEN ? ELSE canonical_keyword_id_source END,
                        canonical_term = CASE WHEN ? THEN ? ELSE canonical_term END,
                        last_seen_at = ?
                    WHERE canonical_keyword_id = ?
                    """,
                    (
                        effective_tier,
                        effective_tier,
                        bool(row.get("is_canonical_keyword", False)),
                        keyword_id,
                        bool(row.get("is_canonical_keyword", False)),
                        str(row.get("display_name") or row.get("source_term") or normalized),
                        now,
                        canonical_id,
                    ),
                )
                result = conn.execute(
                    """
                    INSERT OR IGNORE INTO keyword_associations (
                        keyword_id, canonical_keyword_id, query_definition_id,
                        accepted_taxon_id, original_trust_tier, source, name_class,
                        language, registry_version, first_seen_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        keyword_id,
                        canonical_id,
                        str(row.get("query_definition_id") or ""),
                        str(row.get("accepted_taxon_key") or row.get("accepted_taxon_id") or ""),
                        original_tier,
                        str(row.get("source") or ""),
                        str(row.get("name_class") or ""),
                        str(row.get("language") or ""),
                        str(row.get("registry_version") or ""),
                        now,
                        now,
                    ),
                )
                associations_inserted += int(result.rowcount)
                if result.rowcount:
                    backfilled += _backfill_keyword_evidence(conn, keyword_id=keyword_id, canonical_keyword_id=canonical_id)
            for row in definitions.iter_rows(named=True):
                normalized = str(row.get("normalized_match_key") or row.get("normalized_query_term") or "")
                field = str(row.get("search_field") or "")
                if not normalized or field not in {"tags", "text"}:
                    continue
                canonical_id = str(row.get("canonical_keyword_id") or "") or stable_identity("canonical-keyword", normalized)
                logical_id = str(row.get("logical_query_id") or row.get("query_definition_id") or "") or stable_identity("flickr-logical-query", normalized, field)
                tier = str(row.get("effective_trust_tier") or row.get("trust_tier") or "T4")
                result = conn.execute(
                    """
                    INSERT OR IGNORE INTO flickr_logical_queries (
                        logical_query_id, canonical_keyword_id, search_field,
                        effective_trust_tier, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'active', ?, ?)
                    """,
                    (logical_id, canonical_id, field, tier, _timestamp(), _timestamp()),
                )
                logical_inserted += int(result.rowcount)
                conn.execute(
                    """
                    UPDATE flickr_logical_queries
                    SET effective_trust_tier = ?, updated_at = ?
                    WHERE logical_query_id = ?
                    """,
                    (tier, _timestamp(), logical_id),
                )
            conn.execute("COMMIT")
        return {
            "canonical_keywords_inserted": canonical_inserted,
            "keyword_associations_inserted": associations_inserted,
            "logical_queries_inserted": logical_inserted,
            "existing_results_backfilled": backfilled,
        }

    def work_item_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT count(*) FROM flickr_work_items").fetchone()[0])

    def enqueue_work_item(self, query: FlickrQuery) -> int:
        with self._connect() as conn:
            result = _insert_work_item(conn, query)
        return int(result.rowcount)

    def ensure_reported_pages(
        self,
        query: FlickrQuery,
        *,
        response_pages: int,
        response_perpage: int | None = None,
    ) -> PageEnsureResult:
        if query.lane == "count_probe" or response_pages <= 0:
            return PageEnsureResult(
                reported_pages=response_pages,
                target_pages=0,
                accessible_pages=0,
                new_pages_enqueued=0,
                total_known_work_items=0,
                highest_known_page=0,
                missing_pages=(),
                warnings=(),
            )
        effective_perpage = response_perpage or query.per_page
        if effective_perpage <= 0:
            effective_perpage = query.per_page
        accessible_pages = _accessible_page_window(effective_perpage)
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
            reported_target_pages = max(response_pages, previous_reported_page_count)
            target_pages = min(reported_target_pages, accessible_pages)
            if reported_target_pages > accessible_pages:
                warnings.append(
                    {
                        "event": "pagination_over_accessible_window",
                        "level": "warning",
                        "response_pages": reported_target_pages,
                        "accessible_pages": accessible_pages,
                        "response_perpage": effective_perpage,
                        "min_date": query_min_date(query),
                        "max_date": query_max_date(query),
                    }
                )
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
            if target_pages > highest_after:
                warnings.append(
                    {
                        "event": "pagination_invariant_failed",
                        "level": "error",
                        "response_pages": reported_target_pages,
                        "target_pages": target_pages,
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
                        "response_pages": reported_target_pages,
                        "target_pages": target_pages,
                        "missing_pages": list(missing_pages),
                        "min_date": query_min_date(query),
                        "max_date": query_max_date(query),
                    }
                )
            conn.execute("COMMIT")
        return PageEnsureResult(
            reported_pages=reported_target_pages,
            target_pages=target_pages,
            accessible_pages=accessible_pages,
            new_pages_enqueued=inserted,
            total_known_work_items=len(after),
            highest_known_page=highest_after,
            missing_pages=missing_pages,
            warnings=tuple(warnings),
        )

    def requeue_stale_claims(
        self,
        *,
        stale_after_seconds: int = DEFAULT_STALE_CLAIM_SECONDS,
        now: datetime | None = None,
    ) -> int:
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
            active_tier = conn.execute(
                """
                SELECT MIN(CASE upper(COALESCE(effective_trust_tier, trust_tier, 'T4'))
                    WHEN 'T1' THEN 1 WHEN 'T2' THEN 2 WHEN 'T3' THEN 3 WHEN 'T4' THEN 4 WHEN 'T5' THEN 5 ELSE 4 END)
                FROM flickr_work_items WHERE status = ?
                """,
                (PENDING,),
            ).fetchone()[0]
            rows = conn.execute(
                """
                SELECT work_item_id, query_json
                FROM flickr_work_items
                WHERE status = ?
                  AND CASE upper(COALESCE(effective_trust_tier, trust_tier, 'T4'))
                    WHEN 'T1' THEN 1 WHEN 'T2' THEN 2 WHEN 'T3' THEN 3 WHEN 'T4' THEN 4 WHEN 'T5' THEN 5 ELSE 4 END = ?
                ORDER BY
                    CASE lane WHEN 'count_probe' THEN 0 WHEN 'normal_page' THEN 1 WHEN 'bbox_page' THEN 1 ELSE 99 END,
                    COALESCE(query_priority, 999999),
                    COALESCE(split_depth, 0),
                    COALESCE(split_priority, 99),
                    COALESCE(date_kind, ''),
                    COALESCE(min_date, ''),
                    COALESCE(max_date, ''),
                    COALESCE(slice_index, 999999),
                    COALESCE(bbox_index, 999999),
                    COALESCE(bbox_label, ''),
                    page,
                    COALESCE(term, ''),
                    COALESCE(search_field, ''),
                    COALESCE(query_hash, work_item_id)
                LIMIT ?
                """,
                (PENDING, active_tier, claim_limit),
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
                    (
                        endpoint,
                        row["work_item_id"],
                        "reserved",
                        unix_timestamp,
                        unix_timestamp,
                    ),
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
                (
                    endpoint,
                    work_item_id,
                    "reserved",
                    _unix_timestamp(now),
                    _unix_timestamp(now),
                ),
            )
            conn.execute("COMMIT")

    def _api_calls_in_window(self, conn: sqlite3.Connection, now: float) -> int:
        cutoff = now - 3600
        return int(conn.execute("SELECT count(*) FROM api_call_ledger WHERE created_at >= ?", (cutoff,)).fetchone()[0])

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
            conn.execute(
                "UPDATE flickr_physical_requests SET status = ?, completed_at = ?, error = NULL WHERE physical_request_id = ?",
                (COMPLETED, _timestamp(), work_item_id),
            )
            conn.execute(
                """
                UPDATE flickr_logical_queries
                SET last_completed_upload_date = CASE
                        WHEN COALESCE(last_completed_upload_date, '') >= COALESCE((
                            SELECT max_date FROM flickr_work_items WHERE work_item_id = ?
                        ), '') THEN last_completed_upload_date
                        ELSE (SELECT max_date FROM flickr_work_items WHERE work_item_id = ?) END,
                    updated_at = ?
                WHERE logical_query_id = (
                    SELECT logical_query_id FROM flickr_work_items WHERE work_item_id = ?
                )
                """,
                (work_item_id, work_item_id, _timestamp(), work_item_id),
            )

    def fail_work_item(self, work_item_id: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE flickr_work_items SET status = ?, completed_at = ?, error = ? WHERE work_item_id = ?",
                (FAILED, _timestamp(), error, work_item_id),
            )
            conn.execute(
                """
                UPDATE flickr_physical_requests
                SET status = 'deferred', completed_at = ?, error = ?, retry_count = retry_count + 1
                WHERE physical_request_id = ?
                """,
                (_timestamp(), error, work_item_id),
            )

    def insert_source_records(self, records: list[dict[str, Any]], *, source_query: FlickrQuery) -> tuple[int, int, int, int, int]:
        inserted = 0
        unique_records = deduplicate_photo_records(records)
        skipped = len(records) - len(unique_records)
        queued = 0
        query_terms_added = 0
        duplicate_query_terms = 0
        with self._connect() as conn:
            for record in unique_records:
                photo_id = str(record.get("id") or "")
                image_url = str(record.get("url_l") or record.get("url_m") or "")
                if not photo_id or not image_url:
                    skipped += 1
                    continue
                source_record_hash = _source_record_hash(record)
                image_url_kind = "url_l" if record.get("url_l") else "url_m"
                image_url_result = conn.execute(
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
                terms_added, duplicate_terms = _merge_query_provenance(conn, "flickr", photo_id, source_query)
                query_terms_added += terms_added
                duplicate_query_terms += duplicate_terms
                _record_query_and_keyword_evidence(
                    conn,
                    source="flickr",
                    photo_id=photo_id,
                    record=record,
                    query=source_query,
                )
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
                    if image_url_result.rowcount:
                        queue_result = conn.execute(
                            """
                            INSERT OR IGNORE INTO image_triage_queue (
                                source, flickr_photo_id, image_url, image_url_kind,
                                source_record_hash, status, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                "flickr",
                                photo_id,
                                image_url,
                                image_url_kind,
                                source_record_hash,
                                PENDING,
                                _timestamp(),
                            ),
                        )
                        queued += int(queue_result.rowcount)
        return inserted, skipped, queued, query_terms_added, duplicate_query_terms

    def photo_keyword_evidence_frame(self) -> pl.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT flickr_photo_id, keyword_id, canonical_keyword_id,
                       query_execution_id, accepted_taxon_id,
                       original_trust_tier, effective_trust_tier, search_field,
                       match_basis, returned_by_query, metadata_match,
                       first_seen_at, last_seen_at
                FROM photo_keyword_evidence
                ORDER BY flickr_photo_id, keyword_id, search_field, match_basis
                """
            ).fetchall()
        return pl.DataFrame([dict(row) for row in rows]) if rows else pl.DataFrame()

    def source_records_with_query_provenance(self) -> pl.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    source,
                    flickr_photo_id,
                    image_url,
                    image_url_kind,
                    query_field AS first_query_field,
                    query_term AS first_query_term,
                    query_language AS first_query_language,
                    text_search_terms_json,
                    tag_search_terms_json,
                    all_query_labels_json,
                    query_hit_count,
                    duplicate_query_hit_count
                FROM source_records
                ORDER BY source, flickr_photo_id
                """
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            first_query_label = f"{row['first_query_field']}:{row['first_query_term']}"
            text_terms = _json_list(row["text_search_terms_json"])
            tag_terms = _json_list(row["tag_search_terms_json"])
            labels = _json_list(row["all_query_labels_json"])
            all_terms = [*text_terms, *tag_terms]
            output.append(
                {
                    "source": row["source"],
                    "flickr_photo_id": row["flickr_photo_id"],
                    "image_url": row["image_url"],
                    "image_url_kind": row["image_url_kind"],
                    "first_query_field": row["first_query_field"],
                    "first_query_term": row["first_query_term"],
                    "first_query_language": row["first_query_language"],
                    "first_query_label": first_query_label,
                    "text_search_terms": text_terms,
                    "tag_search_terms": tag_terms,
                    "all_query_labels": labels,
                    "all_query_terms": all_terms,
                    "all_query_fields": _query_fields_from_labels(labels),
                    "query_hit_count": int(row["query_hit_count"] or len(labels)),
                    "duplicate_query_hit_count": int(row["duplicate_query_hit_count"] or 0),
                }
            )
        return pl.DataFrame(output) if output else pl.DataFrame()

    def canonical_source_records_frame(self, *, photo_ids: set[str] | None = None) -> pl.DataFrame:
        empty = build_evidence_frame([], species_query="multilingual_lepidoptera")
        if photo_ids is not None and not photo_ids:
            return empty
        photo_filter = ""
        params: tuple[Any, ...] = ()
        if photo_ids is not None:
            placeholders = ", ".join("?" for _ in photo_ids)
            photo_filter = f"AND flickr_photo_id IN ({placeholders})"
            params = tuple(sorted(photo_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    source,
                    flickr_photo_id,
                    image_url,
                    image_url_kind,
                    source_record_hash,
                    query_field,
                    query_term,
                    query_language,
                    raw_json,
                    text_search_terms_json,
                    tag_search_terms_json,
                    all_query_labels_json,
                    query_definition_ids_json,
                    accepted_taxon_keys_json,
                    family_keys_json,
                    genus_keys_json,
                    species_keys_json,
                    registry_versions_json,
                    query_hit_count,
                    duplicate_query_hit_count
                FROM source_records
                WHERE image_url != ''
                  {photo_filter}
                ORDER BY source, flickr_photo_id
                """,
                params,
            ).fetchall()
        evidence_rows: list[dict[str, Any]] = []
        for row in rows:
            try:
                photo = json.loads(str(row["raw_json"]))
            except json.JSONDecodeError:
                continue
            evidence = extract_photo_evidence(photo, species_query="multilingual_lepidoptera")
            text_terms = _json_list(row["text_search_terms_json"])
            tag_terms = _json_list(row["tag_search_terms_json"])
            labels = _json_list(row["all_query_labels_json"])
            evidence.update(
                {
                    "source": row["source"],
                    "source_record_hash": row["source_record_hash"],
                    "first_query_field": row["query_field"],
                    "first_query_term": row["query_term"],
                    "first_query_language": row["query_language"],
                    "text_search_terms": text_terms,
                    "tag_search_terms": tag_terms,
                    "all_query_labels": labels,
                    "all_query_terms": [*text_terms, *tag_terms],
                    "all_query_fields": _query_fields_from_labels(labels),
                    "query_hit_count": int(row["query_hit_count"] or len(labels)),
                    "duplicate_query_hit_count": int(row["duplicate_query_hit_count"] or 0),
                    "query_definition_ids": _json_list(row["query_definition_ids_json"]),
                    "discovery_accepted_taxon_keys": _json_list(row["accepted_taxon_keys_json"]),
                    "discovery_family_keys": _json_list(row["family_keys_json"]),
                    "discovery_genus_keys": _json_list(row["genus_keys_json"]),
                    "discovery_species_keys": _json_list(row["species_keys_json"]),
                    "registry_versions": _json_list(row["registry_versions_json"]),
                }
            )
            evidence_rows.append(evidence)
        return pl.DataFrame(evidence_rows, schema=empty.schema) if evidence_rows else empty

    def export_canonical_evidence(self, output_path: str | Path) -> int:
        frame = self.canonical_source_records_frame()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(frame, output)
        return frame.height

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS canonical_keywords (
                    canonical_keyword_id TEXT PRIMARY KEY,
                    normalized_term TEXT NOT NULL UNIQUE,
                    canonical_keyword_id_source TEXT NOT NULL,
                    canonical_term TEXT NOT NULL,
                    effective_trust_tier TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS keyword_associations (
                    keyword_id TEXT PRIMARY KEY,
                    canonical_keyword_id TEXT NOT NULL REFERENCES canonical_keywords(canonical_keyword_id),
                    query_definition_id TEXT NOT NULL,
                    accepted_taxon_id TEXT NOT NULL,
                    original_trust_tier TEXT NOT NULL,
                    source TEXT NOT NULL,
                    name_class TEXT NOT NULL,
                    language TEXT NOT NULL,
                    registry_version TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS keyword_associations_canonical_idx
                    ON keyword_associations(canonical_keyword_id);
                CREATE TABLE IF NOT EXISTS flickr_logical_queries (
                    logical_query_id TEXT PRIMARY KEY,
                    canonical_keyword_id TEXT NOT NULL REFERENCES canonical_keywords(canonical_keyword_id),
                    search_field TEXT NOT NULL,
                    effective_trust_tier TEXT NOT NULL,
                    status TEXT NOT NULL,
                    last_completed_upload_date TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(canonical_keyword_id, search_field)
                );
                CREATE TABLE IF NOT EXISTS flickr_physical_requests (
                    physical_request_id TEXT PRIMARY KEY,
                    logical_query_id TEXT NOT NULL,
                    search_field TEXT NOT NULL,
                    lane TEXT NOT NULL,
                    min_upload_date TEXT NOT NULL,
                    max_upload_date TEXT NOT NULL,
                    bbox TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    UNIQUE(logical_query_id, lane, min_upload_date, max_upload_date, bbox, page)
                );
                CREATE TABLE IF NOT EXISTS flickr_query_results (
                    query_execution_id TEXT NOT NULL,
                    physical_request_id TEXT NOT NULL,
                    logical_query_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    flickr_photo_id TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    PRIMARY KEY(physical_request_id, source, flickr_photo_id)
                );
                CREATE INDEX IF NOT EXISTS flickr_query_results_logical_idx
                    ON flickr_query_results(logical_query_id, source, flickr_photo_id);
                CREATE TABLE IF NOT EXISTS photo_keyword_evidence (
                    source TEXT NOT NULL,
                    flickr_photo_id TEXT NOT NULL,
                    keyword_id TEXT NOT NULL,
                    canonical_keyword_id TEXT NOT NULL,
                    query_execution_id TEXT NOT NULL,
                    accepted_taxon_id TEXT NOT NULL,
                    original_trust_tier TEXT NOT NULL,
                    effective_trust_tier TEXT NOT NULL,
                    search_field TEXT NOT NULL,
                    match_basis TEXT NOT NULL,
                    returned_by_query INTEGER NOT NULL,
                    metadata_match INTEGER NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    PRIMARY KEY(source, flickr_photo_id, keyword_id, search_field, match_basis)
                );
                CREATE INDEX IF NOT EXISTS photo_keyword_evidence_photo_idx
                    ON photo_keyword_evidence(source, flickr_photo_id);
                """
            )
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
                    search_field TEXT,
                    query_language TEXT,
                    term_type TEXT,
                    term_confidence TEXT,
                    trust_tier TEXT,
                    effective_trust_tier TEXT,
                    logical_query_id TEXT,
                    canonical_keyword_id TEXT,
                    query_hash TEXT,
                    registry_version TEXT,
                    query_definition_id TEXT,
                    accepted_taxon_key TEXT,
                    accepted_scientific_name TEXT,
                    family_key TEXT,
                    genus_key TEXT,
                    species_key TEXT,
                    query_priority INTEGER,
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
            self._ensure_source_record_columns(conn)
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
            "search_field": "TEXT",
            "query_language": "TEXT",
            "term_type": "TEXT",
            "term_confidence": "TEXT",
            "trust_tier": "TEXT",
            "effective_trust_tier": "TEXT",
            "logical_query_id": "TEXT",
            "canonical_keyword_id": "TEXT",
            "query_hash": "TEXT",
            "registry_version": "TEXT",
            "query_definition_id": "TEXT",
            "accepted_taxon_key": "TEXT",
            "accepted_scientific_name": "TEXT",
            "family_key": "TEXT",
            "genus_key": "TEXT",
            "species_key": "TEXT",
            "query_priority": "INTEGER",
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

    def _ensure_source_record_columns(self, conn: sqlite3.Connection) -> None:
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(source_records)").fetchall()}
        columns = {
            "text_search_terms_json": "TEXT NOT NULL DEFAULT '[]'",
            "tag_search_terms_json": "TEXT NOT NULL DEFAULT '[]'",
            "all_query_labels_json": "TEXT NOT NULL DEFAULT '[]'",
            "query_definition_ids_json": "TEXT NOT NULL DEFAULT '[]'",
            "accepted_taxon_keys_json": "TEXT NOT NULL DEFAULT '[]'",
            "family_keys_json": "TEXT NOT NULL DEFAULT '[]'",
            "genus_keys_json": "TEXT NOT NULL DEFAULT '[]'",
            "species_keys_json": "TEXT NOT NULL DEFAULT '[]'",
            "registry_versions_json": "TEXT NOT NULL DEFAULT '[]'",
            "query_hit_count": "INTEGER NOT NULL DEFAULT 0",
            "duplicate_query_hit_count": "INTEGER NOT NULL DEFAULT 0",
            "last_seen_at": "TEXT",
        }
        for name, sql_type in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE source_records ADD COLUMN {name} {sql_type}")

    def _connect(self) -> sqlite3.Connection:
        conn = connect_closing(self.path, timeout=30, isolation_level=None)
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
    run_id: str | None = None,
    worker_id: str | None = None,
    storage_backend: str = "local",
    storage_prefix: str | Path | None = None,
    evidence_stage: str = "poll_once",
    compact_after_run: bool = True,
    storage: CloudStorage | None = None,
    work_store: WorkStore | None = None,
    claim_once: bool = False,
) -> PollOnceResult:
    state = MetadataPollState(state_db)
    effective_run_id = run_id or _default_run_id()
    effective_worker_id = worker_id or os.environ.get("BIOMINER_WORKER_ID") or "local"
    output_storage = storage or _storage_backend_from_name(storage_backend)
    raw_base_prefix = _raw_base_prefix(
        raw_root=raw_root,
        storage_prefix=storage_prefix,
        storage_backend=storage_backend,
    )
    evidence_base_prefix = _evidence_base_prefix(evidence_output=evidence_output, storage_prefix=storage_prefix)
    stale_requeued = state.requeue_stale_claims(stale_after_seconds=stale_claim_seconds)
    _progress(progress_callback, {"event": "stale_claims_requeued", "count": stale_requeued})
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
    evidence_rows_total = 0
    evidence_shard_rows = 0
    evidence_shard_uri: str | None = None
    evidence_shard_checksum: str | None = None
    evidence_shard_bytes: int | None = None
    shard_registry_version: str | None = None
    delta_photo_ids_by_work_item: dict[str, set[str]] = {}
    shard_registry_versions_by_work_item: dict[str, str | None] = {}
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
            _progress(
                progress_callback,
                {
                    "event": "work_claimed",
                    "claimed": len(claimed),
                    "claim_limit": claim_limit,
                },
            )
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
                        _write_raw_response(
                            storage=output_storage,
                            raw_base_prefix=str(raw_base_prefix),
                            run_id=effective_run_id,
                            work_item_id=work_item_id,
                            query=query,
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
                            (
                                inserted,
                                skipped,
                                queued_count,
                                query_hits,
                                duplicate_hits,
                            ) = state.insert_source_records(records, source_query=query)
                            photo_ids = _photo_ids_from_records(records)
                            if photo_ids:
                                for existing_photo_ids in delta_photo_ids_by_work_item.values():
                                    existing_photo_ids.difference_update(photo_ids)
                                delta_photo_ids_by_work_item[work_item_id] = photo_ids
                                shard_registry_versions_by_work_item[work_item_id] = query.registry_version
                            if query.registry_version and shard_registry_version is None:
                                shard_registry_version = query.registry_version
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
                                    "reported_pages": page_ensure.reported_pages,
                                    "accessible_pages": page_ensure.accessible_pages,
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
                                        "reported_pages": page_ensure.reported_pages,
                                        "target_pages": page_ensure.target_pages,
                                        "accessible_pages": page_ensure.accessible_pages,
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
            if claim_once:
                break

    if compact_after_run and not is_cloud_uri(evidence_output):
        evidence_rows_total = state.export_canonical_evidence(evidence_output)
        evidence_rows_written = evidence_rows_total
    else:
        for batch_work_item_id, photo_ids in sorted(delta_photo_ids_by_work_item.items()):
            if not photo_ids:
                continue
            canonical_frame = state.canonical_source_records_frame(photo_ids=photo_ids)
            (
                evidence_shard_uri,
                evidence_shard_rows,
                evidence_shard_checksum,
                evidence_shard_bytes,
            ) = _write_evidence_shard(
                storage=output_storage,
                evidence_base_prefix=str(evidence_base_prefix),
                stage=evidence_stage,
                run_id=effective_run_id,
                worker_id=effective_worker_id,
                batch_id=batch_work_item_id,
                frame=canonical_frame,
            )
            if evidence_shard_uri and work_store:
                work_store.register_shard(
                    job_name="poll_once",
                    registry_version=shard_registry_versions_by_work_item.get(batch_work_item_id, shard_registry_version),
                    stage=evidence_stage,
                    run_id=effective_run_id,
                    worker_id=effective_worker_id,
                    uri=evidence_shard_uri,
                    checksum=evidence_shard_checksum,
                    row_count=evidence_shard_rows,
                    byte_count=evidence_shard_bytes,
                )
            evidence_rows_total += evidence_shard_rows
            evidence_rows_written += evidence_shard_rows
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
    _progress(
        progress_callback,
        {
            "event": "poll_completed",
            **{**result.__dict__, "state_db": str(result.state_db)},
        },
    )
    return result


def _json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if item not in (None, "")]


def _record_query_and_keyword_evidence(
    conn: sqlite3.Connection,
    *,
    source: str,
    photo_id: str,
    record: dict[str, Any],
    query: FlickrQuery,
) -> int:
    normalized = normalize_name_key(query.normalized_term or query.term)
    canonical_id = query.canonical_keyword_id or stable_identity("canonical-keyword", normalized)
    logical_id = query.logical_query_id or stable_identity("flickr-logical-query", normalized, query.search_field)
    execution_id = _work_item_id(query)
    now = _timestamp()
    conn.execute(
        """
        INSERT OR IGNORE INTO flickr_query_results (
            query_execution_id, physical_request_id, logical_query_id,
            source, flickr_photo_id, first_seen_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (execution_id, execution_id, logical_id, source, photo_id, now),
    )
    inserted = _insert_evidence_for_canonical(
        conn,
        source=source,
        photo_id=photo_id,
        canonical_keyword_id=canonical_id,
        execution_id=execution_id,
        search_field=query.search_field,
        match_basis="query_return",
        returned_by_query=True,
        metadata_match=False,
    )
    for metadata_field, metadata_text in _metadata_text_fields(record):
        normalized_text = normalize_name_key(metadata_text)
        if not normalized_text:
            continue
        candidates = conn.execute(
            """
            SELECT canonical_keyword_id, normalized_term
            FROM canonical_keywords
            WHERE normalized_term <> '' AND instr(?, normalized_term) > 0
            ORDER BY length(normalized_term) DESC, canonical_keyword_id
            """,
            (normalized_text,),
        ).fetchall()
        for candidate in candidates:
            inserted += _insert_evidence_for_canonical(
                conn,
                source=source,
                photo_id=photo_id,
                canonical_keyword_id=str(candidate["canonical_keyword_id"]),
                execution_id=execution_id,
                search_field=query.search_field,
                match_basis=f"metadata:{metadata_field}",
                returned_by_query=False,
                metadata_match=True,
            )
    return inserted


def _insert_evidence_for_canonical(
    conn: sqlite3.Connection,
    *,
    source: str,
    photo_id: str,
    canonical_keyword_id: str,
    execution_id: str,
    search_field: str,
    match_basis: str,
    returned_by_query: bool,
    metadata_match: bool,
    keyword_id: str | None = None,
) -> int:
    associations = conn.execute(
        """
        SELECT a.keyword_id, a.accepted_taxon_id, a.original_trust_tier,
               c.effective_trust_tier
        FROM keyword_associations AS a
        JOIN canonical_keywords AS c USING (canonical_keyword_id)
        WHERE a.canonical_keyword_id = ? AND (? IS NULL OR a.keyword_id = ?)
        ORDER BY a.keyword_id
        """,
        (canonical_keyword_id, keyword_id, keyword_id),
    ).fetchall()
    inserted = 0
    now = _timestamp()
    for association in associations:
        result = conn.execute(
            """
            INSERT INTO photo_keyword_evidence (
                source, flickr_photo_id, keyword_id, canonical_keyword_id,
                query_execution_id, accepted_taxon_id, original_trust_tier,
                effective_trust_tier, search_field, match_basis,
                returned_by_query, metadata_match, first_seen_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source, flickr_photo_id, keyword_id, search_field, match_basis)
            DO UPDATE SET last_seen_at = excluded.last_seen_at,
                          returned_by_query = MAX(returned_by_query, excluded.returned_by_query),
                          metadata_match = MAX(metadata_match, excluded.metadata_match),
                          effective_trust_tier = excluded.effective_trust_tier
            """,
            (
                source,
                photo_id,
                association["keyword_id"],
                canonical_keyword_id,
                execution_id,
                association["accepted_taxon_id"],
                association["original_trust_tier"],
                association["effective_trust_tier"],
                search_field,
                match_basis,
                int(returned_by_query),
                int(metadata_match),
                now,
                now,
            ),
        )
        inserted += int(result.rowcount)
    return inserted


def _backfill_keyword_evidence(
    conn: sqlite3.Connection,
    *,
    keyword_id: str,
    canonical_keyword_id: str,
) -> int:
    rows = conn.execute(
        """
        SELECT r.source, r.flickr_photo_id, r.query_execution_id, q.search_field
        FROM flickr_query_results AS r
        JOIN flickr_logical_queries AS q USING (logical_query_id)
        WHERE q.canonical_keyword_id = ?
        """,
        (canonical_keyword_id,),
    ).fetchall()
    inserted = 0
    for row in rows:
        inserted += _insert_evidence_for_canonical(
            conn,
            source=str(row["source"]),
            photo_id=str(row["flickr_photo_id"]),
            canonical_keyword_id=canonical_keyword_id,
            execution_id=str(row["query_execution_id"]),
            search_field=str(row["search_field"]),
            match_basis="query_return",
            returned_by_query=True,
            metadata_match=False,
            keyword_id=keyword_id,
        )
    return inserted


def _metadata_text_fields(record: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    def text(value: object) -> str:
        if isinstance(value, dict):
            return str(value.get("_content") or "")
        if isinstance(value, list):
            return " ".join(text(item) for item in value)
        return str(value or "")

    return tuple((field, text(record.get(field))) for field in ("title", "tags", "machine_tags", "description", "comments") if text(record.get(field)))


def _json_dump_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _append_unique(values: list[str], value: object | None) -> bool:
    if value in (None, ""):
        return False
    item = str(value)
    if item in values:
        return False
    values.append(item)
    return True


def _merge_query_provenance(
    conn: sqlite3.Connection,
    source: str,
    flickr_photo_id: str,
    source_query: FlickrQuery,
) -> tuple[int, int]:
    row = conn.execute(
        """
        SELECT
            text_search_terms_json,
            tag_search_terms_json,
            all_query_labels_json,
            query_definition_ids_json,
            accepted_taxon_keys_json,
            family_keys_json,
            genus_keys_json,
            species_keys_json,
            registry_versions_json,
            query_hit_count,
            duplicate_query_hit_count
        FROM source_records
        WHERE source = ? AND flickr_photo_id = ?
        """,
        (source, flickr_photo_id),
    ).fetchone()
    if row is None:
        return 0, 0
    text_terms = _json_list(row["text_search_terms_json"])
    tag_terms = _json_list(row["tag_search_terms_json"])
    labels = _json_list(row["all_query_labels_json"])
    query_definition_ids = _json_list(row["query_definition_ids_json"])
    accepted_taxon_keys = _json_list(row["accepted_taxon_keys_json"])
    family_keys = _json_list(row["family_keys_json"])
    genus_keys = _json_list(row["genus_keys_json"])
    species_keys = _json_list(row["species_keys_json"])
    registry_versions = _json_list(row["registry_versions_json"])

    label = f"{source_query.search_field}:{source_query.term}"
    if source_query.search_field == "text":
        _append_unique(text_terms, source_query.term)
    elif source_query.search_field == "tags":
        _append_unique(tag_terms, source_query.term)
    label_added = _append_unique(labels, label)
    _append_unique(query_definition_ids, source_query.query_definition_id)
    _append_unique(accepted_taxon_keys, source_query.accepted_taxon_key)
    _append_unique(family_keys, source_query.family_key)
    _append_unique(genus_keys, source_query.genus_key)
    _append_unique(species_keys, source_query.species_key)
    _append_unique(registry_versions, source_query.registry_version)

    query_hit_count = int(row["query_hit_count"] or 0) + (1 if label_added else 0)
    duplicate_query_hit_count = int(row["duplicate_query_hit_count"] or 0) + (0 if label_added else 1)
    conn.execute(
        """
        UPDATE source_records
        SET text_search_terms_json = ?,
            tag_search_terms_json = ?,
            all_query_labels_json = ?,
            query_definition_ids_json = ?,
            accepted_taxon_keys_json = ?,
            family_keys_json = ?,
            genus_keys_json = ?,
            species_keys_json = ?,
            registry_versions_json = ?,
            query_hit_count = ?,
            duplicate_query_hit_count = ?,
            last_seen_at = ?
        WHERE source = ? AND flickr_photo_id = ?
        """,
        (
            _json_dump_list(text_terms),
            _json_dump_list(tag_terms),
            _json_dump_list(labels),
            _json_dump_list(query_definition_ids),
            _json_dump_list(accepted_taxon_keys),
            _json_dump_list(family_keys),
            _json_dump_list(genus_keys),
            _json_dump_list(species_keys),
            _json_dump_list(registry_versions),
            query_hit_count,
            duplicate_query_hit_count,
            _timestamp(),
            source,
            flickr_photo_id,
        ),
    )
    return (1, 0) if label_added else (0, 1)


def _query_fields_from_labels(labels: list[str]) -> list[str]:
    fields: list[str] = []
    for label in labels:
        field, separator, _term = label.partition(":")
        if separator:
            fields.append(field)
    return fields


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


def _write_raw_response(
    *,
    storage: CloudStorage,
    raw_base_prefix: str,
    run_id: str,
    work_item_id: str,
    query: FlickrQuery,
    payload: dict[str, Any],
) -> str:
    uri = build_raw_flickr_response_uri(
        raw_base_prefix,
        run_id=run_id,
        query_field=query.search_field,
        query_term=query.term,
        lane=query.lane,
        page=query.page,
        work_item_id=work_item_id,
    )
    return storage.write_json(uri, payload)


def _write_evidence_shard(
    *,
    storage: CloudStorage,
    evidence_base_prefix: str,
    stage: str,
    run_id: str,
    worker_id: str,
    batch_id: str,
    frame: pl.DataFrame,
) -> tuple[str | None, int, str | None, int | None]:
    if frame.is_empty():
        return None, 0, None, None
    uri = build_evidence_shard_uri(
        evidence_base_prefix,
        stage=stage,
        run_id=run_id,
        worker_id=worker_id,
        batch_id=batch_id,
    )
    written = storage.write_parquet_shard(uri, frame)
    checksum, byte_count = _local_artifact_metadata(written)
    return written, frame.height, checksum, byte_count


def _storage_backend_from_name(storage_backend: str) -> CloudStorage:
    backend = storage_backend.lower()
    if backend == "local":
        return LocalStorageBackend()
    config = load_storage_config_from_env()
    return create_storage_backend(StorageConfig(**{**config.__dict__, "backend": backend}))


def _raw_base_prefix(*, raw_root: str | Path, storage_prefix: str | Path | None, storage_backend: str) -> str:
    if storage_backend.lower() != "local" and storage_prefix is not None:
        return str(storage_prefix)
    raw_value = str(raw_root).rstrip("/")
    if is_cloud_uri(raw_value):
        return raw_value.removesuffix("/raw")
    raw = Path(raw_root)
    return str(raw.parent) if raw.name == "raw" else str(raw)


def _evidence_base_prefix(*, evidence_output: str | Path, storage_prefix: str | Path | None) -> str:
    if storage_prefix is not None:
        return str(storage_prefix)
    output_value = str(evidence_output).rstrip("/")
    if is_cloud_uri(output_value):
        prefix, separator, _filename = output_value.rpartition("/")
        if not separator:
            return output_value
        return prefix.removesuffix("/evidence")
    output = Path(evidence_output)
    parent = output.parent
    return str(parent.parent) if parent.name == "evidence" else str(parent)


def _default_run_id() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")


def _local_artifact_metadata(uri: str) -> tuple[str | None, int | None]:
    if is_cloud_uri(uri):
        return None, None
    try:
        path = normalize_local_uri(uri)
    except ValueError:
        return None, None
    if not path.exists() or not path.is_file():
        return None, None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}", path.stat().st_size


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


def _accessible_page_window(per_page: int) -> int:
    if per_page <= 0:
        return 1
    return max(1, FLICKR_SEARCH_RESULT_WINDOW // per_page)


def _insert_work_item(conn: sqlite3.Connection, query: FlickrQuery) -> sqlite3.Cursor:
    work_item_id = _work_item_id(query)
    normalized = normalize_name_key(query.normalized_term or query.term)
    canonical_id = query.canonical_keyword_id or stable_identity("canonical-keyword", normalized)
    logical_id = query.logical_query_id or stable_identity("flickr-logical-query", normalized, query.search_field)
    effective_tier = query.effective_trust_tier or query.trust_tier or "T4"
    now = _timestamp()
    conn.execute(
        """
        INSERT OR IGNORE INTO canonical_keywords (
            canonical_keyword_id, normalized_term, canonical_keyword_id_source,
            canonical_term, effective_trust_tier, first_seen_at, last_seen_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            canonical_id,
            normalized,
            query.keyword_id or canonical_id,
            query.term,
            effective_tier,
            now,
            now,
        ),
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO flickr_logical_queries (
            logical_query_id, canonical_keyword_id, search_field,
            effective_trust_tier, status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, 'active', ?, ?)
        """,
        (logical_id, canonical_id, query.search_field, effective_tier, now, now),
    )
    if query.keyword_id:
        conn.execute(
            """
            INSERT OR IGNORE INTO keyword_associations (
                keyword_id, canonical_keyword_id, query_definition_id,
                accepted_taxon_id, original_trust_tier, source, name_class,
                language, registry_version, first_seen_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?)
            """,
            (
                query.keyword_id,
                canonical_id,
                query.query_definition_id or logical_id,
                query.accepted_taxon_key or "",
                query.original_trust_tier or effective_tier,
                query.term_type or "",
                query.language,
                query.registry_version or "",
                now,
                now,
            ),
        )
    overlap = conn.execute(
        """
        SELECT 1 FROM flickr_physical_requests
        WHERE logical_query_id = ? AND ? <> '' AND ? <> ''
          AND min_upload_date <> '' AND max_upload_date <> ''
          AND NOT (min_upload_date = ? AND max_upload_date = ?)
          AND min_upload_date <= ? AND max_upload_date >= ?
        LIMIT 1
        """,
        (
            logical_id,
            query.min_upload_date or "",
            query.max_upload_date or "",
            query.min_upload_date or "",
            query.max_upload_date or "",
            query.max_upload_date or "",
            query.min_upload_date or "",
        ),
    ).fetchone()
    if overlap is not None:
        raise ValueError(f"overlapping upload-date interval for logical query {logical_id}")
    conn.execute(
        """
        INSERT OR IGNORE INTO flickr_physical_requests (
            physical_request_id, logical_query_id, search_field, lane,
            min_upload_date, max_upload_date, bbox, page, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            work_item_id,
            logical_id,
            query.search_field,
            query.lane,
            query.min_upload_date or "",
            query.max_upload_date or "",
            query.bbox or "",
            query.page,
            PENDING,
            now,
        ),
    )
    return conn.execute(
        """
        INSERT OR IGNORE INTO flickr_work_items (
            work_item_id, status, query_json, lane, page, per_page,
            split_depth, split_priority, split_reason, parent_query_hash,
            parent_total, date_kind, min_date, max_date, bbox_index,
            slice_index, bbox_label, term, search_field, query_language,
            term_type, term_confidence, trust_tier, effective_trust_tier,
            logical_query_id, canonical_keyword_id, query_hash, registry_version,
            query_definition_id, accepted_taxon_key, accepted_scientific_name,
            family_key, genus_key, species_key, query_priority, claimed_at,
            completed_at, error, records_returned, response_total,
            response_pages, response_page, response_perpage, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?)
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
            query.search_field,
            query.language,
            query.term_type,
            query.term_confidence,
            query.trust_tier,
            effective_tier,
            logical_id,
            canonical_id,
            query_hash(query),
            query.registry_version,
            query.query_definition_id,
            query.accepted_taxon_key,
            query.accepted_scientific_name,
            query.family_key,
            query.genus_key,
            query.species_key,
            query.query_priority,
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
    return {
        "normalized_term": normalize_name_key(query.normalized_term or query.term),
        "search_field": query.search_field,
        "lane": query.lane,
        "has_geo": query.has_geo,
        "bbox": query.bbox,
        "min_taken_date": query.min_taken_date,
        "max_taken_date": query.max_taken_date,
        "min_upload_date": query.min_upload_date,
        "max_upload_date": query.max_upload_date,
    }


def _payload_photo_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    photos = payload.get("photos", {})
    rows = photos.get("photo", []) if isinstance(photos, dict) else []
    return [row for row in rows if isinstance(row, dict)]


def _photo_ids_from_records(records: list[dict[str, Any]]) -> set[str]:
    return {str(record.get("id")) for record in records if record.get("id") not in (None, "")}


def _work_item_id(query: FlickrQuery) -> str:
    return query_hash(query)


def _source_record_hash(record: dict[str, Any]) -> str:
    import hashlib

    payload = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _query_from_json(payload: str) -> FlickrQuery:
    data = json.loads(payload)
    data.setdefault("query_priority", 999999)
    return FlickrQuery(**data)


def _timestamp(value: datetime | None = None) -> str:
    return (value or datetime.now(UTC)).isoformat()


def _unix_timestamp(value: datetime | None = None) -> float:
    return (value or datetime.now(UTC)).timestamp()
