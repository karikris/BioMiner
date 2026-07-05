from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Callable

import polars as pl

from biominer.flickr_fetch.metadata_poller import (
    DEFAULT_MAX_RETRIES,
    HARD_API_CALLS_PER_HOUR,
    SOFT_API_CALLS_PER_HOUR,
    FlickrFetchFailure,
    _accessible_page_window,
    _classify_fetch_error,
    _http_fetcher,
    _payload_page,
    _payload_pages,
    _payload_perpage,
    _payload_photo_records,
    _payload_total,
    _source_record_hash,
    _validate_flickr_search_payload,
    _write_evidence_shard,
    _write_raw_response,
)
from biominer.filter.extractor import build_evidence_frame, extract_photo_evidence
from biominer.flickr_fetch.endpoints import SEARCH_METHOD
from biominer.flickr_fetch.query_planner import FlickrQuery, deduplicate_photo_records, plan_queries_from_count, query_hash
from biominer.storage.cloud import CloudStorage
from biominer.workstore.base import WorkStore

FetchMetadata = Callable[[FlickrQuery], dict[str, Any]]


@dataclass(frozen=True)
class CloudPollResult:
    raw_responses_written: int = 0
    evidence_rows_written: int = 0
    evidence_rows_total: int = 0
    source_records_inserted: int = 0
    duplicate_records_skipped: int = 0
    query_hits_inserted: int = 0
    duplicate_query_hits_skipped: int = 0
    image_urls_queued: int = 0
    work_items_claimed: int = 0
    api_calls_made: int = 0
    remaining_soft_budget: int = SOFT_API_CALLS_PER_HOUR
    remaining_hard_budget: int = HARD_API_CALLS_PER_HOUR
    workstore_work_items_completed: int = 0
    workstore_work_items_failed: int = 0
    workstore_followup_work_items_enqueued: int = 0
    source_record_shard_uris: tuple[str, ...] = ()


class CloudMetadataPoller:
    """Cloud-native Flickr metadata poller.

    This poller claims production work directly from ``WorkStore`` and writes
    raw Flickr JSON plus canonical source-record Parquet shards directly to the
    configured ``CloudStorage`` backend. It deliberately does not create a
    local ``MetadataPollState`` database.
    """

    def __init__(
        self,
        *,
        storage: CloudStorage,
        workstore: WorkStore,
        job_name: str,
        stage: str,
        registry_version: str | None,
        run_id: str,
        worker_id: str,
        storage_prefix: str,
        fetch_metadata: FetchMetadata | None = None,
        api_key: str | None = None,
        max_api_calls: int = SOFT_API_CALLS_PER_HOUR,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = 0.0,
    ) -> None:
        self.storage = storage
        self.workstore = workstore
        self.job_name = job_name
        self.stage = stage
        self.registry_version = registry_version
        self.run_id = run_id
        self.worker_id = worker_id
        self.storage_prefix = storage_prefix
        self.fetch_metadata = fetch_metadata or _http_fetcher(api_key=api_key)
        self.max_api_calls = max(0, int(max_api_calls))
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._canonical_rows_by_photo: dict[tuple[str, str], dict[str, Any]] = {}

    def run_once(self, *, claim_limit: int) -> CloudPollResult:
        limit = min(max(0, int(claim_limit)), self.max_api_calls)
        if limit <= 0:
            return CloudPollResult(remaining_soft_budget=self.max_api_calls, remaining_hard_budget=HARD_API_CALLS_PER_HOUR)
        claimed = self.workstore.claim_next_batch(
            self.worker_id,
            limit,
            job_name=self.job_name,
            stage=self.stage,
            registry_version=self.registry_version,
        )
        metrics = _MutableCloudPollMetrics(work_items_claimed=len(claimed))
        for item in claimed:
            if metrics.api_calls_made >= self.max_api_calls:
                self.workstore.mark_failed(str(item.get("work_key") or ""), "flickr_poll_api_budget_exhausted")
                metrics.workstore_work_items_failed += 1
                continue
            self._process_work_item(item, metrics)
        return metrics.to_result(max_api_calls=self.max_api_calls)

    def _process_work_item(self, item: dict[str, Any], metrics: _MutableCloudPollMetrics) -> None:
        work_key = str(item.get("work_key") or "")
        query = flickr_query_from_work_item(item)
        try:
            payload, attempts = _fetch_with_retries_without_local_state(
                query=query,
                fetcher=self.fetch_metadata,
                max_retries=self.max_retries,
                retry_backoff_seconds=self.retry_backoff_seconds,
            )
            metrics.api_calls_made += attempts
            _write_raw_response(
                storage=self.storage,
                raw_base_prefix=self.storage_prefix,
                run_id=self.run_id,
                work_item_id=work_key,
                query=query,
                payload=payload,
            )
            metrics.raw_responses_written += 1
            page_shard_uri: str | None = None
            page_row_count = 0
            if query.lane == "count_probe":
                followups = tuple(plan_queries_from_count(query, total=_payload_total(payload)))
            else:
                records = _payload_photo_records(payload)
                page_rows, inserted, skipped, query_hits, duplicate_hits, queued = self._canonical_delta_rows(records, query=query)
                metrics.source_records_inserted += inserted
                metrics.duplicate_records_skipped += skipped
                metrics.query_hits_inserted += query_hits
                metrics.duplicate_query_hits_skipped += duplicate_hits
                metrics.image_urls_queued += queued
                frame = _evidence_frame(page_rows)
                page_shard_uri, page_row_count, checksum, byte_count = _write_evidence_shard(
                    storage=self.storage,
                    evidence_base_prefix=self.storage_prefix,
                    stage=self.stage,
                    run_id=self.run_id,
                    worker_id=self.worker_id,
                    batch_id=work_key,
                    frame=frame,
                )
                if page_shard_uri is not None:
                    self.workstore.register_shard(
                        job_name=self.job_name,
                        registry_version=query.registry_version or self.registry_version,
                        stage=self.stage,
                        run_id=self.run_id,
                        worker_id=self.worker_id,
                        uri=page_shard_uri,
                        checksum=checksum,
                        row_count=page_row_count,
                        byte_count=byte_count,
                        metadata={
                            "work_key": work_key,
                            "query_definition_id": query.query_definition_id,
                            "search_field": query.search_field,
                            "term": query.term,
                        },
                    )
                    metrics.source_record_shard_uris.append(page_shard_uri)
                    metrics.evidence_rows_written += page_row_count
                    metrics.evidence_rows_total += page_row_count
                followups = _followup_page_queries(query, payload)
            metrics.workstore_followup_work_items_enqueued += self._enqueue_followups(followups)
            self.workstore.mark_completed(work_key, output_uri=page_shard_uri, checksum=None, row_count=page_row_count)
            metrics.workstore_work_items_completed += 1
        except FlickrFetchFailure as exc:
            metrics.api_calls_made += exc.attempts
            self.workstore.mark_failed(work_key, str(exc))
            metrics.workstore_work_items_failed += 1
        except Exception as exc:  # noqa: BLE001 - work item failure must return to control-plane state.
            self.workstore.mark_failed(work_key, str(exc) or exc.__class__.__name__)
            metrics.workstore_work_items_failed += 1

    def _canonical_delta_rows(
        self,
        records: list[dict[str, Any]],
        *,
        query: FlickrQuery,
    ) -> tuple[list[dict[str, Any]], int, int, int, int, int]:
        unique = deduplicate_photo_records(records)
        skipped = len(records) - len(unique)
        inserted = 0
        queued = 0
        query_hits = 0
        duplicate_hits = 0
        touched: list[dict[str, Any]] = []
        touched_keys: set[tuple[str, str]] = set()
        for record in unique:
            source = str(record.get("source") or "flickr")
            photo_id = str(record.get("id") or record.get("flickr_photo_id") or "")
            if not photo_id:
                skipped += 1
                continue
            prepared = {**record, "source": source, "source_record_hash": _source_record_hash(record)}
            row = extract_photo_evidence(prepared, species_query="multilingual_lepidoptera")
            if not row.get("image_url"):
                skipped += 1
                continue
            key = (source, photo_id)
            canonical = self._canonical_rows_by_photo.get(key)
            if canonical is None:
                canonical = row
                self._canonical_rows_by_photo[key] = canonical
                inserted += 1
                queued += 1
            added, duplicate = _fold_query_provenance(canonical, query=query)
            query_hits += added
            duplicate_hits += duplicate
            if key not in touched_keys:
                touched.append(canonical)
                touched_keys.add(key)
        return touched, inserted, skipped, query_hits, duplicate_hits, queued

    def _enqueue_followups(self, queries: tuple[FlickrQuery, ...]) -> int:
        if not queries:
            return 0
        return self.workstore.enqueue_work(
            self.job_name,
            self.registry_version,
            [flickr_query_work_item(query, run_id=self.run_id) for query in queries],
            stage=self.stage,
        )


@dataclass
class _MutableCloudPollMetrics:
    raw_responses_written: int = 0
    evidence_rows_written: int = 0
    evidence_rows_total: int = 0
    source_records_inserted: int = 0
    duplicate_records_skipped: int = 0
    query_hits_inserted: int = 0
    duplicate_query_hits_skipped: int = 0
    image_urls_queued: int = 0
    work_items_claimed: int = 0
    api_calls_made: int = 0
    workstore_work_items_completed: int = 0
    workstore_work_items_failed: int = 0
    workstore_followup_work_items_enqueued: int = 0
    source_record_shard_uris: list[str] | None = None

    def __post_init__(self) -> None:
        if self.source_record_shard_uris is None:
            self.source_record_shard_uris = []

    def to_result(self, *, max_api_calls: int) -> CloudPollResult:
        soft_remaining = max(0, max_api_calls - self.api_calls_made)
        hard_remaining = max(0, HARD_API_CALLS_PER_HOUR - self.api_calls_made)
        return CloudPollResult(
            raw_responses_written=self.raw_responses_written,
            evidence_rows_written=self.evidence_rows_written,
            evidence_rows_total=self.evidence_rows_total,
            source_records_inserted=self.source_records_inserted,
            duplicate_records_skipped=self.duplicate_records_skipped,
            query_hits_inserted=self.query_hits_inserted,
            duplicate_query_hits_skipped=self.duplicate_query_hits_skipped,
            image_urls_queued=self.image_urls_queued,
            work_items_claimed=self.work_items_claimed,
            api_calls_made=self.api_calls_made,
            remaining_soft_budget=soft_remaining,
            remaining_hard_budget=hard_remaining,
            workstore_work_items_completed=self.workstore_work_items_completed,
            workstore_work_items_failed=self.workstore_work_items_failed,
            workstore_followup_work_items_enqueued=self.workstore_followup_work_items_enqueued,
            source_record_shard_uris=tuple(self.source_record_shard_uris or ()),
        )


def flickr_query_work_item(query: FlickrQuery, *, run_id: str) -> dict[str, Any]:
    return {
        "work_key": f"{run_id}:flickr:{query_hash(query)}",
        "run_id": run_id,
        "query": asdict(query),
    }


def flickr_query_from_work_item(item: dict[str, Any]) -> FlickrQuery:
    payload = item.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"work item {item.get('work_key')} has invalid payload")
    query_payload = payload.get("query")
    if isinstance(query_payload, FlickrQuery):
        return query_payload
    if not isinstance(query_payload, dict):
        raise ValueError(f"work item {item.get('work_key')} has no Flickr query payload")
    return FlickrQuery(**query_payload)


def _fetch_with_retries_without_local_state(
    *,
    query: FlickrQuery,
    fetcher: FetchMetadata,
    max_retries: int,
    retry_backoff_seconds: float,
) -> tuple[dict[str, Any], int]:
    attempts = 0
    while True:
        attempts += 1
        try:
            payload = fetcher(query)
            _validate_flickr_search_payload(payload)
            return payload, attempts
        except Exception as exc:  # noqa: BLE001 - retry classification mirrors metadata_poller.
            error = _classify_fetch_error(exc)
            if not error.retryable or attempts > max_retries:
                raise FlickrFetchFailure(error, attempts=attempts) from exc
            if retry_backoff_seconds > 0:
                import time

                time.sleep(retry_backoff_seconds * (2 ** (attempts - 1)))


def _fold_query_provenance(row: dict[str, Any], *, query: FlickrQuery) -> tuple[int, int]:
    if not row.get("first_query_field"):
        row["first_query_field"] = query.search_field
        row["first_query_term"] = query.term
        row["first_query_language"] = query.language
    if query.search_field == "text":
        _append_unique(row["text_search_terms"], query.term)
    elif query.search_field == "tags":
        _append_unique(row["tag_search_terms"], query.term)
    label = f"{query.search_field}:{query.term}"
    label_added = _append_unique(row["all_query_labels"], label)
    _append_unique(row["all_query_terms"], query.term)
    _append_unique(row["all_query_fields"], query.search_field)
    _append_unique(row["query_definition_ids"], query.query_definition_id)
    _append_unique(row["discovery_accepted_taxon_keys"], query.accepted_taxon_key)
    _append_unique(row["discovery_family_keys"], query.family_key)
    _append_unique(row["discovery_genus_keys"], query.genus_key)
    _append_unique(row["discovery_species_keys"], query.species_key)
    _append_unique(row["registry_versions"], query.registry_version)
    if label_added:
        row["query_hit_count"] = int(row.get("query_hit_count") or 0) + 1
        return 1, 0
    row["duplicate_query_hit_count"] = int(row.get("duplicate_query_hit_count") or 0) + 1
    return 0, 1


def _append_unique(values: list[str], value: object | None) -> bool:
    if value in (None, ""):
        return False
    item = str(value)
    if item in values:
        return False
    values.append(item)
    return True


def _followup_page_queries(query: FlickrQuery, payload: dict[str, Any]) -> tuple[FlickrQuery, ...]:
    response_pages = _payload_pages(payload)
    response_page = _payload_page(payload)
    response_perpage = _payload_perpage(payload) or query.per_page
    if response_pages <= response_page:
        return ()
    accessible_pages = min(response_pages, _accessible_page_window(response_perpage))
    if accessible_pages <= response_page:
        return ()
    lane = "bbox_page" if query.bbox else "normal_page"
    return tuple(replace(query, page=page, per_page=response_perpage, lane=lane) for page in range(response_page + 1, accessible_pages + 1))


def _evidence_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    empty = build_evidence_frame([], species_query="multilingual_lepidoptera")
    return pl.DataFrame(rows, schema=empty.schema) if rows else empty


__all__ = [
    "CloudMetadataPoller",
    "CloudPollResult",
    "flickr_query_from_work_item",
    "flickr_query_work_item",
]
