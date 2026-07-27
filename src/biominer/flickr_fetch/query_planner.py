from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
import json
from math import ceil
from pathlib import Path
from typing import Any, Iterable, Literal

import polars as pl

from biominer.registry.normalize import normalize_name_key
from biominer.registry.unified import stable_identity


SearchField = Literal["text", "tags"]
QueryLane = Literal["count_probe", "normal_page", "bbox_page"]

NORMAL_PAGE_SIZE = 500
GEO_PAGE_SIZE = 250
BBOX_PAGE_SIZE = GEO_PAGE_SIZE
COUNT_PROBE_PAGE_SIZE = 1
FLICKR_SEARCH_RESULT_WINDOW = 4000
STABLE_RESULT_THRESHOLD = FLICKR_SEARCH_RESULT_WINDOW
MAX_ACCESSIBLE_RESULTS_PER_QUERY = FLICKR_SEARCH_RESULT_WINDOW
SPLIT_REASON_PRIORITY = {
    None: 0,
    "taken_date": 1,
    "upload_date": 2,
    "bbox": 3,
    "narrower_term": 4,
}
LANE_PRIORITY = {"count_probe": 0, "normal_page": 1, "bbox_page": 1}
DEFAULT_EXTRAS = "description,license,date_upload,date_taken,geo,tags,machine_tags,owner_name,o_dims,url_l,url_m,last_update,media,views"
DEFAULT_FIXED_SLICE_START_DATE = "2004-02-10"
DEFAULT_FIXED_SLICE_END_DATE = date.today().isoformat()
DEFAULT_COARSE_SLICE_END_DATE: str | None = None
DEFAULT_COARSE_SLICE_DAYS: int | None = None
DEFAULT_FIXED_SLICE_DAYS = 5
@dataclass(frozen=True)
class FlickrQuery:
    term: str
    language: str
    search_field: SearchField
    lane: QueryLane
    normalized_term: str | None = None
    logical_query_id: str | None = None
    canonical_keyword_id: str | None = None
    keyword_id: str | None = None
    original_trust_tier: str | None = None
    effective_trust_tier: str | None = None
    api_language_code: str | None = None
    bcp47: str | None = None
    page: int = 1
    per_page: int = COUNT_PROBE_PAGE_SIZE
    has_geo: int = 1
    bbox: str | None = None
    place_id: str | None = None
    woe_id: str | None = None
    min_taken_date: str | None = None
    max_taken_date: str | None = None
    min_upload_date: str | None = None
    max_upload_date: str | None = None
    split_reason: str | None = None
    parent_total: int | None = None
    region: str | None = None
    term_type: str | None = None
    term_confidence: str | None = None
    trust_tier: str | None = None
    notes: str | None = None
    parent_query_hash: str | None = None
    split_depth: int = 0
    bbox_index: int | None = None
    slice_index: int | None = None
    registry_version: str | None = None
    query_definition_id: str | None = None
    accepted_taxon_key: str | None = None
    accepted_scientific_name: str | None = None
    keyword_owner_taxon_key: str | None = None
    keyword_owner_rank: str | None = None
    keyword_ownership_basis: str | None = None
    query_stage: str | None = None
    query_stage_order: int = 99
    family_key: str | None = None
    genus_key: str | None = None
    species_key: str | None = None
    query_priority: int = 999999


def load_registry_flickr_queries(
    path: str | Path,
    *,
    start_date: str = DEFAULT_FIXED_SLICE_START_DATE,
    end_date: str = DEFAULT_FIXED_SLICE_END_DATE,
    slice_days: int = DEFAULT_FIXED_SLICE_DAYS,
) -> tuple[FlickrQuery, ...]:
    return load_registry_flickr_queries_from_frame(
        pl.read_parquet(path),
        start_date=start_date,
        end_date=end_date,
        slice_days=slice_days,
    )


def load_registry_flickr_queries_from_frame(
    frame: pl.DataFrame,
    *,
    start_date: str = DEFAULT_FIXED_SLICE_START_DATE,
    end_date: str = DEFAULT_FIXED_SLICE_END_DATE,
    slice_days: int = DEFAULT_FIXED_SLICE_DAYS,
) -> tuple[FlickrQuery, ...]:
    del start_date, end_date, slice_days
    if frame.is_empty():
        return ()
    if "normalized_match_key" not in frame.columns:
        frame = frame.with_columns(pl.col("source_term").str.to_lowercase().alias("normalized_match_key"))
    if "query_eligible" not in frame.columns:
        raise ValueError("flickr query definitions require the query_eligible field")
    enabled_filter = pl.col("enabled") if "enabled" in frame.columns else pl.lit(True)
    query_eligible_filter = pl.col("query_eligible")
    rows = frame.filter(enabled_filter & query_eligible_filter).sort(["search_priority", "normalized_match_key", "query_definition_id"]).to_dicts()
    queries: list[FlickrQuery] = []
    seen_logical_queries: set[tuple[str, str]] = set()
    for row in rows:
        field = str(row.get("search_field") or "text")
        if field not in {"text", "tags"}:
            continue
        normalized_term = str(row.get("normalized_match_key") or normalize_name_key(row.get("source_term")))
        logical_key = (normalized_term, field)
        if not normalized_term or logical_key in seen_logical_queries:
            continue
        seen_logical_queries.add(logical_key)
        bbox = str(row.get("bbox") or "") or None
        logical_query_id = str(row.get("logical_query_id") or "") or stable_identity("flickr-logical-query", normalized_term, field)
        queries.append(
            FlickrQuery(
                term=str(row.get("source_term") or row.get("normalized_query_term") or ""),
                normalized_term=normalized_term,
                logical_query_id=logical_query_id,
                canonical_keyword_id=str(row.get("canonical_keyword_id") or "") or stable_identity("canonical-keyword", normalized_term),
                keyword_id=str(row.get("keyword_id") or row.get("name_id") or "") or None,
                original_trust_tier=str(row.get("original_trust_tier") or row.get("trust_tier") or "") or None,
                effective_trust_tier=str(row.get("effective_trust_tier") or row.get("trust_tier") or "") or None,
                language=str(row.get("language") or "und"),
                api_language_code=str(row.get("api_language_code") or "") or None,
                bcp47=str(row.get("bcp47") or "") or None,
                search_field=field,
                lane="bbox_page" if bbox else "normal_page",
                page=1,
                per_page=BBOX_PAGE_SIZE if bbox else NORMAL_PAGE_SIZE,
                has_geo=1 if bbox else 0,
                bbox=bbox,
                region=str(row.get("region") or "") or None,
                term_type=str(row.get("name_class") or "") or None,
                term_confidence=str(row.get("confidence") or "") or None,
                trust_tier=str(row.get("trust_tier") or "") or None,
                registry_version=str(row.get("registry_version") or "") or None,
                query_definition_id=str(row.get("query_definition_id") or "") or None,
                accepted_taxon_key=str(row.get("accepted_taxon_key") or "") or None,
                accepted_scientific_name=str(row.get("accepted_scientific_name") or "") or None,
                keyword_owner_taxon_key=str(row.get("keyword_owner_taxon_key") or "") or None,
                keyword_owner_rank=str(row.get("keyword_owner_rank") or "") or None,
                keyword_ownership_basis=str(row.get("keyword_ownership_basis") or "") or None,
                query_stage=str(row.get("query_stage") or "") or None,
                query_stage_order=_int_or_default(row.get("query_stage_order"), 99),
                family_key=str(row.get("family_key") or "") or None,
                genus_key=str(row.get("genus_key") or "") or None,
                species_key=str(row.get("species_key") or "") or None,
                query_priority=_int_or_default(row.get("search_priority"), 999999),
            )
        )
    return _sort_queries(queries)


def plan_pages_from_count(probe: FlickrQuery, *, total: int) -> tuple[FlickrQuery, ...]:
    if total <= 0:
        return ()
    accessible_total = min(total, MAX_ACCESSIBLE_RESULTS_PER_QUERY)
    per_page = page_size_for_query(probe)
    lane: QueryLane = "bbox_page" if probe.bbox else "normal_page"
    return tuple(_page_query(probe, page=page, per_page=per_page, lane=lane) for page in range(1, ceil(accessible_total / per_page) + 1))


def fixed_upload_date_slices(
    *,
    start_date: str,
    end_date: str,
    slice_days: int = DEFAULT_FIXED_SLICE_DAYS,
    coarse_end_date: str | None = None,
    coarse_slice_days: int | None = None,
) -> tuple[tuple[str, str], ...]:
    if slice_days <= 0:
        raise ValueError("slice_days must be positive")
    end = date.fromisoformat(end_date)
    current = date.fromisoformat(start_date)
    if coarse_end_date and coarse_slice_days:
        if coarse_slice_days <= 0:
            raise ValueError("coarse_slice_days must be positive")
        coarse_end = min(date.fromisoformat(coarse_end_date), end)
        slices = _date_slices(current=current, end=coarse_end, slice_days=coarse_slice_days) if current <= coarse_end else ()
        current = max(current, coarse_end + timedelta(days=1))
        if current > end:
            return slices
        return (*slices, *_date_slices(current=current, end=end, slice_days=slice_days))
    return _date_slices(current=current, end=end, slice_days=slice_days)


def _date_slices(*, current: date, end: date, slice_days: int) -> tuple[tuple[str, str], ...]:
    slices: list[tuple[str, str]] = []
    while current <= end:
        slice_end = min(current + timedelta(days=slice_days - 1), end)
        slices.append((current.isoformat(), slice_end.isoformat()))
        current = slice_end + timedelta(days=1)
    return tuple(slices)


def plan_fixed_upload_slice_pages(
    *,
    term: str,
    search_field: SearchField,
    start_date: str = DEFAULT_FIXED_SLICE_START_DATE,
    end_date: str = DEFAULT_FIXED_SLICE_END_DATE,
    slice_days: int = DEFAULT_FIXED_SLICE_DAYS,
    coarse_end_date: str | None = DEFAULT_COARSE_SLICE_END_DATE,
    coarse_slice_days: int | None = DEFAULT_COARSE_SLICE_DAYS,
    language: str = "en",
) -> tuple[FlickrQuery, ...]:
    pages: list[FlickrQuery] = []
    for slice_index, (start, end) in enumerate(
        fixed_upload_date_slices(
            start_date=start_date,
            end_date=end_date,
            slice_days=slice_days,
            coarse_end_date=coarse_end_date,
            coarse_slice_days=coarse_slice_days,
        )
    ):
        pages.append(
            FlickrQuery(
                term=term,
                language=language,
                search_field=search_field,
                lane="normal_page",
                page=1,
                per_page=NORMAL_PAGE_SIZE,
                has_geo=0,
                min_upload_date=start,
                max_upload_date=end,
                split_reason="upload_date",
                split_depth=1,
                slice_index=slice_index,
            )
        )
    return _sort_queries(pages)


def result_pages_for_total(total: int, *, per_page: int = NORMAL_PAGE_SIZE) -> int:
    if total <= 0:
        return 0
    return ceil(total / per_page)


def page_size_for_query(query: FlickrQuery) -> int:
    if query.has_geo or query.bbox:
        return GEO_PAGE_SIZE
    return NORMAL_PAGE_SIZE


def plan_queries_from_count(
    probe: FlickrQuery,
    *,
    total: int,
    taken_date_ranges: Iterable[tuple[str, str]] = (),
    upload_date_ranges: Iterable[tuple[str, str]] = (),
    bboxes: Iterable[str] = (),
    narrower_terms: Iterable[str] = (),
) -> tuple[FlickrQuery, ...]:
    del taken_date_ranges, upload_date_ranges, bboxes, narrower_terms
    unsliced_probe = replace(
        probe,
        min_taken_date=None,
        max_taken_date=None,
        min_upload_date=None,
        max_upload_date=None,
        split_reason=None,
        split_depth=0,
        slice_index=None,
        parent_total=total,
    )
    return plan_pages_from_count(unsliced_probe, total=total)


def split_priority(query: FlickrQuery) -> int:
    return SPLIT_REASON_PRIORITY.get(query.split_reason, 99)


def lane_priority(query: FlickrQuery) -> int:
    return LANE_PRIORITY.get(query.lane, 99)


def query_hash(query: FlickrQuery) -> str:
    return _query_hash(query)


def query_date_kind(query: FlickrQuery) -> str:
    if query.min_taken_date or query.max_taken_date:
        return "taken_date"
    if query.min_upload_date or query.max_upload_date:
        return "upload_date"
    return ""


def query_min_date(query: FlickrQuery) -> str:
    return query.min_taken_date or query.min_upload_date or ""


def query_max_date(query: FlickrQuery) -> str:
    return query.max_taken_date or query.max_upload_date or ""


def _sort_queries(queries: Iterable[FlickrQuery]) -> tuple[FlickrQuery, ...]:
    return tuple(sorted(queries, key=_query_sort_key))


def _query_sort_key(query: FlickrQuery) -> tuple[object, ...]:
    return (
        query.query_priority,
        lane_priority(query),
        query.split_depth,
        split_priority(query),
        query_date_kind(query),
        query_min_date(query),
        query_max_date(query),
        query.bbox_index if query.bbox_index is not None else 999999,
        query.slice_index if query.slice_index is not None else 999999,
        query.region or "",
        query.page,
        query.term.casefold(),
        query.search_field,
        _query_hash(query),
    )


def flickr_search_params(query: FlickrQuery) -> dict[str, str | int]:
    params: dict[str, str | int] = {
        query.search_field: query.term,
        "has_geo": query.has_geo,
        "media": "photos",
        "content_types": "0",
        "safe_search": 1,
        "extras": DEFAULT_EXTRAS,
        "per_page": query.per_page,
        "page": query.page,
    }
    for key in (
        "bbox",
        "place_id",
        "woe_id",
        "min_taken_date",
        "max_taken_date",
        "min_upload_date",
        "max_upload_date",
    ):
        value = getattr(query, key)
        if value:
            params[key] = value
    return params


def deduplicate_photo_records(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for record in records:
        key = str(record.get("id") or record.get("flickr_photo_id") or "")
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def _page_query(probe: FlickrQuery, *, page: int, per_page: int, lane: QueryLane) -> FlickrQuery:
    return FlickrQuery(
        term=probe.term,
        normalized_term=probe.normalized_term,
        logical_query_id=probe.logical_query_id,
        canonical_keyword_id=probe.canonical_keyword_id,
        keyword_id=probe.keyword_id,
        original_trust_tier=probe.original_trust_tier,
        effective_trust_tier=probe.effective_trust_tier,
        language=probe.language,
        api_language_code=probe.api_language_code,
        bcp47=probe.bcp47,
        search_field=probe.search_field,
        lane=lane,
        page=page,
        per_page=per_page,
        has_geo=probe.has_geo,
        bbox=probe.bbox,
        place_id=probe.place_id,
        woe_id=probe.woe_id,
        min_taken_date=probe.min_taken_date,
        max_taken_date=probe.max_taken_date,
        min_upload_date=probe.min_upload_date,
        max_upload_date=probe.max_upload_date,
        split_reason=probe.split_reason,
        parent_total=probe.parent_total,
        region=probe.region,
        term_type=probe.term_type,
        term_confidence=probe.term_confidence,
        trust_tier=probe.trust_tier,
        notes=probe.notes,
        parent_query_hash=probe.parent_query_hash,
        split_depth=probe.split_depth,
        bbox_index=probe.bbox_index,
        slice_index=probe.slice_index,
        registry_version=probe.registry_version,
        query_definition_id=probe.query_definition_id,
        accepted_taxon_key=probe.accepted_taxon_key,
        accepted_scientific_name=probe.accepted_scientific_name,
        family_key=probe.family_key,
        genus_key=probe.genus_key,
        species_key=probe.species_key,
        query_priority=probe.query_priority,
    )


def _query_hash(query: FlickrQuery) -> str:
    import hashlib

    payload_dict = {
        "normalized_term": normalize_name_key(query.normalized_term or query.term),
        "search_field": query.search_field,
        "lane": query.lane,
        "page": query.page,
        "per_page": query.per_page,
        "has_geo": query.has_geo,
        "bbox": query.bbox,
        "place_id": query.place_id,
        "woe_id": query.woe_id,
        "min_taken_date": query.min_taken_date,
        "max_taken_date": query.max_taken_date,
        "min_upload_date": query.min_upload_date,
        "max_upload_date": query.max_upload_date,
    }
    payload = json.dumps(payload_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_upload_interval(query: FlickrQuery, *, now: datetime | None = None) -> tuple[FlickrQuery, FlickrQuery] | None:
    """Bisect an inclusive Flickr upload interval into disjoint UTC-second ranges.

    Flickr exposes only the first 4,000 search results.  This operation is used
    after a normal page-one response proves a term/time slice is too large; it
    never performs a separate count request.
    """

    start = _upload_bound_to_epoch(query.min_upload_date, is_start=True, now=now)
    end = _upload_bound_to_epoch(query.max_upload_date, is_start=False, now=now)
    if start >= end:
        return None
    midpoint = start + (end - start) // 2
    if midpoint >= end:
        return None
    parent = query_hash(query)
    common = {
        "lane": "bbox_page" if query.has_geo or query.bbox or query.place_id or query.woe_id else "normal_page",
        "page": 1,
        "per_page": page_size_for_query(query),
        "min_taken_date": None,
        "max_taken_date": None,
        "split_reason": "upload_date",
        "parent_total": query.parent_total,
        "parent_query_hash": parent,
        "split_depth": query.split_depth + 1,
        "slice_index": None,
    }
    return (
        replace(query, **common, min_upload_date=str(start), max_upload_date=str(midpoint)),
        replace(query, **common, min_upload_date=str(midpoint + 1), max_upload_date=str(end)),
    )


def _upload_bound_to_epoch(value: str | None, *, is_start: bool, now: datetime | None) -> int:
    if not value:
        value = DEFAULT_FIXED_SLICE_START_DATE if is_start else (now or datetime.now(UTC)).date().isoformat()
    try:
        return int(value)
    except ValueError:
        parsed = datetime.fromisoformat(value).replace(tzinfo=UTC)
        if len(value) == 10 and not is_start:
            parsed += timedelta(days=1, seconds=-1)
        return int(parsed.timestamp())


def _int_or_default(value: object, default: int) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except TypeError, ValueError:
        return default
