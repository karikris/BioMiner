from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
import json
from math import ceil
from pathlib import Path
from typing import Any, Iterable, Literal

import polars as pl


SearchField = Literal["text", "tags"]
QueryLane = Literal["count_probe", "normal_page", "bbox_page"]

NORMAL_PAGE_SIZE = 500
GEO_PAGE_SIZE = 250
BBOX_PAGE_SIZE = GEO_PAGE_SIZE
COUNT_PROBE_PAGE_SIZE = 1
FLICKR_SEARCH_RESULT_WINDOW = 4000
STABLE_RESULT_THRESHOLD = FLICKR_SEARCH_RESULT_WINDOW
MAX_ACCESSIBLE_RESULTS_PER_QUERY = FLICKR_SEARCH_RESULT_WINDOW
SPLIT_TOTAL_THRESHOLD = STABLE_RESULT_THRESHOLD
SPLIT_REASON_PRIORITY = {
    None: 0,
    "taken_date": 1,
    "upload_date": 2,
    "bbox": 3,
    "narrower_term": 4,
}
LANE_PRIORITY = {"count_probe": 0, "normal_page": 1, "bbox_page": 1}
DEFAULT_EXTRAS = "description,license,date_upload,date_taken,geo,tags,machine_tags,owner_name,o_dims,url_l,url_m,last_update,media,views"
DEFAULT_UPLOAD_DATE_RANGES: tuple[tuple[str, str], ...] = (
    ("2004-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2026-12-31"),
)
DEFAULT_FIXED_SLICE_START_DATE = "2004-02-10"
DEFAULT_FIXED_SLICE_END_DATE = date.today().isoformat()
DEFAULT_COARSE_SLICE_END_DATE: str | None = None
DEFAULT_COARSE_SLICE_DAYS: int | None = None
DEFAULT_FIXED_SLICE_DAYS = 5

MULTILINGUAL_SEED_TERMS: tuple[tuple[str, str], ...] = (
    ("en", "butterfly"),
    ("en", "butterflies"),
    ("en", "lepidoptera"),
    ("en", "swallowtail"),
    ("en", "moth"),
    ("en", "caterpillar"),
    ("en", "chrysalis"),
    ("en", "pupa"),
    ("en", "egg"),
    ("zh", "蝴蝶"),
    ("zh", "凤蝶"),
    ("zh", "鳞翅目"),
    ("zh", "毛虫"),
    ("zh", "蛹"),
    ("zh", "卵"),
    ("es", "mariposa"),
    ("es", "mariposas"),
    ("es", "lepidóptero"),
    ("es", "oruga"),
    ("es", "crisálida"),
    ("es", "pupa"),
    ("es", "huevo"),
    ("ar", "فراشة"),
    ("ar", "فراشات"),
    ("ar", "يرقة"),
    ("ar", "شرنقة"),
    ("ar", "بيضة"),
    ("id", "kupu-kupu"),
    ("id", "kupukupu"),
    ("id", "ulat"),
    ("id", "kepompong"),
    ("id", "telur"),
    ("pt", "borboleta"),
    ("pt", "borboletas"),
    ("pt", "lagarta"),
    ("pt", "crisálida"),
    ("pt", "pupa"),
    ("pt", "ovo"),
    ("fr", "papillon"),
    ("fr", "papillons"),
    ("fr", "chenille"),
    ("fr", "chrysalide"),
    ("fr", "œuf"),
    ("ja", "蝶"),
    ("ja", "チョウ"),
    ("ja", "蝶々"),
    ("ja", "アゲハチョウ"),
    ("ja", "毛虫"),
    ("ja", "蛹"),
    ("ja", "卵"),
    ("ru", "бабочка"),
    ("ru", "бабочки"),
    ("ru", "чешуекрылые"),
    ("ru", "гусеница"),
    ("ru", "куколка"),
    ("ru", "яйцо"),
    ("de", "Schmetterling"),
    ("de", "Schmetterlinge"),
    ("de", "Tagfalter"),
    ("de", "Raupe"),
    ("de", "Puppe"),
    ("de", "Ei"),
)


@dataclass(frozen=True)
class MultilingualSearchTerm:
    language: str
    term: str
    region: str | None = None
    bbox: str | None = None
    term_type: str | None = None
    term_confidence: str = "high"
    notes: str | None = None


@dataclass(frozen=True)
class FlickrQuery:
    term: str
    language: str
    search_field: SearchField
    lane: QueryLane
    page: int = 1
    per_page: int = COUNT_PROBE_PAGE_SIZE
    has_geo: int = 1
    bbox: str | None = None
    min_taken_date: str | None = None
    max_taken_date: str | None = None
    min_upload_date: str | None = None
    max_upload_date: str | None = None
    split_reason: str | None = None
    parent_total: int | None = None
    region: str | None = None
    term_type: str | None = None
    term_confidence: str | None = None
    notes: str | None = None
    parent_query_hash: str | None = None
    split_depth: int = 0
    bbox_index: int | None = None
    slice_index: int | None = None
    registry_version: str | None = None
    query_definition_id: str | None = None
    accepted_taxon_key: str | None = None
    accepted_scientific_name: str | None = None
    family_key: str | None = None
    genus_key: str | None = None
    species_key: str | None = None


@dataclass(frozen=True)
class QueryPlan:
    count_probes: tuple[FlickrQuery, ...]
    page_queries: tuple[FlickrQuery, ...]


def multilingual_seed_terms() -> tuple[MultilingualSearchTerm, ...]:
    seen: set[str] = set()
    terms: list[MultilingualSearchTerm] = []
    for language, term in MULTILINGUAL_SEED_TERMS:
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(MultilingualSearchTerm(language=language, term=term))
    return tuple(terms)


def build_count_probes(
    *,
    terms: Iterable[MultilingualSearchTerm],
    search_fields: Iterable[SearchField] = ("text", "tags"),
) -> tuple[FlickrQuery, ...]:
    return tuple(
        FlickrQuery(
            term=term.term,
            language=term.language,
            search_field=field,
            lane="count_probe",
            per_page=COUNT_PROBE_PAGE_SIZE,
            bbox=term.bbox,
            region=term.region,
            term_type=term.term_type,
            term_confidence=term.term_confidence,
            notes=term.notes,
        )
        for term in terms
        for field in search_fields
    )


def load_registry_flickr_queries(
    path: str | Path,
    *,
    start_date: str = DEFAULT_FIXED_SLICE_START_DATE,
    end_date: str = DEFAULT_FIXED_SLICE_END_DATE,
    slice_days: int = DEFAULT_FIXED_SLICE_DAYS,
) -> tuple[FlickrQuery, ...]:
    frame = pl.read_parquet(path)
    if frame.is_empty():
        return ()
    if "normalized_match_key" not in frame.columns:
        frame = frame.with_columns(pl.col("source_term").str.to_lowercase().alias("normalized_match_key"))
    rows = frame.filter(pl.col("enabled") if "enabled" in frame.columns else pl.lit(True)).sort(
        ["search_priority", "normalized_match_key", "query_definition_id"]
    ).to_dicts()
    queries: list[FlickrQuery] = []
    for row in rows:
        field = str(row.get("search_field") or "text")
        if field not in {"text", "tags"}:
            continue
        bbox = str(row.get("bbox") or "") or None
        for slice_index, (slice_start, slice_end) in enumerate(
            fixed_upload_date_slices(start_date=start_date, end_date=end_date, slice_days=slice_days)
        ):
            queries.append(
                FlickrQuery(
                    term=str(row.get("source_term") or row.get("normalized_query_term") or ""),
                    language=str(row.get("language") or "und"),
                    search_field=field,
                    lane="bbox_page" if bbox else "normal_page",
                    page=1,
                    per_page=BBOX_PAGE_SIZE if bbox else NORMAL_PAGE_SIZE,
                    has_geo=1 if bbox else 0,
                    bbox=bbox,
                    min_upload_date=slice_start,
                    max_upload_date=slice_end,
                    split_reason="upload_date",
                    region=str(row.get("region") or "") or None,
                    term_type=str(row.get("name_class") or "") or None,
                    term_confidence=str(row.get("confidence") or "") or None,
                    split_depth=1,
                    slice_index=slice_index,
                    registry_version=str(row.get("registry_version") or "") or None,
                    query_definition_id=str(row.get("query_definition_id") or "") or None,
                    accepted_taxon_key=str(row.get("accepted_taxon_key") or "") or None,
                    accepted_scientific_name=str(row.get("accepted_scientific_name") or "") or None,
                    family_key=str(row.get("family_key") or "") or None,
                    genus_key=str(row.get("genus_key") or "") or None,
                    species_key=str(row.get("species_key") or "") or None,
                )
            )
    return tuple(queries)


def plan_pages_from_count(probe: FlickrQuery, *, total: int) -> tuple[FlickrQuery, ...]:
    if total <= 0:
        return ()
    per_page = page_size_for_query(probe)
    lane: QueryLane = "bbox_page" if probe.bbox else "normal_page"
    return tuple(
        _page_query(probe, page=page, per_page=per_page, lane=lane)
        for page in range(1, ceil(total / per_page) + 1)
    )


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


def query_fits_result_window(total: int) -> bool:
    return total <= STABLE_RESULT_THRESHOLD


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
    if query_fits_result_window(total):
        return plan_pages_from_count(probe, total=total)
    return tuple(
        _copy_registry_provenance(page, probe)
        for page in plan_fixed_upload_slice_pages(
            term=probe.term,
            search_field=probe.search_field,
            start_date=probe.min_upload_date or DEFAULT_FIXED_SLICE_START_DATE,
            end_date=probe.max_upload_date or DEFAULT_FIXED_SLICE_END_DATE,
            language=probe.language,
        )
    )


def sort_queries_for_resume(queries: Iterable[FlickrQuery]) -> tuple[FlickrQuery, ...]:
    return _sort_queries(queries)


def query_sort_key(query: FlickrQuery) -> tuple[object, ...]:
    return _query_sort_key(query)


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
        query.split_depth,
        split_priority(query),
        query_date_kind(query),
        query_min_date(query),
        query_max_date(query),
        query.bbox_index if query.bbox_index is not None else 999999,
        query.slice_index if query.slice_index is not None else 999999,
        query.region or "",
        query.term.casefold(),
        lane_priority(query),
        query.page,
        _query_hash(query),
    )


def _split_with_bboxes(probe: FlickrQuery, *, total: int, bboxes: tuple[str, ...]) -> tuple[FlickrQuery, ...]:
    return tuple(_split_probe(probe, bbox=bbox, bbox_index=index, reason="bbox", total=total) for index, bbox in enumerate(bboxes))


def _split_with_narrower_terms(probe: FlickrQuery, *, total: int, narrower_terms: tuple[str, ...]) -> tuple[FlickrQuery, ...]:
    return tuple(_split_probe(probe, term=term, reason="narrower_term", total=total) for term in narrower_terms)


def _split_with_taken_ranges(probe: FlickrQuery, *, total: int, taken: tuple[tuple[str, str], ...]) -> tuple[FlickrQuery, ...]:
    return tuple(_split_probe(probe, min_taken_date=start, max_taken_date=end, reason="taken_date", total=total) for start, end in taken)


def _split_with_upload_ranges(probe: FlickrQuery, *, total: int, upload: tuple[tuple[str, str], ...]) -> tuple[FlickrQuery, ...]:
    return tuple(_split_probe(probe, min_upload_date=start, max_upload_date=end, reason="upload_date", total=total) for start, end in upload)


def split_high_volume_query(
    probe: FlickrQuery,
    *,
    total: int,
    taken_date_ranges: Iterable[tuple[str, str]] = (),
    upload_date_ranges: Iterable[tuple[str, str]] = (),
    bboxes: Iterable[str] = (),
    narrower_terms: Iterable[str] = (),
) -> tuple[FlickrQuery, ...]:
    if total <= STABLE_RESULT_THRESHOLD:
        return plan_pages_from_count(probe, total=total)
    taken = tuple(taken_date_ranges)
    if taken:
        return _sort_queries(_split_with_taken_ranges(probe, total=total, taken=taken))
    upload = tuple(upload_date_ranges)
    if upload:
        return _sort_queries(_split_with_upload_ranges(probe, total=total, upload=upload))
    bbox_values = tuple(bboxes)
    if bbox_values:
        return _sort_queries(_split_with_bboxes(probe, total=total, bboxes=bbox_values))
    return _sort_queries(_split_with_narrower_terms(probe, total=total, narrower_terms=tuple(narrower_terms)))


def build_worldwide_discovery_plan(*, terms: Iterable[MultilingualSearchTerm]) -> QueryPlan:
    probes = build_count_probes(terms=terms)
    return QueryPlan(count_probes=probes, page_queries=())


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
    for key in ("bbox", "min_taken_date", "max_taken_date", "min_upload_date", "max_upload_date"):
        value = getattr(query, key)
        if value:
            params[key] = value
    return params


def deduplicate_photo_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
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
        language=probe.language,
        search_field=probe.search_field,
        lane=lane,
        page=page,
        per_page=per_page,
        has_geo=probe.has_geo,
        bbox=probe.bbox,
        min_taken_date=probe.min_taken_date,
        max_taken_date=probe.max_taken_date,
        min_upload_date=probe.min_upload_date,
        max_upload_date=probe.max_upload_date,
        split_reason=probe.split_reason,
        parent_total=probe.parent_total,
        region=probe.region,
        term_type=probe.term_type,
        term_confidence=probe.term_confidence,
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
    )


def _split_probe(
    probe: FlickrQuery,
    *,
    reason: str,
    total: int,
    term: str | None = None,
    bbox: str | None = None,
    min_taken_date: str | None = None,
    max_taken_date: str | None = None,
    min_upload_date: str | None = None,
    max_upload_date: str | None = None,
    bbox_index: int | None = None,
) -> FlickrQuery:
    return FlickrQuery(
        term=term or probe.term,
        language=probe.language,
        search_field=probe.search_field,
        lane="count_probe",
        per_page=COUNT_PROBE_PAGE_SIZE,
        has_geo=probe.has_geo,
        bbox=bbox or probe.bbox,
        min_taken_date=min_taken_date or probe.min_taken_date,
        max_taken_date=max_taken_date or probe.max_taken_date,
        min_upload_date=min_upload_date or probe.min_upload_date,
        max_upload_date=max_upload_date or probe.max_upload_date,
        split_reason=reason,
        parent_total=total,
        region=probe.region,
        term_type=probe.term_type,
        term_confidence=probe.term_confidence,
        notes=probe.notes,
        parent_query_hash=probe.parent_query_hash or _query_hash(probe),
        split_depth=probe.split_depth + 1,
        bbox_index=bbox_index if bbox_index is not None else probe.bbox_index,
        slice_index=probe.slice_index,
        registry_version=probe.registry_version,
        query_definition_id=probe.query_definition_id,
        accepted_taxon_key=probe.accepted_taxon_key,
        accepted_scientific_name=probe.accepted_scientific_name,
        family_key=probe.family_key,
        genus_key=probe.genus_key,
        species_key=probe.species_key,
    )


def _copy_registry_provenance(query: FlickrQuery, source: FlickrQuery) -> FlickrQuery:
    return replace(
        query,
        registry_version=source.registry_version,
        query_definition_id=source.query_definition_id,
        accepted_taxon_key=source.accepted_taxon_key,
        accepted_scientific_name=source.accepted_scientific_name,
        family_key=source.family_key,
        genus_key=source.genus_key,
        species_key=source.species_key,
    )


def _query_hash(query: FlickrQuery) -> str:
    import hashlib

    payload = json.dumps(query.__dict__, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
