from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Iterable, Literal


SearchField = Literal["text", "tags"]
QueryLane = Literal["count_probe", "normal_page", "bbox_page"]

NORMAL_PAGE_SIZE = 500
BBOX_PAGE_SIZE = 250
COUNT_PROBE_PAGE_SIZE = 1
SPLIT_TOTAL_THRESHOLD = 3500

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
    terms: Iterable[MultilingualSearchTerm] | None = None,
    search_fields: Iterable[SearchField] = ("text", "tags"),
) -> tuple[FlickrQuery, ...]:
    return tuple(
        FlickrQuery(
            term=term.term,
            language=term.language,
            search_field=field,
            lane="count_probe",
            per_page=COUNT_PROBE_PAGE_SIZE,
        )
        for term in (terms or multilingual_seed_terms())
        for field in search_fields
    )


def plan_pages_from_count(probe: FlickrQuery, *, total: int) -> tuple[FlickrQuery, ...]:
    if total <= 0:
        return ()
    per_page = BBOX_PAGE_SIZE if probe.bbox else NORMAL_PAGE_SIZE
    lane: QueryLane = "bbox_page" if probe.bbox else "normal_page"
    return tuple(
        _page_query(probe, page=page, per_page=per_page, lane=lane)
        for page in range(1, ceil(total / per_page) + 1)
    )


def split_high_volume_query(
    probe: FlickrQuery,
    *,
    total: int,
    taken_date_ranges: Iterable[tuple[str, str]] = (),
    upload_date_ranges: Iterable[tuple[str, str]] = (),
    bboxes: Iterable[str] = (),
    narrower_terms: Iterable[str] = (),
) -> tuple[FlickrQuery, ...]:
    if total <= SPLIT_TOTAL_THRESHOLD:
        return plan_pages_from_count(probe, total=total)
    taken = tuple(taken_date_ranges)
    if taken:
        return tuple(_split_probe(probe, min_taken_date=start, max_taken_date=end, reason="taken_date", total=total) for start, end in taken)
    upload = tuple(upload_date_ranges)
    if upload:
        return tuple(_split_probe(probe, min_upload_date=start, max_upload_date=end, reason="upload_date", total=total) for start, end in upload)
    bbox_values = tuple(bboxes)
    if bbox_values:
        return tuple(_split_probe(probe, bbox=bbox, reason="bbox", total=total) for bbox in bbox_values)
    return tuple(_split_probe(probe, term=term, reason="narrower_term", total=total) for term in narrower_terms)


def build_worldwide_discovery_plan() -> QueryPlan:
    probes = build_count_probes()
    return QueryPlan(count_probes=probes, page_queries=())


def flickr_search_params(query: FlickrQuery) -> dict[str, str | int]:
    params: dict[str, str | int] = {
        query.search_field: query.term,
        "has_geo": query.has_geo,
        "media": "photos",
        "content_types": "0",
        "safe_search": 1,
        "extras": "url_l,url_m",
        "per_page": query.per_page,
        "page": query.page,
    }
    for key in ("bbox", "min_taken_date", "max_taken_date", "min_upload_date", "max_upload_date"):
        value = getattr(query, key)
        if value:
            params[key] = value
    return params


def deduplicate_photo_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str | None, str | None]] = set()
    unique: list[dict[str, Any]] = []
    for record in records:
        key = (str(record.get("id") or record.get("flickr_photo_id") or ""), str(record.get("url_l") or record.get("url_m") or record.get("image_url") or ""))
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
        bbox=probe.bbox,
        min_taken_date=probe.min_taken_date,
        max_taken_date=probe.max_taken_date,
        min_upload_date=probe.min_upload_date,
        max_upload_date=probe.max_upload_date,
        split_reason=probe.split_reason,
        parent_total=probe.parent_total,
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
) -> FlickrQuery:
    return FlickrQuery(
        term=term or probe.term,
        language=probe.language,
        search_field=probe.search_field,
        lane="count_probe",
        per_page=COUNT_PROBE_PAGE_SIZE,
        bbox=bbox or probe.bbox,
        min_taken_date=min_taken_date or probe.min_taken_date,
        max_taken_date=max_taken_date or probe.max_taken_date,
        min_upload_date=min_upload_date or probe.min_upload_date,
        max_upload_date=max_upload_date or probe.max_upload_date,
        split_reason=reason,
        parent_total=total,
    )
