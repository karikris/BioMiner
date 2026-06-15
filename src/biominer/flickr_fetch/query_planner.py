from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json
from math import ceil
from pathlib import Path
from typing import Any, Iterable, Literal


SearchField = Literal["text", "tags"]
QueryLane = Literal["count_probe", "normal_page", "bbox_page"]

NORMAL_PAGE_SIZE = 500
GEO_PAGE_SIZE = 250
BBOX_PAGE_SIZE = GEO_PAGE_SIZE
COUNT_PROBE_PAGE_SIZE = 1
FLICKR_SEARCH_RESULT_WINDOW = 4000
STABLE_RESULT_THRESHOLD = FLICKR_SEARCH_RESULT_WINDOW
MAX_RESULT_PAGES_PER_QUERY = FLICKR_SEARCH_RESULT_WINDOW
MAX_RESULTS_PER_QUERY = FLICKR_SEARCH_RESULT_WINDOW
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
PAPILIO_DEMOLEUS_ANCHOR_TERMS = (
    "Papilio demoleus",
    "demoleus",
    "lime butterfly",
    "common lime butterfly",
)
DEFAULT_UPLOAD_DATE_RANGES: tuple[tuple[str, str], ...] = (
    ("2004-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2026-12-31"),
)
DEFAULT_FIXED_SLICE_START_DATE = "2004-02-10"
DEFAULT_FIXED_SLICE_END_DATE = date.today().isoformat()
DEFAULT_COARSE_SLICE_END_DATE = "2015-12-31"
DEFAULT_COARSE_SLICE_DAYS = 10
DEFAULT_FIXED_SLICE_DAYS = 5
DEFAULT_FIXED_SLICE_PAGES = FLICKR_SEARCH_RESULT_WINDOW // NORMAL_PAGE_SIZE
PAPILIO_DEMOLEUS_REGION_BBOXES: dict[str, str] = {
    "India": "68.11,6.55,97.40,35.67",
    "Pakistan": "60.87,23.63,77.84,37.08",
    "Bangladesh": "88.01,20.59,92.67,26.63",
    "Sri Lanka": "79.52,5.92,81.88,9.84",
    "Nepal": "80.06,26.35,88.20,30.45",
    "Myanmar": "92.17,9.78,101.17,28.55",
    "Thailand": "97.34,5.61,105.64,20.46",
    "Malaysia": "99.64,0.85,119.27,7.36",
    "Singapore": "103.59,1.13,104.10,1.48",
    "Indonesia": "95.01,-11.01,141.02,6.08",
    "Philippines": "116.93,4.59,126.60,21.12",
    "Vietnam": "102.14,8.18,109.47,23.39",
    "Cambodia": "102.33,10.41,107.63,14.69",
    "Laos": "100.08,13.91,107.70,22.51",
    "China": "73.50,18.16,134.77,53.56",
    "Taiwan": "119.31,21.89,122.00,25.30",
    "Hong Kong": "113.82,22.13,114.43,22.57",
    "Japan": "129.00,24.00,146.00,46.00",
    "Australia": "112.92,-43.74,153.64,-10.05",
    "Papua New Guinea": "140.84,-11.66,156.02,-1.32",
    "Iraq": "38.79,29.06,48.58,37.39",
    "Iran": "44.03,25.06,63.33,39.78",
    "Saudi Arabia": "34.50,16.29,55.67,32.16",
    "United Arab Emirates": "51.58,22.63,56.40,26.08",
    "Oman": "52.00,16.64,59.84,26.39",
    "Yemen": "42.55,12.11,54.54,18.99",
    "Israel": "34.22,29.45,35.90,33.34",
    "Jordan": "34.96,29.19,39.30,33.38",
    "Egypt": "24.70,22.00,36.90,31.67",
    "Seychelles": "46.20,-10.50,56.60,-3.70",
    "Mauritius": "56.50,-20.70,63.60,-10.00",
    "Dominican Republic": "-72.01,17.36,-68.32,19.98",
    "Haiti": "-74.48,18.02,-71.62,20.09",
    "Jamaica": "-78.37,17.64,-76.18,18.53",
    "Caribbean": "-89.50,9.00,-59.00,27.00",
    "Florida": "-87.64,24.40,-79.97,31.00",
    "Central America": "-92.30,7.00,-77.00,18.50",
    "South America": "-81.40,-55.98,-34.70,13.40",
}

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
            bbox=term.bbox,
            region=term.region,
            term_type=term.term_type,
            term_confidence=term.term_confidence,
            notes=term.notes,
        )
        for term in (terms or multilingual_seed_terms())
        for field in search_fields
    )


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
    pages_per_slice: int = DEFAULT_FIXED_SLICE_PAGES,
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
        for page in range(1, pages_per_slice + 1):
            pages.append(
                FlickrQuery(
                    term=term,
                    language=language,
                    search_field=search_field,
                    lane="normal_page",
                    page=page,
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


def query_fits_page_limit(total: int, *, per_page: int = NORMAL_PAGE_SIZE, max_pages: int = MAX_RESULT_PAGES_PER_QUERY) -> bool:
    return total <= STABLE_RESULT_THRESHOLD and result_pages_for_total(total, per_page=per_page) <= max_pages


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
    max_pages: int = MAX_RESULT_PAGES_PER_QUERY,
) -> tuple[FlickrQuery, ...]:
    if total <= STABLE_RESULT_THRESHOLD and query_fits_page_limit(total, per_page=page_size_for_query(probe), max_pages=max_pages):
        return plan_pages_from_count(probe, total=total)
    return plan_fixed_upload_slice_pages(
        term=probe.term,
        search_field=probe.search_field,
        start_date=probe.min_upload_date or DEFAULT_FIXED_SLICE_START_DATE,
        end_date=probe.max_upload_date or DEFAULT_FIXED_SLICE_END_DATE,
        language=probe.language,
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


def build_worldwide_discovery_plan() -> QueryPlan:
    probes = build_count_probes()
    return QueryPlan(count_probes=probes, page_queries=())


def load_papilio_demoleus_terms_from_json(path: str | Path) -> tuple[MultilingualSearchTerm, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = _keyword_entries(data)
    terms: list[MultilingualSearchTerm] = []
    seen: set[tuple[str, str, str | None, str | None]] = set()
    for entry in entries:
        if not entry.get("use_for_flickr", True):
            continue
        raw_term = _clean_term(entry.get("term"))
        if not raw_term:
            continue
        regions = entry.get("regions") or [None]
        for region in regions:
            confidence = _term_confidence(entry)
            term = _gated_term(raw_term, confidence=confidence)
            key = (term.casefold(), str(entry.get("language") or "und"), region, entry.get("term_type"))
            if key in seen:
                continue
            seen.add(key)
            terms.append(
                MultilingualSearchTerm(
                    language=str(entry.get("language") or "und"),
                    term=term,
                    region=region,
                    bbox=PAPILIO_DEMOLEUS_REGION_BBOXES.get(str(region)) if region else None,
                    term_type=str(entry.get("term_type") or "unknown"),
                    term_confidence=confidence,
                    notes=_notes(entry, raw_term=raw_term, gated_term=term, confidence=confidence),
                )
            )
    return tuple(terms)


def build_papilio_demoleus_count_probes_from_json(
    path: str | Path,
    *,
    search_fields: Iterable[SearchField] = ("text", "tags"),
) -> tuple[FlickrQuery, ...]:
    return build_count_probes(terms=load_papilio_demoleus_terms_from_json(path), search_fields=search_fields)


def papilio_demoleus_known_region_for_coordinate(latitude: float, longitude: float) -> str | None:
    for region, bbox in PAPILIO_DEMOLEUS_REGION_BBOXES.items():
        if coordinate_in_bbox(latitude=latitude, longitude=longitude, bbox=bbox):
            return region
    return None


def outside_known_papilio_demoleus_regions(record: dict[str, Any]) -> bool | None:
    latitude = _optional_float(record.get("latitude", record.get("decimalLatitude")))
    longitude = _optional_float(record.get("longitude", record.get("decimalLongitude")))
    if latitude is None or longitude is None:
        return None
    return papilio_demoleus_known_region_for_coordinate(latitude, longitude) is None


def coordinate_in_bbox(*, latitude: float, longitude: float, bbox: str) -> bool:
    min_lon, min_lat, max_lon, max_lat = (float(value) for value in bbox.split(","))
    return min_lat <= latitude <= max_lat and min_lon <= longitude <= max_lon


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
    )


def _keyword_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    groups = data.get("dictionary_groups", {})
    entries: list[dict[str, Any]] = []
    for group in groups.values():
        if isinstance(group, list):
            entries.extend(item for item in group if isinstance(item, dict))
        elif isinstance(group, dict):
            for region, items in group.items():
                region_items = items if isinstance(items, list) else ()
                for item in region_items:
                    if isinstance(item, dict):
                        regions = item.get("regions") or [region]
                        entries.append({**item, "regions": regions})
    return entries


def _clean_term(value: object) -> str:
    return " ".join(str(value or "").replace('"', "").split())


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _term_confidence(entry: dict[str, Any]) -> str:
    precision = str(entry.get("precision_tier") or "").casefold()
    confidence = str(entry.get("confidence") or "").casefold()
    term_type = str(entry.get("term_type") or "").casefold()
    if precision == "low" or confidence == "low" or term_type in {"broad_butterfly", "host_plant", "life_stage", "pest_context"}:
        return "broad"
    return confidence or "medium"


def _gated_term(term: str, *, confidence: str) -> str:
    if confidence != "broad":
        return term
    normalized = term.casefold()
    if any(anchor.casefold() in normalized for anchor in PAPILIO_DEMOLEUS_ANCHOR_TERMS):
        return term
    return f"{PAPILIO_DEMOLEUS_ANCHOR_TERMS[0]} {term}"


def _notes(entry: dict[str, Any], *, raw_term: str, gated_term: str, confidence: str) -> str | None:
    values = [str(entry.get("notes") or "").strip()]
    if confidence == "broad" and raw_term != gated_term:
        values.append("Broad/uncertain term gated with Papilio demoleus anchor for Flickr search precision.")
    return " ".join(value for value in values if value) or None


def _year_ranges(start: str, end: str) -> tuple[tuple[str, str], ...]:
    if len(start) < 4 or len(end) < 4 or not start[:4].isdigit() or not end[:4].isdigit():
        return ()
    start_year = int(start[:4])
    end_year = int(end[:4])
    if start_year >= end_year:
        return ()
    return tuple((f"{year}-01-01", f"{year}-12-31") for year in range(start_year, end_year + 1))


def _query_hash(query: FlickrQuery) -> str:
    import hashlib

    payload = json.dumps(query.__dict__, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
