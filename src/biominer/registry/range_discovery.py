from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import polars as pl


HTTPGet = Callable[[str, dict[str, object]], dict[str, Any]]
RANGE_SEED_SCHEMA_VERSION = "range-seed-v1"
RANGE_COUNTRIES_FILE = "range_countries.parquet"


@dataclass(frozen=True)
class OccurrenceCountryFacet:
    country_code: str
    occurrence_count: int


@dataclass(frozen=True)
class AcceptedSpeciesResolution:
    accepted_taxon_key: str
    scientific_name: str
    match_type: str
    confidence: int


@dataclass(frozen=True)
class OccurrenceCountryDetails:
    country_code: str
    georeferenced_count: int | None
    basis_of_record_counts: dict[str, int]
    first_year: int | None
    last_year: int | None


@dataclass(frozen=True)
class RangeSeedCountry:
    country_code: str
    country_name: str
    region: str
    range_status: str
    requires_occurrence_support: bool
    taxonomic_caution: bool


@dataclass(frozen=True)
class RangeSeed:
    accepted_taxon_key: str
    scientific_name: str
    countries: dict[str, RangeSeedCountry]


class GBIFOccurrenceCountryClient:
    def __init__(self, *, http_get: HTTPGet | None = None) -> None:
        self._http_get = http_get or _http_get
        self.call_count = 0

    def resolve_accepted_species_key(self, scientific_name: str) -> AcceptedSpeciesResolution:
        payload = self._get_object(
            "/species/match",
            {"name": scientific_name, "rank": "SPECIES", "strict": "false"},
        )
        accepted_key = payload.get("acceptedUsageKey") or payload.get("usageKey")
        if accepted_key is None:
            raise ValueError(f"GBIF did not resolve species {scientific_name!r}")
        rank = str(payload.get("rank") or "")
        if rank and rank != "SPECIES":
            raise ValueError(f"GBIF resolved {scientific_name!r} to rank {rank!r}, expected SPECIES")
        return AcceptedSpeciesResolution(
            accepted_taxon_key=f"gbif:{accepted_key}",
            scientific_name=str(payload.get("canonicalName") or payload.get("scientificName") or scientific_name),
            match_type=str(payload.get("matchType") or ""),
            confidence=_int_or_zero(payload.get("confidence")),
        )

    def country_facets(self, accepted_taxon_key: str, *, facet_limit: int = 300) -> tuple[OccurrenceCountryFacet, ...]:
        params: dict[str, object] = {
            "taxonKey": _bare_gbif_key(accepted_taxon_key),
            "limit": 0,
            "facet": "country",
            "facetLimit": facet_limit,
        }
        payload = self._get_object("/occurrence/search", params)
        rows: list[OccurrenceCountryFacet] = []
        for item in _facet_counts(payload, "COUNTRY"):
            country_code = _country_code(item.get("name"))
            if not country_code:
                continue
            rows.append(OccurrenceCountryFacet(country_code=country_code, occurrence_count=_int_or_zero(item.get("count"))))
        return tuple(rows)

    def country_details(self, accepted_taxon_key: str, country_code: str) -> OccurrenceCountryDetails:
        code = _country_code(country_code)
        detail_payload = self._get_object(
            "/occurrence/search",
            {
                "taxonKey": _bare_gbif_key(accepted_taxon_key),
                "country": code,
                "limit": 0,
                "facet": ["basisOfRecord", "year"],
                "facetLimit": 1000,
            },
        )
        georef_payload = self._get_object(
            "/occurrence/search",
            {
                "taxonKey": _bare_gbif_key(accepted_taxon_key),
                "country": code,
                "hasCoordinate": "true",
                "limit": 0,
            },
        )
        years = sorted(
            year
            for year in (_int_or_none(item.get("name")) for item in _facet_counts(detail_payload, "YEAR"))
            if year is not None
        )
        return OccurrenceCountryDetails(
            country_code=code,
            georeferenced_count=_int_or_zero(georef_payload.get("count")),
            basis_of_record_counts={
                str(item.get("name") or ""): _int_or_zero(item.get("count"))
                for item in _facet_counts(detail_payload, "BASIS_OF_RECORD")
                if item.get("name")
            },
            first_year=years[0] if years else None,
            last_year=years[-1] if years else None,
        )

    def _get_object(self, path: str, params: dict[str, object]) -> dict[str, Any]:
        self.call_count += 1
        payload = self._http_get(path, params)
        if not isinstance(payload, dict):
            raise ValueError(f"GBIF occurrence response for {path} must be a JSON object")
        return payload


def discover_range_countries(
    *,
    accepted_taxon_key: str,
    scientific_name: str,
    client: GBIFOccurrenceCountryClient,
    seed_json: str | Path | None = None,
    retrieved_at: str | None = None,
    low_count_threshold: int = 2,
    recent_year_cutoff: int | None = None,
    facet_limit: int = 300,
) -> pl.DataFrame:
    seed = load_range_seed(seed_json) if seed_json is not None else RangeSeed("", "", {})
    if seed.accepted_taxon_key and seed.accepted_taxon_key != accepted_taxon_key:
        raise ValueError(f"range seed belongs to {seed.accepted_taxon_key}, not {accepted_taxon_key}")
    retrieved = retrieved_at or datetime.now(UTC).isoformat()
    recent_cutoff = recent_year_cutoff if recent_year_cutoff is not None else datetime.now(UTC).year - 10
    rows: list[dict[str, object]] = []
    for facet in client.country_facets(accepted_taxon_key, facet_limit=facet_limit):
        details = client.country_details(accepted_taxon_key, facet.country_code)
        seed_country = seed.countries.get(facet.country_code)
        range_status = _range_status(
            facet,
            seed_country,
            accepted_taxon_key=accepted_taxon_key,
            seed=seed,
            low_count_threshold=low_count_threshold,
        )
        has_recent_records = details.last_year is not None and details.last_year >= recent_cutoff
        rows.append(
            {
                "accepted_taxon_key": accepted_taxon_key,
                "scientific_name": scientific_name,
                "source": "GBIF",
                "source_taxon_key": accepted_taxon_key,
                "country_code": facet.country_code,
                "country_name": seed_country.country_name if seed_country else facet.country_code,
                "admin1_code": "",
                "admin1_name": "",
                "occurrence_count": facet.occurrence_count,
                "georeferenced_count": details.georeferenced_count,
                "basis_of_record_counts_json": json.dumps(details.basis_of_record_counts, sort_keys=True, separators=(",", ":")),
                "first_year": details.first_year,
                "last_year": details.last_year,
                "has_recent_records": has_recent_records,
                "range_status": range_status,
                "confidence": _confidence(facet.occurrence_count, range_status=range_status, low_count_threshold=low_count_threshold),
                "taxonomic_caution": range_status == "taxonomically_cautionary" or bool(seed_country and seed_country.taxonomic_caution),
                "retrieved_at": retrieved,
                "source_query_hash": _source_query_hash(accepted_taxon_key, facet.country_code),
                "region": seed_country.region if seed_country else "",
            }
        )
    return pl.DataFrame(rows, schema=_range_countries_schema()).sort("country_code")


def write_range_countries(frame: pl.DataFrame, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / RANGE_COUNTRIES_FILE
    frame.write_parquet(path)
    return path


def load_range_seed(path: str | Path) -> RangeSeed:
    seed_path = Path(path)
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    if str(payload.get("schema_version") or "") != RANGE_SEED_SCHEMA_VERSION:
        raise ValueError(f"range seed schema_version must be {RANGE_SEED_SCHEMA_VERSION}")
    countries: dict[str, RangeSeedCountry] = {}
    for region in payload.get("regions") or []:
        if not isinstance(region, dict):
            continue
        region_name = str(region.get("region") or "")
        range_status = str(region.get("range_status") or "occurrence_supported")
        requires_occurrence_support = _boolish(region.get("requires_occurrence_support", False))
        taxonomic_caution = _boolish(region.get("taxonomic_caution", False)) or range_status == "taxonomically_cautionary"
        for country in region.get("countries") or []:
            if not isinstance(country, dict):
                continue
            code = _country_code(country.get("code"))
            if not code:
                continue
            countries[code] = RangeSeedCountry(
                country_code=code,
                country_name=str(country.get("name") or code),
                region=region_name,
                range_status=range_status,
                requires_occurrence_support=requires_occurrence_support,
                taxonomic_caution=taxonomic_caution,
            )
    return RangeSeed(
        accepted_taxon_key=str(payload.get("accepted_taxon_key") or ""),
        scientific_name=str(payload.get("scientific_name") or ""),
        countries=countries,
    )


def _range_status(
    facet: OccurrenceCountryFacet,
    seed_country: RangeSeedCountry | None,
    *,
    accepted_taxon_key: str,
    seed: RangeSeed,
    low_count_threshold: int,
) -> str:
    if seed_country and seed_country.taxonomic_caution and accepted_taxon_key == seed.accepted_taxon_key:
        return "taxonomically_cautionary"
    if facet.occurrence_count < low_count_threshold:
        return "single_or_uncertain_record"
    if seed_country is None:
        return "occurrence_supported"
    if seed_country.requires_occurrence_support and facet.occurrence_count <= 0:
        return "predicted_suitable_not_recorded"
    return seed_country.range_status


def _confidence(occurrence_count: int, *, range_status: str, low_count_threshold: int) -> str:
    if range_status == "taxonomically_cautionary":
        return "taxonomic_caution"
    if occurrence_count < low_count_threshold:
        return "low"
    if occurrence_count >= 10:
        return "high"
    return "medium"


def _facet_counts(payload: dict[str, Any], field: str) -> list[dict[str, Any]]:
    for facet in payload.get("facets") or []:
        if not isinstance(facet, dict):
            continue
        if str(facet.get("field") or "").upper() != field:
            continue
        counts = facet.get("counts") or []
        return [item for item in counts if isinstance(item, dict)]
    return []


def _range_countries_schema() -> dict[str, pl.DataType]:
    return {
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "source": pl.String,
        "source_taxon_key": pl.String,
        "country_code": pl.String,
        "country_name": pl.String,
        "admin1_code": pl.String,
        "admin1_name": pl.String,
        "occurrence_count": pl.Int64,
        "georeferenced_count": pl.Int64,
        "basis_of_record_counts_json": pl.String,
        "first_year": pl.Int64,
        "last_year": pl.Int64,
        "has_recent_records": pl.Boolean,
        "range_status": pl.String,
        "confidence": pl.String,
        "taxonomic_caution": pl.Boolean,
        "retrieved_at": pl.String,
        "source_query_hash": pl.String,
        "region": pl.String,
    }


def _source_query_hash(accepted_taxon_key: str, country_code: str) -> str:
    payload = json.dumps({"accepted_taxon_key": accepted_taxon_key, "country_code": country_code, "source": "GBIF"}, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bare_gbif_key(value: str) -> str:
    return str(value).removeprefix("gbif:")


def _country_code(value: object) -> str:
    code = str(value or "").strip().upper()
    return code if len(code) == 2 and code.isalpha() else ""


def _int_or_zero(value: object) -> int:
    parsed = _int_or_none(value)
    return parsed if parsed is not None else 0


def _int_or_none(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "enabled"}


def _http_get(path: str, params: dict[str, object]) -> dict[str, Any]:
    import httpx

    with httpx.Client(timeout=30) as client:
        response = client.get(f"https://api.gbif.org/v1{path}", params=params)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"GBIF occurrence response for {path} must be a JSON object")
    return payload


__all__ = [
    "AcceptedSpeciesResolution",
    "GBIFOccurrenceCountryClient",
    "OccurrenceCountryDetails",
    "OccurrenceCountryFacet",
    "RANGE_COUNTRIES_FILE",
    "RangeSeed",
    "RangeSeedCountry",
    "discover_range_countries",
    "load_range_seed",
    "write_range_countries",
]
