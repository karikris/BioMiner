from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from biominer.registry.range_discovery import (
    AcceptedSpeciesResolution,
    GBIFOccurrenceCountryClient,
    OccurrenceCountryDetails,
    OccurrenceCountryFacet,
    discover_range_countries,
    write_range_countries,
)


class FakeGBIFHTTP:
    def __init__(self, payloads: dict[tuple[str, str], dict[str, object]]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, path: str, params: dict[str, object]) -> dict[str, object]:
        self.calls.append((path, dict(params)))
        country = str(params.get("country") or "")
        return self.payloads[(path, country)]


class FakeOccurrenceClient:
    def __init__(
        self,
        facets: tuple[OccurrenceCountryFacet, ...],
        details: dict[str, OccurrenceCountryDetails] | None = None,
    ) -> None:
        self.facets = facets
        self.details = details or {}

    def country_facets(self, accepted_taxon_key: str, *, facet_limit: int = 300) -> tuple[OccurrenceCountryFacet, ...]:
        return self.facets

    def country_details(self, accepted_taxon_key: str, country_code: str) -> OccurrenceCountryDetails:
        return self.details.get(
            country_code,
            OccurrenceCountryDetails(
                country_code=country_code,
                georeferenced_count=None,
                basis_of_record_counts={},
                first_year=None,
                last_year=None,
            ),
        )


def test_gbif_occurrence_country_facets_normalize_country_codes() -> None:
    http = FakeGBIFHTTP(
        {
            (
                "/occurrence/search",
                "",
            ): {
                "count": 17,
                "facets": [
                    {
                        "field": "COUNTRY",
                        "counts": [
                            {"name": " in ", "count": 12},
                            {"name": "id", "count": 5},
                            {"name": "", "count": 1},
                        ],
                    }
                ],
            }
        }
    )

    rows = GBIFOccurrenceCountryClient(http_get=http).country_facets("gbif:1938069", facet_limit=10)

    assert rows == (
        OccurrenceCountryFacet(country_code="IN", occurrence_count=12),
        OccurrenceCountryFacet(country_code="ID", occurrence_count=5),
    )
    assert http.calls == [
        (
            "/occurrence/search",
            {"taxonKey": "1938069", "limit": 0, "facet": "country", "facetLimit": 10},
        )
    ]


def test_gbif_occurrence_client_resolves_accepted_species_key() -> None:
    http = FakeGBIFHTTP(
        {
            (
                "/species/match",
                "",
            ): {
                "usageKey": 1938069,
                "acceptedUsageKey": 1938069,
                "canonicalName": "Papilio demoleus",
                "rank": "SPECIES",
                "matchType": "EXACT",
                "confidence": 99,
            }
        }
    )

    resolution = GBIFOccurrenceCountryClient(http_get=http).resolve_accepted_species_key("Papilio demoleus")

    assert resolution == AcceptedSpeciesResolution(
        accepted_taxon_key="gbif:1938069",
        scientific_name="Papilio demoleus",
        match_type="EXACT",
        confidence=99,
    )
    assert http.calls == [
        (
            "/species/match",
            {"name": "Papilio demoleus", "rank": "SPECIES", "strict": "false"},
        )
    ]


def test_range_discovery_marks_low_count_countries_single_or_uncertain(tmp_path) -> None:
    seed = _write_seed(
        tmp_path,
        [
            {
                "region": "South Asia",
                "countries": [{"code": "IN", "name": "India"}, {"code": "BT", "name": "Bhutan"}],
                "range_status": "native_or_long_established",
            }
        ],
    )
    client = FakeOccurrenceClient(
        (
            OccurrenceCountryFacet(country_code="IN", occurrence_count=25),
            OccurrenceCountryFacet(country_code="BT", occurrence_count=1),
        )
    )

    frame = discover_range_countries(
        accepted_taxon_key="gbif:1938069",
        scientific_name="Papilio demoleus",
        client=client,
        seed_json=seed,
        retrieved_at="2026-07-07T00:00:00+00:00",
        low_count_threshold=2,
    )

    rows = {row["country_code"]: row for row in frame.sort("country_code").to_dicts()}
    assert rows["IN"]["range_status"] == "native_or_long_established"
    assert rows["BT"]["range_status"] == "single_or_uncertain_record"
    assert rows["BT"]["confidence"] == "low"


def test_caribbean_seed_requires_occurrence_support_for_introduced_established(tmp_path) -> None:
    seed = _write_seed(
        tmp_path,
        [
            {
                "region": "Caribbean introduced range",
                "countries": [{"code": "DO", "name": "Dominican Republic"}, {"code": "HT", "name": "Haiti"}],
                "range_status": "introduced_established",
                "requires_occurrence_support": True,
            }
        ],
    )
    client = FakeOccurrenceClient(
        (OccurrenceCountryFacet(country_code="DO", occurrence_count=6),),
        details={
            "DO": OccurrenceCountryDetails(
                country_code="DO",
                georeferenced_count=4,
                basis_of_record_counts={"HUMAN_OBSERVATION": 6},
                first_year=2020,
                last_year=2025,
            )
        },
    )

    frame = discover_range_countries(
        accepted_taxon_key="gbif:1938069",
        scientific_name="Papilio demoleus",
        client=client,
        seed_json=seed,
        retrieved_at="2026-07-07T00:00:00+00:00",
    )

    assert frame.select("country_code").to_series().to_list() == ["DO"]
    row = frame.to_dicts()[0]
    assert row["range_status"] == "introduced_established"
    assert row["has_recent_records"] is True
    assert row["basis_of_record_counts_json"] == '{"HUMAN_OBSERVATION":6}'


def test_papilio_demoleus_seed_marks_australia_new_guinea_taxonomically_cautionary() -> None:
    client = FakeOccurrenceClient(
        (
            OccurrenceCountryFacet(country_code="AU", occurrence_count=3425),
            OccurrenceCountryFacet(country_code="PG", occurrence_count=12),
        )
    )

    frame = discover_range_countries(
        accepted_taxon_key="gbif:1938069",
        scientific_name="Papilio demoleus",
        client=client,
        seed_json=Path("config/range_seed/papilio_demoleus.json"),
        retrieved_at="2026-07-07T00:00:00+00:00",
    )

    assert frame.select("country_code").to_series().to_list() == ["AU", "PG"]
    assert frame.select("range_status").to_series().to_list() == ["taxonomically_cautionary", "taxonomically_cautionary"]
    assert frame.select("taxonomic_caution").to_series().to_list() == [True, True]


def test_species_specific_range_seed_rejects_accepted_taxon_mismatch(tmp_path) -> None:
    seed = _write_seed(
        tmp_path,
        [
            {
                "region": "South Asia",
                "countries": [{"code": "IN", "name": "India"}],
                "range_status": "native_or_long_established",
            }
        ],
    )

    with pytest.raises(ValueError, match="range seed belongs to gbif:1938069"):
        discover_range_countries(
            accepted_taxon_key="gbif:999",
            scientific_name="Papilio other",
            client=FakeOccurrenceClient((OccurrenceCountryFacet(country_code="IN", occurrence_count=25),)),
            seed_json=seed,
            retrieved_at="2026-07-07T00:00:00+00:00",
        )


def _write_seed(tmp_path, regions: list[dict[str, object]]) -> Path:
    path = tmp_path / "range_seed.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "range-seed-v1",
                "accepted_taxon_key": "gbif:1938069",
                "scientific_name": "Papilio demoleus",
                "regions": regions,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_range_countries_output_schema_includes_required_columns(tmp_path) -> None:
    seed = _write_seed(
        tmp_path,
        [
            {
                "region": "South Asia",
                "countries": [{"code": "IN", "name": "India"}],
                "range_status": "native_or_long_established",
            }
        ],
    )
    frame = discover_range_countries(
        accepted_taxon_key="gbif:1938069",
        scientific_name="Papilio demoleus",
        client=FakeOccurrenceClient((OccurrenceCountryFacet(country_code="IN", occurrence_count=25),)),
        seed_json=seed,
        retrieved_at="2026-07-07T00:00:00+00:00",
    )

    assert frame.schema == {
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


def test_write_range_countries_persists_parquet_output(tmp_path) -> None:
    seed = _write_seed(
        tmp_path,
        [
            {
                "region": "South Asia",
                "countries": [{"code": "IN", "name": "India"}],
                "range_status": "native_or_long_established",
            }
        ],
    )
    frame = discover_range_countries(
        accepted_taxon_key="gbif:1938069",
        scientific_name="Papilio demoleus",
        client=FakeOccurrenceClient((OccurrenceCountryFacet(country_code="IN", occurrence_count=25),)),
        seed_json=seed,
        retrieved_at="2026-07-07T00:00:00+00:00",
    )

    path = write_range_countries(frame, tmp_path / "registry")

    assert path == tmp_path / "registry" / "range_countries.parquet"
    assert pl.read_parquet(path).select("country_code").to_series().to_list() == ["IN"]
