from __future__ import annotations

import sqlite3

import httpx
import polars as pl
import pytest

from biominer.geo.builder import build_geo_candidate_tables
from biominer.geo.gbif_candidates import build_gbif_geo_candidates
from biominer.geo.grid import candidate_set_for_point, geocell_id, neighbour_geocell_ids


def test_geo_candidate_fallback_by_grid_level() -> None:
    latitude = -27.0
    longitude = 153.0
    local_cell = geocell_id("G4_5deg", latitude, longitude)
    world_cell = geocell_id("G0_world", latitude, longitude)
    index = pl.DataFrame(
        [
            {
                "grid_level": "G4_5deg",
                "geocell_id": local_cell,
                "species_key": "1",
                "scientific_name": "Danaus plexippus",
                "candidate_rank_prior": 1.0,
            },
            {
                "grid_level": "G0_world",
                "geocell_id": world_cell,
                "species_key": "1",
                "scientific_name": "Danaus plexippus",
                "candidate_rank_prior": 0.5,
            },
            {
                "grid_level": "G0_world",
                "geocell_id": world_cell,
                "species_key": "2",
                "scientific_name": "Vanessa cardui",
                "candidate_rank_prior": 0.5,
            },
        ]
    )

    result = candidate_set_for_point(
        index,
        latitude=latitude,
        longitude=longitude,
        preferred_grid_level="G4_5deg",
        min_species_per_cell=2,
    )

    assert result.selected_grid_level == "G0_world"
    assert result.fallback_reason == "local_cell_below_min_species"
    assert result.candidates.height == 2


def test_neighbour_cells_are_deterministic() -> None:
    cells = neighbour_geocell_ids("G4_5deg", -27.0, 153.0)

    assert cells == tuple(sorted(cells))
    assert geocell_id("G4_5deg", -27.0, 153.0) in cells


def test_geo_candidate_builder_writes_required_schema(tmp_path) -> None:
    occurrences = pl.DataFrame(
        [
            {
                "speciesKey": "1",
                "scientificName": "Danaus plexippus",
                "family": "Nymphalidae",
                "genus": "Danaus",
                "decimalLatitude": -27.0,
                "decimalLongitude": 153.0,
                "coordinateUncertaintyInMeters": 10000,
                "basisOfRecord": "HUMAN_OBSERVATION",
                "datasetKey": "dataset-a",
                "year": 2024,
            }
        ]
    )

    outputs = build_geo_candidate_tables(occurrences, output_dir=tmp_path, geo_version="test-v1", grid_levels=("G4_5deg",))

    species_index = pl.read_parquet(outputs["geo_species_index"])
    row = species_index.filter(pl.col("grid_level") == "G4_5deg").to_dicts()[0]
    assert row["geo_version"] == "test-v1"
    assert row["occurrence_count"] == 1
    assert row["source_dataset_count"] == 1
    assert row["basis_of_record_counts"] == '{"HUMAN_OBSERVATION": 1}'


def test_gbif_geo_ingestion_paginates_retries_and_consolidates(tmp_path) -> None:
    taxa = tmp_path / "taxa.parquet"
    _taxa_frame().write_parquet(taxa)
    clients: list[FakeOccurrenceClient] = []

    def factory() -> FakeOccurrenceClient:
        client = FakeOccurrenceClient(
            [
                _http_error(429),
                _occurrence_payload(
                    count=2,
                    end=False,
                    key="occ-1",
                    latitude=-27.0,
                    longitude=153.0,
                    year=2024,
                ),
                _occurrence_payload(
                    count=2,
                    end=True,
                    key="occ-2",
                    latitude=-28.0,
                    longitude=152.0,
                    year=2023,
                ),
            ]
        )
        clients.append(client)
        return client

    result = build_gbif_geo_candidates(
        taxa_path=taxa,
        output_dir=tmp_path / "geo",
        geo_version="geo-v1",
        state_db=tmp_path / "state.sqlite",
        workers=1,
        page_size=1,
        max_retries=3,
        client_factory=factory,
    )

    reference = pl.read_parquet(result["outputs"]["gbif_occurrence_reference"])
    species_index = pl.read_parquet(result["outputs"]["geo_species_index"])
    state = sqlite3.connect(tmp_path / "state.sqlite").execute(
        "SELECT offset, status, attempts FROM gbif_geo_occurrence_progress WHERE species_key = '100'"
    ).fetchone()
    assert result["species_completed"] == 1
    assert result["occurrence_rows"] == 2
    assert state == (2, "completed", 1)
    assert [call["offset"] for call in clients[0].calls] == [0, 0, 1]
    assert reference.select("year").to_series().to_list() == [2024, 2023]
    assert species_index.filter(pl.col("grid_level") == "G0_world").select("occurrence_count").to_series().to_list() == [2]


def test_gbif_geo_ingestion_resumes_from_sqlite_offset(tmp_path) -> None:
    taxa = tmp_path / "taxa.parquet"
    state_db = tmp_path / "state.sqlite"
    output_dir = tmp_path / "geo"
    _taxa_frame().write_parquet(taxa)

    build_gbif_geo_candidates(
        taxa_path=taxa,
        output_dir=output_dir,
        geo_version="geo-v1",
        state_db=state_db,
        workers=1,
        page_size=1,
        max_retries=1,
        client_factory=lambda: FakeOccurrenceClient(
            [
                _occurrence_payload(count=2, end=False, key="occ-1", latitude=-27.0, longitude=153.0, year=2024),
                _http_error(500),
            ]
        ),
    )

    result = build_gbif_geo_candidates(
        taxa_path=taxa,
        output_dir=output_dir,
        geo_version="geo-v1",
        state_db=state_db,
        workers=1,
        page_size=1,
        max_retries=2,
        client_factory=lambda: FakeOccurrenceClient(
            [
                _occurrence_payload(count=2, end=True, key="occ-2", latitude=-28.0, longitude=152.0, year=2023),
            ]
        ),
    )

    reference = pl.read_parquet(result["outputs"]["gbif_occurrence_reference"])
    state = sqlite3.connect(state_db).execute(
        "SELECT offset, status FROM gbif_geo_occurrence_progress WHERE species_key = '100'"
    ).fetchone()
    assert state == (2, "completed")
    assert reference.select("year").to_series().to_list() == [2024, 2023]


def test_gbif_geo_ingestion_refuses_fingerprint_mismatch(tmp_path) -> None:
    taxa = tmp_path / "taxa.parquet"
    state_db = tmp_path / "state.sqlite"
    output_dir = tmp_path / "geo"
    _taxa_frame().write_parquet(taxa)

    build_gbif_geo_candidates(
        taxa_path=taxa,
        output_dir=output_dir,
        geo_version="geo-v1",
        state_db=state_db,
        workers=1,
        page_size=1,
        client_factory=lambda: FakeOccurrenceClient(
            [_occurrence_payload(count=1, end=True, key="occ-1", latitude=-27.0, longitude=153.0, year=2024)]
        ),
    )

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        build_gbif_geo_candidates(
            taxa_path=taxa,
            output_dir=output_dir,
            geo_version="geo-v1",
            state_db=state_db,
            workers=1,
            page_size=2,
            client_factory=lambda: FakeOccurrenceClient([]),
        )


def _taxa_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:100",
                "species_key": "gbif:100",
                "scientific_name": "Papilio demoleus",
                "rank": "SPECIES",
                "family": "Papilionidae",
                "genus": "Papilio",
            }
        ]
    )


def _occurrence_payload(
    *,
    count: int,
    end: bool,
    key: str,
    latitude: float,
    longitude: float,
    year: int,
) -> dict[str, object]:
    return {
        "count": count,
        "endOfRecords": end,
        "results": [
            {
                "key": key,
                "datasetKey": "dataset-a",
                "basisOfRecord": "HUMAN_OBSERVATION",
                "occurrenceID": f"urn:test:{key}",
                "taxonKey": 100,
                "speciesKey": 100,
                "scientificName": "Papilio demoleus",
                "family": "Papilionidae",
                "genus": "Papilio",
                "decimalLatitude": latitude,
                "decimalLongitude": longitude,
                "coordinateUncertaintyInMeters": 10000,
                "countryCode": "AU",
                "year": year,
                "eventDate": f"{year}-01-01",
                "issues": ["COORDINATE_ROUNDED"],
            }
        ],
    }


def _http_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://api.gbif.org/v1/occurrence/search")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(f"HTTP {status_code}", request=request, response=response)


class FakeOccurrenceClient:
    def __init__(self, outcomes: list[dict[str, object] | BaseException]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def occurrence_search(self, params: dict[str, object]) -> dict[str, object]:
        self.calls.append(dict(params))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self) -> None:
        self.closed = True
