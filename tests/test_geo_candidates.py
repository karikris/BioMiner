from __future__ import annotations

import polars as pl

from biominer.geo.builder import build_geo_candidate_tables
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
