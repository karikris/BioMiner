from __future__ import annotations

import pytest
import polars as pl

from biominer.bioclip.candidate_sets import (
    CandidateMode,
    CandidateStrategy,
    build_candidate_set,
    candidate_set_signature,
    family_labels,
    genus_candidates_by_family,
    parse_candidate_mode,
    parse_candidate_strategy,
)
from biominer.bioclip.species_candidates import SpeciesCandidate
from biominer.geo.grid import geocell_id


def _candidate(name: str, *, family: str, genus: str) -> SpeciesCandidate:
    return SpeciesCandidate(
        scientific_name=name,
        canonical_name=name,
        rank="species",
        family=family,
        genus=genus,
        source="test",
        source_taxon_id=name,
        is_target_species=False,
    )


def test_candidate_mode_and_strategy_parsing() -> None:
    assert parse_candidate_mode("triage") is CandidateMode.TRIAGE
    assert parse_candidate_mode("rescue_full_species") is CandidateMode.RESCUE_FULL_SPECIES
    assert parse_candidate_strategy("hierarchical") is CandidateStrategy.HIERARCHICAL
    with pytest.raises(ValueError):
        parse_candidate_mode("invalid")


def test_family_topk_candidate_generation() -> None:
    candidate_set = build_candidate_set(
        {},
        species_candidates=[
            _candidate("Danaus plexippus", family="Nymphalidae", genus="Danaus"),
            _candidate("Papilio machaon", family="Papilionidae", genus="Papilio"),
        ],
        mode="family",
        strategy="all",
    )

    assert candidate_set.family_candidates == ("Papilionidae", "Nymphalidae")
    assert candidate_set.label_sets["family"] == family_labels(("Papilionidae", "Nymphalidae"))


def test_genus_topk_per_family() -> None:
    candidates = [
        _candidate("Danaus plexippus", family="Nymphalidae", genus="Danaus"),
        _candidate("Vanessa cardui", family="Nymphalidae", genus="Vanessa"),
        _candidate("Papilio machaon", family="Papilionidae", genus="Papilio"),
    ]

    grouped = genus_candidates_by_family(candidates, families=("Nymphalidae",), per_family_limit=8)

    assert grouped == {"Nymphalidae": ("Danaus", "Vanessa")}


def test_candidate_set_signature_is_deterministic() -> None:
    left = candidate_set_signature({"species": ("b", "a"), "triage": ("x",)})
    right = candidate_set_signature({"triage": ("x",), "species": ("b", "a")})
    changed_order = candidate_set_signature({"species": ("a", "b"), "triage": ("x",)})

    assert left == right
    assert left != changed_order


def test_geo_strategy_uses_geo_index_for_geolocated_record() -> None:
    latitude = -27.0
    longitude = 153.0
    cell_id = geocell_id("G4_5deg", latitude, longitude)
    candidates = [
        _candidate("Danaus plexippus", family="Nymphalidae", genus="Danaus"),
        _candidate("Papilio machaon", family="Papilionidae", genus="Papilio"),
    ]
    geo_index = pl.DataFrame(
        [
            {
                "grid_level": "G4_5deg",
                "geocell_id": cell_id,
                "species_key": "1",
                "scientific_name": "Danaus plexippus",
                "candidate_rank_prior": 1.0,
            }
        ]
    )

    candidate_set = build_candidate_set(
        {"latitude": latitude, "longitude": longitude},
        species_candidates=candidates,
        mode="species",
        strategy="geo",
        geo_species_index=geo_index,
        geo_min_species_per_cell=1,
    )

    assert [candidate.scientific_name for candidate in candidate_set.species_candidates] == ["Danaus plexippus"]
    assert candidate_set.geo_candidate_cell_id == cell_id
    assert candidate_set.geo_candidate_grid_level == "G4_5deg"
    assert "gbif_geo" in candidate_set.species_candidate_sources_json


def test_hierarchical_strategy_intersects_geo_gate_and_rescues_metadata_match() -> None:
    latitude = -27.0
    longitude = 153.0
    cell_id = geocell_id("G4_5deg", latitude, longitude)
    candidates = [
        _candidate("Danaus plexippus", family="Nymphalidae", genus="Danaus"),
        _candidate("Papilio machaon", family="Papilionidae", genus="Papilio"),
        _candidate("Vanessa cardui", family="Nymphalidae", genus="Vanessa"),
    ]
    geo_index = pl.DataFrame(
        [
            {
                "grid_level": "G4_5deg",
                "geocell_id": cell_id,
                "species_key": "1",
                "scientific_name": "Danaus plexippus",
                "candidate_rank_prior": 0.6,
            },
            {
                "grid_level": "G4_5deg",
                "geocell_id": cell_id,
                "species_key": "2",
                "scientific_name": "Papilio machaon",
                "candidate_rank_prior": 0.4,
            },
        ]
    )

    candidate_set = build_candidate_set(
        {
            "latitude": latitude,
            "longitude": longitude,
            "title": "Danaus plexippus",
            "family_topk_json": [{"label": "a photo of a Papilionidae butterfly", "score": 0.9}],
        },
        species_candidates=candidates,
        mode="species",
        strategy="hierarchical",
        geo_species_index=geo_index,
        geo_min_species_per_cell=1,
    )

    assert [candidate.scientific_name for candidate in candidate_set.species_candidates] == [
        "Papilio machaon",
        "Danaus plexippus",
    ]
    assert "metadata_rescue" in candidate_set.species_candidate_sources_json
