from __future__ import annotations

import pytest

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
