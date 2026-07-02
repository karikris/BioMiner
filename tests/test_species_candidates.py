from __future__ import annotations

import polars as pl
import pytest

from biominer.bioclip.species_candidates import load_species_candidates, species_labels, species_prompt_variants
from biominer.species.context import SpeciesContext


def test_load_species_candidates_orders_explicit_target_and_limits_species_rows(tmp_path) -> None:
    path = tmp_path / "global_checklist.csv"
    path.write_text(
        "scientificName,rank,family,genus,source,taxon_id\n"
        "Danaus plexippus,species,Nymphalidae,Danaus,global,1\n"
        "Papilio demoleus,species,Papilionidae,Papilio,global,2\n"
        "Vanessa cardui,species,Nymphalidae,Vanessa,global,3\n",
        encoding="utf-8",
    )

    candidates = load_species_candidates(path, limit=2, target_species="Papilio demoleus")

    assert [candidate.scientific_name for candidate in candidates] == ["Papilio demoleus", "Danaus plexippus"]
    assert candidates[0].is_target_species is True
    assert species_labels(candidates)[0] == "a photo of Papilio demoleus"


def test_load_species_candidates_rejects_missing_target_without_explicit_context(tmp_path) -> None:
    path = tmp_path / "global_checklist.csv"
    path.write_text(
        "scientificName,rank,family,genus,source,taxon_id\n"
        "Danaus plexippus,species,Nymphalidae,Danaus,global,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target species is absent"):
        load_species_candidates(path, target_species="Papilio demoleus")


def test_load_species_candidates_allows_unregistered_target_from_context(tmp_path) -> None:
    path = tmp_path / "global_checklist.csv"
    path.write_text(
        "scientificName,rank,family,genus,source,taxon_id\n"
        "Danaus plexippus,species,Nymphalidae,Danaus,global,1\n",
        encoding="utf-8",
    )
    context = SpeciesContext(
        scientific_name="Papilio demoleus",
        accepted_taxon_key="gbif:100",
        canonical_name="Papilio demoleus",
        family="Papilionidae",
        genus="Papilio",
        family_key="gbif:10",
        genus_key="gbif:90",
        species_key="gbif:100",
        registry_version="fixture",
    )

    candidates = load_species_candidates(
        path,
        target_species="Papilio demoleus",
        allow_unregistered_target=True,
        target_context=context,
    )

    assert candidates[0].scientific_name == "Papilio demoleus"
    assert candidates[0].source == "species_context"


def test_load_species_candidates_reads_parquet_and_dedupes_names(tmp_path) -> None:
    path = tmp_path / "global_checklist.parquet"
    pl.DataFrame(
        {
            "scientific_name": ["Papilio demoleus", "Papilio demoleus", "Papilio machaon"],
            "rank": ["species", "species", "species"],
            "family": ["Papilionidae", "Papilionidae", "Papilionidae"],
            "genus": ["Papilio", "Papilio", "Papilio"],
        }
    ).write_parquet(path)

    candidates = load_species_candidates(path, limit=10)

    assert [candidate.scientific_name for candidate in candidates] == ["Papilio demoleus", "Papilio machaon"]


def test_species_candidates_read_common_names_and_build_prompt_variants(tmp_path) -> None:
    path = tmp_path / "candidates.csv"
    path.write_text(
        "scientific_name,rank,family,genus,common_names\n"
        "Papilio demoleus,species,Papilionidae,Papilio,lime butterfly|chequered swallowtail\n",
        encoding="utf-8",
    )

    candidates = load_species_candidates(path)
    assert candidates[0].common_names == ("lime butterfly", "chequered swallowtail")

    variants = species_prompt_variants(candidates)
    labels = [variant.label for variant in variants]
    assert "a photo of Papilio demoleus" in labels
    assert "a photo of lime butterfly" in labels
