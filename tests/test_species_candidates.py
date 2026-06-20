from __future__ import annotations

import polars as pl
import pytest

from biominer.bioclip.species_candidates import load_species_candidates, species_labels, species_prompt_variants


def test_load_species_candidates_limits_registry_species_rows_without_pinned_injection(tmp_path) -> None:
    path = tmp_path / "global_checklist.parquet"
    pl.DataFrame(
        {
            "scientificName": ["Danaus plexippus", "Papilio demoleus", "Vanessa cardui"],
            "rank": ["species", "subspecies", "species"],
            "family": ["Nymphalidae", "Papilionidae", "Nymphalidae"],
            "genus": ["Danaus", "Papilio", "Vanessa"],
            "source": ["registry", "registry", "registry"],
            "taxon_id": ["1", "2", "3"],
        }
    ).write_parquet(path)

    candidates = load_species_candidates(path, limit=2)

    assert [candidate.scientific_name for candidate in candidates] == ["Danaus plexippus", "Vanessa cardui"]
    assert all(candidate.is_target_species is False for candidate in candidates)
    assert species_labels(candidates)[0] == "a photo of Danaus plexippus"


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
    path = tmp_path / "candidates.parquet"
    pl.DataFrame(
        {
            "scientific_name": ["Papilio demoleus"],
            "rank": ["species"],
            "family": ["Papilionidae"],
            "genus": ["Papilio"],
            "common_names": ["lime butterfly|chequered swallowtail"],
        }
    ).write_parquet(path)

    candidates = load_species_candidates(path)
    assert candidates[0].common_names == ("lime butterfly", "chequered swallowtail")

    variants = species_prompt_variants(candidates)
    labels = [variant.label for variant in variants]
    assert "a photo of Papilio demoleus" in labels
    assert "a photo of lime butterfly" in labels


def test_species_candidates_reject_csv_inputs(tmp_path) -> None:
    path = tmp_path / "candidates.csv"
    path.write_text("scientific_name,rank\nPapilio demoleus,species\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Parquet"):
        load_species_candidates(path)
