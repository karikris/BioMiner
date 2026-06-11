from __future__ import annotations

import polars as pl

from biominer.bioclip.species_candidates import load_species_candidates, species_labels


def test_load_species_candidates_pins_target_and_limits_species_rows(tmp_path) -> None:
    path = tmp_path / "global_checklist.csv"
    path.write_text(
        "scientificName,rank,family,genus,source,taxon_id\n"
        "Danaus plexippus,species,Nymphalidae,Danaus,global,1\n"
        "Papilio demoleus,subspecies,Papilionidae,Papilio,global,2\n"
        "Vanessa cardui,species,Nymphalidae,Vanessa,global,3\n",
        encoding="utf-8",
    )

    candidates = load_species_candidates(path, limit=2)

    assert [candidate.scientific_name for candidate in candidates] == ["Papilio demoleus", "Danaus plexippus"]
    assert candidates[0].is_target_species is True
    assert species_labels(candidates)[0] == "a photo of Papilio demoleus"


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
