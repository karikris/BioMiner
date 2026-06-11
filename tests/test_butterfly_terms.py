from __future__ import annotations

import polars as pl
from pathlib import Path

from flickr_bio_occurrence.flickr.butterfly_terms import (
    estimate_minimum_fetch_hours,
    load_butterfly_dashboard_terms,
    safe_query_variant,
)


def test_load_butterfly_dashboard_terms_includes_species_common_names_and_common_words(tmp_path) -> None:
    data_dir = tmp_path / "dashboard" / "data"
    reference_dir = data_dir / "reference"
    reference_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "scientificName": ["Papilio demoleus", "Papilio demoleus sthenelus"],
            "species": ["Papilio demoleus", "Papilio demoleus"],
            "genus": ["Papilio", "Papilio"],
            "family": ["Papilionidae", "Papilionidae"],
        }
    ).write_parquet(data_dir / "butterfly_sa2_bins.parquet")
    (reference_dir / "butterfly_conservation_status.csv").write_text(
        "accepted_taxon,match_names,rank,common_name\n"
        "Papilio demoleus,Papilio demoleus,species,Lime Butterfly / Citrus Swallowtail\n",
        encoding="utf-8",
    )

    terms = load_butterfly_dashboard_terms(data_dir, common_terms=("butterfly",))
    values = {term.term: term.source for term in terms}

    assert values["Papilio demoleus"] == "scientificName"
    assert values["Papilio demoleus sthenelus"] == "scientificName"
    assert values["Papilio"] == "genus"
    assert values["Papilionidae"] == "family"
    assert values["Lime Butterfly"] == "common_name"
    assert values["Citrus Swallowtail"] == "common_name"
    assert values["butterfly"] == "common_word"


def test_safe_query_variant_is_stable_slug() -> None:
    assert safe_query_variant("Papilio (Eleppone) anactus") == "papilio_eleppone_anactus"


def test_estimate_minimum_fetch_hours_uses_hourly_api_cap() -> None:
    assert estimate_minimum_fetch_hours(planned_api_calls=7200, api_calls_per_hour=3600) == 2.0
