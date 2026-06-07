from __future__ import annotations

from pathlib import Path


def test_required_config_assets_exist() -> None:
    for path in [
        ".env.example",
        "config/species_seed.csv",
        "config/regions.csv",
        "config/dwc_schema.yml",
    ]:
        assert Path(path).exists()


def test_env_example_contains_only_variable_names() -> None:
    text = Path(".env.example").read_text(encoding="utf-8")

    assert "FLICKR_API_KEY=" in text
    assert "FLICKR_API_KEY=your_flickr_api_key_here" in text


def test_dwc_schema_lists_required_occurrence_fields() -> None:
    text = Path("config/dwc_schema.yml").read_text(encoding="utf-8")

    for field in ["occurrenceID", "basisOfRecord", "eventDate", "scientificName", "dynamicProperties"]:
        assert f"- {field}" in text


def test_regions_include_broad_australia_bbox() -> None:
    text = Path("config/regions.csv").read_text(encoding="utf-8")

    assert "AU_ALL,Australia" in text


def test_regions_include_all_australian_states_and_territories() -> None:
    text = Path("config/regions.csv").read_text(encoding="utf-8")

    for region_id in ["AU_ACT", "AU_NSW", "AU_NT", "AU_QLD", "AU_SA", "AU_TAS", "AU_VIC", "AU_WA"]:
        assert f"{region_id}," in text


def test_local_papilio_demoleus_config_files_are_ignored() -> None:
    text = Path(".gitignore").read_text(encoding="utf-8")

    assert "config/papilio_demoleus_flickr_estimator.sh2" in text
    assert "config/papilio_demoleus_multilingual_keywords.json" in text
