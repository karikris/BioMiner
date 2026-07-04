from __future__ import annotations

from pathlib import Path


def test_env_example_exists() -> None:
    assert Path(".env.example").exists()


def test_env_example_contains_only_variable_names() -> None:
    text = Path(".env.example").read_text(encoding="utf-8")

    assert "FLICKR_API_KEY=" in text
    assert "FLICKR_API_KEY=your_flickr_api_key_here" in text


def test_removed_papilio_demoleus_seed_config_paths_are_not_documented() -> None:
    text = Path(".gitignore").read_text(encoding="utf-8")

    assert "config/papilio_demoleus_flickr_estimator.sh2" not in text
    assert "config/papilio_demoleus_multilingual_keywords.json" not in text


def test_papilio_species_example_documents_generic_object_pipeline() -> None:
    path = Path("examples/species/papilio_demoleus/object_pipeline.md")
    assert path.exists()

    text = path.read_text(encoding="utf-8")
    for command in (
        "biominer vision detect",
        "biominer vision score",
        "biominer vision ablate",
        "biominer evidence join",
    ):
        assert command in text
    assert '--species-context staging/species_runs/papilio_demoleus/species_context.json' in text
    assert "--image-max-side-px" in text
    assert "fetch_papilio_demoleus_multilingual_metadata.py" not in text
    assert "classify_papilio_demoleus" not in text
