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


def test_anti_keyword_config_path_is_not_documented_or_checked_in() -> None:
    assert not Path("config/anti_keywords.json").exists()

    tracked_text = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "docs/cloud_storage.md",
            "docs/cloud_provider_config.md",
            "src/biominer/filter/metadata_flags.py",
            "src/biominer/filter/__init__.py",
        )
    )
    assert "config/anti_keywords.json" not in tracked_text
    assert "load_metadata_keyword_groups" not in tracked_text
    assert "flag_metadata_parquet" not in tracked_text


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


def test_production_workflow_docs_exist_and_match_current_surface() -> None:
    required_docs = (
        Path("docs/production_workflow.md"),
        Path("docs/registry_trust_tiers.md"),
        Path("docs/vision_workflow.md"),
        Path("docs/storage_postgres_s3.md"),
    )
    for path in required_docs:
        assert path.exists(), path

    production = Path("docs/production_workflow.md").read_text(encoding="utf-8")
    for expected in (
        "biominer run",
        "auto",
        "family",
        "genus",
        "species",
        "S3",
        "Postgres",
        "metadata flags",
    ):
        assert expected in production

    registry = Path("docs/registry_trust_tiers.md").read_text(encoding="utf-8")
    for expected in ("T1", "T5", "disabled", "query_definition_id"):
        assert expected in registry

    vision = Path("docs/vision_workflow.md").read_text(encoding="utf-8")
    for expected in (
        "biominer vision detect",
        "biominer dev vision yoloe26-runtime-check",
        "BioCLIP",
        "object finder",
        "detector_crop_segmentation",
    ):
        assert expected in vision
    assert "YOLOE/YOLO26 output must not be interpreted as species classification" in vision
    assert "does not store reviewed boxes" in vision

    storage = Path("docs/storage_postgres_s3.md").read_text(encoding="utf-8")
    for expected in (
        "BIOMINER_S3_ENDPOINT_URL",
        "BIOMINER_WORKSTORE_DSN",
        "--storage-backend local",
        "--workstore-backend sqlite",
    ):
        assert expected in storage


def test_production_examples_cover_family_genus_species_runs() -> None:
    path = Path("examples/production_workflow.md")
    assert path.exists()

    text = path.read_text(encoding="utf-8")
    for expected in (
        "BIOMINER_S3_ENDPOINT_URL",
        "BIOMINER_WORKSTORE_DSN",
        "FLICKR_API_KEY",
        "--taxon \"Papilio demoleus\"",
        "--rank species",
        "--taxon \"Papilio\"",
        "--rank genus",
        "--taxon \"Papilionidae\"",
        "--rank family",
        "--storage-backend s3",
        "--workstore-backend postgres",
        "--vision-backend yoloe26",
        "imageomics/bioclip-2.5-vith14",
    ):
        assert expected in text
    assert "broad seed" not in text.lower()
    assert "YOLO species" not in text
