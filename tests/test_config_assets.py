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


def test_metadata_keyword_path_helpers_are_not_documented() -> None:
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
    assert '--species-context runs/local_debug/papilio_demoleus/species_context.json' in text
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
    for expected in (
        "T1",
        "T5",
        "enabled",
        "query_definition_id",
        "enabled_t5_name_rows",
        "t5_query_definition_rows",
    ):
        assert expected in registry
    assert "T3  Wikidata labels and aliases with confident external taxon links" in registry
    assert "external taxon linkage is confident" in registry
    assert "weaker interpretation" not in registry

    vision = Path("docs/vision_workflow.md").read_text(encoding="utf-8")
    for expected in (
        "biominer vision detect",
        "biominer dev vision yoloe26-runtime-check",
        "BioCLIP",
        "object finder",
        "user-provided coarse-object checkpoint",
        "detector_crop_segmentation",
        "--species-candidates data/registry/current/species_candidates.parquet",
        "runs/local_debug/papilio_demoleus",
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


def test_final_command_surface_audit_includes_migration_notes() -> None:
    text = Path("reports/refactor_final_command_surface.md").read_text(encoding="utf-8")

    for expected in (
        "## Migration Notes",
        "biominer run --taxon <name> --rank auto|family|genus|species",
        "flickr_query_definitions.parquet",
        "--storage-backend local --workstore-backend sqlite --dry-run",
    ):
        assert expected in text


def test_completion_audit_records_current_refactor_invariants() -> None:
    text = Path("reports/refactor_completion_audit.md").read_text(encoding="utf-8")

    for expected in (
        "Public production workflow is rank-aware",
        "Production storage/workstore defaults",
        "T5 generated translations are retained as name evidence and query-gated before Flickr retrieval",
        "Metadata keyword logic is flags, not hard pre-visual drop",
        "YOLOE/YOLO26 are object proposal backends only",
        "BioCLIP remains the species scorer",
        "Generated artifacts are not tracked",
    ):
        assert expected in text


def test_refactor_phase_commit_map_records_pushed_main_phases() -> None:
    text = Path("reports/refactor_phase_commit_map.md").read_text(encoding="utf-8")

    for expected in (
        "Current branch: `main`",
        "All commits listed below are present on",
        "Phase 0 - Repository and environment audit",
        "Phase 10 - Final audits and PR readiness evidence",
        "`f28aa84` chore: audit refactor environment and workflow surface",
        "`e6a498b` docs: clarify verified refactor audit base",
        "T5 translations",
        "requiring query eligibility",
    ):
        assert expected in text


def test_mcp_environment_report_uses_generic_runtime_layout() -> None:
    text = Path("reports/refactor_mcp_environment.md").read_text(encoding="utf-8")

    assert "/Users/merm0001/Repos" not in text
    assert "BIOMINER_BASE_PATH" in text
    assert "/Applications/secrets/secrets.env" in text
    assert "/mnt/c/Applications/secrets/secrets.env" in text


def test_yoloe26_docs_do_not_recommend_training_dataset_storage() -> None:
    text = Path("docs/yoloe26_prototype.md").read_text(encoding="utf-8")

    for forbidden in (
        "reviewed box dataset",
        "supervised YOLO",
        "fine-tuning",
        "training dataset",
    ):
        assert forbidden not in text
    assert "rather than detector training artifacts" in text


def test_readme_keeps_broad_probe_recipe_out_of_production_workflow() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    for removed_phrase in (
        "Explicit broad-probe coverage",
        "reviewed anchored broad terms",
        "Broad probes are not implicit production seeds",
        "Broad searches do not recursively count-probe",
        "A broad term such as `butterfly`",
    ):
        assert removed_phrase not in text
    assert "work items must be created from registry-derived query definitions" in text
    assert "reviewed/corroborated translation names retained with low-trust provenance" in text
    assert "query-eligible species-level registry name" in text
    assert "register-based processing" not in text
    assert "register-based classification helpers" not in text
    assert "object-first scoring" in text
