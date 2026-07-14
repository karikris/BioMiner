from __future__ import annotations

from pathlib import Path

from biominer.run import ProductionRunRequest, RunArtifactUris, RunPaths, TaxonScope
from biominer.run.orchestrator import build_run_plan
from biominer.run.paths import (
    REFERENCE_FIRST_ARTIFACT_KEYS,
    RUN_ARTIFACT_DIRECTORY_KEYS,
    RUN_ARTIFACT_LAYOUT_VERSION,
    RUN_ARTIFACT_RELATIVE_PATHS,
)
from biominer.species.context import SpeciesContext


def test_reference_first_artifact_inventory_covers_every_durable_category() -> None:
    assert REFERENCE_FIRST_ARTIFACT_KEYS <= set(RUN_ARTIFACT_RELATIVE_PATHS)
    assert REFERENCE_FIRST_ARTIFACT_KEYS == {
        "geographic_registry",
        "taxon_geographic_spread",
        "geographic_occurrence_evidence",
        "taxon_geographic_summary",
        "geographic_qa_findings",
        "flickr_geography",
        "flickr_geo_clusters",
        "flickr_geo_assignments",
        "regional_candidates",
        "reference_observations",
        "reference_media",
        "reference_media_candidates",
        "reference_media_objects",
        "reference_review",
        "reference_review_queue",
        "reference_review_decisions",
        "reference_readiness",
        "reference_embeddings",
        "reference_prototypes",
        "feature_matrix",
        "classifiers",
        "calibrators",
        "flickr_embeddings",
        "target_aware_object_scores",
        "target_aware_candidate_scores",
        "reports",
    }
    assert {
        "geographic_registry",
        "reference_media",
        "reference_review",
        "reference_readiness",
        "classifiers",
        "calibrators",
        "reports",
    } <= RUN_ARTIFACT_DIRECTORY_KEYS


def test_local_and_cloud_artifact_layouts_have_identical_relative_paths(
    tmp_path: Path,
) -> None:
    run_id = "Papilio: demoleus"
    paths = RunPaths.from_root(tmp_path, run_id=run_id)
    uris = RunArtifactUris.from_prefix("s3://biominer/runs", run_id=run_id)

    local = paths.to_dict()
    cloud = uris.to_dict()
    assert local.keys() == cloud.keys()
    for key, relative in RUN_ARTIFACT_RELATIVE_PATHS.items():
        assert Path(local[key]).relative_to(paths.run_root).as_posix() == relative
        assert cloud[key] == f"{uris.run_root_uri}/{relative}"

    assert paths.reference_embeddings_path.name == "reference_embeddings.parquet"
    assert uris.reference_embeddings_uri.endswith(
        "/references/embeddings/reference_embeddings.parquet"
    )
    assert paths.target_aware_candidate_scores_path.name == (
        "target_aware_candidate_scores.parquet"
    )
    assert uris.flickr_geo_assignments_uri.endswith(
        "/flickr/geography/flickr_geo_assignments.parquet"
    )


def test_scoped_classifier_and_calibrator_paths_are_safe_and_immutable(
    tmp_path: Path,
) -> None:
    paths = RunPaths.from_root(tmp_path, run_id="pilot")
    uris = RunArtifactUris.from_prefix("s3://biominer/runs", run_id="pilot")

    classifier = paths.classifier_artifact_dir(
        target_task="Target vs Other",
        route="Adult/Field",
        artifact_fingerprint="sha256:abcdef",
    )
    calibrator = uris.calibrator_artifact_uri(
        target_task="Target vs Other",
        route="Adult/Field",
        artifact_fingerprint="sha256:123456",
    )

    assert classifier.relative_to(paths.classifiers_dir).as_posix() == (
        "task=target_vs_other/route=adult_field/artifact=sha256_abcdef"
    )
    assert calibrator.endswith(
        "/ml/calibrators/task=target_vs_other/route=adult_field/"
        "artifact=sha256_123456"
    )


def test_ensure_directories_materializes_every_local_artifact_parent(
    tmp_path: Path,
) -> None:
    paths = RunPaths.from_root(tmp_path, run_id="pilot")

    paths.ensure_directories()

    for key, relative in RUN_ARTIFACT_RELATIVE_PATHS.items():
        artifact = paths.run_root / relative
        expected_directory = artifact if key in RUN_ARTIFACT_DIRECTORY_KEYS else artifact.parent
        assert expected_directory.is_dir(), key


def test_run_plan_manifests_complete_artifact_inventory_and_audit_metrics(
    tmp_path: Path,
) -> None:
    request = ProductionRunRequest(
        taxon="Papilio demoleus",
        rank="species",
        output_root=tmp_path,
        run_id="artifact-layout",
        dry_run=True,
    )
    plan = build_run_plan(request, taxon_scope=_taxon_scope())

    expected_outputs = plan.artifact_uris.to_dict()
    expected_metrics = plan.artifact_uris.audit_metrics()
    assert plan.manifest.outputs == expected_outputs
    assert plan.to_dict()["paths"] == plan.paths.to_dict()
    assert plan.manifest.metrics["artifact_layout_version"] == (
        RUN_ARTIFACT_LAYOUT_VERSION
    )
    assert plan.manifest.metrics["artifact_location_count"] == len(expected_outputs)
    assert plan.manifest.metrics["artifact_directory_count"] == len(
        RUN_ARTIFACT_DIRECTORY_KEYS
    )
    assert expected_metrics.items() <= plan.manifest.metrics.items()
    assert plan.manifest.model_configs["artifact_layout_version"] == (
        RUN_ARTIFACT_LAYOUT_VERSION
    )


def _taxon_scope() -> TaxonScope:
    context = SpeciesContext(
        scientific_name="Papilio demoleus",
        accepted_taxon_key="gbif:1941315",
        canonical_name="Papilio demoleus",
        family="Papilionidae",
        genus="Papilio",
        family_key="gbif:9417",
        genus_key="gbif:1920490",
        species_key="gbif:1941315",
        registry_version="registry-v1",
    )
    return TaxonScope.from_species_context(context)
