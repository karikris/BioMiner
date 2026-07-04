from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from biominer.bioclip.object_runner import OBJECT_VISUAL_MODES, PRIMARY_VISUAL_CLASSIFIER
from biominer.detection.detector_base import DecodedImage, DetectionCandidate, FakeObjectDetector
from biominer.evidence import evidence_count_metrics
from biominer.evidence.join import write_object_evidence_outputs
from biominer.registry.trust_policy import (
    TrustTier,
    decide_name_trust,
    disabled_reason_for_candidate,
    should_enable_name_by_default,
    source_default_trust_tier,
)
from biominer.run import (
    ProductionRunOrchestrator,
    ProductionRunRequest,
    RunArtifactUris,
    RunManifest,
    RunPaths,
    RunStage,
    StageExecutionResult,
    StageRecord,
    StageStatus,
    TaxonScope,
    resolve_taxon_scope_from_registry,
)
from biominer.species.context import CommonName, SpeciesContext
from biominer.workstore.sqlite import SQLiteWorkStore


def test_taxon_scope_construction_and_roundtrip() -> None:
    context = _species_context()
    scope = TaxonScope.from_species_context(context)

    assert scope.input_name == "Danaus plexippus"
    assert scope.input_rank == "species"
    assert scope.accepted_rank == "species"
    assert scope.species_count == 1
    assert scope.species_names == ("Danaus plexippus",)
    assert TaxonScope.from_dict(scope.to_dict()) == scope


def test_taxon_scope_validates_rank_and_species_contexts() -> None:
    with pytest.raises(ValueError, match="input_rank"):
        TaxonScope(
            input_name="Danaus",
            input_rank="subgenus",  # type: ignore[arg-type]
            accepted_taxon_key="gbif:5131",
            accepted_scientific_name="Danaus",
            accepted_rank="genus",
            registry_version="test-v1",
            species_contexts=(_species_context(),),
        )
    with pytest.raises(ValueError, match="species_contexts"):
        TaxonScope(
            input_name="Danaus",
            input_rank="genus",
            accepted_taxon_key="gbif:5131",
            accepted_scientific_name="Danaus",
            accepted_rank="genus",
            registry_version="test-v1",
            species_contexts=(),
        )


def test_resolve_taxon_scope_from_registry_expands_species_genus_and_family(tmp_path) -> None:
    registry = _write_rank_registry(tmp_path / "registry")

    species_scope = resolve_taxon_scope_from_registry(registry_dir=registry, input_name="Papilio demoleus", input_rank="species")
    genus_scope = resolve_taxon_scope_from_registry(registry_dir=registry, input_name="Papilio", input_rank="genus")
    family_scope = resolve_taxon_scope_from_registry(registry_dir=registry, input_name="Papilionidae", input_rank="family")
    auto_scope = resolve_taxon_scope_from_registry(registry_dir=registry, input_name="Papilio", input_rank="auto")

    assert species_scope.accepted_rank == "species"
    assert species_scope.species_names == ("Papilio demoleus",)
    assert species_scope.species_contexts[0].common_names[0].name == "Lime butterfly"
    assert genus_scope.accepted_rank == "genus"
    assert genus_scope.accepted_taxon_key == "gbif:90"
    assert genus_scope.species_names == ("Papilio demoleus", "Papilio machaon")
    assert family_scope.accepted_rank == "family"
    assert family_scope.species_names == ("Papilio demoleus", "Papilio machaon", "Shared name")
    assert auto_scope.accepted_rank == "genus"
    assert auto_scope.species_count == 2


def test_resolve_taxon_scope_reports_ambiguous_or_empty_registry_matches(tmp_path) -> None:
    registry = _write_rank_registry(tmp_path / "registry")

    with pytest.raises(ValueError, match="ambiguous taxon match"):
        resolve_taxon_scope_from_registry(registry_dir=registry, input_name="Shared name", input_rank="auto")

    with pytest.raises(ValueError, match="no species found under genus"):
        resolve_taxon_scope_from_registry(registry_dir=registry, input_name="Emptygenus", input_rank="genus")

    with pytest.raises(ValueError, match="species not found"):
        resolve_taxon_scope_from_registry(registry_dir=registry, input_name="Missing species", input_rank="species")


def test_run_paths_and_dry_run_manifest(tmp_path) -> None:
    scope = TaxonScope.from_species_context(_species_context())
    request = ProductionRunRequest(taxon="Danaus plexippus", rank="species", output_root=tmp_path, dry_run=True)
    orchestrator = ProductionRunOrchestrator(request, taxon_scope=scope)
    plan = orchestrator.plan()

    manifest_path = orchestrator.write_dry_run_manifest()
    manifest = RunManifest.read_json(manifest_path)

    assert manifest_path == tmp_path / "run_id=species_danaus_plexippus" / "run_manifest.json"
    assert plan.artifact_uris.query_definitions_uri.endswith("/registry/flickr_query_definitions.parquet")
    assert manifest.storage_backend == "s3"
    assert manifest.workstore_backend == "postgres"
    assert manifest.taxon_scope == scope
    assert manifest.model_configs["primary_visual_classifier"] == PRIMARY_VISUAL_CLASSIFIER
    assert manifest.model_configs["visual_modes"] == list(OBJECT_VISUAL_MODES)
    assert manifest.query_counts == {"compiled_definitions": 0, "enqueued_work_items": 0}
    assert manifest.detection_counts == {"images_seen": 0, "detections": 0, "crops_created": 0}
    assert manifest.bioclip_counts == {"objects_scored": 0, "whole_images_scored": 0, "segmentation_crops_scored": 0}
    assert manifest.evidence_counts == {"object_evidence_rows": 0, "photo_summary_rows": 0}
    assert manifest.outputs["manifest"].endswith("/run_manifest.json")
    assert [stage.stage for stage in manifest.stages][:3] == [
        RunStage.RESOLVE_TAXON_SCOPE,
        RunStage.BUILD_REGISTRY,
        RunStage.COMPILE_QUERIES,
    ]
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["species_count"] == 1


def test_run_artifact_uris_are_s3_safe_and_species_scoped() -> None:
    uris = RunArtifactUris.from_prefix("s3://biominer/runs", run_id="Family: Papilionidae")

    assert uris.run_root_uri == "s3://biominer/runs/run_id=family_papilionidae"
    assert uris.manifest_uri == "s3://biominer/runs/run_id=family_papilionidae/run_manifest.json"
    assert uris.query_definitions_uri == "s3://biominer/runs/run_id=family_papilionidae/registry/flickr_query_definitions.parquet"
    assert uris.object_detections_uri == "s3://biominer/runs/run_id=family_papilionidae/staging/object_detections.parquet"
    assert uris.species_uri("Papilio demoleus") == "s3://biominer/runs/run_id=family_papilionidae/species/papilio_demoleus"
    assert uris.species_context_uri("Papilio demoleus").endswith("/species/papilio_demoleus/species_context.json")
    assert uris.to_dict()["photo_summary"].endswith("/staging/photo_evidence_summary.parquet")


def test_run_manifest_stage_status_and_count_roundtrip() -> None:
    scope = TaxonScope.from_species_context(_species_context())
    manifest = RunManifest(
        run_id="run-1",
        taxon_scope=scope,
        stages=(StageRecord(stage=RunStage.COMPILE_QUERIES),),
        query_counts={"compiled_definitions": 0},
    )

    manifest = manifest.with_stage_status(
        RunStage.COMPILE_QUERIES,
        StageStatus.COMPLETE,
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:00:01Z",
        metrics={"compiled_definitions": 12},
        outputs={"query_definitions": "s3://biominer/runs/run_id=run-1/registry/flickr_query_definitions.parquet"},
    )
    payload = manifest.to_dict()
    roundtrip = RunManifest.from_dict(payload)

    assert roundtrip.query_counts == {"compiled_definitions": 0}
    assert roundtrip.stages[0].status is StageStatus.COMPLETE
    assert roundtrip.stages[0].metrics == {"compiled_definitions": 12}
    assert roundtrip.stages[0].outputs["query_definitions"].startswith("s3://")


def test_orchestrator_resolves_scope_and_runs_stage_subset_with_fake_handlers(tmp_path) -> None:
    registry = _write_rank_registry(tmp_path / "registry")
    calls: list[int] = []

    def fake_compile(plan):  # noqa: ANN001 - test double mirrors the stage handler protocol.
        calls.append(plan.manifest.taxon_scope.species_count)
        return StageExecutionResult(
            metrics={"compiled_definitions": 4},
            outputs={"query_definitions": plan.artifact_uris.query_definitions_uri},
        )

    request = ProductionRunRequest(
        taxon="Papilio",
        rank="genus",
        registry_dir=str(registry),
        output_root=tmp_path / "runs",
        stages=(RunStage.RESOLVE_TAXON_SCOPE, RunStage.COMPILE_QUERIES, RunStage.BUILD_REGISTRY),
    )
    plan = ProductionRunOrchestrator(
        request,
        stage_handlers={RunStage.COMPILE_QUERIES: fake_compile},
    ).run()

    assert calls == [2]
    assert plan.manifest.status == "complete"
    assert [(stage.stage, stage.status, stage.message) for stage in plan.manifest.stages] == [
        (RunStage.RESOLVE_TAXON_SCOPE, StageStatus.COMPLETE, None),
        (RunStage.COMPILE_QUERIES, StageStatus.COMPLETE, None),
        (RunStage.BUILD_REGISTRY, StageStatus.COMPLETE, None),
    ]
    assert plan.manifest.stages[1].metrics == {"compiled_definitions": 4}
    assert plan.manifest.stages[2].metrics["registry_reused"] is True
    assert plan.manifest.stages[2].metrics["taxa_rows"] == 10
    assert plan.paths.manifest_path.exists()


def test_orchestrator_dry_run_marks_unimplemented_stages_skipped(tmp_path) -> None:
    scope = TaxonScope.from_species_context(_species_context())
    request = ProductionRunRequest(
        taxon="Danaus plexippus",
        rank="species",
        output_root=tmp_path / "runs",
        stages=(RunStage.RESOLVE_TAXON_SCOPE, RunStage.POLL_FLICKR),
        dry_run=True,
    )

    plan = ProductionRunOrchestrator(request, taxon_scope=scope).run()

    assert [stage.status for stage in plan.manifest.stages] == [StageStatus.COMPLETE, StageStatus.SKIPPED]
    assert plan.manifest.stages[1].message == "dry_run"
    assert plan.paths.manifest_path.exists()


def test_orchestrator_compiles_registry_queries_and_enqueues_flickr_work(tmp_path) -> None:
    registry = _write_rank_registry(tmp_path / "registry")
    _write_query_definitions(registry)
    workstore = SQLiteWorkStore(tmp_path / "workstore.sqlite")
    request = ProductionRunRequest(
        taxon="Papilio demoleus",
        rank="species",
        registry_dir=str(registry),
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        stages=(RunStage.RESOLVE_TAXON_SCOPE, RunStage.COMPILE_QUERIES, RunStage.ENQUEUE_FLICKR_WORK),
        limits={"records": 2},
    )

    plan = ProductionRunOrchestrator(request, workstore=workstore).run()

    assert plan.manifest.status == "complete"
    assert plan.manifest.query_counts == {"compiled_definitions": 2, "flickr_work_items": 2, "enqueued_work_items": 2}
    assert plan.manifest.stages[1].outputs["local_query_definitions"].endswith("/registry/flickr_query_definitions.parquet")
    work_items = workstore.list_work_items(
        job_name="biominer_production_run",
        stage=RunStage.POLL_FLICKR.value,
        registry_version="rank-registry-v1",
    )
    assert len(work_items) == 2
    assert work_items[0]["payload"]["run_id"] == "species_papilio_demoleus"
    assert work_items[0]["payload"]["query"]["accepted_taxon_key"] == "gbif:100"


def test_orchestrator_build_registry_stage_validates_local_registry(tmp_path) -> None:
    registry = _write_rank_registry(tmp_path / "registry")
    _write_query_definitions(registry)
    request = ProductionRunRequest(
        taxon="Papilio demoleus",
        rank="species",
        registry_dir=str(registry),
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        stages=(RunStage.BUILD_REGISTRY,),
    )

    result = ProductionRunOrchestrator(request).run()

    assert result.manifest.status == "complete"
    stage = result.manifest.stages[0]
    assert stage.status is StageStatus.COMPLETE
    assert stage.metrics["registry_reused"] is True
    assert stage.metrics["registry_version"] == "rank-registry-v1"
    assert stage.metrics["query_definition_rows"] == 2
    assert stage.outputs["query_definitions"].endswith("flickr_query_definitions.parquet")
    assert result.manifest.metrics["taxa_rows"] == 10


def test_orchestrator_build_registry_stage_fails_when_registry_artifacts_are_missing(tmp_path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    scope = TaxonScope.from_species_context(_species_context())
    request = ProductionRunRequest(
        taxon="Papilio demoleus",
        rank="species",
        registry_dir=str(registry),
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        stages=(RunStage.BUILD_REGISTRY,),
    )

    result = ProductionRunOrchestrator(request, taxon_scope=scope).run()

    assert result.manifest.status == "failed"
    assert result.manifest.stages[0].status is StageStatus.FAILED
    assert result.manifest.stages[0].message is not None
    assert "missing_registry_inputs:" in result.manifest.stages[0].message
    assert "taxa.parquet" in result.manifest.stages[0].message


def test_orchestrator_enqueue_is_idempotent_for_same_run_and_registry_queries(tmp_path) -> None:
    registry = _write_rank_registry(tmp_path / "registry")
    _write_query_definitions(registry)
    workstore = SQLiteWorkStore(tmp_path / "workstore.sqlite")
    request = ProductionRunRequest(
        taxon="Papilio demoleus",
        rank="species",
        registry_dir=str(registry),
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        stages=(RunStage.RESOLVE_TAXON_SCOPE, RunStage.ENQUEUE_FLICKR_WORK),
        limits={"records": 1},
    )

    first = ProductionRunOrchestrator(request, workstore=workstore).run()
    second = ProductionRunOrchestrator(request, workstore=workstore).run()

    assert first.manifest.query_counts["enqueued_work_items"] == 1
    assert second.manifest.query_counts["enqueued_work_items"] == 0
    assert second.manifest.stages[1].metrics["duplicate_work_items"] == 1


def test_orchestrator_polls_flickr_with_local_sqlite_and_fake_fetcher(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FLICKR_API_KEY", raising=False)
    registry = _write_rank_registry(tmp_path / "registry")
    _write_query_definitions(registry)
    request = ProductionRunRequest(
        taxon="Papilio demoleus",
        rank="species",
        registry_dir=str(registry),
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        stages=(RunStage.POLL_FLICKR,),
        limits={"records": 1, "api_calls": 5},
    )

    result = ProductionRunOrchestrator(request, metadata_fetcher=_fake_flickr_fetch).run()

    assert result.manifest.status == "complete"
    assert result.paths.source_records_path.exists()
    assert result.manifest.query_counts["polled_work_items"] == 1
    assert result.manifest.query_counts["api_calls_made"] == 1
    assert result.manifest.metrics["source_records_inserted"] == 1
    assert result.manifest.stages[0].outputs["source_records"] == str(result.paths.source_records_path)
    row = pl.read_parquet(result.paths.source_records_path).to_dicts()[0]
    assert row["flickr_photo_id"] == "poll-photo-1"
    assert row["tag_search_terms"] == ["Papilio demoleus"]
    assert row["query_hit_count"] == 1


def test_orchestrator_poll_stage_requires_fetcher_or_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FLICKR_API_KEY", raising=False)
    registry = _write_rank_registry(tmp_path / "registry")
    _write_query_definitions(registry)
    request = ProductionRunRequest(
        taxon="Papilio demoleus",
        rank="species",
        registry_dir=str(registry),
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        stages=(RunStage.POLL_FLICKR,),
        limits={"records": 1},
    )

    result = ProductionRunOrchestrator(request).run()

    assert result.manifest.status == "failed"
    assert result.manifest.stages[0].status is StageStatus.FAILED
    assert result.manifest.stages[0].message == "flickr_fetcher_or_api_key_required_for_poll_flickr"


def test_orchestrator_poll_stage_refuses_cloud_until_workstore_claiming_is_wired(tmp_path) -> None:
    registry = _write_rank_registry(tmp_path / "registry")
    _write_query_definitions(registry)
    request = ProductionRunRequest(
        taxon="Papilio demoleus",
        rank="species",
        registry_dir=str(registry),
        output_root="s3://biominer/runs",
        stages=(RunStage.POLL_FLICKR,),
    )

    result = ProductionRunOrchestrator(request, metadata_fetcher=_fake_flickr_fetch).run()

    assert result.manifest.status == "failed"
    assert result.manifest.stages[0].message == "poll_flickr_requires_local_sqlite_until_workstore_claiming_is_wired"


def test_orchestrator_joins_evidence_and_writes_summary_metrics(tmp_path) -> None:
    scope = TaxonScope.from_species_context(_species_context())
    request = ProductionRunRequest(
        taxon="Danaus plexippus",
        rank="species",
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        stages=(RunStage.JOIN_EVIDENCE, RunStage.SUMMARIZE),
    )
    plan = ProductionRunOrchestrator(request, taxon_scope=scope).plan()
    _write_join_stage_inputs(plan.paths)

    result = ProductionRunOrchestrator(request, taxon_scope=scope).run()

    assert result.manifest.status == "complete"
    assert result.paths.object_evidence_path.exists()
    assert result.paths.photo_summary_path.exists()
    assert result.paths.metrics_path.exists()
    assert result.manifest.evidence_counts == {"object_evidence_rows": 1, "photo_summary_rows": 1}
    assert result.manifest.metrics["object_occurrence_bin_counts"] == {"gold": 1}
    assert result.manifest.metrics["photo_occurrence_bin_counts"] == {"gold": 1}
    assert result.manifest.stages[0].outputs == {
        "object_evidence": str(result.paths.object_evidence_path),
        "photo_summary": str(result.paths.photo_summary_path),
    }
    assert result.manifest.stages[1].outputs == {"metrics": str(result.paths.metrics_path)}
    assert json.loads(result.paths.metrics_path.read_text(encoding="utf-8")) == {
        "object_evidence_rows": 1,
        "object_occurrence_bin_counts": {"gold": 1},
        "photo_occurrence_bin_counts": {"gold": 1},
        "photo_summary_rows": 1,
    }


def test_orchestrator_join_evidence_fails_when_local_inputs_are_missing(tmp_path) -> None:
    scope = TaxonScope.from_species_context(_species_context())
    request = ProductionRunRequest(
        taxon="Danaus plexippus",
        rank="species",
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        stages=(RunStage.JOIN_EVIDENCE,),
    )

    result = ProductionRunOrchestrator(request, taxon_scope=scope).run()

    assert result.manifest.status == "failed"
    assert result.manifest.stages[0].status is StageStatus.FAILED
    assert result.manifest.stages[0].message is not None
    assert result.manifest.stages[0].message.startswith("missing_join_inputs:")
    assert "canonical_source_records.parquet" in result.manifest.stages[0].message


def test_orchestrator_runs_local_detection_and_object_scoring_with_injected_fakes(tmp_path) -> None:
    scope = TaxonScope.from_species_context(_species_context())
    request = ProductionRunRequest(
        taxon="Danaus plexippus",
        rank="species",
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        stages=(RunStage.DETECT_OBJECTS, RunStage.SCORE_BIOCLIP),
    )
    plan = ProductionRunOrchestrator(request, taxon_scope=scope).plan()
    _write_source_records(plan.paths)
    detector = FakeObjectDetector(
        [[DetectionCandidate(label="butterfly_like", score=0.91, bbox_xyxy=(0.0, 0.0, 4.0, 4.0), objectness_score=0.91)]]
    )
    scorer = _ConstantObjectScorer(
        {
            "Danaus plexippus": 0.82,
            "a photo of Danaus plexippus": 0.81,
            "Monarch": 0.50,
            "Nymphalidae": 0.93,
            "Danaus": 0.90,
        }
    )

    result = ProductionRunOrchestrator(
        request,
        taxon_scope=scope,
        object_detector=detector,
        image_loader=lambda _record: _tiny_rgb_image(),
        object_scorer=scorer,
        allow_single_target_fixture=True,
    ).run()

    assert result.manifest.status == "complete"
    assert result.paths.object_detections_path.exists()
    assert result.paths.object_scores_path.exists()
    assert result.manifest.detection_counts == {
        "images_seen": 1,
        "detections": 1,
        "crops_created": 1,
        "images_loaded": 1,
        "image_failures": 0,
    }
    assert result.manifest.bioclip_counts["objects_scored"] == 1
    score_row = pl.read_parquet(result.paths.object_scores_path).to_dicts()[0]
    assert score_row["species_top1_scientific_name"] == "Danaus plexippus"
    assert score_row["target_species_score"] == 0.82
    assert result.manifest.stages[0].outputs["object_detections"] == str(result.paths.object_detections_path)
    assert result.manifest.stages[1].outputs["object_scores"] == str(result.paths.object_scores_path)


def test_orchestrator_detect_stage_fails_without_detector_runtime(tmp_path) -> None:
    scope = TaxonScope.from_species_context(_species_context())
    request = ProductionRunRequest(
        taxon="Danaus plexippus",
        rank="species",
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        stages=(RunStage.DETECT_OBJECTS,),
    )
    plan = ProductionRunOrchestrator(request, taxon_scope=scope).plan()
    _write_source_records(plan.paths)

    result = ProductionRunOrchestrator(request, taxon_scope=scope).run()

    assert result.manifest.status == "failed"
    assert result.manifest.stages[0].status is StageStatus.FAILED
    assert result.manifest.stages[0].message == "detector_runtime_required_for_detect_objects"


def test_run_paths_are_stable(tmp_path) -> None:
    paths = RunPaths.from_root(tmp_path, run_id="Family: Papilionidae")

    assert paths.run_root == tmp_path / "run_id=family_papilionidae"
    assert paths.query_definitions_path.name == "flickr_query_definitions.parquet"
    assert paths.object_evidence_path.name == "object_evidence_joined.parquet"
    assert paths.photo_summary_path.name == "photo_evidence_summary.parquet"
    assert paths.species_dir("Danaus plexippus") == paths.run_root / "species" / "danaus_plexippus"


def test_evidence_package_imports_and_metrics(tmp_path) -> None:
    joined = pl.DataFrame([{"occurrence_bin": "gold"}, {"occurrence_bin": "in_review"}, {"occurrence_bin": "gold"}])
    summary = pl.DataFrame([{"photo_occurrence_bin": "gold"}])

    assert evidence_count_metrics(joined, summary) == {
        "object_evidence_rows": 3,
        "photo_summary_rows": 1,
        "object_occurrence_bin_counts": {"gold": 2, "in_review": 1},
        "photo_occurrence_bin_counts": {"gold": 1},
    }
    assert callable(write_object_evidence_outputs)


def test_trust_policy_default_tiers_and_enablement() -> None:
    assert source_default_trust_tier("GBIF", "vernacular") is TrustTier.T2
    assert source_default_trust_tier("CoL", "scientific_synonym") is TrustTier.T1
    assert source_default_trust_tier("Wikidata", "vernacular") is TrustTier.T3
    assert source_default_trust_tier("iNaturalist", "vernacular_alias") is TrustTier.T4
    assert source_default_trust_tier("LibreTranslate", "generated_translation") is TrustTier.T5

    assert should_enable_name_by_default(TrustTier.T1, "low", "collision") is True
    assert should_enable_name_by_default(TrustTier.T2, "medium", "collision") is True
    assert should_enable_name_by_default(TrustTier.T3, "high", "none", external_taxon_link_confident=True) is True
    assert should_enable_name_by_default(TrustTier.T3, "high", "none") is False
    assert should_enable_name_by_default(TrustTier.T4, "medium", "none") is True
    assert should_enable_name_by_default(TrustTier.T4, "low", "none") is False
    assert should_enable_name_by_default(TrustTier.T5, "high", "none") is False
    assert should_enable_name_by_default(TrustTier.T5, "high", "none", review_state="accepted") is True


def test_trust_policy_disabled_reasons() -> None:
    assert disabled_reason_for_candidate(TrustTier.T3, "high", "none") == "wikidata_name_requires_confident_taxon_link"
    assert disabled_reason_for_candidate(TrustTier.T4, "medium", "ambiguous") == "name_collision_requires_review"
    assert disabled_reason_for_candidate(TrustTier.T5, "high", "none") == "generated_translation_requires_review"
    assert decide_name_trust(source="Wikidata", name_class="vernacular", confidence="high", collision_status="none").enabled is False


def _species_context() -> SpeciesContext:
    return SpeciesContext(
        scientific_name="Danaus plexippus",
        accepted_taxon_key="gbif:5130",
        canonical_name="Danaus plexippus",
        family="Nymphalidae",
        genus="Danaus",
        family_key="gbif:7017",
        genus_key="gbif:5131",
        species_key="gbif:5130",
        registry_version="test-v1",
        common_names=(CommonName(name="Monarch", language="en", source="GBIF", trust_tier="T2"),),
    )


def _write_rank_registry(registry: Path) -> Path:
    registry.mkdir(parents=True, exist_ok=True)
    taxa_rows = [
        _taxon_row("gbif:10", "Papilionidae", "FAMILY", family_key="gbif:10", family="Papilionidae"),
        _taxon_row("gbif:90", "Papilio", "GENUS", parent_key="gbif:10", family_key="gbif:10", family="Papilionidae", genus_key="gbif:90", genus="Papilio"),
        _taxon_row("gbif:91", "Emptygenus", "GENUS", parent_key="gbif:10", family_key="gbif:10", family="Papilionidae", genus_key="gbif:91", genus="Emptygenus"),
        _taxon_row("gbif:100", "Papilio demoleus", "SPECIES", parent_key="gbif:90", family_key="gbif:10", family="Papilionidae", genus_key="gbif:90", genus="Papilio", species_key="gbif:100", species="Papilio demoleus"),
        _taxon_row("gbif:101", "Papilio machaon", "SPECIES", parent_key="gbif:90", family_key="gbif:10", family="Papilionidae", genus_key="gbif:90", genus="Papilio", species_key="gbif:101", species="Papilio machaon"),
        _taxon_row("gbif:20", "Nymphalidae", "FAMILY", family_key="gbif:20", family="Nymphalidae"),
        _taxon_row("gbif:190", "Danaus", "GENUS", parent_key="gbif:20", family_key="gbif:20", family="Nymphalidae", genus_key="gbif:190", genus="Danaus"),
        _taxon_row("gbif:200", "Danaus plexippus", "SPECIES", parent_key="gbif:190", family_key="gbif:20", family="Nymphalidae", genus_key="gbif:190", genus="Danaus", species_key="gbif:200", species="Danaus plexippus"),
        _taxon_row("gbif:300", "Shared name", "GENUS", parent_key="gbif:10", family_key="gbif:10", family="Papilionidae", genus_key="gbif:300", genus="Shared name"),
        _taxon_row("gbif:301", "Shared name", "SPECIES", parent_key="gbif:300", family_key="gbif:10", family="Papilionidae", genus_key="gbif:300", genus="Shared name", species_key="gbif:301", species="Shared name"),
    ]
    pl.DataFrame(taxa_rows).write_parquet(registry / "taxa.parquet")
    pl.DataFrame(
        [
            _name_row("gbif:100", "Papilio demoleus", "accepted_scientific", "la", "T1"),
            _name_row("gbif:100", "Lime butterfly", "vernacular", "eng", "T2"),
            _name_row("gbif:101", "Papilio machaon", "accepted_scientific", "la", "T1"),
            _name_row("gbif:200", "Danaus plexippus", "accepted_scientific", "la", "T1"),
            _name_row("gbif:301", "Shared name", "accepted_scientific", "la", "T1"),
        ]
    ).write_parquet(registry / "names.parquet")
    pl.DataFrame([{"source": "GBIF", "source_version": "fixture", "retrieved_at": "2026-01-01T00:00:00Z"}]).write_parquet(registry / "source_snapshots.parquet")
    (registry / "manifest.json").write_text(json.dumps({"registry_version": "rank-registry-v1"}), encoding="utf-8")
    return registry


def _write_query_definitions(registry: Path) -> None:
    pl.DataFrame(
        [
            _query_definition_row("q-tags", "Papilio demoleus", "tags", 10),
            _query_definition_row("q-text", "Lime butterfly", "text", 20),
        ]
    ).write_parquet(registry / "flickr_query_definitions.parquet")


def _write_source_records(paths: RunPaths) -> None:
    paths.ensure_directories()
    pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "source_record_hash": "sha256:source-1",
                "image_url": "memory://photo-1",
                "photo_page_url": "https://www.flickr.com/photos/u/photo-1",
                "title": "Monarch butterfly",
                "description": "Danaus plexippus on milkweed",
                "tags": ["monarch", "butterfly"],
                "latitude": 42.1,
                "longitude": -83.0,
                "date_taken": "2025-07-01",
            }
        ]
    ).write_parquet(paths.source_records_path)


def _write_join_stage_inputs(paths: RunPaths) -> None:
    paths.ensure_directories()
    _write_source_records(paths)
    pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "detection_id": "det-1",
                "crop_hash": "sha256:crop-1",
                "bbox_xyxy": [10.0, 12.0, 100.0, 120.0],
                "bbox_xyxyn": [0.1, 0.12, 0.8, 0.9],
                "detector_label": "butterfly_like",
                "detector_score": 0.91,
                "objectness_score": 0.91,
                "detection_status": "detected",
                "backend": "fake",
                "model_id": "fake-detector",
                "model_version": "test",
                "checkpoint": "none",
            }
        ]
    ).write_parquet(paths.object_detections_path)
    pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "detection_id": "det-1",
                "crop_hash": "sha256:crop-1",
                "model_id": "fake-bioclip",
                "model_version": "test",
                "model_checkpoint": "fake-checkpoint",
                "candidate_set_id": "fixture",
                "classified_at": "2026-01-01T00:00:00Z",
                "ablation_mode": "detector_crop",
                "family_top3": ["Nymphalidae"],
                "family_top1": "Nymphalidae",
                "family_top1_score": 0.95,
                "family_margin": 0.40,
                "genus_top8": ["Danaus"],
                "genus_top1": "Danaus",
                "genus_top1_score": 0.90,
                "genus_margin": 0.35,
                "species_top20": ["Danaus plexippus"],
                "species_top20_accepted_taxon_keys": ["gbif:5130"],
                "species_top5": ["Danaus plexippus"],
                "species_top5_accepted_taxon_keys": ["gbif:5130"],
                "species_top1": "Danaus plexippus",
                "species_top1_scientific_name": "Danaus plexippus",
                "species_top1_accepted_taxon_key": "gbif:5130",
                "accepted_taxon_key": "gbif:5130",
                "species_top1_score": 0.82,
                "species_top1_margin": 0.31,
                "target_accepted_taxon_key": "gbif:5130",
                "target_species_score": 0.82,
                "target_species_rank": 1,
                "geospatial_prior_score": 0.10,
                "geospatial_prior_reason": "within_context_region",
                "text_evidence_score": 0.50,
                "comment_evidence_score": 0.0,
                "is_target_positive": True,
                "is_negative_material": False,
                "occurrence_bin": "gold",
                "bin_reason": "target_species_score_ge_070",
            }
        ]
    ).write_parquet(paths.object_scores_path)


def _tiny_rgb_image() -> DecodedImage:
    return DecodedImage(width=4, height=4, mode="RGB", data=bytes([255, 255, 255] * 16), source_uri="memory://photo-1")


def _fake_flickr_fetch(_query: object) -> dict[str, object]:
    return {
        "photos": {
            "total": "1",
            "pages": "1",
            "page": "1",
            "perpage": "500",
            "photo": [
                {
                    "id": "poll-photo-1",
                    "title": "Papilio demoleus on citrus",
                    "url_l": "https://live.staticflickr.com/poll-photo-1.jpg",
                    "datetaken": "2025-03-01 10:30:00",
                }
            ],
        }
    }


class _ConstantObjectScorer:
    model_id = "fake-bioclip"
    model_version = "test"
    model_checkpoint = "fake-checkpoint"

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def score(self, _item: dict[str, object], labels: tuple[str, ...]) -> dict[str, float]:
        return {label: float(self.scores.get(label, 0.0)) for label in labels}


def _query_definition_row(query_definition_id: str, source_term: str, search_field: str, search_priority: int) -> dict[str, object]:
    return {
        "query_definition_id": query_definition_id,
        "registry_version": "rank-registry-v1",
        "accepted_taxon_key": "gbif:100",
        "accepted_scientific_name": "Papilio demoleus",
        "family_key": "gbif:10",
        "genus_key": "gbif:90",
        "species_key": "gbif:100",
        "source_term": source_term,
        "language": "en",
        "search_field": search_field,
        "search_priority": search_priority,
        "bbox": "",
        "region": "",
        "name_class": "accepted_scientific" if search_field == "tags" else "vernacular",
        "confidence": "high",
        "enabled": True,
    }


def _taxon_row(
    accepted_taxon_key: str,
    scientific_name: str,
    rank: str,
    *,
    parent_key: str = "",
    family_key: str = "",
    family: str = "",
    genus_key: str = "",
    genus: str = "",
    species_key: str = "",
    species: str = "",
) -> dict[str, str]:
    return {
        "accepted_taxon_key": accepted_taxon_key,
        "scientific_name": scientific_name,
        "rank": rank,
        "parent_key": parent_key,
        "family_key": family_key,
        "family": family,
        "genus_key": genus_key,
        "genus": genus,
        "species_key": species_key,
        "species": species,
    }


def _name_row(accepted_taxon_key: str, display_name: str, name_class: str, language: str, trust_tier: str) -> dict[str, object]:
    return {
        "name_id": f"name:{accepted_taxon_key}:{display_name}",
        "registry_version": "rank-registry-v1",
        "accepted_taxon_key": accepted_taxon_key,
        "verbatim_name": display_name,
        "display_name": display_name,
        "language": language,
        "script": "Latn",
        "region": "",
        "bbox": "",
        "name_class": name_class,
        "source": "GBIF",
        "source_record_id": accepted_taxon_key,
        "trust_tier": trust_tier,
        "precision_tier": "high",
        "confidence": "high",
        "enabled": True,
        "disabled_reason": "",
    }
