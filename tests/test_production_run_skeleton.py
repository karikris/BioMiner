from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

import biominer.run.orchestrator as run_orchestrator_module
from biominer.bioclip.object_runner import OBJECT_VISUAL_MODES, PRIMARY_VISUAL_CLASSIFIER
from biominer.detection.detector_base import DecodedImage, DetectionCandidate, FakeObjectDetector
from biominer.evidence import build_object_evidence_frames, build_review_queue, evidence_count_metrics
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
    resolve_taxon_scope_from_registry_frames,
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


def test_resolve_taxon_scope_from_registry_frames_matches_path_resolver(tmp_path) -> None:
    registry = _write_rank_registry(tmp_path / "registry")

    path_scope = resolve_taxon_scope_from_registry(registry_dir=registry, input_name="Papilio", input_rank="genus")
    frame_scope = resolve_taxon_scope_from_registry_frames(
        taxa=pl.read_parquet(registry / "taxa.parquet"),
        names=pl.read_parquet(registry / "names.parquet"),
        source_snapshots=pl.read_parquet(registry / "source_snapshots.parquet"),
        manifest=json.loads((registry / "manifest.json").read_text(encoding="utf-8")),
        input_name="Papilio",
        input_rank="genus",
    )

    assert frame_scope == path_scope


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
    assert manifest.evidence_counts == {"object_evidence_rows": 0, "photo_summary_rows": 0, "review_queue_rows": 0}
    assert manifest.outputs["manifest"].endswith("/run_manifest.json")
    assert manifest.outputs["review_queue"].endswith("/reports/review_queue.parquet")
    assert [stage.stage for stage in manifest.stages][:3] == [
        RunStage.RESOLVE_TAXON_SCOPE,
        RunStage.BUILD_REGISTRY,
        RunStage.COMPILE_QUERIES,
    ]
    assert [stage.stage for stage in manifest.stages][-3:] == [
        RunStage.QUEUE_COMMENT_REVIEW,
        RunStage.REVIEW_COMMENTS,
        RunStage.APPLY_COMMENT_REVIEW,
    ]
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["species_count"] == 1


def test_production_run_plan_defaults_to_detector_crop_only(tmp_path) -> None:
    scope = TaxonScope.from_species_context(_species_context())
    request = ProductionRunRequest(taxon="Danaus plexippus", rank="species", output_root=tmp_path, dry_run=True)
    plan = ProductionRunOrchestrator(request, taxon_scope=scope).plan()

    assert request.bioclip_ablation_modes == ("detector_crop",)
    assert run_orchestrator_module._request_bioclip_modes(request) == ("detector_crop",)
    assert plan.manifest.model_configs["bioclip_ablation_modes"] == ["detector_crop"]
    assert "whole_image" not in plan.manifest.model_configs["bioclip_ablation_modes"]


def test_run_artifact_uris_are_s3_safe_and_species_scoped() -> None:
    uris = RunArtifactUris.from_prefix("s3://biominer/runs", run_id="Family: Papilionidae")

    assert uris.run_root_uri == "s3://biominer/runs/run_id=family_papilionidae"
    assert uris.manifest_uri == "s3://biominer/runs/run_id=family_papilionidae/run_manifest.json"
    assert uris.review_queue_uri == "s3://biominer/runs/run_id=family_papilionidae/reports/review_queue.parquet"
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
    _write_query_definitions(registry)
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


def test_orchestrator_enqueues_t5_registry_queries_for_flickr_api(tmp_path, monkeypatch) -> None:
    registry = _write_rank_registry(tmp_path / "registry")
    _write_t5_query_definitions(registry)
    workstore = SQLiteWorkStore(tmp_path / "workstore.sqlite")
    real_query_loader = run_orchestrator_module.load_registry_flickr_queries_from_frame

    def one_day_query_window(frame):  # noqa: ANN001 - test double mirrors the query loader signature used by the orchestrator.
        return real_query_loader(
            frame,
            start_date="2026-01-01",
            end_date="2026-01-01",
            slice_days=1,
        )

    monkeypatch.setattr(run_orchestrator_module, "load_registry_flickr_queries_from_frame", one_day_query_window)
    request = ProductionRunRequest(
        taxon="Papilio demoleus",
        rank="species",
        registry_dir=str(registry),
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        stages=(RunStage.COMPILE_QUERIES, RunStage.ENQUEUE_FLICKR_WORK),
        limits={"records": 2},
    )

    plan = ProductionRunOrchestrator(request, workstore=workstore).run()
    work_items = workstore.list_work_items(
        job_name="biominer_production_run",
        stage=RunStage.POLL_FLICKR.value,
        registry_version="rank-registry-v1",
    )
    queued_queries = sorted((item["payload"]["query"] for item in work_items), key=lambda query: query["search_field"])

    assert plan.manifest.status == "complete"
    assert plan.manifest.query_counts == {"compiled_definitions": 2, "flickr_work_items": 2, "enqueued_work_items": 2}
    assert [query["term"] for query in queued_queries] == ["Translated Lime", "Translated Lime"]
    assert [query["search_field"] for query in queued_queries] == ["tags", "text"]
    assert [query["term_type"] for query in queued_queries] == ["generated_translation", "generated_translation"]
    assert [query["trust_tier"] for query in queued_queries] == ["T5", "T5"]
    assert [query["query_definition_id"] for query in queued_queries] == ["q-t5-tags", "q-t5-text"]


def test_orchestrator_compile_queries_filters_registry_definitions_to_species_scope(tmp_path) -> None:
    registry = _write_rank_registry(tmp_path / "registry")
    _write_query_definitions_with_out_of_scope_taxa(registry)
    request = ProductionRunRequest(
        taxon="Papilio demoleus",
        rank="species",
        registry_dir=str(registry),
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        stages=(RunStage.COMPILE_QUERIES,),
    )

    plan = ProductionRunOrchestrator(request).run()
    compiled = pl.read_parquet(plan.paths.query_definitions_path).sort("query_definition_id")

    assert plan.manifest.status == "complete"
    assert plan.manifest.stages[0].metrics["registry_query_definition_source_rows"] == 4
    assert plan.manifest.stages[0].metrics["registry_query_definition_rows"] == 2
    assert compiled.select("query_definition_id").to_series().to_list() == ["q-demoleus-tags", "q-demoleus-text"]
    assert compiled.select("accepted_taxon_key").to_series().unique().to_list() == ["gbif:100"]


def test_orchestrator_genus_run_enqueues_only_expanded_species_query_definitions(tmp_path) -> None:
    registry = _write_rank_registry(tmp_path / "registry")
    _write_query_definitions_with_out_of_scope_taxa(registry)
    workstore = SQLiteWorkStore(tmp_path / "workstore.sqlite")
    request = ProductionRunRequest(
        taxon="Papilio",
        rank="genus",
        registry_dir=str(registry),
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        stages=(RunStage.COMPILE_QUERIES, RunStage.ENQUEUE_FLICKR_WORK),
        limits={"records": 3},
    )

    plan = ProductionRunOrchestrator(request, workstore=workstore).run()
    work_items = workstore.list_work_items(
        job_name="biominer_production_run",
        stage=RunStage.POLL_FLICKR.value,
        registry_version="rank-registry-v1",
    )

    assert plan.manifest.status == "complete"
    assert plan.manifest.taxon_scope.species_names == ("Papilio demoleus", "Papilio machaon")
    assert plan.manifest.query_counts == {"compiled_definitions": 3, "flickr_work_items": 3, "enqueued_work_items": 3}
    compiled = pl.read_parquet(plan.paths.query_definitions_path)
    assert sorted(compiled.select("accepted_taxon_key").to_series().to_list()) == ["gbif:100", "gbif:100", "gbif:101"]
    queued_keys = {item["payload"]["query"]["accepted_taxon_key"] for item in work_items}
    assert queued_keys <= {"gbif:100", "gbif:101"}
    assert "gbif:200" not in queued_keys


def test_orchestrator_limit_species_bounds_resolved_scope_and_compiled_queries(tmp_path) -> None:
    registry = _write_rank_registry(tmp_path / "registry")
    _write_query_definitions_with_out_of_scope_taxa(registry)
    request = ProductionRunRequest(
        taxon="Papilio",
        rank="genus",
        registry_dir=str(registry),
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        stages=(RunStage.RESOLVE_TAXON_SCOPE, RunStage.COMPILE_QUERIES),
        limits={"species": 1, "records": 2},
    )

    plan = ProductionRunOrchestrator(request).run()
    compiled = pl.read_parquet(plan.paths.query_definitions_path).sort("query_definition_id")

    assert plan.manifest.status == "complete"
    assert plan.manifest.taxon_scope.species_names == ("Papilio demoleus",)
    assert plan.manifest.metrics["expanded_species_count"] == 1
    assert plan.manifest.query_counts["compiled_definitions"] == 2
    assert plan.manifest.query_counts["flickr_work_items"] == 2
    assert compiled.select("query_definition_id").to_series().to_list() == ["q-demoleus-tags", "q-demoleus-text"]
    assert compiled.select("accepted_taxon_key").to_series().unique().to_list() == ["gbif:100"]


def test_orchestrator_reads_registry_inputs_from_cloud_storage(tmp_path) -> None:
    registry = _write_rank_registry(tmp_path / "registry")
    _write_query_definitions(registry)
    storage = _FakeRunStorage()
    registry_uri = "s3://biominer/registry/current"
    _seed_cloud_registry(storage, registry_uri, registry)
    request = ProductionRunRequest(
        taxon="Papilio",
        rank="genus",
        registry_dir=registry_uri,
        output_root="s3://biominer/runs",
        stages=(RunStage.RESOLVE_TAXON_SCOPE, RunStage.BUILD_REGISTRY, RunStage.COMPILE_QUERIES),
        limits={"records": 1},
    )

    plan = ProductionRunOrchestrator(request, storage=storage).run()

    assert plan.manifest.status == "complete"
    assert plan.manifest.taxon_scope.accepted_rank == "genus"
    assert plan.manifest.taxon_scope.species_names == ("Papilio demoleus", "Papilio machaon")
    assert plan.manifest.stages[1].outputs["taxa"] == f"{registry_uri}/taxa.parquet"
    assert plan.manifest.stages[1].metrics["registry_version"] == "rank-registry-v1"
    assert plan.manifest.stages[2].outputs["source_query_definitions"] == f"{registry_uri}/flickr_query_definitions.parquet"
    assert plan.manifest.query_counts["flickr_work_items"] == 1
    assert storage.parquet_payloads[plan.artifact_uris.query_definitions_uri].height == 2


def test_orchestrator_no_longer_reports_s3_registry_reads_as_unimplemented() -> None:
    source = Path("src/biominer/run/orchestrator.py").read_text(encoding="utf-8")

    assert "S3 registry reads are not implemented" not in source


def test_orchestrator_writes_compiled_queries_to_cloud_storage(tmp_path) -> None:
    registry = _write_rank_registry(tmp_path / "registry")
    _write_query_definitions(registry)
    storage = _FakeRunStorage()
    request = ProductionRunRequest(
        taxon="Papilio demoleus",
        rank="species",
        registry_dir=str(registry),
        output_root="s3://biominer/runs",
        stages=(RunStage.COMPILE_QUERIES,),
        limits={"records": 1},
    )

    plan = ProductionRunOrchestrator(request, storage=storage).run()

    expected_uri = "s3://biominer/runs/run_id=species_papilio_demoleus/registry/flickr_query_definitions.parquet"
    assert plan.manifest.status == "complete"
    assert plan.manifest.query_counts == {"compiled_definitions": 2, "flickr_work_items": 1, "enqueued_work_items": 0}
    assert plan.manifest.stages[0].outputs["query_definitions"] == expected_uri
    assert list(storage.parquet_payloads) == [expected_uri]
    assert storage.parquet_payloads[expected_uri].select("query_definition_id").to_series().to_list() == ["q-tags", "q-text"]
    assert storage.json_payloads[plan.artifact_uris.manifest_uri]["status"] == "complete"


def test_orchestrator_cloud_compile_requires_storage_backend(tmp_path) -> None:
    registry = _write_rank_registry(tmp_path / "registry")
    _write_query_definitions(registry)
    request = ProductionRunRequest(
        taxon="Papilio demoleus",
        rank="species",
        registry_dir=str(registry),
        output_root="s3://biominer/runs",
        stages=(RunStage.COMPILE_QUERIES,),
    )

    plan = ProductionRunOrchestrator(request).run()

    assert plan.manifest.status == "failed"
    assert plan.manifest.stages[0].message == "storage_backend_required_for_compile_queries"


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


def test_orchestrator_build_registry_stage_can_build_missing_registry_when_enabled(tmp_path) -> None:
    registry = tmp_path / "built-registry"
    scope = TaxonScope.from_species_context(_species_context())
    request = ProductionRunRequest(
        taxon="Danaus plexippus",
        rank="species",
        registry_dir=str(registry),
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        stages=(RunStage.BUILD_REGISTRY,),
        build_registry_if_missing=True,
    )
    calls: list[Path] = []

    def fake_build(target: Path) -> dict[str, object]:
        calls.append(target)
        _write_rank_registry(target)
        _write_query_definitions(target)
        return {"registry_version": "rank-registry-v1"}

    result = ProductionRunOrchestrator(request, taxon_scope=scope, registry_builder=fake_build).run()

    assert result.manifest.status == "complete"
    assert calls == [registry]
    assert result.manifest.stages[0].metrics["registry_reused"] is False
    assert result.manifest.stages[0].metrics["query_definition_rows"] == 2


def test_orchestrator_build_registry_stage_requires_registry_query_definitions(tmp_path) -> None:
    registry = _write_rank_registry(tmp_path / "registry")
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

    assert result.manifest.status == "failed"
    assert result.manifest.stages[0].status is StageStatus.FAILED
    assert result.manifest.stages[0].message is not None
    assert "missing_registry_inputs:" in result.manifest.stages[0].message
    assert "flickr_query_definitions.parquet" in result.manifest.stages[0].message


def test_orchestrator_compile_queries_fails_cleanly_when_registry_query_definitions_missing(tmp_path) -> None:
    registry = _write_rank_registry(tmp_path / "registry")
    request = ProductionRunRequest(
        taxon="Papilio demoleus",
        rank="species",
        registry_dir=str(registry),
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        stages=(RunStage.COMPILE_QUERIES,),
    )

    result = ProductionRunOrchestrator(request).run()

    assert result.manifest.status == "failed"
    assert result.manifest.stages[0].status is StageStatus.FAILED
    assert result.manifest.stages[0].message is not None
    assert result.manifest.stages[0].message.startswith("missing_registry_query_definitions:")
    assert "flickr_query_definitions.parquet" in result.manifest.stages[0].message


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


def test_orchestrator_polls_t5_registry_queries_through_flickr_api(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FLICKR_API_KEY", raising=False)
    registry = _write_rank_registry(tmp_path / "registry")
    _write_t5_query_definitions(registry)
    real_query_loader = run_orchestrator_module.load_registry_flickr_queries_from_frame
    seen_queries = []

    def one_day_query_window(frame):  # noqa: ANN001 - test double mirrors the query loader signature used by the orchestrator.
        return real_query_loader(
            frame,
            start_date="2026-01-01",
            end_date="2026-01-01",
            slice_days=1,
        )

    def fake_t5_fetch(query):  # noqa: ANN001 - test double records the exact poller query.
        seen_queries.append(query)
        return {
            "photos": {
                "page": str(query.page),
                "pages": "1",
                "perpage": str(query.per_page),
                "total": "1",
                "photo": [
                    {
                        "id": "t5-photo-1",
                        "title": "retrieved by generated translation",
                        "url_l": "https://live.staticflickr.com/t5-photo-1.jpg",
                    }
                ],
            }
        }

    monkeypatch.setattr(run_orchestrator_module, "load_registry_flickr_queries_from_frame", one_day_query_window)
    request = ProductionRunRequest(
        taxon="Papilio demoleus",
        rank="species",
        registry_dir=str(registry),
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        stages=(RunStage.POLL_FLICKR,),
        limits={"records": 2, "api_calls": 5, "workers": 1},
    )

    result = ProductionRunOrchestrator(request, metadata_fetcher=fake_t5_fetch).run()
    queries = sorted(seen_queries, key=lambda query: query.search_field)

    assert result.manifest.status == "complete"
    assert [(query.search_field, query.term) for query in queries] == [
        ("tags", "Translated Lime"),
        ("text", "Translated Lime"),
    ]
    assert [query.trust_tier for query in queries] == ["T5", "T5"]
    assert [query.term_type for query in queries] == ["generated_translation", "generated_translation"]
    assert [query.query_definition_id for query in queries] == ["q-t5-tags", "q-t5-text"]
    frame = pl.read_parquet(result.paths.source_records_path)
    assert frame.height == 1
    row = frame.to_dicts()[0]
    assert row["flickr_photo_id"] == "t5-photo-1"
    assert row["tag_search_terms"] == ["Translated Lime"]
    assert row["text_search_terms"] == ["Translated Lime"]
    assert row["all_query_labels"] == ["tags:Translated Lime", "text:Translated Lime"]
    assert row["query_definition_ids"] == ["q-t5-tags", "q-t5-text"]
    assert row["query_hit_count"] == 2


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


def test_orchestrator_poll_stage_claims_workstore_and_writes_cloud_source_records(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FLICKR_API_KEY", raising=False)
    registry = _write_rank_registry(tmp_path / "registry")
    _write_query_definitions(registry)
    storage = _FakeRunStorage()
    workstore = SQLiteWorkStore(tmp_path / "workstore.sqlite")
    request = ProductionRunRequest(
        taxon="Papilio demoleus",
        rank="species",
        registry_dir=str(registry),
        output_root="s3://biominer/runs",
        storage_backend="s3",
        workstore_backend="postgres",
        worker_id="worker-from-request",
        stages=(RunStage.ENQUEUE_FLICKR_WORK, RunStage.POLL_FLICKR),
        limits={"records": 1, "api_calls": 5},
    )

    result = ProductionRunOrchestrator(request, storage=storage, workstore=workstore, metadata_fetcher=_fake_flickr_fetch).run()

    assert result.manifest.status == "complete"
    assert result.manifest.query_counts["enqueued_work_items"] == 1
    assert result.manifest.query_counts["polled_work_items"] == 1
    poll_stage = result.manifest.stages[1]
    assert poll_stage.metrics["workstore_work_items_completed"] == 1
    assert poll_stage.metrics["source_record_shards"] == 1
    source_records_uri = poll_stage.outputs["source_records"]
    assert source_records_uri.startswith(result.artifact_uris.staging_uri + "/evidence/stage=poll_flickr/")
    assert source_records_uri in storage.parquet_payloads
    row = storage.parquet_payloads[source_records_uri].to_dicts()[0]
    assert row["flickr_photo_id"] == "poll-photo-1"
    assert row["tag_search_terms"] == ["Papilio demoleus"]
    work_items = workstore.list_work_items(
        job_name="biominer_production_run",
        stage=RunStage.POLL_FLICKR.value,
        registry_version="rank-registry-v1",
    )
    assert [item["status"] for item in work_items] == ["completed"]
    assert work_items[0]["claimed_by"] == "worker-from-request"
    assert work_items[0]["output_uri"] == source_records_uri


def test_orchestrator_cloud_poll_reenqueues_reported_followup_pages(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FLICKR_API_KEY", raising=False)
    registry = _write_rank_registry(tmp_path / "registry")
    _write_query_definitions(registry)
    storage = _FakeRunStorage()
    workstore = SQLiteWorkStore(tmp_path / "workstore.sqlite")
    request = ProductionRunRequest(
        taxon="Papilio demoleus",
        rank="species",
        registry_dir=str(registry),
        output_root="s3://biominer/runs",
        storage_backend="s3",
        workstore_backend="postgres",
        stages=(RunStage.ENQUEUE_FLICKR_WORK, RunStage.POLL_FLICKR),
        limits={"records": 1, "api_calls": 5},
    )

    result = ProductionRunOrchestrator(request, storage=storage, workstore=workstore, metadata_fetcher=_fake_flickr_fetch_two_pages).run()

    assert result.manifest.status == "complete"
    poll_stage = result.manifest.stages[1]
    assert poll_stage.metrics["workstore_work_items_completed"] == 1
    assert poll_stage.metrics["workstore_followup_work_items_enqueued"] == 1
    work_items = workstore.list_work_items(
        job_name="biominer_production_run",
        stage=RunStage.POLL_FLICKR.value,
        registry_version="rank-registry-v1",
    )
    statuses_by_page = {item["payload"]["query"]["page"]: item["status"] for item in work_items}
    assert statuses_by_page == {1: "completed", 2: "pending"}


def test_orchestrator_runs_fake_backed_cloud_workflow_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FLICKR_API_KEY", raising=False)
    registry = _write_rank_registry(tmp_path / "registry")
    _write_query_definitions(registry)
    storage = _FakeRunStorage()
    workstore = SQLiteWorkStore(tmp_path / "workstore.sqlite")
    detector = FakeObjectDetector(
        [[DetectionCandidate(label="butterfly_like", score=0.91, bbox_xyxy=(0.0, 0.0, 4.0, 4.0), objectness_score=0.91)]]
    )
    scorer = _ConstantObjectScorer(
        {
            "Papilio demoleus": 0.84,
            "a photo of Papilio demoleus": 0.83,
            "Lime butterfly": 0.61,
            "Papilionidae": 0.91,
            "Papilio": 0.88,
        }
    )
    request = ProductionRunRequest(
        taxon="Papilio demoleus",
        rank="species",
        registry_dir=str(registry),
        output_root="s3://biominer/runs",
        storage_backend="s3",
        workstore_backend="postgres",
        stages=(
            RunStage.COMPILE_QUERIES,
            RunStage.ENQUEUE_FLICKR_WORK,
            RunStage.POLL_FLICKR,
            RunStage.DETECT_OBJECTS,
            RunStage.SCORE_BIOCLIP,
            RunStage.JOIN_EVIDENCE,
            RunStage.SUMMARIZE,
        ),
        limits={"records": 1, "api_calls": 5},
    )

    result = ProductionRunOrchestrator(
        request,
        storage=storage,
        workstore=workstore,
        metadata_fetcher=_fake_flickr_fetch,
        object_detector=detector,
        image_loader=lambda _record: _tiny_rgb_image(),
        object_scorer=scorer,
        allow_single_target_fixture=True,
    ).run()

    assert result.manifest.status == "complete"
    assert result.manifest.query_counts["compiled_definitions"] == 2
    assert result.manifest.query_counts["enqueued_work_items"] == 1
    assert result.manifest.query_counts["polled_work_items"] == 1
    assert result.manifest.detection_counts["detections"] == 1
    assert result.manifest.bioclip_counts["objects_scored"] == 1
    assert result.manifest.bioclip_counts["whole_images_scored"] == 0
    assert result.manifest.bioclip_counts["detector_crops_scored"] == 1
    assert result.manifest.bioclip_counts["segmentation_crops_scored"] == 0
    assert result.manifest.evidence_counts["object_evidence_rows"] == 1
    assert result.manifest.evidence_counts["photo_summary_rows"] == 1
    poll_shard_uri = result.manifest.stages[2].outputs["source_records"]
    assert poll_shard_uri.startswith(result.artifact_uris.staging_uri + "/evidence/stage=poll_flickr/")
    assert poll_shard_uri in storage.parquet_payloads
    detection_uri = result.manifest.stages[3].outputs["object_detections"]
    score_uri = result.manifest.stages[4].outputs["object_scores"]
    object_evidence_uri = result.manifest.stages[5].outputs["object_evidence"]
    photo_summary_uri = result.manifest.stages[6].outputs["photo_summary"]
    review_queue_uri = result.manifest.stages[6].outputs["review_queue"]
    for uri in (
        result.artifact_uris.query_definitions_uri,
        detection_uri,
        score_uri,
        object_evidence_uri,
        photo_summary_uri,
        review_queue_uri,
    ):
        assert uri in storage.parquet_payloads
    assert detection_uri.startswith(result.artifact_uris.staging_uri + "/evidence/stage=detect_objects/")
    assert score_uri.startswith(result.artifact_uris.staging_uri + "/evidence/stage=score_bioclip/")
    assert object_evidence_uri.startswith(result.artifact_uris.staging_uri + "/evidence/stage=join_evidence/")
    assert photo_summary_uri.startswith(result.artifact_uris.staging_uri + "/evidence/stage=photo_summary/")
    assert review_queue_uri.startswith(result.artifact_uris.staging_uri + "/evidence/stage=review_queue/")
    assert storage.json_payloads[result.artifact_uris.manifest_uri]["status"] == "complete"
    summary = storage.parquet_payloads[photo_summary_uri].to_dicts()[0]
    assert summary["flickr_photo_id"] == "poll-photo-1"
    assert summary["best_object_species_top1"] == "Papilio demoleus"


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
    assert result.paths.review_queue_path.exists()
    assert pl.read_parquet(result.paths.review_queue_path).height == 0
    assert result.manifest.evidence_counts == {"object_evidence_rows": 1, "photo_summary_rows": 1, "review_queue_rows": 0}
    assert result.manifest.metrics["object_occurrence_bin_counts"] == {"gold": 1}
    assert result.manifest.metrics["photo_occurrence_bin_counts"] == {"gold": 1}
    assert result.manifest.metrics["review_queue_bin_counts"] == {}
    assert result.manifest.stages[0].outputs == {
        "object_evidence": str(result.paths.object_evidence_path),
        "photo_summary": str(result.paths.photo_summary_path),
    }
    assert result.manifest.stages[1].outputs == {
        "metrics": str(result.paths.metrics_path),
        "review_queue": str(result.paths.review_queue_path),
    }
    assert json.loads(result.paths.metrics_path.read_text(encoding="utf-8")) == {
        "object_evidence_rows": 1,
        "object_occurrence_bin_counts": {"gold": 1},
        "photo_occurrence_bin_counts": {"gold": 1},
        "photo_summary_rows": 1,
        "review_queue_bin_counts": {},
        "review_queue_rows": 0,
    }


def test_orchestrator_joins_evidence_from_cloud_shard_inventory(tmp_path) -> None:
    scope = TaxonScope.from_species_context(_species_context())
    storage = _FakeRunStorage()
    workstore = SQLiteWorkStore(tmp_path / "workstore.sqlite")
    request = ProductionRunRequest(
        taxon="Danaus plexippus",
        rank="species",
        output_root="s3://biominer/runs",
        stages=(RunStage.JOIN_EVIDENCE,),
    )
    plan = ProductionRunOrchestrator(request, taxon_scope=scope, storage=storage).plan()
    canonical, detections, scores = _join_stage_input_frames()
    source_uri = plan.artifact_uris.staging_uri + "/evidence/stage=poll_flickr/run_id=species_danaus_plexippus/worker=poller/batch=001.parquet"
    detection_uri = plan.artifact_uris.staging_uri + "/evidence/stage=detect_objects/run_id=species_danaus_plexippus/worker=detector/batch=001.parquet"
    score_uri = plan.artifact_uris.staging_uri + "/evidence/stage=score_bioclip/run_id=species_danaus_plexippus/worker=bioclip/batch=001.parquet"
    storage.parquet_payloads[source_uri] = canonical
    storage.parquet_payloads[detection_uri] = detections
    storage.parquet_payloads[score_uri] = scores
    for stage, uri, frame, worker_id in (
        (RunStage.POLL_FLICKR.value, source_uri, canonical, "poller"),
        (RunStage.DETECT_OBJECTS.value, detection_uri, detections, "detector"),
        (RunStage.SCORE_BIOCLIP.value, score_uri, scores, "bioclip"),
    ):
        workstore.register_shard(
            job_name="biominer_production_run",
            registry_version=scope.registry_version,
            stage=stage,
            run_id=plan.manifest.run_id,
            worker_id=worker_id,
            uri=uri,
            checksum=None,
            row_count=frame.height,
        )

    result = ProductionRunOrchestrator(request, taxon_scope=scope, storage=storage, workstore=workstore).run()

    assert result.manifest.status == "complete"
    assert result.manifest.evidence_counts == {"object_evidence_rows": 1, "photo_summary_rows": 0, "review_queue_rows": 0}
    object_evidence_uri = result.manifest.stages[0].outputs["object_evidence"]
    assert object_evidence_uri.startswith(plan.artifact_uris.staging_uri + "/evidence/stage=join_evidence/")
    assert plan.artifact_uris.object_evidence_uri not in storage.parquet_payloads
    assert plan.artifact_uris.photo_summary_uri not in storage.parquet_payloads
    assert storage.json_payloads[plan.artifact_uris.manifest_uri]["evidence_counts"]["object_evidence_rows"] == 1
    joined = storage.parquet_payloads[object_evidence_uri]
    assert joined.select("flickr_photo_id").to_series().to_list() == ["photo-1"]
    join_shards = workstore.list_committed_shards(
        job_name="biominer_production_run",
        stage=RunStage.JOIN_EVIDENCE.value,
        registry_version=scope.registry_version,
        run_id=plan.manifest.run_id,
    )
    assert [shard["uri"] for shard in join_shards] == [object_evidence_uri]


def test_orchestrator_cloud_join_requires_storage_backend() -> None:
    scope = TaxonScope.from_species_context(_species_context())
    request = ProductionRunRequest(
        taxon="Danaus plexippus",
        rank="species",
        output_root="s3://biominer/runs",
        stages=(RunStage.JOIN_EVIDENCE,),
    )

    result = ProductionRunOrchestrator(request, taxon_scope=scope).run()

    assert result.manifest.status == "failed"
    assert result.manifest.stages[0].message == "storage_backend_required_for_join_evidence"


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


def test_orchestrator_summarize_writes_review_queue_for_ambiguous_photos(tmp_path) -> None:
    scope = TaxonScope.from_species_context(_species_context())
    request = ProductionRunRequest(
        taxon="Danaus plexippus",
        rank="species",
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        stages=(RunStage.SUMMARIZE,),
    )
    plan = ProductionRunOrchestrator(request, taxon_scope=scope).plan()
    plan.paths.ensure_directories()
    pl.DataFrame([{"occurrence_bin": "in_review"}]).write_parquet(plan.paths.object_evidence_path)
    pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "review-1",
                "best_detection_id": "det-r",
                "detection_count": 1,
                "best_object_occurrence_bin": "in_review",
                "best_object_species_top1": "Danaus plexippus",
                "best_object_score": 0.42,
                "photo_occurrence_bin": "in_review",
                "photo_bin_reason": "ambiguous_species_margin",
                "all_detection_ids": ["det-r"],
                "all_candidate_species": ["Danaus plexippus", "Danaus eresimus"],
            },
            {
                "source": "flickr",
                "flickr_photo_id": "bronze-1",
                "best_detection_id": "det-b",
                "detection_count": 1,
                "best_object_occurrence_bin": "bronze",
                "best_object_species_top1": "Danaus plexippus",
                "best_object_score": 0.25,
                "photo_occurrence_bin": "bronze",
                "photo_bin_reason": "weak_species_score",
                "all_detection_ids": ["det-b"],
                "all_candidate_species": ["Danaus plexippus"],
            },
        ]
    ).write_parquet(plan.paths.photo_summary_path)

    result = ProductionRunOrchestrator(request, taxon_scope=scope).run()
    queue = pl.read_parquet(result.paths.review_queue_path)

    assert result.manifest.status == "complete"
    assert result.manifest.evidence_counts["review_queue_rows"] == 2
    assert result.manifest.metrics["review_queue_bin_counts"] == {"bronze": 1, "in_review": 1}
    assert result.manifest.stages[0].outputs["review_queue"] == str(result.paths.review_queue_path)
    assert queue.select("flickr_photo_id").to_series().to_list() == ["review-1", "bronze-1"]


def test_orchestrator_summarizes_photo_evidence_from_cloud_join_shards(tmp_path) -> None:
    scope = TaxonScope.from_species_context(_species_context())
    storage = _FakeRunStorage()
    workstore = SQLiteWorkStore(tmp_path / "workstore.sqlite")
    request = ProductionRunRequest(
        taxon="Danaus plexippus",
        rank="species",
        output_root="s3://biominer/runs",
        stages=(RunStage.SUMMARIZE,),
    )
    plan = ProductionRunOrchestrator(request, taxon_scope=scope, storage=storage).plan()
    canonical, detections, scores = _join_stage_input_frames()
    joined, _summary = build_object_evidence_frames(
        canonical_source_records=canonical,
        object_detections=detections,
        object_scores=scores,
    )
    joined_uri = plan.artifact_uris.staging_uri + "/evidence/stage=join_evidence/run_id=species_danaus_plexippus/worker=joiner/batch=001.parquet"
    storage.parquet_payloads[joined_uri] = joined
    workstore.register_shard(
        job_name="biominer_production_run",
        registry_version=scope.registry_version,
        stage=RunStage.JOIN_EVIDENCE.value,
        run_id=plan.manifest.run_id,
        worker_id="joiner",
        uri=joined_uri,
        checksum=None,
        row_count=joined.height,
    )

    result = ProductionRunOrchestrator(request, taxon_scope=scope, storage=storage, workstore=workstore).run()

    assert result.manifest.status == "complete"
    assert result.manifest.evidence_counts == {"object_evidence_rows": 1, "photo_summary_rows": 1, "review_queue_rows": 0}
    photo_summary_uri = result.manifest.stages[0].outputs["photo_summary"]
    assert result.manifest.stages[0].outputs["metrics"] == plan.artifact_uris.metrics_uri
    assert photo_summary_uri.startswith(plan.artifact_uris.staging_uri + "/evidence/stage=photo_summary/")
    assert plan.artifact_uris.photo_summary_uri not in storage.parquet_payloads
    assert storage.json_payloads[plan.artifact_uris.manifest_uri]["evidence_counts"]["photo_summary_rows"] == 1
    assert storage.json_payloads[plan.artifact_uris.metrics_uri]["photo_occurrence_bin_counts"] == {"gold": 1}
    summary = storage.parquet_payloads[photo_summary_uri]
    assert summary.select("flickr_photo_id").to_series().to_list() == ["photo-1"]
    assert summary.select("photo_occurrence_bin").to_series().to_list() == ["gold"]
    summary_shards = workstore.list_committed_shards(
        job_name="biominer_production_run",
        stage="photo_summary",
        registry_version=scope.registry_version,
        run_id=plan.manifest.run_id,
    )
    assert [shard["uri"] for shard in summary_shards] == [photo_summary_uri]


def test_orchestrator_builds_review_queue_from_cloud_summary_shards(tmp_path) -> None:
    scope = TaxonScope.from_species_context(_species_context())
    storage = _FakeRunStorage()
    workstore = SQLiteWorkStore(tmp_path / "workstore.sqlite")
    request = ProductionRunRequest(
        taxon="Danaus plexippus",
        rank="species",
        output_root="s3://biominer/runs",
        stages=(RunStage.SUMMARIZE,),
    )
    plan = ProductionRunOrchestrator(request, taxon_scope=scope, storage=storage).plan()
    summary_uri = plan.artifact_uris.staging_uri + "/evidence/stage=photo_summary/run_id=species_danaus_plexippus/worker=summarizer/batch=001.parquet"
    storage.parquet_payloads[summary_uri] = pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "review-1",
                "best_detection_id": "det-r",
                "detection_count": 1,
                "best_object_occurrence_bin": "in_review",
                "best_object_species_top1": "Danaus plexippus",
                "best_object_score": 0.42,
                "photo_occurrence_bin": "in_review",
                "photo_bin_reason": "ambiguous_species_margin",
                "all_detection_ids": ["det-r"],
                "all_candidate_species": ["Danaus plexippus", "Danaus eresimus"],
            },
            {
                "source": "flickr",
                "flickr_photo_id": "gold-1",
                "best_detection_id": "det-g",
                "detection_count": 1,
                "best_object_occurrence_bin": "gold",
                "best_object_species_top1": "Danaus plexippus",
                "best_object_score": 0.82,
                "photo_occurrence_bin": "gold",
                "photo_bin_reason": "target_species_score_ge_070",
                "all_detection_ids": ["det-g"],
                "all_candidate_species": ["Danaus plexippus"],
            },
        ]
    )
    workstore.register_shard(
        job_name="biominer_production_run",
        registry_version=scope.registry_version,
        stage="photo_summary",
        run_id=plan.manifest.run_id,
        worker_id="summarizer",
        uri=summary_uri,
        checksum=None,
        row_count=2,
    )

    result = ProductionRunOrchestrator(request, taxon_scope=scope, storage=storage, workstore=workstore).run()

    assert result.manifest.status == "complete"
    assert result.manifest.evidence_counts == {"object_evidence_rows": 0, "photo_summary_rows": 2, "review_queue_rows": 1}
    review_queue_uri = result.manifest.stages[0].outputs["review_queue"]
    assert review_queue_uri.startswith(plan.artifact_uris.staging_uri + "/evidence/stage=review_queue/")
    assert plan.artifact_uris.review_queue_uri not in storage.parquet_payloads
    assert storage.json_payloads[plan.artifact_uris.metrics_uri]["review_queue_bin_counts"] == {"in_review": 1}
    queue = storage.parquet_payloads[review_queue_uri]
    assert queue.select("flickr_photo_id").to_series().to_list() == ["review-1"]
    queue_shards = workstore.list_committed_shards(
        job_name="biominer_production_run",
        stage="review_queue",
        registry_version=scope.registry_version,
        run_id=plan.manifest.run_id,
    )
    assert [shard["uri"] for shard in queue_shards] == [review_queue_uri]


def test_orchestrator_local_comment_review_stages_process_and_apply_promotions(tmp_path) -> None:
    scope = TaxonScope.from_species_context(_species_context())
    request = ProductionRunRequest(
        taxon="Danaus plexippus",
        rank="species",
        output_root=tmp_path / "runs",
        storage_backend="local",
        workstore_backend="sqlite",
        stages=(RunStage.QUEUE_COMMENT_REVIEW, RunStage.REVIEW_COMMENTS, RunStage.APPLY_COMMENT_REVIEW),
    )
    orchestrator = ProductionRunOrchestrator(
        request,
        taxon_scope=scope,
        comment_fetcher=lambda photo_id: [{"author": "u1", "_content": "Confirmed Danaus plexippus at -27.4698, 153.0251"}],
    )
    plan = orchestrator.plan()
    plan.paths.ensure_directories()
    pl.DataFrame(
        [
            {
                "source": "flickr",
                "source_record_id": "bronze-1",
                "source_record_hash": "sha256:bronze-1",
                "flickr_photo_id": "bronze-1",
                "photo_page_url": "https://www.flickr.com/photos/example/bronze-1",
                "image_url": "https://live.staticflickr.com/bronze-1.jpg",
                "raw_title": "Danaus plexippus",
                "raw_tags": "Danaus plexippus monarch",
                "bioclip_top1_label": "a photo of Danaus plexippus",
                "species_top1_score": 0.92,
                "bioclip_top1_score": 0.92,
                "is_target_positive": True,
                "occurrence_bin": "bronze",
                "triage_bin": "bronze",
                "image_category": "adult_butterfly",
                "life_stage": "adult_butterfly",
                "date_taken": "2024-01-15",
                "latitude": None,
                "longitude": None,
            }
        ]
    ).write_parquet(plan.paths.object_evidence_path)

    result = orchestrator.run()

    reviewed = pl.read_parquet(result.paths.reviewed_object_evidence_path)
    assert result.manifest.status == "complete"
    assert result.manifest.metrics["comment_review_queue_created"] == 1
    assert result.manifest.metrics["records_moved_to_gold"] == 1
    assert reviewed.select("occurrence_bin").to_series().to_list() == ["gold"]


def test_orchestrator_cloud_summarize_requires_storage_backend() -> None:
    scope = TaxonScope.from_species_context(_species_context())
    request = ProductionRunRequest(
        taxon="Danaus plexippus",
        rank="species",
        output_root="s3://biominer/runs",
        stages=(RunStage.SUMMARIZE,),
    )

    result = ProductionRunOrchestrator(request, taxon_scope=scope).run()

    assert result.manifest.status == "failed"
    assert result.manifest.stages[0].message == "storage_backend_required_for_summarize"


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
    assert result.manifest.bioclip_counts["whole_images_scored"] == 0
    assert result.manifest.bioclip_counts["detector_crops_scored"] == 1
    assert result.manifest.bioclip_counts["segmentation_crops_scored"] == 0
    assert result.manifest.metrics["visual_modes_requested"] == ["detector_crop"]
    assert result.manifest.metrics["visual_modes_scored"] == ["detector_crop"]
    scores = pl.read_parquet(result.paths.object_scores_path).sort("ablation_mode")
    assert scores.height == 1
    assert scores.select("ablation_mode").to_series().to_list() == ["detector_crop"]
    assert scores.select("species_top1_scientific_name").to_series().to_list() == ["Danaus plexippus"]
    assert scores.select("target_species_score").to_series().to_list() == [0.82]
    assert result.manifest.stages[0].outputs["object_detections"] == str(result.paths.object_detections_path)
    assert result.manifest.stages[1].outputs["object_scores"] == str(result.paths.object_scores_path)


def test_orchestrator_detects_objects_from_cloud_storage(tmp_path) -> None:
    scope = TaxonScope.from_species_context(_species_context())
    storage = _FakeRunStorage()
    workstore = SQLiteWorkStore(tmp_path / "workstore.sqlite")
    request = ProductionRunRequest(
        taxon="Danaus plexippus",
        rank="species",
        output_root="s3://biominer/runs",
        stages=(RunStage.DETECT_OBJECTS,),
    )
    plan = ProductionRunOrchestrator(request, taxon_scope=scope, storage=storage).plan()
    canonical, _, _ = _join_stage_input_frames()
    source_uri = plan.artifact_uris.staging_uri + "/evidence/stage=poll_flickr/run_id=species_danaus_plexippus/worker=poller/batch=001.parquet"
    storage.parquet_payloads[source_uri] = canonical
    workstore.register_shard(
        job_name="biominer_production_run",
        registry_version=scope.registry_version,
        stage=RunStage.POLL_FLICKR.value,
        run_id=plan.manifest.run_id,
        worker_id="poller",
        uri=source_uri,
        checksum=None,
        row_count=canonical.height,
    )
    detector = FakeObjectDetector(
        [[DetectionCandidate(label="butterfly_like", score=0.91, bbox_xyxy=(0.0, 0.0, 4.0, 4.0), objectness_score=0.91)]]
    )

    result = ProductionRunOrchestrator(
        request,
        taxon_scope=scope,
        storage=storage,
        workstore=workstore,
        object_detector=detector,
        image_loader=lambda _record: _tiny_rgb_image(),
    ).run()

    assert result.manifest.status == "complete"
    assert result.manifest.stages[0].metrics["detection_work_items_enqueued"] == 1
    assert result.manifest.stages[0].metrics["workstore_work_items_completed"] == 1
    assert result.manifest.detection_counts == {
        "images_seen": 1,
        "detections": 1,
        "crops_created": 1,
        "images_loaded": 1,
        "image_failures": 0,
    }
    detection_uri = result.manifest.stages[0].outputs["object_detections"]
    assert detection_uri.startswith(plan.artifact_uris.staging_uri + "/evidence/stage=detect_objects/")
    detections = storage.parquet_payloads[detection_uri]
    assert detections.select("detector_label").to_series().to_list() == ["butterfly_like"]
    assert storage.json_payloads[plan.artifact_uris.manifest_uri]["detection_counts"]["detections"] == 1
    work_items = workstore.list_work_items(
        job_name="biominer_production_run",
        stage=RunStage.DETECT_OBJECTS.value,
        registry_version=scope.registry_version,
    )
    assert [item["status"] for item in work_items] == ["completed"]
    shards = workstore.list_committed_shards(
        job_name="biominer_production_run",
        stage=RunStage.DETECT_OBJECTS.value,
        registry_version=scope.registry_version,
        run_id=plan.manifest.run_id,
    )
    assert [shard["uri"] for shard in shards] == [detection_uri]


def test_orchestrator_cloud_detect_requires_storage_backend() -> None:
    scope = TaxonScope.from_species_context(_species_context())
    request = ProductionRunRequest(
        taxon="Danaus plexippus",
        rank="species",
        output_root="s3://biominer/runs",
        stages=(RunStage.DETECT_OBJECTS,),
    )

    result = ProductionRunOrchestrator(
        request,
        taxon_scope=scope,
        object_detector=FakeObjectDetector(),
        image_loader=lambda _record: _tiny_rgb_image(),
    ).run()

    assert result.manifest.status == "failed"
    assert result.manifest.stages[0].message == "storage_backend_required_for_detect_objects"


def test_orchestrator_scores_bioclip_from_cloud_storage(tmp_path) -> None:
    scope = TaxonScope.from_species_context(_species_context())
    storage = _FakeRunStorage()
    workstore = SQLiteWorkStore(tmp_path / "workstore.sqlite")
    request = ProductionRunRequest(
        taxon="Danaus plexippus",
        rank="species",
        output_root="s3://biominer/runs",
        stages=(RunStage.SCORE_BIOCLIP,),
    )
    plan = ProductionRunOrchestrator(request, taxon_scope=scope, storage=storage).plan()
    canonical, detections, _ = _join_stage_input_frames()
    source_uri = plan.artifact_uris.staging_uri + "/evidence/stage=poll_flickr/run_id=species_danaus_plexippus/worker=poller/batch=001.parquet"
    detection_uri = plan.artifact_uris.staging_uri + "/evidence/stage=detect_objects/run_id=species_danaus_plexippus/worker=detector/batch=001.parquet"
    storage.parquet_payloads[source_uri] = canonical
    storage.parquet_payloads[detection_uri] = detections
    workstore.register_shard(
        job_name="biominer_production_run",
        registry_version=scope.registry_version,
        stage=RunStage.POLL_FLICKR.value,
        run_id=plan.manifest.run_id,
        worker_id="poller",
        uri=source_uri,
        checksum=None,
        row_count=canonical.height,
    )
    workstore.register_shard(
        job_name="biominer_production_run",
        registry_version=scope.registry_version,
        stage=RunStage.DETECT_OBJECTS.value,
        run_id=plan.manifest.run_id,
        worker_id="detector",
        uri=detection_uri,
        checksum=None,
        row_count=detections.height,
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
        storage=storage,
        workstore=workstore,
        object_scorer=scorer,
        allow_single_target_fixture=True,
    ).run()

    assert result.manifest.status == "complete"
    assert result.manifest.stages[0].metrics["score_work_items_enqueued"] == 1
    assert result.manifest.stages[0].metrics["workstore_work_items_completed"] == 1
    assert result.manifest.bioclip_counts["objects_scored"] == 1
    assert result.manifest.bioclip_counts["whole_images_scored"] == 0
    assert result.manifest.bioclip_counts["detector_crops_scored"] == 1
    assert result.manifest.bioclip_counts["segmentation_crops_scored"] == 0
    assert result.manifest.metrics["visual_modes_requested"] == ["detector_crop"]
    assert result.manifest.metrics["visual_modes_scored"] == ["detector_crop"]
    score_uri = result.manifest.stages[0].outputs["object_scores"]
    assert score_uri.startswith(plan.artifact_uris.staging_uri + "/evidence/stage=score_bioclip/")
    scores = storage.parquet_payloads[score_uri].sort("ablation_mode")
    assert scores.height == 1
    assert scores.select("ablation_mode").to_series().to_list() == ["detector_crop"]
    assert scores.select("species_top1_scientific_name").to_series().to_list() == ["Danaus plexippus"]
    assert storage.json_payloads[plan.artifact_uris.manifest_uri]["bioclip_counts"]["objects_scored"] == 1
    work_items = workstore.list_work_items(
        job_name="biominer_production_run",
        stage=RunStage.SCORE_BIOCLIP.value,
        registry_version=scope.registry_version,
    )
    assert [item["status"] for item in work_items] == ["completed"]
    shards = workstore.list_committed_shards(
        job_name="biominer_production_run",
        stage=RunStage.SCORE_BIOCLIP.value,
        registry_version=scope.registry_version,
        run_id=plan.manifest.run_id,
    )
    assert [shard["uri"] for shard in shards] == [score_uri]


def test_production_cloud_run_does_not_write_durable_local_artifacts(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    scope = TaxonScope.from_species_context(_species_context())
    storage = _FakeRunStorage()
    workstore = SQLiteWorkStore(tmp_path.parent / f"{tmp_path.name}-workstore.sqlite")
    request = ProductionRunRequest(
        taxon="Danaus plexippus",
        rank="species",
        output_root="s3://biominer/runs",
        stages=(RunStage.DETECT_OBJECTS, RunStage.SCORE_BIOCLIP, RunStage.JOIN_EVIDENCE, RunStage.SUMMARIZE),
    )
    plan = ProductionRunOrchestrator(request, taxon_scope=scope, storage=storage).plan()
    canonical, _, _ = _join_stage_input_frames()
    source_uri = plan.artifact_uris.staging_uri + "/evidence/stage=poll_flickr/run_id=species_danaus_plexippus/worker=poller/batch=001.parquet"
    storage.parquet_payloads[source_uri] = canonical
    workstore.register_shard(
        job_name="biominer_production_run",
        registry_version=scope.registry_version,
        stage=RunStage.POLL_FLICKR.value,
        run_id=plan.manifest.run_id,
        worker_id="poller",
        uri=source_uri,
        checksum=None,
        row_count=canonical.height,
    )
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

    before = _durable_local_artifacts(tmp_path)
    _install_forbidden_local_write_guard(monkeypatch, root=tmp_path)
    result = ProductionRunOrchestrator(
        request,
        taxon_scope=scope,
        storage=storage,
        workstore=workstore,
        object_detector=detector,
        image_loader=lambda _record: _tiny_rgb_image(),
        object_scorer=scorer,
        allow_single_target_fixture=True,
    ).run()
    after = _durable_local_artifacts(tmp_path)

    assert result.manifest.status == "complete"
    assert after == before == set()
    assert result.manifest.stages[0].outputs["object_detections"] in storage.parquet_payloads
    assert result.manifest.stages[1].outputs["object_scores"] in storage.parquet_payloads
    assert result.manifest.stages[2].outputs["object_evidence"] in storage.parquet_payloads
    assert result.manifest.stages[3].outputs["photo_summary"] in storage.parquet_payloads
    assert result.manifest.stages[3].outputs["review_queue"] in storage.parquet_payloads
    assert plan.artifact_uris.metrics_uri in storage.json_payloads
    assert plan.artifact_uris.manifest_uri in storage.json_payloads


def test_orchestrator_cloud_score_requires_storage_backend() -> None:
    scope = TaxonScope.from_species_context(_species_context())
    request = ProductionRunRequest(
        taxon="Danaus plexippus",
        rank="species",
        output_root="s3://biominer/runs",
        stages=(RunStage.SCORE_BIOCLIP,),
    )

    result = ProductionRunOrchestrator(request, taxon_scope=scope, object_scorer=_ConstantObjectScorer({})).run()

    assert result.manifest.status == "failed"
    assert result.manifest.stages[0].message == "storage_backend_required_for_score_bioclip"


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
    assert paths.review_queue_path.name == "review_queue.parquet"
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
    assert callable(build_object_evidence_frames)
    assert callable(build_review_queue)


def test_review_queue_keeps_bronze_and_review_photo_summaries() -> None:
    summary = pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "gold-1",
                "best_detection_id": "det-g",
                "detection_count": 1,
                "best_object_occurrence_bin": "gold",
                "best_object_species_top1": "Danaus plexippus",
                "best_object_score": 0.9,
                "photo_occurrence_bin": "gold",
                "photo_bin_reason": "target_species_score_ge_070",
                "all_detection_ids": ["det-g"],
                "all_candidate_species": ["Danaus plexippus"],
            },
            {
                "source": "flickr",
                "flickr_photo_id": "review-1",
                "best_detection_id": "det-r",
                "detection_count": 1,
                "best_object_occurrence_bin": "in_review",
                "best_object_species_top1": "Danaus plexippus",
                "best_object_score": 0.42,
                "photo_occurrence_bin": "in_review",
                "photo_bin_reason": "ambiguous_species_margin",
                "all_detection_ids": ["det-r"],
                "all_candidate_species": ["Danaus plexippus", "Danaus eresimus"],
            },
            {
                "source": "flickr",
                "flickr_photo_id": "bronze-1",
                "best_detection_id": "det-b",
                "detection_count": 1,
                "best_object_occurrence_bin": "bronze",
                "best_object_species_top1": "Danaus plexippus",
                "best_object_score": 0.25,
                "photo_occurrence_bin": "bronze",
                "photo_bin_reason": "weak_species_score",
                "all_detection_ids": ["det-b"],
                "all_candidate_species": ["Danaus plexippus"],
            },
        ]
    )

    queue = build_review_queue(summary)

    assert queue.select("flickr_photo_id").to_series().to_list() == ["review-1", "bronze-1"]
    assert queue.select("review_priority").to_series().to_list() == [10, 20]
    assert queue.select("review_reason").to_series().to_list() == ["ambiguous_species_margin", "weak_species_score"]


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
    assert should_enable_name_by_default(TrustTier.T5, "high", "none") is True
    assert should_enable_name_by_default(TrustTier.T5, "high", "none", review_state="accepted") is True


def test_trust_policy_disabled_reasons() -> None:
    assert disabled_reason_for_candidate(TrustTier.T3, "high", "none") == "wikidata_name_requires_confident_taxon_link"
    assert disabled_reason_for_candidate(TrustTier.T4, "medium", "ambiguous") == "name_collision_requires_review"
    assert disabled_reason_for_candidate(TrustTier.T5, "high", "none") == ""
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


def _write_t5_query_definitions(registry: Path) -> None:
    pl.DataFrame(
        [
            _query_definition_row(
                "q-t5-tags",
                "Translated Lime",
                "tags",
                5,
                name_class="generated_translation",
                confidence="low",
                trust_tier="T5",
                query_eligible=True,
            ),
            _query_definition_row(
                "q-t5-text",
                "Translated Lime",
                "text",
                5,
                name_class="generated_translation",
                confidence="low",
                trust_tier="T5",
                query_eligible=True,
            ),
        ]
    ).write_parquet(registry / "flickr_query_definitions.parquet")


def _write_query_definitions_with_out_of_scope_taxa(registry: Path) -> None:
    pl.DataFrame(
        [
            _query_definition_row("q-demoleus-tags", "Papilio demoleus", "tags", 10),
            _query_definition_row("q-demoleus-text", "Lime butterfly", "text", 20),
            _query_definition_row(
                "q-machaon-tags",
                "Papilio machaon",
                "tags",
                10,
                accepted_taxon_key="gbif:101",
                accepted_scientific_name="Papilio machaon",
                species_key="gbif:101",
                species="Papilio machaon",
            ),
            _query_definition_row(
                "q-danaus-tags",
                "Danaus plexippus",
                "tags",
                10,
                accepted_taxon_key="gbif:200",
                accepted_scientific_name="Danaus plexippus",
                family_key="gbif:20",
                family="Nymphalidae",
                genus_key="gbif:190",
                genus="Danaus",
                species_key="gbif:200",
                species="Danaus plexippus",
            ),
        ]
    ).write_parquet(registry / "flickr_query_definitions.parquet")


def _write_source_records(paths: RunPaths) -> None:
    paths.ensure_directories()
    canonical, _, _ = _join_stage_input_frames()
    canonical.write_parquet(paths.source_records_path)


def _write_join_stage_inputs(paths: RunPaths) -> None:
    paths.ensure_directories()
    canonical, detections, scores = _join_stage_input_frames()
    canonical.write_parquet(paths.source_records_path)
    detections.write_parquet(paths.object_detections_path)
    scores.write_parquet(paths.object_scores_path)


def _join_stage_input_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    canonical = pl.DataFrame(
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
    )
    detections = pl.DataFrame(
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
    )
    scores = pl.DataFrame(
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
    )
    return canonical, detections, scores


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


def _fake_flickr_fetch_two_pages(_query: object) -> dict[str, object]:
    return {
        "photos": {
            "total": "501",
            "pages": "2",
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


class _FakeRunStorage:
    def __init__(self) -> None:
        self.parquet_payloads: dict[str, pl.DataFrame] = {}
        self.json_payloads: dict[str, dict[str, object]] = {}

    def read_parquet(self, uri: str) -> pl.DataFrame:
        return self.parquet_payloads[uri]

    def read_json(self, uri: str) -> dict[str, object]:
        return self.json_payloads[uri]

    def write_parquet_shard(self, uri: str, frame: pl.DataFrame) -> str:
        self.parquet_payloads[uri] = frame
        return uri

    def write_json(self, uri: str, payload: dict[str, object]) -> str:
        self.json_payloads[uri] = payload
        return uri

    def exists(self, uri: str) -> bool:
        return uri in self.parquet_payloads or uri in self.json_payloads


def _install_forbidden_local_write_guard(monkeypatch, *, root: Path) -> None:  # noqa: ANN001 - pytest monkeypatch fixture.
    for method_name in ("write_text", "write_bytes", "mkdir", "touch"):
        original = getattr(Path, method_name)

        def guarded_path_method(self, *args, _original=original, _method_name=method_name, **kwargs):  # noqa: ANN001, ANN202
            if _is_forbidden_local_artifact(self, root=root):
                raise AssertionError(f"production cloud mode attempted local {_method_name}: {self}")
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(Path, method_name, guarded_path_method)

    original_write_parquet = pl.DataFrame.write_parquet

    def guarded_write_parquet(self, file=None, *args, **kwargs):  # noqa: ANN001, ANN202
        if isinstance(file, str | Path) and _is_forbidden_local_artifact(Path(file), root=root):
            raise AssertionError(f"production cloud mode attempted local parquet write: {file}")
        return original_write_parquet(self, file, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", guarded_write_parquet)


def _durable_local_artifacts(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if _is_forbidden_local_artifact(path, root=root)
    }


def _is_forbidden_local_artifact(path: str | Path, *, root: Path) -> bool:
    candidate = Path(path)
    absolute = candidate if candidate.is_absolute() else root / candidate
    try:
        relative = absolute.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    if relative.suffix == ".sqlite":
        return True
    return bool(relative.parts and relative.parts[0] in {"data", "staging", "reports", "runs"})


def _seed_cloud_registry(storage: _FakeRunStorage, registry_uri: str, registry: Path) -> None:
    storage.parquet_payloads[f"{registry_uri}/taxa.parquet"] = pl.read_parquet(registry / "taxa.parquet")
    storage.parquet_payloads[f"{registry_uri}/names.parquet"] = pl.read_parquet(registry / "names.parquet")
    storage.parquet_payloads[f"{registry_uri}/source_snapshots.parquet"] = pl.read_parquet(registry / "source_snapshots.parquet")
    query_definitions = registry / "flickr_query_definitions.parquet"
    if query_definitions.exists():
        storage.parquet_payloads[f"{registry_uri}/flickr_query_definitions.parquet"] = pl.read_parquet(query_definitions)
    storage.json_payloads[f"{registry_uri}/manifest.json"] = json.loads((registry / "manifest.json").read_text(encoding="utf-8"))


def _query_definition_row(
    query_definition_id: str,
    source_term: str,
    search_field: str,
    search_priority: int,
    *,
    accepted_taxon_key: str = "gbif:100",
    accepted_scientific_name: str = "Papilio demoleus",
    family_key: str = "gbif:10",
    family: str = "Papilionidae",
    genus_key: str = "gbif:90",
    genus: str = "Papilio",
    species_key: str = "gbif:100",
    species: str = "Papilio demoleus",
    name_class: str | None = None,
    confidence: str = "high",
    trust_tier: str = "T1",
    query_eligible: bool | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "query_definition_id": query_definition_id,
        "registry_version": "rank-registry-v1",
        "accepted_taxon_key": accepted_taxon_key,
        "accepted_scientific_name": accepted_scientific_name,
        "family_key": family_key,
        "family": family,
        "genus_key": genus_key,
        "genus": genus,
        "species_key": species_key,
        "species": species,
        "source_term": source_term,
        "language": "en",
        "search_field": search_field,
        "search_priority": search_priority,
        "bbox": "",
        "region": "",
        "name_class": name_class or ("accepted_scientific" if search_field == "tags" else "vernacular"),
        "confidence": confidence,
        "trust_tier": trust_tier,
        "enabled": True,
    }
    if query_eligible is not None:
        row["query_eligible"] = query_eligible
    return row


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
