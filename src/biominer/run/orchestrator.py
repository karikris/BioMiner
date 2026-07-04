from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from biominer.bioclip.object_runner import OBJECT_VISUAL_MODES, PRIMARY_VISUAL_CLASSIFIER
from biominer.common.status import CLAIMED, COMPLETED, FAILED, PENDING
from biominer.evidence.join import build_object_evidence_frames, write_object_evidence_outputs
from biominer.evidence.metrics import build_review_queue, evidence_count_metrics
from biominer.storage.parquet import write_parquet
from biominer.flickr_fetch.metadata_poller import DEFAULT_STALE_CLAIM_SECONDS, SOFT_API_CALLS_PER_HOUR, MetadataPollState, poll_once
from biominer.flickr_fetch.query_planner import FlickrQuery, load_registry_flickr_queries, load_registry_flickr_queries_from_frame, query_hash
from biominer.run.manifest import RunManifest, utc_now_iso
from biominer.run.paths import RunArtifactUris, RunPaths
from biominer.run.stages import DEFAULT_PRODUCTION_STAGES, RunStage, StageStatus, default_stage_records
from biominer.run.taxon_scope import InputRank, TaxonScope, resolve_taxon_scope_from_registry, resolve_taxon_scope_from_registry_frames
from biominer.storage.cloud import CloudStorage
from biominer.storage.paths import safe_path_component
from biominer.storage.uri import is_cloud_uri, join_uri
from biominer.workstore.base import WorkStore


DEFAULT_BIOCLIP_MODEL = "imageomics/bioclip-2.5-vith14"
DEFAULT_VISION_BACKEND = "yoloe26"
PRODUCTION_JOB_NAME = "biominer_production_run"


@dataclass(frozen=True)
class StageExecutionResult:
    status: StageStatus = StageStatus.COMPLETE
    message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)


StageHandler = Callable[[Any], StageExecutionResult]


@dataclass(frozen=True)
class ProductionRunRequest:
    taxon: str
    rank: InputRank = "auto"
    registry_dir: str | None = None
    output_root: str | Path = "runs"
    run_id: str | None = None
    storage_backend: str = "s3"
    workstore_backend: str = "postgres"
    vision_backend: str = DEFAULT_VISION_BACKEND
    bioclip_model: str = DEFAULT_BIOCLIP_MODEL
    bioclip_ablation_mode: str = "detector_crop"
    stages: tuple[RunStage, ...] = DEFAULT_PRODUCTION_STAGES
    dry_run: bool = False
    limits: dict[str, int] = field(default_factory=dict)

    def resolved_run_id(self) -> str:
        if self.run_id:
            return safe_path_component(self.run_id)
        return safe_path_component(f"{self.rank}_{self.taxon}")


@dataclass(frozen=True)
class ProductionRunPlan:
    request: ProductionRunRequest
    paths: RunPaths
    artifact_uris: RunArtifactUris
    manifest: RunManifest

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": {
                "taxon": self.request.taxon,
                "rank": self.request.rank,
                "registry_dir": self.request.registry_dir,
                "output_root": str(self.request.output_root),
                "run_id": self.request.resolved_run_id(),
                "storage_backend": self.request.storage_backend,
                "workstore_backend": self.request.workstore_backend,
                "vision_backend": self.request.vision_backend,
                "bioclip_model": self.request.bioclip_model,
                "bioclip_ablation_mode": self.request.bioclip_ablation_mode,
                "stages": [stage.value for stage in self.request.stages],
                "dry_run": self.request.dry_run,
                "limits": dict(self.request.limits),
            },
            "paths": {
                "run_root": str(self.paths.run_root),
                "manifest": str(self.paths.manifest_path),
                "query_definitions": str(self.paths.query_definitions_path),
                "object_detections": str(self.paths.object_detections_path),
                "object_scores": str(self.paths.object_scores_path),
                "object_evidence": str(self.paths.object_evidence_path),
                "photo_summary": str(self.paths.photo_summary_path),
            },
            "artifact_uris": self.artifact_uris.to_dict(),
            "species_artifacts": {
                context.scientific_name: {
                    "root": self.artifact_uris.species_uri(context.scientific_name),
                    "context": self.artifact_uris.species_context_uri(context.scientific_name),
                    "query_definitions": self.artifact_uris.species_query_definitions_uri(context.scientific_name),
                }
                for context in self.manifest.taxon_scope.species_contexts
            },
            "manifest": self.manifest.to_dict(),
        }


def build_run_plan(request: ProductionRunRequest, *, taxon_scope: TaxonScope) -> ProductionRunPlan:
    run_id = request.resolved_run_id()
    paths = RunPaths.from_root(request.output_root, run_id=run_id)
    artifact_uris = RunArtifactUris.from_prefix(request.output_root, run_id=run_id)
    manifest = RunManifest(
        run_id=run_id,
        taxon_scope=taxon_scope,
        status="planned",
        storage_backend=request.storage_backend,
        workstore_backend=request.workstore_backend,
        started_at=utc_now_iso(),
        stages=default_stage_records(request.stages),
        model_configs={
            "vision_backend": request.vision_backend,
            "bioclip_model": request.bioclip_model,
            "bioclip_ablation_mode": request.bioclip_ablation_mode,
            "primary_visual_classifier": PRIMARY_VISUAL_CLASSIFIER,
            "visual_modes": list(OBJECT_VISUAL_MODES),
        },
        query_counts={"compiled_definitions": 0, "enqueued_work_items": 0},
        detection_counts={"images_seen": 0, "detections": 0, "crops_created": 0},
        bioclip_counts={"objects_scored": 0, "whole_images_scored": 0, "segmentation_crops_scored": 0},
        evidence_counts={"object_evidence_rows": 0, "photo_summary_rows": 0, "review_queue_rows": 0},
        metrics={"expanded_species_count": taxon_scope.species_count},
        outputs=artifact_uris.to_dict(),
    )
    return ProductionRunPlan(request=request, paths=paths, artifact_uris=artifact_uris, manifest=manifest)


class ProductionRunOrchestrator:
    """Rank-aware production run coordinator.

    Cloud-backed stages still fail clearly where S3/Postgres IO is not fully
    wired, while local/dev runs can execute the implemented stages with fakes.
    """

    def __init__(
        self,
        request: ProductionRunRequest,
        *,
        taxon_scope: TaxonScope | None = None,
        storage: CloudStorage | None = None,
        workstore: WorkStore | None = None,
        object_detector: Any | None = None,
        image_loader: Callable[[dict[str, Any]], Any] | None = None,
        object_scorer: Any | None = None,
        metadata_fetcher: Callable[[FlickrQuery], dict[str, Any]] | None = None,
        flickr_api_key: str | None = None,
        species_candidate_path: str | Path | None = None,
        allow_single_target_fixture: bool = False,
        stage_handlers: Mapping[RunStage, StageHandler] | None = None,
    ) -> None:
        self.request = request
        self.taxon_scope = taxon_scope
        self.storage = storage
        self.workstore = workstore
        self.object_detector = object_detector
        self.image_loader = image_loader
        self.object_scorer = object_scorer
        self.metadata_fetcher = metadata_fetcher
        self.flickr_api_key = flickr_api_key
        self.species_candidate_path = species_candidate_path
        self.allow_single_target_fixture = allow_single_target_fixture
        self.stage_handlers = dict(stage_handlers or {})

    def plan(self) -> ProductionRunPlan:
        return build_run_plan(self.request, taxon_scope=self._resolve_taxon_scope())

    def write_dry_run_manifest(self) -> Path:
        plan = self.plan()
        plan.paths.ensure_directories()
        return plan.manifest.write_json(plan.paths.manifest_path)

    def run(self) -> ProductionRunPlan:
        plan = self.plan()
        manifest = plan.manifest.with_status("running")
        plan = replace(plan, manifest=manifest)
        for stage in self.request.stages:
            started_at = utc_now_iso()
            manifest = plan.manifest.with_stage_status(stage, StageStatus.RUNNING, started_at=started_at)
            plan = replace(plan, manifest=manifest)
            result = self._run_stage(plan, stage)
            manifest = plan.manifest.with_stage_status(
                stage,
                result.status,
                ended_at=utc_now_iso(),
                message=result.message,
                metrics=result.metrics,
                outputs=result.outputs,
            )
            manifest = _merge_stage_counts(manifest, stage=stage, result=result)
            plan = replace(plan, manifest=manifest)
        final_status = "failed" if any(stage.status == StageStatus.FAILED for stage in plan.manifest.stages) else "complete"
        plan = replace(plan, manifest=plan.manifest.with_status(final_status, ended_at=utc_now_iso()))
        self._write_manifest_if_local(plan)
        return plan

    def _resolve_taxon_scope(self) -> TaxonScope:
        if self.taxon_scope is not None:
            return self.taxon_scope
        if not self.request.registry_dir:
            raise ValueError("registry_dir is required when taxon_scope is not provided")
        if is_cloud_uri(str(self.request.registry_dir)):
            self.taxon_scope = resolve_taxon_scope_from_registry_frames(
                taxa=self._read_registry_parquet("taxa.parquet"),
                names=self._read_registry_optional_parquet("names.parquet"),
                source_snapshots=self._read_registry_optional_parquet("source_snapshots.parquet"),
                manifest=self._read_registry_manifest(),
                input_name=self.request.taxon,
                input_rank=self.request.rank,
            )
        else:
            self.taxon_scope = resolve_taxon_scope_from_registry(
                registry_dir=self.request.registry_dir,
                input_name=self.request.taxon,
                input_rank=self.request.rank,
            )
        return self.taxon_scope

    def _run_stage(self, plan: ProductionRunPlan, stage: RunStage) -> StageExecutionResult:
        handler = self.stage_handlers.get(stage)
        if handler is not None:
            return handler(plan)
        if stage == RunStage.RESOLVE_TAXON_SCOPE:
            return StageExecutionResult(
                metrics={
                    "accepted_rank": plan.manifest.taxon_scope.accepted_rank,
                    "expanded_species_count": plan.manifest.taxon_scope.species_count,
                }
            )
        if self.request.dry_run:
            return StageExecutionResult(status=StageStatus.SKIPPED, message="dry_run")
        if stage == RunStage.BUILD_REGISTRY:
            return self._run_build_registry_stage(plan)
        if stage == RunStage.COMPILE_QUERIES:
            return self._run_compile_queries_stage(plan)
        if stage == RunStage.ENQUEUE_FLICKR_WORK:
            return self._run_enqueue_flickr_work_stage(plan)
        if stage == RunStage.POLL_FLICKR:
            return self._run_poll_flickr_stage(plan)
        if stage == RunStage.DETECT_OBJECTS:
            return self._run_detect_objects_stage(plan)
        if stage == RunStage.SCORE_BIOCLIP:
            return self._run_score_bioclip_stage(plan)
        if stage == RunStage.JOIN_EVIDENCE:
            return self._run_join_evidence_stage(plan)
        if stage == RunStage.SUMMARIZE:
            return self._run_summarize_stage(plan)
        return StageExecutionResult(status=StageStatus.SKIPPED, message="stage_not_implemented")

    def _run_build_registry_stage(self, plan: ProductionRunPlan) -> StageExecutionResult:
        if self._registry_is_cloud():
            if self.storage is None:
                return StageExecutionResult(status=StageStatus.FAILED, message="storage_backend_required_for_build_registry")
            required = (
                self._registry_artifact_uri("taxa.parquet"),
                self._registry_artifact_uri("names.parquet"),
                self._registry_artifact_uri("manifest.json"),
            )
            missing = _missing_uris(self.storage, *required)
            if missing:
                return StageExecutionResult(status=StageStatus.FAILED, message="missing_registry_inputs: " + ", ".join(missing))
            taxa = self.storage.read_parquet(self._registry_artifact_uri("taxa.parquet"))
            names = self.storage.read_parquet(self._registry_artifact_uri("names.parquet"))
            query_definitions_uri = self._registry_artifact_uri("flickr_query_definitions.parquet")
            query_definition_rows = self.storage.read_parquet(query_definitions_uri).height if self.storage.exists(query_definitions_uri) else 0
            manifest = self.storage.read_json(self._registry_artifact_uri("manifest.json"))
            outputs = {
                "registry_dir": str(self.request.registry_dir),
                "manifest": self._registry_artifact_uri("manifest.json"),
                "taxa": self._registry_artifact_uri("taxa.parquet"),
                "names": self._registry_artifact_uri("names.parquet"),
            }
            if self.storage.exists(query_definitions_uri):
                outputs["query_definitions"] = query_definitions_uri
            return StageExecutionResult(
                metrics={
                    "registry_reused": True,
                    "taxa_rows": taxa.height,
                    "name_rows": names.height,
                    "query_definition_rows": query_definition_rows,
                    "expanded_species_count": plan.manifest.taxon_scope.species_count,
                    "registry_version": str(manifest.get("registry_version") or ""),
                },
                outputs=outputs,
            )
        registry = self._registry_dir_path(stage_name="build_registry")
        required = (
            registry / "taxa.parquet",
            registry / "names.parquet",
            registry / "manifest.json",
        )
        missing = _missing_paths(*required)
        if missing:
            return StageExecutionResult(status=StageStatus.FAILED, message="missing_registry_inputs: " + ", ".join(missing))
        query_definitions = registry / "flickr_query_definitions.parquet"
        metrics = {
            "registry_reused": True,
            "taxa_rows": _parquet_row_count(registry / "taxa.parquet"),
            "name_rows": _parquet_row_count(registry / "names.parquet"),
            "query_definition_rows": _parquet_row_count(query_definitions) if query_definitions.exists() else 0,
            "expanded_species_count": plan.manifest.taxon_scope.species_count,
            "registry_version": _registry_manifest_version(registry),
        }
        outputs = {
            "registry_dir": str(registry),
            "manifest": str(registry / "manifest.json"),
            "taxa": str(registry / "taxa.parquet"),
            "names": str(registry / "names.parquet"),
        }
        if query_definitions.exists():
            outputs["query_definitions"] = str(query_definitions)
        return StageExecutionResult(metrics=metrics, outputs=outputs)

    def _run_compile_queries_stage(self, plan: ProductionRunPlan) -> StageExecutionResult:
        source_frame, source_ref, source_path = self._registry_query_definitions_source()
        queries = self._load_flickr_work_queries()
        outputs = {
            "source_query_definitions": source_ref,
            "query_definitions": plan.artifact_uris.query_definitions_uri,
        }
        if not is_cloud_uri(self.request.output_root):
            plan.paths.ensure_directories()
            if source_path is not None and source_path.resolve() != plan.paths.query_definitions_path.resolve():
                shutil.copyfile(source_path, plan.paths.query_definitions_path)
            else:
                write_parquet(source_frame, plan.paths.query_definitions_path)
            outputs["local_query_definitions"] = str(plan.paths.query_definitions_path)
        else:
            if self.storage is None:
                return StageExecutionResult(status=StageStatus.FAILED, message="storage_backend_required_for_compile_queries")

            uri = self.storage.write_parquet_shard(plan.artifact_uris.query_definitions_uri, source_frame)
            outputs["query_definitions"] = uri
        return StageExecutionResult(
            metrics={
                "registry_query_definition_rows": source_frame.height,
                "flickr_work_items": len(queries),
            },
            outputs=outputs,
        )

    def _run_enqueue_flickr_work_stage(self, plan: ProductionRunPlan) -> StageExecutionResult:
        if self.workstore is None:
            return StageExecutionResult(status=StageStatus.FAILED, message="workstore_required_for_enqueue_flickr_work")
        queries = self._load_flickr_work_queries()
        registry_version = plan.manifest.taxon_scope.registry_version
        self.workstore.get_or_create_run(
            job_name=PRODUCTION_JOB_NAME,
            stage=RunStage.ENQUEUE_FLICKR_WORK.value,
            run_id=plan.manifest.run_id,
            registry_version=registry_version,
            config=plan.to_dict()["request"],
        )
        inserted = self.workstore.enqueue_work(
            PRODUCTION_JOB_NAME,
            registry_version,
            [_flickr_query_work_item(query, run_id=plan.manifest.run_id) for query in queries],
            stage=RunStage.POLL_FLICKR.value,
        )
        return StageExecutionResult(
            metrics={
                "flickr_work_items": len(queries),
                "enqueued_work_items": inserted,
                "duplicate_work_items": len(queries) - inserted,
            },
            outputs={"workstore_stage": RunStage.POLL_FLICKR.value},
        )

    def _run_poll_flickr_stage(self, plan: ProductionRunPlan) -> StageExecutionResult:
        if is_cloud_uri(self.request.output_root) or self.request.storage_backend != "local" or self.request.workstore_backend != "sqlite":
            return self._run_cloud_poll_flickr_stage(plan)
        queries = self._load_flickr_work_queries()
        plan.paths.ensure_directories()
        state_db = plan.paths.run_root / "state" / "flickr_poller.sqlite"
        state_db.parent.mkdir(parents=True, exist_ok=True)
        state = MetadataPollState(state_db)
        seeded = state.ensure_seed_work_items(queries)
        api_key = self.flickr_api_key or os.environ.get("FLICKR_API_KEY")
        if state.work_item_count() > 0 and self.metadata_fetcher is None and not api_key:
            return StageExecutionResult(status=StageStatus.FAILED, message="flickr_fetcher_or_api_key_required_for_poll_flickr")
        result = poll_once(
            state_db=state.path,
            raw_root=plan.paths.run_root / "raw",
            evidence_output=plan.paths.source_records_path,
            max_api_calls=int(self.request.limits.get("api_calls") or SOFT_API_CALLS_PER_HOUR),
            api_key=api_key,
            fetch_metadata=self.metadata_fetcher,
            workers=int(self.request.limits.get("workers") or 1),
            run_id=plan.manifest.run_id,
            worker_id=os.environ.get("BIOMINER_WORKER_ID") or "local",
            storage_backend="local",
            compact_after_run=True,
        )
        return StageExecutionResult(
            metrics={
                "seeded_work_items": seeded,
                "raw_responses_written": result.raw_responses_written,
                "evidence_rows_written": result.evidence_rows_written,
                "evidence_rows_total": result.evidence_rows_total,
                "source_records_inserted": result.source_records_inserted,
                "duplicate_records_skipped": result.duplicate_records_skipped,
                "query_terms_added": result.query_hits_inserted,
                "duplicate_query_terms": result.duplicate_query_hits_skipped,
                "image_urls_queued": result.image_urls_queued,
                "work_items_claimed": result.work_items_claimed,
                "api_calls_made": result.api_calls_made,
                "remaining_soft_budget": result.remaining_soft_budget,
                "remaining_hard_budget": result.remaining_hard_budget,
                "stale_claims_requeued": result.stale_claims_requeued,
            },
            outputs={
                "state_db": str(state.path),
                "raw_root": str(plan.paths.run_root / "raw"),
                "source_records": str(plan.paths.source_records_path),
            },
        )

    def _run_cloud_poll_flickr_stage(self, plan: ProductionRunPlan) -> StageExecutionResult:
        if self.storage is None:
            return StageExecutionResult(status=StageStatus.FAILED, message="storage_backend_required_for_poll_flickr")
        if self.workstore is None:
            return StageExecutionResult(status=StageStatus.FAILED, message="workstore_required_for_poll_flickr")
        registry_version = plan.manifest.taxon_scope.registry_version
        worker_id = os.environ.get("BIOMINER_WORKER_ID") or "local"
        stale_requeued = self.workstore.requeue_stale_claims(
            job_name=PRODUCTION_JOB_NAME,
            stage=RunStage.POLL_FLICKR.value,
            registry_version=registry_version,
            stale_after_seconds=int(self.request.limits.get("stale_claim_seconds") or DEFAULT_STALE_CLAIM_SECONDS),
        )
        pending_preview = self.workstore.list_work_items(
            job_name=PRODUCTION_JOB_NAME,
            stage=RunStage.POLL_FLICKR.value,
            registry_version=registry_version,
            statuses=[PENDING],
            limit=1,
        )
        if not pending_preview:
            return StageExecutionResult(
                metrics={
                    "seeded_work_items": 0,
                    "work_items_claimed": 0,
                    "api_calls_made": 0,
                    "workstore_stale_claims_requeued": stale_requeued,
                },
                outputs={"workstore_stage": RunStage.POLL_FLICKR.value},
            )
        api_key = self.flickr_api_key or os.environ.get("FLICKR_API_KEY")
        if self.metadata_fetcher is None and not api_key:
            return StageExecutionResult(status=StageStatus.FAILED, message="flickr_fetcher_or_api_key_required_for_poll_flickr")
        claim_limit = _cloud_poll_claim_limit(self.request.limits)
        claimed = self.workstore.claim_next_batch(
            worker_id,
            claim_limit,
            job_name=PRODUCTION_JOB_NAME,
            stage=RunStage.POLL_FLICKR.value,
            registry_version=registry_version,
        )
        if not claimed:
            return StageExecutionResult(
                metrics={
                    "seeded_work_items": 0,
                    "work_items_claimed": 0,
                    "api_calls_made": 0,
                    "workstore_stale_claims_requeued": stale_requeued,
                },
                outputs={"workstore_stage": RunStage.POLL_FLICKR.value},
            )
        with tempfile.TemporaryDirectory(prefix="biominer-poll-") as tmp_dir:
            state = MetadataPollState(Path(tmp_dir) / "flickr_poller.sqlite")
            local_work_keys: dict[str, str] = {}
            for item in claimed:
                query = _flickr_query_from_work_item(item)
                state.enqueue_work_item(query)
                local_work_keys[_poller_work_item_id(query)] = str(item["work_key"])
            result = poll_once(
                state_db=state.path,
                raw_root=join_uri(plan.artifact_uris.staging_uri, "raw"),
                evidence_output=plan.artifact_uris.source_records_uri,
                max_api_calls=int(self.request.limits.get("api_calls") or SOFT_API_CALLS_PER_HOUR),
                api_key=api_key,
                fetch_metadata=self.metadata_fetcher,
                workers=max(1, min(int(self.request.limits.get("workers") or 1), len(claimed))),
                run_id=plan.manifest.run_id,
                worker_id=worker_id,
                storage_backend=self.request.storage_backend,
                storage_prefix=plan.artifact_uris.staging_uri,
                evidence_stage=RunStage.POLL_FLICKR.value,
                compact_after_run=False,
                storage=self.storage,
                work_store=self.workstore,
                claim_once=True,
            )
            source_frame = state.canonical_source_records_frame()
            source_records_uri = self.storage.write_parquet_shard(plan.artifact_uris.source_records_uri, source_frame)
            mirror = _mirror_cloud_poll_state_to_workstore(
                workstore=self.workstore,
                snapshot=state.work_items_snapshot(),
                local_work_keys=local_work_keys,
                run_id=plan.manifest.run_id,
                registry_version=registry_version,
                output_uri=source_records_uri,
            )
        return StageExecutionResult(
            metrics={
                "seeded_work_items": len(claimed),
                "raw_responses_written": result.raw_responses_written,
                "evidence_rows_written": result.evidence_rows_written,
                "evidence_rows_total": result.evidence_rows_total,
                "source_records_inserted": result.source_records_inserted,
                "duplicate_records_skipped": result.duplicate_records_skipped,
                "query_terms_added": result.query_hits_inserted,
                "duplicate_query_terms": result.duplicate_query_hits_skipped,
                "image_urls_queued": result.image_urls_queued,
                "work_items_claimed": result.work_items_claimed,
                "api_calls_made": result.api_calls_made,
                "remaining_soft_budget": result.remaining_soft_budget,
                "remaining_hard_budget": result.remaining_hard_budget,
                "stale_claims_requeued": result.stale_claims_requeued,
                "workstore_stale_claims_requeued": stale_requeued,
                **mirror,
            },
            outputs={
                "workstore_stage": RunStage.POLL_FLICKR.value,
                "raw_prefix": join_uri(plan.artifact_uris.staging_uri, "source=flickr"),
                "evidence_prefix": join_uri(plan.artifact_uris.staging_uri, "evidence"),
                "source_records": source_records_uri,
            },
        )

    def _run_detect_objects_stage(self, plan: ProductionRunPlan) -> StageExecutionResult:
        if is_cloud_uri(self.request.output_root):
            if self.storage is None:
                return StageExecutionResult(status=StageStatus.FAILED, message="storage_backend_required_for_detect_objects")
            missing = _missing_uris(self.storage, plan.artifact_uris.source_records_uri)
            if missing:
                return StageExecutionResult(status=StageStatus.FAILED, message="missing_detection_inputs: " + ", ".join(missing))
            if self.object_detector is None or self.image_loader is None:
                return StageExecutionResult(status=StageStatus.FAILED, message="detector_runtime_required_for_detect_objects")
            from biominer.detection.pipeline import run_detection_pipeline

            records = self.storage.read_parquet(plan.artifact_uris.source_records_uri).to_dicts()
            with tempfile.TemporaryDirectory(prefix="biominer-detect-") as tmp_dir:
                result = run_detection_pipeline(
                    records=records,
                    detector=self.object_detector,
                    output_path=Path(tmp_dir) / "object_detections.parquet",
                    image_loader=self.image_loader,
                )
            output_uri = self.storage.write_parquet_shard(plan.artifact_uris.object_detections_uri, result.frame)
            return StageExecutionResult(
                metrics={
                    "records_seen": result.records_seen,
                    "images_loaded": result.images_loaded,
                    "image_failures": result.image_failures,
                    "detections_written": result.detections_written,
                    "crops_created": result.crops_created,
                    "parquet_batches_written": result.parquet_batches_written,
                },
                outputs={"object_detections": output_uri},
            )
        missing = _missing_paths(plan.paths.source_records_path)
        if missing:
            return StageExecutionResult(status=StageStatus.FAILED, message="missing_detection_inputs: " + ", ".join(missing))
        if self.object_detector is None or self.image_loader is None:
            return StageExecutionResult(status=StageStatus.FAILED, message="detector_runtime_required_for_detect_objects")
        import polars as pl
        from biominer.detection.pipeline import run_detection_pipeline

        plan.paths.ensure_directories()
        records = pl.read_parquet(plan.paths.source_records_path).to_dicts()
        result = run_detection_pipeline(
            records=records,
            detector=self.object_detector,
            output_path=plan.paths.object_detections_path,
            image_loader=self.image_loader,
        )
        return StageExecutionResult(
            metrics={
                "records_seen": result.records_seen,
                "images_loaded": result.images_loaded,
                "image_failures": result.image_failures,
                "detections_written": result.detections_written,
                "crops_created": result.crops_created,
                "parquet_batches_written": result.parquet_batches_written,
            },
            outputs={"object_detections": str(result.output_path)},
        )

    def _run_score_bioclip_stage(self, plan: ProductionRunPlan) -> StageExecutionResult:
        if is_cloud_uri(self.request.output_root):
            if self.storage is None:
                return StageExecutionResult(status=StageStatus.FAILED, message="storage_backend_required_for_score_bioclip")
            missing = _missing_uris(
                self.storage,
                plan.artifact_uris.source_records_uri,
                plan.artifact_uris.object_detections_uri,
            )
            if missing:
                return StageExecutionResult(status=StageStatus.FAILED, message="missing_score_inputs: " + ", ".join(missing))
            if self.object_scorer is None:
                return StageExecutionResult(status=StageStatus.FAILED, message="bioclip_runtime_required_for_score_bioclip")
            from biominer.bioclip.candidate_sets import build_candidate_set_for_taxon_scope
            from biominer.bioclip.object_runner import screen_object_detections

            canonical = self.storage.read_parquet(plan.artifact_uris.source_records_uri)
            detections = self.storage.read_parquet(plan.artifact_uris.object_detections_uri)
            target_context = plan.manifest.taxon_scope.species_contexts[0]
            candidate_set = build_candidate_set_for_taxon_scope(
                plan.manifest.taxon_scope,
                target_context=target_context,
                species_candidate_path=self.species_candidate_path,
                records=canonical.to_dicts(),
                allow_single_target_fixture=self.allow_single_target_fixture,
            )
            with tempfile.TemporaryDirectory(prefix="biominer-score-") as tmp_dir:
                result = screen_object_detections(
                    canonical_records=canonical,
                    detections=detections,
                    species_context=target_context,
                    candidate_set=candidate_set,
                    scorer=self.object_scorer,
                    output_path=Path(tmp_dir) / "object_bioclip_scores.parquet",
                    ablation_mode=self.request.bioclip_ablation_mode,  # type: ignore[arg-type]
                )
            output_uri = self.storage.write_parquet_shard(plan.artifact_uris.object_scores_uri, result.frame)
            return StageExecutionResult(metrics=_object_score_metrics(result), outputs={"object_scores": output_uri})
        missing = _missing_paths(plan.paths.source_records_path, plan.paths.object_detections_path)
        if missing:
            return StageExecutionResult(status=StageStatus.FAILED, message="missing_score_inputs: " + ", ".join(missing))
        if self.object_scorer is None:
            return StageExecutionResult(status=StageStatus.FAILED, message="bioclip_runtime_required_for_score_bioclip")
        import polars as pl
        from biominer.bioclip.candidate_sets import build_candidate_set_for_taxon_scope
        from biominer.bioclip.object_runner import screen_object_detections

        plan.paths.ensure_directories()
        canonical = pl.read_parquet(plan.paths.source_records_path)
        detections = pl.read_parquet(plan.paths.object_detections_path)
        target_context = plan.manifest.taxon_scope.species_contexts[0]
        candidate_set = build_candidate_set_for_taxon_scope(
            plan.manifest.taxon_scope,
            target_context=target_context,
            species_candidate_path=self.species_candidate_path,
            records=canonical.to_dicts(),
            allow_single_target_fixture=self.allow_single_target_fixture,
        )
        result = screen_object_detections(
            canonical_records=canonical,
            detections=detections,
            species_context=target_context,
            candidate_set=candidate_set,
            scorer=self.object_scorer,
            output_path=plan.paths.object_scores_path,
            ablation_mode=self.request.bioclip_ablation_mode,  # type: ignore[arg-type]
        )
        return StageExecutionResult(
            metrics=_object_score_metrics(result),
            outputs={"object_scores": str(result.output_path or plan.paths.object_scores_path)},
        )

    def _registry_query_definitions_source(self) -> tuple[Any, str, Path | None]:
        if self._registry_is_cloud():
            uri = self._registry_artifact_uri("flickr_query_definitions.parquet")
            return self._read_registry_parquet("flickr_query_definitions.parquet"), uri, None
        path = self._registry_query_definitions_path()
        import polars as pl

        return pl.read_parquet(path), str(path), path

    def _registry_query_definitions_path(self) -> Path:
        registry = self._registry_dir_path(stage_name="compile_queries")
        path = registry / "flickr_query_definitions.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def _registry_dir_path(self, *, stage_name: str) -> Path:
        if not self.request.registry_dir:
            raise ValueError(f"registry_dir is required for {stage_name}")
        registry_dir = str(self.request.registry_dir)
        if is_cloud_uri(registry_dir):
            raise ValueError(f"S3 registry reads are not implemented for {stage_name} in the local orchestrator yet")
        return Path(registry_dir)

    def _load_flickr_work_queries(self) -> tuple[FlickrQuery, ...]:
        if self._registry_is_cloud():
            queries = load_registry_flickr_queries_from_frame(self._read_registry_parquet("flickr_query_definitions.parquet"))
        else:
            queries = load_registry_flickr_queries(self._registry_query_definitions_path())
        limit = int(self.request.limits.get("records") or 0)
        return tuple(queries[:limit]) if limit > 0 else queries

    def _registry_is_cloud(self) -> bool:
        return bool(self.request.registry_dir and is_cloud_uri(str(self.request.registry_dir)))

    def _registry_artifact_uri(self, filename: str) -> str:
        if not self.request.registry_dir:
            raise ValueError("registry_dir is required")
        return join_uri(str(self.request.registry_dir), filename)

    def _read_registry_parquet(self, filename: str) -> Any:
        if not self._registry_is_cloud():
            import polars as pl

            return pl.read_parquet(self._registry_dir_path(stage_name="registry_read") / filename)
        if self.storage is None:
            raise ValueError("storage_backend_required_for_registry_reads")
        uri = self._registry_artifact_uri(filename)
        if not self.storage.exists(uri):
            raise FileNotFoundError(uri)
        return self.storage.read_parquet(uri)

    def _read_registry_optional_parquet(self, filename: str) -> Any:
        if not self._registry_is_cloud():
            path = self._registry_dir_path(stage_name="registry_read") / filename
            if not path.exists():
                import polars as pl

                return pl.DataFrame()
            import polars as pl

            return pl.read_parquet(path)
        if self.storage is None:
            raise ValueError("storage_backend_required_for_registry_reads")
        uri = self._registry_artifact_uri(filename)
        if not self.storage.exists(uri):
            import polars as pl

            return pl.DataFrame()
        return self.storage.read_parquet(uri)

    def _read_registry_manifest(self) -> dict[str, Any]:
        if not self._registry_is_cloud():
            path = self._registry_dir_path(stage_name="registry_read") / "manifest.json"
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return {}
        if self.storage is None:
            raise ValueError("storage_backend_required_for_registry_reads")
        uri = self._registry_artifact_uri("manifest.json")
        if not self.storage.exists(uri):
            return {}
        return self.storage.read_json(uri)

    def _run_join_evidence_stage(self, plan: ProductionRunPlan) -> StageExecutionResult:
        if is_cloud_uri(self.request.output_root):
            if self.storage is None:
                return StageExecutionResult(status=StageStatus.FAILED, message="storage_backend_required_for_join_evidence")
            missing = _missing_uris(
                self.storage,
                plan.artifact_uris.source_records_uri,
                plan.artifact_uris.object_detections_uri,
                plan.artifact_uris.object_scores_uri,
            )
            if missing:
                return StageExecutionResult(status=StageStatus.FAILED, message="missing_join_inputs: " + ", ".join(missing))
            joined, photo_summary = build_object_evidence_frames(
                canonical_source_records=self.storage.read_parquet(plan.artifact_uris.source_records_uri),
                object_detections=self.storage.read_parquet(plan.artifact_uris.object_detections_uri),
                object_scores=self.storage.read_parquet(plan.artifact_uris.object_scores_uri),
            )
            object_evidence_uri = self.storage.write_parquet_shard(plan.artifact_uris.object_evidence_uri, joined)
            photo_summary_uri = self.storage.write_parquet_shard(plan.artifact_uris.photo_summary_uri, photo_summary)
            metrics = evidence_count_metrics(joined, photo_summary)
            return StageExecutionResult(
                metrics=metrics,
                outputs={
                    "object_evidence": object_evidence_uri,
                    "photo_summary": photo_summary_uri,
                },
            )
        missing = _missing_paths(plan.paths.source_records_path, plan.paths.object_detections_path, plan.paths.object_scores_path)
        if missing:
            return StageExecutionResult(status=StageStatus.FAILED, message="missing_join_inputs: " + ", ".join(missing))
        import polars as pl

        plan.paths.ensure_directories()
        outputs = write_object_evidence_outputs(
            canonical_source_records=pl.read_parquet(plan.paths.source_records_path),
            object_detections=pl.read_parquet(plan.paths.object_detections_path),
            object_scores=pl.read_parquet(plan.paths.object_scores_path),
            joined_output_path=plan.paths.object_evidence_path,
            photo_summary_output_path=plan.paths.photo_summary_path,
        )
        joined = pl.read_parquet(outputs.object_evidence_joined)
        photo_summary = pl.read_parquet(outputs.photo_evidence_summary)
        metrics = evidence_count_metrics(joined, photo_summary)
        return StageExecutionResult(
            metrics=metrics,
            outputs={
                "object_evidence": str(outputs.object_evidence_joined),
                "photo_summary": str(outputs.photo_evidence_summary),
            },
        )

    def _run_summarize_stage(self, plan: ProductionRunPlan) -> StageExecutionResult:
        if is_cloud_uri(self.request.output_root):
            if self.storage is None:
                return StageExecutionResult(status=StageStatus.FAILED, message="storage_backend_required_for_summarize")
            missing = _missing_uris(
                self.storage,
                plan.artifact_uris.object_evidence_uri,
                plan.artifact_uris.photo_summary_uri,
            )
            if missing:
                return StageExecutionResult(status=StageStatus.FAILED, message="missing_summary_inputs: " + ", ".join(missing))
            joined = self.storage.read_parquet(plan.artifact_uris.object_evidence_uri)
            photo_summary = self.storage.read_parquet(plan.artifact_uris.photo_summary_uri)
            review_queue = build_review_queue(photo_summary)
            metrics = evidence_count_metrics(joined, photo_summary)
            metrics["review_queue_rows"] = review_queue.height
            metrics["review_queue_bin_counts"] = _value_counts(review_queue, "review_bucket")
            metrics_uri = self.storage.write_json(plan.artifact_uris.metrics_uri, metrics)
            review_queue_uri = self.storage.write_parquet_shard(plan.artifact_uris.review_queue_uri, review_queue)
            return StageExecutionResult(metrics=metrics, outputs={"metrics": metrics_uri, "review_queue": review_queue_uri})
        missing = _missing_paths(plan.paths.object_evidence_path, plan.paths.photo_summary_path)
        if missing:
            return StageExecutionResult(status=StageStatus.FAILED, message="missing_summary_inputs: " + ", ".join(missing))
        import json
        import polars as pl

        joined = pl.read_parquet(plan.paths.object_evidence_path)
        photo_summary = pl.read_parquet(plan.paths.photo_summary_path)
        review_queue = build_review_queue(photo_summary)
        metrics = evidence_count_metrics(joined, photo_summary)
        metrics["review_queue_rows"] = review_queue.height
        metrics["review_queue_bin_counts"] = _value_counts(review_queue, "review_bucket")
        plan.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        plan.paths.metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        write_parquet(review_queue, plan.paths.review_queue_path)
        return StageExecutionResult(
            metrics=metrics,
            outputs={
                "metrics": str(plan.paths.metrics_path),
                "review_queue": str(plan.paths.review_queue_path),
            },
        )

    def _write_manifest_if_local(self, plan: ProductionRunPlan) -> Path | str | None:
        if is_cloud_uri(self.request.output_root):
            if self.storage is not None:
                return self.storage.write_json(plan.artifact_uris.manifest_uri, plan.manifest.to_dict())
            return None
        plan.paths.ensure_directories()
        return plan.manifest.write_json(plan.paths.manifest_path)


def _flickr_query_work_item(query: FlickrQuery, *, run_id: str) -> dict[str, Any]:
    return {
        "work_key": f"{run_id}:flickr:{query_hash(query)}",
        "run_id": run_id,
        "query": query.__dict__,
    }


def _flickr_query_from_work_item(item: dict[str, Any]) -> FlickrQuery:
    payload = item.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"work item {item.get('work_key')} has invalid payload")
    query_payload = payload.get("query")
    if isinstance(query_payload, FlickrQuery):
        return query_payload
    if not isinstance(query_payload, dict):
        raise ValueError(f"work item {item.get('work_key')} has no Flickr query payload")
    return FlickrQuery(**query_payload)


def _poller_work_item_id(query: FlickrQuery) -> str:
    payload = json.dumps(asdict(query), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cloud_poll_claim_limit(limits: dict[str, int]) -> int:
    candidates = [
        int(limits.get("api_calls") or SOFT_API_CALLS_PER_HOUR),
        int(limits.get("workers") or 1),
    ]
    records_limit = int(limits.get("records") or 0)
    if records_limit > 0:
        candidates.append(records_limit)
    return max(1, min(value for value in candidates if value > 0))


def _mirror_cloud_poll_state_to_workstore(
    *,
    workstore: WorkStore,
    snapshot: list[dict[str, Any]],
    local_work_keys: dict[str, str],
    run_id: str,
    registry_version: str | None,
    output_uri: str,
) -> dict[str, int]:
    completed = 0
    failed = 0
    unfinished_claimed = 0
    followup_enqueued = 0
    for row in snapshot:
        local_id = str(row["work_item_id"])
        status = str(row["status"])
        query = row["query"]
        work_key = local_work_keys.get(local_id)
        if work_key:
            if status == COMPLETED:
                workstore.mark_completed(
                    work_key,
                    output_uri=output_uri,
                    checksum=None,
                    row_count=int(row["records_returned"] or 0),
                )
                completed += 1
            elif status == FAILED:
                workstore.mark_failed(work_key, str(row["error"] or "flickr_poll_failed"))
                failed += 1
            else:
                workstore.mark_failed(work_key, f"flickr_poll_left_work_item_{status}")
                unfinished_claimed += 1
            continue
        if status == PENDING:
            followup_enqueued += workstore.enqueue_work(
                PRODUCTION_JOB_NAME,
                registry_version,
                [_flickr_query_work_item(query, run_id=run_id)],
                stage=RunStage.POLL_FLICKR.value,
            )
        elif status in {CLAIMED, COMPLETED, FAILED}:
            followup_enqueued += workstore.enqueue_work(
                PRODUCTION_JOB_NAME,
                registry_version,
                [_flickr_query_work_item(query, run_id=run_id)],
                stage=RunStage.POLL_FLICKR.value,
            )
    return {
        "workstore_work_items_completed": completed,
        "workstore_work_items_failed": failed,
        "workstore_unfinished_claimed_items": unfinished_claimed,
        "workstore_followup_work_items_enqueued": followup_enqueued,
    }


def _parquet_row_count(path: Path) -> int:
    import polars as pl

    return pl.scan_parquet(path).select(pl.len()).collect().item()


def _registry_manifest_version(registry: Path) -> str:
    try:
        return str(json.loads((registry / "manifest.json").read_text(encoding="utf-8")).get("registry_version") or "")
    except (FileNotFoundError, json.JSONDecodeError):
        return ""


def _missing_paths(*paths: Path) -> list[str]:
    return [str(path) for path in paths if not path.exists()]


def _missing_uris(storage: CloudStorage, *uris: str) -> list[str]:
    return [uri for uri in uris if not storage.exists(uri)]


def _object_score_metrics(result: Any) -> dict[str, Any]:
    return {
        "records_seen": result.records_seen,
        "detections_seen": result.detections_seen,
        "crops_scored": result.crops_scored,
        "objects_scored": result.crops_scored,
        "score_batches_written": result.score_batches_written,
        "segmentation_unavailable_count": result.segmentation_unavailable_count,
        "segmentation_status": result.segmentation_status,
        "visual_mode": result.visual_mode,
        "visual_mode_status": result.visual_mode_status,
    }


def _merge_stage_counts(manifest: RunManifest, *, stage: RunStage, result: StageExecutionResult) -> RunManifest:
    if stage == RunStage.COMPILE_QUERIES and result.status == StageStatus.COMPLETE:
        return replace(
            manifest,
            query_counts={
                **manifest.query_counts,
                "compiled_definitions": int(result.metrics.get("registry_query_definition_rows", 0)),
                "flickr_work_items": int(result.metrics.get("flickr_work_items", 0)),
            },
        )
    if stage == RunStage.ENQUEUE_FLICKR_WORK and result.status == StageStatus.COMPLETE:
        return replace(
            manifest,
            query_counts={
                **manifest.query_counts,
                "enqueued_work_items": int(result.metrics.get("enqueued_work_items", 0)),
            },
        )
    if stage == RunStage.BUILD_REGISTRY and result.status == StageStatus.COMPLETE:
        return replace(manifest, metrics={**manifest.metrics, **result.metrics})
    if stage == RunStage.POLL_FLICKR and result.status == StageStatus.COMPLETE:
        return replace(
            manifest,
            query_counts={
                **manifest.query_counts,
                "polled_work_items": int(result.metrics.get("work_items_claimed", 0)),
                "api_calls_made": int(result.metrics.get("api_calls_made", 0)),
            },
            metrics={**manifest.metrics, **result.metrics},
        )
    if stage == RunStage.DETECT_OBJECTS and result.status == StageStatus.COMPLETE:
        return replace(
            manifest,
            detection_counts={
                **manifest.detection_counts,
                "images_seen": int(result.metrics.get("records_seen", manifest.detection_counts.get("images_seen", 0))),
                "images_loaded": int(result.metrics.get("images_loaded", manifest.detection_counts.get("images_loaded", 0))),
                "image_failures": int(result.metrics.get("image_failures", manifest.detection_counts.get("image_failures", 0))),
                "detections": int(result.metrics.get("detections_written", manifest.detection_counts.get("detections", 0))),
                "crops_created": int(result.metrics.get("crops_created", manifest.detection_counts.get("crops_created", 0))),
            },
            metrics={**manifest.metrics, **result.metrics},
        )
    if stage == RunStage.SCORE_BIOCLIP and result.status == StageStatus.COMPLETE:
        mode = str(result.metrics.get("visual_mode") or "")
        counts = {
            **manifest.bioclip_counts,
            "objects_scored": int(result.metrics.get("objects_scored", manifest.bioclip_counts.get("objects_scored", 0))),
        }
        if mode == "whole_image":
            counts["whole_images_scored"] = int(result.metrics.get("objects_scored", manifest.bioclip_counts.get("whole_images_scored", 0)))
        if mode == "detector_crop_segmentation":
            counts["segmentation_crops_scored"] = int(result.metrics.get("objects_scored", manifest.bioclip_counts.get("segmentation_crops_scored", 0)))
        return replace(manifest, bioclip_counts=counts, metrics={**manifest.metrics, **result.metrics})
    if stage in {RunStage.JOIN_EVIDENCE, RunStage.SUMMARIZE} and result.status == StageStatus.COMPLETE:
        return replace(
            manifest,
            evidence_counts={
                **manifest.evidence_counts,
                "object_evidence_rows": int(result.metrics.get("object_evidence_rows", manifest.evidence_counts.get("object_evidence_rows", 0))),
                "photo_summary_rows": int(result.metrics.get("photo_summary_rows", manifest.evidence_counts.get("photo_summary_rows", 0))),
                "review_queue_rows": int(result.metrics.get("review_queue_rows", manifest.evidence_counts.get("review_queue_rows", 0))),
            },
            metrics={**manifest.metrics, **result.metrics},
        )
    return manifest


def _value_counts(frame: Any, column: str) -> dict[str, int]:
    if frame.is_empty() or column not in frame.columns:
        return {}
    counts = frame.group_by(column).len(name="count").sort(column).to_dicts()
    return {str(row[column] or ""): int(row["count"]) for row in counts}
