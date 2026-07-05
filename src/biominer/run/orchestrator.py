from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from biominer.bioclip.object_runner import OBJECT_VISUAL_MODES, PRIMARY_VISUAL_CLASSIFIER
from biominer.evidence.join import build_object_evidence_frames, write_object_evidence_outputs
from biominer.evidence.metrics import build_review_queue, evidence_count_metrics
from biominer.flickr_comments.comment_review import CommentReviewState
from biominer.flickr_comments.comments_enrichment import fetch_flickr_comments
from biominer.flickr_fetch.cloud_poller import CloudMetadataPoller, flickr_query_work_item
from biominer.storage.parquet import write_parquet
from biominer.flickr_fetch.metadata_poller import DEFAULT_STALE_CLAIM_SECONDS, SOFT_API_CALLS_PER_HOUR, MetadataPollState, poll_once
from biominer.flickr_fetch.query_planner import FlickrQuery, load_registry_flickr_queries, load_registry_flickr_queries_from_frame
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
REQUIRED_REGISTRY_ARTIFACTS = (
    "taxa.parquet",
    "names.parquet",
    "manifest.json",
    "flickr_query_definitions.parquet",
)


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
    bioclip_ablation_modes: tuple[str, ...] = OBJECT_VISUAL_MODES
    worker_id: str = "local"
    stages: tuple[RunStage, ...] = DEFAULT_PRODUCTION_STAGES
    dry_run: bool = False
    limits: dict[str, int] = field(default_factory=dict)
    build_registry_if_missing: bool = False

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
                "bioclip_ablation_modes": list(self.request.bioclip_ablation_modes),
                "worker_id": self.request.worker_id,
                "stages": [stage.value for stage in self.request.stages],
                "dry_run": self.request.dry_run,
                "limits": dict(self.request.limits),
                "build_registry_if_missing": self.request.build_registry_if_missing,
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
            "bioclip_ablation_modes": list(request.bioclip_ablation_modes),
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
        comment_fetcher: Callable[[str], list[dict[str, Any]]] | None = None,
        registry_builder: Callable[[Path], dict[str, Any]] | None = None,
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
        self.comment_fetcher = comment_fetcher
        self.registry_builder = registry_builder
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
            self.taxon_scope = _limit_taxon_scope(self.taxon_scope, self.request.limits)
            return self.taxon_scope
        if not self.request.registry_dir:
            raise ValueError("registry_dir is required when taxon_scope is not provided")
        if is_cloud_uri(str(self.request.registry_dir)):
            resolved = resolve_taxon_scope_from_registry_frames(
                taxa=self._read_registry_parquet("taxa.parquet"),
                names=self._read_registry_optional_parquet("names.parquet"),
                source_snapshots=self._read_registry_optional_parquet("source_snapshots.parquet"),
                manifest=self._read_registry_manifest(),
                input_name=self.request.taxon,
                input_rank=self.request.rank,
            )
        else:
            resolved = resolve_taxon_scope_from_registry(
                registry_dir=self.request.registry_dir,
                input_name=self.request.taxon,
                input_rank=self.request.rank,
            )
        self.taxon_scope = _limit_taxon_scope(resolved, self.request.limits)
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
        if stage == RunStage.QUEUE_COMMENT_REVIEW:
            return self._run_queue_comment_review_stage(plan)
        if stage == RunStage.REVIEW_COMMENTS:
            return self._run_review_comments_stage(plan)
        if stage == RunStage.APPLY_COMMENT_REVIEW:
            return self._run_apply_comment_review_stage(plan)
        return StageExecutionResult(status=StageStatus.SKIPPED, message="stage_not_implemented")

    def _run_build_registry_stage(self, plan: ProductionRunPlan) -> StageExecutionResult:
        if self._registry_is_cloud():
            if self.storage is None:
                return StageExecutionResult(status=StageStatus.FAILED, message="storage_backend_required_for_build_registry")
            required = tuple(self._registry_artifact_uri(filename) for filename in REQUIRED_REGISTRY_ARTIFACTS)
            missing = _missing_uris(self.storage, *required)
            if missing:
                return StageExecutionResult(status=StageStatus.FAILED, message="missing_registry_inputs: " + ", ".join(missing))
            taxa = self.storage.read_parquet(self._registry_artifact_uri("taxa.parquet"))
            names = self.storage.read_parquet(self._registry_artifact_uri("names.parquet"))
            query_definitions_uri = self._registry_artifact_uri("flickr_query_definitions.parquet")
            query_definition_rows = self.storage.read_parquet(query_definitions_uri).height
            manifest = self.storage.read_json(self._registry_artifact_uri("manifest.json"))
            outputs = {
                "registry_dir": str(self.request.registry_dir),
                "manifest": self._registry_artifact_uri("manifest.json"),
                "taxa": self._registry_artifact_uri("taxa.parquet"),
                "names": self._registry_artifact_uri("names.parquet"),
                "query_definitions": query_definitions_uri,
            }
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
        required = tuple(registry / filename for filename in REQUIRED_REGISTRY_ARTIFACTS)
        missing = _missing_paths(*required)
        built_registry = False
        if missing:
            if not self.request.build_registry_if_missing:
                return StageExecutionResult(status=StageStatus.FAILED, message="missing_registry_inputs: " + ", ".join(missing))
            if self.registry_builder is None:
                return StageExecutionResult(status=StageStatus.FAILED, message="registry_builder_required_for_build_registry_if_missing")
            self.registry_builder(registry)
            built_registry = True
            missing = _missing_paths(*required)
            if missing:
                return StageExecutionResult(status=StageStatus.FAILED, message="missing_registry_inputs_after_build: " + ", ".join(missing))
        query_definitions = registry / "flickr_query_definitions.parquet"
        metrics = {
            "registry_reused": not built_registry,
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
            "query_definitions": str(query_definitions),
        }
        return StageExecutionResult(metrics=metrics, outputs=outputs)

    def _run_compile_queries_stage(self, plan: ProductionRunPlan) -> StageExecutionResult:
        try:
            source_frame, source_ref, _source_path = self._registry_query_definitions_source()
            scoped_frame = _query_definitions_for_taxon_scope(source_frame, plan.manifest.taxon_scope)
            if scoped_frame.is_empty():
                return StageExecutionResult(
                    status=StageStatus.FAILED,
                    message=f"no_registry_query_definitions_for_scope: {plan.manifest.taxon_scope.accepted_scientific_name}",
                    metrics={
                        "registry_query_definition_source_rows": source_frame.height,
                        "registry_query_definition_rows": 0,
                    },
                    outputs={"source_query_definitions": source_ref},
                )
            queries = self._load_flickr_work_queries(scoped_frame)
        except (FileNotFoundError, ValueError) as exc:
            return StageExecutionResult(status=StageStatus.FAILED, message=f"missing_registry_query_definitions: {exc}")
        outputs = {
            "source_query_definitions": source_ref,
            "query_definitions": plan.artifact_uris.query_definitions_uri,
        }
        if not is_cloud_uri(self.request.output_root):
            plan.paths.ensure_directories()
            write_parquet(scoped_frame, plan.paths.query_definitions_path)
            outputs["local_query_definitions"] = str(plan.paths.query_definitions_path)
        else:
            if self.storage is None:
                return StageExecutionResult(status=StageStatus.FAILED, message="storage_backend_required_for_compile_queries")

            uri = self.storage.write_parquet_shard(plan.artifact_uris.query_definitions_uri, scoped_frame)
            outputs["query_definitions"] = uri
        return StageExecutionResult(
            metrics={
                "registry_query_definition_source_rows": source_frame.height,
                "registry_query_definition_rows": scoped_frame.height,
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
            [flickr_query_work_item(query, run_id=plan.manifest.run_id) for query in queries],
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
        initial_work_items_enqueued = state.enqueue_initial_work_items(queries)
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
            worker_id=self.request.worker_id,
            storage_backend="local",
            compact_after_run=True,
        )
        return StageExecutionResult(
            metrics={
                "initial_work_items_enqueued": initial_work_items_enqueued,
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
        worker_id = self.request.worker_id
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
            statuses=["pending"],
            limit=1,
        )
        if not pending_preview:
            return StageExecutionResult(
                metrics={
                    "initial_work_items_enqueued": 0,
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
        poller = CloudMetadataPoller(
            storage=self.storage,
            workstore=self.workstore,
            job_name=PRODUCTION_JOB_NAME,
            stage=RunStage.POLL_FLICKR.value,
            registry_version=registry_version,
            run_id=plan.manifest.run_id,
            worker_id=worker_id,
            storage_prefix=plan.artifact_uris.staging_uri,
            fetch_metadata=self.metadata_fetcher,
            api_key=api_key,
            max_api_calls=int(self.request.limits.get("api_calls") or SOFT_API_CALLS_PER_HOUR),
        )
        result = poller.run_once(claim_limit=claim_limit)
        source_record_shards = result.source_record_shard_uris
        source_records_output = source_record_shards[0] if source_record_shards else ""
        return StageExecutionResult(
            metrics={
                "initial_work_items_enqueued": result.work_items_claimed,
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
                "stale_claims_requeued": 0,
                "workstore_stale_claims_requeued": stale_requeued,
                "workstore_work_items_completed": result.workstore_work_items_completed,
                "workstore_work_items_failed": result.workstore_work_items_failed,
                "workstore_followup_work_items_enqueued": result.workstore_followup_work_items_enqueued,
                "source_record_shards": len(source_record_shards),
            },
            outputs={
                "workstore_stage": RunStage.POLL_FLICKR.value,
                "raw_prefix": join_uri(plan.artifact_uris.staging_uri, "raw", "source=flickr"),
                "evidence_prefix": join_uri(plan.artifact_uris.staging_uri, "evidence"),
                "source_records": source_records_output,
                "source_record_shards": ",".join(source_record_shards),
            },
        )

    def _run_detect_objects_stage(self, plan: ProductionRunPlan) -> StageExecutionResult:
        if is_cloud_uri(self.request.output_root):
            if self.storage is None:
                return StageExecutionResult(status=StageStatus.FAILED, message="storage_backend_required_for_detect_objects")
            if not _cloud_source_records_available(self.storage, self.workstore, plan):
                return StageExecutionResult(status=StageStatus.FAILED, message="missing_detection_inputs: source_records")
            if self.object_detector is None or self.image_loader is None:
                return StageExecutionResult(status=StageStatus.FAILED, message="detector_runtime_required_for_detect_objects")
            from biominer.detection.pipeline import run_detection_pipeline

            records = _read_cloud_source_records(self.storage, self.workstore, plan).to_dicts()
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
            missing = _missing_uris(self.storage, plan.artifact_uris.object_detections_uri)
            if not _cloud_source_records_available(self.storage, self.workstore, plan):
                missing.append("source_records")
            if missing:
                return StageExecutionResult(status=StageStatus.FAILED, message="missing_score_inputs: " + ", ".join(missing))
            if self.object_scorer is None:
                return StageExecutionResult(status=StageStatus.FAILED, message="bioclip_runtime_required_for_score_bioclip")
            from biominer.bioclip.candidate_sets import build_candidate_set_for_taxon_scope

            canonical = _read_cloud_source_records(self.storage, self.workstore, plan)
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
                frame, metrics = _score_object_visual_modes(
                    canonical_records=canonical,
                    detections=detections,
                    species_context=target_context,
                    candidate_set=candidate_set,
                    scorer=self.object_scorer,
                    output_dir=Path(tmp_dir),
                    modes=_request_bioclip_modes(self.request),
                )
            output_uri = self.storage.write_parquet_shard(plan.artifact_uris.object_scores_uri, frame)
            return StageExecutionResult(metrics=metrics, outputs={"object_scores": output_uri})
        missing = _missing_paths(plan.paths.source_records_path, plan.paths.object_detections_path)
        if missing:
            return StageExecutionResult(status=StageStatus.FAILED, message="missing_score_inputs: " + ", ".join(missing))
        if self.object_scorer is None:
            return StageExecutionResult(status=StageStatus.FAILED, message="bioclip_runtime_required_for_score_bioclip")
        import polars as pl
        from biominer.bioclip.candidate_sets import build_candidate_set_for_taxon_scope

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
        mode_output_dir = plan.paths.staging_dir / "object_bioclip_scores_by_mode"
        frame, metrics = _score_object_visual_modes(
            canonical_records=canonical,
            detections=detections,
            species_context=target_context,
            candidate_set=candidate_set,
            scorer=self.object_scorer,
            output_dir=mode_output_dir,
            modes=_request_bioclip_modes(self.request),
        )
        write_parquet(frame, plan.paths.object_scores_path)
        return StageExecutionResult(
            metrics=metrics,
            outputs={"object_scores": str(plan.paths.object_scores_path)},
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
            raise ValueError(f"registry_dir is a cloud URI for {stage_name}; read registry artifacts through the configured storage backend")
        return Path(registry_dir)

    def _load_flickr_work_queries(self, query_definitions: Any | None = None) -> tuple[FlickrQuery, ...]:
        if query_definitions is None:
            if self._registry_is_cloud():
                frame = self._read_registry_parquet("flickr_query_definitions.parquet")
            else:
                import polars as pl

                frame = pl.read_parquet(self._registry_query_definitions_path())
            query_definitions = _query_definitions_for_taxon_scope(frame, self._resolve_taxon_scope())
        queries = load_registry_flickr_queries_from_frame(query_definitions)
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
                plan.artifact_uris.object_detections_uri,
                plan.artifact_uris.object_scores_uri,
            )
            if not _cloud_source_records_available(self.storage, self.workstore, plan):
                missing.append("source_records")
            if missing:
                return StageExecutionResult(status=StageStatus.FAILED, message="missing_join_inputs: " + ", ".join(missing))
            joined, photo_summary = build_object_evidence_frames(
                canonical_source_records=_read_cloud_source_records(self.storage, self.workstore, plan),
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

    def _run_queue_comment_review_stage(self, plan: ProductionRunPlan) -> StageExecutionResult:
        target_context = plan.manifest.taxon_scope.species_contexts[0]
        if is_cloud_uri(self.request.output_root):
            return StageExecutionResult(status=StageStatus.FAILED, message="cloud_comment_review_state_not_implemented")
        missing = _missing_paths(plan.paths.object_evidence_path)
        if missing:
            return StageExecutionResult(status=StageStatus.FAILED, message="missing_comment_review_inputs: " + ", ".join(missing))
        import polars as pl

        plan.paths.ensure_directories()
        frame = pl.read_parquet(plan.paths.object_evidence_path)
        state = CommentReviewState(plan.paths.comment_review_state_path, species_context=target_context)
        created = state.enqueue_records(frame.to_dicts())
        metrics = {**state.summary(), "comment_review_queue_created": created}
        return StageExecutionResult(
            metrics=metrics,
            outputs={"comment_review_state": str(plan.paths.comment_review_state_path)},
        )

    def _run_review_comments_stage(self, plan: ProductionRunPlan) -> StageExecutionResult:
        target_context = plan.manifest.taxon_scope.species_contexts[0]
        if is_cloud_uri(self.request.output_root):
            return StageExecutionResult(status=StageStatus.FAILED, message="cloud_comment_review_state_not_implemented")
        if not plan.paths.comment_review_state_path.exists():
            return StageExecutionResult(status=StageStatus.FAILED, message=f"missing_comment_review_state: {plan.paths.comment_review_state_path}")
        fetcher = self.comment_fetcher
        api_key = self.flickr_api_key or os.environ.get("FLICKR_API_KEY")
        if fetcher is None:
            if not api_key:
                return StageExecutionResult(status=StageStatus.FAILED, message="flickr_fetcher_or_api_key_required_for_review_comments")
            fetcher = fetch_flickr_comments(api_key=api_key)
        state = CommentReviewState(plan.paths.comment_review_state_path, species_context=target_context)
        result = state.process_pending(
            fetch_comments=fetcher,
            max_api_calls=int(self.request.limits.get("comment_api_calls") or 300),
        )
        metrics = {**state.summary(), **result}
        return StageExecutionResult(
            metrics=metrics,
            outputs={"comment_review_state": str(plan.paths.comment_review_state_path)},
        )

    def _run_apply_comment_review_stage(self, plan: ProductionRunPlan) -> StageExecutionResult:
        target_context = plan.manifest.taxon_scope.species_contexts[0]
        if is_cloud_uri(self.request.output_root):
            return StageExecutionResult(status=StageStatus.FAILED, message="cloud_comment_review_state_not_implemented")
        missing = _missing_paths(plan.paths.object_evidence_path, plan.paths.comment_review_state_path)
        if missing:
            return StageExecutionResult(status=StageStatus.FAILED, message="missing_comment_review_apply_inputs: " + ", ".join(missing))
        import polars as pl

        frame = pl.read_parquet(plan.paths.object_evidence_path)
        state = CommentReviewState(plan.paths.comment_review_state_path, species_context=target_context)
        rows = state.apply_decisions_to_records(frame.to_dicts())
        output = write_parquet(pl.DataFrame(rows), plan.paths.reviewed_object_evidence_path)
        moved_gold = sum(1 for row in rows if row.get("comment_review_decision") == "move_to_gold")
        moved_silver = sum(1 for row in rows if row.get("comment_review_decision") == "move_to_silver")
        return StageExecutionResult(
            metrics={
                **state.summary(),
                "records_moved_to_gold": moved_gold,
                "records_moved_to_silver": moved_silver,
                "reviewed_object_evidence_rows": len(rows),
            },
            outputs={"reviewed_object_evidence": str(output)},
        )

    def _write_manifest_if_local(self, plan: ProductionRunPlan) -> Path | str | None:
        if is_cloud_uri(self.request.output_root):
            if self.storage is not None:
                return self.storage.write_json(plan.artifact_uris.manifest_uri, plan.manifest.to_dict())
            return None
        plan.paths.ensure_directories()
        return plan.manifest.write_json(plan.paths.manifest_path)


def _cloud_poll_claim_limit(limits: dict[str, int]) -> int:
    candidates = [
        int(limits.get("api_calls") or SOFT_API_CALLS_PER_HOUR),
        int(limits.get("workers") or 1),
    ]
    records_limit = int(limits.get("records") or 0)
    if records_limit > 0:
        candidates.append(records_limit)
    return max(1, min(value for value in candidates if value > 0))


def _cloud_source_records_available(storage: CloudStorage, workstore: WorkStore | None, plan: ProductionRunPlan) -> bool:
    if storage.exists(plan.artifact_uris.source_records_uri):
        return True
    return bool(_cloud_source_record_shard_uris(workstore, plan))


def _read_cloud_source_records(storage: CloudStorage, workstore: WorkStore | None, plan: ProductionRunPlan) -> Any:
    if storage.exists(plan.artifact_uris.source_records_uri):
        return storage.read_parquet(plan.artifact_uris.source_records_uri)
    shard_uris = _cloud_source_record_shard_uris(workstore, plan)
    if not shard_uris:
        raise FileNotFoundError("source_records")
    import polars as pl

    frames = [storage.read_parquet(uri) for uri in shard_uris]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def _cloud_source_record_shard_uris(workstore: WorkStore | None, plan: ProductionRunPlan) -> list[str]:
    if workstore is None:
        return []
    shards = workstore.list_committed_shards(
        job_name=PRODUCTION_JOB_NAME,
        stage=RunStage.POLL_FLICKR.value,
        registry_version=plan.manifest.taxon_scope.registry_version,
        run_id=plan.manifest.run_id,
    )
    return [str(shard["uri"]) for shard in shards]


def _parquet_row_count(path: Path) -> int:
    import polars as pl

    return pl.scan_parquet(path).select(pl.len()).collect().item()


def _query_definitions_for_taxon_scope(frame: Any, taxon_scope: TaxonScope) -> Any:
    if frame.is_empty():
        return frame
    scope_keys = sorted(
        {
            str(context.accepted_taxon_key or "")
            for context in taxon_scope.species_contexts
            if str(context.accepted_taxon_key or "")
        }
        | {
            str(context.species_key or "")
            for context in taxon_scope.species_contexts
            if str(context.species_key or "")
        }
    )
    if not scope_keys:
        return frame.head(0)
    predicates = []
    if "accepted_taxon_key" in frame.columns:
        predicates.append(frame["accepted_taxon_key"].is_in(scope_keys))
    if "species_key" in frame.columns:
        predicates.append(frame["species_key"].is_in(scope_keys))
    if not predicates:
        raise ValueError("registry query definitions must include accepted_taxon_key or species_key for scoped production runs")
    predicate = predicates[0]
    for extra in predicates[1:]:
        predicate = predicate | extra
    return frame.filter(predicate)


def _limit_taxon_scope(taxon_scope: TaxonScope, limits: Mapping[str, int]) -> TaxonScope:
    limit = int(limits.get("species") or 0)
    if limit <= 0 or taxon_scope.species_count <= limit:
        return taxon_scope
    return replace(taxon_scope, species_contexts=taxon_scope.species_contexts[:limit])


def _registry_manifest_version(registry: Path) -> str:
    try:
        return str(json.loads((registry / "manifest.json").read_text(encoding="utf-8")).get("registry_version") or "")
    except (FileNotFoundError, json.JSONDecodeError):
        return ""


def _missing_paths(*paths: Path) -> list[str]:
    return [str(path) for path in paths if not path.exists()]


def _missing_uris(storage: CloudStorage, *uris: str) -> list[str]:
    return [uri for uri in uris if not storage.exists(uri)]


def _request_bioclip_modes(request: ProductionRunRequest) -> tuple[Any, ...]:
    raw_modes = request.bioclip_ablation_modes or (request.bioclip_ablation_mode,)
    modes = tuple(str(mode) for mode in raw_modes if str(mode).strip())
    if not modes:
        modes = (request.bioclip_ablation_mode,)
    unsupported = [mode for mode in modes if mode not in OBJECT_VISUAL_MODES]
    if unsupported:
        raise ValueError("unsupported BioCLIP visual mode(s): " + ", ".join(unsupported))
    return modes


def _score_object_visual_modes(
    *,
    canonical_records: Any,
    detections: Any,
    species_context: Any,
    candidate_set: Any,
    scorer: Any,
    output_dir: Path,
    modes: tuple[Any, ...],
) -> tuple[Any, dict[str, Any]]:
    import polars as pl
    from biominer.bioclip.ablation import run_object_ablations

    report = run_object_ablations(
        canonical_records=canonical_records,
        detections=detections,
        species_context=species_context,
        candidate_set=candidate_set,
        scorer=scorer,
        output_dir=output_dir,
        modes=modes,  # type: ignore[arg-type]
    )
    frames = [
        pl.read_parquet(path)
        for mode in report.modes
        if (path := report.output_dir / f"object_bioclip_scores_{mode}.parquet").exists()
    ]
    frame = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    return frame, _object_ablation_metrics(report.report, frame=frame)


def _object_ablation_metrics(report: dict[str, Any], *, frame: Any) -> dict[str, Any]:
    return {
        "records_seen": int(report.get("records_seen") or 0),
        "detections_seen": int(report.get("detections_seen") or 0),
        "crops_scored": int(report.get("crops_scored") or 0),
        "objects_scored": int(report.get("crops_scored") or 0),
        "whole_images_scored": _mode_row_count(frame, "whole_image"),
        "detector_crops_scored": _mode_row_count(frame, "detector_crop"),
        "segmentation_crops_scored": _mode_row_count(frame, "detector_crop_segmentation"),
        "score_batches_written": int(report.get("score_batches_written") or 0),
        "primary_visual_classifier": report.get("primary_visual_classifier"),
        "visual_modes_requested": list(report.get("visual_modes_requested") or []),
        "visual_modes_scored": list(report.get("visual_modes_scored") or []),
        "visual_mode_status_by_mode": dict(report.get("visual_mode_status_by_mode") or {}),
        "segmentation_status_by_mode": dict(report.get("segmentation_status_by_mode") or {}),
        "segmentation_unavailable_count_by_mode": dict(report.get("segmentation_unavailable_count_by_mode") or {}),
        "segmentation_unavailable_reason_by_mode": dict(report.get("segmentation_unavailable_reason_by_mode") or {}),
        "segmentation_unavailable_count": sum(int(value or 0) for value in dict(report.get("segmentation_unavailable_count_by_mode") or {}).values()),
    }


def _mode_row_count(frame: Any, mode: str) -> int:
    if frame.is_empty() or "ablation_mode" not in frame.columns:
        return 0
    return frame.filter(frame["ablation_mode"] == mode).height


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
        counts = {
            **manifest.bioclip_counts,
            "objects_scored": int(result.metrics.get("objects_scored", manifest.bioclip_counts.get("objects_scored", 0))),
            "detector_crops_scored": int(result.metrics.get("detector_crops_scored", manifest.bioclip_counts.get("detector_crops_scored", 0))),
            "whole_images_scored": int(result.metrics.get("whole_images_scored", manifest.bioclip_counts.get("whole_images_scored", 0))),
            "segmentation_crops_scored": int(result.metrics.get("segmentation_crops_scored", manifest.bioclip_counts.get("segmentation_crops_scored", 0))),
        }
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
    if stage in {RunStage.QUEUE_COMMENT_REVIEW, RunStage.REVIEW_COMMENTS, RunStage.APPLY_COMMENT_REVIEW} and result.status == StageStatus.COMPLETE:
        return replace(manifest, metrics={**manifest.metrics, **result.metrics})
    return manifest


def _value_counts(frame: Any, column: str) -> dict[str, int]:
    if frame.is_empty() or column not in frame.columns:
        return {}
    counts = frame.group_by(column).len(name="count").sort(column).to_dicts()
    return {str(row[column] or ""): int(row["count"]) for row in counts}
