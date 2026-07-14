from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
import json
import os
from pathlib import Path
from typing import Any

from biominer.bioclip.cascade_contract import (
    DEFAULT_RANK_BEAM_WIDTH,
    DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    DEFAULT_SPECIES_REPORT_TOP_K,
    DEFAULT_SPECIES_RERANK_TOP_K,
    GLOBAL_RANK_TOP_K_BEAM_STRATEGY,
    production_cascade_work_identity,
    validate_production_cascade_settings,
)
from biominer.bioclip.classification_modes import (
    DEFAULT_CLASSIFICATION_MODE,
    HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    ClassificationMode,
    normalize_classification_mode,
)
from biominer.bioclip.object_runner import OBJECT_VISUAL_MODES, PRIMARY_VISUAL_CLASSIFIER, object_score_audit_metrics
from biominer.bioclip.path_taxonomy_store import PathTaxonomyStore
from biominer.bioclip.taxonomy_embedding_cache import TaxonomyTextEmbeddingIndex
from biominer.detection.policy import DetectionPolicy, VisionRuntimeSettings
from biominer.evidence.cloud_work import (
    CloudReviewQueueResult,
    join_evidence_batch_id,
    join_object_evidence_from_cloud_shards,
    photo_summary_batch_id,
    review_queue_batch_id,
    summarize_photo_evidence_from_cloud_shards,
)
from biominer.evidence.join import write_object_evidence_outputs
from biominer.evidence.metrics import build_review_queue, evidence_count_metrics
from biominer.evaluation.qa import build_visual_qa_findings
from biominer.evaluation.review_queue import build_hierarchical_review_queue
from biominer.flickr_comments.comment_review import CommentReviewState
from biominer.flickr_comments.comments_enrichment import fetch_flickr_comments
from biominer.flickr_fetch.cloud_poller import CloudMetadataPoller, flickr_query_work_item
from biominer.storage.parquet import write_parquet
from biominer.flickr_fetch.metadata_poller import DEFAULT_STALE_CLAIM_SECONDS, SOFT_API_CALLS_PER_HOUR, MetadataPollState, poll_once
from biominer.flickr_fetch.query_planner import FlickrQuery, load_registry_flickr_queries_from_frame
from biominer.reports.vision import build_vision_stage_metrics, write_vision_stage_reports
from biominer.references.readiness import (
    ReferenceBankReadinessPermit,
    load_reference_bank_readiness,
    reference_readiness_allows_vision,
)
from biominer.run.manifest import RunManifest, utc_now_iso
from biominer.run.paths import RunArtifactUris, RunPaths
from biominer.run.stages import (
    DEFAULT_PRODUCTION_STAGES,
    MANUAL_REVIEW_STAGES,
    RunStage,
    StageStatus,
    default_stage_records,
)
from biominer.run.taxon_scope import InputRank, TaxonScope, resolve_taxon_scope_from_registry, resolve_taxon_scope_from_registry_frames
from biominer.storage.cloud import CloudStorage
from biominer.storage.paths import build_evidence_shard_uri, safe_path_component
from biominer.storage.uri import is_cloud_uri, join_uri
from biominer.vision.cloud_work import (
    ROLLING_VISION_ARTIFACT_ORDER,
    RollingVisionPipelineError,
    commit_rolling_vision_batch_shards,
    detect_cloud_rolling_vision_batch,
    enqueue_rolling_vision_work_from_source_shards,
    rolling_vision_batch_id,
    rolling_vision_part_id,
    run_bounded_cloud_rolling_pipeline,
    score_cloud_rolling_detection_batch,
)
from biominer.vision.gates import BioClipGatePolicy
from biominer.workstore.base import WorkStore


DEFAULT_BIOCLIP_MODEL = "imageomics/bioclip-2.5-vith14"
DEFAULT_VISION_BACKEND = "yoloe26"
PRODUCTION_VISUAL_MODE = "detector_crop"
PRODUCTION_BIOCLIP_COMPARISON_ROUTES = ("adult_field",)
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
VisionRuntimeFactory = Callable[[], tuple[Any, Callable[[dict[str, Any]], Any], Any, list[Any]]]


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
    vision_profile: str | None = None
    vision_settings: VisionRuntimeSettings = field(default_factory=VisionRuntimeSettings)
    classification_mode: ClassificationMode = DEFAULT_CLASSIFICATION_MODE
    taxonomy_candidate_table: str | Path | None = None  # Deprecated; registry_dir is canonical.
    taxonomy_text_embedding_cache: str | Path | None = None
    reference_bank_readiness: str | Path | None = None
    reference_bank_readiness_sha256: str | None = None
    beam_strategy: str = GLOBAL_RANK_TOP_K_BEAM_STRATEGY
    rank_beam_width: int = DEFAULT_RANK_BEAM_WIDTH
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K
    species_report_top_k: int = DEFAULT_SPECIES_REPORT_TOP_K
    worker_id: str = "local"
    stages: tuple[RunStage, ...] = DEFAULT_PRODUCTION_STAGES
    dry_run: bool = False
    limits: dict[str, int] = field(default_factory=dict)
    build_registry_if_missing: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "classification_mode", normalize_classification_mode(self.classification_mode))
        settings = validate_production_cascade_settings(
            beam_strategy=self.beam_strategy,
            rank_beam_width=self.rank_beam_width,
            species_first_pass_top_k=self.species_first_pass_top_k,
            species_rerank_top_k=self.species_rerank_top_k,
            species_report_top_k=self.species_report_top_k,
        )
        for field_name, value in zip(
            (
                "beam_strategy",
                "rank_beam_width",
                "species_first_pass_top_k",
                "species_rerank_top_k",
                "species_report_top_k",
            ),
            settings,
            strict=True,
        ):
            object.__setattr__(self, field_name, value)

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
                "vision_worker": "rolling" if is_cloud_uri(str(self.request.output_root)) else "local_dev",
                "bioclip_model": self.request.bioclip_model,
                "vision_profile": self.request.vision_profile,
                "vision_settings": asdict(self.request.vision_settings),
                "classification_mode": self.request.classification_mode,
                "registry_taxonomy": self.request.registry_dir,
                "taxonomy_candidate_table": str(self.request.taxonomy_candidate_table) if self.request.taxonomy_candidate_table else None,
                "taxonomy_text_embedding_cache": str(self.request.taxonomy_text_embedding_cache)
                if self.request.taxonomy_text_embedding_cache
                else None,
                "reference_bank_readiness": str(self.request.reference_bank_readiness)
                if self.request.reference_bank_readiness
                else None,
                "reference_bank_readiness_sha256": (
                    self.request.reference_bank_readiness_sha256
                ),
                "beam_strategy": self.request.beam_strategy,
                "rank_beam_width": self.request.rank_beam_width,
                "species_first_pass_top_k": self.request.species_first_pass_top_k,
                "species_rerank_top_k": self.request.species_rerank_top_k,
                "species_report_top_k": self.request.species_report_top_k,
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
            "vision_worker": "rolling" if is_cloud_uri(str(request.output_root)) else "local_dev",
            "bioclip_model": request.bioclip_model,
            "vision_profile": request.vision_profile,
            "vision_settings": asdict(request.vision_settings),
            "classification_mode": request.classification_mode,
            "registry_taxonomy": request.registry_dir,
            "taxonomy_candidate_table": str(request.taxonomy_candidate_table) if request.taxonomy_candidate_table else None,
            "taxonomy_text_embedding_cache": str(request.taxonomy_text_embedding_cache)
            if request.taxonomy_text_embedding_cache
            else None,
            "reference_bank_readiness": (
                {
                    "artifact_path": str(request.reference_bank_readiness),
                    "expected_sha256": request.reference_bank_readiness_sha256,
                    "validation_status": "not_validated",
                }
                if request.reference_bank_readiness
                else None
            ),
            "beam_strategy": request.beam_strategy,
            "rank_beam_width": request.rank_beam_width,
            "species_first_pass_top_k": request.species_first_pass_top_k,
            "species_rerank_top_k": request.species_rerank_top_k,
            "species_report_top_k": request.species_report_top_k,
            "primary_visual_classifier": PRIMARY_VISUAL_CLASSIFIER,
            "visual_modes": [PRODUCTION_VISUAL_MODE],
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
        vision_runtime_factory: VisionRuntimeFactory | None = None,
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
        self.species_candidate_path = self._resolve_species_candidate_path(
            registry_taxonomy_override=species_candidate_path or self.request.taxonomy_candidate_table,
            resolved_registry_dir=self.request.registry_dir,
        )
        self.allow_single_target_fixture = allow_single_target_fixture
        self.stage_handlers = dict(stage_handlers or {})
        self.vision_runtime_factory = vision_runtime_factory
        self._vision_runtime_initialized = False
        self._vision_runtime_resources: list[Any] = []
        self._vision_runtime_resources_closed = False
        self._reference_bank_readiness_permit: ReferenceBankReadinessPermit | None = None

    def _resolve_species_candidate_path(
        self,
        *,
        registry_taxonomy_override: str | Path | None,
        resolved_registry_dir: str | None,
    ) -> str | Path | None:
        if registry_taxonomy_override is not None:
            return registry_taxonomy_override
        if not resolved_registry_dir:
            return None
        if self.request.classification_mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION:
            return resolved_registry_dir if is_cloud_uri(str(resolved_registry_dir)) else Path(resolved_registry_dir)
        if is_cloud_uri(str(resolved_registry_dir)):
            return join_uri(str(resolved_registry_dir), "taxa.parquet")
        return Path(resolved_registry_dir) / "taxa.parquet"

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
        try:
            for stage in self.request.stages:
                started_at = utc_now_iso()
                manifest = plan.manifest.with_stage_status(stage, StageStatus.RUNNING, started_at=started_at)
                plan = replace(plan, manifest=manifest)
                result = self._run_stage(plan, stage)
                if stage in MANUAL_REVIEW_STAGES and result.status is StageStatus.COMPLETE:
                    result = replace(
                        result,
                        status=StageStatus.AWAITING_MANUAL_REVIEW,
                        message=result.message or "manual_review_required",
                    )
                manifest = self._manifest_with_reference_bank_permit(plan.manifest)
                manifest = manifest.with_stage_status(
                    stage,
                    result.status,
                    ended_at=(
                        None
                        if result.status is StageStatus.AWAITING_MANUAL_REVIEW
                        else utc_now_iso()
                    ),
                    message=result.message,
                    metrics=result.metrics,
                    outputs=result.outputs,
                )
                manifest = _merge_stage_counts(manifest, stage=stage, result=result)
                plan = replace(plan, manifest=manifest)
                if result.status is StageStatus.AWAITING_MANUAL_REVIEW:
                    break
            if any(stage.status is StageStatus.FAILED for stage in plan.manifest.stages):
                final_status = StageStatus.FAILED.value
                ended_at = utc_now_iso()
            elif any(
                stage.status is StageStatus.AWAITING_MANUAL_REVIEW
                for stage in plan.manifest.stages
            ):
                final_status = StageStatus.AWAITING_MANUAL_REVIEW.value
                ended_at = None
            else:
                final_status = StageStatus.COMPLETE.value
                ended_at = utc_now_iso()
            plan = replace(
                plan,
                manifest=plan.manifest.with_status(final_status, ended_at=ended_at),
            )
            self._write_manifest_if_local(plan)
            return plan
        finally:
            self._close_vision_runtime_resources()

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
        if not self.request.dry_run and stage in {RunStage.DETECT_OBJECTS, RunStage.SCORE_BIOCLIP}:
            self._load_reference_bank_readiness_permit(plan)
            plan = replace(plan, manifest=self._manifest_with_reference_bank_permit(plan.manifest))
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
        if stage in MANUAL_REVIEW_STAGES:
            return StageExecutionResult(
                status=StageStatus.AWAITING_MANUAL_REVIEW,
                message="manual_review_required",
            )
        if stage == RunStage.BUILD_REGISTRY:
            return self._run_build_registry_stage(plan)
        if stage == RunStage.COMPILE_QUERIES:
            return self._run_compile_queries_stage(plan)
        if stage == RunStage.ENQUEUE_FLICKR_WORK:
            return self._run_enqueue_flickr_work_stage(plan)
        if stage == RunStage.POLL_FLICKR:
            return self._run_poll_flickr_stage(plan)
        if stage == RunStage.DETECT_OBJECTS:
            self._initialize_vision_runtime()
            return self._run_detect_objects_stage(plan)
        if stage == RunStage.SCORE_BIOCLIP:
            self._initialize_vision_runtime()
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

    def _load_reference_bank_readiness_permit(
        self,
        plan: ProductionRunPlan,
    ) -> ReferenceBankReadinessPermit:
        if self._reference_bank_readiness_permit is not None:
            return self._reference_bank_readiness_permit
        readiness_path = self.request.reference_bank_readiness
        if readiness_path is None:
            raise ValueError(
                "reference_bank_readiness is required before detect_objects or score_bioclip"
            )
        readiness_sha256 = self.request.reference_bank_readiness_sha256
        if readiness_sha256 is None:
            raise ValueError(
                "reference_bank_readiness_sha256 is required before detect_objects or score_bioclip"
            )
        runtime_model_name = str(
            getattr(self.object_scorer, "model_id", "") or self.request.bioclip_model
        )
        runtime_preprocessing_version = str(
            getattr(self.object_scorer, "preprocessing_version", "") or ""
        )
        runtime_model_input_fingerprint = str(
            getattr(self.object_scorer, "model_input_fingerprint", "") or ""
        )
        identity_expectations = {
            "expected_registry_version": plan.manifest.taxon_scope.registry_version,
            "expected_target_accepted_taxon_key": (
                plan.manifest.taxon_scope.accepted_taxon_key
            ),
            "expected_model_name": runtime_model_name,
            "expected_readiness_sha256": readiness_sha256,
        }
        if runtime_preprocessing_version:
            identity_expectations["expected_preprocessing_version"] = (
                runtime_preprocessing_version
            )
        if runtime_model_input_fingerprint:
            identity_expectations["expected_model_input_fingerprint"] = (
                runtime_model_input_fingerprint
            )
        permit = load_reference_bank_readiness(
            readiness_path,
            **identity_expectations,
        )
        if not reference_readiness_allows_vision(permit.status):
            raise ValueError(
                "reference bank readiness does not permit vision: "
                f"status={permit.status}"
            )
        if (
            self.object_scorer is not None
            and self.request.classification_mode
            != HIERARCHICAL_BUTTERFLY_CLASSIFICATION
        ):
            bind_readiness = getattr(
                self.object_scorer,
                "bind_reference_readiness",
                None,
            )
            if callable(bind_readiness):
                bind_readiness(permit)
            else:
                self._validate_reference_runtime_identity(
                    scorer=self.object_scorer,
                    permit=permit,
                )
                if not self.allow_single_target_fixture:
                    raise ValueError(
                        "injected production BioCLIP scorer cannot attest its "
                        "loaded reference model identity"
                    )
            self._validate_reference_runtime_identity(
                scorer=self.object_scorer,
                permit=permit,
            )
        self._reference_bank_readiness_permit = permit
        return permit

    def _initialize_vision_runtime(self) -> None:
        if self._vision_runtime_initialized or self.vision_runtime_factory is None:
            return
        detector, image_loader, scorer, resources = self.vision_runtime_factory()
        self._vision_runtime_resources.extend(resources)
        permit = self._reference_bank_readiness_permit
        if permit is None:
            raise ValueError(
                "reference bank readiness must be validated before vision runtime initialization"
            )
        if (
            self.request.classification_mode
            != HIERARCHICAL_BUTTERFLY_CLASSIFICATION
        ):
            bind_readiness = getattr(scorer, "bind_reference_readiness", None)
            if callable(bind_readiness):
                bind_readiness(permit)
            else:
                self._validate_reference_runtime_identity(
                    scorer=scorer,
                    permit=permit,
                )
                if not self.allow_single_target_fixture:
                    raise ValueError(
                        "production BioCLIP scorer cannot attest its loaded reference "
                        "model identity"
                    )
            self._validate_reference_runtime_identity(scorer=scorer, permit=permit)
        self._vision_runtime_initialized = True
        if self.object_detector is None:
            self.object_detector = detector
        if self.image_loader is None:
            self.image_loader = image_loader
        if self.object_scorer is None:
            self.object_scorer = scorer

    def _manifest_with_reference_bank_permit(self, manifest: RunManifest) -> RunManifest:
        permit = self._reference_bank_readiness_permit
        if permit is None:
            return manifest
        readiness_identity = {
            "artifact_path": str(self.request.reference_bank_readiness),
            "expected_sha256": self.request.reference_bank_readiness_sha256,
            "validation_status": "validated",
            **asdict(permit),
        }
        return replace(
            manifest,
            model_configs={
                **manifest.model_configs,
                "reference_bank_readiness": readiness_identity,
            },
            metrics={
                **manifest.metrics,
                "reference_bank_readiness_status": str(permit.status),
                "reference_bank_readiness_sha256": permit.readiness_sha256,
            },
        )

    def _close_vision_runtime_resources(self) -> None:
        if self._vision_runtime_resources_closed:
            return
        self._vision_runtime_resources_closed = True
        for resource in reversed(self._vision_runtime_resources):
            close = getattr(resource, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _validate_reference_runtime_identity(
        *,
        scorer: Any,
        permit: ReferenceBankReadinessPermit,
    ) -> None:
        runtime_identity = {
            "model_name": str(
                getattr(scorer, "model_id", "")
                or getattr(scorer, "model_name", "")
                or ""
            ),
            "model_version": str(getattr(scorer, "model_version", "") or ""),
            "checkpoint_sha256": str(
                getattr(scorer, "checkpoint_sha256", "") or ""
            ),
            "preprocessing_version": str(
                getattr(scorer, "preprocessing_version", "") or ""
            ),
            "input_contract_version": str(
                getattr(scorer, "input_contract_version", "") or ""
            ),
            "model_input_fingerprint": str(
                getattr(scorer, "model_input_fingerprint", "") or ""
            ),
        }
        expected_identity = {
            field_name: str(getattr(permit, field_name))
            for field_name in runtime_identity
        }
        missing = sorted(
            field_name
            for field_name, value in runtime_identity.items()
            if not value
        )
        if missing:
            raise ValueError(
                "vision runtime does not declare the full reference model input identity: "
                + ", ".join(missing)
            )
        mismatches = sorted(
            field_name
            for field_name, value in runtime_identity.items()
            if (
                _canonical_bioclip_model_id(value)
                != _canonical_bioclip_model_id(expected_identity[field_name])
                if field_name == "model_name"
                else value != expected_identity[field_name]
            )
        )
        if mismatches:
            raise ValueError(
                "vision runtime reference model input identity mismatch: "
                + ", ".join(mismatches)
            )

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
            return self._run_cloud_rolling_vision_stage(plan)
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
            detection_policy=self.request.vision_settings.to_detection_policy(DetectionPolicy(backend=self.object_detector.backend)),
            run_policy=self.request.vision_settings.to_detection_run_policy(),
        )
        return StageExecutionResult(
            metrics={
                "records_seen": result.records_seen,
                "images_loaded": result.images_loaded,
                "image_failures": result.image_failures,
                "detections_written": result.detections_written,
                "crops_created": result.crops_created,
                "adaptive_batching_enabled": result.adaptive_batching_enabled,
                "detector_batch_retries": result.detector_batch_retries,
                "detector_batch_size_initial": result.detector_batch_size_initial,
                "detector_batch_size_final": result.detector_batch_size_final,
                "detector_batch_size_min": result.detector_batch_size_min,
                "parquet_batches_written": result.parquet_batches_written,
            },
            outputs={"object_detections": str(result.output_path)},
        )

    def _run_cloud_rolling_vision_stage(self, plan: ProductionRunPlan) -> StageExecutionResult:
        if self.storage is None:
            return StageExecutionResult(status=StageStatus.FAILED, message="storage_backend_required_for_detect_objects")
        if self.workstore is None:
            return StageExecutionResult(status=StageStatus.FAILED, message="workstore_required_for_detect_objects")
        if not _cloud_source_records_available(self.storage, self.workstore, plan):
            return StageExecutionResult(status=StageStatus.FAILED, message="missing_detection_inputs: source_records")
        if self.object_detector is None or self.image_loader is None or self.object_scorer is None:
            return StageExecutionResult(status=StageStatus.FAILED, message="rolling_vision_runtime_required_for_detect_objects")

        path_taxonomy_store: PathTaxonomyStore | None = None
        taxonomy_text_embedding_index: TaxonomyTextEmbeddingIndex | None = None
        cascade_identity: dict[str, Any] | None = None
        taxonomy_metrics: dict[str, Any] = {}
        if self.request.classification_mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION:
            path_taxonomy_store, taxonomy_status = self._load_valid_hierarchical_taxonomy_store()
            if taxonomy_status.status is StageStatus.FAILED:
                return taxonomy_status
            taxonomy_metrics = dict(taxonomy_status.metrics)
            assert path_taxonomy_store is not None
            taxonomy_text_embedding_index, cache_status = (
                self._load_required_taxonomy_embedding_index(path_taxonomy_store)
            )
            taxonomy_metrics.update(cache_status.metrics)
            if cache_status.status is StageStatus.FAILED:
                return StageExecutionResult(
                    status=cache_status.status,
                    message=cache_status.message,
                    metrics=taxonomy_metrics,
                )
            assert taxonomy_text_embedding_index is not None
            cascade_identity = production_cascade_work_identity(
                classification_version=path_taxonomy_store.classification_version,
                prompt_version=path_taxonomy_store.prompt_version,
                taxonomy_fingerprint=path_taxonomy_store.classification_fingerprint,
                hierarchy_fingerprint=path_taxonomy_store.hierarchy_fingerprint,
                embedding_cache_fingerprint=taxonomy_text_embedding_index.cache_fingerprint,
            )

        from biominer.bioclip.candidate_sets import build_candidate_set_for_taxon_scope

        target_context = plan.manifest.taxon_scope.species_contexts[0]
        candidate_set = build_candidate_set_for_taxon_scope(
            plan.manifest.taxon_scope,
            target_context=target_context,
            species_candidate_path=None
            if self.request.classification_mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
            else self.species_candidate_path,
            allow_single_target_fixture=self.allow_single_target_fixture,
        )
        registry_version = plan.manifest.taxon_scope.registry_version
        stale_requeued = self.workstore.requeue_stale_claims(
            job_name=PRODUCTION_JOB_NAME,
            stage=RunStage.DETECT_OBJECTS.value,
            registry_version=registry_version,
            stale_after_seconds=int(self.request.limits.get("stale_claim_seconds") or DEFAULT_STALE_CLAIM_SECONDS),
        )
        detection_policy = self.request.vision_settings.to_detection_policy(DetectionPolicy(backend=self.object_detector.backend))
        gate_mode = "routed_visual_domain"
        score_no_detection_whole_image = False
        plan_result = enqueue_rolling_vision_work_from_source_shards(
            storage=self.storage,
            workstore=self.workstore,
            job_name=PRODUCTION_JOB_NAME,
            registry_version=registry_version,
            run_id=plan.manifest.run_id,
            source_stage=RunStage.POLL_FLICKR.value,
            vision_stage=RunStage.DETECT_OBJECTS.value,
            vision_batch_rows=self.request.vision_settings.parquet_part_rows,
            detector={
                "backend": self.object_detector.backend,
                "model_id": self.object_detector.model_id,
                "model_version": self.object_detector.model_version,
                "checkpoint": self.object_detector.checkpoint,
                "prompt_classes": list(
                    getattr(self.object_detector, "prompt_classes", ())
                ),
                "prompt_set_fingerprint": str(
                    getattr(self.object_detector, "prompt_set_fingerprint", "")
                    or ""
                ),
            },
            vision_settings=self.request.vision_settings,
            bioclip_gate_mode=gate_mode,
            score_no_detection_whole_image=score_no_detection_whole_image,
            supported_comparison_routes=PRODUCTION_BIOCLIP_COMPARISON_ROUTES,
            bioclip_model={
                "model_id": self.object_scorer.model_id,
                "model_version": self.object_scorer.model_version,
                "checkpoint": self.object_scorer.model_checkpoint,
            },
            candidate_set_id=candidate_set.candidate_set_id,
            classification_mode=self.request.classification_mode,
            taxonomy_table_version=_taxonomy_manifest_value(path_taxonomy_store, "classification_version"),
            taxonomy_prompt_variant_version=_taxonomy_manifest_value(path_taxonomy_store, "prompt_version"),
            species_first_pass_top_k=self.request.species_first_pass_top_k,
            species_rerank_top_k=self.request.species_rerank_top_k,
            cascade_identity=cascade_identity,
            limit=int(self.request.limits.get("records") or 0) or None,
        )
        claimed = self.workstore.claim_next_batch(
            self.request.worker_id,
            _cloud_rolling_vision_claim_limit(self.request.limits),
            job_name=PRODUCTION_JOB_NAME,
            stage=RunStage.DETECT_OBJECTS.value,
            registry_version=registry_version,
        )
        if not claimed:
            return StageExecutionResult(
                metrics={
                    **taxonomy_metrics,
                    "rolling_vision_work_items_enqueued": plan_result.enqueued_work_items,
                    "duplicate_rolling_vision_work_items": plan_result.duplicate_work_items,
                    "rolling_vision_batches_planned": plan_result.batches_planned,
                    "source_records_seen": plan_result.source_records_seen,
                    "source_record_shards": plan_result.source_shards_seen,
                    "work_items_claimed": 0,
                    "workstore_work_items_completed": 0,
                    "workstore_stale_claims_requeued": stale_requeued,
                    "vision_worker": "rolling",
                },
                outputs={"workstore_stage": RunStage.DETECT_OBJECTS.value},
            )

        committed_outputs: list[dict[str, str]] = []
        accumulated_metrics: dict[str, int] = {
            "records_seen": 0,
            "images_loaded": 0,
            "image_failures": 0,
            "detections_written": 0,
            "crops_created": 0,
            "score_inputs": 0,
            "objects_scored": 0,
            "whole_images_scored": 0,
            "detector_crops_scored": 0,
            "segmentation_crops_scored": 0,
            "object_evidence_rows": 0,
            "photo_summary_rows": 0,
            "parquet_parts_written": 0,
            "parquet_parts_reused": 0,
            "detector_batch_retries": 0,
            "bioclip_batch_retries": 0,
        }
        gate_policy = BioClipGatePolicy(
            mode=gate_mode,
            score_no_detection_whole_image=score_no_detection_whole_image,
            supported_comparison_routes=PRODUCTION_BIOCLIP_COMPARISON_ROUTES,
        )

        def detect(item: dict[str, Any]) -> Any:
            return detect_cloud_rolling_vision_batch(
                work_item=item,
                detector=self.object_detector,
                image_loader=self.image_loader,
                detection_policy=detection_policy,
                detector_batch_size=self.request.vision_settings.detector_batch_size,
                adaptive_detector_batching=self.request.vision_settings.adaptive_batching,
                min_detector_batch_size=self.request.vision_settings.min_detector_batch_size,
            )

        def score(detected: Any) -> Any:
            temp_dir = Path("/tmp") / "biominer_rolling_vision" / plan.manifest.run_id / detected.batch_id
            return score_cloud_rolling_detection_batch(
                batch=detected,
                image_loader=self.image_loader,
                species_context=target_context,
                candidate_set=candidate_set,
                scorer=self.object_scorer,
                bioclip_gate_policy=gate_policy,
                temp_dir=temp_dir,
                crop_padding_ratio=self.request.vision_settings.crop_padding_ratio,
                crop_target_px=self.request.vision_settings.crop_target_px,
                bioclip_batch_size=self.request.vision_settings.crop_batch_size,
                adaptive_bioclip_batching=self.request.vision_settings.adaptive_batching,
                min_bioclip_batch_size=self.request.vision_settings.min_crop_batch_size,
                classification_mode=self.request.classification_mode,
                taxonomy_table_version=_taxonomy_manifest_value(path_taxonomy_store, "classification_version"),
                taxonomy_prompt_variant_version=_taxonomy_manifest_value(path_taxonomy_store, "prompt_version"),
                rank_beam_width=self.request.rank_beam_width,
                species_first_pass_top_k=self.request.species_first_pass_top_k,
                species_rerank_top_k=self.request.species_rerank_top_k,
                path_taxonomy_store=path_taxonomy_store,
                taxonomy_text_embedding_index=taxonomy_text_embedding_index,
                cascade_identity=cascade_identity,
            )

        def commit(item: dict[str, Any], batch_result: Any) -> Any:
            batch_id = rolling_vision_batch_id(item)
            part_id = rolling_vision_part_id(item)
            commit_result = commit_rolling_vision_batch_shards(
                    storage=self.storage,
                    workstore=self.workstore,
                    job_name=PRODUCTION_JOB_NAME,
                    registry_version=registry_version,
                    run_id=plan.manifest.run_id,
                    worker_id=self.request.worker_id,
                    base_prefix=plan.artifact_uris.staging_uri,
                    work_key=str(item["work_key"]),
                    batch_id=batch_id,
                    part_id=part_id,
                    frames=batch_result.frames,
                    compression=self.request.vision_settings.parquet_compression,
                    metadata={
                        "vision_worker": "rolling",
                        "claimed_work_items": 1,
                        "source_record_count": batch_result.metrics.get("records_seen"),
                    },
                )
            if self.request.vision_settings.delete_images_after_commit:
                batch_result.cleanup_after_commit()
            committed_outputs.append(commit_result.output_uris)
            for key in accumulated_metrics:
                accumulated_metrics[key] += int(batch_result.metrics.get(key, 0) or 0)
            accumulated_metrics["parquet_parts_written"] += commit_result.parts_written
            accumulated_metrics["parquet_parts_reused"] += commit_result.parts_reused
            return commit_result

        try:
            pipeline_result = run_bounded_cloud_rolling_pipeline(claimed, detect=detect, score=score, commit=commit)
        except RollingVisionPipelineError as exc:
            completed_keys = {
                str(item.get("work_key") or "") for item in claimed[: len(committed_outputs)]
            }
            for item in claimed:
                work_key = str(item.get("work_key") or "")
                if work_key not in completed_keys:
                    self.workstore.mark_failed(work_key, str(exc) or exc.__class__.__name__)
            return StageExecutionResult(status=StageStatus.FAILED, message=f"rolling_vision_failed: {exc}")
        completed = pipeline_result.batches_committed

        outputs_by_artifact = {
            artifact: ",".join(output[artifact] for output in committed_outputs if output.get(artifact))
            for artifact in ROLLING_VISION_ARTIFACT_ORDER
        }
        return StageExecutionResult(
            metrics={
                **taxonomy_metrics,
                **accumulated_metrics,
                "rolling_vision_work_items_enqueued": plan_result.enqueued_work_items,
                "duplicate_rolling_vision_work_items": plan_result.duplicate_work_items,
                "rolling_vision_batches_planned": plan_result.batches_planned,
                "rolling_vision_batches_claimed": len(claimed),
                "rolling_vision_batches_committed": completed,
                "rolling_vision_buffer_capacity": pipeline_result.buffer_capacity,
                "source_record_shards": plan_result.source_shards_seen,
                "work_items_claimed": len(claimed),
                "workstore_work_items_completed": completed,
                "workstore_stale_claims_requeued": stale_requeued,
                "vision_worker": "rolling",
                "bioclip_gate_mode": gate_mode,
                "score_no_detection_whole_image": score_no_detection_whole_image,
                "supported_bioclip_comparison_routes": list(
                    PRODUCTION_BIOCLIP_COMPARISON_ROUTES
                ),
                "parquet_part_count": completed,
                "parquet_part_rows": accumulated_metrics["detections_written"],
                "parquet_compression": self.request.vision_settings.parquet_compression,
                "detection_shards": completed,
                "score_input_shards": completed,
                "score_shards": completed,
                "object_evidence_shards": completed,
                "photo_summary_shards": completed,
            },
            outputs={
                "object_detections": outputs_by_artifact["object_detections"],
                "bioclip_score_inputs": outputs_by_artifact["bioclip_score_inputs"],
                "object_scores": outputs_by_artifact["object_bioclip_scores"],
                "object_evidence": outputs_by_artifact["object_evidence_joined"],
                "photo_summary": outputs_by_artifact["photo_evidence_summary"],
                "image_batch_manifest": outputs_by_artifact["image_batch_manifest"],
            },
        )

    def _run_score_bioclip_stage(self, plan: ProductionRunPlan) -> StageExecutionResult:
        if is_cloud_uri(self.request.output_root):
            if self.storage is None:
                return StageExecutionResult(status=StageStatus.FAILED, message="storage_backend_required_for_score_bioclip")
            if self.workstore is None:
                return StageExecutionResult(status=StageStatus.FAILED, message="workstore_required_for_score_bioclip")
            score_uris = _cloud_stage_shard_uris(self.workstore, plan, RunStage.SCORE_BIOCLIP.value)
            if not score_uris:
                return StageExecutionResult(status=StageStatus.FAILED, message="missing_score_inputs: object_bioclip_scores")
            frame = _read_cloud_stage_frame(self.storage, self.workstore, plan, RunStage.SCORE_BIOCLIP.value)
            row_count = frame.height
            return StageExecutionResult(
                metrics={
                    **_visual_classification_config_metrics_from_paths(self.request, species_candidate_path=self.species_candidate_path),
                    "records_seen": row_count,
                    "detections_seen": row_count,
                    "crops_scored": row_count,
                    "objects_scored": row_count,
                    "whole_images_scored": _mode_row_count(frame, "whole_image"),
                    "detector_crops_scored": _mode_row_count(frame, "detector_crop"),
                    "segmentation_crops_scored": _mode_row_count(frame, "detector_crop_segmentation"),
                    "primary_visual_classifier": PRIMARY_VISUAL_CLASSIFIER,
                    "visual_modes_requested": [PRODUCTION_VISUAL_MODE],
                    "visual_modes_scored": [mode for mode in OBJECT_VISUAL_MODES if _mode_row_count(frame, mode) > 0],
                    "score_shards": len(score_uris),
                    "score_batches_written": 0,
                    "parquet_parts_written": 0,
                    "parquet_parts_reused": len(score_uris),
                    "parquet_part_count": len(score_uris),
                    "parquet_part_rows": row_count,
                    "parquet_compression": self.request.vision_settings.parquet_compression,
                    "score_part_count": len(score_uris),
                    "score_part_rows": row_count,
                    "vision_worker": "rolling",
                    "rolling_vision_shards_reused": True,
                    **object_score_audit_metrics(frame),
                },
                outputs={"object_scores": ",".join(score_uris)},
            )
        path_taxonomy_store: PathTaxonomyStore | None = None
        taxonomy_text_embedding_index: TaxonomyTextEmbeddingIndex | None = None
        taxonomy_metrics: dict[str, Any] = {}
        if self.request.classification_mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION:
            path_taxonomy_store, taxonomy_status = self._load_valid_hierarchical_taxonomy_store()
            if taxonomy_status.status is StageStatus.FAILED:
                return taxonomy_status
            taxonomy_metrics = dict(taxonomy_status.metrics)
        missing = _missing_paths(plan.paths.source_records_path, plan.paths.object_detections_path)
        if missing:
            return StageExecutionResult(
                status=StageStatus.FAILED,
                message="missing_score_inputs: " + ", ".join(missing),
                metrics=taxonomy_metrics,
            )
        if self.object_scorer is None:
            return StageExecutionResult(
                status=StageStatus.FAILED,
                message="bioclip_runtime_required_for_score_bioclip",
                metrics=taxonomy_metrics,
            )
        if path_taxonomy_store is not None:
            taxonomy_text_embedding_index, cache_status = (
                self._load_required_taxonomy_embedding_index(path_taxonomy_store)
            )
            taxonomy_metrics.update(cache_status.metrics)
            if cache_status.status is StageStatus.FAILED:
                return StageExecutionResult(
                    status=cache_status.status,
                    message=cache_status.message,
                    metrics=taxonomy_metrics,
                )
        import polars as pl
        from biominer.bioclip.candidate_sets import build_candidate_set_for_taxon_scope

        plan.paths.ensure_directories()
        canonical = pl.read_parquet(plan.paths.source_records_path)
        detections = pl.read_parquet(plan.paths.object_detections_path)
        target_context = plan.manifest.taxon_scope.species_contexts[0]
        candidate_set = build_candidate_set_for_taxon_scope(
            plan.manifest.taxon_scope,
            target_context=target_context,
            species_candidate_path=None
            if self.request.classification_mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
            else self.species_candidate_path,
            records=canonical.to_dicts(),
            allow_single_target_fixture=self.allow_single_target_fixture,
        )
        mode_output_dir = plan.paths.staging_dir / "object_bioclip_scores_by_mode"
        frame, metrics = _score_primary_visual_mode(
            canonical_records=canonical,
            detections=detections,
            species_context=target_context,
            candidate_set=candidate_set,
            scorer=self.object_scorer,
            output_dir=mode_output_dir,
            bioclip_batch_size=self.request.vision_settings.crop_batch_size,
            adaptive_batching=self.request.vision_settings.adaptive_batching,
            min_bioclip_batch_size=self.request.vision_settings.min_crop_batch_size,
            classification_mode=self.request.classification_mode,
            rank_beam_width=self.request.rank_beam_width,
            species_first_pass_top_k=self.request.species_first_pass_top_k,
            species_rerank_top_k=self.request.species_rerank_top_k,
            species_report_top_k=self.request.species_report_top_k,
            path_taxonomy_store=path_taxonomy_store,
            taxonomy_text_embedding_index=taxonomy_text_embedding_index,
        )
        write_parquet(frame, plan.paths.object_scores_path)
        metrics.update(_visual_classification_config_metrics_from_paths(self.request, species_candidate_path=self.species_candidate_path))
        metrics.update(taxonomy_metrics)
        return StageExecutionResult(
            metrics=metrics,
            outputs={"object_scores": str(plan.paths.object_scores_path)},
        )

    def _validate_hierarchical_taxonomy_table(self) -> StageExecutionResult:
        _store, result = self._load_valid_hierarchical_taxonomy_store()
        return result

    def _load_valid_hierarchical_taxonomy_store(
        self,
    ) -> tuple[PathTaxonomyStore | None, StageExecutionResult]:
        base_metrics = _visual_classification_config_metrics_from_paths(self.request, species_candidate_path=self.species_candidate_path)
        if self.species_candidate_path is None:
            return (
                None,
                StageExecutionResult(
                    status=StageStatus.FAILED,
                    message="missing_taxonomy_candidate_table",
                    metrics={**base_metrics, "registry_taxonomy_status": "missing", "taxonomy_candidate_table_status": "missing"},
                ),
            )
        try:
            store = self._read_butterfly_taxonomy_store(self.species_candidate_path)
        except FileNotFoundError as exc:
            return (
                None,
                StageExecutionResult(
                    status=StageStatus.FAILED,
                    message=f"missing_taxonomy_candidate_table: {exc}",
                    metrics={**base_metrics, "registry_taxonomy_status": "missing", "taxonomy_candidate_table_status": "missing"},
                ),
            )
        except ValueError as exc:
            return (
                None,
                StageExecutionResult(
                    status=StageStatus.FAILED,
                    message=f"invalid_taxonomy_candidate_table: {exc}",
                    metrics={**base_metrics, "registry_taxonomy_status": "invalid", "taxonomy_candidate_table_status": "invalid"},
                ),
            )
        findings = store.validation_findings()
        fatal = [finding for finding in findings if finding.get("severity") == "fatal"]
        if fatal:
            return (
                None,
                StageExecutionResult(
                    status=StageStatus.FAILED,
                    message="invalid_taxonomy_candidate_table: " + ", ".join(str(finding.get("code")) for finding in fatal),
                    metrics={
                        **base_metrics,
                        **_butterfly_taxonomy_store_metrics(store, registry_taxonomy_status="invalid"),
                        "registry_taxonomy_fatal_findings": len(fatal),
                    },
                ),
            )
        return (
            store,
            StageExecutionResult(
                metrics={
                    **base_metrics,
                    **_butterfly_taxonomy_store_metrics(store, registry_taxonomy_status="valid"),
                    "registry_taxonomy_fatal_findings": 0,
                    "registry_taxonomy_warning_findings": sum(1 for finding in findings if finding.get("severity") == "warning"),
                }
            ),
        )

    def _read_butterfly_taxonomy_store(self, location: str | Path) -> PathTaxonomyStore:
        if is_cloud_uri(str(location)):
            if self.storage is None:
                raise ValueError("storage_backend_required_for_registry_taxonomy")
            paths_uri = join_uri(str(location), "species_paths.parquet")
            if not self.storage.exists(paths_uri):
                raise FileNotFoundError(paths_uri)
            return PathTaxonomyStore.from_species_paths(self.storage.read_parquet(paths_uri))
        v3_root = Path(location).parent if Path(location).suffix == ".parquet" else Path(location)
        return PathTaxonomyStore.read(v3_root)

    def _load_required_taxonomy_embedding_index(
        self,
        taxonomy_store: PathTaxonomyStore,
    ) -> tuple[TaxonomyTextEmbeddingIndex | None, StageExecutionResult]:
        location = self.request.taxonomy_text_embedding_cache
        if not location:
            return (
                None,
                StageExecutionResult(
                    status=StageStatus.FAILED,
                    message=(
                        "missing_taxonomy_text_embedding_cache: hierarchical production "
                        "requires an embedding cache built from registry species_paths.parquet"
                    ),
                    metrics={
                        "taxonomy_text_embedding_cache_rows": 0,
                        "taxonomy_text_embedding_cache_status": "missing",
                    },
                ),
            )
        if self.object_scorer is None:
            return (
                None,
                StageExecutionResult(
                    status=StageStatus.FAILED,
                    message="bioclip_runtime_required_for_taxonomy_text_embedding_cache",
                ),
            )
        frame = None
        try:
            frame = self._read_optional_parquet_location(location)
            index = TaxonomyTextEmbeddingIndex.from_frame(
                frame,
                taxonomy_store=taxonomy_store,
                model_id=str(self.object_scorer.model_id or ""),
                model_checkpoint=str(self.object_scorer.model_checkpoint or ""),
            )
        except FileNotFoundError as exc:
            return (
                None,
                StageExecutionResult(
                    status=StageStatus.FAILED,
                    message=f"missing_taxonomy_text_embedding_cache: {exc}",
                    metrics={
                        "taxonomy_text_embedding_cache_rows": 0,
                        "taxonomy_text_embedding_cache_status": "missing",
                    },
                ),
            )
        except ValueError as exc:
            return (
                None,
                StageExecutionResult(
                    status=StageStatus.FAILED,
                    message=f"invalid_taxonomy_text_embedding_cache: {exc}",
                    metrics={
                        "taxonomy_text_embedding_cache_rows": frame.height if frame is not None else 0,
                        "taxonomy_text_embedding_cache_status": "invalid",
                    },
                ),
            )
        return (
            index,
            StageExecutionResult(
                metrics={
                    "taxonomy_text_embedding_cache_rows": frame.height,
                    "taxonomy_text_embedding_cache_status": "validated",
                    "taxonomy_text_embedding_cache_fingerprint": index.cache_fingerprint,
                    "taxonomy_text_embedding_dimension": index.embedding_dim,
                    "taxonomy_text_embedding_label_count": index.label_count,
                }
            ),
        )

    def _read_optional_parquet_location(self, location: str | Path) -> Any:
        if is_cloud_uri(str(location)):
            if self.storage is None:
                raise ValueError("storage_backend_required_for_cloud_parquet")
            if not self.storage.exists(str(location)):
                raise FileNotFoundError(str(location))
            return self.storage.read_parquet(str(location))
        import polars as pl

        path = Path(location)
        if not path.exists():
            raise FileNotFoundError(path)
        return pl.read_parquet(path)

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
            if self.workstore is None:
                return StageExecutionResult(status=StageStatus.FAILED, message="workstore_required_for_join_evidence")
            joined_uris = _cloud_stage_shard_uris(self.workstore, plan, RunStage.JOIN_EVIDENCE.value)
            if joined_uris:
                joined = _read_cloud_stage_frame(self.storage, self.workstore, plan, RunStage.JOIN_EVIDENCE.value)
                return StageExecutionResult(
                    metrics={
                        "object_evidence_rows": joined.height,
                        "object_occurrence_bin_counts": _value_counts(joined, "occurrence_bin"),
                        "object_evidence_shards": len(joined_uris),
                        "parquet_parts_reused": len(joined_uris),
                        "vision_worker": "rolling",
                        "rolling_vision_shards_reused": True,
                    },
                    outputs={"object_evidence": ",".join(joined_uris)},
                )
            missing = []
            if not _cloud_stage_shard_uris(self.workstore, plan, RunStage.POLL_FLICKR.value):
                missing.append("source_records")
            if not _cloud_stage_shard_uris(self.workstore, plan, RunStage.DETECT_OBJECTS.value):
                missing.append("object_detections")
            if not _cloud_stage_shard_uris(self.workstore, plan, RunStage.SCORE_BIOCLIP.value):
                missing.append("object_scores")
            if missing:
                return StageExecutionResult(status=StageStatus.FAILED, message="missing_join_inputs: " + ", ".join(missing))
            result = join_object_evidence_from_cloud_shards(
                storage=self.storage,
                workstore=self.workstore,
                job_name=PRODUCTION_JOB_NAME,
                registry_version=plan.manifest.taxon_scope.registry_version,
                run_id=plan.manifest.run_id,
                source_stage=RunStage.POLL_FLICKR.value,
                detection_stage=RunStage.DETECT_OBJECTS.value,
                score_stage=RunStage.SCORE_BIOCLIP.value,
            )
            object_evidence_uri = self.storage.write_parquet_shard(
                build_evidence_shard_uri(
                    plan.artifact_uris.staging_uri,
                    stage=RunStage.JOIN_EVIDENCE.value,
                    run_id=plan.manifest.run_id,
                    worker_id=self.request.worker_id,
                    batch_id=join_evidence_batch_id(result),
                ),
                result.frame,
            )
            self.workstore.register_shard(
                job_name=PRODUCTION_JOB_NAME,
                registry_version=plan.manifest.taxon_scope.registry_version,
                stage=RunStage.JOIN_EVIDENCE.value,
                run_id=plan.manifest.run_id,
                worker_id=self.request.worker_id,
                uri=object_evidence_uri,
                checksum=None,
                row_count=result.frame.height,
                metadata={
                    "source_shards_seen": result.source_shards_seen,
                    "detection_shards_seen": result.detection_shards_seen,
                    "score_shards_seen": result.score_shards_seen,
                },
            )
            metrics = {
                "object_evidence_rows": result.frame.height,
                "object_occurrence_bin_counts": _value_counts(result.frame, "occurrence_bin"),
                "source_record_shards": result.source_shards_seen,
                "detection_shards": result.detection_shards_seen,
                "score_shards": result.score_shards_seen,
                "source_records_seen": result.source_records_seen,
                "detections_seen": result.detections_seen,
                "scores_seen": result.scores_seen,
                "object_evidence_shards": 1,
            }
            return StageExecutionResult(
                metrics=metrics,
                outputs={
                    "object_evidence": object_evidence_uri,
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
            if self.workstore is None:
                return StageExecutionResult(status=StageStatus.FAILED, message="workstore_required_for_summarize")
            outputs: dict[str, str] = {}
            summary_result = None
            existing_photo_summary_uris = _cloud_stage_shard_uris(self.workstore, plan, "photo_summary")
            should_materialize_summary = not existing_photo_summary_uris
            if should_materialize_summary and _cloud_stage_shard_uris(self.workstore, plan, RunStage.JOIN_EVIDENCE.value):
                summary_result = summarize_photo_evidence_from_cloud_shards(
                    storage=self.storage,
                    workstore=self.workstore,
                    job_name=PRODUCTION_JOB_NAME,
                    registry_version=plan.manifest.taxon_scope.registry_version,
                    run_id=plan.manifest.run_id,
                    joined_stage=RunStage.JOIN_EVIDENCE.value,
                    species_context=plan.manifest.taxon_scope.species_contexts[0],
                )
                photo_summary_uri = self.storage.write_parquet_shard(
                    build_evidence_shard_uri(
                        plan.artifact_uris.staging_uri,
                        stage="photo_summary",
                        run_id=plan.manifest.run_id,
                        worker_id=self.request.worker_id,
                        batch_id=photo_summary_batch_id(summary_result),
                    ),
                    summary_result.frame,
                )
                self.workstore.register_shard(
                    job_name=PRODUCTION_JOB_NAME,
                    registry_version=plan.manifest.taxon_scope.registry_version,
                    stage="photo_summary",
                    run_id=plan.manifest.run_id,
                    worker_id=self.request.worker_id,
                    uri=photo_summary_uri,
                    checksum=None,
                    row_count=summary_result.frame.height,
                    metadata={
                        "joined_shards_seen": summary_result.joined_shards_seen,
                        "object_evidence_rows_seen": summary_result.object_evidence_rows_seen,
                    },
                )
                outputs["photo_summary"] = photo_summary_uri
            if existing_photo_summary_uris and "photo_summary" not in outputs:
                outputs["photo_summary"] = ",".join(existing_photo_summary_uris)
            if not _cloud_stage_shard_uris(self.workstore, plan, "photo_summary"):
                return StageExecutionResult(status=StageStatus.FAILED, message="missing_summary_inputs: object_evidence, photo_summary")
            summary_shards = self.workstore.list_committed_shards(
                job_name=PRODUCTION_JOB_NAME,
                stage="photo_summary",
                registry_version=plan.manifest.taxon_scope.registry_version,
                run_id=plan.manifest.run_id,
            )
            joined = _read_cloud_stage_frame(self.storage, self.workstore, plan, RunStage.JOIN_EVIDENCE.value)
            photo_summary = (
                summary_result.frame
                if summary_result is not None
                else _read_cloud_stage_frame(self.storage, self.workstore, plan, "photo_summary")
            )
            review_queue, review_queue_mode = _build_production_review_queue(joined=joined, photo_summary=photo_summary)
            visual_qa_findings = build_visual_qa_findings(object_evidence=joined, photo_summary=photo_summary)
            queue_result = CloudReviewQueueResult(
                frame=review_queue,
                summary_shards=tuple(summary_shards),
                photo_summary_rows_seen=photo_summary.height,
                photo_occurrence_bin_counts=_value_counts(photo_summary, "photo_occurrence_bin"),
            )
            review_queue_uri = self.storage.write_parquet_shard(
                build_evidence_shard_uri(
                    plan.artifact_uris.staging_uri,
                    stage="review_queue",
                    run_id=plan.manifest.run_id,
                    worker_id=self.request.worker_id,
                    batch_id=review_queue_batch_id(queue_result),
                ),
                queue_result.frame,
            )
            self.workstore.register_shard(
                job_name=PRODUCTION_JOB_NAME,
                registry_version=plan.manifest.taxon_scope.registry_version,
                stage="review_queue",
                run_id=plan.manifest.run_id,
                worker_id=self.request.worker_id,
                uri=review_queue_uri,
                checksum=None,
                row_count=queue_result.frame.height,
                metadata={
                    "summary_shards_seen": queue_result.summary_shards_seen,
                    "review_queue_mode": review_queue_mode,
                },
            )
            visual_qa_uri = self.storage.write_parquet_shard(
                plan.artifact_uris.visual_qa_findings_uri,
                visual_qa_findings,
            )
            metrics = evidence_count_metrics(joined, photo_summary)
            metrics.update(
                {
                    "photo_occurrence_bin_counts": queue_result.photo_occurrence_bin_counts,
                    "object_occurrence_bin_counts": _value_counts(joined, "occurrence_bin"),
                    "summary_shards": queue_result.summary_shards_seen,
                    "review_queue_shards": 1,
                }
            )
            metrics.update(_review_queue_metrics(queue_result.frame, review_queue_mode=review_queue_mode))
            metrics.update(_visual_qa_metrics(visual_qa_findings))
            if summary_result is not None:
                metrics.update(
                    {
                        "object_evidence_rows": summary_result.object_evidence_rows_seen,
                        "object_occurrence_bin_counts": summary_result.object_occurrence_bin_counts,
                        "joined_evidence_shards": summary_result.joined_shards_seen,
                        "photo_summary_shards": 1,
                    }
                )
            metrics_uri = self.storage.write_json(plan.artifact_uris.metrics_uri, metrics)
            vision_stage_metrics = build_vision_stage_metrics(
                detections=_read_cloud_stage_frame(self.storage, self.workstore, plan, RunStage.DETECT_OBJECTS.value),
                scores=_read_cloud_stage_frame(self.storage, self.workstore, plan, RunStage.SCORE_BIOCLIP.value),
                joined=joined,
                photo_summary=photo_summary,
                stage_metrics=plan.manifest.metrics,
                detection_policy=self.request.vision_settings.to_detection_policy(DetectionPolicy()),
            )
            vision_stage_metrics_uri = self.storage.write_json(plan.artifact_uris.vision_stage_metrics_uri, vision_stage_metrics)
            outputs.update(
                {
                    "metrics": metrics_uri,
                    "review_queue": review_queue_uri,
                    "visual_qa_findings": visual_qa_uri,
                    "vision_stage_metrics": vision_stage_metrics_uri,
                }
            )
            return StageExecutionResult(metrics=metrics, outputs=outputs)
        missing = _missing_paths(plan.paths.object_evidence_path)
        if missing:
            return StageExecutionResult(status=StageStatus.FAILED, message="missing_summary_inputs: " + ", ".join(missing))
        import json
        import polars as pl

        joined = pl.read_parquet(plan.paths.object_evidence_path)
        photo_summary = _read_optional_parquet(plan.paths.photo_summary_path)
        review_queue, review_queue_mode = _build_production_review_queue(joined=joined, photo_summary=photo_summary)
        visual_qa_findings = build_visual_qa_findings(object_evidence=joined, photo_summary=photo_summary)
        metrics = evidence_count_metrics(joined, photo_summary)
        metrics.update(_review_queue_metrics(review_queue, review_queue_mode=review_queue_mode))
        metrics.update(_visual_qa_metrics(visual_qa_findings))
        plan.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        plan.paths.metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        vision_stage_metrics = build_vision_stage_metrics(
            detections=_read_optional_parquet(plan.paths.object_detections_path),
            scores=_read_optional_parquet(plan.paths.object_scores_path),
            joined=joined,
            photo_summary=photo_summary,
            stage_metrics=plan.manifest.metrics,
            detection_policy=self.request.vision_settings.to_detection_policy(DetectionPolicy()),
        )
        vision_report_paths = write_vision_stage_reports(vision_stage_metrics, plan.paths.reports_dir)
        write_parquet(review_queue, plan.paths.review_queue_path)
        write_parquet(visual_qa_findings, plan.paths.visual_qa_findings_path)
        return StageExecutionResult(
            metrics=metrics,
            outputs={
                "metrics": str(plan.paths.metrics_path),
                "vision_stage_metrics": str(vision_report_paths["metrics"]),
                "vision_stage_summary": str(vision_report_paths["summary"]),
                "review_queue": str(plan.paths.review_queue_path),
                "visual_qa_findings": str(plan.paths.visual_qa_findings_path),
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


def _cloud_rolling_vision_claim_limit(limits: dict[str, int]) -> int:
    batch_limit = int(limits.get("rolling_vision_batches") or 0)
    worker_limit = int(limits.get("workers") or 0)
    candidates = [value for value in (batch_limit, worker_limit) if value > 0]
    return min(candidates) if candidates else 4


def _cloud_source_records_available(storage: CloudStorage, workstore: WorkStore | None, plan: ProductionRunPlan) -> bool:
    if storage.exists(plan.artifact_uris.source_records_uri):
        return True
    return bool(_cloud_source_record_shard_uris(workstore, plan))


def _cloud_source_record_shard_uris(workstore: WorkStore | None, plan: ProductionRunPlan) -> list[str]:
    return _cloud_stage_shard_uris(workstore, plan, RunStage.POLL_FLICKR.value)


def _cloud_stage_shard_uris(workstore: WorkStore | None, plan: ProductionRunPlan, stage: str) -> list[str]:
    if workstore is None:
        return []
    shards = workstore.list_committed_shards(
        job_name=PRODUCTION_JOB_NAME,
        stage=stage,
        registry_version=plan.manifest.taxon_scope.registry_version,
        run_id=plan.manifest.run_id,
    )
    return [str(shard["uri"]) for shard in shards]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_cloud_stage_frame(storage: CloudStorage, workstore: WorkStore, plan: ProductionRunPlan, stage: str) -> Any:
    import polars as pl

    uris = _cloud_stage_shard_uris(workstore, plan, stage)
    scans = [storage.scan_parquet(uri) for uri in uris]
    return pl.concat(scans, how="diagonal_relaxed").collect(engine="streaming") if scans else pl.DataFrame()


def _read_optional_parquet(path: Path) -> Any:
    import polars as pl

    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


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


def _score_primary_visual_mode(
    *,
    canonical_records: Any,
    detections: Any,
    species_context: Any,
    candidate_set: Any,
    scorer: Any,
    output_dir: Path,
    bioclip_batch_size: int = 24,
    adaptive_batching: bool = False,
    min_bioclip_batch_size: int = 1,
    classification_mode: ClassificationMode = DEFAULT_CLASSIFICATION_MODE,
    rank_beam_width: int = DEFAULT_RANK_BEAM_WIDTH,
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K,
    species_report_top_k: int = DEFAULT_SPECIES_REPORT_TOP_K,
    path_taxonomy_store: PathTaxonomyStore | None = None,
    taxonomy_text_embedding_index: TaxonomyTextEmbeddingIndex | None = None,
) -> tuple[Any, dict[str, Any]]:
    from biominer.bioclip.object_runner import screen_object_detections

    (
        _beam_strategy,
        rank_beam_width,
        species_first_pass_top_k,
        species_rerank_top_k,
        _species_report_top_k,
    ) = validate_production_cascade_settings(
        beam_strategy=GLOBAL_RANK_TOP_K_BEAM_STRATEGY,
        rank_beam_width=rank_beam_width,
        species_first_pass_top_k=species_first_pass_top_k,
        species_rerank_top_k=species_rerank_top_k,
        species_report_top_k=species_report_top_k,
    )
    result = screen_object_detections(
        canonical_records=canonical_records,
        detections=detections,
        species_context=species_context,
        candidate_set=candidate_set,
        scorer=scorer,
        output_path=output_dir / "object_bioclip_scores_detector_crop.parquet",
        ablation_mode=PRODUCTION_VISUAL_MODE,
        bioclip_gate_policy=BioClipGatePolicy(
            supported_comparison_routes=PRODUCTION_BIOCLIP_COMPARISON_ROUTES
        ),
        bioclip_batch_size=bioclip_batch_size,
        adaptive_batching=adaptive_batching,
        min_bioclip_batch_size=min_bioclip_batch_size,
        classification_mode=classification_mode,
        rank_beam_width=rank_beam_width,
        species_first_pass_top_k=species_first_pass_top_k,
        species_rerank_top_k=species_rerank_top_k,
        path_taxonomy_store=path_taxonomy_store,
        taxonomy_text_embedding_index=taxonomy_text_embedding_index,
    )
    frame = result.frame
    scored = [PRODUCTION_VISUAL_MODE] if result.crops_scored else []
    return frame, {
        "records_seen": result.records_seen,
        "detections_seen": result.detections_seen,
        "crops_scored": result.crops_scored,
        "objects_scored": result.crops_scored,
        "whole_images_scored": 0,
        "detector_crops_scored": result.crops_scored,
        "segmentation_crops_scored": 0,
        "primary_visual_classifier": PRIMARY_VISUAL_CLASSIFIER,
        "visual_modes_requested": [PRODUCTION_VISUAL_MODE],
        "visual_modes_scored": scored,
        "visual_mode_status_by_mode": {PRODUCTION_VISUAL_MODE: result.visual_mode_status},
        "segmentation_status_by_mode": {},
        "segmentation_unavailable_count_by_mode": {},
        "segmentation_unavailable_reason_by_mode": {},
        "segmentation_unavailable_count": 0,
        "segmentation_unavailable_reason": None,
        "adaptive_batching_enabled": result.adaptive_batching_enabled,
        "bioclip_batch_retries": result.bioclip_batch_retries,
        "bioclip_batch_size_initial": result.bioclip_batch_size_initial,
        "bioclip_batch_size_final": result.bioclip_batch_size_final,
        "bioclip_batch_size_min": result.bioclip_batch_size_min,
        "score_batches_written": result.score_batches_written,
        **object_score_audit_metrics(frame),
    }


def _visual_classification_config_metrics_from_paths(
    request: ProductionRunRequest,
    *,
    species_candidate_path: str | Path | None,
) -> dict[str, Any]:
    return _visual_classification_config_metrics(request=request, species_candidate_path=species_candidate_path)


def _visual_classification_config_metrics(
    request: ProductionRunRequest,
    *,
    species_candidate_path: str | Path | None = None,
) -> dict[str, Any]:
    candidate_table = (
        str(species_candidate_path)
        if species_candidate_path is not None
        else request.registry_dir
    )
    return {
        "classification_mode": request.classification_mode,
        "registry_taxonomy": candidate_table,
        "taxonomy_candidate_table": candidate_table,
        "beam_strategy": request.beam_strategy,
        "rank_beam_width": request.rank_beam_width,
        "species_first_pass_top_k": request.species_first_pass_top_k,
        "species_rerank_top_k": request.species_rerank_top_k,
        "species_report_top_k": request.species_report_top_k,
    }


def _butterfly_taxonomy_store_metrics(
    store: PathTaxonomyStore,
    *,
    registry_taxonomy_status: str,
) -> dict[str, Any]:
    manifest = dict(store.manifest or {})
    family_count = store.rank_candidates("FAMILY").height
    species_count = store.leaf_paths.filter(store.leaf_paths["enabled"]).height
    return {
        "registry_taxonomy_status": registry_taxonomy_status,
        "taxonomy_candidate_table_status": registry_taxonomy_status,
        "classification_table_version": manifest.get("classification_version"),
        "classification_prompt_variant_version": manifest.get("prompt_version"),
        "classification_registry_version": manifest.get("registry_version"),
        "classification_retrieved_at_min": _min_non_blank_frame_value(store.nodes, "retrieved_at"),
        "classification_retrieved_at_max": _max_non_blank_frame_value(store.nodes, "retrieved_at"),
        "classification_family_count": family_count,
        "classification_species_count": species_count,
        "taxonomy_family_candidate_count": family_count,
        "taxonomy_species_candidate_count": species_count,
        "classification_family_label_count": store.prompt_labels.filter(
            store.prompt_labels["enabled"] & (store.prompt_labels["rank"] == "FAMILY")
        ).height,
        "classification_species_label_count": store.prompt_labels.filter(
            store.prompt_labels["enabled"] & (store.prompt_labels["rank"] == "SPECIES")
        ).height,
        "classification_taxonomy_fingerprint": store.classification_fingerprint,
        "classification_hierarchy_fingerprint": store.hierarchy_fingerprint,
    }


def _taxonomy_manifest_value(
    store: PathTaxonomyStore | None,
    key: str,
) -> str | None:
    if store is None:
        return None
    manifest = dict(store.manifest or {})
    aliases = {
        "classification_table_version": "classification_version",
        "prompt_variant_version": "prompt_version",
    }
    value = manifest.get(key)
    if value is None:
        value = manifest.get(aliases.get(key, ""))
    return str(value) if value is not None and str(value).strip() else None


def _first_non_blank_frame_value(frame: Any, column: str) -> str | None:
    values = _non_blank_frame_values(frame, column)
    return values[0] if values else None


def _min_non_blank_frame_value(frame: Any, column: str) -> str | None:
    values = _non_blank_frame_values(frame, column)
    return min(values) if values else None


def _max_non_blank_frame_value(frame: Any, column: str) -> str | None:
    values = _non_blank_frame_values(frame, column)
    return max(values) if values else None


def _non_blank_frame_values(frame: Any, column: str) -> list[str]:
    if frame.is_empty() or column not in frame.columns:
        return []
    return [str(value) for value in frame.get_column(column).drop_nulls().to_list() if str(value or "").strip()]


def _mode_row_count(frame: Any, mode: str) -> int:
    if frame.is_empty() or "ablation_mode" not in frame.columns:
        return 0
    return frame.filter(frame["ablation_mode"] == mode).height


def _build_production_review_queue(*, joined: Any, photo_summary: Any) -> tuple[Any, str]:
    if _has_hierarchical_classification_rows(joined):
        return build_hierarchical_review_queue(object_evidence=joined, photo_summary=photo_summary), "hierarchical"
    return build_review_queue(photo_summary), "target_scope"


def _has_hierarchical_classification_rows(frame: Any) -> bool:
    if frame.is_empty() or "classification_mode" not in frame.columns:
        return False
    return any(
        str(value or "") == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
        for value in frame.get_column("classification_mode").drop_nulls().to_list()
    )


def _canonical_bioclip_model_id(value: object) -> str:
    return str(value or "").strip().removeprefix("hf-hub:")


def _review_queue_metrics(frame: Any, *, review_queue_mode: str) -> dict[str, Any]:
    return {
        "review_queue_mode": review_queue_mode,
        "review_queue_rows": frame.height,
        "high_priority_review_rows": _high_priority_review_rows(frame),
        "review_priority_counts": _value_counts(frame, "review_priority"),
        "review_queue_bin_counts": _value_counts(frame, "review_bucket"),
        "top_review_reasons": _review_reason_counts(frame),
    }


def _visual_qa_metrics(frame: Any) -> dict[str, Any]:
    severity_counts = _value_counts(frame, "severity")
    fatal_count = int(severity_counts.get("fatal", 0))
    return {
        "visual_qa_findings": frame.height,
        "visual_qa_severity_counts": severity_counts,
        "visual_qa_finding_type_counts": _value_counts(frame, "finding_type"),
        "visual_qa_fatal_count": fatal_count,
        "visual_qa_warning_count": int(severity_counts.get("warning", 0)),
        "visual_qa_info_count": int(severity_counts.get("info", 0)),
        "visual_qa_status": "failed" if fatal_count > 0 else "passed",
    }


def _high_priority_review_rows(frame: Any) -> int:
    if frame.is_empty() or "review_priority" not in frame.columns:
        return 0
    count = 0
    for value in frame.get_column("review_priority").drop_nulls().to_list():
        priority = _optional_int(value)
        if priority is not None and priority >= 80:
            count += 1
    return count


def _review_reason_counts(frame: Any) -> dict[str, int]:
    if frame.is_empty() or "review_reason" not in frame.columns:
        return {}
    counts: dict[str, int] = {}
    for value in frame.get_column("review_reason").drop_nulls().to_list():
        for reason in str(value or "").split(";"):
            reason = reason.strip()
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


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
