"""Fail-closed planning and stage governance for the adaptive production graph."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from biominer.references.readiness import ReferenceBankReadinessPermit
from biominer.run.adaptive_config import (
    AdaptiveReferenceSettings,
    AdaptiveReferenceValidationContext,
    validate_adaptive_reference_settings,
)
from biominer.run.manifest import RunManifest, utc_now_iso
from biominer.run.paths import RUN_ARTIFACT_LAYOUT_VERSION, RunArtifactUris, RunPaths
from biominer.run.stages import (
    ADAPTIVE_REFERENCE_PRODUCTION_STAGES,
    MANUAL_REVIEW_STAGES,
    RunStage,
    StageStatus,
    default_stage_records,
)
from biominer.run.support_dependencies import (
    SUPPORT_DEPENDENT_STAGES,
    SUPPORT_SCORING_MODES,
    SupportDependencyPermit,
    validate_support_readiness_dependencies,
)
from biominer.run.taxon_scope import (
    InputRank,
    TaxonScope,
    resolve_taxon_scope_from_registry,
    resolve_taxon_scope_from_registry_frames,
)
from biominer.storage.cloud import CloudStorage
from biominer.storage.paths import safe_path_component
from biominer.storage.uri import is_cloud_uri, join_uri


DEFAULT_BIOCLIP_MODEL = "imageomics/bioclip-2.5-vith14"
DEFAULT_VISION_BACKEND = "yoloe26"
ADAPTIVE_VISUAL_INPUT = "raw_full_image"


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
    reference_bank_readiness: str | Path | None = None
    reference_bank_readiness_sha256: str | None = None
    regional_candidates: str | Path | None = None
    reference_embeddings: str | Path | None = None
    classifier_artifact: str | Path | None = None
    calibrator_artifact: str | Path | None = None
    reference_admission_mode: str = "adaptive_gbif_fast_start"
    reference_source: str = "gbif"
    initial_scoring_mode: str = "provisional_reference_ranking"
    flickr_release_requires_human_review: bool = True
    statistical_reference_audit: bool = True
    strict_reference_readiness_claim: bool = False
    reference_split_uses: tuple[str, ...] = ()
    support_scoring_mode: str = "calibrated"
    worker_id: str = "local"
    stages: tuple[RunStage, ...] = ADAPTIVE_REFERENCE_PRODUCTION_STAGES
    dry_run: bool = False
    limits: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validation_context = AdaptiveReferenceValidationContext(
            strict_readiness_claim=self.strict_reference_readiness_claim,
            reference_split_uses=self.reference_split_uses,
            final_flickr_export_requested=(RunStage.FINAL_QUALITY_GATE in self.stages),
            calibrator_available=self.calibrator_artifact is not None,
        )
        adaptive = validate_adaptive_reference_settings(
            AdaptiveReferenceSettings(
                reference_admission_mode=self.reference_admission_mode,
                reference_source=self.reference_source,
                initial_scoring_mode=self.initial_scoring_mode,
                flickr_release_requires_human_review=(
                    self.flickr_release_requires_human_review
                ),
                statistical_reference_audit=self.statistical_reference_audit,
            ),
            context=validation_context,
        )
        for field_name in (
            "reference_admission_mode",
            "reference_source",
            "initial_scoring_mode",
            "flickr_release_requires_human_review",
            "statistical_reference_audit",
        ):
            object.__setattr__(self, field_name, getattr(adaptive, field_name))
        object.__setattr__(
            self,
            "reference_split_uses",
            validation_context.reference_split_uses,
        )
        scoring_mode = str(self.support_scoring_mode).strip().casefold()
        if scoring_mode not in SUPPORT_SCORING_MODES:
            raise ValueError(
                f"unsupported support_scoring_mode: {self.support_scoring_mode!r}"
            )
        object.__setattr__(self, "support_scoring_mode", scoring_mode)
        if any(stage not in ADAPTIVE_REFERENCE_PRODUCTION_STAGES for stage in self.stages):
            invalid = sorted(
                stage.value
                for stage in self.stages
                if stage not in ADAPTIVE_REFERENCE_PRODUCTION_STAGES
            )
            raise ValueError(
                "production stages must belong to the adaptive graph: "
                + ", ".join(invalid)
            )

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
                "reference_bank_readiness": (
                    str(self.request.reference_bank_readiness)
                    if self.request.reference_bank_readiness
                    else None
                ),
                "reference_bank_readiness_sha256": (
                    self.request.reference_bank_readiness_sha256
                ),
                "regional_candidates": (
                    str(self.request.regional_candidates)
                    if self.request.regional_candidates
                    else None
                ),
                "reference_embeddings": (
                    str(self.request.reference_embeddings)
                    if self.request.reference_embeddings
                    else None
                ),
                "classifier_artifact": (
                    str(self.request.classifier_artifact)
                    if self.request.classifier_artifact
                    else None
                ),
                "calibrator_artifact": (
                    str(self.request.calibrator_artifact)
                    if self.request.calibrator_artifact
                    else None
                ),
                "reference_admission_mode": self.request.reference_admission_mode,
                "reference_source": self.request.reference_source,
                "initial_scoring_mode": self.request.initial_scoring_mode,
                "flickr_release_requires_human_review": (
                    self.request.flickr_release_requires_human_review
                ),
                "statistical_reference_audit": (
                    self.request.statistical_reference_audit
                ),
                "strict_reference_readiness_claim": (
                    self.request.strict_reference_readiness_claim
                ),
                "reference_split_uses": list(self.request.reference_split_uses),
                "support_scoring_mode": self.request.support_scoring_mode,
                "worker_id": self.request.worker_id,
                "stages": [stage.value for stage in self.request.stages],
                "dry_run": self.request.dry_run,
                "limits": dict(self.request.limits),
            },
            "paths": self.paths.to_dict(),
            "artifact_uris": self.artifact_uris.to_dict(),
            "species_artifacts": {
                context.scientific_name: {
                    "root": self.artifact_uris.species_uri(context.scientific_name),
                    "context": self.artifact_uris.species_context_uri(
                        context.scientific_name
                    ),
                    "query_definitions": (
                        self.artifact_uris.species_query_definitions_uri(
                            context.scientific_name
                        )
                    ),
                }
                for context in self.manifest.taxon_scope.species_contexts
            },
            "manifest": self.manifest.to_dict(),
        }


def build_run_plan(
    request: ProductionRunRequest,
    *,
    taxon_scope: TaxonScope,
) -> ProductionRunPlan:
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
            "vision_backend": DEFAULT_VISION_BACKEND,
            "bioclip_model": DEFAULT_BIOCLIP_MODEL,
            "visual_input": ADAPTIVE_VISUAL_INPUT,
            "spatial_crop_permitted": False,
            "reference_bank_readiness": (
                {
                    "artifact_path": str(request.reference_bank_readiness),
                    "expected_sha256": request.reference_bank_readiness_sha256,
                    "validation_status": "not_validated",
                }
                if request.reference_bank_readiness
                else None
            ),
            "support_dependencies": {
                "regional_candidates": (
                    str(request.regional_candidates)
                    if request.regional_candidates
                    else None
                ),
                "reference_embeddings": (
                    str(request.reference_embeddings)
                    if request.reference_embeddings
                    else None
                ),
                "classifier_artifact": (
                    str(request.classifier_artifact)
                    if request.classifier_artifact
                    else None
                ),
                "calibrator_artifact": (
                    str(request.calibrator_artifact)
                    if request.calibrator_artifact
                    else None
                ),
                "support_scoring_mode": request.support_scoring_mode,
                "validation_status": "not_validated",
            },
            "adaptive_reference_workflow": {
                "reference_admission_mode": request.reference_admission_mode,
                "reference_source": request.reference_source,
                "initial_scoring_mode": request.initial_scoring_mode,
                "flickr_release_requires_human_review": (
                    request.flickr_release_requires_human_review
                ),
                "statistical_reference_audit": request.statistical_reference_audit,
                "strict_reference_readiness_claim": (
                    request.strict_reference_readiness_claim
                ),
                "reference_split_uses": list(request.reference_split_uses),
            },
            "artifact_layout_version": RUN_ARTIFACT_LAYOUT_VERSION,
        },
        metrics={
            "expanded_species_count": taxon_scope.species_count,
            **artifact_uris.audit_metrics(),
        },
        outputs=artifact_uris.to_dict(),
    )
    return ProductionRunPlan(
        request=request,
        paths=paths,
        artifact_uris=artifact_uris,
        manifest=manifest,
    )


class ProductionRunOrchestrator:
    """Coordinate one adaptive graph through explicit, injected stage owners."""

    def __init__(
        self,
        request: ProductionRunRequest,
        *,
        taxon_scope: TaxonScope | None = None,
        storage: CloudStorage | None = None,
        stage_handlers: Mapping[RunStage, StageHandler] | None = None,
    ) -> None:
        self.request = request
        self.taxon_scope = taxon_scope
        self.storage = storage
        self.stage_handlers = dict(stage_handlers or {})
        self._support_dependency_permit: SupportDependencyPermit | None = None
        self._reference_bank_readiness_permit: (
            ReferenceBankReadinessPermit | None
        ) = None

    def plan(self) -> ProductionRunPlan:
        return build_run_plan(self.request, taxon_scope=self._resolve_taxon_scope())

    def write_dry_run_manifest(self) -> Path:
        plan = self.plan()
        plan.paths.ensure_directories()
        return plan.manifest.write_json(plan.paths.manifest_path)

    def run(self) -> ProductionRunPlan:
        plan = self.plan()
        plan = replace(plan, manifest=plan.manifest.with_status("running"))
        for stage in self.request.stages:
            started_at = utc_now_iso()
            manifest = plan.manifest.with_stage_status(
                stage,
                StageStatus.RUNNING,
                started_at=started_at,
            )
            plan = replace(plan, manifest=manifest)
            result = self._run_stage(plan, stage)
            if stage in MANUAL_REVIEW_STAGES and result.status is StageStatus.COMPLETE:
                result = replace(
                    result,
                    status=StageStatus.AWAITING_MANUAL_REVIEW,
                    message=result.message or "manual_review_required",
                )
            manifest = self._manifest_with_support_permit(plan.manifest)
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
            plan = replace(plan, manifest=manifest)
            if result.status in {
                StageStatus.AWAITING_MANUAL_REVIEW,
                StageStatus.FAILED,
            }:
                break
        if any(stage.status is StageStatus.FAILED for stage in plan.manifest.stages):
            status = StageStatus.FAILED.value
            ended_at = utc_now_iso()
        elif any(
            stage.status is StageStatus.AWAITING_MANUAL_REVIEW
            for stage in plan.manifest.stages
        ):
            status = StageStatus.AWAITING_MANUAL_REVIEW.value
            ended_at = None
        else:
            status = StageStatus.COMPLETE.value
            ended_at = utc_now_iso()
        plan = replace(
            plan,
            manifest=plan.manifest.with_status(status, ended_at=ended_at),
        )
        self._write_manifest(plan)
        return plan

    def _run_stage(
        self,
        plan: ProductionRunPlan,
        stage: RunStage,
    ) -> StageExecutionResult:
        if not self.request.dry_run and stage in SUPPORT_DEPENDENT_STAGES:
            self._load_support_dependency_permit(plan, stage=stage)
        handler = self.stage_handlers.get(stage)
        if handler is not None:
            return handler(plan)
        if stage is RunStage.RESOLVE_TAXON_SCOPE:
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
        return StageExecutionResult(
            status=StageStatus.FAILED,
            message=f"stage_handler_not_configured:{stage.value}",
        )

    def _load_support_dependency_permit(
        self,
        plan: ProductionRunPlan,
        *,
        stage: RunStage,
    ) -> SupportDependencyPermit:
        if self._support_dependency_permit is not None:
            return self._support_dependency_permit
        permit = validate_support_readiness_dependencies(
            stage=stage,
            regional_candidates=self.request.regional_candidates,
            reference_bank_readiness=self.request.reference_bank_readiness,
            reference_bank_readiness_sha256=(
                self.request.reference_bank_readiness_sha256
            ),
            reference_embeddings=self.request.reference_embeddings,
            classifier_artifact=self.request.classifier_artifact,
            calibrator_artifact=self.request.calibrator_artifact,
            expected_registry_version=plan.manifest.taxon_scope.registry_version,
            expected_target_accepted_taxon_key=(
                plan.manifest.taxon_scope.accepted_taxon_key
            ),
            expected_model_name=DEFAULT_BIOCLIP_MODEL,
            scoring_mode=self.request.support_scoring_mode,
        )
        self._support_dependency_permit = permit
        self._reference_bank_readiness_permit = permit.readiness
        return permit

    def _manifest_with_support_permit(self, manifest: RunManifest) -> RunManifest:
        permit = self._reference_bank_readiness_permit
        if permit is None:
            return manifest
        model_configs = {
            **manifest.model_configs,
            "reference_bank_readiness": {
                "artifact_path": str(self.request.reference_bank_readiness),
                "expected_sha256": self.request.reference_bank_readiness_sha256,
                "validation_status": "validated",
                **asdict(permit),
            },
        }
        metrics = {
            **manifest.metrics,
            "reference_bank_readiness_status": str(permit.status),
            "reference_bank_readiness_sha256": permit.readiness_sha256,
        }
        dependencies = self._support_dependency_permit
        if dependencies is not None:
            model_configs["support_dependencies"] = {
                **dict(manifest.model_configs.get("support_dependencies") or {}),
                "validation_status": "validated",
                "candidate_set_fingerprints": list(
                    dependencies.candidate_set_fingerprints
                ),
                "reference_embedding_fingerprint": (
                    dependencies.reference_embedding_fingerprint
                ),
                "model_fingerprint": dependencies.model_fingerprint,
                "classifier_fingerprint": dependencies.classifier_fingerprint,
                "calibration_fingerprint": dependencies.calibration_fingerprint,
                "scoring_mode": getattr(
                    dependencies,
                    "scoring_mode",
                    self.request.support_scoring_mode,
                ),
                "score_semantics": getattr(
                    dependencies,
                    "score_semantics",
                    "uncalibrated_similarity_and_margin_not_probability",
                ),
            }
            metrics.update(
                {
                    "reference_embedding_fingerprint": (
                        dependencies.reference_embedding_fingerprint
                    ),
                    "classifier_fingerprint": dependencies.classifier_fingerprint,
                    "calibration_fingerprint": (
                        dependencies.calibration_fingerprint
                    ),
                }
            )
        return replace(manifest, model_configs=model_configs, metrics=metrics)

    def _resolve_taxon_scope(self) -> TaxonScope:
        if self.taxon_scope is not None:
            self.taxon_scope = _limit_taxon_scope(
                self.taxon_scope,
                self.request.limits,
            )
            return self.taxon_scope
        if not self.request.registry_dir:
            raise ValueError("registry_dir is required when taxon_scope is not provided")
        if is_cloud_uri(str(self.request.registry_dir)):
            resolved = resolve_taxon_scope_from_registry_frames(
                taxa=self._read_registry_parquet("taxa.parquet"),
                names=self._read_registry_optional_parquet("names.parquet"),
                source_snapshots=self._read_registry_optional_parquet(
                    "source_snapshots.parquet"
                ),
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

    def _registry_artifact_uri(self, filename: str) -> str:
        if not self.request.registry_dir:
            raise ValueError("registry_dir is required")
        return join_uri(str(self.request.registry_dir), filename)

    def _read_registry_parquet(self, filename: str) -> Any:
        if self.storage is None:
            raise ValueError("storage_backend_required_for_registry_reads")
        uri = self._registry_artifact_uri(filename)
        if not self.storage.exists(uri):
            raise FileNotFoundError(uri)
        return self.storage.read_parquet(uri)

    def _read_registry_optional_parquet(self, filename: str) -> Any:
        import polars as pl

        if self.storage is None:
            raise ValueError("storage_backend_required_for_registry_reads")
        uri = self._registry_artifact_uri(filename)
        return self.storage.read_parquet(uri) if self.storage.exists(uri) else pl.DataFrame()

    def _read_registry_manifest(self) -> dict[str, Any]:
        if self.storage is None:
            raise ValueError("storage_backend_required_for_registry_reads")
        uri = self._registry_artifact_uri("manifest.json")
        return self.storage.read_json(uri) if self.storage.exists(uri) else {}

    def _write_manifest(self, plan: ProductionRunPlan) -> Path | str | None:
        if is_cloud_uri(str(self.request.output_root)):
            if self.storage is None:
                return None
            return self.storage.write_json(
                plan.artifact_uris.manifest_uri,
                plan.manifest.to_dict(),
            )
        plan.paths.ensure_directories()
        return plan.manifest.write_json(plan.paths.manifest_path)


def _limit_taxon_scope(
    taxon_scope: TaxonScope,
    limits: Mapping[str, int],
) -> TaxonScope:
    limit = int(limits.get("species") or 0)
    if limit <= 0 or taxon_scope.species_count <= limit:
        return taxon_scope
    return replace(taxon_scope, species_contexts=taxon_scope.species_contexts[:limit])


__all__ = [
    "ADAPTIVE_VISUAL_INPUT",
    "ProductionRunOrchestrator",
    "ProductionRunPlan",
    "ProductionRunRequest",
    "StageExecutionResult",
    "build_run_plan",
]
