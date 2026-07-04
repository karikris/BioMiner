from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from biominer.bioclip.object_runner import OBJECT_VISUAL_MODES, PRIMARY_VISUAL_CLASSIFIER
from biominer.run.manifest import RunManifest, utc_now_iso
from biominer.run.paths import RunArtifactUris, RunPaths
from biominer.run.stages import DEFAULT_PRODUCTION_STAGES, RunStage, StageStatus, default_stage_records
from biominer.run.taxon_scope import InputRank, TaxonScope, resolve_taxon_scope_from_registry
from biominer.storage.paths import safe_path_component
from biominer.storage.uri import is_cloud_uri


DEFAULT_BIOCLIP_MODEL = "imageomics/bioclip-2.5-vith14"
DEFAULT_VISION_BACKEND = "yoloe26"


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
            "primary_visual_classifier": PRIMARY_VISUAL_CLASSIFIER,
            "visual_modes": list(OBJECT_VISUAL_MODES),
        },
        query_counts={"compiled_definitions": 0, "enqueued_work_items": 0},
        detection_counts={"images_seen": 0, "detections": 0, "crops_created": 0},
        bioclip_counts={"objects_scored": 0, "whole_images_scored": 0, "segmentation_crops_scored": 0},
        evidence_counts={"object_evidence_rows": 0, "photo_summary_rows": 0},
        metrics={"expanded_species_count": taxon_scope.species_count},
        outputs=artifact_uris.to_dict(),
    )
    return ProductionRunPlan(request=request, paths=paths, artifact_uris=artifact_uris, manifest=manifest)


class ProductionRunOrchestrator:
    """Non-executing shell for the production run workflow.

    Later phases will connect this class to registry compilation, cloud workstores,
    object detection, BioCLIP scoring, and evidence aggregation. This first
    skeleton deliberately only builds a plan and dry-run manifest.
    """

    def __init__(
        self,
        request: ProductionRunRequest,
        *,
        taxon_scope: TaxonScope | None = None,
        stage_handlers: Mapping[RunStage, StageHandler] | None = None,
    ) -> None:
        self.request = request
        self.taxon_scope = taxon_scope
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
        return StageExecutionResult(status=StageStatus.SKIPPED, message="stage_not_implemented")

    def _write_manifest_if_local(self, plan: ProductionRunPlan) -> Path | None:
        if is_cloud_uri(self.request.output_root):
            return None
        plan.paths.ensure_directories()
        return plan.manifest.write_json(plan.paths.manifest_path)
