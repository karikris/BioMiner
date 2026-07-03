from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from biominer.run.manifest import RunManifest, utc_now_iso
from biominer.run.paths import RunPaths
from biominer.run.stages import DEFAULT_PRODUCTION_STAGES, RunStage, default_stage_records
from biominer.run.taxon_scope import InputRank, TaxonScope
from biominer.storage.paths import safe_path_component


DEFAULT_BIOCLIP_MODEL = "imageomics/bioclip-2.5-vith14"
DEFAULT_VISION_BACKEND = "yoloe26"


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
            "manifest": self.manifest.to_dict(),
        }


def build_run_plan(request: ProductionRunRequest, *, taxon_scope: TaxonScope) -> ProductionRunPlan:
    run_id = request.resolved_run_id()
    paths = RunPaths.from_root(request.output_root, run_id=run_id)
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
        },
    )
    return ProductionRunPlan(request=request, paths=paths, manifest=manifest)


class ProductionRunOrchestrator:
    """Non-executing shell for the production run workflow.

    Later phases will connect this class to registry compilation, cloud workstores,
    object detection, BioCLIP scoring, and evidence aggregation. This first
    skeleton deliberately only builds a plan and dry-run manifest.
    """

    def __init__(self, request: ProductionRunRequest, *, taxon_scope: TaxonScope) -> None:
        self.request = request
        self.taxon_scope = taxon_scope

    def plan(self) -> ProductionRunPlan:
        return build_run_plan(self.request, taxon_scope=self.taxon_scope)

    def write_dry_run_manifest(self) -> Path:
        plan = self.plan()
        plan.paths.ensure_directories()
        return plan.manifest.write_json(plan.paths.manifest_path)

    def run(self) -> ProductionRunPlan:
        if self.request.dry_run:
            self.write_dry_run_manifest()
            return self.plan()
        raise NotImplementedError("production run execution will be wired in a later phase")
