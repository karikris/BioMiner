from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from biominer.storage.paths import safe_path_component
from biominer.storage.uri import join_uri


@dataclass(frozen=True)
class RunArtifactUris:
    """URI-oriented artifact layout for local or S3-compatible production runs."""

    root_uri: str
    run_id: str

    @classmethod
    def from_prefix(cls, prefix: str | Path, *, run_id: str) -> RunArtifactUris:
        return cls(root_uri=str(prefix).rstrip("/"), run_id=safe_path_component(run_id))

    @property
    def run_root_uri(self) -> str:
        return join_uri(self.root_uri, f"run_id={self.run_id}")

    @property
    def registry_uri(self) -> str:
        return join_uri(self.run_root_uri, "registry")

    @property
    def staging_uri(self) -> str:
        return join_uri(self.run_root_uri, "staging")

    @property
    def reports_uri(self) -> str:
        return join_uri(self.run_root_uri, "reports")

    @property
    def manifest_uri(self) -> str:
        return join_uri(self.run_root_uri, "run_manifest.json")

    @property
    def metrics_uri(self) -> str:
        return join_uri(self.reports_uri, "run_metrics.json")

    @property
    def vision_stage_metrics_uri(self) -> str:
        return join_uri(self.reports_uri, "vision_stage_metrics.json")

    @property
    def vision_stage_summary_uri(self) -> str:
        return join_uri(self.reports_uri, "vision_stage_summary.md")

    @property
    def review_queue_uri(self) -> str:
        return join_uri(self.reports_uri, "review_queue.parquet")

    @property
    def visual_qa_findings_uri(self) -> str:
        return join_uri(self.reports_uri, "visual_qa_findings.parquet")

    @property
    def reviewed_object_evidence_uri(self) -> str:
        return join_uri(self.staging_uri, "object_evidence_reviewed.parquet")

    @property
    def query_definitions_uri(self) -> str:
        return join_uri(self.registry_uri, "flickr_query_definitions.parquet")

    @property
    def source_records_uri(self) -> str:
        return join_uri(self.staging_uri, "canonical_source_records.parquet")

    @property
    def object_detections_uri(self) -> str:
        return join_uri(self.staging_uri, "object_detections.parquet")

    @property
    def object_scores_uri(self) -> str:
        return join_uri(self.staging_uri, "object_bioclip_scores.parquet")

    @property
    def object_evidence_uri(self) -> str:
        return join_uri(self.staging_uri, "object_evidence_joined.parquet")

    @property
    def photo_summary_uri(self) -> str:
        return join_uri(self.staging_uri, "photo_evidence_summary.parquet")

    def species_uri(self, scientific_name: str) -> str:
        return join_uri(self.run_root_uri, "species", safe_path_component(scientific_name))

    def species_context_uri(self, scientific_name: str) -> str:
        return join_uri(self.species_uri(scientific_name), "species_context.json")

    def species_query_definitions_uri(self, scientific_name: str) -> str:
        return join_uri(self.species_uri(scientific_name), "flickr_query_definitions.parquet")

    def to_dict(self) -> dict[str, str]:
        return {
            "run_root": self.run_root_uri,
            "manifest": self.manifest_uri,
            "metrics": self.metrics_uri,
            "vision_stage_metrics": self.vision_stage_metrics_uri,
            "vision_stage_summary": self.vision_stage_summary_uri,
            "review_queue": self.review_queue_uri,
            "visual_qa_findings": self.visual_qa_findings_uri,
            "reviewed_object_evidence": self.reviewed_object_evidence_uri,
            "query_definitions": self.query_definitions_uri,
            "source_records": self.source_records_uri,
            "object_detections": self.object_detections_uri,
            "object_scores": self.object_scores_uri,
            "object_evidence": self.object_evidence_uri,
            "photo_summary": self.photo_summary_uri,
        }


@dataclass(frozen=True)
class RunPaths:
    """Filesystem-oriented artifact layout for local dry-runs and tests."""

    root: Path
    run_id: str

    @classmethod
    def from_root(cls, root: str | Path, *, run_id: str) -> RunPaths:
        return cls(root=Path(root), run_id=safe_path_component(run_id))

    @property
    def run_root(self) -> Path:
        return self.root / f"run_id={self.run_id}"

    @property
    def registry_dir(self) -> Path:
        return self.run_root / "registry"

    @property
    def staging_dir(self) -> Path:
        return self.run_root / "staging"

    @property
    def reports_dir(self) -> Path:
        return self.run_root / "reports"

    @property
    def manifest_path(self) -> Path:
        return self.run_root / "run_manifest.json"

    @property
    def metrics_path(self) -> Path:
        return self.reports_dir / "run_metrics.json"

    @property
    def vision_stage_metrics_path(self) -> Path:
        return self.reports_dir / "vision_stage_metrics.json"

    @property
    def vision_stage_summary_path(self) -> Path:
        return self.reports_dir / "vision_stage_summary.md"

    @property
    def review_queue_path(self) -> Path:
        return self.reports_dir / "review_queue.parquet"

    @property
    def visual_qa_findings_path(self) -> Path:
        return self.reports_dir / "visual_qa_findings.parquet"

    @property
    def comment_review_state_path(self) -> Path:
        return self.run_root / "state" / "comment_review.sqlite"

    @property
    def reviewed_object_evidence_path(self) -> Path:
        return self.staging_dir / "object_evidence_reviewed.parquet"

    @property
    def query_definitions_path(self) -> Path:
        return self.registry_dir / "flickr_query_definitions.parquet"

    @property
    def source_records_path(self) -> Path:
        return self.staging_dir / "canonical_source_records.parquet"

    @property
    def object_detections_path(self) -> Path:
        return self.staging_dir / "object_detections.parquet"

    @property
    def object_scores_path(self) -> Path:
        return self.staging_dir / "object_bioclip_scores.parquet"

    @property
    def object_evidence_path(self) -> Path:
        return self.staging_dir / "object_evidence_joined.parquet"

    @property
    def photo_summary_path(self) -> Path:
        return self.staging_dir / "photo_evidence_summary.parquet"

    def species_dir(self, scientific_name: str) -> Path:
        return self.run_root / "species" / safe_path_component(scientific_name)

    def ensure_directories(self) -> None:
        for path in (self.registry_dir, self.staging_dir, self.reports_dir, self.run_root / "species", self.run_root / "state"):
            path.mkdir(parents=True, exist_ok=True)
