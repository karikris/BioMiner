from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from biominer.storage.paths import safe_path_component


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
        for path in (self.registry_dir, self.staging_dir, self.reports_dir, self.run_root / "species"):
            path.mkdir(parents=True, exist_ok=True)
