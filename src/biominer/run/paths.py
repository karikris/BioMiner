from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from biominer.storage.paths import safe_path_component
from biominer.storage.uri import join_uri


RUN_ARTIFACT_LAYOUT_VERSION = "reference-first-run-artifacts-v1.0.0"

RUN_ARTIFACT_RELATIVE_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "manifest": "run_manifest.json",
        "registry": "registry",
        "staging": "staging",
        "reports": "reports",
        "species": "species",
        "state": "state",
        "metrics": "reports/run_metrics.json",
        "query_definitions": "registry/flickr_query_definitions.parquet",
        "source_records": "staging/canonical_source_records.parquet",
        "object_detections": "staging/object_detections.parquet",
        "object_scores": "staging/object_bioclip_scores.parquet",
        "photo_summary": "staging/photo_evidence_summary.parquet",
        "geographic_registry": "registry/geography",
        "taxon_geographic_spread": (
            "registry/geography/taxon_geographic_spread.parquet"
        ),
        "geographic_occurrence_evidence": (
            "registry/geography/geographic_occurrence_evidence.parquet"
        ),
        "geographic_spread_manifest": (
            "registry/geography/geographic_spread_manifest.json"
        ),
        "taxon_geographic_summary": (
            "registry/geography/taxon_geographic_summary.parquet"
        ),
        "geographic_qa_findings": (
            "registry/geography/geographic_qa_findings.parquet"
        ),
        "geographic_summary_manifest": (
            "registry/geography/geographic_summary_manifest.json"
        ),
        "flickr_geography": "flickr/geography/flickr_geography.parquet",
        "flickr_geo_clusters": "flickr/geography/flickr_geo_clusters.parquet",
        "flickr_geo_assignments": (
            "flickr/geography/flickr_geo_assignments.parquet"
        ),
        "flickr_embeddings": "flickr/embeddings/flickr_image_embeddings.parquet",
        "regional_candidates": (
            "candidates/regional_candidate_species.parquet"
        ),
        "reference_observations": (
            "references/metadata/reference_observations.parquet"
        ),
        "reference_media": "references/media",
        "reference_media_candidates": (
            "references/media/reference_media_candidates.parquet"
        ),
        "reference_media_objects": (
            "references/media/reference_media_objects.parquet"
        ),
        "reference_review": "references/review",
        "reference_review_queue": (
            "references/review/reference_review_queue.parquet"
        ),
        "reference_review_decisions": (
            "references/review/reference_review_decisions.parquet"
        ),
        "reference_readiness": "references/readiness",
        "reference_readiness_manifest": (
            "references/readiness/reference_bank_readiness.json"
        ),
        "reference_support_manifest": (
            "references/readiness/reference_support_manifest.parquet"
        ),
        "reference_bank_summary": (
            "references/readiness/reference_bank_summary.parquet"
        ),
        "reference_embeddings": (
            "references/embeddings/reference_embeddings.parquet"
        ),
        "reference_embeddings_manifest": "references/embeddings/manifest.json",
        "reference_embeddings_report": (
            "references/embeddings/reference_embeddings_report.json"
        ),
        "reference_prototypes": (
            "references/prototypes/reference_prototypes.parquet"
        ),
        "feature_matrix": "ml/features/few_shot_training_features.parquet",
        "classifiers": "ml/classifiers",
        "calibrators": "ml/calibrators",
        "target_aware_object_scores": (
            "scores/target_aware_object_scores.parquet"
        ),
        "target_aware_candidate_scores": (
            "scores/target_aware_candidate_scores.parquet"
        ),
    }
)

RUN_ARTIFACT_DIRECTORY_KEYS: frozenset[str] = frozenset(
    {
        "registry",
        "staging",
        "reports",
        "species",
        "state",
        "geographic_registry",
        "reference_media",
        "reference_review",
        "reference_readiness",
        "classifiers",
        "calibrators",
    }
)

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

    def artifact_uri(self, key: str) -> str:
        try:
            relative = RUN_ARTIFACT_RELATIVE_PATHS[key]
        except KeyError as exc:
            raise KeyError(f"unknown run artifact key: {key}") from exc
        return join_uri(self.run_root_uri, relative)

    @property
    def registry_uri(self) -> str:
        return self.artifact_uri("registry")

    @property
    def staging_uri(self) -> str:
        return self.artifact_uri("staging")

    @property
    def reports_uri(self) -> str:
        return self.artifact_uri("reports")

    @property
    def manifest_uri(self) -> str:
        return self.artifact_uri("manifest")

    @property
    def metrics_uri(self) -> str:
        return self.artifact_uri("metrics")

    @property
    def query_definitions_uri(self) -> str:
        return self.artifact_uri("query_definitions")

    @property
    def source_records_uri(self) -> str:
        return self.artifact_uri("source_records")

    @property
    def object_detections_uri(self) -> str:
        return self.artifact_uri("object_detections")

    @property
    def object_scores_uri(self) -> str:
        return self.artifact_uri("object_scores")

    @property
    def photo_summary_uri(self) -> str:
        return self.artifact_uri("photo_summary")

    @property
    def geographic_registry_uri(self) -> str:
        return self.artifact_uri("geographic_registry")

    @property
    def taxon_geographic_spread_uri(self) -> str:
        return self.artifact_uri("taxon_geographic_spread")

    @property
    def geographic_occurrence_evidence_uri(self) -> str:
        return self.artifact_uri("geographic_occurrence_evidence")

    @property
    def taxon_geographic_summary_uri(self) -> str:
        return self.artifact_uri("taxon_geographic_summary")

    @property
    def geographic_qa_findings_uri(self) -> str:
        return self.artifact_uri("geographic_qa_findings")

    @property
    def flickr_geography_uri(self) -> str:
        return self.artifact_uri("flickr_geography")

    @property
    def flickr_geo_clusters_uri(self) -> str:
        return self.artifact_uri("flickr_geo_clusters")

    @property
    def flickr_geo_assignments_uri(self) -> str:
        return self.artifact_uri("flickr_geo_assignments")

    @property
    def regional_candidates_uri(self) -> str:
        return self.artifact_uri("regional_candidates")

    @property
    def reference_observations_uri(self) -> str:
        return self.artifact_uri("reference_observations")

    @property
    def reference_media_uri(self) -> str:
        return self.artifact_uri("reference_media")

    @property
    def reference_media_candidates_uri(self) -> str:
        return self.artifact_uri("reference_media_candidates")

    @property
    def reference_media_objects_uri(self) -> str:
        return self.artifact_uri("reference_media_objects")

    @property
    def reference_review_uri(self) -> str:
        return self.artifact_uri("reference_review")

    @property
    def reference_review_queue_uri(self) -> str:
        return self.artifact_uri("reference_review_queue")

    @property
    def reference_review_decisions_uri(self) -> str:
        return self.artifact_uri("reference_review_decisions")

    @property
    def reference_readiness_uri(self) -> str:
        return self.artifact_uri("reference_readiness")

    @property
    def reference_embeddings_uri(self) -> str:
        return self.artifact_uri("reference_embeddings")

    @property
    def reference_prototypes_uri(self) -> str:
        return self.artifact_uri("reference_prototypes")

    @property
    def feature_matrix_uri(self) -> str:
        return self.artifact_uri("feature_matrix")

    @property
    def classifiers_uri(self) -> str:
        return self.artifact_uri("classifiers")

    @property
    def calibrators_uri(self) -> str:
        return self.artifact_uri("calibrators")

    @property
    def flickr_embeddings_uri(self) -> str:
        return self.artifact_uri("flickr_embeddings")

    @property
    def target_aware_object_scores_uri(self) -> str:
        return self.artifact_uri("target_aware_object_scores")

    @property
    def target_aware_candidate_scores_uri(self) -> str:
        return self.artifact_uri("target_aware_candidate_scores")

    def classifier_artifact_uri(
        self,
        *,
        target_task: str,
        route: str,
        artifact_fingerprint: str,
    ) -> str:
        return _scoped_model_artifact(
            self.classifiers_uri,
            target_task=target_task,
            route=route,
            artifact_fingerprint=artifact_fingerprint,
        )

    def calibrator_artifact_uri(
        self,
        *,
        target_task: str,
        route: str,
        artifact_fingerprint: str,
    ) -> str:
        return _scoped_model_artifact(
            self.calibrators_uri,
            target_task=target_task,
            route=route,
            artifact_fingerprint=artifact_fingerprint,
        )

    def species_uri(self, scientific_name: str) -> str:
        return join_uri(self.artifact_uri("species"), safe_path_component(scientific_name))

    def species_context_uri(self, scientific_name: str) -> str:
        return join_uri(self.species_uri(scientific_name), "species_context.json")

    def species_query_definitions_uri(self, scientific_name: str) -> str:
        return join_uri(
            self.species_uri(scientific_name),
            "flickr_query_definitions.parquet",
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "run_root": self.run_root_uri,
            **{
                key: self.artifact_uri(key)
                for key in RUN_ARTIFACT_RELATIVE_PATHS
            },
        }

    def audit_metrics(self) -> dict[str, str | int]:
        return {
            "artifact_layout_version": RUN_ARTIFACT_LAYOUT_VERSION,
            "artifact_location_count": len(self.to_dict()),
            "artifact_directory_count": len(RUN_ARTIFACT_DIRECTORY_KEYS),
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

    def artifact_path(self, key: str) -> Path:
        try:
            relative = RUN_ARTIFACT_RELATIVE_PATHS[key]
        except KeyError as exc:
            raise KeyError(f"unknown run artifact key: {key}") from exc
        return self.run_root / relative

    @property
    def registry_dir(self) -> Path:
        return self.artifact_path("registry")

    @property
    def staging_dir(self) -> Path:
        return self.artifact_path("staging")

    @property
    def reports_dir(self) -> Path:
        return self.artifact_path("reports")

    @property
    def manifest_path(self) -> Path:
        return self.artifact_path("manifest")

    @property
    def metrics_path(self) -> Path:
        return self.artifact_path("metrics")

    @property
    def query_definitions_path(self) -> Path:
        return self.artifact_path("query_definitions")

    @property
    def source_records_path(self) -> Path:
        return self.artifact_path("source_records")

    @property
    def object_detections_path(self) -> Path:
        return self.artifact_path("object_detections")

    @property
    def object_scores_path(self) -> Path:
        return self.artifact_path("object_scores")

    @property
    def photo_summary_path(self) -> Path:
        return self.artifact_path("photo_summary")

    @property
    def geographic_registry_dir(self) -> Path:
        return self.artifact_path("geographic_registry")

    @property
    def taxon_geographic_spread_path(self) -> Path:
        return self.artifact_path("taxon_geographic_spread")

    @property
    def geographic_occurrence_evidence_path(self) -> Path:
        return self.artifact_path("geographic_occurrence_evidence")

    @property
    def taxon_geographic_summary_path(self) -> Path:
        return self.artifact_path("taxon_geographic_summary")

    @property
    def geographic_qa_findings_path(self) -> Path:
        return self.artifact_path("geographic_qa_findings")

    @property
    def flickr_geography_path(self) -> Path:
        return self.artifact_path("flickr_geography")

    @property
    def flickr_geo_clusters_path(self) -> Path:
        return self.artifact_path("flickr_geo_clusters")

    @property
    def flickr_geo_assignments_path(self) -> Path:
        return self.artifact_path("flickr_geo_assignments")

    @property
    def regional_candidates_path(self) -> Path:
        return self.artifact_path("regional_candidates")

    @property
    def reference_observations_path(self) -> Path:
        return self.artifact_path("reference_observations")

    @property
    def reference_media_dir(self) -> Path:
        return self.artifact_path("reference_media")

    @property
    def reference_media_candidates_path(self) -> Path:
        return self.artifact_path("reference_media_candidates")

    @property
    def reference_media_objects_path(self) -> Path:
        return self.artifact_path("reference_media_objects")

    @property
    def reference_review_dir(self) -> Path:
        return self.artifact_path("reference_review")

    @property
    def reference_review_queue_path(self) -> Path:
        return self.artifact_path("reference_review_queue")

    @property
    def reference_review_decisions_path(self) -> Path:
        return self.artifact_path("reference_review_decisions")

    @property
    def reference_readiness_dir(self) -> Path:
        return self.artifact_path("reference_readiness")

    @property
    def reference_embeddings_path(self) -> Path:
        return self.artifact_path("reference_embeddings")

    @property
    def reference_prototypes_path(self) -> Path:
        return self.artifact_path("reference_prototypes")

    @property
    def feature_matrix_path(self) -> Path:
        return self.artifact_path("feature_matrix")

    @property
    def classifiers_dir(self) -> Path:
        return self.artifact_path("classifiers")

    @property
    def calibrators_dir(self) -> Path:
        return self.artifact_path("calibrators")

    @property
    def flickr_embeddings_path(self) -> Path:
        return self.artifact_path("flickr_embeddings")

    @property
    def target_aware_object_scores_path(self) -> Path:
        return self.artifact_path("target_aware_object_scores")

    @property
    def target_aware_candidate_scores_path(self) -> Path:
        return self.artifact_path("target_aware_candidate_scores")

    def classifier_artifact_dir(
        self,
        *,
        target_task: str,
        route: str,
        artifact_fingerprint: str,
    ) -> Path:
        return Path(
            _scoped_model_artifact(
                str(self.classifiers_dir),
                target_task=target_task,
                route=route,
                artifact_fingerprint=artifact_fingerprint,
            )
        )

    def calibrator_artifact_dir(
        self,
        *,
        target_task: str,
        route: str,
        artifact_fingerprint: str,
    ) -> Path:
        return Path(
            _scoped_model_artifact(
                str(self.calibrators_dir),
                target_task=target_task,
                route=route,
                artifact_fingerprint=artifact_fingerprint,
            )
        )

    def species_dir(self, scientific_name: str) -> Path:
        return self.artifact_path("species") / safe_path_component(scientific_name)

    def to_dict(self) -> dict[str, str]:
        return {
            "run_root": str(self.run_root),
            **{
                key: str(self.artifact_path(key))
                for key in RUN_ARTIFACT_RELATIVE_PATHS
            },
        }

    def ensure_directories(self) -> None:
        for key, relative in RUN_ARTIFACT_RELATIVE_PATHS.items():
            artifact = self.run_root / relative
            directory = (
                artifact if key in RUN_ARTIFACT_DIRECTORY_KEYS else artifact.parent
            )
            directory.mkdir(parents=True, exist_ok=True)


def _scoped_model_artifact(
    root: str | Path,
    *,
    target_task: str,
    route: str,
    artifact_fingerprint: str,
) -> str:
    return join_uri(
        root,
        f"task={safe_path_component(target_task)}",
        f"route={safe_path_component(route)}",
        f"artifact={safe_path_component(artifact_fingerprint)}",
    )


__all__ = [
    "RUN_ARTIFACT_DIRECTORY_KEYS",
    "RUN_ARTIFACT_LAYOUT_VERSION",
    "RUN_ARTIFACT_RELATIVE_PATHS",
    "RunArtifactUris",
    "RunPaths",
]
