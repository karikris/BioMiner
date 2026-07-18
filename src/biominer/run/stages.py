from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class RunStage(StrEnum):
    RESOLVE_TAXON_SCOPE = "resolve_taxon_scope"
    BUILD_REGISTRY = "build_registry"
    GEOGRAPHIC_SPREAD = "geographic_spread"
    COMPILE_QUERIES = "compile_queries"
    ENQUEUE_FLICKR_WORK = "enqueue_flickr_work"
    POLL_FLICKR = "poll_flickr"
    FLICKR_GEO_CLUSTERING = "flickr_geo_clustering"
    REGIONAL_CANDIDATE_GENERATION = "regional_candidate_generation"
    REFERENCE_METADATA = "reference_metadata"
    REFERENCE_MEDIA = "reference_media"
    REFERENCE_DEDUPLICATION = "reference_deduplication"
    REFERENCE_QUALITY_ROUTING = "reference_quality_routing"
    REFERENCE_ADMISSION = "reference_admission"
    REFERENCE_GEOGRAPHY_INDEX = "reference_geography_index"
    REFERENCE_REVIEW = "reference_review"
    REFERENCE_EMBEDDINGS = "reference_embeddings"
    REFERENCE_PROTOTYPES = "reference_prototypes"
    CLASSIFIER_TRAINING = "classifier_training"
    CLASSIFIER_CALIBRATION = "classifier_calibration"
    REFERENCE_READINESS = "reference_readiness"
    FLICKR_DETECTION = "flickr_detection"
    FLICKR_EMBEDDING = "flickr_embedding"
    FLICKR_GEO_TAXON_PARTITIONING = "flickr_geo_taxon_partitioning"
    FAMILY_ROUTING = "family_routing"
    DYNAMIC_POOL_PLANNING = "dynamic_pool_planning"
    DYNAMIC_POOL_SCORING = "dynamic_pool_scoring"
    TARGET_AWARE_SCORING = "target_aware_scoring"
    PROVISIONAL_FLICKR_SCORING = "provisional_flickr_scoring"
    REVIEW_SAMPLE_PLANNING = "review_sample_planning"
    FLICKR_HUMAN_VERIFICATION = "flickr_human_verification"
    RISK_CONTROLLED_AUDIT = "risk_controlled_audit"
    STATISTICAL_REFERENCE_AUDIT = "statistical_reference_audit"
    TARGETED_REFERENCE_REVIEW = "targeted_reference_review"
    AFFECTED_REFERENCE_REBUILD = "affected_reference_rebuild"
    AFFECTED_RECORD_RESCORE = "affected_record_rescore"
    FINAL_QUALITY_GATE = "final_quality_gate"
    EVIDENCE = "evidence"
    EVALUATION = "evaluation"
    DETECT_OBJECTS = "detect_objects"
    SCORE_BIOCLIP = "score_bioclip"
    JOIN_EVIDENCE = "join_evidence"
    SUMMARIZE = "summarize"
    QUEUE_COMMENT_REVIEW = "queue_comment_review"
    REVIEW_COMMENTS = "review_comments"
    APPLY_COMMENT_REVIEW = "apply_comment_review"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_MANUAL_REVIEW = "awaiting_manual_review"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


DEFAULT_PRODUCTION_STAGES: tuple[RunStage, ...] = (
    RunStage.RESOLVE_TAXON_SCOPE,
    RunStage.BUILD_REGISTRY,
    RunStage.COMPILE_QUERIES,
    RunStage.ENQUEUE_FLICKR_WORK,
    RunStage.POLL_FLICKR,
    RunStage.DETECT_OBJECTS,
    RunStage.SCORE_BIOCLIP,
    RunStage.JOIN_EVIDENCE,
    RunStage.SUMMARIZE,
    RunStage.QUEUE_COMMENT_REVIEW,
    RunStage.REVIEW_COMMENTS,
    RunStage.APPLY_COMMENT_REVIEW,
)


REFERENCE_FIRST_PRODUCTION_STAGES: tuple[RunStage, ...] = (
    RunStage.RESOLVE_TAXON_SCOPE,
    RunStage.BUILD_REGISTRY,
    RunStage.GEOGRAPHIC_SPREAD,
    RunStage.COMPILE_QUERIES,
    RunStage.ENQUEUE_FLICKR_WORK,
    RunStage.POLL_FLICKR,
    RunStage.FLICKR_GEO_CLUSTERING,
    RunStage.REGIONAL_CANDIDATE_GENERATION,
    RunStage.REFERENCE_METADATA,
    RunStage.REFERENCE_MEDIA,
    RunStage.REFERENCE_REVIEW,
    RunStage.REFERENCE_EMBEDDINGS,
    RunStage.REFERENCE_PROTOTYPES,
    RunStage.CLASSIFIER_TRAINING,
    RunStage.CLASSIFIER_CALIBRATION,
    RunStage.REFERENCE_READINESS,
    RunStage.FLICKR_DETECTION,
    RunStage.FLICKR_EMBEDDING,
    RunStage.TARGET_AWARE_SCORING,
    RunStage.EVIDENCE,
    RunStage.EVALUATION,
)


ADAPTIVE_REFERENCE_PRODUCTION_STAGES: tuple[RunStage, ...] = (
    RunStage.RESOLVE_TAXON_SCOPE,
    RunStage.BUILD_REGISTRY,
    RunStage.GEOGRAPHIC_SPREAD,
    RunStage.COMPILE_QUERIES,
    RunStage.ENQUEUE_FLICKR_WORK,
    RunStage.POLL_FLICKR,
    RunStage.FLICKR_GEO_CLUSTERING,
    RunStage.FLICKR_DETECTION,
    RunStage.FLICKR_EMBEDDING,
    RunStage.REGIONAL_CANDIDATE_GENERATION,
    RunStage.REFERENCE_METADATA,
    RunStage.REFERENCE_MEDIA,
    RunStage.REFERENCE_DEDUPLICATION,
    RunStage.REFERENCE_QUALITY_ROUTING,
    RunStage.REFERENCE_ADMISSION,
    RunStage.REFERENCE_EMBEDDINGS,
    RunStage.REFERENCE_GEOGRAPHY_INDEX,
    RunStage.REFERENCE_PROTOTYPES,
    RunStage.FLICKR_GEO_TAXON_PARTITIONING,
    RunStage.FAMILY_ROUTING,
    RunStage.DYNAMIC_POOL_PLANNING,
    RunStage.DYNAMIC_POOL_SCORING,
    RunStage.PROVISIONAL_FLICKR_SCORING,
    RunStage.REVIEW_SAMPLE_PLANNING,
    RunStage.FLICKR_HUMAN_VERIFICATION,
    RunStage.RISK_CONTROLLED_AUDIT,
    RunStage.STATISTICAL_REFERENCE_AUDIT,
    RunStage.TARGETED_REFERENCE_REVIEW,
    RunStage.AFFECTED_REFERENCE_REBUILD,
    RunStage.AFFECTED_RECORD_RESCORE,
    RunStage.FINAL_QUALITY_GATE,
)


MANUAL_REVIEW_STAGES: frozenset[RunStage] = frozenset(
    {
        RunStage.REFERENCE_REVIEW,
        RunStage.FLICKR_HUMAN_VERIFICATION,
        RunStage.TARGETED_REFERENCE_REVIEW,
    }
)


@dataclass(frozen=True)
class StageRecord:
    stage: RunStage
    status: StageStatus = StageStatus.PENDING
    started_at: str | None = None
    ended_at: str | None = None
    message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "message": self.message,
            "metrics": dict(self.metrics),
            "outputs": dict(self.outputs),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StageRecord:
        return cls(
            stage=RunStage(str(payload["stage"])),
            status=StageStatus(str(payload.get("status") or StageStatus.PENDING.value)),
            started_at=payload.get("started_at"),
            ended_at=payload.get("ended_at"),
            message=payload.get("message"),
            metrics=dict(payload.get("metrics") or {}),
            outputs={
                str(key): str(value)
                for key, value in dict(payload.get("outputs") or {}).items()
            },
        )


def default_stage_records(
    stages: tuple[RunStage, ...] = DEFAULT_PRODUCTION_STAGES,
) -> tuple[StageRecord, ...]:
    return tuple(StageRecord(stage=stage) for stage in stages)
