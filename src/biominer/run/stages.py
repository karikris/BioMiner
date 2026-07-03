from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class RunStage(StrEnum):
    RESOLVE_TAXON_SCOPE = "resolve_taxon_scope"
    BUILD_REGISTRY = "build_registry"
    COMPILE_QUERIES = "compile_queries"
    ENQUEUE_FLICKR_WORK = "enqueue_flickr_work"
    POLL_FLICKR = "poll_flickr"
    DETECT_OBJECTS = "detect_objects"
    SCORE_BIOCLIP = "score_bioclip"
    JOIN_EVIDENCE = "join_evidence"
    SUMMARIZE = "summarize"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
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
            outputs={str(key): str(value) for key, value in dict(payload.get("outputs") or {}).items()},
        )


def default_stage_records(stages: tuple[RunStage, ...] = DEFAULT_PRODUCTION_STAGES) -> tuple[StageRecord, ...]:
    return tuple(StageRecord(stage=stage) for stage in stages)
