from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from biominer.run.stages import MANUAL_REVIEW_STAGES, RunStage, StageRecord, StageStatus
from biominer.run.taxon_scope import TaxonScope


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    taxon_scope: TaxonScope
    status: str = "planned"
    command: tuple[str, ...] = ()
    git_sha: str | None = None
    storage_backend: str | None = None
    workstore_backend: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    stages: tuple[StageRecord, ...] = ()
    model_configs: dict[str, Any] = field(default_factory=dict)
    query_counts: dict[str, int] = field(default_factory=dict)
    detection_counts: dict[str, int] = field(default_factory=dict)
    bioclip_counts: dict[str, int] = field(default_factory=dict)
    evidence_counts: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)

    def with_status(self, status: str, *, ended_at: str | None = None) -> RunManifest:
        return replace(self, status=status, ended_at=ended_at)

    def with_stage_status(
        self,
        stage: RunStage | str,
        status: StageStatus | str,
        *,
        started_at: str | None = None,
        ended_at: str | None = None,
        message: str | None = None,
        metrics: dict[str, Any] | None = None,
        outputs: dict[str, str] | None = None,
    ) -> RunManifest:
        return self._with_stage_status(
            stage,
            status,
            started_at=started_at,
            ended_at=ended_at,
            message=message,
            metrics=metrics,
            outputs=outputs,
            allow_manual_completion=False,
        )

    def with_manual_review_approval(
        self,
        stage: RunStage | str,
        *,
        reviewer: str,
        approved_at: str | None = None,
        message: str = "manual_review_approved",
        metrics: dict[str, Any] | None = None,
        outputs: dict[str, str] | None = None,
    ) -> RunManifest:
        target = RunStage(str(stage))
        if target not in MANUAL_REVIEW_STAGES:
            raise ValueError(f"stage is not a manual-review stage: {target.value}")
        current = next((record for record in self.stages if record.stage == target), None)
        if current is None or current.status is not StageStatus.AWAITING_MANUAL_REVIEW:
            raise ValueError(
                f"manual-review stage must be awaiting manual review before approval: {target.value}"
            )
        approved_by = " ".join(str(reviewer or "").split())
        if not approved_by:
            raise ValueError("reviewer must be non-empty")
        approval_time = approved_at or utc_now_iso()
        approval_metrics = {
            **(metrics or {}),
            "manual_review_approved_by": approved_by,
            "manual_review_approved_at": approval_time,
        }
        return self._with_stage_status(
            target,
            StageStatus.COMPLETE,
            ended_at=approval_time,
            message=message,
            metrics=approval_metrics,
            outputs=outputs,
            allow_manual_completion=True,
        )

    def _with_stage_status(
        self,
        stage: RunStage | str,
        status: StageStatus | str,
        *,
        started_at: str | None = None,
        ended_at: str | None = None,
        message: str | None = None,
        metrics: dict[str, Any] | None = None,
        outputs: dict[str, str] | None = None,
        allow_manual_completion: bool,
    ) -> RunManifest:
        target = RunStage(str(stage))
        next_status = StageStatus(str(status))
        if (
            target in MANUAL_REVIEW_STAGES
            and next_status is StageStatus.COMPLETE
            and not allow_manual_completion
        ):
            raise ValueError(
                f"manual-review stage cannot be completed automatically: {target.value}"
            )
        updated: list[StageRecord] = []
        replaced = False
        for record in self.stages:
            if record.stage == target:
                updated.append(
                    StageRecord(
                        stage=target,
                        status=next_status,
                        started_at=started_at if started_at is not None else record.started_at,
                        ended_at=ended_at if ended_at is not None else record.ended_at,
                        message=message if message is not None else record.message,
                        metrics={**record.metrics, **(metrics or {})},
                        outputs={**record.outputs, **(outputs or {})},
                    )
                )
                replaced = True
            else:
                updated.append(record)
        if not replaced:
            updated.append(
                StageRecord(
                    stage=target,
                    status=next_status,
                    started_at=started_at,
                    ended_at=ended_at,
                    message=message,
                    metrics=dict(metrics or {}),
                    outputs=dict(outputs or {}),
                )
            )
        return replace(self, stages=tuple(updated))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": self.status,
            "command": list(self.command),
            "git_sha": self.git_sha,
            "storage_backend": self.storage_backend,
            "workstore_backend": self.workstore_backend,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "taxon_scope": self.taxon_scope.to_dict(),
            "species_count": self.taxon_scope.species_count,
            "stages": [stage.to_dict() for stage in self.stages],
            "model_configs": dict(self.model_configs),
            "query_counts": dict(self.query_counts),
            "detection_counts": dict(self.detection_counts),
            "bioclip_counts": dict(self.bioclip_counts),
            "evidence_counts": dict(self.evidence_counts),
            "metrics": dict(self.metrics),
            "outputs": dict(self.outputs),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunManifest:
        return cls(
            run_id=str(payload["run_id"]),
            taxon_scope=TaxonScope.from_dict(dict(payload["taxon_scope"])),
            status=str(payload.get("status") or "planned"),
            command=tuple(str(item) for item in payload.get("command", ())),
            git_sha=payload.get("git_sha"),
            storage_backend=payload.get("storage_backend"),
            workstore_backend=payload.get("workstore_backend"),
            started_at=payload.get("started_at"),
            ended_at=payload.get("ended_at"),
            stages=tuple(StageRecord.from_dict(item) for item in payload.get("stages", ())),
            model_configs=dict(payload.get("model_configs") or {}),
            query_counts={str(key): int(value) for key, value in dict(payload.get("query_counts") or {}).items()},
            detection_counts={str(key): int(value) for key, value in dict(payload.get("detection_counts") or {}).items()},
            bioclip_counts={str(key): int(value) for key, value in dict(payload.get("bioclip_counts") or {}).items()},
            evidence_counts={str(key): int(value) for key, value in dict(payload.get("evidence_counts") or {}).items()},
            metrics=dict(payload.get("metrics") or {}),
            outputs={str(key): str(value) for key, value in dict(payload.get("outputs") or {}).items()},
        )

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return output

    @classmethod
    def read_json(cls, path: str | Path) -> RunManifest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
