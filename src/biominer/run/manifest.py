from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from biominer.run.stages import StageRecord
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
    metrics: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)

    def with_status(self, status: str, *, ended_at: str | None = None) -> RunManifest:
        return replace(self, status=status, ended_at=ended_at)

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
