from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    display_name: str
    role: str
    status: str
    task: str
    notes: str = ""


class ModelRegistry:
    def __init__(self, models: dict[str, ModelConfig]) -> None:
        self.models = models

    @classmethod
    def from_config(cls, path: str | Path) -> "ModelRegistry":
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        models = {
            model_id: ModelConfig(model_id=model_id, **values)
            for model_id, values in data.get("models", {}).items()
        }
        return cls(models)

    def resolve_preferred_bioclip(self) -> ModelConfig:
        for model_id in ("bioclip2", "bioclip_newest", "bioclip1"):
            model = self.models.get(model_id)
            if model is not None and model.status in {"use_if_available", "available", "fallback_only"}:
                return model
        raise RuntimeError("No BioCLIP-family model is configured")
