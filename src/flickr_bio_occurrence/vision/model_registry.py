from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    display_name: str
    role: str
    status: str
    task: str
    model_name: str
    checkpoint: str
    package_name: str
    package_version: str
    model_hash: str


@dataclass(frozen=True)
class BioClipRuntime:
    model: ModelConfig
    home: Path
    venv_python: Path | None
    package_version: str
    available: bool
    unavailable_reason: str = ""
