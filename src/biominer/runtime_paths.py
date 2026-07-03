from __future__ import annotations

import os
from pathlib import Path


BASE_PATH_ENV = "BIOMINER_BASE_PATH"
LEGACY_BASE_PATH_ENV = "BIOMINER_RUNTIME_BASE_PATH"


def resolve_runtime_base_path(*, source_file: str | Path | None = None) -> Path:
    configured = os.environ.get(BASE_PATH_ENV) or os.environ.get(LEGACY_BASE_PATH_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    repo_root = _repo_root_from_source(source_file or __file__)
    if repo_root.name == "BioMiner":
        return repo_root.parent.resolve()
    cwd = Path.cwd().resolve()
    if cwd.name == "BioMiner":
        return cwd.parent
    return cwd


def runtime_dir(name: str) -> Path:
    return BASE_PATH / name


def _repo_root_from_source(source_file: str | Path) -> Path:
    path = Path(source_file).resolve()
    for parent in path.parents:
        if parent.name == "BioMiner" and (parent / "pyproject.toml").exists():
            return parent
    return path.parents[2]


BASE_PATH = resolve_runtime_base_path()
BIOMINER_DIR = runtime_dir("BioMiner")
YOLOE26_DIR = runtime_dir("YOLO26")
BIOCLIP25_DIR = runtime_dir("BioCLIP25")

