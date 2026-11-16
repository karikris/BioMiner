from __future__ import annotations

import os
from pathlib import Path


BASE_PATH_ENV = "BIOMINER_BASE_PATH"


def resolve_runtime_base_path(*, source_file: str | Path | None = None) -> Path:
    configured = os.environ.get(BASE_PATH_ENV)
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


def embeddinggemma2_runtime_env(runtime_python: str | Path) -> dict[str, str]:
    """Share immutable source and cache locations across all Gemma entry paths."""
    runtime = Path(runtime_python).expanduser().absolute()
    venv = runtime.parent.parent
    known_venv = venv.name in {"venv", ".venv"} or (venv / "pyvenv.cfg").is_file()
    root = runtime.parents[2] if runtime.parent.name == "bin" and known_venv and len(runtime.parents) >= 3 else EMBEDDINGGEMMA2_DIR
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(root / "cache/huggingface"))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(env["HF_HOME"]) / "hub"))
    env.setdefault("TORCH_HOME", str(root / "cache/torch"))
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    source = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _repo_root_from_source(source_file: str | Path) -> Path:
    path = Path(source_file).resolve()
    for parent in path.parents:
        if parent.name == "BioMiner" and (parent / "pyproject.toml").exists():
            return parent
    return path.parents[2]


BASE_PATH = resolve_runtime_base_path()
EMBEDDINGGEMMA2_DIR = runtime_dir("EmbeddingGemma2")
BIOCLIP25_DIR = runtime_dir("BioCLIP25")
