from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    display_name: str
    role: str
    status: str
    task: str
    model_name: str = ""
    checkpoint: str = ""
    package_name: str = ""
    package_version: str = ""
    model_hash: str = ""
    source_url: str = ""
    local_install_path_env: str = ""
    local_install_path: str = ""
    notes: str = ""


@dataclass(frozen=True)
class BioClipRuntime:
    model: ModelConfig
    home: Path | None
    venv_python: Path | None
    package_version: str
    available: bool
    unavailable_reason: str = ""


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
        for model_id in ("bioclip2_5_huge", "bioclip_newest", "bioclip2", "bioclip1"):
            model = self.models.get(model_id)
            if model is not None and model.status in {"use_if_available", "available", "fallback_only"}:
                return model
        raise RuntimeError("No BioCLIP-family model is configured")

    def resolve_preferred_bioclip_runtime(self) -> BioClipRuntime:
        model = self.resolve_preferred_bioclip()
        home = _resolve_local_home(model)
        if home is None:
            return BioClipRuntime(
                model=model,
                home=None,
                venv_python=None,
                package_version="",
                available=False,
                unavailable_reason=f"{model.model_id} has no local install path configured",
            )

        venv_python = home / ".venv" / "bin" / "python"
        if not home.exists():
            return BioClipRuntime(
                model=model,
                home=home,
                venv_python=venv_python,
                package_version="",
                available=False,
                unavailable_reason=f"{home} does not exist",
            )
        if not venv_python.exists():
            return BioClipRuntime(
                model=model,
                home=home,
                venv_python=venv_python,
                package_version="",
                available=False,
                unavailable_reason=f"{venv_python} does not exist",
            )

        package_version = _read_venv_package_version(home, model.package_name)
        if not package_version:
            return BioClipRuntime(
                model=model,
                home=home,
                venv_python=venv_python,
                package_version="",
                available=False,
                unavailable_reason=f"{model.package_name} is not installed in {home}",
            )

        return BioClipRuntime(
            model=model,
            home=home,
            venv_python=venv_python,
            package_version=package_version,
            available=True,
        )

    def require_preferred_bioclip_runtime(self) -> BioClipRuntime:
        runtime = self.resolve_preferred_bioclip_runtime()
        if not runtime.available:
            raise RuntimeError(f"BioCLIP runtime is not available: {runtime.unavailable_reason}")
        return runtime


def _resolve_local_home(model: ModelConfig) -> Path | None:
    if model.local_install_path_env:
        env_value = os.environ.get(model.local_install_path_env)
        if env_value:
            return Path(env_value)
    if model.local_install_path:
        return Path(model.local_install_path)
    return None


def _read_venv_package_version(home: Path, package_name: str) -> str:
    site_packages_dirs = (home / ".venv" / "lib").glob("python*/site-packages")
    normalized = package_name.replace("-", "_").lower()
    for site_packages in site_packages_dirs:
        for dist_info in site_packages.glob("*.dist-info"):
            metadata = dist_info / "METADATA"
            if not metadata.exists():
                continue
            found_name = ""
            found_version = ""
            for line in metadata.read_text(encoding="utf-8").splitlines():
                if line.startswith("Name:"):
                    found_name = line.split(":", 1)[1].strip().replace("-", "_").lower()
                if line.startswith("Version:"):
                    found_version = line.split(":", 1)[1].strip()
            if found_name == normalized:
                return found_version
    return ""
