from __future__ import annotations

from pathlib import Path

import pytest

from flickr_bio_occurrence.review.rules import review_status_for_candidate
from flickr_bio_occurrence.vision.model_registry import ModelRegistry


def test_bioclip2_or_newest_model_is_preferred() -> None:
    registry = ModelRegistry.from_config("config/model_registry.toml")
    selected = registry.resolve_preferred_bioclip()

    assert selected.model_id == "bioclip2_5_huge"
    assert selected.role == "preferred"
    assert selected.model_name == "imageomics/bioclip-2"
    assert selected.package_name == "open_clip_torch"
    assert selected.local_install_path_env == "BIOCLIP25_HOME"


def test_bioclip2_remains_available_as_fallback_to_newest() -> None:
    registry = ModelRegistry.from_config("config/model_registry.toml")

    assert registry.models["bioclip2"].role == "fallback"
    assert registry.models["bioclip1"].role == "fallback"


def test_bioclip_conflict_routes_to_review() -> None:
    status = review_status_for_candidate(species_agreement_status="text_vision_conflict", range_extension_candidate=False)

    assert status == "needs_review"


def test_local_bioclip_runtime_uses_configured_env_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_home = tmp_path / "bioclip25"
    site_packages = runtime_home / ".venv" / "lib" / "python3.14" / "site-packages"
    dist_info = site_packages / "open_clip_torch-3.3.0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text("Name: open_clip_torch\nVersion: 3.3.0\n", encoding="utf-8")
    venv_python = runtime_home / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    monkeypatch.setenv("BIOCLIP25_HOME", str(runtime_home))

    registry = ModelRegistry.from_config("config/model_registry.toml")
    runtime = registry.resolve_preferred_bioclip_runtime()

    assert runtime.model.model_id == "bioclip2_5_huge"
    assert runtime.home == runtime_home
    assert runtime.venv_python == venv_python
    assert runtime.package_version == "3.3.0"
    assert runtime.available is True


def test_local_bioclip_runtime_fails_clearly_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIOCLIP25_HOME", "/definitely/missing/bioclip25")
    registry = ModelRegistry.from_config("config/model_registry.toml")

    with pytest.raises(RuntimeError, match="BioCLIP runtime is not available"):
        registry.require_preferred_bioclip_runtime()
