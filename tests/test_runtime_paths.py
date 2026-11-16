from __future__ import annotations

from biominer.runtime_paths import resolve_runtime_base_path

import pytest


def test_runtime_base_path_uses_environment_override(tmp_path, monkeypatch) -> None:
    base = tmp_path / "runtime-base"
    monkeypatch.setenv("BIOMINER_BASE_PATH", str(base))

    assert resolve_runtime_base_path() == base.resolve()


def test_runtime_base_path_defaults_to_parent_of_biominer_repo(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("BIOMINER_BASE_PATH", raising=False)
    repo = tmp_path / "BioMiner"
    module_path = repo / "src" / "biominer" / "cli.py"
    module_path.parent.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='biominer'\n", encoding="utf-8")
    module_path.write_text("# test source marker\n", encoding="utf-8")

    assert resolve_runtime_base_path(source_file=module_path) == tmp_path.resolve()


@pytest.mark.parametrize("runtime", ["/usr/bin/python3", "/usr/local/bin/python3.14"])
def test_embedding_system_runtime_uses_configured_cache_root(tmp_path, monkeypatch, runtime):
    from biominer import runtime_paths

    for name in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runtime_paths, "EMBEDDINGGEMMA2_DIR", tmp_path / "EmbeddingGemma2")
    env = runtime_paths.embeddinggemma2_runtime_env(runtime)
    assert env["HF_HOME"] == str(tmp_path / "EmbeddingGemma2/cache/huggingface")
