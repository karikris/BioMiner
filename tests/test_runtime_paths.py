from __future__ import annotations

from biominer.runtime_paths import resolve_runtime_base_path


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
