from __future__ import annotations

import os
import sys

import pytest

from biominer import cli
from biominer.secrets_loader import load_runtime_secrets_env


def test_load_runtime_secrets_env_loads_flickr_key_and_secret(tmp_path, monkeypatch) -> None:
    secrets = tmp_path / "secrets.env"
    secrets.write_text(
        "\n".join(
            [
                "# local development secrets",
                "FLICKR_API_KEY=flickr-api-key-value",
                "export FLICKR_SECRET_KEY='flickr secret value'",
                'MCP_TOKEN="abc#def" # comment outside the value',
                "GITHUB_TOKEN=from-file",
                "BAD-NAME=invalid",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("FLICKR_API_KEY", raising=False)
    monkeypatch.delenv("FLICKR_SECRET_KEY", raising=False)
    monkeypatch.delenv("MCP_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "already-exported")

    result = load_runtime_secrets_env(secrets)

    assert result.exists is True
    assert set(result.loaded_names) == {"FLICKR_API_KEY", "FLICKR_SECRET_KEY", "MCP_TOKEN"}
    assert result.skipped_existing_names == ("GITHUB_TOKEN",)
    assert result.skipped_invalid_lines == 1
    assert result.error is None
    assert result.path == secrets
    assert "flickr-api-key-value" == os.environ["FLICKR_API_KEY"]
    assert "flickr secret value" == os.environ["FLICKR_SECRET_KEY"]
    assert "abc#def" == os.environ["MCP_TOKEN"]
    assert "already-exported" == os.environ["GITHUB_TOKEN"]


def test_load_runtime_secrets_env_uses_path_override(tmp_path, monkeypatch) -> None:
    secrets = tmp_path / "override.env"
    secrets.write_text("FLICKR_API_KEY=from-override\n", encoding="utf-8")
    monkeypatch.setenv("BIOMINER_SECRETS_ENV", str(secrets))
    monkeypatch.delenv("FLICKR_API_KEY", raising=False)

    result = load_runtime_secrets_env()

    assert result.path == secrets
    assert result.loaded_names == ("FLICKR_API_KEY",)
    assert os.environ["FLICKR_API_KEY"] == "from-override"


def test_load_runtime_secrets_env_can_be_disabled(tmp_path, monkeypatch) -> None:
    secrets = tmp_path / "disabled.env"
    secrets.write_text("FLICKR_API_KEY=not-loaded\n", encoding="utf-8")
    monkeypatch.setenv("BIOMINER_LOAD_SECRETS_ENV", "0")
    monkeypatch.delenv("FLICKR_API_KEY", raising=False)

    result = load_runtime_secrets_env(secrets)

    assert result.exists is True
    assert result.loaded_names == ()
    assert "FLICKR_API_KEY" not in os.environ


def test_load_runtime_secrets_env_falls_back_to_sibling_secrets(tmp_path, monkeypatch) -> None:
    base = tmp_path / "base"
    secrets_dir = base / "secrets"
    secrets_dir.mkdir(parents=True)
    secrets = secrets_dir / "secrets.env"
    secrets.write_text("FLICKR_API_KEY=from-sibling\nFLICKR_SECRET_KEY=secret\n", encoding="utf-8")
    monkeypatch.setenv("BIOMINER_BASE_PATH", str(base))
    monkeypatch.delenv("BIOMINER_SECRETS_ENV", raising=False)
    monkeypatch.delenv("BIOMINER_LOAD_SECRETS_ENV", raising=False)
    monkeypatch.delenv("FLICKR_API_KEY", raising=False)
    monkeypatch.delenv("FLICKR_SECRET_KEY", raising=False)

    result = load_runtime_secrets_env()

    assert result.path == secrets
    assert result.loaded_names == ("FLICKR_API_KEY", "FLICKR_SECRET_KEY")
    assert os.environ["FLICKR_API_KEY"] == "from-sibling"
    assert os.environ["FLICKR_SECRET_KEY"] == "secret"


def test_cli_main_loads_runtime_secrets_before_dispatch(monkeypatch, capsys) -> None:
    calls: list[str] = []

    def fake_load() -> None:
        calls.append("loaded")

    monkeypatch.setattr(cli, "load_runtime_secrets_env", fake_load)
    monkeypatch.setattr(sys, "argv", ["biominer", "--version"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    assert calls == ["loaded"]
    assert "biominer 0.1.0" in capsys.readouterr().out
