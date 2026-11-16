from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from biominer.cli import build_parser, run
from biominer.detection.embeddinggemma2 import DEFAULT_EMBEDDINGGEMMA2_REVISION


@pytest.mark.parametrize("operation", ["runtime-check", "prefetch", "smoke"])
def test_embedding_runtime_commands_pin_model(operation):
    command = ["dev", "vision", f"embeddinggemma2-{operation}"]
    if operation == "smoke":
        command.extend(["--image", "fixture.jpg"])
    args = build_parser().parse_args(command)
    assert args.revision == DEFAULT_EMBEDDINGGEMMA2_REVISION
    assert args.runtime_python.endswith("/EmbeddingGemma2/venv/bin/python")
    assert args.dtype == "float32"


@pytest.mark.parametrize("command", ["yoloe26-runtime-check", "yoloe26-prefetch", "yoloe26-smoke", "yoloe-flickr-sample", "yoloe-flickr-sharded"])
def test_retired_yoloe_execution_commands_are_rejected(command):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["dev", "vision", command])


def test_embedding_sharded_command_uses_one_model_per_shard():
    args = build_parser().parse_args(["dev", "vision", "embedding-flickr-sharded", "--state-db", "new.sqlite", "--output-dir", "new-output"])
    assert args.shards == 4
    assert args.batch_size == 8
    assert args.device == "auto"


def test_embedding_runtime_check_uses_requested_interpreter_without_loading_weights(tmp_path, monkeypatch, capsys):
    from biominer import embedding_cli

    interpreter = tmp_path / "EmbeddingGemma2" / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"model_weights_loaded":false}', stderr="")

    monkeypatch.setattr(embedding_cli.subprocess, "run", execute)
    args = build_parser().parse_args(["dev", "vision", "embeddinggemma2-runtime-check", "--runtime-python", str(interpreter)])
    assert run(args) == 0
    assert json.loads(capsys.readouterr().out)["model_weights_loaded"] is False
    command, options = calls[0]
    assert command[:3] == [str(interpreter), "-m", "biominer.detection.embedding_runtime"]
    assert "ultralytics" not in str(command)
    assert options["env"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_embedding_smoke_resolves_input_and_output_before_runtime(tmp_path, monkeypatch):
    from biominer import embedding_cli

    interpreter = tmp_path / "python"
    interpreter.touch()
    calls = []
    monkeypatch.setattr(embedding_cli.subprocess, "run", lambda command, **kwargs:
        calls.append(command) or SimpleNamespace(returncode=0, stdout="{}", stderr=""))
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["dev", "vision", "embeddinggemma2-smoke", "--runtime-python", str(interpreter), "--image", "fixture.jpg", "--output-dir", "smoke"])
    assert run(args) == 0
    assert str(tmp_path / "fixture.jpg") in calls[0]
    assert str(tmp_path / "smoke") in calls[0]
