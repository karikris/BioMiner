"""Public CLI and dotenv controls; no database or provider calls."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/enrich_butterfly_image_species.py"
LAUNCHER = ROOT / "scripts/start_butterfly_keyword_enrichment.sh"


def _cli(*arguments):
    environment = dict(os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE="1", POLARS_MAX_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    return subprocess.run(
        [sys.executable, "-I", "-B", str(SCRIPT), *map(str, arguments)],
        cwd=ROOT, env=environment, text=True, capture_output=True,
        timeout=60, check=False,
    )


def test_public_script_help_describes_explicit_local_resumable_controls():
    result = _cli("--help")
    assert result.returncode == 0, result.stderr
    for flag in ("--local", "--database", "--database-manifest", "--output", "--sources",
                 "--collect-only", "--secrets-file", "--max-runtime-hours", "--max-http-requests"):
        assert flag in result.stdout
    assert "filesystem/SQLite" in result.stdout
    assert "production storage defaults" in result.stdout


def test_local_opt_in_is_required_before_creating_output_or_reading_database(tmp_path):
    output = tmp_path / "must-not-be-created"
    result = _cli("--output", output, "--database", tmp_path / "absent.duckdb")
    assert result.returncode == 2
    assert "--local" in result.stderr
    assert not output.exists()


def test_unknown_source_is_rejected_without_starting_collection(tmp_path):
    output = tmp_path / "must-not-be-created"
    result = _cli("--local", "--output", output, "--sources", "invented-source")
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert not output.exists()


def test_background_launcher_requires_explicit_interpreter(tmp_path):
    environment = dict(os.environ)
    environment.pop("BIOMINER_PYTHON", None)
    environment["XDG_STATE_HOME"] = str(tmp_path / "uncreated-state")
    result = subprocess.run(["bash", str(LAUNCHER), "--help"], cwd=ROOT,
                            env=environment, text=True, capture_output=True,
                            timeout=10, check=False)
    assert result.returncode != 0
    assert "BIOMINER_PYTHON" in result.stderr
    assert not (tmp_path / "uncreated-state").exists()


def test_background_launcher_passes_script_arguments_to_public_supervisor(tmp_path):
    # This executable records argv and exits: it cannot import BioMiner, read
    # the real database, contact providers, or start another enrichment worker.
    record = tmp_path / "recorded-arguments"
    stub = tmp_path / "fake-qualified-python"
    stub.write_text(
        "#!/bin/sh\n"
        "record=" + shlex.quote(str(record)) + "\n"
        "printf '%s\\n' \"$@\" > \"$record.tmp\"\n"
        "mv -- \"$record.tmp\" \"$record\"\n",
        encoding="utf-8",
    )
    stub.chmod(0o700)
    environment = dict(os.environ)
    environment.update(BIOMINER_PYTHON=str(stub), XDG_STATE_HOME=str(tmp_path / "state"))
    output = tmp_path / "enrichment-output"
    arguments = ["--local", "--output", str(output), "--collect-only"]
    # Keep the finite recording child owned by this test shell. The production
    # launcher remains detached, while the qualification supervisor sees no
    # orphan descendants after pytest exits.
    result = subprocess.run(["bash", "-c",
                            'launch_script=$1; shift; source "$launch_script" "$@"; wait',
                            "fixture", str(LAUNCHER), *arguments], cwd=ROOT,
                            env=environment, text=True, capture_output=True,
                            timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    assert record.exists(), result.stdout
    observed = record.read_text(encoding="utf-8").splitlines()
    assert observed[:4] == ["-I", "-B", "-m", "biominer.polars_runtime"]
    boundary = observed.index("--")
    # The public supervisor supplies its own Python interpreter and -I -B.
    # Passing the executable here instead makes Python parse its ELF bytes.
    assert observed[boundary + 1:] == [str(SCRIPT), *arguments]
    assert observed[boundary + 1] != str(stub)
    assert observed[observed.index("--timeout") + 1] == "86400"
    assert observed[observed.index("--grace-seconds") + 1] == "30"
    scratch = Path(observed[observed.index("--scratch-parent") + 1])
    reports = Path(observed[observed.index("--report-dir") + 1])
    assert scratch.is_dir()
    assert reports.parent == scratch.parent
    assert (scratch.parent / "pid").read_text().strip().isdigit()
    assert "Supervisor PID:" in result.stdout
    assert "Launcher log:" in result.stdout
    assert not output.exists()


@pytest.fixture
def credential_loader(monkeypatch):
    # runpy uses a non-main name, so importing the public script cannot start it.
    # Its source-tree import path is restored by pytest after this fixture.
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = runpy.run_path(str(SCRIPT), run_name="image_keyword_cli_test")
    return namespace["load_selected_credentials"]


def test_dotenv_reads_only_selected_credentials_without_echoing_them(tmp_path, monkeypatch, capsys, credential_loader):
    monkeypatch.setenv("UNRELATED_TOKEN", "unchanged")
    monkeypatch.setenv("MYMEMORY_API_KEY", "old")
    monkeypatch.setenv("MYMEMORY_EMAIL", "old@example.invalid")
    path = tmp_path / "fixture.env"
    path.write_text(
        "# fixture-only values\n"
        "UNRELATED_TOKEN=must-not-load\n"
        "export MYMEMORY_API_KEY = 'fixture-private-key'\n"
        'MYMEMORY_EMAIL="fixture-private@example.invalid"\n'
        "not-an-assignment\n",
        encoding="utf-8",
    )
    credential_loader(path)
    assert os.environ["MYMEMORY_API_KEY"] == "fixture-private-key"
    assert os.environ["MYMEMORY_EMAIL"] == "fixture-private@example.invalid"
    assert os.environ["UNRELATED_TOKEN"] == "unchanged"
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_dotenv_never_evaluates_shell_substitution_or_expands_variables(tmp_path, monkeypatch, credential_loader):
    marker = tmp_path / "must-not-exist"
    literal = "$(touch " + str(marker) + ") ${HOME} `whoami`"
    monkeypatch.setenv("MYMEMORY_API_KEY", "old")
    path = tmp_path / "fixture.env"
    path.write_text("MYMEMORY_API_KEY='" + literal + "'\n", encoding="utf-8")
    credential_loader(path)
    assert os.environ["MYMEMORY_API_KEY"] == literal
    assert not marker.exists()


def test_missing_optional_dotenv_does_not_mutate_existing_environment(monkeypatch, credential_loader):
    monkeypatch.setenv("MYMEMORY_API_KEY", "fixture-existing-key")
    monkeypatch.setenv("MYMEMORY_EMAIL", "fixture-existing@example.invalid")
    credential_loader(None)
    assert os.environ["MYMEMORY_API_KEY"] == "fixture-existing-key"
    assert os.environ["MYMEMORY_EMAIL"] == "fixture-existing@example.invalid"
