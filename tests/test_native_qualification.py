from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import qualify_native_runtime as qualification


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/qualify_native_runtime.py"


def test_qualification_cli_requires_explicit_snapshot_and_context():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--context" in result.stdout
    assert "--snapshot-sha256" in result.stdout


def ref(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return ref(path)


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE,
    ).strip()


@pytest.fixture
def scenario(tmp_path):
    """Synthetic orchestration only; native case execution is mocked explicitly."""
    repo = tmp_path / "repo"
    repo.mkdir()
    source_root = SCRIPT.parents[1]
    sources = {}
    for name in qualification.REQUIRED_SOURCES:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source_root / name).read_bytes())
        sources[name] = ref(path)["sha256"]
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "synthetic qualification context")
    context = {"repo": str(repo), "tested_commit": git(repo, "rev-parse", "HEAD"), "sources": sources}
    snapshot = tmp_path / "snapshot"
    profile = write_json(snapshot / "profile.json", {"python": ref(Path(sys.executable).resolve())})
    launcher = snapshot / "scripts/freeze_native_profile.py"
    launcher.parent.mkdir()
    launcher.write_text("# fixture only; never executed\n")
    manifest = write_json(snapshot / "snapshot.json", {
        "files": {"profile.json": profile["sha256"], "scripts/freeze_native_profile.py": ref(launcher)["sha256"]},
    })
    generation = tmp_path / "generation"
    write_json(generation / "generation.json", {"profile_sha256": profile["sha256"]})
    return {
        "context": context, "snapshot": snapshot, "manifest_hash": manifest["sha256"],
        "generation": generation, "python": Path(sys.executable), "output": tmp_path / "result",
    }


def mocked_case(argv, cwd, environment, timeout):
    """Default-suite test double, never a native qualification observation."""
    config = json.loads(Path(environment["BIOMINER_NATIVE_QUALIFICATION_CONTEXT"]).read_bytes())
    case = argv[-1].rsplit("::test_", 1)[1]
    write_json(Path(config["evidence_directory"]) / f"{case}.json", {
        "case": case, "runtime_profile": config["runtime_profile"],
        "generation_receipt": config["generation_receipt"],
        "details": {"synthetic_unit_test_double": True},
    })
    return subprocess.CompletedProcess(argv, 0, b"1 passed in 0.01s\n")


def test_committed_keyword_expressions_are_validated_as_source_bytes(scenario):
    context = scenario["context"]
    repo = Path(context["repo"])
    keyword = "api_" + "key"
    path = repo / "source_expression.py"
    path.write_text(
        "import os\n"
        f"first = Client({keyword}=os.environ.get(args.{keyword}_env))\n"
        f"second = Client({keyword}={keyword})\n"
    )
    git(repo, "add", path.name)
    git(repo, "commit", "-m", "add harmless keyword expressions")
    context["tested_commit"] = git(repo, "rev-parse", "HEAD")
    context["sources"][path.name] = ref(path)["sha256"]

    assert qualification.validate_context(context) == repo


def test_producer_emits_five_distinct_source_bound_gate_receipts(scenario, monkeypatch):
    calls = []

    def observe(*args):
        calls.append(args)
        return mocked_case(*args)

    monkeypatch.setattr(qualification, "run_test", observe)
    report = qualification.qualify(**scenario)
    assert set(report["cases"]) == set(qualification.CASES)
    assert len(calls) == 5
    assert len({call[0][-1] for call in calls}) == 5
    for case, item in report["cases"].items():
        receipt = qualification.gate.ref_json(item["receipt"])
        assert receipt["check"] == f"native_verify.{case}"
        assert receipt["tested_commit"] == scenario["context"]["tested_commit"]
        assert receipt["source_digest"] == qualification.gate.source_digest(scenario["context"]["sources"])
        assert receipt["cwd"] == scenario["context"]["repo"]
        assert receipt["runtime_profile"] == report["runtime_profile"]
        assert receipt["argv"][-1].endswith(f"::test_{case}")
    assert (scenario["output"] / "native_verify.json").exists()
    qualification.gate.verify_report(
        qualification.gate.canonical(report), "native_verify", 2,
        context=scenario["context"],
        check={"runtime_profile": report["runtime_profile"]},
    )


@pytest.mark.parametrize("exit_code,log", [
    (1, b"1 passed in 0.01s\n"),
    (False, b"1 passed in 0.01s\n"),
    (0, b"1 skipped in 0.01s\n"),
    (0, b"1 failed in 0.01s\n"),
    (0, b"1 passed, 1 deselected in 0.01s\n"),
    (0, b"2 passed in 0.01s\n"),
    (0, b'{"passed": true}\n'),
])
def test_failed_skipped_or_false_success_does_not_publish_report(scenario, monkeypatch, exit_code, log):
    monkeypatch.setattr(
        qualification, "run_test",
        lambda *args: subprocess.CompletedProcess(args[0], exit_code, log),
    )
    with pytest.raises(qualification.gate.GateError):
        qualification.qualify(**scenario)
    assert not (scenario["output"] / "native_verify.json").exists()
    assert (scenario["output"] / "artifacts.log").read_bytes() == log


def test_timeout_is_preserved_as_failed_result(scenario, monkeypatch):
    def timeout(*args):
        raise subprocess.TimeoutExpired(args[0], 1, output=b"partial native output\n")

    monkeypatch.setattr(qualification, "run_test", timeout)
    with pytest.raises(qualification.gate.GateError, match="case failed"):
        qualification.qualify(**scenario)
    receipt = json.loads((scenario["output"] / "artifacts.result.json").read_bytes())
    assert receipt["exit_code"] == 124
    assert not (scenario["output"] / "native_verify.json").exists()


@pytest.mark.parametrize("parent_exits", [False, True])
def test_real_timeout_reaps_children_in_separate_sessions(tmp_path, parent_exits):
    script = """
import json, os, subprocess, sys, time
child = subprocess.Popen(
    [sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
print(json.dumps({'parent': os.getpid(), 'child': child.pid}), flush=True)
if sys.argv[1] == 'exit':
    raise SystemExit(0)
time.sleep(60)
"""
    result = qualification.run_test(
        [sys.executable, "-c", script, "exit" if parent_exits else "wait"],
        tmp_path, dict(os.environ), 1,
    )
    assert result.returncode == 124, result.stdout
    pids = json.loads(result.stdout.splitlines()[0])
    assert not Path(f"/proc/{pids['parent']}").exists()
    assert not Path(f"/proc/{pids['child']}").exists()
    assert b"descendants cleaned up" in result.stdout


@pytest.mark.parametrize("parent_exit", [0, 7])
def test_exited_adopted_child_preserves_parent_result(tmp_path, parent_exit):
    script = """
import json, os, sys
child = os.fork()
if child == 0:
    os._exit(0)
os.waitid(os.P_PID, child, os.WEXITED | os.WNOWAIT)
print(json.dumps({'parent': os.getpid(), 'child': child}), flush=True)
raise SystemExit(int(sys.argv[1]))
"""
    result = qualification.run_test(
        [sys.executable, "-c", script, str(parent_exit)],
        tmp_path, dict(os.environ), 5,
    )
    assert result.returncode == parent_exit, result.stdout
    pids = json.loads(result.stdout.splitlines()[0])
    assert not Path(f"/proc/{pids['parent']}").exists()
    assert not Path(f"/proc/{pids['child']}").exists()
    assert b"left descendant processes" not in result.stdout


def test_live_adopted_child_blocks_success_and_is_reaped(tmp_path):
    script = """
import json, os, subprocess, sys
child = subprocess.Popen(
    [sys.executable, '-c', 'import time; time.sleep(60)'],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL, start_new_session=True)
print(json.dumps({'parent': os.getpid(), 'child': child.pid}), flush=True)
"""
    result = qualification.run_test(
        [sys.executable, "-c", script], tmp_path, dict(os.environ), 5,
    )
    assert result.returncode == 125, result.stdout
    pids = json.loads(result.stdout.splitlines()[0])
    assert not Path(f"/proc/{pids['parent']}").exists()
    assert not Path(f"/proc/{pids['child']}").exists()
    assert b"left descendant processes; cleaned up" in result.stdout


@pytest.mark.parametrize("damage", ["missing", "wrong_profile", "wrong_generation", "wrong_case"])
def test_passing_pytest_without_bound_native_observation_is_rejected(scenario, monkeypatch, damage):
    def changed(*args):
        result = mocked_case(*args)
        path = scenario["output"] / "native-evidence/artifacts.json"
        if damage == "missing":
            path.unlink()
        else:
            value = json.loads(path.read_bytes())
            key = {"wrong_profile": "runtime_profile", "wrong_generation": "generation_receipt", "wrong_case": "case"}[damage]
            value[key] = "different"
            path.write_text(json.dumps(value))
        return result

    monkeypatch.setattr(qualification, "run_test", changed)
    with pytest.raises(qualification.gate.GateError):
        qualification.qualify(**scenario)
    assert not (scenario["output"] / "native_verify.json").exists()


@pytest.mark.parametrize("damage", ["working_source", "committed_hash", "omitted_source", "manifest", "generation"])
def test_stale_source_and_runtime_bindings_fail_before_execution(scenario, monkeypatch, damage):
    context = scenario["context"]
    if damage == "working_source":
        (Path(context["repo"]) / qualification.CASE_MODULE).write_text("changed")
    elif damage == "committed_hash":
        context["sources"][qualification.CASE_MODULE] = "0" * 64
    elif damage == "omitted_source":
        context["sources"].pop(qualification.CASE_MODULE)
    elif damage == "manifest":
        scenario["manifest_hash"] = "0" * 64
    else:
        write_json(scenario["generation"] / "generation.json", {"profile_sha256": "0" * 64})
    monkeypatch.setattr(qualification, "run_test", lambda *args: pytest.fail("must not execute stale input"))
    with pytest.raises(qualification.gate.GateError):
        qualification.qualify(**scenario)
    assert not scenario["output"].exists()


def test_source_change_during_case_blocks_report(scenario, monkeypatch):
    def mutate(*args):
        result = mocked_case(*args)
        (Path(scenario["context"]["repo"]) / qualification.CASE_MODULE).write_text("mutated during check")
        return result

    monkeypatch.setattr(qualification, "run_test", mutate)
    with pytest.raises(qualification.gate.GateError, match="execution qualification source"):
        qualification.qualify(**scenario)
    assert not (scenario["output"] / "native_verify.json").exists()


def test_retained_checkpoint_artifacts_are_hash_bound_to_report(scenario, monkeypatch):
    def retain(*args):
        result = mocked_case(*args)
        if args[0][-1].endswith("::test_features"):
            write_json(scenario["output"] / "native-evidence/fixture/checkpoint.json", {
                "synthetic_unit_test_double": True,
            })
        return result

    monkeypatch.setattr(qualification, "run_test", retain)
    report = qualification.qualify(**scenario)
    checkpoint = scenario["output"] / "native-evidence/fixture/checkpoint.json"
    assert ref(checkpoint) in report["evidence"]
    checkpoint.write_text("tampered")
    with pytest.raises(qualification.gate.GateError, match="hash mismatch"):
        qualification.gate.verify_report(
            qualification.gate.canonical(report), "native_verify", 2,
            context=scenario["context"], check={"runtime_profile": report["runtime_profile"]},
        )


def test_output_is_create_only_and_outside_git(scenario):
    scenario["output"].mkdir()
    sentinel = scenario["output"] / "preserve"
    sentinel.write_text("unchanged")
    with pytest.raises(qualification.gate.GateError, match="already exists"):
        qualification.qualify(**scenario)
    assert sentinel.read_text() == "unchanged"
    scenario["output"] = Path(scenario["context"]["repo"]) / "forbidden-output"
    with pytest.raises(qualification.gate.GateError, match="outside Git"):
        qualification.qualify(**scenario)
    assert not scenario["output"].exists()


def test_credential_output_is_not_published_or_passed_to_tests(scenario, monkeypatch):
    secret = "fixture-sensitive-api-value"
    monkeypatch.setenv("SUPABASE_KEY", secret)

    def leak(argv, cwd, environment, timeout):
        assert "SUPABASE_KEY" not in environment
        return subprocess.CompletedProcess(argv, 0, secret.encode() + b"\n1 passed in 0.01s\n")

    monkeypatch.setattr(qualification, "run_test", leak)
    with pytest.raises(qualification.gate.GateError, match="credential"):
        qualification.qualify(**scenario)
    assert not (scenario["output"] / "artifacts.log").exists()
    assert not (scenario["output"] / "native_verify.json").exists()


def test_opt_in_cases_fail_without_explicit_runtime_context(monkeypatch):
    spec = importlib.util.spec_from_file_location("native_cases", SCRIPT.parent.parent / qualification.CASE_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("BIOMINER_NATIVE_QUALIFICATION_CONTEXT", raising=False)
    with pytest.raises(AssertionError, match="explicit native qualification"):
        module.test_artifacts()
