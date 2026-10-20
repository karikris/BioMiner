from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_upgrade_phase.py"


def test_phase_cli_exposes_validation_and_post_push_handshake():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "finalize" in result.stdout
    assert "continue" in result.stdout


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def ref(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path, value):
    path.write_text(json.dumps(value))
    return ref(path)


def source_digest(sources):
    return hashlib.sha256(
        json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.fixture
def phase(tmp_path):
    from scripts import check_upgrade_phase as gate

    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "config", "user.name", "Fixture")
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (repo / "uv.lock").write_text("version = 1\n")
    (repo / "produce_report.py").write_text(
        "import pathlib, sys\nprint(pathlib.Path(sys.argv[1]).read_text())\n"
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_example.py").write_text("def test_ok():\n    assert True\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "baseline")
    base = git(repo, "rev-parse", "HEAD")
    tasks = {}
    for number in range(1, 4):
        (repo / f"input{number}.py").write_text(f"VALUE = {number}\n")
        git(repo, "add", f"input{number}.py")
        git(repo, "commit", "-m", f"task 1.{number}")
        tasks[f"1.{number}"] = git(repo, "rev-parse", "HEAD")
    tested = tasks["1.3"]
    sources = {
        name: ref(repo / name)["sha256"]
        for name in git(repo, "ls-files").splitlines()
    }
    artifact = tmp_path / "evidence.txt"
    artifact.write_text("Recorded fixture evidence; no live services were used.")
    artifact_ref = ref(artifact)
    reviews = {}
    for kind in ("spec_review", "quality_review"):
        report = tmp_path / f"{kind}.txt"
        report.write_text(f"Independent {kind} examines the committed fixture inputs.")
        reviews[kind] = write_json(
            tmp_path / f"{kind}.json",
            {
                "schema_version": 1,
                "kind": kind,
                "reviewer": f"independent-{kind}",
                "implementer": "implementation-agent",
                "task_commits": tasks,
                "source_digest": source_digest(sources),
                "verdict": "approved",
                "findings": [],
                "report": ref(report),
            },
        )
    checks = {}
    for name in ("inventory", "baseline", "security", "phase_gate_tests"):
        checker = "pytest" if name in ("baseline", "phase_gate_tests") else "report"
        if checker == "pytest":
            argv = [sys.executable, "-m", "pytest", "-q"]
            log = tmp_path / f"{name}.log"
            log.write_text("2 passed in 0.03s\n")
        else:
            log = tmp_path / f"{name}.log"
            argv = [sys.executable, "produce_report.py", str(log)]
            report = {
                "schema_version": 1,
                "kind": name,
                "evidence": [artifact_ref],
            }
            if name == "inventory":
                report["entries"] = [
                    {"id": "fixture-runtime", "runtime": artifact_ref, "owner": "fixture"}
                ]
            else:
                report.update(findings=[], coverage_gaps=[], scopes=["biominer-runtime"])
            write_json(log, report)
        receipt = {
            "schema_version": 1,
            "check": name,
            "argv": argv,
            "cwd": str(repo),
            "exit_code": 0,
            "tested_commit": tested,
            "source_digest": source_digest(sources),
            "log": ref(log),
        }
        checks[name] = {
            "argv": argv,
            "checker": checker,
            "receipt": write_json(tmp_path / f"{name}.result.json", receipt),
        }
    manifest = {
        "schema_version": 1,
        "phase": 1,
        "repo": str(repo),
        "base_commit": base,
        "tested_commit": tested,
        "tasks": tasks,
        "sources": sources,
        "artifacts": [artifact_ref],
        "implementer": "implementation-agent",
        "reviews": reviews,
        "checks": checks,
        "blockers": [],
        "previous": None,
    }
    path = tmp_path / "phase.json"
    write_json(path, manifest)
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    git(repo, "remote", "add", "origin", str(remote))
    return gate, repo, path, manifest


def save_phase(phase):
    _, _, path, manifest = phase
    write_json(path, manifest)
    return path


def alter_check(phase, name, *, summary=None, **changes):
    _, _, _, manifest = phase
    item = manifest["checks"][name]
    receipt_path = Path(item["receipt"]["path"])
    receipt = json.loads(receipt_path.read_text())
    receipt.update(changes)
    if summary is not None:
        Path(receipt["log"]["path"]).write_text(summary)
        receipt["log"] = ref(Path(receipt["log"]["path"]))
    item["receipt"] = write_json(receipt_path, receipt)
    save_phase(phase)


def summary_commit(phase):
    _, repo, _, _ = phase
    (repo / "docs").mkdir(exist_ok=True)
    (repo / "docs" / "phase-1.md").write_text("Phase 1 verification summary.\n")
    git(repo, "add", "docs/phase-1.md")
    git(repo, "commit", "-m", "phase 1 summary")
    return git(repo, "rev-parse", "HEAD")


def test_valid_candidate_retains_unpushed_state(phase):
    gate, _, path, _ = phase
    assert gate.validate_phase(path)["state"] == "validated"


@pytest.mark.parametrize("name", ["inventory", "baseline", "security", "phase_gate_tests"])
def test_missing_mandatory_check_rejected(phase, name):
    gate, _, _, manifest = phase
    del manifest["checks"][name]
    with pytest.raises(gate.GateError, match="mandatory"):
        gate.validate_phase(save_phase(phase))


@pytest.mark.parametrize(
    ("summary", "exit_code"),
    [
        ("2 passed, 1 skipped in 0.03s\n", 0),
        ("1 failed, 2 passed in 0.03s\n", 0),
        ("all checks passed\n", 0),
        ("2 passed in 0.03s\n", 1),
    ],
)
def test_failed_skipped_or_unparsed_tests_rejected(phase, summary, exit_code):
    gate, _, path, _ = phase
    alter_check(phase, "baseline", summary=summary, exit_code=exit_code)
    with pytest.raises(gate.GateError, match="pytest|exit"):
        gate.validate_phase(path)


def test_log_hash_tampering_rejected(phase):
    gate, _, path, manifest = phase
    receipt = json.loads(Path(manifest["checks"]["baseline"]["receipt"]["path"]).read_text())
    Path(receipt["log"]["path"]).write_text("6000 passed in 0.01s\n")
    with pytest.raises(gate.GateError, match="hash"):
        gate.validate_phase(path)


def test_changed_tested_source_rejected(phase):
    gate, repo, path, _ = phase
    (repo / "input1.py").write_text("VALUE = 'modified after testing'\n")
    with pytest.raises(gate.GateError, match="source"):
        gate.validate_phase(path)


def test_omitted_task_source_rejected(phase):
    gate, _, _, manifest = phase
    del manifest["sources"]["input1.py"]
    with pytest.raises(gate.GateError, match="source"):
        gate.validate_phase(save_phase(phase))


def test_review_report_hash_tampering_rejected(phase):
    gate, _, path, manifest = phase
    review = json.loads(Path(manifest["reviews"]["spec_review"]["path"]).read_text())
    Path(review["report"]["path"]).write_text("Review changed after approval.")
    with pytest.raises(gate.GateError, match="hash"):
        gate.validate_phase(path)


def test_self_review_rejected(phase):
    gate, _, _, manifest = phase
    review_path = Path(manifest["reviews"]["spec_review"]["path"])
    review = json.loads(review_path.read_text())
    review["reviewer"] = manifest["implementer"]
    manifest["reviews"]["spec_review"] = write_json(review_path, review)
    with pytest.raises(gate.GateError, match="independent"):
        gate.validate_phase(save_phase(phase))


def test_phase_one_inventory_may_report_open_security_findings(phase):
    gate, _, path, _ = phase
    report = {
        "schema_version": 1,
        "kind": "security",
        "evidence": phase[3]["artifacts"],
        "findings": [{"id": "fixture", "scope": "biominer-runtime", "status": "open"}],
        "coverage_gaps": [],
        "scopes": ["biominer-runtime"],
    }
    alter_check(phase, "security", summary=json.dumps(report))
    assert gate.validate_phase(path)["state"] == "validated"


def test_credentials_in_evidence_rejected_without_echoing_secret(phase):
    gate, _, path, _ = phase
    secret = "fixture-super-sensitive-password"
    alter_check(
        phase,
        "baseline",
        summary="".join(["postgresql", "://user:", secret, "@localhost/db\n2 passed in 0.03s\n"]),
    )
    with pytest.raises(gate.GateError, match="credential") as exc:
        gate.validate_phase(path)
    assert secret not in str(exc.value)


def test_unknown_phase_and_task_ids_rejected(phase):
    gate, _, _, manifest = phase
    manifest["phase"] = 7
    with pytest.raises(gate.GateError, match="phase"):
        gate.validate_phase(save_phase(phase))


def test_post_push_handshake_requires_observed_remote_match(phase, tmp_path):
    gate, repo, path, _ = phase
    git(repo, "push", "origin", "main")
    final = summary_commit(phase)
    with pytest.raises(gate.GateError, match="remote"):
        gate.finalize_phase(path, final, tmp_path / "handshake.json")
    assert not (tmp_path / "handshake.json").exists()


def test_final_handshake_is_create_only_and_outside_git(phase, tmp_path):
    gate, repo, path, _ = phase
    final = summary_commit(phase)
    git(repo, "push", "origin", "main")
    target = tmp_path / "handshake.json"
    result = gate.finalize_phase(path, final, target)
    assert result["observed_remote_sha"] == final
    assert result["state"] == "pushed"
    original = target.read_bytes()
    with pytest.raises(gate.GateError, match="exist"):
        gate.finalize_phase(path, final, target)
    assert target.read_bytes() == original
    with pytest.raises(gate.GateError, match="outside"):
        gate.finalize_phase(path, final, repo / "handshake.json")


def test_final_summary_cannot_change_tested_code(phase, tmp_path):
    gate, repo, path, _ = phase
    (repo / "input1.py").write_text("VALUE = 'untested'\n")
    git(repo, "add", "input1.py")
    git(repo, "commit", "-m", "untested final change")
    final = git(repo, "rev-parse", "HEAD")
    git(repo, "push", "origin", "main")
    with pytest.raises(gate.GateError, match="source|summary"):
        gate.finalize_phase(path, final, tmp_path / "handshake.json")


def test_continuation_rejects_changed_predecessor_evidence(phase, tmp_path):
    gate, repo, path, _ = phase
    final = summary_commit(phase)
    git(repo, "push", "origin", "main")
    target = tmp_path / "handshake.json"
    gate.finalize_phase(path, final, target)
    Path(phase[3]["artifacts"][0]["path"]).write_text("stale")
    with pytest.raises(gate.GateError, match="hash"):
        gate.continue_phase(target, 2)


def test_continuation_rejects_remote_moved_after_receipt(phase, tmp_path):
    gate, repo, path, _ = phase
    final = summary_commit(phase)
    git(repo, "push", "origin", "main")
    target = tmp_path / "handshake.json"
    gate.finalize_phase(path, final, target)
    git(repo, "commit", "--allow-empty", "-m", "remote moved")
    git(repo, "push", "origin", "main")
    with pytest.raises(gate.GateError, match="remote"):
        gate.continue_phase(target, 2)


def test_continuation_accepts_only_immediate_next_phase(phase, tmp_path):
    gate, repo, path, _ = phase
    final = summary_commit(phase)
    git(repo, "push", "origin", "main")
    target = tmp_path / "handshake.json"
    gate.finalize_phase(path, final, target)
    assert gate.continue_phase(target, 2)["state"] == "ready"
    with pytest.raises(gate.GateError, match="next phase"):
        gate.continue_phase(target, 3)


def test_runner_executes_commands_and_binds_actual_result_logs(phase, tmp_path):
    gate, _, path, _ = phase
    result = gate.run_phase(path, tmp_path / "run")
    candidate = Path(result["receipt"]["path"])
    manifest = json.loads(candidate.read_text())
    receipt = json.loads(Path(manifest["checks"]["baseline"]["receipt"]["path"]).read_text())
    assert receipt["exit_code"] == 0
    assert "1 passed" in Path(receipt["log"]["path"]).read_text()
    assert gate.validate_phase(candidate)["state"] == "validated"


def test_runner_does_not_overwrite_evidence_directory(phase, tmp_path):
    gate, _, path, _ = phase
    output = tmp_path / "run"
    output.mkdir()
    (output / "existing.log").write_text("preserved")
    with pytest.raises(gate.GateError, match="exist"):
        gate.run_phase(path, output)
    assert (output / "existing.log").read_text() == "preserved"


def test_runner_rejects_secret_output_before_writing_it(phase, tmp_path):
    gate, _, path, manifest = phase
    secret = "unique-sensitive-password"
    manifest["checks"]["inventory"]["argv"] = [
        sys.executable,
        "-c",
        "print('postgresql' + '://user:' + 'unique-sensitive-' + 'password' + '@localhost/db')",
    ]
    save_phase(phase)
    with pytest.raises(gate.GateError, match="credential"):
        gate.run_phase(path, tmp_path / "run")
    for item in (tmp_path / "run").rglob("*"):
        if item.is_file():
            assert secret.encode() not in item.read_bytes()


def test_full_suite_command_cannot_be_replaced_by_echo(phase):
    gate, _, _, manifest = phase
    manifest["checks"]["baseline"]["argv"] = ["echo", "2 passed in 0.03s"]
    with pytest.raises(gate.GateError, match="pytest"):
        gate.validate_phase(save_phase(phase))


def test_full_suite_cannot_select_only_one_test(phase):
    gate, _, _, manifest = phase
    manifest["checks"]["baseline"]["argv"].append("tests/test_example.py")
    with pytest.raises(gate.GateError, match="filtered"):
        gate.validate_phase(save_phase(phase))


def test_unknown_security_success_flag_does_not_replace_report(phase):
    gate, _, path, _ = phase
    alter_check(phase, "security", summary=json.dumps({"passed": True}))
    with pytest.raises(gate.GateError, match="schema"):
        gate.validate_phase(path)


def test_later_phase_security_rejects_unresolved_runtime_exposure(phase):
    gate, _, _, manifest = phase
    report = {
        "schema_version": 1,
        "kind": "security",
        "evidence": manifest["artifacts"],
        "scopes": ["biominer-runtime"],
        "findings": [{"id": "fixture", "scope": "biominer-runtime", "status": "open"}],
        "coverage_gaps": [],
    }
    with pytest.raises(gate.GateError, match="security finding"):
        gate.verify_report(json.dumps(report).encode(), "security", 3)


def test_cleanup_requires_actual_hashed_inspection(phase):
    gate, _, _, manifest = phase
    report = {
        "schema_version": 1,
        "kind": "live_cleanup",
        "evidence": manifest["artifacts"],
        "namespaces": {
            service: {"namespace": f"biominer_upgrade_{service}", "remaining": 0}
            for service in ("postgres", "s3")
        },
    }
    with pytest.raises(gate.GateError, match="reference"):
        gate.verify_report(json.dumps(report).encode(), "live_cleanup", 5)


def test_runner_uses_clean_snapshot_for_baseline(phase, tmp_path):
    gate, repo, _, manifest = phase
    snapshot = tmp_path / "snapshot"
    subprocess.run(
        ["git", "clone", "--local", str(repo), str(snapshot)],
        check=True, capture_output=True,
    )
    manifest["checks"]["baseline"]["cwd"] = str(snapshot)
    result = gate.run_phase(save_phase(phase), tmp_path / "run")
    candidate = json.loads(Path(result["receipt"]["path"]).read_text())
    receipt = json.loads(Path(candidate["checks"]["baseline"]["receipt"]["path"]).read_text())
    assert receipt["cwd"] == str(snapshot)


def test_runner_rejects_dirty_baseline_snapshot_even_outside_declared_scope(phase, tmp_path):
    gate, repo, _, manifest = phase
    # Existing unmodified source may be outside the operator's narrow scope.
    del manifest["sources"]["tests/test_example.py"]
    for reference in manifest["reviews"].values():
        review_path = Path(reference["path"])
        review = json.loads(review_path.read_text())
        review["source_digest"] = source_digest(manifest["sources"])
        reference.update(write_json(review_path, review))
    (repo / "tests" / "test_example.py").write_text("def test_hidden():\n    assert True\n")
    with pytest.raises(gate.GateError, match="snapshot"):
        gate.run_phase(save_phase(phase), tmp_path / "run")


def test_failed_runner_command_keeps_exit_and_log_without_candidate(phase, tmp_path):
    gate, _, _, manifest = phase
    manifest["checks"]["inventory"]["argv"] = [
        sys.executable, "-c", "print('fixture command failed'); raise SystemExit(3)",
    ]
    output = tmp_path / "run"
    with pytest.raises(gate.GateError, match="exit"):
        gate.run_phase(save_phase(phase), output)
    receipt = json.loads((output / "inventory.result.json").read_text())
    assert receipt["exit_code"] == 3
    assert (output / "inventory.log").read_text() == "fixture command failed\n"
    assert not (output / "phase.json").exists()


def test_runner_rejects_inherited_credential_value(phase, tmp_path, monkeypatch):
    gate, _, _, manifest = phase
    monkeypatch.setenv("FIXTURE_ACCESS_TOKEN", "sensitive-runtime-value-only")
    manifest["checks"]["inventory"]["argv"] = [
        sys.executable, "-c", "import os; print(os.environ['FIXTURE_ACCESS_TOKEN'])",
    ]
    with pytest.raises(gate.GateError, match="credential"):
        gate.run_phase(save_phase(phase), tmp_path / "run")
    assert not (tmp_path / "run" / "inventory.log").exists()


def test_phase_two_accepts_only_evidence_bound_containment(phase):
    gate, _, _, manifest = phase
    report = {
        "schema_version": 1,
        "kind": "security",
        "evidence": manifest["artifacts"],
        "scopes": ["biominer-runtime"],
        "findings": [{
            "id": "fixture", "scope": "biominer-runtime", "status": "contained",
            "evidence": manifest["artifacts"][0],
        }],
        "coverage_gaps": [],
    }
    assert gate.verify_report(json.dumps(report).encode(), "security", 2)["findings"]
    with pytest.raises(gate.GateError, match="security finding"):
        gate.verify_report(json.dumps(report).encode(), "security", 3)


@pytest.mark.parametrize("bad", ["compiler_jobs", "version", "features"])
def test_arrow_build_contract_rejects_unqualified_build(phase, bad):
    gate, _, _, manifest = phase
    artifact = manifest["artifacts"][0]
    report = {
        "schema_version": 1, "kind": "arrow_build", "evidence": [artifact],
        "compiler_jobs": 4, "version": "26.0.0+biominer.fixture",
        "source": artifact, "recipe": artifact, "wheel": artifact,
        "native_receipt": artifact, "features": ["s3", "flight", "encryption"],
    }
    report[bad] = {"compiler_jobs": 5, "version": "26.0.0", "features": ["s3"]}[bad]
    with pytest.raises(gate.GateError, match="Arrow"):
        gate.verify_report(json.dumps(report).encode(), "arrow_build", 4)


def test_benchmark_gate_rejects_median_regression(phase):
    gate, _, _, manifest = phase
    report = {
        "schema_version": 1, "kind": "benchmark", "evidence": manifest["artifacts"],
        "samples": [
            {
                "baseline": {"wall_seconds": 1, "peak_rss_bytes": 100},
                "target": {"wall_seconds": 1.2, "peak_rss_bytes": 100},
                "evidence": manifest["artifacts"][0],
            }
            for _ in range(3)
        ],
        "budget_violations": [], "resume_losses": [], "provenance_losses": [],
    }
    with pytest.raises(gate.GateError, match="15 percent"):
        gate.verify_report(json.dumps(report).encode(), "benchmark", 5)


def test_live_cleanup_blocks_remaining_objects(phase):
    gate, _, _, manifest = phase
    report = {
        "schema_version": 1,
        "kind": "live_cleanup",
        "evidence": manifest["artifacts"],
        "namespaces": {
            service: {
                "namespace": f"biominer_upgrade_{service}", "remaining": 1,
                "inspection": manifest["artifacts"][0],
            }
            for service in ("postgres", "s3")
        },
    }
    with pytest.raises(gate.GateError, match="cleanup incomplete"):
        gate.verify_report(json.dumps(report).encode(), "live_cleanup", 5)


def test_fake_live_report_without_required_cases_rejected(phase):
    gate, _, _, manifest = phase
    report = {
        "schema_version": 1, "kind": "live_postgres", "evidence": manifest["artifacts"],
        "execution": "live", "namespace": "biominer_upgrade_fixture",
        "cases": {}, "passed": True,
    }
    with pytest.raises(gate.GateError, match="mandatory cases"):
        gate.verify_report(json.dumps(report).encode(), "live_postgres", 5)


def test_final_summary_credentials_block_handshake(phase, tmp_path):
    gate, repo, path, _ = phase
    (repo / "docs").mkdir()
    summary = repo / "docs" / "summary.md"
    summary.write_text("".join(["postgresql", "://user:", "fixture-secret", "@localhost/db"]))
    git(repo, "add", "docs/summary.md")
    git(repo, "commit", "-m", "unsafe summary fixture")
    final = git(repo, "rev-parse", "HEAD")
    git(repo, "push", "origin", "main")
    with pytest.raises(gate.GateError, match="credential"):
        gate.finalize_phase(path, final, tmp_path / "handshake.json")
    assert not (tmp_path / "handshake.json").exists()


def test_cli_reports_malformed_report_without_traceback(phase):
    _, _, path, _ = phase
    alter_check(phase, "inventory", summary=json.dumps({
        "schema_version": 1, "kind": "inventory", "evidence": phase[3]["artifacts"],
        "entries": [42],
    }))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "validate", str(path)], capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert json.loads(result.stderr)["state"] == "blocked"


@pytest.mark.parametrize("name", ["SUPABASE_KEY", "SUPABASE_SERVICE_ROLE_KEY", "BIOMINER_POSTGRES_DSN"])
def test_provider_credential_environment_names_are_scanned(phase, monkeypatch, name):
    gate, _, _, _ = phase
    value = "unique-sensitive-provider-value"
    monkeypatch.setenv(name, value)
    with pytest.raises(gate.GateError, match="credential"):
        gate.scan_credentials(value.encode())


@pytest.mark.parametrize("name", ["pytest.ini", "setup.cfg", "tox.ini", "pytest.toml", ".pytest.ini", "conftest.py"])
@pytest.mark.parametrize("ignored", [False, True])
def test_full_suite_rejects_untracked_or_ignored_pytest_configuration(phase, name, ignored):
    gate, repo, path, _ = phase
    if ignored:
        (repo / ".git" / "info" / "exclude").write_text(f"{name}\n")
    (repo / name).write_text("# unreviewed test configuration\n")
    with pytest.raises(gate.GateError, match="snapshot"):
        gate.validate_phase(path)


def test_native_cases_cannot_reuse_bare_unbound_logs(phase, tmp_path):
    gate, _, _, manifest = phase
    log = tmp_path / "generic.log"
    log.write_text("1 passed in 0.01s\n")
    report = {
        "schema_version": 1, "kind": "native_verify", "evidence": manifest["artifacts"],
        "cases": {
            name: {"exit_code": 0, "log": ref(log)}
            for name in ("artifacts", "loaded_versions", "tls", "import_order", "features")
        },
    }
    with pytest.raises(gate.GateError, match="receipt|context"):
        gate.verify_report(json.dumps(report).encode(), "native_verify", 2)


def native_report(phase, tmp_path):
    _, repo, _, manifest = phase
    profile = manifest["artifacts"][0]
    report = {
        "schema_version": 1, "kind": "native_verify",
        "evidence": [profile], "runtime_profile": profile, "cases": {},
    }
    for case in ("artifacts", "loaded_versions", "tls", "import_order", "features"):
        log = tmp_path / f"native-{case}.log"
        log.write_text("1 passed in 0.01s\n")
        receipt = {
            "schema_version": 1, "check": f"native_verify.{case}",
            "argv": [sys.executable, "-m", "pytest", "-q", "tests/test_example.py"],
            "cwd": str(repo), "exit_code": 0,
            "tested_commit": manifest["tested_commit"],
            "source_digest": source_digest(manifest["sources"]),
            "runtime_profile": profile, "log": ref(log),
        }
        report["cases"][case] = {"receipt": write_json(tmp_path / f"native-{case}.json", receipt)}
    return report, {"runtime_profile": profile}


def test_native_cases_accept_source_and_runtime_bound_command_receipts(phase, tmp_path):
    gate, _, _, manifest = phase
    report, check = native_report(phase, tmp_path)
    result = gate.verify_report(
        json.dumps(report).encode(), "native_verify", 2, context=manifest, check=check,
    )
    assert result["runtime_profile"] == check["runtime_profile"]


@pytest.mark.parametrize("field", ["tested_commit", "source_digest", "runtime_profile", "check"])
def test_native_case_rejects_stale_source_profile_or_reused_case(phase, tmp_path, field):
    gate, _, _, manifest = phase
    report, check = native_report(phase, tmp_path)
    path = Path(report["cases"]["tls"]["receipt"]["path"])
    receipt = json.loads(path.read_text())
    receipt[field] = {
        "tested_commit": "a" * 40, "source_digest": "b" * 64,
        "runtime_profile": {"path": manifest["artifacts"][0]["path"], "sha256": "c" * 64},
        "check": "native_verify.features",
    }[field]
    report["cases"]["tls"]["receipt"] = write_json(path, receipt)
    with pytest.raises(gate.GateError, match="nested|runtime"):
        gate.verify_report(
            json.dumps(report).encode(), "native_verify", 2, context=manifest, check=check,
        )


@pytest.mark.parametrize("category", ["findings", "coverage_gaps"])
def test_later_security_report_cannot_omit_predecessor_ids(phase, category):
    gate, _, _, manifest = phase
    earlier = {
        "findings": [], "coverage_gaps": [],
    }
    earlier[category] = [{"id": "native-lxml", "scope": "biominer-runtime", "status": "open"}]
    report = {
        "schema_version": 1, "kind": "security", "evidence": manifest["artifacts"],
        "scopes": ["biominer-runtime"], "findings": [], "coverage_gaps": [],
    }
    with pytest.raises(gate.GateError, match="predecessor"):
        gate.verify_report(
            json.dumps(report).encode(), "security", 2, previous_security=earlier,
        )


def test_native_coverage_gap_requires_evidence_bound_resolution(phase):
    gate, _, _, manifest = phase
    earlier = {
        "findings": [],
        "coverage_gaps": [{"id": "native-lxml", "scope": "biominer-runtime", "status": "open"}],
    }
    report = {
        "schema_version": 1, "kind": "security", "evidence": manifest["artifacts"],
        "scopes": ["biominer-runtime"], "findings": [],
        "coverage_gaps": [{
            "id": "native-lxml", "scope": "biominer-runtime", "status": "resolved",
            "evidence": manifest["artifacts"][0],
        }],
    }
    assert gate.verify_report(
        json.dumps(report).encode(), "security", 2, previous_security=earlier,
    )["coverage_gaps"][0]["status"] == "resolved"
    report["coverage_gaps"][0]["status"] = "contained"
    with pytest.raises(gate.GateError, match="coverage gap"):
        gate.verify_report(
            json.dumps(report).encode(), "security", 2, previous_security=earlier,
        )
