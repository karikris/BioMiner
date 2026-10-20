from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


MODULE = Path(__file__).resolve().parents[1] / "src/biominer/polars_runtime.py"
OBSERVE = """
import json,os,pathlib,sys
root=pathlib.Path(os.environ['POLARS_OOC_SPILL_DIR'])
(root/'worker-data').write_text('temporary')
pathlib.Path(sys.argv[1]).write_text(json.dumps({
 'environment':{k:v for k,v in os.environ.items() if k.startswith('POLARS_')},
 'scratch':str(root),'mode':root.stat().st_mode & 0o777,
 'executable':sys.executable,'prefix':sys.prefix}))
"""


@pytest.fixture
def runtime():
    assert MODULE.is_file(), "the per-run Polars supervisor is missing"
    return importlib.import_module("biominer.polars_runtime")


def execute(runtime, tmp_path, *, env=None, code=OBSERVE, extra=(), report="report", **kwargs):
    output = tmp_path / f"{report}-observed.json"
    result = runtime.run(
        ["-c", code, str(output), *extra], environment={} if env is None else env,
        scratch_parent=tmp_path, report_dir=tmp_path / report, timeout=10, **kwargs,
    )
    return result, json.loads(output.read_text()) if output.exists() else None


def test_default_policy_reaches_fresh_child_and_reports_decimal_budget_bytes(runtime, tmp_path):
    result, observed = execute(runtime, tmp_path)
    assert observed["environment"] == {
        "POLARS_MAX_THREADS": "4", "POLARS_OOC_MEMORY_BUDGET_MB": "1073",
        "POLARS_OOC_DISK_BUDGET_MB": "4294", "POLARS_OOC_SPILL_DIR": observed["scratch"],
    }
    assert observed["mode"] == 0o700
    assert result["policy"]["memory"]["requested_default_bytes"] == 1024**3
    assert result["policy"]["memory"]["absolute_bytes"] == 1_073_000_000
    assert result["policy"]["disk"]["requested_default_bytes"] == 4 * 1024**3
    assert result["policy"]["disk"]["bytes"] == 4_294_000_000
    assert result["status"] == "passed" and result["exit_code"] == 0
    assert result["cleanup"]["status"] == "removed"
    assert not Path(observed["scratch"]).exists()
    assert json.loads((tmp_path / "report/finished.json").read_text()) == result
    assert (tmp_path / "report/started.json").is_file()


@pytest.mark.parametrize("memory", [
    {"POLARS_OOC_MEMORY_BUDGET_MB": "2000"},
    {"POLARS_OOC_MEMORY_BUDGET_FRACTION": "0.25"},
    {"POLARS_OOC_MEMORY_BUDGET_MB": "2", "POLARS_OOC_MEMORY_BUDGET_FRACTION": "0.1"},
])
def test_declared_budgets_are_preserved_without_inventing_an_absolute_fraction_cap(runtime, tmp_path, memory):
    env = {**memory, "POLARS_MAX_THREADS": "2", "POLARS_OOC_DISK_BUDGET_MB": "12"}
    original = env.copy()
    result, observed = execute(runtime, tmp_path, env=env)
    assert env == original
    for key, value in env.items():
        assert observed["environment"][key] == value
    if "POLARS_OOC_MEMORY_BUDGET_MB" not in memory:
        assert "POLARS_OOC_MEMORY_BUDGET_MB" not in observed["environment"]
        assert result["policy"]["memory"]["absolute_bytes"] is None
    assert result["policy"]["memory"]["origin"] == "explicit"


def test_numeric_overrides_are_canonicalized_without_changing_their_budget(runtime, tmp_path):
    _, observed = execute(runtime, tmp_path, env={
        "POLARS_MAX_THREADS": "  +02\t", "POLARS_OOC_MEMORY_BUDGET_MB": "0012",
        "POLARS_OOC_DISK_BUDGET_MB": " 32 ", "POLARS_OOC_MEMORY_BUDGET_FRACTION": " 0.25 ",
    })
    assert observed["environment"]["POLARS_MAX_THREADS"] == "2"
    assert observed["environment"]["POLARS_OOC_MEMORY_BUDGET_MB"] == "12"
    assert observed["environment"]["POLARS_OOC_DISK_BUDGET_MB"] == "32"
    assert observed["environment"]["POLARS_OOC_MEMORY_BUDGET_FRACTION"] == "0.25"


@pytest.mark.parametrize(("key", "value"), [
    ("POLARS_MAX_THREADS", "0"), ("POLARS_MAX_THREADS", "nan"),
    ("POLARS_OOC_MEMORY_BUDGET_MB", "-1"), ("POLARS_OOC_MEMORY_BUDGET_MB", ""),
    ("POLARS_OOC_DISK_BUDGET_MB", "1.5"), ("POLARS_OOC_DISK_BUDGET_MB", str(2**64)),
    ("POLARS_OOC_MEMORY_BUDGET_FRACTION", "nan"), ("POLARS_OOC_MEMORY_BUDGET_FRACTION", "inf"),
    ("POLARS_OOC_MEMORY_BUDGET_FRACTION", "0"), ("POLARS_OOC_MEMORY_BUDGET_FRACTION", "1.1"),
])
def test_invalid_policy_fails_before_launch_or_creating_scratch(runtime, tmp_path, key, value):
    with pytest.raises(runtime.PolicyError, match=key):
        execute(runtime, tmp_path, env={key: value})
    assert list(tmp_path.iterdir()) == []


def test_sequential_runs_get_distinct_owned_leaves_and_preserve_existing_parent_files(runtime, tmp_path):
    sibling = tmp_path / "123456789"
    sibling.mkdir()
    (sibling / "keep").write_text("not owned by this run")
    first, first_observed = execute(runtime, tmp_path, report="first")
    second, second_observed = execute(runtime, tmp_path, report="second")
    assert first_observed["scratch"] != second_observed["scratch"]
    assert first["cleanup"]["status"] == second["cleanup"]["status"] == "removed"
    assert (sibling / "keep").read_text() == "not owned by this run"


def test_explicit_spill_directory_is_an_isolated_parent_not_deleted(runtime, tmp_path):
    parent = tmp_path / "declared"
    parent.mkdir()
    (parent / "keep").write_text("retain")
    result, observed = execute(runtime, tmp_path, env={"POLARS_OOC_SPILL_DIR": str(parent)})
    assert Path(observed["scratch"]).parent == parent
    assert result["policy"]["spill_parent_origin"] == "environment"
    assert (parent / "keep").read_text() == "retain"


@pytest.mark.parametrize(("ending", "returncode"), [
    ("raise RuntimeError('synthetic failure')", 1),
    ("sys.exit(7)", 7),
    ("os.kill(os.getpid(), 15)", -15),
])
def test_worker_failure_is_reported_and_scratch_is_cleaned(runtime, tmp_path, ending, returncode):
    result, observed = execute(runtime, tmp_path, code=OBSERVE + "\n" + ending)
    assert result["status"] == "failed"
    assert result["worker"]["returncode"] == returncode
    assert result["exit_code"] != 0
    assert result["cleanup"]["status"] == "removed"
    assert not Path(observed["scratch"]).exists()


def test_cleanup_never_follows_a_worker_symlink_to_unowned_content(runtime, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("retain")
    code = OBSERVE + "\n(root/'external').symlink_to(sys.argv[2], target_is_directory=True)"
    result, _ = execute(runtime, tmp_path, code=code, extra=[str(outside)])
    assert result["cleanup"]["status"] == "removed"
    assert (outside / "keep").read_text() == "retain"


def test_cleanup_failure_is_a_failed_receipt_not_success(runtime, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise PermissionError("synthetic cleanup failure with secret-like-text")
    monkeypatch.setattr(runtime, "_remove_owned_scratch", fail)
    result, observed = execute(runtime, tmp_path)
    assert result["status"] == "failed" and result["exit_code"] != 0
    assert result["cleanup"] == {"status": "failed", "error_type": "PermissionError"}
    assert "secret-like-text" not in json.dumps(result)
    import shutil
    shutil.rmtree(observed["scratch"])


def test_existing_report_directory_is_never_overwritten_or_launched(runtime, tmp_path):
    (tmp_path / "report").mkdir()
    with pytest.raises((runtime.PolicyError, FileExistsError)):
        execute(runtime, tmp_path)
    assert list(tmp_path.iterdir()) == [tmp_path / "report"]


def test_initial_receipt_failure_removes_scratch_without_starting_a_worker(runtime, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("synthetic receipt write failure")
    def unexpected_worker(*args, **kwargs):
        pytest.fail("worker must not launch before initial receipt publication")
    monkeypatch.setattr(runtime, "_json", fail)
    monkeypatch.setattr(runtime.subprocess, "Popen", unexpected_worker)
    with pytest.raises(OSError, match="receipt write"):
        execute(runtime, tmp_path)
    assert list(tmp_path.iterdir()) == [tmp_path / "report"]


@pytest.mark.parametrize("ignore_term", [False, True])
def test_timeout_reaps_worker_and_reports_cleanup(runtime, tmp_path, ignore_term):
    code = "import time,signal;"
    if ignore_term:
        code += "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
    code += "time.sleep(30)"
    result = runtime.run(
        ["-c", code], environment={},
        scratch_parent=tmp_path, report_dir=tmp_path / "report", timeout=0.1, grace_seconds=0.1,
    )
    assert result["termination_reason"] == "timeout"
    assert result["exit_code"] == 124 and result["cleanup"]["status"] == "removed"
    assert result["process_group"]["status"] == "clear"


def test_surviving_child_in_owned_group_is_stopped_before_cleanup(runtime, tmp_path):
    code = OBSERVE + """
import subprocess
p=subprocess.Popen([sys.executable,'-I','-B','-c','import time;time.sleep(30)'])
pathlib.Path(sys.argv[2]).write_text(str(p.pid))
"""
    pidfile = tmp_path / "descendant.pid"
    result, _ = execute(runtime, tmp_path, code=code, extra=[str(pidfile)], grace_seconds=0.1)
    pid = int(pidfile.read_text())
    stat_path = Path(f"/proc/{pid}/stat")
    assert not stat_path.exists() or stat_path.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    assert result["process_group"]["status"] == "clear"
    assert result["cleanup"]["status"] == "removed"


def test_detached_descendant_is_reaped_before_owned_scratch_is_removed(runtime, tmp_path):
    code = OBSERVE + """
import subprocess
p=subprocess.Popen([sys.executable,'-I','-B','-c','import time;time.sleep(30)'], start_new_session=True)
pathlib.Path(sys.argv[2]).write_text(str(p.pid))
"""
    pidfile = tmp_path / "detached.pid"
    try:
        result, _ = execute(runtime, tmp_path, code=code, extra=[str(pidfile)], grace_seconds=0.1)
        assert not Path(f"/proc/{int(pidfile.read_text())}").exists()
        assert result["process_group"]["detached_descendants_terminated"] is True
        assert result["exit_code"] != 0
        assert result["cleanup"]["status"] == "removed"
    finally:
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_public_run_rejects_a_caller_with_unrelated_children(runtime, tmp_path):
    other = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    try:
        with pytest.raises(runtime.PolicyError, match="existing descendants"):
            execute(runtime, tmp_path)
        assert other.poll() is None
        assert list(tmp_path.iterdir()) == []
    finally:
        other.terminate()
        other.wait()


def test_supervisor_sigterm_forwards_signal_and_publishes_cleanup_receipt(runtime, tmp_path):
    report = tmp_path / "report"
    ready = tmp_path / "ready"
    code = "import pathlib,time,sys;pathlib.Path(sys.argv[1]).touch();time.sleep(30)"
    process = subprocess.Popen([
        sys.executable, "-I", "-B", str(MODULE), "--scratch-parent", str(tmp_path),
        "--report-dir", str(report), "--grace-seconds", "0.1", "--", "-c", code, str(ready),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "worker never started"
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) == 128 + signal.SIGTERM
        receipt = json.loads((report / "finished.json").read_text())
        assert receipt["termination_reason"] == "signal:15"
        assert receipt["cleanup"]["status"] == "removed"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_cli_entry_bootstrap_does_not_import_scientific_cli_in_supervisor(runtime, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runtime, "run", lambda args, **kwargs: calls.append(args) or {
        "exit_code": 0, "report_dir": str(tmp_path / "report"), "status": "passed",
    })
    monkeypatch.setattr(sys, "argv", ["biominer", "--help"])
    before = sys.modules.get("biominer.cli")
    assert runtime.cli_main() == 0
    assert calls == [["-m", "biominer.cli", "--help"]]
    assert sys.modules.get("biominer.cli") is before


def test_symlink_scratch_parent_is_rejected_before_launch(runtime, tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(parent, target_is_directory=True)
    with pytest.raises(runtime.PolicyError, match="canonical"):
        execute(runtime, tmp_path, env={"POLARS_OOC_SPILL_DIR": str(alias)})
    assert list(parent.iterdir()) == []


def test_nested_supervisor_uses_its_own_leaf_and_retains_declared_budgets(runtime, tmp_path):
    inner_report = tmp_path / "inner-report"
    result, observed = execute(
        runtime, tmp_path, code=(
            "import sys;from biominer.polars_runtime import run;"
            f"r=run(['-c',{OBSERVE!r},sys.argv[1]],report_dir=sys.argv[2]);"
            "raise SystemExit(r['exit_code'])"
        ), extra=[str(inner_report)], env={"POLARS_MAX_THREADS": "2"},
    )
    inner = json.loads((inner_report / "finished.json").read_text())
    assert inner["scratch"]["path"] != result["scratch"]["path"]
    assert Path(inner["scratch"]["path"]).parent == Path(result["scratch"]["path"])
    assert observed["environment"]["POLARS_MAX_THREADS"] == "2"
    assert inner["cleanup"]["status"] == result["cleanup"]["status"] == "removed"


def test_nested_default_report_survives_outer_scratch_cleanup(runtime, tmp_path):
    result, observed = execute(
        runtime, tmp_path, code=(
            "import json,pathlib,sys;from biominer.polars_runtime import run;"
            f"r=run(['-c',{OBSERVE!r},sys.argv[1]]);"
            "p=pathlib.Path(sys.argv[1]);v=json.loads(p.read_text());"
            "v['inner_report']=r['report_dir'];p.write_text(json.dumps(v));"
            "raise SystemExit(r['exit_code'])"
        ), env={"TMPDIR": str(tmp_path)},
    )
    assert not Path(result["scratch"]["path"]).exists()
    assert (Path(observed["inner_report"]) / "finished.json").is_file()
