"""Opt-in real Polars 2 spill qualification; missing context fails, never skips.

Engine budgets are neither RSS bounds nor aggregate child-process limits.
The committed module also supplies its own synthetic worker, so context hashes
cover the complete probe rather than an independently editable helper.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


_spec = importlib.util.spec_from_file_location(
    "resource_qualification", Path(__file__).resolve().parents[1] / "qualify_native_runtime.py",
)
native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native)
CASE_MODULE = "scripts/upgrade_checks/test_polars_resource_cases.py"
SUPERVISOR_MODULE = "src/biominer/polars_runtime.py"
SUPERVISOR_SHA256 = "461bc3c4c412679090291fa8968781307c9c685c7b2576e9fffbfed301f3b171"
VERSIONS = {
    "polars": "2.0.0", "polars-runtime-32": "2.0.0",
    "pyarrow": "25.0.1+biominer.f918dfd84c8c",
}


def _reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _checked(reference):
    assert _reference(reference["path"]) == reference, "resource qualification input changed"
    return json.loads(Path(reference["path"]).read_bytes())


def _save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _environment(output):
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(output / "home"),
        "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "PY_COLORS": "0",
        "POLARS_MAX_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "UV_NO_CACHE": "1",
    }


def _command(argv, output, name, cwd, environment, expected_exit=0):
    # The existing dedicated subreaper retains partial timeout output and stops
    # detached descendants. Raw credentials are never inherited by the probe.
    result = native.run_test(argv, cwd, environment, 180)
    log = output / f"{name}.log"
    with log.open("xb") as stream:
        stream.write(result.stdout)
    _save(output / f"{name}-command.json", {
        "argv": argv, "cwd": str(cwd), "exit_code": result.returncode,
        "expected_exit": expected_exit, "log": _reference(log),
        "polars_environment": {key: value for key, value in environment.items()
                               if key.startswith("POLARS_")},
    })
    assert result.returncode == expected_exit, f"{name} failed; retained {log}"
    return result.stdout


def _validate_sources(context):
    cwd = native.validate_context(context)
    production = subprocess.check_output([
        "git", "-C", context["repo"], "ls-tree", "-r", "--name-only",
        context["tested_commit"], "--", "src/biominer",
    ], text=True, timeout=60).splitlines()
    required = {CASE_MODULE, SUPERVISOR_MODULE, "scripts/development_runtime.py"}
    required.update(name for name in production if name.endswith((".py", ".json")))
    assert required <= context["sources"].keys(), "resource source closure incomplete"
    assert context["sources"][SUPERVISOR_MODULE] == SUPERVISOR_SHA256
    assert Path(__file__).resolve() == cwd / CASE_MODULE
    assert _reference(__file__)["sha256"] == context["sources"][CASE_MODULE]
    return cwd


def _runtime_inputs(context):
    runtime = context["runtime"]
    profile = _checked(runtime["profile"])
    development = _checked(runtime["development_receipt"])
    assert development["profile_sha256"] == runtime["profile"]["sha256"]
    assert development["production_ready"] is False
    assert development["source_is_editable"] is True
    checkout = Path(development["source_checkout"]).resolve()
    # This case targets the P2 runtime; unlike the paired P1 overlay, even its
    # dependency configuration must match the tested commit.
    for name, expected in context["sources"].items():
        assert _reference(checkout / name)["sha256"] == expected
    return profile, development, checkout


def _verify(context, output, cwd, name):
    profile, development, checkout = _runtime_inputs(context)
    raw = _command([
        profile["python"]["path"], "-I", "-B", str(cwd / "scripts/development_runtime.py"),
        "verify", "--root", development["root"],
        "--receipt-sha256", context["runtime"]["development_receipt"]["sha256"],
    ], output, name, cwd, _environment(output))
    observed = json.loads(raw)
    assert observed["status"] == "verified-development"
    assert observed["production_ready"] is False
    assert observed["source_checkout"] == str(checkout)
    return development, checkout


def _worker(output):
    """Run only under the installed supervisor; observe actual synthetic spills."""
    import importlib.metadata
    import biominer
    import biominer.polars_runtime as supervisor
    import polars as pl
    import pyarrow

    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    assert versions == VERSIONS
    assert pl.__version__ == VERSIONS["polars"] and pyarrow.__version__ == VERSIONS["pyarrow"]
    assert pl.thread_pool_size() == 4
    assert _reference(supervisor.__file__)["sha256"] == SUPERVISOR_SHA256
    root = Path(os.environ["POLARS_OOC_SPILL_DIR"])
    assert root.is_dir() and not root.is_symlink()
    assert root.stat().st_mode & 0o777 == 0o700
    seen = {}
    maximum_bytes = 0
    observer_errors = []
    stop = threading.Event()

    def observe():
        nonlocal maximum_bytes
        try:
            while not stop.is_set():
                current = 0
                for path in root.rglob("*.ipc"):
                    try:
                        size = path.stat().st_size
                    except FileNotFoundError:
                        continue
                    seen[str(path.relative_to(root))] = size
                    current += size
                maximum_bytes = max(maximum_bytes, current)
                stop.wait(0.001)
        except BaseException as error:
            observer_errors.append(type(error).__name__)

    watcher = threading.Thread(target=observe)
    watcher.start()
    result = {
        "schema": "biominer-polars-resource-observation/v1",
        "versions": versions, "threads": pl.thread_pool_size(),
        "python": sys.executable, "prefix": sys.prefix,
        "application": str(Path(biominer.__file__).resolve()),
        "supervisor": _reference(supervisor.__file__), "probe": _reference(__file__),
        "scratch": str(root), "fixture_rows": 200_000,
        "production_ready": False,
        "limits": "Engine accounting, not RSS/aggregate caps; sampled filesystem bytes are lower bounds",
    }
    try:
        frame = pl.DataFrame({
            "key": [(i * 48271) % 200003 for i in range(200_000)],
            "payload": [str(i % 997) + "x" * 80 for i in range(200_000)],
        })
        sorted_frame = frame.lazy().sort("key").collect(engine="streaming")
        assert sorted_frame.height == 200_000
        assert sorted_frame["key"].is_sorted()
        assert sorted_frame["key"].sum() == 20_000_189_620
        result.update(outcome="passed", key_sum=20_000_189_620)
    except BaseException as error:
        result.update(
            outcome="failed", error_type=type(error).__name__,
            disk_budget_error="POLARS_OOC_DISK_BUDGET_MB" in str(error),
        )
    finally:
        stop.wait(0.05)
        stop.set()
        watcher.join()
        result.update(
            observed_spill_files=len(seen),
            maximum_observed_spill_bytes=maximum_bytes,
            maximum_file_bytes=max(seen.values(), default=0),
            observer_errors=observer_errors,
        )
        _save(output, result)
    return 0 if result["outcome"] == "passed" else 7


def test_polars_engine_spill_and_disk_exhaustion():
    context_path = os.environ.get("BIOMINER_POLARS_RESOURCE_CONTEXT")
    assert context_path, "explicit Polars resource qualification context required"
    context = json.loads(Path(context_path).read_bytes())
    cwd = _validate_sources(context)
    output = native.gate.outside_git(context["output"], context["repo"])
    output.mkdir(mode=0o700)
    development, checkout = _verify(context, output, cwd, "before-verify")
    python = str(Path(development["root"]) / "env/bin/python")
    observations = {}
    scratch_paths = set()
    for label, disk, expected_exit in (("success", "64", 0), ("disk-exhaustion", "16", 7)):
        report_dir = output / f"{label}-report"
        observation = output / f"{label}-observation.json"
        environment = {
            **_environment(output), "POLARS_OOC_MEMORY_BUDGET_MB": "4",
            "POLARS_OOC_DISK_BUDGET_MB": disk, "POLARS_OOC_DRIFT_THRESHOLD": "1",
            "POLARS_OOC_SPILL_MIN_BYTES": "1",
        }
        argv = [
            python, "-I", "-B", "-m", "biominer.polars_runtime",
            "--scratch-parent", str(output), "--report-dir", str(report_dir),
            "--timeout", "60", "--", str(cwd / CASE_MODULE), "--worker", str(observation),
        ]
        _command(argv, output, label, cwd, environment, expected_exit)
        observed = json.loads(observation.read_bytes())
        finished = json.loads((report_dir / "finished.json").read_bytes())
        assert observed["versions"] == VERSIONS and observed["threads"] == 4
        assert observed["python"] == python and observed["prefix"] == str(Path(development["root"]) / "env")
        assert Path(observed["application"]).is_relative_to(checkout / "src/biominer")
        assert observed["supervisor"] == {
            "path": str(checkout / SUPERVISOR_MODULE), "sha256": SUPERVISOR_SHA256,
        }
        assert observed["probe"] == _reference(cwd / CASE_MODULE)
        assert observed["production_ready"] is False
        assert observed["observed_spill_files"] > 0 and observed["maximum_observed_spill_bytes"] > 0
        assert observed["observer_errors"] == []
        scratch = Path(observed["scratch"])
        assert scratch.parent == output and str(scratch) == finished["scratch"]["path"]
        assert not scratch.exists() and not scratch.is_symlink()
        assert str(scratch) not in scratch_paths
        scratch_paths.add(str(scratch))
        assert finished["interpreter"] == python
        assert finished["exit_code"] == expected_exit and finished["worker"]["returncode"] == expected_exit
        assert finished["termination_reason"] == "worker-exit"
        assert finished["cleanup"] == {"status": "removed"}
        assert finished["process_group"]["status"] == "clear"
        assert finished["process_group"]["remaining_descendants_reaped"] is False
        assert finished["policy"]["memory"]["absolute_bytes"] == 4_000_000
        assert finished["policy"]["disk"]["bytes"] == int(disk) * 1_000_000
        assert finished["policy"]["threads"] == {"origin": "explicit", "value": 4}
        if expected_exit:
            assert observed["outcome"] == "failed"
            assert observed["error_type"] == "RuntimeError" and observed["disk_budget_error"] is True
            assert finished["status"] == "failed"
        else:
            assert observed["outcome"] == "passed" and observed["key_sum"] == 20_000_189_620
            assert finished["status"] == "passed"
        observations[label] = {
            "observation": _reference(observation),
            "started": _reference(report_dir / "started.json"),
            "finished": _reference(report_dir / "finished.json"),
            "command": _reference(output / f"{label}-command.json"),
            "expected_worker_exit": expected_exit,
        }
    _verify(context, output, cwd, "after-verify")
    _validate_sources(context)
    _runtime_inputs(context)
    _save(output / "qualification.json", {
        "schema": "biominer-polars-resource-qualification/v1", "status": "passed",
        "tested_commit": context["tested_commit"], "runtime": context["runtime"],
        "sources": context["sources"], "observations": observations,
        "verification": {name: _reference(output / f"{name}-verify-command.json")
                         for name in ("before", "after")},
        "production_ready": False,
        "limits": "Qualified development source and engine behavior; neither production admission nor RSS/aggregate caps",
    })
    print(json.dumps({"qualification": _reference(output / "qualification.json")}))


if __name__ == "__main__":
    assert len(sys.argv) == 3 and sys.argv[1] == "--worker", "explicit synthetic worker invocation required"
    raise SystemExit(_worker(Path(sys.argv[2])))
