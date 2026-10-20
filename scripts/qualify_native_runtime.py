#!/usr/bin/env python3
"""Produce phase-gate native_verify receipts from five real, opt-in cases.

Run after committing the tested sources. --context accepts a phase manifest
or its repo/tested_commit/sources fields plus an optional absolute cwd.
The pytest interpreter needs pytest; all native probes run through the frozen
profile's approved Python and generation. Outputs must be new and outside Git.
This qualifies local native behavior, not hosted PostgreSQL/S3 or workloads.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


_spec = importlib.util.spec_from_file_location(
    "qualification_gate", Path(__file__).with_name("check_upgrade_phase.py")
)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)
CASES = ("artifacts", "loaded_versions", "tls", "import_order", "features")
CASE_MODULE = "scripts/upgrade_checks/test_native_qualification_cases.py"
REQUIRED_SOURCES = {
    "scripts/qualify_native_runtime.py", "scripts/check_upgrade_phase.py",
    CASE_MODULE, "pyproject.toml", "uv.lock",
}


def executable_ref(path):
    """Hash executable bytes without interpreting a binary as textual evidence."""
    path = Path(path).resolve()
    gate.require(path.is_file(), "runtime executable missing")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def source_bytes(path):
    """Read hash-bound source without interpreting expressions as output data."""
    path = Path(path)
    gate.require(path.is_file(), "qualification source is missing")
    return path.read_bytes()


def validate_context(context):
    repo = Path(context["repo"]).resolve()
    commit = context["tested_commit"]
    sources = context["sources"]
    gate.require(isinstance(commit, str) and gate.COMMIT.fullmatch(commit), "full tested commit required")
    gate.require(gate.git(repo, "rev-parse", f"{commit}^{{commit}}") == commit, "tested commit missing")
    gate.require(isinstance(sources, dict) and REQUIRED_SOURCES <= sources.keys(), "qualification source coverage incomplete")
    check = context.get("checks", {}).get("native_verify", {})
    cwd = Path(check.get("cwd", context.get("cwd", repo)))
    gate.require(cwd.is_absolute() and cwd.is_dir(), "absolute execution cwd required")
    cwd = cwd.resolve()
    for name, expected in sources.items():
        path = Path(name)
        gate.require(not path.is_absolute() and ".." not in path.parts, "unsafe source path")
        gate.require(isinstance(expected, str) and gate.SHA.fullmatch(expected), "invalid source hash")
        committed = subprocess.run(
            ["git", "-C", str(repo), "show", f"{commit}:{name}"],
            capture_output=True, timeout=60,
        )
        gate.require(committed.returncode == 0 and gate.digest(committed.stdout) == expected, "committed qualification source mismatch")
        gate.require(gate.digest(source_bytes(cwd / name)) == expected, "execution qualification source mismatch")
    for name in ("qualify_native_runtime.py", "check_upgrade_phase.py"):
        gate.require(
            gate.digest(source_bytes(Path(__file__).with_name(name))) == sources[f"scripts/{name}"],
            "loaded qualification code differs from tested source",
        )
    return cwd


def runtime_configuration(snapshot, manifest_hash, generation):
    snapshot, generation = Path(snapshot).resolve(), Path(generation).resolve()
    manifest_ref = {"path": str(snapshot / "snapshot.json"), "sha256": manifest_hash}
    manifest = gate.ref_json(manifest_ref)
    profile_ref = {
        "path": str(snapshot / "profile.json"),
        "sha256": manifest["files"]["profile.json"],
    }
    profile = gate.ref_json(profile_ref)
    launcher_ref = {
        "path": str(snapshot / "scripts/freeze_native_profile.py"),
        "sha256": manifest["files"]["scripts/freeze_native_profile.py"],
    }
    gate.verify_ref(launcher_ref)
    python_ref = {key: profile["python"][key] for key in ("path", "sha256")}
    gate.require(executable_ref(python_ref["path"])["sha256"] == python_ref["sha256"], "approved Python hash mismatch")
    generation_ref = gate.file_ref(generation / "generation.json")
    receipt = gate.ref_json(generation_ref)
    gate.require(receipt.get("profile_sha256") == profile_ref["sha256"], "generation belongs to another profile")
    return {
        "snapshot": str(snapshot), "snapshot_manifest": manifest_ref,
        "runtime_profile": profile_ref, "launcher": launcher_ref,
        "approved_python": python_ref, "generation": str(generation),
        "generation_receipt": generation_ref,
    }


def test_environment(config):
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "PYTHON", "UV_", "PIP_"))
        and not re.search(r"(?:PASSWORD|PASSWD|TOKEN|SECRET|(?:^|_)KEY(?:_|$)|DSN|CONNECTION_STRING)", key)
    }
    environment.update({
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": "",
        "PYTHONDONTWRITEBYTECODE": "1", "PY_COLORS": "0",
        "BIOMINER_NATIVE_QUALIFICATION_CONTEXT": str(config),
    })
    return environment


def descendants(parent):
    """Return this supervisor's process tree, including adopted session leaders."""
    processes = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                processes[int(entry.name)] = (int(fields[1]), fields[19])
            except (OSError, ValueError, IndexError):
                continue
    found = {}
    parents = {parent}
    while True:
        children = {pid: start for pid, (ppid, start) in processes.items()
                    if ppid in parents and pid not in found}
        if not children:
            return found
        found.update(children)
        parents.update(children)


def signal_owned(pid, start, sig):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[19] == start:
            os.kill(pid, sig)
    except (ProcessLookupError, FileNotFoundError):
        pass


def stop_descendants():
    # Freeze parents before killing them so a new-session child cannot escape
    # the ownership scan by being spawned while its parent is being removed.
    owned = {}
    for _ in range(32):
        current = descendants(os.getpid())
        for pid, start in current.items():
            signal_owned(pid, start, signal.SIGSTOP)
        if current == owned:
            break
        owned = current
    for pid, start in owned.items():
        signal_owned(pid, start, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            time.sleep(0.01)
    raise RuntimeError("native descendant cleanup did not finish")


def supervise(argv, timeout):
    # Linux is part of native-runtime's platform contract. This dedicated
    # process becomes a subreaper without changing the coordinator's process
    # ownership. Orphaned grandchildren remain ours even after pytest exits.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot enable native child supervision")
    process = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
        code = process.returncode
        if descendants(os.getpid()):
            stop_descendants()
            output += b"\nnative case left descendant processes; cleaned up\n"
            code = 125
    except subprocess.TimeoutExpired:
        stop_descendants()
        output, _ = process.communicate(timeout=5)
        output += b"\nnative case timed out; descendants cleaned up\n"
        code = 124
    except BaseException:
        stop_descendants()
        raise
    sys.stdout.buffer.write(output)
    sys.stdout.buffer.flush()
    return code if code >= 0 else 128 - code


def run_test(argv, cwd, environment, timeout):
    supervised = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
         "--_supervise", str(timeout), json.dumps(argv)],
        cwd=cwd, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return subprocess.CompletedProcess(argv, supervised.returncode, supervised.stdout)


def qualify(context, snapshot, manifest_hash, generation, python, output, timeout=900):
    cwd = validate_context(context)
    configuration = runtime_configuration(snapshot, manifest_hash, generation)
    gate.require(type(timeout) is int and 1 <= timeout <= 3600, "case timeout must be 1..3600")
    output = gate.outside_git(output, context["repo"])
    gate.require(not output.exists() and not output.is_symlink(), "immutable qualification output already exists")
    python = Path(python).absolute()
    gate.require(python.is_file() and python.name.startswith("python"), "pytest Python executable required")
    pytest_python = executable_ref(python)
    output.mkdir(mode=0o700)
    configuration["evidence_directory"] = str(output / "native-evidence")
    Path(configuration["evidence_directory"]).mkdir(mode=0o700)
    config_path = output / "runtime-context.json"
    gate.create_file(config_path, gate.canonical(configuration))
    environment = test_environment(config_path)
    report = {
        "schema_version": 1, "kind": "native_verify", "execution": "local_native",
        "runtime_profile": configuration["runtime_profile"], "cases": {},
        "evidence": [gate.file_ref(config_path), configuration["snapshot_manifest"],
                     configuration["generation_receipt"]],
        "limitations": ["Hosted PostgreSQL/S3 unrun.", "Resume uses bounded synthetic Parquet fixtures, not a scientific workload."],
    }
    for case in CASES:
        validate_context(context)
        argv = [
            str(python), "-m", "pytest", "-q", "-s", "--color=no", "--noconftest",
            "-p", "no:cacheprovider", "-c", "/dev/null", "--rootdir", str(cwd),
            f"{cwd / CASE_MODULE}::test_{case}",
        ]
        timed_out = False
        try:
            result = run_test(argv, cwd, environment, timeout)
            log_bytes, exit_code = result.stdout, result.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            log_bytes, exit_code = (exc.stdout or b"") + b"\nqualification case timed out\n", 124
        log = output / f"{case}.log"
        gate.create_file(log, log_bytes)
        receipt = {
            "schema_version": 1, "check": f"native_verify.{case}", "argv": argv,
            "cwd": str(cwd), "exit_code": exit_code, "tested_commit": context["tested_commit"],
            "source_digest": gate.source_digest(context["sources"]),
            "runtime_profile": configuration["runtime_profile"], "log": gate.file_ref(log),
            "pytest_python": pytest_python,
        }
        receipt_path = output / f"{case}.result.json"
        gate.create_file(receipt_path, gate.canonical(receipt))
        gate.require(not timed_out and type(exit_code) is int and exit_code == 0, "native case failed; evidence preserved")
        outcomes = gate.pytest_outcomes(log_bytes)
        gate.require(outcomes["passed"] == 1, "each native case must execute exactly one test")
        observed = Path(configuration["evidence_directory"]) / f"{case}.json"
        observation = gate.load_json(observed)
        gate.require(
            observation.get("case") == case
            and observation.get("runtime_profile") == configuration["runtime_profile"]
            and observation.get("generation_receipt") == configuration["generation_receipt"],
            "native observation binding mismatch",
        )
        report["evidence"].append(gate.file_ref(observed))
        report["cases"][case] = {"receipt": gate.file_ref(receipt_path)}
        validate_context(context)
        gate.require(executable_ref(python) == pytest_python, "pytest executable changed during qualification")
        gate.require(
            runtime_configuration(snapshot, manifest_hash, generation) == {
                key: value for key, value in configuration.items() if key != "evidence_directory"
            }, "runtime evidence changed during qualification",
        )
    observations = {f"{case}.json" for case in CASES}
    for path in sorted(Path(configuration["evidence_directory"]).rglob("*")):
        gate.require(not path.is_symlink(), "native evidence cannot contain symlinks")
        if path.is_file() and path.name not in observations:
            report["evidence"].append(gate.file_ref(path))
    gate.verify_report(
        gate.canonical(report), "native_verify", 2, context=context,
        check={"cwd": str(cwd), "runtime_profile": configuration["runtime_profile"]},
    )
    gate.create_file(output / "native_verify.json", gate.canonical(report))
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--_supervise"]:
        return supervise(json.loads(argv[2]), int(argv[1]))
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    try:
        result = qualify(
            gate.load_json(args.context), args.snapshot, args.snapshot_sha256,
            args.generation, args.python, args.output, args.timeout,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        # Avoid reflecting arbitrary external exception text into phase logs.
        print(json.dumps({"status": "blocked", "ready": False, "error": type(exc).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
