"""Supervise one Python run with Polars policy established before imports.

Run ``python -I -B -m biominer.polars_runtime -- -m biominer.cli ...`` with
the permanent environment interpreter. ``POLARS_OOC_SPILL_DIR`` denotes a
requested parent: each invocation creates and removes its own private leaf.

These are per-process Polars engine budgets, not RSS or aggregate job limits.
Ordinary Python/module/library entrypoints are covered only when wrapped.
Nested wrappers allocate separate leaves; inherited explicit budgets remain.
Linux subreaper supervision includes detached descendants. Callers must have
one Python thread and no existing children; concurrent child creation is not
supported. The supervisor cannot clean up after its own SIGKILL or host failure.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time


THREADS = "POLARS_MAX_THREADS"
MEMORY = "POLARS_OOC_MEMORY_BUDGET_MB"
FRACTION = "POLARS_OOC_MEMORY_BUDGET_FRACTION"
DISK = "POLARS_OOC_DISK_BUDGET_MB"
SPILL = "POLARS_OOC_SPILL_DIR"
MILLION = 1_000_000
DEFAULT_MEMORY_BYTES = 1024**3
DEFAULT_DISK_BYTES = 4 * 1024**3


class PolicyError(ValueError):
    """Invalid policy rejected before starting a worker."""


def _positive_integer(environment, key, default=None):
    value = environment.get(key, default)
    limit = (2**64 - 1) // (MILLION if key in {MEMORY, DISK} else 1)
    if not isinstance(value, str) or not re.fullmatch(r"\+?[0-9]+", value.strip()):
        raise PolicyError(f"{key} must be a positive integer")
    parsed = int(value)
    if not 0 < parsed <= limit:
        raise PolicyError(f"{key} is outside the supported positive range")
    environment[key] = str(parsed)
    return parsed


def _policy(source):
    environment = dict(source)
    has_memory, has_fraction = MEMORY in environment, FRACTION in environment
    threads = _positive_integer(environment, THREADS, "4")
    disk = _positive_integer(environment, DISK, str(DEFAULT_DISK_BYTES // MILLION))
    memory = None
    if has_memory or not has_fraction:
        memory = _positive_integer(environment, MEMORY, str(DEFAULT_MEMORY_BYTES // MILLION))
    fraction = 0.8
    if has_fraction:
        try:
            fraction = float(environment[FRACTION])
        except (TypeError, ValueError):
            raise PolicyError(f"{FRACTION} must be a finite fraction in (0, 1]") from None
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise PolicyError(f"{FRACTION} must be a finite fraction in (0, 1]")
        environment[FRACTION] = str(fraction)
    policy = {
        "threads": {"value": threads, "origin": "explicit" if THREADS in source else "default"},
        "memory": {
            "origin": "explicit" if has_memory or has_fraction else "default",
            "absolute_bytes": memory * MILLION if memory is not None else None,
            "fraction": fraction, "fraction_origin": "explicit" if has_fraction else "upstream-default",
            "requested_default_bytes": None if has_memory or has_fraction else DEFAULT_MEMORY_BYTES,
            "semantics": "minimum of absolute bytes (when set) and fraction times Polars total memory",
        },
        "disk": {
            "bytes": disk * MILLION, "origin": "explicit" if DISK in source else "default",
            "requested_default_bytes": None if DISK in source else DEFAULT_DISK_BYTES,
        },
        "scope": "per-process Polars engine accounting; neither RSS nor aggregate job limits",
    }
    return environment, policy


def _directory(path):
    path = Path(path).absolute()
    if path.resolve() != path or not path.is_dir():
        raise PolicyError("scratch/report parent must be an existing canonical directory")
    return path


def _identity(path):
    info = path.lstat()
    return {"device": info.st_dev, "inode": info.st_ino, "uid": info.st_uid}


def _json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _remove_owned_scratch(root, expected):
    info = root.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or _identity(root) != expected):
        raise PolicyError("scratch identity changed; cleanup refused")
    if not shutil.rmtree.avoids_symlink_attacks:
        raise PolicyError("safe directory cleanup is unavailable on this platform")
    shutil.rmtree(root)


def _live_group(pgid):
    members = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                members.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return members


def _signal_group(pgid, signum):
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


def _finish_group(pgid, grace):
    if not _live_group(pgid):
        return {"status": "clear", "remaining_workers_terminated": False}
    _signal_group(pgid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while _live_group(pgid) and time.monotonic() < deadline:
        time.sleep(0.02)
    if _live_group(pgid):
        _signal_group(pgid, signal.SIGKILL)
        deadline = time.monotonic() + grace
        while _live_group(pgid) and time.monotonic() < deadline:
            time.sleep(0.02)
    return {"status": "failed" if _live_group(pgid) else "clear",
            "remaining_workers_terminated": True}


def _descendants(parent):
    processes = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            processes[int(directory.name)] = (int(fields[1]), fields[19], int(fields[2]))
        except (FileNotFoundError, ProcessLookupError):
            pass
    owned = {}
    parents = {parent}
    while parents:
        children = {pid for pid, (ppid, _, _) in processes.items() if ppid in parents}
        owned.update({pid: processes[pid][1:] for pid in children})
        parents = children
    return owned


def _subreaper(value=None):
    libc = ctypes.CDLL(None, use_errno=True)
    if value is None:
        observed = ctypes.c_int()
        status = libc.prctl(37, ctypes.byref(observed), 0, 0, 0)  # PR_GET_CHILD_SUBREAPER
        result = observed.value
    else:
        status = libc.prctl(36, value, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
        result = value
    if status:
        raise OSError(ctypes.get_errno(), "child supervision unavailable")
    return result


def _signal_owned(pid, start, signum):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[19] == start:
            os.kill(pid, signum)
    except (FileNotFoundError, ProcessLookupError):
        pass


def _reap_descendants(grace):
    # Freeze parents before killing so a setsid grandchild cannot race out of
    # the ownership scan. This process was admitted with no existing children.
    owned = {}
    for _ in range(32):
        current = _descendants(os.getpid())
        for pid, (start, _) in current.items():
            _signal_owned(pid, start, signal.SIGSTOP)
        if current == owned:
            break
        owned = current
    for pid, (start, _) in owned.items():
        _signal_owned(pid, start, signal.SIGKILL)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            time.sleep(0.01)
    raise PolicyError("descendant cleanup did not finish")


def _python():
    # Resolve a directory alias without resolving env/bin/python to its base
    # interpreter, which would discard virtual-environment package identity.
    if sys.prefix != sys.base_prefix:
        candidate = Path(sys.prefix).resolve() / "bin/python"
        if candidate.is_file():
            return str(candidate)
    return str(Path(sys.executable).absolute())


def run(args, *, environment=None, scratch_parent=None, report_dir=None,
        timeout=None, grace_seconds=2.0):
    """Run Python arguments in a fresh process; return a durable outcome receipt.

    Worker stdin/stdout/stderr are inherited. Receipts contain no raw argv,
    arbitrary environment values, worker output, or exception messages.
    """
    if (sys.platform != "linux" or threading.current_thread() is not threading.main_thread()
            or threading.active_count() != 1):
        raise PolicyError("supervision requires a single Python thread on the Linux main thread")
    if _descendants(os.getpid()):
        raise PolicyError("supervision refuses a caller with existing descendants")
    if not args or any(not isinstance(arg, str) for arg in args):
        raise PolicyError("worker Python arguments are required")
    if (not math.isfinite(grace_seconds) or grace_seconds <= 0
            or timeout is not None and (not math.isfinite(timeout) or timeout <= 0)):
        raise PolicyError("timeouts must be finite and positive")
    source = dict(os.environ if environment is None else environment)
    env, policy = _policy(source)
    if SPILL in source and not source[SPILL].strip():
        raise PolicyError(f"{SPILL} must name a scratch parent")
    parent = _directory(source.get(SPILL, scratch_parent or "/var/tmp"))
    policy["spill_parent_origin"] = "environment" if SPILL in source else (
        "argument" if scratch_parent is not None else "default")
    if report_dir is not None:
        report_dir = Path(report_dir).absolute()
        _directory(report_dir.parent)
        if report_dir.exists() or report_dir.is_symlink():
            raise PolicyError("report directory already exists")
    # A nested wrapper inherits its parent's spill leaf. Default reports must
    # live outside that leaf so outer cleanup cannot delete inner evidence.
    report_parent = _directory(source.get("TMPDIR", "/var/tmp")) if report_dir is None else report_dir.parent
    root = Path(tempfile.mkdtemp(prefix="biominer-polars-", dir=parent))
    expected = _identity(root)
    report_dir = report_dir or report_parent / (root.name + ".report")
    try:
        report_dir.mkdir(mode=0o700)
    except BaseException:
        _remove_owned_scratch(root, expected)
        raise
    env[SPILL] = str(root)
    argv = [_python(), "-I", "-B", *args]
    initial = {
        "schema": "biominer-polars-run/v1", "status": "started", "policy": policy,
        "scratch": {"path": str(root), "identity": expected}, "report_dir": str(report_dir),
        "command_sha256": hashlib.sha256(json.dumps(argv).encode()).hexdigest(),
        "interpreter": argv[0], "supervisor_pid": os.getpid(),
        "limitations": "Supervisor SIGKILL/host failure can orphan scratch; engine budgets are not aggregate/RSS caps",
    }
    try:
        _json(report_dir / "started.json", initial)
    except BaseException:
        _remove_owned_scratch(root, expected)
        raise
    process = None
    requested_signal = None
    signal_time = None
    reason = "worker-exit"
    old_handlers = {}
    old_subreaper = None
    error_type = None
    group = {"status": "not-started", "remaining_workers_terminated": False}
    result = {**initial, "status": "failed"}

    def handle(signum, _frame):
        nonlocal requested_signal, signal_time
        if requested_signal is None:
            requested_signal, signal_time = signum, time.monotonic()
        if process is not None:
            _signal_group(process.pid, signum)

    try:
        old_subreaper = _subreaper()
        _subreaper(1)
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, handle)
        process = subprocess.Popen(argv, env=env, start_new_session=True)
        if requested_signal is not None:
            _signal_group(process.pid, requested_signal)
        deadline = time.monotonic() + timeout if timeout is not None else None
        while process.poll() is None:
            now = time.monotonic()
            if requested_signal is not None:
                if reason != "timeout":
                    reason = f"signal:{requested_signal}"
                if now - signal_time >= grace_seconds:
                    _signal_group(process.pid, signal.SIGKILL)
            elif deadline is not None and now >= deadline:
                reason = "timeout"
                requested_signal, signal_time = signal.SIGTERM, now
                _signal_group(process.pid, signal.SIGTERM)
                deadline = None
            try:
                process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                pass
        if reason != "timeout" and requested_signal is not None:
            reason = f"signal:{requested_signal}"
    except BaseException as exc:
        error_type = type(exc).__name__
        reason = "supervisor-error"
        if process is not None:
            _signal_group(process.pid, signal.SIGKILL)
            process.wait()
    finally:
        try:
            if process is not None:
                group = _finish_group(process.pid, grace_seconds)
                remaining = _descendants(os.getpid())
                group["detached_descendants_terminated"] = any(
                    pgid != process.pid for _, pgid in remaining.values())
                group["remaining_descendants_reaped"] = bool(remaining)
                if remaining:
                    _reap_descendants(grace_seconds)
        except (OSError, ValueError) as exc:
            group = {"status": "failed", "error_type": type(exc).__name__}
        for signum, previous in old_handlers.items():
            signal.signal(signum, previous)
        if old_subreaper is not None:
            _subreaper(old_subreaper)
    cleanup = {"status": "not-attempted-active-workers"}
    if group["status"] in {"clear", "not-started"}:
        try:
            _remove_owned_scratch(root, expected)
            cleanup = {"status": "removed"}
        except (OSError, ValueError) as exc:
            cleanup = {"status": "failed", "error_type": type(exc).__name__}
    returncode = process.returncode if process is not None else None
    exit_code = (128 - returncode if returncode < 0 else returncode) if returncode is not None else 125
    if reason == "timeout":
        exit_code = 124
    elif reason.startswith("signal:"):
        exit_code = 128 + int(reason.split(":")[1])
    if (error_type or cleanup["status"] != "removed" or group["status"] == "failed"
            or group.get("remaining_descendants_reaped")):
        exit_code = 125
    result.update(
        status="passed" if exit_code == 0 else "failed", exit_code=exit_code,
        worker={"pid": process.pid if process else None, "returncode": returncode},
        termination_reason=reason, process_group=group, cleanup=cleanup,
        error_type=error_type,
    )
    _json(report_dir / "finished.json", result)
    return result


def cli_main():
    """Console bootstrap for installation in a newly qualified environment."""
    result = run(["-m", "biominer.cli", *sys.argv[1:]])
    print(json.dumps({"resource_report": str(Path(result["report_dir"]) / "finished.json"),
                      "status": result["status"]}), file=sys.stderr)
    return result["exit_code"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--scratch-parent", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--grace-seconds", type=float, default=2.0)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    values = parser.parse_args(argv)
    args = values.args[1:] if values.args[:1] == ["--"] else values.args
    try:
        result = run(args, scratch_parent=values.scratch_parent, report_dir=values.report_dir,
                     timeout=values.timeout, grace_seconds=values.grace_seconds)
    except (PolicyError, OSError) as exc:
        print(json.dumps({"status": "rejected", "error_type": type(exc).__name__,
                          "reason": str(exc) if isinstance(exc, PolicyError) else "run setup failed"}),
              file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"],
                      "resource_report": str(Path(result["report_dir"]) / "finished.json")}), file=sys.stderr)
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
