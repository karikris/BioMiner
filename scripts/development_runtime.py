#!/usr/bin/env python3
"""Create an immutable dependency environment with an explicitly editable checkout.

This is a development runtime, never a production generation admission. Build at
its permanent path; scripts and subprocesses must use ROOT/env/bin/python.
Activation is an explicit, stopped-consumer operation. It preserves the old
environment under an inaccessible quarantine and never rolls back to those
unqualified bytes. Read-only modes prevent accidental uv replacement, not a
malicious owner deliberately changing permissions.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tomllib
import types


SCHEMA = "biominer-development-runtime/v1"
RECEIPT = "development.json"
NATIVE = {"psycopg-c", "pyarrow", "duckdb", "lxml"}
BACKEND = "setuptools==84.2.0"
SHA = re.compile(r"[0-9a-f]{64}")


class DevelopmentError(ValueError):
    """Development runtime evidence is incomplete or changed."""


def require(condition, reason):
    if not condition:
        raise DevelopmentError(reason)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def payload(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def create_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(payload(value))
        stream.flush()
        os.fsync(stream.fileno())


def identity(path):
    path = Path(path)
    info = path.lstat()
    return {"device": info.st_dev, "inode": info.st_ino,
            "uid": info.st_uid, "mode": stat.S_IMODE(info.st_mode),
            "symlink": os.readlink(path) if path.is_symlink() else None}


def inventory(root):
    """Do not follow directory links or chmod external interpreter targets."""
    root = Path(root)
    result = {}
    for path in [root, *sorted(root.rglob("*"))]:
        name = str(path.relative_to(root))
        if name == RECEIPT:
            continue
        info = path.lstat()
        row = {"mode": stat.S_IMODE(info.st_mode), "uid": info.st_uid}
        if stat.S_ISLNK(info.st_mode):
            row.update(kind="symlink", target=os.readlink(path))
            if path.is_file():
                row["target_sha256"] = digest(path)
            elif path.exists():
                require(path.resolve().is_relative_to(root.resolve()),
                        "external directory symlink in dependency tree")
        elif stat.S_ISDIR(info.st_mode):
            row["kind"] = "directory"
        else:
            require(stat.S_ISREG(info.st_mode), "special file in dependency tree")
            row.update(kind="file", sha256=digest(path), size=info.st_size)
        result[name] = row
    return result


def freeze(root):
    root = Path(root)
    for path in [*sorted(root.rglob("*"), reverse=True), root]:
        info = path.lstat()
        require(info.st_uid == os.getuid(), "dependency tree contains another owner's file")
        if stat.S_ISLNK(info.st_mode):
            continue
        require(stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode),
                "dependency tree contains a special file")
        path.chmod(0o555 if stat.S_ISDIR(info.st_mode) or info.st_mode & 0o111 else 0o444)


def _load_verified(path, expected):
    data = path.read_bytes()
    require(isinstance(expected, str) and SHA.fullmatch(expected)
            and hashlib.sha256(data).hexdigest() == expected, "snapshot launcher hash mismatch")
    module = types.ModuleType("approved_development_dependency_launcher")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def snapshot_context(snapshot, expected):
    snapshot = Path(snapshot).absolute()
    manifest = snapshot / "snapshot.json"
    require(isinstance(expected, str) and SHA.fullmatch(expected)
            and digest(manifest) == expected, "snapshot manifest hash mismatch")
    doc = json.loads(manifest.read_text())
    freezer = _load_verified(snapshot / "scripts/freeze_native_profile.py",
                             doc["files"]["scripts/freeze_native_profile.py"])
    freezer.verify_snapshot(snapshot, expected)
    native = _load_verified(snapshot / "scripts/native_runtime.py",
                           doc["files"]["scripts/native_runtime.py"])
    profile = snapshot / "profile.json"
    profile_doc, profile = native.load_profile(profile)
    native._check_inputs(profile_doc, profile.parent)
    require({row["name"] for row in profile_doc["artifacts"]} == NATIVE | {"biominer"},
            "unexpected native artifact set")
    return native, profile_doc, profile


def environment(native, root):
    result = native.launch_environment()
    result.update(UV_CACHE_DIR=str(root / "cache"), UV_LINK_MODE="copy",
                  UV_PROJECT_ENVIRONMENT=str(root / "env"), UV_PYTHON_DOWNLOADS="never")
    return result


def command(args, env, records, *, cwd=None, timeout=600):
    """Retain exact non-secret argv and output hashes, never raw installer output."""
    process = subprocess.Popen(
        [str(arg) for arg in args], env=env, cwd=cwd, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise
    records.append({
        "argv": [str(arg) for arg in args], "returncode": process.returncode,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
    })
    require(process.returncode == 0, f"development command failed ({len(records)}, exit {process.returncode})")
    return stdout


IDENTITY_PROBE = """
import importlib.metadata as m,json,pathlib,subprocess,sys
import biominer
child=json.loads(subprocess.check_output([sys.executable,'-I','-B','-c',
    'import json,sys;print(json.dumps([sys.executable,sys.prefix]))'],text=True))
d=m.distribution('biominer')
print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,
    'child':child,'module':str(pathlib.Path(biominer.__file__).resolve()),
    'direct_url':json.loads(d.read_text('direct_url.json')),
    'packages':sorted((x.metadata['Name'],x.version) for x in m.distributions())}))
"""


def runtime_checks(native, doc, profile, root, checkout, records):
    env = environment(native, root)
    # Verification must not recreate a cache inside the frozen root.
    env["UV_NO_CACHE"] = "1"
    native._host(doc)
    native._python(doc, profile.parent)
    for item in doc["native_files"]:
        native.verify_native_file(item, profile, root)
    # Only the application is editable. The four approved native wheels retain
    # the exact production wheel-member verification, without changing its policy.
    native._installed_wheels(
        {**doc, "artifacts": [row for row in doc["artifacts"] if row["name"] in NATIVE]},
        profile, root,
    )
    checks = []
    for check in doc["checks"]:
        script = native._file(check["script"], profile.parent, "runtime check")
        args = [arg.replace("{generation}", str(root)).replace("{profile}", str(profile))
                for arg in check["args"]]
        raw = command([root / "env/bin/python", "-I", "-B", script, *args], env, records)
        observed = json.loads(raw)
        require(observed.get("status") == "passed" and observed.get("profile_sha256") == digest(profile),
                f"development runtime check failed: {check['name']}")
        if check["name"] == "extension_policy":
            require(observed.get("policy_id") == doc["extension_policy"]["policy_id"],
                    "extension policy differs")
        checks.append(check["name"])
    require(native.INSTALL_CHECKS <= set(checks), "mandatory development checks missing")
    uv = native._file(doc["uv"], profile.parent, "uv")
    command([uv, "--no-config", "pip", "check", "--python", root / "env/bin/python"], env, records)
    observed = json.loads(command(
        [root / "env/bin/python", "-I", "-B", "-c", IDENTITY_PROBE], env, records,
    ))
    expected = [str(root / "env/bin/python"), str(root / "env")]
    require([observed["executable"], observed["prefix"]] == expected
            and observed["child"] == expected, "permanent interpreter identity differs")
    require(Path(observed["module"]).is_relative_to(checkout / "src")
            and observed["direct_url"].get("url") == checkout.as_uri()
            and observed["direct_url"].get("dir_info", {}).get("editable") is True,
            "BioMiner import is not the intended editable checkout")
    return {"checks": checks, "identity": observed}


def _new_directory(path):
    path = Path(path).absolute()
    require(not path.exists() and not path.is_symlink(), "output already exists")
    require(path.parent.is_dir() and path.parent.resolve() == path.parent
            and path.parent.stat().st_uid == os.getuid(),
            "output parent must be canonical, existing and owned")
    path.mkdir(mode=0o700)
    return path


def build(snapshot, snapshot_sha256, checkout, root):
    builder_sha256 = digest(Path(__file__))
    root = Path(root).absolute()
    require(not root.exists() and not root.is_symlink(), "output already exists")
    checkout = Path(checkout).resolve()
    native, doc, profile = snapshot_context(snapshot, snapshot_sha256)
    for name, item in (("pyproject.toml", doc["project"]), ("uv.lock", doc["base_lock"])):
        require(digest(checkout / name) == item["sha256"], f"checkout {name} differs from frozen baseline")
    project_doc = tomllib.loads((checkout / "pyproject.toml").read_text())
    require(project_doc.get("build-system") == {
        "requires": ["setuptools>=84.2"], "build-backend": "setuptools.build_meta"},
        "editable build backend requires a separately reviewed exact constraint")
    python = native._python(doc, profile.parent)
    uv = native._file(doc["uv"], profile.parent, "uv")
    native._host(doc)
    _new_directory(root)
    records = []
    try:
        project = root / "project"
        project.mkdir()
        for name, item in (("pyproject.toml", doc["project"]), ("uv.lock", doc["base_lock"])):
            shutil.copyfile(native._file(item, profile.parent, name), project / name)
        constraints = root / "build-constraints.txt"
        constraints.write_text(BACKEND + "\n")
        env = environment(native, root)
        command([uv, "--no-config", "venv", "--python", python, root / "env"], env, records)
        args = [uv, "--no-config", "sync", "--locked", "--no-editable", "--no-install-project"]
        for extra in ("postgres", "geo", "ml", "test"):
            args += ["--extra", extra]
        for name in sorted(NATIVE):
            args += ["--no-install-package", name]
        command([*args, "--project", project], env, records, timeout=3600)
        requirements = root / "native-wheels.txt"
        requirements.write_text("".join(
            f'{native._file(row, profile.parent, "native wheel").as_uri()} --hash=sha256:{row["sha256"]}\n'
            for row in doc["artifacts"] if row["name"] in NATIVE
        ))
        command([uv, "--no-config", "pip", "install", "--python", root / "env/bin/python",
                 "--no-index", "--no-deps", "--require-hashes", "-r", requirements], env, records)
        command([uv, "--no-config", "pip", "install", "--python", root / "env/bin/python",
                 "--no-deps", "--build-constraints", constraints, "--editable", checkout], env, records)
        observed = runtime_checks(native, doc, profile, root, checkout, records)
        if (root / "cache").is_dir() and not (root / "cache").is_symlink():
            shutil.rmtree(root / "cache")
        receipt = {
            "schema": SCHEMA, "status": "built-development", "production_ready": False,
            "root": str(root), "source_checkout": str(checkout), "source_is_editable": True,
            "live_services": "not_attested", "script_sha256": builder_sha256,
            "snapshot": str(Path(snapshot).absolute()), "snapshot_sha256": snapshot_sha256,
            "profile_sha256": digest(profile), "backend_constraint": BACKEND,
            "source_configuration": {name: digest(checkout / name) for name in ("pyproject.toml", "uv.lock")},
            "installer_policy": {"link_mode": "copy", "cache": "private-removed-before-freeze",
                                 "python_downloads": "never", "user_configuration": "disabled"},
            "commands": records, **observed,
        }
        # Inventory describes final modes; the receipt is published last and is
        # separately pinned by its caller, avoiding a circular self-hash.
        freeze(root)
        receipt["inventory"] = inventory(root)
        root.chmod(0o700)
        create_json(root / RECEIPT, receipt)
        (root / RECEIPT).chmod(0o444)
        root.chmod(0o555)
        return verify(root, digest(root / RECEIPT))
    except BaseException:
        if root.is_dir() and os.access(root, os.W_OK) and not (root / "FAILED.json").exists():
            create_json(root / "FAILED.json", {"status": "failed-development", "production_ready": False,
                                               "commands": records})
        raise


def verify(root, receipt_sha256):
    root = Path(root).absolute()
    receipt = root / RECEIPT
    require(isinstance(receipt_sha256, str) and SHA.fullmatch(receipt_sha256)
            and digest(receipt) == receipt_sha256, "development receipt hash mismatch")
    doc = json.loads(receipt.read_text())
    require(doc.get("schema") == SCHEMA and doc.get("status") == "built-development"
            and doc.get("root") == str(root) and doc.get("production_ready") is False
            and doc.get("source_is_editable") is True, "development receipt contract differs")
    require(not root.is_symlink() and root.resolve() == root, "development root must retain its permanent path")
    require(inventory(root) == doc["inventory"], "development dependency inventory changed")
    for path in [root, *root.rglob("*")]:
        info = path.lstat()
        require(info.st_uid == os.getuid() and (path.is_symlink() or not info.st_mode & 0o222),
                "development tree is not owned and read-only")
    checkout = Path(doc["source_checkout"])
    require(checkout.resolve() == checkout, "editable checkout path changed")
    for name, expected in doc["source_configuration"].items():
        require(name in {"pyproject.toml", "uv.lock"} and digest(checkout / name) == expected,
                "editable dependency configuration changed; build a new version")
    native, profile_doc, profile = snapshot_context(doc["snapshot"], doc["snapshot_sha256"])
    require(digest(profile) == doc["profile_sha256"], "development profile changed")
    records = []
    observed = runtime_checks(native, profile_doc, profile, root, checkout, records)
    require(observed == {"checks": doc["checks"], "identity": doc["identity"]},
            "development runtime observation changed")
    return {"status": "verified-development", "production_ready": False,
            "root": str(root), "source_checkout": str(checkout),
            "receipt_sha256": receipt_sha256, "checks": observed["checks"],
            "live_services": "not_attested", "source_is_editable": True}


def _references(value, entry, cwd):
    if not value or any(char.isspace() for char in value) or "\0" in value:
        return False
    path = Path(value)
    if not path.is_absolute():
        if "/" not in value:
            return False
        path = cwd / path
    absolute = Path(os.path.abspath(path))
    return absolute.is_relative_to(entry) or absolute.resolve().is_relative_to(entry.resolve())


def _systemd_observation(entry):
    """Independent service-manager evidence, with a fixed local D-Bus endpoint."""
    controller = Path("/usr/bin/systemctl")

    def root_file(path):
        path = Path(path)
        info = path.stat()
        require(path.is_file() and info.st_uid == 0 and not info.st_mode & 0o022,
                "systemd bootstrap evidence file is not root-controlled")
        return {"path": str(path), "sha256": digest(path)}

    controller_record = root_file(controller)
    uid = os.getuid()
    env = {
        "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
        "XDG_RUNTIME_DIR": f"/run/user/{uid}",
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{uid}/bus",
    }

    def query(args):
        result = subprocess.run([str(controller), *args], env=env, capture_output=True, timeout=15)
        require(result.returncode == 0, "systemd bootstrap observation unavailable")
        return result.stdout.decode()

    unit = f"user@{uid}.service"
    properties = dict(line.split("=", 1) for line in query([
        "show", unit, "--property=MainPID,ControlGroup,ExecStart,User,FragmentPath,DropInPaths",
    ]).splitlines() if "=" in line)
    require(properties["User"] == str(uid), "systemd bootstrap service owner differs")
    executable = re.search(r"(?:^|[ {])path=([^ ;]+) ;", properties["ExecStart"])
    require(executable is not None, "systemd bootstrap executable missing")
    executable = executable[1]
    require(executable == "/usr/lib/systemd/systemd"
            and f"argv[]={executable} --user ;" in properties["ExecStart"],
            "systemd bootstrap executable or arguments differ")
    files = [root_file(executable), root_file(properties["FragmentPath"])]
    files += [root_file(path) for path in properties["DropInPaths"].split()]
    selected = []
    for line in query(["--user", "show-environment"]).splitlines():
        key, sep, value = line.partition("=")
        if sep and key in {"VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"}:
            # systemctl may quote values; any alias text also fails closed.
            selected.append({
                "key": key, "references_entry": str(entry) in value or ".venv" in value
                or any(_references(part.strip("'\""), entry, entry.parent) for part in value.split(":")),
            })
    return {
        "unit": unit, "main_pid": int(properties["MainPID"]), "uid": uid,
        "control_group": properties["ControlGroup"], "executable": executable,
        "controller_sha256": controller_record["sha256"],
        "unit_files": files, "manager_environment": sorted(selected, key=lambda row: row["key"]),
    }


def _bootstrap_process(directory):
    status = (directory / "status").read_text()
    owner = int(re.search(r"^Uid:\s+(\d+)", status, re.M)[1])
    parent = int(re.search(r"^PPid:\s+(\d+)", status, re.M)[1])
    fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    require(int(fields[1]) == parent, "bootstrap parent identity changed")
    args = [arg.decode() for arg in (directory / "cmdline").read_bytes().split(b"\0") if arg]
    return {
        "pid": int(directory.name), "parent_pid": parent, "uid": owner,
        "start_ticks": int(fields[19]), "cgroup": (directory / "cgroup").read_text().strip(),
        "argv": args,
    }


def bootstrap_scope(entry, *, proc_root=Path("/proc")):
    """Opt-in scope exclusion of two exact bootstrap identities, never children."""
    entry, proc_root = Path(entry).absolute(), Path(proc_root)
    observation = _systemd_observation(entry)
    require(not any(row["references_entry"] for row in observation["manager_environment"]),
            "systemd manager environment references the old environment")
    main_pid = observation["main_pid"]
    require(main_pid > 1, "bootstrap MainPID is unavailable")
    manager = _bootstrap_process(proc_root / str(main_pid))
    expected_cgroup = "0::" + observation["control_group"] + "/init.scope"
    require(manager["parent_pid"] == 1 and manager["uid"] == os.getuid()
            and manager["cgroup"] == expected_cgroup
            and manager["argv"] == [observation["executable"], "--user"],
            "bootstrap manager identity differs")
    helpers = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit() or directory.name == str(main_pid):
            continue
        try:
            status = (directory / "status").read_text()
            parent = re.search(r"^PPid:\s+(\d+)", status, re.M)
            if not parent or int(parent[1]) != main_pid:
                continue
            helper = _bootstrap_process(directory)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if helper["argv"] == ["(sd-pam)"]:
            require(helper["uid"] == os.getuid() and helper["cgroup"] == expected_cgroup,
                    "bootstrap helper ownership or cgroup differs")
            helpers.append(helper)
    require(len(helpers) == 1, "bootstrap requires exactly one identified PAM helper")
    processes = []
    for role, process in (("user-manager", manager), ("pam-helper", helpers[0])):
        process = process.copy()
        process["argv_sha256"] = hashlib.sha256(payload(process.pop("argv"))).hexdigest()
        processes.append({"role": role, **process})
    return {
        "schema": "biominer-bootstrap-scope/v1", "service": observation, "processes": processes,
        "limitation": "Private bootstrap maps/environment were not inspected; exact bootstrap identities only. "
        "This excludes no spawned services, queued jobs or future launches. Launchers must remain stopped.",
    }


def inspect_consumers(entry, *, proc_root=Path("/proc"), bootstrap_scope_sha256=None):
    """Report evidence categories only; never return command lines or env values.

    Other users' processes are explicitly outside readable coverage. Unreadable
    same-user processes fail closed. This is a point-in-time check, not a scheduler
    lock: operators must first stop launchers that could create new consumers.
    """
    entry, proc_root = Path(entry).absolute(), Path(proc_root)
    consumers, unknown = [], []
    other_users = 0
    scope = None
    if bootstrap_scope_sha256 is not None:
        scope = bootstrap_scope(entry, proc_root=proc_root)
        require(hashlib.sha256(payload(scope)).hexdigest() == bootstrap_scope_sha256,
                "bootstrap scope identity or evidence changed")
    excluded = {row["pid"] for row in scope["processes"]} if scope else set()
    for directory in sorted(proc_root.iterdir()):
        if not directory.name.isdigit() or int(directory.name) in excluded | {os.getpid()}:
            continue
        pid = int(directory.name)
        try:
            status = (directory / "status").read_text()
            owner = int(re.search(r"^Uid:\s+(\d+)", status, re.M)[1])
            if owner != os.getuid():
                other_users += 1
                continue
            reasons = set()
            cwd = Path(os.readlink(directory / "cwd"))
            if _references(str(cwd), entry, cwd):
                reasons.add("working_directory")
            executable = directory / "exe"
            if executable.is_symlink() and _references(os.readlink(executable), entry, cwd):
                reasons.add("executable")
            argv = (directory / "cmdline").read_bytes().decode(errors="replace").split("\0")
            skip_source = False
            for arg in argv:
                if skip_source:
                    skip_source = False
                    continue
                if arg in {"-c", "--command"}:
                    skip_source = True
                    continue
                value = arg.partition("=")[2] if arg.startswith("-") and "=" in arg else arg
                if _references(value, entry, cwd):
                    reasons.add("argument")
            for item in (directory / "environ").read_bytes().split(b"\0"):
                key, sep, value = item.partition(b"=")
                if sep and key in {b"VIRTUAL_ENV", b"PYTHONHOME", b"PYTHONPATH"}:
                    if any(_references(part.decode(errors="replace"), entry, cwd) for part in value.split(b":")):
                        reasons.add("environment:" + key.decode())
            for line in (directory / "maps").read_text().splitlines():
                parts = line.split(maxsplit=5)
                if len(parts) == 6 and _references(parts[5].removesuffix(" (deleted)"), entry, cwd):
                    reasons.add("mapped_file")
            for fd in (directory / "fd").iterdir():
                try:
                    if _references(os.readlink(fd).removesuffix(" (deleted)"), entry, cwd):
                        reasons.add("open_file")
                except FileNotFoundError:
                    pass
            if reasons:
                consumers.append({"pid": pid, "uid": owner, "reasons": sorted(reasons)})
        except (PermissionError, OSError, ValueError, TypeError):
            # Processes which exited during scanning do not become false unknowns.
            if directory.exists():
                unknown.append({"pid": pid, "reason": "process evidence unreadable"})
    return {"consumers": consumers, "unknown": unknown,
            "scope": "readable same-user processes; launchers must remain stopped",
            "other_user_processes": other_users, "bootstrap_scope": scope,
            "bootstrap_scope_sha256": bootstrap_scope_sha256}


def activate(*, root, receipt_sha256, checkout, expected_device, expected_inode, quarantine, records,
             bootstrap_scope_sha256=None):
    checkout = Path(checkout).resolve()
    root, quarantine, records = (Path(path).absolute() for path in (root, quarantine, records))
    observed = verify(root, receipt_sha256)
    require(observed["source_checkout"] == str(checkout), "activation checkout differs")
    entry = checkout / ".venv"
    expected = (expected_device, expected_inode)

    def check_identity():
        info = entry.lstat()
        require((info.st_dev, info.st_ino) == expected and stat.S_ISDIR(info.st_mode)
                and info.st_uid == os.getuid(), "old environment identity changed")

    def check_consumers():
        scan = inspect_consumers(entry, bootstrap_scope_sha256=bootstrap_scope_sha256)
        require(not scan["consumers"] and not scan["unknown"], "environment consumer scan is not clear")
        return scan

    check_identity()
    require(not records.exists() and not records.is_symlink(), "activation records already exist")
    require(not quarantine.exists() and not quarantine.is_symlink(), "quarantine already exists")
    require(not quarantine.is_relative_to(entry) and not records.is_relative_to(entry)
            and not root.is_relative_to(entry), "activation paths overlap the old environment")
    lock = checkout / ".native-runtime/development-cutover.lock"
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        require(os.fstat(fd).st_uid == os.getuid(), "cutover lock owner differs")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_identity()
        scan = check_consumers()
        _new_directory(records)
        _new_directory(quarantine)
        create_json(records / "before.json", {
            "status": "prepared-development", "old_identity": identity(entry), "scan": scan,
            "new_runtime": observed, "quarantine": str(quarantine), "rollback_to_old_allowed": False,
        })
        check_identity()
        final_scan = check_consumers()
        moved = False
        try:
            # Destinations live under a freshly created private directory; no
            # existing environment or receipt is replaced by this operation.
            os.rename(entry, quarantine / "env")
            moved = True
            quarantine.chmod(0)
            os.symlink(root / "env", entry, target_is_directory=True)
            result = {"status": "activated-development", "production_ready": False,
                      "entry_identity": identity(entry), "new_runtime": observed,
                      "quarantine": str(quarantine), "rollback_to_old_allowed": False,
                      "final_scan": final_scan}
            create_json(records / "activated.json", result)
            return result
        except BaseException:
            if moved:
                quarantine.chmod(0)
            create_json(records / "failed.json", {
                "status": "activation-failed", "old_quarantined": moved,
                "rollback_to_old_allowed": False,
            })
            raise
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", allow_abbrev=False)
    build_parser.add_argument("--snapshot", required=True, type=Path)
    build_parser.add_argument("--snapshot-sha256", required=True)
    build_parser.add_argument("--checkout", required=True, type=Path)
    build_parser.add_argument("--output", required=True, type=Path)
    for name in ("verify", "activate"):
        child = commands.add_parser(name, allow_abbrev=False)
        child.add_argument("--root", required=True, type=Path)
        child.add_argument("--receipt-sha256", required=True)
        if name == "activate":
            child.add_argument("--checkout", required=True, type=Path)
            child.add_argument("--expected-device", required=True, type=int)
            child.add_argument("--expected-inode", required=True, type=int)
            child.add_argument("--quarantine", required=True, type=Path)
            child.add_argument("--records", required=True, type=Path)
            child.add_argument("--bootstrap-scope-sha256")
    child = commands.add_parser("inspect", allow_abbrev=False)
    child.add_argument("--entry", required=True, type=Path)
    child.add_argument("--systemd-bootstrap", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build(args.snapshot, args.snapshot_sha256, args.checkout, args.output)
        elif args.command == "verify":
            result = verify(args.root, args.receipt_sha256)
        elif args.command == "inspect":
            pin = None
            if args.systemd_bootstrap:
                pin = hashlib.sha256(payload(bootstrap_scope(args.entry))).hexdigest()
            result = {"identity": identity(args.entry),
                      **inspect_consumers(args.entry, bootstrap_scope_sha256=pin)}
        else:
            values = vars(args).copy()
            values.pop("command")
            result = activate(**values)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        # Exception type only: external output and malformed records may contain
        # secrets. Detailed local diagnosis can inspect the numbered command hash.
        print(json.dumps({"status": "blocked-development", "production_ready": False,
                          "reason_type": type(exc).__name__,
                          "reason": str(exc) if isinstance(exc, DevelopmentError) else "external verification failed"}),
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
