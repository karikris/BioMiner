from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/development_runtime.py"


@pytest.fixture
def runtime():
    assert SCRIPT.is_file(), "development runtime interface is missing"
    spec = importlib.util.spec_from_file_location("development_runtime", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def thaw(root):
    if root.is_dir() and not root.is_symlink():
        root.chmod(0o700)
        for path in root.rglob("*"):
            if not path.is_symlink():
                path.chmod(0o700 if path.is_dir() else 0o600)


def test_cli_separates_build_verify_inspect_and_activate(runtime):
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(SCRIPT), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert all(word in result.stdout for word in ("build", "verify", "inspect", "activate"))


def test_freeze_preserves_external_symlink_target(runtime, tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("external interpreter")
    outside.chmod(0o755)
    root = tmp_path / "version"
    (root / "env/bin").mkdir(parents=True)
    (root / "env/bin/python").symlink_to(outside)
    (root / "env/dependency").write_text("fixed bytes")
    try:
        runtime.freeze(root)
        assert outside.stat().st_mode & 0o777 == 0o755
        assert not root.stat().st_mode & 0o222
        assert not (root / "env/dependency").stat().st_mode & 0o222
    finally:
        thaw(root)


def test_inventory_detects_modes_and_symlink_retargeting(runtime, tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    data = root / "data"
    data.write_text("same")
    link = root / "alias"
    link.symlink_to("data")
    before = runtime.inventory(root)
    data.chmod(0o444)
    assert runtime.inventory(root) != before
    before = runtime.inventory(root)
    link.unlink()
    link.symlink_to("missing")
    assert runtime.inventory(root) != before


def fake_process(proc, pid, *, args, env=b"", cwd=None, maps=""):
    directory = proc / str(pid)
    directory.mkdir(parents=True)
    (directory / "status").write_text(f"Name:\tworker\nUid:\t{os.getuid()}\t{os.getuid()}\n")
    (directory / "comm").write_text("worker\n")
    (directory / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in args) + b"\0")
    (directory / "environ").write_bytes(env)
    (directory / "maps").write_text(maps)
    (directory / "cwd").symlink_to(cwd or directory, target_is_directory=True)
    (directory / "fd").mkdir()
    return directory


@pytest.mark.parametrize(
    "case",
    ("installer_argument", "relative_argument", "virtual_env", "pythonpath", "pythonhome", "mapping", "fd"),
)
def test_consumer_scan_blocks_actual_environment_references(runtime, tmp_path, case):
    entry = tmp_path / "repo/.venv"
    entry.mkdir(parents=True)
    proc = tmp_path / "proc"
    args, env, maps = ["worker"], b"", ""
    if case == "installer_argument":
        args = ["uv", "pip", "install", "--python", str(entry / "bin/python")]
    elif case == "relative_argument":
        args = ["uv", "--python=.venv/bin/python"]
    elif case == "virtual_env":
        env = b"VIRTUAL_ENV=" + str(entry).encode() + b"\0SECRET=never-output\0"
    elif case == "pythonpath":
        env = b"PYTHONPATH=/other:" + str(entry / "lib").encode() + b"\0"
    elif case == "pythonhome":
        env = b"PYTHONHOME=" + str(entry).encode() + b"\0"
    elif case == "mapping":
        maps = f"123-456 r-xp 0 0:0 1 {entry}/lib/native.so\n"
    p = fake_process(proc, 424242, args=args, env=env, cwd=entry.parent, maps=maps)
    if case == "fd":
        (p / "fd/3").symlink_to(entry / "open-output")
    result = runtime.inspect_consumers(entry, proc_root=proc)
    assert [row["pid"] for row in result["consumers"]] == [424242]
    assert "never-output" not in json.dumps(result)
    assert "VIRTUAL_ENV=" not in json.dumps(result)


def test_consumer_scan_does_not_match_literal_shell_source(runtime, tmp_path):
    entry = tmp_path / "repo/.venv"
    entry.mkdir(parents=True)
    proc = tmp_path / "proc"
    fake_process(proc, 424242, args=["bash", "-c", f"printf '{entry}/bin/python'"], cwd=tmp_path)
    result = runtime.inspect_consumers(entry, proc_root=proc)
    assert result["consumers"] == []


@pytest.fixture
def bootstrap(runtime, tmp_path, monkeypatch):
    entry = tmp_path / "repo/.venv"
    entry.mkdir(parents=True)
    proc = tmp_path / "proc"
    unit_cgroup = f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service"
    observation = {
        "unit": f"user@{os.getuid()}.service", "main_pid": 424240, "uid": os.getuid(),
        "control_group": unit_cgroup, "executable": "/usr/lib/systemd/systemd",
        "manager_environment": [], "controller_sha256": "c" * 64,
        "unit_files": [{"path": "/usr/lib/systemd/system/user@.service", "sha256": "d" * 64}],
    }
    monkeypatch.setattr(runtime, "_systemd_observation", lambda *a: observation)
    for pid, parent, argv in (
        (424240, 1, ["/usr/lib/systemd/systemd", "--user"]),
        (424241, 424240, ["(sd-pam)"]),
    ):
        directory = fake_process(proc, pid, args=argv, cwd=entry.parent)
        (directory / "status").write_text(f"Name:\tbootstrap\nPPid:\t{parent}\nUid:\t{os.getuid()}\t{os.getuid()}\n")
        fields = ["S", str(parent), *["0"] * 17, str(pid * 10)]
        (directory / "stat").write_text(f"{pid} (bootstrap) " + " ".join(fields))
        (directory / "cgroup").write_text("0::" + unit_cgroup + "/init.scope\n")
        # Private proc evidence remains unreadable/absent, just as on this host.
        (directory / "cwd").unlink()
    return entry, proc, observation


def test_bootstrap_exclusion_binds_independent_unit_and_exact_process_identities(runtime, bootstrap):
    entry, proc, _ = bootstrap
    scope = runtime.bootstrap_scope(entry, proc_root=proc)
    pin = hashlib.sha256(runtime.payload(scope)).hexdigest()
    result = runtime.inspect_consumers(entry, proc_root=proc, bootstrap_scope_sha256=pin)
    assert result["unknown"] == []
    assert {row["pid"] for row in result["bootstrap_scope"]["processes"]} == {424240, 424241}
    assert result["bootstrap_scope_sha256"] == pin
    assert "not inspected" in result["bootstrap_scope"]["limitation"]


def test_bootstrap_exclusion_never_covers_other_service_descendants(runtime, bootstrap):
    entry, proc, _ = bootstrap
    scope = runtime.bootstrap_scope(entry, proc_root=proc)
    pin = hashlib.sha256(runtime.payload(scope)).hexdigest()
    directory = fake_process(proc, 424242, args=["worker"], cwd=entry.parent)
    (directory / "cwd").unlink()
    result = runtime.inspect_consumers(entry, proc_root=proc, bootstrap_scope_sha256=pin)
    assert result["unknown"] == [{"pid": 424242, "reason": "process evidence unreadable"}]


def test_bootstrap_pid_reuse_invalidates_scope_pin(runtime, bootstrap):
    entry, proc, _ = bootstrap
    scope = runtime.bootstrap_scope(entry, proc_root=proc)
    pin = hashlib.sha256(runtime.payload(scope)).hexdigest()
    path = proc / "424241/stat"
    fields = path.read_text().split()
    fields[-1] = "999999"
    path.write_text(" ".join(fields))
    with pytest.raises(runtime.DevelopmentError, match="bootstrap scope"):
        runtime.inspect_consumers(entry, proc_root=proc, bootstrap_scope_sha256=pin)


def test_bootstrap_environment_reference_blocks_exclusion(runtime, bootstrap):
    entry, proc, observation = bootstrap
    observation["manager_environment"] = [{"key": "VIRTUAL_ENV", "references_entry": True}]
    with pytest.raises(runtime.DevelopmentError, match="manager environment"):
        runtime.bootstrap_scope(entry, proc_root=proc)


def test_bootstrap_wrong_cgroup_cannot_be_excluded_by_name(runtime, bootstrap):
    entry, proc, _ = bootstrap
    (proc / "424241/cgroup").write_text("0::/an-unrelated-service\n")
    with pytest.raises(runtime.DevelopmentError, match="bootstrap"):
        runtime.bootstrap_scope(entry, proc_root=proc)


@pytest.fixture
def cutover(runtime, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".native-runtime").mkdir(parents=True)
    old = repo / ".venv"
    old.mkdir()
    (old / "preserved").write_text("old artifact bytes")
    dev = tmp_path / "new-version"
    (dev / "env").mkdir(parents=True)
    result = {"root": str(dev), "source_checkout": str(repo), "status": "verified-development"}
    monkeypatch.setattr(runtime, "verify", lambda *a, **kw: result)
    monkeypatch.setattr(runtime, "inspect_consumers", lambda *a, **kw: {"consumers": [], "unknown": []})
    quarantine = tmp_path / "quarantine"
    records = tmp_path / "cutover-records"
    yield repo, old, dev, quarantine, records
    thaw(quarantine)


def perform(runtime, cutover, **overrides):
    repo, old, dev, quarantine, records = cutover
    args = dict(
        root=dev, receipt_sha256="a" * 64, checkout=repo,
        expected_device=old.lstat().st_dev, expected_inode=old.lstat().st_ino,
        quarantine=quarantine, records=records,
    )
    args.update(overrides)
    return runtime.activate(**args)


def test_cutover_preserves_old_bytes_beneath_inaccessible_quarantine(runtime, cutover):
    repo, old, dev, quarantine, records = cutover
    result = perform(runtime, cutover)
    assert old.is_symlink()
    assert old.resolve() == dev / "env"
    assert quarantine.stat().st_mode & 0o777 == 0
    quarantine.chmod(0o700)
    assert (quarantine / "env/preserved").read_text() == "old artifact bytes"
    assert result["status"] == "activated-development"
    assert (records / "activated.json").is_file()


def test_cutover_rejects_changed_inode_before_touching_environment(runtime, cutover):
    with pytest.raises(runtime.DevelopmentError, match="identity"):
        perform(runtime, cutover, expected_inode=0)
    assert cutover[1].is_dir() and not cutover[1].is_symlink()
    assert not cutover[3].exists()


@pytest.mark.parametrize("key", ["consumers", "unknown"])
def test_cutover_rejects_consumers_or_unknown_ownership(runtime, cutover, monkeypatch, key):
    scan = {"consumers": [], "unknown": []}
    scan[key] = [{"pid": 123}]
    monkeypatch.setattr(runtime, "inspect_consumers", lambda *a, **kw: scan)
    with pytest.raises(runtime.DevelopmentError, match="consumer"):
        perform(runtime, cutover)
    assert cutover[1].is_dir() and not cutover[1].is_symlink()


def test_cutover_never_overwrites_an_existing_record_directory(runtime, cutover):
    cutover[4].mkdir()
    with pytest.raises((runtime.DevelopmentError, FileExistsError)):
        perform(runtime, cutover)
    assert not cutover[1].is_symlink()


def test_cutover_failure_never_reactivates_quarantined_environment(runtime, cutover, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("publication failure")
    monkeypatch.setattr(runtime.os, "symlink", fail)
    with pytest.raises(OSError, match="publication"):
        perform(runtime, cutover)
    assert not cutover[1].exists()
    assert cutover[3].stat().st_mode & 0o777 == 0
    assert (cutover[4] / "failed.json").is_file()


def test_build_rejects_existing_output_before_any_install(runtime, tmp_path):
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(runtime.DevelopmentError, match="exists"):
        runtime.build(tmp_path / "snapshot", "a" * 64, tmp_path, output)


def test_build_installs_exact_native_wheels_after_locked_dependencies_and_editable_last(runtime, tmp_path, monkeypatch):
    builder = tmp_path / "builder.py"
    builder.write_text("loaded builder source")
    builder_hash = runtime.digest(builder)
    monkeypatch.setattr(runtime, "__file__", str(builder))
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text(
        '[build-system]\nrequires=["setuptools>=84.2"]\nbuild-backend="setuptools.build_meta"\n'
    )
    (checkout / "uv.lock").write_text("frozen dependency lock")
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    artifacts = []
    for name in ("biominer", *sorted(runtime.NATIVE)):
        path = tmp_path / f"{name}.whl"
        path.write_bytes(name.encode())
        artifacts.append({"name": name, "path": str(path), "sha256": runtime.digest(path)})
    doc = {
        "project": {"path": str(checkout / "pyproject.toml"), "sha256": runtime.digest(checkout / "pyproject.toml")},
        "base_lock": {"path": str(checkout / "uv.lock"), "sha256": runtime.digest(checkout / "uv.lock")},
        "artifacts": artifacts, "uv": {"path": "/approved/uv"},
    }
    native = types.SimpleNamespace(
        _python=lambda *a: Path("/approved/python"),
        _file=lambda item, *a: Path(item["path"]),
        _host=lambda *a: None, launch_environment=lambda: {},
    )
    monkeypatch.setattr(runtime, "snapshot_context", lambda *a: (native, doc, profile))
    calls = []
    def command(args, env, records, **kwargs):
        calls.append(([str(arg) for arg in args], env.copy()))
        builder.write_text("later builder edit")
        assert env["UV_LINK_MODE"] == "copy"
        assert env["UV_CACHE_DIR"] == str(tmp_path / "new/cache")
        records.append({"argv": calls[-1][0], "returncode": 0})
    monkeypatch.setattr(runtime, "command", command)
    monkeypatch.setattr(runtime, "runtime_checks", lambda *a: {"checks": ["fixture"], "identity": {}})
    monkeypatch.setattr(runtime, "verify", lambda root, sha: {"receipt_sha256": sha})
    root = tmp_path / "new"
    try:
        result = runtime.build(tmp_path / "snapshot", "a" * 64, checkout, root)
        assert len(calls) == 4
        sync = calls[1][0]
        assert "--locked" in sync and "--no-install-project" in sync
        exclusions = {sync[index + 1] for index, arg in enumerate(sync) if arg == "--no-install-package"}
        assert exclusions == runtime.NATIVE
        requirements = (root / "native-wheels.txt").read_text()
        assert len(requirements.splitlines()) == 4
        assert "biominer.whl" not in requirements
        assert all(f'--hash=sha256:{row["sha256"]}' in requirements for row in artifacts if row["name"] != "biominer")
        assert all(flag in calls[2][0] for flag in ("--no-index", "--no-deps", "--require-hashes"))
        assert all(flag in calls[3][0] for flag in ("--no-deps", "--build-constraints", "--editable"))
        assert (root / "build-constraints.txt").read_text() == runtime.BACKEND + "\n"
        assert runtime.digest(root / runtime.RECEIPT) == result["receipt_sha256"]
        receipt = json.loads((root / runtime.RECEIPT).read_text())
        assert receipt["production_ready"] is False and receipt["source_is_editable"] is True
        assert receipt["script_sha256"] == builder_hash
        assert receipt["inventory"] == runtime.inventory(root)
        assert checkout.stat().st_mode & 0o200
    finally:
        thaw(root)


def test_verify_rejects_dependency_change_before_running_checks(runtime, tmp_path, monkeypatch):
    root = tmp_path / "version"
    root.mkdir()
    dependency = root / "dependency"
    dependency.write_text("approved")
    runtime.freeze(root)
    receipt = {
        "schema": runtime.SCHEMA, "status": "built-development", "root": str(root),
        "production_ready": False, "source_is_editable": True, "inventory": runtime.inventory(root),
    }
    root.chmod(0o700)
    runtime.create_json(root / runtime.RECEIPT, receipt)
    (root / runtime.RECEIPT).chmod(0o444)
    root.chmod(0o555)
    expected = runtime.digest(root / runtime.RECEIPT)
    dependency.chmod(0o644)
    dependency.write_text("unapproved")
    dependency.chmod(0o444)
    monkeypatch.setattr(runtime, "snapshot_context", lambda *a: pytest.fail("checked a changed dependency tree"))
    try:
        with pytest.raises(runtime.DevelopmentError, match="inventory changed"):
            runtime.verify(root, expected)
    finally:
        thaw(root)


def test_verify_requires_externally_pinned_receipt(runtime, tmp_path):
    (tmp_path / runtime.RECEIPT).write_text("{}")
    with pytest.raises(runtime.DevelopmentError, match="receipt hash"):
        runtime.verify(tmp_path, "a" * 64)


def wheel(path, marker):
    dist = "guard_fixture-1.0.dist-info"
    files = {
        "guard_fixture.py": f"VALUE = {marker!r}\n".encode(),
        f"{dist}/METADATA": b"Metadata-Version: 2.1\nName: guard-fixture\nVersion: 1.0\n",
        f"{dist}/WHEEL": b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    rows = [
        f"{name},sha256={base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()},{len(data)}"
        for name, data in files.items()
    ]
    files[f"{dist}/RECORD"] = ("\n".join(rows) + f"\n{dist}/RECORD,,\n").encode()
    path.parent.mkdir()
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)


@pytest.fixture
def uv_guard(runtime, tmp_path):
    uv = shutil.which("uv")
    if uv is None or sys.platform != "linux" or os.getuid() == 0:
        pytest.skip("Read-only protection qualification requires Linux, uv and a non-root user")
    version = tmp_path / "version"
    version.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    custom = tmp_path / "custom/guard_fixture-1.0-py3-none-any.whl"
    stock = tmp_path / "stock/guard_fixture-1.0-py3-none-any.whl"
    wheel(custom, "custom-native")
    wheel(stock, "registry-replacement")
    (project / "pyproject.toml").write_text(
        '[project]\nname="guard-project"\nversion="1.0"\n'
        f'requires-python=">={sys.version_info.major}.{sys.version_info.minor}"\n'
        f'dependencies=["guard-fixture @ {stock.as_uri()}"]\n'
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("UV_", "PYTHON")) and k != "VIRTUAL_ENV"}
    env.update(UV_CACHE_DIR=str(tmp_path / "cache"), UV_LINK_MODE="copy", UV_PYTHON_DOWNLOADS="never")
    def command(*args):
        return subprocess.run([uv, *args], env=env, cwd=project, capture_output=True, text=True, timeout=40)
    for args in [
        ("venv", "--python", sys.executable, str(version / "env")),
        ("pip", "install", "--python", str(version / "env/bin/python"), "--no-index", "--no-deps", str(custom)),
        ("lock", "--offline", "--python", sys.executable),
    ]:
        result = command(*args)
        assert result.returncode == 0, result.stderr
    (project / ".venv").symlink_to(version / "env", target_is_directory=True)
    yield version, project, command
    thaw(version)


@pytest.mark.parametrize("operation", ["sync", "run", "replace"])
def test_actual_uv_cannot_mutate_frozen_version_or_alias(runtime, uv_guard, operation):
    version, project, command = uv_guard
    if operation == "replace":
        # A missing original interpreter makes uv try to recreate this venv
        # using the available explicit interpreter, with no network involved.
        python = version / "env/bin/python"
        python.unlink()
        python.symlink_to(version / "missing-interpreter")
    runtime.freeze(version)
    before = runtime.inventory(version)
    alias = runtime.identity(project / ".venv")
    lock = (project / "uv.lock").read_bytes()
    args = ["sync", "--offline"]
    if operation == "run":
        args = ["run", "--offline", "python", "-c", "import guard_fixture"]
    elif operation == "replace":
        args += ["--python", sys.executable]
    result = command(*args)
    # -rA exposes bounded offline fixture output for the migration evidence log.
    print(json.dumps({"guard_operation": operation, "returncode": result.returncode,
                      "stdout": result.stdout, "stderr": result.stderr}, sort_keys=True))
    assert result.returncode != 0, result.stdout
    assert "Permission denied" in result.stderr, result.stderr
    assert runtime.inventory(version) == before
    assert runtime.identity(project / ".venv") == alias
    assert (project / "uv.lock").read_bytes() == lock


def test_permanent_interpreter_keeps_subprocess_identity_through_directory_alias(runtime, uv_guard):
    version, project, _ = uv_guard
    python = version / "env/bin/python"
    code = (
        "import sys,subprocess,json;"
        "child=subprocess.check_output([sys.executable,'-I','-B','-c',"
        "'import sys,json;print(json.dumps([sys.executable,sys.prefix]))'],text=True);"
        "print(json.dumps([sys.executable,sys.prefix,json.loads(child)]))"
    )
    result = subprocess.run([str(python), "-I", "-B", "-c", code], capture_output=True, text=True, check=True)
    executable, prefix, child = json.loads(result.stdout)
    assert executable == str(python)
    assert prefix == str(version / "env")
    assert child == [executable, prefix]
    assert (project / ".venv").resolve() == version / "env"
