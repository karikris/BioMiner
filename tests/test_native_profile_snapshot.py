from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest

from test_native_runtime import wheel as native_wheel

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "freeze_native_profile.py"


def test_snapshot_cli_exposes_freeze_verify_and_run():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert all(command in result.stdout for command in ("freeze", "verify", "run"))


def record(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def blob(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return record(path)


def unlock(path):
    path.chmod(0o755)
    for item in path.rglob("*"):
        if not item.is_symlink():
            item.chmod(0o755 if item.is_dir() else 0o644)


@pytest.fixture
def candidate(tmp_path):
    """Real native admission with synthetic wheels and a fixture-pinned patch."""
    repo = tmp_path / "repo"
    materials = tmp_path / "materials"
    materials.mkdir()
    project = blob(repo / "pyproject.toml", '[project]\nname="biominer"\nversion="0.1.0"\n')
    lock = blob(repo / "uv.lock", "version = 1\n")
    patch = blob(materials / "zlib-fix.patch", "reviewed patch fixture")
    source = (SCRIPT.parent / "native_runtime.py").read_text()
    source = re.sub(
        r'(LXML_ZLIB_PATCH_SHA256\s*=\s*)["\'][a-f0-9]{64}["\']',
        lambda match: match[1] + repr(patch["sha256"]),
        source,
    )
    launcher = blob(repo / "scripts" / "native_runtime.py", source)
    policy = types.ModuleType("fixture_native_policy")
    policy.__file__ = launcher["path"]
    exec(compile(source, launcher["path"], "exec"), policy.__dict__)
    # Reuse the admission suite's complete receipt closure (including lxml
    # licenses/relink qualification); the copied policy pins its fixture patch.
    blob(materials / "pyproject.toml", Path(project["path"]).read_text())
    artifacts = [
        native_wheel(materials, name, version)
        for name, version in {**policy.VERSIONS, "biominer": "0.1.0"}.items()
    ]
    helper = blob(repo / "scripts" / "check_helper.py", "frozen sibling evidence")
    check = blob(
        repo / "scripts" / "check_fixture.py",
        "from pathlib import Path\nprint(Path(__file__).with_name('check_helper.py').read_text())\n",
    )
    recipe = blob(repo / "scripts" / "build_fixture.py", "raise SystemExit('build must remain unsupported')\n")
    doc = {
        "schema_version": 1, "status": "candidate", "ready": False,
        "platform": {"system": "Linux", "machine": "x86_64"},
        "project": project, "base_lock": lock,
        "inputs": [dict(launcher, kind="recipe"), dict(check, kind="check"), dict(helper, kind="check"), dict(recipe, kind="recipe")],
        "artifacts": artifacts,
        "python": {**record(Path(sys.executable).resolve()), "version": "3.14.7", "openssl_version": "3.5.9"},
        "uv": blob(materials / "uv", "fixture uv"),
        "native_files": [dict(blob(materials / "libfixture.so", "fixture native dependency"), owner="dpkg:fixture")],
        "host_packages": [{"name": "fixture", "minimum_version": "1.0", "origin": "fixture"}],
        "checks": [{"name": name, "script": check, "args": ["{profile}"]} for name in sorted(policy.INSTALL_CHECKS)],
        "recipes": [{"script": recipe, "args": ["--root", str(repo), "--lock", lock["path"]]}],
        "extension_policy": {"policy_id": "c" * 64, "allowlist": []},
    }
    path = repo / "candidate.json"
    path.write_text(json.dumps(doc))
    assert policy.verify_artifacts(path)["ready"] is False
    blob(tmp_path / "historical-generation.json",
         json.dumps({"profile_sha256": record(path)["sha256"]}))
    return path, doc, repo


@pytest.fixture
def snapshot(candidate, tmp_path):
    from scripts import freeze_native_profile as freezer

    output = tmp_path / "snapshot"
    freezer.freeze(candidate[0], output)
    yield freezer, output
    if output.exists():
        unlock(output)


def test_freeze_is_candidate_evidence_not_runtime_admission(snapshot):
    freezer, output = snapshot
    result = freezer.verify_snapshot(output)
    assert result["status"] == "candidate"
    assert result["ready"] is False
    assert result["rebuild_supported"] is False


def test_frozen_artifacts_survive_original_repository_removal(snapshot, candidate, tmp_path):
    _, output = snapshot
    _, _, repo = candidate
    shutil.rmtree(repo)
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(output / "scripts" / SCRIPT.name),
         "run", str(output), "artifacts"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "candidate"


def test_snapshot_preserves_original_profile_and_generation_receipt(snapshot, candidate, tmp_path):
    _, output = snapshot
    original = candidate[0].read_bytes()
    generation = tmp_path / "historical-generation.json"
    assert (output / "original-profile.json").read_bytes() == original
    assert candidate[0].read_bytes() == original
    assert generation.read_text() == json.dumps({"profile_sha256": record(candidate[0])["sha256"]})


def test_frozen_recipe_execution_is_disabled_but_provenance_preserved(snapshot, candidate):
    freezer, output = snapshot
    profile = json.loads((output / "profile.json").read_text())
    manifest = json.loads((output / "snapshot.json").read_text())
    assert profile["recipes"] == []
    assert manifest["archived_recipes"] == candidate[1]["recipes"]
    with pytest.raises(freezer.SnapshotError, match="rebuild"):
        freezer.run_snapshot(output, ["build"])


def test_direct_frozen_runtime_also_rejects_build(snapshot, tmp_path):
    _, output = snapshot
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(output / "scripts" / "native_runtime.py"),
         "build", "--profile", str(output / "profile.json"), "--output", str(tmp_path / "rebuild")],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "no explicit build recipes" in result.stderr
    assert not (tmp_path / "rebuild").exists()


def test_frozen_check_resolves_its_sibling_after_repository_removal(snapshot, candidate):
    _, output = snapshot
    shutil.rmtree(candidate[2])
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(output / "scripts" / "check_fixture.py")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "frozen sibling evidence"


def test_snapshot_is_owned_and_read_only(snapshot):
    _, output = snapshot
    assert all(not item.stat().st_mode & 0o222 for item in [output, *output.rglob("*")])


@pytest.mark.parametrize("damage", ["bytes", "missing", "extra", "symlink", "writable", "manifest"])
def test_snapshot_tampering_fails_closed(snapshot, tmp_path, damage):
    freezer, output = snapshot
    target = output / "project" / "pyproject.toml"
    if damage == "bytes":
        target.chmod(0o644)
        target.write_text("changed")
        target.chmod(0o444)
    elif damage == "missing":
        target.parent.chmod(0o755)
        target.unlink()
        target.parent.chmod(0o555)
    elif damage in {"extra", "symlink"}:
        output.chmod(0o755)
        extra = output / "unexpected"
        if damage == "extra":
            extra.write_text("extra")
            extra.chmod(0o444)
        else:
            extra.symlink_to(tmp_path)
        output.chmod(0o555)
    elif damage == "writable":
        target.chmod(0o644)
    else:
        target = output / "snapshot.json"
        target.chmod(0o644)
        doc = json.loads(target.read_text())
        doc["rebuild_supported"] = True
        target.write_text(json.dumps(doc))
        target.chmod(0o444)
    with pytest.raises(freezer.SnapshotError):
        freezer.verify_snapshot(output)


@pytest.mark.parametrize("dependency", ["wheel", "uv", "native_file", "build_input"])
def test_external_dependency_tampering_still_blocks_snapshot(snapshot, candidate, dependency):
    freezer, output = snapshot
    doc = candidate[1]
    retained = {
        "wheel": doc["artifacts"][0], "uv": doc["uv"],
        "native_file": doc["native_files"][0],
        "build_input": json.loads(Path(doc["artifacts"][0]["source_receipt"]["path"]).read_text())["inputs"][0],
    }[dependency]
    Path(retained["path"]).write_bytes(b"changed retained dependency")
    with pytest.raises(freezer.SnapshotError, match="retained|artifact|SHA256"):
        freezer.verify_snapshot(output)


def test_frozen_profile_cannot_override_pinned_configuration(snapshot):
    freezer, output = snapshot
    with pytest.raises(freezer.SnapshotError, match="profile"):
        freezer.run_snapshot(output, ["artifacts", "--profile", "/different/profile.json"])


def test_freeze_refuses_existing_output_without_altering_it(snapshot, candidate):
    freezer, output = snapshot
    before = (output / "snapshot.json").read_bytes()
    with pytest.raises(freezer.SnapshotError, match="exist"):
        freezer.freeze(candidate[0], output)
    assert (output / "snapshot.json").read_bytes() == before


def test_invalid_candidate_creates_no_snapshot(candidate, tmp_path):
    from scripts import freeze_native_profile as freezer

    Path(candidate[1]["base_lock"]["path"]).write_text("changed lock")
    output = tmp_path / "snapshot"
    with pytest.raises(freezer.SnapshotError, match="SHA256|hash"):
        freezer.freeze(candidate[0], output)
    assert not output.exists()


def test_external_manifest_pin_is_checked(snapshot):
    freezer, output = snapshot
    digest = record(output / "snapshot.json")["sha256"]
    assert freezer.verify_snapshot(output, digest)["manifest"]["sha256"] == digest
    with pytest.raises(freezer.SnapshotError, match="manifest hash"):
        freezer.verify_snapshot(output, "f" * 64)


def test_relative_controls_relocate_and_generation_templates_stay_intact(candidate, tmp_path):
    from scripts import freeze_native_profile as freezer

    profile, doc, repo = candidate
    for item in freezer.records(doc):
        if Path(item["path"]).is_relative_to(repo):
            item["path"] = str(Path(item["path"]).relative_to(repo))
    generation_file = {"path": "{generation}/env/lib/example.so", "sha256": "a" * 64}
    doc["native_files"].append(generation_file)
    profile.write_text(json.dumps(doc))
    output = tmp_path / "snapshot"
    try:
        freezer.freeze(profile, output)
        frozen = json.loads((output / "profile.json").read_text())
        assert frozen["project"]["path"] == str(output / "project/pyproject.toml")
        assert frozen["checks"][0]["script"]["path"] == str(output / "scripts/check_fixture.py")
        assert frozen["native_files"][-1] == generation_file
        shutil.rmtree(repo)
        assert freezer.verify_snapshot(output)["status"] == "candidate"
    finally:
        if output.exists():
            unlock(output)


@pytest.mark.parametrize("operation", ["install", "verify", "exec"])
def test_frozen_operations_pass_pinned_profile_to_preserved_launcher(snapshot, monkeypatch, operation):
    freezer, output = snapshot
    load = freezer.load_launcher
    calls = []

    def observe(*args):
        module = load(*args)
        module.main = lambda argv: calls.append(argv) or 0
        return module

    monkeypatch.setattr(freezer, "load_launcher", observe)
    trailing = ["--generation", "/retained/generation"]
    if operation == "exec":
        trailing += ["--", "python", "-c", "print('bounded application')"]
    assert freezer.run_snapshot(output, [operation, *trailing]) == 0
    assert calls == [[operation, "--profile", str(output / "profile.json"), *trailing]]


def test_frozen_launcher_creates_generation_under_retained_persistent_root(candidate, tmp_path):
    from scripts import freeze_native_profile as freezer

    # A /tmp fixture would pass the launcher's legacy output exception and miss
    # the relocation bug. Exercise the real frozen path guard outside /tmp.
    with tempfile.TemporaryDirectory(prefix=".biominer-freezer-test-", dir=Path.home()) as home:
        generation_root = Path(home).resolve() / ".native-runtime" / "generations"
        assert not generation_root.is_relative_to(Path("/tmp"))
        profile, doc, repo = candidate
        doc["generation_root"] = str(generation_root)
        profile.write_text(json.dumps(doc))
        output = tmp_path / "snapshot"
        try:
            freezer.freeze(profile, output)
            shutil.rmtree(repo)
            manifest = json.loads((output / "snapshot.json").read_text())
            launcher = freezer.load_launcher(
                output / "scripts/native_runtime.py",
                manifest["files"]["scripts/native_runtime.py"],
            )
            frozen, _ = launcher.load_profile(output / "profile.json")
            assert frozen["generation_root"] == str(generation_root)
            generation = generation_root / "fallback-1"
            with pytest.raises(launcher.ProfileError, match="outputs must be under"):
                launcher._new_output(generation)
            launcher._new_output(generation, frozen["generation_root"])
            assert generation.is_dir()
            with pytest.raises(launcher.ProfileError, match="already exists"):
                launcher._new_output(generation, frozen["generation_root"])
        finally:
            if output.exists():
                unlock(output)


@pytest.mark.parametrize("option", ["--prof", "--prof="])
def test_frozen_cli_rejects_abbreviated_profile_override(snapshot, candidate, option):
    _, output = snapshot
    override = [option + str(candidate[0])] if option.endswith("=") else [option, str(candidate[0])]
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(output / "scripts" / SCRIPT.name),
         "run", str(output), "artifacts", *override],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr
