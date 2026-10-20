#!/usr/bin/env python3
"""Freeze a candidate's fallback control files without rewriting its history.

Use the profile's approved Python executable (APPROVED_PYTHON below):

    APPROVED_PYTHON -I -B scripts/freeze_native_profile.py freeze CANDIDATE NEW_DIRECTORY
    APPROVED_PYTHON -I -B NEW_DIRECTORY/scripts/freeze_native_profile.py verify NEW_DIRECTORY
    APPROVED_PYTHON -I -B NEW_DIRECTORY/scripts/freeze_native_profile.py run NEW_DIRECTORY artifacts
    APPROVED_PYTHON -I -B NEW_DIRECTORY/scripts/freeze_native_profile.py run NEW_DIRECTORY install --generation NEW_GENERATION

The snapshot preserves scripts, project configuration and lock bytes. Existing
wheels, build materials, Python, tools and host dependencies stay at their
hash-bound paths. This is not a host-independent runtime or a readiness claim.
Rebuilding is disabled: original recipe records are retained as provenance.
Use a new generation with the frozen profile; old generation receipts are never
rewritten. Keep the returned manifest hash as an independent verification pin.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import types


SHA = re.compile(r"[0-9a-f]{64}\Z")
OPERATIONS = {"artifacts", "install", "verify", "exec"}
LIMITATION = (
    "Requires retained hash-bound artifacts, build materials, Python, tools and "
    "qualified host dependencies; rebuild and host independence are not supported."
)


class SnapshotError(ValueError):
    """Snapshot creation or verification failed closed."""


def require(condition, message):
    if not condition:
        raise SnapshotError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def payload(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def reference(path):
    return {"path": str(Path(path).absolute()), "sha256": sha256(path)}


def read_json(path):
    try:
        result = json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise SnapshotError("missing or malformed snapshot/profile JSON") from exc
    require(isinstance(result, dict), "snapshot/profile must be an object")
    return result


def resolve_record(item, base):
    require(isinstance(item, dict) and isinstance(item.get("path"), str), "file reference requires a path")
    require(isinstance(item.get("sha256"), str) and SHA.fullmatch(item["sha256"]), "file reference requires a SHA256 hash")
    name = item["path"]
    if "{generation}" in name:
        return None
    require("{" not in name and "}" not in name, "unsupported file reference placeholder")
    path = Path(name)
    return (path if path.is_absolute() else base / path).absolute()


def check_record(item, base):
    path = resolve_record(item, base)
    if path is not None:
        require(path.is_file() and sha256(path) == item["sha256"], "retained file missing or SHA256 hash mismatch")
    return path


def records(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield value
        for child in value.values():
            yield from records(child)
    elif isinstance(value, list):
        for child in value:
            yield from records(child)


def load_launcher(path, expected_hash):
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected_hash, "launcher hash mismatch")
    module = types.ModuleType("frozen_native_runtime")
    module.__file__ = str(path)
    # Execute the verified bytes, never an import cache or the live repo module.
    exec(compile(data, str(path), "exec"), module.__dict__)
    require(callable(getattr(module, "verify_artifacts", None)) and callable(getattr(module, "main", None)), "launcher interface unavailable")
    return module


def candidate(path):
    path = Path(path).absolute()
    doc = read_json(path)
    require(isinstance(doc.get("inputs"), list), "candidate input records required")
    launchers = [
        item for item in doc["inputs"]
        if isinstance(item, dict) and Path(item.get("path", "")).name == "native_runtime.py"
    ]
    require(len(launchers) == 1, "candidate requires one hash-bound native_runtime.py launcher")
    launcher = check_record(launchers[0], path.parent)
    require(launcher is not None, "launcher cannot be a generation placeholder")
    for item in records(doc):
        check_record(item, path.parent)
    module = load_launcher(launcher, launchers[0]["sha256"])
    try:
        result = module.verify_artifacts(path)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise SnapshotError(f"candidate artifact verification failed: {exc}") from exc
    require(
        isinstance(result, dict) and result.get("status") == "candidate"
        and result.get("ready") is False and result.get("profile_sha256") == sha256(path),
        "launcher did not validate this candidate profile",
    )
    return doc, module


def relocate(value, base, mapping):
    """Change references only; copied file contents and historical receipts stay intact."""
    if isinstance(value, dict):
        result = {key: relocate(child, base, mapping) for key, child in value.items()}
        if "path" in value and "sha256" in value:
            old = resolve_record(value, base)
            if old is not None:
                result["path"] = str(mapping.get(str(old), old))
        return result
    if isinstance(value, list):
        return [relocate(child, base, mapping) for child in value]
    if isinstance(value, str) and value in mapping:
        return str(mapping[value])
    return value


def retained_references(doc, root):
    retained = {}
    for item in records(doc):
        path = resolve_record(item, root)
        if path is not None and not path.is_relative_to(root):
            require(str(path) not in retained or retained[str(path)] == item["sha256"], "conflicting retained hashes")
            retained[str(path)] = item["sha256"]
    return [{"path": name, "sha256": value} for name, value in sorted(retained.items())]


def create_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def freeze(profile, output):
    profile, output = Path(profile).absolute(), Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), "snapshot output already exists")
    require(output.parent.is_dir(), "snapshot parent directory must exist")
    original_bytes = profile.read_bytes()
    doc, _ = candidate(profile)
    require(sha256(profile) == hashlib.sha256(original_bytes).hexdigest(), "candidate changed during validation")
    planned = {}

    def add(item, relative):
        source = check_record(item, profile.parent)
        require(source is not None and not source.is_symlink(), "copied snapshot sources must be regular files, not symlinks")
        old = planned.get(relative)
        require(old is None or old == (source, item["sha256"]), "snapshot script filename collision")
        planned[relative] = (source, item["sha256"])

    add(doc["project"], "project/pyproject.toml")
    add(doc["base_lock"], "project/uv.lock")
    for item in doc["inputs"]:
        if item.get("kind") in {"recipe", "check"}:
            add(item, "scripts/" + Path(item["path"]).name)
    for item in [*doc["checks"], *doc["recipes"]]:
        add(item["script"], "scripts/" + Path(item["script"]["path"]).name)
    require("scripts/native_runtime.py" in planned, "launcher must be a copied recipe/check input")
    add(reference(Path(__file__).absolute()), "scripts/freeze_native_profile.py")
    mapping = {str(source): output / name for name, (source, _) in planned.items()}
    require(len(mapping) == len(planned), "one source cannot have multiple snapshot destinations")
    frozen = relocate(doc, profile.parent, mapping)
    frozen["recipes"] = []
    output.mkdir(mode=0o700)
    for name, (source, expected) in sorted(planned.items()):
        data = source.read_bytes()
        require(hashlib.sha256(data).hexdigest() == expected, "source changed while freezing")
        create_file(output / name, data)
    create_file(output / "original-profile.json", original_bytes)
    create_file(output / "profile.json", payload(frozen))
    candidate(output / "profile.json")
    files = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    manifest = {
        "schema_version": 1, "status": "candidate", "ready": False,
        "rebuild_supported": False, "limitation": LIMITATION,
        "original_profile": {"path": str(profile), "sha256": hashlib.sha256(original_bytes).hexdigest()},
        "files": files, "archived_recipes": copy.deepcopy(doc["recipes"]),
        "relocations": [
            {"original_path": str(source), "frozen_path": name, "sha256": expected}
            for name, (source, expected) in sorted(planned.items())
        ],
        "retained_references": retained_references(frozen, output),
    }
    # Manifest last. A partial directory cannot be admitted by verify_snapshot.
    create_file(output / "snapshot.json", payload(manifest))
    for path in sorted(output.rglob("*"), reverse=True):
        require(not path.is_symlink(), "snapshot symlink detected")
        path.chmod(0o555 if path.is_dir() else 0o444)
    output.chmod(0o555)
    return verify_snapshot(output)


def verify_snapshot(output, expected_manifest_sha256=None):
    output = Path(output).absolute()
    require(output.is_dir() and not output.is_symlink(), "snapshot directory missing or symlinked")
    manifest_path = output / "snapshot.json"
    if expected_manifest_sha256 is not None:
        require(SHA.fullmatch(expected_manifest_sha256) and sha256(manifest_path) == expected_manifest_sha256, "snapshot manifest hash mismatch")
    manifest = read_json(manifest_path)
    require(
        manifest.get("schema_version") == 1 and manifest.get("status") == "candidate"
        and manifest.get("ready") is False and manifest.get("rebuild_supported") is False
        and manifest.get("limitation") == LIMITATION,
        "snapshot manifest contract changed",
    )
    files = manifest.get("files")
    require(isinstance(files, dict) and files, "snapshot file inventory missing")
    allowed = {"snapshot.json"}
    for name, expected in files.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts and name != "snapshot.json", "unsafe snapshot inventory path")
        require(isinstance(expected, str) and SHA.fullmatch(expected), "invalid snapshot inventory hash")
        allowed.add(name)
        allowed.update(str(parent) for parent in path.parents if str(parent) != ".")
    seen = set()
    for path in [output, *output.rglob("*")]:
        require(not path.is_symlink(), "snapshot symlink detected")
        require(path.is_file() or path.is_dir(), "snapshot special file detected")
        info = path.stat()
        require(info.st_uid == os.getuid() and not info.st_mode & 0o222, "snapshot is not owned and read-only")
        if path != output:
            name = str(path.relative_to(output))
            seen.add(name)
            require(name in allowed, "unexpected snapshot file or directory")
            if name in files:
                require(path.is_file() and sha256(path) == files[name], "snapshot file hash mismatch")
    require(seen == allowed, "snapshot inventory file missing")
    require({"profile.json", "original-profile.json", "scripts/native_runtime.py", "scripts/freeze_native_profile.py"} <= files.keys(), "snapshot control file missing")
    original = read_json(output / "original-profile.json")
    origin = manifest["original_profile"]
    require(files["original-profile.json"] == origin["sha256"], "original profile hash mismatch")
    mapping = {}
    for item in manifest["relocations"]:
        name = item["frozen_path"]
        require(name in files and files[name] == item["sha256"], "relocation bytes do not match snapshot inventory")
        require(item["original_path"] not in mapping, "duplicate snapshot relocation")
        mapping[item["original_path"]] = output / name
    expected = relocate(original, Path(origin["path"]).parent, mapping)
    expected["recipes"] = []
    frozen = read_json(output / "profile.json")
    require(expected == frozen and manifest["archived_recipes"] == original["recipes"], "frozen profile differs beyond authorized relocation and disabled rebuilds")
    require(manifest["retained_references"] == retained_references(frozen, output), "retained dependency inventory changed")
    candidate(output / "profile.json")
    return {
        "status": "candidate", "ready": False, "rebuild_supported": False,
        "snapshot": str(output), "profile": reference(output / "profile.json"),
        "manifest": reference(manifest_path), "limitation": LIMITATION,
    }


def run_snapshot(output, args):
    require(args and args[0] in OPERATIONS, "snapshot rebuild is unsupported; use artifacts/install/verify/exec")
    options = args[:args.index("--")] if "--" in args else args
    require(not any(arg == "--profile" or arg.startswith("--profile=") for arg in options), "snapshot profile cannot be overridden")
    verify_snapshot(output)
    output = Path(output).absolute()
    manifest = read_json(output / "snapshot.json")
    module = load_launcher(output / "scripts/native_runtime.py", manifest["files"]["scripts/native_runtime.py"])
    return module.main([args[0], "--profile", str(output / "profile.json"), *args[1:]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("freeze")
    command.add_argument("profile", type=Path)
    command.add_argument("output", type=Path)
    command = commands.add_parser("verify")
    command.add_argument("snapshot", type=Path)
    command.add_argument("--manifest-sha256")
    command = commands.add_parser("run")
    command.add_argument("snapshot", type=Path)
    command.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze(args.profile, args.output)
        elif args.command == "verify":
            result = verify_snapshot(args.snapshot, args.manifest_sha256)
        else:
            return run_snapshot(args.snapshot, args.args[1:] if args.args[:1] == ["--"] else args.args)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "blocked", "ready": False, "reason": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
