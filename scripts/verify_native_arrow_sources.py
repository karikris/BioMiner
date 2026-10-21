#!/usr/bin/env python3
"""Verify supplemental Arrow source evidence without executing recipes or tools.

The producer captures a separate receipt for the finite bundled cohort and
selected ExternalProjects. Verification replays exact source, generated-output,
object, archive, linker-map and wheel evidence using original or retained bytes.
"""
from __future__ import annotations

from collections import deque
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shlex
import stat
import sys
import tarfile
import zipfile


COHORT_SHA256 = "584b47d20f2ec9ce3d3f98ed4e616c9b866de96d5033cd8f8406060b7d88a69f"
SDK_NAMES = (
    "aws_sdk_cpp", "aws_crt_cpp", "aws_c_auth", "aws_c_cal", "aws_c_common",
    "aws_c_compression", "aws_c_event_stream", "aws_c_http", "aws_c_io",
    "aws_c_mqtt", "aws_c_s3", "aws_c_sdkutils", "aws_checksums", "s2n_tls",
    "thrift", "grpc", "protobuf", "absl", "cares", "rapidjson", "jemalloc",
)
FAMILIES = {
    "25": {
        "source_policy": "arrow25-two-compute-patches-private-network-v2",
        "source_revision": "beccec0d0c451b7aa3e4530416ac431b3c035c69",
        "recipe_sha256": "881740f1bbac88722230b307a876ca18f009fe79f4406e2ca9baf17e615c413a",
        "legacy": frozenset({"boost"}),
    },
    "26": {
        "source_policy": "arrow26-integrated-private-network-v3",
        "source_revision": "446167169a0dc547e00e1bc29f417a40c0ca267a",
        "recipe_sha256": "ff52c1e7ca5ee5d146a4cc83af9a618862fe59de29d55e88f9c867f733f73473",
        "legacy": frozenset({"simdjson", "uriparser", "boost"}),
    },
}
EP_ARCHIVE_SHA256 = {
    "lz4": "537512904744b35e232912055ccf8ec66d768639ff3abe5788d90d792ec5f48b",
    "mimalloc": "37107a52c16baa80c5f74861dddda7b27bb9949e41a6637691867a94c88ca446",
    "orc_format": "584dfe2a4202946178fd8fc7d1239be7805b9ed4596ab2042dee739e7880992b",
    "snappy": "90f74bc1fbf78a6c56b3c4a082a05103b3a56bb17bca1a27e052ea11723292dc",
    "substrait": "f989a862f694e7dbb695925ddb7c4ce06aa6c51aca945105c075139aed7e55a2",
    "utf8proc": "6f4f1b639daa6dca9f80bc5db1233e9cbaa31a67790887106160b33ef743f136",
}
ORC_ARCHIVE_SHA256 = "52fc762332442e8b05d7182f8c035f9e04d945b9a52be22ab69f28b3f37d4500"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             "file hash must be an exact SHA256")
    return value


def _logical_path(value):
    # Historical identities are lexical. In retained mode they must never be
    # resolved, statted, opened, or traversed on the original filesystem.
    _require(isinstance(value, str) and value.startswith("/") and "\x00" not in value
             and "\\" not in value and posixpath.normpath(value) == value,
             "file/source identity must be a canonical absolute path")
    return PurePosixPath(value)


def _relative(value):
    _require(isinstance(value, str) and value not in ("", ".") and "\x00" not in value
             and "\\" not in value and not value.startswith("/")
             and all(part not in ("", ".", "..") for part in value.split("/")),
             "source inventory contains an unsafe relative path")
    return value


def _reference(value):
    _require(isinstance(value, dict), "file reference must be an object")
    _logical_path(value.get("path"))
    return {"path": value["path"], "sha256": _sha256(value.get("sha256"))}


class _Resolver:
    def __init__(self, retained_inputs=None):
        self.retained = None
        if retained_inputs is not None:
            _require(isinstance(retained_inputs, list), "retained inputs must be a file list")
            self.retained = {}
            for value in retained_inputs:
                row = _reference(value)
                self.retained[row["sha256"]] = row
        self.files = {}
        self.originals = {}

    def read(self, reference):
        reference = _reference(reference)
        selected = reference
        if self.retained is not None:
            selected = self.retained.get(reference["sha256"])
            _require(selected is not None, "required retained closure file is missing")
        path = Path(selected["path"])
        try:
            _require(not path.is_symlink() and path.is_file(),
                     "retained/source file must be a regular non-symlink")
            data = path.read_bytes()
        except OSError as exc:
            raise ValueError("required retained/source file cannot be read") from exc
        _require(hashlib.sha256(data).hexdigest() == reference["sha256"],
                 "retained/source file checksum mismatch")
        self.files[(selected["path"], reference["sha256"])] = dict(selected)
        self.originals[(reference["path"], reference["sha256"])] = reference
        return data

    def json(self, reference):
        try:
            value = json.loads(self.read(reference))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("retained JSON evidence is malformed") from exc
        _require(isinstance(value, dict), "retained JSON evidence must be an object")
        return value

    def recheck(self):
        for reference in list(self.files.values()):
            path = Path(reference["path"])
            try:
                _require(not path.is_symlink() and path.is_file(),
                         "retained closure file is no longer regular")
                with path.open("rb") as stream:
                    observed = hashlib.file_digest(stream, "sha256").hexdigest()
            except OSError as exc:
                raise ValueError("retained closure file disappeared") from exc
            _require(observed == reference["sha256"], "retained closure checksum changed")


def read_record(reference, *, retained_inputs=None):
    """Read exactly the referenced bytes, using only retained copies if given."""
    return _Resolver(retained_inputs).read(reference)


def verify_proof_union(schema_kind, legacy, supplemental):
    _require(isinstance(schema_kind, str) and schema_kind in FAMILIES, "unknown source proof family")
    _require(isinstance(legacy, dict) and set(legacy) == FAMILIES[schema_kind]["legacy"],
             "legacy source proof set differs from the exact family contract")
    _require(isinstance(supplemental, dict) and set(supplemental) == set(SDK_NAMES),
             "supplemental source proof cohort must contain exactly21 components")
    _require(set(legacy).isdisjoint(supplemental), "source proof union overlaps")
    combined = {**legacy, **supplemental}
    expected = FAMILIES[schema_kind]["legacy"] | set(SDK_NAMES)
    _require(set(combined) == expected and len(combined) == (24 if schema_kind == "26" else 22),
             "source proof union is incomplete")
    return combined


def _inventory(files):
    _require(isinstance(files, list) and bool(files), "source inventory must contain records")
    result = {}
    for value in files:
        _require(isinstance(value, dict), "source inventory entry must be an object")
        name = _relative(value.get("relative"))
        _require(name not in result, "source inventory contains a duplicate path")
        kind = value.get("type", "file")
        _require(kind in {"file", "generated", "directory", "symlink"},
                 "source inventory contains an unsupported type")
        row = {"relative": name, "type": kind}
        if kind in {"file", "generated"}:
            row["sha256"] = _sha256(value.get("sha256"))
        elif kind == "symlink":
            target = value.get("target")
            _require(isinstance(target, str) and bool(target) and not target.startswith("/")
                     and "\\" not in target and "\x00" not in target,
                     "source symlink target is unsafe")
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
            _require(resolved not in ("", ".", "..") and not resolved.startswith("../"),
                     "source symlink escapes its source root")
            row["target"] = target
        result[name] = row
    _validate_inventory_links(result)
    return result


def _validate_inventory_links(inventory):
    """Expand inventory links before parent components, without filesystem I/O."""
    completed = {}
    for initial, row in inventory.items():
        if row["type"] != "symlink":
            continue
        pending = deque(initial.split("/"))
        resolved, active = [], set()
        while pending:
            component = pending.popleft()
            if isinstance(component, tuple):
                link = component[0]
                active.remove(link)
                completed[link] = tuple(resolved)
                continue
            if component in ("", "."):
                continue
            if component == "..":
                _require(bool(resolved), "source symlink chain escapes its source root")
                resolved.pop()
                continue
            candidate = "/".join([*resolved, component])
            target = inventory.get(candidate)
            if target is None or target["type"] != "symlink":
                resolved.append(component)
                continue
            _require(candidate not in active, "source symlink chain contains a cycle")
            if candidate in completed:
                resolved = list(completed[candidate])
                continue
            active.add(candidate)
            # The marker completes this link before the original suffix is
            # interpreted. A later use can reuse its completed path safely.
            pending.appendleft((candidate,))
            pending.extendleft(reversed(target["target"].split("/")))


def _parents(names):
    result = set()
    for name in names:
        parent = PurePosixPath(name).parent
        while str(parent) != ".":
            result.add(str(parent))
            parent = parent.parent
    return result


def _check_current_layout(root, inventory):
    root = Path(root)
    _require(not root.is_symlink() and root.is_dir(), "source root is not a regular directory")
    observed = {}
    try:
        for directory, directories, files in os.walk(root, followlinks=False):
            for name in directories + files:
                path = Path(directory) / name
                relative = path.relative_to(root).as_posix()
                if path.is_symlink():
                    observed[relative] = ("symlink", os.readlink(path))
                elif path.is_dir():
                    observed[relative] = ("directory", None)
                elif path.is_file():
                    observed[relative] = ("file", None)
                else:
                    raise ValueError("source tree contains an unsupported object")
    except OSError as exc:
        raise ValueError("source tree cannot be inventoried") from exc
    implicit = _parents(inventory)
    _require(set(inventory) <= set(observed)
             and all(name in inventory or (kind == "directory" and name in implicit)
                     for name, (kind, _) in observed.items()),
             "source tree layout differs from its complete inventory")
    for name, row in inventory.items():
        kind, target = observed[name]
        _require(kind == ("file" if row["type"] == "generated" else row["type"]),
                 "source object type differs from its inventory")
        if kind == "symlink":
            _require(target == row["target"], "source symlink differs from its inventory")


def _verify_archive_source(archive_record, source_root, files, generated_files, resolver,
                           *, verified_replacements=None, qualified_directories=()):
    root = _logical_path(source_root)
    inventory = _inventory(files)
    _require(isinstance(generated_files, dict), "generated source derivations must be explicit")
    generated = {}
    replacements = verified_replacements or {}
    _require(isinstance(replacements, dict), "source replacements must be individually qualified")
    for relative, data in generated_files.items():
        name = _relative(relative)
        _require(isinstance(data, bytes), "generated source derivation must provide exact bytes")
        generated[name] = data
    if resolver.retained is None:
        _check_current_layout(str(root), inventory)

    with tarfile.open(fileobj=io.BytesIO(resolver.read(archive_record)), mode="r:*") as archive:
        members = archive.getmembers()
        archive_root = None
        expected = {}
        for member in members:
            name = member.name
            while name.startswith("./"):
                name = name[2:]
            name = name.rstrip("/")
            if not name:
                continue
            _relative(name)
            parts = name.split("/", 1)
            if archive_root is None:
                archive_root = parts[0]
            _require(parts[0] == archive_root, "source archive has multiple roots")
            if len(parts) == 1:
                _require(member.isdir(), "source archive root must be a directory")
                continue
            relative = parts[1]
            _require(relative not in expected, "source archive repeats a member path")
            _require(member.isfile() or member.isdir() or member.issym(),
                     "source archive contains an unqualified member type")
            expected[relative] = member
        _require(bool(expected), "source archive has no members")
        _require(set(expected) & set(generated) == set(replacements),
                 "generated source derivation overlaps unqualified original archive members")
        allowed_directories = _parents(set(expected) | set(generated))
        allowed_directories.update(_relative(name) for name in qualified_directories)
        _require(all(name in expected or name in generated
                     or (row["type"] == "directory" and name in allowed_directories)
                     for name, row in inventory.items()),
                 "source inventory contains unqualified extra/generated paths")
        _require(set(expected) <= set(inventory) | allowed_directories
                 and set(generated) <= set(inventory),
                 "source inventory omits archive/generated members")

        # getmembers() leaves compressed input at EOF. Replay regular members
        # in physical archive order so gzip only needs one initial rewind.
        for name, member in expected.items():
            row = inventory.get(name)
            if member.isdir():
                _require(row is None or row["type"] == "directory", "source directory layout changed")
            elif member.issym():
                _require(row is not None and row["type"] == "symlink"
                         and row["target"] == member.linkname, "source symlink differs from archive")
            else:
                _require(row is not None and row["type"] == "file", "source archive file is missing")
                stream = archive.extractfile(member)
                _require(stream is not None, "source archive regular member cannot be read")
                original_hash = hashlib.sha256(stream.read()).hexdigest()
                if name in replacements:
                    _require(replacements[name] == original_hash,
                             "generated replacement does not bind its original archive bytes")
                else:
                    _require(row["sha256"] == original_hash, "source bytes differ from pinned archive")
                resolver.read({"path": str(root / name), "sha256": row["sha256"]})

    for name, expected_bytes in generated.items():
        row = inventory[name]
        _require(row["type"] in {"file", "generated"}, "generated source is not a regular file")
        actual = resolver.read({"path": str(root / name), "sha256": row["sha256"]})
        _require(actual == expected_bytes, "generated source differs from its exact derivation")
        row["type"] = "file"
    return [inventory[name] for name in sorted(inventory)]


def verify_archive_source(archive_record, source_root, files, *, generated_files=None, retained_inputs=None):
    """Compare a complete source inventory to pinned archive and derived bytes."""
    try:
        return _verify_archive_source(archive_record, source_root, files,
                                      generated_files or {}, _Resolver(retained_inputs))
    except (tarfile.TarError, OSError) as exc:
        raise ValueError("source archive/tree proof could not be read") from exc


def derive_aws_sdk_config(template, *, cache, root_cmake, core_cmake, target_kind):
    """Derive the one SDKConfig.h output for the captured static/default branch."""
    _require(all(isinstance(value, bytes) for value in (template, root_cmake, core_cmake))
             and isinstance(cache, dict), "AWS SDK derivation requires pinned source bytes and cache")
    _require(target_kind == "STATIC_LIBRARY" and cache.get("BUILD_SHARED_LIBS") in {"OFF", "FALSE", "0"},
             "AWS SDK derivation requires the actual static build")
    _require(cache.get("CUSTOM_MEMORY_MANAGEMENT", "OFF") in {"OFF", "FALSE", "0"},
             "AWS custom memory management branch is not the reviewed OFF branch")
    try:
        root_text, core_text = root_cmake.decode(), core_cmake.decode()
    except UnicodeDecodeError as exc:
        raise ValueError("AWS SDK CMake derivation source is not UTF-8") from exc
    defaults = re.findall(r'option\(\s*CUSTOM_MEMORY_MANAGEMENT\s+"[^"]*"\s+(\w+)\s*\)',
                          root_text, re.S | re.I)
    _require(defaults == ["OFF"], "AWS memory management source default differs")
    branch = re.compile(
        r"if\(\s*CUSTOM_MEMORY_MANAGEMENT\s+OR\s*\(\s*BUILD_SHARED_LIBS\s+AND\s+NOT\s+DEFINED\s+CUSTOM_MEMORY_MANAGEMENT\s*\)\s*\)"
        r"\s*set\(\s*USE_AWS_MEMORY_MANAGEMENT\s+ON\s*\).*?"
        r"else\(\)\s*set\(\s*USE_AWS_MEMORY_MANAGEMENT\s+OFF\s*\).*?endif\(\)", re.S)
    _require(len(branch.findall(core_text)) == 1, "AWS SDK memory-management generator branch differs")
    generator = re.compile(
        r'configure_file\(\s*"\$\{CMAKE_CURRENT_SOURCE_DIR\}/include/aws/core/SDKConfig\.h\.in"'
        r'\s*"\$\{CMAKE_CURRENT_SOURCE_DIR\}/include/aws/core/SDKConfig\.h"\s+NEWLINE_STYLE\s+UNIX\s*\)', re.S)
    _require(len(generator.findall(core_text)) == 1, "AWS SDKConfig header generator differs")
    pattern = rb"(?m)^#cmakedefine USE_AWS_MEMORY_MANAGEMENT$"
    _require(len(re.findall(pattern, template)) == 1 and template.count(b"#cmakedefine") == 1,
             "AWS SDKConfig template differs or repeats its directive")
    return re.sub(pattern, b"/* #undef USE_AWS_MEMORY_MANAGEMENT */", template)


def _cache(data):
    try:
        lines = data.decode().splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("captured CMake cache is not UTF-8") from exc
    result = {}
    for line in lines:
        if not line or line.startswith(("#", "//")):
            continue
        match = re.fullmatch(r"([^:]+):[^=]*=(.*)", line)
        if match:
            key, value = match.groups()
            _require(key not in result, "captured CMake cache repeats an entry")
            result[key] = value
    return result


def _generated_files(name, entry, manifest, selected, resolver):
    result = {}
    derivations = entry.get("generated_derivations", [])
    _require(isinstance(derivations, list), "generated source derivations must be a list")
    for proof in derivations:
        _require(name == "aws_sdk_cpp" and isinstance(proof, dict)
                 and proof.get("kind") == "aws-sdkconfig-static-v1"
                 and not result, "unqualified ExternalProject/generated source derivation")
        root = _logical_path(entry["source_root"])
        inventory = _inventory(entry["files"])
        paths = {
            "template": "src/aws-cpp-sdk-core/include/aws/core/SDKConfig.h.in",
            "root_cmake": "CMakeLists.txt",
            "core_cmake": "src/aws-cpp-sdk-core/CMakeLists.txt",
        }
        captured = {}
        for field, relative in paths.items():
            row = inventory.get(relative)
            _require(row is not None and row["type"] == "file", "AWS derivation source is missing")
            reference = {"path": str(root / relative), "sha256": row["sha256"]}
            _require(proof.get(field) == reference, "AWS derivation source ownership differs")
            captured[field] = resolver.read(reference)
        cache_ref = _reference(proof.get("cache"))
        _require(cache_ref["path"] == manifest["context"]["work"] + "/cpp-build/CMakeCache.txt",
                 "AWS derivation cache belongs to another build")
        targets = [row for row in selected.get("targets", [])
                   if row.get("target") == "aws-cpp-sdk-core"]
        _require(len(targets) == 1 and targets[0].get("source", "").startswith(str(root) + "/"),
                 "AWS static target lacks exact source ownership")
        result["src/aws-cpp-sdk-core/include/aws/core/SDKConfig.h"] = derive_aws_sdk_config(
            captured["template"], cache=_cache(resolver.read(cache_ref)),
            root_cmake=captured["root_cmake"], core_cmake=captured["core_cmake"],
            target_kind=targets[0].get("kind"))
    return result


def _join(directory, value):
    _require(isinstance(value, str) and bool(value) and "\x00" not in value and "\\" not in value,
             "build graph path is malformed")
    return posixpath.normpath(value if value.startswith("/") else posixpath.join(str(directory), value))


def _under(path, root):
    return path == root or path.startswith(root + "/")


def _text(data):
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("source/command evidence must be UTF-8") from exc


def _physical_record(path):
    path = Path(path)
    try:
        _require(not path.is_symlink() and path.is_file(), "captured input/output must be a regular file")
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        raise ValueError("captured input/output cannot be read") from exc
    return _reference({"path": str(path), "sha256": checksum})


def _tree_files(root):
    root = Path(root)
    _require(not root.is_symlink() and root.is_dir(), "captured source root is not a regular directory")
    result = []
    try:
        for directory, directories, files in os.walk(root, followlinks=False):
            for name in directories + files:
                path = Path(directory) / name
                row = {"relative": path.relative_to(root).as_posix()}
                if path.is_symlink():
                    row.update(type="symlink", target=os.readlink(path))
                elif path.is_dir():
                    row["type"] = "directory"
                elif path.is_file():
                    row.update(type="file", sha256=_physical_record(path)["sha256"])
                else:
                    raise ValueError("captured source tree contains an unsupported object")
                result.append(row)
    except OSError as exc:
        raise ValueError("captured source tree cannot be inventoried") from exc
    result.sort(key=lambda row: row["relative"])
    _inventory(result)
    return result


class _Evidence:
    """Capture direct files, or replay only the declared hash-addressed closure."""

    def __init__(self, resolver, captured=None):
        self.resolver = resolver
        self.captured = captured
        self.references = None
        if captured is not None:
            closure = captured.get("closure")
            _require(isinstance(closure, list), "source attestation retained closure is missing")
            self.references = {}
            for value in closure:
                reference = _reference(value)
                _require(reference["path"] not in self.references,
                         "source attestation closure repeats an original path")
                self.references[reference["path"]] = reference
        self.tools = {}
        self.aliases = {}

    def reference(self, path, *, optional=False):
        path = str(_logical_path(str(path)))
        if self.references is not None:
            reference = self.references.get(path)
            if reference is None and optional:
                return None
            _require(reference is not None, "required captured source/command/artifact reference is missing: " + path)
        else:
            if optional and not Path(path).exists():
                return None
            reference = _physical_record(path)
        self.resolver.read(reference)
        return reference

    def read(self, path, *, optional=False):
        reference = self.reference(path, optional=optional)
        return None if reference is None else self.resolver.read(reference)

    def tree(self, root, captured_files=None):
        if self.captured is None:
            return _tree_files(root)
        _require(captured_files is not None, "captured ExternalProject inventory is missing")
        rows = _inventory(captured_files)
        result = [rows[name] for name in sorted(rows)]
        if self.resolver.retained is None:
            _check_current_layout(root, rows)
        return result

    def _tool_file(self, token):
        alias = None
        if self.captured is not None:
            alias = self.captured.get("artifact_closure", {}).get("generator_aliases", {}).get(token)
        else:
            current, links, seen = token, [], set()
            while Path(current).is_symlink():
                _require(current not in seen and len(seen) < 64, "generator tool alias is cyclic")
                seen.add(current)
                target = os.readlink(current)
                links.append({"path": current, "target": target})
                current = _join(posixpath.dirname(current), target)
            if links:
                alias = {"links": links, "file": self.reference(current)}
        if alias is None:
            return self.reference(token)
        _require(isinstance(alias, dict) and isinstance(alias.get("links"), list)
                 and bool(alias["links"]), "generator tool alias is malformed")
        current, seen = token, set()
        for link in alias["links"]:
            _require(isinstance(link, dict) and link.get("path") == current
                     and current not in seen and isinstance(link.get("target"), str),
                     "generator tool alias chain differs")
            seen.add(current)
            current = _join(posixpath.dirname(current), link["target"])
        reference = _reference(alias.get("file"))
        _require(current == reference["path"], "generator tool alias target differs")
        self.resolver.read(reference)
        self.aliases[token] = alias
        return reference

    def tool(self, role, token, toolchain):
        _require(isinstance(token, str) and bool(token), "generator/compiler tool is missing")
        if not token.startswith("/"):
            _require(token in {"nm", "gawk", "awk", "sh", "make", "install"},
                     "unqualified generator tool alias")
            token = "/usr/bin/" + token
        reference = self._tool_file(token)
        pinned = toolchain.get(role)
        _require(pinned is None or reference == _reference(pinned),
                 "generator/compiler differs from pinned toolchain: " + role)
        previous = self.tools.setdefault(role, reference)
        _require(previous == reference, "generator tool identity changed within proof")
        return reference


def _archive_index(reference, resolver):
    result, root = {}, None
    try:
        with tarfile.open(fileobj=io.BytesIO(resolver.read(reference)), mode="r:*") as bundle:
            for member in bundle.getmembers():
                name = member.name
                while name.startswith("./"):
                    name = name[2:]
                name = name.rstrip("/")
                if not name:
                    continue
                _relative(name)
                parts = name.split("/", 1)
                if root is None:
                    root = parts[0]
                _require(parts[0] == root, "ExternalProject source archive has multiple roots")
                if len(parts) == 1:
                    _require(member.isdir(), "ExternalProject source archive root is not a directory")
                    continue
                relative = parts[1]
                _require(relative not in result, "ExternalProject source archive repeats a path")
                if member.isfile():
                    stream = bundle.extractfile(member)
                    _require(stream is not None, "ExternalProject archive member cannot be read")
                    data = stream.read()
                    result[relative] = {"type": "file", "sha256": hashlib.sha256(data).hexdigest(), "data": data}
                elif member.isdir():
                    result[relative] = {"type": "directory"}
                elif member.issym():
                    result[relative] = {"type": "symlink", "target": member.linkname}
                else:
                    raise ValueError("ExternalProject archive has an unqualified member type")
    except (tarfile.TarError, OSError) as exc:
        raise ValueError("ExternalProject archive could not be inspected") from exc
    _require(bool(result), "ExternalProject source archive is empty")
    return result


def _ar_members(data):
    """Read regular GNU/BSD ar bytes without running ar or a native reader."""
    _require(data.startswith(b"!<arch>\n"), "static archive is not a regular ar container")
    position, names, result = 8, b"", []
    while position < len(data):
        _require(position + 60 <= len(data), "static archive member header is truncated")
        header = data[position:position + 60]
        _require(header[58:] == b"`\n", "static archive member header differs")
        try:
            size = int(header[48:58].strip())
            name = header[:16].decode("ascii").rstrip()
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("static archive member metadata is malformed") from exc
        _require(size >= 0 and position + 60 + size <= len(data), "static archive member is truncated")
        payload = data[position + 60:position + 60 + size]
        position += 60 + size + (size % 2)
        _require(position <= len(data), "static archive padding is truncated")
        if name == "//":
            names = payload
            continue
        if name in {"/", "/SYM64/", "__.SYMDEF", "__.SYMDEF SORTED"}:
            continue
        if name.startswith("#1/"):
            try:
                length = int(name[3:])
                _require(0 < length <= len(payload), "BSD archive member name is truncated")
                name = payload[:length].rstrip(b"\0").decode("utf-8")
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError("BSD archive member name is malformed") from exc
            payload = payload[length:]
        elif re.fullmatch(r"/\d+", name):
            offset = int(name[1:])
            end = names.find(b"/\n", offset)
            _require(0 <= offset < len(names) and end > offset,
                     "GNU archive long-name reference is malformed")
            name = _text(names[offset:end])
        else:
            name = name.removesuffix("/")
        _relative(name)
        result.append({"name": name, "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)})
    return result


def _ninja_tokens(value):
    # CMake emits escaped spaces/colons in paths and $ continuations. This is
    # lexical parsing only; variables are expanded only for known rule fields.
    marker = "\x01"
    value = value.replace("$ ", marker).replace("$:", ":").replace("$$", "$")
    return [token.replace(marker, " ") for token in value.split()]


def _ninja_blocks(data):
    lines = _text(data).replace("$\n", "").splitlines()
    blocks, current = [], None
    for line in lines:
        if line.startswith("build "):
            match = re.fullmatch(r"build (.*?):\s*(\S+)(?:\s+(.*))?", line)
            _require(match is not None, "captured Ninja build record is malformed")
            outputs, rule, inputs = match.groups()
            output_tokens = _ninja_tokens(outputs.split(" | ", 1)[0])
            input_tokens = _ninja_tokens((inputs or "").split(" |", 1)[0])
            current = {"outputs": output_tokens, "rule": rule, "inputs": input_tokens, "fields": {}}
            blocks.append(current)
        elif current is not None and line.startswith("  ") and " = " in line:
            key, value = line.strip().split(" = ", 1)
            _require(key not in current["fields"], "Ninja record repeats a field")
            current["fields"][key] = value
        elif line and not line.startswith((" ", "#")):
            current = None
    return blocks


def _ninja_rules(data):
    result, current = {}, None
    for line in _text(data).replace("$\n", "").splitlines():
        if line.startswith("rule "):
            current = line[5:].strip()
            _require(current not in result, "Ninja command rules repeat a name")
            result[current] = {}
        elif current is not None and line.startswith("  ") and " = " in line:
            key, value = line.strip().split(" = ", 1)
            result[current][key] = value
        elif line and not line.startswith((" ", "#")):
            current = None
    return result


def _compile_record(command, directory, command_record, source_root, evidence, toolchain,
                    *, source=None, output=None, require_pic=False):
    try:
        tokens = shlex.split(command) if isinstance(command, str) else list(command)
    except ValueError as exc:
        raise ValueError("captured compiler command is malformed") from exc
    _require(tokens and "-c" in tokens and "-o" in tokens,
             "selected object has no explicit compiler source/output command")
    compiler = tokens[0]
    roles = [role for role in ("cc", "cxx") if toolchain.get(role, {}).get("path") == compiler]
    if not roles:
        _require(compiler.startswith(("/usr/bin/", "/bin/")),
                 "selected object compiler is not the pinned compiler or its recorded system alias")
        roles = ["cxx" if "++" in PurePosixPath(compiler).name else "cc"]
    compiler_ref = evidence.tool(roles[0], compiler, toolchain)
    index = tokens.index("-o")
    _require(index + 1 < len(tokens), "compiler command omits its object output")
    object_path = _join(directory, tokens[index + 1])
    if source is None:
        inputs = [value for value in tokens[1:] if re.search(r"\.(c|cc|cpp|cxx|C)$", value)]
        _require(len(inputs) == 1, "compiler command source identity is ambiguous")
        source = _join(directory, inputs[0])
    else:
        source = _join(directory, source)
        _require(any(_join(directory, value) == source for value in tokens[1:] if not value.startswith("-")),
                 "compiler command does not name its selected source")
    _require(_under(source, source_root), "selected source/object belongs to another source tree")
    if output is not None:
        _require(object_path == _join(directory, output), "compiler graph object identity differs")
    pic = "-fPIC" in tokens or "-fpic" in tokens
    _require(not require_pic or pic, "selected static object command does not enable PIC")
    source_ref = evidence.reference(source)
    object_ref = evidence.reference(object_path)
    _require(evidence.resolver.read(object_ref).startswith(b"\x7fELF"), "selected compiler output is not an ELF object")
    return {"source": source_ref, "object": object_ref, "compiler": compiler_ref,
            "directory": str(directory), "command": tokens, "command_record": command_record, "pic": pic}


def _archive_objects(archive_ref, objects, evidence, *, require_pic=True):
    members = _ar_members(evidence.resolver.read(archive_ref))
    _require(bool(members), "selected ExternalProject static archive contains no objects")
    expected = {(PurePosixPath(row["object"]["path"]).name, row["object"]["sha256"]): row for row in objects}
    _require(len(expected) == len(objects), "selected archive object identities repeat")
    observed = [(row["name"], row["sha256"]) for row in members]
    _require(len(observed) == len(expected) and len(set(observed)) == len(observed)
             and set(observed) == set(expected), "static archive members differ from selected source/object mappings")
    _require(not require_pic or all(row["pic"] for row in objects), "selected archive contains a non-PIC object")
    return {"file": archive_ref, "members": members, "objects": [row["object"] for row in objects]}


def _unique_references(values):
    rows = {_reference(row)["path"]: _reference(row) for row in values}
    return [rows[name] for name in sorted(rows)]


def _generated_output(rule, path, inputs, command_record, evidence, *, original=None, command=None):
    _require(bool(inputs), "generated output has no pinned generator/template inputs")
    result = {"rule": rule, "output": evidence.reference(path),
              "inputs": _unique_references(inputs), "command_record": command_record}
    if original is not None:
        _require(original.get("type") == "file", "generated output replaces a non-file archive member")
        result["original_sha256"] = original["sha256"]
    if command is not None:
        result["command"] = command
    return result


def _ep_layout(name, work):
    target = "orc-format_ep" if name == "orc_format" else name + "_ep"
    base = work + "/cpp-build" + ("/_deps/orc-build" if name in {"lz4", "orc_format"} else "")
    source = base + "/" + target + "-prefix/src/" + target
    binary = source if name in {"jemalloc", "snappy"} else source + "-build"
    installed = {
        "jemalloc": source + "/dist/lib/libjemalloc_pic.a",
        "lz4": base + "/c++/libs/thirdparty/lz4_ep-install/lib/liblz4.a",
        "mimalloc": base + "/mimalloc_ep/src/mimalloc_ep/lib/libmimalloc.a",
        "snappy": base + "/snappy_ep/src/snappy_ep-install/lib/libsnappy.a",
        "utf8proc": base + "/utf8proc_ep-install/lib/libutf8proc.a",
        "substrait": base + "/release/libsubstrait.a",
        "orc_format": base + "/c++/src/liborc.a",
    }[name]
    return {"target": target, "base": base, "source": source, "binary": binary,
            "stamp": source + "-stamp", "installed": installed,
            "cfg": base + "/" + target + "-prefix/tmp/" + target + "-cfgcmd.txt"}


def _source_reference(relative, source, original, evidence):
    _require(relative in original and original[relative]["type"] == "file",
             "generator/template is not a pinned original archive member")
    reference = evidence.reference(source + "/" + relative)
    _require(reference["sha256"] == original[relative]["sha256"],
             "generator/template source bytes differ from original archive")
    return reference


def _cmake_entrypoint(name, layout, original, selected, evidence):
    relative, subdir, declaration, download_policy = "CMakeLists.txt", "", None, None
    if name == "lz4" and relative not in original:
        relative, subdir = "build/cmake/CMakeLists.txt", "build/cmake"
        owner_root = layout["base"].removesuffix("/_deps/orc-build") + "/_deps/orc-src"
        owner_relative = "cmake_modules/ThirdpartyToolchain.cmake"
        trees = [row for row in selected.get("source_trees", []) if row.get("path") == owner_root]
        _require(len(trees) == 1, "LZ4 source_subdir declaration lacks frozen ORC source ownership")
        inventory = _inventory(trees[0].get("files"))
        row = inventory.get(owner_relative)
        _require(row is not None and row["type"] == "file", "LZ4 source_subdir declaration is missing")
        declaration = evidence.reference(owner_root + "/" + owner_relative)
        _require(declaration["sha256"] == row["sha256"], "LZ4 source_subdir declaration bytes changed")
        text = _text(evidence.resolver.read(declaration))
        definitions = re.findall(r'set\(\s*LZ4_CONFIGURE\s+SOURCE_SUBDIR\s+"build/cmake"\s+'
                                 r'CMAKE_ARGS\s+\$\{LZ4_CMAKE_ARGS\}\s*\)', text, re.I)
        calls = re.findall(r"ExternalProject_Add\(\s*lz4_ep\b([^()]*)\)", text, re.I | re.S)
        _require(len(definitions) == 1 and len(calls) == 1 and "${LZ4_CONFIGURE}" in calls[0],
                 "LZ4 source-owned nested CMake entrypoint declaration differs")
        if not re.search(r"\bURL_HASH\b", calls[0]):
            # ORC 2.2.1 does not declare a CMake download hash for LZ4.
            # Prove that omission from its original pinned archive; the LZ4
            # archive still has its independent pin and full source proof.
            expected_url = "https://github.com/lz4/lz4/archive/v1.10.0.tar.gz"
            _require(re.findall(r'set\(\s*LZ4_VERSION\s+"([^"]+)"\s*\)', text) == ["1.10.0"]
                     and shlex.split(calls[0], comments=True) == [
                         "URL", "https://github.com/lz4/lz4/archive/v${LZ4_VERSION}.tar.gz",
                         "${LZ4_CONFIGURE}", "${THIRDPARTY_LOG_OPTIONS}",
                         "BUILD_BYPRODUCTS", "${LZ4_STATIC_LIB}"],
                     "LZ4 source-owned unhashed download declaration differs")
            owner_archive_path = (layout["base"].removesuffix("/_deps/orc-build")
                                  + "/_deps/orc-subbuild/orc-populate-prefix/src/orc-2.2.1.tar.gz")
            archives = [row for row in selected.get("downloaded_archives", [])
                        if row.get("path") == owner_archive_path and row.get("sha256") == ORC_ARCHIVE_SHA256]
            _require(len(archives) == 1, "LZ4 unhashed download lacks its pinned ORC source archive")
            owner_archive = _reference(archives[0])
            owner_original = _archive_index(owner_archive, evidence.resolver)
            _require(_source_reference(owner_relative, owner_root, owner_original, evidence) == declaration,
                     "LZ4 download declaration differs from original ORC source")
            download_policy = {"kind": "orc-lz4-source-owned-no-url-hash-v1", "upstream_hash": None,
                               "url": expected_url, "owner_archive": owner_archive,
                               "owner_declaration": declaration}
    reference = _source_reference(relative, layout["source"], original, evidence)
    result = {"source_subdir": subdir, "source_entrypoint": reference}
    if declaration:
        result["source_subdir_declaration"] = declaration
    if download_policy:
        result["download_hash_policy"] = download_policy
    return result


def _ep_configuration(name, layout, archive_ref, evidence, *, cmake_entrypoint=None):
    cfg_ref = evidence.reference(layout["cfg"])
    text = _text(evidence.resolver.read(cfg_ref)).strip()
    match = re.fullmatch(r"cmd='(.*)'", text, re.S)
    _require(match is not None, "ExternalProject configuration record is malformed")
    template = match[1].split(";") if match[1] else []
    command = list(template)
    if cmake_entrypoint is not None:
        subdir = cmake_entrypoint["source_subdir"]
        expected_source = layout["source"] + ("/" + subdir if subdir else "")
        placeholders = re.findall(r"<[^<>]*>", match[1])
        _require(set(placeholders) <= {"<SOURCE_DIR>", "<SOURCE_SUBDIR>", "<BINARY_DIR>"},
                 "ExternalProject configuration contains an unqualified placeholder")
        if placeholders:
            _require(template.count("-S") == 1 and template.count("-B") == 1,
                     "ExternalProject source/binary placeholder positions differ")
            source_index, binary_index = template.index("-S") + 1, template.index("-B") + 1
            _require(source_index < len(template) and binary_index < len(template)
                     and template[source_index] == "<SOURCE_DIR><SOURCE_SUBDIR>"
                     and template[binary_index] == "<BINARY_DIR>"
                     and all("<" not in value and ">" not in value for index, value in enumerate(template)
                             if index not in {source_index, binary_index}),
                     "ExternalProject configuration source/binary placeholder ownership differs")
            command[source_index], command[binary_index] = expected_source, layout["binary"]
        for flag, expected in (("-S", expected_source), ("-B", layout["binary"])):
            if flag in command:
                index = command.index(flag) + 1
                _require(command.count(flag) == 1 and index < len(command) and command[index] == expected,
                         "ExternalProject configuration rebinds its source or binary directory")
    else:
        _require(not re.search(r"[<>]", match[1]), "non-CMake ExternalProject configuration contains a placeholder")
    command_record = evidence.reference(layout["stamp"] + "/" + layout["target"]
                                         + "-configure-Release.cmake", optional=True)
    if command_record:
        recorded = re.findall(r'^set\(command "(.*)"\)$',
                              _text(evidence.resolver.read(command_record)), re.M)
        _require(recorded == [";".join(command)], "ExternalProject configure command differs from its captured stamp")
    urlinfo = evidence.reference(layout["stamp"] + "/" + layout["target"] + "-urlinfo.txt")
    info = _text(evidence.resolver.read(urlinfo))
    policy = cmake_entrypoint.get("download_hash_policy") if cmake_entrypoint else None
    hashes = re.findall(r"^hash=(.*)$", info, re.M)
    if policy is None:
        hash_matches = hashes == ["SHA256=" + archive_ref["sha256"]]
    else:
        hash_matches = (name == "lz4" and archive_ref["sha256"] == EP_ARCHIVE_SHA256["lz4"]
                        and hashes == [""] and re.findall(r"^url\(s\)=(.*)$", info, re.M) == [policy["url"]])
    _require(re.findall(r"^source_dir=(.*)$", info, re.M) == [layout["source"]]
             and hash_matches,
             "ExternalProject archive/source configuration ownership differs")
    result = {"configuration": cfg_ref, "configure_template": template, "configure_command": command,
            "configure_command_record": command_record, "download_record": urlinfo}
    if cmake_entrypoint is not None:
        result.update(cmake_entrypoint)
    return result


def _ninja_objects(layout, source_root, evidence, toolchain):
    graph_ref = evidence.reference(layout["binary"] + "/build.ninja")
    rules_ref = evidence.reference(layout["binary"] + "/CMakeFiles/rules.ninja")
    blocks = _ninja_blocks(evidence.resolver.read(graph_ref))
    rules = _ninja_rules(evidence.resolver.read(rules_ref))
    objects = {}
    for block in blocks:
        if "_COMPILER__" not in block["rule"]:
            continue
        _require(len(block["outputs"]) == 1 and len(block["inputs"]) == 1,
                 "ExternalProject compile graph source/output is ambiguous")
        output = _join(layout["binary"], block["outputs"][0])
        if evidence.reference(output, optional=True) is None:
            continue
        command = rules.get(block["rule"], {}).get("command")
        _require(isinstance(command, str), "ExternalProject compiler rule is missing")
        values = {**block["fields"], "out": block["outputs"][0], "in": block["inputs"][0],
                  "LAUNCHER": "", "CODE_CHECK": ""}

        def expand(match):
            key = match[1] or match[2]
            _require(key in values, "ExternalProject compiler rule has an unbound field: " + key)
            return values[key]

        command = re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}|\$([A-Za-z_][A-Za-z_0-9]*)", expand, command)
        row = _compile_record(command, layout["binary"], graph_ref, source_root, evidence, toolchain,
                              source=block["inputs"][0], output=block["outputs"][0])
        _require(output not in objects, "ExternalProject compile graph repeats an object")
        row["rule_record"] = rules_ref
        objects[output] = row
    archives = []
    for block in blocks:
        if "_STATIC_LIBRARY_LINKER__" not in block["rule"]:
            continue
        _require(len(block["outputs"]) == 1, "ExternalProject archive graph output is ambiguous")
        output = _join(layout["binary"], block["outputs"][0])
        reference = evidence.reference(output, optional=True)
        if reference is None:
            continue
        inputs = [_join(layout["binary"], value) for value in block["inputs"]]
        _require(inputs and all(value in objects for value in inputs),
                 "ExternalProject archive lacks selected object/compiler provenance")
        linker = rules.get(block["rule"], {}).get("command", "")
        # The source-owned Ninja rule must name the same archiver selected by
        # the captured configure command/cache; it is never executed here.
        archivers = re.findall(r"(/[^\s]+(?:-|/)ar)\s", linker)
        if toolchain.get("ar"):
            archivers = [toolchain["ar"]["path"]] if toolchain["ar"]["path"] in linker else []
        _require(len(set(archivers)) == 1, "ExternalProject static archiver command is unqualified")
        ar_ref = evidence.tool("ar", archivers[0], toolchain)
        proof = _archive_objects(reference, [objects[value] for value in inputs], evidence)
        proof.update(command_record=graph_ref, rule_record=rules_ref, archiver=ar_ref)
        archives.append(proof)
    _require(bool(archives), "ExternalProject static archive graph produced no qualified archive")
    return graph_ref, rules_ref, list(objects.values()), archives


def _make_variable(data, key):
    text = _text(data).replace("\\\n", " ")
    values = re.findall(r"^" + re.escape(key) + r"\s*(?::=|=)\s*(.*)$", text, re.M)
    _require(len(values) == 1, "source-owned Makefile variable differs: " + key)
    return values[0]


def _shell_list(data, key):
    result = ""
    for value in re.findall(r'^' + re.escape(key) + r'="([^"\n]*)"$', _text(data), re.M):
        result = value.replace("${" + key + "}", result).replace("${install_suffix}", "")
        _require("$" not in result, "Jemalloc generator list has an unqualified substitution")
    return result.split()


def _jemalloc_project(layout, original, configuration, evidence, toolchain):
    source = layout["source"]
    configure = _source_reference("configure", source, original, evidence)
    template = _source_reference("Makefile.in", source, original, evidence)
    command = configuration["configure_command"]
    _require(command and command[0] == "./configure", "Jemalloc generator command differs")
    for role, prefix in (("cc", "CC="), ("ar", "AR=")):
        selected = [value[len(prefix):] for value in command if value.startswith(prefix)]
        _require(len(selected) == 1, "Jemalloc configure tool selection is missing or repeated")
        evidence.tool(role, selected[0], toolchain)
    flags = {value for value in command if value.startswith("--") and "=" not in value}
    _require(flags == {"--without-export", "--disable-shared", "--disable-cxx", "--disable-libdl",
                       "--disable-initial-exec-tls"}, "Jemalloc configure policy differs")
    for flag, expected in (("--with-jemalloc-prefix", "je_arrow_"),
                           ("--with-private-namespace", "je_arrow_private_")):
        _require([value for value in command if value.startswith(flag + "=")] == [flag + "=" + expected],
                 "Jemalloc source-owned namespace configuration differs")
    for flag, expected in (("--prefix", source + "/dist"), ("--libdir", source + "/dist/lib")):
        values = [value.split("=", 1)[1] for value in command if value.startswith(flag + "=")]
        _require(len(values) == 1 and posixpath.normpath(values[0]) == expected,
                 "Jemalloc installation belongs to another build")
    makefile = evidence.reference(source + "/Makefile")
    make_data = evidence.resolver.read(makefile)
    configure_data = original["configure"]["data"]
    generated_names = set()
    generator_inputs = [configure, template, configuration["configuration"]]
    for key in ("cfghdrs_out", "cfgoutputs_out"):
        declared = _shell_list(configure_data, key)
        observed = _make_variable(make_data, key).split()
        if not declared:
            # A small literal Makefile.in is useful for independent fixtures;
            # real Autoconf output lists are declared in pinned configure.
            declared = _make_variable(original["Makefile.in"]["data"], key).split()
        _require(declared == observed and all("$" not in value and "@" not in value for value in declared),
                 "Jemalloc generated output declarations differ from pinned source")
        generated_names.update(_relative(value) for value in declared)
    for key in ("cfghdrs_in", "cfgoutputs_in"):
        for relative in _shell_list(configure_data, key):
            generator_inputs.append(_source_reference(relative, source, original, evidence))
    bins = ("bin/jemalloc-config", "bin/jemalloc.sh", "bin/jeprof")
    if any(name in original or name + ".in" in original for name in bins):
        declarations = re.findall(
            r'^ac_config_files="\$ac_config_files \$cfgoutputs_tup config\.stamp ([^"\n]+)"$',
            _text(configure_data), re.M)
        expected_bins = " ".join("$(objroot)" + name for name in bins)
        _require(declarations == [" ".join(bins)]
                 and _make_variable(original["Makefile.in"]["data"], "BINS") == expected_bins
                 and _make_variable(make_data, "BINS") == expected_bins,
                 "Jemalloc configured bin declarations differ from pinned source")
        for relative in bins:
            _require(original.get(relative, {}).get("type") == "file",
                     "Jemalloc configured bin has no original release source")
            generator_inputs.append(_source_reference(relative + ".in", source, original, evidence))
        generated_names.update(bins)
    c_sources = re.findall(r"\$\(srcroot\)([^\s]+\.c)",
                           _make_variable(original["Makefile.in"]["data"], "C_SRCS"))
    _require(bool(c_sources), "Jemalloc source-owned C source list is missing")
    log_ref = evidence.reference(layout["stamp"] + "/jemalloc_ep-build-out.log")
    log = _text(evidence.resolver.read(log_ref))
    objects = {}
    for line in log.splitlines():
        if not line.startswith(toolchain["cc"]["path"] + " ") or " -c " not in line:
            continue
        row = _compile_record(line, source, log_ref, source, evidence, toolchain)
        relative = posixpath.relpath(row["source"]["path"], source)
        _require(relative in c_sources, "Jemalloc object consumes a source outside its pinned source list")
        _source_reference(relative, source, original, evidence)
        _require(row["object"]["path"] not in objects, "Jemalloc log repeats a compiler output")
        objects[row["object"]["path"]] = row
    _require(bool(objects), "Jemalloc completed compiler evidence is missing")
    archives = []
    for line in log.splitlines():
        tokens = shlex.split(line)
        if not tokens or tokens[0] != evidence.tools["ar"]["path"]:
            continue
        _require(len(tokens) > 3 and re.fullmatch(r"[a-zA-Z]+", tokens[1]),
                 "Jemalloc archiver command differs")
        archive_path = _join(source, tokens[2])
        selected = [_join(source, token) for token in tokens[3:]]
        _require(all(path in objects for path in selected), "Jemalloc archive has an unselected object")
        pic = archive_path.endswith("_pic.a")
        proof = _archive_objects(evidence.reference(archive_path), [objects[path] for path in selected],
                                 evidence, require_pic=pic)
        proof.update(command=tokens, command_record=log_ref, archiver=evidence.tools["ar"])
        archives.append(proof)
    _require(any(row["file"]["path"].endswith("/lib/libjemalloc_pic.a") for row in archives),
             "Jemalloc PIC archive/object evidence is missing")
    generated_names.update({"config.log", "config.status"})
    generated = []
    command_record = configuration["configure_command_record"] or configuration["configuration"]
    for relative in sorted(generated_names):
        reference = evidence.reference(source + "/" + relative, optional=True)
        _require(reference is not None or relative == "config.log", "Jemalloc declared generated source is missing")
        if reference is None:
            continue
        inputs = list(generator_inputs)
        candidate = relative + ".in"
        if candidate in original:
            inputs.append(_source_reference(candidate, source, original, evidence))
        generated.append(_generated_output("jemalloc-configure-v1", source + "/" + relative, inputs,
                                           command_record, evidence, original=original.get(relative), command=command))
    for row in objects.values():
        generated.append(_generated_output("compile-object-v1", row["object"]["path"],
                                           [row["source"], row["compiler"]], log_ref, evidence,
                                           command=row["command"]))
    for archive in archives:
        generated.append(_generated_output("archive-objects-v1", archive["file"]["path"],
                                           [archive["archiver"], *archive["objects"]], log_ref, evidence,
                                           command=archive["command"]))
    # nm/awk dependency and symbol outputs are declared by Makefile's pinned
    # source rules. Only names corresponding to recorded compiler outputs exist.
    source_objects = list(objects.values())
    for row in source_objects:
        object_path = row["object"]["path"]
        dependency = object_path[:-2] + ".d"
        if evidence.reference(dependency, optional=True):
            generated.append(_generated_output("jemalloc-dependencies-v1", dependency,
                                               [row["source"], row["compiler"], template], log_ref, evidence))
        if object_path.endswith(".sym.o"):
            symbols = object_path[:-2]
            if evidence.reference(symbols, optional=True):
                script = source + "/include/jemalloc/internal/private_symbols.awk"
                inputs = [row["object"], evidence.reference(script), template]
                for role, token in (("nm", "nm"), ("awk", "gawk")):
                    inputs.append(evidence.tool(role, token, toolchain))
                generated.append(_generated_output("jemalloc-symbols-v1", symbols, inputs, log_ref, evidence))
    namespace_script = "include/jemalloc/internal/private_namespace.sh"
    if namespace_script in original:
        script_ref = _source_reference(namespace_script, source, original, evidence)
        symbols = [row["output"] for row in generated if row["rule"] == "jemalloc-symbols-v1"]
        _require(symbols, "Jemalloc private namespace has no selected symbol inputs")
        generated_name = source + "/include/jemalloc/internal/private_namespace.gen.h"
        header_name = source + "/include/jemalloc/internal/private_namespace.h"
        generated.append(_generated_output("jemalloc-private-namespace-v1", generated_name,
                                           [script_ref, template, *symbols], log_ref, evidence))
        generated_ref, header_ref = evidence.reference(generated_name), evidence.reference(header_name)
        _require(generated_ref["sha256"] == header_ref["sha256"], "Jemalloc generated namespace header copy differs")
        generated.append(_generated_output("generated-copy-v1", header_name, [generated_ref, template], log_ref, evidence))
    return generated, list(objects.values()), archives, generator_inputs


def _cmake_configure_outputs(source, binary, original, evidence, *, entrypoint="CMakeLists.txt"):
    declared = {}
    project = ""
    root_text = _text(original.get(entrypoint, {}).get("data", b""))
    match = re.search(r"\bproject\(\s*([A-Za-z_][A-Za-z_0-9-]*)", root_text, re.I)
    if match:
        project = match[1]
    project_source = _join(source, posixpath.dirname(entrypoint) or ".")
    values = {"PROJECT_BINARY_DIR": binary, "CMAKE_BINARY_DIR": binary, "CMAKE_CURRENT_BINARY_DIR": binary,
              "CMAKE_CURRENT_SOURCE_DIR": project_source, "PROJECT_SOURCE_DIR": project_source, "PROJECT_NAME": project}

    def expand(value):
        for key, replacement in values.items():
            value = value.replace("${" + key + "}", replacement)
        return None if "$" in value else value

    for relative, row in original.items():
        if row["type"] != "file" or not (relative.endswith(".cmake") or relative.endswith("CMakeLists.txt")):
            continue
        text = _text(row["data"])
        cmake_ref = _source_reference(relative, source, original, evidence)
        current_source = _join(source, posixpath.dirname(relative) or ".")
        values["CMAKE_CURRENT_SOURCE_DIR"] = current_source
        for match in re.finditer(r"\b(configure_file|configure_package_config_file|write_basic_package_version_file)"
                                 r"\s*\(([^()]*)\)", text, re.S | re.I):
            try:
                args = shlex.split(match[2], comments=True)
            except ValueError as exc:
                raise ValueError("source-owned CMake output declaration is malformed") from exc
            command = match[1].lower()
            needed = 1 if command == "write_basic_package_version_file" else 2
            if len(args) < needed:
                continue
            output = expand(args[needed - 1])
            if output is None:
                continue
            output = _join(binary, output)
            _require(_under(output, binary), "CMake configure output escapes its build directory")
            inputs = [cmake_ref]
            if needed == 2:
                template = expand(args[0])
                if template is None:
                    continue
                template_path = _join(current_source, template)
                if not _under(template_path, source):
                    continue
                template_relative = posixpath.relpath(template_path, source)
                if template_relative not in original:
                    continue
                inputs.append(_source_reference(template_relative, source, original, evidence))
            declared[output] = _unique_references(inputs)
    return declared


def _cmake_metadata(relative, source_cmake, version):
    fixed = {".ninja_deps", ".ninja_log", "CMakeCache.txt", "build.ninja", "cmake_install.cmake",
             "install_manifest.txt", "CMakeFiles/rules.ninja", "CMakeFiles/InstallScripts.json",
             "CMakeFiles/TargetDirectories.txt", "CMakeFiles/cmake.check_cache",
             "CMakeFiles/CMakeConfigureLog.yaml", "CMakeFiles/CMakeOutput.log", "CMakeFiles/CMakeError.log"}
    if relative in fixed:
        return True
    if version:
        prefix = "CMakeFiles/" + version + "/"
        probes = {"CMakeCCompiler.cmake", "CMakeCXXCompiler.cmake", "CMakeSystem.cmake",
                  "CMakeDetermineCompilerABI_C.bin", "CMakeDetermineCompilerABI_CXX.bin",
                  "CompilerIdC/CMakeCCompilerId.c", "CompilerIdC/a.out",
                  "CompilerIdCXX/CMakeCXXCompilerId.cpp", "CompilerIdCXX/a.out"}
        if relative.startswith(prefix) and relative[len(prefix):] in probes:
            return True
    export = re.fullmatch(r"CMakeFiles/Export/[0-9a-f]{32}/([^/]+?)(?:-release)?\.cmake", relative)
    if export:
        # Export filenames/target names must be literally declared by the
        # pinned package CMake source; a directory-wide exemption is forbidden.
        name = export[1]
        return re.search(r"(?i)\b(?:EXPORT|FILE)\s+" + re.escape(name) + r"(?:\.cmake)?\b", source_cmake) is not None
    return False


def _cmake_project(name, layout, original, configuration, evidence, toolchain, version, captured):
    source, binary = layout["source"], layout["binary"]
    entrypoint = posixpath.relpath(configuration["source_entrypoint"]["path"], source)
    cmake_ref = _source_reference(entrypoint, source, original, evidence)
    configure = configuration["configure_command"]
    _require(configure and configure[0] == toolchain["cmake"]["path"]
             and [value for value in configure if value.startswith("-DBUILD_SHARED_LIBS=")] == ["-DBUILD_SHARED_LIBS=OFF"],
             "ExternalProject CMake/static source configuration differs")
    cmake_tool = evidence.tool("cmake", configure[0], toolchain)
    graph, rules, objects, archives = _ninja_objects(layout, source, evidence, toolchain)
    selected = [row for row in archives if PurePosixPath(row["file"]["path"]).name
                == PurePosixPath(layout["installed"]).name]
    _require(len(selected) == 1, "ExternalProject selected archive identity is ambiguous")
    installed = evidence.reference(layout["installed"])
    _require(installed["sha256"] == selected[0]["file"]["sha256"],
             "ExternalProject installed archive differs from its selected build archive")
    selected[0]["installed"] = installed
    build_files = evidence.tree(binary, captured.get("build_files") if captured else None)
    declared = _cmake_configure_outputs(source, binary, original, evidence, entrypoint=entrypoint)
    all_cmake = "\n".join(_text(row["data"]) for relative, row in original.items()
                           if row["type"] == "file" and (relative.endswith(".cmake")
                                                          or relative.endswith("CMakeLists.txt")))
    inputs = [cmake_ref, cmake_tool, configuration["configuration"]]
    if configuration.get("source_subdir_declaration"):
        inputs.append(configuration["source_subdir_declaration"])
    command_record = configuration["configure_command_record"] or configuration["configuration"]
    generated, represented = [], set()
    for row in objects:
        generated.append(_generated_output("compile-object-v1", row["object"]["path"],
                                           [row["source"], row["compiler"], rules], graph, evidence,
                                           command=row["command"]))
        represented.add(row["object"]["path"])
    for archive in archives:
        generated.append(_generated_output("archive-objects-v1", archive["file"]["path"],
                                           [archive["archiver"], *archive["objects"], rules], graph, evidence))
        represented.add(archive["file"]["path"])
    if installed["path"] not in represented:
        generated.append(_generated_output("installed-copy-v1", installed["path"],
                                           [selected[0]["file"], cmake_ref], command_record, evidence))
    blocks = _ninja_blocks(evidence.resolver.read(graph))
    executable_outputs = {value: block for block in blocks if "_EXECUTABLE_LINKER__" in block["rule"]
                          for value in (_join(binary, name) for name in block["outputs"])}
    object_by_path = {row["object"]["path"]: row for row in objects}
    # CMake's own generator templates are captured when compiler-identification
    # outputs are retained inside an in-source EP. They are never executed.
    module_inputs = []
    cache_data = evidence.read(binary + "/CMakeCache.txt")
    cache = _cache(cache_data)
    _require(cache.get("BUILD_SHARED_LIBS") in {"OFF", "FALSE", "0"},
             "ExternalProject actual cache selects shared libraries")
    expected_home = posixpath.dirname(configuration["source_entrypoint"]["path"])
    _require(cache.get("CMAKE_HOME_DIRECTORY", expected_home) == expected_home
             and cache.get("CMAKE_CACHEFILE_DIR", binary) == binary,
             "ExternalProject actual cache source/binary ownership differs")
    cmake_root = cache.get("CMAKE_ROOT")
    if cmake_root:
        for relative in ("CMakeCCompiler.cmake.in", "CMakeCXXCompiler.cmake.in", "CMakeSystem.cmake.in",
                         "CMakeCCompilerId.c.in", "CMakeCXXCompilerId.cpp.in",
                         "CMakeCCompilerABI.c", "CMakeCXXCompilerABI.cpp", "CMakeDetermineCompilerId.cmake",
                         "CMakeDetermineCompilerABI.cmake", "CMakePackageConfigHelpers.cmake",
                         "WriteBasicConfigVersionFile.cmake"):
            reference = evidence.reference(cmake_root + "/Modules/" + relative, optional=True)
            if reference:
                module_inputs.append(reference)
    inputs += module_inputs
    for row in build_files:
        if row["type"] == "directory":
            continue
        if row["type"] == "symlink":
            _require(binary == source and row["relative"] in original,
                     "unqualified ExternalProject generated build symlink")
            continue
        path = binary + "/" + row["relative"]
        if binary == source and row["relative"] in original:
            continue
        if path in represented:
            continue
        if path in declared:
            generated.append(_generated_output("cmake-configure-file-v1", path,
                                               [*inputs, *declared[path]], command_record, evidence,
                                               original=original.get(row["relative"])))
        elif _cmake_metadata(row["relative"], all_cmake, version):
            generated.append(_generated_output("cmake-build-metadata-v1", path, inputs, command_record, evidence))
        elif path in executable_outputs or path.removesuffix(".map") in executable_outputs:
            block = executable_outputs.get(path, executable_outputs.get(path.removesuffix(".map")))
            selected_objects = [_join(binary, value) for value in block["inputs"]]
            _require(selected_objects and all(value in object_by_path for value in selected_objects),
                     "ExternalProject executable output lacks selected source/object provenance")
            generated.append(_generated_output("cmake-executable-v1", path,
                                               [*inputs, *(object_by_path[value]["object"] for value in selected_objects)],
                                               graph, evidence))
        else:
            raise ValueError("unqualified ExternalProject generated build/source path: " + row["relative"])
    qualified_directories = set()
    if version:
        qualified_directories.update({"CMakeFiles/" + version + "/CompilerIdC/tmp",
                                      "CMakeFiles/" + version + "/CompilerIdCXX/tmp"})
    qualified_directories.update({"CMakeFiles/CMakeScratch", "CMakeFiles/pkgRedirects"})
    known_parents = _parents(row["relative"] for row in build_files if row["type"] != "directory")
    for row in build_files:
        if row["type"] == "directory":
            _require(row["relative"] in known_parents or row["relative"] in qualified_directories
                     or (binary == source and original.get(row["relative"], {}).get("type") == "directory"),
                     "unqualified ExternalProject generated build directory")
    return generated, objects, archives, build_files, qualified_directories


def _command_tokens(command):
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise ValueError("captured native generator command is malformed") from exc
    if tokens[:1] == ["cd"]:
        _require(len(tokens) > 3 and tokens[2] == "&&", "captured generator working directory differs")
        return tokens[1], tokens[3:]
    return None, tokens


def _main_object_records(compile_commands, graph_blocks, archive_path, work, evidence, toolchain, command_record):
    rows = {}
    for entry in compile_commands:
        _require(isinstance(entry, dict), "main compile database entry is malformed")
        directory = entry.get("directory")
        _logical_path(directory)
        command = entry.get("command", entry.get("arguments"))
        if not command:
            continue
        tokens = shlex.split(command) if isinstance(command, str) else list(command)
        if "-o" not in tokens:
            continue
        index = tokens.index("-o")
        _require(index + 1 < len(tokens), "main compile database omits its object output")
        output = _join(directory, tokens[index + 1])
        rows[output] = entry
    archive_blocks = [row for row in graph_blocks
                      if "_STATIC_LIBRARY_LINKER__" in row["rule"]
                      and any(_join(work + "/cpp-build", output) == archive_path for output in row["outputs"])]
    if archive_blocks:
        _require(len(archive_blocks) == 1, "main static archive build graph is ambiguous")
        selected = [_join(work + "/cpp-build", value) for value in archive_blocks[0]["inputs"]]
    else:
        # Independently authored small fixtures can expose the same object
        # ownership through the compile database and exact ar payload hashes.
        members = _ar_members(evidence.read(archive_path))
        identities = {(row["name"], row["sha256"]) for row in members}
        selected = [path for path in rows if any(PurePosixPath(path).name == row["name"] for row in members)
                    and (PurePosixPath(path).name, evidence.reference(path)["sha256"]) in identities]
    _require(selected and all(path in rows for path in selected),
             "main archive lacks selected source/object compile mappings")
    result = []
    for output in selected:
        entry = rows[output]
        row = _compile_record(entry.get("command", entry.get("arguments")), entry["directory"], command_record,
                              work, evidence, toolchain, source=entry["file"], output=output, require_pic=True)
        source_data = evidence.resolver.read(row["source"])
        includes = re.findall(r'^#include "([^"\n]+)"', _text(source_data), re.M)
        if "Unity" in PurePosixPath(row["source"]["path"]).parts:
            _require(bool(includes), "Unity compilation unit has no captured source inputs")
            row["included_sources"] = []
            for include in includes:
                path = _join(posixpath.dirname(row["source"]["path"]), include)
                _require(_under(path, work), "Unity source input belongs to another build")
                row["included_sources"].append(evidence.reference(path))
        result.append(row)
    return result


def _proto_project(name, layout, original, evidence, toolchain, selected, work, arrow, compile_info):
    graph_ref = evidence.reference(work + "/cpp-build/build.ninja")
    graph_blocks = _ninja_blocks(evidence.resolver.read(graph_ref))
    output_root = (work + "/cpp-build/substrait_ep-generated" if name == "substrait"
                   else work + "/cpp-build/_deps/orc-build/c++/src")
    source_root = layout["source"]
    targets = [row for row in selected.get("targets", []) if row.get("target") == "protoc"]
    _require(len(targets) == 1 and _under(targets[0].get("source", ""), work + "/cpp-build/_deps/protobuf-src"),
             "native protobuf generator has no exact cohort source ownership")
    target_artifacts = set(targets[0].get("artifacts", []))
    generated = []
    for block in graph_blocks:
        outputs = [_join(work + "/cpp-build", value) for value in block["outputs"]]
        if not outputs or not all(_under(path, output_root) and path.endswith((".pb.cc", ".pb.h")) for path in outputs):
            continue
        _require(block["rule"] == "CUSTOM_COMMAND" and isinstance(block["fields"].get("COMMAND"), str),
                 "protobuf generated source has no exact native generator command")
        directory, tokens = _command_tokens(block["fields"]["COMMAND"])
        directory = directory or work + "/cpp-build"
        _require(tokens and tokens[0].startswith(work + "/cpp-build/_deps/protobuf-build/"),
                 "protobuf generated source selects another native generator")
        generator = evidence.tool("protoc", tokens[0], toolchain)
        _require(tokens[0] in target_artifacts or generator["path"] in target_artifacts,
                 "protobuf generator artifact differs from its selected target")
        proto_inputs = [_join(directory, value) for value in tokens[1:] if value.endswith(".proto")]
        _require(len(proto_inputs) == 1 and (_under(proto_inputs[0], source_root)
                 or _under(proto_inputs[0], arrow + "/cpp/proto")),
                 "protobuf generator input belongs to another source tree")
        proto = evidence.reference(proto_inputs[0])
        relative = posixpath.relpath(proto["path"], source_root)
        if _under(proto["path"], source_root):
            _require(relative in original and original[relative].get("sha256") == proto["sha256"],
                     "protobuf generator input differs from its pinned source archive")
        output_flags = [value.split("=", 1)[1] for value in tokens if value.startswith("--cpp_out=")]
        _require(len(output_flags) == 1 and _join(directory, output_flags[0]) == output_root,
                 "protobuf generator output directory differs")
        include_roots = []
        for index, token in enumerate(tokens[1:], 1):
            value = tokens[index + 1] if token == "-I" and index + 1 < len(tokens) else token[2:] if token.startswith("-I") else None
            if value:
                include_roots.append(_join(directory, value))
        matching_roots = [root for root in include_roots if _under(proto["path"], root)]
        _require(bool(matching_roots), "protobuf input has no source-owned include root")
        relative = posixpath.relpath(proto["path"], max(matching_roots, key=len))
        stem = output_root + "/" + relative.removesuffix(".proto")
        _require(set(outputs) == {stem + ".pb.cc", stem + ".pb.h"}, "protobuf source/output identity differs")
        inputs = [proto, generator]
        for path in outputs:
            generated.append(_generated_output("protoc-cpp-v1", path, inputs, graph_ref, evidence, command=tokens))
    _require(bool(generated), "selected protobuf source generator output evidence is missing")
    archive_ref = evidence.reference(layout["installed"])
    command_ref = _reference(compile_info.get("compile_commands"))
    try:
        database = json.loads(evidence.resolver.read(command_ref))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("main compile database is malformed") from exc
    _require(isinstance(database, list), "main compile database must contain entries")
    objects = _main_object_records(database, graph_blocks, layout["installed"], work, evidence, toolchain, command_ref)
    archive = _archive_objects(archive_ref, objects, evidence)
    archive["command_record"] = graph_ref
    for row in objects:
        generated.append(_generated_output("compile-object-v1", row["object"]["path"],
                                           [row["source"], row["compiler"]], command_ref, evidence,
                                           command=row["command"]))
    generated.append(_generated_output("archive-objects-v1", archive_ref["path"],
                                       [row["object"] for row in objects], graph_ref, evidence))
    return generated, objects, [archive]


def _finish_ep_sources(source, files, original, generated, evidence, qualified_directories=()):
    rows = _inventory(files)
    qualified, replacements = {}, {}
    for proof in generated:
        path = proof["output"]["path"]
        if not _under(path, source):
            continue
        relative = posixpath.relpath(path, source)
        _require(relative not in qualified, "ExternalProject generated output repeats a source path")
        qualified[relative] = evidence.resolver.read(proof["output"])
        if relative in original:
            _require(proof.get("original_sha256") == original[relative].get("sha256"),
                     "generated replacement omits exact original source bytes")
            replacements[relative] = proof["original_sha256"]
    for name, row in rows.items():
        if name in original:
            continue
        _require(name in qualified or (row["type"] == "directory"
                 and name in _parents(set(original) | set(qualified)) | set(qualified_directories)),
                 "ExternalProject source inventory contains an unqualified generated path: " + name)
    return qualified, replacements


def _external_project(name, layout, manifest, selected, compile_info, evidence, toolchain, version, work, arrow):
    archive_ref = (manifest["dependencies"][name]["archive"] if name == "jemalloc" else None)
    if archive_ref is None:
        candidates = [row for row in selected.get("downloaded_archives", [])
                      if posixpath.dirname(row.get("path", "")) == posixpath.dirname(layout["source"])
                      and row.get("sha256") == EP_ARCHIVE_SHA256[name]]
        _require(len(candidates) == 1, "ExternalProject source archive pin/ownership differs: " + name)
        archive_ref = _reference(candidates[0])
    original = _archive_index(archive_ref, evidence.resolver)
    captured = evidence.captured.get("external_projects", {}).get(name) if evidence.captured else None
    files = evidence.tree(layout["source"], captured.get("files") if captured else None)
    entrypoint = (_cmake_entrypoint(name, layout, original, selected, evidence)
                  if name in {"lz4", "mimalloc", "snappy", "utf8proc"} else None)
    configuration = _ep_configuration(name, layout, archive_ref, evidence, cmake_entrypoint=entrypoint)
    if name == "jemalloc":
        generated, objects, archives, _ = _jemalloc_project(layout, original, configuration, evidence, toolchain)
        build_files, directories = None, set()
    elif name in {"substrait", "orc_format"}:
        _require(configuration["configure_command"] == [], "download-only ExternalProject unexpectedly configures code")
        generated, objects, archives = _proto_project(name, layout, original, evidence, toolchain,
                                                     selected, work, arrow, compile_info)
        build_files, directories = None, set()
    else:
        generated, objects, archives, build_files, directories = _cmake_project(
            name, layout, original, configuration, evidence, toolchain, version, captured)
    if name == "jemalloc":
        # Installation copies are checked byte-for-byte against their source,
        # not accepted merely because they happen to be under dist/.
        copies = {
            "dist/bin/jemalloc-config": "bin/jemalloc-config", "dist/bin/jemalloc.sh": "bin/jemalloc.sh",
            "dist/bin/jeprof": "bin/jeprof", "dist/include/jemalloc/jemalloc.h": "include/jemalloc/jemalloc.h",
            "dist/lib/libjemalloc.a": "lib/libjemalloc.a", "dist/lib/libjemalloc_pic.a": "lib/libjemalloc_pic.a",
            "dist/lib/pkgconfig/jemalloc.pc": "jemalloc.pc",
            "dist/share/doc/jemalloc/jemalloc.html": "doc/jemalloc.html",
            "dist/share/man/man3/jemalloc.3": "doc/jemalloc.3",
        }
        for destination, source_name in copies.items():
            reference = evidence.reference(layout["source"] + "/" + destination, optional=True)
            if reference is None:
                continue
            source_ref = evidence.reference(layout["source"] + "/" + source_name)
            _require(reference["sha256"] == source_ref["sha256"], "Jemalloc installed copy bytes differ")
            generated.append(_generated_output("installed-copy-v1", reference["path"], [source_ref],
                                               configuration["configuration"], evidence))
        map_path = layout["source"] + "/a.out.map"
        if evidence.reference(map_path, optional=True):
            generated.append(_generated_output("jemalloc-configure-map-v1", map_path,
                                               [evidence.reference(layout["source"] + "/configure")],
                                               configuration["configure_command_record"] or configuration["configuration"], evidence))
    generated.sort(key=lambda row: row["output"]["path"])
    qualified, replacements = _finish_ep_sources(layout["source"], files, original, generated, evidence, directories)
    proved = _verify_archive_source(archive_ref, layout["source"], files, qualified, evidence.resolver,
                                   verified_replacements=replacements, qualified_directories=directories)
    result = {"kind": "external_project", "target": layout["target"], "archive": _reference(archive_ref),
              "source_root": layout["source"], "files": proved, **configuration,
              "generated_outputs": generated, "objects": objects, "archives": archives}
    if build_files is not None:
        result["build_root"] = layout["binary"]
        result["build_files"] = build_files
    return result


def _artifact_closure(raw, work, evidence):
    result = {"linked_static_archives": raw.get("linked_static_archives"),
              "link_maps": raw.get("native_link_artifacts"),
              "final_link_maps": raw.get("final_native_link_artifacts"),
              "native_files": raw.get("native_files")}
    _require(all(isinstance(value, list) and bool(value) for value in result.values()),
             "compiled subject static/map/native artifact closure is incomplete")
    static = {}
    archive_members = {}
    for row in result["linked_static_archives"]:
        reference = _reference(row)
        _require(reference["path"].endswith(".a") and reference["path"] not in static,
                 "compiled static artifact inventory repeats or mislabels an archive")
        static[reference["path"]] = reference
        archive_members[reference["path"]] = _ar_members(evidence.resolver.read(reference))
    maps = {_reference(row)["path"]: _reference(row) for row in result["link_maps"]}
    final = {_reference(row)["path"]: _reference(row) for row in result["final_link_maps"]}
    _require(len(maps) == len(result["link_maps"]) and len(final) == len(result["final_link_maps"])
             and all(path in maps and maps[path] == row for path, row in final.items()),
             "compiled final linker map inventory differs")
    linked, consumed = set(), []
    for path, reference in maps.items():
        text = _text(evidence.resolver.read(reference))
        _require("Linker script and memory map" in text, "captured linker map has no final memory-map record")
        if path not in final:
            continue
        stage = work + ("/python-build" if _under(path, work + "/python-build") else "/cpp-build")
        _require(_under(path, stage) and not {"CMakeScratch", "CMakeTmp", "_deps"}.intersection(
            PurePosixPath(posixpath.relpath(path, stage)).parts), "final linker map belongs to a configure/dependency probe")
        for value in re.findall(r"(?:^|\s)([^\s()]+\.a)(?=[\s()]|$)", text, re.M):
            linked.add(_join(stage, value))
        for value, member in re.findall(r"(?:^|\s)([^\s()]+\.a)\(([^\s()]+)\)", text, re.M):
            archive_path = _join(stage, value)
            _require(archive_path in static, "final linker map consumes an unrecorded static archive")
            matches = [row for row in archive_members[archive_path] if row["name"] == member]
            _require(bool(matches), "final linker map member is absent from its hash-bound static archive")
            consumed.append({"map": reference, "archive": static[archive_path], "member": member,
                             "member_sha256": sorted({row["sha256"] for row in matches})})
    _require(linked == set(static), "final map/static artifact closure differs from compiled subject")
    result["consumed_members"] = sorted(consumed, key=lambda row: (
        row["map"]["path"], row["archive"]["path"], row["member"]))
    wheel_ref = _reference(raw.get("wheel"))
    _require(raw.get("wheel_sha256") == wheel_ref["sha256"], "compiled wheel hash differs from its subject reference")
    wheel_rows = {}
    external = {}
    prefix = "{generation}/env/lib/python3.14/site-packages/"
    for row in result["native_files"]:
        _require(isinstance(row, dict) and isinstance(row.get("path"), str), "native artifact row is malformed")
        checksum = _sha256(row.get("sha256"))
        if row.get("owner") == "wheel:pyarrow":
            _require(row["path"].startswith(prefix), "wheel native artifact path differs")
            member = _relative(row["path"][len(prefix):])
            _require(member.startswith("pyarrow/") and member not in wheel_rows,
                     "wheel native artifact ownership repeats or differs")
            wheel_rows[member] = checksum
        else:
            reference = _reference(row)
            _require(reference["path"] not in external, "external native artifact repeats")
            _require(evidence.resolver.read(reference).startswith(b"\x7fELF"),
                     "external native artifact is not ELF")
            external[reference["path"]] = reference
    _require(bool(wheel_rows), "compiled wheel native inventory is empty")
    try:
        with zipfile.ZipFile(io.BytesIO(evidence.resolver.read(wheel_ref))) as bundle:
            names, elfs = set(), {}
            for item in bundle.infolist():
                name = item.filename.rstrip("/")
                _relative(name)
                _require(item.filename not in names and not stat.S_ISLNK(item.external_attr >> 16),
                         "wheel archive repeats a member or contains a symlink")
                names.add(item.filename)
                if item.is_dir():
                    continue
                with bundle.open(item) as stream:
                    data = stream.read()
                if data.startswith(b"\x7fELF"):
                    elfs[item.filename] = hashlib.sha256(data).hexdigest()
            _require(elfs == wheel_rows, "wheel ELF/native artifact inventory or bytes differ")
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError("compiled wheel artifact cannot be read") from exc
    linkage = raw.get("linkage")
    _require(isinstance(linkage, dict) and set(linkage) == set(wheel_rows),
             "wheel ELF/linkage artifact closure differs")
    for value in linkage.values():
        _require(isinstance(value, str) and "not found" not in value, "compiled ELF linkage is unresolved")
        for path in re.findall(r"=> (/\S+)", value) + re.findall(r"^\s*(/\S+) \(", value, re.M):
            normalized = _join(work, path)
            if _under(normalized, work + "/wheel-inventory"):
                member = posixpath.relpath(normalized, work + "/wheel-inventory")
                _require(member in wheel_rows, "ELF linkage selects an unrecorded wheel native")
            else:
                # ldd may print the provider's symlink spelling. The raw native
                # inventory uses canonical names; retain the captured alias
                # chain and require its canonical bytes to equal that inventory.
                if normalized in external:
                    evidence.resolver.read(external[normalized])
                else:
                    canonical = evidence._tool_file(normalized)
                    _require(canonical["path"] in external and external[canonical["path"]] == canonical,
                             "ELF linkage selects an unrecorded external native")
    stages = raw.get("build", {}).get("stages")
    _require(isinstance(stages, list) and bool(stages), "completed subject has no recorded build commands")
    for stage in stages:
        _require(isinstance(stage, dict) and stage.get("exit_code") == 0
                 and isinstance(stage.get("command"), list) and bool(stage["command"]),
                 "completed subject contains a failed or missing build command")
        evidence.resolver.read(stage.get("log"))
    result["build_commands"] = stages
    return result


def _verify_inventory_files(root, files, resolver, *, check_layout=True):
    inventory = _inventory(files)
    if check_layout and resolver.retained is None:
        _check_current_layout(root, inventory)
    for name, row in inventory.items():
        if row["type"] in {"file", "generated"}:
            resolver.read({"path": root + "/" + name, "sha256": row["sha256"]})
    return [inventory[name] for name in sorted(inventory)]


def _assemble_attestation(raw_ref, cohort_ref, resolver, *, captured=None, producer_ref=None):
    raw = resolver.json(raw_ref)
    kinds = [kind for kind, family in FAMILIES.items()
             if raw.get("source_policy") == family["source_policy"]
             and raw.get("source_revision") == family["source_revision"]]
    _require(len(kinds) == 1 and raw.get("schema_version") == 1
             and raw.get("name") == "pyarrow" and raw.get("status") == "compiled_unverified",
             "source attestation subject is not the exact completed compiled family")
    kind, family = kinds[0], FAMILIES[kinds[0]]
    subject = {"raw_receipt": _reference(raw_ref)}
    for field in ("recipe", "build_inputs", "compiled_dependencies", "compiled_sources", "wheel"):
        subject[field] = _reference(raw.get(field))
        resolver.read(subject[field])
    _require(subject["recipe"]["sha256"] == family["recipe_sha256"], "source attestation recipe bytes differ")
    if producer_ref is None:
        producer_ref = _physical_record(Path(__file__).absolute())
    producer_ref = _reference(producer_ref)
    _require(producer_ref["sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "source attestation producer bytes differ")
    resolver.read(producer_ref)
    _require(_reference(cohort_ref)["sha256"] == COHORT_SHA256, "source attestation cohort is unreviewed")
    cohort = resolver.json(cohort_ref).get("sources")
    _require(isinstance(cohort, list) and len(cohort) == len(SDK_NAMES)
             and all(isinstance(row, dict) for row in cohort)
             and {row.get("name") for row in cohort} == set(SDK_NAMES),
             "source attestation cohort metadata is incomplete")
    manifest, selected, compile_info = (resolver.json(subject[field]) for field in (
        "build_inputs", "compiled_dependencies", "compiled_sources"))
    _require(manifest.get("source_policy") == family["source_policy"]
             and manifest.get("recipe") == subject["recipe"] and selected.get("stage") == "compiled",
             "source attestation build/dependency subject differs")
    work = str(_logical_path(manifest.get("context", {}).get("work")))
    _require(raw.get("build", {}).get("context", {}).get("work") == work,
             "completed raw/manifest work context differs")
    _require(subject["raw_receipt"]["path"] == work + "/build-receipt.json",
             "completed raw receipt belongs to another build")
    arrow = work + "/arrow-" + family["source_revision"]
    evidence = _Evidence(resolver, captured)
    toolchain = raw.get("toolchain")
    _require(isinstance(toolchain, dict) and {"cc", "cxx", "cmake", "ninja", "ld"} <= set(toolchain),
             "completed subject toolchain is incomplete")
    for role, reference in toolchain.items():
        reference = _reference(reference)
        _require(role in {"cc", "cxx", "cmake", "ninja", "ld", "ar", "ranlib", "make"},
                 "completed subject has an unqualified toolchain role")
        evidence.tool(role, reference["path"], toolchain)
    _verify_inventory_files(arrow, compile_info.get("files"), resolver)
    resolver.read(manifest.get("source_archive"))
    if manifest.get("source_verification") is not None:
        resolver.read(manifest["source_verification"])
    for field in ("translation_units", "integrated_header_consumers"):
        for reference in compile_info.get(field, []):
            resolver.read(reference)
    file_api = selected.get("file_api")
    _require(isinstance(file_api, list), "compiled CMake source/target evidence is missing")
    for reference in file_api:
        resolver.read(reference)
    legacy = selected.get("forced_bundled_source_verification")
    _require(isinstance(legacy, dict) and set(legacy) == family["legacy"], "legacy source proof set differs")
    for name, files in legacy.items():
        archive = manifest.get("dependencies", {}).get(name, {}).get("archive")
        proved = _verify_archive_source(archive, work + "/cpp-build/_deps/" + name + "-src", files, {}, resolver)
        _require(proved == files, "legacy source proof is not the unchanged canonical inventory")
    trees = {row.get("path"): row for row in selected.get("source_trees", []) if isinstance(row, dict)}
    supplemental = {}
    for row in cohort:
        name = row["name"]
        expected = {key: value for key, value in row.items() if key not in {"name", "path"}}
        dependency = manifest.get("dependencies", {}).get(name)
        _require(isinstance(dependency, dict) and dependency.get("role") == "bundled_provider"
                 and all(dependency.get(key) == value for key, value in expected.items()),
                 "source component metadata differs: " + name)
        archive = _reference(dependency.get("archive"))
        _require(archive["sha256"] == row["sha256"], "source archive cohort pin differs")
        if name == "jemalloc":
            continue
        source = work + "/cpp-build/_deps/" + name.replace("_", "-") + "-src"
        _require(source in trees, "compiled source inventory omitted cohort component: " + name)
        entry = {"kind": "fetchcontent", "archive": archive, "source_root": source,
                 "files": trees[source].get("files"), "generated_derivations": []}
        if name == "aws_sdk_cpp":
            root = source + "/src/aws-cpp-sdk-core/"
            entry["generated_derivations"] = [{"kind": "aws-sdkconfig-static-v1",
                "template": evidence.reference(root + "include/aws/core/SDKConfig.h.in"),
                "root_cmake": evidence.reference(source + "/CMakeLists.txt"),
                "core_cmake": evidence.reference(root + "CMakeLists.txt"),
                "cache": evidence.reference(work + "/cpp-build/CMakeCache.txt")}]
        generated = _generated_files(name, entry, manifest, selected, resolver)
        entry["files"] = _verify_archive_source(archive, source, entry["files"], generated, resolver)
        _require(entry["files"] == trees[source]["files"], "compiled source inventory is not canonical: " + name)
        supplemental[name] = entry
    layouts = {name: _ep_layout(name, work) for name in ("jemalloc", *EP_ARCHIVE_SHA256)}
    selected_ep = {row.get("target") for row in selected.get("targets", [])
                   if isinstance(row, dict) and isinstance(row.get("target"), str) and row["target"].endswith("_ep")}
    _require(selected_ep == {row["target"] for row in layouts.values()},
             "selected ExternalProject source closure is missing or unqualified")
    version_match = re.fullmatch(r"cmake version ([0-9]+\.[0-9]+\.[0-9]+)",
                                 raw.get("toolchain_versions", {}).get("cmake", ""))
    version = version_match[1] if version_match else None
    external = {name: _external_project(name, layout, manifest, selected, compile_info,
                                       evidence, toolchain, version, work, arrow)
                for name, layout in layouts.items()}
    supplemental["jemalloc"] = external["jemalloc"]
    verify_proof_union(kind, legacy, supplemental)
    artifacts = _artifact_closure(raw, work, evidence)
    artifacts["generator_tools"] = {role: evidence.tools[role] for role in sorted(evidence.tools)}
    artifacts["generator_aliases"] = {name: evidence.aliases[name] for name in sorted(evidence.aliases)}
    result = {"schema_version": 1, "kind": "bundled_native_source_attestation", "status": "source_attested",
              "ready": False, "schema_kind": kind, "source_policy": family["source_policy"],
              "producer": producer_ref, "subject": subject, "cohort": _reference(cohort_ref),
              "legacy": legacy, "supplemental": supplemental, "external_projects": external,
              "artifact_closure": artifacts}
    result["closure"] = [resolver.originals[key] for key in sorted(resolver.originals)
                         if captured is None or key != getattr(resolver, "receipt_identity", None)]
    resolver.recheck()
    return result


def capture_completed_attestation(raw_receipt_path, expected_sha256, *, cohort_path):
    """Capture a separate, pure provenance receipt; never mutate frozen work."""
    resolver = _Resolver()
    raw_ref = {"path": str(Path(raw_receipt_path).absolute()), "sha256": _sha256(expected_sha256)}
    cohort_ref = {"path": str(Path(cohort_path).absolute()), "sha256": COHORT_SHA256}
    try:
        return _assemble_attestation(raw_ref, cohort_ref, resolver)
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise ValueError("completed source attestation inputs could not be read") from exc


def verify_completed_attestation(receipt_path, expected_sha256, *, retained_inputs=None):
    """Validate a completed supplemental receipt using a closed retained resolver.

    The reader never executes a recipe, compiler, generator, or native tool.
    Reconstruct all finite source/generated/artifact records from the exact
    subject and captured closure. Historical filesystem paths are never opened
    when retained copies are supplied.
    """
    resolver = _Resolver(retained_inputs)
    receipt = {"path": str(receipt_path), "sha256": _sha256(expected_sha256)}
    doc = resolver.json(receipt)
    _require(doc.get("schema_version") == 1
             and doc.get("kind") == "bundled_native_source_attestation"
             and doc.get("status") == "source_attested" and doc.get("ready") is False,
             "completed source attestation schema/status differs")
    subject = doc.get("subject")
    _require(isinstance(subject, dict), "source attestation subject is missing")
    raw_ref = _reference(subject.get("raw_receipt"))
    raw = resolver.json(raw_ref)
    for field in ("recipe", "build_inputs", "compiled_dependencies", "compiled_sources", "wheel"):
        _require(subject.get(field) == raw.get(field), "source attestation subject reference differs: " + field)
    resolver.receipt_identity = (receipt["path"], receipt["sha256"])
    try:
        expected = _assemble_attestation(raw_ref, doc.get("cohort"), resolver,
                                         captured=doc, producer_ref=doc.get("producer"))
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise ValueError("completed source/artifact proof could not be read") from exc
    _require(doc == expected,
             "completed source/generated rule/artifact/subject/closure records differ from their exact derivation")
    return {"receipt": receipt, "attestation": doc,
            "files": [resolver.files[key] for key in sorted(resolver.files)],
            "supplemental": doc["supplemental"], "legacy": doc["legacy"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Capture supplemental Arrow source/build provenance without executing tools")
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--build-receipt-sha256", required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        raw_ref = {"path": str(args.build_receipt.absolute()), "sha256": _sha256(args.build_receipt_sha256)}
        resolver = _Resolver()
        raw = resolver.json(raw_ref)
        manifest = resolver.json(raw.get("build_inputs"))
        work = str(_logical_path(manifest.get("context", {}).get("work")))
        _require(raw.get("build", {}).get("context", {}).get("work") == work,
                 "completed raw/manifest work context differs")
        output = args.output.absolute()
        _require(not output.exists() and not output.is_symlink(), "source attestation output already exists; preserve it")
        _require(not _under(str(output.resolve()), str(Path(work).resolve())),
                 "source attestation output must be outside the frozen build tree")
        doc = capture_completed_attestation(raw_ref["path"], raw_ref["sha256"], cohort_path=args.cohort)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with output.open("xb") as stream:
            stream.write(data)
        print(json.dumps({"path": str(output), "sha256": hashlib.sha256(data).hexdigest(),
                          "status": doc["status"], "ready": False}, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
