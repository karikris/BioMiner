#!/usr/bin/env python3
"""Verify supplemental Arrow source evidence without executing recipes or tools.

This module supplies pure source-proof primitives and the completed-attestation
reader. Capture/CLI orchestration and additional ExternalProject derivations
are qualified separately against actual build evidence.
"""
from __future__ import annotations

from collections import deque
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import tarfile


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
        "recipe_sha256": "4f7df429cc94697f155b7e252a8d0241225b1ea5800319f44bb6fb7b08da9d9b",
        "legacy": frozenset({"boost"}),
    },
    "26": {
        "source_policy": "arrow26-integrated-private-network-v3",
        "source_revision": "446167169a0dc547e00e1bc29f417a40c0ca267a",
        "recipe_sha256": "1bc69557acd83f150040f5001fd8243b6bcf98250023990e6d22e179cf6264b1",
        "legacy": frozenset({"simdjson", "uriparser", "boost"}),
    },
}


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


def _verify_archive_source(archive_record, source_root, files, generated_files, resolver):
    root = _logical_path(source_root)
    inventory = _inventory(files)
    _require(isinstance(generated_files, dict), "generated source derivations must be explicit")
    generated = {}
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
        _require(not (set(expected) & set(generated)),
                 "generated source derivation overlaps original archive members")
        allowed_directories = _parents(set(expected) | set(generated))
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


def verify_completed_attestation(receipt_path, expected_sha256, *, retained_inputs=None):
    """Validate a completed supplemental receipt using a closed retained resolver.

    The reader never executes a recipe, compiler, generator, or native tool.
    Unknown ExternalProject generated rules remain rejected until separately
    qualified; this function cannot turn incomplete source evidence into proof.
    """
    resolver = _Resolver(retained_inputs)
    receipt = {"path": str(receipt_path), "sha256": _sha256(expected_sha256)}
    doc = resolver.json(receipt)
    _require(doc.get("schema_version") == 1
             and doc.get("kind") == "bundled_native_source_attestation"
             and doc.get("status") == "source_attested" and doc.get("ready") is False,
             "completed source attestation schema/status differs")
    kind = doc.get("schema_kind")
    _require(isinstance(kind, str) and kind in FAMILIES, "completed source attestation family differs")
    family = FAMILIES[kind]
    _require(doc.get("source_policy") == family["source_policy"], "source attestation policy differs")
    subject = doc.get("subject")
    _require(isinstance(subject, dict), "source attestation subject is missing")
    raw = resolver.json(subject.get("raw_receipt"))
    _require(raw.get("schema_version") == 1 and raw.get("status") == "compiled_unverified"
             and raw.get("name") == "pyarrow" and raw.get("source_policy") == family["source_policy"]
             and raw.get("source_revision") == family["source_revision"],
             "source attestation subject is not the exact compiled family")
    for field in ("recipe", "build_inputs", "compiled_dependencies", "compiled_sources", "wheel"):
        _require(subject.get(field) == raw.get(field), "source attestation subject reference differs: " + field)
        resolver.read(subject[field])
    _require(subject["recipe"]["sha256"] == family["recipe_sha256"],
             "source attestation recipe bytes differ")
    producer = _reference(doc.get("producer"))
    _require(producer["sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "source attestation producer bytes differ")
    resolver.read(producer)
    cohort_ref = _reference(doc.get("cohort"))
    _require(cohort_ref["sha256"] == COHORT_SHA256, "source attestation cohort is unreviewed")
    cohort = resolver.json(cohort_ref).get("sources")
    _require(isinstance(cohort, list) and len(cohort) == len(SDK_NAMES)
             and all(isinstance(row, dict) for row in cohort)
             and {row.get("name") for row in cohort} == set(SDK_NAMES),
             "source attestation cohort metadata is incomplete")
    manifest = resolver.json(subject["build_inputs"])
    selected = resolver.json(subject["compiled_dependencies"])
    _require(manifest.get("source_policy") == family["source_policy"]
             and selected.get("stage") == "compiled", "source attestation build/dependency subject differs")
    legacy, supplemental = doc.get("legacy"), doc.get("supplemental")
    verify_proof_union(kind, legacy, supplemental)
    _require(legacy == selected.get("forced_bundled_source_verification"),
             "legacy source proof differs from the unchanged raw receipt")
    for row in cohort:
        name = row["name"]
        expected = {key: value for key, value in row.items() if key not in {"name", "path"}}
        dependency = manifest.get("dependencies", {}).get(name)
        _require(isinstance(dependency, dict)
                 and all(dependency.get(key) == value for key, value in expected.items())
                 and dependency.get("role") == "bundled_provider", "source component metadata differs: " + name)
        entry = supplemental[name]
        _require(isinstance(entry, dict) and entry.get("archive") == dependency.get("archive"),
                 "source component archive ownership differs: " + name)
        _require(entry["archive"]["sha256"] == row["sha256"], "source archive cohort pin differs")
        work = str(_logical_path(manifest["context"]["work"]))
        expected_root = (work + "/cpp-build/jemalloc_ep-prefix/src/jemalloc_ep" if name == "jemalloc"
                         else work + "/cpp-build/_deps/" + name.replace("_", "-") + "-src")
        _require(entry.get("source_root") == expected_root
                 and entry.get("kind") == ("external_project" if name == "jemalloc" else "fetchcontent"),
                 "source component root/kind differs: " + name)
        generated = _generated_files(name, entry, manifest, selected, resolver)
        proved = _verify_archive_source(entry["archive"], entry["source_root"], entry.get("files"),
                                       generated, resolver)
        _require(proved == entry["files"], "source proof rows are not canonical: " + name)
    external = doc.get("external_projects")
    selected_ep = {row.get("target") for row in selected.get("targets", [])
                   if isinstance(row, dict) and isinstance(row.get("target"), str)
                   and row["target"].endswith("_ep")}
    _require(selected_ep <= {"jemalloc_ep"},
             "selected ExternalProject source rules are not yet qualified")
    _require(isinstance(external, dict) and set(external) == {"jemalloc"}
             and external["jemalloc"] == supplemental["jemalloc"],
             "ExternalProject source closure is incomplete or unqualified")
    # More EP generators are intentionally not inferred from arbitrary extras.
    # Their exact records/rules must be qualified by the orchestration task.
    closure = doc.get("closure")
    _require(isinstance(closure, list), "source attestation retained closure is missing")
    references = [_reference(value) for value in closure]
    observed = {(row["path"], row["sha256"]) for row in references}
    required = set(resolver.originals) - {(receipt["path"], receipt["sha256"])}
    _require(len(observed) == len(references) and observed == required,
             "source attestation retained closure is incomplete or duplicated")
    resolver.recheck()
    return {"receipt": receipt, "attestation": doc,
            "files": [resolver.files[key] for key in sorted(resolver.files)],
            "supplemental": supplemental, "legacy": legacy}
