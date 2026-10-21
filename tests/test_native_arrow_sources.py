"""Pure Arrow source-proof contracts; synthetic archives and retained copies only."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_native_arrow_sources.py"
SDK_NAMES = (
    "aws_sdk_cpp", "aws_crt_cpp", "aws_c_auth", "aws_c_cal", "aws_c_common",
    "aws_c_compression", "aws_c_event_stream", "aws_c_http", "aws_c_io",
    "aws_c_mqtt", "aws_c_s3", "aws_c_sdkutils", "aws_checksums", "s2n_tls",
    "thrift", "grpc", "protobuf", "absl", "cares", "rapidjson", "jemalloc",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def record(path):
    return {"path": str(path), "sha256": digest(path.read_bytes())}


@pytest.fixture
def verifier(monkeypatch):
    assert SCRIPT.is_file(), "the standalone source attestation implementation is required"
    spec = importlib.util.spec_from_file_location("native_arrow_sources_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    def forbidden(*args, **kwargs):
        raise AssertionError("source verification must not execute a native tool or recipe")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    return module


@pytest.fixture
def source_case(tmp_path):
    historical = tmp_path / "historical"
    source = historical / "source"
    source.mkdir(parents=True)
    contents = {"z.c": b"int last_source;\n", "a.h": b"#define FIRST 1\n",
                "m.txt": b"upstream notice\n"}
    archive = historical / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for relative, data in contents.items():
            item = tarfile.TarInfo("upstream/" + relative)
            item.size = len(data)
            bundle.addfile(item, io.BytesIO(data))
        link = tarfile.TarInfo("upstream/alias.h")
        link.type = tarfile.SYMTYPE
        link.linkname = "a.h"
        bundle.addfile(link)
    for relative, data in contents.items():
        (source / relative).write_bytes(data)
    (source / "alias.h").symlink_to("a.h")
    files = [{"relative": name, "type": "file", "sha256": digest(data)}
             for name, data in sorted(contents.items())]
    files.append({"relative": "alias.h", "type": "symlink", "target": "a.h"})
    files.sort(key=lambda row: row["relative"])
    return {"historical": historical, "source": source, "archive": record(archive),
            "files": files, "contents": contents}


def add_file(case, relative, data):
    path = case["source"] / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    case["files"].append({"relative": relative, "type": "file", "sha256": digest(data)})
    case["files"].sort(key=lambda row: row["relative"])


def relocate(case, target):
    target.mkdir()
    references = [case["archive"]]
    references.extend({"path": str(case["source"] / row["relative"]), "sha256": row["sha256"]}
                      for row in case["files"] if row["type"] == "file")
    retained = {}
    for ref in references:
        copied = target / ref["sha256"]
        copied.write_bytes(Path(ref["path"]).read_bytes())
        retained[ref["sha256"]] = record(copied)
    shutil.rmtree(case["historical"])
    return list(retained.values())


def forbid_history(monkeypatch, historical):
    for name in ("open", "stat", "lstat", "readlink", "iterdir", "resolve"):
        original = getattr(Path, name)

        def guarded(path, *args, _original=original, **kwargs):
            if path.is_relative_to(historical):
                raise AssertionError("historical paths must never be traversed after relocation")
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(Path, name, guarded)


@pytest.mark.parametrize("schema_kind,legacy_names,total", [
    ("26", ("simdjson", "uriparser", "boost"), 24),
    ("25", ("boost",), 22),
])
def test_proof_union_requires_exact_disjoint_family_sets(verifier, schema_kind, legacy_names, total):
    legacy = {name: [{"relative": name + ".h", "type": "file", "sha256": "a" * 64}]
              for name in legacy_names}
    supplemental = {name: {"kind": "external_project" if name == "jemalloc" else "fetchcontent"}
                    for name in SDK_NAMES}
    before = json.dumps([legacy, supplemental], sort_keys=True)
    result = verifier.verify_proof_union(schema_kind, legacy, supplemental)
    assert set(result) == set(legacy_names) | set(SDK_NAMES)
    assert len(result) == total
    assert json.dumps([legacy, supplemental], sort_keys=True) == before


@pytest.mark.parametrize("schema_kind", ["25", "26"])
@pytest.mark.parametrize("change", ["missing_sdk", "extra_sdk", "overlap", "wrong_legacy"])
def test_proof_union_rejects_incomplete_or_relabeled_evidence(verifier, schema_kind, change):
    names = ("boost",) if schema_kind == "25" else ("simdjson", "uriparser", "boost")
    legacy = {name: [{"relative": name + ".h", "type": "file", "sha256": "a" * 64}]
              for name in names}
    supplemental = {name: {"kind": "fetchcontent"} for name in SDK_NAMES}
    if change == "missing_sdk":
        supplemental.pop("grpc")
    elif change == "extra_sdk":
        supplemental["unowned"] = {"kind": "fetchcontent"}
    elif change == "overlap":
        supplemental["boost"] = supplemental.pop("grpc")
    else:
        legacy["unowned"] = legacy.pop("boost")
    with pytest.raises(ValueError, match="source|proof|cohort|legacy|union|component"):
        verifier.verify_proof_union(schema_kind, legacy, supplemental)


def test_archive_proof_preserves_exact_canonical_source_inventory(verifier, source_case):
    case = source_case
    observed = verifier.verify_archive_source(case["archive"], str(case["source"]), case["files"])
    assert observed == case["files"]


def test_archive_proof_hashes_current_source_not_only_inventory(verifier, source_case):
    case = source_case
    (case["source"] / "a.h").write_bytes(b"changed despite retained inventory")
    with pytest.raises(ValueError, match="source|checksum|hash"):
        verifier.verify_archive_source(case["archive"], str(case["source"]), case["files"])


@pytest.mark.parametrize("relative", ["extra.c", "include/unqualified-generated.h", "obj/unqualified.o"])
def test_deferred_source_extras_need_explicit_byte_derivation(verifier, source_case, relative):
    case = source_case
    add_file(case, relative, b"unqualified generated source or artifact")
    with pytest.raises(ValueError, match="source|extra|generated|layout"):
        verifier.verify_archive_source(case["archive"], str(case["source"]), case["files"])


def test_one_exact_generated_file_is_verified_without_exempting_the_tree(verifier, source_case):
    case = source_case
    expected = b"/* derived from the pinned upstream template */\n"
    add_file(case, "generated.h", expected)
    assert verifier.verify_archive_source(
        case["archive"], str(case["source"]), case["files"],
        generated_files={"generated.h": expected}) == case["files"]
    add_file(case, "another.h", b"not covered by the one derivation")
    with pytest.raises(ValueError, match="source|extra|generated|layout"):
        verifier.verify_archive_source(
            case["archive"], str(case["source"]), case["files"],
            generated_files={"generated.h": expected})


def test_generated_bytes_cannot_be_changed_by_updating_their_inventory_hash(verifier, source_case):
    case = source_case
    add_file(case, "generated.h", b"unapproved generated bytes")
    with pytest.raises(ValueError, match="source|generated|deriv"):
        verifier.verify_archive_source(
            case["archive"], str(case["source"]), case["files"],
            generated_files={"generated.h": b"exact template output\n"})


def test_retained_resolver_replays_proof_without_touching_historical_paths(
        verifier, source_case, tmp_path, monkeypatch):
    case = source_case
    retained = relocate(case, tmp_path / "retained")
    forbid_history(monkeypatch, case["historical"])
    assert verifier.read_record(case["archive"], retained_inputs=retained).startswith(b"\x1f\x8b")
    assert verifier.verify_archive_source(
        case["archive"], str(case["source"]), case["files"],
        retained_inputs=retained) == case["files"]


def test_missing_retained_copy_never_falls_back_to_the_original_path(
        verifier, source_case, tmp_path, monkeypatch):
    case = source_case
    retained = relocate(case, tmp_path / "retained")
    retained = [row for row in retained if row["sha256"] != case["archive"]["sha256"]]
    forbid_history(monkeypatch, case["historical"])
    with pytest.raises(ValueError, match="retained|closure|missing"):
        verifier.read_record(case["archive"], retained_inputs=retained)


def test_retained_copy_is_rehashed_before_use(verifier, source_case, tmp_path):
    case = source_case
    retained = relocate(case, tmp_path / "retained")
    copied = next(row for row in retained if row["sha256"] == case["archive"]["sha256"])
    Path(copied["path"]).write_bytes(b"changed retained bytes")
    with pytest.raises(ValueError, match="checksum|hash|retained"):
        verifier.read_record(case["archive"], retained_inputs=retained)


AWS_TEMPLATE = b"// retained upstream notice\n\n#cmakedefine USE_AWS_MEMORY_MANAGEMENT\n"
AWS_DEFAULT = b'option(CUSTOM_MEMORY_MANAGEMENT "Use custom memory management" OFF)\n'
AWS_CORE = b"""if(CUSTOM_MEMORY_MANAGEMENT OR (BUILD_SHARED_LIBS AND NOT DEFINED CUSTOM_MEMORY_MANAGEMENT))
  set(USE_AWS_MEMORY_MANAGEMENT ON)
else()
  set(USE_AWS_MEMORY_MANAGEMENT OFF)
endif()
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/aws/core/SDKConfig.h.in"
               "${CMAKE_CURRENT_SOURCE_DIR}/include/aws/core/SDKConfig.h"
               NEWLINE_STYLE UNIX)
"""


@pytest.mark.parametrize("explicit_off", [True, False], ids=["25-explicit-off", "26-source-default-off"])
def test_aws_header_derivation_uses_actual_static_cache_and_pinned_source_default(
        verifier, explicit_off):
    cache = {"BUILD_SHARED_LIBS": "OFF"}
    if explicit_off:
        cache["CUSTOM_MEMORY_MANAGEMENT"] = "OFF"
    assert verifier.derive_aws_sdk_config(
        AWS_TEMPLATE, cache=cache, root_cmake=AWS_DEFAULT, core_cmake=AWS_CORE,
        target_kind="STATIC_LIBRARY") == (
            b"// retained upstream notice\n\n/* #undef USE_AWS_MEMORY_MANAGEMENT */\n")


@pytest.mark.parametrize("change", ["custom_on", "shared_cache", "shared_target", "changed_default",
                                    "changed_generator", "duplicate_define"])
def test_aws_derivation_rejects_unproved_branch_template_or_generator(verifier, change):
    cache = {"BUILD_SHARED_LIBS": "OFF"}
    target = "STATIC_LIBRARY"
    default, core, template = AWS_DEFAULT, AWS_CORE, AWS_TEMPLATE
    if change == "custom_on":
        cache["CUSTOM_MEMORY_MANAGEMENT"] = "ON"
    elif change == "shared_cache":
        cache["BUILD_SHARED_LIBS"] = "ON"
    elif change == "shared_target":
        target = "SHARED_LIBRARY"
    elif change == "changed_default":
        default = AWS_DEFAULT.replace(b" OFF)", b" ON)")
    elif change == "changed_generator":
        core = AWS_CORE.replace(b"NEWLINE_STYLE UNIX", b"@ONLY")
    else:
        template += b"#cmakedefine USE_AWS_MEMORY_MANAGEMENT\n"
    with pytest.raises(ValueError, match="AWS|SDK|template|memory|static|default|generator"):
        verifier.derive_aws_sdk_config(
            template, cache=cache, root_cmake=default, core_cmake=core, target_kind=target)


def test_completed_attestation_checks_expected_hash_before_interpreting_metadata(verifier, tmp_path):
    path = tmp_path / "attestation.json"
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="checksum|hash"):
        verifier.verify_completed_attestation(path, "0" * 64)


def chain_source_case(tmp_path, change):
    """Synthetic upstream links, including directory links followed by '..'."""
    historical = tmp_path / "historical-chain"
    source = historical / "source"
    (source / "dir").mkdir(parents=True)
    (source / "other").mkdir()
    (source / "other/value.h").write_bytes(b"#define RETAINED_CHAIN 1\n")
    (historical / "outside.h").write_bytes(b"outside the declared source root\n")
    links = {"dir/link": "../other", "alias.h": "dir/link/value.h", "header.h": "alias.h"}
    if change == "self_cycle":
        links["loop.h"] = "loop.h"
    elif change == "two_node_cycle":
        links["first.h"], links["second.h"] = "second.h", "first.h"
    elif change == "directory_chain_escape":
        # Lexical normalization alone maps this to root/outside.h. Following
        # dir/link first gives root/other/../../outside.h, outside the root.
        links["dir/escape.h"] = "link/../../outside.h"
    for relative, target in links.items():
        (source / relative).symlink_to(target)
    files = [
        {"relative": "dir", "type": "directory"},
        {"relative": "other", "type": "directory"},
        {"relative": "other/value.h", "type": "file",
         "sha256": digest((source / "other/value.h").read_bytes())},
        *({"relative": relative, "type": "symlink", "target": target}
          for relative, target in links.items()),
    ]
    files.sort(key=lambda row: row["relative"])
    archive = historical / "upstream.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for row in files:
            item = tarfile.TarInfo("upstream/" + row["relative"])
            if row["type"] == "directory":
                item.type = tarfile.DIRTYPE
                bundle.addfile(item)
            elif row["type"] == "symlink":
                item.type = tarfile.SYMTYPE
                item.linkname = row["target"]
                bundle.addfile(item)
            else:
                data = (source / row["relative"]).read_bytes()
                item.size = len(data)
                bundle.addfile(item, io.BytesIO(data))
    return {"historical": historical, "source": source, "archive": record(archive), "files": files}


@pytest.mark.parametrize("retained", [False, True], ids=["original-files", "relocated-forbid-history"])
@pytest.mark.parametrize("change", ["self_cycle", "two_node_cycle", "directory_chain_escape"])
def test_archive_source_rejects_cyclic_or_escaping_symlink_chains(
        verifier, tmp_path, monkeypatch, retained, change):
    case = chain_source_case(tmp_path, change)
    retained_inputs = None
    if retained:
        retained_inputs = relocate(case, tmp_path / "retained-chain")
        forbid_history(monkeypatch, case["historical"])
    with pytest.raises(ValueError, match="symlink|chain|cycle|escap"):
        verifier.verify_archive_source(
            case["archive"], str(case["source"]), case["files"], retained_inputs=retained_inputs)


@pytest.mark.parametrize("retained", [False, True], ids=["original-files", "relocated-forbid-history"])
def test_archive_source_accepts_confined_file_and_directory_symlink_chains(
        verifier, tmp_path, monkeypatch, retained):
    case = chain_source_case(tmp_path, "valid")
    retained_inputs = None
    if retained:
        retained_inputs = relocate(case, tmp_path / "retained-chain")
        forbid_history(monkeypatch, case["historical"])
    assert verifier.verify_archive_source(
        case["archive"], str(case["source"]), case["files"], retained_inputs=retained_inputs) == case["files"]


EP_NAMES = ("jemalloc", "lz4", "mimalloc", "orc_format", "snappy", "substrait", "utf8proc")


def ar_bytes(members):
    """A real regular-ar container; its ELF payloads remain synthetic bytes."""
    result = bytearray(b"!<arch>\n")
    names, offsets = bytearray(), {}
    for name, _ in members:
        if len((name + "/").encode()) > 16:
            offsets[name] = len(names)
            names.extend(name.encode() + b"/\n")
    rows = ([(b"//", bytes(names))] if names else [])
    rows += [(("/" + str(offsets[name])).encode() if name in offsets else (name + "/").encode(), data)
             for name, data in members]
    for encoded, data in rows:
        result.extend(encoded.ljust(16) + b"0".ljust(12) + b"0".ljust(6)
                      + b"0".ljust(6) + b"100644".ljust(8)
                      + str(len(data)).encode().ljust(10) + b"`\n" + data)
        if len(data) % 2:
            result.extend(b"\n")
    return bytes(result)


class CompletedSourceCase:
    """Minimal completed-build shapes, independent of live native outputs."""

    def __init__(self, root, schema_kind, verifier, monkeypatch):
        self.root = root / "historical"
        self.work = self.root / "build"
        self.cpp = self.work / "cpp-build"
        self.kind = schema_kind
        self.family = verifier.FAMILIES[schema_kind]
        self.arrow = self.work / ("arrow-" + self.family["source_revision"])
        self.paths = {}
        self.sources = {}
        self.ep = {}
        self.targets = []
        self.archives = []
        self.compile_commands = []
        self.ninja_commands = []
        self.toolchain = {
            name: self.put(self.root / "tools" / filename, b"synthetic tool: " + name.encode())
            for name, filename in (("cc", "gcc"), ("cxx", "g++"), ("cmake", "cmake"),
                                   ("ninja", "ninja"), ("ld", "ld"), ("ar", "ar"),
                                   ("ranlib", "ranlib"), ("make", "make"))
        }
        recipe = self.put(self.work / "build-recipe.py", b"# frozen synthetic recipe; never import\n")
        monkeypatch.setitem(verifier.FAMILIES, schema_kind,
                            {**self.family, "recipe_sha256": recipe["sha256"]})
        self.family = verifier.FAMILIES[schema_kind]
        self.manifest = {
            "schema_version": 1, "source_policy": self.family["source_policy"], "recipe": recipe,
            "context": {"work": str(self.work), "cc": self.toolchain["cc"]["path"],
                        "cxx": self.toolchain["cxx"]["path"]},
            "dependencies": {}, "builder": {"toolchain": self.toolchain},
        }
        if schema_kind == "25":
            for unused in ("ranlib", "make"):
                self.toolchain.pop(unused)
            builder_python = self.put(self.root / "tools/python", b"synthetic frozen builder Python\n")
            builder = {"schema_version": 1, "kind": "arrow26_python_builder", "status": "frozen_verified",
                       "python": builder_python, "toolchain": {name: self.toolchain[name] for name in (
                           "cc", "cxx", "cmake", "ninja", "ld")}}
            builder_ref = self.json(self.root / "builder-receipt.json", builder)
            self.manifest["builder"] = {**builder, "receipt": builder_ref}
            self.manifest["context"]["python"] = builder_python["path"]
        self.cache = self.put(self.cpp / "CMakeCache.txt", b"BUILD_SHARED_LIBS:BOOL=OFF\n" + (
            b"CUSTOM_MEMORY_MANAGEMENT:BOOL=OFF\n" if schema_kind == "25" else b""))
        self.cohort_rows = []
        for index, name in enumerate(SDK_NAMES):
            source = (self.cpp / "jemalloc_ep-prefix/src/jemalloc_ep" if name == "jemalloc"
                      else self.cpp / "_deps" / (name.replace("_", "-") + "-src"))
            contents = {"CMakeLists.txt": b"# synthetic upstream source\n",
                        "source.c": ("int " + name + ";\n").encode()}
            if name == "aws_sdk_cpp":
                contents = {"CMakeLists.txt": AWS_DEFAULT,
                            "src/aws-cpp-sdk-core/CMakeLists.txt": AWS_CORE,
                            "src/aws-cpp-sdk-core/include/aws/core/SDKConfig.h.in": AWS_TEMPLATE}
            elif name == "jemalloc":
                contents = self.jemalloc_originals()
            archive = self.bundle(self.work / "inputs" / (name + ".tar.gz"), source, contents)
            row = {"name": name, "variable": name.upper(), "version": "1.0",
                   "build_version": "1.0", "commit": f"{index + 1:040x}",
                   "sha256": archive["sha256"], "filename": name + ".tar.gz",
                   "path": str(self.root / "acquisition" / (name + ".tar.gz")),
                   "url": "https://example.invalid/" + name + ".tar.gz",
                   "env": "ARROW_" + name.upper() + "_URL", "old_version": "0.9",
                   "old_sha256": digest(("old " + name).encode())}
            self.cohort_rows.append(row)
            self.manifest["dependencies"][name] = {
                **{key: value for key, value in row.items() if key not in {"name", "path"}},
                "archive": archive, "role": "bundled_provider"}
            self.sources[name] = source
        self.put(self.sources["aws_sdk_cpp"] /
                 "src/aws-cpp-sdk-core/include/aws/core/SDKConfig.h",
                 AWS_TEMPLATE.replace(b"#cmakedefine USE_AWS_MEMORY_MANAGEMENT",
                                      b"/* #undef USE_AWS_MEMORY_MANAGEMENT */"))
        self.targets.append({"target": "aws-cpp-sdk-core", "kind": "STATIC_LIBRARY",
                             "source": str(self.sources["aws_sdk_cpp"] / "src/aws-cpp-sdk-core"),
                             "artifacts": [], "links": []})
        self.legacy = {}
        for name in sorted(self.family["legacy"]):
            source = self.cpp / "_deps" / (name + "-src")
            archive = self.bundle(self.work / "inputs" / (name + ".tar.gz"), source,
                                  {"original.h": ("// " + name + " original\n").encode()})
            self.manifest["dependencies"][name] = {"archive": archive, "filename": name + ".tar.gz"}
            self.legacy[name] = self.inventory(source)
        self.cohort = self.json(self.root / "cohort.json", {"sources": self.cohort_rows})
        monkeypatch.setattr(verifier, "COHORT_SHA256", self.cohort["sha256"])
        self.make_external_projects()
        monkeypatch.setattr(verifier, "EP_ARCHIVE_SHA256", {
            name: value["archive"]["sha256"] for name, value in self.ep.items() if name != "jemalloc"},
                            raising=False)
        self.make_arrow_inputs()
        if schema_kind == "25":
            add_arrow25_header_only_xsimd(self, verifier, monkeypatch)
        self.make_final_artifacts()
        self.refresh()

    def put(self, path, data):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.paths[str(path)] = path
        return record(path)

    def json(self, path, value):
        return self.put(path, (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())

    def bundle(self, path, source, contents):
        path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "w:gz") as bundle:
            for relative, data in contents.items():
                item = tarfile.TarInfo("upstream/" + relative)
                item.size = len(data)
                bundle.addfile(item, io.BytesIO(data))
                self.put(source / relative, data)
        self.paths[str(path)] = path
        return record(path)

    @staticmethod
    def inventory(root):
        return [{"relative": path.relative_to(root).as_posix(), "type": "file",
                 "sha256": record(path)["sha256"]}
                for path in sorted(root.rglob("*")) if path.is_file() and not path.is_symlink()]

    @staticmethod
    def jemalloc_originals():
        return {
            "configure": b"#!/bin/sh\n# pinned Autoconf generator\n",
            "Makefile.in": b"C_SRCS := $(srcroot)src/jemalloc.c\n"
                           b"cfghdrs_out := include/jemalloc/internal/jemalloc_internal_defs.h\n"
                           b"cfgoutputs_out := Makefile\n",
            "Makefile": b"# generated upstream release Makefile, replaced by configure\n",
            "config.status": b"# upstream release configuration, replaced by configure\n",
            "include/jemalloc/internal/jemalloc_internal_defs.h.in": b"#undef JEMALLOC_PREFIX\n",
            "include/jemalloc/internal/jemalloc_internal_defs.h": b"#define JEMALLOC_PREFIX je_\n",
            "src/jemalloc.c": b"int jemalloc_source;\n",
        }

    def make_external_projects(self):
        for name in EP_NAMES:
            target = "orc-format_ep" if name == "orc_format" else name + "_ep"
            base = self.cpp / "_deps/orc-build" if name in {"lz4", "orc_format"} else self.cpp
            source = base / (target + "-prefix/src/" + target)
            binary = source if name in {"jemalloc", "snappy"} else source.with_name(target + "-build")
            stamp = source.with_name(target + "-stamp")
            if name == "jemalloc":
                archive = self.manifest["dependencies"][name]["archive"]
                static = source / "dist/lib/libjemalloc_pic.a"
            else:
                contents = ({"src/main/proto/orc/proto/orc_proto.proto": b'syntax = "proto3";\n'}
                            if name == "orc_format" else
                            {"proto/substrait/type.proto": b'syntax = "proto3";\n'}
                            if name == "substrait" else
                            {"CMakeLists.txt": b"cmake_minimum_required(VERSION 3.16)\n"
                                               b"configure_file(config.h.in config.h)\n",
                             "config.h.in": b"#define BUILD_STATIC 1\n",
                             "source.c": ("int " + name + "_source;\n").encode()})
                archive = self.bundle(base / (target + "-prefix/src/" + name + ".tar.gz"),
                                      source, contents)
                self.archives.append({**archive, "matches_arrow_versions_pin": name != "orc_format"})
                install = ({"mimalloc": self.cpp / "mimalloc_ep/src/mimalloc_ep",
                            "snappy": self.cpp / "snappy_ep/src/snappy_ep-install",
                            "utf8proc": self.cpp / "utf8proc_ep-install",
                            "lz4": self.cpp / "_deps/orc-build/c++/libs/thirdparty/lz4_ep-install"}
                           .get(name))
                static = install / ("lib/lib" + name + ".a") if install else None
            self.targets.append({"target": target, "kind": "UTILITY", "source": str(self.arrow / "cpp"),
                                 "artifacts": [], "links": []})
            urlinfo = self.put(stamp / (target + "-urlinfo.txt"),
                               ("source_dir=" + str(source) + "\nhash=SHA256=" + archive["sha256"]
                                + "\nurl(s)=https://example.invalid/" + name + ".tar.gz\n").encode())
            cfg = ("./configure;AR=" + self.toolchain["ar"]["path"]
                   + ";CC=" + self.toolchain["cc"]["path"]
                   + ";--prefix=" + str(source / "dist") + ";--libdir=" + str(source / "dist/lib")
                   + ";--with-jemalloc-prefix=je_arrow_;--with-private-namespace=je_arrow_private_"
                   + ";--without-export;--disable-shared;--disable-cxx;--disable-libdl;--disable-initial-exec-tls"
                   if name == "jemalloc" else "" if static is None else
                   self.toolchain["cmake"]["path"] + ";-DBUILD_SHARED_LIBS=OFF;-DCMAKE_C_COMPILER="
                   + self.toolchain["cc"]["path"] + ";-DCMAKE_C_FLAGS=-fPIC;-GNinja")
            cfg_ref = self.put(base / (target + "-prefix/tmp/" + target + "-cfgcmd.txt"),
                               ("cmd='" + cfg + "'\n").encode())
            command_ref = self.put(stamp / (target + "-configure-Release.cmake"),
                                   ('set(command "' + cfg + '")\n').encode())
            self.ep[name] = {"target": target, "source": source, "binary": binary, "archive": archive,
                             "cfg": cfg_ref, "command_record": command_ref, "urlinfo": urlinfo,
                             "static": static, "generated": []}
            if static is None:
                self.make_proto_project(name)
                continue
            generated = (source / "include/jemalloc/internal/jemalloc_internal_defs.h" if name == "jemalloc"
                         else binary / "config.h")
            self.put(generated, b'#define JEMALLOC_PREFIX "je_arrow_"\n' if name == "jemalloc"
                     else b"#define BUILD_STATIC 1\n")
            self.ep[name]["generated"].append(generated)
            obj = binary / ("src/jemalloc.pic.o" if name == "jemalloc" else
                            "CMakeFiles/provider.dir/snappy-stubs-internal.cc.o" if name == "snappy"
                            else "CMakeFiles/provider.dir/source.c.o")
            source_file = source / ("src/jemalloc.c" if name == "jemalloc" else "source.c")
            obj_bytes = b"\x7fELF\x02\x01\x01\x00" + name.encode() + b" synthetic object\n"
            self.put(obj, obj_bytes)
            self.put(static, ar_bytes([(obj.name, obj_bytes)]))
            self.put(source / "lib/libjemalloc_pic.a" if name == "jemalloc" else binary / ("lib" + name + ".a"),
                     ar_bytes([(obj.name, obj_bytes)]))
            self.ep[name].update(object=obj, input=source_file, member=obj.name)
            self.targets.append({"target": name + "::" + name, "kind": "STATIC_LIBRARY",
                                 "source": str(self.arrow / "cpp"), "artifacts": [str(static)], "links": []})
            if name == "jemalloc":
                self.put(source / "Makefile", b"C_SRCS := $(srcroot)src/jemalloc.c\n"
                         b"cfghdrs_out := include/jemalloc/internal/jemalloc_internal_defs.h\n"
                         b"cfgoutputs_out := Makefile\n")
                self.put(source / "config.status", b"#!/bin/sh\n# captured configured generator\n")
                log = (self.toolchain["cc"]["path"] + " -fPIC -DPIC -c -o src/jemalloc.pic.o src/jemalloc.c\n"
                       + self.toolchain["ar"]["path"]
                       + " crus lib/libjemalloc_pic.a src/jemalloc.pic.o\n")
                self.put(stamp / (target + "-build-out.log"), log.encode())
            else:
                graph = ("build " + obj.relative_to(binary).as_posix()
                         + ": C_COMPILER__provider " + str(source_file) + "\n  FLAGS = -fPIC\n"
                         + "build lib" + name + ".a: C_STATIC_LIBRARY_LINKER__provider "
                         + obj.relative_to(binary).as_posix() + "\n")
                self.put(binary / "build.ninja", graph.encode())
                self.put(binary / "CMakeFiles/rules.ninja", (
                    "rule C_COMPILER__provider\n  command = " + self.toolchain["cc"]["path"]
                    + " $FLAGS -o $out -c $in\nrule C_STATIC_LIBRARY_LINKER__provider\n  command = "
                    + self.toolchain["ar"]["path"] + " qc $out $in\n").encode())
                self.put(binary / "CMakeCache.txt", ("BUILD_SHARED_LIBS:BOOL=OFF\nCMAKE_C_COMPILER:FILEPATH="
                         + self.toolchain["cc"]["path"] + "\n").encode())
                self.put(binary / "install_manifest.txt", (str(static) + "\n").encode())

    def make_proto_project(self, name):
        source = self.ep[name]["source"]
        if name == "substrait":
            proto = source / "proto/substrait/type.proto"
            output = self.cpp / "substrait_ep-generated/substrait/type.pb.cc"
        else:
            proto = source / "src/main/proto/orc/proto/orc_proto.proto"
            output = self.cpp / "_deps/orc-build/c++/src/orc_proto.pb.cc"
        header = output.with_suffix(".h")
        self.put(output, b"// captured generated protobuf implementation\n")
        self.put(header, b"// captured generated protobuf header\n")
        protoc = self.put(self.cpp / "_deps/protobuf-build/protoc", b"\x7fELF synthetic protoc35.1\n")
        if not any(row["target"] == "protoc" for row in self.targets):
            self.targets.append({"target": "protoc", "kind": "EXECUTABLE", "source": str(self.sources["protobuf"]),
                                 "artifacts": [protoc["path"]], "links": []})
        obj = self.cpp / ("CMakeFiles/" + name + ".dir/generated.o")
        obj_bytes = b"\x7fELF synthetic generated object " + name.encode()
        self.put(obj, obj_bytes)
        static = self.cpp / ("release/libsubstrait.a" if name == "substrait" else "_deps/orc-build/c++/src/liborc.a")
        self.put(static, ar_bytes([(obj.name, obj_bytes)]))
        self.ep[name].update(input=proto, generated=[output, header], object=obj, static=static,
                             member=obj.name, protoc=protoc)
        proto_root = source / "proto" if name == "substrait" else proto.parent
        cpp_root = self.cpp / "substrait_ep-generated" if name == "substrait" else output.parent
        command = protoc["path"] + " -I" + str(proto_root) + " --cpp_out=" + str(cpp_root) + " " + str(proto)
        self.ninja_commands.append("build " + output.relative_to(self.cpp).as_posix() + " "
                                   + header.relative_to(self.cpp).as_posix() + ": CUSTOM_COMMAND "
                                   + str(proto) + "\n  COMMAND = " + command + "\n")
        self.compile_commands.append({"directory": str(self.cpp), "file": str(output),
                                      "command": self.toolchain["cxx"]["path"] + " -fPIC -c "
                                      + str(output) + " -o " + str(obj), "output": str(obj)})
        self.targets.append({"target": "substrait" if name == "substrait" else "orc",
                             "kind": "STATIC_LIBRARY", "source": str(self.arrow / "cpp"),
                             "artifacts": [str(static)], "links": []})

    def make_arrow_inputs(self):
        cmake = []
        for name in EP_NAMES:
            project = self.ep[name]
            target = project["target"]
            cmake.append("ExternalProject_Add(" + target + " URL https://example.invalid/" + name
                         + ".tar.gz URL_HASH SHA256=" + project["archive"]["sha256"] + ")\n")
        self.put(self.arrow / "cpp/cmake_modules/ThirdpartyToolchain.cmake", "".join(cmake).encode())
        self.put(self.arrow / "cpp/thirdparty/versions.txt", "".join(
            name.upper() + "_BUILD_SHA256_CHECKSUM=" + project["archive"]["sha256"] + "\n"
            for name, project in self.ep.items()).encode())
        self.main_graph = self.put(self.cpp / "build.ninja", "".join(self.ninja_commands).encode())
        self.main_compile = self.json(self.cpp / "compile_commands.json", self.compile_commands)
        self.patched_consumers, self.patched_units = [], []
        if self.kind == "25":
            for name in ("scalar_if_else.cc", "vector_replace.cc"):
                self.patched_consumers.append(self.put(
                    self.arrow / "cpp/src/arrow/compute/kernels" / name,
                    ("// original ordered-patch header consumer " + name + "\n").encode()))
            self.patched_units.append(self.put(
                self.cpp / "src/arrow/CMakeFiles/arrow_compute_objlib.dir/Unity/unity_0_cxx.cxx",
                "".join('#include "' + row["path"] + '"\n' for row in self.patched_consumers).encode()))
        self.manifest["source_archive"] = self.bundle(self.root / "arrow-original.tar.gz", self.arrow,
                                                     {"NOTICE.txt": b"synthetic Arrow upstream notice\n"})

    def make_final_artifacts(self):
        self.native = self.put(self.work / "wheel-inventory/pyarrow/libarrow.so",
                               b"\x7fELF synthetic final Arrow library\n")
        self.system_native = self.put(self.root / "system/libc.so", b"\x7fELF synthetic system dependency\n")
        self.wheel_path = self.work / "wheelhouse/pyarrow-1.0-cp314-cp314-linux_x86_64.whl"
        self.wheel_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.wheel_path, "w") as wheel:
            wheel.writestr("pyarrow/libarrow.so", Path(self.native["path"]).read_bytes())
            wheel.writestr("pyarrow-1.0.dist-info/METADATA", b"Name: pyarrow\nVersion: 1.0\n")
            wheel.writestr("pyarrow-1.0.dist-info/WHEEL", b"Tag: cp314-cp314-linux_x86_64\n")
        self.paths[str(self.wheel_path)] = self.wheel_path
        map_text = "Archive member included to satisfy reference by file (symbol)\n"
        for project in self.ep.values():
            map_text += str(project["static"]) + "(" + project["member"] + ")\n"
        map_text += "Linker script and memory map\n"
        map_text += "".join("LOAD " + str(project["static"]) + "\n" for project in self.ep.values())
        self.map_ref = self.put(self.cpp / "release/libarrow.so.map", map_text.encode())
        self.build_log = self.put(self.work / "logs/cpp-build-1.log", b"[1/1] Linking completed\n")

    def refresh(self):
        self.selected = {"schema_version": 1, "stage": "compiled", "targets": self.targets,
                         "source_trees": [{"path": str(source), "files": self.inventory(source)}
                                          for name, source in self.sources.items() if name != "jemalloc"],
                         "forced_bundled_source_verification": self.legacy,
                         "static_archives": [record(project["static"]) for project in self.ep.values()],
                         "downloaded_archives": self.archives, "native_inputs": [], "file_api": []}
        self.selected_ref = self.json(self.work / "compiled-dependencies.json", self.selected)
        compilation = {"compile_commands": self.main_compile,
                       "translation_units": self.patched_units if self.kind == "25" else [],
                       ("patched_header_consumers" if self.kind == "25" else "integrated_header_consumers"):
                           self.patched_consumers if self.kind == "25" else []}
        self.source_ref = self.json(self.work / "compiled-sources.json", {
            "files": self.inventory(self.arrow), **compilation})
        if self.kind == "25":
            self.manifest["commands"] = {
                "cpp_configure": [self.toolchain["cmake"]["path"], "-S", str(self.arrow / "cpp"),
                                  "-B", str(self.cpp)],
                "cpp_build": [self.toolchain["cmake"]["path"], "--build", str(self.cpp),
                              "--target", "install", "--parallel", "4"],
                "python_build": [self.manifest["builder"]["python"]["path"], "-I", "-B", "-m", "build", "--wheel"],
            }
        manifest_ref = self.json(self.work / "build-inputs.json", self.manifest)
        self.raw = {"schema_version": 1, "name": "pyarrow", "status": "compiled_unverified",
                    "source_policy": self.family["source_policy"], "source_revision": self.family["source_revision"],
                    "recipe": self.manifest["recipe"], "build_inputs": manifest_ref,
                    "compiled_dependencies": self.selected_ref, "compiled_sources": self.source_ref,
                    "wheel": record(self.wheel_path), "wheel_sha256": record(self.wheel_path)["sha256"],
                    "toolchain": self.toolchain, "native_link_artifacts": [self.map_ref],
                    "final_native_link_artifacts": [self.map_ref],
                    "linked_static_archives": [record(project["static"]) for project in self.ep.values()],
                    "native_files": [{"path": "{generation}/env/lib/python3.14/site-packages/pyarrow/libarrow.so",
                                      "sha256": self.native["sha256"], "owner": "wheel:pyarrow"},
                                     {**self.system_native, "owner": "synthetic:system"}],
                    "linkage": {"pyarrow/libarrow.so": "libc.so => " + self.system_native["path"] + " (0x1)\n"},
                    "build": {"context": self.manifest["context"], "toolchain": self.toolchain,
                              "stages": [{"command": [self.toolchain["cmake"]["path"], "--build", str(self.cpp),
                                                       "--target", "install", "--parallel", "4"],
                                          "log": self.build_log, "exit_code": 0}]}}
        if self.kind == "25":
            self.raw.pop("build")
            self.raw.pop("toolchain")
            self.raw.update({"kind": "corrective_arrow25_private_network_build", "ready": False,
                             "context": dict(self.manifest["context"]),
                             "builder": json.loads(json.dumps(self.manifest["builder"])),
                             "commands": json.loads(json.dumps(self.manifest["commands"])),
                             "compile_source_evidence": compilation,
                             "stages": [{"command": self.manifest["commands"][name], "exit_code": 0,
                                         "log": self.build_log if name == "cpp_build" else self.put(
                                             self.work / "logs/python-build-1.log", b"wheel build completed\n")}
                                        for name in ("cpp_build", "python_build")]})
        self.raw_ref = self.json(self.work / "build-receipt.json", self.raw)

    def capture(self, verifier):
        return verifier.capture_completed_attestation(self.raw_ref["path"], self.raw_ref["sha256"],
                                                      cohort_path=self.cohort["path"])

    def publish(self, verifier, path):
        return self.json(path, self.capture(verifier))


@pytest.fixture(params=["26", "25"], ids=["arrow26", "arrow25"])
def completed_source_case(tmp_path, request, verifier, monkeypatch):
    return CompletedSourceCase(tmp_path, request.param, verifier, monkeypatch)


def test_completed_capture_binds_exact_subject_cohort_and_all_selected_external_projects(
        verifier, completed_source_case, tmp_path):
    case = completed_source_case
    original_raw = Path(case.raw_ref["path"]).read_bytes()
    attestation = case.publish(verifier, tmp_path / "attestation.json")
    result = verifier.verify_completed_attestation(attestation["path"], attestation["sha256"])
    doc = result["attestation"]
    assert doc["kind"] == "bundled_native_source_attestation"
    assert doc["status"] == "source_attested" and doc["ready"] is False
    assert doc["schema_kind"] == case.kind and doc["source_policy"] == case.family["source_policy"]
    assert doc["subject"] == {"raw_receipt": case.raw_ref,
                              **{field: case.raw[field] for field in (
                                  "recipe", "build_inputs", "compiled_dependencies", "compiled_sources", "wheel")}}
    assert doc["cohort"] == case.cohort
    assert doc["legacy"] == case.legacy
    assert set(doc["supplemental"]) == set(SDK_NAMES)
    assert len(set(doc["legacy"]) | set(doc["supplemental"])) == (24 if case.kind == "26" else 22)
    assert set(doc["external_projects"]) == set(EP_NAMES) | ({"xsimd"} if case.kind == "25" else set())
    assert {row["archive"]["sha256"] for row in doc["external_projects"].values()} == {
        row["archive"]["sha256"] for row in case.ep.values()} | (
            {case.xsimd["archive"]["sha256"]} if case.kind == "25" else set())
    assert Path(case.raw_ref["path"]).read_bytes() == original_raw


def test_completed_capture_closes_static_maps_native_wheel_and_generated_records(
        verifier, completed_source_case, tmp_path):
    case = completed_source_case
    attestation = case.publish(verifier, tmp_path / "attestation.json")
    result = verifier.verify_completed_attestation(attestation["path"], attestation["sha256"])
    refs = {(row["path"], row["sha256"]) for row in result["files"]}
    required = [case.raw_ref, case.cohort, case.map_ref, case.system_native,
                case.main_graph, case.main_compile, case.raw["wheel"], *case.toolchain.values()]
    required += [record(project["static"]) for project in case.ep.values()]
    required += [record(project["object"]) for project in case.ep.values()]
    required += [record(path) for project in case.ep.values() for path in project["generated"]]
    required += [project["cfg"] for project in case.ep.values()]
    assert all((row["path"], row["sha256"]) in refs for row in required)
    artifact = result["attestation"]["artifact_closure"]
    assert artifact["linked_static_archives"] == case.raw["linked_static_archives"]
    assert artifact["link_maps"] == case.raw["native_link_artifacts"]
    assert artifact["final_link_maps"] == case.raw["final_native_link_artifacts"]
    assert artifact["native_files"] == case.raw["native_files"]
    for project in result["attestation"]["external_projects"].values():
        assert project["generated_outputs"]
        for generated in project["generated_outputs"]:
            assert generated["inputs"] and generated["command_record"] and generated["output"]


@pytest.mark.parametrize("field", ["recipe", "build_inputs", "compiled_dependencies", "compiled_sources", "wheel"])
def test_completed_reader_rejects_rebound_subject_reference(
        verifier, completed_source_case, tmp_path, field):
    case = completed_source_case
    doc = case.capture(verifier)
    doc["subject"][field] = case.put(case.root / ("foreign-" + field), b"foreign subject\n")
    attestation = case.json(tmp_path / "rebound.json", doc)
    with pytest.raises(ValueError, match="subject|reference|recipe|wheel"):
        verifier.verify_completed_attestation(attestation["path"], attestation["sha256"])


@pytest.mark.parametrize("artifact", ["wheel", "static", "map", "native", "object", "generated", "generator"])
def test_completed_reader_rehashes_every_final_and_generated_artifact(
        verifier, completed_source_case, tmp_path, artifact):
    case = completed_source_case
    attestation = case.publish(verifier, tmp_path / "attestation.json")
    project = case.ep["jemalloc"]
    target = {"wheel": case.wheel_path, "static": project["static"],
              "map": Path(case.map_ref["path"]), "native": Path(case.system_native["path"]),
              "object": project["object"], "generated": project["generated"][0],
              "generator": project["source"] / "configure"}[artifact]
    target.write_bytes(target.read_bytes() + b"changed after capture\n")
    with pytest.raises(ValueError, match="checksum|hash|source|archive|generated|wheel|artifact"):
        verifier.verify_completed_attestation(attestation["path"], attestation["sha256"])


@pytest.mark.parametrize("change", ["unknown_generated_header", "wrong_compiler", "missing_pic",
                                    "source_object_mismatch", "extra_archive_member", "unselected_ep"])
def test_completed_capture_rejects_unowned_ep_source_or_object_provenance(
        verifier, completed_source_case, change):
    case = completed_source_case
    project = case.ep["utf8proc"]
    graph = project["binary"] / "build.ninja"
    if change == "unknown_generated_header":
        case.put(case.ep["jemalloc"]["source"] / "include/unknown.h", b"unowned extra header\n")
    elif change == "wrong_compiler":
        rules = project["binary"] / "CMakeFiles/rules.ninja"
        rules.write_text(rules.read_text().replace(case.toolchain["cc"]["path"], "/foreign/gcc"))
    elif change == "missing_pic":
        graph.write_text(graph.read_text().replace("FLAGS = -fPIC", "FLAGS = -O2"))
    elif change == "source_object_mismatch":
        graph.write_text(graph.read_text().replace(str(project["input"]), str(case.sources["grpc"] / "source.c")))
    elif change == "extra_archive_member":
        project["static"].write_bytes(ar_bytes([
            (project["member"], project["object"].read_bytes()), ("foreign.o", b"\x7fELF unselected object")]))
    else:
        case.targets.append({"target": "unreviewed_ep", "kind": "UTILITY", "source": str(case.arrow / "cpp"),
                             "artifacts": [], "links": []})
    case.refresh()
    with pytest.raises(ValueError, match="source|generated|provenance|compiler|PIC|object|archive|ExternalProject|unqualified"):
        case.capture(verifier)


def test_completed_reader_rejects_caller_labeled_generated_tree_exemption(
        verifier, completed_source_case, tmp_path):
    case = completed_source_case
    doc = case.capture(verifier)
    project = doc["external_projects"]["jemalloc"]
    project["generated_outputs"][0]["rule"] = "trust-all-current-generated-files"
    attestation = case.json(tmp_path / "unowned-generated-rule.json", doc)
    with pytest.raises(ValueError, match="rule|generated|derivation|qualified"):
        verifier.verify_completed_attestation(attestation["path"], attestation["sha256"])


@pytest.mark.parametrize("name", ["aws_c_http", "grpc", "rapidjson"])
def test_completed_capture_compares_every_sdk_to_its_owned_archive(
        verifier, completed_source_case, name):
    case = completed_source_case
    source = case.sources[name] / "source.c"
    source.write_bytes(source.read_bytes() + b"unreviewed source change\n")
    case.refresh()
    with pytest.raises(ValueError, match="source|archive|checksum|bytes"):
        case.capture(verifier)


def test_completed_reader_rejects_incomplete_static_map_closure(
        verifier, completed_source_case, tmp_path):
    case = completed_source_case
    doc = case.capture(verifier)
    doc["artifact_closure"]["linked_static_archives"].pop()
    attestation = case.json(tmp_path / "incomplete-artifacts.json", doc)
    with pytest.raises(ValueError, match="artifact|static|closure|archive"):
        verifier.verify_completed_attestation(attestation["path"], attestation["sha256"])


def test_completed_proof_replays_only_retained_copies_after_originals_are_removed(
        verifier, completed_source_case, tmp_path, monkeypatch):
    case = completed_source_case
    attestation = case.publish(verifier, tmp_path / "attestation.json")
    initial = verifier.verify_completed_attestation(attestation["path"], attestation["sha256"])
    retained_root = tmp_path / "retained"
    retained_root.mkdir()
    retained = {}
    for ref in initial["files"]:
        copied = retained_root / ref["sha256"]
        copied.write_bytes(Path(ref["path"]).read_bytes())
        retained[ref["sha256"]] = record(copied)
    shutil.rmtree(case.root)
    attestation_path = Path(attestation["path"])
    attestation_path.unlink()
    forbid_history(monkeypatch, case.root)
    result = verifier.verify_completed_attestation(attestation["path"], attestation["sha256"],
                                                  retained_inputs=list(retained.values()))
    assert result["attestation"] == initial["attestation"]
    assert result["supplemental"] == initial["supplemental"]
    assert result["legacy"] == initial["legacy"]
    assert all(Path(ref["path"]).is_relative_to(retained_root) for ref in result["files"])


def test_completed_capture_cli_publishes_once_and_never_changes_the_compiled_receipt(
        verifier, completed_source_case, tmp_path):
    case = completed_source_case
    before = Path(case.raw_ref["path"]).read_bytes()
    output = tmp_path / "published-attestation.json"
    argv = ["--build-receipt", case.raw_ref["path"], "--build-receipt-sha256", case.raw_ref["sha256"],
            "--cohort", case.cohort["path"], "--output", str(output)]
    assert verifier.main(argv) == 0
    first = output.read_bytes()
    assert verifier.verify_completed_attestation(str(output), digest(first))["attestation"]["ready"] is False
    with pytest.raises((ValueError, SystemExit), match="exist|preserv|overwrite|2"):
        verifier.main(argv)
    assert output.read_bytes() == first
    assert Path(case.raw_ref["path"]).read_bytes() == before


def test_completed_capture_cli_rejects_an_output_inside_the_frozen_build(
        verifier, completed_source_case):
    case = completed_source_case
    output = case.work / "new-attestation.json"
    argv = ["--build-receipt", case.raw_ref["path"], "--build-receipt-sha256", case.raw_ref["sha256"],
            "--cohort", case.cohort["path"], "--output", str(output)]
    with pytest.raises((ValueError, SystemExit), match="frozen|build|outside|2"):
        verifier.main(argv)
    assert not output.exists()


def test_completed_capture_rejects_inconsistent_work_context(
        verifier, completed_source_case):
    case = completed_source_case
    context_owner = case.raw if case.kind == "25" else case.raw["build"]
    context_owner["context"] = {**case.manifest["context"], "work": str(case.root / "other-build")}
    case.raw_ref = case.json(case.work / "build-receipt.json", case.raw)
    with pytest.raises(ValueError, match="context|work|build"):
        case.capture(verifier)


def test_completed_cli_never_publishes_in_manifest_work_with_inconsistent_work_context(
        verifier, completed_source_case):
    case = completed_source_case
    context_owner = case.raw if case.kind == "25" else case.raw["build"]
    context_owner["context"] = {**case.manifest["context"], "work": str(case.root / "other-build")}
    case.raw_ref = case.json(case.work / "build-receipt.json", case.raw)
    output = case.work / "new-attestation.json"
    argv = ["--build-receipt", case.raw_ref["path"], "--build-receipt-sha256", case.raw_ref["sha256"],
            "--cohort", case.cohort["path"], "--output", str(output)]
    with pytest.raises((ValueError, SystemExit), match="context|work|build|frozen|outside|2"):
        verifier.main(argv)
    assert not output.exists()


def use_actual_cmake_ep_configuration(case, name, verifier, monkeypatch):
    """Keep cfgcmd placeholders distinct from CMake's expanded stamp command."""
    project = case.ep[name]
    source_subdir = ""
    owner = None
    if name == "lz4":
        source_subdir = "/build/cmake"
        source = project["source"]
        cmake_bytes = (source / "CMakeLists.txt").read_bytes()
        template_bytes = (source / "config.h.in").read_bytes()
        (source / "CMakeLists.txt").unlink()
        (source / "config.h.in").unlink()
        old = project["archive"]
        archive = case.bundle(Path(old["path"]), source, {
            "build/cmake/CMakeLists.txt": cmake_bytes,
            "build/cmake/config.h.in": template_bytes,
            "source.c": (source / "source.c").read_bytes()})
        project["archive"] = archive
        download = Path(project["urlinfo"]["path"])
        project["urlinfo"] = case.put(download, download.read_bytes().replace(
            old["sha256"].encode(), archive["sha256"].encode()))
        case.archives = [{**row, "sha256": archive["sha256"]} if row["path"] == archive["path"] else row
                         for row in case.archives]
        monkeypatch.setitem(verifier.EP_ARCHIVE_SHA256, "lz4", archive["sha256"])
        for relative in ("cpp/cmake_modules/ThirdpartyToolchain.cmake", "cpp/thirdparty/versions.txt"):
            path = case.arrow / relative
            path.write_text(path.read_text().replace(old["sha256"], archive["sha256"]))
        owner = case.cpp / "_deps/orc-src/cmake_modules/ThirdpartyToolchain.cmake"
        case.put(owner, b'set(LZ4_CONFIGURE SOURCE_SUBDIR "build/cmake" CMAKE_ARGS ${LZ4_CMAKE_ARGS})\n'
                 b'ExternalProject_Add(lz4_ep\n  URL "https://example.invalid/lz4.tar.gz"\n'
                 + ('  URL_HASH "SHA256=' + archive["sha256"] + '"\n').encode()
                 + b'  ${LZ4_CONFIGURE})\n')
    cfg_path = Path(project["cfg"]["path"])
    command = cfg_path.read_text().removeprefix("cmd='").removesuffix("'\n")
    command += ";-S;<SOURCE_DIR><SOURCE_SUBDIR>;-B;<BINARY_DIR>"
    expanded = (command.replace("<SOURCE_DIR>", str(project["source"]))
                .replace("<SOURCE_SUBDIR>", source_subdir)
                .replace("<BINARY_DIR>", str(project["binary"])))
    project["cfg"] = case.put(cfg_path, ("cmd='" + command + "'\n").encode())
    project["command_record"] = case.put(Path(project["command_record"]["path"]),
                                         ('set(command "' + expanded + '")\n').encode())
    cache = project["binary"] / "CMakeCache.txt"
    cache.write_text(cache.read_text() + "CMAKE_HOME_DIRECTORY:INTERNAL=" + str(project["source"])
                     + source_subdir + "\nCMAKE_CACHEFILE_DIR:INTERNAL=" + str(project["binary"]) + "\n")
    case.refresh()
    if owner is not None:
        case.selected["source_trees"].append({"path": str(owner.parent.parent),
                                              "files": case.inventory(owner.parent.parent)})
        case.selected_ref = case.json(case.work / "compiled-dependencies.json", case.selected)
        case.raw["compiled_dependencies"] = case.selected_ref
        case.raw_ref = case.json(case.work / "build-receipt.json", case.raw)
    return project, expanded.split(";"), owner


@pytest.mark.parametrize("name", ["lz4", "mimalloc", "snappy", "utf8proc"])
@pytest.mark.parametrize("retained", [False, True], ids=["original-files", "retained-forbid-history"])
def test_completed_capture_accepts_actual_cmake_ep_configuration(
        verifier, completed_source_case, tmp_path, monkeypatch, name, retained):
    case = completed_source_case
    project, expanded, owner = use_actual_cmake_ep_configuration(case, name, verifier, monkeypatch)
    if name == "lz4":
        assert not (project["source"] / "CMakeLists.txt").exists()
        assert (project["source"] / "build/cmake/CMakeLists.txt").is_file()
    attestation = case.publish(verifier, tmp_path / "actual-cmake-config-attestation.json")
    result = verifier.verify_completed_attestation(attestation["path"], attestation["sha256"])
    proof = result["attestation"]["external_projects"][name]
    assert proof["configuration"] == project["cfg"]
    assert proof["configure_command_record"] == project["command_record"]
    assert proof["configure_command"] == expanded
    entrypoint = project["source"] / ("build/cmake/CMakeLists.txt" if name == "lz4" else "CMakeLists.txt")
    inputs = {row["path"] for generated in proof["generated_outputs"] for row in generated["inputs"]}
    assert str(entrypoint) in inputs
    if owner is not None:
        assert record(owner) in result["files"]
    if retained:
        retained_root = tmp_path / "retained-cmake"
        retained_root.mkdir()
        retained_inputs = {}
        for reference in result["files"]:
            copied = retained_root / reference["sha256"]
            copied.write_bytes(Path(reference["path"]).read_bytes())
            retained_inputs[reference["sha256"]] = record(copied)
        shutil.rmtree(case.root)
        Path(attestation["path"]).unlink()
        forbid_history(monkeypatch, case.root)
        replay = verifier.verify_completed_attestation(attestation["path"], attestation["sha256"],
                                                      retained_inputs=list(retained_inputs.values()))
        assert replay["attestation"] == result["attestation"]
        assert all(Path(reference["path"]).is_relative_to(retained_root) for reference in replay["files"])


@pytest.mark.parametrize("name", ["lz4", "mimalloc", "snappy", "utf8proc"])
@pytest.mark.parametrize("change", ["unknown_placeholder", "rebound_source", "rebound_binary"])
def test_completed_capture_rejects_unqualified_cmake_ep_configuration(
        verifier, completed_source_case, monkeypatch, name, change):
    case = completed_source_case
    project, _, _ = use_actual_cmake_ep_configuration(case, name, verifier, monkeypatch)
    if change == "unknown_placeholder":
        path = Path(project["cfg"]["path"])
        path.write_text(path.read_text().replace("<SOURCE_SUBDIR>", "<UNOWNED_SUBDIR>"))
    else:
        path = Path(project["command_record"]["path"])
        before = str(project["source"]) if change == "rebound_source" else str(project["binary"])
        after = str(case.root / ("foreign-source" if change == "rebound_source" else "foreign-binary"))
        path.write_text(path.read_text().replace(before, after))
    with pytest.raises(ValueError, match="configur|source|placeholder|entrypoint|binary|ownership"):
        case.capture(verifier)


def test_completed_capture_rejects_lz4_entrypoint_outside_its_pinned_source_declaration(
        verifier, completed_source_case, monkeypatch):
    case = completed_source_case
    _, _, owner = use_actual_cmake_ep_configuration(case, "lz4", verifier, monkeypatch)
    owner.write_text(owner.read_text().replace('SOURCE_SUBDIR "build/cmake"', 'SOURCE_SUBDIR "other/cmake"'))
    tree = next(row for row in case.selected["source_trees"] if row["path"] == str(owner.parent.parent))
    tree["files"] = case.inventory(owner.parent.parent)
    case.raw["compiled_dependencies"] = case.json(case.work / "compiled-dependencies.json", case.selected)
    case.raw_ref = case.json(case.work / "build-receipt.json", case.raw)
    with pytest.raises(ValueError, match="configur|source|entrypoint|subdir|ownership"):
        case.capture(verifier)


JEMALLOC_CONFIGURED_BINS = ("bin/jemalloc-config", "bin/jemalloc.sh", "bin/jeprof")


def use_jemalloc_configured_bins(case, verifier, monkeypatch, *, change=None):
    """Model the separate AC_CONFIG_FILES/BINS outputs of the pinned release."""
    project = case.ep["jemalloc"]
    source = project["source"]
    current = {path.relative_to(source).as_posix(): path.read_bytes()
               for path in source.rglob("*") if path.is_file()}
    previous_originals = case.jemalloc_originals()
    originals = dict(previous_originals)
    declared = JEMALLOC_CONFIGURED_BINS[:-1] if change == "undeclared" else JEMALLOC_CONFIGURED_BINS
    originals["configure"] += (
        'ac_config_files="$ac_config_files $cfgoutputs_tup config.stamp '
        + " ".join(declared) + '"\n').encode()
    originals["configure.ac"] = (
        'AC_CONFIG_FILES([$cfgoutputs_tup config.stamp '
        + " ".join(declared) + '])\n').encode()
    bins = ("BINS := " + " ".join("$(objroot)" + name for name in JEMALLOC_CONFIGURED_BINS)
            + "\n").encode()
    originals["Makefile.in"] += bins
    for name in JEMALLOC_CONFIGURED_BINS:
        originals[name] = ("# original release " + name + "\n").encode()
        originals[name + ".in"] = ("#!/bin/sh\n# template " + name + " @prefix@\n").encode()
    if change == "extra_bin":
        originals["bin/unowned"] = b"# original unowned script\n"
    old = project["archive"]
    archive = case.bundle(Path(old["path"]), source, originals)
    for name, data in current.items():
        if name not in previous_originals or data != previous_originals[name]:
            case.put(source / name, data + bins if name == "Makefile" else data)
    for name in JEMALLOC_CONFIGURED_BINS:
        data = ("#!/bin/sh\n# configured " + name + " " + str(source / "dist") + "\n").encode()
        case.put(source / name, data)
        case.put(source / "dist" / name, data)
    if change == "template_mismatch":
        case.put(source / (JEMALLOC_CONFIGURED_BINS[0] + ".in"), b"unqualified template mutation\n")
    elif change == "extra_bin":
        case.put(source / "bin/unowned", b"unqualified changed script\n")
    project["archive"] = archive
    case.manifest["dependencies"]["jemalloc"].update(archive=archive, sha256=archive["sha256"])
    next(row for row in case.cohort_rows if row["name"] == "jemalloc")["sha256"] = archive["sha256"]
    case.cohort = case.json(Path(case.cohort["path"]), {"sources": case.cohort_rows})
    monkeypatch.setattr(verifier, "COHORT_SHA256", case.cohort["sha256"])
    path = Path(project["urlinfo"]["path"])
    project["urlinfo"] = case.put(path, path.read_bytes().replace(
        old["sha256"].encode(), archive["sha256"].encode()))
    for relative in ("cpp/cmake_modules/ThirdpartyToolchain.cmake", "cpp/thirdparty/versions.txt"):
        path = case.arrow / relative
        path.write_text(path.read_text().replace(old["sha256"], archive["sha256"]))
    case.refresh()
    return project, originals


def replay_completed_source_only_from_copies(case, attestation, initial, verifier, tmp_path, monkeypatch):
    retained_root = tmp_path / "retained-actual-ep"
    retained_root.mkdir()
    retained = {}
    for reference in initial["files"]:
        copied = retained_root / reference["sha256"]
        copied.write_bytes(Path(reference["path"]).read_bytes())
        retained[reference["sha256"]] = record(copied)
    shutil.rmtree(case.root)
    Path(attestation["path"]).unlink()
    forbid_history(monkeypatch, case.root)
    replay = verifier.verify_completed_attestation(attestation["path"], attestation["sha256"],
                                                  retained_inputs=list(retained.values()))
    assert replay["attestation"] == initial["attestation"]
    assert all(Path(row["path"]).is_relative_to(retained_root) for row in replay["files"])


@pytest.mark.parametrize("retained", [False, True], ids=["original-files", "retained-forbid-history"])
def test_jemalloc_configured_bins_preserve_original_and_template_evidence(
        verifier, completed_source_case, tmp_path, monkeypatch, retained):
    case = completed_source_case
    project, originals = use_jemalloc_configured_bins(case, verifier, monkeypatch)
    attestation = case.publish(verifier, tmp_path / "jemalloc-bin-attestation.json")
    initial = verifier.verify_completed_attestation(attestation["path"], attestation["sha256"])
    proof = initial["attestation"]["external_projects"]["jemalloc"]
    generated = {row["output"]["path"]: row for row in proof["generated_outputs"]}
    for name in JEMALLOC_CONFIGURED_BINS:
        row = generated[str(project["source"] / name)]
        assert row["rule"] == "jemalloc-configure-v1"
        assert row["original_sha256"] == digest(originals[name])
        assert row["output"] == record(project["source"] / name)
        inputs = {ref["path"]: ref for ref in row["inputs"]}
        for relative in ("configure", "Makefile.in", name + ".in"):
            ref = record(project["source"] / relative)
            assert inputs[ref["path"]] == ref
        assert digest(originals[name]) != row["output"]["sha256"]
    if retained:
        replay_completed_source_only_from_copies(case, attestation, initial, verifier, tmp_path, monkeypatch)


@pytest.mark.parametrize("change", ["undeclared", "template_mismatch", "extra_bin"])
def test_jemalloc_configured_bins_reject_unqualified_replacements(
        verifier, completed_source_case, monkeypatch, change):
    case = completed_source_case
    use_jemalloc_configured_bins(case, verifier, monkeypatch, change=change)
    with pytest.raises(ValueError, match="source|template|generated|declar|archive|Jemalloc"):
        case.capture(verifier)


def refresh_lz4_owner(case, owner):
    case.refresh()
    case.selected["source_trees"].append({"path": str(owner.parent.parent),
                                          "files": case.inventory(owner.parent.parent)})
    case.raw["compiled_dependencies"] = case.json(case.work / "compiled-dependencies.json", case.selected)
    case.raw_ref = case.json(case.work / "build-receipt.json", case.raw)


def replace_lz4_owner_archive(case, owner, archive_path, archive_bytes, verifier, monkeypatch):
    current = owner.read_bytes()
    archive = case.bundle(archive_path, owner.parent.parent,
                          {"cmake_modules/ThirdpartyToolchain.cmake": archive_bytes})
    case.put(owner, current)
    case.archives = [row for row in case.archives if row["path"] != archive["path"]]
    case.archives.append({**archive, "matches_arrow_versions_pin": True})
    monkeypatch.setattr(verifier, "ORC_ARCHIVE_SHA256", archive["sha256"], raising=False)
    refresh_lz4_owner(case, owner)
    return archive


def use_lz4_unhashed_download(case, verifier, monkeypatch):
    """ORC 2.2.1 owns this no-URL_HASH declaration; LZ4 is pinned separately."""
    project, _, owner = use_actual_cmake_ep_configuration(case, "lz4", verifier, monkeypatch)
    declaration = (
        b'set(LZ4_VERSION "1.10.0")\n'
        b'set(LZ4_CONFIGURE SOURCE_SUBDIR "build/cmake" CMAKE_ARGS ${LZ4_CMAKE_ARGS})\n'
        b'ExternalProject_Add(lz4_ep\n'
        b'  URL "https://github.com/lz4/lz4/archive/v${LZ4_VERSION}.tar.gz"\n'
        b'  ${LZ4_CONFIGURE}\n  ${THIRDPARTY_LOG_OPTIONS}\n'
        b'  BUILD_BYPRODUCTS ${LZ4_STATIC_LIB})\n')
    case.put(owner, declaration)
    info = Path(project["urlinfo"]["path"])
    project["urlinfo"] = case.put(info, (
        "source_dir=" + str(project["source"]) + "\nhash=\n"
        "url(s)=https://github.com/lz4/lz4/archive/v1.10.0.tar.gz\n").encode())
    archive_path = (case.cpp / "_deps/orc-subbuild/orc-populate-prefix/src/orc-2.2.1.tar.gz")
    owner_archive = replace_lz4_owner_archive(case, owner, archive_path, declaration, verifier, monkeypatch)
    return project, owner, owner_archive


@pytest.mark.parametrize("retained", [False, True], ids=["original-files", "retained-forbid-history"])
def test_lz4_unhashed_download_retains_independent_pin_and_original_orc_ownership(
        verifier, completed_source_case, tmp_path, monkeypatch, retained):
    case = completed_source_case
    project, owner, owner_archive = use_lz4_unhashed_download(case, verifier, monkeypatch)
    attestation = case.publish(verifier, tmp_path / "lz4-empty-hash-attestation.json")
    initial = verifier.verify_completed_attestation(attestation["path"], attestation["sha256"])
    proof = initial["attestation"]["external_projects"]["lz4"]
    assert proof["archive"] == project["archive"]
    assert proof["archive"]["sha256"] == verifier.EP_ARCHIVE_SHA256["lz4"]
    assert proof["download_record"] == project["urlinfo"]
    assert b"\nhash=\n" in Path(proof["download_record"]["path"]).read_bytes()
    assert proof["source_subdir_declaration"] == record(owner)
    assert owner_archive in initial["files"]
    assert record(owner) in initial["files"]
    if retained:
        replay_completed_source_only_from_copies(case, attestation, initial, verifier, tmp_path, monkeypatch)


@pytest.mark.parametrize("change", ["other_ep_empty_hash", "wrong_independent_pin", "foreign_owner",
                                    "owner_archive_mismatch", "upstream_hash_conflict", "foreign_url"])
def test_lz4_unhashed_download_rejects_unowned_or_conflicting_download_evidence(
        verifier, completed_source_case, monkeypatch, change):
    case = completed_source_case
    project, owner, owner_archive = use_lz4_unhashed_download(case, verifier, monkeypatch)
    if change == "other_ep_empty_hash":
        other = case.ep["mimalloc"]
        path = Path(other["urlinfo"]["path"])
        path.write_text(path.read_text().replace("hash=SHA256=" + other["archive"]["sha256"], "hash="))
    elif change == "wrong_independent_pin":
        monkeypatch.setitem(verifier.EP_ARCHIVE_SHA256, "lz4", "0" * 64)
    elif change == "foreign_owner":
        tree = next(row for row in case.selected["source_trees"] if row["path"] == str(owner.parent.parent))
        tree["path"] = str(case.root / "foreign-orc-source")
        case.raw["compiled_dependencies"] = case.json(case.work / "compiled-dependencies.json", case.selected)
        case.raw_ref = case.json(case.work / "build-receipt.json", case.raw)
    elif change == "owner_archive_mismatch":
        replace_lz4_owner_archive(case, owner, Path(owner_archive["path"]),
                                  owner.read_bytes() + b"# different original member\n", verifier, monkeypatch)
    elif change == "upstream_hash_conflict":
        data = owner.read_bytes().replace(b"  ${LZ4_CONFIGURE}\n", (
            '  URL_HASH "SHA256=' + project["archive"]["sha256"] + '"\n  ${LZ4_CONFIGURE}\n').encode())
        case.put(owner, data)
        replace_lz4_owner_archive(case, owner, Path(owner_archive["path"]), data, verifier, monkeypatch)
    elif change == "foreign_url":
        path = Path(project["urlinfo"]["path"])
        path.write_text(path.read_text().replace("https://github.com/lz4/lz4/archive/v1.10.0.tar.gz",
                                               "https://example.invalid/foreign.tar.gz"))
    with pytest.raises(ValueError, match="archive|source|configur|ownership|download|LZ4|hash"):
        case.capture(verifier)


ARROW_ARCHIVE_EMPTY_DIRECTORIES = ("testing", "cpp/submodules", "cpp/submodules/parquet-testing")


def use_arrow_archive_empty_directories(case, *, directory_member=True):
    """The frozen recipes validate directories but emit only files/symlinks."""
    path = Path(case.manifest["source_archive"]["path"])
    with tarfile.open(path) as original:
        contents = [(member.name, original.extractfile(member).read())
                    for member in original.getmembers() if member.isfile()]
    with tarfile.open(path, "w:gz") as archive:
        for name, data in contents:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        for relative in ARROW_ARCHIVE_EMPTY_DIRECTORIES:
            member = tarfile.TarInfo("upstream/" + relative)
            if directory_member:
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
            else:
                data = b"original archive file, not a directory\n"
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            (case.arrow / relative).mkdir(parents=True, exist_ok=True)
    case.manifest["source_archive"] = record(path)
    case.refresh()


@pytest.mark.parametrize("retained", [False, True], ids=["original-files", "retained-forbid-history"])
def test_completed_arrow_source_accepts_only_archive_owned_empty_directories(
        verifier, completed_source_case, tmp_path, monkeypatch, retained):
    case = completed_source_case
    use_arrow_archive_empty_directories(case)
    compiled = json.loads(Path(case.source_ref["path"]).read_bytes())
    assert not set(ARROW_ARCHIVE_EMPTY_DIRECTORIES).intersection(row["relative"] for row in compiled["files"])
    attestation = case.publish(verifier, tmp_path / "archive-empty-directory-attestation.json")
    initial = verifier.verify_completed_attestation(attestation["path"], attestation["sha256"])
    assert case.manifest["source_archive"] in initial["files"]
    assert initial["attestation"]["subject"]["compiled_sources"] == case.source_ref
    if retained:
        replay_completed_source_only_from_copies(case, attestation, initial, verifier, tmp_path, monkeypatch)


@pytest.mark.parametrize("change", ["unknown_directory", "wrong_type", "missing_directory",
                                    "archive_file_member", "archive_hash"])
def test_completed_arrow_source_rejects_unqualified_empty_directory_layout(
        verifier, completed_source_case, change):
    case = completed_source_case
    use_arrow_archive_empty_directories(case, directory_member=change != "archive_file_member")
    if change == "unknown_directory":
        (case.arrow / "unrecorded-empty-directory").mkdir()
    elif change == "wrong_type":
        path = case.arrow / "testing"
        path.rmdir()
        path.write_bytes(b"directory replaced with a file\n")
    elif change == "missing_directory":
        (case.arrow / "testing").rmdir()
    elif change == "archive_hash":
        path = Path(case.manifest["source_archive"]["path"])
        path.write_bytes(path.read_bytes() + b"changed referenced archive bytes")
    with pytest.raises(ValueError, match="source|archive|layout|checksum|director|object"):
        case.capture(verifier)


def use_cmake_optional_ninja_compiler_fields(case, *, explicit=False, change=None):
    """LZ4's built CLI objects omit empty flags but do not enter its PIC archive."""
    project = case.ep["lz4"]
    binary = project["binary"]
    static_object = project["object"].relative_to(binary).as_posix()
    cli_object = "CMakeFiles/provider_cli.dir/source.c.o"
    case.put(binary / cli_object, b"\x7fELF\x02\x01\x01\x00synthetic non-PIC CLI object\n")
    graph = ""
    if change == "inherited_global":
        graph += "DEFINES = -DUNQUALIFIED_GLOBAL=1\n"
    for obj, static in ((static_object, True), (cli_object, False)):
        graph += "build " + obj + ": C_COMPILER__provider " + str(project["input"]) + "\n"
        graph += "  DEP_FILE = " + obj + ".d\n"
        if static and change != "static_missing_pic":
            graph += "  FLAGS = -fPIC\n"
        elif explicit:
            graph += "  FLAGS = -O2\n"
        if explicit:
            graph += "  DEFINES = -DRETAINED_OPTION=7\n  INCLUDES = -I" + str(project["source"]) + "\n"
    graph += "build liblz4.a: C_STATIC_LIBRARY_LINKER__provider " + static_object + "\n"
    command = ("${LAUNCHER}${CODE_CHECK}" + case.toolchain["cc"]["path"]
               + " $DEFINES $INCLUDES $FLAGS -MD -MT $out -MF $DEP_FILE -o $out -c $in")
    if change in {"DEP_FILE", "SOURCE", "OBJECT", "UNOWNED"}:
        if change == "DEP_FILE":
            graph = "\n".join(line for line in graph.split("\n") if not line.startswith("  DEP_FILE = "))
        else:
            command += " $" + change
    rules = "rule C_COMPILER__provider\n  command = " + command + "\n"
    if change == "inherited_rule":
        rules += "  DEFINES = -DUNQUALIFIED_RULE=1\n"
    rules += ("rule C_STATIC_LIBRARY_LINKER__provider\n  command = "
              + case.toolchain["ar"]["path"] + " qc $out $in\n")
    graph_ref = case.put(binary / "build.ninja", graph.encode())
    rules_ref = case.put(binary / "CMakeFiles/rules.ninja", rules.encode())
    return project, binary / cli_object, graph_ref, rules_ref


@pytest.mark.parametrize("explicit", [False, True], ids=["omitted-empty-values", "preserve-explicit-values"])
@pytest.mark.parametrize("retained", [False, True], ids=["original-files", "retained-forbid-history"])
def test_cmake_optional_ninja_compiler_fields_preserve_static_and_cli_evidence(
        verifier, completed_source_case, tmp_path, monkeypatch, retained, explicit):
    case = completed_source_case
    project, cli_object, graph, rules = use_cmake_optional_ninja_compiler_fields(case, explicit=explicit)
    attestation = case.publish(verifier, tmp_path / "ninja-optional-fields-attestation.json")
    initial = verifier.verify_completed_attestation(attestation["path"], attestation["sha256"])
    proof = initial["attestation"]["external_projects"]["lz4"]
    objects = {row["object"]["path"]: row for row in proof["objects"]}
    assert set(objects) == {str(project["object"]), str(cli_object)}
    assert objects[str(project["object"])]["pic"] is True
    assert objects[str(cli_object)]["pic"] is False
    for row in objects.values():
        assert row["command_record"] == graph and row["rule_record"] == rules
        assert row["compiler"] == case.toolchain["cc"]
        assert not any("$" in value for value in row["command"])
        assert ("-DRETAINED_OPTION=7" in row["command"]) is explicit
        assert ("-I" + str(project["source"]) in row["command"]) is explicit
    archive_objects = {ref["path"] for archive in proof["archives"] for ref in archive["objects"]}
    assert archive_objects == {str(project["object"])}
    if retained:
        replay_completed_source_only_from_copies(case, attestation, initial, verifier, tmp_path, monkeypatch)


@pytest.mark.parametrize("change", ["DEP_FILE", "SOURCE", "OBJECT", "UNOWNED", "static_missing_pic",
                                    "inherited_global", "inherited_rule"])
def test_cmake_optional_ninja_compiler_fields_reject_unqualified_or_required_values(
        verifier, completed_source_case, change):
    case = completed_source_case
    use_cmake_optional_ninja_compiler_fields(case, change=change)
    with pytest.raises(ValueError, match="field|PIC|pic|compiler|variable|binding"):
        case.capture(verifier)


def use_source_owned_cmake_exports(case, verifier, monkeypatch, *, configuration="", change=None):
    project, _, owner = use_actual_cmake_ep_configuration(case, "lz4", verifier, monkeypatch)
    old = project["archive"]
    with tarfile.open(old["path"]) as archive:
        contents = {member.name.split("/", 1)[1]: archive.extractfile(member).read()
                    for member in archive.getmembers() if member.isfile()}
    entrypoint = "build/cmake/CMakeLists.txt"
    contents[entrypoint] += (
        b'set(LZ4_PKG_INSTALLDIR "${CMAKE_INSTALL_LIBDIR}/cmake/lz4")\n'
        b'install(TARGETS provider EXPORT lz4Targets ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")\n'
        b'export(EXPORT lz4Targets FILE ${CMAKE_CURRENT_BINARY_DIR}/lz4Targets.cmake NAMESPACE LZ4::)\n')
    if change != "missing_install_declaration":
        contents[entrypoint] += (
            b'install(EXPORT lz4Targets FILE lz4Targets.cmake NAMESPACE LZ4:: '
            b'DESTINATION ${LZ4_PKG_INSTALLDIR})\n')
    archive = case.bundle(Path(old["path"]), project["source"], contents)
    project["archive"] = archive
    case.archives = [{**row, "sha256": archive["sha256"]} if row["path"] == archive["path"] else row
                     for row in case.archives]
    monkeypatch.setitem(verifier.EP_ARCHIVE_SHA256, "lz4", archive["sha256"])
    for path in (Path(project["urlinfo"]["path"]), owner,
                 case.arrow / "cpp/cmake_modules/ThirdpartyToolchain.cmake",
                 case.arrow / "cpp/thirdparty/versions.txt"):
        path.write_text(path.read_text().replace(old["sha256"], archive["sha256"]))
    cache = project["binary"] / "CMakeCache.txt"
    case.put(cache, cache.read_bytes() + (
        "CMAKE_BUILD_TYPE:STRING=" + configuration + "\nCMAKE_INSTALL_LIBDIR:PATH=lib\n"
        "CMAKE_INSTALL_PREFIX:PATH=" + str(project["static"].parent.parent) + "\n").encode())
    destination = "lib/cmake/lz4"
    folder = hashlib.md5(destination.encode(), usedforsecurity=False).hexdigest()
    stem = "lz4Targets" if change != "wrong_name" else "unownedTargets"
    if change == "wrong_digest":
        folder = "0" * 32
    suffix = configuration.lower() if configuration else "noconfig"
    if change == "wrong_configuration":
        suffix = "debug"
    paths = [project["binary"] / (stem + ".cmake"),
             project["binary"] / "CMakeFiles/Export" / folder / (stem + ".cmake"),
             project["binary"] / "CMakeFiles/Export" / folder / (stem + "-" + suffix + ".cmake")]
    for path in paths:
        case.put(path, ("# Generated CMake export: " + path.name + "\n").encode())
    script = ""
    for index, path in enumerate(paths[1:]):
        if change == "unreferenced_install_source" and index == 1:
            continue
        installed_source = path if change != "foreign_install_source" else case.root / "foreign" / path.name
        script += ('file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/' + destination
                   + '" TYPE FILE FILES "' + str(installed_source) + '")\n')
    install_script = case.put(project["binary"] / "cmake_install.cmake", script.encode())
    refresh_lz4_owner(case, owner)
    return project, paths, record(cache), install_script


@pytest.mark.parametrize("configuration", ["", "Release"], ids=["noconfig", "release"])
@pytest.mark.parametrize("retained", [False, True], ids=["original-files", "retained-forbid-history"])
def test_source_owned_cmake_exports_bind_original_declarations_and_install_records(
        verifier, completed_source_case, tmp_path, monkeypatch, configuration, retained):
    case = completed_source_case
    project, paths, cache, install_script = use_source_owned_cmake_exports(
        case, verifier, monkeypatch, configuration=configuration)
    attestation = case.publish(verifier, tmp_path / "cmake-export-attestation.json")
    initial = verifier.verify_completed_attestation(attestation["path"], attestation["sha256"])
    proof = initial["attestation"]["external_projects"]["lz4"]
    generated = {row["output"]["path"]: row for row in proof["generated_outputs"]}
    original_cmake = record(project["source"] / "build/cmake/CMakeLists.txt")
    for index, path in enumerate(paths):
        row = generated[str(path)]
        assert row["rule"] == "cmake-export-file-v1"
        assert row["output"] == record(path)
        assert original_cmake in row["inputs"] and cache in row["inputs"]
        if index:
            assert install_script in row["inputs"]
    if retained:
        replay_completed_source_only_from_copies(case, attestation, initial, verifier, tmp_path, monkeypatch)


@pytest.mark.parametrize("change", ["wrong_name", "wrong_digest", "wrong_configuration",
                                    "unreferenced_install_source", "foreign_install_source",
                                    "missing_install_declaration"])
def test_source_owned_cmake_exports_reject_unqualified_names_configurations_and_paths(
        verifier, completed_source_case, monkeypatch, change):
    case = completed_source_case
    use_source_owned_cmake_exports(case, verifier, monkeypatch, change=change)
    with pytest.raises(ValueError, match="unqualified|export|Export|install|source|configuration"):
        case.capture(verifier)


def use_remaining_source_owned_lz4_metadata(case, verifier, monkeypatch, *, change=None):
    project, _, _, _ = use_source_owned_cmake_exports(case, verifier, monkeypatch)
    old = project["archive"]
    with tarfile.open(old["path"]) as archive:
        contents = {member.name.split("/", 1)[1]: archive.extractfile(member).read()
                    for member in archive.getmembers() if member.isfile()}
    cmake = "build/cmake/CMakeLists.txt"
    contents[cmake] += (
        b'set(LZ4_TOP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")\n'
        b'set(LZ4_LIB_SOURCE_DIR "${LZ4_TOP_SOURCE_DIR}/lib")\n'
        + (b'include(CPack)\n' if change != "missing_cpack_include" else b'')
        + b'configure_package_config_file("${CMAKE_CURRENT_LIST_DIR}/lz4Config.cmake.in" '
        b'"${CMAKE_CURRENT_BINARY_DIR}/lz4Config.cmake" INSTALL_DESTINATION ${LZ4_PKG_INSTALLDIR})\n'
        b'configure_file(${LZ4_LIB_SOURCE_DIR}/liblz4.pc.in liblz4.pc @ONLY)\n'
        b'foreach(cli_tool IN ITEMS lz4cat unlz4)\n'
        b'  add_custom_target("create_${cli_tool}_symlink" ALL\n'
        b'    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:lz4cli> ${cli_tool}\n'
        b'    COMMENT "Creating symlink for ${cli_tool}" VERBATIM)\nendforeach()\n'
        b'foreach(f lz4cat unlz4)\n'
        b'  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${f}.1" ".so man1/lz4.1\\n")\nendforeach()\n')
    contents["README.md"] = b"LZ4 project description\n"
    contents["LICENSE"] = b"LZ4 source license\n"
    contents["lib/lz4.h"] = (b"#define LZ4_VERSION_MAJOR 1\n#define LZ4_VERSION_MINOR 10\n"
                              b"#define LZ4_VERSION_RELEASE 0\n")
    contents["lib/liblz4.pc.in"] = (b"prefix=@PREFIX@\nlibdir=@LIBDIR@\nincludedir=@INCLUDEDIR@\n"
                                    b"Version: @VERSION@\nLibs: -L${libdir} -llz4\n")
    contents["build/cmake/lz4Config.cmake.in"] = (
        b'@PACKAGE_INIT@\ninclude("${CMAKE_CURRENT_LIST_DIR}/lz4Targets.cmake")\n')
    archive = case.bundle(Path(old["path"]), project["source"], contents)
    project["archive"] = archive
    case.archives = [{**row, "sha256": archive["sha256"]} if row["path"] == archive["path"] else row
                     for row in case.archives]
    monkeypatch.setitem(verifier.EP_ARCHIVE_SHA256, "lz4", archive["sha256"])
    owner = case.cpp / "_deps/orc-src/cmake_modules/ThirdpartyToolchain.cmake"
    for path in (Path(project["urlinfo"]["path"]), owner,
                 case.arrow / "cpp/cmake_modules/ThirdpartyToolchain.cmake",
                 case.arrow / "cpp/thirdparty/versions.txt"):
        path.write_text(path.read_text().replace(old["sha256"], archive["sha256"]))
    cmake_root = case.root / "cmake-data"
    module_bytes = {
        "Modules/CPack.cmake": (
            b'set(cpack_input_file "${CMAKE_ROOT}/Templates/CPackConfig.cmake.in")\n'
            b'set(cpack_source_input_file "${CMAKE_ROOT}/Templates/CPackConfig.cmake.in")\n'
            b'_cpack_set_default(CPACK_OUTPUT_CONFIG_FILE "${CMAKE_BINARY_DIR}/CPackConfig.cmake")\n'
            b'_cpack_set_default(CPACK_SOURCE_OUTPUT_CONFIG_FILE "${CMAKE_BINARY_DIR}/CPackSourceConfig.cmake")\n'
            b'configure_file("${cpack_input_file}" "${CPACK_OUTPUT_CONFIG_FILE}" @ONLY)\n'
            b'configure_file("${cpack_source_input_file}" "${CPACK_SOURCE_OUTPUT_CONFIG_FILE}" @ONLY)\n'),
        "Templates/CPackConfig.cmake.in": (
            b"# CPack generated configuration\n@_CPACK_OTHER_VARIABLES_@\n"
            b'set(CPACK_PROPERTIES_FILE "@CMAKE_BINARY_DIR@/CPackProperties.cmake")\n'),
        "Modules/CMakePackageConfigHelpers.cmake": b"# pinned package-config generator\n",
    }
    module_refs = {name: case.put(cmake_root / name, data) for name, data in module_bytes.items()}
    monkeypatch.setattr(verifier, "CMAKE_LZ4_METADATA_SHA256", {
        "4.2.3": {name: row["sha256"] for name, row in module_refs.items()}}, raising=False)
    cache = project["binary"] / "CMakeCache.txt"
    case.put(cache, cache.read_bytes() + ("CMAKE_ROOT:INTERNAL=" + str(cmake_root)
                                        + "\nCMAKE_INSTALL_INCLUDEDIR:PATH=include\n").encode())
    outputs = {}
    for name in ("CPackConfig.cmake", "CPackSourceConfig.cmake"):
        variables = {
            "CPACK_BUILD_SOURCE_DIRS": str(project["source"] / "build/cmake") + ";" + str(project["binary"]),
            "CPACK_OUTPUT_CONFIG_FILE": str(project["binary"] / "CPackConfig.cmake"),
            "CPACK_SOURCE_OUTPUT_CONFIG_FILE": str(project["binary"] / "CPackSourceConfig.cmake"),
            "CPACK_INSTALL_PREFIX": str(project["static"].parent.parent),
        }
        if change == "rebound_cpack_output":
            variables["CPACK_OUTPUT_CONFIG_FILE"] = str(case.root / "foreign-config.cmake")
        body = "\n".join('set(' + key + ' "' + value + '")' for key, value in variables.items()).encode()
        data = module_bytes["Templates/CPackConfig.cmake.in"].replace(
            b"@_CPACK_OTHER_VARIABLES_@", body).replace(b"@CMAKE_BINARY_DIR@", str(project["binary"]).encode())
        outputs[name] = case.put(project["binary"] / name, data)
    template = contents["lib/liblz4.pc.in"]
    for key, value in {"PREFIX": str(project["static"].parent.parent), "LIBDIR": "${prefix}/lib",
                       "INCLUDEDIR": "${prefix}/include", "VERSION": "1.10.0"}.items():
        template = template.replace(("@" + key + "@").encode(), value.encode())
    outputs["liblz4.pc"] = case.put(project["binary"] / "liblz4.pc", template)
    outputs["lz4Config.cmake"] = case.put(project["binary"] / "lz4Config.cmake", (
        b"\n####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######\n"
        b"# configured package prefix metadata\n"
        + contents["build/cmake/lz4Config.cmake.in"].replace(b"@PACKAGE_INIT@\n", b"")))
    for name in ("lz4cat.1", "unlz4.1"):
        outputs[name] = case.put(project["binary"] / name, b".so man1/lz4.1\n")
    _, cli_object, graph_ref, _ = use_cmake_optional_ninja_compiler_fields(case)
    executable = case.put(project["binary"] / "lz4", b"\x7fELF synthetic built CLI\n")
    graph = Path(graph_ref["path"]).read_text()
    graph += ("build lz4: C_EXECUTABLE_LINKER__lz4cli " + str(cli_object.relative_to(project["binary"])) + "\n")
    for name in ("lz4cat", "unlz4"):
        graph += ("build CMakeFiles/create_" + name + "_symlink: CUSTOM_COMMAND || lz4\n"
                  "  COMMAND = cd " + str(project["binary"]) + " && " + case.toolchain["cmake"]["path"]
                  + " -E create_symlink " + ("foreign" if change == "foreign_symlink_command" else "lz4")
                  + " " + name + "\n")
        (project["binary"] / name).symlink_to("wrong" if change == "wrong_link_target" else "lz4")
    graph_ref = case.put(Path(graph_ref["path"]), graph.encode())
    if change == "unknown_name":
        case.put(project["binary"] / "CPackUnownedConfig.cmake", b"# unqualified packaging output\n")
    elif change == "foreign_template":
        case.put(cmake_root / "Templates/CPackConfig.cmake.in", b"# foreign CPack template\n")
    elif change == "foreign_module":
        case.put(cmake_root / "Modules/CPack.cmake", b"# foreign CPack module\n")
    elif change == "pkg_bytes":
        case.put(project["binary"] / "liblz4.pc", b"wrong configured package metadata\n")
    elif change == "template_bytes":
        case.put(project["source"] / "lib/liblz4.pc.in", b"unqualified original template mutation\n")
    elif change == "man_bytes":
        case.put(project["binary"] / "lz4cat.1", b"wrong manpage alias\n")
    refresh_lz4_owner(case, owner)
    case.raw["toolchain_versions"] = {"cmake": "cmake version 4.2.3"}
    case.raw_ref = case.json(case.work / "build-receipt.json", case.raw)
    return project, outputs, module_refs, record(cache), graph_ref, executable


@pytest.mark.parametrize("retained", [False, True], ids=["original-files", "retained-forbid-history"])
def test_remaining_lz4_metadata_proves_all_eight_original_module_template_and_graph_outputs(
        verifier, completed_source_case, tmp_path, monkeypatch, retained):
    case = completed_source_case
    project, outputs, modules, cache, graph, executable = use_remaining_source_owned_lz4_metadata(
        case, verifier, monkeypatch)
    attestation = case.publish(verifier, tmp_path / "remaining-lz4-metadata-attestation.json")
    initial = verifier.verify_completed_attestation(attestation["path"], attestation["sha256"])
    proof = initial["attestation"]["external_projects"]["lz4"]
    generated = {row["output"]["path"]: row for row in proof["generated_outputs"]}
    original_cmake = record(project["source"] / "build/cmake/CMakeLists.txt")
    assert len(outputs) == 6
    for name, output in outputs.items():
        row = generated[output["path"]]
        assert row["rule"] == "lz4-source-metadata-v1"
        assert original_cmake in row["inputs"] and cache in row["inputs"]
        if name.startswith("CPack"):
            assert modules["Modules/CPack.cmake"] in row["inputs"]
            assert modules["Templates/CPackConfig.cmake.in"] in row["inputs"]
        elif name == "liblz4.pc":
            assert record(project["source"] / "lib/liblz4.pc.in") in row["inputs"]
        elif name == "lz4Config.cmake":
            assert record(project["source"] / "build/cmake/lz4Config.cmake.in") in row["inputs"]
            assert modules["Modules/CMakePackageConfigHelpers.cmake"] in row["inputs"]
    links = {row["path"]: row for row in proof["generated_symlinks"]}
    assert set(links) == {str(project["binary"] / name) for name in ("lz4cat", "unlz4")}
    for row in links.values():
        assert row["rule"] == "lz4-source-symlink-v1" and row["target"] == "lz4"
        assert original_cmake in row["inputs"] and graph in row["inputs"] and executable in row["inputs"]
        assert row["command_record"] == graph
    if retained:
        replay_completed_source_only_from_copies(case, attestation, initial, verifier, tmp_path, monkeypatch)


@pytest.mark.parametrize("change", ["unknown_name", "missing_cpack_include", "foreign_template", "foreign_module",
                                    "rebound_cpack_output", "pkg_bytes", "template_bytes", "man_bytes",
                                    "wrong_link_target", "foreign_symlink_command"])
def test_remaining_lz4_metadata_rejects_foreign_names_templates_bytes_paths_and_link_targets(
        verifier, completed_source_case, monkeypatch, change):
    case = completed_source_case
    use_remaining_source_owned_lz4_metadata(case, verifier, monkeypatch, change=change)
    with pytest.raises(ValueError, match="unqualified|metadata|template|module|CPack|source|symlink|alias"):
        case.capture(verifier)


@pytest.fixture
def actual_flat_arrow25_case(tmp_path, verifier, monkeypatch):
    return CompletedSourceCase(tmp_path, "25", verifier, monkeypatch)


@pytest.mark.parametrize("retained", [False, True], ids=["original-files", "retained-forbid-history"])
def test_arrow25_flat_receipt_binds_builder_stages_and_patched_source_consumers(
        verifier, actual_flat_arrow25_case, tmp_path, monkeypatch, retained):
    case = actual_flat_arrow25_case
    assert "build" not in case.raw and "toolchain" not in case.raw
    assert case.raw["builder"] == case.manifest["builder"]
    attestation = case.publish(verifier, tmp_path / "flat-arrow25-attestation.json")
    initial = verifier.verify_completed_attestation(attestation["path"], attestation["sha256"])
    assert initial["attestation"]["schema_kind"] == "25"
    assert initial["attestation"]["artifact_closure"]["build_commands"] == case.raw["stages"]
    required = [case.manifest["builder"]["receipt"], *case.manifest["builder"]["toolchain"].values(),
                *case.patched_consumers, *case.patched_units]
    assert all(row in initial["files"] for row in required)
    if retained:
        replay_completed_source_only_from_copies(case, attestation, initial, verifier, tmp_path, monkeypatch)


def test_arrow25_flat_receipt_cli_publishes_only_outside_its_actual_manifest_work(
        verifier, actual_flat_arrow25_case, tmp_path):
    case = actual_flat_arrow25_case
    output = tmp_path / "flat-arrow25-cli-attestation.json"
    raw_before = Path(case.raw_ref["path"]).read_bytes()
    argv = ["--build-receipt", case.raw_ref["path"], "--build-receipt-sha256", case.raw_ref["sha256"],
            "--cohort", case.cohort["path"], "--output", str(output)]
    assert verifier.main(argv) == 0
    assert verifier.verify_completed_attestation(str(output), record(output)["sha256"])["attestation"]["schema_kind"] == "25"
    assert Path(case.raw_ref["path"]).read_bytes() == raw_before


@pytest.mark.parametrize("change", ["nested_only", "ambiguous_layout", "context", "commands", "stage_command",
                                    "failed_stage", "builder_backlink", "builder_content", "tool_context",
                                    "patched_field", "compile_evidence"])
def test_arrow25_flat_receipt_rejects_wrong_shape_or_unbound_build_source_evidence(
        verifier, actual_flat_arrow25_case, change):
    case = actual_flat_arrow25_case
    if change in {"nested_only", "ambiguous_layout"}:
        case.raw["build"] = {"context": case.raw["context"], "stages": case.raw["stages"]}
        case.raw["toolchain"] = case.raw["builder"]["toolchain"]
        if change == "nested_only":
            for key in ("context", "stages", "commands", "builder"):
                case.raw.pop(key)
    elif change == "context":
        case.raw["context"]["work"] = str(case.root / "foreign-build")
    elif change == "commands":
        case.raw["commands"]["cpp_build"][1] = "--foreign-build"
    elif change == "stage_command":
        case.raw["stages"][0]["command"] = [case.toolchain["cmake"]["path"], "--foreign-build"]
    elif change == "failed_stage":
        case.raw["stages"][0]["exit_code"] = 1
    elif change == "builder_backlink":
        Path(case.manifest["builder"]["receipt"]["path"]).write_bytes(b"changed frozen builder receipt\n")
    elif change == "builder_content":
        case.raw["builder"]["toolchain"]["cc"] = case.put(case.root / "foreign-cc", b"foreign compiler\n")
    elif change == "tool_context":
        context = {**case.manifest["context"], "cc": str(case.root / "foreign-cc")}
        case.manifest["context"] = context
        case.raw["context"] = dict(context)
        case.raw["build_inputs"] = case.json(case.work / "build-inputs.json", case.manifest)
    elif change == "patched_field":
        compilation = json.loads(Path(case.source_ref["path"]).read_bytes())
        compilation["integrated_header_consumers"] = compilation.pop("patched_header_consumers")
        case.raw["compiled_sources"] = case.json(case.work / "compiled-sources.json", compilation)
        case.raw["compile_source_evidence"] = {key: value for key, value in compilation.items() if key != "files"}
    elif change == "compile_evidence":
        case.raw["compile_source_evidence"] = {**case.raw["compile_source_evidence"], "patched_header_consumers": []}
    case.raw_ref = case.json(case.work / "build-receipt.json", case.raw)
    with pytest.raises(ValueError, match="shape|context|builder|toolchain|command|stage|evidence|consumer|receipt|header|compiled|checksum|family|failed"):
        case.capture(verifier)


def test_arrow25_flat_receipt_layout_is_not_a_fallback_for_arrow26(
        verifier, tmp_path, monkeypatch):
    case = CompletedSourceCase(tmp_path, "26", verifier, monkeypatch)
    build = case.raw.pop("build")
    case.raw["context"], case.raw["stages"] = build["context"], build["stages"]
    case.raw["builder"] = {"toolchain": case.raw.pop("toolchain")}
    case.raw_ref = case.json(case.work / "build-receipt.json", case.raw)
    with pytest.raises(ValueError, match="context|build|toolchain|shape"):
        case.capture(verifier)


XSIMD_ARROW25_DECLARATION = b'''macro(build_xsimd)
  message(STATUS "Building xsimd from source")
  set(XSIMD_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/xsimd_ep/src/xsimd_ep-install")
  set(XSIMD_CMAKE_ARGS ${EP_COMMON_CMAKE_ARGS} "-DCMAKE_INSTALL_PREFIX=${XSIMD_PREFIX}")

  externalproject_add(xsimd_ep
                      ${EP_COMMON_OPTIONS}
                      PREFIX "${CMAKE_BINARY_DIR}"
                      URL ${XSIMD_SOURCE_URL}
                      URL_HASH "SHA256=${ARROW_XSIMD_BUILD_SHA256_CHECKSUM}"
                      CMAKE_ARGS ${XSIMD_CMAKE_ARGS})

  set(XSIMD_INCLUDE_DIR "${XSIMD_PREFIX}/include")
  # The include directory must exist before it is referenced by a target.
  file(MAKE_DIRECTORY "${XSIMD_INCLUDE_DIR}")

  add_library(arrow::xsimd INTERFACE IMPORTED)
  target_include_directories(arrow::xsimd INTERFACE "${XSIMD_INCLUDE_DIR}")
  add_dependencies(arrow::xsimd xsimd_ep)

  set(XSIMD_VENDORED TRUE)
endmacro()
'''


def add_arrow25_header_only_xsimd(case, verifier, monkeypatch):
    """Arrow25 uses PREFIX cpp-build, not the common *_ep-prefix layout."""
    source, binary = case.cpp / "src/xsimd_ep", case.cpp / "src/xsimd_ep-build"
    prefix = case.cpp / "xsimd_ep/src/xsimd_ep-install"
    include = prefix / "include"
    original = {
        "CMakeLists.txt": b"project(xsimd)\nadd_library(xsimd INTERFACE)\n"
                          b"install(DIRECTORY include/xsimd DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})\n",
        "include/xsimd/xsimd.hpp": b"#include <xsimd/config/xsimd_config.hpp>\n",
        "include/xsimd/config/xsimd_config.hpp": b"#define XSIMD_VERSION_MAJOR 14\n"
                                                   b"#define XSIMD_VERSION_MINOR 2\n"
                                                   b"#define XSIMD_VERSION_PATCH 0\n",
        "LICENSE": b"synthetic upstream license\n",
    }
    archive = case.bundle(case.cpp / "src/14.2.0.tar.gz", source, original)
    monkeypatch.setattr(verifier, "XSIMD_ARCHIVE_SHA256", archive["sha256"], raising=False)
    case.archives.append({**archive, "matches_arrow_versions_pin": True})
    installed = [case.put(include / name.removeprefix("include/"), data)
                 for name, data in original.items() if name.startswith("include/")]
    # These unused package exports are outside the claimed installed-header payload.
    unused_metadata = case.put(prefix / "share/cmake/xsimd/xsimdTargets.cmake", b"unused package export\n")
    stamp = case.cpp / "src/xsimd_ep-stamp"
    args = [case.toolchain["cmake"]["path"], "-DBUILD_SHARED_LIBS=OFF", "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_C_COMPILER=" + case.toolchain["cc"]["path"],
            "-DCMAKE_CXX_COMPILER=" + case.toolchain["cxx"]["path"], "-DCMAKE_CXX_STANDARD=20",
            "-DCMAKE_INSTALL_PREFIX=" + str(prefix), "-GNinja", "-S", "<SOURCE_DIR><SOURCE_SUBDIR>",
            "-B", "<BINARY_DIR>"]
    cfg = case.put(case.cpp / "tmp/xsimd_ep-cfgcmd.txt", ("cmd='" + ";".join(args) + "'\n").encode())
    expanded = [value.replace("<SOURCE_DIR><SOURCE_SUBDIR>", str(source)).replace("<BINARY_DIR>", str(binary))
                for value in args]
    command = case.put(stamp / "xsimd_ep-configure-Release.cmake",
                       ('set(command "' + ";".join(expanded) + '")\n').encode())
    download = case.put(stamp / "xsimd_ep-urlinfo.txt", (
        "source_dir=" + str(source) + "\nhash=SHA256=" + archive["sha256"]
        + "\nurl(s)=https://github.com/xtensor-stack/xsimd/archive/14.2.0.tar.gz\n").encode())
    cache_values = {
        "CMAKE_HOME_DIRECTORY": str(source), "CMAKE_CACHEFILE_DIR": str(binary),
        "CMAKE_INSTALL_PREFIX": str(prefix), "CMAKE_BUILD_TYPE": "Release", "BUILD_SHARED_LIBS": "OFF",
        "CMAKE_C_COMPILER": case.toolchain["cc"]["path"], "CMAKE_CXX_COMPILER": case.toolchain["cxx"]["path"],
        "CMAKE_COMMAND": case.toolchain["cmake"]["path"], "CMAKE_MAKE_PROGRAM": case.toolchain["ninja"]["path"],
        "CMAKE_CXX_STANDARD": "20", "BUILD_TESTS": "OFF", "BUILD_BENCHMARK": "OFF", "BUILD_EXAMPLES": "OFF",
        "XSIMD_SKIP_INSTALL": "OFF", "XSIMD_VERSION_MAJOR": "14", "XSIMD_VERSION_MINOR": "2",
        "XSIMD_VERSION_PATCH": "0",
    }
    cache = case.put(binary / "CMakeCache.txt", "".join(
        key + ":STRING=" + value + "\n" for key, value in cache_values.items()).encode())
    declaration_path = case.arrow / "cpp/cmake_modules/ThirdpartyToolchain.cmake"
    declaration = case.put(declaration_path, declaration_path.read_bytes() + XSIMD_ARROW25_DECLARATION)
    versions_path = case.arrow / "cpp/thirdparty/versions.txt"
    versions = case.put(versions_path, versions_path.read_bytes() + (
        "ARROW_XSIMD_BUILD_VERSION=14.2.0\nARROW_XSIMD_BUILD_SHA256_CHECKSUM=" + archive["sha256"] + "\n").encode())
    case.manifest["source_archive"] = case.bundle(Path(case.manifest["source_archive"]["path"]), case.arrow, {
        "NOTICE.txt": b"synthetic Arrow upstream notice\n",
        "cpp/cmake_modules/ThirdpartyToolchain.cmake": Path(declaration["path"]).read_bytes(),
        "cpp/thirdparty/versions.txt": Path(versions["path"]).read_bytes(),
    })
    case.targets.extend([
        {"target": "xsimd_ep", "kind": "UTILITY", "source": str(case.arrow / "cpp"),
         "binary": str(case.cpp), "imported": False, "artifacts": []},
        {"target": "arrow::xsimd", "kind": "INTERFACE_LIBRARY", "source": str(case.arrow / "cpp"),
         "binary": str(case.cpp), "imported": True, "artifacts": []},
    ])
    consumer = case.put(case.arrow / "cpp/src/arrow/util/cpu_info.cc", b"#include <xsimd/xsimd.hpp>\n")
    unity = case.put(case.cpp / "src/arrow/CMakeFiles/arrow_util.dir/Unity/unity_0_cxx.cxx",
                     ('#include "' + consumer["path"] + '"\n').encode())
    objects = []
    for index, selected_source in enumerate((consumer, unity)):
        obj = case.put(case.cpp / ("src/arrow/CMakeFiles/arrow_util.dir/xsimd_consumer" + str(index) + ".o"),
                       b"\x7fELF synthetic Arrow xsimd header consumer " + str(index).encode())
        objects.append(obj)
        case.compile_commands.append({"directory": str(case.cpp), "file": selected_source["path"],
            "command": case.toolchain["cxx"]["path"] + " -isystem " + str(include) + " -fPIC -c "
                       + selected_source["path"] + " -o " + obj["path"], "output": obj["path"]})
        case.ninja_commands.append("build " + str(Path(obj["path"]).relative_to(case.cpp))
            + ": CXX_COMPILER__arrow_util " + selected_source["path"] + "\n")
    case.main_compile = case.json(case.cpp / "compile_commands.json", case.compile_commands)
    case.main_graph = case.put(case.cpp / "build.ninja", "".join(case.ninja_commands).encode())
    case.xsimd = {"source": source, "binary": binary, "prefix": prefix, "include": include,
                   "archive": archive, "installed": installed, "unused_metadata": unused_metadata,
                   "cfg": cfg, "command": command, "download": download, "cache": cache,
                   "declaration": declaration, "versions": versions, "consumer": consumer,
                   "unity": unity, "objects": objects}


@pytest.mark.parametrize("retained", [False, True], ids=["original-files", "retained-forbid-history"])
def test_arrow25_xsimd_header_ep_binds_original_source_installed_headers_and_real_consumers(
        verifier, actual_flat_arrow25_case, tmp_path, monkeypatch, retained):
    case, project = actual_flat_arrow25_case, actual_flat_arrow25_case.xsimd
    attestation = case.publish(verifier, tmp_path / "arrow25-xsimd-attestation.json")
    initial = verifier.verify_completed_attestation(attestation["path"], attestation["sha256"])
    proof = initial["attestation"]["external_projects"]["xsimd"]
    assert proof["kind"] == "header_only_external_project" and proof["target"] == "xsimd_ep"
    assert proof["archive"] == project["archive"] and proof["source_root"] == str(project["source"])
    assert [row for row in proof["files"] if row["type"] == "file"] == case.inventory(project["source"])
    assert proof["installed_include_root"] == str(project["include"])
    installed_files = {row["relative"]: row for row in proof["installed_files"] if row["type"] == "file"}
    assert set(installed_files) == {str(Path(row["path"]).relative_to(project["include"]))
                                    for row in project["installed"]}
    assert proof["objects"] == [] and proof["archives"] == []
    assert {row["object"]["path"] for row in proof["consumers"]} == {row["path"] for row in project["objects"]}
    assert all(row["pic"] and row["command_record"] == case.main_compile for row in proof["consumers"])
    assert any(project["consumer"] in row.get("included_sources", []) for row in proof["consumers"])
    assert all(row["rule"] == "xsimd-installed-header-copy-v1" for row in proof["generated_outputs"])
    for reference in (project["archive"], project["cfg"], project["command"], project["download"],
                      project["cache"], project["declaration"], project["versions"],
                      project["consumer"], project["unity"], *project["installed"], *project["objects"]):
        assert reference in initial["files"]
    assert project["unused_metadata"] not in initial["files"]
    assert set(initial["supplemental"]) == set(SDK_NAMES)
    if retained:
        replay_completed_source_only_from_copies(case, attestation, initial, verifier, tmp_path, monkeypatch)


@pytest.mark.parametrize("change", ["missing_target", "foreign_target_source", "wrong_interface_type", "extra_ep",
                                    "archive_pin", "archive_owner", "source_bytes", "extra_source", "declaration",
                                    "version_pin", "cache_prefix", "cache_compiler", "configure_prefix",
                                    "installed_bytes", "missing_header", "extra_header", "header_symlink",
                                    "missing_include", "foreign_include", "missing_object", "foreign_consumer"])
def test_arrow25_xsimd_header_ep_rejects_unbound_sources_headers_configuration_and_consumers(
        verifier, actual_flat_arrow25_case, change):
    case, project = actual_flat_arrow25_case, actual_flat_arrow25_case.xsimd
    if change == "missing_target":
        case.targets = [row for row in case.targets if row["target"] != "xsimd_ep"]
    elif change == "foreign_target_source":
        next(row for row in case.targets if row["target"] == "xsimd_ep")["source"] = str(case.root / "foreign")
    elif change == "wrong_interface_type":
        next(row for row in case.targets if row["target"] == "arrow::xsimd")["kind"] = "STATIC_LIBRARY"
    elif change == "extra_ep":
        case.targets.append({"target": "unqualified_ep", "kind": "UTILITY"})
    elif change == "archive_pin":
        next(row for row in case.archives if row["path"] == project["archive"]["path"])["sha256"] = "0" * 64
    elif change == "archive_owner":
        next(row for row in case.archives if row["path"] == project["archive"]["path"])["path"] = str(case.root / "14.2.0.tar.gz")
    elif change in {"source_bytes", "extra_source"}:
        case.put(project["source"] / ("include/xsimd/xsimd.hpp" if change == "source_bytes" else "unowned.h"), b"changed source\n")
    elif change == "declaration":
        path = Path(project["declaration"]["path"])
        case.put(path, path.read_bytes().replace(b'PREFIX "${CMAKE_BINARY_DIR}"', b'PREFIX "foreign"'))
    elif change == "version_pin":
        path = Path(project["versions"]["path"])
        case.put(path, path.read_bytes().replace(b"ARROW_XSIMD_BUILD_VERSION=14.2.0", b"ARROW_XSIMD_BUILD_VERSION=14.3.0"))
    elif change in {"cache_prefix", "cache_compiler"}:
        path = Path(project["cache"]["path"])
        before = str(project["prefix"]) if change == "cache_prefix" else case.toolchain["cxx"]["path"]
        case.put(path, path.read_text().replace(before, str(case.root / "foreign")).encode())
    elif change == "configure_prefix":
        for reference in (project["cfg"], project["command"]):
            path = Path(reference["path"])
            case.put(path, path.read_text().replace(str(project["prefix"]), str(case.root / "foreign")).encode())
    elif change in {"installed_bytes", "missing_header", "header_symlink"}:
        path = Path(project["installed"][0]["path"])
        if change == "installed_bytes":
            case.put(path, b"changed installed header\n")
        else:
            path.unlink()
            if change == "header_symlink":
                path.symlink_to("config/xsimd_config.hpp")
    elif change == "extra_header":
        case.put(project["include"] / "xsimd/unowned.hpp", b"unowned installed header\n")
    elif change in {"missing_include", "foreign_include", "foreign_consumer"}:
        for row in case.compile_commands:
            if str(project["include"]) not in row["command"]:
                continue
            if change == "missing_include":
                row["command"] = row["command"].replace(" -isystem " + str(project["include"]), "")
            elif change == "foreign_include":
                row["command"] = row["command"].replace(str(project["include"]), str(case.root / "foreign/include"))
            else:
                before = row["file"]
                row["file"] = case.put(case.root / "foreign/consumer.cc", b"#include <xsimd/xsimd.hpp>\n")["path"]
                row["command"] = row["command"].replace(before, row["file"])
        case.main_compile = case.json(case.cpp / "compile_commands.json", case.compile_commands)
    elif change == "missing_object":
        Path(project["objects"][0]["path"]).unlink()
    case.refresh()
    with pytest.raises((ValueError, FileNotFoundError), match="xsimd|ExternalProject|source|header|configur|compiler|archive|include|object|target|tree|layout|file"):
        case.capture(verifier)


def test_arrow26_rejects_arrow25_only_xsimd_external_project(verifier, tmp_path, monkeypatch):
    case = CompletedSourceCase(tmp_path, "26", verifier, monkeypatch)
    add_arrow25_header_only_xsimd(case, verifier, monkeypatch)
    case.refresh()
    with pytest.raises(ValueError, match="ExternalProject|unqualified"):
        case.capture(verifier)


def use_native_lf_lz4_package_configuration(case, verifier, monkeypatch, *, template_final_lf=False, change=None):
    """Pinned CMake PACKAGE_INIT begins LF and configure_file terminates each line."""
    project, _, _, _, _, _ = use_remaining_source_owned_lz4_metadata(case, verifier, monkeypatch)
    old = project["archive"]
    with tarfile.open(old["path"]) as archive:
        contents = {member.name.split("/", 1)[1]: archive.extractfile(member).read()
                    for member in archive.getmembers() if member.isfile()}
    relative = "build/cmake/lz4Config.cmake.in"
    template = b'@PACKAGE_INIT@\ninclude( "${CMAKE_CURRENT_LIST_DIR}/lz4Targets.cmake" )'
    if template_final_lf:
        template += b"\n"
    contents[relative] = template
    archive = case.bundle(Path(old["path"]), project["source"], contents)
    project["archive"] = archive
    case.archives = [{**row, "sha256": archive["sha256"]} if row["path"] == archive["path"] else row
                     for row in case.archives]
    monkeypatch.setitem(verifier.EP_ARCHIVE_SHA256, "lz4", archive["sha256"])
    owner = case.cpp / "_deps/orc-src/cmake_modules/ThirdpartyToolchain.cmake"
    for path in (Path(project["urlinfo"]["path"]), owner,
                 case.arrow / "cpp/cmake_modules/ThirdpartyToolchain.cmake",
                 case.arrow / "cpp/thirdparty/versions.txt"):
        path.write_text(path.read_text().replace(old["sha256"], archive["sha256"]))
    expansion = (b"\n####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######\n"
                 b"# configured by the pinned CMake module\n")
    before, after = template.split(b"@PACKAGE_INIT@")
    expected = before + expansion + after
    if not expected.endswith(b"\n"):
        expected += b"\n"
    data = expected
    if change == "missing_leading_lf":
        data = data.removeprefix(b"\n")
    elif change == "extra_leading_lf":
        data = b"\n" + data
    elif change == "missing_final_lf":
        data = data.removesuffix(b"\n")
    elif change == "extra_final_lf":
        data += b"\n"
    elif change == "changed_suffix":
        data = data.replace(b"lz4Targets.cmake", b"foreignTargets.cmake")
    elif change == "changed_template":
        case.put(project["source"] / relative, template.replace(b"lz4Targets.cmake", b"foreignTargets.cmake"))
    output = case.put(project["binary"] / "lz4Config.cmake", data)
    refresh_lz4_owner(case, owner)
    case.raw["toolchain_versions"] = {"cmake": "cmake version 4.2.3"}
    case.raw_ref = case.json(case.work / "build-receipt.json", case.raw)
    return project, record(project["source"] / relative), output, expected


@pytest.mark.parametrize("template_final_lf", [False, True], ids=["actual-no-template-eof-lf", "template-has-eof-lf"])
@pytest.mark.parametrize("retained", [False, True], ids=["original-files", "retained-forbid-history"])
def test_lz4_package_config_uses_pinned_module_leading_lf_and_native_configure_file_final_lf(
        verifier, completed_source_case, tmp_path, monkeypatch, template_final_lf, retained):
    case = completed_source_case
    project, template, output, expected = use_native_lf_lz4_package_configuration(
        case, verifier, monkeypatch, template_final_lf=template_final_lf)
    assert Path(output["path"]).read_bytes() == expected
    assert expected.startswith(b"\n####### Expanded") and expected.endswith(b"\n")
    attestation = case.publish(verifier, tmp_path / "lz4-native-configure-file-lf.json")
    initial = verifier.verify_completed_attestation(attestation["path"], attestation["sha256"])
    generated = {row["output"]["path"]: row for row in initial["attestation"]["external_projects"]["lz4"]["generated_outputs"]}
    proof = generated[output["path"]]
    assert proof["rule"] == "lz4-source-metadata-v1" and template in proof["inputs"]
    assert record(project["binary"] / "CMakeCache.txt") in proof["inputs"]
    assert output in initial["files"]
    if retained:
        replay_completed_source_only_from_copies(case, attestation, initial, verifier, tmp_path, monkeypatch)


@pytest.mark.parametrize("change", ["missing_leading_lf", "extra_leading_lf", "missing_final_lf",
                                    "extra_final_lf", "changed_suffix", "changed_template"])
def test_lz4_package_config_rejects_non_native_newline_prefix_suffix_or_changed_source_template(
        verifier, completed_source_case, monkeypatch, change):
    case = completed_source_case
    use_native_lf_lz4_package_configuration(case, verifier, monkeypatch, change=change)
    with pytest.raises(ValueError, match="template|source|package|LZ4"):
        case.capture(verifier)
