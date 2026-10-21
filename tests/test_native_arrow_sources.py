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
        self.source_ref = self.json(self.work / "compiled-sources.json", {
            "files": self.inventory(self.arrow), "compile_commands": self.main_compile,
            "translation_units": [], "integrated_header_consumers": []})
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
    assert set(doc["external_projects"]) == set(EP_NAMES)
    assert {row["archive"]["sha256"] for row in doc["external_projects"].values()} == {
        row["archive"]["sha256"] for row in case.ep.values()}
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
    case.raw["build"]["context"] = {**case.manifest["context"], "work": str(case.root / "other-build")}
    case.raw_ref = case.json(case.work / "build-receipt.json", case.raw)
    with pytest.raises(ValueError, match="context|work|build"):
        case.capture(verifier)


def test_completed_cli_never_publishes_in_manifest_work_with_inconsistent_work_context(
        verifier, completed_source_case):
    case = completed_source_case
    case.raw["build"]["context"] = {**case.manifest["context"], "work": str(case.root / "other-build")}
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
