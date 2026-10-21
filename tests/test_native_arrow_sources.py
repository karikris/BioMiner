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
    for name in ("open", "stat", "lstat", "readlink", "iterdir"):
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
