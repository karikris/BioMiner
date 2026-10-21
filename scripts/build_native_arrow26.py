#!/usr/bin/env python3
"""Build one pinned, host-specific Arrow 26 candidate; never admit a runtime.

First use --configure-only. Review the actual dependency selection, then resume
with --configured-inputs-sha256. Compilation receipts remain unqualified. This
independent recipe deliberately leaves the retained Arrow 25 fallback unchanged.
"""
from __future__ import annotations

import argparse
import base64
import email
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import sys
import tarfile
import zipfile

ARROW_COMMIT = "446167169a0dc547e00e1bc29f417a40c0ca267a"
ARROW_ARCHIVE_SHA256 = "764adbb5601a4d0d9d67a10cf04892d33cfac9fd43012f08a2d7c9956a7a90d2"
INTEGRATED_SOURCES = {
    "cpp/src/arrow/compute/kernels/copy_data_internal.h": "21b1f8acce63e0ca301a968339d731eddb5efd8e8eeb1a55638ffe665a046621",
    "cpp/src/arrow/compute/kernels/vector_replace.cc": "12cd5bd3ab8e487aba96743c92cea9f234e3efba279f1200126029f84717d9c1",
    "cpp/src/arrow/compute/kernels/vector_replace_test.cc": "c087392bbd8114e903608b36c1e4e88a087677cbcfda484cb270d227b385788e",
    "cpp/src/arrow/compute/kernels/scalar_if_else.cc": "87f5ce8b675c9dcbaf1f45b25e2515a161fe3bf4b1b567acba0ecea287510fa8",
}
FEATURES = ("ACERO", "AZURE", "COMPUTE", "CSV", "DATASET", "FILESYSTEM", "FLIGHT", "GCS", "HDFS", "JSON", "ORC", "PARQUET", "S3", "SUBSTRAIT")
CODECS = ("BROTLI", "BZ2", "LZ4", "SNAPPY", "ZLIB", "ZSTD")
PYTHON_FEATURES = ("ACERO", "AZURE", "DATASET", "FLIGHT", "GCS", "HDFS", "ORC", "PARQUET", "S3", "SUBSTRAIT", "PARQUET_ENCRYPTION")
REQUIRED_ARROW_LIBRARIES = tuple(f"lib{name}.so.2600" for name in (
    "arrow", "arrow_compute", "arrow_acero", "arrow_dataset", "arrow_flight",
    "arrow_s3", "arrow_substrait", "parquet"))
BUILDER_PINS = {
    "scikit-build-core": "1.1.1", "cython": "3.3.0", "numpy": "2.5.4", "libcst": "1.9.0",
    "setuptools-scm": "10.3.4", "setuptools": "84.2.1", "build": "1.6.1",
    "packaging": "26.3", "pyproject-hooks": "1.3.3", "pathspec": "1.1.1",
    "pyyaml": "6.0.3", "vcs-versioning": "2.6.0",
}
DEPENDENCIES = {
    "simdjson": {"version": "4.6.11", "commit": "f5de14f09256982933af2849beb43778bd421ca7",
                 "sha256": "9157ef4acb368d727873b4dd592f7b1d67d19fcb9201e8bc5a6b093e54e5171a",
                 "filename": "simdjson-f5de14f09256982933af2849beb43778bd421ca7.tar.gz", "env": "ARROW_SIMDJSON_URL"},
    "uriparser": {"version": "1.0.2", "commit": "9b2bed92f5deecf740819f9bf27724bee2fe9c12",
                  "sha256": "dd2e4843f43de6f5aedf430e530f1e2d159eba8ca4d64c2787af1c20f707a9e4",
                  "filename": "uriparser-1.0.2.tar.bz2", "env": "ARROW_URIPARSER_URL"},
    "boost": {"version": "1.92.0", "sha256": "f51707c27359a0df0cac1beada86de31bb5eed5e8285592dadec384df99c2984",
              "filename": "boost-1.92.0-cmake.tar.gz", "env": "ARROW_BOOST_URL"},
    "zlib": {"version": "1.3.2", "sha256": "bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16",
             "filename": "zlib-1.3.2.tar.gz", "env": "ARROW_ZLIB_URL"},
}
SOURCE_POLICY = "arrow26-integrated-private-network-v2"
# Actual configuration remains closed until the reviewed PIC-provider helper
# has a committed source pin. Synthetic tests substitute their exact helper.
NETWORK_HELPER_SHA256 = "a8cb0355e3abfa4415093268c17678d63ea1e745a48f4a5dd750a6a01d037414"
NETWORK_ARCHIVES = ("libcurl.a", "libxml2.a", "libiconv.a", "libcharset.a", "libz.a", "libnghttp2.a")
NETWORK_HEADERS = ("curl/curl.h", "libxml2/libxml/parser.h", "iconv.h", "zlib.h", "nghttp2/nghttp2.h")
NETWORK_COMPONENTS = {
    "libcurl": ("curl", "8.23.0"), "libxml2": ("libxml2", "2.15.4"),
    "libiconv": ("libiconv", "1.19"), "libcharset": ("libiconv", "1.19"),
    "zlib": ("zlib", "1.3.2"), "libnghttp2": ("nghttp2", "1.68.0-2ubuntu0.2"),
}
TOOLCHAIN_PATCHES = (
    ("list(APPEND AWSSDK_PRODUCTS aws-lc s2n-tls)",
     "list(APPEND AWSSDK_PRODUCTS s2n-tls)\n    add_library(crypto ALIAS OpenSSL::Crypto)"),
    ("  fetchcontent_makeavailable(cares)",
     "  fetchcontent_makeavailable(cares)\n  set_target_properties(c-ares PROPERTIES UNITY_BUILD OFF)"),
    ("  fetchcontent_makeavailable(grpc)",
     "  set(CMAKE_UNITY_BUILD OFF)\n  fetchcontent_makeavailable(grpc)"),
    ('set(MAKE_BUILD_ARGS "-j${NPROC}")', 'set(MAKE_BUILD_ARGS "-j1")'),
)
VERSION_PATCHES = (
    ("ARROW_SIMDJSON_BUILD_VERSION=v4.6.4", "ARROW_SIMDJSON_BUILD_VERSION=v4.6.11"),
    ("ARROW_SIMDJSON_BUILD_SHA256_CHECKSUM=b091107844fe928158c5c2265c20360fff312889ddf7ebc4528a0f0f8f2ff9cd",
     "ARROW_SIMDJSON_BUILD_SHA256_CHECKSUM=" + DEPENDENCIES["simdjson"]["sha256"]),
)
BOOST_SELECTION_ANCHOR = "  fetchcontent_makeavailable(boost)"

def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path.absolute()), "sha256": sha256(path)}


def canonical_json(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def check_record(record: dict, label: str = "input") -> Path:
    path = Path(record["path"])
    if not path.is_absolute() or not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"{label} checksum mismatch: {path}")
    return path


def preserve(path: Path, data: bytes, *, resume: bool, label: str) -> dict:
    if resume:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"{label} changed since preparation")
    else:
        with path.open("xb") as stream:
            stream.write(data)
    return file_record(path)


def preserve_recipe(work: Path, source_bytes: bytes, *, resume: bool) -> dict:
    return preserve(work / "build-recipe.py", source_bytes, resume=resume, label="recipe")


def preserve_build_inputs(work: Path, inputs: dict, *, resume: bool) -> tuple:
    record = preserve(work / "build-inputs.json", canonical_json(inputs), resume=resume, label="build inputs")
    return record, "26.0.0+biominer." + record["sha256"][:12]


def archive_members(bundle: tarfile.TarFile) -> tuple:
    members, root, names = bundle.getmembers(), None, set()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts or member.name in names:
            raise ValueError("unsafe or duplicate archive path")
        names.add(member.name)
        root = root or path.parts[0]
        if path.parts[0] != root or not (member.isfile() or member.isdir() or member.issym()):
            raise ValueError("unsupported archive layout or member type")
        if member.issym():
            target = PurePosixPath(member.linkname)
            if target.is_absolute():
                raise ValueError("absolute archive symlink")
            normalized = os.path.normpath(str(path.parent / target))
            if normalized != root and not normalized.startswith(root + "/"):
                raise ValueError("escaping archive symlink")
    symlinks = {PurePosixPath(m.name) for m in members if m.issym()}
    if any(any(parent in symlinks for parent in PurePosixPath(m.name).parents) for m in members):
        raise ValueError("archive member traverses a symlink")
    if root is None:
        raise ValueError("empty source archive")
    return root, members


def tree_paths(root: Path) -> dict:
    result = {}
    def unreadable(error):
        raise ValueError(f"unreadable source/build tree: {error.filename}") from error
    for directory, folders, files in os.walk(root, followlinks=False, onerror=unreadable):
        for name in folders + files:
            path = Path(directory) / name
            result[path.relative_to(root).as_posix()] = path
    return result


def verify_archive_tree(bundle: tarfile.TarFile, source: Path, replacements: dict | None = None,
                        generated_files: dict | None = None) -> list:
    archive_root, members = archive_members(bundle)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("source directory missing or symlink")
    expected, replacements = {}, replacements or {}
    for member in members:
        relative = PurePosixPath(member.name).relative_to(archive_root)
        if str(relative) == ".":
            continue
        expected[str(relative)] = member
        for parent in relative.parents:
            if str(parent) != ".":
                expected.setdefault(str(parent), None)
    actual = tree_paths(source)
    generated_files = generated_files or {}
    if set(generated_files) & set(expected) or set(actual) != set(expected) | set(generated_files):
        raise ValueError("source layout changed since extraction")
    evidence = []
    for relative, member in sorted(expected.items()):
        path = actual[relative]
        if member is not None and member.issym():
            if not path.is_symlink() or os.readlink(path) != member.linkname:
                raise ValueError(f"source symlink layout changed: {relative}")
            evidence.append({"relative": relative, "type": "symlink", "target": member.linkname})
        elif member is None or member.isdir():
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"source directory layout changed: {relative}")
        else:
            stream = bundle.extractfile(member)
            original = stream.read()
            expected_bytes = replacements.get(relative, original)
            if path.is_symlink() or not path.is_file() or sha256(path) != hashlib.sha256(expected_bytes).hexdigest():
                raise ValueError(f"source changed since extraction: {relative}")
            evidence.append({"relative": relative, "type": "file", "sha256": sha256(path)})
    for relative, data in sorted(generated_files.items()):
        path = actual[relative]
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"generated source differs from frozen builder template: {relative}")
        evidence.append({"relative": relative, "type": "generated", "sha256": sha256(path)})
    return evidence


def exact_replace(data: bytes, before: str, after: str) -> bytes:
    if data.count(before.encode()) != 1:
        raise ValueError("Arrow source transformation anchor must occur exactly once")
    return data.replace(before.encode(), after.encode())


def source_transformations(bundle: tarfile.TarFile, root: str) -> dict:
    toolchain_path = "cpp/cmake_modules/ThirdpartyToolchain.cmake"
    toolchain = bundle.extractfile(root + "/" + toolchain_path).read()
    for before, after in TOOLCHAIN_PATCHES:
        toolchain = exact_replace(toolchain, before, after)
    toolchain = exact_replace(toolchain, BOOST_SELECTION_ANCHOR, BOOST_SELECTION_ANCHOR +
        '\n  file(WRITE "${CMAKE_BINARY_DIR}/biominer-boost-selection.txt" "${BOOST_INCLUDE_LIBRARIES}\\n")')
    versions_path = "cpp/thirdparty/versions.txt"
    versions = bundle.extractfile(root + "/" + versions_path).read()
    for before, after in VERSION_PATCHES:
        versions = exact_replace(versions, before, after)
    return {toolchain_path: toolchain, versions_path: versions}


def prepare_source(archive: Path, work: Path, *, resume: bool) -> tuple:
    if sha256(archive) != ARROW_ARCHIVE_SHA256:
        raise ValueError("Arrow source archive checksum mismatch")
    work = work.absolute()
    if any(path.is_symlink() for path in (work, *work.parents)):
        raise ValueError("work path contains a symlink")
    if work.exists() and (not work.is_dir() or (not resume and any(work.iterdir()))):
        raise ValueError("work directory already exists and is not empty")
    if resume:
        for name in ("cpp-build", "python-build", "arrow-prefix", "wheelhouse", "wheel-inventory",
                     "inputs", "build-receipt.json", "source-verification.json", "configured-inputs.json",
                     "resume-state.json", "logs", "private-network-helper.py", "private-network.cmake"):
            if (work / name).is_symlink():
                raise ValueError(f"resume output path is a symlink: {name}")
        receipt = work / "build-receipt.json"
        if receipt.exists():
            previous = json.loads(receipt.read_bytes())
            if previous.get("status") not in {"building", "failed", "configured_unverified"}:
                raise ValueError("resume cannot modify completed or unrecognized receipt")
        if list((work / "wheelhouse").glob("*.whl")):
            raise ValueError("resume cannot modify completed wheel output")
    source = work / f"arrow-{ARROW_COMMIT}"
    with tarfile.open(archive) as bundle:
        root, _ = archive_members(bundle)
        if root != source.name:
            raise ValueError("Arrow source archive root mismatch")
        transformed = source_transformations(bundle, root)
        for relative, expected in INTEGRATED_SOURCES.items():
            if hashlib.sha256(bundle.extractfile(root + "/" + relative).read()).hexdigest() != expected:
                raise ValueError(f"integrated Arrow source checksum mismatch: {relative}")
        if not resume:
            work.mkdir(parents=True, exist_ok=True)
            bundle.extractall(work, filter="data")
            verify_archive_tree(bundle, source)
            for relative, data in transformed.items():
                (source / relative).write_bytes(data)
        inventory = verify_archive_tree(bundle, source, transformed)
        evidence = {
            "schema_version": 1, "source_revision": ARROW_COMMIT, "archive": file_record(archive),
            "integrated_sources": [{"relative": name, "sha256": value} for name, value in sorted(INTEGRATED_SOURCES.items())],
            "transformations": [{"relative": name,
                "original_sha256": hashlib.sha256(bundle.extractfile(root + "/" + name).read()).hexdigest(),
                "sha256": hashlib.sha256(data).hexdigest()} for name, data in transformed.items()],
            "files": inventory,
        }
    preserve(work / "source-verification.json", canonical_json(evidence), resume=resume, label="source receipt")
    return source, evidence


def validate_builder(receipt_path: Path, receipt_sha256: str, *, python: str, cc: str, cxx: str) -> dict:
    check_record({"path": str(receipt_path), "sha256": receipt_sha256}, "builder receipt")
    receipt = json.loads(receipt_path.read_bytes())
    if (receipt.get("schema_version"), receipt.get("kind"), receipt.get("status")) != (
            1, "arrow26_python_builder", "frozen_verified"):
        raise ValueError("unapproved builder receipt")
    if python != receipt["python"]["path"]:
        raise ValueError("builder Python path differs from approved receipt")
    for name, selected in (("cc", cc), ("cxx", cxx)):
        if selected != receipt["toolchain"][name]["path"]:
            raise ValueError(f"compiler path differs from builder receipt: {name}")
    for name in ("python", "base_python", "requirements", "inventory", "verification"):
        check_record(receipt[name], "builder " + name)
    for name in ("cc", "cxx", "cmake", "ninja", "ld"):
        check_record(receipt["toolchain"][name], "builder toolchain " + name)
    inventory = json.loads(Path(receipt["inventory"]["path"]).read_bytes())
    root = Path(inventory["root"])
    if inventory.get("schema_version") != 1 or not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("invalid builder inventory root")
    if Path(python) != root / "bin/python" or any(parent.is_symlink() for parent in root.parents):
        raise ValueError("builder Python is outside the approved inventory root")
    rows = {}
    for kind in ("files", "directories", "symlinks"):
        for row in inventory[kind]:
            relative = PurePosixPath(row["relative"])
            if relative.is_absolute() or ".." in relative.parts or str(relative) in rows:
                raise ValueError("invalid builder inventory path")
            rows[str(relative)] = (kind, row)
    actual = tree_paths(root) | {".": root}
    if set(actual) != set(rows):
        raise ValueError("builder inventory layout mismatch")
    base_python = Path(receipt["base_python"]["path"]).resolve()
    if Path(python).resolve() != base_python:
        raise ValueError("builder Python symlink does not resolve to approved base executable")
    for relative, (kind, row) in rows.items():
        path = actual[relative]
        if kind == "symlinks":
            if not path.is_symlink() or os.readlink(path) != row["target"]:
                raise ValueError(f"builder symlink changed: {relative}")
            resolved = path.resolve()
            if not resolved.is_relative_to(root) and not (
                    relative.startswith("bin/python") and resolved == base_python):
                raise ValueError(f"builder symlink escapes approved environment: {relative}")
        else:
            mode = stat.S_IMODE(path.lstat().st_mode)
            if path.is_symlink() or mode != row["mode"] or mode & 0o222:
                raise ValueError(f"builder inventory mode changed or writable: {relative}")
            if kind == "files":
                if not path.is_file() or sha256(path) != row["sha256"]:
                    raise ValueError(f"builder inventory file changed: {relative}")
            elif not path.is_dir():
                raise ValueError(f"builder inventory directory changed: {relative}")
    distributions = receipt["distributions"]
    actual_pins = {row["name"]: row["version"] for row in distributions}
    if actual_pins != BUILDER_PINS or len(distributions) != len(BUILDER_PINS):
        raise ValueError("builder distributions do not match exact approved pins")
    for row in distributions:
        for field in ("wheel", "metadata", "record"):
            path = check_record(row[field], "builder distribution " + field)
            if field != "wheel" and not path.resolve().is_relative_to(root):
                raise ValueError("builder distribution metadata outside frozen inventory")
        metadata = email.message_from_bytes(Path(row["metadata"]["path"]).read_bytes())
        name = re.sub(r"[-_.]+", "-", metadata["Name"].lower())
        if name != row["name"] or metadata["Version"] != row["version"]:
            raise ValueError("builder distribution metadata identity mismatch")
    return {**receipt, "receipt": file_record(receipt_path)}


def probe_builder(builder: dict) -> dict:
    probe = (
        "import sys,sysconfig,ssl,importlib.metadata as m,json; "
        "print(json.dumps({'executable':sys.executable,'base_executable':sys._base_executable,"
        "'version':list(sys.version_info[:3]),'implementation':sys.implementation.name,"
        "'gil':sys._is_gil_enabled(),'platform':sysconfig.get_platform(),'openssl':ssl.OPENSSL_VERSION,"
        "'distributions':[{'name':d.metadata['Name'],'version':d.version,'metadata':str(d._path/'METADATA')}"
        " for d in m.distributions(path=list(set([sysconfig.get_path('purelib'),sysconfig.get_path('platlib')])))]},sort_keys=True))")
    observed = json.loads(subprocess.check_output([builder["python"]["path"], "-I", "-B", "-c", probe], text=True))
    pins = {re.sub(r"[-_.]+", "-", d["name"].lower()): d["version"] for d in observed["distributions"]}
    if (observed["version"] != [3, 14, 7] or observed["implementation"] != "cpython"
            or not observed["gil"] or observed["platform"] != "linux-x86_64"
            or observed["executable"] != builder["python"]["path"]
            or Path(observed["base_executable"]).resolve() != Path(builder["base_python"]["path"]).resolve()
            or pins != BUILDER_PINS or len(observed["distributions"]) != len(BUILDER_PINS)):
        raise ValueError("observed builder Python or distribution identity mismatch")
    expected_metadata = {row["metadata"]["path"] for row in builder["distributions"]}
    if {row["metadata"] for row in observed["distributions"]} != expected_metadata:
        raise ValueError("observed builder distributions came from unexpected paths")
    return observed


def validate_network(receipt_path: Path, receipt_sha256: str, helper_path: Path, *, openssl_prefix: Path) -> dict:
    """Load only approved helper source and retain its complete verified closure."""
    helper_path, receipt_path = Path(helper_path).absolute(), Path(receipt_path).absolute()
    if any(path.is_symlink() for path in (helper_path, *helper_path.parents)):
        raise ValueError("network helper checksum path contains a symlink")
    helper_bytes = helper_path.read_bytes()
    helper_hash = hashlib.sha256(helper_bytes).hexdigest()
    if not isinstance(NETWORK_HELPER_SHA256, str) or helper_hash != NETWORK_HELPER_SHA256:
        raise ValueError("network helper checksum mismatch or committed helper pin unavailable")
    reference = {"path": str(receipt_path), "sha256": receipt_sha256}
    check_record(reference, "network receipt")
    namespace = {"__name__": "biominer_private_network_helper", "__file__": str(helper_path)}
    # Execute exactly the bytes just hashed, never a separately resolved module
    # or a timestamp-valid pyc from the mutable source directory.
    exec(compile(helper_bytes, str(helper_path), "exec"), namespace)
    verified = namespace["verify_completed_prefix"](receipt_path, receipt_sha256)
    if (set(verified) != {"receipt", "prefix", "build", "files", "private_files"}
            or verified["receipt"] != reference):
        raise ValueError("network descriptor does not bind the approved receipt")
    raw, prefix = verified["build"], Path(verified["prefix"])
    if raw != json.loads(receipt_path.read_bytes()):
        raise ValueError("network descriptor raw build differs from the approved receipt")
    if (raw.get("schema_version") != 1 or raw.get("kind") != "private_network_static_build"
            or raw.get("status") != "compiled_unverified" or raw.get("ready") is not False
            or raw.get("prefix") != str(prefix) or not prefix.is_absolute()
            or prefix != receipt_path.parent / "prefix" or prefix.resolve() != prefix):
        raise ValueError("network descriptor has an unapproved private prefix or build identity")
    if raw.get("openssl", {}).get("prefix") != str(Path(openssl_prefix).absolute()):
        raise ValueError("network descriptor selected different private OpenSSL inputs")
    files = {}
    for row in verified["files"]:
        path = check_record(row, "network closure")
        if str(path) in files:
            raise ValueError("network closure has duplicate file identities")
        files[str(path)] = row["sha256"]
    if files.get(str(receipt_path)) != receipt_sha256:
        raise ValueError("network closure omitted the approved receipt")
    for name, metadata in verified["private_files"].items():
        path = Path(name)
        identity = (metadata.get("source_package"), metadata.get("source_version"))
        if (not path.is_absolute() or not path.is_relative_to(prefix) or path.resolve() != path
                or metadata.get("component") not in NETWORK_COMPONENTS
                or identity != NETWORK_COMPONENTS[metadata["component"]]
                or files.get(name) != metadata.get("sha256")):
            raise ValueError("network private file lacks exact component/source/closure ownership")
    required = [prefix / "lib" / name for name in NETWORK_ARCHIVES]
    required += [prefix / "include" / name for name in NETWORK_HEADERS]
    if any(str(path) not in verified["private_files"] for path in required):
        raise ValueError("network closure lacks a required private archive or header")
    return {**verified, "helper": {"path": str(helper_path), "sha256": helper_hash}}


def verify_network(network: dict) -> None:
    check_record(network["helper"], "network helper")
    current = validate_network(Path(network["receipt"]["path"]), network["receipt"]["sha256"],
                               Path(network["helper"]["path"]),
                               openssl_prefix=Path(network["build"]["openssl"]["prefix"]))
    if current != network:
        raise ValueError("network descriptor or complete source/provider closure changed")


def private_network_cmake(prefix: Path, openssl: Path) -> bytes:
    """Bind global static imports before upstream find modules can omit links."""
    for path in (prefix, openssl):
        if not path.is_absolute() or any(character in str(path) for character in '\t\n\r;"$\\{}'):
            raise ValueError("network/OpenSSL path contains unsupported CMake characters")
    targets = (
        ("OpenSSL::Crypto", openssl / "lib/libcrypto.a", openssl / "include", "Threads::Threads;${CMAKE_DL_LIBS}"),
        ("OpenSSL::SSL", openssl / "lib/libssl.a", openssl / "include", "OpenSSL::Crypto"),
        ("ZLIB::ZLIB", prefix / "lib/libz.a", prefix / "include", ""),
        ("biominer_nghttp2", prefix / "lib/libnghttp2.a", prefix / "include", ""),
        ("biominer_charset", prefix / "lib/libcharset.a", prefix / "include", ""),
        ("Iconv::Iconv", prefix / "lib/libiconv.a", prefix / "include", ""),
        ("LibXml2::LibXml2", prefix / "lib/libxml2.a", prefix / "include/libxml2", "Iconv::Iconv;ZLIB::ZLIB;m"),
        ("CURL::libcurl", prefix / "lib/libcurl.a", prefix / "include", "ZLIB::ZLIB;OpenSSL::SSL;biominer_nghttp2;Threads::Threads;${CMAKE_DL_LIBS}"),
    )
    lines = ["# Recipe-owned exact private imports; no host pkg-config interfaces.",
             "include_guard(GLOBAL)", "find_package(Threads REQUIRED)"]
    for target, archive, include, dependencies in targets:
        lines += [f"if(TARGET {target})", f'  message(FATAL_ERROR "private provider target already exists: {target}")',
                  "endif()", f"add_library({target} STATIC IMPORTED GLOBAL)",
                  f"set_target_properties({target} PROPERTIES", f'  IMPORTED_LOCATION "{archive}"',
                  '  IMPORTED_LINK_INTERFACE_LANGUAGES "C"', f'  INTERFACE_INCLUDE_DIRECTORIES "{include}"',
                  f'  INTERFACE_LINK_LIBRARIES "{dependencies}")']
    lines += ['set_property(TARGET CURL::libcurl APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS "CURL_STATICLIB")',
              'set_property(TARGET LibXml2::LibXml2 APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS "LIBXML_STATIC")']
    return ("\n".join(lines) + "\n").encode()


def build_environment(jobs: int, *, python: str, cc: str, cxx: str) -> dict:
    if not 1 <= jobs <= 4:
        raise ValueError("--jobs must be between 1 and 4")
    # An allowlist prevents inherited compiler/backend/dependency overrides.
    env = {name: os.environ[name] for name in ("PATH", "HOME", "SSL_CERT_FILE", "SSL_CERT_DIR") if name in os.environ}
    env.update(LANG="C.UTF-8", LC_ALL="C.UTF-8", CC=cc, CXX=cxx,
               CMAKE_BUILD_PARALLEL_LEVEL="1", MAKEFLAGS="-j1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", PYARROW_BUNDLE_ARROW_CPP="ON",
               PYARROW_WITH_PARQUET_ENCRYPTION="ON", PYARROW_WITH_GANDIVA="OFF")
    for feature in FEATURES:
        env[f"PYARROW_WITH_{feature}"] = "ON"
    return env


def expected_cache(context: dict, *, stage: str) -> dict:
    expected = {"CMAKE_C_COMPILER": context["cc"], "CMAKE_CXX_COMPILER": context["cxx"]}
    expected.update({name: context["python"] for name in ("Python3_EXECUTABLE", "Python_EXECUTABLE", "PYTHON_EXECUTABLE")})
    prefix, network = Path(context["openssl_prefix"]), Path(context["network_prefix"])
    expected.update({
        "CMAKE_PROJECT_INCLUDE": str(Path(context["work"]) / "private-network.cmake"),
        "CMAKE_FIND_PACKAGE_PREFER_CONFIG": "OFF", "ZLIB_SOURCE": "SYSTEM",
        "ZLIB_LIBRARY": str(network / "lib/libz.a"), "ZLIB_LIBRARY_RELEASE": str(network / "lib/libz.a"),
        "ZLIB_LIBRARY_DEBUG": str(network / "lib/libz.a"), "ZLIB_INCLUDE_DIR": str(network / "include"),
        "CURL_NO_CURL_CMAKE": "TRUE", "CURL_USE_STATIC_LIBS": "TRUE",
        "CURL_LIBRARY": str(network / "lib/libcurl.a"), "CURL_LIBRARY_RELEASE": str(network / "lib/libcurl.a"),
        "CURL_LIBRARY_DEBUG": str(network / "lib/libcurl.a"), "CURL_INCLUDE_DIR": str(network / "include"),
        "LIBXML2_LIBRARY": str(network / "lib/libxml2.a"), "LIBXML2_INCLUDE_DIR": str(network / "include/libxml2"),
        "Iconv_LIBRARY": str(network / "lib/libiconv.a"), "Iconv_INCLUDE_DIR": str(network / "include"),
        "Iconv_IS_BUILT_IN": "FALSE", "NGHTTP2_LIBRARY": str(network / "lib/libnghttp2.a"),
        "NGHTTP2_INCLUDE_DIR": str(network / "include"), "OPENSSL_USE_STATIC_LIBS": "TRUE",
        "OPENSSL_ROOT_DIR": str(prefix), "OPENSSL_INCLUDE_DIR": str(prefix / "include"),
        "OPENSSL_CRYPTO_LIBRARY": str(prefix / "lib/libcrypto.a"), "OPENSSL_SSL_LIBRARY": str(prefix / "lib/libssl.a"),
    })
    if stage == "cpp":
        expected.update({f"ARROW_{name}": "ON" for name in FEATURES})
        expected.update({f"ARROW_WITH_{name}": "ON" for name in CODECS})
        expected.update({f"ARROW_{name}": "OFF" for name in ("TESTING", "BUILD_TESTS", "BUILD_BENCHMARKS", "BUILD_INTEGRATION", "GANDIVA")})
        expected.update({name + "_SOURCE": "BUNDLED" for name in ("simdjson", "uriparser", "Boost")})
        expected.update(ARROW_BUILD_SHARED="ON", ARROW_BUILD_STATIC="OFF", ARROW_DEPENDENCY_SOURCE="AUTO",
                        ARROW_DEPENDENCY_USE_SHARED="OFF", ARROW_JEMALLOC="ON", ARROW_MIMALLOC="ON",
                        PARQUET_REQUIRE_ENCRYPTION="ON", URIPARSER_BUILD_TESTS="OFF")
    elif stage == "python":
        expected.update({"PYARROW_" + name: "ON" for name in PYTHON_FEATURES})
        expected.update(PYARROW_GANDIVA="OFF", PYARROW_BUNDLE_ARROW_CPP="ON")
        for package in ("Arrow", "ArrowCompute", "ArrowAcero", "ArrowDataset", "ArrowFlight", "ArrowS3", "ArrowSubstrait", "Parquet"):
            expected[package + "_DIR"] = str(Path(context["work"]) / "arrow-prefix/lib/cmake" / package)
    else:
        raise ValueError("unknown CMake cache stage")
    return expected


def verify_zlib_cache_alias(path: Path, context: dict) -> None:
    """Bind the ORC cache normalization to the exact private imported archive."""
    try:
        work = Path(context["work"])
        build = work / "cpp-build"
        if path != build / "CMakeCache.txt":
            raise ValueError("cache is outside the selected C++ build")
        manifest = work / "build-inputs.json"
        if manifest.is_symlink():
            raise ValueError("build input manifest is a symlink")
        inputs = json.loads(manifest.read_bytes())
        if (inputs.get("schema_version") != 1 or inputs.get("source_policy") != SOURCE_POLICY
                or inputs.get("arrow_commit") != ARROW_COMMIT or inputs.get("context") != context):
            raise ValueError("build input identity differs from the requested configuration")
        network = inputs["network"]
        if (network["prefix"] != context["network_prefix"]
                or network["build"]["openssl"]["prefix"] != context["openssl_prefix"]):
            raise ValueError("private network/OpenSSL prefix differs from the requested configuration")
        verify_network(network)
        metadata = work / "private-network.cmake"
        if (metadata.is_symlink() or inputs["provider_cmake"] != file_record(metadata)
                or metadata.read_bytes() != private_network_cmake(Path(network["prefix"]), Path(context["openssl_prefix"]))):
            raise ValueError("private imported-target metadata differs from its pinned definition")
        archive = Path(network["prefix"]) / "lib/libz.a"
        provider = native_input_provider(archive, None, network=network)
        if ({key: provider.get(key) for key in ("kind", "component", "source_package", "source_version")}
                != {"kind": "private_network", "component": "zlib", "source_package": "zlib", "source_version": "1.3.2"}):
            raise ValueError("zlib archive lacks exact private source ownership")
        source = work / f"arrow-{ARROW_COMMIT}/cpp"
        targets, _ = capture_cmake_targets(build, source)
        selected = [row for row in targets if row["target"] == "ZLIB::ZLIB"]
        if len(selected) != 1:
            raise ValueError("one scoped ZLIB::ZLIB target is required")
        target = selected[0]
        original = json.loads(check_record(target["record"], "ZLIB_LIBRARY target").read_bytes())
        if (target["kind"] != "STATIC_LIBRARY" or target["imported"] is not True
                or target["local"] is not False or original.get("abstract") is not True
                or target["source"] != str(source) or target["binary"] != str(build)
                or target["artifacts"] != [str(archive)]):
            raise ValueError("ZLIB::ZLIB is not the global imported abstract private static archive")
    except (KeyError, TypeError, OSError, ValueError) as error:
        raise ValueError(f"cpp CMake cache rejected ZLIB_LIBRARY alias binding: {error}") from error


def verify_cache(path: Path, context: dict, *, stage: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlink {stage} CMake cache")
    cache = {}
    for line in path.read_text().splitlines():
        match = re.match(r"([^/#][^:]*):[^=]+=(.*)$", line)
        if match:
            if match[1] in cache:
                raise ValueError(f"duplicate CMake cache key: {match[1]}")
            cache[match[1]] = match[2]
    for name, expected in expected_cache(context, stage=stage).items():
        actual = cache.get(name)
        if stage == "cpp" and name == "ZLIB_LIBRARY" and actual == "ZLIB::ZLIB":
            verify_zlib_cache_alias(path, context)
            continue
        if actual != expected:
            raise ValueError(f"{stage} CMake cache rejected {name}: {actual!r}, expected {expected!r}")
    if cache.get("CMAKE_GENERATOR") not in (None, "Ninja"):
        raise ValueError("CMake generator is not bounded Ninja")
    return cache


def build_commands(source: Path, work: Path, prefix: Path, *, network_prefix: Path, python: str, cc: str, cxx: str, jobs: int,
                   cmake: str = "cmake", ninja: str = "ninja") -> dict:
    build_environment(jobs, python=python, cc=cc, cxx=cxx)
    context = {"python": python, "cc": cc, "cxx": cxx, "openssl_prefix": str(prefix),
               "network_prefix": str(network_prefix), "work": str(work)}
    flags = expected_cache(context, stage="cpp") | {
        "CMAKE_PREFIX_PATH": str(network_prefix) + ";" + str(prefix), "CMAKE_BUILD_TYPE": "Release", "CMAKE_INSTALL_LIBDIR": "lib",
        "CMAKE_INSTALL_PREFIX": str(work / "arrow-prefix"), "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "CMAKE_UNITY_BUILD": "ON", "ARROW_RPATH_ORIGIN": "ON", "CMAKE_MAKE_PROGRAM": ninja,
        "CMAKE_FIND_USE_PACKAGE_REGISTRY": "OFF", "CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY": "OFF",
        "CMAKE_SHARED_LINKER_FLAGS": "-Wl,-Map,%", "CMAKE_MODULE_LINKER_FLAGS": "-Wl,-Map,%",
    }
    python_flags = expected_cache(context, stage="python") | {
        "CMAKE_PREFIX_PATH": ";".join(map(str, (work / "arrow-prefix", network_prefix, prefix))),
        "CMAKE_MAKE_PROGRAM": ninja, "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "CMAKE_SHARED_LINKER_FLAGS": "-Wl,-Map,%", "CMAKE_MODULE_LINKER_FLAGS": "-Wl,-Map,%",
        "CMAKE_FIND_USE_PACKAGE_REGISTRY": "OFF", "CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY": "OFF",
    }
    return {
        "cpp_configure": [cmake, "-S", str(source / "cpp"), "-B", str(work / "cpp-build"), "-G", "Ninja",
                          *[f"-D{name}={value}" for name, value in flags.items()]],
        "cpp_build": [cmake, "--build", str(work / "cpp-build"), "--target", "install", "--parallel", str(jobs)],
        "python_build": [python, "-I", "-B", "-m", "build", "--wheel", "--no-isolation", "--outdir", str(work / "wheelhouse"),
                         f"--config-setting=build-dir={work / 'python-build'}", f"--config-setting=build.tool-args=-j{jobs}",
                         *[f"--config-setting=cmake.define.{name}={value}" for name, value in python_flags.items()], str(source / "python")],
    }


def tree_inventory(root: Path) -> list:
    rows = []
    for relative, path in sorted(tree_paths(root).items()):
        if path.is_symlink():
            rows.append({"relative": relative, "type": "symlink", "target": os.readlink(path)})
        elif path.is_file():
            rows.append({"relative": relative, "type": "file", "sha256": sha256(path)})
        elif path.is_dir():
            rows.append({"relative": relative, "type": "directory"})
        else:
            raise ValueError(f"unsupported source/build object: {path}")
    return rows


def system_provider(path: Path) -> dict:
    ownership = subprocess.run(["/usr/bin/dpkg-query", "-S", str(path)], capture_output=True, text=True)
    if ownership.returncode:
        raise ValueError(f"unowned system native input: {path}")
    packages = {line.split(": ", 1)[0] for line in ownership.stdout.splitlines() if ": " in line}
    if len(packages) != 1:
        raise ValueError(f"ambiguous system native input provider: {path}")
    package = next(iter(packages))
    fields = subprocess.check_output([
        "/usr/bin/dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${source:Package}\t${source:Version}", package], text=True).split("\t")
    if len(fields) != 4 or not all(fields):
        raise ValueError(f"missing system source-package/version identity: {path}")
    return {"package": fields[0], "version": fields[1], "source_package": fields[2], "source_version": fields[3],
            "source_and_advisory_review": "required_before_runtime_admission"}


def validate_python_runtime(base_python: Path) -> dict:
    """Reuse the exact standalone runtime attestation, including native inputs."""
    prefix = base_python.resolve().parent.parent
    receipt = prefix.parent / "receipt.json"
    check_record({"path": str(receipt), "sha256": "43a7e96c319752ec519a3c7e213cbb063bb8d140bbb598e69ceb6b49b2c787f4"},
                 "Python runtime source receipt")
    runtime = json.loads(receipt.read_bytes())
    if (runtime.get("schema_version") != 1 or runtime.get("source_build") != "Astral20260929"
            or runtime.get("identity") != {"version": "3.14.7", "openssl_version": "3.5.9"}
            or runtime.get("python") != file_record(base_python)
            or runtime.get("archive", {}).get("sha256") != "da5357f47b1d9c8d4c439004d463d893ef688a44c27438a62b47073d26d1377e"):
        raise ValueError("Python runtime source identity rejected")
    check_record(runtime["archive"], "Python runtime source archive")
    files = []
    for row in runtime["verified_archive_files"]:
        path = check_record(row, "Python runtime archive member")
        if not path.is_relative_to(prefix):
            raise ValueError("Python runtime source member escapes approved prefix")
        if path.is_relative_to(prefix / "include") or re.search(r"\.(a|so(?:\.[0-9]+)*)$", path.name):
            files.append(row)
    return {"source_build": runtime["source_build"], "receipt": file_record(receipt),
            "archive": runtime["archive"], "python": runtime["python"], "files": files}


def protected_native_name(path: Path) -> bool:
    return bool(re.fullmatch(r"lib(?:curl(?:-[^.]+)?|xml2|z|iconv|charset|ssl|crypto|nghttp2)"
                             r"(?:\.a|\.so(?:\.[0-9]+)*)", path.name))


def native_input_provider(path: Path, python_runtime: dict | None, *, network: dict | None = None,
                          openssl: dict | None = None) -> dict:
    path = path.resolve()
    if network:
        metadata = network["private_files"].get(str(path))
        if metadata:
            reference = {"path": str(path), "sha256": metadata["sha256"]}
            check_record(reference, "network private input")
            check_record(network["receipt"], "network receipt")
            check_record(network["helper"], "network helper")
            if network["helper"]["sha256"] != NETWORK_HELPER_SHA256:
                raise ValueError("network helper checksum is not the approved source pin")
            return {"kind": "private_network", "helper": network["helper"], "receipt": network["receipt"],
                    "file": reference, **{name: metadata[name] for name in ("component", "source_package", "source_version")}}
        if path.is_relative_to(Path(network["prefix"])):
            raise ValueError(f"unapproved private network provider: {path}")
    if openssl:
        for row in openssl["static_archives"]:
            if Path(row["path"]).resolve() == path:
                check_record(row, "private OpenSSL archive")
                check_record(openssl["receipt"], "private OpenSSL receipt")
                return {"kind": "private_openssl", "version": "3.6.5", "receipt": openssl["receipt"],
                        "source_commit": openssl["source_commit"]}
    if python_runtime:
        for row in python_runtime["files"]:
            if Path(row["path"]).resolve() == path.resolve():
                check_record(row, "Python native input")
                check_record(python_runtime["receipt"], "Python runtime source receipt")
                return {"kind": "python_runtime", "source_build": python_runtime["source_build"],
                        "receipt": python_runtime["receipt"]}
    if protected_native_name(path):
        raise ValueError(f"unapproved network/crypto provider: {path}")
    return system_provider(path)


def capture_cmake_targets(build: Path, expected_source: Path) -> tuple[list, list]:
    """Use CMake's scoped target identities and already evaluated link inputs."""
    reply = build / ".cmake/api/v1/reply"
    if reply.resolve() != reply:
        raise ValueError("File API reply directory contains a symlink")
    indexes = list(reply.glob("index-*.json"))
    if len(indexes) != 1:
        raise ValueError("File API requires one current successful index")
    records = {}

    def read(path):
        if path.parent != reply or path.is_symlink() or not path.is_file():
            raise ValueError("File API reply reference is missing or escapes its directory")
        records[str(path)] = file_record(path)
        try:
            return json.loads(path.read_bytes())
        except (ValueError, UnicodeError) as error:
            raise ValueError("File API reply is not valid JSON") from error

    def referenced(item):
        name = item.get("jsonFile")
        if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
            raise ValueError("File API reply reference is invalid")
        return reply / name

    index = read(indexes[0])
    model = read(referenced(index.get("reply", {}).get("codemodel-v2", {})))
    version = model.get("version", {})
    if (model.get("kind") != "codemodel" or version.get("major") != 2
            or type(version.get("minor")) is not int or version["minor"] < 9
            or model.get("paths") != {"source": str(expected_source), "build": str(build)}):
        raise ValueError("File API codemodel version or source/build identity rejected")
    configurations = model.get("configurations", [])
    if len(configurations) != 1:
        raise ValueError("File API requires a single Ninja configuration")
    configuration = configurations[0]
    rows, identities = [], set()
    for entry in configuration.get("targets", []) + configuration.get("abstractTargets", []):
        path = referenced(entry)
        target = read(path)
        identity = target.get("id")
        if (not isinstance(identity, str) or not identity or identity in identities
                or identity != entry.get("id") or target.get("name") != entry.get("name")):
            raise ValueError("File API target identity is missing, duplicated or mismatched")
        identities.add(identity)
        paths = target.get("paths", {})
        if not all(isinstance(paths.get(key), str) for key in ("source", "build")):
            raise ValueError("File API target source/build paths are missing")
        artifacts = [str((build / item["path"]).absolute()) for item in target.get("artifacts", [])]
        rows.append({
            "target": target["name"], "id": identity, "kind": target["type"],
            "source": str((expected_source / paths["source"]).absolute()),
            "binary": str((build / paths["build"]).absolute()),
            "artifact": artifacts[0] if artifacts else "", "artifacts": artifacts,
            "imported": target.get("imported", False), "local": target.get("local", False),
            "includes": [item for group in target.get("compileGroups", []) for item in group.get("includes", [])],
            "links": target.get("link", {}).get("commandFragments", []),
            "interface_links": target.get("interfaceLinkLibraries", []),
            "record": file_record(path),
        })
    if not rows:
        raise ValueError("File API target inventory is empty")
    return sorted(rows, key=lambda row: row["id"]), [records[name] for name in sorted(records)]


def capture_dependencies(work: Path, *, compiled: bool) -> dict:
    build = work / "cpp-build"
    inputs = json.loads((work / "build-inputs.json").read_bytes())
    network = inputs.get("network")
    if network:
        verify_network(network)
    if (build / "zlib_ep").exists():
        raise ValueError("unapproved second bundled zlib provider selected")
    targets, file_api = capture_cmake_targets(build, work / f"arrow-{ARROW_COMMIT}/cpp")
    boost = build / "biominer-boost-selection.txt"
    if not boost.is_file():
        raise ValueError("actual Boost component selection missing")
    components = sorted(set(boost.read_text().strip().split(";")))
    required_boost = {"algorithm", "crc", "numeric/conversion", "scope_exit", "throw_exception", "tokenizer", "locale", "thread", "uuid"}
    if not required_boost.issubset(components):
        raise ValueError("actual Boost component selection lacks required components")
    static_archives, native_inputs, source_roots = {}, {}, set()
    for row in targets:
        target, source = row["target"], row["source"]
        if re.search(r"(^|::)(gtest|gmock)(_|$)", target, re.I) or "aws-lc" in source:
            raise ValueError(f"unapproved testing/crypto dependency selected: {target}")
        candidates = {Path(value) for value in row["artifacts"]}
        for fragment in row["links"]:
            if fragment.get("role") == "libraries":
                for value in shlex.split(fragment["fragment"]):
                    if not value.startswith("-") and re.search(r"\.(a|so(?:\.[0-9]+)*)$", value):
                        candidates.add((build / value).absolute())
        for candidate in sorted(candidates):
            if not re.search(r"\.(a|so(?:\.[0-9]+)*)$", candidate.name) or not candidate.is_file():
                continue
            item = file_record(candidate)
            if protected_native_name(candidate.resolve()) or not candidate.resolve().is_relative_to(work):
                item["provider"] = native_input_provider(candidate.resolve(), inputs.get("python_runtime"),
                                                         network=network, openssl=inputs.get("openssl"))
            native_inputs[str(candidate)] = item
            if candidate.name.endswith(".a"):
                static_archives[str(candidate)] = item
        source_path = Path(source)
        if source_path.is_dir() and source_path.is_relative_to(build):
            source_roots.add(source_path)
    # Capture each fetched source tree once, including headers used by interface
    # targets. Keep configure-generated changes explicit, rather than guessing
    # that upstream source versions describe a system-selected package.
    source_roots.update(path for path in (build / "_deps").glob("*-src") if path.is_dir())
    source_roots = {path for path in source_roots if not any(parent in source_roots for parent in path.parents)}
    forced = {}
    for name in ("simdjson", "uriparser", "boost"):
        source = build / "_deps" / (name + "-src")
        if not source.is_dir():
            raise ValueError(f"required bundled {name} source missing")
        with tarfile.open(work / "inputs" / DEPENDENCIES[name]["filename"]) as archive:
            forced[name] = verify_archive_tree(archive, source)
    if network:
        selected = {str(Path(row["path"]).resolve()) for row in static_archives.values()}
        if any(str(Path(network["prefix"]) / "lib" / name) not in selected for name in NETWORK_ARCHIVES):
            raise ValueError("configured target inventory omitted a required private network static archive")
    versions = (work / f"arrow-{ARROW_COMMIT}/cpp/thirdparty/versions.txt").read_text()
    reviewed_hashes = set(re.findall(r"_SHA256_CHECKSUM=([0-9a-f]{64})", versions))
    archives = []
    for path in build.rglob("*"):
        if path.is_file() and not path.is_symlink() and path.name.endswith((".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".zip")):
            item = file_record(path)
            item["matches_arrow_versions_pin"] = item["sha256"] in reviewed_hashes
            archives.append(item)
    return {"schema_version": 1, "stage": "compiled" if compiled else "configured", "targets": targets,
            "private_network": {name: network[name] for name in ("receipt", "helper", "prefix")} if network else None,
            "file_api": file_api, "native_inputs": sorted(native_inputs.values(), key=lambda row: row["path"]),
            "boost_components": components, "boost_selection": file_record(boost),
            "static_archives": sorted(static_archives.values(), key=lambda row: row["path"]),
            "source_trees": [{"path": str(path), "files": tree_inventory(path)} for path in sorted(source_roots)],
            "forced_bundled_source_verification": forced, "downloaded_archives": archives,
            "coverage": "CMake File API 2.9 includes build and scoped imported targets plus resolved link fragments. Private network inputs are receipt/source-bound. Bare -l flags require final linker-map resolution. Targets are a superset; final maps establish actual linkage. Other system source/advisory review remains required."}


def verify_configured_selection(configured: dict, current: dict) -> None:
    """Keep configured inputs fixed while permitting checkpointed build outputs.

    The caller verifies the complete, explicitly hash-approved resume state
    before this comparison. New archives/generated files are therefore retained
    evidence, not an exemption from resume integrity checking.
    """
    for name in ("targets", "boost_components", "boost_selection", "forced_bundled_source_verification"):
        if configured[name] != current[name]:
            raise ValueError(f"configured dependency selection changed: {name}")
    if configured.get("file_api") != current.get("file_api"):
        raise ValueError("configured File API evidence changed")
    if configured.get("private_network") != current.get("private_network"):
        raise ValueError("configured private network provider changed")
    for name in ("static_archives", "native_inputs", "downloaded_archives"):
        observed = {row["path"]: row for row in current.get(name, [])}
        for row in configured.get(name, []):
            if observed.get(row["path"]) != row:
                raise ValueError(f"configured dependency input changed: {row['path']}")
    observed_trees = {row["path"]: row for row in current["source_trees"]}
    for tree in configured["source_trees"]:
        current_tree = observed_trees.get(tree["path"])
        if current_tree is None:
            raise ValueError(f"configured source tree disappeared: {tree['path']}")
        files = {row["relative"]: row for row in current_tree["files"]}
        for row in tree["files"]:
            if files.get(row["relative"]) != row:
                raise ValueError(f"configured source changed: {tree['path']}/{row['relative']}")


def inspect_wheel(work: Path, build_record: dict) -> dict:
    if file_record(work / "build-inputs.json") != build_record.get("build_inputs"):
        raise ValueError("immutable build manifest changed during compilation")
    wheels = list((work / "wheelhouse").glob("*.whl"))
    if len(wheels) != 1 or wheels[0].is_symlink():
        raise ValueError("expected exactly one regular Arrow wheel")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as bundle:
        names = bundle.namelist()
        if len(set(names)) != len(names):
            raise ValueError("duplicate wheel member")
        for info in bundle.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("unsafe wheel member")
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA") and len(PurePosixPath(name).parts) == 2]
        if len(metadata_names) != 1:
            raise ValueError("wheel distribution metadata is ambiguous")
        metadata = email.message_from_bytes(bundle.read(metadata_names[0]))
        if metadata["Name"] != "pyarrow" or metadata["Version"] != build_record["wheel_version"]:
            raise ValueError("wheel version does not match immutable build inputs")
        elfs = [name for name in names if not name.endswith("/") and bundle.open(name).read(4) == b"\x7fELF"]
        if not elfs:
            raise ValueError("wheel contains no native ELF")
        for name in REQUIRED_ARROW_LIBRARIES:
            if "pyarrow/" + name not in elfs:
                raise ValueError(f"required wheel native ELF missing: {name}")
        wheel_metadata = bundle.read(metadata_names[0].replace("METADATA", "WHEEL")).decode()
        if "Tag: cp314-cp314-linux_x86_64" not in wheel_metadata:
            raise ValueError("wheel Python/ABI/platform tag is not the approved cp314 Linux build")
        unpack = work / "wheel-inventory"
        unpack.mkdir()
        bundle.extractall(unpack)
    native_files, linkage = {}, {}
    wheel_paths = {(unpack / name).resolve(): name for name in elfs}
    for path, member in wheel_paths.items():
        native_files[str(path)] = {"path": "{generation}/env/lib/python3.14/site-packages/" + member,
                                   "sha256": sha256(path), "owner": "wheel:pyarrow"}
        result = subprocess.run(["/usr/bin/ldd", str(path)], capture_output=True, text=True, check=True,
                                env={"PATH": "/usr/bin:/bin", "LANG": "C"})
        if "not found" in result.stdout:
            raise ValueError(f"unresolved ELF dependency: {member}")
        linkage[member] = result.stdout
        dependencies = re.findall(r"=> (/\S+)", result.stdout) + re.findall(r"^\s*(/\S+) \(", result.stdout, re.M)
        for dependency in dependencies:
            resolved = Path(dependency).resolve()
            if resolved in wheel_paths or str(resolved) in native_files:
                continue
            provider = native_input_provider(resolved, build_record.get("python_runtime"),
                                             network=build_record.get("network"), openssl=build_record.get("openssl"))
            owner = ("python-runtime:" + provider["source_build"] if provider.get("kind") == "python_runtime"
                     else "private-network:" + provider["component"] if provider.get("kind") == "private_network"
                     else "private-openssl:3.6.5" if provider.get("kind") == "private_openssl"
                     else "dpkg:" + provider["package"])
            native_files[str(resolved)] = file_record(resolved) | {"owner": owner, "provider": provider}
    return {"wheel": file_record(wheel), "version": metadata["Version"],
            "native_files": list(native_files.values()), "linkage": linkage}


def write_resume_state(work: Path) -> dict:
    path = work / "resume-state.json"
    number = 2
    while path.exists() or path.is_symlink():
        path = work / f"resume-state-{number}.json"
        number += 1
    files = [row for row in tree_inventory(work) if row["relative"] != "build-receipt.json"]
    return preserve(path, canonical_json({"schema_version": 1, "files": files}), resume=False, label="resume state")


def verify_resume_state(work: Path, state: dict | None = None) -> None:
    path = check_record(state, "resume state") if state else work / "resume-state.json"
    if path.is_symlink() or path.parent != work or not path.is_file():
        raise ValueError("resume state must be a regular file in the owned work directory")
    value = json.loads(path.read_bytes())
    current = [row for row in tree_inventory(work) if row["relative"] not in ("build-receipt.json", path.name)]
    if value.get("schema_version") != 1 or value.get("files") != current:
        raise ValueError("resume state changed: intermediate outputs or inputs differ")


def verify_link_maps(work: Path, prefix: Path, dependencies: dict, *, network: dict) -> dict:
    verify_network(network)
    inputs_path = work / "build-inputs.json"
    inputs = json.loads(inputs_path.read_bytes()) if inputs_path.is_file() else {}
    maps, final_maps, linked, contributed = [], [], set(), set()
    closure = {row["path"]: row for row in network["files"]}
    required = [prefix / "lib/libcrypto.a", prefix / "lib/libssl.a"]
    required += [Path(network["prefix"]) / "lib" / name for name in NETWORK_ARCHIVES if name != "libcharset.a"]
    for path in required:
        if str(path) not in closure:
            raise ValueError(f"private network closure omitted required static input: {path.name}")
        check_record(closure[str(path)], "network static link input")
    for path in sorted(work.rglob("*.map")):
        if path.is_symlink() or not path.is_file():
            continue
        text = path.read_text(errors="replace")
        if "Linker script and memory map" not in text:
            continue
        maps.append(file_record(path))
        if (not (path.is_relative_to(work / "cpp-build") or path.is_relative_to(work / "python-build"))
                or any(part in {"CMakeScratch", "CMakeTmp", "_deps"} for part in path.relative_to(work).parts)
                or not re.fullmatch(r".+\.so(?:\.[0-9]+)*\.map", path.name)):
            continue
        if path.resolve() != path:
            raise ValueError("final linker map path contains a symlink")
        final_maps.append(file_record(path))
        stage_build = work / ("python-build" if path.is_relative_to(work / "python-build") else "cpp-build")
        for archive in re.findall(r"(?:^|\s)([^\s()]+\.a)\([^\s()]+\)", text, re.M):
            contributed.add(str((stage_build / archive).resolve()))
        libraries = re.findall(r"(?:^|\s)([^\s()]+(?:\.a|\.so(?:\.[0-9]+)*))(?=[\s()]|$)", text, re.M)
        for name in libraries:
            library = (stage_build / name).resolve()
            if protected_native_name(library):
                if library in (prefix / "lib/libssl.a", prefix / "lib/libcrypto.a"):
                    check_record(closure[str(library)], "network private OpenSSL link input")
                else:
                    native_input_provider(library, inputs.get("python_runtime"), network=network,
                                          openssl=inputs.get("openssl"))
            if library.suffix == ".a":
                linked.add(str(library))
    if not maps:
        raise ValueError("build produced no per-output linker maps")
    resolved = linked
    for path in required:
        if str(path.resolve()) not in resolved:
            raise ValueError(f"final link maps do not prove actual static linkage: {path.name}")
        if str(path.resolve()) not in contributed:
            raise ValueError(f"final link maps do not prove a consumed archive member: {path.name}")
    if len([name for name in resolved if Path(name).name == "libz.a"]) != 1:
        raise ValueError("exactly one approved private libz.a required in final linkage")
    known = {str(Path(row["path"]).resolve()): row for row in dependencies["static_archives"]}
    records = []
    for name in sorted(resolved):
        path = Path(name)
        if name in known:
            check_record(known[name], "linked static input")
            item = known[name]
        else:
            item = file_record(path)
            if not path.is_relative_to(work) and not path.is_relative_to(prefix):
                item["provider"] = native_input_provider(path, inputs.get("python_runtime"),
                                                         network=network, openssl=inputs.get("openssl"))
        if name in network["private_files"]:
            item = {**item, "provider": native_input_provider(path, inputs.get("python_runtime"), network=network,
                                                             openssl=inputs.get("openssl"))}
        records.append(item)
    return {"maps": maps, "final_maps": final_maps, "linked_static_archives": records,
            "qualification": "Every required private archive contributes an object member in final Arrow/Python consumer maps, excluding configure probes; LOAD entries alone do not establish consumption. Final ELF closure and all other system source/provider reviews remain required."}


def validate_openssl(prefix: Path, source_archive: Path) -> dict:
    header = prefix / "include/openssl/opensslv.h"
    if not header.is_file() or '"OpenSSL 3.6.5 ' not in header.read_text():
        raise ValueError("OpenSSL 3.6.5 required")
    check_record({"path": str(header), "sha256": "80d9b9534fdbdab12c29cb8c62887ede14bbb7112b93e838a63d6815400f5f0b"}, "OpenSSL header")
    expected = {"libcrypto.a": "e2d8161275df2868b63e28d8d1f5b49efa3e40a3697d1a21a07890c58a2d3bc2",
                "libssl.a": "3ac9b8f49ce67cc03e9a1073754faaec92226195d610bd59800e4df5c41a82b6"}
    archives = [file_record(check_record({"path": str(prefix / "lib" / name), "sha256": value}, "approved static OpenSSL archive"))
                for name, value in expected.items()]
    receipt = prefix.parent / "openssl-build-receipt.json"
    check_record({"path": str(receipt), "sha256": "66d41e5fa8cee1cdb865a7df41620685776c96fd8eba9abcc6a70505ca3da13a"}, "OpenSSL source receipt")
    check_record({"path": str(source_archive), "sha256": "f3091b7ccb41a11d877b7406b7afc9dd9c5003d299c6bb33cff882127bc06c91"}, "OpenSSL source archive")
    return {"version": "3.6.5", "source_commit": "c8bd5a57108599ac650bbae77fcabe3109dab2e8",
            "receipt": file_record(receipt), "source_archive": file_record(source_archive),
            "header": file_record(header), "headers": tree_inventory(prefix / "include"), "static_archives": archives}


def expected_generated_version(python: str, version: str) -> bytes:
    # This private API is bound to the exact frozen vcs-versioning 2.6.0 wheel.
    # Formatting prints bytes and does not write into or import the Arrow tree.
    code = (
        "import sys,base64; from vcs_versioning._dump_version import DEFAULT_TEMPLATES,DummyScmVersion; "
        "from vcs_versioning._version_cls import _version_as_tuple; "
        "v=sys.argv[1]; content=DEFAULT_TEMPLATES['.py'].format(version=v,version_tuple=_version_as_tuple(v),"
        "scm_version=DummyScmVersion()); print(base64.b64encode(content.encode()).decode())")
    return base64.b64decode(subprocess.check_output([python, "-I", "-B", "-c", code, version], text=True).strip(), validate=True)


def compile_source_evidence(work: Path) -> dict:
    path = work / "cpp-build/compile_commands.json"
    entries = json.loads(path.read_bytes())
    expected = {work / f"arrow-{ARROW_COMMIT}/cpp/src/arrow/compute/kernels" / name
                for name in ("vector_replace.cc", "scalar_if_else.cc")}
    found, units = set(), []
    for row in entries:
        source = Path(row["file"])
        if not source.is_absolute():
            source = Path(row["directory"]) / source
        if source in expected:
            found.add(source)
            units.append(file_record(source))
        elif "Unity" in source.parts and source.is_file():
            includes = {Path(name) for name in re.findall(r'^#include "([^"]+)"', source.read_text(), re.M)}
            relevant = includes & expected
            if relevant:
                found.update(relevant)
                units.append(file_record(source))
    if found != expected:
        raise ValueError("compile commands do not bind both integrated header consumers")
    return {"compile_commands": file_record(path), "translation_units": units,
            "integrated_header_consumers": [file_record(path) for path in sorted(found)]}


def run_stage(work: Path, name: str, command: list, env: dict) -> dict:
    logs = work / "logs"
    logs.mkdir(exist_ok=True)
    number = 1
    while (logs / f"{name}-{number}.log").exists():
        number += 1
    path = logs / f"{name}-{number}.log"
    with path.open("x") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return {"command": command, "log": file_record(path), "exit_code": 0}


def run_network_stage(work: Path, name: str, command: list, env: dict, network: dict, *, frozen_inputs: tuple = ()) -> dict:
    def guard():
        verify_network(network)
        for reference in frozen_inputs:
            check_record(reference, "immutable build input")
    guard()
    try:
        return run_stage(work, name, command, env)
    finally:
        guard()


def main(argv: list | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    recipe_bytes = Path(__file__).read_bytes()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("openssl-prefix", "archive", "work", "builder-receipt", "simdjson-archive", "uriparser-archive", "boost-archive", "zlib-archive",
                 "network-receipt", "network-helper"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("builder-receipt-sha256", "network-receipt-sha256", "python", "cc", "cxx"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--openssl-source-archive", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--check", action="store_true", help="verify approved inputs without creating a build generation")
    parser.add_argument("--configure-only", action="store_true", help="stop for review of actual native dependency selection")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-state-sha256")
    parser.add_argument("--configured-inputs-sha256")
    args = parser.parse_args(argv)
    work, archive, prefix = args.work.absolute(), args.archive.absolute(), args.openssl_prefix.absolute()
    builder, record, return_code = None, None, 1
    try:
        env = build_environment(args.jobs, python=args.python, cc=args.cc, cxx=args.cxx)
        if work.is_relative_to("/tmp") or work.is_relative_to("/var/tmp"):
            raise ValueError("build work must use persistent storage outside /tmp and /var/tmp")
        if any(character in str(work) for character in "\t\n\r{}"):
            raise ValueError("work path contains unsupported CMake/template characters")
        if not args.check and not args.configure_only and (not args.resume or not args.configured_inputs_sha256):
            raise ValueError("compilation requires --resume and reviewed --configured-inputs-sha256")
        builder = validate_builder(args.builder_receipt.absolute(), args.builder_receipt_sha256,
                                   python=args.python, cc=args.cc, cxx=args.cxx)
        observed_builder = probe_builder(builder)
        python_runtime = validate_python_runtime(Path(builder["base_python"]["path"]))
        openssl_archive = args.openssl_source_archive.absolute() if args.openssl_source_archive else prefix.parent / "downloads/openssl.tar.gz"
        openssl = validate_openssl(prefix, openssl_archive)
        network = validate_network(args.network_receipt, args.network_receipt_sha256, args.network_helper,
                                   openssl_prefix=prefix)
        if sha256(archive) != ARROW_ARCHIVE_SHA256:
            raise ValueError("Arrow source archive checksum mismatch")
        archives = {name: getattr(args, name + "_archive").absolute() for name in DEPENDENCIES}
        for name, path in archives.items():
            check_record({"path": str(path), "sha256": DEPENDENCIES[name]["sha256"]}, name + " archive")
        if args.check:
            return 0
        if args.resume:
            if not args.resume_state_sha256:
                raise ValueError("resume requires approved --resume-state-sha256")
            previous = json.loads((work / "build-receipt.json").read_bytes())
            state = previous["resume_state"]
            if state["sha256"] != args.resume_state_sha256:
                raise ValueError("resume state does not match approved hash")
            verify_resume_state(work, state)
        source, source_verification = prepare_source(archive, work, resume=args.resume)
        recipe_record = preserve_recipe(work, recipe_bytes, resume=args.resume)
        helper_record = preserve(work / "private-network-helper.py", Path(network["helper"]["path"]).read_bytes(),
                                 resume=args.resume, label="network helper source")
        captured_network = validate_network(args.network_receipt, args.network_receipt_sha256, Path(helper_record["path"]),
                                            openssl_prefix=prefix)
        if {key: value for key, value in captured_network.items() if key != "helper"} != {
                key: value for key, value in network.items() if key != "helper"}:
            raise ValueError("network source/provider closure changed during preparation")
        network = captured_network
        network_prefix = Path(network["prefix"])
        provider_cmake = preserve(work / "private-network.cmake", private_network_cmake(network_prefix, prefix),
                                  resume=args.resume, label="private network CMake metadata")
        (work / "inputs").mkdir(exist_ok=True)
        query = work / "cpp-build/.cmake/api/v1/query/codemodel-v2"
        query.parent.mkdir(parents=True, exist_ok=True)
        query_record = preserve(query, b"", resume=args.resume, label="CMake File API query")
        dependencies = {}
        for name, path in archives.items():
            copied = preserve(work / "inputs" / DEPENDENCIES[name]["filename"], path.read_bytes(),
                              resume=args.resume, label=name + " source archive")
            dependencies[name] = {**DEPENDENCIES[name], "archive": copied,
                                  "role": "retained_source_pin_only" if name == "zlib" else "bundled_provider"}
        context = {"python": args.python, "cc": args.cc, "cxx": args.cxx,
                   "openssl_prefix": str(prefix), "network_prefix": str(network_prefix), "work": str(work)}
        commands = build_commands(source, work, prefix, network_prefix=network_prefix,
                                   python=args.python, cc=args.cc, cxx=args.cxx, jobs=args.jobs,
                                   cmake=builder["toolchain"]["cmake"]["path"], ninja=builder["toolchain"]["ninja"]["path"])
        tool_versions = {name: subprocess.check_output([row["path"], "--version"], text=True).splitlines()[0]
                         for name, row in builder["toolchain"].items()}
        inputs = {
            "schema_version": 1, "source_policy": SOURCE_POLICY, "arrow_commit": ARROW_COMMIT,
            "source_archive": file_record(archive), "source_verification": file_record(work / "source-verification.json"),
            "integrated_sources": source_verification["integrated_sources"], "recipe": recipe_record,
            "builder": builder, "observed_builder": observed_builder, "toolchain_versions": tool_versions,
            "python_runtime": python_runtime,
            "openssl": openssl, "network": network, "provider_cmake": provider_cmake, "dependencies": dependencies,
            "commands": commands, "context": context,
            "cmake_file_api_query": query_record,
            "compiler_budget": {"maximum_jobs": args.jobs, "outer_ninja_jobs": args.jobs, "nested_cmake_jobs": 1, "nested_make_jobs": 1},
            "features": FEATURES, "codecs": CODECS,
            "version_environment": {"SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYARROW": "{wheel_version}"},
            "python_feature_environment": {key: value for key, value in env.items() if key.startswith("PYARROW_")},
            "generated_version_policy": "frozen vcs-versioning 2.6.0 default Python template; None SCM node; exact derived version",
            "resume_limit": "Generated Python source from failed wheel builds is preserved and requires a fresh generation.",
            "qualification": "unqualified host-specific candidate; all selected system sources and advisory gaps require independent review",
        }
        manifest, version = preserve_build_inputs(work, inputs, resume=args.resume)
        frozen_inputs = (recipe_record, provider_cmake, manifest, inputs["source_verification"], query_record,
                         *(dependency["archive"] for dependency in dependencies.values()))
        expected_version = expected_generated_version(args.python, version)
        preserve(work / "expected-generated-version.py", expected_version, resume=args.resume, label="generated version template")
        # Validate the reviewed configuration before replacing the mutable
        # incomplete-build receipt or adding a new checkpoint/log. A rejected
        # approval must leave the prior evidence exactly as it was.
        if not args.configure_only:
            configured_path = check_record({"path": str(work / "configured-inputs.json"), "sha256": args.configured_inputs_sha256}, "reviewed configured inputs")
            configured = json.loads(configured_path.read_bytes())
            check_record(configured["cache"], "reviewed configured cache")
            if configured["build_inputs"] != manifest:
                raise ValueError("configured dependency selection changed before compilation")
            verify_configured_selection(configured["dependencies"], capture_dependencies(work, compiled=False))
            verify_cache(work / "cpp-build/CMakeCache.txt", context, stage="cpp")
        record = {**inputs, "status": "building", "build_inputs": manifest, "wheel_version": version, "stages": []}
        receipt_path = work / "build-receipt.json"
        receipt_path.write_bytes(canonical_json(record))
        env.update(GIT_CEILING_DIRECTORIES=str(work), OPENSSL_ROOT_DIR=str(prefix),
                   PKG_CONFIG_PATH=os.pathsep.join(map(str, (network_prefix / "lib/pkgconfig", prefix / "lib/pkgconfig"))),
                   ARROW_HOME=str(work / "arrow-prefix"),
                   CMAKE_PREFIX_PATH=";".join(map(str, (work / "arrow-prefix", network_prefix, prefix))),
                   SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYARROW=version, LDFLAGS="-Wl,-Map,%", TMPDIR=str(work / "tmp"))
        env["PATH"] = os.pathsep.join(dict.fromkeys([str(Path(row["path"]).parent) for row in builder["toolchain"].values()] +
                                                   [str(Path(args.python).parent), "/usr/bin", "/bin"]))
        for name, dependency in dependencies.items():
            if dependency["role"] == "bundled_provider":
                env[dependency["env"]] = dependency["archive"]["path"]
        (work / "tmp").mkdir(exist_ok=True)
        if args.configure_only:
            record["stages"].append(run_network_stage(work, "configure", commands["cpp_configure"], env, network,
                                                     frozen_inputs=frozen_inputs))
            verify_cache(work / "cpp-build/CMakeCache.txt", context, stage="cpp")
            configured = {"schema_version": 1, "build_inputs": manifest,
                          "cache": file_record(work / "cpp-build/CMakeCache.txt"),
                          "dependencies": capture_dependencies(work, compiled=False)}
            configured_record = preserve(work / "configured-inputs.json", canonical_json(configured), resume=False, label="configured inputs")
            record.update(status="configured_unverified", configured_inputs=configured_record)
            return_code = 0
        else:
            record["stages"].append(run_network_stage(work, "cpp-build", commands["cpp_build"], env, network,
                                                     frozen_inputs=frozen_inputs))
            verify_cache(work / "cpp-build/CMakeCache.txt", context, stage="cpp")
            dependencies_compiled = capture_dependencies(work, compiled=True)
            source_compilation = compile_source_evidence(work)
            record["stages"].append(run_network_stage(work, "python-build", commands["python_build"], env, network,
                                                     frozen_inputs=frozen_inputs))
            verify_cache(work / "python-build/CMakeCache.txt", context, stage="python")
            validate_builder(args.builder_receipt.absolute(), args.builder_receipt_sha256,
                             python=args.python, cc=args.cc, cxx=args.cxx)
            if validate_python_runtime(Path(builder["base_python"]["path"])) != python_runtime:
                raise ValueError("Python runtime source inputs changed during build")
            if validate_openssl(prefix, openssl_archive) != openssl:
                raise ValueError("approved OpenSSL inputs changed during build")
            with tarfile.open(archive) as bundle:
                root, _ = archive_members(bundle)
                final_sources = verify_archive_tree(bundle, source, source_transformations(bundle, root),
                    {"python/pyarrow/_generated_version.py": expected_version})
            maps = verify_link_maps(work, prefix, dependencies_compiled, network=network)
            wheel = inspect_wheel(work, record)
            verify_network(network)
            for reference in frozen_inputs:
                check_record(reference, "immutable build input")
            dependency_record = preserve(work / "compiled-dependencies.json", canonical_json(dependencies_compiled), resume=False, label="compiled dependencies")
            final_source_record = preserve(work / "compiled-sources.json", canonical_json({"files": final_sources, **source_compilation}), resume=False, label="compiled sources")
            record = {
                "schema_version": 1, "source_policy": SOURCE_POLICY, "status": "compiled_unverified", "name": "pyarrow", **wheel,
                "wheel_sha256": wheel["wheel"]["sha256"], "source_revision": ARROW_COMMIT,
                "recipe": recipe_record, "build_inputs": manifest, "source_verification": inputs["source_verification"],
                "network": network, "openssl": openssl, "provider_cmake": provider_cmake,
                "compiled_dependencies": dependency_record, "compiled_sources": final_source_record,
                "native_link_artifacts": maps["maps"], "final_native_link_artifacts": maps["final_maps"],
                "linked_static_archives": maps["linked_static_archives"],
                "toolchain": builder["toolchain"], "toolchain_versions": tool_versions,
                "python_build_distributions": builder["distributions"],
                "python_runtime": python_runtime,
                "effective_versions": {"pyarrow": version, "openssl": "3.6.5", **{name: dep["version"] for name, dep in DEPENDENCIES.items()}},
                "build": record, "qualification": "compiled host-specific candidate; native/source/security/API qualification and admission still required",
            }
            return_code = 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError, tarfile.TarError, zipfile.BadZipFile) as error:
        if record is None:
            parser.error(str(error))
        record.update(status="failed", error=str(error))
        print(str(error), file=sys.stderr)
    finally:
        if record is not None:
            if record["status"] != "compiled_unverified":
                record["resume_state"] = write_resume_state(work)
            (work / "build-receipt.json").write_bytes(canonical_json(record))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
