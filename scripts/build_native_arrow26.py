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
ZLIB_PATCH_COMMIT = "df84af25dc1942490e1d1c899a07619152a46148"
ZLIB_PATCH_SHA256 = "110ff14375733173d8aa54574473424fbd7dfe4b81f1ca34a759c6fe14b15b14"
ZLIB_ORIGINAL_GZWRITE_SHA256 = "2ce783294c688330c716bfe106f43167d3dae5be84994176bc304d205d4f5b04"
ZLIB_PATCHED_GZWRITE_SHA256 = "31d0da140edf382b82b13e2d87a22f59e158d9014ee026459e01cf437d0ab07d"
ZLIB_PATCH_PATH = Path(__file__).parent / "native_patches/zlib-df84af25.patch"
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
    ("ARROW_ZLIB_BUILD_VERSION=1.3.1", "ARROW_ZLIB_BUILD_VERSION=1.3.2"),
    ("ARROW_ZLIB_BUILD_SHA256_CHECKSUM=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23",
     "ARROW_ZLIB_BUILD_SHA256_CHECKSUM=" + DEPENDENCIES["zlib"]["sha256"]),
)
ZLIB_CMAKE_ANCHOR = '    set(ZLIB_CMAKE_ARGS ${EP_COMMON_CMAKE_ARGS} "-DCMAKE_INSTALL_PREFIX=${ZLIB_PREFIX}")'
ZLIB_PROJECT_ANCHOR = '                        URL_HASH "SHA256=${ARROW_ZLIB_BUILD_SHA256_CHECKSUM}"'
BOOST_SELECTION_ANCHOR = "  fetchcontent_makeavailable(boost)"

# Deferred until Arrow's top-level directory is complete. The target properties
# record configured selection; actual linker maps separately establish linkage.
DEPENDENCY_CAPTURE_CMAKE = r'''
# BioMiner: retain actual configured targets and their source/link inputs.
function(biominer_capture_directory directory)
  get_property(local_targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
  get_property(imported_targets DIRECTORY "${directory}" PROPERTY IMPORTED_TARGETS)
  foreach(target IN LISTS local_targets imported_targets)
    if(NOT TARGET "${target}")
      continue()
    endif()
    get_target_property(kind "${target}" TYPE)
    get_target_property(source_dir "${target}" SOURCE_DIR)
    get_target_property(binary_dir "${target}" BINARY_DIR)
    get_target_property(includes "${target}" INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(links "${target}" INTERFACE_LINK_LIBRARIES)
    string(MAKE_C_IDENTIFIER "${target}" safe_target)
    set(artifact "")
    if(kind MATCHES "^(STATIC|SHARED|MODULE|UNKNOWN)_LIBRARY$")
      set(artifact "$<TARGET_FILE:${target}>")
    endif()
    file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/biominer-targets/${safe_target}.tsv"
         CONTENT "${target}\t${kind}\t${source_dir}\t${binary_dir}\t${artifact}\t${includes}\t${links}\n")
  endforeach()
  get_property(children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
  foreach(child IN LISTS children)
    biominer_capture_directory("${child}")
  endforeach()
endfunction()
cmake_language(DEFER CALL biominer_capture_directory "${CMAKE_SOURCE_DIR}")
'''


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
    toolchain = exact_replace(toolchain, ZLIB_CMAKE_ANCHOR, ZLIB_CMAKE_ANCHOR[:-1] +
        ' "-DZLIB_BUILD_TESTING=OFF" "-DZLIB_BUILD_SHARED=OFF" "-DZLIB_BUILD_STATIC=ON"'
        ' "-DZLIB_INSTALL=ON" "-DCMAKE_INSTALL_LIBDIR=lib")')
    toolchain = exact_replace(toolchain, ZLIB_PROJECT_ANCHOR, ZLIB_PROJECT_ANCHOR +
        '\n                        PATCH_COMMAND "${Python3_EXECUTABLE}" -I -B "${BIOMINER_RECIPE}"'
        ' --apply-zlib-patch --work "${BIOMINER_WORK}" --zlib-source <SOURCE_DIR>'
        ' --wheel-version "${BIOMINER_WHEEL_VERSION}"')
    toolchain = exact_replace(toolchain, BOOST_SELECTION_ANCHOR, BOOST_SELECTION_ANCHOR +
        '\n  file(WRITE "${CMAKE_BINARY_DIR}/biominer-boost-selection.txt" "${BOOST_INCLUDE_LIBRARIES}\\n")')
    toolchain += DEPENDENCY_CAPTURE_CMAKE.encode()
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
                     "resume-state.json", "logs", "zlib-patch-application.json"):
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


def apply_zlib_patch(source: Path, archive: Path, patch: Path, receipt: Path) -> dict:
    if sha256(archive) != DEPENDENCIES["zlib"]["sha256"]:
        raise ValueError("zlib archive checksum mismatch")
    if sha256(patch) != ZLIB_PATCH_SHA256:
        raise ValueError("zlib patch checksum mismatch")
    if receipt.is_symlink():
        raise ValueError("zlib patch receipt is a symlink")
    resumed = receipt.exists()
    with tarfile.open(archive) as bundle:
        root, _ = archive_members(bundle)
        original = bundle.extractfile(root + "/gzwrite.c").read()
        if hashlib.sha256(original).hexdigest() != ZLIB_ORIGINAL_GZWRITE_SHA256:
            raise ValueError("zlib original gzwrite source checksum mismatch")
        replacements = {}
        if resumed:
            current = source / "gzwrite.c"
            if not current.is_file() or current.is_symlink() or sha256(current) != ZLIB_PATCHED_GZWRITE_SHA256:
                raise ValueError("zlib patched gzwrite source checksum mismatch")
            replacements["gzwrite.c"] = current.read_bytes()
        verify_archive_tree(bundle, source, replacements)
        if not resumed:
            result = subprocess.run(["/usr/bin/patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch)],
                                    cwd=source, capture_output=True, text=True)
            output = result.stdout + result.stderr
            if result.returncode or re.search(r"offset|fuzz|failed|reversed", output, re.I):
                raise ValueError("zlib patch did not apply exactly")
            if sha256(source / "gzwrite.c") != ZLIB_PATCHED_GZWRITE_SHA256:
                raise ValueError("zlib patched gzwrite source checksum mismatch")
            replacements["gzwrite.c"] = (source / "gzwrite.c").read_bytes()
        files = verify_archive_tree(bundle, source, replacements)
    application = {"schema_version": 1, "patch_commit": ZLIB_PATCH_COMMIT,
                   "source": str(source.absolute()), "archive": file_record(archive), "patch": file_record(patch),
                   "original_gzwrite_sha256": ZLIB_ORIGINAL_GZWRITE_SHA256,
                   "patched_gzwrite_sha256": ZLIB_PATCHED_GZWRITE_SHA256, "files": files}
    preserve(receipt, canonical_json(application), resume=resumed, label="zlib patch receipt")
    return application


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
    if stage == "cpp":
        expected.update({f"ARROW_{name}": "ON" for name in FEATURES})
        expected.update({f"ARROW_WITH_{name}": "ON" for name in CODECS})
        expected.update({f"ARROW_{name}": "OFF" for name in ("TESTING", "BUILD_TESTS", "BUILD_BENCHMARKS", "BUILD_INTEGRATION", "GANDIVA")})
        expected.update({name + "_SOURCE": "BUNDLED" for name in ("simdjson", "uriparser", "Boost", "ZLIB")})
        prefix = Path(context["openssl_prefix"])
        expected.update(ARROW_BUILD_SHARED="ON", ARROW_BUILD_STATIC="OFF", ARROW_DEPENDENCY_SOURCE="AUTO",
                        ARROW_DEPENDENCY_USE_SHARED="OFF", ARROW_JEMALLOC="ON", ARROW_MIMALLOC="ON",
                        PARQUET_REQUIRE_ENCRYPTION="ON", OPENSSL_USE_STATIC_LIBS="TRUE",
                        OPENSSL_ROOT_DIR=str(prefix), OPENSSL_CRYPTO_LIBRARY=str(prefix / "lib/libcrypto.a"),
                        OPENSSL_SSL_LIBRARY=str(prefix / "lib/libssl.a"), URIPARSER_BUILD_TESTS="OFF")
    elif stage == "python":
        expected.update({"PYARROW_" + name: "ON" for name in PYTHON_FEATURES})
        expected.update(PYARROW_GANDIVA="OFF", PYARROW_BUNDLE_ARROW_CPP="ON")
        for package in ("Arrow", "ArrowCompute", "ArrowAcero", "ArrowDataset", "ArrowFlight", "ArrowS3", "ArrowSubstrait", "Parquet"):
            expected[package + "_DIR"] = str(Path(context["work"]) / "arrow-prefix/lib/cmake" / package)
    else:
        raise ValueError("unknown CMake cache stage")
    return expected


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
        if actual != expected:
            raise ValueError(f"{stage} CMake cache rejected {name}: {actual!r}, expected {expected!r}")
    if cache.get("CMAKE_GENERATOR") not in (None, "Ninja"):
        raise ValueError("CMake generator is not bounded Ninja")
    return cache


def build_commands(source: Path, work: Path, prefix: Path, *, python: str, cc: str, cxx: str, jobs: int,
                   cmake: str = "cmake", ninja: str = "ninja") -> dict:
    build_environment(jobs, python=python, cc=cc, cxx=cxx)
    context = {"python": python, "cc": cc, "cxx": cxx, "openssl_prefix": str(prefix), "work": str(work)}
    flags = expected_cache(context, stage="cpp") | {
        "CMAKE_PREFIX_PATH": str(prefix), "CMAKE_BUILD_TYPE": "Release", "CMAKE_INSTALL_LIBDIR": "lib",
        "CMAKE_INSTALL_PREFIX": str(work / "arrow-prefix"), "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "CMAKE_UNITY_BUILD": "ON", "ARROW_RPATH_ORIGIN": "ON", "CMAKE_MAKE_PROGRAM": ninja,
        "CMAKE_FIND_USE_PACKAGE_REGISTRY": "OFF", "CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY": "OFF",
        "CMAKE_SHARED_LINKER_FLAGS": "-Wl,-Map,%", "CMAKE_MODULE_LINKER_FLAGS": "-Wl,-Map,%",
        "BIOMINER_RECIPE": str(work / "build-recipe.py"), "BIOMINER_WORK": str(work),
        "BIOMINER_WHEEL_VERSION": "{wheel_version}",
    }
    python_flags = expected_cache(context, stage="python") | {
        "CMAKE_PREFIX_PATH": str(work / "arrow-prefix") + ";" + str(prefix),
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


def capture_dependencies(work: Path, *, compiled: bool) -> dict:
    build = work / "cpp-build"
    inputs = json.loads((work / "build-inputs.json").read_bytes())
    private_archives = {str(Path(row["path"]).resolve()): row for row in inputs["openssl"]["static_archives"]}
    target_paths = sorted((build / "biominer-targets").glob("*.tsv"))
    if not target_paths:
        raise ValueError("configured dependency target inventory missing")
    boost = build / "biominer-boost-selection.txt"
    if not boost.is_file():
        raise ValueError("actual Boost component selection missing")
    components = sorted(set(boost.read_text().strip().split(";")))
    required_boost = {"algorithm", "crc", "numeric/conversion", "scope_exit", "throw_exception", "tokenizer", "locale", "thread", "uuid"}
    if not required_boost.issubset(components):
        raise ValueError("actual Boost component selection lacks required components")
    targets, static_archives, source_roots = [], {}, set()
    for path in target_paths:
        fields = path.read_text().rstrip("\n").split("\t")
        if len(fields) != 7:
            raise ValueError(f"malformed dependency target inventory: {path}")
        target, kind, source, binary, artifact, includes, links = fields
        if re.search(r"(^|::)(gtest|gmock)(_|$)", target, re.I) or "aws-lc" in source:
            raise ValueError(f"unapproved testing/crypto dependency selected: {target}")
        row = {"target": target, "kind": kind, "source": source, "binary": binary,
               "artifact": artifact, "includes": includes, "links": links, "record": file_record(path)}
        targets.append(row)
        candidate = Path(artifact)
        if artifact.endswith(".a") and candidate.is_file():
            item = file_record(candidate)
            if str(candidate.resolve()) in private_archives:
                check_record(private_archives[str(candidate.resolve())], "private OpenSSL archive")
                item["provider"] = {"kind": "private_openssl", "version": "3.6.5",
                                    "receipt": inputs["openssl"]["receipt"], "source_commit": inputs["openssl"]["source_commit"]}
            elif not candidate.resolve().is_relative_to(work):
                item["provider"] = system_provider(candidate.resolve())
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
    zlib_source = build / "zlib_ep/src/zlib_ep"
    if compiled:
        application = apply_zlib_patch(zlib_source, work / "inputs" / DEPENDENCIES["zlib"]["filename"],
                                       work / "zlib-df84af25.patch", work / "zlib-patch-application.json")
        forced["zlib"] = application["files"]
        zcache = zlib_source.parent / "zlib_ep-build/CMakeCache.txt"
        values = dict(re.findall(r"^([^/#][^:]*):[^=]+=(.*)$", zcache.read_text(), re.M))
        for key, value in {"ZLIB_BUILD_TESTING": "OFF", "ZLIB_BUILD_SHARED": "OFF", "ZLIB_BUILD_STATIC": "ON", "ZLIB_INSTALL": "ON"}.items():
            if values.get(key) != value:
                raise ValueError(f"zlib CMake cache rejected {key}")
        source_roots.add(zlib_source)
        zarchive = build / "zlib_ep/src/zlib_ep-install/lib/libz.a"
        if not zarchive.is_file() or str(zarchive) not in static_archives:
            raise ValueError("actual bundled patched zlib static archive missing")
    versions = (work / f"arrow-{ARROW_COMMIT}/cpp/thirdparty/versions.txt").read_text()
    reviewed_hashes = set(re.findall(r"_SHA256_CHECKSUM=([0-9a-f]{64})", versions))
    archives = []
    for path in build.rglob("*"):
        if path.is_file() and not path.is_symlink() and path.name.endswith((".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".zip")):
            item = file_record(path)
            item["matches_arrow_versions_pin"] = item["sha256"] in reviewed_hashes
            archives.append(item)
    return {"schema_version": 1, "stage": "compiled" if compiled else "configured", "targets": targets,
            "boost_components": components, "boost_selection": file_record(boost),
            "static_archives": sorted(static_archives.values(), key=lambda row: row["path"]),
            "source_trees": [{"path": str(path), "files": tree_inventory(path)} for path in sorted(source_roots)],
            "forced_bundled_source_verification": forced, "downloaded_archives": archives,
            "coverage": "Configured targets are a superset. Final link maps identify used archives. System source and advisory review remains required; unsupported coverage is not clearance."}


def verify_configured_selection(configured: dict, current: dict) -> None:
    """Keep configured inputs fixed while permitting checkpointed build outputs.

    The caller verifies the complete, explicitly hash-approved resume state
    before this comparison. New archives/generated files are therefore retained
    evidence, not an exemption from resume integrity checking.
    """
    for name in ("targets", "boost_components", "boost_selection", "forced_bundled_source_verification"):
        if configured[name] != current[name]:
            raise ValueError(f"configured dependency selection changed: {name}")
    for name in ("static_archives", "downloaded_archives"):
        observed = {row["path"]: row for row in current[name]}
        for row in configured[name]:
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
            provider = system_provider(resolved)
            native_files[str(resolved)] = file_record(resolved) | {"owner": "dpkg:" + provider["package"], "provider": provider}
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


def verify_link_maps(work: Path, prefix: Path, dependencies: dict) -> dict:
    maps, linked = [], set()
    for path in sorted(work.rglob("*.map")):
        if path.is_symlink() or not path.is_file():
            continue
        text = path.read_text(errors="replace")
        if "Linker script and memory map" not in text:
            continue
        maps.append(file_record(path))
        linked.update(re.findall(r"(?:^|\s)(/[^\s()]+\.a)(?:\(|\s|$)", text, re.M))
    if not maps:
        raise ValueError("build produced no per-output linker maps")
    required = [prefix / "lib/libcrypto.a", prefix / "lib/libssl.a"]
    zlib = [Path(row["path"]) for row in dependencies["static_archives"] if Path(row["path"]).name == "libz.a"]
    if len(zlib) != 1:
        raise ValueError("exactly one approved bundled libz.a required")
    required.extend(zlib)
    resolved = {str(Path(path).resolve()) for path in linked}
    for path in required:
        if str(path.resolve()) not in resolved:
            raise ValueError(f"link maps do not prove actual static linkage: {path.name}")
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
                item["provider"] = system_provider(path)
        records.append(item)
    return {"maps": maps, "linked_static_archives": records,
            "qualification": "Link maps include build outputs; runtime admission must review final wheel library maps and every system source/provider."}


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


def private_zlib_main(argv: list) -> int:
    parser = argparse.ArgumentParser(description="Guarded Arrow 26 ExternalProject patch step")
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--zlib-source", required=True, type=Path)
    parser.add_argument("--wheel-version", required=True)
    args = parser.parse_args(argv)
    work = args.work.absolute()
    manifest = work / "build-inputs.json"
    if args.wheel_version != "26.0.0+biominer." + sha256(manifest)[:12]:
        raise ValueError("zlib patch step build manifest version mismatch")
    inputs = json.loads(manifest.read_bytes())
    if file_record(Path(__file__)) != inputs["recipe"]:
        raise ValueError("zlib patch step is not the captured executing recipe")
    expected = work / "cpp-build/zlib_ep/src/zlib_ep"
    if args.zlib_source.absolute() != expected or any(path.is_symlink() for path in (expected, *expected.parents)):
        raise ValueError("zlib patch source is outside the owned ExternalProject")
    check_record(inputs["patch_tool"], "zlib patch tool")
    check_record(inputs["dependencies"]["zlib"]["archive"], "zlib archive")
    check_record(inputs["zlib_patch"], "zlib patch")
    apply_zlib_patch(expected, work / "inputs" / DEPENDENCIES["zlib"]["filename"],
                     work / "zlib-df84af25.patch", work / "zlib-patch-application.json")
    return 0


def main(argv: list | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--apply-zlib-patch":
        return private_zlib_main(argv[1:])
    recipe_bytes = Path(__file__).read_bytes()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("openssl-prefix", "archive", "work", "builder-receipt", "simdjson-archive", "uriparser-archive", "boost-archive", "zlib-archive"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("builder-receipt-sha256", "python", "cc", "cxx"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--openssl-source-archive", type=Path)
    parser.add_argument("--zlib-patch", type=Path, default=ZLIB_PATCH_PATH)
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
        openssl_archive = args.openssl_source_archive.absolute() if args.openssl_source_archive else prefix.parent / "downloads/openssl.tar.gz"
        openssl = validate_openssl(prefix, openssl_archive)
        if sha256(archive) != ARROW_ARCHIVE_SHA256:
            raise ValueError("Arrow source archive checksum mismatch")
        archives = {name: getattr(args, name + "_archive").absolute() for name in DEPENDENCIES}
        for name, path in archives.items():
            check_record({"path": str(path), "sha256": DEPENDENCIES[name]["sha256"]}, name + " archive")
        check_record({"path": str(args.zlib_patch.absolute()), "sha256": ZLIB_PATCH_SHA256}, "zlib patch")
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
        (work / "inputs").mkdir(exist_ok=True)
        dependencies = {}
        for name, path in archives.items():
            copied = preserve(work / "inputs" / DEPENDENCIES[name]["filename"], path.read_bytes(),
                              resume=args.resume, label=name + " source archive")
            dependencies[name] = {**DEPENDENCIES[name], "archive": copied}
        zlib_patch = preserve(work / "zlib-df84af25.patch", args.zlib_patch.read_bytes(),
                              resume=args.resume, label="zlib patch")
        context = {"python": args.python, "cc": args.cc, "cxx": args.cxx,
                   "openssl_prefix": str(prefix), "work": str(work)}
        commands = build_commands(source, work, prefix, python=args.python, cc=args.cc, cxx=args.cxx, jobs=args.jobs,
                                   cmake=builder["toolchain"]["cmake"]["path"], ninja=builder["toolchain"]["ninja"]["path"])
        tool_versions = {name: subprocess.check_output([row["path"], "--version"], text=True).splitlines()[0]
                         for name, row in builder["toolchain"].items()}
        inputs = {
            "schema_version": 1, "source_policy": "arrow26-integrated-v1", "arrow_commit": ARROW_COMMIT,
            "source_archive": file_record(archive), "source_verification": file_record(work / "source-verification.json"),
            "integrated_sources": source_verification["integrated_sources"], "recipe": recipe_record,
            "builder": builder, "observed_builder": observed_builder, "toolchain_versions": tool_versions,
            "openssl": openssl, "dependencies": dependencies, "zlib_patch": zlib_patch,
            "patch_tool": file_record(Path("/usr/bin/patch")), "commands": commands, "context": context,
            "compiler_budget": {"maximum_jobs": args.jobs, "outer_ninja_jobs": args.jobs, "nested_cmake_jobs": 1, "nested_make_jobs": 1},
            "features": FEATURES, "codecs": CODECS,
            "version_environment": {"SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYARROW": "{wheel_version}"},
            "python_feature_environment": {key: value for key, value in env.items() if key.startswith("PYARROW_")},
            "generated_version_policy": "frozen vcs-versioning 2.6.0 default Python template; None SCM node; exact derived version",
            "resume_limit": "Generated Python source from failed wheel builds is preserved and requires a fresh generation.",
            "qualification": "unqualified host-specific candidate; all selected system sources and advisory gaps require independent review",
        }
        manifest, version = preserve_build_inputs(work, inputs, resume=args.resume)
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
        env.update(GIT_CEILING_DIRECTORIES=str(work), OPENSSL_ROOT_DIR=str(prefix), PKG_CONFIG_PATH=str(prefix / "lib/pkgconfig"),
                   ARROW_HOME=str(work / "arrow-prefix"), CMAKE_PREFIX_PATH=str(work / "arrow-prefix") + ";" + str(prefix),
                   SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYARROW=version, LDFLAGS="-Wl,-Map,%", TMPDIR=str(work / "tmp"))
        env["PATH"] = os.pathsep.join(dict.fromkeys([str(Path(row["path"]).parent) for row in builder["toolchain"].values()] +
                                                   [str(Path(args.python).parent), "/usr/bin", "/bin"]))
        for name, dependency in dependencies.items():
            env[dependency["env"]] = dependency["archive"]["path"]
        (work / "tmp").mkdir(exist_ok=True)
        if args.configure_only:
            command = [item.replace("{wheel_version}", version) if item == "-DBIOMINER_WHEEL_VERSION={wheel_version}" else item
                       for item in commands["cpp_configure"]]
            record["stages"].append(run_stage(work, "configure", command, env))
            verify_cache(work / "cpp-build/CMakeCache.txt", context, stage="cpp")
            configured = {"schema_version": 1, "build_inputs": manifest,
                          "cache": file_record(work / "cpp-build/CMakeCache.txt"),
                          "dependencies": capture_dependencies(work, compiled=False)}
            configured_record = preserve(work / "configured-inputs.json", canonical_json(configured), resume=False, label="configured inputs")
            record.update(status="configured_unverified", configured_inputs=configured_record)
            return_code = 0
        else:
            record["stages"].append(run_stage(work, "cpp-build", commands["cpp_build"], env))
            dependencies_compiled = capture_dependencies(work, compiled=True)
            source_compilation = compile_source_evidence(work)
            record["stages"].append(run_stage(work, "python-build", commands["python_build"], env))
            verify_cache(work / "python-build/CMakeCache.txt", context, stage="python")
            validate_builder(args.builder_receipt.absolute(), args.builder_receipt_sha256,
                             python=args.python, cc=args.cc, cxx=args.cxx)
            if validate_openssl(prefix, openssl_archive) != openssl:
                raise ValueError("approved OpenSSL inputs changed during build")
            with tarfile.open(archive) as bundle:
                root, _ = archive_members(bundle)
                final_sources = verify_archive_tree(bundle, source, source_transformations(bundle, root),
                    {"python/pyarrow/_generated_version.py": expected_version})
            maps = verify_link_maps(work, prefix, dependencies_compiled)
            wheel = inspect_wheel(work, record)
            dependency_record = preserve(work / "compiled-dependencies.json", canonical_json(dependencies_compiled), resume=False, label="compiled dependencies")
            final_source_record = preserve(work / "compiled-sources.json", canonical_json({"files": final_sources, **source_compilation}), resume=False, label="compiled sources")
            record = {
                "schema_version": 1, "status": "compiled_unverified", "name": "pyarrow", **wheel,
                "wheel_sha256": wheel["wheel"]["sha256"], "source_revision": ARROW_COMMIT,
                "recipe": recipe_record, "build_inputs": manifest, "source_verification": inputs["source_verification"],
                "patch_application": file_record(work / "zlib-patch-application.json"),
                "compiled_dependencies": dependency_record, "compiled_sources": final_source_record,
                "native_link_artifacts": maps["maps"], "linked_static_archives": maps["linked_static_archives"],
                "toolchain": builder["toolchain"], "toolchain_versions": tool_versions,
                "python_build_distributions": builder["distributions"],
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
