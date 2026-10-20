#!/usr/bin/env python3
"""Build the Linux cp314 Arrow wheel from approved source and static OpenSSL.

This is a host-specific build, not a manylinux portability claim. The source
archive is pinned; CMake's pinned dependency downloads retain upstream hashes.
Admission still requires independent library/feature verification by the runtime
profile. A completed compile receipt alone does not authorize production use.
"""
from __future__ import annotations

import argparse
import email
import hashlib
import json
import os
import platform
import re
import shutil
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

ARROW_COMMIT = "beccec0d0c451b7aa3e4530416ac431b3c035c69"
ARROW_ARCHIVE_SHA256 = "a2cb8711cb74251ae8c7e9051f4a85b5034fd689867e202d774e417d532bdcf7"
SECURITY_PATCH_COMMIT = "a769c291e01093b73d03a075179cf7a09bf92ad8"
SECURITY_PATCH_SHA256 = "d51722f4ddb3934a0c846b691a60a730cb77d5a814d7077cab1c5d145c54aaa1"
SECURITY_PATCH_PATH = Path(__file__).parent / "native_patches" / "arrow25-a769c291.patch"
SECURITY_PATCH_FILES = {
    "cpp/src/arrow/compute/kernels/vector_replace.cc": {
        "original_sha256": "201ee04949830bdd07c1f2f87efda6201b39e6aad2d6a76d8dd868e77d6f156e",
        "patched_sha256": "ddd28a89eb41b79e3dc5ecec4cc0e26eb7cec535516a05831b625c8b598afd13",
    },
    "cpp/src/arrow/compute/kernels/vector_replace_test.cc": {
        "original_sha256": "24fdcfad027471f6833dc6efd1f07ba04a3adab7f1edb7aa485c9d2175a28e73",
        "patched_sha256": "2f926df7f6fbce5f7c9c0a32575e4eda39f6c446fa25ff0fa083046b63bb4360",
    },
}
OFFSET_PATCH_COMMIT = "0efbdec7a10da1a4043fc1210c61e73c88bc9b4f"
OFFSET_PATCH_SHA256 = "6c64e115aa53621ae7e1992417b26ab7c6c51fb265d3e64216580f7dcae87d50"
OFFSET_PATCH_PATH = Path(__file__).parent / "native_patches" / "arrow25-0efbdec7.patch"
OFFSET_PATCH_FILES = {
    "cpp/src/arrow/compute/kernels/copy_data_internal.h": {
        "original_sha256": "d1f2979d5a4d4f73f4ab0a40043d377962871f0d85ff3b4a494bcbf94c948c99",
        "patched_sha256": "21b1f8acce63e0ca301a968339d731eddb5efd8e8eeb1a55638ffe665a046621",
    },
    "cpp/src/arrow/compute/kernels/vector_replace_test.cc": {
        "original_sha256": "2f926df7f6fbce5f7c9c0a32575e4eda39f6c446fa25ff0fa083046b63bb4360",
        "patched_sha256": "c087392bbd8114e903608b36c1e4e88a087677cbcfda484cb270d227b385788e",
    },
}
FEATURES = ("ACERO", "AZURE", "COMPUTE", "CSV", "DATASET", "FILESYSTEM", "FLIGHT", "GCS", "HDFS", "JSON", "ORC", "PARQUET", "S3", "SUBSTRAIT")
TOOLCHAIN_PATCHES = (
    ("list(APPEND AWSSDK_PRODUCTS aws-lc s2n-tls)",
     "list(APPEND AWSSDK_PRODUCTS s2n-tls)\n    add_library(crypto ALIAS OpenSSL::Crypto)"),
    ("  fetchcontent_makeavailable(cares)",
     "  fetchcontent_makeavailable(cares)\n  set_target_properties(c-ares PROPERTIES UNITY_BUILD OFF)"),
    ("  fetchcontent_makeavailable(grpc)",
     "  set(CMAKE_UNITY_BUILD OFF)\n  fetchcontent_makeavailable(grpc)"),
    ('set(MAKE_BUILD_ARGS "-j${NPROC}")', 'set(MAKE_BUILD_ARGS "-j1")'),
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path.absolute()), "sha256": sha256(path)}


def preserve_recipe(work: Path, source_bytes: bytes, *, resume: bool) -> dict:
    """Capture the executing recipe once; a resume must use identical bytes."""
    path = work / "build-recipe.py"
    if resume:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != source_bytes:
            raise ValueError("recipe changed since build preparation")
    else:
        with path.open("xb") as stream:
            stream.write(source_bytes)
    return file_record(path)


def preserve_build_inputs(work: Path, inputs: dict, *, resume: bool) -> tuple:
    """The exact immutable manifest bytes determine the custom wheel suffix."""
    path = work / "build-inputs.json"
    data = (json.dumps(inputs, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if resume:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise ValueError("build inputs changed since preparation")
    else:
        with path.open("xb") as stream:
            stream.write(data)
    record = file_record(path)
    return record, "25.0.1+biominer." + record["sha256"][:12]


def build_environment(jobs: int) -> dict:
    if not 1 <= jobs <= 4:
        raise ValueError("--jobs must be between 1 and 4")
    # The explicit outer Ninja limit is jobs. Each ExternalProject is serial:
    # otherwise four outer edges could each start four more compilers.
    return dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL="1", MAKEFLAGS="-j1")


def verify_source_layout(source: Path, members: list) -> None:
    """Require exactly the archive paths, types and symlink targets."""
    expected = {}
    for member in members:
        relative = Path(member.name).relative_to(source.name)
        if relative == Path("."):
            continue
        if not (member.isdir() or member.isfile() or member.issym()):
            raise ValueError("unsupported Arrow archive member")
        expected[relative] = member
        for parent in relative.parents:
            if parent != Path("."):
                expected.setdefault(parent, None)
    actual = set()
    for directory, folders, files in os.walk(source, followlinks=False):
        actual.update((Path(directory) / name).relative_to(source)
                      for name in folders + files)
    if actual != set(expected):
        raise ValueError("source layout changed since extraction")
    for relative, member in expected.items():
        path = source / relative
        if member is not None and member.issym():
            valid = path.is_symlink() and os.readlink(path) == member.linkname
        elif member is None or member.isdir():
            valid = not path.is_symlink() and path.is_dir()
        else:
            valid = not path.is_symlink() and path.is_file()
        if not valid:
            raise ValueError(f"source layout changed since extraction: {relative}")


def prepare_source(archive: Path, work: Path, patch: Path, *, resume: bool,
                   offset_patch: Path | None = None) -> tuple:
    """Accept only the original archive plus every exact reviewed transformation."""
    if sha256(archive) != ARROW_ARCHIVE_SHA256:
        raise ValueError("Arrow source archive checksum mismatch")
    stages = (
        (SECURITY_PATCH_COMMIT, SECURITY_PATCH_SHA256, patch, SECURITY_PATCH_FILES,
         "arrow-a769c291.patch"),
        (OFFSET_PATCH_COMMIT, OFFSET_PATCH_SHA256, offset_patch or OFFSET_PATCH_PATH,
         OFFSET_PATCH_FILES, "arrow-0efbdec7.patch"),
    )
    patch_bytes, original_hashes, final_hashes = {}, {}, {}
    for commit, expected_hash, path, files, preserved_name in stages:
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError("Arrow security patch checksum mismatch")
        patch_bytes[preserved_name] = data
        for relative, pins in files.items():
            if relative in final_hashes and final_hashes[relative] != pins["original_sha256"]:
                raise ValueError("reviewed patch chain is discontinuous")
            original_hashes.setdefault(relative, pins["original_sha256"])
            final_hashes[relative] = pins["patched_sha256"]
    work = work.absolute()
    if any(path.is_symlink() for path in (work, *work.parents)):
        raise ValueError("work path contains a symlink")
    if work.exists() and (not work.is_dir() or (not resume and any(work.iterdir()))):
        raise ValueError("work directory already exists and is not empty")
    if resume:
        for name in ("build-receipt.json", "cpp-build", "arrow-prefix", "wheelhouse",
                     "wheel-inventory", "configure.log", "cpp-build.log", "python-build.log"):
            if (work / name).is_symlink():
                raise ValueError(f"resume output path is a symlink: {name}")
        receipt = work / "build-receipt.json"
        if receipt.exists() or receipt.is_symlink():
            if receipt.is_symlink() or not receipt.is_file():
                raise ValueError("resume cannot modify completed or unrecognized build receipt")
            previous = json.loads(receipt.read_bytes())
            if not isinstance(previous, dict) or previous.get("status") not in {"building", "failed"}:
                raise ValueError("resume cannot modify completed or unrecognized build receipt")
        wheels = work / "wheelhouse"
        if wheels.is_symlink() or (wheels.exists() and any(wheels.glob("*.whl"))):
            raise ValueError("resume cannot modify completed wheel output")
    source = work / f"arrow-{ARROW_COMMIT}"
    if source.exists() and not resume:
        raise ValueError("source directory already exists; use --resume or a fresh build generation")
    if resume and (source.is_symlink() or not source.is_dir()):
        raise ValueError("resume requires an existing source directory")
    work.mkdir(parents=True, exist_ok=True)
    for name, data in patch_bytes.items():
        preserved_patch = work / name
        if resume:
            if preserved_patch.is_symlink() or not preserved_patch.is_file() or preserved_patch.read_bytes() != data:
                raise ValueError("preserved security patch changed since extraction")
        else:
            with preserved_patch.open("xb") as stream:
                stream.write(data)
    toolchain_relative = "cpp/cmake_modules/ThirdpartyToolchain.cmake"
    original_toolchain = None
    patched_toolchain = None
    found_security_files = set()
    with tarfile.open(archive) as bundle:
        if not resume:
            bundle.extractall(work, filter="data")
        verify_source_layout(source, bundle.getmembers())
        for member in bundle.getmembers():
            if not member.isfile():
                continue
            relative = Path(member.name).relative_to(source.name).as_posix()
            stream = bundle.extractfile(member)
            assert stream is not None
            expected = stream.read()
            if relative == toolchain_relative:
                original_toolchain = expected
                patched_toolchain = expected
                for before, after in TOOLCHAIN_PATCHES:
                    if patched_toolchain.count(before.encode()) != 1:
                        raise ValueError("Arrow toolchain patch context mismatch")
                    patched_toolchain = patched_toolchain.replace(before.encode(), after.encode())
                if resume:
                    expected = patched_toolchain
            current = source / relative
            if current.is_symlink() or not current.is_file():
                raise ValueError(f"source changed since extraction: {relative}")
            if relative in original_hashes:
                found_security_files.add(relative)
                if hashlib.sha256(expected).hexdigest() != original_hashes[relative]:
                    raise ValueError(f"original security source checksum mismatch: {relative}")
                expected_hash = (final_hashes if resume else original_hashes)[relative]
                valid = sha256(current) == expected_hash
            else:
                valid = current.read_bytes() == expected
            if not valid:
                raise ValueError(f"source changed since extraction: {relative}")
    if found_security_files != set(original_hashes) or original_toolchain is None:
        raise ValueError("reviewed source files missing from Arrow archive")
    if not resume:
        (source / toolchain_relative).write_bytes(patched_toolchain)
        for commit, expected_hash, path, files, preserved_name in stages:
            for relative, pins in files.items():
                if sha256(source / relative) != pins["original_sha256"]:
                    raise ValueError(f"patch chain input checksum mismatch: {relative}")
            result = subprocess.run(
                ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(work / preserved_name)],
                cwd=source, capture_output=True, text=True,
            )
            output = result.stdout + result.stderr
            if result.returncode or any(word in output.lower() for word in ("offset", "fuzz", "failed", "reversed")):
                raise ValueError("Arrow security patch did not apply exactly")
            for relative, pins in files.items():
                if sha256(source / relative) != pins["patched_sha256"]:
                    raise ValueError(f"patch chain output checksum mismatch: {relative}")
    for relative, expected_hash in final_hashes.items():
        if sha256(source / relative) != expected_hash:
            raise ValueError(f"patched security source checksum mismatch: {relative}")
    toolchain_record = {
        "file": toolchain_relative,
        "original_sha256": hashlib.sha256(original_toolchain).hexdigest(),
        "patched_sha256": sha256(source / toolchain_relative),
        "before": TOOLCHAIN_PATCHES[0][0], "after": TOOLCHAIN_PATCHES[0][1],
        "build_workarounds": [{"before": before, "after": after}
                             for before, after in TOOLCHAIN_PATCHES[1:]],
    }
    security_record = {
        "schema_version": 1, "source_revision": ARROW_COMMIT,
        "patches": [{
            "commit": commit, "patch": file_record(work / preserved_name),
            "files": [{"relative": relative, **pins} for relative, pins in files.items()],
        } for commit, expected_hash, path, files, preserved_name in stages],
        "final_sources": [{"relative": relative, "sha256": final_hashes[relative]}
                          for relative in sorted(final_hashes)],
    }
    receipt = work / "arrow-security-patches.json"
    if resume:
        if receipt.is_symlink() or not receipt.is_file() or json.loads(receipt.read_bytes()) != security_record:
            raise ValueError("security patch receipt changed since extraction")
    else:
        with receipt.open("x") as stream:
            json.dump(security_record, stream, sort_keys=True, indent=2)
            stream.write("\n")
    return source, toolchain_record, security_record



def completed_receipt(work: Path, archive: Path, prefix: Path, python: str, build_record: dict) -> dict:
    """Bind the real wheel and its complete resolved ELF closure to build inputs."""
    wheels = list((work / "wheelhouse").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("expected exactly one built Arrow wheel")
    wheel = wheels[0]
    unpack = work / "wheel-inventory"
    unpack.mkdir(exist_ok=True)
    with zipfile.ZipFile(wheel) as bundle:
        bundle.extractall(unpack)
        metadata_name = next(name for name in bundle.namelist() if name.endswith(".dist-info/METADATA"))
        version = email.message_from_bytes(bundle.read(metadata_name))["Version"]
    file_record = lambda path: {"path": str(Path(path).absolute()), "sha256": sha256(Path(path))}
    native_files = {}
    wheel_elfs = {}
    for path in unpack.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                if stream.read(4) != b"\x7fELF":
                    continue
            wheel_elfs[path.resolve()] = path.relative_to(unpack).as_posix()
    if not wheel_elfs:
        raise RuntimeError("wheel contains no native ELF")
    if version != build_record.get("wheel_version"):
        raise RuntimeError("wheel version does not match immutable build inputs")
    if file_record(work / "build-inputs.json") != build_record.get("build_inputs"):
        raise RuntimeError("build input manifest changed during compilation")
    linkage = {}
    for path, member in wheel_elfs.items():
        native_files[str(path)] = {"path": "{generation}/env/lib/python3.14/site-packages/" + member,
                                  "sha256": sha256(path), "owner": "wheel:pyarrow"}
        result = subprocess.run(["ldd", str(path)], capture_output=True, text=True, check=True)
        if "not found" in result.stdout:
            raise RuntimeError(f"unresolved ELF dependency in {member}: {result.stdout}")
        linkage[member] = result.stdout
        dependencies = re.findall(r"=> (/\S+)", result.stdout)
        dependencies += re.findall(r"^\s*(/\S+) \(", result.stdout, re.MULTILINE)
        for dependency in dependencies:
            resolved = Path(dependency).resolve()
            if resolved in wheel_elfs or str(resolved) in native_files:
                continue
            owner = subprocess.run(["dpkg-query", "-S", str(resolved)], capture_output=True, text=True)
            if owner.returncode:
                owner = subprocess.run(["dpkg-query", "-S", dependency], capture_output=True, text=True)
            if owner.returncode:
                raise RuntimeError(f"unapproved non-wheel native dependency: {resolved}")
            package = owner.stdout.split(": ", 1)[0].split(":", 1)[0]
            native_files[str(resolved)] = file_record(resolved) | {"owner": "dpkg:" + package}
    input_files = [archive, prefix / "include/openssl/opensslv.h",
                   prefix / "lib/libcrypto.a", prefix / "lib/libssl.a",
                   work / f"arrow-{ARROW_COMMIT}/cpp/cmake_modules/ThirdpartyToolchain.cmake",
                   work / "build-recipe.py", work / "arrow-a769c291.patch", work / "arrow-0efbdec7.patch",
                   work / "arrow-security-patches.json", work / "build-inputs.json"]
    input_files.extend(work / f"arrow-{ARROW_COMMIT}" / row["relative"]
                       for row in build_record["security_patches"]["final_sources"])
    openssl_receipt = prefix.parent / "openssl-build-receipt.json"
    if not openssl_receipt.is_file():
        raise RuntimeError("OpenSSL source build receipt missing")
    input_files.append(openssl_receipt)
    distribution_probe = "import importlib.metadata as m,json; print(json.dumps([{'name':d.metadata['Name'],'version':d.version,'record':str(d._path/'RECORD'),'metadata':str(d._path/'METADATA')} for d in m.distributions()],sort_keys=True))"
    distributions = json.loads(subprocess.check_output([python, "-c", distribution_probe], text=True))
    for distribution in distributions:
        for field in ("record", "metadata"):
            path = Path(distribution[field])
            if not path.is_file():
                raise RuntimeError(f"build distribution lacks {field}: {distribution['name']}")
            input_files.append(path)
    input_records = [file_record(path) for path in input_files]
    toolchain = {tool: file_record(Path(shutil.which(tool)).resolve()) for tool in ("cmake", "ninja", "gcc", "g++", "ld")}
    toolchain["python"] = file_record(Path(python))
    maps = [path for path in work.rglob("*.map")
            if "Linker script and memory map" in path.read_text(errors="replace")]
    if not maps:
        raise RuntimeError("build produced no per-output linker maps")
    return {
        "schema_version": 1, "status": "compiled_unverified", "name": "pyarrow", "version": version,
        "wheel": file_record(wheel), "wheel_sha256": sha256(wheel), "source_revision": ARROW_COMMIT,
        "toolchain": toolchain, "toolchain_versions": build_record.get("toolchain", {}),
        "python_build_distributions": distributions,
        "inputs": input_records,
        "inputs_sha256": hashlib.sha256(json.dumps(input_records, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "recipe": file_record(work / "build-recipe.py"),
        "patch_application": file_record(work / "arrow-security-patches.json"),
        "build_inputs": file_record(work / "build-inputs.json"),
        "native_link_artifacts": [file_record(path) for path in maps],
        "effective_versions": {"pyarrow": version, "openssl": "3.6.5"},
        "native_files": list(native_files.values()), "linkage": linkage, "build": build_record,
        "qualification": "compiled host-specific candidate; TLS, geometry, host patch and runtime admission still required",
    }


def main() -> int:
    recipe_bytes = Path(__file__).read_bytes()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openssl-prefix", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--security-patch", type=Path, default=SECURITY_PATCH_PATH)
    parser.add_argument("--offset-patch", type=Path, default=OFFSET_PATCH_PATH)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true", help="reuse a verified source/build generation")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error("--jobs must be between 1 and 4")
    prefix = args.openssl_prefix.resolve()
    header = prefix / "include/openssl/opensslv.h"
    if not header.is_file() or '"OpenSSL 3.6.5 ' not in header.read_text():
        parser.error("OpenSSL 3.6.5 required; reject an unpatched or missing prefix")
    for library in ("libcrypto.a", "libssl.a"):
        if not (prefix / "lib" / library).is_file():
            parser.error(f"approved static OpenSSL archive missing: {library}")
    if args.check:
        return 0
    if args.archive is None or args.work is None:
        parser.error("--archive and --work required for a build")
    archive = args.archive.resolve()
    work = args.work.absolute()
    try:
        source, patch_receipt, security_patches = prepare_source(
            archive, work, args.security_patch.resolve(), resume=args.resume,
            offset_patch=args.offset_patch.resolve(),
        )
        recipe = preserve_recipe(work, recipe_bytes, resume=args.resume)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    install = work / "arrow-prefix"
    build = work / "cpp-build"
    approved_python = str(Path(args.python).absolute())
    flags = [f"-DARROW_{feature}=ON" for feature in FEATURES]
    flags += [f"-D{name}={approved_python}" for name in ("Python3_EXECUTABLE", "Python_EXECUTABLE", "PYTHON_EXECUTABLE")]
    flags += [
        "-DARROW_BUILD_SHARED=ON", "-DARROW_BUILD_STATIC=OFF",
        "-DARROW_BUILD_TESTS=OFF", "-DARROW_DEPENDENCY_SOURCE=AUTO",
        "-DARROW_DEPENDENCY_USE_SHARED=OFF", "-DARROW_RPATH_ORIGIN=ON",
        "-DARROW_WITH_BROTLI=ON", "-DARROW_WITH_BZ2=ON",
        "-DARROW_WITH_LZ4=ON", "-DARROW_WITH_SNAPPY=ON",
        "-DARROW_WITH_ZLIB=ON", "-DARROW_WITH_ZSTD=ON",
        "-DPARQUET_REQUIRE_ENCRYPTION=ON", "-DARROW_JEMALLOC=ON",
        "-DARROW_MIMALLOC=ON", "-DARROW_GANDIVA=OFF",
        "-DOPENSSL_USE_STATIC_LIBS=TRUE", f"-DOPENSSL_ROOT_DIR={prefix}",
        f"-DCMAKE_PREFIX_PATH={prefix}", "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_LIBDIR=lib", f"-DCMAKE_INSTALL_PREFIX={install}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DCMAKE_UNITY_BUILD=ON",
        "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-Map,%",
        "-DCMAKE_MODULE_LINKER_FLAGS=-Wl,-Map,%",
    ]
    env = build_environment(args.jobs)
    # Archive builds must not discover BioMiner's enclosing Git checkout.
    env["GIT_CEILING_DIRECTORIES"] = str(work)
    env["OPENSSL_ROOT_DIR"] = str(prefix)
    env["PKG_CONFIG_PATH"] = str(prefix / "lib/pkgconfig")
    commands = [
        ["cmake", "-S", str(source / "cpp"), "-B", str(build), "-G", "Ninja", *flags],
        ["cmake", "--build", str(build), "--target", "install", "--parallel", str(args.jobs)],
    ]
    inputs = {
        "schema_version": 1, "status": "building", "jobs": args.jobs, "arrow_commit": ARROW_COMMIT,
        "arrow_archive_sha256": ARROW_ARCHIVE_SHA256, "features": FEATURES,
        "recipe": recipe, "security_patches": security_patches,
        "source_archive": file_record(archive),
        "patch_application": file_record(work / "arrow-security-patches.json"),
        "patched_sources": [file_record(source / row["relative"])
                            for row in security_patches["final_sources"]],
        "compiler_budget": {
            "maximum_jobs": args.jobs, "outer_ninja_jobs": args.jobs,
            "nested_cmake_jobs": 1, "nested_make_jobs": 1,
            "make_override": 'set(MAKE_BUILD_ARGS "-j1")',
        },
        "openssl_archives": {name: sha256(prefix / "lib" / name) for name in ("libcrypto.a", "libssl.a")},
        "cmake_flags": flags, "commands": commands, "crypto_patch": patch_receipt,
        "platform": platform.platform(), "machine": platform.machine(),
        "toolchain": {tool: subprocess.check_output([tool, "--version"], text=True).splitlines()[0]
                      for tool in ("cmake", "ninja", "gcc", "g++")},
        "python": subprocess.check_output([args.python, "-c", "import sys,ssl; print(sys.version); print(ssl.OPENSSL_VERSION)"], text=True),
        "qualification": "unverified host-specific build; runtime admission required",
    }
    try:
        build_inputs, wheel_version = preserve_build_inputs(work, inputs, resume=args.resume)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    inputs["build_inputs"] = build_inputs
    inputs["wheel_version"] = wheel_version
    receipt = work / "build-receipt.json"
    receipt.write_text(json.dumps(inputs, indent=2) + "\n")
    try:
        for stage, command in zip(("configure", "cpp-build"), commands, strict=True):
            with (work / f"{stage}.log").open("w") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        cache = (build / "CMakeCache.txt").read_text()
        for name in ("OPENSSL_CRYPTO_LIBRARY", "OPENSSL_SSL_LIBRARY"):
            matches = [line for line in cache.splitlines() if line.startswith(name + ":")]
            if not matches or any(str(prefix / "lib") not in line or not line.endswith(".a") for line in matches):
                raise RuntimeError(f"CMake did not bind {name} to approved static prefix")
        env.update(ARROW_HOME=str(install), CMAKE_PREFIX_PATH=str(install), PYARROW_BUNDLE_ARROW_CPP="ON")
        for feature in FEATURES:
            env[f"PYARROW_WITH_{feature}"] = "ON"
        env["PYARROW_WITH_PARQUET_ENCRYPTION"] = "ON"
        env["LDFLAGS"] = "-Wl,-Map,%"
        env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYARROW"] = wheel_version
        command = [args.python, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(work / "wheelhouse"), str(source / "python")]
        inputs["commands"].append(command)
        with (work / "python-build.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        inputs["wheels"] = {str(wheel): sha256(wheel) for wheel in (work / "wheelhouse").glob("*.whl")}
        if not inputs["wheels"]:
            raise RuntimeError("build returned without producing a wheel")
        inputs = completed_receipt(work, archive, prefix, args.python, inputs)
    except (subprocess.CalledProcessError, RuntimeError) as error:
        inputs["status"] = "failed"
        inputs["error"] = str(error)
        print(str(error), file=sys.stderr)
        return 1
    finally:
        receipt.write_text(json.dumps(inputs, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
