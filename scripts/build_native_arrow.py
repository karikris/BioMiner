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
FEATURES = ("ACERO", "AZURE", "COMPUTE", "CSV", "DATASET", "FILESYSTEM", "FLIGHT", "GCS", "HDFS", "JSON", "ORC", "PARQUET", "S3", "SUBSTRAIT")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()



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
                   work / f"arrow-{ARROW_COMMIT}/cpp/cmake_modules/ThirdpartyToolchain.cmake"]
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
        "native_link_artifacts": [file_record(path) for path in maps],
        "effective_versions": {"pyarrow": version, "openssl": "3.6.5"},
        "native_files": list(native_files.values()), "linkage": linkage, "build": build_record,
        "qualification": "compiled host-specific candidate; TLS, geometry, host patch and runtime admission still required",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openssl-prefix", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true", help="reuse a verified source/build generation")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8:
        parser.error("--jobs must be between 1 and 8")
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
    if sha256(archive) != ARROW_ARCHIVE_SHA256:
        parser.error("Arrow source archive checksum mismatch")
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    source = work / f"arrow-{ARROW_COMMIT}"
    if source.exists() and not args.resume:
        parser.error("source directory already exists; use --resume or a fresh build generation")
    with tarfile.open(archive) as bundle:
        if source.exists():
            # Compare every original source file, permitting only our exact
            # recorded crypto patch. This prevents an unreviewed tree on resume.
            for member in bundle.getmembers():
                if not member.isfile():
                    continue
                stream = bundle.extractfile(member)
                assert stream is not None
                expected = stream.read()
                if member.name.endswith("cpp/cmake_modules/ThirdpartyToolchain.cmake"):
                    expected = expected.replace(
                        b"list(APPEND AWSSDK_PRODUCTS aws-lc s2n-tls)",
                        b"list(APPEND AWSSDK_PRODUCTS s2n-tls)\n    add_library(crypto ALIAS OpenSSL::Crypto)",
                    )
                current = work / member.name
                variants = [expected]
                if member.name.endswith("cpp/cmake_modules/ThirdpartyToolchain.cmake"):
                    variants.append(expected.replace(
                        b"  fetchcontent_makeavailable(cares)",
                        b"  fetchcontent_makeavailable(cares)\n  set_target_properties(c-ares PROPERTIES UNITY_BUILD OFF)",
                    ))
                if member.name.endswith("cpp/cmake_modules/ThirdpartyToolchain.cmake"):
                    variants.append(variants[-1].replace(
                        b"  fetchcontent_makeavailable(grpc)",
                        b"  set(CMAKE_UNITY_BUILD OFF)\n  fetchcontent_makeavailable(grpc)",
                    ))
                if not current.is_file() or current.read_bytes() not in variants:
                    parser.error(f"source changed since extraction: {member.name}")
        else:
            bundle.extractall(work, filter="data")
    toolchain = source / "cpp/cmake_modules/ThirdpartyToolchain.cmake"
    original = toolchain.read_text()
    # Linux upstream bundles AWS-LC for s2n. Bind that crypto target to our
    # approved archive instead; aws-c-cal already selects OpenSSL explicitly.
    before = "list(APPEND AWSSDK_PRODUCTS aws-lc s2n-tls)"
    after = "list(APPEND AWSSDK_PRODUCTS s2n-tls)\n    add_library(crypto ALIAS OpenSSL::Crypto)"
    if args.resume:
        if original.count(after) != 1:
            parser.error("Arrow AWS crypto patch context mismatch")
    else:
        if original.count(before) != 1:
            parser.error("Arrow AWS crypto patch context mismatch")
        toolchain.write_text(original.replace(before, after))
    cares_before = "  fetchcontent_makeavailable(cares)"
    cares_after = cares_before + "\n  set_target_properties(c-ares PROPERTIES UNITY_BUILD OFF)"
    patched = toolchain.read_text()
    if cares_after not in patched:
        if patched.count(cares_before) != 1:
            parser.error("c-ares unity workaround context mismatch")
        toolchain.write_text(patched.replace(cares_before, cares_after))
    grpc_before = "  fetchcontent_makeavailable(grpc)"
    grpc_after = "  set(CMAKE_UNITY_BUILD OFF)\n" + grpc_before
    patched = toolchain.read_text()
    if grpc_after not in patched:
        if patched.count(grpc_before) != 1:
            parser.error("gRPC unity workaround context mismatch")
        toolchain.write_text(patched.replace(grpc_before, grpc_after))
    patch_receipt = {
        "file": "cpp/cmake_modules/ThirdpartyToolchain.cmake",
        "before": before, "after": after,
        "patched_sha256": sha256(toolchain),
        "build_workarounds": [{"before": cares_before, "after": cares_after},
                              {"before": grpc_before, "after": grpc_after}],
    }
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
    # All nested builds inherit the same bound, including Make ExternalProjects.
    env = dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs), MAKEFLAGS=f"-j{args.jobs}")
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
        "openssl_archives": {name: sha256(prefix / "lib" / name) for name in ("libcrypto.a", "libssl.a")},
        "cmake_flags": flags, "commands": commands, "crypto_patch": patch_receipt,
        "platform": platform.platform(), "machine": platform.machine(),
        "toolchain": {tool: subprocess.check_output([tool, "--version"], text=True).splitlines()[0]
                      for tool in ("cmake", "ninja", "gcc", "g++")},
        "python": subprocess.check_output([args.python, "-c", "import sys,ssl; print(sys.version); print(ssl.OPENSSL_VERSION)"], text=True),
        "qualification": "unverified host-specific build; runtime admission required",
    }
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
        input_digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:12]
        env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYARROW"] = f"25.0.1+biominer.{input_digest}"
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
