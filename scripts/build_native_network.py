#!/usr/bin/env python3
"""Build one source-bound static curl/XML prefix; never admit a runtime.

Inputs are local and pinned. The work directory must be new; failures remain
available for inspection and are never resumed or overwritten. --check validates
inputs without creating work. The resulting receipt is compiled_unverified:
downstream Arrow linkage, advisories, capabilities and runtime admission still
require independent review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time


SOURCES = {
    "curl": {"version": "8.23.0", "root": "curl-8.23.0", "filename": "curl-8.23.0.tar.xz",
             "sha256": "8dadd7d2536a314f982ad05a651b713f426bf7d02cc2c76cffb9796f2218e6c5",
             "commit": "3163ec065643ba20709f190111f42c3e1fe95273", "url": "https://curl.se/download/curl-8.23.0.tar.xz"},
    "zlib": {"version": "1.3.2", "root": "zlib-1.3.2", "filename": "zlib-1.3.2.tar.gz",
             "sha256": "bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16", "url": "https://zlib.net/zlib-1.3.2.tar.gz"},
    "libiconv": {"version": "1.19", "root": "libiconv-1.19", "filename": "libiconv-1.19.tar.gz",
                 "sha256": "88dd96a8c0464eca144fc791ae60cd31cd8ee78321e67397e25fc095c4a19aa6", "url": "https://ftp.gnu.org/pub/gnu/libiconv/libiconv-1.19.tar.gz"},
    "libxml2": {"version": "2.15.4", "root": "libxml2-2.15.4", "filename": "libxml2-2.15.4.tar.xz",
                "sha256": "98087fd181d9070724f3fbc65c7377db03038eb92bd882374daff44940138821", "url": "https://download.gnome.org/sources/libxml2/2.15/libxml2-2.15.4.tar.xz"},
    "nghttp2": {"version": "1.68.0", "root": "nghttp2-1.68.0", "filename": "nghttp2_1.68.0.orig.tar.gz",
                "sha256": "2c16ffc588ad3f9e2613c3fad72db48ecb5ce15bc362fcc85b342e48daf51013",
                "url": "https://api.launchpad.net/1.0/ubuntu/+archive/primary/+sourcepub/18550650"},
}
ZLIB_PATCH_COMMIT = "df84af25dc1942490e1d1c899a07619152a46148"
ZLIB_PATCH_SHA256 = "110ff14375733173d8aa54574473424fbd7dfe4b81f1ca34a759c6fe14b15b14"
ZLIB_ORIGINAL_GZWRITE_SHA256 = "2ce783294c688330c716bfe106f43167d3dae5be84994176bc304d205d4f5b04"
ZLIB_PATCHED_GZWRITE_SHA256 = "31d0da140edf382b82b13e2d87a22f59e158d9014ee026459e01cf437d0ab07d"
OPENSSL_FILES = {
    "include/openssl/opensslv.h": "80d9b9534fdbdab12c29cb8c62887ede14bbb7112b93e838a63d6815400f5f0b",
    "lib/libcrypto.a": "e2d8161275df2868b63e28d8d1f5b49efa3e40a3697d1a21a07890c58a2d3bc2",
    "lib/libssl.a": "3ac9b8f49ce67cc03e9a1073754faaec92226195d610bd59800e4df5c41a82b6",
}
OPENSSL_RECEIPT_SHA256 = "66d41e5fa8cee1cdb865a7df41620685776c96fd8eba9abcc6a70505ca3da13a"
OPENSSL_ARCHIVE_SHA256 = "f3091b7ccb41a11d877b7406b7afc9dd9c5003d299c6bb33cff882127bc06c91"
NGHTTP2_SOURCE_VERSION = "1.68.0-2ubuntu0.2"
NGHTTP2_PUBLICATION = "https://api.launchpad.net/1.0/ubuntu/+archive/primary/+sourcepub/18550650"
NGHTTP2_DIST_INPUTS = {
    "nghttp2_debian": {"filename": "nghttp2_1.68.0-2ubuntu0.2.debian.tar.xz", "size": 18304,
                       "sha256": "3b25d46909e26fa04d6d81d5ff7032b094b33e086ef2060889a2d029cc75f54f"},
    "nghttp2_dsc": {"filename": "nghttp2_1.68.0-2ubuntu0.2.dsc", "size": 2868,
                    "sha256": "45a9d539b2abc000cbe6d8b40a1d26f10c29550e59393c33acc84e5fc3919e69"},
    "nghttp2_signature": {"filename": "nghttp2_1.68.0.orig.tar.gz.asc", "size": 833,
                          "sha256": "be8b3ff79aa9f374cb5b1127ac686520e0291627e4e9c5faac6d730640dc3155"},
}
NGHTTP2_PATCH_STAGES = [
    {"name": "0002-Workaround-for-963648.patch", "sha256": "4fe75c6f2192dacaf4cb120bcb58798b90dedc2132fdb404629a715878f528eb", "files": {
        "doc/_exts/rubydomain/rubydomain.py": {"before": "7aa1ff9ff33d58d088513859f4aca3908718643c9a95db4d6ae92ef26ad14d86", "after": "aabbb2a20744f7dcf1195ff9453eb20b12049dfbe7884bf9675e74e90dc3c1a0"}}},
    {"name": "CVE-2026-27135.patch", "sha256": "1e76f98cf16e42236d98ebbf4bc41caf49869efecd6fb8ea0eef52e54509b9cb", "files": {
        "lib/nghttp2_session.c": {"before": "ecc071771e0b310b51d927b31bcdd4f89ed1fa5c1785c1ae5ad1a4fd2a369542", "after": "7a999fc52572b73c518fd158cfc1cd381d915fb624f3d4808c54b08269a4b2cf"}}},
    {"name": "CVE-2026-58055.patch", "sha256": "ed11cd1968301c8ca9478dc6657bcffb31f07b528794c0184e859447e1ac61fc", "files": {
        "src/shrpx_downstream.cc": {"before": "8397d76201b72a8b5abbdfa6c7a8a74226fe34172c81def1cd4b941a7513a069", "after": "d343115184c386fc56d2124b5c630b8adc00939927b16ffda3de3f67158178e5"},
        "src/shrpx_downstream.h": {"before": "0a9de104b29da60de9e8eed2b3637b809ba281cff50d93c3011911d4fc0f9b5c", "after": "89a63e756612cb25c21bd57a1c75d628b0c8fe8dd876d545a32bc884b7473bcb"},
        "src/shrpx_http2_upstream.cc": {"before": "71dcd24ec829d7da718a30a2627e26552de5af88a062288cf8cee7ea3ec11655", "after": "99f88fea9a7b709ba6a27daba888e7b67b76f1fcbde5fbc0da3636428cf40ae0"},
        "src/shrpx_http3_upstream.cc": {"before": "6411fadeda6fdd78ab2ba5eac5fdd44eea042711eb5769cc690fe0629ed92262", "after": "74173704c35ba459518d092d20c5a8d327e521c3dfb6d796ba97bbf273c350a2"},
        "src/shrpx_http_downstream_connection.cc": {"before": "2bfb49b7782b137355bb31acfaf4a6cc28238dea332b8b884500b05a1247868a", "after": "87c4962288d1c134d532b8fd76967b2503156b6675f984025731770007574cc1"},
        "src/shrpx_http_downstream_connection.h": {"before": "9d13d4e3da8927bf196b9d54a23c042a738f2220e8813eb77a24f8b527e25785", "after": "090176782b318d2f69b59941d51d7cc6bdaa064ca5748b0af07ea6beb8af8649"},
        "src/shrpx_https_upstream.cc": {"before": "43dcc5a5a30cdee17c8037e7206847482ccf58b71031c0c4f64750a97c5c788b", "after": "a6103e84fe89ae29f07e1f5f467bfb8f4e39d2900928f7c995df524a275c60b8"}}},
]
PINNED_TOOLS = {
    "cc": ("/usr/bin/x86_64-linux-gnu-gcc-15", "b5f1b773a7c733738352000c92a077dc5852a1a2fc6d836b1e411be1e9ec5f88"),
    "cxx": ("/usr/bin/x86_64-linux-gnu-g++-15", "e6718f7e0c7d057c3ff77b550c603da9bc4030e3ede3c053705acce1293dbe4d"),
    "cmake": ("/usr/bin/cmake", "6e1dccda39845415d68eabb934c598998949c99ec4668625d571aee1827b05c7"),
    "ninja": ("/usr/bin/ninja", "91e9548850cda2799facfdaa7abe33f9e978832f85e98212255708a7bbe437f2"),
    "ld": ("/usr/bin/x86_64-linux-gnu-ld.bfd", "97f48d93b8b076a92d2809ec29dcb17f0f37c8827358f832255e2ed22fef6075"),
}
DISABLED_PROTOCOLS = ("DICT", "FILE", "FTP", "GOPHER", "IMAP", "IPFS", "LDAP", "LDAPS", "MQTT", "POP3", "RTSP", "SMTP", "TELNET", "TFTP", "WEBSOCKETS")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path):
    return {"path": str(Path(path).absolute()), "sha256": sha256(path)}


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def save(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    return record(path)


def checked(path, expected, label):
    path = Path(path).absolute()
    if path.resolve() != path or path.is_symlink():
        raise ValueError(f"{label} symlink or noncanonical path")
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"{label} checksum mismatch: {path}")
    return record(path)


def archive_members(bundle, root):
    members, names, symlinks = bundle.getmembers(), set(), set()
    for member in members:
        path = PurePosixPath(member.name)
        if (path.is_absolute() or ".." in path.parts or not path.parts
                or path.parts[0] != root or str(path) in names):
            raise ValueError("unsafe, wrong-root or duplicate source archive path")
        names.add(str(path))
        if not (member.isfile() or member.isdir() or member.issym()):
            raise ValueError("unsupported source archive member")
        if member.issym():
            target = PurePosixPath(member.linkname)
            normalized = os.path.normpath(str(path.parent / target))
            if target.is_absolute() or not (normalized == root or normalized.startswith(root + "/")):
                raise ValueError("escaping source archive symlink")
            symlinks.add(path)
    if not members:
        raise ValueError("empty source archive")
    if any(any(parent in symlinks for parent in PurePosixPath(item.name).parents) for item in members):
        raise ValueError("source archive member traverses a symlink")
    return members


def inventory(root, *, reject_shared=False):
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise ValueError("inventory root missing, aliased or symlinked")
    entries = []
    def unreadable(error):
        raise ValueError(f"unreadable inventory: {error.filename}") from error
    paths = [root]
    for directory, folders, files in os.walk(root, followlinks=False, onerror=unreadable):
        paths.extend(Path(directory) / name for name in folders + files)
    for path in sorted(paths):
        info = path.lstat()
        relative = path.relative_to(root).as_posix()
        row = {"relative": relative, "mode": stat.S_IMODE(info.st_mode)}
        if info.st_uid != os.getuid():
            raise ValueError("inventory contains a different owner's file")
        if reject_shared and re.search(r"\.so(?:\.|$)", path.name) and not path.is_dir():
            raise ValueError("private prefix contains a shared library or alias")
        if path.is_symlink():
            if not path.resolve().is_relative_to(root) or not path.exists():
                raise ValueError("escaping or broken inventory symlink")
            row.update(kind="symlink", target=os.readlink(path))
        elif stat.S_ISDIR(info.st_mode):
            row.update(kind="directory")
        elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            row.update(kind="file", size=info.st_size, sha256=sha256(path))
        else:
            raise ValueError("inventory contains a special file or hardlink")
        entries.append(row)
    return {"root": str(root), "files": entries}


def prefix_inventory(root):
    return inventory(root, reject_shared=True)


def nghttp2_series():
    return "".join(stage["name"] + "\n" for stage in NGHTTP2_PATCH_STAGES)


def validate_nghttp2_debian(bundle):
    members = archive_members(bundle, "debian")
    by_name = {member.name: member for member in members}
    expected = {"debian/patches/series": nghttp2_series().encode()}
    for name, data in expected.items():
        member = by_name.get(name)
        if member is None or not member.isfile() or bundle.extractfile(member).read() != data:
            raise ValueError("nghttp2 distro patch series/order mismatch")
    for stage in NGHTTP2_PATCH_STAGES:
        member = by_name.get("debian/patches/" + stage["name"])
        if (member is None or not member.isfile()
                or hashlib.sha256(bundle.extractfile(member).read()).hexdigest() != stage["sha256"]):
            raise ValueError("nghttp2 distro patch checksum mismatch")
    return members


def nghttp2_distribution_inputs(directory):
    """Pin the complete DSC checksum closure; no signature claim is inferred."""
    directory = Path(directory).absolute()
    pin = SOURCES["nghttp2"]
    result = {"nghttp2": checked(directory / pin["filename"], pin["sha256"], "nghttp2 source archive")}
    for name, metadata in NGHTTP2_DIST_INPUTS.items():
        path = directory / metadata["filename"]
        result[name] = checked(path, metadata["sha256"], "nghttp2 distro input")
        if path.stat().st_size != metadata["size"]:
            raise ValueError("nghttp2 distro input size mismatch")
    text = Path(result["nghttp2_dsc"]["path"]).read_text()
    if (not re.search(r"^Source: nghttp2$", text, re.M)
            or not re.search(r"^Version: " + re.escape(NGHTTP2_SOURCE_VERSION) + r"$", text, re.M)):
        raise ValueError("nghttp2 distro DSC source identity mismatch")
    section = re.search(r"^Checksums-Sha256:\n((?:[ \t].*\n)+)", text, re.M)
    if section is None:
        raise ValueError("nghttp2 distro DSC SHA256 closure missing")
    checksums = {}
    for line in section.group(1).splitlines():
        digest, size, name = line.split()
        if name in checksums:
            raise ValueError("duplicate nghttp2 distro DSC checksum")
        checksums[name] = (digest, int(size))
    expected = {Path(row["path"]).name: (row["sha256"], Path(row["path"]).stat().st_size)
                for name, row in result.items() if name != "nghttp2_dsc"}
    if checksums != expected:
        raise ValueError("nghttp2 distro DSC checksum/size closure mismatch")
    with tarfile.open(result["nghttp2_debian"]["path"]) as bundle:
        validate_nghttp2_debian(bundle)
    return result


def validate_archive_dir(directory):
    directory = Path(directory).absolute()
    result = {}
    for name, pin in SOURCES.items():
        path = directory / pin["filename"]
        result[name] = checked(path, pin["sha256"], name + " archive")
        with tarfile.open(path) as bundle:
            archive_members(bundle, pin["root"])
    result["zlib_patch"] = checked(directory / "zlib-df84af25.patch", ZLIB_PATCH_SHA256, "zlib patch")
    result.update(nghttp2_distribution_inputs(directory))
    return result


def build_environment(work, tools):
    work = Path(work).absolute()
    return {
        "PATH": "/usr/bin:/bin", "HOME": str(work / "home"), "TMPDIR": str(work / "tmp"),
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "CC": tools["cc"], "CXX": tools["cxx"],
        "AR": tools["ar"], "RANLIB": tools["ranlib"], "CFLAGS": "-O2 -g0 -fPIC",
        "CXXFLAGS": "-O2 -g0 -fPIC", "MAKEFLAGS": "-j1", "CMAKE_BUILD_PARALLEL_LEVEL": "1",
        "CONFIG_SITE": "/dev/null", "PKG_CONFIG_LIBDIR": str(work / "prefix/lib/pkgconfig"),
        "OPENSSL_CONF": "/dev/null", "OPENSSL_MODULES": "/nonexistent",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
    }


def _live_group(pid):
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            candidate = int(directory.name)
            if os.getpgid(candidate) != pid and os.getsid(candidate) != pid:
                continue
            if directory.joinpath("stat").read_text().rsplit(")", 1)[1].split()[0] != "Z":
                return True
        except (ProcessLookupError, FileNotFoundError):
            pass
    return False


def run_command(argv, *, cwd, env, work, name, timeout=1800):
    if not 0 < timeout <= 1800:
        raise ValueError("command timeout must be bounded by1800seconds")
    log = Path(work) / "logs" / (name + ".log")
    process, error, code, cleanup = None, None, None, "not-started"
    started = time.monotonic()
    with log.open("xb") as stream:
        try:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait(timeout=timeout)
            if _live_group(process.pid):
                raise RuntimeError("build command left live process-group/session members")
            cleanup = "terminal"
        except BaseException as caught:
            error = caught
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
                deadline = time.monotonic() + 5
                while _live_group(process.pid) and time.monotonic() < deadline:
                    time.sleep(0.01)
                cleanup = "failed" if _live_group(process.pid) else "terminal"
                code = 124 if isinstance(caught, subprocess.TimeoutExpired) else process.returncode
        finally:
            stream.flush()
            os.fsync(stream.fileno())
    receipt = save(Path(work) / "logs" / (name + "-command.json"), {
        "argv": argv, "cwd": str(Path(cwd).absolute()), "timeout_seconds": timeout,
        "exit_code": code, "cleanup": cleanup, "log": record(log),
        "duration_seconds": round(time.monotonic() - started, 6),
        "error_type": type(error).__name__ if error else None,
    })
    if error is not None:
        raise error
    if code:
        raise subprocess.CalledProcessError(code, argv)
    return receipt


def apply_nghttp2_patches(source, debian, work, archive):
    """Apply all distro patches, including disabled proxy-code fixes, exactly."""
    source, debian, work = Path(source), Path(debian), Path(work)
    checked(archive, SOURCES["nghttp2"]["sha256"], "nghttp2 original source")
    series = debian / "patches/series"
    if series.is_symlink() or series.read_text() != nghttp2_series():
        raise ValueError("nghttp2 distro patch series/order mismatch")
    patches = [checked(debian / "patches" / stage["name"], stage["sha256"], "nghttp2 distro patch")
               for stage in NGHTTP2_PATCH_STAGES]
    _verify_original_source(archive, source, "nghttp2", nghttp2_stages=[])
    original, stages = inventory(source), []
    for index, (stage, patch) in enumerate(zip(NGHTTP2_PATCH_STAGES, patches, strict=True)):
        before = {name: checked(source / name, pin["before"], "nghttp2 patch preimage")["sha256"]
                  for name, pin in stage["files"].items()}
        command = run_command([
            "/usr/bin/patch", "--batch", "--forward", "--fuzz=0", "--no-backup-if-mismatch", "-p1", "-i", patch["path"],
        ], cwd=source, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, work=work,
            name=f"nghttp2-patch-{index + 1}", timeout=30)
        log = json.loads(Path(command["path"]).read_bytes())["log"]
        if re.search(r"offset|fuzz|failed|reversed", Path(log["path"]).read_text(), re.I):
            raise ValueError("nghttp2 patch must apply exactly without fuzz or offsets")
        after = {name: checked(source / name, pin["after"], "nghttp2 patch result")["sha256"]
                 for name, pin in stage["files"].items()}
        _verify_original_source(archive, source, "nghttp2", nghttp2_stages=NGHTTP2_PATCH_STAGES[:index + 1])
        stages.append({"name": stage["name"], "patch": patch, "before_files": before,
                       "after_files": after, "command": command})
    return {"source_package": "nghttp2", "source_version": NGHTTP2_SOURCE_VERSION,
            "archive": record(archive), "debian": inventory(debian), "original_source": original,
            "final_source": inventory(source), "stages": stages}


def prepare_sources(archive_dir, work, *, owned=None):
    work = Path(work).absolute()
    if work.exists() or work.is_symlink():
        raise FileExistsError(f"build work already exists: {work}")
    if work.resolve() != work:
        raise ValueError("canonical work path required")
    archives = validate_archive_dir(archive_dir)
    work.mkdir(parents=True, mode=0o700)
    if owned is not None:
        info = work.lstat()
        owned.update(device=info.st_dev, inode=info.st_ino, uid=info.st_uid)
    for name in ("inputs", "sources", "distribution", "logs", "home", "tmp", "prefix", "build"):
        (work / name).mkdir(mode=0o700)
    copied = {}
    for name, reference in archives.items():
        original = Path(reference["path"])
        target = work / "inputs" / original.name
        shutil.copyfile(original, target)
        copied[name] = checked(target, reference["sha256"], "copied source")
    recipe = work / "build-recipe.py"
    recipe.write_bytes(Path(__file__).read_bytes())
    preparation_inputs = save(work / "preparation-inputs.json", {"recipe": record(recipe), "archives": copied, "patch_tool": record(Path("/usr/bin/patch").resolve())})
    for name, pin in SOURCES.items():
        with tarfile.open(copied[name]["path"]) as bundle:
            archive_members(bundle, pin["root"])
            bundle.extractall(work / "sources", filter="data")
    with tarfile.open(copied["nghttp2_debian"]["path"]) as bundle:
        validate_nghttp2_debian(bundle)
        bundle.extractall(work / "distribution", filter="data")
    nghttp2_application = apply_nghttp2_patches(
        work / "sources" / SOURCES["nghttp2"]["root"], work / "distribution/debian", work,
        copied["nghttp2"]["path"],
    )
    zlib = work / "sources" / SOURCES["zlib"]["root"]
    checked(zlib / "gzwrite.c", ZLIB_ORIGINAL_GZWRITE_SHA256, "original zlib gzwrite source")
    patch_command = run_command([
        "/usr/bin/patch", "--batch", "--forward", "--fuzz=0", "--no-backup-if-mismatch", "-p1", "-i", copied["zlib_patch"]["path"],
    ], cwd=zlib, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, work=work, name="zlib-patch", timeout=30)
    if re.search(r"offset|fuzz|failed|reversed", (work / "logs/zlib-patch.log").read_text(), re.I):
        raise ValueError("zlib patch must apply exactly without fuzz or offsets")
    checked(zlib / "gzwrite.c", ZLIB_PATCHED_GZWRITE_SHA256, "patched zlib gzwrite source")
    application = {"patch_commit": ZLIB_PATCH_COMMIT, "patch": copied["zlib_patch"],
                   "archive": copied["zlib"], "original_gzwrite_sha256": ZLIB_ORIGINAL_GZWRITE_SHA256,
                   "patched_gzwrite_sha256": ZLIB_PATCHED_GZWRITE_SHA256, "command": patch_command}
    prepared = {"archives": copied, "recipe": record(recipe), "preparation_inputs": preparation_inputs, "patch_application": application,
                "nghttp2_patch_application": nghttp2_application,
                "sources": {name: inventory(work / "sources" / pin["root"]) for name, pin in SOURCES.items()}}
    save(work / "prepared-sources.json", prepared)
    return prepared


def verify_sources(prepared):
    for source in prepared["sources"].values():
        if inventory(source["root"]) != source:
            raise ValueError("prepared source tree changed during build")
    debian = prepared["nghttp2_patch_application"]["debian"]
    if inventory(debian["root"]) != debian:
        raise ValueError("nghttp2 distro source metadata changed during build")
    for reference in [*prepared["archives"].values(), prepared["recipe"]]:
        checked(reference["path"], reference["sha256"], "retained source input")


def validate_openssl(prefix, archive):
    prefix = Path(prefix).absolute()
    files = [checked(prefix / name, value, "approved private OpenSSL") for name, value in OPENSSL_FILES.items()]
    receipt = checked(prefix.parent / "openssl-build-receipt.json", OPENSSL_RECEIPT_SHA256, "OpenSSL source receipt")
    source = checked(archive, OPENSSL_ARCHIVE_SHA256, "OpenSSL source archive")
    return {"version": "3.6.5", "source_commit": "c8bd5a57108599ac650bbae77fcabe3109dab2e8",
            "prefix": str(prefix), "files": files, "headers": inventory(prefix / "include"),
            "receipt": receipt, "source_archive": source}


def validate_nghttp2_source(path, expected, archive_dir):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("explicit nghttp2 source receipt SHA256 required")
    reference = checked(path, expected, "nghttp2 source receipt")
    value = json.loads(Path(path).read_bytes())
    if (value.get("schema") != "biominer-nghttp2-distro-source-access/v1"
            or value.get("source_package") != "nghttp2" or value.get("source_version") != NGHTTP2_SOURCE_VERSION
            or value.get("source_publication") != NGHTTP2_PUBLICATION):
        raise ValueError("unapproved nghttp2 distro source identity")
    distribution = nghttp2_distribution_inputs(archive_dir)
    if checked(value["dsc"]["path"], distribution["nghttp2_dsc"]["sha256"], "nghttp2 source DSC") != value["dsc"]:
        raise ValueError("nghttp2 source DSC binding mismatch")
    rows = value["dsc_sha256_and_sizes_checked"]
    approved = {Path(row["path"]).name: row for name, row in distribution.items() if name != "nghttp2_dsc"}
    if len(rows) != len(approved) or {Path(row["path"]).name for row in rows} != set(approved):
        raise ValueError("nghttp2 source DSC input closure incomplete")
    for row in rows:
        digest = approved[Path(row["path"]).name]["sha256"]
        checked(row["path"], digest, "nghttp2 source DSC input")
        if row["sha256"] != digest or row["size"] != Path(row["path"]).stat().st_size:
            raise ValueError("nghttp2 source DSC input identity mismatch")
    if value["patch_series"] != nghttp2_series() or len(value["patches"]) != len(NGHTTP2_PATCH_STAGES):
        raise ValueError("nghttp2 source patch series/order mismatch")
    for row, stage in zip(value["patches"], NGHTTP2_PATCH_STAGES, strict=True):
        checked(row["path"], stage["sha256"], "nghttp2 source patch evidence")
        if row["name"] != stage["name"] or row["sha256"] != stage["sha256"]:
            raise ValueError("nghttp2 source patch identity mismatch")
    header_names = ("nghttp2.h", "nghttp2ver.h.in")
    if len(value["source_headers"]) != len(header_names):
        raise ValueError("nghttp2 source header evidence incomplete")
    with tarfile.open(distribution["nghttp2"]["path"]) as bundle:
        archive_members(bundle, SOURCES["nghttp2"]["root"])
        for row, name in zip(value["source_headers"], header_names, strict=True):
            original = bundle.extractfile(SOURCES["nghttp2"]["root"] + "/lib/includes/nghttp2/" + name).read()
            if Path(row["path"]).name != name or row["sha256"] != hashlib.sha256(original).hexdigest():
                raise ValueError("nghttp2 original source header identity mismatch")
            checked(row["path"], row["sha256"], "nghttp2 original source header evidence")
    signature = value["signature"]
    for key in ("keyring", "log"):
        checked(signature[key]["path"], signature[key]["sha256"], "nghttp2 signature evidence")
    if signature["exit_code"] != 2:
        raise ValueError("nghttp2 source signature limitation differs from reviewed evidence")
    return {"receipt": reference, "source_package": "nghttp2", "source_version": NGHTTP2_SOURCE_VERSION,
            "version": "1.68.0", "source_publication": NGHTTP2_PUBLICATION,
            "distribution_inputs": distribution, "source_access": value,
            "signature_verified": False,
            "qualification": "Primary HTTPS publication and exact DSC checksum closure; gpgv could not verify the signing key. Private source build, whole-archive PIC probe and independent review required."}


def verify_nghttp2_outputs(work):
    """Bind installed HTTP2 outputs to this source/build, never system binaries."""
    work = Path(work)
    source = work / "sources" / SOURCES["nghttp2"]["root"] / "lib/includes/nghttp2"
    build = work / "build/nghttp2"
    archive_path = build / "lib/libnghttp2.a"
    archive = checked(archive_path, sha256(archive_path), "nghttp2 built archive")
    header, template = record(source / "nghttp2.h"), record(source / "nghttp2ver.h.in")
    generated = build / "lib/includes/nghttp2/nghttp2ver.h"
    expected = Path(template["path"]).read_bytes().replace(b"@PACKAGE_VERSION@", b"1.68.0").replace(b"@PACKAGE_VERSION_NUM@", b"0x014400")
    if generated.is_symlink() or generated.read_bytes() != expected:
        raise ValueError("nghttp2 generated version header differs from source template")
    installed = [checked(work / "prefix/lib/libnghttp2.a", archive["sha256"], "nghttp2 source-built archive"),
                 checked(work / "prefix/include/nghttp2/nghttp2.h", header["sha256"], "nghttp2 source public header"),
                 checked(work / "prefix/include/nghttp2/nghttp2ver.h", sha256(generated), "nghttp2 generated installed header")]
    source_manifest = source.parent.parent / "CMakeLists.txt"
    definition = re.search(r"set\(NGHTTP2_SOURCES\s+(.*?)\s*\)", source_manifest.read_text(), re.S)
    names = definition.group(1).split() if definition else []
    if not names or len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9_]+\.c", name) for name in names):
        raise ValueError("nghttp2 literal static target source manifest missing")
    expected_sources = {str(source_manifest.parent / name) for name in names}
    target = build / "lib/CMakeFiles/nghttp2_static.dir"
    objects = [checked(path, sha256(path), "nghttp2 target object") for path in sorted(target.rglob("*.o"))]
    compile_commands = checked(build / "compile_commands.json", sha256(build / "compile_commands.json"), "nghttp2 compile commands")
    rows = json.loads(Path(compile_commands["path"]).read_bytes())
    if not isinstance(rows, list) or not rows or not objects:
        raise ValueError("nghttp2 source compile/object closure missing")
    outputs, compiled = set(), set()
    for row in rows:
        argv = row.get("arguments") if isinstance(row.get("arguments"), list) else shlex.split(row.get("command", ""))
        if (not argv or not all(isinstance(arg, str) for arg in argv) or argv[0] != PINNED_TOOLS["cc"][0]
                or "-fPIC" not in argv or any(arg.lower() in ("-fno-pic", "-fpie", "-fno-pie") for arg in argv)
                or not {"-DBUILDING_NGHTTP2", "-DNGHTTP2_STATICLIB", "-c", "-o"} <= set(argv)
                or row.get("directory") != str(build)):
            raise ValueError("nghttp2 compile command lacks pinned compiler/static PIC policy")
        source_path = row.get("file")
        if (source_path not in expected_sources or source_path in compiled
                or argv.index("-c") + 1 >= len(argv) or argv[argv.index("-c") + 1] != source_path
                or argv.index("-o") + 1 >= len(argv)):
            raise ValueError("nghttp2 compile command source closure mismatch")
        output = Path(argv[argv.index("-o") + 1])
        if not output.is_absolute():
            output = build / output
        expected_output = target / (Path(source_path).name + ".o")
        if output != expected_output or str(output) in outputs:
            raise ValueError("nghttp2 compile target object mapping mismatch")
        compiled.add(source_path)
        outputs.add(str(output))
    if compiled != expected_sources or outputs != {row["path"] for row in objects}:
        raise ValueError("nghttp2 complete source/target object closure mismatch")
    return {"archive": archive, "public_header": header, "version_template": template,
            "generated_version_header": record(generated), "installed": installed,
            "objects": objects, "compile_commands": compile_commands,
            "target_source_manifest": record(source_manifest), "compiled_sources": [record(path) for path in sorted(compiled)]}


def validate_tools():
    result = {}
    for name, (path, digest) in PINNED_TOOLS.items():
        result[name] = checked(path, digest, "pinned " + name)
    for name in ("make", "ar", "ranlib", "patch", "readelf", "nm", "install"):
        found = shutil.which(name, path="/usr/bin:/bin")
        if not found:
            raise ValueError("required native tool missing: " + name)
        result[name] = record(Path(found).resolve())
    for row in result.values():
        observed = subprocess.run([row["path"], "--version"], env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, capture_output=True, timeout=30, check=True)
        row["version_output"] = (observed.stdout + observed.stderr).decode()
    return result


def build_plan(work, tools, openssl_prefix, *, jobs, http2):
    if isinstance(jobs, bool) or not 1 <= jobs <= 4:
        raise ValueError("jobs must be between1and4")
    work, crypto = Path(work), Path(openssl_prefix)
    prefix, build, sources = work / "prefix", work / "build", work / "sources"
    common = ["-GNinja", "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_C_COMPILER={tools['cc']}",
              f"-DCMAKE_MAKE_PROGRAM={tools['ninja']}",
              f"-DCMAKE_INSTALL_PREFIX={prefix}", "-DCMAKE_INSTALL_LIBDIR=lib", "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
              "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]
    plan = []
    def add(name, argv, cwd=work, timeout=1800):
        plan.append({"name": name, "argv": [str(arg) for arg in argv], "cwd": str(cwd), "timeout": timeout})
    def cmake(name, options):
        directory = build / name
        add(name + "-configure", [tools["cmake"], "-S", sources / SOURCES[name]["root"], "-B", directory, *common, *options], timeout=300)
        add(name + "-build", [tools["cmake"], "--build", directory, "--parallel", str(jobs)])
        add(name + "-install", [tools["cmake"], "--install", directory], timeout=300)
    cmake("zlib", ["-DZLIB_BUILD_TESTING=OFF", "-DZLIB_BUILD_SHARED=OFF", "-DZLIB_BUILD_STATIC=ON", "-DZLIB_INSTALL=ON"])
    iconv = build / "libiconv"
    add("libiconv-configure", [sources / SOURCES["libiconv"]["root"] / "configure", "--disable-shared", "--enable-static", "--with-pic", "--disable-nls", f"--prefix={prefix}", f"--libdir={prefix / 'lib'}"], iconv, 300)
    add("libiconv-charset-bootstrap", [tools["make"], "-j1", "lib/localcharset.h"], iconv)
    add("libiconv-build", [tools["make"], "-C", iconv / "lib", f"-j{jobs}", "all"], iconv)
    for component in ("libcharset", "lib"):
        add("libiconv-install-" + component, [tools["make"], "-C", iconv / component, "-j1", "install-lib", f"libdir={prefix / 'lib'}", f"includedir={prefix / 'include'}"], iconv, 300)
    add("libiconv-install-header", [tools.get("install", "/usr/bin/install"), "-m", "644", iconv / "include/iconv.h.inst", prefix / "include/iconv.h"], iconv, 30)
    zlib = [f"-DZLIB_LIBRARY={prefix / 'lib/libz.a'}", f"-DZLIB_INCLUDE_DIR={prefix / 'include'}", "-DZLIB_USE_STATIC_LIBS=ON"]
    cmake("libxml2", ["-DBUILD_SHARED_LIBS=OFF", "-DLIBXML2_WITH_ICU=OFF", "-DLIBXML2_WITH_ICONV=ON", "-DIconv_IS_BUILT_IN=FALSE",
                      f"-DIconv_LIBRARY={prefix / 'lib/libiconv.a'}", f"-DIconv_INCLUDE_DIR={prefix / 'include'}",
                      "-DLIBXML2_WITH_ZLIB=ON", "-DLIBXML2_WITH_PROGRAMS=OFF", "-DLIBXML2_WITH_TESTS=OFF",
                      "-DLIBXML2_WITH_PYTHON=OFF", "-DLIBXML2_WITH_DOCS=OFF", *zlib])
    cmake("nghttp2", ["-DBUILD_SHARED_LIBS=OFF", "-DBUILD_STATIC_LIBS=ON", "-DENABLE_LIB_ONLY=ON",
                     "-DBUILD_TESTING=OFF", "-DENABLE_APP=OFF", "-DENABLE_HPACK_TOOLS=OFF",
                     "-DENABLE_EXAMPLES=OFF", "-DENABLE_DOC=OFF", "-DENABLE_FAILMALLOC=OFF", "-DENABLE_HTTP3=OFF",
                     "-DWITH_LIBXML2=OFF", "-DWITH_JEMALLOC=OFF", "-DWITH_MRUBY=OFF", "-DWITH_NEVERBLEED=OFF",
                     "-DWITH_LIBBPF=OFF", "-DWITH_WOLFSSL=OFF",
                     *[f"-DCMAKE_DISABLE_FIND_PACKAGE_{name}=TRUE" for name in
                       ("Python3", "OpenSSL", "Libngtcp2", "Libnghttp3", "Systemd", "Jansson", "Libevent", "LibXml2", "Jemalloc")]])
    options = ["-DBUILD_SHARED_LIBS=OFF", "-DBUILD_STATIC_LIBS=ON", "-DBUILD_CURL_EXE=OFF", "-DBUILD_TESTING=OFF", "-DBUILD_EXAMPLES=OFF",
               "-DBUILD_LIBCURL_DOCS=OFF", "-DBUILD_MISC_DOCS=OFF", "-DCURL_USE_CMAKECONFIG=OFF", "-DCURL_USE_PKGCONFIG=OFF",
               "-DCURL_ENABLE_SSL=ON", "-DCURL_USE_OPENSSL=ON", "-DOPENSSL_USE_STATIC_LIBS=TRUE",
               f"-DOPENSSL_ROOT_DIR={crypto}", f"-DOPENSSL_INCLUDE_DIR={crypto / 'include'}",
               f"-DOPENSSL_SSL_LIBRARY={crypto / 'lib/libssl.a'}", f"-DOPENSSL_CRYPTO_LIBRARY={crypto / 'lib/libcrypto.a'}",
               "-DCURL_DISABLE_OPENSSL_AUTO_LOAD_CONFIG=ON", "-DCURL_DISABLE_HTTP=OFF", "-DCURL_ZLIB=ON", *zlib,
               "-DCURL_BROTLI=OFF", "-DCURL_ZSTD=OFF", "-DUSE_LIBIDN2=OFF", "-DENABLE_ARES=OFF",
               "-DCURL_USE_LIBPSL=OFF", "-DCURL_USE_LIBSSH2=OFF", "-DCURL_USE_LIBSSH=OFF", "-DCURL_USE_GSSAPI=OFF",
               "-DCURL_USE_GSASL=OFF", "-DUSE_NGTCP2=OFF", "-DUSE_QUICHE=OFF", "-DUSE_PROXY_HTTP3=OFF",
               "-DCURL_USE_MBEDTLS=OFF", "-DCURL_USE_WOLFSSL=OFF", "-DCURL_USE_GNUTLS=OFF", "-DCURL_USE_RUSTLS=OFF",
               "-DCURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt", "-DCURL_CA_PATH=/etc/ssl/certs",
               *[f"-DCURL_DISABLE_{name}=ON" for name in DISABLED_PROTOCOLS]]
    if http2:
        options += ["-DUSE_NGHTTP2=ON", "-DNGHTTP2_USE_STATIC_LIBS=ON", f"-DNGHTTP2_LIBRARY={http2['archive']}", f"-DNGHTTP2_INCLUDE_DIR={http2['include']}"]
    else:
        options += ["-DUSE_NGHTTP2=OFF"]
    cmake("curl", options)
    return plan


def validate_cache(path, expected):
    values = {}
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        key = line.split(":", 1)[0]
        if key in values:
            raise ValueError("duplicate CMake cache key: " + key)
        values[key] = line.split("=", 1)[1]
    for key, value in expected.items():
        if values.get(key) != str(value):
            raise ValueError("CMake cache rejected " + key)
    return record(path)


def protected_native_name(path):
    return bool(re.fullmatch(r"lib(?:curl(?:-[^.]+)?|xml2|z|iconv|charset|ssl|crypto|nghttp2)"
                             r"(?:\.a|\.so(?:\.[0-9]+)*)", Path(path).name))


def validate_linkage(dynamic, maps, archives):
    if "TEXTREL" in dynamic or any(protected_native_name(name) for name in re.findall(r"\[([^\]\n]+)\]", dynamic)):
        raise ValueError("dynamic or text-relocated private native dependency")
    expected = {str(Path(path).absolute()) for path in archives}
    text = "\n".join(Path(path).read_text() for path in maps)
    consumed = {path for path in expected if path + "(" in text}
    if consumed != expected:
        raise ValueError("private archive member not consumed by link maps")
    loaded = {line[5:].strip() for line in text.splitlines() if line.startswith("LOAD ")}
    for path in loaded:
        if protected_native_name(path) and path not in expected:
            raise ValueError("foreign private native provider in link map: " + path)
    for match in re.finditer(r"(?:^|\s)([^\s()]+\.a)\(", text, re.M):
        path = match.group(1)
        if protected_native_name(path) and path not in expected:
            raise ValueError("foreign private static archive member: " + path)
    return {"consumed_private_archives": sorted(consumed), "maps": [record(path) for path in maps], "dynamic_private_dependencies": []}


def _verify_original_source(archive, source, name, *, nghttp2_stages=None):
    """Compare every original member and only explicitly pinned transforms."""
    source = Path(source)
    transforms = {}
    if name == "nghttp2":
        for stage in NGHTTP2_PATCH_STAGES if nghttp2_stages is None else nghttp2_stages:
            for relative, pin in stage["files"].items():
                previous = transforms.get(relative)
                if previous is not None and previous["after"] != pin["before"]:
                    raise ValueError("nghttp2 source patch chain mismatch")
                transforms[relative] = {"before": previous["before"] if previous else pin["before"], "after": pin["after"]}
    if name == "zlib":
        transforms["gzwrite.c"] = {"before": ZLIB_ORIGINAL_GZWRITE_SHA256, "after": ZLIB_PATCHED_GZWRITE_SHA256}
    root = "debian" if name == "nghttp2_debian" else SOURCES[name]["root"]
    with tarfile.open(archive) as bundle:
        members = archive_members(bundle, root)
        expected = {"."}
        for member in members:
            relative = PurePosixPath(member.name).relative_to(root)
            expected.add(str(relative))
            expected.update(str(parent) for parent in relative.parents)
            path = source / str(relative)
            if member.isfile():
                original = hashlib.sha256(bundle.extractfile(member).read()).hexdigest()
                transform = transforms.get(str(relative))
                current = transform["after"] if transform else original
                if (path.is_symlink() or not path.is_file() or sha256(path) != current
                        or transform is not None and original != transform["before"]):
                    raise ValueError("original source member mismatch: " + str(path))
            elif member.issym():
                if not path.is_symlink() or os.readlink(path) != member.linkname:
                    raise ValueError("original source symlink mismatch")
            elif not path.is_dir() or path.is_symlink():
                raise ValueError("original source directory mismatch")
        if {row["relative"] for row in inventory(source)["files"]} != expected:
            raise ValueError("original source archive layout mismatch")


def verify_nghttp2_patch_application(application, work, archives):
    """Pure check of the source transformation and all ordered command evidence."""
    work = Path(work)
    source, debian = work / "sources" / SOURCES["nghttp2"]["root"], work / "distribution/debian"
    if (application["source_package"] != "nghttp2" or application["source_version"] != NGHTTP2_SOURCE_VERSION
            or application["archive"] != archives["nghttp2"] or application["debian"] != inventory(debian)
            or application["final_source"] != inventory(source) or len(application["stages"]) != len(NGHTTP2_PATCH_STAGES)):
        raise ValueError("nghttp2 source/patch application binding mismatch")
    _verify_original_source(archives["nghttp2_debian"]["path"], debian, "nghttp2_debian")
    _verify_original_source(archives["nghttp2"]["path"], source, "nghttp2")
    original = json.loads(json.dumps(application["final_source"]))
    with tarfile.open(archives["nghttp2"]["path"]) as bundle:
        for row in original["files"]:
            if row["kind"] == "file":
                data = bundle.extractfile(SOURCES["nghttp2"]["root"] + "/" + row["relative"]).read()
                row.update(sha256=hashlib.sha256(data).hexdigest(), size=len(data))
    if original != application["original_source"]:
        raise ValueError("nghttp2 original full-source inventory mismatch")
    for index, (row, stage) in enumerate(zip(application["stages"], NGHTTP2_PATCH_STAGES, strict=True)):
        patch = checked(debian / "patches" / stage["name"], stage["sha256"], "nghttp2 retained distro patch")
        command = json.loads(Path(row["command"]["path"]).read_bytes())
        argv = ["/usr/bin/patch", "--batch", "--forward", "--fuzz=0", "--no-backup-if-mismatch", "-p1", "-i", patch["path"]]
        if (row["name"] != stage["name"] or row["patch"] != patch
                or row["before_files"] != {name: pin["before"] for name, pin in stage["files"].items()}
                or row["after_files"] != {name: pin["after"] for name, pin in stage["files"].items()}
                or row["command"]["path"] != str(work / "logs" / f"nghttp2-patch-{index + 1}-command.json")
                or command["argv"] != argv or command["cwd"] != str(source) or command["timeout_seconds"] != 30
                or command["exit_code"] != 0 or command["cleanup"] != "terminal"
                or re.search(r"offset|fuzz|failed|reversed", Path(command["log"]["path"]).read_text(), re.I)):
            raise ValueError("nghttp2 ordered zero-fuzz patch command evidence mismatch")


def verify_completed_prefix(receipt_path, expected_sha256):
    """Pure byte/policy verification of a previously reviewed build descriptor.

    Never starts a subprocess, builds, writes, admits a runtime or refreshes a
    receipt. The caller must separately pin this captured helper's exact bytes
    and approve the concrete build receipt hash. Returned file references are
    suitable for a downstream before/after dependency closure check.
    """
    try:
        return _verify_completed_prefix(receipt_path, expected_sha256)
    except (KeyError, TypeError, OSError, tarfile.TarError) as error:
        raise ValueError("invalid private network closure: " + str(error)) from error


def _verify_completed_prefix(receipt_path, expected_sha256):
    receipt_ref = checked(receipt_path, expected_sha256, "private network receipt")
    value = json.loads(Path(receipt_path).read_bytes())
    if (value.get("schema_version") != 1 or value.get("kind") != "private_network_static_build"
            or value.get("status") != "compiled_unverified" or value.get("ready") is not False):
        raise ValueError("completed compiled_unverified private network receipt required")
    work = Path(receipt_ref["path"]).parent
    prefix = work / "prefix"
    if value.get("prefix") != str(prefix) or Path(receipt_path).name != "build-receipt.json" or (work / "FAILED.json").exists():
        raise ValueError("private prefix path or failed build state rejected")
    for key, name in (("input_manifest", "build-inputs.json"), ("recipe", "build-recipe.py"),
                      ("prefix_inventory", "prefix-inventory.json"), ("build_inventory", "build-inventory.json")):
        if value[key]["path"] != str(work / name):
            raise ValueError("private build control path mismatch")
    files, parsed = {}, {}
    def consume(reference):
        if not isinstance(reference, dict) or not isinstance(reference.get("sha256"), str):
            raise ValueError("private network file reference required")
        actual = checked(reference["path"], reference["sha256"], "private network closure")
        previous = files.setdefault(actual["path"], actual["sha256"])
        if previous != actual["sha256"]:
            raise ValueError("conflicting private network file reference")
        return actual
    def walk(document):
        if isinstance(document, dict):
            if "path" in document and "sha256" in document:
                actual = consume(document)
                path = Path(actual["path"])
                if path.suffix == ".json" and path.is_relative_to(work) and actual["path"] not in parsed:
                    nested = json.loads(path.read_bytes())
                    parsed[actual["path"]] = nested
                    walk(nested)
            for child in document.values():
                walk(child)
        elif isinstance(document, list):
            for child in document:
                walk(child)
    consume(receipt_ref)
    parsed[receipt_ref["path"]] = value
    walk(value)
    def read(reference):
        return json.loads(Path(consume(reference)["path"]).read_bytes())
    def tree(reference, actual_root, *, frozen=False):
        expected = read(reference)
        actual = prefix_inventory(actual_root) if frozen else inventory(actual_root)
        if expected != actual:
            raise ValueError("private prefix/build inventory changed")
        for row in actual["files"]:
            if frozen and row["kind"] != "symlink" and row["mode"] & 0o222:
                raise ValueError("private prefix is writable")
            if row["kind"] == "file":
                consume({"path": str(Path(actual_root) / row["relative"]), "sha256": row["sha256"]})
        return actual
    if value["recipe"]["sha256"] != sha256(Path(__file__)):
        raise ValueError("private prefix helper differs from captured build recipe")
    inputs = read(value["input_manifest"])
    prepared = read(inputs["prepared_sources"])
    if (inputs.get("schema_version") != 1 or inputs.get("kind") != "private_network_build_inputs"
            or inputs.get("source_pins") != SOURCES or inputs["recipe"] != value["recipe"]
            or set(prepared["sources"]) != set(SOURCES)
            or value["sources"] != prepared["sources"] or value["source_archives"] != prepared["archives"]
            or value["zlib_patch"] != prepared["patch_application"] or value["nghttp2_patch"] != prepared["nghttp2_patch_application"]
            or prepared["recipe"] != value["recipe"]):
        raise ValueError("private network source/input manifest mismatch")
    if prepared["preparation_inputs"]["path"] != str(work / "preparation-inputs.json"):
        raise ValueError("private source preparation manifest path mismatch")
    preparation_inputs = read(prepared["preparation_inputs"])
    if preparation_inputs["recipe"] != value["recipe"] or preparation_inputs["archives"] != prepared["archives"]:
        raise ValueError("private source preparation input binding mismatch")
    if inputs["probe_source"] != value["probe"]["source"] or Path(inputs["probe_source"]["path"]).read_bytes() != PROBE.encode():
        raise ValueError("private network probe source changed")
    archive_inputs = validate_archive_dir(work / "inputs")
    if archive_inputs != value["source_archives"]:
        raise ValueError("private source archive set mismatch")
    verify_nghttp2_patch_application(value["nghttp2_patch"], work, archive_inputs)
    patch = value["zlib_patch"]
    if (patch["patch_commit"] != ZLIB_PATCH_COMMIT or patch["patch"] != archive_inputs["zlib_patch"]
            or patch["archive"] != archive_inputs["zlib"] or patch["original_gzwrite_sha256"] != ZLIB_ORIGINAL_GZWRITE_SHA256
            or patch["patched_gzwrite_sha256"] != ZLIB_PATCHED_GZWRITE_SHA256):
        raise ValueError("private zlib patch chain mismatch")
    verify_sources(prepared)
    for name, source in prepared["sources"].items():
        if source["root"] != str(work / "sources" / SOURCES[name]["root"]):
            raise ValueError("private source path mismatch")
        _verify_original_source(archive_inputs[name]["path"], source["root"], name)
        for row in source["files"]:
            if row["kind"] == "file":
                consume({"path": str(Path(source["root"]) / row["relative"]), "sha256": row["sha256"]})
    openssl, nghttp2 = value["openssl"], value["nghttp2"]
    if (inputs["openssl"] != openssl or inputs["nghttp2"] != nghttp2
            or validate_openssl(openssl["prefix"], openssl["source_archive"]["path"]) != openssl
            or validate_nghttp2_source(nghttp2["receipt"]["path"], nghttp2["receipt"]["sha256"], work / "inputs") != nghttp2):
        raise ValueError("private OpenSSL/HTTP2 source-provider binding changed")
    if value["nghttp2_build"] != verify_nghttp2_outputs(work):
        raise ValueError("nghttp2 private source-build output binding changed")
    for row in value["nghttp2_patch"]["debian"]["files"]:
        if row["kind"] == "file":
            consume({"path": str(work / "distribution/debian" / row["relative"]), "sha256": row["sha256"]})
    for row in openssl["headers"]["files"]:
        if row["kind"] == "file":
            consume({"path": str(Path(openssl["headers"]["root"]) / row["relative"]), "sha256": row["sha256"]})
    tools = value["toolchain"]
    if inputs["toolchain"] != tools:
        raise ValueError("private toolchain input binding mismatch")
    for name, (path, digest) in PINNED_TOOLS.items():
        if tools[name]["path"] != path or tools[name]["sha256"] != digest:
            raise ValueError("private build toolchain pin mismatch")
    if preparation_inputs["patch_tool"] != {key: tools["patch"][key] for key in ("path", "sha256")}:
        raise ValueError("private source patch toolchain binding mismatch")
    paths = {name: row["path"] for name, row in tools.items()}
    http2 = {"version": "1.68.0", "archive": str(prefix / "lib/libnghttp2.a"), "include": str(prefix / "include")}
    plan = build_plan(work, paths, openssl["prefix"], jobs=inputs["jobs"], http2=http2)
    if inputs["commands"] != plan or inputs["environment"] != build_environment(work, paths):
        raise ValueError("private build command/environment policy mismatch")
    command_data = {}
    for reference in value["commands"]:
        command = read(reference)
        name = Path(reference["path"]).name
        if name in command_data or command.get("exit_code") != 0 or command.get("cleanup") != "terminal":
            raise ValueError("private build command missing success or terminal cleanup")
        command_data[name] = command
    for item in plan:
        command = command_data.get(item["name"] + "-command.json")
        if command is None or any(command[key] != item[other] for key, other in (("argv", "argv"), ("cwd", "cwd"), ("timeout_seconds", "timeout"))):
            raise ValueError("private build command evidence incomplete")
    if Path(patch["command"]["path"]).name not in command_data:
        raise ValueError("private zlib patch command omitted")
    if any(Path(stage["command"]["path"]).name not in command_data for stage in value["nghttp2_patch"]["stages"]):
        raise ValueError("nghttp2 ordered source patch command omitted")
    expected_caches = []
    for item in plan:
        if item["name"].endswith("-configure") and item["name"] != "libiconv-configure":
            name = item["name"][:-10]
            required = {arg[2:].split("=",1)[0]: arg.split("=",1)[1] for arg in item["argv"] if arg.startswith("-D")}
            expected_caches.append(validate_cache(work / "build" / name / "CMakeCache.txt", required))
    if value["caches"] != expected_caches:
        raise ValueError("private configured cache inventory mismatch")
    prefix_tree = tree(value["prefix_inventory"], prefix, frozen=True)
    tree(value["build_inventory"], work / "build")
    if not value["objects"] or {row["path"] for row in value["objects"]} != {str(path) for path in (work / "build").rglob("*.o") if path.is_file() and not path.is_symlink()}:
        raise ValueError("private build object closure mismatch")
    archives = [prefix / "lib" / name for name in ("libcurl.a", "libxml2.a", "libiconv.a", "libcharset.a", "libz.a", "libnghttp2.a")]
    archives += [Path(openssl["prefix"]) / "lib" / name for name in ("libssl.a", "libcrypto.a")]
    if len(value["static_archives"]) != len(archives) or {row["path"] for row in value["static_archives"]} != {str(path) for path in archives}:
        raise ValueError("private static archive set mismatch")
    for row in value["static_archives"]:
        with Path(row["path"]).open("rb") as stream:
            magic = stream.read(8)
        if not row.get("members") or magic != b"!<arch>\n":
            raise ValueError("private static archive member inventory missing")
        if (row["path"] == str(prefix / "lib/libnghttp2.a")
                and sorted(row["members"]) != sorted(Path(item["path"]).name for item in value["nghttp2_build"]["objects"])):
            raise ValueError("nghttp2 private archive/compiled object membership mismatch")
    expected_observation = {"curl": "8.23.0", "openssl": "3.6.5", "zlib": "1.3.2", "libxml2": "2.15.4", "libiconv": "1.19", "nghttp2": "1.68.0",
                            **{key: True for key in ("http", "https", "http2", "xml", "xpath", "encoding", "gzip_xml")}}
    if value["probe"]["observation"] != expected_observation or value["effective_versions"] != expected_observation:
        raise ValueError("private static probe capability/version mismatch")
    for name, expected in (("probe-whole-archive-pic-link", set(archives) - {prefix / "lib/libcharset.a"}), ("probe-charset-whole-archive-pic-link", {prefix / "lib/libcharset.a"})):
        command = command_data.get(name + "-command.json", {})
        argv = command.get("argv", [])
        if (not {"-shared", "-Wl,-z,defs", "-Wl,-z,text", "-Wl,--whole-archive", "-Wl,--no-whole-archive"} <= set(argv)
                or not {str(path) for path in expected} <= set(argv)):
            raise ValueError("private whole-archive PIC probe command missing")
    required_probes = {"probe-compile", "probe-link", "probe-observation", "probe-dynamic", "probe-whole-dynamic", "probe-charset-whole-dynamic"}
    if not {name + "-command.json" for name in required_probes} <= command_data.keys():
        raise ValueError("private consumer probe command evidence incomplete")
    observed = command_data["probe-observation-command.json"]["log"]
    if json.loads(Path(observed["path"]).read_bytes()) != value["probe"]["observation"]:
        raise ValueError("private consumer probe observation log mismatch")
    for index, archive in enumerate(archives):
        for action, argv in (("members", [paths["ar"], "t", str(archive)]),
                             ("symbols", [paths["nm"], "-g", "--defined-only", str(archive)]),
                             ("relocations", [paths["readelf"], "-r", str(archive)])):
            command = command_data.get(f"archive-{index}-{action}-command.json", {})
            if command.get("argv") != argv:
                raise ValueError("private static archive command evidence missing")
            if action == "members" and Path(command["log"]["path"]).read_text().splitlines() != value["static_archives"][index]["members"]:
                raise ValueError("private static archive member log mismatch")
    dynamic = "".join(Path(command_data[name + "-command.json"]["log"]["path"]).read_text() for name in ("probe-dynamic", "probe-whole-dynamic", "probe-charset-whole-dynamic"))
    maps = [Path(item["path"]) for item in value["linkage"]["maps"]]
    if validate_linkage(dynamic, maps, archives) != value["linkage"]:
        raise ValueError("private static link-map identity mismatch")
    private_files = {}
    for row in prefix_tree["files"]:
        if row["kind"] != "file":
            continue
        relative = row["relative"]
        component = next((name for name in ("libiconv", "libcharset", "libxml2", "libcurl", "libnghttp2") if Path(relative).name == name + ".a"), None)
        if relative.startswith("include/curl/"):
            component = "libcurl"
        elif relative.startswith("include/libxml2/"):
            component = "libxml2"
        elif relative.startswith("include/nghttp2/"):
            component = "libnghttp2"
        elif Path(relative).name in ("libz.a", "zlib.h", "zconf.h"):
            component = "zlib"
        elif Path(relative).name in ("iconv.h", "localcharset.h"):
            component = "libiconv" if Path(relative).name == "iconv.h" else "libcharset"
        if component:
            source_name = {"libcurl": "curl", "libcharset": "libiconv", "libnghttp2": "nghttp2"}.get(component, component)
            source_version = nghttp2["source_version"] if source_name == "nghttp2" else SOURCES[source_name]["version"]
            private_files[str(prefix / relative)] = {"sha256": row["sha256"], "component": component,
                                                    "source_package": source_name, "source_version": source_version}
    return {"receipt": receipt_ref, "prefix": str(prefix), "build": value,
            "files": [{"path": path, "sha256": digest} for path, digest in sorted(files.items())], "private_files": private_files}


PROBE = r'''#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <curl/curl.h>
#include <openssl/crypto.h>
#include <openssl/ssl.h>
#include <zlib.h>
#include <iconv.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <nghttp2/nghttp2.h>
int main(int argc, char **argv) {
  if(argc != 2) return 1;
  if(curl_global_init(CURL_GLOBAL_DEFAULT)) return 2;
  const curl_version_info_data *c = curl_version_info(CURLVERSION_NOW);
  if(!c || strcmp(c->version,"8.23.0") || !c->ssl_version || strncmp(c->ssl_version,"OpenSSL/3.6.5",13)) return 3;
  if(!c->libz_version || strcmp(c->libz_version,"1.3.2") || strcmp(zlibVersion(),"1.3.2")) return 4;
  if(!(c->features & CURL_VERSION_SSL) || !(c->features & CURL_VERSION_LIBZ) || !(c->features & CURL_VERSION_HTTP2)) return 5;
  int http=0, https=0;
  for(const char * const *p=c->protocols; *p; ++p) {
    if(!strcmp(*p,"http")) http=1; else if(!strcmp(*p,"https")) https=1; else return 6;
  }
  if(!http || !https || strcmp(nghttp2_version(0)->version_str,"1.68.0")) return 7;
  if(strncmp(OpenSSL_version(OPENSSL_VERSION),"OpenSSL 3.6.5 ",14)) return 8;
  SSL_CTX *ssl=SSL_CTX_new(TLS_client_method()); if(!ssl) return 9; SSL_CTX_free(ssl);
  if(_libiconv_version != 0x0113) return 10;
  iconv_t converter=iconv_open("UTF-8","ISO-8859-1"); if(converter == (iconv_t)-1) return 11;
  char input[1]={(char)0xe9}, output[8]={0}, *in=input, *out=output; size_t ni=1,no=sizeof(output);
  if(iconv(converter,&in,&ni,&out,&no)==(size_t)-1 || ni || memcmp(output,"\xc3\xa9",2)) return 12;
  iconv_close(converter);
  if(strcmp(xmlParserVersion,"21504") || !xmlHasFeature(XML_WITH_ICONV) || !xmlHasFeature(XML_WITH_ZLIB) || !xmlHasFeature(XML_WITH_XPATH)) return 13;
  const char *document="<root><item>source-bound</item></root>";
  xmlDocPtr doc=xmlReadMemory(document,(int)strlen(document),"offline.xml",NULL,XML_PARSE_NONET);
  if(!doc) return 14;
  xmlXPathContextPtr ctx=xmlXPathNewContext(doc); if(!ctx) return 15;
  xmlXPathObjectPtr found=xmlXPathEvalExpression((const xmlChar*)"/root/item",ctx);
  if(!found || !found->nodesetval || found->nodesetval->nodeNr!=1) return 16;
  xmlXPathFreeObject(found); xmlXPathFreeContext(ctx); xmlFreeDoc(doc);
  gzFile gzip=gzopen(argv[1],"wb"); if(!gzip) return 17;
  if(gzwrite(gzip,document,(unsigned)strlen(document))!=(int)strlen(document) || gzclose(gzip)!=Z_OK) return 18;
  doc=xmlReadFile(argv[1],NULL,XML_PARSE_NONET|XML_PARSE_UNZIP); if(!doc) return 19; xmlFreeDoc(doc);
  curl_global_cleanup(); xmlCleanupParser();
  printf("{\"curl\":\"8.23.0\",\"openssl\":\"3.6.5\",\"zlib\":\"1.3.2\",\"libxml2\":\"2.15.4\",\"libiconv\":\"1.19\",\"nghttp2\":\"1.68.0\",\"http\":true,\"https\":true,\"http2\":true,\"xml\":true,\"xpath\":true,\"encoding\":true,\"gzip_xml\":true}\n");
  return 0;
}
'''


def _copy(reference, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as output, Path(reference["path"]).open("rb") as source:
        shutil.copyfileobj(source, output)
    return checked(target, reference["sha256"], "captured input")


def _freeze(root):
    for path in sorted([Path(root), *Path(root).rglob("*")], reverse=True):
        if not path.is_symlink():
            mode = stat.S_IMODE(path.stat().st_mode)
            path.chmod(mode & ~0o222)


def build(work, archive_dir, openssl, nghttp2, tools, jobs, *, owned=None):
    work = Path(work).absolute()
    prepared = prepare_sources(archive_dir, work, owned=owned)
    paths = {name: row["path"] for name, row in tools.items()}
    env = build_environment(work, paths)
    prefix = work / "prefix"
    access = _copy(nghttp2["receipt"], work / "inputs/nghttp2-source-access.json")
    nghttp2 = validate_nghttp2_source(access["path"], access["sha256"], work / "inputs")
    _copy(openssl["receipt"], work / "inputs/openssl-build-receipt.json")
    _copy(openssl["source_archive"], work / "inputs/openssl-3.6.5.tar.gz")
    http2 = {"version": "1.68.0", "archive": str(prefix / "lib/libnghttp2.a"), "include": str(prefix / "include")}
    plan = build_plan(work, paths, openssl["prefix"], jobs=jobs, http2=http2)
    source = work / "native-probe.c"
    source.write_text(PROBE)
    inputs = {"schema_version": 1, "kind": "private_network_build_inputs", "recipe": prepared["recipe"],
              "prepared_sources": record(work / "prepared-sources.json"), "source_pins": SOURCES,
              "openssl": openssl, "nghttp2": nghttp2, "toolchain": tools, "jobs": jobs,
              "environment": env, "commands": plan, "probe_source": record(source),
              "ca_bundle": record(Path("/etc/ssl/certs/ca-certificates.crt"))}
    input_ref = save(work / "build-inputs.json", inputs)
    command_refs = [prepared["patch_application"]["command"],
                    *[stage["command"] for stage in prepared["nghttp2_patch_application"]["stages"]]]
    caches = []
    for item in plan:
        Path(item["cwd"]).mkdir(parents=True, exist_ok=True)
        command_refs.append(run_command(item["argv"], cwd=item["cwd"], env=env, work=work, name=item["name"], timeout=item["timeout"]))
        if item["name"].endswith("-configure") and item["name"] != "libiconv-configure":
            name = item["name"][:-10]
            expected = {arg[2:].split("=",1)[0]: arg.split("=",1)[1] for arg in item["argv"] if arg.startswith("-D")}
            caches.append(validate_cache(work / "build" / name / "CMakeCache.txt", expected))
            if "Manually-specified variables were not used" in (work / "logs" / (item["name"] + ".log")).read_text():
                raise ValueError("native configuration ignored a pinned option")
    verify_sources(prepared)
    verify_nghttp2_patch_application(prepared["nghttp2_patch_application"], work, prepared["archives"])
    nghttp2_build = verify_nghttp2_outputs(work)
    crypto = Path(openssl["prefix"])
    libraries = [prefix / "lib" / name for name in ("libcurl.a", "libxml2.a", "libiconv.a", "libcharset.a", "libz.a", "libnghttp2.a")]
    libraries += [crypto / "lib/libssl.a", crypto / "lib/libcrypto.a"]
    for archive in libraries:
        if archive.is_symlink() or not archive.is_file() or archive.read_bytes()[:8] != b"!<arch>\n":
            raise ValueError("private static archive missing or invalid: " + str(archive))
    object_file, executable, shared = work / "native-probe.o", work / "native-probe", work / "native-probe.so"
    executable_map, shared_map = work / "native-probe.map", work / "native-probe-whole.map"
    charset_map, charset_shared = work / "charset-probe-whole.map", work / "charset-probe.so"
    def run(name, argv, timeout=300):
        ref = run_command([str(arg) for arg in argv], cwd=work, env=env, work=work, name=name, timeout=timeout)
        command_refs.append(ref)
        return (work / "logs" / (name + ".log")).read_text()
    run("probe-compile", [paths["cc"], "-O2", "-g0", "-fPIC", "-DCURL_STATICLIB", "-DNGHTTP2_STATICLIB", "-I" + str(prefix / "include"), "-I" + str(prefix / "include/libxml2"), "-I" + str(crypto / "include"), "-c", source, "-o", object_file])
    # Whole archives prove PIC for all private archive members, including the
    # source-built private nghttp2 archive rather than only used HTTP2 symbols.
    # libiconv already incorporates localcharset.o. Its separately installed
    # libcharset archive has the same symbol and needs its own whole-PIC link.
    consumer_libraries = [path for path in libraries if path.name != "libcharset.a"]
    common_link = ["-Wl,--no-as-needed", "-Wl,--start-group", *consumer_libraries, "-Wl,--end-group", "-pthread", "-ldl", "-lm"]
    run("probe-link", [paths["cc"], object_file, "-Wl,-Map," + str(executable_map), "-o", executable, *common_link])
    run("probe-whole-archive-pic-link", [paths["cc"], "-shared", "-Wl,-z,defs", "-Wl,-z,text", "-Wl,-Map," + str(shared_map), "-o", shared,
        "-Wl,--whole-archive", *consumer_libraries, "-Wl,--no-whole-archive", "-pthread", "-ldl", "-lm"])
    run("probe-charset-whole-archive-pic-link", [paths["cc"], "-shared", "-Wl,-z,defs", "-Wl,-z,text", "-Wl,-Map," + str(charset_map), "-o", charset_shared,
        "-Wl,--whole-archive", prefix / "lib/libcharset.a", "-Wl,--no-whole-archive", "-pthread", "-ldl", "-lm"])
    observation = json.loads(run("probe-observation", [executable, work / "probe.xml.gz"], 30))
    expected_observation = {"curl": "8.23.0", "openssl": "3.6.5", "zlib": "1.3.2", "libxml2": "2.15.4", "libiconv": "1.19", "nghttp2": "1.68.0",
                            **{key: True for key in ("http", "https", "http2", "xml", "xpath", "encoding", "gzip_xml")}}
    if observation != expected_observation:
        raise ValueError("native static consumer returned unexpected capability/version evidence")
    dynamic = run("probe-dynamic", [paths["readelf"], "-d", executable])
    dynamic += run("probe-whole-dynamic", [paths["readelf"], "-d", shared])
    dynamic += run("probe-charset-whole-dynamic", [paths["readelf"], "-d", charset_shared])
    linkage = validate_linkage(dynamic, [executable_map, shared_map, charset_map], libraries)
    static_archives = []
    for index, archive in enumerate(libraries):
        names = run(f"archive-{index}-members", [paths["ar"], "t", archive]).splitlines()
        if not names:
            raise ValueError("empty private static archive")
        if (archive == prefix / "lib/libnghttp2.a"
                and sorted(names) != sorted(Path(item["path"]).name for item in nghttp2_build["objects"])):
            raise ValueError("nghttp2 built archive/compiled object membership mismatch")
        run(f"archive-{index}-symbols", [paths["nm"], "-g", "--defined-only", archive])
        run(f"archive-{index}-relocations", [paths["readelf"], "-r", archive])
        static_archives.append({**record(archive), "members": names})
    licences = {}
    for name, relative in (("curl", "COPYING"), ("zlib", "LICENSE"), ("libxml2", "Copyright"), ("libiconv", "COPYING.LIB"), ("nghttp2", "COPYING")):
        original = work / "sources" / SOURCES[name]["root"] / relative
        licences[name] = _copy(record(original), prefix / "licenses" / (name + "-" + relative))
    licences["libcharset"] = _copy(record(work / "sources/libiconv-1.19/libcharset/COPYING.LIB"), prefix / "licenses/libcharset-COPYING.LIB")
    licences["nghttp2_distribution"] = _copy(record(work / "distribution/debian/copyright"), prefix / "licenses/nghttp2-distribution-copyright")
    verify_sources(prepared)
    if validate_openssl(crypto, openssl["source_archive"]["path"]) != openssl:
        raise ValueError("private OpenSSL inputs changed during build")
    if validate_nghttp2_source(nghttp2["receipt"]["path"], nghttp2["receipt"]["sha256"], work / "inputs") != nghttp2:
        raise ValueError("nghttp2 distro source inputs changed during build")
    if verify_nghttp2_outputs(work) != nghttp2_build:
        raise ValueError("nghttp2 source build outputs changed during qualification")
    for row in tools.values():
        checked(row["path"], row["sha256"], "build tool")
    checked(inputs["ca_bundle"]["path"], inputs["ca_bundle"]["sha256"], "CA bundle")
    objects = [record(path) for path in sorted((work / "build").rglob("*.o")) if path.is_file() and not path.is_symlink()]
    if not objects:
        raise ValueError("native build object evidence missing")
    build_files = save(work / "build-inventory.json", inventory(work / "build"))
    _freeze(prefix)
    frozen = save(work / "prefix-inventory.json", prefix_inventory(prefix))
    result = {
        "schema_version": 1, "kind": "private_network_static_build", "status": "compiled_unverified", "ready": False,
        "input_manifest": input_ref, "recipe": prepared["recipe"], "prefix": str(prefix), "prefix_inventory": frozen,
        "sources": prepared["sources"], "source_archives": prepared["archives"], "zlib_patch": prepared["patch_application"],
        "nghttp2_patch": prepared["nghttp2_patch_application"], "nghttp2_build": nghttp2_build,
        "openssl": openssl, "nghttp2": nghttp2, "effective_versions": expected_observation,
        "toolchain": tools, "commands": command_refs, "caches": caches, "objects": objects, "build_inventory": build_files,
        "static_archives": static_archives, "linkage": linkage, "probe": {"source": record(source), "object": record(object_file), "executable": record(executable), "whole_archive_shared": record(shared), "charset_whole_archive_shared": record(charset_shared), "observation": observation},
        "licenses": licences,
        "limits": ["Candidate build only; no Arrow runtime admission, hosted-service qualification or general security clearance.",
                   "nghttp2 is privately rebuilt from the exact Ubuntu 1.68.0-2ubuntu0.2 source and ordered patches; this does not claim reproducibility of the installed Ubuntu binary.",
                   "nghttp2 DSC checksums and primary HTTPS source publication are pinned; gpgv reported an unknown signing key, so no signature verification is asserted.",
                   "Sources, object files, licences and exact links are retained; downstream static-linked distribution must carry appropriate relink materials and notices.",
                   "Command cleanup covers the original process group/session; the coordinator must provide descendant supervision for escaped processes and interruptions."],
    }
    save(work / "build-receipt.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("work", "archive-dir", "openssl-prefix"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--openssl-source-archive", type=Path)
    parser.add_argument("--nghttp2-source-receipt", type=Path)
    parser.add_argument("--nghttp2-source-receipt-sha256")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    work, owned = args.work.absolute(), {}
    try:
        if work.exists() or work.is_symlink():
            raise FileExistsError(f"build work already exists: {work}")
        if work.resolve() != work:
            raise ValueError("canonical work path required")
        if isinstance(args.jobs, bool) or not 1 <= args.jobs <= 4:
            raise ValueError("jobs must be between1and4")
        archives = validate_archive_dir(args.archive_dir)
        archive = args.openssl_source_archive or args.openssl_prefix.parent / "downloads/openssl.tar.gz"
        openssl = validate_openssl(args.openssl_prefix, archive)
        if args.nghttp2_source_receipt is None:
            raise ValueError("explicit --nghttp2-source-receipt and SHA256 required to preserve HTTP2")
        http2 = validate_nghttp2_source(args.nghttp2_source_receipt, args.nghttp2_source_receipt_sha256, args.archive_dir)
        tools = validate_tools()
        if args.check:
            print(json.dumps({"status": "inputs_verified", "ready": False, "archives": archives, "openssl": openssl, "nghttp2": http2, "toolchain": tools}, sort_keys=True))
        else:
            result = build(work, args.archive_dir, openssl, http2, tools, args.jobs, owned=owned)
            print(json.dumps({"status": result["status"], "ready": False, "receipt": record(work / "build-receipt.json")}, sort_keys=True))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError, tarfile.TarError, RuntimeError) as error:
        if owned and work.is_dir() and not work.is_symlink():
            info = work.lstat()
            if owned == {"device": info.st_dev, "inode": info.st_ino, "uid": info.st_uid} and not (work / "FAILED.json").exists():
                save(work / "FAILED.json", {"status": "failed", "ready": False, "error_type": type(error).__name__, "error": str(error)})
        print(json.dumps({"status": "blocked", "ready": False, "reason": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
