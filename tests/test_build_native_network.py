"""Private native network build guards; synthetic sources, no actual build."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_native_network.py"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def record(path):
    return {"path": str(path.resolve()), "sha256": digest(path.read_bytes())}


def archive_at(path, root, files, links=None):
    with tarfile.open(path, "w:gz") as bundle:
        for name, data in files.items():
            member = tarfile.TarInfo(f"{root}/{name}")
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
        for name, target in (links or {}).items():
            member = tarfile.TarInfo(f"{root}/{name}")
            member.type = tarfile.SYMTYPE
            member.linkname = target
            bundle.addfile(member)
    return path


def test_network_cli_requires_explicit_isolated_inputs():
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(SCRIPT)], capture_output=True, text=True,
    )

    assert result.returncode == 2
    assert all(flag in result.stderr for flag in ("--work", "--archive-dir", "--openssl-prefix"))


def test_network_cli_rejects_an_existing_work_directory(tmp_path):
    work = tmp_path / "existing"
    work.mkdir()
    sentinel = work / "owned-by-previous-attempt"
    sentinel.write_bytes(b"retain failed or prior work")
    prepared = work / "preparation-inputs.json"
    prepared.write_bytes(b"previous attempt owns this directory")
    result = subprocess.run([
        sys.executable, "-I", "-B", str(SCRIPT), "--work", str(work),
        "--archive-dir", str(tmp_path / "absent-inputs"),
        "--openssl-prefix", str(tmp_path / "absent-openssl"),
    ], capture_output=True, text=True)

    assert result.returncode == 1
    assert "already exists" in result.stderr
    assert sentinel.read_bytes() == b"retain failed or prior work"
    assert set(work.iterdir()) == {sentinel, prepared}
    assert prepared.read_bytes() == b"previous attempt owns this directory"


@pytest.fixture
def recipe():
    spec = importlib.util.spec_from_file_location("private_network_recipe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source_inputs(tmp_path, monkeypatch, recipe):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    before = b"int gzwrite_guard = 0;\n"
    after = b"int gzwrite_guard = 1;\n"
    patch = (
        b"--- a/gzwrite.c\n+++ b/gzwrite.c\n@@ -1 +1 @@\n"
        b"-int gzwrite_guard = 0;\n+int gzwrite_guard = 1;\n"
    )
    pins = {}
    ng_before = b"int source_guard = 0;\n"
    ng_headers = {"nghttp2.h": b"synthetic original public header\n",
                  "nghttp2ver.h.in": b'#define NGHTTP2_VERSION "@PACKAGE_VERSION@"\n#define NGHTTP2_VERSION_NUM @PACKAGE_VERSION_NUM@\n'}
    for name, pin in recipe.SOURCES.items():
        files = {"COPYING": b"synthetic upstream licence\n", "source.c": name.encode()}
        if name == "zlib":
            files["gzwrite.c"] = before
        elif name == "nghttp2":
            files["lib/nghttp2_session.c"] = ng_before
            files["lib/CMakeLists.txt"] = b"set(NGHTTP2_SOURCES nghttp2_session.c)\n"
            files.update({"lib/includes/nghttp2/" + name: data for name, data in ng_headers.items()})
        path = archive_at(inputs / pin["filename"], pin["root"], files)
        pins[name] = {**pin, "sha256": digest(path.read_bytes())}
    monkeypatch.setattr(recipe, "SOURCES", pins)
    (inputs / "zlib-df84af25.patch").write_bytes(patch)
    monkeypatch.setattr(recipe, "ZLIB_PATCH_SHA256", digest(patch))
    monkeypatch.setattr(recipe, "ZLIB_ORIGINAL_GZWRITE_SHA256", digest(before))
    monkeypatch.setattr(recipe, "ZLIB_PATCHED_GZWRITE_SHA256", digest(after))
    stages, debian_files = [], {"copyright": b"synthetic distribution licence\n"}
    for index, stage in enumerate(recipe.NGHTTP2_PATCH_STAGES):
        left, right = f"int source_guard = {index};\n".encode(), f"int source_guard = {index + 1};\n".encode()
        data = b"--- a/lib/nghttp2_session.c\n+++ b/lib/nghttp2_session.c\n@@ -1 +1 @@\n-" + left + b"+" + right
        debian_files["patches/" + stage["name"]] = data
        stages.append({"name": stage["name"], "sha256": digest(data), "files": {"lib/nghttp2_session.c": {"before": digest(left), "after": digest(right)}}})
    monkeypatch.setattr(recipe, "NGHTTP2_PATCH_STAGES", stages)
    debian_files["patches/series"] = recipe.nghttp2_series().encode()
    distro_pins = {name: dict(pin) for name, pin in recipe.NGHTTP2_DIST_INPUTS.items()}
    archive_at(inputs / distro_pins["nghttp2_debian"]["filename"], "debian", debian_files)
    (inputs / distro_pins["nghttp2_signature"]["filename"]).write_bytes(b"synthetic detached signature; not verified\n")
    for name in ("nghttp2_debian", "nghttp2_signature"):
        path = inputs / distro_pins[name]["filename"]
        distro_pins[name].update(sha256=digest(path.read_bytes()), size=path.stat().st_size)
    dsc = "Source: nghttp2\nVersion: " + recipe.NGHTTP2_SOURCE_VERSION + "\nChecksums-Sha256:\n"
    for pin in (pins["nghttp2"], distro_pins["nghttp2_signature"], distro_pins["nghttp2_debian"]):
        path = inputs / pin["filename"]
        dsc += f" {digest(path.read_bytes())} {path.stat().st_size} {path.name}\n"
    path = inputs / distro_pins["nghttp2_dsc"]["filename"]
    path.write_text(dsc)
    distro_pins["nghttp2_dsc"].update(sha256=digest(path.read_bytes()), size=path.stat().st_size)
    monkeypatch.setattr(recipe, "NGHTTP2_DIST_INPUTS", distro_pins)
    return inputs, tmp_path / "work", before, after


@pytest.mark.parametrize("change", ("archive", "patch", "archive-symlink"))
def test_input_checksum_rejection_precedes_work_creation(recipe, source_inputs, change):
    inputs, work, _, _ = source_inputs
    path = inputs / recipe.SOURCES["curl"]["filename"]
    if change == "archive":
        path.write_bytes(b"unreviewed archive")
    elif change == "patch":
        (inputs / "zlib-df84af25.patch").write_bytes(b"unreviewed patch")
    else:
        moved = inputs / "moved.tar.gz"
        path.rename(moved)
        path.symlink_to(moved)

    with pytest.raises(ValueError, match="checksum|symlink"):
        recipe.prepare_sources(inputs, work)

    assert not work.exists()


@pytest.mark.parametrize("member", ("../outside", "/absolute", "a/../../outside"))
def test_source_archive_rejects_path_escape_before_extraction(recipe, tmp_path, member):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        entry = tarfile.TarInfo(member if member.startswith("/") else f"root/{member}")
        entry.size = 3
        bundle.addfile(entry, io.BytesIO(b"bad"))

    with tarfile.open(archive) as bundle, pytest.raises(ValueError, match="archive"):
        recipe.archive_members(bundle, "root")

    assert set(tmp_path.iterdir()) == {archive}


def test_source_archive_rejects_symlink_traversal(recipe, tmp_path):
    archive = archive_at(
        tmp_path / "unsafe.tar.gz", "root", {"directory/file": b"bad"},
        {"directory": "inside"},
    )

    with tarfile.open(archive) as bundle, pytest.raises(ValueError, match="symlink"):
        recipe.archive_members(bundle, "root")


def test_preparation_preserves_all_sources_and_applies_exact_zlib_patch(recipe, source_inputs):
    inputs, work, before, after = source_inputs

    prepared = recipe.prepare_sources(inputs, work)

    assert (work / "sources/zlib-1.3.2/gzwrite.c").read_bytes() == after
    assert (work / "inputs/zlib-1.3.2.tar.gz").read_bytes() == (inputs / "zlib-1.3.2.tar.gz").read_bytes()
    assert prepared["patch_application"]["original_gzwrite_sha256"] == digest(before)
    assert prepared["patch_application"]["patched_gzwrite_sha256"] == digest(after)
    assert set(prepared["sources"]) == {"curl", "libxml2", "libiconv", "zlib", "nghttp2"}
    assert all(item["files"] for item in prepared["sources"].values())
    assert not list(work.rglob("*.orig"))
    assert not list(work.rglob("*.rej"))

    with pytest.raises(FileExistsError, match="already exists"):
        recipe.prepare_sources(inputs, work)


def test_source_inventory_detects_non_zlib_mutation(recipe, source_inputs):
    inputs, work, _, _ = source_inputs
    prepared = recipe.prepare_sources(inputs, work)
    source = work / "sources/curl-8.23.0/source.c"
    source.write_bytes(b"changed after preparation")

    with pytest.raises(ValueError, match="source"):
        recipe.verify_sources(prepared)


def test_build_environment_discards_inherited_compiler_loader_and_provider_overrides(
    recipe, tmp_path, monkeypatch,
):
    for key in ("CFLAGS", "LDFLAGS", "CPATH", "LD_PRELOAD", "CMAKE_ARGS", "MAKEFLAGS", "PKG_CONFIG_PATH", "OPENSSL_CONF", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(key, "untrusted inherited setting")
    tools = {"cc": "/pinned/gcc", "cxx": "/pinned/g++", "ar": "/pinned/ar", "ranlib": "/pinned/ranlib"}

    env = recipe.build_environment(tmp_path, tools)

    assert not any("untrusted" in value for value in env.values())
    assert env["CC"] == "/pinned/gcc" and env["CXX"] == "/pinned/g++"
    assert env["MAKEFLAGS"] == "-j1" and env["CMAKE_BUILD_PARALLEL_LEVEL"] == "1"
    assert env["CONFIG_SITE"] == "/dev/null"
    assert env["OPENSSL_CONF"] == "/dev/null"
    assert env["HOME"] == str(tmp_path / "home")
    assert env["TMPDIR"] == str(tmp_path / "tmp")


@pytest.mark.parametrize("jobs", (0, 5, -1))
def test_build_plan_rejects_unbounded_jobs(recipe, tmp_path, jobs):
    with pytest.raises(ValueError, match="jobs"):
        recipe.build_plan(tmp_path, {}, tmp_path / "crypto", jobs=jobs, http2=None)


def test_build_plan_forces_static_private_providers_and_http_protocols(recipe, tmp_path):
    tools = {name: f"/pinned/{name}" for name in ("cc", "cxx", "cmake", "ninja", "make")}
    prefix = tmp_path / "prefix"
    crypto = tmp_path / "crypto"
    http2 = {"version": "1.68.0", "archive": str(prefix / "lib/libnghttp2.a"), "include": str(prefix / "include")}

    plan = recipe.build_plan(tmp_path, tools, crypto, jobs=4, http2=http2)

    curl = next(item["argv"] for item in plan if item["name"] == "curl-configure")
    xml = next(item["argv"] for item in plan if item["name"] == "libxml2-configure")
    assert "-DBUILD_SHARED_LIBS=OFF" in curl and "-DBUILD_STATIC_LIBS=ON" in curl
    assert "-DCURL_USE_OPENSSL=ON" in curl and "-DCURL_DISABLE_HTTP=OFF" in curl
    assert "-DCURL_ZLIB=ON" in curl and "-DUSE_NGHTTP2=ON" in curl
    assert f"-DZLIB_LIBRARY={prefix / 'lib/libz.a'}" in curl
    assert f"-DOPENSSL_SSL_LIBRARY={crypto / 'lib/libssl.a'}" in curl
    assert f"-DOPENSSL_CRYPTO_LIBRARY={crypto / 'lib/libcrypto.a'}" in curl
    assert "-DNGHTTP2_USE_STATIC_LIBS=ON" in curl
    assert "-DHTTP_ONLY=ON" not in curl
    for name in ("LDAP", "LDAPS", "DICT", "FTP", "FILE", "GOPHER", "IMAP", "MQTT", "POP3", "RTSP", "SMTP", "TELNET", "TFTP", "WEBSOCKETS"):
        assert f"-DCURL_DISABLE_{name}=ON" in curl
    for name in ("LIBSSH2", "LIBSSH", "LIBPSL", "GSSAPI"):
        assert f"-DCURL_USE_{name}=OFF" in curl
    assert "-DCURL_USE_PKGCONFIG=OFF" in curl
    assert "-DCURL_DISABLE_OPENSSL_AUTO_LOAD_CONFIG=ON" in curl
    assert "-DLIBXML2_WITH_ICU=OFF" in xml and "-DLIBXML2_WITH_ZLIB=ON" in xml
    assert "-DLIBXML2_WITH_ICONV=ON" in xml and "-DIconv_IS_BUILT_IN=FALSE" in xml
    assert f"-DIconv_LIBRARY={prefix / 'lib/libiconv.a'}" in xml
    assert "-DLIBXML2_WITH_PROGRAMS=OFF" in xml and "-DLIBXML2_WITH_TESTS=OFF" in xml
    for item in plan:
        assert 0 < item["timeout"] <= 1800
        assert not any(arg.startswith("-j") and arg not in ("-j1", "-j4") for arg in item["argv"])
    assert all("--parallel" in item["argv"] for item in plan if item["name"].endswith("-build") and item["argv"][0] == tools["cmake"])


def test_cmake_cache_rejects_wrong_static_provider(recipe, tmp_path):
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text("ZLIB_LIBRARY:FILEPATH=/usr/lib/libz.so\nCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON\n")

    with pytest.raises(ValueError, match="ZLIB_LIBRARY"):
        recipe.validate_cache(cache, {"ZLIB_LIBRARY": "/private/lib/libz.a"})


def test_link_map_requires_consumed_private_archive_members(recipe, tmp_path):
    archive = tmp_path / "prefix/lib/libz.a"
    mapping = tmp_path / "probe.map"
    mapping.write_text(f"LOAD {archive}\n")

    with pytest.raises(ValueError, match="consumed|member"):
        recipe.validate_linkage("", [mapping], [archive])


@pytest.mark.parametrize("foreign", ("/usr/lib/libz.a", "/usr/lib/libcurl.a", "/usr/lib/libssl.a", "/usr/lib/libxml2.a", "/usr/lib/libiconv.a"))
def test_link_map_rejects_foreign_native_providers(recipe, tmp_path, foreign):
    archive = tmp_path / "prefix/lib/libz.a"
    mapping = tmp_path / "probe.map"
    mapping.write_text(f"{archive}(gzwrite.o)\n{foreign}(foreign.o)\nLOAD {foreign}\n")

    with pytest.raises(ValueError, match="foreign|private"):
        recipe.validate_linkage("", [mapping], [archive])


@pytest.mark.parametrize("library", ("libz.so.1", "libcurl.so.4", "libssl.so.3", "libcrypto.so.3", "libxml2.so.16", "libnghttp2.so.14"))
def test_linkage_rejects_dynamic_private_dependency(recipe, tmp_path, library):
    archive = tmp_path / "prefix/lib/libz.a"
    mapping = tmp_path / "probe.map"
    mapping.write_text(f"{archive}(gzwrite.o)\nLOAD {archive}\n")

    with pytest.raises(ValueError, match="dynamic|private"):
        recipe.validate_linkage(f"(NEEDED) Shared library: [{library}]", [mapping], [archive])


@pytest.mark.parametrize("evidence", ("needed", "load", "member", "relative-member"))
def test_linkage_rejects_foreign_curl_backend_variant(recipe, tmp_path, evidence):
    archive = tmp_path / "prefix/lib/libcurl.a"
    mapping = tmp_path / "probe.map"
    text, dynamic = f"{archive}(private-curl.o)\nLOAD {archive}\n", ""
    if evidence == "needed":
        dynamic = "(NEEDED) Shared library: [libcurl-gnutls.so.4]\n"
    elif evidence == "load":
        text += "LOAD /usr/lib/x86_64-linux-gnu/libcurl-gnutls.so.4\n"
    elif evidence == "member":
        text += "/usr/lib/x86_64-linux-gnu/libcurl-gnutls.a(foreign.o)\n"
    else:
        text += "../../host/libcurl-gnutls.a(foreign.o)\n"
    mapping.write_text(text)

    with pytest.raises(ValueError, match="dynamic|foreign|private"):
        recipe.validate_linkage(dynamic, [mapping], [archive])


def test_linkage_accepts_only_consumed_exact_static_archives(recipe, tmp_path):
    archives = [tmp_path / "private" / name for name in ("libz.a", "libcurl.a", "libssl.a", "libcrypto.a", "libxml2.a", "libiconv.a", "libnghttp2.a")]
    mapping = tmp_path / "probe.map"
    mapping.write_text("\n".join(f"{path}(member.o)\nLOAD {path}" for path in archives))

    result = recipe.validate_linkage("(NEEDED) Shared library: [libc.so.6]", [mapping], archives)

    assert set(result["consumed_private_archives"]) == {str(path) for path in archives}


def test_prefix_inventory_rejects_escape_and_shared_library(recipe, tmp_path):
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    (prefix / "libz.a").write_bytes(b"!<arch>\n")
    (prefix / "escape").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink"):
        recipe.prefix_inventory(prefix)
    (prefix / "escape").unlink()
    (prefix / "libz.so.1").write_bytes(b"dynamic")
    with pytest.raises(ValueError, match="shared"):
        recipe.prefix_inventory(prefix)


def test_prefix_inventory_rejects_internal_shared_library_alias(recipe, tmp_path):
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    (prefix / "libz.a").write_bytes(b"!<arch>\n")
    (prefix / "libz.so").symlink_to("libz.a")

    with pytest.raises(ValueError, match="shared"):
        recipe.prefix_inventory(prefix)


@pytest.fixture
def http2_provider(tmp_path):
    """Obsolete installed-byte evidence cannot qualify a source build."""
    receipt = tmp_path / "provider.json"
    receipt.write_text(json.dumps({
        "kind": "nghttp2_installed_static_provider_identity", "source_package": "nghttp2",
        "source_version": "1.68.0-2ubuntu0.2", "package": "libnghttp2-dev:amd64",
        "files": [],
        "archive_status": "system static bytes captured, exact distro upstream/debian source archives not downloaded",
    }))
    return receipt


@pytest.fixture
def http2_source(recipe, source_inputs, tmp_path):
    inputs, _, _, _ = source_inputs
    evidence = tmp_path / "http2-source-evidence"
    evidence.mkdir()
    headers = []
    with tarfile.open(inputs / recipe.SOURCES["nghttp2"]["filename"]) as bundle:
        for name in ("nghttp2.h", "nghttp2ver.h.in"):
            path = evidence / name
            path.write_bytes(bundle.extractfile(recipe.SOURCES["nghttp2"]["root"] + "/lib/includes/nghttp2/" + name).read())
            headers.append(record(path))
    patches = []
    with tarfile.open(inputs / recipe.NGHTTP2_DIST_INPUTS["nghttp2_debian"]["filename"]) as bundle:
        for stage in recipe.NGHTTP2_PATCH_STAGES:
            path = evidence / stage["name"]
            path.write_bytes(bundle.extractfile("debian/patches/" + stage["name"]).read())
            patches.append({**record(path), "name": stage["name"]})
    keyring, log = evidence / "keyring", evidence / "signature.log"
    keyring.write_bytes(b"synthetic keyring; not a signature verification\n")
    log.write_bytes(b"gpgv: Can't check signature: No public key\n")
    dsc_inputs = [inputs / recipe.SOURCES["nghttp2"]["filename"],
                  inputs / recipe.NGHTTP2_DIST_INPUTS["nghttp2_signature"]["filename"],
                  inputs / recipe.NGHTTP2_DIST_INPUTS["nghttp2_debian"]["filename"]]
    receipt = evidence / "source-access.json"
    receipt.write_text(json.dumps({
        "schema": "biominer-nghttp2-distro-source-access/v1", "source_package": "nghttp2",
        "source_version": recipe.NGHTTP2_SOURCE_VERSION, "source_publication": recipe.NGHTTP2_PUBLICATION,
        "dsc": record(inputs / recipe.NGHTTP2_DIST_INPUTS["nghttp2_dsc"]["filename"]),
        "dsc_sha256_and_sizes_checked": [{**record(path), "size": path.stat().st_size} for path in dsc_inputs],
        "patch_series": recipe.nghttp2_series(), "patches": patches, "source_headers": headers,
        "signature": {"exit_code": 2, "keyring": record(keyring), "log": record(log)},
    }))
    return receipt


@pytest.mark.parametrize("change", ("receipt", "source-version", "header", "missing-file"))
def test_http2_source_rejects_changed_identity_or_bytes(recipe, http2_source, source_inputs, change):
    receipt = http2_source
    expected = digest(receipt.read_bytes())
    value = json.loads(receipt.read_bytes())
    if change == "receipt":
        receipt.write_text(receipt.read_text() + " ")
    elif change == "source-version":
        value["source_version"] = "1.68.0-2ubuntu0.1"
        receipt.write_text(json.dumps(value))
        expected = digest(receipt.read_bytes())
    elif change == "header":
        Path(value["source_headers"][0]["path"]).write_bytes(b"unexpected header")
    else:
        value["dsc_sha256_and_sizes_checked"].pop()
        receipt.write_text(json.dumps(value))
        expected = digest(receipt.read_bytes())

    with pytest.raises(ValueError, match="nghttp2|HTTP2|checksum|source"):
        recipe.validate_nghttp2_source(receipt, expected, source_inputs[0])


def test_http2_source_preserves_signature_limit_without_importing_installed_binary(recipe, http2_source, source_inputs):
    expected = digest(http2_source.read_bytes())

    provider = recipe.validate_nghttp2_source(http2_source, expected, source_inputs[0])

    assert provider["source_version"] == "1.68.0-2ubuntu0.2"
    assert provider["signature_verified"] is False
    assert "signing key" in provider["qualification"]
    assert provider["receipt"] == record(http2_source)
    assert len(provider["distribution_inputs"]) == 4
    assert not any(Path(row["path"]).suffix == ".a" for row in provider["distribution_inputs"].values())


def test_command_failure_retains_exact_exit_and_log(recipe, tmp_path):
    (tmp_path / "logs").mkdir()
    command = [sys.executable, "-I", "-B", "-c", "print('bounded failure evidence'); raise SystemExit(7)"]

    with pytest.raises(subprocess.CalledProcessError) as caught:
        recipe.run_command(command, cwd=tmp_path, env={"PATH": "/usr/bin:/bin"}, work=tmp_path, name="fixture", timeout=10)

    assert caught.value.returncode == 7
    result = json.loads((tmp_path / "logs/fixture-command.json").read_bytes())
    assert result["exit_code"] == 7 and result["argv"] == command
    assert "bounded failure evidence" in Path(result["log"]["path"]).read_text()
    assert result["cleanup"] == "terminal"


def test_completed_prefix_rejects_wrong_receipt_hash(recipe, tmp_path):
    receipt = tmp_path / "build-receipt.json"
    receipt.write_text(json.dumps({"schema_version": 1, "kind": "private_network_static_build"}))

    with pytest.raises(ValueError, match="checksum"):
        recipe.verify_completed_prefix(receipt, "0" * 64)


def test_completed_prefix_rejects_uncompiled_receipt(recipe, tmp_path):
    receipt = tmp_path / "build-receipt.json"
    receipt.write_text(json.dumps({
        "schema_version": 1, "kind": "private_network_static_build",
        "status": "inputs_verified", "ready": False,
    }))

    with pytest.raises(ValueError, match="compiled"):
        recipe.verify_completed_prefix(receipt, digest(receipt.read_bytes()))


@pytest.fixture
def completed_prefix(recipe, source_inputs, http2_source, tmp_path, monkeypatch):
    """Synthetic validator protocol, never evidence that a native build ran."""
    inputs_dir, work, _, _ = source_inputs
    prepared = recipe.prepare_sources(inputs_dir, work)
    crypto = tmp_path / "crypto/prefix"
    crypto_files = {}
    for relative in ("include/openssl/opensslv.h", "lib/libssl.a", "lib/libcrypto.a"):
        path = crypto / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((b"!<arch>\n" if path.suffix == ".a" else b"") + relative.encode())
        crypto_files[relative] = digest(path.read_bytes())
    monkeypatch.setattr(recipe, "OPENSSL_FILES", crypto_files)
    crypto_receipt = crypto.parent / "openssl-build-receipt.json"
    crypto_receipt.write_text("{}\n")
    monkeypatch.setattr(recipe, "OPENSSL_RECEIPT_SHA256", digest(crypto_receipt.read_bytes()))
    crypto_archive = tmp_path / "openssl.tar.gz"
    crypto_archive.write_bytes(b"synthetic source archive input")
    monkeypatch.setattr(recipe, "OPENSSL_ARCHIVE_SHA256", digest(crypto_archive.read_bytes()))
    openssl = recipe.validate_openssl(crypto, crypto_archive)
    source_receipt = recipe._copy(record(http2_source), work / "inputs/nghttp2-source-access.json")
    nghttp2 = recipe.validate_nghttp2_source(source_receipt["path"], source_receipt["sha256"], work / "inputs")
    prefix = work / "prefix"
    for name in ("libcurl.a", "libxml2.a", "libiconv.a", "libcharset.a", "libz.a", "libnghttp2.a"):
        path = prefix / "lib" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"!<arch>\n" + name.encode())
    ng_source = work / "sources" / recipe.SOURCES["nghttp2"]["root"] / "lib/includes/nghttp2"
    ng_build = work / "build/nghttp2"
    archive = ng_build / "lib/libnghttp2.a"
    archive.parent.mkdir(parents=True)
    archive.write_bytes((prefix / "lib/libnghttp2.a").read_bytes())
    generated = ng_build / "lib/includes/nghttp2/nghttp2ver.h"
    generated.parent.mkdir(parents=True)
    generated.write_bytes((ng_source / "nghttp2ver.h.in").read_bytes().replace(b"@PACKAGE_VERSION@", b"1.68.0").replace(b"@PACKAGE_VERSION_NUM@", b"0x014400"))
    for name, original in (("nghttp2.h", ng_source / "nghttp2.h"), ("nghttp2ver.h", generated)):
        target = prefix / "include/nghttp2" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.read_bytes())
    ng_object = ng_build / "lib/CMakeFiles/nghttp2_static.dir/nghttp2_session.c.o"
    ng_object.parent.mkdir(parents=True)
    ng_object.write_bytes(b"synthetic nghttp2 object; not compiled")
    ng_compile = [{"directory": str(ng_build), "file": str(ng_source.parent.parent / "nghttp2_session.c"),
                   "arguments": [str(tmp_path / "tools/cc"), "-DBUILDING_NGHTTP2", "-DNGHTTP2_STATICLIB", "-fPIC", "-o", str(ng_object), "-c", str(ng_source.parent.parent / "nghttp2_session.c")]}]
    (ng_build / "compile_commands.json").write_text(json.dumps(ng_compile))
    for name in ("include/zlib.h", "include/iconv.h", "include/curl/curl.h", "include/libxml2/libxml/parser.h"):
        path = prefix / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic header\n")
    tools, pinned = {}, {}
    for name in ("cc", "cxx", "cmake", "ninja", "ld", "make", "ar", "ranlib", "patch", "readelf", "nm", "install"):
        path = tmp_path / "tools" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("synthetic tool bytes; do not execute\n" + name)
        tools[name] = {**record(path), "version_output": "synthetic unit fixture"}
        if name in recipe.PINNED_TOOLS:
            pinned[name] = (str(path), digest(path.read_bytes()))
    tools["patch"] = {**record(Path("/usr/bin/patch").resolve()), "version_output": "fixture binds the actual source-preparation patch tool"}
    monkeypatch.setattr(recipe, "PINNED_TOOLS", pinned)
    ng_outputs = recipe.verify_nghttp2_outputs(work)
    paths = {name: item["path"] for name, item in tools.items()}
    http2 = {"version": "1.68.0", "archive": str(prefix / "lib/libnghttp2.a"), "include": str(prefix / "include")}
    plan = recipe.build_plan(work, paths, crypto, jobs=4, http2=http2)
    probe_source = work / "native-probe.c"
    probe_source.write_text(recipe.PROBE)
    inputs = {"schema_version": 1, "kind": "private_network_build_inputs", "recipe": prepared["recipe"],
              "prepared_sources": record(work / "prepared-sources.json"), "source_pins": recipe.SOURCES,
              "openssl": openssl, "nghttp2": nghttp2, "toolchain": tools, "jobs": 4,
              "environment": recipe.build_environment(work, paths), "commands": plan, "probe_source": record(probe_source)}
    inputs_ref = recipe.save(work / "build-inputs.json", inputs)
    commands = [prepared["patch_application"]["command"],
                *[stage["command"] for stage in prepared["nghttp2_patch_application"]["stages"]]]
    def command(name, argv, cwd=work, timeout=300, log="synthetic protocol only; command was not executed\n"):
        logfile = work / "logs" / (name + ".log")
        logfile.write_text(log)
        commands.append(recipe.save(work / "logs" / (name + "-command.json"), {
            "argv": argv, "cwd": str(cwd), "timeout_seconds": timeout,
            "exit_code": 0, "cleanup": "terminal", "log": record(logfile),
        }))
    caches = []
    for item in plan:
        command(item["name"], item["argv"], item["cwd"], item["timeout"])
        if item["name"].endswith("-configure") and item["name"] != "libiconv-configure":
            path = work / "build" / item["name"][:-10] / "CMakeCache.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(arg[2:].split("=", 1)[0] + ":STRING=" + arg.split("=", 1)[1] for arg in item["argv"] if arg.startswith("-D")) + "\n")
            caches.append(record(path))
    obj = work / "build/unit-fixture.o"
    obj.write_bytes(b"synthetic object bytes; not a compiled native object")
    libraries = [prefix / "lib" / name for name in ("libcurl.a", "libxml2.a", "libiconv.a", "libcharset.a", "libz.a", "libnghttp2.a")]
    libraries += [crypto / "lib/libssl.a", crypto / "lib/libcrypto.a"]
    maps = []
    for name in ("native-probe.map", "native-probe-whole.map", "charset-probe-whole.map"):
        path = work / name
        path.write_text("\n".join(f"{item}(unit-fixture.o)\nLOAD {item}" for item in libraries))
        maps.append(path)
    observation = {"curl": "8.23.0", "openssl": "3.6.5", "zlib": "1.3.2", "libxml2": "2.15.4", "libiconv": "1.19", "nghttp2": "1.68.0",
                   **{key: True for key in ("http", "https", "http2", "xml", "xpath", "encoding", "gzip_xml")}}
    for name in ("probe-compile", "probe-link", "probe-dynamic", "probe-whole-dynamic", "probe-charset-whole-dynamic"):
        command(name, [paths["cc"], "unit-fixture-only"], log="(NEEDED) Shared library: [libc.so.6]\n" if name.endswith("dynamic") else "not executed\n")
    command("probe-observation", [str(work / "native-probe")], log=json.dumps(observation) + "\n")
    for name, selected in (("probe-whole-archive-pic-link", [item for item in libraries if item.name != "libcharset.a"]), ("probe-charset-whole-archive-pic-link", [prefix / "lib/libcharset.a"])):
        command(name, [paths["cc"], "-shared", "-Wl,-z,defs", "-Wl,-z,text", "-Wl,--whole-archive", *map(str, selected), "-Wl,--no-whole-archive"])
    for index, archive in enumerate(libraries):
        members = "nghttp2_session.c.o\n" if archive.name == "libnghttp2.a" else "unit-fixture.o\n"
        command(f"archive-{index}-members", [paths["ar"], "t", str(archive)], log=members)
        command(f"archive-{index}-symbols", [paths["nm"], "-g", "--defined-only", str(archive)])
        command(f"archive-{index}-relocations", [paths["readelf"], "-r", str(archive)])
    artifacts = {}
    for key, filename in (("object", "native-probe.o"), ("executable", "native-probe"), ("whole_archive_shared", "native-probe.so"), ("charset_whole_archive_shared", "charset-probe.so")):
        path = work / filename
        path.write_bytes(b"synthetic probe bytes; not executable")
        artifacts[key] = record(path)
    license_path = prefix / "licenses/fixture-license"
    license_path.parent.mkdir()
    license_path.write_text("synthetic licence; fixture is not distributable\n")
    build_inventory = recipe.save(work / "build-inventory.json", recipe.inventory(work / "build"))
    recipe._freeze(prefix)
    prefix_inventory = recipe.save(work / "prefix-inventory.json", recipe.prefix_inventory(prefix))
    value = {
        "schema_version": 1, "kind": "private_network_static_build", "status": "compiled_unverified", "ready": False,
        "input_manifest": inputs_ref, "recipe": prepared["recipe"], "prefix": str(prefix), "prefix_inventory": prefix_inventory,
        "sources": prepared["sources"], "source_archives": prepared["archives"], "zlib_patch": prepared["patch_application"],
        "nghttp2_patch": prepared["nghttp2_patch_application"], "nghttp2_build": ng_outputs,
        "openssl": openssl, "nghttp2": nghttp2, "effective_versions": observation, "toolchain": tools,
        "commands": commands, "caches": caches, "objects": [record(obj), record(ng_object)], "build_inventory": build_inventory,
        "static_archives": [{**record(path), "members": ["nghttp2_session.c.o" if path.name == "libnghttp2.a" else "unit-fixture.o"]} for path in libraries],
        "linkage": recipe.validate_linkage("", maps, libraries),
        "probe": {"source": record(probe_source), **artifacts, "observation": observation},
        "licenses": {"unit_fixture": record(license_path)},
    }
    receipt = work / "build-receipt.json"
    recipe.save(receipt, value)
    return receipt, value


def test_completed_prefix_verification_is_pure_and_returns_complete_closure(recipe, completed_prefix, monkeypatch):
    receipt, value = completed_prefix
    def no_execution(*args, **kwargs):
        raise AssertionError("completed-prefix verification must not execute a process")
    monkeypatch.setattr(recipe.subprocess, "run", no_execution)
    monkeypatch.setattr(recipe.subprocess, "Popen", no_execution)
    monkeypatch.setattr(recipe.subprocess, "check_output", no_execution)

    descriptor = recipe.verify_completed_prefix(receipt, digest(receipt.read_bytes()))

    assert descriptor["receipt"] == record(receipt)
    assert descriptor["prefix"] == value["prefix"]
    assert descriptor["build"] == value
    actual = {item["path"]: item["sha256"] for item in descriptor["files"]}
    for item in value["static_archives"] + value["objects"] + value["caches"]:
        assert actual[item["path"]] == item["sha256"]
    assert str(Path(value["prefix"]) / "include/zlib.h") in descriptor["private_files"]


@pytest.mark.parametrize("change", ("prefix-bytes", "prefix-mode", "prefix-extra", "source", "archive", "input-manifest", "command-log", "cache", "object", "map", "probe", "license"))
def test_completed_prefix_rejects_changed_closure(recipe, completed_prefix, change):
    receipt, value = completed_prefix
    if change.startswith("prefix-"):
        path = Path(value["prefix"]) / "include/zlib.h"
        path.chmod(0o644)
        if change == "prefix-bytes":
            path.write_bytes(b"unreviewed header")
        elif change == "prefix-extra":
            Path(value["prefix"]).chmod(0o755)
            Path(value["prefix"], "extra").write_bytes(b"unreviewed extra file")
    else:
        paths = {
            "source": Path(value["sources"]["curl"]["root"]) / "source.c",
            "archive": Path(value["source_archives"]["curl"]["path"]),
            "input-manifest": Path(value["input_manifest"]["path"]),
            "command-log": Path(json.loads(Path(value["commands"][0]["path"]).read_bytes())["log"]["path"]),
            "cache": Path(value["caches"][0]["path"]), "object": Path(value["objects"][0]["path"]),
            "map": Path(value["linkage"]["maps"][0]["path"]), "probe": Path(value["probe"]["executable"]["path"]),
            "license": Path(value["licenses"]["unit_fixture"]["path"]),
        }
        path = paths[change]
        path.chmod(0o644)
        path.write_bytes(b"unreviewed replacement")

    with pytest.raises(ValueError, match="checksum|inventory|source|writable"):
        recipe.verify_completed_prefix(receipt, digest(receipt.read_bytes()))


@pytest.mark.parametrize("change", ("failed-command", "missing-command", "wrong-version", "foreign-static", "foreign-map"))
def test_completed_prefix_rejects_invalid_policy_even_with_fresh_receipt_hash(recipe, completed_prefix, tmp_path, change):
    receipt, value = completed_prefix
    if change == "failed-command":
        ref = value["commands"][-1]
        path = Path(ref["path"])
        command = json.loads(path.read_bytes())
        command["exit_code"] = 7
        path.write_text(json.dumps(command))
        ref["sha256"] = digest(path.read_bytes())
    elif change == "missing-command":
        value["commands"] = value["commands"][:1]
    elif change == "wrong-version":
        value["probe"]["observation"]["curl"] = "8.18.0"
    elif change == "foreign-static":
        path = tmp_path / "foreign/libz.a"
        path.parent.mkdir()
        path.write_bytes(b"!<arch>\nforeign provider")
        value["static_archives"][4].update(record(path))
    else:
        ref = value["linkage"]["maps"][0]
        path = Path(ref["path"])
        path.write_text(path.read_text() + "\nLOAD /usr/lib/libz.a\n/usr/lib/libz.a(foreign.o)\n")
        ref["sha256"] = digest(path.read_bytes())
    receipt.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="command|version|static|foreign|private"):
        recipe.verify_completed_prefix(receipt, digest(receipt.read_bytes()))


@pytest.mark.parametrize("copied", ("lib/libnghttp2.a", "include/nghttp2/nghttp2.h", "include/nghttp2/nghttp2ver.h"))
def test_completed_prefix_rejects_rehashed_nghttp2_copy_substitution(recipe, completed_prefix, copied):
    receipt, value = completed_prefix
    prefix = Path(value["prefix"])
    path = prefix / copied
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"\nunapproved copied provider bytes\n")
    path.chmod(0o444)
    # Refresh every output reference so this exercises the source/provider
    # binding rather than rejection of a stale output inventory or receipt.
    for item in value["static_archives"]:
        if item["path"] == str(path):
            item["sha256"] = digest(path.read_bytes())
    inventory = Path(value["prefix_inventory"]["path"])
    inventory.write_text(json.dumps(recipe.prefix_inventory(prefix)))
    value["prefix_inventory"]["sha256"] = digest(inventory.read_bytes())
    receipt.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="nghttp2|HTTP2|provider"):
        recipe.verify_completed_prefix(receipt, digest(receipt.read_bytes()))


def test_nghttp2_private_pic_library_is_in_build_plan(recipe, tmp_path):
    tools = {name: f"/pinned/{name}" for name in ("cc", "cxx", "cmake", "ninja", "make")}
    prefix = tmp_path / "prefix"
    plan = recipe.build_plan(tmp_path, tools, tmp_path / "crypto", jobs=4, http2={
        "version": "1.68.0", "archive": str(prefix / "lib/libnghttp2.a"), "include": str(prefix / "include"),
    })

    assert "nghttp2-configure" in {item["name"] for item in plan}
    configure = next(item["argv"] for item in plan if item["name"] == "nghttp2-configure")
    assert "-DCMAKE_POSITION_INDEPENDENT_CODE=ON" in configure
    assert "-DBUILD_STATIC_LIBS=ON" in configure and "-DBUILD_SHARED_LIBS=OFF" in configure
    assert "-DENABLE_LIB_ONLY=ON" in configure
    for name in ("APP", "HPACK_TOOLS", "EXAMPLES", "DOC", "FAILMALLOC", "HTTP3"):
        assert f"-DENABLE_{name}=OFF" in configure
    assert "-DBUILD_TESTING=OFF" in configure
    assert plan.index(next(item for item in plan if item["name"] == "nghttp2-install")) < plan.index(next(item for item in plan if item["name"] == "curl-configure"))


def test_installed_nghttp2_identity_cannot_qualify_private_source(recipe, http2_provider, source_inputs):
    inputs, _, _, _ = source_inputs

    with pytest.raises(ValueError, match="source|distro"):
        recipe.validate_nghttp2_source(http2_provider, digest(http2_provider.read_bytes()), inputs)


@pytest.fixture
def ordered_nghttp2_sources(recipe, tmp_path, monkeypatch):
    """Three tiny successive patches prove application order and source bounds."""
    names = ("0002-Workaround-for-963648.patch", "CVE-2026-27135.patch", "CVE-2026-58055.patch")
    source = tmp_path / "nghttp2-1.68.0"
    (source / "lib").mkdir(parents=True)
    member = source / "lib/nghttp2_session.c"
    member.write_bytes(b"int source_guard = 0;\n")
    archive = archive_at(tmp_path / "nghttp2.tar.gz", "nghttp2-1.68.0", {"lib/nghttp2_session.c": member.read_bytes()})
    debian = tmp_path / "debian"
    patches = debian / "patches"
    patches.mkdir(parents=True)
    (patches / "series").write_text("\n".join(names) + "\n")
    stages = []
    for index, name in enumerate(names):
        before, after = f"int source_guard = {index};\n".encode(), f"int source_guard = {index + 1};\n".encode()
        data = b"--- a/lib/nghttp2_session.c\n+++ b/lib/nghttp2_session.c\n@@ -1 +1 @@\n-" + before + b"+" + after
        (patches / name).write_bytes(data)
        stages.append({"name": name, "sha256": digest(data), "files": {"lib/nghttp2_session.c": {"before": digest(before), "after": digest(after)}}})
    monkeypatch.setattr(recipe, "NGHTTP2_PATCH_STAGES", stages, raising=False)
    monkeypatch.setitem(recipe.SOURCES, "nghttp2", {"version": "1.68.0", "root": "nghttp2-1.68.0", "filename": archive.name, "sha256": digest(archive.read_bytes())})
    work = tmp_path / "patch-work"
    work.mkdir()
    (work / "logs").mkdir()
    return source, debian, work, archive, names


def test_nghttp2_distro_patches_apply_in_exact_order_with_source_evidence(recipe, ordered_nghttp2_sources):
    source, debian, work, archive, names = ordered_nghttp2_sources

    applied = recipe.apply_nghttp2_patches(source, debian, work, archive)

    assert (source / "lib/nghttp2_session.c").read_bytes() == b"int source_guard = 3;\n"
    assert [item["name"] for item in applied["stages"]] == list(names)
    assert all(item["before_files"] and item["after_files"] and item["command"] for item in applied["stages"])
    assert applied["source_version"] == "1.68.0-2ubuntu0.2"


@pytest.mark.parametrize("change", ("order", "patch-bytes"))
def test_nghttp2_distro_patch_rejection_precedes_source_mutation(recipe, ordered_nghttp2_sources, change):
    source, debian, work, archive, names = ordered_nghttp2_sources
    if change == "order":
        (debian / "patches/series").write_text("\n".join(reversed(names)) + "\n")
    else:
        (debian / "patches" / names[1]).write_bytes(b"unapproved security patch")

    with pytest.raises(ValueError, match="series|order|patch|checksum"):
        recipe.apply_nghttp2_patches(source, debian, work, archive)

    assert (source / "lib/nghttp2_session.c").read_bytes() == b"int source_guard = 0;\n"
    assert list((work / "logs").iterdir()) == []


@pytest.mark.parametrize("change", ("pic", "source", "target"))
def test_nghttp2_output_binding_rejects_unqualified_compile_evidence(recipe, completed_prefix, change):
    receipt, value = completed_prefix
    path = Path(value["nghttp2_build"]["compile_commands"]["path"])
    rows = json.loads(path.read_bytes())
    if change == "pic":
        rows[0]["arguments"].remove("-fPIC")
    elif change == "source":
        rows[0]["file"] = "/usr/include/nghttp2/unapproved.c"
        rows[0]["arguments"][-1] = rows[0]["file"]
    else:
        rows[0]["arguments"][rows[0]["arguments"].index("-o") + 1] = str(receipt.parent / "build/unrelated.o")
    path.write_text(json.dumps(rows))

    with pytest.raises(ValueError, match="nghttp2|PIC|source|object|compile"):
        recipe.verify_nghttp2_outputs(receipt.parent)
