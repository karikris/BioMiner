"""Corrective Arrow 25 guards use fixtures, never an installed runtime."""
from __future__ import annotations

import difflib
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_native_arrow25_network.py"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def record(path):
    return {"path": str(path), "sha256": digest(path.read_bytes())}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return record(path)


@pytest.fixture
def recipe():
    spec = importlib.util.spec_from_file_location("arrow25_network_tests", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def unordered_source_archive(tmp_path):
    """Physical member order differs from names by more than the gzip buffer."""
    files = {name: name.encode() + b"\n" + bytes(range(251)) * 2048
             for name in ("z.txt", "a.txt", "nested/y.txt", "b.txt")}
    archive = tmp_path / "unordered.tar.gz"
    packed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=packed, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as bundle:
            for relative, data in files.items():
                member = tarfile.TarInfo("source/" + relative)
                member.size = len(data)
                bundle.addfile(member, io.BytesIO(data))
            member = tarfile.TarInfo("source/link.txt")
            member.type, member.linkname = tarfile.SYMTYPE, "a.txt"
            bundle.addfile(member)
            member = tarfile.TarInfo("source/empty")
            member.type = tarfile.DIRTYPE
            bundle.addfile(member)
    archive.write_bytes(packed.getvalue())
    source = tmp_path / "source"
    for relative, data in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (source / "empty").mkdir()
    (source / "link.txt").symlink_to("a.txt")
    replacements = {"a.txt": b"reviewed transformation\n"}
    generated = {"000-generated.py": b"frozen builder template\n"}
    for relative, data in (replacements | generated).items():
        (source / relative).write_bytes(data)
    return archive, source, files, replacements, generated


def test_archive_verification_reads_compressed_members_forward_and_keeps_canonical_evidence(
        recipe, unordered_source_archive, monkeypatch):
    archive, source, original, replacements, generated = unordered_source_archive
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.getmembers()
        # Count decompressor work after the required complete metadata scan.
        reader = bundle.fileobj._buffer.raw
        original_seek, original_rewind = reader.seek, reader._rewind
        seeks, rewinds = [], []

        def seek(offset, whence=io.SEEK_SET):
            assert whence == io.SEEK_SET
            seeks.append(offset)
            return original_seek(offset, whence)

        def rewind():
            rewinds.append(reader.tell())
            return original_rewind()

        monkeypatch.setattr(reader, "seek", seek)
        monkeypatch.setattr(reader, "_rewind", rewind)

        observed = recipe.verify_archive_tree(bundle, source, replacements, generated)

    assert observed == [
        {"relative": "a.txt", "type": "file", "sha256": digest(replacements["a.txt"])},
        {"relative": "b.txt", "type": "file", "sha256": digest(original["b.txt"])},
        {"relative": "link.txt", "type": "symlink", "target": "a.txt"},
        {"relative": "nested/y.txt", "type": "file", "sha256": digest(original["nested/y.txt"])},
        {"relative": "z.txt", "type": "file", "sha256": digest(original["z.txt"])},
        {"relative": "000-generated.py", "type": "generated", "sha256": digest(generated["000-generated.py"])},
    ]
    assert seeks
    assert len(rewinds) <= 1, "member verification restarted gzip decompression"
    assert seeks == sorted(seeks), "member data was requested out of physical archive order"


@pytest.mark.parametrize("change", ["file", "replacement", "generated", "symlink", "directory", "extra", "missing", "file_symlink"])
def test_physical_archive_traversal_preserves_every_source_guard(recipe, unordered_source_archive, change):
    archive, source, original, replacements, generated = unordered_source_archive
    if change == "file":
        (source / "z.txt").write_bytes(b"changed regular source")
    elif change == "replacement":
        (source / "a.txt").write_bytes(original["a.txt"])
    elif change == "generated":
        (source / "000-generated.py").write_bytes(b"unapproved generated source")
    elif change == "symlink":
        (source / "link.txt").unlink()
        (source / "link.txt").symlink_to("b.txt")
    elif change == "directory":
        (source / "empty").rmdir()
        (source / "empty").write_bytes(b"replaced directory")
    elif change == "extra":
        (source / "extra.txt").write_bytes(b"unlisted source")
    elif change == "missing":
        (source / "z.txt").unlink()
    else:
        (source / "z.txt").unlink()
        (source / "z.txt").symlink_to("b.txt")

    with tarfile.open(archive, "r:gz") as bundle:
        with pytest.raises(ValueError, match="source|layout|generated"):
            recipe.verify_archive_tree(bundle, source, replacements, generated)


@pytest.fixture
def source_inputs(tmp_path, monkeypatch, recipe):
    allocation = "cpp/src/arrow/compute/kernels/vector_replace.cc"
    tests = "cpp/src/arrow/compute/kernels/vector_replace_test.cc"
    header = "cpp/src/arrow/compute/kernels/copy_data_internal.h"
    original = {allocation: b"before\nold allocation\nafter\n",
                tests: b"before\nold tests\nafter\n", header: b"before\nold copy\nafter\n"}
    first = {allocation: b"before\nsafe allocation\nafter\n", tests: b"before\nallocation tests\nafter\n"}
    final = {**first, tests: b"before\nallocation and slice tests\nafter\n",
             header: b"before\ncopy slice offset\nafter\n"}
    patches = []
    for name, before, after in (("allocation", {key: original[key] for key in first}, first),
                                ("offset", {tests: first[tests], header: original[header]},
                                 {tests: final[tests], header: final[header]})):
        path = tmp_path / (name + ".patch")
        path.write_text("".join("".join(difflib.unified_diff(
            data.decode().splitlines(keepends=True), after[key].decode().splitlines(keepends=True),
            fromfile="a/" + key, tofile="b/" + key)) for key, data in before.items()))
        prefix = "SECURITY" if name == "allocation" else "OFFSET"
        monkeypatch.setattr(recipe, prefix + "_PATCH_SHA256", digest(path.read_bytes()))
        monkeypatch.setattr(recipe, prefix + "_PATCH_FILES", {
            key: {"original_sha256": digest(data), "patched_sha256": digest(after[key])}
            for key, data in before.items()})
        patches.append(path)
    original["cpp/cmake_modules/ThirdpartyToolchain.cmake"] = (
        "\n".join(before for before, _ in recipe.TOOLCHAIN_PATCHES) + "\n"
        + recipe.BOOST_SELECTION_ANCHOR + "\n").encode()
    original["cpp/thirdparty/versions.txt"] = (
        "\n".join(before for before, _ in recipe.VERSION_PATCHES) + "\n").encode()
    original["python/pyproject.toml"] = (
        'build-backend = "_build_backend"\nrequires = ["scikit-build-core"]\n').encode()
    original["NOTICE.txt"] = b"upstream notice\n"
    archive = tmp_path / "arrow.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, data in original.items():
            member = tarfile.TarInfo(f"arrow-{recipe.ARROW_COMMIT}/{name}")
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
        member = tarfile.TarInfo(f"arrow-{recipe.ARROW_COMMIT}/python/NOTICE.txt")
        member.type, member.linkname = tarfile.SYMTYPE, "../NOTICE.txt"
        bundle.addfile(member)
    monkeypatch.setattr(recipe, "ARROW_ARCHIVE_SHA256", digest(archive.read_bytes()))
    return archive, tmp_path / "generation", patches, original, final


def prepare(recipe, inputs, *, resume=False):
    archive, work, patches, _, _ = inputs
    return recipe.prepare_source(archive, work, patches[0], offset_patch=patches[1], resume=resume)


def test_separate_source_identity_keeps_both_ordered_compute_fixes(recipe, source_inputs):
    source, evidence = prepare(recipe, source_inputs)
    archive, work, patches, original, final = source_inputs
    assert recipe.ARROW_COMMIT == "beccec0d0c451b7aa3e4530416ac431b3c035c69"
    assert recipe.SOURCE_POLICY == "arrow25-two-compute-patches-private-network-v1"
    assert [row["commit"] for row in evidence["security_patches"]["patches"]] == [
        recipe.SECURITY_PATCH_COMMIT, recipe.OFFSET_PATCH_COMMIT]
    for relative, data in final.items():
        assert (source / relative).read_bytes() == data
    assert (source / "NOTICE.txt").read_bytes() == original["NOTICE.txt"]
    assert (source / "python/NOTICE.txt").is_symlink()
    assert prepare(recipe, source_inputs, resume=True) == (source, evidence)
    assert "--apply-zlib-patch" not in (source / "cpp/cmake_modules/ThirdpartyToolchain.cmake").read_text()
    assert len(evidence["files"]) >= len(original)


@pytest.mark.parametrize("change", ["unrelated-source", "extra", "patch", "source-receipt"])
def test_resume_rejects_any_changed_source_or_ordered_patch_evidence(recipe, source_inputs, change):
    source, _ = prepare(recipe, source_inputs)
    _, work, _, _, _ = source_inputs
    target = {"unrelated-source": source / "NOTICE.txt", "extra": source / "extra.cc",
              "patch": work / "arrow-a769c291.patch",
              "source-receipt": work / "source-verification.json"}[change]
    target.write_bytes(b"changed evidence")
    with pytest.raises(ValueError, match="source|patch|layout"):
        prepare(recipe, source_inputs, resume=True)


@pytest.mark.parametrize("change", ["archive", "allocation", "offset"])
def test_rejected_source_input_creates_no_generation(recipe, source_inputs, change):
    archive, work, patches, _, _ = source_inputs
    {"archive": archive, "allocation": patches[0], "offset": patches[1]}[change].write_bytes(b"wrong")
    with pytest.raises(ValueError, match="checksum"):
        prepare(recipe, source_inputs)
    assert not work.exists()


def test_original_recipe_and_patch_bytes_are_not_redefined():
    root = SCRIPT.parent
    assert digest((root / "build_native_arrow.py").read_bytes()) == "65077fc43ba4542eb051b0a07a53ba147614551d192e774e13773aef0567da83"
    assert digest((root / "native_patches/arrow25-a769c291.patch").read_bytes()) == "d51722f4ddb3934a0c846b691a60a730cb77d5a814d7077cab1c5d145c54aaa1"
    assert digest((root / "native_patches/arrow25-0efbdec7.patch").read_bytes()) == "6c64e115aa53621ae7e1992417b26ab7c6c51fb265d3e64216580f7dcae87d50"


def test_version_is_derived_from_new_manifest(recipe, tmp_path):
    saved, version = recipe.preserve_build_inputs(tmp_path, {"source_policy": recipe.SOURCE_POLICY}, resume=False)
    assert version == "25.0.1+biominer." + saved["sha256"][:12]
    with pytest.raises(ValueError, match="build inputs"):
        recipe.preserve_build_inputs(tmp_path, {"source_policy": "old-policy"}, resume=True)


@pytest.mark.parametrize("jobs", [0, 5, 16])
def test_outer_build_budget_is_at_most_four(recipe, jobs):
    with pytest.raises(ValueError, match="between 1 and 4"):
        recipe.build_environment(jobs, python="/python", cc="/cc", cxx="/cxx")


@pytest.mark.parametrize("stage", ["cpp", "python"])
def test_both_stages_bind_exact_private_providers_and_persistent_backend(recipe, tmp_path, stage):
    work, network, openssl = (tmp_path / name for name in ("work", "network", "openssl"))
    commands = recipe.build_commands(tmp_path / "source", work, openssl, network_prefix=network,
                                     python="/python", cc="/cc", cxx="/cxx", jobs=4)
    command = commands["cpp_configure" if stage == "cpp" else "python_build"]
    marker = "-D" if stage == "cpp" else "--config-setting=cmake.define."
    for key, value in {"ZLIB_SOURCE": "SYSTEM", "ZLIB_LIBRARY": network / "lib/libz.a",
                       "CURL_LIBRARY": network / "lib/libcurl.a", "LIBXML2_LIBRARY": network / "lib/libxml2.a",
                       "Iconv_LIBRARY": network / "lib/libiconv.a", "NGHTTP2_LIBRARY": network / "lib/libnghttp2.a",
                       "OPENSSL_CRYPTO_LIBRARY": openssl / "lib/libcrypto.a",
                       "OPENSSL_SSL_LIBRARY": openssl / "lib/libssl.a",
                       "CMAKE_PROJECT_INCLUDE": work / "private-network.cmake"}.items():
        assert f"{marker}{key}={value}" in command
    assert "--config-setting=build-dir=" + str(work / "python-build") in commands["python_build"]
    assert "--config-setting=build.tool-args=-j4" in commands["python_build"]
    assert "-DARROW_GANDIVA=OFF" in commands["cpp_configure"]
    assert "-DPARQUET_REQUIRE_ENCRYPTION=ON" in commands["cpp_configure"]
    assert recipe.build_environment(4, python="/python", cc="/cc", cxx="/cxx")["MAKEFLAGS"] == "-j1"


def write_zlib_alias_target(recipe, work, archive):
    build = work / "cpp-build"
    reply = build / ".cmake/api/v1/reply"
    reply.mkdir(parents=True)
    target = {"name": "ZLIB::ZLIB", "id": "private-zlib", "type": "STATIC_LIBRARY", "imported": True,
              "abstract": True, "paths": {"source": ".", "build": "."},
              "artifacts": [{"path": str(archive)}]}
    write_json(reply / "zlib.json", target)
    write_json(reply / "model.json", {"kind": "codemodel", "version": {"major": 2, "minor": 9},
        "paths": {"source": str(work / f"arrow-{recipe.ARROW_COMMIT}/cpp"), "build": str(build)},
        "configurations": [{"name": "Release", "targets": [], "abstractTargets": [
            {"name": target["name"], "id": target["id"], "jsonFile": "zlib.json"}]}]})
    write_json(reply / "index-test.json", {"reply": {"codemodel-v2": {
        "kind": "codemodel", "version": {"major": 2, "minor": 9}, "jsonFile": "model.json"}}})
    return reply


def zlib_alias_cache_fixture(recipe, network_inputs, tmp_path):
    network = verified_network(recipe, network_inputs)
    work = tmp_path / "alias-work"
    reply = write_zlib_alias_target(recipe, work, Path(network["prefix"]) / "lib/libz.a")
    context = {"work": str(work), "python": "/approved/python", "cc": "/approved/cc", "cxx": "/approved/cxx",
               "openssl_prefix": str(network_inputs[2]), "network_prefix": network["prefix"]}
    metadata = work / "private-network.cmake"
    metadata.write_bytes(recipe.private_network_cmake(Path(network["prefix"]), network_inputs[2]))
    write_json(work / "build-inputs.json", {"schema_version": 1, "source_policy": recipe.SOURCE_POLICY,
        "arrow_commit": recipe.ARROW_COMMIT, "context": context, "network": network,
        "private_network_cmake": record(metadata)})
    cache = work / "cpp-build/CMakeCache.txt"
    entries = recipe.expected_cache(context, stage="cpp") | {"ZLIB_LIBRARY": "ZLIB::ZLIB"}
    cache.write_text("\n".join(f"{name}:STRING={value}" for name, value in entries.items()) + "\n")
    return cache, context, network, reply


def test_cpp_cache_accepts_upstream_zlib_alias_only_with_private_target_binding(recipe, network_inputs, tmp_path):
    cache, context, network, _ = zlib_alias_cache_fixture(recipe, network_inputs, tmp_path)

    observed = recipe.verify_cache(cache, context, stage="cpp")

    assert observed["ZLIB_LIBRARY"] == "ZLIB::ZLIB"
    requested = str(Path(network["prefix"]) / "lib/libz.a")
    assert recipe.expected_cache(context, stage="cpp")["ZLIB_LIBRARY"] == requested
    commands = recipe.build_commands(tmp_path / "source", Path(context["work"]), network_inputs[2],
        network_prefix=Path(network["prefix"]), python=context["python"], cc=context["cc"], cxx=context["cxx"], jobs=4)
    assert "-DZLIB_LIBRARY=" + requested in commands["cpp_configure"]
    assert "--config-setting=cmake.define.ZLIB_LIBRARY=" + requested in commands["python_build"]


@pytest.mark.parametrize("change", ["foreign", "extra_artifact", "missing", "duplicate", "not_imported", "local", "shared",
                                 "not_abstract", "metadata", "repinned_metadata", "unowned", "wrong_source",
                                 "replaced_archive", "missing_network", "missing_file_api"])
def test_cpp_cache_rejects_zlib_alias_without_exact_private_binding(recipe, network_inputs, tmp_path, change):
    cache, context, network, reply = zlib_alias_cache_fixture(recipe, network_inputs, tmp_path)
    work = Path(context["work"])
    target = json.loads((reply / "zlib.json").read_bytes())
    model = json.loads((reply / "model.json").read_bytes())
    inputs = json.loads((work / "build-inputs.json").read_bytes())
    archive = Path(network["prefix"]) / "lib/libz.a"
    if change == "foreign":
        foreign = tmp_path / "host/libz.a"
        foreign.parent.mkdir()
        foreign.write_bytes(b"host zlib")
        target["artifacts"] = [{"path": str(foreign)}]
    elif change == "extra_artifact":
        target["artifacts"].append({"path": str(tmp_path / "second-libz.a")})
    elif change == "missing":
        model["configurations"][0]["abstractTargets"] = []
    elif change == "duplicate":
        write_json(reply / "second-zlib.json", target | {"id": "second-private-zlib"})
        model["configurations"][0]["abstractTargets"].append(
            {"name": target["name"], "id": "second-private-zlib", "jsonFile": "second-zlib.json"})
    elif change in {"not_imported", "local", "shared", "not_abstract"}:
        key, value = {"not_imported": ("imported", False), "local": ("local", True),
                      "shared": ("type", "SHARED_LIBRARY"), "not_abstract": ("abstract", False)}[change]
        target[key] = value
    elif change in {"metadata", "repinned_metadata"}:
        metadata = work / "private-network.cmake"
        metadata.write_bytes(metadata.read_bytes() + b"set_property(TARGET ZLIB::ZLIB PROPERTY IMPORTED_LOCATION /host/libz.a)\n")
        if change == "repinned_metadata":
            inputs["private_network_cmake"] = record(metadata)
    elif change == "unowned":
        del inputs["network"]["private_files"][str(archive)]
    elif change == "wrong_source":
        inputs["network"]["private_files"][str(archive)]["source_version"] = "1.3.1"
    elif change == "replaced_archive":
        archive.write_bytes(b"replaced after provider verification")
    elif change == "missing_network":
        del inputs["network"]
    else:
        (reply / "index-test.json").unlink()
    write_json(reply / "zlib.json", target)
    write_json(reply / "model.json", model)
    write_json(work / "build-inputs.json", inputs)

    with pytest.raises(ValueError, match="ZLIB_LIBRARY"):
        recipe.verify_cache(cache, context, stage="cpp")


@pytest.mark.parametrize("alias", ["ZLIB::OTHER", "zlib::zlib", "$<TARGET_FILE:ZLIB::ZLIB>", "ZLIB::ZLIB;foreign"])
def test_cpp_cache_rejects_other_zlib_alias_spellings(recipe, network_inputs, tmp_path, alias):
    cache, context, _, _ = zlib_alias_cache_fixture(recipe, network_inputs, tmp_path)
    cache.write_text(cache.read_text().replace("ZLIB_LIBRARY:STRING=ZLIB::ZLIB\n", "ZLIB_LIBRARY:STRING=" + alias + "\n"))
    with pytest.raises(ValueError, match="ZLIB_LIBRARY"):
        recipe.verify_cache(cache, context, stage="cpp")


def test_python_cache_does_not_accept_unproven_zlib_alias_normalization(recipe, network_inputs, tmp_path):
    cache, context, _, _ = zlib_alias_cache_fixture(recipe, network_inputs, tmp_path)
    entries = recipe.expected_cache(context, stage="python") | {"ZLIB_LIBRARY": "ZLIB::ZLIB"}
    cache.write_text("\n".join(f"{name}:STRING={value}" for name, value in entries.items()) + "\n")
    with pytest.raises(ValueError, match="ZLIB_LIBRARY"):
        recipe.verify_cache(cache, context, stage="python")


def test_network_helper_cannot_execute_before_source_hash_approval(recipe, tmp_path, monkeypatch):
    marker, helper = tmp_path / "executed", tmp_path / "helper.py"
    helper.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    monkeypatch.setattr(recipe, "NETWORK_HELPER_SHA256", None)
    with pytest.raises(ValueError, match="network helper.*checksum"):
        recipe.validate_network(tmp_path / "receipt.json", "0" * 64, helper, openssl_prefix=tmp_path)
    assert not marker.exists()


@pytest.mark.parametrize("name", ["libcurl.so.4", "libcurl-gnutls.so.4", "libxml2.so.16", "libz.so.1",
                                 "libnghttp2.a", "libssl.so.3", "libcrypto.a"])
def test_affected_host_provider_is_refused_before_package_lookup(recipe, tmp_path, monkeypatch, name):
    path = tmp_path / name
    path.write_bytes(b"unapproved")
    monkeypatch.setattr(recipe, "system_provider", lambda path: pytest.fail("host provider accepted"))
    with pytest.raises(ValueError, match="unapproved.*provider"):
        recipe.native_input_provider(path, None)


def test_resume_checkpoint_rejects_changed_partial_build_outputs(recipe, tmp_path):
    (tmp_path / "partial.o").write_bytes(b"original")
    saved = recipe.write_resume_state(tmp_path)
    recipe.verify_resume_state(tmp_path, saved)
    (tmp_path / "partial.o").write_bytes(b"changed")
    with pytest.raises(ValueError, match="resume state"):
        recipe.verify_resume_state(tmp_path, saved)


@pytest.fixture
def network_inputs(tmp_path, monkeypatch, recipe):
    """An approved helper fixture returns the same public closure schema."""
    root = tmp_path / "network"
    prefix = root / "prefix"
    openssl = tmp_path / "openssl"
    private_files = {}
    paths = {
        "lib/libcurl.a": ("libcurl", "curl", "8.23.0"),
        "lib/libxml2.a": ("libxml2", "libxml2", "2.15.4"),
        "lib/libiconv.a": ("libiconv", "libiconv", "1.19"),
        "lib/libcharset.a": ("libcharset", "libiconv", "1.19"),
        "lib/libz.a": ("zlib", "zlib", "1.3.2"),
        "lib/libnghttp2.a": ("libnghttp2", "nghttp2", "1.68.0-2ubuntu0.2"),
        "include/curl/curl.h": ("libcurl", "curl", "8.23.0"),
        "include/libxml2/libxml/parser.h": ("libxml2", "libxml2", "2.15.4"),
        "include/iconv.h": ("libiconv", "libiconv", "1.19"),
        "include/zlib.h": ("zlib", "zlib", "1.3.2"),
        "include/nghttp2/nghttp2.h": ("libnghttp2", "nghttp2", "1.68.0-2ubuntu0.2"),
    }
    files = []
    for relative, (component, source_package, source_version) in paths.items():
        path = prefix / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("approved " + relative).encode())
        files.append(record(path))
        private_files[str(path)] = {"sha256": digest(path.read_bytes()), "component": component,
                                    "source_package": source_package, "source_version": source_version}
    for relative in ("lib/libssl.a", "lib/libcrypto.a", "include/openssl/ssl.h"):
        path = openssl / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("approved " + relative).encode())
        files.append(record(path))
    for name in ("original-source.c", "source.tar.gz", "build-command.log", "df84af25.patch"):
        path = root / name
        path.write_bytes(("retained " + name).encode())
        files.append(record(path))
    raw = {"schema_version": 1, "kind": "private_network_static_build", "status": "compiled_unverified",
           "ready": False, "prefix": str(prefix), "openssl": {"prefix": str(openssl)},
           "zlib_patch": {"patch_commit": "df84af25dc1942490e1d1c899a07619152a46148"}}
    receipt = root / "build-receipt.json"
    write_json(receipt, raw)
    files.append(record(receipt))
    descriptor = {"receipt": record(receipt), "prefix": str(prefix), "build": raw,
                  "files": sorted(files, key=lambda row: row["path"]), "private_files": private_files}
    helper = tmp_path / "approved-network-helper.py"
    helper.write_text("import json\nfrom pathlib import Path\n"
                      "def verify_completed_prefix(receipt_path, expected_sha256):\n"
                      "    return " + repr(descriptor) + "\n")
    monkeypatch.setattr(recipe, "NETWORK_HELPER_SHA256", digest(helper.read_bytes()), raising=False)
    return receipt, helper, openssl, descriptor


def verified_network(recipe, network_inputs):
    receipt, helper, openssl, _ = network_inputs
    return recipe.validate_network(receipt, digest(receipt.read_bytes()), helper, openssl_prefix=openssl)


def test_network_helper_is_hashed_before_any_source_is_executed(recipe, tmp_path, monkeypatch):
    marker = tmp_path / "executed"
    helper = tmp_path / "unapproved.py"
    helper.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    monkeypatch.setattr(recipe, "NETWORK_HELPER_SHA256", "0" * 64, raising=False)
    with pytest.raises(ValueError, match="network helper.*checksum"):
        recipe.validate_network(tmp_path / "receipt.json", "0" * 64, helper,
                                openssl_prefix=tmp_path / "openssl")
    assert not marker.exists()


def test_network_descriptor_binds_approved_helper_and_complete_closure(recipe, network_inputs, monkeypatch):
    receipt, helper, _, expected = network_inputs
    monkeypatch.setattr(recipe.subprocess, "run", lambda *args, **kwargs: pytest.fail("file guard ran a command"))
    observed = verified_network(recipe, network_inputs)
    assert observed == {**expected, "helper": record(helper)}
    recipe.verify_network(observed)
    with pytest.raises(ValueError, match="network receipt.*checksum"):
        recipe.validate_network(receipt, "0" * 64, helper, openssl_prefix=network_inputs[2])


@pytest.mark.parametrize("changed", ["libcurl.a", "libxml2.a", "libz.a", "libnghttp2.a", "zlib.h",
                                    "original-source.c", "source.tar.gz", "build-command.log", "df84af25.patch",
                                    "build-receipt.json", "approved-network-helper.py"])
def test_network_closure_rechecks_every_consumed_file(recipe, network_inputs, changed):
    network = verified_network(recipe, network_inputs)
    rows = [*network["files"], network["helper"]]
    target = next(Path(row["path"]) for row in rows if Path(row["path"]).name == changed)
    target.write_bytes(b"changed after review")
    with pytest.raises(ValueError, match="network.*checksum"):
        recipe.verify_network(network)


def test_network_descriptor_requires_the_same_private_openssl(recipe, network_inputs, tmp_path):
    receipt, helper, _, _ = network_inputs
    with pytest.raises(ValueError, match="network.*OpenSSL"):
        recipe.validate_network(receipt, digest(receipt.read_bytes()), helper,
                                openssl_prefix=tmp_path / "unapproved-openssl")


def test_private_network_provider_has_source_bound_identity(recipe, network_inputs):
    network = verified_network(recipe, network_inputs)
    archive = Path(network["prefix"]) / "lib/libz.a"
    observed = recipe.native_input_provider(archive, None, network=network)
    assert observed == {"kind": "private_network", "helper": network["helper"],
                        "receipt": network["receipt"], "file": record(archive),
                        "component": "zlib", "source_package": "zlib", "source_version": "1.3.2"}
    archive.write_bytes(b"unapproved replacement")
    with pytest.raises(ValueError, match="network.*checksum"):
        recipe.native_input_provider(archive, None, network=network)


@pytest.mark.parametrize("name", ["libcurl.so.4", "libcurl-gnutls.so.4", "libxml2.so.16", "libz.so.1",
                                 "libiconv.a", "libcharset.a", "libssl.so.3", "libcrypto.a", "libnghttp2.so.14"])
def test_host_network_crypto_provider_is_rejected_before_dpkg(recipe, tmp_path, monkeypatch, name):
    path = tmp_path / name
    path.write_bytes(b"affected or unapproved host provider")
    monkeypatch.setattr(recipe, "system_provider", lambda value: pytest.fail("unapproved host provider was accepted"))
    with pytest.raises(ValueError, match="unapproved.*provider"):
        recipe.native_input_provider(path, None)

def test_final_link_maps_reject_affected_host_libraries(recipe, network_inputs, tmp_path):
    network = verified_network(recipe, network_inputs)
    work = tmp_path / "arrow-work"
    (work / "cpp-build/release").mkdir(parents=True)
    archives = [row for row in network["files"] if Path(row["path"]).suffix == ".a"]
    host = tmp_path / "host/libz.so.1"
    host.parent.mkdir()
    host.write_bytes(b"host zlib")
    (work / "cpp-build/release/libarrow.so.2500.1.0.map").write_text(
        "Linker script and memory map\n" + "\n".join(row["path"] + "(member.o)" for row in archives)
        + "\nLOAD " + str(host) + "\n")
    with pytest.raises(ValueError, match="unapproved.*provider"):
        recipe.verify_link_maps(work, network_inputs[2], {"static_archives": archives}, network=network)


def test_probe_maps_cannot_substitute_for_final_private_linkage(recipe, network_inputs, tmp_path):
    network = verified_network(recipe, network_inputs)
    work = tmp_path / "arrow-work"
    probe = work / "cpp-build/CMakeFiles/CMakeScratch/TryCompile-test/probe.map"
    probe.parent.mkdir(parents=True)
    archives = [row for row in network["files"] if Path(row["path"]).suffix == ".a"]
    probe.write_text("Linker script and memory map\n" +
                     "\n".join(row["path"] + "(member.o)" for row in archives))
    final = work / "cpp-build/release/libarrow.so.2500.1.0.map"
    final.parent.mkdir(parents=True)
    final.write_text("Linker script and memory map\n")
    with pytest.raises(ValueError, match="final.*static linkage"):
        recipe.verify_link_maps(work, network_inputs[2], {"static_archives": archives}, network=network)


@pytest.mark.parametrize("mutation", [None, "metadata", "boost", "patch", "alias", "foreign_alias"])
def test_cli_rejected_approval_preserves_configure_receipt_and_checkpoint(
        recipe, source_inputs, network_inputs, tmp_path, monkeypatch, capsys, mutation):
    """Real orchestration, source patches and helper verification; no compiler."""
    archive, work, patches, _, _ = source_inputs
    receipt_path, helper, openssl, descriptor = network_inputs
    original_relative_to = Path.is_relative_to
    monkeypatch.setattr(Path, "is_relative_to", lambda self, *other:
        False if other in (("/tmp",), ("/var/tmp",)) else original_relative_to(self, *other))
    tools = {name: {"path": "/approved/" + name, "sha256": "0" * 64}
             for name in ("cc", "cxx", "cmake", "ninja", "ld")}
    builder = {"python": {"path": "/approved/python"}, "base_python": {"path": "/approved/base-python"},
               "toolchain": tools}
    monkeypatch.setattr(recipe, "validate_builder", lambda *args, **kwargs: builder)
    monkeypatch.setattr(recipe, "probe_builder", lambda *args: {"identity": "synthetic frozen builder"})
    monkeypatch.setattr(recipe, "validate_python_runtime", lambda *args: {"files": []})
    monkeypatch.setattr(recipe, "validate_openssl", lambda *args: {"version": "3.6.5"})
    monkeypatch.setattr(recipe, "expected_generated_version", lambda *args: b"exact generated fixture\n")
    monkeypatch.setattr(recipe.subprocess, "check_output", lambda *args, **kwargs: "fixture version\n")
    selected = {"targets": [], "boost_components": [], "boost_selection": {},
                "forced_bundled_source_verification": {}, "static_archives": [],
                "downloaded_archives": [], "source_trees": []}
    monkeypatch.setattr(recipe, "capture_dependencies", lambda *args, **kwargs: selected)
    stages = []

    def configure(stage_work, name, command, env):
        stages.append(name)
        assert name == "configure"
        context = {"work": str(work), "openssl_prefix": str(openssl), "network_prefix": descriptor["prefix"],
                   "python": "/approved/python", "cc": tools["cc"]["path"], "cxx": tools["cxx"]["path"]}
        cache = work / "cpp-build/CMakeCache.txt"
        entries = recipe.expected_cache(context, stage="cpp")
        if mutation in {"alias", "foreign_alias"}:
            entries["ZLIB_LIBRARY"] = "ZLIB::ZLIB"
            artifact = Path(descriptor["prefix"]) / "lib/libz.a"
            if mutation == "foreign_alias":
                artifact = tmp_path / "foreign-libz.a"
                artifact.write_bytes(b"unapproved zlib")
            write_zlib_alias_target(recipe, work, artifact)
        cache.write_text("\n".join(f"{key}:STRING={value}" for key, value in
                                  entries.items()) + "\n")
        if mutation in {"metadata", "boost", "patch"}:
            selected = {"metadata": work / "private-network.cmake",
                        "boost": work / "inputs" / recipe.DEPENDENCIES["boost"]["filename"],
                        "patch": work / "arrow-a769c291.patch"}[mutation]
            selected.write_bytes(b"changed during configuration")
        return {"command": command, "exit_code": 0}

    monkeypatch.setattr(recipe, "run_stage", configure)
    boost = tmp_path / "boost.tar.gz"
    boost.write_bytes(b"approved Boost fixture")
    monkeypatch.setitem(recipe.DEPENDENCIES["boost"], "sha256", digest(boost.read_bytes()))
    argv = ["--archive", str(archive), "--work", str(work), "--openssl-prefix", str(openssl),
            "--builder-receipt", str(tmp_path / "builder.json"), "--builder-receipt-sha256", "0" * 64,
            "--network-receipt", str(receipt_path), "--network-receipt-sha256", digest(receipt_path.read_bytes()),
            "--network-helper", str(helper), "--python", "/approved/python", "--cc", tools["cc"]["path"],
            "--cxx", tools["cxx"]["path"], "--boost-archive", str(boost),
            "--security-patch", str(patches[0]), "--offset-patch", str(patches[1])]
    configured_result = recipe.main(argv + ["--configure-only"])
    if mutation not in {None, "alias"}:
        assert configured_result != 0
        expected_error = "ZLIB_LIBRARY" if mutation == "foreign_alias" else "immutable build input checksum"
        assert expected_error in capsys.readouterr().err
        assert not (work / "configured-inputs.json").exists()
        assert json.loads((work / "build-receipt.json").read_bytes())["status"] == "failed"
        return
    assert configured_result == 0
    manifest = json.loads((work / "build-inputs.json").read_bytes())
    assert manifest["source_policy"] == recipe.SOURCE_POLICY
    assert manifest["network"]["helper"] == record(work / "private-network-helper.py")
    assert manifest["network"]["receipt"] == record(receipt_path)
    assert "zlib_patch" not in manifest
    assert not (work / "zlib-patch-application.json").exists()
    receipt = work / "build-receipt.json"
    before, tree = receipt.read_bytes(), recipe.tree_inventory(work)
    state_hash = json.loads(before)["resume_state"]["sha256"]
    result = recipe.main(argv + ["--resume", "--resume-state-sha256", state_hash,
                                "--configured-inputs-sha256", "0" * 64])
    assert result != 0
    assert "reviewed configured inputs checksum mismatch" in capsys.readouterr().err
    assert receipt.read_bytes() == before
    assert recipe.tree_inventory(work) == tree
    assert stages == ["configure"]


@pytest.mark.parametrize("fail", [False, True])
def test_network_stage_rechecks_complete_closure_even_after_failure(recipe, network_inputs, tmp_path, monkeypatch, fail):
    network = verified_network(recipe, network_inputs)
    events = []
    guard = recipe.verify_network

    def observe(value):
        events.append("verify")
        guard(value)

    def stage(*args):
        events.append("stage")
        if fail:
            raise subprocess.CalledProcessError(1, ["synthetic-stage"])
        return {"exit_code": 0}

    monkeypatch.setattr(recipe, "verify_network", observe)
    monkeypatch.setattr(recipe, "run_stage", stage)
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            recipe.run_network_stage(tmp_path, "fixture", ["fixture"], {}, network)
    else:
        assert recipe.run_network_stage(tmp_path, "fixture", ["fixture"], {}, network) == {"exit_code": 0}
    assert events == ["verify", "stage", "verify"]


@pytest.mark.parametrize("fail", [False, True])
def test_stage_provider_mutation_is_rejected_after_success_or_failure(recipe, network_inputs, tmp_path, monkeypatch, fail):
    network = verified_network(recipe, network_inputs)

    def stage(*args):
        (Path(network["prefix"]) / "include/zlib.h").write_bytes(b"changed during stage")
        if fail:
            raise subprocess.CalledProcessError(1, ["synthetic-stage"])
        return {"exit_code": 0}

    monkeypatch.setattr(recipe, "run_stage", stage)
    with pytest.raises(ValueError, match="network.*checksum"):
        recipe.run_network_stage(tmp_path, "fixture", ["fixture"], {}, network)


def test_load_only_map_does_not_prove_actual_private_archive_contribution(recipe, network_inputs, tmp_path):
    network = verified_network(recipe, network_inputs)
    work = tmp_path / "arrow-work"
    final = work / "cpp-build/release/libarrow.so.2500.1.0.map"
    final.parent.mkdir(parents=True)
    archives = [row for row in network["files"] if Path(row["path"]).suffix == ".a"]
    final.write_text("Linker script and memory map\n" + "\n".join("LOAD " + row["path"] for row in archives) + "\n")
    with pytest.raises(ValueError, match="final.*static linkage|archive contribution"):
        recipe.verify_link_maps(work, network_inputs[2], {"static_archives": archives}, network=network)
