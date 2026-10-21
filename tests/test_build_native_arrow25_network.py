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
    assert recipe.SOURCE_POLICY == "arrow25-two-compute-patches-private-network-v2"
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
    archives = {}
    for dependency, pins in recipe.DEPENDENCIES.items():
        path = tmp_path / (dependency + ".tar.gz")
        path.write_bytes(("approved " + dependency + " fixture").encode())
        monkeypatch.setitem(pins, "sha256", digest(path.read_bytes()))
        archives[dependency] = path
    argv = ["--archive", str(archive), "--work", str(work), "--openssl-prefix", str(openssl),
            "--builder-receipt", str(tmp_path / "builder.json"), "--builder-receipt-sha256", "0" * 64,
            "--network-receipt", str(receipt_path), "--network-receipt-sha256", digest(receipt_path.read_bytes()),
            "--network-helper", str(helper), "--python", "/approved/python", "--cc", tools["cc"]["path"],
            "--cxx", tools["cxx"]["path"],
            "--security-patch", str(patches[0]), "--offset-patch", str(patches[1])]
    for dependency, path in archives.items():
        argv += ["--" + dependency.replace("_", "-") + "-archive", str(path)]
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


# Frozen reviewed cohort identities. Tests never load the external acquisition draft.
EXPECTED_SDK_COHORT = {
    "aws_sdk_cpp": {
        "variable": "AWSSDK",
        "version": "1.11.914",
        "build_version": "d207a37ecb8dc33b5e131d0d59843d264b7f7e86",
        "commit": "d207a37ecb8dc33b5e131d0d59843d264b7f7e86",
        "sha256": "108382fed71e70ada73d4aaa45e0ba54777f7eca6bbd44ac24b9af9ef54f069e",
        "filename": "aws-sdk-cpp-d207a37ecb8dc33b5e131d0d59843d264b7f7e86.tar.gz",
        "url": "https://github.com/aws/aws-sdk-cpp/archive/d207a37ecb8dc33b5e131d0d59843d264b7f7e86.tar.gz",
        "env": "ARROW_AWSSDK_URL",
        "old_version": "1.11.778",
        "old_sha256": "06371849792f1a8633c3e0592a17b4700f39c748ad374f4e9a6592739ecc4d9c"
    },
    "aws_crt_cpp": {
        "variable": "AWS_CRT_CPP",
        "version": "0.43.8",
        "build_version": "9c319786d1d83d5ace6a9de817dcfe1861003609",
        "commit": "9c319786d1d83d5ace6a9de817dcfe1861003609",
        "sha256": "f107357efa7d6b9be783d96d49cdaf000fe881417c431ae9fe27faf1bb55f431",
        "filename": "aws-crt-cpp-9c319786d1d83d5ace6a9de817dcfe1861003609.tar.gz",
        "url": "https://github.com/awslabs/aws-crt-cpp/archive/9c319786d1d83d5ace6a9de817dcfe1861003609.tar.gz",
        "env": "ARROW_AWS_CRT_CPP_URL",
        "old_version": "v0.38.0",
        "old_sha256": "f57b82f0a9e6890e4d47aa1c48b9324b5beb5b58000715c4c6ef9f147be13891"
    },
    "aws_c_auth": {
        "variable": "AWS_C_AUTH",
        "version": "4b5d524bf1a511b05e0fffe5bdc51800770b9427",
        "build_version": "4b5d524bf1a511b05e0fffe5bdc51800770b9427",
        "commit": "4b5d524bf1a511b05e0fffe5bdc51800770b9427",
        "sha256": "b4f6690b382abe049d543804f40230e667a95a18f64506ef7647925e8c6b9b67",
        "filename": "aws-c-auth-4b5d524bf1a511b05e0fffe5bdc51800770b9427.tar.gz",
        "url": "https://github.com/awslabs/aws-c-auth/archive/4b5d524bf1a511b05e0fffe5bdc51800770b9427.tar.gz",
        "env": "ARROW_AWS_C_AUTH_URL",
        "old_version": "v0.10.1",
        "old_sha256": "85d737f0f735256f1931e85e4cadbe228d88698f7b59a9b390b49ef5d0778a43"
    },
    "aws_c_cal": {
        "variable": "AWS_C_CAL",
        "version": "8aa2a48a09f93c65d4cf06388e143a6584de6321",
        "build_version": "8aa2a48a09f93c65d4cf06388e143a6584de6321",
        "commit": "8aa2a48a09f93c65d4cf06388e143a6584de6321",
        "sha256": "29706ab10a4d95a74db278925b82c85077c08224c0efe837841d2354555c8817",
        "filename": "aws-c-cal-8aa2a48a09f93c65d4cf06388e143a6584de6321.tar.gz",
        "url": "https://github.com/awslabs/aws-c-cal/archive/8aa2a48a09f93c65d4cf06388e143a6584de6321.tar.gz",
        "env": "ARROW_AWS_C_CAL_URL",
        "old_version": "v0.9.13",
        "old_sha256": "80b7c6087b0af461b4483e4c9483aea2e0dac5d9fb2289b057159ea6032409e1"
    },
    "aws_c_common": {
        "variable": "AWS_C_COMMON",
        "version": "f7d471b1feaea9a310851672580f9ac7e031cfc0",
        "build_version": "f7d471b1feaea9a310851672580f9ac7e031cfc0",
        "commit": "f7d471b1feaea9a310851672580f9ac7e031cfc0",
        "sha256": "a87a89fa6678fda59dd66ea9246e70d837bf71ecec39829ec712e6863d1453e4",
        "filename": "aws-c-common-f7d471b1feaea9a310851672580f9ac7e031cfc0.tar.gz",
        "url": "https://github.com/awslabs/aws-c-common/archive/f7d471b1feaea9a310851672580f9ac7e031cfc0.tar.gz",
        "env": "ARROW_AWS_C_COMMON_URL",
        "old_version": "v0.12.6",
        "old_sha256": "138822ecdcaff1d702f37d4751f245847d088592724921cc6bf61c232b198d6b"
    },
    "aws_c_compression": {
        "variable": "AWS_C_COMPRESSION",
        "version": "d8264e64f698341eb03039b96b4f44702a9b3f83",
        "build_version": "d8264e64f698341eb03039b96b4f44702a9b3f83",
        "commit": "d8264e64f698341eb03039b96b4f44702a9b3f83",
        "sha256": "537971fe42e989b638e0f436683357ed3d07f07b06b910b1c37ab3a8094bb8fd",
        "filename": "aws-c-compression-d8264e64f698341eb03039b96b4f44702a9b3f83.tar.gz",
        "url": "https://github.com/awslabs/aws-c-compression/archive/d8264e64f698341eb03039b96b4f44702a9b3f83.tar.gz",
        "env": "ARROW_AWS_C_COMPRESSION_URL",
        "old_version": "v0.3.2",
        "old_sha256": "f93f5a5d8b3fee3a6d97b14ba279efacd4d4016ef9cc7dc4be7d43519ecfbe93"
    },
    "aws_c_event_stream": {
        "variable": "AWS_C_EVENT_STREAM",
        "version": "51bef3c44e1058b1689751539170b2e0f589ccdb",
        "build_version": "51bef3c44e1058b1689751539170b2e0f589ccdb",
        "commit": "51bef3c44e1058b1689751539170b2e0f589ccdb",
        "sha256": "b890d4940957d1ddade1a9d7de4adc37b7cb98dfdfdfec7436527fec8cb558df",
        "filename": "aws-c-event-stream-51bef3c44e1058b1689751539170b2e0f589ccdb.tar.gz",
        "url": "https://github.com/awslabs/aws-c-event-stream/archive/51bef3c44e1058b1689751539170b2e0f589ccdb.tar.gz",
        "env": "ARROW_AWS_C_EVENT_STREAM_URL",
        "old_version": "v0.7.0",
        "old_sha256": "88835b4c78462547917f622fd9dda45c991b7e356d9c07e2f0537d4d97fbd4fb"
    },
    "aws_c_http": {
        "variable": "AWS_C_HTTP",
        "version": "8aefd899fc3210bfd0e3fd414011a3cb708bf6e4",
        "build_version": "8aefd899fc3210bfd0e3fd414011a3cb708bf6e4",
        "commit": "8aefd899fc3210bfd0e3fd414011a3cb708bf6e4",
        "sha256": "61b1770e2e3a2bf046005cbfa8e26d621d8f266f05d03589ebee36bb4a0ac39f",
        "filename": "aws-c-http-8aefd899fc3210bfd0e3fd414011a3cb708bf6e4.tar.gz",
        "url": "https://github.com/awslabs/aws-c-http/archive/8aefd899fc3210bfd0e3fd414011a3cb708bf6e4.tar.gz",
        "env": "ARROW_AWS_C_HTTP_URL",
        "old_version": "v0.10.13",
        "old_sha256": "d8352e7a1fb1996694a4dc31219ce03452882abf8d0858c104727f975e11b9c7"
    },
    "aws_c_io": {
        "variable": "AWS_C_IO",
        "version": "28e4eb351a536daf1627bf69279e50671f7729b5",
        "build_version": "28e4eb351a536daf1627bf69279e50671f7729b5",
        "commit": "28e4eb351a536daf1627bf69279e50671f7729b5",
        "sha256": "ea51cbe20981ed34b7a8c01a60c848e4ccb716a29cd740ead07ffd61be5169c3",
        "filename": "aws-c-io-28e4eb351a536daf1627bf69279e50671f7729b5.tar.gz",
        "url": "https://github.com/awslabs/aws-c-io/archive/28e4eb351a536daf1627bf69279e50671f7729b5.tar.gz",
        "env": "ARROW_AWS_C_IO_URL",
        "old_version": "v0.26.3",
        "old_sha256": "521fd0848fca661130bbb7278a414d7a38bdcb9bc8ffa89f6660d84e5838a303"
    },
    "aws_c_mqtt": {
        "variable": "AWS_C_MQTT",
        "version": "e35b9ca3f9fcbf1a972c831c5e79046ce56959d1",
        "build_version": "e35b9ca3f9fcbf1a972c831c5e79046ce56959d1",
        "commit": "e35b9ca3f9fcbf1a972c831c5e79046ce56959d1",
        "sha256": "54d1987ce06eb7b39faace854818e9322d3df79cb310ada350beca0b12cb89cc",
        "filename": "aws-c-mqtt-e35b9ca3f9fcbf1a972c831c5e79046ce56959d1.tar.gz",
        "url": "https://github.com/awslabs/aws-c-mqtt/archive/e35b9ca3f9fcbf1a972c831c5e79046ce56959d1.tar.gz",
        "env": "ARROW_AWS_C_MQTT_URL",
        "old_version": "v0.15.2",
        "old_sha256": "66f3f5edff4ad1f765a86d3342b6017d0f29f950c1c24f8c1edacdc895202edc"
    },
    "aws_c_s3": {
        "variable": "AWS_C_S3",
        "version": "226c3e6937e3d5f7db9e10651f7bd20575e98187",
        "build_version": "226c3e6937e3d5f7db9e10651f7bd20575e98187",
        "commit": "226c3e6937e3d5f7db9e10651f7bd20575e98187",
        "sha256": "a8f33816bf7aff38467fb15ca233a33a9113be78dad47c8a7ee30d323b42ba74",
        "filename": "aws-c-s3-226c3e6937e3d5f7db9e10651f7bd20575e98187.tar.gz",
        "url": "https://github.com/awslabs/aws-c-s3/archive/226c3e6937e3d5f7db9e10651f7bd20575e98187.tar.gz",
        "env": "ARROW_AWS_C_S3_URL",
        "old_version": "v0.12.0",
        "old_sha256": "1a8a8ceda0585d52028a1f3daa5861f924e7d8d2f6a17bec05813dc0b74d6eed"
    },
    "aws_c_sdkutils": {
        "variable": "AWS_C_SDKUTILS",
        "version": "a1cc19f53b63658f1b1400b36f199eafeeb895a6",
        "build_version": "a1cc19f53b63658f1b1400b36f199eafeeb895a6",
        "commit": "a1cc19f53b63658f1b1400b36f199eafeeb895a6",
        "sha256": "f8ef9e93b30f9833d9c9ea474110c8fa2d29fd78b885546b07e68558ece3bb1c",
        "filename": "aws-c-sdkutils-a1cc19f53b63658f1b1400b36f199eafeeb895a6.tar.gz",
        "url": "https://github.com/awslabs/aws-c-sdkutils/archive/a1cc19f53b63658f1b1400b36f199eafeeb895a6.tar.gz",
        "env": "ARROW_AWS_C_SDKUTILS_URL",
        "old_version": "v0.2.4",
        "old_sha256": "493cbed4fa57e0d4622fcff044e11305eb4fc12445f32c8861025597939175fc"
    },
    "aws_checksums": {
        "variable": "AWS_CHECKSUMS",
        "version": "1d5f2f1f3e5d013aae8810878ceb5b3f6f258c4e",
        "build_version": "1d5f2f1f3e5d013aae8810878ceb5b3f6f258c4e",
        "commit": "1d5f2f1f3e5d013aae8810878ceb5b3f6f258c4e",
        "sha256": "c58d8c5319319dcade0926da44ad12effecb21372983e22835a657a32f03c9c9",
        "filename": "aws-checksums-1d5f2f1f3e5d013aae8810878ceb5b3f6f258c4e.tar.gz",
        "url": "https://github.com/awslabs/aws-checksums/archive/1d5f2f1f3e5d013aae8810878ceb5b3f6f258c4e.tar.gz",
        "env": "ARROW_AWS_CHECKSUMS_URL",
        "old_version": "v0.2.10",
        "old_sha256": "cb6509f75e42ee25c372a6d379e8582ce5179e5335183842e808f7d8abb0c314"
    },
    "s2n_tls": {
        "variable": "S2N_TLS",
        "version": "1.7.12",
        "build_version": "6459002cd61e4bf3bbc53db23f068b21da9ff6bd",
        "commit": "6459002cd61e4bf3bbc53db23f068b21da9ff6bd",
        "sha256": "94a57c4f4713cc20088b723bcdf0bb254e32a559ffcbed583e67fe3d8ddec442",
        "filename": "s2n-tls-6459002cd61e4bf3bbc53db23f068b21da9ff6bd.tar.gz",
        "url": "https://github.com/aws/s2n-tls/archive/6459002cd61e4bf3bbc53db23f068b21da9ff6bd.tar.gz",
        "env": "ARROW_S2N_TLS_URL",
        "old_version": "v1.7.0",
        "old_sha256": "a6e8228e238239bb3c17b1eda3ed702bcbb2eaebc792eac4d754cc5619b0ea06"
    },
    "thrift": {
        "variable": "THRIFT",
        "version": "0.25.0",
        "build_version": "0.25.0",
        "commit": "27e8a425ffb498e190df3a12e239326bf5ba9ed6",
        "sha256": "66da4707214c54c94bac082103dc67adaf9e08925662700f269170a7b534b214",
        "filename": "thrift-27e8a425ffb498e190df3a12e239326bf5ba9ed6.tar.gz",
        "url": "https://www.apache.org/dyn/closer.lua/thrift/0.25.0/thrift-0.25.0.tar.gz?action=download",
        "env": "ARROW_THRIFT_URL",
        "old_version": "0.22.0",
        "old_sha256": "794a0e455787960d9f27ab92c38e34da27e8deeda7a5db0e59dc64a00df8a1e5"
    },
    "grpc": {
        "variable": "GRPC",
        "version": "1.83.1",
        "build_version": "v1.83.1",
        "commit": "aae267021b1ac256f8b9038d0ef528c3798cc137",
        "sha256": "60caa8397426d8a500e3e57ab6a49d1cb5aa62a36f2591eb9da3d77fa38ad8c9",
        "filename": "grpc-v1.83.1.tar.gz",
        "url": "https://github.com/grpc/grpc/archive/v1.83.1.tar.gz",
        "env": "ARROW_GRPC_URL",
        "old_version": "v1.76.0",
        "old_sha256": "0af37b800953130b47c075b56683ee60bdc3eda3c37fc6004193f5b569758204"
    },
    "protobuf": {
        "variable": "PROTOBUF",
        "version": "35.1",
        "build_version": "v35.1",
        "commit": "35cd01f9fe9afbeea38cc7b979a3b6bfcde82c03",
        "sha256": "f0b6838e7522a8da96126d487068c959bc624926368f3024ac8fd03abd0a1ac4",
        "filename": "protobuf-35.1.tar.gz",
        "url": "https://github.com/protocolbuffers/protobuf/releases/download/v35.1/protobuf-35.1.tar.gz",
        "env": "ARROW_PROTOBUF_URL",
        "old_version": "v31.1",
        "old_sha256": "12bfd76d27b9ac3d65c00966901609e020481b9474ef75c7ff4601ac06fa0b82"
    },
    "absl": {
        "variable": "ABSL",
        "version": "20250512.1",
        "build_version": "20250512.1",
        "commit": "76bb24329e8bf5f39704eb10d21b9a80befa7c81",
        "sha256": "9b7a064305e9fd94d124ffa6cc358592eb42b5da588fb4e07d09254aa40086db",
        "filename": "absl-20250512.1.tar.gz",
        "url": "https://github.com/abseil/abseil-cpp/archive/20250512.1.tar.gz",
        "env": "ARROW_ABSL_URL",
        "old_version": "20250127.0",
        "old_sha256": "16242f394245627e508ec6bb296b433c90f8d914f73b9c026fddb905e27276e8"
    },
    "cares": {
        "variable": "CARES",
        "version": "1.34.9",
        "build_version": "1.34.9",
        "commit": "bc91829cce983a35ec01f5dbc05465858a3df469",
        "sha256": "9364b70a12eb59a06830fb488d9357f90100dff0d98b23ea3593a183d43e6f2f",
        "filename": "c-ares-1.34.9.tar.gz",
        "url": "https://github.com/c-ares/c-ares/releases/download/v1.34.9/c-ares-1.34.9.tar.gz",
        "env": "ARROW_CARES_URL",
        "old_version": "1.34.6",
        "old_sha256": "912dd7cc3b3e8a79c52fd7fb9c0f4ecf0aaa73e45efda880266a2d6e26b84ef5"
    },
    "rapidjson": {
        "variable": "RAPIDJSON",
        "version": "24b5e7a8b27f42fa16b96fc70aade9106cf7102f",
        "build_version": "24b5e7a8b27f42fa16b96fc70aade9106cf7102f",
        "commit": "24b5e7a8b27f42fa16b96fc70aade9106cf7102f",
        "sha256": "2d2601a82d2d3b7e143a3c8d43ef616671391034bc46891a9816b79cf2d3e7a8",
        "filename": "rapidjson-24b5e7a8b27f42fa16b96fc70aade9106cf7102f.tar.gz",
        "url": "https://github.com/Tencent/rapidjson/archive/24b5e7a8b27f42fa16b96fc70aade9106cf7102f.tar.gz",
        "env": "ARROW_RAPIDJSON_URL",
        "old_version": "232389d4f1012dddec4ef84861face2d2ba85709",
        "old_sha256": "b9290a9a6d444c8e049bd589ab804e0ccf2b05dc5984a19ed5ae75d090064806"
    },
    "jemalloc": {
        "variable": "JEMALLOC",
        "version": "5.3.1",
        "build_version": "5.3.1",
        "commit": "81034ce1f1373e37dc865038e1bc8eeecf559ce8",
        "sha256": "3826bc80232f22ed5c4662f3034f799ca316e819103bdc7bb99018a421706f92",
        "filename": "jemalloc-5.3.1.tar.bz2",
        "url": "https://github.com/jemalloc/jemalloc/releases/download/5.3.1/jemalloc-5.3.1.tar.bz2",
        "env": "ARROW_JEMALLOC_URL",
        "old_version": "5.3.0",
        "old_sha256": "2db82d1e7119df3e71b7640219b6dfe84789bc0537983c3b7ac4f7189aecfeaa"
    }
}


def source_transform_fixture(recipe, source_inputs, *, versions=None, toolchain=None):
    original = source_inputs[3]
    members = {
        "cpp/cmake_modules/ThirdpartyToolchain.cmake":
            original["cpp/cmake_modules/ThirdpartyToolchain.cmake"],
        "cpp/thirdparty/versions.txt": (
            "\n".join(f"ARROW_{row['variable']}_BUILD_VERSION={row['old_version']}\n"
                      f"ARROW_{row['variable']}_BUILD_SHA256_CHECKSUM={row['old_sha256']}"
                      for row in EXPECTED_SDK_COHORT.values()) + "\n"
            + "ARROW_BOOST_BUILD_VERSION=1.88.0\n"
            + "ARROW_BOOST_BUILD_SHA256_CHECKSUM=dcea50f40ba1ecfc448fdf886c0165cf3e525fef2c9e3e080b9804e8117b9694\n"
            + "ARROW_UNRELATED_BUILD_VERSION=keep-this-source-identity\n").encode(),
    }
    if versions is not None:
        members["cpp/thirdparty/versions.txt"] = versions
    if toolchain is not None:
        members["cpp/cmake_modules/ThirdpartyToolchain.cmake"] = toolchain
    packed = io.BytesIO()
    with tarfile.open(fileobj=packed, mode="w") as archive:
        for relative, data in members.items():
            member = tarfile.TarInfo("arrow/" + relative)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    packed.seek(0)
    return packed, members


RAPIDJSON_TARGET_BLOCK = (
    '  add_library(RapidJSON INTERFACE IMPORTED)\n'
    '  target_include_directories(RapidJSON INTERFACE "${rapidjson_SOURCE_DIR}/include")\n'
    '  add_dependencies(RapidJSON rapidjson)'
)


def rapidjson_transform_fixture(recipe, source_inputs, *, target_block=RAPIDJSON_TARGET_BLOCK):
    # Independent Arrow25 wrapper: the common fixture already carries every
    # other transformation anchor, and may also carry this exact block later.
    original = source_inputs[3]["cpp/cmake_modules/ThirdpartyToolchain.cmake"]
    original = original.replace(RAPIDJSON_TARGET_BLOCK.encode(), b"")
    wrapper = (
        '\nfunction(build_rapidjson)\n'
        '  set(RAPIDJSON_BUILD_DOC OFF)\n'
        '  set(RAPIDJSON_BUILD_EXAMPLES OFF)\n'
        '  set(RAPIDJSON_BUILD_TESTS OFF)\n'
        '  fetchcontent_makeavailable(rapidjson)\n'
        + target_block + '\n'
        '  set(RAPIDJSON_VENDORED TRUE PARENT_SCOPE)\n'
        'endfunction()\n'
    )
    return source_transform_fixture(recipe, source_inputs, toolchain=original + wrapper.encode())


@pytest.mark.parametrize("upstream_target", [True, False], ids=["upstream-exported-interface", "legacy-fallback"])
def test_rapidjson_target_compatibility_preserves_build_headers_and_export(
        recipe, source_inputs, tmp_path, upstream_target):
    packed, _ = rapidjson_transform_fixture(recipe, source_inputs)
    with tarfile.open(fileobj=packed) as archive:
        transformed = recipe.source_transformations(archive, "arrow")[
            "cpp/cmake_modules/ThirdpartyToolchain.cmake"].decode()
    wrapper = "function(build_rapidjson)\n" + transformed.split(
        "function(build_rapidjson)\n", 1)[1].split("\nendfunction()", 1)[0] + "\nendfunction()\n"
    harness = tmp_path / "rapidjson-cmake"
    headers = harness / "upstream/include"
    headers.mkdir(parents=True)
    (harness / "CMakeLists.txt").write_text(
        'cmake_minimum_required(VERSION 3.29)\n'
        'project(rapidjson_compatibility LANGUAGES NONE)\n'
        f'set(PROVIDES_RAPIDJSON {"ON" if upstream_target else "OFF"})\n'
        'set(rapidjson_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/upstream")\n'
        'function(fetchcontent_makeavailable dependency)\n'
        '  if(NOT dependency STREQUAL "rapidjson")\n'
        '    message(FATAL_ERROR "unexpected dependency")\n'
        '  endif()\n'
        '  add_custom_target(rapidjson)\n'
        '  if(PROVIDES_RAPIDJSON)\n'
        '    add_library(RapidJSON INTERFACE)\n'
        '    target_include_directories(RapidJSON INTERFACE "$<INSTALL_INTERFACE:include>")\n'
        '    set_property(TARGET RapidJSON PROPERTY BIOMINER_UPSTREAM_MARKER preserved)\n'
        '    install(TARGETS RapidJSON EXPORT RapidJSON-targets)\n'
        '    install(EXPORT RapidJSON-targets DESTINATION lib/cmake/RapidJSON)\n'
        '  endif()\n'
        'endfunction()\n'
        'add_library(unrelated INTERFACE)\n'
        'set_property(TARGET unrelated PROPERTY BIOMINER_UPSTREAM_MARKER untouched)\n'
        + wrapper +
        'build_rapidjson()\n'
        'add_library(consumer INTERFACE)\n'
        'target_link_libraries(consumer INTERFACE RapidJSON)\n'
        'get_target_property(kind RapidJSON TYPE)\n'
        'get_target_property(imported RapidJSON IMPORTED)\n'
        'get_target_property(marker RapidJSON BIOMINER_UPSTREAM_MARKER)\n'
        'get_target_property(includes RapidJSON INTERFACE_INCLUDE_DIRECTORIES)\n'
        'get_target_property(dependencies RapidJSON MANUALLY_ADDED_DEPENDENCIES)\n'
        'get_target_property(unrelated_marker unrelated BIOMINER_UPSTREAM_MARKER)\n'
        'file(WRITE "${CMAKE_BINARY_DIR}/properties.txt" '
        '"${kind}\\n${imported}\\n${marker}\\n${includes}\\n${dependencies}\\n${RAPIDJSON_VENDORED}\\n${unrelated_marker}\\n")\n'
        'file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/effective-includes.txt" '
        'CONTENT "$<TARGET_GENEX_EVAL:consumer,$<TARGET_PROPERTY:RapidJSON,INTERFACE_INCLUDE_DIRECTORIES>>")\n'
    )
    build = tmp_path / "rapidjson-cmake-build"
    result = subprocess.run(["/usr/bin/cmake", "-S", str(harness), "-B", str(build)],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    kind, imported, marker, includes, dependencies, vendored, unrelated = (
        build / "properties.txt").read_text().splitlines()
    assert kind == "INTERFACE_LIBRARY"
    assert dependencies == "rapidjson"
    assert vendored == "TRUE"
    assert unrelated == "untouched"
    assert [value for value in (build / "effective-includes.txt").read_text().split(";") if value] == [str(headers)]
    if upstream_target:
        assert imported == "FALSE"
        assert marker == "preserved"
        assert "$<INSTALL_INTERFACE:include>" in includes
        exports = list((build / "CMakeFiles/Export").rglob("RapidJSON-targets.cmake"))
        assert len(exports) == 1
        assert str(headers) not in exports[0].read_text()
    else:
        assert imported == "TRUE"


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_rapidjson_target_compatibility_requires_exact_original_block(recipe, source_inputs, mutation):
    block = "" if mutation == "missing" else RAPIDJSON_TARGET_BLOCK + "\n" + RAPIDJSON_TARGET_BLOCK
    packed, _ = rapidjson_transform_fixture(recipe, source_inputs, target_block=block)
    with tarfile.open(fileobj=packed) as archive:
        with pytest.raises(ValueError, match="anchor|transformation"):
            recipe.source_transformations(archive, "arrow")


AZURE_FETCH_BLOCK = (
    '  fetchcontent_makeavailable(azure_sdk)\n'
    '  if(CMAKE_VERSION VERSION_LESS 3.28)'
)


def azure_xml_transform_fixture(recipe, source_inputs, *, fetch_block=AZURE_FETCH_BLOCK):
    original = source_inputs[3]["cpp/cmake_modules/ThirdpartyToolchain.cmake"]
    original = original.replace(AZURE_FETCH_BLOCK.encode(), b"")
    wrapper = (
        '\nfunction(build_azure_sdk)\n'
        + fetch_block + '\n'
        '    set_property(DIRECTORY ${azure_sdk_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL TRUE)\n'
        '  endif()\n'
        '  set(AZURE_SDK_VENDORED TRUE PARENT_SCOPE)\n'
        'endfunction()\n'
    )
    packed, _ = source_transform_fixture(recipe, source_inputs, toolchain=original + wrapper.encode())
    with tarfile.open(fileobj=packed) as archive:
        transformed = recipe.source_transformations(archive, "arrow")[
            "cpp/cmake_modules/ThirdpartyToolchain.cmake"].decode()
    transformed_wrapper = "function(build_azure_sdk)\n" + transformed.split(
        "function(build_azure_sdk)\n", 1)[1].split("\nendfunction()", 1)[0] + "\nendfunction()\n"
    return wrapper.lstrip("\n"), transformed_wrapper


def azure_xml_cmake_graph(tmp_path, wrapper, *, target_variant="valid", windows=False):
    """Model Azure's raw XML archive link without borrowing its target interface."""
    harness = tmp_path / "source"
    expected = harness / "sdk-source/sdk/storage/azure-storage-common"
    actual = harness / "foreign-source" if target_variant == "foreign_source" else expected
    actual.mkdir(parents=True)
    expected.mkdir(parents=True, exist_ok=True)
    (harness / "iconv.c").write_text("int biominer_iconv_open(void) { return 19; }\n")
    (harness / "xml.c").write_text(
        "extern int biominer_iconv_open(void);\n"
        "int biominer_xml_read(void) { return biominer_iconv_open(); }\n")
    (harness / "consumer.c").write_text(
        "extern int biominer_azure_read(void);\n"
        "int biominer_consume(void) { return biominer_azure_read(); }\n")
    (actual / "common.c").write_text(
        "extern int biominer_xml_read(void);\n"
        "int biominer_azure_read(void) { return biominer_xml_read(); }\n")
    if target_variant == "missing":
        target = 'add_library(unrelated_common STATIC common.c)\n'
    elif target_variant == "alias":
        target = (
            'add_library(real_common STATIC common.c)\n'
            'add_library(azure-storage-common ALIAS real_common)\n')
    elif target_variant == "imported":
        target = (
            'add_library(azure-storage-common STATIC IMPORTED GLOBAL)\n'
            'set_property(TARGET azure-storage-common PROPERTY IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/libxml_stub.a")\n')
    else:
        kind = "SHARED" if target_variant == "shared" else "STATIC"
        target = (
            f'add_library(azure-storage-common {kind} common.c)\n'
            'set_property(TARGET azure-storage-common PROPERTY POSITION_INDEPENDENT_CODE ON)\n'
            'target_link_libraries(azure-storage-common PRIVATE ${LIBXML2_LIBRARIES})\n'
            'add_dependencies(azure-storage-common owned_xml)\n')
    common_source = (
        '# Exact Azure-like upstream raw-variable dependency remains unchanged.\n' + target)
    common_cmake = actual / "CMakeLists.txt"
    common_cmake.write_text(common_source)
    (harness / "CMakeLists.txt").write_text(
        'cmake_minimum_required(VERSION 3.29)\n'
        'project(azure_private_xml LANGUAGES C)\n'
        'set(CMAKE_C_COMPILER /usr/bin/cc)\n'
        'add_library(owned_iconv STATIC iconv.c)\n'
        'set_target_properties(owned_iconv PROPERTIES OUTPUT_NAME iconv_stub POSITION_INDEPENDENT_CODE ON)\n'
        'add_library(owned_xml STATIC xml.c)\n'
        'set_target_properties(owned_xml PROPERTIES OUTPUT_NAME xml_stub POSITION_INDEPENDENT_CODE ON)\n'
        'add_library(Iconv::Iconv STATIC IMPORTED GLOBAL)\n'
        'set_target_properties(Iconv::Iconv PROPERTIES IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/libiconv_stub.a")\n'
        'add_dependencies(Iconv::Iconv owned_iconv)\n'
        'add_library(LibXml2::LibXml2 STATIC IMPORTED GLOBAL)\n'
        'set_target_properties(LibXml2::LibXml2 PROPERTIES IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/libxml_stub.a" '
        'INTERFACE_LINK_LIBRARIES Iconv::Iconv)\n'
        'add_dependencies(LibXml2::LibXml2 owned_xml)\n'
        'set(LIBXML2_LIBRARIES "${CMAKE_BINARY_DIR}/libxml_stub.a")\n'
        'macro(fetchcontent_makeavailable dependency)\n'
        '  if(NOT "${dependency}" STREQUAL "azure_sdk")\n'
        '    message(FATAL_ERROR "unexpected dependency")\n'
        '  endif()\n'
        '  set(azure_sdk_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sdk-source")\n'
        f'  add_subdirectory("{actual}" "${{CMAKE_BINARY_DIR}}/azure-common")\n'
        'endmacro()\n'
        f'set(WIN32 {"TRUE" if windows else "FALSE"})\n'
        + wrapper +
        'build_azure_sdk()\n'
        'if(TARGET azure-storage-common)\n'
        '  get_target_property(links azure-storage-common LINK_LIBRARIES)\n'
        '  file(WRITE "${CMAKE_BINARY_DIR}/azure-links.txt" "${links}")\n'
        'endif()\n'
        'add_library(consumer SHARED consumer.c)\n'
        'target_link_libraries(consumer PRIVATE azure-storage-common)\n'
        'target_link_options(consumer PRIVATE "-Wl,-z,defs")\n')
    build = tmp_path / "build"
    configure = subprocess.run(
        ["/usr/bin/cmake", "-S", str(harness), "-B", str(build), "-G", "Ninja",
         "-DCMAKE_MAKE_PROGRAM=/usr/bin/ninja", "-DCMAKE_C_COMPILER=/usr/bin/cc"],
        text=True, capture_output=True, timeout=30)
    assert common_cmake.read_text() == common_source
    return configure, build


def test_azure_private_xml_interface_resolves_actual_final_shared_consumer(
        recipe, source_inputs, tmp_path):
    original, transformed = azure_xml_transform_fixture(recipe, source_inputs)
    before, before_build = azure_xml_cmake_graph(tmp_path / "before", original)
    assert before.returncode == 0, before.stdout + before.stderr
    broken = subprocess.run(
        ["/usr/bin/cmake", "--build", str(before_build), "--target", "consumer", "--parallel", "1"],
        text=True, capture_output=True, timeout=30)
    assert broken.returncode != 0
    assert "biominer_iconv_open" in broken.stdout + broken.stderr

    after, after_build = azure_xml_cmake_graph(tmp_path / "after", transformed)
    assert after.returncode == 0, after.stdout + after.stderr
    fixed = subprocess.run(
        ["/usr/bin/cmake", "--build", str(after_build), "--target", "consumer", "--parallel", "1"],
        text=True, capture_output=True, timeout=30)
    assert fixed.returncode == 0, fixed.stdout + fixed.stderr
    links = (after_build / "azure-links.txt").read_text().split(";")
    assert str(after_build / "libxml_stub.a") in links
    assert links.count("LibXml2::LibXml2") == 1
    assert (after_build / "libconsumer.so").read_bytes().startswith(b"\x7fELF")


@pytest.mark.parametrize("variant", ["foreign_source", "shared", "imported", "alias", "missing"])
def test_azure_private_xml_interface_rejects_unowned_or_wrong_kind_target(
        recipe, source_inputs, tmp_path, variant):
    _, transformed = azure_xml_transform_fixture(recipe, source_inputs)
    configured, _ = azure_xml_cmake_graph(tmp_path, transformed, target_variant=variant)
    assert configured.returncode != 0, "unproven Azure target was accepted"
    assert "private Azure XML dependency" in configured.stdout + configured.stderr


def test_azure_private_xml_interface_leaves_windows_dependency_unchanged(
        recipe, source_inputs, tmp_path):
    _, transformed = azure_xml_transform_fixture(recipe, source_inputs)
    configured, build = azure_xml_cmake_graph(tmp_path, transformed, windows=True)
    assert configured.returncode == 0, configured.stdout + configured.stderr
    assert "LibXml2::LibXml2" not in (build / "azure-links.txt").read_text().split(";")


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_azure_private_xml_interface_requires_exact_unique_source_anchor(
        recipe, source_inputs, mutation):
    block = "" if mutation == "missing" else AZURE_FETCH_BLOCK + "\n" + AZURE_FETCH_BLOCK
    with pytest.raises(ValueError, match="anchor|transformation"):
        azure_xml_transform_fixture(recipe, source_inputs, fetch_block=block)


def test_sdk_cohort_pins_are_the_complete_reviewed_finite_table(recipe):
    assert getattr(recipe, "SDK_COHORT", {}) == EXPECTED_SDK_COHORT
    for name, pins in EXPECTED_SDK_COHORT.items():
        assert recipe.DEPENDENCIES[name] == pins


def test_sdk_source_transformation_changes_every_exact_version_and_checksum(recipe, source_inputs):
    packed, original = source_transform_fixture(recipe, source_inputs)
    with tarfile.open(fileobj=packed) as archive:
        observed = recipe.source_transformations(archive, "arrow")
    lines = observed["cpp/thirdparty/versions.txt"].decode().splitlines()
    for pins in EXPECTED_SDK_COHORT.values():
        assert lines.count(f"ARROW_{pins['variable']}_BUILD_VERSION={pins['build_version']}") == 1
        assert lines.count(f"ARROW_{pins['variable']}_BUILD_SHA256_CHECKSUM={pins['sha256']}") == 1
        assert f"ARROW_{pins['variable']}_BUILD_VERSION={pins['old_version']}" not in lines
        assert f"ARROW_{pins['variable']}_BUILD_SHA256_CHECKSUM={pins['old_sha256']}" not in lines
    assert "ARROW_UNRELATED_BUILD_VERSION=keep-this-source-identity" in lines
    assert "ARROW_BOOST_BUILD_VERSION=1.92.0" in lines
    assert set(observed) == set(original)


@pytest.mark.parametrize("field", ["version", "sha256"])
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "prefixed", "suffixed"])
def test_sdk_source_transformation_requires_whole_unique_assignment_lines(
        recipe, source_inputs, field, mutation):
    packed, members = source_transform_fixture(recipe, source_inputs)
    pins = EXPECTED_SDK_COHORT["grpc"]
    anchor = (f"ARROW_GRPC_BUILD_VERSION={pins['old_version']}" if field == "version"
              else f"ARROW_GRPC_BUILD_SHA256_CHECKSUM={pins['old_sha256']}")
    replacement = {"missing": "", "duplicate": anchor + "\n" + anchor,
                   "prefixed": "UNREVIEWED_" + anchor, "suffixed": anchor + "_unreviewed"}[mutation]
    poisoned = members["cpp/thirdparty/versions.txt"].replace(anchor.encode(), replacement.encode())
    packed, _ = source_transform_fixture(recipe, source_inputs, versions=poisoned)
    with tarfile.open(fileobj=packed) as archive:
        with pytest.raises(ValueError, match="anchor|assignment|transformation"):
            recipe.source_transformations(archive, "arrow")


@pytest.mark.parametrize("field", ["version", "sha256"])
@pytest.mark.parametrize("position", ["before", "after"])
def test_sdk_source_transformation_rejects_changed_value_duplicate_assignments(
        recipe, source_inputs, field, position):
    _, members = source_transform_fixture(recipe, source_inputs)
    pins = EXPECTED_SDK_COHORT["grpc"]
    key = "ARROW_GRPC_BUILD_VERSION" if field == "version" else "ARROW_GRPC_BUILD_SHA256_CHECKSUM"
    old_value = pins["old_version"] if field == "version" else pins["old_sha256"]
    anchor = f"{key}={old_value}"
    duplicate = f"{key}=unreviewed-version" if field == "version" else f"{key}={'0' * 64}"
    replacement = (duplicate + "\n" + anchor if position == "before"
                   else anchor + "\n" + duplicate)
    poisoned = members["cpp/thirdparty/versions.txt"].replace(anchor.encode(), replacement.encode())
    packed, _ = source_transform_fixture(recipe, source_inputs, versions=poisoned)
    with tarfile.open(fileobj=packed) as archive:
        with pytest.raises(ValueError, match="anchor|assignment|transformation"):
            recipe.source_transformations(archive, "arrow")


def test_corrective25_keeps_uri_absence_and_all_existing_capabilities(recipe, source_inputs):
    source, evidence = prepare(recipe, source_inputs)
    toolchain = (source / "cpp/cmake_modules/ThirdpartyToolchain.cmake").read_text()
    # Exact Arrow25 has no uriparser source/target. Its target-only unity fix is26-only.
    assert "uriparser" not in toolchain
    assert tuple(recipe.FEATURES) == (
        "ACERO", "AZURE", "COMPUTE", "CSV", "DATASET", "FILESYSTEM", "FLIGHT",
        "GCS", "HDFS", "JSON", "ORC", "PARQUET", "S3", "SUBSTRAIT")
    assert tuple(recipe.CODECS) == ("BROTLI", "BZ2", "LZ4", "SNAPPY", "ZLIB", "ZSTD")
    assert [row["commit"] for row in evidence["security_patches"]["patches"]] == [
        "a769c291e01093b73d03a075179cf7a09bf92ad8",
        "0efbdec7a10da1a4043fc1210c61e73c88bc9b4f"]
    assert "set_target_properties(c-ares PROPERTIES UNITY_BUILD OFF)" in toolchain
    assert "  set(CMAKE_UNITY_BUILD OFF)\n  fetchcontent_makeavailable(grpc)" in toolchain


def test_all_direct_sdk_packages_are_forced_to_the_reviewed_bundled_sources(recipe, tmp_path):
    context = {"cc": "/cc", "cxx": "/cxx", "python": "/python", "work": str(tmp_path),
               "openssl_prefix": "/private/openssl", "network_prefix": "/private/network"}
    expected = recipe.expected_cache(context, stage="cpp")
    for package in ("AWSSDK", "gRPC", "Protobuf", "absl", "c-ares", "Thrift", "jemalloc", "RapidJSON"):
        assert expected[package + "_SOURCE"] == "BUNDLED"
    assert expected["CUSTOM_MEMORY_MANAGEMENT"] == "OFF"


@pytest.mark.parametrize("name", [*EXPECTED_SDK_COHORT, "unreviewed"])
def test_inherited_dependency_urls_cannot_influence_the_build(recipe, monkeypatch, name):
    variable = EXPECTED_SDK_COHORT[name]["env"] if name != "unreviewed" else "ARROW_UNREVIEWED_URL"
    monkeypatch.setenv(variable, "/poisoned/unreviewed-input.tar.gz")
    env = recipe.build_environment(4, python="/python", cc="/cc", cxx="/cxx")
    assert variable not in env


@pytest.fixture
def sdk_configure_inputs(recipe, source_inputs, network_inputs, tmp_path, monkeypatch):
    """Exercise real configure orchestration, source patching and immutable inputs."""
    archive, work, patches, _, _ = source_inputs
    receipt, helper, openssl, descriptor = network_inputs
    original_relative_to = Path.is_relative_to
    monkeypatch.setattr(Path, "is_relative_to", lambda self, *other:
        False if other in (("/tmp",), ("/var/tmp",)) else original_relative_to(self, *other))
    tools = {name: {"path": "/approved/" + name, "sha256": "0" * 64}
             for name in ("cc", "cxx", "cmake", "ninja", "ld")}
    builder = {"base_python": {"path": "/approved/base-python"}, "toolchain": tools}
    monkeypatch.setattr(recipe, "validate_builder", lambda *args, **kwargs: builder)
    monkeypatch.setattr(recipe, "probe_builder", lambda *args: {"identity": "frozen fixture"})
    monkeypatch.setattr(recipe, "validate_python_runtime", lambda *args: {"files": []})
    monkeypatch.setattr(recipe, "validate_openssl", lambda *args: {"version": "3.6.5"})
    monkeypatch.setattr(recipe, "expected_generated_version", lambda *args: b"fixture generated version\n")
    monkeypatch.setattr(recipe.subprocess, "check_output", lambda *args, **kwargs: "fixture version\n")
    monkeypatch.setattr(recipe, "capture_dependencies", lambda *args, **kwargs: {
        "targets": [], "boost_components": [], "boost_selection": {}, "source_trees": [],
        "forced_bundled_source_verification": {}, "static_archives": [], "downloaded_archives": []})
    # Use independent fixture bytes while preserving the production table's other fields.
    monkeypatch.setattr(recipe, "DEPENDENCIES", {name: dict(pins) for name, pins in recipe.DEPENDENCIES.items()})
    expected = {**recipe.DEPENDENCIES, **EXPECTED_SDK_COHORT}
    archives = {}
    for name, pins in expected.items():
        path = tmp_path / ("sdk-input-" + name + ".tar.gz")
        path.write_bytes(("approved synthetic archive " + name).encode())
        archives[name] = path
        if name in recipe.DEPENDENCIES:
            monkeypatch.setitem(recipe.DEPENDENCIES[name], "sha256", digest(path.read_bytes()))
    argv = ["--archive", str(archive), "--work", str(work), "--openssl-prefix", str(openssl),
            "--builder-receipt", str(tmp_path / "builder.json"), "--builder-receipt-sha256", "0" * 64,
            "--network-receipt", str(receipt), "--network-receipt-sha256", digest(receipt.read_bytes()),
            "--network-helper", str(helper), "--python", "/approved/python", "--cc", tools["cc"]["path"],
            "--cxx", tools["cxx"]["path"], "--security-patch", str(patches[0]),
            "--offset-patch", str(patches[1])]
    for name, path in archives.items():
        argv += ["--" + name.replace("_", "-") + "-archive", str(path)]
    stages = []
    mutation = {"name": None}

    def configure(stage_work, name, command, env):
        stages.append({"name": name, "env": dict(env)})
        assert name == "configure"
        context = {"work": str(work), "openssl_prefix": str(openssl),
                   "network_prefix": descriptor["prefix"], "python": "/approved/python",
                   "cc": tools["cc"]["path"], "cxx": tools["cxx"]["path"]}
        cache = work / "cpp-build/CMakeCache.txt"
        cache.write_text("\n".join(f"{key}:STRING={value}" for key, value in
                                  recipe.expected_cache(context, stage="cpp").items()) + "\n")
        if mutation["name"]:
            name = mutation["name"]
            (work / "inputs" / expected[name]["filename"]).write_bytes(b"changed copied SDK archive")
        return {"command": command, "exit_code": 0}

    monkeypatch.setattr(recipe, "run_stage", configure)
    return {"argv": argv, "work": work, "archives": archives, "stages": stages,
            "expected": expected, "mutation": mutation}


def invoke_sdk_cli(recipe, argv):
    try:
        return recipe.main(argv)
    except SystemExit as error:
        # argparse rejection is an observable pre-implementation failure too.
        return error.code


def test_every_sdk_archive_is_captured_and_source_owned_urls_override_poison(
        recipe, sdk_configure_inputs, monkeypatch):
    inputs = sdk_configure_inputs
    for name, pins in EXPECTED_SDK_COHORT.items():
        monkeypatch.setenv(pins["env"], "/poisoned/" + name + ".tar.gz")
    assert invoke_sdk_cli(recipe, inputs["argv"] + ["--configure-only"]) == 0
    work = inputs["work"]
    manifest = json.loads((work / "build-inputs.json").read_bytes())
    assert manifest["source_policy"] == "arrow25-two-compute-patches-private-network-v2"
    assert set(manifest["dependencies"]) == set(inputs["expected"])
    assert [row["name"] for row in inputs["stages"]] == ["configure"]
    env = inputs["stages"][0]["env"]
    for name, pins in EXPECTED_SDK_COHORT.items():
        saved = work / "inputs" / pins["filename"]
        assert saved.read_bytes() == inputs["archives"][name].read_bytes()
        assert manifest["dependencies"][name] == {
            **pins, "sha256": digest(saved.read_bytes()),
            "archive": record(saved), "role": "bundled_provider"}
        assert env[pins["env"]] == str(saved)
    assert manifest["compiler_budget"] == {
        "maximum_jobs": 4, "outer_ninja_jobs": 4, "nested_cmake_jobs": 1, "nested_make_jobs": 1}
    assert manifest["security_patches"]["patches"][0]["commit"] == recipe.SECURITY_PATCH_COMMIT
    assert manifest["security_patches"]["patches"][1]["commit"] == recipe.OFFSET_PATCH_COMMIT


@pytest.mark.parametrize("name", list(EXPECTED_SDK_COHORT))
def test_any_sdk_archive_checksum_failure_creates_no_generation(
        recipe, sdk_configure_inputs, name, capsys):
    inputs = sdk_configure_inputs
    inputs["archives"][name].write_bytes(b"wrong source input")
    assert invoke_sdk_cli(recipe, inputs["argv"] + ["--check"]) != 0
    assert "checksum" in capsys.readouterr().err
    assert not inputs["work"].exists()
    assert inputs["stages"] == []


@pytest.mark.parametrize("name", ["aws_sdk_cpp", "grpc", "cares"])
def test_copied_sdk_input_mutation_is_rejected_after_configure(
        recipe, sdk_configure_inputs, name, capsys):
    inputs = sdk_configure_inputs
    inputs["mutation"]["name"] = name
    assert invoke_sdk_cli(recipe, inputs["argv"] + ["--configure-only"]) != 0
    assert "immutable build input checksum" in capsys.readouterr().err
    assert not (inputs["work"] / "configured-inputs.json").exists()
    assert json.loads((inputs["work"] / "build-receipt.json").read_bytes())["status"] == "failed"
