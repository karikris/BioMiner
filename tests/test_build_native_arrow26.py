"""Arrow 26 build guards use isolated fixtures and bounded local CMake stubs."""
from __future__ import annotations

import gzip
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


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_native_arrow26.py"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def record(path):
    return {"path": str(path), "sha256": digest(path.read_bytes())}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return record(path)


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


def test_arrow26_recipe_requires_explicit_build_inputs():
    result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "--builder-receipt" in result.stderr


@pytest.fixture
def recipe():
    spec = importlib.util.spec_from_file_location("arrow26_recipe_tests", SCRIPT)
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
    toolchain = "\n".join(before for before, _ in recipe.TOOLCHAIN_PATCHES)
    toolchain += '\n    set(ZLIB_CMAKE_ARGS ${EP_COMMON_CMAKE_ARGS} "-DCMAKE_INSTALL_PREFIX=${ZLIB_PREFIX}")\n'
    toolchain += '                        URL_HASH "SHA256=${ARROW_ZLIB_BUILD_SHA256_CHECKSUM}"\n'
    toolchain += "\n" + recipe.BOOST_SELECTION_ANCHOR + "\n"
    versions = "\n".join(before for before, _ in recipe.VERSION_PATCHES) + "\nARROW_ZLIB_BUILD_VERSION=1.3.1\n"
    files = {
        "cpp/cmake_modules/ThirdpartyToolchain.cmake": toolchain.encode(),
        "cpp/thirdparty/versions.txt": versions.encode(),
        "python/pyproject.toml": b'build-backend = "scikit_build_core.build"\n',
        "cpp/src/arrow/compute/kernels/copy_data_internal.h": b"integrated sliced Boolean fix\n",
        "cpp/src/arrow/compute/kernels/vector_replace.cc": b"integrated allocation fix\n",
        "cpp/src/arrow/compute/kernels/vector_replace_test.cc": b"integrated regressions\n",
        "cpp/src/arrow/compute/kernels/scalar_if_else.cc": b"second header consumer\n",
        "NOTICE.txt": b"retain upstream notice\n",
    }
    monkeypatch.setattr(recipe, "INTEGRATED_SOURCES", {
        name: digest(data) for name, data in files.items() if name.startswith("cpp/src/")})
    source_archive = archive_at(tmp_path / "arrow.tar.gz", f"arrow-{recipe.ARROW_COMMIT}",
                                files, {"python/NOTICE.txt": "../NOTICE.txt"})
    monkeypatch.setattr(recipe, "ARROW_ARCHIVE_SHA256", digest(source_archive.read_bytes()))
    return source_archive, tmp_path / "generation", files


def test_source_preserves_integrated_fixes_and_transforms_exact_inputs(recipe, source_inputs):
    archive, work, files = source_inputs
    source, evidence = recipe.prepare_source(archive, work, resume=False)

    for name, expected in files.items():
        if name not in {"cpp/cmake_modules/ThirdpartyToolchain.cmake", "cpp/thirdparty/versions.txt"}:
            assert (source / name).read_bytes() == expected
    assert (source / "python/NOTICE.txt").is_symlink()
    text = (source / "cpp/cmake_modules/ThirdpartyToolchain.cmake").read_text()
    assert 'set(MAKE_BUILD_ARGS "-j1")' in text
    assert "ZLIB_BUILD_SHARED=OFF" not in text
    assert "PATCH_COMMAND" not in text
    versions = (source / "cpp/thirdparty/versions.txt").read_text()
    assert recipe.DEPENDENCIES["simdjson"]["sha256"] in versions
    assert "ARROW_ZLIB_BUILD_VERSION=1.3.1" in versions
    assert evidence["integrated_sources"] == [
        {"relative": name, "sha256": value} for name, value in sorted(recipe.INTEGRATED_SOURCES.items())]
    assert not list(work.glob("arrow25*.patch"))
    assert recipe.prepare_source(archive, work, resume=True) == (source, evidence)


@pytest.mark.parametrize("change", ["file", "extra", "directory", "symlink", "toolchain", "versions"])
def test_source_resume_rejects_any_unreviewed_source_change(recipe, source_inputs, change):
    archive, work, _ = source_inputs
    source, _ = recipe.prepare_source(archive, work, resume=False)
    if change == "file":
        (source / "NOTICE.txt").write_text("changed")
    elif change == "extra":
        (source / "extra.cc").write_text("unreviewed")
    elif change == "directory":
        (source / "extra").mkdir()
    elif change == "symlink":
        (source / "python/NOTICE.txt").unlink()
        (source / "python/NOTICE.txt").symlink_to("../cpp")
    elif change == "toolchain":
        (source / "cpp/cmake_modules/ThirdpartyToolchain.cmake").write_text("unreviewed")
    else:
        (source / "cpp/thirdparty/versions.txt").write_text("unreviewed")

    with pytest.raises(ValueError, match="source|layout"):
        recipe.prepare_source(archive, work, resume=True)


def test_source_rejects_bad_archive_before_creating_outputs(recipe, source_inputs):
    archive, work, _ = source_inputs
    archive.write_bytes(b"unapproved")
    with pytest.raises(ValueError, match="archive checksum"):
        recipe.prepare_source(archive, work, resume=False)
    assert not work.exists()


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_source_rejects_inexact_transformation_anchor(recipe, source_inputs, monkeypatch, change):
    archive, work, files = source_inputs
    key = "cpp/cmake_modules/ThirdpartyToolchain.cmake"
    anchor = recipe.TOOLCHAIN_PATCHES[0][0].encode()
    files[key] = files[key].replace(anchor, b"removed" if change == "missing" else anchor + b"\n" + anchor)
    archive_at(archive, f"arrow-{recipe.ARROW_COMMIT}", files, {"python/NOTICE.txt": "../NOTICE.txt"})
    monkeypatch.setattr(recipe, "ARROW_ARCHIVE_SHA256", digest(archive.read_bytes()))
    with pytest.raises(ValueError, match="anchor"):
        recipe.prepare_source(archive, work, resume=False)
    assert not work.exists()


@pytest.mark.parametrize("kind", ["receipt", "wheel", "symlink", "nonempty"])
def test_source_preserves_existing_or_completed_outputs(recipe, source_inputs, tmp_path, kind):
    archive, work, _ = source_inputs
    if kind == "nonempty":
        work.mkdir()
        marker = work / "unowned.txt"
        marker.write_text("keep")
    else:
        recipe.prepare_source(archive, work, resume=False)
        if kind == "receipt":
            marker = work / "build-receipt.json"
            marker.write_text('{"status":"compiled_unverified"}')
        elif kind == "wheel":
            marker = work / "wheelhouse/pyarrow.whl"
            marker.parent.mkdir()
            marker.write_bytes(b"keep wheel")
        else:
            marker = work / "python-build"
            marker.symlink_to(tmp_path / "outside")
    before = None if marker.is_symlink() else marker.read_bytes()

    with pytest.raises(ValueError, match="already exists|completed|symlink"):
        recipe.prepare_source(archive, work, resume=kind != "nonempty")
    assert marker.is_symlink() if before is None else marker.read_bytes() == before


def test_manifest_derives_version_and_binds_command_template(recipe, tmp_path):
    inputs = {"commands": {"python_build": ["build", "{wheel_version}"]}, "jobs": 4}
    manifest, version = recipe.preserve_build_inputs(tmp_path, inputs, resume=False)
    assert version == "26.0.0+biominer." + manifest["sha256"][:12]
    assert recipe.preserve_build_inputs(tmp_path, inputs, resume=True) == (manifest, version)
    with pytest.raises(ValueError, match="build inputs changed"):
        recipe.preserve_build_inputs(tmp_path, inputs | {"jobs": 3}, resume=True)


def test_captured_recipe_resume_cannot_change(recipe, tmp_path):
    recipe.preserve_recipe(tmp_path, b"original", resume=False)
    with pytest.raises(ValueError, match="recipe changed"):
        recipe.preserve_recipe(tmp_path, b"changed", resume=True)
    assert (tmp_path / "build-recipe.py").read_bytes() == b"original"


@pytest.mark.parametrize("jobs", [0, 5, 8])
def test_job_budget_rejects_unsafe_outer_concurrency(recipe, jobs):
    with pytest.raises(ValueError, match="between 1 and 4"):
        recipe.build_environment(jobs, python="/approved/python", cc="/approved/cc", cxx="/approved/cxx")


def test_build_commands_bind_both_stages_and_persistent_python_tree(recipe, tmp_path, monkeypatch):
    monkeypatch.setenv("CC", "/wrong/compiler")
    monkeypatch.setenv("CMAKE_ARGS", "-DARROW_S3=OFF")
    env = recipe.build_environment(4, python="/approved/python", cc="/approved/cc", cxx="/approved/cxx")
    commands = recipe.build_commands(tmp_path / "source", tmp_path / "work", tmp_path / "openssl", network_prefix=tmp_path / "network",
                                     python="/approved/python", cc="/approved/cc", cxx="/approved/cxx", jobs=4)
    assert env["CC"] == "/approved/cc"
    assert env["CXX"] == "/approved/cxx"
    assert env["MAKEFLAGS"] == "-j1"
    assert env["CMAKE_BUILD_PARALLEL_LEVEL"] == "1"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "CMAKE_ARGS" not in env
    configure = commands["cpp_configure"]
    for feature in recipe.FEATURES:
        assert f"-DARROW_{feature}=ON" in configure
        assert env[f"PYARROW_WITH_{feature}"] == "ON"
    for flag in ("TESTING", "BUILD_TESTS", "BUILD_BENCHMARKS", "BUILD_INTEGRATION", "GANDIVA"):
        assert f"-DARROW_{flag}=OFF" in configure
    for name in ("simdjson", "uriparser", "Boost"):
        assert f"-D{name}_SOURCE=BUNDLED" in configure
    assert "-DZLIB_SOURCE=SYSTEM" in configure
    wheel_command = commands["python_build"]
    assert "--no-isolation" in wheel_command
    assert f"--config-setting=build-dir={tmp_path / 'work/python-build'}" in wheel_command
    assert "--config-setting=build.tool-args=-j4" in wheel_command
    assert "--config-setting=cmake.define.CMAKE_C_COMPILER=/approved/cc" in wheel_command
    assert "--config-setting=cmake.define.CMAKE_CXX_COMPILER=/approved/cxx" in wheel_command
    assert "--parallel" in commands["cpp_build"]


@pytest.fixture
def builder_inputs(tmp_path, recipe):
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    base = tmp_path / "approved-python"
    base.write_bytes(b"approved interpreter")
    base.chmod(0o555)
    python = env / "bin/python"
    python.symlink_to(base)
    tools = {}
    for name in ("cc", "cxx", "cmake", "ninja", "ld"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        path.chmod(0o555)
        tools[name] = record(path)
    distributions = []
    for name, version in recipe.BUILDER_PINS.items():
        package = env / f"{name}-{version}.dist-info"
        package.mkdir()
        (package / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
        (package / "RECORD").write_text("fixture record\n")
        wheel = tmp_path / f"{name}.whl"
        wheel.write_bytes(f"fixture wheel {name}".encode())
        distributions.append({"name": name, "version": version, "wheel": record(wheel),
                              "metadata": record(package / "METADATA"), "record": record(package / "RECORD")})
    for path in env.rglob("*"):
        if not path.is_symlink():
            path.chmod(0o555 if path.is_dir() else 0o444)
    env.chmod(0o555)
    inventory = {
        "schema_version": 1, "root": str(env),
        "files": [{"relative": str(p.relative_to(env)), "sha256": digest(p.read_bytes()),
                   "mode": p.stat().st_mode & 0o7777} for p in sorted(env.rglob("*")) if p.is_file() and not p.is_symlink()],
        "directories": [{"relative": ".", "mode": 0o555}] + [
            {"relative": str(p.relative_to(env)), "mode": 0o555} for p in sorted(env.rglob("*"))
            if p.is_dir() and not p.is_symlink()],
        "symlinks": [{"relative": "bin/python", "target": str(base)}],
    }
    inventory_record = write_json(tmp_path / "inventory.json", inventory)
    req = tmp_path / "requirements.txt"
    req.write_text("pinned fixture\n")
    verification = tmp_path / "verification.json"
    verification.write_text('{"status":"passed"}\n')
    value = {"schema_version": 1, "kind": "arrow26_python_builder", "status": "frozen_verified",
             "python": record(python), "base_python": record(base), "requirements": record(req),
             "distributions": distributions, "inventory": inventory_record,
             "toolchain": tools, "verification": record(verification)}
    receipt = tmp_path / "builder.json"
    receipt_record = write_json(receipt, value)
    yield receipt, receipt_record["sha256"], value, env
    env.chmod(0o755)
    for path in env.rglob("*"):
        if not path.is_symlink():
            path.chmod(0o755 if path.is_dir() else 0o644)


def test_builder_validates_frozen_exact_inventory_before_probe(recipe, builder_inputs):
    receipt, expected, value, _ = builder_inputs
    checked = recipe.validate_builder(receipt, expected, python=value["python"]["path"],
                                      cc=value["toolchain"]["cc"]["path"], cxx=value["toolchain"]["cxx"]["path"])
    assert checked["receipt"] == record(receipt)
    assert checked["distributions"] == value["distributions"]


@pytest.mark.parametrize("change", ["hash", "extra", "writable", "compiler", "pin", "symlink"])
def test_builder_rejects_unapproved_or_mutated_inputs(recipe, builder_inputs, change):
    receipt, expected, value, env = builder_inputs
    cc = value["toolchain"]["cc"]["path"]
    if change == "hash":
        expected = "0" * 64
    elif change == "extra":
        env.chmod(0o755)
        (env / "injected.py").write_text("bad")
        env.chmod(0o555)
    elif change == "writable":
        Path(value["distributions"][0]["metadata"]["path"]).chmod(0o644)
    elif change == "compiler":
        cc = "/wrong/compiler"
    elif change == "pin":
        value["distributions"][0]["version"] = "0.0.0"
        expected = write_json(receipt, value)["sha256"]
    else:
        (env / "bin").chmod(0o755)
        (env / "bin/python").unlink()
        (env / "bin/python").symlink_to(sys.executable)
        (env / "bin").chmod(0o555)
    with pytest.raises(ValueError, match="builder|compiler|inventory|distribution|checksum|symlink"):
        recipe.validate_builder(receipt, expected, python=value["python"]["path"], cc=cc,
                                cxx=value["toolchain"]["cxx"]["path"])


def cache_fixture(recipe, tmp_path):
    expected = {"python": "/approved/python", "cc": "/approved/cc", "cxx": "/approved/cxx",
                "openssl_prefix": str(tmp_path / "openssl"), "network_prefix": str(tmp_path / "network"),
                "work": str(tmp_path / "work")}
    entries = recipe.expected_cache(expected, stage="cpp")
    path = tmp_path / "CMakeCache.txt"
    path.write_text("\n".join(f"{k}:STRING={v}" for k, v in entries.items()) + "\n")
    return path, entries, expected


@pytest.mark.parametrize("key,value", [("simdjson_SOURCE", "SYSTEM"), ("ZLIB_SOURCE", "AUTO"),
                                       ("ARROW_TESTING", "ON"), ("ARROW_S3", "OFF"),
                                       ("CMAKE_CXX_COMPILER", "/unapproved/clang++"),
                                       ("OPENSSL_CRYPTO_LIBRARY", "/usr/lib/libcrypto.so")])
def test_configured_cache_must_match_required_contract(recipe, tmp_path, key, value):
    path, entries, expected = cache_fixture(recipe, tmp_path)
    recipe.verify_cache(path, expected, stage="cpp")
    entries[key] = value
    path.write_text("\n".join(f"{k}:STRING={v}" for k, v in entries.items()) + "\n")
    with pytest.raises(ValueError, match=key):
        recipe.verify_cache(path, expected, stage="cpp")


@pytest.mark.parametrize("change", ["version", "manifest", "missing_elf", "missing_s3"])
def test_wheel_rejection_precedes_native_admission(recipe, tmp_path, change):
    manifest, version = recipe.preserve_build_inputs(tmp_path, {"jobs": 4}, resume=False)
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    with zipfile.ZipFile(wheelhouse / "pyarrow.whl", "w") as wheel:
        actual = "26.0.0" if change == "version" else version
        wheel.writestr("pyarrow.dist-info/METADATA", f"Name: pyarrow\nVersion: {actual}\n")
        if change != "missing_elf":
            for name in recipe.REQUIRED_ARROW_LIBRARIES:
                if change != "missing_s3" or "s3" not in name:
                    wheel.writestr(f"pyarrow/{name}", b"\x7fELFfixture")
    if change == "manifest":
        (tmp_path / "build-inputs.json").write_text("changed")
    with pytest.raises((ValueError, RuntimeError), match="version|manifest|ELF|libarrow_s3"):
        recipe.inspect_wheel(tmp_path, {"build_inputs": manifest, "wheel_version": version})


def test_resume_state_rejects_modified_intermediate_outputs(recipe, tmp_path):
    build = tmp_path / "cpp-build"
    build.mkdir()
    intermediate = build / "object.o"
    intermediate.write_bytes(b"compiled partial output")
    recipe.write_resume_state(tmp_path)
    recipe.verify_resume_state(tmp_path)
    intermediate.write_bytes(b"injected partial output")
    with pytest.raises(ValueError, match="resume state"):
        recipe.verify_resume_state(tmp_path)


def test_resume_preserves_failed_evidence_when_python_changed_source(recipe, source_inputs):
    archive, work, _ = source_inputs
    source, _ = recipe.prepare_source(archive, work, resume=False)
    generated = source / "python/pyarrow/_generated_version.py"
    generated.parent.mkdir()
    generated.write_text("unreviewed generated version")
    with pytest.raises(ValueError, match="source layout"):
        recipe.prepare_source(archive, work, resume=True)
    assert generated.read_text() == "unreviewed generated version"


@pytest.mark.parametrize("missing", ["libcrypto.a", "libssl.a", "libz.a", "libcurl.a", "libxml2.a", "libiconv.a", "libnghttp2.a"])
def test_link_maps_require_actual_static_crypto_and_private_network(recipe, network_inputs, tmp_path, missing):
    network = verified_network(recipe, network_inputs)
    prefix = network_inputs[2]
    static = [row for row in network["files"] if Path(row["path"]).suffix == ".a"]
    linked = [row["path"] for row in static if not row["path"].endswith(missing)]
    work = tmp_path / "arrow-work"
    (work / "cpp-build/release").mkdir(parents=True)
    (work / "cpp-build/release/libarrow.so.2600.0.0.map").write_text(
        "Linker script and memory map\n" + "\n".join(path + "(member.o)" for path in linked))
    with pytest.raises(ValueError, match=missing):
        recipe.verify_link_maps(work, prefix, {"static_archives": static}, network=network)


def selection_fixture(recipe, tmp_path):
    source = tmp_path / "cpp-build/_deps/example-src"
    source.mkdir(parents=True)
    (source / "input.cc").write_bytes(b"approved source")
    selection = {"targets": [{"target": "example", "artifact": str(tmp_path / "libexample.a")}],
                 "boost_components": ["thread"], "boost_selection": {"sha256": "fixture"},
                 "forced_bundled_source_verification": {"simdjson": [{"relative": "input.cc", "sha256": "fixture"}]},
                 "static_archives": [], "downloaded_archives": [],
                 "source_trees": [{"path": str(source), "files": recipe.tree_inventory(source)}]}
    return source, selection


def test_configured_selection_allows_hash_bound_partial_build_outputs(recipe, tmp_path):
    import copy

    source, configured = selection_fixture(recipe, tmp_path)
    archive = tmp_path / "libexample.a"
    archive.write_bytes(b"partial build archive")
    (source / "generated.h").write_bytes(b"generated during build")
    state = recipe.write_resume_state(tmp_path)
    recipe.verify_resume_state(tmp_path, state)
    current = copy.deepcopy(configured)
    current["static_archives"] = [record(archive)]
    current["source_trees"][0]["files"] = recipe.tree_inventory(source)

    recipe.verify_configured_selection(configured, current)


@pytest.mark.parametrize("change", ["source", "target", "bundled_source"])
def test_configured_selection_still_rejects_changed_immutable_inputs(recipe, tmp_path, change):
    import copy

    source, configured = selection_fixture(recipe, tmp_path)
    current = copy.deepcopy(configured)
    if change == "source":
        (source / "input.cc").write_bytes(b"changed source")
        current["source_trees"][0]["files"] = recipe.tree_inventory(source)
    elif change == "target":
        current["targets"][0]["target"] = "unapproved"
    else:
        current["forced_bundled_source_verification"]["simdjson"][0]["sha256"] = "changed"
    with pytest.raises(ValueError, match="configured|source"):
        recipe.verify_configured_selection(configured, current)


def test_openssl_rejects_changed_header_even_with_approved_version_text(recipe, tmp_path):
    prefix = tmp_path / "openssl"
    header = prefix / "include/openssl/opensslv.h"
    header.parent.mkdir(parents=True)
    header.write_text('#define OPENSSL_VERSION_TEXT "OpenSSL 3.6.5 29 Sep 2026"\n/* changed header */\n')

    with pytest.raises(ValueError, match="OpenSSL header checksum"):
        recipe.validate_openssl(prefix, tmp_path / "archive.tar.gz")


@pytest.mark.parametrize("zlib_cache", ["archive", "alias", "foreign_alias"])
def test_cli_bad_configure_approval_preserves_incomplete_receipt_and_checkpoint(
        recipe, source_inputs, builder_inputs, network_inputs, tmp_path, monkeypatch, capsys, zlib_cache):
    """Exercise actual orchestration; only external builder/configure work is fake."""
    archive, work, _ = source_inputs
    builder_receipt, builder_hash, builder, _ = builder_inputs
    receipt_path, helper, prefix, _ = network_inputs
    original_relative_to = Path.is_relative_to
    # Fixture work is deliberately temporary; production still requires disk.
    monkeypatch.setattr(Path, "is_relative_to", lambda self, *other:
                        False if other in (("/tmp",), ("/var/tmp",)) else original_relative_to(self, *other))
    monkeypatch.setattr(recipe, "probe_builder", lambda value: {"identity": "frozen test builder"})
    monkeypatch.setattr(recipe, "validate_python_runtime", lambda value: {"source_build": "synthetic-python", "files": []})
    monkeypatch.setattr(recipe, "validate_openssl", lambda *args: {"version": "3.6.5"})
    monkeypatch.setattr(recipe, "expected_generated_version", lambda *args: b"exact generated test version\n")
    monkeypatch.setattr(recipe.subprocess, "check_output", lambda *args, **kwargs: "fixture tool version\n")
    selected = {"targets": [], "boost_components": [], "boost_selection": {},
                "forced_bundled_source_verification": {}, "static_archives": [],
                "downloaded_archives": [], "source_trees": []}
    monkeypatch.setattr(recipe, "capture_dependencies", lambda *args, **kwargs: selected)
    stages = []

    def configure_fixture(stage_work, name, command, env):
        stages.append(name)
        assert name == "configure"
        context = {"work": str(work), "openssl_prefix": str(prefix), "network_prefix": str(network_inputs[3]["prefix"]),
                   "python": builder["python"]["path"],
                   "cc": builder["toolchain"]["cc"]["path"], "cxx": builder["toolchain"]["cxx"]["path"]}
        cache = work / "cpp-build/CMakeCache.txt"
        cache.parent.mkdir(exist_ok=True)
        entries = recipe.expected_cache(context, stage="cpp")
        if zlib_cache != "archive":
            entries["ZLIB_LIBRARY"] = "ZLIB::ZLIB"
            artifact = Path(context["network_prefix"]) / "lib/libz.a"
            if zlib_cache == "foreign_alias":
                artifact = tmp_path / "foreign-libz.a"
                artifact.write_bytes(b"unapproved zlib")
            write_zlib_alias_target(recipe, work, artifact)
        cache.write_text("\n".join(f"{key}:STRING={value}" for key, value in
                                   entries.items()) + "\n")
        return {"command": command, "exit_code": 0}

    monkeypatch.setattr(recipe, "run_stage", configure_fixture)
    argv = ["--archive", str(archive), "--work", str(work), "--openssl-prefix", str(prefix),
            "--builder-receipt", str(builder_receipt), "--builder-receipt-sha256", builder_hash,
            "--network-receipt", str(receipt_path), "--network-receipt-sha256", digest(receipt_path.read_bytes()),
            "--network-helper", str(helper),
            "--python", builder["python"]["path"], "--cc", builder["toolchain"]["cc"]["path"],
            "--cxx", builder["toolchain"]["cxx"]["path"]]
    for name in recipe.DEPENDENCIES:
        dependency_archive = tmp_path / f"{name}.tar.gz"
        dependency_archive.write_bytes(f"approved {name} fixture".encode())
        monkeypatch.setitem(recipe.DEPENDENCIES[name], "sha256", digest(dependency_archive.read_bytes()))
        argv.extend(["--" + name.replace("_", "-") + "-archive", str(dependency_archive)])
    configured_result = recipe.main(argv + ["--configure-only"])
    if zlib_cache == "foreign_alias":
        assert configured_result != 0
        assert "ZLIB_LIBRARY" in capsys.readouterr().err
        assert not (work / "configured-inputs.json").exists()
        assert json.loads((work / "build-receipt.json").read_bytes())["status"] == "failed"
        return
    assert configured_result == 0
    manifest = json.loads((work / "build-inputs.json").read_bytes())
    assert manifest["source_policy"] == "arrow26-integrated-private-network-v3"
    assert manifest["network"]["helper"] == record(work / "private-network-helper.py")
    assert manifest["network"]["receipt"] == record(receipt_path)
    assert manifest["provider_cmake"] == record(work / "private-network.cmake")
    assert manifest["dependencies"]["zlib"]["role"] == "retained_source_pin_only"
    assert "zlib_patch" not in manifest
    assert not (work / "zlib-patch-application.json").exists()
    receipt = work / "build-receipt.json"
    before = receipt.read_bytes()
    before_tree = recipe.tree_inventory(work)
    state_hash = json.loads(before)["resume_state"]["sha256"]
    try:
        result = recipe.main(argv + ["--resume", "--resume-state-sha256", state_hash,
                                    "--configured-inputs-sha256", "0" * 64])
    except SystemExit as error:
        result = error.code

    assert result != 0
    assert "reviewed configured inputs checksum mismatch" in capsys.readouterr().err
    assert receipt.read_bytes() == before
    assert recipe.tree_inventory(work) == before_tree
    assert stages == ["configure"]


def configured_target_fixture(recipe, tmp_path):
    work = tmp_path / "work"
    build = work / "cpp-build"
    reply = build / ".cmake/api/v1/reply"
    reply.mkdir(parents=True)
    write_json(work / "build-inputs.json", {"openssl": {"static_archives": []}})
    (build / "biominer-boost-selection.txt").write_text(
        "algorithm;crc;numeric/conversion;scope_exit;throw_exception;tokenizer;locale;thread;uuid\n"
    )
    for name in ("simdjson", "uriparser", "boost"):
        source = build / "_deps" / (name + "-src")
        source.mkdir(parents=True)
        (source / "input.h").write_bytes(b"approved source\n")
        archive = work / "inputs" / recipe.DEPENDENCIES[name]["filename"]
        archive.parent.mkdir(exist_ok=True)
        archive_at(archive, name, {"input.h": b"approved source\n"})
    source = work / f"arrow-{recipe.ARROW_COMMIT}/cpp"
    versions = source / "thirdparty/versions.txt"
    versions.parent.mkdir(parents=True)
    versions.write_text("")
    archives = []
    targets = []
    for name in ("one", "two"):
        archive = work / (name + ".a")
        archive.write_bytes(name.encode())
        archives.append(archive)
        target = {"name": "Repeated", "id": name, "type": "STATIC_LIBRARY",
                  "imported": True, "local": True,
                  "paths": {"source": name, "build": name},
                  "artifacts": [{"path": str(archive)}]}
        write_json(reply / (name + ".json"), target)
        targets.append({"name": "Repeated", "id": name, "jsonFile": name + ".json"})
    write_json(reply / "model.json", {"kind": "codemodel", "version": {"major": 2, "minor": 9},
        "paths": {"source": str(source), "build": str(build)},
        "configurations": [{"name": "Release", "targets": [], "abstractTargets": targets}]})
    write_json(reply / "index-test.json", {"reply": {"codemodel-v2": {
        "kind": "codemodel", "version": {"major": 2, "minor": 9}, "jsonFile": "model.json"}}})
    return work, reply, archives


def test_dependency_inventory_preserves_scoped_imported_target_identity(recipe, tmp_path):
    work, reply, archives = configured_target_fixture(recipe, tmp_path)

    observed = recipe.capture_dependencies(work, compiled=False)

    assert [row["target"] for row in observed["targets"]] == ["Repeated", "Repeated"]
    assert [row["id"] for row in observed["targets"]] == ["one", "two"]
    assert [row["artifact"] for row in observed["targets"]] == list(map(str, archives))
    assert [row["record"] for row in observed["targets"]] == [record(reply / "one.json"), record(reply / "two.json")]
    assert observed["static_archives"] == [record(path) for path in archives]
    assert len(observed["file_api"]) == 4


@pytest.mark.parametrize("change", ["missing", "escape", "identity", "version"])
def test_dependency_inventory_rejects_invalid_file_api_evidence(recipe, tmp_path, change):
    work, reply, _ = configured_target_fixture(recipe, tmp_path)
    if change == "missing":
        (reply / "one.json").unlink()
    elif change == "escape":
        model = json.loads((reply / "model.json").read_bytes())
        model["configurations"][0]["abstractTargets"][0]["jsonFile"] = "../one.json"
        write_json(reply / "model.json", model)
    elif change == "identity":
        target = json.loads((reply / "one.json").read_bytes())
        target["id"] = "different"
        write_json(reply / "one.json", target)
    else:
        model = json.loads((reply / "model.json").read_bytes())
        model["version"]["minor"] = 8
        write_json(reply / "model.json", model)

    with pytest.raises(ValueError, match="File API"):
        recipe.capture_dependencies(work, compiled=False)


def test_private_python_native_input_requires_source_attested_bytes(recipe, tmp_path):
    library = tmp_path / "python/lib/libpython3.14.so.1.0"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"approved standalone Python library")
    receipt = tmp_path / "python-receipt.json"
    receipt.write_text('{"source_build":"synthetic-python"}\n')
    runtime = {"source_build": "synthetic-python", "receipt": record(receipt),
               "files": [record(library)]}

    provider = recipe.native_input_provider(library, runtime)

    assert provider == {"kind": "python_runtime", "source_build": "synthetic-python",
                        "receipt": record(receipt)}
    library.write_bytes(b"changed standalone Python library")
    with pytest.raises(ValueError, match="Python native input checksum"):
        recipe.native_input_provider(library, runtime)


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


@pytest.mark.parametrize("stage", ["cpp", "python"])
def test_both_build_stages_force_the_same_exact_private_network_paths(recipe, tmp_path, stage):
    prefix, openssl, work = tmp_path / "network", tmp_path / "openssl", tmp_path / "work"
    context = {"python": "/approved/python", "cc": "/approved/cc", "cxx": "/approved/cxx",
               "work": str(work), "openssl_prefix": str(openssl), "network_prefix": str(prefix)}
    actual = recipe.expected_cache(context, stage=stage)
    required = {"ZLIB_SOURCE": "SYSTEM", "ZLIB_LIBRARY": str(prefix / "lib/libz.a"),
                "ZLIB_INCLUDE_DIR": str(prefix / "include"), "CURL_LIBRARY_RELEASE": str(prefix / "lib/libcurl.a"),
                "CURL_LIBRARY": str(prefix / "lib/libcurl.a"), "CURL_INCLUDE_DIR": str(prefix / "include"),
                "CURL_NO_CURL_CMAKE": "TRUE", "CURL_USE_STATIC_LIBS": "TRUE",
                "LIBXML2_LIBRARY": str(prefix / "lib/libxml2.a"),
                "LIBXML2_INCLUDE_DIR": str(prefix / "include/libxml2"),
                "Iconv_LIBRARY": str(prefix / "lib/libiconv.a"), "Iconv_INCLUDE_DIR": str(prefix / "include"),
                "Iconv_IS_BUILT_IN": "FALSE", "OPENSSL_USE_STATIC_LIBS": "TRUE",
                "OPENSSL_SSL_LIBRARY": str(openssl / "lib/libssl.a"),
                "OPENSSL_CRYPTO_LIBRARY": str(openssl / "lib/libcrypto.a"),
                "NGHTTP2_LIBRARY": str(prefix / "lib/libnghttp2.a"), "NGHTTP2_INCLUDE_DIR": str(prefix / "include"),
                "CMAKE_PROJECT_INCLUDE": str(work / "private-network.cmake")}
    assert {key: actual.get(key) for key in required} == required
    commands = recipe.build_commands(tmp_path / "source", work, openssl, network_prefix=prefix,
                                     python=context["python"], cc=context["cc"], cxx=context["cxx"], jobs=4)
    command = commands["cpp_configure" if stage == "cpp" else "python_build"]
    marker = "-D" if stage == "cpp" else "--config-setting=cmake.define."
    assert all(f"{marker}{key}={value}" in command for key, value in required.items())


def test_private_network_cmake_wires_static_dependency_interfaces(recipe, tmp_path):
    prefix, openssl = tmp_path / "network", tmp_path / "openssl"
    text = recipe.private_network_cmake(prefix, openssl).decode()
    for target in ("CURL::libcurl", "LibXml2::LibXml2", "Iconv::Iconv", "ZLIB::ZLIB",
                   "OpenSSL::SSL", "OpenSSL::Crypto", "biominer_nghttp2"):
        assert f"add_library({target} STATIC IMPORTED GLOBAL)" in text
    for path in (prefix / "lib/libcurl.a", prefix / "lib/libxml2.a", prefix / "lib/libiconv.a",
                 prefix / "lib/libz.a", prefix / "lib/libnghttp2.a", openssl / "lib/libssl.a", openssl / "lib/libcrypto.a"):
        assert str(path) in text
    assert "INTERFACE_LINK_LIBRARIES" in text
    assert "ZLIB::ZLIB;OpenSSL::SSL;biominer_nghttp2" in text
    assert "Iconv::Iconv;ZLIB::ZLIB" in text


def test_arrow26_does_not_execute_an_unused_bundled_zlib_patch(recipe, source_inputs):
    archive, work, _ = source_inputs
    source, _ = recipe.prepare_source(archive, work, resume=False)
    text = (source / "cpp/cmake_modules/ThirdpartyToolchain.cmake").read_text()
    assert "--apply-zlib-patch" not in text
    assert "PATCH_COMMAND" not in text
    assert "-DZLIB_BUILD_SHARED=OFF" not in text


def test_final_link_maps_reject_affected_host_libraries(recipe, network_inputs, tmp_path):
    network = verified_network(recipe, network_inputs)
    work = tmp_path / "arrow-work"
    (work / "cpp-build/release").mkdir(parents=True)
    archives = [row for row in network["files"] if Path(row["path"]).suffix == ".a"]
    host = tmp_path / "host/libz.so.1"
    host.parent.mkdir()
    host.write_bytes(b"host zlib")
    (work / "cpp-build/release/libarrow.so.2600.0.0.map").write_text(
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
    final = work / "cpp-build/release/libarrow.so.2600.0.0.map"
    final.parent.mkdir(parents=True)
    final.write_text("Linker script and memory map\n")
    with pytest.raises(ValueError, match="final.*static linkage"):
        recipe.verify_link_maps(work, network_inputs[2], {"static_archives": archives}, network=network)


def test_network_helper_pin_pending_keeps_real_configuration_closed(recipe, network_inputs, monkeypatch):
    receipt, helper, openssl, _ = network_inputs
    monkeypatch.setattr(recipe, "NETWORK_HELPER_SHA256", None)
    with pytest.raises(ValueError, match="network helper checksum"):
        recipe.validate_network(receipt, digest(receipt.read_bytes()), helper, openssl_prefix=openssl)


@pytest.mark.parametrize("change", ["raw_receipt", "duplicate", "missing_archive", "private_source", "unowned_file"])
def test_network_descriptor_rejects_incomplete_or_conflicting_provider_ownership(recipe, network_inputs, monkeypatch, change):
    import copy

    receipt, helper, openssl, original = network_inputs
    value = copy.deepcopy(original)
    if change == "raw_receipt":
        value["build"]["ready"] = True
    elif change == "duplicate":
        value["files"].append(value["files"][0])
    elif change == "missing_archive":
        del value["private_files"][str(Path(value["prefix"]) / "lib/libz.a")]
    elif change == "private_source":
        value["private_files"][str(Path(value["prefix"]) / "lib/libcurl.a")]["source_version"] = "8.18.0"
    else:
        value["private_files"][str(openssl / "lib/libcrypto.a")] = {
            "component": "zlib", "source_package": "zlib", "source_version": "1.3.2",
            "sha256": digest((openssl / "lib/libcrypto.a").read_bytes())}
    helper.write_text("def verify_completed_prefix(receipt_path, expected_sha256):\n    return " + repr(value) + "\n")
    monkeypatch.setattr(recipe, "NETWORK_HELPER_SHA256", digest(helper.read_bytes()))
    with pytest.raises(ValueError, match="network"):
        recipe.validate_network(receipt, digest(receipt.read_bytes()), helper, openssl_prefix=openssl)


@pytest.mark.parametrize("stage", ["cpp", "python"])
@pytest.mark.parametrize("key", ["ZLIB_LIBRARY", "ZLIB_INCLUDE_DIR", "CURL_LIBRARY_RELEASE", "LIBXML2_LIBRARY",
                                 "Iconv_LIBRARY", "OPENSSL_SSL_LIBRARY", "NGHTTP2_LIBRARY", "CMAKE_PROJECT_INCLUDE"])
def test_both_cache_guards_reject_host_or_replaced_network_inputs(recipe, tmp_path, stage, key):
    context = {"work": str(tmp_path / "work"), "python": "/approved/python", "cc": "/approved/cc", "cxx": "/approved/cxx",
               "openssl_prefix": str(tmp_path / "openssl"), "network_prefix": str(tmp_path / "network")}
    entries = recipe.expected_cache(context, stage=stage)
    entries[key] = "/usr/unapproved/provider"
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text("\n".join(f"{name}:STRING={value}" for name, value in entries.items()) + "\n")
    with pytest.raises(ValueError, match=key):
        recipe.verify_cache(cache, context, stage=stage)


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
        "provider_cmake": record(metadata)})
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
            inputs["provider_cmake"] = record(metadata)
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


def test_dependency_capture_preserves_private_network_provider_identities(recipe, network_inputs, tmp_path):
    network = verified_network(recipe, network_inputs)
    work, reply, _ = configured_target_fixture(recipe, tmp_path)
    inputs = json.loads((work / "build-inputs.json").read_bytes())
    inputs["network"] = network
    write_json(work / "build-inputs.json", inputs)
    model = json.loads((reply / "model.json").read_bytes())
    for index, name in enumerate(recipe.NETWORK_ARCHIVES):
        identity = "private-" + str(index)
        path = Path(network["prefix"]) / "lib" / name
        write_json(reply / (identity + ".json"), {"name": name, "id": identity, "type": "STATIC_LIBRARY",
            "imported": True, "local": False, "paths": {"source": ".", "build": "."},
            "artifacts": [{"path": str(path)}]})
        model["configurations"][0]["abstractTargets"].append({"name": name, "id": identity, "jsonFile": identity + ".json"})
    write_json(reply / "model.json", model)

    result = recipe.capture_dependencies(work, compiled=False)

    assert result["private_network"] == {key: network[key] for key in ("receipt", "helper", "prefix")}
    assert set(result["forced_bundled_source_verification"]) == {"simdjson", "uriparser", "boost"}
    providers = {Path(row["path"]).name: row["provider"] for row in result["static_archives"] if "provider" in row}
    assert set(providers) == set(recipe.NETWORK_ARCHIVES)
    assert all(value["kind"] == "private_network" for value in providers.values())
    assert providers["libz.a"]["file"] == record(Path(network["prefix"]) / "lib/libz.a")


def test_dependency_capture_rejects_a_second_bundled_zlib_provider(recipe, tmp_path):
    work, _, _ = configured_target_fixture(recipe, tmp_path)
    (work / "cpp-build/zlib_ep").mkdir()
    with pytest.raises(ValueError, match="second bundled zlib provider"):
        recipe.capture_dependencies(work, compiled=False)


@pytest.mark.parametrize("fail", [False, True])
def test_network_stage_rechecks_closure_after_success_or_subprocess_failure(recipe, network_inputs, tmp_path, monkeypatch, fail):
    network = verified_network(recipe, network_inputs)
    events = []
    guard = recipe.verify_network

    def observed_guard(value):
        events.append("verify")
        guard(value)

    def stage(*args):
        events.append("stage")
        if fail:
            raise subprocess.CalledProcessError(1, ["fixture"])
        return {"exit_code": 0}

    monkeypatch.setattr(recipe, "verify_network", observed_guard)
    monkeypatch.setattr(recipe, "run_stage", stage)
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            recipe.run_network_stage(tmp_path, "fixture", ["fixture"], {}, network)
    else:
        assert recipe.run_network_stage(tmp_path, "fixture", ["fixture"], {}, network) == {"exit_code": 0}
    assert events == ["verify", "stage", "verify"]


def test_network_stage_cannot_publish_success_after_provider_mutation(recipe, network_inputs, tmp_path, monkeypatch):
    network = verified_network(recipe, network_inputs)

    def stage(*args):
        (Path(network["prefix"]) / "include/zlib.h").write_bytes(b"changed during compilation")
        return {"exit_code": 0}

    monkeypatch.setattr(recipe, "run_stage", stage)
    with pytest.raises(ValueError, match="network.*checksum"):
        recipe.run_network_stage(tmp_path, "fixture", ["fixture"], {}, network)


def test_network_stage_rechecks_captured_cmake_metadata_after_compilation(recipe, network_inputs, tmp_path, monkeypatch):
    network = verified_network(recipe, network_inputs)
    metadata = tmp_path / "private-network.cmake"
    metadata.write_bytes(recipe.private_network_cmake(Path(network["prefix"]), network_inputs[2]))
    original = record(metadata)

    def stage(*args):
        metadata.write_text('add_library(unapproved SHARED IMPORTED)\n')
        return {"exit_code": 0}

    monkeypatch.setattr(recipe, "run_stage", stage)
    with pytest.raises(ValueError, match="immutable build input checksum"):
        recipe.run_network_stage(tmp_path, "fixture", ["fixture"], {}, network, frozen_inputs=(original,))


@pytest.mark.parametrize("relative_host", [False, True])
def test_final_link_maps_require_actual_private_closure_and_allow_unused_charset(recipe, network_inputs, tmp_path, relative_host):
    network = verified_network(recipe, network_inputs)
    work = tmp_path / "arrow-work"
    (work / "cpp-build/release").mkdir(parents=True)
    archives = [row for row in network["files"] if Path(row["path"]).suffix == ".a" and not row["path"].endswith("libcharset.a")]
    lines = ["Linker script and memory map", *[row["path"] + "(member.o)" for row in archives]]
    if relative_host:
        host = work / "cpp-build/host/libcurl.so.4"
        host.parent.mkdir()
        host.write_bytes(b"unapproved duplicate host curl")
        lines.append("LOAD host/libcurl.so.4")
    final = work / "cpp-build/release/libarrow.so.2600.0.0.map"
    final.write_text("\n".join(lines) + "\n")
    if relative_host:
        with pytest.raises(ValueError, match="unapproved.*provider"):
            recipe.verify_link_maps(work, network_inputs[2], {"static_archives": archives}, network=network)
    else:
        result = recipe.verify_link_maps(work, network_inputs[2], {"static_archives": archives}, network=network)
        assert result["final_maps"] == [record(final)]
        assert len(result["linked_static_archives"]) == 7
        assert all(row["provider"]["kind"] == "private_network" for row in result["linked_static_archives"]
                   if Path(row["path"]).is_relative_to(Path(network["prefix"])))


@pytest.mark.parametrize("unused", [None, "libcurl.a", "libxml2.a", "libiconv.a", "libz.a", "libnghttp2.a", "libssl.a", "libcrypto.a"])
def test_final_link_maps_cannot_treat_an_unused_load_as_consumed_private_code(recipe, network_inputs, tmp_path, unused):
    network = verified_network(recipe, network_inputs)
    work = tmp_path / "arrow-work"
    (work / "cpp-build/release").mkdir(parents=True)
    archives = [row for row in network["files"] if Path(row["path"]).suffix == ".a" and not row["path"].endswith("libcharset.a")]
    lines = ["Linker script and memory map"]
    for row in archives:
        # GNU ld prints LOAD even when an archive contributes no object.
        lines.append("LOAD " + row["path"])
        if unused is not None and Path(row["path"]).name != unused:
            lines.append(row["path"] + "(member.o)")
    final = work / "cpp-build/release/libarrow.so.2600.0.0.map"
    final.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="final.*consumed archive member" + (".*" + unused if unused else "")):
        recipe.verify_link_maps(work, network_inputs[2], {"static_archives": archives}, network=network)


# These security source pins are an independent test oracle. They were reviewed
# in recipe-cohort-1.json (584b47d2...); tests never read that external artifact.
EXPECTED_SDK_COHORT = {
    "aws_sdk_cpp": {
        "variable": "AWSSDK",
        "version": "1.11.914",
        "build_version": "d207a37ecb8dc33b5e131d0d59843d264b7f7e86",
        "commit": "d207a37ecb8dc33b5e131d0d59843d264b7f7e86",
        "sha256": "108382fed71e70ada73d4aaa45e0ba54777f7eca6bbd44ac24b9af9ef54f069e",
        "filename": "aws-sdk-cpp-d207a37ecb8dc33b5e131d0d59843d264b7f7e86.tar.gz",
        "env": "ARROW_AWSSDK_URL",
        "old_version": "1.11.778",
        "old_sha256": "06371849792f1a8633c3e0592a17b4700f39c748ad374f4e9a6592739ecc4d9c",
        "url": "https://github.com/aws/aws-sdk-cpp/archive/d207a37ecb8dc33b5e131d0d59843d264b7f7e86.tar.gz"
    },
    "aws_crt_cpp": {
        "variable": "AWS_CRT_CPP",
        "version": "0.43.8",
        "build_version": "9c319786d1d83d5ace6a9de817dcfe1861003609",
        "commit": "9c319786d1d83d5ace6a9de817dcfe1861003609",
        "sha256": "f107357efa7d6b9be783d96d49cdaf000fe881417c431ae9fe27faf1bb55f431",
        "filename": "aws-crt-cpp-9c319786d1d83d5ace6a9de817dcfe1861003609.tar.gz",
        "env": "ARROW_AWS_CRT_CPP_URL",
        "old_version": "v0.38.0",
        "old_sha256": "f57b82f0a9e6890e4d47aa1c48b9324b5beb5b58000715c4c6ef9f147be13891",
        "url": "https://github.com/awslabs/aws-crt-cpp/archive/9c319786d1d83d5ace6a9de817dcfe1861003609.tar.gz"
    },
    "aws_c_auth": {
        "variable": "AWS_C_AUTH",
        "version": "4b5d524bf1a511b05e0fffe5bdc51800770b9427",
        "build_version": "4b5d524bf1a511b05e0fffe5bdc51800770b9427",
        "commit": "4b5d524bf1a511b05e0fffe5bdc51800770b9427",
        "sha256": "b4f6690b382abe049d543804f40230e667a95a18f64506ef7647925e8c6b9b67",
        "filename": "aws-c-auth-4b5d524bf1a511b05e0fffe5bdc51800770b9427.tar.gz",
        "env": "ARROW_AWS_C_AUTH_URL",
        "old_version": "v0.10.1",
        "old_sha256": "85d737f0f735256f1931e85e4cadbe228d88698f7b59a9b390b49ef5d0778a43",
        "url": "https://github.com/awslabs/aws-c-auth/archive/4b5d524bf1a511b05e0fffe5bdc51800770b9427.tar.gz"
    },
    "aws_c_cal": {
        "variable": "AWS_C_CAL",
        "version": "8aa2a48a09f93c65d4cf06388e143a6584de6321",
        "build_version": "8aa2a48a09f93c65d4cf06388e143a6584de6321",
        "commit": "8aa2a48a09f93c65d4cf06388e143a6584de6321",
        "sha256": "29706ab10a4d95a74db278925b82c85077c08224c0efe837841d2354555c8817",
        "filename": "aws-c-cal-8aa2a48a09f93c65d4cf06388e143a6584de6321.tar.gz",
        "env": "ARROW_AWS_C_CAL_URL",
        "old_version": "v0.9.13",
        "old_sha256": "80b7c6087b0af461b4483e4c9483aea2e0dac5d9fb2289b057159ea6032409e1",
        "url": "https://github.com/awslabs/aws-c-cal/archive/8aa2a48a09f93c65d4cf06388e143a6584de6321.tar.gz"
    },
    "aws_c_common": {
        "variable": "AWS_C_COMMON",
        "version": "f7d471b1feaea9a310851672580f9ac7e031cfc0",
        "build_version": "f7d471b1feaea9a310851672580f9ac7e031cfc0",
        "commit": "f7d471b1feaea9a310851672580f9ac7e031cfc0",
        "sha256": "a87a89fa6678fda59dd66ea9246e70d837bf71ecec39829ec712e6863d1453e4",
        "filename": "aws-c-common-f7d471b1feaea9a310851672580f9ac7e031cfc0.tar.gz",
        "env": "ARROW_AWS_C_COMMON_URL",
        "old_version": "v0.12.6",
        "old_sha256": "138822ecdcaff1d702f37d4751f245847d088592724921cc6bf61c232b198d6b",
        "url": "https://github.com/awslabs/aws-c-common/archive/f7d471b1feaea9a310851672580f9ac7e031cfc0.tar.gz"
    },
    "aws_c_compression": {
        "variable": "AWS_C_COMPRESSION",
        "version": "d8264e64f698341eb03039b96b4f44702a9b3f83",
        "build_version": "d8264e64f698341eb03039b96b4f44702a9b3f83",
        "commit": "d8264e64f698341eb03039b96b4f44702a9b3f83",
        "sha256": "537971fe42e989b638e0f436683357ed3d07f07b06b910b1c37ab3a8094bb8fd",
        "filename": "aws-c-compression-d8264e64f698341eb03039b96b4f44702a9b3f83.tar.gz",
        "env": "ARROW_AWS_C_COMPRESSION_URL",
        "old_version": "v0.3.2",
        "old_sha256": "f93f5a5d8b3fee3a6d97b14ba279efacd4d4016ef9cc7dc4be7d43519ecfbe93",
        "url": "https://github.com/awslabs/aws-c-compression/archive/d8264e64f698341eb03039b96b4f44702a9b3f83.tar.gz"
    },
    "aws_c_event_stream": {
        "variable": "AWS_C_EVENT_STREAM",
        "version": "51bef3c44e1058b1689751539170b2e0f589ccdb",
        "build_version": "51bef3c44e1058b1689751539170b2e0f589ccdb",
        "commit": "51bef3c44e1058b1689751539170b2e0f589ccdb",
        "sha256": "b890d4940957d1ddade1a9d7de4adc37b7cb98dfdfdfec7436527fec8cb558df",
        "filename": "aws-c-event-stream-51bef3c44e1058b1689751539170b2e0f589ccdb.tar.gz",
        "env": "ARROW_AWS_C_EVENT_STREAM_URL",
        "old_version": "v0.7.0",
        "old_sha256": "88835b4c78462547917f622fd9dda45c991b7e356d9c07e2f0537d4d97fbd4fb",
        "url": "https://github.com/awslabs/aws-c-event-stream/archive/51bef3c44e1058b1689751539170b2e0f589ccdb.tar.gz"
    },
    "aws_c_http": {
        "variable": "AWS_C_HTTP",
        "version": "8aefd899fc3210bfd0e3fd414011a3cb708bf6e4",
        "build_version": "8aefd899fc3210bfd0e3fd414011a3cb708bf6e4",
        "commit": "8aefd899fc3210bfd0e3fd414011a3cb708bf6e4",
        "sha256": "61b1770e2e3a2bf046005cbfa8e26d621d8f266f05d03589ebee36bb4a0ac39f",
        "filename": "aws-c-http-8aefd899fc3210bfd0e3fd414011a3cb708bf6e4.tar.gz",
        "env": "ARROW_AWS_C_HTTP_URL",
        "old_version": "v0.10.13",
        "old_sha256": "d8352e7a1fb1996694a4dc31219ce03452882abf8d0858c104727f975e11b9c7",
        "url": "https://github.com/awslabs/aws-c-http/archive/8aefd899fc3210bfd0e3fd414011a3cb708bf6e4.tar.gz"
    },
    "aws_c_io": {
        "variable": "AWS_C_IO",
        "version": "28e4eb351a536daf1627bf69279e50671f7729b5",
        "build_version": "28e4eb351a536daf1627bf69279e50671f7729b5",
        "commit": "28e4eb351a536daf1627bf69279e50671f7729b5",
        "sha256": "ea51cbe20981ed34b7a8c01a60c848e4ccb716a29cd740ead07ffd61be5169c3",
        "filename": "aws-c-io-28e4eb351a536daf1627bf69279e50671f7729b5.tar.gz",
        "env": "ARROW_AWS_C_IO_URL",
        "old_version": "v0.26.3",
        "old_sha256": "521fd0848fca661130bbb7278a414d7a38bdcb9bc8ffa89f6660d84e5838a303",
        "url": "https://github.com/awslabs/aws-c-io/archive/28e4eb351a536daf1627bf69279e50671f7729b5.tar.gz"
    },
    "aws_c_mqtt": {
        "variable": "AWS_C_MQTT",
        "version": "e35b9ca3f9fcbf1a972c831c5e79046ce56959d1",
        "build_version": "e35b9ca3f9fcbf1a972c831c5e79046ce56959d1",
        "commit": "e35b9ca3f9fcbf1a972c831c5e79046ce56959d1",
        "sha256": "54d1987ce06eb7b39faace854818e9322d3df79cb310ada350beca0b12cb89cc",
        "filename": "aws-c-mqtt-e35b9ca3f9fcbf1a972c831c5e79046ce56959d1.tar.gz",
        "env": "ARROW_AWS_C_MQTT_URL",
        "old_version": "v0.15.2",
        "old_sha256": "66f3f5edff4ad1f765a86d3342b6017d0f29f950c1c24f8c1edacdc895202edc",
        "url": "https://github.com/awslabs/aws-c-mqtt/archive/e35b9ca3f9fcbf1a972c831c5e79046ce56959d1.tar.gz"
    },
    "aws_c_s3": {
        "variable": "AWS_C_S3",
        "version": "226c3e6937e3d5f7db9e10651f7bd20575e98187",
        "build_version": "226c3e6937e3d5f7db9e10651f7bd20575e98187",
        "commit": "226c3e6937e3d5f7db9e10651f7bd20575e98187",
        "sha256": "a8f33816bf7aff38467fb15ca233a33a9113be78dad47c8a7ee30d323b42ba74",
        "filename": "aws-c-s3-226c3e6937e3d5f7db9e10651f7bd20575e98187.tar.gz",
        "env": "ARROW_AWS_C_S3_URL",
        "old_version": "v0.12.0",
        "old_sha256": "1a8a8ceda0585d52028a1f3daa5861f924e7d8d2f6a17bec05813dc0b74d6eed",
        "url": "https://github.com/awslabs/aws-c-s3/archive/226c3e6937e3d5f7db9e10651f7bd20575e98187.tar.gz"
    },
    "aws_c_sdkutils": {
        "variable": "AWS_C_SDKUTILS",
        "version": "a1cc19f53b63658f1b1400b36f199eafeeb895a6",
        "build_version": "a1cc19f53b63658f1b1400b36f199eafeeb895a6",
        "commit": "a1cc19f53b63658f1b1400b36f199eafeeb895a6",
        "sha256": "f8ef9e93b30f9833d9c9ea474110c8fa2d29fd78b885546b07e68558ece3bb1c",
        "filename": "aws-c-sdkutils-a1cc19f53b63658f1b1400b36f199eafeeb895a6.tar.gz",
        "env": "ARROW_AWS_C_SDKUTILS_URL",
        "old_version": "v0.2.4",
        "old_sha256": "493cbed4fa57e0d4622fcff044e11305eb4fc12445f32c8861025597939175fc",
        "url": "https://github.com/awslabs/aws-c-sdkutils/archive/a1cc19f53b63658f1b1400b36f199eafeeb895a6.tar.gz"
    },
    "aws_checksums": {
        "variable": "AWS_CHECKSUMS",
        "version": "1d5f2f1f3e5d013aae8810878ceb5b3f6f258c4e",
        "build_version": "1d5f2f1f3e5d013aae8810878ceb5b3f6f258c4e",
        "commit": "1d5f2f1f3e5d013aae8810878ceb5b3f6f258c4e",
        "sha256": "c58d8c5319319dcade0926da44ad12effecb21372983e22835a657a32f03c9c9",
        "filename": "aws-checksums-1d5f2f1f3e5d013aae8810878ceb5b3f6f258c4e.tar.gz",
        "env": "ARROW_AWS_CHECKSUMS_URL",
        "old_version": "v0.2.10",
        "old_sha256": "cb6509f75e42ee25c372a6d379e8582ce5179e5335183842e808f7d8abb0c314",
        "url": "https://github.com/awslabs/aws-checksums/archive/1d5f2f1f3e5d013aae8810878ceb5b3f6f258c4e.tar.gz"
    },
    "s2n_tls": {
        "variable": "S2N_TLS",
        "version": "1.7.12",
        "build_version": "6459002cd61e4bf3bbc53db23f068b21da9ff6bd",
        "commit": "6459002cd61e4bf3bbc53db23f068b21da9ff6bd",
        "sha256": "94a57c4f4713cc20088b723bcdf0bb254e32a559ffcbed583e67fe3d8ddec442",
        "filename": "s2n-tls-6459002cd61e4bf3bbc53db23f068b21da9ff6bd.tar.gz",
        "env": "ARROW_S2N_TLS_URL",
        "old_version": "v1.7.0",
        "old_sha256": "a6e8228e238239bb3c17b1eda3ed702bcbb2eaebc792eac4d754cc5619b0ea06",
        "url": "https://github.com/aws/s2n-tls/archive/6459002cd61e4bf3bbc53db23f068b21da9ff6bd.tar.gz"
    },
    "thrift": {
        "variable": "THRIFT",
        "version": "0.25.0",
        "build_version": "0.25.0",
        "commit": "27e8a425ffb498e190df3a12e239326bf5ba9ed6",
        "sha256": "66da4707214c54c94bac082103dc67adaf9e08925662700f269170a7b534b214",
        "filename": "thrift-27e8a425ffb498e190df3a12e239326bf5ba9ed6.tar.gz",
        "env": "ARROW_THRIFT_URL",
        "old_version": "0.22.0",
        "old_sha256": "794a0e455787960d9f27ab92c38e34da27e8deeda7a5db0e59dc64a00df8a1e5",
        "url": "https://www.apache.org/dyn/closer.lua/thrift/0.25.0/thrift-0.25.0.tar.gz?action=download"
    },
    "grpc": {
        "variable": "GRPC",
        "version": "1.83.1",
        "build_version": "v1.83.1",
        "commit": "aae267021b1ac256f8b9038d0ef528c3798cc137",
        "sha256": "60caa8397426d8a500e3e57ab6a49d1cb5aa62a36f2591eb9da3d77fa38ad8c9",
        "filename": "grpc-v1.83.1.tar.gz",
        "env": "ARROW_GRPC_URL",
        "old_version": "v1.76.0",
        "old_sha256": "0af37b800953130b47c075b56683ee60bdc3eda3c37fc6004193f5b569758204",
        "url": "https://github.com/grpc/grpc/archive/v1.83.1.tar.gz"
    },
    "protobuf": {
        "variable": "PROTOBUF",
        "version": "35.1",
        "build_version": "v35.1",
        "commit": "35cd01f9fe9afbeea38cc7b979a3b6bfcde82c03",
        "sha256": "f0b6838e7522a8da96126d487068c959bc624926368f3024ac8fd03abd0a1ac4",
        "filename": "protobuf-35.1.tar.gz",
        "env": "ARROW_PROTOBUF_URL",
        "old_version": "v31.1",
        "old_sha256": "12bfd76d27b9ac3d65c00966901609e020481b9474ef75c7ff4601ac06fa0b82",
        "url": "https://github.com/protocolbuffers/protobuf/releases/download/v35.1/protobuf-35.1.tar.gz"
    },
    "absl": {
        "variable": "ABSL",
        "version": "20250512.1",
        "build_version": "20250512.1",
        "commit": "76bb24329e8bf5f39704eb10d21b9a80befa7c81",
        "sha256": "9b7a064305e9fd94d124ffa6cc358592eb42b5da588fb4e07d09254aa40086db",
        "filename": "absl-20250512.1.tar.gz",
        "env": "ARROW_ABSL_URL",
        "old_version": "20250127.0",
        "old_sha256": "16242f394245627e508ec6bb296b433c90f8d914f73b9c026fddb905e27276e8",
        "url": "https://github.com/abseil/abseil-cpp/archive/20250512.1.tar.gz"
    },
    "cares": {
        "variable": "CARES",
        "version": "1.34.9",
        "build_version": "1.34.9",
        "commit": "bc91829cce983a35ec01f5dbc05465858a3df469",
        "sha256": "9364b70a12eb59a06830fb488d9357f90100dff0d98b23ea3593a183d43e6f2f",
        "filename": "c-ares-1.34.9.tar.gz",
        "env": "ARROW_CARES_URL",
        "old_version": "1.34.6",
        "old_sha256": "912dd7cc3b3e8a79c52fd7fb9c0f4ecf0aaa73e45efda880266a2d6e26b84ef5",
        "url": "https://github.com/c-ares/c-ares/releases/download/v1.34.9/c-ares-1.34.9.tar.gz"
    },
    "rapidjson": {
        "variable": "RAPIDJSON",
        "version": "24b5e7a8b27f42fa16b96fc70aade9106cf7102f",
        "build_version": "24b5e7a8b27f42fa16b96fc70aade9106cf7102f",
        "commit": "24b5e7a8b27f42fa16b96fc70aade9106cf7102f",
        "sha256": "2d2601a82d2d3b7e143a3c8d43ef616671391034bc46891a9816b79cf2d3e7a8",
        "filename": "rapidjson-24b5e7a8b27f42fa16b96fc70aade9106cf7102f.tar.gz",
        "env": "ARROW_RAPIDJSON_URL",
        "old_version": "232389d4f1012dddec4ef84861face2d2ba85709",
        "old_sha256": "b9290a9a6d444c8e049bd589ab804e0ccf2b05dc5984a19ed5ae75d090064806",
        "url": "https://github.com/Tencent/rapidjson/archive/24b5e7a8b27f42fa16b96fc70aade9106cf7102f.tar.gz"
    },
    "jemalloc": {
        "variable": "JEMALLOC",
        "version": "5.3.1",
        "build_version": "5.3.1",
        "commit": "81034ce1f1373e37dc865038e1bc8eeecf559ce8",
        "sha256": "3826bc80232f22ed5c4662f3034f799ca316e819103bdc7bb99018a421706f92",
        "filename": "jemalloc-5.3.1.tar.bz2",
        "env": "ARROW_JEMALLOC_URL",
        "old_version": "5.3.0",
        "old_sha256": "2db82d1e7119df3e71b7640219b6dfe84789bc0537983c3b7ac4f7189aecfeaa",
        "url": "https://github.com/jemalloc/jemalloc/releases/download/5.3.1/jemalloc-5.3.1.tar.bz2"
    }
}


def test_sdk_cohort_requires_exact_reviewed_source_inputs(recipe):
    assert recipe.SOURCE_POLICY == "arrow26-integrated-private-network-v3"
    assert set(recipe.DEPENDENCIES) == {
        "simdjson", "uriparser", "boost", "zlib", *EXPECTED_SDK_COHORT,
    }
    for name, expected in EXPECTED_SDK_COHORT.items():
        assert {key: recipe.DEPENDENCIES[name].get(key) for key in expected} == expected
        assert "path" not in recipe.DEPENDENCIES[name]


@pytest.fixture
def cohort_source_inputs(recipe, source_inputs, monkeypatch):
    """Use independent original assignments, including when recipe pins drift."""
    archive, work, files = source_inputs
    versions_path = "cpp/thirdparty/versions.txt"
    lines = files[versions_path].decode().splitlines()
    for expected in EXPECTED_SDK_COHORT.values():
        for assignment in (
            f"ARROW_{expected['variable']}_BUILD_VERSION={expected['old_version']}",
            f"ARROW_{expected['variable']}_BUILD_SHA256_CHECKSUM={expected['old_sha256']}",
        ):
            if assignment not in lines:
                lines.append(assignment)
    lines.append("ARROW_UNRELATED_BUILD_VERSION=keep-this-exactly")
    files[versions_path] = ("\n".join(lines) + "\n").encode()
    toolchain_path = "cpp/cmake_modules/ThirdpartyToolchain.cmake"
    toolchain = files[toolchain_path].decode()
    uri_anchor = "  fetchcontent_makeavailable(uriparser)"
    if uri_anchor in toolchain:
        toolchain = toolchain.replace(uri_anchor + "\n", "")
    toolchain += "function(build_uriparser)\n" + uri_anchor + "\nendfunction()\n"
    toolchain += "function(unrelated_target)\n  set(CMAKE_UNITY_BUILD ON)\nendfunction()\n"
    files[toolchain_path] = toolchain.encode()
    files["cpp/thirdparty/uri-feature-test-fixture.c"] = (
        b"#define _DEFAULT_SOURCE\n#include <stdlib.h>\n"
        b"void *unchanged_feature_test(void *p) { return reallocarray(p, 2, 4); }\n")
    archive_at(archive, f"arrow-{recipe.ARROW_COMMIT}", files,
               {"python/NOTICE.txt": "../NOTICE.txt"})
    monkeypatch.setattr(recipe, "ARROW_ARCHIVE_SHA256", digest(archive.read_bytes()))
    return archive, work, files


def test_sdk_cohort_source_transformation_updates_every_reviewed_assignment(
        recipe, cohort_source_inputs):
    archive, work, files = cohort_source_inputs
    source, evidence = recipe.prepare_source(archive, work, resume=False)
    versions = (source / "cpp/thirdparty/versions.txt").read_text()
    assignments = dict(line.split("=", 1) for line in versions.splitlines()
                       if line.startswith("ARROW_") and "=" in line)
    for expected in EXPECTED_SDK_COHORT.values():
        assert assignments[f"ARROW_{expected['variable']}_BUILD_VERSION"] == expected["build_version"]
        assert assignments[f"ARROW_{expected['variable']}_BUILD_SHA256_CHECKSUM"] == expected["sha256"]
    assert assignments["ARROW_UNRELATED_BUILD_VERSION"] == "keep-this-exactly"
    for name, data in files.items():
        if name not in {"cpp/cmake_modules/ThirdpartyToolchain.cmake", "cpp/thirdparty/versions.txt"}:
            assert (source / name).read_bytes() == data
    assert recipe.prepare_source(archive, work, resume=True) == (source, evidence)


def test_uriparser_unity_fix_only_changes_its_target(recipe, cohort_source_inputs):
    archive, work, files = cohort_source_inputs
    source, _ = recipe.prepare_source(archive, work, resume=False)
    toolchain = (source / "cpp/cmake_modules/ThirdpartyToolchain.cmake").read_text()
    uri_body = toolchain.split("function(build_uriparser)\n", 1)[1].split("endfunction()", 1)[0]
    assert uri_body == (
        "  fetchcontent_makeavailable(uriparser)\n"
        "  set_target_properties(uriparser PROPERTIES UNITY_BUILD OFF)\n")
    assert "function(unrelated_target)\n  set(CMAKE_UNITY_BUILD ON)\nendfunction()\n" in toolchain
    uri_source = "cpp/thirdparty/uri-feature-test-fixture.c"
    assert (source / uri_source).read_bytes() == files[uri_source]
    commands = recipe.build_commands(source, work, work / "openssl",
        network_prefix=work / "network", python="/python", cc="/cc", cxx="/cxx", jobs=4)
    assert "-DCMAKE_UNITY_BUILD=ON" in commands["cpp_configure"]


@pytest.mark.parametrize("name", ["aws_sdk_cpp", "grpc", "rapidjson", "jemalloc"])
@pytest.mark.parametrize("field", ["version", "checksum"])
@pytest.mark.parametrize("change", ["missing", "duplicate", "prefixed", "suffixed"])
def test_sdk_source_rejects_changed_or_ambiguous_original_assignments(
        recipe, cohort_source_inputs, monkeypatch, name, field, change):
    archive, work, files = cohort_source_inputs
    pin = EXPECTED_SDK_COHORT[name]
    suffix = "BUILD_VERSION" if field == "version" else "BUILD_SHA256_CHECKSUM"
    original = pin["old_version"] if field == "version" else pin["old_sha256"]
    assignment = f"ARROW_{pin['variable']}_{suffix}={original}"
    if change == "missing":
        replacement = "REMOVED_ASSIGNMENT=unreviewed"
    elif change == "duplicate":
        replacement = assignment + "\n" + assignment
    elif change == "prefixed":
        replacement = "UNOWNED_" + assignment
    else:
        replacement = assignment + "0"
    path = "cpp/thirdparty/versions.txt"
    assert files[path].count((assignment + "\n").encode()) == 1
    files[path] = files[path].replace((assignment + "\n").encode(), (replacement + "\n").encode())
    archive_at(archive, f"arrow-{recipe.ARROW_COMMIT}", files,
               {"python/NOTICE.txt": "../NOTICE.txt"})
    monkeypatch.setattr(recipe, "ARROW_ARCHIVE_SHA256", digest(archive.read_bytes()))

    with pytest.raises(ValueError, match="anchor|assignment"):
        recipe.prepare_source(archive, work, resume=False)
    assert not work.exists()


@pytest.mark.parametrize("name", [
    "ARROW_AWSSDK_URL", "ARROW_GRPC_URL", "ARROW_SUBSTRAIT_URL", "ARROW_UNREVIEWED_URL",
])
def test_inherited_fetch_urls_cannot_enter_the_build_environment(recipe, monkeypatch, name):
    monkeypatch.setenv(name, "/unowned/poisoned-source.tar.gz")
    env = recipe.build_environment(4, python="/python", cc="/cc", cxx="/cxx")
    assert name not in env
    assert not any(key.startswith("ARROW_") and key.endswith("_URL") for key in env)


@pytest.fixture
def cohort_cli(recipe, cohort_source_inputs, builder_inputs, network_inputs, tmp_path, monkeypatch):
    """Run real argument/input/copy/manifest/stage guards with fake tool work."""
    archive, work, _ = cohort_source_inputs
    builder_receipt, builder_hash, builder, _ = builder_inputs
    network_receipt, helper, openssl, descriptor = network_inputs
    original_relative_to = Path.is_relative_to
    monkeypatch.setattr(Path, "is_relative_to", lambda self, *other:
        False if other in (("/tmp",), ("/var/tmp",)) else original_relative_to(self, *other))
    monkeypatch.setattr(recipe, "probe_builder", lambda value: {"identity": "frozen test builder"})
    monkeypatch.setattr(recipe, "validate_python_runtime", lambda value:
        {"source_build": "synthetic-python", "files": []})
    monkeypatch.setattr(recipe, "validate_openssl", lambda *args: {"version": "3.6.5"})
    monkeypatch.setattr(recipe, "expected_generated_version", lambda *args: b"exact test version\n")
    monkeypatch.setattr(recipe.subprocess, "check_output", lambda *args, **kwargs: "fixture tool version\n")
    selected = {"targets": [], "boost_components": [], "boost_selection": {},
                "forced_bundled_source_verification": {}, "static_archives": [],
                "downloaded_archives": [], "source_trees": []}
    monkeypatch.setattr(recipe, "capture_dependencies", lambda *args, **kwargs: selected)
    argv = ["--archive", str(archive), "--work", str(work), "--openssl-prefix", str(openssl),
            "--builder-receipt", str(builder_receipt), "--builder-receipt-sha256", builder_hash,
            "--network-receipt", str(network_receipt),
            "--network-receipt-sha256", digest(network_receipt.read_bytes()),
            "--network-helper", str(helper), "--python", builder["python"]["path"],
            "--cc", builder["toolchain"]["cc"]["path"], "--cxx", builder["toolchain"]["cxx"]["path"]]
    archives, payloads = {}, {}
    for name in ("simdjson", "uriparser", "boost", "zlib", *EXPECTED_SDK_COHORT):
        path = tmp_path / (name + ".fixture.tar.gz")
        payloads[name] = ("reviewed source fixture: " + name).encode()
        path.write_bytes(payloads[name])
        archives[name] = path
        if name in recipe.DEPENDENCIES:
            monkeypatch.setitem(recipe.DEPENDENCIES[name], "sha256", digest(payloads[name]))
        argv.extend(["--" + name.replace("_", "-") + "-archive", str(path)])
    for pin in EXPECTED_SDK_COHORT.values():
        monkeypatch.setenv(pin["env"], "/unowned/poisoned-source.tar.gz")
    monkeypatch.setenv("ARROW_UNREVIEWED_URL", "/unowned/not-in-cohort.tar.gz")
    state = {"argv": argv, "archives": archives, "payloads": payloads,
             "work": work, "stages": [], "mutate_copy": None}

    def configure(stage_work, name, command, env):
        assert name == "configure"
        state["stages"].append({"name": name, "command": command, "env": dict(env)})
        context = {"work": str(work), "openssl_prefix": str(openssl),
                   "network_prefix": descriptor["prefix"], "python": builder["python"]["path"],
                   "cc": builder["toolchain"]["cc"]["path"], "cxx": builder["toolchain"]["cxx"]["path"]}
        cache = work / "cpp-build/CMakeCache.txt"
        cache.parent.mkdir(exist_ok=True)
        cache.write_text("\n".join(f"{key}:STRING={value}" for key, value in
                                   recipe.expected_cache(context, stage="cpp").items()) + "\n")
        if state["mutate_copy"]:
            pin = EXPECTED_SDK_COHORT[state["mutate_copy"]]
            (work / "inputs" / pin["filename"]).write_bytes(b"changed after configure")
        return {"command": command, "exit_code": 0}

    monkeypatch.setattr(recipe, "run_stage", configure)
    return state


def test_sdk_cohort_cli_captures_all_owned_inputs_and_ignores_inherited_urls(recipe, cohort_cli):
    state = cohort_cli
    assert recipe.main(state["argv"] + ["--configure-only"]) == 0
    work = state["work"]
    manifest = json.loads((work / "build-inputs.json").read_bytes())
    receipt = json.loads((work / "build-receipt.json").read_bytes())
    assert manifest["source_policy"] == "arrow26-integrated-private-network-v3"
    assert set(manifest["dependencies"]) == {"simdjson", "uriparser", "boost", "zlib", *EXPECTED_SDK_COHORT}
    assert receipt["build_inputs"] == record(work / "build-inputs.json")
    assert receipt["dependencies"] == manifest["dependencies"]
    assert receipt["status"] == "configured_unverified"
    assert len(state["stages"]) == 1
    env = state["stages"][0]["env"]
    assert "ARROW_UNREVIEWED_URL" not in env
    for name, pin in EXPECTED_SDK_COHORT.items():
        captured = manifest["dependencies"][name]
        copied = work / "inputs" / pin["filename"]
        assert copied.read_bytes() == state["payloads"][name]
        assert captured["archive"] == record(copied)
        assert captured["sha256"] == digest(state["payloads"][name])
        assert captured["role"] == "bundled_provider"
        for key, expected in pin.items():
            if key != "sha256":
                assert captured[key] == expected
        assert env[pin["env"]] == str(copied)
        assert copied != state["archives"][name]
    assert set(manifest["features"]) == {
        "ACERO", "AZURE", "COMPUTE", "CSV", "DATASET", "FILESYSTEM", "FLIGHT",
        "GCS", "HDFS", "JSON", "ORC", "PARQUET", "S3", "SUBSTRAIT",
    }
    assert set(manifest["codecs"]) == {"BROTLI", "BZ2", "LZ4", "SNAPPY", "ZLIB", "ZSTD"}
    assert manifest["dependencies"]["zlib"]["role"] == "retained_source_pin_only"
    assert manifest["network"]["helper"] == record(work / "private-network-helper.py")
    assert manifest["provider_cmake"] == record(work / "private-network.cmake")
    assert manifest["compiler_budget"] == {
        "maximum_jobs": 4, "outer_ninja_jobs": 4, "nested_cmake_jobs": 1, "nested_make_jobs": 1,
    }


@pytest.mark.parametrize("name", tuple(EXPECTED_SDK_COHORT))
def test_each_sdk_archive_is_a_required_explicit_cli_input(recipe, cohort_cli, name, capsys):
    state = cohort_cli
    argv = list(state["argv"])
    option = "--" + name.replace("_", "-") + "-archive"
    offset = argv.index(option)
    del argv[offset:offset + 2]
    with pytest.raises(SystemExit) as error:
        recipe.main(argv + ["--check"])
    assert error.value.code == 2
    assert option in capsys.readouterr().err
    assert not state["work"].exists()
    assert state["stages"] == []


@pytest.mark.parametrize("name", tuple(EXPECTED_SDK_COHORT))
def test_sdk_archive_checksum_is_checked_before_any_generation(recipe, cohort_cli, name, capsys):
    state = cohort_cli
    state["archives"][name].write_bytes(b"unreviewed source")
    with pytest.raises(SystemExit) as error:
        recipe.main(state["argv"] + ["--check"])
    assert error.value.code == 2
    assert name + " archive checksum mismatch" in capsys.readouterr().err
    assert not state["work"].exists()
    assert state["stages"] == []


@pytest.mark.parametrize("name", tuple(EXPECTED_SDK_COHORT))
def test_every_captured_sdk_archive_is_immutable_across_configure(recipe, cohort_cli, name, capsys):
    state = cohort_cli
    state["mutate_copy"] = name
    assert recipe.main(state["argv"] + ["--configure-only"]) != 0
    assert "immutable build input checksum mismatch" in capsys.readouterr().err
    assert len(state["stages"]) == 1
    receipt = json.loads((state["work"] / "build-receipt.json").read_bytes())
    assert receipt["status"] == "failed"
    assert not (state["work"] / "configured-inputs.json").exists()


@pytest.mark.parametrize("package", [
    "AWSSDK", "gRPC", "Protobuf", "absl", "c-ares", "Thrift", "jemalloc", "RapidJSON",
])
def test_reviewed_sdk_sources_cannot_resolve_to_host_providers(recipe, tmp_path, package):
    commands = recipe.build_commands(tmp_path / "source", tmp_path / "work", tmp_path / "openssl",
        network_prefix=tmp_path / "network", python="/python", cc="/cc", cxx="/cxx", jobs=4)
    assert f"-D{package}_SOURCE=BUNDLED" in commands["cpp_configure"]
    cache, entries, context = cache_fixture(recipe, tmp_path)
    recipe.verify_cache(cache, context, stage="cpp")
    entries[package + "_SOURCE"] = "SYSTEM"
    cache.write_text("\n".join(f"{key}:STRING={value}" for key, value in entries.items()) + "\n")
    with pytest.raises(ValueError, match=package + "_SOURCE"):
        recipe.verify_cache(cache, context, stage="cpp")


RAPIDJSON_LEGACY_TARGET = '''  add_library(RapidJSON INTERFACE IMPORTED)
  target_include_directories(RapidJSON INTERFACE "${rapidjson_SOURCE_DIR}/include")
  add_dependencies(RapidJSON rapidjson)'''

# Exact Arrow26 wrapper shape, independent of the recipe transformation table.
RAPIDJSON_WRAPPER = '''function(build_rapidjson)
  list(APPEND CMAKE_MESSAGE_INDENT "RapidJSON: ")
  message(STATUS "Building from source")

  fetchcontent_declare(rapidjson
                       ${FC_DECLARE_COMMON_OPTIONS} OVERRIDE_FIND_PACKAGE
                       URL ${RAPIDJSON_SOURCE_URL}
                       URL_HASH "SHA256=${ARROW_RAPIDJSON_BUILD_SHA256_CHECKSUM}")
  prepare_fetchcontent()
  set(CCACHE_FOUND OFF)
  set(LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib")
  set(RAPIDJSON_BUILD_DOC OFF)
  set(RAPIDJSON_BUILD_EXAMPLES OFF)
  set(RAPIDJSON_BUILD_TESTS OFF)
  fetchcontent_makeavailable(rapidjson)
  if(CMAKE_VERSION VERSION_LESS 3.28)
    set_property(DIRECTORY ${rapidjson_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL TRUE)
  endif()

''' + RAPIDJSON_LEGACY_TARGET + '''

  set(RAPIDJSON_VENDORED
      TRUE
      PARENT_SCOPE)
  list(POP_BACK CMAKE_MESSAGE_INDENT)
endfunction()
'''


@pytest.fixture
def rapidjson_source_inputs(recipe, source_inputs, monkeypatch):
    archive, work, original = source_inputs
    files = dict(original)
    key = "cpp/cmake_modules/ThirdpartyToolchain.cmake"
    # source_inputs includes each registered transform anchor. Replace the
    # independent old RapidJSON fragment once if a future recipe adds it.
    toolchain = files[key].decode().replace(RAPIDJSON_LEGACY_TARGET + "\n", "")
    files[key] = (toolchain + "\n" + RAPIDJSON_WRAPPER).encode()
    archive_at(archive, f"arrow-{recipe.ARROW_COMMIT}", files,
               {"python/NOTICE.txt": "../NOTICE.txt"})
    monkeypatch.setattr(recipe, "ARROW_ARCHIVE_SHA256", digest(archive.read_bytes()))
    return archive, work, files


@pytest.mark.parametrize("upstream_target", [True, False], ids=["upstream-export", "legacy-fallback"])
def test_rapidjson_transformed_wrapper_preserves_target_and_effective_source_includes(
        recipe, rapidjson_source_inputs, tmp_path, upstream_target):
    cmake = shutil.which("cmake")
    assert cmake, "CMake is required for this compiler-free target configuration test"
    archive, work, original = rapidjson_source_inputs
    source, evidence = recipe.prepare_source(archive, work, resume=False)
    key = "cpp/cmake_modules/ThirdpartyToolchain.cmake"
    toolchain = (source / key).read_text()
    wrapper = "function(build_rapidjson)\n" + toolchain.split(
        "function(build_rapidjson)\n", 1)[1].split("endfunction()", 1)[0] + "endfunction()\n"
    cmake_source = tmp_path / "cmake-source"
    vendor = cmake_source / "rapidjson-source"
    (vendor / "include/rapidjson").mkdir(parents=True)
    # This is the exact target/export behavior of pinned RapidJSON24b5. It has
    # no BUILD_INTERFACE include of its own; directory includes are not an
    # interface for Arrow's parent-directory consumers.
    (vendor / "CMakeLists.txt").write_text('''add_library(RapidJSON INTERFACE)
target_include_directories(RapidJSON INTERFACE $<INSTALL_INTERFACE:include>)
set_property(TARGET RapidJSON PROPERTY BIOMINER_UPSTREAM_MARKER "preserved")
install(TARGETS RapidJSON EXPORT RapidJSON-targets)
install(EXPORT RapidJSON-targets DESTINATION lib/cmake/RapidJSON)
''')
    (cmake_source / "wrapper.cmake").write_text(wrapper)
    (cmake_source / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.16)
project(rapidjson_target_compatibility LANGUAGES NONE)
set(rapidjson_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rapidjson-source")
function(fetchcontent_declare)
endfunction()
function(prepare_fetchcontent)
endfunction()
function(fetchcontent_makeavailable name)
  add_custom_target(rapidjson)
  if(UPSTREAM_TARGET)
    add_subdirectory("${rapidjson_SOURCE_DIR}")
  endif()
endfunction()
add_library(unrelated INTERFACE)
set_property(TARGET unrelated PROPERTY BIOMINER_UNRELATED_MARKER "unchanged")
include(wrapper.cmake)
build_rapidjson()
if(NOT RAPIDJSON_VENDORED)
  message(FATAL_ERROR "lost bundled selection")
endif()
get_target_property(kind RapidJSON TYPE)
get_target_property(imported RapidJSON IMPORTED)
get_target_property(marker RapidJSON BIOMINER_UPSTREAM_MARKER)
get_target_property(includes RapidJSON INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(dependencies RapidJSON MANUALLY_ADDED_DEPENDENCIES)
get_target_property(unrelated_marker unrelated BIOMINER_UNRELATED_MARKER)
file(WRITE "${CMAKE_BINARY_DIR}/target-state.txt"
  "${kind}\\n${imported}\\n${marker}\\n${includes}\\n${dependencies}\\n${unrelated_marker}\\n")
add_library(arrow_consumer INTERFACE)
target_link_libraries(arrow_consumer INTERFACE RapidJSON)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/consumer-includes.txt"
  CONTENT "$<TARGET_PROPERTY:arrow_consumer,INTERFACE_INCLUDE_DIRECTORIES>")
''')
    build = tmp_path / "cmake-build"
    completed = subprocess.run([cmake, "-S", str(cmake_source), "-B", str(build),
                                "-DUPSTREAM_TARGET=" + ("ON" if upstream_target else "OFF")],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    kind, imported, marker, includes, dependencies, unrelated = (
        build / "target-state.txt").read_text().splitlines()
    assert kind == "INTERFACE_LIBRARY"
    assert imported == ("FALSE" if upstream_target else "TRUE")
    assert marker == ("preserved" if upstream_target else "marker-NOTFOUND")
    if upstream_target:
        assert "$<INSTALL_INTERFACE:include>" in includes.split(";")
    assert dependencies.split(";") == ["rapidjson"]
    assert unrelated == "unchanged"
    effective = [item for item in (build / "consumer-includes.txt").read_text().split(";") if item]
    assert effective == [str(vendor / "include")]
    if upstream_target:
        exports = list(build.rglob("RapidJSON-targets.cmake"))
        assert len(exports) == 1
        assert str(vendor) not in exports[0].read_text()
    assert source.is_relative_to(work)
    assert (source / "NOTICE.txt").read_bytes() == original["NOTICE.txt"]
    assert evidence["integrated_sources"] == [
        {"relative": name, "sha256": value} for name, value in sorted(recipe.INTEGRATED_SOURCES.items())]
    assert recipe.prepare_source(archive, work, resume=True) == (source, evidence)


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_rapidjson_transform_requires_exact_unique_target_fragment(
        recipe, rapidjson_source_inputs, monkeypatch, change):
    archive, work, original = rapidjson_source_inputs
    files = dict(original)
    key = "cpp/cmake_modules/ThirdpartyToolchain.cmake"
    replacement = ("  # reviewed target fragment absent" if change == "missing"
                   else RAPIDJSON_LEGACY_TARGET + "\n" + RAPIDJSON_LEGACY_TARGET)
    files[key] = files[key].replace(RAPIDJSON_LEGACY_TARGET.encode(), replacement.encode())
    archive_at(archive, f"arrow-{recipe.ARROW_COMMIT}", files,
               {"python/NOTICE.txt": "../NOTICE.txt"})
    monkeypatch.setattr(recipe, "ARROW_ARCHIVE_SHA256", digest(archive.read_bytes()))
    with pytest.raises(ValueError, match="anchor"):
        recipe.prepare_source(archive, work, resume=False)
    assert not work.exists()


# Independent exact Arrow 26 anchor; the SDK's raw-archive link stays intact.
AZURE_FETCH_BLOCK = (
    '  fetchcontent_makeavailable(azure_sdk)\n'
    '  if(CMAKE_VERSION VERSION_LESS 3.28)'
)


def azure_xml_transform_fixture(recipe, source_inputs, monkeypatch, *, fetch_block=AZURE_FETCH_BLOCK):
    archive, work, original = source_inputs
    files = dict(original)
    key = "cpp/cmake_modules/ThirdpartyToolchain.cmake"
    # source_inputs contains each registered anchor outside any function.
    # Replace that synthetic copy with an independent upstream-shaped wrapper.
    toolchain = files[key].replace(AZURE_FETCH_BLOCK.encode(), b"")
    wrapper = (
        'function(build_azure_sdk)\n'
        + fetch_block + '\n'
        '    set_property(DIRECTORY ${azure_sdk_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL TRUE)\n'
        '  endif()\n'
        '  set(AZURE_SDK_VENDORED TRUE PARENT_SCOPE)\n'
        'endfunction()\n'
    )
    files[key] = toolchain + b"\n" + wrapper.encode()
    archive_at(archive, f"arrow-{recipe.ARROW_COMMIT}", files,
               {"python/NOTICE.txt": "../NOTICE.txt"})
    monkeypatch.setattr(recipe, "ARROW_ARCHIVE_SHA256", digest(archive.read_bytes()))
    source, _ = recipe.prepare_source(archive, work, resume=False)
    transformed = (source / key).read_text()
    transformed_wrapper = "function(build_azure_sdk)\n" + transformed.split(
        "function(build_azure_sdk)\n", 1)[1].split("\nendfunction()", 1)[0] + "\nendfunction()\n"
    return wrapper, transformed_wrapper


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
    common_source = '# Azure-like upstream raw-variable dependency stays unchanged.\n' + target
    common_cmake = actual / "CMakeLists.txt"
    common_cmake.write_text(common_source)
    (harness / "CMakeLists.txt").write_text(
        'cmake_minimum_required(VERSION 3.29)\n'
        'project(azure_private_xml LANGUAGES C)\n'
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
        recipe, source_inputs, monkeypatch, tmp_path):
    original, transformed = azure_xml_transform_fixture(recipe, source_inputs, monkeypatch)
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
        recipe, source_inputs, monkeypatch, tmp_path, variant):
    _, transformed = azure_xml_transform_fixture(recipe, source_inputs, monkeypatch)
    configured, _ = azure_xml_cmake_graph(tmp_path, transformed, target_variant=variant)
    assert configured.returncode != 0, "unproven Azure target was accepted"
    assert "private Azure XML dependency" in configured.stdout + configured.stderr


def test_azure_private_xml_interface_leaves_windows_dependency_unchanged(
        recipe, source_inputs, monkeypatch, tmp_path):
    _, transformed = azure_xml_transform_fixture(recipe, source_inputs, monkeypatch)
    configured, build = azure_xml_cmake_graph(tmp_path, transformed, windows=True)
    assert configured.returncode == 0, configured.stdout + configured.stderr
    assert "LibXml2::LibXml2" not in (build / "azure-links.txt").read_text().split(";")


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_azure_private_xml_interface_requires_exact_unique_source_anchor(
        recipe, source_inputs, monkeypatch, mutation):
    block = "" if mutation == "missing" else AZURE_FETCH_BLOCK + "\n" + AZURE_FETCH_BLOCK
    with pytest.raises(ValueError, match="anchor|transformation"):
        azure_xml_transform_fixture(recipe, source_inputs, monkeypatch, fetch_block=block)
    assert not source_inputs[1].exists()
