"""Arrow 26 build guards use isolated fixtures; no compiler or network is needed."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
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
def source_inputs(tmp_path, monkeypatch, recipe):
    toolchain = "\n".join(before for before, _ in recipe.TOOLCHAIN_PATCHES)
    toolchain += "\n" + recipe.ZLIB_CMAKE_ANCHOR + "\n" + recipe.ZLIB_PROJECT_ANCHOR
    toolchain += "\n" + recipe.BOOST_SELECTION_ANCHOR + "\n"
    versions = "\n".join(before for before, _ in recipe.VERSION_PATCHES) + "\n"
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
    assert "ZLIB_BUILD_TESTING=OFF" in text
    assert "ZLIB_BUILD_SHARED=OFF" in text
    assert "ZLIB_BUILD_STATIC=ON" in text
    assert "ZLIB_INSTALL=ON" in text
    assert "PATCH_COMMAND" in text
    versions = (source / "cpp/thirdparty/versions.txt").read_text()
    assert recipe.DEPENDENCIES["simdjson"]["sha256"] in versions
    assert recipe.DEPENDENCIES["zlib"]["sha256"] in versions
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
    commands = recipe.build_commands(tmp_path / "source", tmp_path / "work", tmp_path / "openssl",
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
    for name in ("simdjson", "uriparser", "Boost", "ZLIB"):
        assert f"-D{name}_SOURCE=BUNDLED" in configure
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


@pytest.fixture
def zlib_inputs(tmp_path, monkeypatch, recipe):
    import difflib

    original = b"before\nold write\nafter\n"
    patched = b"before\nsafe write\nafter\n"
    archive = archive_at(tmp_path / "zlib.tar.gz", "zlib-1.3.2",
                         {"gzwrite.c": original, "LICENSE": b"retain"})
    patch = tmp_path / "zlib.patch"
    patch.write_text("".join(difflib.unified_diff(original.decode().splitlines(True),
                                                patched.decode().splitlines(True),
                                                fromfile="a/gzwrite.c", tofile="b/gzwrite.c")))
    monkeypatch.setitem(recipe.DEPENDENCIES["zlib"], "sha256", digest(archive.read_bytes()))
    monkeypatch.setattr(recipe, "ZLIB_PATCH_SHA256", digest(patch.read_bytes()))
    monkeypatch.setattr(recipe, "ZLIB_ORIGINAL_GZWRITE_SHA256", digest(original))
    monkeypatch.setattr(recipe, "ZLIB_PATCHED_GZWRITE_SHA256", digest(patched))
    with tarfile.open(archive) as bundle:
        bundle.extractall(tmp_path / "extracted", filter="data")
    return tmp_path / "extracted/zlib-1.3.2", archive, patch, tmp_path / "zlib-application.json", patched


def test_zlib_guard_applies_exact_patch_and_retains_application(recipe, zlib_inputs):
    source, archive, patch, receipt, expected = zlib_inputs
    result = recipe.apply_zlib_patch(source, archive, patch, receipt)
    assert (source / "gzwrite.c").read_bytes() == expected
    assert result["patch_commit"] == recipe.ZLIB_PATCH_COMMIT
    assert result["patch"] == record(patch)
    assert result["archive"] == record(archive)
    assert recipe.apply_zlib_patch(source, archive, patch, receipt) == result


@pytest.mark.parametrize("change", ["archive", "patch", "source", "extra", "receipt"])
def test_zlib_guard_rejects_tampered_source_chain(recipe, zlib_inputs, change):
    source, archive, patch, receipt, _ = zlib_inputs
    recipe.apply_zlib_patch(source, archive, patch, receipt)
    target = {"archive": archive, "patch": patch, "source": source / "gzwrite.c",
              "extra": source / "unexpected.c", "receipt": receipt}[change]
    target.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="zlib|source|archive|patch|receipt"):
        recipe.apply_zlib_patch(source, archive, patch, receipt)


def cache_fixture(recipe, tmp_path):
    expected = {"python": "/approved/python", "cc": "/approved/cc", "cxx": "/approved/cxx",
                "openssl_prefix": str(tmp_path / "openssl"), "work": str(tmp_path / "work")}
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


@pytest.mark.parametrize("missing", ["libcrypto.a", "libssl.a", "libz.a"])
def test_link_maps_require_actual_static_crypto_and_patched_zlib(recipe, tmp_path, missing):
    prefix = tmp_path / "openssl"
    (prefix / "lib").mkdir(parents=True)
    static = []
    for name in ("libcrypto.a", "libssl.a", "libz.a"):
        path = prefix / "lib" / name
        path.write_bytes(name.encode())
        static.append(record(path))
    linked = [row["path"] for row in static if not row["path"].endswith(missing)]
    (tmp_path / "candidate.map").write_text("Linker script and memory map\n" +
                                            "\n".join(path + "(member.o)" for path in linked))
    with pytest.raises(ValueError, match=missing):
        recipe.verify_link_maps(tmp_path, prefix, {"static_archives": static})


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


def test_cli_bad_configure_approval_preserves_incomplete_receipt_and_checkpoint(
        recipe, source_inputs, builder_inputs, tmp_path, monkeypatch, capsys):
    """Exercise actual orchestration; only external builder/configure work is fake."""
    archive, work, _ = source_inputs
    builder_receipt, builder_hash, builder, _ = builder_inputs
    prefix = tmp_path / "openssl"
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
        context = {"work": str(work), "openssl_prefix": str(prefix), "python": builder["python"]["path"],
                   "cc": builder["toolchain"]["cc"]["path"], "cxx": builder["toolchain"]["cxx"]["path"]}
        cache = work / "cpp-build/CMakeCache.txt"
        cache.parent.mkdir(exist_ok=True)
        cache.write_text("\n".join(f"{key}:STRING={value}" for key, value in
                                   recipe.expected_cache(context, stage="cpp").items()) + "\n")
        return {"command": command, "exit_code": 0}

    monkeypatch.setattr(recipe, "run_stage", configure_fixture)
    argv = ["--archive", str(archive), "--work", str(work), "--openssl-prefix", str(prefix),
            "--builder-receipt", str(builder_receipt), "--builder-receipt-sha256", builder_hash,
            "--python", builder["python"]["path"], "--cc", builder["toolchain"]["cc"]["path"],
            "--cxx", builder["toolchain"]["cxx"]["path"]]
    for name in recipe.DEPENDENCIES:
        dependency_archive = tmp_path / f"{name}.tar.gz"
        dependency_archive.write_bytes(f"approved {name} fixture".encode())
        monkeypatch.setitem(recipe.DEPENDENCIES[name], "sha256", digest(dependency_archive.read_bytes()))
        argv.extend([f"--{name}-archive", str(dependency_archive)])
    patch = tmp_path / "approved-zlib.patch"
    patch.write_bytes(b"approved patch fixture")
    monkeypatch.setattr(recipe, "ZLIB_PATCH_SHA256", digest(patch.read_bytes()))
    argv.extend(["--zlib-patch", str(patch)])
    assert recipe.main(argv + ["--configure-only"]) == 0
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
