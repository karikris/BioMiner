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


def test_cli_bad_configure_approval_preserves_incomplete_receipt_and_checkpoint(
        recipe, source_inputs, builder_inputs, network_inputs, tmp_path, monkeypatch, capsys):
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
        cache.write_text("\n".join(f"{key}:STRING={value}" for key, value in
                                   recipe.expected_cache(context, stage="cpp").items()) + "\n")
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
        argv.extend([f"--{name}-archive", str(dependency_archive)])
    assert recipe.main(argv + ["--configure-only"]) == 0
    manifest = json.loads((work / "build-inputs.json").read_bytes())
    assert manifest["source_policy"] == "arrow26-integrated-private-network-v2"
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
