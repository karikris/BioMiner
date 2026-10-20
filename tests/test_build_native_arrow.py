"""The Arrow build must reject unapproved crypto before invoking a compiler."""
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def arrow_recipe():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "build_native_arrow.py"
    spec = importlib.util.spec_from_file_location("arrow_recipe_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source_inputs(tmp_path, monkeypatch, arrow_recipe):
    """Small source archive exercises real extraction, patching and resume."""
    import difflib
    import hashlib
    import io
    import tarfile

    relative_files = (
        "cpp/src/arrow/compute/kernels/vector_replace.cc",
        "cpp/src/arrow/compute/kernels/vector_replace_test.cc",
    )
    originals = {name: b"before\nold allocation\nafter\n" for name in relative_files}
    patched = {name: b"before\nsafe allocation\nafter\n" for name in relative_files}
    patch = tmp_path / "reviewed.patch"
    patch.write_text("".join(
        "".join(difflib.unified_diff(
            originals[name].decode().splitlines(keepends=True),
            patched[name].decode().splitlines(keepends=True),
            fromfile="a/" + name, tofile="b/" + name,
        ))
        for name in relative_files
    ))
    sha = lambda data: hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(arrow_recipe, "SECURITY_PATCH_SHA256", sha(patch.read_bytes()), raising=False)
    monkeypatch.setattr(arrow_recipe, "SECURITY_PATCH_FILES", {
        name: {"original_sha256": sha(originals[name]), "patched_sha256": sha(patched[name])}
        for name in relative_files
    }, raising=False)
    originals["cpp/cmake_modules/ThirdpartyToolchain.cmake"] = (
        b"list(APPEND AWSSDK_PRODUCTS aws-lc s2n-tls)\n"
        b"  fetchcontent_makeavailable(cares)\n"
        b"  fetchcontent_makeavailable(grpc)\n"
        b'set(MAKE_BUILD_ARGS "-j${NPROC}")\n'
    )
    originals["NOTICE.txt"] = b"Preserve upstream notices.\n"
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, data in originals.items():
            entry = tarfile.TarInfo(f"arrow-{arrow_recipe.ARROW_COMMIT}/{name}")
            entry.size = len(data)
            bundle.addfile(entry, io.BytesIO(data))
        link = tarfile.TarInfo(f"arrow-{arrow_recipe.ARROW_COMMIT}/python/NOTICE.txt")
        link.type = tarfile.SYMTYPE
        link.linkname = "../NOTICE.txt"
        bundle.addfile(link)
    monkeypatch.setattr(arrow_recipe, "ARROW_ARCHIVE_SHA256", sha(archive.read_bytes()))
    return archive, patch, tmp_path / "work", originals, patched


def test_arrow_source_preparation_applies_reviewed_patch(arrow_recipe, source_inputs):
    archive, patch, work, originals, expected = source_inputs
    source, toolchain, security = arrow_recipe.prepare_source(archive, work, patch, resume=False)

    for name, data in expected.items():
        assert (source / name).read_bytes() == data
    assert (source / "NOTICE.txt").read_bytes() == originals["NOTICE.txt"]
    assert (source / "python/NOTICE.txt").is_symlink()
    assert (source / "python/NOTICE.txt").read_bytes() == originals["NOTICE.txt"]
    assert security["patches"][0]["patch"]["sha256"] == arrow_recipe.sha256(patch)
    assert security["patches"][0]["commit"] == arrow_recipe.SECURITY_PATCH_COMMIT
    assert {row["relative"] for row in security["patches"][0]["files"]} == set(expected)
    assert toolchain["original_sha256"] != toolchain["patched_sha256"]
    assert 'set(MAKE_BUILD_ARGS "-j1")' in (
        source / "cpp/cmake_modules/ThirdpartyToolchain.cmake"
    ).read_text()


def test_arrow_source_preparation_rejects_changed_patch_before_extracting(arrow_recipe, source_inputs):
    archive, patch, work, _, _ = source_inputs
    patch.write_bytes(patch.read_bytes() + b"\nchanged\n")

    with pytest.raises(ValueError, match="security patch checksum"):
        arrow_recipe.prepare_source(archive, work, patch, resume=False)
    assert not work.exists()


def test_arrow_source_resume_accepts_complete_reviewed_state(arrow_recipe, source_inputs):
    archive, patch, work, _, _ = source_inputs
    first = arrow_recipe.prepare_source(archive, work, patch, resume=False)

    second = arrow_recipe.prepare_source(archive, work, patch, resume=True)

    assert first == second


@pytest.mark.parametrize("changed", [
    "cpp/src/arrow/compute/kernels/vector_replace.cc",
    "cpp/src/arrow/compute/kernels/vector_replace_test.cc",
    "cpp/cmake_modules/ThirdpartyToolchain.cmake",
    "NOTICE.txt",
])
def test_arrow_source_resume_rejects_partial_or_changed_source(arrow_recipe, source_inputs, changed):
    archive, patch, work, originals, _ = source_inputs
    source, _, _ = arrow_recipe.prepare_source(archive, work, patch, resume=False)
    (source / changed).write_bytes(originals[changed] + (b"changed" if changed == "NOTICE.txt" else b""))

    with pytest.raises(ValueError, match="source changed since extraction"):
        arrow_recipe.prepare_source(archive, work, patch, resume=True)


def test_arrow_source_preparation_preserves_existing_generation(arrow_recipe, source_inputs):
    archive, patch, work, _, _ = source_inputs
    source, _, _ = arrow_recipe.prepare_source(archive, work, patch, resume=False)
    before = (source / "NOTICE.txt").read_bytes()

    with pytest.raises(ValueError, match="already exists"):
        arrow_recipe.prepare_source(archive, work, patch, resume=False)
    assert (source / "NOTICE.txt").read_bytes() == before


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_arrow_source_resume_rejects_additional_paths(arrow_recipe, source_inputs, kind):
    archive, patch, work, _, _ = source_inputs
    source, _, _ = arrow_recipe.prepare_source(archive, work, patch, resume=False)
    extra = source / "unexpected-cmake-input"
    if kind == "file":
        extra.write_text("unreviewed source")
    elif kind == "directory":
        extra.mkdir()
    else:
        extra.symlink_to("NOTICE.txt")

    with pytest.raises(ValueError, match="source layout changed"):
        arrow_recipe.prepare_source(archive, work, patch, resume=True)
    assert extra.exists()


def test_arrow_source_preparation_preserves_nonempty_work(arrow_recipe, source_inputs):
    archive, patch, work, _, _ = source_inputs
    work.mkdir()
    receipt = work / "build-receipt.json"
    receipt.write_text("previous receipt")
    log = work / "configure.log"
    log.write_text("previous log")

    with pytest.raises(ValueError, match="work directory already exists"):
        arrow_recipe.prepare_source(archive, work, patch, resume=False)
    assert receipt.read_text() == "previous receipt"
    assert log.read_text() == "previous log"
    assert set(work.iterdir()) == {receipt, log}


def test_arrow_source_preparation_rejects_work_alias(arrow_recipe, source_inputs, tmp_path):
    archive, patch, work, _, _ = source_inputs
    work.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(work, target_is_directory=True)

    with pytest.raises(ValueError, match="work path contains a symlink"):
        arrow_recipe.prepare_source(archive, alias, patch, resume=False)
    assert not list(work.iterdir())


def test_arrow_source_resume_rejects_changed_archive_symlink(arrow_recipe, source_inputs):
    archive, patch, work, _, _ = source_inputs
    source, _, _ = arrow_recipe.prepare_source(archive, work, patch, resume=False)
    link = source / "python/NOTICE.txt"
    link.unlink()
    link.symlink_to("../cpp")

    with pytest.raises(ValueError, match="source layout changed"):
        arrow_recipe.prepare_source(archive, work, patch, resume=True)


@pytest.mark.parametrize("completed", ["receipt", "wheel"])
def test_arrow_source_resume_preserves_completed_outputs(arrow_recipe, source_inputs, completed):
    archive, patch, work, _, _ = source_inputs
    arrow_recipe.prepare_source(archive, work, patch, resume=False)
    log = work / "configure.log"
    log.write_text("previous log")
    if completed == "receipt":
        artifact = work / "build-receipt.json"
        artifact.write_text('{"status":"compiled_unverified"}\n')
    else:
        artifact = work / "wheelhouse" / "pyarrow.whl"
        artifact.parent.mkdir()
        artifact.write_bytes(b"previous wheel")
    before = artifact.read_bytes()

    with pytest.raises(ValueError, match="resume cannot modify completed"):
        arrow_recipe.prepare_source(archive, work, patch, resume=True)
    assert artifact.read_bytes() == before
    assert log.read_text() == "previous log"


@pytest.mark.parametrize("name", [
    "build-receipt.json", "cpp-build", "arrow-prefix", "wheelhouse",
    "wheel-inventory", "configure.log", "cpp-build.log", "python-build.log",
])
def test_arrow_source_resume_rejects_output_symlinks(arrow_recipe, source_inputs, tmp_path, name):
    archive, patch, work, _, _ = source_inputs
    arrow_recipe.prepare_source(archive, work, patch, resume=False)
    outside = tmp_path / "unowned-missing-target"
    (work / name).symlink_to(outside)

    with pytest.raises(ValueError, match="resume output path is a symlink"):
        arrow_recipe.prepare_source(archive, work, patch, resume=True)
    assert not outside.exists()
    assert (work / name).is_symlink()


def test_arrow_build_budget_serializes_nested_builds(arrow_recipe):
    environment = arrow_recipe.build_environment(4)

    assert environment["CMAKE_BUILD_PARALLEL_LEVEL"] == "1"
    assert environment["MAKEFLAGS"] == "-j1"


@pytest.mark.parametrize("jobs", [0, 5, 8])
def test_arrow_builder_rejects_more_than_four_jobs(tmp_path, jobs):
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_native_arrow.py"
    result = subprocess.run(
        [sys.executable, str(script), "--openssl-prefix", str(tmp_path), "--check",
         "--jobs", str(jobs)],
        capture_output=True, text=True,
    )

    assert result.returncode == 2
    assert "--jobs must be between 1 and 4" in result.stderr


def test_arrow_recipe_copy_cannot_change_on_resume(tmp_path, arrow_recipe):
    work = tmp_path / "work"
    work.mkdir()
    expected = arrow_recipe.preserve_recipe(work, b"original recipe", resume=False)

    with pytest.raises(ValueError, match="recipe changed"):
        arrow_recipe.preserve_recipe(work, b"changed recipe", resume=True)
    assert Path(expected["path"]).read_bytes() == b"original recipe"


def test_arrow_input_manifest_binds_wheel_version(tmp_path, arrow_recipe):
    inputs = {"recipe": {"sha256": "a" * 64}, "security_patches": {"patches": []}, "jobs": 4}

    record, version = arrow_recipe.preserve_build_inputs(tmp_path, inputs, resume=False)

    assert version == "25.0.1+biominer." + arrow_recipe.sha256(Path(record["path"]))[:12]
    assert record["sha256"] == arrow_recipe.sha256(Path(record["path"]))
    assert arrow_recipe.preserve_build_inputs(tmp_path, inputs, resume=True) == (record, version)


def test_arrow_resume_rejects_changed_build_inputs(tmp_path, arrow_recipe):
    arrow_recipe.preserve_build_inputs(tmp_path, {"jobs": 4}, resume=False)

    with pytest.raises(ValueError, match="build inputs changed"):
        arrow_recipe.preserve_build_inputs(tmp_path, {"jobs": 3}, resume=True)


def test_arrow_receipt_rejects_wheel_version_not_bound_to_inputs(tmp_path, arrow_recipe):
    import zipfile

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    with zipfile.ZipFile(wheelhouse / "pyarrow.whl", "w") as wheel:
        wheel.writestr("pyarrow-25.0.1.dist-info/METADATA", "Name: pyarrow\nVersion: 25.0.1\n")
        wheel.writestr("pyarrow/lib.so", b"\x7fELFnot a compiled library")

    with pytest.raises(RuntimeError, match="wheel version does not match"):
        arrow_recipe.completed_receipt(
            tmp_path, tmp_path / "source", tmp_path / "openssl", sys.executable,
            {"wheel_version": "25.0.1+biominer.123456789abc"},
        )


def test_arrow_builder_rejects_unpatched_openssl(tmp_path):
    prefix = tmp_path / "openssl"
    header = prefix / "include" / "openssl" / "opensslv.h"
    header.parent.mkdir(parents=True)
    header.write_text('#define OPENSSL_VERSION_TEXT "OpenSSL 3.6.0"\n')
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_native_arrow.py"
    result = subprocess.run(
        [sys.executable, str(script), "--openssl-prefix", str(prefix), "--check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "OpenSSL 3.6.5 required" in result.stderr


def test_arrow_builder_rejects_changed_source_archive(tmp_path):
    prefix = tmp_path / "openssl"
    header = prefix / "include" / "openssl" / "opensslv.h"
    header.parent.mkdir(parents=True)
    header.write_text('#define OPENSSL_VERSION_TEXT "OpenSSL 3.6.5 29 Sep 2026"\n')
    (prefix / "lib").mkdir()
    for name in ("libcrypto.a", "libssl.a"):
        (prefix / "lib" / name).write_bytes(b"fixture")
    archive = tmp_path / "arrow.tar.gz"
    archive.write_bytes(b"changed upstream source")
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_native_arrow.py"
    result = subprocess.run(
        [sys.executable, str(script), "--openssl-prefix", str(prefix),
         "--archive", str(archive), "--work", str(tmp_path / "generation")],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "source archive checksum mismatch" in result.stderr


def test_arrow_receipt_rejects_wheel_without_native_elf(tmp_path):
    import runpy
    import zipfile
    import pytest

    script = Path(__file__).resolve().parents[1] / "scripts" / "build_native_arrow.py"
    builder = runpy.run_path(str(script))
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    with zipfile.ZipFile(wheelhouse / "pyarrow.whl", "w") as wheel:
        wheel.writestr("pyarrow-25.0.1.dist-info/METADATA", "Name: pyarrow\nVersion: 25.0.1\n")
        wheel.writestr("pyarrow/__init__.py", "")
    with pytest.raises(RuntimeError, match="wheel contains no native ELF"):
        builder["completed_receipt"](tmp_path, tmp_path / "source.tar.gz", tmp_path / "openssl", sys.executable, {})
