"""Admission, output isolation, and evidence for the private XML library build."""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


@pytest.fixture
def builder():
    path = Path(__file__).parents[1] / "scripts/build_native_lxml.py"
    assert path.is_file(), "the reviewed private lxml build recipe is required"
    spec = importlib.util.spec_from_file_location("native_lxml_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_output_is_untouched_before_input_checks(builder, tmp_path):
    output = tmp_path / "previous"
    output.mkdir()
    sentinel = output / "receipt.json"
    sentinel.write_bytes(b"previous immutable receipt")
    with pytest.raises(FileExistsError):
        builder.build(tmp_path / "absent-python", tmp_path / "absent-inputs", output, 4)
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"previous immutable receipt"


@pytest.mark.parametrize("jobs", [0, 5, -1])
def test_compiler_budget_rejects_invalid_values(builder, tmp_path, jobs):
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="jobs"):
        builder.build(Path(sys.executable), tmp_path, output, jobs)
    assert not output.exists()


def test_changed_source_stops_before_output_creation(builder, tmp_path):
    source = next(iter(builder.SOURCES.values()))
    (tmp_path / source["filename"]).write_bytes(b"unapproved archive")
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="checksum"):
        builder.build(Path(sys.executable), tmp_path, output, 1)
    assert not output.exists()


def test_source_policy_includes_reviewed_zlib_fix_and_iconv_upgrade(builder):
    assert builder.SOURCES["libiconv"]["version"] == "1.19"
    assert builder.SOURCES["zlib_patch"]["sha256"] == (
        "110ff14375733173d8aa54574473424fbd7dfe4b81f1ca34a759c6fe14b15b14"
    )
    assert builder.ZLIB_PATCH_COMMIT == "df84af25dc1942490e1d1c899a07619152a46148"


def test_archive_cannot_write_outside_source_root(builder, tmp_path):
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as bundle:
        member = tarfile.TarInfo("../escaped")
        member.size = 3
        bundle.addfile(member, io.BytesIO(b"bad"))
    output = tmp_path / "source"
    output.mkdir()
    with pytest.raises((ValueError, tarfile.FilterError)):
        builder.extract_source(archive, output)
    assert not (tmp_path / "escaped").exists()


def valid_probe():
    return {
        "lxml": "6.1.3",
        "lxml_compiled": "6.1.3",
        "libxml2_runtime": [2, 15, 4],
        "libxml2_compiled": [2, 15, 4],
        "libxslt_runtime": [1, 1, 45],
        "libxslt_compiled": [1, 1, 45],
        "libiconv": "1.19",
        "zlib": "1.3.2",
        "libxml_features": [
            "catalog", "html", "iconv", "regexp", "schematron",
            "xmlschema", "xpath", "zlib",
        ],
        "libxml_compiled_features": [
            "catalog", "html", "iconv", "regexp", "schematron",
            "xmlschema", "xpath", "zlib",
        ],
        "checks": {
            "xml": True, "xpath": True, "xmlschema": True, "schematron": True,
            "xslt": True, "html": True, "objectify": True, "encoding": True,
            "gzip": True, "gzip_default_disabled": True,
            "external_parameter_entity_blocked": True,
        },
    }


def test_exact_native_versions_and_capabilities_are_admitted(builder):
    builder.validate_probe(valid_probe())


@pytest.mark.parametrize("field,value", [
    ("lxml_compiled", "6.1.2"),
    ("libxml2_runtime", [2, 14, 6]),
    ("libxml2_compiled", [2, 14, 6]),
    ("libxslt_runtime", [1, 1, 43]),
    ("libiconv", "1.18"),
    ("zlib", "1.3.1"),
    ("libxml_features", ["xmlschema", "xpath"]),
    ("libxml_compiled_features", ["xmlschema", "xpath"]),
    ("checks", {}),
])
def test_native_probe_rejects_missing_or_stale_evidence(builder, field, value):
    probe = valid_probe()
    probe[field] = value
    with pytest.raises(ValueError, match="probe"):
        builder.validate_probe(probe)


def test_linkage_rejects_external_xml_or_missing_libraries(builder):
    with pytest.raises(ValueError, match="private"):
        builder.validate_linkage("libxml2.so.16 => /usr/lib/libxml2.so.16 (0x1)")
    with pytest.raises(ValueError, match="unresolved"):
        builder.validate_linkage("libz.so.1 => not found")


def test_linker_retains_hardening_and_exposes_library_versions(builder):
    command = builder.linker_command(
        "gcc -shared -Wl,-z,relro -Wl,-z,now -Wl,--exclude-libs,ALL", "/usr/bin/cc"
    )
    assert command[0] == "/usr/bin/cc"
    assert "-Wl,-z,relro" in command
    assert "-Wl,-z,now" in command
    assert "-Wl,--exclude-libs,ALL" not in command
    assert "-Wl,--exclude-libs,libexslt.a:libxslt.a:libxml2.a" in command


def test_relink_rejects_unused_replacement_library(builder, tmp_path):
    replacement = tmp_path / "libiconv.a"
    replacement.write_bytes(b"replacement")
    (tmp_path / "relink-manifest.json").write_text(json.dumps({
        "original_root": "/original", "inputs": [],
        "commands": [{"argv": ["cc", "object.o", "-o", "extension.so"],
                      "cwd": "/original", "output": "extension.so"}],
    }))
    script = tmp_path / "relink.py"
    script.write_text(builder.RELINK_SCRIPT)
    output = tmp_path / "new-links"
    result = subprocess.run(
        [sys.executable, "-B", str(script), "--iconv-library", str(replacement),
         "--output", str(output)], capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "replacement library is not used" in result.stderr
    assert not output.exists()


def test_command_timeout_is_bounded_and_leaves_no_success(builder, tmp_path):
    log = tmp_path / "command.log"
    with pytest.raises(subprocess.TimeoutExpired):
        builder.run_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path, env={}, log=log, timeout=0.1,
        )
    assert log.is_file()


@pytest.mark.parametrize("relative_object", [False, True])
def test_relink_accepts_replacement_library_and_preserves_inputs(builder, tmp_path, relative_object):
    original = tmp_path / "build-root"
    bundle = tmp_path / "relocatable"
    (bundle / "prefix/lib").mkdir(parents=True)
    (bundle / "sources/lxml/objects").mkdir(parents=True)
    (bundle / "prefix/lib/libiconv.a").write_bytes(b"original iconv")
    (bundle / "sources/lxml/objects/extension.o").write_bytes(b"extension object")
    replacement = tmp_path / "replacement.a"
    replacement.write_bytes(b"replacement iconv")
    fake_linker = tmp_path / "linker.py"
    fake_linker.write_text(
        "import pathlib,sys\n"
        "args=sys.argv[1:]\n"
        "out=pathlib.Path(args[args.index('-o')+1])\n"
        "out.parent.mkdir(parents=True,exist_ok=True)\n"
        "out.write_bytes(b'|'.join(pathlib.Path(x).read_bytes() for x in args "
        "if x.endswith(('.a','.o'))))\n"
    )
    commands = [{
        "argv": [
            sys.executable, str(fake_linker),
            "objects/extension.o" if relative_object else str(original / "sources/lxml/objects/extension.o"),
            str(original / "prefix/lib/libiconv.a"),
            "-o", str(original / "built/extension.so"),
        ],
        "cwd": str(original / "sources/lxml"),
        "output": str(original / "built/extension.so"),
    }]
    manifest = {
        "original_root": str(original), "commands": commands,
        "inputs": [
            {"path": "sources/lxml/objects/extension.o", "sha256": builder.sha256(bundle / "sources/lxml/objects/extension.o")},
            {"path": "prefix/lib/libiconv.a", "sha256": builder.sha256(bundle / "prefix/lib/libiconv.a")},
        ],
    }
    (bundle / "relink-manifest.json").write_text(json.dumps(manifest))
    script = bundle / "relink.py"
    script.write_text(builder.RELINK_SCRIPT)
    output = tmp_path / "new-links"
    result = subprocess.run(
        [sys.executable, "-B", str(script), "--iconv-library", str(replacement),
         "--output", str(output)], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "extension.so").read_bytes() == b"extension object|replacement iconv"
    assert (bundle / "prefix/lib/libiconv.a").read_bytes() == b"original iconv"
