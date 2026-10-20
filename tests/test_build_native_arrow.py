"""The Arrow build must reject unapproved crypto before invoking a compiler."""
import subprocess
import sys
from pathlib import Path


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
