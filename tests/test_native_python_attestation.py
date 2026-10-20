"""Interpreter artifacts require the official immutable archive digest."""
import importlib.util
from pathlib import Path

import pytest
import io
import tarfile


def test_python_attestation_rejects_modified_archive(tmp_path):
    script = Path(__file__).parents[1] / 'scripts/attest_native_python.py'
    assert script.exists(), 'interpreter attestation missing'
    spec = importlib.util.spec_from_file_location('native_python', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = tmp_path / 'python.tar.gz'
    archive.write_bytes(b'changed runtime')
    with pytest.raises(ValueError, match='archive checksum'):
        module.verify_archive(archive)


def test_python_attestation_rejects_extra_importable_file(tmp_path):
    script = Path(__file__).parents[1] / 'scripts/attest_native_python.py'
    spec = importlib.util.spec_from_file_location('native_python', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = tmp_path / 'source.tar'
    with tarfile.open(archive, 'w') as bundle:
        member = tarfile.TarInfo('python/approved.py')
        member.size = 4
        bundle.addfile(member, io.BytesIO(b'pass'))
    prefix = tmp_path / 'python'
    prefix.mkdir()
    (prefix / 'approved.py').write_bytes(b'pass')
    (prefix / 'sitecustomize.py').write_bytes(b'pass')
    with pytest.raises(ValueError, match='unexpected runtime file'):
        module.verify_prefix(archive, prefix)


def test_python_attestation_rejects_extra_directory_symlink(tmp_path):
    script = Path(__file__).parents[1] / 'scripts/attest_native_python.py'
    spec = importlib.util.spec_from_file_location('native_python_symlink', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = tmp_path / 'source.tar'
    with tarfile.open(archive, 'w') as bundle:
        member = tarfile.TarInfo('python/approved.py')
        member.size = 4
        bundle.addfile(member, io.BytesIO(b'pass'))
    prefix = tmp_path / 'python'
    prefix.mkdir()
    (prefix / 'approved.py').write_bytes(b'pass')
    external = tmp_path / 'external'
    external.mkdir()
    (prefix / 'sitecustomize').symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match='unexpected runtime file'):
        module.verify_prefix(archive, prefix)


def test_python_attestation_rejects_replaced_regular_file_symlink(tmp_path):
    script = Path(__file__).parents[1] / 'scripts/attest_native_python.py'
    spec = importlib.util.spec_from_file_location('native_python_file_symlink', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = tmp_path / 'source.tar'
    with tarfile.open(archive, 'w') as bundle:
        member = tarfile.TarInfo('python/approved.py')
        member.size = 4
        bundle.addfile(member, io.BytesIO(b'pass'))
    prefix = tmp_path / 'python'
    prefix.mkdir()
    external = tmp_path / 'external.py'
    external.write_bytes(b'pass')
    (prefix / 'approved.py').symlink_to(external)
    with pytest.raises(ValueError, match='replaced by symlink'):
        module.verify_prefix(archive, prefix)
