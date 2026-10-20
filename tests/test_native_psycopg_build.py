"""Source admission for the system-linked PostgreSQL adapter build."""
import hashlib
import importlib.util
from pathlib import Path

import pytest


def test_psycopg_build_rejects_changed_source(tmp_path):
    script = Path(__file__).parents[1] / 'scripts/build_native_psycopg.py'
    assert script.exists(), 'system-linked Psycopg builder is missing'
    spec = importlib.util.spec_from_file_location('native_psycopg', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / 'source.tar.gz'
    source.write_bytes(b'changed source')
    with pytest.raises(ValueError, match='checksum'):
        module.verify_source(source, hashlib.sha256(b'approved source').hexdigest())


def test_build_preserves_virtual_environment_interpreter(tmp_path):
    script = Path(__file__).parents[1] / 'scripts/build_native_psycopg.py'
    spec = importlib.util.spec_from_file_location('native_psycopg', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = tmp_path / 'python'
    original.symlink_to('/usr/bin/python3')
    assert module.interpreter_path(original) == str(original.absolute())


def test_linkage_inventory_includes_loader(tmp_path):
    script = Path(__file__).parents[1] / 'scripts/build_native_psycopg.py'
    spec = importlib.util.spec_from_file_location('native_psycopg', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.linkage_paths('libpq.so => /lib/libpq.so (0x1)\n/lib/ld-linux.so (0x2)') == ['/lib/libpq.so', '/lib/ld-linux.so']


def test_linkage_inventory_rejects_missing_library(tmp_path):
    script = Path(__file__).parents[1] / 'scripts/build_native_psycopg.py'
    spec = importlib.util.spec_from_file_location('native_psycopg', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match='unresolved'):
        module.linkage_paths('libpq.so => not found')
