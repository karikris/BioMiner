from __future__ import annotations

import hashlib
import json

import pytest

from scripts import refresh_native_host as refresh
runtime = refresh.runtime
from tests.test_native_runtime import profile  # shared valid artifact fixture


def setup_host(profile, monkeypatch):
    path, doc = profile
    # refresh loads its own hash-bound copy of native_runtime, while the shared
    # fixture replaces only the test patch digest in the normal imported copy.
    monkeypatch.setattr(runtime, 'LXML_ZLIB_PATCH_SHA256',
                        hashlib.sha256(b'reviewed patch fixture').hexdigest())
    lib = path.parent / 'libssl.so'
    lib.write_bytes(b'old package bytes')
    doc['ready'] = True
    doc['status'] = 'admitted'
    doc['native_files'] = [{'path': str(lib), 'sha256': runtime.sha256(lib), 'owner': 'dpkg:libssl3t64'}]
    path.write_text(json.dumps(doc))
    old = path.read_bytes()
    lib.write_bytes(b'new patched package bytes')
    def run(command, **kwargs):
        if command[1] == '-S':
            return f'libssl3t64:amd64: {lib}\n'
        if command[1] == '-W':
            return '3.5.5-1ubuntu3.7\tamd64\tinstall ok installed'
        return 'official installed-version Ubuntu policy'
    monkeypatch.setattr(runtime, '_run', run)
    monkeypatch.setattr(runtime, '_host', lambda doc: [{'name': 'libssl3t64', 'version': '3.5.5-1ubuntu3.7', 'architecture': 'amd64'}])
    return path, doc, lib, old


def test_refresh_preserves_build_evidence_and_requires_new_generation(profile, monkeypatch):
    path, doc, lib, old = setup_host(profile, monkeypatch)
    output = path.parent / 'refreshed.lock'
    result = refresh.refresh_host(path, output)
    new = json.loads(output.read_text())
    assert path.read_bytes() == old
    for key in ('artifacts', 'inputs', 'extension_policy', 'python', 'recipes', 'base_lock'):
        assert new[key] == doc[key]
    assert new['native_files'][0]['sha256'] == runtime.sha256(lib)
    assert new['host_packages'][0]['apt_policy_sha256'] == hashlib.sha256(b'official installed-version Ubuntu policy').hexdigest()
    assert new['ready'] is False
    assert new['status'] == 'candidate'
    assert result['ready'] is False
    assert result['status'] == 'candidate'
    assert result['changes'][0]['old_sha256'] != result['changes'][0]['sha256']
    assert json.loads(output.with_suffix('.lock.host-refresh.json').read_text())['profile_sha256'] == runtime.sha256(output)


@pytest.mark.parametrize('reason', ['below fixed floor', 'untrusted Ubuntu origin'])
def test_failed_host_gate_writes_no_candidate(profile, monkeypatch, reason):
    path, _, _, _ = setup_host(profile, monkeypatch)
    def reject(doc):
        raise runtime.ProfileError(reason)
    monkeypatch.setattr(runtime, '_host', reject)
    output = path.parent / 'refreshed.lock'
    with pytest.raises(runtime.ProfileError, match=reason):
        refresh.refresh_host(path, output)
    assert not output.exists()


def test_refresh_rejects_changed_owner(profile, monkeypatch):
    path, _, lib, _ = setup_host(profile, monkeypatch)
    monkeypatch.setattr(runtime, '_run', lambda command, **kw: f'evil-package: {lib}')
    with pytest.raises(runtime.ProfileError, match='owner mismatch'):
        refresh.refresh_host(path, path.parent / 'new.lock')


def test_refresh_cannot_bless_tampered_static_artifact(profile, monkeypatch):
    path, doc, _, _ = setup_host(profile, monkeypatch)
    from pathlib import Path
    Path(doc['artifacts'][0]['path']).write_bytes(b'tamper')
    with pytest.raises(runtime.ProfileError, match='SHA256 mismatch'):
        refresh.refresh_host(path, path.parent / 'new.lock')


def test_refresh_never_overwrites_existing_profile(profile, monkeypatch):
    path, _, _, old = setup_host(profile, monkeypatch)
    with pytest.raises(runtime.ProfileError, match='new profile'):
        refresh.refresh_host(path, path)
    assert path.read_bytes() == old


def test_refresh_rejects_malformed_native_owner(profile, monkeypatch):
    path, doc, _, _ = setup_host(profile, monkeypatch)
    doc['native_files'][0]['owner'] = 7
    path.write_text(json.dumps(doc))
    with pytest.raises(runtime.ProfileError, match='path and owner'):
        refresh.refresh_host(path, path.parent / 'new.lock')
