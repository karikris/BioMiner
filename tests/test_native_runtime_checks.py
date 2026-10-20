"""Native runtime admission checks must inspect actual imported providers."""
from pathlib import Path

import pytest


def test_runtime_checks_reject_binary_psycopg(checks):
    with pytest.raises(ValueError, match='system-linked'):
        checks.check_psycopg_implementation('binary')


def test_native_tls_fixture_verifies_certificates(tmp_path):
    import sys
    import subprocess
    import json

    path = Path(__file__).parents[1] / 'scripts/check_native_tls.py'
    result = subprocess.run([sys.executable, str(path), '--output', str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    for name in ('libpq_valid_certificate', 'libpq_wrong_hostname_rejected',
                 'arrow_valid_certificate', 'arrow_wrong_hostname_rejected',
                 'libpq_untrusted_issuer_rejected', 'arrow_untrusted_issuer_rejected'):
        assert receipt[name] is True


@pytest.fixture
def checks():
    from scripts import check_native_runtime
    return check_native_runtime


def test_geometry_requires_spatial_provider(checks, tmp_path):
    import duckdb
    class MissingSpatial:
        def execute(self, *args):
            raise duckdb.CatalogException('Spatial provider unavailable')
    with pytest.raises(duckdb.Error, match='Spatial provider unavailable'):
        checks.check_geometry(MissingSpatial(), tmp_path)


@pytest.mark.parametrize('accepted', ['allow_unsigned_extensions', 'allow_extensions_metadata_mismatch'])
def test_extension_policy_rejects_each_signature_bypass(checks, accepted):
    class UnsafeDuckDB:
        Error = RuntimeError
        @staticmethod
        def connect(*, config):
            if config.get(accepted) == 'true':
                return object()
            raise RuntimeError('setting rejected')
    with pytest.raises(ValueError, match=accepted):
        checks.check_extension_settings(UnsafeDuckDB)


def probe_profile(checks, tmp_path):
    import hashlib
    import json
    path = tmp_path / 'native-runtime.lock'
    script = Path(checks.__file__).resolve()
    doc = {'extension_policy': {'policy_id': 'a' * 64},
           'inputs': [{'path': str(script), 'sha256': hashlib.sha256(script.read_bytes()).hexdigest()}]}
    path.write_text(json.dumps(doc))
    return path, doc


@pytest.mark.parametrize('damage', ['wrong_profile', 'wrong_order', 'wrong_policy', 'failed'])
def test_import_order_probes_reject_unbound_receipts(checks, tmp_path, monkeypatch, damage):
    import hashlib
    import json
    import subprocess
    path, doc = probe_profile(checks, tmp_path)
    def run(argv, **kwargs):
        order = argv[argv.index('--native-order') + 1].split(',')
        receipt = {'status': 'passed', 'profile_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'policy_id': doc['extension_policy']['policy_id'], 'import_order': order}
        if damage == 'wrong_profile':
            receipt['profile_sha256'] = 'f' * 64
        elif damage == 'wrong_order':
            receipt['import_order'] = list(reversed(order))
        elif damage == 'wrong_policy':
            receipt['policy_id'] = 'f' * 64
        else:
            receipt['status'] = 'failed'
        return subprocess.CompletedProcess(argv, 0, json.dumps(receipt), '')
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(ValueError, match='probe'):
        checks.check_import_orders(path, tmp_path)


def test_import_order_probes_require_hashed_script_input(checks, tmp_path):
    import json
    path, doc = probe_profile(checks, tmp_path)
    doc['inputs'][0]['sha256'] = 'f' * 64
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='source'):
        checks.check_import_orders(path, tmp_path)


@pytest.mark.parametrize('accepted', ['allow_unsigned_extensions', 'allow_extensions_metadata_mismatch'])
def test_extension_policy_rejects_runtime_setting_bypass(checks, accepted):
    class Connection:
        def execute(self, statement):
            if accepted in statement:
                return self
            raise RuntimeError('setting rejected')
        def close(self):
            pass
    class UnsafeDuckDB:
        Error = RuntimeError
        @staticmethod
        def connect(*, config=None):
            if config:
                raise RuntimeError('constructor setting rejected')
            return Connection()
    with pytest.raises(ValueError, match=accepted):
        checks.check_extension_settings(UnsafeDuckDB)


def test_arrow_cannot_report_python_only_crypto_line(checks):
    with pytest.raises(ValueError, match='crypto'):
        checks.check_crypto_versions('wheel:pyarrow', {'3.5.9'})


def test_python_crypto_line_is_accepted_only_for_python_owner(checks):
    checks.check_crypto_versions('python-runtime:Astral20260929', {'3.5.9'})
    checks.check_crypto_versions('wheel:duckdb', {'3.6.5'})
    checks.check_crypto_versions('dpkg:libssl3t64', {'3.5.5'})


def test_import_orders_accept_opposite_bound_process_receipts(checks, tmp_path, monkeypatch):
    import hashlib
    import json
    import subprocess
    import sys
    path, doc = probe_profile(checks, tmp_path)
    calls = []
    def run(argv, **kwargs):
        assert argv[:3] == [sys.executable, '-I', '-B']
        calls.append(argv)
        order = argv[argv.index('--native-order') + 1].split(',')
        receipt = {'status': 'passed', 'profile_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'policy_id': doc['extension_policy']['policy_id'], 'import_order': order}
        return subprocess.CompletedProcess(argv, 0, json.dumps(receipt), '')
    monkeypatch.setattr(subprocess, 'run', run)
    result = checks.check_import_orders(path, tmp_path)
    assert len(calls) == 2
    assert result[0] == list(reversed(result[1]))


def test_tls_check_requires_its_own_hashed_source(checks, tmp_path):
    path, _ = probe_profile(checks, tmp_path)
    with pytest.raises(ValueError, match='source'):
        checks.profile_check('tls', path, tmp_path)


def test_python_fips_docstring_is_not_a_crypto_provider_version(checks, tmp_path):
    library = tmp_path / 'python.elf'
    library.write_bytes(b'\0' + checks.PYTHON_FIPS_DOC + b'\0OpenSSL 3.5.9\0')
    assert checks.observed_crypto_versions(library, 'python-runtime:Astral20260929') == {'3.5.9'}


def test_fips_docstring_cannot_hide_actual_old_crypto(checks, tmp_path):
    library = tmp_path / 'python.elf'
    library.write_bytes(b'\0' + checks.PYTHON_FIPS_DOC + b'\0OpenSSL 3.0.0 7 Sep 2021\0')
    assert '3.0.0' in checks.observed_crypto_versions(library, 'python-runtime:Astral20260929')


def test_python_docstring_exception_cannot_apply_to_arrow(checks, tmp_path):
    library = tmp_path / 'arrow.elf'
    library.write_bytes(b'\0' + checks.PYTHON_FIPS_DOC + b'\0')
    assert '3.0.0' in checks.observed_crypto_versions(library, 'wheel:pyarrow')
