"""Incomplete native builds must remain visibly unpromotable."""
import json
from pathlib import Path
import subprocess
import sys
import importlib.util
import zipfile

import pytest

from scripts import assemble_native_profile as assembler
from scripts import freeze_native_profile as freezer
from scripts import native_runtime


def test_assembly_reports_missing_wheels_without_inventing_hashes(tmp_path):
    script = Path(__file__).parents[1] / 'scripts/assemble_native_profile.py'
    result = subprocess.run([sys.executable, str(script), '--root', str(tmp_path),
                             '--output', str(tmp_path / 'native-runtime.lock')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    doc = json.loads((tmp_path / 'native-runtime.lock').read_text())
    assert doc['ready'] is False
    assert set(doc['pending_artifacts']) == {'biominer', 'psycopg-c', 'pyarrow', 'duckdb', 'lxml'}
    assert doc['artifacts'] == []


def test_assembly_does_not_infer_success_from_project_wheel_presence(tmp_path):
    work = tmp_path / '.native-runtime/biominer/wheelhouse'
    work.mkdir(parents=True)
    (work / 'biominer-0.1.0-py3-none-any.whl').write_bytes(b'unattributed wheel')
    (work.parent / 'source.tar').write_bytes(b'unattributed source')
    script = Path(__file__).parents[1] / 'scripts/assemble_native_profile.py'
    result = subprocess.run([sys.executable, str(script), '--root', str(tmp_path),
                             '--output', str(tmp_path / 'native-runtime.lock')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'biominer' in json.loads((tmp_path / 'native-runtime.lock').read_text())['pending_artifacts']


def test_preservation_keeps_exact_toolchain_bytes(tmp_path):
    script = Path(__file__).parents[1] / 'scripts/assemble_native_profile.py'
    spec = importlib.util.spec_from_file_location('assembler', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tool = tmp_path / 'temporary-python'
    tool.write_bytes(b'approved interpreter bytes')
    receipt = {'toolchain': {'python': module.file_record(tool)}, 'inputs': [], 'native_link_artifacts': []}
    preserved = module.preserve_inputs(receipt, tmp_path / 'materials')
    tool.unlink()
    item = preserved['toolchain']['python']
    assert Path(item['path']).read_bytes() == b'approved interpreter bytes'
    assert item['original_path'] == str(tool)


def test_preservation_retains_backend_manifest_dependencies(tmp_path):
    script = Path(__file__).parents[1] / 'scripts/assemble_native_profile.py'
    spec = importlib.util.spec_from_file_location('assembler_nested', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = tmp_path / 'backend.py'
    backend.write_bytes(b'actual build backend')
    manifest = tmp_path / 'build-backend-inputs.json'
    manifest.write_text(json.dumps([module.file_record(backend)]))
    receipt = {'toolchain': {}, 'inputs': [module.file_record(manifest)], 'native_link_artifacts': []}
    preserved = module.preserve_inputs(receipt, tmp_path / 'materials')
    backend.unlink()
    nested = [item for item in preserved['inputs'] if item.get('kind') == 'backend-input']
    assert len(nested) == 1
    assert Path(nested[0]['path']).read_bytes() == b'actual build backend'


def test_preserved_backend_inputs_do_not_require_retired_build_environment(tmp_path):
    backend = tmp_path / 'backend.py'
    backend.write_bytes(b'exact historical backend')
    manifest = tmp_path / 'build-backend-inputs.json'
    manifest.write_text(json.dumps([assembler.file_record(backend)]))
    receipt = {'toolchain': {}, 'inputs': [assembler.file_record(manifest)],
               'native_link_artifacts': []}
    retained = assembler.preserve_inputs(receipt, tmp_path / 'first')
    backend.unlink()
    manifest.unlink()

    again = assembler.preserve_inputs(retained, tmp_path / 'second')

    assert all(Path(item['path']).is_file() for item in again['inputs'])
    assert len(again['inputs']) == len(retained['inputs'])


def test_new_assembly_preserves_previous_candidate_receipt(tmp_path):
    script = Path(__file__).parents[1] / 'scripts/assemble_native_profile.py'
    spec = importlib.util.spec_from_file_location('assembler_immutable', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.preserve_receipt({'wheel_sha256': 'first'}, tmp_path, 'duckdb')
    approved = module.file_record(original)
    newer = module.preserve_receipt({'wheel_sha256': 'second'}, tmp_path, 'duckdb')
    assert original != newer
    assert module.file_record(original) == approved
    assert json.loads(original.read_text())['wheel_sha256'] == 'first'


@pytest.mark.parametrize('explicit_override', [False, True])
@pytest.mark.parametrize('package', ['biominer', 'lxml', 'psycopg-c', 'pyarrow'])
def test_assembly_selects_build_receipt(tmp_path, explicit_override, package):
    receipts = []
    version = {'biominer': '0.1.0', 'lxml': '6.1.3', 'psycopg-c': '3.3.6',
               'pyarrow': '25.0.1+biominer.1'}[package]
    base = {'psycopg-c': 'psycopg', 'pyarrow': 'arrow/generation-1'}.get(package, package)
    for name in (base, package + '-upgrade'):
        build = tmp_path / '.native-runtime' / name
        build.mkdir(parents=True)
        wheel = build / f'{package}-{version}-py3-none-any.whl'
        with zipfile.ZipFile(wheel, 'w') as archive:
            archive.writestr(f'{package}/__init__.py', name)
        receipt = build / ('build-receipt.json' if package == 'pyarrow' else 'receipt.json')
        receipt.write_text(json.dumps({'version': version, 'wheel': assembler.file_record(wheel)}))
        receipts.append(receipt)
    output = tmp_path / 'new-profile.lock'
    command = [sys.executable, assembler.__file__, '--root', str(tmp_path),
               '--output', str(output), '--python', str(tmp_path / 'absent-python')]
    selected = receipts[0]
    if explicit_override:
        selected = receipts[1]
        option = {'psycopg-c': 'psycopg', 'pyarrow': 'arrow'}.get(package, package)
        command.extend([f'--{option}-receipt', str(selected)])
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    artifact = next(item for item in json.loads(output.read_text())['artifacts'] if item['name'] == package)
    assert artifact['source_receipt'] == assembler.file_record(selected)
    assert artifact['sha256'] == json.loads(selected.read_text())['wheel']['sha256']


def test_assembly_includes_bounded_lxml_recipe(tmp_path):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    (scripts / 'build_native_lxml.py').write_text('raise SystemExit("fixture")\n')
    output = tmp_path / 'candidate.lock'
    assembler.assemble(tmp_path, output, tmp_path / 'absent-python')
    recipe = json.loads(output.read_text())['recipes'][0]
    assert Path(recipe['script']['path']).name == 'build_native_lxml.py'
    assert recipe['args'] == [
        '--python', str(tmp_path / '.native-runtime/lxml/build-env/bin/python'),
        '--sources', str(tmp_path / '.native-runtime/lxml/downloads'),
        '--output', '{output}/lxml', '--jobs', '4',
    ]


@pytest.mark.parametrize('existing_kind', ['file', 'dangling-symlink'])
def test_assembly_rejects_existing_output_before_side_effects(tmp_path, monkeypatch, existing_kind):
    output = tmp_path / 'existing-profile.lock'
    if existing_kind == 'file':
        output.write_bytes(b'previous immutable profile')
    else:
        output.symlink_to(tmp_path / 'absent-target')
    def unexpected(*args, **kwargs):
        pytest.fail('existing output must be rejected before probing or preserving inputs')
    monkeypatch.setattr(assembler.subprocess, 'check_output', unexpected)
    monkeypatch.setattr(assembler.subprocess, 'run', unexpected)
    monkeypatch.setattr(assembler.shutil, 'which', unexpected)
    monkeypatch.setattr(assembler, 'preserve_inputs', unexpected)
    with pytest.raises(FileExistsError, match='exist'):
        assembler.assemble(tmp_path, output, Path(sys.executable), preserve=True)
    if existing_kind == 'file':
        assert output.read_bytes() == b'previous immutable profile'
    else:
        assert output.is_symlink()
        assert output.readlink() == tmp_path / 'absent-target'


def test_assembly_never_overwrites_a_concurrently_created_output(tmp_path, monkeypatch):
    output = tmp_path / 'new-profile.lock'
    monkeypatch.setattr(assembler.shutil, 'which', lambda name: None)
    def probe(*args, **kwargs):
        output.write_bytes(b'concurrent profile')
        return subprocess.CompletedProcess(args, 0, stdout='package policy')
    monkeypatch.setattr(assembler.subprocess, 'run', probe)
    with pytest.raises(FileExistsError):
        assembler.assemble(tmp_path, output, tmp_path / 'absent-python')
    assert output.read_bytes() == b'concurrent profile'


@pytest.fixture
def mixed_host_receipts(tmp_path, monkeypatch):
    """Synthetic wheel receipts; real file hash/owner checker, mocked dpkg/apt."""
    library = tmp_path / 'libfixture.so'
    library.write_bytes(b'current deployment library')
    old = dict(path=str(library), owner='dpkg:libfixture', sha256='a' * 64)
    current = assembler.file_record(library) | {'owner': 'dpkg:libfixture'}
    receipts = []
    for package, row in [('psycopg-c', old), ('pyarrow', current)]:
        wheel = tmp_path / f'{package}.whl'
        with zipfile.ZipFile(wheel, 'w') as archive:
            archive.writestr(f'{package}/__init__.py', 'fixture')
        receipt = tmp_path / f'{package}.json'
        receipt.write_text(json.dumps({'version': '1.0', 'wheel': assembler.file_record(wheel),
                                      'native_files': [row]}))
        receipts.append(receipt)
    host = tmp_path / 'deployment.json'
    host.write_text(json.dumps({'schema_version': 1, 'native_files': [current],
                               'host_packages': [{'name': 'libfixture'}]}))
    monkeypatch.setattr(assembler.shutil, 'which', lambda name: None)
    monkeypatch.setattr(assembler.subprocess, 'run',
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout='fixture apt policy'))
    monkeypatch.setattr(native_runtime, '_host',
                        lambda doc: [{'name': 'libfixture', 'version': '2', 'architecture': 'amd64'}])
    monkeypatch.setattr(native_runtime, '_run',
                        lambda argv: f'libfixture:amd64: {library}\n')
    return {'root': tmp_path, 'library': library, 'old': old, 'current': current,
            'receipts': receipts, 'host': host}


def assemble_mixed(fixture, **kwargs):
    return assembler.assemble(
        fixture['root'], fixture['root'] / 'candidate.json', fixture['root'] / 'absent-python',
        psycopg_receipt=fixture['receipts'][0], arrow_receipt=fixture['receipts'][1], **kwargs)


def test_mixed_host_build_history_still_fails_without_explicit_deployment_profile(mixed_host_receipts):
    with pytest.raises(ValueError, match='conflicting native'):
        assemble_mixed(mixed_host_receipts)


def test_explicit_host_profile_reconciles_deployment_without_rewriting_build_history(mixed_host_receipts):
    fixture = mixed_host_receipts
    originals = {path: path.read_bytes() for path in fixture['receipts']}
    original_host = fixture['host'].read_bytes()
    result = assemble_mixed(fixture, deployment_host_profile=fixture['host'],
                            deployment_host_profile_sha256=assembler.sha256(fixture['host']))
    assert result['ready'] is False
    doc = json.loads((fixture['root'] / 'candidate.json').read_bytes())
    assert doc['native_files'] == [fixture['current']]
    assert all(path.read_bytes() == value for path, value in originals.items())
    assert fixture['host'].read_bytes() == original_host
    reconciliation = doc['deployment_host_reconciliation']
    retained = reconciliation['deployment_profile']
    assert Path(retained['path']).read_bytes() == original_host
    assert retained['sha256'] == assembler.sha256(fixture['host'])
    assert any(item['path'] == retained['path'] for item in doc['inputs'])
    assert reconciliation['mappings'][0]['build_records'] == [
        {'recorded_path': item['path'], 'recorded_sha256': item['sha256'], 'owner': item['owner']}
        for item in (fixture['old'], fixture['current'])]
    assert reconciliation['mappings'][0]['selected'] == fixture['current']
    assert {item['name'] for item in doc['host_packages']} == {
        'openssl', 'libssl3t64', 'libssl-dev', 'libpq5', 'libfixture'}
    assert next(item for item in doc['host_packages'] if item['name'] == 'libpq5')['minimum_version'] == '18.6-0ubuntu0.26.04.1'


def freeze_mixed_candidate(fixture):
    """Exercise the real freezer with explicitly synthetic artifact admission."""
    root = fixture['root']
    (root / 'scripts').mkdir()
    (root / 'scripts/native_runtime.py').write_text(
        'import hashlib\nfrom pathlib import Path\n'
        'def verify_artifacts(path):\n'
        '    return {"status": "candidate", "ready": False,\n'
        '            "profile_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}\n'
        'def main(argv=None):\n    return 0\n')
    (root / 'pyproject.toml').write_text('# synthetic freezer control fixture\n')
    (root / 'uv.lock').write_text('# synthetic freezer lock fixture\n')
    assemble_mixed(fixture, deployment_host_profile=fixture['host'],
                   deployment_host_profile_sha256=assembler.sha256(fixture['host']))
    output = root / 'snapshot'
    freezer.freeze(root / 'candidate.json', output)
    return output


def test_reconciled_host_history_survives_unchanged_freezer(mixed_host_receipts):
    fixture = mixed_host_receipts
    assert assembler.sha256(fixture['library']) != fixture['old']['sha256']
    snapshot = freeze_mixed_candidate(fixture)
    frozen = json.loads((snapshot / 'profile.json').read_bytes())
    mapping = frozen['deployment_host_reconciliation']['mappings'][0]
    assert mapping['build_records'][0] == {
        'recorded_path': fixture['old']['path'], 'recorded_sha256': fixture['old']['sha256'],
        'owner': fixture['old']['owner']}
    assert mapping['selected'] == fixture['current']
    references = list(freezer.records(frozen))
    assert mapping['selected'] in references
    assert not any(item['sha256'] == fixture['old']['sha256'] for item in references)
    assert freezer.verify_snapshot(snapshot)['ready'] is False


def test_frozen_reconciled_host_still_rejects_changed_selected_library(mixed_host_receipts):
    fixture = mixed_host_receipts
    snapshot = freeze_mixed_candidate(fixture)
    fixture['library'].write_bytes(b'changed selected deployment library')
    with pytest.raises(freezer.SnapshotError, match='retained file missing or SHA256 hash mismatch'):
        freezer.verify_snapshot(snapshot)


@pytest.mark.parametrize('bad', ['missing_hash', 'missing_profile', 'changed_profile', 'missing_path',
                               'duplicate_path', 'selected_owner', 'missing_package',
                               'changed_live_hash', 'invalid_selected_hash', 'relative_path',
                               'alias_path', 'incoming_owner', 'invalid_incoming_hash',
                               'actual_owner', 'package_checks'])
def test_explicit_host_inventory_fails_closed(mixed_host_receipts, monkeypatch, bad):
    fixture = mixed_host_receipts
    host = json.loads(fixture['host'].read_bytes())
    kwargs = {'deployment_host_profile': fixture['host']}
    if bad == 'missing_path':
        host['native_files'] = []
    elif bad == 'duplicate_path':
        host['native_files'].append(dict(host['native_files'][0]))
    elif bad == 'selected_owner':
        host['native_files'][0]['owner'] = 'dpkg:other'
    elif bad == 'missing_package':
        host['host_packages'] = []
    elif bad == 'invalid_selected_hash':
        host['native_files'][0]['sha256'] = 'unverified'
    elif bad == 'relative_path':
        host['native_files'][0]['path'] = 'libfixture.so'
    elif bad == 'alias_path':
        alias = fixture['root'] / 'alias.so'
        alias.symlink_to(fixture['library'])
        host['native_files'][0]['path'] = str(alias)
    elif bad in {'incoming_owner', 'invalid_incoming_hash'}:
        receipt = json.loads(fixture['receipts'][0].read_bytes())
        receipt['native_files'][0]['owner' if bad == 'incoming_owner' else 'sha256'] = (
            'dpkg:other' if bad == 'incoming_owner' else 'unverified')
        fixture['receipts'][0].write_text(json.dumps(receipt))
    elif bad == 'actual_owner':
        monkeypatch.setattr(native_runtime, '_run', lambda argv: f'other:amd64: {fixture["library"]}\n')
    elif bad == 'package_checks':
        def reject(doc):
            raise native_runtime.ProfileError('fixture package version/origin mismatch')
        monkeypatch.setattr(native_runtime, '_host', reject)
    fixture['host'].write_text(json.dumps(host))
    kwargs['deployment_host_profile_sha256'] = assembler.sha256(fixture['host'])
    if bad == 'missing_hash':
        kwargs.pop('deployment_host_profile_sha256')
    elif bad == 'missing_profile':
        kwargs.pop('deployment_host_profile')
    elif bad == 'changed_profile':
        fixture['host'].write_text('{}')
    elif bad == 'changed_live_hash':
        fixture['library'].write_bytes(b'unapproved replacement')
    with pytest.raises(ValueError):
        assemble_mixed(fixture, **kwargs)
    assert not (fixture['root'] / 'candidate.json').exists()


@pytest.mark.parametrize('owner', ['wheel:fixture', 'python-runtime:fixture'])
def test_explicit_host_profile_never_resolves_non_dpkg_conflicts(mixed_host_receipts, owner):
    fixture = mixed_host_receipts
    for path in fixture['receipts']:
        receipt = json.loads(path.read_bytes())
        receipt['native_files'][0]['owner'] = owner
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='conflicting native'):
        assemble_mixed(fixture, deployment_host_profile=fixture['host'],
                       deployment_host_profile_sha256=assembler.sha256(fixture['host']))
    assert not (fixture['root'] / 'candidate.json').exists()


@pytest.mark.parametrize('deployment_minimum,expected', [
    ('0', '18.6-0ubuntu0.26.04.1'), ('999', '999')])
def test_deployment_inventory_keeps_stronger_mandatory_package_floor(
        mixed_host_receipts, monkeypatch, deployment_minimum, expected):
    fixture = mixed_host_receipts
    host = json.loads(fixture['host'].read_bytes())
    host['host_packages'].append({'name': 'libpq5', 'minimum_version': deployment_minimum})
    fixture['host'].write_text(json.dumps(host))
    def query(argv, **kwargs):
        result = 1 if argv[:2] == ['/usr/bin/dpkg', '--compare-versions'] and argv[2] == '0' else 0
        return subprocess.CompletedProcess(argv, result, stdout='fixture apt policy')
    monkeypatch.setattr(assembler.subprocess, 'run', query)
    assemble_mixed(fixture, deployment_host_profile=fixture['host'],
                   deployment_host_profile_sha256=assembler.sha256(fixture['host']))
    doc = json.loads((fixture['root'] / 'candidate.json').read_bytes())
    assert next(item for item in doc['host_packages'] if item['name'] == 'libpq5')['minimum_version'] == expected


def test_live_deployment_bytes_changed_during_assembly_block_publication(mixed_host_receipts, monkeypatch):
    fixture = mixed_host_receipts
    def changed(argv, **kwargs):
        fixture['library'].write_bytes(b'changed after initial host inventory check')
        return subprocess.CompletedProcess(argv, 0, stdout='fixture apt policy')
    monkeypatch.setattr(assembler.subprocess, 'run', changed)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        assemble_mixed(fixture, deployment_host_profile=fixture['host'],
                       deployment_host_profile_sha256=assembler.sha256(fixture['host']))
    assert not (fixture['root'] / 'candidate.json').exists()
