"""Incomplete native builds must remain visibly unpromotable."""
import json
from pathlib import Path
import subprocess
import sys
import importlib.util
import zipfile

import pytest

from scripts import assemble_native_profile as assembler


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
