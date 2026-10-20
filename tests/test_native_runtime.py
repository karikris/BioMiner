from __future__ import annotations

import hashlib
import json
import platform
import sys
import zipfile
from pathlib import Path

import pytest

from scripts import native_runtime as runtime


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blob(root, name, text='input'):
    path = root / name
    path.write_text(text)
    return {'path': str(path), 'sha256': digest(path)}


def wheel(root, name, version):
    stem = name.replace('-', '_')
    path = root / f'{stem}-{version}-cp314-cp314-linux_x86_64.whl'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(f'{stem}-{version}.dist-info/METADATA', f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n')
        archive.writestr(f'{stem}-{version}.dist-info/WHEEL', 'Wheel-Version: 1.0\nTag: cp314-cp314-linux_x86_64\n')
    inputs = [blob(root, name + '.source.tar')]
    if name == 'biominer':
        overlay = root / 'biominer-build'
        overlay.mkdir()
        inputs.append(blob(overlay, 'pyproject.toml', (root / 'pyproject.toml').read_text()))
    receipt = blob(root, name + '.receipt.json', json.dumps({'wheel_sha256': digest(path), 'source_revision': 'a' * 40, 'toolchain': {'compiler': blob(root, name + '.compiler')}, 'inputs': inputs, 'inputs_sha256': hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest(), 'native_link_artifacts': [blob(root, name + '.link-map')], 'effective_versions': {'openssl': '3.6.5'}}))
    if name == 'lxml':
        patch = blob(root, 'zlib-fix.patch', 'reviewed patch fixture')
        value = json.loads(Path(receipt['path']).read_text())
        value['inputs'].append(patch)
        licenses = [blob(root, filename, filename + ' test content') for filename in (
            'libiconv-COPYING.LIB', 'libcharset-COPYING.LIB', 'NOTICE.txt',
            'zlib-LICENSE', 'libxml2-Copyright', 'libxslt-Copyright')]
        extension = b'fixture native extension'
        with zipfile.ZipFile(path, 'a') as archive:
            archive.writestr('lxml/etree.cp314.so', extension)
            for item in licenses:
                archive.writestr('lxml-6.1.3.dist-info/licenses/' + Path(item['path']).name,
                                 Path(item['path']).read_bytes())
        value['wheel_sha256'] = digest(path)
        value['license_manifest'] = blob(root, 'license-manifest.json', json.dumps(licenses))
        value['relink'] = {
            'bundle': blob(root, 'relink-materials.tar.gz', 'fixture bundle'),
            'manifest': blob(root, 'relink-manifest.json', '{}'),
            'qualification': blob(root, 'relink-qualification.json', json.dumps({
                'status': 'passed', 'relocated_bundle': True,
                'replacement_library_sha256': 'a' * 64,
                'outputs': [{'name': 'etree.cp314.so',
                             'original_sha256': hashlib.sha256(extension).hexdigest(),
                             'relinked_sha256': hashlib.sha256(extension).hexdigest()}],
            })),
        }
        value['inputs'].extend([*licenses, value['license_manifest'], *value['relink'].values()])
        value['inputs_sha256'] = hashlib.sha256(json.dumps(value['inputs'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        value['effective_versions'] = {
            'lxml': '6.1.3', 'libxml2': '2.15.4', 'libxslt': '1.1.45',
            'libiconv': '1.19', 'zlib': '1.3.2',
        }
        value['native_policy'] = {
            'zlib_patch_commit': 'df84af25dc1942490e1d1c899a07619152a46148',
            'zlib_patch_sha256': patch['sha256'],
            'libxml_features': ['catalog', 'html', 'iconv', 'regexp', 'schematron',
                                'xmlschema', 'xpath', 'zlib'],
        }
        receipt = blob(root, name + '.receipt.json', json.dumps(value))
    return {'name': name, 'version': version, 'path': str(path), 'sha256': digest(path), 'source_receipt': receipt}


@pytest.fixture
def profile(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'LXML_ZLIB_PATCH_SHA256',
                        hashlib.sha256(b'reviewed patch fixture').hexdigest())
    doc = {
        'schema_version': 1,
        'platform': {'system': 'Linux', 'machine': 'x86_64'},
        'base_lock': blob(tmp_path, 'uv.lock'),
        'project': blob(tmp_path, 'pyproject.toml', '[project]\nname="biominer"\nversion="0.1.0"\nrequires-python=">=3.14.7"\n'),
        'inputs': [dict(blob(tmp_path, 'sources.tar'), kind='source', source_revision='a' * 40)],
        'artifacts': [wheel(tmp_path, 'psycopg-c', '3.3.6'), wheel(tmp_path, 'pyarrow', '25.0.1+biominer.1'), wheel(tmp_path, 'duckdb', '1.5.6+biominer.1'), wheel(tmp_path, 'biominer', '0.1.0'), wheel(tmp_path, 'lxml', '6.1.3')],
        'python': {'path': str(Path(sys.executable).resolve()), 'sha256': digest(Path(sys.executable).resolve()), 'version': '3.14.7', 'openssl_version': '3.5.9'},
        'uv': blob(tmp_path, 'uv'),
        'native_files': [],
        'host_packages': [{'name': 'libssl3t64', 'minimum_version': '3.5.5-1ubuntu3.7', 'architecture': 'amd64', 'origin': 'Ubuntu'}],
        'extension_policy': {'policy_id': 'c' * 64, 'allowlist': []},
        'checks': [],
        'recipes': [],
    }
    path = tmp_path / 'native-runtime.lock'
    path.write_text(json.dumps(doc))
    return path, doc


def rewrite(profile):
    path, doc = profile
    path.write_text(json.dumps(doc))
    return path


def test_valid_artifact_inventory_is_candidate_not_ready(profile):
    result = runtime.verify_artifacts(profile[0])
    assert result['status'] == 'candidate'
    assert result['ready'] is False


def test_unpatched_registry_lxml_is_not_an_admitted_native_runtime(profile):
    # The lock's registry wheel is insufficient even when its Python version
    # matches: the patched native closure must be admitted separately.
    profile[1]['artifacts'] = [item for item in profile[1]['artifacts'] if item['name'] != 'lxml']
    with pytest.raises(runtime.ProfileError, match='lxml'):
        runtime.verify_artifacts(rewrite(profile))


@pytest.mark.parametrize('damage', ['libxml2', 'libxslt', 'libiconv', 'zlib',
                                   'patch_identity', 'patch_input', 'feature'])
def test_lxml_receipt_cannot_claim_an_unbound_patch(profile, damage):
    _, doc = profile
    artifact = next(item for item in doc['artifacts'] if item['name'] == 'lxml')
    receipt_path = Path(artifact['source_receipt']['path'])
    receipt = json.loads(receipt_path.read_text())
    if damage in receipt['effective_versions']:
        receipt['effective_versions'][damage] = '0.0.0'
    elif damage == 'patch_identity':
        receipt['native_policy']['zlib_patch_commit'] = 'f' * 40
    elif damage == 'patch_input':
        receipt['inputs'] = [item for item in receipt['inputs']
                             if Path(item['path']).name != 'zlib-fix.patch']
    else:
        receipt['native_policy']['libxml_features'].remove('schematron')
    receipt['inputs_sha256'] = hashlib.sha256(json.dumps(receipt['inputs'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    artifact['source_receipt']['sha256'] = digest(receipt_path)
    with pytest.raises(runtime.ProfileError, match='lxml'):
        runtime.verify_artifacts(rewrite(profile))


@pytest.mark.parametrize('damage', ['missing_bundle', 'unbound_license', 'missing_license',
                                   'not_relocated', 'different_extension',
                                   'invalid_license_path', 'invalid_output_name'])
def test_lxml_requires_retained_qualified_relink_materials(profile, damage):
    artifact = next(item for item in profile[1]['artifacts'] if item['name'] == 'lxml')
    path = Path(artifact['source_receipt']['path'])
    receipt = json.loads(path.read_text())
    if damage == 'missing_bundle':
        receipt['inputs'].remove(receipt['relink']['bundle'])
    elif damage == 'unbound_license':
        receipt['inputs'].remove(receipt['license_manifest'])
    elif damage in ('missing_license', 'invalid_license_path'):
        manifest = receipt['license_manifest']
        items = json.loads(Path(manifest['path']).read_text())
        if damage == 'missing_license':
            items.pop()
        else:
            items[0]['path'] = None
        Path(manifest['path']).write_text(json.dumps(items))
        for item in receipt['inputs']:
            if item['path'] == manifest['path']:
                item['sha256'] = digest(Path(item['path']))
        manifest['sha256'] = digest(Path(manifest['path']))
    else:
        record = receipt['relink']['qualification']
        data = json.loads(Path(record['path']).read_text())
        if damage == 'not_relocated':
            data['relocated_bundle'] = False
        elif damage == 'invalid_output_name':
            data['outputs'][0]['name'] = []
        else:
            data['outputs'][0]['relinked_sha256'] = 'f' * 64
        Path(record['path']).write_text(json.dumps(data))
        for item in receipt['inputs']:
            if item['path'] == record['path']:
                item['sha256'] = digest(Path(item['path']))
        record['sha256'] = digest(Path(record['path']))
    receipt['inputs_sha256'] = hashlib.sha256(json.dumps(receipt['inputs'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    path.write_text(json.dumps(receipt))
    artifact['source_receipt']['sha256'] = digest(path)
    with pytest.raises(runtime.ProfileError, match='lxml'):
        runtime.verify_artifacts(rewrite(profile))


def test_lxml_relink_materials_remain_valid_after_input_preservation(profile):
    from scripts import assemble_native_profile as assembler

    artifact = next(item for item in profile[1]['artifacts'] if item['name'] == 'lxml')
    path = Path(artifact['source_receipt']['path'])
    receipt = json.loads(path.read_text())
    preserved = assembler.preserve_inputs(receipt, path.parent / 'retained')
    originals = {item['path'] for item in (
        *receipt['inputs'], *receipt['native_link_artifacts'], *receipt['toolchain'].values())}
    for original in originals:
        Path(original).unlink()
    path.write_text(json.dumps(preserved))
    artifact['source_receipt']['sha256'] = digest(path)

    assert runtime.verify_artifacts(rewrite(profile))['status'] == 'candidate'


def test_biominer_receipt_accepts_preserved_matching_project_overlay(profile):
    _, doc = profile
    artifact = next(item for item in doc['artifacts'] if item['name'] == 'biominer')
    receipt = json.loads(Path(artifact['source_receipt']['path']).read_text())
    overlay = next(item for item in receipt['inputs'] if Path(item['path']).name == 'pyproject.toml')
    assert overlay['path'] != doc['project']['path']
    assert overlay['sha256'] == doc['project']['sha256']
    assert runtime.verify_artifacts(profile[0])['ready'] is False


def test_biominer_receipt_rejects_a_new_project_with_an_old_wheel(profile):
    _, doc = profile
    project = Path(doc['project']['path'])
    project.write_text(project.read_text() + 'dependencies=["packaging==26.2"]\n')
    doc['project']['sha256'] = digest(project)
    with pytest.raises(runtime.ProfileError, match='BioMiner.*pyproject.toml'):
        runtime.verify_artifacts(rewrite(profile))


@pytest.mark.parametrize('damage', ['missing', 'ambiguous'])
def test_biominer_receipt_requires_one_project_overlay(profile, damage):
    _, doc = profile
    artifact = next(item for item in doc['artifacts'] if item['name'] == 'biominer')
    receipt_path = Path(artifact['source_receipt']['path'])
    receipt = json.loads(receipt_path.read_text())
    overlay = next(item for item in receipt['inputs'] if Path(item['path']).name == 'pyproject.toml')
    if damage == 'missing':
        receipt['inputs'].remove(overlay)
    else:
        receipt['inputs'].append(dict(overlay))
    receipt['inputs_sha256'] = hashlib.sha256(json.dumps(receipt['inputs'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    artifact['source_receipt']['sha256'] = digest(receipt_path)
    with pytest.raises(runtime.ProfileError, match='BioMiner.*pyproject.toml'):
        runtime.verify_artifacts(rewrite(profile))


@pytest.mark.parametrize('damage', ['missing', 'tampered', 'wrong_version', 'wrong_platform', 'base_lock', 'missing_receipt'])
def test_artifact_verification_rejects_invalid_evidence(profile, damage):
    path, doc = profile
    artifact = doc['artifacts'][1]
    if damage == 'missing':
        Path(artifact['path']).unlink()
    elif damage == 'tampered':
        with Path(artifact['path']).open('ab') as output:
            output.write(b'tampering')
    elif damage == 'wrong_version':
        artifact['version'] = '25.0.1'
    elif damage == 'wrong_platform':
        doc['platform']['machine'] = 'aarch64'
    elif damage == 'base_lock':
        Path(doc['base_lock']['path']).write_text('new lock')
    else:
        Path(artifact['source_receipt']['path']).unlink()
    rewrite(profile)
    with pytest.raises(runtime.ProfileError):
        runtime.verify_artifacts(path)


@pytest.mark.parametrize('value', [{}, {'schema_version': 2}, {'schema_version': 1, 'platform': None}])
def test_malformed_profile_has_useful_error(tmp_path, value):
    path = tmp_path / 'bad.lock'
    path.write_text(json.dumps(value))
    with pytest.raises(runtime.ProfileError, match='profile|schema|platform'):
        runtime.verify_artifacts(path)


def test_absent_runtime_evidence_blocks_readiness(profile):
    with pytest.raises(runtime.ProfileError, match='generation'):
        runtime.verify(profile[0])


@pytest.mark.parametrize('damage', ['outside', 'root_itself', 'symlink', 'invalid_root'])
def test_profile_generation_root_cannot_escape_or_alias(tmp_path, damage):
    root = tmp_path / '.native-runtime/generations'
    root.mkdir(parents=True)
    output = root / 'new'
    if damage == 'outside':
        output = tmp_path / 'outside'
    elif damage == 'root_itself':
        output = root
    elif damage == 'symlink':
        alias = tmp_path / 'alias'
        alias.symlink_to(root, target_is_directory=True)
        output = alias / 'new'
    else:
        root = tmp_path
    with pytest.raises(runtime.ProfileError):
        runtime._new_output(output, str(root))
    assert not (tmp_path / 'outside').exists()
    assert not (tmp_path / '.native-runtime/generations/new').exists()


@pytest.mark.parametrize('option', ['--prof', '--prof='])
def test_native_cli_rejects_abbreviated_profile_override(profile, option):
    arguments = ['artifacts', '--profile', str(profile[0])]
    arguments.extend([option + str(profile[0])] if option.endswith('=') else [option, str(profile[0])])
    with pytest.raises(SystemExit) as error:
        runtime.main(arguments)
    assert error.value.code == 2


def test_install_never_reuses_generation(profile, tmp_path):
    target = tmp_path / 'generation'
    target.mkdir()
    sentinel = target / 'owned-by-another-run'
    sentinel.write_text('preserve')
    with pytest.raises(runtime.ProfileError, match='exist'):
        runtime.install(profile[0], target)
    assert sentinel.read_text() == 'preserve'


@pytest.mark.parametrize('key', ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'GDAL_DRIVER_PATH', 'OPENSSL_MODULES', 'OPENSSL_CONF'])
def test_launch_environment_rejects_loader_overrides(key):
    with pytest.raises(runtime.ProfileError, match=key):
        runtime.launch_environment({key: '/unapproved'})


def test_build_missing_recipe_fails_without_creating_output(profile, tmp_path):
    output = tmp_path / 'build'
    with pytest.raises(runtime.ProfileError, match='recipe'):
        runtime.build(profile[0], output)
    assert not output.exists()


def test_cli_accepts_options_after_command(profile, capsys):
    assert runtime.main(['artifacts', '--profile', str(profile[0])]) == 0
    assert json.loads(capsys.readouterr().out)['ready'] is False


def test_build_executes_pinned_recipe_and_preserves_candidate_status(profile, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, '_python', lambda doc, base: Path(sys.executable))
    path, doc = profile
    recipe = blob(tmp_path, 'recipe.py', 'import pathlib,sys; pathlib.Path(sys.argv[1], "result.txt").write_text("built")')
    doc['recipes'] = [{'script': recipe, 'args': ['{output}']}]
    rewrite(profile)
    output = tmp_path / 'build'
    result = runtime.build(path, output)
    assert (output / 'result.txt').read_text() == 'built'
    assert result['status'] == 'candidate'
    assert result['ready'] is False
    assert {'path': 'result.txt', 'sha256': digest(output / 'result.txt')} in result['outputs']


def test_build_failure_preserves_failure_without_success_receipt(profile, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, '_python', lambda doc, base: Path(sys.executable))
    path, doc = profile
    doc['recipes'] = [{'script': blob(tmp_path, 'fail.py', 'raise SystemExit(4)'), 'args': []}]
    rewrite(profile)
    output = tmp_path / 'failed-build'
    with pytest.raises(runtime.ProfileError, match='command failed'):
        runtime.build(path, output)
    assert json.loads((output / 'FAILED.json').read_text())['ready'] is False
    assert not (output / 'build-receipt.json').exists()


def test_arbitrary_receipt_fields_do_not_substitute_native_link_evidence(profile):
    _, doc = profile
    receipt_path = Path(doc['artifacts'][1]['source_receipt']['path'])
    receipt = json.loads(receipt_path.read_text())
    receipt['native_link_artifacts'] = 'trust me'
    receipt_path.write_text(json.dumps(receipt))
    doc['artifacts'][1]['source_receipt']['sha256'] = digest(receipt_path)
    rewrite(profile)
    with pytest.raises(runtime.ProfileError, match='native_link'):
        runtime.verify_artifacts(profile[0])


def test_missing_mandatory_checks_rejected_before_environment_creation(profile, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, '_host', lambda doc: [])
    monkeypatch.setattr(runtime, '_python', lambda doc, base: Path(sys.executable))
    output = tmp_path / 'env-generation'
    with pytest.raises(runtime.ProfileError, match='mandatory checks'):
        runtime.install(profile[0], output)
    assert not output.exists()


@pytest.fixture
def installable_profile(profile, tmp_path, monkeypatch):
    path, doc = profile
    monkeypatch.setattr(runtime, '_host', lambda doc: [{'name': 'test-host', 'version': 'approved'}])
    monkeypatch.setattr(runtime, '_python', lambda doc, base: Path(sys.executable))
    real_run = runtime._run
    def run(argv, **kwargs):
        if argv[:2] == ['/usr/bin/dpkg-query', '-S']:
            owner = 'libpq5' if Path(argv[-1]).name == 'libpq.so' else 'libssl3t64'
            return owner + ': ' + argv[-1]
        return real_run(argv, **kwargs)
    monkeypatch.setattr(runtime, '_run', run)
    # A deterministic installer stand-in performs real venv creation and wheel
    # extraction. This tests generation admission, not uv's implementation.
    fake_uv = blob(tmp_path, 'uv', f'#!{sys.executable}\n' + '''
import hashlib,json,pathlib,sys,urllib.parse,venv,zipfile
args=sys.argv[1:]
if args[0]=='venv':
    venv.EnvBuilder(with_pip=False).create(args[-1])
elif args[0]=='sync':
    required=['--locked','--no-editable','--no-install-project','ml']
    assert all(value in args for value in required)
    assert all(args.count(name)==1 for name in ['psycopg-c','pyarrow','duckdb','lxml'])
else:
    assert all(value in args for value in ['--no-index','--no-deps','--require-hashes'])
    target=pathlib.Path(args[args.index('--python')+1]).parent.parent/'lib/python3.14/site-packages'
    for line in pathlib.Path(args[args.index('-r')+1]).read_text().splitlines():
        uri,expected=line.split(' --hash=sha256:')
        wheel=pathlib.Path(urllib.parse.unquote(urllib.parse.urlparse(uri).path))
        assert hashlib.sha256(wheel.read_bytes()).hexdigest()==expected
        with zipfile.ZipFile(wheel) as archive: archive.extractall(target)
''')
    Path(fake_uv['path']).chmod(0o755)
    doc['uv'] = fake_uv
    check = blob(tmp_path, 'check.py', '''import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text())
print(json.dumps({'status':'passed','profile_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'policy_id':d['extension_policy']['policy_id']}))
''')
    doc['checks'] = [{'name': name, 'script': check, 'args': ['{profile}']} for name in sorted(runtime.INSTALL_CHECKS)]
    doc['native_files'] = [dict(blob(tmp_path, 'libpq.so'), owner='dpkg:libpq5'), dict(blob(tmp_path, 'libssl.so'), owner='dpkg:libssl3t64')]
    rewrite(profile)
    return profile


def unfreeze(root):
    root.chmod(0o755)
    for path in root.rglob('*'):
        if not path.is_symlink():
            path.chmod(0o755 if path.is_dir() else 0o644)


def test_fresh_generation_is_frozen_and_reverified(installable_profile, tmp_path):
    path, _ = installable_profile
    output = tmp_path / 'generation'
    try:
        result = runtime.install(path, output)
        assert result['ready'] is True
        assert result['live_services'] == 'not_attested'
        assert runtime.verify(path, output)['status'] == 'verified'
        assert not output.stat().st_mode & 0o222
    finally:
        if output.exists():
            unfreeze(output)


@pytest.mark.parametrize('damage', ['wheel_bytes', 'new_file', 'profile', 'writable'])
def test_generation_changes_block_execution(installable_profile, tmp_path, damage):
    path, doc = installable_profile
    output = tmp_path / 'generation'
    try:
        runtime.install(path, output)
        if damage == 'profile':
            doc['extension_policy']['policy_id'] = 'd' * 64
            rewrite(installable_profile)
        elif damage == 'writable':
            output.chmod(0o755)
        else:
            unfreeze(output)
            target = output / 'env/lib/python3.14/site-packages/duckdb-1.5.6+biominer.1.dist-info/METADATA' if damage == 'wheel_bytes' else output / 'env/lib/python3.14/site-packages/unapproved.py'
            target.write_text('tampered')
            runtime._freeze(output)
        with pytest.raises(runtime.ProfileError, match='generation'):
            runtime.verify(path, output)
    finally:
        if output.exists():
            unfreeze(output)


def test_source_receipt_is_bound_to_wheel(profile):
    _, doc = profile
    item = doc['artifacts'][1]['source_receipt']
    receipt_path = Path(item['path'])
    receipt = json.loads(receipt_path.read_text())
    receipt['wheel_sha256'] = 'f' * 64
    receipt_path.write_text(json.dumps(receipt))
    item['sha256'] = digest(receipt_path)
    rewrite(profile)
    with pytest.raises(runtime.ProfileError, match='wheel_sha256'):
        runtime.verify_artifacts(profile[0])


def test_missing_native_inventory_blocks_install(installable_profile, tmp_path):
    path, doc = installable_profile
    doc['native_files'] = []
    rewrite(installable_profile)
    output = tmp_path / 'generation'
    with pytest.raises(runtime.ProfileError, match='native_files'):
        runtime.install(path, output)
    assert not output.exists()


def test_compiled_policy_probe_mismatch_blocks_install(installable_profile, tmp_path):
    path, doc = installable_profile
    item = doc['checks'][0]['script']
    check = Path(item['path'])
    check.write_text(check.read_text().replace("d['extension_policy']['policy_id']", "'f'*64"))
    for entry in doc['checks']:
        entry['script']['sha256'] = digest(check)
    rewrite(installable_profile)
    output = tmp_path / 'generation'
    with pytest.raises(runtime.ProfileError, match='policy_id'):
        runtime.install(path, output)
    assert not (output / 'generation.json').exists()
    assert json.loads((output / 'FAILED.json').read_text())['ready'] is False


def test_host_gate_cannot_lower_reviewed_ssl_floor(profile, monkeypatch):
    _, doc = profile
    names = ['libssl3t64', 'libssl-dev', 'openssl', 'libpq5']
    doc['host_packages'] = [{'name': name, 'minimum_version': '3.5.5-1ubuntu3.6', 'architecture': 'amd64', 'origin': 'Ubuntu', 'advisory': 'reviewed advisory'} for name in names]
    original = runtime._run
    def command(argv, **kwargs):
        if argv[:2] == ['/usr/bin/apt-cache', 'policy']:
            return 'release o=Ubuntu'
        return original(argv, **kwargs)
    monkeypatch.setattr(runtime, '_run', command)
    with pytest.raises(runtime.ProfileError):
        runtime._host(doc)


def test_old_official_repository_entry_cannot_prove_installed_origin(profile, monkeypatch):
    _, doc = profile
    def policy(name):
        return f'{name}:\n *** 3.5.5-1ubuntu3.6 100\n        100 /var/lib/dpkg/status\n     3.5.5-1ubuntu3 500\n        500 http://archive.ubuntu.com/ubuntu resolute/main amd64 Packages\n'
    names = ['libssl3t64', 'libssl-dev', 'openssl', 'libpq5']
    doc['host_packages'] = [{'name': name, 'minimum_version': '3.5.5-1ubuntu3.6', 'architecture': 'amd64', 'origin': 'Ubuntu', 'advisory': 'reviewed advisory', 'apt_policy_sha256': hashlib.sha256(policy(name).encode()).hexdigest()} for name in names]
    def command(argv, **kwargs):
        if argv[0].endswith('dpkg-query'):
            return '3.5.5-1ubuntu3.6\tamd64\tinstall ok installed'
        if argv[0].endswith('dpkg'):
            return ''
        return 'release o=Ubuntu' if len(argv) == 2 else policy(argv[-1])
    monkeypatch.setattr(runtime, '_run', command)
    with pytest.raises(runtime.ProfileError, match='origin'):
        runtime._host(doc)


def test_native_file_owner_cannot_be_invented(profile, monkeypatch, tmp_path):
    _, doc = profile
    native = dict(blob(tmp_path, 'libssl.so'), owner='dpkg:libssl3t64')
    monkeypatch.setattr(runtime, '_run', lambda argv, **kwargs: 'unrelated-package: ' + native['path'])
    with pytest.raises(runtime.ProfileError, match='owner'):
        runtime.verify_native_file(native, profile[0], tmp_path)


def test_native_file_requires_known_owner_namespace(profile, tmp_path):
    native = dict(blob(tmp_path, 'unknown.so'), owner='trusted')
    with pytest.raises(runtime.ProfileError, match='owner'):
        runtime.verify_native_file(native, profile[0], tmp_path)


def test_build_timeout_preserves_failed_candidate(profile, tmp_path, monkeypatch):
    path, doc = profile
    monkeypatch.setattr(runtime, '_python', lambda doc, base: Path(sys.executable))
    doc['recipes'] = [{'script': blob(tmp_path, 'blocked.py', 'import time;time.sleep(10)'), 'args': [], 'timeout_seconds': 1}]
    rewrite(profile)
    output = tmp_path / 'bounded-build'
    with pytest.raises(runtime.ProfileError, match='timed out'):
        runtime.build(path, output)
    assert not (output / 'build-receipt.json').exists()
    assert json.loads((output / 'FAILED.json').read_text())['ready'] is False


def test_python_runtime_native_owner_is_bound_to_approved_prefix(profile, tmp_path):
    path, doc = profile
    doc['python']['source_build'] = 'Astral20260929'
    rewrite(profile)
    native = dict(blob(tmp_path, 'foreign.so'), owner='python-runtime:Astral20260929')
    with pytest.raises(runtime.ProfileError, match='Python runtime owner'):
        runtime.verify_native_file(native, path, tmp_path)


def test_python_runtime_native_owner_is_bound_to_source_build(profile, tmp_path):
    path, doc = profile
    doc['python']['source_build'] = 'Astral20260929'
    rewrite(profile)
    native = dict(doc['python'], owner='python-runtime:unapproved')
    with pytest.raises(runtime.ProfileError, match='Python runtime owner'):
        runtime.verify_native_file(native, path, tmp_path)


def test_timed_out_build_terminates_compiler_descendants(profile, tmp_path, monkeypatch):
    import os
    import signal
    path, doc = profile
    monkeypatch.setattr(runtime, '_python', lambda doc, base: Path(sys.executable))
    recipe = blob(tmp_path, 'children.py', 'import pathlib,subprocess,sys,time\np=subprocess.Popen([sys.executable,"-c","import time;time.sleep(100)"])\npathlib.Path(sys.argv[1],"child.pid").write_text(str(p.pid))\ntime.sleep(100)')
    doc['recipes'] = [{'script': recipe, 'args': ['{output}'], 'timeout_seconds': 1}]
    rewrite(profile)
    output = tmp_path / 'bounded-children'
    with pytest.raises(runtime.ProfileError, match='timed out'):
        runtime.build(path, output)
    pid = int((output / 'child.pid').read_text())
    child = Path(f'/proc/{pid}/stat')
    try:
        assert not child.exists() or child.read_text().rsplit(')', 1)[1].split()[0] == 'Z'
    finally:
        if child.exists():
            os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize('field', ['host_packages', 'native_files', 'checks', 'recipes'])
def test_profile_record_lists_reject_nonobjects(profile, field):
    path, doc = profile
    doc[field] = [None]
    rewrite(profile)
    with pytest.raises(runtime.ProfileError, match=field):
        runtime.verify_artifacts(path)


def test_probe_nonobject_result_is_useful_failure(installable_profile, tmp_path):
    path, doc = installable_profile
    script = Path(doc['checks'][0]['script']['path'])
    script.write_text('print("[]")')
    for check in doc['checks']:
        check['script']['sha256'] = digest(script)
    rewrite(installable_profile)
    with pytest.raises(runtime.ProfileError, match='check'):
        runtime.install(path, tmp_path / 'generation')


def test_clean_launch_disables_external_gdal_plugins():
    assert runtime.launch_environment({})['GDAL_DRIVER_PATH'] == 'disable'


def test_static_spatial_wheel_cannot_bundle_external_extension(profile):
    path, doc = profile
    artifact = next(item for item in doc['artifacts'] if item['name'] == 'duckdb')
    with zipfile.ZipFile(artifact['path'], 'a') as archive:
        archive.writestr('duckdb/spatial.duckdb_extension', b'unreviewed extension')
    artifact['sha256'] = digest(Path(artifact['path']))
    receipt_path = Path(artifact['source_receipt']['path'])
    receipt = json.loads(receipt_path.read_text())
    receipt['wheel_sha256'] = artifact['sha256']
    receipt_path.write_text(json.dumps(receipt))
    artifact['source_receipt']['sha256'] = digest(receipt_path)
    rewrite(profile)
    with pytest.raises(runtime.ProfileError, match='external extension'):
        runtime.verify_artifacts(path)


def test_python_stdlib_tampering_cannot_spoof_identity(profile, tmp_path, monkeypatch):
    import shutil
    import tarfile
    from scripts import attest_native_python as attestor
    path, doc = profile
    prefix = tmp_path / 'python'
    (prefix / 'bin').mkdir(parents=True)
    (prefix / 'lib/python3.14').mkdir(parents=True)
    executable = prefix / 'bin/python3.14'
    shutil.copyfile(sys.executable, executable)
    executable.chmod(0o755)
    ssl_source = prefix / 'lib/python3.14/ssl.py'
    ssl_source.write_text("OPENSSL_VERSION = 'OpenSSL 3.5.9'\n")
    archive = tmp_path / 'python.tar'
    with tarfile.open(archive, 'w') as output:
        output.add(prefix, arcname='python')
    matched = attestor.verify_prefix(archive, prefix)
    receipt = {'schema_version': 1, 'source_build': 'Astral20260929', 'url': attestor.URL,
               'archive': {'path': str(archive), 'sha256': digest(archive)},
               'python': {'path': str(executable), 'sha256': digest(executable)},
               'identity': {'version': '3.14.7', 'openssl_version': '3.5.9'},
               'verified_archive_files': matched}
    receipt_record = blob(tmp_path, 'python-receipt.json', json.dumps(receipt))
    doc['inputs'].append(dict(receipt_record, kind='python-source-receipt'))
    attestor_file = Path(attestor.__file__).resolve()
    doc['inputs'].append({'kind': 'recipe', 'path': str(attestor_file), 'sha256': digest(attestor_file)})
    doc['python'] = {'path': str(executable), 'sha256': digest(executable), 'version': '3.14.7', 'openssl_version': '3.5.9', 'source_build': 'Astral20260929'}
    ssl_source.write_text("OPENSSL_VERSION = 'OpenSSL 3.5.9'\n# injected code\n")
    # Simulate a forged identity result. The real archive/prefix checker must
    # reject the altered source before this subprocess could be invoked.
    monkeypatch.setattr(runtime, '_run', lambda *args, **kwargs: json.dumps(receipt['identity']))
    monkeypatch.setattr(attestor, 'verify_archive', lambda file: None)
    monkeypatch.setattr(runtime, '_load_python_attestor', lambda record, base: attestor, raising=False)
    rewrite(profile)
    with pytest.raises(runtime.ProfileError, match='Python runtime'):
        runtime._python(doc, path.parent)
