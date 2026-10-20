from __future__ import annotations

import hashlib
import json
import platform
import sys
import zipfile
from pathlib import Path

import pytest

from scripts import native_runtime as runtime


def test_execute_verifies_then_uses_the_installed_policy_supervisor(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(runtime, "verify", lambda *args: events.append("verified"))
    monkeypatch.setattr(runtime, "launch_environment", lambda: {"POLARS_MAX_THREADS": "2"})
    def execve(path, argv, env):
        events.append((path, argv, env))
    monkeypatch.setattr(runtime.os, "execve", execve)
    generation = tmp_path / "generation"
    runtime.execute(tmp_path / "profile", generation, ["-m", "biominer.cli", "--help"])
    python = str(generation / "env/bin/python")
    assert events == ["verified", (
        python, [python, "-I", "-B", "-m", "biominer.polars_runtime", "--",
                 "-m", "biominer.cli", "--help"], {"POLARS_MAX_THREADS": "2"},
    )]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blob(root, name, text='input'):
    path = root / name
    path.write_text(text)
    return {'path': str(path), 'sha256': digest(path)}


def arrow_fixture_pins():
    """Tiny synthetic inputs; never substitute these pins in a real profile."""
    return {
        'ARROW_ARCHIVE_SHA256': hashlib.sha256(b'fixture Arrow archive').hexdigest(),
        'ARROW_RECIPE_SHA256': hashlib.sha256(b'fixture Arrow recipe').hexdigest(),
        'ARROW_SECURITY_PATCH_SHA256': hashlib.sha256(b'fixture Arrow patch').hexdigest(),
        'ARROW_OFFSET_PATCH_SHA256': hashlib.sha256(b'fixture Arrow offset patch').hexdigest(),
        'ARROW_SECURITY_FILES': {
            'cpp/src/arrow/compute/kernels/vector_replace.cc': {
                'original_sha256': 'a' * 64,
                'patched_sha256': hashlib.sha256(b'fixture patched kernel').hexdigest(),
            },
            'cpp/src/arrow/compute/kernels/vector_replace_test.cc': {
                'original_sha256': 'b' * 64,
                'patched_sha256': hashlib.sha256(b'fixture patched regression').hexdigest(),
            },
        },
        'ARROW_OFFSET_FILES': {
            'cpp/src/arrow/compute/kernels/copy_data_internal.h': {
                'original_sha256': 'c' * 64,
                'patched_sha256': hashlib.sha256(b'fixture patched copy header').hexdigest(),
            },
            'cpp/src/arrow/compute/kernels/vector_replace_test.cc': {
                'original_sha256': hashlib.sha256(b'fixture patched regression').hexdigest(),
                'patched_sha256': hashlib.sha256(b'fixture final regression').hexdigest(),
            },
        },
    }


def arrow_build_fixture(root):
    archive = blob(root, 'arrow-source.tar', 'fixture Arrow archive')
    recipe = blob(root, 'build-recipe.py', 'fixture Arrow recipe')
    patch = blob(root, 'arrow-security.patch', 'fixture Arrow patch')
    offset_patch = blob(root, 'arrow-offset.patch', 'fixture Arrow offset patch')
    files = arrow_fixture_pins()['ARROW_SECURITY_FILES']
    offset_files = arrow_fixture_pins()['ARROW_OFFSET_FILES']
    sources = [blob(root, 'vector_replace.cc', 'fixture patched kernel'),
               blob(root, 'vector_replace_test.cc', 'fixture final regression'),
               blob(root, 'copy_data_internal.h', 'fixture patched copy header')]
    application = {
        'schema_version': 1,
        'source_revision': 'beccec0d0c451b7aa3e4530416ac431b3c035c69',
        'patches': [{
            'commit': 'a769c291e01093b73d03a075179cf7a09bf92ad8',
            'patch': patch,
            'files': [{'relative': name, **pins} for name, pins in files.items()],
        }, {
            'commit': '0efbdec7a10da1a4043fc1210c61e73c88bc9b4f',
            'patch': offset_patch,
            'files': [{'relative': name, **pins} for name, pins in offset_files.items()],
        }],
        'final_sources': [{'relative': name, 'sha256': pins['patched_sha256']}
                          for name, pins in (files | offset_files).items()],
    }
    patch_application = blob(root, 'arrow-security-patches.json', json.dumps(application))
    build = {
        'schema_version': 1, 'arrow_commit': application['source_revision'],
        'arrow_archive_sha256': archive['sha256'], 'source_archive': archive,
        'recipe': recipe, 'security_patches': application,
        'patch_application': patch_application, 'patched_sources': sources,
    }
    build_inputs = blob(root, 'build-inputs.json',
                        json.dumps(build, sort_keys=True, separators=(',', ':')) + '\n')
    version = '25.0.1+biominer.' + build_inputs['sha256'][:12]
    return {
        'source_revision': application['source_revision'],
        'recipe': recipe, 'patch_application': patch_application, 'build_inputs': build_inputs,
        'version': version, 'effective_versions': {'pyarrow': version, 'openssl': '3.6.5'},
        'inputs': [archive, recipe, patch, offset_patch, patch_application, build_inputs, *sources],
    }


def wheel(root, name, version):
    arrow = arrow_build_fixture(root) if name == 'pyarrow' else None
    if arrow:
        version = arrow['version']
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
    if arrow:
        value = json.loads(Path(receipt['path']).read_text()) | arrow
        value['inputs_sha256'] = hashlib.sha256(json.dumps(
            value['inputs'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        receipt = blob(root, name + '.receipt.json', json.dumps(value))
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
    for name, value in arrow_fixture_pins().items():
        monkeypatch.setattr(runtime, name, value, raising=False)
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


def test_arrow_chain_keeps_intermediate_source_hash_historical(profile):
    artifact = profile[1]['artifacts'][1]
    receipt = json.loads(Path(artifact['source_receipt']['path']).read_text())
    application = json.loads(Path(receipt['patch_application']['path']).read_text())
    first, second = application['patches']
    relative = 'cpp/src/arrow/compute/kernels/vector_replace_test.cc'
    intermediate = next(row['patched_sha256'] for row in first['files']
                        if row['relative'] == relative)
    before = next(row['original_sha256'] for row in second['files']
                  if row['relative'] == relative)
    assert intermediate == before
    assert intermediate not in {row['sha256'] for row in receipt['inputs']}
    assert len(application['final_sources']) == 3
    assert runtime.verify_artifacts(profile[0])['ready'] is False


def test_arrow25_receipt_requires_reviewed_security_patch(profile):
    # The historical build has the expected version and OpenSSL but no
    # evidence that the vulnerable chunked-boolean source was corrected.
    artifact = profile[1]['artifacts'][1]
    receipt = json.loads(Path(artifact['source_receipt']['path']).read_text())
    receipt.pop('patch_application')
    replace_arrow_receipt(profile, receipt)
    with pytest.raises(runtime.ProfileError, match='Arrow|arrow'):
        runtime.verify_artifacts(profile[0])


def replace_arrow_receipt(profile, receipt):
    artifact = profile[1]['artifacts'][1]
    path = Path(artifact['source_receipt']['path'])
    receipt['inputs_sha256'] = hashlib.sha256(json.dumps(
        receipt['inputs'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    path.write_text(json.dumps(receipt))
    artifact['source_receipt']['sha256'] = digest(path)
    rewrite(profile)


@pytest.mark.parametrize('damage', [
    'source_revision', 'patch_commit', 'patch_digest', 'original_source',
    'patched_source', 'missing_kernel', 'duplicate_kernel', 'patch_not_retained',
    'source_not_retained', 'recipe_not_retained', 'unreviewed_recipe',
    'build_inputs_not_retained', 'build_source_revision', 'build_archive',
    'build_recipe', 'build_patch', 'build_sources', 'malformed_build_source',
    'build_version', 'effective_version', 'allocation_only', 'offset_only',
    'reversed_patches', 'duplicate_patch', 'offset_commit', 'offset_digest',
    'offset_original_source', 'offset_patched_source', 'offset_not_retained',
    'offset_source_not_retained', 'final_sources_missing', 'final_sources_duplicate',
    'final_sources_wrong_path', 'final_sources_intermediate', 'build_sources_intermediate',
])
def test_arrow25_security_evidence_cannot_be_relabelled(profile, damage):
    artifact = profile[1]['artifacts'][1]
    receipt = json.loads(Path(artifact['source_receipt']['path']).read_text())
    application_path = Path(receipt['patch_application']['path'])
    application = json.loads(application_path.read_text())
    build_path = Path(receipt['build_inputs']['path'])
    build = json.loads(build_path.read_text())
    patch = application['patches'][0]

    if damage == 'source_revision':
        receipt['source_revision'] = 'f' * 40
    elif damage == 'patch_commit':
        patch['commit'] = 'f' * 40
    elif damage == 'patch_digest':
        patch['patch'] = receipt['recipe']
    elif damage in ('original_source', 'patched_source'):
        patch['files'][0][damage.replace('_source', '_sha256')] = 'f' * 64
    elif damage == 'missing_kernel':
        patch['files'].pop(0)
    elif damage == 'duplicate_kernel':
        patch['files'].append(patch['files'][0])
    elif damage == 'offset_not_retained':
        key = application['patches'][1]['patch']['sha256']
        receipt['inputs'] = [item for item in receipt['inputs'] if item['sha256'] != key]
    elif damage == 'offset_source_not_retained':
        key = arrow_fixture_pins()['ARROW_OFFSET_FILES'][
            'cpp/src/arrow/compute/kernels/copy_data_internal.h']['patched_sha256']
        receipt['inputs'] = [item for item in receipt['inputs'] if item['sha256'] != key]
    elif damage.endswith('_not_retained'):
        key = {'patch': patch['patch'], 'source': build['patched_sources'][0],
               'recipe': receipt['recipe'], 'build_inputs': receipt['build_inputs']}[
                   damage.removesuffix('_not_retained')]
        receipt['inputs'] = [item for item in receipt['inputs']
                             if item['sha256'] != key['sha256']]
    elif damage == 'unreviewed_recipe':
        receipt['recipe'] = patch['patch']
    elif damage == 'build_source_revision':
        build['arrow_commit'] = 'f' * 40
    elif damage == 'build_archive':
        build['source_archive'] = receipt['recipe']
    elif damage == 'build_recipe':
        build['recipe'] = patch['patch']
    elif damage == 'build_patch':
        build['security_patches'] = {}
    elif damage == 'build_sources':
        build['patched_sources'] = []
    elif damage == 'malformed_build_source':
        build['patched_sources'][0]['sha256'] = ['not-a-digest']
    elif damage == 'build_version':
        receipt['version'] = '25.0.1+biominer.' + 'f' * 12
    elif damage == 'effective_version':
        receipt['effective_versions']['pyarrow'] = '25.0.1'
    elif damage == 'allocation_only':
        application['patches'] = application['patches'][:1]
        application.pop('final_sources', None)
    elif damage == 'offset_only':
        application['patches'] = application['patches'][1:]
    elif damage == 'reversed_patches':
        application['patches'].reverse()
    elif damage == 'duplicate_patch':
        application['patches'].append(application['patches'][1])
    elif damage == 'offset_commit':
        application['patches'][1]['commit'] = 'f' * 40
    elif damage == 'offset_digest':
        application['patches'][1]['patch'] = patch['patch']
    elif damage in ('offset_original_source', 'offset_patched_source'):
        # The overlap must continue from the first patch's exact output.
        application['patches'][1]['files'][1][
            damage.removeprefix('offset_').replace('_source', '_sha256')] = 'f' * 64
    elif damage == 'final_sources_missing':
        application.pop('final_sources')
    elif damage == 'final_sources_duplicate':
        application['final_sources'][2] = application['final_sources'][1]
    elif damage == 'final_sources_wrong_path':
        application['final_sources'][0]['relative'] = 'unreviewed/source.cc'
    elif damage == 'final_sources_intermediate':
        application['final_sources'][1]['sha256'] = patch['files'][1]['patched_sha256']
    elif damage == 'build_sources_intermediate':
        build['patched_sources'][1]['sha256'] = patch['files'][1]['patched_sha256']

    # Refresh envelopes and wheel metadata so each rejection exercises the
    # changed policy field, rather than an incidental stale envelope/version.
    def refresh(descriptor, path, value):
        old = descriptor['sha256']
        path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n')
        descriptor['sha256'] = digest(path)
        for item in receipt['inputs']:
            if item['sha256'] == old:
                item['sha256'] = descriptor['sha256']
    refresh(receipt['patch_application'], application_path, application)
    build['patch_application'] = receipt['patch_application']
    if damage != 'build_patch':
        build['security_patches'] = application
    refresh(receipt['build_inputs'], build_path, build)
    version = '25.0.1+biominer.' + receipt['build_inputs']['sha256'][:12]
    artifact['version'] = version
    if damage != 'build_version':
        receipt['version'] = version
    if damage != 'effective_version':
        receipt['effective_versions']['pyarrow'] = version
    wheel_path = Path(artifact['path'])
    with zipfile.ZipFile(wheel_path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(wheel_path, 'w') as archive:
        for name, data in members.items():
            if name.endswith('.dist-info/METADATA'):
                data = f'Metadata-Version: 2.1\nName: pyarrow\nVersion: {version}\n'.encode()
            archive.writestr(name, data)
    artifact['sha256'] = receipt['wheel_sha256'] = digest(wheel_path)
    replace_arrow_receipt(profile, receipt)
    with pytest.raises(runtime.ProfileError, match='Arrow|arrow'):
        runtime.verify_artifacts(profile[0])


def test_arrow_security_materials_survive_input_preservation(profile, tmp_path):
    from scripts import assemble_native_profile

    artifact = profile[1]['artifacts'][1]
    receipt = json.loads(Path(artifact['source_receipt']['path']).read_text())
    preserved = assemble_native_profile.preserve_inputs(receipt, tmp_path / 'retained')
    for item in receipt['inputs']:
        Path(item['path']).unlink()
    replace_arrow_receipt(profile, preserved)

    assert runtime.verify_artifacts(profile[0])['ready'] is False


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
