from __future__ import annotations

import hashlib
import io
import tarfile
import json
import platform
import sys
import zipfile
import copy
import shutil
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


def test_arrow26_family_reaches_its_own_receipt_policy(profile):
    """Selecting 26 must not run the preserved Arrow 25 backport policy."""
    artifact = profile[1]['artifacts'][1]
    artifact['version'] = '26.0.0+biominer.0123456789ab'
    path = Path(artifact['path'])
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('pyarrow.dist-info/METADATA',
                        'Metadata-Version: 2.1\nName: pyarrow\nVersion: ' + artifact['version'] + '\n')
        archive.writestr('pyarrow.dist-info/WHEEL',
                        'Wheel-Version: 1.0\nTag: cp314-cp314-linux_x86_64\n')
    receipt = json.loads(Path(artifact['source_receipt']['path']).read_text())
    artifact['sha256'] = receipt['wheel_sha256'] = digest(path)
    replace_arrow_receipt(profile, receipt)

    with pytest.raises(runtime.ProfileError, match='Arrow 26.*raw receipt'):
        runtime.verify_artifacts(profile[0])


def test_private_arrow25_family_has_a_separate_closed_admission_policy(tmp_path):
    with pytest.raises(runtime.ProfileError, match='Arrow 25.*raw receipt'):
        runtime._arrow25_network_security({'inputs': []}, tmp_path, '25.0.1+biominer.0123456789ab')


@pytest.mark.parametrize('family,kind,recipe_sha256', [
    ('arrow26-integrated-private-network-v3', '26',
     '1bc69557acd83f150040f5001fd8243b6bcf98250023990e6d22e179cf6264b1'),
    ('arrow25-two-compute-patches-private-network-v2', '25',
     '4f7df429cc94697f155b7e252a8d0241225b1ea5800319f44bb6fb7b08da9d9b'),
])
def test_cohort_arrow_family_has_distinct_closed_pins_and_schema_kind(family, kind, recipe_sha256):
    pins = runtime._arrow_private_pins(family)
    assert pins['schema_kind'] == kind
    assert pins['recipe_sha256'] == recipe_sha256
    assert pins['raw_sha256'] is None
    assert pins['attestation_sha256'] is None
    assert runtime.ARROW26_BUILD_RECEIPT_SHA256 is None
    assert runtime.ARROW25_NETWORK_BUILD_RECEIPT_SHA256 is None
    assert runtime.ARROW26_RECIPE_SHA256 == '3bd6d35656f89fa71d1eaccd63477f5a94fbe074a56368af5b372660bbeac567'
    assert runtime.ARROW25_NETWORK_RECIPE_SHA256 == 'c73a4e5b11c3abea2391a65587362c073c6b97ee6f36c83bc8e0c909ff870a4d'


def test_private_arrow25_rebuild_rejects_before_inputs_probes_or_outputs(profile, tmp_path, monkeypatch):
    artifact = profile[1]['artifacts'][1]
    path = Path(artifact['source_receipt']['path'])
    receipt = json.loads(path.read_bytes())
    receipt['source_policy'] = 'arrow25-two-compute-patches-private-network-v1'
    receipt['arrow25_network_build_receipt'] = {'path': '/historical/raw.json', 'sha256': '0' * 64}
    path.write_text(json.dumps(receipt))
    artifact['source_receipt']['sha256'] = digest(path)
    monkeypatch.setattr(runtime, '_inputs', lambda *args: pytest.fail('private25 rebuild reached legacy inputs'))
    monkeypatch.setattr(runtime, '_python', lambda *args: pytest.fail('private25 rebuild reached environment probe'))
    output = tmp_path / 'uncreated-rebuild'
    with pytest.raises(runtime.ProfileError, match='Arrow 25.*rebuild.*unsupported'):
        runtime.build(rewrite(profile), output)
    assert not output.exists()


def test_obsolete_arrow26_bundled_zlib_family_cannot_be_normalized(tmp_path, monkeypatch):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'obsolete-arrow26', monkeypatch)
    fixture['raw']['build']['source_policy'] = 'arrow26-integrated-v1'
    refresh_arrow26_fixture(fixture, monkeypatch)
    with pytest.raises(runtime.ProfileError, match='Arrow 26.*family|Arrow 26.*policy'):
        assembler.normalize_arrow26_receipt(fixture['raw'], Path(fixture['record']['path']))



def private_network_build_fixture(root, material, openssl, monkeypatch):
    """Tiny source-bound private prefix; its helper must never execute."""
    prefix = root / 'network/prefix'
    helper = material('private-network-helper.py', 'raise RuntimeError("admission executed a helper")\n')
    recipe = material('network/build-recipe.py', Path(helper['path']).read_text())
    private, archives = {}, []
    names = {'libcurl.a': ('libcurl', 'curl', '8.23.0'), 'libxml2.a': ('libxml2', 'libxml2', '2.15.4'),
             'libiconv.a': ('libiconv', 'libiconv', '1.19'), 'libcharset.a': ('libcharset', 'libiconv', '1.19'),
             'libz.a': ('zlib', 'zlib', '1.3.2'), 'libnghttp2.a': ('libnghttp2', 'nghttp2', '1.68.0-2ubuntu0.2')}
    paths = {'lib/' + name: values for name, values in names.items()}
    paths.update({'include/curl/curl.h': names['libcurl.a'], 'include/libxml2/libxml/parser.h': names['libxml2.a'],
                  'include/iconv.h': names['libiconv.a'], 'include/zlib.h': names['libz.a'],
                  'include/nghttp2/nghttp2.h': names['libnghttp2.a']})
    for relative, (component, package, version) in paths.items():
        reference = material('network/prefix/' + relative, 'approved private ' + relative)
        private[reference['path']] = {'sha256': reference['sha256'], 'component': component,
                                     'source_package': package, 'source_version': version}
        if relative.startswith('lib/'):
            archives.append(reference)
    original, final = b'original nghttp2 source\n', b'patched nghttp2 source\n'
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w:gz') as bundle:
        entry = tarfile.TarInfo('nghttp2-1.68.0/lib/source.c')
        entry.size = len(original)
        bundle.addfile(entry, io.BytesIO(original))
    archive = material('network/inputs/nghttp2.tar.gz', data.getvalue())
    source = material('network/sources/nghttp2-1.68.0/lib/source.c', final)
    source_root = str(root / 'network/sources/nghttp2-1.68.0')
    old_inventory = {'root': source_root, 'files': [{'relative': 'lib/source.c', 'kind': 'file',
                     'sha256': hashlib.sha256(original).hexdigest()}]}
    final_inventory = {'root': source_root, 'files': [{'relative': 'lib/source.c', 'kind': 'file',
                       'sha256': source['sha256']}]}
    patch = material('network/inputs/nghttp2-security.patch', 'reviewed Ubuntu security patch')
    command = material('network/nghttp2-patch-command.json', {'exit_code': 0, 'cleanup': 'terminal',
                       'log': material('network/nghttp2-patch.log', 'exact patch applied')})
    ngpatch = {'source_package': 'nghttp2', 'source_version': '1.68.0-2ubuntu0.2', 'archive': archive,
               'original_source': old_inventory, 'final_source': final_inventory,
               'debian': {'root': str(root / 'network/distribution/debian'), 'files': [{
                   'relative': 'patches/security.patch', 'kind': 'file',
                   'sha256': material('network/distribution/debian/patches/security.patch', 'retained distro patch')['sha256']}]},
               'stages': [{'name': 'distro-security', 'patch': patch, 'command': command,
                   'before_files': {'lib/source.c': hashlib.sha256(original).hexdigest()},
                   'after_files': {'lib/source.c': source['sha256']}}]}
    compiled = {name: material('network/build/' + name, 'source-bound nghttp2 ' + name)
                for name in ('public_header', 'version_template', 'generated_version_header',
                             'target_source_manifest')}
    compiled['compile_commands'] = material('network/build/nghttp2/compile_commands.json', [{
        'file': source['path'], 'directory': str(root / 'network/build/nghttp2'),
        'command': 'cc -DBUILDING_NGHTTP2 -DNGHTTP2_STATICLIB -fPIC -c ' + source['path']}])
    compiled.update(archive=next(item for item in archives if item['path'].endswith('libnghttp2.a')),
                    objects=[material('network/build/source.c.o', 'PIC nghttp2 object')],
                    compiled_sources=[source], installed=[compiled['public_header']])
    zlib_patch = material('network/inputs/zlib-df84af25.patch', 'reviewed zlib stale pointer patch')
    gzwrite = material('network/sources/zlib/gzwrite.c', 'patched private zlib')
    zlib = {'patch_commit': runtime.LXML_ZLIB_PATCH_COMMIT, 'patch': zlib_patch,
            'original_gzwrite_sha256': 'b' * 64, 'patched_gzwrite_sha256': gzwrite['sha256'],
            'source': str(root / 'network/sources/zlib'), 'files': [{
                'relative': 'gzwrite.c', 'kind': 'file', 'sha256': gzwrite['sha256']}]}
    commands = []
    for name in ('probe-whole-archive-pic-link', 'probe-charset-whole-archive-pic-link'):
        commands.append(material('network/' + name + '-command.json', {
            'exit_code': 0, 'cleanup': 'terminal', 'argv': ['cc', '-shared', '-Wl,-z,defs', '-Wl,-z,text', '-Wl,--whole-archive'],
            'log': material('network/' + name + '.log', 'whole archive PIC link succeeded')}))
    observation = {'curl': '8.23.0', 'openssl': '3.6.5', 'zlib': '1.3.2', 'libxml2': '2.15.4',
                   'libiconv': '1.19', 'nghttp2': '1.68.0',
                   **{key: True for key in ('http', 'https', 'http2', 'xml', 'xpath', 'encoding', 'gzip_xml')}}
    build = {'schema_version': 1, 'kind': 'private_network_static_build', 'status': 'compiled_unverified',
             'ready': False, 'recipe': recipe, 'prefix': str(prefix),
             'openssl': {'prefix': str(root / 'openssl'), 'static_archives': openssl['static_archives']},
             'sources': {'nghttp2': final_inventory, 'zlib': {'root': str(root / 'network/sources/zlib'),
                         'files': [{'relative': 'gzwrite.c', 'kind': 'file', 'sha256': gzwrite['sha256']}]}},
             'nghttp2_patch': ngpatch, 'nghttp2_build': compiled, 'zlib_patch': zlib,
             'commands': commands, 'linkage': {'dynamic_private_dependencies': []},
             'probe': {'observation': observation}, 'effective_versions': observation}
    receipt = material('network/build-receipt.json', build)
    files, seen = {}, set()

    def collect(value, key=''):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                files[value['path']] = {name: value[name] for name in ('path', 'sha256')}
                if value['path'].endswith('.json') and value['path'] not in seen:
                    seen.add(value['path'])
                    collect(json.loads(Path(value['path']).read_bytes()))
            if key != 'original_source' and isinstance(value.get('root'), str) and isinstance(value.get('files'), list):
                for row in value['files']:
                    if row.get('kind') == 'file':
                        path = str(Path(value['root']) / row['relative'])
                        files[path] = {'path': path, 'sha256': row['sha256']}
            for name, child in value.items():
                collect(child, name)
        elif isinstance(value, list):
            for child in value:
                collect(child, key)
    collect(build)
    files[receipt['path']] = receipt
    for path, metadata in private.items():
        files[path] = {'path': path, 'sha256': metadata['sha256']}
    network = {'receipt': receipt, 'helper': helper, 'prefix': str(prefix), 'build': build,
               'files': list(files.values()), 'private_files': private}
    for name, value in {'ARROW_NETWORK_HELPER_SHA256': helper['sha256'],
                        'ARROW_NETWORK_BUILD_RECEIPT_SHA256': receipt['sha256'],
                        'ARROW_NETWORK_ZLIB_PATCH_SHA256': zlib_patch['sha256'],
                        'ARROW_NETWORK_ZLIB_ORIGINAL_SHA256': 'b' * 64,
                        'ARROW_NETWORK_ZLIB_PATCHED_SHA256': gzwrite['sha256']}.items():
        monkeypatch.setattr(runtime, name, value)
    return network, archives


def arrow26_build_fixture(root, monkeypatch, *, family='26'):
    """Complete tiny synthetic 26 chain; these module-local pins admit no real build."""
    root.mkdir()

    def material(name, data):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            path.write_bytes(data)
            return {'path': str(path), 'sha256': digest(path)}
        if not isinstance(data, str):
            data = json.dumps(data, sort_keys=True, separators=(',', ':')) + '\n'
        path.write_text(data)
        return {'path': str(path), 'sha256': digest(path)}

    def inventory(directory):
        return [{'relative': p.relative_to(directory).as_posix(), 'type': 'file',
                 'sha256': digest(p)} for p in sorted(directory.rglob('*')) if p.is_file()]

    private25 = family == '25'
    revision = runtime.ARROW_SOURCE_REVISION if private25 else runtime.ARROW26_SOURCE_REVISION
    source_policy = runtime.ARROW25_NETWORK_FAMILY if private25 else runtime.ARROW26_PRIVATE_FAMILY
    source = root / ('arrow-' + revision)
    source.mkdir()
    integrated = {}
    for name in ('copy_data_internal.h', 'scalar_if_else.cc', 'vector_replace.cc',
                 'vector_replace_test.cc'):
        relative = 'cpp/src/arrow/compute/kernels/' + name
        payload = {'copy_data_internal.h': 'fixture patched copy header',
                   'vector_replace.cc': 'fixture patched kernel',
                   'vector_replace_test.cc': 'fixture final regression'}.get(name, 'synthetic integrated ' + name)
        integrated[relative] = material(source.relative_to(root).as_posix() + '/' + relative,
                                        payload if private25 else 'synthetic integrated ' + name)['sha256']
    transformations = []
    for relative in ('cpp/cmake_modules/ThirdpartyToolchain.cmake', 'cpp/thirdparty/versions.txt'):
        item = material(source.relative_to(root).as_posix() + '/' + relative,
                        'synthetic reviewed transformed ' + relative)
        transformations.append({'relative': relative, 'original_sha256': 'a' * 64,
                                'sha256': item['sha256']})
    archive = material('arrow.tar.gz', 'synthetic Arrow 26 archive')
    recipe = material('build-recipe.py', 'synthetic exact reviewed Arrow26 recipe')
    backend = material(source.relative_to(root).as_posix() + '/python/pyproject.toml',
                       'build-backend = "_build_backend"\nrequires = ["scikit-build-core"]\n')
    source_document = {
        'schema_version': 1, 'source_revision': revision, 'archive': archive,
        'transformations': transformations, 'files': inventory(source),
    }
    if private25:
        for name, value in arrow_fixture_pins().items():
            monkeypatch.setattr(runtime, name, value)
        stages, finals = [], {}
        for commit, filename, contents, files in (
                (runtime.ARROW_SECURITY_PATCH_COMMIT, 'arrow-a769c291.patch', 'fixture Arrow patch', runtime.ARROW_SECURITY_FILES),
                (runtime.ARROW_OFFSET_PATCH_COMMIT, 'arrow-0efbdec7.patch', 'fixture Arrow offset patch', runtime.ARROW_OFFSET_FILES)):
            patch_record = material(filename, contents)
            stages.append({'commit': commit, 'patch': patch_record,
                           'files': [{'relative': key, **value} for key, value in files.items()]})
            finals.update({key: value['patched_sha256'] for key, value in files.items()})
        security = {'schema_version': 1, 'source_revision': revision, 'patches': stages,
                    'final_sources': [{'relative': key, 'sha256': value} for key, value in sorted(finals.items())]}
        patch_application = material('arrow-security-patches.json', security)
        source_document.update(source_policy=source_policy, security_patches=security,
            backend={'path': 'python/pyproject.toml', 'sha256': backend['sha256'],
                     'backend': '_build_backend', 'implementation': 'scikit-build-core'})
    else:
        source_document['integrated_sources'] = [{'relative': k, 'sha256': v} for k, v in sorted(integrated.items())]
    source_verification = material('source-verification.json', source_document)
    toolchain = {name: material('tools/' + name, 'synthetic approved ' + name)
                 for name in ('cc', 'cxx', 'cmake', 'ninja', 'ld')}
    build_python = material('builder/bin/python', 'synthetic approved Python')
    builder_pins = {'build': '1.6.1', 'cython': '3.3.0', 'libcst': '1.9.0',
                    'numpy': '2.5.4', 'packaging': '26.3', 'pathspec': '1.1.1',
                    'pyproject-hooks': '1.3.3', 'pyyaml': '6.0.3',
                    'scikit-build-core': '1.1.1', 'setuptools': '84.2.1',
                    'setuptools-scm': '10.3.4', 'vcs-versioning': '2.6.0'}
    distributions = []
    for name, version in builder_pins.items():
        distributions.append({'name': name, 'version': version,
            'metadata': material('builder/' + name + '/METADATA',
                                 f'Name: {name}\nVersion: {version}\n'),
            'record': material('builder/' + name + '/RECORD', 'synthetic package record'),
            'wheel': material('builder-wheels/' + name + '.whl', 'synthetic builder wheel ' + name)})
    builder = {'schema_version': 1, 'kind': 'arrow26_python_builder', 'status': 'frozen_verified',
               'python': build_python, 'base_python': build_python,
               'requirements': material('requirements.txt', 'synthetic pinned requirements'),
               'inventory': material('builder-inventory.json', {'schema_version': 1,
                   'root': str(root / 'builder'), 'files': inventory(root / 'builder'),
                   'directories': [], 'symlinks': []}),
               'verification': material('builder-verification.json', {'status': 'passed'}),
               'distributions': distributions, 'toolchain': toolchain}
    builder['receipt'] = material('builder.json', builder)
    openssl_archives = [material('openssl/lib/' + name, 'synthetic approved ' + name)
                        for name in ('libcrypto.a', 'libssl.a')]
    openssl = {'version': '3.6.5', 'source_commit': 'c8bd5a57108599ac650bbae77fcabe3109dab2e8',
               'receipt': material('openssl-receipt.json', {'synthetic': True}),
               'source_archive': material('openssl.tar.gz', 'synthetic OpenSSL source'),
               'header': material('openssl/include/openssl/opensslv.h', 'OpenSSL 3.6.5'),
               'headers': inventory(root / 'openssl/include'), 'static_archives': openssl_archives}
    dependencies = {}
    forced, trees = {}, []
    selected_versions = {'boost': '1.92.0'} if private25 else {
        'simdjson': '4.6.11', 'uriparser': '1.0.2', 'boost': '1.92.0', 'zlib': '1.3.2'}
    for name, version in selected_versions.items():
        descriptor = material('inputs/' + name + '.tar.gz', 'synthetic ' + name + ' archive')
        dependencies[name] = {'version': version, 'archive': descriptor, 'sha256': descriptor['sha256']}
        if not private25:
            dependencies[name]['role'] = 'retained_source_pin_only' if name == 'zlib' else 'bundled_provider'
        if name == 'zlib':
            continue
        directory = root / 'cpp-build' / name
        material('cpp-build/' + name + '/' + ('gzwrite.c' if name == 'zlib' else 'source.cc'),
                 'synthetic final ' + name + ' source')
        forced[name] = inventory(directory)
        trees.append({'path': str(directory), 'files': inventory(directory)})
    network, network_archives = private_network_build_fixture(root, material, openssl, monkeypatch)
    provider_cmake = material('private-network.cmake', runtime._arrow_private_cmake(
        Path(network['prefix']), root / 'openssl'))
    context = {'work': str(root), 'python': build_python['path'],
               'cc': toolchain['cc']['path'], 'cxx': toolchain['cxx']['path'],
               'openssl_prefix': str(root / 'openssl'), 'network_prefix': network['prefix']}
    features = ['ACERO', 'AZURE', 'COMPUTE', 'CSV', 'DATASET', 'FILESYSTEM', 'FLIGHT',
                'GCS', 'HDFS', 'JSON', 'ORC', 'PARQUET', 'S3', 'SUBSTRAIT']
    cpp_flags = {**{'ARROW_' + name: 'ON' for name in features},
                 **{'ARROW_WITH_' + name: 'ON' for name in ('BROTLI', 'BZ2', 'LZ4', 'SNAPPY', 'ZLIB', 'ZSTD')},
                 **{'ARROW_' + name: 'OFF' for name in ('TESTING', 'BUILD_TESTS', 'BUILD_BENCHMARKS', 'BUILD_INTEGRATION', 'GANDIVA')},
                 **{name + '_SOURCE': 'BUNDLED' for name in (('Boost',) if private25 else ('simdjson', 'uriparser', 'Boost'))},
                 'ARROW_BUILD_SHARED': 'ON', 'ARROW_BUILD_STATIC': 'OFF',
                 'ARROW_DEPENDENCY_SOURCE': 'AUTO', 'ARROW_DEPENDENCY_USE_SHARED': 'OFF',
                 'ARROW_JEMALLOC': 'ON', 'ARROW_MIMALLOC': 'ON',
                 'PARQUET_REQUIRE_ENCRYPTION': 'ON', 'OPENSSL_USE_STATIC_LIBS': 'TRUE',
                 'OPENSSL_ROOT_DIR': str(root / 'openssl'),
                 'OPENSSL_CRYPTO_LIBRARY': openssl_archives[0]['path'],
                 'OPENSSL_SSL_LIBRARY': openssl_archives[1]['path'],
                 'CMAKE_C_COMPILER': toolchain['cc']['path'],
                 'CMAKE_CXX_COMPILER': toolchain['cxx']['path'],
                 'CMAKE_MAKE_PROGRAM': toolchain['ninja']['path'],
                 'Python3_EXECUTABLE': build_python['path'],
                 'Python_EXECUTABLE': build_python['path'],
                 'PYTHON_EXECUTABLE': build_python['path'],
                 **({} if private25 else {'URIPARSER_BUILD_TESTS': 'OFF'})}
    cpp_flags.update(runtime._arrow_private_flags(context))
    python_cmake_flags = {**{'PYARROW_' + name: 'ON' for name in (
        'ACERO', 'AZURE', 'DATASET', 'FLIGHT', 'GCS', 'HDFS', 'ORC', 'PARQUET', 'S3',
        'SUBSTRAIT', 'PARQUET_ENCRYPTION')}, 'PYARROW_GANDIVA': 'OFF',
        'PYARROW_BUNDLE_ARROW_CPP': 'ON', 'CMAKE_C_COMPILER': toolchain['cc']['path'],
        'CMAKE_CXX_COMPILER': toolchain['cxx']['path'],
        'Python3_EXECUTABLE': build_python['path'], 'Python_EXECUTABLE': build_python['path'],
        'PYTHON_EXECUTABLE': build_python['path']}
    python_cmake_flags.update(runtime._arrow_private_flags(context))
    for name in ('Arrow', 'ArrowCompute', 'ArrowAcero', 'ArrowDataset', 'ArrowFlight',
                 *(() if private25 else ('ArrowS3',)), 'ArrowSubstrait', 'Parquet'):
        python_cmake_flags[name + '_DIR'] = str(root / 'arrow-prefix/lib/cmake' / name)
    commands = {'cpp_configure': [toolchain['cmake']['path'], '-S', str(source / 'cpp'),
                                  '-B', str(root / 'cpp-build'), '-G', 'Ninja',
                                  *['-D' + k + '=' + v for k, v in cpp_flags.items()]],
                'cpp_build': [toolchain['cmake']['path'], '--build', str(root / 'cpp-build'),
                              '--target', 'install', '--parallel', '4'],
                'python_build': [build_python['path'], '-I', '-B', '-m', 'build', '--wheel',
                                 '--no-isolation', '--config-setting=build-dir=' + str(root / 'python-build'),
                                 '--config-setting=build.tool-args=-j4',
                                 *['--config-setting=cmake.define.' + k + '=' + v
                                   for k, v in python_cmake_flags.items()], str(source / 'python')]}
    python_features = {'PYARROW_WITH_' + name: 'ON' for name in features}
    python_features.update(PYARROW_WITH_GANDIVA='OFF', PYARROW_WITH_PARQUET_ENCRYPTION='ON',
                           PYARROW_BUNDLE_ARROW_CPP='ON')
    build = {'schema_version': 1, 'source_policy': source_policy, 'arrow_commit': revision,
             'source_archive': archive, 'source_verification': source_verification,
             'integrated_sources': [{'relative': k, 'sha256': v} for k, v in sorted(integrated.items())],
             'recipe': recipe, 'builder': builder, 'openssl': openssl,
             'dependencies': dependencies, 'network': network,
             ('private_network_cmake' if private25 else 'provider_cmake'): provider_cmake,
             'patch_tool': material('tools/patch', 'synthetic patch tool'),
             'cmake_file_api_query': material('cpp-build/.cmake/api/v1/query/codemodel-v2', ''),
             'compiler_budget': {'maximum_jobs': 4, 'outer_ninja_jobs': 4,
                                 'nested_cmake_jobs': 1, 'nested_make_jobs': 1},
             'features': features, 'codecs': ['BROTLI', 'BZ2', 'LZ4', 'SNAPPY', 'ZLIB', 'ZSTD'],
             'context': context, 'commands': commands, 'python_feature_environment': python_features,
             'toolchain_versions': {name: 'synthetic reviewed ' + name for name in toolchain},
             'version_environment': {'SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYARROW': '{wheel_version}'}}
    if private25:
        build.pop('integrated_sources')
        build.update(security_patches=security, patch_application=patch_application)
    inputs = material('build-inputs.json', build)
    version = ('25.0.1' if private25 else '26.0.0') + '+biominer.' + inputs['sha256'][:12]
    static = [{**row, 'provider': {'kind': 'private_openssl', 'version': '3.6.5',
                'receipt': openssl['receipt'], 'source_commit': openssl['source_commit']}}
              for row in openssl_archives]
    static.extend({**row, 'provider': {'kind': 'private_network', 'receipt': network['receipt'],
                   'helper': network['helper'], 'file': row,
                   **{key: network['private_files'][row['path']][key]
                      for key in ('component', 'source_package', 'source_version')}}} for row in network_archives)
    map_record = material('cpp-build/release/libarrow.so.' + ('2500' if private25 else '2600') + '.map', 'Linker script and memory map\n' +
                          ''.join(row['path'] + '(member.o)\n' for row in static))
    target_record = material('cpp-build/.cmake/api/v1/reply/target-arrow.json', {
        'id': 'arrow::@synthetic-scope', 'name': 'arrow', 'type': 'SHARED_LIBRARY',
        'paths': {'source': '.', 'build': '.'}})
    model = material('cpp-build/.cmake/api/v1/reply/codemodel.json', {
        'kind': 'codemodel', 'version': {'major': 2, 'minor': 9},
        'paths': {'source': str(source / 'cpp'), 'build': str(root / 'cpp-build')},
        'configurations': [{'targets': [{'id': 'arrow::@synthetic-scope', 'name': 'arrow',
                                        'jsonFile': 'target-arrow.json'}]}]})
    index = material('cpp-build/.cmake/api/v1/reply/index-synthetic.json', {
        'reply': {'codemodel-v2': {'jsonFile': 'codemodel.json'}}})
    compiled_dependencies = material('compiled-dependencies.json', {
        'schema_version': 1, 'stage': 'compiled', 'targets': [{
            'target': 'arrow', 'id': 'arrow::@synthetic-scope', 'kind': 'SHARED_LIBRARY',
            'source': str(source / 'cpp'), 'binary': str(root / 'cpp-build'),
            'record': target_record, 'artifacts': [], 'links': []}],
        'file_api': [index, model, target_record], 'native_inputs': static,
        'static_archives': static, 'source_trees': trees,
        'forced_bundled_source_verification': forced, 'downloaded_archives': [],
        'boost_selection': material('boost-selection.txt', 'algorithm;crc;numeric/conversion;scope_exit;throw_exception;tokenizer;locale;thread;uuid\n'),
        'boost_components': ['algorithm', 'crc', 'numeric/conversion', 'scope_exit', 'throw_exception', 'tokenizer', 'locale', 'thread', 'uuid'],
    })
    consumers = [{'path': str(source / relative), 'sha256': integrated[relative]}
                 for relative in ('cpp/src/arrow/compute/kernels/scalar_if_else.cc',
                                  'cpp/src/arrow/compute/kernels/vector_replace.cc')]
    compile_commands = material('compile_commands.json', [{'file': item['path'],
                                'directory': str(root / 'cpp-build'), 'command': 'synthetic compile'}
                               for item in consumers])
    compiled_sources = material('compiled-sources.json', {'files': inventory(source),
        'compile_commands': compile_commands, 'translation_units': consumers,
        ('patched_header_consumers' if private25 else 'integrated_header_consumers'): consumers})
    wheel_path = root / ('pyarrow-' + version + '-cp314-cp314-linux_x86_64.whl')
    with zipfile.ZipFile(wheel_path, 'w') as wheel_zip:
        wheel_zip.writestr('pyarrow.dist-info/METADATA', 'Name: pyarrow\nVersion: ' + version + '\n')
        wheel_zip.writestr('pyarrow.dist-info/WHEEL', 'Wheel-Version: 1.0\nTag: cp314-cp314-linux_x86_64\n')
        for library in ('arrow', 'arrow_compute', 'arrow_acero', 'arrow_dataset', 'arrow_flight',
                        *(() if private25 else ('arrow_s3',)), 'arrow_substrait', 'parquet'):
            wheel_zip.writestr('pyarrow/lib' + library + '.so.' + ('2500' if private25 else '2600'), b'\x7fELFsynthetic ' + library.encode())
    wheel_record = {'path': str(wheel_path), 'sha256': digest(wheel_path)}
    raw = {'schema_version': 1, 'status': 'compiled_unverified', 'name': 'pyarrow',
           'version': version, 'wheel': wheel_record, 'wheel_sha256': wheel_record['sha256'],
           'source_revision': revision, 'recipe': recipe, 'build_inputs': inputs,
           'source_verification': source_verification, 'source_policy': source_policy, 'network': network,
           'compiled_dependencies': compiled_dependencies, 'compiled_sources': compiled_sources,
           'native_link_artifacts': [map_record], 'final_native_link_artifacts': [map_record], 'linked_static_archives': static,
           'toolchain': toolchain, 'toolchain_versions': build['toolchain_versions'],
           'python_build_distributions': distributions,
           'effective_versions': {'pyarrow': version, 'openssl': '3.6.5',
                                  **{k: v['version'] for k, v in dependencies.items()}},
           'build': {**build, 'status': 'building', 'build_inputs': inputs, 'wheel_version': version,
                     'stages': [{'command': commands[name], 'exit_code': 0,
                                 'log': material(name + '.log', 'synthetic successful stage')}
                                for name in ('cpp_build', 'python_build')]},
           'native_files': [], 'linkage': {}}
    if private25:
        raw = {**build, **{key: value for key, value in raw.items() if key not in (
            'build', 'effective_versions', 'python_build_distributions', 'toolchain')},
            'kind': 'corrective_arrow25_private_network_build', 'ready': False,
            'wheel_version': version, 'stages': raw['build']['stages']}
    raw_record = material('build-receipt.json', raw)
    prefix_pin = 'ARROW25_NETWORK' if private25 else 'ARROW26'
    pins = {prefix_pin + '_ARCHIVE_SHA256': archive['sha256'], prefix_pin + '_RECIPE_SHA256': recipe['sha256'],
            prefix_pin + '_BUILD_RECEIPT_SHA256': raw_record['sha256'],
            'ARROW26_BUILDER_RECEIPT_SHA256': builder['receipt']['sha256'],
            prefix_pin + ('_SOURCES' if private25 else '_INTEGRATED_SOURCES'): integrated,
            prefix_pin + '_TRANSFORMATIONS': transformations,
            prefix_pin + '_DEPENDENCIES': {k: {'version': v['version'], 'sha256': v['sha256']}
                                           for k, v in dependencies.items()},
            'ARROW26_OPENSSL_PINS': {k: v['sha256'] for k, v in openssl.items()
                                     if k in ('receipt', 'source_archive', 'header')},
            'ARROW26_OPENSSL_STATIC': {Path(v['path']).name: v['sha256'] for v in openssl_archives}}
    for name, value in pins.items():
        monkeypatch.setattr(runtime, name, value, raising=False)
    fixture = {'root': root, 'raw': raw, 'record': raw_record, 'material': material,
               'pins': pins, 'source': source}
    private_python_arrow26_fixture(fixture, monkeypatch)
    return fixture


def test_complete_arrow26_raw_chain_is_normalized_and_admitted(profile, tmp_path, monkeypatch):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'arrow26', monkeypatch)
    wrapper = assembler.normalize_arrow26_receipt(fixture['raw'], Path(fixture['record']['path']))
    wrapper_path = assembler.preserve_receipt(wrapper, tmp_path / 'materials', 'pyarrow')
    profile[1]['artifacts'][1] = {'name': 'pyarrow', 'version': fixture['raw']['version'],
        **fixture['raw']['wheel'], 'source_receipt': assembler.file_record(wrapper_path)}

    assert runtime.verify_artifacts(rewrite(profile))['ready'] is False
    assert wrapper['arrow26_build_receipt'] == fixture['record']
    assert 'source_receipt' not in wrapper
    assert Path(fixture['record']['path']).read_bytes() == Path(
        wrapper['arrow26_build_receipt']['path']).read_bytes()


def refresh_arrow26_fixture(fixture, monkeypatch):
    """Refresh envelopes/local version so corruption reaches nested policies."""
    raw, material = fixture['raw'], fixture['material']
    private25 = raw.get('source_policy') in (runtime.ARROW25_NETWORK_FAMILY, runtime.ARROW25_COHORT_FAMILY)
    if private25:
        previous = json.loads(Path(raw['build_inputs']['path']).read_bytes())
        build = {key: raw[key] for key in previous}
        if 'python_runtime' in raw:
            build['python_runtime'] = raw['python_runtime']
    else:
        build = {k: v for k, v in raw['build'].items()
                 if k not in ('status', 'build_inputs', 'wheel_version', 'stages')}
    raw['build_inputs'] = material('build-inputs.json', build)
    history = raw if private25 else raw['build']
    history['build_inputs'] = raw['build_inputs']
    version = ('25.0.1' if private25 else '26.0.0') + '+biominer.' + raw['build_inputs']['sha256'][:12]
    raw['version'] = history['wheel_version'] = version
    if not private25:
        raw['effective_versions']['pyarrow'] = version
    wheel_path = Path(raw['wheel']['path'])
    with zipfile.ZipFile(wheel_path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(wheel_path, 'w') as archive:
        for name, value in members.items():
            if name.endswith('.dist-info/METADATA'):
                value = ('Name: pyarrow\nVersion: ' + version + '\n').encode()
            archive.writestr(name, value)
    raw['wheel']['sha256'] = raw['wheel_sha256'] = digest(wheel_path)
    fixture['record'] = material('build-receipt.json', raw)
    prefix = {runtime.ARROW25_COHORT_FAMILY: 'ARROW25_COHORT',
              runtime.ARROW26_COHORT_FAMILY: 'ARROW26_COHORT'}.get(
                  raw.get('source_policy'), 'ARROW25_NETWORK' if private25 else 'ARROW26')
    monkeypatch.setattr(runtime, prefix + '_BUILD_RECEIPT_SHA256',
                        fixture['record']['sha256'])


EXPECTED_COHORT_COMPONENTS = {
    'aws_sdk_cpp', 'aws_crt_cpp', 'aws_c_auth', 'aws_c_cal', 'aws_c_common',
    'aws_c_compression', 'aws_c_event_stream', 'aws_c_http', 'aws_c_io', 'aws_c_mqtt',
    'aws_c_s3', 'aws_c_sdkutils', 'aws_checksums', 's2n_tls', 'thrift', 'grpc',
    'protobuf', 'absl', 'cares', 'rapidjson', 'jemalloc',
}


def cohort_arrow_build_fixture(root, monkeypatch, kind, *, symlinks=False):
    """Exercise adapter trust/retention; the standalone verifier has its own proof tests."""
    fixture = arrow26_build_fixture(root, monkeypatch, family=kind)
    raw, material = fixture['raw'], fixture['material']
    family = runtime.ARROW25_COHORT_FAMILY if kind == '25' else runtime.ARROW26_COHORT_FAMILY
    prefix = 'ARROW25_COHORT' if kind == '25' else 'ARROW26_COHORT'
    old_prefix = 'ARROW25_NETWORK' if kind == '25' else 'ARROW26'
    build = raw if kind == '25' else raw['build']
    raw['source_policy'] = build['source_policy'] = family
    source = json.loads(Path(raw['source_verification']['path']).read_bytes())
    source['source_policy'] = family
    raw['source_verification'] = build['source_verification'] = material('source-verification.json', source)
    selected = json.loads(Path(raw['compiled_dependencies']['path']).read_bytes())
    sdk = copy.deepcopy(runtime.ARROW_SDK_COHORT)
    assert set(sdk) == EXPECTED_COHORT_COMPONENTS
    supplemental, closure = {}, []
    for name, metadata in sdk.items():
        source_root = (root / 'cpp-build/jemalloc_ep-prefix/src/jemalloc_ep' if name == 'jemalloc'
                       else root / 'cpp-build/_deps' / (name.replace('_', '-') + '-src'))
        source_file = material(str(source_root.relative_to(root)) + '/source.h', 'source for ' + name)
        links = {'alias.h': 'source.h', 'chain.h': 'alias.h'} if symlinks and name == 'grpc' else {}
        for relative, target in links.items():
            (source_root / relative).symlink_to(target)
        packed = io.BytesIO()
        with tarfile.open(fileobj=packed, mode='w:gz') as archive:
            data = Path(source_file['path']).read_bytes()
            member = tarfile.TarInfo(name + '/source.h')
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
            for relative, target in links.items():
                member = tarfile.TarInfo(name + '/' + relative)
                member.type, member.linkname = tarfile.SYMTYPE, target
                archive.addfile(member)
        archive_record = material('inputs/' + metadata['filename'], packed.getvalue())
        metadata['sha256'] = archive_record['sha256']
        build['dependencies'][name] = {**metadata, 'archive': archive_record, 'role': 'bundled_provider'}
        files = [{'relative': 'source.h', 'type': 'file', 'sha256': source_file['sha256']}]
        files.extend({'relative': relative, 'type': 'symlink', 'target': target}
                     for relative, target in links.items())
        files.sort(key=lambda row: row['relative'])
        supplemental[name] = {'kind': 'external_project' if name == 'jemalloc' else 'fetchcontent',
                              'archive': archive_record, 'source_root': str(source_root),
                              'files': files, 'generated_derivations': []}
        selected['source_trees'].append({'path': str(source_root), 'files': files})
        closure.extend((archive_record, source_file))
    raw['compiled_dependencies'] = material('compiled-dependencies.json', selected)
    if kind == '26':
        raw['effective_versions'].update({name: row['version'] for name, row in sdk.items()})
    build['commands']['cpp_configure'].extend('-D' + name + '_SOURCE=BUNDLED' for name in (
        'AWSSDK', 'gRPC', 'Protobuf', 'absl', 'c-ares', 'Thrift', 'jemalloc', 'RapidJSON'))
    if kind == '25':
        build['commands']['cpp_configure'].append('-DCUSTOM_MEMORY_MANAGEMENT=OFF')
    for suffix in ('ARCHIVE_SHA256', 'RECIPE_SHA256', 'TRANSFORMATIONS'):
        monkeypatch.setattr(runtime, prefix + '_' + suffix, getattr(runtime, old_prefix + '_' + suffix))
    monkeypatch.setattr(runtime, 'ARROW_SDK_COHORT', sdk)
    dependency_pins = {**getattr(runtime, old_prefix + '_DEPENDENCIES'), **sdk}
    monkeypatch.setattr(runtime, prefix + '_DEPENDENCIES', dependency_pins)
    refresh_arrow26_fixture(fixture, monkeypatch)
    cohort = material('cohort.json', {'sources': [dict(row, name=name,
        path=build['dependencies'][name]['archive']['path']) for name, row in sdk.items()]})
    monkeypatch.setattr(runtime, 'ARROW_COHORT_METADATA_SHA256', cohort['sha256'])
    # A deliberately tiny reviewed verifier fixture verifies closure hashes. The
    # production proof engine is independently tested with real archive trees.
    producer = material('source-verifier.py', '''import hashlib,json
from pathlib import Path
def verify_completed_attestation(path, expected_sha256, *, retained_inputs=None):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
        raise ValueError('attestation changed')
    doc=json.loads(path.read_bytes())
    index={row['sha256']:row for row in retained_inputs or []}
    files=[]
    for record in doc['closure']:
        selected=index.get(record['sha256']) if retained_inputs is not None else record
        if selected is None or hashlib.sha256(Path(selected['path']).read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('source closure changed')
        files.append(selected)
    return {'receipt':{'path':str(path),'sha256':expected_sha256},'attestation':doc,
            'files':files,'supplemental':doc['supplemental'],'legacy':doc['legacy']}
''')
    monkeypatch.setattr(runtime, 'ARROW_SOURCE_ATTESTER_SHA256', producer['sha256'])
    subject = {name: raw[name] for name in (
        'recipe', 'build_inputs', 'compiled_dependencies', 'compiled_sources', 'wheel')}
    subject['raw_receipt'] = fixture['record']
    document = {'schema_version': 1, 'kind': 'bundled_native_source_attestation',
        'status': 'source_attested', 'ready': False, 'schema_kind': kind, 'source_policy': family,
        'producer': producer, 'subject': subject, 'cohort': cohort,
        'legacy': selected['forced_bundled_source_verification'], 'supplemental': supplemental,
        'external_projects': {}, 'closure': [*closure, cohort, producer, *subject.values()]}
    attestation = material('source-attestation.json', document)
    monkeypatch.setattr(runtime, prefix + '_ATTESTATION_SHA256', attestation['sha256'])
    fixture.update(attestation=attestation, attestation_doc=document, prefix=prefix, kind=kind)
    return fixture


def retained_cohort_with_symlinks(tmp_path, monkeypatch):
    from scripts import assemble_native_profile as assembler
    fixture = cohort_arrow_build_fixture(tmp_path / 'historical', monkeypatch, '26', symlinks=True)
    wrapper = assembler.normalize_arrow26_receipt(fixture['raw'], Path(fixture['record']['path']),
        source_attestation=Path(fixture['attestation']['path']),
        source_attestation_sha256=fixture['attestation']['sha256'])
    wrapper = assembler.preserve_inputs(wrapper, tmp_path / 'retained')
    shutil.rmtree(fixture['root'])
    return fixture, wrapper


def test_cohort_arrow_retained_symlink_chain_never_traverses_historical_paths(tmp_path, monkeypatch):
    fixture, wrapper = retained_cohort_with_symlinks(tmp_path, monkeypatch)
    historical = fixture['root']
    for name in ('resolve', 'stat', 'lstat', 'readlink'):
        original = getattr(Path, name)

        def guarded(path, *args, _original=original, **kwargs):
            if path.is_relative_to(historical):
                pytest.fail('retained source proof traversed a historical path: ' + str(path))
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(Path, name, guarded)
    runtime._arrow_private_security(wrapper, tmp_path, fixture['raw']['version'], runtime.ARROW26_COHORT_FAMILY)


@pytest.mark.parametrize('damage', ['absolute', 'escaping', 'self_cycle', 'pair_cycle', 'escaping_chain'])
def test_cohort_arrow_retained_inventory_rejects_invalid_symlink_chains(tmp_path, monkeypatch, damage):
    from scripts import assemble_native_profile as assembler
    fixture, wrapper = retained_cohort_with_symlinks(tmp_path, monkeypatch)
    document = fixture['attestation_doc']
    files = document['supplemental']['grpc']['files']
    by_name = {row['relative']: row for row in files}
    if damage == 'absolute':
        by_name['alias.h']['target'] = '/outside/source.h'
    elif damage == 'escaping':
        by_name['alias.h']['target'] = '../outside/source.h'
    elif damage == 'self_cycle':
        by_name['alias.h']['target'] = 'alias.h'
    elif damage == 'pair_cycle':
        by_name['alias.h']['target'] = 'chain.h'
    else:
        files.extend([{'relative': 'a', 'type': 'directory'}, {'relative': 'c', 'type': 'directory'},
                      {'relative': 'a/b', 'type': 'symlink', 'target': '../c'}])
        by_name['alias.h']['target'] = 'a/b/../../outside'
    files.sort(key=lambda row: row['relative'])
    path = tmp_path / 'changed-proof.json'
    path.write_text(json.dumps(document))
    attestation = assembler.file_record(path)
    monkeypatch.setattr(runtime, fixture['prefix'] + '_ATTESTATION_SHA256', attestation['sha256'])
    wrapper['bundled_source_attestation'] = attestation
    wrapper['inputs'].append(attestation)
    with pytest.raises(runtime.ProfileError, match='symlink|source tree'):
        runtime._arrow_private_security(wrapper, tmp_path, fixture['raw']['version'], runtime.ARROW26_COHORT_FAMILY)


@pytest.mark.parametrize('kind', ['25', '26'])
@pytest.mark.parametrize('preserve', [False, True])
def test_complete_cohort_arrow_retains_separate_proof_and_raw_bytes(profile, tmp_path, monkeypatch, kind, preserve):
    from scripts import assemble_native_profile as assembler
    fixture = cohort_arrow_build_fixture(tmp_path / 'cohort', monkeypatch, kind)
    raw_bytes = Path(fixture['record']['path']).read_bytes()
    normalize = assembler.normalize_arrow25_network_receipt if kind == '25' else assembler.normalize_arrow26_receipt
    wrapper = normalize(fixture['raw'], Path(fixture['record']['path']),
        source_attestation=Path(fixture['attestation']['path']),
        source_attestation_sha256=fixture['attestation']['sha256'])
    assert wrapper['bundled_source_attestation'] == fixture['attestation']
    assert Path(fixture['record']['path']).read_bytes() == raw_bytes
    forced = fixture['attestation_doc']['legacy']
    assert set(forced) == ({'boost'} if kind == '25' else {'simdjson', 'uriparser', 'boost'})
    assert set(fixture['attestation_doc']['supplemental']) == EXPECTED_COHORT_COMPONENTS
    if preserve:
        historical = copy.deepcopy(wrapper['inputs'])
        wrapper = assembler.preserve_inputs(wrapper, tmp_path / 'retained')
        for row in historical:
            Path(row['path']).unlink(missing_ok=True)
    wheel_record = (next(row for row in wrapper['inputs']
                         if row['sha256'] == fixture['raw']['wheel_sha256'])
                    if preserve else fixture['raw']['wheel'])
    wrapper_path = assembler.preserve_receipt(wrapper, tmp_path / 'wrappers', 'pyarrow')
    profile[1]['artifacts'][1] = {'name': 'pyarrow', 'version': fixture['raw']['version'],
        **wheel_record, 'source_receipt': assembler.file_record(wrapper_path)}
    assert runtime.verify_artifacts(rewrite(profile))['ready'] is False


@pytest.mark.parametrize('kind', ['25', '26'])
@pytest.mark.parametrize('damage', ['family', 'subject', 'missing', 'extra', 'legacy', 'archive', 'source', 'producer'])
def test_cohort_arrow_attestation_cannot_relabel_or_omit_inputs(tmp_path, monkeypatch, kind, damage):
    from scripts import assemble_native_profile as assembler
    fixture = cohort_arrow_build_fixture(tmp_path / 'cohort', monkeypatch, kind)
    doc = fixture['attestation_doc']
    if damage == 'family':
        doc['source_policy'] = runtime.ARROW25_COHORT_FAMILY if kind == '26' else runtime.ARROW26_COHORT_FAMILY
    elif damage == 'subject':
        doc['subject']['wheel'] = doc['subject']['recipe']
    elif damage == 'missing':
        doc['supplemental'].pop('grpc')
    elif damage == 'extra':
        doc['supplemental']['unreviewed'] = doc['supplemental']['grpc']
    elif damage == 'legacy':
        doc['legacy'] = {}
    elif damage == 'archive':
        doc['supplemental']['grpc']['archive'] = doc['supplemental']['protobuf']['archive']
    elif damage == 'source':
        Path(doc['supplemental']['grpc']['source_root'], 'source.h').write_text('changed original source')
    else:
        Path(doc['producer']['path']).write_text('raise RuntimeError("unreviewed verifier executed")\n')
    fixture['attestation'] = fixture['material']('source-attestation.json', doc)
    monkeypatch.setattr(runtime, fixture['prefix'] + '_ATTESTATION_SHA256', fixture['attestation']['sha256'])
    normalize = assembler.normalize_arrow25_network_receipt if kind == '25' else assembler.normalize_arrow26_receipt
    with pytest.raises(runtime.ProfileError):
        normalize(fixture['raw'], Path(fixture['record']['path']),
            source_attestation=Path(fixture['attestation']['path']),
            source_attestation_sha256=fixture['attestation']['sha256'])


@pytest.mark.parametrize('kind', ['25', '26'])
def test_cohort_arrow_rebuild_rejects_before_legacy_probes(profile, tmp_path, monkeypatch, kind):
    artifact = profile[1]['artifacts'][1]
    artifact['version'] = ('25.0.1' if kind == '25' else '26.0.0') + '+biominer.0123456789ab'
    path = Path(artifact['source_receipt']['path'])
    receipt = json.loads(path.read_bytes())
    receipt['source_policy'] = runtime.ARROW25_COHORT_FAMILY if kind == '25' else runtime.ARROW26_COHORT_FAMILY
    path.write_text(json.dumps(receipt));artifact['source_receipt']['sha256'] = digest(path)
    monkeypatch.setattr(runtime, '_inputs', lambda *args: pytest.fail('cohort rebuild reached legacy inputs'))
    monkeypatch.setattr(runtime, '_python', lambda *args: pytest.fail('cohort rebuild reached Python probe'))
    output = tmp_path / 'uncreated'
    with pytest.raises(runtime.ProfileError, match='rebuild.*unsupported'):
        runtime.build(rewrite(profile), output)
    assert not output.exists()


@pytest.mark.parametrize('damage', [
    'failed', 'source_revision', 'source_policy', 'recipe', 'archive',
    'integrated_source', 'zlib_patch', 'zlib_final_source', 'static_archive',
    'link_maps', 'budget', 'features', 'file_api_identity', 'header_consumer',
])
def test_arrow26_nested_evidence_cannot_be_relabelled(tmp_path, monkeypatch, damage):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'arrow26', monkeypatch)
    raw, material = fixture['raw'], fixture['material']
    if damage == 'failed':
        raw['status'] = 'failed'
    elif damage == 'source_revision':
        raw['source_revision'] = 'f' * 40
    elif damage == 'source_policy':
        raw['build']['source_policy'] = 'arrow25-backport'
    elif damage == 'recipe':
        raw['recipe'] = raw['build']['recipe'] = material('other-recipe.py', 'unreviewed recipe')
    elif damage == 'archive':
        raw['build']['source_archive'] = material('other.tar.gz', 'unreviewed source')
    elif damage == 'integrated_source':
        doc = json.loads(Path(raw['source_verification']['path']).read_bytes())
        doc['integrated_sources'].pop()
        raw['source_verification'] = raw['build']['source_verification'] = material('source-verification.json', doc)
    elif damage == 'zlib_patch':
        raw['build']['zlib_patch'] = material('other-zlib.patch', 'unreviewed zlib patch')
    elif damage == 'zlib_final_source':
        raw['build']['network']['build']['zlib_patch']['patched_gzwrite_sha256'] = 'f' * 64
    elif damage == 'static_archive':
        Path(raw['linked_static_archives'][-1]['path']).write_bytes(b'changed selected libz archive')
    elif damage == 'link_maps':
        raw['native_link_artifacts'] = []
    elif damage == 'budget':
        raw['build']['compiler_budget']['maximum_jobs'] = 8
    elif damage == 'features':
        raw['build']['features'].remove('FLIGHT')
    elif damage == 'file_api_identity':
        doc = json.loads(Path(raw['compiled_dependencies']['path']).read_bytes())
        doc['targets'][0]['id'] = 'arrow::@different-scope'
        raw['compiled_dependencies'] = material('compiled-dependencies.json', doc)
    elif damage == 'header_consumer':
        doc = json.loads(Path(raw['compiled_sources']['path']).read_bytes())
        doc['integrated_header_consumers'].pop()
        raw['compiled_sources'] = material('compiled-sources.json', doc)
    refresh_arrow26_fixture(fixture, monkeypatch)

    with pytest.raises(runtime.ProfileError, match='Arrow 26'):
        assembler.normalize_arrow26_receipt(raw, Path(fixture['record']['path']))


def test_arrow26_retains_nested_json_and_inventory_files_after_relocation(profile, tmp_path, monkeypatch):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'arrow26', monkeypatch)
    wrapper = assembler.normalize_arrow26_receipt(fixture['raw'], Path(fixture['record']['path']))
    old_inputs = copy.deepcopy(wrapper['inputs'])
    original_raw = Path(fixture['record']['path']).read_bytes()
    preserved = assembler.preserve_inputs(wrapper, tmp_path / 'retained')
    for item in old_inputs:
        Path(item['path']).unlink(missing_ok=True)
    wrapper_path = assembler.preserve_receipt(preserved, tmp_path / 'retained', 'pyarrow')
    profile[1]['artifacts'][1] = {'name': 'pyarrow', 'version': fixture['raw']['version'],
        **fixture['raw']['wheel'], 'source_receipt': assembler.file_record(wrapper_path)}

    assert runtime.verify_artifacts(rewrite(profile))['ready'] is False
    retained_raw = next(row for row in preserved['inputs']
                        if row['sha256'] == fixture['record']['sha256'])
    assert Path(retained_raw['path']).read_bytes() == original_raw


def test_arrow26_changed_retained_static_bytes_are_rejected(profile, tmp_path, monkeypatch):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'arrow26', monkeypatch)
    wrapper = assembler.normalize_arrow26_receipt(fixture['raw'], Path(fixture['record']['path']))
    preserved = assembler.preserve_inputs(wrapper, tmp_path / 'retained')
    libz = fixture['raw']['linked_static_archives'][-1]
    selected = next(row for row in preserved['inputs'] if row['sha256'] == libz['sha256'])
    Path(selected['path']).write_bytes(b'tampered retained static archive')
    wrapper_path = assembler.preserve_receipt(preserved, tmp_path / 'retained', 'pyarrow')
    profile[1]['artifacts'][1] = {'name': 'pyarrow', 'version': fixture['raw']['version'],
        **fixture['raw']['wheel'], 'source_receipt': assembler.file_record(wrapper_path)}

    with pytest.raises(runtime.ProfileError, match='SHA256 mismatch'):
        runtime.verify_artifacts(rewrite(profile))


@pytest.mark.parametrize('command,old,new', [
    ('cpp_build', '4', '16'),
    ('python_build', '--config-setting=build.tool-args=-j4', '--config-setting=build.tool-args=-j16'),
    ('cpp_configure', '-DARROW_S3=ON', '-DARROW_S3=OFF'),
    ('python_build', '--config-setting=cmake.define.PYARROW_PARQUET_ENCRYPTION=ON',
     '--config-setting=cmake.define.PYARROW_PARQUET_ENCRYPTION=OFF'),
])
def test_arrow26_commands_cannot_override_declared_features_or_budget(
        tmp_path, monkeypatch, command, old, new):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'arrow26', monkeypatch)
    build = fixture['raw']['build']
    argv = build['commands'][command]
    argv[argv.index(old)] = new
    if command != 'cpp_configure':
        stage = 0 if command == 'cpp_build' else 1
        build['stages'][stage]['command'] = argv
    refresh_arrow26_fixture(fixture, monkeypatch)

    with pytest.raises(runtime.ProfileError, match='Arrow 26'):
        assembler.normalize_arrow26_receipt(fixture['raw'], Path(fixture['record']['path']))


def private_python_arrow26_fixture(fixture, monkeypatch):
    raw, material = fixture['raw'], fixture['material']
    if 'python_runtime' in raw:
        return
    history = raw['build'] if 'build' in raw else raw
    python = history['builder']['base_python']
    library = material('python-runtime/libpython3.14.so.1.0', 'synthetic source-attested Python library')
    archive = material('python-runtime.tar.gz', 'synthetic exact approved Python archive')
    receipt = material('python-runtime-receipt.json', {'schema_version': 1,
        'source_build': 'Astral20260929', 'python': python, 'archive': archive,
        'verified_archive_files': [python, library]})
    runtime_record = {'source_build': 'Astral20260929', 'receipt': receipt,
                      'archive': archive, 'python': python, 'files': [library]}
    raw['python_runtime'] = history['python_runtime'] = runtime_record
    selected = json.loads(Path(raw['compiled_dependencies']['path']).read_bytes())
    selected['native_inputs'].append({**library, 'provider': {'kind': 'python_runtime',
        'source_build': 'Astral20260929', 'receipt': receipt}})
    raw['compiled_dependencies'] = material('compiled-dependencies.json', selected)
    monkeypatch.setattr(runtime, 'ARROW26_PYTHON_RUNTIME_PINS', {
        'receipt': receipt['sha256'], 'archive': archive['sha256'], 'python': python['sha256']},
        raising=False)
    refresh_arrow26_fixture(fixture, monkeypatch)


def test_arrow26_accepts_only_source_attested_private_python_provider(tmp_path, monkeypatch):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'arrow26', monkeypatch)
    private_python_arrow26_fixture(fixture, monkeypatch)

    wrapper = assembler.normalize_arrow26_receipt(fixture['raw'], Path(fixture['record']['path']))

    expected = fixture['raw']['python_runtime']['files'][0]
    assert any(item['sha256'] == expected['sha256'] for item in wrapper['inputs'])


def test_arrow26_provider_kind_cannot_claim_a_system_package(tmp_path, monkeypatch):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'arrow26', monkeypatch)
    selected = json.loads(Path(fixture['raw']['compiled_dependencies']['path']).read_bytes())
    selected['native_inputs'][0]['provider'] = {'kind': 'unreviewed_private',
        'package': 'libunknown', 'version': '1', 'source_package': 'libunknown', 'source_version': '1'}
    fixture['raw']['compiled_dependencies'] = fixture['material']('compiled-dependencies.json', selected)
    refresh_arrow26_fixture(fixture, monkeypatch)

    with pytest.raises(runtime.ProfileError, match='Arrow 26'):
        assembler.normalize_arrow26_receipt(fixture['raw'], Path(fixture['record']['path']))


@pytest.mark.parametrize('field', ['effective_versions', 'compiler_budget', 'toolchain_versions'])
def test_arrow26_malformed_record_objects_have_a_policy_error(tmp_path, monkeypatch, field):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'arrow26', monkeypatch)
    if field == 'effective_versions':
        fixture['raw'][field] = []
        fixture['record'] = fixture['material']('build-receipt.json', fixture['raw'])
        monkeypatch.setattr(runtime, 'ARROW26_BUILD_RECEIPT_SHA256', fixture['record']['sha256'])
    else:
        fixture['raw']['build'][field] = []
        if field == 'toolchain_versions':
            fixture['raw'][field] = []
        refresh_arrow26_fixture(fixture, monkeypatch)

    with pytest.raises(runtime.ProfileError, match='Arrow 26'):
        assembler.normalize_arrow26_receipt(fixture['raw'], Path(fixture['record']['path']))


@pytest.mark.parametrize('field', ['features', 'codecs', 'boost_components'])
def test_arrow26_unhashable_array_members_have_a_policy_error(tmp_path, monkeypatch, field):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'arrow26', monkeypatch)
    if field == 'boost_components':
        selected = json.loads(Path(fixture['raw']['compiled_dependencies']['path']).read_bytes())
        selected[field][0] = {}
        fixture['raw']['compiled_dependencies'] = fixture['material']('compiled-dependencies.json', selected)
    else:
        fixture['raw']['build'][field][0] = {}
    refresh_arrow26_fixture(fixture, monkeypatch)

    with pytest.raises(runtime.ProfileError, match='Arrow 26'):
        assembler.normalize_arrow26_receipt(fixture['raw'], Path(fixture['record']['path']))


@pytest.mark.parametrize('damage', ['missing_translation_path', 'non_string_private_archive_path'])
def test_arrow26_malformed_file_records_have_a_policy_error(tmp_path, monkeypatch, damage):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'arrow26', monkeypatch)
    if damage == 'missing_translation_path':
        compiled = json.loads(Path(fixture['raw']['compiled_sources']['path']).read_bytes())
        del compiled['translation_units'][0]['path']
        fixture['raw']['compiled_sources'] = fixture['material']('compiled-sources.json', compiled)
    else:
        fixture['raw']['build']['openssl']['static_archives'][0]['path'] = []
    refresh_arrow26_fixture(fixture, monkeypatch)

    with pytest.raises(runtime.ProfileError, match='Arrow 26'):
        assembler.normalize_arrow26_receipt(fixture['raw'], Path(fixture['record']['path']))


@pytest.mark.parametrize('damage', [
    'missing_openssl_prefix', 'non_string_openssl_prefix', 'file_api_path',
    'index_reply', 'codemodel_reference', 'model_version', 'target_entries',
    'target_id', 'entry_id', 'entry_json_file', 'binary_link_map', 'binary_translation_unit',
])
def test_arrow26_malformed_nested_metadata_has_a_policy_error(tmp_path, monkeypatch, damage):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'arrow26', monkeypatch)
    raw = fixture['raw']
    selected = json.loads(Path(raw['compiled_dependencies']['path']).read_bytes())

    def change_api(name, change):
        index = next(i for i, row in enumerate(selected['file_api']) if Path(row['path']).name == name)
        row = selected['file_api'][index]
        document = json.loads(Path(row['path']).read_bytes())
        change(document)
        selected['file_api'][index] = fixture['material'](
            str(Path(row['path']).relative_to(fixture['root'])), document)

    def binary_material(name):
        row = fixture['material'](name, 'synthetic placeholder')
        path = Path(row['path'])
        path.write_bytes(b'\xffbinary')
        return {**row, 'sha256': digest(path)}

    if damage == 'missing_openssl_prefix':
        del raw['build']['context']['openssl_prefix']
    elif damage == 'non_string_openssl_prefix':
        raw['build']['context']['openssl_prefix'] = []
    elif damage == 'file_api_path':
        selected['file_api'][0]['path'] = []
    elif damage == 'index_reply':
        change_api('index-synthetic.json', lambda doc: doc.update(reply=[]))
    elif damage == 'codemodel_reference':
        change_api('index-synthetic.json', lambda doc: doc['reply'].update({'codemodel-v2': []}))
    elif damage == 'model_version':
        change_api('codemodel.json', lambda doc: doc.update(version=[]))
    elif damage == 'target_entries':
        change_api('codemodel.json', lambda doc: doc['configurations'][0].update(targets={}))
    elif damage == 'target_id':
        selected['targets'][0]['id'] = []
    elif damage == 'entry_id':
        change_api('codemodel.json', lambda doc: doc['configurations'][0]['targets'][0].update(id={}))
    elif damage == 'entry_json_file':
        change_api('codemodel.json', lambda doc: doc['configurations'][0]['targets'][0].update(jsonFile=[]))
    elif damage == 'binary_link_map':
        raw['native_link_artifacts'][0] = binary_material('libarrow.map')
    elif damage == 'binary_translation_unit':
        compiled = json.loads(Path(raw['compiled_sources']['path']).read_bytes())
        unit = binary_material('cpp-build/unity.cxx')
        compiled['translation_units'].append(unit)
        commands = json.loads(Path(compiled['compile_commands']['path']).read_bytes())
        commands.append({'file': unit['path'], 'directory': str(fixture['root'] / 'cpp-build')})
        compiled['compile_commands'] = fixture['material']('compile_commands.json', commands)
        raw['compiled_sources'] = fixture['material']('compiled-sources.json', compiled)
    raw['compiled_dependencies'] = fixture['material']('compiled-dependencies.json', selected)
    refresh_arrow26_fixture(fixture, monkeypatch)

    with pytest.raises(runtime.ProfileError, match='Arrow 26'):
        assembler.normalize_arrow26_receipt(raw, Path(fixture['record']['path']))


@pytest.mark.parametrize('preserve', [False, True])
def test_assembly_normalizes_arrow26_in_both_input_modes(profile, tmp_path, monkeypatch, preserve):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'arrow26', monkeypatch)
    original_raw = Path(fixture['record']['path']).read_bytes()
    monkeypatch.setattr(assembler.shutil, 'which', lambda name: None)
    monkeypatch.setattr(assembler.subprocess, 'run', lambda *args, **kwargs:
                        runtime.subprocess.CompletedProcess(args, 0, stdout='synthetic host policy'))
    output = tmp_path / 'assembled.lock'

    assembler.assemble(tmp_path, output, tmp_path / 'absent-python', preserve=preserve,
                       arrow_receipt=Path(fixture['record']['path']))

    artifact = next(row for row in json.loads(output.read_bytes())['artifacts'] if row['name'] == 'pyarrow')
    assert artifact['version'] == fixture['raw']['version']
    assert artifact['source_receipt'] != fixture['record']
    wrapper = json.loads(Path(artifact['source_receipt']['path']).read_bytes())
    assert wrapper['arrow26_build_receipt'] == fixture['record']
    raw_input = next(row for row in wrapper['inputs'] if row['sha256'] == fixture['record']['sha256'])
    assert Path(raw_input['path']).read_bytes() == original_raw
    assert Path(fixture['record']['path']).read_bytes() == original_raw
    assert (raw_input['path'] != fixture['record']['path']) is preserve


def freeze_arrow26_fixture(profile, tmp_path, monkeypatch):
    from scripts import assemble_native_profile as assembler
    from scripts import freeze_native_profile as freezer
    fixture = arrow26_build_fixture(tmp_path / 'arrow26', monkeypatch)
    wrapper = assembler.normalize_arrow26_receipt(fixture['raw'], Path(fixture['record']['path']))
    preserved = assembler.preserve_inputs(wrapper, tmp_path / 'retained')
    wrapper_path = assembler.preserve_receipt(preserved, tmp_path / 'retained', 'pyarrow')
    profile[1]['artifacts'][1] = {'name': 'pyarrow', 'version': fixture['raw']['version'],
        **fixture['raw']['wheel'], 'source_receipt': assembler.file_record(wrapper_path)}
    launcher = tmp_path / 'scripts/native_runtime.py'
    launcher.parent.mkdir()
    # Execute the actual dual-policy implementation in the unchanged freezer.
    # Only visibly synthetic fixture pins differ from the production source.
    pin_names = [name for name in vars(runtime) if name.startswith(('ARROW26_', 'ARROW_NETWORK_', 'ARROW25_NETWORK_'))]
    pin_names.extend([*arrow_fixture_pins(), 'LXML_ZLIB_PATCH_SHA256'])
    launcher.write_text(Path(runtime.__file__).read_text() +
        '\n# Synthetic unit-test policy pins; never a production launcher.\n' +
        ''.join(name + ' = ' + repr(getattr(runtime, name)) + '\n' for name in pin_names))
    profile[1]['inputs'].append({**assembler.file_record(launcher), 'kind': 'recipe'})
    snapshot = tmp_path / 'snapshot'
    freezer.freeze(rewrite(profile), snapshot)
    return fixture, preserved, snapshot


def test_arrow26_candidate_survives_the_unchanged_freezer(profile, tmp_path, monkeypatch):
    from scripts import freeze_native_profile as freezer
    fixture, preserved, snapshot = freeze_arrow26_fixture(profile, tmp_path, monkeypatch)

    assert freezer.verify_snapshot(snapshot)['ready'] is False
    assert json.loads((snapshot / 'profile.json').read_bytes())['artifacts'][1]['version'] == fixture['raw']['version']


def test_frozen_arrow26_candidate_rejects_changed_retained_source(profile, tmp_path, monkeypatch):
    from scripts import freeze_native_profile as freezer
    fixture, preserved, snapshot = freeze_arrow26_fixture(profile, tmp_path, monkeypatch)
    source_hash = fixture['pins']['ARROW26_INTEGRATED_SOURCES'][
        'cpp/src/arrow/compute/kernels/copy_data_internal.h']
    selected = next(row for row in preserved['inputs'] if row['sha256'] == source_hash)
    Path(selected['path']).write_bytes(b'changed retained integrated header')

    with pytest.raises(freezer.SnapshotError, match='SHA256 mismatch'):
        freezer.verify_snapshot(snapshot)


def test_arrow26_candidate_does_not_register_a_legacy_rebuild(tmp_path, monkeypatch):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'arrow26', monkeypatch)
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    (scripts / 'build_native_arrow.py').write_text('raise RuntimeError("legacy25 fixture")\n')
    monkeypatch.setattr(assembler.shutil, 'which', lambda name: None)
    monkeypatch.setattr(assembler.subprocess, 'run', lambda *args, **kwargs:
                        runtime.subprocess.CompletedProcess(args, 0, stdout='synthetic host policy'))
    output = tmp_path / 'assembled.lock'

    assembler.assemble(tmp_path, output, tmp_path / 'absent-python',
                       arrow_receipt=Path(fixture['record']['path']))

    doc = json.loads(output.read_bytes())
    assert not any(Path(row['script']['path']).name == 'build_native_arrow.py' for row in doc['recipes'])
    assert doc['rebuild_supported'] is False


def test_arrow26_rebuild_rejects_before_environment_probes(profile, tmp_path, monkeypatch):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'arrow26', monkeypatch)
    wrapper = assembler.normalize_arrow26_receipt(fixture['raw'], Path(fixture['record']['path']))
    wrapper_path = assembler.preserve_receipt(wrapper, tmp_path / 'materials', 'pyarrow')
    profile[1]['artifacts'][1] = {'name': 'pyarrow', 'version': fixture['raw']['version'],
        **fixture['raw']['wheel'], 'source_receipt': assembler.file_record(wrapper_path)}
    profile[1]['recipes'] = [{'script': blob(tmp_path, 'legacy-build.py', 'legacy25 fixture'), 'args': []}]
    monkeypatch.setattr(runtime, '_python', lambda *args:
                        pytest.fail('an unsupported26 rebuild must reject before environment probes'))
    output = tmp_path / 'rebuild-output'

    with pytest.raises(runtime.ProfileError, match='Arrow 26.*rebuild.*unsupported'):
        runtime.build(rewrite(profile), output)

    assert not output.exists()


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


@pytest.fixture
def timeout_process(profile, tmp_path, monkeypatch):
    from types import SimpleNamespace

    path, doc = profile
    monkeypatch.setattr(runtime, '_python', lambda doc, base: Path(sys.executable))
    doc['recipes'] = [{'script': blob(tmp_path, 'worker.py', 'pass'), 'args': [], 'timeout_seconds': 1}]
    rewrite(profile)
    output = tmp_path / 'bounded-fake-children'
    observed = SimpleNamespace(states=['R', 'Z'], reads=[], kills=[], waits=[],
                               elapsed=0.0, leader_delay=0.0, leader_timeout=False,
                               original=runtime.subprocess.TimeoutExpired(['worker'], 1),
                               build=lambda: runtime.build(path, output), output=output)

    class Process:
        pid = 4242
        returncode = None

        def communicate(self, timeout=None):
            observed.waits.append(timeout)
            if len(observed.waits) == 1:
                raise observed.original
            observed.elapsed += observed.leader_delay
            if observed.leader_timeout:
                raise runtime.subprocess.TimeoutExpired(['worker'], timeout)
            self.returncode = -9
            return '', None

    def sleep(seconds):
        observed.elapsed += seconds

    original_iterdir, original_read = Path.iterdir, Path.read_text

    def iterdir(path):
        if path == Path('/proc'):
            return iter([Path('/proc/4243'), Path('/proc/9000')])
        return original_iterdir(path)

    def read_text(path, *args, **kwargs):
        if path == Path('/proc/9000/stat'):
            pytest.fail('unrelated process stat must not be inspected')
        if path == Path('/proc/4243/stat'):
            state = observed.states[0]
            if len(observed.states) > 1:
                observed.states.pop(0)
            observed.reads.append(state)
            if isinstance(state, Exception):
                raise state
            if state == 'malformed':
                return 'unparseable'
            group = 9999 if state == 'moved' else 4242
            return f'4243 (child ) name) {"R" if state == "moved" else state} 1 {group} {group} 0'
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(runtime.subprocess, 'Popen', lambda *args, **kwargs: Process())
    monkeypatch.setattr(runtime.os, 'killpg', lambda group, sig: observed.kills.append((group, sig)))
    monkeypatch.setattr(runtime.os, 'getpgid', lambda pid: 4242 if pid == 4243 else 9000)
    monkeypatch.setattr(runtime.os, 'getsid', lambda pid: 4242 if pid == 4243 else 9000)
    monkeypatch.setattr(runtime, 'time', SimpleNamespace(monotonic=lambda: observed.elapsed, sleep=sleep), raising=False)
    monkeypatch.setattr(Path, 'iterdir', iterdir)
    monkeypatch.setattr(Path, 'read_text', read_text)
    return observed


@pytest.mark.parametrize('terminal', ['Z', 'X'])
def test_timeout_waits_for_descendants_after_leader_reaped(timeout_process, tmp_path, terminal):
    timeout_process.states = ['R', terminal]
    with pytest.raises(runtime.ProfileError, match='command timed out'):
        timeout_process.build()
    assert timeout_process.reads == ['R', terminal]
    assert timeout_process.kills == [(4242, runtime.signal.SIGKILL)]
    assert timeout_process.waits == [1, 5.0]
    assert timeout_process.elapsed > 0
    assert not (timeout_process.output / 'build-receipt.json').exists()
    assert json.loads((timeout_process.output / 'FAILED.json').read_text())['ready'] is False


def test_timeout_refuses_live_descendant_at_shared_cleanup_deadline(timeout_process, tmp_path):
    timeout_process.states = ['R']
    timeout_process.leader_delay = 4.9
    with pytest.raises(runtime.ProfileError, match='cleanup.*terminal'):
        timeout_process.build()
    assert timeout_process.reads
    assert timeout_process.elapsed == pytest.approx(5.0)
    assert timeout_process.kills == [(4242, runtime.signal.SIGKILL)]


@pytest.mark.parametrize('state', [PermissionError('owned stat denied'), 'malformed'])
def test_timeout_refuses_unverifiable_owned_descendant(timeout_process, tmp_path, state):
    timeout_process.states = [state]
    with pytest.raises(runtime.ProfileError, match='cleanup.*inspect'):
        timeout_process.build()
    assert timeout_process.reads == [state]


@pytest.mark.parametrize('state', [FileNotFoundError(), 'moved'])
def test_timeout_allows_disappeared_or_reclassified_process(timeout_process, tmp_path, state):
    timeout_process.states = [state]
    with pytest.raises(runtime.ProfileError, match='command timed out'):
        timeout_process.build()
    assert timeout_process.reads == [state]


def test_timeout_bounds_leader_cleanup_communication(timeout_process, tmp_path):
    timeout_process.leader_timeout = True
    with pytest.raises(runtime.ProfileError, match='cleanup.*leader'):
        timeout_process.build()
    assert timeout_process.waits == [1, 5.0]


def test_interrupt_preserved_after_terminal_cleanup(timeout_process, tmp_path):
    timeout_process.original = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as caught:
        timeout_process.build()
    assert caught.value is timeout_process.original
    assert timeout_process.reads == ['R', 'Z']


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


@pytest.mark.parametrize('preserve', [False, True])
def test_corrective_arrow25_private_family_keeps_distinct_raw_evidence(profile, tmp_path, monkeypatch, preserve):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'private25', monkeypatch, family='25')
    original = Path(fixture['record']['path']).read_bytes()
    wrapper = assembler.normalize_arrow25_network_receipt(fixture['raw'], Path(fixture['record']['path']))
    assert wrapper['source_policy'] == runtime.ARROW25_NETWORK_FAMILY
    assert wrapper['arrow25_network_build_receipt'] == fixture['record']
    assert 'source_receipt' not in wrapper and 'arrow26_build_receipt' not in wrapper
    assert 'build' not in fixture['raw'] and 'effective_versions' not in fixture['raw']
    assert Path(fixture['record']['path']).read_bytes() == original
    if preserve:
        old_inputs = copy.deepcopy(wrapper['inputs'])
        wrapper = assembler.preserve_inputs(wrapper, tmp_path / 'retained')
        for row in old_inputs:
            Path(row['path']).unlink(missing_ok=True)
    wrapper_path = assembler.preserve_receipt(wrapper, tmp_path / 'wrappers', 'pyarrow')
    profile[1]['artifacts'][1] = {'name': 'pyarrow', 'version': fixture['raw']['version'],
        **fixture['raw']['wheel'], 'source_receipt': assembler.file_record(wrapper_path)}
    assert runtime.verify_artifacts(rewrite(profile))['ready'] is False


@pytest.mark.parametrize('family', ['25', '26'])
def test_private_prefix_retains_actual_json_array_build_evidence(tmp_path, monkeypatch, family):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'private', monkeypatch, family=family)
    network = fixture['raw']['network']
    commands = network['build']['nghttp2_build']['compile_commands']
    assert Path(commands['path']).name == 'compile_commands.json'
    assert isinstance(json.loads(Path(commands['path']).read_bytes()), list)
    normalize = assembler.normalize_arrow25_network_receipt if family == '25' else assembler.normalize_arrow26_receipt
    wrapper = normalize(fixture['raw'], Path(fixture['record']['path']))
    assert any(row['sha256'] == commands['sha256'] for row in wrapper['inputs'])


@pytest.mark.parametrize('family', ['25', '26'])
def test_private_arrow_family_cannot_be_renormalized_as_the_other(family, tmp_path, monkeypatch):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'private', monkeypatch, family=family)
    normalize = assembler.normalize_arrow26_receipt if family == '25' else assembler.normalize_arrow25_network_receipt
    with pytest.raises(runtime.ProfileError, match='Arrow (25|26)'):
        normalize(fixture['raw'], Path(fixture['record']['path']))


@pytest.mark.parametrize('damage', ['compute_chain', 'source_policy', 'recipe', 'private_source', 'prefix_receipt', 'static_bytes'])
def test_corrective_arrow25_private_evidence_cannot_be_relabelled(tmp_path, monkeypatch, damage):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'private25', monkeypatch, family='25')
    raw, material = fixture['raw'], fixture['material']
    if damage == 'compute_chain':
        raw['security_patches']['patches'][0]['files'][0]['patched_sha256'] = 'f' * 64
        refresh_arrow26_fixture(fixture, monkeypatch)
    elif damage == 'source_policy':
        raw['source_policy'] = runtime.ARROW26_PRIVATE_FAMILY
        fixture['record'] = material('build-receipt.json', raw)
        monkeypatch.setattr(runtime, 'ARROW25_NETWORK_BUILD_RECEIPT_SHA256', fixture['record']['sha256'])
    elif damage == 'recipe':
        raw['recipe'] = material('unreviewed25.py', 'unreviewed corrective recipe')
        refresh_arrow26_fixture(fixture, monkeypatch)
    elif damage == 'private_source':
        raw['network']['private_files'][str(Path(raw['network']['prefix']) / 'lib/libcurl.a')]['source_version'] = '8.18.0'
        refresh_arrow26_fixture(fixture, monkeypatch)
    elif damage == 'prefix_receipt':
        raw['network']['receipt']['sha256'] = 'f' * 64
        refresh_arrow26_fixture(fixture, monkeypatch)
    else:
        Path(next(row['path'] for row in raw['linked_static_archives'] if row['path'].endswith('libz.a'))).write_bytes(b'tampered private bytes')
    with pytest.raises(runtime.ProfileError, match='Arrow 25'):
        assembler.normalize_arrow25_network_receipt(raw, Path(fixture['record']['path']))


@pytest.mark.parametrize('library', ['libcurl.so.4', 'libcurl-gnutls.so.4', 'libxml2.so.16', 'libz.so.1',
                                   'libiconv.a', 'libcharset.a', 'libnghttp2.a', 'libssl.so.3', 'libcrypto.a'])
@pytest.mark.parametrize('location', ['selected', 'final_map'])
def test_private_arrow_rejects_all_host_provider_variants(tmp_path, monkeypatch, library, location):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'private26', monkeypatch)
    raw, material = fixture['raw'], fixture['material']
    foreign = material('host/' + library, 'affected or unapproved host provider')
    if location == 'selected':
        selected = json.loads(Path(raw['compiled_dependencies']['path']).read_bytes())
        selected['native_inputs'].append({**foreign, 'provider': {'package': 'host', 'version': '1',
            'source_package': 'host', 'source_version': '1'}})
        raw['compiled_dependencies'] = material('compiled-dependencies.json', selected)
    else:
        mapping = raw['native_link_artifacts'][0]
        text = Path(mapping['path']).read_text() + 'LOAD ' + foreign['path'] + '\n'
        changed = material('cpp-build/release/libarrow.so.2600.map', text)
        raw['native_link_artifacts'] = raw['final_native_link_artifacts'] = [changed]
    refresh_arrow26_fixture(fixture, monkeypatch)
    with pytest.raises(runtime.ProfileError, match='unapproved host network/crypto'):
        assembler.normalize_arrow26_receipt(raw, Path(fixture['record']['path']))


@pytest.mark.parametrize('unused', [None, 'libcurl.a', 'libxml2.a', 'libiconv.a', 'libz.a', 'libnghttp2.a', 'libssl.a', 'libcrypto.a'])
def test_private_arrow_admission_requires_every_core_archive_member(tmp_path, monkeypatch, unused):
    from scripts import assemble_native_profile as assembler
    fixture = arrow26_build_fixture(tmp_path / 'private26', monkeypatch)
    raw, material = fixture['raw'], fixture['material']
    lines = ['Linker script and memory map']
    for row in raw['linked_static_archives']:
        lines.append('LOAD ' + row['path'])
        if unused is not None and Path(row['path']).name != unused:
            lines.append(row['path'] + '(member.o)')
    mapping = material('cpp-build/release/libarrow.so.2600.map', '\n'.join(lines) + '\n')
    raw['native_link_artifacts'] = raw['final_native_link_artifacts'] = [mapping]
    refresh_arrow26_fixture(fixture, monkeypatch)
    with pytest.raises(runtime.ProfileError, match='consumed private archive member'):
        assembler.normalize_arrow26_receipt(raw, Path(fixture['record']['path']))
