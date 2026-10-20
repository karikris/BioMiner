#!/usr/bin/env python3
"""Assemble observed build receipts; absent artifacts remain explicitly pending."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def file_record(path):
    path = Path(path).absolute()
    return {'path': str(path), 'sha256': sha256(path)}


def preserve_receipt(receipt, directory, name):
    payload = (json.dumps(receipt, indent=2) + '\n').encode()
    digest = hashlib.sha256(payload).hexdigest()
    path = directory / digest / f'{name}-receipt.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError('preserved receipt bytes differ')
    else:
        with path.open('xb') as stream:
            stream.write(payload)
    return path


def preserve_inputs(receipt, directory):
    """Retain temporary build inputs without changing their claimed bytes."""
    receipt = json.loads(json.dumps(receipt))
    directory.mkdir(parents=True, exist_ok=True)
    # Backend manifests bind actual imported files and headers. Preserve their
    # dependencies too: retaining the JSON alone would lose those bytes after
    # a temporary build environment disappears.
    for manifest in list(receipt['inputs']):
        source = Path(manifest['path'])
        if source.name == 'build-backend-inputs.json':
            if sha256(source) != manifest['sha256']:
                raise ValueError(f'build input changed before preservation: {source}')
            dependencies = json.loads(source.read_text())
            if not isinstance(dependencies, list):
                raise ValueError('backend input manifest must contain file records')
            for item in dependencies:
                # A previously preserved receipt already binds these bytes at
                # durable paths. Its original manifest intentionally retains
                # historical paths, which may no longer exist.
                retained = [entry for entry in receipt['inputs']
                            if entry.get('sha256') == item.get('sha256')]
                if not retained:
                    receipt['inputs'].append(dict(item, kind='backend-input'))
    records = [*receipt['inputs'], *receipt['native_link_artifacts'], *receipt['toolchain'].values()]
    for item in records:
        source = Path(item['path'])
        if sha256(source) != item['sha256']:
            raise ValueError(f'build input changed before preservation: {source}')
        target = directory / item['sha256'] / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        if sha256(target) != item['sha256']:
            raise ValueError(f'preserved build input checksum mismatch: {target}')
        item['original_path'] = item['path']
        item['path'] = str(target.absolute())
    receipt['inputs_sha256'] = hashlib.sha256(json.dumps(receipt['inputs'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return receipt


def assemble(root, output, python, preserve=False, duckdb_receipt=None, biominer_receipt=None,
             lxml_receipt=None, psycopg_receipt=None, arrow_receipt=None):
    root, output, python = root.absolute(), output.absolute(), python.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'profile already exists; choose a new output path: {output}')
    sources = {
        'biominer': root / '.native-runtime/biominer/receipt.json',
        'psycopg-c': root / '.native-runtime/psycopg/receipt.json',
        'pyarrow': root / '.native-runtime/arrow/generation-1/build-receipt.json',
        'duckdb': root / '.native-runtime/duckdb/build-receipt.json',
        'lxml': root / '.native-runtime/lxml/receipt.json',
    }
    if duckdb_receipt is not None:
        sources['duckdb'] = duckdb_receipt.absolute()
    if biominer_receipt is not None:
        sources['biominer'] = biominer_receipt.absolute()
    if lxml_receipt is not None:
        sources['lxml'] = lxml_receipt.absolute()
    if psycopg_receipt is not None:
        sources['psycopg-c'] = psycopg_receipt.absolute()
    if arrow_receipt is not None:
        sources['pyarrow'] = arrow_receipt.absolute()
    doc = {'schema_version': 1, 'status': 'candidate', 'ready': False,
           'platform': {'system': 'Linux', 'machine': 'x86_64'},
           'generation_root': str(root / '.native-runtime/generations'),
           'base_lock': {}, 'project': {}, 'python': {}, 'uv': {},
           'inputs': [], 'artifacts': [], 'native_files': [],
           'host_packages': [], 'checks': [], 'recipes': [],
           'extension_policy': {}, 'pending_artifacts': [],
           'limitations': ['Host patching, complete acceptance, entrypoint migration and promotion required.',
                           'Live PostgreSQL and S3 checks unrun.']}
    for key, name in (('base_lock', 'uv.lock'), ('project', 'pyproject.toml')):
        if (root / name).is_file():
            doc[key] = file_record(root / name)
    if python.is_file():
        identity = json.loads(subprocess.check_output([str(python), '-I', '-c',
            'import json,platform,ssl;print(json.dumps({"version":platform.python_version(),"openssl_version":ssl.OPENSSL_VERSION.split()[1]}))'], text=True))
        source_receipt = root / '.native-runtime/python-runtime/receipt.json'
        source_build = 'unattested'
        if source_receipt.is_file():
            provenance = json.loads(source_receipt.read_text())
            if provenance['python']['sha256'] != sha256(python) or Path(provenance['python']['path']).resolve() != python.resolve():
                raise ValueError('approved Python does not match its archive attestation')
            source_build = provenance['source_build']
            doc['inputs'].append(file_record(source_receipt) | {'kind': 'python-source-receipt'})
            doc['inputs'].append(provenance['archive'] | {'kind': 'source', 'source_revision': source_build})
            # Imports used by later checks load additional standard-library
            # extensions, including ctypes. Bind all archive-attested ELF files,
            # rather than only those mapped by the initial ssl probe.
            for item in provenance['verified_archive_files']:
                native_path = Path(item['path'])
                with native_path.open('rb') as stream:
                    is_elf = stream.read(4) == b'\x7fELF'
                if is_elf:
                    if sha256(native_path) != item['sha256']:
                        raise ValueError(f'archive-attested Python ELF changed: {native_path}')
                    doc['native_files'].append(dict(item, owner='python-runtime:' + source_build))
        doc['python'] = file_record(python) | identity | {'source_build': source_build}
        mapped = json.loads(subprocess.check_output([str(python), '-I', '-c',
            'import json,ssl,hashlib,pathlib; print(json.dumps(sorted({line.split()[-1] for line in pathlib.Path("/proc/self/maps").read_text().splitlines() if len(line.split())>=6 and line.split()[-1].startswith("/")})))'], text=True))
        runtime_root = python.resolve().parents[1]
        for filename in mapped:
            path = Path(filename).resolve()
            if not path.is_file():
                raise ValueError(f'Python loaded file disappeared: {path}')
            if path.is_relative_to(runtime_root):
                owner = 'python-runtime:' + source_build
            else:
                ownership = subprocess.run(['dpkg-query', '-S', str(path)], capture_output=True, text=True)
                if ownership.returncode:
                    raise ValueError(f'Python native library lacks package ownership: {path}')
                owner = 'dpkg:' + ownership.stdout.split(':', 1)[0]
            doc['native_files'].append(file_record(path) | {'owner': owner})
    uv = shutil.which('uv')
    if uv:
        doc['uv'] = file_record(uv)
    for name, path in sources.items():
        if not path.is_file():
            doc['pending_artifacts'].append(name)
            continue
        receipt = json.loads(path.read_text())
        if name == 'duckdb' and receipt.get('policy_id'):
            doc['extension_policy'] = {'policy_id': receipt['policy_id'],
                                       'allowlist': receipt.get('extension_policy', {}).get('extensions', [])}
        wheel = receipt.get('wheel')
        # Never infer a successful compile from logs or pending recipe metadata.
        if not isinstance(wheel, dict) or not Path(wheel.get('path', '')).is_file():
            doc['pending_artifacts'].append(name)
            continue
        if sha256(wheel['path']) != wheel.get('sha256'):
            raise ValueError(f'{name} wheel differs from build receipt')
        version = receipt.get('version')
        if not isinstance(version, str):
            raise ValueError(f'{name} receipt needs effective wheel version')
        source_descriptor = receipt.get('source_receipt')
        if isinstance(source_descriptor, dict):
            source_path = Path(source_descriptor['path'])
            if sha256(source_path) != source_descriptor['sha256']:
                raise ValueError(f'{name} source receipt changed')
            receipt = json.loads(source_path.read_text())
            # Build history and compiled policy are separate from source inputs.
            receipt['policy_id'] = doc['extension_policy'].get('policy_id')
            path = source_path
        if preserve:
            receipt = preserve_inputs(receipt, root / '.native-runtime/build-materials')
            original = file_record(path)
            receipt['preserved_from'] = original
            archived = root / '.native-runtime/build-materials' / original['sha256'] / path.name
            archived.parent.mkdir(parents=True, exist_ok=True)
            if not archived.exists():
                shutil.copyfile(path, archived)
            if sha256(archived) != original['sha256']:
                raise ValueError('archived source receipt checksum mismatch')
            receipt['inputs'].append(file_record(archived))
            receipt['inputs_sha256'] = hashlib.sha256(json.dumps(receipt['inputs'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            path = preserve_receipt(receipt, root / '.native-runtime/build-materials', name)
        doc['artifacts'].append({'name': name, 'version': version,
                                 **wheel, 'source_receipt': file_record(path)})
        doc['inputs'].append(file_record(path) | {'kind': 'build-receipt'})
        for item in receipt.get('native_files', []):
            item = dict(item)
            if not item['owner'].startswith(('dpkg:', 'wheel:', 'python-runtime')):
                item['owner'] = 'dpkg:' + item['owner'].split(':', 1)[0]
            doc['native_files'].append(item)
        with zipfile.ZipFile(wheel['path']) as archive:
            for member in archive.namelist():
                if '.so' in Path(member).name and not member.endswith('/'):
                    doc['native_files'].append({'path': '{generation}/env/lib/python3.14/site-packages/' + member,
                        'sha256': hashlib.sha256(archive.read(member)).hexdigest(), 'owner': 'wheel:' + name})
        if name == 'duckdb':
            doc['extension_policy'] = {'policy_id': receipt['policy_id'],
                                       'allowlist': receipt.get('extension_policy', {}).get('extensions', [])}
    for name in ('openssl', 'libssl3t64', 'libssl-dev', 'libpq5'):
        policy = subprocess.run(['apt-cache', 'policy', name], capture_output=True, text=True, check=True).stdout
        doc['host_packages'].append({'name': name, 'architecture': 'amd64', 'origin': 'Ubuntu',
            'minimum_version': '18.6-0ubuntu0.26.04.1' if name == 'libpq5' else '3.5.5-1ubuntu3.7',
            'advisory': 'Ubuntu vendor package provenance' if name == 'libpq5' else 'USN-8861-1',
            'apt_policy_sha256': hashlib.sha256(policy.encode()).hexdigest()})
    checks = root / 'scripts/check_native_runtime.py'
    if checks.is_file():
        for name in ('effective_versions', 'native_inventory', 'extension_policy', 'tls', 'geometry'):
            doc['checks'].append({'name': name, 'script': file_record(checks),
                'args': ['--check', name, '--profile', '{profile}', '--generation', '{generation}']})
    for script in ('native_runtime.py', 'check_native_runtime.py', 'check_native_tls.py',
                   'build_native_psycopg.py', 'build_native_arrow.py', 'build_native_duckdb.py',
                   'build_native_lxml.py', 'build_native_project.py', 'attest_native_python.py',
                   'refresh_native_host.py', 'assemble_native_profile.py'):
        path = root / 'scripts' / script
        if path.is_file():
            doc['inputs'].append(file_record(path) | {'kind': 'check' if script.startswith('check_') else 'recipe'})
    build_root = root / '.native-runtime'
    definitions = [
        ('build_native_project.py', ['--python', str(build_root / 'biominer/build-env/bin/python'),
                                    '--root', str(root), '--output', '{output}/biominer']),
        ('build_native_psycopg.py', ['--python', str(build_root / 'psycopg/build-env/bin/python'),
                                   '--output', '{output}/psycopg', '--lock', str(root / 'uv.lock')]),
        ('build_native_arrow.py', ['--python', str(build_root / 'arrow/build-env/bin/python'),
                                  '--openssl-prefix', str(build_root / 'arrow/openssl-prefix'),
                                  '--archive', str(build_root / 'arrow/downloads/arrow.tar.gz'),
                                  '--work', '{output}/arrow']),
        ('build_native_duckdb.py', ['--python', str(python), '--work-dir', '{output}/duckdb']),
        ('build_native_lxml.py', ['--python', str(build_root / 'lxml/build-env/bin/python'),
                                 '--sources', str(build_root / 'lxml/downloads'),
                                 '--output', '{output}/lxml', '--jobs', '4']),
    ]
    for script, args in definitions:
        if (root / 'scripts' / script).is_file():
            doc['recipes'].append({'script': file_record(root / 'scripts' / script), 'args': args})
    if not doc['extension_policy']:
        doc['extension_policy'] = {'policy_id': None, 'allowlist': [], 'status': 'pending_duckdb_wheel'}
    native = {}
    for item in doc['native_files']:
        previous = native.get(item['path'])
        if previous is not None and previous != item:
            raise ValueError(f'conflicting native ownership/hash records: {item["path"]}')
        native[item['path']] = item
    doc['native_files'] = list(native.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    return {'status': 'candidate', 'ready': False, 'pending_artifacts': doc['pending_artifacts'], 'profile': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('native-runtime.lock'))
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--preserve-inputs', action='store_true')
    parser.add_argument('--duckdb-receipt', type=Path,
                        help='Explicit preserved production receipt after separate test-policy builds')
    parser.add_argument('--biominer-receipt', type=Path,
                        help='Explicit BioMiner build receipt for a new configuration generation')
    parser.add_argument('--lxml-receipt', type=Path,
                        help='Explicit lxml build receipt with its patched native closure')
    parser.add_argument('--psycopg-receipt', type=Path,
                        help='Explicit preserved Psycopg build receipt')
    parser.add_argument('--arrow-receipt', type=Path,
                        help='Explicit preserved or newly qualified Arrow build receipt')
    args = parser.parse_args()
    print(json.dumps(assemble(args.root, args.output, args.python, args.preserve_inputs,
                              args.duckdb_receipt, args.biominer_receipt, args.lxml_receipt,
                              args.psycopg_receipt, args.arrow_receipt), sort_keys=True))


if __name__ == '__main__':
    main()
