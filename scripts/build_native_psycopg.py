#!/usr/bin/env python3
"""Build a hashed Psycopg C wheel; host security admission happens at launch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
import subprocess
import tomllib
import urllib.request
import zipfile


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_source(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError('Psycopg source checksum mismatch')


def interpreter_path(path: Path) -> str:
    # Resolving a venv executable symlink selects its base environment instead.
    return str(path.absolute())


def linkage_paths(text: str) -> list[str]:
    if 'not found' in text:
        raise ValueError('unresolved native library in Psycopg wheel')
    return re.findall(r'(?:=>\s+)?(/\S+)\s+\(0x[0-9a-fA-F]+\)', text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('.native-runtime/psycopg'))
    parser.add_argument('--lock', type=Path, default=Path('uv.lock'))
    args = parser.parse_args()
    package = next(p for p in tomllib.loads(args.lock.read_text())['package']
                   if p['name'] == 'psycopg-c')
    if package['version'] != '3.3.6':
        parser.error('this reviewed recipe requires psycopg-c 3.3.6')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / 'psycopg_c-3.3.6.tar.gz'
    expected = package['sdist']['hash'].removeprefix('sha256:')
    if not source.exists():
        urllib.request.urlretrieve(package['sdist']['url'], source)
    verify_source(source, expected)
    wheelhouse = output / 'wheelhouse'
    wheelhouse.mkdir(exist_ok=True)
    log = output / 'build.log'
    paths = json.loads(subprocess.check_output([interpreter_path(args.python), '-I', '-c',
        'import importlib.metadata,json,pathlib; names=("setuptools","wheel","pip","packaging");print(json.dumps([str(importlib.metadata.distribution(name).locate_file(p)) for name in names for p in importlib.metadata.files(name) if pathlib.Path(importlib.metadata.distribution(name).locate_file(p)).is_file() and not str(p).endswith(".pyc")]))'], text=True))
    include = Path(subprocess.check_output([interpreter_path(args.python), '-I', '-c',
        'import sysconfig;print(sysconfig.get_path("include"))'], text=True).strip())
    paths.extend(str(path) for directory in (include, Path('/usr/include/postgresql'))
                 for path in directory.rglob('*.h'))
    manifest = output / 'build-backend-inputs.json'
    manifest.write_text(json.dumps([{'path': path, 'sha256': sha256(Path(path))}
                                   for path in sorted(set(paths))], indent=2) + '\n')
    env = dict(os.environ, MAX_JOBS='1', PIP_CACHE_DIR=str(output / 'pip-cache'))
    with log.open('w') as stream:
        subprocess.run([interpreter_path(args.python), '-m', 'pip', 'wheel',
                        '--no-deps', '--no-build-isolation', '--wheel-dir', str(wheelhouse),
                        str(source)], check=True, env=env, stdout=stream, stderr=subprocess.STDOUT)
    wheels = list(wheelhouse.glob('psycopg_c-3.3.6-*.whl'))
    if len(wheels) != 1:
        raise ValueError('expected exactly one Psycopg C wheel')
    libraries = {}
    native_link_artifacts = []
    resolved_libraries = {}
    with zipfile.ZipFile(wheels[0]) as archive:
        for name in archive.namelist():
            if name.endswith('.so'):
                target = output / Path(name).name
                target.write_bytes(archive.read(name))
                libraries[name] = subprocess.check_output(['ldd', str(target)], text=True)
                native_link_artifacts.append({'path': str(target), 'sha256': sha256(target)})
                for dependency in linkage_paths(libraries[name]):
                    library = Path(dependency).resolve()
                    if library not in resolved_libraries:
                        resolved_libraries[library] = {'path': str(library), 'sha256': sha256(library),
                            'owner': 'dpkg:' + subprocess.check_output(['dpkg-query', '-S', str(library)], text=True).split(':', 1)[0]}
    if not libraries or not all('libpq.so' in linkage for linkage in libraries.values()):
        raise ValueError('Psycopg wheel does not link to system libpq')
    inputs = [{'path': str(source), 'sha256': expected},
              {'path': str(args.lock.resolve()), 'sha256': sha256(args.lock)},
              {'path': str(manifest), 'sha256': sha256(manifest)},
              {'path': str(log), 'sha256': sha256(log)}]
    compiler = Path(shutil.which('cc')).resolve()
    pg_config = Path(shutil.which('pg_config')).resolve()
    receipt = {'schema_version': 1, 'status': 'candidate_host_update_required',
               'wheel_sha256': sha256(wheels[0]),
               'inputs': inputs,
               'source_revision': expected,
               'toolchain': {'compiler': {'path': str(compiler), 'sha256': sha256(compiler)},
                             'pg_config': {'path': str(pg_config), 'sha256': sha256(pg_config)},
                             'python': {'path': interpreter_path(args.python), 'sha256': sha256(args.python)}},
               'toolchain_versions': {'compiler': subprocess.check_output(['cc', '--version'], text=True).splitlines()[0],
                             'python': subprocess.check_output([interpreter_path(args.python), '--version'], text=True).strip(),
                             'setuptools': subprocess.check_output([interpreter_path(args.python), '-c',
                                'import importlib.metadata; print(importlib.metadata.version("setuptools"))'], text=True).strip()},
               'inputs_sha256': hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
               'native_link_artifacts': native_link_artifacts,
               'effective_versions': {'psycopg-c': package['version'], 'libpq': subprocess.check_output(['pg_config', '--version'], text=True).strip()},
               'native_files': list(resolved_libraries.values()),
               'name': 'psycopg-c', 'version': package['version'],
               'source': {'url': package['sdist']['url'], 'sha256': expected},
               'wheel': {'path': str(wheels[0]), 'sha256': sha256(wheels[0])},
               'compiler': subprocess.check_output(['cc', '--version'], text=True).splitlines()[0],
               'pg_config': subprocess.check_output(['pg_config', '--version'], text=True).strip(),
               'system_packages': subprocess.check_output(['dpkg-query', '-W', 'libpq5',
                                                          'libpq-dev', 'libssl3t64', 'libssl-dev'], text=True),
               'linkage': libraries, 'build_log_sha256': sha256(log)}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt['wheel']))


if __name__ == '__main__':
    main()
