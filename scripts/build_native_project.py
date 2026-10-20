#!/usr/bin/env python3
"""Build frozen committed application code with an explicit configuration overlay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile


def record(path):
    path = Path(path).absolute()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest}


def source_revision(path):
    with tarfile.open(path) as archive:
        revision = archive.pax_headers.get('comment', '')
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('frozen source archive lacks exact Git revision')
    return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('.native-runtime/biominer'))
    args = parser.parse_args()
    work, root, python = args.output.absolute(), args.root.absolute(), args.python.absolute()
    work.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    archive = work / f'source-{revision}.tar'
    subprocess.run(['git', 'archive', revision, '--output', str(archive)], cwd=root, check=True)
    if source_revision(archive) != revision:
        raise ValueError('Git archive revision mismatch')
    config = record(root / 'pyproject.toml')
    source = work / f'source-{revision}-{config["sha256"][:12]}'
    if source.exists():
        raise ValueError('source generation already exists; choose a new output directory')
    source.mkdir()
    with tarfile.open(archive) as bundle:
        bundle.extractall(source, filter='data')
    shutil.copyfile(root / 'pyproject.toml', source / 'pyproject.toml')
    if record(source / 'pyproject.toml')['sha256'] != config['sha256']:
        raise ValueError('configuration overlay changed during snapshot')
    # Snapshot actual build backend inputs before invoking them; no inferred
    # wheel/source association from unrelated files left in a directory.
    paths = json.loads(subprocess.check_output([str(python), '-I', '-c',
        'import importlib.metadata,json,pathlib; names=("setuptools","wheel","build","packaging","pyproject-hooks");print(json.dumps([str(importlib.metadata.distribution(name).locate_file(p)) for name in names for p in importlib.metadata.files(name) if pathlib.Path(importlib.metadata.distribution(name).locate_file(p)).is_file() and not str(p).endswith(".pyc")]))'], text=True))
    backend = work / 'build-backend-inputs.json'
    backend.write_text(json.dumps([record(path) for path in sorted(set(paths))], indent=2) + '\n')
    wheelhouse = work / 'wheelhouse'
    wheelhouse.mkdir(exist_ok=True)
    command = [str(python), '-I', '-m', 'build', '--wheel', '--no-isolation',
               '--outdir', str(wheelhouse), str(source)]
    log = work / 'build.log'
    with log.open('w') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    wheels = list(wheelhouse.glob('biominer-0.1.0-*.whl'))
    if len(wheels) != 1:
        raise ValueError('expected exactly one frozen BioMiner wheel')
    inputs = [record(archive), record(source / 'pyproject.toml'), record(backend), record(log)]
    receipt = {'schema_version': 1, 'status': 'built_unverified', 'name': 'biominer', 'version': '0.1.0',
               'wheel': record(wheels[0]), 'wheel_sha256': record(wheels[0])['sha256'],
               'source_revision': revision, 'command': command,
               'source_overlay': 'Task-owned pyproject.toml; application source from git archive',
               'toolchain': {'python': record(python)}, 'inputs': inputs,
               'inputs_sha256': hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
               'native_link_artifacts': [], 'effective_versions': {}}
    (work / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt['wheel']))


if __name__ == '__main__':
    main()
