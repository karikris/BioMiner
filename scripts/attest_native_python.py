#!/usr/bin/env python3
"""Extract and attest the approved standalone interpreter into persistent storage."""
from __future__ import annotations

import argparse
import hashlib
import json
import marshal
import os
import re
from pathlib import Path
import subprocess
import tarfile
import urllib.request

URL = 'https://github.com/astral-sh/python-build-standalone/releases/download/20260929/cpython-3.14.7%2B20260929-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz'
ARCHIVE_SHA256 = 'da5357f47b1d9c8d4c439004d463d893ef688a44c27438a62b47073d26d1377e'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_archive(path):
    if sha256(path) != ARCHIVE_SHA256:
        raise ValueError('official Python archive checksum mismatch')


def verify_prefix(archive, prefix):
    expected_paths = set()
    expected_symlinks = set()
    matched = []
    with tarfile.open(archive) as source:
        for member in source:
            relative = Path(member.name).relative_to('python')
            installed = prefix / relative
            expected_paths.add(relative)
            if member.issym():
                expected_symlinks.add(relative)
                if not installed.is_symlink() or os.readlink(installed) != member.linkname:
                    raise ValueError(f'Python runtime symlink differs: {relative}')
            elif member.isdir():
                if installed.is_symlink() or not installed.is_dir():
                    raise ValueError(f'Python runtime directory differs: {relative}')
            elif member.isfile() or member.islnk():
                if installed.is_symlink():
                    raise ValueError(f'Python runtime file replaced by symlink: {relative}')
                if relative.suffix == '.pyc':
                    continue  # Validate executable bytecode against source below.
                expected = hashlib.file_digest(source.extractfile(member), 'sha256').hexdigest()
                if sha256(installed) != expected:
                    raise ValueError(f'Python runtime differs from official archive: {relative}')
                matched.append({'path': str(installed), 'sha256': expected})
    for path in prefix.rglob('*'):
        relative = path.relative_to(prefix)
        if path.is_symlink() and relative not in expected_symlinks:
            raise ValueError(f'unexpected runtime file: {relative}')
        if path.is_dir() or (relative in expected_paths and path.suffix != '.pyc'):
            continue
        cache = re.fullmatch(r'(.+)\.cpython-314(?:\.opt-[12])?\.pyc', path.name)
        source = relative.parent.parent / (cache[1] + '.py') if cache else None
        if path.parent.name == '__pycache__' and source in expected_paths and not path.is_symlink():
            optimization = 2 if '.opt-2.' in path.name else 1 if '.opt-1.' in path.name else 0
            cached = path.read_bytes()
            if len(cached) < 16:
                raise ValueError(f'invalid runtime bytecode: {relative}')
            expected = compile((prefix / source).read_bytes(), str(prefix / source), 'exec',
                               dont_inherit=True, optimize=optimization)
            if marshal.loads(cached[16:]) != expected:
                raise ValueError(f'runtime bytecode differs from approved source: {relative}')
            continue
        raise ValueError(f'unexpected runtime file: {relative}')
    return matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.native-runtime/python-runtime'))
    args = parser.parse_args()
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'cpython-3.14.7+20260929.tar.gz'
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    verify_archive(archive)
    prefix = output / 'python'
    if not prefix.exists():
        with tarfile.open(archive) as source:
            source.extractall(output, filter='data')
    matched = verify_prefix(archive, prefix)
    python = prefix / 'bin/python3.14'
    identity = json.loads(subprocess.check_output([str(python), '-I', '-S', '-B', '-c',
        'import ssl,platform,json;print(json.dumps({"version":platform.python_version(),"openssl_version":ssl.OPENSSL_VERSION.split()[1]}))'], text=True))
    if identity != {'version': '3.14.7', 'openssl_version': '3.5.9'}:
        raise ValueError(f'actual interpreter identity changed: {identity}')
    receipt = {'schema_version': 1, 'source_build': 'Astral20260929', 'url': URL,
               'archive': {'path': str(archive), 'sha256': ARCHIVE_SHA256},
               'python': {'path': str(python), 'sha256': sha256(python)},
               'identity': identity, 'verified_archive_files': matched}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'source_build': receipt['source_build'], 'python': str(python),
                      'matched_files': len(matched), 'archive_sha256': ARCHIVE_SHA256}))


if __name__ == '__main__':
    main()
