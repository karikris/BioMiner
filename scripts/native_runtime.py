#!/usr/bin/env python3
"""Fail-closed, profile-bound native runtime generations.

Profiles are operator-approved input, not self-authenticating certificates. Build
receipts are candidates; only an installed, immutable generation passing host and
runtime checks is admitted. No command updates an existing environment.
"""
from __future__ import annotations

import argparse
import email
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


class ProfileError(ValueError):
    """Missing, incompatible, or unverifiable native runtime evidence."""


VERSIONS = {'psycopg-c': '3.3.6', 'pyarrow': '25.0.1+biominer.1',
            'duckdb': '1.5.6+biominer.1', 'lxml': '6.1.3'}
LXML_NATIVE_VERSIONS = {'lxml': '6.1.3', 'libxml2': '2.15.4', 'libxslt': '1.1.45',
                        'libiconv': '1.19', 'zlib': '1.3.2'}
LXML_FEATURES = {'catalog', 'html', 'iconv', 'regexp', 'schematron',
                 'xmlschema', 'xpath', 'zlib'}
LXML_ZLIB_PATCH_COMMIT = 'df84af25dc1942490e1d1c899a07619152a46148'
LXML_ZLIB_PATCH_SHA256 = '110ff14375733173d8aa54574473424fbd7dfe4b81f1ca34a759c6fe14b15b14'
RUNTIME_CHECKS = {'effective_versions', 'native_inventory', 'extension_policy'}
INSTALL_CHECKS = RUNTIME_CHECKS | {'tls', 'geometry'}
BLOCKED_ENV = {'LD_PRELOAD', 'LD_LIBRARY_PATH', 'LD_AUDIT', 'PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'GDAL_DRIVER_PATH', 'OPENSSL_MODULES', 'OPENSSL_CONF'}
SHA = re.compile(r'[0-9a-f]{64}')


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ProfileError(f'cannot read profile/receipt {path}: {exc}') from exc
    if not isinstance(result, dict):
        raise ProfileError(f'profile/receipt must be an object: {path}')
    return result


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ProfileError(f'{label}: expected lowercase SHA256')
    return value


def _file(item: Any, base: Path, label: str) -> Path:
    if not isinstance(item, dict) or not isinstance(item.get('path'), str):
        raise ProfileError(f'{label}: expected path and sha256')
    path = Path(item['path'])
    path = (base / path).resolve() if not path.is_absolute() else path.resolve()
    expected = _hash(item.get('sha256'), label)
    try:
        if not path.is_file() or sha256(path) != expected:
            raise ProfileError(f'{label}: missing file or SHA256 mismatch: {path}')
    except OSError as exc:
        raise ProfileError(f'{label}: {exc}') from exc
    return path


def load_profile(path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path).resolve()
    doc = _read_json(path)
    if doc.get('schema_version') != 1:
        raise ProfileError('profile schema_version must be 1')
    expected = {'system': 'Linux', 'machine': 'x86_64'}
    if doc.get('platform') != expected or platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ProfileError('profile platform must match Linux x86_64 host')
    for name in ('inputs', 'artifacts', 'native_files', 'host_packages', 'checks', 'recipes'):
        if not isinstance(doc.get(name), list):
            raise ProfileError(f'profile {name} must be a list')
        if any(not isinstance(item, dict) for item in doc[name]):
            raise ProfileError(f'profile {name} entries must be objects')
    for name in ('base_lock', 'project', 'python', 'uv', 'extension_policy'):
        if not isinstance(doc.get(name), dict):
            raise ProfileError(f'profile {name} must be an object')
    policy = doc['extension_policy']
    _hash(policy.get('policy_id'), 'extension_policy.policy_id')
    if not isinstance(policy.get('allowlist'), list):
        raise ProfileError('extension_policy.allowlist must be a list')
    for extension in policy['allowlist']:
        if not isinstance(extension, dict) or not isinstance(extension.get('name'), str):
            raise ProfileError('extension allowlist requires name and sha256')
        _hash(extension.get('sha256'), 'extension allowlist')
    return doc, path


def _inputs(doc: dict[str, Any], base: Path) -> None:
    _file(doc['base_lock'], base, 'base_lock')
    _file(doc['project'], base, 'project')
    if not doc['inputs']:
        raise ProfileError('profile requires checksummed source/build inputs')
    for item in doc['inputs']:
        _file(item, base, 'input')
        if not isinstance(item.get('kind'), str):
            raise ProfileError('input requires kind')
        if item['kind'] == 'source' and not item.get('source_revision'):
            raise ProfileError('source input requires exact source_revision')


def _lxml_materials(receipt: dict[str, Any], base: Path, wheel: Path) -> None:
    """Require retained, qualified relink material for the static native closure."""
    def retained(record, label):
        if not isinstance(record, dict):
            raise ProfileError(f'lxml receipt requires {label}')
        digest = _hash(record.get('sha256'), f'lxml {label} sha256')
        matches = [item for item in receipt['inputs'] if item.get('sha256') == digest]
        if not matches:
            raise ProfileError(f'lxml {label} must be a checksummed retained input')
        return _file(matches[0], base, f'lxml {label}')

    licenses = _read_json_list(retained(receipt.get('license_manifest'), 'license_manifest'))
    relink = receipt.get('relink')
    if not isinstance(relink, dict):
        raise ProfileError('lxml receipt requires relink materials')
    for name in ('bundle', 'manifest'):
        retained(relink.get(name), f'relink {name}')
    result = _read_json(retained(relink.get('qualification'), 'relink qualification'))
    if result.get('status') != 'passed' or result.get('relocated_bundle') is not True:
        raise ProfileError('lxml relocated relink qualification did not pass')
    _hash(result.get('replacement_library_sha256'), 'lxml replacement library sha256')
    outputs = result.get('outputs')
    if (not isinstance(outputs, list) or not outputs
            or not all(isinstance(item, dict) and isinstance(item.get('name'), str)
                       for item in outputs)):
        raise ProfileError('lxml relink qualification requires extension outputs')
    with zipfile.ZipFile(wheel) as archive:
        members = [name for name in archive.namelist() if name.endswith('.so')]
        native = {Path(name).name: hashlib.sha256(archive.read(name)).hexdigest() for name in members}
        if (len(native) != len(members) or len(outputs) != len(native)
                or {item['name'] for item in outputs} != set(native)):
            raise ProfileError('lxml relink outputs differ from the wheel extensions')
        for item in outputs:
            if (item.get('original_sha256') != native[item['name']]
                    or item.get('relinked_sha256') != native[item['name']]):
                raise ProfileError('lxml relink output differs from the admitted wheel')
        required = {'libiconv-COPYING.LIB', 'libcharset-COPYING.LIB', 'NOTICE.txt',
                    'zlib-LICENSE', 'libxml2-Copyright', 'libxslt-Copyright'}
        if (len(licenses) != len(required)
                or not all(isinstance(item.get('path'), str) for item in licenses)
                or {Path(item['path']).name for item in licenses} != required):
            raise ProfileError('lxml native license manifest is incomplete')
        for item in licenses:
            retained(item, 'native license')
            members = [name for name in archive.namelist()
                       if name.endswith('/' + Path(item['path']).name)]
            if len(members) != 1 or hashlib.sha256(archive.read(members[0])).hexdigest() != item['sha256']:
                raise ProfileError('lxml native license differs from the wheel')


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ProfileError(f'lxml invalid native license manifest: {path}') from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ProfileError('lxml native license manifest must contain file records')
    return value


def _wheel(artifact: dict[str, Any], base: Path, project: dict[str, Any]) -> Path:
    path = _file(artifact, base, f'wheel {artifact.get("name")}')
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            for name in names:
                pure = Path(name)
                if pure.is_absolute() or '..' in pure.parts or '\\' in name:
                    raise ProfileError(f'unsafe wheel member: {name}')
                if name.lower().endswith('.duckdb_extension'):
                    raise ProfileError(f'wheel bundles an external extension: {name}')
                if any(part.endswith('.data') for part in pure.parts):
                    raise ProfileError('wheel .data relocation is unsupported by this profile')
            metadata_names = [name for name in names if name.endswith('.dist-info/METADATA')]
            wheel_names = [name for name in names if name.endswith('.dist-info/WHEEL')]
            if len(metadata_names) != 1 or len(wheel_names) != 1:
                raise ProfileError(f'invalid wheel metadata: {path}')
            metadata = email.message_from_bytes(archive.read(metadata_names[0]))
            name = metadata['Name'].lower().replace('_', '-')
            if name != artifact['name'] or metadata['Version'] != artifact['version']:
                raise ProfileError(f'wheel identity mismatch: {path}')
            tags = email.message_from_bytes(archive.read(wheel_names[0])).get_all('Tag', [])
            supported = {'cp314-cp314-linux_x86_64', 'cp314-cp314-manylinux_2_39_x86_64', 'cp314-cp314-manylinux_2_28_x86_64', 'py3-none-any'}
            if not tags or not any(tag in supported for tag in tags):
                raise ProfileError(f'unsupported wheel platform/ABI: {tags}')
            if name in VERSIONS and not any(tag.startswith('cp314-cp314-') for tag in tags):
                raise ProfileError(f'native wheel must target CPython 3.14: {name}')
    except (OSError, zipfile.BadZipFile, KeyError, AttributeError) as exc:
        raise ProfileError(f'invalid wheel {path}: {exc}') from exc
    receipt_path = _file(artifact.get('source_receipt'), base, 'source_receipt')
    receipt = _read_json(receipt_path)
    for key in ('wheel_sha256', 'source_revision', 'toolchain', 'inputs', 'inputs_sha256', 'native_link_artifacts', 'effective_versions'):
        if key not in receipt:
            raise ProfileError(f'source receipt requires {key}: {receipt_path}')
    if receipt['wheel_sha256'] != artifact['sha256']:
        raise ProfileError('source receipt wheel_sha256 mismatch')
    if not isinstance(receipt['source_revision'], str) or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', receipt['source_revision']):
        raise ProfileError('source receipt requires exact source revision/hash')
    if not isinstance(receipt['toolchain'], dict) or not receipt['toolchain']:
        raise ProfileError('source receipt requires checksummed toolchain')
    for tool, item in receipt['toolchain'].items():
        _file(item, receipt_path.parent, f'toolchain {tool}')
    if not isinstance(receipt['inputs'], list) or not receipt['inputs']:
        raise ProfileError('source receipt requires checksummed inputs')
    for item in receipt['inputs']:
        _file(item, receipt_path.parent, 'receipt input')
    _hash(receipt['inputs_sha256'], 'receipt inputs_sha256')
    actual_inputs = hashlib.sha256(json.dumps(receipt['inputs'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if receipt['inputs_sha256'] != actual_inputs:
        raise ProfileError('receipt inputs_sha256 mismatch')
    if name == 'biominer':
        overlays = [item for item in receipt['inputs'] if Path(item['path']).name == 'pyproject.toml']
        if len(overlays) != 1:
            raise ProfileError('BioMiner source receipt requires exactly one pyproject.toml overlay')
        if overlays[0]['sha256'] != project.get('sha256'):
            raise ProfileError('BioMiner source receipt pyproject.toml does not match profile project')
    if not isinstance(receipt['native_link_artifacts'], list) or (name in VERSIONS and not receipt['native_link_artifacts']):
        raise ProfileError('source receipt requires checksummed native_link_artifacts')
    for item in receipt['native_link_artifacts']:
        _file(item, receipt_path.parent, 'native_link_artifacts')
    if not isinstance(receipt['effective_versions'], dict):
        raise ProfileError('source receipt effective_versions must be an object')
    if name in ('pyarrow', 'duckdb'):
        version = receipt['effective_versions'].get('openssl')
        if not isinstance(version, str) or not re.fullmatch(r'3\.6\.(?:[5-9]|[1-9][0-9]+)', version):
            raise ProfileError(f'{name} receipt requires patched OpenSSL 3.6.5+ on the approved line')
    if name == 'lxml':
        if any(receipt['effective_versions'].get(key) != value
               for key, value in LXML_NATIVE_VERSIONS.items()):
            raise ProfileError('lxml receipt requires the reviewed patched native versions')
        policy = receipt.get('native_policy')
        if (not isinstance(policy, dict)
                or policy.get('zlib_patch_commit') != LXML_ZLIB_PATCH_COMMIT
                or policy.get('zlib_patch_sha256') != LXML_ZLIB_PATCH_SHA256
                or not any(item.get('sha256') == LXML_ZLIB_PATCH_SHA256
                           for item in receipt['inputs'])):
            raise ProfileError('lxml receipt requires the checksummed zlib security patch')
        features = policy.get('libxml_features')
        if (not isinstance(features, list)
                or not all(isinstance(item, str) for item in features)
                or not LXML_FEATURES <= set(features)):
            raise ProfileError('lxml receipt requires the reviewed libxml features')
        _lxml_materials(receipt, receipt_path.parent, path)
    return path


def verify_artifacts(profile: str | Path) -> dict[str, Any]:
    doc, path = load_profile(profile)
    _inputs(doc, path.parent)
    seen: set[str] = set()
    for artifact in doc['artifacts']:
        if not isinstance(artifact, dict) or not isinstance(artifact.get('name'), str):
            raise ProfileError('profile artifact requires name')
        name = artifact['name']
        if name in seen or name == 'psycopg-binary':
            raise ProfileError(f'duplicate or prohibited artifact: {name}')
        seen.add(name)
        if name in VERSIONS:
            version = artifact.get('version')
            accepted = version == VERSIONS[name] if name in ('psycopg-c', 'lxml') else isinstance(version, str) and re.fullmatch(re.escape(VERSIONS[name].split('+')[0]) + r'\+biominer\.(?:1|[0-9a-f]{12})', version)
            if not accepted:
                raise ProfileError(f'unapproved native artifact version: {name}')
        _wheel(artifact, path.parent, doc['project'])
    if seen != set(VERSIONS) | {'biominer'}:
        raise ProfileError('profile requires exactly psycopg-c, pyarrow, duckdb, lxml, biominer wheels')
    return {'status': 'candidate', 'ready': False, 'profile_sha256': sha256(path), 'artifacts': sorted(seen)}


def launch_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    source = dict(os.environ if source is None else source)
    for key in BLOCKED_ENV:
        if source.get(key):
            raise ProfileError(f'unapproved loader environment: {key}')
    # Discard inherited uv/pip settings, shell startup paths, and user-site state.
    environment = {key: value for key, value in source.items() if not key.startswith(('UV_', 'PIP_', 'PYTHON', 'LD_')) and key not in ('VIRTUAL_ENV', 'HOME')}
    environment['PYTHONNOUSERSITE'] = '1'
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['HOME'] = '/nonexistent'
    environment['GDAL_DRIVER_PATH'] = 'disable'
    return environment


def _bounded_process(argv: list[str], *, cwd: Path | None, env: dict[str, str], timeout: int, output: Any = subprocess.PIPE) -> str:
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except BaseException:
        # The recipe may have spawned pip/CMake/compiler workers. Killing just
        # its interpreter would leave those workers modifying failed outputs.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, argv)
    return stdout or ''


def _run(argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int = 300, log: Path | None = None) -> str:
    try:
        if log is not None:
            with log.open('x') as target:
                _bounded_process(argv, cwd=cwd, env=env or launch_environment(), timeout=timeout, output=target)
            return ''
        result = _bounded_process(argv, cwd=cwd, env=env or launch_environment(), timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ProfileError(f'command timed out after {timeout}s: {argv[0]}') from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        # Do not echo subprocess output, which may contain connection secrets.
        raise ProfileError(f'command failed: {argv[0]} ({getattr(exc, "returncode", "unavailable")})') from exc
    return result


def _host(doc: dict[str, Any]) -> list[dict[str, str]]:
    try:
        release = dict(line.split('=', 1) for line in Path('/etc/os-release').read_text().splitlines() if '=' in line)
    except OSError as exc:
        raise ProfileError('cannot attest host release') from exc
    if release.get('ID', '').strip('"') != 'ubuntu' or release.get('VERSION_ID', '').strip('"') != '26.04':
        raise ProfileError('host requires Ubuntu 26.04 security package provenance')
    required = {'libssl3t64', 'libssl-dev', 'openssl', 'libpq5'}
    names = {item.get('name') for item in doc['host_packages'] if isinstance(item, dict)}
    if not required <= names:
        raise ProfileError('host_packages requires libssl3t64, libssl-dev, openssl, libpq5')
    global_policy = _run(['/usr/bin/apt-cache', 'policy'])
    if 'o=Ubuntu' not in global_policy:
        raise ProfileError('host apt metadata lacks Ubuntu Origin')
    observed = []
    for item in doc['host_packages']:
        if item.get('origin') != 'Ubuntu' or item.get('architecture') != 'amd64' or not item.get('advisory'):
            raise ProfileError('host package requires Ubuntu amd64 origin and advisory')
        name = item['name']
        if not re.fullmatch(r'[a-z0-9][a-z0-9+.-]*', name):
            raise ProfileError('invalid host package name')
        minimum = item.get('minimum_version')
        if not isinstance(minimum, str) or not minimum:
            raise ProfileError('host package requires minimum_version')
        if name in {'libssl3t64', 'libssl-dev', 'openssl'}:
            try:
                _run(['/usr/bin/dpkg', '--compare-versions', minimum, 'ge', '3.5.5-1ubuntu3.7'])
            except ProfileError as exc:
                raise ProfileError(f'host {name} profile minimum {minimum} is below reviewed 3.5.5-1ubuntu3.7 floor') from exc
        output = _run(['/usr/bin/dpkg-query', '-W', '-f=${Version}\t${Architecture}\t${Status}', name]).strip().split('\t')
        if len(output) != 3 or output[1] != 'amd64' or output[2] != 'install ok installed':
            raise ProfileError(f'host package unavailable: {name}')
        try:
            _run(['/usr/bin/dpkg', '--compare-versions', output[0], 'ge', item['minimum_version']])
        except ProfileError as exc:
            raise ProfileError(f'host {name} installed {output[0]} below required {minimum}; operator update required') from exc
        policy = _run(['/usr/bin/apt-cache', 'policy', name])
        stanza = []
        inside = False
        for line in policy.splitlines():
            header = re.fullmatch(r'\s*(\*\*\*\s+)?(\S+)\s+\d+\s*', line)
            if header:
                if inside:
                    break
                inside = bool(header.group(1)) and header.group(2) == output[0]
            elif inside:
                stanza.append(line)
        if not inside:
            raise ProfileError(f'host installed-version apt evidence absent: {name}')
        repositories = re.findall(r'\b(https?://\S+)\s+resolute(?:-[a-z]+)?/', '\n'.join(stanza))
        official = re.compile(r'https?://(?:(?:[a-z]{2}\.)?archive\.ubuntu\.com|security\.ubuntu\.com|ports\.ubuntu\.com)/ubuntu/?$')
        if not repositories or any(not official.fullmatch(url) for url in repositories):
            raise ProfileError(f'host installed-version lacks exclusive official Ubuntu origin: {name}')
        # Require receipt pinning of the policy that established package origin.
        if not isinstance(item.get('apt_policy_sha256'), str) or hashlib.sha256(policy.encode()).hexdigest() != item['apt_policy_sha256']:
            raise ProfileError(f'host apt origin evidence changed or absent: {name}')
        observed.append({'name': name, 'version': output[0], 'architecture': output[1]})
    return observed


def _load_python_attestor(record: dict[str, Any], base: Path):
    script = _file(record, base, 'Python runtime attestor')
    spec = importlib.util.spec_from_file_location('biominer_python_attestor', script)
    if spec is None or spec.loader is None:
        raise ProfileError('cannot load Python runtime attestor')
    module = importlib.util.module_from_spec(spec)
    exec(compile(script.read_bytes(), str(script), "exec"), module.__dict__)
    return module


def verify_python_prefix(doc: dict[str, Any], base: Path, python: Path) -> None:
    receipts = [item for item in doc['inputs'] if item.get('kind') == 'python-source-receipt']
    attestors = [item for item in doc['inputs'] if Path(item.get('path', '')).name == 'attest_native_python.py']
    if len(receipts) != 1 or len(attestors) != 1:
        raise ProfileError('Python runtime requires one source receipt and hashed archive attestor')
    receipt = _read_json(_file(receipts[0], base, 'Python runtime source receipt'))
    module = _load_python_attestor(attestors[0], base)
    if (receipt.get('schema_version') != 1 or receipt.get('source_build') != doc['python'].get('source_build')
            or receipt.get('url') != module.URL
            or receipt.get('identity') != {'version': '3.14.7', 'openssl_version': '3.5.9'}):
        raise ProfileError('Python runtime source identity differs')
    if _file(receipt.get('python'), base, 'Python runtime receipt executable') != python:
        raise ProfileError('Python runtime executable differs from source receipt')
    archive = _file(receipt.get('archive'), base, 'Python runtime archive')
    try:
        module.verify_archive(archive)
        matched = module.verify_prefix(archive, python.parent.parent)
    except (ValueError, OSError) as exc:
        raise ProfileError(f'Python runtime archive/prefix verification failed: {exc}') from exc
    if matched != receipt.get('verified_archive_files'):
        raise ProfileError('Python runtime archive file inventory differs from source receipt')


def _python(doc: dict[str, Any], base: Path) -> Path:
    python = _file(doc['python'], base, 'python')
    if doc['python'].get('version') != '3.14.7' or doc['python'].get('openssl_version') != '3.5.9':
        raise ProfileError('approved Python identity must be 3.14.7 / OpenSSL 3.5.9')
    verify_python_prefix(doc, base, python)
    output = _run([str(python), '-I', '-S', '-B', '-c', 'import json,ssl,platform; print(json.dumps({"version":platform.python_version(),"openssl_version":ssl.OPENSSL_VERSION.split()[1]}))'])
    try:
        observed = json.loads(output)
    except ValueError as exc:
        raise ProfileError('Python identity probe did not return JSON') from exc
    if not isinstance(observed, dict) or any(observed.get(key) != doc['python'][key] for key in ('version', 'openssl_version')):
        raise ProfileError(f'actual Python/OpenSSL identity differs: {observed}')
    return python


def _write(path: Path, value: Any) -> None:
    with path.open('x') as target:
        json.dump(value, target, indent=2, sort_keys=True)
        target.write('\n')


def _new_output(path: Path, generation_root: str | None = None) -> None:
    if path.exists() or path.is_symlink():
        raise ProfileError(f'generation/output already exists: {path}')
    resolved = path.resolve()
    if generation_root is not None:
        if not isinstance(generation_root, str):
            raise ProfileError('profile generation_root must be an absolute directory')
        root = Path(generation_root)
        if (not root.is_absolute() or root.name != 'generations'
                or root.parent.name != '.native-runtime' or root.resolve() != root):
            raise ProfileError('profile generation_root must be an unaliased .native-runtime/generations directory')
        if resolved == root or not resolved.is_relative_to(root) or path.absolute() != resolved:
            raise ProfileError('generation must be below the profile generation_root without symlinks')
    else:
        repository_root = Path(__file__).resolve().parents[1]
        if not (resolved.is_relative_to(Path('/tmp')) or resolved.is_relative_to(repository_root / '.native-runtime')):
            raise ProfileError('outputs must be under /tmp or repository .native-runtime')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(mode=0o700)


def build(profile: str | Path, output: str | Path) -> dict[str, Any]:
    doc, path = load_profile(profile)
    _inputs(doc, path.parent)
    if not doc['recipes']:
        raise ProfileError('profile has no explicit build recipes')
    python = _python(doc, path.parent)
    scripts = []
    for recipe in doc['recipes']:
        if not isinstance(recipe, dict) or not isinstance(recipe.get('args', []), list):
            raise ProfileError('build recipe requires script and args array')
        timeout = recipe.get('timeout_seconds', 14400)
        if not isinstance(timeout, int) or not 1 <= timeout <= 43200:
            raise ProfileError('recipe timeout_seconds must be 1..43200')
        scripts.append((_file(recipe.get('script'), path.parent, 'recipe script'), recipe.get('args', []), timeout))
    output = Path(output).absolute()
    _new_output(output)
    try:
        for index, (script, args, timeout) in enumerate(scripts):
            if any(not isinstance(arg, str) for arg in args):
                raise ProfileError('recipe arguments must be strings')
            _run([str(python), '-I', str(script), *[arg.replace('{output}', str(output)).replace('{profile}', str(path)) for arg in args]], cwd=output, timeout=timeout, log=output / f'recipe-{index}.log')
        files = [{'path': str(file.relative_to(output)), 'sha256': sha256(file)} for file in sorted(output.rglob('*')) if file.is_file() and not file.is_symlink()]
        receipt = {'schema_version': 1, 'status': 'candidate', 'ready': False, 'profile_sha256': sha256(path), 'outputs': files}
        _write(output / 'build-receipt.json', receipt)
        return receipt
    except Exception:
        _write(output / 'FAILED.json', {'status': 'failed', 'ready': False, 'phase': 'build'})
        raise


def _inventory(root: Path) -> dict[str, dict[str, str]]:
    result = {}
    for file in sorted(root.rglob('*')):
        if file.is_symlink():
            result[str(file.relative_to(root))] = {'symlink': os.readlink(file), 'target_sha256': sha256(file.resolve()) if file.resolve().is_file() else ''}
        elif file.is_file():
            result[str(file.relative_to(root))] = {'sha256': sha256(file)}
    return result


def _freeze(root: Path) -> None:
    for path in sorted(root.rglob('*'), reverse=True):
        if not path.is_symlink():
            mode = path.stat().st_mode
            path.chmod(0o555 if path.is_dir() or mode & 0o111 else 0o444)
    root.chmod(0o555)


def _checks(doc: dict[str, Any], profile: Path, generation: Path, *, installation: bool) -> list[str]:
    required = INSTALL_CHECKS if installation else RUNTIME_CHECKS
    applicable = [check for check in doc['checks'] if installation or check.get('name') in RUNTIME_CHECKS]
    names = [check.get('name') for check in applicable]
    if not required <= set(names) or len(names) != len(set(names)):
        raise ProfileError(f'missing/duplicate required checks: {sorted(required - set(names))}')
    for check in applicable:
        script = _file(check.get('script'), profile.parent, f'check {check["name"]}')
        args = check.get('args', [])
        if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
            raise ProfileError('check args must be a string array')
        output = _run([str(generation / 'env/bin/python'), '-I', '-B', str(script), *[arg.replace('{generation}', str(generation)).replace('{profile}', str(profile)) for arg in args]])
        try:
            receipt = json.loads(output)
        except ValueError as exc:
            raise ProfileError(f'check {check["name"]} returned invalid JSON') from exc
        if not isinstance(receipt, dict) or receipt.get('status') != 'passed' or receipt.get('profile_sha256') != sha256(profile):
            raise ProfileError(f'check {check["name"]} did not attest this profile')
        if check['name'] == 'extension_policy' and receipt.get('policy_id') != doc['extension_policy']['policy_id']:
            raise ProfileError('compiled extension policy_id mismatch')
    return names


def _check_inputs(doc: dict[str, Any], base: Path) -> None:
    if any(not isinstance(check, dict) or not isinstance(check.get('name'), str) for check in doc['checks']):
        raise ProfileError('profile checks require named objects')
    names = [check['name'] for check in doc['checks']]
    if not INSTALL_CHECKS <= set(names) or len(names) != len(set(names)):
        raise ProfileError('missing/duplicate mandatory checks')
    for check in doc['checks']:
        _file(check.get('script'), base, f'check {check["name"]}')
    if not doc['native_files']:
        raise ProfileError('profile native_files must contain approved libpq/SSL/crypto mappings')
    owners = set()
    for native in doc['native_files']:
        if not isinstance(native, dict) or not isinstance(native.get('owner'), str):
            raise ProfileError('native file requires locked wheel or dpkg owner')
        owners.add(native['owner'])
        _hash(native.get('sha256'), 'native file')
        if not isinstance(native.get('path'), str):
            raise ProfileError('native file requires path')
    if not {'dpkg:libpq5', 'dpkg:libssl3t64'} <= owners:
        raise ProfileError('native_files requires dpkg:libpq5 and dpkg:libssl3t64')


def _installed_wheels(doc: dict[str, Any], profile: Path, generation: Path) -> None:
    site = generation / 'env/lib/python3.14/site-packages'
    if not site.is_dir():
        raise ProfileError('generation lacks CPython 3.14 site-packages')
    if list(site.glob('psycopg_binary*')) or list(site.glob('psycopg-binary*')):
        raise ProfileError('psycopg-binary is prohibited')
    for artifact in doc['artifacts']:
        wheel = _wheel(artifact, profile.parent, doc['project'])
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                if member.is_dir() or member.filename.endswith('/RECORD'):
                    continue
                installed = site / member.filename
                if not installed.is_file() or sha256(installed) != hashlib.sha256(archive.read(member)).hexdigest():
                    raise ProfileError(f'installed wheel bytes differ: {member.filename}')


def verify_native_file(native: dict[str, Any], profile: str | Path, generation: str | Path) -> None:
    """Verify a declared native file's bytes and real package ownership."""
    profile = Path(profile).resolve()
    generation = Path(generation).resolve()
    item = dict(native)
    if not isinstance(item.get('path'), str) or not isinstance(item.get('owner'), str):
        raise ProfileError('native file requires path and owner')
    item['path'] = item['path'].replace('{generation}', str(generation))
    file = _file(item, profile.parent, 'native file')
    owner = item['owner']
    if owner.startswith('dpkg:'):
        package = owner.removeprefix('dpkg:')
        if not re.fullmatch(r'[a-z0-9][a-z0-9+.-]*', package):
            raise ProfileError('invalid native dpkg owner')
        output = _run(['/usr/bin/dpkg-query', '-S', str(file)])
        owners = {line.split(': ', 1)[0].split(':', 1)[0] for line in output.splitlines() if ': ' in line}
        if owners != {package}:
            raise ProfileError(f'native file dpkg owner mismatch: {file}')
    elif owner.startswith('python-runtime:'):
        doc = _read_json(profile)
        python = _file(doc.get('python'), profile.parent, 'python runtime')
        source_build = doc['python'].get('source_build')
        if not source_build or owner != f'python-runtime:{source_build}' or not file.is_relative_to(python.parent.parent):
            raise ProfileError(f'native file Python runtime owner/prefix mismatch: {file}')
    elif owner.startswith('wheel:'):
        package = owner.removeprefix('wheel:')
        if not re.fullmatch(r'[a-z0-9][a-z0-9_.-]*', package):
            raise ProfileError('invalid native wheel owner')
        code = 'import importlib.metadata as m,pathlib,sys; d=m.distribution(sys.argv[1]); p=pathlib.Path(sys.argv[2]).resolve(); sys.exit(0 if any(pathlib.Path(d.locate_file(f)).resolve()==p for f in (d.files or [])) else 1)'
        try:
            _run([str(generation / 'env/bin/python'), '-I', '-B', '-c', code, package, str(file)])
        except ProfileError as exc:
            raise ProfileError(f'native file wheel owner mismatch: {file}') from exc
    else:
        raise ProfileError('native file owner must use dpkg:, wheel:, or python-runtime: namespace')


def install(profile: str | Path, generation: str | Path) -> dict[str, Any]:
    generation = Path(generation).absolute()
    if generation.exists() or generation.is_symlink():
        raise ProfileError(f'generation already exists: {generation}')
    verify_artifacts(profile)
    doc, path = load_profile(profile)
    _check_inputs(doc, path.parent)
    host = _host(doc)
    python = _python(doc, path.parent)
    uv = _file(doc['uv'], path.parent, 'uv')
    _new_output(generation, doc.get('generation_root'))
    try:
        project = generation / 'project'
        project.mkdir()
        shutil.copyfile(_file(doc['project'], path.parent, 'project'), project / 'pyproject.toml')
        shutil.copyfile(_file(doc['base_lock'], path.parent, 'base_lock'), project / 'uv.lock')
        environment = launch_environment()
        environment['UV_PROJECT_ENVIRONMENT'] = str(generation / 'env')
        environment['UV_CACHE_DIR'] = str(generation / 'cache')
        _run([str(uv), 'venv', '--python', str(python), str(generation / 'env')], env=environment)
        _run([str(uv), 'sync', '--locked', '--no-editable', '--no-install-project', '--extra', 'postgres', '--extra', 'geo', '--extra', 'ml', '--no-install-package', 'psycopg-c', '--no-install-package', 'pyarrow', '--no-install-package', 'duckdb', '--no-install-package', 'lxml', '--project', str(project)], env=environment, timeout=3600)
        requirements = generation / 'native-wheels.txt'
        requirements.write_text('\n'.join(f'{_file(artifact, path.parent, "wheel").as_uri()} --hash=sha256:{artifact["sha256"]}' for artifact in doc['artifacts']) + '\n')
        _run([str(uv), 'pip', 'install', '--python', str(generation / 'env/bin/python'), '--no-index', '--no-deps', '--require-hashes', '-r', str(requirements)], env=environment)
        _installed_wheels(doc, path, generation)
        checks = _checks(doc, path, generation, installation=True)
        # This cache belongs to this new generation; installed files remain.
        if (generation / 'cache').is_dir() and not (generation / 'cache').is_symlink():
            shutil.rmtree(generation / 'cache')
        receipt = {'schema_version': 1, 'status': 'installed', 'ready': False, 'profile_sha256': sha256(path), 'host_packages': host, 'checks': checks, 'files': _inventory(generation)}
        _write(generation / 'generation.json', receipt)
        _freeze(generation)
        return verify(path, generation)
    except Exception:
        if os.access(generation, os.W_OK) and not (generation / 'FAILED.json').exists():
            _write(generation / 'FAILED.json', {'status': 'failed', 'ready': False, 'phase': 'install'})
        raise


def verify(profile: str | Path, generation: str | Path | None = None) -> dict[str, Any]:
    verify_artifacts(profile)
    if generation is None:
        raise ProfileError('generation is required for readiness; artifacts alone are candidate evidence')
    doc, path = load_profile(profile)
    _check_inputs(doc, path.parent)
    generation = Path(generation).resolve()
    receipt = _read_json(generation / 'generation.json')
    if receipt.get('status') != 'installed' or receipt.get('profile_sha256') != sha256(path):
        raise ProfileError('generation receipt/profile mismatch')
    if not INSTALL_CHECKS <= set(receipt.get('checks', [])):
        raise ProfileError('generation lacks mandatory installation checks')
    inventory = _inventory(generation)
    inventory.pop('generation.json', None)
    if inventory != receipt.get('files'):
        raise ProfileError('generation inventory changed')
    for file in [generation, *generation.rglob('*')]:
        if not file.is_symlink() and (file.stat().st_uid != os.getuid() or file.stat().st_mode & 0o222):
            raise ProfileError(f'generation is not owned and read-only: {file}')
    _host(doc)
    _python(doc, path.parent)
    for native in doc['native_files']:
        verify_native_file(native, path, generation)
    _installed_wheels(doc, path, generation)
    checks = _checks(doc, path, generation, installation=False)
    return {'status': 'verified', 'ready': True, 'profile_sha256': sha256(path), 'generation': str(generation), 'checks': checks, 'live_services': 'not_attested'}


def execute(profile: str | Path, generation: str | Path, args: list[str]) -> None:
    environment = launch_environment()
    verify(profile, generation)
    if not args:
        raise ProfileError('exec requires Python arguments, for example -m biominer.cli')
    python = str(Path(generation).resolve() / 'env/bin/python')
    os.execve(python, [python, '-I', '-B', *args], environment)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('build', 'install', 'verify', 'exec', 'artifacts'):
        child = commands.add_parser(command, allow_abbrev=False)
        child.add_argument('--profile', required=True, type=Path)
        child.add_argument('--generation', type=Path)
        child.add_argument('--output', type=Path)
        if command == 'exec':
            child.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.command == 'build':
            if args.output is None:
                raise ProfileError('build requires --output')
            result = build(args.profile, args.output)
        elif args.command == 'artifacts':
            result = verify_artifacts(args.profile)
        elif args.command == 'verify':
            result = verify(args.profile, args.generation)
        else:
            if args.generation is None:
                raise ProfileError(f'{args.command} requires --generation')
            if args.command == 'exec':
                execute(args.profile, args.generation, args.args[1:] if args.args[:1] == ['--'] else args.args)
                return 0
            result = install(args.profile, args.generation)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ProfileError, OSError) as exc:
        print(json.dumps({'status': 'blocked', 'ready': False, 'reason': str(exc)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
