#!/usr/bin/env python3
"""Rebind a candidate's deployment libraries after an operator Ubuntu update.

Build receipts remain historical evidence. A refreshed profile requires a new
immutable installation and full verification before it can run production.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

# Direct invocation with isolated Python does not put this directory on sys.path.
_spec = importlib.util.spec_from_file_location('native_runtime', Path(__file__).with_name('native_runtime.py'))
assert _spec and _spec.loader
runtime = importlib.util.module_from_spec(_spec)
exec(compile(Path(_spec.origin).read_bytes(), _spec.origin, 'exec'), runtime.__dict__)


def refresh_host(profile: str | Path, output: str | Path) -> dict:
    doc, profile = runtime.load_profile(profile)
    output = Path(output).absolute()
    receipt_path = output.with_suffix(output.suffix + '.host-refresh.json')
    if output.exists() or receipt_path.exists() or output == profile:
        raise runtime.ProfileError('host refresh requires a new profile and receipt')
    # Preserve relative file references and their exact recorded build provenance.
    if output.parent.resolve() != profile.parent:
        raise runtime.ProfileError('refreshed profile must share the original profile directory')
    initial_hash = runtime.sha256(profile)
    runtime.verify_artifacts(profile)
    updated = copy.deepcopy(doc)
    updated["ready"] = False
    updated["status"] = "candidate"
    packages = {item['name']: item for item in updated['host_packages']}
    changes = []
    if not updated['native_files']:
        raise runtime.ProfileError('host refresh requires native file inventory')
    for item in updated['native_files']:
        if not isinstance(item.get('path'), str) or not isinstance(item.get('owner'), str):
            raise runtime.ProfileError('native file requires path and owner')
        owner = item['owner']
        if not owner.startswith('dpkg:'):
            # Generation placeholders are verified during new generation install.
            if '{generation}' not in item.get('path', ''):
                runtime._file(item, profile.parent, 'unchanged native file')
            continue
        package = owner.removeprefix('dpkg:')
        if not re.fullmatch(r'[a-z0-9][a-z0-9+.-]*', package):
            raise runtime.ProfileError('invalid native dpkg owner')
        path = Path(item['path'])
        path = (profile.parent / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file():
            raise runtime.ProfileError(f'host native file missing: {path}')
        owners = {line.split(': ', 1)[0].split(':', 1)[0] for line in runtime._run(['/usr/bin/dpkg-query', '-S', str(path)]).splitlines() if ': ' in line}
        if owners != {package}:
            raise runtime.ProfileError(f'host native file owner mismatch: {path}')
        observed = runtime._run(['/usr/bin/dpkg-query', '-W', '-f=${Version}\t${Architecture}\t${Status}', package]).strip().split('\t')
        if len(observed) != 3 or observed[1:] != ['amd64', 'install ok installed']:
            raise runtime.ProfileError(f'host native package unavailable: {package}')
        new_hash = runtime.sha256(path)
        if package not in packages:
            packages[package] = {'name': package, 'minimum_version': observed[0], 'architecture': 'amd64', 'origin': 'Ubuntu', 'advisory': 'Native dependency deployment version pin; no security-fix assertion'}
            updated['host_packages'].append(packages[package])
        if new_hash != item['sha256']:
            changes.append({'path': item['path'], 'owner': owner, 'old_sha256': item['sha256'], 'sha256': new_hash, 'version': observed[0]})
        item['sha256'] = new_hash
    for package in updated['host_packages']:
        policy = runtime._run(['/usr/bin/apt-cache', 'policy', package['name']])
        package['apt_policy_sha256'] = hashlib.sha256(policy.encode()).hexdigest()
    observed_host = runtime._host(updated)
    # Recheck all artifacts and source receipts after observing host state.
    runtime.verify_artifacts(profile)
    if runtime.sha256(profile) != initial_hash:
        raise runtime.ProfileError('source profile changed during host refresh')
    for item in updated['native_files']:
        if '{generation}' not in item.get('path', ''):
            runtime._file(item, profile.parent, 'refreshed native file')
    runtime._host(updated)
    result = {'schema_version': 1, 'status': 'candidate', 'ready': False, 'source_profile_sha256': initial_hash, 'changes': changes, 'host_packages': observed_host, 'required_next_step': 'install a new immutable generation and pass full verification'}
    # Exclusive creation protects source profiles and pre-existing generations.
    with output.open('x') as handle:
        json.dump(updated, handle, indent=2, sort_keys=True)
        handle.write('\n')
    result['profile_sha256'] = runtime.sha256(output)
    with receipt_path.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = refresh_host(args.profile, args.output)
    except (runtime.ProfileError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'blocked', 'ready': False, 'error': str(exc)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
