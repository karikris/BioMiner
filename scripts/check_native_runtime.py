#!/usr/bin/env python3
"""Offline native capability evidence; this script alone never admits a host."""
from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import mmap
from pathlib import Path
import re
import ssl
import subprocess
import sys
import tempfile


IMPORT_ORDERS = (
    ('pyarrow.flight', 'pyarrow.parquet.encryption', 'psycopg.pq', 'duckdb'),
    ('duckdb', 'psycopg.pq', 'pyarrow.parquet.encryption', 'pyarrow.flight'),
)
BYPASS_SETTINGS = ('allow_unsigned_extensions', 'allow_extensions_metadata_mismatch')
# CPython v3.14.7, Modules/clinic/_hashopenssl.c.h:1877-1888; GitHits
# resolved source github:python/cpython@823f0323. This exact docstring describes
# FIPS API behavior, not the linked provider version. PSF-2.0 source.
PYTHON_FIPS_DOC = (
    b'get_fips_mode($module, /)\n--\n\nDetermine the OpenSSL FIPS mode of operation.\n\n'
    b'For OpenSSL 3.0.0 and newer it returns the state of the default provider\n'
    b"in the default OSSL context. It's not quite the same as FIPS_mode() but\n"
    b'good enough for unittests.\n\n'
    b'Effectively any non-zero return value indicates FIPS mode;\n'
    b'values other than 1 may have additional significance.'
)


def observed_crypto_versions(path: Path, owner: str) -> set[str]:
    versions = set()
    marker = re.compile(rb'OpenSSL (3\.[0-9]+\.[0-9]+)(?: |\x00)')
    offset = PYTHON_FIPS_DOC.index(b'OpenSSL 3.0.0')
    with path.open('rb') as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as contents:
        for match in marker.finditer(contents):
            version = match[1].decode()
            start = match.start() - offset
            if (version == '3.0.0' and owner.startswith('python-runtime:') and start >= 0
                    and (start == 0 or contents[start - 1] == 0)
                    and contents[start:start + len(PYTHON_FIPS_DOC) + 1] == PYTHON_FIPS_DOC + b'\0'):
                continue
            versions.add(version)
    return versions


def check_psycopg_implementation(implementation: str) -> None:
    if implementation != 'c':
        raise ValueError('Psycopg must use the system-linked C implementation')


def check_extension_settings(duckdb) -> None:
    for setting in BYPASS_SETTINGS:
        try:
            connection = duckdb.connect(config={setting: 'true'})
        except duckdb.Error:
            continue
        if hasattr(connection, 'close'):
            connection.close()
        raise ValueError(f'signature bypass configuration accepted: {setting}')
    for setting in BYPASS_SETTINGS:
        connection = duckdb.connect()
        try:
            try:
                connection.execute(f'SET {setting} = true')
            except duckdb.Error:
                continue
            raise ValueError(f'signature bypass SET accepted: {setting}')
        finally:
            connection.close()


def check_compiled_policy(policy_id: str) -> None:
    import _duckdb
    exported = ctypes.CDLL(_duckdb.__file__).biominer_extension_policy_id
    exported.restype = ctypes.c_char_p
    if exported().decode('ascii') != policy_id:
        raise ValueError('compiled extension policy does not match native profile')


def check_geometry(connection, directory: Path) -> dict:
    """Exercise local geometry formats and current planar predicate semantics."""
    directory.mkdir(parents=True, exist_ok=True)
    geometry = {'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'properties': {'country': 'fixture'},
         'geometry': {'type': 'Polygon', 'coordinates': [
             [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
             [[1, 1], [1, 3], [3, 3], [3, 1], [1, 1]]]}},
        {'type': 'Feature', 'properties': {'country': 'multipolygon'},
         'geometry': {'type': 'MultiPolygon', 'coordinates': [
             [[[5, 5], [6, 5], [6, 6], [5, 6], [5, 5]]],
             [[[8, 8], [9, 8], [9, 9], [8, 9], [8, 8]]]]}},
    ]}
    boundary = directory / 'boundary.geojson'
    boundary.write_text(json.dumps(geometry))
    results = connection.execute(
        'SELECT ST_Intersects(geom, ST_Point(0.5,0.5)), '
        'ST_Intersects(geom, ST_Point(2,2)), '
        'ST_Intersects(geom, ST_Point(4,2)) FROM ST_Read(?) WHERE country=?',
        [str(boundary), 'fixture'],
    ).fetchone()
    if results != (True, False, True):
        raise ValueError(f'Spatial polygon/hole/border behavior changed: {results}')
    multi = connection.execute(
        'SELECT ST_Intersects(geom, ST_Point(5.5,5.5)), '
        'ST_Intersects(geom, ST_Point(8.5,8.5)), '
        'ST_Intersects(geom, ST_Point(7,7)) FROM ST_Read(?) WHERE country=?',
        [str(boundary), 'multipolygon'],
    ).fetchone()
    if multi != (True, True, False):
        raise ValueError(f'Spatial multipolygon behavior changed: {multi}')
    shapefile = directory / 'boundary.shp'
    quoted = str(shapefile).replace("'", "''")
    connection.execute(
        f"COPY (SELECT * FROM ST_Read(?)) TO '{quoted}' "
        "WITH (FORMAT GDAL, DRIVER 'ESRI Shapefile')", [str(boundary)],
    )
    names = 'SELECT country FROM ST_Read(?) ORDER BY country'
    if connection.execute(names, [str(shapefile)]).fetchall() != connection.execute(names, [str(boundary)]).fetchall():
        raise ValueError('Spatial shapefile attribute behavior changed')
    query = 'SELECT country, ST_Intersects(geom, ST_Point(?,?)) FROM ST_Read(?) ORDER BY country'
    for x, y in ((0.5, 0.5), (2, 2), (4, 2), (5.5, 5.5), (8.5, 8.5), (7, 7)):
        expected = connection.execute(query, [x, y, str(boundary)]).fetchall()
        actual = connection.execute(query, [x, y, str(shapefile)]).fetchall()
        if actual != expected:
            raise ValueError(f'Spatial shapefile geometry differs at {x},{y}')
    dateline = connection.execute(
        "SELECT ST_Intersects(ST_GeomFromText("
        "'POLYGON ((170 -10,-170 -10,-170 10,170 10,170 -10))'), ST_Point(0,0)), "
        "ST_Intersects(ST_GeomFromText('POLYGON ((170 -10,-170 -10,-170 10,170 10,170 -10))'), ST_Point(180,0))",
    ).fetchone()
    if dateline != (True, False):
        raise ValueError('Existing planar dateline behavior changed')
    return {'polygon_hole_border': True, 'multipolygon': True, 'shapefile': True, 'planar_dateline': True}


def check_capabilities(policy_id: str) -> dict:
    import duckdb
    import psycopg.pq
    import pyarrow as pa
    import pyarrow.fs as fs
    import pyarrow.parquet as parquet

    check_psycopg_implementation(psycopg.pq.__impl__)
    versions = {name: importlib.metadata.version(name) for name in ('duckdb', 'pyarrow', 'psycopg-c')}
    for name in ('duckdb', 'pyarrow'):
        if '+biominer.' not in versions[name]:
            raise ValueError(f'{name} is not an explicitly versioned custom wheel')
    for name in ('pyarrow.flight', 'pyarrow.parquet.encryption', 'pyarrow.dataset',
                 'pyarrow.compute', 'pyarrow.acero', 'pyarrow.orc', 'pyarrow.json'):
        importlib.import_module(name)
    for name in ('S3FileSystem', 'AzureFileSystem', 'GcsFileSystem', 'HadoopFileSystem'):
        if not hasattr(fs, name):
            raise ValueError(f'Arrow filesystem capability missing: {name}')
    table = pa.table({'value': [1, 2]})
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        parquet.write_table(table, path / 'sample.parquet')
        if not parquet.read_table(path / 'sample.parquet').equals(table):
            raise ValueError('Arrow Parquet round trip failed')
        with duckdb.connect() as connection:
            connection.execute('LOAD spatial')
            check_compiled_policy(policy_id)
            geometry = check_geometry(connection, path)
    return {'status': 'capabilities_passed', 'versions': versions,
            'python_openssl': ssl.OPENSSL_VERSION, 'libpq_version': psycopg.pq.version(),
            'extension_policy_id': policy_id, 'geometry': geometry,
            'live_postgres': 'unrun', 'live_s3': 'unrun'}


def check_input_binding(profile: dict, path: Path, script: Path) -> None:
    script = script.resolve()
    expected = hashlib.sha256(script.read_bytes()).hexdigest()
    for item in profile.get('inputs', []):
        candidate = Path(item.get('path', ''))
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        if candidate.resolve() == script and item.get('sha256') == expected:
            return
    raise ValueError('runtime check source is absent or changed in profile inputs')


def check_source_binding(profile: dict, path: Path) -> None:
    check_input_binding(profile, path, Path(__file__))


def check_crypto_versions(owner: str, versions: set[str]) -> None:
    if owner == 'dpkg:libssl3t64':
        # The helper separately validates the Ubuntu backport revision, origin,
        # file hash and ownership; its upstream version remains 3.5.5.
        return
    approved = {'3.5.9'} if owner.startswith('python-runtime:') else {'3.6.5'}
    if not versions <= approved:
        raise ValueError(f'affected or wrong-line embedded crypto: {owner}: {versions}')


def check_effective_versions(profile: dict) -> None:
    for artifact in profile['artifacts']:
        if importlib.metadata.version(artifact['name']) != artifact['version']:
            raise ValueError(f'effective package version changed: {artifact["name"]}')
    import psycopg.pq
    check_psycopg_implementation(psycopg.pq.__impl__)
    if ssl.OPENSSL_VERSION.split()[1] != profile['python']['openssl_version']:
        raise ValueError('effective Python OpenSSL changed')


def check_native_inventory(profile: dict, generation: Path, order: tuple[str, ...]) -> list[dict]:
    if order not in IMPORT_ORDERS:
        raise ValueError('unsupported native import order')
    for name in order:
        importlib.import_module(name)
    check_effective_versions(profile)
    import duckdb
    with duckdb.connect() as connection:
        connection.execute('LOAD spatial')
    check_compiled_policy(profile['extension_policy']['policy_id'])
    approved = {str(Path(item['path'].replace('{generation}', str(generation))).resolve()): item
                for item in profile['native_files']}
    mapped = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/'):
            mapped.add(fields[5])
    distribution_files = {}
    for distribution in importlib.metadata.distributions():
        for entry in distribution.files or ():
            if entry.hash and entry.hash.mode == 'sha256':
                target = Path(distribution.locate_file(entry)).resolve()
                if target.is_relative_to(generation.resolve()):
                    distribution_files[str(target)] = entry.hash.value
    found = []
    for filename in sorted(mapped):
        library = Path(filename).resolve()
        if not library.is_file():
            raise ValueError(f'loaded native file disappeared or cannot be attested: {library}')
        with library.open('rb') as source:
            if source.read(4) != b'\x7fELF':
                continue
            source.seek(0)
            actual = hashlib.file_digest(source, 'sha256').hexdigest()
        item = approved.get(str(library))
        if item is not None:
            if actual != item['sha256']:
                raise ValueError(f'loaded native file hash changed: {library}')
        else:
            record = distribution_files.get(str(library))
            encoded = base64.urlsafe_b64encode(bytes.fromhex(actual)).rstrip(b'=').decode()
            if record != encoded:
                raise ValueError(f'loaded native file lacks approved ownership/hash: {library}')
        versions = observed_crypto_versions(library, item['owner'] if item else '')
        crypto = library.name.startswith(('libssl.', 'libcrypto.'))
        if not versions and not crypto:
            continue
        if item is None:
            raise ValueError(f'unapproved loaded crypto provider: {library}')
        check_crypto_versions(item['owner'], versions)
        found.append({'path': str(library), 'openssl_markers': sorted(versions)})
    if not found:
        raise ValueError('no loaded crypto provider evidence')
    return found


def check_import_orders(path: Path, generation: Path) -> list[list[str]]:
    profile = json.loads(path.read_text())
    check_source_binding(profile, path)
    profile_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    completed = []
    for order in IMPORT_ORDERS:
        command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()),
                   '--check', 'native_inventory', '--profile', str(path),
                   '--generation', str(generation), '--native-order', ','.join(order)]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
            receipt = json.loads(result.stdout)
        except (subprocess.SubprocessError, ValueError) as exc:
            raise ValueError(f'native import-order probe failed: {order}') from exc
        if (not isinstance(receipt, dict) or receipt.get('status') != 'passed'
                or receipt.get('profile_sha256') != profile_hash
                or receipt.get('policy_id') != profile['extension_policy']['policy_id']
                or receipt.get('import_order') != list(order)):
            raise ValueError(f'native import-order probe receipt mismatch: {order}')
        completed.append(list(order))
    return completed


def profile_check(name: str, path: Path, generation: Path, *, native_order: tuple[str, ...] | None = None) -> dict:
    profile = json.loads(path.read_text())
    check_source_binding(profile, path)
    policy_id = profile['extension_policy']['policy_id']
    evidence = {}
    if name == 'effective_versions':
        check_effective_versions(profile)
    elif name == 'native_inventory':
        if native_order is None:
            evidence['import_orders'] = check_import_orders(path, generation)
        else:
            evidence['native_files'] = check_native_inventory(profile, generation, native_order)
            evidence['import_order'] = list(native_order)
    elif name == 'extension_policy':
        import duckdb
        check_compiled_policy(policy_id)
        check_extension_settings(duckdb)
    elif name == 'geometry':
        evidence['capabilities'] = check_capabilities(policy_id)
    elif name == 'tls':
        script = Path(__file__).resolve().with_name('check_native_tls.py')
        check_input_binding(profile, path, script)
        with tempfile.TemporaryDirectory() as output:
            result = subprocess.run([sys.executable, '-I', '-B', str(script), '--output', output],
                                    check=True, capture_output=True, text=True, timeout=90)
            receipt = json.loads(result.stdout)
            expected = {'libpq_valid_certificate', 'libpq_wrong_hostname_rejected',
                        'arrow_valid_certificate', 'arrow_wrong_hostname_rejected',
                        'libpq_untrusted_issuer_rejected', 'arrow_untrusted_issuer_rejected'}
            if not expected <= receipt.keys() or not all(receipt[key] is True for key in expected):
                raise ValueError('native TLS certificate checks incomplete')
            evidence['tls'] = receipt
    else:
        raise ValueError(f'unknown mandatory check: {name}')
    return {'status': 'passed', 'check': name, 'policy_id': policy_id,
            'profile_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **evidence}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-id')
    parser.add_argument('--check', choices=('effective_versions', 'native_inventory', 'extension_policy', 'tls', 'geometry'))
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--generation', type=Path)
    parser.add_argument('--native-order')
    args = parser.parse_args()
    if args.native_order and args.check != 'native_inventory':
        parser.error('--native-order is only valid with native_inventory')
    if args.check:
        if args.profile is None or args.generation is None:
            parser.error('--check requires --profile and --generation')
        order = tuple(args.native_order.split(',')) if args.native_order else None
        print(json.dumps(profile_check(args.check, args.profile, args.generation, native_order=order), sort_keys=True))
    else:
        if not args.policy_id:
            parser.error('--policy-id required')
        print(json.dumps(check_capabilities(args.policy_id), sort_keys=True))


if __name__ == '__main__':
    main()
