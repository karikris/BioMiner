#!/usr/bin/env python3
"""Bounded loopback TLS checks; these do not test hosted S3 or database CRUD."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import socket
import ssl
import struct
import subprocess
import threading


@contextmanager
def server(cert: Path, key: Path, postgres: bool):
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(4)
    listener.settimeout(0.2)
    port = listener.getsockname()[1]
    stop = threading.Event()
    handshake = threading.Event()
    errors = []

    def serve():
        while not stop.is_set():
            try:
                raw, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                with raw:
                    raw.settimeout(3)
                    if postgres:
                        request = bytearray()
                        while len(request) < 8:
                            chunk = raw.recv(8-len(request))
                            if not chunk:
                                raise ValueError('incomplete PostgreSQL SSLRequest')
                            request.extend(chunk)
                        if request != struct.pack('!II', 8, 80877103):
                            raise ValueError('unexpected PostgreSQL SSLRequest')
                        raw.sendall(b'S')
                    with context.wrap_socket(raw, server_side=True) as secure:
                        handshake.set()
                        request = secure.recv(65536)
                        if not postgres:
                            while b'\r\n\r\n' not in request:
                                chunk = secure.recv(65536)
                                if not chunk:
                                    break
                                request += chunk
                            secure.sendall(b'HTTP/1.1 404 Not Found\r\n'
                                           b'Content-Length: 0\r\nConnection: close\r\n\r\n')
            except (ssl.SSLError, ConnectionError):
                # Invalid-certificate clients intentionally abort TLS.
                pass
            except Exception as exc:
                errors.append(str(exc))

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield port, handshake
    finally:
        stop.set()
        listener.close()
        thread.join(timeout=4)
        if thread.is_alive() or errors:
            raise ValueError(f'TLS fixture failed: {errors}')


def check_tls(directory: Path) -> dict:
    import psycopg
    import pyarrow.fs as fs

    directory.mkdir(parents=True, exist_ok=True)
    cert, key = directory / 'tls-cert.pem', directory / 'tls-key.pem'
    subprocess.run(['/usr/bin/openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                    '-keyout', str(key), '-out', str(cert), '-days', '1',
                    '-subj', '/CN=localhost', '-addext', 'subjectAltName=DNS:localhost'],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    key.chmod(0o600)
    other_cert, other_key = directory / 'other-ca.pem', directory / 'other-ca-key.pem'
    subprocess.run(['/usr/bin/openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                    '-keyout', str(other_key), '-out', str(other_cert), '-days', '1',
                    '-subj', '/CN=Untrusted fixture issuer'], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    other_key.chmod(0o600)
    result = {}
    for case in ('valid', 'wrong_hostname', 'untrusted_issuer'):
        valid = case == 'valid'
        host = 'wrong.example' if case == 'wrong_hostname' else 'localhost'
        trusted = other_cert if case == 'untrusted_issuer' else cert
        with server(cert, key, True) as (port, handshake):
            try:
                psycopg.connect(host=host, hostaddr='127.0.0.1', port=port,
                                user='tls-fixture', dbname='tls-fixture',
                                sslmode='verify-full', sslrootcert=str(trusted), connect_timeout=3)
            except psycopg.OperationalError as exc:
                error = str(exc).lower()
            else:
                raise ValueError('TLS fixture unexpectedly acted as a database')
            if valid:
                if not handshake.wait(1):
                    raise ValueError(f'libpq valid certificate failed: {error}')
                result['libpq_valid_certificate'] = True
            else:
                if 'certificate' not in error or ('match' not in error and 'verify' not in error):
                    raise ValueError(f'libpq did not reject hostname: {error}')
                result[f'libpq_{case}_rejected'] = True
        with server(cert, key, False) as (port, handshake):
            endpoint = '127.0.0.1' if case == 'wrong_hostname' else 'localhost'
            filesystem = fs.S3FileSystem(anonymous=True, scheme='https', region='us-east-1',
                                        endpoint_override=f'{endpoint}:{port}',
                                        force_virtual_addressing=False,
                                        tls_ca_file_path=str(trusted), request_timeout=2,
                                        connect_timeout=2,
                                        retry_strategy=fs.AwsStandardS3RetryStrategy(max_attempts=1))
            try:
                info = filesystem.get_file_info('fixture/object')
            except OSError as exc:
                if valid:
                    raise ValueError(f'Arrow valid certificate failed: {exc}') from exc
                if 'ssl' not in str(exc).lower() and 'certificate' not in str(exc).lower():
                    raise ValueError(f'Arrow failure was not certificate verification: {exc}') from exc
                result[f'arrow_{case}_rejected'] = True
            else:
                if not valid:
                    raise ValueError('Arrow accepted wrong-host certificate')
                if not handshake.wait(1) or info.type != fs.FileType.NotFound:
                    raise ValueError('Arrow TLS/HTTP fixture returned unexpected result')
                result['arrow_valid_certificate'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(check_tls(args.output), sort_keys=True))


if __name__ == '__main__':
    main()
