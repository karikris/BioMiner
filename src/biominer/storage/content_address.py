from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
from urllib.parse import urlparse


SHA256_PATTERN = re.compile(r"^sha256:([0-9a-f]{64})$")


def normalize_sha256(value: str) -> str:
    normalized = str(value).strip().casefold()
    match = SHA256_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError("expected SHA-256 formatted as sha256:<64 lowercase hex chars>")
    return f"sha256:{match.group(1)}"


def sha256_stream(stream) -> str:  # noqa: ANN001
    digest = sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return sha256_stream(stream)


def validate_content_addressed_uri(uri: str | Path, expected_sha256: str) -> str:
    normalized = normalize_sha256(expected_sha256)
    digest = normalized.removeprefix("sha256:")
    final_name = Path(urlparse(str(uri)).path).name.casefold()
    if digest not in final_name:
        raise ValueError(
            "content-addressed transfer URI final key must contain the full SHA-256"
        )
    return normalized


__all__ = [
    "normalize_sha256",
    "sha256_file",
    "sha256_stream",
    "validate_content_addressed_uri",
]
