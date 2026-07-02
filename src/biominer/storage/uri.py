from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse


def is_s3_uri(uri: str | Path) -> bool:
    return str(uri).startswith("s3://")


def is_cloud_uri(uri: str | Path) -> bool:
    return is_s3_uri(uri)


def normalize_local_uri(uri: str | Path) -> Path:
    value = str(uri)
    parsed = urlparse(value)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError(f"unsupported file URI host: {parsed.netloc}")
        return Path(unquote(parsed.path))
    if parsed.scheme and parsed.scheme != "file":
        raise ValueError(f"not a local URI: {value}")
    return Path(value)


def join_uri(prefix: str | Path, *parts: str | Path) -> str:
    base = str(prefix).rstrip("/")
    clean_parts = [str(part).strip("/") for part in parts if str(part).strip("/")]
    if not clean_parts:
        return base
    suffix = "/".join(clean_parts)
    return f"{base}/{suffix}" if base else suffix
