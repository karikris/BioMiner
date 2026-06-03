from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from urllib.parse import urlparse

import httpx


@dataclass(frozen=True)
class CachedImage:
    source_url: str
    path: Path
    image_hash: str
    content_type: str
    byte_size: int


def cache_image_from_url(
    image_url: str,
    *,
    cache_root: str | Path = "data/cache/images",
    http_client: httpx.Client | None = None,
) -> CachedImage:
    parsed = urlparse(image_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http and https image URLs can be cached")

    client = http_client or httpx.Client(timeout=30)
    response = client.get(image_url)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if not content_type.startswith("image/"):
        raise ValueError(f"{image_url} does not point to an image response")

    content = response.content
    digest = hashlib.sha256(content).hexdigest()
    image_hash = f"sha256:{digest}"
    target_dir = Path(cache_root) / digest[:2] / digest[2:4]
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{digest}{_extension_for_content_type(content_type, parsed.path)}"
    if not target.exists():
        target.write_bytes(content)

    return CachedImage(
        source_url=image_url,
        path=target,
        image_hash=image_hash,
        content_type=content_type,
        byte_size=len(content),
    )


def _extension_for_content_type(content_type: str, path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get(content_type, ".img")
