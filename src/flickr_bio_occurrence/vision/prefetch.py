from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Iterable

import polars as pl

from flickr_bio_occurrence.vision.image_cache import CachedImage, cache_image_from_url


CacheImage = Callable[..., CachedImage]


@dataclass(frozen=True)
class ImagePrefetchResult:
    manifest_path: Path
    requested_urls: int
    already_cached: int
    newly_cached: int
    failed: int
    errors: list[dict[str, str]]


def prefetch_image_urls(
    rows: Iterable[dict[str, object]],
    *,
    cache_root: str | Path,
    manifest_path: str | Path,
    max_workers: int = 16,
    cache_image: CacheImage = cache_image_from_url,
    fail_on_error: bool = False,
) -> ImagePrefetchResult:
    urls = _unique_image_urls(rows)
    manifest = load_image_cache_manifest(manifest_path)
    cached_by_url = {
        url: cached
        for url, cached in manifest.items()
        if cached.path.exists()
    }
    missing_urls = [url for url in urls if url not in cached_by_url]

    newly_cached: dict[str, CachedImage] = {}
    errors: list[dict[str, str]] = []
    workers = max(1, max_workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(cache_image, url, cache_root=cache_root): url
            for url in missing_urls
        }
        for future in as_completed(futures):
            url = futures[future]
            try:
                newly_cached[url] = future.result()
            except Exception as exc:  # noqa: BLE001 - errors are recorded per URL for resumable prefetching.
                errors.append({"source_url": url, "error": f"{type(exc).__name__}: {exc}"})
                if fail_on_error:
                    raise

    merged = {**cached_by_url, **newly_cached}
    write_image_cache_manifest(merged.values(), manifest_path)
    return ImagePrefetchResult(
        manifest_path=Path(manifest_path),
        requested_urls=len(urls),
        already_cached=len(cached_by_url),
        newly_cached=len(newly_cached),
        failed=len(errors),
        errors=errors,
    )


def load_image_cache_manifest(manifest_path: str | Path) -> dict[str, CachedImage]:
    path = Path(manifest_path)
    if not path.exists():
        return {}
    frame = pl.read_parquet(path)
    if frame.is_empty():
        return {}
    return {
        str(row["source_url"]): CachedImage(
            source_url=str(row["source_url"]),
            path=Path(str(row["path"])),
            image_hash=str(row["image_hash"]),
            content_type=str(row["content_type"]),
            byte_size=int(row["byte_size"]),
        )
        for row in frame.to_dicts()
    }


def write_image_cache_manifest(cached_images: Iterable[CachedImage], manifest_path: str | Path) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "source_url": cached.source_url,
            "path": str(cached.path),
            "image_hash": cached.image_hash,
            "content_type": cached.content_type,
            "byte_size": cached.byte_size,
            "manifest_updated_at": time.time(),
        }
        for cached in cached_images
    ]
    schema = {
        "source_url": pl.String,
        "path": pl.String,
        "image_hash": pl.String,
        "content_type": pl.String,
        "byte_size": pl.Int64,
        "manifest_updated_at": pl.Float64,
    }
    pl.DataFrame(rows, schema=schema).write_parquet(path)


def build_manifest_cache_image(
    *,
    manifest_path: str | Path,
    fallback_cache_image: CacheImage = cache_image_from_url,
) -> CacheImage:
    manifest = load_image_cache_manifest(manifest_path)

    def cache_image(image_url: str, *, cache_root: str | Path) -> CachedImage:
        cached = manifest.get(image_url)
        if cached is not None and cached.path.exists():
            return cached
        return fallback_cache_image(image_url, cache_root=cache_root)

    return cache_image


def _unique_image_urls(rows: Iterable[dict[str, object]]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for row in rows:
        image_url = row.get("image_url")
        if not image_url:
            continue
        url = str(image_url)
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls
