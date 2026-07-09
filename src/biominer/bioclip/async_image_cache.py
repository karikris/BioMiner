"""Asynchronous image download for BioCLIP register fills.

Downloads multiple images concurrently using ``httpx.AsyncClient`` with a
configurable concurrency limit and exponential-backoff retries.  The public
API mirrors :func:`~biominer.bioclip.image_cache.cache_image_from_url` but
operates on batches, returning a list of either :class:`CachedImage` results
or captured exceptions so that the caller can handle partial failures
without aborting the entire register fill.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from urllib.parse import urlparse

import httpx

from biominer.bioclip.image_cache import CachedImage, _extension_for_content_type


async def _download_one(
    client: httpx.AsyncClient,
    image_url: str,
    *,
    cache_root: Path,
    max_retries: int,
    retry_backoff_seconds: float,
    semaphore: asyncio.Semaphore,
) -> CachedImage:
    """Download a single image with retries, respecting the semaphore."""
    parsed = urlparse(image_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http and https image URLs can be cached")

    async with semaphore:
        response = await _async_get_with_retries(
            client,
            image_url,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )

    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if not content_type.startswith("image/"):
        raise ValueError(f"{image_url} does not point to an image response")

    content = response.content
    digest = hashlib.sha256(content).hexdigest()
    image_hash = f"sha256:{digest}"
    target_dir = cache_root / digest[:2] / digest[2:4]
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


async def _async_get_with_retries(
    client: httpx.AsyncClient,
    image_url: str,
    *,
    max_retries: int,
    retry_backoff_seconds: float,
) -> httpx.Response:
    attempts = max(1, max_retries)
    last_error: httpx.HTTPError | None = None
    for attempt in range(attempts):
        try:
            return await client.get(image_url)
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            await asyncio.sleep(retry_backoff_seconds * (2 ** attempt))
    assert last_error is not None
    raise last_error


async def _download_batch(
    urls: list[str],
    *,
    cache_root: Path,
    concurrency: int,
    max_retries: int,
    retry_backoff_seconds: float,
    timeout: float,
) -> list[CachedImage | Exception]:
    """Download *urls* concurrently, returning results positionally.

    Each slot contains either a :class:`CachedImage` on success or the
    captured :class:`Exception` on failure.
    """
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            asyncio.create_task(
                _download_one(
                    client,
                    url,
                    cache_root=cache_root,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                    semaphore=semaphore,
                )
            )
            for url in urls
        ]
        raw = await asyncio.gather(*tasks, return_exceptions=True)
    return list(raw)


def cache_images_async(
    urls: list[str],
    *,
    cache_root: str | Path = "data/cache/images",
    concurrency: int = 8,
    max_retries: int = 5,
    retry_backoff_seconds: float = 1.0,
    timeout: float = 30.0,
) -> list[CachedImage | Exception]:
    """Download *urls* concurrently from a synchronous context.

    Spins up an ``asyncio`` event loop (or uses the running one when called
    inside an existing loop) and returns one :class:`CachedImage` or captured
    ``Exception`` per URL, in the same order as *urls*.

    Parameters
    ----------
    urls:
        HTTP(S) image URLs to download.
    cache_root:
        Local directory for content-addressed image storage.
    concurrency:
        Maximum parallel downloads (default 8).
    max_retries:
        Per-URL retry budget (default 5).
    retry_backoff_seconds:
        Base delay between retries — doubles each attempt.
    timeout:
        ``httpx`` per-request timeout in seconds.
    """
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # Already inside an event loop (e.g. Jupyter) — run in a new thread.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run,
                _download_batch(
                    urls,
                    cache_root=root,
                    concurrency=concurrency,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                    timeout=timeout,
                ),
            ).result()

    return asyncio.run(
        _download_batch(
            urls,
            cache_root=root,
            concurrency=concurrency,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            timeout=timeout,
        )
    )
