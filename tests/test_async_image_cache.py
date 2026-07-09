from __future__ import annotations

import hashlib

import httpx

from biominer.bioclip.async_image_cache import cache_images_async
from biominer.bioclip.image_cache import CachedImage


def test_cache_images_async_downloads_batch(tmp_path) -> None:
    """Async batch downloader writes content-addressed files for each URL."""
    image_bytes_a = b"\xff\xd8batch-a\xff\xd9"
    image_bytes_b = b"\xff\xd8batch-b\xff\xd9"

    transport = httpx.MockTransport(_mock_handler({
        "https://live.staticflickr.com/a.jpg": (image_bytes_a, "image/jpeg"),
        "https://live.staticflickr.com/b.jpg": (image_bytes_b, "image/jpeg"),
    }))
    # Monkeypatch httpx.AsyncClient to use MockTransport
    import biominer.bioclip.async_image_cache as mod
    original = mod._download_batch

    import asyncio

    async def patched_batch(urls, *, cache_root, concurrency, max_retries, retry_backoff_seconds, timeout):
        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(transport=transport, timeout=timeout) as client:
            tasks = [
                asyncio.create_task(
                    mod._download_one(
                        client, url,
                        cache_root=cache_root,
                        max_retries=max_retries,
                        retry_backoff_seconds=retry_backoff_seconds,
                        semaphore=sem,
                    )
                )
                for url in urls
            ]
            raw = await asyncio.gather(*tasks, return_exceptions=True)
        return list(raw)

    mod._download_batch = patched_batch
    try:
        results = cache_images_async(
            ["https://live.staticflickr.com/a.jpg", "https://live.staticflickr.com/b.jpg"],
            cache_root=tmp_path,
            concurrency=4,
            max_retries=1,
            retry_backoff_seconds=0,
        )
    finally:
        mod._download_batch = original

    assert len(results) == 2
    for result in results:
        assert isinstance(result, CachedImage)
        assert result.path.exists()
    assert results[0].image_hash == "sha256:" + hashlib.sha256(image_bytes_a).hexdigest()
    assert results[1].image_hash == "sha256:" + hashlib.sha256(image_bytes_b).hexdigest()


def test_cache_images_async_captures_failures(tmp_path) -> None:
    """Failed downloads are returned as Exception instances, not raised."""
    transport = httpx.MockTransport(_mock_handler({
        "https://live.staticflickr.com/good.jpg": (b"\xff\xd8ok\xff\xd9", "image/jpeg"),
        # bad.jpg is missing → will get a 404
    }))
    import biominer.bioclip.async_image_cache as mod
    import asyncio

    async def patched_batch(urls, *, cache_root, concurrency, max_retries, retry_backoff_seconds, timeout):
        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(transport=transport, timeout=timeout) as client:
            tasks = [
                asyncio.create_task(
                    mod._download_one(
                        client, url,
                        cache_root=cache_root,
                        max_retries=max_retries,
                        retry_backoff_seconds=retry_backoff_seconds,
                        semaphore=sem,
                    )
                )
                for url in urls
            ]
            raw = await asyncio.gather(*tasks, return_exceptions=True)
        return list(raw)

    original = mod._download_batch
    mod._download_batch = patched_batch
    try:
        results = cache_images_async(
            ["https://live.staticflickr.com/good.jpg", "https://live.staticflickr.com/bad.jpg"],
            cache_root=tmp_path,
            max_retries=1,
            retry_backoff_seconds=0,
        )
    finally:
        mod._download_batch = original

    assert isinstance(results[0], CachedImage)
    assert isinstance(results[1], Exception)


def test_cache_images_async_empty_list(tmp_path) -> None:
    """Empty URL list returns empty results."""
    results = cache_images_async([], cache_root=tmp_path)
    assert results == []


def _mock_handler(url_map: dict[str, tuple[bytes, str]]):
    """Build an httpx transport handler for async tests."""
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in url_map:
            content, content_type = url_map[url]
            return httpx.Response(200, content=content, headers={"Content-Type": content_type})
        return httpx.Response(404, content=b"not found", headers={"Content-Type": "text/plain"})
    return handler
