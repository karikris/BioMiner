from __future__ import annotations

import hashlib

import httpx
import pytest

from flickr_bio_occurrence.vision.image_cache import cache_image_from_url


def test_cache_image_from_url_writes_content_addressed_file(tmp_path) -> None:
    image_bytes = b"\xff\xd8fake-jpeg\xff\xd9"

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://live.staticflickr.com/example.jpg"
        return httpx.Response(200, content=image_bytes, headers={"Content-Type": "image/jpeg"})

    cached = cache_image_from_url(
        "https://live.staticflickr.com/example.jpg",
        cache_root=tmp_path,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    expected_hash = "sha256:" + hashlib.sha256(image_bytes).hexdigest()
    assert cached.image_hash == expected_hash
    assert cached.path.exists()
    assert cached.path.read_bytes() == image_bytes
    assert cached.path.suffix == ".jpg"


def test_cache_image_from_url_rejects_non_http_urls(tmp_path) -> None:
    with pytest.raises(ValueError, match="Only http and https image URLs"):
        cache_image_from_url("file:///tmp/image.jpg", cache_root=tmp_path)


def test_cache_image_from_url_rejects_non_image_response(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html></html>", headers={"Content-Type": "text/html"})

    with pytest.raises(ValueError, match="does not point to an image"):
        cache_image_from_url(
            "https://live.staticflickr.com/example",
            cache_root=tmp_path,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )


def test_cache_image_from_url_retries_transient_request_errors(tmp_path) -> None:
    image_bytes = b"\xff\xd8retry-jpeg\xff\xd9"
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("temporary dns failure", request=request)
        return httpx.Response(200, content=image_bytes, headers={"Content-Type": "image/jpeg"})

    cached = cache_image_from_url(
        "https://live.staticflickr.com/example.jpg",
        cache_root=tmp_path,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=2,
        retry_sleep_seconds=0,
    )

    assert attempts == 2
    assert cached.path.exists()
