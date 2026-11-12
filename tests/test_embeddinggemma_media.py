from __future__ import annotations

import hashlib
import importlib
from collections import Counter
from io import BytesIO
from pathlib import Path

import httpx
import polars as pl
import pytest
from PIL import Image


def _cache(items, cache_root, **kwargs):
    try:
        module = importlib.import_module("biominer.embeddinggemma.media")
    except ModuleNotFoundError:
        pytest.fail("EmbeddingGemma media caching is not implemented")
    return module.cache_items(items, cache_root, **kwargs)


def _png(color="red", *, orientation=None):
    data = BytesIO()
    image = Image.new("RGB", (8, 5), color)
    if orientation:
        exif = image.getexif()
        exif[274] = orientation
        image.save(data, format="JPEG", exif=exif)
    else:
        image.save(data, format="PNG")
    return data.getvalue()


def _items(ids=("a",), **extra):
    return pl.from_dicts(
        [
            {
                "item_id": item_id,
                "image_url": f"https://images.example/{item_id}.jpg",
                "fallback_url": None,
                **extra,
            }
            for item_id in ids
        ]
    )


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_cache_validates_images_and_deduplicates_bytes_by_content(tmp_path):
    image = _png()
    with _client(lambda request: httpx.Response(200, content=image)) as client:
        result = _cache(_items(("a", "b")), tmp_path, http_client=client, workers=2)

    assert result["status"].to_list() == ["ok", "ok"]
    assert (
        result["image_hash"].to_list()
        == ["sha256:" + hashlib.sha256(image).hexdigest()] * 2
    )
    assert result["local_path"].n_unique() == 1
    assert Path(result["local_path"][0]).read_bytes() == image
    assert result["width"].to_list() == [8, 8]
    assert result["height"].to_list() == [5, 5]
    assert pl.read_parquet(tmp_path / "index.parquet").height == 2


def test_cache_records_orientation_corrected_dimensions_without_changing_source_bytes(
    tmp_path,
):
    image = _png(orientation=6)
    with _client(lambda request: httpx.Response(200, content=image)) as client:
        result = _cache(_items(), tmp_path, http_client=client)

    assert (result["width"][0], result["height"][0]) == (5, 8)
    assert Path(result["local_path"][0]).read_bytes() == image


def test_resume_verifies_cached_hash_and_avoids_network_requests(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, content=_png())

    with _client(handler) as client:
        first = _cache(_items(), tmp_path, http_client=client)
        before = Path(first["local_path"][0]).stat().st_mtime_ns
        second = _cache(_items(), tmp_path, http_client=client)

    assert calls == ["/a.jpg"]
    assert first.to_dicts() == second.to_dicts()
    assert Path(first["local_path"][0]).stat().st_mtime_ns == before


def test_resume_detects_corrupt_cached_bytes_and_preserves_them(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, content=_png())

    with _client(handler) as client:
        first = _cache(_items(), tmp_path, http_client=client)
        path = Path(first["local_path"][0])
        path.write_bytes(b"corrupt-media")
        second = _cache(_items(), tmp_path, http_client=client)

    assert second["status"].to_list() == ["cache_corrupt"]
    assert second["error_type"].to_list() == ["content_hash_mismatch"]
    assert path.read_bytes() == b"corrupt-media"
    assert calls == ["/a.jpg"]


def test_cache_tries_fallback_url_after_primary_unavailable(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return (
            httpx.Response(404)
            if request.url.path == "/a.jpg"
            else httpx.Response(200, content=_png())
        )

    with _client(handler) as client:
        result = _cache(
            _items(fallback_url="https://fallback.example/a.png"),
            tmp_path,
            http_client=client,
        )

    assert result["status"].to_list() == ["ok"]
    assert calls == ["/a.jpg", "/a.png"]


def test_rate_limited_download_retries_and_honors_retry_after(tmp_path, monkeypatch):
    calls = []
    sleeps = []

    def handler(request):
        calls.append(request.url.path)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0.25"})
        return httpx.Response(200, content=_png())

    monkeypatch.setattr("time.sleep", sleeps.append)
    with _client(handler) as client:
        result = _cache(_items(), tmp_path, http_client=client, retries=1)

    assert result["status"].to_list() == ["ok"]
    assert len(calls) == 2
    assert sleeps == [0.25]
    assert result["attempts"].to_list() == [2]


def test_rate_limit_longer_than_retry_budget_stops_without_early_retry(
    tmp_path, monkeypatch
):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(429, headers={"Retry-After": "3600"})

    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    with _client(handler) as client:
        result = _cache(_items(), tmp_path, http_client=client)

    assert result["status"].to_list() == ["rate_limited"]
    assert result["error_type"].to_list() == ["retry_after_exceeds_budget"]
    assert len(calls) == 1
    assert sleeps == []


def test_decode_error_never_persists_unvalidated_bytes_or_url_in_index(tmp_path):
    items = _items(
        image_url="https://images.example/private-token.jpg?secret=do-not-leak"
    )
    with _client(
        lambda request: httpx.Response(200, content=b"not an image")
    ) as client:
        result = _cache(items, tmp_path, http_client=client)

    assert result["status"].to_list() == ["decode_error"]
    assert result["local_path"].to_list() == [None]
    assert result["image_hash"].to_list() == [None]
    assert not list((tmp_path / "objects").rglob("*.image"))
    assert "private-token" not in str(result.to_dicts())
    assert "do-not-leak" not in str(
        pl.read_parquet(tmp_path / "index.parquet").to_dicts()
    )


class _Chunks(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.read_count = 0

    def __iter__(self):
        for chunk in self.chunks:
            self.read_count += 1
            yield chunk


def test_streaming_byte_limit_stops_reading_before_consuming_whole_response(tmp_path):
    stream = _Chunks([b"x" * 8, b"y" * 8, b"z" * 8])
    with _client(lambda request: httpx.Response(200, stream=stream)) as client:
        result = _cache(_items(), tmp_path, http_client=client, max_image_bytes=10)

    assert result["status"].to_list() == ["too_large"]
    assert stream.read_count == 2
    assert not list((tmp_path / "objects").rglob("*.image"))


def test_declared_oversize_response_is_rejected_without_reading_body(tmp_path):
    stream = _Chunks([b"x" * 20])
    with _client(
        lambda request: httpx.Response(
            200, stream=stream, headers={"Content-Length": "20"}
        )
    ) as client:
        result = _cache(_items(), tmp_path, http_client=client, max_image_bytes=10)

    assert result["status"].to_list() == ["too_large"]
    assert stream.read_count == 0


def test_failed_item_is_retried_on_resume_without_disturbing_good_bytes(tmp_path):
    counts = Counter()

    def handler(request):
        counts[request.url.path] += 1
        if request.url.path == "/b.jpg" and counts[request.url.path] == 1:
            return httpx.Response(404)
        return httpx.Response(200, content=_png())

    with _client(handler) as client:
        first = _cache(_items(("a", "b")), tmp_path, http_client=client, workers=1)
        path = Path(first.filter(pl.col("item_id") == "a")["local_path"][0])
        before = path.stat().st_mtime_ns
        second = _cache(_items(("a", "b")), tmp_path, http_client=client, workers=1)

    assert second["status"].to_list() == ["ok", "ok"]
    assert counts == {"/a.jpg": 1, "/b.jpg": 2}
    assert path.stat().st_mtime_ns == before


def test_limit_bounds_requested_work_and_keeps_previous_index_entries(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, content=_png())

    with _client(handler) as client:
        result = _cache(_items(("a", "b", "c")), tmp_path, http_client=client, limit=1)
        second = _cache(_items(("b", "c")), tmp_path, http_client=client, limit=1)

    assert result["item_id"].to_list() == ["a"]
    assert second["item_id"].to_list() == ["b"]
    assert calls == ["/a.jpg", "/b.jpg"]
    assert set(pl.read_parquet(tmp_path / "index.parquet")["item_id"]) == {"a", "b"}


def test_same_item_identity_with_changed_source_fingerprint_fails_closed(tmp_path):
    with _client(lambda request: httpx.Response(200, content=_png())) as client:
        _cache(_items(), tmp_path, http_client=client)
        with pytest.raises(ValueError, match="source fingerprint"):
            _cache(
                _items(image_url="https://images.example/new.jpg"),
                tmp_path,
                http_client=client,
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"workers": 0},
        {"retries": -1},
        {"max_image_bytes": 0},
        {"timeout": 0},
        {"limit": -1},
    ],
)
def test_invalid_download_bounds_fail_before_network(tmp_path, kwargs):
    def unexpected(request):
        pytest.fail("invalid bounds must fail before network use")

    with _client(unexpected) as client, pytest.raises(ValueError):
        _cache(_items(), tmp_path, http_client=client, **kwargs)


def test_oversized_primary_does_not_block_valid_fallback(tmp_path):
    image = _png()

    def handler(request):
        if request.url.path == "/a.jpg":
            return httpx.Response(
                200, headers={"Content-Length": "10000"}, content=b"x"
            )
        return httpx.Response(200, content=image)

    with _client(handler) as client:
        result = _cache(
            _items(fallback_url="https://fallback.example/a.png"),
            tmp_path,
            http_client=client,
            max_image_bytes=1000,
        )

    assert result["status"].to_list() == ["ok"]


def test_parallel_cache_writer_is_rejected(tmp_path):
    from biominer.embeddinggemma.common import run_lock
    from biominer.embeddinggemma.media import cache_items

    empty = pl.DataFrame(schema={"item_id": pl.String, "image_url": pl.String})
    with run_lock(tmp_path), pytest.raises(RuntimeError, match="worker"):
        cache_items(empty, tmp_path)


def test_read_only_cache_rejects_changed_source(tmp_path):
    import io

    from PIL import Image

    from biominer.embeddinggemma.media import read_cached_items

    buffer = io.BytesIO()
    Image.new("RGB", (20, 20)).save(buffer, format="PNG")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=buffer.getvalue())
        )
    )
    items = pl.DataFrame(
        {"item_id": ["one"], "image_url": ["https://example.test/old.png"]}
    )
    _cache(items, tmp_path, http_client=client)
    changed = items.with_columns(
        pl.lit("https://example.test/new.png").alias("image_url")
    )
    with pytest.raises(ValueError, match="source fingerprint"):
        read_cached_items(changed, tmp_path)
    client.close()


def test_read_only_cache_excludes_corrupt_bytes_without_network(tmp_path):
    import io

    from PIL import Image

    from biominer.embeddinggemma.media import read_cached_items

    buffer = io.BytesIO()
    Image.new("RGB", (20, 20)).save(buffer, format="PNG")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=buffer.getvalue())
        )
    )
    items = pl.DataFrame(
        {"item_id": ["one"], "image_url": ["https://example.test/image.png"]}
    )
    result = _cache(items, tmp_path, http_client=client)
    Path(result["local_path"].item()).write_bytes(b"changed")
    assert read_cached_items(items, tmp_path)["status"].item() == "cache_corrupt"
    client.close()
