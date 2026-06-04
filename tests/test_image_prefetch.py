from __future__ import annotations

from pathlib import Path

from flickr_bio_occurrence.vision.image_cache import CachedImage
from flickr_bio_occurrence.vision.prefetch import (
    build_manifest_cache_image,
    load_image_cache_manifest,
    prefetch_image_urls,
)


def test_prefetch_image_urls_writes_manifest_and_skips_existing_urls(tmp_path) -> None:
    calls: list[str] = []

    def fake_cache(image_url: str, *, cache_root: str | Path) -> CachedImage:
        calls.append(image_url)
        target = Path(cache_root) / f"{len(calls)}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"jpeg")
        return CachedImage(
            source_url=image_url,
            path=target,
            image_hash=f"sha256:{len(calls)}",
            content_type="image/jpeg",
            byte_size=4,
        )

    rows = [
        {"image_url": "https://live.staticflickr.com/1.jpg"},
        {"image_url": "https://live.staticflickr.com/1.jpg"},
        {"image_url": "https://live.staticflickr.com/2.jpg"},
        {"image_url": None},
    ]
    manifest_path = tmp_path / "cache" / "image_url_cache.parquet"

    first = prefetch_image_urls(
        rows,
        cache_root=tmp_path / "cache" / "images",
        manifest_path=manifest_path,
        max_workers=1,
        cache_image=fake_cache,
    )
    second = prefetch_image_urls(
        rows,
        cache_root=tmp_path / "cache" / "images",
        manifest_path=manifest_path,
        max_workers=1,
        cache_image=fake_cache,
    )

    assert first.requested_urls == 2
    assert first.newly_cached == 2
    assert first.failed == 0
    assert second.already_cached == 2
    assert second.newly_cached == 0
    assert sorted(calls) == ["https://live.staticflickr.com/1.jpg", "https://live.staticflickr.com/2.jpg"]
    assert set(load_image_cache_manifest(manifest_path)) == {
        "https://live.staticflickr.com/1.jpg",
        "https://live.staticflickr.com/2.jpg",
    }


def test_manifest_cache_image_uses_prefetched_file_without_fallback_download(tmp_path) -> None:
    cached_path = tmp_path / "cache" / "image.jpg"
    cached_path.parent.mkdir(parents=True)
    cached_path.write_bytes(b"jpeg")
    manifest_path = tmp_path / "image_url_cache.parquet"
    prefetch_image_urls(
        [{"image_url": "https://live.staticflickr.com/prefetched.jpg"}],
        cache_root=tmp_path / "unused",
        manifest_path=manifest_path,
        cache_image=lambda image_url, *, cache_root: CachedImage(
            source_url=image_url,
            path=cached_path,
            image_hash="sha256:prefetched",
            content_type="image/jpeg",
            byte_size=4,
        ),
    )

    def forbidden_fallback(image_url: str, *, cache_root: str | Path) -> CachedImage:
        raise AssertionError("fallback download should not be called for prefetched images")

    cache_image = build_manifest_cache_image(
        manifest_path=manifest_path,
        fallback_cache_image=forbidden_fallback,
    )

    cached = cache_image("https://live.staticflickr.com/prefetched.jpg", cache_root=tmp_path / "cache")

    assert cached.path == cached_path
    assert cached.image_hash == "sha256:prefetched"


def test_manifest_cache_image_falls_back_for_missing_urls(tmp_path) -> None:
    fallback_path = tmp_path / "cache" / "fallback.jpg"

    def fake_fallback(image_url: str, *, cache_root: str | Path) -> CachedImage:
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        fallback_path.write_bytes(b"jpeg")
        return CachedImage(
            source_url=image_url,
            path=fallback_path,
            image_hash="sha256:fallback",
            content_type="image/jpeg",
            byte_size=4,
        )

    cache_image = build_manifest_cache_image(
        manifest_path=tmp_path / "missing.parquet",
        fallback_cache_image=fake_fallback,
    )

    cached = cache_image(
        "https://live.staticflickr.com/fallback.jpg",
        cache_root=tmp_path / "cache",
    )

    assert cached.path.exists()
