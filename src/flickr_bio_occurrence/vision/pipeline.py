from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

from flickr_bio_occurrence.vision.image_cache import CachedImage, cache_image_from_url
from flickr_bio_occurrence.vision.temp_image_store import cleanup_cached_image


class ImageClassifier(Protocol):
    def classify_image(self, **kwargs: object) -> dict[str, object]:
        ...


CacheImage = Callable[..., CachedImage]
RowClassifier = Callable[[dict[str, Any]], dict[str, object]]


def classify_bronze_photo_row(
    row: dict[str, Any],
    *,
    classifier: ImageClassifier,
    cache_root: str | Path = "data/cache/images",
    cache_image: CacheImage = cache_image_from_url,
    delete_after_success: bool = True,
    keep_failed_images: bool = False,
) -> dict[str, object]:
    image_url = row.get("image_url")
    if not image_url:
        raise ValueError(f"Bronze photo row {row.get('flickr_photo_id', '<unknown>')} does not include an image_url")

    cached = cache_image(str(image_url), cache_root=cache_root)
    try:
        prediction = classifier.classify_image(
            flickr_photo_id=str(row["flickr_photo_id"]),
            image_path=cached.path,
            image_hash=cached.image_hash,
            image_url_used=cached.source_url,
            resolved_scientific_name=str(row.get("species_query") or ""),
            text_evidence_present=_text_evidence_present(row),
        )
    except Exception:
        if not keep_failed_images:
            cleanup_cached_image(cached, cache_root=cache_root, delete_after_success=True)
        raise
    cleanup_cached_image(cached, cache_root=cache_root, delete_after_success=delete_after_success)
    return prediction


def _text_evidence_present(row: dict[str, Any]) -> bool:
    return bool(row.get("raw_title") or row.get("raw_description") or row.get("raw_tags") or row.get("machine_tags"))
