from __future__ import annotations

from pathlib import Path

import pytest

from flickr_bio_occurrence.vision.image_cache import CachedImage
from flickr_bio_occurrence.vision.pipeline import classify_bronze_photo_row


def test_classify_bronze_photo_row_caches_image_and_runs_classifier(tmp_path) -> None:
    calls: dict[str, object] = {}
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"jpeg")

    def fake_cache(image_url: str, *, cache_root: str | Path) -> CachedImage:
        calls["image_url"] = image_url
        calls["cache_root"] = cache_root
        return CachedImage(
            source_url=image_url,
            path=image_path,
            image_hash="sha256:image",
            content_type="image/jpeg",
            byte_size=10,
        )

    class FakeClassifier:
        def classify_image(self, **kwargs: object) -> dict[str, object]:
            calls["classifier_kwargs"] = kwargs
            return {
                "flickr_photo_id": kwargs["flickr_photo_id"],
                "image_hash": kwargs["image_hash"],
                "image_url_used": kwargs["image_url_used"],
                "top1_label": "a photo of Papilio demoleus",
                "top1_score": 0.9,
                "species_agreement_status": "exact_species_agreement",
                "vision_review_required": False,
            }

    record = classify_bronze_photo_row(
        {
            "flickr_photo_id": "123",
            "image_url": "https://live.staticflickr.com/example.jpg",
            "species_query": "Papilio demoleus",
            "raw_title": "Papilio demoleus",
        },
        classifier=FakeClassifier(),
        cache_root=tmp_path,
        cache_image=fake_cache,
    )

    assert record["flickr_photo_id"] == "123"
    assert record["image_hash"] == "sha256:image"
    assert calls["image_url"] == "https://live.staticflickr.com/example.jpg"
    assert calls["classifier_kwargs"]["resolved_scientific_name"] == "Papilio demoleus"
    assert calls["classifier_kwargs"]["text_evidence_present"] is True
    assert not image_path.exists()


def test_classify_bronze_photo_row_can_keep_permanent_cache_file(tmp_path) -> None:
    image_path = tmp_path / "cache" / "image.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(b"jpeg")

    def fake_cache(image_url: str, *, cache_root: str | Path) -> CachedImage:
        return CachedImage(
            source_url=image_url,
            path=image_path,
            image_hash="sha256:image",
            content_type="image/jpeg",
            byte_size=10,
        )

    class FakeClassifier:
        def classify_image(self, **kwargs: object) -> dict[str, object]:
            return {"flickr_photo_id": kwargs["flickr_photo_id"]}

    classify_bronze_photo_row(
        {"flickr_photo_id": "123", "image_url": "https://live.staticflickr.com/example.jpg"},
        classifier=FakeClassifier(),
        cache_root=image_path.parent,
        cache_image=fake_cache,
        delete_after_success=False,
    )

    assert image_path.exists()


def test_classify_bronze_photo_row_retains_failed_image_when_configured(tmp_path) -> None:
    image_path = tmp_path / "cache" / "failed.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(b"jpeg")

    def fake_cache(image_url: str, *, cache_root: str | Path) -> CachedImage:
        return CachedImage(
            source_url=image_url,
            path=image_path,
            image_hash="sha256:image",
            content_type="image/jpeg",
            byte_size=10,
        )

    class FailingClassifier:
        def classify_image(self, **kwargs: object) -> dict[str, object]:
            raise RuntimeError("classification failed")

    with pytest.raises(RuntimeError, match="classification failed"):
        classify_bronze_photo_row(
            {"flickr_photo_id": "123", "image_url": "https://live.staticflickr.com/example.jpg"},
            classifier=FailingClassifier(),
            cache_root=image_path.parent,
            cache_image=fake_cache,
            keep_failed_images=True,
        )

    assert image_path.exists()


def test_classify_bronze_photo_row_deletes_failed_image_by_default(tmp_path) -> None:
    image_path = tmp_path / "cache" / "failed.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(b"jpeg")

    def fake_cache(image_url: str, *, cache_root: str | Path) -> CachedImage:
        return CachedImage(
            source_url=image_url,
            path=image_path,
            image_hash="sha256:image",
            content_type="image/jpeg",
            byte_size=10,
        )

    class FailingClassifier:
        def classify_image(self, **kwargs: object) -> dict[str, object]:
            raise RuntimeError("classification failed")

    with pytest.raises(RuntimeError, match="classification failed"):
        classify_bronze_photo_row(
            {"flickr_photo_id": "123", "image_url": "https://live.staticflickr.com/example.jpg"},
            classifier=FailingClassifier(),
            cache_root=image_path.parent,
            cache_image=fake_cache,
        )

    assert not image_path.exists()


def test_classify_bronze_photo_row_does_not_delete_huggingface_cache(tmp_path) -> None:
    image_path = tmp_path / "cache" / "huggingface" / "image.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"jpeg")

    def fake_cache(image_url: str, *, cache_root: str | Path) -> CachedImage:
        return CachedImage(
            source_url=image_url,
            path=image_path,
            image_hash="sha256:image",
            content_type="image/jpeg",
            byte_size=10,
        )

    class FakeClassifier:
        def classify_image(self, **kwargs: object) -> dict[str, object]:
            return {"flickr_photo_id": kwargs["flickr_photo_id"]}

    classify_bronze_photo_row(
        {"flickr_photo_id": "123", "image_url": "https://live.staticflickr.com/example.jpg"},
        classifier=FakeClassifier(),
        cache_root=image_path.parent,
        cache_image=fake_cache,
    )

    assert image_path.exists()


def test_classify_bronze_photo_row_requires_image_url(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not include an image_url"):
        classify_bronze_photo_row(
            {"flickr_photo_id": "123", "species_query": "Papilio demoleus"},
            classifier=object(),
            cache_root=tmp_path,
        )
