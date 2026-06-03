from __future__ import annotations

from pathlib import Path

import pytest

from flickr_bio_occurrence.vision.image_cache import CachedImage
from flickr_bio_occurrence.vision.pipeline import build_bioclip_row_classifier, classify_bronze_photo_row


def test_classify_bronze_photo_row_caches_image_and_runs_classifier(tmp_path) -> None:
    calls: dict[str, object] = {}

    def fake_cache(image_url: str, *, cache_root: str | Path) -> CachedImage:
        calls["image_url"] = image_url
        calls["cache_root"] = cache_root
        return CachedImage(
            source_url=image_url,
            path=tmp_path / "image.jpg",
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


def test_classify_bronze_photo_row_requires_image_url(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not include an image_url"):
        classify_bronze_photo_row(
            {"flickr_photo_id": "123", "species_query": "Papilio demoleus"},
            classifier=object(),
            cache_root=tmp_path,
        )


def test_build_bioclip_row_classifier_uses_registry_runtime_and_cache_root(tmp_path) -> None:
    row_classifier = build_bioclip_row_classifier(
        model_registry_path="config/model_registry.toml",
        cache_root=tmp_path,
        scorer=lambda image_path, labels: {
            "a photo of Papilio demoleus": 0.99,
            "a photo of a moth": 0.01,
        },
        cache_image=lambda image_url, *, cache_root: CachedImage(
            source_url=image_url,
            path=tmp_path / "image.jpg",
            image_hash="sha256:image",
            content_type="image/jpeg",
            byte_size=10,
        ),
    )

    record = row_classifier(
        {
            "flickr_photo_id": "123",
            "image_url": "https://live.staticflickr.com/example.jpg",
            "species_query": "Papilio demoleus",
            "raw_title": "Papilio demoleus",
        }
    )

    assert record["model_version"] == "bioclip2_5_huge"
    assert record["top1_label"] == "a photo of Papilio demoleus"
    assert record["species_agreement_status"] == "exact_species_agreement"
