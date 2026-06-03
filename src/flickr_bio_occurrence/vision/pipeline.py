from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

from flickr_bio_occurrence.vision.bioclip import BioClipClassifier, BioClipScorer, PersistentBioClipScorer
from flickr_bio_occurrence.vision.image_cache import CachedImage, cache_image_from_url
from flickr_bio_occurrence.vision.model_registry import ModelRegistry


class ImageClassifier(Protocol):
    def classify_image(self, **kwargs: object) -> dict[str, object]:
        ...


CacheImage = Callable[..., CachedImage]
RowClassifier = Callable[[dict[str, Any]], dict[str, object]]


def build_bioclip_row_classifier(
    *,
    model_registry_path: str | Path = "config/model_registry.toml",
    cache_root: str | Path = "data/cache/images",
    scorer: BioClipScorer | None = None,
    cache_image: CacheImage = cache_image_from_url,
) -> RowClassifier:
    runtime = ModelRegistry.from_config(model_registry_path).require_preferred_bioclip_runtime()
    effective_scorer = scorer or PersistentBioClipScorer(runtime=runtime)
    classifier = BioClipClassifier(runtime=runtime, scorer=effective_scorer)

    def classify_row(row: dict[str, Any]) -> dict[str, object]:
        return classify_bronze_photo_row(
            row,
            classifier=classifier,
            cache_root=cache_root,
            cache_image=cache_image,
        )

    close = getattr(effective_scorer, "close", None)
    if callable(close):
        setattr(classify_row, "close", close)
    setattr(classify_row, "scorer", effective_scorer)
    return classify_row


def classify_bronze_photo_row(
    row: dict[str, Any],
    *,
    classifier: ImageClassifier,
    cache_root: str | Path = "data/cache/images",
    cache_image: CacheImage = cache_image_from_url,
) -> dict[str, object]:
    image_url = row.get("image_url")
    if not image_url:
        raise ValueError(f"Bronze photo row {row.get('flickr_photo_id', '<unknown>')} does not include an image_url")

    cached = cache_image(str(image_url), cache_root=cache_root)
    return classifier.classify_image(
        flickr_photo_id=str(row["flickr_photo_id"]),
        image_path=cached.path,
        image_hash=cached.image_hash,
        image_url_used=cached.source_url,
        resolved_scientific_name=str(row.get("species_query") or ""),
        text_evidence_present=_text_evidence_present(row),
    )


def _text_evidence_present(row: dict[str, Any]) -> bool:
    return bool(row.get("raw_title") or row.get("raw_description") or row.get("raw_tags") or row.get("machine_tags"))
