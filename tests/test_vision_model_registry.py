from __future__ import annotations

from flickr_bio_occurrence.review.rules import review_status_for_candidate
from flickr_bio_occurrence.vision.model_registry import ModelRegistry


def test_bioclip2_or_newest_model_is_preferred() -> None:
    registry = ModelRegistry.from_config("config/model_registry.toml")
    selected = registry.resolve_preferred_bioclip()

    assert selected.model_id == "bioclip2_5_huge"
    assert selected.role == "preferred"
    assert selected.model_name == "imageomics/bioclip-2"
    assert selected.package_name == "open_clip_torch"


def test_bioclip2_remains_available_as_fallback_to_newest() -> None:
    registry = ModelRegistry.from_config("config/model_registry.toml")

    assert registry.models["bioclip2"].role == "fallback"
    assert registry.models["bioclip1"].role == "fallback"


def test_bioclip_conflict_routes_to_review() -> None:
    status = review_status_for_candidate(species_agreement_status="text_vision_conflict", range_extension_candidate=False)

    assert status == "needs_review"
