from __future__ import annotations

from flickr_bio_occurrence.review.rules import review_status_for_candidate
from flickr_bio_occurrence.vision.model_registry import ModelRegistry


def test_bioclip2_or_newest_model_is_preferred() -> None:
    registry = ModelRegistry.from_config("config/model_registry.toml")
    selected = registry.resolve_preferred_bioclip()

    assert selected.model_id == "bioclip2"
    assert selected.role == "preferred"


def test_bioclip_conflict_routes_to_review() -> None:
    status = review_status_for_candidate(species_agreement_status="text_vision_conflict", range_extension_candidate=False)

    assert status == "needs_review"
