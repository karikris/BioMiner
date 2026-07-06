from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import polars as pl

from biominer.evidence.buckets import (
    classify_evidence_frame,
    classify_evidence_row,
    object_metadata_review_reason,
    object_occurrence_bucket,
    photo_bucket_and_reason,
)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "flickr_photo_id": "1",
        "image_url": "https://live.staticflickr.com/large.jpg",
        "species_query": "Papilio demoleus",
        "species_text_match": True,
        "scientific_names_detected": ["Papilio demoleus"],
        "human_verification_detected": False,
        "comments_count": 1,
        "museum_detected": False,
        "artwork_detected": False,
        "specimen_detected": False,
        "collection_detected": False,
        "captive_detected": False,
        "non_target_order_detected": False,
        "review_flags": [],
        "bioclip_top1_label": "a photo of Papilio demoleus",
        "bioclip_top1_score": 0.75,
        "bioclip_species_agreement_status": "exact_species_agreement",
    }
    row.update(overrides)
    return row


def test_legacy_filter_rules_and_bucket_report_wrappers_are_removed() -> None:
    assert importlib.util.find_spec("biominer.filter.rules") is None
    assert importlib.util.find_spec("biominer.reports.buckets") is None


def test_gold_score_gte_070_target_positive() -> None:
    result = classify_evidence_row(_row(human_verification_detected=False, bioclip_top1_score=0.70))

    assert result["publication_state"] == "gold"
    assert result["publication_state_reason"] == "target_positive_score_gte_070"
    assert result["occurrence_bin"] == "gold"
    assert result["bin_reason"] == "target_positive_score_gte_070"
    assert result["image_category"] == "unknown"
    assert result["life_stage"] == "unknown"
    assert result["review_reason"] == []


def test_object_occurrence_bucket_policy_lives_in_evidence_package() -> None:
    item = {"latitude": 45.0, "longitude": -93.0, "date_taken": "2024-07-01"}

    assert object_occurrence_bucket(item=item, target_score=0.72, target_rank=1, margin=0.25, geo=SimpleNamespace(route_to_review=False)) == (
        "gold",
        "target_species_score_ge_070",
    )
    assert object_occurrence_bucket(item=item, target_score=0.72, target_rank=2, margin=0.25, geo=SimpleNamespace(route_to_review=False)) == (
        "in_review",
        "species_conflict",
    )
    assert object_occurrence_bucket(
        item=item,
        target_score=0.72,
        target_rank=1,
        margin=0.25,
        geo=SimpleNamespace(route_to_review=True, reason="geospatial_conflict"),
    ) == ("in_review", "geospatial_conflict")
    assert object_occurrence_bucket(
        item={"image_category": "artwork", "latitude": 45.0, "longitude": -93.0, "date_taken": "2024-07-01"},
        target_score=0.99,
        target_rank=1,
        margin=0.5,
        geo=SimpleNamespace(route_to_review=False),
    ) == (
        "gold",
        "target_species_score_ge_070",
    )
    assert object_occurrence_bucket(item={"detector_label": "hard_negative"}, target_score=0.99, target_rank=1, margin=0.5, geo=SimpleNamespace(route_to_review=False)) == (
        "bin",
        "negative_material_hard_negative_object",
    )


def test_photo_bucket_policy_lives_in_evidence_package() -> None:
    rows = [
        {"occurrence_bin": "gold", "bin_reason": "target_species_score_ge_070"},
        {"occurrence_bin": "in_review", "bin_reason": "geospatial_conflict"},
    ]

    assert photo_bucket_and_reason(rows, {}) == ("in_review", "geospatial_conflict")
    assert photo_bucket_and_reason(rows, {"is_negative_material": True, "negative_filter_reason": "artwork"}) == ("in_review", "geospatial_conflict")


def test_object_metadata_hints_are_review_context_not_hard_negative_bins() -> None:
    assert object_metadata_review_reason({"is_negative_material": True, "negative_filter_reason": "artwork"}) == "artwork"
    assert object_metadata_review_reason({"artwork_hint": True}) == "artwork"
    assert object_metadata_review_reason({"artwork_detected": True}) == "artwork"
    assert object_metadata_review_reason({"specimen_detected": True}) == "museum_specimen"
    assert object_metadata_review_reason({"metadata_negative_reason_hint": "other_insect"}) == "non_target_order"
    assert object_occurrence_bucket(
        item={"artwork_hint": True, "latitude": "", "longitude": "", "date_taken": ""},
        target_score=0.2,
        target_rank=1,
        margin=0.1,
        geo=SimpleNamespace(route_to_review=False),
    ) == ("in_review", "artwork")


def test_silver_score_035_to_070_target_positive() -> None:
    result = classify_evidence_row(_row(human_verification_detected=False, bioclip_top1_score=0.49))

    assert result["publication_state"] == "silver"
    assert result["publication_state_reason"] == "target_positive_score_035_to_070"
    assert result["review_reason"] == []


def test_bronze_negative_material_museum_art_ai_other_insect() -> None:
    cases = [
        (_row(museum_detected=True, bioclip_top1_score=0.2, bioclip_top1_label="a photo of a butterfly"), "negative_material_museum_specimen"),
        (_row(artwork_detected=True, bioclip_top1_score=0.2, bioclip_top1_label="a photo of a butterfly"), "negative_material_artwork"),
        (_row(ai_generated_detected=True, bioclip_top1_score=0.2, bioclip_top1_label="a photo of a butterfly"), "negative_material_ai_generated"),
        (_row(non_target_order_detected=True, bioclip_top1_score=0.2, bioclip_top1_label="a photo of a butterfly"), "negative_material_non_target_order"),
        (_row(bioclip_top1_label="a photo of a moth", bioclip_species_agreement_status="text_vision_conflict"), "negative_material_non_butterfly"),
    ]

    for row, reason in cases:
        result = classify_evidence_row(row)
        assert result["publication_state"] == "bronze"
        assert result["publication_state_reason"] == reason
        assert result["occurrence_bin"] == "bronze"
        assert result["bin_reason"] == reason
        assert result["negative_filter_reason"] is not None
        assert result["review_reason"] == []


def test_existing_art_tattoo_museum_logic_uses_image_category() -> None:
    artwork = classify_evidence_row(_row(artwork_detected=True))
    tattoo = classify_evidence_row(_row(tattoo_detected=True))
    museum = classify_evidence_row(_row(museum_detected=True))

    assert artwork["image_category"] == "artwork"
    assert artwork["publication_state_reason"] == "target_positive_score_gte_070"
    assert tattoo["image_category"] == "tattoo"
    assert tattoo["publication_state_reason"] == "target_positive_score_gte_070"
    assert museum["image_category"] == "museum_specimen"
    assert museum["publication_state_reason"] == "target_positive_score_gte_070"


def test_bronze_is_not_bioclip_positive_without_negative_material() -> None:
    result = classify_evidence_row(_row())

    assert result["publication_state"] == "gold"
    assert result["publication_state_reason"] == "target_positive_score_gte_070"


def test_generic_butterfly_label_goes_to_bronze_not_gold() -> None:
    result = classify_evidence_row(
        _row(
            bioclip_top1_label="a photo of a butterfly",
            bioclip_species_agreement_status="same_family_agreement",
            bioclip_top1_score=0.95,
        )
    )

    assert result["publication_state"] == "bronze"
    assert result["publication_state_reason"] == "below_50"


def test_metadata_negative_flags_do_not_demote_strong_visual_target_evidence() -> None:
    result = classify_evidence_row(_row(artwork_detected=True))

    assert result["publication_state"] == "gold"
    assert result["publication_state_reason"] == "target_positive_score_gte_070"
    assert result["review_reason"] == []


def test_metadata_flags_do_not_demote_strong_visual_target_evidence() -> None:
    result = classify_evidence_row(
        _row(
            artwork_hint=True,
            logo_or_brand_hint=True,
            textile_or_pattern_hint=True,
            object_or_product_hint=True,
            hard_negative_text_hint=True,
            matched_keyword_groups=["artwork"],
            matched_keywords=["illustration"],
        )
    )

    assert result["publication_state"] == "gold"
    assert result["publication_state_reason"] == "target_positive_score_gte_070"
    assert result["review_reason"] == []


def test_metadata_flags_are_review_context_when_visual_scoring_is_missing() -> None:
    result = classify_evidence_row(
        _row(
            bioclip_top1_score=None,
            artwork_hint=True,
            hard_negative_text_hint=True,
            matched_keyword_groups=["artwork"],
            matched_keywords=["illustration"],
        )
    )

    assert result["publication_state"] == "in_review"
    assert "artwork" in result["review_reason"]
    assert "missing_bioclip" in result["review_reason"]


def test_all_metadata_hard_negative_hints_are_review_context_when_visual_scoring_is_missing() -> None:
    result = classify_evidence_row(
        _row(
            bioclip_top1_score=None,
            logo_or_brand_hint=True,
            textile_or_pattern_hint=True,
            object_or_product_hint=True,
            hard_negative_text_hint=True,
            matched_keyword_groups=["logo_or_brand", "textile_or_pattern", "object_or_product"],
            matched_keywords=["logo", "pattern", "toy"],
        )
    )

    assert result["publication_state"] == "in_review"
    assert "logo_or_brand" in result["review_reason"]
    assert "textile_or_pattern" in result["review_reason"]
    assert "object_or_product" in result["review_reason"]
    assert "missing_bioclip" in result["review_reason"]


def test_inferred_metadata_object_categories_are_review_context_not_hard_drops() -> None:
    result = classify_evidence_row(
        _row(
            raw_title="Danaus plexippus butterfly sticker logo",
            bioclip_top1_score=None,
            hard_negative_text_hint=True,
            metadata_negative_reason_hint="logo_or_brand",
        )
    )

    assert result["publication_state"] == "in_review"
    assert result["image_category"] == "logo_or_brand"
    assert "logo_or_brand" in result["review_reason"]
    assert "missing_bioclip" in result["review_reason"]


def test_in_review_rows_get_precedent_review_reasons() -> None:
    result = classify_evidence_row(
        _row(
            image_url=None,
            artwork_detected=True,
            museum_detected=True,
            specimen_detected=True,
            non_target_order_detected=True,
            bioclip_top1_label="a photo of a moth",
            bioclip_species_agreement_status="text_vision_conflict",
            scientific_names_detected=["Papilio demoleus", "Danaus plexippus"],
            captive_detected=True,
            bioclip_top1_score=0.2,
            human_verification_detected=False,
            comments_count=0,
            api_error=True,
        )
    )

    assert result["publication_state"] == "in_review"
    assert result["review_reason"] == [
        "missing_image",
        "artwork",
        "museum_specimen",
        "non_target_order",
        "species_conflict",
        "multiple_species",
        "captivity_suspected",
        "low_confidence",
        "api_error",
    ]


def test_missing_image_prevents_bronze_and_provides_review_reason() -> None:
    result = classify_evidence_row(_row(image_url=None, human_verification_detected=False))

    assert result["publication_state"] == "in_review"
    assert result["review_reason"][0] == "missing_image"


def test_classify_evidence_frame_adds_exactly_one_state_per_row() -> None:
    frame = classify_evidence_frame(
        pl.DataFrame(
            [
                _row(flickr_photo_id="gold"),
                _row(flickr_photo_id="silver", bioclip_top1_score=0.49),
                _row(flickr_photo_id="bronze", artwork_detected=True, bioclip_top1_score=0.2, bioclip_top1_label="a photo of a butterfly"),
                _row(flickr_photo_id="review", image_url=None),
            ]
        )
    )

    states = frame["publication_state"].to_list()
    assert states == ["gold", "silver", "bronze", "in_review"]
    assert all(state in {"gold", "silver", "bronze", "in_review"} for state in states)
    review_rows = frame.filter(pl.col("publication_state") == "in_review")
    assert review_rows.to_dicts()[0]["review_reason"]


def test_no_legacy_human_verification_gold_gate_remains() -> None:
    result = classify_evidence_row(_row(human_verification_detected=False))

    assert result["publication_state"] == "gold"
    assert "human_verified" not in result["publication_state_reason"]
