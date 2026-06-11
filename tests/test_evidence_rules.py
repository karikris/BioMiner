from __future__ import annotations

import polars as pl

from biominer.filter.rules import classify_evidence_frame, classify_evidence_row


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


def test_gold_score_gte_050_target_positive() -> None:
    result = classify_evidence_row(_row(human_verification_detected=False, bioclip_top1_score=0.50))

    assert result["publication_state"] == "gold"
    assert result["publication_state_reason"] == "target_positive_score_gte_050"
    assert result["occurrence_bin"] == "gold"
    assert result["bin_reason"] == "target_positive_score_gte_050"
    assert result["image_category"] == "adult_butterfly"
    assert result["life_stage"] == "adult_butterfly"
    assert result["review_reason"] == []


def test_silver_score_lt_050_target_positive() -> None:
    result = classify_evidence_row(_row(human_verification_detected=False, bioclip_top1_score=0.49))

    assert result["publication_state"] == "silver"
    assert result["publication_state_reason"] == "target_positive_score_lt_050"
    assert result["review_reason"] == []


def test_bronze_negative_material_museum_art_ai_other_insect() -> None:
    cases = [
        (_row(museum_detected=True), "negative_material_museum_specimen"),
        (_row(artwork_detected=True), "negative_material_artwork"),
        (_row(ai_generated_detected=True), "negative_material_ai_generated"),
        (_row(non_target_order_detected=True), "negative_material_non_target_order"),
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
    assert artwork["publication_state_reason"] == "negative_material_artwork"
    assert tattoo["image_category"] == "tattoo"
    assert tattoo["publication_state_reason"] == "negative_material_tattoo"
    assert museum["image_category"] == "museum_specimen"
    assert museum["publication_state_reason"] == "negative_material_museum_specimen"


def test_bronze_is_not_bioclip_positive_without_negative_material() -> None:
    result = classify_evidence_row(_row())

    assert result["publication_state"] == "gold"
    assert result["publication_state_reason"] == "target_positive_score_gte_050"


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


def test_hard_exclusion_flags_force_bronze_even_when_otherwise_gold() -> None:
    result = classify_evidence_row(_row(artwork_detected=True))

    assert result["publication_state"] == "bronze"
    assert result["publication_state_reason"] == "negative_material_artwork"
    assert result["review_reason"] == []


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
                _row(flickr_photo_id="silver", bioclip_top1_score=0.25),
                _row(flickr_photo_id="bronze", artwork_detected=True),
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
