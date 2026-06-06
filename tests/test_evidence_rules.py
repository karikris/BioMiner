from __future__ import annotations

import polars as pl

from flickr_bio_occurrence.evidence.rules import REVIEW_REASON_PRECEDENCE, classify_evidence_frame, classify_evidence_row


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


def test_gold_requires_human_verification_and_positive_bioclip_agreement() -> None:
    result = classify_evidence_row(_row(human_verification_detected=True))

    assert result["publication_state"] == "gold"
    assert result["publication_state_reason"] == "human_verified_bioclip_positive"
    assert result["review_reason"] == []


def test_silver_allows_human_verification_with_missing_or_weak_or_conflicting_bioclip() -> None:
    missing = classify_evidence_row(_row(human_verification_detected=True, bioclip_top1_score=None, bioclip_species_agreement_status=""))
    weak = classify_evidence_row(_row(human_verification_detected=True, bioclip_top1_score=0.25))
    conflict = classify_evidence_row(
        _row(
            human_verification_detected=True,
            bioclip_top1_label="a photo of a moth",
            bioclip_species_agreement_status="text_vision_conflict",
        )
    )

    assert missing["publication_state"] == "silver"
    assert missing["publication_state_reason"] == "human_verified_bioclip_missing"
    assert weak["publication_state"] == "silver"
    assert weak["publication_state_reason"] == "human_verified_bioclip_low_confidence"
    assert conflict["publication_state"] == "silver"
    assert conflict["publication_state_reason"] == "human_verified_bioclip_conflict"


def test_bronze_is_bioclip_positive_without_human_verification() -> None:
    result = classify_evidence_row(_row(human_verification_detected=False))

    assert result["publication_state"] == "bronze"
    assert result["publication_state_reason"] == "bioclip_positive_without_human_verification"
    assert result["review_reason"] == []


def test_hard_exclusion_flags_force_in_review_even_when_otherwise_gold() -> None:
    result = classify_evidence_row(_row(human_verification_detected=True, artwork_detected=True))

    assert result["publication_state"] == "in_review"
    assert result["review_reason"] == ["artwork"]


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
    assert result["review_reason"] == list(REVIEW_REASON_PRECEDENCE)


def test_missing_image_prevents_bronze_and_provides_review_reason() -> None:
    result = classify_evidence_row(_row(image_url=None, human_verification_detected=False))

    assert result["publication_state"] == "in_review"
    assert result["review_reason"][0] == "missing_image"


def test_classify_evidence_frame_adds_exactly_one_state_per_row() -> None:
    frame = classify_evidence_frame(
        pl.DataFrame(
            [
                _row(flickr_photo_id="gold", human_verification_detected=True),
                _row(flickr_photo_id="silver", human_verification_detected=True, bioclip_top1_score=None),
                _row(flickr_photo_id="bronze"),
                _row(flickr_photo_id="review", image_url=None),
            ]
        )
    )

    states = frame["publication_state"].to_list()
    assert states == ["gold", "silver", "bronze", "in_review"]
    assert all(state in {"gold", "silver", "bronze", "in_review"} for state in states)
    review_rows = frame.filter(pl.col("publication_state") == "in_review")
    assert review_rows.to_dicts()[0]["review_reason"]
