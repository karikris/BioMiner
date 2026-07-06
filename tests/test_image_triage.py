from __future__ import annotations

from pathlib import Path

from biominer.bioclip.triage import classify_bioclip_triage


def _record(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source_record_id": "src-1",
        "flickr_photo_id": "1",
        "photo_page_url": "https://www.flickr.com/photos/user/1",
        "image_url": "https://live.staticflickr.com/1_large.jpg",
        "image_url_kind": "url_l",
        "latitude": "-27.0",
        "longitude": "153.0",
        "date_taken": "2024-05-06 10:30:00",
        "date_upload": "1715000000",
        "title": "Papilio demoleus",
    }
    row.update(overrides)
    return row


def test_gold_score_gt_070_with_text_species_match() -> None:
    result = classify_bioclip_triage(
        record=_record(),
        prediction={
            "species_top1_label": "a photo of Papilio demoleus",
            "species_top1_scientific_name": "Papilio demoleus",
            "species_top1_score": 0.71,
            "triage_top1_label": "a photo of an adult butterfly",
            "triage_top1_score": 0.88,
        },
    )

    assert result["triage_bin"] == "gold"
    assert result["triage_reason"] == "adult_butterfly_species_match_score_gt_070"
    assert result["occurrence_bin"] == "gold"
    assert result["bin_reason"] == "adult_butterfly_species_match_score_gt_070"
    assert result["image_category"] == "adult_butterfly"
    assert result["life_stage"] == "adult_butterfly"
    assert result["is_target_positive"] is True


def test_generic_butterfly_label_goes_to_bronze_not_gold() -> None:
    result = classify_bioclip_triage(
        record=_record(),
        prediction={"top1_label": "a photo of a butterfly", "top1_score": 0.99, "topk_json": []},
    )

    assert result["triage_bin"] == "bronze"
    assert result["triage_reason"] == "below_50"
    assert result["is_target_positive"] is False


def test_species_label_without_visual_triage_does_not_assert_adult_butterfly() -> None:
    result = classify_bioclip_triage(
        record=_record(title="Papilio demoleus lime butterfly"),
        prediction={
            "bioclip_top1_label": "a photo of lime butterfly",
            "bioclip_top1_score": 0.93,
            "species_top1_label": "a photo of lime butterfly",
            "species_top1_scientific_name": "Papilio demoleus",
            "species_top1_score": 0.93,
        },
    )

    assert result["triage_bin"] == "in_review"
    assert result["triage_reason"] == "missing_visual_butterfly_evidence"
    assert result["image_category"] == "unknown"
    assert result["life_stage"] == "unknown"
    assert result["is_butterfly_life_stage"] is False


def test_other_species_label_goes_to_bronze_not_gold() -> None:
    result = classify_bioclip_triage(
        record=_record(),
        prediction={
            "species_top1_label": "a photo of Papilio machaon",
            "species_top1_scientific_name": "Papilio machaon",
            "species_top1_score": 0.92,
            "triage_top1_label": "a photo of an adult butterfly",
            "triage_top1_score": 0.88,
        },
    )

    assert result["triage_bin"] == "bronze"
    assert result["triage_reason"] == "below_50"
    assert result["image_category"] == "adult_butterfly"


def test_target_species_below_silver_threshold_goes_to_bronze() -> None:
    result = classify_bioclip_triage(
        record=_record(),
        prediction={
            "species_top1_label": "a photo of Papilio demoleus",
            "species_top1_scientific_name": "Papilio demoleus",
            "species_top1_score": 0.34,
            "triage_top1_label": "a photo of an adult butterfly",
            "triage_top1_score": 0.88,
        },
    )

    assert result["triage_bin"] == "bronze"
    assert result["triage_reason"] == "below_50"


def test_missing_date_goes_to_silver() -> None:
    result = classify_bioclip_triage(
        record=_record(date_taken=None, date_upload=None, captured_at=None),
        prediction={
            "species_top1_label": "a photo of Papilio demoleus",
            "species_top1_scientific_name": "Papilio demoleus",
            "species_top1_score": 0.99,
            "triage_top1_label": "a photo of an adult butterfly",
            "triage_top1_score": 0.88,
        },
    )

    assert result["triage_bin"] == "silver"
    assert result["triage_reason"] == "missing_event_date"


def test_missing_geo_goes_to_in_review_no_geo() -> None:
    result = classify_bioclip_triage(
        record=_record(latitude=None, longitude=None),
        prediction={
            "species_top1_label": "a photo of Papilio demoleus",
            "species_top1_scientific_name": "Papilio demoleus",
            "species_top1_score": 0.99,
            "triage_top1_label": "a photo of an adult butterfly",
            "triage_top1_score": 0.88,
        },
    )

    assert result["occurrence_bin"] == "silver"
    assert result["triage_bin"] == "silver"
    assert result["bin_reason"] == "missing_geo"


def test_bronze_negative_material_museum_art_ai_other_insect() -> None:
    cases = [
        (_record(), "pinned_specimen", {"top1_label": "a photo of a pinned museum specimen"}),
        (_record(), "artwork", {"top1_label": "a photo of artwork or illustration"}),
        (_record(), "tattoo", {"top1_label": "a photo of a tattoo"}),
        (_record(), "ai_generated", {"top1_label": "an ai generated image"}),
        (_record(), "other_insect", {"top1_label": "a photo of an insect"}),
    ]

    for record, reason, *prediction_override in cases:
        prediction = {"top1_label": "a photo of Papilio demoleus", "top1_score": 0.99, "topk_json": []}
        prediction.update(prediction_override[0] if prediction_override else {})
        result = classify_bioclip_triage(record=record, prediction=prediction)
        assert result["triage_bin"] == "bin"
        assert result["triage_reason"] == reason
        assert result["occurrence_bin"] == "bin"
        assert result["bin_reason"] == reason
        assert result["negative_filter_reason"] == reason
        assert result["is_negative_material"] is True


def test_metadata_negative_flags_do_not_override_strong_species_prediction() -> None:
    result = classify_bioclip_triage(
        record=_record(museum_detected=True, artwork_detected=True, ai_generated_detected=True, other_insect_detected=True),
        prediction={
            "species_top1_label": "a photo of Papilio demoleus",
            "species_top1_scientific_name": "Papilio demoleus",
            "species_top1_score": 0.99,
            "triage_top1_label": "a photo of an adult butterfly",
            "triage_top1_score": 0.88,
        },
    )

    assert result["triage_bin"] == "gold"
    assert result["triage_reason"] == "adult_butterfly_species_match_score_gt_070"
    assert result["negative_filter_reason"] is None
    assert result["is_negative_material"] is False


def test_life_stage_labels_go_to_bronze_with_specific_stage() -> None:
    cases = [
        ("a photo of an egg", "egg"),
        ("a photo of a caterpillar", "caterpillar"),
        ("a photo of a larva", "larva"),
        ("a photo of a pupa", "pupa"),
        ("a photo of a chrysalis", "chrysalis"),
    ]

    for label, life_stage in cases:
        result = classify_bioclip_triage(
            record=_record(),
            prediction={"top1_label": label, "top1_score": 0.88, "topk_json": []},
        )

        assert result["triage_bin"] == "bronze"
        assert result["image_category"] == "life_stage_non_adult"
        assert result["life_stage"] == life_stage


def test_high_species_score_with_small_margin_goes_to_review() -> None:
    result = classify_bioclip_triage(
        record=_record(),
        prediction={
            "species_top1_label": "a photo of Papilio demoleus",
            "species_top1_scientific_name": "Papilio demoleus",
            "species_top1_score": 0.92,
            "species_top1_top2_margin": 0.02,
            "triage_group_top": "adult_butterfly",
            "triage_group_scores": {"adult_butterfly": 0.91, "hard_negative": 0.04},
        },
    )

    assert result["occurrence_bin"] == "in_review"
    assert result["bin_reason"] == "ambiguous_species_margin"


def test_hard_negative_group_overrides_species_score() -> None:
    result = classify_bioclip_triage(
        record=_record(),
        prediction={
            "species_top1_label": "a photo of Papilio demoleus",
            "species_top1_scientific_name": "Papilio demoleus",
            "species_top1_score": 0.95,
            "species_top1_top2_margin": 0.50,
            "triage_group_top": "hard_negative",
            "triage_group_scores": {"adult_butterfly": 0.08, "hard_negative": 0.87},
        },
    )

    assert result["occurrence_bin"] == "bin"
    assert result["bin_reason"] == "hard_negative_group"


def test_gold_requires_genus_consistency_when_available() -> None:
    result = classify_bioclip_triage(
        record=_record(),
        prediction={
            "species_top1_label": "a photo of Papilio demoleus",
            "species_top1_scientific_name": "Papilio demoleus",
            "species_top1_score": 0.93,
            "species_top1_top2_margin": 0.30,
            "species_top1_genus": "Papilio",
            "species_top1_family": "Papilionidae",
            "genus_top1": "Danaus",
            "family_top1": "Nymphalidae",
        },
    )

    assert result["occurrence_bin"] == "in_review"
    assert result["bin_reason"] == "taxonomy_inconsistent"


def test_no_new_darwin_core_logic_added() -> None:
    triage_source = Path("src/biominer/bioclip/triage.py").read_text(encoding="utf-8")

    assert "Darwin" not in triage_source
    assert "identificationVerificationStatus" not in triage_source
    assert "dwc" not in triage_source.casefold()
