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
    }
    row.update(overrides)
    return row


def test_gold_score_gte_050_target_positive() -> None:
    result = classify_bioclip_triage(
        record=_record(),
        prediction={"top1_label": "a photo of Papilio demoleus", "top1_score": 0.50, "topk_json": []},
    )

    assert result["triage_bin"] == "gold"
    assert result["triage_reason"] == "adult_lepidoptera_with_date_geo"
    assert result["occurrence_bin"] == "gold"
    assert result["bin_reason"] == "adult_lepidoptera_with_date_geo"
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


def test_target_species_below_threshold_goes_to_bronze() -> None:
    result = classify_bioclip_triage(
        record=_record(),
        prediction={
            "species_top1_label": "a photo of Papilio demoleus",
            "species_top1_scientific_name": "Papilio demoleus",
            "species_top1_score": 0.49,
            "triage_top1_label": "a photo of an adult butterfly",
            "triage_top1_score": 0.88,
        },
    )

    assert result["triage_bin"] == "bronze"
    assert result["triage_reason"] == "below_50"


def test_missing_date_goes_to_silver() -> None:
    result = classify_bioclip_triage(
        record=_record(date_taken=None, date_upload=None, captured_at=None),
        prediction={"top1_label": "a photo of Papilio demoleus", "top1_score": 0.99, "topk_json": []},
    )

    assert result["triage_bin"] == "silver"
    assert result["triage_reason"] == "missing_event_date"


def test_missing_geo_goes_to_in_review_no_geo() -> None:
    result = classify_bioclip_triage(
        record=_record(latitude=None, longitude=None),
        prediction={"top1_label": "a photo of Papilio demoleus", "top1_score": 0.99, "topk_json": []},
    )

    assert result["occurrence_bin"] == "in_review/no_geo"
    assert result["triage_bin"] == "in_review/no_geo"
    assert result["bin_reason"] == "no_geo"


def test_bronze_negative_material_museum_art_ai_other_insect() -> None:
    cases = [
        (_record(museum_detected=True), "museum_specimen"),
        (_record(), "artwork", {"top1_label": "a photo of artwork or illustration"}),
        (_record(tattoo_detected=True), "tattoo"),
        (_record(ai_generated_detected=True), "AI_generated"),
        (_record(other_insect_detected=True), "other_insect"),
    ]

    for record, reason, *prediction_override in cases:
        prediction = {"top1_label": "a photo of Papilio demoleus", "top1_score": 0.99, "topk_json": []}
        prediction.update(prediction_override[0] if prediction_override else {})
        result = classify_bioclip_triage(record=record, prediction=prediction)
        assert result["triage_bin"] == "bronze"
        assert result["triage_reason"] == reason
        assert result["occurrence_bin"] == "bronze"
        assert result["bin_reason"] == reason
        assert result["negative_filter_reason"] == reason
        assert result["is_negative_material"] is True


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


def test_no_new_darwin_core_logic_added() -> None:
    triage_source = Path("src/biominer/vision/triage.py").read_text(encoding="utf-8")

    assert "Darwin" not in triage_source
    assert "identificationVerificationStatus" not in triage_source
    assert "dwc" not in triage_source.casefold()
