from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from flickr_bio_occurrence.vision.image_cache import CachedImage
from flickr_bio_occurrence.vision.triage import classify_bioclip_triage, process_image_triage_records


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


def test_in_review_missing_image_url(tmp_path) -> None:
    run = process_image_triage_records(
        [_record(image_url=None)],
        classifier=FakeClassifier(),
        output_path=tmp_path / "image_triage.parquet",
        cache_image=forbidden_cache,
    )

    row = run.frame.to_dicts()[0]
    assert row["triage_bin"] == "in_review"
    assert row["classification_status"] == "invalid_record"
    assert "missing image URL" in row["classification_error"]


def test_downloaded_image_deleted_after_successful_classification(tmp_path) -> None:
    image_path = tmp_path / "cache" / "image.jpg"

    run = process_image_triage_records(
        [_record()],
        classifier=FakeClassifier(),
        output_path=tmp_path / "image_triage.parquet",
        cache_root=tmp_path / "cache",
        cache_image=fake_cache(image_path),
    )

    row = run.frame.to_dicts()[0]
    assert row["classification_status"] == "success"
    assert row["image_deleted_after_classification"] is True
    assert run.images_deleted_after_classification == 1
    assert not image_path.exists()


def test_hash_or_identifier_stored_after_classification(tmp_path) -> None:
    image_path = tmp_path / "cache" / "image.jpg"

    run = process_image_triage_records(
        [_record()],
        classifier=FakeClassifier(),
        output_path=tmp_path / "image_triage.parquet",
        cache_root=tmp_path / "cache",
        cache_image=fake_cache(image_path),
        now=datetime(2026, 6, 7, tzinfo=UTC),
    )
    row = run.frame.to_dicts()[0]

    assert row["source"] == "flickr"
    assert row["flickr_photo_id"] == "1"
    assert row["image_url_kind"] == "url_l"
    assert row["image_hash"] == "sha256:image"
    assert row["source_record_hash"].startswith("sha256:")
    assert row["classified_at"] == "2026-06-07T00:00:00+00:00"
    assert row["latitude"] == -27.0
    assert row["longitude"] == 153.0
    assert row["year"] == 2024
    assert row["month"] == 5


def test_successfully_processed_record_skipped_on_rerun(tmp_path) -> None:
    output = tmp_path / "image_triage.parquet"
    image_path = tmp_path / "cache" / "image.jpg"
    first_classifier = FakeClassifier()
    second_classifier = FakeClassifier()

    first = process_image_triage_records(
        [_record()],
        classifier=first_classifier,
        output_path=output,
        cache_root=tmp_path / "cache",
        cache_image=fake_cache(image_path),
    )
    second = process_image_triage_records(
        [_record()],
        classifier=second_classifier,
        output_path=output,
        cache_root=tmp_path / "cache",
        cache_image=forbidden_cache,
    )

    assert first.records_classified == 1
    assert second.records_skipped_existing == 1
    assert second_classifier.calls == 0
    assert second.frame.filter(pl.col("classification_status") == "success").height == 1
    assert second.frame.filter(pl.col("classification_status") == "skipped_existing").height == 1
    assert second.frame.filter(pl.col("classification_status") == "skipped_existing").to_dicts()[0]["triage_reason"] == "duplicate_successful_record"


def test_same_image_with_different_model_checkpoint_is_reprocessed(tmp_path) -> None:
    output = tmp_path / "image_triage.parquet"
    image_path = tmp_path / "cache" / "image.jpg"
    first_classifier = FakeClassifier()
    second_classifier = FakeClassifier()

    first = process_image_triage_records(
        [_record()],
        classifier=first_classifier,
        output_path=output,
        cache_root=tmp_path / "cache",
        cache_image=fake_cache(image_path),
        model_checkpoint="checkpoint-a",
    )
    second = process_image_triage_records(
        [_record()],
        classifier=second_classifier,
        output_path=output,
        cache_root=tmp_path / "cache",
        cache_image=fake_cache(image_path),
        model_checkpoint="checkpoint-b",
    )

    assert first.records_classified == 1
    assert second.records_classified == 1
    assert second.records_skipped_existing == 0
    assert second_classifier.calls == 1


def test_failed_download_is_recorded_without_image_file(tmp_path) -> None:
    def failing_cache(*args, **kwargs):  # noqa: ANN002, ANN003 - test fake.
        raise RuntimeError("download failed")

    run = process_image_triage_records(
        [_record()],
        classifier=FakeClassifier(),
        output_path=tmp_path / "image_triage.parquet",
        cache_root=tmp_path / "cache",
        cache_image=failing_cache,
    )
    row = run.frame.to_dicts()[0]

    assert run.download_failures == 1
    assert row["classification_status"] == "failed_download"
    assert row["classification_error"] == "download failed"
    assert row["image_downloaded"] is False
    assert row["image_deleted_after_classification"] is False


def test_failed_bioclip_deletes_downloaded_image_and_records_failure(tmp_path) -> None:
    image_path = tmp_path / "cache" / "failed.jpg"

    run = process_image_triage_records(
        [_record()],
        classifier=FailingClassifier(),
        output_path=tmp_path / "image_triage.parquet",
        cache_root=tmp_path / "cache",
        cache_image=fake_cache(image_path),
    )
    row = run.frame.to_dicts()[0]

    assert run.bioclip_failures == 1
    assert row["classification_status"] == "failed_bioclip"
    assert row["image_downloaded"] is True
    assert row["image_deleted_after_classification"] is True
    assert not image_path.exists()


def test_no_new_darwin_core_logic_added() -> None:
    triage_source = Path("src/flickr_bio_occurrence/vision/triage.py").read_text(encoding="utf-8")

    assert "Darwin" not in triage_source
    assert "identificationVerificationStatus" not in triage_source
    assert "dwc" not in triage_source.casefold()


class FakeClassifier:
    def __init__(self) -> None:
        self.calls = 0

    def classify_image(self, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        return {
            "top1_label": "a photo of Papilio demoleus",
            "top1_score": 0.91,
            "topk_json": [{"label": "a photo of Papilio demoleus", "score": 0.91}],
        }


class FailingClassifier:
    def classify_image(self, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("bioclip failed")


def fake_cache(image_path: Path):
    def cache_image(image_url: str, *, cache_root: str | Path) -> CachedImage:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"jpeg")
        return CachedImage(
            source_url=image_url,
            path=image_path,
            image_hash="sha256:image",
            content_type="image/jpeg",
            byte_size=4,
        )

    return cache_image


def forbidden_cache(*args, **kwargs):  # noqa: ANN002, ANN003 - test guard.
    raise AssertionError("cache should not be called")
