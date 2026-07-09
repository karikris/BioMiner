from __future__ import annotations

import json

import polars as pl

from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION
from biominer.evaluation.reports import (
    CALIBRATION_BINS_FILE,
    EVALUATION_METRICS_FILE,
    EVALUATION_SUMMARY_FILE,
    FAMILY_CONFUSION_FILE,
    REVIEW_ERROR_EXAMPLES_FILE,
    SPECIES_CONFUSION_FILE,
    write_evaluation_report,
)


def test_write_evaluation_report_writes_json_parquet_and_markdown(tmp_path) -> None:
    paths = write_evaluation_report(
        object_scores=pl.DataFrame([_prediction()]),
        reviewed_labels=pl.DataFrame([_label()]),
        output_dir=tmp_path,
        run_manifest={"run_id": "run-1"},
    )

    assert sorted(paths) == [
        "calibration_bins",
        "family_confusion_matrix",
        "metrics",
        "review_error_examples",
        "species_confusion_matrix",
        "summary",
    ]
    for filename in (
        EVALUATION_METRICS_FILE,
        FAMILY_CONFUSION_FILE,
        SPECIES_CONFUSION_FILE,
        EVALUATION_SUMMARY_FILE,
        CALIBRATION_BINS_FILE,
        REVIEW_ERROR_EXAMPLES_FILE,
    ):
        assert (tmp_path / filename).exists()

    metrics = json.loads((tmp_path / EVALUATION_METRICS_FILE).read_text(encoding="utf-8"))
    family_confusion = pl.read_parquet(tmp_path / FAMILY_CONFUSION_FILE)
    species_confusion = pl.read_parquet(tmp_path / SPECIES_CONFUSION_FILE)
    calibration_bins = pl.read_parquet(tmp_path / CALIBRATION_BINS_FILE)
    review_errors = pl.read_parquet(tmp_path / REVIEW_ERROR_EXAMPLES_FILE)
    markdown = (tmp_path / EVALUATION_SUMMARY_FILE).read_text(encoding="utf-8")

    assert metrics["schema_version"] == "evaluation_metrics_v1"
    assert metrics["run"]["run_id"] == "run-1"
    assert metrics["metrics"]["species_top1_accuracy"] == 1.0
    assert metrics["metrics"]["species_top20_recall"] == 1.0
    assert metrics["calibration"]["sample_count"] == 1
    assert family_confusion.to_dicts()[0]["count"] == 1
    assert species_confusion.to_dicts()[0]["predicted_name"] == "Papilio demoleus"
    assert calibration_bins.height == 10
    assert review_errors.is_empty()
    assert "Family top1 accuracy" in markdown
    assert "Species top20 recall" in markdown
    assert "Species MRR" in markdown


def test_write_evaluation_report_handles_empty_inputs(tmp_path) -> None:
    write_evaluation_report(
        object_scores=pl.DataFrame(),
        reviewed_labels=pl.DataFrame(),
        output_dir=tmp_path,
    )

    metrics = json.loads((tmp_path / EVALUATION_METRICS_FILE).read_text(encoding="utf-8"))
    markdown = (tmp_path / EVALUATION_SUMMARY_FILE).read_text(encoding="utf-8")

    assert metrics["metrics"]["evaluated_objects"] == 0
    assert metrics["metrics"]["missing_prediction_count"] == 0
    assert pl.read_parquet(tmp_path / FAMILY_CONFUSION_FILE).is_empty()
    assert pl.read_parquet(tmp_path / SPECIES_CONFUSION_FILE).is_empty()
    assert "No reviewed object labels were available for evaluation." in markdown


def _prediction() -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": "1",
        "detection_id": "d1",
        "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        "taxonomy_table_version": "taxonomy-v1",
        "model_id": "bioclip",
        "model_checkpoint": "checkpoint-a",
        "family_top3": ["Papilionidae", "Nymphalidae", "Pieridae"],
        "family_top3_accepted_taxon_keys": ["gbif:9417", "gbif:7017", "gbif:5481"],
        "selected_family": "Papilionidae",
        "selected_family_key": "gbif:9417",
        "species_top1_scientific_name": "Papilio demoleus",
        "species_top1_accepted_taxon_key": "gbif:100",
        "accepted_taxon_key": "gbif:100",
        "species_top1_score": 0.91,
        "species_top5": ["Papilio demoleus", "Papilio machaon"],
        "species_top5_accepted_taxon_keys": ["gbif:100", "gbif:200"],
        "species_top20": ["Papilio demoleus", "Papilio machaon"],
        "species_top20_accepted_taxon_keys": ["gbif:100", "gbif:200"],
    }


def _label() -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": "1",
        "detection_id": "d1",
        "crop_hash": "sha256:d1",
        "label_level": "species",
        "is_butterfly": True,
        "accepted_taxon_key": "gbif:100",
        "scientific_name": "Papilio demoleus",
        "family_key": "gbif:9417",
        "family": "Papilionidae",
        "genus_key": "gbif:90",
        "genus": "Papilio",
        "label_source": "fixture",
        "reviewer_id": "reviewer-a",
        "reviewed_at": "2026-07-10T00:00:00Z",
        "review_confidence": "high",
        "review_notes": "synthetic",
    }
