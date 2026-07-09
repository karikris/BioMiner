from __future__ import annotations

import pytest
import polars as pl

from biominer.evaluation.charts import (
    CALIBRATION_RELIABILITY_CHART_FILE,
    FAMILY_CONFUSION_CHART_FILE,
    REVIEW_REASON_COUNTS_CHART_FILE,
    SPECIES_ACCURACY_BY_FAMILY_CHART_FILE,
    write_evaluation_charts,
)


pytest.importorskip("matplotlib")


def test_write_evaluation_charts_writes_png_files(tmp_path) -> None:
    paths = write_evaluation_charts(
        family_confusion=pl.DataFrame(
            [
                {
                    "true_key": "gbif:9417",
                    "true_name": "Papilionidae",
                    "predicted_key": "gbif:9417",
                    "predicted_name": "Papilionidae",
                    "count": 2,
                    "classification_mode": "hierarchical_butterfly_classification",
                },
                {
                    "true_key": "gbif:7017",
                    "true_name": "Nymphalidae",
                    "predicted_key": "gbif:9417",
                    "predicted_name": "Papilionidae",
                    "count": 1,
                    "classification_mode": "hierarchical_butterfly_classification",
                },
            ]
        ),
        species_accuracy_by_family=pl.DataFrame(
            [
                {
                    "family_key": "gbif:9417",
                    "family": "Papilionidae",
                    "correct": 1,
                    "total": 2,
                    "accuracy": 0.5,
                },
                {
                    "family_key": "gbif:7017",
                    "family": "Nymphalidae",
                    "correct": 1,
                    "total": 1,
                    "accuracy": 1.0,
                },
            ]
        ),
        calibration_bins=pl.DataFrame(
            [
                {
                    "bin_index": 0,
                    "lower": 0.0,
                    "upper": 0.5,
                    "count": 1,
                    "avg_confidence": 0.25,
                    "accuracy": 0.0,
                    "gap": 0.25,
                    "weight": 0.5,
                },
                {
                    "bin_index": 1,
                    "lower": 0.5,
                    "upper": 1.0,
                    "count": 1,
                    "avg_confidence": 0.75,
                    "accuracy": 1.0,
                    "gap": 0.25,
                    "weight": 0.5,
                },
            ]
        ),
        review_error_examples=pl.DataFrame(
            [
                {"error_type": "species_top1_mismatch"},
                {"error_type": "species_top1_mismatch"},
                {"error_type": "missing_species_prediction"},
            ]
        ),
        output_dir=tmp_path,
    )

    assert paths == {
        "calibration_reliability_chart": str(tmp_path / CALIBRATION_RELIABILITY_CHART_FILE),
        "family_confusion_chart": str(tmp_path / FAMILY_CONFUSION_CHART_FILE),
        "review_reason_counts_chart": str(tmp_path / REVIEW_REASON_COUNTS_CHART_FILE),
        "species_accuracy_by_family_chart": str(tmp_path / SPECIES_ACCURACY_BY_FAMILY_CHART_FILE),
    }
    for filename in (
        FAMILY_CONFUSION_CHART_FILE,
        SPECIES_ACCURACY_BY_FAMILY_CHART_FILE,
        CALIBRATION_RELIABILITY_CHART_FILE,
        REVIEW_REASON_COUNTS_CHART_FILE,
    ):
        assert (tmp_path / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
