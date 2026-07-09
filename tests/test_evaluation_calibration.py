from __future__ import annotations

import json

import polars as pl
import pytest

from biominer.evaluation.calibration import (
    CALIBRATION_MODE,
    add_uncertainty_fields,
    expected_calibration_error,
    score_margin,
    topk_entropy,
)


def test_topk_entropy_is_higher_for_flat_scores_than_peaked_scores() -> None:
    flat = topk_entropy([1.0, 1.0, 1.0])
    peaked = topk_entropy([0.98, 0.01, 0.01])

    assert flat is not None
    assert peaked is not None
    assert flat > peaked


def test_score_margin_uses_ordered_top_two_scores() -> None:
    assert score_margin({"species_top5_scores": [0.8, 0.55, 0.1]}) == pytest.approx(0.25)
    assert score_margin({"species_top5_scores": json.dumps([0.9, 0.4])}) == pytest.approx(0.5)
    assert score_margin({"species_top5_scores": [0.9]}) is None


def test_expected_calibration_error_bins_are_stable() -> None:
    result = expected_calibration_error(
        predictions=pl.DataFrame(
            [
                {"species_top1_score": 0.9, "species_top1_correct": True},
                {"species_top1_score": 0.8, "species_top1_correct": False},
                {"species_top1_score": 0.2, "species_top1_correct": False},
            ]
        ),
        labels=pl.DataFrame([{"label": "a"}, {"label": "b"}, {"label": "c"}]),
        score_column="species_top1_score",
        correct_column="species_top1_correct",
        bins=5,
    )

    assert result["calibration_mode"] == CALIBRATION_MODE
    assert result["sample_count"] == 3
    assert result["ece"] == pytest.approx(0.3)
    assert result["bins"][1]["count"] == 1
    assert result["bins"][4]["count"] == 2
    assert result["bins"][4]["accuracy"] == pytest.approx(0.5)


def test_expected_calibration_error_skips_missing_scores_without_crashing() -> None:
    result = expected_calibration_error(
        predictions=pl.DataFrame(
            [
                {"species_top1_score": None, "species_top1_correct": True},
                {"species_top1_score": 0.7, "species_top1_correct": None},
            ]
        ),
        labels=pl.DataFrame([{"label": "a"}]),
        score_column="species_top1_score",
        correct_column="species_top1_correct",
        bins=3,
    )

    assert result["sample_count"] == 0
    assert result["ece"] == 0.0
    assert [row["count"] for row in result["bins"]] == [0, 0, 0]


def test_add_uncertainty_fields_marks_low_margin_and_family_conflict() -> None:
    frame = add_uncertainty_fields(
        pl.DataFrame(
            [
                {
                    "species_top5_scores": [0.51, 0.50, 0.10],
                    "family_top3_scores": [0.80, 0.20],
                    "selected_family_key": "gbif:9417",
                    "species_candidate_family_key": "gbif:7017",
                }
            ]
        ),
        low_margin_threshold=0.05,
    )
    row = frame.to_dicts()[0]

    assert row["species_top1_margin"] == pytest.approx(0.01)
    assert row["family_margin"] == pytest.approx(0.60)
    assert row["species_top5_entropy"] is not None
    assert row["low_margin_flag"] is True
    assert row["family_species_conflict_flag"] is True
