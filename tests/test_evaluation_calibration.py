from __future__ import annotations

import hashlib

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from biominer.evaluation.calibration import (
    TARGET_CALIBRATION_RELIABILITY_SCHEMA,
    TARGET_THRESHOLD_OPERATING_POINT_SCHEMA,
    build_target_calibration_diagnostics,
    validate_target_calibration_diagnostics,
)


def test_target_calibration_bins_and_operating_points_are_weighted() -> None:
    diagnostics = build_target_calibration_diagnostics(
        pl.DataFrame(
            [
                _calibration_row("a", target=True, probability=0.9, weight=2.0),
                _calibration_row("b", target=False, probability=0.8, weight=1.0),
                _calibration_row("c", target=False, probability=0.2, weight=1.0),
            ]
        ),
        bin_count=5,
        thresholds=(0.5,),
        confidence_level=0.95,
    )

    validate_target_calibration_diagnostics(diagnostics)
    assert diagnostics.reliability.schema == TARGET_CALIBRATION_RELIABILITY_SCHEMA
    assert (
        diagnostics.operating_points.schema == TARGET_THRESHOLD_OPERATING_POINT_SCHEMA
    )
    assert diagnostics.reliability.height == 5
    low = diagnostics.reliability.filter(pl.col("bin_index") == 1).row(0, named=True)
    high = diagnostics.reliability.filter(pl.col("bin_index") == 4).row(0, named=True)
    assert low["item_count"] == 1
    assert low["mean_predicted_probability"] == pytest.approx(0.2)
    assert low["observed_target_rate"] == pytest.approx(0.0)
    assert high["item_count"] == 2
    assert high["weighted_item_count"] == pytest.approx(3.0)
    assert high["mean_predicted_probability"] == pytest.approx(2.6 / 3.0)
    assert high["observed_target_rate"] == pytest.approx(2.0 / 3.0)
    assert 0.0 <= high["observed_rate_ci_lower"] <= high["observed_target_rate"]
    assert high["observed_target_rate"] <= high["observed_rate_ci_upper"] <= 1.0
    assert diagnostics.reliability["ece_contribution"].drop_nulls().sum() == (
        pytest.approx(0.2)
    )

    point = diagnostics.operating_points.row(0, named=True)
    assert point["threshold"] == 0.5
    assert point["true_positive_weight"] == pytest.approx(2.0)
    assert point["false_positive_weight"] == pytest.approx(1.0)
    assert point["true_negative_weight"] == pytest.approx(1.0)
    assert point["false_negative_weight"] == pytest.approx(0.0)
    assert point["precision"] == pytest.approx(2.0 / 3.0)
    assert point["recall"] == pytest.approx(1.0)
    assert point["specificity"] == pytest.approx(0.5)
    assert point["calibration_method"] == "sigmoid"
    assert point["calibration_split_fingerprint"] == _sha("split")


def test_target_calibration_reports_missing_probability_coverage() -> None:
    diagnostics = build_target_calibration_diagnostics(
        pl.DataFrame(
            [
                _calibration_row("a", target=True, probability=0.9, weight=2.0),
                _calibration_row("b", target=False, probability=None, weight=1.0),
            ]
        ),
        bin_count=2,
        thresholds=(0.5,),
    )

    assert diagnostics.reliability["evaluation_item_count"].unique().to_list() == [2]
    assert diagnostics.reliability["probability_sample_count"].unique().to_list() == [1]
    assert diagnostics.reliability["missing_probability_count"].unique().to_list() == [
        1
    ]
    assert diagnostics.reliability[
        "weighted_probability_coverage"
    ].unique().to_list() == pytest.approx([2.0 / 3.0])


def test_target_calibration_rejects_raw_scores_and_is_deterministic() -> None:
    raw_only = pl.DataFrame(
        {
            "species_top1_score": [0.99],
            "target_present": [True],
        }
    )
    with pytest.raises(ValueError, match="calibrated_target_probability"):
        build_target_calibration_diagnostics(raw_only)

    rows = [
        _calibration_row("a", target=True, probability=0.9, weight=1.0),
        _calibration_row("b", target=False, probability=0.1, weight=1.0),
    ]
    first = build_target_calibration_diagnostics(pl.DataFrame(rows), bin_count=2)
    second = build_target_calibration_diagnostics(
        pl.DataFrame(list(reversed(rows))), bin_count=2
    )

    assert_frame_equal(first.reliability, second.reliability)
    assert_frame_equal(first.operating_points, second.operating_points)
    assert first.diagnostics_fingerprint == second.diagnostics_fingerprint


def test_target_calibration_rejects_invalid_provenance() -> None:
    invalid = _calibration_row("a", target=True, probability=0.9, weight=1.0)
    invalid["calibration_split_fingerprint"] = "split-v1"
    with pytest.raises(ValueError, match="calibration_split_fingerprint"):
        build_target_calibration_diagnostics(pl.DataFrame([invalid]))

    invalid = _calibration_row("a", target=True, probability=0.9, weight=1.0)
    invalid["calibration_method"] = "raw_cosine"
    with pytest.raises(ValueError, match="calibration_method"):
        build_target_calibration_diagnostics(pl.DataFrame([invalid]))


def _calibration_row(
    item_id: str,
    *,
    target: bool,
    probability: float | None,
    weight: float,
) -> dict[str, object]:
    return {
        "evaluation_item_id": item_id,
        "evaluation_set": "natural_stream",
        "sampling_weight": weight,
        "target_present": target,
        "calibrated_target_probability": probability,
        "calibration_method": "sigmoid",
        "calibration_split_fingerprint": _sha("split"),
        "calibrator_fingerprint": _sha("calibrator"),
    }


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
