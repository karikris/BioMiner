from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION
from support.evaluation_fixtures import (
    SYNTHETIC_EVALUATION_FIXTURE_VERSION,
    build_synthetic_evaluation_fixture,
)
from biominer.evaluation.metrics import evaluate_hierarchical_predictions
from biominer.evaluation.review_queue import build_hierarchical_review_queue


GOLDEN_EXPECTED_PATH = Path("tests/fixtures/evaluation_synthetic_expected_metrics.json")


def test_synthetic_hierarchical_evaluation_matches_golden_metrics() -> None:
    fixture = build_synthetic_evaluation_fixture()
    metrics = evaluate_hierarchical_predictions(
        object_scores=fixture.object_scores,
        reviewed_labels=fixture.reviewed_labels,
    )
    queue = build_hierarchical_review_queue(object_evidence=fixture.object_scores)

    actual = {
        "schema_version": "hierarchical-golden-evaluation-v1",
        "fixture_version": SYNTHETIC_EVALUATION_FIXTURE_VERSION,
        "metrics": _selected_metric_payload(metrics),
        "review_queue": {
            "rows": queue.height,
            "high_priority_rows": queue.filter(pl.col("review_priority") >= 80).height,
        },
    }

    expected = json.loads(GOLDEN_EXPECTED_PATH.read_text(encoding="utf-8"))
    assert actual == expected


def test_synthetic_hierarchical_regression_invariants_hold() -> None:
    fixture = build_synthetic_evaluation_fixture()

    eligible_detection_ids = set(
        fixture.object_detections.filter(pl.col("detector_label") == "butterfly_like")
        .select("detection_id")
        .to_series()
        .to_list()
    )
    scored_detection_ids = set(fixture.object_scores.select("detection_id").to_series().to_list())

    assert scored_detection_ids == eligible_detection_ids
    assert fixture.object_scores.select("classification_mode").to_series().unique().to_list() == [
        HIERARCHICAL_BUTTERFLY_CLASSIFICATION
    ]
    assert "target_species_score" not in fixture.object_scores.columns
    assert "target_accepted_taxon_key" not in fixture.object_scores.columns


def _selected_metric_payload(metrics: dict[str, object]) -> dict[str, object]:
    keys = [
        "butterfly_positive_labels",
        "family_top1_accuracy",
        "family_top3_recall",
        "missing_prediction_count",
        "negative_correct_count",
        "negative_labels",
        "selected_family_accuracy",
        "species_mrr",
        "species_top1_accuracy",
        "species_top20_recall",
        "species_top5_recall",
    ]
    return {key: _stable_value(metrics[key]) for key in keys}


def _stable_value(value: Any) -> object:
    if isinstance(value, float):
        return round(value, 6)
    return value
