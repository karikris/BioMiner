from __future__ import annotations

import polars as pl
import pytest

from support.evaluation_fixtures import (
    SYNTHETIC_EVALUATION_FIXTURE_VERSION,
    build_synthetic_evaluation_fixture,
    write_synthetic_evaluation_fixture,
)
from biominer.evaluation.metrics import evaluate_hierarchical_predictions
from biominer.evaluation.review_queue import build_hierarchical_review_queue


def test_synthetic_evaluation_fixture_builder_is_deterministic() -> None:
    first = build_synthetic_evaluation_fixture()
    second = build_synthetic_evaluation_fixture()

    for name in first.to_frames():
        assert first.to_frames()[name].to_dicts() == second.to_frames()[name].to_dicts()


def test_synthetic_evaluation_fixture_has_expected_taxonomy_shape() -> None:
    fixture = build_synthetic_evaluation_fixture()
    taxa = fixture.classification_taxa

    assert taxa.filter(pl.col("rank") == "FAMILY").height == 3
    assert taxa.filter(pl.col("rank") == "SPECIES").height == 30
    assert set(taxa.select("fixture_version").to_series().drop_nulls().to_list()) == {
        SYNTHETIC_EVALUATION_FIXTURE_VERSION
    }


def test_synthetic_evaluation_fixture_metrics_are_exact() -> None:
    fixture = build_synthetic_evaluation_fixture()

    metrics = evaluate_hierarchical_predictions(
        object_scores=fixture.object_scores,
        reviewed_labels=fixture.reviewed_labels,
    )

    assert metrics["butterfly_positive_labels"] == 4
    assert metrics["negative_labels"] == 1
    assert metrics["negative_correct_count"] == 1
    assert metrics["family_top1_accuracy"] == 0.75
    assert metrics["family_top3_recall"] == 1.0
    assert metrics["selected_family_accuracy"] == 0.75
    assert metrics["species_top1_accuracy"] == 0.25
    assert metrics["species_top5_recall"] == 0.5
    assert metrics["species_top20_recall"] == 0.75
    assert metrics["species_mrr"] == pytest.approx(0.4)


def test_synthetic_evaluation_fixture_review_queue_contains_expected_rows() -> None:
    fixture = build_synthetic_evaluation_fixture()

    queue = build_hierarchical_review_queue(object_evidence=fixture.object_scores)

    assert queue.height == 4
    rows = {row["flickr_photo_id"]: row for row in queue.to_dicts()}
    assert rows["photo-wrong"]["review_priority"] == 90
    assert "metadata_species_conflict" in rows["photo-wrong"]["review_reason"]
    assert rows["photo-top5"]["review_priority"] == 70
    assert rows["photo-top1"]["review_priority"] == 10


def test_write_synthetic_evaluation_fixture_writes_parquet_files(tmp_path) -> None:
    paths = write_synthetic_evaluation_fixture(tmp_path)

    assert sorted(paths) == [
        "classification_taxa",
        "object_detections",
        "object_scores",
        "reviewed_labels",
    ]
    assert pl.read_parquet(paths["classification_taxa"]).height == 33
    assert pl.read_parquet(paths["object_detections"]).height == 5
    assert pl.read_parquet(paths["object_scores"]).height == 4
    assert pl.read_parquet(paths["reviewed_labels"]).height == 5
