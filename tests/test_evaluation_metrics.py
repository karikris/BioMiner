from __future__ import annotations

import json

import polars as pl
import pytest

from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION, TARGET_SCOPE_OBJECT_SCREENING
from biominer.evaluation.metrics import evaluate_hierarchical_predictions


def test_evaluate_hierarchical_predictions_counts_species_top1_correct() -> None:
    metrics = evaluate_hierarchical_predictions(
        object_scores=pl.DataFrame([_prediction(species_top1="Papilio demoleus")]),
        reviewed_labels=pl.DataFrame([_label(scientific_name="Papilio demoleus", accepted_taxon_key="gbif:100")]),
    )

    assert metrics["evaluated_objects"] == 1
    assert metrics["butterfly_positive_labels"] == 1
    assert metrics["family_top1_accuracy"] == 1.0
    assert metrics["family_top3_recall"] == 1.0
    assert metrics["selected_family_accuracy"] == 1.0
    assert metrics["species_top1_accuracy"] == 1.0
    assert metrics["species_top5_recall"] == 1.0
    assert metrics["species_top20_recall"] == 1.0
    assert metrics["species_mrr"] == 1.0


def test_evaluate_hierarchical_predictions_counts_species_top5_when_top1_wrong() -> None:
    metrics = evaluate_hierarchical_predictions(
        object_scores=pl.DataFrame(
            [
                _prediction(
                    species_top1="Papilio machaon",
                    species_top1_key="gbif:200",
                    species_top5_json=True,
                    species_top20_json=True,
                    species_top5=["Papilio machaon", "Papilio demoleus"],
                    species_top5_keys=["gbif:200", "gbif:100"],
                    species_top20=["Papilio machaon", "Papilio demoleus"],
                    species_top20_keys=["gbif:200", "gbif:100"],
                )
            ]
        ),
        reviewed_labels=pl.DataFrame([_label(scientific_name="Papilio demoleus", accepted_taxon_key="gbif:100")]),
    )

    assert metrics["species_top1_accuracy"] == 0.0
    assert metrics["species_top5_recall"] == 1.0
    assert metrics["species_top20_recall"] == 1.0
    assert metrics["species_mrr"] == pytest.approx(0.5)


def test_evaluate_hierarchical_predictions_counts_species_top20_when_top5_wrong() -> None:
    metrics = evaluate_hierarchical_predictions(
        object_scores=pl.DataFrame(
            [
                _prediction(
                    species_top1="Papilio machaon",
                    species_top1_key="gbif:200",
                    species_top5=["Papilio machaon"],
                    species_top5_keys=["gbif:200"],
                    species_top20=["Papilio machaon", "Papilio demoleus"],
                    species_top20_keys=["gbif:200", "gbif:100"],
                )
            ]
        ),
        reviewed_labels=pl.DataFrame([_label(scientific_name="Papilio demoleus", accepted_taxon_key="gbif:100")]),
    )

    assert metrics["species_top1_accuracy"] == 0.0
    assert metrics["species_top5_recall"] == 0.0
    assert metrics["species_top20_recall"] == 1.0
    assert metrics["species_mrr"] == pytest.approx(0.5)


def test_evaluate_hierarchical_predictions_counts_family_top3_when_top1_wrong() -> None:
    metrics = evaluate_hierarchical_predictions(
        object_scores=pl.DataFrame(
            [
                _prediction(
                    family_top3=["Nymphalidae", "Papilionidae", "Pieridae"],
                    family_top3_keys=["gbif:7017", "gbif:9417", "gbif:5481"],
                    selected_family="Nymphalidae",
                    selected_family_key="gbif:7017",
                )
            ]
        ),
        reviewed_labels=pl.DataFrame([_label(scientific_name="Papilio demoleus", accepted_taxon_key="gbif:100")]),
    )

    assert metrics["family_top1_accuracy"] == 0.0
    assert metrics["family_top3_recall"] == 1.0
    assert metrics["selected_family_accuracy"] == 0.0


def test_path_cascade_family_metrics_use_rank_and_selected_names_not_overlay_ids() -> None:
    prediction = _prediction(
        family_top3=["Nymphalidae", "Papilionidae", "Pieridae"],
        family_top3_keys=["fixture:family:nymph", "fixture:family:papilio"],
        selected_family="Nymphalidae",
        selected_family_key="fixture:family:nymph",
    )
    prediction.update(
        {
            "classifier_schema_version": "butterfly-cascade-output-v1.0.0",
            "family_top1": "Nymphalidae",
            "family_top3_node_ids": [
                "fixture:family:nymph",
                "fixture:family:papilio",
                "fixture:family:pieris",
            ],
        }
    )
    metrics = evaluate_hierarchical_predictions(
        object_scores=pl.DataFrame([prediction]),
        reviewed_labels=pl.DataFrame([_label()]),
    )

    assert metrics["family_top1_accuracy"] == 0.0
    assert metrics["family_top3_recall"] == 1.0
    assert metrics["selected_family_accuracy"] == 0.0


def test_evaluate_hierarchical_predictions_handles_missing_prediction() -> None:
    metrics = evaluate_hierarchical_predictions(
        object_scores=pl.DataFrame([]),
        reviewed_labels=pl.DataFrame([_label(scientific_name="Papilio demoleus", accepted_taxon_key="gbif:100")]),
    )

    assert metrics["missing_prediction_count"] == 1
    assert metrics["false_negative_butterfly_count"] == 1
    assert metrics["family_top1_accuracy"] == 0.0
    assert metrics["species_top20_recall"] == 0.0


def test_evaluate_hierarchical_predictions_handles_negative_labels() -> None:
    metrics = evaluate_hierarchical_predictions(
        object_scores=pl.DataFrame([_prediction(flickr_photo_id="n2", detection_id="dn2")]),
        reviewed_labels=pl.DataFrame(
            [
                _label(flickr_photo_id="n1", detection_id="dn1", is_butterfly=False, label_level="negative"),
                _label(flickr_photo_id="n2", detection_id="dn2", is_butterfly=False, label_level="negative"),
            ]
        ),
    )

    assert metrics["negative_labels"] == 2
    assert metrics["negative_correct_count"] == 1
    assert metrics["false_positive_butterfly_count"] == 1
    assert metrics["false_negative_butterfly_count"] == 0


def test_evaluate_hierarchical_predictions_skips_target_scope_rows() -> None:
    metrics = evaluate_hierarchical_predictions(
        object_scores=pl.DataFrame(
            [
                _prediction(flickr_photo_id="h1", detection_id="dh1"),
                _prediction(
                    flickr_photo_id="t1",
                    detection_id="dt1",
                    classification_mode=TARGET_SCOPE_OBJECT_SCREENING,
                ),
            ]
        ),
        reviewed_labels=pl.DataFrame(
            [
                _label(flickr_photo_id="h1", detection_id="dh1"),
                _label(flickr_photo_id="t1", detection_id="dt1"),
            ]
        ),
    )

    assert metrics["hierarchical_prediction_count"] == 1
    assert metrics["target_scope_prediction_count"] == 1
    assert metrics["non_hierarchical_prediction_count"] == 1
    assert metrics["matched_hierarchical_objects"] == 1
    assert metrics["missing_prediction_count"] == 1
    assert metrics["false_negative_butterfly_count"] == 1
    assert metrics["species_top1_accuracy"] == 1.0


def _prediction(
    *,
    source: str = "flickr",
    flickr_photo_id: str = "1",
    detection_id: str = "d1",
    classification_mode: str = HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    family_top3: list[str] | None = None,
    family_top3_keys: list[str] | None = None,
    selected_family: str = "Papilionidae",
    selected_family_key: str = "gbif:9417",
    species_top1: str = "Papilio demoleus",
    species_top1_key: str = "gbif:100",
    species_top5: list[str] | None = None,
    species_top5_keys: list[str] | None = None,
    species_top20: list[str] | None = None,
    species_top20_keys: list[str] | None = None,
    species_top5_json: bool = False,
    species_top20_json: bool = False,
) -> dict[str, object]:
    top5 = species_top5 or [species_top1, "Papilio machaon"]
    top5_keys = species_top5_keys or [species_top1_key, "gbif:200"]
    top20 = species_top20 or top5
    top20_keys = species_top20_keys or top5_keys
    return {
        "source": source,
        "flickr_photo_id": flickr_photo_id,
        "detection_id": detection_id,
        "classification_mode": classification_mode,
        "family_top3": family_top3 or ["Papilionidae", "Nymphalidae", "Pieridae"],
        "family_top3_accepted_taxon_keys": family_top3_keys or ["gbif:9417", "gbif:7017", "gbif:5481"],
        "selected_family": selected_family,
        "selected_family_key": selected_family_key,
        "species_top1_scientific_name": species_top1,
        "species_top1_accepted_taxon_key": species_top1_key,
        "species_top5": json.dumps(top5) if species_top5_json else top5,
        "species_top5_accepted_taxon_keys": json.dumps(top5_keys) if species_top5_json else top5_keys,
        "species_top20": json.dumps(top20) if species_top20_json else top20,
        "species_top20_accepted_taxon_keys": json.dumps(top20_keys) if species_top20_json else top20_keys,
    }


def _label(
    *,
    source: str = "flickr",
    flickr_photo_id: str = "1",
    detection_id: str = "d1",
    label_level: str = "species",
    is_butterfly: bool = True,
    accepted_taxon_key: str = "gbif:100",
    scientific_name: str = "Papilio demoleus",
    family_key: str = "gbif:9417",
    family: str = "Papilionidae",
) -> dict[str, object]:
    return {
        "source": source,
        "flickr_photo_id": flickr_photo_id,
        "detection_id": detection_id,
        "crop_hash": f"sha256:{detection_id}",
        "label_level": label_level,
        "is_butterfly": is_butterfly,
        "accepted_taxon_key": accepted_taxon_key if is_butterfly else "",
        "scientific_name": scientific_name if is_butterfly else "",
        "family_key": family_key if is_butterfly else "",
        "family": family if is_butterfly else "",
        "genus_key": "gbif:90" if is_butterfly else "",
        "genus": "Papilio" if is_butterfly else "",
        "label_source": "fixture",
        "reviewer_id": "reviewer-a",
        "reviewed_at": "2026-07-10T00:00:00Z",
        "review_confidence": "high",
        "review_notes": "synthetic",
    }
