from __future__ import annotations

import json

import polars as pl

from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION
from biominer.evaluation.review_queue import build_hierarchical_review_queue
from biominer.evaluation.thresholds import (
    VISION_BUCKET_POLICY_SCHEMA_VERSION,
    VisionBucketPolicy,
    load_vision_bucket_policy,
)


def test_default_vision_bucket_policy_loads_from_config() -> None:
    policy = load_vision_bucket_policy()

    assert policy.schema_version == VISION_BUCKET_POLICY_SCHEMA_VERSION
    assert policy.high_confidence_species_top1_score == 0.70
    assert policy.minimum_species_margin == 0.05
    assert policy.low_confidence_species_score == 0.35


def test_invalid_vision_bucket_policy_threshold_fails_clearly(tmp_path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"minimum_species_margin": 1.5}), encoding="utf-8")

    try:
        load_vision_bucket_policy(path)
    except ValueError as exc:
        assert "minimum_species_margin" in str(exc)
    else:  # pragma: no cover - keeps failure message explicit without pytest.raises match noise.
        raise AssertionError("invalid policy should fail")


def test_review_queue_priority_changes_according_to_policy() -> None:
    frame = pl.DataFrame([_row(species_top1_margin=0.08)])

    default_queue = build_hierarchical_review_queue(object_evidence=frame)
    stricter_queue = build_hierarchical_review_queue(
        object_evidence=frame,
        policy=VisionBucketPolicy(minimum_species_margin=0.10),
    )

    assert default_queue.to_dicts()[0]["review_priority"] == 10
    assert default_queue.to_dicts()[0]["review_reason"] == "clean_confident_prediction"
    assert stricter_queue.to_dicts()[0]["review_priority"] == 70
    assert stricter_queue.to_dicts()[0]["review_reason"] == "low_species_margin"


def test_conservative_default_does_not_treat_uncertain_prediction_as_clean() -> None:
    queue = build_hierarchical_review_queue(
        object_evidence=pl.DataFrame(
            [
                _row(
                    species_top1_score=0.69,
                    species_top1_margin=0.04,
                    species_top5_scores=[0.69, 0.65],
                )
            ]
        )
    )

    row = queue.to_dicts()[0]
    assert row["review_priority"] == 70
    assert row["review_reason"] == "low_species_margin"


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source": "flickr",
        "flickr_photo_id": "policy-photo",
        "detection_id": "policy-det",
        "crop_hash": "sha256:policy",
        "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        "selected_family": "Papilionidae",
        "selected_family_key": "gbif:9417",
        "family_top3": ["Papilionidae", "Pieridae", "Nymphalidae"],
        "family_top3_scores": [0.90, 0.05, 0.02],
        "species_top5": ["Papilio demoleus", "Papilio machaon"],
        "species_top5_scores": [0.80, 0.72],
        "species_top1_scientific_name": "Papilio demoleus",
        "species_top1_accepted_taxon_key": "gbif:100",
        "species_top1_score": 0.80,
        "species_top1_margin": 0.08,
        "detector_label": "butterfly_like",
        "detector_score": 0.82,
        "occurrence_bin": "in_review",
        "bin_reason": "policy_fixture",
    }
    row.update(overrides)
    return row
