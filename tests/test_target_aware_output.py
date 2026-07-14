from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from biominer.bioclip.target_aware_output import (
    TARGET_AWARE_CANDIDATE_SCORE_SCHEMA,
    TARGET_AWARE_CANDIDATE_SCORES_SCHEMA_VERSION,
    TARGET_AWARE_OBJECT_SCORE_SCHEMA,
    TARGET_AWARE_OBJECT_SCORES_SCHEMA_VERSION,
    empty_target_aware_candidate_scores_frame,
    empty_target_aware_object_scores_frame,
    target_aware_candidate_scores_frame,
    target_aware_object_scores_frame,
    validate_target_aware_candidate_scores,
    validate_target_aware_object_scores,
    write_target_aware_candidate_scores,
    write_target_aware_object_scores,
)


def _fp(character: str) -> str:
    return "sha256:" + character * 64


def _candidate(
    key: str,
    name: str,
    *,
    target: bool,
    priority: int,
    probability: float,
    rank: int,
) -> dict[str, object]:
    return {
        "accepted_taxon_key": key,
        "scientific_name": name,
        "target_candidate": target,
        "candidate_priority": priority,
        "candidate_reasons": ["configured_target" if target else "regional_occurrence"],
        "text_ensemble_similarity": 0.45 if target else 0.5,
        "reference_centroid_similarity": 0.8 if target else 0.6,
        "nearest_reference_similarity": 0.82 if target else 0.62,
        "top_k_reference_similarity": 0.79 if target else 0.59,
        "reference_top_k": 3,
        "local_prototype_similarity": 0.77 if target else 0.58,
        "global_prototype_similarity": 0.71 if target else 0.57,
        "classifier_task": "binary_target_verifier"
        if target
        else "regional_multiclass",
        "classifier_decision_score": 0.75 if target else 0.25,
        "calibrated_probability": 0.86 if target else probability,
        "regional_classifier_decision_score": 0.7 if target else 0.3,
        "regional_calibrated_probability": probability,
        "regional_rank": rank,
        "global_rank": rank,
        "geographic_scope": "regional",
        "geographic_evidence_score": 0.9 if target else 0.7,
        "occurrence_support": 10 if target else 7,
        "life_stage": "adult",
        "visual_domain": "adult_field",
        "life_stage_compatible": True,
        "visual_domain_compatible": True,
        "best_competitor_accepted_taxon_key": ("gbif:2" if target else "gbif:1"),
        "best_competitor_scientific_name": (
            "Papilio polytes" if target else "Papilio demoleus"
        ),
        "competitor_margin": 0.2 if target else -0.2,
        "nonmatch_score": 0.3 if target else 0.7,
        "abstention_reason": None,
        "prompt_result_fingerprint": _fp("1"),
        "reference_embedding_fingerprint": _fp("2"),
        "reference_prototype_fingerprint": _fp("3"),
        "support_manifest_fingerprint": _fp("4"),
        "classifier_fingerprint": _fp("5"),
        "calibration_fingerprint": _fp("6"),
        "structured_evidence_fingerprint": _fp("7"),
        "candidate_set_fingerprint": _fp("8"),
    }


def _visual_input() -> dict[str, object]:
    return {
        "visual_input_id": "visual:raw",
        "visual_input_kind": "raw_full_image",
        "transformation_fingerprint": _fp("1"),
        "embedding_id": "embedding:raw",
        "embedding_fingerprint": _fp("2"),
        "source_fusion_fingerprint": _fp("3"),
        "target_classifier_decision_score": 0.75,
        "target_regional_classifier_decision_score": 0.7,
        "best_competitor_accepted_taxon_key": "gbif:2",
        "best_competitor_scientific_name": "Papilio polytes",
        "best_competitor_regional_classifier_decision_score": 0.3,
        "regional_competitor_margin": 0.4,
        "target_reference_centroid_similarity": 0.8,
        "best_competitor_reference_centroid_similarity": 0.6,
        "reference_competitor_margin": 0.2,
        "aggregation_weight": 1.0,
        "input_score_fingerprint": _fp("4"),
    }


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source": "flickr",
        "flickr_photo_id": "photo-1",
        "source_record_hash": _fp("a"),
        "scoring_unit_id": "photo-1:adult-field",
        "detection_id": "detection-1",
        "route": "adult_field",
        "geo_cluster_id": "geo:sydney",
        "candidate_set_id": "candidate-set:1",
        "target_accepted_taxon_key": "gbif:1",
        "reference_bank_version": "reference-bank-v1",
        "classifier_version": "classifier-v1",
        "calibration_version": "calibration-v1",
        "prompt_version": "prompt-v1",
        "visual_input_version": "visual-input-v1",
        "classification_mode": "target_aware_few_shot",
        "registry_fingerprint": _fp("0"),
        "geographic_spread_fingerprint": _fp("1"),
        "flickr_cluster_fingerprint": _fp("2"),
        "candidate_set_fingerprint": _fp("8"),
        "reference_bank_fingerprint": _fp("3"),
        "reference_embedding_fingerprint": _fp("4"),
        "prototype_fingerprint": _fp("5"),
        "support_manifest_fingerprint": _fp("6"),
        "model_fingerprint": _fp("7"),
        "preprocessing_fingerprint": _fp("9"),
        "prompt_fingerprint": _fp("a"),
        "classifier_fingerprint": _fp("b"),
        "calibrator_fingerprint": _fp("c"),
        "regional_classifier_fingerprint": _fp("d"),
        "regional_calibrator_fingerprint": _fp("e"),
        "visual_input_fusion_fingerprint": _fp("f"),
        "decision_policy_fingerprint": _fp("0"),
        "scored_at": datetime(2026, 7, 14, 4, 30, tzinfo=UTC),
        "producer_git_sha": "0123456789abcdef0123456789abcdef01234567",
        "target_raw_text_similarity": 0.45,
        "target_reference_centroid_similarity": 0.8,
        "target_nearest_reference_similarity": 0.82,
        "target_top_k_reference_similarity": 0.79,
        "target_local_prototype_similarity": 0.77,
        "target_global_prototype_similarity": 0.71,
        "target_classifier_margin": 0.5,
        "calibrated_target_probability": 0.86,
        "target_regional_rank": 1,
        "target_global_rank": 1,
        "best_competitor_accepted_taxon_key": "gbif:2",
        "best_competitor_scientific_name": "Papilio polytes",
        "calibrated_best_competitor_probability": 0.3,
        "best_competitor_reference_similarity": 0.6,
        "target_competitor_margin": 0.2,
        "best_domain_negative_similarity": 0.1,
        "calibrated_non_target_probability": 0.2,
        "nonmatch_score": 0.3,
        "nonmatch_margin": 0.56,
        "competitor_reason": "regional_occurrence",
        "yoloe_route": "adult_field",
        "detector_score": 0.95,
        "subject_area_ratio": 0.4,
        "mask_coverage": 0.35,
        "visual_input_disagreement": 0.0,
        "reference_coverage": 1.0,
        "geo_evidence": 0.9,
        "ood_score": 0.05,
        "route_compatible": True,
        "reference_coverage_sufficient": True,
        "domain_negative_detected": False,
        "out_of_distribution": False,
        "visual_detail_sufficient": True,
        "no_geo_global_fallback": False,
        "quality_flags": [],
        "visual_input_evidence": [_visual_input()],
        "regional_candidate_evidence": [
            _candidate(
                "gbif:2",
                "Papilio polytes",
                target=False,
                priority=1,
                probability=0.3,
                rank=2,
            ),
            _candidate(
                "gbif:1",
                "Papilio demoleus",
                target=True,
                priority=0,
                probability=0.7,
                rank=1,
            ),
        ],
        "classification_decision": "target_confirmed",
        "abstained": False,
        "abstention_reason": None,
        "review_priority": "none",
        "model_decision_threshold": 0.8,
        "competitor_margin_threshold": 0.1,
        "threshold_provenance": _fp("0"),
    }
    row.update(overrides)
    return row


def test_target_aware_output_exposes_required_versioned_physical_types() -> None:
    assert TARGET_AWARE_OBJECT_SCORES_SCHEMA_VERSION == (
        "target-aware-object-scores-v1.0.0"
    )
    assert TARGET_AWARE_CANDIDATE_SCORES_SCHEMA_VERSION == (
        "target-aware-candidate-scores-v1.0.0"
    )
    assert TARGET_AWARE_OBJECT_SCORE_SCHEMA["calibrated_target_probability"] == (
        pl.Float64
    )
    assert TARGET_AWARE_OBJECT_SCORE_SCHEMA["target_regional_rank"] == pl.UInt32
    assert TARGET_AWARE_OBJECT_SCORE_SCHEMA["visual_input_evidence"] == pl.List(
        pl.Struct(
            {
                "visual_input_id": pl.String,
                "visual_input_kind": pl.String,
                "transformation_fingerprint": pl.String,
                "embedding_id": pl.String,
                "embedding_fingerprint": pl.String,
                "source_fusion_fingerprint": pl.String,
                "target_classifier_decision_score": pl.Float32,
                "target_regional_classifier_decision_score": pl.Float32,
                "best_competitor_accepted_taxon_key": pl.String,
                "best_competitor_scientific_name": pl.String,
                "best_competitor_regional_classifier_decision_score": pl.Float32,
                "regional_competitor_margin": pl.Float32,
                "target_reference_centroid_similarity": pl.Float32,
                "best_competitor_reference_centroid_similarity": pl.Float32,
                "reference_competitor_margin": pl.Float32,
                "aggregation_weight": pl.Float32,
                "input_score_fingerprint": pl.String,
            }
        )
    )


def test_empty_frames_preserve_exact_schemas() -> None:
    objects = empty_target_aware_object_scores_frame()
    candidates = empty_target_aware_candidate_scores_frame()

    assert dict(objects.schema) == TARGET_AWARE_OBJECT_SCORE_SCHEMA
    assert dict(candidates.schema) == TARGET_AWARE_CANDIDATE_SCORE_SCHEMA
    validate_target_aware_object_scores(objects)
    validate_target_aware_candidate_scores(candidates, object_scores=objects)


def test_frame_derives_stable_ids_and_complete_candidate_projection() -> None:
    first = target_aware_object_scores_frame([_row()])
    second = target_aware_object_scores_frame([_row()])
    row = first.row(0, named=True)

    assert first.equals(second)
    assert row["schema_version"] == TARGET_AWARE_OBJECT_SCORES_SCHEMA_VERSION
    assert row["target_score_id"].startswith("target-score:")
    assert row["target_score_fingerprint"].startswith("sha256:")
    assert [
        item["candidate_priority"] for item in row["regional_candidate_evidence"]
    ] == [
        0,
        1,
    ]
    assert all(
        item["candidate_evidence_fingerprint"].startswith("sha256:")
        for item in row["regional_candidate_evidence"]
    )

    candidates = target_aware_candidate_scores_frame(first)
    assert candidates.height == 2
    assert candidates["accepted_taxon_key"].to_list() == ["gbif:1", "gbif:2"]
    assert candidates["schema_version"].unique().to_list() == [
        TARGET_AWARE_CANDIDATE_SCORES_SCHEMA_VERSION
    ]
    assert candidates["target_score_id"].unique().to_list() == [row["target_score_id"]]
    validate_target_aware_candidate_scores(candidates, object_scores=first)


def test_output_round_trip_preserves_nested_evidence_and_zstd_schema(tmp_path) -> None:
    objects = target_aware_object_scores_frame([_row()])
    candidates = target_aware_candidate_scores_frame(objects)

    object_path = write_target_aware_object_scores(
        objects, tmp_path / "target_aware_object_scores.parquet"
    )
    candidate_path = write_target_aware_candidate_scores(
        candidates,
        tmp_path / "target_aware_candidate_scores.parquet",
        object_scores=objects,
    )

    restored_objects = pl.read_parquet(object_path)
    restored_candidates = pl.read_parquet(candidate_path)
    assert dict(restored_objects.schema) == TARGET_AWARE_OBJECT_SCORE_SCHEMA
    assert dict(restored_candidates.schema) == TARGET_AWARE_CANDIDATE_SCORE_SCHEMA
    assert restored_objects.to_dicts() == objects.to_dicts()
    assert restored_candidates.to_dicts() == candidates.to_dicts()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("classifier_fingerprint", "unsupported", "classifier_fingerprint"),
        ("calibrated_target_probability", 1.1, "calibrated_target_probability"),
        ("reference_coverage", -0.1, "reference_coverage"),
        ("producer_git_sha", "deadbeef", "producer_git_sha"),
    ],
)
def test_output_rejects_invalid_provenance_and_numeric_ranges(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        target_aware_object_scores_frame([_row(**{field: value})])


def test_output_rejects_incomplete_candidate_union_and_visual_weights() -> None:
    competitor_only = [
        _candidate(
            "gbif:2",
            "Papilio polytes",
            target=False,
            priority=0,
            probability=1.0,
            rank=1,
        )
    ]
    with pytest.raises(ValueError, match="exactly one target"):
        target_aware_object_scores_frame(
            [_row(regional_candidate_evidence=competitor_only)]
        )

    visual = _visual_input()
    visual["aggregation_weight"] = 0.5
    with pytest.raises(ValueError, match="aggregation weights"):
        target_aware_object_scores_frame([_row(visual_input_evidence=[visual])])


def test_target_confirmation_fails_closed_on_quality_or_provenance_drift() -> None:
    with pytest.raises(ValueError, match="target_confirmed.*compatible route"):
        target_aware_object_scores_frame([_row(route_compatible=False)])

    with pytest.raises(ValueError, match="threshold provenance"):
        target_aware_object_scores_frame([_row(threshold_provenance=_fp("1"))])

    candidates = _row()["regional_candidate_evidence"]
    assert isinstance(candidates, list)
    target = next(item for item in candidates if item["target_candidate"])
    target["geographic_scope"] = "nonregional"
    target["geographic_evidence_score"] = None
    target["occurrence_support"] = 0
    with pytest.raises(ValueError, match="regional geographic evidence"):
        target_aware_object_scores_frame([_row(regional_candidate_evidence=candidates)])


def test_validator_rejects_physical_schema_and_derived_identity_tampering() -> None:
    frame = target_aware_object_scores_frame([_row()])
    with pytest.raises(ValueError, match="physical schema"):
        validate_target_aware_object_scores(
            frame.with_columns(pl.col("target_regional_rank").cast(pl.UInt64))
        )

    tampered = frame.with_columns(
        pl.lit("target-score:" + "0" * 64).alias("target_score_id")
    )
    with pytest.raises(ValueError, match="target_score_id"):
        validate_target_aware_object_scores(tampered)

    with pytest.raises(ValueError, match="unknown fields"):
        target_aware_object_scores_frame([_row(legacy_species_score=0.9)])
