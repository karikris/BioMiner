"""Tests for leakage-safe dynamic-pool evidence calibration."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from biominer.evaluation.dynamic_pool_splits import (
    DynamicPoolEvaluationSplitPolicy,
    ReviewedFlickrSplitItem,
    build_dynamic_pool_evaluation_splits,
    build_reviewed_flickr_components,
)
from biominer.ml.dynamic_pool_calibration import (
    DYNAMIC_POOL_CALIBRATION_PREDICTION_SCHEMA,
    DynamicPoolCalibrationConfig,
    FrozenDynamicPoolEvidenceModel,
    fit_dynamic_pool_evidence_calibrator,
)
from biominer.ml.dynamic_pool_features import (
    DynamicPoolFeatureInput,
    build_dynamic_pool_feature_table,
)
from biominer.ml.dynamic_pool_thresholds import (
    AUDITED_SCREENING_THRESHOLD_AUDIT_SCHEMA,
    LOWER_BOUND_METHOD,
    SCREENING_CANDIDATE_LABEL,
    AuditedScreeningThresholdPolicy,
    select_audited_screening_threshold,
)


def _sha(index: int) -> str:
    return f"sha256:{format(index % 16, 'x') * 64}"


def _split_item(index: int) -> ReviewedFlickrSplitItem:
    no_geo = index % 5 == 0
    return ReviewedFlickrSplitItem(
        item_id=f"cal-item-{index:02d}",
        source_record_hash=_sha(index),
        source_artifact_fingerprint=_sha(index + 1),
        review_decision_fingerprint=_sha(index + 2),
        flickr_photo_id=f"cal-photo-{index}",
        owner_group_id=f"cal-owner-{index}",
        duplicate_group_id=f"cal-duplicate-{index}",
        observation_group_id=f"cal-observation-{index}",
        geographic_cluster_id=None if no_geo else f"cal-geo-{index}",
        no_geo=no_geo,
        source_mirror_group_id=f"cal-mirror-{index}",
        stratum_id=f"cal-stratum-{index % 4}",
        candidate_species_key=f"cal-species-{index % 3}",
        human_supported=index % 2 == 0,
        sampling_weight=1.0 + (index % 4) * 0.15,
    )


def _feature_input(index: int) -> DynamicPoolFeatureInput:
    supported = index % 2 == 0
    no_geo = index % 5 == 0
    local_available = not no_geo and index % 4 != 0
    jitter = ((index % 7) - 3) * 0.01
    global_similarity = (0.76 if supported else 0.34) + jitter
    nearest = global_similarity + 0.05
    top_k = global_similarity + 0.02
    return DynamicPoolFeatureInput(
        item_id=f"cal-item-{index:02d}",
        candidate_species_key=f"cal-species-{index % 3}",
        score_component_fingerprint=_sha(index + 3),
        model_fingerprint=_sha(14),
        reference_evidence_fingerprint=_sha(13),
        query_fingerprint=_sha(index + 4),
        global_prototype_similarity=global_similarity,
        global_nearest_reference_similarity=nearest,
        global_top_k_mean_similarity=top_k,
        raw_competitor_margin=(0.22 if supported else -0.08) + jitter,
        local_evidence_available=local_available,
        local_evidence_unavailable_reason=(
            None if local_available else "no_eligible_local_support"
        ),
        geographic_cluster_id=None if no_geo else f"cal-geo-{index}",
        no_geo=no_geo,
        route="adult_field",
        visual_domain="live_field",
        route_compatible=True,
        quality_flag_count=index % 3,
        global_support_coverage_fraction=0.8,
        global_top_k_coverage_fraction=1.0,
        global_observation_independence_fraction=1.0,
        global_reference_count=4,
        global_configured_reference_count=5,
        global_independent_observation_count=4,
        global_reference_shortfall_count=1,
        local_reference_count=3 if local_available else 0,
        local_configured_reference_count=4,
        local_independent_observation_count=3 if local_available else 0,
        local_reference_shortfall_count=1 if local_available else 4,
        primary_query_tier=f"T{index % 5 + 1}",
        query_hit_count=index % 9 + 1,
        family_similarity=global_similarity - 0.03,
        family_rank=1 if supported else 2,
        family_margin_to_next_raw=0.18 if supported else 0.04,
        local_prototype_similarity=(
            global_similarity + 0.03 if local_available else None
        ),
        local_nearest_reference_similarity=(
            nearest + 0.02 if local_available else None
        ),
        local_top_k_mean_similarity=(top_k + 0.02 if local_available else None),
        prototype_absolute_disagreement=0.03 if local_available else None,
        nearest_absolute_disagreement=0.02 if local_available else None,
        top_k_absolute_disagreement=0.02 if local_available else None,
        prototype_rank_movement=0 if local_available else None,
        nearest_rank_movement=0 if local_available else None,
        top_k_rank_movement=0 if local_available else None,
        local_support_coverage_fraction=0.75 if local_available else None,
        local_top_k_coverage_fraction=1.0 if local_available else None,
        local_observation_independence_fraction=1.0 if local_available else None,
        subject_area_ratio=0.18 + (index % 5) * 0.03,
        query_text_similarity=0.50 + (0.08 if supported else -0.05),
        query_text_margin=0.06 if supported else -0.02,
    )


def _fixture():
    split_manifest = build_dynamic_pool_evaluation_splits(
        build_reviewed_flickr_components(
            [_split_item(index) for index in range(36)]
        ).register,
        DynamicPoolEvaluationSplitPolicy(
            split_version="dynamic-pool-calibration-fixture-v1",
            random_seed=91,
        ),
    ).manifest
    inputs = [_feature_input(index) for index in range(36)]
    table = build_dynamic_pool_feature_table(inputs, split_manifest).table
    return split_manifest, inputs, table


def _config() -> DynamicPoolCalibrationConfig:
    return DynamicPoolCalibrationConfig(
        route="adult_field",
        random_seed=17,
        maximum_cross_validation_folds=4,
        reliability_bin_count=5,
    )


def test_calibrator_uses_grouped_oof_calibration_and_validation_reliability() -> None:
    _, _, table = _fixture()

    fit = fit_dynamic_pool_evidence_calibrator(table, _config())

    assert isinstance(fit.evidence_model, FrozenDynamicPoolEvidenceModel)
    assert fit.predictions.schema == DYNAMIC_POOL_CALIBRATION_PREDICTION_SCHEMA
    assert set(fit.predictions["evaluation_split"].to_list()) == {
        "calibration",
        "validation",
    }
    assert "final_test" not in fit.predictions["evaluation_split"].to_list()
    assert fit.final_test_prediction_count == 0
    calibration = fit.predictions.filter(
        fit.predictions["evaluation_split"] == "calibration"
    )
    validation = fit.predictions.filter(
        fit.predictions["evaluation_split"] == "validation"
    )
    assert calibration["fold_index"].null_count() == 0
    assert set(calibration["prediction_role"].to_list()) == {"grouped_oof_calibration"}
    assert validation["fold_index"].null_count() == validation.height
    assert set(validation["prediction_role"].to_list()) == {"independent_validation"}
    assert set(fit.probability_calibration.report["dataset_split"].to_list()) == {
        "calibration"
    }
    assert set(fit.validation_diagnostics.reliability["evaluation_set"].to_list()) == {
        "validation"
    }


def test_calibrated_probabilities_and_validation_metrics_are_finite() -> None:
    _, _, table = _fixture()
    fit = fit_dynamic_pool_evidence_calibrator(table, _config())
    probabilities = fit.predictions["calibrated_target_probability"].to_list()

    assert all(0.0 <= value <= 1.0 for value in probabilities)
    assert 0.0 <= fit.validation_metrics["weighted_brier_score"] <= 1.0
    assert fit.validation_metrics["weighted_log_loss"] >= 0.0
    assert 0.0 <= fit.validation_metrics["weighted_expected_calibration_error"] <= 1.0
    assert fit.validation_diagnostics.reliability.height == 5
    assert (
        fit.calibration_sample_count
        == table.filter(table["evaluation_split"] == "calibration").height
    )
    assert (
        fit.validation_sample_count
        == table.filter(table["evaluation_split"] == "validation").height
    )


def test_calibration_fit_is_deterministic() -> None:
    _, _, table = _fixture()

    first = fit_dynamic_pool_evidence_calibrator(table, _config())
    second = fit_dynamic_pool_evidence_calibrator(table, _config())

    assert (
        first.evidence_model.model_fingerprint
        == second.evidence_model.model_fingerprint
    )
    assert (
        first.probability_calibration.calibration_fingerprint
        == second.probability_calibration.calibration_fingerprint
    )
    assert first.predictions.equals(second.predictions)
    assert first.fit_fingerprint == second.fit_fingerprint


def test_final_test_features_do_not_affect_fit_or_validation() -> None:
    manifest, inputs, table = _fixture()
    final_ids = set(
        manifest.filter(manifest["evaluation_split"] == "final_test")[
            "item_id"
        ].to_list()
    )
    changed_inputs = [
        replace(
            item,
            global_prototype_similarity=-0.9,
            global_nearest_reference_similarity=-0.8,
            global_top_k_mean_similarity=-0.85,
            raw_competitor_margin=-1.5,
        )
        if item.item_id in final_ids
        else item
        for item in inputs
    ]
    changed_table = build_dynamic_pool_feature_table(changed_inputs, manifest).table

    first = fit_dynamic_pool_evidence_calibrator(table, _config())
    second = fit_dynamic_pool_evidence_calibrator(changed_table, _config())

    assert table["feature_table_fingerprint"].item(0) != changed_table[
        "feature_table_fingerprint"
    ].item(0)
    assert (
        first.evidence_model.model_fingerprint
        == second.evidence_model.model_fingerprint
    )
    assert (
        first.probability_calibration.calibration_fingerprint
        == second.probability_calibration.calibration_fingerprint
    )
    assert first.predictions.equals(second.predictions)
    assert first.fit_fingerprint == second.fit_fingerprint


def test_validation_features_change_metrics_not_fitted_parameters() -> None:
    manifest, inputs, table = _fixture()
    validation_ids = set(
        manifest.filter(manifest["evaluation_split"] == "validation")[
            "item_id"
        ].to_list()
    )
    changed_inputs = [
        replace(
            item,
            global_prototype_similarity=-item.global_prototype_similarity,
            global_nearest_reference_similarity=-item.global_nearest_reference_similarity,
            global_top_k_mean_similarity=-item.global_top_k_mean_similarity,
            raw_competitor_margin=-item.raw_competitor_margin,
        )
        if item.item_id in validation_ids
        else item
        for item in inputs
    ]
    changed_table = build_dynamic_pool_feature_table(changed_inputs, manifest).table

    first = fit_dynamic_pool_evidence_calibrator(table, _config())
    second = fit_dynamic_pool_evidence_calibrator(changed_table, _config())

    assert (
        first.evidence_model.model_fingerprint
        == second.evidence_model.model_fingerprint
    )
    assert (
        first.probability_calibration.calibration_fingerprint
        == second.probability_calibration.calibration_fingerprint
    )
    assert first.validation_metrics != second.validation_metrics
    assert first.fit_fingerprint != second.fit_fingerprint


def test_runtime_rejects_wrong_feature_width_and_foreign_calibrator() -> None:
    _, _, table = _fixture()
    fit = fit_dynamic_pool_evidence_calibrator(table, _config())

    with pytest.raises(ValueError, match="wrong shape"):
        fit.evidence_model.decision_function(np.zeros((1, 3)))
    foreign = replace(
        fit.probability_calibration.calibrator,
        classifier_fingerprint=_sha(1),
    )
    with pytest.raises(ValueError, match="another evidence model"):
        fit.evidence_model.predict_supported_probability(
            np.zeros((1, len(fit.evidence_model.feature_names))), foreign
        )


def test_route_without_independent_partitions_fails_closed() -> None:
    _, _, table = _fixture()

    with pytest.raises(ValueError, match="no calibration evidence"):
        fit_dynamic_pool_evidence_calibrator(
            table,
            DynamicPoolCalibrationConfig(route="larval"),
        )


def test_threshold_selection_uses_only_independent_validation_bounds() -> None:
    _, _, table = _fixture()
    fit = fit_dynamic_pool_evidence_calibrator(table, _config())
    policy = AuditedScreeningThresholdPolicy(
        minimum_precision_lower_bound=0.50,
        confidence_level=0.95,
        minimum_selected_items=2,
        minimum_selected_components=2,
    )

    selection = select_audited_screening_threshold(fit, policy)

    assert selection.status == "selected"
    assert selection.threshold is not None
    assert selection.audited_precision_lower_bound >= 0.50
    assert selection.threshold_audit.schema == AUDITED_SCREENING_THRESHOLD_AUDIT_SCHEMA
    assert set(selection.threshold_audit["evaluation_split"].to_list()) == {
        "validation"
    }
    assert set(selection.threshold_audit["lower_bound_method"].to_list()) == {
        LOWER_BOUND_METHOD
    }
    assert selection.threshold_audit["selected"].sum() == 1
    selected = selection.threshold_audit.filter(
        selection.threshold_audit["selected"]
    ).row(0, named=True)
    assert selected["threshold"] == selection.threshold
    assert selected["audited_precision_lower_bound"] == min(
        selected["weight_adjusted_lower_bound"],
        selected["component_exact_lower_bound"],
    )
    assert selected["precision_lower_bound_passed"] is True


def test_threshold_maximizes_coverage_among_passing_candidates() -> None:
    _, _, table = _fixture()
    fit = fit_dynamic_pool_evidence_calibrator(table, _config())
    selection = select_audited_screening_threshold(
        fit,
        AuditedScreeningThresholdPolicy(
            minimum_precision_lower_bound=0.30,
            minimum_selected_items=2,
            minimum_selected_components=2,
        ),
    )
    eligible = selection.threshold_audit.filter(
        selection.threshold_audit["threshold_eligible"]
    )

    assert selection.weighted_validation_coverage == pytest.approx(
        eligible["weighted_validation_coverage"].max()
    )


def test_high_precision_policy_fails_closed_with_small_validation_set() -> None:
    _, _, table = _fixture()
    fit = fit_dynamic_pool_evidence_calibrator(table, _config())

    selection = select_audited_screening_threshold(
        fit,
        AuditedScreeningThresholdPolicy(),
    )

    assert selection.status == "infeasible"
    assert selection.status_reason == "insufficient_independent_validation_evidence"
    assert selection.threshold is None
    assert selection.threshold_audit["selected"].sum() == 0

    bound_limited = select_audited_screening_threshold(
        fit,
        AuditedScreeningThresholdPolicy(
            minimum_precision_lower_bound=0.70,
            minimum_selected_items=2,
            minimum_selected_components=2,
        ),
    )
    assert bound_limited.status_reason == "no_threshold_satisfies_precision_lower_bound"


def test_screening_threshold_never_authorizes_occurrence_release() -> None:
    _, _, table = _fixture()
    fit = fit_dynamic_pool_evidence_calibrator(table, _config())
    selection = select_audited_screening_threshold(
        fit,
        AuditedScreeningThresholdPolicy(
            minimum_precision_lower_bound=0.50,
            minimum_selected_items=2,
            minimum_selected_components=2,
        ),
    )

    assert selection.screening_candidate_label == SCREENING_CANDIDATE_LABEL
    assert selection.fit_partition == "calibration"
    assert selection.selection_partition == "validation"
    assert selection.final_test_prediction_count == 0
    assert selection.occurrence_release_authorized is False


def test_threshold_selection_is_deterministic() -> None:
    _, _, table = _fixture()
    fit = fit_dynamic_pool_evidence_calibrator(table, _config())
    policy = AuditedScreeningThresholdPolicy(
        minimum_precision_lower_bound=0.50,
        minimum_selected_items=2,
        minimum_selected_components=2,
    )

    first = select_audited_screening_threshold(fit, policy)
    second = select_audited_screening_threshold(fit, policy)

    assert first.selection_fingerprint == second.selection_fingerprint
    assert first.threshold_audit.equals(second.threshold_audit)
