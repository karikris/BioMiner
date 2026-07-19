"""Tests for fail-closed dynamic-pool remediation triggers."""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from biominer.evaluation.dynamic_pool_escalation import (
    POOLING_ESCALATION_SCHEMA,
    DynamicPoolEscalationPolicy,
    define_pooling_escalations,
    validate_pooling_escalations,
)
from biominer.evaluation.dynamic_pool_quality import (
    DynamicPoolQualityPolicy,
    report_family_pooling_quality,
    report_geographic_pooling_quality,
    report_overall_pooling_quality,
)
from test_dynamic_pool_quality import _observation


def _quality_policy() -> DynamicPoolQualityPolicy:
    return DynamicPoolQualityPolicy(
        minimum_group_items=2,
        minimum_group_components=2,
        minimum_group_effective_sample_size=2.0,
        minimum_metric_denominator_items=1,
        minimum_metric_denominator_components=1,
        minimum_metric_effective_sample_size=1.0,
    )


def _permissive_escalation_policy() -> DynamicPoolEscalationPolicy:
    return DynamicPoolEscalationPolicy(
        minimum_precision_lower_bound=0.0,
        maximum_family_routing_error_rate=1.0,
        maximum_global_local_disagreement_rate=1.0,
        maximum_local_support_insufficiency_rate=1.0,
        maximum_reference_outlier_error_influence_rate=1.0,
        maximum_weighted_brier_score=1.0,
        maximum_weighted_ece=1.0,
        maximum_ood_false_positive_incidence=1.0,
    )


def test_complete_passing_quality_requires_no_action() -> None:
    quality = report_overall_pooling_quality(
        [_observation(index) for index in range(4)],
        policy=_quality_policy(),
    )

    escalations = define_pooling_escalations(
        [quality],
        policy=_permissive_escalation_policy(),
    )

    assert escalations.schema == POOLING_ESCALATION_SCHEMA
    row = escalations.row(0, named=True)
    assert row["escalation_status"] == "no_action"
    assert row["flagged_for_remediation"] is False
    assert row["triggered_rules"] == []
    assert row["human_review_required"] is False


def test_precision_uses_the_audited_lower_bound() -> None:
    quality = report_overall_pooling_quality(
        [_observation(index) for index in range(4)],
        policy=_quality_policy(),
    )
    policy = replace(
        _permissive_escalation_policy(),
        minimum_precision_lower_bound=0.95,
    )

    row = define_pooling_escalations([quality], policy=policy).row(0, named=True)

    rule = next(
        rule
        for rule in row["triggered_rules"]
        if rule["reason"] == "precision_lower_bound_below_objective"
    )
    assert rule["comparison_basis"] == "confidence_interval_lower"
    assert rule["operator"] == "<"
    assert rule["threshold"] == pytest.approx(0.95)
    assert row["additional_flickr_audit_candidate"] is True


@pytest.mark.parametrize(
    ("changes", "policy_field", "reason", "reference_candidate"),
    [
        (
            {"family_routing_correct": False},
            "maximum_family_routing_error_rate",
            "family_misrouting_above_objective",
            True,
        ),
        (
            {"global_local_disagreed": True},
            "maximum_global_local_disagreement_rate",
            "global_local_disagreement_above_objective",
            False,
        ),
        (
            {"local_support_available": False},
            "maximum_local_support_insufficiency_rate",
            "local_support_insufficiency_above_objective",
            True,
        ),
        (
            {
                "human_supported": False,
                "screening_selected": True,
                "reference_outlier_influenced_error": True,
            },
            "maximum_reference_outlier_error_influence_rate",
            "reference_outlier_influence_above_objective",
            True,
        ),
        (
            {
                "human_supported": False,
                "screening_selected": True,
                "out_of_distribution": True,
            },
            "maximum_ood_false_positive_incidence",
            "ood_false_positive_incidence_above_objective",
            False,
        ),
    ],
)
def test_risk_metrics_create_typed_review_triggers(
    changes: dict[str, object],
    policy_field: str,
    reason: str,
    reference_candidate: bool,
) -> None:
    observations = [replace(_observation(index), **changes) for index in range(4)]
    quality = report_overall_pooling_quality(
        observations,
        policy=_quality_policy(),
    )
    policy = replace(_permissive_escalation_policy(), **{policy_field: 0.0})

    row = define_pooling_escalations([quality], policy=policy).row(0, named=True)

    assert reason in row["trigger_reasons"]
    assert row["escalation_status"] == "remediation_review_required"
    assert row["reference_review_candidate"] is reference_candidate
    assert row["automatic_reference_disposition"] is False
    assert row["taxon_misidentification_conclusion"] == "not_assessed"


def test_calibration_diagnostics_create_review_not_release_authority() -> None:
    observations = [
        replace(_observation(index), calibrated_supported_probability=0.0)
        for index in range(4)
    ]
    quality = report_overall_pooling_quality(
        observations,
        policy=_quality_policy(),
    )
    policy = replace(
        _permissive_escalation_policy(),
        maximum_weighted_brier_score=0.1,
        maximum_weighted_ece=0.1,
    )

    row = define_pooling_escalations([quality], policy=policy).row(0, named=True)

    assert "weighted_brier_score_above_objective" in row["trigger_reasons"]
    assert "weighted_ece_above_objective" in row["trigger_reasons"]
    assert row["recommended_actions"] == ["review_calibration"]
    assert row["authorizes_occurrence_release"] is False


def test_geographic_precision_has_an_explicit_geography_trigger() -> None:
    quality = report_geographic_pooling_quality(
        [_observation(index) for index in range(4)],
        policy=_quality_policy(),
    )
    policy = replace(
        _permissive_escalation_policy(),
        minimum_precision_lower_bound=0.95,
    )

    escalations = define_pooling_escalations([quality], policy=policy)

    flagged = escalations.filter(pl.col("flagged_for_remediation"))
    assert flagged.height
    assert all(
        "geography_precision_lower_bound_below_objective" in reasons
        or any(reason.startswith("insufficient_") for reason in reasons)
        for reasons in flagged["trigger_reasons"]
    )
    assert flagged.filter(pl.col("additional_flickr_audit_candidate")).height


def test_insufficient_quality_requests_evidence_without_underperformance_claim() -> (
    None
):
    quality = report_family_pooling_quality([_observation(0)])

    row = define_pooling_escalations([quality]).row(0, named=True)

    assert row["quality_group_status"] == "insufficient_sample"
    assert row["escalation_status"] == "evidence_collection_required"
    assert row["trigger_reasons"] == ["insufficient_representative_evidence"]
    assert row["reference_review_candidate"] is False
    assert row["additional_flickr_audit_candidate"] is True
    assert row["automatic_reference_disposition"] is False


def test_escalation_is_deterministic_and_tamper_evident() -> None:
    observations = [_observation(index) for index in range(8)]
    reports = [
        report_overall_pooling_quality(observations, policy=_quality_policy()),
        report_family_pooling_quality(observations, policy=_quality_policy()),
    ]
    first = define_pooling_escalations(
        reports,
        policy=_permissive_escalation_policy(),
    )
    second = define_pooling_escalations(
        list(reversed(reports)),
        policy=_permissive_escalation_policy(),
    )

    assert first.equals(second)
    tampered = first.with_columns(
        pl.lit("misidentified").alias("taxon_misidentification_conclusion")
    )
    with pytest.raises(ValueError, match="authority contract"):
        validate_pooling_escalations(tampered)
