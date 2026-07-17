from __future__ import annotations

import polars as pl
import pytest

from biominer.evaluation.reference_escalation import (
    ReferenceEscalationPolicy,
    flag_species_for_reference_review,
)


KEYS = {
    "target_species": "Papilio demoleus",
    "competitor_species": "Papilio polytes",
    "region": "geo:qld",
    "route": "adult_field",
}


def _performance(**overrides: object) -> pl.DataFrame:
    row = {
        **KEYS,
        "metric_status": "complete",
        "reviewed_record_count": 50,
        "precision_ci_lower": 0.9,
        "recall": 0.9,
        "false_positive_rate": 0.05,
        "competitor_confusion_rate": 0.05,
        **overrides,
    }
    return pl.DataFrame([row])


def _reference_evidence(**overrides: object) -> pl.DataFrame:
    row = {
        **KEYS,
        "prototype_dispersion_max": 0.1,
        "high_influence_reference_count": 0,
        "reference_outlier_count": 0,
        "route_imbalance_ratio": 0.0,
        "target_reference_count": 10,
        "reference_identity_conclusion": "not_assessed",
        **overrides,
    }
    return pl.DataFrame([row])


def test_species_within_every_objective_is_not_flagged() -> None:
    decision = flag_species_for_reference_review(
        _performance(),
        _reference_evidence(),
    ).row(0, named=True)

    assert decision["flagged_for_reference_review"] is False
    assert decision["review_scope"] == "none"
    assert decision["flag_reasons"] == []
    assert decision["statistical_identity_conclusion"] == "not_assessed"


def test_every_triggered_reason_persists_observation_operator_and_threshold() -> None:
    decision = flag_species_for_reference_review(
        _performance(
            precision_ci_lower=0.4,
            recall=0.3,
            false_positive_rate=0.5,
            competitor_confusion_rate=0.4,
        ),
        _reference_evidence(
            prototype_dispersion_max=0.8,
            high_influence_reference_count=3,
            route_imbalance_ratio=0.9,
            target_reference_count=4,
        ),
    ).row(0, named=True)

    expected = {
        "precision_lower_bound_below_objective",
        "false_positive_rate_above_objective",
        "target_recall_below_objective",
        "competitor_confusion_above_objective",
        "prototype_dispersion_above_objective",
        "high_influence_outlier_rate_above_objective",
        "route_imbalance_above_objective",
        "reference_support_shortfall",
    }
    assert decision["flagged_for_reference_review"] is True
    assert set(decision["flag_reasons"]) == expected
    assert {rule["reason"] for rule in decision["triggered_rules"]} == expected
    assert all(rule["operator"] in {"<", ">"} for rule in decision["triggered_rules"])
    assert all(rule["threshold"] is not None for rule in decision["triggered_rules"])
    assert decision["policy_fingerprint"].startswith("sha256:")


def test_insufficient_sample_is_flagged_without_fabricating_metrics() -> None:
    decision = flag_species_for_reference_review(
        _performance(
            metric_status="insufficient_sample",
            reviewed_record_count=4,
            precision_ci_lower=None,
            recall=None,
            false_positive_rate=None,
            competitor_confusion_rate=None,
        ),
        _reference_evidence(),
    ).row(0, named=True)

    assert decision["flag_reasons"] == ["insufficient_human_audit_sample"]
    assert decision["triggered_rules"][0]["threshold"] is None
    assert decision["statistical_identity_conclusion"] == "not_assessed"


def test_policy_is_versioned_fingerprinted_and_validated() -> None:
    first = ReferenceEscalationPolicy(policy_version="pilot-v2")
    second = ReferenceEscalationPolicy(
        policy_version="pilot-v2",
        minimum_target_recall=0.8,
    )

    assert first.fingerprint != second.fingerprint
    with pytest.raises(ValueError, match="minimum_target_recall"):
        ReferenceEscalationPolicy(minimum_target_recall=1.1)
