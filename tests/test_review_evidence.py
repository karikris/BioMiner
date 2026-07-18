"""Tests for dynamic human-review evidence planning."""

from __future__ import annotations

import pytest

from biominer.evaluation.review_evidence import (
    COMPLEX_DESIGN_ADJUSTMENT,
    INTERVAL_METHOD,
    MILESTONE_POLICY,
    STOPPING_RULE,
    TARGET_METRIC,
    ReviewEvidencePolicy,
)


def test_review_evidence_policy_exposes_complete_preregistered_design() -> None:
    policy = ReviewEvidencePolicy()

    assert policy.target_metric == TARGET_METRIC
    assert policy.target_precision == 0.95
    assert policy.confidence_level == 0.95
    assert policy.lower_bound_objective == 0.95
    assert policy.minimum_represented_strata == 1
    assert policy.maximum_review_budget == 1_000
    assert policy.milestone_policy == MILESTONE_POLICY
    assert policy.milestone_information_fractions == (0.25, 0.5, 0.75, 1.0)
    assert policy.grouping_design == (
        "duplicate_group_id",
        "observation_group_id",
        "owner_group_id",
        "reviewer_group_id",
    )
    assert policy.interval_method == INTERVAL_METHOD
    assert policy.complex_design_adjustment == COMPLEX_DESIGN_ADJUSTMENT
    assert policy.stopping_rule == STOPPING_RULE
    assert policy.per_milestone_alpha == pytest.approx(0.0125)
    assert policy.per_milestone_confidence_level == pytest.approx(0.9875)
    assert policy.fingerprint.startswith("sha256:")


def test_one_look_policy_preserves_the_requested_confidence_level() -> None:
    policy = ReviewEvidencePolicy(milestone_information_fractions=(1.0,))

    assert policy.per_milestone_confidence_level == pytest.approx(0.95)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"target_precision": 1.0}, "target_precision"),
        (
            {"target_precision": 0.90, "lower_bound_objective": 0.95},
            "cannot exceed",
        ),
        ({"confidence_level": 0.0}, "confidence_level"),
        (
            {
                "minimum_represented_strata": 3,
                "minimum_decisive_reviews_per_stratum": 2,
                "maximum_review_budget": 5,
            },
            "cannot cover",
        ),
        (
            {"milestone_information_fractions": (0.5, 0.25, 1.0)},
            "unique and increasing",
        ),
        ({"milestone_information_fractions": (0.5,)}, "must equal one"),
        ({"grouping_design": ()}, "must not be empty"),
        ({"grouping_design": ("unknown_group",)}, "unsupported grouping"),
        ({"interval_method": "wilson"}, "unsupported interval_method"),
        ({"stopping_rule": "point_estimate_only"}, "unsupported stopping_rule"),
    ],
)
def test_review_evidence_policy_fails_closed_on_invalid_design(
    changes: dict[str, object], match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        ReviewEvidencePolicy(**changes)


def test_policy_fingerprint_changes_with_material_design_choices() -> None:
    baseline = ReviewEvidencePolicy()
    changed = ReviewEvidencePolicy(
        maximum_review_budget=500,
        minimum_represented_strata=5,
        milestone_information_fractions=(0.5, 1.0),
    )

    assert changed.fingerprint != baseline.fingerprint
