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
    ReviewEvidenceObservation,
    ReviewGroupingProfile,
    calculate_review_requirements,
    clopper_pearson_lower_bound,
    update_review_milestones,
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


def test_exact_one_sided_binomial_reference_values() -> None:
    assert clopper_pearson_lower_bound(9, 10, confidence_level=0.95) == pytest.approx(
        0.6058366975634952
    )
    assert clopper_pearson_lower_bound(18, 20, confidence_level=0.95) == pytest.approx(
        0.7173814751141391
    )


def test_all_success_one_look_requires_59_not_a_universal_count() -> None:
    policy = ReviewEvidencePolicy(
        milestone_information_fractions=(1.0,),
        maximum_review_budget=200,
    )
    plan = calculate_review_requirements(policy, anticipated_error_rate=0.0)

    assert clopper_pearson_lower_bound(58, 58, confidence_level=0.95) < 0.95
    assert clopper_pearson_lower_bound(59, 59, confidence_level=0.95) >= 0.95
    assert plan.independent_effective_decisive_reviews == 59
    assert plan.required_nominal_decisive_reviews == 59
    assert plan.assumed_successes_at_effective_requirement == 59
    assert plan.assumed_errors_at_effective_requirement == 0
    assert plan.status == "planned"


def test_observed_error_assumption_changes_the_required_count() -> None:
    policy = ReviewEvidencePolicy(
        target_precision=0.97,
        lower_bound_objective=0.95,
        milestone_information_fractions=(1.0,),
        maximum_review_budget=2_000,
    )
    all_success = calculate_review_requirements(policy, anticipated_error_rate=0.0)
    errors_expected = calculate_review_requirements(policy, anticipated_error_rate=0.02)

    assert errors_expected.independent_effective_decisive_reviews is not None
    assert all_success.independent_effective_decisive_reviews is not None
    assert (
        errors_expected.independent_effective_decisive_reviews
        > all_success.independent_effective_decisive_reviews
    )
    assert errors_expected.assumed_errors_at_effective_requirement > 0


def test_weighted_and_clustered_design_inflates_nominal_reviews() -> None:
    policy = ReviewEvidencePolicy(
        milestone_information_fractions=(1.0,),
        maximum_review_budget=1_000,
    )
    grouping = ReviewGroupingProfile(
        owner_cluster_sizes=(3, 3, 2),
        owner_intraclass_correlation=0.20,
        duplicate_cluster_sizes=(2, 2, 1, 1),
        duplicate_intraclass_correlation=0.30,
    )
    plan = calculate_review_requirements(
        policy,
        anticipated_error_rate=0.0,
        sampling_weights=(1.0, 1.0, 4.0, 4.0),
        external_design_effect=1.20,
        grouping_profile=grouping,
    )

    assert plan.weight_design_effect > 1.0
    assert plan.grouping_design_effect > 1.0
    assert plan.combined_design_effect > 1.0
    assert plan.statistical_nominal_decisive_reviews > 59
    assert plan.interval_semantics.startswith("exact_independent_binomial")


def test_required_strata_floor_can_determine_the_review_requirement() -> None:
    policy = ReviewEvidencePolicy(
        target_precision=0.80,
        lower_bound_objective=0.75,
        minimum_represented_strata=10,
        minimum_decisive_reviews_per_stratum=10,
        maximum_review_budget=500,
        milestone_information_fractions=(1.0,),
    )
    plan = calculate_review_requirements(
        policy,
        anticipated_error_rate=0.0,
        required_stratum_count=10,
    )

    assert plan.stratum_minimum_decisive_reviews == 100
    assert plan.statistical_nominal_decisive_reviews < 100
    assert plan.required_nominal_decisive_reviews == 100


def test_planner_reports_infeasible_objective_and_budget() -> None:
    objective_infeasible = calculate_review_requirements(
        ReviewEvidencePolicy(milestone_information_fractions=(1.0,)),
        anticipated_error_rate=0.06,
    )
    assert objective_infeasible.status == "objective_infeasible"
    assert objective_infeasible.required_nominal_decisive_reviews is None

    budget_insufficient = calculate_review_requirements(
        ReviewEvidencePolicy(
            maximum_review_budget=60,
            milestone_information_fractions=(1.0,),
        ),
        anticipated_error_rate=0.0,
        external_design_effect=2.0,
    )
    assert budget_insufficient.status == "budget_insufficient"
    assert budget_insufficient.required_nominal_decisive_reviews == 118
    assert budget_insufficient.recommended_review_count == 60


def test_planner_subtracts_existing_decisive_reviews_from_recommendation() -> None:
    plan = calculate_review_requirements(
        ReviewEvidencePolicy(
            maximum_review_budget=200,
            milestone_information_fractions=(1.0,),
        ),
        anticipated_error_rate=0.0,
        observed_decisive_reviews=20,
    )

    assert plan.recommended_review_count == 59
    assert plan.additional_decisive_reviews_needed == 39


def _sha(character: str) -> str:
    return f"sha256:{character * 64}"


def _review(
    sequence: int,
    *,
    supported: bool | None = True,
    purpose: str = "representative_audit",
    eligible: bool = True,
    weight: float | None = 1.0,
    stratum: str = "stratum-1",
    owner: str | None = None,
    duplicate: str | None = None,
) -> ReviewEvidenceObservation:
    digit = format(sequence % 16, "x")
    return ReviewEvidenceObservation(
        review_sequence=sequence,
        review_unit_id=f"review-{sequence}",
        source_record_hash=_sha(digit),
        review_decision_fingerprint=_sha(format((sequence + 1) % 16, "x")),
        stratum_id=stratum,
        reviewer_group_id=f"reviewer-{sequence}",
        owner_group_id=owner or f"owner-{sequence}",
        duplicate_group_id=duplicate or f"duplicate-{sequence}",
        observation_group_id=f"observation-{sequence}",
        sampling_purpose=purpose,
        representative_estimation_eligible=eligible,
        human_supported=supported,
        sampling_weight=weight,
    )


def _one_look_plan():
    policy = ReviewEvidencePolicy(
        maximum_review_budget=200,
        milestone_information_fractions=(1.0,),
    )
    plan = calculate_review_requirements(policy, anticipated_error_rate=0.0)
    return policy, plan


def test_milestone_stops_only_when_preregistered_59_record_bound_passes() -> None:
    policy, plan = _one_look_plan()
    before = update_review_milestones(
        policy,
        plan,
        [_review(sequence) for sequence in range(1, 59)],
    )
    reached = update_review_milestones(
        policy,
        plan,
        [_review(sequence) for sequence in range(1, 60)],
    )

    assert before.decision == "continue_to_next_milestone"
    assert before.additional_decisive_reviews_to_next_milestone == 1
    assert not before.stop_authorized
    assert reached.decision == "stop_objective_met"
    assert reached.stop_authorized
    assert reached.representative_support_authorized
    assert not reached.occurrence_release_authorized
    assert reached.milestones[0].precision_lower_bound >= 0.95
    assert reached.milestones[0].interval_semantics == (
        "exact_one_sided_clopper_pearson"
    )


def test_failed_final_milestone_requests_replanning_from_observed_error() -> None:
    policy, plan = _one_look_plan()
    reviews = [_review(sequence) for sequence in range(1, 60)]
    reviews[-1] = _review(59, supported=False)

    update = update_review_milestones(policy, plan, reviews)

    assert update.eligible_error_reviews == 1
    assert update.observed_weighted_error_rate == pytest.approx(1 / 59)
    assert update.decision == "replan_from_observed_error"
    assert not update.stop_authorized
    assert not update.milestones[0].lower_bound_objective_met


def test_targeted_and_nondecisive_events_do_not_advance_audit_milestone() -> None:
    policy, plan = _one_look_plan()
    reviews = [_review(sequence) for sequence in range(1, 58)]
    reviews.extend(
        [
            _review(
                58,
                purpose="targeted_failure_discovery",
                eligible=False,
                weight=None,
            ),
            _review(59, supported=None),
            _review(
                60,
                purpose="occurrence_release_review",
                eligible=False,
                weight=None,
            ),
        ]
    )

    update = update_review_milestones(policy, plan, reviews)

    assert update.total_review_events == 60
    assert update.eligible_decisive_reviews == 57
    assert update.targeted_events_excluded == 1
    assert update.nondecisive_events_excluded == 1
    assert update.other_ineligible_events_excluded == 1
    assert update.next_milestone_count == 59


def test_represented_strata_gate_can_block_an_otherwise_passing_bound() -> None:
    policy = ReviewEvidencePolicy(
        minimum_represented_strata=2,
        minimum_decisive_reviews_per_stratum=2,
        maximum_review_budget=200,
        milestone_information_fractions=(1.0,),
    )
    plan = calculate_review_requirements(
        policy,
        anticipated_error_rate=0.0,
        required_stratum_count=2,
    )
    reviews = [_review(sequence, stratum="only-stratum") for sequence in range(1, 60)]

    update = update_review_milestones(policy, plan, reviews)

    assert update.milestones[0].lower_bound_objective_met
    assert not update.milestones[0].represented_strata_met
    assert not update.stop_authorized


def test_weighted_grouped_milestone_uses_conservative_effective_count() -> None:
    policy = ReviewEvidencePolicy(
        maximum_review_budget=500,
        milestone_information_fractions=(1.0,),
    )
    grouping = ReviewGroupingProfile(
        owner_cluster_sizes=(2, 2),
        owner_intraclass_correlation=0.25,
    )
    plan = calculate_review_requirements(
        policy,
        anticipated_error_rate=0.0,
        sampling_weights=(1.0, 3.0),
        grouping_profile=grouping,
    )
    count = plan.recommended_review_count
    reviews = [
        _review(
            sequence,
            weight=1.0 if sequence % 2 else 3.0,
            owner=f"owner-{sequence // 2}",
        )
        for sequence in range(1, count + 1)
    ]

    update = update_review_milestones(
        policy,
        plan,
        reviews,
        grouping_profile=grouping,
    )
    milestone = update.milestones[0]

    assert milestone.combined_design_effect > 1.0
    assert milestone.effective_decisive_reviews < count
    assert milestone.interval_semantics.startswith(
        "conservative_integer_effective_sample"
    )


def test_milestone_update_is_event_order_independent_but_sequence_bound() -> None:
    policy, plan = _one_look_plan()
    reviews = [_review(sequence) for sequence in range(1, 60)]
    first = update_review_milestones(policy, plan, reviews)
    second = update_review_milestones(policy, plan, list(reversed(reviews)))

    assert first == second
    with pytest.raises(ValueError, match="review_sequence must be unique"):
        update_review_milestones(policy, plan, [reviews[0], _review(1)])
