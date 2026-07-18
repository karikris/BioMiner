"""Dynamic human-review evidence planning and stopping contracts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import math

from biominer.common.semantic_hash import canonical_semantic_fingerprint


REVIEW_EVIDENCE_POLICY_SCHEMA_VERSION = "review-evidence-policy-v1.0.0"
REVIEW_EVIDENCE_POLICY_FILE = "review_evidence_policy.json"
REVIEW_REQUIREMENT_PLAN_SCHEMA_VERSION = "review-requirement-plan-v1.0.0"
REVIEW_REQUIREMENT_PLAN_FILE = "review_requirement_plan.json"
REVIEW_MILESTONE_UPDATE_SCHEMA_VERSION = "review-milestone-update-v1.0.0"
REVIEW_MILESTONE_UPDATE_FILE = "review_milestone_update.json"

TARGET_METRIC = "precision_of_selected_occurrence_candidates"
INTERVAL_METHOD = "one_sided_clopper_pearson"
MILESTONE_POLICY = "prespecified_bonferroni_information_fractions"
STOPPING_RULE = "global_lower_bound_and_required_strata_at_milestone"
COMPLEX_DESIGN_ADJUSTMENT = "kish_effective_sample_size_times_design_effect"

GROUPING_DIMENSIONS = frozenset(
    {
        "reviewer_group_id",
        "owner_group_id",
        "duplicate_group_id",
        "observation_group_id",
    }
)


@dataclass(frozen=True, slots=True)
class ReviewEvidencePolicy:
    """Preregistered precision objective and sequential review design.

    The exact binomial interval applies to an independent unweighted design.
    Weighted and grouped planning is explicitly adjusted through effective
    sample size and a supplied design effect; it is not called exact binomial
    evidence.
    """

    schema_version: str = REVIEW_EVIDENCE_POLICY_SCHEMA_VERSION
    target_precision: float = 0.95
    confidence_level: float = 0.95
    lower_bound_objective: float = 0.95
    minimum_represented_strata: int = 1
    minimum_decisive_reviews_per_stratum: int = 1
    maximum_review_budget: int = 1_000
    milestone_policy: str = MILESTONE_POLICY
    milestone_information_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
    grouping_design: tuple[str, ...] = (
        "reviewer_group_id",
        "owner_group_id",
        "duplicate_group_id",
        "observation_group_id",
    )
    interval_method: str = INTERVAL_METHOD
    stopping_rule: str = STOPPING_RULE
    target_metric: str = TARGET_METRIC
    complex_design_adjustment: str = COMPLEX_DESIGN_ADJUSTMENT

    def __post_init__(self) -> None:
        for field, expected in (
            ("schema_version", REVIEW_EVIDENCE_POLICY_SCHEMA_VERSION),
            ("milestone_policy", MILESTONE_POLICY),
            ("interval_method", INTERVAL_METHOD),
            ("stopping_rule", STOPPING_RULE),
            ("target_metric", TARGET_METRIC),
            ("complex_design_adjustment", COMPLEX_DESIGN_ADJUSTMENT),
        ):
            value = _required_text(getattr(self, field), field=field)
            if value != expected:
                raise ValueError(f"unsupported {field}: {value}")
            object.__setattr__(self, field, value)
        for field in (
            "target_precision",
            "confidence_level",
            "lower_bound_objective",
        ):
            value = _open_probability(getattr(self, field), field=field)
            object.__setattr__(self, field, value)
        if self.lower_bound_objective > self.target_precision:
            raise ValueError("lower_bound_objective cannot exceed target_precision")
        for field in (
            "minimum_represented_strata",
            "minimum_decisive_reviews_per_stratum",
            "maximum_review_budget",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        minimum_required = (
            self.minimum_represented_strata * self.minimum_decisive_reviews_per_stratum
        )
        if minimum_required > self.maximum_review_budget:
            raise ValueError(
                "maximum_review_budget cannot cover required stratum representation"
            )
        fractions = _milestone_fractions(self.milestone_information_fractions)
        object.__setattr__(self, "milestone_information_fractions", fractions)
        groups = tuple(
            sorted(
                {
                    _required_text(value, field="grouping_design")
                    for value in self.grouping_design
                }
            )
        )
        if not groups:
            raise ValueError("grouping_design must not be empty")
        unsupported = sorted(set(groups) - GROUPING_DIMENSIONS)
        if unsupported:
            raise ValueError(f"unsupported grouping dimensions: {unsupported}")
        object.__setattr__(self, "grouping_design", groups)

    @property
    def familywise_alpha(self) -> float:
        return 1.0 - self.confidence_level

    @property
    def per_milestone_alpha(self) -> float:
        return self.familywise_alpha / len(self.milestone_information_fractions)

    @property
    def per_milestone_confidence_level(self) -> float:
        return 1.0 - self.per_milestone_alpha

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": self.schema_version,
                "target_precision": self.target_precision,
                "confidence_level": self.confidence_level,
                "lower_bound_objective": self.lower_bound_objective,
                "minimum_represented_strata": self.minimum_represented_strata,
                "minimum_decisive_reviews_per_stratum": (
                    self.minimum_decisive_reviews_per_stratum
                ),
                "maximum_review_budget": self.maximum_review_budget,
                "milestone_policy": self.milestone_policy,
                "milestone_information_fractions": (
                    self.milestone_information_fractions
                ),
                "familywise_alpha": self.familywise_alpha,
                "per_milestone_alpha": self.per_milestone_alpha,
                "grouping_design": self.grouping_design,
                "interval_method": self.interval_method,
                "stopping_rule": self.stopping_rule,
                "target_metric": self.target_metric,
                "complex_design_adjustment": self.complex_design_adjustment,
                "targeted_failure_evidence": ("ineligible_for_representative_stopping"),
            }
        )


@dataclass(frozen=True, slots=True)
class ReviewGroupingProfile:
    """Observed or anticipated cluster shapes for planning design effect."""

    reviewer_cluster_sizes: tuple[int, ...] = (1,)
    owner_cluster_sizes: tuple[int, ...] = (1,)
    duplicate_cluster_sizes: tuple[int, ...] = (1,)
    observation_cluster_sizes: tuple[int, ...] = (1,)
    reviewer_intraclass_correlation: float = 0.0
    owner_intraclass_correlation: float = 0.0
    duplicate_intraclass_correlation: float = 0.0
    observation_intraclass_correlation: float = 0.0

    def __post_init__(self) -> None:
        for field in (
            "reviewer_cluster_sizes",
            "owner_cluster_sizes",
            "duplicate_cluster_sizes",
            "observation_cluster_sizes",
        ):
            sizes = _cluster_sizes(getattr(self, field), field=field)
            object.__setattr__(self, field, sizes)
        for field in (
            "reviewer_intraclass_correlation",
            "owner_intraclass_correlation",
            "duplicate_intraclass_correlation",
            "observation_intraclass_correlation",
        ):
            value = _closed_open_probability(getattr(self, field), field=field)
            object.__setattr__(self, field, value)

    @property
    def dimension_design_effects(self) -> tuple[tuple[str, float], ...]:
        effects = []
        for dimension in ("reviewer", "owner", "duplicate", "observation"):
            sizes = getattr(self, f"{dimension}_cluster_sizes")
            correlation = getattr(self, f"{dimension}_intraclass_correlation")
            size_biased_mean = sum(size * size for size in sizes) / sum(sizes)
            effect = 1.0 + (size_biased_mean - 1.0) * correlation
            effects.append((f"{dimension}_group_id", effect))
        return tuple(effects)

    @property
    def design_effect(self) -> float:
        # Grouping dimensions overlap, so use the most conservative observed
        # dimension rather than multiplying and double-counting the same rows.
        return max(effect for _, effect in self.dimension_design_effects)

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "reviewer_cluster_sizes": self.reviewer_cluster_sizes,
                "owner_cluster_sizes": self.owner_cluster_sizes,
                "duplicate_cluster_sizes": self.duplicate_cluster_sizes,
                "observation_cluster_sizes": self.observation_cluster_sizes,
                "reviewer_intraclass_correlation": (
                    self.reviewer_intraclass_correlation
                ),
                "owner_intraclass_correlation": self.owner_intraclass_correlation,
                "duplicate_intraclass_correlation": (
                    self.duplicate_intraclass_correlation
                ),
                "observation_intraclass_correlation": (
                    self.observation_intraclass_correlation
                ),
                "combination_rule": "maximum_over_overlapping_dimensions",
            }
        )


@dataclass(frozen=True, slots=True)
class ReviewRequirementPlan:
    """Evidence requirement and every adjustment used to calculate it."""

    schema_version: str
    plan_fingerprint: str
    policy_fingerprint: str
    grouping_profile_fingerprint: str
    status: str
    status_reason: str
    anticipated_error_rate: float
    anticipated_precision: float
    planning_confidence_level: float
    planning_alpha: float
    required_stratum_count: int
    stratum_minimum_decisive_reviews: int
    independent_effective_decisive_reviews: int | None
    assumed_successes_at_effective_requirement: int | None
    assumed_errors_at_effective_requirement: int | None
    lower_bound_at_effective_requirement: float | None
    weight_profile_count: int
    weight_design_effect: float
    grouping_dimension_design_effects: tuple[tuple[str, float], ...]
    grouping_design_effect: float
    external_design_effect: float
    combined_design_effect: float
    statistical_nominal_decisive_reviews: int | None
    required_nominal_decisive_reviews: int | None
    maximum_review_budget: int
    recommended_review_count: int
    observed_decisive_reviews: int
    additional_decisive_reviews_needed: int
    interval_semantics: str


@dataclass(frozen=True, slots=True)
class ReviewEvidenceObservation:
    """One effective, source-bound review outcome for audit planning."""

    review_sequence: int
    review_unit_id: str
    source_record_hash: str
    review_decision_fingerprint: str
    stratum_id: str
    reviewer_group_id: str
    owner_group_id: str
    duplicate_group_id: str
    observation_group_id: str
    sampling_purpose: str
    representative_estimation_eligible: bool
    human_supported: bool | None
    sampling_weight: float | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.review_sequence, int)
            or isinstance(self.review_sequence, bool)
            or self.review_sequence < 1
        ):
            raise ValueError("review_sequence must be a positive integer")
        for field in (
            "review_unit_id",
            "stratum_id",
            "reviewer_group_id",
            "owner_group_id",
            "duplicate_group_id",
            "observation_group_id",
        ):
            object.__setattr__(
                self, field, _required_text(getattr(self, field), field=field)
            )
        for field in ("source_record_hash", "review_decision_fingerprint"):
            object.__setattr__(self, field, _sha256(getattr(self, field), field=field))
        purpose = _required_text(self.sampling_purpose, field="sampling_purpose")
        if purpose not in {
            "representative_audit",
            "targeted_failure_discovery",
            "occurrence_release_review",
        }:
            raise ValueError(f"unsupported sampling_purpose: {purpose}")
        object.__setattr__(self, "sampling_purpose", purpose)
        if not isinstance(self.representative_estimation_eligible, bool):
            raise TypeError("representative_estimation_eligible must be a boolean")
        if self.human_supported is not None and not isinstance(
            self.human_supported, bool
        ):
            raise TypeError("human_supported must be a boolean or null")
        weight = self.sampling_weight
        if weight is not None:
            weight = _positive_float(weight, field="sampling_weight")
            object.__setattr__(self, "sampling_weight", weight)
        if self.representative_estimation_eligible:
            if purpose != "representative_audit":
                raise ValueError(
                    "only representative_audit evidence can be estimation eligible"
                )
            if weight is None:
                raise ValueError(
                    "representative audit evidence requires a sampling weight"
                )
        elif purpose == "representative_audit":
            raise ValueError(
                "representative_audit evidence must be estimation eligible"
            )

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class ReviewMilestoneEvaluation:
    """One preregistered information-fraction evaluation."""

    information_fraction: float
    target_decisive_reviews: int
    status: str
    decisive_reviews_evaluated: int
    supported_reviews: int | None
    error_reviews: int | None
    weighted_precision: float | None
    weighted_error_rate: float | None
    represented_strata: int | None
    required_strata: int
    weight_design_effect: float | None
    grouping_dimension_design_effects: tuple[tuple[str, float], ...]
    grouping_design_effect: float | None
    external_design_effect: float
    combined_design_effect: float | None
    effective_decisive_reviews: int | None
    effective_supported_reviews: int | None
    confidence_level: float
    precision_lower_bound: float | None
    interval_semantics: str | None
    target_precision_met: bool
    lower_bound_objective_met: bool
    represented_strata_met: bool
    stop_authorized: bool


@dataclass(frozen=True, slots=True)
class ReviewMilestoneUpdate:
    """Current evidence state across every preregistered milestone."""

    schema_version: str
    update_fingerprint: str
    policy_fingerprint: str
    requirement_plan_fingerprint: str
    evidence_fingerprint: str
    total_review_events: int
    eligible_decisive_reviews: int
    eligible_supported_reviews: int
    eligible_error_reviews: int
    targeted_events_excluded: int
    nondecisive_events_excluded: int
    other_ineligible_events_excluded: int
    observed_weighted_error_rate: float | None
    milestones: tuple[ReviewMilestoneEvaluation, ...]
    decision: str
    decision_reason: str
    next_milestone_count: int | None
    additional_decisive_reviews_to_next_milestone: int
    stop_authorized: bool
    representative_support_authorized: bool
    occurrence_release_authorized: bool


def clopper_pearson_lower_bound(
    successes: int,
    trials: int,
    *,
    confidence_level: float,
) -> float:
    """Return the exact one-sided binomial lower confidence bound."""

    for value, field in ((successes, "successes"), (trials, "trials")):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{field} must be an integer")
    if trials < 1:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be within [0, trials]")
    confidence = _open_probability(confidence_level, field="confidence_level")
    alpha = 1.0 - confidence
    if successes == 0:
        return 0.0
    if successes == trials:
        return alpha ** (1.0 / trials)
    low = 0.0
    high = successes / trials
    for _ in range(80):
        midpoint = (low + high) / 2.0
        tail = _binomial_upper_tail(successes, trials, midpoint)
        if tail < alpha:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def calculate_review_requirements(
    policy: ReviewEvidencePolicy,
    *,
    anticipated_error_rate: float,
    sampling_weights: Sequence[float] = (),
    external_design_effect: float = 1.0,
    required_stratum_count: int | None = None,
    grouping_profile: ReviewGroupingProfile | None = None,
    observed_decisive_reviews: int = 0,
) -> ReviewRequirementPlan:
    """Calculate evidence dynamically; no universal review count is assumed."""

    if not isinstance(policy, ReviewEvidencePolicy):
        raise TypeError("policy must be a ReviewEvidencePolicy")
    error_rate = _closed_open_probability(
        anticipated_error_rate, field="anticipated_error_rate"
    )
    design_effect = _finite_at_least_one(
        external_design_effect, field="external_design_effect"
    )
    if not isinstance(observed_decisive_reviews, int) or isinstance(
        observed_decisive_reviews, bool
    ):
        raise TypeError("observed_decisive_reviews must be an integer")
    if observed_decisive_reviews < 0:
        raise ValueError("observed_decisive_reviews cannot be negative")
    stratum_count = (
        policy.minimum_represented_strata
        if required_stratum_count is None
        else required_stratum_count
    )
    if not isinstance(stratum_count, int) or isinstance(stratum_count, bool):
        raise TypeError("required_stratum_count must be an integer")
    if stratum_count < policy.minimum_represented_strata:
        raise ValueError("required_stratum_count cannot be below the policy minimum")
    profile = grouping_profile or ReviewGroupingProfile()
    if not isinstance(profile, ReviewGroupingProfile):
        raise TypeError("grouping_profile must be a ReviewGroupingProfile")
    normalized_weights = _sampling_weights(sampling_weights)
    weight_effect = _weight_design_effect(normalized_weights)
    combined_effect = weight_effect * profile.design_effect * design_effect
    anticipated_precision = 1.0 - error_rate
    planning_confidence = policy.per_milestone_confidence_level
    planning_alpha = 1.0 - planning_confidence
    stratum_floor = stratum_count * policy.minimum_decisive_reviews_per_stratum

    effective_requirement: int | None = None
    assumed_successes: int | None = None
    assumed_errors: int | None = None
    lower_bound: float | None = None
    if anticipated_precision < policy.target_precision:
        status = "objective_infeasible"
        reason = "anticipated_precision_below_target_precision"
    elif anticipated_precision <= policy.lower_bound_objective:
        status = "objective_infeasible"
        reason = "anticipated_precision_cannot_exceed_lower_bound_objective"
    else:
        for trials in range(1, policy.maximum_review_budget + 1):
            errors = min(trials, math.ceil(trials * error_rate - 1e-12))
            successes = trials - errors
            candidate_bound = clopper_pearson_lower_bound(
                successes,
                trials,
                confidence_level=planning_confidence,
            )
            if candidate_bound >= policy.lower_bound_objective:
                effective_requirement = trials
                assumed_successes = successes
                assumed_errors = errors
                lower_bound = candidate_bound
                break
        if effective_requirement is None:
            status = "budget_insufficient"
            reason = "effective_requirement_exceeds_maximum_review_budget"
        else:
            status = "planned"
            reason = "requirement_within_budget"

    statistical_nominal = (
        None
        if effective_requirement is None
        else math.ceil(effective_requirement * combined_effect - 1e-12)
    )
    required_nominal = (
        None if statistical_nominal is None else max(statistical_nominal, stratum_floor)
    )
    if required_nominal is not None and required_nominal > policy.maximum_review_budget:
        status = "budget_insufficient"
        reason = "design_adjusted_requirement_exceeds_maximum_review_budget"
    recommended = (
        policy.maximum_review_budget
        if required_nominal is None
        else min(required_nominal, policy.maximum_review_budget)
    )
    additional = max(0, recommended - observed_decisive_reviews)
    semantic_plan = {
        "schema_version": REVIEW_REQUIREMENT_PLAN_SCHEMA_VERSION,
        "policy_fingerprint": policy.fingerprint,
        "grouping_profile_fingerprint": profile.fingerprint,
        "status": status,
        "status_reason": reason,
        "anticipated_error_rate": error_rate,
        "anticipated_precision": anticipated_precision,
        "planning_confidence_level": planning_confidence,
        "planning_alpha": planning_alpha,
        "required_stratum_count": stratum_count,
        "stratum_minimum_decisive_reviews": stratum_floor,
        "independent_effective_decisive_reviews": effective_requirement,
        "assumed_successes_at_effective_requirement": assumed_successes,
        "assumed_errors_at_effective_requirement": assumed_errors,
        "lower_bound_at_effective_requirement": lower_bound,
        "weight_profile_count": len(normalized_weights),
        "weight_design_effect": weight_effect,
        "grouping_dimension_design_effects": profile.dimension_design_effects,
        "grouping_design_effect": profile.design_effect,
        "external_design_effect": design_effect,
        "combined_design_effect": combined_effect,
        "statistical_nominal_decisive_reviews": statistical_nominal,
        "required_nominal_decisive_reviews": required_nominal,
        "maximum_review_budget": policy.maximum_review_budget,
        "recommended_review_count": recommended,
        "observed_decisive_reviews": observed_decisive_reviews,
        "additional_decisive_reviews_needed": additional,
        "interval_semantics": (
            "exact_independent_binomial_reference_then_explicit_"
            "effective_sample_size_inflation_for_complex_design"
        ),
    }
    return ReviewRequirementPlan(
        **semantic_plan,
        plan_fingerprint=canonical_semantic_fingerprint(semantic_plan),
    )


def update_review_milestones(
    policy: ReviewEvidencePolicy,
    requirement_plan: ReviewRequirementPlan,
    evidence: Sequence[ReviewEvidenceObservation],
    *,
    grouping_profile: ReviewGroupingProfile | None = None,
) -> ReviewMilestoneUpdate:
    """Evaluate immutable evidence prefixes only at preregistered milestones."""

    if not isinstance(policy, ReviewEvidencePolicy):
        raise TypeError("policy must be a ReviewEvidencePolicy")
    if not isinstance(requirement_plan, ReviewRequirementPlan):
        raise TypeError("requirement_plan must be a ReviewRequirementPlan")
    if requirement_plan.policy_fingerprint != policy.fingerprint:
        raise ValueError("requirement plan references a different review policy")
    profile = grouping_profile or ReviewGroupingProfile()
    if not isinstance(profile, ReviewGroupingProfile):
        raise TypeError("grouping_profile must be a ReviewGroupingProfile")
    if requirement_plan.grouping_profile_fingerprint != profile.fingerprint:
        raise ValueError("grouping profile does not match the requirement plan")
    observations = tuple(evidence)
    if any(not isinstance(item, ReviewEvidenceObservation) for item in observations):
        raise TypeError("evidence must contain ReviewEvidenceObservation values")
    sequences = [item.review_sequence for item in observations]
    unit_ids = [item.review_unit_id for item in observations]
    if len(sequences) != len(set(sequences)):
        raise ValueError("review_sequence must be unique")
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("review_unit_id must be unique effective evidence")
    ordered = tuple(sorted(observations, key=lambda item: item.review_sequence))
    eligible = tuple(
        item
        for item in ordered
        if item.representative_estimation_eligible
        and item.sampling_purpose == "representative_audit"
        and item.human_supported is not None
    )
    targeted_excluded = sum(
        item.sampling_purpose == "targeted_failure_discovery" for item in ordered
    )
    nondecisive_excluded = sum(
        item.sampling_purpose != "targeted_failure_discovery"
        and item.human_supported is None
        for item in ordered
    )
    other_ineligible_excluded = sum(
        item.human_supported is not None
        and item.sampling_purpose != "targeted_failure_discovery"
        and not item.representative_estimation_eligible
        for item in ordered
    )
    weighted_error_rate = _weighted_error_rate(eligible)
    milestone_targets: list[tuple[float, int]] = []
    seen_targets: set[int] = set()
    for fraction in policy.milestone_information_fractions:
        target = max(
            1,
            math.ceil(requirement_plan.recommended_review_count * fraction),
        )
        if target not in seen_targets:
            milestone_targets.append((fraction, target))
            seen_targets.add(target)
    evaluations = tuple(
        _evaluate_milestone(
            information_fraction=fraction,
            target=target,
            eligible=eligible,
            policy=policy,
            requirement_plan=requirement_plan,
            grouping_profile=profile,
        )
        for fraction, target in milestone_targets
    )
    supported = next(
        (evaluation for evaluation in evaluations if evaluation.stop_authorized),
        None,
    )
    next_milestone = next(
        (
            evaluation.target_decisive_reviews
            for evaluation in evaluations
            if evaluation.status == "pending"
        ),
        None,
    )
    if supported is not None:
        decision = "stop_objective_met"
        reason = (
            "precision_point_lower_bound_and_required_strata_met_at_"
            "preregistered_milestone"
        )
        next_milestone = None
    elif len(eligible) >= policy.maximum_review_budget:
        decision = "budget_exhausted_without_support"
        reason = "maximum_review_budget_reached_without_stopping_criterion"
        next_milestone = None
    elif next_milestone is not None:
        decision = (
            "await_decisive_evidence" if not eligible else "continue_to_next_milestone"
        )
        reason = "next_preregistered_milestone_not_yet_reached"
    else:
        decision = "replan_from_observed_error"
        reason = "final_planned_milestone_failed_stopping_criterion"
    additional_to_next = (
        0 if next_milestone is None else max(0, next_milestone - len(eligible))
    )
    evidence_fingerprint = canonical_semantic_fingerprint(
        {
            "ordered_observations": [item.fingerprint for item in ordered],
        }
    )
    semantic_update = {
        "schema_version": REVIEW_MILESTONE_UPDATE_SCHEMA_VERSION,
        "policy_fingerprint": policy.fingerprint,
        "requirement_plan_fingerprint": requirement_plan.plan_fingerprint,
        "evidence_fingerprint": evidence_fingerprint,
        "total_review_events": len(ordered),
        "eligible_decisive_reviews": len(eligible),
        "eligible_supported_reviews": sum(
            item.human_supported is True for item in eligible
        ),
        "eligible_error_reviews": sum(
            item.human_supported is False for item in eligible
        ),
        "targeted_events_excluded": targeted_excluded,
        "nondecisive_events_excluded": nondecisive_excluded,
        "other_ineligible_events_excluded": other_ineligible_excluded,
        "observed_weighted_error_rate": weighted_error_rate,
        "milestones": tuple(asdict(evaluation) for evaluation in evaluations),
        "decision": decision,
        "decision_reason": reason,
        "next_milestone_count": next_milestone,
        "additional_decisive_reviews_to_next_milestone": additional_to_next,
        "stop_authorized": supported is not None,
        "representative_support_authorized": supported is not None,
        "occurrence_release_authorized": False,
    }
    return ReviewMilestoneUpdate(
        **{key: value for key, value in semantic_update.items() if key != "milestones"},
        milestones=evaluations,
        update_fingerprint=canonical_semantic_fingerprint(semantic_update),
    )


def _evaluate_milestone(
    *,
    information_fraction: float,
    target: int,
    eligible: tuple[ReviewEvidenceObservation, ...],
    policy: ReviewEvidencePolicy,
    requirement_plan: ReviewRequirementPlan,
    grouping_profile: ReviewGroupingProfile,
) -> ReviewMilestoneEvaluation:
    common = {
        "information_fraction": information_fraction,
        "target_decisive_reviews": target,
        "required_strata": requirement_plan.required_stratum_count,
        "external_design_effect": requirement_plan.external_design_effect,
        "confidence_level": policy.per_milestone_confidence_level,
    }
    if len(eligible) < target:
        return ReviewMilestoneEvaluation(
            **common,
            status="pending",
            decisive_reviews_evaluated=0,
            supported_reviews=None,
            error_reviews=None,
            weighted_precision=None,
            weighted_error_rate=None,
            represented_strata=None,
            weight_design_effect=None,
            grouping_dimension_design_effects=(),
            grouping_design_effect=None,
            combined_design_effect=None,
            effective_decisive_reviews=None,
            effective_supported_reviews=None,
            precision_lower_bound=None,
            interval_semantics=None,
            target_precision_met=False,
            lower_bound_objective_met=False,
            represented_strata_met=False,
            stop_authorized=False,
        )
    prefix = eligible[:target]
    weights = tuple(float(item.sampling_weight) for item in prefix)
    total_weight = math.fsum(weights)
    supported_weight = math.fsum(
        weight
        for item, weight in zip(prefix, weights, strict=True)
        if item.human_supported
    )
    precision = supported_weight / total_weight
    weight_effect = _weight_design_effect(weights)
    observed_grouping = _observed_grouping_profile(prefix, grouping_profile)
    combined_effect = (
        weight_effect
        * observed_grouping.design_effect
        * requirement_plan.external_design_effect
    )
    effective_trials = math.floor(target / combined_effect + 1e-12)
    supported_count = sum(item.human_supported is True for item in prefix)
    error_count = target - supported_count
    stratum_counts = Counter(item.stratum_id for item in prefix)
    represented_strata = sum(
        count >= policy.minimum_decisive_reviews_per_stratum
        for count in stratum_counts.values()
    )
    if effective_trials < 1:
        lower_bound = None
        effective_supported = None
        interval_semantics = "insufficient_effective_sample"
        status = "insufficient_effective_evidence"
    else:
        effective_supported = min(
            effective_trials,
            math.floor(precision * effective_trials + 1e-12),
        )
        lower_bound = clopper_pearson_lower_bound(
            effective_supported,
            effective_trials,
            confidence_level=policy.per_milestone_confidence_level,
        )
        independent = (
            math.isclose(combined_effect, 1.0, rel_tol=0.0, abs_tol=1e-12)
            and len(set(weights)) == 1
        )
        interval_semantics = (
            "exact_one_sided_clopper_pearson"
            if independent
            else (
                "conservative_integer_effective_sample_clopper_pearson_"
                "approximation_for_weighted_grouped_design"
            )
        )
        status = "evaluated"
    target_met = precision >= policy.target_precision
    lower_met = lower_bound is not None and lower_bound >= policy.lower_bound_objective
    strata_met = represented_strata >= requirement_plan.required_stratum_count
    stop = target_met and lower_met and strata_met
    if status == "evaluated":
        status = "evaluated_supported" if stop else "evaluated_not_supported"
    return ReviewMilestoneEvaluation(
        **common,
        status=status,
        decisive_reviews_evaluated=target,
        supported_reviews=supported_count,
        error_reviews=error_count,
        weighted_precision=precision,
        weighted_error_rate=1.0 - precision,
        represented_strata=represented_strata,
        weight_design_effect=weight_effect,
        grouping_dimension_design_effects=(observed_grouping.dimension_design_effects),
        grouping_design_effect=observed_grouping.design_effect,
        combined_design_effect=combined_effect,
        effective_decisive_reviews=effective_trials,
        effective_supported_reviews=effective_supported,
        precision_lower_bound=lower_bound,
        interval_semantics=interval_semantics,
        target_precision_met=target_met,
        lower_bound_objective_met=lower_met,
        represented_strata_met=strata_met,
        stop_authorized=stop,
    )


def _observed_grouping_profile(
    evidence: tuple[ReviewEvidenceObservation, ...],
    assumption: ReviewGroupingProfile,
) -> ReviewGroupingProfile:
    cluster_sizes: dict[str, tuple[int, ...]] = {}
    for dimension in ("reviewer", "owner", "duplicate", "observation"):
        counts = Counter(getattr(item, f"{dimension}_group_id") for item in evidence)
        cluster_sizes[dimension] = tuple(sorted(counts.values()))
    return ReviewGroupingProfile(
        reviewer_cluster_sizes=cluster_sizes["reviewer"],
        owner_cluster_sizes=cluster_sizes["owner"],
        duplicate_cluster_sizes=cluster_sizes["duplicate"],
        observation_cluster_sizes=cluster_sizes["observation"],
        reviewer_intraclass_correlation=(assumption.reviewer_intraclass_correlation),
        owner_intraclass_correlation=assumption.owner_intraclass_correlation,
        duplicate_intraclass_correlation=(assumption.duplicate_intraclass_correlation),
        observation_intraclass_correlation=(
            assumption.observation_intraclass_correlation
        ),
    )


def _weighted_error_rate(
    evidence: tuple[ReviewEvidenceObservation, ...],
) -> float | None:
    if not evidence:
        return None
    total = math.fsum(float(item.sampling_weight) for item in evidence)
    errors = math.fsum(
        float(item.sampling_weight)
        for item in evidence
        if item.human_supported is False
    )
    return errors / total


def _binomial_upper_tail(successes: int, trials: int, probability: float) -> float:
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    log_probability = math.log(probability)
    log_complement = math.log1p(-probability)
    terms = [
        math.lgamma(trials + 1)
        - math.lgamma(success_count + 1)
        - math.lgamma(trials - success_count + 1)
        + success_count * log_probability
        + (trials - success_count) * log_complement
        for success_count in range(successes, trials + 1)
    ]
    maximum = max(terms)
    return math.exp(maximum) * math.fsum(math.exp(term - maximum) for term in terms)


def _sampling_weights(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("sampling_weights must be a numeric sequence")
    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("sampling_weights must be numeric")
        weight = float(value)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("sampling_weights must be finite and positive")
        normalized.append(weight)
    return tuple(normalized)


def _weight_design_effect(weights: tuple[float, ...]) -> float:
    if not weights:
        return 1.0
    return (
        len(weights)
        * math.fsum(weight * weight for weight in weights)
        / (math.fsum(weights) ** 2)
    )


def _cluster_sizes(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise TypeError(f"{field} must be a nonempty tuple")
    if any(
        not isinstance(size, int) or isinstance(size, bool) or size < 1
        for size in value
    ):
        raise ValueError(f"{field} must contain positive integers")
    return tuple(value)


def _closed_open_probability(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized < 1.0:
        raise ValueError(f"{field} must be within [0, 1)")
    return normalized


def _finite_at_least_one(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 1.0:
        raise ValueError(f"{field} must be finite and at least one")
    return normalized


def _positive_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return normalized


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field).casefold()
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValueError(f"{field} must be a sha256 fingerprint")
    try:
        int(text.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a sha256 fingerprint") from exc
    return text


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value.strip()


def _open_probability(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 < normalized < 1.0:
        raise ValueError(f"{field} must be within (0, 1)")
    return normalized


def _milestone_fractions(value: object) -> tuple[float, ...]:
    if not isinstance(value, tuple) or not value:
        raise TypeError("milestone_information_fractions must be a nonempty tuple")
    fractions: list[float] = []
    for fraction in value:
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            raise TypeError("milestone_information_fractions must be numeric")
        normalized = float(fraction)
        if not math.isfinite(normalized) or not 0.0 < normalized <= 1.0:
            raise ValueError("milestone_information_fractions must be within (0, 1]")
        fractions.append(normalized)
    normalized_fractions = tuple(fractions)
    if tuple(sorted(set(normalized_fractions))) != normalized_fractions:
        raise ValueError(
            "milestone_information_fractions must be unique and increasing"
        )
    if normalized_fractions[-1] != 1.0:
        raise ValueError("final milestone information fraction must equal one")
    return normalized_fractions


__all__ = [
    "COMPLEX_DESIGN_ADJUSTMENT",
    "GROUPING_DIMENSIONS",
    "INTERVAL_METHOD",
    "MILESTONE_POLICY",
    "REVIEW_EVIDENCE_POLICY_FILE",
    "REVIEW_EVIDENCE_POLICY_SCHEMA_VERSION",
    "REVIEW_MILESTONE_UPDATE_FILE",
    "REVIEW_MILESTONE_UPDATE_SCHEMA_VERSION",
    "REVIEW_REQUIREMENT_PLAN_FILE",
    "REVIEW_REQUIREMENT_PLAN_SCHEMA_VERSION",
    "STOPPING_RULE",
    "TARGET_METRIC",
    "ReviewEvidencePolicy",
    "ReviewEvidenceObservation",
    "ReviewGroupingProfile",
    "ReviewMilestoneEvaluation",
    "ReviewMilestoneUpdate",
    "ReviewRequirementPlan",
    "calculate_review_requirements",
    "clopper_pearson_lower_bound",
    "update_review_milestones",
]
