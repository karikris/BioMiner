"""Dynamic human-review evidence planning and stopping contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

from biominer.common.semantic_hash import canonical_semantic_fingerprint


REVIEW_EVIDENCE_POLICY_SCHEMA_VERSION = "review-evidence-policy-v1.0.0"
REVIEW_EVIDENCE_POLICY_FILE = "review_evidence_policy.json"
REVIEW_REQUIREMENT_PLAN_SCHEMA_VERSION = "review-requirement-plan-v1.0.0"
REVIEW_REQUIREMENT_PLAN_FILE = "review_requirement_plan.json"

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
    "REVIEW_REQUIREMENT_PLAN_FILE",
    "REVIEW_REQUIREMENT_PLAN_SCHEMA_VERSION",
    "STOPPING_RULE",
    "TARGET_METRIC",
    "ReviewEvidencePolicy",
    "ReviewGroupingProfile",
    "ReviewRequirementPlan",
    "calculate_review_requirements",
    "clopper_pearson_lower_bound",
]
