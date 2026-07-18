"""Dynamic human-review evidence planning and stopping contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math

from biominer.common.semantic_hash import canonical_semantic_fingerprint


REVIEW_EVIDENCE_POLICY_SCHEMA_VERSION = "review-evidence-policy-v1.0.0"
REVIEW_EVIDENCE_POLICY_FILE = "review_evidence_policy.json"

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
    "STOPPING_RULE",
    "TARGET_METRIC",
    "ReviewEvidencePolicy",
]
