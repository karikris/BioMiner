"""Precision-constrained selective prediction and calibrated abstention policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
import re

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.ml.nonmatch import (
    ABSTAIN,
    CLASSIFICATION_OUTCOMES,
    DOMAIN_NEGATIVE_OUTCOMES,
    INSUFFICIENT_REFERENCE_COVERAGE,
    INSUFFICIENT_VISUAL_DETAIL,
    KNOWN_NONREGIONAL_COMPETITOR,
    KNOWN_REGIONAL_COMPETITOR,
    NO_GEO_GLOBAL_FALLBACK,
    OUT_OF_DISTRIBUTION,
    TARGET_CONFIRMED,
    TARGET_PROBABLE_REVIEW,
    ExplicitNonMatchScore,
)
from biominer.ml.training_features import TARGET_TASKS
from biominer.references.readiness import REFERENCE_ROUTES


DECISION_POLICY_MANIFEST_SCHEMA_VERSION = "few-shot-decision-policy-v1.0.0"
DECISION_POLICY_VERSION = "precision-constrained-selective-policy-v1.0.0"
DECISION_POLICY_CALIBRATION_SAMPLE_VERSION = (
    "decision-policy-calibration-samples-v1.0.0"
)
DECISION_POLICY_OPTIMIZATION_METRIC = "weighted_target_recall_at_precision"
DECISION_POLICY_STATUSES = frozenset({"fitted", "infeasible"})
CALIBRATION_PARTITION = "calibration"
MAX_THRESHOLD_GRID_CELLS = 4_000_000

_REQUIREMENTS = (
    ("route_compatible", True),
    ("reference_coverage_sufficient", True),
    ("geographic_evidence_sufficient", True),
    ("domain_negative_absent", True),
    ("out_of_distribution_absent", True),
    ("visual_detail_sufficient", True),
    ("no_geo_global_fallback_absent", True),
)
_ABSTENTION_RULES = (
    "decision_policy_infeasible",
    "incompatible_route",
    "domain_negative_without_supported_outcome",
    "out_of_distribution",
    "insufficient_visual_detail",
    "insufficient_reference_coverage",
    "no_geo_global_fallback",
    "weak_geographic_evidence",
    "calibrated_non_target_dominates",
    "missing_calibrated_target_probability",
    "target_probability_below_threshold",
    "missing_competitor_margin",
    "competitor_margin_below_threshold",
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class DecisionPolicyCalibrationSample:
    """One reviewed calibration-split row used only for policy threshold fitting."""

    sample_id: str
    leakage_component_id: str
    dataset_split: str
    true_target: bool
    calibrated_target_probability: float | None
    competitor_margin: float | None
    sample_weight: float = 1.0
    route_compatible: bool = True
    reference_coverage_sufficient: bool = True
    geographic_evidence_sufficient: bool = True
    domain_negative_detected: bool = False
    out_of_distribution: bool = False
    visual_detail_sufficient: bool = True
    no_geo_global_fallback: bool = False


@dataclass(frozen=True, slots=True)
class SelectiveDecisionPolicyConfig:
    """Immutable policy objective and all model/split identities."""

    target_task: str
    route: str
    target_precision_objective: float
    model_fingerprint: str
    classifier_fingerprint: str
    calibration_fingerprint: str
    split_fingerprint: str
    optimization_metric: str = DECISION_POLICY_OPTIMIZATION_METRIC
    max_threshold_grid_cells: int = MAX_THRESHOLD_GRID_CELLS

    def __post_init__(self) -> None:
        task = _required_choice(
            self.target_task,
            field="target_task",
            allowed=TARGET_TASKS,
        )
        route = _required_choice(self.route, field="route", allowed=REFERENCE_ROUTES)
        objective = _unit_interval(
            self.target_precision_objective,
            field="target_precision_objective",
        )
        if objective <= 0.0:
            raise ValueError("target_precision_objective must be greater than zero")
        if self.optimization_metric != DECISION_POLICY_OPTIMIZATION_METRIC:
            raise ValueError("unsupported decision-policy optimization metric")
        maximum = _positive_integer(
            self.max_threshold_grid_cells,
            field="max_threshold_grid_cells",
        )
        object.__setattr__(self, "target_task", task)
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "target_precision_objective", objective)
        object.__setattr__(
            self,
            "model_fingerprint",
            _sha256(self.model_fingerprint, field="model_fingerprint"),
        )
        object.__setattr__(
            self,
            "classifier_fingerprint",
            _sha256(self.classifier_fingerprint, field="classifier_fingerprint"),
        )
        object.__setattr__(
            self,
            "calibration_fingerprint",
            _sha256(self.calibration_fingerprint, field="calibration_fingerprint"),
        )
        object.__setattr__(
            self,
            "split_fingerprint",
            _sha256(self.split_fingerprint, field="split_fingerprint"),
        )
        object.__setattr__(self, "max_threshold_grid_cells", maximum)


@dataclass(frozen=True, slots=True)
class SelectiveDecisionPolicy:
    """Learned confirmation thresholds and their calibration-only provenance."""

    schema_version: str
    policy_version: str
    status: str
    status_reason: str
    target_confirmation_enabled: bool
    target_task: str
    route: str
    target_probability_threshold: float | None
    competitor_margin_threshold: float | None
    optimization_metric: str
    target_precision_objective: float
    achieved_weighted_precision: float | None
    achieved_weighted_recall: float | None
    achieved_weighted_coverage: float | None
    achieved_unweighted_precision: float | None
    achieved_unweighted_recall: float | None
    achieved_unweighted_coverage: float | None
    calibration_sample_count: int
    eligible_calibration_sample_count: int
    calibration_group_count: int
    positive_sample_count: int
    negative_sample_count: int
    threshold_grid_size: int
    calibration_sample_fingerprint: str
    model_fingerprint: str
    classifier_fingerprint: str
    calibration_fingerprint: str
    split_fingerprint: str
    requirements: tuple[tuple[str, bool], ...]
    abstention_rules: tuple[str, ...]
    decision_policy_fingerprint: str


@dataclass(frozen=True, slots=True)
class SelectivePredictionEvidence:
    """Runtime non-match score plus fail-closed quality and identity gates."""

    nonmatch_score: ExplicitNonMatchScore
    model_fingerprint: str
    classifier_fingerprint: str
    calibration_fingerprint: str
    route_compatible: bool
    reference_coverage_sufficient: bool
    geographic_evidence_sufficient: bool
    domain_negative_detected: bool
    out_of_distribution: bool
    visual_detail_sufficient: bool
    no_geo_global_fallback: bool


@dataclass(frozen=True, slots=True)
class SelectiveDecision:
    """One policy decision with explicit abstention and threshold provenance."""

    classification_decision: str
    abstained: bool
    abstention_reason: str | None
    review_priority: str
    model_decision_threshold: float | None
    competitor_margin_threshold: float | None
    threshold_provenance: str
    target_probability: float | None
    competitor_margin: float | None
    winning_nonmatch_accepted_taxon_key: str | None
    decision_policy_fingerprint: str


@dataclass(frozen=True, slots=True)
class _ThresholdMetrics:
    probability_threshold: float
    margin_threshold: float
    weighted_precision: float
    weighted_recall: float
    weighted_coverage: float
    unweighted_precision: float
    unweighted_recall: float
    unweighted_coverage: float


def fit_selective_decision_policy(
    samples: Sequence[DecisionPolicyCalibrationSample],
    config: SelectiveDecisionPolicyConfig,
) -> SelectiveDecisionPolicy:
    """Jointly select confirmation thresholds on calibration-only grouped rows."""

    if not isinstance(config, SelectiveDecisionPolicyConfig):
        raise TypeError("config must be a SelectiveDecisionPolicyConfig")
    normalized = _validate_samples(samples)
    eligible = tuple(item for item in normalized if _threshold_eligible(item))
    probability_thresholds = tuple(
        sorted(
            {
                float(item.calibrated_target_probability)
                for item in eligible
                if item.calibrated_target_probability is not None
            }
        )
    )
    margin_thresholds = tuple(
        sorted(
            {
                float(item.competitor_margin)
                for item in eligible
                if item.competitor_margin is not None
            }
        )
    )
    grid_size = len(probability_thresholds) * len(margin_thresholds)
    if grid_size > config.max_threshold_grid_cells:
        raise ValueError(
            "decision-policy threshold grid exceeds max_threshold_grid_cells"
        )
    total_positive_weight = sum(
        item.sample_weight for item in normalized if item.true_target
    )
    total_weight = sum(item.sample_weight for item in normalized)
    total_positive_count = sum(item.true_target for item in normalized)
    best: _ThresholdMetrics | None = None
    for probability_threshold in probability_thresholds:
        for margin_threshold in margin_thresholds:
            metrics = _threshold_metrics(
                normalized,
                probability_threshold=probability_threshold,
                margin_threshold=margin_threshold,
                total_positive_weight=total_positive_weight,
                total_weight=total_weight,
                total_positive_count=total_positive_count,
            )
            if (
                metrics is None
                or metrics.weighted_precision + 1e-15
                < config.target_precision_objective
            ):
                continue
            if best is None or _threshold_rank(metrics) > _threshold_rank(best):
                best = metrics

    sample_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DECISION_POLICY_CALIBRATION_SAMPLE_VERSION,
            "split_fingerprint": config.split_fingerprint,
            "calibration_fingerprint": config.calibration_fingerprint,
            "samples": [_sample_payload(item) for item in normalized],
        }
    )
    common: dict[str, object] = {
        "schema_version": DECISION_POLICY_MANIFEST_SCHEMA_VERSION,
        "policy_version": DECISION_POLICY_VERSION,
        "target_task": config.target_task,
        "route": config.route,
        "optimization_metric": config.optimization_metric,
        "target_precision_objective": config.target_precision_objective,
        "calibration_sample_count": len(normalized),
        "eligible_calibration_sample_count": len(eligible),
        "calibration_group_count": len(
            {item.leakage_component_id for item in normalized}
        ),
        "positive_sample_count": total_positive_count,
        "negative_sample_count": len(normalized) - total_positive_count,
        "threshold_grid_size": grid_size,
        "calibration_sample_fingerprint": sample_fingerprint,
        "model_fingerprint": config.model_fingerprint,
        "classifier_fingerprint": config.classifier_fingerprint,
        "calibration_fingerprint": config.calibration_fingerprint,
        "split_fingerprint": config.split_fingerprint,
        "requirements": _REQUIREMENTS,
        "abstention_rules": _ABSTENTION_RULES,
    }
    if best is None:
        values = {
            **common,
            "status": "infeasible",
            "status_reason": "no_threshold_pair_meets_target_precision_objective",
            "target_confirmation_enabled": False,
            "target_probability_threshold": None,
            "competitor_margin_threshold": None,
            "achieved_weighted_precision": None,
            "achieved_weighted_recall": None,
            "achieved_weighted_coverage": None,
            "achieved_unweighted_precision": None,
            "achieved_unweighted_recall": None,
            "achieved_unweighted_coverage": None,
        }
    else:
        values = {
            **common,
            "status": "fitted",
            "status_reason": "precision_objective_met",
            "target_confirmation_enabled": True,
            "target_probability_threshold": best.probability_threshold,
            "competitor_margin_threshold": best.margin_threshold,
            "achieved_weighted_precision": best.weighted_precision,
            "achieved_weighted_recall": best.weighted_recall,
            "achieved_weighted_coverage": best.weighted_coverage,
            "achieved_unweighted_precision": best.unweighted_precision,
            "achieved_unweighted_recall": best.unweighted_recall,
            "achieved_unweighted_coverage": best.unweighted_coverage,
        }
    fingerprint = canonical_semantic_fingerprint(_policy_semantic_payload(values))
    return SelectiveDecisionPolicy(
        **values,
        decision_policy_fingerprint=fingerprint,
    )


def decision_policy_manifest_payload(
    policy: SelectiveDecisionPolicy,
) -> dict[str, object]:
    """Return the canonical JSON-like decision-policy payload for calibration."""

    if not isinstance(policy, SelectiveDecisionPolicy):
        raise TypeError("policy must be a SelectiveDecisionPolicy")
    values = _policy_values(policy)
    expected = canonical_semantic_fingerprint(_policy_semantic_payload(values))
    if expected != policy.decision_policy_fingerprint:
        raise ValueError("decision policy fingerprint is inconsistent")
    return {
        **_policy_semantic_payload(values),
        "decision_policy_fingerprint": policy.decision_policy_fingerprint,
    }


def apply_selective_decision_policy(
    policy: SelectiveDecisionPolicy,
    evidence: SelectivePredictionEvidence,
) -> SelectiveDecision:
    """Apply non-match and quality gates before considering target confirmation."""

    decision_policy_manifest_payload(policy)
    normalized = _validate_prediction_evidence(policy, evidence)
    score = normalized.nonmatch_score
    if policy.status != "fitted" or not policy.target_confirmation_enabled:
        return _decision(
            policy,
            score,
            outcome=ABSTAIN,
            abstained=True,
            reason="decision_policy_infeasible",
        )
    if not normalized.route_compatible or score.route != policy.route:
        return _decision(
            policy,
            score,
            outcome=ABSTAIN,
            abstained=True,
            reason="incompatible_route",
        )
    if normalized.domain_negative_detected:
        outcome = score.best_domain_negative_outcome
        if outcome not in DOMAIN_NEGATIVE_OUTCOMES:
            return _decision(
                policy,
                score,
                outcome=ABSTAIN,
                abstained=True,
                reason="domain_negative_without_supported_outcome",
            )
        return _decision(
            policy,
            score,
            outcome=outcome,
            abstained=False,
            reason=None,
        )
    if normalized.out_of_distribution:
        return _decision(
            policy,
            score,
            outcome=OUT_OF_DISTRIBUTION,
            abstained=True,
            reason="out_of_distribution",
        )
    if not normalized.visual_detail_sufficient:
        return _decision(
            policy,
            score,
            outcome=INSUFFICIENT_VISUAL_DETAIL,
            abstained=True,
            reason="insufficient_visual_detail",
        )
    if not normalized.reference_coverage_sufficient:
        return _decision(
            policy,
            score,
            outcome=INSUFFICIENT_REFERENCE_COVERAGE,
            abstained=True,
            reason="insufficient_reference_coverage",
        )
    if normalized.no_geo_global_fallback:
        return _decision(
            policy,
            score,
            outcome=NO_GEO_GLOBAL_FALLBACK,
            abstained=True,
            reason="no_geo_global_fallback",
        )

    if score.nonmatch_margin is not None and score.nonmatch_margin <= 0.0:
        leading = score.calibrated_best_non_target_outcome
        if leading in {
            KNOWN_REGIONAL_COMPETITOR,
            KNOWN_NONREGIONAL_COMPETITOR,
            *DOMAIN_NEGATIVE_OUTCOMES,
        }:
            return _decision(
                policy,
                score,
                outcome=leading,
                abstained=False,
                reason=None,
            )
        return _decision(
            policy,
            score,
            outcome=ABSTAIN,
            abstained=True,
            reason="calibrated_non_target_dominates",
        )

    target_probability = score.calibrated_target_probability
    if target_probability is None:
        return _decision(
            policy,
            score,
            outcome=ABSTAIN,
            abstained=True,
            reason="missing_calibrated_target_probability",
        )
    assert policy.target_probability_threshold is not None
    assert policy.competitor_margin_threshold is not None
    if target_probability < policy.target_probability_threshold:
        return _decision(
            policy,
            score,
            outcome=TARGET_PROBABLE_REVIEW,
            abstained=True,
            reason="target_probability_below_threshold",
        )
    if score.competitor_margin is None:
        return _decision(
            policy,
            score,
            outcome=ABSTAIN,
            abstained=True,
            reason="missing_competitor_margin",
        )
    if score.competitor_margin < policy.competitor_margin_threshold:
        return _decision(
            policy,
            score,
            outcome=ABSTAIN,
            abstained=True,
            reason="competitor_margin_below_threshold",
        )
    if not normalized.geographic_evidence_sufficient:
        return _decision(
            policy,
            score,
            outcome=TARGET_PROBABLE_REVIEW,
            abstained=True,
            reason="weak_geographic_evidence",
        )
    return _decision(
        policy,
        score,
        outcome=TARGET_CONFIRMED,
        abstained=False,
        reason=None,
    )


def _decision(
    policy: SelectiveDecisionPolicy,
    score: ExplicitNonMatchScore,
    *,
    outcome: str,
    abstained: bool,
    reason: str | None,
) -> SelectiveDecision:
    if outcome not in CLASSIFICATION_OUTCOMES:
        raise AssertionError(f"unsupported decision outcome: {outcome}")
    if outcome == TARGET_CONFIRMED:
        priority = "none"
    elif outcome == TARGET_PROBABLE_REVIEW:
        priority = "high"
    elif abstained:
        priority = "normal"
    else:
        priority = "none"
    return SelectiveDecision(
        classification_decision=outcome,
        abstained=abstained,
        abstention_reason=reason,
        review_priority=priority,
        model_decision_threshold=policy.target_probability_threshold,
        competitor_margin_threshold=policy.competitor_margin_threshold,
        threshold_provenance=policy.decision_policy_fingerprint,
        target_probability=score.calibrated_target_probability,
        competitor_margin=score.competitor_margin,
        winning_nonmatch_accepted_taxon_key=(
            score.calibrated_best_non_target_accepted_taxon_key
            if outcome in {KNOWN_REGIONAL_COMPETITOR, KNOWN_NONREGIONAL_COMPETITOR}
            else None
        ),
        decision_policy_fingerprint=policy.decision_policy_fingerprint,
    )


def _validate_samples(
    samples: Sequence[DecisionPolicyCalibrationSample],
) -> tuple[DecisionPolicyCalibrationSample, ...]:
    if not samples:
        raise ValueError("decision-policy calibration samples must not be empty")
    normalized = []
    sample_ids: set[str] = set()
    for item in samples:
        if not isinstance(item, DecisionPolicyCalibrationSample):
            raise TypeError(
                "samples must contain DecisionPolicyCalibrationSample values"
            )
        sample_id = _required_text(item.sample_id, field="sample_id")
        if sample_id in sample_ids:
            raise ValueError("decision-policy sample IDs must be unique")
        sample_ids.add(sample_id)
        component_id = _required_text(
            item.leakage_component_id,
            field="leakage_component_id",
        )
        if item.dataset_split != CALIBRATION_PARTITION:
            raise ValueError("decision-policy fitting requires calibration split rows")
        if not isinstance(item.true_target, bool):
            raise TypeError("true_target must be boolean")
        probability = _optional_unit_interval(
            item.calibrated_target_probability,
            field="calibrated_target_probability",
        )
        margin = _optional_competitor_margin(
            item.competitor_margin,
            field="competitor_margin",
        )
        weight = _finite_number(item.sample_weight, field="sample_weight")
        if weight <= 0.0:
            raise ValueError("sample_weight must be positive")
        booleans = {
            "route_compatible": item.route_compatible,
            "reference_coverage_sufficient": item.reference_coverage_sufficient,
            "geographic_evidence_sufficient": (item.geographic_evidence_sufficient),
            "domain_negative_detected": item.domain_negative_detected,
            "out_of_distribution": item.out_of_distribution,
            "visual_detail_sufficient": item.visual_detail_sufficient,
            "no_geo_global_fallback": item.no_geo_global_fallback,
        }
        for field, value in booleans.items():
            if not isinstance(value, bool):
                raise TypeError(f"{field} must be boolean")
        normalized.append(
            DecisionPolicyCalibrationSample(
                sample_id=sample_id,
                leakage_component_id=component_id,
                dataset_split=CALIBRATION_PARTITION,
                true_target=item.true_target,
                calibrated_target_probability=probability,
                competitor_margin=margin,
                sample_weight=weight,
                **booleans,
            )
        )
    class_values = {item.true_target for item in normalized}
    if class_values != {False, True}:
        raise ValueError("decision-policy calibration requires both target classes")
    if len({item.leakage_component_id for item in normalized}) < 2:
        raise ValueError("decision-policy calibration requires independent groups")
    return tuple(sorted(normalized, key=lambda item: item.sample_id))


def _threshold_eligible(item: DecisionPolicyCalibrationSample) -> bool:
    return bool(
        item.route_compatible
        and item.reference_coverage_sufficient
        and item.geographic_evidence_sufficient
        and not item.domain_negative_detected
        and not item.out_of_distribution
        and item.visual_detail_sufficient
        and not item.no_geo_global_fallback
        and item.calibrated_target_probability is not None
        and item.competitor_margin is not None
    )


def _threshold_metrics(
    samples: Sequence[DecisionPolicyCalibrationSample],
    *,
    probability_threshold: float,
    margin_threshold: float,
    total_positive_weight: float,
    total_weight: float,
    total_positive_count: int,
) -> _ThresholdMetrics | None:
    confirmed = tuple(
        item
        for item in samples
        if _threshold_eligible(item)
        and float(item.calibrated_target_probability) >= probability_threshold
        and float(item.competitor_margin) >= margin_threshold
    )
    if not confirmed:
        return None
    confirmed_weight = sum(item.sample_weight for item in confirmed)
    true_positive_weight = sum(
        item.sample_weight for item in confirmed if item.true_target
    )
    true_positive_count = sum(item.true_target for item in confirmed)
    return _ThresholdMetrics(
        probability_threshold=probability_threshold,
        margin_threshold=margin_threshold,
        weighted_precision=true_positive_weight / confirmed_weight,
        weighted_recall=true_positive_weight / total_positive_weight,
        weighted_coverage=confirmed_weight / total_weight,
        unweighted_precision=true_positive_count / len(confirmed),
        unweighted_recall=true_positive_count / total_positive_count,
        unweighted_coverage=len(confirmed) / len(samples),
    )


def _threshold_rank(metrics: _ThresholdMetrics) -> tuple[float, ...]:
    return (
        metrics.weighted_recall,
        metrics.weighted_precision,
        metrics.weighted_coverage,
        metrics.unweighted_precision,
        metrics.probability_threshold,
        metrics.margin_threshold,
    )


def _sample_payload(item: DecisionPolicyCalibrationSample) -> dict[str, object]:
    return {
        "sample_id": item.sample_id,
        "leakage_component_id": item.leakage_component_id,
        "dataset_split": item.dataset_split,
        "true_target": item.true_target,
        "calibrated_target_probability": item.calibrated_target_probability,
        "competitor_margin": item.competitor_margin,
        "sample_weight": item.sample_weight,
        "route_compatible": item.route_compatible,
        "reference_coverage_sufficient": item.reference_coverage_sufficient,
        "geographic_evidence_sufficient": item.geographic_evidence_sufficient,
        "domain_negative_detected": item.domain_negative_detected,
        "out_of_distribution": item.out_of_distribution,
        "visual_detail_sufficient": item.visual_detail_sufficient,
        "no_geo_global_fallback": item.no_geo_global_fallback,
    }


def _policy_values(policy: SelectiveDecisionPolicy) -> dict[str, object]:
    return {
        "schema_version": policy.schema_version,
        "policy_version": policy.policy_version,
        "status": policy.status,
        "status_reason": policy.status_reason,
        "target_confirmation_enabled": policy.target_confirmation_enabled,
        "target_task": policy.target_task,
        "route": policy.route,
        "target_probability_threshold": policy.target_probability_threshold,
        "competitor_margin_threshold": policy.competitor_margin_threshold,
        "optimization_metric": policy.optimization_metric,
        "target_precision_objective": policy.target_precision_objective,
        "achieved_weighted_precision": policy.achieved_weighted_precision,
        "achieved_weighted_recall": policy.achieved_weighted_recall,
        "achieved_weighted_coverage": policy.achieved_weighted_coverage,
        "achieved_unweighted_precision": policy.achieved_unweighted_precision,
        "achieved_unweighted_recall": policy.achieved_unweighted_recall,
        "achieved_unweighted_coverage": policy.achieved_unweighted_coverage,
        "calibration_sample_count": policy.calibration_sample_count,
        "eligible_calibration_sample_count": policy.eligible_calibration_sample_count,
        "calibration_group_count": policy.calibration_group_count,
        "positive_sample_count": policy.positive_sample_count,
        "negative_sample_count": policy.negative_sample_count,
        "threshold_grid_size": policy.threshold_grid_size,
        "calibration_sample_fingerprint": policy.calibration_sample_fingerprint,
        "model_fingerprint": policy.model_fingerprint,
        "classifier_fingerprint": policy.classifier_fingerprint,
        "calibration_fingerprint": policy.calibration_fingerprint,
        "split_fingerprint": policy.split_fingerprint,
        "requirements": policy.requirements,
        "abstention_rules": policy.abstention_rules,
    }


def _policy_semantic_payload(values: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": values["schema_version"],
        "policy_version": values["policy_version"],
        "status": values["status"],
        "status_reason": values["status_reason"],
        "target_confirmation_enabled": values["target_confirmation_enabled"],
        "target_task": values["target_task"],
        "route": values["route"],
        "target_probability_threshold": values["target_probability_threshold"],
        "competitor_margin_threshold": values["competitor_margin_threshold"],
        "optimization_metric": values["optimization_metric"],
        "target_precision_objective": values["target_precision_objective"],
        "achieved_metrics": {
            "weighted_precision": values["achieved_weighted_precision"],
            "weighted_recall": values["achieved_weighted_recall"],
            "weighted_coverage": values["achieved_weighted_coverage"],
            "unweighted_precision": values["achieved_unweighted_precision"],
            "unweighted_recall": values["achieved_unweighted_recall"],
            "unweighted_coverage": values["achieved_unweighted_coverage"],
        },
        "calibration_sample_count": values["calibration_sample_count"],
        "eligible_calibration_sample_count": values[
            "eligible_calibration_sample_count"
        ],
        "calibration_group_count": values["calibration_group_count"],
        "positive_sample_count": values["positive_sample_count"],
        "negative_sample_count": values["negative_sample_count"],
        "threshold_grid_size": values["threshold_grid_size"],
        "calibration_sample_fingerprint": values["calibration_sample_fingerprint"],
        "model_fingerprint": values["model_fingerprint"],
        "classifier_fingerprint": values["classifier_fingerprint"],
        "calibration_fingerprint": values["calibration_fingerprint"],
        "split_fingerprint": values["split_fingerprint"],
        "requirements": dict(values["requirements"]),
        "abstention_rules": list(values["abstention_rules"]),
    }


def _validate_prediction_evidence(
    policy: SelectiveDecisionPolicy,
    evidence: SelectivePredictionEvidence,
) -> SelectivePredictionEvidence:
    if not isinstance(evidence, SelectivePredictionEvidence):
        raise TypeError("evidence must be SelectivePredictionEvidence")
    if not isinstance(evidence.nonmatch_score, ExplicitNonMatchScore):
        raise TypeError("nonmatch_score must be ExplicitNonMatchScore")
    model = _sha256(evidence.model_fingerprint, field="model_fingerprint")
    classifier = _sha256(
        evidence.classifier_fingerprint,
        field="classifier_fingerprint",
    )
    calibration = _sha256(
        evidence.calibration_fingerprint,
        field="calibration_fingerprint",
    )
    if model != policy.model_fingerprint:
        raise ValueError("runtime model fingerprint does not match decision policy")
    if classifier != policy.classifier_fingerprint:
        raise ValueError(
            "runtime classifier fingerprint does not match decision policy"
        )
    if calibration != policy.calibration_fingerprint:
        raise ValueError(
            "runtime calibration fingerprint does not match decision policy"
        )
    score_calibrator = evidence.nonmatch_score.target_calibrator_fingerprint
    if (
        score_calibrator is not None
        and score_calibrator != policy.calibration_fingerprint
    ):
        raise ValueError("target score calibrator does not match decision policy")
    booleans = {
        "route_compatible": evidence.route_compatible,
        "reference_coverage_sufficient": evidence.reference_coverage_sufficient,
        "geographic_evidence_sufficient": (evidence.geographic_evidence_sufficient),
        "domain_negative_detected": evidence.domain_negative_detected,
        "out_of_distribution": evidence.out_of_distribution,
        "visual_detail_sufficient": evidence.visual_detail_sufficient,
        "no_geo_global_fallback": evidence.no_geo_global_fallback,
    }
    for field, value in booleans.items():
        if not isinstance(value, bool):
            raise TypeError(f"{field} must be boolean")
    return SelectivePredictionEvidence(
        nonmatch_score=evidence.nonmatch_score,
        model_fingerprint=model,
        classifier_fingerprint=classifier,
        calibration_fingerprint=calibration,
        **booleans,
    )


def _optional_unit_interval(value: object, *, field: str) -> float | None:
    return None if value is None else _unit_interval(value, field=field)


def _optional_competitor_margin(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    result = _finite_number(value, field=field)
    if not -2.0 <= result <= 2.0:
        raise ValueError(f"{field} must be between -2 and 2")
    return result


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be non-empty canonical text")
    return value


def _required_choice(
    value: object,
    *,
    field: str,
    allowed: frozenset[str] | set[str],
) -> str:
    result = _required_text(value, field=field)
    if result not in allowed:
        raise ValueError(f"{field} is not supported: {result}")
    return result


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full lowercase sha256 fingerprint")
    return text


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _unit_interval(value: object, *, field: str) -> float:
    result = _finite_number(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be between zero and one")
    return result


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


__all__ = [
    "CALIBRATION_PARTITION",
    "DECISION_POLICY_CALIBRATION_SAMPLE_VERSION",
    "DECISION_POLICY_MANIFEST_SCHEMA_VERSION",
    "DECISION_POLICY_OPTIMIZATION_METRIC",
    "DECISION_POLICY_STATUSES",
    "DECISION_POLICY_VERSION",
    "DecisionPolicyCalibrationSample",
    "SelectiveDecision",
    "SelectiveDecisionPolicy",
    "SelectiveDecisionPolicyConfig",
    "SelectivePredictionEvidence",
    "apply_selective_decision_policy",
    "decision_policy_manifest_payload",
    "fit_selective_decision_policy",
]
