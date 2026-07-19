"""Risk-controlled screening thresholds from independent validation evidence."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import NormalDist

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.calibration import CALIBRATED_TARGET_PROBABILITY_KIND
from biominer.evaluation.review_evidence import clopper_pearson_lower_bound
from biominer.ml.dynamic_pool_calibration import (
    DYNAMIC_POOL_CALIBRATION_PREDICTION_SCHEMA,
    DynamicPoolCalibrationFit,
)


AUDITED_SCREENING_THRESHOLD_VERSION = "audited-screening-threshold-v1.0.0"
AUDITED_SCREENING_THRESHOLD_AUDIT_SCHEMA_VERSION = (
    "audited-screening-threshold-audit-v1.0.0"
)
AUDITED_SCREENING_THRESHOLD_STATUSES = frozenset({"selected", "infeasible"})
SCREENING_CANDIDATE_LABEL = "statistically_supported_screening_candidate"
LOWER_BOUND_METHOD = (
    "minimum_of_one_sided_kish_wilson_and_exact_all_rows_component_clopper_pearson"
)

AUDITED_SCREENING_THRESHOLD_AUDIT_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "policy_fingerprint": pl.String,
    "fit_fingerprint": pl.String,
    "threshold_audit_fingerprint": pl.String,
    "threshold_row_fingerprint": pl.String,
    "evidence_model_fingerprint": pl.String,
    "calibrator_fingerprint": pl.String,
    "split_fingerprint": pl.String,
    "validation_prediction_artifact_fingerprint": pl.String,
    "evaluation_split": pl.String,
    "probability_kind": pl.String,
    "lower_bound_method": pl.String,
    "confidence_level": pl.Float64,
    "minimum_precision_lower_bound": pl.Float64,
    "threshold": pl.Float64,
    "validation_item_count": pl.UInt32,
    "validation_component_count": pl.UInt32,
    "validation_weight": pl.Float64,
    "selected_item_count": pl.UInt32,
    "selected_component_count": pl.UInt32,
    "selected_weight": pl.Float64,
    "supported_item_count": pl.UInt32,
    "error_item_count": pl.UInt32,
    "supported_weight": pl.Float64,
    "error_weight": pl.Float64,
    "weighted_precision": pl.Float64,
    "weight_effective_sample_size": pl.Float64,
    "weight_adjusted_lower_bound": pl.Float64,
    "component_success_count": pl.UInt32,
    "component_trial_count": pl.UInt32,
    "component_exact_lower_bound": pl.Float64,
    "audited_precision_lower_bound": pl.Float64,
    "weighted_validation_coverage": pl.Float64,
    "minimum_item_count_passed": pl.Boolean,
    "minimum_component_count_passed": pl.Boolean,
    "precision_lower_bound_passed": pl.Boolean,
    "threshold_eligible": pl.Boolean,
    "selected": pl.Boolean,
}


@dataclass(frozen=True, slots=True)
class AuditedScreeningThresholdPolicy:
    """Preregistered validation-only precision and evidence requirements."""

    minimum_precision_lower_bound: float = 0.95
    confidence_level: float = 0.95
    minimum_selected_items: int = 30
    minimum_selected_components: int = 30

    def __post_init__(self) -> None:
        for field in ("minimum_precision_lower_bound", "confidence_level"):
            value = _open_probability(getattr(self, field), field=field)
            object.__setattr__(self, field, value)
        for field in ("minimum_selected_items", "minimum_selected_components"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": AUDITED_SCREENING_THRESHOLD_VERSION,
                "minimum_precision_lower_bound": (self.minimum_precision_lower_bound),
                "confidence_level": self.confidence_level,
                "minimum_selected_items": self.minimum_selected_items,
                "minimum_selected_components": self.minimum_selected_components,
                "threshold_candidates": "distinct_independent_validation_probabilities",
                "selection_objective": "maximum_weighted_validation_coverage",
                "lower_bound_method": LOWER_BOUND_METHOD,
                "component_success_semantics": (
                    "every_selected_row_in_component_human_supported"
                ),
                "authority": "screening_only_not_occurrence_release",
            }
        )


@dataclass(frozen=True, slots=True)
class AuditedScreeningThresholdSelection:
    schema_version: str
    status: str
    status_reason: str
    threshold: float | None
    weighted_precision: float | None
    audited_precision_lower_bound: float | None
    weighted_validation_coverage: float | None
    selected_item_count: int
    selected_component_count: int
    policy_fingerprint: str
    fit_fingerprint: str
    evidence_model_fingerprint: str
    calibrator_fingerprint: str
    split_fingerprint: str
    validation_prediction_artifact_fingerprint: str
    threshold_audit_fingerprint: str
    selection_fingerprint: str
    threshold_audit: pl.DataFrame
    screening_candidate_label: str
    fit_partition: str
    selection_partition: str
    final_test_prediction_count: int
    occurrence_release_authorized: bool


def select_audited_screening_threshold(
    fit: DynamicPoolCalibrationFit,
    policy: AuditedScreeningThresholdPolicy,
) -> AuditedScreeningThresholdSelection:
    """Select maximum validation coverage subject to a conservative precision LCB."""

    if not isinstance(fit, DynamicPoolCalibrationFit):
        raise TypeError("fit must be a DynamicPoolCalibrationFit")
    if not isinstance(policy, AuditedScreeningThresholdPolicy):
        raise TypeError("policy must be an AuditedScreeningThresholdPolicy")
    predictions = fit.predictions
    if predictions.schema != DYNAMIC_POOL_CALIBRATION_PREDICTION_SCHEMA:
        raise ValueError("dynamic-pool calibration prediction schema mismatch")
    if set(predictions["evaluation_split"].to_list()) != {
        "calibration",
        "validation",
    }:
        raise ValueError("threshold input must exclude final_test predictions")
    validation = predictions.filter(pl.col("evaluation_split") == "validation")
    if not validation.height:
        raise ValueError("threshold selection requires validation predictions")
    validation_prediction_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": "dynamic-pool-validation-predictions-v1.0.0",
            "source_prediction_artifact_fingerprint": (
                fit.prediction_artifact_fingerprint
            ),
            "rows": validation.to_dicts(),
        }
    )
    model_fingerprint = fit.evidence_model.model_fingerprint
    calibrator_fingerprint = fit.probability_calibration.calibration_fingerprint
    split_fingerprint = fit.evidence_model.split_fingerprint
    probabilities = tuple(
        sorted(
            {
                float(value)
                for value in validation[CALIBRATED_TARGET_PROBABILITY_KIND].to_list()
            }
        )
    )
    total_weight = float(validation["sampling_weight"].sum())
    validation_components = validation["independence_component_id"].n_unique()
    rows = []
    for threshold in probabilities:
        selected = validation.filter(
            pl.col(CALIBRATED_TARGET_PROBABILITY_KIND) >= threshold
        )
        selected_rows = selected.to_dicts()
        weights = [float(row["sampling_weight"]) for row in selected_rows]
        supported_rows = [row for row in selected_rows if row["human_supported"]]
        supported_weight = sum(float(row["sampling_weight"]) for row in supported_rows)
        selected_weight = sum(weights)
        precision = supported_weight / selected_weight
        effective_n = _kish_effective_sample_size(weights)
        weight_lower = _one_sided_wilson_lower_bound(
            precision,
            effective_n=effective_n,
            confidence_level=policy.confidence_level,
        )
        component_outcomes: dict[str, list[bool]] = defaultdict(list)
        for row in selected_rows:
            component_outcomes[str(row["independence_component_id"])].append(
                bool(row["human_supported"])
            )
        component_successes = sum(all(values) for values in component_outcomes.values())
        component_trials = len(component_outcomes)
        component_lower = clopper_pearson_lower_bound(
            component_successes,
            component_trials,
            confidence_level=policy.confidence_level,
        )
        audited_lower = min(weight_lower, component_lower)
        item_gate = selected.height >= policy.minimum_selected_items
        component_gate = component_trials >= policy.minimum_selected_components
        precision_gate = audited_lower >= policy.minimum_precision_lower_bound
        eligible = item_gate and component_gate and precision_gate
        base = {
            "policy_fingerprint": policy.fingerprint,
            "fit_fingerprint": fit.fit_fingerprint,
            "evidence_model_fingerprint": model_fingerprint,
            "calibrator_fingerprint": calibrator_fingerprint,
            "split_fingerprint": split_fingerprint,
            "validation_prediction_artifact_fingerprint": (
                validation_prediction_fingerprint
            ),
            "evaluation_split": "validation",
            "probability_kind": CALIBRATED_TARGET_PROBABILITY_KIND,
            "lower_bound_method": LOWER_BOUND_METHOD,
            "confidence_level": policy.confidence_level,
            "minimum_precision_lower_bound": (policy.minimum_precision_lower_bound),
            "threshold": threshold,
            "validation_item_count": validation.height,
            "validation_component_count": validation_components,
            "validation_weight": total_weight,
            "selected_item_count": selected.height,
            "selected_component_count": component_trials,
            "selected_weight": selected_weight,
            "supported_item_count": len(supported_rows),
            "error_item_count": selected.height - len(supported_rows),
            "supported_weight": supported_weight,
            "error_weight": selected_weight - supported_weight,
            "weighted_precision": precision,
            "weight_effective_sample_size": effective_n,
            "weight_adjusted_lower_bound": weight_lower,
            "component_success_count": component_successes,
            "component_trial_count": component_trials,
            "component_exact_lower_bound": component_lower,
            "audited_precision_lower_bound": audited_lower,
            "weighted_validation_coverage": selected_weight / total_weight,
            "minimum_item_count_passed": item_gate,
            "minimum_component_count_passed": component_gate,
            "precision_lower_bound_passed": precision_gate,
            "threshold_eligible": eligible,
        }
        rows.append(
            {
                "schema_version": (AUDITED_SCREENING_THRESHOLD_AUDIT_SCHEMA_VERSION),
                **base,
                "threshold_audit_fingerprint": "",
                "threshold_row_fingerprint": canonical_semantic_fingerprint(base),
                "selected": False,
            }
        )
    eligible_rows = [row for row in rows if row["threshold_eligible"]]
    selected_row = (
        max(
            eligible_rows,
            key=lambda row: (
                float(row["weighted_validation_coverage"]),
                float(row["audited_precision_lower_bound"]),
                -float(row["threshold"]),
            ),
        )
        if eligible_rows
        else None
    )
    if selected_row is not None:
        selected_row["selected"] = True
    audit_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": AUDITED_SCREENING_THRESHOLD_AUDIT_SCHEMA_VERSION,
            "policy_fingerprint": policy.fingerprint,
            "fit_fingerprint": fit.fit_fingerprint,
            "row_fingerprints": [row["threshold_row_fingerprint"] for row in rows],
            "selected_threshold": (
                None if selected_row is None else selected_row["threshold"]
            ),
        }
    )
    for row in rows:
        row["threshold_audit_fingerprint"] = audit_fingerprint
    audit = pl.DataFrame(
        rows,
        schema=AUDITED_SCREENING_THRESHOLD_AUDIT_SCHEMA,
        strict=True,
    ).sort("threshold")
    if selected_row is None:
        minimum_evidence_passed = any(
            row["minimum_item_count_passed"] and row["minimum_component_count_passed"]
            for row in rows
        )
        status = "infeasible"
        status_reason = (
            "no_threshold_satisfies_precision_lower_bound"
            if minimum_evidence_passed
            else "insufficient_independent_validation_evidence"
        )
        threshold = None
        precision = None
        lower_bound = None
        coverage = None
        selected_items = 0
        selected_components = 0
    else:
        status = "selected"
        status_reason = "precision_lower_bound_and_evidence_requirements_satisfied"
        threshold = float(selected_row["threshold"])
        precision = float(selected_row["weighted_precision"])
        lower_bound = float(selected_row["audited_precision_lower_bound"])
        coverage = float(selected_row["weighted_validation_coverage"])
        selected_items = int(selected_row["selected_item_count"])
        selected_components = int(selected_row["selected_component_count"])
    selection_semantics = {
        "schema_version": AUDITED_SCREENING_THRESHOLD_VERSION,
        "status": status,
        "status_reason": status_reason,
        "threshold": threshold,
        "weighted_precision": precision,
        "audited_precision_lower_bound": lower_bound,
        "weighted_validation_coverage": coverage,
        "selected_item_count": selected_items,
        "selected_component_count": selected_components,
        "policy_fingerprint": policy.fingerprint,
        "fit_fingerprint": fit.fit_fingerprint,
        "evidence_model_fingerprint": model_fingerprint,
        "calibrator_fingerprint": calibrator_fingerprint,
        "split_fingerprint": split_fingerprint,
        "validation_prediction_artifact_fingerprint": (
            validation_prediction_fingerprint
        ),
        "threshold_audit_fingerprint": audit_fingerprint,
        "screening_candidate_label": SCREENING_CANDIDATE_LABEL,
        "fit_partition": "calibration",
        "selection_partition": "validation",
        "final_test_prediction_count": 0,
        "occurrence_release_authorized": False,
    }
    return AuditedScreeningThresholdSelection(
        **selection_semantics,
        selection_fingerprint=canonical_semantic_fingerprint(selection_semantics),
        threshold_audit=audit,
    )


def _kish_effective_sample_size(weights: list[float]) -> float:
    total = sum(weights)
    squared = sum(weight * weight for weight in weights)
    if total <= 0.0 or squared <= 0.0:
        raise ValueError("threshold weights must be positive")
    return total * total / squared


def _one_sided_wilson_lower_bound(
    proportion: float,
    *,
    effective_n: float,
    confidence_level: float,
) -> float:
    if not 0.0 <= proportion <= 1.0 or effective_n <= 0.0:
        raise ValueError("Wilson bound inputs are invalid")
    z_score = NormalDist().inv_cdf(confidence_level)
    z_squared = z_score * z_score
    denominator = 1.0 + z_squared / effective_n
    center = (proportion + z_squared / (2.0 * effective_n)) / denominator
    half_width = (
        z_score
        * sqrt(
            proportion * (1.0 - proportion) / effective_n
            + z_squared / (4.0 * effective_n * effective_n)
        )
        / denominator
    )
    return max(0.0, center - half_width)


def _open_probability(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{field} must be in (0, 1)")
    return result


__all__ = [
    "AUDITED_SCREENING_THRESHOLD_AUDIT_SCHEMA",
    "AUDITED_SCREENING_THRESHOLD_AUDIT_SCHEMA_VERSION",
    "AUDITED_SCREENING_THRESHOLD_STATUSES",
    "AUDITED_SCREENING_THRESHOLD_VERSION",
    "LOWER_BOUND_METHOD",
    "SCREENING_CANDIDATE_LABEL",
    "AuditedScreeningThresholdPolicy",
    "AuditedScreeningThresholdSelection",
    "select_audited_screening_threshold",
]
