"""Weighted species-level evaluation of provisional reference banks."""

from __future__ import annotations

from math import isfinite, sqrt
from statistics import NormalDist

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss

from biominer.evaluation.reference_bank_audit import (
    AUDIT_DIMENSIONS,
    REFERENCE_BANK_QUALITY_AUDIT_SCHEMA,
    REFERENCE_BANK_QUALITY_SUMMARY_SCHEMA,
    ReferenceBankQualityPolicy,
)


HUMAN_REVIEWED_FLICKR_BASIS = "human_reviewed_flickr"
REPRESENTATIVE_CAMPAIGN = "representative_quality_audit"
WEIGHTED_WILSON_METHOD = "weighted_wilson_effective_sample_size"


def measure_reference_bank_performance(
    audit: pl.DataFrame,
    *,
    policy: ReferenceBankQualityPolicy | None = None,
) -> pl.DataFrame:
    """Measure each complete audit stratum or return an unavailable state."""

    selected_policy = policy or ReferenceBankQualityPolicy()
    _validate_audit_input(audit)
    if audit.is_empty():
        return pl.DataFrame(schema=REFERENCE_BANK_QUALITY_SUMMARY_SCHEMA)
    rows: list[dict[str, object]] = []
    for key, group in audit.group_by(list(AUDIT_DIMENSIONS), maintain_order=False):
        dimensions = dict(zip(AUDIT_DIMENSIONS, key, strict=True))
        rows.append(_measure_group(group, dimensions, selected_policy))
    return pl.DataFrame(
        rows,
        schema=REFERENCE_BANK_QUALITY_SUMMARY_SCHEMA,
        orient="row",
    ).sort(list(AUDIT_DIMENSIONS))


def _validate_audit_input(audit: pl.DataFrame) -> None:
    missing = sorted(set(REFERENCE_BANK_QUALITY_AUDIT_SCHEMA) - set(audit.columns))
    if missing:
        raise ValueError(f"reference-bank audit missing columns: {missing}")
    if audit["audit_record_id"].null_count() or audit["audit_record_id"].n_unique() != audit.height:
        raise ValueError("audit_record_id must be nonnull and unique")
    if audit.filter(pl.col("verification_basis") != HUMAN_REVIEWED_FLICKR_BASIS).height:
        raise ValueError("species audit may use only human-reviewed Flickr labels")
    if audit.filter(
        pl.col("inclusion_probability").is_null()
        | ~pl.col("inclusion_probability").is_finite()
        | ~pl.col("inclusion_probability").is_between(0.0, 1.0, closed="right")
    ).height:
        raise ValueError("inclusion_probability must be within (0, 1]")
    if audit.filter(
        pl.col("sampling_weight").is_not_null()
        & (
            ~pl.col("sampling_weight").is_finite()
            | (pl.col("sampling_weight") <= 0.0)
        )
    ).height:
        raise ValueError("sampling_weight must be positive and finite when present")
    if audit.filter(
        pl.col("prediction_abstained") & pl.col("predicted_target").is_not_null()
    ).height:
        raise ValueError("abstained rows must not carry a target decision")


def _measure_group(
    group: pl.DataFrame,
    dimensions: dict[str, object],
    policy: ReferenceBankQualityPolicy,
) -> dict[str, object]:
    targeted = group.filter(pl.col("sampling_campaign") != REPRESENTATIVE_CAMPAIGN)
    missing_targeted_weights = bool(
        targeted.height and targeted["sampling_weight"].null_count()
    )
    if missing_targeted_weights and policy.require_sampling_weights_for_targeted_queues:
        return _unavailable_row(
            dimensions,
            group.height,
            "unavailable_missing_sampling_weights",
        )
    if group.height < policy.minimum_group_sample_size:
        return _unavailable_row(dimensions, group.height, "insufficient_sample")

    weights = np.asarray(
        [
            float(weight) if weight is not None else 1.0 / float(probability)
            for weight, probability in zip(
                group["sampling_weight"].to_list(),
                group["inclusion_probability"].to_list(),
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    truth = np.asarray(group["human_target_supported"].to_list(), dtype=np.bool_)
    abstained = np.asarray(group["prediction_abstained"].to_list(), dtype=np.bool_)
    decisions = group["predicted_target"].to_list()
    predicted = np.asarray([value is True for value in decisions], dtype=np.bool_)
    retained = ~abstained
    tp = float(weights[retained & truth & predicted].sum())
    tn = float(weights[retained & ~truth & ~predicted].sum())
    fp = float(weights[retained & ~truth & predicted].sum())
    fn = float(weights[retained & truth & ~predicted].sum())
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    precision_interval = _weighted_wilson(tp, tp + fp, weights, policy.confidence_level)
    recall_interval = _weighted_wilson(tp, tp + fn, weights, policy.confidence_level)
    total_weight = float(weights.sum())
    retained_weight = float(weights[retained].sum())
    competitor = group["predicted_competitor_species"].to_list()
    expected_competitor = str(dimensions["competitor_species"])
    competitor_confusion = float(
        weights[
            retained
            & truth
            & ~predicted
            & np.asarray([value == expected_competitor for value in competitor])
        ].sum()
    )

    probabilities = group["calibrated_probability"].to_list()
    calibrators = group["calibrator_validated"].to_list()
    probability_available = all(value is not None for value in probabilities) and all(
        value is True for value in calibrators
    )
    margins = np.asarray(group["provisional_margin"].to_list(), dtype=np.float64)
    if not np.isfinite(margins).all():
        raise ValueError("provisional_margin must contain finite raw scores")
    ranking_score = (
        np.asarray(probabilities, dtype=np.float64)
        if probability_available
        else margins
    )
    pr_auc = (
        float(average_precision_score(truth, ranking_score, sample_weight=weights))
        if truth.any() and (~truth).any()
        else None
    )
    if probability_available:
        probability_array = np.asarray(probabilities, dtype=np.float64)
        if not np.isfinite(probability_array).all() or np.any(
            (probability_array < 0.0) | (probability_array > 1.0)
        ):
            raise ValueError("calibrated_probability must be finite within [0, 1]")
        brier = float(
            brier_score_loss(
                truth,
                probability_array,
                sample_weight=weights,
                pos_label=True,
                scale_by_half=True,
            )
        )
        ece = _expected_calibration_error(truth, probability_array, weights)
        margin_quantiles: tuple[float | None, float | None, float | None] = (
            None,
            None,
            None,
        )
    else:
        brier = None
        ece = None
        margin_quantiles = tuple(
            float(value)
            for value in np.quantile(margins, (0.05, 0.5, 0.95), method="linear")
        )
    return {
        **dimensions,
        "reviewed_record_count": group.height,
        "weighted_record_count": total_weight,
        "metric_status": "complete",
        "quality_approval_state": "eligible_for_policy_evaluation",
        "weights_applied": True,
        "confidence_interval_method": WEIGHTED_WILSON_METHOD,
        "precision": precision,
        "precision_ci_lower": precision_interval[0],
        "precision_ci_upper": precision_interval[1],
        "recall": recall,
        "recall_ci_lower": recall_interval[0],
        "recall_ci_upper": recall_interval[1],
        "false_positive_rate": _ratio(fp, fp + tn),
        "false_negative_rate": _ratio(fn, fn + tp),
        "pr_auc": pr_auc,
        "coverage": _ratio(retained_weight, total_weight),
        "abstention_rate": _ratio(total_weight - retained_weight, total_weight),
        "competitor_confusion_rate": _ratio(competitor_confusion, tp + fn),
        "probability_available": probability_available,
        "brier_score": brier,
        "expected_calibration_error": ece,
        "margin_q05": margin_quantiles[0],
        "margin_median": margin_quantiles[1],
        "margin_q95": margin_quantiles[2],
    }


def _unavailable_row(
    dimensions: dict[str, object],
    count: int,
    status: str,
) -> dict[str, object]:
    return {
        **dimensions,
        "reviewed_record_count": count,
        "weighted_record_count": None,
        "metric_status": status,
        "quality_approval_state": "unavailable",
        "weights_applied": False,
        "confidence_interval_method": WEIGHTED_WILSON_METHOD,
        **{
            name: None
            for name in (
                "precision",
                "precision_ci_lower",
                "precision_ci_upper",
                "recall",
                "recall_ci_lower",
                "recall_ci_upper",
                "false_positive_rate",
                "false_negative_rate",
                "pr_auc",
                "coverage",
                "abstention_rate",
                "competitor_confusion_rate",
                "brier_score",
                "expected_calibration_error",
                "margin_q05",
                "margin_median",
                "margin_q95",
            )
        },
        "probability_available": False,
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def _weighted_wilson(
    successes: float,
    total: float,
    weights: np.ndarray,
    confidence_level: float,
) -> tuple[float | None, float | None]:
    if total <= 0.0:
        return None, None
    effective_n = float(weights.sum() ** 2 / np.square(weights).sum())
    proportion = successes / total
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    denominator = 1.0 + z * z / effective_n
    center = (proportion + z * z / (2.0 * effective_n)) / denominator
    spread = (
        z
        * sqrt(
            proportion * (1.0 - proportion) / effective_n
            + z * z / (4.0 * effective_n * effective_n)
        )
        / denominator
    )
    return max(0.0, center - spread), min(1.0, center + spread)


def _expected_calibration_error(
    truth: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    total = float(weights.sum())
    error = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        mask = (probability >= lower) & (
            probability <= upper if upper >= 1.0 else probability < upper
        )
        if not mask.any():
            continue
        bin_weight = float(weights[mask].sum())
        accuracy = float(np.average(truth[mask], weights=weights[mask]))
        confidence = float(np.average(probability[mask], weights=weights[mask]))
        error += bin_weight / total * abs(accuracy - confidence)
    if not isfinite(error):
        raise ValueError("expected calibration error is not finite")
    return error


__all__ = [
    "HUMAN_REVIEWED_FLICKR_BASIS",
    "REPRESENTATIVE_CAMPAIGN",
    "WEIGHTED_WILSON_METHOD",
    "measure_reference_bank_performance",
]
