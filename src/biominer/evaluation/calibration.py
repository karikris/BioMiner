from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
import re
from statistics import NormalDist
from typing import Mapping, Sequence

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.ml.calibration import CALIBRATION_METHODS


TARGET_CALIBRATION_RELIABILITY_SCHEMA_VERSION = "target-calibration-reliability-v1.0.0"
TARGET_THRESHOLD_OPERATING_POINT_SCHEMA_VERSION = (
    "target-threshold-operating-point-v1.0.0"
)
CALIBRATED_TARGET_PROBABILITY_KIND = "calibrated_target_probability"
CALIBRATION_CONFIDENCE_INTERVAL_METHOD = "kish_effective_n_wilson_score"
DEFAULT_CALIBRATION_THRESHOLDS = (0.50, 0.70, 0.90, 0.95, 0.99)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")

TARGET_CALIBRATION_RELIABILITY_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "input_fingerprint": pl.String,
    "configuration_fingerprint": pl.String,
    "evaluation_set": pl.String,
    "probability_kind": pl.String,
    "calibration_method": pl.String,
    "calibration_split_fingerprint": pl.String,
    "calibrator_fingerprint": pl.String,
    "confidence_interval_method": pl.String,
    "confidence_level": pl.Float64,
    "bin_index": pl.UInt32,
    "bin_lower_bound": pl.Float64,
    "bin_upper_bound": pl.Float64,
    "evaluation_item_count": pl.UInt32,
    "probability_sample_count": pl.UInt32,
    "missing_probability_count": pl.UInt32,
    "weighted_evaluation_item_count": pl.Float64,
    "weighted_probability_sample_count": pl.Float64,
    "weighted_probability_coverage": pl.Float64,
    "item_count": pl.UInt32,
    "weighted_item_count": pl.Float64,
    "effective_sample_size": pl.Float64,
    "mean_predicted_probability": pl.Float64,
    "observed_target_rate": pl.Float64,
    "observed_rate_ci_lower": pl.Float64,
    "observed_rate_ci_upper": pl.Float64,
    "absolute_gap": pl.Float64,
    "ece_contribution": pl.Float64,
}

TARGET_THRESHOLD_OPERATING_POINT_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "input_fingerprint": pl.String,
    "configuration_fingerprint": pl.String,
    "evaluation_set": pl.String,
    "probability_kind": pl.String,
    "calibration_method": pl.String,
    "calibration_split_fingerprint": pl.String,
    "calibrator_fingerprint": pl.String,
    "confidence_interval_method": pl.String,
    "confidence_level": pl.Float64,
    "evaluation_item_count": pl.UInt32,
    "probability_sample_count": pl.UInt32,
    "missing_probability_count": pl.UInt32,
    "weighted_evaluation_item_count": pl.Float64,
    "weighted_probability_sample_count": pl.Float64,
    "weighted_probability_coverage": pl.Float64,
    "threshold": pl.Float64,
    "true_positive_count": pl.UInt32,
    "true_negative_count": pl.UInt32,
    "false_positive_count": pl.UInt32,
    "false_negative_count": pl.UInt32,
    "true_positive_weight": pl.Float64,
    "true_negative_weight": pl.Float64,
    "false_positive_weight": pl.Float64,
    "false_negative_weight": pl.Float64,
    "precision": pl.Float64,
    "precision_ci_lower": pl.Float64,
    "precision_ci_upper": pl.Float64,
    "precision_undefined_reason": pl.String,
    "recall": pl.Float64,
    "recall_ci_lower": pl.Float64,
    "recall_ci_upper": pl.Float64,
    "recall_undefined_reason": pl.String,
    "specificity": pl.Float64,
    "specificity_ci_lower": pl.Float64,
    "specificity_ci_upper": pl.Float64,
    "specificity_undefined_reason": pl.String,
    "false_positive_rate": pl.Float64,
    "false_negative_rate": pl.Float64,
}

_TARGET_CALIBRATION_INPUT_FIELDS = (
    "evaluation_item_id",
    "evaluation_set",
    "sampling_weight",
    "target_present",
    CALIBRATED_TARGET_PROBABILITY_KIND,
    "calibration_method",
    "calibration_split_fingerprint",
    "calibrator_fingerprint",
)


@dataclass(frozen=True, slots=True)
class TargetCalibrationDiagnostics:
    reliability: pl.DataFrame
    operating_points: pl.DataFrame
    input_fingerprint: str
    configuration_fingerprint: str
    diagnostics_fingerprint: str


def build_target_calibration_diagnostics(
    frame: pl.DataFrame,
    *,
    bin_count: int = 10,
    thresholds: Sequence[float] = DEFAULT_CALIBRATION_THRESHOLDS,
    confidence_level: float = 0.95,
) -> TargetCalibrationDiagnostics:
    """Build reliability and operating-point tables from calibrated probabilities."""

    normalized = _validated_target_calibration_input(frame)
    bins = _integer_at_least(bin_count, minimum=2, field="bin_count")
    operating_thresholds = _probability_values(thresholds, field="thresholds")
    confidence = _probability(
        confidence_level,
        field="confidence_level",
        open_lower=True,
    )
    if confidence >= 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    input_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": "target-calibration-input-v1.0.0",
            "rows": normalized.to_dicts(),
        }
    )
    configuration_fingerprint = canonical_semantic_fingerprint(
        {
            "reliability_schema_version": (
                TARGET_CALIBRATION_RELIABILITY_SCHEMA_VERSION
            ),
            "operating_point_schema_version": (
                TARGET_THRESHOLD_OPERATING_POINT_SCHEMA_VERSION
            ),
            "probability_kind": CALIBRATED_TARGET_PROBABILITY_KIND,
            "bin_count": bins,
            "thresholds": list(operating_thresholds),
            "confidence_level": confidence,
            "confidence_interval_method": CALIBRATION_CONFIDENCE_INTERVAL_METHOD,
        }
    )
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in normalized.iter_rows(named=True):
        grouped[
            (
                str(row["evaluation_set"]),
                str(row["calibration_method"]),
                str(row["calibration_split_fingerprint"]),
                str(row["calibrator_fingerprint"]),
            )
        ].append(row)

    reliability_rows: list[dict[str, object]] = []
    operating_rows: list[dict[str, object]] = []
    z_score = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    for identity in sorted(grouped):
        rows = grouped[identity]
        available = [
            row for row in rows if row[CALIBRATED_TARGET_PROBABILITY_KIND] is not None
        ]
        common = _calibration_group_base(
            rows,
            available,
            identity=identity,
            input_fingerprint=input_fingerprint,
            configuration_fingerprint=configuration_fingerprint,
            confidence_level=confidence,
        )
        probability_weight = float(common["weighted_probability_sample_count"])
        bin_groups: list[list[dict[str, object]]] = [[] for _ in range(bins)]
        for row in available:
            probability = float(row[CALIBRATED_TARGET_PROBABILITY_KIND])
            bin_groups[min(int(probability * bins), bins - 1)].append(row)
        for bin_index, selected in enumerate(bin_groups):
            selected_weights = [float(row["sampling_weight"]) for row in selected]
            weighted_count = sum(selected_weights)
            if selected:
                mean_probability = (
                    sum(
                        float(row["sampling_weight"])
                        * float(row[CALIBRATED_TARGET_PROBABILITY_KIND])
                        for row in selected
                    )
                    / weighted_count
                )
                target_weight = sum(
                    float(row["sampling_weight"])
                    for row in selected
                    if bool(row["target_present"])
                )
                observed = target_weight / weighted_count
                ci_lower, ci_upper = _weighted_wilson_interval(
                    target_weight,
                    weighted_count,
                    selected_weights,
                    z_score=z_score,
                )
                absolute_gap = abs(mean_probability - observed)
                contribution = (
                    weighted_count / probability_weight * absolute_gap
                    if probability_weight > 0.0
                    else None
                )
            else:
                mean_probability = None
                observed = None
                ci_lower = None
                ci_upper = None
                absolute_gap = None
                contribution = None
            reliability_rows.append(
                {
                    "schema_version": TARGET_CALIBRATION_RELIABILITY_SCHEMA_VERSION,
                    **common,
                    "bin_index": bin_index,
                    "bin_lower_bound": bin_index / bins,
                    "bin_upper_bound": (bin_index + 1) / bins,
                    "item_count": len(selected),
                    "weighted_item_count": weighted_count,
                    "effective_sample_size": _effective_sample_size(selected_weights),
                    "mean_predicted_probability": mean_probability,
                    "observed_target_rate": observed,
                    "observed_rate_ci_lower": ci_lower,
                    "observed_rate_ci_upper": ci_upper,
                    "absolute_gap": absolute_gap,
                    "ece_contribution": contribution,
                }
            )
        for threshold in operating_thresholds:
            operating_rows.append(
                _operating_point_row(
                    available,
                    common=common,
                    threshold=threshold,
                    z_score=z_score,
                )
            )

    reliability = pl.DataFrame(
        reliability_rows,
        schema=TARGET_CALIBRATION_RELIABILITY_SCHEMA,
        orient="row",
    ).sort(
        "evaluation_set",
        "calibration_method",
        "calibration_split_fingerprint",
        "calibrator_fingerprint",
        "bin_index",
    )
    operating_points = pl.DataFrame(
        operating_rows,
        schema=TARGET_THRESHOLD_OPERATING_POINT_SCHEMA,
        orient="row",
    ).sort(
        "evaluation_set",
        "calibration_method",
        "calibration_split_fingerprint",
        "calibrator_fingerprint",
        "threshold",
    )
    diagnostics_fingerprint = _target_calibration_diagnostics_fingerprint(
        reliability,
        operating_points,
        input_fingerprint=input_fingerprint,
        configuration_fingerprint=configuration_fingerprint,
    )
    result = TargetCalibrationDiagnostics(
        reliability=reliability,
        operating_points=operating_points,
        input_fingerprint=input_fingerprint,
        configuration_fingerprint=configuration_fingerprint,
        diagnostics_fingerprint=diagnostics_fingerprint,
    )
    validate_target_calibration_diagnostics(result)
    return result


def validate_target_calibration_diagnostics(
    diagnostics: TargetCalibrationDiagnostics,
) -> None:
    if not isinstance(diagnostics, TargetCalibrationDiagnostics):
        raise TypeError("diagnostics must be a TargetCalibrationDiagnostics")
    if dict(diagnostics.reliability.schema) != TARGET_CALIBRATION_RELIABILITY_SCHEMA:
        raise ValueError("target calibration reliability physical schema mismatch")
    if (
        dict(diagnostics.operating_points.schema)
        != TARGET_THRESHOLD_OPERATING_POINT_SCHEMA
    ):
        raise ValueError("target threshold operating-point physical schema mismatch")
    if diagnostics.reliability.is_empty() or diagnostics.operating_points.is_empty():
        raise ValueError("target calibration diagnostic tables must not be empty")
    expected_reliability = diagnostics.reliability.sort(
        "evaluation_set",
        "calibration_method",
        "calibration_split_fingerprint",
        "calibrator_fingerprint",
        "bin_index",
    )
    expected_points = diagnostics.operating_points.sort(
        "evaluation_set",
        "calibration_method",
        "calibration_split_fingerprint",
        "calibrator_fingerprint",
        "threshold",
    )
    if not diagnostics.reliability.equals(expected_reliability):
        raise ValueError("target calibration reliability rows are not sorted")
    if not diagnostics.operating_points.equals(expected_points):
        raise ValueError("target threshold operating points are not sorted")
    for value, field in (
        (diagnostics.input_fingerprint, "input_fingerprint"),
        (diagnostics.configuration_fingerprint, "configuration_fingerprint"),
        (diagnostics.diagnostics_fingerprint, "diagnostics_fingerprint"),
    ):
        _sha256(value, field=field)
    for table in (diagnostics.reliability, diagnostics.operating_points):
        if set(table["input_fingerprint"].to_list()) != {diagnostics.input_fingerprint}:
            raise ValueError("target calibration input_fingerprint is inconsistent")
        if set(table["configuration_fingerprint"].to_list()) != {
            diagnostics.configuration_fingerprint
        }:
            raise ValueError(
                "target calibration configuration_fingerprint is inconsistent"
            )
    for field in (
        "weighted_probability_coverage",
        "confidence_level",
    ):
        if diagnostics.reliability.filter(
            (pl.col(field) < 0.0) | (pl.col(field) > 1.0)
        ).height:
            raise ValueError(f"target calibration {field} is outside [0, 1]")
    expected_fingerprint = _target_calibration_diagnostics_fingerprint(
        diagnostics.reliability,
        diagnostics.operating_points,
        input_fingerprint=diagnostics.input_fingerprint,
        configuration_fingerprint=diagnostics.configuration_fingerprint,
    )
    if diagnostics.diagnostics_fingerprint != expected_fingerprint:
        raise ValueError("target calibration diagnostics_fingerprint is invalid")


def _validated_target_calibration_input(frame: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("target calibration input must be a Polars DataFrame")
    missing = sorted(set(_TARGET_CALIBRATION_INPUT_FIELDS) - set(frame.columns))
    if missing:
        raise ValueError(f"target calibration input is missing columns: {missing}")
    if frame.is_empty():
        raise ValueError("target calibration input must not be empty")
    normalized_rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    calibrator_identities: dict[str, tuple[str, str]] = {}
    for row in frame.select(_TARGET_CALIBRATION_INPUT_FIELDS).iter_rows(named=True):
        item_id = _required_text(row["evaluation_item_id"], field="evaluation_item_id")
        if item_id in seen_ids:
            raise ValueError(
                "target calibration evaluation_item_id values must be unique"
            )
        seen_ids.add(item_id)
        evaluation_set = _required_text(row["evaluation_set"], field="evaluation_set")
        weight = _positive_float(row["sampling_weight"], field="sampling_weight")
        target = row["target_present"]
        if not isinstance(target, bool):
            raise TypeError("target_present must contain Boolean values")
        probability_value = row[CALIBRATED_TARGET_PROBABILITY_KIND]
        probability = (
            None
            if probability_value is None
            else _probability(
                probability_value,
                field=CALIBRATED_TARGET_PROBABILITY_KIND,
            )
        )
        method = _required_text(row["calibration_method"], field="calibration_method")
        if method not in CALIBRATION_METHODS:
            raise ValueError(f"unsupported calibration_method: {method}")
        split = _sha256(
            row["calibration_split_fingerprint"],
            field="calibration_split_fingerprint",
        )
        calibrator = _sha256(
            row["calibrator_fingerprint"],
            field="calibrator_fingerprint",
        )
        identity = (method, split)
        previous_identity = calibrator_identities.setdefault(calibrator, identity)
        if previous_identity != identity:
            raise ValueError(
                "calibrator_fingerprint maps to conflicting method or split provenance"
            )
        normalized_rows.append(
            {
                "evaluation_item_id": item_id,
                "evaluation_set": evaluation_set,
                "sampling_weight": weight,
                "target_present": target,
                CALIBRATED_TARGET_PROBABILITY_KIND: probability,
                "calibration_method": method,
                "calibration_split_fingerprint": split,
                "calibrator_fingerprint": calibrator,
            }
        )
    schema = {
        "evaluation_item_id": pl.String,
        "evaluation_set": pl.String,
        "sampling_weight": pl.Float64,
        "target_present": pl.Boolean,
        CALIBRATED_TARGET_PROBABILITY_KIND: pl.Float64,
        "calibration_method": pl.String,
        "calibration_split_fingerprint": pl.String,
        "calibrator_fingerprint": pl.String,
    }
    return pl.DataFrame(normalized_rows, schema=schema, orient="row").sort(
        "evaluation_set", "evaluation_item_id"
    )


def _calibration_group_base(
    rows: Sequence[Mapping[str, object]],
    available: Sequence[Mapping[str, object]],
    *,
    identity: tuple[str, str, str, str],
    input_fingerprint: str,
    configuration_fingerprint: str,
    confidence_level: float,
) -> dict[str, object]:
    total_weight = sum(float(row["sampling_weight"]) for row in rows)
    probability_weight = sum(float(row["sampling_weight"]) for row in available)
    evaluation_set, method, split, calibrator = identity
    return {
        "input_fingerprint": input_fingerprint,
        "configuration_fingerprint": configuration_fingerprint,
        "evaluation_set": evaluation_set,
        "probability_kind": CALIBRATED_TARGET_PROBABILITY_KIND,
        "calibration_method": method,
        "calibration_split_fingerprint": split,
        "calibrator_fingerprint": calibrator,
        "confidence_interval_method": CALIBRATION_CONFIDENCE_INTERVAL_METHOD,
        "confidence_level": confidence_level,
        "evaluation_item_count": len(rows),
        "probability_sample_count": len(available),
        "missing_probability_count": len(rows) - len(available),
        "weighted_evaluation_item_count": total_weight,
        "weighted_probability_sample_count": probability_weight,
        "weighted_probability_coverage": (
            probability_weight / total_weight if total_weight > 0.0 else None
        ),
    }


def _operating_point_row(
    rows: Sequence[Mapping[str, object]],
    *,
    common: Mapping[str, object],
    threshold: float,
    z_score: float,
) -> dict[str, object]:
    tp_rows: list[Mapping[str, object]] = []
    tn_rows: list[Mapping[str, object]] = []
    fp_rows: list[Mapping[str, object]] = []
    fn_rows: list[Mapping[str, object]] = []
    for row in rows:
        target = bool(row["target_present"])
        predicted = float(row[CALIBRATED_TARGET_PROBABILITY_KIND]) >= threshold
        if target and predicted:
            tp_rows.append(row)
        elif target:
            fn_rows.append(row)
        elif predicted:
            fp_rows.append(row)
        else:
            tn_rows.append(row)
    tp = _row_weight(tp_rows)
    tn = _row_weight(tn_rows)
    fp = _row_weight(fp_rows)
    fn = _row_weight(fn_rows)
    precision, precision_low, precision_high, precision_reason = _rate_interval(
        tp,
        tp + fp,
        [float(row["sampling_weight"]) for row in (*tp_rows, *fp_rows)],
        z_score=z_score,
    )
    recall, recall_low, recall_high, recall_reason = _rate_interval(
        tp,
        tp + fn,
        [float(row["sampling_weight"]) for row in (*tp_rows, *fn_rows)],
        z_score=z_score,
    )
    specificity, specificity_low, specificity_high, specificity_reason = _rate_interval(
        tn,
        tn + fp,
        [float(row["sampling_weight"]) for row in (*tn_rows, *fp_rows)],
        z_score=z_score,
    )
    return {
        "schema_version": TARGET_THRESHOLD_OPERATING_POINT_SCHEMA_VERSION,
        **common,
        "threshold": threshold,
        "true_positive_count": len(tp_rows),
        "true_negative_count": len(tn_rows),
        "false_positive_count": len(fp_rows),
        "false_negative_count": len(fn_rows),
        "true_positive_weight": tp,
        "true_negative_weight": tn,
        "false_positive_weight": fp,
        "false_negative_weight": fn,
        "precision": precision,
        "precision_ci_lower": precision_low,
        "precision_ci_upper": precision_high,
        "precision_undefined_reason": precision_reason,
        "recall": recall,
        "recall_ci_lower": recall_low,
        "recall_ci_upper": recall_high,
        "recall_undefined_reason": recall_reason,
        "specificity": specificity,
        "specificity_ci_lower": specificity_low,
        "specificity_ci_upper": specificity_high,
        "specificity_undefined_reason": specificity_reason,
        "false_positive_rate": _ratio(fp, fp + tn),
        "false_negative_rate": _ratio(fn, fn + tp),
    }


def _rate_interval(
    numerator: float,
    denominator: float,
    weights: Sequence[float],
    *,
    z_score: float,
) -> tuple[float | None, float | None, float | None, str | None]:
    if denominator <= 0.0:
        return None, None, None, "zero_denominator"
    lower, upper = _weighted_wilson_interval(
        numerator,
        denominator,
        weights,
        z_score=z_score,
    )
    return numerator / denominator, lower, upper, None


def _weighted_wilson_interval(
    success_weight: float,
    total_weight: float,
    weights: Sequence[float],
    *,
    z_score: float,
) -> tuple[float | None, float | None]:
    effective_n = _effective_sample_size(weights)
    if total_weight <= 0.0 or effective_n <= 0.0:
        return None, None
    proportion = min(1.0, max(0.0, success_weight / total_weight))
    z_squared = z_score * z_score
    denominator = 1.0 + z_squared / effective_n
    center = (proportion + z_squared / (2.0 * effective_n)) / denominator
    half_width = (
        z_score
        * (
            proportion * (1.0 - proportion) / effective_n
            + z_squared / (4.0 * effective_n * effective_n)
        )
        ** 0.5
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _effective_sample_size(weights: Sequence[float]) -> float:
    total = sum(weights)
    squared = sum(weight * weight for weight in weights)
    return total * total / squared if total > 0.0 and squared > 0.0 else 0.0


def _row_weight(rows: Sequence[Mapping[str, object]]) -> float:
    return sum(float(row["sampling_weight"]) for row in rows)


def _target_calibration_diagnostics_fingerprint(
    reliability: pl.DataFrame,
    operating_points: pl.DataFrame,
    *,
    input_fingerprint: str,
    configuration_fingerprint: str,
) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema_version": "target-calibration-diagnostics-v1.0.0",
            "input_fingerprint": input_fingerprint,
            "configuration_fingerprint": configuration_fingerprint,
            "reliability_rows": reliability.to_dicts(),
            "operating_point_rows": operating_points.to_dicts(),
        }
    )


def _integer_at_least(value: object, *, minimum: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _probability_values(values: Sequence[float], *, field: str) -> tuple[float, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError(f"{field} must be a sequence")
    normalized = tuple(sorted({_probability(value, field=field) for value in values}))
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def _probability(
    value: object,
    *,
    field: str,
    open_lower: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must contain numeric probabilities")
    result = float(value)
    lower_valid = result > 0.0 if open_lower else result >= 0.0
    if not isfinite(result) or not lower_valid or result > 1.0:
        interval = "(0, 1]" if open_lower else "[0, 1]"
        raise ValueError(f"{field} must contain values in {interval}")
    return result


def _positive_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must contain numeric values")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must contain positive finite values")
    return result


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must contain canonical nonblank text")
    return value


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full lowercase sha256 fingerprint")
    return text


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


__all__ = [
    "CALIBRATED_TARGET_PROBABILITY_KIND",
    "CALIBRATION_CONFIDENCE_INTERVAL_METHOD",
    "DEFAULT_CALIBRATION_THRESHOLDS",
    "TARGET_CALIBRATION_RELIABILITY_SCHEMA",
    "TARGET_CALIBRATION_RELIABILITY_SCHEMA_VERSION",
    "TARGET_THRESHOLD_OPERATING_POINT_SCHEMA",
    "TARGET_THRESHOLD_OPERATING_POINT_SCHEMA_VERSION",
    "TargetCalibrationDiagnostics",
    "build_target_calibration_diagnostics",
    "validate_target_calibration_diagnostics",
]
