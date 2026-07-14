"""Deterministic whole-component bootstrap intervals for evaluation metrics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from math import ceil, isfinite
import re

import numpy as np
import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.leakage import (
    EVALUATION_PARTITIONS,
    EVALUATION_IDENTITY_COMPONENT_SCHEMA,
    EVALUATION_IDENTITY_COMPONENT_SCHEMA_VERSION,
)


TARGET_METRIC_CONFIDENCE_INTERVAL_SCHEMA_VERSION = (
    "target-metric-confidence-interval-v1.0.0"
)
TARGET_METRIC_CONFIDENCE_INTERVAL_FILE = "target_metric_confidence_intervals.parquet"
GROUPED_BOOTSTRAP_METHOD = "within-evaluation-set-identity-component-percentile"
IMPROVEMENT_CLAIM_POLICY = "point_estimates_alone_do_not_establish_improvement"
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")

TARGET_METRIC_CONFIDENCE_INTERVAL_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "input_fingerprint": pl.String,
    "metric_configuration_fingerprint": pl.String,
    "bootstrap_configuration_fingerprint": pl.String,
    "leakage_register_fingerprint": pl.String,
    "evaluation_set": pl.String,
    "metric_name": pl.String,
    "point_estimate": pl.Float64,
    "confidence_interval_lower": pl.Float64,
    "confidence_interval_upper": pl.Float64,
    "confidence_level": pl.Float64,
    "interval_method": pl.String,
    "interval_status": pl.String,
    "undefined_reason": pl.String,
    "bootstrap_replicates": pl.UInt32,
    "minimum_valid_replicates": pl.UInt32,
    "valid_replicates": pl.UInt32,
    "undefined_replicates": pl.UInt32,
    "independent_component_count": pl.UInt32,
    "minimum_component_size": pl.UInt32,
    "maximum_component_size": pl.UInt32,
    "random_seed": pl.UInt64,
    "improvement_claim_policy": pl.String,
}


@dataclass(frozen=True, slots=True)
class GroupedBootstrapConfig:
    replicate_count: int = 2_000
    confidence_level: float = 0.95
    random_seed: int = 20_260_714
    minimum_valid_fraction: float = 0.80
    minimum_component_count: int = 2

    def __post_init__(self) -> None:
        if (
            isinstance(self.replicate_count, bool)
            or not isinstance(self.replicate_count, int)
            or self.replicate_count < 2
        ):
            raise ValueError("replicate_count must be an integer >= 2")
        for value, field in (
            (self.confidence_level, "confidence_level"),
            (self.minimum_valid_fraction, "minimum_valid_fraction"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not isfinite(float(value))
                or not 0.0 < float(value) < 1.0
            ):
                raise ValueError(f"{field} must be in (0, 1)")
            object.__setattr__(self, field, float(value))
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or not 0 <= self.random_seed < 2**64
        ):
            raise ValueError("random_seed must be an unsigned 64-bit integer")
        if (
            isinstance(self.minimum_component_count, bool)
            or not isinstance(self.minimum_component_count, int)
            or self.minimum_component_count < 2
        ):
            raise ValueError("minimum_component_count must be an integer >= 2")

    @property
    def minimum_valid_replicates(self) -> int:
        return ceil(self.replicate_count * self.minimum_valid_fraction)

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": TARGET_METRIC_CONFIDENCE_INTERVAL_SCHEMA_VERSION,
                "interval_method": GROUPED_BOOTSTRAP_METHOD,
                "replicate_count": self.replicate_count,
                "confidence_level": self.confidence_level,
                "random_seed": self.random_seed,
                "minimum_valid_fraction": self.minimum_valid_fraction,
                "minimum_component_count": self.minimum_component_count,
                "improvement_claim_policy": IMPROVEMENT_CLAIM_POLICY,
            }
        )


@dataclass(frozen=True, slots=True)
class GroupedBootstrapResult:
    intervals: pl.DataFrame
    components: pl.DataFrame
    bootstrap_configuration_fingerprint: str
    uncertainty_fingerprint: str


MetricEvaluator = Callable[[pl.DataFrame], Mapping[str, float | None]]


def empty_target_metric_confidence_intervals() -> pl.DataFrame:
    return pl.DataFrame(schema=TARGET_METRIC_CONFIDENCE_INTERVAL_SCHEMA)


def build_grouped_metric_confidence_intervals(
    frame: pl.DataFrame,
    components: pl.DataFrame,
    *,
    metric_names: Sequence[str],
    point_estimates: Mapping[tuple[str, str], float | None],
    metric_evaluator: MetricEvaluator,
    input_fingerprint: str,
    metric_configuration_fingerprint: str,
    config: GroupedBootstrapConfig | None = None,
) -> GroupedBootstrapResult:
    """Resample whole identity components independently within each holdout."""

    active = config or GroupedBootstrapConfig()
    if not isinstance(active, GroupedBootstrapConfig):
        raise TypeError("config must be a GroupedBootstrapConfig")
    input_hash = _sha256(input_fingerprint, field="input_fingerprint")
    metric_hash = _sha256(
        metric_configuration_fingerprint,
        field="metric_configuration_fingerprint",
    )
    names = tuple(
        sorted({_required_text(name, field="metric_names") for name in metric_names})
    )
    if not names:
        raise ValueError("metric_names must not be empty")
    if not callable(metric_evaluator):
        raise TypeError("metric_evaluator must be callable")
    normalized_frame = _validate_bootstrap_frame(frame)
    normalized_components = _validate_components(components)
    _validate_component_coverage(normalized_frame, normalized_components)
    evaluation_sets = sorted(set(normalized_frame["evaluation_set"].to_list()))
    expected_point_keys = {
        (evaluation_set, metric_name)
        for evaluation_set in evaluation_sets
        for metric_name in names
    }
    if set(point_estimates) != expected_point_keys:
        raise ValueError(
            "point_estimates do not exactly cover evaluation sets and metrics"
        )

    interval_rows: list[dict[str, object]] = []
    register_fingerprint = _single_text(
        normalized_components,
        "register_fingerprint",
    )
    for evaluation_set in evaluation_sets:
        evaluation = normalized_frame.filter(
            pl.col("evaluation_set") == evaluation_set
        ).sort("evaluation_item_id")
        assignments = normalized_components.filter(
            pl.col("partition") == evaluation_set
        ).sort("item_id")
        component_ids = sorted(set(assignments["bootstrap_component_id"].to_list()))
        if len(component_ids) < active.minimum_component_count:
            raise ValueError(
                f"{evaluation_set} requires at least "
                f"{active.minimum_component_count} independent components"
            )
        component_index = {
            component_id: index for index, component_id in enumerate(component_ids)
        }
        component_by_item = {
            str(row["item_id"]): component_index[str(row["bootstrap_component_id"])]
            for row in assignments.iter_rows(named=True)
        }
        row_components = np.asarray(
            [
                component_by_item[str(item_id)]
                for item_id in evaluation["evaluation_item_id"]
            ],
            dtype=np.int64,
        )
        original_weights = np.asarray(
            evaluation["sampling_weight"].to_list(),
            dtype=np.float64,
        )
        draws: dict[str, list[float]] = {name: [] for name in names}
        rng = np.random.default_rng(
            _evaluation_seed(active.random_seed, evaluation_set)
        )
        probabilities = np.full(
            len(component_ids),
            1.0 / len(component_ids),
            dtype=np.float64,
        )
        for _replicate in range(active.replicate_count):
            component_counts = rng.multinomial(len(component_ids), probabilities)
            row_counts = component_counts[row_components]
            selected = row_counts > 0
            sample = evaluation.filter(pl.Series(selected)).with_columns(
                pl.Series(
                    "sampling_weight",
                    original_weights[selected] * row_counts[selected],
                    dtype=pl.Float64,
                )
            )
            values = metric_evaluator(sample)
            if not isinstance(values, Mapping):
                raise TypeError("metric_evaluator must return a mapping")
            missing_metrics = sorted(set(names) - set(values))
            if missing_metrics:
                raise ValueError(f"metric_evaluator omitted metrics: {missing_metrics}")
            for name in names:
                value = values[name]
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise TypeError(f"metric_evaluator returned nonnumeric {name}")
                numeric = float(value)
                if isfinite(numeric):
                    draws[name].append(numeric)
        component_sizes = (
            assignments.group_by("bootstrap_component_id")
            .len()
            .sort("bootstrap_component_id")["len"]
            .to_list()
        )
        for name in names:
            point = _optional_finite_float(
                point_estimates[(evaluation_set, name)],
                field=f"point_estimates[{evaluation_set},{name}]",
            )
            valid = np.asarray(draws[name], dtype=np.float64)
            valid_count = int(valid.size)
            undefined_count = active.replicate_count - valid_count
            if point is None:
                lower = None
                upper = None
                status = "point_estimate_undefined"
                reason = "point_estimate_undefined"
            elif valid_count < active.minimum_valid_replicates:
                lower = None
                upper = None
                status = "insufficient_valid_replicates"
                reason = "insufficient_valid_replicates"
            else:
                alpha = 1.0 - active.confidence_level
                lower, upper = (
                    float(value)
                    for value in np.quantile(
                        valid,
                        (alpha / 2.0, 1.0 - alpha / 2.0),
                        method="linear",
                    )
                )
                status = "complete"
                reason = None
            interval_rows.append(
                {
                    "schema_version": TARGET_METRIC_CONFIDENCE_INTERVAL_SCHEMA_VERSION,
                    "input_fingerprint": input_hash,
                    "metric_configuration_fingerprint": metric_hash,
                    "bootstrap_configuration_fingerprint": active.fingerprint,
                    "leakage_register_fingerprint": register_fingerprint,
                    "evaluation_set": evaluation_set,
                    "metric_name": name,
                    "point_estimate": point,
                    "confidence_interval_lower": lower,
                    "confidence_interval_upper": upper,
                    "confidence_level": active.confidence_level,
                    "interval_method": GROUPED_BOOTSTRAP_METHOD,
                    "interval_status": status,
                    "undefined_reason": reason,
                    "bootstrap_replicates": active.replicate_count,
                    "minimum_valid_replicates": active.minimum_valid_replicates,
                    "valid_replicates": valid_count,
                    "undefined_replicates": undefined_count,
                    "independent_component_count": len(component_ids),
                    "minimum_component_size": min(component_sizes),
                    "maximum_component_size": max(component_sizes),
                    "random_seed": active.random_seed,
                    "improvement_claim_policy": IMPROVEMENT_CLAIM_POLICY,
                }
            )
    intervals = pl.DataFrame(
        interval_rows,
        schema=TARGET_METRIC_CONFIDENCE_INTERVAL_SCHEMA,
        orient="row",
    ).sort("evaluation_set", "metric_name")
    uncertainty_fingerprint = _uncertainty_fingerprint(
        intervals,
        normalized_components,
        bootstrap_configuration_fingerprint=active.fingerprint,
    )
    result = GroupedBootstrapResult(
        intervals=intervals,
        components=normalized_components,
        bootstrap_configuration_fingerprint=active.fingerprint,
        uncertainty_fingerprint=uncertainty_fingerprint,
    )
    validate_grouped_bootstrap_result(result)
    return result


def validate_grouped_bootstrap_result(result: GroupedBootstrapResult) -> None:
    if not isinstance(result, GroupedBootstrapResult):
        raise TypeError("result must be a GroupedBootstrapResult")
    if dict(result.intervals.schema) != TARGET_METRIC_CONFIDENCE_INTERVAL_SCHEMA:
        raise ValueError("target metric confidence-interval physical schema mismatch")
    if result.intervals.is_empty():
        raise ValueError("target metric confidence intervals must not be empty")
    if not result.intervals.equals(
        result.intervals.sort("evaluation_set", "metric_name")
    ):
        raise ValueError("target metric confidence intervals are not sorted")
    keys = list(
        zip(
            result.intervals["evaluation_set"].to_list(),
            result.intervals["metric_name"].to_list(),
            strict=True,
        )
    )
    if len(keys) != len(set(keys)):
        raise ValueError("target metric confidence-interval keys must be unique")
    normalized_components = _validate_components(result.components)
    if not result.components.equals(normalized_components):
        raise ValueError("bootstrap components are not sorted")
    _sha256(
        result.bootstrap_configuration_fingerprint,
        field="bootstrap_configuration_fingerprint",
    )
    _sha256(result.uncertainty_fingerprint, field="uncertainty_fingerprint")
    if set(result.intervals["bootstrap_configuration_fingerprint"].to_list()) != {
        result.bootstrap_configuration_fingerprint
    }:
        raise ValueError("bootstrap configuration fingerprint is inconsistent")
    _validate_interval_semantics(result.intervals, result.components)
    expected = _uncertainty_fingerprint(
        result.intervals,
        result.components,
        bootstrap_configuration_fingerprint=(
            result.bootstrap_configuration_fingerprint
        ),
    )
    if result.uncertainty_fingerprint != expected:
        raise ValueError("uncertainty_fingerprint is invalid")


def _validate_bootstrap_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a Polars DataFrame")
    required = {"evaluation_item_id", "evaluation_set", "sampling_weight"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"bootstrap frame is missing columns: {missing}")
    if frame.is_empty():
        raise ValueError("bootstrap frame must not be empty")
    normalized = frame.sort("evaluation_set", "evaluation_item_id")
    if normalized["evaluation_item_id"].n_unique() != normalized.height:
        raise ValueError("bootstrap evaluation_item_id values must be unique")
    for field in ("evaluation_item_id", "evaluation_set"):
        if normalized.filter(
            pl.col(field).is_null()
            | (pl.col(field).str.strip_chars() == "")
            | (pl.col(field) != pl.col(field).str.strip_chars())
        ).height:
            raise ValueError(f"bootstrap {field} must contain canonical text")
    invalid_sets = sorted(
        set(normalized["evaluation_set"].to_list()) - EVALUATION_PARTITIONS
    )
    if invalid_sets:
        raise ValueError(f"unsupported bootstrap evaluation sets: {invalid_sets}")
    if normalized.filter(
        pl.col("sampling_weight").is_null()
        | ~pl.col("sampling_weight").is_finite()
        | (pl.col("sampling_weight") <= 0.0)
    ).height:
        raise ValueError("bootstrap sampling_weight must be positive and finite")
    return normalized


def _validate_components(components: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(components, pl.DataFrame):
        raise TypeError("components must be a Polars DataFrame")
    if dict(components.schema) != EVALUATION_IDENTITY_COMPONENT_SCHEMA:
        raise ValueError("evaluation identity-component physical schema mismatch")
    if components.is_empty():
        raise ValueError("evaluation identity components must not be empty")
    normalized = components.sort("partition", "bootstrap_component_id", "item_id")
    if normalized["item_id"].n_unique() != normalized.height:
        raise ValueError("evaluation identity-component item IDs must be unique")
    for field in ("partition", "bootstrap_component_id", "item_id"):
        if normalized.filter(
            pl.col(field).is_null()
            | (pl.col(field).str.strip_chars() == "")
            | (pl.col(field) != pl.col(field).str.strip_chars())
        ).height:
            raise ValueError(f"evaluation identity-component {field} is invalid")
    invalid_partitions = sorted(
        set(normalized["partition"].to_list()) - EVALUATION_PARTITIONS
    )
    if invalid_partitions:
        raise ValueError(
            f"unsupported identity-component partitions: {invalid_partitions}"
        )
    if set(normalized["schema_version"].to_list()) != {
        EVALUATION_IDENTITY_COMPONENT_SCHEMA_VERSION
    }:
        raise ValueError("evaluation identity-component schema is incompatible")
    register_fingerprint = _single_text(normalized, "register_fingerprint")
    _sha256(register_fingerprint, field="register_fingerprint")
    for component_id in sorted(set(normalized["bootstrap_component_id"].to_list())):
        selected = normalized.filter(pl.col("bootstrap_component_id") == component_id)
        partition = _single_text(selected, "partition")
        item_ids = selected["item_id"].sort().to_list()
        expected_id = canonical_semantic_fingerprint(
            {
                "schema_version": EVALUATION_IDENTITY_COMPONENT_SCHEMA_VERSION,
                "register_fingerprint": register_fingerprint,
                "partition": partition,
                "item_ids": item_ids,
            }
        )
        if component_id != expected_id:
            raise ValueError("bootstrap_component_id is invalid")
        if set(selected["component_size"].to_list()) != {selected.height}:
            raise ValueError("bootstrap component_size is inconsistent")
    return normalized


def _validate_interval_semantics(
    intervals: pl.DataFrame,
    components: pl.DataFrame,
) -> None:
    if set(intervals["schema_version"].to_list()) != {
        TARGET_METRIC_CONFIDENCE_INTERVAL_SCHEMA_VERSION
    }:
        raise ValueError("target metric confidence-interval schema is incompatible")
    for field in (
        "input_fingerprint",
        "metric_configuration_fingerprint",
        "bootstrap_configuration_fingerprint",
        "leakage_register_fingerprint",
    ):
        _sha256(_single_text(intervals, field), field=field)
    if _single_text(intervals, "leakage_register_fingerprint") != _single_text(
        components,
        "register_fingerprint",
    ):
        raise ValueError("confidence intervals reference a different leakage register")
    for field, expected in (
        ("interval_method", GROUPED_BOOTSTRAP_METHOD),
        ("improvement_claim_policy", IMPROVEMENT_CLAIM_POLICY),
    ):
        if set(intervals[field].to_list()) != {expected}:
            raise ValueError(f"confidence-interval {field} is invalid")
    interval_sets = set(intervals["evaluation_set"].to_list())
    component_sets = set(components["partition"].to_list())
    if interval_sets != component_sets:
        raise ValueError("confidence-interval evaluation-set coverage is inconsistent")
    metric_sets = {
        evaluation_set: set(
            intervals.filter(pl.col("evaluation_set") == evaluation_set)[
                "metric_name"
            ].to_list()
        )
        for evaluation_set in sorted(interval_sets)
    }
    if len({frozenset(values) for values in metric_sets.values()}) != 1:
        raise ValueError(
            "confidence-interval metric coverage differs by evaluation set"
        )
    statuses = {
        "complete",
        "point_estimate_undefined",
        "insufficient_valid_replicates",
    }
    for row in intervals.iter_rows(named=True):
        evaluation_set = _required_text(
            row["evaluation_set"],
            field="evaluation_set",
        )
        _required_text(row["metric_name"], field="metric_name")
        if evaluation_set not in EVALUATION_PARTITIONS:
            raise ValueError(f"unsupported evaluation set: {evaluation_set}")
        status = row["interval_status"]
        if status not in statuses:
            raise ValueError(f"unsupported confidence-interval status: {status}")
        confidence = _bounded_probability(
            row["confidence_level"],
            field="confidence_level",
        )
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        replicates = int(row["bootstrap_replicates"])
        minimum_valid = int(row["minimum_valid_replicates"])
        valid = int(row["valid_replicates"])
        undefined = int(row["undefined_replicates"])
        if replicates < 2 or not 1 <= minimum_valid <= replicates:
            raise ValueError("bootstrap replicate thresholds are invalid")
        if valid + undefined != replicates:
            raise ValueError("bootstrap replicate accounting is inconsistent")
        assignments = components.filter(pl.col("partition") == evaluation_set)
        component_sizes = (
            assignments.group_by("bootstrap_component_id").len()["len"].to_list()
        )
        expected_component_count = len(component_sizes)
        if (
            expected_component_count < 2
            or int(row["independent_component_count"]) != expected_component_count
            or int(row["minimum_component_size"]) != min(component_sizes)
            or int(row["maximum_component_size"]) != max(component_sizes)
        ):
            raise ValueError("bootstrap component accounting is inconsistent")
        point = _optional_finite_float(row["point_estimate"], field="point_estimate")
        lower = _optional_finite_float(
            row["confidence_interval_lower"],
            field="confidence_interval_lower",
        )
        upper = _optional_finite_float(
            row["confidence_interval_upper"],
            field="confidence_interval_upper",
        )
        reason = row["undefined_reason"]
        if status == "complete":
            if (
                point is None
                or lower is None
                or upper is None
                or lower > upper
                or valid < minimum_valid
                or reason is not None
            ):
                raise ValueError("complete confidence interval is inconsistent")
        elif status == "point_estimate_undefined":
            if (
                point is not None
                or lower is not None
                or upper is not None
                or reason != status
            ):
                raise ValueError("undefined point-estimate interval is inconsistent")
        elif (
            point is None
            or lower is not None
            or upper is not None
            or valid >= minimum_valid
            or reason != status
        ):
            raise ValueError("insufficient-replicate interval is inconsistent")


def _bounded_probability(value: object, *, field: str) -> float:
    result = _optional_finite_float(value, field=field)
    if result is None or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return result


def _validate_component_coverage(
    frame: pl.DataFrame,
    components: pl.DataFrame,
) -> None:
    frame_ids = set(frame["evaluation_item_id"].to_list())
    component_ids = set(components["item_id"].to_list())
    if frame_ids != component_ids:
        raise ValueError(
            "bootstrap component coverage mismatch: "
            f"missing={sorted(frame_ids - component_ids)[:10]}, "
            f"unexpected={sorted(component_ids - frame_ids)[:10]}"
        )
    partition_by_item = {
        str(row["item_id"]): str(row["partition"])
        for row in components.iter_rows(named=True)
    }
    for row in frame.iter_rows(named=True):
        item_id = str(row["evaluation_item_id"])
        if str(row["evaluation_set"]) != partition_by_item[item_id]:
            raise ValueError(f"bootstrap item {item_id} has a partition mismatch")


def _uncertainty_fingerprint(
    intervals: pl.DataFrame,
    components: pl.DataFrame,
    *,
    bootstrap_configuration_fingerprint: str,
) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema_version": TARGET_METRIC_CONFIDENCE_INTERVAL_SCHEMA_VERSION,
            "bootstrap_configuration_fingerprint": (
                bootstrap_configuration_fingerprint
            ),
            "interval_rows": intervals.to_dicts(),
            "component_rows": components.to_dicts(),
        }
    )


def _evaluation_seed(seed: int, evaluation_set: str) -> int:
    digest = hashlib.sha256(f"{seed}:{evaluation_set}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _optional_finite_float(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric or null")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite or null")
    return result


def _single_text(frame: pl.DataFrame, field: str) -> str:
    values = frame[field].unique().to_list()
    if len(values) != 1:
        raise ValueError(f"{field} must have one consistent value")
    return _required_text(values[0], field=field)


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must contain canonical nonblank text")
    return value


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full lowercase sha256 fingerprint")
    return text


__all__ = [
    "GROUPED_BOOTSTRAP_METHOD",
    "IMPROVEMENT_CLAIM_POLICY",
    "TARGET_METRIC_CONFIDENCE_INTERVAL_FILE",
    "TARGET_METRIC_CONFIDENCE_INTERVAL_SCHEMA",
    "TARGET_METRIC_CONFIDENCE_INTERVAL_SCHEMA_VERSION",
    "GroupedBootstrapConfig",
    "GroupedBootstrapResult",
    "build_grouped_metric_confidence_intervals",
    "empty_target_metric_confidence_intervals",
    "validate_grouped_bootstrap_result",
]
