"""Weighted, abstention-aware metrics for calibrated target verification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
from math import isfinite
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.calibration import (
    DEFAULT_CALIBRATION_THRESHOLDS,
    TARGET_CALIBRATION_RELIABILITY_SCHEMA,
    TARGET_THRESHOLD_OPERATING_POINT_SCHEMA,
    TargetCalibrationDiagnostics,
    build_target_calibration_diagnostics,
    validate_target_calibration_diagnostics,
)
from biominer.evaluation.leakage import validate_reference_and_holdout_leakage
from biominer.flickr_fetch.geographic_clustering import NO_GEO_CLUSTER_ID
from biominer.ml.calibration import CALIBRATION_METHODS
from biominer.ml.nonmatch import ABSTAIN, CLASSIFICATION_OUTCOMES, TARGET_CONFIRMED
from biominer.reports.flickr_fetch import current_git_sha
from biominer.storage.parquet import write_parquet


TARGET_VERIFICATION_EVALUATION_SCHEMA_VERSION = "target-verification-evaluation-v1.1.0"
TARGET_VERIFICATION_METRIC_SCHEMA_VERSION = "target-verification-metric-v1.1.0"
TARGET_MARGIN_DISTRIBUTION_SCHEMA_VERSION = "target-margin-distribution-v1.0.0"
TARGET_VERIFICATION_REPORT_SCHEMA_VERSION = "target-verification-report-v1.1.0"

TARGET_VERIFICATION_METRICS_FILE = "target_verification_metrics.parquet"
TARGET_MARGIN_DISTRIBUTION_FILE = "target_competitor_margin_distribution.parquet"
TARGET_CALIBRATION_RELIABILITY_FILE = "target_calibration_reliability.parquet"
TARGET_THRESHOLD_OPERATING_POINTS_FILE = "target_threshold_operating_points.parquet"
TARGET_VERIFICATION_REPORT_FILE = "target_verification_report.json"
TARGET_VERIFICATION_REPORT_MARKDOWN_FILE = "target_verification_report.md"

EVALUATION_SET_VALUES = frozenset({"balanced_challenge", "natural_stream"})

TARGET_VERIFICATION_EVALUATION_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "evaluation_item_id": pl.String,
    "evaluation_set": pl.String,
    "sampling_weight": pl.Float64,
    "target_present": pl.Boolean,
    "calibrated_target_probability": pl.Float64,
    "calibration_method": pl.String,
    "calibration_split_fingerprint": pl.String,
    "calibrator_fingerprint": pl.String,
    "classification_decision": pl.String,
    "abstained": pl.Boolean,
    "target_competitor_margin": pl.Float64,
    "ground_truth_out_of_distribution": pl.Boolean,
    "detector_gate_required": pl.Boolean,
    "detector_gate_passed": pl.Boolean,
    "family_evaluable": pl.Boolean,
    "true_family_rank": pl.UInt32,
    "genus_evaluable": pl.Boolean,
    "true_genus_rank": pl.UInt32,
    "species_evaluable": pl.Boolean,
    "true_species_rank": pl.UInt32,
    "old_classifier_target_pruned": pl.Boolean,
    "geo_cluster_id": pl.String,
    "country_code": pl.String,
    "no_geo": pl.Boolean,
    "route": pl.String,
    "life_stage": pl.String,
    "visual_domain": pl.String,
    "subject_area_band": pl.String,
    "source_query_tier": pl.String,
    "source_query_term": pl.String,
    "source_provider": pl.String,
    "visual_input_kind": pl.String,
}

TARGET_VERIFICATION_METRIC_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "input_fingerprint": pl.String,
    "configuration_fingerprint": pl.String,
    "evaluation_set": pl.String,
    "scope": pl.String,
    "stratum_dimension": pl.String,
    "stratum_value": pl.String,
    "metric_family": pl.String,
    "metric_name": pl.String,
    "metric_value": pl.Float64,
    "numerator": pl.Float64,
    "denominator": pl.Float64,
    "undefined_reason": pl.String,
    "item_count": pl.UInt32,
    "weighted_item_count": pl.Float64,
    "target_item_count": pl.UInt32,
    "weighted_target_item_count": pl.Float64,
}

TARGET_MARGIN_DISTRIBUTION_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "input_fingerprint": pl.String,
    "configuration_fingerprint": pl.String,
    "evaluation_set": pl.String,
    "scope": pl.String,
    "stratum_dimension": pl.String,
    "stratum_value": pl.String,
    "population": pl.String,
    "item_count": pl.UInt32,
    "missing_margin_count": pl.UInt32,
    "weighted_item_count": pl.Float64,
    "margin_mean": pl.Float64,
    "margin_stddev": pl.Float64,
    "margin_min": pl.Float64,
    "margin_p05": pl.Float64,
    "margin_p25": pl.Float64,
    "margin_median": pl.Float64,
    "margin_p75": pl.Float64,
    "margin_p95": pl.Float64,
    "margin_max": pl.Float64,
}

STRATIFICATION_FIELDS: tuple[tuple[str, str], ...] = (
    ("geographic_cluster", "geo_cluster_id"),
    ("country", "country_code"),
    ("no_geo", "no_geo"),
    ("route", "route"),
    ("life_stage", "life_stage"),
    ("visual_domain", "visual_domain"),
    ("subject_size_band", "subject_area_band"),
    ("query_tier", "source_query_tier"),
    ("search_term", "source_query_term"),
    ("source_provider", "source_provider"),
    ("visual_input_kind", "visual_input_kind"),
)

_REQUIRED_TEXT_FIELDS = (
    "schema_version",
    "evaluation_item_id",
    "evaluation_set",
    "classification_decision",
    "calibration_method",
    "calibration_split_fingerprint",
    "calibrator_fingerprint",
    "geo_cluster_id",
    "country_code",
    "route",
    "life_stage",
    "visual_domain",
    "subject_area_band",
    "source_query_tier",
    "source_query_term",
    "source_provider",
    "visual_input_kind",
)
_RANK_FIELDS = (
    ("family_evaluable", "true_family_rank"),
    ("genus_evaluable", "true_genus_rank"),
    ("species_evaluable", "true_species_rank"),
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_LOGGER = logging.getLogger(__name__)
_REQUIRED_NON_NULL_FIELDS = (
    "target_present",
    "abstained",
    "ground_truth_out_of_distribution",
    "detector_gate_required",
    "family_evaluable",
    "genus_evaluable",
    "species_evaluable",
    "no_geo",
)


@dataclass(frozen=True, slots=True)
class TargetVerificationMetricsConfig:
    precision_targets: tuple[float, ...] = (0.90, 0.95, 0.99)
    ece_bin_count: int = 10
    threshold_operating_points: tuple[float, ...] = DEFAULT_CALIBRATION_THRESHOLDS
    calibration_confidence_level: float = 0.95
    family_recall_ks: tuple[int, ...] = (1, 3, 5)
    genus_recall_ks: tuple[int, ...] = (1, 3, 5)
    species_recall_ks: tuple[int, ...] = (1, 5, 20)

    def __post_init__(self) -> None:
        targets = tuple(sorted(set(self.precision_targets)))
        if not targets or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or not 0.0 < float(value) <= 1.0
            for value in targets
        ):
            raise ValueError("precision_targets must contain values in (0, 1]")
        object.__setattr__(
            self,
            "precision_targets",
            tuple(float(value) for value in targets),
        )
        if (
            isinstance(self.ece_bin_count, bool)
            or not isinstance(self.ece_bin_count, int)
            or self.ece_bin_count < 2
        ):
            raise ValueError("ece_bin_count must be an integer >= 2")
        operating_points = tuple(sorted(set(self.threshold_operating_points)))
        if not operating_points or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in operating_points
        ):
            raise ValueError("threshold_operating_points must contain values in [0, 1]")
        object.__setattr__(
            self,
            "threshold_operating_points",
            tuple(float(value) for value in operating_points),
        )
        confidence = self.calibration_confidence_level
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not isfinite(float(confidence))
            or not 0.0 < float(confidence) < 1.0
        ):
            raise ValueError("calibration_confidence_level must be in (0, 1)")
        object.__setattr__(
            self,
            "calibration_confidence_level",
            float(confidence),
        )
        for field in (
            "family_recall_ks",
            "genus_recall_ks",
            "species_recall_ks",
        ):
            values = tuple(sorted(set(getattr(self, field))))
            if not values or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in values
            ):
                raise ValueError(f"{field} must contain positive integers")
            object.__setattr__(self, field, values)

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": TARGET_VERIFICATION_METRIC_SCHEMA_VERSION,
                "precision_targets": list(self.precision_targets),
                "ece_bin_count": self.ece_bin_count,
                "threshold_operating_points": list(self.threshold_operating_points),
                "calibration_confidence_level": self.calibration_confidence_level,
                "family_recall_ks": list(self.family_recall_ks),
                "genus_recall_ks": list(self.genus_recall_ks),
                "species_recall_ks": list(self.species_recall_ks),
                "positive_decision": TARGET_CONFIRMED,
                "pr_auc_definition": "noninterpolated_average_precision",
                "abstention_semantics": (
                    "system_recall_penalizes_abstention; selective_risk_is_retained_only"
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class TargetVerificationMetricReport:
    metrics: pl.DataFrame
    margin_distribution: pl.DataFrame
    calibration_diagnostics: TargetCalibrationDiagnostics
    input_fingerprint: str
    configuration_fingerprint: str
    report_fingerprint: str


@dataclass(frozen=True, slots=True)
class TargetVerificationMetricPublication:
    output_dir: Path
    metrics_path: Path
    margin_distribution_path: Path
    calibration_reliability_path: Path
    threshold_operating_points_path: Path
    report_json_path: Path
    report_markdown_path: Path
    report: Mapping[str, object]


def empty_target_verification_evaluation_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=TARGET_VERIFICATION_EVALUATION_SCHEMA)


def target_verification_evaluation_frame(
    rows: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    materialized = tuple(rows)
    expected_fields = set(TARGET_VERIFICATION_EVALUATION_SCHEMA)
    for index, row in enumerate(materialized):
        if not isinstance(row, Mapping):
            raise TypeError(f"target verification row {index} must be a mapping")
        missing = sorted(expected_fields - set(row))
        unexpected = sorted(set(row) - expected_fields)
        if missing or unexpected:
            raise ValueError(
                f"target verification row {index} fields differ: "
                f"missing={missing}, unexpected={unexpected}"
            )
    frame = pl.DataFrame(
        materialized,
        schema=TARGET_VERIFICATION_EVALUATION_SCHEMA,
    ).sort("evaluation_set", "evaluation_item_id")
    validate_target_verification_evaluation_frame(frame)
    return frame


def validate_target_verification_evaluation_frame(frame: pl.DataFrame) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(
            "target verification evaluation input must be a Polars DataFrame"
        )
    if dict(frame.schema) != TARGET_VERIFICATION_EVALUATION_SCHEMA:
        raise ValueError("target verification evaluation physical schema mismatch")
    if frame.is_empty():
        raise ValueError("target verification evaluation input must not be empty")
    expected = frame.sort("evaluation_set", "evaluation_item_id")
    if not frame.equals(expected):
        raise ValueError("target verification evaluation input is not sorted")
    if frame["evaluation_item_id"].n_unique() != frame.height:
        raise ValueError("target verification evaluation item IDs must be unique")
    for field in _REQUIRED_TEXT_FIELDS:
        if frame.filter(
            pl.col(field).is_null()
            | (pl.col(field).str.strip_chars() == "")
            | (pl.col(field) != pl.col(field).str.strip_chars())
        ).height:
            raise ValueError(f"{field} must contain canonical nonblank text")
    if set(frame["schema_version"].to_list()) != {
        TARGET_VERIFICATION_EVALUATION_SCHEMA_VERSION
    }:
        raise ValueError("target verification evaluation schema is incompatible")
    invalid_sets = sorted(
        set(frame["evaluation_set"].to_list()) - EVALUATION_SET_VALUES
    )
    if invalid_sets:
        raise ValueError(f"unsupported evaluation sets: {invalid_sets}")
    invalid_decisions = sorted(
        set(frame["classification_decision"].to_list()) - CLASSIFICATION_OUTCOMES
    )
    if invalid_decisions:
        raise ValueError(f"unsupported classification decisions: {invalid_decisions}")
    invalid_methods = sorted(
        set(frame["calibration_method"].to_list()) - CALIBRATION_METHODS
    )
    if invalid_methods:
        raise ValueError(f"unsupported calibration methods: {invalid_methods}")
    for field in ("calibration_split_fingerprint", "calibrator_fingerprint"):
        if any(
            _SHA256_PATTERN.fullmatch(str(value)) is None
            for value in frame[field].to_list()
        ):
            raise ValueError(f"{field} must contain full lowercase sha256 fingerprints")
    for field in _REQUIRED_NON_NULL_FIELDS:
        if frame[field].null_count():
            raise ValueError(f"{field} cannot contain null values")
    if frame.filter(
        pl.col("sampling_weight").is_null()
        | ~pl.col("sampling_weight").is_finite()
        | (pl.col("sampling_weight") <= 0.0)
    ).height:
        raise ValueError("sampling_weight must contain positive finite values")
    probability = pl.col("calibrated_target_probability")
    if frame.filter(
        probability.is_not_null()
        & (~probability.is_finite() | (probability < 0.0) | (probability > 1.0))
    ).height:
        raise ValueError("calibrated_target_probability must be null or in [0, 1]")
    margin = pl.col("target_competitor_margin")
    if frame.filter(margin.is_not_null() & ~margin.is_finite()).height:
        raise ValueError("target_competitor_margin must be null or finite")
    if frame.filter(
        (pl.col("classification_decision") == TARGET_CONFIRMED) & pl.col("abstained")
    ).height:
        raise ValueError("target_confirmed decisions cannot be abstained")
    if frame.filter(
        (pl.col("classification_decision") == ABSTAIN) & ~pl.col("abstained")
    ).height:
        raise ValueError("abstain decisions must set abstained=true")
    if frame.filter(
        pl.col("detector_gate_required") & pl.col("detector_gate_passed").is_null()
    ).height:
        raise ValueError("required detector-gate outcomes cannot be null")
    for evaluable_field, rank_field in _RANK_FIELDS:
        if frame.filter(
            pl.col(rank_field).is_not_null() & (pl.col(rank_field) == 0)
        ).height:
            raise ValueError(f"{rank_field} must be positive when present")
        if frame.filter(
            ~pl.col(evaluable_field) & pl.col(rank_field).is_not_null()
        ).height:
            raise ValueError(
                f"{rank_field} cannot be present when {evaluable_field} is false"
            )
    if (
        frame.filter(
            pl.col("no_geo") & (pl.col("geo_cluster_id") != NO_GEO_CLUSTER_ID)
        ).height
        or frame.filter(
            ~pl.col("no_geo") & (pl.col("geo_cluster_id") == NO_GEO_CLUSTER_ID)
        ).height
    ):
        raise ValueError("no_geo and geo_cluster_id are inconsistent")


def evaluate_target_verification(
    frame: pl.DataFrame,
    balanced_challenge: pl.DataFrame,
    natural_stream: pl.DataFrame,
    leakage_register: pl.DataFrame,
    config: TargetVerificationMetricsConfig | None = None,
) -> TargetVerificationMetricReport:
    """Evaluate complete frozen holdouts after revalidating leakage."""

    validate_reference_and_holdout_leakage(
        leakage_register,
        balanced_challenge,
        natural_stream,
    )
    validate_target_verification_evaluation_frame(frame)
    _validate_evaluation_matches_holdouts(
        frame,
        balanced_challenge,
        natural_stream,
    )
    return compute_target_verification_metrics(frame, config)


def compute_target_verification_metrics(
    frame: pl.DataFrame,
    config: TargetVerificationMetricsConfig | None = None,
) -> TargetVerificationMetricReport:
    """Compute each frozen set independently and across required strata."""

    validate_target_verification_evaluation_frame(frame)
    active = config or TargetVerificationMetricsConfig()
    if not isinstance(active, TargetVerificationMetricsConfig):
        raise TypeError("config must be a TargetVerificationMetricsConfig")
    input_fingerprint = _input_fingerprint(frame)
    metric_rows: list[dict[str, object]] = []
    margin_rows: list[dict[str, object]] = []
    for evaluation_set in sorted(set(frame["evaluation_set"].to_list())):
        evaluation = frame.filter(pl.col("evaluation_set") == evaluation_set)
        groups: list[tuple[str, str, str, pl.DataFrame]] = [
            ("overall", "overall", "all", evaluation)
        ]
        for dimension, field in STRATIFICATION_FIELDS:
            values = sorted(
                evaluation[field].unique().to_list(),
                key=lambda value: _stratum_value(value),
            )
            groups.extend(
                (
                    "stratum",
                    dimension,
                    _stratum_value(value),
                    evaluation.filter(pl.col(field) == value),
                )
                for value in values
            )
        for scope, dimension, value, group in groups:
            metric_rows.extend(
                _evaluate_group(
                    group,
                    evaluation_set=evaluation_set,
                    scope=scope,
                    dimension=dimension,
                    value=value,
                    input_fingerprint=input_fingerprint,
                    config=active,
                )
            )
            margin_rows.extend(
                _margin_distribution_rows(
                    group,
                    evaluation_set=evaluation_set,
                    scope=scope,
                    dimension=dimension,
                    value=value,
                    input_fingerprint=input_fingerprint,
                    config=active,
                )
            )
    metrics = pl.DataFrame(
        metric_rows,
        schema=TARGET_VERIFICATION_METRIC_SCHEMA,
    ).sort(
        "evaluation_set",
        "scope",
        "stratum_dimension",
        "stratum_value",
        "metric_name",
    )
    margins = pl.DataFrame(
        margin_rows,
        schema=TARGET_MARGIN_DISTRIBUTION_SCHEMA,
    ).sort(
        "evaluation_set",
        "scope",
        "stratum_dimension",
        "stratum_value",
        "population",
    )
    calibration_diagnostics = build_target_calibration_diagnostics(
        frame,
        bin_count=active.ece_bin_count,
        thresholds=active.threshold_operating_points,
        confidence_level=active.calibration_confidence_level,
    )
    report_fingerprint = _report_fingerprint(
        metrics,
        margins,
        calibration_diagnostics,
        input_fingerprint=input_fingerprint,
        configuration_fingerprint=active.fingerprint,
    )
    return TargetVerificationMetricReport(
        metrics=metrics,
        margin_distribution=margins,
        calibration_diagnostics=calibration_diagnostics,
        input_fingerprint=input_fingerprint,
        configuration_fingerprint=active.fingerprint,
        report_fingerprint=report_fingerprint,
    )


def validate_target_verification_metric_report(
    report: TargetVerificationMetricReport,
) -> None:
    if not isinstance(report, TargetVerificationMetricReport):
        raise TypeError("report must be a TargetVerificationMetricReport")
    if dict(report.metrics.schema) != TARGET_VERIFICATION_METRIC_SCHEMA:
        raise ValueError("target verification metric physical schema mismatch")
    if dict(report.margin_distribution.schema) != TARGET_MARGIN_DISTRIBUTION_SCHEMA:
        raise ValueError("target margin distribution physical schema mismatch")
    if report.metrics.is_empty() or report.margin_distribution.is_empty():
        raise ValueError("target verification report artifacts must not be empty")
    validate_target_calibration_diagnostics(report.calibration_diagnostics)
    metric_sort = report.metrics.sort(
        "evaluation_set",
        "scope",
        "stratum_dimension",
        "stratum_value",
        "metric_name",
    )
    if not report.metrics.equals(metric_sort):
        raise ValueError("target verification metrics are not sorted")
    margin_sort = report.margin_distribution.sort(
        "evaluation_set",
        "scope",
        "stratum_dimension",
        "stratum_value",
        "population",
    )
    if not report.margin_distribution.equals(margin_sort):
        raise ValueError("target margin distributions are not sorted")
    for value, field in (
        (report.input_fingerprint, "input_fingerprint"),
        (report.configuration_fingerprint, "configuration_fingerprint"),
        (report.report_fingerprint, "report_fingerprint"),
    ):
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{field} must be a full lowercase sha256 fingerprint")
    if set(report.metrics["input_fingerprint"].to_list()) != {
        report.input_fingerprint
    } or set(report.margin_distribution["input_fingerprint"].to_list()) != {
        report.input_fingerprint
    }:
        raise ValueError("target verification input_fingerprint is inconsistent")
    if set(report.metrics["configuration_fingerprint"].to_list()) != {
        report.configuration_fingerprint
    } or set(report.margin_distribution["configuration_fingerprint"].to_list()) != {
        report.configuration_fingerprint
    }:
        raise ValueError(
            "target verification configuration_fingerprint is inconsistent"
        )
    expected = _report_fingerprint(
        report.metrics,
        report.margin_distribution,
        report.calibration_diagnostics,
        input_fingerprint=report.input_fingerprint,
        configuration_fingerprint=report.configuration_fingerprint,
    )
    if report.report_fingerprint != expected:
        raise ValueError("target verification report_fingerprint is invalid")


def publish_target_verification_metric_report(
    report: TargetVerificationMetricReport,
    output_dir: str | Path,
    *,
    run_id: str | None = None,
) -> TargetVerificationMetricPublication:
    """Atomically publish metric tables and compact JSON/Markdown audit reports."""

    validate_target_verification_metric_report(report)
    started_at = datetime.now(UTC)
    effective_run_id = str(
        run_id
        or "target-verification-"
        + started_at.strftime("%Y%m%dT%H%M%S%fZ-")
        + uuid4().hex[:12]
    ).strip()
    if not effective_run_id:
        raise ValueError("run_id cannot be blank")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
    _log_event(
        "target_verification_report_publish_started",
        command="evaluation.publish_target_verification_metrics",
        run_id=effective_run_id,
        output_dir=str(destination),
        input_fingerprint=report.input_fingerprint,
        report_fingerprint=report.report_fingerprint,
    )
    try:
        staging.mkdir(parents=False, exist_ok=False)
        metrics_staged = write_parquet(
            report.metrics,
            staging / TARGET_VERIFICATION_METRICS_FILE,
            overwrite=False,
        )
        margins_staged = write_parquet(
            report.margin_distribution,
            staging / TARGET_MARGIN_DISTRIBUTION_FILE,
            overwrite=False,
        )
        calibration_staged = write_parquet(
            report.calibration_diagnostics.reliability,
            staging / TARGET_CALIBRATION_RELIABILITY_FILE,
            overwrite=False,
        )
        operating_points_staged = write_parquet(
            report.calibration_diagnostics.operating_points,
            staging / TARGET_THRESHOLD_OPERATING_POINTS_FILE,
            overwrite=False,
        )
        ended_at = datetime.now(UTC)
        payload = _publication_payload(
            report,
            metrics_path=metrics_staged,
            margins_path=margins_staged,
            calibration_path=calibration_staged,
            operating_points_path=operating_points_staged,
            final_output_dir=destination,
            run_id=effective_run_id,
            started_at=started_at,
            ended_at=ended_at,
        )
        (staging / TARGET_VERIFICATION_REPORT_FILE).write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        (staging / TARGET_VERIFICATION_REPORT_MARKDOWN_FILE).write_text(
            _publication_markdown(payload),
            encoding="utf-8",
        )
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    metrics_path = destination / TARGET_VERIFICATION_METRICS_FILE
    margins_path = destination / TARGET_MARGIN_DISTRIBUTION_FILE
    calibration_path = destination / TARGET_CALIBRATION_RELIABILITY_FILE
    operating_points_path = destination / TARGET_THRESHOLD_OPERATING_POINTS_FILE
    loaded_metrics = pl.read_parquet(metrics_path)
    loaded_margins = pl.read_parquet(margins_path)
    loaded_calibration = pl.read_parquet(calibration_path)
    loaded_operating_points = pl.read_parquet(operating_points_path)
    if not report.metrics.equals(loaded_metrics):
        raise ValueError("target verification metrics Parquet round-trip mismatch")
    if not report.margin_distribution.equals(loaded_margins):
        raise ValueError("target margin distribution Parquet round-trip mismatch")
    if not report.calibration_diagnostics.reliability.equals(loaded_calibration):
        raise ValueError("target calibration reliability Parquet round-trip mismatch")
    if not report.calibration_diagnostics.operating_points.equals(
        loaded_operating_points
    ):
        raise ValueError("target threshold operating-point Parquet round-trip mismatch")
    _log_event(
        "target_verification_report_publish_completed",
        command="evaluation.publish_target_verification_metrics",
        run_id=effective_run_id,
        output_dir=str(destination),
        metric_rows=report.metrics.height,
        margin_rows=report.margin_distribution.height,
        calibration_rows=report.calibration_diagnostics.reliability.height,
        operating_point_rows=report.calibration_diagnostics.operating_points.height,
        report_fingerprint=report.report_fingerprint,
    )
    return TargetVerificationMetricPublication(
        output_dir=destination,
        metrics_path=metrics_path,
        margin_distribution_path=margins_path,
        calibration_reliability_path=calibration_path,
        threshold_operating_points_path=operating_points_path,
        report_json_path=destination / TARGET_VERIFICATION_REPORT_FILE,
        report_markdown_path=(destination / TARGET_VERIFICATION_REPORT_MARKDOWN_FILE),
        report=payload,
    )


def _validate_evaluation_matches_holdouts(
    frame: pl.DataFrame,
    balanced_challenge: pl.DataFrame,
    natural_stream: pl.DataFrame,
) -> None:
    expected: dict[str, dict[str, object]] = {}
    for evaluation_set, holdout in (
        ("balanced_challenge", balanced_challenge),
        ("natural_stream", natural_stream),
    ):
        for row in holdout.iter_rows(named=True):
            item_id = str(row["evaluation_item_id"])
            sampling_weight = (
                1.0
                if evaluation_set == "balanced_challenge"
                else float(row["sampling_weight"])
            )
            expected[item_id] = {
                "evaluation_set": evaluation_set,
                "sampling_weight": sampling_weight,
                "target_present": row["target_present"],
                "geo_cluster_id": row["geo_cluster_id"],
                "route": row["route"] or "not_applicable",
                "life_stage": row["life_stage"],
                "visual_domain": row["visual_domain"],
                "source_query_tier": row["source_query_tier"],
                "source_query_term": row["source_query_term"],
                "source_provider": str(row["source"]).casefold(),
            }
    actual_ids = set(frame["evaluation_item_id"].to_list())
    expected_ids = set(expected)
    if actual_ids != expected_ids:
        raise ValueError(
            "target verification input does not cover the frozen holdouts: "
            f"missing={sorted(expected_ids - actual_ids)[:10]}, "
            f"unexpected={sorted(actual_ids - expected_ids)[:10]}"
        )
    fields = (
        "evaluation_set",
        "target_present",
        "geo_cluster_id",
        "route",
        "life_stage",
        "visual_domain",
        "source_query_tier",
        "source_query_term",
        "source_provider",
    )
    for row in frame.iter_rows(named=True):
        item_id = str(row["evaluation_item_id"])
        expected_row = expected[item_id]
        for field in fields:
            if row[field] != expected_row[field]:
                raise ValueError(
                    f"target verification item {item_id} has a {field} mismatch"
                )
        if (
            abs(float(row["sampling_weight"]) - float(expected_row["sampling_weight"]))
            > 1e-12
        ):
            raise ValueError(
                f"target verification item {item_id} has a sampling_weight mismatch"
            )


def _evaluate_group(
    frame: pl.DataFrame,
    *,
    evaluation_set: str,
    scope: str,
    dimension: str,
    value: str,
    input_fingerprint: str,
    config: TargetVerificationMetricsConfig,
) -> list[dict[str, object]]:
    weights = np.asarray(frame["sampling_weight"].to_list(), dtype=np.float64)
    truth = np.asarray(frame["target_present"].to_list(), dtype=np.bool_)
    decisions = frame["classification_decision"].to_list()
    predicted = np.asarray(
        [decision == TARGET_CONFIRMED for decision in decisions],
        dtype=np.bool_,
    )
    abstained = np.asarray(frame["abstained"].to_list(), dtype=np.bool_)
    total_weight = float(weights.sum())
    target_weight = float(weights[truth].sum())
    base = {
        "schema_version": TARGET_VERIFICATION_METRIC_SCHEMA_VERSION,
        "input_fingerprint": input_fingerprint,
        "configuration_fingerprint": config.fingerprint,
        "evaluation_set": evaluation_set,
        "scope": scope,
        "stratum_dimension": dimension,
        "stratum_value": value,
        "item_count": frame.height,
        "weighted_item_count": total_weight,
        "target_item_count": int(truth.sum()),
        "weighted_target_item_count": target_weight,
    }
    rows: list[dict[str, object]] = []

    def add(
        family: str,
        name: str,
        metric_value: float | None,
        *,
        numerator: float | None = None,
        denominator: float | None = None,
        undefined_reason: str | None = None,
    ) -> None:
        rows.append(
            {
                **base,
                "metric_family": family,
                "metric_name": name,
                "metric_value": metric_value,
                "numerator": numerator,
                "denominator": denominator,
                "undefined_reason": undefined_reason,
            }
        )

    tp = float(weights[truth & predicted].sum())
    tn = float(weights[~truth & ~predicted].sum())
    fp = float(weights[~truth & predicted].sum())
    fn = float(weights[truth & ~predicted].sum())
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    fpr = _ratio(fp, fp + tn)
    fnr = _ratio(fn, fn + tp)
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    add(
        "classification",
        "precision",
        precision,
        numerator=tp,
        denominator=tp + fp,
        undefined_reason=_zero_reason(tp + fp),
    )
    add(
        "classification",
        "recall",
        recall,
        numerator=tp,
        denominator=tp + fn,
        undefined_reason=_zero_reason(tp + fn),
    )
    add(
        "classification",
        "f1",
        f1,
        undefined_reason=(
            "undefined_precision_or_recall"
            if precision is None or recall is None
            else None
        ),
    )
    add(
        "classification",
        "specificity",
        specificity,
        numerator=tn,
        denominator=tn + fp,
        undefined_reason=_zero_reason(tn + fp),
    )
    add(
        "classification",
        "false_positive_rate",
        fpr,
        numerator=fp,
        denominator=fp + tn,
        undefined_reason=_zero_reason(fp + tn),
    )
    add(
        "classification",
        "false_negative_rate",
        fnr,
        numerator=fn,
        denominator=fn + tp,
        undefined_reason=_zero_reason(fn + tp),
    )
    for name, count in (
        ("true_positive_weight", tp),
        ("true_negative_weight", tn),
        ("false_positive_weight", fp),
        ("false_negative_weight", fn),
    ):
        add("confusion", name, count, numerator=count, denominator=total_weight)

    retained = ~abstained
    retained_weight = float(weights[retained].sum())
    abstained_weight = float(weights[abstained].sum())
    retained_errors = float(weights[retained & (truth != predicted)].sum())
    add(
        "selective",
        "coverage",
        _ratio(retained_weight, total_weight),
        numerator=retained_weight,
        denominator=total_weight,
        undefined_reason=_zero_reason(total_weight),
    )
    add(
        "selective",
        "abstention_rate",
        _ratio(abstained_weight, total_weight),
        numerator=abstained_weight,
        denominator=total_weight,
        undefined_reason=_zero_reason(total_weight),
    )
    add(
        "selective",
        "selective_risk",
        _ratio(retained_errors, retained_weight),
        numerator=retained_errors,
        denominator=retained_weight,
        undefined_reason=_zero_reason(retained_weight),
    )

    probabilities = frame["calibrated_target_probability"].to_list()
    complete_probability = all(value is not None for value in probabilities)
    probability_mask = np.asarray(
        [value is not None for value in probabilities],
        dtype=np.bool_,
    )
    probability_weight = float(weights[probability_mask].sum())
    add(
        "calibration",
        "probability_coverage",
        _ratio(probability_weight, total_weight),
        numerator=probability_weight,
        denominator=total_weight,
        undefined_reason=_zero_reason(total_weight),
    )
    both_classes = bool(truth.any() and (~truth).any())
    if complete_probability:
        score = np.asarray(probabilities, dtype=np.float64)
        brier = float(
            brier_score_loss(
                truth,
                score,
                sample_weight=weights,
                pos_label=True,
                scale_by_half=True,
            )
        )
        loss = float(
            log_loss(
                truth,
                score,
                sample_weight=weights,
                labels=[False, True],
            )
        )
        ece = _expected_calibration_error(
            truth,
            score,
            weights,
            bin_count=config.ece_bin_count,
        )
        add("calibration", "brier_score", brier)
        add("calibration", "log_loss", loss)
        add("calibration", "expected_calibration_error", ece)
        if both_classes:
            add(
                "ranking",
                "pr_auc",
                float(average_precision_score(truth, score, sample_weight=weights)),
            )
            add(
                "ranking",
                "roc_auc",
                float(roc_auc_score(truth, score, sample_weight=weights)),
            )
            precision_curve, recall_curve, _ = precision_recall_curve(
                truth,
                score,
                sample_weight=weights,
            )
            for target_precision in config.precision_targets:
                valid = precision_curve >= target_precision
                recall_at_precision = float(recall_curve[valid].max())
                add(
                    "ranking",
                    _precision_metric_name(target_precision),
                    recall_at_precision,
                )
        else:
            add("ranking", "pr_auc", None, undefined_reason="single_class")
            add("ranking", "roc_auc", None, undefined_reason="single_class")
            for target_precision in config.precision_targets:
                add(
                    "ranking",
                    _precision_metric_name(target_precision),
                    None,
                    undefined_reason="single_class",
                )
    else:
        for name in (
            "brier_score",
            "log_loss",
            "expected_calibration_error",
        ):
            add(
                "calibration",
                name,
                None,
                undefined_reason="incomplete_probability_coverage",
            )
        for name in ("pr_auc", "roc_auc"):
            add(
                "ranking",
                name,
                None,
                undefined_reason="incomplete_probability_coverage",
            )
        for target_precision in config.precision_targets:
            add(
                "ranking",
                _precision_metric_name(target_precision),
                None,
                undefined_reason="incomplete_probability_coverage",
            )

    ood = np.asarray(
        frame["ground_truth_out_of_distribution"].to_list(),
        dtype=np.bool_,
    )
    ood_weight = float(weights[ood].sum())
    ood_false_positive_weight = float(weights[ood & predicted].sum())
    add(
        "ood",
        "ood_false_positive_rate",
        _ratio(ood_false_positive_weight, ood_weight),
        numerator=ood_false_positive_weight,
        denominator=ood_weight,
        undefined_reason=_zero_reason(ood_weight),
    )

    gate_required = np.asarray(
        frame["detector_gate_required"].fill_null(False).to_list(),
        dtype=np.bool_,
    )
    gate_passed = np.asarray(
        frame["detector_gate_passed"].fill_null(False).to_list(),
        dtype=np.bool_,
    )
    gate_required_weight = float(weights[gate_required].sum())
    gate_passed_weight = float(weights[gate_required & gate_passed].sum())
    add(
        "detector",
        "detector_gate_recall",
        _ratio(gate_passed_weight, gate_required_weight),
        numerator=gate_passed_weight,
        denominator=gate_required_weight,
        undefined_reason=_zero_reason(gate_required_weight),
    )

    for level, evaluable_field, rank_field, ks in (
        ("family", "family_evaluable", "true_family_rank", config.family_recall_ks),
        ("genus", "genus_evaluable", "true_genus_rank", config.genus_recall_ks),
        (
            "species",
            "species_evaluable",
            "true_species_rank",
            config.species_recall_ks,
        ),
    ):
        evaluable = np.asarray(frame[evaluable_field].to_list(), dtype=np.bool_)
        ranks = frame[rank_field].to_list()
        evaluable_weight = float(weights[evaluable].sum())
        for k in ks:
            hits = np.asarray(
                [
                    bool(is_evaluable and rank is not None and int(rank) <= k)
                    for is_evaluable, rank in zip(evaluable, ranks, strict=True)
                ],
                dtype=np.bool_,
            )
            hit_weight = float(weights[hits].sum())
            add(
                "diagnostic_recall",
                f"{level}_recall_at_{k}",
                _ratio(hit_weight, evaluable_weight),
                numerator=hit_weight,
                denominator=evaluable_weight,
                undefined_reason=_zero_reason(evaluable_weight),
            )

    old_pruned_values = frame["old_classifier_target_pruned"].to_list()
    counterfactual_mask = np.asarray(
        [
            truth_value and value is not None
            for truth_value, value in zip(truth, old_pruned_values, strict=True)
        ],
        dtype=np.bool_,
    )
    pruned = np.asarray(
        [bool(value) if value is not None else False for value in old_pruned_values],
        dtype=np.bool_,
    )
    counterfactual_weight = float(weights[counterfactual_mask].sum())
    pruned_weight = float(weights[counterfactual_mask & pruned].sum())
    add(
        "counterfactual",
        "old_classifier_target_pruning_rate",
        _ratio(pruned_weight, counterfactual_weight),
        numerator=pruned_weight,
        denominator=counterfactual_weight,
        undefined_reason=_zero_reason(counterfactual_weight),
    )
    return rows


def _margin_distribution_rows(
    frame: pl.DataFrame,
    *,
    evaluation_set: str,
    scope: str,
    dimension: str,
    value: str,
    input_fingerprint: str,
    config: TargetVerificationMetricsConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for population, selected in (
        ("all", frame),
        ("target_present", frame.filter(pl.col("target_present"))),
        ("target_absent", frame.filter(~pl.col("target_present"))),
    ):
        available = selected.filter(pl.col("target_competitor_margin").is_not_null())
        margins = np.asarray(
            available["target_competitor_margin"].to_list(),
            dtype=np.float64,
        )
        weights = np.asarray(
            available["sampling_weight"].to_list(),
            dtype=np.float64,
        )
        statistics = _weighted_distribution(margins, weights)
        rows.append(
            {
                "schema_version": TARGET_MARGIN_DISTRIBUTION_SCHEMA_VERSION,
                "input_fingerprint": input_fingerprint,
                "configuration_fingerprint": config.fingerprint,
                "evaluation_set": evaluation_set,
                "scope": scope,
                "stratum_dimension": dimension,
                "stratum_value": value,
                "population": population,
                "item_count": available.height,
                "missing_margin_count": selected.height - available.height,
                "weighted_item_count": float(weights.sum()),
                **statistics,
            }
        )
    return rows


def _weighted_distribution(
    values: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | None]:
    names = (
        "margin_mean",
        "margin_stddev",
        "margin_min",
        "margin_p05",
        "margin_p25",
        "margin_median",
        "margin_p75",
        "margin_p95",
        "margin_max",
    )
    if values.size == 0:
        return dict.fromkeys(names)
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    quantiles = _weighted_quantiles(values, weights, (0.05, 0.25, 0.5, 0.75, 0.95))
    return {
        "margin_mean": mean,
        "margin_stddev": variance**0.5,
        "margin_min": float(values.min()),
        "margin_p05": quantiles[0],
        "margin_p25": quantiles[1],
        "margin_median": quantiles[2],
        "margin_p75": quantiles[3],
        "margin_p95": quantiles[4],
        "margin_max": float(values.max()),
    }


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Sequence[float],
) -> tuple[float, ...]:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    total_weight = float(cumulative[-1])
    return tuple(
        float(
            ordered_values[
                min(
                    int(
                        np.searchsorted(
                            cumulative, quantile * total_weight, side="left"
                        )
                    ),
                    ordered_values.size - 1,
                )
            ]
        )
        for quantile in quantiles
    )


def _expected_calibration_error(
    truth: np.ndarray,
    score: np.ndarray,
    weights: np.ndarray,
    *,
    bin_count: int,
) -> float:
    total_weight = float(weights.sum())
    indices = np.minimum((score * bin_count).astype(np.int64), bin_count - 1)
    result = 0.0
    for index in range(bin_count):
        mask = indices == index
        if not mask.any():
            continue
        bin_weight = float(weights[mask].sum())
        confidence = float(np.average(score[mask], weights=weights[mask]))
        observed = float(np.average(truth[mask], weights=weights[mask]))
        result += (bin_weight / total_weight) * abs(confidence - observed)
    return result


def _input_fingerprint(frame: pl.DataFrame) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema_version": TARGET_VERIFICATION_EVALUATION_SCHEMA_VERSION,
            "rows": frame.to_dicts(),
        }
    )


def _report_fingerprint(
    metrics: pl.DataFrame,
    margins: pl.DataFrame,
    calibration_diagnostics: TargetCalibrationDiagnostics,
    *,
    input_fingerprint: str,
    configuration_fingerprint: str,
) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema_version": TARGET_VERIFICATION_METRIC_SCHEMA_VERSION,
            "input_fingerprint": input_fingerprint,
            "configuration_fingerprint": configuration_fingerprint,
            "metric_rows": metrics.to_dicts(),
            "margin_rows": margins.to_dicts(),
            "calibration_diagnostics_fingerprint": (
                calibration_diagnostics.diagnostics_fingerprint
            ),
            "calibration_reliability_rows": (
                calibration_diagnostics.reliability.to_dicts()
            ),
            "threshold_operating_point_rows": (
                calibration_diagnostics.operating_points.to_dicts()
            ),
        }
    )


def _publication_payload(
    report: TargetVerificationMetricReport,
    *,
    metrics_path: Path,
    margins_path: Path,
    calibration_path: Path,
    operating_points_path: Path,
    final_output_dir: Path,
    run_id: str,
    started_at: datetime,
    ended_at: datetime,
) -> dict[str, object]:
    overall = report.metrics.filter(pl.col("scope") == "overall")
    summary: dict[str, dict[str, object]] = {}
    for evaluation_set in sorted(set(overall["evaluation_set"].to_list())):
        rows = overall.filter(pl.col("evaluation_set") == evaluation_set)
        summary[evaluation_set] = {
            str(row["metric_name"]): {
                "value": row["metric_value"],
                "undefined_reason": row["undefined_reason"],
            }
            for row in rows.iter_rows(named=True)
        }
    return {
        "schema_version": TARGET_VERIFICATION_REPORT_SCHEMA_VERSION,
        "command": "evaluation.publish_target_verification_metrics",
        "run_id": run_id,
        "pid": os.getpid(),
        "git_sha": current_git_sha(),
        "status": "complete",
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "elapsed_seconds": max(0.0, (ended_at - started_at).total_seconds()),
        "network_requests": 0,
        "input_fingerprint": report.input_fingerprint,
        "configuration_fingerprint": report.configuration_fingerprint,
        "report_fingerprint": report.report_fingerprint,
        "metric_row_count": report.metrics.height,
        "margin_row_count": report.margin_distribution.height,
        "calibration_reliability_row_count": (
            report.calibration_diagnostics.reliability.height
        ),
        "threshold_operating_point_row_count": (
            report.calibration_diagnostics.operating_points.height
        ),
        "calibration_diagnostics_fingerprint": (
            report.calibration_diagnostics.diagnostics_fingerprint
        ),
        "calibration_reports": _calibration_publication_summary(
            report.calibration_diagnostics.reliability
        ),
        "overall_metrics": summary,
        "artifacts": {
            "metrics": {
                "path": str(final_output_dir / metrics_path.name),
                "row_count": report.metrics.height,
                "byte_count": metrics_path.stat().st_size,
                "sha256": _file_sha256(metrics_path),
            },
            "margin_distribution": {
                "path": str(final_output_dir / margins_path.name),
                "row_count": report.margin_distribution.height,
                "byte_count": margins_path.stat().st_size,
                "sha256": _file_sha256(margins_path),
            },
            "calibration_reliability": {
                "path": str(final_output_dir / calibration_path.name),
                "row_count": report.calibration_diagnostics.reliability.height,
                "byte_count": calibration_path.stat().st_size,
                "sha256": _file_sha256(calibration_path),
            },
            "threshold_operating_points": {
                "path": str(final_output_dir / operating_points_path.name),
                "row_count": report.calibration_diagnostics.operating_points.height,
                "byte_count": operating_points_path.stat().st_size,
                "sha256": _file_sha256(operating_points_path),
            },
        },
    }


def _calibration_publication_summary(
    reliability: pl.DataFrame,
) -> list[dict[str, object]]:
    identity_fields = (
        "evaluation_set",
        "calibration_method",
        "calibration_split_fingerprint",
        "calibrator_fingerprint",
    )
    summaries: list[dict[str, object]] = []
    identities = sorted(
        {
            tuple(str(row[field]) for field in identity_fields)
            for row in reliability.select(identity_fields).iter_rows(named=True)
        }
    )
    for identity in identities:
        selected = reliability
        for field, value in zip(identity_fields, identity, strict=True):
            selected = selected.filter(pl.col(field) == value)
        first = selected.row(0, named=True)
        contributions = selected["ece_contribution"].drop_nulls().to_list()
        summaries.append(
            {
                **dict(zip(identity_fields, identity, strict=True)),
                "probability_kind": first["probability_kind"],
                "calibration_sample_size": first["probability_sample_count"],
                "evaluation_item_count": first["evaluation_item_count"],
                "probability_sample_count": first["probability_sample_count"],
                "missing_probability_count": first["missing_probability_count"],
                "weighted_evaluation_item_count": first[
                    "weighted_evaluation_item_count"
                ],
                "weighted_probability_sample_count": first[
                    "weighted_probability_sample_count"
                ],
                "weighted_probability_coverage": first["weighted_probability_coverage"],
                "reliability_bin_count": selected.height,
                "expected_calibration_error": (
                    float(sum(contributions)) if contributions else None
                ),
                "confidence_level": first["confidence_level"],
                "confidence_interval_method": first["confidence_interval_method"],
            }
        )
    return summaries


def _publication_markdown(payload: Mapping[str, object]) -> str:
    overall = payload["overall_metrics"]
    assert isinstance(overall, Mapping)
    lines = [
        "# Target verification metrics",
        "",
        f"- Status: `{payload['status']}`",
        f"- Run ID: `{payload['run_id']}`",
        f"- Report fingerprint: `{payload['report_fingerprint']}`",
        f"- Metric rows: `{payload['metric_row_count']}`",
        f"- Margin rows: `{payload['margin_row_count']}`",
        (
            "- Calibration reliability rows: "
            f"`{payload['calibration_reliability_row_count']}`"
        ),
        (
            "- Threshold operating-point rows: "
            f"`{payload['threshold_operating_point_row_count']}`"
        ),
    ]
    for evaluation_set, metrics in sorted(overall.items()):
        assert isinstance(metrics, Mapping)
        lines.extend(["", f"## {evaluation_set}", ""])
        for name in (
            "precision",
            "recall",
            "f1",
            "pr_auc",
            "roc_auc",
            "coverage",
            "abstention_rate",
            "brier_score",
            "expected_calibration_error",
        ):
            entry = metrics.get(name)
            if not isinstance(entry, Mapping):
                continue
            value = entry.get("value")
            reason = entry.get("undefined_reason")
            display = value if value is not None else f"undefined ({reason})"
            lines.append(f"- {name}: `{display}`")
    calibration_reports = payload.get("calibration_reports")
    if isinstance(calibration_reports, list):
        lines.extend(["", "## Calibration provenance", ""])
        for entry in calibration_reports:
            if not isinstance(entry, Mapping):
                continue
            lines.append(
                "- "
                f"{entry.get('evaluation_set')} / {entry.get('calibration_method')}: "
                f"n={entry.get('probability_sample_count')}, "
                f"split={entry.get('calibration_split_fingerprint')}, "
                f"ECE={entry.get('expected_calibration_error')}"
            )
    return "\n".join([*lines, ""])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _log_event(event: str, **values: object) -> None:
    _LOGGER.info(
        "%s",
        json.dumps(
            {"event": event, **values},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ),
    )


def _stratum_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _precision_metric_name(value: float) -> str:
    basis_points = round(value * 10_000)
    return f"recall_at_precision_{basis_points:04d}bp"


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def _zero_reason(denominator: float) -> str | None:
    return "zero_denominator" if denominator <= 0.0 else None


__all__ = [
    "EVALUATION_SET_VALUES",
    "STRATIFICATION_FIELDS",
    "TARGET_CALIBRATION_RELIABILITY_FILE",
    "TARGET_CALIBRATION_RELIABILITY_SCHEMA",
    "TARGET_MARGIN_DISTRIBUTION_FILE",
    "TARGET_MARGIN_DISTRIBUTION_SCHEMA",
    "TARGET_MARGIN_DISTRIBUTION_SCHEMA_VERSION",
    "TARGET_THRESHOLD_OPERATING_POINTS_FILE",
    "TARGET_THRESHOLD_OPERATING_POINT_SCHEMA",
    "TARGET_VERIFICATION_EVALUATION_SCHEMA",
    "TARGET_VERIFICATION_EVALUATION_SCHEMA_VERSION",
    "TARGET_VERIFICATION_METRIC_SCHEMA",
    "TARGET_VERIFICATION_METRIC_SCHEMA_VERSION",
    "TARGET_VERIFICATION_METRICS_FILE",
    "TARGET_VERIFICATION_REPORT_FILE",
    "TARGET_VERIFICATION_REPORT_MARKDOWN_FILE",
    "TARGET_VERIFICATION_REPORT_SCHEMA_VERSION",
    "TargetVerificationMetricReport",
    "TargetVerificationMetricPublication",
    "TargetVerificationMetricsConfig",
    "compute_target_verification_metrics",
    "empty_target_verification_evaluation_frame",
    "evaluate_target_verification",
    "publish_target_verification_metric_report",
    "target_verification_evaluation_frame",
    "validate_target_verification_evaluation_frame",
    "validate_target_verification_metric_report",
]
