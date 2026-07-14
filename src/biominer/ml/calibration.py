"""Leakage-audited probability calibration for frozen classifier scores."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import logging
from math import exp, isfinite, log
import os
from pathlib import Path
import re
import shutil
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.ml._numeric_artifacts import (
    bytes_sha256,
    deterministic_numeric_npz,
    load_numeric_npz,
    numeric_array_manifest_entry,
)
from biominer.ml.training_features import TARGET_TASKS
from biominer.references.readiness import REFERENCE_ROUTES


CALIBRATION_MANIFEST_SCHEMA_VERSION = "few-shot-calibration-manifest-v1.0.0"
CALIBRATION_VERSION = "few-shot-probability-calibration-v1.0.0"
CALIBRATION_ARRAY_SCHEMA_VERSION = "few-shot-calibration-arrays-v1.0.0"
CALIBRATION_REPORT_SCHEMA_VERSION = "few-shot-calibration-report-v1.0.0"
CALIBRATION_MANIFEST_FILE = "calibration_artifacts.json"
CALIBRATION_ARRAYS_FILE = "calibration_arrays.npz"
CALIBRATION_REPORT_FILE = "calibration_report.parquet"
CALIBRATION_PARTITION = "calibration"
CALIBRATION_OOF_POLICY = "group-aware-out-of-fold-predictions-v1"
CALIBRATION_SCORE_INPUT_KIND = "estimator_decision_score"
CALIBRATED_PROBABILITY_KIND = "calibrated_probability"

SIGMOID_METHOD = "sigmoid"
ISOTONIC_METHOD = "isotonic"
TEMPERATURE_METHOD = "temperature"
AUTO_METHOD = "auto"
CALIBRATION_METHODS = frozenset({SIGMOID_METHOD, ISOTONIC_METHOD, TEMPERATURE_METHOD})
CALIBRATION_CONFIG_METHODS = frozenset({*CALIBRATION_METHODS, AUTO_METHOD})

MIN_ISOTONIC_SAMPLE_COUNT = 1_000
MIN_ISOTONIC_GROUP_COUNT = 200
DEFAULT_RELIABILITY_BIN_COUNT = 10
DEFAULT_MIN_CLASS_GROUP_COUNT = 2
DEFAULT_TEMPERATURE_OPTIMIZATION_ITERATIONS = 128
TEMPERATURE_LOG_INVERSE_MIN = -10.0
TEMPERATURE_LOG_INVERSE_MAX = 10.0

ISOTONIC_THRESHOLDS_ARRAY = "isotonic_thresholds"
ISOTONIC_VALUES_ARRAY = "isotonic_values"

MAX_CALIBRATION_MANIFEST_BYTES = 1_048_576
MAX_CALIBRATION_ARRAY_ARCHIVE_BYTES = 67_108_864
MAX_CALIBRATION_ARRAY_UNCOMPRESSED_BYTES = 67_108_864
MAX_CALIBRATION_REPORT_BYTES = 67_108_864
MAX_NUMPY_HEADER_BYTES = 4_096

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CalibrationPrediction:
    """One estimator score produced for a held-out leakage component."""

    prediction_id: str
    source_item_id: str
    leakage_component_id: str
    fold_index: int
    dataset_split: str
    true_class_label: str
    decision_scores: tuple[float, ...]
    sample_weight: float = 1.0


@dataclass(frozen=True, slots=True)
class CalibrationFoldAudit:
    """Group identities used to fit and validate one score-producing estimator."""

    fold_index: int
    estimator_fit_group_ids: tuple[str, ...]
    validation_group_ids: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(_fold_semantics(self))


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Calibration identity and method-selection policy for one classifier route."""

    classifier_fingerprint: str
    split_fingerprint: str
    target_task: str
    route: str
    class_labels: tuple[str, ...]
    method: str = AUTO_METHOD
    positive_class_label: str | None = None
    reliability_bin_count: int = DEFAULT_RELIABILITY_BIN_COUNT
    minimum_class_group_count: int = DEFAULT_MIN_CLASS_GROUP_COUNT

    def __post_init__(self) -> None:
        classifier = _sha256(
            self.classifier_fingerprint,
            field="classifier_fingerprint",
        )
        split = _sha256(self.split_fingerprint, field="split_fingerprint")
        task = _required_choice(
            self.target_task,
            field="target_task",
            allowed=TARGET_TASKS,
        )
        route = _required_choice(self.route, field="route", allowed=REFERENCE_ROUTES)
        labels = tuple(
            _required_text(label, field="class_labels") for label in self.class_labels
        )
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError("class_labels must contain at least two unique labels")
        method = _required_choice(
            self.method,
            field="method",
            allowed=CALIBRATION_CONFIG_METHODS,
        )
        positive = (
            None
            if self.positive_class_label is None
            else _required_text(
                self.positive_class_label,
                field="positive_class_label",
            )
        )
        if len(labels) == 2:
            if positive is None or positive not in labels:
                raise ValueError(
                    "binary calibration requires a positive_class_label in class_labels"
                )
            if method == TEMPERATURE_METHOD:
                raise ValueError(
                    "temperature calibration requires at least three classes"
                )
        elif positive is not None:
            raise ValueError(
                "positive_class_label is only valid for binary calibration"
            )
        if method in {SIGMOID_METHOD, ISOTONIC_METHOD} and len(labels) != 2:
            raise ValueError(f"{method} calibration requires exactly two classes")
        if task == "regional_multiclass" and len(labels) < 3:
            raise ValueError(
                "regional multiclass calibration requires at least three classes"
            )
        bins = _integer_at_least(
            self.reliability_bin_count,
            minimum=2,
            field="reliability_bin_count",
        )
        minimum_groups = _integer_at_least(
            self.minimum_class_group_count,
            minimum=2,
            field="minimum_class_group_count",
        )
        object.__setattr__(self, "classifier_fingerprint", classifier)
        object.__setattr__(self, "split_fingerprint", split)
        object.__setattr__(self, "target_task", task)
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "class_labels", labels)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "positive_class_label", positive)
        object.__setattr__(self, "reliability_bin_count", bins)
        object.__setattr__(self, "minimum_class_group_count", minimum_groups)

    @property
    def resolved_method(self) -> str:
        if self.method != AUTO_METHOD:
            return self.method
        return SIGMOID_METHOD if len(self.class_labels) == 2 else TEMPERATURE_METHOD


@dataclass(frozen=True, slots=True)
class FrozenProbabilityCalibrator:
    """Transparent NumPy runtime for persisted calibration parameters."""

    calibration_fingerprint: str
    classifier_fingerprint: str
    split_fingerprint: str
    target_task: str
    route: str
    method: str
    class_labels: tuple[str, ...]
    positive_class_label: str | None
    scalar_parameters: Mapping[str, float]
    array_parameters: Mapping[str, Any]

    def predict_proba(self, decision_scores: object) -> Any:
        """Convert estimator decision scores into ordered class probabilities."""

        np = _load_numpy()
        values = np.asarray(decision_scores, dtype=np.float64)
        if not bool(np.isfinite(values).all()):
            raise ValueError("calibrator decision scores must be finite")
        if self.method in {SIGMOID_METHOD, ISOTONIC_METHOD}:
            if values.ndim == 1:
                margins = values
            elif values.ndim == 2 and values.shape[1] == 1:
                margins = values[:, 0]
            else:
                raise ValueError(
                    "binary calibrators require one decision margin per row"
                )
            if self.method == SIGMOID_METHOD:
                slope = float(self.scalar_parameters["slope"])
                intercept = float(self.scalar_parameters["intercept"])
                positive = _stable_sigmoid_array(slope * margins + intercept)
            else:
                thresholds = self.array_parameters[ISOTONIC_THRESHOLDS_ARRAY]
                probabilities = self.array_parameters[ISOTONIC_VALUES_ARRAY]
                positive = np.interp(
                    margins,
                    thresholds,
                    probabilities,
                    left=float(probabilities[0]),
                    right=float(probabilities[-1]),
                )
            result = np.empty((margins.shape[0], 2), dtype=np.float64)
            positive_index = self.class_labels.index(str(self.positive_class_label))
            result[:, positive_index] = positive
            result[:, 1 - positive_index] = 1.0 - positive
            return result

        if values.ndim != 2 or values.shape[1] != len(self.class_labels):
            raise ValueError(
                "temperature calibration requires one score per ordered class"
            )
        inverse_temperature = float(self.scalar_parameters["inverse_temperature"])
        return _softmax(values * inverse_temperature)


@dataclass(frozen=True, slots=True)
class CalibrationFit:
    """Fitted calibrator, audited provenance, and deterministic report payload."""

    calibrator: FrozenProbabilityCalibrator
    method: str
    sample_count: int
    group_count: int
    class_sample_counts: tuple[tuple[str, int], ...]
    class_group_counts: tuple[tuple[str, int], ...]
    independent_prediction_artifact_fingerprint: str
    calibration_fingerprint: str
    scalar_parameters: Mapping[str, float]
    array_parameters: Mapping[str, Any]
    metrics: Mapping[str, float]
    report: pl.DataFrame
    semantic_payload: Mapping[str, object]
    array_archive_bytes: bytes


@dataclass(frozen=True, slots=True)
class CalibrationArtifactPaths:
    """Committed paths and identity for one immutable calibration artifact."""

    directory: Path
    manifest_path: Path
    arrays_path: Path
    report_path: Path
    calibration_fingerprint: str


@dataclass(frozen=True, slots=True)
class LoadedCalibration:
    """Validated runtime plus its immutable reliability report and manifest."""

    calibrator: FrozenProbabilityCalibrator
    report: pl.DataFrame
    manifest: Mapping[str, object]


def calibration_report_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "calibration_version": pl.String,
        "calibration_fingerprint": pl.String,
        "classifier_fingerprint": pl.String,
        "independent_prediction_artifact_fingerprint": pl.String,
        "split_fingerprint": pl.String,
        "target_task": pl.String,
        "route": pl.String,
        "dataset_split": pl.String,
        "method": pl.String,
        "probability_kind": pl.String,
        "class_label": pl.String,
        "class_index": pl.Int32,
        "bin_index": pl.Int32,
        "bin_lower_bound": pl.Float64,
        "bin_upper_bound": pl.Float64,
        "unweighted_count": pl.Int64,
        "weighted_count": pl.Float64,
        "mean_prediction": pl.Float64,
        "observed_frequency": pl.Float64,
        "brier_contribution": pl.Float64,
        "log_loss_contribution": pl.Float64,
        "ece_contribution": pl.Float64,
    }


def fit_probability_calibrator(
    predictions: Sequence[CalibrationPrediction],
    fold_audits: Sequence[CalibrationFoldAudit],
    config: CalibrationConfig,
) -> CalibrationFit:
    """Fit a calibrator only after proving group-aware held-out score provenance."""

    if not isinstance(config, CalibrationConfig):
        raise TypeError("config must be a CalibrationConfig")
    validated = _validate_calibration_inputs(predictions, fold_audits, config)
    np = _load_numpy()
    scores = validated["scores"]
    class_indices = validated["class_indices"]
    weights = validated["weights"]
    assert isinstance(scores, np.ndarray)
    assert isinstance(class_indices, np.ndarray)
    assert isinstance(weights, np.ndarray)

    method = config.resolved_method
    scalar_parameters: dict[str, float] = {}
    array_parameters: dict[str, Any] = {}
    optimizer: dict[str, object]
    if method == SIGMOID_METHOD:
        positive_index = config.class_labels.index(str(config.positive_class_label))
        targets = (class_indices == positive_index).astype(np.float64)
        slope, intercept, iterations, converged = _fit_platt_sigmoid(
            scores[:, 0], targets, weights
        )
        scalar_parameters = {"intercept": intercept, "slope": slope}
        optimizer = {
            "algorithm": "platt-smoothed-newton-line-search-v1",
            "converged": converged,
            "iteration_count": iterations,
            "maximum_iterations": 100,
        }
    elif method == ISOTONIC_METHOD:
        sample_count = int(scores.shape[0])
        group_count = int(validated["group_count"])
        if sample_count < MIN_ISOTONIC_SAMPLE_COUNT:
            raise ValueError(
                "isotonic calibration requires at least 1000 independent predictions"
            )
        if group_count < MIN_ISOTONIC_GROUP_COUNT:
            raise ValueError(
                "isotonic calibration requires at least 200 independent groups"
            )
        positive_index = config.class_labels.index(str(config.positive_class_label))
        targets = (class_indices == positive_index).astype(np.float64)
        thresholds, values = _fit_isotonic(scores[:, 0], targets, weights)
        array_parameters = {
            ISOTONIC_THRESHOLDS_ARRAY: thresholds,
            ISOTONIC_VALUES_ARRAY: values,
        }
        optimizer = {
            "algorithm": "weighted-pava-piecewise-linear-v1",
            "minimum_group_count": MIN_ISOTONIC_GROUP_COUNT,
            "minimum_sample_count": MIN_ISOTONIC_SAMPLE_COUNT,
        }
    else:
        inverse_temperature, iterations = _fit_temperature(
            scores,
            class_indices,
            weights,
        )
        scalar_parameters = {
            "inverse_temperature": inverse_temperature,
            "temperature": 1.0 / inverse_temperature,
        }
        optimizer = {
            "algorithm": "bounded-golden-section-log-inverse-temperature-v1",
            "iteration_count": iterations,
            "log_inverse_temperature_bounds": [
                TEMPERATURE_LOG_INVERSE_MIN,
                TEMPERATURE_LOG_INVERSE_MAX,
            ],
        }

    arrays = {
        name: np.ascontiguousarray(value, dtype=np.dtype("<f8"))
        for name, value in sorted(array_parameters.items())
    }
    archive_bytes = deterministic_numeric_npz(arrays)
    if len(archive_bytes) > MAX_CALIBRATION_ARRAY_ARCHIVE_BYTES:
        raise ValueError("calibration array archive exceeds the configured size limit")
    array_entries = {
        name: numeric_array_manifest_entry(value)
        for name, value in sorted(arrays.items())
    }
    provisional = FrozenProbabilityCalibrator(
        calibration_fingerprint="sha256:" + "0" * 64,
        classifier_fingerprint=config.classifier_fingerprint,
        split_fingerprint=config.split_fingerprint,
        target_task=config.target_task,
        route=config.route,
        method=method,
        class_labels=config.class_labels,
        positive_class_label=config.positive_class_label,
        scalar_parameters=MappingProxyType(dict(scalar_parameters)),
        array_parameters=MappingProxyType(arrays),
    )
    probabilities = provisional.predict_proba(scores)
    metrics = _probability_metrics(
        probabilities,
        class_indices,
        weights,
        reliability_bin_count=config.reliability_bin_count,
        uncalibrated_scores=scores if method == TEMPERATURE_METHOD else None,
    )
    semantic_payload = _fit_semantic_payload(
        config=config,
        method=method,
        validated=validated,
        scalar_parameters=scalar_parameters,
        optimizer=optimizer,
        array_entries=array_entries,
        archive_bytes=archive_bytes,
        metrics=metrics,
    )
    calibration_fingerprint = canonical_semantic_fingerprint(semantic_payload)
    calibrator = FrozenProbabilityCalibrator(
        calibration_fingerprint=calibration_fingerprint,
        classifier_fingerprint=config.classifier_fingerprint,
        split_fingerprint=config.split_fingerprint,
        target_task=config.target_task,
        route=config.route,
        method=method,
        class_labels=config.class_labels,
        positive_class_label=config.positive_class_label,
        scalar_parameters=MappingProxyType(dict(scalar_parameters)),
        array_parameters=MappingProxyType(arrays),
    )
    report = _build_calibration_report(
        probabilities=probabilities,
        class_indices=class_indices,
        weights=weights,
        config=config,
        method=method,
        calibration_fingerprint=calibration_fingerprint,
        prediction_fingerprint=str(
            validated["independent_prediction_artifact_fingerprint"]
        ),
    )
    _LOGGER.info(
        "calibration_fit_complete method=%s task=%s route=%s samples=%d groups=%d "
        "calibration_fingerprint=%s",
        method,
        config.target_task,
        config.route,
        int(validated["sample_count"]),
        int(validated["group_count"]),
        calibration_fingerprint,
    )
    return CalibrationFit(
        calibrator=calibrator,
        method=method,
        sample_count=int(validated["sample_count"]),
        group_count=int(validated["group_count"]),
        class_sample_counts=tuple(validated["class_sample_counts"]),
        class_group_counts=tuple(validated["class_group_counts"]),
        independent_prediction_artifact_fingerprint=str(
            validated["independent_prediction_artifact_fingerprint"]
        ),
        calibration_fingerprint=calibration_fingerprint,
        scalar_parameters=MappingProxyType(dict(scalar_parameters)),
        array_parameters=MappingProxyType(arrays),
        metrics=MappingProxyType(dict(metrics)),
        report=report,
        semantic_payload=MappingProxyType(dict(semantic_payload)),
        array_archive_bytes=archive_bytes,
    )


def write_probability_calibrator(
    fit: CalibrationFit,
    directory: str | Path,
    *,
    git_sha: str,
    created_at: datetime | None = None,
    decision_policy: Mapping[str, object] | None = None,
) -> CalibrationArtifactPaths:
    """Write arrays, reliability report, then the canonical commit manifest."""

    if not isinstance(fit, CalibrationFit):
        raise TypeError("fit must be a CalibrationFit")
    if (
        canonical_semantic_fingerprint(fit.semantic_payload)
        != fit.calibration_fingerprint
    ):
        raise ValueError("calibration fit fingerprint is inconsistent")
    if (
        bytes_sha256(fit.array_archive_bytes)
        != _mapping(fit.semantic_payload["arrays"], field="semantic_payload.arrays")[
            "sha256"
        ]
    ):
        raise ValueError("calibration fit array archive is inconsistent")
    timestamp = _creation_timestamp(created_at)
    commit_sha = _git_sha(git_sha)
    report_bytes = _report_parquet_bytes(fit.report)
    if len(report_bytes) > MAX_CALIBRATION_REPORT_BYTES:
        raise ValueError("calibration report exceeds the configured size limit")
    policy_payload = (
        _pending_decision_policy(fit)
        if decision_policy is None
        else _validated_embedded_decision_policy(
            decision_policy,
            calibration_fingerprint=fit.calibration_fingerprint,
            classifier_fingerprint=fit.calibrator.classifier_fingerprint,
            split_fingerprint=fit.calibrator.split_fingerprint,
            target_task=fit.calibrator.target_task,
            route=fit.calibrator.route,
        )
    )
    manifest = {
        "schema_version": CALIBRATION_MANIFEST_SCHEMA_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "calibration_fingerprint": fit.calibration_fingerprint,
        "created_at": timestamp,
        "git_sha": commit_sha,
        "identity": fit.semantic_payload["identity"],
        "provenance": fit.semantic_payload["provenance"],
        "fitting": fit.semantic_payload["fitting"],
        "parameters": fit.semantic_payload["parameters"],
        "metrics": fit.semantic_payload["metrics"],
        "arrays": fit.semantic_payload["arrays"],
        "decision_policy": policy_payload,
        "report": {
            "schema_version": CALIBRATION_REPORT_SCHEMA_VERSION,
            "file_name": CALIBRATION_REPORT_FILE,
            "uri": CALIBRATION_REPORT_FILE,
            "sha256": bytes_sha256(report_bytes),
            "size_bytes": len(report_bytes),
            "row_count": fit.report.height,
        },
        "libraries": _library_versions(),
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    if len(manifest_bytes) > MAX_CALIBRATION_MANIFEST_BYTES:
        raise ValueError("calibration manifest exceeds the configured size limit")

    output = Path(directory)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir(mode=0o700, exist_ok=False)
        arrays_path = output / CALIBRATION_ARRAYS_FILE
        report_path = output / CALIBRATION_REPORT_FILE
        manifest_path = output / CALIBRATION_MANIFEST_FILE
        _write_exclusive_atomic(arrays_path, fit.array_archive_bytes)
        _write_exclusive_atomic(report_path, report_bytes)
        _fsync_directory(output)
        _write_exclusive_atomic(manifest_path, manifest_bytes)
        _fsync_directory(output)
        _fsync_directory(output.parent)
    except BaseException:
        if output.is_dir() and not output.is_symlink():
            shutil.rmtree(output)
        raise
    return CalibrationArtifactPaths(
        directory=output,
        manifest_path=manifest_path,
        arrays_path=arrays_path,
        report_path=report_path,
        calibration_fingerprint=fit.calibration_fingerprint,
    )


def load_probability_calibrator(
    directory: str | Path,
    *,
    expected_calibration_fingerprint: str | None = None,
    expected_classifier_fingerprint: str | None = None,
    expected_split_fingerprint: str | None = None,
) -> LoadedCalibration:
    """Load a strict numeric calibrator and report without importing sklearn."""

    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("calibration artifact path must be a real directory")
    expected_files = {
        CALIBRATION_MANIFEST_FILE,
        CALIBRATION_ARRAYS_FILE,
        CALIBRATION_REPORT_FILE,
    }
    try:
        entries = tuple(root.iterdir())
    except OSError as exc:
        raise ValueError("calibration artifact directory is unreadable") from exc
    if {item.name for item in entries} != expected_files:
        raise ValueError("calibration artifact directory has unexpected files")
    if any(item.is_symlink() or not item.is_file() for item in entries):
        raise ValueError("calibration artifact files must be regular files")

    manifest = _read_manifest(root / CALIBRATION_MANIFEST_FILE)
    try:
        validated = _validate_manifest(manifest)
    except ValueError as exc:
        raise ValueError(f"calibration manifest validation failed: {exc}") from exc
    calibration_fingerprint = str(manifest["calibration_fingerprint"])
    semantic_payload = {
        key: manifest[key]
        for key in (
            "calibration_version",
            "identity",
            "provenance",
            "fitting",
            "parameters",
            "metrics",
            "arrays",
        )
    }
    if canonical_semantic_fingerprint(semantic_payload) != calibration_fingerprint:
        raise ValueError("calibration fingerprint does not match the manifest")
    _match_expected_fingerprint(
        calibration_fingerprint,
        expected_calibration_fingerprint,
        field="calibration_fingerprint",
    )

    identity = validated["identity"]
    provenance = validated["provenance"]
    arrays_metadata = validated["arrays"]
    parameters = validated["parameters"]
    assert isinstance(identity, Mapping)
    assert isinstance(provenance, Mapping)
    assert isinstance(arrays_metadata, Mapping)
    assert isinstance(parameters, Mapping)
    classifier_fingerprint = str(identity["classifier_fingerprint"])
    split_fingerprint = str(provenance["split_fingerprint"])
    _match_expected_fingerprint(
        classifier_fingerprint,
        expected_classifier_fingerprint,
        field="classifier_fingerprint",
    )
    _match_expected_fingerprint(
        split_fingerprint,
        expected_split_fingerprint,
        field="split_fingerprint",
    )

    arrays_path = root / CALIBRATION_ARRAYS_FILE
    arrays_value = _bounded_file_bytes(
        arrays_path,
        maximum=MAX_CALIBRATION_ARRAY_ARCHIVE_BYTES,
        label="calibration array archive",
    )
    if len(arrays_value) != arrays_metadata["size_bytes"]:
        raise ValueError("calibration array archive size does not match the manifest")
    if bytes_sha256(arrays_value) != arrays_metadata["sha256"]:
        raise ValueError(
            "calibration array archive checksum does not match the manifest"
        )
    array_specs = _mapping(arrays_metadata["entries"], field="arrays.entries")
    method = str(identity["method"])
    expected_shapes, expected_dtypes = _expected_array_contract(method, array_specs)
    arrays = load_numeric_npz(
        arrays_value,
        specs={
            name: _mapping(spec, field=f"arrays.entries.{name}")
            for name, spec in array_specs.items()
        },
        expected_dtypes=expected_dtypes,
        expected_shapes=expected_shapes,
        max_uncompressed_bytes=MAX_CALIBRATION_ARRAY_UNCOMPRESSED_BYTES,
        artifact_label="calibration",
        max_header_bytes=MAX_NUMPY_HEADER_BYTES,
    )
    _validate_loaded_arrays(method, arrays)
    for array in arrays.values():
        array.setflags(write=False)

    scalar_parameters = {
        str(key): _finite_number(value, field=f"parameters.scalar.{key}")
        for key, value in _mapping(
            parameters["scalar"], field="parameters.scalar"
        ).items()
    }
    _validate_scalar_parameters(method, scalar_parameters)
    class_labels = _string_tuple(
        identity["class_labels"], field="identity.class_labels"
    )
    positive = identity["positive_class_label"]
    calibrator = FrozenProbabilityCalibrator(
        calibration_fingerprint=calibration_fingerprint,
        classifier_fingerprint=classifier_fingerprint,
        split_fingerprint=split_fingerprint,
        target_task=str(identity["target_task"]),
        route=str(identity["route"]),
        method=method,
        class_labels=class_labels,
        positive_class_label=None if positive is None else str(positive),
        scalar_parameters=MappingProxyType(scalar_parameters),
        array_parameters=MappingProxyType(arrays),
    )

    report_metadata = validated["report"]
    assert isinstance(report_metadata, Mapping)
    report_path = root / CALIBRATION_REPORT_FILE
    report_value = _bounded_file_bytes(
        report_path,
        maximum=MAX_CALIBRATION_REPORT_BYTES,
        label="calibration report",
    )
    if len(report_value) != report_metadata["size_bytes"]:
        raise ValueError("calibration report size does not match the manifest")
    if bytes_sha256(report_value) != report_metadata["sha256"]:
        raise ValueError("calibration report checksum does not match the manifest")
    try:
        report = pl.read_parquet(io.BytesIO(report_value))
    except Exception as exc:
        raise ValueError("calibration report is invalid Parquet") from exc
    _validate_loaded_report(
        report,
        calibration_fingerprint=calibration_fingerprint,
        classifier_fingerprint=classifier_fingerprint,
        split_fingerprint=split_fingerprint,
        prediction_fingerprint=str(
            provenance["independent_prediction_artifact_fingerprint"]
        ),
        target_task=str(identity["target_task"]),
        route=str(identity["route"]),
        row_count=int(report_metadata["row_count"]),
        class_labels=class_labels,
        method=method,
        reliability_bin_count=int(
            _mapping(validated["fitting"], field="fitting")["reliability_bin_count"]
        ),
    )
    return LoadedCalibration(
        calibrator=calibrator,
        report=report,
        manifest=MappingProxyType(dict(manifest)),
    )


def _validate_calibration_inputs(
    predictions: Sequence[CalibrationPrediction],
    fold_audits: Sequence[CalibrationFoldAudit],
    config: CalibrationConfig,
) -> dict[str, object]:
    np = _load_numpy()
    if not predictions:
        raise ValueError("calibration predictions must not be empty")
    if len(fold_audits) < 2:
        raise ValueError("group-aware calibration requires at least two folds")
    class_to_index = {label: index for index, label in enumerate(config.class_labels)}
    expected_score_count = (
        1 if len(config.class_labels) == 2 else len(config.class_labels)
    )
    normalized_predictions: list[dict[str, object]] = []
    prediction_ids: set[str] = set()
    group_folds: dict[str, int] = {}
    source_item_groups: dict[str, str] = {}
    class_groups: dict[str, set[str]] = defaultdict(set)
    class_counts: Counter[str] = Counter()
    for prediction in predictions:
        if not isinstance(prediction, CalibrationPrediction):
            raise TypeError("predictions must contain CalibrationPrediction values")
        prediction_id = _required_text(prediction.prediction_id, field="prediction_id")
        if prediction_id in prediction_ids:
            raise ValueError("calibration prediction IDs must be unique")
        prediction_ids.add(prediction_id)
        source_item_id = _required_text(
            prediction.source_item_id, field="source_item_id"
        )
        group_id = _required_text(
            prediction.leakage_component_id,
            field="leakage_component_id",
        )
        prior_group = source_item_groups.setdefault(source_item_id, group_id)
        if prior_group != group_id:
            raise ValueError("one source item cannot cross leakage components")
        fold_index = _nonnegative_integer(prediction.fold_index, field="fold_index")
        if prediction.dataset_split != CALIBRATION_PARTITION:
            raise ValueError(
                "calibrator input must come from the calibration partition"
            )
        class_label = _required_text(
            prediction.true_class_label,
            field="true_class_label",
        )
        if class_label not in class_to_index:
            raise ValueError("calibration prediction has an unknown true class")
        scores = tuple(
            _finite_number(score, field="decision_scores")
            for score in prediction.decision_scores
        )
        if len(scores) != expected_score_count:
            raise ValueError("calibration decision-score width is inconsistent")
        weight = _finite_number(prediction.sample_weight, field="sample_weight")
        if weight <= 0.0:
            raise ValueError("calibration sample weights must be positive")
        prior_fold = group_folds.setdefault(group_id, fold_index)
        if prior_fold != fold_index:
            raise ValueError("one leakage component cannot cross validation folds")
        class_groups[class_label].add(group_id)
        class_counts[class_label] += 1
        normalized_predictions.append(
            {
                "prediction_id": prediction_id,
                "source_item_id": source_item_id,
                "leakage_component_id": group_id,
                "fold_index": fold_index,
                "dataset_split": CALIBRATION_PARTITION,
                "true_class_label": class_label,
                "decision_scores": list(scores),
                "sample_weight": weight,
            }
        )

    normalized_audits: list[dict[str, object]] = []
    audits_by_fold: dict[int, tuple[set[str], set[str]]] = {}
    validation_owner: dict[str, int] = {}
    for audit in fold_audits:
        if not isinstance(audit, CalibrationFoldAudit):
            raise TypeError("fold_audits must contain CalibrationFoldAudit values")
        fold_index = _nonnegative_integer(audit.fold_index, field="fold_index")
        if fold_index in audits_by_fold:
            raise ValueError("calibration fold indices must be unique")
        fit_groups = _canonical_group_ids(
            audit.estimator_fit_group_ids,
            field="estimator_fit_group_ids",
        )
        validation_groups = _canonical_group_ids(
            audit.validation_group_ids,
            field="validation_group_ids",
        )
        if not fit_groups or not validation_groups:
            raise ValueError("calibration folds require fit and validation groups")
        overlap = set(fit_groups) & set(validation_groups)
        if overlap:
            raise ValueError(
                "estimator-fit and validation groups overlap within a fold"
            )
        for group_id in validation_groups:
            owner = validation_owner.setdefault(group_id, fold_index)
            if owner != fold_index:
                raise ValueError("one validation group cannot belong to multiple folds")
        audits_by_fold[fold_index] = (set(fit_groups), set(validation_groups))
        semantics = {
            "fold_index": fold_index,
            "estimator_fit_group_ids": list(fit_groups),
            "validation_group_ids": list(validation_groups),
        }
        normalized_audits.append(
            {
                "fold_index": fold_index,
                "fold_fingerprint": canonical_semantic_fingerprint(semantics),
                "estimator_fit_group_count": len(fit_groups),
                "validation_group_count": len(validation_groups),
                "estimator_fit_groups_fingerprint": canonical_semantic_fingerprint(
                    {"group_ids": list(fit_groups)}
                ),
                "validation_groups_fingerprint": canonical_semantic_fingerprint(
                    {"group_ids": list(validation_groups)}
                ),
                "_semantics": semantics,
            }
        )

    expected_fold_indices = set(range(len(audits_by_fold)))
    if set(audits_by_fold) != expected_fold_indices:
        raise ValueError("calibration fold indices must be contiguous from zero")

    prediction_groups_by_fold: dict[int, set[str]] = defaultdict(set)
    for row in normalized_predictions:
        fold_index = int(row["fold_index"])
        group_id = str(row["leakage_component_id"])
        if fold_index not in audits_by_fold:
            raise ValueError("calibration prediction references an unaudited fold")
        _fit, validation = audits_by_fold[fold_index]
        if group_id not in validation:
            raise ValueError("prediction validation group does not belong to its fold")
        prediction_groups_by_fold[fold_index].add(group_id)
    for fold_index, (_fit, validation) in audits_by_fold.items():
        if prediction_groups_by_fold[fold_index] != validation:
            raise ValueError(
                "calibration fold validation groups do not exactly match predictions"
            )
    if set(validation_owner) != set(group_folds):
        raise ValueError("calibration validation groups do not cover every prediction")

    class_sample_counts = tuple(
        (label, class_counts[label]) for label in config.class_labels
    )
    class_group_counts = tuple(
        (label, len(class_groups[label])) for label in config.class_labels
    )
    for label, count in class_sample_counts:
        if count == 0:
            raise ValueError(f"calibration class has no predictions: {label}")
    for label, count in class_group_counts:
        if count < config.minimum_class_group_count:
            raise ValueError(
                "each calibration class requires the configured independent group count"
            )

    normalized_predictions.sort(key=lambda row: str(row["prediction_id"]))
    normalized_audits.sort(key=lambda row: int(row["fold_index"]))
    prediction_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": "calibration-oof-predictions-v1.0.0",
            "classifier_fingerprint": config.classifier_fingerprint,
            "split_fingerprint": config.split_fingerprint,
            "oof_policy": CALIBRATION_OOF_POLICY,
            "predictions": normalized_predictions,
            "folds": [row["_semantics"] for row in normalized_audits],
        }
    )
    scores = np.asarray(
        [row["decision_scores"] for row in normalized_predictions],
        dtype=np.float64,
    )
    indices = np.asarray(
        [
            class_to_index[str(row["true_class_label"])]
            for row in normalized_predictions
        ],
        dtype=np.int64,
    )
    weights = np.asarray(
        [row["sample_weight"] for row in normalized_predictions],
        dtype=np.float64,
    )
    return {
        "scores": scores,
        "class_indices": indices,
        "weights": weights,
        "sample_count": len(normalized_predictions),
        "group_count": len(group_folds),
        "class_sample_counts": class_sample_counts,
        "class_group_counts": class_group_counts,
        "independent_prediction_artifact_fingerprint": prediction_fingerprint,
        "folds": tuple(
            {key: value for key, value in row.items() if key != "_semantics"}
            for row in normalized_audits
        ),
    }


def _fit_platt_sigmoid(
    scores: Any,
    targets: Any,
    weights: Any,
) -> tuple[float, float, int, bool]:
    """Fit Platt-smoothed logistic parameters with deterministic Newton steps."""

    np = _load_numpy()
    positive_prior = float(weights[targets == 1.0].sum())
    negative_prior = float(weights[targets == 0.0].sum())
    high_target = (positive_prior + 1.0) / (positive_prior + 2.0)
    low_target = 1.0 / (negative_prior + 2.0)
    smoothed = np.where(targets == 1.0, high_target, low_target)
    slope = 0.0
    intercept = log((positive_prior + 1.0) / (negative_prior + 1.0))
    ridge = 1e-12
    tolerance = 1e-10

    def objective(candidate_slope: float, candidate_intercept: float) -> float:
        logits = candidate_slope * scores + candidate_intercept
        losses = np.logaddexp(0.0, logits) - smoothed * logits
        return float((weights * losses).sum() + 0.5 * ridge * candidate_slope**2)

    current = objective(slope, intercept)
    converged = False
    iterations = 0
    for iteration in range(1, 101):
        iterations = iteration
        logits = slope * scores + intercept
        probabilities = _stable_sigmoid_array(logits)
        residual = weights * (probabilities - smoothed)
        curvature = weights * probabilities * (1.0 - probabilities)
        gradient_slope = float((residual * scores).sum() + ridge * slope)
        gradient_intercept = float(residual.sum())
        if max(abs(gradient_slope), abs(gradient_intercept)) < tolerance:
            converged = True
            break
        hessian_ss = float((curvature * scores * scores).sum() + ridge)
        hessian_si = float((curvature * scores).sum())
        hessian_ii = float(curvature.sum() + ridge)
        determinant = hessian_ss * hessian_ii - hessian_si * hessian_si
        if determinant <= 0.0 or not isfinite(determinant):
            raise ValueError("sigmoid calibration Hessian is singular")
        delta_slope = (
            hessian_ii * gradient_slope - hessian_si * gradient_intercept
        ) / determinant
        delta_intercept = (
            hessian_ss * gradient_intercept - hessian_si * gradient_slope
        ) / determinant
        directional_derivative = -(
            gradient_slope * delta_slope + gradient_intercept * delta_intercept
        )
        step = 1.0
        accepted = False
        while step >= 1e-10:
            candidate_slope = slope - step * delta_slope
            candidate_intercept = intercept - step * delta_intercept
            candidate = objective(candidate_slope, candidate_intercept)
            if candidate <= current + 1e-4 * step * directional_derivative:
                slope = candidate_slope
                intercept = candidate_intercept
                current = candidate
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    if not isfinite(slope) or not isfinite(intercept):
        raise ValueError("sigmoid calibration produced non-finite parameters")
    return float(slope), float(intercept), iterations, converged


def _fit_isotonic(scores: Any, targets: Any, weights: Any) -> tuple[Any, Any]:
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "scikit-learn is required to fit isotonic calibration"
        ) from exc

    np = _load_numpy()
    estimator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        increasing=True,
        out_of_bounds="clip",
    )
    estimator.fit(scores, targets, sample_weight=weights)
    thresholds = np.ascontiguousarray(estimator.X_thresholds_, dtype=np.dtype("<f8"))
    values = np.ascontiguousarray(estimator.y_thresholds_, dtype=np.dtype("<f8"))
    if thresholds.size < 2 or thresholds.shape != values.shape:
        raise ValueError("isotonic calibration produced an invalid knot sequence")
    return thresholds, values


def _fit_temperature(
    scores: Any,
    class_indices: Any,
    weights: Any,
) -> tuple[float, int]:
    np = _load_numpy()
    total_weight = float(weights.sum())

    def objective(log_inverse_temperature: float) -> float:
        inverse = exp(log_inverse_temperature)
        scaled = scores * inverse
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        log_partition = np.log(np.exp(shifted).sum(axis=1))
        selected = shifted[np.arange(shifted.shape[0]), class_indices]
        return float((weights * (log_partition - selected)).sum() / total_weight)

    left = TEMPERATURE_LOG_INVERSE_MIN
    right = TEMPERATURE_LOG_INVERSE_MAX
    ratio = (5.0**0.5 - 1.0) / 2.0
    inner_left = right - ratio * (right - left)
    inner_right = left + ratio * (right - left)
    value_left = objective(inner_left)
    value_right = objective(inner_right)
    for _iteration in range(DEFAULT_TEMPERATURE_OPTIMIZATION_ITERATIONS):
        if value_left <= value_right:
            right = inner_right
            inner_right = inner_left
            value_right = value_left
            inner_left = right - ratio * (right - left)
            value_left = objective(inner_left)
        else:
            left = inner_left
            inner_left = inner_right
            value_left = value_right
            inner_right = left + ratio * (right - left)
            value_right = objective(inner_right)
    candidates = (
        (0.0, objective(0.0)),
        (left, objective(left)),
        (right, objective(right)),
        (inner_left, value_left),
        (inner_right, value_right),
    )
    best_log_inverse, _best_loss = min(candidates, key=lambda item: (item[1], item[0]))
    inverse_temperature = exp(best_log_inverse)
    if not isfinite(inverse_temperature) or inverse_temperature <= 0.0:
        raise ValueError("temperature calibration produced an invalid scale")
    return float(inverse_temperature), DEFAULT_TEMPERATURE_OPTIMIZATION_ITERATIONS


def _probability_metrics(
    probabilities: Any,
    class_indices: Any,
    weights: Any,
    *,
    reliability_bin_count: int,
    uncalibrated_scores: Any | None,
) -> dict[str, float]:
    np = _load_numpy()
    class_count = int(probabilities.shape[1])
    one_hot = np.eye(class_count, dtype=np.float64)[class_indices]
    total_weight = float(weights.sum())
    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    brier = float(
        (weights[:, None] * (probabilities - one_hot) ** 2).sum()
        / (total_weight * class_count)
    )
    log_loss = float(
        -(
            weights * np.log(clipped[np.arange(probabilities.shape[0]), class_indices])
        ).sum()
        / total_weight
    )
    classwise_log_loss = float(
        -(
            weights[:, None]
            * (one_hot * np.log(clipped) + (1.0 - one_hot) * np.log(1.0 - clipped))
        ).sum()
        / (total_weight * class_count)
    )
    metrics = {
        "brier_score": brier,
        "classwise_log_loss": classwise_log_loss,
        "expected_calibration_error": _macro_ece(
            probabilities,
            one_hot,
            weights,
            reliability_bin_count,
        ),
        "log_loss": log_loss,
    }
    if uncalibrated_scores is not None:
        uncalibrated = _softmax(uncalibrated_scores)
        uncalibrated_clipped = np.clip(uncalibrated, 1e-15, 1.0 - 1e-15)
        metrics["uncalibrated_softmax_log_loss"] = float(
            -(
                weights
                * np.log(
                    uncalibrated_clipped[
                        np.arange(uncalibrated.shape[0]), class_indices
                    ]
                )
            ).sum()
            / total_weight
        )
    return metrics


def _macro_ece(
    probabilities: Any,
    one_hot: Any,
    weights: Any,
    bin_count: int,
) -> float:
    np = _load_numpy()
    total_weight = float(weights.sum())
    class_ece = []
    for class_index in range(probabilities.shape[1]):
        values = probabilities[:, class_index]
        labels = one_hot[:, class_index]
        bins = np.minimum((values * bin_count).astype(np.int64), bin_count - 1)
        ece = 0.0
        for bin_index in range(bin_count):
            selected = bins == bin_index
            if not bool(selected.any()):
                continue
            selected_weights = weights[selected]
            weight = float(selected_weights.sum())
            mean_prediction = float(
                (selected_weights * values[selected]).sum() / weight
            )
            observed = float((selected_weights * labels[selected]).sum() / weight)
            ece += weight / total_weight * abs(mean_prediction - observed)
        class_ece.append(ece)
    return float(sum(class_ece) / len(class_ece))


def _fit_semantic_payload(
    *,
    config: CalibrationConfig,
    method: str,
    validated: Mapping[str, object],
    scalar_parameters: Mapping[str, float],
    optimizer: Mapping[str, object],
    array_entries: Mapping[str, Mapping[str, object]],
    archive_bytes: bytes,
    metrics: Mapping[str, float],
) -> dict[str, object]:
    return {
        "calibration_version": CALIBRATION_VERSION,
        "identity": {
            "classifier_fingerprint": config.classifier_fingerprint,
            "target_task": config.target_task,
            "route": config.route,
            "method": method,
            "class_labels": list(config.class_labels),
            "positive_class_label": config.positive_class_label,
        },
        "provenance": {
            "independent_prediction_artifact_fingerprint": validated[
                "independent_prediction_artifact_fingerprint"
            ],
            "split_fingerprint": config.split_fingerprint,
            "dataset_split": CALIBRATION_PARTITION,
            "oof_policy": CALIBRATION_OOF_POLICY,
            "folds": list(validated["folds"]),
        },
        "fitting": {
            "sample_count": validated["sample_count"],
            "class_count": len(config.class_labels),
            "group_count": validated["group_count"],
            "class_sample_counts": _counts_payload(validated["class_sample_counts"]),
            "class_group_counts": _counts_payload(validated["class_group_counts"]),
            "minimum_class_group_count": config.minimum_class_group_count,
            "reliability_bin_count": config.reliability_bin_count,
        },
        "parameters": {
            "score_input_kind": CALIBRATION_SCORE_INPUT_KIND,
            "scalar": dict(sorted(scalar_parameters.items())),
            "array_names": sorted(array_entries),
            "optimizer": dict(optimizer),
        },
        "metrics": dict(sorted(metrics.items())),
        "arrays": {
            "schema_version": CALIBRATION_ARRAY_SCHEMA_VERSION,
            "file_name": CALIBRATION_ARRAYS_FILE,
            "uri": CALIBRATION_ARRAYS_FILE,
            "sha256": bytes_sha256(archive_bytes),
            "size_bytes": len(archive_bytes),
            "entries": {
                name: dict(spec) for name, spec in sorted(array_entries.items())
            },
        },
    }


def _build_calibration_report(
    *,
    probabilities: Any,
    class_indices: Any,
    weights: Any,
    config: CalibrationConfig,
    method: str,
    calibration_fingerprint: str,
    prediction_fingerprint: str,
) -> pl.DataFrame:
    np = _load_numpy()
    class_count = len(config.class_labels)
    one_hot = np.eye(class_count, dtype=np.float64)[class_indices]
    total_weight = float(weights.sum())
    rows: list[dict[str, object]] = []
    for class_index, class_label in enumerate(config.class_labels):
        values = probabilities[:, class_index]
        labels = one_hot[:, class_index]
        bin_indices = np.minimum(
            (values * config.reliability_bin_count).astype(np.int64),
            config.reliability_bin_count - 1,
        )
        for bin_index in range(config.reliability_bin_count):
            selected = bin_indices == bin_index
            count = int(selected.sum())
            selected_weights = weights[selected]
            weighted_count = float(selected_weights.sum())
            if count:
                selected_values = values[selected]
                selected_labels = labels[selected]
                mean_prediction = float(
                    (selected_weights * selected_values).sum() / weighted_count
                )
                observed = float(
                    (selected_weights * selected_labels).sum() / weighted_count
                )
                clipped = np.clip(selected_values, 1e-15, 1.0 - 1e-15)
                brier_contribution = float(
                    (selected_weights * (selected_values - selected_labels) ** 2).sum()
                    / (total_weight * class_count)
                )
                log_loss_contribution = float(
                    -(
                        selected_weights
                        * (
                            selected_labels * np.log(clipped)
                            + (1.0 - selected_labels) * np.log(1.0 - clipped)
                        )
                    ).sum()
                    / (total_weight * class_count)
                )
                ece_contribution = float(
                    weighted_count
                    / total_weight
                    * abs(mean_prediction - observed)
                    / class_count
                )
            else:
                mean_prediction = None
                observed = None
                brier_contribution = 0.0
                log_loss_contribution = 0.0
                ece_contribution = 0.0
            rows.append(
                {
                    "schema_version": CALIBRATION_REPORT_SCHEMA_VERSION,
                    "calibration_version": CALIBRATION_VERSION,
                    "calibration_fingerprint": calibration_fingerprint,
                    "classifier_fingerprint": config.classifier_fingerprint,
                    "independent_prediction_artifact_fingerprint": prediction_fingerprint,
                    "split_fingerprint": config.split_fingerprint,
                    "target_task": config.target_task,
                    "route": config.route,
                    "dataset_split": CALIBRATION_PARTITION,
                    "method": method,
                    "probability_kind": CALIBRATED_PROBABILITY_KIND,
                    "class_label": class_label,
                    "class_index": class_index,
                    "bin_index": bin_index,
                    "bin_lower_bound": bin_index / config.reliability_bin_count,
                    "bin_upper_bound": (bin_index + 1) / config.reliability_bin_count,
                    "unweighted_count": count,
                    "weighted_count": weighted_count,
                    "mean_prediction": mean_prediction,
                    "observed_frequency": observed,
                    "brier_contribution": brier_contribution,
                    "log_loss_contribution": log_loss_contribution,
                    "ece_contribution": ece_contribution,
                }
            )
    return pl.DataFrame(rows, schema=calibration_report_schema(), orient="row").sort(
        "class_index", "bin_index"
    )


def _validate_manifest(payload: Mapping[str, object]) -> dict[str, object]:
    _expect_keys(
        payload,
        {
            "arrays",
            "calibration_fingerprint",
            "calibration_version",
            "created_at",
            "decision_policy",
            "fitting",
            "git_sha",
            "identity",
            "libraries",
            "metrics",
            "parameters",
            "provenance",
            "report",
            "schema_version",
        },
        field="manifest",
    )
    if payload["schema_version"] != CALIBRATION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("calibration manifest schema version is incompatible")
    if payload["calibration_version"] != CALIBRATION_VERSION:
        raise ValueError("calibration version is incompatible")
    _sha256(payload["calibration_fingerprint"], field="calibration_fingerprint")
    _validate_creation_timestamp(payload["created_at"])
    _git_sha(payload["git_sha"])

    identity = _mapping(payload["identity"], field="identity")
    _expect_keys(
        identity,
        {
            "classifier_fingerprint",
            "target_task",
            "route",
            "method",
            "class_labels",
            "positive_class_label",
        },
        field="identity",
    )
    _sha256(identity["classifier_fingerprint"], field="identity.classifier_fingerprint")
    task = _required_choice(
        identity["target_task"], field="identity.target_task", allowed=TARGET_TASKS
    )
    route = _required_choice(
        identity["route"], field="identity.route", allowed=REFERENCE_ROUTES
    )
    method = _required_choice(
        identity["method"], field="identity.method", allowed=CALIBRATION_METHODS
    )
    labels = _string_tuple(identity["class_labels"], field="identity.class_labels")
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("identity.class_labels are invalid")
    positive = identity["positive_class_label"]
    if len(labels) == 2:
        if not isinstance(positive, str) or positive not in labels:
            raise ValueError("binary calibration positive class is invalid")
        if method == TEMPERATURE_METHOD:
            raise ValueError("binary calibration method is invalid")
    elif positive is not None or method != TEMPERATURE_METHOD:
        raise ValueError("multiclass calibration identity is invalid")
    if task == "regional_multiclass" and len(labels) < 3:
        raise ValueError("regional calibration class count is invalid")
    del task, route

    provenance = _mapping(payload["provenance"], field="provenance")
    _expect_keys(
        provenance,
        {
            "dataset_split",
            "folds",
            "independent_prediction_artifact_fingerprint",
            "oof_policy",
            "split_fingerprint",
        },
        field="provenance",
    )
    _sha256(
        provenance["independent_prediction_artifact_fingerprint"],
        field="provenance.independent_prediction_artifact_fingerprint",
    )
    _sha256(provenance["split_fingerprint"], field="provenance.split_fingerprint")
    if provenance["dataset_split"] != CALIBRATION_PARTITION:
        raise ValueError("calibration provenance partition is invalid")
    if provenance["oof_policy"] != CALIBRATION_OOF_POLICY:
        raise ValueError("calibration OOF policy is incompatible")
    folds = provenance["folds"]
    if not isinstance(folds, list) or len(folds) < 2:
        raise ValueError("calibration fold provenance is invalid")
    seen_folds: set[int] = set()
    for position, raw_fold in enumerate(folds):
        fold = _mapping(raw_fold, field=f"provenance.folds[{position}]")
        _expect_keys(
            fold,
            {
                "estimator_fit_group_count",
                "estimator_fit_groups_fingerprint",
                "fold_fingerprint",
                "fold_index",
                "validation_group_count",
                "validation_groups_fingerprint",
            },
            field=f"provenance.folds[{position}]",
        )
        fold_index = _nonnegative_integer(
            fold["fold_index"], field="provenance.folds.fold_index"
        )
        if fold_index in seen_folds or fold_index != position:
            raise ValueError("calibration fold order is invalid")
        seen_folds.add(fold_index)
        _positive_integer(
            fold["estimator_fit_group_count"],
            field="provenance.folds.estimator_fit_group_count",
        )
        _positive_integer(
            fold["validation_group_count"],
            field="provenance.folds.validation_group_count",
        )
        for field in (
            "estimator_fit_groups_fingerprint",
            "fold_fingerprint",
            "validation_groups_fingerprint",
        ):
            _sha256(fold[field], field=f"provenance.folds.{field}")

    fitting = _mapping(payload["fitting"], field="fitting")
    _expect_keys(
        fitting,
        {
            "class_count",
            "class_group_counts",
            "class_sample_counts",
            "group_count",
            "minimum_class_group_count",
            "reliability_bin_count",
            "sample_count",
        },
        field="fitting",
    )
    sample_count = _positive_integer(
        fitting["sample_count"], field="fitting.sample_count"
    )
    class_count = _positive_integer(fitting["class_count"], field="fitting.class_count")
    group_count = _positive_integer(fitting["group_count"], field="fitting.group_count")
    minimum_groups = _integer_at_least(
        fitting["minimum_class_group_count"],
        minimum=2,
        field="fitting.minimum_class_group_count",
    )
    _integer_at_least(
        fitting["reliability_bin_count"],
        minimum=2,
        field="fitting.reliability_bin_count",
    )
    if class_count != len(labels) or group_count > sample_count:
        raise ValueError("calibration fitting counts are inconsistent")
    _validate_counts_payload(
        fitting["class_sample_counts"],
        labels=labels,
        total=sample_count,
        minimum=1,
        field="fitting.class_sample_counts",
    )
    group_counts = _validate_counts_payload(
        fitting["class_group_counts"],
        labels=labels,
        total=None,
        minimum=minimum_groups,
        field="fitting.class_group_counts",
    )
    if any(count > group_count for count in group_counts.values()):
        raise ValueError("calibration class group counts are inconsistent")
    if method == ISOTONIC_METHOD and (
        sample_count < MIN_ISOTONIC_SAMPLE_COUNT
        or group_count < MIN_ISOTONIC_GROUP_COUNT
    ):
        raise ValueError("isotonic calibration evidence is insufficient")

    parameters = _mapping(payload["parameters"], field="parameters")
    _expect_keys(
        parameters,
        {"array_names", "optimizer", "scalar", "score_input_kind"},
        field="parameters",
    )
    if parameters["score_input_kind"] != CALIBRATION_SCORE_INPUT_KIND:
        raise ValueError("calibration score input kind is invalid")
    scalar = _mapping(parameters["scalar"], field="parameters.scalar")
    scalar_values = {
        key: _finite_number(value, field=f"parameters.scalar.{key}")
        for key, value in scalar.items()
    }
    _validate_scalar_parameters(method, scalar_values)
    expected_names = (
        [ISOTONIC_THRESHOLDS_ARRAY, ISOTONIC_VALUES_ARRAY]
        if method == ISOTONIC_METHOD
        else []
    )
    if parameters["array_names"] != expected_names:
        raise ValueError("calibration parameter array names are invalid")
    optimizer = _mapping(parameters["optimizer"], field="parameters.optimizer")
    _validate_optimizer(method, optimizer)

    metrics = _mapping(payload["metrics"], field="metrics")
    required_metrics = {
        "brier_score",
        "classwise_log_loss",
        "expected_calibration_error",
        "log_loss",
    }
    if method == TEMPERATURE_METHOD:
        required_metrics.add("uncalibrated_softmax_log_loss")
    if set(metrics) != required_metrics:
        raise ValueError("calibration metric keys are invalid")
    for key, value in metrics.items():
        metric = _finite_number(value, field=f"metrics.{key}")
        if metric < 0.0:
            raise ValueError("calibration metrics cannot be negative")

    arrays = _mapping(payload["arrays"], field="arrays")
    _validate_arrays_manifest(arrays, method)
    decision_policy = _mapping(payload["decision_policy"], field="decision_policy")
    if decision_policy.get("schema_version") == (
        "few-shot-decision-policy-pending-v1.0.0"
    ):
        _validate_pending_decision_policy(decision_policy, payload)
    else:
        _validated_embedded_decision_policy(
            decision_policy,
            calibration_fingerprint=str(payload["calibration_fingerprint"]),
            classifier_fingerprint=str(identity["classifier_fingerprint"]),
            split_fingerprint=str(provenance["split_fingerprint"]),
            target_task=str(identity["target_task"]),
            route=str(identity["route"]),
        )
    report = _mapping(payload["report"], field="report")
    _validate_report_manifest(report)
    libraries = _mapping(payload["libraries"], field="libraries")
    if set(libraries) != {"biominer", "numpy", "polars", "scikit_learn"}:
        raise ValueError("calibration library provenance is invalid")
    for name, version in libraries.items():
        _required_text(version, field=f"libraries.{name}")
    return {
        "identity": identity,
        "provenance": provenance,
        "fitting": fitting,
        "parameters": parameters,
        "arrays": arrays,
        "report": report,
    }


def _validate_arrays_manifest(payload: Mapping[str, object], method: str) -> None:
    _expect_keys(
        payload,
        {"entries", "file_name", "schema_version", "sha256", "size_bytes", "uri"},
        field="arrays",
    )
    if payload["schema_version"] != CALIBRATION_ARRAY_SCHEMA_VERSION:
        raise ValueError("calibration array schema version is incompatible")
    if (
        payload["file_name"] != CALIBRATION_ARRAYS_FILE
        or payload["uri"] != CALIBRATION_ARRAYS_FILE
    ):
        raise ValueError("calibration array archive path is invalid")
    _sha256(payload["sha256"], field="arrays.sha256")
    size = _positive_integer(payload["size_bytes"], field="arrays.size_bytes")
    if size > MAX_CALIBRATION_ARRAY_ARCHIVE_BYTES:
        raise ValueError("calibration array archive exceeds the configured size limit")
    entries = _mapping(payload["entries"], field="arrays.entries")
    expected_names = (
        {ISOTONIC_THRESHOLDS_ARRAY, ISOTONIC_VALUES_ARRAY}
        if method == ISOTONIC_METHOD
        else set()
    )
    if set(entries) != expected_names:
        raise ValueError("calibration array manifest keys are invalid")
    expected_length: int | None = None
    for name, raw_spec in entries.items():
        spec = _mapping(raw_spec, field=f"arrays.entries.{name}")
        _expect_keys(
            spec,
            {"dtype", "raw_sha256", "shape", "size_bytes"},
            field=f"arrays.entries.{name}",
        )
        if spec["dtype"] != "<f8":
            raise ValueError(f"calibration array {name} dtype is invalid")
        shape = spec["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 1
            or isinstance(shape[0], bool)
            or not isinstance(shape[0], int)
            or shape[0] < 2
        ):
            raise ValueError(f"calibration array {name} shape is invalid")
        if expected_length is None:
            expected_length = shape[0]
        elif shape[0] != expected_length:
            raise ValueError("isotonic calibration array lengths differ")
        _sha256(spec["raw_sha256"], field=f"arrays.entries.{name}.raw_sha256")
        if (
            _positive_integer(
                spec["size_bytes"], field=f"arrays.entries.{name}.size_bytes"
            )
            != shape[0] * 8
        ):
            raise ValueError(f"calibration array {name} byte size is invalid")


def _expected_array_contract(
    method: str,
    specs: Mapping[str, object],
) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    for name, raw_spec in specs.items():
        spec = _mapping(raw_spec, field=f"arrays.entries.{name}")
        shape = spec["shape"]
        assert isinstance(shape, list)
        shapes[name] = tuple(int(value) for value in shape)
        dtypes[name] = str(spec["dtype"])
    expected = (
        {ISOTONIC_THRESHOLDS_ARRAY, ISOTONIC_VALUES_ARRAY}
        if method == ISOTONIC_METHOD
        else set()
    )
    if set(shapes) != expected:
        raise ValueError("calibration array contract is inconsistent")
    return shapes, dtypes


def _validate_loaded_arrays(method: str, arrays: Mapping[str, Any]) -> None:
    if method != ISOTONIC_METHOD:
        if arrays:
            raise ValueError(
                "non-isotonic calibrators cannot contain vector parameters"
            )
        return
    np = _load_numpy()
    thresholds = arrays[ISOTONIC_THRESHOLDS_ARRAY]
    values = arrays[ISOTONIC_VALUES_ARRAY]
    if not bool((np.diff(thresholds) > 0.0).all()):
        raise ValueError("isotonic thresholds must be strictly increasing")
    if not bool((np.diff(values) >= 0.0).all()):
        raise ValueError("isotonic values must be nondecreasing")
    if bool(((values < 0.0) | (values > 1.0)).any()):
        raise ValueError("isotonic values must be probabilities")


def _validate_scalar_parameters(method: str, values: Mapping[str, float]) -> None:
    expected = {
        SIGMOID_METHOD: {"intercept", "slope"},
        ISOTONIC_METHOD: set(),
        TEMPERATURE_METHOD: {"inverse_temperature", "temperature"},
    }[method]
    if set(values) != expected:
        raise ValueError("calibration scalar parameter keys are invalid")
    if method == TEMPERATURE_METHOD:
        inverse = values["inverse_temperature"]
        temperature = values["temperature"]
        if inverse <= 0.0 or temperature <= 0.0:
            raise ValueError("temperature parameters must be positive")
        if abs(inverse * temperature - 1.0) > 1e-12:
            raise ValueError("temperature parameters are inconsistent")


def _validate_optimizer(method: str, payload: Mapping[str, object]) -> None:
    if method == SIGMOID_METHOD:
        _expect_keys(
            payload,
            {"algorithm", "converged", "iteration_count", "maximum_iterations"},
            field="parameters.optimizer",
        )
        if payload["algorithm"] != "platt-smoothed-newton-line-search-v1":
            raise ValueError("sigmoid optimizer identity is incompatible")
        if not isinstance(payload["converged"], bool):
            raise ValueError("sigmoid optimizer convergence flag is invalid")
        iterations = _positive_integer(
            payload["iteration_count"], field="parameters.optimizer.iteration_count"
        )
        maximum = _positive_integer(
            payload["maximum_iterations"],
            field="parameters.optimizer.maximum_iterations",
        )
        if maximum != 100 or iterations > maximum:
            raise ValueError("sigmoid optimizer iteration metadata is invalid")
    elif method == ISOTONIC_METHOD:
        _expect_keys(
            payload,
            {"algorithm", "minimum_group_count", "minimum_sample_count"},
            field="parameters.optimizer",
        )
        if payload["algorithm"] != "weighted-pava-piecewise-linear-v1":
            raise ValueError("isotonic optimizer identity is incompatible")
        if (
            payload["minimum_group_count"] != MIN_ISOTONIC_GROUP_COUNT
            or payload["minimum_sample_count"] != MIN_ISOTONIC_SAMPLE_COUNT
        ):
            raise ValueError("isotonic evidence policy is incompatible")
    else:
        _expect_keys(
            payload,
            {"algorithm", "iteration_count", "log_inverse_temperature_bounds"},
            field="parameters.optimizer",
        )
        if (
            payload["algorithm"] != "bounded-golden-section-log-inverse-temperature-v1"
            or payload["iteration_count"] != DEFAULT_TEMPERATURE_OPTIMIZATION_ITERATIONS
            or payload["log_inverse_temperature_bounds"]
            != [TEMPERATURE_LOG_INVERSE_MIN, TEMPERATURE_LOG_INVERSE_MAX]
        ):
            raise ValueError("temperature optimizer identity is incompatible")


def _pending_decision_policy(fit: CalibrationFit) -> dict[str, object]:
    semantics = {
        "schema_version": "few-shot-decision-policy-pending-v1.0.0",
        "status": "not_fitted",
        "target_confirmation_enabled": False,
        "calibration_fingerprint": fit.calibration_fingerprint,
        "reason": "Task 9.4 threshold and abstention policy has not been fitted.",
    }
    return {
        **semantics,
        "decision_policy_fingerprint": canonical_semantic_fingerprint(semantics),
    }


def _validate_pending_decision_policy(
    payload: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    _expect_keys(
        payload,
        {
            "calibration_fingerprint",
            "decision_policy_fingerprint",
            "reason",
            "schema_version",
            "status",
            "target_confirmation_enabled",
        },
        field="decision_policy",
    )
    if (
        payload["schema_version"] != "few-shot-decision-policy-pending-v1.0.0"
        or payload["status"] != "not_fitted"
        or payload["target_confirmation_enabled"] is not False
        or payload["calibration_fingerprint"] != manifest["calibration_fingerprint"]
        or payload["reason"]
        != "Task 9.4 threshold and abstention policy has not been fitted."
    ):
        raise ValueError("pending decision policy is invalid")
    fingerprint = _sha256(
        payload["decision_policy_fingerprint"],
        field="decision_policy.decision_policy_fingerprint",
    )
    semantics = dict(payload)
    semantics.pop("decision_policy_fingerprint")
    if canonical_semantic_fingerprint(semantics) != fingerprint:
        raise ValueError("pending decision policy fingerprint is invalid")


def _validated_embedded_decision_policy(
    value: Mapping[str, object],
    *,
    calibration_fingerprint: str,
    classifier_fingerprint: str,
    split_fingerprint: str,
    target_task: str,
    route: str,
) -> dict[str, object]:
    payload = _mapping(value, field="decision_policy")
    _expect_keys(
        payload,
        {
            "abstention_rules",
            "achieved_metrics",
            "calibration_fingerprint",
            "calibration_group_count",
            "calibration_sample_count",
            "calibration_sample_fingerprint",
            "classifier_fingerprint",
            "competitor_margin_threshold",
            "decision_policy_fingerprint",
            "eligible_calibration_sample_count",
            "model_fingerprint",
            "negative_sample_count",
            "optimization_metric",
            "policy_version",
            "positive_sample_count",
            "requirements",
            "route",
            "schema_version",
            "split_fingerprint",
            "status",
            "status_reason",
            "target_confirmation_enabled",
            "target_precision_objective",
            "target_probability_threshold",
            "target_task",
            "threshold_grid_size",
        },
        field="decision_policy",
    )
    if payload["schema_version"] != "few-shot-decision-policy-v1.0.0":
        raise ValueError("decision policy schema version is incompatible")
    if payload["policy_version"] != "precision-constrained-selective-policy-v1.0.0":
        raise ValueError("decision policy version is incompatible")
    if payload["optimization_metric"] != "weighted_target_recall_at_precision":
        raise ValueError("decision policy optimization metric is incompatible")
    if (
        payload["calibration_fingerprint"] != calibration_fingerprint
        or payload["classifier_fingerprint"] != classifier_fingerprint
        or payload["split_fingerprint"] != split_fingerprint
        or payload["target_task"] != target_task
        or payload["route"] != route
    ):
        raise ValueError("decision policy identity does not match calibration")
    _sha256(payload["model_fingerprint"], field="decision_policy.model_fingerprint")
    _sha256(
        payload["calibration_sample_fingerprint"],
        field="decision_policy.calibration_sample_fingerprint",
    )
    for field in (
        "calibration_fingerprint",
        "classifier_fingerprint",
        "split_fingerprint",
    ):
        _sha256(payload[field], field=f"decision_policy.{field}")
    status = _required_choice(
        payload["status"],
        field="decision_policy.status",
        allowed=frozenset({"fitted", "infeasible"}),
    )
    enabled = payload["target_confirmation_enabled"]
    if not isinstance(enabled, bool):
        raise ValueError("decision policy confirmation flag must be boolean")
    objective = _unit_interval(
        payload["target_precision_objective"],
        field="decision_policy.target_precision_objective",
    )
    if objective <= 0.0:
        raise ValueError("decision policy precision objective must be positive")
    sample_count = _positive_integer(
        payload["calibration_sample_count"],
        field="decision_policy.calibration_sample_count",
    )
    eligible_count = _nonnegative_integer(
        payload["eligible_calibration_sample_count"],
        field="decision_policy.eligible_calibration_sample_count",
    )
    group_count = _integer_at_least(
        payload["calibration_group_count"],
        minimum=2,
        field="decision_policy.calibration_group_count",
    )
    positive_count = _positive_integer(
        payload["positive_sample_count"],
        field="decision_policy.positive_sample_count",
    )
    negative_count = _positive_integer(
        payload["negative_sample_count"],
        field="decision_policy.negative_sample_count",
    )
    grid_size = _nonnegative_integer(
        payload["threshold_grid_size"],
        field="decision_policy.threshold_grid_size",
    )
    if (
        eligible_count > sample_count
        or group_count > sample_count
        or positive_count + negative_count != sample_count
        or grid_size > eligible_count * eligible_count
        or (eligible_count == 0) != (grid_size == 0)
    ):
        raise ValueError("decision policy calibration counts are inconsistent")
    requirements = _mapping(
        payload["requirements"], field="decision_policy.requirements"
    )
    expected_requirements = {
        "route_compatible": True,
        "reference_coverage_sufficient": True,
        "domain_negative_absent": True,
        "out_of_distribution_absent": True,
        "visual_detail_sufficient": True,
        "no_geo_global_fallback_absent": True,
    }
    if dict(requirements) != expected_requirements:
        raise ValueError("decision policy requirements are incompatible")
    expected_rules = [
        "decision_policy_infeasible",
        "incompatible_route",
        "domain_negative_without_supported_outcome",
        "out_of_distribution",
        "insufficient_visual_detail",
        "insufficient_reference_coverage",
        "no_geo_global_fallback",
        "calibrated_non_target_dominates",
        "missing_calibrated_target_probability",
        "target_probability_below_threshold",
        "missing_competitor_margin",
        "competitor_margin_below_threshold",
    ]
    if payload["abstention_rules"] != expected_rules:
        raise ValueError("decision policy abstention rules are incompatible")
    metrics = _mapping(
        payload["achieved_metrics"], field="decision_policy.achieved_metrics"
    )
    metric_fields = {
        "weighted_precision",
        "weighted_recall",
        "weighted_coverage",
        "unweighted_precision",
        "unweighted_recall",
        "unweighted_coverage",
    }
    _expect_keys(metrics, metric_fields, field="decision_policy.achieved_metrics")
    probability_threshold = payload["target_probability_threshold"]
    margin_threshold = payload["competitor_margin_threshold"]
    if status == "fitted":
        if not enabled or payload["status_reason"] != "precision_objective_met":
            raise ValueError("fitted decision policy status metadata is invalid")
        _unit_interval(
            probability_threshold,
            field="decision_policy.target_probability_threshold",
        )
        margin = _finite_number(
            margin_threshold,
            field="decision_policy.competitor_margin_threshold",
        )
        if not -2.0 <= margin <= 2.0:
            raise ValueError("decision policy competitor margin threshold is invalid")
        achieved = {
            field: _unit_interval(
                metrics[field],
                field=f"decision_policy.achieved_metrics.{field}",
            )
            for field in metric_fields
        }
        if achieved["weighted_precision"] + 1e-15 < objective or grid_size == 0:
            raise ValueError("decision policy does not meet its precision objective")
    else:
        if (
            enabled
            or payload["status_reason"]
            != "no_threshold_pair_meets_target_precision_objective"
            or probability_threshold is not None
            or margin_threshold is not None
            or any(metrics[field] is not None for field in metric_fields)
        ):
            raise ValueError("infeasible decision policy metadata is invalid")
    fingerprint = _sha256(
        payload["decision_policy_fingerprint"],
        field="decision_policy.decision_policy_fingerprint",
    )
    semantics = dict(payload)
    semantics.pop("decision_policy_fingerprint")
    if canonical_semantic_fingerprint(semantics) != fingerprint:
        raise ValueError("decision policy fingerprint is invalid")
    return dict(payload)


def _validate_report_manifest(payload: Mapping[str, object]) -> None:
    _expect_keys(
        payload,
        {"file_name", "row_count", "schema_version", "sha256", "size_bytes", "uri"},
        field="report",
    )
    if payload["schema_version"] != CALIBRATION_REPORT_SCHEMA_VERSION:
        raise ValueError("calibration report schema version is incompatible")
    if (
        payload["file_name"] != CALIBRATION_REPORT_FILE
        or payload["uri"] != CALIBRATION_REPORT_FILE
    ):
        raise ValueError("calibration report path is invalid")
    _sha256(payload["sha256"], field="report.sha256")
    size = _positive_integer(payload["size_bytes"], field="report.size_bytes")
    if size > MAX_CALIBRATION_REPORT_BYTES:
        raise ValueError("calibration report exceeds the configured size limit")
    _positive_integer(payload["row_count"], field="report.row_count")


def _validate_loaded_report(
    report: pl.DataFrame,
    *,
    calibration_fingerprint: str,
    classifier_fingerprint: str,
    split_fingerprint: str,
    prediction_fingerprint: str,
    target_task: str,
    route: str,
    row_count: int,
    class_labels: tuple[str, ...],
    method: str,
    reliability_bin_count: int,
) -> None:
    expected_schema = calibration_report_schema()
    if report.schema != expected_schema:
        raise ValueError("calibration report schema is incompatible")
    if (
        report.height != row_count
        or row_count != len(class_labels) * reliability_bin_count
    ):
        raise ValueError("calibration report row count is inconsistent")
    rows = report.sort("class_index", "bin_index").to_dicts()
    for index, row in enumerate(rows):
        class_index = index // reliability_bin_count
        bin_index = index % reliability_bin_count
        if (
            row["schema_version"] != CALIBRATION_REPORT_SCHEMA_VERSION
            or row["calibration_version"] != CALIBRATION_VERSION
            or row["calibration_fingerprint"] != calibration_fingerprint
            or row["classifier_fingerprint"] != classifier_fingerprint
            or row["independent_prediction_artifact_fingerprint"]
            != prediction_fingerprint
            or row["split_fingerprint"] != split_fingerprint
            or row["target_task"] != target_task
            or row["route"] != route
            or row["dataset_split"] != CALIBRATION_PARTITION
            or row["method"] != method
            or row["probability_kind"] != CALIBRATED_PROBABILITY_KIND
            or row["class_index"] != class_index
            or row["class_label"] != class_labels[class_index]
            or row["bin_index"] != bin_index
            or row["bin_lower_bound"] != bin_index / reliability_bin_count
            or row["bin_upper_bound"] != (bin_index + 1) / reliability_bin_count
        ):
            raise ValueError("calibration report identity or bin ordering is invalid")
        count = _nonnegative_integer(
            row["unweighted_count"], field="report.unweighted_count"
        )
        weighted_count = _nonnegative_number(
            row["weighted_count"], field="report.weighted_count"
        )
        mean_prediction = row["mean_prediction"]
        observed = row["observed_frequency"]
        if count == 0:
            if (
                weighted_count != 0.0
                or mean_prediction is not None
                or observed is not None
            ):
                raise ValueError("empty calibration report bins are inconsistent")
        else:
            if weighted_count <= 0.0:
                raise ValueError(
                    "populated calibration report bins need positive weight"
                )
            _unit_interval(mean_prediction, field="report.mean_prediction")
            _unit_interval(observed, field="report.observed_frequency")
        for field in (
            "brier_contribution",
            "log_loss_contribution",
            "ece_contribution",
        ):
            _nonnegative_number(row[field], field=f"report.{field}")


def _report_parquet_bytes(report: pl.DataFrame) -> bytes:
    if report.schema != calibration_report_schema():
        raise ValueError("calibration report schema is incompatible")
    output = io.BytesIO()
    report.write_parquet(output, compression="zstd", statistics=True)
    return output.getvalue()


def _read_manifest(path: Path) -> dict[str, object]:
    raw = _bounded_file_bytes(
        path,
        maximum=MAX_CALIBRATION_MANIFEST_BYTES,
        label="calibration manifest",
    )
    try:
        text = raw.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("calibration manifest is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("calibration manifest must be a JSON object")
    if raw != _canonical_json_bytes(payload):
        raise ValueError("calibration manifest is not canonical JSON")
    return payload


def _bounded_file_bytes(path: Path, *, maximum: int, label: str) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if size <= 0 or size > maximum:
        raise ValueError(f"{label} size is invalid")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} is unreadable") from exc


def _library_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    def package_version(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "not_installed"

    return {
        "biominer": package_version("biominer"),
        "numpy": package_version("numpy"),
        "polars": package_version("polars"),
        "scikit_learn": package_version("scikit-learn"),
    }


def _fold_semantics(audit: CalibrationFoldAudit) -> dict[str, object]:
    return {
        "fold_index": _nonnegative_integer(audit.fold_index, field="fold_index"),
        "estimator_fit_group_ids": list(
            _canonical_group_ids(
                audit.estimator_fit_group_ids,
                field="estimator_fit_group_ids",
            )
        ),
        "validation_group_ids": list(
            _canonical_group_ids(
                audit.validation_group_ids,
                field="validation_group_ids",
            )
        ),
    }


def _canonical_group_ids(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    result = tuple(sorted(_required_text(value, field=field) for value in values))
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must not contain duplicate groups")
    return result


def _counts_payload(values: object) -> list[dict[str, object]]:
    assert isinstance(values, tuple)
    return [{"class_label": label, "count": count} for label, count in values]


def _validate_counts_payload(
    value: object,
    *,
    labels: tuple[str, ...],
    total: int | None,
    minimum: int,
    field: str,
) -> dict[str, int]:
    if not isinstance(value, list) or len(value) != len(labels):
        raise ValueError(f"{field} is invalid")
    result: dict[str, int] = {}
    for expected_label, raw in zip(labels, value, strict=True):
        row = _mapping(raw, field=field)
        _expect_keys(row, {"class_label", "count"}, field=field)
        if row["class_label"] != expected_label:
            raise ValueError(f"{field} class order is invalid")
        result[expected_label] = _integer_at_least(
            row["count"], minimum=minimum, field=f"{field}.count"
        )
    if total is not None and sum(result.values()) != total:
        raise ValueError(f"{field} total is inconsistent")
    return result


def _stable_sigmoid_array(values: Any) -> Any:
    np = _load_numpy()
    result = np.empty_like(values, dtype=np.float64)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponent = np.exp(values[~nonnegative])
    result[~nonnegative] = exponent / (1.0 + exponent)
    return result


def _softmax(values: Any) -> Any:
    np = _load_numpy()
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _unique_json_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _write_exclusive_atomic(path: Path, value: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(path) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _expect_keys(
    value: Mapping[str, object],
    expected: set[str] | frozenset[str],
    *,
    field: str,
) -> None:
    if set(value) != set(expected):
        raise ValueError(f"{field} has an incompatible key set")


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


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    return tuple(_required_text(item, field=field) for item in value)


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full lowercase sha256 fingerprint")
    return text


def _git_sha(value: object) -> str:
    text = _required_text(value, field="git_sha")
    if _GIT_SHA_PATTERN.fullmatch(text) is None:
        raise ValueError("git_sha must be a full lowercase commit SHA")
    return text


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_integer(value: object, *, field: str) -> int:
    result = _nonnegative_integer(value, field=field)
    if result == 0:
        raise ValueError(f"{field} must be positive")
    return result


def _integer_at_least(value: object, *, minimum: int, field: str) -> int:
    result = _nonnegative_integer(value, field=field)
    if result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        item_method = getattr(value, "item", None)
        if not callable(item_method):
            raise ValueError(f"{field} must be numeric")
        value = item_method()
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _nonnegative_number(value: object, *, field: str) -> float:
    result = _finite_number(value, field=field)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _unit_interval(value: object, *, field: str) -> float:
    result = _finite_number(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be between zero and one")
    return result


def _creation_timestamp(value: datetime | None) -> str:
    timestamp = datetime.now(timezone.utc) if value is None else value
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
        raise ValueError("created_at must be a timezone-aware datetime")
    normalized = timestamp.astimezone(timezone.utc)
    return normalized.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _validate_creation_timestamp(value: object) -> str:
    text = _required_text(value, field="created_at")
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise ValueError(
            "created_at must use canonical UTC microsecond format"
        ) from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ") != text:
        raise ValueError("created_at must use canonical UTC microsecond format")
    return text


def _match_expected_fingerprint(
    actual: str,
    expected: str | None,
    *,
    field: str,
) -> None:
    if expected is None:
        return
    if actual != _sha256(expected, field=f"expected_{field}"):
        raise ValueError(f"{field} does not match the expected fingerprint")


def _load_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("NumPy is required for probability calibration") from exc
    return np


__all__ = [
    "AUTO_METHOD",
    "CALIBRATED_PROBABILITY_KIND",
    "CALIBRATION_ARRAYS_FILE",
    "CALIBRATION_MANIFEST_FILE",
    "CALIBRATION_MANIFEST_SCHEMA_VERSION",
    "CALIBRATION_OOF_POLICY",
    "CALIBRATION_REPORT_FILE",
    "CALIBRATION_REPORT_SCHEMA_VERSION",
    "CALIBRATION_VERSION",
    "ISOTONIC_METHOD",
    "SIGMOID_METHOD",
    "TEMPERATURE_METHOD",
    "CalibrationArtifactPaths",
    "CalibrationConfig",
    "CalibrationFit",
    "CalibrationFoldAudit",
    "CalibrationPrediction",
    "FrozenProbabilityCalibrator",
    "LoadedCalibration",
    "calibration_report_schema",
    "fit_probability_calibrator",
    "load_probability_calibrator",
    "write_probability_calibrator",
]
