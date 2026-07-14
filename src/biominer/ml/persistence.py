"""Versioned, non-executable persistence for frozen linear classifiers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from math import isfinite
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Any
from uuid import uuid4
import zipfile

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.ml.classifiers import (
    CLASSIFIER_TRAINING_VERSION,
    CV_SPLIT_VERSION,
    EMBEDDING_ONLY_FEATURE_SET,
    EMBEDDING_PLUS_STRUCTURED_FEATURE_SET,
    ESTIMATOR_LINEAR_SVC,
    ESTIMATOR_LOGISTIC_REGRESSION,
    FEATURE_LAYOUT_VERSION,
    LINEAR_SVC_EMBEDDING_MODEL,
    LINEAR_SVC_STRUCTURED_MODEL,
    LOGISTIC_REGRESSION_MODEL,
    PRIMARY_CV_METRIC,
    QUALITY_FLAG_HASH_BUCKET_COUNT,
    QUALITY_FLAG_HASH_VERSION,
    SECONDARY_CV_METRIC,
    ClassifierCandidateResult,
    ClassifierFeatureLayout,
    ClassifierTrainingRun,
    classifier_feature_layout,
)
from biominer.ml.training_features import TARGET_TASKS
from biominer.references.readiness import REFERENCE_ROUTES


CLASSIFIER_MANIFEST_SCHEMA_VERSION = "few-shot-classifier-manifest-v1.0.0"
CLASSIFIER_VERSION = "frozen-bioclip-linear-classifier-v1.0.0"
CLASSIFIER_MANIFEST_FILE = "classifier_manifest.json"
CLASSIFIER_ARRAYS_FILE = "classifier_arrays.npz"
CLASSIFIER_VALIDATION_POLICY = "group-aware-stratified-k-fold-v1"
CLASSIFIER_QA_STATUS = "passed"

COEFFICIENTS_ARRAY = "coefficients"
INTERCEPTS_ARRAY = "intercepts"
CLASS_INDICES_ARRAY = "class_indices"
CONTINUOUS_IMPUTER_STATISTICS_ARRAY = "continuous_imputer_statistics"
CONTINUOUS_SCALER_MEAN_ARRAY = "continuous_scaler_mean"
CONTINUOUS_SCALER_SCALE_ARRAY = "continuous_scaler_scale"
CONTINUOUS_SCALER_VARIANCE_ARRAY = "continuous_scaler_variance"

MAX_CLASSIFIER_MANIFEST_BYTES = 1_048_576
MAX_CLASSIFIER_ARRAY_ARCHIVE_BYTES = 268_435_456
MAX_CLASSIFIER_ARRAY_UNCOMPRESSED_BYTES = 268_435_456
MAX_NUMPY_HEADER_BYTES = 4_096

_BASE_ARRAY_KEYS = frozenset(
    {COEFFICIENTS_ARRAY, INTERCEPTS_ARRAY, CLASS_INDICES_ARRAY}
)
_STRUCTURED_ARRAY_KEYS = frozenset(
    {
        CONTINUOUS_IMPUTER_STATISTICS_ARRAY,
        CONTINUOUS_SCALER_MEAN_ARRAY,
        CONTINUOUS_SCALER_SCALE_ARRAY,
        CONTINUOUS_SCALER_VARIANCE_ARRAY,
    }
)
_LINEAR_MODEL_IDENTITIES = {
    LOGISTIC_REGRESSION_MODEL: (
        ESTIMATOR_LOGISTIC_REGRESSION,
        EMBEDDING_ONLY_FEATURE_SET,
    ),
    LINEAR_SVC_EMBEDDING_MODEL: (
        ESTIMATOR_LINEAR_SVC,
        EMBEDDING_ONLY_FEATURE_SET,
    ),
    LINEAR_SVC_STRUCTURED_MODEL: (
        ESTIMATOR_LINEAR_SVC,
        EMBEDDING_PLUS_STRUCTURED_FEATURE_SET,
    ),
}
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")

_TOP_LEVEL_KEYS = frozenset(
    {
        "arrays",
        "classifier_fingerprint",
        "classifier_version",
        "created_at",
        "features",
        "git_sha",
        "identity",
        "preprocessing",
        "qa_status",
        "schema_version",
        "training",
        "validation",
    }
)


@dataclass(frozen=True, slots=True)
class ClassifierArtifactPaths:
    """Committed paths and identity for one immutable classifier artifact."""

    directory: Path
    manifest_path: Path
    arrays_path: Path
    classifier_fingerprint: str


@dataclass(frozen=True, slots=True)
class FrozenLinearClassifier:
    """Validated linear decision function reconstructed without scikit-learn."""

    classifier_version: str
    classifier_fingerprint: str
    model_name: str
    estimator_family: str
    target_task: str
    target_accepted_taxon_key: str
    route: str
    class_labels: tuple[str, ...]
    feature_layout: ClassifierFeatureLayout
    feature_schema_fingerprint: str
    model_fingerprint: str
    preprocessing_fingerprint: str
    reference_bank_version: str
    reference_bank_fingerprint: str
    training_data_fingerprint: str
    probability_calibrated: bool
    coefficients: Any
    intercepts: Any
    class_indices: Any
    continuous_imputer_statistics: Any
    continuous_scaler_mean: Any
    continuous_scaler_scale: Any
    continuous_scaler_variance: Any

    def transform_features(self, raw_features: object) -> Any:
        """Apply the persisted median-imputation and scaling policy."""

        np = _load_numpy()
        try:
            matrix = np.asarray(raw_features, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("classifier features must be a numeric matrix") from exc
        if matrix.ndim != 2:
            raise ValueError("classifier features must be two-dimensional")
        if matrix.shape[1] != len(self.feature_layout.raw_feature_names):
            raise ValueError(
                "classifier feature width does not match the persisted layout"
            )
        transformed = np.array(matrix, dtype=np.float64, order="C", copy=True)
        continuous = self.feature_layout.continuous_column_indices
        noncontinuous = (
            *self.feature_layout.embedding_column_indices,
            *self.feature_layout.indicator_column_indices,
        )
        if noncontinuous and not bool(
            np.isfinite(transformed[:, list(noncontinuous)]).all()
        ):
            raise ValueError("embedding and indicator features must be finite")
        indicators = self.feature_layout.indicator_column_indices
        if indicators:
            indicator_values = transformed[:, list(indicators)]
            if not bool(((indicator_values == 0.0) | (indicator_values == 1.0)).all()):
                raise ValueError("indicator features must contain only zero or one")
        if continuous:
            indices = list(continuous)
            values = transformed[:, indices]
            if bool(np.isinf(values).any()):
                raise ValueError("continuous classifier features cannot be infinite")
            values = np.where(
                np.isnan(values),
                self.continuous_imputer_statistics,
                values,
            )
            transformed[:, indices] = (
                values - self.continuous_scaler_mean
            ) / self.continuous_scaler_scale
        if not bool(np.isfinite(transformed).all()):
            raise ValueError("transformed classifier features must be finite")
        return transformed

    def decision_function(self, raw_features: object) -> Any:
        """Return uncalibrated linear margins/logits in fitted class order."""

        transformed = self.transform_features(raw_features)
        scores = transformed @ self.coefficients.T + self.intercepts.reshape(1, -1)
        return scores.ravel() if self.coefficients.shape[0] == 1 else scores

    def predict(self, raw_features: object) -> tuple[str, ...]:
        """Return class labels using scikit-learn's linear class ordering."""

        scores = self.decision_function(raw_features)
        if scores.ndim == 1:
            indices = (scores > 0.0).astype("intp")
        else:
            indices = scores.argmax(axis=1)
        return tuple(self.class_labels[int(index)] for index in indices)


def write_frozen_classifier(
    run: ClassifierTrainingRun,
    directory: str | Path,
    *,
    preprocessing_fingerprint: str,
    reference_bank_version: str,
    reference_bank_fingerprint: str,
    git_sha: str,
    model_name: str | None = None,
    created_at: datetime | None = None,
) -> ClassifierArtifactPaths:
    """Persist one fitted linear candidate as canonical JSON and numeric NPZ."""

    if not isinstance(run, ClassifierTrainingRun):
        raise TypeError("run must be a ClassifierTrainingRun")
    preprocessing = _sha256_fingerprint(
        preprocessing_fingerprint,
        field="preprocessing_fingerprint",
    )
    bank_version = _required_text(
        reference_bank_version,
        field="reference_bank_version",
    )
    bank_fingerprint = _sha256_fingerprint(
        reference_bank_fingerprint,
        field="reference_bank_fingerprint",
    )
    commit_sha = _git_sha(git_sha)
    timestamp = _creation_timestamp(created_at)
    candidate = _select_candidate(run, model_name)
    expected_family, expected_feature_set = _linear_model_identity(candidate.model_name)
    if (
        candidate.estimator_family != expected_family
        or candidate.feature_set != expected_feature_set
    ):
        raise ValueError("only supported fitted linear classifiers can be persisted")

    arrays = _extract_candidate_arrays(run, candidate)
    archive_bytes = _deterministic_npz(arrays)
    if len(archive_bytes) > MAX_CLASSIFIER_ARRAY_ARCHIVE_BYTES:
        raise ValueError("classifier array archive exceeds the configured size limit")
    arrays_sha256 = _bytes_sha256(archive_bytes)
    array_entries = {
        name: _array_manifest_entry(array) for name, array in sorted(arrays.items())
    }
    manifest = _build_manifest(
        run=run,
        candidate=candidate,
        preprocessing_fingerprint=preprocessing,
        reference_bank_version=bank_version,
        reference_bank_fingerprint=bank_fingerprint,
        git_sha=commit_sha,
        created_at=timestamp,
        arrays_sha256=arrays_sha256,
        arrays_size_bytes=len(archive_bytes),
        array_entries=array_entries,
    )
    classifier_fingerprint = _classifier_fingerprint(manifest)
    manifest["classifier_fingerprint"] = classifier_fingerprint
    manifest_bytes = _canonical_json_bytes(manifest)
    if len(manifest_bytes) > MAX_CLASSIFIER_MANIFEST_BYTES:
        raise ValueError("classifier manifest exceeds the configured size limit")

    output = Path(directory)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir(mode=0o700, exist_ok=False)
        arrays_path = output / CLASSIFIER_ARRAYS_FILE
        manifest_path = output / CLASSIFIER_MANIFEST_FILE
        _write_exclusive_atomic(arrays_path, archive_bytes)
        _fsync_directory(output)
        _write_exclusive_atomic(manifest_path, manifest_bytes)
        _fsync_directory(output)
        _fsync_directory(output.parent)
    except BaseException:
        if output.is_dir() and not output.is_symlink():
            shutil.rmtree(output)
        raise
    return ClassifierArtifactPaths(
        directory=output,
        manifest_path=manifest_path,
        arrays_path=arrays_path,
        classifier_fingerprint=classifier_fingerprint,
    )


def load_frozen_classifier(
    directory: str | Path,
    *,
    expected_classifier_fingerprint: str | None = None,
    expected_model_fingerprint: str | None = None,
    expected_preprocessing_fingerprint: str | None = None,
    expected_reference_bank_fingerprint: str | None = None,
    expected_training_data_fingerprint: str | None = None,
) -> FrozenLinearClassifier:
    """Load and validate a transparent linear artifact without executable state."""

    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("classifier artifact path must be a real directory")
    expected_files = {CLASSIFIER_MANIFEST_FILE, CLASSIFIER_ARRAYS_FILE}
    try:
        entries = tuple(root.iterdir())
    except OSError as exc:
        raise ValueError("classifier artifact directory is unreadable") from exc
    if {item.name for item in entries} != expected_files:
        raise ValueError("classifier artifact directory has unexpected files")
    if any(item.is_symlink() or not item.is_file() for item in entries):
        raise ValueError("classifier artifact files must be regular files")

    manifest_path = root / CLASSIFIER_MANIFEST_FILE
    arrays_path = root / CLASSIFIER_ARRAYS_FILE
    manifest = _read_manifest(manifest_path)
    metadata = _validate_manifest(manifest)
    classifier_fingerprint = str(manifest["classifier_fingerprint"])
    if _classifier_fingerprint(manifest) != classifier_fingerprint:
        raise ValueError("classifier fingerprint does not match the manifest")
    _match_expected_fingerprint(
        classifier_fingerprint,
        expected_classifier_fingerprint,
        field="classifier_fingerprint",
    )

    arrays_metadata = metadata["arrays"]
    assert isinstance(arrays_metadata, Mapping)
    expected_size = int(arrays_metadata["size_bytes"])
    try:
        actual_size = arrays_path.stat().st_size
    except OSError as exc:
        raise ValueError("classifier array archive is unreadable") from exc
    if actual_size != expected_size:
        raise ValueError("classifier array archive size does not match the manifest")
    if actual_size > MAX_CLASSIFIER_ARRAY_ARCHIVE_BYTES:
        raise ValueError("classifier array archive exceeds the configured size limit")
    try:
        archive_bytes = arrays_path.read_bytes()
    except OSError as exc:
        raise ValueError("classifier array archive is unreadable") from exc
    if _bytes_sha256(archive_bytes) != arrays_metadata["sha256"]:
        raise ValueError(
            "classifier array archive checksum does not match the manifest"
        )

    feature_layout = metadata["feature_layout"]
    assert isinstance(feature_layout, ClassifierFeatureLayout)
    array_specs = arrays_metadata["entries"]
    assert isinstance(array_specs, Mapping)
    arrays = _load_array_archive(
        archive_bytes,
        specs=array_specs,
        feature_layout=feature_layout,
        class_count=len(metadata["class_labels"]),
    )
    training = metadata["training"]
    assert isinstance(training, Mapping)
    _match_expected_fingerprint(
        str(training["foundation_model_fingerprint"]),
        expected_model_fingerprint,
        field="model_fingerprint",
    )
    _match_expected_fingerprint(
        str(training["preprocessing_fingerprint"]),
        expected_preprocessing_fingerprint,
        field="preprocessing_fingerprint",
    )
    _match_expected_fingerprint(
        str(training["reference_bank_fingerprint"]),
        expected_reference_bank_fingerprint,
        field="reference_bank_fingerprint",
    )
    _match_expected_fingerprint(
        str(training["training_data_fingerprint"]),
        expected_training_data_fingerprint,
        field="training_data_fingerprint",
    )
    for array in arrays.values():
        array.setflags(write=False)
    identity = metadata["identity"]
    assert isinstance(identity, Mapping)
    return FrozenLinearClassifier(
        classifier_version=CLASSIFIER_VERSION,
        classifier_fingerprint=classifier_fingerprint,
        model_name=str(identity["model_name"]),
        estimator_family=str(identity["estimator_family"]),
        target_task=str(identity["target_task"]),
        target_accepted_taxon_key=str(identity["target_accepted_taxon_key"]),
        route=str(identity["route"]),
        class_labels=tuple(metadata["class_labels"]),
        feature_layout=feature_layout,
        feature_schema_fingerprint=str(training["feature_schema_fingerprint"]),
        model_fingerprint=str(training["foundation_model_fingerprint"]),
        preprocessing_fingerprint=str(training["preprocessing_fingerprint"]),
        reference_bank_version=str(training["reference_bank_version"]),
        reference_bank_fingerprint=str(training["reference_bank_fingerprint"]),
        training_data_fingerprint=str(training["training_data_fingerprint"]),
        probability_calibrated=False,
        coefficients=arrays[COEFFICIENTS_ARRAY],
        intercepts=arrays[INTERCEPTS_ARRAY],
        class_indices=arrays[CLASS_INDICES_ARRAY],
        continuous_imputer_statistics=arrays.get(
            CONTINUOUS_IMPUTER_STATISTICS_ARRAY,
            _empty_float_array(),
        ),
        continuous_scaler_mean=arrays.get(
            CONTINUOUS_SCALER_MEAN_ARRAY,
            _empty_float_array(),
        ),
        continuous_scaler_scale=arrays.get(
            CONTINUOUS_SCALER_SCALE_ARRAY,
            _empty_float_array(),
        ),
        continuous_scaler_variance=arrays.get(
            CONTINUOUS_SCALER_VARIANCE_ARRAY,
            _empty_float_array(),
        ),
    )


def _build_manifest(
    *,
    run: ClassifierTrainingRun,
    candidate: ClassifierCandidateResult,
    preprocessing_fingerprint: str,
    reference_bank_version: str,
    reference_bank_fingerprint: str,
    git_sha: str,
    created_at: str,
    arrays_sha256: str,
    arrays_size_bytes: int,
    array_entries: Mapping[str, object],
) -> dict[str, object]:
    layout = candidate.feature_layout
    estimator_configuration = _pairs_to_json_mapping(candidate.estimator_configuration)
    deterministic_seed = estimator_configuration.get("random_seed")
    if isinstance(deterministic_seed, bool) or not isinstance(deterministic_seed, int):
        raise ValueError("candidate estimator configuration lacks a random seed")
    class_weight_policy = estimator_configuration.get("class_weight")
    model_selection_metrics = candidate.model_selection_metrics
    structured = layout.feature_set == EMBEDDING_PLUS_STRUCTURED_FEATURE_SET
    manifest: dict[str, object] = {
        "schema_version": CLASSIFIER_MANIFEST_SCHEMA_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "created_at": created_at,
        "git_sha": git_sha,
        "qa_status": CLASSIFIER_QA_STATUS,
        "identity": {
            "model_name": candidate.model_name,
            "estimator_family": candidate.estimator_family,
            "target_task": run.target_task,
            "target_accepted_taxon_key": run.target_accepted_taxon_key,
            "route": run.route,
            "class_labels": list(run.class_labels),
            "probability_calibrated": False,
        },
        "features": {
            "feature_layout_version": FEATURE_LAYOUT_VERSION,
            "feature_set": layout.feature_set,
            "embedding_dimension": layout.embedding_dimension,
            "source_feature_names": list(layout.source_feature_names),
            "raw_feature_names": list(layout.raw_feature_names),
            "raw_feature_dtypes": ["<f8"] * len(layout.raw_feature_names),
            "transformed_feature_names": list(layout.transformed_feature_names),
            "transformed_feature_dtypes": ["<f8"]
            * len(layout.transformed_feature_names),
            "embedding_column_indices": list(layout.embedding_column_indices),
            "continuous_column_indices": list(layout.continuous_column_indices),
            "indicator_column_indices": list(layout.indicator_column_indices),
            "feature_layout_fingerprint": layout.fingerprint,
            "quality_flag_hash_version": (
                QUALITY_FLAG_HASH_VERSION if structured else None
            ),
            "quality_flag_hash_bucket_count": (
                QUALITY_FLAG_HASH_BUCKET_COUNT if structured else None
            ),
        },
        "preprocessing": {
            "embedding_policy": "passthrough",
            "continuous_imputation_policy": "median" if structured else "none",
            "continuous_scaling_policy": "standard" if structured else "none",
            "indicator_policy": "passthrough" if structured else "none",
            "continuous_imputer_statistics_array": (
                CONTINUOUS_IMPUTER_STATISTICS_ARRAY if structured else None
            ),
            "continuous_scaler_mean_array": (
                CONTINUOUS_SCALER_MEAN_ARRAY if structured else None
            ),
            "continuous_scaler_scale_array": (
                CONTINUOUS_SCALER_SCALE_ARRAY if structured else None
            ),
            "continuous_scaler_variance_array": (
                CONTINUOUS_SCALER_VARIANCE_ARRAY if structured else None
            ),
        },
        "training": {
            "training_version": run.training_version,
            "configuration_fingerprint": run.configuration_fingerprint,
            "training_run_fingerprint": run.training_run_fingerprint,
            "candidate_fingerprint": candidate.candidate_fingerprint,
            "feature_schema_fingerprint": run.feature_schema_fingerprint,
            "training_data_fingerprint": run.training_data_fingerprint,
            "fit_partition_fingerprint": run.fit_partition_fingerprint,
            "model_selection_partition_fingerprint": (
                run.model_selection_partition_fingerprint
            ),
            "cv_split_fingerprint": run.cv_split_fingerprint,
            "candidate_set_fingerprint": run.candidate_set_fingerprint,
            "foundation_model_fingerprint": run.model_fingerprint,
            "preprocessing_fingerprint": preprocessing_fingerprint,
            "reference_bank_version": reference_bank_version,
            "reference_bank_fingerprint": reference_bank_fingerprint,
            "support_manifest_fingerprint": run.support_manifest_fingerprint,
            "reference_embedding_fingerprint": run.reference_embedding_fingerprint,
            "reference_prototype_fingerprint": run.reference_prototype_fingerprint,
            "foundation_model_trainable": False,
            "numpy_version": run.numpy_version,
            "scikit_learn_version": run.scikit_learn_version,
            "estimator_configuration": estimator_configuration,
            "deterministic_seed": deterministic_seed,
            "class_weight_policy": class_weight_policy,
            "search_grid": {
                name: [_json_value(item) for item in values]
                for name, values in candidate.parameter_grid
            },
            "selected_parameters": _pairs_to_json_mapping(
                candidate.selected_parameters
            ),
        },
        "validation": {
            "qa_status": CLASSIFIER_QA_STATUS,
            "validation_policy": CLASSIFIER_VALIDATION_POLICY,
            "cv_split_version": CV_SPLIT_VERSION,
            "splitter": "StratifiedGroupKFold",
            "group_field": "leakage_group_id",
            "primary_metric": PRIMARY_CV_METRIC,
            "secondary_metric": SECONDARY_CV_METRIC,
            "fold_count": len(run.folds),
            "fit_sample_count": run.fit_sample_count,
            "fit_group_count": run.fit_group_count,
            "fit_class_sample_counts": _counts_payload(run.fit_class_sample_counts),
            "fit_class_group_counts": _counts_payload(run.fit_class_group_counts),
            "model_selection_sample_count": run.model_selection_sample_count,
            "calibration_sample_count": run.calibration_sample_count,
            "final_test_sample_count": run.final_test_sample_count,
            "best_cv_balanced_accuracy": candidate.best_cv_balanced_accuracy,
            "best_cv_macro_f1": candidate.best_cv_macro_f1,
            "model_selection_metrics": (
                {
                    "sample_count": model_selection_metrics.sample_count,
                    "class_sample_counts": _counts_payload(
                        model_selection_metrics.class_sample_counts
                    ),
                    "balanced_accuracy": model_selection_metrics.balanced_accuracy,
                    "macro_f1": model_selection_metrics.macro_f1,
                }
                if model_selection_metrics is not None
                else None
            ),
            "probability_calibrated": False,
        },
        "arrays": {
            "file_name": CLASSIFIER_ARRAYS_FILE,
            "uri": CLASSIFIER_ARRAYS_FILE,
            "sha256": arrays_sha256,
            "size_bytes": arrays_size_bytes,
            "entries": dict(array_entries),
        },
    }
    return manifest


def _extract_candidate_arrays(
    run: ClassifierTrainingRun,
    candidate: ClassifierCandidateResult,
) -> dict[str, Any]:
    np = _load_numpy()
    pipeline = candidate.pipeline
    named_steps = getattr(pipeline, "named_steps", None)
    if not isinstance(named_steps, Mapping) or "classifier" not in named_steps:
        raise ValueError("candidate does not contain a fitted classifier pipeline")
    classifier = named_steps["classifier"]
    try:
        fitted_labels = tuple(str(value) for value in classifier.classes_.tolist())
        coefficients = _float_array(classifier.coef_, name=COEFFICIENTS_ARRAY)
        intercepts = _float_array(classifier.intercept_, name=INTERCEPTS_ARRAY)
    except AttributeError as exc:
        raise ValueError("candidate linear classifier is not fitted") from exc
    if fitted_labels != run.class_labels:
        raise ValueError("fitted class order does not match the training run")
    class_indices = np.arange(len(fitted_labels), dtype=np.dtype("<i8"))
    arrays: dict[str, Any] = {
        COEFFICIENTS_ARRAY: coefficients,
        INTERCEPTS_ARRAY: intercepts,
        CLASS_INDICES_ARRAY: class_indices,
    }
    if candidate.feature_set == EMBEDDING_PLUS_STRUCTURED_FEATURE_SET:
        features = named_steps.get("features")
        transformers = getattr(features, "named_transformers_", None)
        if not isinstance(transformers, Mapping) or "continuous" not in transformers:
            raise ValueError("structured classifier lacks fitted preprocessing")
        continuous = transformers["continuous"]
        continuous_steps = getattr(continuous, "named_steps", None)
        if not isinstance(continuous_steps, Mapping):
            raise ValueError("structured classifier preprocessing is malformed")
        try:
            imputer = continuous_steps["imputer"]
            scaler = continuous_steps["scaler"]
            arrays.update(
                {
                    CONTINUOUS_IMPUTER_STATISTICS_ARRAY: _float_array(
                        imputer.statistics_,
                        name=CONTINUOUS_IMPUTER_STATISTICS_ARRAY,
                    ),
                    CONTINUOUS_SCALER_MEAN_ARRAY: _float_array(
                        scaler.mean_,
                        name=CONTINUOUS_SCALER_MEAN_ARRAY,
                    ),
                    CONTINUOUS_SCALER_SCALE_ARRAY: _float_array(
                        scaler.scale_,
                        name=CONTINUOUS_SCALER_SCALE_ARRAY,
                    ),
                    CONTINUOUS_SCALER_VARIANCE_ARRAY: _float_array(
                        scaler.var_,
                        name=CONTINUOUS_SCALER_VARIANCE_ARRAY,
                    ),
                }
            )
        except (AttributeError, KeyError) as exc:
            raise ValueError(
                "structured classifier preprocessing is not fitted"
            ) from exc
    _validate_extracted_array_shapes(
        arrays,
        feature_layout=candidate.feature_layout,
        class_count=len(run.class_labels),
    )
    return arrays


def _validate_extracted_array_shapes(
    arrays: Mapping[str, Any],
    *,
    feature_layout: ClassifierFeatureLayout,
    class_count: int,
) -> None:
    expected_keys = _expected_array_keys(feature_layout.feature_set)
    if set(arrays) != expected_keys:
        raise ValueError("fitted classifier arrays do not match the feature set")
    if class_count < 2:
        raise ValueError("classifier artifacts require at least two classes")
    coefficient_rows = 1 if class_count == 2 else class_count
    feature_count = len(feature_layout.transformed_feature_names)
    if arrays[COEFFICIENTS_ARRAY].shape != (coefficient_rows, feature_count):
        raise ValueError("classifier coefficient shape is inconsistent")
    if arrays[INTERCEPTS_ARRAY].shape != (coefficient_rows,):
        raise ValueError("classifier intercept shape is inconsistent")
    if arrays[CLASS_INDICES_ARRAY].shape != (class_count,):
        raise ValueError("classifier class-index shape is inconsistent")
    continuous_count = len(feature_layout.continuous_column_indices)
    for name in _STRUCTURED_ARRAY_KEYS:
        if name in arrays and arrays[name].shape != (continuous_count,):
            raise ValueError(f"classifier {name} shape is inconsistent")
    if CONTINUOUS_SCALER_SCALE_ARRAY in arrays and bool(
        (arrays[CONTINUOUS_SCALER_SCALE_ARRAY] <= 0.0).any()
    ):
        raise ValueError("classifier scaling values must be positive")
    if CONTINUOUS_SCALER_VARIANCE_ARRAY in arrays and bool(
        (arrays[CONTINUOUS_SCALER_VARIANCE_ARRAY] < 0.0).any()
    ):
        raise ValueError("classifier variance values cannot be negative")


def _deterministic_npz(arrays: Mapping[str, Any]) -> bytes:
    np = _load_numpy()
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_STORED,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(arrays):
            array = arrays[name]
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer,
                array,
                version=(2, 0),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, buffer.getvalue())
    return output.getvalue()


def _array_manifest_entry(array: Any) -> dict[str, object]:
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "size_bytes": int(array.nbytes),
        "raw_sha256": _bytes_sha256(array.tobytes(order="C")),
    }


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError("classifier manifest is unreadable") from exc
    if size <= 0 or size > MAX_CLASSIFIER_MANIFEST_BYTES:
        raise ValueError("classifier manifest size is invalid")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("classifier manifest is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("classifier manifest must be a JSON object")
    if raw != _canonical_json_bytes(payload):
        raise ValueError("classifier manifest is not canonical JSON")
    return payload


def _validate_manifest(payload: Mapping[str, object]) -> dict[str, object]:
    _expect_keys(payload, _TOP_LEVEL_KEYS, field="manifest")
    if payload["schema_version"] != CLASSIFIER_MANIFEST_SCHEMA_VERSION:
        raise ValueError("classifier manifest schema version is incompatible")
    if payload["classifier_version"] != CLASSIFIER_VERSION:
        raise ValueError("classifier version is incompatible")
    _sha256_fingerprint(
        payload["classifier_fingerprint"], field="classifier_fingerprint"
    )
    _validate_creation_timestamp(payload["created_at"])
    _git_sha(payload["git_sha"])
    if payload["qa_status"] != CLASSIFIER_QA_STATUS:
        raise ValueError("classifier artifact QA status is not passed")

    identity = _mapping(payload["identity"], field="identity")
    _expect_keys(
        identity,
        {
            "class_labels",
            "estimator_family",
            "model_name",
            "probability_calibrated",
            "route",
            "target_accepted_taxon_key",
            "target_task",
        },
        field="identity",
    )
    model_name = _required_text(identity["model_name"], field="identity.model_name")
    expected_family, expected_feature_set = _linear_model_identity(model_name)
    if identity["estimator_family"] != expected_family:
        raise ValueError("classifier estimator family conflicts with its model name")
    task = _required_text(identity["target_task"], field="identity.target_task")
    if task not in TARGET_TASKS:
        raise ValueError("classifier target task is unsupported")
    route = _required_text(identity["route"], field="identity.route")
    if route not in REFERENCE_ROUTES:
        raise ValueError("classifier route is unsupported")
    if task == "larval_target_verifier" and route != "larval":
        raise ValueError("larval classifier artifact requires the larval route")
    _required_text(
        identity["target_accepted_taxon_key"],
        field="identity.target_accepted_taxon_key",
    )
    class_labels = _string_tuple(
        identity["class_labels"], field="identity.class_labels"
    )
    if len(class_labels) < 2 or class_labels != tuple(sorted(set(class_labels))):
        raise ValueError("classifier class labels must be unique and sorted")
    if identity["probability_calibrated"] is not False:
        raise ValueError("classifier artifact cannot claim calibrated probabilities")

    features = _mapping(payload["features"], field="features")
    _expect_keys(
        features,
        {
            "continuous_column_indices",
            "embedding_column_indices",
            "embedding_dimension",
            "feature_layout_fingerprint",
            "feature_layout_version",
            "feature_set",
            "indicator_column_indices",
            "quality_flag_hash_bucket_count",
            "quality_flag_hash_version",
            "raw_feature_dtypes",
            "raw_feature_names",
            "source_feature_names",
            "transformed_feature_dtypes",
            "transformed_feature_names",
        },
        field="features",
    )
    if features["feature_layout_version"] != FEATURE_LAYOUT_VERSION:
        raise ValueError("classifier feature layout version is incompatible")
    feature_set = _required_text(features["feature_set"], field="features.feature_set")
    if feature_set != expected_feature_set:
        raise ValueError("classifier feature set conflicts with its model name")
    embedding_dimension = _positive_integer(
        features["embedding_dimension"],
        field="features.embedding_dimension",
    )
    feature_layout = classifier_feature_layout(feature_set, embedding_dimension)
    _validate_feature_layout_payload(features, feature_layout)

    preprocessing = _mapping(payload["preprocessing"], field="preprocessing")
    _validate_preprocessing_payload(preprocessing, feature_layout)
    training = _mapping(payload["training"], field="training")
    _validate_training_payload(training)
    validation = _mapping(payload["validation"], field="validation")
    _validate_validation_payload(validation, class_labels)
    arrays = _mapping(payload["arrays"], field="arrays")
    _validate_arrays_payload(arrays, feature_layout, len(class_labels))
    return {
        "identity": identity,
        "class_labels": class_labels,
        "feature_layout": feature_layout,
        "training": training,
        "arrays": arrays,
    }


def _validate_feature_layout_payload(
    payload: Mapping[str, object],
    layout: ClassifierFeatureLayout,
) -> None:
    expected_sequences = {
        "source_feature_names": layout.source_feature_names,
        "raw_feature_names": layout.raw_feature_names,
        "transformed_feature_names": layout.transformed_feature_names,
        "embedding_column_indices": layout.embedding_column_indices,
        "continuous_column_indices": layout.continuous_column_indices,
        "indicator_column_indices": layout.indicator_column_indices,
    }
    for field, expected in expected_sequences.items():
        value = payload[field]
        if not isinstance(value, list) or tuple(value) != expected:
            raise ValueError(f"classifier {field} does not match the feature layout")
    raw_dtypes = payload["raw_feature_dtypes"]
    transformed_dtypes = payload["transformed_feature_dtypes"]
    if raw_dtypes != ["<f8"] * len(layout.raw_feature_names):
        raise ValueError("classifier raw feature dtypes are invalid")
    if transformed_dtypes != ["<f8"] * len(layout.transformed_feature_names):
        raise ValueError("classifier transformed feature dtypes are invalid")
    if payload["feature_layout_fingerprint"] != layout.fingerprint:
        raise ValueError("classifier feature layout fingerprint is invalid")
    structured = layout.feature_set == EMBEDDING_PLUS_STRUCTURED_FEATURE_SET
    expected_hash_version = QUALITY_FLAG_HASH_VERSION if structured else None
    expected_bucket_count = QUALITY_FLAG_HASH_BUCKET_COUNT if structured else None
    if (
        payload["quality_flag_hash_version"] != expected_hash_version
        or payload["quality_flag_hash_bucket_count"] != expected_bucket_count
    ):
        raise ValueError("classifier quality-flag feature policy is invalid")


def _validate_preprocessing_payload(
    payload: Mapping[str, object],
    layout: ClassifierFeatureLayout,
) -> None:
    _expect_keys(
        payload,
        {
            "continuous_imputation_policy",
            "continuous_imputer_statistics_array",
            "continuous_scaler_mean_array",
            "continuous_scaler_scale_array",
            "continuous_scaler_variance_array",
            "continuous_scaling_policy",
            "embedding_policy",
            "indicator_policy",
        },
        field="preprocessing",
    )
    structured = layout.feature_set == EMBEDDING_PLUS_STRUCTURED_FEATURE_SET
    expected = {
        "embedding_policy": "passthrough",
        "continuous_imputation_policy": "median" if structured else "none",
        "continuous_scaling_policy": "standard" if structured else "none",
        "indicator_policy": "passthrough" if structured else "none",
        "continuous_imputer_statistics_array": (
            CONTINUOUS_IMPUTER_STATISTICS_ARRAY if structured else None
        ),
        "continuous_scaler_mean_array": (
            CONTINUOUS_SCALER_MEAN_ARRAY if structured else None
        ),
        "continuous_scaler_scale_array": (
            CONTINUOUS_SCALER_SCALE_ARRAY if structured else None
        ),
        "continuous_scaler_variance_array": (
            CONTINUOUS_SCALER_VARIANCE_ARRAY if structured else None
        ),
    }
    if dict(payload) != expected:
        raise ValueError("classifier preprocessing policy is invalid")


def _validate_training_payload(payload: Mapping[str, object]) -> None:
    fingerprint_fields = {
        "candidate_fingerprint",
        "candidate_set_fingerprint",
        "configuration_fingerprint",
        "cv_split_fingerprint",
        "feature_schema_fingerprint",
        "fit_partition_fingerprint",
        "foundation_model_fingerprint",
        "preprocessing_fingerprint",
        "reference_bank_fingerprint",
        "reference_embedding_fingerprint",
        "reference_prototype_fingerprint",
        "support_manifest_fingerprint",
        "training_data_fingerprint",
        "training_run_fingerprint",
    }
    _expect_keys(
        payload,
        {
            *fingerprint_fields,
            "class_weight_policy",
            "deterministic_seed",
            "estimator_configuration",
            "foundation_model_trainable",
            "model_selection_partition_fingerprint",
            "numpy_version",
            "reference_bank_version",
            "scikit_learn_version",
            "search_grid",
            "selected_parameters",
            "training_version",
        },
        field="training",
    )
    if payload["training_version"] != CLASSIFIER_TRAINING_VERSION:
        raise ValueError("classifier training version is incompatible")
    for field in fingerprint_fields:
        _sha256_fingerprint(payload[field], field=f"training.{field}")
    selection_fingerprint = payload["model_selection_partition_fingerprint"]
    if selection_fingerprint is not None:
        _sha256_fingerprint(
            selection_fingerprint,
            field="training.model_selection_partition_fingerprint",
        )
    _required_text(payload["reference_bank_version"], field="reference_bank_version")
    _required_text(payload["numpy_version"], field="training.numpy_version")
    _required_text(
        payload["scikit_learn_version"],
        field="training.scikit_learn_version",
    )
    if payload["foundation_model_trainable"] is not False:
        raise ValueError("persisted classifier cannot make BioCLIP trainable")
    seed = _nonnegative_integer(
        payload["deterministic_seed"],
        field="training.deterministic_seed",
    )
    configuration = _mapping(
        payload["estimator_configuration"],
        field="training.estimator_configuration",
    )
    if configuration.get("random_seed") != seed:
        raise ValueError("classifier deterministic seed conflicts with configuration")
    if configuration.get("probability_calibrated") is not False:
        raise ValueError("classifier estimator configuration is not uncalibrated")
    if configuration.get("class_weight") != payload["class_weight_policy"]:
        raise ValueError("classifier class-weight policy conflicts with configuration")
    search_grid = _mapping(payload["search_grid"], field="training.search_grid")
    selected = _mapping(
        payload["selected_parameters"],
        field="training.selected_parameters",
    )
    if not search_grid or set(selected) != set(search_grid):
        raise ValueError("classifier selected parameters do not match the search grid")
    for name, values in search_grid.items():
        if not isinstance(name, str) or not isinstance(values, list) or not values:
            raise ValueError("classifier search grid is malformed")
        if selected[name] not in values:
            raise ValueError("classifier selected parameter is outside the search grid")


def _validate_validation_payload(
    payload: Mapping[str, object],
    class_labels: tuple[str, ...],
) -> None:
    _expect_keys(
        payload,
        {
            "best_cv_balanced_accuracy",
            "best_cv_macro_f1",
            "calibration_sample_count",
            "cv_split_version",
            "final_test_sample_count",
            "fit_class_group_counts",
            "fit_class_sample_counts",
            "fit_group_count",
            "fit_sample_count",
            "fold_count",
            "group_field",
            "model_selection_metrics",
            "model_selection_sample_count",
            "primary_metric",
            "probability_calibrated",
            "qa_status",
            "secondary_metric",
            "splitter",
            "validation_policy",
        },
        field="validation",
    )
    expected_values = {
        "qa_status": CLASSIFIER_QA_STATUS,
        "validation_policy": CLASSIFIER_VALIDATION_POLICY,
        "cv_split_version": CV_SPLIT_VERSION,
        "splitter": "StratifiedGroupKFold",
        "group_field": "leakage_group_id",
        "primary_metric": PRIMARY_CV_METRIC,
        "secondary_metric": SECONDARY_CV_METRIC,
        "probability_calibrated": False,
    }
    if any(payload[name] != value for name, value in expected_values.items()):
        raise ValueError("classifier validation policy is invalid")
    fold_count = _positive_integer(payload["fold_count"], field="validation.fold_count")
    if fold_count < 2:
        raise ValueError("classifier validation requires at least two folds")
    fit_sample_count = _positive_integer(
        payload["fit_sample_count"],
        field="validation.fit_sample_count",
    )
    fit_group_count = _positive_integer(
        payload["fit_group_count"],
        field="validation.fit_group_count",
    )
    sample_counts = _validate_count_payload(
        payload["fit_class_sample_counts"],
        class_labels,
        field="validation.fit_class_sample_counts",
    )
    group_counts = _validate_count_payload(
        payload["fit_class_group_counts"],
        class_labels,
        field="validation.fit_class_group_counts",
    )
    if sum(sample_counts) != fit_sample_count or sum(group_counts) != fit_group_count:
        raise ValueError("classifier fit counts are internally inconsistent")
    if any(count < fold_count for count in group_counts):
        raise ValueError("classifier group counts cannot support the persisted folds")
    selection_count = _nonnegative_integer(
        payload["model_selection_sample_count"],
        field="validation.model_selection_sample_count",
    )
    _nonnegative_integer(
        payload["calibration_sample_count"],
        field="validation.calibration_sample_count",
    )
    _nonnegative_integer(
        payload["final_test_sample_count"],
        field="validation.final_test_sample_count",
    )
    _unit_metric(
        payload["best_cv_balanced_accuracy"],
        field="validation.best_cv_balanced_accuracy",
    )
    _unit_metric(
        payload["best_cv_macro_f1"],
        field="validation.best_cv_macro_f1",
    )
    metrics = payload["model_selection_metrics"]
    if selection_count == 0:
        if metrics is not None:
            raise ValueError(
                "classifier has metrics for an empty model-selection split"
            )
        return
    metric_payload = _mapping(metrics, field="validation.model_selection_metrics")
    _expect_keys(
        metric_payload,
        {"balanced_accuracy", "class_sample_counts", "macro_f1", "sample_count"},
        field="validation.model_selection_metrics",
    )
    if (
        _positive_integer(
            metric_payload["sample_count"],
            field="validation.model_selection_metrics.sample_count",
        )
        != selection_count
    ):
        raise ValueError("classifier model-selection sample count is inconsistent")
    metric_counts = _validate_count_payload(
        metric_payload["class_sample_counts"],
        class_labels,
        field="validation.model_selection_metrics.class_sample_counts",
    )
    if sum(metric_counts) != selection_count:
        raise ValueError("classifier model-selection class counts are inconsistent")
    _unit_metric(
        metric_payload["balanced_accuracy"],
        field="validation.model_selection_metrics.balanced_accuracy",
    )
    _unit_metric(
        metric_payload["macro_f1"],
        field="validation.model_selection_metrics.macro_f1",
    )


def _validate_arrays_payload(
    payload: Mapping[str, object],
    layout: ClassifierFeatureLayout,
    class_count: int,
) -> None:
    _expect_keys(
        payload,
        {"entries", "file_name", "sha256", "size_bytes", "uri"},
        field="arrays",
    )
    if (
        payload["file_name"] != CLASSIFIER_ARRAYS_FILE
        or payload["uri"] != CLASSIFIER_ARRAYS_FILE
    ):
        raise ValueError("classifier array archive path is invalid")
    _sha256_fingerprint(payload["sha256"], field="arrays.sha256")
    size = _positive_integer(payload["size_bytes"], field="arrays.size_bytes")
    if size > MAX_CLASSIFIER_ARRAY_ARCHIVE_BYTES:
        raise ValueError("classifier array archive exceeds the configured size limit")
    entries = _mapping(payload["entries"], field="arrays.entries")
    expected_keys = _expected_array_keys(layout.feature_set)
    if set(entries) != expected_keys:
        raise ValueError("classifier array manifest keys are invalid")
    coefficient_rows = 1 if class_count == 2 else class_count
    continuous_count = len(layout.continuous_column_indices)
    expected_shapes = {
        COEFFICIENTS_ARRAY: (coefficient_rows, len(layout.transformed_feature_names)),
        INTERCEPTS_ARRAY: (coefficient_rows,),
        CLASS_INDICES_ARRAY: (class_count,),
        CONTINUOUS_IMPUTER_STATISTICS_ARRAY: (continuous_count,),
        CONTINUOUS_SCALER_MEAN_ARRAY: (continuous_count,),
        CONTINUOUS_SCALER_SCALE_ARRAY: (continuous_count,),
        CONTINUOUS_SCALER_VARIANCE_ARRAY: (continuous_count,),
    }
    for name, raw_spec in entries.items():
        spec = _mapping(raw_spec, field=f"arrays.entries.{name}")
        _expect_keys(
            spec,
            {"dtype", "raw_sha256", "shape", "size_bytes"},
            field=f"arrays.entries.{name}",
        )
        expected_dtype = "<i8" if name == CLASS_INDICES_ARRAY else "<f8"
        if spec["dtype"] != expected_dtype:
            raise ValueError(f"classifier array {name} dtype is invalid")
        shape = spec["shape"]
        if not isinstance(shape, list) or tuple(shape) != expected_shapes[name]:
            raise ValueError(f"classifier array {name} shape is invalid")
        _sha256_fingerprint(
            spec["raw_sha256"],
            field=f"arrays.entries.{name}.raw_sha256",
        )
        expected_bytes = 8
        for dimension in expected_shapes[name]:
            expected_bytes *= dimension
        if (
            _nonnegative_integer(
                spec["size_bytes"],
                field=f"arrays.entries.{name}.size_bytes",
            )
            != expected_bytes
        ):
            raise ValueError(f"classifier array {name} byte size is invalid")


def _load_array_archive(
    value: bytes,
    *,
    specs: Mapping[str, object],
    feature_layout: ClassifierFeatureLayout,
    class_count: int,
) -> dict[str, Any]:
    expected_members = tuple(sorted(f"{name}.npy" for name in specs))
    try:
        with zipfile.ZipFile(io.BytesIO(value), mode="r") as archive:
            infos = archive.infolist()
            member_names = tuple(info.filename for info in infos)
            if member_names != expected_members or len(set(member_names)) != len(infos):
                raise ValueError("classifier archive members do not match the manifest")
            total_size = 0
            for info in infos:
                if (
                    info.is_dir()
                    or info.flag_bits & 0x1
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.file_size != info.compress_size
                ):
                    raise ValueError(
                        "classifier archive members are not safe numeric files"
                    )
                total_size += info.file_size
            if total_size > MAX_CLASSIFIER_ARRAY_UNCOMPRESSED_BYTES:
                raise ValueError("classifier archive expands beyond the size limit")
    except zipfile.BadZipFile as exc:
        raise ValueError("invalid classifier array archive") from exc

    np = _load_numpy()
    arrays: dict[str, Any] = {}
    try:
        with np.load(
            io.BytesIO(value),
            allow_pickle=False,
            max_header_size=MAX_NUMPY_HEADER_BYTES,
        ) as loaded:
            if tuple(sorted(loaded.files)) != tuple(sorted(specs)):
                raise ValueError("classifier archive members do not match the manifest")
            for name in sorted(specs):
                arrays[name] = np.array(loaded[name], copy=True, order="C")
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ValueError) and "archive members" in str(exc):
            raise
        raise ValueError("invalid classifier array archive") from exc

    for name, array in arrays.items():
        spec = _mapping(specs[name], field=f"arrays.entries.{name}")
        if array.dtype.hasobject or array.dtype.kind not in "fiu":
            raise ValueError(f"classifier array {name} is not numeric")
        if array.dtype.str != spec["dtype"] or list(array.shape) != spec["shape"]:
            raise ValueError(f"classifier array {name} does not match its manifest")
        if int(array.nbytes) != spec["size_bytes"]:
            raise ValueError(f"classifier array {name} byte size is invalid")
        if _bytes_sha256(array.tobytes(order="C")) != spec["raw_sha256"]:
            raise ValueError(f"classifier array {name} checksum is invalid")
        if array.dtype.kind == "f" and not bool(np.isfinite(array).all()):
            raise ValueError(f"classifier array {name} contains non-finite values")
    _validate_extracted_array_shapes(
        arrays,
        feature_layout=feature_layout,
        class_count=class_count,
    )
    expected_indices = np.arange(class_count, dtype=np.dtype("<i8"))
    if not bool(np.array_equal(arrays[CLASS_INDICES_ARRAY], expected_indices)):
        raise ValueError("classifier numeric class indices are invalid")
    return arrays


def _classifier_fingerprint(manifest: Mapping[str, object]) -> str:
    without_fingerprint = dict(manifest)
    without_fingerprint.pop("classifier_fingerprint", None)
    arrays = _mapping(manifest["arrays"], field="arrays")
    return canonical_semantic_fingerprint(
        {
            "manifest": without_fingerprint,
            "classifier_arrays_sha256": arrays["sha256"],
        }
    )


def _select_candidate(
    run: ClassifierTrainingRun,
    model_name: str | None,
) -> ClassifierCandidateResult:
    selected_name = (
        run.selected_model_name
        if model_name is None
        else _required_text(
            model_name,
            field="model_name",
        )
    )
    matches = tuple(item for item in run.candidates if item.model_name == selected_name)
    if len(matches) != 1:
        raise ValueError("requested classifier candidate is not unique in the run")
    return matches[0]


def _linear_model_identity(model_name: object) -> tuple[str, str]:
    name = _required_text(model_name, field="model_name")
    try:
        return _LINEAR_MODEL_IDENTITIES[name]
    except KeyError as exc:
        raise ValueError(
            "only supported fitted linear classifiers can be persisted"
        ) from exc


def _expected_array_keys(feature_set: str) -> frozenset[str]:
    if feature_set == EMBEDDING_ONLY_FEATURE_SET:
        return _BASE_ARRAY_KEYS
    if feature_set == EMBEDDING_PLUS_STRUCTURED_FEATURE_SET:
        return _BASE_ARRAY_KEYS | _STRUCTURED_ARRAY_KEYS
    raise ValueError("unsupported classifier feature set")


def _float_array(value: object, *, name: str) -> Any:
    np = _load_numpy()
    try:
        result = np.asarray(value, dtype=np.dtype("<f8"), order="C")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"classifier {name} is not a numeric array") from exc
    result = np.ascontiguousarray(result)
    if result.dtype.hasobject or not bool(np.isfinite(result).all()):
        raise ValueError(f"classifier {name} must contain finite numeric values")
    return result


def _empty_float_array() -> Any:
    np = _load_numpy()
    result = np.empty((0,), dtype=np.dtype("<f8"))
    result.setflags(write=False)
    return result


def _load_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency guard.
        raise RuntimeError(
            "classifier persistence requires the 'ml' dependency group"
        ) from exc
    return np


def _counts_payload(values: Sequence[tuple[str, int]]) -> list[dict[str, object]]:
    return [
        {"class_label": str(class_label), "count": int(count)}
        for class_label, count in values
    ]


def _validate_count_payload(
    value: object,
    class_labels: tuple[str, ...],
    *,
    field: str,
) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != len(class_labels):
        raise ValueError(f"{field} must contain one row per class")
    counts: list[int] = []
    for expected_label, raw_item in zip(class_labels, value, strict=True):
        item = _mapping(raw_item, field=field)
        _expect_keys(item, {"class_label", "count"}, field=field)
        if item["class_label"] != expected_label:
            raise ValueError(f"{field} class order is invalid")
        counts.append(_positive_integer(item["count"], field=f"{field}.count"))
    return tuple(counts)


def _pairs_to_json_mapping(
    values: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in values:
        key = _required_text(name, field="configuration key")
        if key in result:
            raise ValueError("classifier configuration has duplicate keys")
        result[key] = _json_value(value)
    return result


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("classifier metadata cannot contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        return {
            _required_text(key, field="metadata key"): _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        return _json_value(item_method())
    raise TypeError(
        f"classifier metadata value is not JSON-compatible: {type(value)!r}"
    )


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


def _bytes_sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


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


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    return tuple(_required_text(item, field=field) for item in value)


def _sha256_fingerprint(value: object, *, field: str) -> str:
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


def _unit_metric(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be finite and between zero and one")
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
    validated = _sha256_fingerprint(expected, field=f"expected_{field}")
    if actual != validated:
        raise ValueError(f"{field} does not match the expected fingerprint")


__all__ = [
    "CLASSIFIER_ARRAYS_FILE",
    "CLASSIFIER_MANIFEST_FILE",
    "CLASSIFIER_MANIFEST_SCHEMA_VERSION",
    "CLASSIFIER_QA_STATUS",
    "CLASSIFIER_VALIDATION_POLICY",
    "CLASSIFIER_VERSION",
    "MAX_CLASSIFIER_ARRAY_ARCHIVE_BYTES",
    "MAX_CLASSIFIER_MANIFEST_BYTES",
    "ClassifierArtifactPaths",
    "FrozenLinearClassifier",
    "load_frozen_classifier",
    "write_frozen_classifier",
]
