"""Leakage-safe calibrated evidence models for frozen dynamic-pool features."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, log
from types import MappingProxyType
import warnings

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.calibration import (
    CALIBRATED_TARGET_PROBABILITY_KIND,
    TargetCalibrationDiagnostics,
    build_target_calibration_diagnostics,
)
from biominer.ml.calibration import (
    SIGMOID_METHOD,
    CalibrationConfig,
    CalibrationFit,
    CalibrationFoldAudit,
    CalibrationPrediction,
    FrozenProbabilityCalibrator,
    fit_probability_calibrator,
)
from biominer.ml.dynamic_pool_features import (
    DYNAMIC_POOL_MODEL_FEATURE_NAMES,
    validate_dynamic_pool_feature_table,
)
from biominer.references.schemas import REFERENCE_ROUTES


DYNAMIC_POOL_CALIBRATION_VERSION = "dynamic-pool-calibrated-evidence-v1.0.0"
DYNAMIC_POOL_EVIDENCE_MODEL_VERSION = "standardized-l2-logistic-evidence-v1.0.0"
DYNAMIC_POOL_CALIBRATION_PREDICTION_SCHEMA_VERSION = (
    "dynamic-pool-calibration-prediction-v1.0.0"
)
CALIBRATION_OUTCOME_LABELS = ("error", "supported")

DYNAMIC_POOL_CALIBRATION_PREDICTION_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "evidence_model_fingerprint": pl.String,
    "calibrator_fingerprint": pl.String,
    "split_fingerprint": pl.String,
    "feature_schema_fingerprint": pl.String,
    "source_feature_row_fingerprint": pl.String,
    "prediction_fingerprint": pl.String,
    "item_id": pl.String,
    "independence_component_id": pl.String,
    "evaluation_split": pl.String,
    "prediction_role": pl.String,
    "fold_index": pl.Int32,
    "raw_evidence_logit": pl.Float64,
    CALIBRATED_TARGET_PROBABILITY_KIND: pl.Float64,
    "human_supported": pl.Boolean,
    "sampling_weight": pl.Float64,
}


@dataclass(frozen=True, slots=True)
class DynamicPoolCalibrationConfig:
    """Versioned fitting and validation policy for one visual route."""

    route: str
    random_seed: int = 42
    maximum_cross_validation_folds: int = 5
    regularization_c: float = 1.0
    maximum_iterations: int = 2_000
    convergence_tolerance: float = 1e-10
    reliability_bin_count: int = 10
    validation_confidence_level: float = 0.95

    def __post_init__(self) -> None:
        route = _required_text(self.route, field="route")
        if route not in REFERENCE_ROUTES:
            raise ValueError(f"unsupported route: {route}")
        object.__setattr__(self, "route", route)
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or not 0 <= self.random_seed < 2**32
        ):
            raise ValueError("random_seed must be an unsigned 32-bit integer")
        for field, minimum in (
            ("maximum_cross_validation_folds", 2),
            ("maximum_iterations", 1),
            ("reliability_bin_count", 2),
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{field} must be an integer >= {minimum}")
        for field in ("regularization_c", "convergence_tolerance"):
            value = _positive_float(getattr(self, field), field=field)
            object.__setattr__(self, field, value)
        confidence = _finite_float(
            self.validation_confidence_level,
            field="validation_confidence_level",
        )
        if not 0.0 < confidence < 1.0:
            raise ValueError("validation_confidence_level must be in (0, 1)")
        object.__setattr__(self, "validation_confidence_level", confidence)

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": DYNAMIC_POOL_CALIBRATION_VERSION,
                "evidence_model_version": DYNAMIC_POOL_EVIDENCE_MODEL_VERSION,
                "route": self.route,
                "random_seed": self.random_seed,
                "maximum_cross_validation_folds": (self.maximum_cross_validation_folds),
                "regularization_c": self.regularization_c,
                "maximum_iterations": self.maximum_iterations,
                "convergence_tolerance": self.convergence_tolerance,
                "reliability_bin_count": self.reliability_bin_count,
                "validation_confidence_level": self.validation_confidence_level,
                "fit_partition": "calibration",
                "reliability_partition": "validation",
                "final_test_policy": "untouched",
            }
        )


@dataclass(frozen=True, slots=True)
class FrozenDynamicPoolEvidenceModel:
    """Transparent standardized linear logit runtime without sklearn state."""

    model_fingerprint: str
    config_fingerprint: str
    split_fingerprint: str
    feature_schema_fingerprint: str
    route: str
    feature_names: tuple[str, ...]
    calibration_row_fingerprints: tuple[str, ...]
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float

    def __post_init__(self) -> None:
        for field in (
            "model_fingerprint",
            "config_fingerprint",
            "split_fingerprint",
            "feature_schema_fingerprint",
        ):
            _sha256(getattr(self, field), field=field)
        route = _required_text(self.route, field="route")
        if route not in REFERENCE_ROUTES:
            raise ValueError("evidence model has an unsupported route")
        names = tuple(self.feature_names)
        if names != DYNAMIC_POOL_MODEL_FEATURE_NAMES:
            raise ValueError("evidence model feature names differ from contract")
        width = len(names)
        for field in ("scaler_mean", "scaler_scale", "coefficients"):
            values = _finite_tuple(getattr(self, field), field=field)
            if len(values) != width:
                raise ValueError(f"{field} width differs from feature contract")
            if field == "scaler_scale" and any(value <= 0.0 for value in values):
                raise ValueError("scaler_scale values must be positive")
            object.__setattr__(self, field, values)
        row_fingerprints = tuple(
            _sha256(value, field="calibration_row_fingerprints")
            for value in self.calibration_row_fingerprints
        )
        if not row_fingerprints or len(row_fingerprints) != len(set(row_fingerprints)):
            raise ValueError("calibration row fingerprints must be unique and nonempty")
        object.__setattr__(self, "calibration_row_fingerprints", row_fingerprints)
        object.__setattr__(
            self, "intercept", _finite_float(self.intercept, field="intercept")
        )
        expected = _evidence_model_fingerprint(
            config_fingerprint=self.config_fingerprint,
            split_fingerprint=self.split_fingerprint,
            feature_schema_fingerprint=self.feature_schema_fingerprint,
            route=route,
            calibration_row_fingerprints=row_fingerprints,
            scaler_mean=self.scaler_mean,
            scaler_scale=self.scaler_scale,
            coefficients=self.coefficients,
            intercept=self.intercept,
        )
        if self.model_fingerprint != expected:
            raise ValueError("evidence model fingerprint mismatch")

    def decision_function(self, feature_vectors: object) -> object:
        """Return raw linear logits; these are not probabilities."""

        np = _load_numpy()
        values = np.asarray(feature_vectors, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("dynamic-pool feature matrix has the wrong shape")
        if not bool(np.isfinite(values).all()):
            raise ValueError("dynamic-pool feature matrix must be finite")
        mean = np.asarray(self.scaler_mean, dtype=np.float64)
        scale = np.asarray(self.scaler_scale, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        return ((values - mean) / scale) @ coefficients + self.intercept

    def predict_supported_probability(
        self,
        feature_vectors: object,
        calibrator: FrozenProbabilityCalibrator,
    ) -> object:
        """Apply the separately fingerprinted sigmoid calibrator."""

        if calibrator.classifier_fingerprint != self.model_fingerprint:
            raise ValueError("calibrator is bound to another evidence model")
        if calibrator.split_fingerprint != self.split_fingerprint:
            raise ValueError("calibrator is bound to another split")
        scores = self.decision_function(feature_vectors)
        probabilities = calibrator.predict_proba(scores)
        supported_index = calibrator.class_labels.index("supported")
        return probabilities[:, supported_index]


@dataclass(frozen=True, slots=True)
class DynamicPoolCalibrationFit:
    evidence_model: FrozenDynamicPoolEvidenceModel
    probability_calibration: CalibrationFit
    predictions: pl.DataFrame
    validation_diagnostics: TargetCalibrationDiagnostics
    validation_metrics: Mapping[str, float]
    config_fingerprint: str
    prediction_artifact_fingerprint: str
    fit_fingerprint: str
    cross_validation_fold_count: int
    calibration_sample_count: int
    validation_sample_count: int
    final_test_prediction_count: int


def fit_dynamic_pool_evidence_calibrator(
    feature_table: pl.DataFrame,
    config: DynamicPoolCalibrationConfig,
) -> DynamicPoolCalibrationFit:
    """Fit on calibration labels and assess reliability on validation labels."""

    validate_dynamic_pool_feature_table(feature_table)
    if not isinstance(config, DynamicPoolCalibrationConfig):
        raise TypeError("config must be a DynamicPoolCalibrationConfig")
    route_rows = feature_table.filter(pl.col("route") == config.route)
    calibration = route_rows.filter(pl.col("evaluation_split") == "calibration")
    validation = route_rows.filter(pl.col("evaluation_split") == "validation")
    if not calibration.height:
        raise ValueError("route has no calibration evidence")
    if not validation.height:
        raise ValueError("route has no validation evidence")
    _require_both_outcomes(calibration, partition="calibration")
    _require_both_outcomes(validation, partition="validation")
    feature_schema_fingerprint = _single_value(
        feature_table, "feature_schema_fingerprint"
    )
    split_fingerprint = _single_value(feature_table, "split_fingerprint")
    calibration_matrix = _feature_matrix(calibration)
    calibration_labels = _labels(calibration)
    calibration_weights = _weights(calibration)
    calibration_groups = tuple(calibration["independence_component_id"].to_list())
    folds = _grouped_folds(
        calibration_labels,
        calibration_groups,
        maximum_folds=config.maximum_cross_validation_folds,
        random_seed=config.random_seed,
    )
    np = _load_numpy()
    oof_scores = np.full(calibration.height, np.nan, dtype=np.float64)
    oof_fold_indices = np.full(calibration.height, -1, dtype=np.int32)
    fold_audits = []
    for fold_index, (fit_indices, held_out_indices) in enumerate(folds):
        fitted = _fit_logistic(
            calibration_matrix[fit_indices],
            calibration_labels[fit_indices],
            calibration_weights[fit_indices],
            config=config,
        )
        oof_scores[held_out_indices] = _decision_scores(
            fitted,
            calibration_matrix[held_out_indices],
        )
        oof_fold_indices[held_out_indices] = fold_index
        fit_groups = tuple(sorted({calibration_groups[index] for index in fit_indices}))
        validation_groups = tuple(
            sorted({calibration_groups[index] for index in held_out_indices})
        )
        fold_audits.append(
            CalibrationFoldAudit(
                fold_index=fold_index,
                estimator_fit_group_ids=fit_groups,
                validation_group_ids=validation_groups,
            )
        )
    if not bool(np.isfinite(oof_scores).all()) or bool((oof_fold_indices < 0).any()):
        raise AssertionError("grouped OOF prediction coverage is incomplete")
    full_fitted = _fit_logistic(
        calibration_matrix,
        calibration_labels,
        calibration_weights,
        config=config,
    )
    calibration_row_fingerprints = tuple(
        sorted(calibration["feature_row_fingerprint"].to_list())
    )
    model_fingerprint = _evidence_model_fingerprint(
        config_fingerprint=config.fingerprint,
        split_fingerprint=str(split_fingerprint),
        feature_schema_fingerprint=str(feature_schema_fingerprint),
        route=config.route,
        calibration_row_fingerprints=calibration_row_fingerprints,
        scaler_mean=full_fitted["mean"],
        scaler_scale=full_fitted["scale"],
        coefficients=full_fitted["coefficients"],
        intercept=float(full_fitted["intercept"]),
    )
    evidence_model = FrozenDynamicPoolEvidenceModel(
        model_fingerprint=model_fingerprint,
        config_fingerprint=config.fingerprint,
        split_fingerprint=str(split_fingerprint),
        feature_schema_fingerprint=str(feature_schema_fingerprint),
        route=config.route,
        feature_names=DYNAMIC_POOL_MODEL_FEATURE_NAMES,
        calibration_row_fingerprints=calibration_row_fingerprints,
        scaler_mean=tuple(full_fitted["mean"]),
        scaler_scale=tuple(full_fitted["scale"]),
        coefficients=tuple(full_fitted["coefficients"]),
        intercept=float(full_fitted["intercept"]),
    )
    calibration_predictions = tuple(
        CalibrationPrediction(
            prediction_id=f"dynamic-pool-oof:{row['item_id']}",
            source_item_id=str(row["item_id"]),
            leakage_component_id=str(row["independence_component_id"]),
            fold_index=int(oof_fold_indices[index]),
            dataset_split="calibration",
            true_class_label="supported" if row["human_supported"] else "error",
            decision_scores=(float(oof_scores[index]),),
            sample_weight=float(row["sampling_weight"]),
        )
        for index, row in enumerate(calibration.iter_rows(named=True))
    )
    probability_calibration = fit_probability_calibrator(
        calibration_predictions,
        tuple(fold_audits),
        CalibrationConfig(
            classifier_fingerprint=model_fingerprint,
            split_fingerprint=str(split_fingerprint),
            target_task="binary_target_verifier",
            route=config.route,
            class_labels=CALIBRATION_OUTCOME_LABELS,
            method=SIGMOID_METHOD,
            positive_class_label="supported",
            reliability_bin_count=config.reliability_bin_count,
            minimum_class_group_count=2,
        ),
    )
    oof_probabilities = probability_calibration.calibrator.predict_proba(oof_scores)[
        :, 1
    ]
    validation_matrix = _feature_matrix(validation)
    validation_scores = evidence_model.decision_function(validation_matrix)
    validation_probabilities = evidence_model.predict_supported_probability(
        validation_matrix,
        probability_calibration.calibrator,
    )
    predictions = _prediction_table(
        calibration=calibration,
        validation=validation,
        oof_scores=oof_scores,
        oof_probabilities=oof_probabilities,
        oof_fold_indices=oof_fold_indices,
        validation_scores=validation_scores,
        validation_probabilities=validation_probabilities,
        evidence_model=evidence_model,
        calibrator=probability_calibration.calibrator,
    )
    validation_rows = predictions.filter(pl.col("evaluation_split") == "validation")
    diagnostics = build_target_calibration_diagnostics(
        validation_rows.select(
            pl.col("item_id").alias("evaluation_item_id"),
            pl.col("evaluation_split").alias("evaluation_set"),
            "sampling_weight",
            pl.col("human_supported").alias("target_present"),
            CALIBRATED_TARGET_PROBABILITY_KIND,
            pl.lit(SIGMOID_METHOD).alias("calibration_method"),
            pl.col("split_fingerprint").alias("calibration_split_fingerprint"),
            pl.col("calibrator_fingerprint"),
        ),
        bin_count=config.reliability_bin_count,
        confidence_level=config.validation_confidence_level,
    )
    metrics = _validation_metrics(validation_rows, diagnostics=diagnostics)
    prediction_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_CALIBRATION_PREDICTION_SCHEMA_VERSION,
            "rows": predictions.to_dicts(),
        }
    )
    fit_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_CALIBRATION_VERSION,
            "config_fingerprint": config.fingerprint,
            "evidence_model_fingerprint": model_fingerprint,
            "calibrator_fingerprint": (probability_calibration.calibration_fingerprint),
            "prediction_artifact_fingerprint": prediction_fingerprint,
            "validation_diagnostics_fingerprint": diagnostics.diagnostics_fingerprint,
            "validation_metrics": dict(metrics),
            "fit_partition": "calibration",
            "reliability_partition": "validation",
            "final_test_prediction_count": 0,
        }
    )
    return DynamicPoolCalibrationFit(
        evidence_model=evidence_model,
        probability_calibration=probability_calibration,
        predictions=predictions,
        validation_diagnostics=diagnostics,
        validation_metrics=MappingProxyType(metrics),
        config_fingerprint=config.fingerprint,
        prediction_artifact_fingerprint=prediction_fingerprint,
        fit_fingerprint=fit_fingerprint,
        cross_validation_fold_count=len(folds),
        calibration_sample_count=calibration.height,
        validation_sample_count=validation.height,
        final_test_prediction_count=0,
    )


def _grouped_folds(
    labels: object,
    groups: tuple[str, ...],
    *,
    maximum_folds: int,
    random_seed: int,
) -> tuple[tuple[object, object], ...]:
    np = _load_numpy()
    labels_array = np.asarray(labels, dtype=np.int8)
    class_groups = {
        outcome: {groups[index] for index in np.flatnonzero(labels_array == outcome)}
        for outcome in (0, 1)
    }
    minimum = min(len(values) for values in class_groups.values())
    for fold_count in range(min(maximum_folds, minimum), 1, -1):
        sklearn_model_selection = _load_sklearn_model_selection()
        splitter = sklearn_model_selection.StratifiedGroupKFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=random_seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            candidates = tuple(
                (fit_indices, held_out_indices)
                for fit_indices, held_out_indices in splitter.split(
                    np.zeros(labels_array.shape[0]), labels_array, groups
                )
            )
        if all(
            set(labels_array[fit_indices].tolist()) == {0, 1}
            and not (
                {groups[index] for index in fit_indices}
                & {groups[index] for index in held_out_indices}
            )
            for fit_indices, held_out_indices in candidates
        ):
            return candidates
    raise ValueError(
        "calibration evidence cannot form two leakage-safe folds with both outcomes"
    )


def _fit_logistic(
    matrix: object,
    labels: object,
    weights: object,
    *,
    config: DynamicPoolCalibrationConfig,
) -> dict[str, object]:
    np = _load_numpy()
    linear_model = _load_sklearn_linear_model()
    exceptions = _load_sklearn_exceptions()
    values = np.asarray(matrix, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    total_weight = float(sample_weights.sum())
    mean = (sample_weights[:, None] * values).sum(axis=0) / total_weight
    centered = values - mean
    variance = (sample_weights[:, None] * centered * centered).sum(
        axis=0
    ) / total_weight
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale = np.where(scale <= np.finfo(np.float64).eps, 1.0, scale)
    standardized = centered / scale
    estimator = linear_model.LogisticRegression(
        C=config.regularization_c,
        solver="lbfgs",
        max_iter=config.maximum_iterations,
        tol=config.convergence_tolerance,
        random_state=config.random_seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", exceptions.ConvergenceWarning)
        estimator.fit(standardized, labels, sample_weight=weights)
    if any(issubclass(item.category, exceptions.ConvergenceWarning) for item in caught):
        raise ValueError("dynamic-pool evidence model did not converge")
    if tuple(estimator.classes_.tolist()) != (0, 1):
        raise ValueError("dynamic-pool evidence model requires both ordered outcomes")
    coefficients = np.asarray(estimator.coef_[0], dtype=np.float64)
    intercept = float(estimator.intercept_[0])
    if not (
        bool(np.isfinite(mean).all())
        and bool(np.isfinite(scale).all())
        and bool(np.isfinite(coefficients).all())
        and isfinite(intercept)
        and bool((scale > 0.0).all())
    ):
        raise ValueError("dynamic-pool evidence model produced invalid parameters")
    return {
        "mean": tuple(float(value) for value in mean),
        "scale": tuple(float(value) for value in scale),
        "coefficients": tuple(float(value) for value in coefficients),
        "intercept": intercept,
    }


def _decision_scores(fitted: Mapping[str, object], matrix: object) -> object:
    np = _load_numpy()
    values = np.asarray(matrix, dtype=np.float64)
    mean = np.asarray(fitted["mean"], dtype=np.float64)
    scale = np.asarray(fitted["scale"], dtype=np.float64)
    coefficients = np.asarray(fitted["coefficients"], dtype=np.float64)
    return ((values - mean) / scale) @ coefficients + float(fitted["intercept"])


def _prediction_table(
    *,
    calibration: pl.DataFrame,
    validation: pl.DataFrame,
    oof_scores: object,
    oof_probabilities: object,
    oof_fold_indices: object,
    validation_scores: object,
    validation_probabilities: object,
    evidence_model: FrozenDynamicPoolEvidenceModel,
    calibrator: FrozenProbabilityCalibrator,
) -> pl.DataFrame:
    rows = []
    partitions = (
        (
            calibration,
            "grouped_oof_calibration",
            oof_scores,
            oof_probabilities,
            oof_fold_indices,
        ),
        (
            validation,
            "independent_validation",
            validation_scores,
            validation_probabilities,
            None,
        ),
    )
    for frame, role, scores, probabilities, fold_indices in partitions:
        for index, source in enumerate(frame.iter_rows(named=True)):
            base = {
                "evidence_model_fingerprint": evidence_model.model_fingerprint,
                "calibrator_fingerprint": calibrator.calibration_fingerprint,
                "split_fingerprint": evidence_model.split_fingerprint,
                "source_feature_row_fingerprint": source["feature_row_fingerprint"],
                "item_id": source["item_id"],
                "evaluation_split": source["evaluation_split"],
                "prediction_role": role,
                "fold_index": (
                    None if fold_indices is None else int(fold_indices[index])
                ),
                "raw_evidence_logit": float(scores[index]),
                CALIBRATED_TARGET_PROBABILITY_KIND: float(probabilities[index]),
            }
            rows.append(
                {
                    "schema_version": (
                        DYNAMIC_POOL_CALIBRATION_PREDICTION_SCHEMA_VERSION
                    ),
                    **base,
                    "feature_schema_fingerprint": (
                        evidence_model.feature_schema_fingerprint
                    ),
                    "prediction_fingerprint": canonical_semantic_fingerprint(base),
                    "independence_component_id": source["independence_component_id"],
                    "human_supported": source["human_supported"],
                    "sampling_weight": source["sampling_weight"],
                }
            )
    table = pl.DataFrame(
        rows,
        schema=DYNAMIC_POOL_CALIBRATION_PREDICTION_SCHEMA,
        strict=True,
    ).sort("evaluation_split", "item_id")
    _validate_prediction_table(table)
    return table


def _validate_prediction_table(table: pl.DataFrame) -> None:
    if table.schema != DYNAMIC_POOL_CALIBRATION_PREDICTION_SCHEMA:
        raise ValueError("dynamic-pool calibration prediction schema mismatch")
    if table["item_id"].n_unique() != table.height:
        raise ValueError("dynamic-pool calibration prediction item IDs repeat")
    if set(table["evaluation_split"].to_list()) != {"calibration", "validation"}:
        raise ValueError("calibration predictions must exclude final_test")
    for row in table.iter_rows(named=True):
        probability = float(row[CALIBRATED_TARGET_PROBABILITY_KIND])
        if not 0.0 <= probability <= 1.0:
            raise ValueError("calibrated probability is outside [0, 1]")


def _validation_metrics(
    validation: pl.DataFrame,
    *,
    diagnostics: TargetCalibrationDiagnostics,
) -> dict[str, float]:
    rows = validation.to_dicts()
    total_weight = sum(float(row["sampling_weight"]) for row in rows)
    brier = (
        sum(
            float(row["sampling_weight"])
            * (
                float(row[CALIBRATED_TARGET_PROBABILITY_KIND])
                - float(bool(row["human_supported"]))
            )
            ** 2
            for row in rows
        )
        / total_weight
    )
    log_loss = (
        -sum(
            float(row["sampling_weight"])
            * (
                log(
                    min(
                        1.0 - 1e-15,
                        max(1e-15, float(row[CALIBRATED_TARGET_PROBABILITY_KIND])),
                    )
                )
                if row["human_supported"]
                else log(
                    min(
                        1.0 - 1e-15,
                        max(
                            1e-15,
                            1.0 - float(row[CALIBRATED_TARGET_PROBABILITY_KIND]),
                        ),
                    )
                )
            )
            for row in rows
        )
        / total_weight
    )
    ece = float(diagnostics.reliability["ece_contribution"].drop_nulls().sum())
    return {
        "weighted_brier_score": float(brier),
        "weighted_log_loss": float(log_loss),
        "weighted_expected_calibration_error": ece,
        "weighted_sample_count": total_weight,
    }


def _evidence_model_fingerprint(
    *,
    config_fingerprint: str,
    split_fingerprint: str,
    feature_schema_fingerprint: str,
    route: str,
    calibration_row_fingerprints: tuple[str, ...],
    scaler_mean: tuple[float, ...],
    scaler_scale: tuple[float, ...],
    coefficients: tuple[float, ...],
    intercept: float,
) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_EVIDENCE_MODEL_VERSION,
            "config_fingerprint": config_fingerprint,
            "split_fingerprint": split_fingerprint,
            "feature_schema_fingerprint": feature_schema_fingerprint,
            "route": route,
            "fit_partition": "calibration",
            "calibration_row_fingerprints": calibration_row_fingerprints,
            "feature_names": DYNAMIC_POOL_MODEL_FEATURE_NAMES,
            "scaler_mean": scaler_mean,
            "scaler_scale": scaler_scale,
            "coefficients": coefficients,
            "intercept": intercept,
        }
    )


def _feature_matrix(frame: pl.DataFrame) -> object:
    np = _load_numpy()
    values = np.asarray(frame["feature_vector"].to_list(), dtype=np.float64)
    if values.shape != (frame.height, len(DYNAMIC_POOL_MODEL_FEATURE_NAMES)):
        raise ValueError("dynamic-pool feature matrix has an invalid shape")
    if not bool(np.isfinite(values).all()):
        raise ValueError("dynamic-pool feature matrix must be finite")
    return values


def _labels(frame: pl.DataFrame) -> object:
    return _load_numpy().asarray(frame["human_supported"].to_list(), dtype="int8")


def _weights(frame: pl.DataFrame) -> object:
    return _load_numpy().asarray(frame["sampling_weight"].to_list(), dtype="float64")


def _require_both_outcomes(frame: pl.DataFrame, *, partition: str) -> None:
    if set(frame["human_supported"].to_list()) != {False, True}:
        raise ValueError(f"{partition} route evidence requires both outcomes")
    groups_by_outcome: dict[bool, set[str]] = defaultdict(set)
    for row in frame.iter_rows(named=True):
        groups_by_outcome[bool(row["human_supported"])].add(
            str(row["independence_component_id"])
        )
    if partition == "calibration" and any(
        len(groups_by_outcome[outcome]) < 2 for outcome in (False, True)
    ):
        raise ValueError(
            "calibration route evidence requires two independent groups per outcome"
        )


def _single_value(frame: pl.DataFrame, field: str) -> object:
    values = frame[field].unique().to_list()
    if len(values) != 1:
        raise ValueError(f"{field} must have one value")
    return values[0]


def _finite_tuple(values: object, *, field: str) -> tuple[float, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, tuple | list):
        raise TypeError(f"{field} must be a numeric sequence")
    return tuple(_finite_float(value, field=field) for value in values)


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _positive_float(value: object, *, field: str) -> float:
    result = _finite_float(value, field=field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value.strip()


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValueError(f"{field} must be a sha256 fingerprint")
    try:
        int(text.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a sha256 fingerprint") from exc
    return text


def _load_numpy() -> object:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("dynamic-pool calibration requires the 'ml' extra") from exc
    return np


def _load_sklearn_model_selection() -> object:
    try:
        from sklearn import model_selection
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("dynamic-pool calibration requires the 'ml' extra") from exc
    return model_selection


def _load_sklearn_linear_model() -> object:
    try:
        from sklearn import linear_model
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("dynamic-pool calibration requires the 'ml' extra") from exc
    return linear_model


def _load_sklearn_exceptions() -> object:
    try:
        from sklearn import exceptions
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("dynamic-pool calibration requires the 'ml' extra") from exc
    return exceptions


__all__ = [
    "CALIBRATION_OUTCOME_LABELS",
    "DYNAMIC_POOL_CALIBRATION_PREDICTION_SCHEMA",
    "DYNAMIC_POOL_CALIBRATION_PREDICTION_SCHEMA_VERSION",
    "DYNAMIC_POOL_CALIBRATION_VERSION",
    "DYNAMIC_POOL_EVIDENCE_MODEL_VERSION",
    "DynamicPoolCalibrationConfig",
    "DynamicPoolCalibrationFit",
    "FrozenDynamicPoolEvidenceModel",
    "fit_dynamic_pool_evidence_calibrator",
]
