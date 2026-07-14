"""Leakage-safe conventional classifiers over frozen BioCLIP features."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import logging
from math import isfinite
from pathlib import Path
from typing import Any

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.detection.routing import DETECTION_ROUTES
from biominer.ml.training_features import (
    LABEL_CERTAINTIES,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_MODEL_FEATURE_COLUMNS,
    TARGET_TASKS,
    load_few_shot_training_features,
    validate_few_shot_training_features,
)
from biominer.references.readiness import REFERENCE_ROUTES
from biominer.vision.full_frame_attention import (
    FOCUSED_FULL_FRAME_KIND,
    MASKED_FULL_FRAME_KIND,
    MULTI_OBJECT_FULL_FRAME_KIND,
    RAW_FULL_IMAGE_KIND,
)


CLASSIFIER_TRAINING_VERSION = "frozen-embedding-classifier-training-v1.0.0"
CLASSIFIER_SEARCH_GRID_VERSION = "frozen-embedding-search-grid-v1"
FEATURE_LAYOUT_VERSION = "frozen-embedding-feature-layout-v1"
QUALITY_FLAG_HASH_VERSION = "visual-quality-flag-sha256-buckets-v1"
CV_SPLIT_VERSION = "stratified-group-kfold-audit-v1"

LOGISTIC_REGRESSION_MODEL = "logistic_regression_embedding"
LINEAR_SVC_EMBEDDING_MODEL = "linear_svc_embedding"
LINEAR_SVC_STRUCTURED_MODEL = "linear_svc_embedding_structured"
RBF_SVC_PILOT_MODEL = "rbf_svc_embedding_pilot"

DEFAULT_CLASSIFIER_MODELS = (
    LOGISTIC_REGRESSION_MODEL,
    LINEAR_SVC_EMBEDDING_MODEL,
    LINEAR_SVC_STRUCTURED_MODEL,
)
CLASSIFIER_MODELS = frozenset((*DEFAULT_CLASSIFIER_MODELS, RBF_SVC_PILOT_MODEL))
MODEL_SELECTION_PREFERENCE = (
    LINEAR_SVC_EMBEDDING_MODEL,
    LINEAR_SVC_STRUCTURED_MODEL,
    LOGISTIC_REGRESSION_MODEL,
    RBF_SVC_PILOT_MODEL,
)

EMBEDDING_ONLY_FEATURE_SET = "embedding_only"
EMBEDDING_PLUS_STRUCTURED_FEATURE_SET = "embedding_plus_structured"
CLASSIFIER_FEATURE_SETS = frozenset(
    {EMBEDDING_ONLY_FEATURE_SET, EMBEDDING_PLUS_STRUCTURED_FEATURE_SET}
)

ESTIMATOR_LOGISTIC_REGRESSION = "logistic_regression"
ESTIMATOR_LINEAR_SVC = "linear_svc"
ESTIMATOR_RBF_SVC = "rbf_svc"

NON_TARGET_CLASS_LABEL = "__non_target__"
PRIMARY_CV_METRIC = "balanced_accuracy"
SECONDARY_CV_METRIC = "macro_f1"

REGULARIZATION_C_GRID = (0.01, 0.1, 1.0, 10.0)
RBF_C_GRID = (0.1, 1.0, 10.0)
RBF_GAMMA_GRID: tuple[str | float, ...] = ("scale", 0.01, 0.1)
DEFAULT_RBF_MAX_FIT_SAMPLES = 2_000
QUALITY_FLAG_HASH_BUCKET_COUNT = 32

_VISUAL_INPUT_KINDS = (
    RAW_FULL_IMAGE_KIND,
    FOCUSED_FULL_FRAME_KIND,
    MASKED_FULL_FRAME_KIND,
    MULTI_OBJECT_FULL_FRAME_KIND,
)
_STRUCTURED_INDICATOR_COLUMNS = (
    "missing_geo",
    "multiple_organism_indicator",
    "low_resolution_indicator",
)
_STRUCTURED_CONTINUOUS_COLUMNS = tuple(
    name
    for name in NUMERIC_MODEL_FEATURE_COLUMNS
    if name != "embedding" and name not in _STRUCTURED_INDICATOR_COLUMNS
)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ClassifierTrainingConfig:
    """Deterministic estimator-search policy for one task and visual route."""

    target_task: str
    target_accepted_taxon_key: str
    route: str
    n_splits: int = 3
    random_seed: int = 42
    class_weight: str | tuple[tuple[str, float], ...] | Mapping[str, float] | None = (
        "balanced"
    )
    included_label_certainties: tuple[str, ...] = ("high", "medium")
    enabled_models: tuple[str, ...] = DEFAULT_CLASSIFIER_MODELS
    enable_rbf_pilot: bool = False
    rbf_max_fit_samples: int = DEFAULT_RBF_MAX_FIT_SAMPLES
    n_jobs: int = 1

    def __post_init__(self) -> None:
        task = _required_choice(
            self.target_task,
            field="target_task",
            allowed=TARGET_TASKS,
        )
        target_key = _required_text(
            self.target_accepted_taxon_key,
            field="target_accepted_taxon_key",
        )
        route = _required_choice(self.route, field="route", allowed=REFERENCE_ROUTES)
        if task == "larval_target_verifier" and route != "larval":
            raise ValueError("larval target verifier requires route='larval'")
        n_splits = _integer_at_least(self.n_splits, minimum=2, field="n_splits")
        seed = _random_seed(self.random_seed)
        certainties = _sorted_unique_choices(
            self.included_label_certainties,
            field="included_label_certainties",
            allowed=LABEL_CERTAINTIES,
        )
        if not certainties:
            raise ValueError("included_label_certainties must not be empty")
        models = _unique_choices(
            self.enabled_models,
            field="enabled_models",
            allowed=frozenset(DEFAULT_CLASSIFIER_MODELS),
        )
        if not isinstance(self.enable_rbf_pilot, bool):
            raise TypeError("enable_rbf_pilot must be boolean")
        if not models and not self.enable_rbf_pilot:
            raise ValueError("at least one classifier model must be enabled")
        cap = _positive_integer(self.rbf_max_fit_samples, field="rbf_max_fit_samples")
        n_jobs = _positive_integer(self.n_jobs, field="n_jobs")
        weight = _normalized_class_weight(self.class_weight)
        object.__setattr__(self, "target_task", task)
        object.__setattr__(self, "target_accepted_taxon_key", target_key)
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "n_splits", n_splits)
        object.__setattr__(self, "random_seed", seed)
        object.__setattr__(self, "class_weight", weight)
        object.__setattr__(self, "included_label_certainties", certainties)
        object.__setattr__(self, "enabled_models", models)
        object.__setattr__(self, "rbf_max_fit_samples", cap)
        object.__setattr__(self, "n_jobs", n_jobs)

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": CLASSIFIER_TRAINING_VERSION,
                "search_grid_version": CLASSIFIER_SEARCH_GRID_VERSION,
                "target_task": self.target_task,
                "target_accepted_taxon_key": self.target_accepted_taxon_key,
                "route": self.route,
                "n_splits": self.n_splits,
                "random_seed": self.random_seed,
                "class_weight": _class_weight_semantic_value(self.class_weight),
                "included_label_certainties": list(self.included_label_certainties),
                "enabled_models": list(self.enabled_models),
                "enable_rbf_pilot": self.enable_rbf_pilot,
                "rbf_max_fit_samples": self.rbf_max_fit_samples,
                "n_jobs": self.n_jobs,
                "primary_cv_metric": PRIMARY_CV_METRIC,
                "secondary_cv_metric": SECONDARY_CV_METRIC,
                "model_selection_preference": list(MODEL_SELECTION_PREFERENCE),
            }
        )

    def sklearn_class_weight(self) -> str | dict[str, float] | None:
        if isinstance(self.class_weight, tuple):
            return dict(self.class_weight)
        return self.class_weight


@dataclass(frozen=True, slots=True)
class ClassifierFeatureLayout:
    """Exact numeric input and transformed feature order for one feature set."""

    feature_set: str
    embedding_dimension: int
    source_feature_names: tuple[str, ...]
    raw_feature_names: tuple[str, ...]
    transformed_feature_names: tuple[str, ...]
    embedding_column_indices: tuple[int, ...]
    continuous_column_indices: tuple[int, ...]
    indicator_column_indices: tuple[int, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class GroupFoldAudit:
    fold_index: int
    train_sample_count: int
    validation_sample_count: int
    train_group_ids: tuple[str, ...]
    validation_group_ids: tuple[str, ...]
    train_class_sample_counts: tuple[tuple[str, int], ...]
    validation_class_sample_counts: tuple[tuple[str, int], ...]
    train_class_group_counts: tuple[tuple[str, int], ...]
    validation_class_group_counts: tuple[tuple[str, int], ...]
    fold_fingerprint: str


@dataclass(frozen=True, slots=True)
class HyperparameterScore:
    parameters: tuple[tuple[str, object], ...]
    mean_balanced_accuracy: float
    std_balanced_accuracy: float
    mean_macro_f1: float
    std_macro_f1: float
    balanced_accuracy_rank: int


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    sample_count: int
    class_sample_counts: tuple[tuple[str, int], ...]
    balanced_accuracy: float
    macro_f1: float


@dataclass(frozen=True, slots=True)
class ClassifierCandidateResult:
    """One refitted in-memory estimator plus non-executable audit metadata."""

    model_name: str
    estimator_family: str
    feature_set: str
    feature_layout: ClassifierFeatureLayout
    estimator_configuration: tuple[tuple[str, object], ...]
    parameter_grid: tuple[tuple[str, tuple[object, ...]], ...]
    selected_parameters: tuple[tuple[str, object], ...]
    grid_scores: tuple[HyperparameterScore, ...]
    best_cv_balanced_accuracy: float
    best_cv_macro_f1: float
    model_selection_metrics: ClassificationMetrics | None
    fit_partition_fingerprint: str
    model_selection_partition_fingerprint: str | None
    cv_split_fingerprint: str
    probability_calibrated: bool
    candidate_fingerprint: str
    pipeline: Any = field(repr=False, compare=False)

    @property
    def selected_parameter_dict(self) -> dict[str, object]:
        return dict(self.selected_parameters)


@dataclass(frozen=True, slots=True)
class ClassifierTrainingRun:
    """Audited comparison for one task without calibration or persistence."""

    training_version: str
    configuration_fingerprint: str
    target_task: str
    target_accepted_taxon_key: str
    route: str
    class_labels: tuple[str, ...]
    fit_sample_count: int
    fit_group_count: int
    fit_class_sample_counts: tuple[tuple[str, int], ...]
    fit_class_group_counts: tuple[tuple[str, int], ...]
    model_selection_sample_count: int
    calibration_sample_count: int
    final_test_sample_count: int
    folds: tuple[GroupFoldAudit, ...]
    cv_split_fingerprint: str
    feature_schema_fingerprint: str
    training_data_fingerprint: str
    fit_partition_fingerprint: str
    model_selection_partition_fingerprint: str | None
    model_fingerprint: str
    support_manifest_fingerprint: str
    reference_embedding_fingerprint: str
    reference_prototype_fingerprint: str
    candidate_set_fingerprint: str
    numpy_version: str
    scikit_learn_version: str
    foundation_model_trainable: bool
    candidates: tuple[ClassifierCandidateResult, ...]
    selected_model_name: str
    training_run_fingerprint: str

    @property
    def selected_candidate(self) -> ClassifierCandidateResult:
        return next(
            item
            for item in self.candidates
            if item.model_name == self.selected_model_name
        )


@dataclass(frozen=True, slots=True)
class _ModelSpec:
    model_name: str
    estimator_family: str
    feature_set: str
    parameter_grid: tuple[tuple[str, tuple[object, ...]], ...]


@dataclass(frozen=True, slots=True)
class _MLDependencies:
    np: Any
    sklearn_version: str
    ColumnTransformer: Any
    SimpleImputer: Any
    LogisticRegression: Any
    GridSearchCV: Any
    StratifiedGroupKFold: Any
    Pipeline: Any
    StandardScaler: Any
    LinearSVC: Any
    SVC: Any
    balanced_accuracy_score: Any
    f1_score: Any
    config_context: Any


def train_frozen_embedding_classifiers(
    training_features: pl.DataFrame | str | Path,
    config: ClassifierTrainingConfig,
) -> ClassifierTrainingRun:
    """Search and refit conventional estimators without touching BioCLIP weights."""

    if not isinstance(config, ClassifierTrainingConfig):
        raise TypeError("config must be a ClassifierTrainingConfig")
    frame = _training_feature_frame(training_features)
    eligible = _eligible_task_rows(frame, config)
    fit_frame = eligible.filter(pl.col("dataset_split") == "support_train")
    selection_frame = eligible.filter(pl.col("dataset_split") == "model_selection")
    if fit_frame.is_empty():
        raise ValueError("classifier training requires eligible support_train rows")

    fit_labels = _labels(fit_frame, config)
    class_labels = tuple(sorted(set(fit_labels)))
    _validate_required_classes(class_labels, config)
    groups = tuple(str(value) for value in fit_frame["leakage_group_id"].to_list())
    _validate_group_labels(groups, fit_labels)
    class_group_counts = _class_group_counts(groups, fit_labels, class_labels)
    insufficient = {
        label: count for label, count in class_group_counts if count < config.n_splits
    }
    if insufficient:
        raise ValueError(
            "groups per class cannot support "
            f"n_splits={config.n_splits}: {sorted(insufficient.items())}"
        )
    _validate_class_weight_labels(config.class_weight, class_labels)

    selection_labels = (
        _labels(selection_frame, config) if selection_frame.height else ()
    )
    if selection_labels and set(selection_labels) != set(class_labels):
        raise ValueError(
            "model_selection rows must contain every fitted class and no unseen class"
        )
    if selection_labels:
        selection_groups = tuple(
            str(value) for value in selection_frame["leakage_group_id"].to_list()
        )
        _validate_group_labels(selection_groups, selection_labels)

    dependencies = _load_ml_dependencies()
    model_specs = _model_specs(config)
    if config.enable_rbf_pilot and fit_frame.height > config.rbf_max_fit_samples:
        raise ValueError(
            "RBF pilot sample cap exceeded: "
            f"{fit_frame.height}/{config.rbf_max_fit_samples}"
        )

    embedding_dimension = int(fit_frame["embedding_dimension"][0])
    layouts = {
        feature_set: _feature_layout(feature_set, embedding_dimension)
        for feature_set in {item.feature_set for item in model_specs}
    }
    fit_matrices = {
        feature_set: _feature_matrix(
            fit_frame,
            layout=layouts[feature_set],
            np=dependencies.np,
        )
        for feature_set in layouts
    }
    selection_matrices = {
        feature_set: _feature_matrix(
            selection_frame,
            layout=layouts[feature_set],
            np=dependencies.np,
        )
        for feature_set in layouts
        if selection_frame.height
    }
    y = dependencies.np.asarray(fit_labels, dtype=str)
    group_array = dependencies.np.asarray(groups, dtype=str)
    audit_splitter = dependencies.StratifiedGroupKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_seed,
    )
    representative_matrix = fit_matrices[model_specs[0].feature_set]
    audited_indices = tuple(
        (
            dependencies.np.asarray(train_indices, dtype=int),
            dependencies.np.asarray(validation_indices, dtype=int),
        )
        for train_indices, validation_indices in audit_splitter.split(
            representative_matrix,
            y,
            groups=group_array,
        )
    )
    folds = _audit_group_folds(
        audited_indices,
        labels=fit_labels,
        groups=groups,
        class_labels=class_labels,
    )
    cv_split_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": CV_SPLIT_VERSION,
            "configuration_fingerprint": config.fingerprint,
            "fold_fingerprints": [item.fold_fingerprint for item in folds],
        }
    )

    _log_event(
        "frozen_classifier_training_start",
        target_task=config.target_task,
        target_accepted_taxon_key=config.target_accepted_taxon_key,
        route=config.route,
        fit_sample_count=fit_frame.height,
        fit_group_count=len(set(groups)),
        class_count=len(class_labels),
        candidate_count=len(model_specs),
        n_splits=config.n_splits,
        configuration_fingerprint=config.fingerprint,
        cv_split_fingerprint=cv_split_fingerprint,
    )

    fit_partition_fingerprint = _partition_fingerprint(
        fit_frame,
        split="support_train",
        config=config,
    )
    selection_partition_fingerprint = (
        _partition_fingerprint(
            selection_frame,
            split="model_selection",
            config=config,
        )
        if selection_frame.height
        else None
    )
    candidates = tuple(
        _fit_model_candidate(
            spec=spec,
            config=config,
            dependencies=dependencies,
            layout=layouts[spec.feature_set],
            fit_matrix=fit_matrices[spec.feature_set],
            labels=y,
            groups=group_array,
            class_labels=class_labels,
            selection_matrix=selection_matrices.get(spec.feature_set),
            selection_labels=selection_labels,
            fit_partition_fingerprint=fit_partition_fingerprint,
            model_selection_partition_fingerprint=(selection_partition_fingerprint),
            cv_split_fingerprint=cv_split_fingerprint,
        )
        for spec in model_specs
    )
    selected_model_name = _select_model(candidates)
    training_data_fingerprint = _single_value(frame, "training_data_fingerprint")
    feature_schema_fingerprint = _single_value(frame, "feature_schema_fingerprint")
    consumed_rows = eligible.filter(
        pl.col("dataset_split").is_in(("support_train", "model_selection"))
    )
    candidate_set_fingerprint = _candidate_set_fingerprint(consumed_rows)
    model_fingerprint = _single_value(frame, "model_fingerprint")
    support_manifest_fingerprint = _single_value(
        frame,
        "support_manifest_fingerprint",
    )
    reference_embedding_fingerprint = _single_value(
        frame,
        "reference_embedding_fingerprint",
    )
    reference_prototype_fingerprint = _single_value(
        frame,
        "reference_prototype_fingerprint",
    )
    run_semantics = {
        "schema_version": CLASSIFIER_TRAINING_VERSION,
        "configuration_fingerprint": config.fingerprint,
        "target_task": config.target_task,
        "target_accepted_taxon_key": config.target_accepted_taxon_key,
        "route": config.route,
        "class_labels": list(class_labels),
        "fit_partition_fingerprint": fit_partition_fingerprint,
        "model_selection_partition_fingerprint": selection_partition_fingerprint,
        "cv_split_fingerprint": cv_split_fingerprint,
        "candidate_fingerprints": [item.candidate_fingerprint for item in candidates],
        "selected_model_name": selected_model_name,
        "feature_schema_fingerprint": feature_schema_fingerprint,
        "training_data_fingerprint": training_data_fingerprint,
        "model_fingerprint": model_fingerprint,
        "support_manifest_fingerprint": support_manifest_fingerprint,
        "reference_embedding_fingerprint": reference_embedding_fingerprint,
        "reference_prototype_fingerprint": reference_prototype_fingerprint,
        "candidate_set_fingerprint": candidate_set_fingerprint,
        "numpy_version": dependencies.np.__version__,
        "scikit_learn_version": dependencies.sklearn_version,
        "foundation_model_trainable": False,
    }
    training_run_fingerprint = canonical_semantic_fingerprint(run_semantics)
    result = ClassifierTrainingRun(
        training_version=CLASSIFIER_TRAINING_VERSION,
        configuration_fingerprint=config.fingerprint,
        target_task=config.target_task,
        target_accepted_taxon_key=config.target_accepted_taxon_key,
        route=config.route,
        class_labels=class_labels,
        fit_sample_count=fit_frame.height,
        fit_group_count=len(set(groups)),
        fit_class_sample_counts=_class_sample_counts(fit_labels, class_labels),
        fit_class_group_counts=class_group_counts,
        model_selection_sample_count=selection_frame.height,
        calibration_sample_count=eligible.filter(
            pl.col("dataset_split") == "calibration"
        ).height,
        final_test_sample_count=eligible.filter(
            pl.col("dataset_split") == "final_test"
        ).height,
        folds=folds,
        cv_split_fingerprint=cv_split_fingerprint,
        feature_schema_fingerprint=feature_schema_fingerprint,
        training_data_fingerprint=training_data_fingerprint,
        fit_partition_fingerprint=fit_partition_fingerprint,
        model_selection_partition_fingerprint=selection_partition_fingerprint,
        model_fingerprint=model_fingerprint,
        support_manifest_fingerprint=support_manifest_fingerprint,
        reference_embedding_fingerprint=reference_embedding_fingerprint,
        reference_prototype_fingerprint=reference_prototype_fingerprint,
        candidate_set_fingerprint=candidate_set_fingerprint,
        numpy_version=dependencies.np.__version__,
        scikit_learn_version=dependencies.sklearn_version,
        foundation_model_trainable=False,
        candidates=candidates,
        selected_model_name=selected_model_name,
        training_run_fingerprint=training_run_fingerprint,
    )
    _log_event(
        "frozen_classifier_training_complete",
        target_task=config.target_task,
        route=config.route,
        fit_sample_count=result.fit_sample_count,
        model_selection_sample_count=result.model_selection_sample_count,
        candidate_count=len(result.candidates),
        selected_model_name=result.selected_model_name,
        training_run_fingerprint=result.training_run_fingerprint,
    )
    return result


def _training_feature_frame(
    source: pl.DataFrame | str | Path,
) -> pl.DataFrame:
    if isinstance(source, pl.DataFrame):
        validate_few_shot_training_features(source)
        return source
    return load_few_shot_training_features(source)


def _eligible_task_rows(
    frame: pl.DataFrame,
    config: ClassifierTrainingConfig,
) -> pl.DataFrame:
    result = frame.filter(
        (pl.col("target_task") == config.target_task)
        & (pl.col("target_accepted_taxon_key") == config.target_accepted_taxon_key)
        & (pl.col("route") == config.route)
        & pl.col("label_certainty").is_in(config.included_label_certainties)
    )
    if config.target_task == "regional_multiclass":
        result = result.filter(pl.col("species_training_suitable"))
    if result.is_empty():
        raise ValueError("no eligible training rows match the classifier configuration")
    return result


def _labels(
    frame: pl.DataFrame,
    config: ClassifierTrainingConfig,
) -> tuple[str, ...]:
    labels: list[str] = []
    for row in frame.iter_rows(named=True):
        if config.target_task in {
            "binary_target_verifier",
            "larval_target_verifier",
        }:
            labels.append(
                config.target_accepted_taxon_key
                if bool(row["target_present"])
                else NON_TARGET_CLASS_LABEL
            )
        elif config.target_task == "regional_multiclass":
            value = row["accepted_class_taxon_key"]
            labels.append(_required_text(value, field="accepted_class_taxon_key"))
        elif config.target_task == "visual_domain":
            labels.append(
                _required_text(row["visual_domain_label"], field="visual_domain_label")
            )
        else:  # pragma: no cover - config validation owns this invariant.
            raise AssertionError(f"unsupported target task: {config.target_task}")
    return tuple(labels)


def _validate_required_classes(
    class_labels: Sequence[str],
    config: ClassifierTrainingConfig,
) -> None:
    if len(class_labels) < 2:
        raise ValueError("classifier fitting requires at least two classes")
    if config.target_task in {
        "binary_target_verifier",
        "larval_target_verifier",
    }:
        expected = {NON_TARGET_CLASS_LABEL, config.target_accepted_taxon_key}
        if set(class_labels) != expected:
            raise ValueError("target verifier requires both target and non-target rows")
    if (
        config.target_task == "regional_multiclass"
        and config.target_accepted_taxon_key not in class_labels
    ):
        raise ValueError("regional multiclass fitting requires target support")


def _validate_group_labels(groups: Sequence[str], labels: Sequence[str]) -> None:
    label_by_group: dict[str, str] = {}
    for group, label in zip(groups, labels, strict=True):
        previous = label_by_group.setdefault(group, label)
        if previous != label:
            raise ValueError("leakage group maps to multiple labels")


def _class_sample_counts(
    labels: Sequence[str],
    class_labels: Sequence[str],
) -> tuple[tuple[str, int], ...]:
    counts = Counter(labels)
    return tuple((label, int(counts[label])) for label in class_labels)


def _class_group_counts(
    groups: Sequence[str],
    labels: Sequence[str],
    class_labels: Sequence[str],
) -> tuple[tuple[str, int], ...]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for group, label in zip(groups, labels, strict=True):
        grouped[label].add(group)
    return tuple((label, len(grouped[label])) for label in class_labels)


def _audit_group_folds(
    splits: Sequence[tuple[Any, Any]],
    *,
    labels: Sequence[str],
    groups: Sequence[str],
    class_labels: Sequence[str],
) -> tuple[GroupFoldAudit, ...]:
    expected_classes = set(class_labels)
    all_groups = set(groups)
    validation_group_occurrences: Counter[str] = Counter()
    result: list[GroupFoldAudit] = []
    for fold_index, (train_indices, validation_indices) in enumerate(splits):
        train_ids = tuple(int(index) for index in train_indices)
        validation_ids = tuple(int(index) for index in validation_indices)
        train_groups = tuple(sorted({groups[index] for index in train_ids}))
        validation_groups = tuple(sorted({groups[index] for index in validation_ids}))
        if set(train_groups).intersection(validation_groups):
            raise ValueError("group-aware fold leaks a group across train/validation")
        if set(train_groups).union(validation_groups) != all_groups:
            raise ValueError("group-aware fold does not cover every fitting group")
        train_labels = tuple(labels[index] for index in train_ids)
        validation_labels = tuple(labels[index] for index in validation_ids)
        if set(train_labels) != expected_classes:
            raise ValueError("group-aware fold train partition omits a class")
        if set(validation_labels) != expected_classes:
            raise ValueError("group-aware fold validation partition omits a class")
        validation_group_occurrences.update(validation_groups)
        train_group_labels = _labels_for_groups(train_groups, groups, labels)
        validation_group_labels = _labels_for_groups(
            validation_groups,
            groups,
            labels,
        )
        semantic = {
            "schema_version": CV_SPLIT_VERSION,
            "fold_index": fold_index,
            "train_indices": list(train_ids),
            "validation_indices": list(validation_ids),
            "train_group_ids": list(train_groups),
            "validation_group_ids": list(validation_groups),
            "train_class_sample_counts": list(
                _class_sample_counts(train_labels, class_labels)
            ),
            "validation_class_sample_counts": list(
                _class_sample_counts(validation_labels, class_labels)
            ),
            "train_class_group_counts": list(
                _class_sample_counts(train_group_labels, class_labels)
            ),
            "validation_class_group_counts": list(
                _class_sample_counts(validation_group_labels, class_labels)
            ),
        }
        result.append(
            GroupFoldAudit(
                fold_index=fold_index,
                train_sample_count=len(train_ids),
                validation_sample_count=len(validation_ids),
                train_group_ids=train_groups,
                validation_group_ids=validation_groups,
                train_class_sample_counts=tuple(
                    tuple(item) for item in semantic["train_class_sample_counts"]
                ),
                validation_class_sample_counts=tuple(
                    tuple(item) for item in semantic["validation_class_sample_counts"]
                ),
                train_class_group_counts=tuple(
                    tuple(item) for item in semantic["train_class_group_counts"]
                ),
                validation_class_group_counts=tuple(
                    tuple(item) for item in semantic["validation_class_group_counts"]
                ),
                fold_fingerprint=canonical_semantic_fingerprint(semantic),
            )
        )
    if set(validation_group_occurrences) != all_groups or any(
        count != 1 for count in validation_group_occurrences.values()
    ):
        raise ValueError("group-aware folds must validate every group exactly once")
    return tuple(result)


def _labels_for_groups(
    selected_groups: Sequence[str],
    groups: Sequence[str],
    labels: Sequence[str],
) -> tuple[str, ...]:
    label_by_group: dict[str, str] = {}
    for group, label in zip(groups, labels, strict=True):
        previous = label_by_group.setdefault(group, label)
        if previous != label:
            raise ValueError("leakage group maps to multiple labels")
    return tuple(label_by_group[group] for group in selected_groups)


def _feature_layout(
    feature_set: str,
    embedding_dimension: int,
) -> ClassifierFeatureLayout:
    if feature_set not in CLASSIFIER_FEATURE_SETS:
        raise ValueError(f"unsupported classifier feature set: {feature_set}")
    dimension = _positive_integer(embedding_dimension, field="embedding_dimension")
    embedding_names = tuple(f"embedding_{index:05d}" for index in range(dimension))
    if feature_set == EMBEDDING_ONLY_FEATURE_SET:
        source_names = ("embedding",)
        raw_names = embedding_names
        continuous_names: tuple[str, ...] = ()
        indicator_names: tuple[str, ...] = ()
    else:
        source_names = tuple(MODEL_FEATURE_COLUMNS)
        continuous_names = _STRUCTURED_CONTINUOUS_COLUMNS
        indicator_names = (
            *_STRUCTURED_INDICATOR_COLUMNS,
            *(f"route={value}" for value in sorted(REFERENCE_ROUTES)),
            *(f"visual_input_kind={value}" for value in sorted(_VISUAL_INPUT_KINDS)),
            *(f"yoloe_route={value}" for value in sorted(DETECTION_ROUTES)),
            *(
                f"visual_input_quality_flag_hash_{index:02d}"
                for index in range(QUALITY_FLAG_HASH_BUCKET_COUNT)
            ),
        )
        raw_names = (*embedding_names, *continuous_names, *indicator_names)
    embedding_indices = tuple(range(dimension))
    continuous_indices = tuple(range(dimension, dimension + len(continuous_names)))
    indicator_start = dimension + len(continuous_names)
    indicator_indices = tuple(
        range(indicator_start, indicator_start + len(indicator_names))
    )
    semantic = {
        "schema_version": FEATURE_LAYOUT_VERSION,
        "feature_set": feature_set,
        "embedding_dimension": dimension,
        "source_feature_names": list(source_names),
        "raw_feature_names": list(raw_names),
        "transformed_feature_names": list(raw_names),
        "embedding_column_indices": list(embedding_indices),
        "continuous_column_indices": list(continuous_indices),
        "indicator_column_indices": list(indicator_indices),
        "quality_flag_hash_version": (
            QUALITY_FLAG_HASH_VERSION
            if feature_set == EMBEDDING_PLUS_STRUCTURED_FEATURE_SET
            else None
        ),
        "quality_flag_hash_bucket_count": (
            QUALITY_FLAG_HASH_BUCKET_COUNT
            if feature_set == EMBEDDING_PLUS_STRUCTURED_FEATURE_SET
            else None
        ),
    }
    return ClassifierFeatureLayout(
        feature_set=feature_set,
        embedding_dimension=dimension,
        source_feature_names=source_names,
        raw_feature_names=tuple(raw_names),
        transformed_feature_names=tuple(raw_names),
        embedding_column_indices=embedding_indices,
        continuous_column_indices=continuous_indices,
        indicator_column_indices=indicator_indices,
        fingerprint=canonical_semantic_fingerprint(semantic),
    )


def classifier_feature_layout(
    feature_set: str,
    embedding_dimension: int,
) -> ClassifierFeatureLayout:
    """Return the versioned, exact numeric layout for classifier inputs."""

    return _feature_layout(feature_set, embedding_dimension)


def materialize_classifier_feature_matrix(
    frame: pl.DataFrame,
    layout: ClassifierFeatureLayout,
) -> Any:
    """Materialize raw float64 features without importing scikit-learn."""

    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a polars DataFrame")
    if not isinstance(layout, ClassifierFeatureLayout):
        raise TypeError("layout must be a ClassifierFeatureLayout")
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency guard.
        raise RuntimeError(
            "classifier feature materialization requires the 'ml' dependency group"
        ) from exc
    return _feature_matrix(frame, layout=layout, np=np)


def _feature_matrix(
    frame: pl.DataFrame,
    *,
    layout: ClassifierFeatureLayout,
    np: Any,
) -> Any:
    rows: list[list[float]] = []
    for row in frame.iter_rows(named=True):
        values = [float(value) for value in row["embedding"]]
        if len(values) != layout.embedding_dimension:
            raise ValueError("training row embedding dimension is inconsistent")
        if layout.feature_set == EMBEDDING_PLUS_STRUCTURED_FEATURE_SET:
            for field_name in _STRUCTURED_CONTINUOUS_COLUMNS:
                value = row[field_name]
                values.append(float(value) if value is not None else float("nan"))
            values.extend(
                1.0 if bool(row[field_name]) else 0.0
                for field_name in _STRUCTURED_INDICATOR_COLUMNS
            )
            for category in sorted(REFERENCE_ROUTES):
                values.append(1.0 if row["route"] == category else 0.0)
            for category in sorted(_VISUAL_INPUT_KINDS):
                values.append(1.0 if row["visual_input_kind"] == category else 0.0)
            for category in sorted(DETECTION_ROUTES):
                values.append(1.0 if row["yoloe_route"] == category else 0.0)
            quality_buckets = [0.0] * QUALITY_FLAG_HASH_BUCKET_COUNT
            for flag in row["visual_input_quality_flags"]:
                quality_buckets[_quality_flag_bucket(str(flag))] = 1.0
            values.extend(quality_buckets)
        if len(values) != len(layout.raw_feature_names):
            raise AssertionError("classifier feature matrix width mismatch")
        rows.append(values)
    if not rows:
        return np.empty((0, len(layout.raw_feature_names)), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _quality_flag_bucket(value: str) -> int:
    digest = hashlib.sha256(
        (QUALITY_FLAG_HASH_VERSION + "\0" + value).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % QUALITY_FLAG_HASH_BUCKET_COUNT


def _model_specs(config: ClassifierTrainingConfig) -> tuple[_ModelSpec, ...]:
    result: list[_ModelSpec] = []
    for model_name in config.enabled_models:
        if model_name == LOGISTIC_REGRESSION_MODEL:
            result.append(
                _ModelSpec(
                    model_name=model_name,
                    estimator_family=ESTIMATOR_LOGISTIC_REGRESSION,
                    feature_set=EMBEDDING_ONLY_FEATURE_SET,
                    parameter_grid=(("classifier__C", REGULARIZATION_C_GRID),),
                )
            )
        elif model_name == LINEAR_SVC_EMBEDDING_MODEL:
            result.append(
                _ModelSpec(
                    model_name=model_name,
                    estimator_family=ESTIMATOR_LINEAR_SVC,
                    feature_set=EMBEDDING_ONLY_FEATURE_SET,
                    parameter_grid=(("classifier__C", REGULARIZATION_C_GRID),),
                )
            )
        elif model_name == LINEAR_SVC_STRUCTURED_MODEL:
            result.append(
                _ModelSpec(
                    model_name=model_name,
                    estimator_family=ESTIMATOR_LINEAR_SVC,
                    feature_set=EMBEDDING_PLUS_STRUCTURED_FEATURE_SET,
                    parameter_grid=(("classifier__C", REGULARIZATION_C_GRID),),
                )
            )
        else:  # pragma: no cover - config validation owns this invariant.
            raise AssertionError(f"unsupported model name: {model_name}")
    if config.enable_rbf_pilot:
        result.append(
            _ModelSpec(
                model_name=RBF_SVC_PILOT_MODEL,
                estimator_family=ESTIMATOR_RBF_SVC,
                feature_set=EMBEDDING_ONLY_FEATURE_SET,
                parameter_grid=(
                    ("classifier__C", RBF_C_GRID),
                    ("classifier__gamma", RBF_GAMMA_GRID),
                ),
            )
        )
    return tuple(result)


def _fit_model_candidate(
    *,
    spec: _ModelSpec,
    config: ClassifierTrainingConfig,
    dependencies: _MLDependencies,
    layout: ClassifierFeatureLayout,
    fit_matrix: Any,
    labels: Any,
    groups: Any,
    class_labels: tuple[str, ...],
    selection_matrix: Any | None,
    selection_labels: Sequence[str],
    fit_partition_fingerprint: str,
    model_selection_partition_fingerprint: str | None,
    cv_split_fingerprint: str,
) -> ClassifierCandidateResult:
    pipeline = _classifier_pipeline(
        spec,
        config=config,
        dependencies=dependencies,
        layout=layout,
    )
    search_splitter = dependencies.StratifiedGroupKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_seed,
    )
    search = dependencies.GridSearchCV(
        estimator=pipeline,
        param_grid={name: list(values) for name, values in spec.parameter_grid},
        scoring={
            PRIMARY_CV_METRIC: PRIMARY_CV_METRIC,
            SECONDARY_CV_METRIC: "f1_macro",
        },
        refit=PRIMARY_CV_METRIC,
        cv=search_splitter,
        n_jobs=config.n_jobs,
        return_train_score=False,
        error_score="raise",
    )
    with dependencies.config_context(enable_metadata_routing=False):
        search.fit(fit_matrix, labels, groups=groups)
    if int(search.n_splits_) != config.n_splits:
        raise ValueError("grid search used an unexpected number of folds")
    fitted_pipeline = search.best_estimator_
    fitted_classes = tuple(
        str(value)
        for value in fitted_pipeline.named_steps["classifier"].classes_.tolist()
    )
    if fitted_classes != class_labels:
        raise ValueError("fitted estimator class order does not match training labels")
    if layout.feature_set == EMBEDDING_PLUS_STRUCTURED_FEATURE_SET:
        transformed_width = int(
            fitted_pipeline.named_steps["features"].transform(fit_matrix[:1]).shape[1]
        )
        if transformed_width != len(layout.transformed_feature_names):
            raise ValueError("fitted structured feature width is inconsistent")

    grid_scores = _grid_scores(search.cv_results_)
    best_index = int(search.best_index_)
    best_cv_balanced_accuracy = _finite_metric(
        search.cv_results_[f"mean_test_{PRIMARY_CV_METRIC}"][best_index],
        field="best_cv_balanced_accuracy",
    )
    best_cv_macro_f1 = _finite_metric(
        search.cv_results_[f"mean_test_{SECONDARY_CV_METRIC}"][best_index],
        field="best_cv_macro_f1",
    )
    selection_metrics = None
    if selection_matrix is not None:
        predictions = tuple(
            str(value) for value in fitted_pipeline.predict(selection_matrix)
        )
        selection_metrics = _classification_metrics(
            expected=selection_labels,
            predicted=predictions,
            class_labels=class_labels,
            dependencies=dependencies,
        )
    selected_parameters = _parameter_items(search.best_params_)
    estimator_configuration = _estimator_configuration(spec, config)
    candidate_semantics = {
        "schema_version": CLASSIFIER_TRAINING_VERSION,
        "search_grid_version": CLASSIFIER_SEARCH_GRID_VERSION,
        "configuration_fingerprint": config.fingerprint,
        "model_name": spec.model_name,
        "estimator_family": spec.estimator_family,
        "feature_set": spec.feature_set,
        "feature_layout_fingerprint": layout.fingerprint,
        "class_labels": list(class_labels),
        "fit_partition_fingerprint": fit_partition_fingerprint,
        "model_selection_partition_fingerprint": (
            model_selection_partition_fingerprint
        ),
        "cv_split_fingerprint": cv_split_fingerprint,
        "estimator_configuration": [
            [name, value] for name, value in estimator_configuration
        ],
        "parameter_grid": [
            [name, list(values)] for name, values in spec.parameter_grid
        ],
        "selected_parameters": [list(item) for item in selected_parameters],
        "grid_scores": [
            {
                "parameters": [list(item) for item in score.parameters],
                "mean_balanced_accuracy": score.mean_balanced_accuracy,
                "std_balanced_accuracy": score.std_balanced_accuracy,
                "mean_macro_f1": score.mean_macro_f1,
                "std_macro_f1": score.std_macro_f1,
                "balanced_accuracy_rank": score.balanced_accuracy_rank,
            }
            for score in grid_scores
        ],
        "best_cv_balanced_accuracy": best_cv_balanced_accuracy,
        "best_cv_macro_f1": best_cv_macro_f1,
        "probability_calibrated": False,
        "model_selection_metrics": (
            {
                "sample_count": selection_metrics.sample_count,
                "class_sample_counts": list(selection_metrics.class_sample_counts),
                "balanced_accuracy": selection_metrics.balanced_accuracy,
                "macro_f1": selection_metrics.macro_f1,
            }
            if selection_metrics is not None
            else None
        ),
    }
    return ClassifierCandidateResult(
        model_name=spec.model_name,
        estimator_family=spec.estimator_family,
        feature_set=spec.feature_set,
        feature_layout=layout,
        estimator_configuration=estimator_configuration,
        parameter_grid=spec.parameter_grid,
        selected_parameters=selected_parameters,
        grid_scores=grid_scores,
        best_cv_balanced_accuracy=best_cv_balanced_accuracy,
        best_cv_macro_f1=best_cv_macro_f1,
        model_selection_metrics=selection_metrics,
        fit_partition_fingerprint=fit_partition_fingerprint,
        model_selection_partition_fingerprint=(model_selection_partition_fingerprint),
        cv_split_fingerprint=cv_split_fingerprint,
        probability_calibrated=False,
        candidate_fingerprint=canonical_semantic_fingerprint(candidate_semantics),
        pipeline=fitted_pipeline,
    )


def _classifier_pipeline(
    spec: _ModelSpec,
    *,
    config: ClassifierTrainingConfig,
    dependencies: _MLDependencies,
    layout: ClassifierFeatureLayout,
) -> Any:
    class_weight = config.sklearn_class_weight()
    if spec.estimator_family == ESTIMATOR_LOGISTIC_REGRESSION:
        classifier = dependencies.LogisticRegression(
            solver="lbfgs",
            class_weight=class_weight,
            random_state=config.random_seed,
            max_iter=10_000,
        )
    elif spec.estimator_family == ESTIMATOR_LINEAR_SVC:
        classifier = dependencies.LinearSVC(
            penalty="l2",
            loss="squared_hinge",
            dual="auto",
            class_weight=class_weight,
            random_state=config.random_seed,
            max_iter=10_000,
        )
    elif spec.estimator_family == ESTIMATOR_RBF_SVC:
        classifier = dependencies.SVC(
            kernel="rbf",
            class_weight=class_weight,
            decision_function_shape="ovr",
            random_state=config.random_seed,
            cache_size=512.0,
        )
    else:  # pragma: no cover - model specs own this invariant.
        raise AssertionError(f"unsupported estimator family: {spec.estimator_family}")

    steps: list[tuple[str, object]] = []
    if layout.feature_set == EMBEDDING_PLUS_STRUCTURED_FEATURE_SET:
        continuous_pipeline = dependencies.Pipeline(
            [
                (
                    "imputer",
                    dependencies.SimpleImputer(
                        strategy="median",
                        keep_empty_features=True,
                    ),
                ),
                ("scaler", dependencies.StandardScaler()),
            ]
        )
        preprocessor = dependencies.ColumnTransformer(
            [
                ("embedding", "passthrough", list(layout.embedding_column_indices)),
                (
                    "continuous",
                    continuous_pipeline,
                    list(layout.continuous_column_indices),
                ),
                ("indicator", "passthrough", list(layout.indicator_column_indices)),
            ],
            remainder="drop",
            sparse_threshold=0.0,
            verbose_feature_names_out=False,
        )
        steps.append(("features", preprocessor))
    steps.append(("classifier", classifier))
    return dependencies.Pipeline(steps)


def _estimator_configuration(
    spec: _ModelSpec,
    config: ClassifierTrainingConfig,
) -> tuple[tuple[str, object], ...]:
    common: dict[str, object] = {
        "class_weight": _class_weight_semantic_value(config.class_weight),
        "probability_calibrated": False,
        "random_seed": config.random_seed,
    }
    if spec.estimator_family == ESTIMATOR_LOGISTIC_REGRESSION:
        values = {
            **common,
            "effective_penalty": "l2",
            "fit_intercept": True,
            "l1_ratio": 0.0,
            "max_iter": 10_000,
            "solver": "lbfgs",
            "tol": 0.0001,
        }
    elif spec.estimator_family == ESTIMATOR_LINEAR_SVC:
        values = {
            **common,
            "dual": "auto",
            "fit_intercept": True,
            "loss": "squared_hinge",
            "max_iter": 10_000,
            "penalty": "l2",
            "tol": 0.0001,
        }
    elif spec.estimator_family == ESTIMATOR_RBF_SVC:
        values = {
            **common,
            "cache_size_mb": 512.0,
            "decision_function_shape": "ovr",
            "kernel": "rbf",
            "max_iter": -1,
            "probability_output_enabled": False,
            "shrinking": True,
            "tol": 0.001,
        }
    else:  # pragma: no cover - model specs own this invariant.
        raise AssertionError(f"unsupported estimator family: {spec.estimator_family}")
    return tuple(sorted(values.items()))


def _grid_scores(results: Mapping[str, object]) -> tuple[HyperparameterScore, ...]:
    parameters = results["params"]
    output: list[HyperparameterScore] = []
    for index, params in enumerate(parameters):
        output.append(
            HyperparameterScore(
                parameters=_parameter_items(params),
                mean_balanced_accuracy=_finite_metric(
                    results[f"mean_test_{PRIMARY_CV_METRIC}"][index],
                    field="mean_balanced_accuracy",
                ),
                std_balanced_accuracy=_finite_metric(
                    results[f"std_test_{PRIMARY_CV_METRIC}"][index],
                    field="std_balanced_accuracy",
                ),
                mean_macro_f1=_finite_metric(
                    results[f"mean_test_{SECONDARY_CV_METRIC}"][index],
                    field="mean_macro_f1",
                ),
                std_macro_f1=_finite_metric(
                    results[f"std_test_{SECONDARY_CV_METRIC}"][index],
                    field="std_macro_f1",
                ),
                balanced_accuracy_rank=int(
                    results[f"rank_test_{PRIMARY_CV_METRIC}"][index]
                ),
            )
        )
    return tuple(output)


def _classification_metrics(
    *,
    expected: Sequence[str],
    predicted: Sequence[str],
    class_labels: Sequence[str],
    dependencies: _MLDependencies,
) -> ClassificationMetrics:
    return ClassificationMetrics(
        sample_count=len(expected),
        class_sample_counts=_class_sample_counts(expected, class_labels),
        balanced_accuracy=_finite_metric(
            dependencies.balanced_accuracy_score(expected, predicted),
            field="model_selection balanced_accuracy",
        ),
        macro_f1=_finite_metric(
            dependencies.f1_score(
                expected,
                predicted,
                labels=list(class_labels),
                average="macro",
                zero_division=0,
            ),
            field="model_selection macro_f1",
        ),
    )


def _select_model(candidates: Sequence[ClassifierCandidateResult]) -> str:
    if not candidates:
        raise ValueError("classifier comparison produced no candidates")

    def rank(item: ClassifierCandidateResult) -> tuple[object, ...]:
        metrics = item.model_selection_metrics
        return (
            -(
                metrics.balanced_accuracy
                if metrics is not None
                else item.best_cv_balanced_accuracy
            ),
            -(metrics.macro_f1 if metrics is not None else item.best_cv_macro_f1),
            -item.best_cv_balanced_accuracy,
            -item.best_cv_macro_f1,
            MODEL_SELECTION_PREFERENCE.index(item.model_name),
        )

    return min(candidates, key=rank).model_name


def _partition_fingerprint(
    frame: pl.DataFrame,
    *,
    split: str,
    config: ClassifierTrainingConfig,
) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema_version": CLASSIFIER_TRAINING_VERSION,
            "configuration_fingerprint": config.fingerprint,
            "dataset_split": split,
            "training_example_fingerprints": frame[
                "training_example_fingerprint"
            ].to_list(),
        }
    )


def _candidate_set_fingerprint(frame: pl.DataFrame) -> str:
    values = sorted(
        {
            str(value)
            for value in frame["candidate_set_fingerprint"].to_list()
            if value is not None
        }
    )
    return canonical_semantic_fingerprint(
        {
            "schema_version": CLASSIFIER_TRAINING_VERSION,
            "candidate_set_fingerprints": values,
        }
    )


def _load_ml_dependencies() -> _MLDependencies:
    try:
        import numpy as np
        import sklearn
        from sklearn import config_context
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, f1_score
        from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC, SVC
    except ImportError as exc:  # pragma: no cover - exercised in a minimal install.
        raise RuntimeError(
            "frozen classifier training requires the optional biominer[ml] dependencies"
        ) from exc
    return _MLDependencies(
        np=np,
        sklearn_version=sklearn.__version__,
        ColumnTransformer=ColumnTransformer,
        SimpleImputer=SimpleImputer,
        LogisticRegression=LogisticRegression,
        GridSearchCV=GridSearchCV,
        StratifiedGroupKFold=StratifiedGroupKFold,
        Pipeline=Pipeline,
        StandardScaler=StandardScaler,
        LinearSVC=LinearSVC,
        SVC=SVC,
        balanced_accuracy_score=balanced_accuracy_score,
        f1_score=f1_score,
        config_context=config_context,
    )


def _validate_class_weight_labels(
    class_weight: str | tuple[tuple[str, float], ...] | None,
    class_labels: Sequence[str],
) -> None:
    if isinstance(class_weight, tuple):
        keys = {key for key, _ in class_weight}
        if keys != set(class_labels):
            raise ValueError(
                "explicit class_weight keys must exactly match fitted classes"
            )


def _normalized_class_weight(
    value: object,
) -> str | tuple[tuple[str, float], ...] | None:
    if value is None or value == "balanced":
        return value
    if isinstance(value, str):
        raise ValueError("class_weight string must be 'balanced'")
    items = tuple(value.items()) if isinstance(value, Mapping) else value
    if isinstance(items, (str, bytes, bytearray)) or not isinstance(items, Sequence):
        raise TypeError("class_weight must be None, 'balanced', or label/weight pairs")
    parsed: dict[str, float] = {}
    for item in items:
        if not isinstance(item, Sequence) or len(item) != 2:
            raise TypeError("class_weight entries must be label/weight pairs")
        key = _required_text(item[0], field="class_weight label")
        weight = float(item[1])
        if not isfinite(weight) or weight <= 0.0:
            raise ValueError("class_weight values must be finite and positive")
        if key in parsed:
            raise ValueError("class_weight labels must be unique")
        parsed[key] = weight
    if not parsed:
        raise ValueError("explicit class_weight must not be empty")
    return tuple(sorted(parsed.items()))


def _class_weight_semantic_value(
    value: str | tuple[tuple[str, float], ...] | None,
) -> object:
    if isinstance(value, tuple):
        return [[key, weight] for key, weight in value]
    return value


def _parameter_items(values: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple(
        (str(key), _parameter_value(value)) for key, value in sorted(values.items())
    )


def _parameter_value(value: object) -> object:
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    if hasattr(value, "item"):
        return _parameter_value(value.item())
    raise TypeError(f"unsupported hyperparameter value: {type(value).__name__}")


def _finite_metric(value: object, *, field: str) -> float:
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _single_value(frame: pl.DataFrame, field: str) -> str:
    values = frame[field].unique().to_list()
    if len(values) != 1:
        raise ValueError(f"training features require one {field}")
    return _required_text(values[0], field=field)


def _sorted_unique_choices(
    values: object,
    *,
    field: str,
    allowed: frozenset[str],
) -> tuple[str, ...]:
    result = _unique_choices(values, field=field, allowed=allowed)
    return tuple(sorted(result))


def _unique_choices(
    values: object,
    *,
    field: str,
    allowed: frozenset[str],
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError(f"{field} must be a sequence")
    result = tuple(_required_text(value, field=field) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} must be unique")
    invalid = sorted(set(result) - allowed)
    if invalid:
        raise ValueError(f"{field} contains unsupported values: {invalid}")
    return result


def _required_choice(value: object, *, field: str, allowed: frozenset[str]) -> str:
    parsed = _required_text(value, field=field)
    if parsed not in allowed:
        raise ValueError(f"unsupported {field}: {parsed}")
    return parsed


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


def _random_seed(value: object) -> int:
    parsed = _integer_at_least(value, minimum=0, field="random_seed")
    if parsed > 2**32 - 1:
        raise ValueError("random_seed exceeds NumPy RandomState bounds")
    return parsed


def _positive_integer(value: object, *, field: str) -> int:
    return _integer_at_least(value, minimum=1, field=field)


def _integer_at_least(value: object, *, minimum: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _log_event(event: str, **fields: object) -> None:
    _LOGGER.info(
        "%s",
        json.dumps(
            {"event": event, **fields},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ),
    )


__all__ = [
    "CLASSIFIER_FEATURE_SETS",
    "CLASSIFIER_MODELS",
    "CLASSIFIER_SEARCH_GRID_VERSION",
    "CLASSIFIER_TRAINING_VERSION",
    "CV_SPLIT_VERSION",
    "DEFAULT_CLASSIFIER_MODELS",
    "DEFAULT_RBF_MAX_FIT_SAMPLES",
    "EMBEDDING_ONLY_FEATURE_SET",
    "EMBEDDING_PLUS_STRUCTURED_FEATURE_SET",
    "ESTIMATOR_LINEAR_SVC",
    "ESTIMATOR_LOGISTIC_REGRESSION",
    "ESTIMATOR_RBF_SVC",
    "FEATURE_LAYOUT_VERSION",
    "LINEAR_SVC_EMBEDDING_MODEL",
    "LINEAR_SVC_STRUCTURED_MODEL",
    "LOGISTIC_REGRESSION_MODEL",
    "MODEL_SELECTION_PREFERENCE",
    "NON_TARGET_CLASS_LABEL",
    "PRIMARY_CV_METRIC",
    "QUALITY_FLAG_HASH_BUCKET_COUNT",
    "QUALITY_FLAG_HASH_VERSION",
    "RBF_C_GRID",
    "RBF_GAMMA_GRID",
    "RBF_SVC_PILOT_MODEL",
    "REGULARIZATION_C_GRID",
    "SECONDARY_CV_METRIC",
    "ClassificationMetrics",
    "ClassifierCandidateResult",
    "ClassifierFeatureLayout",
    "ClassifierTrainingConfig",
    "ClassifierTrainingRun",
    "GroupFoldAudit",
    "HyperparameterScore",
    "classifier_feature_layout",
    "materialize_classifier_feature_matrix",
    "train_frozen_embedding_classifiers",
]
