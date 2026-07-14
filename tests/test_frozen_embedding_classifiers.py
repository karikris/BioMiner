from __future__ import annotations

from dataclasses import replace
from math import sqrt
import subprocess
import sys

import pytest

from biominer.ml.classifiers import (
    DEFAULT_CLASSIFIER_MODELS,
    EMBEDDING_ONLY_FEATURE_SET,
    EMBEDDING_PLUS_STRUCTURED_FEATURE_SET,
    LINEAR_SVC_EMBEDDING_MODEL,
    LINEAR_SVC_STRUCTURED_MODEL,
    LOGISTIC_REGRESSION_MODEL,
    NON_TARGET_CLASS_LABEL,
    RBF_SVC_PILOT_MODEL,
    ClassifierTrainingConfig,
    train_frozen_embedding_classifiers,
)
from biominer.ml.training_features import build_few_shot_training_features
from test_few_shot_training_features import COMPETITOR, TARGET, _example, _sha


THIRD_SPECIES = "gbif:1938071"
pytestmark = pytest.mark.filterwarnings("error")


def test_trains_default_binary_comparators_with_group_audited_folds() -> None:
    frame = _training_frame(
        task="binary_target_verifier",
        classes=(TARGET, COMPETITOR),
        fit_groups_per_class=4,
        model_selection_groups_per_class=2,
    )
    config = ClassifierTrainingConfig(
        target_task="binary_target_verifier",
        target_accepted_taxon_key=TARGET,
        route="adult_field",
        n_splits=2,
        random_seed=17,
        class_weight="balanced",
    )

    run = train_frozen_embedding_classifiers(frame, config)

    assert tuple(item.model_name for item in run.candidates) == (
        LOGISTIC_REGRESSION_MODEL,
        LINEAR_SVC_EMBEDDING_MODEL,
        LINEAR_SVC_STRUCTURED_MODEL,
    )
    assert config.enabled_models == DEFAULT_CLASSIFIER_MODELS
    assert run.class_labels == (NON_TARGET_CLASS_LABEL, TARGET)
    assert run.fit_sample_count == 8
    assert run.fit_group_count == 8
    assert run.model_selection_sample_count == 4
    assert run.calibration_sample_count == 0
    assert run.final_test_sample_count == 0
    assert run.foundation_model_trainable is False
    assert run.training_run_fingerprint.startswith("sha256:")
    assert run.cv_split_fingerprint.startswith("sha256:")
    assert len(run.folds) == 2
    for fold in run.folds:
        assert set(fold.train_group_ids).isdisjoint(fold.validation_group_ids)
        assert dict(fold.train_class_sample_counts).keys() == set(run.class_labels)
        assert dict(fold.validation_class_sample_counts).keys() == set(run.class_labels)

    candidates = {item.model_name: item for item in run.candidates}
    assert candidates[LOGISTIC_REGRESSION_MODEL].feature_set == (
        EMBEDDING_ONLY_FEATURE_SET
    )
    assert candidates[LINEAR_SVC_EMBEDDING_MODEL].feature_set == (
        EMBEDDING_ONLY_FEATURE_SET
    )
    structured = candidates[LINEAR_SVC_STRUCTURED_MODEL]
    assert structured.feature_set == EMBEDDING_PLUS_STRUCTURED_FEATURE_SET
    assert "target_reference_centroid_similarity" in (
        structured.feature_layout.source_feature_names
    )
    assert "target_present" not in structured.feature_layout.source_feature_names
    assert structured.feature_layout.fingerprint.startswith("sha256:")
    assert all(item.model_selection_metrics is not None for item in run.candidates)
    assert run.selected_model_name == LINEAR_SVC_EMBEDDING_MODEL

    for candidate in run.candidates:
        classifier = candidate.pipeline.named_steps["classifier"]
        assert classifier.class_weight == "balanced"
        assert candidate.probability_calibrated is False
        assert candidate.best_cv_balanced_accuracy >= 0.5
        assert candidate.selected_parameter_dict["classifier__C"] in {
            0.01,
            0.1,
            1.0,
            10.0,
        }
        assert candidate.candidate_fingerprint.startswith("sha256:")
    logistic_configuration = dict(
        candidates[LOGISTIC_REGRESSION_MODEL].estimator_configuration
    )
    assert logistic_configuration["effective_penalty"] == "l2"
    assert logistic_configuration["solver"] == "lbfgs"


@pytest.mark.parametrize(
    ("task", "route", "classes", "expected_labels"),
    (
        (
            "regional_multiclass",
            "adult_field",
            (TARGET, COMPETITOR, THIRD_SPECIES),
            (TARGET, COMPETITOR, THIRD_SPECIES),
        ),
        (
            "visual_domain",
            "adult_field",
            ("live_field", "pinned_specimen"),
            ("live_field", "pinned_specimen"),
        ),
        (
            "larval_target_verifier",
            "larval",
            (TARGET, COMPETITOR),
            (NON_TARGET_CLASS_LABEL, TARGET),
        ),
    ),
)
def test_routes_labels_for_every_required_classifier_task(
    task: str,
    route: str,
    classes: tuple[str, ...],
    expected_labels: tuple[str, ...],
) -> None:
    frame = _training_frame(
        task=task,
        route=route,
        classes=classes,
        fit_groups_per_class=3,
    )
    config = ClassifierTrainingConfig(
        target_task=task,
        target_accepted_taxon_key=TARGET,
        route=route,
        n_splits=3,
        enabled_models=(LINEAR_SVC_EMBEDDING_MODEL,),
    )

    run = train_frozen_embedding_classifiers(frame, config)

    assert run.class_labels == tuple(sorted(expected_labels))
    assert run.fit_sample_count == len(classes) * 3
    assert run.selected_model_name == LINEAR_SVC_EMBEDDING_MODEL
    assert run.candidates[0].model_selection_metrics is None


def test_rejects_conflicting_group_labels_and_too_few_groups_per_class() -> None:
    with pytest.raises(ValueError, match="included_label_certainties"):
        ClassifierTrainingConfig(
            target_task="binary_target_verifier",
            target_accepted_taxon_key=TARGET,
            route="adult_field",
            included_label_certainties=(),
        )

    conflicting = _training_frame(
        task="binary_target_verifier",
        classes=(TARGET, COMPETITOR),
        fit_groups_per_class=3,
        conflicting_group_labels=True,
    )
    config = ClassifierTrainingConfig(
        target_task="binary_target_verifier",
        target_accepted_taxon_key=TARGET,
        route="adult_field",
        n_splits=3,
        enabled_models=(LINEAR_SVC_EMBEDDING_MODEL,),
    )
    with pytest.raises(ValueError, match="leakage group maps to multiple labels"):
        train_frozen_embedding_classifiers(conflicting, config)

    insufficient = _training_frame(
        task="binary_target_verifier",
        classes=(TARGET, COMPETITOR),
        fit_groups_per_class=2,
    )
    with pytest.raises(ValueError, match="groups per class.*n_splits=3"):
        train_frozen_embedding_classifiers(insufficient, config)


def test_rbf_is_explicitly_enabled_bounded_and_never_estimates_probability() -> None:
    frame = _training_frame(
        task="binary_target_verifier",
        classes=(TARGET, COMPETITOR),
        fit_groups_per_class=3,
    )
    too_small = ClassifierTrainingConfig(
        target_task="binary_target_verifier",
        target_accepted_taxon_key=TARGET,
        route="adult_field",
        n_splits=3,
        enabled_models=(),
        enable_rbf_pilot=True,
        rbf_max_fit_samples=5,
    )
    with pytest.raises(ValueError, match="RBF pilot sample cap"):
        train_frozen_embedding_classifiers(frame, too_small)

    bounded = replace(too_small, rbf_max_fit_samples=6)
    run = train_frozen_embedding_classifiers(frame, bounded)

    assert tuple(item.model_name for item in run.candidates) == (RBF_SVC_PILOT_MODEL,)
    classifier = run.candidates[0].pipeline.named_steps["classifier"]
    assert classifier.kernel == "rbf"
    assert classifier.probability == "deprecated"
    assert not hasattr(classifier, "predict_proba")
    assert run.candidates[0].probability_calibrated is False


def test_custom_class_weights_and_seed_produce_repeatable_search() -> None:
    frame = _training_frame(
        task="binary_target_verifier",
        classes=(TARGET, COMPETITOR),
        fit_groups_per_class=3,
    )
    config = ClassifierTrainingConfig(
        target_task="binary_target_verifier",
        target_accepted_taxon_key=TARGET,
        route="adult_field",
        n_splits=3,
        random_seed=91,
        class_weight=((NON_TARGET_CLASS_LABEL, 1.5), (TARGET, 2.0)),
        enabled_models=(LOGISTIC_REGRESSION_MODEL,),
    )

    first = train_frozen_embedding_classifiers(frame, config)
    second = train_frozen_embedding_classifiers(frame, config)

    assert first.cv_split_fingerprint == second.cv_split_fingerprint
    assert first.training_run_fingerprint == second.training_run_fingerprint
    assert first.candidates[0].selected_parameters == (
        second.candidates[0].selected_parameters
    )
    classifier = first.candidates[0].pipeline.named_steps["classifier"]
    assert classifier.class_weight == {
        NON_TARGET_CLASS_LABEL: 1.5,
        TARGET: 2.0,
    }


def test_calibration_and_final_test_rows_never_enter_fitting_or_selection() -> None:
    support_only = _training_frame(
        task="binary_target_verifier",
        classes=(TARGET, COMPETITOR),
        fit_groups_per_class=3,
    )
    with_holdouts = _training_frame(
        task="binary_target_verifier",
        classes=(TARGET, COMPETITOR),
        fit_groups_per_class=3,
        calibration_groups_per_class=2,
        final_test_groups_per_class=2,
    )
    config = ClassifierTrainingConfig(
        target_task="binary_target_verifier",
        target_accepted_taxon_key=TARGET,
        route="adult_field",
        n_splits=3,
        enabled_models=(LINEAR_SVC_EMBEDDING_MODEL,),
    )

    baseline = train_frozen_embedding_classifiers(support_only, config)
    audited = train_frozen_embedding_classifiers(with_holdouts, config)

    assert audited.fit_sample_count == baseline.fit_sample_count == 6
    assert audited.calibration_sample_count == 4
    assert audited.final_test_sample_count == 4
    assert audited.cv_split_fingerprint == baseline.cv_split_fingerprint
    assert audited.fit_partition_fingerprint == baseline.fit_partition_fingerprint
    assert audited.selected_model_name == baseline.selected_model_name
    assert audited.candidates[0].candidate_fingerprint == (
        baseline.candidates[0].candidate_fingerprint
    )
    assert audited.candidates[0].selected_parameters == (
        baseline.candidates[0].selected_parameters
    )


def test_classifier_contract_import_does_not_load_optional_ml_stack() -> None:
    script = """
import sys
import biominer.ml.classifiers as classifiers
assert classifiers.DEFAULT_CLASSIFIER_MODELS
assert 'numpy' not in sys.modules
assert 'sklearn' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def _training_frame(
    *,
    task: str,
    classes: tuple[str, ...],
    route: str = "adult_field",
    fit_groups_per_class: int,
    model_selection_groups_per_class: int = 0,
    calibration_groups_per_class: int = 0,
    final_test_groups_per_class: int = 0,
    conflicting_group_labels: bool = False,
):
    examples = []
    for class_index, class_label in enumerate(classes):
        for group_index in range(fit_groups_per_class):
            group_id = f"fit-group:{class_index}:{group_index}"
            if conflicting_group_labels and class_index == 1 and group_index == 0:
                group_id = "fit-group:0:0"
            examples.append(
                _classifier_example(
                    item_id=f"fit:{class_index}:{group_index}",
                    group_id=group_id,
                    dataset_split="support_train",
                    task=task,
                    route=route,
                    class_label=class_label,
                    class_index=class_index,
                    sample_index=group_index,
                )
            )
        for group_index in range(model_selection_groups_per_class):
            examples.append(
                _classifier_example(
                    item_id=f"selection:{class_index}:{group_index}",
                    group_id=f"selection-group:{class_index}:{group_index}",
                    dataset_split="model_selection",
                    task=task,
                    route=route,
                    class_label=class_label,
                    class_index=class_index,
                    sample_index=group_index,
                )
            )
        for split, count in (
            ("calibration", calibration_groups_per_class),
            ("final_test", final_test_groups_per_class),
        ):
            for group_index in range(count):
                examples.append(
                    _classifier_example(
                        item_id=f"{split}:{class_index}:{group_index}",
                        group_id=f"{split}-group:{class_index}:{group_index}",
                        dataset_split=split,
                        task=task,
                        route=route,
                        class_label=class_label,
                        class_index=class_index,
                        sample_index=group_index,
                    )
                )
    return build_few_shot_training_features(tuple(examples))


def _classifier_example(
    *,
    item_id: str,
    group_id: str,
    dataset_split: str,
    task: str,
    route: str,
    class_label: str,
    class_index: int,
    sample_index: int,
):
    base = _example(item_id, target_present=class_index == 0)
    target_present = (
        class_label == TARGET if task != "visual_domain" else class_index == 0
    )
    accepted_class = (
        TARGET
        if target_present
        else (class_label if task == "regional_multiclass" else COMPETITOR)
    )
    visual_domain = class_label if task == "visual_domain" else "live_field"
    vector = _unit(
        (
            1.0 if class_index == 0 else -0.4 * class_index,
            0.15 * sample_index,
            0.5 * class_index + 0.05,
        )
    )
    yoloe_route = (
        "caterpillar_field"
        if route == "larval"
        else (
            "pinned_specimen" if route == "pinned_specimen" else "adult_butterfly_field"
        )
    )
    provenance = replace(
        base.provenance,
        source_item_id=item_id,
        source_observation_id=f"observation:{item_id}",
        source_owner_id=f"owner:{item_id}",
        duplicate_group_id=f"duplicate:{item_id}",
        burst_group_id=f"burst:{item_id}",
        provider_mirror_group_id=f"mirror:{item_id}",
        leakage_group_id=group_id,
        dataset_split=dataset_split,
        candidate_set_fingerprint=_sha(f"candidate-set:{item_id}"),
    )
    label = replace(
        base.label,
        reviewed_label_id=f"label:{item_id}",
        reviewed_label_fingerprint=_sha(f"label:{item_id}"),
        target_present=target_present,
        accepted_class_taxon_key=accepted_class,
        visual_domain_label=visual_domain,
    )
    embedding = replace(
        base.embedding,
        visual_input_id=f"visual-input:{item_id}",
        embedding=vector,
        embedding_fingerprint=_sha(f"embedding:{item_id}"),
    )
    reference = replace(base.reference, route=route)
    detection = replace(
        base.detection,
        yoloe_route=yoloe_route,
        detector_confidence=0.75 + 0.01 * sample_index,
        visual_input_quality_flags=(
            ("blurred",) if sample_index % 2 else ("small_subject",)
        ),
    )
    return replace(
        base,
        target_task=task,
        route=route,
        provenance=provenance,
        label=label,
        embedding=embedding,
        reference=reference,
        detection=detection,
    )


def _unit(values: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values)  # type: ignore[return-value]
