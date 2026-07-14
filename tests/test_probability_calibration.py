from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from biominer.ml.calibration import (
    CALIBRATION_ARRAYS_FILE,
    CALIBRATION_MANIFEST_FILE,
    CALIBRATION_MANIFEST_SCHEMA_VERSION,
    CALIBRATION_REPORT_FILE,
    CALIBRATION_REPORT_SCHEMA_VERSION,
    CalibrationConfig,
    CalibrationFoldAudit,
    CalibrationPrediction,
    fit_probability_calibrator,
    load_probability_calibrator,
    write_probability_calibrator,
)


NON_TARGET = "__non_target__"
TARGET = "gbif:6432573"
COMPETITOR = "gbif:1939773"
THIRD_SPECIES = "gbif:5139051"
CREATED_AT = datetime(2026, 7, 14, 5, 6, 7, 890123, tzinfo=timezone.utc)
GIT_SHA = "a" * 40


def _sha(value: str) -> str:
    import hashlib

    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _binary_inputs(
    *,
    group_count: int = 24,
    fold_count: int = 3,
) -> tuple[tuple[CalibrationPrediction, ...], tuple[CalibrationFoldAudit, ...]]:
    groups = tuple(f"component-{index:04d}" for index in range(group_count))
    predictions = []
    for index, group_id in enumerate(groups):
        positive = index % 2 == 1
        # Include difficult examples so calibration is not a perfect separator.
        base = 1.4 if positive else -1.4
        score = base + ((index % 5) - 2) * 0.55
        predictions.append(
            CalibrationPrediction(
                prediction_id=f"prediction-{index:04d}",
                source_item_id=f"item-{index:04d}",
                leakage_component_id=group_id,
                fold_index=index % fold_count,
                dataset_split="calibration",
                true_class_label=TARGET if positive else NON_TARGET,
                decision_scores=(score,),
                sample_weight=1.0 + (index % 3) * 0.25,
            )
        )
    audits = tuple(
        CalibrationFoldAudit(
            fold_index=fold_index,
            estimator_fit_group_ids=tuple(
                group_id
                for index, group_id in enumerate(groups)
                if index % fold_count != fold_index
            ),
            validation_group_ids=tuple(
                group_id
                for index, group_id in enumerate(groups)
                if index % fold_count == fold_index
            ),
        )
        for fold_index in range(fold_count)
    )
    return tuple(predictions), audits


def _binary_config(*, method: str = "auto") -> CalibrationConfig:
    return CalibrationConfig(
        classifier_fingerprint=_sha("binary-classifier"),
        split_fingerprint=_sha("dataset-split"),
        target_task="binary_target_verifier",
        route="adult_field",
        class_labels=(NON_TARGET, TARGET),
        positive_class_label=TARGET,
        method=method,
        reliability_bin_count=8,
    )


def _multiclass_inputs() -> tuple[
    tuple[CalibrationPrediction, ...], tuple[CalibrationFoldAudit, ...]
]:
    class_labels = (TARGET, COMPETITOR, THIRD_SPECIES)
    groups = tuple(f"regional-component-{index:03d}" for index in range(36))
    predictions = []
    for index, group_id in enumerate(groups):
        class_index = index % len(class_labels)
        scores = [-0.4, -0.4, -0.4]
        scores[class_index] = 0.7 + (index % 4) * 0.08
        scores[(class_index + 1) % 3] += 0.25
        predictions.append(
            CalibrationPrediction(
                prediction_id=f"regional-prediction-{index:03d}",
                source_item_id=f"regional-item-{index:03d}",
                leakage_component_id=group_id,
                fold_index=index % 3,
                dataset_split="calibration",
                true_class_label=class_labels[class_index],
                decision_scores=tuple(scores),
            )
        )
    audits = tuple(
        CalibrationFoldAudit(
            fold_index=fold_index,
            estimator_fit_group_ids=tuple(
                group_id
                for index, group_id in enumerate(groups)
                if index % 3 != fold_index
            ),
            validation_group_ids=tuple(
                group_id
                for index, group_id in enumerate(groups)
                if index % 3 == fold_index
            ),
        )
        for fold_index in range(3)
    )
    return tuple(predictions), audits


def test_binary_sigmoid_uses_group_aware_oof_predictions() -> None:
    predictions, audits = _binary_inputs()

    fit = fit_probability_calibrator(predictions, audits, _binary_config())
    probabilities = fit.calibrator.predict_proba(
        np.asarray([[-3.0], [0.0], [3.0]], dtype=np.float64)
    )

    assert fit.method == "sigmoid"
    assert fit.sample_count == len(predictions)
    assert fit.group_count == len(predictions)
    assert fit.independent_prediction_artifact_fingerprint.startswith("sha256:")
    assert fit.calibration_fingerprint.startswith("sha256:")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[0, 1] < probabilities[1, 1] < probabilities[2, 1]
    assert set(fit.report["schema_version"].unique()) == {
        CALIBRATION_REPORT_SCHEMA_VERSION
    }
    assert set(fit.report["dataset_split"].unique()) == {"calibration"}
    assert set(fit.report["probability_kind"].unique()) == {"calibrated_probability"}
    assert fit.report.height == 2 * 8
    assert fit.report["brier_contribution"].sum() == pytest.approx(
        fit.metrics["brier_score"]
    )
    assert fit.report["log_loss_contribution"].sum() == pytest.approx(
        fit.metrics["classwise_log_loss"]
    )
    assert fit.report["ece_contribution"].sum() == pytest.approx(
        fit.metrics["expected_calibration_error"]
    )


def test_fold_audit_rejects_estimator_fit_group_in_its_validation_fold() -> None:
    predictions, audits = _binary_inputs()
    leaked = replace(
        audits[0],
        estimator_fit_group_ids=(
            *audits[0].estimator_fit_group_ids,
            audits[0].validation_group_ids[0],
        ),
    )

    with pytest.raises(ValueError, match="estimator-fit and validation groups overlap"):
        fit_probability_calibrator(
            predictions,
            (leaked, *audits[1:]),
            _binary_config(),
        )


def test_fold_audit_rejects_non_calibration_and_cross_fold_predictions() -> None:
    predictions, audits = _binary_inputs()
    wrong_partition = replace(predictions[0], dataset_split="support_train")
    with pytest.raises(ValueError, match="calibration partition"):
        fit_probability_calibrator(
            (wrong_partition, *predictions[1:]),
            audits,
            _binary_config(),
        )

    wrong_fold = replace(predictions[0], fold_index=1)
    with pytest.raises(ValueError, match="validation group does not belong"):
        fit_probability_calibrator(
            (wrong_fold, *predictions[1:]),
            audits,
            _binary_config(),
        )


def test_source_item_cannot_claim_multiple_leakage_components() -> None:
    predictions, audits = _binary_inputs()
    conflicting = replace(predictions[1], source_item_id=predictions[0].source_item_id)

    with pytest.raises(ValueError, match="source item cannot cross leakage components"):
        fit_probability_calibrator(
            (predictions[0], conflicting, *predictions[2:]),
            audits,
            _binary_config(),
        )


def test_isotonic_requires_at_least_one_thousand_independent_predictions() -> None:
    predictions, audits = _binary_inputs(group_count=999)

    with pytest.raises(ValueError, match="at least 1000 independent predictions"):
        fit_probability_calibrator(
            predictions,
            audits,
            _binary_config(method="isotonic"),
        )


def test_isotonic_knots_are_numeric_arrays_when_evidence_is_sufficient() -> None:
    groups = tuple(f"isotonic-component-{index:03d}" for index in range(200))
    predictions = tuple(
        CalibrationPrediction(
            prediction_id=f"isotonic-{index:04d}",
            source_item_id=f"isotonic-item-{index:04d}",
            leakage_component_id=groups[index // 5],
            fold_index=(index // 5) % 5,
            dataset_split="calibration",
            true_class_label=TARGET if index % 2 else NON_TARGET,
            decision_scores=((index % 101) / 25.0 - 2.0,),
        )
        for index in range(1000)
    )
    audits = tuple(
        CalibrationFoldAudit(
            fold_index=fold_index,
            estimator_fit_group_ids=tuple(
                group_id
                for index, group_id in enumerate(groups)
                if index % 5 != fold_index
            ),
            validation_group_ids=tuple(
                group_id
                for index, group_id in enumerate(groups)
                if index % 5 == fold_index
            ),
        )
        for fold_index in range(5)
    )

    fit = fit_probability_calibrator(
        predictions,
        audits,
        _binary_config(method="isotonic"),
    )

    assert fit.method == "isotonic"
    assert set(fit.array_parameters) == {"isotonic_thresholds", "isotonic_values"}
    probabilities = fit.calibrator.predict_proba(np.asarray([[-4.0], [4.0]]))
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_multiclass_auto_method_fits_temperature_scaling() -> None:
    predictions, audits = _multiclass_inputs()
    config = CalibrationConfig(
        classifier_fingerprint=_sha("regional-classifier"),
        split_fingerprint=_sha("regional-split"),
        target_task="regional_multiclass",
        route="adult_field",
        class_labels=(TARGET, COMPETITOR, THIRD_SPECIES),
        method="auto",
        reliability_bin_count=6,
    )

    fit = fit_probability_calibrator(predictions, audits, config)
    probabilities = fit.calibrator.predict_proba(
        np.asarray([[1.0, 0.0, -1.0], [-0.2, 0.1, 0.4]], dtype=np.float64)
    )

    assert fit.method == "temperature"
    assert fit.scalar_parameters["inverse_temperature"] > 0.0
    assert fit.metrics["log_loss"] <= fit.metrics["uncalibrated_softmax_log_loss"]
    assert probabilities.shape == (2, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert fit.report.height == 3 * 6


def test_artifact_round_trip_is_deterministic_and_keeps_policy_disabled(
    tmp_path: Path,
) -> None:
    predictions, audits = _binary_inputs()
    fit = fit_probability_calibrator(predictions, audits, _binary_config())

    first = write_probability_calibrator(
        fit,
        tmp_path / "first",
        git_sha=GIT_SHA,
        created_at=CREATED_AT,
    )
    second = write_probability_calibrator(
        fit,
        tmp_path / "second",
        git_sha=GIT_SHA,
        created_at=CREATED_AT,
    )

    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.arrays_path.read_bytes() == second.arrays_path.read_bytes()
    assert first.report_path.read_bytes() == second.report_path.read_bytes()
    payload = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == CALIBRATION_MANIFEST_SCHEMA_VERSION
    assert payload["calibration_fingerprint"] == fit.calibration_fingerprint
    assert payload["decision_policy"]["status"] == "not_fitted"
    assert payload["decision_policy"]["target_confirmation_enabled"] is False
    assert payload["provenance"]["split_fingerprint"] == _sha("dataset-split")
    assert payload["provenance"]["oof_policy"].startswith("group-aware")
    assert payload["parameters"]["score_input_kind"] == "estimator_decision_score"
    assert set(path.name for path in first.directory.iterdir()) == {
        CALIBRATION_MANIFEST_FILE,
        CALIBRATION_ARRAYS_FILE,
        CALIBRATION_REPORT_FILE,
    }

    loaded = load_probability_calibrator(
        first.directory,
        expected_calibration_fingerprint=fit.calibration_fingerprint,
        expected_classifier_fingerprint=_sha("binary-classifier"),
        expected_split_fingerprint=_sha("dataset-split"),
    )
    raw = np.asarray([[-2.0], [0.2], [2.5]], dtype=np.float64)
    np.testing.assert_allclose(
        loaded.calibrator.predict_proba(raw),
        fit.calibrator.predict_proba(raw),
        rtol=0.0,
        atol=0.0,
    )
    assert loaded.report.equals(fit.report)

    with pytest.raises(FileExistsError):
        write_probability_calibrator(
            fit,
            first.directory,
            git_sha=GIT_SHA,
            created_at=CREATED_AT,
        )


@pytest.mark.parametrize("artifact", ("manifest", "arrays", "report"))
def test_artifact_tampering_fails_closed(tmp_path: Path, artifact: str) -> None:
    predictions, audits = _binary_inputs()
    fit = fit_probability_calibrator(predictions, audits, _binary_config())
    paths = write_probability_calibrator(
        fit,
        tmp_path / artifact,
        git_sha=GIT_SHA,
        created_at=CREATED_AT,
    )

    path = {
        "manifest": paths.manifest_path,
        "arrays": paths.arrays_path,
        "report": paths.report_path,
    }[artifact]
    value = bytearray(path.read_bytes())
    value[len(value) // 2] ^= 1
    path.write_bytes(value)

    message = {
        "manifest": "manifest",
        "arrays": "array archive checksum",
        "report": "report checksum",
    }[artifact]
    with pytest.raises(ValueError, match=message):
        load_probability_calibrator(paths.directory)


def test_loaded_runtime_does_not_import_sklearn(tmp_path: Path) -> None:
    predictions, audits = _binary_inputs()
    fit = fit_probability_calibrator(predictions, audits, _binary_config())
    paths = write_probability_calibrator(
        fit,
        tmp_path / "runtime",
        git_sha=GIT_SHA,
        created_at=CREATED_AT,
    )
    script = """
import builtins
import pathlib

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'sklearn' or name.startswith('sklearn.'):
        raise AssertionError('runtime imported sklearn')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

from biominer.ml.calibration import load_probability_calibrator
loaded = load_probability_calibrator(pathlib.Path(__import__('sys').argv[1]))
values = loaded.calibrator.predict_proba([[-1.0], [1.0]])
assert values.shape == (2, 2)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(paths.directory)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
