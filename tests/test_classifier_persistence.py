from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.ml.classifiers import (
    ESTIMATOR_LINEAR_SVC,
    ESTIMATOR_RBF_SVC,
    LINEAR_SVC_EMBEDDING_MODEL,
    LINEAR_SVC_STRUCTURED_MODEL,
    LOGISTIC_REGRESSION_MODEL,
    RBF_SVC_PILOT_MODEL,
    ClassifierTrainingConfig,
    materialize_classifier_feature_matrix,
    train_frozen_embedding_classifiers,
)
from biominer.ml.persistence import (
    CLASSIFIER_ARRAYS_FILE,
    CLASSIFIER_MANIFEST_FILE,
    CLASSIFIER_MANIFEST_SCHEMA_VERSION,
    load_frozen_classifier,
    write_frozen_classifier,
)
from test_few_shot_training_features import COMPETITOR, TARGET, _sha
from test_frozen_embedding_classifiers import THIRD_SPECIES, _training_frame


CREATED_AT = datetime(2026, 7, 14, 3, 4, 5, 678901, tzinfo=timezone.utc)
GIT_SHA = "a" * 40
PREPROCESSING_FINGERPRINT = _sha("preprocessing")
REFERENCE_BANK_FINGERPRINT = _sha("reference-bank")
REFERENCE_BANK_VERSION = "reference-bank-test-v1"


@pytest.fixture(scope="module")
def binary_training():
    frame = _training_frame(
        task="binary_target_verifier",
        classes=(TARGET, COMPETITOR),
        fit_groups_per_class=4,
        model_selection_groups_per_class=2,
    )
    run = train_frozen_embedding_classifiers(
        frame,
        ClassifierTrainingConfig(
            target_task="binary_target_verifier",
            target_accepted_taxon_key=TARGET,
            route="adult_field",
            n_splits=2,
            random_seed=17,
        ),
    )
    return frame, run


@pytest.fixture(scope="module")
def multiclass_training():
    frame = _training_frame(
        task="regional_multiclass",
        classes=(TARGET, COMPETITOR, THIRD_SPECIES),
        fit_groups_per_class=3,
    )
    run = train_frozen_embedding_classifiers(
        frame,
        ClassifierTrainingConfig(
            target_task="regional_multiclass",
            target_accepted_taxon_key=TARGET,
            route="adult_field",
            n_splits=3,
            enabled_models=(LINEAR_SVC_EMBEDDING_MODEL,),
        ),
    )
    return frame, run


@pytest.mark.parametrize(
    "model_name",
    (
        LOGISTIC_REGRESSION_MODEL,
        LINEAR_SVC_EMBEDDING_MODEL,
        LINEAR_SVC_STRUCTURED_MODEL,
    ),
)
def test_round_trip_matches_fitted_binary_pipeline(
    tmp_path: Path,
    binary_training,
    model_name: str,
) -> None:
    frame, run = binary_training
    candidate = next(item for item in run.candidates if item.model_name == model_name)
    paths = _write(run, tmp_path / model_name, model_name=model_name)

    loaded = load_frozen_classifier(
        paths.directory,
        expected_classifier_fingerprint=paths.classifier_fingerprint,
        expected_model_fingerprint=run.model_fingerprint,
        expected_preprocessing_fingerprint=PREPROCESSING_FINGERPRINT,
        expected_reference_bank_fingerprint=REFERENCE_BANK_FINGERPRINT,
        expected_training_data_fingerprint=run.training_data_fingerprint,
    )
    raw = materialize_classifier_feature_matrix(frame, candidate.feature_layout)

    np.testing.assert_allclose(
        loaded.decision_function(raw),
        candidate.pipeline.decision_function(raw),
        rtol=1e-12,
        atol=1e-12,
    )
    assert loaded.predict(raw) == tuple(
        str(value) for value in candidate.pipeline.predict(raw)
    )
    assert loaded.class_labels == run.class_labels
    assert loaded.model_name == model_name
    assert loaded.probability_calibrated is False
    assert paths.manifest_path.name == CLASSIFIER_MANIFEST_FILE
    assert paths.arrays_path.name == CLASSIFIER_ARRAYS_FILE


def test_svc_decision_margin_is_not_exposed_as_probability(
    tmp_path: Path,
    binary_training,
) -> None:
    frame, run = binary_training
    candidate = next(
        item for item in run.candidates if item.model_name == LINEAR_SVC_EMBEDDING_MODEL
    )
    paths = _write(
        run,
        tmp_path / "linear-svc-margin",
        model_name=LINEAR_SVC_EMBEDDING_MODEL,
    )
    loaded = load_frozen_classifier(paths.directory)
    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    raw = materialize_classifier_feature_matrix(frame, candidate.feature_layout)

    assert loaded.estimator_family == ESTIMATOR_LINEAR_SVC
    assert loaded.decision_function(raw).ndim == 1
    assert loaded.probability_calibrated is False
    assert manifest["identity"]["probability_calibrated"] is False
    assert not hasattr(loaded, "predict_proba")


def test_round_trip_preserves_multiclass_class_order(
    tmp_path: Path,
    multiclass_training,
) -> None:
    frame, run = multiclass_training
    candidate = run.selected_candidate
    paths = _write(run, tmp_path / "multiclass")
    loaded = load_frozen_classifier(paths.directory)
    raw = materialize_classifier_feature_matrix(frame, candidate.feature_layout)

    assert loaded.class_labels == tuple(sorted((TARGET, COMPETITOR, THIRD_SPECIES)))
    assert loaded.predict(raw) == tuple(
        str(value) for value in candidate.pipeline.predict(raw)
    )
    assert loaded.decision_function(raw).shape == (frame.height, 3)


def test_artifact_bytes_are_deterministic_and_manifest_is_complete(
    tmp_path: Path,
    binary_training,
) -> None:
    _, run = binary_training
    first = _write(
        run,
        tmp_path / "first",
        model_name=LINEAR_SVC_STRUCTURED_MODEL,
    )
    second = _write(
        run,
        tmp_path / "second",
        model_name=LINEAR_SVC_STRUCTURED_MODEL,
    )

    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.arrays_path.read_bytes() == second.arrays_path.read_bytes()
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == CLASSIFIER_MANIFEST_SCHEMA_VERSION
    assert manifest["classifier_fingerprint"] == first.classifier_fingerprint
    assert manifest["identity"]["class_labels"] == list(run.class_labels)
    assert manifest["identity"]["probability_calibrated"] is False
    assert (
        manifest["features"]["raw_feature_names"]
        == manifest["features"]["transformed_feature_names"]
    )
    assert set(manifest["arrays"]["entries"]) == {
        "class_indices",
        "coefficients",
        "continuous_imputer_statistics",
        "continuous_scaler_mean",
        "continuous_scaler_scale",
        "continuous_scaler_variance",
        "intercepts",
    }
    with np.load(first.arrays_path, allow_pickle=False) as arrays:
        assert set(arrays.files) == set(manifest["arrays"]["entries"])
        assert arrays["class_indices"].tolist() == list(range(len(run.class_labels)))
        assert all(arrays[name].dtype.kind in "fiu" for name in arrays.files)
        assert not any(arrays[name].dtype.hasobject for name in arrays.files)


def test_existing_artifact_is_never_overwritten(
    tmp_path: Path,
    binary_training,
) -> None:
    _, run = binary_training
    paths = _write(run, tmp_path / "immutable")
    original_manifest = paths.manifest_path.read_bytes()
    original_arrays = paths.arrays_path.read_bytes()

    with pytest.raises(FileExistsError):
        _write(run, paths.directory)

    assert paths.manifest_path.read_bytes() == original_manifest
    assert paths.arrays_path.read_bytes() == original_arrays


def test_rbf_pipeline_cannot_be_persisted_as_a_linear_artifact(
    tmp_path: Path,
    binary_training,
) -> None:
    _, run = binary_training
    source = run.selected_candidate
    rbf = replace(
        source,
        model_name=RBF_SVC_PILOT_MODEL,
        estimator_family=ESTIMATOR_RBF_SVC,
    )
    incompatible = replace(
        run,
        candidates=(rbf,),
        selected_model_name=RBF_SVC_PILOT_MODEL,
    )

    with pytest.raises(ValueError, match="linear classifiers"):
        _write(incompatible, tmp_path / "rbf")
    assert not (tmp_path / "rbf").exists()


def test_manifest_and_array_tampering_fail_closed(
    tmp_path: Path,
    binary_training,
) -> None:
    _, run = binary_training
    manifest_paths = _write(run, tmp_path / "manifest-tamper")
    payload = json.loads(manifest_paths.manifest_path.read_text(encoding="utf-8"))
    payload["training"]["training_data_fingerprint"] = _sha("tampered")
    _write_manifest(manifest_paths.manifest_path, payload, recompute_fingerprint=False)
    with pytest.raises(ValueError, match="classifier fingerprint"):
        load_frozen_classifier(manifest_paths.directory)

    array_paths = _write(run, tmp_path / "array-tamper")
    corrupted = bytearray(array_paths.arrays_path.read_bytes())
    corrupted[len(corrupted) // 2] ^= 1
    array_paths.arrays_path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="array archive checksum"):
        load_frozen_classifier(array_paths.directory)


@pytest.mark.parametrize("mutation", ("object_array", "unknown_key"))
def test_malicious_npz_payloads_are_rejected_after_valid_outer_checksums(
    tmp_path: Path,
    binary_training,
    mutation: str,
) -> None:
    _, run = binary_training
    paths = _write(run, tmp_path / mutation)
    with np.load(paths.arrays_path, allow_pickle=False) as original:
        arrays = {name: np.array(original[name], copy=True) for name in original.files}
    if mutation == "object_array":
        arrays["coefficients"] = np.asarray([object()], dtype=object)
    else:
        arrays["unexpected"] = np.asarray([1.0], dtype=np.float64)
    np.savez(paths.arrays_path, **arrays)
    payload = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    archive = paths.arrays_path.read_bytes()
    payload["arrays"]["sha256"] = _bytes_sha256(archive)
    payload["arrays"]["size_bytes"] = len(archive)
    _write_manifest(paths.manifest_path, payload, recompute_fingerprint=True)

    expected = (
        "invalid classifier array archive"
        if mutation == "object_array"
        else ("archive members")
    )
    with pytest.raises(ValueError, match=expected):
        load_frozen_classifier(paths.directory)


def test_duplicate_manifest_keys_are_rejected(tmp_path: Path, binary_training) -> None:
    _, run = binary_training
    paths = _write(run, tmp_path / "duplicate-json-key")
    raw = paths.manifest_path.read_text(encoding="utf-8")
    paths.manifest_path.write_text(
        raw.replace(
            '{"arrays":',
            '{"schema_version":"few-shot-classifier-manifest-v1.0.0","arrays":',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_frozen_classifier(paths.directory)


def test_persistence_contract_import_keeps_ml_dependencies_lazy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import biominer.ml.persistence; "
                "assert 'numpy' not in sys.modules; "
                "assert 'sklearn' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def _write(run, directory: Path, *, model_name: str | None = None):
    return write_frozen_classifier(
        run,
        directory,
        model_name=model_name,
        preprocessing_fingerprint=PREPROCESSING_FINGERPRINT,
        reference_bank_version=REFERENCE_BANK_VERSION,
        reference_bank_fingerprint=REFERENCE_BANK_FINGERPRINT,
        git_sha=GIT_SHA,
        created_at=CREATED_AT,
    )


def _write_manifest(
    path: Path,
    payload: dict[str, object],
    *,
    recompute_fingerprint: bool,
) -> None:
    if recompute_fingerprint:
        without_fingerprint = dict(payload)
        without_fingerprint.pop("classifier_fingerprint", None)
        arrays = payload["arrays"]
        assert isinstance(arrays, dict)
        payload["classifier_fingerprint"] = canonical_semantic_fingerprint(
            {
                "manifest": without_fingerprint,
                "classifier_arrays_sha256": arrays["sha256"],
            }
        )
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _bytes_sha256(value: bytes) -> str:
    import hashlib

    return f"sha256:{hashlib.sha256(value).hexdigest()}"
