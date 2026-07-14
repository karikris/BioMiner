from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from biominer.bioclip.reference_scoring import ReferenceEvidenceIndex
from biominer.ml.calibration import FrozenProbabilityCalibrator
from biominer.ml.persistence import FrozenLinearClassifier
from biominer.vision.model_state import (
    VersionedVisionModelStateCache,
    VisionModelStateRequest,
)


@dataclass
class _FakeBioClip:
    key: tuple[str, str]
    close_calls: int = 0
    attestation_calls: int = 0

    def ensure_model_attestation(self) -> None:
        self.attestation_calls += 1

    def close(self) -> None:
        self.close_calls += 1


@dataclass(frozen=True)
class _FakeReferenceIndex:
    model_fingerprint: str
    reference_embedding_fingerprint: str
    reference_prototype_fingerprint: str


@dataclass(frozen=True)
class _FakeClassifier:
    classifier_fingerprint: str
    model_fingerprint: str
    preprocessing_fingerprint: str
    reference_bank_version: str
    reference_embedding_fingerprint: str
    reference_prototype_fingerprint: str
    target_task: str = "regional_multiclass"
    route: str = "adult_field"
    class_labels: tuple[str, ...] = ("1", "2")


@dataclass(frozen=True)
class _FakeCalibrator:
    calibration_fingerprint: str
    classifier_fingerprint: str
    target_task: str = "regional_multiclass"
    route: str = "adult_field"
    class_labels: tuple[str, ...] = ("1", "2")


def test_versioned_model_state_loads_once_and_reuses_every_component(tmp_path) -> None:
    calls = _LoaderCalls()
    cache = _cache(calls)
    request = _request(tmp_path)

    first = cache.resolve(request)
    second = cache.resolve(request)

    assert first.bioclip is second.bioclip
    assert first.reference_index is second.reference_index
    assert first.classifier is second.classifier
    assert first.calibrator is second.calibrator
    assert cast(_FakeBioClip, first.bioclip).attestation_calls == 1
    assert calls.counts == {
        "bioclip": 1,
        "reference_index": 1,
        "classifier": 1,
        "calibrator": 1,
    }
    metrics = cache.metrics()
    assert metrics["model_state_resolutions"] == 2
    assert metrics["model_state_total_loads"] == 4
    assert metrics["model_state_total_cache_hits"] == 4
    assert metrics["model_state_cache_hit_rate"] == 0.5
    for component in (
        "bioclip",
        "reference_index",
        "classifier",
        "calibrator",
    ):
        assert metrics[f"{component}_loads"] == 1
        assert metrics[f"{component}_cache_hits"] == 1
        assert metrics[f"{component}_refreshes"] == 0


def test_versioned_model_state_refreshes_only_changed_artifacts(tmp_path) -> None:
    calls = _LoaderCalls()
    cache = _cache(calls)
    initial = _request(tmp_path)
    initial_state = cache.resolve(initial)

    calibration_only = _request(
        tmp_path,
        calibration_fingerprint=_fp("9"),
    )
    calibrated_state = cache.resolve(calibration_only)

    assert calibrated_state.bioclip is initial_state.bioclip
    assert calibrated_state.reference_index is initial_state.reference_index
    assert calibrated_state.classifier is initial_state.classifier
    assert calibrated_state.calibrator is not initial_state.calibrator
    assert calls.counts == {
        "bioclip": 1,
        "reference_index": 1,
        "classifier": 1,
        "calibrator": 2,
    }
    metrics = cache.metrics()
    assert metrics["calibrator_refreshes"] == 1
    assert metrics["bioclip_cache_hits"] == 1
    assert metrics["reference_index_cache_hits"] == 1
    assert metrics["classifier_cache_hits"] == 1


def test_versioned_model_state_refreshes_bound_few_shot_stack_on_bank_change(
    tmp_path,
) -> None:
    calls = _LoaderCalls()
    cache = _cache(calls)
    initial = _request(tmp_path)
    initial_state = cache.resolve(initial)
    changed = _request(
        tmp_path,
        reference_bank_version="bank-v2",
        reference_embedding_fingerprint=_fp("6"),
        reference_prototype_fingerprint=_fp("7"),
        classifier_fingerprint=_fp("8"),
        calibration_fingerprint=_fp("9"),
    )

    changed_state = cache.resolve(changed)

    assert changed_state.bioclip is initial_state.bioclip
    assert changed_state.reference_index is not initial_state.reference_index
    assert changed_state.classifier is not initial_state.classifier
    assert changed_state.calibrator is not initial_state.calibrator
    metrics = cache.metrics()
    assert metrics["bioclip_loads"] == 1
    assert metrics["bioclip_cache_hits"] == 1
    assert metrics["reference_index_loads"] == 2
    assert metrics["classifier_loads"] == 2
    assert metrics["calibrator_loads"] == 2
    assert metrics["active_reference_bank_version"] == "bank-v2"


def test_versioned_model_state_closes_bioclip_only_when_model_identity_changes(
    tmp_path,
) -> None:
    calls = _LoaderCalls()
    cache = _cache(calls)
    first = cache.resolve(_request(tmp_path))
    old_bioclip = cast(_FakeBioClip, first.bioclip)
    changed_request = _request(
        tmp_path,
        model_fingerprint=_fp("a"),
        preprocessing_fingerprint=_fp("b"),
        reference_embedding_fingerprint=_fp("c"),
        reference_prototype_fingerprint=_fp("d"),
        classifier_fingerprint=_fp("e"),
        calibration_fingerprint=_fp("f"),
    )

    second = cache.resolve(changed_request)

    assert second.bioclip is not first.bioclip
    assert old_bioclip.close_calls == 1
    assert cache.metrics()["bioclip_refreshes"] == 1
    active = cast(_FakeBioClip, second.bioclip)
    cache.close()
    assert active.close_calls == 1


def test_versioned_model_state_rejects_cross_artifact_identity_mismatch(
    tmp_path,
) -> None:
    calls = _LoaderCalls()

    def bad_classifier(request: VisionModelStateRequest) -> FrozenLinearClassifier:
        calls.counts["classifier"] += 1
        value = _fake_classifier(request)
        value = _FakeClassifier(
            **{
                **value.__dict__,
                "reference_bank_version": "wrong-bank",
            }
        )
        return cast(FrozenLinearClassifier, value)

    cache = VersionedVisionModelStateCache(
        bioclip_loader=calls.bioclip,
        reference_index_loader=calls.reference_index,
        classifier_loader=bad_classifier,
        calibrator_loader=calls.calibrator,
    )
    request = _request(tmp_path)

    with pytest.raises(ValueError, match="classifier.*reference_bank_version"):
        cache.resolve(request)
    with pytest.raises(ValueError, match="classifier.*reference_bank_version"):
        cache.resolve(request)

    assert calls.counts["classifier"] == 2


def test_model_state_progress_requires_checkpoint_before_cache_cleanup(
    tmp_path,
) -> None:
    cache = _cache(_LoaderCalls())
    cache.resolve(_request(tmp_path))

    with pytest.raises(ValueError, match="before a Parquet shard checkpoint"):
        cache.record_batch_completed(
            input_rows=10,
            output_rows=8,
            checkpointed_parquet_shards=0,
            deleted_source_cache_files=2,
        )

    cache.record_batch_completed(
        input_rows=10,
        output_rows=8,
        checkpointed_parquet_shards=2,
        deleted_source_cache_files=2,
    )
    metrics = cache.metrics()
    assert metrics["model_state_batches_completed"] == 1
    assert metrics["model_state_input_rows"] == 10
    assert metrics["model_state_output_rows"] == 8
    assert metrics["checkpointed_parquet_shards"] == 2
    assert metrics["deleted_source_cache_files"] == 2


class _LoaderCalls:
    def __init__(self) -> None:
        self.counts = {
            "bioclip": 0,
            "reference_index": 0,
            "classifier": 0,
            "calibrator": 0,
        }

    def bioclip(self, request: VisionModelStateRequest) -> object:
        self.counts["bioclip"] += 1
        return _FakeBioClip(request.bioclip_key)

    def reference_index(
        self,
        request: VisionModelStateRequest,
    ) -> ReferenceEvidenceIndex:
        self.counts["reference_index"] += 1
        return cast(
            ReferenceEvidenceIndex,
            _FakeReferenceIndex(
                model_fingerprint=request.model_fingerprint,
                reference_embedding_fingerprint=(
                    request.reference_embedding_fingerprint
                ),
                reference_prototype_fingerprint=(
                    request.reference_prototype_fingerprint
                ),
            ),
        )

    def classifier(
        self,
        request: VisionModelStateRequest,
    ) -> FrozenLinearClassifier:
        self.counts["classifier"] += 1
        return cast(FrozenLinearClassifier, _fake_classifier(request))

    def calibrator(
        self,
        request: VisionModelStateRequest,
    ) -> FrozenProbabilityCalibrator:
        self.counts["calibrator"] += 1
        return cast(
            FrozenProbabilityCalibrator,
            _FakeCalibrator(
                calibration_fingerprint=request.calibration_fingerprint,
                classifier_fingerprint=request.classifier_fingerprint,
            ),
        )


def _cache(calls: _LoaderCalls) -> VersionedVisionModelStateCache:
    return VersionedVisionModelStateCache(
        bioclip_loader=calls.bioclip,
        reference_index_loader=calls.reference_index,
        classifier_loader=calls.classifier,
        calibrator_loader=calls.calibrator,
    )


def _fp(character: str) -> str:
    return "sha256:" + character * 64


def _request(
    tmp_path: Path,
    *,
    model_fingerprint: str = _fp("0"),
    preprocessing_fingerprint: str = _fp("1"),
    reference_bank_version: str = "bank-v1",
    reference_embedding_fingerprint: str = _fp("2"),
    reference_prototype_fingerprint: str = _fp("3"),
    classifier_fingerprint: str = _fp("4"),
    calibration_fingerprint: str = _fp("5"),
) -> VisionModelStateRequest:
    return VisionModelStateRequest(
        model_fingerprint=model_fingerprint,
        preprocessing_fingerprint=preprocessing_fingerprint,
        reference_bank_version=reference_bank_version,
        reference_embeddings_path=tmp_path / "reference_embeddings.parquet",
        reference_embedding_fingerprint=reference_embedding_fingerprint,
        reference_prototypes_path=tmp_path / "reference_prototypes.parquet",
        reference_prototype_fingerprint=reference_prototype_fingerprint,
        classifier_artifact_path=tmp_path / "classifier" / classifier_fingerprint,
        classifier_fingerprint=classifier_fingerprint,
        calibrator_artifact_path=tmp_path / "calibrator" / calibration_fingerprint,
        calibration_fingerprint=calibration_fingerprint,
    )


def _fake_classifier(request: VisionModelStateRequest) -> _FakeClassifier:
    return _FakeClassifier(
        classifier_fingerprint=request.classifier_fingerprint,
        model_fingerprint=request.model_fingerprint,
        preprocessing_fingerprint=request.preprocessing_fingerprint,
        reference_bank_version=request.reference_bank_version,
        reference_embedding_fingerprint=request.reference_embedding_fingerprint,
        reference_prototype_fingerprint=request.reference_prototype_fingerprint,
    )
