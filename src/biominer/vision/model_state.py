from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
import re

from biominer.bioclip.reference_scoring import ReferenceEvidenceIndex
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.ml.calibration import (
    FrozenProbabilityCalibrator,
    load_probability_calibrator,
)
from biominer.ml.persistence import FrozenLinearClassifier, load_frozen_classifier


_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CACHE_COMPONENTS = (
    "bioclip",
    "reference_index",
    "classifier",
    "calibrator",
)


@dataclass(frozen=True, slots=True)
class VisionModelStateRequest:
    model_fingerprint: str
    preprocessing_fingerprint: str
    reference_bank_version: str
    reference_embeddings_path: Path
    reference_embedding_fingerprint: str
    reference_prototypes_path: Path
    reference_prototype_fingerprint: str
    classifier_artifact_path: Path
    classifier_fingerprint: str
    calibrator_artifact_path: Path
    calibration_fingerprint: str

    def __post_init__(self) -> None:
        for field_name in (
            "model_fingerprint",
            "preprocessing_fingerprint",
            "reference_embedding_fingerprint",
            "reference_prototype_fingerprint",
            "classifier_fingerprint",
            "calibration_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _fingerprint(getattr(self, field_name), field=field_name),
            )
        object.__setattr__(
            self,
            "reference_bank_version",
            _required_text(
                self.reference_bank_version,
                field="reference_bank_version",
            ),
        )
        for field_name in (
            "reference_embeddings_path",
            "reference_prototypes_path",
            "classifier_artifact_path",
            "calibrator_artifact_path",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, (str, Path)):
                raise TypeError(f"{field_name} must be a path")
            object.__setattr__(self, field_name, Path(value))

    @property
    def state_fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "model_fingerprint": self.model_fingerprint,
                "preprocessing_fingerprint": self.preprocessing_fingerprint,
                "reference_bank_version": self.reference_bank_version,
                "reference_embedding_fingerprint": (
                    self.reference_embedding_fingerprint
                ),
                "reference_prototype_fingerprint": (
                    self.reference_prototype_fingerprint
                ),
                "classifier_fingerprint": self.classifier_fingerprint,
                "calibration_fingerprint": self.calibration_fingerprint,
            }
        )

    @property
    def bioclip_key(self) -> tuple[str, str]:
        return (self.model_fingerprint, self.preprocessing_fingerprint)

    @property
    def reference_index_key(self) -> tuple[str, ...]:
        return (
            self.reference_bank_version,
            self.reference_embedding_fingerprint,
            self.reference_prototype_fingerprint,
            *self.bioclip_key,
        )

    @property
    def classifier_key(self) -> tuple[str, ...]:
        return (
            self.classifier_fingerprint,
            *self.reference_index_key,
        )

    @property
    def calibrator_key(self) -> tuple[str, ...]:
        return (
            self.calibration_fingerprint,
            *self.classifier_key,
        )


@dataclass(frozen=True, slots=True)
class LoadedVisionModelState:
    state_fingerprint: str
    bioclip: object
    reference_index: ReferenceEvidenceIndex
    classifier: FrozenLinearClassifier
    calibrator: FrozenProbabilityCalibrator


@dataclass(slots=True)
class _ComponentMetrics:
    loads: int = 0
    cache_hits: int = 0
    refreshes: int = 0


BioClipLoader = Callable[[VisionModelStateRequest], object]
ReferenceIndexLoader = Callable[[VisionModelStateRequest], ReferenceEvidenceIndex]
ClassifierLoader = Callable[[VisionModelStateRequest], FrozenLinearClassifier]
CalibratorLoader = Callable[
    [VisionModelStateRequest],
    FrozenProbabilityCalibrator,
]


class VersionedVisionModelStateCache:
    """Retain immutable inference state and refresh only changed components."""

    def __init__(
        self,
        *,
        bioclip_loader: BioClipLoader,
        reference_index_loader: ReferenceIndexLoader | None = None,
        classifier_loader: ClassifierLoader | None = None,
        calibrator_loader: CalibratorLoader | None = None,
    ) -> None:
        if not callable(bioclip_loader):
            raise TypeError("bioclip_loader must be callable")
        self._bioclip_loader = bioclip_loader
        self._reference_index_loader = reference_index_loader or _load_reference_index
        self._classifier_loader = classifier_loader or _load_classifier
        self._calibrator_loader = calibrator_loader or _load_calibrator
        self._slots: dict[str, tuple[tuple[str, ...], object]] = {}
        self._metrics = {
            component: _ComponentMetrics() for component in _CACHE_COMPONENTS
        }
        self._state_resolutions = 0
        self._batches_completed = 0
        self._input_rows = 0
        self._output_rows = 0
        self._checkpointed_parquet_shards = 0
        self._deleted_source_cache_files = 0
        self._active_request: VisionModelStateRequest | None = None
        self._lock = Lock()

    def __enter__(self) -> VersionedVisionModelStateCache:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        self.close()

    def resolve(
        self,
        request: VisionModelStateRequest,
    ) -> LoadedVisionModelState:
        if not isinstance(request, VisionModelStateRequest):
            raise TypeError("request must be a VisionModelStateRequest")
        with self._lock:
            try:
                bioclip = self._resolve_component(
                    "bioclip",
                    request.bioclip_key,
                    lambda: _load_bioclip_state(
                        self._bioclip_loader,
                        request,
                    ),
                    close_before_refresh=True,
                )
                reference_index = self._resolve_component(
                    "reference_index",
                    request.reference_index_key,
                    lambda: self._reference_index_loader(request),
                )
                classifier = self._resolve_component(
                    "classifier",
                    request.classifier_key,
                    lambda: self._classifier_loader(request),
                )
                calibrator = self._resolve_component(
                    "calibrator",
                    request.calibrator_key,
                    lambda: self._calibrator_loader(request),
                )
                _validate_loaded_state(
                    request,
                    reference_index=reference_index,
                    classifier=classifier,
                    calibrator=calibrator,
                )
            except BaseException:
                for component in ("reference_index", "classifier", "calibrator"):
                    self._slots.pop(component, None)
                self._active_request = None
                raise
            self._state_resolutions += 1
            self._active_request = request
            return LoadedVisionModelState(
                state_fingerprint=request.state_fingerprint,
                bioclip=bioclip,
                reference_index=reference_index,
                classifier=classifier,
                calibrator=calibrator,
            )

    def record_batch_completed(
        self,
        *,
        input_rows: int,
        output_rows: int,
        checkpointed_parquet_shards: int,
        deleted_source_cache_files: int,
    ) -> None:
        values = {
            "input_rows": input_rows,
            "output_rows": output_rows,
            "checkpointed_parquet_shards": checkpointed_parquet_shards,
            "deleted_source_cache_files": deleted_source_cache_files,
        }
        for field_name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if deleted_source_cache_files and not checkpointed_parquet_shards:
            raise ValueError(
                "source cache files cannot be deleted before a Parquet shard checkpoint"
            )
        with self._lock:
            self._batches_completed += 1
            self._input_rows += input_rows
            self._output_rows += output_rows
            self._checkpointed_parquet_shards += checkpointed_parquet_shards
            self._deleted_source_cache_files += deleted_source_cache_files

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            payload: dict[str, Any] = {
                "model_state_resolutions": self._state_resolutions,
                "model_state_batches_completed": self._batches_completed,
                "model_state_input_rows": self._input_rows,
                "model_state_output_rows": self._output_rows,
                "checkpointed_parquet_shards": (self._checkpointed_parquet_shards),
                "deleted_source_cache_files": self._deleted_source_cache_files,
            }
            total_loads = 0
            total_hits = 0
            for component in _CACHE_COMPONENTS:
                metrics = self._metrics[component]
                payload[f"{component}_loads"] = metrics.loads
                payload[f"{component}_cache_hits"] = metrics.cache_hits
                payload[f"{component}_refreshes"] = metrics.refreshes
                payload[f"{component}_cache_hit_rate"] = _ratio(
                    metrics.cache_hits,
                    metrics.cache_hits + metrics.loads,
                )
                total_loads += metrics.loads
                total_hits += metrics.cache_hits
            payload["model_state_total_loads"] = total_loads
            payload["model_state_total_cache_hits"] = total_hits
            payload["model_state_cache_hit_rate"] = _ratio(
                total_hits,
                total_hits + total_loads,
            )
            active = self._active_request
            payload["active_model_state_fingerprint"] = (
                active.state_fingerprint if active is not None else None
            )
            payload["active_reference_bank_version"] = (
                active.reference_bank_version if active is not None else None
            )
            payload["active_classifier_fingerprint"] = (
                active.classifier_fingerprint if active is not None else None
            )
            payload["active_calibration_fingerprint"] = (
                active.calibration_fingerprint if active is not None else None
            )
            return payload

    def close(self) -> None:
        with self._lock:
            slot = self._slots.pop("bioclip", None)
            if slot is not None:
                _close_resource(slot[1])
            self._slots.clear()
            self._active_request = None

    def _resolve_component(
        self,
        component: str,
        key: tuple[str, ...],
        loader: Callable[[], object],
        *,
        close_before_refresh: bool = False,
    ) -> Any:
        current = self._slots.get(component)
        metrics = self._metrics[component]
        if current is not None and current[0] == key:
            metrics.cache_hits += 1
            return current[1]
        if current is not None:
            metrics.refreshes += 1
            if close_before_refresh:
                _close_resource(current[1])
                self._slots.pop(component, None)
        loaded = loader()
        if loaded is None:
            raise RuntimeError(f"{component} loader returned no state")
        self._slots[component] = (key, loaded)
        metrics.loads += 1
        return loaded


def _load_reference_index(
    request: VisionModelStateRequest,
) -> ReferenceEvidenceIndex:
    return ReferenceEvidenceIndex(
        request.reference_embeddings_path,
        request.reference_prototypes_path,
    )


def _load_bioclip_state(
    loader: BioClipLoader,
    request: VisionModelStateRequest,
) -> object:
    loaded = loader(request)
    if loaded is None:
        raise RuntimeError("bioclip loader returned no state")
    try:
        ensure_attestation = getattr(loaded, "ensure_model_attestation", None)
        if callable(ensure_attestation):
            ensure_attestation()
        reported_model_fingerprint = getattr(loaded, "model_fingerprint", None)
        if (
            reported_model_fingerprint is not None
            and reported_model_fingerprint != request.model_fingerprint
        ):
            raise ValueError("loaded BioCLIP model fingerprint does not match request")
        reported_preprocessing_fingerprint = getattr(
            loaded,
            "preprocessing_fingerprint",
            None,
        )
        if (
            reported_preprocessing_fingerprint is not None
            and reported_preprocessing_fingerprint != request.preprocessing_fingerprint
        ):
            raise ValueError(
                "loaded BioCLIP preprocessing fingerprint does not match request"
            )
    except BaseException:
        _close_resource(loaded)
        raise
    return loaded


def _load_classifier(request: VisionModelStateRequest) -> FrozenLinearClassifier:
    return load_frozen_classifier(
        request.classifier_artifact_path,
        expected_classifier_fingerprint=request.classifier_fingerprint,
        expected_model_fingerprint=request.model_fingerprint,
        expected_preprocessing_fingerprint=request.preprocessing_fingerprint,
    )


def _load_calibrator(
    request: VisionModelStateRequest,
) -> FrozenProbabilityCalibrator:
    loaded = load_probability_calibrator(
        request.calibrator_artifact_path,
        expected_calibration_fingerprint=request.calibration_fingerprint,
        expected_classifier_fingerprint=request.classifier_fingerprint,
    )
    return loaded.calibrator


def _validate_loaded_state(
    request: VisionModelStateRequest,
    *,
    reference_index: object,
    classifier: object,
    calibrator: object,
) -> None:
    expected_reference = {
        "model_fingerprint": request.model_fingerprint,
        "reference_embedding_fingerprint": (request.reference_embedding_fingerprint),
        "reference_prototype_fingerprint": (request.reference_prototype_fingerprint),
    }
    _match_attributes(reference_index, expected_reference, component="reference index")
    expected_classifier = {
        "classifier_fingerprint": request.classifier_fingerprint,
        "model_fingerprint": request.model_fingerprint,
        "preprocessing_fingerprint": request.preprocessing_fingerprint,
        "reference_bank_version": request.reference_bank_version,
        "reference_embedding_fingerprint": (request.reference_embedding_fingerprint),
        "reference_prototype_fingerprint": (request.reference_prototype_fingerprint),
    }
    _match_attributes(classifier, expected_classifier, component="classifier")
    expected_calibrator = {
        "calibration_fingerprint": request.calibration_fingerprint,
        "classifier_fingerprint": request.classifier_fingerprint,
        "target_task": getattr(classifier, "target_task", None),
        "route": getattr(classifier, "route", None),
        "class_labels": getattr(classifier, "class_labels", None),
    }
    _match_attributes(calibrator, expected_calibrator, component="calibrator")


def _match_attributes(
    value: object,
    expected: Mapping[str, object],
    *,
    component: str,
) -> None:
    mismatches = [
        field_name
        for field_name, expected_value in expected.items()
        if getattr(value, field_name, None) != expected_value
    ]
    if mismatches:
        raise ValueError(
            f"loaded {component} does not match requested identity: "
            + ", ".join(sorted(mismatches))
        )


def _close_resource(value: object) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _fingerprint(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a sha256 fingerprint")
    return text


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


__all__ = [
    "LoadedVisionModelState",
    "VersionedVisionModelStateCache",
    "VisionModelStateRequest",
]
