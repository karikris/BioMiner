"""Canonical detector and routing identity for every production entry path."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
import re

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.detection.policy import DetectionPolicy


DETECTOR_ROUTE_CONTRACT_SCHEMA_VERSION = "detector-route-contract-v1.0.0"
DETECTOR_ROUTE_CONTRACT_VERSION = "canonical-yoloe-routing-v1"
DETECTOR_EXECUTION_MODES = frozenset(
    {"in_process", "persistent_sidecar", "injected"}
)

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FIELDS = frozenset(
    {
        "schema_version",
        "contract_version",
        "backend",
        "model_id",
        "model_version",
        "checkpoint",
        "prompt_classes",
        "prompt_set_fingerprint",
        "execution_mode",
        "transport",
        "detector_image_size",
        "detector_confidence_threshold",
        "detector_nms_iou_threshold",
        "detector_max_detections",
        "policy_box_score_threshold",
        "policy_nms_iou_threshold",
        "policy_min_box_area_ratio",
        "policy_max_boxes_per_image",
        "routing_policy_version",
        "routing_policy_fingerprint",
        "contract_fingerprint",
    }
)


@dataclass(frozen=True, slots=True)
class DetectorRouteContract:
    """Complete immutable identity for detector execution and route decisions."""

    schema_version: str
    contract_version: str
    backend: str
    model_id: str
    model_version: str
    checkpoint: str
    prompt_classes: tuple[str, ...]
    prompt_set_fingerprint: str | None
    execution_mode: str
    transport: str | None
    detector_image_size: int | None
    detector_confidence_threshold: float | None
    detector_nms_iou_threshold: float | None
    detector_max_detections: int | None
    policy_box_score_threshold: float
    policy_nms_iou_threshold: float
    policy_min_box_area_ratio: float
    policy_max_boxes_per_image: int
    routing_policy_version: str
    routing_policy_fingerprint: str

    def __post_init__(self) -> None:
        if self.schema_version != DETECTOR_ROUTE_CONTRACT_SCHEMA_VERSION:
            raise ValueError("unsupported detector route contract schema")
        if self.contract_version != DETECTOR_ROUTE_CONTRACT_VERSION:
            raise ValueError("unsupported detector route contract version")
        for field in (
            "backend",
            "model_id",
            "model_version",
            "checkpoint",
            "routing_policy_version",
        ):
            object.__setattr__(
                self, field, _required_text(getattr(self, field), field=field)
            )
        prompts = _canonical_prompts(self.prompt_classes)
        object.__setattr__(self, "prompt_classes", prompts)
        prompt_fingerprint = _optional_sha256(
            self.prompt_set_fingerprint, field="prompt_set_fingerprint"
        )
        object.__setattr__(self, "prompt_set_fingerprint", prompt_fingerprint)
        if self.execution_mode not in DETECTOR_EXECUTION_MODES:
            raise ValueError("unsupported detector execution mode")
        transport = _optional_text(self.transport, field="transport")
        object.__setattr__(self, "transport", transport)
        if self.execution_mode == "persistent_sidecar" and transport is None:
            raise ValueError("persistent sidecar contract requires a transport")
        if self.execution_mode != "persistent_sidecar" and transport is not None:
            raise ValueError("only a persistent sidecar contract may name a transport")
        for field in ("detector_image_size", "detector_max_detections"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _positive_int(value, field=field))
        for field in (
            "detector_confidence_threshold",
            "detector_nms_iou_threshold",
            "policy_box_score_threshold",
            "policy_nms_iou_threshold",
            "policy_min_box_area_ratio",
        ):
            object.__setattr__(
                self,
                field,
                _optional_unit_float(getattr(self, field), field=field)
                if field.startswith("detector_")
                else _unit_float(getattr(self, field), field=field),
            )
        object.__setattr__(
            self,
            "policy_max_boxes_per_image",
            _positive_int(
                self.policy_max_boxes_per_image,
                field="policy_max_boxes_per_image",
            ),
        )
        _sha256(
            self.routing_policy_fingerprint,
            field="routing_policy_fingerprint",
        )
        self._validate_yoloe_identity()

    def _validate_yoloe_identity(self) -> None:
        if self.backend != "yoloe26":
            return
        if not self.model_id.startswith("yoloe26:"):
            raise ValueError("YOLOE-26 model_id must start with yoloe26:")
        if not self.prompt_classes or self.prompt_set_fingerprint is None:
            raise ValueError("YOLOE-26 contract requires prompts and fingerprint")
        from biominer.detection.yoloe26_detector import (
            yoloe26_prompt_set_fingerprint,
        )

        if self.prompt_set_fingerprint != yoloe26_prompt_set_fingerprint(
            self.prompt_classes
        ):
            raise ValueError("YOLOE-26 prompt-set fingerprint does not match prompts")
        if self.execution_mode not in {"in_process", "persistent_sidecar"}:
            raise ValueError("YOLOE-26 contract requires an explicit runtime mode")
        required_detector_values = (
            self.detector_image_size,
            self.detector_confidence_threshold,
            self.detector_nms_iou_threshold,
            self.detector_max_detections,
        )
        if any(value is None for value in required_detector_values):
            raise ValueError("YOLOE-26 contract requires complete detector settings")
        if self.detector_confidence_threshold != self.policy_box_score_threshold:
            raise ValueError("YOLOE-26 detector and policy confidence thresholds differ")
        if self.detector_nms_iou_threshold != self.policy_nms_iou_threshold:
            raise ValueError("YOLOE-26 detector and policy NMS thresholds differ")
        if self.detector_max_detections != self.policy_max_boxes_per_image:
            raise ValueError("YOLOE-26 detector and policy maximum detections differ")

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(self.identity_payload())

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_version": self.contract_version,
            "backend": self.backend,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "checkpoint": self.checkpoint,
            "prompt_classes": list(self.prompt_classes),
            "prompt_set_fingerprint": self.prompt_set_fingerprint,
            "execution_mode": self.execution_mode,
            "transport": self.transport,
            "detector_image_size": self.detector_image_size,
            "detector_confidence_threshold": self.detector_confidence_threshold,
            "detector_nms_iou_threshold": self.detector_nms_iou_threshold,
            "detector_max_detections": self.detector_max_detections,
            "policy_box_score_threshold": self.policy_box_score_threshold,
            "policy_nms_iou_threshold": self.policy_nms_iou_threshold,
            "policy_min_box_area_ratio": self.policy_min_box_area_ratio,
            "policy_max_boxes_per_image": self.policy_max_boxes_per_image,
            "routing_policy_version": self.routing_policy_version,
            "routing_policy_fingerprint": self.routing_policy_fingerprint,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "contract_fingerprint": self.fingerprint,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> DetectorRouteContract:
        if not isinstance(values, Mapping):
            raise TypeError("detector route contract must be a mapping")
        _require_exact_fields(values, set(_FIELDS))
        contract = cls(
            schema_version=_required_text(
                values["schema_version"], field="schema_version"
            ),
            contract_version=_required_text(
                values["contract_version"], field="contract_version"
            ),
            backend=_required_text(values["backend"], field="backend"),
            model_id=_required_text(values["model_id"], field="model_id"),
            model_version=_required_text(
                values["model_version"], field="model_version"
            ),
            checkpoint=_required_text(values["checkpoint"], field="checkpoint"),
            prompt_classes=_sequence_of_text(
                values["prompt_classes"], field="prompt_classes"
            ),
            prompt_set_fingerprint=_optional_text(
                values["prompt_set_fingerprint"], field="prompt_set_fingerprint"
            ),
            execution_mode=_required_text(
                values["execution_mode"], field="execution_mode"
            ),
            transport=_optional_text(values["transport"], field="transport"),
            detector_image_size=_optional_int(
                values["detector_image_size"], field="detector_image_size"
            ),
            detector_confidence_threshold=_optional_number(
                values["detector_confidence_threshold"],
                field="detector_confidence_threshold",
            ),
            detector_nms_iou_threshold=_optional_number(
                values["detector_nms_iou_threshold"],
                field="detector_nms_iou_threshold",
            ),
            detector_max_detections=_optional_int(
                values["detector_max_detections"],
                field="detector_max_detections",
            ),
            policy_box_score_threshold=_number(
                values["policy_box_score_threshold"],
                field="policy_box_score_threshold",
            ),
            policy_nms_iou_threshold=_number(
                values["policy_nms_iou_threshold"],
                field="policy_nms_iou_threshold",
            ),
            policy_min_box_area_ratio=_number(
                values["policy_min_box_area_ratio"],
                field="policy_min_box_area_ratio",
            ),
            policy_max_boxes_per_image=_integer(
                values["policy_max_boxes_per_image"],
                field="policy_max_boxes_per_image",
            ),
            routing_policy_version=_required_text(
                values["routing_policy_version"],
                field="routing_policy_version",
            ),
            routing_policy_fingerprint=_required_text(
                values["routing_policy_fingerprint"],
                field="routing_policy_fingerprint",
            ),
        )
        if values["contract_fingerprint"] != contract.fingerprint:
            raise ValueError("detector route contract fingerprint mismatch")
        return contract


def build_detector_route_contract(
    detector: object,
    policy: DetectionPolicy,
) -> DetectorRouteContract:
    """Bind an object/dictionary detector to one canonical routing policy."""

    if not isinstance(policy, DetectionPolicy):
        raise TypeError("policy must be a DetectionPolicy")
    backend = _detector_value(detector, "backend")
    if backend != policy.backend:
        raise ValueError("detector backend and detection policy backend differ")
    execution_mode = _optional_detector_value(detector, "execution_mode") or (
        "injected" if backend != "yoloe26" else "persistent_sidecar"
    )
    prompt_classes = _detector_sequence(detector, "prompt_classes")
    return DetectorRouteContract(
        schema_version=DETECTOR_ROUTE_CONTRACT_SCHEMA_VERSION,
        contract_version=DETECTOR_ROUTE_CONTRACT_VERSION,
        backend=backend,
        model_id=_detector_value(detector, "model_id"),
        model_version=_detector_value(detector, "model_version"),
        checkpoint=_detector_value(detector, "checkpoint"),
        prompt_classes=prompt_classes,
        prompt_set_fingerprint=_optional_detector_value(
            detector, "prompt_set_fingerprint"
        ),
        execution_mode=execution_mode,
        transport=_optional_detector_value(detector, "transport"),
        detector_image_size=_optional_detector_int(detector, "imgsz"),
        detector_confidence_threshold=_optional_detector_number(detector, "conf"),
        detector_nms_iou_threshold=_optional_detector_number(detector, "iou"),
        detector_max_detections=_optional_detector_int(detector, "max_det"),
        policy_box_score_threshold=policy.box_score_threshold,
        policy_nms_iou_threshold=policy.nms_iou_threshold,
        policy_min_box_area_ratio=policy.min_box_area_ratio,
        policy_max_boxes_per_image=policy.max_boxes_per_image,
        routing_policy_version=policy.routing_policy.version,
        routing_policy_fingerprint=policy.routing_policy.fingerprint,
    )


def _detector_value(detector: object, field: str) -> str:
    value = _raw_detector_value(detector, field)
    return _required_text(value, field=f"detector.{field}")


def _optional_detector_value(detector: object, field: str) -> str | None:
    return _optional_text(_raw_detector_value(detector, field), field=f"detector.{field}")


def _optional_detector_int(detector: object, field: str) -> int | None:
    return _optional_int(_raw_detector_value(detector, field), field=f"detector.{field}")


def _optional_detector_number(detector: object, field: str) -> float | None:
    return _optional_number(
        _raw_detector_value(detector, field), field=f"detector.{field}"
    )


def _detector_sequence(detector: object, field: str) -> tuple[str, ...]:
    value = _raw_detector_value(detector, field)
    return () if value is None else _sequence_of_text(value, field=f"detector.{field}")


def _raw_detector_value(detector: object, field: str) -> object:
    return detector.get(field) if isinstance(detector, Mapping) else getattr(detector, field, None)


def _canonical_prompts(values: tuple[str, ...]) -> tuple[str, ...]:
    prompts = tuple(_required_text(value, field="prompt") for value in values)
    if len(prompts) != len(set(prompts)):
        raise ValueError("detector prompt classes contain duplicates")
    return prompts


def _sequence_of_text(value: object, *, field: str) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, list | tuple):
        raise TypeError(f"{field} must be a sequence")
    return tuple(_required_text(item, field=field) for item in value)


def _require_exact_fields(values: Mapping[str, object], expected: set[str]) -> None:
    missing = expected - set(values)
    unexpected = set(values) - expected
    if missing or unexpected:
        raise ValueError(
            "detector route contract fields mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None or value == "":
        return None
    return _required_text(value, field=field)


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _optional_number(value: object, *, field: str) -> float | None:
    return None if value is None else _number(value, field=field)


def _unit_float(value: object, *, field: str) -> float:
    result = _number(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be within [0, 1]")
    return result


def _optional_unit_float(value: object, *, field: str) -> float | None:
    return None if value is None else _unit_float(value, field=field)


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _optional_int(value: object, *, field: str) -> int | None:
    return None if value is None else _integer(value, field=field)


def _positive_int(value: object, *, field: str) -> int:
    result = _integer(value, field=field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a canonical SHA-256 fingerprint")
    return text


def _optional_sha256(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, field=field)


__all__ = [
    "DETECTOR_EXECUTION_MODES",
    "DETECTOR_ROUTE_CONTRACT_SCHEMA_VERSION",
    "DETECTOR_ROUTE_CONTRACT_VERSION",
    "DetectorRouteContract",
    "build_detector_route_contract",
]
