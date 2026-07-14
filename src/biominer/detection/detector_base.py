from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Sequence


COARSE_DETECTOR_LABELS: tuple[str, ...] = (
    "adult_butterfly",
    "possible_adult_butterfly",
    "pinned_specimen",
    "butterfly_like",
    "moth_like",
    "caterpillar",
    "pupa",
    "insect_like",
    "artifact",
    "no_relevant_organism",
    "hard_negative",
)

LEGACY_DETECTOR_LABEL_MAP: dict[str, str] = {
    "butterfly": "butterfly_like",
    "adult_butterfly": "butterfly_like",
    "butterfly_wing": "butterfly_like",
    "butterfly_specimen": "pinned_specimen",
    "butterfly specimen": "pinned_specimen",
    "pinned_butterfly_specimen": "pinned_specimen",
    "lepidoptera": "butterfly_like",
    "moth": "moth_like",
    "larva": "caterpillar",
    "life_stage": "caterpillar",
    "chrysalis": "pupa",
    "object_proposal": "insect_like",
    "insect": "insect_like",
    "other_insect": "insect_like",
    "flower": "hard_negative",
    "leaf": "hard_negative",
    "person": "hard_negative",
    "hand": "hard_negative",
    "drawing": "artifact",
    "painting": "artifact",
    "artwork": "artifact",
    "logo": "artifact",
    "text": "artifact",
    "sign": "artifact",
    "museum_label": "artifact",
    "museum label": "artifact",
}


@dataclass(frozen=True)
class DecodedImage:
    width: int
    height: int
    mode: str
    data: bytes
    source_uri: str | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("DecodedImage width and height must be positive")
        if self.mode != "RGB":
            raise ValueError("DecodedImage currently supports RGB bytes only")
        expected = self.width * self.height * 3
        if len(self.data) != expected:
            raise ValueError(f"DecodedImage RGB data length {len(self.data)} does not match expected {expected}")


@dataclass(frozen=True)
class DetectionCandidate:
    label: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    objectness_score: float | None = None
    detector_prompt: str | None = None
    detector_class_id: int | None = None
    detector_prompt_set_fingerprint: str | None = None
    mask_polygon_xyn: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", normalize_detector_label(self.label))
        if self.detector_prompt is not None:
            object.__setattr__(self, "detector_prompt", normalize_detector_prompt(self.detector_prompt))
        if self.detector_class_id is not None:
            if isinstance(self.detector_class_id, bool) or not isinstance(self.detector_class_id, int):
                raise ValueError("detector_class_id must be a non-negative integer")
            if self.detector_class_id < 0:
                raise ValueError("detector_class_id must be a non-negative integer")
        if self.detector_prompt_set_fingerprint is not None:
            _validate_sha256_fingerprint(
                self.detector_prompt_set_fingerprint,
                name="detector_prompt_set_fingerprint",
            )
        object.__setattr__(self, "mask_polygon_xyn", normalize_mask_polygon_xyn(self.mask_polygon_xyn))


def normalize_mask_polygon_xyn(value: object) -> tuple[tuple[float, float], ...] | None:
    """Canonicalize one instance polygon in normalized original-image coordinates."""
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list | tuple):
        raise ValueError("mask_polygon_xyn must be a sequence of [x, y] points")

    points: list[tuple[float, float]] = []
    for raw_point in value:
        if hasattr(raw_point, "tolist"):
            raw_point = raw_point.tolist()
        if not isinstance(raw_point, list | tuple) or len(raw_point) != 2:
            raise ValueError("mask_polygon_xyn points must contain exactly two coordinates")
        coordinates: list[float] = []
        for raw_coordinate in raw_point:
            if isinstance(raw_coordinate, bool):
                raise ValueError("mask_polygon_xyn coordinates must be finite numbers")
            try:
                coordinate = round(float(raw_coordinate), 6)
            except (TypeError, ValueError) as exc:
                raise ValueError("mask_polygon_xyn coordinates must be finite numbers") from exc
            if not math.isfinite(coordinate):
                raise ValueError("mask_polygon_xyn coordinates must be finite numbers")
            if not 0.0 <= coordinate <= 1.0:
                raise ValueError("mask_polygon_xyn coordinates must be normalized to [0, 1]")
            coordinates.append(0.0 if coordinate == 0.0 else coordinate)
        points.append((coordinates[0], coordinates[1]))

    if not points:
        return None
    if len(points) < 3:
        raise ValueError("mask_polygon_xyn must contain at least three points")
    return tuple(points)


def normalize_detector_label(label: object) -> str:
    normalized = "_".join(str(label or "").strip().casefold().split())
    if normalized in COARSE_DETECTOR_LABELS:
        return normalized
    mapped = LEGACY_DETECTOR_LABEL_MAP.get(normalized) or LEGACY_DETECTOR_LABEL_MAP.get(str(label or "").strip().casefold())
    if mapped:
        return mapped
    if detector_label_is_taxon_like(label):
        raise ValueError(f"detector label appears taxonomic, not coarse object class: {label!r}")
    raise ValueError(f"detector label must be a BioMiner coarse object label, got {label!r}")


def normalize_detector_prompt(prompt: object) -> str:
    normalized = " ".join(str(prompt or "").strip().casefold().split())
    if not normalized:
        raise ValueError("detector prompt must be non-empty")
    return normalized


def _validate_sha256_fingerprint(value: str, *, name: str) -> None:
    prefix = "sha256:"
    digest = str(value)
    if not digest.startswith(prefix) or len(digest) != len(prefix) + 64:
        raise ValueError(f"{name} must be a sha256 fingerprint")
    try:
        int(digest[len(prefix) :], 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a sha256 fingerprint") from exc


def detector_label_is_taxon_like(label: object) -> bool:
    parts = [part for part in str(label or "").strip().split() if part]
    if len(parts) < 2:
        return False
    return parts[0][:1].isupper() and parts[1][:1].islower()


class ObjectDetector(Protocol):
    model_id: str
    model_version: str
    checkpoint: str
    backend: str

    def detect_batch(self, images: Sequence[DecodedImage]) -> list[list[DetectionCandidate]]:
        ...


class FakeObjectDetector:
    model_id = "fake-detector"
    model_version = "test"
    checkpoint = "fake-checkpoint"
    backend = "fake"

    def __init__(self, detections: Sequence[Sequence[DetectionCandidate]] = ()) -> None:
        self._detections = [list(batch) for batch in detections]

    def detect_batch(self, images: Sequence[DecodedImage]) -> list[list[DetectionCandidate]]:
        output: list[list[DetectionCandidate]] = []
        for index, _image in enumerate(images):
            output.append(list(self._detections[index]) if index < len(self._detections) else [])
        return output
