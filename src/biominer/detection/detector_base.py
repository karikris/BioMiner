from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


COARSE_DETECTOR_LABELS: tuple[str, ...] = (
    "butterfly_like",
    "moth_like",
    "caterpillar",
    "pupa",
    "insect_like",
    "hard_negative",
)

LEGACY_DETECTOR_LABEL_MAP: dict[str, str] = {
    "butterfly": "butterfly_like",
    "adult_butterfly": "butterfly_like",
    "butterfly_wing": "butterfly_like",
    "butterfly_specimen": "butterfly_like",
    "butterfly specimen": "butterfly_like",
    "pinned_butterfly_specimen": "butterfly_like",
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
    "drawing": "hard_negative",
    "painting": "hard_negative",
    "artwork": "hard_negative",
    "logo": "hard_negative",
    "text": "hard_negative",
    "sign": "hard_negative",
    "museum_label": "hard_negative",
    "museum label": "hard_negative",
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", normalize_detector_label(self.label))


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
