from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


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
