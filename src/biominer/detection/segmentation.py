from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from biominer.detection.cropper import CropResult


class Segmenter(Protocol):
    backend: str

    def segment_crop(self, crop: CropResult) -> bytes | None:
        ...


@dataclass(frozen=True)
class NoneSegmenter:
    backend: str = "none"

    def segment_crop(self, crop: CropResult) -> bytes | None:
        return None


class SamLikeSegmenter:
    backend = "sam"

    def __init__(self, *args: object, **kwargs: object) -> None:
        try:
            import segment_anything  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency path.
            raise RuntimeError(
                "SAM/SAM2-style segmentation requires an optional vision environment; segmentation is not required for detector crops"
            ) from exc
        raise RuntimeError("SAM/SAM2 adapter is a placeholder; use the none segmenter or provide a project-specific adapter")

    def segment_crop(self, crop: CropResult) -> bytes | None:
        return None
