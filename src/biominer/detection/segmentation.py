from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Protocol

from biominer.detection.cropper import CropResult


DETECTOR_CROP_MASK_KEYS = (
    "detector_crop_mask",
    "detector_crop_mask_json",
    "segmentation_crop_mask",
    "segmentation_crop_mask_json",
)
DETECTOR_CROP_RGB_KEYS = (
    "detector_crop_segmentation_rgb",
    "segmentation_crop_rgb",
)


class SegmentationUnavailable(RuntimeError):
    pass


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


def make_segmenter(backend: str = "none") -> Segmenter:
    normalized = backend.strip().casefold()
    if normalized == "none":
        return NoneSegmenter()
    if normalized in {"sam", "sam2"}:
        return SamLikeSegmenter()
    raise ValueError(f"unknown segmenter backend {backend!r}; expected one of: none, sam, sam2")


def detector_crop_mask_available(item: dict[str, Any]) -> bool:
    return any(item.get(key) not in (None, "", []) for key in (*DETECTOR_CROP_MASK_KEYS, *DETECTOR_CROP_RGB_KEYS))


def detector_masked_crop_bytes(item: dict[str, Any], crop: CropResult) -> bytes | None:
    for key in DETECTOR_CROP_RGB_KEYS:
        value = item.get(key)
        if value in (None, ""):
            continue
        data = bytes(value) if isinstance(value, bytearray) else value
        if not isinstance(data, bytes):
            raise SegmentationUnavailable(f"{key}_must_be_bytes")
        _validate_rgb_bytes(data, crop=crop, key=key)
        return data

    for key in DETECTOR_CROP_MASK_KEYS:
        value = item.get(key)
        if value in (None, "", []):
            continue
        mask = _mask_values(value, key=key)
        return _apply_mask(crop.encoded_bytes, mask=mask, crop=crop, key=key)
    return None


def _mask_values(value: object, *, key: str) -> list[bool]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SegmentationUnavailable(f"{key}_invalid_json") from exc
    if not isinstance(value, list | tuple):
        raise SegmentationUnavailable(f"{key}_must_be_list")
    flattened: list[object] = []
    for item in value:
        if isinstance(item, list | tuple):
            flattened.extend(item)
        else:
            flattened.append(item)
    return [bool(item) for item in flattened]


def _apply_mask(data: bytes, *, mask: list[bool], crop: CropResult, key: str) -> bytes:
    expected_pixels = crop.crop_width * crop.crop_height
    if len(mask) != expected_pixels:
        raise SegmentationUnavailable(f"{key}_pixel_count_mismatch")
    _validate_rgb_bytes(data, crop=crop, key="crop")
    output = bytearray(data)
    for index, keep in enumerate(mask):
        if keep:
            continue
        offset = index * 3
        output[offset : offset + 3] = b"\x00\x00\x00"
    return bytes(output)


def _validate_rgb_bytes(data: bytes, *, crop: CropResult, key: str) -> None:
    expected = crop.crop_width * crop.crop_height * 3
    if len(data) != expected:
        raise SegmentationUnavailable(f"{key}_rgb_byte_count_mismatch")
