from __future__ import annotations

from dataclasses import dataclass
import hashlib

from biominer.detection.detector_base import DecodedImage


@dataclass(frozen=True)
class CropResult:
    encoded_bytes: bytes
    crop_hash: str
    crop_width: int
    crop_height: int
    clamped_bbox_xyxy: list[float]
    padded_bbox_xyxy: list[float]
    storage_policy: str = "ephemeral"


def crop_with_padding(
    image: DecodedImage,
    bbox_xyxy: tuple[float, float, float, float],
    padding_ratio: float,
    target_px: int,
) -> CropResult:
    if target_px <= 0:
        raise ValueError("target_px must be positive")
    clamped = _clamped_bbox(image, bbox_xyxy)
    padded = _padded_bbox(image, bbox_xyxy, padding_ratio=padding_ratio)
    resized = _resize_nearest(image, padded, target_px=target_px)
    digest = "sha256:" + hashlib.sha256(resized).hexdigest()
    return CropResult(
        encoded_bytes=resized,
        crop_hash=digest,
        crop_width=target_px,
        crop_height=target_px,
        clamped_bbox_xyxy=clamped,
        padded_bbox_xyxy=padded,
    )


def _clamped_bbox(image: DecodedImage, bbox: tuple[float, float, float, float]) -> list[float]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    left = _clamp(min(x1, x2), 0.0, float(image.width))
    top = _clamp(min(y1, y2), 0.0, float(image.height))
    right = _clamp(max(x1, x2), 0.0, float(image.width))
    bottom = _clamp(max(y1, y2), 0.0, float(image.height))
    if right <= left:
        right = min(float(image.width), left + 1.0)
    if bottom <= top:
        bottom = min(float(image.height), top + 1.0)
    return [_round(left), _round(top), _round(right), _round(bottom)]


def _padded_bbox(image: DecodedImage, bbox: tuple[float, float, float, float], *, padding_ratio: float) -> list[float]:
    clamped = _clamped_bbox(image, bbox)
    raw_width = abs(float(bbox[2]) - float(bbox[0]))
    raw_height = abs(float(bbox[3]) - float(bbox[1]))
    pad_x = raw_width * max(0.0, padding_ratio)
    pad_y = raw_height * max(0.0, padding_ratio)
    left = _clamp(clamped[0] - pad_x, 0.0, float(image.width))
    top = _clamp(clamped[1] - pad_y, 0.0, float(image.height))
    right = _clamp(clamped[2] + pad_x, 0.0, float(image.width))
    bottom = _clamp(clamped[3] + pad_y, 0.0, float(image.height))
    return [_round(left), _round(top), _round(right), _round(bottom)]


def _resize_nearest(image: DecodedImage, bbox: list[float], *, target_px: int) -> bytes:
    left, top, right, bottom = bbox
    width = max(1e-9, right - left)
    height = max(1e-9, bottom - top)
    if width >= height:
        content_width = target_px
        content_height = max(1, round(target_px * (height / width)))
    else:
        content_height = target_px
        content_width = max(1, round(target_px * (width / height)))
    x_offset = (target_px - content_width) // 2
    y_offset = (target_px - content_height) // 2
    output = bytearray(bytes(target_px * target_px * 3))
    for y in range(content_height):
        source_y = top + ((y + 0.5) / content_height) * height
        pixel_y = min(image.height - 1, max(0, int(source_y)))
        for x in range(content_width):
            source_x = left + ((x + 0.5) / content_width) * width
            pixel_x = min(image.width - 1, max(0, int(source_x)))
            source_offset = (pixel_y * image.width + pixel_x) * 3
            target_offset = ((y + y_offset) * target_px + (x + x_offset)) * 3
            output[target_offset : target_offset + 3] = image.data[source_offset : source_offset + 3]
    return bytes(output)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _round(value: float) -> float:
    return round(value, 6)
