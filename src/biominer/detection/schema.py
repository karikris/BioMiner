from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from typing import Any, Iterable

from biominer.detection.detector_base import DecodedImage, DetectionCandidate
from biominer.detection.policy import DetectionPolicy


DETECTION_SCHEMA_VERSION = "object-detection-v1"


def detection_id_for(
    *,
    source: str,
    flickr_photo_id: str,
    detector_checkpoint: str,
    bbox_xyxyn: Iterable[float | None],
    detector_label: str,
) -> str:
    payload = json.dumps(
        {
            "source": source,
            "flickr_photo_id": flickr_photo_id,
            "detector_checkpoint": detector_checkpoint,
            "bbox_xyxyn": [None if value is None else round(float(value), 6) for value in bbox_xyxyn],
            "detector_label": detector_label,
        },
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_detection_rows(
    *,
    record: dict[str, Any],
    image: DecodedImage,
    detections: Iterable[DetectionCandidate],
    detector_backend: str,
    detector_model_id: str,
    detector_model_version: str,
    detector_checkpoint: str,
    detected_at: datetime | str | None = None,
    policy: DetectionPolicy | None = None,
) -> list[dict[str, Any]]:
    source = str(record.get("source") or "flickr")
    photo_id = str(record.get("flickr_photo_id") or record.get("id") or "")
    if not source or not photo_id:
        raise ValueError("Detection rows require source and flickr_photo_id")
    timestamp = _timestamp(detected_at)
    kept = _filter_detections(detections, image=image, policy=policy or DetectionPolicy(backend=detector_backend))
    if not kept:
        return [
            _base_row(
                record,
                source=source,
                photo_id=photo_id,
                detector_backend=detector_backend,
                detector_model_id=detector_model_id,
                detector_model_version=detector_model_version,
                detector_checkpoint=detector_checkpoint,
                detected_at=timestamp,
                detection_id=detection_id_for(
                    source=source,
                    flickr_photo_id=photo_id,
                    detector_checkpoint=detector_checkpoint,
                    bbox_xyxyn=(None, None, None, None),
                    detector_label="no_detection",
                ),
                detection_status="no_detection",
                failure_reason="no_butterfly_like_object",
            )
        ]
    rows: list[dict[str, Any]] = []
    for candidate in kept:
        bbox = _bbox_xyxy(candidate.bbox_xyxy, image=image)
        xyxyn = _bbox_xyxyn(bbox, image=image)
        xywhn = _bbox_xywhn(xyxyn)
        rows.append(
            {
                **_base_row(
                    record,
                    source=source,
                    photo_id=photo_id,
                    detector_backend=detector_backend,
                    detector_model_id=detector_model_id,
                    detector_model_version=detector_model_version,
                    detector_checkpoint=detector_checkpoint,
                    detected_at=timestamp,
                    detection_id=detection_id_for(
                        source=source,
                        flickr_photo_id=photo_id,
                        detector_checkpoint=detector_checkpoint,
                        bbox_xyxyn=xyxyn,
                        detector_label=candidate.label,
                    ),
                    detection_status="detected",
                    failure_reason=None,
                ),
                "bbox_xyxy": bbox,
                "bbox_xyxyn": xyxyn,
                "bbox_xywhn": xywhn,
                "box_area_ratio": _box_area_ratio(bbox, image=image),
                "detector_label": candidate.label,
                "detector_score": float(candidate.score),
                "objectness_score": None if candidate.objectness_score is None else float(candidate.objectness_score),
            }
        )
    return rows


def _base_row(
    record: dict[str, Any],
    *,
    source: str,
    photo_id: str,
    detector_backend: str,
    detector_model_id: str,
    detector_model_version: str,
    detector_checkpoint: str,
    detected_at: str,
    detection_id: str,
    detection_status: str,
    failure_reason: str | None,
) -> dict[str, Any]:
    return {
        "source": source,
        "flickr_photo_id": photo_id,
        "source_record_hash": _optional_string(record.get("source_record_hash")),
        "image_url": str(record.get("image_url") or ""),
        "photo_page_url": _optional_string(record.get("photo_page_url")),
        "detection_id": detection_id,
        "detector_backend": detector_backend,
        "prediction_source": f"object_detector:{detector_backend}",
        "detector_model_id": detector_model_id,
        "detector_model_version": detector_model_version,
        "detector_checkpoint": detector_checkpoint,
        "detected_at": detected_at,
        "bbox_xyxy": [],
        "bbox_xyxyn": [],
        "bbox_xywhn": [],
        "box_area_ratio": 0.0,
        "detector_label": None,
        "detector_score": 0.0,
        "objectness_score": None,
        "nms_group_id": None,
        "crop_padding_ratio": 0.0,
        "crop_hash": None,
        "crop_width": None,
        "crop_height": None,
        "crop_storage_policy": "not_created",
        "detection_status": detection_status,
        "failure_reason": failure_reason,
        "schema_version": DETECTION_SCHEMA_VERSION,
    }


def _filter_detections(
    detections: Iterable[DetectionCandidate],
    *,
    image: DecodedImage,
    policy: DetectionPolicy,
) -> list[DetectionCandidate]:
    output: list[tuple[DetectionCandidate, list[float]]] = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        bbox = _bbox_xyxy(detection.bbox_xyxy, image=image)
        if detection.score < policy.box_score_threshold:
            continue
        if _box_area_ratio(bbox, image=image) < policy.min_box_area_ratio:
            continue
        if any(_iou_xyxy(bbox, kept_bbox) > policy.nms_iou_threshold for _kept, kept_bbox in output):
            continue
        output.append((detection, bbox))
        if len(output) >= policy.max_boxes_per_image:
            break
    return [detection for detection, _bbox in output]


def _bbox_xyxy(values: tuple[float, float, float, float], *, image: DecodedImage) -> list[float]:
    x1, y1, x2, y2 = (float(value) for value in values)
    left = _clamp(min(x1, x2), 0.0, float(image.width))
    top = _clamp(min(y1, y2), 0.0, float(image.height))
    right = _clamp(max(x1, x2), 0.0, float(image.width))
    bottom = _clamp(max(y1, y2), 0.0, float(image.height))
    return [left, top, right, bottom]


def _bbox_xyxyn(bbox: list[float], *, image: DecodedImage) -> list[float]:
    return [
        round(bbox[0] / image.width, 6),
        round(bbox[1] / image.height, 6),
        round(bbox[2] / image.width, 6),
        round(bbox[3] / image.height, 6),
    ]


def _bbox_xywhn(xyxyn: list[float]) -> list[float]:
    width = max(0.0, xyxyn[2] - xyxyn[0])
    height = max(0.0, xyxyn[3] - xyxyn[1])
    return [round(xyxyn[0] + width / 2, 6), round(xyxyn[1] + height / 2, 6), round(width, 6), round(height, 6)]


def _box_area_ratio(bbox: list[float], *, image: DecodedImage) -> float:
    area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    return round(area / (image.width * image.height), 6)


def _iou_xyxy(left: list[float], right: list[float]) -> float:
    intersection_left = max(left[0], right[0])
    intersection_top = max(left[1], right[1])
    intersection_right = min(left[2], right[2])
    intersection_bottom = min(left[3], right[3])
    intersection = max(0.0, intersection_right - intersection_left) * max(0.0, intersection_bottom - intersection_top)
    if intersection <= 0.0:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _timestamp(value: datetime | str | None) -> str:
    if isinstance(value, str):
        return value
    return (value or datetime.now(UTC)).astimezone(UTC).isoformat()


def _optional_string(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
