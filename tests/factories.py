from __future__ import annotations

import polars as pl

from biominer.detection.detector_base import DetectionCandidate


def flickr_source_record(
    photo_id: str = "photo-1",
    *,
    image_url: str | None = None,
    title: str = "monarch butterfly on milkweed",
    raw_tags: str = "monarch Danaus plexippus",
    latitude: float | None = 45.0,
    longitude: float | None = -93.0,
    date_taken: str | None = "2024-07-01",
    source_record_hash: str | None = None,
) -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "source_record_hash": source_record_hash or f"sha256:source-{photo_id}",
        "image_url": image_url or f"https://live.staticflickr.com/{photo_id}.jpg",
        "photo_page_url": f"https://www.flickr.com/photos/u/{photo_id}",
        "title": title,
        "raw_tags": raw_tags,
        "latitude": latitude,
        "longitude": longitude,
        "date_taken": date_taken,
    }


def canonical_records(*records: dict[str, object]) -> pl.DataFrame:
    return pl.DataFrame(list(records) or [flickr_source_record()])


def object_detection_row(
    photo_id: str = "photo-1",
    *,
    detection_id: str = "det-1",
    label: str = "butterfly_like",
    score: float = 0.9,
    bbox_xyxy: list[float] | None = None,
    crop_hash: str = "sha256:crop-1",
    crop_padding_ratio: float = 0.12,
    detection_status: str = "detected",
    failure_reason: str | None = None,
) -> dict[str, object]:
    bbox = bbox_xyxy or [0.0, 0.0, 10.0, 10.0]
    record = flickr_source_record(photo_id)
    return {
        "source": record["source"],
        "flickr_photo_id": record["flickr_photo_id"],
        "source_record_hash": record["source_record_hash"],
        "image_url": record["image_url"],
        "photo_page_url": record["photo_page_url"],
        "detection_id": detection_id,
        "detector_backend": "fake",
        "detector_model_id": "fake-detector",
        "detector_model_version": "v1",
        "detector_checkpoint": "checkpoint-a",
        "detected_at": "2026-01-01T00:00:00+00:00",
        "bbox_xyxy": bbox,
        "bbox_xyxyn": [0.0, 0.0, 0.5, 0.5],
        "bbox_xywhn": [0.25, 0.25, 0.5, 0.5],
        "box_area_ratio": 0.25,
        "detector_label": label,
        "detector_score": score,
        "objectness_score": score,
        "nms_group_id": None,
        "crop_padding_ratio": crop_padding_ratio,
        "crop_hash": crop_hash,
        "crop_width": 336,
        "crop_height": 336,
        "crop_storage_policy": "ephemeral",
        "detection_status": detection_status,
        "failure_reason": failure_reason,
    }


def object_detections(*rows: dict[str, object]) -> pl.DataFrame:
    return pl.DataFrame(list(rows) or [object_detection_row()])


def detection_candidate(
    label: str = "butterfly_like",
    *,
    score: float = 0.91,
    bbox_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 4.0, 4.0),
) -> DetectionCandidate:
    return DetectionCandidate(label=label, score=score, bbox_xyxy=bbox_xyxy, objectness_score=score)
