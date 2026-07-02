from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionPolicy:
    backend: str = "yolo"
    box_score_threshold: float = 0.20
    nms_iou_threshold: float = 0.50
    min_box_area_ratio: float = 0.0005
    max_boxes_per_image: int = 8
    crop_padding_ratio: float = 0.12
    image_max_side_px: int = 1280
    crop_target_px: int = 336
    retain_debug_crops: bool = False
    debug_crop_limit: int = 500


@dataclass(frozen=True)
class DetectionRunPolicy:
    download_workers: int = 4
    decode_workers: int = 4
    detector_workers: int = 1
    max_inflight_images: int = 32
    max_inflight_crops: int = 96
    detector_batch_size: int = 4
    crop_batch_size: int = 24
    parquet_batch_rows: int = 10000
