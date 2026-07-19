from __future__ import annotations

from dataclasses import dataclass, field

from biominer.detection.routing import DetectionRoutingPolicy


@dataclass(frozen=True, slots=True)
class DetectionPolicy:
    """Detector filtering and routing controls used by every detection pass."""

    backend: str = "yoloe26"
    box_score_threshold: float = 0.20
    nms_iou_threshold: float = 0.50
    min_box_area_ratio: float = 0.0005
    max_boxes_per_image: int = 8
    routing_policy: DetectionRoutingPolicy = field(default_factory=DetectionRoutingPolicy)
    image_max_side_px: int = 1280


@dataclass(frozen=True, slots=True)
class DetectionRunPolicy:
    """Bounded I/O, detector batching, and Parquet publication controls."""

    download_workers: int = 4
    max_inflight_images: int = 32
    detector_batch_size: int = 4
    parquet_batch_rows: int = 10000
    adaptive_batching: bool = False
    min_detector_batch_size: int = 1
