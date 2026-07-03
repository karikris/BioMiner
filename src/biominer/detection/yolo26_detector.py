from __future__ import annotations

from pathlib import Path
from typing import Sequence

from biominer.detection.detector_base import DecodedImage, DetectionCandidate, normalize_detector_label


DEFAULT_YOLO26_IMGSZ = 640
DEFAULT_YOLO26_CONF = 0.20
DEFAULT_YOLO26_IOU = 0.50
DEFAULT_YOLO26_MAX_DET = 8


class Yolo26ObjectDetector:
    """YOLO26-compatible inference adapter for user-provided coarse object checkpoints."""

    backend = "yolo26"

    def __init__(
        self,
        *,
        checkpoint: str,
        device: str = "auto",
        imgsz: int = DEFAULT_YOLO26_IMGSZ,
        conf: float = DEFAULT_YOLO26_CONF,
        iou: float = DEFAULT_YOLO26_IOU,
        max_det: int = DEFAULT_YOLO26_MAX_DET,
    ) -> None:
        checkpoint = str(checkpoint or "").strip()
        if not checkpoint:
            raise ValueError("YOLO26 inference requires an explicit user-provided checkpoint")
        try:
            import ultralytics
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - optional dependency path.
            raise RuntimeError("ultralytics is required for YOLO26 inference; install the optional vision runtime") from exc

        self.checkpoint = checkpoint
        self.device = None if device == "auto" else device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.model_id = f"yolo26:{Path(checkpoint).stem}"
        self.model_version = f"ultralytics:{getattr(ultralytics, '__version__', 'unknown')}"
        self._model = YOLO(checkpoint)

    def detect_batch(self, images: Sequence[DecodedImage]) -> list[list[DetectionCandidate]]:
        if not images:
            return []
        pil_images = [_decoded_image_to_pil(image) for image in images]
        results = self._model.predict(
            pil_images,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            verbose=False,
        )
        if not isinstance(results, list | tuple):
            results = [results]
        return [detections_from_yolo26_result(result) for result in results]


def detections_from_yolo26_result(result: object) -> list[DetectionCandidate]:
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    rows: list[DetectionCandidate] = []
    for box in boxes:
        xyxy = _first_vector(getattr(box, "xyxy", ()))
        if len(xyxy) != 4:
            continue
        cls_id = int(_first_scalar(getattr(box, "cls", -1), default=-1))
        raw_label = str(names.get(cls_id, cls_id)) if isinstance(names, dict) else str(cls_id)
        score = float(_first_scalar(getattr(box, "conf", 0.0), default=0.0))
        rows.append(
            DetectionCandidate(
                label=yolo26_coarse_label(raw_label),
                score=score,
                bbox_xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                objectness_score=score,
            )
        )
    return rows


def yolo26_coarse_label(label: object) -> str:
    try:
        return normalize_detector_label(label)
    except ValueError:
        return "insect_like"


def _decoded_image_to_pil(image: DecodedImage) -> object:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency path.
        raise RuntimeError("Pillow is required to prepare decoded images for YOLO26") from exc
    return Image.frombytes("RGB", (image.width, image.height), image.data)


def _first_vector(value: object) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list | tuple) and value and isinstance(value[0], list | tuple):
        value = value[0]
    if not isinstance(value, list | tuple):
        return []
    return [float(item) for item in value]


def _first_scalar(value: object, *, default: float) -> float:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list | tuple):
        if not value:
            return default
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
