from __future__ import annotations

from typing import Sequence

from biominer.detection.detector_base import DecodedImage, DetectionCandidate


class YoloObjectDetector:
    backend = "yolo"

    def __init__(self, *, model_path: str = "yolov8n.pt", device: str = "auto") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - optional dependency path.
            raise RuntimeError(
                "ultralytics is required for the YOLO detector backend; install BioMiner's optional vision environment"
            ) from exc
        self._model = YOLO(model_path)
        self.model_id = "ultralytics-yolo"
        self.model_version = str(getattr(self._model, "task", "") or "unknown")
        self.checkpoint = model_path
        self.device = None if device == "auto" else device

    def detect_batch(self, images: Sequence[DecodedImage]) -> list[list[DetectionCandidate]]:
        arrays = [_rgb_rows(image) for image in images]
        results = self._model.predict(arrays, device=self.device, verbose=False)
        output: list[list[DetectionCandidate]] = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            rows: list[DetectionCandidate] = []
            for box in getattr(result, "boxes", []) or []:
                xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
                cls_id = int(box.cls[0]) if getattr(box, "cls", None) is not None else -1
                rows.append(
                    DetectionCandidate(
                        label=str(names.get(cls_id, cls_id)),
                        score=float(box.conf[0]),
                        bbox_xyxy=xyxy,  # type: ignore[arg-type]
                        objectness_score=float(box.conf[0]),
                    )
                )
            output.append(rows)
        return output


def _rgb_rows(image: DecodedImage) -> list[list[tuple[int, int, int]]]:
    rows: list[list[tuple[int, int, int]]] = []
    for y in range(image.height):
        row: list[tuple[int, int, int]] = []
        for x in range(image.width):
            offset = (y * image.width + x) * 3
            row.append((image.data[offset], image.data[offset + 1], image.data[offset + 2]))
        rows.append(row)
    return rows
