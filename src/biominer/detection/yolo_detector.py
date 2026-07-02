from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
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


class YoloSidecarObjectDetector:
    backend = "yolo"
    model_id = "ultralytics-yolo-sidecar"
    model_version = "sidecar"

    def __init__(self, *, runtime_python: str, model_path: str = "yolov8n.pt", device: str = "auto") -> None:
        self.runtime_python = str(runtime_python)
        self.checkpoint = model_path
        self.device = device

    def detect_batch(self, images: Sequence[DecodedImage]) -> list[list[DetectionCandidate]]:
        payload = {
            "model_path": self.checkpoint,
            "device": self.device,
            "images": [_image_to_payload(image) for image in images],
        }
        result = subprocess.run(
            [self.runtime_python, "-m", "biominer.detection.yolo_detector"],
            input=json.dumps(payload, sort_keys=True),
            text=True,
            capture_output=True,
            cwd=str(Path.cwd()),
            env=_sidecar_env(),
            check=False,
        )
        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"YOLO sidecar detection failed: {error}")
        return _detections_from_payload(json.loads(result.stdout or "{}"))


def _run_sidecar() -> None:
    import sys

    request = json.loads(sys.stdin.read() or "{}")
    detector = YoloObjectDetector(model_path=str(request.get("model_path") or "yolov8n.pt"), device=str(request.get("device") or "auto"))
    images = [_image_from_payload(item) for item in request.get("images", [])]
    detections = detector.detect_batch(images)
    print(json.dumps({"detections": [[_candidate_to_payload(candidate) for candidate in batch] for batch in detections]}, sort_keys=True))


def _image_to_payload(image: DecodedImage) -> dict[str, object]:
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "source_uri": image.source_uri,
        "data_b64": base64.b64encode(image.data).decode("ascii"),
    }


def _image_from_payload(payload: dict[str, object]) -> DecodedImage:
    return DecodedImage(
        width=int(payload["width"]),
        height=int(payload["height"]),
        mode=str(payload.get("mode") or "RGB"),
        data=base64.b64decode(str(payload.get("data_b64") or "")),
        source_uri=str(payload.get("source_uri") or "") or None,
    )


def _candidate_to_payload(candidate: DetectionCandidate) -> dict[str, object]:
    return {
        "label": candidate.label,
        "score": candidate.score,
        "bbox_xyxy": list(candidate.bbox_xyxy),
        "objectness_score": candidate.objectness_score,
    }


def _detections_from_payload(payload: dict[str, object]) -> list[list[DetectionCandidate]]:
    return [
        [
            DetectionCandidate(
                label=str(candidate.get("label") or ""),
                score=float(candidate.get("score") or 0.0),
                bbox_xyxy=tuple(float(value) for value in candidate.get("bbox_xyxy", ())),  # type: ignore[arg-type]
                objectness_score=None
                if candidate.get("objectness_score") is None
                else float(candidate.get("objectness_score")),
            )
            for candidate in batch
        ]
        for batch in payload.get("detections", [])
    ]


def _sidecar_env() -> dict[str, str]:
    env = dict(os.environ)
    source_path = str(Path.cwd() / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source_path if not current else f"{source_path}{os.pathsep}{current}"
    return env


def _rgb_rows(image: DecodedImage) -> list[list[tuple[int, int, int]]]:
    rows: list[list[tuple[int, int, int]]] = []
    for y in range(image.height):
        row: list[tuple[int, int, int]] = []
        for x in range(image.width):
            offset = (y * image.width + x) * 3
            row.append((image.data[offset], image.data[offset + 1], image.data[offset + 2]))
        rows.append(row)
    return rows


if __name__ == "__main__":  # pragma: no cover - exercised through sidecar subprocesses.
    _run_sidecar()
