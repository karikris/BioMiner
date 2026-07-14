from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
from typing import Sequence

from biominer.detection.detector_base import (
    DecodedImage,
    DetectionCandidate,
    normalize_detector_label,
    normalize_mask_polygon_xyn,
)
from biominer.runtime_paths import YOLOE26_DIR as VISION_RUNTIME_DIR


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


class Yolo26SidecarObjectDetector:
    """YOLO26 inference through BioMiner's optional Python 3.12 vision runtime."""

    backend = "yolo26"

    def __init__(
        self,
        *,
        runtime_python: str,
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
        self.runtime_python = str(Path(runtime_python).expanduser())
        self.checkpoint = checkpoint
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.model_id = f"yolo26:{Path(checkpoint).stem}"
        self.model_version = "ultralytics:unknown"

    def detect_batch(self, images: Sequence[DecodedImage]) -> list[list[DetectionCandidate]]:
        payload = {
            "checkpoint": self.checkpoint,
            "device": self.device,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "max_det": self.max_det,
            "images": [_image_to_payload(image) for image in images],
        }
        result = subprocess.run(
            [self.runtime_python, "-m", "biominer.detection.yolo26_detector"],
            input=json.dumps(payload, sort_keys=True),
            text=True,
            capture_output=True,
            cwd=str(_sidecar_cwd(self.runtime_python)),
            env=_sidecar_env(self.runtime_python),
            check=False,
        )
        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"YOLO26 sidecar detection failed: {error}")
        response = json.loads(result.stdout or "{}")
        metadata = response.get("metadata") or {}
        self.model_id = str(metadata.get("model_id") or self.model_id)
        self.model_version = str(metadata.get("model_version") or self.model_version)
        return _detections_from_payload(response)


def detections_from_yolo26_result(result: object) -> list[DetectionCandidate]:
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    box_rows = list(boxes)
    mask_polygons = _result_mask_polygons_xyn(result, expected_count=len(box_rows))
    rows: list[DetectionCandidate] = []
    for index, box in enumerate(box_rows):
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
                mask_polygon_xyn=mask_polygons[index],
            )
        )
    return rows


def _result_mask_polygons_xyn(
    result: object,
    *,
    expected_count: int,
) -> list[tuple[tuple[float, float], ...] | None]:
    masks = getattr(result, "masks", None)
    if masks is None:
        return [None] * expected_count
    polygons = getattr(masks, "xyn", None)
    if not isinstance(polygons, list | tuple):
        raise ValueError("YOLO26 result masks.xyn must be an ordered sequence")
    if len(polygons) != expected_count:
        raise ValueError("YOLO26 result masks must align one-to-one with boxes")
    return [normalize_mask_polygon_xyn(polygon) for polygon in polygons]


def yolo26_coarse_label(label: object) -> str:
    return normalize_detector_label(label)


def _run_sidecar() -> None:
    import sys

    request = json.loads(sys.stdin.read() or "{}")
    detector = Yolo26ObjectDetector(
        checkpoint=str(request.get("checkpoint") or ""),
        device=str(request.get("device") or "auto"),
        imgsz=int(request.get("imgsz") or DEFAULT_YOLO26_IMGSZ),
        conf=float(request.get("conf") or DEFAULT_YOLO26_CONF),
        iou=float(request.get("iou") or DEFAULT_YOLO26_IOU),
        max_det=int(request.get("max_det") or DEFAULT_YOLO26_MAX_DET),
    )
    images = [_image_from_payload(item) for item in request.get("images", [])]
    detections = detector.detect_batch(images)
    print(
        json.dumps(
            {
                "metadata": {
                    "backend": detector.backend,
                    "model_id": detector.model_id,
                    "model_version": detector.model_version,
                    "checkpoint": detector.checkpoint,
                },
                "detections": [[_candidate_to_payload(candidate) for candidate in batch] for batch in detections],
            },
            sort_keys=True,
        )
    )


def _decoded_image_to_pil(image: DecodedImage) -> object:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency path.
        raise RuntimeError("Pillow is required to prepare decoded images for YOLO26") from exc
    return Image.frombytes("RGB", (image.width, image.height), image.data)


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
        "mask_polygon_xyn": None
        if candidate.mask_polygon_xyn is None
        else [list(point) for point in candidate.mask_polygon_xyn],
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
                mask_polygon_xyn=normalize_mask_polygon_xyn(candidate.get("mask_polygon_xyn")),
            )
            for candidate in batch
        ]
        for batch in payload.get("detections", [])
    ]


def _sidecar_env(runtime_python: str | None = None) -> dict[str, str]:
    env = dict(os.environ)
    source_path = str(Path.cwd() / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source_path if not current else f"{source_path}{os.pathsep}{current}"
    root = _runtime_root(runtime_python)
    cache_root = root / "cache"
    defaults = {
        "HF_HOME": cache_root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache_root / "huggingface" / "hub",
        "TORCH_HOME": cache_root / "torch",
        "YOLO_CONFIG_DIR": cache_root / "ultralytics",
        "BIOMINER_YOLO26_MODEL_DIR": root / "models",
    }
    for key, value in defaults.items():
        env.setdefault(key, str(value))
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def _sidecar_cwd(runtime_python: str | None = None) -> Path:
    model_dir = Path(_sidecar_env(runtime_python)["BIOMINER_YOLO26_MODEL_DIR"])
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _runtime_root(runtime_python: str | None = None) -> Path:
    if runtime_python:
        path = Path(runtime_python).expanduser()
        if len(path.parents) >= 3 and path.parent.name == "bin":
            return path.parents[2]
    return VISION_RUNTIME_DIR


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


if __name__ == "__main__":  # pragma: no cover - exercised through sidecar subprocesses.
    _run_sidecar()
