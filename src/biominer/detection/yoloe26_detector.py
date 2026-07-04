from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

from biominer.detection.detector_base import DecodedImage, DetectionCandidate, detector_label_is_taxon_like
from biominer.runtime_paths import YOLOE26_DIR


DEFAULT_YOLOE26_CHECKPOINT = "yoloe-26s-seg.pt"
ALLOWED_YOLOE26_CHECKPOINTS = (
    "yoloe-26n-seg.pt",
    "yoloe-26s-seg.pt",
    "yoloe-26m-seg.pt",
    "yoloe-26l-seg.pt",
    "yoloe-26x-seg.pt",
)
DEFAULT_YOLOE26_PROMPTS = (
    "butterfly",
    "moth",
    "caterpillar",
    "chrysalis",
    "pupa",
    "insect",
    "butterfly wing",
    "pinned butterfly specimen",
    "butterfly specimen",
    "lepidoptera",
    "flower",
    "leaf",
    "person",
    "hand",
    "drawing",
    "painting",
    "logo",
    "text",
    "sign",
    "museum label",
)
YOLOE26_PROMPT_LABEL_MAP = {
    "butterfly": "butterfly_like",
    "butterfly wing": "butterfly_like",
    "pinned butterfly specimen": "butterfly_like",
    "butterfly specimen": "butterfly_like",
    "lepidoptera": "butterfly_like",
    "moth": "moth_like",
    "caterpillar": "caterpillar",
    "chrysalis": "pupa",
    "pupa": "pupa",
    "insect": "insect_like",
    "flower": "hard_negative",
    "leaf": "hard_negative",
    "person": "hard_negative",
    "hand": "hard_negative",
    "drawing": "hard_negative",
    "painting": "hard_negative",
    "logo": "hard_negative",
    "text": "hard_negative",
    "sign": "hard_negative",
    "museum label": "hard_negative",
}


def default_yoloe26_prompts(*, include_hard_negative_prompts: bool = True) -> tuple[str, ...]:
    if include_hard_negative_prompts:
        return DEFAULT_YOLOE26_PROMPTS
    return tuple(prompt for prompt in DEFAULT_YOLOE26_PROMPTS if yoloe26_coarse_label(prompt) != "hard_negative")


def yoloe26_coarse_label(prompt: str) -> str:
    mapped = YOLOE26_PROMPT_LABEL_MAP.get(_normalise_prompt(prompt))
    if mapped:
        return mapped
    if detector_label_is_taxon_like(prompt):
        raise ValueError(f"YOLOE-26 prompts must be object proposals, not taxon labels: {prompt!r}")
    return "insect_like"


class YoloE26ObjectDetector:
    backend = "yoloe26"

    def __init__(
        self,
        *,
        checkpoint: str = DEFAULT_YOLOE26_CHECKPOINT,
        device: str = "auto",
        imgsz: int = 640,
        conf: float = 0.20,
        iou: float = 0.50,
        max_det: int = 8,
        prompt_classes: Sequence[str] | None = None,
    ) -> None:
        _validate_checkpoint(checkpoint)
        try:
            import ultralytics
            from ultralytics import YOLOE
        except ImportError as exc:  # pragma: no cover - optional dependency path.
            raise RuntimeError(
                "ultralytics with YOLOE support is required for the yoloe26 detector backend; "
                "install BioMiner's optional YOLOE-26 vision runtime"
            ) from exc

        self.checkpoint = checkpoint
        self.device = None if device == "auto" else device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.prompt_classes = tuple(prompt_classes or default_yoloe26_prompts())
        self.model_id = f"yoloe26:{Path(checkpoint).stem}"
        self.model_version = f"ultralytics:{getattr(ultralytics, '__version__', 'unknown')}"
        self._model = YOLOE(_checkpoint_reference(checkpoint))
        self._model.set_classes(list(self.prompt_classes))

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
        return [detections_from_yoloe_result(result, prompt_classes=self.prompt_classes) for result in results]


class YoloE26SidecarObjectDetector:
    backend = "yoloe26"

    def __init__(
        self,
        *,
        runtime_python: str,
        checkpoint: str = DEFAULT_YOLOE26_CHECKPOINT,
        device: str = "auto",
        imgsz: int = 640,
        conf: float = 0.20,
        iou: float = 0.50,
        max_det: int = 8,
        prompt_classes: Sequence[str] | None = None,
    ) -> None:
        _validate_checkpoint(checkpoint)
        self.runtime_python = str(Path(runtime_python).expanduser())
        self.checkpoint = checkpoint
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.prompt_classes = tuple(prompt_classes or default_yoloe26_prompts())
        self.model_id = f"yoloe26:{Path(checkpoint).stem}"
        self.model_version = "ultralytics:unknown"

    def detect_batch(self, images: Sequence[DecodedImage]) -> list[list[DetectionCandidate]]:
        payload = {
            "checkpoint": self.checkpoint,
            "device": self.device,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "max_det": self.max_det,
            "prompt_classes": list(self.prompt_classes),
            "images": [_image_to_payload(image) for image in images],
        }
        result = subprocess.run(
            [self.runtime_python, "-m", "biominer.detection.yoloe26_detector"],
            input=json.dumps(payload, sort_keys=True),
            text=True,
            capture_output=True,
            cwd=str(_sidecar_cwd(self.runtime_python)),
            env=_sidecar_env(self.runtime_python),
            check=False,
        )
        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"YOLOE-26 sidecar detection failed: {error}")
        response = json.loads(result.stdout or "{}")
        metadata = response.get("metadata") or {}
        self.model_id = str(metadata.get("model_id") or self.model_id)
        self.model_version = str(metadata.get("model_version") or self.model_version)
        return _detections_from_payload(response)


def detections_from_yoloe_result(result: object, *, prompt_classes: Sequence[str] | None = None) -> list[DetectionCandidate]:
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
        prompt = _prompt_for_class(cls_id, names=names, prompt_classes=prompt_classes)
        score = float(_first_scalar(getattr(box, "conf", 0.0), default=0.0))
        rows.append(
            DetectionCandidate(
                label=yoloe26_coarse_label(prompt),
                score=score,
                bbox_xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                objectness_score=score,
            )
        )
    return rows


def _run_sidecar() -> None:
    import sys

    request = json.loads(sys.stdin.read() or "{}")
    detector = YoloE26ObjectDetector(
        checkpoint=str(request.get("checkpoint") or DEFAULT_YOLOE26_CHECKPOINT),
        device=str(request.get("device") or "auto"),
        imgsz=int(request.get("imgsz") or 640),
        conf=float(request.get("conf") or 0.20),
        iou=float(request.get("iou") or 0.50),
        max_det=int(request.get("max_det") or 8),
        prompt_classes=tuple(str(value) for value in request.get("prompt_classes", []) if str(value).strip())
        or default_yoloe26_prompts(),
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


def _validate_checkpoint(checkpoint: str) -> None:
    path = Path(checkpoint).expanduser()
    if path.name in ALLOWED_YOLOE26_CHECKPOINTS or path.exists():
        return
    if path.parent != Path("."):
        return
    raise ValueError(
        "unsupported YOLOE-26 checkpoint; expected one of "
        + ", ".join(ALLOWED_YOLOE26_CHECKPOINTS)
        + " or an explicit checkpoint path"
    )


def _checkpoint_reference(checkpoint: str) -> str:
    path = Path(checkpoint).expanduser()
    if path.is_absolute() or path.parent != Path("."):
        return str(path)
    model_dir = os.environ.get("BIOMINER_YOLO26_MODEL_DIR")
    if model_dir:
        candidate = Path(model_dir).expanduser() / checkpoint
        if candidate.exists():
            return str(candidate)
    return checkpoint


def _decoded_image_to_pil(image: DecodedImage) -> object:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency path.
        raise RuntimeError("Pillow is required to prepare decoded images for YOLOE-26") from exc
    return Image.frombytes("RGB", (image.width, image.height), image.data)


def _prompt_for_class(cls_id: int, *, names: object, prompt_classes: Sequence[str] | None = None) -> str:
    if isinstance(names, dict) and cls_id in names:
        return str(names[cls_id])
    if prompt_classes is not None and 0 <= cls_id < len(prompt_classes):
        return str(prompt_classes[cls_id])
    return str(cls_id)


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


def _normalise_prompt(prompt: str) -> str:
    return " ".join(str(prompt or "").strip().casefold().split())


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
    return YOLOE26_DIR


if __name__ == "__main__":  # pragma: no cover - exercised through sidecar subprocesses.
    _run_sidecar()
