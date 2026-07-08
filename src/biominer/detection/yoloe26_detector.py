from __future__ import annotations

import base64
from collections import deque
import json
import os
from pathlib import Path
import subprocess
from threading import Thread
from typing import IO, Any, Callable, Sequence

from biominer.detection.detector_base import DecodedImage, DetectionCandidate, detector_label_is_taxon_like
from biominer.runtime_paths import YOLOE26_DIR


PopenFactory = Callable[..., subprocess.Popen[str]]

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
        prompt_tuple = tuple(prompt_classes or default_yoloe26_prompts())
        _validate_prompt_classes(prompt_tuple)
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
        self.prompt_classes = prompt_tuple
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
        popen: PopenFactory = subprocess.Popen,
    ) -> None:
        _validate_checkpoint(checkpoint)
        prompt_tuple = tuple(prompt_classes or default_yoloe26_prompts())
        _validate_prompt_classes(prompt_tuple)
        self.runtime_python = str(Path(runtime_python).expanduser())
        self.checkpoint = checkpoint
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.prompt_classes = prompt_tuple
        self.model_id = f"yoloe26:{Path(checkpoint).stem}"
        self.model_version = "ultralytics:unknown"
        self.popen = popen
        self._process: subprocess.Popen[str] | None = None
        self._stdin: IO[str] | None = None
        self._stdout: IO[str] | None = None
        self._stderr_lines: deque[str] = deque(maxlen=50)
        self._stderr_thread: Thread | None = None

    def __enter__(self) -> "YoloE26SidecarObjectDetector":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001 - context manager protocol.
        self.close()

    def detect_batch(self, images: Sequence[DecodedImage]) -> list[list[DetectionCandidate]]:
        if not images:
            return []
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
        response = self._request(payload)
        return _detections_from_payload(response)

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None and self._stdin is not None:
            try:
                self._stdin.write(json.dumps({"shutdown": True}, sort_keys=True) + "\n")
                self._stdin.flush()
                process.wait(timeout=10)
            except Exception:  # noqa: BLE001 - close must not mask caller errors.
                process.terminate()
                process.wait(timeout=10)
        self._process = None
        self._stdin = None
        self._stdout = None

    def _request(self, payload: dict[str, object]) -> dict[str, object]:
        process = self._ensure_process()
        if process.poll() is not None:
            raise RuntimeError(_sidecar_exit_message("YOLOE-26 persistent sidecar exited early", process, self._stderr_tail()))
        assert self._stdin is not None
        assert self._stdout is not None
        try:
            self._stdin.write(json.dumps(payload, sort_keys=True) + "\n")
            self._stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(_sidecar_exit_message("YOLOE-26 persistent sidecar pipe closed", process, self._stderr_tail())) from exc
        line = self._stdout.readline()
        if not line:
            raise RuntimeError(_sidecar_exit_message("YOLOE-26 persistent sidecar closed stdout before returning detections", process, self._stderr_tail()))
        response = json.loads(line)
        if "error" in response:
            error_type = str(response.get("error_type") or "error")
            raise RuntimeError(f"YOLOE-26 sidecar detection failed ({error_type}): {response['error']}")
        metadata = response.get("metadata") or response
        if isinstance(metadata, dict):
            self.model_id = str(metadata.get("model_id") or self.model_id)
            self.model_version = str(metadata.get("model_version") or self.model_version)
            self.checkpoint = str(metadata.get("checkpoint") or self.checkpoint)
        return response

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is None:
            process = self.popen(
                [self.runtime_python, "-m", "biominer.detection.yoloe26_detector", "--persistent"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=str(_sidecar_cwd(self.runtime_python)),
                env=_sidecar_env(self.runtime_python),
            )
            if process.stdin is None or process.stdout is None:
                process.terminate()
                raise RuntimeError("YOLOE-26 persistent sidecar did not expose stdin/stdout pipes")
            self._process = process
            self._stdin = process.stdin
            self._stdout = process.stdout
            if process.stderr is not None:
                self._stderr_thread = Thread(target=_drain_stderr, args=(process.stderr, self._stderr_lines), daemon=True)
                self._stderr_thread.start()
        return self._process

    def _stderr_tail(self) -> str:
        return "\n".join(self._stderr_lines)


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


def _drain_stderr(stderr: IO[str], lines: deque[str]) -> None:
    try:
        for line in stderr:
            stripped = line.rstrip()
            if stripped:
                lines.append(stripped)
    except Exception:  # noqa: BLE001 - diagnostic drain must not affect worker requests.
        return


def _sidecar_exit_message(prefix: str, process: subprocess.Popen[str], stderr_tail: str) -> str:
    code = process.poll()
    message = f"{prefix} with code {code}"
    if stderr_tail:
        message = f"{message}: {stderr_tail}"
    return message


def _run_sidecar() -> None:
    import sys

    request = json.loads(sys.stdin.read() or "{}")
    detector = _detector_from_request(request)
    images = [_image_from_payload(item) for item in request.get("images", [])]
    detections = detector.detect_batch(images)
    print(json.dumps(_sidecar_response(detector, detections), sort_keys=True))


def _run_persistent_sidecar() -> None:
    import sys

    detector: YoloE26ObjectDetector | None = None
    loaded_key: tuple[object, ...] | None = None
    for line in sys.stdin:
        try:
            request = json.loads(line or "{}")
            if request.get("shutdown"):
                return
            key = _detector_request_key(request)
            if detector is None or loaded_key != key:
                detector = _detector_from_request(request)
                loaded_key = key
            images = [_image_from_payload(item) for item in request.get("images", [])]
            detections = detector.detect_batch(images)
            print(json.dumps(_sidecar_response(detector, detections), sort_keys=True), flush=True)
        except Exception as exc:  # noqa: BLE001 - persistent worker reports JSON errors to the controller.
            print(
                json.dumps(
                    {
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


def _detector_from_request(request: dict[str, object]) -> YoloE26ObjectDetector:
    kwargs = _detector_kwargs_from_request(request)
    return YoloE26ObjectDetector(**kwargs)


def _detector_request_key(request: dict[str, object]) -> tuple[object, ...]:
    kwargs = _detector_kwargs_from_request(request)
    return (
        kwargs["checkpoint"],
        kwargs["device"],
        kwargs["imgsz"],
        kwargs["conf"],
        kwargs["iou"],
        kwargs["max_det"],
        kwargs["prompt_classes"],
    )


def _detector_kwargs_from_request(request: dict[str, object]) -> dict[str, object]:
    prompt_classes = tuple(str(value) for value in request.get("prompt_classes", []) if str(value).strip()) or default_yoloe26_prompts()
    return {
        "checkpoint": str(request.get("checkpoint") or DEFAULT_YOLOE26_CHECKPOINT),
        "device": str(request.get("device") or "auto"),
        "imgsz": int(request.get("imgsz") or 640),
        "conf": float(request.get("conf") or 0.20),
        "iou": float(request.get("iou") or 0.50),
        "max_det": int(request.get("max_det") or 8),
        "prompt_classes": prompt_classes,
    }


def _sidecar_response(detector: YoloE26ObjectDetector, detections: list[list[DetectionCandidate]]) -> dict[str, object]:
    metadata = {
        "backend": detector.backend,
        "model_id": detector.model_id,
        "model_version": detector.model_version,
        "checkpoint": detector.checkpoint,
    }
    return {
        **metadata,
        "metadata": metadata,
        "detections": [[_candidate_to_payload(candidate) for candidate in batch] for batch in detections],
    }


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


def _validate_prompt_classes(prompt_classes: Sequence[str]) -> None:
    for prompt in prompt_classes:
        yoloe26_coarse_label(str(prompt))


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
    import sys

    if "--persistent" in sys.argv[1:]:
        _run_persistent_sidecar()
    else:
        _run_sidecar()
