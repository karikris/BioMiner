from __future__ import annotations

import base64
from collections import deque
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from shutil import rmtree
import subprocess
from tempfile import mkdtemp
from threading import Thread
from typing import IO, Callable, Iterator, Sequence

from biominer.detection.detector_base import (
    DecodedImage,
    DetectionCandidate,
    detector_label_is_taxon_like,
    normalize_detector_prompt,
    normalize_mask_polygon_xyn,
)
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
    "ant": "insect_like",
    "bee": "insect_like",
    "beetle": "insect_like",
    "butterfly": "adult_butterfly",
    "butterfly wing": "possible_adult_butterfly",
    "caddisfly": "insect_like",
    "cicada": "insect_like",
    "cockroach": "insect_like",
    "cricket": "insect_like",
    "damselfly": "insect_like",
    "dragonfly": "insect_like",
    "earwig": "insect_like",
    "fly": "insect_like",
    "grasshopper": "insect_like",
    "lacewing": "insect_like",
    "mantis": "insect_like",
    "mayfly": "insect_like",
    "mosquito": "insect_like",
    "pinned insect specimen": "pinned_specimen",
    "pinned butterfly specimen": "pinned_specimen",
    "butterfly specimen": "pinned_specimen",
    "stick insect": "insect_like",
    "stonefly": "insect_like",
    "termite": "insect_like",
    "true bug": "insect_like",
    "wasp": "insect_like",
    "lepidoptera": "possible_adult_butterfly",
    "moth": "moth_like",
    "caterpillar": "caterpillar",
    "chrysalis": "pupa",
    "pupa": "pupa",
    "insect": "insect_like",
    "flower": "no_relevant_organism",
    "leaf": "no_relevant_organism",
    "person": "no_relevant_organism",
    "hand": "no_relevant_organism",
    "drawing": "artifact",
    "painting": "artifact",
    "logo": "artifact",
    "text": "artifact",
    "sign": "artifact",
    "museum label": "artifact",
}
YOLOE26_SIDECAR_TRANSPORTS = ("json_b64", "image_path")


def default_yoloe26_prompts(*, include_hard_negative_prompts: bool = True) -> tuple[str, ...]:
    if include_hard_negative_prompts:
        return DEFAULT_YOLOE26_PROMPTS
    return tuple(
        prompt
        for prompt in DEFAULT_YOLOE26_PROMPTS
        if yoloe26_coarse_label(prompt) not in {"artifact", "no_relevant_organism", "hard_negative"}
    )


def yoloe26_coarse_label(prompt: str) -> str:
    if detector_label_is_taxon_like(prompt):
        raise ValueError(f"YOLOE-26 prompts must be object proposals, not taxon labels: {prompt!r}")
    normalized = normalize_detector_prompt(prompt)
    mapped = YOLOE26_PROMPT_LABEL_MAP.get(normalized)
    if mapped:
        return mapped
    raise ValueError(f"unsupported YOLOE-26 object prompt: {prompt!r}")


def yoloe26_prompt_set_fingerprint(prompt_classes: Sequence[str]) -> str:
    prompts = list(_normalized_prompt_classes(prompt_classes))
    payload = json.dumps(
        {"prompts": prompts, "schema_version": "yoloe26-prompt-set-v1"},
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class YoloE26ObjectDetector:
    backend = "yoloe26"
    execution_mode = "in_process"

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
        prompt_tuple = _normalized_prompt_classes(prompt_classes or default_yoloe26_prompts())
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
        self.prompt_set_fingerprint = yoloe26_prompt_set_fingerprint(prompt_tuple)
        self.model_id = f"yoloe26:{Path(checkpoint).stem}"
        self.model_version = f"ultralytics:{getattr(ultralytics, '__version__', 'unknown')}"
        self._model = YOLOE(_checkpoint_reference(checkpoint))
        self._model.set_classes(list(self.prompt_classes))
        self.model_load_count = 1

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
    execution_mode = "persistent_sidecar"

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
        transport: str = "json_b64",
        temp_dir: str | Path | None = None,
        retain_temp_images: bool = False,
        popen: PopenFactory = subprocess.Popen,
    ) -> None:
        _validate_checkpoint(checkpoint)
        prompt_tuple = _normalized_prompt_classes(prompt_classes or default_yoloe26_prompts())
        _validate_prompt_classes(prompt_tuple)
        self.runtime_python = str(Path(runtime_python).expanduser())
        self.checkpoint = checkpoint
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.prompt_classes = prompt_tuple
        self.prompt_set_fingerprint = yoloe26_prompt_set_fingerprint(prompt_tuple)
        self.transport = _validate_sidecar_transport(transport)
        self.temp_dir = Path(temp_dir).expanduser() if temp_dir is not None else None
        self.retain_temp_images = bool(retain_temp_images)
        self.model_id = f"yoloe26:{Path(checkpoint).stem}"
        self.model_version = "ultralytics:unknown"
        self.popen = popen
        self._process: subprocess.Popen[str] | None = None
        self._stdin: IO[str] | None = None
        self._stdout: IO[str] | None = None
        self._stderr_lines: deque[str] = deque(maxlen=50)
        self._stderr_thread: Thread | None = None
        self.worker_process_starts = 0
        self.worker_request_count = 0

    def __enter__(self) -> "YoloE26SidecarObjectDetector":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001 - context manager protocol.
        self.close()

    def detect_batch(self, images: Sequence[DecodedImage]) -> list[list[DetectionCandidate]]:
        if not images:
            return []
        base = self._base_request_payload()
        if self.transport == "json_b64":
            payload = {**base, "images": [_image_to_payload(image) for image in images]}
            response = self._request(payload)
            return _detections_from_payload(response)
        with _temporary_sidecar_image_paths(
            images,
            root=self.temp_dir,
            retain=self.retain_temp_images,
        ) as image_paths:
            payload = {**base, "image_paths": [str(path) for path in image_paths]}
            response = self._request(payload)
            return _detections_from_payload(response)

    def _base_request_payload(self) -> dict[str, object]:
        return {
            "checkpoint": self.checkpoint,
            "device": self.device,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "max_det": self.max_det,
            "prompt_classes": list(self.prompt_classes),
            "transport": self.transport,
        }

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
            self.worker_request_count += 1
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
            actual_model_id = str(metadata.get("model_id") or self.model_id)
            actual_checkpoint = str(metadata.get("checkpoint") or self.checkpoint)
            if actual_model_id != self.model_id:
                raise RuntimeError("YOLOE-26 sidecar returned a different model ID")
            if actual_checkpoint != self.checkpoint:
                raise RuntimeError("YOLOE-26 sidecar returned a different checkpoint")
            self.model_version = str(metadata.get("model_version") or self.model_version)
            actual_prompt_fingerprint = metadata.get("prompt_set_fingerprint")
            if actual_prompt_fingerprint is not None and str(actual_prompt_fingerprint) != self.prompt_set_fingerprint:
                raise RuntimeError("YOLOE-26 sidecar returned a different prompt-set fingerprint")
        return response

    @property
    def model_load_count(self) -> int:
        return self.worker_process_starts

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
            self.worker_process_starts += 1
            self._stdin = process.stdin
            self._stdout = process.stdout
            if process.stderr is not None:
                self._stderr_thread = Thread(target=_drain_stderr, args=(process.stderr, self._stderr_lines), daemon=True)
                self._stderr_thread.start()
        return self._process

    def _stderr_tail(self) -> str:
        return "\n".join(self._stderr_lines)


def detections_from_yoloe_result(result: object, *, prompt_classes: Sequence[str] | None = None) -> list[DetectionCandidate]:
    ordered_prompts = _validated_result_prompts(getattr(result, "names", None))
    if prompt_classes is not None:
        expected_prompts = _normalized_prompt_classes(prompt_classes)
        if sorted(expected_prompts) != sorted(ordered_prompts):
            raise ValueError("YOLOE result.names does not match the configured prompt set")
    prompt_set_fingerprint = yoloe26_prompt_set_fingerprint(ordered_prompts)
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
        cls_id = _result_class_id(getattr(box, "cls", -1))
        if cls_id >= len(ordered_prompts):
            raise ValueError(f"YOLOE result class id {cls_id} is outside result.names")
        prompt = ordered_prompts[cls_id]
        score = float(_first_scalar(getattr(box, "conf", 0.0), default=0.0))
        rows.append(
            DetectionCandidate(
                label=yoloe26_coarse_label(prompt),
                score=score,
                bbox_xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                objectness_score=score,
                detector_prompt=prompt,
                detector_class_id=cls_id,
                detector_prompt_set_fingerprint=prompt_set_fingerprint,
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
        raise ValueError("YOLOE result masks.xyn must be an ordered sequence")
    if len(polygons) != expected_count:
        raise ValueError("YOLOE result masks must align one-to-one with boxes")
    return [normalize_mask_polygon_xyn(polygon) for polygon in polygons]


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
    images = _images_from_request(request)
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
            images = _images_from_request(request)
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


def _images_from_request(request: dict[str, object]) -> list[DecodedImage]:
    image_paths = request.get("image_paths")
    if image_paths is not None:
        if not isinstance(image_paths, list | tuple):
            raise ValueError("YOLOE-26 sidecar image_paths must be a list")
        return [_image_from_path(Path(str(path))) for path in image_paths]
    return [_image_from_payload(item) for item in request.get("images", [])]


def _sidecar_response(detector: YoloE26ObjectDetector, detections: list[list[DetectionCandidate]]) -> dict[str, object]:
    metadata = {
        "backend": detector.backend,
        "model_id": detector.model_id,
        "model_version": detector.model_version,
        "checkpoint": detector.checkpoint,
    }
    prompt_set_fingerprint = getattr(detector, "prompt_set_fingerprint", None)
    if prompt_set_fingerprint is not None:
        metadata["prompt_set_fingerprint"] = str(prompt_set_fingerprint)
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
    raw_prompts = [str(prompt) for prompt in prompt_classes]
    if not raw_prompts:
        raise ValueError("YOLOE-26 prompt classes must not be empty")
    for prompt in raw_prompts:
        yoloe26_coarse_label(prompt)
    normalized = [normalize_detector_prompt(prompt) for prompt in raw_prompts]
    if len(set(normalized)) != len(normalized):
        raise ValueError("YOLOE-26 prompt classes must not contain duplicates")


def _normalized_prompt_classes(prompt_classes: Sequence[str]) -> tuple[str, ...]:
    _validate_prompt_classes(prompt_classes)
    return tuple(normalize_detector_prompt(prompt) for prompt in prompt_classes)


def _validate_sidecar_transport(value: str) -> str:
    transport = str(value or "").strip()
    if transport in YOLOE26_SIDECAR_TRANSPORTS:
        return transport
    raise ValueError("YOLOE-26 sidecar transport must be one of: " + ", ".join(YOLOE26_SIDECAR_TRANSPORTS))


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


def _validated_result_prompts(names: object) -> tuple[str, ...]:
    if isinstance(names, list | tuple):
        mapped_names = dict(enumerate(names))
    elif isinstance(names, dict):
        mapped_names: dict[int, object] = {}
        for raw_key, value in names.items():
            if isinstance(raw_key, bool):
                raise ValueError("YOLOE result.names class ids must be contiguous non-negative integers")
            try:
                key = int(raw_key)
            except (TypeError, ValueError) as exc:
                raise ValueError("YOLOE result.names class ids must be contiguous non-negative integers") from exc
            if str(raw_key).strip() != str(key):
                raise ValueError("YOLOE result.names class ids must be contiguous non-negative integers")
            if key in mapped_names:
                raise ValueError("YOLOE result.names contains duplicate class ids")
            mapped_names[key] = value
    else:
        raise ValueError("YOLOE result.names must be an ordered list or class-id mapping")
    if sorted(mapped_names) != list(range(len(mapped_names))):
        raise ValueError("YOLOE result.names class ids must be contiguous from zero")
    if not mapped_names:
        raise ValueError("YOLOE result.names must contain at least one prompt")
    raw_prompts = tuple(str(mapped_names[index]) for index in range(len(mapped_names)))
    for prompt in raw_prompts:
        yoloe26_coarse_label(prompt)
    prompts = tuple(normalize_detector_prompt(prompt) for prompt in raw_prompts)
    if len(set(prompts)) != len(prompts):
        raise ValueError("YOLOE result.names must not contain duplicate prompts")
    return prompts


def _result_class_id(value: object) -> int:
    raw = _first_scalar(value, default=-1)
    if not float(raw).is_integer() or raw < 0:
        raise ValueError(f"YOLOE result class id must be a non-negative integer, got {raw!r}")
    return int(raw)


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


@contextmanager
def _temporary_sidecar_image_paths(
    images: Sequence[DecodedImage],
    *,
    root: Path | None,
    retain: bool,
) -> Iterator[list[Path]]:
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    request_dir = Path(mkdtemp(prefix="biominer_yoloe26_", dir=str(root) if root is not None else None))
    try:
        paths: list[Path] = []
        for index, image in enumerate(images):
            path = request_dir / f"{index:06d}.ppm"
            path.write_bytes(_ppm_bytes(image))
            paths.append(path)
        yield paths
    finally:
        if not retain:
            rmtree(request_dir, ignore_errors=True)


def _ppm_bytes(image: DecodedImage) -> bytes:
    return f"P6\n{image.width} {image.height}\n255\n".encode("ascii") + image.data


def _image_from_path(path: Path) -> DecodedImage:
    if not path.exists():
        raise FileNotFoundError(f"YOLOE-26 sidecar image path does not exist: {path}")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - sidecar runtime dependency path.
        raise RuntimeError("Pillow is required to read YOLOE-26 sidecar image paths") from exc
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        return DecodedImage(width=int(width), height=int(height), mode="RGB", data=rgb.tobytes(), source_uri=str(path))


def _candidate_to_payload(candidate: DetectionCandidate) -> dict[str, object]:
    return {
        "label": candidate.label,
        "score": candidate.score,
        "bbox_xyxy": list(candidate.bbox_xyxy),
        "objectness_score": candidate.objectness_score,
        "detector_prompt": candidate.detector_prompt,
        "detector_class_id": candidate.detector_class_id,
        "detector_prompt_set_fingerprint": candidate.detector_prompt_set_fingerprint,
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
                detector_prompt=None
                if candidate.get("detector_prompt") is None
                else str(candidate.get("detector_prompt")),
                detector_class_id=None
                if candidate.get("detector_class_id") is None
                else int(candidate.get("detector_class_id")),
                detector_prompt_set_fingerprint=None
                if candidate.get("detector_prompt_set_fingerprint") is None
                else str(candidate.get("detector_prompt_set_fingerprint")),
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
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
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
