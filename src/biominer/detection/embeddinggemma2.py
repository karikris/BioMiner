"""Frozen, full-frame EmbeddingGemma 2 visual-domain evidence.

This adapter emits image embeddings and raw cosine evidence. It cannot measure
object presence, counts, boxes, masks, or subject area, and every proposed
domain requires review. No model dependency is imported until ``load()``.
"""

from __future__ import annotations

import base64
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
from queue import Empty, Queue
import re
import subprocess
from threading import Lock, Thread
from time import monotonic
from typing import Any

from biominer.detection.detector_base import DecodedImage
from biominer.detection.routing import DETECTION_ROUTES
from biominer.runtime_paths import embeddinggemma2_runtime_env


DEFAULT_EMBEDDINGGEMMA2_MODEL = "google/embeddinggemma-2"
DEFAULT_EMBEDDINGGEMMA2_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
EMBEDDINGGEMMA2_PROTOCOL_VERSION = "embeddinggemma2-jsonl-v1"
EMBEDDINGGEMMA2_ADAPTER_VERSION = "embeddinggemma2-full-frame-v1"
EMBEDDING_DIMENSION = 768

# The complete ordered universe is mandatory. Negatives cannot be disabled.
DEFAULT_EMBEDDINGGEMMA2_PROMPTS: tuple[tuple[str, str], ...] = (
    ("A photograph of a live adult butterfly in a natural scene.", "adult_butterfly_field"),
    ("A photograph of a butterfly with open or closed wings outdoors.", "adult_butterfly_field"),
    ("A photograph of a moth.", "possible_moth_or_other_insect"),
    ("A photograph of an insect other than a butterfly.", "possible_moth_or_other_insect"),
    ("A photograph of a live caterpillar on a plant.", "caterpillar_field"),
    ("A photograph of an insect larva.", "caterpillar_field"),
    ("A photograph of an insect pupa or butterfly chrysalis.", "pupa_or_chrysalis"),
    ("A photograph of a pinned butterfly specimen in a collection.", "pinned_specimen"),
    ("A photograph of a preserved insect specimen.", "pinned_specimen"),
    ("A butterfly drawing, painting, or illustration.", "artwork_logo_tattoo_or_other_artifact"),
    ("A butterfly logo or tattoo.", "artwork_logo_tattoo_or_other_artifact"),
    ("A butterfly pattern printed on fabric or a decorative object.", "artwork_logo_tattoo_or_other_artifact"),
    ("A museum label, sign, or printed text.", "artwork_logo_tattoo_or_other_artifact"),
    ("A photograph of flowers or leaves without an insect.", "no_relevant_organism"),
    ("A photograph of a person or hand without an insect.", "no_relevant_organism"),
    ("An image without a relevant organism.", "no_relevant_organism"),
)

_BIOCLIP_MAPPING = {
    "adult_butterfly_field": "adult_field",
    "caterpillar_field": "larval",
    "pinned_specimen": "pinned_specimen",
}
_UNAVAILABLE_FIELDS = (
    "subject_present", "subject_area_ratio", "detection_count", "bbox_xyxy",
    "bbox_xyxyn", "bbox_xywhn", "mask_polygon_xyn", "detector_score", "objectness_score", "box_area_ratio",
)
_PREPROCESSING = {
    "canvas": "complete_original_decoded_canvas", "mode": "RGB", "input": "PIL.Image.frombytes",
    "spatial_crop": False, "image_prompt": "", "text_prompt_name": "Classification",
    "model_processor": "immutable_checkpoint_default", "audio_config": None,
    "embedding_dimension": EMBEDDING_DIMENSION, "normalization": "L2_before_cosine",
}


def _fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EmbeddingGemma2Config:
    model_id: str = DEFAULT_EMBEDDINGGEMMA2_MODEL
    revision: str = DEFAULT_EMBEDDINGGEMMA2_REVISION
    device: str = "auto"
    dtype: str = "float32"
    min_similarity: float = 0.25
    min_margin: float = 0.05
    embedding_dimension: int = field(default=EMBEDDING_DIMENSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip() or self.model_id != self.model_id.strip():
            raise ValueError("model_id must be a non-empty model identifier")
        if not isinstance(self.revision, str) or re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
            raise ValueError("revision must be an immutable lowercase 40-character model commit SHA")
        if not isinstance(self.device, str) or not self.device.strip() or self.device != self.device.strip():
            raise ValueError("device must be non-empty")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16; float16 is unsupported")
        for name, low, high in (("min_similarity", -1.0, 1.0), ("min_margin", 0.0, 2.0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and within [{low}, {high}]")
            object.__setattr__(self, name, float(value))

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.as_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> EmbeddingGemma2Config:
        data = dict(value)
        if data.pop("embedding_dimension", EMBEDDING_DIMENSION) != EMBEDDING_DIMENSION:
            raise ValueError("embedding_dimension must be 768")
        return cls(**data)


def runtime_identity() -> dict[str, object]:
    """Inspect installed distributions without importing or loading model code."""
    packages: dict[str, str | None] = {}
    for name in ("sentence-transformers", "transformers", "torch", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages}


def _identity(config: EmbeddingGemma2Config, runtime: dict[str, object]) -> dict[str, object]:
    prompt_fingerprint = _fingerprint(DEFAULT_EMBEDDINGGEMMA2_PROMPTS)
    model_version = f"{config.model_id}@{config.revision};runtime={_fingerprint(runtime)}"
    identity = {
        "adapter_version": EMBEDDINGGEMMA2_ADAPTER_VERSION,
        "backend": "embeddinggemma2", "model_id": config.model_id, "revision": config.revision,
        "model_version": model_version, "config": config.as_dict(), "runtime": runtime,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "prompt_set_fingerprint": prompt_fingerprint,
        "prompts": DEFAULT_EMBEDDINGGEMMA2_PROMPTS, "preprocessing": _PREPROCESSING,
        "score_policy": "max_prompt_per_domain;top_minus_best_different_domain;all_review;raw_cosine",
    }
    return {
        "backend": "embeddinggemma2", "model_id": config.model_id, "revision": config.revision,
        "model_version": model_version, "config": config.as_dict(), "runtime": runtime,
        "embedding_dimension": EMBEDDING_DIMENSION, "prompt_set_fingerprint": prompt_fingerprint,
        "fingerprint": _fingerprint(identity),
    }


def _load_sentence_transformer(config: EmbeddingGemma2Config) -> object:
    try:
        from packaging.version import Version

        if Version(importlib.metadata.version("sentence-transformers")) < Version("6.1.0"):
            raise RuntimeError("EmbeddingGemma 2 requires sentence-transformers >=6.1.0")
        if Version(importlib.metadata.version("transformers")) < Version("5.18.0"):
            raise RuntimeError("EmbeddingGemma 2 requires transformers >=5.18.0")
        import torch
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            config.model_id, revision=config.revision,
            device=None if config.device == "auto" else config.device,
            trust_remote_code=False, config_kwargs={"audio_config": None},
            model_kwargs={"torch_dtype": getattr(torch, config.dtype)},
        )
    except Exception as exc:
        raise RuntimeError(f"EmbeddingGemma 2 model loading failed: {exc}") from exc


def _embedding_batch(value: object, *, expected: int, kind: str) -> list[list[float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list | tuple) or len(value) != expected:
        raise RuntimeError(f"{kind} embedding batch cardinality does not match {expected}")
    output: list[list[float]] = []
    for row in value:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, list | tuple) or len(row) != EMBEDDING_DIMENSION:
            raise RuntimeError(f"{kind} embedding dimension must be {EMBEDDING_DIMENSION}")
        vector: list[float] = []
        for raw in row:
            if isinstance(raw, bool | str | bytes):
                raise RuntimeError(f"{kind} embedding must contain finite numeric values")
            try:
                number = float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(f"{kind} embedding must contain finite numeric values") from exc
            if not math.isfinite(number):
                raise RuntimeError(f"{kind} embedding must contain finite numeric values")
            vector.append(number)
        norm = math.hypot(*vector)
        if norm == 0.0 or not math.isfinite(norm):
            raise RuntimeError(f"{kind} embedding must have a finite nonzero norm")
        output.append([number / norm for number in vector])
    return output


def _image_hash(image: DecodedImage) -> str:
    from biominer.vision.full_frame_attention import decoded_image_content_hash

    return decoded_image_content_hash(image)


def _domain_decision(domains: dict[str, float], config: EmbeddingGemma2Config) -> tuple[str, float, float, str]:
    ranked = sorted(domains.items(), key=lambda item: item[1], reverse=True)
    domain, score = ranked[0]
    margin = score - ranked[1][1]
    reason = "provisional_embedding_route"
    if score < config.min_similarity:
        domain, reason = "ambiguous_visual_domain", "ambiguous_similarity"
    elif margin < config.min_margin:
        domain, reason = "ambiguous_visual_domain", "ambiguous_margin"
    return domain, score, margin, reason


class EmbeddingGemma2Router:
    backend = "embeddinggemma2"

    def __init__(self, config: EmbeddingGemma2Config | None = None, *, model_factory: Callable[..., object] | None = None) -> None:
        self._config = config or EmbeddingGemma2Config()
        self._metadata = _identity(self.config, runtime_identity())
        self._model_factory = model_factory
        self._model: Any = None
        self._text_embeddings: list[list[float]] | None = None
        self._closed = False
        self._lock = Lock()

    @property
    def config(self) -> EmbeddingGemma2Config:
        return self._config

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def revision(self) -> str:
        return self.config.revision

    @property
    def model_version(self) -> str:
        return str(self._metadata["model_version"])

    @property
    def fingerprint(self) -> str:
        return str(self._metadata["fingerprint"])

    @property
    def prompt_set_fingerprint(self) -> str:
        return str(self._metadata["prompt_set_fingerprint"])

    def metadata(self) -> dict[str, object]:
        return json.loads(json.dumps(self._metadata, allow_nan=False))

    def load(self) -> None:
        """Load frozen text and vision weights explicitly (also used by prefetch)."""
        with self._lock:
            self._load()

    def _load(self) -> None:
        if self._closed:
            raise RuntimeError("EmbeddingGemma 2 router is closed")
        if self._model is not None:
            return
        try:
            if self._model_factory is None:
                model = _load_sentence_transformer(self.config)
            else:
                model = self._model_factory(
                    model_id=self.model_id, revision=self.revision, device=self.config.device,
                    dtype=self.config.dtype, trust_remote_code=False, config_kwargs={"audio_config": None},
                )
            model.requires_grad_(False)
            model.eval()
            self._model = model
        except Exception as exc:
            raise RuntimeError(f"EmbeddingGemma 2 frozen model initialization failed: {exc}") from exc

    def route_batch(self, images: Sequence[DecodedImage]) -> list[dict[str, object]]:
        if self._closed:
            raise RuntimeError("EmbeddingGemma 2 router is closed")
        if not images:
            return []
        if any(not isinstance(image, DecodedImage) for image in images):
            raise ValueError("route_batch requires complete RGB DecodedImage inputs")
        from PIL import Image

        with self._lock:
            self._load()
            try:
                kwargs = {"normalize_embeddings": True, "convert_to_numpy": True, "show_progress_bar": False, "truncate_dim": EMBEDDING_DIMENSION}
                if self._text_embeddings is None:
                    encoded = self._model.encode([text for text, _ in DEFAULT_EMBEDDINGGEMMA2_PROMPTS], prompt_name="Classification", **kwargs)
                    self._text_embeddings = _embedding_batch(encoded, expected=len(DEFAULT_EMBEDDINGGEMMA2_PROMPTS), kind="text")
                payload = [{"image": Image.frombytes("RGB", (image.width, image.height), image.data)} for image in images]
                encoded_images = self._model.encode(payload, prompt="", **kwargs)
                embeddings = _embedding_batch(encoded_images, expected=len(images), kind="image")
                return [self._route(image, vector) for image, vector in zip(images, embeddings, strict=True)]
            except Exception as exc:
                raise RuntimeError(f"EmbeddingGemma 2 embedding inference failed: {exc}") from exc

    def _route(self, image: DecodedImage, vector: list[float]) -> dict[str, object]:
        assert self._text_embeddings is not None
        similarities: list[dict[str, object]] = []
        domains: dict[str, float] = {}
        for (text, domain), reference in zip(DEFAULT_EMBEDDINGGEMMA2_PROMPTS, self._text_embeddings, strict=True):
            score = max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(vector, reference, strict=True))))
            similarities.append({"text": text, "domain": domain, "score": score})
            domains[domain] = max(score, domains.get(domain, -1.0))
        domain, score, margin, reason = _domain_decision(domains, self.config)
        return {
            "embedding": vector, "image_content_hash": _image_hash(image),
            "proposed_domain": domain, "bioclip_route": _BIOCLIP_MAPPING.get(domain),
            "routing_action": "review", "routing_reason": reason,
            "cosine_similarity": score, "similarity_margin": margin,
            "prompt_similarities_json": json.dumps(similarities, separators=(",", ":"), allow_nan=False),
            "evidence_kind": "full_frame_embedding", "localization_status": "unavailable",
            **dict.fromkeys(_UNAVAILABLE_FIELDS),
        }

    def close(self) -> None:
        with self._lock:
            self._model = None
            self._text_embeddings = None
            self._closed = True

    def __enter__(self) -> EmbeddingGemma2Router:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class EmbeddingGemma2SidecarRouter:
    """One owned persistent worker with bounded protocol waits and strict identity."""

    backend = "embeddinggemma2"

    def __init__(self, runtime_python: str | Path, config: EmbeddingGemma2Config | None = None, *, popen: Callable[..., Any] = subprocess.Popen, timeout_seconds: float = 600.0) -> None:
        self.runtime_python = str(runtime_python)
        if not self.runtime_python.strip():
            raise ValueError("runtime_python must be non-empty")
        if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        self._config = config or EmbeddingGemma2Config()
        self._popen = popen
        self._timeout = timeout_seconds
        self._process: Any = None
        self._responses: Queue[str | None] = Queue()
        self._stderr: deque[str] = deque(maxlen=30)
        self._metadata: dict[str, object] | None = None
        self._closed = False
        self._request_id = 0
        self._lock = Lock()
        self.worker_process_starts = 0
        self.worker_request_count = 0

    @property
    def config(self) -> EmbeddingGemma2Config:
        return self._config

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def revision(self) -> str:
        return self.config.revision

    @property
    def fingerprint(self) -> str:
        return str(self.metadata()["fingerprint"])

    @property
    def model_version(self) -> str:
        return str(self.metadata()["model_version"])

    @property
    def prompt_set_fingerprint(self) -> str:
        return _fingerprint(DEFAULT_EMBEDDINGGEMMA2_PROMPTS)

    def metadata(self) -> dict[str, object]:
        with self._lock:
            self._ensure_process()
            assert self._metadata is not None
            return json.loads(json.dumps(self._metadata, allow_nan=False))

    def load(self) -> None:
        with self._lock:
            self._ensure_process()
            self._request({"operation": "load"})

    def _ensure_process(self) -> None:
        if self._closed:
            raise RuntimeError("EmbeddingGemma 2 sidecar is closed")
        if self._process is not None:
            return
        try:
            process = self._popen(
                [self.runtime_python, "-m", "biominer.detection.embeddinggemma2_worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", bufsize=1, env=embeddinggemma2_runtime_env(self.runtime_python),
            )
            self._process = process
            self.worker_process_starts += 1
            if process.stdin is None or process.stdout is None:
                raise RuntimeError("EmbeddingGemma 2 sidecar requires stdin/stdout pipes")
            Thread(target=self._read_stdout, args=(process.stdout,), daemon=True).start()
            if process.stderr is not None:
                Thread(target=self._read_stderr, args=(process.stderr,), daemon=True).start()
            self._request({"operation": "hello", "config": self.config.as_dict()})
        except Exception as exc:
            self._stop_process()
            raise RuntimeError(f"EmbeddingGemma 2 worker initialization failed: {exc}") from exc

    def _read_stdout(self, stream: Any) -> None:
        try:
            for line in stream:
                self._responses.put(line)
        finally:
            self._responses.put(None)

    def _read_stderr(self, stream: Any) -> None:
        for line in stream:
            self._stderr.append(line.rstrip())

    def _request(self, payload: dict[str, object]) -> dict[str, object]:
        assert self._process is not None
        self._request_id += 1
        request = {**payload, "protocol": EMBEDDINGGEMMA2_PROTOCOL_VERSION, "request_id": self._request_id}
        try:
            if self._process.poll() is not None:
                raise RuntimeError("worker exited before request")
            deadline = monotonic() + self._timeout
            write_result: Queue[Exception | None] = Queue()
            message = json.dumps(request, allow_nan=False) + "\n"
            Thread(target=_write_message, args=(self._process.stdin, message, write_result), daemon=True).start()
            try:
                error = write_result.get(timeout=max(0.0, deadline - monotonic()))
            except Empty as exc:
                raise RuntimeError("worker request write timed out") from exc
            if error is not None:
                raise RuntimeError(f"worker request write failed: {error}") from error
            self.worker_request_count += 1
            try:
                line = self._responses.get(timeout=max(0.0, deadline - monotonic()))
            except Empty as exc:
                raise RuntimeError("worker response timed out") from exc
            if line is None:
                raise RuntimeError("worker closed stdout: " + "\n".join(self._stderr))
            response = _strict_json_loads(line)
            if not isinstance(response, dict):
                raise RuntimeError("worker protocol response must be an object")
            response_id = response.get("request_id")
            if isinstance(response_id, bool) or not isinstance(response_id, int) or response_id <= 0:
                raise RuntimeError("worker request identity must be a positive integer")
            if response.get("protocol") != EMBEDDINGGEMMA2_PROTOCOL_VERSION or response_id != self._request_id:
                raise RuntimeError("worker protocol or request identity mismatch")
            if "error" in response:
                raise RuntimeError(f"worker inference failed: {response['error']}")
            self._validate_metadata(response.get("metadata"))
            return response
        except Exception as exc:
            self._stop_process()
            self._closed = True
            raise RuntimeError(f"EmbeddingGemma 2 sidecar request failed: {exc}") from exc

    def _validate_metadata(self, value: object) -> None:
        if not isinstance(value, dict) or not isinstance(value.get("runtime"), dict):
            raise RuntimeError("worker metadata is missing runtime identity")
        expected = _identity(self.config, value["runtime"])
        if value != expected:
            raise RuntimeError("worker metadata model/revision/fingerprint identity mismatch")
        if self._metadata is not None and value != self._metadata:
            raise RuntimeError("worker metadata changed during the session")
        self._metadata = value

    def route_batch(self, images: Sequence[DecodedImage]) -> list[dict[str, object]]:
        if self._closed:
            raise RuntimeError("EmbeddingGemma 2 sidecar is closed")
        if not images:
            return []
        if any(not isinstance(image, DecodedImage) for image in images):
            raise ValueError("route_batch requires complete RGB DecodedImage inputs")
        payload = [{"width": image.width, "height": image.height, "mode": image.mode, "data_b64": base64.b64encode(image.data).decode("ascii")} for image in images]
        with self._lock:
            self._ensure_process()
            response = self._request({"operation": "route", "images": payload})
            rows = response.get("results")
            try:
                if not isinstance(rows, list) or len(rows) != len(images):
                    raise RuntimeError("worker result cardinality does not match image batch")
                for image, row in zip(images, rows, strict=True):
                    _validate_result(row, image=image, config=self.config)
                return rows
            except Exception as exc:
                self._stop_process()
                self._closed = True
                raise RuntimeError(f"EmbeddingGemma 2 sidecar invalid result: {exc}") from exc

    def _stop_process(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            for name in ("stdin", "stdout", "stderr"):
                stream = getattr(process, name, None)
                if stream is not None:
                    stream.close()

    def close(self) -> None:
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    process.stdin.write(json.dumps({"protocol": EMBEDDINGGEMMA2_PROTOCOL_VERSION, "operation": "shutdown"}) + "\n")
                    process.stdin.flush()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            self._stop_process()
            self._closed = True

    def __enter__(self) -> EmbeddingGemma2SidecarRouter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON value: {value}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_loads(value: str) -> Any:
    return json.loads(value, parse_constant=_reject_constant, object_pairs_hook=_unique_json_object)


def _write_message(stream: Any, message: str, outcome: Queue[Exception | None]) -> None:
    try:
        stream.write(message)
        stream.flush()
    except Exception as exc:
        outcome.put(exc)
    else:
        outcome.put(None)


def _validate_result(row: object, *, image: DecodedImage, config: EmbeddingGemma2Config) -> None:
    if not isinstance(row, dict):
        raise RuntimeError("worker result must be an evidence object")
    if row.get("image_content_hash") != _image_hash(image):
        raise RuntimeError("worker returned a different image content identity")
    if row.get("proposed_domain") not in DETECTION_ROUTES or row.get("bioclip_route") != _BIOCLIP_MAPPING.get(row.get("proposed_domain")):
        raise RuntimeError("worker returned an invalid domain route")
    if row.get("routing_action") != "review" or row.get("routing_reason") not in {"ambiguous_similarity", "ambiguous_margin", "provisional_embedding_route"}:
        raise RuntimeError("worker returned invalid provisional review semantics")
    if row.get("evidence_kind") != "full_frame_embedding" or row.get("localization_status") != "unavailable":
        raise RuntimeError("worker returned invalid full-frame localization semantics")
    if any(field not in row or row[field] is not None for field in _UNAVAILABLE_FIELDS):
        raise RuntimeError("worker fabricated unavailable localization or object measurements")
    _embedding_batch([row.get("embedding")], expected=1, kind="worker image")
    if not math.isclose(math.hypot(*row["embedding"]), 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise RuntimeError("worker image embedding is not normalized")
    for key, low, high in (("cosine_similarity", -1, 1), ("similarity_margin", 0, 2)):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or not low <= value <= high:
            raise RuntimeError(f"worker {key} must contain finite raw evidence")
    scores = _strict_json_loads(row.get("prompt_similarities_json", ""))
    if not isinstance(scores, list) or len(scores) != len(DEFAULT_EMBEDDINGGEMMA2_PROMPTS):
        raise RuntimeError("worker prompt similarities do not cover the fixed prompt universe")
    domains: dict[str, float] = {}
    for score, (text, domain) in zip(scores, DEFAULT_EMBEDDINGGEMMA2_PROMPTS, strict=True):
        if not isinstance(score, dict) or score.get("text") != text or score.get("domain") != domain:
            raise RuntimeError("worker prompt similarities changed the fixed prompt universe")
        value = score.get("score")
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or not -1 <= value <= 1:
            raise RuntimeError("worker prompt similarity must be a finite raw cosine")
        domains[domain] = max(value, domains.get(domain, -1.0))
    domain, similarity, margin, reason = _domain_decision(domains, config)
    if row["proposed_domain"] != domain or row["routing_reason"] != reason:
        raise RuntimeError("worker domain or reason conflicts with its prompt score evidence")
    if not math.isclose(row["cosine_similarity"], similarity, abs_tol=1e-10) or not math.isclose(row["similarity_margin"], margin, abs_tol=1e-10):
        raise RuntimeError("worker score evidence conflicts with its prompt similarities")
