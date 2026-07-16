from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from math import isfinite
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

from biominer.common.semantic_hash import canonical_semantic_fingerprint


DEFAULT_DEVICE = "auto"
VALID_DEVICES = {"auto", "cuda", "mps", "cpu"}
VALID_IMAGE_RESIZE_MODES = {"shortest", "longest", "squash"}
DEFAULT_TEXT_FEATURE_BATCH_SIZE = 512
TEXT_FEATURE_BATCH_SIZE_ENV = "BIOMINER_BIOCLIP_TEXT_FEATURE_BATCH_SIZE"
DEFAULT_TEXT_FEATURE_CACHE_ENTRIES = 8
TEXT_FEATURE_CACHE_ENTRIES_ENV = "BIOMINER_BIOCLIP_TEXT_FEATURE_CACHE_ENTRIES"
OPENCLIP_PREPROCESSING_ATTESTATION_VERSION = "openclip-preprocessing-attestation-v2"
DECODED_IMAGE_CONTENT_HASH_VERSION = "decoded-image-content-v1"
_HF_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MPS_MEMORY_FIELDS = (
    "mps_current_allocated_memory",
    "mps_driver_allocated_memory",
    "mps_recommended_max_memory",
)


@dataclass(frozen=True, slots=True)
class OpenClipModelSource:
    loader_model_name: str
    pretrained: str | None
    require_pretrained: bool
    model_id: str
    model_revision: str
    model_weights_sha256: str | None
    open_clip_config_sha256: str | None


def main() -> None:
    if "--persistent" in sys.argv[1:]:
        run_persistent_worker()
        return
    request = json.loads(sys.stdin.read())
    configure_hf_cache_env(
        Path(request.get("hf_cache_dir") or "data/cache/huggingface")
    )
    image_paths = request.get("image_paths")
    device = device_from_request(request)
    image_resize_mode = image_resize_mode_from_request(request)
    preprocess_workers = preprocess_workers_from_request(request)
    label_sets = request.get("label_sets")
    if label_sets is not None:
        scores_by_image_by_label_set = score_image_label_sets(
            image_paths=[Path(path) for path in image_paths],
            label_sets={str(name): list(labels) for name, labels in label_sets.items()},
            model_name=request["model_name"],
            checkpoint=request["checkpoint"],
            device=device,
            image_resize_mode=image_resize_mode,
            preprocess_workers=preprocess_workers,
        )
        print(
            json.dumps(
                {
                    "scores_by_image_by_label_set": scores_by_image_by_label_set,
                    "image_resize_mode": image_resize_mode,
                },
                sort_keys=True,
            )
        )
        return
    if image_paths is None:
        scores = score_image(
            image_path=Path(request["image_path"]),
            labels=request["labels"],
            model_name=request["model_name"],
            checkpoint=request["checkpoint"],
            device=device,
            image_resize_mode=image_resize_mode,
            preprocess_workers=preprocess_workers,
        )
        print(
            json.dumps(
                {"scores": scores, "image_resize_mode": image_resize_mode},
                sort_keys=True,
            )
        )
        return
    scores_by_image = score_images(
        image_paths=[Path(path) for path in image_paths],
        labels=request["labels"],
        model_name=request["model_name"],
        checkpoint=request["checkpoint"],
        device=device,
        image_resize_mode=image_resize_mode,
        preprocess_workers=preprocess_workers,
    )
    print(
        json.dumps(
            {
                "scores_by_image": scores_by_image,
                "image_resize_mode": image_resize_mode,
            },
            sort_keys=True,
        )
    )


def configure_hf_cache_env(cache_dir: str | Path) -> Path:
    cache_path = Path(cache_dir).resolve()
    hub_path = cache_path / "hub"
    hub_path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_path)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_path)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return cache_path


def device_from_request(request: dict[str, object]) -> str:
    if "device" in request and request["device"] not in (None, ""):
        return normalize_device(str(request["device"]))
    return DEFAULT_DEVICE


def preprocess_workers_from_request(request: dict[str, object]) -> int:
    value = int(request.get("preprocess_workers") or 1)
    if value <= 0:
        raise ValueError("preprocess_workers must be positive")
    return value


def image_resize_mode_from_request(request: dict[str, object]) -> str | None:
    value = request.get("image_resize_mode")
    if value is None:
        return None
    return normalize_image_resize_mode(str(value))


def normalize_device(device: str) -> str:
    normalized = device.casefold().strip()
    if normalized not in VALID_DEVICES:
        raise ValueError(
            f"Unsupported BioCLIP device {device!r}; expected one of {sorted(VALID_DEVICES)}"
        )
    return normalized


def normalize_image_resize_mode(image_resize_mode: str | None) -> str | None:
    if image_resize_mode is None:
        return None
    normalized = image_resize_mode.casefold().strip()
    if normalized not in VALID_IMAGE_RESIZE_MODES:
        raise ValueError(
            f"Unsupported BioCLIP image resize mode {image_resize_mode!r}; "
            f"expected one of {sorted(VALID_IMAGE_RESIZE_MODES)}"
        )
    return normalized


def run_persistent_worker() -> None:
    loaded: _LoadedBioClipModel | None = None
    loaded_key: tuple[str, str, str, str | None] | None = None
    request_count = 0
    model_load_count = 0
    model_cache_hit_count = 0
    model_refresh_count = 0
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("shutdown"):
                    return
                request_count += 1
                configure_hf_cache_env(
                    Path(request.get("hf_cache_dir") or "data/cache/huggingface")
                )
                device = device_from_request(request)
                image_resize_mode = image_resize_mode_from_request(request)
                preprocess_workers = preprocess_workers_from_request(request)
                key = (
                    str(request["model_name"]),
                    str(request["checkpoint"]),
                    device,
                    image_resize_mode,
                )
                model_cache_hit = loaded is not None and loaded_key == key
                if not model_cache_hit:
                    if loaded is not None:
                        loaded.close()
                        model_refresh_count += 1
                    loaded = _LoadedBioClipModel.load(
                        model_name=key[0],
                        checkpoint=key[1],
                        device=key[2],
                        image_resize_mode=key[3],
                    )
                    loaded_key = key
                    model_load_count += 1
                else:
                    model_cache_hit_count += 1
                progress = {
                    "worker_request_count": request_count,
                    "model_load_count": model_load_count,
                    "model_cache_hit_count": model_cache_hit_count,
                    "model_refresh_count": model_refresh_count,
                    "model_cache_hit": model_cache_hit,
                }
                if not model_cache_hit:
                    print(
                        json.dumps(
                            {
                                "ready": True,
                                **loaded.worker_metadata,
                                **progress,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if request.get("probe") is True:
                    print(
                        json.dumps(
                            {
                                "probed": True,
                                **loaded.worker_metadata,
                                **progress,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    continue
                text_labels = request.get("text_labels")
                if text_labels is not None:
                    embeddings = loaded.text_embeddings(list(text_labels))
                    print(
                        json.dumps(
                            {
                                "text_embeddings": embeddings,
                                "embedding_dim": len(embeddings[0])
                                if embeddings
                                else 0,
                                **loaded.worker_metadata,
                                **progress,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    continue
                image_embedding_paths = request.get("image_embedding_paths")
                if image_embedding_paths is not None:
                    embeddings, image_content_hashes = loaded.image_embeddings(
                        [Path(path) for path in image_embedding_paths],
                        preprocess_workers=preprocess_workers,
                    )
                    print(
                        json.dumps(
                            {
                                "image_embeddings": embeddings,
                                "embedding_dim": len(embeddings[0])
                                if embeddings
                                else 0,
                                "image_content_hashes": image_content_hashes,
                                **loaded.worker_metadata,
                                **progress,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    continue
                image_paths = request.get("image_paths")
                label_sets = request.get("label_sets")
                if label_sets is not None:
                    scores_by_image_by_label_set = loaded.score_image_label_sets(
                        [Path(path) for path in image_paths],
                        {
                            str(name): list(labels)
                            for name, labels in label_sets.items()
                        },
                        preprocess_workers=preprocess_workers,
                    )
                    print(
                        json.dumps(
                            {
                                "scores_by_image_by_label_set": scores_by_image_by_label_set,
                                **loaded.worker_metadata,
                                **progress,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    continue
                if image_paths is None:
                    scores = loaded.score_images(
                        [Path(request["image_path"])],
                        request["labels"],
                        preprocess_workers=preprocess_workers,
                    )[0]
                    print(
                        json.dumps(
                            {
                                "scores": scores,
                                **loaded.worker_metadata,
                                **progress,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    continue
                scores_by_image = loaded.score_images(
                    [Path(path) for path in image_paths],
                    request["labels"],
                    preprocess_workers=preprocess_workers,
                )
                print(
                    json.dumps(
                        {
                            "scores_by_image": scores_by_image,
                            **loaded.worker_metadata,
                            **progress,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - worker reports errors to controller process.
                print(json.dumps({"error": str(exc)}, sort_keys=True), flush=True)
    finally:
        if loaded is not None:
            loaded.close()


def score_image(
    *,
    image_path: Path,
    labels: Sequence[str],
    model_name: str,
    checkpoint: str,
    device: str = DEFAULT_DEVICE,
    image_resize_mode: str | None = None,
    preprocess_workers: int = 1,
) -> dict[str, float]:
    return score_images(
        image_paths=[image_path],
        labels=labels,
        model_name=model_name,
        checkpoint=checkpoint,
        device=normalize_device(device),
        image_resize_mode=image_resize_mode,
        preprocess_workers=preprocess_workers,
    )[0]


def score_images(
    *,
    image_paths: Sequence[Path],
    labels: Sequence[str],
    model_name: str,
    checkpoint: str,
    device: str = DEFAULT_DEVICE,
    image_resize_mode: str | None = None,
    preprocess_workers: int = 1,
) -> list[dict[str, float]]:
    model = _LoadedBioClipModel.load(
        model_name=model_name,
        checkpoint=checkpoint,
        device=normalize_device(device),
        image_resize_mode=image_resize_mode,
    )
    return model.score_images(
        image_paths, labels, preprocess_workers=preprocess_workers
    )


def score_image_label_sets(
    *,
    image_paths: Sequence[Path],
    label_sets: dict[str, Sequence[str]],
    model_name: str,
    checkpoint: str,
    device: str = DEFAULT_DEVICE,
    image_resize_mode: str | None = None,
    preprocess_workers: int = 1,
) -> dict[str, list[dict[str, float]]]:
    model = _LoadedBioClipModel.load(
        model_name=model_name,
        checkpoint=checkpoint,
        device=normalize_device(device),
        image_resize_mode=image_resize_mode,
    )
    return model.score_image_label_sets(
        image_paths, label_sets, preprocess_workers=preprocess_workers
    )


class _LoadedBioClipModel:
    def __init__(
        self,
        *,
        model,
        preprocess,
        tokenizer,
        torch,
        device: str,
        gpu_name: str,
        image_resize_mode: str | None = None,
        model_id: str = "",
        model_revision: str = "",
        model_weights_sha256: str | None = None,
        open_clip_version: str = "",
        open_clip_config_sha256: str | None = None,
        preprocessing_version: str = "",
        preprocessing_config: Mapping[str, object] | None = None,
        preprocessing_fingerprint: str | None = None,
    ) -> None:  # noqa: ANN001 - external runtime objects.
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.torch = torch
        self.device = device
        self.gpu_name = gpu_name
        self.image_resize_mode = normalize_image_resize_mode(image_resize_mode)
        self.model_id = str(model_id)
        self.model_revision = str(model_revision)
        self.model_weights_sha256 = model_weights_sha256
        self.open_clip_version = str(open_clip_version)
        self.open_clip_config_sha256 = open_clip_config_sha256
        self.preprocessing_version = str(preprocessing_version)
        self.preprocessing_config = (
            canonical_preprocessing_config(preprocessing_config)
            if preprocessing_config is not None
            else None
        )
        self.preprocessing_fingerprint = preprocessing_fingerprint
        self._text_features_by_labels: OrderedDict[tuple[str, ...], object] = (
            OrderedDict()
        )
        self._mps_peak_current_allocated_memory = 0
        self._mps_peak_driver_allocated_memory = 0

    @classmethod
    def load(
        cls,
        *,
        model_name: str,
        checkpoint: str,
        device: str = DEFAULT_DEVICE,
        image_resize_mode: str | None = None,
    ) -> "_LoadedBioClipModel":
        normalized_resize_mode = normalize_image_resize_mode(image_resize_mode)
        try:
            import open_clip
            import torch
        except Exception as exc:  # noqa: BLE001 - executed in the external model runtime.
            raise RuntimeError(f"BioCLIP dependencies are unavailable: {exc}") from exc

        resolved_device = resolve_torch_device(torch, device)
        gpu_name = torch_device_name(torch, resolved_device)
        model_source = resolve_open_clip_model_source(model_name, checkpoint)
        transform_kwargs: dict[str, object] = {
            "pretrained": model_source.pretrained,
            "require_pretrained": model_source.require_pretrained,
        }
        if normalized_resize_mode is not None:
            transform_kwargs["image_resize_mode"] = normalized_resize_mode
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_source.loader_model_name, **transform_kwargs
        )
        tokenizer = open_clip.get_tokenizer(model_source.loader_model_name)
        preprocessing_config = canonical_preprocessing_config(
            open_clip.get_model_preprocess_cfg(model)
        )
        open_clip_version = _required_text(
            getattr(open_clip, "__version__", ""),
            field="OpenCLIP package version",
        )
        preprocessing_fingerprint = preprocessing_attestation_fingerprint(
            open_clip_config_sha256=model_source.open_clip_config_sha256,
            open_clip_version=open_clip_version,
            preprocessing_config=preprocessing_config,
            preprocessing_version=OPENCLIP_PREPROCESSING_ATTESTATION_VERSION,
        )
        effective_resize_mode = normalize_image_resize_mode(
            str(preprocessing_config.get("resize_mode") or "")
        )
        model = model.to(resolved_device)
        model.eval()
        return cls(
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            torch=torch,
            device=resolved_device,
            gpu_name=gpu_name,
            image_resize_mode=effective_resize_mode,
            model_id=model_source.model_id,
            model_revision=model_source.model_revision,
            model_weights_sha256=model_source.model_weights_sha256,
            open_clip_version=open_clip_version,
            open_clip_config_sha256=model_source.open_clip_config_sha256,
            preprocessing_version=OPENCLIP_PREPROCESSING_ATTESTATION_VERSION,
            preprocessing_config=preprocessing_config,
            preprocessing_fingerprint=preprocessing_fingerprint,
        )

    @property
    def worker_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "device": self.device,
            "gpu_name": self.gpu_name,
            "image_resize_mode": self.image_resize_mode,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "open_clip_version": self.open_clip_version,
            "open_clip_config_sha256": self.open_clip_config_sha256,
            "preprocessing_version": self.preprocessing_version,
            "preprocessing_config": self.preprocessing_config,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            **self._memory_metadata(),
        }
        if self.model_weights_sha256 is not None:
            metadata["model_weights_sha256"] = self.model_weights_sha256
        return metadata

    def _memory_metadata(self) -> dict[str, object]:
        snapshot = mps_memory_snapshot(self.torch, self.device)
        current = snapshot["mps_current_allocated_memory"]
        driver = snapshot["mps_driver_allocated_memory"]
        if isinstance(current, int):
            self._mps_peak_current_allocated_memory = max(
                self._mps_peak_current_allocated_memory,
                current,
            )
        if isinstance(driver, int):
            self._mps_peak_driver_allocated_memory = max(
                self._mps_peak_driver_allocated_memory,
                driver,
            )
        return {
            **snapshot,
            "mps_peak_current_allocated_memory": (
                self._mps_peak_current_allocated_memory
                if isinstance(current, int)
                else current
            ),
            "mps_peak_driver_allocated_memory": (
                self._mps_peak_driver_allocated_memory
                if isinstance(driver, int)
                else driver
            ),
        }

    def score_images(
        self,
        image_paths: Sequence[Path],
        labels: Sequence[str],
        *,
        preprocess_workers: int = 1,
    ) -> list[dict[str, float]]:
        try:
            from PIL import Image
        except Exception as exc:  # noqa: BLE001 - executed in the external model runtime.
            raise RuntimeError(f"BioCLIP dependencies are unavailable: {exc}") from exc

        text_features = self._text_features(labels)
        with self.torch.no_grad():
            scores_by_image: list[dict[str, float]] = []
            image_batch = self._image_batch(
                image_paths, Image, preprocess_workers=preprocess_workers
            )
            image_features = self.model.encode_image(image_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            probabilities_by_image = (100.0 * image_features @ text_features.T).softmax(
                dim=-1
            )
            for probabilities in probabilities_by_image:
                scores_by_image.append(
                    {
                        label: float(probabilities[index].detach().cpu())
                        for index, label in enumerate(labels)
                    }
                )
        return scores_by_image

    def score_image_label_sets(
        self,
        image_paths: Sequence[Path],
        label_sets: dict[str, Sequence[str]],
        *,
        preprocess_workers: int = 1,
    ) -> dict[str, list[dict[str, float]]]:
        try:
            from PIL import Image
        except Exception as exc:  # noqa: BLE001 - executed in the external model runtime.
            raise RuntimeError(f"BioCLIP dependencies are unavailable: {exc}") from exc

        text_features_by_set = {
            name: self._text_features(labels) for name, labels in label_sets.items()
        }
        scores_by_label_set: dict[str, list[dict[str, float]]] = {
            name: [] for name in label_sets
        }
        with self.torch.no_grad():
            image_batch = self._image_batch(
                image_paths, Image, preprocess_workers=preprocess_workers
            )
            image_features = self.model.encode_image(image_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            for label_set_name, labels in label_sets.items():
                probabilities_by_image = (
                    100.0 * image_features @ text_features_by_set[label_set_name].T
                ).softmax(dim=-1)
                for probabilities in probabilities_by_image:
                    scores_by_label_set[label_set_name].append(
                        {
                            label: float(probabilities[index].detach().cpu())
                            for index, label in enumerate(labels)
                        }
                    )
        return scores_by_label_set

    def text_embeddings(self, labels: Sequence[str]) -> list[list[float]]:
        text_features = self._text_features(labels, cache=False)
        return [
            [float(value) for value in row.detach().cpu().tolist()]
            for row in text_features
        ]

    def image_embeddings(
        self, image_paths: Sequence[Path], *, preprocess_workers: int = 1
    ) -> tuple[list[list[float]], list[str]]:
        try:
            from PIL import Image
        except Exception as exc:  # noqa: BLE001 - executed in the external model runtime.
            raise RuntimeError(f"BioCLIP dependencies are unavailable: {exc}") from exc

        inference_mode = getattr(self.torch, "inference_mode", self.torch.no_grad)
        with inference_mode():
            image_batch, image_content_hashes = self._image_batch_with_content_hashes(
                image_paths, Image, preprocess_workers=preprocess_workers
            )
            image_features = self.model.encode_image(image_batch, normalize=True)
        return (
            [
                [float(value) for value in row.detach().cpu().tolist()]
                for row in image_features
            ],
            image_content_hashes,
        )

    def _image_batch_with_content_hashes(
        self,
        image_paths: Sequence[Path],
        image_module,
        *,
        preprocess_workers: int = 1,
    ):  # noqa: ANN001 - PIL module and external tensor.
        workers = int(preprocess_workers)
        if workers <= 0:
            raise ValueError("preprocess_workers must be positive")
        if workers == 1 or len(image_paths) <= 1:
            prepared = [
                self._preprocess_image_path_with_content_hash(
                    image_path,
                    image_module,
                )
                for image_path in image_paths
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                prepared = list(
                    executor.map(
                        lambda image_path: (
                            self._preprocess_image_path_with_content_hash(
                                image_path,
                                image_module,
                            )
                        ),
                        image_paths,
                    )
                )
        images = [image for image, _content_hash in prepared]
        content_hashes = [content_hash for _image, content_hash in prepared]
        return self.torch.stack(images).to(self.device), content_hashes

    def _image_batch(
        self, image_paths: Sequence[Path], image_module, *, preprocess_workers: int = 1
    ):  # noqa: ANN001 - PIL module.
        workers = int(preprocess_workers)
        if workers <= 0:
            raise ValueError("preprocess_workers must be positive")
        if workers == 1 or len(image_paths) <= 1:
            images = [
                self._preprocess_image_path(image_path, image_module)
                for image_path in image_paths
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                images = list(
                    executor.map(
                        lambda image_path: self._preprocess_image_path(
                            image_path, image_module
                        ),
                        image_paths,
                    )
                )
        return self.torch.stack(images).to(self.device)

    def _preprocess_image_path(self, image_path: Path, image_module):  # noqa: ANN001 - PIL module.
        image = image_module.open(image_path)
        try:
            return self.preprocess(image.convert("RGB"))
        finally:
            close = getattr(image, "close", None)
            if callable(close):
                close()

    def _preprocess_image_path_with_content_hash(
        self,
        image_path: Path,
        image_module,
    ):  # noqa: ANN001 - PIL module and external tensor.
        source_image = image_module.open(image_path)
        rgb_image = None
        try:
            rgb_image = source_image.convert("RGB")
            rgb_image.load()
            content_hash = decoded_rgb_image_content_hash(rgb_image)
            return self.preprocess(rgb_image), content_hash
        finally:
            if rgb_image is not None and rgb_image is not source_image:
                _close_image(rgb_image)
            _close_image(source_image)

    def _text_features(self, labels: Sequence[str], *, cache: bool = True):
        label_key = tuple(labels)
        if cache:
            cached = self._text_features_by_labels.pop(label_key, None)
            if cached is not None:
                self._text_features_by_labels[label_key] = cached
                return cached
        label_list = list(labels)
        batch_size = text_feature_batch_size()
        if len(label_list) <= batch_size:
            text_features = self._encode_text_features(label_list)
        else:
            batches = [
                self._encode_text_features(label_list[start : start + batch_size])
                for start in range(0, len(label_list), batch_size)
            ]
            text_features = self.torch.cat(batches, dim=0)
        if cache:
            self._text_features_by_labels[label_key] = text_features
            while len(self._text_features_by_labels) > text_feature_cache_entries():
                self._text_features_by_labels.popitem(last=False)
        return text_features

    def _encode_text_features(self, labels: Sequence[str]):
        text = self.tokenizer(list(labels)).to(self.device)
        with self.torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def close(self) -> None:
        self._text_features_by_labels.clear()
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        accelerator = getattr(
            self.torch, "cuda" if self.device.startswith("cuda") else "mps", None
        )
        empty_cache = getattr(accelerator, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()


def text_feature_batch_size() -> int:
    raw_value = os.environ.get(TEXT_FEATURE_BATCH_SIZE_ENV)
    if raw_value is None:
        return DEFAULT_TEXT_FEATURE_BATCH_SIZE
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{TEXT_FEATURE_BATCH_SIZE_ENV} must be a positive integer"
        ) from exc
    if parsed <= 0:
        raise ValueError(f"{TEXT_FEATURE_BATCH_SIZE_ENV} must be a positive integer")
    return parsed


def text_feature_cache_entries() -> int:
    raw_value = os.environ.get(TEXT_FEATURE_CACHE_ENTRIES_ENV)
    if raw_value is None:
        return DEFAULT_TEXT_FEATURE_CACHE_ENTRIES
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{TEXT_FEATURE_CACHE_ENTRIES_ENV} must be a positive integer"
        ) from exc
    if parsed <= 0:
        raise ValueError(f"{TEXT_FEATURE_CACHE_ENTRIES_ENV} must be a positive integer")
    return parsed


def resolve_open_clip_model_source(
    model_name: str,
    checkpoint: str,
) -> OpenClipModelSource:
    normalized_model_name = str(model_name or "").strip()
    normalized_checkpoint = str(checkpoint or "").strip()
    if not normalized_model_name:
        raise ValueError("BioCLIP model name must be non-empty")
    if normalized_model_name.startswith("local-dir:"):
        raise ValueError(
            "BioCLIP local-dir model sources require an explicit canonical model identity"
        )

    model_id = normalized_model_name.removeprefix("hf-hub:")
    if "/" not in model_id:
        return OpenClipModelSource(
            loader_model_name=normalized_model_name,
            pretrained=normalized_checkpoint or None,
            require_pretrained=bool(normalized_checkpoint),
            model_id=normalized_model_name,
            model_revision=normalized_checkpoint,
            model_weights_sha256=None,
            open_clip_config_sha256=None,
        )

    if _HF_COMMIT_PATTERN.fullmatch(normalized_checkpoint) is None:
        raise ValueError(
            "Hugging Face BioCLIP checkpoints must be an immutable "
            "40-character commit revision"
        )
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001 - external model runtime dependency.
        raise RuntimeError(
            f"Hugging Face snapshot support is unavailable: {exc}"
        ) from exc
    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            repo_type="model",
            revision=normalized_checkpoint,
            local_files_only=True,
        )
    ).resolve(strict=True)
    if snapshot.name != normalized_checkpoint:
        raise ValueError(
            "Hugging Face snapshot resolved revision does not match the requested commit"
        )
    config_path = snapshot / "open_clip_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"prefetched BioCLIP snapshot is missing OpenCLIP config: {config_path}"
        )
    weights_path = snapshot / "open_clip_model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"prefetched BioCLIP snapshot is missing frozen weights: {weights_path}"
        )
    return OpenClipModelSource(
        loader_model_name=f"local-dir:{snapshot}",
        pretrained=None,
        require_pretrained=True,
        model_id=model_id,
        model_revision=normalized_checkpoint,
        model_weights_sha256=_file_sha256(weights_path),
        open_clip_config_sha256=_file_sha256(config_path),
    )


def canonical_preprocessing_config(config: object) -> dict[str, object]:
    if not isinstance(config, Mapping) or not config:
        raise ValueError("OpenCLIP preprocessing config must be a non-empty mapping")
    return _canonical_json_mapping(config, field="OpenCLIP preprocessing config")


def preprocessing_attestation_fingerprint(
    *,
    open_clip_config_sha256: str | None,
    open_clip_version: str,
    preprocessing_config: object,
    preprocessing_version: str,
) -> str:
    config_sha256 = _optional_sha256(
        open_clip_config_sha256,
        field="OpenCLIP config SHA-256",
    )
    payload = {
        "open_clip_config_sha256": config_sha256,
        "open_clip_version": _required_text(
            open_clip_version,
            field="OpenCLIP package version",
        ),
        "preprocessing_config": canonical_preprocessing_config(preprocessing_config),
        "preprocessing_version": _required_text(
            preprocessing_version,
            field="OpenCLIP preprocessing attestation version",
        ),
    }
    return canonical_semantic_fingerprint(payload)


def decoded_rgb_image_content_hash(image: object) -> str:
    if str(getattr(image, "mode", "")) != "RGB":
        raise ValueError("decoded image content hash requires RGB image data")
    size = getattr(image, "size", None)
    if (
        not isinstance(size, (tuple, list))
        or len(size) != 2
        or isinstance(size[0], bool)
        or isinstance(size[1], bool)
        or not isinstance(size[0], int)
        or not isinstance(size[1], int)
    ):
        raise ValueError("decoded image content hash requires a valid image size")
    width = size[0]
    height = size[1]
    if width <= 0 or height <= 0:
        raise ValueError("decoded image content hash requires positive dimensions")
    tobytes = getattr(image, "tobytes", None)
    if not callable(tobytes):
        raise TypeError("decoded image content hash requires image byte access")
    data = tobytes()
    if not isinstance(data, bytes) or len(data) != width * height * 3:
        raise ValueError("decoded image content hash requires packed RGB bytes")
    header = json.dumps(
        {
            "height": height,
            "mode": "RGB",
            "version": DECODED_IMAGE_CONTENT_HASH_VERSION,
            "width": width,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(data)
    return "sha256:" + digest.hexdigest()


def _close_image(image: object) -> None:
    close = getattr(image, "close", None)
    if callable(close):
        close()


def mps_memory_snapshot(torch, device: str) -> dict[str, int | str]:  # noqa: ANN001 - torch module.
    if not str(device).startswith("mps"):
        return {field: "not_applicable" for field in _MPS_MEMORY_FIELDS}
    mps = getattr(torch, "mps", None)
    snapshot: dict[str, int | str] = {}
    methods = {
        "mps_current_allocated_memory": "current_allocated_memory",
        "mps_driver_allocated_memory": "driver_allocated_memory",
        "mps_recommended_max_memory": "recommended_max_memory",
    }
    for field, method_name in methods.items():
        method = getattr(mps, method_name, None)
        if not callable(method):
            snapshot[field] = "not_instrumented"
            continue
        try:
            value = method()
        except Exception:  # noqa: BLE001 - optional runtime counter.
            snapshot[field] = "not_instrumented"
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            snapshot[field] = "not_instrumented"
            continue
        snapshot[field] = value
    return snapshot


def _canonical_json_mapping(
    value: Mapping[object, object],
    *,
    field: str,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ValueError(f"{field} keys must be non-empty strings")
        result[raw_key] = _canonical_json_value(
            raw_value,
            field=f"{field}.{raw_key}",
        )
    return {key: result[key] for key in sorted(result)}


def _canonical_json_value(value: object, *, field: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{field} must be finite")
        return value
    if isinstance(value, Mapping):
        return _canonical_json_mapping(value, field=field)
    if isinstance(value, (list, tuple)):
        return [
            _canonical_json_value(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{field} contains unsupported value {type(value).__name__}")


def _required_text(value: object, *, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} must be non-empty")
    return normalized


def _optional_sha256(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{field} must be a valid SHA-256")
    return normalized


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    value = "sha256:" + digest.hexdigest()
    if _SHA256_PATTERN.fullmatch(value) is None:  # pragma: no cover - defensive.
        raise AssertionError("invalid SHA-256 digest")
    return value


def resolve_torch_device(torch, requested_device: str = DEFAULT_DEVICE) -> str:  # noqa: ANN001 - torch module.
    device = normalize_device(requested_device)
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available(torch):
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "BioCLIP requested CUDA, but torch.cuda.is_available() is false"
        )
    if device == "mps" and not _mps_available(torch):
        raise RuntimeError(
            "BioCLIP requested MPS, but torch.backends.mps.is_available() is false"
        )
    return device


def torch_device_name(torch, device: str) -> str:  # noqa: ANN001 - torch module.
    if device == "cuda":
        return str(torch.cuda.get_device_name(0))
    if device == "mps":
        return "Apple MPS"
    return ""


def _mps_available(torch) -> bool:  # noqa: ANN001 - torch module.
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    return bool(mps is not None and mps.is_available())


if __name__ == "__main__":
    main()
