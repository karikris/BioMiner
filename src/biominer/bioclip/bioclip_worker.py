from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence


DEFAULT_DEVICE = "auto"
VALID_DEVICES = {"auto", "cuda", "mps", "cpu"}


def main() -> None:
    if "--persistent" in sys.argv[1:]:
        run_persistent_worker()
        return
    request = json.loads(sys.stdin.read())
    configure_hf_cache_env(Path(request.get("hf_cache_dir") or "data/cache/huggingface"))
    image_paths = request.get("image_paths")
    device = device_from_request(request)
    preprocess_workers = preprocess_workers_from_request(request)
    label_sets = request.get("label_sets")
    if label_sets is not None:
        scores_by_image_by_label_set = score_image_label_sets(
            image_paths=[Path(path) for path in image_paths],
            label_sets={str(name): list(labels) for name, labels in label_sets.items()},
            model_name=request["model_name"],
            checkpoint=request["checkpoint"],
            device=device,
            preprocess_workers=preprocess_workers,
        )
        print(json.dumps({"scores_by_image_by_label_set": scores_by_image_by_label_set}, sort_keys=True))
        return
    if image_paths is None:
        scores = score_image(
            image_path=Path(request["image_path"]),
            labels=request["labels"],
            model_name=request["model_name"],
            checkpoint=request["checkpoint"],
            device=device,
            preprocess_workers=preprocess_workers,
        )
        print(json.dumps({"scores": scores}, sort_keys=True))
        return
    scores_by_image = score_images(
        image_paths=[Path(path) for path in image_paths],
        labels=request["labels"],
        model_name=request["model_name"],
        checkpoint=request["checkpoint"],
        device=device,
        preprocess_workers=preprocess_workers,
    )
    print(json.dumps({"scores_by_image": scores_by_image}, sort_keys=True))


def configure_hf_cache_env(cache_dir: str | Path) -> Path:
    cache_path = Path(cache_dir).resolve()
    hub_path = cache_path / "hub"
    hub_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_path))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_path))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return cache_path


def device_from_request(request: dict[str, object]) -> str:
    if "device" in request and request["device"] not in (None, ""):
        return normalize_device(str(request["device"]))
    if request.get("require_cuda") is True:
        return "cuda"
    return DEFAULT_DEVICE


def preprocess_workers_from_request(request: dict[str, object]) -> int:
    value = int(request.get("preprocess_workers") or 1)
    if value <= 0:
        raise ValueError("preprocess_workers must be positive")
    return value


def normalize_device(device: str) -> str:
    normalized = device.casefold().strip()
    if normalized not in VALID_DEVICES:
        raise ValueError(f"Unsupported BioCLIP device {device!r}; expected one of {sorted(VALID_DEVICES)}")
    return normalized


def run_persistent_worker() -> None:
    loaded: _LoadedBioClipModel | None = None
    loaded_key: tuple[str, str, str] | None = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("shutdown"):
                return
            configure_hf_cache_env(Path(request.get("hf_cache_dir") or "data/cache/huggingface"))
            device = device_from_request(request)
            preprocess_workers = preprocess_workers_from_request(request)
            key = (str(request["model_name"]), str(request["checkpoint"]), device)
            if loaded is None or loaded_key != key:
                loaded = _LoadedBioClipModel.load(
                    model_name=key[0],
                    checkpoint=key[1],
                    device=key[2],
                )
                loaded_key = key
                print(json.dumps({"ready": True, "device": loaded.device, "gpu_name": loaded.gpu_name}, sort_keys=True), flush=True)
            text_labels = request.get("text_labels")
            if text_labels is not None:
                embeddings = loaded.text_embeddings(list(text_labels))
                print(
                    json.dumps(
                        {
                            "text_embeddings": embeddings,
                            "embedding_dim": len(embeddings[0]) if embeddings else 0,
                            "device": loaded.device,
                            "gpu_name": loaded.gpu_name,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            image_embedding_paths = request.get("image_embedding_paths")
            if image_embedding_paths is not None:
                embeddings = loaded.image_embeddings(
                    [Path(path) for path in image_embedding_paths],
                    preprocess_workers=preprocess_workers,
                )
                print(
                    json.dumps(
                        {
                            "image_embeddings": embeddings,
                            "embedding_dim": len(embeddings[0]) if embeddings else 0,
                            "device": loaded.device,
                            "gpu_name": loaded.gpu_name,
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
                    {str(name): list(labels) for name, labels in label_sets.items()},
                    preprocess_workers=preprocess_workers,
                )
                print(
                    json.dumps(
                        {
                            "scores_by_image_by_label_set": scores_by_image_by_label_set,
                            "device": loaded.device,
                            "gpu_name": loaded.gpu_name,
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
                print(json.dumps({"scores": scores, "device": loaded.device, "gpu_name": loaded.gpu_name}, sort_keys=True), flush=True)
                continue
            scores_by_image = loaded.score_images(
                [Path(path) for path in image_paths],
                request["labels"],
                preprocess_workers=preprocess_workers,
            )
            print(
                json.dumps(
                    {"scores_by_image": scores_by_image, "device": loaded.device, "gpu_name": loaded.gpu_name},
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - worker reports errors to controller process.
            print(json.dumps({"error": str(exc)}, sort_keys=True), flush=True)


def score_image(
    *,
    image_path: Path,
    labels: Sequence[str],
    model_name: str,
    checkpoint: str,
    device: str = DEFAULT_DEVICE,
    require_cuda: bool | None = None,
    preprocess_workers: int = 1,
) -> dict[str, float]:
    return score_images(
        image_paths=[image_path],
        labels=labels,
        model_name=model_name,
        checkpoint=checkpoint,
        device=_coerce_device(device=device, require_cuda=require_cuda),
        preprocess_workers=preprocess_workers,
    )[0]


def score_images(
    *,
    image_paths: Sequence[Path],
    labels: Sequence[str],
    model_name: str,
    checkpoint: str,
    device: str = DEFAULT_DEVICE,
    require_cuda: bool | None = None,
    preprocess_workers: int = 1,
) -> list[dict[str, float]]:
    model = _LoadedBioClipModel.load(
        model_name=model_name,
        checkpoint=checkpoint,
        device=_coerce_device(device=device, require_cuda=require_cuda),
    )
    return model.score_images(image_paths, labels, preprocess_workers=preprocess_workers)


def score_image_label_sets(
    *,
    image_paths: Sequence[Path],
    label_sets: dict[str, Sequence[str]],
    model_name: str,
    checkpoint: str,
    device: str = DEFAULT_DEVICE,
    require_cuda: bool | None = None,
    preprocess_workers: int = 1,
) -> dict[str, list[dict[str, float]]]:
    model = _LoadedBioClipModel.load(
        model_name=model_name,
        checkpoint=checkpoint,
        device=_coerce_device(device=device, require_cuda=require_cuda),
    )
    return model.score_image_label_sets(image_paths, label_sets, preprocess_workers=preprocess_workers)


def _coerce_device(*, device: str, require_cuda: bool | None) -> str:
    if require_cuda is True and device == DEFAULT_DEVICE:
        return "cuda"
    return normalize_device(device)


class _LoadedBioClipModel:
    def __init__(self, *, model, preprocess, tokenizer, torch, device: str, gpu_name: str) -> None:  # noqa: ANN001 - external runtime objects.
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.torch = torch
        self.device = device
        self.gpu_name = gpu_name
        self._text_features_by_labels: dict[tuple[str, ...], object] = {}

    @classmethod
    def load(cls, *, model_name: str, checkpoint: str, device: str = DEFAULT_DEVICE) -> "_LoadedBioClipModel":
        try:
            import open_clip
            import torch
        except Exception as exc:  # noqa: BLE001 - executed in the external model runtime.
            raise RuntimeError(f"BioCLIP dependencies are unavailable: {exc}") from exc

        resolved_device = resolve_torch_device(torch, device)
        gpu_name = torch_device_name(torch, resolved_device)
        model_args = open_clip_model_args(model_name, checkpoint)
        model, _, preprocess = open_clip.create_model_and_transforms(model_args["model_name"], pretrained=model_args["pretrained"])
        tokenizer = open_clip.get_tokenizer(model_args["model_name"])
        model = model.to(resolved_device)
        model.eval()
        return cls(model=model, preprocess=preprocess, tokenizer=tokenizer, torch=torch, device=resolved_device, gpu_name=gpu_name)

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
            image_batch = self._image_batch(image_paths, Image, preprocess_workers=preprocess_workers)
            image_features = self.model.encode_image(image_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            probabilities_by_image = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            for probabilities in probabilities_by_image:
                scores_by_image.append({label: float(probabilities[index].detach().cpu()) for index, label in enumerate(labels)})
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

        text_features_by_set = {name: self._text_features(labels) for name, labels in label_sets.items()}
        scores_by_label_set: dict[str, list[dict[str, float]]] = {name: [] for name in label_sets}
        with self.torch.no_grad():
            image_batch = self._image_batch(image_paths, Image, preprocess_workers=preprocess_workers)
            image_features = self.model.encode_image(image_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            for label_set_name, labels in label_sets.items():
                probabilities_by_image = (100.0 * image_features @ text_features_by_set[label_set_name].T).softmax(dim=-1)
                for probabilities in probabilities_by_image:
                    scores_by_label_set[label_set_name].append(
                        {label: float(probabilities[index].detach().cpu()) for index, label in enumerate(labels)}
                    )
        return scores_by_label_set

    def text_embeddings(self, labels: Sequence[str]) -> list[list[float]]:
        text_features = self._text_features(labels)
        return [
            [float(value) for value in row.detach().cpu().tolist()]
            for row in text_features
        ]

    def image_embeddings(self, image_paths: Sequence[Path], *, preprocess_workers: int = 1) -> list[list[float]]:
        try:
            from PIL import Image
        except Exception as exc:  # noqa: BLE001 - executed in the external model runtime.
            raise RuntimeError(f"BioCLIP dependencies are unavailable: {exc}") from exc

        with self.torch.no_grad():
            image_batch = self._image_batch(image_paths, Image, preprocess_workers=preprocess_workers)
            image_features = self.model.encode_image(image_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return [
            [float(value) for value in row.detach().cpu().tolist()]
            for row in image_features
        ]

    def _image_batch(self, image_paths: Sequence[Path], image_module, *, preprocess_workers: int = 1):  # noqa: ANN001 - PIL module.
        workers = int(preprocess_workers)
        if workers <= 0:
            raise ValueError("preprocess_workers must be positive")
        if workers == 1 or len(image_paths) <= 1:
            images = [self._preprocess_image_path(image_path, image_module) for image_path in image_paths]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                images = list(executor.map(lambda image_path: self._preprocess_image_path(image_path, image_module), image_paths))
        return self.torch.stack(images).to(self.device)

    def _preprocess_image_path(self, image_path: Path, image_module):  # noqa: ANN001 - PIL module.
        image = image_module.open(image_path)
        try:
            return self.preprocess(image.convert("RGB"))
        finally:
            close = getattr(image, "close", None)
            if callable(close):
                close()

    def _text_features(self, labels: Sequence[str]):
        label_key = tuple(labels)
        cached = self._text_features_by_labels.get(label_key)
        if cached is not None:
            return cached
        text = self.tokenizer(list(labels)).to(self.device)
        with self.torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self._text_features_by_labels[label_key] = text_features
        return text_features


def open_clip_model_args(model_name: str, checkpoint: str) -> dict[str, str | None]:
    if "/" in model_name and not model_name.startswith("hf-hub:"):
        return {"model_name": f"hf-hub:{model_name}", "pretrained": None}
    if model_name.startswith("hf-hub:"):
        return {"model_name": model_name, "pretrained": None}
    return {"model_name": model_name, "pretrained": checkpoint or None}


def resolve_torch_device(torch, requested_device: str = DEFAULT_DEVICE) -> str:  # noqa: ANN001 - torch module.
    device = normalize_device(requested_device)
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available(torch):
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("BioCLIP requested CUDA, but torch.cuda.is_available() is false")
    if device == "mps" and not _mps_available(torch):
        raise RuntimeError("BioCLIP requested MPS, but torch.backends.mps.is_available() is false")
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
