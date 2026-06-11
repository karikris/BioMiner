from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Sequence


def main() -> None:
    if "--persistent" in sys.argv[1:]:
        run_persistent_worker()
        return
    request = json.loads(sys.stdin.read())
    configure_hf_cache_env(Path(request.get("hf_cache_dir") or "data/cache/huggingface"))
    image_paths = request.get("image_paths")
    require_cuda = bool(request.get("require_cuda", True))
    label_sets = request.get("label_sets")
    if label_sets is not None:
        scores_by_image_by_label_set = score_image_label_sets(
            image_paths=[Path(path) for path in image_paths],
            label_sets={str(name): list(labels) for name, labels in label_sets.items()},
            model_name=request["model_name"],
            checkpoint=request["checkpoint"],
            require_cuda=require_cuda,
        )
        print(json.dumps({"scores_by_image_by_label_set": scores_by_image_by_label_set}, sort_keys=True))
        return
    if image_paths is None:
        scores = score_image(
            image_path=Path(request["image_path"]),
            labels=request["labels"],
            model_name=request["model_name"],
            checkpoint=request["checkpoint"],
            require_cuda=require_cuda,
        )
        print(json.dumps({"scores": scores}, sort_keys=True))
        return
    scores_by_image = score_images(
        image_paths=[Path(path) for path in image_paths],
        labels=request["labels"],
        model_name=request["model_name"],
        checkpoint=request["checkpoint"],
        require_cuda=require_cuda,
    )
    print(json.dumps({"scores_by_image": scores_by_image}, sort_keys=True))


def configure_hf_cache_env(cache_dir: str | Path) -> Path:
    cache_path = Path(cache_dir).resolve()
    hub_path = cache_path / "hub"
    hub_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_path))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_path))
    return cache_path


def run_persistent_worker() -> None:
    loaded: _LoadedBioClipModel | None = None
    loaded_key: tuple[str, str, bool] | None = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("shutdown"):
                return
            configure_hf_cache_env(Path(request.get("hf_cache_dir") or "data/cache/huggingface"))
            require_cuda = bool(request.get("require_cuda", True))
            key = (str(request["model_name"]), str(request["checkpoint"]), require_cuda)
            if loaded is None or loaded_key != key:
                loaded = _LoadedBioClipModel.load(
                    model_name=key[0],
                    checkpoint=key[1],
                    require_cuda=require_cuda,
                )
                loaded_key = key
                print(json.dumps({"ready": True, "device": loaded.device, "gpu_name": loaded.gpu_name}, sort_keys=True), flush=True)
            image_paths = request.get("image_paths")
            label_sets = request.get("label_sets")
            if label_sets is not None:
                scores_by_image_by_label_set = loaded.score_image_label_sets(
                    [Path(path) for path in image_paths],
                    {str(name): list(labels) for name, labels in label_sets.items()},
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
                scores = loaded.score_images([Path(request["image_path"])], request["labels"])[0]
                print(json.dumps({"scores": scores, "device": loaded.device, "gpu_name": loaded.gpu_name}, sort_keys=True), flush=True)
                continue
            scores_by_image = loaded.score_images([Path(path) for path in image_paths], request["labels"])
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
    require_cuda: bool = True,
) -> dict[str, float]:
    return score_images(
        image_paths=[image_path],
        labels=labels,
        model_name=model_name,
        checkpoint=checkpoint,
        require_cuda=require_cuda,
    )[0]


def score_images(
    *,
    image_paths: Sequence[Path],
    labels: Sequence[str],
    model_name: str,
    checkpoint: str,
    require_cuda: bool = True,
) -> list[dict[str, float]]:
    model = _LoadedBioClipModel.load(model_name=model_name, checkpoint=checkpoint, require_cuda=require_cuda)
    return model.score_images(image_paths, labels)


def score_image_label_sets(
    *,
    image_paths: Sequence[Path],
    label_sets: dict[str, Sequence[str]],
    model_name: str,
    checkpoint: str,
    require_cuda: bool = True,
) -> dict[str, list[dict[str, float]]]:
    model = _LoadedBioClipModel.load(model_name=model_name, checkpoint=checkpoint, require_cuda=require_cuda)
    return model.score_image_label_sets(image_paths, label_sets)


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
    def load(cls, *, model_name: str, checkpoint: str, require_cuda: bool = True) -> "_LoadedBioClipModel":
        try:
            import open_clip
            import torch
        except Exception as exc:  # noqa: BLE001 - executed in the external model runtime.
            raise RuntimeError(f"BioCLIP dependencies are unavailable: {exc}") from exc

        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("BioCLIP requires CUDA, but torch.cuda.is_available() is false")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else ""
        model_args = open_clip_model_args(model_name, checkpoint)
        model, _, preprocess = open_clip.create_model_and_transforms(model_args["model_name"], pretrained=model_args["pretrained"])
        tokenizer = open_clip.get_tokenizer(model_args["model_name"])
        model = model.to(device)
        model.eval()
        return cls(model=model, preprocess=preprocess, tokenizer=tokenizer, torch=torch, device=device, gpu_name=gpu_name)

    def score_images(self, image_paths: Sequence[Path], labels: Sequence[str]) -> list[dict[str, float]]:
        try:
            from PIL import Image
        except Exception as exc:  # noqa: BLE001 - executed in the external model runtime.
            raise RuntimeError(f"BioCLIP dependencies are unavailable: {exc}") from exc

        text_features = self._text_features(labels)
        with self.torch.no_grad():
            scores_by_image: list[dict[str, float]] = []
            image_batch = self._image_batch(image_paths, Image)
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
    ) -> dict[str, list[dict[str, float]]]:
        try:
            from PIL import Image
        except Exception as exc:  # noqa: BLE001 - executed in the external model runtime.
            raise RuntimeError(f"BioCLIP dependencies are unavailable: {exc}") from exc

        text_features_by_set = {name: self._text_features(labels) for name, labels in label_sets.items()}
        scores_by_label_set: dict[str, list[dict[str, float]]] = {name: [] for name in label_sets}
        with self.torch.no_grad():
            image_batch = self._image_batch(image_paths, Image)
            image_features = self.model.encode_image(image_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            for label_set_name, labels in label_sets.items():
                probabilities_by_image = (100.0 * image_features @ text_features_by_set[label_set_name].T).softmax(dim=-1)
                for probabilities in probabilities_by_image:
                    scores_by_label_set[label_set_name].append(
                        {label: float(probabilities[index].detach().cpu()) for index, label in enumerate(labels)}
                    )
        return scores_by_label_set

    def _image_batch(self, image_paths: Sequence[Path], image_module):  # noqa: ANN001 - PIL module.
        images = [
            self.preprocess(image_module.open(image_path).convert("RGB"))
            for image_path in image_paths
        ]
        return self.torch.stack(images).to(self.device)

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


if __name__ == "__main__":
    main()
