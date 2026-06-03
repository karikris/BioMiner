from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Sequence


def main() -> None:
    request = json.loads(sys.stdin.read())
    configure_hf_cache_env(Path(request.get("hf_cache_dir") or "data/cache/huggingface"))
    scores = score_image(
        image_path=Path(request["image_path"]),
        labels=request["labels"],
        model_name=request["model_name"],
        checkpoint=request["checkpoint"],
    )
    print(json.dumps({"scores": scores}, sort_keys=True))


def configure_hf_cache_env(cache_dir: str | Path) -> Path:
    cache_path = Path(cache_dir).resolve()
    hub_path = cache_path / "hub"
    hub_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_path))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_path))
    return cache_path


def score_image(*, image_path: Path, labels: Sequence[str], model_name: str, checkpoint: str) -> dict[str, float]:
    try:
        from PIL import Image
        import open_clip
        import torch
    except Exception as exc:  # noqa: BLE001 - executed in the external model runtime.
        raise RuntimeError(f"BioCLIP dependencies are unavailable: {exc}") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_args = open_clip_model_args(model_name, checkpoint)
    model, _, preprocess = open_clip.create_model_and_transforms(model_args["model_name"], pretrained=model_args["pretrained"])
    tokenizer = open_clip.get_tokenizer(model_args["model_name"])
    model = model.to(device)
    model.eval()
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text = tokenizer(list(labels)).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        probabilities = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
    return {label: float(probabilities[index].detach().cpu()) for index, label in enumerate(labels)}


def open_clip_model_args(model_name: str, checkpoint: str) -> dict[str, str | None]:
    if "/" in model_name and not model_name.startswith("hf-hub:"):
        return {"model_name": f"hf-hub:{model_name}", "pretrained": None}
    if model_name.startswith("hf-hub:"):
        return {"model_name": model_name, "pretrained": None}
    return {"model_name": model_name, "pretrained": checkpoint or None}


if __name__ == "__main__":
    main()
