"""Lazy official Sentence Transformers adapter. No fp16 inference is permitted."""

from __future__ import annotations

import platform
import re
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from biominer.common.semantic_hash import canonical_semantic_fingerprint

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
PREPROCESSING_VERSION = "embeddinggemma-full-image-exif-rgb-v1"


@dataclass(frozen=True)
class RuntimeConfig:
    revision: str = MODEL_REVISION
    device: str = "mps"
    dtype: str = "float32"
    model_id: str = MODEL_ID

    def __post_init__(self):
        if self.dtype != "float32":
            raise ValueError("This benchmark requires float32; float16 is forbidden")
        if self.device not in {"mps", "cpu"}:
            raise ValueError("Select mps or cpu explicitly")
        if not re.fullmatch(r"[a-f0-9]{40}", self.revision):
            raise ValueError("Model revision must be an immutable 40-character commit")

    def fingerprint(self, budget: int) -> str:
        if budget not in {280, 1120}:
            raise ValueError("Benchmark token budget must be 280 or 1120")
        from importlib.metadata import PackageNotFoundError, version

        versions = {}
        for name in ("torch", "sentence-transformers", "transformers", "pillow"):
            try:
                versions[name] = version(name)
            except PackageNotFoundError:
                versions[name] = "not_installed"
        return canonical_semantic_fingerprint(
            {
                **asdict(self),
                "budget": budget,
                "versions": versions,
                "transform": PREPROCESSING_VERSION,
                "image_prompt": "",
                "dimensions": 768,
                "pooling": "official-sentence-transformers",
            }
        )


def normalize_vectors(vectors, dimension: int = 768):
    import numpy as np

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < dimension or dimension not in {256, 768}:
        raise ValueError("Embedding shape or dimension is invalid")
    array = array[:, :dimension].copy()
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if (
        not np.isfinite(array).all()
        or not np.isfinite(norms).all()
        or (norms <= 0).any()
    ):
        raise ValueError(
            "Nonfinite or zero embedding; refusing degraded model evidence"
        )
    return array / norms


class SentenceEncoder:
    def __init__(
        self, config: RuntimeConfig | None = None, *, cache_dir: Path | None = None
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        config = config or RuntimeConfig()
        self.config = config
        self.cache_dir = cache_dir
        if config.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; choose --device cpu explicitly")
        if (
            config.device == "mps"
            and __import__("os").environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
        ):
            raise RuntimeError(
                "Disable PYTORCH_ENABLE_MPS_FALLBACK for comparable device timing"
            )
        self.model = SentenceTransformer(
            config.model_id,
            revision=config.revision,
            device=config.device,
            cache_folder=str(cache_dir) if cache_dir else None,
            config_kwargs={"audio_config": None},
            model_kwargs={"torch_dtype": torch.float32},
            trust_remote_code=False,
        )
        self.model.eval()
        # Do not trust a checkpoint dtype or mixed-precision default.
        self.model.float()
        if any(
            p.is_floating_point() and p.dtype != torch.float32
            for p in self.model.parameters()
        ):
            raise RuntimeError("Model contains non-float32 parameters")
        backbone = self.model[0].auto_model
        if getattr(backbone.config, "audio_config", None) is not None:
            raise RuntimeError("Audio tower was not disabled")
        self.loaded_at = time.monotonic()

    def encode_images(self, paths: list[Path], budget: int):
        from PIL import Image, ImageOps

        self.config.fingerprint(budget)
        images = []
        try:
            for path in paths:
                with Image.open(path) as image:
                    images.append(ImageOps.exif_transpose(image).convert("RGB"))
            values = self.model.encode(
                [{"image": image} for image in images],
                batch_size=len(images),
                prompt="",
                normalize_embeddings=False,
                convert_to_numpy=True,
                show_progress_bar=False,
                processing_kwargs={"image": {"max_soft_tokens": budget}},
            )
            return normalize_vectors(values)
        finally:
            for image in images:
                image.close()

    def encode_text(self, sentences: list[str]):
        return normalize_vectors(
            self.model.encode(
                sentences,
                batch_size=32,
                prompt_name="Classification",
                normalize_embeddings=False,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        )

    def metrics(self):
        import torch

        return {
            "device": self.config.device,
            "dtype": self.config.dtype,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "mps_current_allocated_bytes": torch.mps.current_allocated_memory()
            if self.config.device == "mps"
            else None,
            "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory()
            if self.config.device == "mps"
            else None,
            "model_loads": 1,
            "audio_encoder_loaded": False,
        }
