from __future__ import annotations

from pathlib import Path
from typing import Any

from biominer.bioclip.image_cache import cache_image_from_url
from biominer.detection.detector_base import DecodedImage


def load_decoded_image_from_record(record: dict[str, Any], *, cache_root: str | Path = "data/cache/images") -> DecodedImage:
    image_url = str(record.get("image_url") or record.get("image_url_used") or "")
    if not image_url:
        raise ValueError("record is missing image_url")
    cached = cache_image_from_url(image_url, cache_root=cache_root)
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency path.
        raise RuntimeError("Pillow is required to decode images for object BioCLIP crop scoring in this runtime") from exc
    with Image.open(cached.path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        return DecodedImage(width=int(width), height=int(height), mode="RGB", data=rgb.tobytes(), source_uri=str(cached.path))
