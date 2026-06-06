from __future__ import annotations

from pathlib import Path

from flickr_bio_occurrence.vision.image_cache import CachedImage


PROTECTED_CACHE_PARTS = {"huggingface", ".hf-cache", "model", "models"}


def cleanup_cached_image(
    cached: CachedImage,
    *,
    cache_root: str | Path,
    delete_after_success: bool,
) -> bool:
    if not delete_after_success:
        return False
    path = Path(cached.path)
    root = Path(cache_root)
    if not _is_safe_cache_path(path, root):
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def _is_safe_cache_path(path: Path, root: Path) -> bool:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if any(part.casefold() in PROTECTED_CACHE_PARTS for part in resolved_root.parts):
        return False
    if any(part.casefold() in PROTECTED_CACHE_PARTS for part in resolved_path.parts):
        return False
    return resolved_path.is_relative_to(resolved_root)
