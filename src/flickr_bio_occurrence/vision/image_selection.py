from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ImageSelectionMode = Literal["large", "original_diagnostic"]
DEFAULT_ORIGINAL_MAX_PIXELS = 89_478_485


class OriginalImageTooLarge(ValueError):
    """Raised when an original diagnostic image exceeds the configured pixel guardrail."""


@dataclass(frozen=True)
class SelectedImageUrl:
    url: str | None
    kind: str | None


@dataclass(frozen=True)
class ImageSelectionPolicy:
    mode: ImageSelectionMode = "large"
    original_max_pixels: int = DEFAULT_ORIGINAL_MAX_PIXELS


def select_flickr_image_url(photo: dict[str, Any], policy: ImageSelectionPolicy | None = None) -> SelectedImageUrl:
    effective_policy = policy or ImageSelectionPolicy()
    if effective_policy.mode == "original_diagnostic":
        if photo.get("url_o"):
            _enforce_original_guardrail(photo, effective_policy)
            return SelectedImageUrl(str(photo["url_o"]), "url_o")
        return _select_first_available(photo, ("url_l", "url_m"))
    return _select_first_available(photo, ("url_l", "url_m"))


def _select_first_available(photo: dict[str, Any], keys: tuple[str, ...]) -> SelectedImageUrl:
    for key in keys:
        value = photo.get(key)
        if value:
            return SelectedImageUrl(str(value), key)
    return SelectedImageUrl(None, None)


def _enforce_original_guardrail(photo: dict[str, Any], policy: ImageSelectionPolicy) -> None:
    width = _optional_int(photo.get("o_width") or photo.get("width_o"))
    height = _optional_int(photo.get("o_height") or photo.get("height_o"))
    if width is None or height is None:
        return
    pixels = width * height
    if pixels > policy.original_max_pixels:
        raise OriginalImageTooLarge(
            f"Original Flickr image has {pixels} pixels, above guardrail {policy.original_max_pixels}"
        )


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
