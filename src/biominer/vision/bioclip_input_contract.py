"""Explicit visual-input policy for every BioCLIP scoring mode."""

from __future__ import annotations

from dataclasses import dataclass

from biominer.bioclip.classification_modes import (
    HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    TARGET_AWARE_FEW_SHOT_CLASSIFICATION,
    TARGET_SCOPE_OBJECT_SCREENING,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.vision.full_frame_attention import (
    FOCUSED_FULL_FRAME_KIND,
    FULL_FRAME_VISUAL_INPUT_VERSION,
    MASKED_FULL_FRAME_KIND,
    MULTI_OBJECT_FULL_FRAME_KIND,
    RAW_FULL_IMAGE_KIND,
)


BIOCLIP_VISUAL_INPUT_CONTRACT_VERSION = "bioclip-visual-input-contract-v1"
LEGACY_OBJECT_VISUAL_MODE = "legacy_object_screening"
TARGET_AWARE_VISUAL_MODE = "whole_image_reference_ensemble"
DYNAMIC_POOL_VISUAL_MODE = "geography_conditioned_dynamic_pool"

FULL_FRAME_BIOCLIP_VISUAL_INPUT_KINDS = (
    RAW_FULL_IMAGE_KIND,
    FOCUSED_FULL_FRAME_KIND,
    MASKED_FULL_FRAME_KIND,
    MULTI_OBJECT_FULL_FRAME_KIND,
)
LEGACY_BIOCLIP_VISUAL_INPUT_KINDS = (
    "detector_crop",
    "detector_crop_segmentation",
    "whole_image",
)

_TARGET_AWARE_MODES = frozenset(
    {
        TARGET_AWARE_VISUAL_MODE,
        TARGET_AWARE_FEW_SHOT_CLASSIFICATION,
    }
)
_LEGACY_MODES = frozenset(
    {
        LEGACY_OBJECT_VISUAL_MODE,
        TARGET_SCOPE_OBJECT_SCREENING,
        HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    }
)


@dataclass(frozen=True, slots=True)
class BioClipVisualInputContract:
    """Immutable mode-level policy separating crops from full-canvas inputs."""

    contract_version: str
    visual_mode: str
    input_family: str
    allowed_visual_input_kinds: tuple[str, ...]
    required_visual_input_version: str | None
    spatial_crop_permitted: bool

    def __post_init__(self) -> None:
        if self.contract_version != BIOCLIP_VISUAL_INPUT_CONTRACT_VERSION:
            raise ValueError("unsupported BioCLIP visual-input contract version")
        if not self.visual_mode:
            raise ValueError("BioCLIP visual mode must be explicit")
        if self.input_family not in {"legacy_object", "full_frame"}:
            raise ValueError("unsupported BioCLIP visual-input family")
        kinds = tuple(dict.fromkeys(self.allowed_visual_input_kinds))
        if not kinds or any(not value for value in kinds):
            raise ValueError("BioCLIP visual-input kinds must be non-empty")
        object.__setattr__(self, "allowed_visual_input_kinds", kinds)
        if self.input_family == "full_frame":
            if self.spatial_crop_permitted:
                raise ValueError("full-frame BioCLIP mode cannot permit spatial crops")
            if self.required_visual_input_version != FULL_FRAME_VISUAL_INPUT_VERSION:
                raise ValueError(
                    "full-frame BioCLIP mode requires the canonical version"
                )

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(self.identity_payload())

    def identity_payload(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "visual_mode": self.visual_mode,
            "input_family": self.input_family,
            "allowed_visual_input_kinds": list(self.allowed_visual_input_kinds),
            "required_visual_input_version": self.required_visual_input_version,
            "spatial_crop_permitted": self.spatial_crop_permitted,
        }

    def validate_input(
        self,
        *,
        visual_input_kind: str,
        visual_input_version: str | None,
        spatial_crop_applied: bool,
    ) -> None:
        if not isinstance(spatial_crop_applied, bool):
            raise TypeError("spatial_crop_applied must be boolean")
        if visual_input_kind not in self.allowed_visual_input_kinds:
            raise ValueError(
                f"BioCLIP visual-input kind {visual_input_kind!r} is not allowed "
                f"for mode {self.visual_mode!r}"
            )
        if spatial_crop_applied and not self.spatial_crop_permitted:
            raise ValueError(f"BioCLIP mode {self.visual_mode!r} forbids spatial crops")
        if visual_input_version != self.required_visual_input_version:
            raise ValueError(
                f"BioCLIP mode {self.visual_mode!r} requires visual-input version "
                f"{self.required_visual_input_version!r}"
            )


def bioclip_visual_input_contract(visual_mode: str) -> BioClipVisualInputContract:
    """Return the only visual-input policy accepted for ``visual_mode``."""

    mode = str(visual_mode or "").strip()
    if mode in _TARGET_AWARE_MODES or mode == DYNAMIC_POOL_VISUAL_MODE:
        return BioClipVisualInputContract(
            contract_version=BIOCLIP_VISUAL_INPUT_CONTRACT_VERSION,
            visual_mode=mode,
            input_family="full_frame",
            allowed_visual_input_kinds=FULL_FRAME_BIOCLIP_VISUAL_INPUT_KINDS,
            required_visual_input_version=FULL_FRAME_VISUAL_INPUT_VERSION,
            spatial_crop_permitted=False,
        )
    if mode in _LEGACY_MODES:
        return BioClipVisualInputContract(
            contract_version=BIOCLIP_VISUAL_INPUT_CONTRACT_VERSION,
            visual_mode=mode,
            input_family="legacy_object",
            allowed_visual_input_kinds=LEGACY_BIOCLIP_VISUAL_INPUT_KINDS,
            required_visual_input_version=None,
            spatial_crop_permitted=True,
        )
    raise ValueError(f"unsupported BioCLIP visual mode {visual_mode!r}")


__all__ = [
    "BIOCLIP_VISUAL_INPUT_CONTRACT_VERSION",
    "DYNAMIC_POOL_VISUAL_MODE",
    "FULL_FRAME_BIOCLIP_VISUAL_INPUT_KINDS",
    "LEGACY_BIOCLIP_VISUAL_INPUT_KINDS",
    "LEGACY_OBJECT_VISUAL_MODE",
    "TARGET_AWARE_VISUAL_MODE",
    "BioClipVisualInputContract",
    "bioclip_visual_input_contract",
]
