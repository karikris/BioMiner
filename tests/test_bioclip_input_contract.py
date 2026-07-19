"""Tests for explicit crop versus full-frame BioCLIP input policies."""

from __future__ import annotations

import pytest

from biominer.bioclip.classification_modes import TARGET_AWARE_FEW_SHOT_CLASSIFICATION
from biominer.vision.bioclip_input_contract import (
    DYNAMIC_POOL_VISUAL_MODE,
    TARGET_AWARE_VISUAL_MODE,
    bioclip_visual_input_contract,
)
from biominer.vision.full_frame_attention import (
    FULL_FRAME_VISUAL_INPUT_VERSION,
    RAW_FULL_IMAGE_KIND,
)


@pytest.mark.parametrize(
    "mode",
    [
        TARGET_AWARE_VISUAL_MODE,
        TARGET_AWARE_FEW_SHOT_CLASSIFICATION,
        DYNAMIC_POOL_VISUAL_MODE,
    ],
)
def test_target_aware_and_dynamic_pool_contracts_are_full_frame(mode: str) -> None:
    contract = bioclip_visual_input_contract(mode)

    assert contract.input_family == "full_frame"
    assert contract.spatial_crop_permitted is False
    contract.validate_input(
        visual_input_kind=RAW_FULL_IMAGE_KIND,
        visual_input_version=FULL_FRAME_VISUAL_INPUT_VERSION,
        spatial_crop_applied=False,
    )
    with pytest.raises(ValueError, match="not allowed"):
        contract.validate_input(
            visual_input_kind="detector_crop",
            visual_input_version=FULL_FRAME_VISUAL_INPUT_VERSION,
            spatial_crop_applied=True,
        )


@pytest.mark.parametrize(
    "mode",
    ["legacy_object_screening", "hierarchical_butterfly_classification", ""],
)
def test_crop_and_legacy_modes_are_rejected(mode: str) -> None:
    with pytest.raises(ValueError, match="unsupported BioCLIP visual mode"):
        bioclip_visual_input_contract(mode)
