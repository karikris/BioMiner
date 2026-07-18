"""Tests for explicit crop versus full-frame BioCLIP input policies."""

from __future__ import annotations

import polars as pl
import pytest

from biominer.bioclip.classification_modes import (
    BUILD_WEEK_TARGET_AWARE_PROTOTYPE,
    TARGET_AWARE_FEW_SHOT_CLASSIFICATION,
)
from biominer.vision.bioclip_input_contract import (
    DYNAMIC_POOL_VISUAL_MODE,
    LEGACY_OBJECT_VISUAL_MODE,
    TARGET_AWARE_VISUAL_MODE,
    bioclip_visual_input_contract,
)
from biominer.vision.full_frame_attention import (
    FULL_FRAME_VISUAL_INPUT_VERSION,
    RAW_FULL_IMAGE_KIND,
)
from biominer.vision.score_inputs import materialize_bioclip_score_inputs


@pytest.mark.parametrize(
    "mode",
    [
        TARGET_AWARE_VISUAL_MODE,
        TARGET_AWARE_FEW_SHOT_CLASSIFICATION,
        BUILD_WEEK_TARGET_AWARE_PROTOTYPE,
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


def test_legacy_object_contract_keeps_crop_ablation_explicit() -> None:
    contract = bioclip_visual_input_contract(LEGACY_OBJECT_VISUAL_MODE)

    assert contract.input_family == "legacy_object"
    assert contract.spatial_crop_permitted is True
    contract.validate_input(
        visual_input_kind="detector_crop",
        visual_input_version=None,
        spatial_crop_applied=True,
    )


@pytest.mark.parametrize("mode", [TARGET_AWARE_VISUAL_MODE, DYNAMIC_POOL_VISUAL_MODE])
def test_legacy_materializer_blocks_full_frame_modes(tmp_path, mode: str) -> None:
    with pytest.raises(ValueError, match="canonical full-frame planner"):
        materialize_bioclip_score_inputs(
            canonical_records=pl.DataFrame(),
            detections=pl.DataFrame(),
            image_loader=lambda _row: None,
            temp_dir=tmp_path,
            visual_mode=mode,
        )
