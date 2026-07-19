from __future__ import annotations

import biominer.vision as vision
from biominer.bioclip.object_runner import OBJECT_VISUAL_MODES


def test_target_full_frame_api_is_public_without_extending_legacy_modes() -> None:
    assert vision.TARGET_AWARE_VISUAL_MODE == "whole_image_reference_ensemble"
    assert vision.TARGET_FULL_FRAME_REQUIRES_CROP_METADATA is False
    assert vision.TARGET_FULL_FRAME_MATERIALIZES_CROP_FILES is False
    assert callable(vision.build_target_full_frame_plan)
    assert callable(vision.encode_target_full_frame_plan)
    assert callable(vision.encode_images_memory_aware)
    assert callable(vision.generate_full_frame_attention_variants)
    assert callable(vision.generate_target_full_frame_attention_variants)
    assert callable(vision.target_full_frame_detection_run_policy)
    assert vision.TARGET_AWARE_VISUAL_MODE not in OBJECT_VISUAL_MODES
    assert set(vision.__all__) >= {
        "FullFrameImageEncoder",
        "MemoryAwareImageBatchPolicy",
        "MpsMemorySnapshot",
        "TARGET_FULL_FRAME_EMBEDDING_VERSION",
        "TARGET_FULL_FRAME_SCORING_UNIT_VERSION",
        "TARGET_FULL_FRAME_VISUAL_INPUT_VERSION",
    }
