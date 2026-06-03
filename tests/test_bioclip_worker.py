from __future__ import annotations

from flickr_bio_occurrence.vision.bioclip_worker import open_clip_model_args


def test_open_clip_model_args_use_hf_hub_prefix_for_model_ids() -> None:
    assert open_clip_model_args("imageomics/bioclip-2", "BioCLIP 2.5 Huge OpenCLIP ViT-H/14 checkpoint") == {
        "model_name": "hf-hub:imageomics/bioclip-2",
        "pretrained": None,
    }


def test_open_clip_model_args_preserve_explicit_openclip_checkpoint() -> None:
    assert open_clip_model_args("ViT-H-14", "laion2b_s32b_b79k") == {
        "model_name": "ViT-H-14",
        "pretrained": "laion2b_s32b_b79k",
    }
