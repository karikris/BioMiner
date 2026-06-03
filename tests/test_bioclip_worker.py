from __future__ import annotations

from pathlib import Path

from flickr_bio_occurrence.vision.bioclip_worker import configure_hf_cache_env, open_clip_model_args


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


def test_configure_hf_cache_env_sets_writable_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)

    configure_hf_cache_env(tmp_path / "hf")

    assert str(tmp_path / "hf") in __import__("os").environ["HF_HOME"]
    assert str(tmp_path / "hf" / "hub") in __import__("os").environ["HUGGINGFACE_HUB_CACHE"]
