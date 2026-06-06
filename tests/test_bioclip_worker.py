from __future__ import annotations

import io
import json
from pathlib import Path

from flickr_bio_occurrence.vision import bioclip_worker
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


def test_worker_main_accepts_batch_request_without_loading_model(monkeypatch, tmp_path, capsys) -> None:
    calls: dict[str, object] = {}

    def fake_score_images(*, image_paths, labels, model_name, checkpoint, require_cuda):  # noqa: ANN001 - mirrors worker signature.
        calls["image_paths"] = image_paths
        calls["labels"] = labels
        calls["model_name"] = model_name
        calls["checkpoint"] = checkpoint
        calls["require_cuda"] = require_cuda
        return [{"label-a": 0.7}, {"label-a": 0.8}]

    monkeypatch.setattr(bioclip_worker, "score_images", fake_score_images)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps(
                {
                    "image_paths": ["/tmp/1.jpg", "/tmp/2.jpg"],
                    "labels": ["label-a"],
                    "model_name": "ViT-H-14",
                    "checkpoint": "checkpoint",
                    "hf_cache_dir": str(tmp_path / "hf"),
                }
            )
        ),
    )

    bioclip_worker.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["scores_by_image"] == [{"label-a": 0.7}, {"label-a": 0.8}]
    assert [str(path) for path in calls["image_paths"]] == ["/tmp/1.jpg", "/tmp/2.jpg"]
    assert calls["require_cuda"] is True
