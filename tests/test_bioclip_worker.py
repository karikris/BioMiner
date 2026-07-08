from __future__ import annotations

import io
import json
from pathlib import Path

from biominer.bioclip import bioclip_worker
from biominer.bioclip.bioclip_worker import configure_hf_cache_env, device_from_request, open_clip_model_args, resolve_torch_device


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
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    configure_hf_cache_env(tmp_path / "hf")

    assert str(tmp_path / "hf") in __import__("os").environ["HF_HOME"]
    assert str(tmp_path / "hf" / "hub") in __import__("os").environ["HUGGINGFACE_HUB_CACHE"]
    assert __import__("os").environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_device_from_request_defaults_to_auto_and_preserves_legacy_cuda() -> None:
    assert device_from_request({}) == "auto"
    assert device_from_request({"require_cuda": True}) == "cuda"
    assert device_from_request({"require_cuda": True, "device": "mps"}) == "mps"


def test_resolve_torch_device_prefers_mps_when_cuda_is_unavailable() -> None:
    assert resolve_torch_device(FakeTorch(cuda=False, mps=True), "auto") == "mps"


def test_worker_main_accepts_batch_request_without_loading_model(monkeypatch, tmp_path, capsys) -> None:
    calls: dict[str, object] = {}

    def fake_score_images(*, image_paths, labels, model_name, checkpoint, device):  # noqa: ANN001 - mirrors worker signature.
        calls["image_paths"] = image_paths
        calls["labels"] = labels
        calls["model_name"] = model_name
        calls["checkpoint"] = checkpoint
        calls["device"] = device
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
    assert calls["device"] == "auto"


class FakeTorch:
    def __init__(self, *, cuda: bool, mps: bool) -> None:
        self.cuda = FakeCuda(cuda)
        self.backends = FakeBackends(mps)


class FakeCuda:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available


class FakeBackends:
    def __init__(self, mps: bool) -> None:
        self.mps = FakeMps(mps)


class FakeMps:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available
