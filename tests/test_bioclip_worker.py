from __future__ import annotations

import io
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

from biominer.bioclip import bioclip_worker
from biominer.bioclip.bioclip_worker import (
    OPENCLIP_PREPROCESSING_ATTESTATION_VERSION,
    canonical_preprocessing_config,
    configure_hf_cache_env,
    device_from_request,
    mps_memory_snapshot,
    normalize_image_resize_mode,
    preprocessing_attestation_fingerprint,
    resolve_open_clip_model_source,
    resolve_torch_device,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint


REVISION = "191d741545e4c741cdef4b22c6eb69c945c1e592"


def test_mps_memory_snapshot_reports_allocator_driver_and_recommended_limit() -> None:
    fake_torch = types.SimpleNamespace(
        mps=types.SimpleNamespace(
            current_allocated_memory=lambda: 1024,
            driver_allocated_memory=lambda: 2048,
            recommended_max_memory=lambda: 4096,
        )
    )

    assert mps_memory_snapshot(fake_torch, "mps") == {
        "mps_current_allocated_memory": 1024,
        "mps_driver_allocated_memory": 2048,
        "mps_recommended_max_memory": 4096,
    }


def test_mps_memory_snapshot_marks_non_mps_devices_not_applicable() -> None:
    assert mps_memory_snapshot(types.SimpleNamespace(), "cpu") == {
        "mps_current_allocated_memory": "not_applicable",
        "mps_driver_allocated_memory": "not_applicable",
        "mps_recommended_max_memory": "not_applicable",
    }


def test_open_clip_model_source_resolves_exact_prefetched_hf_snapshot(
    monkeypatch,
    tmp_path,
) -> None:
    snapshot = _snapshot(tmp_path)
    calls: list[dict[str, object]] = []
    fake_hub = types.ModuleType("huggingface_hub")

    def snapshot_download(**kwargs):  # noqa: ANN003, ANN202 - fake Hub API.
        calls.append(kwargs)
        return str(snapshot)

    fake_hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    source = resolve_open_clip_model_source(
        "imageomics/bioclip-2.5-vith14",
        REVISION,
    )

    assert source.loader_model_name == f"local-dir:{snapshot.resolve()}"
    assert source.model_id == "imageomics/bioclip-2.5-vith14"
    assert source.model_revision == REVISION
    assert source.model_weights_sha256 == _file_sha256(
        snapshot / "open_clip_model.safetensors"
    )
    assert source.open_clip_config_sha256 == _file_sha256(
        snapshot / "open_clip_config.json"
    )
    assert source.pretrained is None
    assert source.require_pretrained is True
    assert calls == [
        {
            "repo_id": "imageomics/bioclip-2.5-vith14",
            "repo_type": "model",
            "revision": REVISION,
            "local_files_only": True,
        }
    ]


def test_open_clip_model_source_checks_resolved_snapshot_commit(
    monkeypatch,
    tmp_path,
) -> None:
    snapshot = _snapshot(tmp_path)
    snapshot_alias = tmp_path / "snapshot-alias"
    snapshot_alias.symlink_to(snapshot, target_is_directory=True)
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = lambda **_kwargs: str(snapshot_alias)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    source = resolve_open_clip_model_source(
        "imageomics/bioclip-2.5-vith14",
        REVISION,
    )

    assert source.loader_model_name == f"local-dir:{snapshot.resolve()}"


@pytest.mark.parametrize("revision", ["main", "v2.5", "", "a" * 39, "g" * 40])
def test_open_clip_model_source_rejects_mutable_or_invalid_hf_revision(
    revision: str,
) -> None:
    with pytest.raises(ValueError, match="immutable 40-character commit"):
        resolve_open_clip_model_source(
            "imageomics/bioclip-2.5-vith14",
            revision,
        )


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("open_clip_config.json", "config"),
        ("open_clip_model.safetensors", "weights"),
    ],
)
def test_open_clip_model_source_rejects_incomplete_prefetched_snapshot(
    monkeypatch,
    tmp_path,
    missing: str,
    message: str,
) -> None:
    snapshot = _snapshot(tmp_path)
    (snapshot / missing).unlink()
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = lambda **_kwargs: str(snapshot)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    with pytest.raises(FileNotFoundError, match=message):
        resolve_open_clip_model_source(
            "imageomics/bioclip-2.5-vith14",
            REVISION,
        )


def test_open_clip_model_source_preserves_explicit_openclip_checkpoint() -> None:
    source = resolve_open_clip_model_source("ViT-H-14", "laion2b_s32b_b79k")

    assert source.loader_model_name == "ViT-H-14"
    assert source.pretrained == "laion2b_s32b_b79k"
    assert source.model_id == "ViT-H-14"
    assert source.model_revision == "laion2b_s32b_b79k"
    assert source.model_weights_sha256 is None
    assert source.open_clip_config_sha256 is None
    assert source.require_pretrained is True


def test_preprocessing_attestation_fingerprint_uses_canonical_binary() -> None:
    config = {
        "size": (224, 224),
        "resize_mode": "longest",
        "mean": (0.1, 0.2, 0.3),
    }
    canonical = canonical_preprocessing_config(config)
    payload = {
        "open_clip_config_sha256": "sha256:" + "b" * 64,
        "open_clip_version": "3.3.0",
        "preprocessing_config": canonical,
        "preprocessing_version": OPENCLIP_PREPROCESSING_ATTESTATION_VERSION,
    }
    expected = canonical_semantic_fingerprint(payload)

    assert canonical == {
        "mean": [0.1, 0.2, 0.3],
        "resize_mode": "longest",
        "size": [224, 224],
    }
    assert OPENCLIP_PREPROCESSING_ATTESTATION_VERSION == (
        "openclip-preprocessing-attestation-v2"
    )
    assert (
        preprocessing_attestation_fingerprint(
            open_clip_config_sha256="sha256:" + "b" * 64,
            open_clip_version="3.3.0",
            preprocessing_config=config,
            preprocessing_version=OPENCLIP_PREPROCESSING_ATTESTATION_VERSION,
        )
        == expected
    )


def test_configure_hf_cache_env_sets_writable_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HF_HOME", "/stale/home")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", "/stale/hub")
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    configure_hf_cache_env(tmp_path / "hf")

    assert str(tmp_path / "hf") in __import__("os").environ["HF_HOME"]
    assert (
        str(tmp_path / "hf" / "hub")
        in __import__("os").environ["HUGGINGFACE_HUB_CACHE"]
    )
    assert __import__("os").environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_device_from_request_defaults_to_auto_and_preserves_legacy_cuda() -> None:
    assert device_from_request({}) == "auto"
    assert device_from_request({"require_cuda": True}) == "cuda"
    assert device_from_request({"require_cuda": True, "device": "mps"}) == "mps"


def test_image_resize_mode_validation_normalizes_supported_values() -> None:
    assert normalize_image_resize_mode(None) is None
    assert normalize_image_resize_mode(" Longest ") == "longest"
    assert normalize_image_resize_mode("shortest") == "shortest"
    assert normalize_image_resize_mode("SQUASH") == "squash"

    with pytest.raises(ValueError, match="Unsupported BioCLIP image resize mode"):
        normalize_image_resize_mode("center_crop")


def test_resolve_torch_device_prefers_mps_when_cuda_is_unavailable() -> None:
    assert resolve_torch_device(FakeTorch(cuda=False, mps=True), "auto") == "mps"


def test_worker_main_accepts_batch_request_without_loading_model(
    monkeypatch, tmp_path, capsys
) -> None:
    calls: dict[str, object] = {}

    def fake_score_images(  # noqa: ANN001 - mirrors worker signature.
        *,
        image_paths,
        labels,
        model_name,
        checkpoint,
        device,
        image_resize_mode,
        preprocess_workers,
    ):
        calls["image_paths"] = image_paths
        calls["labels"] = labels
        calls["model_name"] = model_name
        calls["checkpoint"] = checkpoint
        calls["device"] = device
        calls["image_resize_mode"] = image_resize_mode
        calls["preprocess_workers"] = preprocess_workers
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
                    "image_resize_mode": "longest",
                }
            )
        ),
    )

    bioclip_worker.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["scores_by_image"] == [{"label-a": 0.7}, {"label-a": 0.8}]
    assert [str(path) for path in calls["image_paths"]] == ["/tmp/1.jpg", "/tmp/2.jpg"]
    assert calls["device"] == "auto"
    assert calls["image_resize_mode"] == "longest"
    assert calls["preprocess_workers"] == 1
    assert payload["image_resize_mode"] == "longest"


def test_loaded_model_only_overrides_open_clip_resize_mode_when_requested(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    tokenizer_calls: list[str] = []
    snapshot = _snapshot(tmp_path)
    fake_open_clip = types.ModuleType("open_clip")
    fake_hub = types.ModuleType("huggingface_hub")

    def create_model_and_transforms(model_name: str, **kwargs):  # noqa: ANN001, ANN202 - fake OpenCLIP API.
        calls.append((model_name, kwargs))
        return (
            FakeOpenClipModel(str(kwargs.get("image_resize_mode") or "shortest")),
            None,
            object(),
        )

    fake_open_clip.__version__ = "3.3.0"
    fake_open_clip.create_model_and_transforms = create_model_and_transforms
    fake_open_clip.get_model_preprocess_cfg = lambda model: model.preprocessing_config
    fake_open_clip.get_tokenizer = lambda model_name: (
        tokenizer_calls.append(model_name) or object()
    )
    fake_hub.snapshot_download = lambda **_kwargs: str(snapshot)
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "torch", FakeTorch(cuda=False, mps=False))

    legacy = bioclip_worker._LoadedBioClipModel.load(  # noqa: SLF001 - worker integration contract.
        model_name="imageomics/bioclip-2",
        checkpoint=REVISION,
        device="cpu",
    )
    target = bioclip_worker._LoadedBioClipModel.load(  # noqa: SLF001 - target preprocessing contract.
        model_name="imageomics/bioclip-2",
        checkpoint=REVISION,
        device="cpu",
        image_resize_mode="longest",
    )

    assert calls == [
        (
            f"local-dir:{snapshot.resolve()}",
            {"pretrained": None, "require_pretrained": True},
        ),
        (
            f"local-dir:{snapshot.resolve()}",
            {
                "pretrained": None,
                "require_pretrained": True,
                "image_resize_mode": "longest",
            },
        ),
    ]
    assert tokenizer_calls == [
        f"local-dir:{snapshot.resolve()}",
        f"local-dir:{snapshot.resolve()}",
    ]
    assert legacy.image_resize_mode == "shortest"
    assert target.image_resize_mode == "longest"
    assert target.model_id == "imageomics/bioclip-2"
    assert target.model_revision == REVISION
    assert target.model_weights_sha256 == _file_sha256(
        snapshot / "open_clip_model.safetensors"
    )
    assert target.open_clip_version == "3.3.0"
    assert target.open_clip_config_sha256 == _file_sha256(
        snapshot / "open_clip_config.json"
    )
    assert target.preprocessing_version == (OPENCLIP_PREPROCESSING_ATTESTATION_VERSION)
    assert target.preprocessing_config == _preprocessing_config("longest")
    assert target.preprocessing_fingerprint == preprocessing_attestation_fingerprint(
        open_clip_config_sha256=target.open_clip_config_sha256,
        open_clip_version="3.3.0",
        preprocessing_config=_preprocessing_config("longest"),
        preprocessing_version=OPENCLIP_PREPROCESSING_ATTESTATION_VERSION,
    )


def test_loaded_model_rejects_invalid_resize_mode_before_importing_runtime() -> None:
    with pytest.raises(ValueError, match="Unsupported BioCLIP image resize mode"):
        bioclip_worker._LoadedBioClipModel.load(  # noqa: SLF001 - validation contract.
            model_name="imageomics/bioclip-2",
            checkpoint="checkpoint",
            image_resize_mode="crop",
        )


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


class FakeOpenClipModel:
    def __init__(self, resize_mode: str = "shortest") -> None:
        self.preprocessing_config = _preprocessing_config(resize_mode)

    def to(self, _device):  # noqa: ANN001, ANN202 - fake OpenCLIP model.
        return self

    def eval(self) -> None:
        return None


def _snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / "hub" / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "open_clip_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "open_clip_model.safetensors").write_bytes(b"frozen-weights")
    return snapshot


def _preprocessing_config(resize_mode: str) -> dict[str, object]:
    return {
        "fill_color": 0,
        "interpolation": "bicubic",
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "mode": "RGB",
        "resize_mode": resize_mode,
        "size": [224, 224],
        "std": [0.26862954, 0.26130258, 0.27577711],
    }


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
