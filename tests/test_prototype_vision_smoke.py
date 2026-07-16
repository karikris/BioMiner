from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
import polars as pl

import biominer.benchmarks.prototype_vision_smoke as smoke_module
from biominer.benchmarks.prototype_vision_smoke import (
    EXPECTED_IMAGE_COUNT,
    PrototypeVisionSmokeConfig,
    run_prototype_vision_smoke,
)
from biominer.bioclip.bioclip_worker import decoded_rgb_image_content_hash
from biominer.bioclip.bioclip import _persistent_worker_env


def test_papilio_prototype_smoke_config_is_exactly_five_local_mps_images() -> None:
    config = PrototypeVisionSmokeConfig.read_json(
        "config/pilot/papilio_demoleus_vision_smoke.json"
    )

    assert len(config.reference_media_ids) == EXPECTED_IMAGE_COUNT == 5
    assert len(set(config.reference_media_ids)) == 5
    assert config.device == "mps"
    assert config.bioclip_batch_size == 5
    assert config.yoloe_batch_size == 3
    assert config.model_name == "imageomics/bioclip-2.5-vith14"
    assert config.model_revision == "191d741545e4c741cdef4b22c6eb69c945c1e592"
    assert config.yoloe_checkpoint == "yoloe-26s-seg.pt"


def test_persistent_bioclip_worker_env_exposes_package_and_mps_fallback(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    env = _persistent_worker_env(tmp_path / "hf")

    assert env["PYTHONPATH"].endswith("/src")
    assert env["HF_HOME"] == str((tmp_path / "hf").resolve())
    assert env["HUGGINGFACE_HUB_CACHE"] == str((tmp_path / "hf" / "hub").resolve())
    assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_prototype_smoke_validates_hashes_embeddings_routes_and_worker_reuse(
    monkeypatch, tmp_path
) -> None:
    rows = []
    media_ids = []
    for index in range(5):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (32 + index, 24 + index), (index * 20, 80, 160)).save(path)
        media_id = f"reference-media:{index:064x}"
        media_ids.append(media_id)
        rows.append(
            {
                "reference_media_id": media_id,
                "dataset_split": "support_train",
                "source_object_uri": str(path),
                "source_image_sha256": _file_sha256(path),
                "accepted_taxon_key": f"gbif:{index}",
                "scientific_name": f"Fixture species {index}",
                "reference_group": f"fixture:{index}",
            }
        )
    support = tmp_path / "support.parquet"
    pl.DataFrame(rows).write_parquet(support)
    readiness = tmp_path / "readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "bank_status": "prototype_only",
                "classification_authorised": True,
                "prototype_readiness_status": "prototype_ready_with_shortfalls",
            }
        ),
        encoding="utf-8",
    )
    runtime_python = tmp_path / "python"
    runtime_python.touch()
    config = PrototypeVisionSmokeConfig(
        support_manifest=support,
        support_manifest_sha256=_file_sha256(support),
        readiness=readiness,
        readiness_sha256=_file_sha256(readiness),
        reference_media_ids=tuple(media_ids),
        output_dir=tmp_path / "output",
        bioclip_runtime_python=runtime_python,
        bioclip_hf_cache_dir=tmp_path / "hf",
        yoloe_runtime_python=runtime_python,
        model_name="imageomics/bioclip-2.5-vith14",
        model_revision="1" * 40,
        open_clip_version="3.3.0",
    )

    class FakeScorer:
        def __init__(self, **_kwargs) -> None:
            self.model_id = "imageomics/bioclip-2.5-vith14"
            self.model_revision = "1" * 40
            self.model_weights_sha256 = "sha256:" + "a" * 64
            self.open_clip_version = "3.3.0"
            self.open_clip_config_sha256 = "sha256:" + "b" * 64
            self.preprocessing_version = "openclip-preprocessing-attestation-v2"
            self.preprocessing_config = {"size": [224, 224]}
            self.preprocessing_fingerprint = "sha256:" + "c" * 64
            self.effective_image_resize_mode = "longest"
            self.device = "mps"
            self.gpu_name = "Apple MPS"
            self.last_image_content_hashes = None
            self.cache_metrics = {
                "bioclip_worker_process_starts": 1,
                "bioclip_worker_requests": 2,
                "bioclip_model_loads": 1,
                "bioclip_model_cache_hits": 1,
                "bioclip_model_refreshes": 0,
                "bioclip_model_cache_hit_rate": 0.5,
                "bioclip_last_request_cache_hit": True,
            }

        def ensure_model_attestation(self) -> None:
            return None

        def embed_image_paths(self, paths):  # noqa: ANN001
            hashes = []
            for path in paths:
                with Image.open(path) as image:
                    hashes.append(decoded_rgb_image_content_hash(image.convert("RGB")))
            self.last_image_content_hashes = hashes
            return [[float(index), 1.0] for index in range(len(paths))]

        def close(self) -> None:
            return None

    class FakeDetector:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.checkpoint = kwargs["checkpoint"]
            self.model_id = "yoloe26:yoloe-26s-seg"
            self.model_version = "ultralytics:test"
            self.prompt_set_fingerprint = "sha256:" + "d" * 64
            self.prompt_classes = ("butterfly",)
            self.worker_process_starts = 0
            self.worker_request_count = 0

        def detect_batch(self, images):  # noqa: ANN001
            self.worker_process_starts = 1
            self.worker_request_count += 1
            return [[] for _image in images]

        def close(self) -> None:
            return None

    monkeypatch.setattr(smoke_module, "PersistentBioClipScorer", FakeScorer)
    monkeypatch.setattr(smoke_module, "YoloE26SidecarObjectDetector", FakeDetector)

    result = run_prototype_vision_smoke(config)

    assert result.report["status"] == "passed"
    assert result.report["bioclip"]["embedding_shape"] == [5, 2]
    assert result.report["bioclip"]["content_hashes_match"] is True
    assert result.report["yoloe"]["persistent_worker_process_starts"] == 1
    assert result.report["yoloe"]["persistent_worker_requests"] == 2
    assert result.report["batch_settings"]["yoloe_actual_batches"] == [3, 2]
    assert {row["route"]["detection_route"] for row in result.report["per_image"]} == {
        "no_relevant_organism"
    }
    assert result.report_path.is_file()
    assert result.summary_path.is_file()


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
