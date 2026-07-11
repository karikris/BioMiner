from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION
from biominer.detection.policy import VisionRuntimeSettings
from biominer.storage.parquet import ParquetPartWrite
from biominer.vision.cloud_work import (
    ROLLING_VISION_ARTIFACT_ORDER,
    ROLLING_VISION_ARTIFACT_STAGES,
    commit_rolling_vision_batch_shards,
    enqueue_rolling_vision_work_from_source_shards,
    rolling_vision_settings_key,
    rolling_vision_work_item,
)
from biominer.workstore.sqlite import SQLiteWorkStore


def test_rolling_vision_work_key_changes_by_output_affecting_settings() -> None:
    records = [
        {
            "source": "flickr",
            "flickr_photo_id": "photo-1",
            "source_record_hash": "sha256:source-1",
            "image_url": "https://live.staticflickr.com/photo-1.jpg",
        }
    ]
    detector = {"backend": "yoloe26", "model_id": "yoloe26", "model_version": "test", "checkpoint": "yoloe-26s-seg.pt"}
    bioclip_model = {"model_id": "bioclip-2.5", "model_version": "test", "checkpoint": "hf-hub:imageomics/bioclip-2.5-vith14"}
    base_settings = rolling_vision_settings_key(
        detector=detector,
        vision_settings=VisionRuntimeSettings(yolo_imgsz=768, yolo_conf=0.20, yolo_iou=0.50, yolo_max_det=8),
        bioclip_gate_mode="exclude_hard_negative",
        score_no_detection_whole_image=True,
        bioclip_model=bioclip_model,
        candidate_set_id="candidate-set-v1",
        classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        taxonomy_table_version="taxonomy-v1",
        taxonomy_prompt_variant_version="prompt-v1",
        family_top_k=3,
        species_first_pass_top_k=20,
        species_rerank_top_k=20,
    )
    changed_settings = rolling_vision_settings_key(
        detector=detector,
        vision_settings=VisionRuntimeSettings(yolo_imgsz=640, yolo_conf=0.20, yolo_iou=0.50, yolo_max_det=8),
        bioclip_gate_mode="exclude_hard_negative",
        score_no_detection_whole_image=True,
        bioclip_model=bioclip_model,
        candidate_set_id="candidate-set-v1",
        classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        taxonomy_table_version="taxonomy-v1",
        taxonomy_prompt_variant_version="prompt-v1",
        family_top_k=3,
        species_first_pass_top_k=20,
        species_rerank_top_k=20,
    )

    base = rolling_vision_work_item(
        records,
        run_id="run-1",
        batch_index=0,
        vision_batch_rows=500,
        source_shard_uris=["s3://biominer/source.parquet"],
        settings_key=base_settings,
    )
    changed = rolling_vision_work_item(
        records,
        run_id="run-1",
        batch_index=0,
        vision_batch_rows=500,
        source_shard_uris=["s3://biominer/source.parquet"],
        settings_key=changed_settings,
    )

    assert base["work_key"] != changed["work_key"]
    assert base["settings_key"]["detector"]["yolo_imgsz"] == 768
    assert base["settings_key"]["crop"]["crop_target_px"] == 336
    assert base["settings_key"]["bioclip_gate"]["mode"] == "exclude_hard_negative"
    assert base["settings_key"]["bioclip_gate"]["score_no_detection_whole_image"] is True
    assert base["settings_key"]["bioclip_model"]["checkpoint"] == "hf-hub:imageomics/bioclip-2.5-vith14"
    assert base["settings_key"]["classification_mode"] == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
    assert base["settings_key"]["taxonomy_table_version"] == "taxonomy-v1"
    assert base["settings_key"]["taxonomy_prompt_variant_version"] == "prompt-v1"
    assert base["settings_key"]["top_k_settings"] == {
        "family_top_k": 3,
        "species_first_pass_top_k": 20,
        "species_rerank_top_k": 20,
    }


def test_enqueue_rolling_vision_work_batches_source_shards_deterministically(tmp_path: Path) -> None:
    storage = _FakeCloudStorage()
    workstore = SQLiteWorkStore(tmp_path / "workstore.sqlite")
    source_a = "s3://biominer/source/a.parquet"
    source_b = "s3://biominer/source/b.parquet"
    storage.parquet_payloads[source_b] = pl.DataFrame([_source_record("photo-b")])
    storage.parquet_payloads[source_a] = pl.DataFrame([_source_record("photo-a")])
    for uri in (source_b, source_a):
        workstore.register_shard(
            job_name="biominer_production_run",
            registry_version="registry-v1",
            stage="poll_flickr",
            run_id="run-1",
            worker_id="poller",
            uri=uri,
            checksum=None,
            row_count=1,
        )

    first = enqueue_rolling_vision_work_from_source_shards(
        storage=storage,
        workstore=workstore,
        job_name="biominer_production_run",
        registry_version="registry-v1",
        run_id="run-1",
        source_stage="poll_flickr",
        vision_batch_rows=1,
        detector={"backend": "yoloe26", "model_id": "yoloe26", "model_version": "test", "checkpoint": "ckpt"},
        vision_settings=VisionRuntimeSettings(yolo_imgsz=768),
        bioclip_model={"model_id": "bioclip", "model_version": "test", "checkpoint": "model"},
    )
    second = enqueue_rolling_vision_work_from_source_shards(
        storage=storage,
        workstore=workstore,
        job_name="biominer_production_run",
        registry_version="registry-v1",
        run_id="run-1",
        source_stage="poll_flickr",
        vision_batch_rows=1,
        detector={"backend": "yoloe26", "model_id": "yoloe26", "model_version": "test", "checkpoint": "ckpt"},
        vision_settings=VisionRuntimeSettings(yolo_imgsz=768),
        bioclip_model={"model_id": "bioclip", "model_version": "test", "checkpoint": "model"},
    )

    assert first.source_shards_seen == 2
    assert first.source_records_seen == 2
    assert first.batches_planned == 2
    assert first.enqueued_work_items == 2
    assert second.enqueued_work_items == 0
    assert second.duplicate_work_items == 2
    items = workstore.list_work_items(
        job_name="biominer_production_run",
        stage="detect_objects",
        registry_version="registry-v1",
    )
    assert [item["payload"]["batch_id"] for item in items] == ["vision-batch-000000", "vision-batch-000001"]
    assert [item["payload"]["source_records"][0]["flickr_photo_id"] for item in items] == ["photo-a", "photo-b"]


def test_commit_rolling_vision_shards_writes_all_parts_before_registering_and_completing() -> None:
    storage = _RecordingStorage()
    workstore = _RecordingWorkStore()
    frames = {artifact: pl.DataFrame({"artifact": [artifact]}) for artifact in ROLLING_VISION_ARTIFACT_ORDER}

    result = commit_rolling_vision_batch_shards(
        storage=storage,
        workstore=workstore,
        job_name="biominer_production_run",
        registry_version="registry-v1",
        run_id="run-1",
        worker_id="worker-1",
        base_prefix="s3://biominer/runs/run_id=run-1/staging",
        work_key="run-1:rolling-vision:abc",
        batch_id="vision-batch-000000",
        part_id="part-000000",
        frames=frames,
        compression="zstd",
    )

    first_register_index = workstore.events.index("register:image_batch_manifest")
    assert storage.events == [f"write:{artifact}" for artifact in ROLLING_VISION_ARTIFACT_ORDER]
    assert first_register_index == 0
    assert workstore.events[-1] == "complete:run-1:rolling-vision:abc"
    assert [event.split(":", 1)[1] for event in workstore.events[:-1]] == list(ROLLING_VISION_ARTIFACT_ORDER)
    assert result.parts_written == len(ROLLING_VISION_ARTIFACT_ORDER)
    assert result.parts_reused == 0
    assert set(result.output_uris) == set(ROLLING_VISION_ARTIFACT_ORDER)
    assert workstore.completed[0]["output_uri"] == result.output_uris["photo_evidence_summary"]
    assert {shard["stage"] for shard in workstore.shards} == set(ROLLING_VISION_ARTIFACT_STAGES.values())


def test_commit_rolling_vision_shards_does_not_register_or_complete_after_write_failure() -> None:
    storage = _RecordingStorage(fail_on_artifact="object_bioclip_scores")
    workstore = _RecordingWorkStore()
    frames = {artifact: pl.DataFrame({"artifact": [artifact]}) for artifact in ROLLING_VISION_ARTIFACT_ORDER}

    with pytest.raises(RuntimeError, match="write failed"):
        commit_rolling_vision_batch_shards(
            storage=storage,
            workstore=workstore,
            job_name="biominer_production_run",
            registry_version="registry-v1",
            run_id="run-1",
            worker_id="worker-1",
            base_prefix="s3://biominer/runs/run_id=run-1/staging",
            work_key="run-1:rolling-vision:abc",
            batch_id="vision-batch-000000",
            part_id="part-000000",
            frames=frames,
        )

    assert storage.events == [
        "write:image_batch_manifest",
        "write:object_detections",
        "write:bioclip_score_inputs",
        "write:object_bioclip_scores",
    ]
    assert workstore.events == []
    assert workstore.shards == []
    assert workstore.completed == []


def _source_record(photo_id: str) -> dict[str, str]:
    return {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "source_record_hash": f"sha256:{photo_id}",
        "image_url": f"https://live.staticflickr.com/{photo_id}.jpg",
    }


class _FakeCloudStorage:
    def __init__(self) -> None:
        self.parquet_payloads: dict[str, pl.DataFrame] = {}

    def read_parquet(self, uri: str) -> pl.DataFrame:
        return self.parquet_payloads[uri]

    def iter_parquet_batches(self, uri: str, *, batch_size: int):  # noqa: ANN201
        yield from self.parquet_payloads[uri].iter_slices(batch_size)


class _RecordingStorage:
    def __init__(self, *, fail_on_artifact: str | None = None) -> None:
        self.fail_on_artifact = fail_on_artifact
        self.events: list[str] = []
        self.parquet_payloads: dict[str, pl.DataFrame] = {}

    def write_parquet_part(
        self,
        uri: str,
        frame: pl.DataFrame,
        *,
        compression: str | None = "zstd",
        overwrite: bool = False,
    ) -> ParquetPartWrite:
        artifact = _artifact_from_uri(uri)
        self.events.append(f"write:{artifact}")
        if artifact == self.fail_on_artifact:
            raise RuntimeError(f"write failed for {artifact}")
        if not overwrite and uri in self.parquet_payloads:
            raise FileExistsError(uri)
        self.parquet_payloads[uri] = frame
        return ParquetPartWrite(uri=uri, row_count=frame.height, byte_count=None, compression=compression)

    def exists(self, uri: str) -> bool:
        return uri in self.parquet_payloads


class _RecordingWorkStore:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.shards: list[dict[str, Any]] = []
        self.completed: list[dict[str, Any]] = []

    def register_shard(self, **kwargs: Any) -> None:
        artifact = str((kwargs.get("metadata") or {}).get("artifact") or "")
        self.events.append(f"register:{artifact}")
        self.shards.append(dict(kwargs))

    def mark_completed(self, work_key: str, output_uri: str | None, checksum: str | None, row_count: int | None) -> None:
        self.events.append(f"complete:{work_key}")
        self.completed.append(
            {
                "work_key": work_key,
                "output_uri": output_uri,
                "checksum": checksum,
                "row_count": row_count,
            }
        )


def _artifact_from_uri(uri: str) -> str:
    marker = "/stage="
    stage = uri.split(marker, 1)[1].split("/", 1)[0]
    return {
        "image_batch_manifest": "image_batch_manifest",
        "object_detections": "object_detections",
        "bioclip_score_inputs": "bioclip_score_inputs",
        "object_bioclip_scores": "object_bioclip_scores",
        "object_evidence_joined": "object_evidence_joined",
        "photo_evidence_summary": "photo_evidence_summary",
    }[stage]
