from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from biominer.detection.cloud_work import detection_work_item, enqueue_detection_work_from_source_shards, run_cloud_detection_batch
from biominer.detection.detector_base import DecodedImage, DetectionCandidate
from biominer.run.stages import RunStage
from biominer.workstore.sqlite import SQLiteWorkStore


def test_enqueue_detection_work_from_source_shard_inventory_is_idempotent(tmp_path: Path) -> None:
    storage = _FakeCloudStorage()
    workstore = SQLiteWorkStore(tmp_path / "workstore.sqlite")
    source_uri = "s3://biominer/runs/run_id=run-1/staging/evidence/stage=poll_flickr/run_id=run-1/worker=w1/batch=001.parquet"
    storage.parquet_payloads[source_uri] = pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "source_record_hash": "sha256:source-1",
                "image_url": "https://live.staticflickr.com/photo-1.jpg",
                "photo_page_url": "https://www.flickr.com/photos/u/photo-1",
            },
            {
                "source": "flickr",
                "flickr_photo_id": "photo-2",
                "source_record_hash": "sha256:source-2",
                "image_url": "https://live.staticflickr.com/photo-2.jpg",
                "photo_page_url": "https://www.flickr.com/photos/u/photo-2",
            },
        ]
    )
    workstore.register_shard(
        job_name="biominer_production_run",
        registry_version="registry-v1",
        stage=RunStage.POLL_FLICKR.value,
        run_id="run-1",
        worker_id="poller-1",
        uri=source_uri,
        checksum=None,
        row_count=2,
    )

    first = enqueue_detection_work_from_source_shards(
        storage=storage,
        workstore=workstore,
        job_name="biominer_production_run",
        registry_version="registry-v1",
        run_id="run-1",
        source_stage=RunStage.POLL_FLICKR.value,
        detection_stage=RunStage.DETECT_OBJECTS.value,
        detector_backend="fake",
        detector_model_id="fake-detector",
        detector_model_version="test",
        detector_checkpoint="fake-checkpoint",
    )
    second = enqueue_detection_work_from_source_shards(
        storage=storage,
        workstore=workstore,
        job_name="biominer_production_run",
        registry_version="registry-v1",
        run_id="run-1",
        source_stage=RunStage.POLL_FLICKR.value,
        detection_stage=RunStage.DETECT_OBJECTS.value,
        detector_backend="fake",
        detector_model_id="fake-detector",
        detector_model_version="test",
        detector_checkpoint="fake-checkpoint",
    )

    assert first.source_shards_seen == 1
    assert first.source_records_seen == 2
    assert first.enqueued_work_items == 2
    assert first.duplicate_work_items == 0
    assert second.enqueued_work_items == 0
    assert second.duplicate_work_items == 2
    items = workstore.list_work_items(
        job_name="biominer_production_run",
        stage=RunStage.DETECT_OBJECTS.value,
        registry_version="registry-v1",
    )
    assert [item["status"] for item in items] == ["pending", "pending"]
    assert {item["payload"]["source_shard_uri"] for item in items} == {source_uri}
    assert {item["payload"]["source_record"]["flickr_photo_id"] for item in items} == {"photo-1", "photo-2"}
    assert all(item["payload"]["detector"]["checkpoint"] == "fake-checkpoint" for item in items)


def test_run_cloud_detection_batch_chunks_loaded_images_by_detector_batch_size() -> None:
    class RecordingDetector:
        backend = "fake"
        model_id = "fake-detector"
        model_version = "test"
        checkpoint = "fake-checkpoint"

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def detect_batch(self, images):  # noqa: ANN001, ANN202 - mirrors detector protocol.
            self.batch_sizes.append(len(images))
            return [
                [DetectionCandidate(label="butterfly_like", score=0.91, bbox_xyxy=(0.0, 0.0, 2.0, 2.0))]
                for _image in images
            ]

    detector = RecordingDetector()
    work_items = []
    for index in range(5):
        payload = detection_work_item(
            {
                "source": "flickr",
                "flickr_photo_id": f"photo-{index}",
                "source_record_hash": f"sha256:source-{index}",
                "image_url": f"https://live.staticflickr.com/photo-{index}.jpg",
            },
            run_id="run-1",
            source_shard_uri="s3://biominer/source.parquet",
            detector={"backend": "fake", "model_id": "fake-detector", "model_version": "test", "checkpoint": "fake-checkpoint"},
        )
        work_items.append({"work_key": payload["work_key"], "payload": payload})

    result = run_cloud_detection_batch(
        work_items=work_items,
        detector=detector,
        image_loader=lambda record: _decoded_image(),
        detector_batch_size=2,
    )

    assert detector.batch_sizes == [2, 2, 1]
    assert result.records_seen == 5
    assert result.images_loaded == 5
    assert result.detections_written == 5


def test_run_cloud_detection_batch_adaptive_batching_halves_after_memory_error() -> None:
    class AdaptiveDetector:
        backend = "fake"
        model_id = "fake-detector"
        model_version = "test"
        checkpoint = "fake-checkpoint"

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def detect_batch(self, images):  # noqa: ANN001, ANN202 - mirrors detector protocol.
            self.batch_sizes.append(len(images))
            if len(images) > 8:
                raise RuntimeError("CUDA out of memory during YOLO inference")
            return [
                [DetectionCandidate(label="butterfly_like", score=0.91, bbox_xyxy=(0.0, 0.0, 2.0, 2.0))]
                for _image in images
            ]

    detector = AdaptiveDetector()
    result = run_cloud_detection_batch(
        work_items=_detection_work_items(16),
        detector=detector,
        image_loader=lambda record: _decoded_image(),
        detector_batch_size=16,
        adaptive_batching=True,
        min_detector_batch_size=1,
    )

    assert detector.batch_sizes == [16, 8, 8]
    assert result.records_seen == 16
    assert result.images_loaded == 16
    assert result.detections_written == 16
    assert result.adaptive_batching_enabled is True
    assert result.detector_batch_retries == 1
    assert result.detector_batch_size_initial == 16
    assert result.detector_batch_size_final == 8
    assert result.detector_batch_size_min == 1
    assert result.frame.get_column("flickr_photo_id").to_list() == [f"photo-{index}" for index in range(16)]


def test_run_cloud_detection_batch_adaptive_batching_does_not_retry_non_memory_error() -> None:
    class NonMemoryDetector:
        backend = "fake"
        model_id = "fake-detector"
        model_version = "test"
        checkpoint = "fake-checkpoint"

        def detect_batch(self, images):  # noqa: ANN001, ANN202 - mirrors detector protocol.
            raise RuntimeError("invalid YOLO tensor shape")

    with pytest.raises(RuntimeError, match="invalid YOLO tensor shape"):
        run_cloud_detection_batch(
            work_items=_detection_work_items(2),
            detector=NonMemoryDetector(),
            image_loader=lambda record: _decoded_image(),
            detector_batch_size=2,
            adaptive_batching=True,
            min_detector_batch_size=1,
        )


def test_run_cloud_detection_batch_adaptive_batching_reports_min_batch_failure() -> None:
    class AlwaysMemoryDetector:
        backend = "fake"
        model_id = "fake-detector"
        model_version = "test"
        checkpoint = "fake-checkpoint"

        def detect_batch(self, images):  # noqa: ANN001, ANN202 - mirrors detector protocol.
            raise RuntimeError(f"MPS memory exhausted at detector batch size {len(images)}")

    with pytest.raises(RuntimeError, match="MPS memory exhausted at detector batch size 1"):
        run_cloud_detection_batch(
            work_items=_detection_work_items(2),
            detector=AlwaysMemoryDetector(),
            image_loader=lambda record: _decoded_image(),
            detector_batch_size=2,
            adaptive_batching=True,
            min_detector_batch_size=1,
        )


def _detection_work_items(count: int) -> list[dict[str, object]]:
    work_items: list[dict[str, object]] = []
    for index in range(count):
        payload = detection_work_item(
            {
                "source": "flickr",
                "flickr_photo_id": f"photo-{index}",
                "source_record_hash": f"sha256:source-{index}",
                "image_url": f"https://live.staticflickr.com/photo-{index}.jpg",
            },
            run_id="run-1",
            source_shard_uri="s3://biominer/source.parquet",
            detector={"backend": "fake", "model_id": "fake-detector", "model_version": "test", "checkpoint": "fake-checkpoint"},
        )
        work_items.append({"work_key": payload["work_key"], "payload": payload})
    return work_items


def _decoded_image() -> DecodedImage:
    return DecodedImage(
        width=2,
        height=2,
        mode="RGB",
        data=bytes([255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0]),
        source_uri="memory://cloud-detection",
    )


class _FakeCloudStorage:
    def __init__(self) -> None:
        self.parquet_payloads: dict[str, pl.DataFrame] = {}

    def read_parquet(self, uri: str) -> pl.DataFrame:
        return self.parquet_payloads[uri]
