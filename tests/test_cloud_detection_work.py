from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from biominer.detection.cloud_work import detection_work_item, enqueue_detection_work_from_source_shards, run_cloud_detection_batch
from biominer.detection.detector_base import DecodedImage, DetectionCandidate
from biominer.detection.policy import DetectionPolicy, VisionRuntimeSettings
from biominer.detection.routing import DetectionRoutingPolicy
from biominer.run.stages import RunStage
from biominer.workstore.sqlite import SQLiteWorkStore


def test_enqueue_detection_work_from_source_shard_inventory_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    storage = _FakeCloudStorage()
    workstore = SQLiteWorkStore(tmp_path / "workstore.sqlite")
    enqueue_batch_sizes: list[int] = []
    original_enqueue_work = workstore.enqueue_work

    def recording_enqueue_work(job_name, registry_version=None, items=None, *, stage="default"):  # noqa: ANN001, ANN202
        enqueue_batch_sizes.append(len(items or []))
        return original_enqueue_work(job_name, registry_version, items, stage=stage)

    monkeypatch.setattr(workstore, "enqueue_work", recording_enqueue_work)
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
        read_batch_size=1,
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
        read_batch_size=1,
    )

    assert first.source_shards_seen == 1
    assert first.source_records_seen == 2
    assert first.enqueued_work_items == 2
    assert first.duplicate_work_items == 0
    assert second.enqueued_work_items == 0
    assert second.duplicate_work_items == 2
    assert enqueue_batch_sizes == [1, 1, 1, 1]
    items = workstore.list_work_items(
        job_name="biominer_production_run",
        stage=RunStage.DETECT_OBJECTS.value,
        registry_version="registry-v1",
    )
    assert [item["status"] for item in items] == ["pending", "pending"]
    assert {item["payload"]["source_shard_uri"] for item in items} == {source_uri}
    assert {item["payload"]["source_record"]["flickr_photo_id"] for item in items} == {"photo-1", "photo-2"}
    assert all(item["payload"]["detector"]["checkpoint"] == "fake-checkpoint" for item in items)


def test_detection_work_item_key_changes_by_detector_policy_and_runtime_settings() -> None:
    record = {
        "source": "flickr",
        "flickr_photo_id": "photo-1",
        "source_record_hash": "sha256:source-1",
        "image_url": "https://live.staticflickr.com/photo-1.jpg",
    }
    detector = {"backend": "fake", "model_id": "fake-detector", "model_version": "test", "checkpoint": "fake-checkpoint"}
    base = detection_work_item(
        record,
        run_id="run-1",
        source_shard_uri="s3://biominer/source.parquet",
        detector=detector,
        detection_policy=DetectionPolicy(backend="fake", crop_padding_ratio=0.12),
        vision_settings=VisionRuntimeSettings(yolo_imgsz=640, yolo_conf=0.20, yolo_iou=0.50, yolo_max_det=8),
    )
    padding_changed = detection_work_item(
        record,
        run_id="run-1",
        source_shard_uri="s3://biominer/source.parquet",
        detector=detector,
        detection_policy=DetectionPolicy(backend="fake", crop_padding_ratio=0.18),
        vision_settings=VisionRuntimeSettings(yolo_imgsz=640, yolo_conf=0.20, yolo_iou=0.50, yolo_max_det=8),
    )
    imgsz_changed = detection_work_item(
        record,
        run_id="run-1",
        source_shard_uri="s3://biominer/source.parquet",
        detector=detector,
        detection_policy=DetectionPolicy(backend="fake", crop_padding_ratio=0.12),
        vision_settings=VisionRuntimeSettings(yolo_imgsz=768, yolo_conf=0.20, yolo_iou=0.50, yolo_max_det=8),
    )

    assert base["work_key"] != padding_changed["work_key"]
    assert base["work_key"] != imgsz_changed["work_key"]
    assert base["detection_policy"]["crop_padding_ratio"] == 0.12
    assert "bioclip_eligible_labels" not in base["detection_policy"]
    assert base["detection_policy"]["routing_policy"] == {
        "version": "detection-routing-policy-v1",
        "fingerprint": DetectionRoutingPolicy().fingerprint,
        "possible_adult_route_enabled": True,
        "possible_adult_route_threshold": 0.35,
        "ambiguous_insect_review_enabled": True,
        "ambiguous_insect_review_threshold": 0.35,
    }
    assert base["vision_runtime"]["yolo_imgsz"] == 640


def test_detection_work_key_binds_ordered_prompt_set_and_profile_routing() -> None:
    record = {
        "source": "flickr",
        "flickr_photo_id": "photo-1",
        "source_record_hash": "sha256:source-1",
        "image_url": "https://live.staticflickr.com/photo-1.jpg",
    }
    base_detector = {
        "backend": "fake",
        "model_id": "fake-detector",
        "model_version": "test",
        "checkpoint": "fake-checkpoint",
        "prompt_classes": ["butterfly", "moth"],
        "prompt_set_fingerprint": "sha256:" + "a" * 64,
    }
    settings = VisionRuntimeSettings(possible_adult_route_threshold=0.55)
    base = detection_work_item(
        record,
        run_id="run-1",
        source_shard_uri="s3://biominer/source.parquet",
        detector=base_detector,
        vision_settings=settings,
    )
    reordered = detection_work_item(
        record,
        run_id="run-1",
        source_shard_uri="s3://biominer/source.parquet",
        detector={
            **base_detector,
            "prompt_classes": ["moth", "butterfly"],
            "prompt_set_fingerprint": "sha256:" + "b" * 64,
        },
        vision_settings=settings,
    )

    assert reordered["work_key"] != base["work_key"]
    assert base["detector"]["prompt_classes"] == ["butterfly", "moth"]
    assert base["vision_runtime"]["possible_adult_route_threshold"] == 0.55
    assert (
        base["detection_policy"]["routing_policy"][
            "possible_adult_route_threshold"
        ]
        == 0.55
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", "detection-routing-policy-v2"),
        ("possible_adult_route_enabled", False),
        ("possible_adult_route_threshold", 0.55),
        ("ambiguous_insect_review_enabled", False),
        ("ambiguous_insect_review_threshold", 0.65),
    ],
)
def test_detection_work_key_changes_by_complete_routing_policy_identity(
    field: str,
    value: object,
) -> None:
    record = {
        "source": "flickr",
        "flickr_photo_id": "photo-1",
        "source_record_hash": "sha256:source-1",
        "image_url": "https://live.staticflickr.com/photo-1.jpg",
    }
    detector = {
        "backend": "fake",
        "model_id": "fake-detector",
        "model_version": "test",
        "checkpoint": "fake-checkpoint",
    }
    base = detection_work_item(
        record,
        run_id="run-1",
        source_shard_uri="s3://biominer/source.parquet",
        detector=detector,
        detection_policy=DetectionPolicy(backend="fake"),
    )
    routing_values = {
        "version": "detection-routing-policy-v1",
        "possible_adult_route_enabled": True,
        "possible_adult_route_threshold": 0.35,
        "ambiguous_insect_review_enabled": True,
        "ambiguous_insect_review_threshold": 0.35,
        field: value,
    }
    changed = detection_work_item(
        record,
        run_id="run-1",
        source_shard_uri="s3://biominer/source.parquet",
        detector=detector,
        detection_policy=DetectionPolicy(
            backend="fake",
            routing_policy=DetectionRoutingPolicy(**routing_values),
        ),
    )

    assert changed["work_key"] != base["work_key"]
    assert changed["detection_policy"]["routing_policy"][field] == value
    assert (
        changed["detection_policy"]["routing_policy"]["fingerprint"]
        != base["detection_policy"]["routing_policy"]["fingerprint"]
    )


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


def test_run_cloud_detection_batch_routes_image_failure_without_scoring() -> None:
    class PromptedDetector:
        backend = "fake"
        model_id = "fake-detector"
        model_version = "test"
        checkpoint = "fake-checkpoint"
        prompt_set_fingerprint = "sha256:" + "c" * 64

        def detect_batch(self, images):  # noqa: ANN001, ANN202
            raise AssertionError("image failure must not reach detection")

    result = run_cloud_detection_batch(
        work_items=_detection_work_items(
            1, prompt_set_fingerprint="sha256:" + "c" * 64
        ),
        detector=PromptedDetector(),
        image_loader=lambda _record: (_ for _ in ()).throw(ValueError("bad image")),
    )

    row = result.frame.to_dicts()[0]
    assert row["detection_status"] == "failed_image_load"
    assert row["detection_route"] == "ambiguous_visual_domain"
    assert row["routing_action"] == "exclude"
    assert row["bioclip_route"] is None
    assert row["detector_prompt_set_fingerprint"] == "sha256:" + "c" * 64


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


def _detection_work_items(
    count: int, *, prompt_set_fingerprint: str = ""
) -> list[dict[str, object]]:
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
            detector={
                "backend": "fake",
                "model_id": "fake-detector",
                "model_version": "test",
                "checkpoint": "fake-checkpoint",
                "prompt_set_fingerprint": prompt_set_fingerprint,
            },
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

    def iter_parquet_batches(self, uri: str, *, batch_size: int):  # noqa: ANN201 - fake protocol implementation.
        yield from self.parquet_payloads[uri].iter_slices(batch_size)
