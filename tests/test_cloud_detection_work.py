from __future__ import annotations

from pathlib import Path

import polars as pl

from biominer.detection.cloud_work import enqueue_detection_work_from_source_shards
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


class _FakeCloudStorage:
    def __init__(self) -> None:
        self.parquet_payloads: dict[str, pl.DataFrame] = {}

    def read_parquet(self, uri: str) -> pl.DataFrame:
        return self.parquet_payloads[uri]
