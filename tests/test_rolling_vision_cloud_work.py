from __future__ import annotations

from pathlib import Path
from threading import Event, get_ident

import pytest

import polars as pl

from biominer.vision.cloud_work import (
    RollingVisionPipelineError,
    enqueue_rolling_vision_work_from_source_shards,
    run_bounded_cloud_rolling_pipeline,
)
from biominer.workstore.sqlite import SQLiteWorkStore


def test_rolling_work_planning_streams_bounded_batches_and_is_resumable(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    storage = StreamingOnlyStorage()
    workstore = SQLiteWorkStore(tmp_path / "workstore.sqlite")
    source_uris = ("s3://bucket/source-1.parquet", "s3://bucket/source-2.parquet")
    for shard_index, uri in enumerate(source_uris):
        storage.frames[uri] = pl.DataFrame(
            [
                {
                    "source": "flickr",
                    "flickr_photo_id": f"photo-{shard_index}-{row_index}",
                    "source_record_hash": f"sha256:{shard_index}-{row_index}",
                    "image_url": f"https://example.invalid/{shard_index}/{row_index}.jpg",
                }
                for row_index in range(3)
            ]
        )
        workstore.register_shard(
            job_name="biominer_production_run",
            registry_version="registry-v2",
            stage="poll_flickr",
            run_id="run-1",
            worker_id="poller",
            uri=uri,
            checksum=None,
            row_count=3,
        )

    enqueue_sizes: list[int] = []
    payload_batch_sizes: list[int] = []
    original_enqueue = workstore.enqueue_work

    def recording_enqueue(job_name, registry_version=None, items=None, *, stage="default"):  # noqa: ANN001, ANN202
        values = list(items or [])
        enqueue_sizes.append(len(values))
        payload_batch_sizes.extend(len(item["source_records"]) for item in values)
        return original_enqueue(job_name, registry_version, values, stage=stage)

    monkeypatch.setattr(workstore, "enqueue_work", recording_enqueue)
    kwargs = {
        "storage": storage,
        "workstore": workstore,
        "job_name": "biominer_production_run",
        "registry_version": "registry-v2",
        "run_id": "run-1",
        "source_stage": "poll_flickr",
        "vision_batch_rows": 2,
        "limit": 5,
    }

    first = enqueue_rolling_vision_work_from_source_shards(**kwargs)
    second = enqueue_rolling_vision_work_from_source_shards(**kwargs)

    assert first.source_shards_seen == 2
    assert first.source_records_seen == 5
    assert first.batches_planned == 3
    assert first.enqueued_work_items == 3
    assert first.duplicate_work_items == 0
    assert second.enqueued_work_items == 0
    assert second.duplicate_work_items == 3
    assert enqueue_sizes == [1, 1, 1, 1, 1, 1]
    assert payload_batch_sizes == [2, 2, 1, 2, 2, 1]
    assert storage.iter_batch_sizes == [2, 2, 2, 2]


class StreamingOnlyStorage:
    def __init__(self) -> None:
        self.frames: dict[str, pl.DataFrame] = {}
        self.iter_batch_sizes: list[int] = []

    def read_parquet(self, _uri: str) -> pl.DataFrame:
        raise AssertionError("rolling planning must not eagerly read a source shard")

    def iter_parquet_batches(self, uri: str, *, batch_size: int):  # noqa: ANN201
        self.iter_batch_sizes.append(batch_size)
        yield from self.frames[uri].iter_slices(batch_size)


def test_bounded_cloud_pipeline_overlaps_next_detection_and_commits_in_order() -> None:
    main_thread = get_ident()
    next_detection_started = Event()
    score_observed_overlap = Event()
    commits: list[str] = []
    items = [{"work_key": f"batch-{index}"} for index in range(3)]

    def detect(item):  # noqa: ANN001, ANN202
        if item["work_key"] == "batch-1":
            next_detection_started.set()
        return {"work_key": item["work_key"]}

    def score(detected):  # noqa: ANN001, ANN202
        if detected["work_key"] == "batch-0":
            score_observed_overlap.set() if next_detection_started.wait(timeout=1) else None
        return detected["work_key"]

    def commit(item, scored):  # noqa: ANN001, ANN202
        assert get_ident() == main_thread
        assert item["work_key"] == scored
        commits.append(scored)
        return scored

    result = run_bounded_cloud_rolling_pipeline(items, detect=detect, score=score, commit=commit)

    assert score_observed_overlap.is_set()
    assert commits == ["batch-0", "batch-1", "batch-2"]
    assert result.commit_results == tuple(commits)
    assert result.buffer_capacity == 1
    assert result.batches_committed == 3


def test_bounded_cloud_pipeline_reports_failing_work_item_and_phase() -> None:
    items = [{"work_key": "batch-0"}, {"work_key": "batch-1"}]

    def score(detected):  # noqa: ANN001, ANN202
        if detected["work_key"] == "batch-1":
            raise RuntimeError("gpu failure")
        return detected

    with pytest.raises(RollingVisionPipelineError, match="score failed for batch-1") as caught:
        run_bounded_cloud_rolling_pipeline(
            items,
            detect=lambda item: item,
            score=score,
            commit=lambda _item, scored: scored,
        )

    assert caught.value.phase == "score"
    assert caught.value.work_item["work_key"] == "batch-1"
