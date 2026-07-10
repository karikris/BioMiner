from __future__ import annotations

from threading import Event
from typing import Any

import polars as pl

from biominer.vision.rolling_worker import (
    BatchPlanner,
    CommitResult,
    DetectionBatch,
    ImageBatch,
    PlannedBatch,
    RollingVisionWorker,
    RollingVisionWorkerSettings,
    ScoreBatch,
    ScoreInputBatch,
)


def test_batch_planner_slices_records_into_deterministic_parts() -> None:
    records = pl.DataFrame({"source": ["flickr"] * 5, "flickr_photo_id": [f"photo-{index}" for index in range(5)]})

    batches = list(BatchPlanner(batch_rows=2).plan(records))

    assert [(batch.batch_index, batch.batch_id, batch.part_id, batch.records.height) for batch in batches] == [
        (0, "vision-batch-000000", "part-000000", 2),
        (1, "vision-batch-000001", "part-000001", 2),
        (2, "vision-batch-000002", "part-000002", 1),
    ]
    assert batches[1].records["flickr_photo_id"].to_list() == ["photo-2", "photo-3"]


def test_rolling_worker_starts_next_yolo_batch_before_bioclip_finishes_previous() -> None:
    records = pl.DataFrame({"source": ["flickr"] * 2, "flickr_photo_id": ["photo-1", "photo-2"]})
    second_yolo_started = Event()
    allow_first_score_to_finish = Event()
    events: list[str] = []

    def image_stage(planned: PlannedBatch) -> ImageBatch:
        events.append(f"image:{planned.batch_index}")
        return ImageBatch(
            batch_index=planned.batch_index,
            batch_id=planned.batch_id,
            part_id=planned.part_id,
            records=planned.records,
        )

    def detection_stage(batch: ImageBatch) -> DetectionBatch:
        events.append(f"yolo:{batch.batch_index}")
        if batch.batch_index == 1:
            second_yolo_started.set()
        return DetectionBatch(image_batch=batch, frame=pl.DataFrame({"detection_id": [f"det-{batch.batch_index}"]}))

    def score_input_stage(batch: DetectionBatch) -> ScoreInputBatch:
        return ScoreInputBatch(detection_batch=batch, frame=pl.DataFrame({"detection_id": batch.frame["detection_id"]}))

    def score_stage(batch: ScoreInputBatch) -> ScoreBatch:
        index = batch.detection_batch.image_batch.batch_index
        events.append(f"score-start:{index}")
        if index == 0:
            assert second_yolo_started.wait(timeout=2.0)
            allow_first_score_to_finish.set()
        events.append(f"score-end:{index}")
        return ScoreBatch(score_input_batch=batch, frame=pl.DataFrame({"detection_id": batch.frame["detection_id"]}))

    def commit_stage(batch: ScoreBatch) -> CommitResult:
        index = batch.score_input_batch.detection_batch.image_batch.batch_index
        events.append(f"commit:{index}")
        return CommitResult(batch_id=f"batch-{index}", part_outputs={"score": f"part-{index}.parquet"})

    worker = RollingVisionWorker(
        settings=RollingVisionWorkerSettings(vision_batch_rows=1, image_prefetch_batches=2),
        image_stage=image_stage,
        detection_stage=detection_stage,
        score_input_stage=score_input_stage,
        score_stage=score_stage,
        commit_stage=commit_stage,
    )

    result = worker.run(records)

    assert allow_first_score_to_finish.is_set()
    assert events.index("yolo:1") < events.index("score-end:0")
    assert result.batches_seen == 2
    assert result.batches_committed == 2


def test_rolling_worker_image_slots_never_exceed_prefetch_limit() -> None:
    records = pl.DataFrame({"source": ["flickr"] * 6, "flickr_photo_id": [f"photo-{index}" for index in range(6)]})

    def image_stage(planned: PlannedBatch) -> ImageBatch:
        return ImageBatch(
            batch_index=planned.batch_index,
            batch_id=planned.batch_id,
            part_id=planned.part_id,
            records=planned.records,
        )

    def detection_stage(batch: ImageBatch) -> DetectionBatch:
        return DetectionBatch(image_batch=batch, frame=pl.DataFrame({"detection_id": [f"det-{batch.batch_index}"]}))

    def score_input_stage(batch: DetectionBatch) -> ScoreInputBatch:
        return ScoreInputBatch(detection_batch=batch, frame=pl.DataFrame({"detection_id": batch.frame["detection_id"]}))

    def score_stage(batch: ScoreInputBatch) -> ScoreBatch:
        return ScoreBatch(score_input_batch=batch, frame=pl.DataFrame({"detection_id": batch.frame["detection_id"]}))

    def commit_stage(batch: ScoreBatch) -> CommitResult:
        index = batch.score_input_batch.detection_batch.image_batch.batch_index
        return CommitResult(batch_id=f"batch-{index}", part_outputs={})

    worker = RollingVisionWorker(
        settings=RollingVisionWorkerSettings(vision_batch_rows=1, image_prefetch_batches=2),
        image_stage=image_stage,
        detection_stage=detection_stage,
        score_input_stage=score_input_stage,
        score_stage=score_stage,
        commit_stage=commit_stage,
    )

    result = worker.run(records)

    assert result.batches_committed == 6
    assert worker.max_resident_image_batches <= 2
    assert result.metrics["max_resident_image_batches"] <= 2
