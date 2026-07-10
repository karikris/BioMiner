from __future__ import annotations

from pathlib import Path
from threading import Event

import httpx
import polars as pl

from biominer.bioclip.object_runner import empty_object_score_frame
from biominer.storage.parquet import write_parquet
from biominer.vision.rolling_worker import (
    BatchPlanner,
    CommitResult,
    CommitWorker,
    DetectionBatch,
    ImageBatch,
    ImageStager,
    PlannedBatch,
    RollingVisionWorker,
    RollingVisionWorkerSettings,
    ScoreBatch,
    ScoreInputBatch,
)
from factories import canonical_records, object_detection_row, object_detections


def test_batch_planner_slices_records_into_deterministic_parts() -> None:
    records = pl.DataFrame({"source": ["flickr"] * 5, "flickr_photo_id": [f"photo-{index}" for index in range(5)]})

    batches = list(BatchPlanner(batch_rows=2).plan(records))

    assert [(batch.batch_index, batch.batch_id, batch.part_id, batch.records.height) for batch in batches] == [
        (0, "vision-batch-000000", "part-000000", 2),
        (1, "vision-batch-000001", "part-000001", 2),
        (2, "vision-batch-000002", "part-000002", 1),
    ]
    assert batches[1].records["flickr_photo_id"].to_list() == ["photo-2", "photo-3"]


def test_image_stager_caches_http_images_and_writes_manifest(tmp_path) -> None:
    requests_seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(str(request.url))
        return httpx.Response(200, headers={"Content-Type": "image/jpeg"}, content=b"fake-jpeg-bytes")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    stager = ImageStager(output_dir=tmp_path / "out", cache_root=tmp_path / "cache", http_client=client)
    planned = PlannedBatch(
        batch_index=0,
        batch_id="vision-batch-000000",
        part_id="part-000000",
        records=pl.DataFrame(
            [
                {
                    "source": "flickr",
                    "flickr_photo_id": "photo-1",
                    "image_url": "https://live.staticflickr.com/photo-1.jpg",
                }
            ]
        ),
    )

    batch = stager(planned)

    manifest = pl.read_parquet(batch.image_batch_manifest)
    assert requests_seen == ["https://live.staticflickr.com/photo-1.jpg"]
    assert batch.records["staged_image_path"].item()
    assert Path(batch.records["staged_image_path"].item()).exists()
    assert batch.cached_image_paths == (Path(batch.records["staged_image_path"].item()),)
    assert batch.failed_image_records == ()
    assert manifest["image_cache_status"].to_list() == ["cached"]
    assert manifest["image_cache_path"].to_list() == [str(batch.cached_image_paths[0])]


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


def test_commit_worker_deletes_cached_images_and_score_inputs_only_after_outputs_commit(tmp_path: Path) -> None:
    cached_image = tmp_path / "cached" / "photo.jpg"
    cached_image.parent.mkdir(parents=True)
    cached_image.write_bytes(b"cached-image")
    score_input_dir = tmp_path / "score-inputs"
    score_input_dir.mkdir()
    (score_input_dir / "input.ppm").write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
    output_dir = tmp_path / "rolling"
    image_batch = ImageBatch(
        batch_index=0,
        batch_id="vision-batch-000000",
        part_id="part-000000",
        records=canonical_records(),
        cached_image_paths=(cached_image,),
    )
    detection_path = write_parquet(object_detections(), tmp_path / "detections.parquet")
    score_input_path = write_parquet(pl.DataFrame([{"detection_id": "det-1"}]), tmp_path / "score-inputs.parquet")
    score_path = write_parquet(empty_object_score_frame(), tmp_path / "scores.parquet")
    score_batch = ScoreBatch(
        score_input_batch=ScoreInputBatch(
            detection_batch=DetectionBatch(image_batch=image_batch, frame=object_detections(), output_path=detection_path),
            frame=pl.DataFrame([{"detection_id": "det-1"}]),
            output_path=score_input_path,
            temp_dir=score_input_dir,
        ),
        frame=empty_object_score_frame(),
        output_path=score_path,
    )

    result = CommitWorker(output_dir=output_dir)(score_batch)

    assert not cached_image.exists()
    assert not score_input_dir.exists()
    assert Path(result.part_outputs["canonical_source_records"]).exists()
    assert Path(result.part_outputs["object_evidence_joined"]).exists()
    assert Path(result.part_outputs["photo_evidence_summary"]).exists()
    assert result.cleanup_paths_deleted == 2


def test_commit_worker_failure_preserves_retryable_inputs(tmp_path: Path) -> None:
    cached_image = tmp_path / "cached" / "photo.jpg"
    cached_image.parent.mkdir(parents=True)
    cached_image.write_bytes(b"cached-image")
    score_input_dir = tmp_path / "score-inputs"
    score_input_dir.mkdir()
    (score_input_dir / "input.ppm").write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
    image_batch = ImageBatch(
        batch_index=0,
        batch_id="vision-batch-000000",
        part_id="part-000000",
        records=canonical_records(),
        cached_image_paths=(cached_image,),
    )
    score_path = write_parquet(empty_object_score_frame(), tmp_path / "scores.parquet")
    score_batch = ScoreBatch(
        score_input_batch=ScoreInputBatch(
            detection_batch=DetectionBatch(
                image_batch=image_batch,
                frame=object_detections(object_detection_row()),
                output_path=None,
            ),
            frame=pl.DataFrame([{"detection_id": "det-1"}]),
            output_path=tmp_path / "score-inputs.parquet",
            temp_dir=score_input_dir,
        ),
        frame=empty_object_score_frame(),
        output_path=score_path,
    )

    try:
        CommitWorker(output_dir=tmp_path / "rolling")(score_batch)
    except ValueError as exc:
        assert "requires detection" in str(exc)
    else:  # pragma: no cover - defensive assertion.
        raise AssertionError("expected commit failure")

    assert cached_image.exists()
    assert score_input_dir.exists()
