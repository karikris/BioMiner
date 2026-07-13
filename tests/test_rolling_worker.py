from __future__ import annotations

from pathlib import Path
import logging
from threading import Event
from time import sleep

import httpx
import polars as pl

from biominer.bioclip.object_runner import empty_object_score_frame
from biominer.storage.parquet import write_parquet
from biominer.vision.rolling_worker import (
    BatchPlanner,
    BioCLIPWorker,
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
import biominer.vision.rolling_worker as rolling_worker_module
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


def test_rolling_worker_consumes_planned_batches_lazily(monkeypatch) -> None:  # noqa: ANN001
    staged_indices: list[int] = []

    def streaming_plan(_planner, records):  # noqa: ANN001, ANN202
        yield PlannedBatch(0, "vision-batch-000000", "part-000000", records.slice(0, 1))
        assert staged_indices == [0]
        yield PlannedBatch(1, "vision-batch-000001", "part-000001", records.slice(1, 1))

    def image_stage(planned: PlannedBatch) -> ImageBatch:
        staged_indices.append(planned.batch_index)
        return ImageBatch(planned.batch_index, planned.batch_id, planned.part_id, planned.records)

    def detection_stage(batch: ImageBatch) -> DetectionBatch:
        return DetectionBatch(batch, pl.DataFrame({"detection_id": [f"det-{batch.batch_index}"]}))

    def score_input_stage(batch: DetectionBatch) -> ScoreInputBatch:
        return ScoreInputBatch(batch, batch.frame)

    def score_stage(batch: ScoreInputBatch) -> ScoreBatch:
        return ScoreBatch(batch, batch.frame)

    def commit_stage(batch: ScoreBatch) -> CommitResult:
        return CommitResult(batch.score_input_batch.detection_batch.image_batch.batch_id, {})

    monkeypatch.setattr(BatchPlanner, "plan", streaming_plan)
    worker = RollingVisionWorker(
        settings=RollingVisionWorkerSettings(vision_batch_rows=1),
        image_stage=image_stage,
        detection_stage=detection_stage,
        score_input_stage=score_input_stage,
        score_stage=score_stage,
        commit_stage=commit_stage,
    )

    result = worker.run(pl.DataFrame({"flickr_photo_id": ["one", "two"]}))

    assert staged_indices == [0, 1]
    assert result.batches_seen == 2


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


def test_image_stager_preserves_typed_list_columns_beyond_inference_window(tmp_path) -> None:
    records = pl.DataFrame(
        {
            "source": pl.Series(["flickr"] * 101, dtype=pl.String),
            "flickr_photo_id": pl.Series([f"photo-{index:03d}" for index in range(101)], dtype=pl.String),
            "image_url": pl.Series([""] * 101, dtype=pl.String),
            "machine_tags": pl.Series([[] for _ in range(100)] + [["uploaded:by=instagram"]], dtype=pl.List(pl.String)),
        }
    )
    stager = ImageStager(output_dir=tmp_path / "out", cache_root=tmp_path / "cache")
    try:
        batch = stager(PlannedBatch(0, "batch-0", "part-0", records))
    finally:
        stager.close()

    assert batch.records.schema["machine_tags"] == pl.List(pl.String)
    assert batch.records["machine_tags"].tail(1).to_list() == [["uploaded:by=instagram"]]
    assert batch.records.schema["staged_image_path"] == pl.String


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


def test_rolling_worker_logs_stage_progress_and_heartbeat(caplog) -> None:  # noqa: ANN001
    def image_stage(planned: PlannedBatch) -> ImageBatch:
        return ImageBatch(planned.batch_index, planned.batch_id, planned.part_id, planned.records)

    def detection_stage(batch: ImageBatch) -> DetectionBatch:
        sleep(0.04)
        return DetectionBatch(batch, pl.DataFrame({"detection_id": ["det-1"]}))

    def score_input_stage(batch: DetectionBatch) -> ScoreInputBatch:
        return ScoreInputBatch(batch, batch.frame)

    def score_stage(batch: ScoreInputBatch) -> ScoreBatch:
        sleep(0.04)
        return ScoreBatch(batch, batch.frame)

    worker = RollingVisionWorker(
        settings=RollingVisionWorkerSettings(vision_batch_rows=1, heartbeat_interval_seconds=0.01),
        image_stage=image_stage,
        detection_stage=detection_stage,
        score_input_stage=score_input_stage,
        score_stage=score_stage,
        commit_stage=lambda batch: CommitResult(
            batch.score_input_batch.detection_batch.image_batch.batch_id,
            {},
        ),
    )

    with caplog.at_level(logging.INFO, logger="biominer.vision.rolling_worker"):
        worker.run(pl.DataFrame({"flickr_photo_id": ["photo-1"]}))

    messages = [record.getMessage() for record in caplog.records]
    assert any("vision_stage_started stage=yolo_detection" in message for message in messages)
    assert any("vision_heartbeat active=" in message for message in messages)
    assert any("vision_stage_finished stage=bioclip_scoring" in message for message in messages)
    assert any("vision_run_finished" in message for message in messages)


def test_bioclip_worker_dispatches_hierarchical_batches(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    calls: list[tuple[int, object, object]] = []

    def fake_hierarchical(*, items, scorer, path_taxonomy_store, taxonomy_text_embedding_index):  # noqa: ANN001, ANN202
        del scorer
        calls.append((len(items), path_taxonomy_store, taxonomy_text_embedding_index))
        return []

    monkeypatch.setattr(rolling_worker_module, "_score_hierarchical_detection_batch", fake_hierarchical)
    taxonomy_store = object()
    embedding_index = object()
    worker = BioCLIPWorker(
        species_context=object(),  # type: ignore[arg-type]
        candidate_set=object(),  # type: ignore[arg-type]
        scorer=object(),  # type: ignore[arg-type]
        output_dir=tmp_path,
        classification_mode="hierarchical_butterfly_classification",
        path_taxonomy_store=taxonomy_store,  # type: ignore[arg-type]
        taxonomy_text_embedding_index=embedding_index,  # type: ignore[arg-type]
    )
    image_batch = ImageBatch(0, "batch-0", "part-0", pl.DataFrame({"flickr_photo_id": ["photo-1"]}))
    detection_batch = DetectionBatch(image_batch, pl.DataFrame({"detection_id": ["det-1"]}))

    result = worker(ScoreInputBatch(detection_batch, detection_batch.frame, items=({},)))

    assert calls == [(1, taxonomy_store, embedding_index)]
    assert result.frame.is_empty()


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


def test_rolling_worker_emits_throughput_queue_cleanup_and_retry_metrics() -> None:
    records = pl.DataFrame({"source": ["flickr"] * 2, "flickr_photo_id": ["photo-1", "photo-2"]})

    def image_stage(planned: PlannedBatch) -> ImageBatch:
        return ImageBatch(
            batch_index=planned.batch_index,
            batch_id=planned.batch_id,
            part_id=planned.part_id,
            records=planned.records,
        )

    def detection_stage(batch: ImageBatch) -> DetectionBatch:
        return DetectionBatch(
            image_batch=batch,
            frame=pl.DataFrame({"detection_id": ["det-1", "det-2"]}),
            metrics={"detector_batch_retries": 2},
        )

    def score_input_stage(batch: DetectionBatch) -> ScoreInputBatch:
        return ScoreInputBatch(
            detection_batch=batch,
            frame=pl.DataFrame({"detection_id": ["det-1", "det-2"]}),
        )

    def score_stage(batch: ScoreInputBatch) -> ScoreBatch:
        return ScoreBatch(
            score_input_batch=batch,
            frame=pl.DataFrame({"detection_id": ["det-1", "det-2"]}),
            metrics={"bioclip_batch_retries": 3},
        )

    def commit_stage(batch: ScoreBatch) -> CommitResult:
        return CommitResult(
            batch_id=batch.score_input_batch.detection_batch.image_batch.batch_id,
            part_outputs={},
            cleanup_paths_deleted=4,
        )

    worker = RollingVisionWorker(
        settings=RollingVisionWorkerSettings(
            vision_batch_rows=2,
            image_prefetch_batches=1,
            accelerator_concurrency=2,
            bioclip_preprocess_workers=4,
        ),
        image_stage=image_stage,
        detection_stage=detection_stage,
        score_input_stage=score_input_stage,
        score_stage=score_stage,
        commit_stage=commit_stage,
    )

    result = worker.run(records)

    assert result.metrics["images_staged"] == 2
    assert result.metrics["images_detected"] == 2
    assert result.metrics["detection_rows"] == 2
    assert result.metrics["bioclip_score_inputs"] == 2
    assert result.metrics["bioclip_inputs_scored"] == 2
    assert result.metrics["detection_rows_per_image"] == 1.0
    assert result.metrics["bioclip_score_inputs_per_image"] == 1.0
    assert result.metrics["cleanup_paths_deleted"] == 4
    assert result.metrics["detector_batch_retries"] == 2
    assert result.metrics["bioclip_batch_retries"] == 3
    assert result.metrics["adaptive_retry_count"] == 5
    assert result.metrics["accelerator_concurrency"] == 2
    assert result.metrics["bioclip_preprocess_workers"] == 4
    assert result.metrics["cache_resident_batch_count"] <= 1
    assert set(result.metrics["queue_wait_seconds_by_stage"]) == {
        "score_to_commit_batches",
        "staged_image_batches",
        "yolo_to_score_batches",
    }
    assert result.metrics["staged_images_per_sec"] > 0.0
    assert result.metrics["yolo_images_per_sec"] > 0.0
    assert result.metrics["bioclip_inputs_per_sec"] > 0.0


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
                output_path=tmp_path / "detections.parquet",
            ),
            frame=pl.DataFrame([{"detection_id": "det-1"}]),
            output_path=tmp_path / "score-inputs.parquet",
            temp_dir=score_input_dir,
        ),
        frame=empty_object_score_frame(),
        output_path=score_path,
    )
    blocked_output_dir = tmp_path / "rolling-file"
    blocked_output_dir.write_text("not a directory", encoding="utf-8")

    try:
        CommitWorker(output_dir=blocked_output_dir)(score_batch)
    except OSError:
        pass
    else:  # pragma: no cover - defensive assertion.
        raise AssertionError("expected commit failure")

    assert cached_image.exists()
    assert score_input_dir.exists()
