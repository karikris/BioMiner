from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from queue import Queue
from threading import BoundedSemaphore, Event, Lock, Thread
from typing import Any
from urllib.parse import urlparse

import httpx
import polars as pl

from biominer.bioclip.image_cache import cache_image_from_url
from biominer.storage.parquet import write_parquet


@dataclass(frozen=True)
class RollingVisionWorkerSettings:
    vision_batch_rows: int = 500
    image_prefetch_batches: int = 4
    target_ready_yolo_batches: int = 3
    yolo_to_score_batches: int = 2
    score_to_commit_batches: int = 2
    accelerator_concurrency: int = 1
    bioclip_preprocess_workers: int = 1
    bioclip_gate_mode: str = "exclude_hard_negative"
    score_no_detection_whole_image: bool = True


@dataclass(frozen=True)
class ImageBatch:
    batch_index: int
    batch_id: str
    part_id: str
    records: pl.DataFrame
    image_batch_manifest: Path | None = None
    cached_image_paths: tuple[Path, ...] = ()
    failed_image_records: tuple[dict[str, Any], ...] = ()
    started_at: str | None = None
    ended_at: str | None = None


@dataclass(frozen=True)
class DetectionBatch:
    image_batch: ImageBatch
    frame: pl.DataFrame
    output_path: Path | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreInputBatch:
    detection_batch: DetectionBatch
    frame: pl.DataFrame
    items: tuple[dict[str, Any], ...] = ()
    temp_dir: Path | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreBatch:
    score_input_batch: ScoreInputBatch
    frame: pl.DataFrame
    output_path: Path | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommitResult:
    batch_id: str
    part_outputs: dict[str, str]
    cleanup_paths_deleted: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RollingVisionWorkerResult:
    started_at: str
    ended_at: str
    status: str
    batches_seen: int
    batches_committed: int
    part_outputs: tuple[dict[str, str], ...]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class PlannedBatch:
    batch_index: int
    batch_id: str
    part_id: str
    records: pl.DataFrame


ImageStage = Callable[[PlannedBatch], ImageBatch]
DetectionStage = Callable[[ImageBatch], DetectionBatch]
ScoreInputStage = Callable[[DetectionBatch], ScoreInputBatch]
ScoreStage = Callable[[ScoreInputBatch], ScoreBatch]
CommitStage = Callable[[ScoreBatch], CommitResult]


class BatchPlanner:
    def __init__(self, *, batch_rows: int = 500) -> None:
        if batch_rows <= 0:
            raise ValueError("batch_rows must be positive")
        self.batch_rows = int(batch_rows)

    def plan(self, records: pl.DataFrame) -> Iterable[PlannedBatch]:
        for batch_index, offset in enumerate(range(0, records.height, self.batch_rows)):
            yield PlannedBatch(
                batch_index=batch_index,
                batch_id=f"vision-batch-{batch_index:06d}",
                part_id=f"part-{batch_index:06d}",
                records=records.slice(offset, self.batch_rows),
            )


class ImageStager:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        cache_root: str | Path = "data/cache/images",
        http_client: httpx.Client | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.cache_root = Path(cache_root)
        self.manifest_dir = self.output_dir / "image_batch_manifest"
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=30)

    def __call__(self, planned: PlannedBatch) -> ImageBatch:
        started_at = datetime.now(UTC).isoformat()
        rows: list[dict[str, Any]] = []
        staged_records: list[dict[str, Any]] = []
        cached_paths: list[Path] = []
        failed: list[dict[str, Any]] = []
        for record in planned.records.to_dicts():
            staged = dict(record)
            image_url = str(record.get("image_url") or record.get("image_url_used") or "")
            row = {
                "source": str(record.get("source") or "flickr"),
                "flickr_photo_id": str(record.get("flickr_photo_id") or record.get("id") or ""),
                "image_url": image_url,
                "image_cache_path": None,
                "image_hash": None,
                "image_cache_status": "skipped",
                "failure_reason": None,
                "batch_id": planned.batch_id,
                "part_id": planned.part_id,
            }
            if _is_http_url(image_url):
                try:
                    cached = cache_image_from_url(image_url, cache_root=self.cache_root, http_client=self._client)
                except Exception as exc:  # noqa: BLE001 - detection stage keeps retryable failure evidence.
                    row["image_cache_status"] = "failed"
                    row["failure_reason"] = str(exc)
                    failed.append({**staged, "image_cache_failure_reason": str(exc)})
                else:
                    row["image_cache_path"] = str(cached.path)
                    row["image_hash"] = cached.image_hash
                    row["image_cache_status"] = "cached"
                    staged["staged_image_path"] = str(cached.path)
                    cached_paths.append(cached.path)
            rows.append(row)
            staged_records.append(staged)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = write_parquet(pl.DataFrame(rows), self.manifest_dir / f"{planned.part_id}.parquet")
        ended_at = datetime.now(UTC).isoformat()
        return ImageBatch(
            batch_index=planned.batch_index,
            batch_id=planned.batch_id,
            part_id=planned.part_id,
            records=pl.DataFrame(staged_records),
            image_batch_manifest=manifest_path,
            cached_image_paths=tuple(cached_paths),
            failed_image_records=tuple(failed),
            started_at=started_at,
            ended_at=ended_at,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


class RollingVisionWorker:
    def __init__(
        self,
        *,
        settings: RollingVisionWorkerSettings | None = None,
        image_stage: ImageStage | None = None,
        detection_stage: DetectionStage | None = None,
        score_input_stage: ScoreInputStage | None = None,
        score_stage: ScoreStage | None = None,
        commit_stage: CommitStage | None = None,
    ) -> None:
        self.settings = settings or RollingVisionWorkerSettings()
        _validate_settings(self.settings)
        self._image_stage = image_stage or _default_image_stage
        self._detection_stage = detection_stage or _missing_stage("detection_stage")
        self._score_input_stage = score_input_stage or _missing_stage("score_input_stage")
        self._score_stage = score_stage or _missing_stage("score_stage")
        self._commit_stage = commit_stage or _missing_stage("commit_stage")
        self.max_resident_image_batches = 0

    def run(self, records: pl.DataFrame) -> RollingVisionWorkerResult:
        started_at = datetime.now(UTC).isoformat()
        planner = BatchPlanner(batch_rows=self.settings.vision_batch_rows)
        planned_batches = list(planner.plan(records))
        staged_image_batches: Queue[ImageBatch | object] = Queue(maxsize=self.settings.image_prefetch_batches)
        yolo_to_score_batches: Queue[DetectionBatch | object] = Queue(maxsize=self.settings.yolo_to_score_batches)
        score_to_commit_batches: Queue[ScoreBatch | object] = Queue(maxsize=self.settings.score_to_commit_batches)
        stop = Event()
        first_error: list[BaseException] = []
        resident = _ResidentBatchCounter()
        image_slots = BoundedSemaphore(self.settings.image_prefetch_batches)
        committed: list[CommitResult] = []
        sentinel = object()

        def remember_error(exc: BaseException) -> None:
            if not first_error:
                first_error.append(exc)
            stop.set()

        def image_producer() -> None:
            try:
                for planned in planned_batches:
                    if stop.is_set():
                        break
                    image_slots.acquire()
                    resident.increment()
                    self.max_resident_image_batches = max(self.max_resident_image_batches, resident.value)
                    staged = self._image_stage(planned)
                    staged_image_batches.put(staged)
            except BaseException as exc:  # noqa: BLE001 - propagate across worker boundary.
                remember_error(exc)
            finally:
                staged_image_batches.put(sentinel)

        def yolo_worker() -> None:
            try:
                while True:
                    item = staged_image_batches.get()
                    if item is sentinel:
                        break
                    yolo_to_score_batches.put(self._detection_stage(item))  # type: ignore[arg-type]
            except BaseException as exc:  # noqa: BLE001
                remember_error(exc)
            finally:
                yolo_to_score_batches.put(sentinel)

        def score_worker() -> None:
            try:
                while True:
                    item = yolo_to_score_batches.get()
                    if item is sentinel:
                        break
                    score_inputs = self._score_input_stage(item)  # type: ignore[arg-type]
                    resident.decrement()
                    image_slots.release()
                    score_to_commit_batches.put(self._score_stage(score_inputs))
            except BaseException as exc:  # noqa: BLE001
                remember_error(exc)
            finally:
                score_to_commit_batches.put(sentinel)

        def commit_worker() -> None:
            try:
                while True:
                    item = score_to_commit_batches.get()
                    if item is sentinel:
                        break
                    committed.append(self._commit_stage(item))  # type: ignore[arg-type]
            except BaseException as exc:  # noqa: BLE001
                remember_error(exc)

        threads = [
            Thread(target=image_producer, name="rolling-image-stager"),
            Thread(target=yolo_worker, name="rolling-yolo-worker"),
            Thread(target=score_worker, name="rolling-bioclip-worker"),
            Thread(target=commit_worker, name="rolling-commit-worker"),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if first_error:
            raise first_error[0]
        ended_at = datetime.now(UTC).isoformat()
        return RollingVisionWorkerResult(
            started_at=started_at,
            ended_at=ended_at,
            status="complete",
            batches_seen=len(planned_batches),
            batches_committed=len(committed),
            part_outputs=tuple(result.part_outputs for result in committed),
            metrics={
                "vision_batch_rows": self.settings.vision_batch_rows,
                "image_prefetch_batches": self.settings.image_prefetch_batches,
                "max_resident_image_batches": self.max_resident_image_batches,
                "yolo_to_score_queue_maxsize": yolo_to_score_batches.maxsize,
                "score_to_commit_queue_maxsize": score_to_commit_batches.maxsize,
            },
        )


class _ResidentBatchCounter:
    def __init__(self) -> None:
        self._value = 0
        self._lock = Lock()

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    def increment(self) -> None:
        with self._lock:
            self._value += 1

    def decrement(self) -> None:
        with self._lock:
            self._value = max(0, self._value - 1)


def _validate_settings(settings: RollingVisionWorkerSettings) -> None:
    for name in (
        "vision_batch_rows",
        "image_prefetch_batches",
        "target_ready_yolo_batches",
        "yolo_to_score_batches",
        "score_to_commit_batches",
        "accelerator_concurrency",
        "bioclip_preprocess_workers",
    ):
        if int(getattr(settings, name)) <= 0:
            raise ValueError(f"{name} must be positive")


def _default_image_stage(planned: PlannedBatch) -> ImageBatch:
    now = datetime.now(UTC).isoformat()
    return ImageBatch(
        batch_index=planned.batch_index,
        batch_id=planned.batch_id,
        part_id=planned.part_id,
        records=planned.records,
        started_at=now,
        ended_at=now,
    )


def _missing_stage(name: str) -> Any:
    def missing(*_args: Any, **_kwargs: Any) -> Any:
        raise NotImplementedError(f"{name} is required for rolling vision worker execution")

    return missing


def _is_http_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


__all__ = [
    "BatchPlanner",
    "CommitResult",
    "DetectionBatch",
    "ImageBatch",
    "ImageStager",
    "PlannedBatch",
    "RollingVisionWorker",
    "RollingVisionWorkerResult",
    "RollingVisionWorkerSettings",
    "ScoreBatch",
    "ScoreInputBatch",
]
