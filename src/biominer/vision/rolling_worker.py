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
from biominer.bioclip.candidate_sets import CandidateSet
from biominer.bioclip.object_runner import (
    OBJECT_SCORE_OUTPUT_SCHEMA,
    ObjectBioClipScorer,
    _ensure_columns,
    _score_detection_batch,
    is_bioclip_memory_error,
    write_object_evidence_outputs,
)
from biominer.detection.detector_base import DecodedImage, ObjectDetector
from biominer.detection.image_io import load_decoded_image_from_record
from biominer.detection.pipeline import DetectionPipelineResult, run_detection_pipeline
from biominer.detection.policy import DetectionPolicy, DetectionRunPolicy
from biominer.species.context import SpeciesContext
from biominer.storage.parquet import write_parquet
from biominer.vision.gates import BioClipGatePolicy
from biominer.vision.score_inputs import materialize_bioclip_score_inputs


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
    output_path: Path | None = None
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


class YOLOWorker:
    def __init__(
        self,
        *,
        detector: ObjectDetector,
        output_dir: str | Path,
        image_loader: Callable[[dict[str, Any]], DecodedImage] | None = None,
        detection_policy: DetectionPolicy | None = None,
        run_policy: DetectionRunPolicy | None = None,
    ) -> None:
        self.detector = detector
        self.detection_dir = Path(output_dir) / "object_detections"
        self.image_loader = image_loader or load_staged_or_cached_image
        self.detection_policy = detection_policy
        self.run_policy = run_policy

    def __call__(self, batch: ImageBatch) -> DetectionBatch:
        self.detection_dir.mkdir(parents=True, exist_ok=True)
        result: DetectionPipelineResult = run_detection_pipeline(
            records=batch.records.to_dicts(),
            detector=self.detector,
            output_path=self.detection_dir / f"{batch.part_id}.parquet",
            image_loader=self.image_loader,
            detection_policy=self.detection_policy,
            run_policy=self.run_policy,
        )
        return DetectionBatch(
            image_batch=batch,
            frame=result.frame,
            output_path=result.output_path,
            metrics={
                "records_seen": result.records_seen,
                "images_loaded": result.images_loaded,
                "image_failures": result.image_failures,
                "detections_written": result.detections_written,
                "crops_created": result.crops_created,
                "detector_batch_retries": result.detector_batch_retries,
                "detector_batch_size_final": result.detector_batch_size_final,
            },
        )


class ScoreInputMaterializer:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        image_loader: Callable[[dict[str, Any]], DecodedImage] | None = None,
        gate_policy: BioClipGatePolicy | None = None,
        crop_padding_ratio: float = 0.08,
        crop_target_px: int = 336,
    ) -> None:
        self.score_input_dir = Path(output_dir) / "bioclip_score_inputs"
        self.temp_root = Path(output_dir) / "bioclip_score_input_files"
        self.image_loader = image_loader or load_staged_or_cached_image
        self.gate_policy = gate_policy or BioClipGatePolicy()
        self.crop_padding_ratio = crop_padding_ratio
        self.crop_target_px = crop_target_px

    def __call__(self, batch: DetectionBatch) -> ScoreInputBatch:
        self.score_input_dir.mkdir(parents=True, exist_ok=True)
        materialized = materialize_bioclip_score_inputs(
            canonical_records=batch.image_batch.records,
            detections=batch.frame,
            image_loader=self.image_loader,
            temp_dir=self.temp_root,
            gate_policy=self.gate_policy,
            crop_padding_ratio=self.crop_padding_ratio,
            crop_target_px=self.crop_target_px,
            batch_id=batch.image_batch.batch_id,
            part_id=batch.image_batch.part_id,
        )
        output_path = write_parquet(materialized.frame, self.score_input_dir / f"{batch.image_batch.part_id}.parquet")
        return ScoreInputBatch(
            detection_batch=batch,
            frame=materialized.frame,
            items=tuple(materialized.items),
            output_path=output_path,
            temp_dir=materialized.temp_dir,
            metrics={"score_inputs": materialized.frame.height},
        )


class BioCLIPWorker:
    def __init__(
        self,
        *,
        species_context: SpeciesContext,
        candidate_set: CandidateSet,
        scorer: ObjectBioClipScorer,
        output_dir: str | Path,
        bioclip_batch_size: int = 24,
        adaptive_batching: bool = False,
        min_bioclip_batch_size: int = 1,
    ) -> None:
        if bioclip_batch_size <= 0:
            raise ValueError("bioclip_batch_size must be positive")
        if min_bioclip_batch_size <= 0:
            raise ValueError("min_bioclip_batch_size must be positive")
        if min_bioclip_batch_size > bioclip_batch_size:
            raise ValueError("min_bioclip_batch_size must be <= bioclip_batch_size")
        self.species_context = species_context
        self.candidate_set = candidate_set
        self.scorer = scorer
        self.score_dir = Path(output_dir) / "object_bioclip_scores"
        self.bioclip_batch_size = bioclip_batch_size
        self.adaptive_batching = adaptive_batching
        self.min_bioclip_batch_size = min_bioclip_batch_size

    def __call__(self, batch: ScoreInputBatch) -> ScoreBatch:
        self.score_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        pending = _chunks(list(batch.items), self.bioclip_batch_size)
        current_batch_size = self.bioclip_batch_size
        retries = 0
        while pending:
            chunk = pending.pop(0)
            try:
                rows.extend(
                    _score_detection_batch(
                        items=chunk,
                        context=self.species_context,
                        candidate_set=self.candidate_set,
                        scorer=self.scorer,
                        ablation_mode="detector_crop",
                    )
                )
            except RuntimeError as exc:
                if (
                    not self.adaptive_batching
                    or len(chunk) <= self.min_bioclip_batch_size
                    or not is_bioclip_memory_error(exc)
                ):
                    raise
                current_batch_size = max(self.min_bioclip_batch_size, min(current_batch_size // 2, len(chunk) // 2))
                retries += 1
                pending = _chunks(chunk, current_batch_size) + pending
        frame = _ensure_columns(pl.DataFrame(rows), OBJECT_SCORE_OUTPUT_SCHEMA) if rows else pl.DataFrame(schema=OBJECT_SCORE_OUTPUT_SCHEMA)
        output_path = write_parquet(frame, self.score_dir / f"{batch.detection_batch.image_batch.part_id}.parquet")
        return ScoreBatch(
            score_input_batch=batch,
            frame=frame,
            output_path=output_path,
            metrics={
                "crops_scored": frame.height,
                "bioclip_batch_retries": retries,
                "bioclip_batch_size_final": current_batch_size,
            },
        )


class CommitWorker:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        species_context: SpeciesContext | None = None,
        delete_images_after_commit: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.canonical_dir = self.output_dir / "canonical_source_records"
        self.detection_dir = self.output_dir / "object_detections"
        self.score_input_dir = self.output_dir / "bioclip_score_inputs"
        self.score_dir = self.output_dir / "object_bioclip_scores"
        self.joined_dir = self.output_dir / "object_evidence_joined"
        self.summary_dir = self.output_dir / "photo_evidence_summary"
        self.species_context = species_context
        self.delete_images_after_commit = delete_images_after_commit

    def __call__(self, batch: ScoreBatch) -> CommitResult:
        image_batch = batch.score_input_batch.detection_batch.image_batch
        self.canonical_dir.mkdir(parents=True, exist_ok=True)
        self.detection_dir.mkdir(parents=True, exist_ok=True)
        self.score_input_dir.mkdir(parents=True, exist_ok=True)
        self.score_dir.mkdir(parents=True, exist_ok=True)
        self.joined_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        canonical_path = write_parquet(image_batch.records, self.canonical_dir / f"{image_batch.part_id}.parquet")
        detection_path = write_parquet(
            batch.score_input_batch.detection_batch.frame,
            self.detection_dir / f"{image_batch.part_id}.parquet",
        )
        score_input_path = write_parquet(batch.score_input_batch.frame, self.score_input_dir / f"{image_batch.part_id}.parquet")
        score_path = write_parquet(batch.frame, self.score_dir / f"{image_batch.part_id}.parquet")
        evidence_outputs = write_object_evidence_outputs(
            canonical_records_path=canonical_path,
            detections_path=detection_path,
            scores_path=score_path,
            joined_output_path=self.joined_dir / f"{image_batch.part_id}.parquet",
            photo_summary_output_path=self.summary_dir / f"{image_batch.part_id}.parquet",
            species_context=self.species_context,
        )
        cleanup_deleted = 0
        if self.delete_images_after_commit:
            cleanup_deleted += _delete_paths(image_batch.cached_image_paths)
        if batch.score_input_batch.temp_dir is not None and batch.score_input_batch.temp_dir.exists():
            cleanup_deleted += _delete_tree(batch.score_input_batch.temp_dir)
        return CommitResult(
            batch_id=image_batch.batch_id,
            part_outputs={
                "image_batch_manifest": str(image_batch.image_batch_manifest) if image_batch.image_batch_manifest else "",
                "canonical_source_records": str(canonical_path),
                "object_detections": str(detection_path),
                "bioclip_score_inputs": str(score_input_path),
                "object_bioclip_scores": str(score_path),
                "object_evidence_joined": str(evidence_outputs.object_evidence_joined),
                "photo_evidence_summary": str(evidence_outputs.photo_evidence_summary),
            },
            cleanup_paths_deleted=cleanup_deleted,
            metrics={
                **batch.score_input_batch.detection_batch.metrics,
                **batch.score_input_batch.metrics,
                **batch.metrics,
                "cleanup_paths_deleted": cleanup_deleted,
            },
        )


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


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [items[index : index + size] for index in range(0, len(items), size)]


def load_staged_or_cached_image(record: dict[str, Any]) -> DecodedImage:
    staged_path = str(record.get("staged_image_path") or "")
    if staged_path:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional dependency path.
            raise RuntimeError("Pillow is required to decode staged images") from exc
        with Image.open(staged_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            return DecodedImage(width=int(width), height=int(height), mode="RGB", data=rgb.tobytes(), source_uri=staged_path)
    return load_decoded_image_from_record(record)


def _delete_paths(paths: tuple[Path, ...]) -> int:
    deleted = 0
    for path in sorted(paths):
        if path.exists():
            path.unlink()
            deleted += 1
    return deleted


def _delete_tree(path: Path) -> int:
    if not path.exists():
        return 0
    files = [child for child in path.rglob("*") if child.is_file()]
    for child in files:
        child.unlink()
    directories = [child for child in path.rglob("*") if child.is_dir()]
    for child in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        child.rmdir()
    path.rmdir()
    return len(files)


__all__ = [
    "BatchPlanner",
    "BioCLIPWorker",
    "CommitResult",
    "CommitWorker",
    "DetectionBatch",
    "ImageBatch",
    "ImageStager",
    "PlannedBatch",
    "RollingVisionWorker",
    "RollingVisionWorkerResult",
    "RollingVisionWorkerSettings",
    "ScoreBatch",
    "ScoreInputBatch",
    "ScoreInputMaterializer",
    "YOLOWorker",
    "load_staged_or_cached_image",
]
