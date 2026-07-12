from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from shutil import rmtree
from threading import Lock
from typing import Any, Protocol

import polars as pl

from biominer.detection.cropper import crop_with_padding
from biominer.detection.detector_base import DecodedImage, ObjectDetector
from biominer.detection.policy import DetectionPolicy, DetectionRunPolicy, detection_is_bioclip_eligible
from biominer.detection.schema import DETECTION_OUTPUT_SCHEMA, build_detection_rows, detection_id_for
from biominer.storage.parquet import write_parquet, write_parquet_batches


ImageLoader = Callable[[dict[str, Any]], DecodedImage]


class ExecutorFactory(Protocol):
    def __call__(self, max_workers: int) -> Any:
        ...


@dataclass(frozen=True)
class DetectionPipelineResult:
    frame: pl.DataFrame
    output_path: Path
    records_seen: int
    images_loaded: int
    image_failures: int
    detections_written: int
    crops_created: int
    parquet_batches_written: int = 0
    adaptive_batching_enabled: bool = False
    detector_batch_retries: int = 0
    detector_batch_size_initial: int = 4
    detector_batch_size_final: int = 4
    detector_batch_size_min: int = 1


@dataclass(frozen=True)
class _LoadedImage:
    record: dict[str, Any]
    image: DecodedImage | None
    failure_reason: str | None = None


@dataclass(frozen=True)
class _CropJob:
    row: dict[str, Any]
    image: DecodedImage


@dataclass
class _AdaptiveDetectorBatchState:
    enabled: bool
    current_batch_size: int
    min_batch_size: int
    retries: int = 0


class _DebugCropWriter:
    def __init__(self, directory: Path, *, limit: int) -> None:
        self.directory = directory
        self.limit = max(0, limit)
        self._written = 0
        self._lock = Lock()

    def write(self, crop_hash: str, *, encoded_bytes: bytes, width: int, height: int) -> bool:
        with self._lock:
            if self._written >= self.limit:
                return False
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{_safe_crop_stem(crop_hash)}.ppm"
            path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + encoded_bytes)
            self._written += 1
            return True


def run_detection_pipeline(
    *,
    records: Iterable[dict[str, Any]],
    detector: ObjectDetector,
    output_path: str | Path,
    image_loader: ImageLoader,
    detection_policy: DetectionPolicy | None = None,
    run_policy: DetectionRunPolicy | None = None,
    executor_factory: ExecutorFactory = ThreadPoolExecutor,
) -> DetectionPipelineResult:
    policy = detection_policy or DetectionPolicy(backend=detector.backend)
    runtime = run_policy or DetectionRunPolicy()
    if runtime.detector_batch_size <= 0:
        raise ValueError("detector_batch_size must be positive")
    if runtime.min_detector_batch_size <= 0:
        raise ValueError("min_detector_batch_size must be positive")
    if runtime.min_detector_batch_size > runtime.detector_batch_size:
        raise ValueError("min_detector_batch_size must be <= detector_batch_size")
    detector_batch_state = _AdaptiveDetectorBatchState(
        enabled=runtime.adaptive_batching,
        current_batch_size=runtime.detector_batch_size,
        min_batch_size=runtime.min_detector_batch_size,
    )
    output_target = Path(output_path)
    batch_dir = _prepare_detection_batch_dir(output_target)
    debug_writer = _prepare_debug_crop_writer(output_target, policy=policy)
    row_buffer: list[dict[str, Any]] = []
    batch_paths: list[Path] = []
    batch: list[_LoadedImage] = []
    records_seen = 0
    images_loaded = 0
    image_failures = 0
    crops_created = 0
    try:
        for loaded in _load_images_bounded(records, image_loader=image_loader, run_policy=runtime, executor_factory=executor_factory):
            records_seen += 1
            if loaded.image is None:
                image_failures += 1
                _buffer_detection_rows(
                    [_image_failure_row(loaded, detector=detector)],
                    row_buffer=row_buffer,
                    batch_paths=batch_paths,
                    batch_dir=batch_dir,
                    parquet_batch_rows=runtime.parquet_batch_rows,
                )
                continue
            loaded = _LoadedImage(record=loaded.record, image=_resize_image_to_max_side(loaded.image, policy.image_max_side_px))
            images_loaded += 1
            batch.append(loaded)
            if len(batch) >= detector_batch_state.current_batch_size:
                enriched = _detect_and_enrich_batch(
                    batch,
                    detector=detector,
                    policy=policy,
                    run_policy=runtime,
                    executor_factory=executor_factory,
                    debug_writer=debug_writer,
                    detector_batch_state=detector_batch_state,
                )
                crops_created += sum(1 for row in enriched if row.get("crop_hash"))
                _buffer_detection_rows(
                    enriched,
                    row_buffer=row_buffer,
                    batch_paths=batch_paths,
                    batch_dir=batch_dir,
                    parquet_batch_rows=runtime.parquet_batch_rows,
                )
                batch = []
        if batch:
            enriched = _detect_and_enrich_batch(
                batch,
                detector=detector,
                policy=policy,
                run_policy=runtime,
                executor_factory=executor_factory,
                debug_writer=debug_writer,
                detector_batch_state=detector_batch_state,
            )
            crops_created += sum(1 for row in enriched if row.get("crop_hash"))
            _buffer_detection_rows(
                enriched,
                row_buffer=row_buffer,
                batch_paths=batch_paths,
                batch_dir=batch_dir,
                parquet_batch_rows=runtime.parquet_batch_rows,
            )
        _flush_detection_row_buffer(row_buffer=row_buffer, batch_paths=batch_paths, batch_dir=batch_dir)
        output = write_parquet_batches(
            (pl.read_parquet(path) for path in batch_paths),
            output_target,
            schema=DETECTION_OUTPUT_SCHEMA,
        )
        frame = pl.read_parquet(output)
        return DetectionPipelineResult(
            frame=frame,
            output_path=output,
            records_seen=records_seen,
            images_loaded=images_loaded,
            image_failures=image_failures,
            detections_written=frame.filter(pl.col("detection_status") == "detected").height
            if frame.height and "detection_status" in frame.columns
            else 0,
            crops_created=crops_created,
            parquet_batches_written=len(batch_paths),
            adaptive_batching_enabled=runtime.adaptive_batching,
            detector_batch_retries=detector_batch_state.retries,
            detector_batch_size_initial=runtime.detector_batch_size,
            detector_batch_size_final=detector_batch_state.current_batch_size,
            detector_batch_size_min=runtime.min_detector_batch_size,
        )
    finally:
        if batch_dir.exists():
            rmtree(batch_dir)


def _prepare_detection_batch_dir(output_path: Path) -> Path:
    batch_dir = output_path.parent / f".{output_path.name}.batches.tmp"
    if batch_dir.exists():
        rmtree(batch_dir)
    return batch_dir


def _prepare_debug_crop_writer(output_path: Path, *, policy: DetectionPolicy) -> _DebugCropWriter | None:
    if not policy.retain_debug_crops:
        return None
    directory = output_path.parent / f"{output_path.stem}_debug_crops"
    if directory.exists():
        rmtree(directory)
    return _DebugCropWriter(directory, limit=policy.debug_crop_limit)


def _buffer_detection_rows(
    rows: list[dict[str, Any]],
    *,
    row_buffer: list[dict[str, Any]],
    batch_paths: list[Path],
    batch_dir: Path,
    parquet_batch_rows: int,
) -> None:
    row_buffer.extend(rows)
    if len(row_buffer) >= max(1, parquet_batch_rows):
        _flush_detection_row_buffer(row_buffer=row_buffer, batch_paths=batch_paths, batch_dir=batch_dir)


def _flush_detection_row_buffer(*, row_buffer: list[dict[str, Any]], batch_paths: list[Path], batch_dir: Path) -> None:
    if not row_buffer:
        return
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_path = batch_dir / f"batch-{len(batch_paths):06d}.parquet"
    # Use the durable schema explicitly. Polars' bounded inference can otherwise
    # infer a Null column from early None values and reject a later real value.
    write_parquet(pl.DataFrame(row_buffer, schema=DETECTION_OUTPUT_SCHEMA), batch_path)
    batch_paths.append(batch_path)
    row_buffer.clear()


def _load_images_bounded(
    records: Iterable[dict[str, Any]],
    *,
    image_loader: ImageLoader,
    run_policy: DetectionRunPolicy,
    executor_factory: ExecutorFactory,
) -> Iterable[_LoadedImage]:
    def load(record: dict[str, Any]) -> _LoadedImage:
        try:
            return _LoadedImage(record=record, image=image_loader(record))
        except Exception as exc:  # noqa: BLE001 - image failures become detection rows.
            return _LoadedImage(record=record, image=None, failure_reason=str(exc))

    with executor_factory(max_workers=run_policy.download_workers) as pool:
        yield from pool.map(load, records, buffersize=run_policy.max_inflight_images)


def _detect_and_enrich_batch(
    batch: list[_LoadedImage],
    *,
    detector: ObjectDetector,
    policy: DetectionPolicy,
    run_policy: DetectionRunPolicy,
    executor_factory: ExecutorFactory,
    debug_writer: _DebugCropWriter | None,
    detector_batch_state: _AdaptiveDetectorBatchState,
) -> list[dict[str, Any]]:
    crop_jobs: list[_CropJob] = []
    for item, detections in _detect_loaded_images_adaptively(batch, detector=detector, state=detector_batch_state):
        image = item.image
        if image is None:
            continue
        detection_rows = build_detection_rows(
            record=item.record,
            image=image,
            detections=detections,
            detector_backend=detector.backend,
            detector_model_id=detector.model_id,
            detector_model_version=detector.model_version,
            detector_checkpoint=detector.checkpoint,
            policy=policy,
        )
        crop_jobs.extend(_CropJob(row=row, image=image) for row in detection_rows)
    return _with_crop_metadata_bounded(
        crop_jobs,
        policy=policy,
        run_policy=run_policy,
        executor_factory=executor_factory,
        debug_writer=debug_writer,
    )


def _detect_loaded_images_adaptively(
    batch: list[_LoadedImage],
    *,
    detector: ObjectDetector,
    state: _AdaptiveDetectorBatchState,
) -> list[tuple[_LoadedImage, list[Any]]]:
    pending = [batch]
    detected: list[tuple[_LoadedImage, list[Any]]] = []
    while pending:
        chunk = pending.pop(0)
        images = [item.image for item in chunk if item.image is not None]
        try:
            detections_by_image = detector.detect_batch(images)
        except RuntimeError as exc:
            if not _should_retry_detector_batch(
                exc,
                adaptive_batching=state.enabled,
                batch_size=len(chunk),
                current_batch_size=state.current_batch_size,
                min_batch_size=state.min_batch_size,
            ):
                raise
            state.current_batch_size = _next_detector_batch_size(
                current_batch_size=state.current_batch_size,
                failed_batch_size=len(chunk),
                min_batch_size=state.min_batch_size,
            )
            state.retries += 1
            pending = _chunks(chunk, state.current_batch_size) + pending
            continue
        if len(detections_by_image) != len(images):
            raise ValueError(f"detector returned {len(detections_by_image)} result rows for {len(images)} images")
        detected.extend(zip(chunk, detections_by_image, strict=True))
    return detected


def _with_crop_metadata_bounded(
    jobs: list[_CropJob],
    *,
    policy: DetectionPolicy,
    run_policy: DetectionRunPolicy,
    executor_factory: ExecutorFactory,
    debug_writer: _DebugCropWriter | None,
) -> list[dict[str, Any]]:
    if not jobs:
        return []

    def enrich(job: _CropJob) -> dict[str, Any]:
        return _with_crop_metadata(job.row, image=job.image, policy=policy, debug_writer=debug_writer)

    rows: list[dict[str, Any]] = []
    with executor_factory(max_workers=run_policy.decode_workers) as pool:
        for chunk in _chunks(jobs, max(1, run_policy.crop_batch_size)):
            rows.extend(pool.map(enrich, chunk, buffersize=run_policy.max_inflight_crops))
    return rows


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [items[start : start + size] for start in range(0, len(items), size)]


def _should_retry_detector_batch(
    exc: RuntimeError,
    *,
    adaptive_batching: bool,
    batch_size: int,
    current_batch_size: int,
    min_batch_size: int,
) -> bool:
    return (
        adaptive_batching
        and batch_size > min_batch_size
        and current_batch_size > min_batch_size
        and is_detector_memory_error(exc)
    )


def _next_detector_batch_size(
    *,
    current_batch_size: int,
    failed_batch_size: int,
    min_batch_size: int,
) -> int:
    if current_batch_size <= min_batch_size or failed_batch_size <= min_batch_size:
        return min_batch_size
    return max(min_batch_size, min(current_batch_size // 2, failed_batch_size // 2))


def is_detector_memory_error(exc: BaseException) -> bool:
    if not isinstance(exc, RuntimeError):
        return False
    message = " ".join(str(exc).casefold().split())
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cuda out of memory",
            "mps memory",
            "allocation failed",
        )
    )


def _with_crop_metadata(
    row: dict[str, Any],
    *,
    image: DecodedImage,
    policy: DetectionPolicy,
    debug_writer: _DebugCropWriter | None,
) -> dict[str, Any]:
    if row.get("detection_status") != "detected":
        return row
    if not detection_is_bioclip_eligible(row, policy) and debug_writer is None:
        return row
    bbox = row.get("bbox_xyxy")
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        return row
    crop = crop_with_padding(
        image,
        bbox_xyxy=tuple(float(value) for value in bbox),  # type: ignore[arg-type]
        padding_ratio=policy.crop_padding_ratio,
        target_px=policy.crop_target_px,
    )
    storage_policy = crop.storage_policy
    if debug_writer is not None and debug_writer.write(
        crop.crop_hash,
        encoded_bytes=crop.encoded_bytes,
        width=crop.crop_width,
        height=crop.crop_height,
    ):
        storage_policy = "debug_retained"
    return {
        **row,
        "crop_padding_ratio": policy.crop_padding_ratio,
        "crop_hash": crop.crop_hash,
        "crop_width": crop.crop_width,
        "crop_height": crop.crop_height,
        "crop_storage_policy": storage_policy,
    }


def _safe_crop_stem(crop_hash: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in crop_hash).strip("_") or "crop"


def _resize_image_to_max_side(image: DecodedImage, max_side_px: int) -> DecodedImage:
    if max_side_px <= 0:
        return image
    current_max_side = max(image.width, image.height)
    if current_max_side <= max_side_px:
        return image
    scale = max_side_px / current_max_side
    target_width = max(1, round(image.width * scale))
    target_height = max(1, round(image.height * scale))
    try:
        from PIL import Image
    except ImportError:
        return _resize_rgb_nearest(
            image.data,
            src_width=image.width,
            src_height=image.height,
            dst_width=target_width,
            dst_height=target_height,
        )
    return DecodedImage(
        width=target_width,
        height=target_height,
        mode=image.mode,
        data=_resize_rgb_lanczos(
            image.data,
            src_width=image.width,
            src_height=image.height,
            dst_width=target_width,
            dst_height=target_height,
            image_factory=Image.frombytes,
            resample=Image.Resampling.LANCZOS,
        ),
        source_uri=image.source_uri,
    )


def _resize_rgb_lanczos(
    data: bytes,
    *,
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
    image_factory,
    resample,
) -> bytes:
    try:
        image = image_factory("RGB", (src_width, src_height), data)
        resized = image.resize((dst_width, dst_height), resample=resample)
        return resized.tobytes()
    except Exception:
        return _resize_rgb_nearest(
            data,
            src_width=src_width,
            src_height=src_height,
            dst_width=dst_width,
            dst_height=dst_height,
        )


def _resize_rgb_nearest(
    data: bytes,
    *,
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> bytes:
    output = bytearray(dst_width * dst_height * 3)
    for y in range(dst_height):
        source_y = min(src_height - 1, int((y + 0.5) * src_height / dst_height))
        for x in range(dst_width):
            source_x = min(src_width - 1, int((x + 0.5) * src_width / dst_width))
            source_offset = (source_y * src_width + source_x) * 3
            target_offset = (y * dst_width + x) * 3
            output[target_offset : target_offset + 3] = data[source_offset : source_offset + 3]
    return bytes(output)


def _image_failure_row(item: _LoadedImage, *, detector: ObjectDetector) -> dict[str, Any]:
    source = str(item.record.get("source") or "flickr")
    photo_id = str(item.record.get("flickr_photo_id") or item.record.get("id") or "")
    if not source or not photo_id:
        raise ValueError("Detection rows require source and flickr_photo_id")
    return {
        "source": source,
        "flickr_photo_id": photo_id,
        "source_record_hash": item.record.get("source_record_hash"),
        "image_url": str(item.record.get("image_url") or ""),
        "photo_page_url": item.record.get("photo_page_url"),
        "detection_id": detection_id_for(
            source=source,
            flickr_photo_id=photo_id,
            detector_checkpoint=detector.checkpoint,
            bbox_xyxyn=(None, None, None, None),
            detector_label="failed_image_load",
        ),
        "detector_backend": detector.backend,
        "prediction_source": f"object_detector:{detector.backend}",
        "detector_model_id": detector.model_id,
        "detector_model_version": detector.model_version,
        "detector_checkpoint": detector.checkpoint,
        "detected_at": datetime.now(UTC).isoformat(),
        "bbox_xyxy": [],
        "bbox_xyxyn": [],
        "bbox_xywhn": [],
        "box_area_ratio": 0.0,
        "detector_label": None,
        "detector_score": 0.0,
        "objectness_score": None,
        "nms_group_id": None,
        "crop_padding_ratio": 0.0,
        "crop_hash": None,
        "crop_width": None,
        "crop_height": None,
        "crop_storage_policy": "not_created",
        "detection_status": "failed_image_load",
        "failure_reason": item.failure_reason or "image_load_failed",
        "schema_version": "object-detection-v1",
    }
