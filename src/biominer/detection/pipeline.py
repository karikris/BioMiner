from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from biominer.detection.cropper import crop_with_padding
from biominer.detection.detector_base import DecodedImage, ObjectDetector
from biominer.detection.policy import DetectionPolicy, DetectionRunPolicy
from biominer.detection.schema import build_detection_rows, detection_id_for
from biominer.storage.parquet import write_parquet


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


@dataclass(frozen=True)
class _LoadedImage:
    record: dict[str, Any]
    image: DecodedImage | None
    failure_reason: str | None = None


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
    rows: list[dict[str, Any]] = []
    batch: list[_LoadedImage] = []
    records_seen = 0
    images_loaded = 0
    image_failures = 0
    crops_created = 0
    for loaded in _load_images_bounded(records, image_loader=image_loader, run_policy=runtime, executor_factory=executor_factory):
        records_seen += 1
        if loaded.image is None:
            image_failures += 1
            rows.append(_image_failure_row(loaded, detector=detector))
            continue
        images_loaded += 1
        batch.append(loaded)
        if len(batch) >= runtime.detector_batch_size:
            enriched = _detect_and_enrich_batch(batch, detector=detector, policy=policy)
            crops_created += sum(1 for row in enriched if row.get("crop_hash"))
            rows.extend(enriched)
            batch = []
    if batch:
        enriched = _detect_and_enrich_batch(batch, detector=detector, policy=policy)
        crops_created += sum(1 for row in enriched if row.get("crop_hash"))
        rows.extend(enriched)
    frame = pl.DataFrame(rows) if rows else pl.DataFrame()
    output = write_parquet(frame, output_path)
    return DetectionPipelineResult(
        frame=frame,
        output_path=output,
        records_seen=records_seen,
        images_loaded=images_loaded,
        image_failures=image_failures,
        detections_written=frame.filter(pl.col("detection_status") == "detected").height if frame.height and "detection_status" in frame.columns else 0,
        crops_created=crops_created,
    )


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
) -> list[dict[str, Any]]:
    detections_by_image = detector.detect_batch([item.image for item in batch if item.image is not None])
    rows: list[dict[str, Any]] = []
    for item, detections in zip(batch, detections_by_image, strict=True):
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
        rows.extend(_with_crop_metadata(row, image=image, policy=policy) for row in detection_rows)
    return rows


def _with_crop_metadata(row: dict[str, Any], *, image: DecodedImage, policy: DetectionPolicy) -> dict[str, Any]:
    if row.get("detection_status") != "detected":
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
    return {
        **row,
        "crop_padding_ratio": policy.crop_padding_ratio,
        "crop_hash": crop.crop_hash,
        "crop_width": crop.crop_width,
        "crop_height": crop.crop_height,
        "crop_storage_policy": crop.storage_policy,
    }


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
        "detector_model_id": detector.model_id,
        "detector_model_version": detector.model_version,
        "detector_checkpoint": detector.checkpoint,
        "detected_at": None,
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
