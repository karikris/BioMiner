from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import polars as pl

from biominer.detection.detector_base import DecodedImage, ObjectDetector
from biominer.detection.pipeline import ImageLoader, _image_failure_row, _resize_image_to_max_side, _with_crop_metadata
from biominer.detection.policy import DetectionPolicy
from biominer.detection.schema import build_detection_rows, empty_detection_frame
from biominer.storage.cloud import CloudStorage
from biominer.workstore.base import WorkStore


@dataclass(frozen=True)
class DetectionWorkPlanResult:
    source_shards_seen: int
    source_records_seen: int
    enqueued_work_items: int
    duplicate_work_items: int


@dataclass(frozen=True)
class CloudDetectionBatchResult:
    frame: pl.DataFrame
    records_seen: int
    images_loaded: int
    image_failures: int
    detections_written: int
    crops_created: int


def enqueue_detection_work_from_source_shards(
    *,
    storage: CloudStorage,
    workstore: WorkStore,
    job_name: str,
    registry_version: str | None,
    run_id: str,
    source_stage: str,
    detection_stage: str,
    detector_backend: str,
    detector_model_id: str,
    detector_model_version: str,
    detector_checkpoint: str,
    limit: int | None = None,
) -> DetectionWorkPlanResult:
    shards = workstore.list_candidate_shards(
        job_name=job_name,
        stage=source_stage,
        registry_version=registry_version,
        run_id=run_id,
    )
    items: list[dict[str, Any]] = []
    records_seen = 0
    remaining = None if limit is None or limit <= 0 else int(limit)
    detector = {
        "backend": detector_backend,
        "model_id": detector_model_id,
        "model_version": detector_model_version,
        "checkpoint": detector_checkpoint,
    }
    for shard in shards:
        if remaining == 0:
            break
        source_shard_uri = str(shard["uri"])
        frame = storage.read_parquet(source_shard_uri)
        for row in frame.iter_rows(named=True):
            if remaining == 0:
                break
            record = dict(row)
            if not _record_is_detectable(record):
                continue
            records_seen += 1
            items.append(
                detection_work_item(
                    record,
                    run_id=run_id,
                    source_shard_uri=source_shard_uri,
                    detector=detector,
                )
            )
            if remaining is not None:
                remaining -= 1
    inserted = workstore.enqueue_work(job_name, registry_version, items, stage=detection_stage) if items else 0
    return DetectionWorkPlanResult(
        source_shards_seen=len(shards),
        source_records_seen=records_seen,
        enqueued_work_items=inserted,
        duplicate_work_items=len(items) - inserted,
    )


def run_cloud_detection_batch(
    *,
    work_items: list[dict[str, Any]],
    detector: ObjectDetector,
    image_loader: ImageLoader,
    detection_policy: DetectionPolicy | None = None,
    detector_batch_size: int = 16,
) -> CloudDetectionBatchResult:
    if detector_batch_size <= 0:
        raise ValueError("detector_batch_size must be positive")
    policy = detection_policy or DetectionPolicy(backend=detector.backend)
    rows: list[dict[str, Any]] = []
    records_seen = 0
    images_loaded = 0
    image_failures = 0
    crops_created = 0
    loaded: list[tuple[dict[str, Any], DecodedImage]] = []
    for item in work_items:
        record = source_record_from_detection_work_item(item)
        records_seen += 1
        try:
            image = _resize_image_to_max_side(image_loader(record), policy.image_max_side_px)
        except Exception as exc:  # noqa: BLE001 - image failures become durable detection rows.
            image_failures += 1
            rows.append(_image_failure_row(_LoadedFailure(record=record, failure_reason=str(exc)), detector=detector))
            continue
        images_loaded += 1
        loaded.append((record, image))
    for loaded_batch in _chunks(loaded, detector_batch_size):
        detections_by_image = detector.detect_batch([image for _record, image in loaded_batch])
        for (record, image), detections in zip(loaded_batch, detections_by_image, strict=True):
            detection_rows = build_detection_rows(
                record=record,
                image=image,
                detections=detections,
                detector_backend=detector.backend,
                detector_model_id=detector.model_id,
                detector_model_version=detector.model_version,
                detector_checkpoint=detector.checkpoint,
                policy=policy,
            )
            for row in detection_rows:
                enriched = _with_crop_metadata(row, image=image, policy=policy, debug_writer=None)
                if enriched.get("crop_hash"):
                    crops_created += 1
                rows.append(enriched)
    frame = pl.DataFrame(rows) if rows else empty_detection_frame()
    detections_written = frame.filter(pl.col("detection_status") == "detected").height if frame.height else 0
    return CloudDetectionBatchResult(
        frame=frame,
        records_seen=records_seen,
        images_loaded=images_loaded,
        image_failures=image_failures,
        detections_written=detections_written,
        crops_created=crops_created,
    )


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


@dataclass(frozen=True)
class _LoadedFailure:
    record: dict[str, Any]
    image: None = None
    failure_reason: str | None = None


def detection_batch_id(work_items: list[dict[str, Any]]) -> str:
    work_keys = [str(item.get("work_key") or "") for item in work_items]
    return _stable_hash({"work_keys": work_keys})


def detection_work_item(
    record: dict[str, Any],
    *,
    run_id: str,
    source_shard_uri: str,
    detector: dict[str, str],
) -> dict[str, Any]:
    source = str(record.get("source") or "flickr")
    flickr_photo_id = str(record.get("flickr_photo_id") or record.get("id") or "")
    image_url = str(record.get("image_url") or "")
    key_payload = {
        "run_id": run_id,
        "source": source,
        "flickr_photo_id": flickr_photo_id,
        "image_url": image_url,
        "detector_backend": detector.get("backend") or "",
        "detector_model_id": detector.get("model_id") or "",
        "detector_model_version": detector.get("model_version") or "",
        "detector_checkpoint": detector.get("checkpoint") or "",
    }
    return {
        "work_key": f"{run_id}:detect:{_stable_hash(key_payload)}",
        "run_id": run_id,
        "source": source,
        "flickr_photo_id": flickr_photo_id,
        "image_url": image_url,
        "source_shard_uri": source_shard_uri,
        "source_record": _jsonable_record(record),
        "detector": dict(detector),
    }


def source_record_from_detection_work_item(item: dict[str, Any]) -> dict[str, Any]:
    payload = item.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"work item {item.get('work_key')} has invalid payload")
    record = payload.get("source_record")
    if not isinstance(record, dict):
        raise ValueError(f"work item {item.get('work_key')} has no source_record payload")
    return dict(record)


def _record_is_detectable(record: dict[str, Any]) -> bool:
    return bool(str(record.get("flickr_photo_id") or record.get("id") or "") and str(record.get("image_url") or ""))


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable_record(record: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable_value(value) for key, value in record.items()}


def _jsonable_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    return str(value)


__all__ = [
    "CloudDetectionBatchResult",
    "DetectionWorkPlanResult",
    "detection_batch_id",
    "detection_work_item",
    "enqueue_detection_work_from_source_shards",
    "run_cloud_detection_batch",
    "source_record_from_detection_work_item",
]
