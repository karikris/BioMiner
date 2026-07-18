from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import polars as pl

from biominer.detection.detector_base import DecodedImage, ObjectDetector
from biominer.detection.pipeline import (
    ImageLoader,
    _image_failure_row,
    _next_detector_batch_size,
    _resize_image_to_max_side,
    _should_retry_detector_batch,
    _with_crop_metadata,
)
from biominer.detection.policy import DetectionPolicy
from biominer.detection.route_contract import (
    DetectorRouteContract,
    build_detector_route_contract,
)
from biominer.detection.schema import build_detection_rows, empty_detection_frame
from biominer.storage.cloud import CloudStorage
from biominer.storage.parquet import DEFAULT_PARQUET_READ_BATCH_SIZE
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
    adaptive_batching_enabled: bool = False
    detector_batch_retries: int = 0
    detector_batch_size_initial: int = 16
    detector_batch_size_final: int = 16
    detector_batch_size_min: int = 1
    detector_route_contract_version: str = ""
    detector_route_contract_fingerprint: str = ""
    detector_execution_mode: str = "injected"
    detector_model_load_count: int = 0


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
    detector_prompt_classes: tuple[str, ...] = (),
    detector_prompt_set_fingerprint: str = "",
    detection_policy: DetectionPolicy | None = None,
    vision_settings: Any | None = None,
    limit: int | None = None,
    read_batch_size: int = DEFAULT_PARQUET_READ_BATCH_SIZE,
) -> DetectionWorkPlanResult:
    shards = workstore.list_candidate_shards(
        job_name=job_name,
        stage=source_stage,
        registry_version=registry_version,
        run_id=run_id,
    )
    records_seen = 0
    attempted = 0
    inserted = 0
    remaining = None if limit is None or limit <= 0 else int(limit)
    detector = {
        "backend": detector_backend,
        "model_id": detector_model_id,
        "model_version": detector_model_version,
        "checkpoint": detector_checkpoint,
        "prompt_classes": list(detector_prompt_classes),
        "prompt_set_fingerprint": detector_prompt_set_fingerprint,
    }
    for shard in shards:
        if remaining == 0:
            break
        source_shard_uri = str(shard["uri"])
        for frame in storage.iter_parquet_batches(source_shard_uri, batch_size=read_batch_size):
            batch_items: list[dict[str, Any]] = []
            for row in frame.iter_rows(named=True):
                if remaining == 0:
                    break
                record = dict(row)
                if not _record_is_detectable(record):
                    continue
                records_seen += 1
                batch_items.append(
                    detection_work_item(
                        record,
                        run_id=run_id,
                        source_shard_uri=source_shard_uri,
                        detector=detector,
                        detection_policy=detection_policy,
                        vision_settings=vision_settings,
                    )
                )
                if remaining is not None:
                    remaining -= 1
            if batch_items:
                attempted += len(batch_items)
                inserted += workstore.enqueue_work(job_name, registry_version, batch_items, stage=detection_stage)
            if remaining == 0:
                break
    return DetectionWorkPlanResult(
        source_shards_seen=len(shards),
        source_records_seen=records_seen,
        enqueued_work_items=inserted,
        duplicate_work_items=attempted - inserted,
    )


def run_cloud_detection_batch(
    *,
    work_items: list[dict[str, Any]],
    detector: ObjectDetector,
    image_loader: ImageLoader,
    detection_policy: DetectionPolicy | None = None,
    detector_batch_size: int = 16,
    adaptive_batching: bool = False,
    min_detector_batch_size: int = 1,
) -> CloudDetectionBatchResult:
    if detector_batch_size <= 0:
        raise ValueError("detector_batch_size must be positive")
    if min_detector_batch_size <= 0:
        raise ValueError("min_detector_batch_size must be positive")
    if min_detector_batch_size > detector_batch_size:
        raise ValueError("min_detector_batch_size must be <= detector_batch_size")
    policy = detection_policy or DetectionPolicy(backend=detector.backend)
    expected_contracts = {
        DetectorRouteContract.from_mapping(_work_item_route_contract(item)).fingerprint
        for item in work_items
    }
    if len(expected_contracts) > 1:
        raise ValueError("cloud detection batch mixes detector route contracts")
    actual_contract = build_detector_route_contract(detector, policy)
    if expected_contracts and expected_contracts != {actual_contract.fingerprint}:
        raise ValueError("cloud detector route contract differs from queued work")
    rows: list[dict[str, Any]] = []
    records_seen = 0
    images_loaded = 0
    image_failures = 0
    crops_created = 0
    current_detector_batch_size = detector_batch_size
    detector_batch_retries = 0
    loaded: list[tuple[dict[str, Any], DecodedImage]] = []
    for item in work_items:
        record = source_record_from_detection_work_item(item)
        records_seen += 1
        try:
            image = _resize_image_to_max_side(image_loader(record), policy.image_max_side_px)
        except Exception as exc:  # noqa: BLE001 - image failures become durable detection rows.
            image_failures += 1
            rows.append(
                _image_failure_row(
                    _LoadedFailure(record=record, failure_reason=str(exc)),
                    detector=detector,
                    policy=policy,
                )
            )
            continue
        images_loaded += 1
        loaded.append((record, image))
    pending_batches = _chunks(loaded, current_detector_batch_size)
    while pending_batches:
        loaded_batch = pending_batches.pop(0)
        try:
            detections_by_image = detector.detect_batch([image for _record, image in loaded_batch])
        except RuntimeError as exc:
            if not _should_retry_detector_batch(
                exc,
                adaptive_batching=adaptive_batching,
                batch_size=len(loaded_batch),
                current_batch_size=current_detector_batch_size,
                min_batch_size=min_detector_batch_size,
            ):
                raise
            current_detector_batch_size = _next_detector_batch_size(
                current_batch_size=current_detector_batch_size,
                failed_batch_size=len(loaded_batch),
                min_batch_size=min_detector_batch_size,
            )
            detector_batch_retries += 1
            pending_batches = _chunks(loaded_batch, current_detector_batch_size) + pending_batches
            continue
        if len(detections_by_image) != len(loaded_batch):
            raise ValueError(f"detector returned {len(detections_by_image)} result rows for {len(loaded_batch)} images")
        for (record, image), detections in zip(loaded_batch, detections_by_image, strict=True):
            detection_rows = build_detection_rows(
                record=record,
                image=image,
                detections=detections,
                detector_backend=detector.backend,
                detector_model_id=detector.model_id,
                detector_model_version=detector.model_version,
                detector_checkpoint=detector.checkpoint,
                detector_prompt_set_fingerprint=getattr(
                    detector, "prompt_set_fingerprint", None
                ),
                policy=policy,
            )
            for row in detection_rows:
                enriched = _with_crop_metadata(row, image=image, policy=policy, debug_writer=None)
                if enriched.get("crop_hash"):
                    crops_created += 1
                rows.append(enriched)
    frame = pl.DataFrame(rows) if rows else empty_detection_frame()
    detections_written = frame.filter(pl.col("detection_status") == "detected").height if frame.height else 0
    realized_contract = build_detector_route_contract(detector, policy)
    return CloudDetectionBatchResult(
        frame=frame,
        records_seen=records_seen,
        images_loaded=images_loaded,
        image_failures=image_failures,
        detections_written=detections_written,
        crops_created=crops_created,
        adaptive_batching_enabled=bool(adaptive_batching),
        detector_batch_retries=detector_batch_retries,
        detector_batch_size_initial=detector_batch_size,
        detector_batch_size_final=current_detector_batch_size,
        detector_batch_size_min=min_detector_batch_size,
        detector_route_contract_version=realized_contract.contract_version,
        detector_route_contract_fingerprint=realized_contract.fingerprint,
        detector_execution_mode=realized_contract.execution_mode,
        detector_model_load_count=_detector_model_load_count(detector),
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
    detector: dict[str, Any],
    detection_policy: DetectionPolicy | None = None,
    vision_settings: Any | None = None,
) -> dict[str, Any]:
    source = str(record.get("source") or "flickr")
    flickr_photo_id = str(record.get("flickr_photo_id") or record.get("id") or "")
    image_url = str(record.get("image_url") or "")
    base_policy = DetectionPolicy(
        backend=str(detector.get("backend") or "yoloe26")
    )
    active_policy = detection_policy
    if active_policy is None and hasattr(vision_settings, "to_detection_policy"):
        active_policy = vision_settings.to_detection_policy(base_policy)
    active_policy = active_policy or base_policy
    policy_key = _detection_policy_key(active_policy)
    runtime_key = _vision_runtime_key(vision_settings)
    detector_identity = {
        "backend": str(detector.get("backend") or ""),
        "model_id": str(detector.get("model_id") or ""),
        "model_version": str(detector.get("model_version") or ""),
        "checkpoint": str(detector.get("checkpoint") or ""),
        "prompt_classes": [
            str(prompt) for prompt in detector.get("prompt_classes") or []
        ],
        "prompt_set_fingerprint": str(
            detector.get("prompt_set_fingerprint") or ""
        ),
    }
    contract_detector = dict(detector_identity)
    if contract_detector["backend"] == "yoloe26":
        contract_detector.update(
            {
                "execution_mode": str(
                    detector.get("execution_mode") or "persistent_sidecar"
                ),
                "transport": str(
                    detector.get("transport")
                    or getattr(vision_settings, "yolo_sidecar_transport", "json_b64")
                ),
                "imgsz": detector.get("imgsz")
                or getattr(vision_settings, "yolo_imgsz", None),
                "conf": detector.get("conf")
                or getattr(vision_settings, "yolo_conf", None),
                "iou": detector.get("iou")
                or getattr(vision_settings, "yolo_iou", None),
                "max_det": detector.get("max_det")
                or getattr(vision_settings, "yolo_max_det", None),
            }
        )
    route_contract = build_detector_route_contract(
        contract_detector, active_policy
    )
    key_payload = {
        "run_id": run_id,
        "source": source,
        "flickr_photo_id": flickr_photo_id,
        "image_url": image_url,
        "detector": detector_identity,
        "detector_route_contract": route_contract.to_dict(),
        "detection_policy": policy_key,
        "vision_runtime": runtime_key,
    }
    return {
        "work_key": f"{run_id}:detect:{_stable_hash(key_payload)}",
        "run_id": run_id,
        "source": source,
        "flickr_photo_id": flickr_photo_id,
        "image_url": image_url,
        "source_shard_uri": source_shard_uri,
        "source_record": _jsonable_record(record),
        "detector": detector_identity,
        "detector_route_contract": route_contract.to_dict(),
        "detection_policy": policy_key,
        "vision_runtime": runtime_key,
    }


def _work_item_route_contract(item: dict[str, Any]) -> dict[str, object]:
    payload = item.get("payload")
    source = payload if isinstance(payload, dict) else item
    contract = source.get("detector_route_contract")
    if not isinstance(contract, dict):
        raise ValueError("cloud detection work item lacks detector route contract")
    return dict(contract)


def _detector_model_load_count(detector: ObjectDetector) -> int:
    value = getattr(detector, "model_load_count", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("detector model_load_count must be a non-negative integer")
    return value


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


def _detection_policy_key(policy: DetectionPolicy) -> dict[str, Any]:
    routing_policy = policy.routing_policy
    return {
        "backend": policy.backend,
        "box_score_threshold": policy.box_score_threshold,
        "nms_iou_threshold": policy.nms_iou_threshold,
        "min_box_area_ratio": policy.min_box_area_ratio,
        "max_boxes_per_image": policy.max_boxes_per_image,
        "routing_policy": {
            "version": routing_policy.version,
            "fingerprint": routing_policy.fingerprint,
            "possible_adult_route_enabled": (
                routing_policy.possible_adult_route_enabled
            ),
            "possible_adult_route_threshold": (
                routing_policy.possible_adult_route_threshold
            ),
            "ambiguous_insect_review_enabled": (
                routing_policy.ambiguous_insect_review_enabled
            ),
            "ambiguous_insect_review_threshold": (
                routing_policy.ambiguous_insect_review_threshold
            ),
        },
        "crop_padding_ratio": policy.crop_padding_ratio,
        "image_max_side_px": policy.image_max_side_px,
        "crop_target_px": policy.crop_target_px,
    }


def _vision_runtime_key(settings: Any | None) -> dict[str, Any]:
    if settings is None:
        return {}
    output_affecting_fields = (
        "yolo_checkpoint",
        "yolo_imgsz",
        "yolo_conf",
        "yolo_iou",
        "yolo_max_det",
        "possible_adult_route_enabled",
        "possible_adult_route_threshold",
        "ambiguous_insect_review_enabled",
        "ambiguous_insect_review_threshold",
        "crop_padding_ratio",
        "crop_target_px",
        "image_max_side_px",
    )
    return {
        field_name: _jsonable_value(getattr(settings, field_name))
        for field_name in output_affecting_fields
        if hasattr(settings, field_name)
    }


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
