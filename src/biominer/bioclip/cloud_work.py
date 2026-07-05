from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from biominer.detection.policy import DetectionPolicy, detection_is_bioclip_eligible
from biominer.storage.cloud import CloudStorage
from biominer.workstore.base import WorkStore


@dataclass(frozen=True)
class BioClipWorkPlanResult:
    detection_shards_seen: int
    detections_seen: int
    eligible_detections_seen: int
    enqueued_work_items: int
    duplicate_work_items: int


def enqueue_bioclip_work_from_detection_shards(
    *,
    storage: CloudStorage,
    workstore: WorkStore,
    job_name: str,
    registry_version: str | None,
    run_id: str,
    detection_stage: str,
    score_stage: str,
    model_id: str,
    model_version: str,
    model_checkpoint: str,
    candidate_set_id: str,
    ablation_modes: tuple[str, ...] = ("detector_crop",),
    detection_policy: DetectionPolicy | None = None,
    limit: int | None = None,
) -> BioClipWorkPlanResult:
    shards = workstore.list_candidate_shards(
        job_name=job_name,
        stage=detection_stage,
        registry_version=registry_version,
        run_id=run_id,
    )
    modes = tuple(mode for mode in ablation_modes if str(mode).strip()) or ("detector_crop",)
    model = {
        "model_id": model_id,
        "model_version": model_version,
        "checkpoint": model_checkpoint,
    }
    items: list[dict[str, Any]] = []
    detections_seen = 0
    eligible_seen = 0
    remaining = None if limit is None or limit <= 0 else int(limit)
    for shard in shards:
        if remaining == 0:
            break
        detection_shard_uri = str(shard["uri"])
        frame = storage.read_parquet(detection_shard_uri)
        for row in frame.iter_rows(named=True):
            detections_seen += 1
            detection = dict(row)
            if not _detection_is_scoreable(detection, detection_policy=detection_policy):
                continue
            eligible_seen += 1
            for mode in modes:
                if remaining == 0:
                    break
                items.append(
                    bioclip_score_work_item(
                        detection,
                        run_id=run_id,
                        detection_shard_uri=detection_shard_uri,
                        model=model,
                        candidate_set_id=candidate_set_id,
                        ablation_mode=mode,
                    )
                )
                if remaining is not None:
                    remaining -= 1
            if remaining == 0:
                break
    inserted = workstore.enqueue_work(job_name, registry_version, items, stage=score_stage) if items else 0
    return BioClipWorkPlanResult(
        detection_shards_seen=len(shards),
        detections_seen=detections_seen,
        eligible_detections_seen=eligible_seen,
        enqueued_work_items=inserted,
        duplicate_work_items=len(items) - inserted,
    )


def bioclip_score_work_item(
    detection: dict[str, Any],
    *,
    run_id: str,
    detection_shard_uri: str,
    model: dict[str, str],
    candidate_set_id: str,
    ablation_mode: str,
) -> dict[str, Any]:
    key_payload = {
        "run_id": run_id,
        "source": str(detection.get("source") or ""),
        "flickr_photo_id": str(detection.get("flickr_photo_id") or ""),
        "detection_id": str(detection.get("detection_id") or ""),
        "crop_hash": str(detection.get("crop_hash") or ""),
        "model_id": str(model.get("model_id") or ""),
        "model_version": str(model.get("model_version") or ""),
        "model_checkpoint": str(model.get("checkpoint") or ""),
        "candidate_set_id": str(candidate_set_id or ""),
        "ablation_mode": str(ablation_mode or ""),
    }
    return {
        "work_key": f"{run_id}:bioclip:{_stable_hash(key_payload)}",
        "run_id": run_id,
        "source": key_payload["source"],
        "flickr_photo_id": key_payload["flickr_photo_id"],
        "detection_id": key_payload["detection_id"],
        "crop_hash": key_payload["crop_hash"],
        "detection_shard_uri": detection_shard_uri,
        "detection": _jsonable_record(detection),
        "model": dict(model),
        "candidate_set_id": candidate_set_id,
        "ablation_mode": ablation_mode,
    }


def detection_from_bioclip_work_item(item: dict[str, Any]) -> dict[str, Any]:
    payload = item.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"work item {item.get('work_key')} has invalid payload")
    detection = payload.get("detection")
    if not isinstance(detection, dict):
        raise ValueError(f"work item {item.get('work_key')} has no detection payload")
    return dict(detection)


def bioclip_score_batch_id(work_items: list[dict[str, Any]]) -> str:
    work_keys = [str(item.get("work_key") or "") for item in work_items]
    return _stable_hash({"work_keys": work_keys})


def _detection_is_scoreable(
    detection: dict[str, Any],
    *,
    detection_policy: DetectionPolicy | None,
) -> bool:
    if not detection_is_bioclip_eligible(detection, detection_policy):
        return False
    return bool(
        str(detection.get("source") or "")
        and str(detection.get("flickr_photo_id") or "")
        and str(detection.get("detection_id") or "")
    )


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
    "BioClipWorkPlanResult",
    "bioclip_score_batch_id",
    "bioclip_score_work_item",
    "detection_from_bioclip_work_item",
    "enqueue_bioclip_work_from_detection_shards",
]
