from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from shutil import rmtree
from typing import Any

import polars as pl

from biominer.bioclip.candidate_sets import CandidateSet
from biominer.bioclip.classification_modes import (
    DEFAULT_CLASSIFICATION_MODE,
    DEFAULT_FAMILY_TOP_K,
    DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    DEFAULT_SPECIES_RERANK_TOP_K,
    ClassificationMode,
    normalize_classification_mode,
)
from biominer.bioclip.cloud_work import bioclip_score_work_item, run_cloud_bioclip_batch
from biominer.bioclip.object_runner import ObjectBioClipScorer
from biominer.bioclip.taxonomy_store import ButterflyTaxonomyStore
from biominer.detection.cloud_work import detection_work_item, run_cloud_detection_batch
from biominer.detection.detector_base import ObjectDetector
from biominer.detection.pipeline import ImageLoader
from biominer.detection.policy import DetectionPolicy
from biominer.evidence.join import build_object_evidence_frames
from biominer.species.context import SpeciesContext
from biominer.storage.cloud import CloudStorage
from biominer.storage.parquet import DEFAULT_PARQUET_COMPRESSION, ParquetPartWrite
from biominer.storage.shard_paths import build_parquet_part_uri
from biominer.vision.gates import BioClipGatePolicy
from biominer.vision.score_inputs import MaterializedBioClipScoreInputs, materialize_bioclip_score_inputs
from biominer.workstore.base import WorkStore


ROLLING_VISION_WORK_STAGE = "detect_objects"

ROLLING_VISION_ARTIFACT_STAGES: dict[str, str] = {
    "image_batch_manifest": "image_batch_manifest",
    "object_detections": "detect_objects",
    "bioclip_score_inputs": "bioclip_score_inputs",
    "object_bioclip_scores": "score_bioclip",
    "object_evidence_joined": "join_evidence",
    "photo_evidence_summary": "photo_summary",
}

ROLLING_VISION_STORAGE_STAGES: dict[str, str] = {
    "image_batch_manifest": "image_batch_manifest",
    "object_detections": "object_detections",
    "bioclip_score_inputs": "bioclip_score_inputs",
    "object_bioclip_scores": "object_bioclip_scores",
    "object_evidence_joined": "object_evidence_joined",
    "photo_evidence_summary": "photo_evidence_summary",
}

ROLLING_VISION_ARTIFACT_ORDER: tuple[str, ...] = tuple(ROLLING_VISION_ARTIFACT_STAGES)


@dataclass(frozen=True)
class RollingVisionWorkPlanResult:
    source_shards_seen: int
    source_records_seen: int
    batches_planned: int
    enqueued_work_items: int
    duplicate_work_items: int


@dataclass(frozen=True)
class CloudRollingVisionBatchResult:
    batch_id: str
    part_id: str
    frames: dict[str, pl.DataFrame]
    metrics: dict[str, Any]
    materialized_score_inputs: MaterializedBioClipScoreInputs | None = None

    def cleanup_after_commit(self) -> None:
        if self.materialized_score_inputs is not None:
            self.materialized_score_inputs.cleanup()


@dataclass(frozen=True)
class RollingVisionShardCommitResult:
    output_uris: dict[str, str]
    rows_by_artifact: dict[str, int]
    parts_written: int
    parts_reused: int


def enqueue_rolling_vision_work_from_source_shards(
    *,
    storage: CloudStorage,
    workstore: WorkStore,
    job_name: str,
    registry_version: str | None,
    run_id: str,
    source_stage: str,
    vision_stage: str = ROLLING_VISION_WORK_STAGE,
    vision_batch_rows: int = 500,
    detector: dict[str, str] | None = None,
    vision_settings: Any | None = None,
    bioclip_gate_mode: str = "exclude_hard_negative",
    score_no_detection_whole_image: bool = True,
    bioclip_model: dict[str, str] | None = None,
    candidate_set_id: str = "",
    classification_mode: ClassificationMode = DEFAULT_CLASSIFICATION_MODE,
    taxonomy_table_version: str | None = None,
    taxonomy_prompt_variant_version: str | None = None,
    family_top_k: int = DEFAULT_FAMILY_TOP_K,
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K,
    limit: int | None = None,
) -> RollingVisionWorkPlanResult:
    if vision_batch_rows <= 0:
        raise ValueError("vision_batch_rows must be positive")
    shards = sorted(
        workstore.list_candidate_shards(
            job_name=job_name,
            stage=source_stage,
            registry_version=registry_version,
            run_id=run_id,
        ),
        key=lambda shard: str(shard.get("uri") or ""),
    )
    records: list[dict[str, Any]] = []
    record_source_uris: list[str] = []
    remaining = None if limit is None or limit <= 0 else int(limit)
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
            records.append(_jsonable_record(record))
            record_source_uris.append(source_shard_uri)
            if remaining is not None:
                remaining -= 1
    settings_key = rolling_vision_settings_key(
        detector=detector or {},
        vision_settings=vision_settings,
        bioclip_gate_mode=bioclip_gate_mode,
        score_no_detection_whole_image=score_no_detection_whole_image,
        bioclip_model=bioclip_model or {},
        candidate_set_id=candidate_set_id,
        classification_mode=classification_mode,
        taxonomy_table_version=taxonomy_table_version,
        taxonomy_prompt_variant_version=taxonomy_prompt_variant_version,
        family_top_k=family_top_k,
        species_first_pass_top_k=species_first_pass_top_k,
        species_rerank_top_k=species_rerank_top_k,
    )
    items: list[dict[str, Any]] = []
    for batch_index, offset in enumerate(range(0, len(records), vision_batch_rows)):
        batch_records = records[offset : offset + vision_batch_rows]
        batch_source_uris = record_source_uris[offset : offset + vision_batch_rows]
        items.append(
            rolling_vision_work_item(
                batch_records,
                run_id=run_id,
                batch_index=batch_index,
                vision_batch_rows=vision_batch_rows,
                source_shard_uris=batch_source_uris,
                settings_key=settings_key,
            )
        )
    inserted = workstore.enqueue_work(job_name, registry_version, items, stage=vision_stage) if items else 0
    return RollingVisionWorkPlanResult(
        source_shards_seen=len(shards),
        source_records_seen=len(records),
        batches_planned=len(items),
        enqueued_work_items=inserted,
        duplicate_work_items=len(items) - inserted,
    )


def rolling_vision_work_item(
    records: list[dict[str, Any]],
    *,
    run_id: str,
    batch_index: int,
    vision_batch_rows: int,
    source_shard_uris: list[str],
    settings_key: dict[str, Any],
) -> dict[str, Any]:
    batch_id = f"vision-batch-{batch_index:06d}"
    part_id = f"part-{batch_index:06d}"
    record_identities = [_source_record_identity(record) for record in records]
    key_payload = {
        "run_id": run_id,
        "batch_id": batch_id,
        "batch_index": int(batch_index),
        "vision_batch_rows": int(vision_batch_rows),
        "source_records": record_identities,
        "settings": settings_key,
    }
    return {
        "work_key": f"{run_id}:rolling-vision:{_stable_hash(key_payload)}",
        "run_id": run_id,
        "batch_id": batch_id,
        "batch_index": int(batch_index),
        "part_id": part_id,
        "vision_batch_rows": int(vision_batch_rows),
        "source_shard_uris": sorted(set(str(uri) for uri in source_shard_uris if str(uri))),
        "source_records": [_jsonable_record(record) for record in records],
        "source_record_identities": record_identities,
        "settings_key": settings_key,
    }


def rolling_vision_settings_key(
    *,
    detector: dict[str, str],
    vision_settings: Any | None,
    bioclip_gate_mode: str,
    score_no_detection_whole_image: bool,
    bioclip_model: dict[str, str],
    candidate_set_id: str,
    classification_mode: ClassificationMode = DEFAULT_CLASSIFICATION_MODE,
    taxonomy_table_version: str | None = None,
    taxonomy_prompt_variant_version: str | None = None,
    family_top_k: int = DEFAULT_FAMILY_TOP_K,
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K,
) -> dict[str, Any]:
    return {
        "detector": {
            "backend": str(detector.get("backend") or ""),
            "model_id": str(detector.get("model_id") or ""),
            "model_version": str(detector.get("model_version") or ""),
            "checkpoint": str(detector.get("checkpoint") or ""),
            "yolo_imgsz": _settings_value(vision_settings, "yolo_imgsz"),
            "yolo_conf": _settings_value(vision_settings, "yolo_conf"),
            "yolo_iou": _settings_value(vision_settings, "yolo_iou"),
            "yolo_max_det": _settings_value(vision_settings, "yolo_max_det"),
        },
        "crop": {
            "crop_padding_ratio": _settings_value(vision_settings, "crop_padding_ratio"),
            "crop_target_px": _settings_value(vision_settings, "crop_target_px"),
        },
        "bioclip_gate": {
            "mode": str(bioclip_gate_mode),
            "score_no_detection_whole_image": bool(score_no_detection_whole_image),
        },
        "bioclip_model": {
            "model_id": str(bioclip_model.get("model_id") or ""),
            "model_version": str(bioclip_model.get("model_version") or ""),
            "checkpoint": str(bioclip_model.get("checkpoint") or ""),
        },
        "candidate_set_id": str(candidate_set_id or ""),
        "classification_mode": normalize_classification_mode(classification_mode),
        "taxonomy_table_version": str(taxonomy_table_version or ""),
        "taxonomy_prompt_variant_version": str(taxonomy_prompt_variant_version or ""),
        "top_k_settings": {
            "family_top_k": int(family_top_k),
            "species_first_pass_top_k": int(species_first_pass_top_k),
            "species_rerank_top_k": int(species_rerank_top_k),
        },
    }


def run_cloud_rolling_vision_batch(
    *,
    work_item: dict[str, Any],
    detector: ObjectDetector,
    image_loader: ImageLoader,
    species_context: SpeciesContext,
    candidate_set: CandidateSet,
    scorer: ObjectBioClipScorer,
    detection_policy: DetectionPolicy,
    bioclip_gate_policy: BioClipGatePolicy,
    temp_dir: str | Path,
    detector_batch_size: int = 16,
    adaptive_detector_batching: bool = False,
    min_detector_batch_size: int = 1,
    crop_padding_ratio: float = 0.12,
    crop_target_px: int = 336,
    bioclip_batch_size: int = 24,
    adaptive_bioclip_batching: bool = False,
    min_bioclip_batch_size: int = 1,
    classification_mode: ClassificationMode = DEFAULT_CLASSIFICATION_MODE,
    taxonomy_table_version: str | None = None,
    taxonomy_prompt_variant_version: str | None = None,
    family_top_k: int = DEFAULT_FAMILY_TOP_K,
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K,
    taxonomy_store: ButterflyTaxonomyStore | None = None,
) -> CloudRollingVisionBatchResult:
    payload = _work_payload(work_item)
    records = [dict(row) for row in payload.get("source_records") or [] if isinstance(row, dict)]
    batch_id = str(payload.get("batch_id") or "")
    part_id = str(payload.get("part_id") or batch_id or "part-000000")
    canonical = pl.DataFrame(records)
    manifest = _image_batch_manifest_frame(records, batch_id=batch_id, part_id=part_id)
    detection_items = [
        {
            "work_key": f"{str(work_item.get('work_key') or batch_id)}:detect:{index:06d}",
            "payload": detection_work_item(
                record,
                run_id=str(payload.get("run_id") or ""),
                source_shard_uri=",".join(str(uri) for uri in payload.get("source_shard_uris") or []),
                detector={
                    "backend": detector.backend,
                    "model_id": detector.model_id,
                    "model_version": detector.model_version,
                    "checkpoint": detector.checkpoint,
                },
                detection_policy=detection_policy,
                vision_settings=None,
            ),
        }
        for index, record in enumerate(records)
    ]
    detection_result = run_cloud_detection_batch(
        work_items=detection_items,
        detector=detector,
        image_loader=image_loader,
        detection_policy=detection_policy,
        detector_batch_size=detector_batch_size,
        adaptive_batching=adaptive_detector_batching,
        min_detector_batch_size=min_detector_batch_size,
    )
    score_inputs = materialize_bioclip_score_inputs(
        canonical_records=canonical,
        detections=detection_result.frame,
        image_loader=image_loader,
        temp_dir=temp_dir,
        gate_policy=bioclip_gate_policy,
        crop_padding_ratio=crop_padding_ratio,
        crop_target_px=crop_target_px,
        batch_id=batch_id,
        part_id=part_id,
    )
    model = {
        "model_id": scorer.model_id,
        "model_version": scorer.model_version,
        "checkpoint": scorer.model_checkpoint,
    }
    score_work_items = [
        {
            "work_key": f"{str(work_item.get('work_key') or batch_id)}:score:{index:06d}",
            "payload": bioclip_score_work_item(
                dict(item),
                run_id=str(payload.get("run_id") or ""),
                detection_shard_uri=f"rolling://{batch_id}/object_detections/{part_id}",
                model=model,
                candidate_set_id=candidate_set.candidate_set_id,
                ablation_mode=str(item.get("visual_input_kind") or item.get("ablation_mode") or "detector_crop"),
                classification_mode=classification_mode,
                taxonomy_table_version=taxonomy_table_version,
                taxonomy_prompt_variant_version=taxonomy_prompt_variant_version,
                family_top_k=family_top_k,
                species_first_pass_top_k=species_first_pass_top_k,
                species_rerank_top_k=species_rerank_top_k,
            ),
        }
        for index, item in enumerate(score_inputs.items)
    ]
    score_result = run_cloud_bioclip_batch(
        work_items=score_work_items,
        species_context=species_context,
        candidate_set=candidate_set,
        scorer=scorer,
        crop_batch_size=bioclip_batch_size,
        adaptive_batching=adaptive_bioclip_batching,
        min_crop_batch_size=min_bioclip_batch_size,
        classification_mode=classification_mode,
        family_top_k=family_top_k,
        species_first_pass_top_k=species_first_pass_top_k,
        species_rerank_top_k=species_rerank_top_k,
        taxonomy_store=taxonomy_store,
    )
    joined, summary = build_object_evidence_frames(
        canonical_source_records=canonical,
        object_detections=detection_result.frame,
        object_scores=score_result.frame,
        species_context=species_context,
    )
    return CloudRollingVisionBatchResult(
        batch_id=batch_id,
        part_id=part_id,
        frames={
            "image_batch_manifest": manifest,
            "object_detections": detection_result.frame,
            "bioclip_score_inputs": score_inputs.frame,
            "object_bioclip_scores": score_result.frame,
            "object_evidence_joined": joined,
            "photo_evidence_summary": summary,
        },
        metrics={
            "records_seen": detection_result.records_seen,
            "images_loaded": detection_result.images_loaded,
            "image_failures": detection_result.image_failures,
            "detections_written": detection_result.detections_written,
            "crops_created": detection_result.crops_created,
            "score_inputs": score_inputs.frame.height,
            "objects_scored": score_result.crops_scored,
            "whole_images_scored": _mode_row_count(score_result.frame, "whole_image"),
            "detector_crops_scored": _mode_row_count(score_result.frame, "detector_crop"),
            "segmentation_crops_scored": _mode_row_count(score_result.frame, "detector_crop_segmentation"),
            "object_evidence_rows": joined.height,
            "photo_summary_rows": summary.height,
            "detector_batch_retries": detection_result.detector_batch_retries,
            "bioclip_batch_retries": score_result.bioclip_batch_retries,
        },
        materialized_score_inputs=score_inputs,
    )


def commit_rolling_vision_batch_shards(
    *,
    storage: CloudStorage,
    workstore: WorkStore,
    job_name: str,
    registry_version: str | None,
    run_id: str,
    worker_id: str,
    base_prefix: str,
    work_key: str,
    batch_id: str,
    part_id: str,
    frames: dict[str, pl.DataFrame],
    compression: str | None = DEFAULT_PARQUET_COMPRESSION,
    metadata: dict[str, Any] | None = None,
) -> RollingVisionShardCommitResult:
    missing = [artifact for artifact in ROLLING_VISION_ARTIFACT_ORDER if artifact not in frames]
    if missing:
        raise ValueError("missing rolling vision frame(s): " + ", ".join(missing))
    writes: dict[str, tuple[ParquetPartWrite, bool]] = {}
    for artifact in ROLLING_VISION_ARTIFACT_ORDER:
        uri = build_parquet_part_uri(
            base_prefix,
            stage=ROLLING_VISION_STORAGE_STAGES[artifact],
            run_id=run_id,
            worker_id=worker_id,
            part_id=part_id,
        )
        writes[artifact] = _write_immutable_parquet_part(storage, uri, frames[artifact], compression=compression)
    for artifact in ROLLING_VISION_ARTIFACT_ORDER:
        part_write, part_written = writes[artifact]
        workstore.register_shard(
            job_name=job_name,
            registry_version=registry_version,
            stage=ROLLING_VISION_ARTIFACT_STAGES[artifact],
            run_id=run_id,
            worker_id=worker_id,
            uri=part_write.uri,
            checksum=None,
            row_count=part_write.row_count,
            byte_count=part_write.byte_count,
            metadata={
                **(metadata or {}),
                "artifact": artifact,
                "batch_id": batch_id,
                "part_id": part_id,
                "part_written": part_written,
                "parquet_compression": part_write.compression,
            },
        )
    output_uris = {artifact: writes[artifact][0].uri for artifact in ROLLING_VISION_ARTIFACT_ORDER}
    rows_by_artifact = {artifact: int(writes[artifact][0].row_count or 0) for artifact in ROLLING_VISION_ARTIFACT_ORDER}
    workstore.mark_completed(
        work_key,
        output_uri=output_uris["photo_evidence_summary"],
        checksum=None,
        row_count=rows_by_artifact["photo_evidence_summary"],
    )
    return RollingVisionShardCommitResult(
        output_uris=output_uris,
        rows_by_artifact=rows_by_artifact,
        parts_written=sum(1 for _artifact, (_write, written) in writes.items() if written),
        parts_reused=sum(1 for _artifact, (_write, written) in writes.items() if not written),
    )


def rolling_vision_batch_id(work_item: dict[str, Any]) -> str:
    payload = _work_payload(work_item)
    return str(payload.get("batch_id") or _stable_hash({"work_key": str(work_item.get("work_key") or "")})[:24])


def rolling_vision_part_id(work_item: dict[str, Any]) -> str:
    payload = _work_payload(work_item)
    return str(payload.get("part_id") or rolling_vision_batch_id(work_item))


def cleanup_rolling_batch_temp_dir(path: str | Path) -> None:
    root = Path(path)
    if root.exists():
        rmtree(root)


def _write_immutable_parquet_part(
    storage: CloudStorage,
    uri: str,
    frame: pl.DataFrame,
    *,
    compression: str | None = DEFAULT_PARQUET_COMPRESSION,
) -> tuple[ParquetPartWrite, bool]:
    try:
        return storage.write_parquet_part(uri, frame, compression=compression, overwrite=False), True
    except FileExistsError:
        if not storage.exists(uri):
            raise
        return ParquetPartWrite(uri=uri, row_count=frame.height, byte_count=None, compression=compression), False


def _image_batch_manifest_frame(records: list[dict[str, Any]], *, batch_id: str, part_id: str) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "source": str(record.get("source") or "flickr"),
                "flickr_photo_id": str(record.get("flickr_photo_id") or record.get("id") or ""),
                "image_url": str(record.get("image_url") or record.get("image_url_used") or ""),
                "image_cache_status": "external_loader",
                "batch_id": batch_id,
                "part_id": part_id,
            }
            for record in records
        ]
    )


def _work_payload(work_item: dict[str, Any]) -> dict[str, Any]:
    payload = work_item.get("payload", work_item)
    if not isinstance(payload, dict):
        raise ValueError(f"work item {work_item.get('work_key')} has invalid payload")
    return payload


def _record_is_detectable(record: dict[str, Any]) -> bool:
    return bool(str(record.get("flickr_photo_id") or record.get("id") or "") and str(record.get("image_url") or ""))


def _source_record_identity(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": str(record.get("source") or "flickr"),
        "flickr_photo_id": str(record.get("flickr_photo_id") or record.get("id") or ""),
        "image_url": str(record.get("image_url") or ""),
        "source_record_hash": str(record.get("source_record_hash") or ""),
    }


def _settings_value(settings: Any | None, field_name: str) -> Any:
    if settings is None or not hasattr(settings, field_name):
        return None
    return _jsonable_value(getattr(settings, field_name))


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


def _mode_row_count(frame: pl.DataFrame, mode: str) -> int:
    if frame.is_empty() or "ablation_mode" not in frame.columns:
        return 0
    return frame.filter(pl.col("ablation_mode") == mode).height


__all__ = [
    "CloudRollingVisionBatchResult",
    "ROLLING_VISION_ARTIFACT_ORDER",
    "ROLLING_VISION_ARTIFACT_STAGES",
    "ROLLING_VISION_STORAGE_STAGES",
    "ROLLING_VISION_WORK_STAGE",
    "RollingVisionShardCommitResult",
    "RollingVisionWorkPlanResult",
    "cleanup_rolling_batch_temp_dir",
    "commit_rolling_vision_batch_shards",
    "enqueue_rolling_vision_work_from_source_shards",
    "rolling_vision_batch_id",
    "rolling_vision_part_id",
    "rolling_vision_settings_key",
    "rolling_vision_work_item",
    "run_cloud_rolling_vision_batch",
]
