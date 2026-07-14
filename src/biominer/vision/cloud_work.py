from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from shutil import rmtree
from typing import Any

import polars as pl

from biominer.bioclip.candidate_sets import CandidateSet
from biominer.bioclip.cascade_contract import validate_cascade_work_identity
from biominer.bioclip.classification_modes import (
    DEFAULT_CLASSIFICATION_MODE,
    DEFAULT_RANK_BEAM_WIDTH,
    DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    DEFAULT_SPECIES_RERANK_TOP_K,
    HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    TARGET_FAMILY_REPORT_TOP_K,
    ClassificationMode,
    normalize_classification_mode,
)
from biominer.bioclip.cloud_work import bioclip_score_work_item, run_cloud_bioclip_batch
from biominer.bioclip.object_runner import ObjectBioClipScorer
from biominer.bioclip.path_taxonomy_store import PathTaxonomyStore
from biominer.bioclip.taxonomy_embedding_cache import TaxonomyTextEmbeddingIndex
from biominer.detection.cloud_work import CloudDetectionBatchResult, detection_work_item, run_cloud_detection_batch
from biominer.detection.detector_base import ObjectDetector
from biominer.detection.pipeline import ImageLoader
from biominer.detection.policy import DetectionPolicy
from biominer.detection.routing import DetectionRoutingPolicy
from biominer.registry.classification_v3 import CLASSIFICATION_V3_VERSION
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
class CloudRollingDetectionBatch:
    work_item: dict[str, Any]
    payload: dict[str, Any]
    batch_id: str
    part_id: str
    canonical: pl.DataFrame
    manifest: pl.DataFrame
    detection_result: CloudDetectionBatchResult


@dataclass(frozen=True)
class RollingVisionShardCommitResult:
    output_uris: dict[str, str]
    rows_by_artifact: dict[str, int]
    parts_written: int
    parts_reused: int


@dataclass(frozen=True)
class RollingVisionPipelineResult:
    commit_results: tuple[Any, ...]
    batches_started: int
    batches_committed: int
    buffer_capacity: int = 1


class RollingVisionPipelineError(RuntimeError):
    def __init__(self, *, phase: str, work_item: dict[str, Any], cause: BaseException) -> None:
        self.phase = phase
        self.work_item = work_item
        self.cause = cause
        work_key = str(work_item.get("work_key") or "unknown")
        super().__init__(f"{phase} failed for {work_key}: {cause}")


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
    detector: dict[str, Any] | None = None,
    vision_settings: Any | None = None,
    bioclip_gate_mode: str = "routed_visual_domain",
    score_no_detection_whole_image: bool = False,
    supported_comparison_routes: tuple[str, ...] = ("adult_field",),
    bioclip_model: dict[str, str] | None = None,
    candidate_set_id: str = "",
    classification_mode: ClassificationMode = DEFAULT_CLASSIFICATION_MODE,
    taxonomy_table_version: str | None = None,
    taxonomy_prompt_variant_version: str | None = None,
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K,
    cascade_identity: dict[str, Any] | None = None,
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
    remaining = None if limit is None or limit <= 0 else int(limit)
    pending_records: list[dict[str, Any]] = []
    pending_source_uris: list[str] = []
    source_records_seen = 0
    source_shards_seen = 0
    work_items_seen = 0
    inserted = 0
    batch_index = 0
    settings_key = rolling_vision_settings_key(
        detector=detector or {},
        vision_settings=vision_settings,
        bioclip_gate_mode=bioclip_gate_mode,
        score_no_detection_whole_image=score_no_detection_whole_image,
        supported_comparison_routes=supported_comparison_routes,
        bioclip_model=bioclip_model or {},
        candidate_set_id=candidate_set_id,
        classification_mode=classification_mode,
        taxonomy_table_version=taxonomy_table_version,
        taxonomy_prompt_variant_version=taxonomy_prompt_variant_version,
        species_first_pass_top_k=species_first_pass_top_k,
        species_rerank_top_k=species_rerank_top_k,
        cascade_identity=cascade_identity,
    )

    def flush_pending() -> None:
        nonlocal batch_index, inserted, work_items_seen
        if not pending_records:
            return
        item = rolling_vision_work_item(
            list(pending_records),
            run_id=run_id,
            batch_index=batch_index,
            vision_batch_rows=vision_batch_rows,
            source_shard_uris=list(pending_source_uris),
            settings_key=settings_key,
        )
        inserted += workstore.enqueue_work(job_name, registry_version, [item], stage=vision_stage)
        work_items_seen += 1
        batch_index += 1
        pending_records.clear()
        pending_source_uris.clear()

    for shard in shards:
        if remaining == 0:
            break
        source_shards_seen += 1
        source_shard_uri = str(shard["uri"])
        for frame in storage.iter_parquet_batches(source_shard_uri, batch_size=vision_batch_rows):
            for row in frame.iter_rows(named=True):
                if remaining == 0:
                    break
                record = dict(row)
                if not _record_is_detectable(record):
                    continue
                pending_records.append(_jsonable_record(record))
                pending_source_uris.append(source_shard_uri)
                source_records_seen += 1
                if remaining is not None:
                    remaining -= 1
                if len(pending_records) == vision_batch_rows:
                    flush_pending()
            if remaining == 0:
                break
    flush_pending()
    return RollingVisionWorkPlanResult(
        source_shards_seen=source_shards_seen,
        source_records_seen=source_records_seen,
        batches_planned=work_items_seen,
        enqueued_work_items=inserted,
        duplicate_work_items=work_items_seen - inserted,
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
    detector: dict[str, Any],
    vision_settings: Any | None,
    bioclip_gate_mode: str = "routed_visual_domain",
    score_no_detection_whole_image: bool = False,
    supported_comparison_routes: tuple[str, ...] = ("adult_field",),
    bioclip_model: dict[str, str],
    candidate_set_id: str,
    classification_mode: ClassificationMode = DEFAULT_CLASSIFICATION_MODE,
    taxonomy_table_version: str | None = None,
    taxonomy_prompt_variant_version: str | None = None,
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K,
    cascade_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_cascade_identity = (
        validate_cascade_work_identity(cascade_identity)
        if cascade_identity is not None
        else None
    )
    normalized_mode = normalize_classification_mode(classification_mode)
    if (
        normalized_cascade_identity is not None
        and normalized_mode != HIERARCHICAL_BUTTERFLY_CLASSIFICATION
    ):
        raise ValueError("cascade identity is valid only for hierarchical classification")
    if (
        normalized_mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
        and str(taxonomy_table_version or "") == CLASSIFICATION_V3_VERSION
        and normalized_cascade_identity is None
    ):
        raise ValueError("classification-v3 rolling work requires cascade identity")
    routing_policy = _routing_policy_settings(vision_settings)
    settings = {
        "detector": {
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
            "yolo_imgsz": _settings_value(vision_settings, "yolo_imgsz"),
            "yolo_conf": _settings_value(vision_settings, "yolo_conf"),
            "yolo_iou": _settings_value(vision_settings, "yolo_iou"),
            "yolo_max_det": _settings_value(vision_settings, "yolo_max_det"),
            "routing_policy": routing_policy,
        },
        "crop": {
            "crop_padding_ratio": _settings_value(vision_settings, "crop_padding_ratio"),
            "crop_target_px": _settings_value(vision_settings, "crop_target_px"),
        },
        "bioclip_gate": {
            "mode": str(bioclip_gate_mode),
            "score_no_detection_whole_image": bool(score_no_detection_whole_image),
            "supported_comparison_routes": list(
                dict.fromkeys(str(route) for route in supported_comparison_routes)
            ),
        },
        "bioclip_model": {
            "model_id": str(bioclip_model.get("model_id") or ""),
            "model_version": str(bioclip_model.get("model_version") or ""),
            "checkpoint": str(bioclip_model.get("checkpoint") or ""),
        },
        "candidate_set_id": str(candidate_set_id or ""),
        "classification_mode": normalized_mode,
        "taxonomy_table_version": str(taxonomy_table_version or ""),
        "taxonomy_prompt_variant_version": str(taxonomy_prompt_variant_version or ""),
        "cascade_identity": normalized_cascade_identity,
    }
    if normalized_cascade_identity is None:
        settings["top_k_settings"] = {
            "target_family_report_top_k": TARGET_FAMILY_REPORT_TOP_K,
            "species_first_pass_top_k": int(species_first_pass_top_k),
            "species_rerank_top_k": int(species_rerank_top_k),
        }
    return settings


def detect_cloud_rolling_vision_batch(
    *,
    work_item: dict[str, Any],
    detector: ObjectDetector,
    image_loader: ImageLoader,
    detection_policy: DetectionPolicy,
    detector_batch_size: int = 16,
    adaptive_detector_batching: bool = False,
    min_detector_batch_size: int = 1,
) -> CloudRollingDetectionBatch:
    payload = _work_payload(work_item)
    _validate_rolling_vision_work_item(work_item, payload)
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
                    "prompt_classes": list(
                        getattr(detector, "prompt_classes", ())
                    ),
                    "prompt_set_fingerprint": str(
                        getattr(detector, "prompt_set_fingerprint", "") or ""
                    ),
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
    return CloudRollingDetectionBatch(
        work_item=work_item,
        payload=payload,
        batch_id=batch_id,
        part_id=part_id,
        canonical=canonical,
        manifest=manifest,
        detection_result=detection_result,
    )


def score_cloud_rolling_detection_batch(
    *,
    batch: CloudRollingDetectionBatch,
    image_loader: ImageLoader,
    species_context: SpeciesContext,
    candidate_set: CandidateSet,
    scorer: ObjectBioClipScorer,
    bioclip_gate_policy: BioClipGatePolicy,
    temp_dir: str | Path,
    crop_padding_ratio: float = 0.12,
    crop_target_px: int = 336,
    bioclip_batch_size: int = 24,
    adaptive_bioclip_batching: bool = False,
    min_bioclip_batch_size: int = 1,
    classification_mode: ClassificationMode = DEFAULT_CLASSIFICATION_MODE,
    taxonomy_table_version: str | None = None,
    taxonomy_prompt_variant_version: str | None = None,
    rank_beam_width: int = DEFAULT_RANK_BEAM_WIDTH,
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K,
    path_taxonomy_store: PathTaxonomyStore | None = None,
    taxonomy_text_embedding_index: TaxonomyTextEmbeddingIndex | None = None,
    cascade_identity: dict[str, Any] | None = None,
) -> CloudRollingVisionBatchResult:
    score_inputs = materialize_bioclip_score_inputs(
        canonical_records=batch.canonical,
        detections=batch.detection_result.frame,
        image_loader=image_loader,
        temp_dir=temp_dir,
        gate_policy=bioclip_gate_policy,
        crop_padding_ratio=crop_padding_ratio,
        crop_target_px=crop_target_px,
        batch_id=batch.batch_id,
        part_id=batch.part_id,
    )
    model = {
        "model_id": scorer.model_id,
        "model_version": scorer.model_version,
        "checkpoint": scorer.model_checkpoint,
    }
    score_work_items: list[dict[str, Any]] = []
    for item in score_inputs.items:
        payload = bioclip_score_work_item(
            dict(item),
            run_id=str(batch.payload.get("run_id") or ""),
            detection_shard_uri=(
                f"rolling://{batch.batch_id}/object_detections/{batch.part_id}"
            ),
            model=model,
            candidate_set_id=candidate_set.candidate_set_id,
            ablation_mode=str(
                item.get("visual_input_kind")
                or item.get("ablation_mode")
                or "detector_crop"
            ),
            classification_mode=classification_mode,
            taxonomy_table_version=taxonomy_table_version,
            taxonomy_prompt_variant_version=taxonomy_prompt_variant_version,
            species_first_pass_top_k=species_first_pass_top_k,
            species_rerank_top_k=species_rerank_top_k,
            cascade_identity=cascade_identity,
        )
        score_work_items.append({"work_key": payload["work_key"], "payload": payload})
    score_result = run_cloud_bioclip_batch(
        work_items=score_work_items,
        species_context=species_context,
        candidate_set=candidate_set,
        scorer=scorer,
        crop_batch_size=bioclip_batch_size,
        adaptive_batching=adaptive_bioclip_batching,
        min_crop_batch_size=min_bioclip_batch_size,
        classification_mode=classification_mode,
        rank_beam_width=rank_beam_width,
        species_first_pass_top_k=species_first_pass_top_k,
        species_rerank_top_k=species_rerank_top_k,
        path_taxonomy_store=path_taxonomy_store,
        taxonomy_text_embedding_index=taxonomy_text_embedding_index,
    )
    joined, summary = build_object_evidence_frames(
        canonical_source_records=batch.canonical,
        object_detections=batch.detection_result.frame,
        object_scores=score_result.frame,
        species_context=species_context,
    )
    return CloudRollingVisionBatchResult(
        batch_id=batch.batch_id,
        part_id=batch.part_id,
        frames={
            "image_batch_manifest": batch.manifest,
            "object_detections": batch.detection_result.frame,
            "bioclip_score_inputs": score_inputs.frame,
            "object_bioclip_scores": score_result.frame,
            "object_evidence_joined": joined,
            "photo_evidence_summary": summary,
        },
        metrics={
            "records_seen": batch.detection_result.records_seen,
            "images_loaded": batch.detection_result.images_loaded,
            "image_failures": batch.detection_result.image_failures,
            "detections_written": batch.detection_result.detections_written,
            "crops_created": batch.detection_result.crops_created,
            "score_inputs": score_inputs.frame.height,
            "objects_scored": score_result.crops_scored,
            "whole_images_scored": _mode_row_count(score_result.frame, "whole_image"),
            "detector_crops_scored": _mode_row_count(score_result.frame, "detector_crop"),
            "segmentation_crops_scored": _mode_row_count(score_result.frame, "detector_crop_segmentation"),
            "object_evidence_rows": joined.height,
            "photo_summary_rows": summary.height,
            "detector_batch_retries": batch.detection_result.detector_batch_retries,
            "bioclip_batch_retries": score_result.bioclip_batch_retries,
        },
        materialized_score_inputs=score_inputs,
    )


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
    rank_beam_width: int = DEFAULT_RANK_BEAM_WIDTH,
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K,
    path_taxonomy_store: PathTaxonomyStore | None = None,
    taxonomy_text_embedding_index: TaxonomyTextEmbeddingIndex | None = None,
    cascade_identity: dict[str, Any] | None = None,
) -> CloudRollingVisionBatchResult:
    detected = detect_cloud_rolling_vision_batch(
        work_item=work_item,
        detector=detector,
        image_loader=image_loader,
        detection_policy=detection_policy,
        detector_batch_size=detector_batch_size,
        adaptive_detector_batching=adaptive_detector_batching,
        min_detector_batch_size=min_detector_batch_size,
    )
    return score_cloud_rolling_detection_batch(
        batch=detected,
        image_loader=image_loader,
        species_context=species_context,
        candidate_set=candidate_set,
        scorer=scorer,
        bioclip_gate_policy=bioclip_gate_policy,
        temp_dir=temp_dir,
        crop_padding_ratio=crop_padding_ratio,
        crop_target_px=crop_target_px,
        bioclip_batch_size=bioclip_batch_size,
        adaptive_bioclip_batching=adaptive_bioclip_batching,
        min_bioclip_batch_size=min_bioclip_batch_size,
        classification_mode=classification_mode,
        taxonomy_table_version=taxonomy_table_version,
        taxonomy_prompt_variant_version=taxonomy_prompt_variant_version,
        rank_beam_width=rank_beam_width,
        species_first_pass_top_k=species_first_pass_top_k,
        species_rerank_top_k=species_rerank_top_k,
        path_taxonomy_store=path_taxonomy_store,
        taxonomy_text_embedding_index=taxonomy_text_embedding_index,
        cascade_identity=cascade_identity,
    )


def run_bounded_cloud_rolling_pipeline(
    work_items: list[dict[str, Any]],
    *,
    detect: Any,
    score: Any,
    commit: Any,
) -> RollingVisionPipelineResult:
    """Run one-slot YOLO -> BioCLIP -> commit buffers with ordered main-thread writes.

    Detection of batch N+1 overlaps scoring of batch N. Only one completed
    detection batch and one scored batch can be resident beyond the active
    calls, and ``commit`` always runs on the caller thread in input order.
    """
    items = list(work_items)
    if not items:
        return RollingVisionPipelineResult(commit_results=(), batches_started=0, batches_committed=0)

    committed: list[Any] = []
    detection_future: Future[Any] | None = None
    detection_item: dict[str, Any] | None = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="biominer-yolo") as detector_pool, ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="biominer-bioclip",
    ) as scorer_pool:
        detection_item = items[0]
        detection_future = detector_pool.submit(detect, detection_item)
        for index, item in enumerate(items):
            try:
                detected = detection_future.result()
            except BaseException as exc:
                raise RollingVisionPipelineError(phase="detect", work_item=detection_item or item, cause=exc) from exc

            next_item = items[index + 1] if index + 1 < len(items) else None
            if next_item is not None:
                detection_item = next_item
                detection_future = detector_pool.submit(detect, next_item)

            score_future = scorer_pool.submit(score, detected)
            try:
                scored = score_future.result()
            except BaseException as exc:
                raise RollingVisionPipelineError(phase="score", work_item=item, cause=exc) from exc
            try:
                committed.append(commit(item, scored))
            except BaseException as exc:
                raise RollingVisionPipelineError(phase="commit", work_item=item, cause=exc) from exc

    return RollingVisionPipelineResult(
        commit_results=tuple(committed),
        batches_started=len(items),
        batches_committed=len(committed),
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


def _validate_rolling_vision_work_item(
    work_item: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    settings = payload.get("settings_key")
    if not isinstance(settings, dict):
        raise ValueError("rolling vision work item has no settings_key mapping")
    cascade_identity = settings.get("cascade_identity")
    if cascade_identity is not None:
        if not isinstance(cascade_identity, dict):
            raise ValueError("rolling vision cascade_identity must be a mapping")
        validate_cascade_work_identity(cascade_identity)
    if (
        normalize_classification_mode(settings.get("classification_mode"))
        == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
        and str(settings.get("taxonomy_table_version") or "")
        == CLASSIFICATION_V3_VERSION
        and cascade_identity is None
    ):
        raise ValueError("classification-v3 rolling work is missing cascade_identity")
    records = [dict(row) for row in payload.get("source_records") or [] if isinstance(row, dict)]
    record_identities = [_source_record_identity(record) for record in records]
    if payload.get("source_record_identities") != record_identities:
        raise ValueError("rolling vision source record identities do not match payload")
    key_payload = {
        "run_id": str(payload.get("run_id") or ""),
        "batch_id": str(payload.get("batch_id") or ""),
        "batch_index": int(payload.get("batch_index") or 0),
        "vision_batch_rows": int(payload.get("vision_batch_rows") or 0),
        "source_records": record_identities,
        "settings": settings,
    }
    expected = f"{key_payload['run_id']}:rolling-vision:{_stable_hash(key_payload)}"
    actual = str(work_item.get("work_key") or payload.get("work_key") or "")
    if actual != expected:
        raise ValueError("rolling vision work key does not match immutable payload identity")


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


def _routing_policy_settings(settings: Any | None) -> dict[str, Any]:
    defaults = DetectionRoutingPolicy()

    def configured(field_name: str) -> Any:
        value = _settings_value(settings, field_name)
        return getattr(defaults, field_name) if value is None else value

    policy = DetectionRoutingPolicy(
        version=defaults.version,
        possible_adult_route_enabled=bool(
            configured("possible_adult_route_enabled")
        ),
        possible_adult_route_threshold=float(
            configured("possible_adult_route_threshold")
        ),
        ambiguous_insect_review_enabled=bool(
            configured("ambiguous_insect_review_enabled")
        ),
        ambiguous_insect_review_threshold=float(
            configured("ambiguous_insect_review_threshold")
        ),
    )
    return {
        "version": policy.version,
        "fingerprint": policy.fingerprint,
        "possible_adult_route_enabled": policy.possible_adult_route_enabled,
        "possible_adult_route_threshold": policy.possible_adult_route_threshold,
        "ambiguous_insect_review_enabled": policy.ambiguous_insect_review_enabled,
        "ambiguous_insect_review_threshold": (
            policy.ambiguous_insect_review_threshold
        ),
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


def _mode_row_count(frame: pl.DataFrame, mode: str) -> int:
    if frame.is_empty() or "ablation_mode" not in frame.columns:
        return 0
    return frame.filter(pl.col("ablation_mode") == mode).height


__all__ = [
    "CloudRollingDetectionBatch",
    "CloudRollingVisionBatchResult",
    "ROLLING_VISION_ARTIFACT_ORDER",
    "ROLLING_VISION_ARTIFACT_STAGES",
    "ROLLING_VISION_STORAGE_STAGES",
    "ROLLING_VISION_WORK_STAGE",
    "RollingVisionShardCommitResult",
    "RollingVisionPipelineError",
    "RollingVisionPipelineResult",
    "RollingVisionWorkPlanResult",
    "cleanup_rolling_batch_temp_dir",
    "commit_rolling_vision_batch_shards",
    "detect_cloud_rolling_vision_batch",
    "enqueue_rolling_vision_work_from_source_shards",
    "rolling_vision_batch_id",
    "rolling_vision_part_id",
    "rolling_vision_settings_key",
    "rolling_vision_work_item",
    "run_cloud_rolling_vision_batch",
    "run_bounded_cloud_rolling_pipeline",
    "score_cloud_rolling_detection_batch",
]
