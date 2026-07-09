from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import polars as pl

from biominer.bioclip.candidate_sets import CandidateSet
from biominer.bioclip.classification_modes import (
    DEFAULT_CLASSIFICATION_MODE,
    DEFAULT_FAMILY_TOP_K,
    DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    DEFAULT_SPECIES_RERANK_TOP_K,
    HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    ClassificationMode,
    normalize_classification_mode,
)
from biominer.bioclip.hierarchical_classifier import (
    classify_butterfly_crops_hierarchical_batch,
    hierarchical_result_to_object_score_row,
)
from biominer.bioclip.object_runner import (
    OBJECT_SCORE_OUTPUT_SCHEMA,
    OBJECT_VISUAL_MODES,
    ObjectBioClipScorer,
    empty_object_score_frame,
    _score_detection,
    _score_detection_batch,
    _ensure_columns,
    _scorer_supports_detector_crop_segmentation,
    _segmentation_status,
    _visual_mode_status,
)
from biominer.bioclip.taxonomy_store import ButterflyTaxonomyStore
from biominer.detection.policy import DetectionPolicy, detection_is_bioclip_eligible
from biominer.detection.segmentation import SegmentationUnavailable
from biominer.species.context import SpeciesContext
from biominer.storage.cloud import CloudStorage
from biominer.workstore.base import WorkStore


@dataclass(frozen=True)
class BioClipWorkPlanResult:
    detection_shards_seen: int
    detections_seen: int
    eligible_detections_seen: int
    enqueued_work_items: int
    duplicate_work_items: int


@dataclass(frozen=True)
class CloudBioClipBatchResult:
    frame: pl.DataFrame
    work_items_seen: int
    detections_seen: int
    crops_scored: int
    segmentation_unavailable_count: int
    segmentation_unavailable_reason: str | None
    visual_modes_requested: tuple[str, ...]
    visual_modes_scored: tuple[str, ...]
    visual_mode_status_by_mode: dict[str, str]
    segmentation_status_by_mode: dict[str, str | None]
    segmentation_unavailable_count_by_mode: dict[str, int]
    segmentation_unavailable_reason_by_mode: dict[str, str | None]


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
    classification_mode: ClassificationMode = DEFAULT_CLASSIFICATION_MODE,
    taxonomy_table_version: str | None = None,
    taxonomy_prompt_variant_version: str | None = None,
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
                        classification_mode=classification_mode,
                        taxonomy_table_version=taxonomy_table_version,
                        taxonomy_prompt_variant_version=taxonomy_prompt_variant_version,
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
    classification_mode: ClassificationMode = DEFAULT_CLASSIFICATION_MODE,
    taxonomy_table_version: str | None = None,
    taxonomy_prompt_variant_version: str | None = None,
) -> dict[str, Any]:
    normalized_mode = normalize_classification_mode(classification_mode)
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
        "classification_mode": normalized_mode,
        "taxonomy_table_version": str(taxonomy_table_version or ""),
        "taxonomy_prompt_variant_version": str(taxonomy_prompt_variant_version or ""),
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
        "classification_mode": normalized_mode,
        "taxonomy_table_version": taxonomy_table_version,
        "taxonomy_prompt_variant_version": taxonomy_prompt_variant_version,
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


def run_cloud_bioclip_batch(
    *,
    work_items: list[dict[str, Any]],
    species_context: SpeciesContext,
    candidate_set: CandidateSet,
    scorer: ObjectBioClipScorer,
    geo_prior_table: pl.DataFrame | None = None,
    crop_batch_size: int = 24,
    classification_mode: ClassificationMode = DEFAULT_CLASSIFICATION_MODE,
    family_top_k: int = DEFAULT_FAMILY_TOP_K,
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K,
    taxonomy_store: ButterflyTaxonomyStore | None = None,
) -> CloudBioClipBatchResult:
    if crop_batch_size <= 0:
        raise ValueError("crop_batch_size must be positive")
    classification_mode = normalize_classification_mode(classification_mode)
    if classification_mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION and taxonomy_store is None:
        raise ValueError("taxonomy_store is required for hierarchical_butterfly_classification")
    rows: list[dict[str, Any]] = []
    requested_modes: list[str] = []
    scored_by_mode: dict[str, int] = {}
    unavailable_by_mode: dict[str, int] = {}
    unavailable_reason_by_mode: dict[str, str | None] = {}
    detection_keys: set[tuple[str, str, str]] = set()
    score_items_by_mode: dict[str, list[dict[str, Any]]] = {}
    for item in work_items:
        payload = item.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(f"work item {item.get('work_key')} has invalid payload")
        mode = str(payload.get("ablation_mode") or "detector_crop")
        if mode not in set(OBJECT_VISUAL_MODES):
            raise ValueError(f"unsupported BioCLIP ablation mode: {mode}")
        requested_modes.append(mode)
        detection = detection_from_bioclip_work_item(item)
        detection_keys.add(
            (
                str(detection.get("source") or ""),
                str(detection.get("flickr_photo_id") or ""),
                str(detection.get("detection_id") or ""),
            )
        )
        score_item = {**detection, "ablation_mode": mode}
        if mode == "detector_crop_segmentation" and not _scorer_supports_detector_crop_segmentation(scorer, score_item):
            _mark_unavailable(
                mode,
                reason="detector_masks_missing",
                unavailable_by_mode=unavailable_by_mode,
                unavailable_reason_by_mode=unavailable_reason_by_mode,
            )
            continue
        score_items_by_mode.setdefault(mode, []).append(score_item)

    for mode, score_items in score_items_by_mode.items():
        if classification_mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION:
            if taxonomy_store is None:
                raise ValueError("taxonomy_store is required for hierarchical_butterfly_classification")
            for score_chunk in _chunks(score_items, 1 if mode == "detector_crop_segmentation" else crop_batch_size):
                try:
                    score_rows = _score_hierarchical_cloud_batch(
                        items=score_chunk,
                        scorer=scorer,
                        taxonomy_store=taxonomy_store,
                        family_top_k=family_top_k,
                        species_first_pass_top_k=species_first_pass_top_k,
                        species_rerank_top_k=species_rerank_top_k,
                    )
                except SegmentationUnavailable as exc:
                    if mode != "detector_crop_segmentation":
                        raise
                    _mark_unavailable(
                        mode,
                        reason=str(exc) or "detector_masks_missing",
                        unavailable_by_mode=unavailable_by_mode,
                        unavailable_reason_by_mode=unavailable_reason_by_mode,
                    )
                    continue
                rows.extend(score_rows)
                scored_by_mode[mode] = scored_by_mode.get(mode, 0) + len(score_rows)
            continue
        if mode == "detector_crop_segmentation":
            for score_item in score_items:
                try:
                    score_row = _score_detection(
                        item=score_item,
                        context=species_context,
                        candidate_set=candidate_set,
                        scorer=scorer,
                        ablation_mode=mode,  # type: ignore[arg-type]
                        geo_prior_table=geo_prior_table,
                        classification_mode=classification_mode,
                        family_top_k=family_top_k,
                        species_first_pass_top_k=species_first_pass_top_k,
                        species_rerank_top_k=species_rerank_top_k,
                    )
                except SegmentationUnavailable as exc:
                    _mark_unavailable(
                        mode,
                        reason=str(exc) or "detector_masks_missing",
                        unavailable_by_mode=unavailable_by_mode,
                        unavailable_reason_by_mode=unavailable_reason_by_mode,
                    )
                    continue
                rows.append(score_row)
                scored_by_mode[mode] = scored_by_mode.get(mode, 0) + 1
            continue
        for score_chunk in _chunks(score_items, crop_batch_size):
            try:
                score_rows = _score_detection_batch(
                    items=score_chunk,
                    context=species_context,
                    candidate_set=candidate_set,
                    scorer=scorer,
                    ablation_mode=mode,  # type: ignore[arg-type]
                    geo_prior_table=geo_prior_table,
                    classification_mode=classification_mode,
                    family_top_k=family_top_k,
                    species_first_pass_top_k=species_first_pass_top_k,
                    species_rerank_top_k=species_rerank_top_k,
                )
            except SegmentationUnavailable:
                raise
            rows.extend(score_rows)
            scored_by_mode[mode] = scored_by_mode.get(mode, 0) + len(score_rows)
    frame = _ensure_columns(pl.DataFrame(rows), OBJECT_SCORE_OUTPUT_SCHEMA) if rows else empty_object_score_frame()
    modes_requested = tuple(_unique(requested_modes))
    modes_scored = tuple(sorted(scored_by_mode))
    visual_status = {
        mode: _visual_mode_status(
            mode=mode,  # type: ignore[arg-type]
            crops_scored=scored_by_mode.get(mode, 0),
            unavailable_count=unavailable_by_mode.get(mode, 0),
        )
        for mode in modes_requested
    }
    segmentation_status = {
        mode: _segmentation_status(
            mode=mode,  # type: ignore[arg-type]
            crops_scored=scored_by_mode.get(mode, 0),
            unavailable_count=unavailable_by_mode.get(mode, 0),
        )
        for mode in modes_requested
    }
    unavailable_total = sum(unavailable_by_mode.values())
    first_unavailable_reason = next((reason for reason in unavailable_reason_by_mode.values() if reason), None)
    return CloudBioClipBatchResult(
        frame=frame,
        work_items_seen=len(work_items),
        detections_seen=len(detection_keys),
        crops_scored=frame.height,
        segmentation_unavailable_count=unavailable_total,
        segmentation_unavailable_reason=first_unavailable_reason,
        visual_modes_requested=modes_requested,
        visual_modes_scored=modes_scored,
        visual_mode_status_by_mode=visual_status,
        segmentation_status_by_mode=segmentation_status,
        segmentation_unavailable_count_by_mode=unavailable_by_mode,
        segmentation_unavailable_reason_by_mode=unavailable_reason_by_mode,
    )


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _score_hierarchical_cloud_batch(
    *,
    items: list[dict[str, Any]],
    scorer: ObjectBioClipScorer,
    taxonomy_store: ButterflyTaxonomyStore,
    family_top_k: int,
    species_first_pass_top_k: int,
    species_rerank_top_k: int,
) -> list[dict[str, Any]]:
    results = classify_butterfly_crops_hierarchical_batch(
        items=items,
        scorer=scorer,
        taxonomy_store=taxonomy_store,
        family_top_k=family_top_k,
        species_first_pass_top_k=species_first_pass_top_k,
        species_rerank_top_k=species_rerank_top_k,
    )
    return [
        hierarchical_result_to_object_score_row(
            item=item,
            result=result,
            scorer=scorer,
            family_top_k=family_top_k,
            species_first_pass_top_k=species_first_pass_top_k,
            species_rerank_top_k=species_rerank_top_k,
        )
        for item, result in zip(items, results, strict=True)
    ]


def bioclip_score_batch_id(work_items: list[dict[str, Any]]) -> str:
    work_keys = [str(item.get("work_key") or "") for item in work_items]
    return _stable_hash({"work_keys": work_keys})


def _mark_unavailable(
    mode: str,
    *,
    reason: str,
    unavailable_by_mode: dict[str, int],
    unavailable_reason_by_mode: dict[str, str | None],
) -> None:
    unavailable_by_mode[mode] = unavailable_by_mode.get(mode, 0) + 1
    unavailable_reason_by_mode.setdefault(mode, reason)


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


def _unique(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return tuple(output)


__all__ = [
    "BioClipWorkPlanResult",
    "CloudBioClipBatchResult",
    "bioclip_score_batch_id",
    "bioclip_score_work_item",
    "detection_from_bioclip_work_item",
    "enqueue_bioclip_work_from_detection_shards",
    "run_cloud_bioclip_batch",
]
