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
from biominer.bioclip.five_rank_classifier import (
    classify_five_rank_crops_batch,
    five_rank_result_to_object_score_row,
)
from biominer.bioclip.five_rank_store import FiveRankTaxonomyStore
from biominer.bioclip.object_runner import (
    OBJECT_SCORE_OUTPUT_SCHEMA,
    OBJECT_VISUAL_MODES,
    ObjectBioClipScorer,
    empty_object_score_frame,
    _score_detection,
    _score_detection_batch,
    _ensure_columns,
    _next_bioclip_batch_size,
    _should_retry_bioclip_batch,
    _scorer_supports_detector_crop_segmentation,
    _segmentation_status,
    _visual_mode_status,
)
from biominer.bioclip.taxonomy_store import ButterflyTaxonomyStore
from biominer.detection.policy import DetectionPolicy
from biominer.detection.segmentation import SegmentationUnavailable
from biominer.registry.classification_table import (
    CLASSIFICATION_TABLE_VERSION,
    PROMPT_VARIANT_VERSION,
)
from biominer.species.context import SpeciesContext
from biominer.storage.cloud import CloudStorage
from biominer.storage.parquet import DEFAULT_PARQUET_READ_BATCH_SIZE
from biominer.vision.gates import BioClipGatePolicy, ScoreInputDecision, bioclip_score_input_decision
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
    adaptive_batching_enabled: bool = False
    bioclip_batch_retries: int = 0
    bioclip_batch_size_initial: int = 24
    bioclip_batch_size_final: int = 24
    bioclip_batch_size_min: int = 1


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
    bioclip_gate_policy: BioClipGatePolicy | None = None,
    limit: int | None = None,
    classification_mode: ClassificationMode = DEFAULT_CLASSIFICATION_MODE,
    taxonomy_table_version: str | None = None,
    taxonomy_prompt_variant_version: str | None = None,
    family_top_k: int = DEFAULT_FAMILY_TOP_K,
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K,
    read_batch_size: int = DEFAULT_PARQUET_READ_BATCH_SIZE,
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
    gate_policy = _active_bioclip_gate_policy(
        detection_policy=detection_policy,
        bioclip_gate_policy=bioclip_gate_policy,
    )
    detections_seen = 0
    eligible_seen = 0
    attempted = 0
    inserted = 0
    remaining = None if limit is None or limit <= 0 else int(limit)
    for shard in shards:
        if remaining == 0:
            break
        detection_shard_uri = str(shard["uri"])
        for frame in storage.iter_parquet_batches(detection_shard_uri, batch_size=read_batch_size):
            batch_items: list[dict[str, Any]] = []
            for row in frame.iter_rows(named=True):
                detections_seen += 1
                detection = dict(row)
                gate_decision = bioclip_score_input_decision(detection, gate_policy)
                if not _detection_is_scoreable(detection, gate_decision=gate_decision):
                    continue
                eligible_seen += 1
                score_modes = ("whole_image",) if gate_decision.visual_input_kind == "whole_image" else modes
                for mode in score_modes:
                    if remaining == 0:
                        break
                    batch_items.append(
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
                            family_top_k=family_top_k,
                            species_first_pass_top_k=species_first_pass_top_k,
                            species_rerank_top_k=species_rerank_top_k,
                            gate_decision=gate_decision,
                        )
                    )
                    if remaining is not None:
                        remaining -= 1
                if remaining == 0:
                    break
            if batch_items:
                attempted += len(batch_items)
                inserted += workstore.enqueue_work(job_name, registry_version, batch_items, stage=score_stage)
            if remaining == 0:
                break
    return BioClipWorkPlanResult(
        detection_shards_seen=len(shards),
        detections_seen=detections_seen,
        eligible_detections_seen=eligible_seen,
        enqueued_work_items=inserted,
        duplicate_work_items=attempted - inserted,
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
    family_top_k: int = DEFAULT_FAMILY_TOP_K,
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K,
    gate_decision: ScoreInputDecision | None = None,
) -> dict[str, Any]:
    normalized_mode = normalize_classification_mode(classification_mode)
    active_gate_decision = gate_decision or bioclip_score_input_decision(
        detection,
        BioClipGatePolicy.legacy_butterfly_like_only(),
    )
    top_k_settings = {
        "family_top_k": int(family_top_k),
        "species_first_pass_top_k": int(species_first_pass_top_k),
        "species_rerank_top_k": int(species_rerank_top_k),
    }
    crop_identity = {
        "crop_hash": str(detection.get("crop_hash") or ""),
        "crop_padding_ratio": _jsonable_value(detection.get("crop_padding_ratio")),
        "crop_width": _jsonable_value(detection.get("crop_width")),
        "crop_height": _jsonable_value(detection.get("crop_height")),
    }
    key_payload = {
        "run_id": run_id,
        "source": str(detection.get("source") or ""),
        "flickr_photo_id": str(detection.get("flickr_photo_id") or ""),
        "detection_id": str(detection.get("detection_id") or ""),
        "crop": crop_identity,
        "model_id": str(model.get("model_id") or ""),
        "model_version": str(model.get("model_version") or ""),
        "model_checkpoint": str(model.get("checkpoint") or ""),
        "candidate_set_id": str(candidate_set_id or ""),
        "classification_mode": normalized_mode,
        "taxonomy_table_version": str(taxonomy_table_version or ""),
        "taxonomy_prompt_variant_version": str(taxonomy_prompt_variant_version or ""),
        "top_k_settings": top_k_settings,
        "ablation_mode": str(ablation_mode or ""),
        "bioclip_gate_mode": active_gate_decision.bioclip_gate_mode,
        "bioclip_gate_decision": active_gate_decision.bioclip_gate_decision,
        "bioclip_gate_reason": active_gate_decision.bioclip_gate_reason,
    }
    return {
        "work_key": f"{run_id}:bioclip:{_stable_hash(key_payload)}",
        "run_id": run_id,
        "source": key_payload["source"],
        "flickr_photo_id": key_payload["flickr_photo_id"],
        "detection_id": key_payload["detection_id"],
        "crop_hash": crop_identity["crop_hash"],
        "detection_shard_uri": detection_shard_uri,
        "detection": _jsonable_record(detection),
        "model": dict(model),
        "candidate_set_id": candidate_set_id,
        "classification_mode": normalized_mode,
        "taxonomy_table_version": taxonomy_table_version,
        "taxonomy_prompt_variant_version": taxonomy_prompt_variant_version,
        "top_k_settings": top_k_settings,
        "ablation_mode": ablation_mode,
        "bioclip_gate_mode": active_gate_decision.bioclip_gate_mode,
        "bioclip_gate_decision": active_gate_decision.bioclip_gate_decision,
        "bioclip_gate_reason": active_gate_decision.bioclip_gate_reason,
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
    adaptive_batching: bool = False,
    min_crop_batch_size: int = 1,
    classification_mode: ClassificationMode = DEFAULT_CLASSIFICATION_MODE,
    family_top_k: int = DEFAULT_FAMILY_TOP_K,
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K,
    taxonomy_store: ButterflyTaxonomyStore | None = None,
    taxonomy_text_embedding_cache: pl.DataFrame | None = None,
) -> CloudBioClipBatchResult:
    if crop_batch_size <= 0:
        raise ValueError("crop_batch_size must be positive")
    if min_crop_batch_size <= 0:
        raise ValueError("min_crop_batch_size must be positive")
    if min_crop_batch_size > crop_batch_size:
        raise ValueError("min_crop_batch_size must be <= crop_batch_size")
    classification_mode = normalize_classification_mode(classification_mode)
    _validate_cloud_bioclip_work_contract(
        work_items=work_items,
        classification_mode=classification_mode,
        family_top_k=family_top_k,
        species_first_pass_top_k=species_first_pass_top_k,
        species_rerank_top_k=species_rerank_top_k,
        taxonomy_store=taxonomy_store,
    )
    if classification_mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION and taxonomy_store is None:
        raise ValueError("taxonomy_store is required for hierarchical_butterfly_classification")
    rows: list[dict[str, Any]] = []
    requested_modes: list[str] = []
    scored_by_mode: dict[str, int] = {}
    unavailable_by_mode: dict[str, int] = {}
    unavailable_reason_by_mode: dict[str, str | None] = {}
    detection_keys: set[tuple[str, str, str]] = set()
    score_items_by_mode: dict[str, list[dict[str, Any]]] = {}
    current_crop_batch_size = crop_batch_size
    bioclip_batch_retries = 0
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
        score_item = {
            **detection,
            "ablation_mode": mode,
            "visual_input_kind": mode,
            "bioclip_gate_mode": payload.get("bioclip_gate_mode"),
            "bioclip_gate_reason": payload.get("bioclip_gate_reason"),
        }
        if mode == "detector_crop_segmentation" and not _scorer_supports_detector_crop_segmentation(scorer, score_item):
            _mark_unavailable(
                mode,
                reason="detector_masks_missing",
                unavailable_by_mode=unavailable_by_mode,
                unavailable_reason_by_mode=unavailable_reason_by_mode,
            )
            continue
        score_items_by_mode.setdefault(mode, []).append(score_item)

    def score_mode_batches(
        *,
        mode: str,
        score_items: list[dict[str, Any]],
        score_batch: Any,
    ) -> None:
        nonlocal bioclip_batch_retries, current_crop_batch_size
        initial_size = 1 if mode == "detector_crop_segmentation" else current_crop_batch_size
        pending = _chunks(score_items, initial_size)
        while pending:
            score_chunk = pending.pop(0)
            try:
                score_rows = score_batch(score_chunk)
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
            except RuntimeError as exc:
                if mode == "detector_crop_segmentation" or not _should_retry_bioclip_batch(
                    exc,
                    adaptive_batching=adaptive_batching,
                    batch_size=len(score_chunk),
                    current_batch_size=current_crop_batch_size,
                    min_batch_size=min_crop_batch_size,
                ):
                    raise
                current_crop_batch_size = _next_bioclip_batch_size(
                    current_batch_size=current_crop_batch_size,
                    failed_batch_size=len(score_chunk),
                    min_batch_size=min_crop_batch_size,
                )
                bioclip_batch_retries += 1
                pending = _chunks(score_chunk, current_crop_batch_size) + pending
                continue
            rows.extend(score_rows)
            scored_by_mode[mode] = scored_by_mode.get(mode, 0) + len(score_rows)

    for mode, score_items in score_items_by_mode.items():
        if classification_mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION:
            if taxonomy_store is None:
                raise ValueError("taxonomy_store is required for hierarchical_butterfly_classification")
            score_mode_batches(
                mode=mode,
                score_items=score_items,
                score_batch=lambda score_chunk: _score_hierarchical_cloud_batch(
                    items=score_chunk,
                    scorer=scorer,
                    taxonomy_store=taxonomy_store,
                    family_top_k=family_top_k,
                    species_first_pass_top_k=species_first_pass_top_k,
                    species_rerank_top_k=species_rerank_top_k,
                    taxonomy_text_embedding_cache=taxonomy_text_embedding_cache,
                ),
            )
            continue
        if mode == "detector_crop_segmentation":
            score_mode_batches(
                mode=mode,
                score_items=score_items,
                score_batch=lambda score_chunk: [
                    _score_detection(
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
                    for score_item in score_chunk
                ],
            )
            continue
        score_mode_batches(
            mode=mode,
            score_items=score_items,
            score_batch=lambda score_chunk: _score_detection_batch(
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
            ),
        )
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
        adaptive_batching_enabled=bool(adaptive_batching),
        bioclip_batch_retries=bioclip_batch_retries,
        bioclip_batch_size_initial=crop_batch_size,
        bioclip_batch_size_final=current_crop_batch_size,
        bioclip_batch_size_min=min_crop_batch_size,
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


def _validate_cloud_bioclip_work_contract(
    *,
    work_items: list[dict[str, Any]],
    classification_mode: ClassificationMode,
    family_top_k: int,
    species_first_pass_top_k: int,
    species_rerank_top_k: int,
    taxonomy_store: ButterflyTaxonomyStore | None,
) -> None:
    expected_mode = normalize_classification_mode(classification_mode)
    expected_top_k = {
        "family_top_k": int(family_top_k),
        "species_first_pass_top_k": int(species_first_pass_top_k),
        "species_rerank_top_k": int(species_rerank_top_k),
    }
    taxonomy_table_version: str | None = None
    prompt_variant_version: str | None = None
    if expected_mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION:
        if taxonomy_store is None:
            raise ValueError("taxonomy_store is required for hierarchical_butterfly_classification")
        taxonomy_table_version = _taxonomy_table_version(taxonomy_store)
        prompt_variant_version = _taxonomy_prompt_variant_version(taxonomy_store)

    for item in work_items:
        work_key = str(item.get("work_key") or "<unknown>")
        payload = item.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(f"work item {work_key} has invalid payload")

        payload_mode = normalize_classification_mode(payload.get("classification_mode"))
        if payload_mode != expected_mode:
            raise ValueError(
                f"work item {work_key} classification_mode {payload_mode!r} "
                f"does not match batch classification_mode {expected_mode!r}"
            )

        top_k_settings = payload.get("top_k_settings")
        if not isinstance(top_k_settings, dict):
            raise ValueError(f"work item {work_key} is missing top_k_settings")
        for key, expected in expected_top_k.items():
            actual = top_k_settings.get(key)
            try:
                actual_int = int(actual)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"work item {work_key} has invalid top_k_settings.{key}") from exc
            if actual_int != expected:
                raise ValueError(
                    f"work item {work_key} top_k_settings.{key}={actual_int} "
                    f"does not match batch {key}={expected}"
                )

        if expected_mode != HIERARCHICAL_BUTTERFLY_CLASSIFICATION:
            continue
        if str(payload.get("taxonomy_table_version") or "") != taxonomy_table_version:
            raise ValueError(
                f"work item {work_key} taxonomy_table_version "
                f"{str(payload.get('taxonomy_table_version') or '')!r} does not match taxonomy store "
                f"{taxonomy_table_version!r}"
            )
        if str(payload.get("taxonomy_prompt_variant_version") or "") != prompt_variant_version:
            raise ValueError(
                f"work item {work_key} taxonomy_prompt_variant_version "
                f"{str(payload.get('taxonomy_prompt_variant_version') or '')!r} does not match taxonomy store "
                f"{prompt_variant_version!r}"
            )


def _taxonomy_table_version(taxonomy_store: ButterflyTaxonomyStore) -> str:
    manifest = taxonomy_store.manifest or {}
    return str(
        manifest.get("classification_table_version")
        or _first_value(taxonomy_store.classification_taxa, "classification_table_version")
        or CLASSIFICATION_TABLE_VERSION
    )


def _taxonomy_prompt_variant_version(taxonomy_store: ButterflyTaxonomyStore) -> str:
    manifest = taxonomy_store.manifest or {}
    return str(
        manifest.get("prompt_variant_version")
        or _first_value(taxonomy_store.family_labels, "prompt_variant_version")
        or PROMPT_VARIANT_VERSION
    )


def _first_value(frame: pl.DataFrame, column: str) -> object:
    if column not in frame.columns or frame.is_empty():
        return None
    values = frame.select(column).drop_nulls().get_column(column)
    return values.item(0) if values.len() else None


def _score_hierarchical_cloud_batch(
    *,
    items: list[dict[str, Any]],
    scorer: ObjectBioClipScorer,
    taxonomy_store: ButterflyTaxonomyStore,
    family_top_k: int,
    species_first_pass_top_k: int,
    species_rerank_top_k: int,
    taxonomy_text_embedding_cache: pl.DataFrame | None = None,
) -> list[dict[str, Any]]:
    if isinstance(taxonomy_store, FiveRankTaxonomyStore):
        results = classify_five_rank_crops_batch(
            items=items,
            scorer=scorer,
            taxonomy_store=taxonomy_store,
            beam_widths={"FAMILY": family_top_k},
            species_first_pass_top_k=species_first_pass_top_k,
            species_rerank_top_k=species_rerank_top_k,
            taxonomy_text_embedding_cache=taxonomy_text_embedding_cache,
        )
        return [
            five_rank_result_to_object_score_row(
                item=item,
                result=result,
                scorer=scorer,
                family_top_k=family_top_k,
                species_first_pass_top_k=species_first_pass_top_k,
                species_rerank_top_k=species_rerank_top_k,
            )
            for item, result in zip(items, results, strict=True)
        ]
    results = classify_butterfly_crops_hierarchical_batch(
        items=items,
        scorer=scorer,
        taxonomy_store=taxonomy_store,
        family_top_k=family_top_k,
        species_first_pass_top_k=species_first_pass_top_k,
        species_rerank_top_k=species_rerank_top_k,
        taxonomy_text_embedding_cache=taxonomy_text_embedding_cache,
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
    gate_decision: ScoreInputDecision,
) -> bool:
    if not gate_decision.should_score:
        return False
    return bool(
        str(detection.get("source") or "")
        and str(detection.get("flickr_photo_id") or "")
        and str(detection.get("detection_id") or "")
    )


def _active_bioclip_gate_policy(
    *,
    detection_policy: DetectionPolicy | None,
    bioclip_gate_policy: BioClipGatePolicy | None,
) -> BioClipGatePolicy:
    if bioclip_gate_policy is not None:
        return bioclip_gate_policy
    active_detection_policy = detection_policy or DetectionPolicy()
    return BioClipGatePolicy.legacy_butterfly_like_only(
        eligible_detector_labels=tuple(active_detection_policy.bioclip_eligible_labels)
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
