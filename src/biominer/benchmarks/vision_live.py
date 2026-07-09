from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import polars as pl

from biominer.bioclip.bioclip import PersistentBioClipScorer
from biominer.bioclip.candidate_sets import build_candidate_set
from biominer.bioclip.classification_modes import (
    DEFAULT_FAMILY_TOP_K,
    DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    DEFAULT_SPECIES_RERANK_TOP_K,
    HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
)
from biominer.bioclip.model_registry import BioClipRuntime
from biominer.bioclip.object_runner import EphemeralCropBioClipScorer, screen_object_detections
from biominer.bioclip.taxonomy_store import ButterflyTaxonomyStore
from biominer.detection.image_io import load_decoded_image_from_record
from biominer.detection.pipeline import run_detection_pipeline
from biominer.detection.policy import DetectionPolicy, DetectionRunPolicy
from biominer.detection.segmentation import make_segmenter
from biominer.detection.yoloe26_detector import YoloE26SidecarObjectDetector
from biominer.evidence.join import write_object_evidence_outputs
from biominer.species.context import SpeciesContext
from biominer.storage.parquet import write_parquet


LIVE_M5PRO_BENCHMARK_KIND = "vision_live_m5pro"


@dataclass(frozen=True)
class LiveM5ProBenchmarkRequest:
    input_path: Path
    taxonomy_candidate_table: Path
    vision_runtime_python: Path
    bioclip_runtime_python: Path
    hf_cache_dir: Path
    checkpoint: str
    yolo_sidecar_transport: str
    device: str
    limit: int
    output_dir: Path
    cache_root: Path
    crop_temp_dir: Path
    imgsz: int
    conf: float
    iou: float
    max_det: int
    yolo_batch: int
    bioclip_batch: int
    crop_padding_ratio: float
    crop_target_px: int
    parquet_batch_rows: int
    prompt_classes: tuple[str, ...]
    family_top_k: int = DEFAULT_FAMILY_TOP_K
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K


@dataclass(frozen=True)
class LiveM5ProBenchmarkResult:
    metrics: dict[str, Any]
    output_dir: Path
    metrics_path: Path
    summary_path: Path


def validate_live_m5pro_benchmark_request(request: LiveM5ProBenchmarkRequest) -> dict[str, Any] | None:
    missing = []
    for field, path in (
        ("input_path", request.input_path),
        ("vision_runtime_python", request.vision_runtime_python),
        ("bioclip_runtime_python", request.bioclip_runtime_python),
    ):
        if not path.exists():
            missing.append({"field": field, "path": str(path)})
    if missing:
        return {
            "error": "missing_required_path",
            "benchmark_kind": LIVE_M5PRO_BENCHMARK_KIND,
            "missing": missing,
        }
    taxonomy_error = validate_live_taxonomy_store(request.taxonomy_candidate_table)
    if taxonomy_error is not None:
        return taxonomy_error
    if request.limit <= 0:
        return {"error": "invalid_limit", "message": "--limit must be positive"}
    if request.yolo_batch <= 0 or request.bioclip_batch <= 0:
        return {"error": "invalid_batch_size", "message": "--yolo-batch and --bioclip-batch must be positive"}
    if request.yolo_sidecar_transport not in {"json_b64", "image_path"}:
        return {"error": "invalid_yolo_sidecar_transport", "message": "--yolo-sidecar-transport must be json_b64 or image_path"}
    return None


def validate_live_taxonomy_store(path: Path) -> dict[str, Any] | None:
    try:
        ButterflyTaxonomyStore.read(path)
    except FileNotFoundError as exc:
        return {
            "error": "missing_taxonomy_candidate_table",
            "benchmark_kind": LIVE_M5PRO_BENCHMARK_KIND,
            "message": str(exc),
            "taxonomy_candidate_table": str(path),
        }
    except ValueError as exc:
        return {
            "error": "invalid_taxonomy_candidate_table",
            "benchmark_kind": LIVE_M5PRO_BENCHMARK_KIND,
            "message": str(exc),
            "taxonomy_candidate_table": str(path),
        }
    return None


def run_live_m5pro_benchmark(
    *,
    request: LiveM5ProBenchmarkRequest,
    bioclip_runtime: BioClipRuntime,
) -> LiveM5ProBenchmarkResult:
    validation = validate_live_m5pro_benchmark_request(request)
    if validation is not None:
        raise ValueError(json.dumps(validation, sort_keys=True))
    request.output_dir.mkdir(parents=True, exist_ok=True)
    stage_seconds: dict[str, float] = {}
    total_start = perf_counter()

    stage_start = perf_counter()
    records = pl.read_parquet(request.input_path).head(request.limit)
    if records.is_empty():
        raise ValueError("input benchmark records are empty after applying --limit")
    canonical_records_path = write_parquet(records, request.output_dir / "canonical_source_records.parquet")
    taxonomy_store = ButterflyTaxonomyStore.read(request.taxonomy_candidate_table)
    species_context = species_context_from_taxonomy_store(taxonomy_store)
    candidate_set = build_candidate_set(species_context, records=records.to_dicts(), allow_single_target_fixture=True)
    stage_seconds["load_inputs"] = _elapsed(stage_start)

    detections_path = request.output_dir / "object_detections.parquet"
    scores_path = request.output_dir / "object_bioclip_scores.parquet"
    joined_path = request.output_dir / "object_evidence_joined.parquet"
    photo_summary_path = request.output_dir / "photo_evidence_summary.parquet"
    metrics_path = request.output_dir / "benchmark_metrics.json"
    summary_path = request.output_dir / "benchmark_summary.md"

    stage_start = perf_counter()
    detector = YoloE26SidecarObjectDetector(
        runtime_python=str(request.vision_runtime_python),
        checkpoint=request.checkpoint,
        transport=request.yolo_sidecar_transport,
        device=request.device,
        imgsz=request.imgsz,
        conf=request.conf,
        iou=request.iou,
        max_det=request.max_det,
        prompt_classes=request.prompt_classes or None,
    )
    try:
        detection_result = run_detection_pipeline(
            records=records.to_dicts(),
            detector=detector,
            output_path=detections_path,
            image_loader=lambda record: load_decoded_image_from_record(record, cache_root=request.cache_root),
            detection_policy=DetectionPolicy(
                backend="yoloe26",
                box_score_threshold=request.conf,
                nms_iou_threshold=request.iou,
                max_boxes_per_image=request.max_det,
                crop_padding_ratio=request.crop_padding_ratio,
                crop_target_px=request.crop_target_px,
            ),
            run_policy=DetectionRunPolicy(
                detector_batch_size=request.yolo_batch,
                crop_batch_size=request.bioclip_batch,
                parquet_batch_rows=request.parquet_batch_rows,
            ),
        )
    finally:
        detector.close()
    stage_seconds["detect_objects"] = _elapsed(stage_start)

    stage_start = perf_counter()
    scorer = PersistentBioClipScorer(
        runtime=bioclip_runtime,
        hf_cache_dir=request.hf_cache_dir,
        device=request.device,
    )
    try:
        object_scorer = EphemeralCropBioClipScorer(
            scorer=scorer,
            image_loader=lambda item: load_decoded_image_from_record(item, cache_root=request.cache_root),
            temp_dir=request.crop_temp_dir,
            crop_padding_ratio=request.crop_padding_ratio,
            crop_target_px=request.crop_target_px,
            model_id=bioclip_runtime.model.model_id,
            model_version=bioclip_runtime.model.model_id,
            model_checkpoint=bioclip_runtime.model.checkpoint,
            retain_debug_crops=False,
            segmenter=make_segmenter("none"),
        )
        score_result = screen_object_detections(
            canonical_records=records,
            detections=detection_result.frame,
            species_context=species_context,
            candidate_set=candidate_set,
            scorer=object_scorer,
            output_path=scores_path,
            classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
            family_top_k=request.family_top_k,
            species_first_pass_top_k=request.species_first_pass_top_k,
            species_rerank_top_k=request.species_rerank_top_k,
            taxonomy_store=taxonomy_store,
            parquet_batch_rows=request.parquet_batch_rows,
            bioclip_batch_size=request.bioclip_batch,
        )
        actual_device = scorer.device
        actual_gpu_name = scorer.gpu_name
    finally:
        scorer.close()
    stage_seconds["score_crops"] = _elapsed(stage_start)

    stage_start = perf_counter()
    evidence_outputs = write_object_evidence_outputs(
        canonical_source_records=records,
        object_detections=detection_result.frame,
        object_scores=score_result.frame,
        joined_output_path=joined_path,
        photo_summary_output_path=photo_summary_path,
        species_context=species_context,
    )
    joined = pl.read_parquet(evidence_outputs.object_evidence_joined)
    photo_summary = pl.read_parquet(evidence_outputs.photo_evidence_summary)
    stage_seconds["join_evidence"] = _elapsed(stage_start)

    metrics = _live_metrics(
        request=request,
        records=records,
        detection_result=detection_result,
        score_result=score_result,
        joined=joined,
        photo_summary=photo_summary,
        bioclip_runtime=bioclip_runtime,
        actual_device=actual_device,
        actual_gpu_name=actual_gpu_name,
        stage_seconds=stage_seconds,
        total_start=total_start,
        outputs={
            "canonical_source_records": canonical_records_path,
            "object_detections": detection_result.output_path,
            "object_bioclip_scores": score_result.output_path,
            "object_evidence_joined": evidence_outputs.object_evidence_joined,
            "photo_evidence_summary": evidence_outputs.photo_evidence_summary,
            "benchmark_metrics": metrics_path,
            "benchmark_summary": summary_path,
        },
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    summary_path.write_text(_live_summary_markdown(metrics), encoding="utf-8")
    return LiveM5ProBenchmarkResult(metrics=metrics, output_dir=request.output_dir, metrics_path=metrics_path, summary_path=summary_path)


def species_context_from_taxonomy_store(taxonomy_store: ButterflyTaxonomyStore) -> SpeciesContext:
    taxa = taxonomy_store.classification_taxa.filter(pl.col("classification_enabled")).sort(
        ["family", "genus", "scientific_name", "accepted_taxon_key"]
    )
    if taxa.is_empty():
        raise ValueError("taxonomy candidate table has no enabled species")
    row = taxa.to_dicts()[0]
    return SpeciesContext(
        scientific_name=str(row.get("scientific_name") or ""),
        accepted_taxon_key=str(row.get("accepted_taxon_key") or ""),
        canonical_name=str(row.get("canonical_name") or row.get("scientific_name") or ""),
        family=str(row.get("family") or ""),
        genus=str(row.get("genus") or ""),
        family_key=str(row.get("family_key") or ""),
        genus_key=str(row.get("genus_key") or ""),
        species_key=str(row.get("species_key") or row.get("accepted_taxon_key") or ""),
        registry_version=str(row.get("registry_version") or ""),
    )


def _live_metrics(
    *,
    request: LiveM5ProBenchmarkRequest,
    records: pl.DataFrame,
    detection_result: Any,
    score_result: Any,
    joined: pl.DataFrame,
    photo_summary: pl.DataFrame,
    bioclip_runtime: BioClipRuntime,
    actual_device: str | None,
    actual_gpu_name: str | None,
    stage_seconds: Mapping[str, float],
    total_start: float,
    outputs: Mapping[str, Path | None],
) -> dict[str, Any]:
    detections = detection_result.frame
    butterfly_like = detections.filter(
        (pl.col("detection_status") == "detected") & (pl.col("detector_label") == "butterfly_like")
    ).height if not detections.is_empty() and {"detection_status", "detector_label"}.issubset(set(detections.columns)) else 0
    return {
        "benchmark_kind": LIVE_M5PRO_BENCHMARK_KIND,
        "status": "ok",
        "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        "input": str(request.input_path),
        "taxonomy_candidate_table": str(request.taxonomy_candidate_table),
        "records_requested_limit": request.limit,
        "records_loaded": records.height,
        "model_metadata": {
            "yolo_checkpoint": request.checkpoint,
            "yolo_sidecar_transport": request.yolo_sidecar_transport,
            "bioclip_model_id": bioclip_runtime.model.model_id,
            "bioclip_model_name": bioclip_runtime.model.model_name,
            "bioclip_checkpoint": bioclip_runtime.model.checkpoint,
        },
        "device": {
            "requested": request.device,
            "actual": actual_device,
            "gpu_name": actual_gpu_name,
            "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        },
        "batch_settings": {
            "yolo_batch": request.yolo_batch,
            "bioclip_batch": request.bioclip_batch,
            "imgsz": request.imgsz,
            "crop_target_px": request.crop_target_px,
            "crop_padding_ratio": request.crop_padding_ratio,
            "parquet_batch_rows": request.parquet_batch_rows,
        },
        "rows": {
            "detections": detections.height,
            "butterfly_like_detections": butterfly_like,
            "scores": score_result.frame.height,
            "object_evidence": joined.height,
            "photo_summary": photo_summary.height,
        },
        "elapsed_seconds_by_stage": {key: round(value, 6) for key, value in stage_seconds.items()},
        "elapsed_seconds": _elapsed(total_start),
        "outputs": {key: str(value) if value is not None else None for key, value in outputs.items()},
        "created_at": datetime.now(UTC).isoformat(),
    }


def _live_summary_markdown(metrics: Mapping[str, Any]) -> str:
    rows = dict(metrics.get("rows") or {})
    stages = dict(metrics.get("elapsed_seconds_by_stage") or {})
    device = dict(metrics.get("device") or {})
    lines = [
        "# Live M5 Pro Vision Benchmark",
        "",
        f"- status: `{metrics.get('status')}`",
        f"- records_loaded: {metrics.get('records_loaded')}",
        f"- requested_device: `{device.get('requested')}`",
        f"- actual_device: `{device.get('actual')}`",
        f"- mps_fallback: `{device.get('pytorch_enable_mps_fallback')}`",
        f"- elapsed_seconds: {metrics.get('elapsed_seconds')}",
        "",
        "## Rows",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(rows.items()))
    lines.extend(["", "## Stage Seconds", ""])
    lines.extend(f"- {key}: {value}" for key, value in sorted(stages.items()))
    return "\n".join(lines) + "\n"


def _elapsed(start: float) -> float:
    return round(perf_counter() - start, 6)
