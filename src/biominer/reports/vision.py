from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import polars as pl

from biominer.detection.policy import DetectionPolicy, detection_is_bioclip_eligible


VISION_STAGE_METRICS_FILE = "vision_stage_metrics.json"
VISION_STAGE_SUMMARY_FILE = "vision_stage_summary.md"


def build_vision_stage_metrics(
    *,
    detections: pl.DataFrame | None = None,
    scores: pl.DataFrame | None = None,
    joined: pl.DataFrame | None = None,
    photo_summary: pl.DataFrame | None = None,
    stage_metrics: dict[str, Any] | None = None,
    detection_policy: DetectionPolicy | None = None,
) -> dict[str, Any]:
    """Build deterministic observability metrics for the detector-first vision stages."""

    runtime = dict(stage_metrics or {})
    detection_metrics = _detection_metrics(detections, runtime=runtime, detection_policy=detection_policy)
    bioclip_metrics = _bioclip_metrics(scores, detections=detections, runtime=runtime)
    evidence_metrics = {
        "object_evidence_rows": _height(joined),
        "photo_summary_rows": _height(photo_summary),
        "object_occurrence_bin_counts": _value_counts(joined, "occurrence_bin"),
        "photo_occurrence_bin_counts": _value_counts(photo_summary, "photo_occurrence_bin"),
    }
    warnings = _warning_flags(detection_metrics=detection_metrics, bioclip_metrics=bioclip_metrics)
    return {
        "schema_version": "vision_stage_metrics_v1",
        "detection": detection_metrics,
        "bioclip": bioclip_metrics,
        "evidence": evidence_metrics,
        "throughput": _throughput_metrics(runtime),
        "warnings": warnings,
    }


def write_vision_stage_reports(metrics: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / VISION_STAGE_METRICS_FILE
    summary_path = output / VISION_STAGE_SUMMARY_FILE
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    summary_path.write_text(vision_stage_summary_markdown(metrics), encoding="utf-8")
    return {"metrics": metrics_path, "summary": summary_path}


def vision_stage_summary_markdown(metrics: dict[str, Any]) -> str:
    detection = _dict(metrics.get("detection"))
    bioclip = _dict(metrics.get("bioclip"))
    evidence = _dict(metrics.get("evidence"))
    throughput = _dict(metrics.get("throughput"))
    warnings = [str(item) for item in metrics.get("warnings") or []]
    warning_lines = [f"- {warning}" for warning in warnings] if warnings else ["- none"]
    lines = [
        "# Vision Stage Metrics",
        "",
        "## Detection",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Images seen | {_display(detection.get('images_seen'))} |",
        f"| Images loaded | {_display(detection.get('images_loaded'))} |",
        f"| Image failures | {_display(detection.get('image_failures'))} |",
        f"| Butterfly-like detections | {_display(detection.get('butterfly_like_detections'))} |",
        f"| Eligible BioCLIP detections | {_display(detection.get('eligible_bioclip_detections'))} |",
        f"| Hard-negative detections | {_display(detection.get('hard_negative_detections'))} |",
        f"| No-detection records | {_display(detection.get('no_detection_count'))} |",
        "",
        "## BioCLIP",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Crops seen | {_display(bioclip.get('crops_seen'))} |",
        f"| Crops scored | {_display(bioclip.get('crops_scored'))} |",
        f"| Family score entries | {_display(bioclip.get('family_scores_computed'))} |",
        f"| Species first-pass candidates | {_display(bioclip.get('species_first_pass_candidates_seen'))} |",
        f"| Species rerank candidates | {_display(bioclip.get('species_rerank_candidates_seen'))} |",
        f"| Adaptive retries | {_display(bioclip.get('bioclip_batch_retries'))} |",
        "",
        "## Evidence",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Object evidence rows | {_display(evidence.get('object_evidence_rows'))} |",
        f"| Photo summary rows | {_display(evidence.get('photo_summary_rows'))} |",
        "",
        "## Top Families",
        "",
        *_top_count_lines(_dict(bioclip.get("selected_family_counts"))),
        "",
        "## Top Species",
        "",
        *_top_count_lines(_dict(bioclip.get("species_top1_counts"))),
        "",
        "## Throughput",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total seconds | {_display(throughput.get('total_seconds'))} |",
        f"| Images/sec | {_display(throughput.get('images_per_second'))} |",
        f"| Crops/sec | {_display(throughput.get('crops_per_second'))} |",
        "",
        "## Warnings",
        "",
        *warning_lines,
        "",
    ]
    return "\n".join(lines)


def _detection_metrics(
    frame: pl.DataFrame | None,
    *,
    runtime: dict[str, Any],
    detection_policy: DetectionPolicy | None,
) -> dict[str, Any]:
    rows = frame.to_dicts() if frame is not None and not frame.is_empty() else []
    eligible = sum(1 for row in rows if detection_is_bioclip_eligible(row, detection_policy))
    unique_images = _unique_record_count(frame)
    detections_by_label = _value_counts(frame, "detector_label")
    status_counts = _value_counts(frame, "detection_status")
    return {
        "images_seen": _runtime_int(runtime, "records_seen", "images_seen", default=unique_images),
        "images_loaded": _runtime_int(runtime, "images_loaded"),
        "image_failures": _runtime_int(runtime, "image_failures", default=status_counts.get("failed_image_load")),
        "detections_by_label": detections_by_label,
        "detection_status_counts": status_counts,
        "butterfly_like_detections": _detected_label_count(rows, "butterfly_like"),
        "eligible_bioclip_detections": eligible,
        "hard_negative_detections": _detected_label_count(rows, "hard_negative"),
        "no_detection_count": int(status_counts.get("no_detection", 0)),
        "crops_created": _runtime_int(runtime, "crops_created", default=_non_null_count(frame, "crop_hash")),
        "detector_batch_size_initial": _runtime_int(runtime, "detector_batch_size_initial", "detector_batch_size"),
        "detector_batch_size_final": _runtime_int(runtime, "detector_batch_size_final"),
        "detector_batch_size_min": _runtime_int(runtime, "detector_batch_size_min"),
        "detector_batch_retries": _runtime_int(runtime, "detector_batch_retries", default=0),
        "detector_runtime_seconds": _runtime_float(runtime, "detector_runtime_seconds"),
    }


def _bioclip_metrics(scores: pl.DataFrame | None, *, detections: pl.DataFrame | None, runtime: dict[str, Any]) -> dict[str, Any]:
    crops_scored = _height(scores)
    candidate_counts = _numeric_values(scores, "species_candidate_count")
    return {
        "crops_seen": _runtime_int(runtime, "crops_seen", default=_eligible_detection_count(detections)),
        "crops_scored": _runtime_int(runtime, "crops_scored", "objects_scored", default=crops_scored),
        "family_scores_computed": _runtime_int(runtime, "family_scores_computed", default=_list_length_sum(scores, "family_top3")),
        "species_first_pass_candidates_seen": _runtime_int(
            runtime,
            "species_first_pass_candidates_seen",
            default=_list_length_sum(scores, "species_top20"),
        ),
        "species_rerank_candidates_seen": _runtime_int(
            runtime,
            "species_rerank_candidates_seen",
            default=_list_length_sum(scores, "species_top20"),
        ),
        "species_top5_returned": _list_length_sum(scores, "species_top5"),
        "selected_family_counts": _value_counts(scores, "selected_family"),
        "species_top1_counts": _value_counts(scores, "species_top1_scientific_name"),
        "species_candidate_count_min": min(candidate_counts) if candidate_counts else None,
        "species_candidate_count_max": max(candidate_counts) if candidate_counts else None,
        "bioclip_batch_size_initial": _runtime_int(runtime, "bioclip_batch_size_initial", "crop_batch_size"),
        "bioclip_batch_size_final": _runtime_int(runtime, "bioclip_batch_size_final"),
        "bioclip_batch_size_min": _runtime_int(runtime, "bioclip_batch_size_min", "min_crop_batch_size"),
        "bioclip_batch_retries": _runtime_int(runtime, "bioclip_batch_retries", default=0),
        "text_embedding_cache_used": _runtime_bool(runtime, "text_embedding_cache_used"),
        "direct_prompt_scoring_used": _runtime_bool(runtime, "direct_prompt_scoring_used", default=_direct_prompt_scoring_used(scores)),
    }


def _throughput_metrics(runtime: dict[str, Any]) -> dict[str, Any]:
    total_seconds = _runtime_float(runtime, "total_seconds", "elapsed_seconds")
    images_seen = _runtime_int(runtime, "records_seen", "images_seen")
    crops_scored = _runtime_int(runtime, "crops_scored", "objects_scored")
    return {
        "total_seconds": total_seconds,
        "images_per_second": _rate(images_seen, total_seconds),
        "crops_per_second": _rate(crops_scored, total_seconds),
    }


def _warning_flags(*, detection_metrics: dict[str, Any], bioclip_metrics: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    eligible = detection_metrics.get("eligible_bioclip_detections")
    scored = bioclip_metrics.get("crops_scored")
    if isinstance(eligible, int) and isinstance(scored, int) and scored > eligible:
        warnings.append("crops_scored_exceeds_eligible_bioclip_detections")
    if int(detection_metrics.get("hard_negative_detections") or 0) > 0:
        warnings.append("hard_negative_detections_present")
    if int(detection_metrics.get("no_detection_count") or 0) > 0:
        warnings.append("no_detection_records_present")
    return warnings


def _value_counts(frame: pl.DataFrame | None, column: str) -> dict[str, int]:
    if frame is None or frame.is_empty() or column not in frame.columns:
        return {}
    counts = frame.group_by(column).len(name="count").sort(column).to_dicts()
    return {str(row[column] or ""): int(row["count"]) for row in counts}


def _height(frame: pl.DataFrame | None) -> int:
    return int(frame.height) if frame is not None else 0


def _unique_record_count(frame: pl.DataFrame | None) -> int | None:
    if frame is None or frame.is_empty():
        return 0 if frame is not None else None
    if {"source", "flickr_photo_id"}.issubset(frame.columns):
        return frame.select(["source", "flickr_photo_id"]).unique().height
    if "flickr_photo_id" in frame.columns:
        return frame.select(["flickr_photo_id"]).unique().height
    return frame.height


def _eligible_detection_count(frame: pl.DataFrame | None) -> int | None:
    if frame is None:
        return None
    return sum(1 for row in frame.to_dicts() if detection_is_bioclip_eligible(row))


def _detected_label_count(rows: list[dict[str, Any]], label: str) -> int:
    return sum(1 for row in rows if str(row.get("detection_status") or "") == "detected" and str(row.get("detector_label") or "") == label)


def _non_null_count(frame: pl.DataFrame | None, column: str) -> int | None:
    if frame is None:
        return None
    if frame.is_empty() or column not in frame.columns:
        return 0
    return int(frame.select(pl.col(column).is_not_null().sum()).item())


def _list_length_sum(frame: pl.DataFrame | None, column: str) -> int | None:
    if frame is None:
        return None
    if frame.is_empty() or column not in frame.columns:
        return 0
    total = 0
    for value in frame.select(column).to_series().to_list():
        if isinstance(value, list | tuple):
            total += len(value)
    return total


def _numeric_values(frame: pl.DataFrame | None, column: str) -> list[float]:
    if frame is None or frame.is_empty() or column not in frame.columns:
        return []
    return [float(value) for value in frame.select(column).to_series().to_list() if value is not None]


def _direct_prompt_scoring_used(frame: pl.DataFrame | None) -> bool | None:
    if frame is None:
        return None
    if frame.is_empty():
        return False
    return "species_rerank_strategy" in frame.columns or "target_species_score" in frame.columns


def _runtime_int(runtime: dict[str, Any], *keys: str, default: int | None = None) -> int | None:
    for key in keys:
        value = runtime.get(key)
        if value is not None:
            return int(value)
    return default


def _runtime_float(runtime: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = runtime.get(key)
        if value is not None:
            return float(value)
    return None


def _runtime_bool(runtime: dict[str, Any], key: str, *, default: bool | None = None) -> bool | None:
    value = runtime.get(key)
    return bool(value) if value is not None else default


def _rate(count: int | None, seconds: float | None) -> float | None:
    if count is None or seconds is None or seconds <= 0:
        return None
    return count / seconds


def _dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _top_count_lines(counts: dict[str, Any], *, limit: int = 10) -> list[str]:
    if not counts:
        return ["- none"]
    ordered = sorted(((str(key), int(value)) for key, value in counts.items()), key=lambda item: (-item[1], item[0]))
    return [f"- {name}: {count}" for name, count in ordered[:limit]]


def _display(value: object) -> str:
    if value is None:
        return "not_instrumented"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


__all__ = [
    "VISION_STAGE_METRICS_FILE",
    "VISION_STAGE_SUMMARY_FILE",
    "build_vision_stage_metrics",
    "vision_stage_summary_markdown",
    "write_vision_stage_reports",
]
