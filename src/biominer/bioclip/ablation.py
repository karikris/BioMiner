from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import polars as pl

from biominer.bioclip.candidate_sets import CandidateSet
from biominer.bioclip.classification_modes import (
    DEFAULT_CLASSIFICATION_MODE,
    DEFAULT_FAMILY_TOP_K,
    DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    DEFAULT_SPECIES_RERANK_TOP_K,
    ClassificationMode,
)
from biominer.bioclip.object_runner import PRIMARY_VISUAL_CLASSIFIER, AblationMode, ObjectBioClipScorer, screen_object_detections
from biominer.bioclip.taxonomy_store import ButterflyTaxonomyStore
from biominer.species.context import SpeciesContext


@dataclass(frozen=True)
class AblationRunReport:
    output_dir: Path
    modes: tuple[AblationMode, ...]
    report: dict[str, Any]


def run_object_ablations(
    *,
    canonical_records: pl.DataFrame,
    detections: pl.DataFrame,
    species_context: SpeciesContext,
    candidate_set: CandidateSet,
    scorer: ObjectBioClipScorer,
    output_dir: str | Path,
    modes: tuple[AblationMode, ...],
    geo_prior_table: pl.DataFrame | None = None,
    parquet_batch_rows: int = 10000,
    bioclip_batch_size: int = 24,
    classification_mode: ClassificationMode = DEFAULT_CLASSIFICATION_MODE,
    family_top_k: int = DEFAULT_FAMILY_TOP_K,
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K,
    taxonomy_store: ButterflyTaxonomyStore | None = None,
) -> AblationRunReport:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    frames: list[pl.DataFrame] = []
    score_batches_by_mode: dict[str, int] = {}
    visual_mode_status_by_mode: dict[str, str | None] = {}
    segmentation_status_by_mode: dict[str, str | None] = {}
    segmentation_unavailable_count_by_mode: dict[str, int] = {}
    segmentation_unavailable_reason_by_mode: dict[str, str | None] = {}
    for mode in modes:
        result = screen_object_detections(
            canonical_records=canonical_records,
            detections=detections,
            species_context=species_context,
            candidate_set=candidate_set,
            scorer=scorer,
            output_path=base / f"object_bioclip_scores_{mode}.parquet",
            ablation_mode=mode,
            geo_prior_table=geo_prior_table,
            parquet_batch_rows=parquet_batch_rows,
            bioclip_batch_size=bioclip_batch_size,
            classification_mode=classification_mode,
            family_top_k=family_top_k,
            species_first_pass_top_k=species_first_pass_top_k,
            species_rerank_top_k=species_rerank_top_k,
            taxonomy_store=taxonomy_store,
        )
        frames.append(result.frame)
        score_batches_by_mode[mode] = result.score_batches_written
        visual_mode_status_by_mode[mode] = result.visual_mode_status
        segmentation_status_by_mode[mode] = result.segmentation_status
        segmentation_unavailable_count_by_mode[mode] = result.segmentation_unavailable_count
        segmentation_unavailable_reason_by_mode[mode] = result.segmentation_unavailable_reason
    combined = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    report = build_ablation_report(combined, canonical_records=canonical_records, detections=detections)
    report["primary_visual_classifier"] = PRIMARY_VISUAL_CLASSIFIER
    report["ablation_mode"] = list(modes)
    report["visual_modes_requested"] = list(modes)
    report["visual_modes_scored"] = sorted(combined.get_column("ablation_mode").unique().to_list()) if "ablation_mode" in combined.columns else []
    report["visual_mode_status_by_mode"] = visual_mode_status_by_mode
    report["score_batches_written_by_mode"] = score_batches_by_mode
    report["score_batches_written"] = sum(score_batches_by_mode.values())
    report["segmentation_status_by_mode"] = segmentation_status_by_mode
    report["segmentation_unavailable_count_by_mode"] = segmentation_unavailable_count_by_mode
    report["segmentation_unavailable_reason_by_mode"] = segmentation_unavailable_reason_by_mode
    (base / "ablation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return AblationRunReport(output_dir=base, modes=modes, report=report)


def build_ablation_report(
    frame: pl.DataFrame,
    *,
    canonical_records: pl.DataFrame | None = None,
    detections: pl.DataFrame | None = None,
) -> dict[str, Any]:
    records_seen = _records_seen(frame, canonical_records)
    detections_seen = _detections_seen(frame, detections)
    no_detection_records = _no_detection_records(detections)
    if frame.is_empty():
        return {
            "ablation_mode": [],
            "records_seen": records_seen,
            "detections_seen": detections_seen,
            "crops_scored": 0,
            "no_detection_records": no_detection_records,
            "mean_target_rank": None,
            "median_target_rank": None,
            "gold_count": 0,
            "silver_count": 0,
            "bronze_count": 0,
            "bin_count": 0,
            "in_review_count": 0,
            "whole_image_vs_crop": 0,
            "crop_vs_segmentation": 0,
            "whole_image_vs_crop_disagreements": 0,
            "crop_vs_segmentation_disagreements": 0,
        }
    ranks = [int(value) for value in frame.get_column("target_species_rank").drop_nulls().to_list()] if "target_species_rank" in frame.columns else []
    counts = _bucket_counts(frame)
    whole_image_vs_crop = _disagreements(frame, "whole_image", "detector_crop")
    crop_vs_segmentation = _disagreements(frame, "detector_crop", "detector_crop_segmentation")
    return {
        "ablation_mode": sorted(frame.get_column("ablation_mode").unique().to_list()) if "ablation_mode" in frame.columns else [],
        "records_seen": records_seen,
        "detections_seen": detections_seen,
        "crops_scored": frame.height,
        "no_detection_records": no_detection_records,
        "mean_target_rank": sum(ranks) / len(ranks) if ranks else None,
        "median_target_rank": _median(ranks),
        "gold_count": counts.get("gold", 0),
        "silver_count": counts.get("silver", 0),
        "bronze_count": counts.get("bronze", 0),
        "bin_count": counts.get("bin", 0),
        "in_review_count": counts.get("in_review", 0),
        "whole_image_vs_crop": whole_image_vs_crop,
        "crop_vs_segmentation": crop_vs_segmentation,
        "whole_image_vs_crop_disagreements": whole_image_vs_crop,
        "crop_vs_segmentation_disagreements": crop_vs_segmentation,
    }


def _records_seen(frame: pl.DataFrame, canonical_records: pl.DataFrame | None) -> int:
    if canonical_records is not None and not canonical_records.is_empty():
        return canonical_records.select(["source", "flickr_photo_id"]).unique().height
    if frame.is_empty() or not {"source", "flickr_photo_id"}.issubset(frame.columns):
        return 0
    return frame.select(["source", "flickr_photo_id"]).unique().height


def _detections_seen(frame: pl.DataFrame, detections: pl.DataFrame | None) -> int:
    columns = ["source", "flickr_photo_id", "detection_id"]
    if detections is not None and not detections.is_empty():
        return detections.select(columns).unique().height
    if frame.is_empty() or not set(columns).issubset(frame.columns):
        return 0
    return frame.select(columns).unique().height


def _no_detection_records(detections: pl.DataFrame | None) -> int:
    if detections is None or detections.is_empty() or "detection_status" not in detections.columns:
        return 0
    return detections.filter(pl.col("detection_status") == "no_detection").select(["source", "flickr_photo_id"]).unique().height


def _bucket_counts(frame: pl.DataFrame) -> dict[str, int]:
    if "occurrence_bin" not in frame.columns:
        return {}
    return {str(row["occurrence_bin"]): int(row["len"]) for row in frame.group_by("occurrence_bin").len().to_dicts()}


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def _disagreements(frame: pl.DataFrame, left: str, right: str) -> int:
    if "ablation_mode" not in frame.columns:
        return 0
    compare_columns = [
        column
        for column in ("occurrence_bin", "species_top1_scientific_name", "species_top1_accepted_taxon_key", "target_species_rank")
        if column in frame.columns
    ]
    if not compare_columns:
        return 0
    select_columns = ["source", "flickr_photo_id", "detection_id", *compare_columns]
    left_frame = frame.filter(pl.col("ablation_mode") == left).select(select_columns)
    right_frame = frame.filter(pl.col("ablation_mode") == right).select(select_columns)
    if left_frame.is_empty() or right_frame.is_empty():
        return 0
    joined = left_frame.join(right_frame, on=["source", "flickr_photo_id", "detection_id"], suffix="_right")
    disagreement = None
    for column in compare_columns:
        current = pl.col(column).fill_null("") != pl.col(f"{column}_right").fill_null("")
        disagreement = current if disagreement is None else disagreement | current
    if disagreement is None:
        return 0
    return joined.filter(disagreement).height
