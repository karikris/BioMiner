from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import polars as pl

from biominer.bioclip.candidate_sets import CandidateSet
from biominer.bioclip.object_runner import AblationMode, ObjectBioClipScorer, screen_object_detections
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
) -> AblationRunReport:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    frames: list[pl.DataFrame] = []
    for mode in modes:
        result = screen_object_detections(
            canonical_records=canonical_records,
            detections=detections,
            species_context=species_context,
            candidate_set=candidate_set,
            scorer=scorer,
            output_path=base / f"object_bioclip_scores_{mode}.parquet",
            ablation_mode=mode,
        )
        frames.append(result.frame)
    combined = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    report = build_ablation_report(combined)
    (base / "ablation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return AblationRunReport(output_dir=base, modes=modes, report=report)


def build_ablation_report(frame: pl.DataFrame) -> dict[str, Any]:
    if frame.is_empty():
        return {
            "records_seen": 0,
            "detections_seen": 0,
            "crops_scored": 0,
            "no_detection_records": 0,
            "mean_target_rank": None,
            "median_target_rank": None,
            "gold_count": 0,
            "silver_count": 0,
            "bronze_count": 0,
            "bin_count": 0,
            "in_review_count": 0,
            "whole_image_vs_crop_disagreements": 0,
            "crop_vs_segmentation_disagreements": 0,
        }
    ranks = [int(value) for value in frame.get_column("target_species_rank").drop_nulls().to_list()] if "target_species_rank" in frame.columns else []
    counts = _bucket_counts(frame)
    return {
        "ablation_mode": sorted(frame.get_column("ablation_mode").unique().to_list()) if "ablation_mode" in frame.columns else [],
        "records_seen": frame.select(["source", "flickr_photo_id"]).unique().height,
        "detections_seen": frame.select(["source", "flickr_photo_id", "detection_id"]).unique().height,
        "crops_scored": frame.height,
        "no_detection_records": 0,
        "mean_target_rank": sum(ranks) / len(ranks) if ranks else None,
        "median_target_rank": _median(ranks),
        "gold_count": counts.get("gold", 0),
        "silver_count": counts.get("silver", 0),
        "bronze_count": counts.get("bronze", 0),
        "bin_count": counts.get("bin", 0),
        "in_review_count": counts.get("in_review", 0),
        "whole_image_vs_crop_disagreements": _disagreements(frame, "whole_image", "detector_crop"),
        "crop_vs_segmentation_disagreements": _disagreements(frame, "detector_crop", "detector_crop_segmentation"),
    }


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
    left_frame = frame.filter(pl.col("ablation_mode") == left).select(["source", "flickr_photo_id", "detection_id", "occurrence_bin"])
    right_frame = frame.filter(pl.col("ablation_mode") == right).select(["source", "flickr_photo_id", "detection_id", "occurrence_bin"])
    if left_frame.is_empty() or right_frame.is_empty():
        return 0
    joined = left_frame.join(right_frame, on=["source", "flickr_photo_id", "detection_id"], suffix="_right")
    return joined.filter(pl.col("occurrence_bin") != pl.col("occurrence_bin_right")).height
