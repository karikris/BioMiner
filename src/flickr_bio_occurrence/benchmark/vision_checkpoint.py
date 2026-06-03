from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import polars as pl

VisionClassifier = Callable[[dict[str, object]], dict[str, object]]


@dataclass(frozen=True)
class VisionCheckpointResult:
    frame: pl.DataFrame
    paths: list[Path]
    completed: int
    newly_completed: int
    skipped_existing: int
    gpu_used: bool = False
    gpu_name: str = ""


def build_checkpointed_vision_predictions(
    bronze: pl.DataFrame,
    vision_classifier: VisionClassifier | None,
    checkpoint_dir: str | Path,
) -> VisionCheckpointResult:
    output_dir = Path(checkpoint_dir)
    if vision_classifier is None or not bronze.height:
        return VisionCheckpointResult(frame=pl.DataFrame(), paths=[], completed=0, newly_completed=0, skipped_existing=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    skipped = 0
    rows_to_process = [row for row in bronze.to_dicts() if row.get("image_url")]
    gpu_used = False
    gpu_name = ""
    try:
        for row in rows_to_process:
            photo_id = str(row["flickr_photo_id"])
            path = vision_prediction_checkpoint_path(output_dir, photo_id)
            if path.exists():
                skipped += 1
                continue
            prediction = vision_classifier(row)
            pl.DataFrame([prediction]).write_parquet(path)
            completed += 1
    finally:
        scorer = getattr(vision_classifier, "scorer", None)
        device = getattr(scorer, "device", "")
        gpu_used = device == "cuda"
        gpu_name = str(getattr(scorer, "gpu_name", "") or "")
        close = getattr(vision_classifier, "close", None)
        if callable(close):
            close()

    paths = sorted(vision_prediction_checkpoint_path(output_dir, str(row["flickr_photo_id"])) for row in rows_to_process)
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return VisionCheckpointResult(
            frame=pl.DataFrame(),
            paths=[],
            completed=0,
            newly_completed=completed,
            skipped_existing=skipped,
            gpu_used=gpu_used,
            gpu_name=gpu_name,
        )
    return VisionCheckpointResult(
        frame=pl.read_parquet(existing_paths),
        paths=existing_paths,
        completed=len(existing_paths),
        newly_completed=completed,
        skipped_existing=skipped,
        gpu_used=gpu_used,
        gpu_name=gpu_name,
    )


def vision_prediction_checkpoint_path(output_dir: Path, flickr_photo_id: str) -> Path:
    safe_id = quote(flickr_photo_id, safe="")
    return output_dir / f"flickr_photo_id={safe_id}.parquet"
