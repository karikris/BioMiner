from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import polars as pl

VisionClassifier = Callable[[dict[str, object]], dict[str, object]]
PREDICTION_KEY = ("flickr_photo_id", "image_hash", "model_version", "model_checkpoint")
DEFAULT_RUN_ID = "default"
DEFAULT_SHARD_SIZE = 1000


@dataclass(frozen=True)
class VisionCheckpointResult:
    frame: pl.DataFrame
    paths: list[Path]
    completed: int
    newly_completed: int
    skipped_existing: int
    gpu_used: bool = False
    gpu_name: str = ""
    rows_per_file: dict[str, int] | None = None


def build_checkpointed_vision_predictions(
    bronze: pl.DataFrame,
    vision_classifier: VisionClassifier | None,
    checkpoint_dir: str | Path,
    *,
    run_id: str = DEFAULT_RUN_ID,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> VisionCheckpointResult:
    output_dir = Path(checkpoint_dir)
    if vision_classifier is None or not bronze.height:
        return VisionCheckpointResult(frame=pl.DataFrame(), paths=[], completed=0, newly_completed=0, skipped_existing=0, rows_per_file={})

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_paths = _prediction_part_paths(output_dir)
    existing_frame = _read_prediction_parts(existing_paths)
    existing_keys = _prediction_keys(existing_frame)
    rows_to_process = [row for row in bronze.to_dicts() if row.get("image_url")]
    new_predictions: list[dict[str, object]] = []
    skipped = 0
    gpu_used = False
    gpu_name = ""
    try:
        for prediction in _classify_rows(rows_to_process, vision_classifier):
            key = _prediction_key(prediction)
            if key in existing_keys:
                skipped += 1
                continue
            existing_keys.add(key)
            new_predictions.append(prediction)
    finally:
        scorer = getattr(vision_classifier, "scorer", None)
        device = getattr(scorer, "device", "")
        gpu_used = device == "cuda"
        gpu_name = str(getattr(scorer, "gpu_name", "") or "")
        close = getattr(vision_classifier, "close", None)
        if callable(close):
            close()

    new_paths = _write_prediction_shards(
        new_predictions,
        output_dir=output_dir,
        run_id=run_id,
        shard_size=shard_size,
        existing_path_count=len(existing_paths),
    )
    paths = sorted([*existing_paths, *new_paths])
    rows_per_file = _rows_per_file(paths)
    if not paths:
        return VisionCheckpointResult(
            frame=pl.DataFrame(),
            paths=[],
            completed=0,
            newly_completed=len(new_predictions),
            skipped_existing=skipped,
            gpu_used=gpu_used,
            gpu_name=gpu_name,
            rows_per_file={},
        )
    frame = pl.read_parquet(paths)
    return VisionCheckpointResult(
        frame=frame,
        paths=paths,
        completed=frame.height,
        newly_completed=len(new_predictions),
        skipped_existing=skipped,
        gpu_used=gpu_used,
        gpu_name=gpu_name,
        rows_per_file=rows_per_file,
    )


def _classify_rows(rows: list[dict[str, object]], vision_classifier: VisionClassifier) -> list[dict[str, object]]:
    classify_rows = getattr(vision_classifier, "classify_rows", None)
    if callable(classify_rows):
        return list(classify_rows(rows))
    return [vision_classifier(row) for row in rows]


def _write_prediction_shards(
    predictions: list[dict[str, object]],
    *,
    output_dir: Path,
    run_id: str,
    shard_size: int,
    existing_path_count: int,
) -> list[Path]:
    if not predictions:
        return []
    written: list[Path] = []
    safe_shard_size = max(1, shard_size)
    for model_version, model_predictions in _group_by_model_version(predictions).items():
        for index in range(0, len(model_predictions), safe_shard_size):
            shard_index = existing_path_count + len(written)
            target_dir = output_dir / f"model_version={model_version}" / f"run_id={run_id}" / f"shard_id={shard_index:05d}"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / "part-00000.parquet"
            pl.DataFrame(model_predictions[index : index + safe_shard_size]).write_parquet(target)
            written.append(target)
    return written


def _group_by_model_version(predictions: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for prediction in predictions:
        model_version = str(prediction.get("model_version") or "unknown")
        grouped.setdefault(model_version, []).append(prediction)
    return grouped


def _prediction_part_paths(output_dir: Path) -> list[Path]:
    return sorted(output_dir.rglob("part-*.parquet"))


def _read_prediction_parts(paths: list[Path]) -> pl.DataFrame:
    if not paths:
        return pl.DataFrame()
    return pl.read_parquet(paths)


def _prediction_keys(frame: pl.DataFrame) -> set[tuple[str, str, str, str]]:
    if frame.is_empty() or not set(PREDICTION_KEY).issubset(set(frame.columns)):
        return set()
    return {_prediction_key(row) for row in frame.to_dicts()}


def _prediction_key(prediction: dict[str, object]) -> tuple[str, str, str, str]:
    return tuple(str(prediction.get(column) or "") for column in PREDICTION_KEY)  # type: ignore[return-value]


def _rows_per_file(paths: list[Path]) -> dict[str, int]:
    return {str(path): pl.read_parquet(path).height for path in paths}
