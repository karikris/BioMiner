from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from biominer.storage.parquet import write_parquet


TEXT_EMBEDDING_COLUMNS = (
    "candidate_set_id",
    "label",
    "accepted_taxon_key",
    "rank",
    "model_id",
    "model_checkpoint",
    "embedding_dim",
    "embedding",
    "created_at",
)
IMAGE_EMBEDDING_COLUMNS = (
    "source",
    "flickr_photo_id",
    "detection_id",
    "crop_hash",
    "model_id",
    "model_checkpoint",
    "embedding_dim",
    "embedding",
    "created_at",
)


def read_embedding_cache(path: str | Path) -> pl.DataFrame:
    source = Path(path)
    return pl.read_parquet(source) if source.exists() else pl.DataFrame()


def write_text_embedding_cache(rows: list[dict[str, Any]], path: str | Path) -> Path:
    frame = _dedupe(pl.DataFrame(rows), ["candidate_set_id", "label", "model_id", "model_checkpoint"])
    return write_parquet(frame, path)


def write_image_embedding_cache(rows: list[dict[str, Any]], path: str | Path) -> Path:
    frame = _dedupe(pl.DataFrame(rows), ["source", "flickr_photo_id", "detection_id", "crop_hash", "model_id", "model_checkpoint"])
    return write_parquet(frame, path)


def cached_text_labels(cache: pl.DataFrame, *, candidate_set_id: str, model_id: str, model_checkpoint: str) -> set[str]:
    if cache.is_empty():
        return set()
    return set(
        cache.filter(
            (pl.col("candidate_set_id") == candidate_set_id)
            & (pl.col("model_id") == model_id)
            & (pl.col("model_checkpoint") == model_checkpoint)
        )
        .get_column("label")
        .to_list()
    )


def cached_crop_hashes(cache: pl.DataFrame, *, model_id: str, model_checkpoint: str) -> set[str]:
    if cache.is_empty():
        return set()
    return set(
        cache.filter((pl.col("model_id") == model_id) & (pl.col("model_checkpoint") == model_checkpoint))
        .get_column("crop_hash")
        .to_list()
    )


def _dedupe(frame: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return frame.unique(subset=keys, maintain_order=True)
