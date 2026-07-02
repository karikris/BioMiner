from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import polars as pl

from biominer.bioclip.candidate_sets import CandidateSet, CandidateTaxon
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


@dataclass(frozen=True)
class EmbeddingCacheUpdate:
    output_path: Path
    frame: pl.DataFrame
    rows_total: int
    rows_added: int
    rows_reused: int
    embeddings_computed: int


def candidate_text_embedding_rows(
    candidate_set: CandidateSet,
    *,
    model_id: str,
    model_checkpoint: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for candidate, rank in (
        *((candidate, "family") for candidate in candidate_set.family_candidates),
        *((candidate, "genus") for candidate in candidate_set.genus_candidates),
        *((candidate, "genus") for candidate in candidate_set.family_candidates),
        *((candidate, "species") for candidate in candidate_set.species_candidates),
    ):
        for label in _candidate_prompt_labels(candidate, rank=rank):
            row = {
                "candidate_set_id": candidate_set.candidate_set_id,
                "label": label,
                "accepted_taxon_key": candidate.accepted_taxon_key if candidate.rank == rank else None,
                "rank": rank,
                "model_id": model_id,
                "model_checkpoint": model_checkpoint,
            }
            row_key = _key(row, ["candidate_set_id", "label", "model_id", "model_checkpoint"])
            if row_key in seen:
                continue
            seen.add(row_key)
            rows.append(row)
    return rows


def prepare_candidate_text_embedding_cache(
    candidate_set: CandidateSet,
    path: str | Path,
    *,
    model_id: str,
    model_checkpoint: str,
    embed_labels: Callable[[list[str]], list[list[float]]],
    created_at: str | None = None,
) -> EmbeddingCacheUpdate:
    rows = candidate_text_embedding_rows(candidate_set, model_id=model_id, model_checkpoint=model_checkpoint)
    return upsert_text_embedding_cache(rows, path, embed_labels=embed_labels, created_at=created_at)


def read_embedding_cache(path: str | Path) -> pl.DataFrame:
    source = Path(path)
    return pl.read_parquet(source) if source.exists() else pl.DataFrame()


def write_text_embedding_cache(rows: list[dict[str, Any]], path: str | Path) -> Path:
    frame = _dedupe(pl.DataFrame(rows), ["candidate_set_id", "label", "model_id", "model_checkpoint"])
    return write_parquet(frame, path)


def write_image_embedding_cache(rows: list[dict[str, Any]], path: str | Path) -> Path:
    frame = _dedupe(pl.DataFrame(rows), ["source", "flickr_photo_id", "detection_id", "crop_hash", "model_id", "model_checkpoint"])
    return write_parquet(frame, path)


def upsert_text_embedding_cache(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    embed_labels: Callable[[list[str]], list[list[float]]],
    created_at: str | None = None,
) -> EmbeddingCacheUpdate:
    cache = read_embedding_cache(path)
    requested = _dedupe_request_rows(rows, ["candidate_set_id", "label", "model_id", "model_checkpoint"])
    existing_keys = _row_keys(cache, ["candidate_set_id", "label", "model_id", "model_checkpoint"])
    missing = [row for row in requested if _key(row, ["candidate_set_id", "label", "model_id", "model_checkpoint"]) not in existing_keys]
    labels = [str(row.get("label") or "") for row in missing]
    embeddings = embed_labels(labels) if labels else []
    if len(embeddings) != len(missing):
        raise ValueError("embed_labels must return one embedding per missing label")
    new_rows = [
        {
            **row,
            "embedding_dim": len(embedding),
            "embedding": [float(value) for value in embedding],
            "created_at": created_at or _now_iso(),
        }
        for row, embedding in zip(missing, embeddings, strict=True)
    ]
    frame = _append_and_dedupe(cache, new_rows, ["candidate_set_id", "label", "model_id", "model_checkpoint"])
    output = write_parquet(frame, path) if new_rows else Path(path)
    return EmbeddingCacheUpdate(
        output_path=output,
        frame=frame,
        rows_total=frame.height,
        rows_added=len(new_rows),
        rows_reused=0,
        embeddings_computed=len(embeddings),
    )


def upsert_image_embedding_cache(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    embed_images: Callable[[list[dict[str, Any]]], list[list[float]]],
    created_at: str | None = None,
) -> EmbeddingCacheUpdate:
    cache = read_embedding_cache(path)
    row_keys = ["source", "flickr_photo_id", "detection_id", "crop_hash", "model_id", "model_checkpoint"]
    requested = _dedupe_request_rows(rows, row_keys)
    existing_row_keys = _row_keys(cache, row_keys)
    crop_embedding_by_key = _crop_embedding_map(cache)
    rows_to_add: list[dict[str, Any]] = []
    rows_to_compute: list[dict[str, Any]] = []
    rows_reused = 0
    pending_crop_keys: set[tuple[str, ...]] = set()
    for row in requested:
        if _key(row, row_keys) in existing_row_keys:
            continue
        crop_key = _key(row, ["crop_hash", "model_id", "model_checkpoint"])
        cached = crop_embedding_by_key.get(crop_key)
        if cached is not None:
            rows_to_add.append(_image_row_with_embedding(row, cached, created_at=created_at))
            rows_reused += 1
            continue
        if crop_key in pending_crop_keys:
            continue
        pending_crop_keys.add(crop_key)
        rows_to_compute.append(row)

    embeddings = embed_images(rows_to_compute) if rows_to_compute else []
    if len(embeddings) != len(rows_to_compute):
        raise ValueError("embed_images must return one embedding per missing crop")
    computed_by_crop_key: dict[tuple[str, ...], list[float]] = {}
    for row, embedding in zip(rows_to_compute, embeddings, strict=True):
        values = [float(value) for value in embedding]
        computed_by_crop_key[_key(row, ["crop_hash", "model_id", "model_checkpoint"])] = values
        rows_to_add.append(_image_row_with_embedding(row, values, created_at=created_at))

    for row in requested:
        if _key(row, row_keys) in existing_row_keys:
            continue
        if any(_key(row, row_keys) == _key(existing, row_keys) for existing in rows_to_add):
            continue
        computed = computed_by_crop_key.get(_key(row, ["crop_hash", "model_id", "model_checkpoint"]))
        if computed is not None:
            rows_to_add.append(_image_row_with_embedding(row, computed, created_at=created_at))
            rows_reused += 1

    frame = _append_and_dedupe(cache, rows_to_add, row_keys)
    output = write_parquet(frame, path) if rows_to_add else Path(path)
    return EmbeddingCacheUpdate(
        output_path=output,
        frame=frame,
        rows_total=frame.height,
        rows_added=len(rows_to_add),
        rows_reused=rows_reused,
        embeddings_computed=len(embeddings),
    )


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


def _append_and_dedupe(cache: pl.DataFrame, new_rows: list[dict[str, Any]], keys: list[str]) -> pl.DataFrame:
    if not new_rows:
        return cache
    new_frame = pl.DataFrame(new_rows)
    if cache.is_empty():
        return _dedupe(new_frame, keys)
    return _dedupe(pl.concat([cache, new_frame], how="diagonal_relaxed"), keys)


def _dedupe_request_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        row_key = _key(row, keys)
        if row_key in seen:
            continue
        seen.add(row_key)
        output.append(row)
    return output


def _row_keys(frame: pl.DataFrame, keys: list[str]) -> set[tuple[str, ...]]:
    if frame.is_empty():
        return set()
    return {_key(row, keys) for row in frame.select(keys).to_dicts()}


def _crop_embedding_map(frame: pl.DataFrame) -> dict[tuple[str, ...], list[float]]:
    if frame.is_empty():
        return {}
    output: dict[tuple[str, ...], list[float]] = {}
    for row in frame.select(["crop_hash", "model_id", "model_checkpoint", "embedding"]).to_dicts():
        output.setdefault(_key(row, ["crop_hash", "model_id", "model_checkpoint"]), [float(value) for value in row["embedding"]])
    return output


def _image_row_with_embedding(row: dict[str, Any], embedding: list[float], *, created_at: str | None) -> dict[str, Any]:
    return {
        **row,
        "embedding_dim": len(embedding),
        "embedding": [float(value) for value in embedding],
        "created_at": created_at or _now_iso(),
    }


def _candidate_prompt_labels(candidate: CandidateTaxon, *, rank: str) -> tuple[str, ...]:
    if rank == "family":
        return _unique_labels([candidate.family or (candidate.scientific_name if candidate.rank == "family" else "")])
    if rank == "genus":
        return _unique_labels([candidate.genus or (candidate.scientific_name if candidate.rank == "genus" else "")])
    labels = [candidate.scientific_name, f"a photo of {candidate.scientific_name}", *candidate.common_names]
    return _unique_labels(labels)


def _unique_labels(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = " ".join(str(value or "").split())
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return tuple(output)


def _key(row: dict[str, Any], keys: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(key) or "") for key in keys)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
