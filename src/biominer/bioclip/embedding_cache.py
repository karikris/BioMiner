from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from pathlib import Path
from typing import Any, Callable

import polars as pl

from biominer.bioclip.candidate_sets import CandidateSet, CandidateTaxon
from biominer.bioclip.taxonomy_store import ButterflyTaxonomyStore
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
TAXONOMY_TEXT_EMBEDDING_COLUMNS = (
    "classification_table_version",
    "prompt_variant_version",
    "label_scope",
    "label",
    "label_hash",
    "accepted_taxon_key",
    "family_key",
    "rank",
    "model_id",
    "model_checkpoint",
    "embedding_dim",
    "embedding_dtype",
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
    batch_size: int | None = None,
    created_at: str | None = None,
) -> EmbeddingCacheUpdate:
    rows = candidate_text_embedding_rows(candidate_set, model_id=model_id, model_checkpoint=model_checkpoint)
    return upsert_text_embedding_cache(rows, path, embed_labels=embed_labels, batch_size=batch_size, created_at=created_at)


def taxonomy_text_embedding_rows(
    taxonomy_store: ButterflyTaxonomyStore,
    *,
    model_id: str,
    model_checkpoint: str,
    embedding_dtype: str = "float32",
) -> list[dict[str, Any]]:
    manifest = dict(taxonomy_store.manifest or {})
    classification_table_version = str(manifest.get("classification_table_version") or _first_value(taxonomy_store.classification_taxa, "classification_table_version") or "")
    prompt_variant_version = str(manifest.get("prompt_variant_version") or _first_value(taxonomy_store.family_labels, "prompt_variant_version") or "")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for row in _taxonomy_label_rows(taxonomy_store):
        label = " ".join(str(row.get("label") or "").split())
        if not label:
            continue
        output = {
            "classification_table_version": classification_table_version,
            "prompt_variant_version": prompt_variant_version,
            "label_scope": row["label_scope"],
            "label": label,
            "label_hash": _label_hash(label),
            "accepted_taxon_key": row["accepted_taxon_key"],
            "family_key": row["family_key"],
            "rank": row["rank"],
            "model_id": model_id,
            "model_checkpoint": model_checkpoint,
            "embedding_dtype": embedding_dtype,
        }
        row_key = _key(output, ["classification_table_version", "prompt_variant_version", "label", "model_id", "model_checkpoint"])
        if row_key in seen:
            continue
        seen.add(row_key)
        rows.append(output)
    return rows


def prepare_taxonomy_text_embedding_cache(
    taxonomy_store: ButterflyTaxonomyStore,
    path: str | Path,
    *,
    model_id: str,
    model_checkpoint: str,
    embed_labels: Callable[[list[str]], list[list[float]]],
    batch_size: int | None = None,
    embedding_dtype: str = "float32",
    created_at: str | None = None,
) -> EmbeddingCacheUpdate:
    rows = taxonomy_text_embedding_rows(
        taxonomy_store,
        model_id=model_id,
        model_checkpoint=model_checkpoint,
        embedding_dtype=embedding_dtype,
    )
    return upsert_taxonomy_text_embedding_cache(
        rows,
        path,
        embed_labels=embed_labels,
        batch_size=batch_size,
        created_at=created_at,
    )


def prepare_object_image_embedding_cache(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    model_id: str,
    model_checkpoint: str,
    crop_path_by_hash: dict[str, Path],
    embed_image_paths: Callable[[list[Path]], list[list[float]]],
    created_at: str | None = None,
) -> EmbeddingCacheUpdate:
    requested = [
        {
            "source": row.get("source"),
            "flickr_photo_id": row.get("flickr_photo_id"),
            "detection_id": row.get("detection_id"),
            "crop_hash": row.get("crop_hash"),
            "model_id": model_id,
            "model_checkpoint": model_checkpoint,
        }
        for row in rows
        if row.get("crop_hash")
    ]

    def embed_missing(missing_rows: list[dict[str, Any]]) -> list[list[float]]:
        paths = [_crop_path_for_row(row, crop_path_by_hash=crop_path_by_hash) for row in missing_rows]
        return embed_image_paths(paths)

    return upsert_image_embedding_cache(requested, path, embed_images=embed_missing, created_at=created_at)


def read_embedding_cache(path: str | Path) -> pl.DataFrame:
    source = Path(path)
    return pl.read_parquet(source) if source.exists() else pl.DataFrame()


def write_text_embedding_cache(rows: list[dict[str, Any]], path: str | Path) -> Path:
    frame = _dedupe(pl.DataFrame(rows), ["candidate_set_id", "label", "model_id", "model_checkpoint"])
    return write_parquet(frame, path)


def write_image_embedding_cache(rows: list[dict[str, Any]], path: str | Path) -> Path:
    frame = _dedupe(pl.DataFrame(rows), ["source", "flickr_photo_id", "detection_id", "crop_hash", "model_id", "model_checkpoint"])
    return write_parquet(frame, path)


def write_taxonomy_text_embedding_cache(rows: list[dict[str, Any]], path: str | Path) -> Path:
    frame = _dedupe(pl.DataFrame(rows), ["classification_table_version", "prompt_variant_version", "label", "model_id", "model_checkpoint"])
    return write_parquet(frame, path)


def upsert_text_embedding_cache(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    embed_labels: Callable[[list[str]], list[list[float]]],
    batch_size: int | None = None,
    created_at: str | None = None,
) -> EmbeddingCacheUpdate:
    cache = read_embedding_cache(path)
    requested = _dedupe_request_rows(rows, ["candidate_set_id", "label", "model_id", "model_checkpoint"])
    existing_keys = _row_keys(cache, ["candidate_set_id", "label", "model_id", "model_checkpoint"])
    missing = [row for row in requested if _key(row, ["candidate_set_id", "label", "model_id", "model_checkpoint"]) not in existing_keys]
    rows_reused = len(requested) - len(missing)
    labels = [str(row.get("label") or "") for row in missing]
    embeddings = _embed_label_batches(labels, embed_labels=embed_labels, batch_size=batch_size)
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
        rows_reused=rows_reused,
        embeddings_computed=len(embeddings),
    )


def upsert_taxonomy_text_embedding_cache(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    embed_labels: Callable[[list[str]], list[list[float]]],
    batch_size: int | None = None,
    created_at: str | None = None,
) -> EmbeddingCacheUpdate:
    cache = read_embedding_cache(path)
    row_keys = ["classification_table_version", "prompt_variant_version", "label", "model_id", "model_checkpoint"]
    requested = _dedupe_request_rows(rows, row_keys)
    existing_keys = _row_keys(cache, row_keys)
    missing = [row for row in requested if _key(row, row_keys) not in existing_keys]
    rows_reused = len(requested) - len(missing)
    labels = [str(row.get("label") or "") for row in missing]
    embeddings = _embed_label_batches(labels, embed_labels=embed_labels, batch_size=batch_size)
    if len(embeddings) != len(missing):
        raise ValueError("embed_labels must return one embedding per missing taxonomy label")
    new_rows = [
        {
            **row,
            "embedding_dim": len(embedding),
            "embedding": [float(value) for value in embedding],
            "created_at": created_at or _now_iso(),
        }
        for row, embedding in zip(missing, embeddings, strict=True)
    ]
    frame = _append_and_dedupe(cache, new_rows, row_keys)
    output = write_parquet(frame, path) if new_rows else Path(path)
    return EmbeddingCacheUpdate(
        output_path=output,
        frame=frame,
        rows_total=frame.height,
        rows_added=len(new_rows),
        rows_reused=rows_reused,
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


def missing_taxonomy_text_embedding_labels(
    cache: pl.DataFrame,
    taxonomy_store: ButterflyTaxonomyStore,
    *,
    model_id: str,
    model_checkpoint: str,
) -> list[str]:
    requested = taxonomy_text_embedding_rows(taxonomy_store, model_id=model_id, model_checkpoint=model_checkpoint)
    if not requested:
        return []
    cache_keys = _row_keys(
        cache.filter((pl.col("model_id") == model_id) & (pl.col("model_checkpoint") == model_checkpoint))
        if not cache.is_empty() and {"model_id", "model_checkpoint"}.issubset(set(cache.columns))
        else pl.DataFrame(),
        ["classification_table_version", "prompt_variant_version", "label", "model_id", "model_checkpoint"],
    )
    return [
        str(row["label"])
        for row in requested
        if _key(row, ["classification_table_version", "prompt_variant_version", "label", "model_id", "model_checkpoint"]) not in cache_keys
    ]


def validate_taxonomy_text_embedding_cache(
    cache: pl.DataFrame,
    taxonomy_store: ButterflyTaxonomyStore,
    *,
    model_id: str,
    model_checkpoint: str,
) -> None:
    if cache.is_empty():
        raise ValueError("taxonomy text embedding cache is empty")
    required = set(TAXONOMY_TEXT_EMBEDDING_COLUMNS)
    missing_columns = sorted(required - set(cache.columns))
    if missing_columns:
        raise ValueError("taxonomy text embedding cache is missing columns: " + ", ".join(missing_columns))
    matching = cache.filter((pl.col("model_id") == model_id) & (pl.col("model_checkpoint") == model_checkpoint))
    if matching.is_empty():
        raise ValueError(f"taxonomy text embedding cache has no rows for model_checkpoint={model_checkpoint!r}")
    missing_labels = missing_taxonomy_text_embedding_labels(cache, taxonomy_store, model_id=model_id, model_checkpoint=model_checkpoint)
    if missing_labels:
        preview = ", ".join(missing_labels[:5])
        suffix = "" if len(missing_labels) <= 5 else f" (+{len(missing_labels) - 5} more)"
        raise ValueError(f"taxonomy text embedding cache missing labels: {preview}{suffix}")
    requested = taxonomy_text_embedding_rows(taxonomy_store, model_id=model_id, model_checkpoint=model_checkpoint)
    _validate_taxonomy_text_embedding_metadata(matching, requested)


def _validate_taxonomy_text_embedding_metadata(
    matching: pl.DataFrame,
    requested: list[dict[str, Any]],
) -> None:
    keys = ["classification_table_version", "prompt_variant_version", "label", "model_id", "model_checkpoint"]
    requested_by_key = {_key(row, keys): row for row in requested}
    expected_dims: set[int] = set()
    embedding_dtypes: set[str] = set()
    for row in matching.to_dicts():
        expected = requested_by_key.get(_key(row, keys))
        if expected is None:
            continue
        label = str(row.get("label") or "")
        expected_label_hash = str(expected.get("label_hash") or "")
        if str(row.get("label_hash") or "") != expected_label_hash:
            raise ValueError(f"taxonomy text embedding cache label_hash mismatch for label={label!r}")
        embedding_dtype = str(row.get("embedding_dtype") or "").strip()
        if not embedding_dtype:
            raise ValueError(f"taxonomy text embedding cache missing embedding_dtype for label={label!r}")
        embedding_dtypes.add(embedding_dtype)
        row_dim = _positive_embedding_dim(row.get("embedding_dim"), label=label)
        embedding = row.get("embedding")
        if embedding is None:
            raise ValueError(f"taxonomy text embedding cache missing embedding for label={label!r}")
        vector = [float(value) for value in embedding]
        if len(vector) != row_dim:
            raise ValueError(f"taxonomy text embedding cache embedding_dim mismatch for label={label!r}")
        expected_dims.add(row_dim)
    if len(expected_dims) > 1:
        raise ValueError("taxonomy text embedding cache has inconsistent embedding_dim values")
    if len(embedding_dtypes) > 1:
        raise ValueError("taxonomy text embedding cache has inconsistent embedding_dtype values")


def _positive_embedding_dim(value: Any, *, label: str) -> int:
    try:
        dim = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"taxonomy text embedding cache invalid embedding_dim for label={label!r}") from exc
    if dim <= 0:
        raise ValueError(f"taxonomy text embedding cache invalid embedding_dim for label={label!r}")
    return dim


def _taxonomy_label_rows(taxonomy_store: ButterflyTaxonomyStore) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in taxonomy_store.family_labels.to_dicts():
        if row.get("enabled") is False:
            continue
        rows.append(
            {
                "label_scope": "family",
                "label": str(row.get("label") or ""),
                "accepted_taxon_key": str(row.get("family_key") or ""),
                "family_key": str(row.get("family_key") or ""),
                "rank": "FAMILY",
            }
        )
    for row in taxonomy_store.species_labels.to_dicts():
        if row.get("enabled") is False:
            continue
        rows.append(
            {
                "label_scope": "species",
                "label": str(row.get("label") or ""),
                "accepted_taxon_key": str(row.get("accepted_taxon_key") or ""),
                "family_key": str(row.get("family_key") or ""),
                "rank": "SPECIES",
            }
        )
    return rows


def _first_value(frame: pl.DataFrame, column: str) -> object:
    if column not in frame.columns or frame.is_empty():
        return None
    values = frame.select(column).to_series().drop_nulls().to_list()
    return values[0] if values else None


def _label_hash(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _dedupe(frame: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return frame.unique(subset=keys, maintain_order=True)


def _embed_label_batches(
    labels: list[str],
    *,
    embed_labels: Callable[[list[str]], list[list[float]]],
    batch_size: int | None,
) -> list[list[float]]:
    if not labels:
        return []
    if batch_size is None:
        return embed_labels(labels)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    embeddings: list[list[float]] = []
    for start in range(0, len(labels), batch_size):
        batch = labels[start : start + batch_size]
        embeddings.extend(embed_labels(batch))
    return embeddings


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


def _crop_path_for_row(row: dict[str, Any], *, crop_path_by_hash: dict[str, Path]) -> Path:
    crop_hash = str(row.get("crop_hash") or "")
    try:
        return crop_path_by_hash[crop_hash]
    except KeyError as exc:
        raise KeyError(f"missing crop path for crop_hash={crop_hash!r}") from exc


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
