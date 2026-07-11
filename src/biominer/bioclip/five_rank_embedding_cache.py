from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
import hashlib
import json
from typing import Callable, Sequence

import polars as pl

from biominer.bioclip.five_rank_store import FiveRankTaxonomyStore


FIVE_RANK_TEXT_EMBEDDING_SCHEMA: dict[str, pl.DataType] = {
    "classification_version": pl.String,
    "prompt_version": pl.String,
    "taxonomy_fingerprint": pl.String,
    "embedding_cache_fingerprint": pl.String,
    "node_id": pl.String,
    "rank": pl.String,
    "label": pl.String,
    "label_hash": pl.String,
    "model_id": pl.String,
    "model_checkpoint": pl.String,
    "embedding_dim": pl.Int64,
    "embedding": pl.List(pl.Float64),
}


@dataclass(frozen=True)
class FiveRankTextEmbeddingIndex:
    model_id: str
    model_checkpoint: str
    taxonomy_fingerprint: str
    cache_fingerprint: str
    vectors_by_label: dict[str, tuple[float, ...]]

    @classmethod
    def from_frame(
        cls,
        frame: pl.DataFrame,
        *,
        taxonomy_store: FiveRankTaxonomyStore,
        model_id: str,
        model_checkpoint: str,
    ) -> FiveRankTextEmbeddingIndex:
        cache = _ensure_schema(frame)
        if cache.is_empty():
            raise ValueError("classification-v2 text embedding cache is empty")
        _require_single(cache, "classification_version", taxonomy_store.classification_version)
        _require_single(cache, "prompt_version", taxonomy_store.prompt_version)
        _require_single(cache, "taxonomy_fingerprint", taxonomy_store.taxonomy_fingerprint)
        _require_single(cache, "model_id", model_id)
        _require_single(cache, "model_checkpoint", model_checkpoint)
        cache_fingerprints = cache["embedding_cache_fingerprint"].unique().to_list()
        if len(cache_fingerprints) != 1 or not str(cache_fingerprints[0] or ""):
            raise ValueError("classification-v2 text embedding cache has mixed fingerprints")
        expected_labels = set(taxonomy_store.prompt_labels.filter(pl.col("enabled"))["label"].to_list())
        actual_labels = set(cache["label"].to_list())
        if actual_labels != expected_labels:
            raise ValueError("classification-v2 text embedding cache label set mismatch")
        vectors: dict[str, tuple[float, ...]] = {}
        dimensions: set[int] = set()
        for row in cache.iter_rows(named=True):
            label = str(row["label"])
            vector = _normalize(row["embedding"])
            dimensions.add(len(vector))
            if label in vectors:
                raise ValueError(f"classification-v2 text embedding cache has duplicate label: {label}")
            if int(row["embedding_dim"]) != len(vector):
                raise ValueError(f"classification-v2 text embedding dimension mismatch: {label}")
            vectors[label] = vector
        if len(dimensions) != 1:
            raise ValueError("classification-v2 text embedding cache has mixed dimensions")
        expected_fingerprint = _cache_fingerprint(cache.drop("embedding_cache_fingerprint"))
        if str(cache_fingerprints[0]) != expected_fingerprint:
            raise ValueError("classification-v2 text embedding cache fingerprint mismatch")
        return cls(
            model_id=model_id,
            model_checkpoint=model_checkpoint,
            taxonomy_fingerprint=taxonomy_store.taxonomy_fingerprint,
            cache_fingerprint=expected_fingerprint,
            vectors_by_label=vectors,
        )

    def score(self, image_embedding: Sequence[float], labels: Sequence[str]) -> dict[str, float]:
        requested = tuple(str(label) for label in labels)
        missing = [label for label in requested if label not in self.vectors_by_label]
        if missing:
            raise ValueError("classification-v2 text embedding cache missing labels: " + ", ".join(missing[:5]))
        image = _normalize(image_embedding)
        logits = [100.0 * _dot(image, self.vectors_by_label[label]) for label in requested]
        probabilities = _softmax(logits)
        return dict(zip(requested, probabilities, strict=True))


def build_five_rank_text_embedding_cache(
    taxonomy_store: FiveRankTaxonomyStore,
    *,
    model_id: str,
    model_checkpoint: str,
    embed_labels: Callable[[list[str]], list[list[float]]],
    batch_size: int = 256,
) -> pl.DataFrame:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    prompts = taxonomy_store.prompt_labels.filter(pl.col("enabled")).sort(["rank", "scientific_name", "label"])
    rows: list[dict[str, object]] = []
    prompt_rows = prompts.iter_rows(named=True)
    batch: list[dict[str, object]] = []
    for row in prompt_rows:
        batch.append(row)
        if len(batch) >= batch_size:
            rows.extend(_embed_prompt_batch(batch, taxonomy_store, model_id, model_checkpoint, embed_labels))
            batch = []
    if batch:
        rows.extend(_embed_prompt_batch(batch, taxonomy_store, model_id, model_checkpoint, embed_labels))
    frame = _ensure_schema(pl.DataFrame(rows))
    fingerprint = _cache_fingerprint(frame.drop("embedding_cache_fingerprint"))
    return frame.with_columns(pl.lit(fingerprint).alias("embedding_cache_fingerprint"))


def _embed_prompt_batch(
    rows: list[dict[str, object]],
    store: FiveRankTaxonomyStore,
    model_id: str,
    model_checkpoint: str,
    embed_labels: Callable[[list[str]], list[list[float]]],
) -> list[dict[str, object]]:
    labels = [str(row["label"]) for row in rows]
    vectors = embed_labels(labels)
    if len(vectors) != len(labels):
        raise ValueError(f"text embedder returned {len(vectors)} rows for {len(labels)} labels")
    output = []
    for row, label, vector in zip(rows, labels, vectors, strict=True):
        normalized = list(_normalize(vector))
        output.append(
            {
                "classification_version": store.classification_version,
                "prompt_version": store.prompt_version,
                "taxonomy_fingerprint": store.taxonomy_fingerprint,
                "embedding_cache_fingerprint": "",
                "node_id": str(row["node_id"]),
                "rank": str(row["rank"]),
                "label": label,
                "label_hash": "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest(),
                "model_id": model_id,
                "model_checkpoint": model_checkpoint,
                "embedding_dim": len(normalized),
                "embedding": normalized,
            }
        )
    return output


def _ensure_schema(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty() and not frame.columns:
        return pl.DataFrame(schema=FIVE_RANK_TEXT_EMBEDDING_SCHEMA)
    expressions = [
        pl.col(name).cast(dtype).alias(name)
        if name in frame.columns
        else pl.lit(_default(dtype), dtype=dtype).alias(name)
        for name, dtype in FIVE_RANK_TEXT_EMBEDDING_SCHEMA.items()
    ]
    return frame.select(expressions).sort(["rank", "node_id", "label"])


def _default(dtype: pl.DataType) -> object:
    if dtype == pl.String:
        return ""
    if dtype == pl.List(pl.Float64):
        return []
    return 0


def _require_single(frame: pl.DataFrame, column: str, expected: str) -> None:
    values = [str(value or "") for value in frame[column].unique().to_list()]
    if values != [expected]:
        raise ValueError(f"classification-v2 text embedding cache {column} mismatch")


def _cache_fingerprint(frame: pl.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.sort(["rank", "node_id", "label"]).iter_rows(named=True):
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def _normalize(values: Sequence[float]) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    norm = sqrt(sum(value * value for value in vector))
    if not vector or norm <= 0:
        raise ValueError("embedding vector must be non-empty with non-zero norm")
    return tuple(value / norm for value in vector)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("image and text embedding dimensions do not match")
    return sum(a * b for a, b in zip(left, right, strict=True))


def _softmax(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        return ()
    maximum = max(values)
    weights = [exp(value - maximum) for value in values]
    total = sum(weights)
    return tuple(weight / total for weight in weights)


__all__ = [
    "FIVE_RANK_TEXT_EMBEDDING_SCHEMA",
    "FiveRankTextEmbeddingIndex",
    "build_five_rank_text_embedding_cache",
]
