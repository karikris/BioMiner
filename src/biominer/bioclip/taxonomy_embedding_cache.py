from __future__ import annotations

from array import array
from dataclasses import dataclass
from math import exp, isfinite, sqrt
import hashlib
import json
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import polars as pl

from biominer.bioclip.path_taxonomy_store import PathTaxonomyStore
from biominer.registry.classification_v3 import CLASSIFICATION_PROMPT_STAGES


TAXONOMY_TEXT_EMBEDDING_SCHEMA: dict[str, pl.DataType] = {
    "classification_version": pl.String,
    "prompt_version": pl.String,
    "prompt_stage": pl.String,
    "hierarchy_fingerprint": pl.String,
    "embedding_cache_fingerprint": pl.String,
    "label": pl.String,
    "label_hash": pl.String,
    "model_id": pl.String,
    "model_checkpoint": pl.String,
    "embedding_dim": pl.Int64,
    "embedding": pl.List(pl.Float32),
}

_UNIT_NORM_TOLERANCE = 1e-5


@dataclass(frozen=True)
class TaxonomyTextEmbeddingIndex:
    classification_version: str
    prompt_version: str
    hierarchy_fingerprint: str
    model_id: str
    model_checkpoint: str
    cache_fingerprint: str
    embedding_dim: int
    prompt_stages: tuple[str, ...]
    _offset_by_label: Mapping[str, int]
    _stage_by_label: Mapping[str, str]
    _vectors: bytes

    @classmethod
    def from_frame(
        cls,
        frame: pl.DataFrame,
        *,
        taxonomy_store: PathTaxonomyStore,
        model_id: str,
        model_checkpoint: str,
    ) -> TaxonomyTextEmbeddingIndex:
        cache = _require_exact_schema(frame)
        if cache.is_empty():
            raise ValueError("taxonomy text embedding cache is empty")
        _require_single(cache, "classification_version", taxonomy_store.classification_version)
        _require_single(cache, "prompt_version", taxonomy_store.prompt_version)
        _require_single(cache, "hierarchy_fingerprint", taxonomy_store.hierarchy_fingerprint)
        _require_single(cache, "model_id", model_id)
        _require_single(cache, "model_checkpoint", model_checkpoint)
        cache_fingerprint = _single_nonblank(cache, "embedding_cache_fingerprint")
        expected_fingerprint = taxonomy_text_embedding_cache_fingerprint(
            cache.drop("embedding_cache_fingerprint")
        )
        if cache_fingerprint != expected_fingerprint:
            raise ValueError("taxonomy text embedding cache fingerprint mismatch")

        expected_identities = _expected_prompt_identities(taxonomy_store)
        actual_identities: set[tuple[str, str, str]] = set()
        stage_by_label: dict[str, str] = {}
        rows = cache.sort(["prompt_stage", "label_hash", "label"]).iter_rows(named=True)
        vector_values = array("f")
        offset_by_label: dict[str, int] = {}
        embedding_dim: int | None = None
        for row in rows:
            stage = _prompt_stage(row["prompt_stage"])
            label = str(row["label"] or "")
            label_hash = str(row["label_hash"] or "")
            if not label:
                raise ValueError("taxonomy text embedding cache contains a blank label")
            expected_label_hash = taxonomy_label_hash(label)
            if label_hash != expected_label_hash:
                raise ValueError(f"taxonomy text embedding label hash mismatch: {label}")
            identity = (stage, label, label_hash)
            if identity in actual_identities:
                raise ValueError(f"taxonomy text embedding cache has duplicate label: {stage}:{label}")
            actual_identities.add(identity)
            previous_stage = stage_by_label.setdefault(label, stage)
            if previous_stage != stage:
                raise ValueError(f"taxonomy text embedding label is reused across prompt stages: {label}")
            vector = _normalized_float32(row["embedding"], require_unit=True)
            row_dim = int(row["embedding_dim"])
            if row_dim != len(vector):
                raise ValueError(f"taxonomy text embedding dimension mismatch: {label}")
            if embedding_dim is None:
                embedding_dim = row_dim
            elif row_dim != embedding_dim:
                raise ValueError("taxonomy text embedding cache has mixed dimensions")
            offset_by_label[label] = len(vector_values)
            vector_values.extend(vector)
        if actual_identities != expected_identities:
            raise ValueError("taxonomy text embedding cache prompt-stage label set mismatch")
        assert embedding_dim is not None
        return cls(
            classification_version=taxonomy_store.classification_version,
            prompt_version=taxonomy_store.prompt_version,
            hierarchy_fingerprint=taxonomy_store.hierarchy_fingerprint,
            model_id=model_id,
            model_checkpoint=model_checkpoint,
            cache_fingerprint=expected_fingerprint,
            embedding_dim=embedding_dim,
            prompt_stages=tuple(sorted(set(stage_by_label.values()), key=_prompt_stage_order)),
            _offset_by_label=MappingProxyType(offset_by_label),
            _stage_by_label=MappingProxyType(stage_by_label),
            _vectors=vector_values.tobytes(),
        )

    @property
    def label_count(self) -> int:
        return len(self._offset_by_label)

    def prompt_stage_for_label(self, label: str) -> str:
        normalized = str(label or "")
        try:
            return self._stage_by_label[normalized]
        except KeyError as exc:
            raise ValueError(f"taxonomy text embedding cache missing label: {normalized}") from exc

    def raw_similarities(
        self,
        image_embedding: Sequence[float],
        labels: Sequence[str],
    ) -> dict[str, float]:
        requested = _requested_labels(labels)
        missing = [label for label in requested if label not in self._offset_by_label]
        if missing:
            raise ValueError("taxonomy text embedding cache missing labels: " + ", ".join(missing[:5]))
        image = _normalize(image_embedding)
        if len(image) != self.embedding_dim:
            raise ValueError("image and text embedding dimensions do not match")
        vectors = memoryview(self._vectors).cast("f")
        scores: dict[str, float] = {}
        for label in requested:
            start = self._offset_by_label[label]
            text = vectors[start : start + self.embedding_dim]
            scores[label] = sum(
                image_value * float(text_value)
                for image_value, text_value in zip(image, text, strict=True)
            )
        return scores

    def diagnostic_probabilities(
        self,
        image_embedding: Sequence[float],
        labels: Sequence[str],
        *,
        logit_scale: float = 100.0,
    ) -> dict[str, float]:
        scale = float(logit_scale)
        if not isfinite(scale) or scale <= 0:
            raise ValueError("logit_scale must be finite and positive")
        similarities = self.raw_similarities(image_embedding, labels)
        probabilities = _softmax(tuple(scale * score for score in similarities.values()))
        return dict(zip(similarities, probabilities, strict=True))


def build_taxonomy_text_embedding_cache(
    taxonomy_store: PathTaxonomyStore,
    *,
    model_id: str,
    model_checkpoint: str,
    embed_labels: Callable[[list[str]], list[list[float]]],
    batch_size: int = 256,
) -> pl.DataFrame:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not str(model_id or "") or not str(model_checkpoint or ""):
        raise ValueError("taxonomy text embedding model identity must be nonblank")
    prompts = _enabled_unique_prompts(taxonomy_store)
    if prompts.is_empty():
        raise ValueError("classification-v3 has no enabled staged prompt labels")
    rows: list[dict[str, object]] = []
    prompt_rows = prompts.iter_rows(named=True)
    batch: list[dict[str, object]] = []
    for row in prompt_rows:
        batch.append(row)
        if len(batch) == batch_size:
            rows.extend(
                _embed_prompt_batch(
                    batch,
                    taxonomy_store=taxonomy_store,
                    model_id=model_id,
                    model_checkpoint=model_checkpoint,
                    embed_labels=embed_labels,
                )
            )
            batch = []
    if batch:
        rows.extend(
            _embed_prompt_batch(
                batch,
                taxonomy_store=taxonomy_store,
                model_id=model_id,
                model_checkpoint=model_checkpoint,
                embed_labels=embed_labels,
            )
        )
    frame = pl.DataFrame(rows, schema=TAXONOMY_TEXT_EMBEDDING_SCHEMA, orient="row")
    frame = _sort_cache(frame)
    fingerprint = taxonomy_text_embedding_cache_fingerprint(
        frame.drop("embedding_cache_fingerprint")
    )
    return frame.with_columns(pl.lit(fingerprint).alias("embedding_cache_fingerprint"))


def raw_embedding_similarities(
    image_embedding: Sequence[float],
    labels: Sequence[str],
    text_embeddings: Sequence[Sequence[float]],
) -> dict[str, float]:
    requested = _requested_labels(labels)
    if len(requested) != len(text_embeddings):
        raise ValueError("text embedding count does not match label count")
    image = _normalize(image_embedding)
    scores: dict[str, float] = {}
    for label, values in zip(requested, text_embeddings, strict=True):
        text = _normalize(values)
        if len(image) != len(text):
            raise ValueError("image and text embedding dimensions do not match")
        scores[label] = sum(
            image_value * text_value
            for image_value, text_value in zip(image, text, strict=True)
        )
    return scores


def normalize_embedding(values: Sequence[float]) -> tuple[float, ...]:
    return _normalize(values)


def taxonomy_label_hash(label: str) -> str:
    return "sha256:" + hashlib.sha256(str(label).encode("utf-8")).hexdigest()


def taxonomy_text_embedding_cache_fingerprint(frame: pl.DataFrame) -> str:
    expected_columns = [
        name for name in TAXONOMY_TEXT_EMBEDDING_SCHEMA if name != "embedding_cache_fingerprint"
    ]
    if frame.columns != expected_columns:
        raise ValueError("taxonomy text embedding cache fingerprint columns mismatch")
    digest = hashlib.sha256()
    for row in _sort_cache(frame).iter_rows(named=True):
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        )
    return "sha256:" + digest.hexdigest()


def _embed_prompt_batch(
    rows: list[dict[str, object]],
    *,
    taxonomy_store: PathTaxonomyStore,
    model_id: str,
    model_checkpoint: str,
    embed_labels: Callable[[list[str]], list[list[float]]],
) -> list[dict[str, object]]:
    labels = [str(row["label"]) for row in rows]
    vectors = embed_labels(labels)
    if len(vectors) != len(labels):
        raise ValueError(f"text embedder returned {len(vectors)} rows for {len(labels)} labels")
    output: list[dict[str, object]] = []
    for row, label, vector in zip(rows, labels, vectors, strict=True):
        normalized = _normalized_float32(vector)
        output.append(
            {
                "classification_version": taxonomy_store.classification_version,
                "prompt_version": taxonomy_store.prompt_version,
                "prompt_stage": str(row["prompt_stage"]),
                "hierarchy_fingerprint": taxonomy_store.hierarchy_fingerprint,
                "embedding_cache_fingerprint": "",
                "label": label,
                "label_hash": taxonomy_label_hash(label),
                "model_id": model_id,
                "model_checkpoint": model_checkpoint,
                "embedding_dim": len(normalized),
                "embedding": list(normalized),
            }
        )
    return output


def _expected_prompt_identities(
    taxonomy_store: PathTaxonomyStore,
) -> set[tuple[str, str, str]]:
    return {
        (str(row["prompt_stage"]), str(row["label"]), taxonomy_label_hash(str(row["label"])))
        for row in _enabled_unique_prompts(taxonomy_store).iter_rows(named=True)
    }


def _enabled_unique_prompts(taxonomy_store: PathTaxonomyStore) -> pl.DataFrame:
    prompts = taxonomy_store.prompt_labels
    if "prompt_stage" not in prompts.columns:
        raise ValueError("classification-v3 prompt labels are missing prompt_stage")
    enabled = prompts.filter(pl.col("enabled")).select("prompt_stage", "label")
    invalid_stages = sorted(
        set(str(value or "") for value in enabled["prompt_stage"].to_list())
        - set(CLASSIFICATION_PROMPT_STAGES)
    )
    if invalid_stages:
        raise ValueError("classification-v3 has invalid prompt stages: " + ", ".join(invalid_stages))
    duplicates_across_stages = (
        enabled.unique()
        .group_by("label")
        .agg(pl.col("prompt_stage").n_unique().alias("stage_count"))
        .filter(pl.col("stage_count") > 1)
    )
    if not duplicates_across_stages.is_empty():
        raise ValueError("classification-v3 reuses prompt labels across stages")
    return enabled.unique().sort(["prompt_stage", "label"])


def _require_exact_schema(frame: pl.DataFrame) -> pl.DataFrame:
    expected_columns = list(TAXONOMY_TEXT_EMBEDDING_SCHEMA)
    if frame.columns != expected_columns or dict(frame.schema) != TAXONOMY_TEXT_EMBEDDING_SCHEMA:
        raise ValueError("taxonomy text embedding cache physical schema mismatch")
    return _sort_cache(frame)


def _sort_cache(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return frame.sort(["prompt_stage", "label_hash", "label"])


def _require_single(frame: pl.DataFrame, column: str, expected: str) -> None:
    values = {str(value or "") for value in frame[column].to_list()}
    if values != {expected}:
        raise ValueError(f"taxonomy text embedding cache {column} mismatch")


def _single_nonblank(frame: pl.DataFrame, column: str) -> str:
    values = {str(value or "") for value in frame[column].to_list()}
    if len(values) != 1 or not next(iter(values), ""):
        raise ValueError(f"taxonomy text embedding cache has mixed or blank {column}")
    return next(iter(values))


def _requested_labels(labels: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(str(label or "") for label in labels)
    if not requested:
        raise ValueError("at least one taxonomy label is required")
    if any(not label for label in requested):
        raise ValueError("taxonomy labels must be nonblank")
    if len(requested) != len(set(requested)):
        raise ValueError("taxonomy labels must be unique")
    return requested


def _prompt_stage(value: object) -> str:
    stage = str(value or "")
    if stage not in CLASSIFICATION_PROMPT_STAGES:
        raise ValueError(f"invalid taxonomy prompt stage: {stage}")
    return stage


def _prompt_stage_order(stage: str) -> int:
    try:
        return CLASSIFICATION_PROMPT_STAGES.index(stage)
    except ValueError as exc:
        raise ValueError(f"invalid taxonomy prompt stage: {stage}") from exc


def _normalize(values: Sequence[float]) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector or any(not isfinite(value) for value in vector):
        raise ValueError("embedding vector must contain finite values")
    norm = sqrt(sum(value * value for value in vector))
    if not isfinite(norm) or norm <= 0:
        raise ValueError("embedding vector must be non-empty with non-zero norm")
    return tuple(value / norm for value in vector)


def _normalized_float32(
    values: Sequence[float],
    *,
    require_unit: bool = False,
) -> array[float]:
    source = tuple(float(value) for value in values)
    if not source or any(not isfinite(value) for value in source):
        raise ValueError("embedding vector must contain finite values")
    norm = sqrt(sum(value * value for value in source))
    if not isfinite(norm) or norm <= 0:
        raise ValueError("embedding vector must be non-empty with non-zero norm")
    if require_unit and abs(norm - 1.0) > _UNIT_NORM_TOLERANCE:
        raise ValueError("taxonomy text embedding cache contains a non-unit vector")
    normalized = source if require_unit else tuple(value / norm for value in source)
    return array("f", normalized)


def _softmax(values: Sequence[float]) -> tuple[float, ...]:
    maximum = max(values)
    weights = [exp(value - maximum) for value in values]
    total = sum(weights)
    return tuple(weight / total for weight in weights)


__all__ = [
    "TAXONOMY_TEXT_EMBEDDING_SCHEMA",
    "TaxonomyTextEmbeddingIndex",
    "build_taxonomy_text_embedding_cache",
    "normalize_embedding",
    "raw_embedding_similarities",
    "taxonomy_label_hash",
    "taxonomy_text_embedding_cache_fingerprint",
]
