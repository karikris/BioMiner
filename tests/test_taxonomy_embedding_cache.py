from __future__ import annotations

from dataclasses import dataclass, replace
from math import isclose

import polars as pl
import pytest

from biominer.bioclip.taxonomy_embedding_cache import (
    TAXONOMY_TEXT_EMBEDDING_SCHEMA,
    TaxonomyTextEmbeddingIndex,
    build_taxonomy_text_embedding_cache,
    taxonomy_text_embedding_cache_fingerprint,
)
from biominer.registry.classification_v3 import (
    CLASSIFICATION_V3_PROMPT_VERSION,
    CLASSIFICATION_V3_VERSION,
    PROMPT_LABEL_SCHEMA,
)


MODEL_ID = "fake-bioclip"
MODEL_CHECKPOINT = "fake-checkpoint"


@dataclass(frozen=True)
class _Store:
    prompt_labels: pl.DataFrame
    classification_version: str = CLASSIFICATION_V3_VERSION
    prompt_version: str = CLASSIFICATION_V3_PROMPT_VERSION
    hierarchy_fingerprint: str = "sha256:hierarchy"


def test_raw_cosines_are_not_candidate_relative_and_probabilities_are_diagnostic() -> None:
    store = _store()
    vectors = {
        "family positive": [1.0, 0.0],
        "family orthogonal": [0.0, 1.0],
        "species first": [-1.0, 0.0],
        "species rerank": [0.6, 0.8],
    }
    cache = build_taxonomy_text_embedding_cache(
        store,
        model_id=MODEL_ID,
        model_checkpoint=MODEL_CHECKPOINT,
        embed_labels=lambda labels: [vectors[label] for label in labels],
    )
    index = _index(cache, store)

    pair = index.raw_similarities([1.0, 0.0], ["family positive", "family orthogonal"])
    expanded = index.raw_similarities(
        [1.0, 0.0],
        ["family positive", "family orthogonal", "species first"],
    )

    assert pair == {"family positive": 1.0, "family orthogonal": 0.0}
    assert expanded["family positive"] == pair["family positive"]
    assert expanded["family orthogonal"] == pair["family orthogonal"]
    assert expanded["species first"] == -1.0
    assert sum(expanded.values()) == 0.0
    probabilities = index.diagnostic_probabilities(
        [1.0, 0.0],
        ["family positive", "family orthogonal", "species first"],
    )
    assert isclose(sum(probabilities.values()), 1.0)


def test_cache_embeds_duplicate_same_stage_label_once() -> None:
    prompts = _prompt_rows()
    duplicate = prompts.filter(pl.col("label") == "family positive").with_columns(
        pl.lit("family:duplicate").alias("node_id"),
        pl.lit("Duplicatidae").alias("scientific_name"),
    )
    store = _Store(pl.concat([prompts, duplicate]))
    calls: list[tuple[str, ...]] = []

    def embed(labels: list[str]) -> list[list[float]]:
        calls.append(tuple(labels))
        return [[1.0, float(index + 1)] for index, _label in enumerate(labels)]

    cache = build_taxonomy_text_embedding_cache(
        store,
        model_id=MODEL_ID,
        model_checkpoint=MODEL_CHECKPOINT,
        embed_labels=embed,
        batch_size=20,
    )

    embedded_labels = [label for call in calls for label in call]
    assert embedded_labels.count("family positive") == 1
    assert cache.height == prompts["label"].n_unique()


def test_cache_fingerprint_mismatch_fails_closed() -> None:
    store, cache = _cache()
    tampered = cache.with_columns(
        pl.lit("sha256:wrong").alias("embedding_cache_fingerprint")
    )

    with pytest.raises(ValueError, match="cache fingerprint mismatch"):
        _index(tampered, store)


def test_prompt_stage_mismatch_fails_even_with_recomputed_cache_fingerprint() -> None:
    store, cache = _cache()
    label = "family positive"
    tampered = cache.with_columns(
        pl.when(pl.col("label") == label)
        .then(pl.lit("species_first_pass"))
        .otherwise(pl.col("prompt_stage"))
        .alias("prompt_stage")
    )
    tampered = _refingerprint(tampered)

    with pytest.raises(ValueError, match="prompt-stage label set mismatch"):
        _index(tampered, store)


def test_taxonomy_fingerprint_mismatch_fails_closed() -> None:
    store, cache = _cache()
    other_store = replace(store, hierarchy_fingerprint="sha256:other-hierarchy")

    with pytest.raises(ValueError, match="hierarchy_fingerprint mismatch"):
        _index(cache, other_store)


@pytest.mark.parametrize(
    ("model_id", "checkpoint", "message"),
    [
        ("other-model", MODEL_CHECKPOINT, "model_id mismatch"),
        (MODEL_ID, "other-checkpoint", "model_checkpoint mismatch"),
    ],
)
def test_model_identity_mismatch_fails_closed(
    model_id: str,
    checkpoint: str,
    message: str,
) -> None:
    store, cache = _cache()

    with pytest.raises(ValueError, match=message):
        TaxonomyTextEmbeddingIndex.from_frame(
            cache,
            taxonomy_store=store,
            model_id=model_id,
            model_checkpoint=checkpoint,
        )


def test_missing_label_and_wrong_image_dimension_fail_clearly() -> None:
    store, cache = _cache()
    index = _index(cache, store)

    with pytest.raises(ValueError, match="missing labels: absent"):
        index.raw_similarities([1.0, 0.0], ["absent"])
    with pytest.raises(ValueError, match="dimensions do not match"):
        index.raw_similarities([1.0, 0.0, 0.0], ["family positive"])


def test_label_hash_and_nonfinite_vectors_fail_closed() -> None:
    store, cache = _cache()
    bad_hash = cache.with_columns(
        pl.when(pl.col("label") == "family positive")
        .then(pl.lit("sha256:wrong"))
        .otherwise(pl.col("label_hash"))
        .alias("label_hash")
    )
    bad_hash = _refingerprint(bad_hash)
    with pytest.raises(ValueError, match="label hash mismatch"):
        _index(bad_hash, store)

    with pytest.raises(ValueError, match="finite values"):
        build_taxonomy_text_embedding_cache(
            store,
            model_id=MODEL_ID,
            model_checkpoint=MODEL_CHECKPOINT,
            embed_labels=lambda labels: [[float("nan"), 1.0] for _label in labels],
        )


def test_cache_requires_exact_physical_schema() -> None:
    store, cache = _cache()
    assert cache.schema == TAXONOMY_TEXT_EMBEDDING_SCHEMA

    with pytest.raises(ValueError, match="physical schema mismatch"):
        _index(cache.with_columns(pl.col("embedding_dim").cast(pl.Int32)), store)


def _cache() -> tuple[_Store, pl.DataFrame]:
    store = _store()
    vectors = {
        "family positive": [1.0, 0.0],
        "family orthogonal": [0.0, 1.0],
        "species first": [-1.0, 0.0],
        "species rerank": [0.6, 0.8],
    }
    return store, build_taxonomy_text_embedding_cache(
        store,
        model_id=MODEL_ID,
        model_checkpoint=MODEL_CHECKPOINT,
        embed_labels=lambda labels: [vectors[label] for label in labels],
        batch_size=2,
    )


def _index(cache: pl.DataFrame, store: _Store) -> TaxonomyTextEmbeddingIndex:
    return TaxonomyTextEmbeddingIndex.from_frame(
        cache,
        taxonomy_store=store,
        model_id=MODEL_ID,
        model_checkpoint=MODEL_CHECKPOINT,
    )


def _store() -> _Store:
    return _Store(_prompt_rows())


def _prompt_rows() -> pl.DataFrame:
    rows = [
        _prompt("family:one", "FAMILY", "Oneidae", "rank_screen", "family positive", 1),
        _prompt("family:one", "FAMILY", "Oneidae", "rank_screen", "family orthogonal", 2),
        _prompt("species:one", "SPECIES", "Onea one", "species_first_pass", "species first", 1),
        _prompt("species:one", "SPECIES", "Onea one", "species_rerank", "species rerank", 1),
    ]
    return pl.DataFrame(rows, schema=PROMPT_LABEL_SCHEMA)


def _prompt(
    node_id: str,
    rank: str,
    name: str,
    stage: str,
    label: str,
    sort_order: int,
) -> dict[str, object]:
    return {
        "classification_version": CLASSIFICATION_V3_VERSION,
        "prompt_version": CLASSIFICATION_V3_PROMPT_VERSION,
        "prompt_stage": stage,
        "node_id": node_id,
        "rank": rank,
        "scientific_name": name,
        "label": label,
        "prompt_template": label,
        "sort_order": sort_order,
        "enabled": True,
    }


def _refingerprint(cache: pl.DataFrame) -> pl.DataFrame:
    fingerprint = taxonomy_text_embedding_cache_fingerprint(
        cache.drop("embedding_cache_fingerprint")
    )
    return cache.with_columns(pl.lit(fingerprint).alias("embedding_cache_fingerprint"))
