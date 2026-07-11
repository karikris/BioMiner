from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import exp
import os
from typing import Any, Iterable, Mapping, Protocol, Sequence

import polars as pl

try:  # NumPy is available with the local BioMiner environment via the ML stack.
    import numpy as _np
except Exception:  # pragma: no cover - fallback keeps the package usable without NumPy.
    _np = None

from biominer.bioclip.classification_modes import (
    DEFAULT_FAMILY_TOP_K,
    DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    DEFAULT_SPECIES_RERANK_TOP_K,
    HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
)
from biominer.bioclip.embedding_cache import validate_taxonomy_text_embedding_cache
from biominer.registry.classification_table import (
    CLASSIFICATION_TABLE_VERSION,
    PROMPT_VARIANT_VERSION,
    ButterflyTaxonomyStore,
)


HIERARCHICAL_CANDIDATE_SELECTION_MODE = "gbif_family_first"
HIERARCHICAL_SPECIES_RERANK_STRATEGY = "rerank_all_first_pass_top20"
DEFAULT_FAMILY_INDEX_CACHE_ENTRIES = 16
FAMILY_INDEX_CACHE_ENTRIES_ENV = "BIOMINER_FAMILY_INDEX_CACHE_ENTRIES"
_CACHE_INDEX_BY_FRAME_AND_FAMILY: OrderedDict[tuple[int, str, str, str], "_CachedFamilySpeciesIndex"] = OrderedDict()

TAXON_SCORE_SCHEMA: dict[str, pl.DataType] = {
    "accepted_taxon_key": pl.String,
    "scientific_name": pl.String,
    "rank": pl.String,
    "family_key": pl.String,
    "family": pl.String,
    "genus_key": pl.String,
    "genus": pl.String,
    "score": pl.Float64,
    "best_label": pl.String,
    "label_count": pl.Int64,
}
TAXON_SCORE_DTYPE = pl.Struct(TAXON_SCORE_SCHEMA)

BUTTERFLY_CASCADE_RESULT_SCHEMA: dict[str, pl.DataType] = {
    "source": pl.String,
    "flickr_photo_id": pl.String,
    "detection_id": pl.String,
    "crop_hash": pl.String,
    "classification_mode": pl.String,
    "candidate_set_id": pl.String,
    "taxonomy_table_version": pl.String,
    "prompt_variant_version": pl.String,
    "family_top3": pl.List(TAXON_SCORE_DTYPE),
    "selected_family_key": pl.String,
    "selected_family": pl.String,
    "species_candidate_count": pl.Int64,
    "species_top20": pl.List(TAXON_SCORE_DTYPE),
    "species_top5": pl.List(TAXON_SCORE_DTYPE),
    "species_top1": TAXON_SCORE_DTYPE,
    "family_top1_score": pl.Float64,
    "species_top1_score": pl.Float64,
    "species_top1_margin": pl.Float64,
    "classified_at": pl.String,
}

HIERARCHICAL_OBJECT_SCORE_SCHEMA_EXTENSIONS: dict[str, pl.DataType] = {
    "taxonomy_table_version": pl.String,
    "taxonomy_prompt_variant_version": pl.String,
    "selected_family_key": pl.String,
    "selected_family": pl.String,
    "family_top3_accepted_taxon_keys": pl.List(pl.String),
    "family_top3_scores": pl.List(pl.Float64),
    "species_candidate_family_key": pl.String,
    "species_candidate_family": pl.String,
    "species_candidate_count": pl.Int64,
    "species_top20_scores": pl.List(pl.Float64),
    "species_top5_scores": pl.List(pl.Float64),
}


class ObjectBioClipScorer(Protocol):
    model_id: str
    model_version: str
    model_checkpoint: str

    def score(self, item: dict[str, Any], labels: tuple[str, ...]) -> dict[str, float]:
        ...


@dataclass(frozen=True)
class TaxonScore:
    accepted_taxon_key: str
    scientific_name: str
    rank: str
    family_key: str | None
    family: str | None
    genus_key: str | None
    genus: str | None
    score: float
    best_label: str
    label_count: int


@dataclass(frozen=True)
class ButterflyCascadeResult:
    source: str
    flickr_photo_id: str
    detection_id: str
    crop_hash: str
    classification_mode: str
    candidate_set_id: str
    taxonomy_table_version: str
    prompt_variant_version: str
    family_top3: tuple[TaxonScore, ...]
    selected_family_key: str | None
    selected_family: str | None
    species_candidate_count: int
    species_top20: tuple[TaxonScore, ...]
    species_top5: tuple[TaxonScore, ...]
    species_top1: TaxonScore | None
    family_top1_score: float
    species_top1_score: float
    species_top1_margin: float | None
    classified_at: str


def taxon_score_to_dict(score: TaxonScore | None) -> dict[str, Any] | None:
    if score is None:
        return None
    row = asdict(score)
    row["score"] = float(score.score)
    row["label_count"] = int(score.label_count)
    return row


def butterfly_cascade_result_to_dict(result: ButterflyCascadeResult) -> dict[str, Any]:
    return {
        "source": result.source,
        "flickr_photo_id": result.flickr_photo_id,
        "detection_id": result.detection_id,
        "crop_hash": result.crop_hash,
        "classification_mode": result.classification_mode or HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        "candidate_set_id": result.candidate_set_id,
        "taxonomy_table_version": result.taxonomy_table_version,
        "prompt_variant_version": result.prompt_variant_version,
        "family_top3": [taxon_score_to_dict(score) for score in result.family_top3],
        "selected_family_key": result.selected_family_key,
        "selected_family": result.selected_family,
        "species_candidate_count": int(result.species_candidate_count),
        "species_top20": [taxon_score_to_dict(score) for score in result.species_top20],
        "species_top5": [taxon_score_to_dict(score) for score in result.species_top5],
        "species_top1": taxon_score_to_dict(result.species_top1),
        "family_top1_score": float(result.family_top1_score),
        "species_top1_score": float(result.species_top1_score),
        "species_top1_margin": result.species_top1_margin,
        "classified_at": result.classified_at,
    }


def butterfly_cascade_results_frame(results: list[ButterflyCascadeResult]) -> pl.DataFrame:
    if not results:
        return pl.DataFrame(schema=BUTTERFLY_CASCADE_RESULT_SCHEMA)
    return pl.DataFrame(
        [butterfly_cascade_result_to_dict(result) for result in results],
        schema=BUTTERFLY_CASCADE_RESULT_SCHEMA,
    )


def aggregate_taxon_prompt_scores(
    *,
    label_scores: Mapping[str, float],
    label_rows: pl.DataFrame,
    taxon_key_column: str,
    taxon_name_column: str,
    aggregation: str = "mean",
) -> list[TaxonScore]:
    if aggregation not in {"max", "mean", "softmax_mean"}:
        raise ValueError("aggregation must be one of: max, mean, softmax_mean")
    _require_label_columns(label_rows, taxon_key_column=taxon_key_column, taxon_name_column=taxon_name_column)
    if label_rows.is_empty():
        return []

    grouped: dict[str, dict[str, Any]] = {}
    for row in label_rows.to_dicts():
        if "enabled" in row and row.get("enabled") is False:
            continue
        label = _clean_text(row.get("label"))
        if not label:
            continue
        taxon_key = _clean_text(row.get(taxon_key_column))
        scientific_name = _clean_text(row.get(taxon_name_column))
        if not taxon_key or not scientific_name:
            continue
        group = grouped.setdefault(
            taxon_key,
            {
                "accepted_taxon_key": taxon_key,
                "scientific_name": scientific_name,
                "rank": _rank_for_label_row(row, taxon_key_column=taxon_key_column),
                "family_key": _taxon_family_key(row, taxon_key=taxon_key, taxon_key_column=taxon_key_column),
                "family": _taxon_family(row, scientific_name=scientific_name, taxon_key_column=taxon_key_column),
                "genus_key": _clean_text(row.get("genus_key")) or None,
                "genus": _clean_text(row.get("genus")) or None,
                "label_scores": {},
            },
        )
        group["label_scores"].setdefault(label, float(label_scores.get(label, 0.0)))

    scores: list[TaxonScore] = []
    for group in grouped.values():
        values_by_label = dict(group["label_scores"])
        if not values_by_label:
            continue
        values = list(values_by_label.values())
        taxon_score = _aggregate_values(values, aggregation=aggregation)
        best_label, _best_score = sorted(values_by_label.items(), key=lambda item: (-item[1], item[0]))[0]
        scores.append(
            TaxonScore(
                accepted_taxon_key=group["accepted_taxon_key"],
                scientific_name=group["scientific_name"],
                rank=group["rank"],
                family_key=group["family_key"],
                family=group["family"],
                genus_key=group["genus_key"],
                genus=group["genus"],
                score=taxon_score,
                best_label=best_label,
                label_count=len(values_by_label),
            )
        )
    return sorted(scores, key=lambda score: (-score.score, score.scientific_name, score.accepted_taxon_key))


def classify_butterfly_crop_hierarchical(
    *,
    item: dict[str, Any],
    scorer: ObjectBioClipScorer,
    taxonomy_store: ButterflyTaxonomyStore,
    family_top_k: int = DEFAULT_FAMILY_TOP_K,
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K,
    prompt_aggregation: str = "mean",
    taxonomy_text_embedding_cache: pl.DataFrame | None = None,
) -> ButterflyCascadeResult:
    family_top_k, species_first_pass_top_k, species_rerank_top_k = _validate_top_k(
        family_top_k=family_top_k,
        species_first_pass_top_k=species_first_pass_top_k,
        species_rerank_top_k=species_rerank_top_k,
    )
    _raise_for_invalid_taxonomy_store(taxonomy_store)

    family_label_rows = taxonomy_store.family_prompt_label_rows()
    family_labels = _label_tuple(family_label_rows)
    if not family_labels:
        raise ValueError("butterfly taxonomy store has no enabled family labels")
    family_label_scores = scorer.score(item, family_labels)
    family_scores = aggregate_taxon_prompt_scores(
        label_scores=family_label_scores,
        label_rows=family_label_rows,
        taxon_key_column="family_key",
        taxon_name_column="family",
        aggregation=prompt_aggregation,
    )
    if not family_scores:
        raise ValueError("butterfly taxonomy store produced no family scores")
    family_top = tuple(family_scores[:family_top_k])
    selected_family = family_top[0]
    selected_family_key = selected_family.accepted_taxon_key

    if taxonomy_text_embedding_cache is not None:
        text_embedding_cache = _validated_taxonomy_text_embedding_cache_for_scorer(
            taxonomy_text_embedding_cache,
            taxonomy_store=taxonomy_store,
            scorer=scorer,
        )
        image_embedding = _embed_image_items_for_cached_taxonomy(scorer, [item])[0]
        species_taxa = taxonomy_store.species_for_family(selected_family_key)
        species_top20 = tuple(
            rank_species_with_cached_text_embeddings(
                image_embedding=image_embedding,
                taxonomy_store=taxonomy_store,
                family_key=selected_family_key,
                text_embedding_cache=text_embedding_cache,
                top_k=species_first_pass_top_k,
            )
        )
    else:
        species_taxa = taxonomy_store.species_for_family(selected_family_key)
        species_label_rows = taxonomy_store.species_label_rows_for_family(selected_family_key)
        species_labels = _label_tuple(species_label_rows)
        if species_taxa.is_empty() or not species_labels:
            raise ValueError(f"butterfly taxonomy store has no enabled species labels for family_key={selected_family_key!r}")
        species_label_scores = scorer.score(item, species_labels)
        species_scores = aggregate_taxon_prompt_scores(
            label_scores=species_label_scores,
            label_rows=species_label_rows,
            taxon_key_column="accepted_taxon_key",
            taxon_name_column="scientific_name",
            aggregation=prompt_aggregation,
        )
        species_top20 = tuple(species_scores[:species_first_pass_top_k])
    _assert_species_top20_family(species_top20, selected_family_key=selected_family_key)

    rerank_keys = [score.accepted_taxon_key for score in species_top20]
    rerank_label_rows = taxonomy_store.species_labels_for_taxa(rerank_keys)
    rerank_labels = _label_tuple(rerank_label_rows)
    rerank_scores = scorer.score(item, rerank_labels) if rerank_labels else {}
    reranked_species = aggregate_taxon_prompt_scores(
        label_scores=rerank_scores,
        label_rows=rerank_label_rows,
        taxon_key_column="accepted_taxon_key",
        taxon_name_column="scientific_name",
        aggregation=prompt_aggregation,
    )
    species_top5 = tuple(reranked_species[:species_rerank_top_k])
    species_top1 = species_top5[0] if species_top5 else None

    return _cascade_result(
        item=item,
        taxonomy_store=taxonomy_store,
        family_top=family_top,
        selected_family_key=selected_family_key,
        species_candidate_count=species_taxa.height,
        species_top20=species_top20,
        species_top5=species_top5,
        species_top1=species_top1,
    )


def classify_butterfly_crops_hierarchical_batch(
    *,
    items: Sequence[dict[str, Any]],
    scorer: ObjectBioClipScorer,
    taxonomy_store: ButterflyTaxonomyStore,
    family_top_k: int = DEFAULT_FAMILY_TOP_K,
    species_first_pass_top_k: int = DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    species_rerank_top_k: int = DEFAULT_SPECIES_RERANK_TOP_K,
    prompt_aggregation: str = "mean",
    taxonomy_text_embedding_cache: pl.DataFrame | None = None,
) -> list[ButterflyCascadeResult]:
    batch_items = list(items)
    if not batch_items:
        return []
    family_top_k, species_first_pass_top_k, species_rerank_top_k = _validate_top_k(
        family_top_k=family_top_k,
        species_first_pass_top_k=species_first_pass_top_k,
        species_rerank_top_k=species_rerank_top_k,
    )
    _raise_for_invalid_taxonomy_store(taxonomy_store)

    family_label_rows = taxonomy_store.family_prompt_label_rows()
    family_labels = _label_tuple(family_label_rows)
    if not family_labels:
        raise ValueError("butterfly taxonomy store has no enabled family labels")
    family_scores_by_item = _score_label_sets_for_items(
        scorer,
        batch_items,
        {"family": family_labels},
    )["family"]
    image_embeddings: list[list[float]] | None = None
    text_embedding_cache: pl.DataFrame | None = None
    cache_model_id: str | None = None
    cache_model_checkpoint: str | None = None
    if taxonomy_text_embedding_cache is not None:
        text_embedding_cache = _validated_taxonomy_text_embedding_cache_for_scorer(
            taxonomy_text_embedding_cache,
            taxonomy_store=taxonomy_store,
            scorer=scorer,
        )
        cache_model_id, cache_model_checkpoint = _single_cache_model_pair(text_embedding_cache)
        image_embeddings = _embed_image_items_for_cached_taxonomy(scorer, batch_items)

    state: dict[int, dict[str, Any]] = {}
    family_groups: dict[str, list[int]] = {}
    for index, label_scores in enumerate(family_scores_by_item):
        family_scores = aggregate_taxon_prompt_scores(
            label_scores=label_scores,
            label_rows=family_label_rows,
            taxon_key_column="family_key",
            taxon_name_column="family",
            aggregation=prompt_aggregation,
        )
        if not family_scores:
            raise ValueError("butterfly taxonomy store produced no family scores")
        family_top = tuple(family_scores[:family_top_k])
        selected_family_key = family_top[0].accepted_taxon_key
        state[index] = {
            "family_top": family_top,
            "selected_family_key": selected_family_key,
        }
        family_groups.setdefault(selected_family_key, []).append(index)

    for family_key, indices in family_groups.items():
        species_taxa = taxonomy_store.species_for_family(family_key)
        if image_embeddings is not None:
            if text_embedding_cache is None or cache_model_id is None or cache_model_checkpoint is None:
                raise AssertionError("taxonomy text embedding cache was not validated before cached species ranking")
            for index in indices:
                species_top20 = tuple(
                    _rank_species_with_validated_cached_text_embeddings(
                        image_embedding=image_embeddings[index],
                        taxonomy_store=taxonomy_store,
                        family_key=family_key,
                        text_embedding_cache=text_embedding_cache,
                        model_id=cache_model_id,
                        model_checkpoint=cache_model_checkpoint,
                        top_k=species_first_pass_top_k,
                    )
                )
                _assert_species_top20_family(species_top20, selected_family_key=family_key)
                state[index]["species_candidate_count"] = species_taxa.height
                state[index]["species_top20"] = species_top20
            continue

        species_label_rows = taxonomy_store.species_label_rows_for_family(family_key)
        species_labels = _label_tuple(species_label_rows)
        if species_taxa.is_empty() or not species_labels:
            raise ValueError(f"butterfly taxonomy store has no enabled species labels for family_key={family_key!r}")
        score_key = f"species:{family_key}"
        species_scores = _score_label_sets_for_items(
            scorer,
            [batch_items[index] for index in indices],
            {score_key: species_labels},
        )[score_key]
        for index, label_scores in zip(indices, species_scores, strict=True):
            species_ranked = aggregate_taxon_prompt_scores(
                label_scores=label_scores,
                label_rows=species_label_rows,
                taxon_key_column="accepted_taxon_key",
                taxon_name_column="scientific_name",
                aggregation=prompt_aggregation,
            )
            species_top20 = tuple(species_ranked[:species_first_pass_top_k])
            _assert_species_top20_family(species_top20, selected_family_key=family_key)
            state[index]["species_candidate_count"] = species_taxa.height
            state[index]["species_top20"] = species_top20

    rerank_groups: dict[tuple[str, ...], list[int]] = {}
    for index in range(len(batch_items)):
        top20_keys = tuple(score.accepted_taxon_key for score in state[index]["species_top20"])
        rerank_groups.setdefault(top20_keys, []).append(index)

    for top20_keys, indices in rerank_groups.items():
        rerank_label_rows = taxonomy_store.species_labels_for_taxa(top20_keys)
        rerank_labels = _label_tuple(rerank_label_rows)
        rerank_scores = (
            _score_label_sets_for_items(
                scorer,
                [batch_items[index] for index in indices],
                {"rerank": rerank_labels},
            )["rerank"]
            if rerank_labels
            else [{} for _index in indices]
        )
        for index, label_scores in zip(indices, rerank_scores, strict=True):
            reranked_species = aggregate_taxon_prompt_scores(
                label_scores=label_scores,
                label_rows=rerank_label_rows,
                taxon_key_column="accepted_taxon_key",
                taxon_name_column="scientific_name",
                aggregation=prompt_aggregation,
            )
            species_top5 = tuple(reranked_species[:species_rerank_top_k])
            state[index]["species_top5"] = species_top5
            state[index]["species_top1"] = species_top5[0] if species_top5 else None

    return [
        _cascade_result(
            item=batch_items[index],
            taxonomy_store=taxonomy_store,
            family_top=state[index]["family_top"],
            selected_family_key=state[index]["selected_family_key"],
            species_candidate_count=int(state[index]["species_candidate_count"]),
            species_top20=state[index]["species_top20"],
            species_top5=state[index]["species_top5"],
            species_top1=state[index]["species_top1"],
        )
        for index in range(len(batch_items))
    ]


def rank_species_with_cached_text_embeddings(
    *,
    image_embedding: Sequence[float],
    taxonomy_store: ButterflyTaxonomyStore,
    family_key: str,
    text_embedding_cache: pl.DataFrame,
    top_k: int,
) -> list[TaxonScore]:
    limit = int(top_k)
    if limit <= 0:
        raise ValueError("top_k must be positive")
    _raise_for_invalid_taxonomy_store(taxonomy_store)
    selected_family_key = _clean_text(family_key)
    if not selected_family_key:
        raise ValueError("family_key is required")
    species_taxa = taxonomy_store.species_for_family(selected_family_key)
    model_id, model_checkpoint = _single_cache_model_pair(text_embedding_cache)
    validate_taxonomy_text_embedding_cache(
        text_embedding_cache,
        taxonomy_store,
        model_id=model_id,
        model_checkpoint=model_checkpoint,
    )
    return _rank_species_with_validated_cached_text_embeddings(
        image_embedding=image_embedding,
        taxonomy_store=taxonomy_store,
        family_key=family_key,
        text_embedding_cache=text_embedding_cache,
        model_id=model_id,
        model_checkpoint=model_checkpoint,
        top_k=limit,
    )


def _rank_species_with_validated_cached_text_embeddings(
    *,
    image_embedding: Sequence[float],
    taxonomy_store: ButterflyTaxonomyStore,
    family_key: str,
    text_embedding_cache: pl.DataFrame,
    model_id: str,
    model_checkpoint: str,
    top_k: int,
) -> list[TaxonScore]:
    limit = int(top_k)
    if limit <= 0:
        raise ValueError("top_k must be positive")
    selected_family_key = _clean_text(family_key)
    if not selected_family_key:
        raise ValueError("family_key is required")
    species_taxa = taxonomy_store.species_for_family(selected_family_key)
    image_vector = _float_vector(image_embedding)
    index = _cached_family_species_index(
        species_taxa=species_taxa,
        family_key=selected_family_key,
        text_embedding_cache=text_embedding_cache,
        model_id=model_id,
        model_checkpoint=model_checkpoint,
    )
    if index.matrix is not None and _np is not None:
        image = _np.asarray(image_vector, dtype=_np.float32)
        norm = float(_np.linalg.norm(image))
        if norm == 0.0:
            label_scores = _np.zeros(len(index.labels), dtype=_np.float32)
        else:
            similarities = index.matrix @ (image / norm)
            logits = 100.0 * similarities
            shifted = logits - _np.max(logits)
            label_scores = _np.exp(shifted) / _np.sum(_np.exp(shifted))
        scores = [
            _taxon_score_from_label_scores(
                metadata=index.taxon_metadata_by_key[taxon_key],
                labels=index.labels,
                label_scores=label_scores,
                label_indices=indices,
            )
            for taxon_key, indices in index.label_indices_by_taxon_key.items()
        ]
    else:
        label_scores = _softmax(
            [100.0 * _cosine_similarity(image_vector, embedding_vector) for embedding_vector in index.embedding_vectors]
        )
        scores = [
            _taxon_score_from_values(
                metadata=index.taxon_metadata_by_key[taxon_key],
                values_by_label={
                    index.labels[index_index]: label_scores[index_index]
                    for index_index in indices
                },
            )
            for taxon_key, indices in index.label_indices_by_taxon_key.items()
        ]
    return sorted(scores, key=lambda score: (-score.score, score.scientific_name, score.accepted_taxon_key))[:limit]


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    weights = [exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights] if total else [0.0 for _value in values]


@dataclass(frozen=True)
class _CachedTaxonMetadata:
    accepted_taxon_key: str
    scientific_name: str
    rank: str
    family_key: str | None
    family: str | None
    genus_key: str | None
    genus: str | None


@dataclass(frozen=True)
class _CachedFamilySpeciesIndex:
    labels: tuple[str, ...]
    embedding_vectors: tuple[tuple[float, ...], ...]
    matrix: Any | None
    label_indices_by_taxon_key: dict[str, tuple[int, ...]]
    taxon_metadata_by_key: dict[str, _CachedTaxonMetadata]


def _cached_family_species_index(
    *,
    species_taxa: pl.DataFrame,
    family_key: str,
    text_embedding_cache: pl.DataFrame,
    model_id: str,
    model_checkpoint: str,
) -> _CachedFamilySpeciesIndex:
    cache_key = (id(text_embedding_cache), family_key, model_id, model_checkpoint)
    cached = _CACHE_INDEX_BY_FRAME_AND_FAMILY.pop(cache_key, None)
    if cached is not None:
        _CACHE_INDEX_BY_FRAME_AND_FAMILY[cache_key] = cached
        return cached
    taxa_by_key = {str(row["accepted_taxon_key"]): row for row in species_taxa.to_dicts()}
    species_rows = (
        text_embedding_cache.filter(
            (pl.col("model_id") == model_id)
            & (pl.col("model_checkpoint") == model_checkpoint)
            & (pl.col("label_scope") == "species")
            & (pl.col("family_key") == family_key)
            & pl.col("accepted_taxon_key").is_in(list(taxa_by_key))
        )
        .sort(["accepted_taxon_key", "label"])
        .to_dicts()
    )
    labels: list[str] = []
    embedding_vectors: list[tuple[float, ...]] = []
    label_indices_by_taxon_key: dict[str, list[int]] = {}
    taxon_metadata_by_key: dict[str, _CachedTaxonMetadata] = {}
    for row in species_rows:
        taxon_key = _clean_text(row.get("accepted_taxon_key"))
        label = _clean_text(row.get("label"))
        if not taxon_key or not label or taxon_key not in taxa_by_key:
            continue
        taxon = taxa_by_key[taxon_key]
        taxon_metadata_by_key.setdefault(
            taxon_key,
            _CachedTaxonMetadata(
                accepted_taxon_key=taxon_key,
                scientific_name=_clean_text(taxon.get("scientific_name")),
                rank=_clean_text(taxon.get("rank")) or "SPECIES",
                family_key=_clean_text(taxon.get("family_key")) or None,
                family=_clean_text(taxon.get("family")) or None,
                genus_key=_clean_text(taxon.get("genus_key")) or None,
                genus=_clean_text(taxon.get("genus")) or None,
            ),
        )
        label_indices_by_taxon_key.setdefault(taxon_key, []).append(len(labels))
        labels.append(label)
        embedding_vectors.append(tuple(_float_vector(row.get("embedding"))))
    matrix = _embedding_matrix(embedding_vectors)
    index = _CachedFamilySpeciesIndex(
        labels=tuple(labels),
        embedding_vectors=tuple(embedding_vectors),
        matrix=matrix,
        label_indices_by_taxon_key={key: tuple(value) for key, value in label_indices_by_taxon_key.items()},
        taxon_metadata_by_key=taxon_metadata_by_key,
    )
    _CACHE_INDEX_BY_FRAME_AND_FAMILY[cache_key] = index
    while len(_CACHE_INDEX_BY_FRAME_AND_FAMILY) > family_index_cache_entries():
        _CACHE_INDEX_BY_FRAME_AND_FAMILY.popitem(last=False)
    return index


def _embedding_matrix(embedding_vectors: Sequence[Sequence[float]]) -> Any | None:
    if _np is None or not embedding_vectors:
        return None
    matrix = _np.asarray(embedding_vectors, dtype=_np.float32)
    norms = _np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / _np.where(norms == 0.0, 1.0, norms)


def family_index_cache_entries() -> int:
    raw_value = os.environ.get(FAMILY_INDEX_CACHE_ENTRIES_ENV)
    if raw_value is None:
        return DEFAULT_FAMILY_INDEX_CACHE_ENTRIES
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{FAMILY_INDEX_CACHE_ENTRIES_ENV} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{FAMILY_INDEX_CACHE_ENTRIES_ENV} must be a positive integer")
    return parsed


def clear_family_embedding_index_cache() -> None:
    _CACHE_INDEX_BY_FRAME_AND_FAMILY.clear()


def _taxon_score_from_label_scores(
    *,
    metadata: _CachedTaxonMetadata,
    labels: Sequence[str],
    label_scores: Sequence[float],
    label_indices: Sequence[int],
) -> TaxonScore:
    values_by_label = {labels[index]: float(label_scores[index]) for index in label_indices}
    return _taxon_score_from_values(metadata=metadata, values_by_label=values_by_label)


def _taxon_score_from_values(
    *,
    metadata: _CachedTaxonMetadata,
    values_by_label: Mapping[str, float],
) -> TaxonScore:
    values = list(values_by_label.values())
    best_label, _best_score = sorted(values_by_label.items(), key=lambda item: (-item[1], item[0]))[0]
    return TaxonScore(
        accepted_taxon_key=metadata.accepted_taxon_key,
        scientific_name=metadata.scientific_name,
        rank=metadata.rank,
        family_key=metadata.family_key,
        family=metadata.family,
        genus_key=metadata.genus_key,
        genus=metadata.genus,
        score=sum(values) / len(values),
        best_label=best_label,
        label_count=len(values_by_label),
    )


def _cached_species_prompt_similarity_groups(
    *,
    image_vector: list[float],
    species_taxa: pl.DataFrame,
    family_key: str,
    text_embedding_cache: pl.DataFrame,
    model_id: str,
    model_checkpoint: str,
) -> dict[str, dict[str, Any]]:
    index = _cached_family_species_index(
        species_taxa=species_taxa,
        family_key=family_key,
        text_embedding_cache=text_embedding_cache,
        model_id=model_id,
        model_checkpoint=model_checkpoint,
    )
    groups: dict[str, dict[str, Any]] = {}
    for taxon_key, indices in index.label_indices_by_taxon_key.items():
        metadata = index.taxon_metadata_by_key[taxon_key]
        group = groups.setdefault(
            taxon_key,
            {
                "accepted_taxon_key": metadata.accepted_taxon_key,
                "scientific_name": metadata.scientific_name,
                "rank": metadata.rank,
                "family_key": metadata.family_key,
                "family": metadata.family,
                "genus_key": metadata.genus_key,
                "genus": metadata.genus,
                "label_scores": {},
            },
        )
        for index_index in indices:
            group["label_scores"].setdefault(
                index.labels[index_index],
                _cosine_similarity(image_vector, index.embedding_vectors[index_index]),
            )
    return groups


def hierarchical_result_to_object_score_row(
    *,
    item: dict[str, Any],
    result: ButterflyCascadeResult,
    scorer: ObjectBioClipScorer,
    family_top_k: int,
    species_first_pass_top_k: int,
    species_rerank_top_k: int,
) -> dict[str, Any]:
    species_top1 = result.species_top1
    genera = _unique_text(score.genus for score in result.species_top5 if score.genus)
    return {
        "source": result.source or str(item.get("source") or ""),
        "flickr_photo_id": result.flickr_photo_id or str(item.get("flickr_photo_id") or ""),
        "detection_id": result.detection_id or str(item.get("detection_id") or ""),
        "crop_hash": result.crop_hash or str(item.get("crop_hash") or ""),
        "model_id": scorer.model_id,
        "model_version": scorer.model_version,
        "model_checkpoint": scorer.model_checkpoint,
        "candidate_set_id": result.candidate_set_id,
        "classified_at": result.classified_at,
        "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        "candidate_selection_mode": HIERARCHICAL_CANDIDATE_SELECTION_MODE,
        "candidate_source": "gbif_classification_table",
        "taxonomy_table_version": result.taxonomy_table_version,
        "taxonomy_prompt_variant_version": result.prompt_variant_version,
        "ablation_mode": str(item.get("ablation_mode") or "detector_crop"),
        "species_first_pass_top_k": int(species_first_pass_top_k),
        "species_rerank_top_k": int(species_rerank_top_k),
        "species_rerank_strategy": HIERARCHICAL_SPECIES_RERANK_STRATEGY,
        "triage_group_top": "butterfly_like",
        "triage_group_scores": {"butterfly_like": float(item.get("detector_score") or 0.0)},
        "family_top3": [score.scientific_name for score in result.family_top3[:family_top_k]],
        "family_top3_accepted_taxon_keys": [score.accepted_taxon_key for score in result.family_top3[:family_top_k]],
        "family_top3_scores": [score.score for score in result.family_top3[:family_top_k]],
        "family_top1": result.selected_family,
        "family_top1_score": result.family_top1_score,
        "family_margin": _margin(result.family_top3),
        "selected_family_key": result.selected_family_key,
        "selected_family": result.selected_family,
        "genus_top8": genera[:8],
        "genus_top1": species_top1.genus if species_top1 is not None else None,
        "genus_top1_score": species_top1.score if species_top1 is not None and species_top1.genus else None,
        "genus_margin": None,
        "species_candidate_family_key": result.selected_family_key,
        "species_candidate_family": result.selected_family,
        "species_candidate_count": int(result.species_candidate_count),
        "species_top20": [score.scientific_name for score in result.species_top20],
        "species_top20_accepted_taxon_keys": [score.accepted_taxon_key for score in result.species_top20],
        "species_top20_scores": [score.score for score in result.species_top20],
        "species_top5": [score.scientific_name for score in result.species_top5],
        "species_top5_accepted_taxon_keys": [score.accepted_taxon_key for score in result.species_top5],
        "species_top5_scores": [score.score for score in result.species_top5],
        "species_top1": species_top1.scientific_name if species_top1 is not None else None,
        "species_top1_scientific_name": species_top1.scientific_name if species_top1 is not None else None,
        "species_top1_accepted_taxon_key": species_top1.accepted_taxon_key if species_top1 is not None else None,
        "accepted_taxon_key": species_top1.accepted_taxon_key if species_top1 is not None else None,
        "species_top1_score": result.species_top1_score,
        "species_top1_margin": result.species_top1_margin,
        "target_accepted_taxon_key": None,
        "target_species_score": None,
        "target_species_rank": None,
        "geospatial_prior_score": 0.0,
        "geospatial_prior_reason": "not_applied_open_classification",
        "text_evidence_score": 0.0,
        "comment_evidence_score": 0.0,
        "is_target_positive": False,
        "is_negative_material": False,
        "occurrence_bin": "in_review",
        "bin_reason": "hierarchical_open_classification_requires_review",
    }


def _cascade_result(
    *,
    item: dict[str, Any],
    taxonomy_store: ButterflyTaxonomyStore,
    family_top: tuple[TaxonScore, ...],
    selected_family_key: str,
    species_candidate_count: int,
    species_top20: tuple[TaxonScore, ...],
    species_top5: tuple[TaxonScore, ...],
    species_top1: TaxonScore | None,
) -> ButterflyCascadeResult:
    if not family_top:
        raise ValueError("butterfly taxonomy store produced no family scores")
    selected_family = family_top[0]
    taxonomy_table_version = _taxonomy_table_version(taxonomy_store)
    prompt_variant_version = _prompt_variant_version(taxonomy_store)
    return ButterflyCascadeResult(
        source=str(item.get("source") or ""),
        flickr_photo_id=str(item.get("flickr_photo_id") or ""),
        detection_id=str(item.get("detection_id") or ""),
        crop_hash=str(item.get("crop_hash") or ""),
        classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        candidate_set_id=_candidate_set_id(taxonomy_store, taxonomy_table_version, prompt_variant_version),
        taxonomy_table_version=taxonomy_table_version,
        prompt_variant_version=prompt_variant_version,
        family_top3=family_top,
        selected_family_key=selected_family_key,
        selected_family=selected_family.scientific_name,
        species_candidate_count=int(species_candidate_count),
        species_top20=species_top20,
        species_top5=species_top5,
        species_top1=species_top1,
        family_top1_score=selected_family.score,
        species_top1_score=species_top1.score if species_top1 is not None else 0.0,
        species_top1_margin=_margin(species_top5),
        classified_at=datetime.now(UTC).isoformat(),
    )


def _validate_top_k(
    *,
    family_top_k: int,
    species_first_pass_top_k: int,
    species_rerank_top_k: int,
) -> tuple[int, int, int]:
    family = int(family_top_k)
    first_pass = int(species_first_pass_top_k)
    rerank = int(species_rerank_top_k)
    if family <= 0:
        raise ValueError("family_top_k must be positive")
    if first_pass <= 0:
        raise ValueError("species_first_pass_top_k must be positive")
    if rerank <= 0:
        raise ValueError("species_rerank_top_k must be positive")
    if rerank > first_pass:
        raise ValueError("species_rerank_top_k must be <= species_first_pass_top_k")
    return family, first_pass, rerank


def _raise_for_invalid_taxonomy_store(taxonomy_store: ButterflyTaxonomyStore) -> None:
    fatal = [finding for finding in taxonomy_store.validation_findings() if finding.get("severity") == "fatal"]
    if fatal:
        codes = ", ".join(str(finding.get("code")) for finding in fatal)
        raise ValueError(f"invalid butterfly taxonomy store: {codes}")


def _validated_taxonomy_text_embedding_cache_for_scorer(
    cache: pl.DataFrame,
    *,
    taxonomy_store: ButterflyTaxonomyStore,
    scorer: ObjectBioClipScorer,
) -> pl.DataFrame:
    validate_taxonomy_text_embedding_cache(
        cache,
        taxonomy_store,
        model_id=scorer.model_id,
        model_checkpoint=scorer.model_checkpoint,
    )
    return cache.filter((pl.col("model_id") == scorer.model_id) & (pl.col("model_checkpoint") == scorer.model_checkpoint))


def _embed_image_items_for_cached_taxonomy(
    scorer: ObjectBioClipScorer,
    items: Sequence[dict[str, Any]],
) -> list[list[float]]:
    embed_items = getattr(scorer, "embed_image_items", None)
    if not callable(embed_items):
        raise ValueError("taxonomy text embedding cache requires a scorer with embed_image_items support")
    embeddings = embed_items(list(items))
    if len(embeddings) != len(items):
        raise ValueError(f"BioCLIP image embedder returned {len(embeddings)} rows for {len(items)} crops")
    return [_float_vector(embedding) for embedding in embeddings]


def _score_label_sets_for_items(
    scorer: ObjectBioClipScorer,
    items: Sequence[dict[str, Any]],
    label_sets: Mapping[str, Sequence[str]],
) -> dict[str, list[dict[str, float]]]:
    label_sets_by_name = {str(name): tuple(str(label) for label in labels) for name, labels in label_sets.items()}
    batch_scorer = getattr(scorer, "score_label_sets_batch", None)
    if callable(batch_scorer):
        return _coerce_label_set_batch_scores(batch_scorer(items, label_sets_by_name), label_sets_by_name, len(items))
    return {
        name: [scorer.score(item, labels) for item in items]
        for name, labels in label_sets_by_name.items()
    }


def _coerce_label_set_batch_scores(
    scores_by_label_set: Mapping[str, Sequence[Mapping[str, Any]]],
    label_sets: Mapping[str, Sequence[str]],
    expected_count: int,
) -> dict[str, list[dict[str, float]]]:
    output: dict[str, list[dict[str, float]]] = {}
    for name in label_sets:
        try:
            raw_scores = list(scores_by_label_set[name])
        except KeyError as exc:
            raise ValueError(f"BioCLIP batch scorer did not return label set {name!r}") from exc
        output[name] = _coerce_score_batch(raw_scores, expected_count=expected_count)
    return output


def _coerce_score_batch(scores_by_item: Sequence[Mapping[str, Any]], *, expected_count: int) -> list[dict[str, float]]:
    scores = list(scores_by_item)
    if len(scores) != expected_count:
        raise ValueError(f"BioCLIP batch scorer returned {len(scores)} rows for {expected_count} images")
    return [{str(label): float(score) for label, score in dict(row).items()} for row in scores]


def _label_tuple(label_rows: pl.DataFrame) -> tuple[str, ...]:
    if "label" not in label_rows.columns or label_rows.is_empty():
        return ()
    return tuple(str(label) for label in label_rows.select("label").to_series().to_list() if str(label or "").strip())


def _assert_species_top20_family(species_top20: Sequence[TaxonScore], *, selected_family_key: str) -> None:
    wrong_family = [
        score.accepted_taxon_key
        for score in species_top20
        if str(score.family_key or "") != str(selected_family_key)
    ]
    if wrong_family:
        raise AssertionError(
            "hierarchical species_top20 contains taxa outside selected family_key="
            f"{selected_family_key!r}: {', '.join(wrong_family)}"
        )


def _single_cache_model_pair(frame: pl.DataFrame) -> tuple[str, str]:
    if frame.is_empty():
        raise ValueError("taxonomy text embedding cache is empty")
    missing = sorted({"model_id", "model_checkpoint"} - set(frame.columns))
    if missing:
        raise ValueError("taxonomy text embedding cache is missing columns: " + ", ".join(missing))
    pairs = {
        (str(row["model_id"]), str(row["model_checkpoint"]))
        for row in frame.select(["model_id", "model_checkpoint"]).unique().to_dicts()
        if str(row["model_id"] or "").strip() and str(row["model_checkpoint"] or "").strip()
    }
    if not pairs:
        raise ValueError("taxonomy text embedding cache has no model_id/model_checkpoint pair")
    if len(pairs) > 1:
        raise ValueError("taxonomy text embedding cache must contain exactly one model_id/model_checkpoint pair")
    return next(iter(pairs))


def _taxonomy_table_version(taxonomy_store: ButterflyTaxonomyStore) -> str:
    manifest = taxonomy_store.manifest or {}
    return str(manifest.get("classification_table_version") or _first_value(taxonomy_store.classification_taxa, "classification_table_version") or CLASSIFICATION_TABLE_VERSION)


def _prompt_variant_version(taxonomy_store: ButterflyTaxonomyStore) -> str:
    manifest = taxonomy_store.manifest or {}
    return str(manifest.get("prompt_variant_version") or _first_value(taxonomy_store.family_labels, "prompt_variant_version") or PROMPT_VARIANT_VERSION)


def _candidate_set_id(taxonomy_store: ButterflyTaxonomyStore, taxonomy_table_version: str, prompt_variant_version: str) -> str:
    manifest = taxonomy_store.manifest or {}
    explicit = str(manifest.get("candidate_set_id") or "").strip()
    if explicit:
        return explicit
    registry_version = str(manifest.get("registry_version") or _first_value(taxonomy_store.classification_taxa, "registry_version") or "").strip()
    parts = [part for part in (registry_version, taxonomy_table_version, prompt_variant_version) if part]
    return ":".join(parts)


def _first_value(frame: pl.DataFrame, column: str) -> object:
    if column not in frame.columns or frame.is_empty():
        return None
    values = frame.select(column).to_series().drop_nulls().to_list()
    return values[0] if values else None


def _margin(scores: Sequence[TaxonScore]) -> float | None:
    if len(scores) < 2:
        return None
    return float(scores[0].score - scores[1].score)


def _unique_text(values: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = _clean_text(value)
        if text and text not in seen:
            output.append(text)
            seen.add(text)
    return output


def _require_label_columns(
    label_rows: pl.DataFrame,
    *,
    taxon_key_column: str,
    taxon_name_column: str,
) -> None:
    missing = [
        column
        for column in ("label", taxon_key_column, taxon_name_column)
        if column not in label_rows.columns
    ]
    if missing:
        raise ValueError(f"taxonomy label rows are missing required columns: {', '.join(missing)}")


def _aggregate_values(values: list[float], *, aggregation: str) -> float:
    if aggregation == "max":
        return max(values)
    if aggregation == "mean":
        return sum(values) / len(values)
    weights = [exp(value) for value in values]
    weight_sum = sum(weights)
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / weight_sum if weight_sum else 0.0


def _rank_for_label_row(row: Mapping[str, Any], *, taxon_key_column: str) -> str:
    rank = _clean_text(row.get("rank"))
    if rank:
        return rank.upper()
    if taxon_key_column == "family_key":
        return "FAMILY"
    return "SPECIES"


def _taxon_family_key(row: Mapping[str, Any], *, taxon_key: str, taxon_key_column: str) -> str | None:
    if taxon_key_column == "family_key":
        return taxon_key
    return _clean_text(row.get("family_key")) or None


def _taxon_family(row: Mapping[str, Any], *, scientific_name: str, taxon_key_column: str) -> str | None:
    if taxon_key_column == "family_key":
        return scientific_name
    return _clean_text(row.get("family")) or None


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _float_vector(value: object) -> list[float]:
    return [float(item) for item in value]  # type: ignore[union-attr]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"embedding dimensions differ: image={len(left)}, text={len(right)}")
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


__all__ = [
    "BUTTERFLY_CASCADE_RESULT_SCHEMA",
    "HIERARCHICAL_CANDIDATE_SELECTION_MODE",
    "HIERARCHICAL_OBJECT_SCORE_SCHEMA_EXTENSIONS",
    "HIERARCHICAL_SPECIES_RERANK_STRATEGY",
    "TAXON_SCORE_DTYPE",
    "TAXON_SCORE_SCHEMA",
    "ButterflyCascadeResult",
    "TaxonScore",
    "aggregate_taxon_prompt_scores",
    "butterfly_cascade_result_to_dict",
    "butterfly_cascade_results_frame",
    "classify_butterfly_crops_hierarchical_batch",
    "classify_butterfly_crop_hierarchical",
    "clear_family_embedding_index_cache",
    "family_index_cache_entries",
    "hierarchical_result_to_object_score_row",
    "rank_species_with_cached_text_embeddings",
    "taxon_score_to_dict",
]
