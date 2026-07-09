from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import exp
from typing import Any, Mapping, Protocol, Sequence

import polars as pl

from biominer.bioclip.classification_modes import (
    DEFAULT_FAMILY_TOP_K,
    DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    DEFAULT_SPECIES_RERANK_TOP_K,
    HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
)
from biominer.registry.classification_table import (
    CLASSIFICATION_TABLE_VERSION,
    PROMPT_VARIANT_VERSION,
    ButterflyTaxonomyStore,
)


HIERARCHICAL_CANDIDATE_SELECTION_MODE = "gbif_family_first"
HIERARCHICAL_SPECIES_RERANK_STRATEGY = "rerank_all_first_pass_top20"

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
) -> ButterflyCascadeResult:
    family_top_k, species_first_pass_top_k, species_rerank_top_k = _validate_top_k(
        family_top_k=family_top_k,
        species_first_pass_top_k=species_first_pass_top_k,
        species_rerank_top_k=species_rerank_top_k,
    )
    _raise_for_invalid_taxonomy_store(taxonomy_store)

    family_label_rows = _enabled_label_rows(taxonomy_store.family_labels)
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

    species_taxa = taxonomy_store.species_for_family(selected_family_key)
    species_label_rows = _enabled_label_rows(taxonomy_store.species_labels).filter(
        pl.col("family_key") == selected_family_key
    )
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

    family_label_rows = _enabled_label_rows(taxonomy_store.family_labels)
    family_labels = _label_tuple(family_label_rows)
    if not family_labels:
        raise ValueError("butterfly taxonomy store has no enabled family labels")
    family_scores_by_item = _score_label_sets_for_items(
        scorer,
        batch_items,
        {"family": family_labels},
    )["family"]

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
        species_label_rows = _enabled_label_rows(taxonomy_store.species_labels).filter(
            pl.col("family_key") == family_key
        )
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


def _enabled_label_rows(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    rows = frame.filter(pl.col("enabled")) if "enabled" in frame.columns else frame
    sort_columns = [column for column in ("family", "genus", "scientific_name", "sort_order", "label") if column in rows.columns]
    return rows.sort(sort_columns) if sort_columns else rows


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
    "taxon_score_to_dict",
]
