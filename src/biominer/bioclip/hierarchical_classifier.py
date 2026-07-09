from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp
from typing import Any, Mapping

import polars as pl

from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION


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
    "taxon_score_to_dict",
]
