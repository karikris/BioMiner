from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from biominer.common.text import normalize_text
from biominer.config.keywords import load_json_config


ACCEPTED_TERM_TYPES = {"scientific_name", "common_name"}
TEXT_COLUMNS = ("raw_title", "raw_description", "raw_tags", "machine_tags")


def accepted_name_terms_from_keyword_json(path: str | Path) -> tuple[str, ...]:
    data = load_json_config(path)
    terms: list[str] = []
    seen: set[str] = set()
    for entry in _keyword_entries(data):
        if str(entry.get("term_type") or "").casefold() not in ACCEPTED_TERM_TYPES:
            continue
        term = str(entry.get("term") or "").strip().strip('"')
        if not term:
            continue
        key = normalize_text(term)
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return tuple(terms)


def build_name_evidence_report(
    *,
    metadata_path: str | Path,
    bioclip_output_path: str | Path,
    keywords_json: str | Path,
    target_species: str,
    score_threshold: float = 0.9,
) -> dict[str, Any]:
    accepted_terms = accepted_name_terms_from_keyword_json(keywords_json)
    metadata = _metadata_with_flags(pl.read_parquet(metadata_path), accepted_terms)
    bioclip = pl.read_parquet(bioclip_output_path)
    frame = bioclip.join(metadata, on="flickr_photo_id", how="left").with_columns(
        [
            pl.col("accepted_name_in_any_metadata_text").fill_null(False),
            pl.col("accepted_name_in_any_query").fill_null(False),
            pl.col("accepted_name_in_text_search_row").fill_null(False),
            pl.col("accepted_name_in_tag_search_row").fill_null(False),
            pl.col("all_query_labels").fill_null([]),
            (pl.col("occurrence_bin") == "gold").alias("_is_gold"),
            (pl.col("species_top1_scientific_name") == target_species).alias("_is_target_species"),
            (pl.col("species_top1_score") > score_threshold).alias("_score_gt_threshold"),
        ]
    )
    candidate = frame.filter(pl.col("_is_gold") & pl.col("_is_target_species") & pl.col("_score_gt_threshold"))
    without_name = candidate.filter(~pl.col("accepted_name_in_any_metadata_text"))
    return {
        "target_species": target_species,
        "score_threshold": score_threshold,
        "accepted_name_terms": list(accepted_terms),
        "gold_records": frame.filter(pl.col("_is_gold")).height,
        "gold_unique_photo_ids": frame.filter(pl.col("_is_gold"))["flickr_photo_id"].n_unique(),
        "gold_target_species_top1": frame.filter(pl.col("_is_gold") & pl.col("_is_target_species")).height,
        "gold_target_species_score_gt_threshold": candidate.height,
        "accepted_name_in_any_metadata_text": candidate.filter(pl.col("accepted_name_in_any_metadata_text")).height,
        "accepted_name_in_any_query": candidate.filter(pl.col("accepted_name_in_any_query")).height,
        "accepted_name_in_text_search_row": candidate.filter(pl.col("accepted_name_in_text_search_row")).height,
        "accepted_name_in_tag_search_row": candidate.filter(pl.col("accepted_name_in_tag_search_row")).height,
        "candidate_dwc_tier_count": candidate.filter(pl.col("accepted_name_in_any_metadata_text")).height,
        "candidate_dwc_tier_strict_text_count": candidate.filter(pl.col("accepted_name_in_text_search_row")).height,
        "candidate_dwc_tier_any_query_count": candidate.filter(pl.col("accepted_name_in_any_query")).height,
        "records_without_accepted_name_count": without_name.height,
        "top_query_labels_for_records_without_accepted_name": _top_list_labels(without_name, "all_query_labels", "query_label"),
        "top_query_terms_for_records_without_accepted_name": _top_list_labels(without_name, "all_query_terms", "query_term"),
        "score_bins": _score_bins(frame.filter(pl.col("_is_gold"))),
        "score_summary": _score_summary(frame.filter(pl.col("_is_gold"))),
    }


def write_name_evidence_report(path: str | Path, report: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _metadata_with_flags(metadata: pl.DataFrame, accepted_terms: tuple[str, ...]) -> pl.DataFrame:
    if metadata.is_empty():
        return pl.DataFrame()
    frame = _ensure_columns(metadata, [*TEXT_COLUMNS, "all_query_terms", "all_query_labels", "query_field", "query_term"])
    rows: list[dict[str, Any]] = []
    for row in frame.to_dicts():
        text_values = [row.get(column) for column in TEXT_COLUMNS]
        query_terms = _list_values(row.get("all_query_terms")) or _list_values(row.get("query_term"))
        query_labels = _list_values(row.get("all_query_labels"))
        if not query_labels and row.get("query_field") and row.get("query_term"):
            query_labels = [f"{row['query_field']}:{row['query_term']}"]
        metadata_text = " ".join(str(value or "") for value in [*text_values, *query_terms])
        rows.append(
            {
                "flickr_photo_id": str(row.get("flickr_photo_id") or ""),
                "accepted_name_in_any_metadata_text": _contains_any(metadata_text, accepted_terms),
                "accepted_name_in_any_query": _contains_any(" ".join(query_terms), accepted_terms),
                "accepted_name_in_text_search_row": _contains_query_label(query_labels, accepted_terms, prefix="text:"),
                "accepted_name_in_tag_search_row": _contains_query_label(query_labels, accepted_terms, prefix="tags:"),
                "all_query_labels": query_labels,
                "all_query_terms": query_terms,
            }
        )
    return pl.DataFrame(rows)


def _keyword_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    groups = data.get("dictionary_groups", data)
    entries: list[dict[str, Any]] = []
    if isinstance(groups, dict):
        for value in groups.values():
            if isinstance(value, list):
                entries.extend(item for item in value if isinstance(item, dict))
            elif isinstance(value, dict):
                for nested in value.values():
                    if isinstance(nested, list):
                        entries.extend(item for item in nested if isinstance(item, dict))
    return entries


def _ensure_columns(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    result = frame
    for column in columns:
        if column not in result.columns:
            result = result.with_columns(pl.lit(None).alias(column))
    return result


def _list_values(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if value in (None, ""):
        return []
    return [str(value)]


def _contains_any(value: object, terms: tuple[str, ...]) -> bool:
    text = normalize_text(value)
    return any(normalize_text(term) in text for term in terms)


def _contains_query_label(labels: list[str], terms: tuple[str, ...], *, prefix: str) -> bool:
    return any(label.casefold().startswith(prefix) and _contains_any(label.split(":", 1)[1], terms) for label in labels if ":" in label)


def _top_list_labels(frame: pl.DataFrame, column: str, output_name: str) -> list[dict[str, Any]]:
    if frame.is_empty() or column not in frame.columns:
        return []
    exploded = frame.select(column).explode(column).drop_nulls()
    if exploded.is_empty():
        return []
    return (
        exploded.group_by(column)
        .len(name="records")
        .sort(["records", column], descending=[True, False])
        .rename({column: output_name})
        .to_dicts()
    )[:12]


def _score_bins(frame: pl.DataFrame) -> list[dict[str, Any]]:
    if frame.is_empty() or "species_top1_score" not in frame.columns:
        return []
    return (
        frame.with_columns(
            pl.when(pl.col("species_top1_score") > 0.99).then(pl.lit(">0.99"))
            .when(pl.col("species_top1_score") > 0.95).then(pl.lit(">0.95-0.99"))
            .when(pl.col("species_top1_score") > 0.90).then(pl.lit(">0.90-0.95"))
            .when(pl.col("species_top1_score") > 0.70).then(pl.lit(">0.70-0.90"))
            .otherwise(pl.lit("<=0.70"))
            .alias("score_bin")
        )
        .group_by("score_bin")
        .len(name="records")
        .sort("score_bin")
        .to_dicts()
    )


def _score_summary(frame: pl.DataFrame) -> dict[str, float | None]:
    if frame.is_empty() or "species_top1_score" not in frame.columns:
        return {"min": None, "median": None, "max": None}
    return frame.select(
        [
            pl.col("species_top1_score").min().alias("min"),
            pl.col("species_top1_score").median().alias("median"),
            pl.col("species_top1_score").max().alias("max"),
        ]
    ).to_dicts()[0]
