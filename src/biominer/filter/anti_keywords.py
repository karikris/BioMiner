from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from biominer.common.text import normalize_text
from biominer.config.keywords import load_json_config
from biominer.filter.category_model import infer_category_from_text
from biominer.storage.parquet import write_parquet


DROP_CATEGORIES = {
    "museum_specimen",
    "artwork",
    "tattoo",
    "ai_generated",
    "logo_or_brand",
    "object_or_product",
    "textile_or_pattern",
    "other_insect",
    "not_lepidoptera",
}
TEXT_COLUMNS = (
    "raw_title",
    "title",
    "raw_description",
    "description",
    "raw_tags",
    "tags",
    "machine_tags",
    "owner_name",
)


def load_anti_keyword_groups(path: str | Path) -> dict[str, tuple[str, ...]]:
    data = load_json_config(path)
    raw_groups = data.get("anti_keywords", data)
    if not isinstance(raw_groups, dict):
        raise ValueError("Anti-keyword config must be an object or contain an anti_keywords object")
    groups: dict[str, tuple[str, ...]] = {}
    for group_name, terms in raw_groups.items():
        if not isinstance(terms, list):
            raise ValueError(f"Anti-keyword group {group_name!r} must be a list")
        groups[str(group_name)] = tuple(str(term).strip() for term in terms if str(term).strip())
    return groups


def filter_biodiversity_records(
    frame: pl.DataFrame,
    *,
    anti_keyword_groups: dict[str, tuple[str, ...]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows = [_filter_row(row, anti_keyword_groups=anti_keyword_groups) for row in frame.to_dicts()]
    filtered = pl.DataFrame(rows) if rows else _empty_filter_frame(frame)
    kept = filtered.filter(pl.col("filter_decision") == "keep") if filtered.height else filtered
    dropped = filtered.filter(pl.col("filter_decision") == "drop") if filtered.height else filtered
    return kept, dropped


def filter_biodiversity_parquet(
    *,
    input_path: str | Path,
    anti_keywords_json: str | Path,
    output_path: str | Path,
    dropped_output_path: str | Path,
) -> dict[str, int | str]:
    frame = pl.read_parquet(input_path)
    kept, dropped = filter_biodiversity_records(frame, anti_keyword_groups=load_anti_keyword_groups(anti_keywords_json))
    write_parquet(kept, output_path)
    write_parquet(dropped, dropped_output_path)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "dropped_output": str(dropped_output_path),
        "input_rows": frame.height,
        "kept_rows": kept.height,
        "dropped_rows": dropped.height,
    }


def _filter_row(row: dict[str, Any], *, anti_keyword_groups: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    text = _combined_text(row)
    matched_group, matched_term = _first_match(text, anti_keyword_groups)
    category = infer_category_from_text(
        text,
        museum_detected=matched_group in {"museum", "specimen", "museum_specimen"},
        specimen_detected=matched_group in {"specimen", "museum_specimen"},
        artwork_detected=matched_group in {"art", "artwork"},
        tattoo_detected=matched_group == "tattoo",
        ai_generated_detected=matched_group in {"ai", "ai_generated"},
        non_target_order_detected=matched_group in {"not_lepidoptera", "other_insect"},
    )
    image_category = str(row.get("image_category") or category["image_category"])
    life_stage = str(row.get("life_stage") or category["life_stage"])
    reason = category["negative_filter_reason"] or matched_group or "biodiversity_candidate"
    if image_category in DROP_CATEGORIES:
        return {
            **row,
            **category,
            "filter_decision": "drop",
            "filter_reason": reason,
            "matched_anti_keyword_group": matched_group,
            "matched_anti_keyword": matched_term,
        }
    return {
        **row,
        "image_category": image_category,
        "life_stage": life_stage,
        "negative_filter_reason": row.get("negative_filter_reason") or category["negative_filter_reason"],
        "filter_decision": "keep",
        "filter_reason": "biodiversity_candidate",
        "matched_anti_keyword_group": matched_group,
        "matched_anti_keyword": matched_term,
    }


def _combined_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(column) or "") for column in TEXT_COLUMNS)


def _first_match(text: str, groups: dict[str, tuple[str, ...]]) -> tuple[str | None, str | None]:
    normalized = normalize_text(text)
    for group_name, terms in groups.items():
        for term in terms:
            if normalize_text(term) in normalized:
                return group_name, term
    return None, None


def _empty_filter_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.Series("filter_decision", [], dtype=pl.String),
        pl.Series("filter_reason", [], dtype=pl.String),
        pl.Series("matched_anti_keyword_group", [], dtype=pl.String),
        pl.Series("matched_anti_keyword", [], dtype=pl.String),
    )
