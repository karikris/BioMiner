from __future__ import annotations

from typing import Any

import polars as pl

from biominer.common.text import normalize_text
from biominer.filter.category_model import infer_category_from_text


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

FLAG_COLUMNS: dict[str, pl.DataType] = {
    "museum_specimen_hint": pl.Boolean,
    "artwork_hint": pl.Boolean,
    "ai_generated_hint": pl.Boolean,
    "logo_or_brand_hint": pl.Boolean,
    "textile_or_pattern_hint": pl.Boolean,
    "object_or_product_hint": pl.Boolean,
    "other_insect_hint": pl.Boolean,
    "hard_negative_text_hint": pl.Boolean,
    "matched_keyword_groups": pl.List(pl.String),
    "matched_keywords": pl.List(pl.String),
    "metadata_image_category_hint": pl.String,
    "metadata_life_stage_hint": pl.String,
    "metadata_negative_reason_hint": pl.String,
}

HARD_NEGATIVE_GROUPS = {
    "museum",
    "specimen",
    "museum_specimen",
    "art",
    "artwork",
    "tattoo",
    "ai",
    "ai_generated",
    "logo",
    "logo_or_brand",
    "brand",
    "textile",
    "textile_or_pattern",
    "pattern",
    "object",
    "object_or_product",
    "product",
    "other_insect",
    "not_lepidoptera",
}


def flag_metadata_records(
    frame: pl.DataFrame,
    *,
    keyword_groups: dict[str, tuple[str, ...]],
) -> pl.DataFrame:
    rows = [_flag_row(row, keyword_groups=keyword_groups) for row in frame.to_dicts()]
    return pl.DataFrame(rows) if rows else _empty_flag_frame(frame)


def _flag_row(row: dict[str, Any], *, keyword_groups: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    text = _combined_text(row)
    matches = _all_matches(text, keyword_groups)
    groups = [group for group, _term in matches]
    terms = [term for _group, term in matches]
    category = infer_category_from_text(
        text,
        museum_detected=_has_group(groups, {"museum", "specimen", "museum_specimen"}),
        specimen_detected=_has_group(groups, {"specimen", "museum_specimen"}),
        artwork_detected=_has_group(groups, {"art", "artwork"}),
        tattoo_detected=_has_group(groups, {"tattoo"}),
        ai_generated_detected=_has_group(groups, {"ai", "ai_generated"}),
        non_target_order_detected=_has_group(groups, {"not_lepidoptera", "other_insect"}),
    )
    hard_negative_hint = any(_normalized_group(group) in HARD_NEGATIVE_GROUPS for group in groups)
    hard_negative_hint = hard_negative_hint or str(category.get("negative_filter_reason") or "") in HARD_NEGATIVE_GROUPS
    return {
        **row,
        "museum_specimen_hint": _has_group(groups, {"museum", "specimen", "museum_specimen"}) or category["image_category"] == "museum_specimen",
        "artwork_hint": _has_group(groups, {"art", "artwork"}) or category["image_category"] == "artwork",
        "ai_generated_hint": _has_group(groups, {"ai", "ai_generated"}) or category["image_category"] == "ai_generated",
        "logo_or_brand_hint": _has_group(groups, {"logo", "brand", "logo_or_brand"}) or category["image_category"] == "logo_or_brand",
        "textile_or_pattern_hint": _has_group(groups, {"textile", "pattern", "textile_or_pattern"}) or category["image_category"] == "textile_or_pattern",
        "object_or_product_hint": _has_group(groups, {"object", "product", "object_or_product"}) or category["image_category"] == "object_or_product",
        "other_insect_hint": _has_group(groups, {"other_insect", "not_lepidoptera"}) or category["image_category"] in {"other_insect", "not_lepidoptera"},
        "hard_negative_text_hint": hard_negative_hint,
        "matched_keyword_groups": _unique(groups),
        "matched_keywords": _unique(terms),
        "metadata_image_category_hint": category["image_category"],
        "metadata_life_stage_hint": category["life_stage"],
        "metadata_negative_reason_hint": category["negative_filter_reason"],
    }


def _combined_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(column) or "") for column in TEXT_COLUMNS)


def _all_matches(text: str, groups: dict[str, tuple[str, ...]]) -> list[tuple[str, str]]:
    normalized = normalize_text(text)
    matches: list[tuple[str, str]] = []
    for group_name, terms in groups.items():
        for term in terms:
            if normalize_text(term) in normalized:
                matches.append((group_name, term))
    return matches


def _has_group(groups: list[str], candidates: set[str]) -> bool:
    return bool({_normalized_group(group) for group in groups} & candidates)


def _normalized_group(group: str) -> str:
    return "_".join(normalize_text(group).split())


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = " ".join(str(value or "").split())
        key = normalize_text(cleaned)
        if not cleaned or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def _empty_flag_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns([pl.Series(name, [], dtype=dtype) for name, dtype in FLAG_COLUMNS.items()])
