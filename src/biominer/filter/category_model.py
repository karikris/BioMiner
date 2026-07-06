from __future__ import annotations

from typing import Any


ALLOWED_IMAGE_CATEGORIES = (
    "adult_butterfly",
    "life_stage_non_adult",
    "museum_specimen",
    "artwork",
    "tattoo",
    "ai_generated",
    "logo_or_brand",
    "object_or_product",
    "textile_or_pattern",
    "other_insect",
    "not_lepidoptera",
    "unknown",
)
ALLOWED_LIFE_STAGES = (
    "adult_butterfly",
    "egg",
    "caterpillar",
    "larva",
    "pupa",
    "chrysalis",
    "unknown",
)
DEFAULT_IMAGE_CATEGORY = "unknown"
DEFAULT_LIFE_STAGE = "unknown"

_LIFE_STAGE_TERMS = (
    ("chrysalis", "chrysalis"),
    ("caterpillar", "caterpillar"),
    ("larva", "larva"),
    ("pupa", "pupa"),
    ("egg", "egg"),
)
def category_defaults() -> dict[str, str | None]:
    return {
        "image_category": DEFAULT_IMAGE_CATEGORY,
        "life_stage": DEFAULT_LIFE_STAGE,
        "negative_filter_reason": None,
    }


def infer_category_from_text(
    text: str,
    *,
    museum_detected: bool = False,
    specimen_detected: bool = False,
    artwork_detected: bool = False,
    tattoo_detected: bool = False,
    ai_generated_detected: bool = False,
    non_target_order_detected: bool = False,
) -> dict[str, str | None]:
    normalized = _normalize(text)
    if museum_detected or specimen_detected:
        return _category("museum_specimen", DEFAULT_LIFE_STAGE, "museum_specimen")
    if artwork_detected:
        return _category("artwork", DEFAULT_LIFE_STAGE, "artwork")
    if tattoo_detected:
        return _category("tattoo", DEFAULT_LIFE_STAGE, "tattoo")
    if ai_generated_detected or _has_any(normalized, ("ai generated", "ai-generated")):
        return _category("ai_generated", DEFAULT_LIFE_STAGE, "ai_generated")
    if _has_any(normalized, ("logo", "brand", "emblem")):
        return _category("logo_or_brand", DEFAULT_LIFE_STAGE, "logo_or_brand")
    if _has_any(normalized, ("textile", "fabric", "pattern")):
        return _category("textile_or_pattern", DEFAULT_LIFE_STAGE, "textile_or_pattern")
    if _has_any(normalized, ("object", "product", "toy", "sticker")):
        return _category("object_or_product", DEFAULT_LIFE_STAGE, "object_or_product")
    life_stage = infer_life_stage_from_text(normalized)
    if life_stage != DEFAULT_LIFE_STAGE:
        return _category("life_stage_non_adult", life_stage, f"life_stage_{life_stage}")
    if non_target_order_detected:
        return _category("not_lepidoptera", "unknown", "non_target_order")
    return category_defaults()


def infer_life_stage_from_text(text: str) -> str:
    normalized = _normalize(text)
    for term, life_stage in _LIFE_STAGE_TERMS:
        if term in normalized:
            return life_stage
    return DEFAULT_LIFE_STAGE


def infer_category_from_record(row: dict[str, Any]) -> dict[str, str | None]:
    if row.get("image_category") or row.get("life_stage") or row.get("negative_filter_reason"):
        return {
            "image_category": str(row.get("image_category") or DEFAULT_IMAGE_CATEGORY),
            "life_stage": str(row.get("life_stage") or DEFAULT_LIFE_STAGE),
            "negative_filter_reason": row.get("negative_filter_reason"),
        }
    category = infer_category_from_text(
        " ".join(str(row.get(key) or "") for key in ("raw_title", "raw_description", "raw_tags", "machine_tags", "comments_text", "bioclip_top1_label", "top1_label")),
        museum_detected=bool(row.get("museum_detected") or row.get("specimen_detected")),
        specimen_detected=bool(row.get("specimen_detected")),
        artwork_detected=bool(row.get("artwork_detected")),
        tattoo_detected=bool(row.get("tattoo_detected")),
        ai_generated_detected=bool(row.get("ai_generated_detected")),
        non_target_order_detected=bool(row.get("non_target_order_detected")),
    )
    if category["image_category"] == "unknown" and _has_visual_species_evidence(row):
        return _category("adult_butterfly", "adult_butterfly", None)
    return category


def category_from_negative_reason(reason: str | None) -> dict[str, str | None]:
    if not reason:
        return category_defaults()
    normalized = _normalize(reason)
    if "museum" in normalized or "specimen" in normalized or "pinned" in normalized:
        return _category("museum_specimen", DEFAULT_LIFE_STAGE, reason)
    if "artwork" in normalized or "illustration" in normalized:
        return _category("artwork", DEFAULT_LIFE_STAGE, reason)
    if "tattoo" in normalized:
        return _category("tattoo", DEFAULT_LIFE_STAGE, reason)
    if "ai" in normalized and "generated" in normalized:
        return _category("ai_generated", DEFAULT_LIFE_STAGE, reason)
    if "caterpillar" in normalized:
        return _category("life_stage_non_adult", "caterpillar", reason)
    if "chrysalis" in normalized:
        return _category("life_stage_non_adult", "chrysalis", reason)
    if "pupa" in normalized:
        return _category("life_stage_non_adult", "pupa", reason)
    if "larva" in normalized:
        return _category("life_stage_non_adult", "larva", reason)
    if "egg" in normalized:
        return _category("life_stage_non_adult", "egg", reason)
    if "moth" in normalized or "not_lepidoptera" in normalized or "not_butterfly" in normalized or "non_target_order" in normalized or "other_order" in normalized:
        return _category("not_lepidoptera", DEFAULT_LIFE_STAGE, reason)
    if "other_insect" in normalized or "beetle" in normalized or "fly" in normalized or "wasp" in normalized:
        return _category("other_insect", DEFAULT_LIFE_STAGE, reason)
    if "object" in normalized or "background" in normalized or "product" in normalized:
        return _category("object_or_product", DEFAULT_LIFE_STAGE, reason)
    return _category("unknown", "unknown", reason)


def _category(image_category: str, life_stage: str, negative_filter_reason: str | None) -> dict[str, str | None]:
    return {
        "image_category": image_category,
        "life_stage": life_stage,
        "negative_filter_reason": negative_filter_reason,
    }


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _normalize(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _has_visual_species_evidence(row: dict[str, Any]) -> bool:
    if row.get("bioclip_species_agreement_status") in {"exact_species_agreement", "accepted_species_agreement"}:
        return True
    if row.get("is_target_positive") is True:
        return True
    label = str(row.get("bioclip_top1_label") or row.get("top1_label") or "")
    return _looks_like_binomial(label)


def _looks_like_binomial(value: str) -> bool:
    parts = value.split()
    return any(left[:1].isupper() and right[:1].islower() for left, right in zip(parts, parts[1:]))
