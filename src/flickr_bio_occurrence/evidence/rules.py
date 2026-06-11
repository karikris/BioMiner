from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import polars as pl

from flickr_bio_occurrence.evidence.category_model import infer_category_from_record


PUBLICATION_STATES = ("gold", "silver", "bronze", "in_review")
REVIEW_REASON_PRECEDENCE = (
    "missing_image",
    "missing_bioclip",
    "artwork",
    "tattoo",
    "museum_specimen",
    "ai_generated",
    "non_target_order",
    "species_conflict",
    "multiple_species",
    "captivity_suspected",
    "low_confidence",
    "ambiguous_classification",
    "api_error",
)
POSITIVE_SPECIES_AGREEMENT = {"exact_species_agreement", "same_genus_agreement", "same_family_agreement", "vision_only"}
CONFLICT_SPECIES_AGREEMENT = {"text_vision_conflict", "non_butterfly"}
TARGET_BUTTERFLY_LABELS = {
    "a photo of Papilio demoleus",
}
TARGET_SPECIES = "Papilio demoleus"
HARD_EXCLUSION_REASONS = {
    "artwork",
    "tattoo",
    "museum_specimen",
    "ai_generated",
    "non_target_order",
    "multiple_species",
    "captivity_suspected",
}
BIOCLIP_CONFIDENCE_THRESHOLD = 0.50


def classify_evidence_frame(evidence: pl.DataFrame) -> pl.DataFrame:
    rows = [classify_evidence_row(row) for row in evidence.to_dicts()]
    return pl.DataFrame(rows) if rows else _empty_classified_evidence_frame(evidence)


def classify_evidence_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [classify_evidence_row(row) for row in rows]


def classify_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    category = infer_category_from_record(row)
    reasons = review_reasons_for_evidence(row)
    negative_reason = _negative_material_reason(row, reasons, category)
    if negative_reason and not category.get("negative_filter_reason"):
        category = {**category, "negative_filter_reason": negative_reason.removeprefix("negative_material_")}
    score = _bioclip_top1_score(row)
    target_positive = bool(score is not None and target_signal_is_positive(row))

    if "missing_image" in reasons or "api_error" in reasons:
        state = "in_review"
        state_reason = "operational_failure"
    elif negative_reason:
        state = "bronze"
        state_reason = negative_reason
    elif target_positive and score is not None and score >= BIOCLIP_CONFIDENCE_THRESHOLD:
        state = "gold"
        state_reason = "target_positive_score_gte_050"
    elif target_positive and score is not None and score < BIOCLIP_CONFIDENCE_THRESHOLD:
        state = "silver"
        state_reason = "target_positive_score_lt_050"
    elif score is not None:
        state = "bronze"
        state_reason = "below_50"
    else:
        state = "in_review"
        state_reason = "missing_bioclip" if score is None else "ambiguous_classification"

    review_reason = reasons if state == "in_review" else []
    if state == "in_review" and not review_reason:
        review_reason = [state_reason]
    return {
        **row,
        **category,
        "occurrence_bin": state,
        "bin_reason": state_reason,
        "publication_state": state,
        "publication_state_reason": state_reason,
        "review_reason": review_reason,
    }


def review_reasons_for_evidence(row: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    category = infer_category_from_record(row)
    image_category = category["image_category"]
    if not row.get("image_url"):
        candidates.append("missing_image")
    if image_category == "artwork" or bool(row.get("artwork_detected")) or _has_review_flag(row, "artwork_context"):
        candidates.append("artwork")
    if image_category == "tattoo" or bool(row.get("tattoo_detected")) or _has_review_flag(row, "tattoo_context"):
        candidates.append("tattoo")
    if image_category == "museum_specimen" or bool(row.get("museum_detected")) or bool(row.get("specimen_detected")) or _has_any_review_flag(row, {"museum_context", "specimen_context"}):
        candidates.append("museum_specimen")
    if image_category == "ai_generated" or bool(row.get("ai_generated_detected")) or _has_any_review_flag(row, {"ai_generated_context", "generated_image_context"}):
        candidates.append("ai_generated")
    if image_category in {"not_lepidoptera", "other_insect"} or bool(row.get("non_target_order_detected")) or _has_review_flag(row, "non_target_order_context"):
        candidates.append("non_target_order")
    if species_agreement_is_conflict(row):
        candidates.append("species_conflict")
    if _multiple_species_detected(row):
        candidates.append("multiple_species")
    if bool(row.get("captive_detected")) or _has_review_flag(row, "captive_context"):
        candidates.append("captivity_suspected")
    if _bioclip_top1_score(row) is None:
        candidates.append("missing_bioclip")
    elif _is_low_confidence(row) and not target_signal_is_positive(row):
        candidates.append("low_confidence")
    if _has_review_flag(row, "api_error") or bool(row.get("api_error")):
        candidates.append("api_error")
    return _ordered_unique_reasons(candidates)


def species_agreement_is_positive(row: dict[str, Any]) -> bool:
    species_name = str(row.get("species_top1_scientific_name") or "")
    if species_name:
        return _normalize_label(species_name) == _normalize_label(TARGET_SPECIES)
    status = _species_agreement_status(row)
    if status in CONFLICT_SPECIES_AGREEMENT:
        return False
    label = _normalize_label(_bioclip_top1_label(row))
    return label in {_normalize_label(value) for value in TARGET_BUTTERFLY_LABELS}


def species_agreement_is_conflict(row: dict[str, Any]) -> bool:
    status = _species_agreement_status(row)
    if status in CONFLICT_SPECIES_AGREEMENT:
        return True
    score = _bioclip_top1_score(row)
    label = _normalize_label(_bioclip_top1_label(row))
    if score is None or score < BIOCLIP_CONFIDENCE_THRESHOLD:
        return False
    return bool(label and label not in {_normalize_label(value) for value in TARGET_BUTTERFLY_LABELS})


def target_signal_is_positive(row: dict[str, Any]) -> bool:
    return species_agreement_is_positive(row)


def _is_low_confidence(row: dict[str, Any]) -> bool:
    score = _bioclip_top1_score(row)
    return score is None or score < BIOCLIP_CONFIDENCE_THRESHOLD


def _multiple_species_detected(row: dict[str, Any]) -> bool:
    names = row.get("scientific_names_detected") or []
    if not isinstance(names, list):
        return False
    species_query = str(row.get("species_query") or "").casefold()
    unique_names = {" ".join(str(name).casefold().split()) for name in names}
    if species_query:
        unique_names.discard(" ".join(species_query.split()))
    return bool(unique_names)


def _ordered_unique_reasons(reasons: Iterable[str]) -> list[str]:
    seen = set(reasons)
    return [reason for reason in REVIEW_REASON_PRECEDENCE if reason in seen]


def _has_review_flag(row: dict[str, Any], flag: str) -> bool:
    return flag in set(row.get("review_flags") or [])


def _has_any_review_flag(row: dict[str, Any], flags: set[str]) -> bool:
    return bool(set(row.get("review_flags") or []) & flags)


def _negative_material_reason(row: dict[str, Any], reasons: list[str], category: dict[str, str | None]) -> str | None:
    image_category = category["image_category"]
    if image_category in {"artwork", "tattoo", "museum_specimen", "ai_generated"}:
        return f"negative_material_{image_category}"
    if image_category in {"not_lepidoptera", "other_insect"}:
        return "negative_material_non_target_order"
    for reason in REVIEW_REASON_PRECEDENCE:
        if reason in HARD_EXCLUSION_REASONS and reason in reasons:
            return f"negative_material_{reason}"
    label = _normalize_label(_bioclip_top1_label(row))
    label_tokens = set(label.split())
    if label_tokens & {"moth", "beetle", "fly", "wasp", "object", "background"}:
        return "negative_material_non_butterfly"
    return None


def _bioclip_top1_score(row: dict[str, Any]) -> float | None:
    value = row.get("bioclip_top1_score", row.get("top1_score"))
    if value in (None, ""):
        return None
    return float(value)


def _bioclip_top1_label(row: dict[str, Any]) -> str:
    return str(row.get("bioclip_top1_label", row.get("top1_label", "")) or "")


def _species_agreement_status(row: dict[str, Any]) -> str:
    return str(row.get("bioclip_species_agreement_status", row.get("species_agreement_status", "")) or "")


def _normalize_label(value: str) -> str:
    return " ".join(value.casefold().split())


def _empty_classified_evidence_frame(evidence: pl.DataFrame) -> pl.DataFrame:
    return evidence.with_columns(
        pl.Series("occurrence_bin", [], dtype=pl.String),
        pl.Series("bin_reason", [], dtype=pl.String),
        pl.Series("image_category", [], dtype=pl.String),
        pl.Series("life_stage", [], dtype=pl.String),
        pl.Series("negative_filter_reason", [], dtype=pl.String),
        pl.Series("publication_state", [], dtype=pl.String),
        pl.Series("publication_state_reason", [], dtype=pl.String),
        pl.Series("review_reason", [], dtype=pl.List(pl.String)),
    )
