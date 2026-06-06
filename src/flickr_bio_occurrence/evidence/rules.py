from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import polars as pl


PUBLICATION_STATES = ("gold", "silver", "bronze", "in_review")
REVIEW_REASON_PRECEDENCE = (
    "missing_image",
    "artwork",
    "museum_specimen",
    "non_target_order",
    "species_conflict",
    "multiple_species",
    "captivity_suspected",
    "low_confidence",
    "no_human_verification",
    "missing_comments",
    "api_error",
)
POSITIVE_SPECIES_AGREEMENT = {"exact_species_agreement", "same_genus_agreement", "same_family_agreement", "vision_only"}
CONFLICT_SPECIES_AGREEMENT = {"text_vision_conflict", "non_butterfly"}
TARGET_BUTTERFLY_LABELS = {
    "a photo of Papilio demoleus",
    "a photo of lime butterfly",
    "a photo of chequered swallowtail",
    "a photo of citrus swallowtail",
    "a photo of a swallowtail butterfly",
    "a photo of a butterfly",
}
HARD_EXCLUSION_REASONS = {
    "missing_image",
    "artwork",
    "museum_specimen",
    "non_target_order",
    "multiple_species",
    "captivity_suspected",
    "api_error",
}
BIOCLIP_CONFIDENCE_THRESHOLD = 0.50


def classify_evidence_frame(evidence: pl.DataFrame) -> pl.DataFrame:
    rows = [classify_evidence_row(row) for row in evidence.to_dicts()]
    return pl.DataFrame(rows) if rows else _empty_classified_evidence_frame(evidence)


def classify_evidence_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [classify_evidence_row(row) for row in rows]


def classify_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    reasons = review_reasons_for_evidence(row)
    hard_exclusion = bool(set(reasons) & HARD_EXCLUSION_REASONS)
    human_verified = bool(row.get("human_verification_detected"))
    score = _bioclip_top1_score(row)
    has_bioclip = score is not None
    high_confidence = bool(score is not None and score >= BIOCLIP_CONFIDENCE_THRESHOLD)
    positive_agreement = species_agreement_is_positive(row)
    species_conflict = "species_conflict" in reasons

    if human_verified and high_confidence and positive_agreement and not hard_exclusion:
        state = "gold"
        state_reason = "human_verified_bioclip_positive"
    elif human_verified and (not has_bioclip or not high_confidence or species_conflict) and not hard_exclusion:
        state = "silver"
        state_reason = _silver_reason(has_bioclip=has_bioclip, high_confidence=high_confidence, species_conflict=species_conflict)
    elif high_confidence and target_signal_is_positive(row) and not human_verified and not hard_exclusion:
        state = "bronze"
        state_reason = "bioclip_positive_without_human_verification"
    else:
        state = "in_review"
        state_reason = "requires_review"

    review_reason = reasons if state == "in_review" else []
    if state == "in_review" and not review_reason:
        review_reason = ["low_confidence"]
    return {
        **row,
        "publication_state": state,
        "publication_state_reason": state_reason,
        "review_reason": review_reason,
    }


def review_reasons_for_evidence(row: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    if not row.get("image_url"):
        candidates.append("missing_image")
    if bool(row.get("artwork_detected")) or _has_review_flag(row, "artwork_context"):
        candidates.append("artwork")
    if bool(row.get("museum_detected")) or bool(row.get("specimen_detected")) or _has_any_review_flag(row, {"museum_context", "specimen_context"}):
        candidates.append("museum_specimen")
    if bool(row.get("non_target_order_detected")) or _has_review_flag(row, "non_target_order_context"):
        candidates.append("non_target_order")
    if species_agreement_is_conflict(row):
        candidates.append("species_conflict")
    if _multiple_species_detected(row):
        candidates.append("multiple_species")
    if bool(row.get("captive_detected")) or _has_review_flag(row, "captive_context"):
        candidates.append("captivity_suspected")
    if _is_low_confidence(row):
        candidates.append("low_confidence")
    if not bool(row.get("human_verification_detected")):
        candidates.append("no_human_verification")
    if _comments_missing(row):
        candidates.append("missing_comments")
    if _has_review_flag(row, "api_error") or bool(row.get("api_error")):
        candidates.append("api_error")
    return _ordered_unique_reasons(candidates)


def species_agreement_is_positive(row: dict[str, Any]) -> bool:
    status = _species_agreement_status(row)
    if status in POSITIVE_SPECIES_AGREEMENT:
        return True
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
    return bool(row.get("species_text_match")) or species_agreement_is_positive(row)


def _silver_reason(*, has_bioclip: bool, high_confidence: bool, species_conflict: bool) -> str:
    if not has_bioclip:
        return "human_verified_bioclip_missing"
    if species_conflict:
        return "human_verified_bioclip_conflict"
    if not high_confidence:
        return "human_verified_bioclip_low_confidence"
    return "human_verified_bioclip_weak"


def _is_low_confidence(row: dict[str, Any]) -> bool:
    score = _bioclip_top1_score(row)
    return score is None or score < BIOCLIP_CONFIDENCE_THRESHOLD


def _comments_missing(row: dict[str, Any]) -> bool:
    count = row.get("comments_count")
    return count in (None, 0)


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
        pl.Series("publication_state", [], dtype=pl.String),
        pl.Series("publication_state_reason", [], dtype=pl.String),
        pl.Series("review_reason", [], dtype=pl.List(pl.String)),
    )
