from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import polars as pl

from biominer.bioclip.policy import DEFAULT_BUCKET_POLICY
from biominer.bioclip.triage import BIN_CATEGORIES, NEGATIVE_RECORD_FIELDS
from biominer.filter.category_model import infer_category_from_record


PUBLICATION_STATES = ("gold", "silver", "bronze", "in_review")
PHOTO_REVIEW_REASONS = {
    "geospatial_conflict",
    "ambiguous_species_margin",
    "species_conflict",
    "taxonomy_inconsistent",
    "detected_object_without_bioclip_score",
}
REVIEW_REASON_PRECEDENCE = (
    "missing_image",
    "missing_bioclip",
    "artwork",
    "tattoo",
    "museum_specimen",
    "ai_generated",
    "logo_or_brand",
    "textile_or_pattern",
    "object_or_product",
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
HARD_EXCLUSION_REASONS = {
    "artwork",
    "tattoo",
    "museum_specimen",
    "ai_generated",
    "non_target_order",
    "multiple_species",
    "captivity_suspected",
}
BIOCLIP_CONFIDENCE_THRESHOLD = DEFAULT_BUCKET_POLICY.silver_species_threshold

METADATA_FLAG_REASON_COLUMNS = {
    "artwork_hint": "artwork",
    "museum_specimen_hint": "museum_specimen",
    "ai_generated_hint": "ai_generated",
    "logo_or_brand_hint": "logo_or_brand",
    "textile_or_pattern_hint": "textile_or_pattern",
    "object_or_product_hint": "object_or_product",
    "other_insect_hint": "non_target_order",
}
METADATA_ONLY_REVIEW_CATEGORIES = {"logo_or_brand", "textile_or_pattern", "object_or_product"}


def bucket_evidence_frame(evidence: pl.DataFrame) -> pl.DataFrame:
    return classify_evidence_frame(evidence)


def bucket_evidence_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return classify_evidence_rows(rows)


def object_occurrence_bucket(
    *,
    item: dict[str, Any],
    target_score: float,
    target_rank: int | None,
    margin: float | None,
    geo: object,
) -> tuple[str, str]:
    hard_negative_reason = object_hard_negative_reason(item)
    if hard_negative_reason:
        return "bin", hard_negative_reason
    if bool(getattr(geo, "route_to_review", False)):
        return "in_review", str(getattr(geo, "reason", "") or "geospatial_conflict")
    if target_rank is not None and target_rank != 1 and target_score >= DEFAULT_BUCKET_POLICY.silver_species_threshold:
        return "in_review", "species_conflict"
    if (
        target_score >= DEFAULT_BUCKET_POLICY.gold_species_threshold
        and margin is not None
        and margin < DEFAULT_BUCKET_POLICY.ambiguous_margin_threshold
    ):
        return "in_review", "ambiguous_species_margin"
    if target_score >= DEFAULT_BUCKET_POLICY.gold_species_threshold and _has_geo(item) and _has_event_date(item):
        return "gold", "target_species_score_ge_070"
    if target_score >= DEFAULT_BUCKET_POLICY.silver_species_threshold:
        if not _has_geo(item):
            return "silver", "missing_geo"
        if not _has_event_date(item):
            return "silver", "missing_event_date"
        return "silver", "target_species_score_ge_035"
    return "bronze", "weak_species_score"


def photo_bucket_and_reason(rows: list[dict[str, Any]], canonical_record: dict[str, Any]) -> tuple[str, str]:
    hard_negative_reason = object_hard_negative_reason(canonical_record)
    if hard_negative_reason:
        return "bin", hard_negative_reason
    bucket = photo_bucket(rows)
    return bucket, photo_bucket_reason(bucket, rows)


def photo_bucket(rows: list[dict[str, Any]]) -> str:
    buckets = [str(row.get("occurrence_bin") or "") for row in rows]
    if any(bucket == "in_review" and str(row.get("bin_reason") or "") in PHOTO_REVIEW_REASONS for row, bucket in zip(rows, buckets, strict=True)):
        return "in_review"
    for bucket in ("gold", "silver", "bronze", "in_review", "bin"):
        if bucket in buckets:
            return bucket
    return "in_review"


def photo_bucket_reason(bucket: str, rows: list[dict[str, Any]]) -> str:
    if bucket == "in_review":
        for row in rows:
            reason = str(row.get("bin_reason") or "")
            if str(row.get("occurrence_bin") or "") == bucket and reason in PHOTO_REVIEW_REASONS:
                return reason
    for row in rows:
        if str(row.get("occurrence_bin") or "") == bucket:
            return str(row.get("bin_reason") or "")
    return str(rows[0].get("bin_reason") or "") if rows else ""


def object_hard_negative_reason(record: dict[str, Any]) -> str | None:
    category = str(record.get("image_category") or "")
    reason = str(record.get("negative_filter_reason") or "")
    if _truthy(record.get("is_negative_material")):
        return _object_negative_material_reason(category=category, reason=reason)
    if category in BIN_CATEGORIES:
        return _object_negative_material_reason(category=category, reason=reason)
    for field, field_reason in NEGATIVE_RECORD_FIELDS:
        if _truthy(record.get(field)):
            return _object_negative_material_reason(category=field_reason, reason=reason)
    return None


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
    elif target_positive and score is not None and score >= DEFAULT_BUCKET_POLICY.gold_species_threshold:
        state = "gold"
        state_reason = "target_positive_score_gte_070"
    elif target_positive and score is not None and score >= DEFAULT_BUCKET_POLICY.silver_species_threshold:
        state = "silver"
        state_reason = "target_positive_score_035_to_070"
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
    if not _strong_visual_target_positive(row):
        candidates.extend(_metadata_flag_review_reasons(row, category=category))
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
    target_species = _target_species_from_row(row)
    species_name = str(row.get("species_top1_scientific_name") or "")
    if species_name and target_species:
        return _normalize_label(species_name) == _normalize_label(target_species)
    status = _species_agreement_status(row)
    if status in CONFLICT_SPECIES_AGREEMENT:
        return False
    if status == "exact_species_agreement" and target_species:
        return True
    label = _normalize_label(_bioclip_top1_label(row))
    return bool(target_species and label == _normalize_label(f"a photo of {target_species}"))


def species_agreement_is_conflict(row: dict[str, Any]) -> bool:
    status = _species_agreement_status(row)
    if status in CONFLICT_SPECIES_AGREEMENT:
        return True
    score = _bioclip_top1_score(row)
    label = _normalize_label(_bioclip_top1_label(row))
    if score is None or score < BIOCLIP_CONFIDENCE_THRESHOLD:
        return False
    target_species = _target_species_from_row(row)
    if not target_species:
        return False
    return bool(label and label != _normalize_label(f"a photo of {target_species}"))


def target_signal_is_positive(row: dict[str, Any]) -> bool:
    return species_agreement_is_positive(row)


def _strong_visual_target_positive(row: dict[str, Any]) -> bool:
    score = _bioclip_top1_score(row)
    return bool(score is not None and score >= DEFAULT_BUCKET_POLICY.gold_species_threshold and target_signal_is_positive(row))


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


def _metadata_flag_review_reasons(row: dict[str, Any], *, category: dict[str, str | None]) -> list[str]:
    reasons = [reason for column, reason in METADATA_FLAG_REASON_COLUMNS.items() if bool(row.get(column))]
    metadata_reason = str(row.get("metadata_negative_reason_hint") or "")
    if metadata_reason in {"artwork", "tattoo", "museum_specimen", "ai_generated", "logo_or_brand", "textile_or_pattern", "object_or_product", "non_target_order"}:
        reasons.append(metadata_reason)
    image_category = str(category.get("image_category") or "")
    if image_category in METADATA_ONLY_REVIEW_CATEGORIES:
        reasons.append(image_category)
    if bool(row.get("hard_negative_text_hint")) and not reasons:
        reasons.append("ambiguous_classification")
    return reasons


def _negative_material_reason(row: dict[str, Any], reasons: list[str], category: dict[str, str | None]) -> str | None:
    image_category = category["image_category"]
    if image_category in {"artwork", "tattoo", "museum_specimen", "ai_generated"}:
        return f"negative_material_{image_category}"
    if image_category in {"not_lepidoptera", "other_insect"}:
        return "negative_material_non_target_order"
    for reason in REVIEW_REASON_PRECEDENCE:
        if reason in HARD_EXCLUSION_REASONS and reason in reasons and _has_explicit_negative_evidence(row, reason):
            return f"negative_material_{reason}"
    label = _normalize_label(_bioclip_top1_label(row))
    label_tokens = set(label.split())
    if label_tokens & {"moth", "beetle", "fly", "wasp", "object", "background"}:
        return "negative_material_non_butterfly"
    return None


def _object_negative_material_reason(*, category: str, reason: str) -> str:
    value = reason or category or "image_material"
    if value in {"non_target_order", "other_order", "not_butterfly", "not_lepidoptera", "other_insect"}:
        value = "non_target_order"
    return f"negative_material_{value}"


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(value)


def _has_geo(item: dict[str, Any]) -> bool:
    return item.get("latitude") not in (None, "") and item.get("longitude") not in (None, "")


def _has_event_date(item: dict[str, Any]) -> bool:
    return bool(item.get("date_taken") or item.get("datetaken") or item.get("captured_at") or item.get("eventDate") or item.get("event_date"))


def _has_explicit_negative_evidence(row: dict[str, Any], reason: str) -> bool:
    if reason == "artwork":
        return bool(row.get("artwork_detected")) or _has_review_flag(row, "artwork_context")
    if reason == "tattoo":
        return bool(row.get("tattoo_detected")) or _has_review_flag(row, "tattoo_context")
    if reason == "museum_specimen":
        return bool(row.get("museum_detected") or row.get("specimen_detected")) or _has_any_review_flag(row, {"museum_context", "specimen_context"})
    if reason == "ai_generated":
        return bool(row.get("ai_generated_detected")) or _has_any_review_flag(row, {"ai_generated_context", "generated_image_context"})
    if reason == "non_target_order":
        return bool(row.get("non_target_order_detected")) or _has_review_flag(row, "non_target_order_context")
    if reason == "multiple_species":
        return _multiple_species_detected(row)
    if reason == "captivity_suspected":
        return bool(row.get("captive_detected")) or _has_review_flag(row, "captive_context")
    return False


def _bioclip_top1_score(row: dict[str, Any]) -> float | None:
    value = row.get("bioclip_top1_score", row.get("top1_score"))
    if value in (None, ""):
        return None
    return float(value)


def _bioclip_top1_label(row: dict[str, Any]) -> str:
    return str(row.get("bioclip_top1_label", row.get("top1_label", "")) or "")


def _species_agreement_status(row: dict[str, Any]) -> str:
    return str(row.get("bioclip_species_agreement_status", row.get("species_agreement_status", "")) or "")


def _target_species_from_row(row: dict[str, Any]) -> str | None:
    for key in ("target_scientific_name", "accepted_scientific_name", "species_query"):
        value = str(row.get(key) or "").strip()
        if value and " " in value:
            return value
    return None


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


__all__ = [
    "bucket_evidence_frame",
    "bucket_evidence_rows",
    "classify_evidence_frame",
    "classify_evidence_row",
    "classify_evidence_rows",
    "object_hard_negative_reason",
    "object_occurrence_bucket",
    "photo_bucket",
    "photo_bucket_and_reason",
    "photo_bucket_reason",
    "review_reasons_for_evidence",
    "species_agreement_is_conflict",
    "species_agreement_is_positive",
    "target_signal_is_positive",
]
