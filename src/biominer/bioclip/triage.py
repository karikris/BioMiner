from __future__ import annotations

from datetime import UTC, datetime
import json
import hashlib
from pathlib import Path
from typing import Any

import polars as pl

from biominer.filter.category_model import (
    DEFAULT_LIFE_STAGE,
    category_defaults,
    category_from_negative_reason,
    infer_life_stage_from_text,
)
from biominer.common.species import species_text_matches
from biominer.bioclip.policy import DEFAULT_BUCKET_POLICY


TRIAGE_BINS = {"gold", "silver", "bronze", "bin", "in_review"}
CLASSIFICATION_STATUSES = {"success", "skipped_existing", "failed_download", "failed_bioclip", "invalid_record"}
GOLD_SPECIES_CONFIDENCE_THRESHOLD = DEFAULT_BUCKET_POLICY.gold_species_threshold
SILVER_SPECIES_CONFIDENCE_THRESHOLD = DEFAULT_BUCKET_POLICY.silver_species_threshold
BIN_CATEGORIES = {
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
NEGATIVE_LABEL_REASONS = {
    "a photo of a pinned museum specimen": "pinned_specimen",
    "a photo of artwork or illustration": "artwork",
    "a photo of a tattoo": "tattoo",
    "a photo of a moth": "moth",
    "a photo of an egg": "egg",
    "a photo of a caterpillar": "caterpillar",
    "a photo of a larva": "larva",
    "a photo of a pupa": "pupa",
    "a photo of a chrysalis": "chrysalis",
    "a photo of an insect": "other_insect",
    "a photo of a beetle": "other_insect",
    "a photo of a fly": "other_insect",
    "a photo of a wasp": "other_insect",
    "a photo of a non-butterfly": "not_butterfly",
    "a photo of a background": "object_background_non_organism",
    "a photo of an object": "object_background_non_organism",
    "a blank image": "object_background_non_organism",
    "an ai generated image": "ai_generated",
}
NEGATIVE_RECORD_FIELDS = (
    ("museum_detected", "museum_specimen"),
    ("specimen_detected", "pinned_specimen"),
    ("artwork_detected", "artwork"),
    ("tattoo_detected", "tattoo"),
    ("ai_generated_detected", "ai_generated"),
    ("other_insect_detected", "other_insect"),
    ("non_target_order_detected", "other_order"),
    ("not_butterfly_detected", "not_butterfly"),
)
VISUAL_ADULT_BUTTERFLY_LABELS = {
    "a photo of an adult butterfly",
    "adult butterfly",
    "a photo of a butterfly",
    "a photo of a swallowtail butterfly",
}


def classify_bioclip_triage(*, record: dict[str, Any], prediction: dict[str, object]) -> dict[str, object]:
    species_top1_label = str(prediction.get("species_top1_label", prediction.get("bioclip_top1_label", prediction.get("top1_label", ""))) or "")
    species_top1_score = _optional_float(prediction.get("species_top1_score", prediction.get("bioclip_top1_score", prediction.get("top1_score"))))
    species_margin = _optional_float(prediction.get("species_top1_top2_margin"))
    species_top1_name = str(prediction.get("species_top1_scientific_name") or _species_name_from_label(species_top1_label) or "")
    triage_top1_label = str(prediction.get("triage_top1_label", prediction.get("bioclip_top1_label", prediction.get("top1_label", ""))) or "")
    triage_group_top = str(prediction.get("triage_group_top") or "")
    triage_group_scores = prediction.get("triage_group_scores") if isinstance(prediction.get("triage_group_scores"), dict) else {}
    hard_negative_score = _optional_float(triage_group_scores.get("hard_negative")) if isinstance(triage_group_scores, dict) else None
    negative_reason = _negative_reason(record, triage_top1_label)
    category = _category_for_prediction(top1_label=triage_top1_label, negative_reason=negative_reason)
    text_species_match = _text_species_match(record, species_top1_name)
    is_species_supported = bool(text_species_match and species_top1_score is not None and species_top1_score >= SILVER_SPECIES_CONFIDENCE_THRESHOLD)
    if triage_group_top == "hard_negative" and hard_negative_score is not None and hard_negative_score >= DEFAULT_BUCKET_POLICY.hard_negative_threshold:
        return _bucket_result(category_defaults(), bucket="bin", reason="hard_negative_group", text_species_match=text_species_match, is_target_positive=False, is_negative_material=True)
    if (
        species_top1_score is not None
        and species_top1_score >= GOLD_SPECIES_CONFIDENCE_THRESHOLD
        and species_margin is not None
        and species_margin < DEFAULT_BUCKET_POLICY.ambiguous_margin_threshold
    ):
        return _bucket_result(category_defaults(), bucket="in_review", reason="ambiguous_species_margin", text_species_match=text_species_match, is_target_positive=False, is_negative_material=False)
    if category["image_category"] in BIN_CATEGORIES:
        reason = str(category["negative_filter_reason"] or negative_reason or category["image_category"])
        return _bucket_result(category, bucket="bin", reason=reason, text_species_match=text_species_match, is_target_positive=False, is_negative_material=True)
    if species_top1_score is None:
        return _bucket_result(category, bucket="in_review", reason="missing_bioclip", text_species_match=text_species_match, is_target_positive=False, is_negative_material=False)
    if category["image_category"] == "life_stage_non_adult":
        return _bucket_result(category, bucket="bronze", reason=str(category["life_stage"]), text_species_match=text_species_match, is_target_positive=is_species_supported, is_negative_material=False)
    if _taxonomy_inconsistent(prediction):
        return _bucket_result(category, bucket="in_review", reason="taxonomy_inconsistent", text_species_match=text_species_match, is_target_positive=False, is_negative_material=False)
    if category["image_category"] == "unknown" and is_species_supported:
        return _bucket_result(
            category,
            bucket="in_review",
            reason="missing_visual_butterfly_evidence",
            text_species_match=text_species_match,
            is_target_positive=is_species_supported,
            is_negative_material=False,
        )
    if category["image_category"] == "adult_butterfly" and text_species_match:
        if not _has_image_url(record):
            return _bucket_result(category, bucket="in_review", reason="missing_image_url", text_species_match=text_species_match, is_target_positive=is_species_supported, is_negative_material=False)
        if species_top1_score > GOLD_SPECIES_CONFIDENCE_THRESHOLD:
            if not _has_geo(record):
                return _bucket_result(category, bucket="silver", reason="missing_geo", text_species_match=True, is_target_positive=True, is_negative_material=False)
            if not _has_event_date(record):
                return _bucket_result(category, bucket="silver", reason="missing_event_date", text_species_match=True, is_target_positive=True, is_negative_material=False)
            return _bucket_result(category, bucket="gold", reason="adult_butterfly_species_match_score_gt_070", text_species_match=True, is_target_positive=True, is_negative_material=False)
        if species_top1_score >= SILVER_SPECIES_CONFIDENCE_THRESHOLD:
            return _bucket_result(category, bucket="silver", reason="species_match_score_035_to_070", text_species_match=True, is_target_positive=True, is_negative_material=False)
    return _bucket_result(category, bucket="bronze", reason="below_50", text_species_match=text_species_match, is_target_positive=False, is_negative_material=False)


def _bucket_result(
    category: dict[str, str | None],
    *,
    bucket: str,
    reason: str,
    text_species_match: bool,
    is_target_positive: bool,
    is_negative_material: bool,
) -> dict[str, object]:
    return {
        **category,
        "occurrence_bin": bucket,
        "bin_reason": reason,
        "triage_bin": bucket,
        "triage_reason": reason,
        "text_species_match": text_species_match,
        "is_butterfly_life_stage": category["image_category"] in {"adult_butterfly", "life_stage_non_adult"},
        "is_target_positive": is_target_positive,
        "is_negative_material": is_negative_material,
    }


def _base_row(
    record: dict[str, Any],
    *,
    source: str,
    model_id: str,
    model_version: str,
    model_checkpoint: str,
    classified_at: str,
) -> dict[str, object]:
    image_url = record.get("image_url")
    date_taken = record.get("date_taken") or record.get("datetaken")
    date_upload = record.get("date_upload") or record.get("dateupload")
    year, month = _year_month(record, date_taken=date_taken, date_upload=date_upload)
    return {
        "source": source,
        "source_record_id": str(record.get("source_record_id") or record.get("flickr_photo_id") or ""),
        "flickr_photo_id": str(record.get("flickr_photo_id") or ""),
        "photo_page_url": record.get("photo_page_url"),
        "image_url": str(image_url) if image_url else None,
        "image_url_kind": record.get("image_url_kind"),
        "latitude": _optional_float(record.get("latitude", record.get("decimalLatitude"))),
        "longitude": _optional_float(record.get("longitude", record.get("decimalLongitude"))),
        "date_taken": date_taken,
        "date_upload": date_upload,
        "captured_at": record.get("captured_at") or date_taken,
        "year": year,
        "month": month,
        "source_record_hash": _source_record_hash(record),
        "image_hash": None,
        "image_downloaded": False,
        "image_deleted_after_classification": False,
        "model_id": model_id,
        "model_version": model_version,
        "model_checkpoint": model_checkpoint,
        "classified_at": classified_at,
    }


def _prediction_fields(prediction: dict[str, object]) -> dict[str, object]:
    topk = prediction.get("bioclip_topk_json", prediction.get("topk_json", []))
    return {
        "bioclip_top1_label": prediction.get("bioclip_top1_label", prediction.get("top1_label")),
        "bioclip_top1_score": _optional_float(prediction.get("bioclip_top1_score", prediction.get("top1_score"))),
        "bioclip_topk_json": topk if isinstance(topk, list) else [],
        "species_top1_label": prediction.get("species_top1_label"),
        "species_top1_scientific_name": prediction.get("species_top1_scientific_name"),
        "species_top1_genus": prediction.get("species_top1_genus"),
        "species_top1_family": prediction.get("species_top1_family"),
        "species_top1_score": _optional_float(prediction.get("species_top1_score")),
        "species_topk_json": prediction.get("species_topk_json", []),
        "triage_top1_label": prediction.get("triage_top1_label"),
        "triage_top1_score": _optional_float(prediction.get("triage_top1_score")),
        "triage_topk_json": prediction.get("triage_topk_json", []),
    }


def _empty_result_fields() -> dict[str, object]:
    return {
        "image_hash": None,
        "image_downloaded": False,
        "image_deleted_after_classification": False,
        "classification_error": None,
        "retry_eligible": False,
        "bioclip_top1_label": None,
        "bioclip_top1_score": None,
        "bioclip_topk_json": [],
        "species_top1_label": None,
        "species_top1_scientific_name": None,
        "species_top1_genus": None,
        "species_top1_family": None,
        "species_top1_score": None,
        "species_topk_json": [],
        "triage_top1_label": None,
        "triage_top1_score": None,
        "triage_topk_json": [],
        "occurrence_bin": None,
        "bin_reason": None,
        **category_defaults(),
        "triage_reason": None,
        "text_species_match": False,
        "is_butterfly_life_stage": False,
        "is_target_positive": False,
        "is_negative_material": False,
    }


def _failure_row(
    base: dict[str, object],
    *,
    status: str,
    error: str,
    retry_eligible: bool = True,
    image_hash: str | None = None,
    image_downloaded: bool = False,
    image_deleted_after_classification: bool = False,
) -> dict[str, object]:
    return {
        **base,
        **_empty_result_fields(),
        "image_hash": image_hash,
        "image_downloaded": image_downloaded,
        "image_deleted_after_classification": image_deleted_after_classification,
        "classification_status": status,
        "classification_error": error,
        "retry_eligible": retry_eligible,
        "occurrence_bin": "in_review",
        "bin_reason": status,
        "triage_bin": "in_review",
        "triage_reason": status,
        "text_species_match": False,
        "is_butterfly_life_stage": False,
    }


def _category_for_prediction(*, top1_label: str, negative_reason: str | None) -> dict[str, str | None]:
    if negative_reason:
        return category_from_negative_reason(negative_reason)
    if _is_visual_adult_butterfly_label(top1_label):
        return {
            "image_category": "adult_butterfly",
            "life_stage": "adult_butterfly",
            "negative_filter_reason": None,
        }
    return category_defaults()


def _negative_reason(record: dict[str, Any], top1_label: str) -> str | None:
    normalized = _normalize(top1_label)
    if normalized in {_normalize(label) for label in NEGATIVE_LABEL_REASONS}:
        return NEGATIVE_LABEL_REASONS[next(label for label in NEGATIVE_LABEL_REASONS if _normalize(label) == normalized)]
    if "museum" in normalized and "specimen" in normalized:
        return "museum_specimen"
    if "pinned" in normalized or "specimen" in normalized:
        return "pinned_specimen"
    if "artwork" in normalized or "illustration" in normalized:
        return "artwork"
    if "tattoo" in normalized:
        return "tattoo"
    if "ai generated" in normalized or "ai-generated" in normalized:
        return "AI_generated"
    if "moth" in normalized:
        return "moth"
    if "not butterfly" in normalized or "non-butterfly" in normalized:
        return "not_butterfly"
    life_stage = infer_life_stage_from_text(top1_label)
    if life_stage != DEFAULT_LIFE_STAGE:
        return life_stage
    return None


def _has_image_url(record: dict[str, Any]) -> bool:
    return bool(record.get("image_url") or record.get("image_url_used"))


def _taxonomy_inconsistent(prediction: dict[str, object]) -> bool:
    species_genus = str(prediction.get("species_top1_genus") or "")
    genus_top1 = str(prediction.get("genus_top1") or "")
    species_family = str(prediction.get("species_top1_family") or "")
    family_top1 = str(prediction.get("family_top1") or "")
    if species_genus and genus_top1 and species_genus != genus_top1:
        return True
    return bool(species_family and family_top1 and species_family != family_top1)


def _has_event_date(record: dict[str, Any]) -> bool:
    return bool(record.get("captured_at") or record.get("date_taken") or record.get("datetaken") or record.get("eventDate"))


def _has_geo(record: dict[str, Any]) -> bool:
    latitude = record.get("latitude", record.get("decimalLatitude"))
    longitude = record.get("longitude", record.get("decimalLongitude"))
    return latitude not in (None, "") and longitude not in (None, "")


def _text_species_match(record: dict[str, Any], species_name: str) -> bool:
    if bool(record.get("text_species_match") or record.get("species_text_match")):
        return True
    return species_text_matches(
        species_name,
        (
            record.get("raw_title"),
            record.get("title"),
            record.get("raw_description"),
            record.get("description"),
            record.get("raw_tags"),
            record.get("tags"),
            record.get("machine_tags"),
        ),
    )


def _successful_keys(frame: pl.DataFrame) -> set[tuple[object, ...]]:
    if frame.is_empty() or "classification_status" not in frame.columns:
        return set()
    return {
        _dedupe_key(row)
        for row in frame.filter(pl.col("classification_status") == "success").to_dicts()
    }


def _dedupe_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row.get("source"),
        row.get("flickr_photo_id"),
        row.get("image_url"),
        row.get("model_id"),
        row.get("model_version"),
        row.get("model_checkpoint"),
    )


def _read_existing(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else _empty_triage_frame()


def _source_record_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _year_month(record: dict[str, Any], *, date_taken: object, date_upload: object) -> tuple[int | None, int | None]:
    year = _optional_int(record.get("year"))
    month = _optional_int(record.get("month"))
    if year is not None and month is not None:
        return year, month
    date_value = str(date_taken or date_upload or "")
    if len(date_value) >= 7 and date_value[0:4].isdigit() and date_value[5:7].isdigit():
        return int(date_value[0:4]), int(date_value[5:7])
    return year, month


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _timestamp(now: datetime | None) -> str:
    value = now or datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _is_visual_adult_butterfly_label(label: str) -> bool:
    normalized = _normalize(label)
    return normalized in VISUAL_ADULT_BUTTERFLY_LABELS


def _species_name_from_label(label: str) -> str | None:
    normalized = label.strip()
    prefix = "a photo of "
    if normalized.casefold().startswith(prefix):
        return normalized[len(prefix):].strip()
    return None


def _empty_triage_frame() -> pl.DataFrame:
    return pl.DataFrame()
