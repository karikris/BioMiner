from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import hashlib
from pathlib import Path
from typing import Any, Callable, Protocol

import polars as pl

from flickr_bio_occurrence.evidence.category_model import category_defaults, category_from_negative_reason, infer_life_stage_from_text
from flickr_bio_occurrence.vision.image_cache import CachedImage, cache_image_from_url
from flickr_bio_occurrence.vision.temp_image_store import cleanup_cached_image


TRIAGE_BINS = {"gold", "silver", "bronze", "in_review"}
CLASSIFICATION_STATUSES = {"success", "skipped_existing", "failed_download", "failed_bioclip", "invalid_record"}
TARGET_LABELS = {
    "a photo of Papilio demoleus",
    "a photo of lime butterfly",
    "a photo of chequered swallowtail",
    "a photo of citrus swallowtail",
    "a photo of a swallowtail butterfly",
    "a photo of a butterfly",
}
VERIFIED_LIFE_STAGE_LABELS = {
    "a photo of a caterpillar",
    "a photo of a pupa or chrysalis",
}
NEGATIVE_LABEL_REASONS = {
    "a photo of a pinned museum specimen": "pinned_specimen",
    "a photo of artwork or illustration": "artwork",
    "a photo of a moth": "moth",
    "a photo of a caterpillar": "caterpillar",
    "a photo of a pupa or chrysalis": "pupa_or_chrysalis",
    "a photo of an insect": "other_insect",
    "a photo of a beetle": "other_insect",
    "a photo of a fly": "other_insect",
    "a photo of a wasp": "other_insect",
    "a photo of a non-butterfly": "not_butterfly",
    "a photo of a background": "object_background_non_organism",
    "a photo of an object": "object_background_non_organism",
    "a blank image": "object_background_non_organism",
    "an ai generated image": "AI_generated",
}
NEGATIVE_RECORD_FIELDS = (
    ("museum_detected", "museum_specimen"),
    ("specimen_detected", "pinned_specimen"),
    ("artwork_detected", "artwork"),
    ("ai_generated_detected", "AI_generated"),
    ("other_insect_detected", "other_insect"),
    ("non_target_order_detected", "other_order"),
    ("not_butterfly_detected", "not_butterfly"),
)


class ImageClassifier(Protocol):
    def classify_image(self, **kwargs: object) -> dict[str, object]:
        ...


CacheImage = Callable[..., CachedImage]


@dataclass(frozen=True)
class ImageTriageRun:
    frame: pl.DataFrame
    output_path: Path
    records_seen: int
    records_classified: int
    records_skipped_existing: int
    download_failures: int
    bioclip_failures: int
    images_deleted_after_classification: int


def process_image_triage_records(
    records: Iterable[dict[str, Any]],
    *,
    classifier: ImageClassifier,
    output_path: str | Path,
    cache_root: str | Path = "data/cache/images",
    cache_image: CacheImage = cache_image_from_url,
    source: str = "flickr",
    model_id: str = "bioclip2_5",
    model_version: str = "bioclip2_5_huge",
    model_checkpoint: str = "unknown",
    now: datetime | None = None,
) -> ImageTriageRun:
    output = Path(output_path)
    existing = _read_existing(output)
    processed_keys = _successful_keys(existing)
    classified_at = _timestamp(now)
    rows: list[dict[str, Any]] = []
    records_seen = 0
    classified = 0
    skipped = 0
    download_failures = 0
    bioclip_failures = 0
    deleted = 0

    for record in records:
        records_seen += 1
        base = _base_row(
            record,
            source=source,
            model_id=model_id,
            model_version=model_version,
            model_checkpoint=model_checkpoint,
            classified_at=classified_at,
        )
        if not base["flickr_photo_id"] or not base["image_url"]:
            rows.append(_failure_row(base, status="invalid_record", error="missing image URL or source record ID"))
            continue
        dedupe_key = _dedupe_key(base)
        if dedupe_key in processed_keys:
            skipped += 1
            rows.append({**base, **_empty_result_fields(), "classification_status": "skipped_existing", "occurrence_bin": "in_review", "bin_reason": "duplicate_successful_record", "triage_bin": "in_review", "triage_reason": "duplicate_successful_record"})
            continue
        try:
            cached = cache_image(str(base["image_url"]), cache_root=cache_root)
        except Exception as exc:  # noqa: BLE001 - failures are recorded for retry.
            download_failures += 1
            rows.append(_failure_row(base, status="failed_download", error=str(exc), retry_eligible=True))
            continue

        image_deleted = False
        try:
            prediction = classifier.classify_image(
                flickr_photo_id=str(base["flickr_photo_id"]),
                image_path=cached.path,
                image_hash=cached.image_hash,
                image_url_used=cached.source_url,
            )
            triage = classify_bioclip_triage(record={**record, **base}, prediction=prediction)
            rows.append(
                {
                    **base,
                    "image_hash": cached.image_hash,
                    "image_downloaded": True,
                    "classification_status": "success",
                    "classification_error": None,
                    "retry_eligible": False,
                    **_prediction_fields(prediction),
                    **triage,
                    "image_deleted_after_classification": cleanup_cached_image(cached, cache_root=cache_root, delete_after_success=True),
                }
            )
            image_deleted = bool(rows[-1]["image_deleted_after_classification"])
            classified += 1
            processed_keys.add(dedupe_key)
        except Exception as exc:  # noqa: BLE001 - model failures are recorded for retry.
            bioclip_failures += 1
            image_deleted = cleanup_cached_image(cached, cache_root=cache_root, delete_after_success=True)
            rows.append(
                _failure_row(
                    base,
                    status="failed_bioclip",
                    error=str(exc),
                    retry_eligible=True,
                    image_hash=cached.image_hash,
                    image_downloaded=True,
                    image_deleted_after_classification=image_deleted,
                )
            )
        if image_deleted:
            deleted += 1

    new_frame = pl.DataFrame(rows) if rows else _empty_triage_frame()
    combined = pl.concat([existing, new_frame], how="diagonal_relaxed") if existing.height else new_frame
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output)
    return ImageTriageRun(
        frame=combined,
        output_path=output,
        records_seen=records_seen,
        records_classified=classified,
        records_skipped_existing=skipped,
        download_failures=download_failures,
        bioclip_failures=bioclip_failures,
        images_deleted_after_classification=deleted,
    )


def classify_bioclip_triage(*, record: dict[str, Any], prediction: dict[str, object]) -> dict[str, object]:
    top1_label = str(prediction.get("bioclip_top1_label", prediction.get("top1_label", "")) or "")
    top1_score = _optional_float(prediction.get("bioclip_top1_score", prediction.get("top1_score")))
    topk = prediction.get("bioclip_topk_json", prediction.get("topk_json", [])) or []
    labels = {_normalize(top1_label)}
    if isinstance(topk, list):
        labels.update(_normalize(str(item.get("label") or "")) for item in topk if isinstance(item, dict))
    verified_life_stage = bool(record.get("human_verification_detected") and record.get("species_text_match") and labels & {_normalize(label) for label in VERIFIED_LIFE_STAGE_LABELS})
    is_target_positive = bool(labels & {_normalize(label) for label in TARGET_LABELS}) or verified_life_stage
    negative_reason = _negative_reason(record, top1_label, ignore_life_stage=verified_life_stage)
    category = _category_for_prediction(top1_label=top1_label, negative_reason=negative_reason, verified_life_stage=verified_life_stage)
    if negative_reason:
        return {**category, "occurrence_bin": "bronze", "bin_reason": negative_reason, "triage_bin": "bronze", "triage_reason": negative_reason, "is_target_positive": is_target_positive, "is_negative_material": True}
    if top1_score is None:
        return {**category, "occurrence_bin": "in_review", "bin_reason": "missing_bioclip", "triage_bin": "in_review", "triage_reason": "missing_bioclip", "is_target_positive": False, "is_negative_material": False}
    if is_target_positive and top1_score >= 0.50:
        return {**category, "occurrence_bin": "gold", "bin_reason": "target_positive_score_gte_050", "triage_bin": "gold", "triage_reason": "target_positive_score_gte_050", "is_target_positive": True, "is_negative_material": False}
    if is_target_positive and top1_score < 0.50:
        return {**category, "occurrence_bin": "silver", "bin_reason": "target_positive_score_lt_050", "triage_bin": "silver", "triage_reason": "target_positive_score_lt_050", "is_target_positive": True, "is_negative_material": False}
    return {**category, "occurrence_bin": "in_review", "bin_reason": "ambiguous_classification", "triage_bin": "in_review", "triage_reason": "ambiguous_classification", "is_target_positive": False, "is_negative_material": False}


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
        "occurrence_bin": None,
        "bin_reason": None,
        **category_defaults(),
        "triage_reason": None,
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
    }


def _category_for_prediction(*, top1_label: str, negative_reason: str | None, verified_life_stage: bool) -> dict[str, str | None]:
    if negative_reason:
        return category_from_negative_reason(negative_reason)
    if verified_life_stage:
        life_stage = infer_life_stage_from_text(top1_label)
        return {
            "image_category": "life_stage_non_adult",
            "life_stage": life_stage,
            "negative_filter_reason": None,
        }
    return category_defaults()


def _negative_reason(record: dict[str, Any], top1_label: str, *, ignore_life_stage: bool) -> str | None:
    for field, reason in NEGATIVE_RECORD_FIELDS:
        if bool(record.get(field)):
            return reason
    normalized = _normalize(top1_label)
    if ignore_life_stage and normalized in {_normalize(label) for label in VERIFIED_LIFE_STAGE_LABELS}:
        return None
    if normalized in {_normalize(label) for label in NEGATIVE_LABEL_REASONS}:
        return NEGATIVE_LABEL_REASONS[next(label for label in NEGATIVE_LABEL_REASONS if _normalize(label) == normalized)]
    if "museum" in normalized and "specimen" in normalized:
        return "museum_specimen"
    if "pinned" in normalized or "specimen" in normalized:
        return "pinned_specimen"
    if "artwork" in normalized or "illustration" in normalized:
        return "artwork"
    if "ai generated" in normalized or "ai-generated" in normalized:
        return "AI_generated"
    if "moth" in normalized:
        return "moth"
    if "not butterfly" in normalized or "non-butterfly" in normalized:
        return "not_butterfly"
    return None


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


def _empty_triage_frame() -> pl.DataFrame:
    return pl.DataFrame()
