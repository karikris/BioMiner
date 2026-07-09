from __future__ import annotations

import json
from typing import Any, Mapping

import polars as pl

from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION


LOW_MARGIN_THRESHOLD = 0.05
HIGH_DETECTOR_SCORE_THRESHOLD = 0.80
WEAK_SPECIES_SCORE_THRESHOLD = 0.35

HIERARCHICAL_REVIEW_QUEUE_SCHEMA: dict[str, pl.DataType] = {
    "source": pl.String,
    "flickr_photo_id": pl.String,
    "detection_id": pl.String,
    "crop_hash": pl.String,
    "photo_page_url": pl.String,
    "image_url": pl.String,
    "classification_mode": pl.String,
    "review_priority": pl.Int64,
    "review_reason": pl.String,
    "selected_family": pl.String,
    "family_top3": pl.List(pl.String),
    "species_top5": pl.List(pl.String),
    "species_top1_scientific_name": pl.String,
    "species_top1_accepted_taxon_key": pl.String,
    "species_top1_score": pl.Float64,
    "species_top1_margin": pl.Float64,
    "detector_score": pl.Float64,
    "occurrence_bin": pl.String,
    "bin_reason": pl.String,
}


def build_hierarchical_review_queue(
    *,
    object_evidence: pl.DataFrame,
    photo_summary: pl.DataFrame | None = None,
    max_rows: int | None = None,
) -> pl.DataFrame:
    """Return object-level review priorities for hierarchical classifier output."""

    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows must be non-negative")
    if object_evidence.is_empty():
        return pl.DataFrame(schema=HIERARCHICAL_REVIEW_QUEUE_SCHEMA)

    photo_rows = _photo_rows_by_key(photo_summary)
    distinct_photo_species = _distinct_species_by_photo(object_evidence)
    rows: list[dict[str, Any]] = []
    for object_row in object_evidence.to_dicts():
        photo_key = (_text(object_row.get("source")), _text(object_row.get("flickr_photo_id")))
        photo_row = photo_rows.get(photo_key, {})
        row = _merge_photo_object_row(photo_row, object_row)
        if not _reviewable_hierarchical_row(row):
            continue
        priority, reasons = _review_priority(row, distinct_photo_species.get(photo_key, set()))
        rows.append(
            {
                "source": _text(row.get("source")),
                "flickr_photo_id": _text(row.get("flickr_photo_id")),
                "detection_id": _text(row.get("detection_id")),
                "crop_hash": _text(row.get("crop_hash")),
                "photo_page_url": _first_text(row, "photo_page_url", "flickr_url"),
                "image_url": _first_text(row, "image_url", "image_url_used"),
                "classification_mode": _text(row.get("classification_mode")),
                "review_priority": priority,
                "review_reason": ";".join(reasons),
                "selected_family": _first_text(row, "selected_family", "family_top1", "family"),
                "family_top3": _taxon_name_list(row.get("family_top3")),
                "species_top5": _taxon_name_list(row.get("species_top5")),
                "species_top1_scientific_name": _first_text(row, "species_top1_scientific_name", "species_top1"),
                "species_top1_accepted_taxon_key": _first_text(
                    row,
                    "species_top1_accepted_taxon_key",
                    "accepted_taxon_key",
                ),
                "species_top1_score": _optional_float(row.get("species_top1_score")),
                "species_top1_margin": _species_margin(row),
                "detector_score": _optional_float(row.get("detector_score")),
                "occurrence_bin": _first_text(row, "occurrence_bin", "photo_occurrence_bin", "triage_bin"),
                "bin_reason": _first_text(row, "bin_reason", "photo_bin_reason", "triage_reason"),
            }
        )

    if not rows:
        return pl.DataFrame(schema=HIERARCHICAL_REVIEW_QUEUE_SCHEMA)
    frame = pl.DataFrame(rows, schema=HIERARCHICAL_REVIEW_QUEUE_SCHEMA)
    frame = frame.sort(
        ["review_priority", "source", "flickr_photo_id", "detection_id", "crop_hash"],
        descending=[True, False, False, False, False],
    )
    return frame.head(max_rows) if max_rows is not None else frame


def _reviewable_hierarchical_row(row: Mapping[str, Any]) -> bool:
    mode = _text(row.get("classification_mode"))
    if mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION:
        return True
    return _is_butterfly_like_detection(row) and _missing_bioclip_score(row)


def _review_priority(row: Mapping[str, Any], photo_species: set[str]) -> tuple[int, list[str]]:
    reasons: list[str] = []
    priority = 10

    def add(reason: str, value: int) -> None:
        nonlocal priority
        if reason not in reasons:
            reasons.append(reason)
        priority = max(priority, value)

    if _missing_bioclip_score(row) and _is_butterfly_like_detection(row):
        add("missing_bioclip_score", 100)
    if _hard_negative_metadata_conflict(row):
        add("hard_negative_metadata_conflict", 95)
    if _metadata_species_conflict(row):
        add("metadata_species_conflict", 90)
    if _family_species_conflict(row):
        add("family_species_conflict", 85)
    if _species_name(row) and len(photo_species) > 1:
        add("multiple_detection_species_conflict", 80)
    if _high_detector_weak_species(row):
        add("high_detector_weak_species_score", 75)
    species_margin = _species_margin(row)
    if species_margin is not None and species_margin <= LOW_MARGIN_THRESHOLD:
        add("low_species_margin", 70)
    family_margin = _family_margin(row)
    if family_margin is not None and family_margin <= LOW_MARGIN_THRESHOLD:
        add("low_family_margin", 65)
    if _geo_prior_conflict(row):
        add("geospatial_prior_conflict", 60)

    if not reasons:
        reasons.append("clean_confident_prediction")
    return priority, reasons


def _photo_rows_by_key(photo_summary: pl.DataFrame | None) -> dict[tuple[str, str], dict[str, Any]]:
    if photo_summary is None or photo_summary.is_empty():
        return {}
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in photo_summary.to_dicts():
        key = (_text(row.get("source")), _text(row.get("flickr_photo_id")))
        if key != ("", ""):
            output.setdefault(key, row)
    return output


def _merge_photo_object_row(photo_row: Mapping[str, Any], object_row: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(photo_row)
    for key, value in object_row.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        merged[key] = value
    return merged


def _distinct_species_by_photo(object_evidence: pl.DataFrame) -> dict[tuple[str, str], set[str]]:
    output: dict[tuple[str, str], set[str]] = {}
    for row in object_evidence.to_dicts():
        species = _species_name(row)
        if not species:
            continue
        key = (_text(row.get("source")), _text(row.get("flickr_photo_id")))
        output.setdefault(key, set()).add(species.casefold())
    return output


def _missing_bioclip_score(row: Mapping[str, Any]) -> bool:
    return _optional_float(row.get("species_top1_score")) is None and not _species_name(row)


def _is_butterfly_like_detection(row: Mapping[str, Any]) -> bool:
    detector_label = _text(row.get("detector_label")).casefold()
    detection_class = _text(row.get("detection_class")).casefold()
    category = _text(row.get("image_category")).casefold()
    status = _text(row.get("detection_status")).casefold()
    if row.get("butterfly_like_detection") is True or row.get("is_butterfly_like") is True:
        return True
    if detector_label in {"butterfly", "butterfly_like", "adult_butterfly"}:
        return True
    if detection_class in {"butterfly", "butterfly_like", "adult_butterfly"}:
        return True
    if category in {"adult_butterfly", "egg", "caterpillar", "larva", "pupa", "chrysalis"}:
        return True
    return status in {"detected", "cropped"} and "butterfly" in detector_label


def _hard_negative_metadata_conflict(row: Mapping[str, Any]) -> bool:
    for field in (
        "is_negative_material",
        "hard_negative",
        "metadata_hard_negative",
        "has_hard_negative",
    ):
        if _truthy(row.get(field)):
            return True
    return bool(
        _first_text(
            row,
            "negative_filter_reason",
            "metadata_negative_reason_hint",
            "hard_negative_reason",
        )
    )


def _metadata_species_conflict(row: Mapping[str, Any]) -> bool:
    if _truthy(row.get("bioclip_tag_conflict")) or _truthy(row.get("metadata_species_conflict")):
        return True
    predicted = _species_name(row).casefold()
    if not predicted:
        return False
    for field in (
        "flickr_text_species_candidate",
        "metadata_species_candidate",
        "text_species_candidate",
        "flickr_tag_species_candidate",
        "comment_species_candidate",
    ):
        candidate = _text(row.get(field)).casefold()
        if candidate and candidate != predicted:
            return True
    return False


def _family_species_conflict(row: Mapping[str, Any]) -> bool:
    selected_key = _first_text(row, "selected_family_key", "family_top1_accepted_taxon_key")
    species_family_key = _first_text(row, "species_candidate_family_key", "species_top1_family_key")
    if selected_key and species_family_key and selected_key != species_family_key:
        return True
    selected_family = _first_text(row, "selected_family", "family_top1", "family").casefold()
    species_family = _first_text(row, "species_candidate_family", "species_top1_family").casefold()
    return bool(selected_family and species_family and selected_family != species_family)


def _high_detector_weak_species(row: Mapping[str, Any]) -> bool:
    detector_score = _optional_float(row.get("detector_score"))
    species_score = _optional_float(row.get("species_top1_score"))
    return (
        detector_score is not None
        and species_score is not None
        and detector_score >= HIGH_DETECTOR_SCORE_THRESHOLD
        and species_score < WEAK_SPECIES_SCORE_THRESHOLD
    )


def _geo_prior_conflict(row: Mapping[str, Any]) -> bool:
    if _truthy(row.get("geo_prior_conflict")) or _truthy(row.get("outside_geospatial_prior")):
        return True
    status = _text(row.get("geospatial_prior_status")).casefold()
    if status in {"outside", "out_of_range", "conflict", "failed"}:
        return True
    reason = _text(row.get("geospatial_prior_reason")).casefold()
    return any(token in reason for token in ("outside", "out_of_range", "conflict"))


def _species_margin(row: Mapping[str, Any]) -> float | None:
    explicit = _optional_float(row.get("species_top1_margin"))
    if explicit is not None:
        return explicit
    explicit = _optional_float(row.get("species_top1_top2_margin"))
    if explicit is not None:
        return explicit
    scores = _float_list(row.get("species_top5_scores"))
    if len(scores) < 2:
        return None
    return float(scores[0] - scores[1])


def _family_margin(row: Mapping[str, Any]) -> float | None:
    explicit = _optional_float(row.get("family_margin"))
    if explicit is not None:
        return explicit
    scores = _float_list(row.get("family_top3_scores"))
    if len(scores) < 2:
        return None
    return float(scores[0] - scores[1])


def _species_name(row: Mapping[str, Any]) -> str:
    return _first_text(row, "species_top1_scientific_name", "species_top1")


def _first_text(row: Mapping[str, Any], *columns: str) -> str:
    for column in columns:
        text = _text(row.get(column))
        if text:
            return text
    return ""


def _taxon_name_list(value: object) -> list[str]:
    values = _list_like(value)
    output: list[str] = []
    seen: set[str] = set()
    for item in values:
        if isinstance(item, Mapping):
            text = _first_text(item, "scientific_name", "name", "label")
        else:
            text = _text(item)
        if text and text not in seen:
            output.append(text)
            seen.add(text)
    return output


def _float_list(value: object) -> list[float]:
    return [number for item in _list_like(value) if (number := _optional_float(item)) is not None]


def _list_like(value: object) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, pl.Series):
        return value.to_list()
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return parsed
    return [value]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    text = str(value).strip().casefold()
    return text in {"1", "true", "yes", "y"}


def _text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


__all__ = [
    "HIERARCHICAL_REVIEW_QUEUE_SCHEMA",
    "build_hierarchical_review_queue",
]
