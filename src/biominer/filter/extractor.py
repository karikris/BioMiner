from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl

from biominer.filter.category_model import infer_category_from_text
from biominer.filter.review_flags import (
    ARTWORK_TERMS,
    CAPTIVE_TERMS,
    COLLECTION_TERMS,
    HUMAN_VERIFICATION_TERMS,
    MUSEUM_TERMS,
    NON_TARGET_ORDER_TERMS,
    SPECIMEN_TERMS,
    TATTOO_TERMS,
    contains_any_term,
    detected_terms,
)


EVIDENCE_OUTPUT_PATH = Path("staging/evidence/staging_evidence.parquet")
SCIENTIFIC_NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+ [a-z][a-z-]+\b")


def build_evidence_frame(
    payloads: Iterable[dict[str, Any]],
    *,
    species_query: str,
) -> pl.DataFrame:
    rows = extract_evidence_rows(payloads, species_query=species_query)
    empty = _empty_evidence_frame()
    return pl.DataFrame(rows, schema=empty.schema) if rows else empty


def extract_evidence_rows(
    payloads: Iterable[dict[str, Any]],
    *,
    species_query: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        for photo in _photo_rows(payload):
            rows.append(extract_photo_evidence(photo, species_query=species_query))
    return rows


def write_staging_evidence(
    payloads: Iterable[dict[str, Any]],
    *,
    species_query: str,
    output_path: str | Path = EVIDENCE_OUTPUT_PATH,
) -> Path:
    frame = build_evidence_frame(payloads, species_query=species_query)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return path


def extract_photo_evidence(photo: dict[str, Any], *, species_query: str) -> dict[str, Any]:
    comments = _comments_text_and_count(photo)
    raw_description = _description_text(photo.get("description"))
    text_sources = {
        "title": _optional_string(photo.get("title")),
        "description": raw_description,
        "tags": _optional_string(photo.get("tags")),
        "machine_tags": _optional_string(photo.get("machine_tags")),
        "comments": comments["comments_text"],
    }
    combined_text = " ".join(value for value in text_sources.values() if value)
    species_sources = _species_text_sources(text_sources, species_query)
    verification_sources = _verification_sources(text_sources)
    verification_terms = sorted({term for terms in verification_sources.values() for term in terms})
    scientific_names = sorted(set(SCIENTIFIC_NAME_PATTERN.findall(combined_text)))
    image_url, image_url_kind = _preferred_flickr_image_url(photo)
    museum_detected = contains_any_term(combined_text, MUSEUM_TERMS)
    artwork_detected = contains_any_term(combined_text, ARTWORK_TERMS)
    specimen_detected = contains_any_term(combined_text, SPECIMEN_TERMS)
    tattoo_detected = contains_any_term(combined_text, TATTOO_TERMS)
    collection_detected = contains_any_term(combined_text, COLLECTION_TERMS)
    captive_detected = contains_any_term(combined_text, CAPTIVE_TERMS)
    non_target_order_detected = contains_any_term(combined_text, NON_TARGET_ORDER_TERMS)
    museum_category_detected = specimen_detected or _contains_museum_specimen_context(combined_text)
    category = infer_category_from_text(
        combined_text,
        museum_detected=museum_category_detected,
        specimen_detected=specimen_detected,
        artwork_detected=artwork_detected,
        tattoo_detected=tattoo_detected,
        non_target_order_detected=non_target_order_detected,
    )
    review_flags = _review_flags(
        image_url=image_url,
        species_text_match=bool(species_sources),
        human_verification_detected=bool(verification_terms),
        museum_detected=museum_detected,
        artwork_detected=artwork_detected,
        specimen_detected=specimen_detected,
        tattoo_detected=tattoo_detected,
        collection_detected=collection_detected,
        captive_detected=captive_detected,
        non_target_order_detected=non_target_order_detected,
    )

    return {
        "source": str(photo.get("source") or "flickr"),
        "source_record_hash": str(photo.get("source_record_hash") or ""),
        "flickr_photo_id": str(photo.get("id", "")),
        "owner_id": photo.get("owner"),
        "owner_name": photo.get("ownername"),
        "photo_page_url": f"https://www.flickr.com/photos/{photo.get('owner', '')}/{photo.get('id', '')}",
        "image_url": image_url,
        "image_url_kind": image_url_kind,
        "date_taken": photo.get("datetaken") or photo.get("datetakenunknown"),
        "date_upload": photo.get("dateupload"),
        "latitude": _optional_float(photo.get("latitude")),
        "longitude": _optional_float(photo.get("longitude")),
        "accuracy": _optional_int(photo.get("accuracy")),
        "license": photo.get("license"),
        "raw_title": text_sources["title"],
        "raw_description": text_sources["description"],
        "raw_tags": text_sources["tags"],
        "machine_tags": text_sources["machine_tags"],
        "comments_text": comments["comments_text"],
        "comments_count": comments["comments_count"],
        "species_query": species_query,
        "first_query_field": None,
        "first_query_term": None,
        "first_query_language": None,
        "text_search_terms": [],
        "tag_search_terms": [],
        "all_query_labels": [],
        "all_query_terms": [],
        "all_query_fields": [],
        "query_hit_count": 0,
        "duplicate_query_hit_count": 0,
        "query_definition_ids": [],
        "discovery_accepted_taxon_keys": [],
        "discovery_family_keys": [],
        "discovery_genus_keys": [],
        "discovery_species_keys": [],
        "registry_versions": [],
        "species_text_match": bool(species_sources),
        "species_text_source": species_sources,
        "scientific_names_detected": scientific_names,
        "human_verification_detected": bool(verification_terms),
        "human_verification_source": sorted(verification_sources),
        "human_verification_terms": verification_terms,
        "human_verification_confidence": 1.0 if verification_terms else 0.0,
        "museum_detected": museum_detected,
        "artwork_detected": artwork_detected,
        "specimen_detected": specimen_detected,
        "tattoo_detected": tattoo_detected,
        "collection_detected": collection_detected,
        "captive_detected": captive_detected,
        "non_target_order_detected": non_target_order_detected,
        "occurrence_bin": "in_review",
        "bin_reason": "unclassified_evidence",
        **category,
        "review_flags": review_flags,
    }


def _photo_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    photos = payload.get("photos")
    if not isinstance(photos, dict):
        return []
    rows = photos.get("photo")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _preferred_flickr_image_url(photo: dict[str, Any]) -> tuple[str | None, str | None]:
    if photo.get("url_l"):
        return str(photo["url_l"]), "url_l"
    if photo.get("url_m"):
        return str(photo["url_m"]), "url_m"
    return None, None


def _description_text(value: Any) -> str | None:
    if isinstance(value, dict):
        return _optional_string(value.get("_content"))
    return _optional_string(value)


def _comments_text_and_count(photo: dict[str, Any]) -> dict[str, Any]:
    comments = photo.get("comments")
    if comments is None:
        return {"comments_text": None, "comments_count": 0}
    if isinstance(comments, str):
        return {"comments_text": comments, "comments_count": 1 if comments else 0}
    if isinstance(comments, list):
        values = [_comment_text(comment) for comment in comments]
        return {"comments_text": _join_text(values), "comments_count": len([value for value in values if value])}
    if isinstance(comments, dict):
        nested = comments.get("comment")
        if isinstance(nested, list):
            values = [_comment_text(comment) for comment in nested]
            return {"comments_text": _join_text(values), "comments_count": len([value for value in values if value])}
        value = _comment_text(comments)
        return {"comments_text": value, "comments_count": 1 if value else 0}
    return {"comments_text": _optional_string(comments), "comments_count": 1}


def _comment_text(value: Any) -> str | None:
    if isinstance(value, dict):
        return _optional_string(value.get("_content") or value.get("text") or value.get("body"))
    return _optional_string(value)


def _species_text_sources(text_sources: dict[str, str | None], species_query: str) -> list[str]:
    species = species_query.casefold()
    return [source for source, value in text_sources.items() if value and species in value.casefold()]


def _verification_sources(text_sources: dict[str, str | None]) -> dict[str, list[str]]:
    return {
        source: detected
        for source, value in text_sources.items()
        if value and (detected := detected_terms(value, HUMAN_VERIFICATION_TERMS))
    }


def _review_flags(
    *,
    image_url: str | None,
    species_text_match: bool,
    human_verification_detected: bool,
    museum_detected: bool,
    artwork_detected: bool,
    specimen_detected: bool,
    tattoo_detected: bool,
    collection_detected: bool,
    captive_detected: bool,
    non_target_order_detected: bool,
) -> list[str]:
    flags: list[str] = []
    if not image_url:
        flags.append("missing_image_url")
    if not species_text_match:
        flags.append("no_species_text_match")
    if human_verification_detected:
        flags.append("human_verification_phrase")
    for enabled, flag in (
        (museum_detected, "museum_context"),
        (artwork_detected, "artwork_context"),
        (specimen_detected, "specimen_context"),
        (tattoo_detected, "tattoo_context"),
        (collection_detected, "collection_context"),
        (captive_detected, "captive_context"),
        (non_target_order_detected, "non_target_order_context"),
    ):
        if enabled:
            flags.append(flag)
    return flags


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _join_text(values: Iterable[str | None]) -> str | None:
    text = "\n".join(value for value in values if value)
    return text or None


def _contains_museum_specimen_context(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return any(term in normalized for term in ("museum specimen", "museum collection", "museum plate"))


def _empty_evidence_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "source": pl.String,
            "source_record_hash": pl.String,
            "flickr_photo_id": pl.String,
            "owner_id": pl.String,
            "owner_name": pl.String,
            "photo_page_url": pl.String,
            "image_url": pl.String,
            "image_url_kind": pl.String,
            "date_taken": pl.String,
            "date_upload": pl.String,
            "latitude": pl.Float64,
            "longitude": pl.Float64,
            "accuracy": pl.Int64,
            "license": pl.String,
            "raw_title": pl.String,
            "raw_description": pl.String,
            "raw_tags": pl.String,
            "machine_tags": pl.String,
            "comments_text": pl.String,
            "comments_count": pl.Int64,
            "species_query": pl.String,
            "first_query_field": pl.String,
            "first_query_term": pl.String,
            "first_query_language": pl.String,
            "text_search_terms": pl.List(pl.String),
            "tag_search_terms": pl.List(pl.String),
            "all_query_labels": pl.List(pl.String),
            "all_query_terms": pl.List(pl.String),
            "all_query_fields": pl.List(pl.String),
            "query_hit_count": pl.Int64,
            "duplicate_query_hit_count": pl.Int64,
            "query_definition_ids": pl.List(pl.String),
            "discovery_accepted_taxon_keys": pl.List(pl.String),
            "discovery_family_keys": pl.List(pl.String),
            "discovery_genus_keys": pl.List(pl.String),
            "discovery_species_keys": pl.List(pl.String),
            "registry_versions": pl.List(pl.String),
            "species_text_match": pl.Boolean,
            "species_text_source": pl.List(pl.String),
            "scientific_names_detected": pl.List(pl.String),
            "human_verification_detected": pl.Boolean,
            "human_verification_source": pl.List(pl.String),
            "human_verification_terms": pl.List(pl.String),
            "human_verification_confidence": pl.Float64,
            "museum_detected": pl.Boolean,
            "artwork_detected": pl.Boolean,
            "specimen_detected": pl.Boolean,
            "tattoo_detected": pl.Boolean,
            "collection_detected": pl.Boolean,
            "captive_detected": pl.Boolean,
            "non_target_order_detected": pl.Boolean,
            "occurrence_bin": pl.String,
            "bin_reason": pl.String,
            "image_category": pl.String,
            "life_stage": pl.String,
            "negative_filter_reason": pl.String,
            "review_flags": pl.List(pl.String),
        }
    )
