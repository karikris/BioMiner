from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl

from flickr_bio_occurrence.dwc.mapper import map_candidate_to_dwc
from flickr_bio_occurrence.vision.image_selection import select_flickr_image_url


def flatten_search_payloads(
    payloads: list[dict[str, Any]],
    *,
    species_name: str,
    region_id: str,
    work_item_id: str,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    fetched_at = datetime.now(UTC).isoformat()
    for payload in payloads:
        for photo in payload.get("photos", {}).get("photo", []):
            rows.append(
                {
                    "flickr_photo_id": str(photo.get("id", "")),
                    "owner_id": photo.get("owner"),
                    "owner_name": photo.get("ownername"),
                    "raw_title": photo.get("title"),
                    "raw_description": _description_text(photo.get("description")),
                    "raw_tags": photo.get("tags"),
                    "machine_tags": photo.get("machine_tags"),
                    "license": photo.get("license"),
                    "date_upload": photo.get("dateupload"),
                    "date_taken": photo.get("datetaken") or photo.get("datetakenunknown"),
                    "last_update": photo.get("lastupdate"),
                    "views": _optional_int(photo.get("views")),
                    "media": photo.get("media"),
                    "image_url": _preferred_flickr_image_url(photo),
                    "photo_page_url": f"https://www.flickr.com/photos/{photo.get('owner', '')}/{photo.get('id', '')}",
                    "decimalLatitude": _optional_float(photo.get("latitude")),
                    "decimalLongitude": _optional_float(photo.get("longitude")),
                    "accuracy": _optional_int(photo.get("accuracy")),
                    "species_query": species_name,
                    "region_id": region_id,
                    "work_item_id": work_item_id,
                    "fetched_at": fetched_at,
                }
            )
    return pl.DataFrame(rows) if rows else _empty_bronze_frame()


def build_silver_candidates(bronze: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in bronze.to_dicts():
        latitude = row.get("decimalLatitude")
        longitude = row.get("decimalLongitude")
        rows.append(
            {
                "flickr_photo_id": row["flickr_photo_id"],
                "resolved_scientific_name": row["species_query"],
                "eventDate": _event_date(row.get("date_taken")),
                "decimalLatitude": latitude,
                "decimalLongitude": longitude,
                "exact_decimalLatitude_internal": latitude,
                "exact_decimalLongitude_internal": longitude,
                "exact_coordinate_source": "flickr_explicit_geotag" if latitude is not None and longitude is not None else None,
                "coordinateUncertaintyInMeters": None,
                "verbatimLocality": None,
                "georeferenceSources": "Flickr photos.search geo extras" if latitude is not None and longitude is not None else None,
                "georeferenceRemarks": "Exact Flickr coordinates stored internally when provided.",
                "verbatimIdentification": row.get("raw_title") or row["species_query"],
                "identificationVerificationStatus": "needs_review",
                "review_status": "needs_review",
                "range_extension_candidate": False,
                "species_agreement_status": "text_only",
                "associatedReferences": row.get("photo_page_url"),
                "associatedMedia": row.get("image_url"),
                "license": row.get("license"),
                "rightsHolder": row.get("owner_name"),
                "human_evidence": bool(row.get("raw_title") or row.get("raw_tags") or row.get("raw_description")),
                "dataGeneralizations": None,
                "informationWithheld": None,
                "occurrenceRemarks": "Flickr-derived occurrence candidate; requires evidence review.",
                "dynamicProperties": {
                    "source": "flickr.photos.search",
                    "work_item_id": row.get("work_item_id"),
                    "species_agreement_status": "text_only",
                },
            }
        )
    return pl.DataFrame(rows) if rows else _empty_silver_frame()


def build_dwc_rows(silver: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame([map_candidate_to_dwc(row) for row in silver.to_dicts()])


def _description_text(value: Any) -> str | None:
    if isinstance(value, dict):
        return value.get("_content")
    return value


def _preferred_flickr_image_url(photo: dict[str, Any]) -> Any:
    return select_flickr_image_url(photo).url


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _event_date(value: Any) -> str | None:
    if not value:
        return None
    return str(value).split(" ", 1)[0]


def _empty_bronze_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "flickr_photo_id": pl.String,
            "raw_title": pl.String,
            "decimalLatitude": pl.Float64,
            "decimalLongitude": pl.Float64,
            "species_query": pl.String,
            "region_id": pl.String,
            "work_item_id": pl.String,
        }
    )


def _empty_silver_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "flickr_photo_id": pl.String,
            "resolved_scientific_name": pl.String,
            "eventDate": pl.String,
            "decimalLatitude": pl.Float64,
            "decimalLongitude": pl.Float64,
            "review_status": pl.String,
            "range_extension_candidate": pl.Boolean,
        }
    )
