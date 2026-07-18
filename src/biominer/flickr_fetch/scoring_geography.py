"""Precision-aware geography bound to canonical Flickr photo scoring units."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.flickr_fetch.geography import (
    FLICKR_GEOGRAPHY_SCHEMA_VERSION,
    flickr_geography_schema,
)
from biominer.flickr_fetch.scoring_units import flickr_photo_embedding_unit_schema
from biominer.storage.parquet import write_parquet


FLICKR_SCORING_GEOGRAPHY_SCHEMA_VERSION = "flickr-scoring-geography-v1.0.0"
FLICKR_SCORING_GEOGRAPHY_FILE = "flickr_scoring_geography.parquet"
FLICKR_SCORING_GEOGRAPHY_AVAILABILITY = frozenset(
    {"available", "no_geo", "invalid", "withheld", "unassigned_geo"}
)

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SORT = ("run_id", "source", "flickr_photo_id", "photo_embedding_unit_id")


def flickr_scoring_geography_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "run_id": pl.String,
        "photo_embedding_unit_id": pl.String,
        "source": pl.String,
        "flickr_photo_id": pl.String,
        "source_record_hash": pl.String,
        "latitude": pl.Float64,
        "longitude": pl.Float64,
        "coordinate_uncertainty_m": pl.Float64,
        "coordinate_uncertainty_source": pl.String,
        "coordinate_accuracy": pl.Float64,
        "coordinate_source": pl.String,
        "coordinate_quality": pl.String,
        "geography_source_quality": pl.String,
        "country_code": pl.String,
        "admin1": pl.String,
        "bioregion": pl.String,
        "bioregion_source": pl.String,
        "coarse_cell_id": pl.String,
        "regional_cell_id": pl.String,
        "local_cell_id": pl.String,
        "supported_cell_resolution": pl.UInt8,
        "geography_availability": pl.String,
        "geography_unavailable_reason": pl.String,
        "geographic_scope": pl.String,
        "geographic_scope_value": pl.String,
        "source_geography_schema_version": pl.String,
        "source_geography_row_fingerprint": pl.String,
        "geography_config_fingerprint": pl.String,
        "bioregion_mapping_version": pl.String,
        "geography_policy_fingerprint": pl.String,
        "geography_signature": pl.String,
        "row_fingerprint": pl.String,
    }


def build_flickr_scoring_geography(
    photo_embedding_units: pl.DataFrame,
    geography: pl.DataFrame,
    *,
    bioregion_by_admin_region: Sequence[tuple[str, str]] = (),
    bioregion_mapping_version: str | None = None,
) -> pl.DataFrame:
    """Join one normalized geography row to each eligible photo unit."""

    _validate_source_frames(photo_embedding_units, geography)
    bioregion_map = _normalize_bioregion_map(bioregion_by_admin_region)
    mapping_version = _optional_text(
        bioregion_mapping_version, field="bioregion_mapping_version"
    )
    if bioregion_map and mapping_version is None:
        raise ValueError("bioregion_mapping_version is required with a mapping")
    if not bioregion_map and mapping_version is not None:
        raise ValueError("bioregion_mapping_version requires a bioregion mapping")
    policy_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": FLICKR_SCORING_GEOGRAPHY_SCHEMA_VERSION,
            "source_schema_version": FLICKR_GEOGRAPHY_SCHEMA_VERSION,
            "bioregion_by_admin_region": [
                [key, bioregion] for key, bioregion in sorted(bioregion_map.items())
            ],
            "bioregion_mapping_version": mapping_version,
            "scope_order": [
                "local_cell",
                "regional_cell",
                "coarse_cell",
                "bioregion",
                "admin1",
                "country",
            ],
        }
    )
    geography_by_photo = {
        (str(row["source"]), str(row["flickr_photo_id"])): row
        for row in geography.iter_rows(named=True)
    }
    rows: list[dict[str, object]] = []
    for photo in photo_embedding_units.iter_rows(named=True):
        key = (str(photo["source"]), str(photo["flickr_photo_id"]))
        try:
            source_row = geography_by_photo[key]
        except KeyError as exc:
            raise ValueError(
                f"eligible Flickr photo is missing geography: {key[0]}:{key[1]}"
            ) from exc
        if source_row["source_record_hash"] != photo["source_record_hash"]:
            raise ValueError(
                f"Flickr photo/geography source_record_hash mismatch: {key[0]}:{key[1]}"
            )
        bioregion = source_row["bioregion"]
        bioregion_source = source_row["bioregion_source"]
        if bioregion is None:
            admin_scope = _admin_scope(source_row)
            mapped = bioregion_map.get(admin_scope or "")
            if mapped is not None:
                bioregion = mapped
                bioregion_source = f"bioregion_mapping:{mapping_version}"
        availability, unavailable_reason = _availability(source_row)
        geographic_scope, scope_value = _best_scope(
            source_row,
            bioregion=bioregion,
            availability=availability,
        )
        signature = canonical_semantic_fingerprint(
            {
                "schema_version": "flickr-scoring-geography-signature-v1",
                "availability": availability,
                "coordinate_quality": source_row["coordinate_quality"],
                "coordinate_uncertainty_m": source_row["coordinate_uncertainty_m"],
                "geographic_scope": geographic_scope,
                "geographic_scope_value": scope_value,
                "source_geography_row_fingerprint": source_row["row_fingerprint"],
                "geography_policy_fingerprint": policy_fingerprint,
            }
        )
        base = {
            "schema_version": FLICKR_SCORING_GEOGRAPHY_SCHEMA_VERSION,
            "run_id": photo["run_id"],
            "photo_embedding_unit_id": photo["photo_embedding_unit_id"],
            "source": photo["source"],
            "flickr_photo_id": photo["flickr_photo_id"],
            "source_record_hash": photo["source_record_hash"],
            "latitude": source_row["latitude"],
            "longitude": source_row["longitude"],
            "coordinate_uncertainty_m": source_row["coordinate_uncertainty_m"],
            "coordinate_uncertainty_source": source_row[
                "coordinate_uncertainty_source"
            ],
            "coordinate_accuracy": source_row["coordinate_accuracy"],
            "coordinate_source": source_row["coordinate_source"],
            "coordinate_quality": source_row["coordinate_quality"],
            "geography_source_quality": source_row["geography_source_quality"],
            "country_code": source_row["country_code"],
            "admin1": source_row["admin1"],
            "bioregion": bioregion,
            "bioregion_source": bioregion_source,
            "coarse_cell_id": source_row["coarse_cell_id"],
            "regional_cell_id": source_row["regional_cell_id"],
            "local_cell_id": source_row["local_cell_id"],
            "supported_cell_resolution": source_row["supported_cell_resolution"],
            "geography_availability": availability,
            "geography_unavailable_reason": unavailable_reason,
            "geographic_scope": geographic_scope,
            "geographic_scope_value": scope_value,
            "source_geography_schema_version": source_row["schema_version"],
            "source_geography_row_fingerprint": source_row["row_fingerprint"],
            "geography_config_fingerprint": source_row[
                "geography_config_fingerprint"
            ],
            "bioregion_mapping_version": mapping_version,
            "geography_policy_fingerprint": policy_fingerprint,
            "geography_signature": signature,
        }
        rows.append(
            {**base, "row_fingerprint": canonical_semantic_fingerprint(base)}
        )
    frame = (
        pl.DataFrame(
            rows,
            schema=flickr_scoring_geography_schema(),
            orient="row",
            strict=True,
        ).sort(*_SORT)
        if rows
        else pl.DataFrame(schema=flickr_scoring_geography_schema())
    )
    validate_flickr_scoring_geography(frame, photo_embedding_units)
    return frame


def validate_flickr_scoring_geography(
    frame: pl.DataFrame,
    photo_embedding_units: pl.DataFrame,
) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("Flickr scoring geography must be a Polars DataFrame")
    if frame.schema != flickr_scoring_geography_schema():
        raise ValueError("Flickr scoring geography schema mismatch")
    if not frame.equals(frame.sort(*_SORT)):
        raise ValueError("Flickr scoring geography is not canonically sorted")
    if frame.height != frame.select(
        "run_id", "source", "flickr_photo_id"
    ).n_unique():
        raise ValueError("Flickr scoring geography photo grain is not unique")
    if frame.height != photo_embedding_units.height:
        raise ValueError("Flickr scoring geography does not cover every photo unit")
    photo_by_id = {
        str(row["photo_embedding_unit_id"]): row
        for row in photo_embedding_units.iter_rows(named=True)
    }
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != FLICKR_SCORING_GEOGRAPHY_SCHEMA_VERSION:
            raise ValueError("unsupported Flickr scoring-geography schema version")
        if row["geography_availability"] not in FLICKR_SCORING_GEOGRAPHY_AVAILABILITY:
            raise ValueError("unsupported Flickr scoring-geography availability")
        try:
            photo = photo_by_id[str(row["photo_embedding_unit_id"])]
        except KeyError as exc:
            raise ValueError("geography references an unknown photo unit") from exc
        for field in (
            "run_id",
            "source",
            "flickr_photo_id",
            "source_record_hash",
        ):
            if row[field] != photo[field]:
                raise ValueError(f"geography/photo unit {field} mismatch")
        for field in (
            "source_record_hash",
            "source_geography_row_fingerprint",
            "geography_config_fingerprint",
            "geography_policy_fingerprint",
            "geography_signature",
            "row_fingerprint",
        ):
            _sha256(row[field], field=field)
        _validate_precision(row)
        _validate_availability(row)
        expected = canonical_semantic_fingerprint(_without(row, "row_fingerprint"))
        if row["row_fingerprint"] != expected:
            raise ValueError("Flickr scoring-geography row fingerprint mismatch")


def write_flickr_scoring_geography(
    frame: pl.DataFrame,
    photo_embedding_units: pl.DataFrame,
    output_path: str | Path,
) -> Path:
    validate_flickr_scoring_geography(frame, photo_embedding_units)
    destination = Path(output_path)
    if destination.suffix.casefold() != ".parquet":
        destination /= FLICKR_SCORING_GEOGRAPHY_FILE
    return write_parquet(frame, destination)


def _validate_source_frames(
    photo_embedding_units: pl.DataFrame,
    geography: pl.DataFrame,
) -> None:
    if not isinstance(photo_embedding_units, pl.DataFrame):
        raise TypeError("photo_embedding_units must be a Polars DataFrame")
    if photo_embedding_units.schema != flickr_photo_embedding_unit_schema():
        raise ValueError("photo embedding-unit schema mismatch")
    if not isinstance(geography, pl.DataFrame):
        raise TypeError("geography must be a Polars DataFrame")
    if geography.schema != flickr_geography_schema():
        raise ValueError("Flickr geography schema mismatch")
    identities = geography.select("source", "flickr_photo_id")
    if identities.n_unique() != geography.height:
        raise ValueError("source Flickr geography photo grain is not unique")
    for row in geography.iter_rows(named=True):
        if row["schema_version"] != FLICKR_GEOGRAPHY_SCHEMA_VERSION:
            raise ValueError("unsupported source Flickr geography schema version")
        expected = canonical_semantic_fingerprint(_without(row, "row_fingerprint"))
        if row["row_fingerprint"] != expected:
            raise ValueError("source Flickr geography row fingerprint mismatch")


def _availability(row: Mapping[str, object]) -> tuple[str, str | None]:
    quality = str(row["coordinate_quality"])
    warning = row["geography_warning"]
    if quality == "withheld":
        return "withheld", str(warning or "coordinates_withheld")
    if quality == "missing":
        return "no_geo", str(warning or "coordinates_missing")
    if quality == "invalid":
        return "invalid", str(warning or "coordinates_invalid")
    if any(
        row[field]
        for field in (
            "local_cell_id",
            "regional_cell_id",
            "coarse_cell_id",
            "bioregion",
            "admin1",
            "country_code",
        )
    ):
        return "available", None
    return "unassigned_geo", "no_supported_geographic_scope"


def _best_scope(
    row: Mapping[str, object],
    *,
    bioregion: object,
    availability: str,
) -> tuple[str, str | None]:
    for scope, field in (
        ("local_cell", "local_cell_id"),
        ("regional_cell", "regional_cell_id"),
        ("coarse_cell", "coarse_cell_id"),
    ):
        if row[field]:
            return scope, str(row[field])
    if bioregion:
        return "bioregion", str(bioregion)
    if row["admin1"]:
        return "admin1", str(row["admin1"])
    if row["country_code"]:
        return "country", str(row["country_code"])
    return availability, None


def _validate_precision(row: Mapping[str, object]) -> None:
    resolution = row["supported_cell_resolution"]
    populated = [
        field
        for field in ("coarse_cell_id", "regional_cell_id", "local_cell_id")
        if row[field] is not None
    ]
    if resolution is None:
        if populated:
            raise ValueError("cells require a supported resolution")
    elif not populated:
        raise ValueError("supported cell resolution requires a populated cell")
    if row["local_cell_id"] is not None and row["regional_cell_id"] is None:
        raise ValueError("local cell requires its regional cell")
    if row["regional_cell_id"] is not None and row["coarse_cell_id"] is None:
        raise ValueError("regional cell requires its coarse cell")
    if row["coordinate_uncertainty_m"] is None:
        if row["coordinate_uncertainty_source"] is not None:
            raise ValueError("coordinate uncertainty source requires a metric value")
    elif float(row["coordinate_uncertainty_m"]) < 0.0:
        raise ValueError("coordinate uncertainty cannot be negative")


def _validate_availability(row: Mapping[str, object]) -> None:
    availability = row["geography_availability"]
    reason = row["geography_unavailable_reason"]
    if availability == "available":
        if reason is not None or row["geographic_scope"] in {
            "no_geo",
            "invalid",
            "withheld",
            "unassigned_geo",
        }:
            raise ValueError("available scoring geography has invalid fallback fields")
    elif reason is None:
        raise ValueError("unavailable scoring geography requires an exact reason")


def _normalize_bioregion_map(
    values: Sequence[tuple[str, str]],
) -> dict[str, str]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError("bioregion_by_admin_region must be a sequence")
    normalized: dict[str, str] = {}
    for value in values:
        if not isinstance(value, Sequence) or len(value) != 2:
            raise ValueError("bioregion map entries must be (admin region, bioregion)")
        key = _required_text(value[0], field="bioregion admin region")
        bioregion = _required_text(value[1], field="bioregion")
        previous = normalized.setdefault(key, bioregion)
        if previous != bioregion:
            raise ValueError(f"admin region {key!r} maps to conflicting bioregions")
    return normalized


def _admin_scope(row: Mapping[str, object]) -> str | None:
    country = row["country_code"]
    admin1 = row["admin1"]
    if country and admin1:
        return f"{country}:{admin1}"
    return str(admin1) if admin1 else None


def _without(row: Mapping[str, object], *fields: str) -> dict[str, object]:
    excluded = set(fields)
    return {key: value for key, value in row.items() if key not in excluded}


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} cannot be blank")
    return text


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase sha256 fingerprint")
    return text


__all__ = [
    "FLICKR_SCORING_GEOGRAPHY_AVAILABILITY",
    "FLICKR_SCORING_GEOGRAPHY_FILE",
    "FLICKR_SCORING_GEOGRAPHY_SCHEMA_VERSION",
    "build_flickr_scoring_geography",
    "flickr_scoring_geography_schema",
    "validate_flickr_scoring_geography",
    "write_flickr_scoring_geography",
]
