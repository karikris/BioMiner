"""Canonical Flickr candidate geography without taxonomic or range inference."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import polars as pl

from biominer.geography import (
    CellGrid,
    GeographicCoordinate,
    GeographicResolutions,
    default_cell_grid,
    project_coordinate,
)
from biominer.storage.parquet import write_parquet


FLICKR_GEOGRAPHY_SCHEMA_VERSION = "flickr-geography-v1.0.0"
FLICKR_ACCURACY_POLICY_VERSION = "flickr-accuracy-v1.0.0"
FLICKR_GEOGRAPHY_FILE = "flickr_geography.parquet"

_WARNING_PRIORITY = (
    "flickr_zero_geo_sentinel",
    "coordinate_pair_incomplete",
    "invalid_latitude",
    "invalid_longitude",
    "coordinates_missing",
    "coordinate_accuracy_invalid",
    "coordinate_accuracy_out_of_range",
    "coordinate_accuracy_nonintegral",
    "coordinate_precision_unknown",
    "coordinate_precision_limits_cells",
    "coordinate_at_null_island",
    "country_code_invalid",
)


@dataclass(frozen=True, slots=True)
class FlickrGeographyConfig:
    resolutions: GeographicResolutions = field(
        default_factory=lambda: GeographicResolutions(coarse=3, regional=5, local=7)
    )
    accuracy_policy_version: str = FLICKR_ACCURACY_POLICY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.resolutions, GeographicResolutions):
            raise TypeError("resolutions must be GeographicResolutions")
        policy = str(self.accuracy_policy_version).strip()
        if not policy:
            raise ValueError("accuracy_policy_version must be nonblank")
        object.__setattr__(self, "accuracy_policy_version", policy)


def flickr_geography_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "source": pl.String,
        "flickr_photo_id": pl.String,
        "source_record_hash": pl.String,
        "latitude": pl.Float64,
        "longitude": pl.Float64,
        "coordinate_accuracy": pl.Float64,
        "coordinate_source": pl.String,
        "geotag_available": pl.Boolean,
        "country_code": pl.String,
        "admin1": pl.String,
        "coarse_cell_id": pl.String,
        "regional_cell_id": pl.String,
        "local_cell_id": pl.String,
        "coordinate_quality": pl.String,
        "geography_warning": pl.String,
        "geography_warnings": pl.List(pl.String),
        "geography_config_fingerprint": pl.String,
    }


def build_flickr_geography_frame(
    records: pl.DataFrame | Iterable[Mapping[str, Any]],
    *,
    config: FlickrGeographyConfig | None = None,
    grid: CellGrid | None = None,
) -> pl.DataFrame:
    effective_config = config or FlickrGeographyConfig()
    if not isinstance(effective_config, FlickrGeographyConfig):
        raise TypeError("config must be a FlickrGeographyConfig")
    backend = grid or default_cell_grid()
    fingerprint = geography_config_fingerprint(effective_config, grid=backend)
    rows = [
        _project_flickr_record(
            record,
            config=effective_config,
            grid=backend,
            config_fingerprint=fingerprint,
        )
        for record in _record_rows(records)
    ]
    rows.sort(key=lambda row: (str(row["source"]), str(row["flickr_photo_id"])))
    _require_unique_identities(rows)
    schema = flickr_geography_schema()
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def write_flickr_geography(
    records: pl.DataFrame | Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    config: FlickrGeographyConfig | None = None,
    grid: CellGrid | None = None,
    overwrite: bool = True,
) -> Path:
    frame = build_flickr_geography_frame(records, config=config, grid=grid)
    return write_parquet(frame, output_path, overwrite=overwrite)


def geography_config_fingerprint(
    config: FlickrGeographyConfig,
    *,
    grid: CellGrid | None = None,
) -> str:
    if not isinstance(config, FlickrGeographyConfig):
        raise TypeError("config must be a FlickrGeographyConfig")
    backend = grid or default_cell_grid()
    payload = {
        "accuracy_policy_version": config.accuracy_policy_version,
        "cell_support": {
            "flickr_city": [config.resolutions.coarse, config.resolutions.regional],
            "flickr_country": [],
            "flickr_region": [config.resolutions.coarse],
            "flickr_street": list(config.resolutions.values),
            "flickr_world": [],
            "unknown_precision": [],
        },
        "grid_name": backend.name,
        "grid_version": backend.version,
        "resolutions": list(config.resolutions.values),
        "schema_version": FLICKR_GEOGRAPHY_SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def _project_flickr_record(
    record: Mapping[str, Any],
    *,
    config: FlickrGeographyConfig,
    grid: CellGrid,
    config_fingerprint: str,
) -> dict[str, object]:
    source = str(record.get("source") or "flickr").strip().casefold()
    if source != "flickr":
        raise ValueError(f"Flickr geography source must be 'flickr', got {source!r}")
    photo_id = str(record.get("flickr_photo_id") or record.get("id") or "").strip()
    if not photo_id:
        raise ValueError("Flickr geography record requires flickr_photo_id")
    source_record_hash = str(record.get("source_record_hash") or "").strip()
    if not source_record_hash:
        raise ValueError(
            f"Flickr geography record {photo_id!r} requires source_record_hash"
        )

    warnings: set[str] = set()
    country_code = _country_code(record, warnings=warnings)
    admin1 = _admin1(record)
    coordinate_accuracy = _coordinate_accuracy(record, warnings=warnings)
    latitude_value, longitude_value, inferred_source = _coordinate_values(record)
    explicit_source = _optional_text(record.get("coordinate_source"))
    coordinate_source = explicit_source or inferred_source

    latitude: float | None = None
    longitude: float | None = None
    geotag_available = False
    coordinate_quality = "missing"
    cells: dict[int, str] = {}
    latitude_present = _is_present(latitude_value)
    longitude_present = _is_present(longitude_value)

    if not latitude_present and not longitude_present:
        warnings.add("coordinates_missing")
        coordinate_source = None
    elif latitude_present != longitude_present:
        warnings.add("coordinate_pair_incomplete")
        coordinate_quality = "invalid"
    else:
        latitude = _coordinate_number(
            latitude_value,
            minimum=-90.0,
            maximum=90.0,
            warning="invalid_latitude",
            warnings=warnings,
        )
        longitude = _coordinate_number(
            longitude_value,
            minimum=-180.0,
            maximum=180.0,
            warning="invalid_longitude",
            warnings=warnings,
        )
        if latitude is None or longitude is None:
            latitude = None
            longitude = None
            coordinate_quality = "invalid"
        else:
            if latitude == 0.0 and longitude == 0.0 and coordinate_accuracy == 0.0:
                latitude = None
                longitude = None
                coordinate_source = None
                coordinate_quality = "missing"
                warnings.add("flickr_zero_geo_sentinel")
            else:
                geotag_available = True
                coordinate_quality, supported_resolutions = _coordinate_quality_and_resolutions(
                    coordinate_accuracy,
                    config=config,
                    warnings=warnings,
                )
                if latitude == 0.0 and longitude == 0.0:
                    warnings.add("coordinate_at_null_island")
                if supported_resolutions:
                    projection = project_coordinate(
                        GeographicCoordinate(latitude=latitude, longitude=longitude),
                        resolutions=config.resolutions,
                        grid=grid,
                    )
                    cells = {
                        resolution: projection.cell_at(resolution)
                        for resolution in supported_resolutions
                    }

    sorted_warnings = sorted(warnings)
    return {
        "schema_version": FLICKR_GEOGRAPHY_SCHEMA_VERSION,
        "source": source,
        "flickr_photo_id": photo_id,
        "source_record_hash": source_record_hash,
        "latitude": latitude,
        "longitude": longitude,
        "coordinate_accuracy": coordinate_accuracy,
        "coordinate_source": coordinate_source,
        "geotag_available": geotag_available,
        "country_code": country_code,
        "admin1": admin1,
        "coarse_cell_id": cells.get(int(config.resolutions.coarse)),
        "regional_cell_id": cells.get(int(config.resolutions.regional)),
        "local_cell_id": cells.get(int(config.resolutions.local)),
        "coordinate_quality": coordinate_quality,
        "geography_warning": _primary_warning(warnings),
        "geography_warnings": sorted_warnings,
        "geography_config_fingerprint": config_fingerprint,
    }


def _record_rows(
    records: pl.DataFrame | Iterable[Mapping[str, Any]],
) -> Iterable[Mapping[str, Any]]:
    if isinstance(records, pl.DataFrame):
        yield from records.iter_rows(named=True)
        return
    if isinstance(records, (str, bytes)):
        raise TypeError("records must be a DataFrame or iterable of mappings")
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("every Flickr geography record must be a mapping")
        yield record


def _coordinate_values(record: Mapping[str, Any]) -> tuple[object, object, str | None]:
    if "latitude" in record or "longitude" in record:
        return record.get("latitude"), record.get("longitude"), "flickr_search_geo"
    location = record.get("location")
    if isinstance(location, Mapping) and ("latitude" in location or "longitude" in location):
        return location.get("latitude"), location.get("longitude"), "flickr_geo_location"
    return None, None, None


def _coordinate_accuracy(
    record: Mapping[str, Any],
    *,
    warnings: set[str],
) -> float | None:
    value = record.get("coordinate_accuracy")
    if not _is_present(value):
        value = record.get("accuracy")
    if not _is_present(value):
        return None
    try:
        accuracy = _finite_float(value)
    except (TypeError, ValueError):
        warnings.add("coordinate_accuracy_invalid")
        return None
    if not 1.0 <= accuracy <= 16.0:
        warnings.add("coordinate_accuracy_out_of_range")
    elif not accuracy.is_integer():
        warnings.add("coordinate_accuracy_nonintegral")
    return accuracy


def _coordinate_quality_and_resolutions(
    accuracy: float | None,
    *,
    config: FlickrGeographyConfig,
    warnings: set[str],
) -> tuple[str, tuple[int, ...]]:
    resolutions = tuple(int(value) for value in config.resolutions.values)
    if accuracy is None or not 1.0 <= accuracy <= 16.0 or not accuracy.is_integer():
        warnings.add("coordinate_precision_unknown")
        return "unknown_precision", ()
    level = int(accuracy)
    if level <= 2:
        quality, supported = "flickr_world", ()
    elif level <= 5:
        quality, supported = "flickr_country", ()
    elif level <= 10:
        quality, supported = "flickr_region", resolutions[:1]
    elif level <= 15:
        quality, supported = "flickr_city", resolutions[:2]
    else:
        quality, supported = "flickr_street", resolutions
    if len(supported) < len(resolutions):
        warnings.add("coordinate_precision_limits_cells")
    return quality, supported


def _coordinate_number(
    value: object,
    *,
    minimum: float,
    maximum: float,
    warning: str,
    warnings: set[str],
) -> float | None:
    try:
        number = _finite_float(value)
    except (TypeError, ValueError):
        warnings.add(warning)
        return None
    if not minimum <= number <= maximum:
        warnings.add(warning)
        return None
    return number


def _country_code(record: Mapping[str, Any], *, warnings: set[str]) -> str | None:
    location = record.get("location")
    nested = location if isinstance(location, Mapping) else {}
    value = _first_present(
        record.get("country_code"),
        record.get("countryCode"),
        nested.get("country_code"),
        nested.get("countryCode"),
    )
    if value is None:
        country = record.get("country")
        if not isinstance(country, Mapping):
            value = country
    text = _optional_text(value)
    if text is None:
        return None
    normalized = text.upper()
    if len(normalized) != 2 or not normalized.isascii() or not normalized.isalpha():
        warnings.add("country_code_invalid")
        return None
    return normalized


def _admin1(record: Mapping[str, Any]) -> str | None:
    location = record.get("location")
    nested = location if isinstance(location, Mapping) else {}
    value = _first_present(
        record.get("admin1"),
        record.get("admin1_name"),
        record.get("stateProvince"),
        record.get("state_province"),
        nested.get("admin1"),
        nested.get("region"),
    )
    if isinstance(value, Mapping):
        value = _first_present(value.get("_content"), value.get("name"))
    return _optional_text(value)


def _first_present(*values: object) -> object | None:
    for value in values:
        if _is_present(value):
            return value
    return None


def _is_present(value: object) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _optional_text(value: object) -> str | None:
    if not _is_present(value):
        return None
    text = str(value).strip()
    return text or None


def _finite_float(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("boolean is not a coordinate number")
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ValueError("coordinate number must be finite")
    return number


def _primary_warning(warnings: set[str]) -> str | None:
    for warning in _WARNING_PRIORITY:
        if warning in warnings:
            return warning
    return min(warnings) if warnings else None


def _require_unique_identities(rows: list[dict[str, object]]) -> None:
    previous: tuple[str, str] | None = None
    for row in rows:
        identity = (str(row["source"]), str(row["flickr_photo_id"]))
        if identity == previous:
            raise ValueError(
                f"duplicate Flickr geography identity: {identity[0]}:{identity[1]}"
            )
        previous = identity


__all__ = [
    "FLICKR_ACCURACY_POLICY_VERSION",
    "FLICKR_GEOGRAPHY_FILE",
    "FLICKR_GEOGRAPHY_SCHEMA_VERSION",
    "FlickrGeographyConfig",
    "build_flickr_geography_frame",
    "flickr_geography_schema",
    "geography_config_fingerprint",
    "write_flickr_geography",
]
