"""Versioned geographic index contract over cached reference embeddings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from math import isfinite
from pathlib import Path
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.admission import REFERENCE_ADMISSION_MODES
from biominer.references.readiness import reference_route_dimensions
from biominer.storage.parquet import write_parquet
from biominer.vision.full_frame_attention import (
    FOCUSED_FULL_FRAME_KIND,
    MASKED_FULL_FRAME_KIND,
    MULTI_OBJECT_FULL_FRAME_KIND,
    RAW_FULL_IMAGE_KIND,
)


REFERENCE_GEOGRAPHY_INDEX_SCHEMA_VERSION = "reference-geography-index-v1.0.0"
REFERENCE_GEOGRAPHY_INDEX_FILE = "reference_geography_index.parquet"
REFERENCE_COORDINATE_QUALITIES = frozenset(
    {
        "local",
        "regional",
        "coarse",
        "country_only",
        "unknown_precision",
        "missing",
        "invalid",
        "withheld",
    }
)

_VISUAL_INPUT_KINDS = frozenset(
    {
        RAW_FULL_IMAGE_KIND,
        FOCUSED_FULL_FRAME_KIND,
        MASKED_FULL_FRAME_KIND,
        MULTI_OBJECT_FULL_FRAME_KIND,
    }
)
_NON_GEOGRAPHIC_QUALITIES = frozenset(
    {"country_only", "missing", "invalid", "withheld"}
)
_GEOGRAPHIC_QUALITIES = frozenset({"local", "regional", "coarse"})
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REFERENCE_MEDIA_ID_PATTERN = re.compile(r"reference-media:[0-9a-f]{64}\Z")
_REFERENCE_OBSERVATION_ID_PATTERN = re.compile(r"reference-observation:[0-9a-f]{64}\Z")
_DUPLICATE_GROUP_ID_PATTERN = re.compile(r"reference-duplicate-group:[0-9a-f]{32}\Z")
_SORT = (
    "accepted_taxon_key",
    "route",
    "country_code",
    "coarse_cell_id",
    "regional_cell_id",
    "local_cell_id",
    "reference_observation_id",
    "reference_media_id",
    "visual_input_kind",
    "embedding_fingerprint",
)
_GRAIN = (
    "reference_media_id",
    "route",
    "visual_input_kind",
    "embedding_fingerprint",
)


def reference_geography_index_schema() -> dict[str, pl.DataType]:
    """Return the closed physical schema for the reference geography index."""

    return {
        "schema_version": pl.String,
        "registry_version": pl.String,
        "reference_bank_version": pl.String,
        "reference_media_id": pl.String,
        "reference_observation_id": pl.String,
        "source": pl.String,
        "source_dataset_key": pl.String,
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "family_key": pl.String,
        "family_name": pl.String,
        "genus_key": pl.String,
        "genus_name": pl.String,
        "route": pl.String,
        "life_stage": pl.String,
        "visual_domain": pl.String,
        "visual_input_kind": pl.String,
        "country_code": pl.String,
        "admin1": pl.String,
        "bioregion": pl.String,
        "geo_cluster_id": pl.String,
        "coarse_cell_id": pl.String,
        "regional_cell_id": pl.String,
        "local_cell_id": pl.String,
        "latitude": pl.Float64,
        "longitude": pl.Float64,
        "coordinate_uncertainty_m": pl.Float64,
        "coordinate_quality": pl.String,
        "global_anchor_eligible": pl.Boolean,
        "local_anchor_eligible": pl.Boolean,
        "duplicate_group_id": pl.String,
        "observer_id_hash": pl.String,
        "observation_date": pl.Date,
        "admission_mode": pl.String,
        "admission_policy_fingerprint": pl.String,
        "reference_quality_flags": pl.List(pl.String),
        "embedding_fingerprint": pl.String,
        "row_fingerprint": pl.String,
    }


def build_reference_geography_index(
    rows: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    """Build a deterministic index without computing or copying embeddings."""

    if isinstance(rows, str | bytes) or not isinstance(rows, Sequence):
        raise TypeError("reference geography rows must be a sequence of mappings")
    schema = reference_geography_index_schema()
    expected_input = set(schema) - {"schema_version", "row_fingerprint"}
    output: list[dict[str, object]] = []
    for position, source_row in enumerate(rows):
        if not isinstance(source_row, Mapping):
            raise TypeError(f"reference geography row {position} must be a mapping")
        unexpected = set(source_row) - expected_input
        missing = expected_input - set(source_row)
        if missing or unexpected:
            raise ValueError(
                "reference geography input fields mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        row = _normalized_row(source_row)
        row["schema_version"] = REFERENCE_GEOGRAPHY_INDEX_SCHEMA_VERSION
        row["row_fingerprint"] = canonical_semantic_fingerprint(row)
        output.append(row)

    frame = (
        pl.DataFrame(output, schema=schema, orient="row", strict=True).sort(*_SORT)
        if output
        else pl.DataFrame(schema=schema)
    )
    validate_reference_geography_index(frame)
    return frame


def validate_reference_geography_index(frame: pl.DataFrame) -> None:
    """Fail closed on schema, semantic, grain, ordering or fingerprint drift."""

    if not isinstance(frame, pl.DataFrame):
        raise TypeError("reference geography index must be a Polars DataFrame")
    if frame.schema != reference_geography_index_schema():
        raise ValueError("reference geography index schema mismatch")
    if frame.is_empty():
        return
    if frame.select(_GRAIN).n_unique() != frame.height:
        raise ValueError("reference geography index grain is not unique")
    if not frame.equals(frame.sort(*_SORT)):
        raise ValueError("reference geography index rows are not canonically sorted")

    media_semantics: dict[str, tuple[object, ...]] = {}
    for row in frame.iter_rows(named=True):
        _validate_row(row)
        media_id = str(row["reference_media_id"])
        semantics = (
            row["reference_observation_id"],
            row["source"],
            row["accepted_taxon_key"],
            row["duplicate_group_id"],
        )
        previous = media_semantics.setdefault(media_id, semantics)
        if previous != semantics:
            raise ValueError("reference media identity has conflicting index semantics")


def reference_geography_index_artifact_fingerprint(frame: pl.DataFrame) -> str:
    """Fingerprint index semantics independently of path and physical bytes."""

    validate_reference_geography_index(frame)
    return canonical_semantic_fingerprint(
        {
            "schema_version": REFERENCE_GEOGRAPHY_INDEX_SCHEMA_VERSION,
            "registry_versions": sorted(frame["registry_version"].unique().to_list()),
            "reference_bank_versions": sorted(
                frame["reference_bank_version"].unique().to_list()
            ),
            "row_fingerprints": sorted(frame["row_fingerprint"].to_list()),
        }
    )


def write_reference_geography_index(
    frame: pl.DataFrame,
    output: str | Path,
) -> Path:
    """Validate and atomically write ``reference_geography_index.parquet``."""

    validate_reference_geography_index(frame)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= REFERENCE_GEOGRAPHY_INDEX_FILE
    return write_parquet(frame, destination)


def _normalized_row(source: Mapping[str, object]) -> dict[str, object]:
    required_text = (
        "registry_version",
        "reference_bank_version",
        "reference_media_id",
        "reference_observation_id",
        "source",
        "source_dataset_key",
        "accepted_taxon_key",
        "scientific_name",
        "family_key",
        "family_name",
        "genus_key",
        "genus_name",
        "route",
        "life_stage",
        "visual_domain",
        "visual_input_kind",
        "coordinate_quality",
        "duplicate_group_id",
        "admission_mode",
        "admission_policy_fingerprint",
        "embedding_fingerprint",
    )
    optional_text = (
        "country_code",
        "admin1",
        "bioregion",
        "geo_cluster_id",
        "coarse_cell_id",
        "regional_cell_id",
        "local_cell_id",
        "observer_id_hash",
    )
    row: dict[str, object] = {
        field: _required_text(source[field], field=field) for field in required_text
    }
    row.update(
        {field: _optional_text(source[field], field=field) for field in optional_text}
    )
    if row["country_code"] is not None:
        row["country_code"] = str(row["country_code"]).upper()
    row.update(
        {
            "latitude": _optional_float(source["latitude"], field="latitude"),
            "longitude": _optional_float(source["longitude"], field="longitude"),
            "coordinate_uncertainty_m": _optional_float(
                source["coordinate_uncertainty_m"],
                field="coordinate_uncertainty_m",
            ),
            "global_anchor_eligible": _boolean(
                source["global_anchor_eligible"], field="global_anchor_eligible"
            ),
            "local_anchor_eligible": _boolean(
                source["local_anchor_eligible"], field="local_anchor_eligible"
            ),
            "observation_date": _optional_date(source["observation_date"]),
            "reference_quality_flags": _canonical_flags(
                source["reference_quality_flags"]
            ),
        }
    )
    return row


def _validate_row(row: Mapping[str, object]) -> None:
    if row["schema_version"] != REFERENCE_GEOGRAPHY_INDEX_SCHEMA_VERSION:
        raise ValueError("unsupported reference geography index schema version")
    for field in (
        "registry_version",
        "reference_bank_version",
        "reference_media_id",
        "reference_observation_id",
        "source",
        "source_dataset_key",
        "accepted_taxon_key",
        "scientific_name",
        "family_key",
        "family_name",
        "genus_key",
        "genus_name",
        "route",
        "life_stage",
        "visual_domain",
        "visual_input_kind",
        "coordinate_quality",
        "duplicate_group_id",
        "admission_mode",
    ):
        if _required_text(row[field], field=field) != row[field]:
            raise ValueError(f"{field} is not canonically normalized")
    for field in (
        "country_code",
        "admin1",
        "bioregion",
        "geo_cluster_id",
        "coarse_cell_id",
        "regional_cell_id",
        "local_cell_id",
        "observer_id_hash",
    ):
        if _optional_text(row[field], field=field) != row[field]:
            raise ValueError(f"{field} is not canonically normalized")

    if not _REFERENCE_MEDIA_ID_PATTERN.fullmatch(str(row["reference_media_id"])):
        raise ValueError("reference_media_id is invalid")
    if not _REFERENCE_OBSERVATION_ID_PATTERN.fullmatch(
        str(row["reference_observation_id"])
    ):
        raise ValueError("reference_observation_id is invalid")
    if not _DUPLICATE_GROUP_ID_PATTERN.fullmatch(str(row["duplicate_group_id"])):
        raise ValueError("duplicate_group_id is invalid")
    for field in (
        "admission_policy_fingerprint",
        "embedding_fingerprint",
        "row_fingerprint",
    ):
        if not _SHA256_PATTERN.fullmatch(str(row[field])):
            raise ValueError(f"{field} is not a canonical SHA-256 fingerprint")
    observer = row["observer_id_hash"]
    if observer is not None and not _SHA256_PATTERN.fullmatch(str(observer)):
        raise ValueError("observer_id_hash is not a canonical SHA-256 fingerprint")

    if row["admission_mode"] not in REFERENCE_ADMISSION_MODES:
        raise ValueError("unsupported reference geography admission_mode")
    expected_stage, expected_domain = reference_route_dimensions(str(row["route"]))
    if row["life_stage"] != expected_stage or row["visual_domain"] != expected_domain:
        raise ValueError("reference route, life_stage and visual_domain conflict")
    if row["visual_input_kind"] not in _VISUAL_INPUT_KINDS:
        raise ValueError("unsupported reference visual_input_kind")

    country = row["country_code"]
    if country is not None and (
        len(str(country)) != 2
        or not str(country).isalpha()
        or not str(country).isupper()
    ):
        raise ValueError("country_code must be an uppercase ISO alpha-2 value")
    quality = str(row["coordinate_quality"])
    if quality not in REFERENCE_COORDINATE_QUALITIES:
        raise ValueError("unsupported reference coordinate_quality")
    _validate_geography(row, quality=quality)

    flags = row["reference_quality_flags"]
    if not isinstance(flags, list) or flags != sorted(set(flags)):
        raise ValueError("reference_quality_flags must be a sorted unique string list")
    if any(
        not isinstance(flag, str) or not flag.strip() or flag != flag.strip()
        for flag in flags
    ):
        raise ValueError("reference_quality_flags contains an invalid value")
    if not isinstance(row["global_anchor_eligible"], bool) or not isinstance(
        row["local_anchor_eligible"], bool
    ):
        raise TypeError("reference anchor eligibility fields must be Boolean")
    if row["observation_date"] is not None and (
        not isinstance(row["observation_date"], date)
        or isinstance(row["observation_date"], datetime)
    ):
        raise ValueError("observation_date must be a date or null")

    payload = dict(row)
    fingerprint = payload.pop("row_fingerprint")
    if fingerprint != canonical_semantic_fingerprint(payload):
        raise ValueError("reference geography row fingerprint mismatch")


def _validate_geography(row: Mapping[str, object], *, quality: str) -> None:
    latitude = row["latitude"]
    longitude = row["longitude"]
    if (latitude is None) != (longitude is None):
        raise ValueError("reference latitude and longitude must both be set or null")
    if latitude is not None:
        if not isfinite(float(latitude)) or not -90 <= float(latitude) <= 90:
            raise ValueError("reference latitude is invalid")
        if not isfinite(float(longitude)) or not -180 <= float(longitude) <= 180:
            raise ValueError("reference longitude is invalid")
    uncertainty = row["coordinate_uncertainty_m"]
    if uncertainty is not None and (
        not isfinite(float(uncertainty)) or float(uncertainty) < 0
    ):
        raise ValueError("coordinate_uncertainty_m must be finite and non-negative")

    coarse = row["coarse_cell_id"]
    regional = row["regional_cell_id"]
    local = row["local_cell_id"]
    if local is not None and (regional is None or coarse is None):
        raise ValueError("local_cell_id requires regional and coarse parents")
    if regional is not None and coarse is None:
        raise ValueError("regional_cell_id requires a coarse parent")
    if quality in _NON_GEOGRAPHIC_QUALITIES:
        if any(
            value is not None
            for value in (latitude, longitude, coarse, regional, local)
        ):
            raise ValueError(
                "non-geographic coordinate quality cannot carry coordinates or cells"
            )
        if row["local_anchor_eligible"]:
            raise ValueError("non-geographic references cannot be local anchors")
        if quality == "country_only" and row["country_code"] is None:
            raise ValueError("country_only quality requires country_code")
    elif latitude is None:
        raise ValueError("usable or unknown-precision geography requires coordinates")

    if quality == "local" and local is None:
        raise ValueError("local coordinate quality requires all cell levels")
    if quality == "regional" and (regional is None or local is not None):
        raise ValueError(
            "regional coordinate quality requires regional/coarse cells only"
        )
    if quality == "coarse" and (
        coarse is None or regional is not None or local is not None
    ):
        raise ValueError("coarse coordinate quality requires only a coarse cell")
    if quality == "unknown_precision" and any(
        value is not None for value in (coarse, regional, local)
    ):
        raise ValueError("unknown_precision geography cannot claim cell precision")
    if row["local_anchor_eligible"] and quality not in _GEOGRAPHIC_QUALITIES:
        raise ValueError("local anchor eligibility requires usable geographic quality")


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field=field)


def _optional_float(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric or null")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be Boolean")
    return value


def _optional_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        raise ValueError("observation_date must not contain a time")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("observation_date must be an ISO date") from exc
    raise ValueError("observation_date must be a date, ISO date string or null")


def _canonical_flags(value: object) -> list[str]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError("reference_quality_flags must be a sequence of strings")
    flags = [_required_text(item, field="reference_quality_flags") for item in value]
    return sorted(set(flags))


__all__ = [
    "REFERENCE_COORDINATE_QUALITIES",
    "REFERENCE_GEOGRAPHY_INDEX_FILE",
    "REFERENCE_GEOGRAPHY_INDEX_SCHEMA_VERSION",
    "build_reference_geography_index",
    "reference_geography_index_artifact_fingerprint",
    "reference_geography_index_schema",
    "validate_reference_geography_index",
    "write_reference_geography_index",
]
