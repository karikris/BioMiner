"""Observation-grained, precision-aware reference geography materialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from pathlib import Path
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.geography import (
    CellGrid,
    GeographicCoordinate,
    GeographicResolutions,
    default_cell_grid,
)
from biominer.references.schemas import validate_reference_observations
from biominer.storage.parquet import write_parquet


NORMALIZED_REFERENCE_GEOGRAPHY_SCHEMA_VERSION = "normalized-reference-geography-v1.0.0"
NORMALIZED_REFERENCE_GEOGRAPHY_FILE = "normalized_reference_geography.parquet"
REFERENCE_GEOGRAPHY_PRECISION_POLICY_VERSION = "reference-geography-uncertainty-v1.0.0"
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

_CONTEXT_FIELDS = frozenset(
    {"reference_observation_id", "continent_code", "admin1", "bioregion"}
)
_GEO_QUALITIES = frozenset({"local", "regional", "coarse"})
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REFERENCE_OBSERVATION_ID_PATTERN = re.compile(r"reference-observation:[0-9a-f]{64}\Z")
_SORT = ("source", "reference_observation_id")


@dataclass(frozen=True, slots=True)
class ReferenceGeographyPrecisionPolicy:
    """Maximum source uncertainty allowed at each published cell level."""

    local_max_uncertainty_m: float = 5_000.0
    regional_max_uncertainty_m: float = 25_000.0
    coarse_max_uncertainty_m: float = 100_000.0
    version: str = REFERENCE_GEOGRAPHY_PRECISION_POLICY_VERSION

    def __post_init__(self) -> None:
        values = (
            self.local_max_uncertainty_m,
            self.regional_max_uncertainty_m,
            self.coarse_max_uncertainty_m,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not isfinite(float(value))
            or float(value) < 0
            for value in values
        ):
            raise ValueError(
                "geography uncertainty thresholds must be finite and nonnegative"
            )
        if not values[0] <= values[1] <= values[2]:
            raise ValueError("geography uncertainty thresholds must be ordered")
        object.__setattr__(
            self,
            "local_max_uncertainty_m",
            float(self.local_max_uncertainty_m),
        )
        object.__setattr__(
            self,
            "regional_max_uncertainty_m",
            float(self.regional_max_uncertainty_m),
        )
        object.__setattr__(
            self,
            "coarse_max_uncertainty_m",
            float(self.coarse_max_uncertainty_m),
        )
        object.__setattr__(self, "version", _required_text(self.version, "version"))


def normalized_reference_geography_schema() -> dict[str, pl.DataType]:
    """Return the closed physical schema at biological-observation grain."""

    return {
        "schema_version": pl.String,
        "registry_version": pl.String,
        "reference_observation_id": pl.String,
        "source": pl.String,
        "source_observation_id": pl.String,
        "source_dataset_key": pl.String,
        "source_snapshot_version": pl.String,
        "source_record_hash": pl.String,
        "source_query_fingerprint": pl.String,
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "observer_id_hash": pl.String,
        "observed_date": pl.Date,
        "country_code": pl.String,
        "country": pl.String,
        "continent_code": pl.String,
        "admin1": pl.String,
        "bioregion": pl.String,
        "source_geo_cluster_id": pl.String,
        "latitude": pl.Float64,
        "longitude": pl.Float64,
        "coordinate_uncertainty_m": pl.Float64,
        "coordinates_obscured": pl.Boolean,
        "geospatial_issue": pl.Boolean,
        "coordinate_quality": pl.String,
        "geography_unavailable_reason": pl.String,
        "cell_grid_name": pl.String,
        "cell_grid_version": pl.String,
        "coarse_cell_resolution": pl.UInt8,
        "regional_cell_resolution": pl.UInt8,
        "local_cell_resolution": pl.UInt8,
        "supported_cell_resolution": pl.UInt8,
        "coarse_cell_id": pl.String,
        "regional_cell_id": pl.String,
        "local_cell_id": pl.String,
        "geography_policy_version": pl.String,
        "local_max_uncertainty_m": pl.Float64,
        "regional_max_uncertainty_m": pl.Float64,
        "coarse_max_uncertainty_m": pl.Float64,
        "geography_policy_fingerprint": pl.String,
        "row_fingerprint": pl.String,
    }


def build_normalized_reference_geography(
    observations: pl.DataFrame,
    *,
    resolutions: GeographicResolutions,
    context_rows: Sequence[Mapping[str, object]] = (),
    policy: ReferenceGeographyPrecisionPolicy | None = None,
    grid: CellGrid | None = None,
) -> pl.DataFrame:
    """Normalize validated source observations without manufacturing precision."""

    validate_reference_observations(observations)
    if not isinstance(resolutions, GeographicResolutions):
        raise TypeError("resolutions must be GeographicResolutions")
    selected_policy = policy or ReferenceGeographyPrecisionPolicy()
    if not isinstance(selected_policy, ReferenceGeographyPrecisionPolicy):
        raise TypeError("policy must be ReferenceGeographyPrecisionPolicy")
    backend = grid or default_cell_grid()
    contexts = _context_by_observation(context_rows)
    observation_ids = set(observations["reference_observation_id"].to_list())
    unknown_context = set(contexts) - observation_ids
    if unknown_context:
        raise ValueError(
            "reference geography context contains unknown observations: "
            f"{sorted(unknown_context)}"
        )
    policy_fingerprint = reference_geography_policy_fingerprint(
        policy=selected_policy,
        resolutions=resolutions,
        grid=backend,
    )
    rows = [
        _normalized_geography_row(
            observation,
            context=contexts.get(str(observation["reference_observation_id"]), {}),
            resolutions=resolutions,
            policy=selected_policy,
            policy_fingerprint=policy_fingerprint,
            grid=backend,
        )
        for observation in observations.iter_rows(named=True)
    ]
    schema = normalized_reference_geography_schema()
    frame = (
        pl.DataFrame(rows, schema=schema, orient="row", strict=True).sort(*_SORT)
        if rows
        else pl.DataFrame(schema=schema)
    )
    validate_normalized_reference_geography(frame)
    return frame


def validate_normalized_reference_geography(frame: pl.DataFrame) -> None:
    """Reject schema, grain, precision, ordering, or identity drift."""

    if not isinstance(frame, pl.DataFrame):
        raise TypeError("normalized reference geography must be a Polars DataFrame")
    if frame.schema != normalized_reference_geography_schema():
        raise ValueError("normalized reference geography schema mismatch")
    if frame.is_empty():
        return
    if frame["reference_observation_id"].n_unique() != frame.height:
        raise ValueError("normalized reference geography duplicates observations")
    if not frame.equals(frame.sort(*_SORT)):
        raise ValueError("normalized reference geography is not canonically sorted")
    for row in frame.iter_rows(named=True):
        _validate_normalized_row(row)


def normalized_reference_geography_artifact_fingerprint(
    frame: pl.DataFrame,
) -> str:
    """Fingerprint semantic geography independently of its file path."""

    validate_normalized_reference_geography(frame)
    return canonical_semantic_fingerprint(
        {
            "schema_version": NORMALIZED_REFERENCE_GEOGRAPHY_SCHEMA_VERSION,
            "source_snapshots": sorted(
                frame["source_snapshot_version"].unique().to_list()
            ),
            "policy_fingerprints": sorted(
                frame["geography_policy_fingerprint"].unique().to_list()
            ),
            "row_fingerprints": sorted(frame["row_fingerprint"].to_list()),
        }
    )


def reference_geography_policy_fingerprint(
    *,
    policy: ReferenceGeographyPrecisionPolicy,
    resolutions: GeographicResolutions,
    grid: CellGrid,
) -> str:
    if not isinstance(policy, ReferenceGeographyPrecisionPolicy):
        raise TypeError("policy must be ReferenceGeographyPrecisionPolicy")
    if not isinstance(resolutions, GeographicResolutions):
        raise TypeError("resolutions must be GeographicResolutions")
    return _policy_fingerprint(
        policy=policy,
        resolutions=resolutions.values,
        grid_name=_required_text(grid.name, "cell_grid_name"),
        grid_version=_required_text(grid.version, "cell_grid_version"),
    )


def write_normalized_reference_geography(
    frame: pl.DataFrame,
    output: str | Path,
) -> Path:
    validate_normalized_reference_geography(frame)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= NORMALIZED_REFERENCE_GEOGRAPHY_FILE
    return write_parquet(frame, destination)


def _normalized_geography_row(
    observation: Mapping[str, object],
    *,
    context: Mapping[str, object],
    resolutions: GeographicResolutions,
    policy: ReferenceGeographyPrecisionPolicy,
    policy_fingerprint: str,
    grid: CellGrid,
) -> dict[str, object]:
    latitude = observation["latitude"]
    longitude = observation["longitude"]
    uncertainty = observation["coordinate_uncertainty"]
    obscured = bool(observation["coordinates_obscured"])
    issue = bool(observation["geospatial_issue"])
    quality, unavailable_reason = _coordinate_quality(
        latitude=latitude,
        longitude=longitude,
        uncertainty=uncertainty,
        obscured=obscured,
        geospatial_issue=issue,
        country_code=observation["country_code"],
        policy=policy,
    )
    normalized_latitude: float | None = None
    normalized_longitude: float | None = None
    cells: dict[str, str | None] = {
        "coarse_cell_id": None,
        "regional_cell_id": None,
        "local_cell_id": None,
    }
    supported_resolution: int | None = None
    if quality in _GEO_QUALITIES or quality == "unknown_precision":
        coordinate = GeographicCoordinate(
            latitude=float(latitude),
            longitude=float(longitude),
            coordinate_uncertainty_m=(
                None if uncertainty is None else float(uncertainty)
            ),
        )
        normalized_latitude = float(coordinate.latitude)
        normalized_longitude = float(longitude)
        if quality in _GEO_QUALITIES:
            cells["coarse_cell_id"] = grid.coordinate_to_cell(
                coordinate, resolution=int(resolutions.coarse)
            )
            supported_resolution = int(resolutions.coarse)
        if quality in {"local", "regional"}:
            cells["regional_cell_id"] = grid.coordinate_to_cell(
                coordinate, resolution=int(resolutions.regional)
            )
            supported_resolution = int(resolutions.regional)
        if quality == "local":
            cells["local_cell_id"] = grid.coordinate_to_cell(
                coordinate, resolution=int(resolutions.local)
            )
            supported_resolution = int(resolutions.local)

    observer_id = _optional_text(observation["observer_id"])
    observed_at = observation["observed_at"]
    row: dict[str, object] = {
        "schema_version": NORMALIZED_REFERENCE_GEOGRAPHY_SCHEMA_VERSION,
        "registry_version": _required_text(
            observation["registry_version"], "registry_version"
        ),
        "reference_observation_id": _required_text(
            observation["reference_observation_id"], "reference_observation_id"
        ),
        "source": _required_text(observation["source"], "source"),
        "source_observation_id": _required_text(
            observation["source_observation_id"], "source_observation_id"
        ),
        "source_dataset_key": _optional_text(observation["source_dataset_key"]),
        "source_snapshot_version": _required_text(
            observation["source_snapshot_version"], "source_snapshot_version"
        ),
        "source_record_hash": _required_fingerprint(
            observation["source_record_hash"], "source_record_hash"
        ),
        "source_query_fingerprint": _required_fingerprint(
            observation["source_query_fingerprint"], "source_query_fingerprint"
        ),
        "accepted_taxon_key": _optional_text(observation["accepted_taxon_key"]),
        "scientific_name": _optional_text(observation["reconciled_scientific_name"]),
        "observer_id_hash": (
            canonical_semantic_fingerprint(
                {
                    "namespace": "reference-observer-v1",
                    "source": str(observation["source"]).casefold(),
                    "observer_id": observer_id,
                }
            )
            if observer_id is not None
            else None
        ),
        "observed_date": observed_at.date()
        if isinstance(observed_at, datetime)
        else None,
        "country_code": _uppercase_optional_code(
            observation["country_code"], field="country_code"
        ),
        "country": _optional_text(observation["country"]),
        "continent_code": _uppercase_optional_code(
            context.get("continent_code"), field="continent_code"
        ),
        "admin1": _optional_text(context.get("admin1")),
        "bioregion": _optional_text(context.get("bioregion")),
        "source_geo_cluster_id": _optional_text(observation["geo_cluster_id"]),
        "latitude": normalized_latitude,
        "longitude": normalized_longitude,
        "coordinate_uncertainty_m": (
            None if uncertainty is None else float(uncertainty)
        ),
        "coordinates_obscured": obscured,
        "geospatial_issue": issue,
        "coordinate_quality": quality,
        "geography_unavailable_reason": unavailable_reason,
        "cell_grid_name": _required_text(grid.name, "cell_grid_name"),
        "cell_grid_version": _required_text(grid.version, "cell_grid_version"),
        "coarse_cell_resolution": int(resolutions.coarse),
        "regional_cell_resolution": int(resolutions.regional),
        "local_cell_resolution": int(resolutions.local),
        "supported_cell_resolution": supported_resolution,
        **cells,
        "geography_policy_version": policy.version,
        "local_max_uncertainty_m": policy.local_max_uncertainty_m,
        "regional_max_uncertainty_m": policy.regional_max_uncertainty_m,
        "coarse_max_uncertainty_m": policy.coarse_max_uncertainty_m,
        "geography_policy_fingerprint": policy_fingerprint,
        "row_fingerprint": "",
    }
    row["row_fingerprint"] = canonical_semantic_fingerprint(
        {key: value for key, value in row.items() if key != "row_fingerprint"}
    )
    return row


def _coordinate_quality(
    *,
    latitude: object,
    longitude: object,
    uncertainty: object,
    obscured: bool,
    geospatial_issue: bool,
    country_code: object,
    policy: ReferenceGeographyPrecisionPolicy,
) -> tuple[str, str | None]:
    if obscured:
        return "withheld", "source_coordinates_obscured"
    if geospatial_issue:
        return "invalid", "source_geospatial_issue"
    if latitude is None or longitude is None:
        if country_code is not None:
            return "country_only", "coordinates_missing_country_available"
        return "missing", "coordinates_missing"
    if uncertainty is None:
        return "unknown_precision", "coordinate_uncertainty_missing"
    uncertainty_m = float(uncertainty)
    if uncertainty_m <= policy.local_max_uncertainty_m:
        return "local", None
    if uncertainty_m <= policy.regional_max_uncertainty_m:
        return "regional", None
    if uncertainty_m <= policy.coarse_max_uncertainty_m:
        return "coarse", None
    return "unknown_precision", "coordinate_uncertainty_exceeds_coarse_policy"


def _validate_normalized_row(row: Mapping[str, object]) -> None:
    if row["schema_version"] != NORMALIZED_REFERENCE_GEOGRAPHY_SCHEMA_VERSION:
        raise ValueError("unsupported normalized reference geography schema")
    for field in (
        "registry_version",
        "reference_observation_id",
        "source",
        "source_observation_id",
        "source_snapshot_version",
        "cell_grid_name",
        "cell_grid_version",
        "geography_policy_version",
    ):
        if _required_text(row[field], field) != row[field]:
            raise ValueError(f"{field} is not canonical")
    if not _REFERENCE_OBSERVATION_ID_PATTERN.fullmatch(
        str(row["reference_observation_id"])
    ):
        raise ValueError("reference_observation_id is invalid")
    for field in (
        "source_record_hash",
        "source_query_fingerprint",
        "geography_policy_fingerprint",
        "row_fingerprint",
    ):
        _required_fingerprint(row[field], field)
    if row["observer_id_hash"] is not None:
        _required_fingerprint(row["observer_id_hash"], "observer_id_hash")
    if (row["accepted_taxon_key"] is None) != (row["scientific_name"] is None):
        raise ValueError("accepted taxon key and scientific name must coexist")
    for field in ("country_code", "continent_code"):
        if row[field] is not None:
            _uppercase_optional_code(row[field], field=field)
    if row["observed_date"] is not None and (
        not isinstance(row["observed_date"], date)
        or isinstance(row["observed_date"], datetime)
    ):
        raise ValueError("observed_date must be a date or null")
    resolutions = (
        int(row["coarse_cell_resolution"]),
        int(row["regional_cell_resolution"]),
        int(row["local_cell_resolution"]),
    )
    if resolutions != tuple(sorted(set(resolutions))):
        raise ValueError("reference geography resolutions are not strictly ordered")
    policy = ReferenceGeographyPrecisionPolicy(
        local_max_uncertainty_m=row["local_max_uncertainty_m"],
        regional_max_uncertainty_m=row["regional_max_uncertainty_m"],
        coarse_max_uncertainty_m=row["coarse_max_uncertainty_m"],
        version=str(row["geography_policy_version"]),
    )
    expected_policy_fingerprint = _policy_fingerprint(
        policy=policy,
        resolutions=resolutions,
        grid_name=str(row["cell_grid_name"]),
        grid_version=str(row["cell_grid_version"]),
    )
    if row["geography_policy_fingerprint"] != expected_policy_fingerprint:
        raise ValueError("reference geography policy fingerprint mismatch")
    quality = str(row["coordinate_quality"])
    if quality not in REFERENCE_COORDINATE_QUALITIES:
        raise ValueError("unsupported reference coordinate quality")
    _validate_quality_projection(row, quality=quality, resolutions=resolutions)
    expected_quality, expected_reason = _coordinate_quality(
        latitude=row["latitude"],
        longitude=row["longitude"],
        uncertainty=row["coordinate_uncertainty_m"],
        obscured=bool(row["coordinates_obscured"]),
        geospatial_issue=bool(row["geospatial_issue"]),
        country_code=row["country_code"],
        policy=policy,
    )
    if (quality, row["geography_unavailable_reason"]) != (
        expected_quality,
        expected_reason,
    ):
        raise ValueError("reference coordinate quality conflicts with precision policy")
    payload = dict(row)
    fingerprint = payload.pop("row_fingerprint")
    if fingerprint != canonical_semantic_fingerprint(payload):
        raise ValueError("normalized reference geography fingerprint mismatch")


def _validate_quality_projection(
    row: Mapping[str, object],
    *,
    quality: str,
    resolutions: tuple[int, int, int],
) -> None:
    latitude = row["latitude"]
    longitude = row["longitude"]
    if (latitude is None) != (longitude is None):
        raise ValueError("normalized latitude and longitude must coexist")
    if latitude is not None:
        GeographicCoordinate(
            latitude=float(latitude),
            longitude=float(longitude),
            coordinate_uncertainty_m=row["coordinate_uncertainty_m"],
        )
    cells = (
        row["coarse_cell_id"],
        row["regional_cell_id"],
        row["local_cell_id"],
    )
    for cell in cells:
        if cell is not None:
            _required_text(cell, "cell identifier")
    supported = row["supported_cell_resolution"]
    reason = row["geography_unavailable_reason"]
    if quality == "local":
        if latitude is None or any(value is None for value in cells):
            raise ValueError("local geography requires coordinates and all cell levels")
        if supported != resolutions[2]:
            raise ValueError("local geography supported resolution differs")
    elif quality == "regional":
        if (
            latitude is None
            or cells[0] is None
            or cells[1] is None
            or cells[2] is not None
        ):
            raise ValueError("regional geography has invalid cell levels")
        if supported != resolutions[1]:
            raise ValueError("regional geography supported resolution differs")
    elif quality == "coarse":
        if (
            latitude is None
            or cells[0] is None
            or any(value is not None for value in cells[1:])
        ):
            raise ValueError("coarse geography has invalid cell levels")
        if supported != resolutions[0]:
            raise ValueError("coarse geography supported resolution differs")
    elif quality == "unknown_precision":
        if (
            latitude is None
            or any(value is not None for value in cells)
            or supported is not None
        ):
            raise ValueError("unknown-precision geography cannot claim cells")
    else:
        if (
            latitude is not None
            or any(value is not None for value in cells)
            or supported is not None
        ):
            raise ValueError("non-geographic state cannot claim coordinates or cells")
    if quality in _GEO_QUALITIES:
        if reason is not None:
            raise ValueError("usable geography cannot have an unavailable reason")
    elif _optional_text(reason) is None:
        raise ValueError("unavailable geography requires an exact reason")
    if row["coordinates_obscured"] and quality != "withheld":
        raise ValueError("obscured coordinates must remain withheld")
    if (
        row["geospatial_issue"]
        and not row["coordinates_obscured"]
        and quality != "invalid"
    ):
        raise ValueError("geospatial issues must remain invalid")
    if quality == "country_only" and row["country_code"] is None:
        raise ValueError("country-only geography requires country_code")


def _context_by_observation(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    if isinstance(rows, str | bytes) or not isinstance(rows, Sequence):
        raise TypeError("reference geography context must be a sequence of mappings")
    output: dict[str, dict[str, object]] = {}
    for position, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise TypeError(
                f"reference geography context row {position} must be a mapping"
            )
        if set(source) != _CONTEXT_FIELDS:
            raise ValueError(
                "reference geography context fields differ from the contract"
            )
        observation_id = _required_text(
            source["reference_observation_id"], "reference_observation_id"
        )
        if observation_id in output:
            raise ValueError("reference geography context duplicates an observation")
        output[observation_id] = {
            "continent_code": _uppercase_optional_code(
                source["continent_code"], field="continent_code"
            ),
            "admin1": _optional_text(source["admin1"]),
            "bioregion": _optional_text(source["bioregion"]),
        }
    return output


def _policy_fingerprint(
    *,
    policy: ReferenceGeographyPrecisionPolicy,
    resolutions: Sequence[int],
    grid_name: str,
    grid_version: str,
) -> str:
    return canonical_semantic_fingerprint(
        {
            "policy_version": policy.version,
            "local_max_uncertainty_m": policy.local_max_uncertainty_m,
            "regional_max_uncertainty_m": policy.regional_max_uncertainty_m,
            "coarse_max_uncertainty_m": policy.coarse_max_uncertainty_m,
            "cell_grid_name": grid_name,
            "cell_grid_version": grid_version,
            "resolutions": list(resolutions),
        }
    )


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonblank string")
    return value.strip()


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return _required_text(value, "optional text")


def _uppercase_optional_code(value: object, *, field: str) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    code = text.upper()
    if len(code) != 2 or not code.isalpha():
        raise ValueError(f"{field} must be a two-letter code")
    return code


def _required_fingerprint(value: object, field: str) -> str:
    text = _required_text(value, field).casefold()
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a sha256: fingerprint")
    return text


__all__ = [
    "NORMALIZED_REFERENCE_GEOGRAPHY_FILE",
    "NORMALIZED_REFERENCE_GEOGRAPHY_SCHEMA_VERSION",
    "REFERENCE_COORDINATE_QUALITIES",
    "REFERENCE_GEOGRAPHY_PRECISION_POLICY_VERSION",
    "ReferenceGeographyPrecisionPolicy",
    "build_normalized_reference_geography",
    "normalized_reference_geography_artifact_fingerprint",
    "normalized_reference_geography_schema",
    "reference_geography_policy_fingerprint",
    "validate_normalized_reference_geography",
    "write_normalized_reference_geography",
]
