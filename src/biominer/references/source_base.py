from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Protocol, runtime_checkable

import polars as pl

from biominer.references.schemas import (
    validate_reference_media_candidates,
    validate_reference_observations,
)


REFERENCE_SOURCE_QUERY_VERSION = "reference-source-query-v2"
REFERENCE_SOURCE_PAGE_VERSION = "reference-source-page-v1"


@dataclass(frozen=True, slots=True)
class ReferenceSourceQuery:
    accepted_taxon_key: str
    scientific_name: str
    geo_cluster_id: str
    fallback_level: int
    source_taxon_id: str | None = None
    spatial_cell_ids: tuple[str, ...] = ()
    country_codes: tuple[str, ...] = ()
    source_place_ids: tuple[str, ...] = ()
    geometry_wkt: str | None = None
    bounding_box: tuple[float, float, float, float] | None = None
    cluster_medoid_latitude: float | None = None
    cluster_medoid_longitude: float | None = None
    page_size: int = 100
    source_snapshot_version: str = ""

    def __post_init__(self) -> None:
        for field in (
            "accepted_taxon_key",
            "scientific_name",
            "geo_cluster_id",
            "source_snapshot_version",
        ):
            value = str(getattr(self, field) or "").strip()
            if not value:
                raise ValueError(f"{field} must be nonblank")
            object.__setattr__(self, field, value)
        source_taxon_id = str(self.source_taxon_id or "").strip() or None
        object.__setattr__(self, "source_taxon_id", source_taxon_id)
        geometry_wkt = str(self.geometry_wkt or "").strip() or None
        object.__setattr__(self, "geometry_wkt", geometry_wkt)
        if isinstance(self.fallback_level, bool) or not isinstance(
            self.fallback_level, int
        ):
            raise TypeError("fallback_level must be an integer")
        if not 0 <= self.fallback_level <= 255:
            raise ValueError("fallback_level must be between 0 and 255")
        if isinstance(self.page_size, bool) or not isinstance(self.page_size, int):
            raise TypeError("page_size must be an integer")
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        cells = _normalized_values(self.spatial_cell_ids, field="spatial_cell_ids")
        countries = tuple(
            sorted(
                {
                    value.upper()
                    for value in _normalized_values(
                        self.country_codes,
                        field="country_codes",
                    )
                }
            )
        )
        if any(len(value) != 2 for value in countries):
            raise ValueError("country_codes must contain ISO alpha-2 values")
        source_place_ids = _normalized_values(
            self.source_place_ids,
            field="source_place_ids",
        )
        bounding_box = _bounding_box(self.bounding_box)
        medoid_latitude, medoid_longitude = _coordinate_pair(
            self.cluster_medoid_latitude,
            self.cluster_medoid_longitude,
        )
        object.__setattr__(self, "spatial_cell_ids", cells)
        object.__setattr__(self, "country_codes", countries)
        object.__setattr__(self, "source_place_ids", source_place_ids)
        object.__setattr__(self, "bounding_box", bounding_box)
        object.__setattr__(self, "cluster_medoid_latitude", medoid_latitude)
        object.__setattr__(self, "cluster_medoid_longitude", medoid_longitude)

    @property
    def query_fingerprint(self) -> str:
        return _fingerprint(
            {
                "contract_version": REFERENCE_SOURCE_QUERY_VERSION,
                "accepted_taxon_key": self.accepted_taxon_key,
                "scientific_name": self.scientific_name,
                "geo_cluster_id": self.geo_cluster_id,
                "fallback_level": self.fallback_level,
                "source_taxon_id": self.source_taxon_id,
                "spatial_cell_ids": self.spatial_cell_ids,
                "country_codes": self.country_codes,
                "source_place_ids": self.source_place_ids,
                "geometry_wkt": self.geometry_wkt,
                "bounding_box": self.bounding_box,
                "cluster_medoid_latitude": self.cluster_medoid_latitude,
                "cluster_medoid_longitude": self.cluster_medoid_longitude,
                "page_size": self.page_size,
                "source_snapshot_version": self.source_snapshot_version,
            }
        )


@dataclass(frozen=True, slots=True)
class ReferenceMetadataPage:
    source: str
    source_version: str
    query_fingerprint: str
    page_cursor: str | None
    next_cursor: str | None
    observations: pl.DataFrame
    media_candidates: pl.DataFrame
    request_count: int
    retry_count: int
    rate_limit_count: int
    complete: bool
    contract_version: str = REFERENCE_SOURCE_PAGE_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != REFERENCE_SOURCE_PAGE_VERSION:
            raise ValueError("unsupported reference source page contract")
        for field in ("source", "source_version", "query_fingerprint"):
            value = str(getattr(self, field) or "").strip()
            if not value:
                raise ValueError(f"{field} must be nonblank")
            object.__setattr__(self, field, value)
        if not self.query_fingerprint.startswith("sha256:") or len(
            self.query_fingerprint
        ) != 71:
            raise ValueError("query_fingerprint must be a full sha256 digest")
        for field in ("request_count", "retry_count", "rate_limit_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field} must be an integer")
            if value < 0:
                raise ValueError(f"{field} must be nonnegative")
        if not isinstance(self.complete, bool):
            raise TypeError("complete must be boolean")
        if self.complete and self.next_cursor is not None:
            raise ValueError("a complete page cannot have a next cursor")
        if not self.complete and not str(self.next_cursor or "").strip():
            raise ValueError("an incomplete page requires a next cursor")
        validate_reference_observations(self.observations)
        validate_reference_media_candidates(self.media_candidates)
        observation_ids = set(self.observations["reference_observation_id"].to_list())
        missing_observations = sorted(
            set(self.media_candidates["reference_observation_id"].to_list())
            - observation_ids
        )
        if missing_observations:
            raise ValueError(
                "media candidates reference observations absent from the page: "
                f"{missing_observations}"
            )
        page_sources = set(self.observations["source"].to_list()) | set(
            self.media_candidates["source"].to_list()
        )
        if page_sources - {self.source}:
            raise ValueError("reference metadata page contains rows from another source")


@runtime_checkable
class ReferenceSourceAdapter(Protocol):
    source: str
    source_version: str
    user_agent: str

    def fetch_page(
        self,
        query: ReferenceSourceQuery,
        *,
        cursor: str | None = None,
    ) -> ReferenceMetadataPage: ...


def validate_source_adapter(adapter: ReferenceSourceAdapter) -> None:
    if not isinstance(adapter, ReferenceSourceAdapter):
        raise TypeError("adapter does not satisfy ReferenceSourceAdapter")
    for field in ("source", "source_version", "user_agent"):
        if not str(getattr(adapter, field, "") or "").strip():
            raise ValueError(f"reference source adapter {field} must be nonblank")


def _normalized_values(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field} must be a tuple")
    normalized = tuple(sorted({str(value or "").strip() for value in values}))
    if "" in normalized:
        raise ValueError(f"{field} cannot contain blank values")
    return normalized


def _fingerprint(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _coordinate_pair(
    latitude: object,
    longitude: object,
) -> tuple[float | None, float | None]:
    if (latitude is None) != (longitude is None):
        raise ValueError("cluster medoid latitude and longitude must be populated together")
    if latitude is None:
        return None, None
    if isinstance(latitude, bool) or isinstance(longitude, bool):
        raise TypeError("cluster medoid coordinates must be numeric")
    lat = float(latitude)
    lon = float(longitude)
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise ValueError("cluster medoid coordinates must be finite")
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise ValueError("cluster medoid coordinates are outside WGS84 bounds")
    return lat, lon


def _bounding_box(
    value: object,
) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 4:
        raise TypeError("bounding_box must be a four-value tuple")
    if any(isinstance(item, bool) for item in value):
        raise TypeError("bounding_box values must be numeric")
    south, west, north, east = (float(item) for item in value)
    if not all(math.isfinite(item) for item in (south, west, north, east)):
        raise ValueError("bounding_box values must be finite")
    if not -90.0 <= south < north <= 90.0:
        raise ValueError("bounding_box latitude bounds are invalid")
    if not -180.0 <= west < east <= 180.0:
        raise ValueError("bounding_box longitude bounds are invalid")
    return south, west, north, east


__all__ = [
    "REFERENCE_SOURCE_PAGE_VERSION",
    "REFERENCE_SOURCE_QUERY_VERSION",
    "ReferenceMetadataPage",
    "ReferenceSourceAdapter",
    "ReferenceSourceQuery",
    "validate_source_adapter",
]
