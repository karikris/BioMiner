from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Protocol, runtime_checkable

import polars as pl

from biominer.references.schemas import (
    validate_reference_media_candidates,
    validate_reference_observations,
)


REFERENCE_SOURCE_QUERY_VERSION = "reference-source-query-v1"
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
        object.__setattr__(self, "spatial_cell_ids", cells)
        object.__setattr__(self, "country_codes", countries)

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


__all__ = [
    "REFERENCE_SOURCE_PAGE_VERSION",
    "REFERENCE_SOURCE_QUERY_VERSION",
    "ReferenceMetadataPage",
    "ReferenceSourceAdapter",
    "ReferenceSourceQuery",
    "validate_source_adapter",
]
