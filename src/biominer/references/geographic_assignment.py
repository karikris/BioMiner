"""Assign globally fetched reference observations to Flickr workload clusters."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import polars as pl

from biominer.flickr_fetch.geographic_clustering import (
    GLOBAL_FALLBACK_CLUSTER_IDS,
    UNASSIGNED_GEO_CLUSTER_ID,
)
from biominer.geography import (
    CellGrid,
    GeographicCoordinate,
    default_cell_grid,
    great_circle_distance_km,
)
from biominer.references.schemas import (
    reference_observations_frame,
    validate_reference_observations,
)


REFERENCE_GEOGRAPHIC_ASSIGNMENT_POLICY_VERSION = (
    "reference-flickr-cell-assignment-v1.0.0"
)


@dataclass(frozen=True, slots=True)
class ReferenceGeographicAssignmentConfig:
    source_resolution: int = 3
    adjacency_grid_distance: int = 1
    maximum_assignment_distance_km: float = 250.0
    policy_version: str = REFERENCE_GEOGRAPHIC_ASSIGNMENT_POLICY_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_resolution, bool)
            or not isinstance(self.source_resolution, int)
            or not 0 <= self.source_resolution <= 15
        ):
            raise ValueError("source_resolution must be an integer from 0 to 15")
        if (
            isinstance(self.adjacency_grid_distance, bool)
            or not isinstance(self.adjacency_grid_distance, int)
            or self.adjacency_grid_distance < 0
        ):
            raise ValueError("adjacency_grid_distance must be a nonnegative integer")
        distance = _positive_finite(
            self.maximum_assignment_distance_km,
            field="maximum_assignment_distance_km",
        )
        policy = str(self.policy_version or "").strip()
        if not policy:
            raise ValueError("policy_version must be nonblank")
        object.__setattr__(self, "maximum_assignment_distance_km", distance)
        object.__setattr__(self, "policy_version", policy)


def assign_reference_observations_to_flickr_clusters(
    observations: pl.DataFrame,
    clusters: pl.DataFrame,
    *,
    config: ReferenceGeographicAssignmentConfig | None = None,
    grid: CellGrid | None = None,
) -> pl.DataFrame:
    """Return the same reference schema with deterministic cluster assignments."""
    if not isinstance(observations, pl.DataFrame):
        raise TypeError("observations must be a Polars DataFrame")
    if not isinstance(clusters, pl.DataFrame):
        raise TypeError("clusters must be a Polars DataFrame")
    validate_reference_observations(observations)
    effective = config or ReferenceGeographicAssignmentConfig()
    if not isinstance(effective, ReferenceGeographicAssignmentConfig):
        raise TypeError("config must be a ReferenceGeographicAssignmentConfig")
    backend = grid or default_cell_grid()
    direct, medoids = _cluster_indexes(
        clusters,
        source_resolution=effective.source_resolution,
        grid=backend,
    )

    assigned: list[dict[str, object]] = []
    for row in observations.iter_rows(named=True):
        updated = dict(row)
        coordinate = _coordinate(row)
        selected: tuple[str, float, int] | None = None
        if coordinate is not None:
            cell_id = backend.coordinate_to_cell(
                coordinate,
                resolution=effective.source_resolution,
            )
            cluster_id = direct.get(cell_id)
            if cluster_id is not None:
                selected = (
                    cluster_id,
                    great_circle_distance_km(coordinate, medoids[cluster_id]),
                    0,
                )
            elif effective.adjacency_grid_distance:
                candidates = {
                    direct[neighbour]
                    for neighbour in backend.neighbours(
                        cell_id,
                        grid_distance=effective.adjacency_grid_distance,
                    )
                    if neighbour in direct
                }
                ranked = sorted(
                    (
                        great_circle_distance_km(coordinate, medoids[candidate]),
                        candidate,
                    )
                    for candidate in candidates
                )
                if ranked and ranked[0][0] <= effective.maximum_assignment_distance_km:
                    selected = (ranked[0][1], ranked[0][0], 1)
        if selected is None:
            updated["geo_cluster_id"] = UNASSIGNED_GEO_CLUSTER_ID
            updated["distance_to_cluster_medoid_km"] = None
            updated["fallback_level"] = 3
        else:
            updated["geo_cluster_id"] = selected[0]
            updated["distance_to_cluster_medoid_km"] = selected[1]
            updated["fallback_level"] = selected[2]
        assigned.append(updated)
    return reference_observations_frame(assigned)


def _cluster_indexes(
    clusters: pl.DataFrame,
    *,
    source_resolution: int,
    grid: CellGrid,
) -> tuple[dict[str, str], dict[str, GeographicCoordinate]]:
    required = {
        "geo_cluster_id",
        "member_cell_ids",
        "medoid",
        "source_resolution",
        "candidate_distribution_only",
    }
    missing = sorted(required - set(clusters.columns))
    if missing:
        raise ValueError(f"Flickr clusters are missing columns: {missing}")
    direct: dict[str, str] = {}
    medoids: dict[str, GeographicCoordinate] = {}
    for row in clusters.sort("geo_cluster_id").iter_rows(named=True):
        cluster_id = str(row["geo_cluster_id"] or "").strip()
        if not cluster_id:
            raise ValueError("Flickr cluster ID must be nonblank")
        if cluster_id in GLOBAL_FALLBACK_CLUSTER_IDS:
            continue
        if row["candidate_distribution_only"] is not True:
            raise ValueError("Flickr clusters must be candidate distributions")
        if row["source_resolution"] != source_resolution:
            raise ValueError(
                f"Flickr cluster {cluster_id} source resolution does not match "
                f"configured resolution {source_resolution}"
            )
        medoid = row["medoid"]
        if not isinstance(medoid, dict):
            raise ValueError(f"Flickr cluster {cluster_id} has no medoid")
        medoids[cluster_id] = GeographicCoordinate(
            latitude=medoid.get("latitude"),
            longitude=medoid.get("longitude"),
        )
        cells = row["member_cell_ids"]
        if not isinstance(cells, list) or not cells:
            raise ValueError(f"Flickr cluster {cluster_id} has no member cells")
        for raw_cell in cells:
            cell_id = str(raw_cell or "").strip()
            if not grid.is_valid(cell_id):
                raise ValueError(f"Flickr cluster {cluster_id} has invalid cell")
            previous = direct.setdefault(cell_id, cluster_id)
            if previous != cluster_id:
                raise ValueError(f"Flickr cell {cell_id} belongs to multiple clusters")
    if not direct:
        raise ValueError("Flickr clusters contain no located member cells")
    return direct, medoids


def _coordinate(row: dict[str, Any]) -> GeographicCoordinate | None:
    latitude = row.get("latitude")
    longitude = row.get("longitude")
    if latitude is None or longitude is None or row.get("coordinates_obscured") is True:
        return None
    try:
        return GeographicCoordinate(latitude=float(latitude), longitude=float(longitude))
    except (TypeError, ValueError) as exc:
        raise ValueError("reference observation contains invalid coordinates") from exc


def _positive_finite(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive finite number")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be a positive finite number")
    return result


__all__ = [
    "REFERENCE_GEOGRAPHIC_ASSIGNMENT_POLICY_VERSION",
    "ReferenceGeographicAssignmentConfig",
    "assign_reference_observations_to_flickr_clusters",
]
