"""Deterministic clusters over Flickr candidate-distribution geography."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import polars as pl

from biominer.flickr_fetch.geography import FLICKR_GEOGRAPHY_SCHEMA_VERSION
from biominer.geography import (
    CellGrid,
    GeographicCoordinate,
    default_cell_grid,
    great_circle_distance_km,
)
from biominer.storage.parquet import write_parquet


FLICKR_GEO_CLUSTERS_SCHEMA_VERSION = "flickr-geo-clusters-v1.0.0"
FLICKR_GEO_ASSIGNMENTS_SCHEMA_VERSION = "flickr-geo-assignments-v1.0.0"
FLICKR_GEO_CLUSTER_METHOD = "h3-density-components-v1.0.0"
FLICKR_GEO_CLUSTERS_FILE = "flickr_geo_clusters.parquet"
FLICKR_GEO_ASSIGNMENTS_FILE = "flickr_geo_assignments.parquet"
NO_GEO_CLUSTER_ID = "no_geo"

_CELL_FIELDS = frozenset({"coarse_cell_id", "regional_cell_id", "local_cell_id"})


@dataclass(frozen=True, slots=True)
class FlickrGeoClusterConfig:
    source_cell_field: str = "regional_cell_id"
    source_resolution: int = 5
    adjacency_grid_distance: int = 1
    minimum_images_per_cell: int = 1
    minimum_cluster_images: int = 2
    maximum_assignment_distance_km: float = 250.0
    bioregion_by_admin_region: tuple[tuple[str, str], ...] = ()
    cluster_method: str = FLICKR_GEO_CLUSTER_METHOD

    def __post_init__(self) -> None:
        field_name = str(self.source_cell_field).strip()
        if field_name not in _CELL_FIELDS:
            raise ValueError(
                "source_cell_field must be coarse_cell_id, regional_cell_id, or local_cell_id"
            )
        resolution = _nonnegative_int(self.source_resolution, field_name="source_resolution")
        if resolution > 15:
            raise ValueError("source_resolution must be between 0 and 15")
        adjacency = _nonnegative_int(
            self.adjacency_grid_distance,
            field_name="adjacency_grid_distance",
        )
        density = _positive_int(
            self.minimum_images_per_cell,
            field_name="minimum_images_per_cell",
        )
        minimum_cluster = _positive_int(
            self.minimum_cluster_images,
            field_name="minimum_cluster_images",
        )
        maximum_distance = _positive_float(
            self.maximum_assignment_distance_km,
            field_name="maximum_assignment_distance_km",
        )
        method = _required_text(self.cluster_method, field_name="cluster_method")
        bioregions = _normalize_bioregion_map(self.bioregion_by_admin_region)

        object.__setattr__(self, "source_cell_field", field_name)
        object.__setattr__(self, "source_resolution", resolution)
        object.__setattr__(self, "adjacency_grid_distance", adjacency)
        object.__setattr__(self, "minimum_images_per_cell", density)
        object.__setattr__(self, "minimum_cluster_images", minimum_cluster)
        object.__setattr__(self, "maximum_assignment_distance_km", maximum_distance)
        object.__setattr__(self, "bioregion_by_admin_region", bioregions)
        object.__setattr__(self, "cluster_method", method)


@dataclass(frozen=True, slots=True)
class FlickrGeoClusterBuildResult:
    clusters: pl.DataFrame
    assignments: pl.DataFrame
    cluster_configuration_hash: str


@dataclass(frozen=True, slots=True)
class _ClusterDefinition:
    cluster_id: str
    member_cells: tuple[str, ...]
    adjacency_cells: tuple[str, ...]
    centroid: GeographicCoordinate
    medoid: GeographicCoordinate
    member_rows: tuple[Mapping[str, Any], ...]


def flickr_geo_clusters_schema() -> dict[str, pl.DataType]:
    coordinate = pl.Struct({"latitude": pl.Float64, "longitude": pl.Float64})
    return {
        "schema_version": pl.String,
        "geo_cluster_id": pl.String,
        "target_accepted_taxon_key": pl.String,
        "member_image_count": pl.UInt64,
        "member_cell_count": pl.UInt64,
        "member_cell_ids": pl.List(pl.String),
        "centroid": coordinate,
        "medoid": coordinate,
        "radius_quantiles_km": pl.Struct(
            {
                "p50": pl.Float64,
                "p90": pl.Float64,
                "p95": pl.Float64,
                "max": pl.Float64,
            }
        ),
        "bounding_geometry": pl.Struct(
            {
                "south": pl.Float64,
                "north": pl.Float64,
                "west": pl.Float64,
                "east": pl.Float64,
                "crosses_dateline": pl.Boolean,
            }
        ),
        "countries": pl.List(pl.String),
        "admin_regions": pl.List(pl.String),
        "source_resolution": pl.UInt8,
        "cluster_method": pl.String,
        "cluster_configuration_hash": pl.String,
        "candidate_distribution_only": pl.Boolean,
        "created_at": pl.Datetime("us", "UTC"),
    }


def flickr_geo_assignments_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "source": pl.String,
        "flickr_photo_id": pl.String,
        "source_record_hash": pl.String,
        "target_accepted_taxon_key": pl.String,
        "geo_cluster_id": pl.String,
        "distance_to_medoid_km": pl.Float64,
        "assignment_method": pl.String,
        "coordinate_quality": pl.String,
        "fallback_scope": pl.String,
        "outlier": pl.Boolean,
        "cluster_configuration_hash": pl.String,
    }


def build_flickr_geo_clusters(
    geography: pl.DataFrame,
    *,
    target_accepted_taxon_key: str,
    config: FlickrGeoClusterConfig | None = None,
    created_at: str | datetime | None = None,
    grid: CellGrid | None = None,
) -> FlickrGeoClusterBuildResult:
    if not isinstance(geography, pl.DataFrame):
        raise TypeError("geography must be a Polars DataFrame")
    target_key = _required_text(
        target_accepted_taxon_key,
        field_name="target_accepted_taxon_key",
    )
    effective_config = config or FlickrGeoClusterConfig()
    if not isinstance(effective_config, FlickrGeoClusterConfig):
        raise TypeError("config must be a FlickrGeoClusterConfig")
    backend = grid or default_cell_grid()
    built_at = _utc_datetime(created_at or datetime.now(UTC))
    rows, geography_fingerprint = _validated_geography_rows(
        geography,
        config=effective_config,
        grid=backend,
    )
    configuration_hash = cluster_configuration_hash(
        effective_config,
        target_accepted_taxon_key=target_key,
        geography_config_fingerprint=geography_fingerprint,
        grid=backend,
    )

    rows_by_cell: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        cell_id = _optional_text(row.get(effective_config.source_cell_field))
        if cell_id:
            rows_by_cell[cell_id].append(row)
    core_cells = {
        cell_id
        for cell_id, members in rows_by_cell.items()
        if len(members) >= effective_config.minimum_images_per_cell
    }
    components = _connected_components(
        core_cells,
        grid=backend,
        grid_distance=effective_config.adjacency_grid_distance,
    )
    retained_components = [
        component
        for component in components
        if sum(len(rows_by_cell[cell_id]) for cell_id in component)
        >= effective_config.minimum_cluster_images
    ]
    core_definitions = tuple(
        _cluster_definition(
            component,
            rows_by_cell=rows_by_cell,
            target_key=target_key,
            configuration_hash=configuration_hash,
            grid=backend,
        )
        for component in retained_components
    )
    definitions = _expand_cluster_definitions(
        core_definitions,
        rows_by_cell=rows_by_cell,
        target_key=target_key,
        configuration_hash=configuration_hash,
        config=effective_config,
        grid=backend,
    )

    direct_cluster_by_cell = {
        cell_id: definition.cluster_id
        for definition in definitions
        for cell_id in definition.member_cells
    }
    assignment_method_by_cell = {
        cell_id: (
            "adjacency"
            if cell_id in definition.adjacency_cells
            else effective_config.source_cell_field.removesuffix("_id")
        )
        for definition in definitions
        for cell_id in definition.member_cells
    }
    definition_by_id = {definition.cluster_id: definition for definition in definitions}
    country_clusters, bioregion_clusters = _fallback_cluster_indexes(
        definitions,
        config=effective_config,
    )

    assignments = [
        _assign_row(
            row,
            target_key=target_key,
            config=effective_config,
            configuration_hash=configuration_hash,
            direct_cluster_by_cell=direct_cluster_by_cell,
            assignment_method_by_cell=assignment_method_by_cell,
            definition_by_id=definition_by_id,
            country_clusters=country_clusters,
            bioregion_clusters=bioregion_clusters,
        )
        for row in rows
    ]
    assignments.sort(key=lambda row: (str(row["source"]), str(row["flickr_photo_id"])))
    assignment_frame = _frame(assignments, schema=flickr_geo_assignments_schema())

    assigned_rows: dict[str, list[tuple[Mapping[str, Any], Mapping[str, object]]]] = defaultdict(list)
    rows_by_identity = {
        (str(row["source"]), str(row["flickr_photo_id"])): row for row in rows
    }
    for assignment in assignments:
        identity = (str(assignment["source"]), str(assignment["flickr_photo_id"]))
        assigned_rows[str(assignment["geo_cluster_id"])].append(
            (rows_by_identity[identity], assignment)
        )

    cluster_rows = [
        _cluster_output_row(
            definition,
            assigned=assigned_rows.get(definition.cluster_id, []),
            target_key=target_key,
            config=effective_config,
            configuration_hash=configuration_hash,
            created_at=built_at,
        )
        for definition in definitions
    ]
    if assigned_rows.get(NO_GEO_CLUSTER_ID):
        cluster_rows.append(
            _no_geo_cluster_row(
                assigned=assigned_rows[NO_GEO_CLUSTER_ID],
                target_key=target_key,
                config=effective_config,
                configuration_hash=configuration_hash,
                created_at=built_at,
            )
        )
    cluster_rows.sort(
        key=lambda row: (str(row["target_accepted_taxon_key"]), str(row["geo_cluster_id"]))
    )
    cluster_frame = _frame(cluster_rows, schema=flickr_geo_clusters_schema())
    return FlickrGeoClusterBuildResult(
        clusters=cluster_frame,
        assignments=assignment_frame,
        cluster_configuration_hash=configuration_hash,
    )


def cluster_configuration_hash(
    config: FlickrGeoClusterConfig,
    *,
    target_accepted_taxon_key: str,
    geography_config_fingerprint: str,
    grid: CellGrid | None = None,
) -> str:
    if not isinstance(config, FlickrGeoClusterConfig):
        raise TypeError("config must be a FlickrGeoClusterConfig")
    backend = grid or default_cell_grid()
    payload = {
        "adjacency_grid_distance": config.adjacency_grid_distance,
        "bioregion_by_admin_region": [list(item) for item in config.bioregion_by_admin_region],
        "cluster_method": config.cluster_method,
        "geography_config_fingerprint": _required_text(
            geography_config_fingerprint,
            field_name="geography_config_fingerprint",
        ),
        "grid_name": backend.name,
        "grid_version": backend.version,
        "maximum_assignment_distance_km": config.maximum_assignment_distance_km,
        "minimum_cluster_images": config.minimum_cluster_images,
        "minimum_images_per_cell": config.minimum_images_per_cell,
        "source_cell_field": config.source_cell_field,
        "source_resolution": config.source_resolution,
        "target_accepted_taxon_key": _required_text(
            target_accepted_taxon_key,
            field_name="target_accepted_taxon_key",
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def write_flickr_geo_cluster_artifacts(
    result: FlickrGeoClusterBuildResult,
    output_dir: str | Path,
    *,
    overwrite: bool = True,
) -> dict[str, Path]:
    if not isinstance(result, FlickrGeoClusterBuildResult):
        raise TypeError("result must be a FlickrGeoClusterBuildResult")
    output = Path(output_dir)
    cluster_path = output / FLICKR_GEO_CLUSTERS_FILE
    assignment_path = output / FLICKR_GEO_ASSIGNMENTS_FILE
    write_parquet(result.clusters, cluster_path, overwrite=overwrite)
    write_parquet(result.assignments, assignment_path, overwrite=overwrite)
    return {"clusters": cluster_path, "assignments": assignment_path}


def _validated_geography_rows(
    geography: pl.DataFrame,
    *,
    config: FlickrGeoClusterConfig,
    grid: CellGrid,
) -> tuple[list[dict[str, Any]], str]:
    required = {
        "schema_version",
        "source",
        "flickr_photo_id",
        "source_record_hash",
        "latitude",
        "longitude",
        "geotag_available",
        "country_code",
        "admin1",
        config.source_cell_field,
        "coordinate_quality",
        "geography_config_fingerprint",
    }
    missing = sorted(required - set(geography.columns))
    if missing:
        raise ValueError(f"Flickr geography is missing required columns: {missing}")
    rows = geography.to_dicts()
    rows.sort(key=lambda row: (str(row["source"]), str(row["flickr_photo_id"])))
    seen: set[tuple[str, str]] = set()
    fingerprints: set[str] = set()
    for row in rows:
        if row.get("schema_version") != FLICKR_GEOGRAPHY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported Flickr geography schema version: {row.get('schema_version')!r}"
            )
        identity = (
            _required_text(row.get("source"), field_name="source"),
            _required_text(row.get("flickr_photo_id"), field_name="flickr_photo_id"),
        )
        if identity in seen:
            raise ValueError(f"duplicate Flickr geography identity: {identity[0]}:{identity[1]}")
        seen.add(identity)
        _required_text(row.get("source_record_hash"), field_name="source_record_hash")
        fingerprint = _required_text(
            row.get("geography_config_fingerprint"),
            field_name="geography_config_fingerprint",
        )
        fingerprints.add(fingerprint)
        cell_id = _optional_text(row.get(config.source_cell_field))
        if cell_id and not grid.is_valid(cell_id):
            raise ValueError(
                f"invalid {config.source_cell_field} for {identity[0]}:{identity[1]}: {cell_id}"
            )
        if row.get("geotag_available") is True:
            _row_coordinate(row)
    if not rows:
        raise ValueError("Flickr geography must contain at least one candidate record")
    if len(fingerprints) != 1:
        raise ValueError("Flickr geography rows must share one geography_config_fingerprint")
    return rows, next(iter(fingerprints))


def _connected_components(
    cells: set[str],
    *,
    grid: CellGrid,
    grid_distance: int,
) -> tuple[tuple[str, ...], ...]:
    remaining = set(cells)
    components: list[tuple[str, ...]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        frontier = [start]
        while frontier:
            current = frontier.pop()
            adjacent = set(
                grid.neighbours(
                    current,
                    grid_distance=grid_distance,
                    include_origin=False,
                )
            )
            discovered = sorted(adjacent & remaining, reverse=True)
            for cell_id in discovered:
                remaining.remove(cell_id)
                component.add(cell_id)
                frontier.append(cell_id)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def _cluster_definition(
    member_cells: tuple[str, ...],
    *,
    rows_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
    target_key: str,
    configuration_hash: str,
    grid: CellGrid,
    adjacency_cells: tuple[str, ...] = (),
) -> _ClusterDefinition:
    weighted_centers = [
        (cell_id, grid.center(cell_id), len(rows_by_cell[cell_id]))
        for cell_id in member_cells
    ]
    centroid = _spherical_centroid(
        [(coordinate, weight) for _cell_id, coordinate, weight in weighted_centers]
    )
    _medoid_cell, medoid, _weight = min(
        weighted_centers,
        key=lambda item: (
            great_circle_distance_km(item[1], centroid),
            item[0],
        ),
    )
    cluster_id = _cluster_id(
        member_cells,
        target_key=target_key,
        configuration_hash=configuration_hash,
    )
    core_rows = tuple(
        row for cell_id in member_cells for row in rows_by_cell[cell_id]
    )
    return _ClusterDefinition(
        cluster_id=cluster_id,
        member_cells=member_cells,
        adjacency_cells=adjacency_cells,
        centroid=centroid,
        medoid=medoid,
        member_rows=core_rows,
    )


def _cluster_id(
    member_cells: tuple[str, ...],
    *,
    target_key: str,
    configuration_hash: str,
) -> str:
    payload = json.dumps(
        {
            "cluster_configuration_hash": configuration_hash,
            "member_cell_ids": list(member_cells),
            "target_accepted_taxon_key": target_key,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "geo:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _expand_cluster_definitions(
    definitions: Sequence[_ClusterDefinition],
    *,
    rows_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
    target_key: str,
    configuration_hash: str,
    config: FlickrGeoClusterConfig,
    grid: CellGrid,
) -> tuple[_ClusterDefinition, ...]:
    if not definitions:
        return ()
    index: dict[str, set[str]] = defaultdict(set)
    member_cells = {
        cell_id for definition in definitions for cell_id in definition.member_cells
    }
    for definition in definitions:
        for member_cell in definition.member_cells:
            for adjacent in grid.neighbours(
                member_cell,
                grid_distance=config.adjacency_grid_distance,
                include_origin=False,
            ):
                if adjacent not in member_cells and adjacent in rows_by_cell:
                    index[adjacent].add(definition.cluster_id)
    definition_by_id = {definition.cluster_id: definition for definition in definitions}
    adjacency_by_cluster: dict[str, set[str]] = defaultdict(set)
    for cell_id in sorted(index):
        candidate_assignments: list[tuple[float, str]] = []
        for cluster_id in sorted(index[cell_id]):
            medoid = definition_by_id[cluster_id].medoid
            maximum_member_distance = max(
                great_circle_distance_km(_row_coordinate(row), medoid)
                for row in rows_by_cell[cell_id]
            )
            if maximum_member_distance <= config.maximum_assignment_distance_km:
                candidate_assignments.append((maximum_member_distance, cluster_id))
        if candidate_assignments:
            _distance, selected_cluster = min(candidate_assignments)
            adjacency_by_cluster[selected_cluster].add(cell_id)

    expanded: list[_ClusterDefinition] = []
    for definition in definitions:
        expanded.append(
            _finalize_adjacency_cells(
                definition,
                adjacency_cells=adjacency_by_cluster.get(definition.cluster_id, set()),
                rows_by_cell=rows_by_cell,
                target_key=target_key,
                configuration_hash=configuration_hash,
                maximum_assignment_distance_km=config.maximum_assignment_distance_km,
                grid=grid,
            )
        )
    return tuple(sorted(expanded, key=lambda item: item.cluster_id))


def _finalize_adjacency_cells(
    core: _ClusterDefinition,
    *,
    adjacency_cells: set[str],
    rows_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
    target_key: str,
    configuration_hash: str,
    maximum_assignment_distance_km: float,
    grid: CellGrid,
) -> _ClusterDefinition:
    accepted = set(adjacency_cells)
    while True:
        border = tuple(sorted(accepted))
        member_cells = tuple(sorted((*core.member_cells, *border)))
        definition = _cluster_definition(
            member_cells,
            rows_by_cell=rows_by_cell,
            target_key=target_key,
            configuration_hash=configuration_hash,
            grid=grid,
            adjacency_cells=border,
        )
        rejected = {
            cell_id
            for cell_id in border
            if max(
                great_circle_distance_km(_row_coordinate(row), definition.medoid)
                for row in rows_by_cell[cell_id]
            )
            > maximum_assignment_distance_km
        }
        if not rejected:
            return definition
        accepted.difference_update(rejected)


def _fallback_cluster_indexes(
    definitions: Sequence[_ClusterDefinition],
    *,
    config: FlickrGeoClusterConfig,
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    country_index: dict[str, set[str]] = defaultdict(set)
    bioregion_index: dict[str, set[str]] = defaultdict(set)
    bioregion_map = dict(config.bioregion_by_admin_region)
    for definition in definitions:
        for row in definition.member_rows:
            country = _optional_text(row.get("country_code"))
            if country:
                country_index[country].add(definition.cluster_id)
            admin_scope = _admin_scope(row)
            bioregion = bioregion_map.get(admin_scope) if admin_scope else None
            if bioregion:
                bioregion_index[bioregion].add(definition.cluster_id)
    return (
        {key: tuple(sorted(value)) for key, value in country_index.items()},
        {key: tuple(sorted(value)) for key, value in bioregion_index.items()},
    )


def _assign_row(
    row: Mapping[str, Any],
    *,
    target_key: str,
    config: FlickrGeoClusterConfig,
    configuration_hash: str,
    direct_cluster_by_cell: Mapping[str, str],
    assignment_method_by_cell: Mapping[str, str],
    definition_by_id: Mapping[str, _ClusterDefinition],
    country_clusters: Mapping[str, tuple[str, ...]],
    bioregion_clusters: Mapping[str, tuple[str, ...]],
) -> dict[str, object]:
    source_cell = _optional_text(row.get(config.source_cell_field))
    cluster_id: str | None = None
    distance: float | None = None
    assignment_method = "no_geo"
    fallback_scope: str | None = None
    outlier = False

    if source_cell and source_cell in direct_cluster_by_cell:
        cluster_id = direct_cluster_by_cell[source_cell]
        distance = great_circle_distance_km(
            _row_coordinate(row),
            definition_by_id[cluster_id].medoid,
        )
        assignment_method = assignment_method_by_cell[source_cell]
    elif source_cell:
        outlier = True
    elif row.get("geotag_available") is True:
        admin_scope = _admin_scope(row)
        bioregion = dict(config.bioregion_by_admin_region).get(admin_scope or "")
        bioregion_candidates = bioregion_clusters.get(bioregion or "", ())
        country = _optional_text(row.get("country_code"))
        country_candidates = country_clusters.get(country or "", ())
        if bioregion and len(bioregion_candidates) == 1:
            cluster_id = bioregion_candidates[0]
            assignment_method = "bioregion"
            fallback_scope = f"bioregion:{bioregion}"
        elif country and len(country_candidates) == 1:
            cluster_id = country_candidates[0]
            assignment_method = "country"
            fallback_scope = f"country:{country}"

    return {
        "schema_version": FLICKR_GEO_ASSIGNMENTS_SCHEMA_VERSION,
        "source": str(row["source"]),
        "flickr_photo_id": str(row["flickr_photo_id"]),
        "source_record_hash": str(row["source_record_hash"]),
        "target_accepted_taxon_key": target_key,
        "geo_cluster_id": cluster_id or NO_GEO_CLUSTER_ID,
        "distance_to_medoid_km": distance,
        "assignment_method": assignment_method,
        "coordinate_quality": str(row["coordinate_quality"]),
        "fallback_scope": fallback_scope,
        "outlier": outlier,
        "cluster_configuration_hash": configuration_hash,
    }


def _cluster_output_row(
    definition: _ClusterDefinition,
    *,
    assigned: Sequence[tuple[Mapping[str, Any], Mapping[str, object]]],
    target_key: str,
    config: FlickrGeoClusterConfig,
    configuration_hash: str,
    created_at: datetime,
) -> dict[str, object]:
    precise = [
        (row, assignment)
        for row, assignment in assigned
        if assignment["assignment_method"]
        in {config.source_cell_field.removesuffix("_id"), "adjacency"}
    ]
    distances = sorted(
        float(assignment["distance_to_medoid_km"])
        for _row, assignment in precise
        if assignment.get("distance_to_medoid_km") is not None
    )
    coordinates = [_row_coordinate(row) for row, _assignment in precise]
    return {
        "schema_version": FLICKR_GEO_CLUSTERS_SCHEMA_VERSION,
        "geo_cluster_id": definition.cluster_id,
        "target_accepted_taxon_key": target_key,
        "member_image_count": len(assigned),
        "member_cell_count": len(definition.member_cells),
        "member_cell_ids": list(definition.member_cells),
        "centroid": _coordinate_struct(definition.centroid),
        "medoid": _coordinate_struct(definition.medoid),
        "radius_quantiles_km": _radius_quantiles(distances),
        "bounding_geometry": _bounding_geometry(coordinates),
        "countries": _countries(assigned),
        "admin_regions": _admin_regions(assigned),
        "source_resolution": config.source_resolution,
        "cluster_method": config.cluster_method,
        "cluster_configuration_hash": configuration_hash,
        "candidate_distribution_only": True,
        "created_at": created_at,
    }


def _no_geo_cluster_row(
    *,
    assigned: Sequence[tuple[Mapping[str, Any], Mapping[str, object]]],
    target_key: str,
    config: FlickrGeoClusterConfig,
    configuration_hash: str,
    created_at: datetime,
) -> dict[str, object]:
    return {
        "schema_version": FLICKR_GEO_CLUSTERS_SCHEMA_VERSION,
        "geo_cluster_id": NO_GEO_CLUSTER_ID,
        "target_accepted_taxon_key": target_key,
        "member_image_count": len(assigned),
        "member_cell_count": 0,
        "member_cell_ids": [],
        "centroid": _coordinate_struct(None),
        "medoid": _coordinate_struct(None),
        "radius_quantiles_km": _radius_quantiles([]),
        "bounding_geometry": _bounding_geometry([]),
        "countries": _countries(assigned),
        "admin_regions": _admin_regions(assigned),
        "source_resolution": None,
        "cluster_method": config.cluster_method,
        "cluster_configuration_hash": configuration_hash,
        "candidate_distribution_only": True,
        "created_at": created_at,
    }


def _countries(
    assigned: Sequence[tuple[Mapping[str, Any], Mapping[str, object]]],
) -> list[str]:
    return sorted(
        {
            country
            for row, _assignment in assigned
            if (country := _optional_text(row.get("country_code")))
        }
    )


def _admin_regions(
    assigned: Sequence[tuple[Mapping[str, Any], Mapping[str, object]]],
) -> list[str]:
    return sorted(
        {
            scope
            for row, _assignment in assigned
            if (scope := _admin_scope(row))
        }
    )


def _admin_scope(row: Mapping[str, Any]) -> str | None:
    admin1 = _optional_text(row.get("admin1"))
    if not admin1:
        return None
    country = _optional_text(row.get("country_code"))
    return f"{country}:{admin1}" if country else admin1


def _spherical_centroid(
    weighted_coordinates: Sequence[tuple[GeographicCoordinate, int]],
) -> GeographicCoordinate:
    if not weighted_coordinates:
        raise ValueError("spherical centroid requires at least one coordinate")
    x = y = z = 0.0
    for coordinate, weight in weighted_coordinates:
        latitude = math.radians(float(coordinate.latitude))
        longitude = math.radians(float(coordinate.longitude))
        x += weight * math.cos(latitude) * math.cos(longitude)
        y += weight * math.cos(latitude) * math.sin(longitude)
        z += weight * math.sin(latitude)
    horizontal = math.hypot(x, y)
    magnitude = math.hypot(horizontal, z)
    if magnitude <= 1e-12:
        return min(
            (coordinate for coordinate, _weight in weighted_coordinates),
            key=lambda coordinate: (float(coordinate.latitude), float(coordinate.longitude)),
        )
    return GeographicCoordinate(
        latitude=math.degrees(math.atan2(z, horizontal)),
        longitude=math.degrees(math.atan2(y, x)),
    )


def _radius_quantiles(distances: Sequence[float]) -> dict[str, float | None]:
    if not distances:
        return {"p50": None, "p90": None, "p95": None, "max": None}
    values = sorted(distances)
    return {
        "p50": _nearest_rank(values, 0.50),
        "p90": _nearest_rank(values, 0.90),
        "p95": _nearest_rank(values, 0.95),
        "max": values[-1],
    }


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    return values[max(0, math.ceil(quantile * len(values)) - 1)]


def _bounding_geometry(
    coordinates: Sequence[GeographicCoordinate],
) -> dict[str, float | bool | None]:
    if not coordinates:
        return {
            "south": None,
            "north": None,
            "west": None,
            "east": None,
            "crosses_dateline": False,
        }
    latitudes = [float(coordinate.latitude) for coordinate in coordinates]
    west, east, crosses_dateline = _longitude_envelope(
        [float(coordinate.longitude) for coordinate in coordinates]
    )
    return {
        "south": min(latitudes),
        "north": max(latitudes),
        "west": west,
        "east": east,
        "crosses_dateline": crosses_dateline,
    }


def _longitude_envelope(longitudes: Sequence[float]) -> tuple[float, float, bool]:
    circular = sorted(longitude % 360.0 for longitude in longitudes)
    if len(circular) == 1:
        longitude = _wrapped_longitude(circular[0])
        return longitude, longitude, False
    gaps = [
        (circular[(index + 1) % len(circular)] - circular[index]) % 360.0
        for index in range(len(circular))
    ]
    gap_index = max(range(len(gaps)), key=lambda index: (gaps[index], -index))
    west = _wrapped_longitude(circular[(gap_index + 1) % len(circular)])
    east = _wrapped_longitude(circular[gap_index])
    return west, east, west > east


def _wrapped_longitude(value: float) -> float:
    wrapped = (value + 180.0) % 360.0 - 180.0
    return 0.0 if wrapped == 0.0 else wrapped


def _coordinate_struct(
    coordinate: GeographicCoordinate | None,
) -> dict[str, float | None]:
    if coordinate is None:
        return {"latitude": None, "longitude": None}
    return {
        "latitude": float(coordinate.latitude),
        "longitude": float(coordinate.longitude),
    }


def _row_coordinate(row: Mapping[str, Any]) -> GeographicCoordinate:
    try:
        return GeographicCoordinate(
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"usable coordinate required for Flickr photo {row.get('flickr_photo_id')!r}"
        ) from exc


def _frame(rows: list[dict[str, object]], *, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def _normalize_bioregion_map(
    values: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    normalized: dict[str, str] = {}
    for value in values:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("bioregion_by_admin_region entries must be (admin_region, bioregion)")
        admin_region = _required_text(value[0], field_name="admin_region")
        bioregion = _required_text(value[1], field_name="bioregion")
        previous = normalized.get(admin_region)
        if previous is not None and previous != bioregion:
            raise ValueError(f"admin region {admin_region!r} maps to conflicting bioregions")
        normalized[admin_region] = bioregion
    return tuple(sorted(normalized.items()))


def _utc_datetime(value: str | datetime) -> datetime:
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError("created_at must be an ISO-8601 timestamp") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TypeError("created_at must be a datetime or ISO-8601 timestamp")
    if parsed.tzinfo is None:
        raise ValueError("created_at must include a timezone")
    return parsed.astimezone(UTC)


def _required_text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be nonblank")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _nonnegative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return normalized


def _positive_int(value: object, *, field_name: str) -> int:
    normalized = _nonnegative_int(value, field_name=field_name)
    if normalized < 1:
        raise ValueError(f"{field_name} must be positive")
    return normalized


def _positive_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive finite number")
    try:
        normalized = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive finite number") from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return normalized


__all__ = [
    "FLICKR_GEO_ASSIGNMENTS_FILE",
    "FLICKR_GEO_ASSIGNMENTS_SCHEMA_VERSION",
    "FLICKR_GEO_CLUSTERS_FILE",
    "FLICKR_GEO_CLUSTERS_SCHEMA_VERSION",
    "FLICKR_GEO_CLUSTER_METHOD",
    "NO_GEO_CLUSTER_ID",
    "FlickrGeoClusterBuildResult",
    "FlickrGeoClusterConfig",
    "build_flickr_geo_clusters",
    "cluster_configuration_hash",
    "flickr_geo_assignments_schema",
    "flickr_geo_clusters_schema",
    "write_flickr_geo_cluster_artifacts",
]
