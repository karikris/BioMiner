"""Reusable deterministic artifacts for geographic reference-index tests."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from biominer.bioclip.geographic_reference_neighbours import (
    build_geographic_reference_neighbours,
)
from biominer.bioclip.global_reference_anchors import (
    select_global_reference_anchors,
)
from biominer.bioclip.reference_geography_index import (
    build_reference_geography_index,
)
from biominer.geography import GeographicCoordinate, GeographicResolutions
from biominer.references.normalized_geography import (
    ReferenceGeographyPrecisionPolicy,
    build_normalized_reference_geography,
)
from biominer.references.schemas import (
    REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
    make_reference_observation_id,
    reference_observations_frame,
)


NOW = datetime(2026, 7, 18, tzinfo=UTC)
RESOLUTIONS = GeographicResolutions(coarse=3, regional=5, local=7)
PRECISION_POLICY = ReferenceGeographyPrecisionPolicy(
    local_max_uncertainty_m=1_000,
    regional_max_uncertainty_m=10_000,
    coarse_max_uncertainty_m=100_000,
)


class FixtureGrid:
    name = "fixture_hierarchical_grid"
    version = "fixture-grid:v1"

    def coordinate_to_cell(
        self, coordinate: GeographicCoordinate, *, resolution: int
    ) -> str:
        return (
            f"fixture-r{resolution}:"
            f"{float(coordinate.latitude):.3f}:{float(coordinate.longitude):.3f}"
        )

    def parent(self, cell_id: str, *, resolution: int | None = None) -> str:
        return f"{cell_id}:parent:{resolution}"

    def neighbours(
        self,
        cell_id: str,
        *,
        grid_distance: int = 1,
        include_origin: bool = False,
    ) -> tuple[str, ...]:
        neighbours = (f"{cell_id}:neighbour:a", f"{cell_id}:neighbour:b")
        return (cell_id, *neighbours) if include_origin else neighbours

    def center(self, cell_id: str) -> GeographicCoordinate:
        return GeographicCoordinate(-33.87, 151.21)

    def is_valid(self, cell_id: object) -> bool:
        return isinstance(cell_id, str) and cell_id.startswith("fixture-r")


def sha(character: str) -> str:
    return f"sha256:{character * 64}"


def observation(source_id: str = "1", **changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": REFERENCE_OBSERVATIONS_SCHEMA_VERSION,
        "reference_observation_id": make_reference_observation_id("GBIF", source_id),
        "source": "GBIF",
        "source_observation_id": source_id,
        "source_taxon_id": "1938069",
        "supplied_scientific_name": "Papilio demoleus",
        "accepted_taxon_key": "gbif:1938069",
        "reconciled_scientific_name": "Papilio demoleus",
        "registry_version": "butterflies-v2-20260718",
        "taxon_reconciliation_status": "accepted_key_exact",
        "identification_quality": "research_grade",
        "community_taxon_status": "species",
        "identification_disagreement": False,
        "captive_or_cultivated": False,
        "observer_id": f"observer-{source_id}",
        "locality": "Sydney",
        "life_stage": "adult",
        "sex": None,
        "observed_at": datetime(2025, 1, 2, 3, 4, tzinfo=UTC),
        "latitude": -33.87,
        "longitude": 151.21,
        "coordinate_uncertainty": 50.0,
        "coordinates_obscured": False,
        "country": "Australia",
        "country_code": "AU",
        "geo_cluster_id": "cluster-au",
        "distance_to_cluster_medoid_km": 4.2,
        "source_dataset_key": "dataset-1",
        "source_dataset_doi": "10.15468/example",
        "source_record_url": f"https://example.test/occurrence/{source_id}",
        "source_record_hash": sha(source_id),
        "retrieved_at": NOW,
        "source_snapshot_version": "gbif-occurrence-2026-07-18",
        "source_query_fingerprint": sha("b"),
        "fallback_level": 0,
        "geospatial_issue": False,
        "preserved_specimen": False,
        "fossil": False,
        "occurrence_absent": False,
        "uncertain_taxon_match": False,
        "basis_of_record_suitable": True,
    }
    row.update(changes)
    if row["latitude"] is None or row["longitude"] is None:
        row["distance_to_cluster_medoid_km"] = None
    return row


def context(observation_id: str, **changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "reference_observation_id": observation_id,
        "continent_code": "OC",
        "admin1": "New South Wales",
        "bioregion": "Sydney Basin",
    }
    row.update(changes)
    return row


def normalized(
    *observations: dict[str, object],
    contexts: list[dict[str, object]] | None = None,
    grid: FixtureGrid | None = None,
) -> pl.DataFrame:
    if contexts is None:
        contexts = [
            context(str(row["reference_observation_id"])) for row in observations
        ]
    return build_normalized_reference_geography(
        reference_observations_frame(list(observations)),
        resolutions=RESOLUTIONS,
        context_rows=contexts,
        policy=PRECISION_POLICY,
        grid=grid or FixtureGrid(),
    )


def index_row(
    geography: dict[str, object],
    suffix: str,
    **changes: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "registry_version": geography["registry_version"],
        "reference_bank_version": "reference-bank-v3",
        "reference_media_id": f"reference-media:{suffix * 64}",
        "reference_observation_id": geography["reference_observation_id"],
        "source": geography["source"],
        "source_dataset_key": geography["source_dataset_key"],
        "accepted_taxon_key": geography["accepted_taxon_key"],
        "scientific_name": geography["scientific_name"],
        "family_key": "gbif:9417",
        "family_name": "Papilionidae",
        "genus_key": "gbif:1920494",
        "genus_name": "Papilio",
        "route": "adult_field",
        "life_stage": "adult",
        "visual_domain": "live_field",
        "visual_input_kind": "raw_full_image",
        "country_code": geography["country_code"],
        "admin1": geography["admin1"],
        "bioregion": geography["bioregion"],
        "geo_cluster_id": geography["source_geo_cluster_id"],
        "coarse_cell_id": geography["coarse_cell_id"],
        "regional_cell_id": geography["regional_cell_id"],
        "local_cell_id": geography["local_cell_id"],
        "latitude": geography["latitude"],
        "longitude": geography["longitude"],
        "coordinate_uncertainty_m": geography["coordinate_uncertainty_m"],
        "coordinate_quality": geography["coordinate_quality"],
        "global_anchor_eligible": True,
        "local_anchor_eligible": geography["coordinate_quality"]
        in {"local", "regional", "coarse"},
        "duplicate_group_id": f"reference-duplicate-group:{suffix * 32}",
        "observer_id_hash": geography["observer_id_hash"],
        "observation_date": geography["observed_date"],
        "admission_mode": "adaptive_gbif_fast_start",
        "admission_policy_fingerprint": sha("a"),
        "reference_quality_flags": ["provisional"],
        "embedding_fingerprint": sha(suffix),
    }
    row.update(changes)
    return row


def artifacts(
    *observations: dict[str, object],
    contexts: list[dict[str, object]] | None = None,
    index_changes: dict[str, dict[str, object]] | None = None,
    grid: FixtureGrid | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    geography = normalized(*observations, contexts=contexts, grid=grid)
    changes = index_changes or {}
    index = build_reference_geography_index(
        [
            index_row(
                row,
                format(position, "x"),
                **changes.get(str(row["reference_observation_id"]), {}),
            )
            for position, row in enumerate(geography.iter_rows(named=True), start=1)
        ]
    )
    anchors = select_global_reference_anchors(index)
    return index, geography, anchors


def build_neighbours(
    *observations: dict[str, object],
    contexts: list[dict[str, object]] | None = None,
    index_changes: dict[str, dict[str, object]] | None = None,
    grid: FixtureGrid | None = None,
) -> pl.DataFrame:
    active_grid = grid or FixtureGrid()
    index, geography, anchors = artifacts(
        *observations,
        contexts=contexts,
        index_changes=index_changes,
        grid=active_grid,
    )
    return build_geographic_reference_neighbours(
        index,
        geography,
        anchors,
        grid=active_grid,
    )


def complete_artifacts(
    *observations: dict[str, object],
    contexts: list[dict[str, object]] | None = None,
    index_changes: dict[str, dict[str, object]] | None = None,
    grid: FixtureGrid | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    active_grid = grid or FixtureGrid()
    index, geography, anchors = artifacts(
        *observations,
        contexts=contexts,
        index_changes=index_changes,
        grid=active_grid,
    )
    neighbours = build_geographic_reference_neighbours(
        index, geography, anchors, grid=active_grid
    )
    return index, geography, anchors, neighbours


__all__ = [
    "FixtureGrid",
    "artifacts",
    "build_neighbours",
    "complete_artifacts",
    "context",
    "index_row",
    "normalized",
    "observation",
    "sha",
]
