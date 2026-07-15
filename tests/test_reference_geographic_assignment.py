from __future__ import annotations

import polars as pl
import pytest

from biominer.flickr_fetch.geographic_clustering import (
    UNASSIGNED_GEO_CLUSTER_ID,
    FlickrGeoClusterConfig,
    build_flickr_geo_clusters,
)
from biominer.flickr_fetch.geography import build_flickr_geography_frame
from biominer.references.geographic_assignment import (
    ReferenceGeographicAssignmentConfig,
    assign_reference_observations_to_flickr_clusters,
)
from biominer.references.schemas import reference_observations_frame
from test_reference_planner import TARGET_KEY, _reference_rows


def _clusters() -> pl.DataFrame:
    geography = build_flickr_geography_frame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "sydney-1",
                "source_record_hash": "sha256:sydney-1",
                "latitude": -33.8,
                "longitude": 151.2,
                "accuracy": 16,
            },
            {
                "source": "flickr",
                "flickr_photo_id": "sydney-2",
                "source_record_hash": "sha256:sydney-2",
                "latitude": -33.81,
                "longitude": 151.21,
                "accuracy": 16,
            },
        ]
    )
    return build_flickr_geo_clusters(
        geography,
        target_accepted_taxon_key=TARGET_KEY,
        config=FlickrGeoClusterConfig(
            source_cell_field="coarse_cell_id",
            source_resolution=3,
            minimum_cluster_images=1,
        ),
        created_at="2026-07-15T00:00:00Z",
    ).clusters


def _observations() -> pl.DataFrame:
    observations, _media, _review = _reference_rows(
        [
            {"taxon_key": TARGET_KEY, "cluster_id": "global", "observation_number": 1},
            {"taxon_key": TARGET_KEY, "cluster_id": "global", "observation_number": 2},
            {"taxon_key": TARGET_KEY, "cluster_id": "global", "observation_number": 3},
        ]
    )
    rows = observations.to_dicts()
    rows[1]["latitude"] = 40.7128
    rows[1]["longitude"] = -74.006
    rows[2]["latitude"] = None
    rows[2]["longitude"] = None
    rows[2]["coordinates_obscured"] = True
    rows[2]["distance_to_cluster_medoid_km"] = None
    return reference_observations_frame(rows)


def test_assigns_exact_cells_and_separates_remote_or_obscured_references() -> None:
    observations = _observations()
    clusters = _clusters()

    assigned = assign_reference_observations_to_flickr_clusters(
        observations,
        clusters,
        config=ReferenceGeographicAssignmentConfig(source_resolution=3),
    )
    reordered = assign_reference_observations_to_flickr_clusters(
        reference_observations_frame(observations.reverse().to_dicts()),
        clusters.reverse(),
        config=ReferenceGeographicAssignmentConfig(source_resolution=3),
    )

    assert assigned.equals(reordered)
    by_source_id = {
        row["source_observation_id"]: row for row in assigned.to_dicts()
    }
    exact = by_source_id[f"{TARGET_KEY}:global:1"]
    assert exact["geo_cluster_id"] != UNASSIGNED_GEO_CLUSTER_ID
    assert exact["fallback_level"] == 0
    assert exact["distance_to_cluster_medoid_km"] is not None
    for number in (2, 3):
        fallback = by_source_id[f"{TARGET_KEY}:global:{number}"]
        assert fallback["geo_cluster_id"] == UNASSIGNED_GEO_CLUSTER_ID
        assert fallback["fallback_level"] == 3
        assert fallback["distance_to_cluster_medoid_km"] is None


def test_rejects_overlapping_cluster_cells() -> None:
    clusters = _clusters()
    duplicate = clusters.with_columns(pl.lit("duplicate-cluster").alias("geo_cluster_id"))

    with pytest.raises(ValueError, match="belongs to multiple clusters"):
        assign_reference_observations_to_flickr_clusters(
            _observations(),
            clusters.vstack(duplicate),
            config=ReferenceGeographicAssignmentConfig(source_resolution=3),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source_resolution": 16}, "source_resolution"),
        ({"adjacency_grid_distance": -1}, "adjacency_grid_distance"),
        ({"maximum_assignment_distance_km": float("nan")}, "maximum_assignment"),
    ],
)
def test_validates_assignment_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ReferenceGeographicAssignmentConfig(**kwargs)  # type: ignore[arg-type]
