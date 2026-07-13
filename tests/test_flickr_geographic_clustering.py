from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from biominer.flickr_fetch.geographic_clustering import (
    FLICKR_GEO_ASSIGNMENTS_FILE,
    FLICKR_GEO_CLUSTERS_FILE,
    NO_GEO_CLUSTER_ID,
    FlickrGeoClusterConfig,
    build_flickr_geo_clusters,
    flickr_geo_assignments_schema,
    flickr_geo_clusters_schema,
    write_flickr_geo_cluster_artifacts,
)
from biominer.flickr_fetch.geography import build_flickr_geography_frame
from biominer.geography import cell_center, coordinate_to_cell, neighbour_cells


TARGET_KEY = "gbif:2734918"
CREATED_AT = datetime(2026, 7, 13, 2, 0, tzinfo=UTC)


def _record(
    photo_id: str,
    latitude: float | None,
    longitude: float | None,
    *,
    accuracy: float | None = 16,
    country_code: str | None = None,
    admin1: str | None = None,
) -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "source_record_hash": f"sha256:{photo_id}",
        "latitude": latitude,
        "longitude": longitude,
        "accuracy": accuracy,
        "country_code": country_code,
        "admin1": admin1,
    }


def _build(
    records: list[dict[str, object]],
    *,
    config: FlickrGeoClusterConfig | None = None,
):
    return build_flickr_geo_clusters(
        build_flickr_geography_frame(records),
        target_accepted_taxon_key=TARGET_KEY,
        config=config,
        created_at=CREATED_AT,
    )


def test_builds_deterministic_candidate_clusters_and_explicit_no_geo() -> None:
    records = [
        _record("brisbane-1", -27.4705, 153.026, country_code="AU", admin1="Queensland"),
        _record("brisbane-2", -27.471, 153.027, country_code="AU", admin1="Queensland"),
        _record("sydney-1", -33.8688, 151.2093, country_code="AU", admin1="New South Wales"),
        _record("sydney-2", -33.869, 151.21, country_code="AU", admin1="New South Wales"),
        _record("missing", None, None, accuracy=None),
    ]
    config = FlickrGeoClusterConfig(minimum_cluster_images=2)

    first = _build(records, config=config)
    second = _build(list(reversed(records)), config=config)

    assert first.clusters.schema == flickr_geo_clusters_schema()
    assert first.assignments.schema == flickr_geo_assignments_schema()
    assert first.clusters.equals(second.clusters)
    assert first.assignments.equals(second.assignments)
    assert first.cluster_configuration_hash == second.cluster_configuration_hash
    assert first.assignments["flickr_photo_id"].to_list() == sorted(
        record["flickr_photo_id"] for record in records
    )

    cluster_rows = first.clusters.to_dicts()
    assert len(cluster_rows) == 3
    assert {row["geo_cluster_id"] for row in cluster_rows} >= {NO_GEO_CLUSTER_ID}
    located = [row for row in cluster_rows if row["geo_cluster_id"] != NO_GEO_CLUSTER_ID]
    assert len(located) == 2
    assert all(row["candidate_distribution_only"] is True for row in cluster_rows)
    assert all(row["target_accepted_taxon_key"] == TARGET_KEY for row in cluster_rows)
    assert all(row["member_image_count"] == 2 for row in located)
    assert all(row["member_cell_count"] == 1 for row in located)
    assert all(row["centroid"]["latitude"] is not None for row in located)
    assert all(row["radius_quantiles_km"]["max"] is not None for row in located)
    no_geo = next(row for row in cluster_rows if row["geo_cluster_id"] == NO_GEO_CLUSTER_ID)
    assert no_geo["member_image_count"] == 1
    assert no_geo["source_resolution"] is None
    assert no_geo["centroid"] == {"latitude": None, "longitude": None}

    missing = next(
        row for row in first.assignments.to_dicts() if row["flickr_photo_id"] == "missing"
    )
    assert missing["geo_cluster_id"] == NO_GEO_CLUSTER_ID
    assert missing["assignment_method"] == "no_geo"
    assert missing["distance_to_medoid_km"] is None
    assert missing["outlier"] is False


def test_density_border_assignment_is_adjacency_gated_and_distance_capped() -> None:
    core_cell = coordinate_to_cell(-27.4705, 153.026, resolution=5)
    adjacent_cell = neighbour_cells(core_cell)[0]
    core = cell_center(core_cell)
    adjacent = cell_center(adjacent_cell)
    remote = cell_center(coordinate_to_cell(-33.8688, 151.2093, resolution=5))
    records = [
        _record("core-1", float(core.latitude), float(core.longitude)),
        _record("core-2", float(core.latitude), float(core.longitude)),
        _record("border", float(adjacent.latitude), float(adjacent.longitude)),
        _record("remote", float(remote.latitude), float(remote.longitude)),
    ]
    permissive = FlickrGeoClusterConfig(
        minimum_images_per_cell=2,
        minimum_cluster_images=2,
        maximum_assignment_distance_km=500,
    )
    result = _build(records, config=permissive)
    assignments = {row["flickr_photo_id"]: row for row in result.assignments.to_dicts()}

    assert assignments["core-1"]["assignment_method"] == "regional_cell"
    assert assignments["border"]["assignment_method"] == "adjacency"
    assert assignments["border"]["distance_to_medoid_km"] > 0
    assert assignments["border"]["outlier"] is False
    located_cluster = result.clusters.filter(pl.col("geo_cluster_id") != NO_GEO_CLUSTER_ID)
    assert located_cluster["member_cell_count"].to_list() == [2]
    assert adjacent_cell in located_cluster["member_cell_ids"].to_list()[0]
    assert assignments["remote"]["geo_cluster_id"] == NO_GEO_CLUSTER_ID
    assert assignments["remote"]["assignment_method"] == "no_geo"
    assert assignments["remote"]["outlier"] is True

    capped = _build(
        records,
        config=FlickrGeoClusterConfig(
            minimum_images_per_cell=2,
            minimum_cluster_images=2,
            maximum_assignment_distance_km=0.001,
        ),
    )
    capped_assignments = {
        row["flickr_photo_id"]: row for row in capped.assignments.to_dicts()
    }
    assert capped_assignments["border"]["geo_cluster_id"] == NO_GEO_CLUSTER_ID
    assert capped_assignments["border"]["outlier"] is True


def test_low_precision_fallback_requires_one_supported_cluster() -> None:
    records = [
        _record("core-1", -27.4705, 153.026, country_code="AU", admin1="Queensland"),
        _record("core-2", -27.471, 153.027, country_code="AU", admin1="Queensland"),
        _record(
            "coarse-bioregion",
            -25.0,
            145.0,
            accuracy=3,
            country_code="AU",
            admin1="Queensland",
        ),
        _record("coarse-country", -25.0, 145.0, accuracy=3, country_code="AU"),
    ]
    config = FlickrGeoClusterConfig(
        minimum_cluster_images=2,
        bioregion_by_admin_region=(("AU:Queensland", "australasia-east"),),
    )
    result = _build(records, config=config)
    assignments = {row["flickr_photo_id"]: row for row in result.assignments.to_dicts()}

    assert assignments["coarse-bioregion"]["assignment_method"] == "bioregion"
    assert assignments["coarse-bioregion"]["fallback_scope"] == (
        "bioregion:australasia-east"
    )
    assert assignments["coarse-bioregion"]["distance_to_medoid_km"] is None
    assert assignments["coarse-country"]["assignment_method"] == "country"
    assert assignments["coarse-country"]["fallback_scope"] == "country:AU"


def test_country_fallback_does_not_choose_between_multiple_clusters() -> None:
    result = _build(
        [
            _record("brisbane-1", -27.4705, 153.026, country_code="AU"),
            _record("brisbane-2", -27.471, 153.027, country_code="AU"),
            _record("sydney-1", -33.8688, 151.2093, country_code="AU"),
            _record("sydney-2", -33.869, 151.21, country_code="AU"),
            _record("coarse", -25.0, 145.0, accuracy=3, country_code="AU"),
        ],
        config=FlickrGeoClusterConfig(minimum_cluster_images=2),
    )
    coarse = next(
        row for row in result.assignments.to_dicts() if row["flickr_photo_id"] == "coarse"
    )
    assert coarse["geo_cluster_id"] == NO_GEO_CLUSTER_ID
    assert coarse["assignment_method"] == "no_geo"
    assert coarse["outlier"] is False


def test_cluster_identity_changes_with_configuration_not_input_order() -> None:
    records = [
        _record("1", -27.4705, 153.026),
        _record("2", -27.471, 153.027),
    ]
    first = _build(records, config=FlickrGeoClusterConfig(minimum_cluster_images=2))
    reordered = _build(
        list(reversed(records)),
        config=FlickrGeoClusterConfig(minimum_cluster_images=2),
    )
    changed = _build(
        records,
        config=FlickrGeoClusterConfig(
            minimum_cluster_images=2,
            maximum_assignment_distance_km=251,
        ),
    )
    first_id = first.clusters["geo_cluster_id"].to_list()[0]
    reordered_id = reordered.clusters["geo_cluster_id"].to_list()[0]
    changed_id = changed.clusters["geo_cluster_id"].to_list()[0]
    assert first_id == reordered_id
    assert first_id != changed_id


def test_writes_typed_cluster_artifacts(tmp_path) -> None:
    result = _build(
        [_record("1", -27.4705, 153.026), _record("2", -27.471, 153.027)],
        config=FlickrGeoClusterConfig(minimum_cluster_images=2),
    )
    paths = write_flickr_geo_cluster_artifacts(result, tmp_path)

    assert paths == {
        "clusters": tmp_path / FLICKR_GEO_CLUSTERS_FILE,
        "assignments": tmp_path / FLICKR_GEO_ASSIGNMENTS_FILE,
    }
    assert pl.read_parquet(paths["clusters"]).schema == flickr_geo_clusters_schema()
    assert pl.read_parquet(paths["assignments"]).schema == flickr_geo_assignments_schema()


@pytest.mark.parametrize(
    "config",
    [
        FlickrGeoClusterConfig(
            source_cell_field="coarse_cell_id",
            source_resolution=3,
            minimum_cluster_images=1,
        ),
        FlickrGeoClusterConfig(
            source_cell_field="local_cell_id",
            source_resolution=7,
            minimum_cluster_images=1,
        ),
    ],
)
def test_supports_configured_cell_resolution(config: FlickrGeoClusterConfig) -> None:
    result = _build([_record("1", -27.4705, 153.026)], config=config)
    located = result.clusters.filter(pl.col("geo_cluster_id") != NO_GEO_CLUSTER_ID)
    assert located["source_resolution"].to_list() == [config.source_resolution]
