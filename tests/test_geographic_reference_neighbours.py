"""Tests for precision-aware geographic reference lookup memberships."""

from __future__ import annotations

import polars as pl
import pytest

from biominer.bioclip.geographic_reference_neighbours import (
    GEOGRAPHIC_REFERENCE_NEIGHBOURS_FILE,
    GeographicReferenceNeighbourPolicy,
    build_geographic_reference_neighbours,
    geographic_reference_neighbours_artifact_fingerprint,
    geographic_reference_neighbours_schema,
    validate_geographic_reference_neighbours,
    write_geographic_reference_neighbours,
)
from biominer.bioclip.global_reference_anchors import (
    select_global_reference_anchors,
)
from biominer.bioclip.reference_geography_index import (
    build_reference_geography_index,
)
from support.reference_geography_fixtures import (
    FixtureGrid as _Grid,
    artifacts as _artifacts,
    build_neighbours as _build,
    context as _context,
    index_row as _index_row,
    normalized as _normalized,
    observation as _observation,
    sha as _sha,
)


def test_local_reference_materializes_complete_ordered_fallback_chain() -> None:
    neighbours = _build(_observation())

    assert neighbours["lookup_scope"].to_list() == [
        "exact_supported_cell",
        "neighbouring_supported_cell",
        "neighbouring_supported_cell",
        "parent_regional_cell",
        "parent_coarse_cell",
        "bioregion",
        "country",
        "continent",
        "global",
    ]
    assert neighbours["fallback_level"].to_list() == list(range(8))[:2] + [1] + list(
        range(2, 8)
    )
    exact = neighbours.filter(pl.col("lookup_scope") == "exact_supported_cell").row(
        0, named=True
    )
    assert exact["supported_cell_level"] == "local"
    assert exact["supported_cell_resolution"] == 7
    assert exact["lookup_cell_id"] == exact["local_cell_id"]
    assert exact["neighbour_grid_distance"] is None
    global_row = neighbours.filter(pl.col("lookup_scope") == "global").row(
        0, named=True
    )
    assert global_row["is_global_anchor"] is True
    assert global_row["lookup_key"] == "global"


@pytest.mark.parametrize(
    ("uncertainty", "expected_level", "expected_scopes"),
    [
        (
            5_000.0,
            "regional",
            {
                "exact_supported_cell",
                "neighbouring_supported_cell",
                "parent_coarse_cell",
                "bioregion",
                "country",
                "continent",
                "global",
            },
        ),
        (
            50_000.0,
            "coarse",
            {
                "exact_supported_cell",
                "neighbouring_supported_cell",
                "bioregion",
                "country",
                "continent",
                "global",
            },
        ),
    ],
)
def test_reference_never_materializes_finer_than_supported_precision(
    uncertainty: float,
    expected_level: str,
    expected_scopes: set[str],
) -> None:
    neighbours = _build(_observation(coordinate_uncertainty=uncertainty))

    assert set(neighbours["lookup_scope"]) == expected_scopes
    assert neighbours["supported_cell_level"].unique().to_list() == [expected_level]
    cell_rows = neighbours.filter(pl.col("lookup_cell_resolution").is_not_null())
    supported = int(cell_rows["supported_cell_resolution"][0])
    assert max(cell_rows["lookup_cell_resolution"].to_list()) <= supported
    if expected_level == "regional":
        assert "parent_regional_cell" not in set(neighbours["lookup_scope"])
        assert "parent_coarse_cell" in set(neighbours["lookup_scope"])
    else:
        assert "parent_coarse_cell" not in set(neighbours["lookup_scope"])


def test_country_only_and_missing_geography_use_named_or_global_fallback_only() -> None:
    country = _observation(
        latitude=None,
        longitude=None,
        coordinate_uncertainty=None,
        country="Australia",
        country_code="AU",
        geo_cluster_id=None,
    )
    country_rows = _build(
        country,
        contexts=[
            _context(
                str(country["reference_observation_id"]),
                bioregion=None,
            )
        ],
    )
    assert set(country_rows["lookup_scope"]) == {"country", "continent", "global"}
    assert country_rows["supported_cell_id"].null_count() == country_rows.height

    missing = _observation(
        latitude=None,
        longitude=None,
        coordinate_uncertainty=None,
        country=None,
        country_code=None,
        geo_cluster_id=None,
    )
    missing_rows = _build(
        missing,
        contexts=[
            _context(
                str(missing["reference_observation_id"]),
                continent_code=None,
                bioregion=None,
            )
        ],
    )
    assert missing_rows["lookup_scope"].to_list() == ["global"]
    assert missing_rows["geography_unavailable_reason"].to_list() == [
        "coordinates_missing"
    ]


def test_local_ineligibility_cannot_manufacture_cell_memberships() -> None:
    observation = _observation()
    observation_id = str(observation["reference_observation_id"])
    neighbours = _build(
        observation,
        index_changes={observation_id: {"local_anchor_eligible": False}},
    )

    assert set(neighbours["lookup_scope"]) == {
        "bioregion",
        "country",
        "continent",
        "global",
    }
    assert neighbours["supported_cell_level"].unique().to_list() == ["local"]
    assert neighbours["lookup_cell_id"].null_count() == neighbours.height


def test_output_is_deterministic_and_lookup_ranks_are_contiguous() -> None:
    first_observation = _observation("1")
    second_observation = _observation(
        "2",
        latitude=-37.81,
        longitude=144.96,
        geo_cluster_id="cluster-vic",
    )
    forward = _build(first_observation, second_observation)
    reverse = _build(second_observation, first_observation)

    assert forward.equals(reverse)
    for group in forward.partition_by(
        ["lookup_scope", "lookup_key", "accepted_taxon_key", "route"]
    ):
        assert group["lookup_rank"].to_list() == list(range(1, group.height + 1))
    assert geographic_reference_neighbours_artifact_fingerprint(forward) == (
        geographic_reference_neighbours_artifact_fingerprint(reverse)
    )


def test_multiple_embeddings_have_distinct_memberships_without_duplicate_grain() -> (
    None
):
    geography = _normalized(_observation())
    geography_row = geography.row(0, named=True)
    index = build_reference_geography_index(
        [
            _index_row(geography_row, "1"),
            _index_row(
                geography_row,
                "2",
                reference_media_id=f"reference-media:{'2' * 64}",
                visual_input_kind="focused_full_frame",
                embedding_fingerprint=_sha("2"),
            ),
        ]
    )
    anchors = select_global_reference_anchors(index)

    neighbours = build_geographic_reference_neighbours(
        index, geography, anchors, grid=_Grid()
    )

    exact = neighbours.filter(pl.col("lookup_scope") == "exact_supported_cell")
    assert exact.height == 2
    assert exact["membership_id"].n_unique() == 2
    assert exact["reference_observation_id"].n_unique() == 1
    assert exact["lookup_rank"].to_list() == [1, 2]
    assert (
        exact.select(
            "reference_geography_row_fingerprint", "lookup_scope", "lookup_key"
        ).n_unique()
        == exact.height
    )


def test_rejects_index_geography_conflict_and_stale_global_anchors() -> None:
    index, geography, anchors = _artifacts(_observation())
    conflicting_index = build_reference_geography_index(
        [
            _index_row(
                geography.row(0, named=True),
                "1",
                country_code="NZ",
            )
        ]
    )
    with pytest.raises(ValueError, match="conflicts with normalized geography"):
        build_geographic_reference_neighbours(
            conflicting_index,
            geography,
            select_global_reference_anchors(conflicting_index),
            grid=_Grid(),
        )

    expanded_index = build_reference_geography_index(
        [
            _index_row(geography.row(0, named=True), "1"),
            _index_row(
                geography.row(0, named=True),
                "2",
                reference_media_id=f"reference-media:{'2' * 64}",
                visual_input_kind="focused_full_frame",
                embedding_fingerprint=_sha("2"),
            ),
        ]
    )
    with pytest.raises(ValueError, match="another reference index"):
        build_geographic_reference_neighbours(
            expanded_index,
            geography,
            anchors,
            grid=_Grid(),
        )
    assert index.height == 1


def test_rejects_grid_identity_drift_and_invalid_neighbour() -> None:
    class _WrongGrid(_Grid):
        version = "fixture-grid:v2"

    index, geography, anchors = _artifacts(_observation())
    with pytest.raises(ValueError, match="grid identity differs"):
        build_geographic_reference_neighbours(
            index, geography, anchors, grid=_WrongGrid()
        )

    class _InvalidNeighbourGrid(_Grid):
        def neighbours(
            self,
            cell_id: str,
            *,
            grid_distance: int = 1,
            include_origin: bool = False,
        ) -> tuple[str, ...]:
            return ("invalid",)

    with pytest.raises(ValueError, match="invalid neighbour"):
        build_geographic_reference_neighbours(
            index, geography, anchors, grid=_InvalidNeighbourGrid()
        )


def test_write_round_trip_and_validator_rejects_tampering(tmp_path) -> None:
    neighbours = _build(_observation())
    path = write_geographic_reference_neighbours(neighbours, tmp_path / "index")
    loaded = pl.read_parquet(path)

    assert path.name == GEOGRAPHIC_REFERENCE_NEIGHBOURS_FILE
    assert loaded.schema == geographic_reference_neighbours_schema()
    validate_geographic_reference_neighbours(loaded)
    assert geographic_reference_neighbours_artifact_fingerprint(loaded) == (
        geographic_reference_neighbours_artifact_fingerprint(neighbours)
    )

    tampered = loaded.with_columns(pl.lit(7).cast(pl.UInt8).alias("fallback_level"))
    with pytest.raises(ValueError, match="canonically sorted|fallback level conflicts"):
        validate_geographic_reference_neighbours(tampered)


def test_empty_inputs_produce_closed_empty_artifact() -> None:
    empty_index = build_reference_geography_index([])
    empty_geography = _normalized()
    empty_anchors = select_global_reference_anchors(empty_index)

    neighbours = build_geographic_reference_neighbours(
        empty_index, empty_geography, empty_anchors
    )

    assert neighbours.is_empty()
    assert neighbours.schema == geographic_reference_neighbours_schema()


@pytest.mark.parametrize("distance", [0, 2, True])
def test_policy_rejects_unbounded_or_invalid_neighbour_distance(
    distance: object,
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        GeographicReferenceNeighbourPolicy(
            neighbour_grid_distance=distance  # type: ignore[arg-type]
        )
