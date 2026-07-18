"""Tests for precision-aware geographic reference lookup memberships."""

from __future__ import annotations

from datetime import UTC, datetime

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


class _Grid:
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


def _sha(character: str) -> str:
    return f"sha256:{character * 64}"


def _observation(source_id: str = "1", **changes: object) -> dict[str, object]:
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
        "source_record_hash": _sha(source_id),
        "retrieved_at": NOW,
        "source_snapshot_version": "gbif-occurrence-2026-07-18",
        "source_query_fingerprint": _sha("b"),
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


def _context(observation_id: str, **changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "reference_observation_id": observation_id,
        "continent_code": "OC",
        "admin1": "New South Wales",
        "bioregion": "Sydney Basin",
    }
    row.update(changes)
    return row


def _normalized(
    *observations: dict[str, object],
    contexts: list[dict[str, object]] | None = None,
    grid: _Grid | None = None,
) -> pl.DataFrame:
    if contexts is None:
        contexts = [
            _context(str(row["reference_observation_id"])) for row in observations
        ]
    return build_normalized_reference_geography(
        reference_observations_frame(list(observations)),
        resolutions=RESOLUTIONS,
        context_rows=contexts,
        policy=PRECISION_POLICY,
        grid=grid or _Grid(),
    )


def _index_row(
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
        "admission_policy_fingerprint": _sha("a"),
        "reference_quality_flags": ["provisional"],
        "embedding_fingerprint": _sha(suffix),
    }
    row.update(changes)
    return row


def _artifacts(
    *observations: dict[str, object],
    contexts: list[dict[str, object]] | None = None,
    index_changes: dict[str, dict[str, object]] | None = None,
    grid: _Grid | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    geography = _normalized(*observations, contexts=contexts, grid=grid)
    changes = index_changes or {}
    index = build_reference_geography_index(
        [
            _index_row(
                row,
                format(position, "x"),
                **changes.get(str(row["reference_observation_id"]), {}),
            )
            for position, row in enumerate(geography.iter_rows(named=True), start=1)
        ]
    )
    anchors = select_global_reference_anchors(index)
    return index, geography, anchors


def _build(
    *observations: dict[str, object],
    contexts: list[dict[str, object]] | None = None,
    index_changes: dict[str, dict[str, object]] | None = None,
    grid: _Grid | None = None,
) -> pl.DataFrame:
    active_grid = grid or _Grid()
    index, geography, anchors = _artifacts(
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
