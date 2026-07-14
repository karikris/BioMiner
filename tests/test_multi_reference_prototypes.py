from __future__ import annotations

from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from biominer.bioclip.reference_prototypes import (
    EMBEDDING_CLUSTER_METHOD_AVERAGE_LINKAGE_COSINE,
    PROTOTYPE_KIND_AGGREGATE,
    PROTOTYPE_KIND_EMBEDDING_CLUSTER,
    PROTOTYPE_KIND_METADATA,
    PROTOTYPE_METHOD_NORMALIZED_MEAN,
    PROTOTYPE_SCOPE_GLOBAL,
    PROTOTYPE_SCOPE_REGIONAL,
    MultiPrototypeConfig,
    build_multi_reference_prototypes,
    load_reference_prototypes,
    write_reference_prototypes,
)
from test_reference_prototypes import _embedding_artifact, _spec


TARGET = "gbif:1938069"
COMPETITOR = "gbif:1938070"


def test_builds_view_and_regional_metadata_prototypes(tmp_path: Path) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec("d1", "od1", TARGET, "Papilio demoleus", "cluster-a", (1, 0, 0)),
            _spec("d2", "od2", TARGET, "Papilio demoleus", "cluster-a", (1, 0.1, 0)),
            _spec("d3", "od3", TARGET, "Papilio demoleus", "cluster-b", (1, 0.2, 0)),
            _spec(
                "v1",
                "ov1",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (0, 1, 0),
                view="ventral",
            ),
            _spec(
                "v2",
                "ov2",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (0.1, 1, 0),
                view="ventral",
            ),
        ),
    )

    prototypes = build_multi_reference_prototypes(
        embeddings,
        config=MultiPrototypeConfig(
            minimum_metadata_observation_count=1,
            enable_embedding_clustering=False,
        ),
        include_mean_centered=False,
    )

    aggregate = prototypes.filter(
        (pl.col("accepted_taxon_key") == TARGET)
        & (pl.col("prototype_kind") == PROTOTYPE_KIND_AGGREGATE)
    )
    assert aggregate.height == 3
    assert set(aggregate["view"]) == {"all"}

    metadata = prototypes.filter(
        (pl.col("accepted_taxon_key") == TARGET)
        & (pl.col("prototype_kind") == PROTOTYPE_KIND_METADATA)
        & (pl.col("prototype_method") == PROTOTYPE_METHOD_NORMALIZED_MEAN)
    )
    assert {
        (str(row["cluster_scope_type"]), str(row["geo_cluster_id"]), str(row["view"]))
        for row in metadata.iter_rows(named=True)
    } == {
        (PROTOTYPE_SCOPE_GLOBAL, "all", "dorsal"),
        (PROTOTYPE_SCOPE_GLOBAL, "all", "ventral"),
        (PROTOTYPE_SCOPE_REGIONAL, "cluster-a", "dorsal"),
        (PROTOTYPE_SCOPE_REGIONAL, "cluster-a", "ventral"),
        (PROTOTYPE_SCOPE_REGIONAL, "cluster-b", "dorsal"),
    }
    assert all(metadata["metadata_group_id"].is_not_null())
    assert all(metadata["embedding_cluster_id"].is_null())
    assert metadata["accepted_taxon_key"].unique().to_list() == [TARGET]


def test_clusters_only_within_verified_species_metadata_group(tmp_path: Path) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec("x1", "ox1", TARGET, "Papilio demoleus", "cluster-a", (1, 0, 0)),
            _spec("x2", "ox2", TARGET, "Papilio demoleus", "cluster-a", (1, 0.04, 0)),
            _spec("x3", "ox3", TARGET, "Papilio demoleus", "cluster-a", (1, -0.04, 0)),
            _spec("y1", "oy1", TARGET, "Papilio demoleus", "cluster-a", (0, 1, 0)),
            _spec("y2", "oy2", TARGET, "Papilio demoleus", "cluster-a", (0.04, 1, 0)),
            _spec("y3", "oy3", TARGET, "Papilio demoleus", "cluster-a", (-0.04, 1, 0)),
        ),
    )
    config = MultiPrototypeConfig(
        minimum_metadata_observation_count=1,
        minimum_clustering_observation_count=6,
        minimum_embedding_cluster_size=3,
        maximum_embedding_cluster_count=4,
        cosine_distance_threshold=0.1,
    )

    prototypes = build_multi_reference_prototypes(
        embeddings,
        config=config,
        include_mean_centered=False,
    )
    clusters = prototypes.filter(
        (pl.col("accepted_taxon_key") == TARGET)
        & (pl.col("prototype_kind") == PROTOTYPE_KIND_EMBEDDING_CLUSTER)
        & (pl.col("cluster_scope_type") == PROTOTYPE_SCOPE_GLOBAL)
        & (pl.col("view") == "dorsal")
    )

    assert clusters.height == 2
    assert set(clusters["independent_observation_count"]) == {3}
    assert set(clusters["clustering_method"]) == {
        EMBEDDING_CLUSTER_METHOD_AVERAGE_LINKAGE_COSINE
    }
    assert set(clusters["clustering_configuration_fingerprint"]) == {config.fingerprint}
    assert set(clusters["accepted_taxon_key"]) == {TARGET}
    member_groups = {
        tuple(row["member_observation_ids"]) for row in clusters.iter_rows(named=True)
    }
    assert member_groups == {
        ("ox1", "ox2", "ox3"),
        ("oy1", "oy2", "oy3"),
    }
    assert clusters["embedding_cluster_id"].n_unique() == 2
    assert all(clusters["embedding_cluster_id"].is_not_null())


def test_does_not_cluster_across_species_or_life_stage(tmp_path: Path) -> None:
    target_adult = tuple(
        _spec(
            f"target-{index}",
            f"target-observation-{index}",
            TARGET,
            "Papilio demoleus",
            "cluster-a",
            (1, index / 20, 0),
        )
        for index in range(3)
    )
    competitor = tuple(
        _spec(
            f"competitor-{index}",
            f"competitor-observation-{index}",
            COMPETITOR,
            "Papilio polytes",
            "cluster-a",
            (0, 1, index / 20),
        )
        for index in range(3)
    )
    target_larval = tuple(
        _spec(
            f"larva-{index}",
            f"larva-observation-{index}",
            TARGET,
            "Papilio demoleus",
            "cluster-a",
            (0, index / 20, 1),
            life_stage="larva",
            route="larval",
            view="lateral",
        )
        for index in range(3)
    )
    embeddings = _embedding_artifact(
        tmp_path,
        (*target_adult, *competitor, *target_larval),
    )

    prototypes = build_multi_reference_prototypes(
        embeddings,
        config=MultiPrototypeConfig(
            minimum_metadata_observation_count=1,
            minimum_clustering_observation_count=6,
            minimum_embedding_cluster_size=3,
        ),
        include_mean_centered=False,
    )

    assert prototypes.filter(
        pl.col("prototype_kind") == PROTOTYPE_KIND_EMBEDDING_CLUSTER
    ).is_empty()
    assert {
        (str(row["accepted_taxon_key"]), str(row["route"]))
        for row in prototypes.filter(
            pl.col("prototype_kind") == PROTOTYPE_KIND_METADATA
        ).iter_rows(named=True)
    } == {
        (TARGET, "adult_field"),
        (TARGET, "larval"),
        (COMPETITOR, "adult_field"),
    }


def test_small_groups_fall_back_to_metadata_centroid(tmp_path: Path) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec("a1", "oa1", TARGET, "Papilio demoleus", "cluster-a", (1, 0, 0)),
            _spec("a2", "oa2", TARGET, "Papilio demoleus", "cluster-a", (0, 1, 0)),
            _spec("a3", "oa3", TARGET, "Papilio demoleus", "cluster-a", (0, 0, 1)),
        ),
    )

    prototypes = build_multi_reference_prototypes(
        embeddings,
        config=MultiPrototypeConfig(
            minimum_metadata_observation_count=1,
            minimum_clustering_observation_count=6,
            minimum_embedding_cluster_size=3,
        ),
        include_mean_centered=False,
    )

    assert prototypes.filter(
        pl.col("prototype_kind") == PROTOTYPE_KIND_EMBEDDING_CLUSTER
    ).is_empty()
    metadata = prototypes.filter(
        (pl.col("prototype_kind") == PROTOTYPE_KIND_METADATA)
        & (pl.col("cluster_scope_type") == PROTOTYPE_SCOPE_GLOBAL)
    )
    assert metadata.height == 1
    assert metadata["member_observation_ids"][0].to_list() == ["oa1", "oa2", "oa3"]


def test_multiple_views_share_one_aggregate_observation_without_losing_view_models(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec(
                "dorsal-media",
                "shared-observation",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (1, 0, 0),
                view="dorsal",
            ),
            _spec(
                "ventral-media",
                "shared-observation",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (0, 1, 0),
                view="ventral",
            ),
        ),
    )

    prototypes = build_multi_reference_prototypes(
        embeddings,
        config=MultiPrototypeConfig(
            minimum_metadata_observation_count=1,
            enable_embedding_clustering=False,
        ),
        include_mean_centered=False,
    )

    aggregate = prototypes.filter(
        (pl.col("prototype_kind") == PROTOTYPE_KIND_AGGREGATE)
        & (pl.col("cluster_scope_type") == PROTOTYPE_SCOPE_GLOBAL)
    )
    metadata = prototypes.filter(
        (pl.col("prototype_kind") == PROTOTYPE_KIND_METADATA)
        & (pl.col("cluster_scope_type") == PROTOTYPE_SCOPE_GLOBAL)
    )
    assert aggregate["independent_observation_count"].to_list() == [1]
    assert aggregate["reference_count"].to_list() == [2]
    assert set(metadata["view"]) == {"dorsal", "ventral"}
    assert set(metadata["independent_observation_count"]) == {1}
    assert all(
        row["member_observation_ids"] == ["shared-observation"]
        for row in metadata.iter_rows(named=True)
    )


def test_multi_prototype_artifact_round_trips_member_provenance(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path / "inputs",
        tuple(
            _spec(
                f"media-{index}",
                f"observation-{index}",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (1, index + 1, 0),
                view="dorsal" if index < 3 else "ventral",
            )
            for index in range(6)
        ),
    )
    prototypes = build_multi_reference_prototypes(
        embeddings,
        config=MultiPrototypeConfig(
            minimum_metadata_observation_count=1,
            enable_embedding_clustering=False,
        ),
        include_mean_centered=False,
    )

    path = write_reference_prototypes(prototypes, tmp_path / "published")
    loaded = load_reference_prototypes(path)

    assert_frame_equal(loaded, prototypes)
    assert all(
        len(row["member_observation_ids"]) == int(row["independent_observation_count"])
        for row in loaded.iter_rows(named=True)
    )
