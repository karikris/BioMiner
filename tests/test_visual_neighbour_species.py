from __future__ import annotations

from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from biominer.bioclip.reference_prototypes import (
    PROTOTYPE_KIND_AGGREGATE,
    MultiPrototypeConfig,
    build_multi_reference_prototypes,
    build_reference_prototypes,
)
from biominer.candidates.regional_union import (
    RegionalCandidateConfig,
    build_regional_candidate_species,
)
from biominer.candidates.visual_neighbours import (
    VISUAL_NEIGHBOUR_SPECIES_FILE,
    VisualNeighbourGraphConfig,
    build_visual_neighbour_species,
    load_visual_neighbour_species,
    validate_visual_neighbour_species,
    visual_neighbour_species_schema,
    write_visual_neighbour_species,
)
from test_reference_prototypes import _embedding_artifact, _spec
from test_regional_candidate_species import (
    COUNTRY_CONGENER,
    LOCAL_CONGENER,
    REGISTRY_VERSION,
    TARGET,
    _clusters,
    _occurrence,
    _taxa,
)


SPECIES_A = "gbif:100"
SPECIES_B = "gbif:200"
SPECIES_C = "gbif:300"
SPECIES_D = "gbif:400"


def test_builds_directed_ranked_cosine_neighbour_edges(tmp_path: Path) -> None:
    prototypes = _prototype_artifact(
        tmp_path,
        (
            (SPECIES_A, "Species alpha", (1, 0, 0)),
            (SPECIES_B, "Species beta", (0.9, 0.1, 0)),
            (SPECIES_C, "Species gamma", (0, 1, 0)),
            (SPECIES_D, "Species delta", (-1, 0, 0)),
        ),
    )

    graph = build_visual_neighbour_species(
        prototypes,
        graph_version="visual-neighbours-test-v1",
        config=VisualNeighbourGraphConfig(top_k_neighbors=2),
    )

    assert graph.schema == visual_neighbour_species_schema()
    alpha = graph.filter(pl.col("subject_accepted_taxon_key") == SPECIES_A)
    assert alpha["neighbour_accepted_taxon_key"].to_list() == [
        SPECIES_B,
        SPECIES_C,
    ]
    assert alpha["neighbour_rank"].to_list() == [1, 2]
    assert alpha["best_prototype_similarity"][0] == pytest.approx(
        0.9 / (0.9**2 + 0.1**2) ** 0.5
    )
    assert alpha["best_prototype_similarity"][1] == pytest.approx(0.0)
    assert all(
        row["subject_accepted_taxon_key"] != row["neighbour_accepted_taxon_key"]
        for row in graph.iter_rows(named=True)
    )
    assert graph["graph_fingerprint"].n_unique() == 1
    assert graph["edge_id"].n_unique() == graph.height
    assert graph["edge_fingerprint"].n_unique() == graph.height


def test_equal_similarity_ties_use_taxon_and_prototype_identity(tmp_path: Path) -> None:
    prototypes = _prototype_artifact(
        tmp_path,
        (
            (SPECIES_A, "Species alpha", (1, 0, 0)),
            (SPECIES_B, "Species beta", (0, 1, 0)),
            (SPECIES_C, "Species gamma", (0, 1, 0)),
        ),
    )

    graph = build_visual_neighbour_species(
        prototypes,
        graph_version="visual-neighbours-test-v1",
        config=VisualNeighbourGraphConfig(top_k_neighbors=2),
    )

    alpha = graph.filter(pl.col("subject_accepted_taxon_key") == SPECIES_A)
    assert alpha["neighbour_accepted_taxon_key"].to_list() == [
        SPECIES_B,
        SPECIES_C,
    ]
    assert alpha["best_prototype_similarity"].to_list() == [0.0, 0.0]


def test_multi_prototypes_do_not_multiply_species_neighbour_opportunities(
    tmp_path: Path,
) -> None:
    specs = []
    for taxon_index, (taxon_key, name, base) in enumerate(
        (
            (SPECIES_A, "Species alpha", (1, 0, 0)),
            (SPECIES_B, "Species beta", (0, 1, 0)),
            (SPECIES_C, "Species gamma", (0, 0, 1)),
        )
    ):
        for index in range(6):
            specs.append(
                _spec(
                    f"media-{taxon_index}-{index}",
                    f"observation-{taxon_index}-{index}",
                    taxon_key,
                    name,
                    "cluster-a",
                    (base[0] + index / 100, base[1], base[2]),
                    view="dorsal" if index < 3 else "ventral",
                )
            )
    embeddings = _embedding_artifact(tmp_path, tuple(specs))
    prototypes = build_multi_reference_prototypes(
        embeddings,
        config=MultiPrototypeConfig(
            minimum_metadata_observation_count=1,
            enable_embedding_clustering=False,
        ),
        include_mean_centered=False,
    )

    graph = build_visual_neighbour_species(
        prototypes,
        graph_version="visual-neighbours-test-v1",
        config=VisualNeighbourGraphConfig(top_k_neighbors=1),
    )

    assert graph.height == 3
    assert set(graph["prototype_kind"]) == {PROTOTYPE_KIND_AGGREGATE}
    aggregate_ids = set(
        prototypes.filter(
            (pl.col("prototype_kind") == PROTOTYPE_KIND_AGGREGATE)
            & (pl.col("cluster_scope_type") == "global")
        )["prototype_id"]
    )
    assert set(graph["subject_prototype_id"]) <= aggregate_ids
    assert set(graph["neighbour_prototype_id"]) <= aggregate_ids


def test_neighbours_never_cross_life_stage_routes(tmp_path: Path) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec(
                "a-adult",
                "oa-adult",
                SPECIES_A,
                "Species alpha",
                "cluster-a",
                (1, 0, 0),
            ),
            _spec(
                "b-adult",
                "ob-adult",
                SPECIES_B,
                "Species beta",
                "cluster-a",
                (0.9, 0.1, 0),
            ),
            _spec(
                "a-larva",
                "oa-larva",
                SPECIES_A,
                "Species alpha",
                "cluster-a",
                (0, 1, 0),
                life_stage="larva",
                route="larval",
                view="lateral",
            ),
            _spec(
                "c-larva",
                "oc-larva",
                SPECIES_C,
                "Species gamma",
                "cluster-a",
                (0, 0.9, 0.1),
                life_stage="larva",
                route="larval",
                view="lateral",
            ),
        ),
    )
    prototypes = build_reference_prototypes(
        embeddings,
        include_mean_centered=False,
    )

    graph = build_visual_neighbour_species(
        prototypes,
        graph_version="visual-neighbours-test-v1",
        config=VisualNeighbourGraphConfig(top_k_neighbors=1),
    )

    assert {
        (
            str(row["route"]),
            str(row["subject_accepted_taxon_key"]),
            str(row["neighbour_accepted_taxon_key"]),
        )
        for row in graph.iter_rows(named=True)
    } == {
        ("adult_field", SPECIES_A, SPECIES_B),
        ("adult_field", SPECIES_B, SPECIES_A),
        ("larval", SPECIES_A, SPECIES_C),
        ("larval", SPECIES_C, SPECIES_A),
    }


def test_graph_round_trip_and_fingerprints_bind_configuration(tmp_path: Path) -> None:
    prototypes = _prototype_artifact(
        tmp_path / "inputs",
        (
            (SPECIES_A, "Species alpha", (1, 0, 0)),
            (SPECIES_B, "Species beta", (0.9, 0.1, 0)),
            (SPECIES_C, "Species gamma", (0, 1, 0)),
        ),
    )
    first = build_visual_neighbour_species(
        prototypes,
        graph_version="visual-neighbours-test-v1",
        config=VisualNeighbourGraphConfig(top_k_neighbors=1),
    )
    second = build_visual_neighbour_species(
        prototypes,
        graph_version="visual-neighbours-test-v1",
        config=VisualNeighbourGraphConfig(top_k_neighbors=2),
    )

    assert first["graph_fingerprint"][0] != second["graph_fingerprint"][0]
    path = write_visual_neighbour_species(first, tmp_path / "published")
    assert path.name == VISUAL_NEIGHBOUR_SPECIES_FILE
    loaded = load_visual_neighbour_species(path)
    assert_frame_equal(loaded, first)

    tampered = first.with_columns(
        pl.when(pl.col("neighbour_rank") == 1)
        .then(pl.lit(2, dtype=pl.UInt32))
        .otherwise(pl.col("neighbour_rank"))
        .alias("neighbour_rank")
    )
    with pytest.raises(ValueError):
        validate_visual_neighbour_species(tampered)


def test_visual_graph_adds_candidates_without_removing_geographic_reasons(
    tmp_path: Path,
) -> None:
    prototypes = _prototype_artifact(
        tmp_path,
        (
            (TARGET, "Papilio demoleus", (1, 0, 0)),
            (COUNTRY_CONGENER, "Papilio machaon", (0.95, 0.05, 0)),
            (LOCAL_CONGENER, "Papilio polytes", (0, 1, 0)),
        ),
    )
    graph = build_visual_neighbour_species(
        prototypes,
        graph_version="visual-neighbours-test-v1",
        config=VisualNeighbourGraphConfig(top_k_neighbors=1),
    )
    occurrences = pl.DataFrame(
        [
            _occurrence("cluster-a", TARGET, "exact"),
            _occurrence("cluster-a", LOCAL_CONGENER, "buffer"),
            _occurrence("cluster-a", COUNTRY_CONGENER, "country"),
        ]
    )

    candidates = build_regional_candidate_species(
        target_accepted_taxon_key=TARGET,
        geo_clusters=_clusters("cluster-a"),
        regional_occurrence=occurrences,
        taxa=_taxa(),
        registry_version=REGISTRY_VERSION,
        visual_neighbour_species=graph,
        config=RegionalCandidateConfig(minimum_local_same_family_candidates=3),
    )

    by_key = {
        str(row["candidate_accepted_taxon_key"]): row
        for row in candidates.iter_rows(named=True)
    }
    assert by_key[LOCAL_CONGENER]["candidate_reason"] == [
        "same_genus_range_overlap",
        "regional_same_family",
    ]
    assert by_key[COUNTRY_CONGENER]["candidate_reason"] == [
        "visually_nearest",
        "same_genus_range_overlap",
        "country_fallback",
    ]
    assert by_key[COUNTRY_CONGENER]["visually_nearest"] is True
    graph_fingerprint = str(graph["graph_fingerprint"][0])
    assert (
        f"visual-neighbours:visual-neighbours-test-v1:{graph_fingerprint}"
        in by_key[TARGET]["source_versions"]
    )


def _prototype_artifact(
    tmp_path: Path,
    species: tuple[tuple[str, str, tuple[float, float, float]], ...],
):
    embeddings = _embedding_artifact(
        tmp_path,
        tuple(
            _spec(
                f"media-{index}",
                f"observation-{index}",
                taxon_key,
                scientific_name,
                "cluster-a",
                vector,
            )
            for index, (taxon_key, scientific_name, vector) in enumerate(species)
        ),
    )
    return build_reference_prototypes(
        embeddings,
        include_mean_centered=False,
    )
