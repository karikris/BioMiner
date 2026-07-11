from __future__ import annotations

import polars as pl

from biominer.benchmarks.path_cascade import (
    BENCHMARK_FIXTURE_NOTICE,
    BENCHMARK_SELECTED_GENUS_NODE_IDS,
    FAMILY_COUNT,
    GENERA_PER_BRANCH,
    SPECIES_PER_SELECTED_GENUS,
    SUBFAMILIES_PER_FAMILY,
    TRIBES_PER_SUBFAMILY,
    build_seven_family_path_cascade_fixture,
)
from biominer.registry.classification_v3 import (
    ASSERTED_PARENT_EDGE,
    EDGE_SCHEMA,
    GBIF_MAPPING_SCHEMA,
    LEAF_PATH_SCHEMA,
    NODE_SCHEMA,
    PROMPT_LABEL_SCHEMA,
    QA_FINDING_SCHEMA,
    REVIEWED_RANK_SKIP_EDGE,
    SOURCE_SCHEMA,
    classification_v3_fingerprint,
    hierarchy_fingerprint,
)


def test_seven_family_fixture_is_a_validated_non_authoritative_v3_store() -> None:
    fixture = build_seven_family_path_cascade_fixture()
    frames = fixture.frames

    assert frames.sources.schema == SOURCE_SCHEMA
    assert frames.nodes.schema == NODE_SCHEMA
    assert frames.edges.schema == EDGE_SCHEMA
    assert frames.gbif_mappings.schema == GBIF_MAPPING_SCHEMA
    assert frames.leaf_paths.schema == LEAF_PATH_SCHEMA
    assert frames.prompt_labels.schema == PROMPT_LABEL_SCHEMA
    assert fixture.qa_findings.schema == QA_FINDING_SCHEMA
    assert fixture.qa_findings.filter(pl.col("severity") == "fatal").is_empty()
    assert set(fixture.qa_findings["code"].to_list()) == {"optional_subtribe_skipped"}

    assert fixture.manifest["qa_status"] == "passed"
    assert fixture.manifest["fatal_finding_count"] == 0
    assert fixture.manifest["benchmark_fixture"] is True
    assert fixture.manifest["authoritative_taxonomy"] is False
    assert fixture.manifest["gbif_authority"] is False
    assert fixture.manifest["fixture_notice"] == BENCHMARK_FIXTURE_NOTICE
    assert "not biological or GBIF authority" in frames.sources["authority"][0]
    assert all(
        str(key).startswith("synthetic-")
        for key in frames.gbif_mappings["gbif_species_key"].to_list()
    )
    assert fixture.taxonomy_store.classification_fingerprint == classification_v3_fingerprint(frames)
    assert fixture.taxonomy_store.hierarchy_fingerprint == hierarchy_fingerprint(frames)
    assert fixture.taxonomy_store.enabled_paths().equals(frames.leaf_paths.filter(pl.col("enabled")))


def test_seven_family_fixture_has_required_branching_and_species_depth() -> None:
    fixture = build_seven_family_path_cascade_fixture()
    nodes = fixture.frames.nodes.filter(pl.col("enabled"))
    paths = fixture.taxonomy_store.enabled_paths()

    assert nodes.filter(pl.col("rank") == "FAMILY").height == FAMILY_COUNT == 7
    subfamilies_per_family = paths.group_by("family_node_id").agg(
        pl.col("subfamily_node_id").n_unique().alias("count")
    )
    assert subfamilies_per_family.height == FAMILY_COUNT
    assert set(subfamilies_per_family["count"].to_list()) == {SUBFAMILIES_PER_FAMILY}
    tribes_per_subfamily = paths.group_by("subfamily_node_id").agg(
        pl.col("tribe_node_id").n_unique().alias("count")
    )
    assert tribes_per_subfamily.height == FAMILY_COUNT * SUBFAMILIES_PER_FAMILY
    assert set(tribes_per_subfamily["count"].to_list()) == {TRIBES_PER_SUBFAMILY}

    assert set(paths["path_completeness"].to_list()) == {
        "complete",
        "reviewed_optional_skip",
    }
    skip_edges = fixture.frames.edges.filter(pl.col("edge_type") == REVIEWED_RANK_SKIP_EDGE)
    assert skip_edges.height == FAMILY_COUNT * SUBFAMILIES_PER_FAMILY * GENERA_PER_BRANCH
    assert skip_edges["skipped_ranks"].to_list() == [["SUBTRIBE"]] * skip_edges.height
    assert (
        fixture.frames.edges.filter(
            (pl.col("edge_type") == ASSERTED_PARENT_EDGE)
            & (pl.col("parent_rank") == "TRIBE")
            & (pl.col("child_rank") == "SUBTRIBE")
        ).height
        == FAMILY_COUNT * SUBFAMILIES_PER_FAMILY
    )
    asserted_paths = paths.filter(pl.col("path_completeness") == "complete")
    skipped_paths = paths.filter(pl.col("path_completeness") == "reviewed_optional_skip")
    assert asserted_paths["subtribe_node_id"].str.len_chars().min() > 0
    assert skipped_paths["subtribe_node_id"].unique().to_list() == [""]
    assert skipped_paths["skipped_ranks"].to_list()[0] == ["SUBTRIBE"]

    branches = paths.with_columns(
        pl.when(pl.col("subtribe_node_id") != "")
        .then(pl.col("subtribe_node_id"))
        .otherwise(pl.col("tribe_node_id"))
        .alias("branch_node_id")
    )
    genera_per_branch = branches.group_by("branch_node_id").agg(
        pl.col("genus_node_id").n_unique().alias("count")
    )
    assert genera_per_branch.height == FAMILY_COUNT * SUBFAMILIES_PER_FAMILY * TRIBES_PER_SUBFAMILY
    assert genera_per_branch["count"].min() == GENERA_PER_BRANCH
    assert genera_per_branch["count"].max() == 3

    selected = paths.filter(pl.col("genus_node_id").is_in(BENCHMARK_SELECTED_GENUS_NODE_IDS))
    selected_species_counts = selected.group_by("genus_node_id").agg(
        pl.col("species_node_id").n_unique().alias("count")
    )
    assert selected_species_counts.height == len(BENCHMARK_SELECTED_GENUS_NODE_IDS) == 3
    assert set(selected_species_counts["count"].to_list()) == {SPECIES_PER_SELECTED_GENUS}
    assert SPECIES_PER_SELECTED_GENUS >= 20
    assert selected["subtribe_node_id"].n_unique() == 1


def test_seven_family_fixture_fingerprints_are_deterministic_across_builds() -> None:
    first = build_seven_family_path_cascade_fixture()
    second = build_seven_family_path_cascade_fixture()

    assert first.manifest["created_at"] == second.manifest["created_at"]
    assert first.taxonomy_store.classification_fingerprint == (
        second.taxonomy_store.classification_fingerprint
    )
    assert first.taxonomy_store.hierarchy_fingerprint == second.taxonomy_store.hierarchy_fingerprint
    assert first.taxonomy_store.enabled_paths()["hierarchy_hash"].to_list() == (
        second.taxonomy_store.enabled_paths()["hierarchy_hash"].to_list()
    )
    assert first.taxonomy_store.rank_candidates("FAMILY")["node_id"].to_list() == (
        second.taxonomy_store.rank_candidates("FAMILY")["node_id"].to_list()
    )
