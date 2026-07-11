from __future__ import annotations

import polars as pl

from biominer.registry.classification_v2 import (
    CLASSIFICATION_RANKS,
    CLASSIFICATION_V2_PROMPT_VERSION,
    CLASSIFICATION_V2_VERSION,
    build_classification_v2_frames,
    build_classification_v2_manifest,
)


def test_classification_v2_builds_exact_reviewed_five_rank_path() -> None:
    frames = build_classification_v2_frames(_taxa(), _reviewed_source())

    assert frames.nodes["rank"].to_list() == ["FAMILY", "GENUS", "SPECIES", "SUBFAMILY", "TRIBE"]
    assert frames.edges.filter(pl.col("enabled")).height == 4
    assert frames.gbif_mappings.select("accepted_taxon_key", "taxonomic_status", "enabled").to_dicts() == [
        {"accepted_taxon_key": "gbif:1938069", "taxonomic_status": "ACCEPTED", "enabled": True}
    ]
    assert frames.leaf_paths.select([rank.casefold() for rank in CLASSIFICATION_RANKS]).to_dicts() == [
        {
            "family": "Papilionidae",
            "subfamily": "Papilioninae",
            "tribe": "Papilionini",
            "genus": "Papilio",
            "species": "Papilio demoleus",
        }
    ]
    assert frames.leaf_paths["enabled"].to_list() == [True]
    assert frames.prompt_labels["rank"].to_list() == ["FAMILY", "GENUS", "SPECIES", "SUBFAMILY", "TRIBE"]
    assert frames.prompt_labels["prompt_version"].unique().to_list() == [CLASSIFICATION_V2_PROMPT_VERSION]


def test_classification_v2_manifest_counts_enabled_ranks_and_paths() -> None:
    frames = build_classification_v2_frames(_taxa(), _reviewed_source())

    manifest = build_classification_v2_manifest(frames, registry_version="butterflies-v2")

    assert manifest["classification_version"] == CLASSIFICATION_V2_VERSION
    assert manifest["registry_version"] == "butterflies-v2"
    assert manifest["rank_order"] == list(CLASSIFICATION_RANKS)
    assert manifest["enabled_node_counts_by_rank"] == {rank: 1 for rank in CLASSIFICATION_RANKS}
    assert manifest["enabled_leaf_path_count"] == 1


def test_classification_v2_disables_nonaccepted_gbif_mapping_and_leaf_path() -> None:
    taxa = _taxa().with_columns(pl.lit("DOUBTFUL").alias("taxonomic_status"))

    frames = build_classification_v2_frames(taxa, _reviewed_source())

    assert frames.gbif_mappings["enabled"].to_list() == [False]
    assert "gbif_species_not_accepted" in frames.gbif_mappings["disabled_reason"][0]
    assert frames.leaf_paths["enabled"].to_list() == [False]
    assert frames.prompt_labels.is_empty()


def _taxa() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "registry_version": "butterflies-v2",
                "accepted_taxon_key": "gbif:1938069",
                "species_key": "gbif:1938069",
                "scientific_name": "Papilio demoleus",
                "species": "Papilio demoleus",
                "rank": "SPECIES",
                "taxonomic_status": "ACCEPTED",
            }
        ]
    )


def _reviewed_source() -> dict[str, object]:
    node = lambda node_id, rank, name: {  # noqa: E731 - compact fixture factory.
        "node_id": node_id,
        "rank": rank,
        "scientific_name": name,
        "source_id": "ncbi-76202",
        "reviewed": True,
        "enabled": True,
    }
    return {
        "classification_version": CLASSIFICATION_V2_VERSION,
        "sources": [
            {
                "source_id": "ncbi-76202",
                "authority": "NCBI Taxonomy",
                "release": "record 76202 updated 2025-06-16",
                "citation": "NCBI Taxonomy record 76202",
                "retrieved_at": "2026-07-11",
                "evidence_url": "https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=76202",
                "evidence": "Papilionidae; Papilioninae; Papilionini; Papilio; Papilio demoleus",
            }
        ],
        "nodes": [
            node("family:papilionidae", "FAMILY", "Papilionidae"),
            node("subfamily:papilioninae", "SUBFAMILY", "Papilioninae"),
            node("tribe:papilionini", "TRIBE", "Papilionini"),
            node("genus:papilio", "GENUS", "Papilio"),
            node("species:papilio-demoleus", "SPECIES", "Papilio demoleus"),
        ],
        "edges": [
            {"parent_node_id": "family:papilionidae", "child_node_id": "subfamily:papilioninae", "source_id": "ncbi-76202", "reviewed": True},
            {"parent_node_id": "subfamily:papilioninae", "child_node_id": "tribe:papilionini", "source_id": "ncbi-76202", "reviewed": True},
            {"parent_node_id": "tribe:papilionini", "child_node_id": "genus:papilio", "source_id": "ncbi-76202", "reviewed": True},
            {"parent_node_id": "genus:papilio", "child_node_id": "species:papilio-demoleus", "source_id": "ncbi-76202", "reviewed": True},
        ],
        "species_mappings": [
            {
                "gbif_species_key": "1938069",
                "accepted_scientific_name": "Papilio demoleus",
                "species_node_id": "species:papilio-demoleus",
                "source_id": "ncbi-76202",
                "reviewed": True,
            }
        ],
    }
