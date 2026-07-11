from __future__ import annotations

import polars as pl
import pytest

from biominer.bioclip.five_rank_store import FiveRankTaxonomyStore
from biominer.registry.classification_v2 import write_classification_v2_artifacts


def test_five_rank_store_exposes_rank_aware_children_and_leaf_candidates(tmp_path) -> None:
    registry = _write_registry(tmp_path)
    write_classification_v2_artifacts(
        registry,
        source_path="config/taxonomy/papilionoidea_classification_v2.json",
    )

    store = FiveRankTaxonomyStore.read(registry)

    family = store.candidates("FAMILY")
    subfamily = store.child_candidates(family["node_id"].to_list(), child_rank="SUBFAMILY")
    tribe = store.child_candidates(subfamily["node_id"].to_list(), child_rank="TRIBE")
    genus = store.child_candidates(tribe["node_id"].to_list(), child_rank="GENUS")
    species = store.species_candidates_for_genera(genus["node_id"].to_list())
    assert family["scientific_name"].to_list() == ["Papilionidae"]
    assert subfamily["scientific_name"].to_list() == ["Papilioninae"]
    assert tribe["scientific_name"].to_list() == ["Papilionini"]
    assert genus["scientific_name"].to_list() == ["Papilio"]
    assert species["scientific_name"].to_list() == ["Papilio demoleus"]
    assert store.gbif_mapping_for_species_nodes(species["node_id"].to_list())["accepted_taxon_key"].to_list() == [
        "gbif:1938069"
    ]


def test_five_rank_store_loads_only_requested_rank_partitions(tmp_path) -> None:
    registry = _write_registry(tmp_path)
    write_classification_v2_artifacts(registry, source_path="config/taxonomy/papilionoidea_classification_v2.json")

    store = FiveRankTaxonomyStore.read(registry, ranks=("FAMILY", "SUBFAMILY"))

    assert set(store.nodes["rank"].to_list()) == {"FAMILY", "SUBFAMILY"}
    assert set(store.prompt_labels["rank"].to_list()) == {"FAMILY", "SUBFAMILY"}
    assert store.edges.select("parent_rank", "child_rank").to_dicts() == [
        {"parent_rank": "FAMILY", "child_rank": "SUBFAMILY"}
    ]


def test_five_rank_store_rejects_tampered_artifact(tmp_path) -> None:
    registry = _write_registry(tmp_path)
    write_classification_v2_artifacts(registry, source_path="config/taxonomy/papilionoidea_classification_v2.json")
    prompts = pl.read_parquet(registry / "classification_prompt_labels.parquet")
    prompts.with_columns(pl.lit("tampered").alias("label")).write_parquet(
        registry / "classification_prompt_labels.parquet"
    )

    with pytest.raises(ValueError, match="artifact checksum mismatch: prompt_labels"):
        FiveRankTaxonomyStore.read(registry)


def _write_registry(tmp_path):  # noqa: ANN001, ANN202 - compact fixture helper.
    registry = tmp_path / "registry"
    registry.mkdir()
    pl.DataFrame(
        [
            {
                "registry_version": "butterflies-v2",
                "accepted_taxon_key": "gbif:1938069",
                "species_key": "gbif:1938069",
                "scientific_name": "Papilio demoleus",
                "species": "Papilio demoleus",
                "family": "Papilionidae",
                "rank": "SPECIES",
                "taxonomic_status": "ACCEPTED",
            }
        ]
    ).write_parquet(registry / "taxa.parquet")
    (registry / "manifest.json").write_text('{"registry_version":"butterflies-v2"}', encoding="utf-8")
    return registry
