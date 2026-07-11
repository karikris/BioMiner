from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from biominer.bioclip.path_taxonomy_store import PathTaxonomyStore
from biominer.registry.classification_v3 import (
    CLASSIFICATION_RANKS,
    CLASSIFICATION_V3_PROMPT_VERSION,
    CLASSIFICATION_V3_VERSION,
    LEAF_PATH_SCHEMA,
    classification_v3_artifact_paths,
    write_classification_v3_artifacts,
)


def test_path_store_loads_verified_v3_artifacts_and_enabled_paths(tmp_path: Path) -> None:
    store = PathTaxonomyStore.read(_write_registry(tmp_path))
    paths = store.enabled_paths()

    assert store.classification_version == CLASSIFICATION_V3_VERSION
    assert store.prompt_version == CLASSIFICATION_V3_PROMPT_VERSION
    assert store.hierarchy_fingerprint == store.manifest["hierarchy_fingerprint"]
    assert store.classification_fingerprint == store.manifest["classification_fingerprint"]
    assert store.manifest["qa_status"] == "passed"
    assert store.manifest["fatal_finding_count"] == 0
    assert paths.height == 7
    assert paths["accepted_taxon_key"].to_list() == [
        "gbif:1001",
        "gbif:1002",
        "gbif:1003",
        "gbif:1004",
        "gbif:1005",
        "gbif:1006",
        "gbif:1007",
    ]
    assert [finding["code"] for finding in store.validation_findings()] == [
        "optional_subtribe_skipped",
        "optional_subtribe_skipped",
        "optional_subtribe_skipped",
    ]


def test_path_store_builds_validated_cloud_equivalent_from_exact_frames(
    tmp_path: Path,
) -> None:
    local = PathTaxonomyStore.read(_write_registry(tmp_path))

    restored = PathTaxonomyStore.from_frames(
        sources=local.sources,
        nodes=local.nodes,
        edges=local.edges,
        gbif_mappings=local.gbif_mappings,
        leaf_paths=local.leaf_paths,
        prompt_labels=local.prompt_labels,
        qa_findings=local.qa_findings,
        manifest=local.manifest,
    )

    assert restored.classification_fingerprint == local.classification_fingerprint
    assert restored.hierarchy_fingerprint == local.hierarchy_fingerprint
    assert restored.enabled_paths().equals(local.enabled_paths())

    with pytest.raises(ValueError, match="artifact schema mismatch: nodes"):
        PathTaxonomyStore.from_frames(
            sources=local.sources,
            nodes=local.nodes.with_columns(pl.col("enabled").cast(pl.Int8)),
            edges=local.edges,
            gbif_mappings=local.gbif_mappings,
            leaf_paths=local.leaf_paths,
            prompt_labels=local.prompt_labels,
            qa_findings=local.qa_findings,
            manifest=local.manifest,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("classification_version", "butterfly-classification-v2.0.0", "manifest version mismatch"),
        ("prompt_version", "wrong-prompts", "manifest prompt version mismatch"),
        ("qa_status", "failed", "did not pass fatal QA"),
        ("fatal_finding_count", 1, "did not pass fatal QA"),
        ("hierarchy_fingerprint", "sha256:wrong", "hierarchy fingerprint mismatch"),
        ("classification_fingerprint", "sha256:wrong", "classification fingerprint mismatch"),
        ("warning_finding_count", 0, "warning QA count mismatch"),
    ],
)
def test_path_store_rejects_invalid_manifest(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    registry = _write_registry(tmp_path)
    path = registry / "classification_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        PathTaxonomyStore.read(registry)


@pytest.mark.parametrize(
    "artifact",
    [
        "sources",
        "nodes",
        "edges",
        "gbif_mappings",
        "leaf_paths",
        "prompt_labels",
        "qa_findings",
    ],
)
def test_path_store_rejects_every_tampered_artifact(tmp_path: Path, artifact: str) -> None:
    registry = _write_registry(tmp_path)
    path = classification_v3_artifact_paths(registry)[artifact]
    with path.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match=rf"artifact (byte count|checksum) mismatch: {artifact}"):
        PathTaxonomyStore.read(registry)


def test_path_store_rejects_wrong_physical_artifact_schema(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path)
    path = classification_v3_artifact_paths(registry)["leaf_paths"]
    pl.read_parquet(path).with_columns(pl.col("enabled").cast(pl.Int8)).write_parquet(path)
    _refresh_artifact_metadata(registry, "leaf_paths")

    with pytest.raises(ValueError, match="artifact schema mismatch: leaf_paths"):
        PathTaxonomyStore.read(registry)


def test_rank_candidates_are_path_reachable_unique_and_id_based(tmp_path: Path) -> None:
    store = PathTaxonomyStore.read(_write_registry(tmp_path))

    assert {
        rank: store.rank_candidates(rank).height
        for rank in CLASSIFICATION_RANKS
    } == {
        "FAMILY": 3,
        "SUBFAMILY": 6,
        "TRIBE": 6,
        "SUBTRIBE": 3,
        "GENUS": 6,
        "SPECIES": 7,
    }
    genera = store.rank_candidates("GENUS").select("node_id", "scientific_name").to_dicts()
    assert genera == [
        {"node_id": "fixture:genus:a1", "scientific_name": "Alpha"},
        {"node_id": "fixture:genus:b2", "scientific_name": "Beta"},
        {"node_id": "fixture:genus:c2", "scientific_name": "Ceres"},
        {"node_id": "fixture:genus:a2", "scientific_name": "Duplicata"},
        {"node_id": "fixture:genus:b1", "scientific_name": "Duplicata"},
        {"node_id": "fixture:genus:c1", "scientific_name": "Gamma"},
    ]


def test_active_path_queries_deduplicate_nodes_and_filter_deterministically(tmp_path: Path) -> None:
    store = PathTaxonomyStore.read(_write_registry(tmp_path))
    enabled = store.enabled_paths()
    duplicated = pl.concat([enabled.reverse(), enabled.head(1)])

    families = store.candidate_nodes_in_paths(duplicated, "FAMILY")
    assert families["node_id"].to_list() == [
        "fixture:family:a",
        "fixture:family:b",
        "fixture:family:c",
    ]
    assert store.candidate_nodes_in_paths(duplicated, "GENUS").height == 6

    active = store.filter_paths_by_rank_nodes(
        duplicated,
        "FAMILY",
        ["fixture:family:c", "fixture:family:a", "fixture:family:a"],
    )
    assert active.height == 5
    assert active["accepted_taxon_key"].to_list() == [
        "gbif:1001",
        "gbif:1002",
        "gbif:1003",
        "gbif:1006",
        "gbif:1007",
    ]
    assert store.candidate_nodes_in_paths(active, "SUBFAMILY")["node_id"].to_list() == [
        "fixture:subfamily:a1",
        "fixture:subfamily:a2",
        "fixture:subfamily:c1",
        "fixture:subfamily:c2",
    ]
    species = store.species_nodes_in_paths(active)
    assert species["node_id"].to_list() == [
        "fixture:species:a1-1",
        "fixture:species:a1-2",
        "fixture:species:c2",
        "fixture:species:a2",
        "fixture:species:c1",
    ]


def test_active_path_queries_reject_foreign_and_nullable_rows(tmp_path: Path) -> None:
    store = PathTaxonomyStore.read(_write_registry(tmp_path))
    active = store.enabled_paths()
    foreign = active.head(1).with_columns(pl.lit("sha256:foreign").alias("hierarchy_hash"))
    nullable = active.head(1).with_columns(pl.lit(None, dtype=pl.Boolean).alias("enabled"))

    with pytest.raises(ValueError, match="do not belong to this taxonomy store"):
        store.candidate_nodes_in_paths(foreign, "FAMILY")
    with pytest.raises(ValueError, match="contain disabled rows"):
        store.candidate_nodes_in_paths(nullable, "FAMILY")


def test_species_path_lookups_are_sorted_and_singular(tmp_path: Path) -> None:
    store = PathTaxonomyStore.read(_write_registry(tmp_path))

    paths = store.paths_for_species_nodes(
        [
            "fixture:species:c2",
            "fixture:species:a1-2",
            "fixture:species:a1-1",
            "fixture:species:a1-1",
        ]
    )
    assert paths["species"].to_list() == ["Alpha alba", "Alpha azurea", "Ceres euca"]
    assert store.path_for_species_node("fixture:species:a1-1")["accepted_taxon_key"] == "gbif:1001"
    with pytest.raises(KeyError, match="species path not found"):
        store.path_for_species_node("fixture:species:missing")

    duplicate = store.leaf_paths.filter(pl.col("species_node_id") == "fixture:species:a1-1").row(
        0,
        named=True,
    )
    duplicate["hierarchy_hash"] = "sha256:synthetic-second-path"
    ambiguous = replace(
        store,
        leaf_paths=pl.concat(
            [store.leaf_paths, pl.DataFrame([duplicate], schema=LEAF_PATH_SCHEMA)]
        ),
    )
    with pytest.raises(ValueError, match="multiple enabled paths"):
        ambiguous.path_for_species_node("fixture:species:a1-1")


def test_prompt_and_mapping_queries_preserve_same_name_node_identities(tmp_path: Path) -> None:
    store = PathTaxonomyStore.read(_write_registry(tmp_path))
    duplicate_name_ids = ["fixture:genus:b1", "fixture:genus:a2", "fixture:genus:a2"]

    prompts = store.prompt_rows_for_nodes(duplicate_name_ids, "rank_screen")
    assert prompts["node_id"].to_list() == [
        "fixture:genus:a2",
        "fixture:genus:a2",
        "fixture:genus:b1",
        "fixture:genus:b1",
    ]
    assert prompts["scientific_name"].to_list() == ["Duplicata"] * 4
    assert prompts["label"].n_unique() == 2
    assert prompts["prompt_stage"].unique().to_list() == ["rank_screen"]
    with pytest.raises(ValueError, match="species_rerank prompts require"):
        store.prompt_rows_for_nodes(duplicate_name_ids, "species_rerank")

    species_ids = ["fixture:species:a1-1", "fixture:species:c2"]
    first_pass = store.prompt_rows_for_nodes(species_ids, "species_first_pass")
    rerank = store.prompt_rows_for_nodes(species_ids, "species_rerank")
    assert first_pass.height == rerank.height == 4
    assert set(first_pass["label"].to_list()).isdisjoint(rerank["label"].to_list())
    assert first_pass["node_id"].unique().sort().to_list() == sorted(species_ids)
    assert rerank["node_id"].unique().sort().to_list() == sorted(species_ids)

    mappings = store.mappings_for_species_nodes(
        ["fixture:species:c2", "fixture:species:a1-1", "fixture:species:c2"]
    )
    assert mappings.select("accepted_taxon_key", "taxonomic_status").to_dicts() == [
        {"accepted_taxon_key": "gbif:1001", "taxonomic_status": "ACCEPTED"},
        {"accepted_taxon_key": "gbif:1007", "taxonomic_status": "ACCEPTED"},
    ]


def test_mixed_optional_subtribe_paths_carry_only_active_reviewed_skips(tmp_path: Path) -> None:
    store = PathTaxonomyStore.read(_write_registry(tmp_path))
    active = store.filter_paths_by_rank_nodes(
        store.enabled_paths(),
        "TRIBE",
        [
            "fixture:tribe:a1",
            "fixture:tribe:a2",
            "fixture:tribe:b1",
            "fixture:tribe:c2",
        ],
    )

    assert active.height == 5
    assert store.candidate_nodes_in_paths(active, "SUBTRIBE")["node_id"].to_list() == [
        "fixture:subtribe:a1",
        "fixture:subtribe:b1",
    ]
    assert store.reviewed_skip_paths(active, "SUBTRIBE")["species_node_id"].to_list() == [
        "fixture:species:a2",
        "fixture:species:c2",
    ]
    carried = store.filter_paths_by_rank_nodes(
        active,
        "SUBTRIBE",
        ["fixture:subtribe:a1"],
        carry_reviewed_skip_paths=True,
    )
    assert carried["accepted_taxon_key"].to_list() == [
        "gbif:1001",
        "gbif:1002",
        "gbif:1003",
        "gbif:1007",
    ]
    asserted_only = store.filter_paths_by_rank_nodes(
        active,
        "SUBTRIBE",
        ["fixture:subtribe:a1"],
    )
    assert asserted_only["accepted_taxon_key"].to_list() == ["gbif:1001", "gbif:1002"]


def test_fully_skipped_optional_rank_preserves_active_paths_without_placeholders(tmp_path: Path) -> None:
    store = PathTaxonomyStore.read(_write_registry(tmp_path))
    active = store.filter_paths_by_rank_nodes(
        store.enabled_paths(),
        "TRIBE",
        ["fixture:tribe:c2"],
    )

    assert store.candidate_nodes_in_paths(active, "SUBTRIBE").is_empty()
    assert store.paths_with_asserted_rank(active, "SUBTRIBE").is_empty()
    skips = store.reviewed_skip_paths(active, "SUBTRIBE")
    assert skips["hierarchy_hash"].to_list() == active["hierarchy_hash"].to_list()
    carried = store.filter_paths_by_rank_nodes(
        active,
        "SUBTRIBE",
        [],
        carry_reviewed_skip_paths=True,
    )
    assert carried["hierarchy_hash"].to_list() == active["hierarchy_hash"].to_list()
    assert not store.nodes.filter(pl.col("node_id").str.contains("skip", literal=True)).height


def _write_registry(tmp_path: Path) -> Path:
    registry = tmp_path / "registry"
    registry.mkdir()
    source, taxa = _classification_fixture()
    source_path = tmp_path / "classification_source.json"
    source_path.write_text(json.dumps(source, indent=2), encoding="utf-8")
    taxa.write_parquet(registry / "taxa.parquet")
    (registry / "manifest.json").write_text(
        json.dumps({"registry_version": "path-store-fixture-v3", "qa_status": "passed"}),
        encoding="utf-8",
    )
    write_classification_v3_artifacts(registry, source_path=source_path)
    return registry


def _refresh_artifact_metadata(registry: Path, artifact: str) -> None:
    paths = classification_v3_artifact_paths(registry)
    artifact_path = paths[artifact]
    manifest_path = paths["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest["artifacts"][artifact]
    metadata["bytes"] = artifact_path.stat().st_size
    metadata["sha256"] = "sha256:" + hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _classification_fixture() -> tuple[dict[str, object], pl.DataFrame]:
    nodes: dict[str, dict[str, object]] = {}
    edges: dict[tuple[str, str], dict[str, object]] = {}
    mappings: list[dict[str, object]] = []
    taxa: list[dict[str, object]] = []
    for path in _path_specs():
        lineage = [
            ("FAMILY", path["family"]),
            ("SUBFAMILY", path["subfamily"]),
            ("TRIBE", path["tribe"]),
        ]
        if path["subtribe"] is not None:
            lineage.append(("SUBTRIBE", path["subtribe"]))
        lineage.extend(
            [
                ("GENUS", path["genus"]),
                ("SPECIES", path["species"]),
            ]
        )
        for rank, value in lineage:
            node_id, name = value
            candidate = {
                "node_id": node_id,
                "rank": rank,
                "scientific_name": name,
                "source_id": "fixture:taxonomy-v1",
                "evidence": f"Reviewed fixture evidence for {rank} {name}",
                **_review(),
            }
            existing = nodes.setdefault(node_id, candidate)
            assert existing["rank"] == rank and existing["scientific_name"] == name
        for (parent_rank, parent), (child_rank, child) in zip(lineage[:-1], lineage[1:], strict=True):
            parent_id, parent_name = parent
            child_id, child_name = child
            skipped = parent_rank == "TRIBE" and child_rank == "GENUS"
            candidate_edge = {
                "parent_node_id": parent_id,
                "child_node_id": child_id,
                "edge_type": "reviewed_rank_skip" if skipped else "asserted_parent",
                "skipped_ranks": ["SUBTRIBE"] if skipped else [],
                "skip_reason": (
                    f"Reviewed fixture lineage places {child_name} below {parent_name} without a SUBTRIBE assertion"
                    if skipped
                    else ""
                ),
                "source_id": "fixture:taxonomy-v1",
                "evidence": f"Reviewed fixture edge {parent_id} to {child_id}",
                **_review(),
            }
            existing_edge = edges.setdefault((parent_id, child_id), candidate_edge)
            assert existing_edge == candidate_edge
        species_id, species_name = path["species"]
        gbif_key = path["gbif_key"]
        mappings.append(
            {
                "gbif_species_key": gbif_key,
                "accepted_scientific_name": species_name,
                "species_node_id": species_id,
                "source_id": "fixture:taxonomy-v1",
                "evidence": f"Exact accepted fixture mapping {gbif_key}",
                **_review(),
            }
        )
        family_id, family_name = path["family"]
        genus_id, genus_name = path["genus"]
        taxa.append(
            {
                "registry_version": "path-store-fixture-v3",
                "accepted_taxon_key": f"gbif:{gbif_key}",
                "species_key": f"gbif:{gbif_key}",
                "scientific_name": species_name,
                "family_key": family_id,
                "family": family_name,
                "genus_key": genus_id,
                "genus": genus_name,
                "species": species_name,
                "rank": "SPECIES",
                "taxonomic_status": "ACCEPTED",
            }
        )
    source = {
        "classification_version": CLASSIFICATION_V3_VERSION,
        "sources": [
            {
                "source_id": "fixture:taxonomy-v1",
                "authority": "BioMiner synthetic taxonomy fixture",
                "release": "v1",
                "citation": "Synthetic Phase 2 path-store fixture",
                "retrieved_at": "2026-07-11",
                "evidence_url": "https://example.test/biominer/path-store-v1",
                "evidence": "Reviewed synthetic branching taxonomy",
            }
        ],
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
        "species_mappings": mappings,
    }
    return source, pl.DataFrame(taxa)


def _path_specs() -> list[dict[str, object]]:
    return [
        _path("a", "a1", "Alphaidae", "Alphinae", "Alphini", "Alphina", "Alpha", "a1-1", "Alpha alba", "1001"),
        _path("a", "a1", "Alphaidae", "Alphinae", "Alphini", "Alphina", "Alpha", "a1-2", "Alpha azurea", "1002"),
        _path("a", "a2", "Alphaidae", "Alphoxinae", "Alphoxini", None, "Duplicata", "a2", "Duplicata alpha", "1003"),
        _path("b", "b1", "Betaidae", "Betinae", "Betini", "Betina", "Duplicata", "b1", "Duplicata beta", "1004"),
        _path("b", "b2", "Betaidae", "Betoxinae", "Betoxini", None, "Beta", "b2", "Beta clara", "1005"),
        _path("c", "c1", "Gammaidae", "Gamminae", "Gammini", "Gammina", "Gamma", "c1", "Gamma densa", "1006"),
        _path("c", "c2", "Gammaidae", "Gammoxinae", "Gammoxini", None, "Ceres", "c2", "Ceres euca", "1007"),
    ]


def _path(
    family_suffix: str,
    branch_suffix: str,
    family_name: str,
    subfamily_name: str,
    tribe_name: str,
    subtribe_name: str | None,
    genus_name: str,
    species_suffix: str,
    species_name: str,
    gbif_key: str,
) -> dict[str, object]:
    return {
        "family": (f"fixture:family:{family_suffix}", family_name),
        "subfamily": (f"fixture:subfamily:{branch_suffix}", subfamily_name),
        "tribe": (f"fixture:tribe:{branch_suffix}", tribe_name),
        "subtribe": (
            (f"fixture:subtribe:{branch_suffix}", subtribe_name)
            if subtribe_name is not None
            else None
        ),
        "genus": (f"fixture:genus:{branch_suffix}", genus_name),
        "species": (f"fixture:species:{species_suffix}", species_name),
        "gbif_key": gbif_key,
    }


def _review() -> dict[str, object]:
    return {
        "reviewed": True,
        "review_status": "reviewed",
        "reviewed_by": "BioMiner Phase 2 fixture reviewer",
        "reviewed_at": "2026-07-11",
        "enabled": True,
    }
