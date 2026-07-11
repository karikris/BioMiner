from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json

import polars as pl
import pytest

from biominer.registry.classification_v2 import (
    CLASSIFICATION_RANKS,
    CLASSIFICATION_V2_PROMPT_VERSION,
    CLASSIFICATION_V2_VERSION,
    build_classification_v2_frames,
    build_classification_v2_manifest,
    classification_v2_artifact_paths,
    classification_v2_qa_frame,
    load_classification_v2_source,
    validate_classification_v2,
    write_classification_v2_artifacts,
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


def test_classification_v2_requires_explicit_reviewer_identity_and_date() -> None:
    source = _reviewed_source()
    source["nodes"][0].pop("reviewed_by")

    frames = build_classification_v2_frames(_taxa(), source)

    family = frames.nodes.filter(pl.col("rank") == "FAMILY").to_dicts()[0]
    assert family["reviewed"] is False
    assert family["enabled"] is False
    assert family["disabled_reason"] == "unreviewed_node"
    assert frames.leaf_paths["enabled"].to_list() == [False]


def test_curated_papilio_demoleus_source_writes_versioned_artifacts(tmp_path) -> None:
    source_path = "config/taxonomy/papilionoidea_classification_v2.json"
    source = load_classification_v2_source(source_path)
    assert source["species_mappings"][0]["gbif_species_key"] == "1938069"
    registry = tmp_path / "registry"
    registry.mkdir()
    _taxa().write_parquet(registry / "taxa.parquet")
    (registry / "manifest.json").write_text('{"registry_version":"butterflies-v2"}', encoding="utf-8")

    manifest = write_classification_v2_artifacts(registry, source_path=source_path)

    paths = classification_v2_artifact_paths(registry)
    assert all(path.exists() for path in paths.values())
    assert manifest["classification_version"] == CLASSIFICATION_V2_VERSION
    assert manifest["enabled_leaf_path_count"] == 1
    assert pl.read_parquet(paths["leaf_paths"]).select(
        "family", "subfamily", "tribe", "genus", "species", "gbif_species_key"
    ).to_dicts() == [
        {
            "family": "Papilionidae",
            "subfamily": "Papilioninae",
            "tribe": "Papilionini",
            "genus": "Papilio",
            "species": "Papilio demoleus",
            "gbif_species_key": "1938069",
        }
    ]


def test_classification_v2_qa_reports_invalid_transition_and_missing_path() -> None:
    source = deepcopy(_reviewed_source())
    source["edges"][1]["child_node_id"] = "genus:papilio"
    frames = build_classification_v2_frames(_taxa(), source)

    findings = validate_classification_v2(frames, taxa=_taxa())
    codes = {finding["code"] for finding in findings if finding["severity"] == "fatal"}

    assert "invalid_edge_rank_transition" in codes
    assert "no_enabled_leaf_path" in codes


def test_classification_v2_qa_emits_explicit_unmapped_species_gap() -> None:
    taxa = pl.concat(
        [
            _taxa(),
            pl.DataFrame(
                [
                    {
                        "registry_version": "butterflies-v2",
                        "accepted_taxon_key": "gbif:999",
                        "species_key": "gbif:999",
                        "scientific_name": "Papilio unmapped",
                        "species": "Papilio unmapped",
                        "rank": "SPECIES",
                        "taxonomic_status": "ACCEPTED",
                    }
                ]
            ),
        ],
        how="diagonal_relaxed",
    )
    frames = build_classification_v2_frames(taxa, _reviewed_source())

    findings = validate_classification_v2(frames, taxa=taxa)
    gaps = [finding for finding in findings if finding["code"] == "unmapped_accepted_species"]

    assert gaps == [
        {
            "severity": "warning",
            "code": "unmapped_accepted_species",
            "table": "classification_leaf_paths",
            "subject": "gbif:999",
            "message": "accepted GBIF species has no complete reviewed five-rank path",
            "details": {"scientific_name": "Papilio unmapped", "family": ""},
        }
    ]
    assert classification_v2_qa_frame(findings).filter(pl.col("code") == "unmapped_accepted_species")["subject"].to_list() == ["gbif:999"]


def test_classification_v2_qa_rejects_multiple_enabled_parents() -> None:
    source = deepcopy(_reviewed_source())
    source["nodes"].append(
        {
            "node_id": "family:alternate",
            "rank": "FAMILY",
            "scientific_name": "Alternateidae",
            "source_id": "ncbi-76202",
            **_review(),
            "enabled": True,
        }
    )
    source["edges"].append(
        {
            "parent_node_id": "family:alternate",
            "child_node_id": "subfamily:papilioninae",
            "source_id": "ncbi-76202",
            **_review(),
        }
    )
    frames = build_classification_v2_frames(_taxa(), source)

    codes = {finding["code"] for finding in validate_classification_v2(frames, taxa=_taxa())}

    assert "enabled_node_has_multiple_parents" in codes


def test_classification_v2_qa_rejects_enabled_cycles() -> None:
    frames = build_classification_v2_frames(_taxa(), _reviewed_source())
    reverse = dict(frames.edges.row(0, named=True))
    reverse.update(
        {
            "parent_node_id": "species:papilio-demoleus",
            "child_node_id": "family:papilionidae",
            "parent_rank": "SPECIES",
            "child_rank": "FAMILY",
            "enabled": True,
        }
    )
    cyclic = replace(frames, edges=pl.concat([frames.edges, pl.DataFrame([reverse], schema=frames.edges.schema)]))

    codes = {finding["code"] for finding in validate_classification_v2(cyclic, taxa=_taxa())}

    assert "invalid_edge_rank_transition" in codes
    assert "enabled_hierarchy_cycle" in codes


def test_classification_v2_fatal_qa_blocks_artifact_promotion(tmp_path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    _taxa().write_parquet(registry / "taxa.parquet")
    source = deepcopy(_reviewed_source())
    source["edges"] = source["edges"][:-1]
    source_path = tmp_path / "invalid-source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="classification-v2 fatal QA"):
        write_classification_v2_artifacts(registry, source_path=source_path)

    paths = classification_v2_artifact_paths(registry)
    assert paths["qa_findings"].exists()
    assert not paths["nodes"].exists()
    assert "no_enabled_leaf_path" in pl.read_parquet(paths["qa_findings"])["code"].to_list()


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
        **_review(),
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
            {"parent_node_id": "family:papilionidae", "child_node_id": "subfamily:papilioninae", "source_id": "ncbi-76202", **_review()},
            {"parent_node_id": "subfamily:papilioninae", "child_node_id": "tribe:papilionini", "source_id": "ncbi-76202", **_review()},
            {"parent_node_id": "tribe:papilionini", "child_node_id": "genus:papilio", "source_id": "ncbi-76202", **_review()},
            {"parent_node_id": "genus:papilio", "child_node_id": "species:papilio-demoleus", "source_id": "ncbi-76202", **_review()},
        ],
        "species_mappings": [
            {
                "gbif_species_key": "1938069",
                "accepted_scientific_name": "Papilio demoleus",
                "species_node_id": "species:papilio-demoleus",
                "source_id": "ncbi-76202",
                **_review(),
            }
        ],
    }


def _review() -> dict[str, object]:
    return {
        "reviewed": True,
        "review_status": "reviewed",
        "reviewed_by": "BioMiner taxonomy review",
        "reviewed_at": "2026-07-11",
    }
