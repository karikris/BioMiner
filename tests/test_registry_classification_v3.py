from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from biominer.registry.classification_v3 import (
    CLASSIFICATION_RANKS,
    CLASSIFICATION_V3_PROMPT_VERSION,
    CLASSIFICATION_V3_VERSION,
    EDGE_SCHEMA,
    GBIF_MAPPING_SCHEMA,
    LEAF_PATH_SCHEMA,
    NODE_SCHEMA,
    PROMPT_LABEL_SCHEMA,
    QA_FINDING_SCHEMA,
    SOURCE_SCHEMA,
    build_classification_v3_frames,
    classification_v3_artifact_paths,
    classification_v3_fingerprint,
    hierarchy_fingerprint,
    load_classification_v3_source,
    validate_classification_v3,
    write_classification_v3_artifacts,
)


def test_classification_v3_builds_complete_six_rank_path_with_exact_schemas() -> None:
    source = _six_rank_source()
    frames = build_classification_v3_frames(_taxa(), source)

    assert frames.sources.schema == SOURCE_SCHEMA
    assert frames.nodes.schema == NODE_SCHEMA
    assert frames.edges.schema == EDGE_SCHEMA
    assert frames.gbif_mappings.schema == GBIF_MAPPING_SCHEMA
    assert frames.leaf_paths.schema == LEAF_PATH_SCHEMA
    assert frames.prompt_labels.schema == PROMPT_LABEL_SCHEMA
    assert frames.nodes["rank"].to_list() == list(CLASSIFICATION_RANKS)
    assert frames.leaf_paths.select(
        "rank_path",
        "rank_path_node_ids",
        "skipped_ranks",
        "path_completeness",
        "enabled",
    ).to_dicts() == [
        {
            "rank_path": list(CLASSIFICATION_RANKS),
            "rank_path_node_ids": [f"test:{rank.casefold()}" for rank in CLASSIFICATION_RANKS],
            "skipped_ranks": [],
            "path_completeness": "complete",
            "enabled": True,
        }
    ]
    assert frames.leaf_paths.select(rank.casefold() for rank in CLASSIFICATION_RANKS).row(0) == (
        "Papilionidae",
        "Papilioninae",
        "Papilionini",
        "Papilionina",
        "Papilio",
        "Papilio demoleus",
    )
    assert frames.leaf_paths["hierarchy_hash"][0].startswith("sha256:")
    assert frames.prompt_labels["rank"].to_list() == list(CLASSIFICATION_RANKS)
    assert frames.prompt_labels["label"][0] == "a photo of a butterfly in family Papilionidae"
    assert validate_classification_v3(frames, taxa=_taxa()) == []


def test_curated_papilio_source_uses_reviewed_optional_subtribe_skip() -> None:
    source = load_classification_v3_source()
    frames = build_classification_v3_frames(_taxa(), source)

    skip = frames.edges.filter(pl.col("edge_type") == "reviewed_rank_skip").to_dicts()
    assert len(skip) == 1
    assert skip[0]["parent_rank"] == "TRIBE"
    assert skip[0]["child_rank"] == "GENUS"
    assert skip[0]["skipped_ranks"] == ["SUBTRIBE"]
    assert skip[0]["reviewed"] is True
    assert skip[0]["skip_reason"]
    assert frames.leaf_paths.select(
        "rank_path",
        "skipped_ranks",
        "subtribe_node_id",
        "subtribe",
        "path_completeness",
        "enabled",
    ).to_dicts() == [
        {
            "rank_path": ["FAMILY", "SUBFAMILY", "TRIBE", "GENUS", "SPECIES"],
            "skipped_ranks": ["SUBTRIBE"],
            "subtribe_node_id": "",
            "subtribe": "",
            "path_completeness": "reviewed_optional_skip",
            "enabled": True,
        }
    ]
    findings = validate_classification_v3(frames, taxa=_taxa())
    assert [(finding["severity"], finding["code"]) for finding in findings] == [
        ("warning", "optional_subtribe_skipped")
    ]


def test_unreviewed_subtribe_skip_is_disabled_and_fatal() -> None:
    source = load_classification_v3_source()
    source["edges"][2].pop("reviewed_by")

    frames = build_classification_v3_frames(_taxa(), source)
    findings = validate_classification_v3(frames, taxa=_taxa())

    skip = frames.edges.filter(pl.col("edge_type") == "reviewed_rank_skip").row(0, named=True)
    assert skip["enabled"] is False
    assert "unreviewed_edge" in skip["disabled_reason"]
    assert frames.leaf_paths["enabled"].to_list() == [False]
    assert "unreviewed_rank_skip" in _codes(findings, severity="fatal")


def test_cycle_is_detected_even_when_invalid_back_edge_is_disabled() -> None:
    source = _six_rank_source()
    source["edges"].append(
        {
            "parent_node_id": "test:species",
            "child_node_id": "test:family",
            "edge_type": "asserted_parent",
            "skipped_ranks": [],
            "skip_reason": "",
            "source_id": "test:taxonomy-v1",
            "evidence": "Synthetic invalid back edge",
            **_review(),
        }
    )

    frames = build_classification_v3_frames(_taxa(), source)
    findings = validate_classification_v3(frames, taxa=_taxa())

    back_edge = frames.edges.filter(pl.col("parent_node_id") == "test:species").row(0, named=True)
    assert back_edge["enabled"] is False
    assert {"cycle_detected", "invalid_rank_transition"} <= _codes(findings, severity="fatal")


def test_multiple_enabled_parents_create_duplicate_paths_and_fail_qa() -> None:
    source = _six_rank_source()
    source["nodes"].append(
        {
            "node_id": "test:alternate-tribe",
            "rank": "TRIBE",
            "scientific_name": "Alternateini",
            "source_id": "test:taxonomy-v1",
            "evidence": "Synthetic alternate tribe",
            **_review(),
        }
    )
    source["edges"].extend(
        [
            {
                "parent_node_id": "test:subfamily",
                "child_node_id": "test:alternate-tribe",
                "edge_type": "asserted_parent",
                "skipped_ranks": [],
                "skip_reason": "",
                "source_id": "test:taxonomy-v1",
                "evidence": "Synthetic adjacent edge",
                **_review(),
            },
            {
                "parent_node_id": "test:alternate-tribe",
                "child_node_id": "test:subtribe",
                "edge_type": "asserted_parent",
                "skipped_ranks": [],
                "skip_reason": "",
                "source_id": "test:taxonomy-v1",
                "evidence": "Synthetic second parent edge",
                **_review(),
            },
        ]
    )

    frames = build_classification_v3_frames(_taxa(), source)
    findings = validate_classification_v3(frames, taxa=_taxa())

    assert frames.leaf_paths.filter(pl.col("enabled")).height == 2
    assert {
        "enabled_node_has_multiple_enabled_parents",
        "duplicate_enabled_species_path",
    } <= _codes(findings, severity="fatal")


@pytest.mark.parametrize(
    ("status", "include_status", "expected_enabled", "expected_status"),
    [
        ("ACCEPTED", True, True, "ACCEPTED"),
        ("DOUBTFUL", True, False, "DOUBTFUL"),
        ("", False, False, ""),
    ],
)
def test_only_explicitly_accepted_gbif_species_enable_mappings(
    status: str,
    include_status: bool,
    expected_enabled: bool,
    expected_status: str,
) -> None:
    taxa = _taxa(status=status, include_status=include_status)
    frames = build_classification_v3_frames(taxa, _six_rank_source())
    mapping = frames.gbif_mappings.row(0, named=True)
    codes = _codes(validate_classification_v3(frames, taxa=taxa))

    assert mapping["taxonomic_status"] == expected_status
    assert mapping["enabled"] is expected_enabled
    if expected_enabled:
        assert "enabled_species_maps_to_nonaccepted_gbif_taxon" not in codes
        assert "enabled_species_missing_gbif_mapping" not in codes
    else:
        assert {
            "enabled_species_maps_to_nonaccepted_gbif_taxon",
            "enabled_species_missing_gbif_mapping",
        } <= codes
    if not include_status:
        assert "unmapped_accepted_species" not in codes


def test_enabled_species_without_mapping_fails_qa() -> None:
    source = _six_rank_source()
    source["species_mappings"] = []

    frames = build_classification_v3_frames(_taxa(), source)

    assert frames.gbif_mappings.is_empty()
    assert "enabled_species_missing_gbif_mapping" in _codes(
        validate_classification_v3(frames, taxa=_taxa()), severity="fatal"
    )


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        ("unknown_rank", "unknown_rank"),
        ("invalid_transition", "invalid_rank_transition"),
        ("invalid_skip", "invalid_rank_skip"),
    ],
)
def test_strict_rank_contract_rejects_malformed_source(mutate: str, expected_code: str) -> None:
    source = _six_rank_source()
    if mutate == "unknown_rank":
        source["nodes"][3]["rank"] = "INFRAFAMILY"
    elif mutate == "invalid_transition":
        source["edges"][2].update(
            {
                "child_node_id": "test:genus",
                "edge_type": "asserted_parent",
            }
        )
    else:
        source["edges"][2].update(
            {
                "child_node_id": "test:genus",
                "edge_type": "reviewed_rank_skip",
                "skipped_ranks": ["SUBFAMILY"],
                "skip_reason": "Synthetic malformed skip",
            }
        )

    frames = build_classification_v3_frames(_taxa(), source)

    assert expected_code in _codes(validate_classification_v3(frames, taxa=_taxa()), severity="fatal")


def test_cross_table_qa_detects_disabled_node_missing_label_and_rank_gap() -> None:
    frames = build_classification_v3_frames(_taxa(), _six_rank_source())
    node_rows = frames.nodes.to_dicts()
    next(row for row in node_rows if row["node_id"] == "test:species")["enabled"] = False
    label_rows = frames.prompt_labels.to_dicts()
    label_rows[0]["node_id"] = "test:missing"
    path_rows = frames.leaf_paths.to_dicts()
    path_rows[0]["tribe_node_id"] = ""
    path_rows[0]["tribe"] = ""
    path_rows[0]["rank_path"] = [rank for rank in path_rows[0]["rank_path"] if rank != "TRIBE"]
    path_rows[0]["rank_path_node_ids"] = [
        node_id for node_id in path_rows[0]["rank_path_node_ids"] if node_id != "test:tribe"
    ]
    tampered = replace(
        frames,
        nodes=pl.DataFrame(node_rows, schema=NODE_SCHEMA),
        leaf_paths=pl.DataFrame(path_rows, schema=LEAF_PATH_SCHEMA),
        prompt_labels=pl.DataFrame(label_rows, schema=PROMPT_LABEL_SCHEMA),
    )

    codes = _codes(validate_classification_v3(tampered, taxa=_taxa()), severity="fatal")

    assert {
        "enabled_path_contains_disabled_node",
        "enabled_label_references_missing_node",
        "missing_mandatory_rank",
    } <= codes


def test_enabled_genus_without_enabled_species_child_warns() -> None:
    frames = build_classification_v3_frames(_taxa(), _six_rank_source())
    edges = frames.edges.filter(pl.col("child_rank") != "SPECIES")
    tampered = replace(frames, edges=edges)

    assert "enabled_genus_has_no_species" in _codes(validate_classification_v3(tampered, taxa=_taxa()))


def test_artifacts_have_exact_checksums_fingerprints_and_nested_schemas(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    _taxa().write_parquet(registry / "taxa.parquet")
    (registry / "manifest.json").write_text(
        json.dumps({"registry_version": "butterflies-v3", "qa_status": "passed"}),
        encoding="utf-8",
    )

    manifest = write_classification_v3_artifacts(registry)
    paths = classification_v3_artifact_paths(registry)
    frames = build_classification_v3_frames(_taxa(), load_classification_v3_source())

    assert manifest["classification_version"] == CLASSIFICATION_V3_VERSION
    assert manifest["prompt_version"] == CLASSIFICATION_V3_PROMPT_VERSION
    assert manifest["rank_order"] == list(CLASSIFICATION_RANKS)
    assert manifest["classification_fingerprint"] == classification_v3_fingerprint(frames)
    assert manifest["hierarchy_fingerprint"] == hierarchy_fingerprint(frames)
    assert manifest["qa_status"] == "passed"
    assert manifest["warning_finding_count"] == 1
    for key, metadata in manifest["artifacts"].items():
        path = paths[key]
        assert path.stat().st_size == metadata["bytes"]
        assert "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
    assert pl.read_parquet(paths["sources"]).schema == SOURCE_SCHEMA
    assert pl.read_parquet(paths["nodes"]).schema == NODE_SCHEMA
    assert pl.read_parquet(paths["edges"]).schema == EDGE_SCHEMA
    assert pl.read_parquet(paths["gbif_mappings"]).schema == GBIF_MAPPING_SCHEMA
    assert pl.read_parquet(paths["leaf_paths"]).schema == LEAF_PATH_SCHEMA
    assert pl.read_parquet(paths["prompt_labels"]).schema == PROMPT_LABEL_SCHEMA
    assert pl.read_parquet(paths["qa_findings"]).schema == QA_FINDING_SCHEMA


def test_fingerprints_are_invariant_to_source_and_frame_row_order() -> None:
    source = _six_rank_source()
    frames = build_classification_v3_frames(_taxa(), source)
    reversed_source = deepcopy(source)
    for key in ("sources", "nodes", "edges", "species_mappings"):
        reversed_source[key].reverse()
    rebuilt = build_classification_v3_frames(_taxa(), reversed_source)
    reversed_frames = replace(
        frames,
        sources=frames.sources.reverse(),
        nodes=frames.nodes.reverse(),
        edges=frames.edges.reverse(),
        gbif_mappings=frames.gbif_mappings.reverse(),
        leaf_paths=frames.leaf_paths.reverse(),
        prompt_labels=frames.prompt_labels.reverse(),
    )

    assert rebuilt.leaf_paths["hierarchy_hash"].to_list() == frames.leaf_paths["hierarchy_hash"].to_list()
    assert classification_v3_fingerprint(rebuilt) == classification_v3_fingerprint(frames)
    assert hierarchy_fingerprint(rebuilt) == hierarchy_fingerprint(frames)
    assert classification_v3_fingerprint(reversed_frames) == classification_v3_fingerprint(frames)
    assert hierarchy_fingerprint(reversed_frames) == hierarchy_fingerprint(frames)


def test_v3_writer_refuses_v2_manifest_without_overwriting_it(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    _taxa().write_parquet(registry / "taxa.parquet")
    existing = {"classification_version": "butterfly-classification-v2.0.0", "sentinel": "unchanged"}
    manifest_path = registry / "classification_manifest.json"
    manifest_path.write_text(json.dumps(existing), encoding="utf-8")

    with pytest.raises(ValueError, match="classification artifact version conflict"):
        write_classification_v3_artifacts(registry)

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == existing
    assert not (registry / "classification_nodes.parquet").exists()


def test_v3_builder_rejects_source_declaring_v2() -> None:
    source = _six_rank_source()
    source["classification_version"] = "butterfly-classification-v2.0.0"

    with pytest.raises(ValueError, match="classification source version mismatch"):
        build_classification_v3_frames(_taxa(), source)


def _taxa(*, status: str = "ACCEPTED", include_status: bool = True) -> pl.DataFrame:
    row: dict[str, object] = {
        "registry_version": "butterflies-v3",
        "accepted_taxon_key": "gbif:1938069",
        "species_key": "gbif:1938069",
        "scientific_name": "Papilio demoleus",
        "family": "Papilionidae",
        "genus": "Papilio",
        "species": "Papilio demoleus",
        "rank": "SPECIES",
    }
    if include_status:
        row["taxonomic_status"] = status
    return pl.DataFrame([row])


def _six_rank_source() -> dict[str, object]:
    rank_names = (
        ("FAMILY", "Papilionidae"),
        ("SUBFAMILY", "Papilioninae"),
        ("TRIBE", "Papilionini"),
        ("SUBTRIBE", "Papilionina"),
        ("GENUS", "Papilio"),
        ("SPECIES", "Papilio demoleus"),
    )
    nodes = [
        {
            "node_id": f"test:{rank.casefold()}",
            "rank": rank,
            "scientific_name": name,
            "source_id": "test:taxonomy-v1",
            "evidence": f"Synthetic reviewed {rank} evidence",
            **_review(),
        }
        for rank, name in rank_names
    ]
    edges = [
        {
            "parent_node_id": f"test:{parent_rank.casefold()}",
            "child_node_id": f"test:{child_rank.casefold()}",
            "edge_type": "asserted_parent",
            "skipped_ranks": [],
            "skip_reason": "",
            "source_id": "test:taxonomy-v1",
            "evidence": f"Synthetic {parent_rank} to {child_rank} evidence",
            **_review(),
        }
        for parent_rank, child_rank in zip(CLASSIFICATION_RANKS[:-1], CLASSIFICATION_RANKS[1:], strict=True)
    ]
    return {
        "classification_version": CLASSIFICATION_V3_VERSION,
        "sources": [
            {
                "source_id": "test:taxonomy-v1",
                "authority": "Synthetic taxonomy fixture",
                "release": "v1",
                "citation": "Synthetic fixture citation",
                "retrieved_at": "2026-07-11",
                "evidence_url": "https://example.test/taxonomy/v1",
                "evidence": "Synthetic reviewed taxonomy evidence",
            }
        ],
        "nodes": nodes,
        "edges": edges,
        "species_mappings": [
            {
                "gbif_species_key": "1938069",
                "accepted_scientific_name": "Papilio demoleus",
                "species_node_id": "test:species",
                "source_id": "test:taxonomy-v1",
                "evidence": "Synthetic exact accepted mapping evidence",
                **_review(),
            }
        ],
    }


def _review() -> dict[str, object]:
    return {
        "reviewed": True,
        "review_status": "reviewed",
        "reviewed_by": "BioMiner test taxonomy reviewer",
        "reviewed_at": "2026-07-11",
        "enabled": True,
    }


def _codes(findings: list[dict[str, object]], *, severity: str | None = None) -> set[str]:
    return {
        str(finding["code"])
        for finding in findings
        if severity is None or finding["severity"] == severity
    }
