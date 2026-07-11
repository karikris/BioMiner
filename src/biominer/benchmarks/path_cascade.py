"""Synthetic, model-free taxonomy inputs for path-cascade benchmarks.

The names and identifiers in this module are deliberately invented.  They are
not biological assertions, GBIF records, or a substitute for the reviewed
classification-v3 registry.  The fixture exists only to exercise branching,
optional-rank skips, and large species candidate sets deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import polars as pl

from biominer.bioclip.path_taxonomy_store import PathTaxonomyStore
from biominer.registry.classification_v3 import (
    ASSERTED_PARENT_EDGE,
    CLASSIFICATION_V3_VERSION,
    REVIEWED_RANK_SKIP_EDGE,
    ClassificationV3Frames,
    build_classification_v3_frames,
    build_classification_v3_manifest,
    classification_v3_fingerprint,
    classification_v3_qa_frame,
    hierarchy_fingerprint,
    validate_classification_v3,
)


BENCHMARK_FIXTURE_NOTICE = (
    "Synthetic developer benchmark only; names and identifiers are neither biological "
    "taxonomy nor GBIF authority."
)
BENCHMARK_REGISTRY_VERSION = "synthetic-seven-family-cascade-v1"
BENCHMARK_SOURCE_ID = "fixture:synthetic-taxonomy-v1"
BENCHMARK_REVIEW_DATE = "2026-07-11"
FAMILY_COUNT = 7
SUBFAMILIES_PER_FAMILY = 2
TRIBES_PER_SUBFAMILY = 2
GENERA_PER_BRANCH = 2
SPECIES_PER_SELECTED_GENUS = 25
BENCHMARK_SELECTED_GENUS_NODE_IDS = (
    "fixture:genus:01:01:01:01",
    "fixture:genus:01:01:01:02",
    "fixture:genus:01:01:01:03",
)


@dataclass(frozen=True)
class SevenFamilyPathCascadeFixture:
    """Validated classification-v3 frames and store for developer benchmarks."""

    taxa: pl.DataFrame
    frames: ClassificationV3Frames
    qa_findings: pl.DataFrame
    manifest: Mapping[str, object]
    taxonomy_store: PathTaxonomyStore


def build_seven_family_path_cascade_fixture() -> SevenFamilyPathCascadeFixture:
    """Build the small, deterministic seven-family path-cascade fixture.

    The first asserted-subtribe branch has three genera with 25 species each.
    Every other branch has two genera with one species each.  This keeps the
    fixture compact while supplying a realistic species top-20 candidate set.
    """
    taxa, source = _synthetic_source_and_taxa()
    frames = build_classification_v3_frames(taxa, source)
    findings = validate_classification_v3(frames, taxa=taxa)
    qa_findings = classification_v3_qa_frame(findings)
    manifest = build_classification_v3_manifest(
        frames,
        registry_version=BENCHMARK_REGISTRY_VERSION,
    )
    fatal_count = sum(finding["severity"] == "fatal" for finding in findings)
    warning_count = sum(finding["severity"] == "warning" for finding in findings)
    manifest.update(
        {
            "created_at": f"{BENCHMARK_REVIEW_DATE}T00:00:00Z",
            "classification_fingerprint": classification_v3_fingerprint(frames),
            "hierarchy_fingerprint": hierarchy_fingerprint(frames),
            "fatal_finding_count": fatal_count,
            "warning_finding_count": warning_count,
            "qa_status": "failed" if fatal_count else "passed",
            "benchmark_fixture": True,
            "authoritative_taxonomy": False,
            "gbif_authority": False,
            "fixture_notice": BENCHMARK_FIXTURE_NOTICE,
            "benchmark_selected_genus_node_ids": list(BENCHMARK_SELECTED_GENUS_NODE_IDS),
        }
    )
    taxonomy_store = PathTaxonomyStore.from_frames(
        sources=frames.sources,
        nodes=frames.nodes,
        edges=frames.edges,
        gbif_mappings=frames.gbif_mappings,
        leaf_paths=frames.leaf_paths,
        prompt_labels=frames.prompt_labels,
        qa_findings=qa_findings,
        manifest=manifest,
    )
    return SevenFamilyPathCascadeFixture(
        taxa=taxa,
        frames=frames,
        qa_findings=qa_findings,
        manifest=taxonomy_store.manifest,
        taxonomy_store=taxonomy_store,
    )


def _synthetic_source_and_taxa() -> tuple[pl.DataFrame, dict[str, object]]:
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    mappings: list[dict[str, object]] = []
    taxa: list[dict[str, object]] = []

    for family_number in range(1, FAMILY_COUNT + 1):
        family_id = f"fixture:family:{family_number:02d}"
        family_name = f"FixtureFamily{family_number:02d}idae"
        nodes.append(_node(family_id, "FAMILY", family_name))
        for subfamily_number in range(1, SUBFAMILIES_PER_FAMILY + 1):
            subfamily_id = f"fixture:subfamily:{family_number:02d}:{subfamily_number:02d}"
            subfamily_name = (
                f"FixtureFamily{family_number:02d}Subfamily{subfamily_number:02d}inae"
            )
            nodes.append(_node(subfamily_id, "SUBFAMILY", subfamily_name))
            edges.append(_asserted_edge(family_id, subfamily_id))
            for tribe_number in range(1, TRIBES_PER_SUBFAMILY + 1):
                tribe_id = (
                    f"fixture:tribe:{family_number:02d}:{subfamily_number:02d}:{tribe_number:02d}"
                )
                tribe_name = (
                    f"FixtureFamily{family_number:02d}Subfamily{subfamily_number:02d}"
                    f"Tribe{tribe_number:02d}ini"
                )
                nodes.append(_node(tribe_id, "TRIBE", tribe_name))
                edges.append(_asserted_edge(subfamily_id, tribe_id))

                genus_parent_id = tribe_id
                if tribe_number == 1:
                    subtribe_id = (
                        f"fixture:subtribe:{family_number:02d}:{subfamily_number:02d}:"
                        f"{tribe_number:02d}"
                    )
                    subtribe_name = (
                        f"FixtureFamily{family_number:02d}Subfamily{subfamily_number:02d}"
                        f"Tribe{tribe_number:02d}ina"
                    )
                    nodes.append(_node(subtribe_id, "SUBTRIBE", subtribe_name))
                    edges.append(_asserted_edge(tribe_id, subtribe_id))
                    genus_parent_id = subtribe_id

                genus_count = (
                    3
                    if (family_number, subfamily_number, tribe_number) == (1, 1, 1)
                    else GENERA_PER_BRANCH
                )
                for genus_number in range(1, genus_count + 1):
                    genus_id = (
                        f"fixture:genus:{family_number:02d}:{subfamily_number:02d}:"
                        f"{tribe_number:02d}:{genus_number:02d}"
                    )
                    genus_name = (
                        f"FixtureGenus{family_number:02d}{subfamily_number:02d}"
                        f"{tribe_number:02d}{genus_number:02d}"
                    )
                    nodes.append(_node(genus_id, "GENUS", genus_name))
                    if tribe_number == 1:
                        edges.append(_asserted_edge(genus_parent_id, genus_id))
                    else:
                        edges.append(_reviewed_subtribe_skip_edge(tribe_id, genus_id))

                    species_count = (
                        SPECIES_PER_SELECTED_GENUS
                        if genus_id in BENCHMARK_SELECTED_GENUS_NODE_IDS
                        else 1
                    )
                    for species_number in range(1, species_count + 1):
                        species_id = f"{genus_id.replace(':genus:', ':species:')}:{species_number:02d}"
                        species_name = f"{genus_name} specimen{species_number:02d}"
                        synthetic_key = (
                            f"synthetic-{family_number:02d}{subfamily_number:02d}"
                            f"{tribe_number:02d}{genus_number:02d}{species_number:02d}"
                        )
                        nodes.append(_node(species_id, "SPECIES", species_name))
                        edges.append(_asserted_edge(genus_id, species_id))
                        mappings.append(_mapping(synthetic_key, species_name, species_id))
                        taxa.append(
                            {
                                "registry_version": BENCHMARK_REGISTRY_VERSION,
                                "accepted_taxon_key": synthetic_key,
                                "species_key": synthetic_key,
                                "scientific_name": species_name,
                                "family": family_name,
                                "genus": genus_name,
                                "species": species_name,
                                "rank": "SPECIES",
                                "taxonomic_status": "ACCEPTED",
                            }
                        )

    return pl.DataFrame(taxa), {
        "classification_version": CLASSIFICATION_V3_VERSION,
        "sources": [
            {
                "source_id": BENCHMARK_SOURCE_ID,
                "authority": "Synthetic benchmark fixture; not biological or GBIF authority",
                "release": BENCHMARK_REGISTRY_VERSION,
                "citation": BENCHMARK_FIXTURE_NOTICE,
                "retrieved_at": BENCHMARK_REVIEW_DATE,
                "evidence_url": "https://example.invalid/biominer/synthetic-path-cascade",
                "evidence": BENCHMARK_FIXTURE_NOTICE,
            }
        ],
        "nodes": nodes,
        "edges": edges,
        "species_mappings": mappings,
    }


def _node(node_id: str, rank: str, scientific_name: str) -> dict[str, object]:
    return {
        "node_id": node_id,
        "rank": rank,
        "scientific_name": scientific_name,
        "source_id": BENCHMARK_SOURCE_ID,
        "evidence": BENCHMARK_FIXTURE_NOTICE,
        **_review(),
    }


def _asserted_edge(parent_node_id: str, child_node_id: str) -> dict[str, object]:
    return {
        "parent_node_id": parent_node_id,
        "child_node_id": child_node_id,
        "edge_type": ASSERTED_PARENT_EDGE,
        "skipped_ranks": [],
        "skip_reason": "",
        "source_id": BENCHMARK_SOURCE_ID,
        "evidence": BENCHMARK_FIXTURE_NOTICE,
        **_review(),
    }


def _reviewed_subtribe_skip_edge(parent_node_id: str, child_node_id: str) -> dict[str, object]:
    return {
        "parent_node_id": parent_node_id,
        "child_node_id": child_node_id,
        "edge_type": REVIEWED_RANK_SKIP_EDGE,
        "skipped_ranks": ["SUBTRIBE"],
        "skip_reason": "Synthetic branch explicitly exercises the optional SUBTRIBE skip contract.",
        "source_id": BENCHMARK_SOURCE_ID,
        "evidence": BENCHMARK_FIXTURE_NOTICE,
        **_review(),
    }


def _mapping(
    synthetic_key: str,
    accepted_scientific_name: str,
    species_node_id: str,
) -> dict[str, object]:
    return {
        "gbif_species_key": synthetic_key,
        "accepted_scientific_name": accepted_scientific_name,
        "species_node_id": species_node_id,
        "source_id": BENCHMARK_SOURCE_ID,
        "evidence": (
            "Synthetic identity mapping required by the classification-v3 test schema; "
            "not a GBIF identifier or assertion."
        ),
        **_review(),
    }


def _review() -> dict[str, object]:
    return {
        "reviewed": True,
        "review_status": "reviewed",
        "reviewed_by": "BioMiner synthetic benchmark builder",
        "reviewed_at": BENCHMARK_REVIEW_DATE,
        "enabled": True,
    }
