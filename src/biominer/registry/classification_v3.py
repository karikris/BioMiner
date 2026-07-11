from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from biominer.storage.uri import join_uri


CLASSIFICATION_V3_VERSION = "butterfly-classification-v3.0.0"
CLASSIFICATION_V3_PROMPT_VERSION = "butterfly-six-rank-prompts-v3"

CLASSIFICATION_RANKS = (
    "FAMILY",
    "SUBFAMILY",
    "TRIBE",
    "SUBTRIBE",
    "GENUS",
    "SPECIES",
)
MANDATORY_CLASSIFICATION_RANKS = (
    "FAMILY",
    "SUBFAMILY",
    "TRIBE",
    "GENUS",
    "SPECIES",
)
OPTIONAL_CLASSIFICATION_RANKS = ("SUBTRIBE",)
ALLOWED_RANK_TRANSITIONS = tuple(zip(CLASSIFICATION_RANKS[:-1], CLASSIFICATION_RANKS[1:], strict=True))

ASSERTED_PARENT_EDGE = "asserted_parent"
REVIEWED_RANK_SKIP_EDGE = "reviewed_rank_skip"
SUPPORTED_EDGE_TYPES = (ASSERTED_PARENT_EDGE, REVIEWED_RANK_SKIP_EDGE)

CLASSIFICATION_V3_SOURCES_FILE = "classification_sources.parquet"
CLASSIFICATION_V3_NODES_FILE = "classification_nodes.parquet"
CLASSIFICATION_V3_EDGES_FILE = "classification_edges.parquet"
CLASSIFICATION_V3_GBIF_MAPPINGS_FILE = "species_gbif_mappings.parquet"
CLASSIFICATION_V3_LEAF_PATHS_FILE = "classification_leaf_paths.parquet"
CLASSIFICATION_V3_PROMPT_LABELS_FILE = "classification_prompt_labels.parquet"
CLASSIFICATION_V3_QA_FINDINGS_FILE = "classification_qa_findings.parquet"
CLASSIFICATION_V3_MANIFEST_FILE = "classification_manifest.json"
DEFAULT_CLASSIFICATION_V3_SOURCE = Path("config/taxonomy/papilionoidea_classification_v3.json")

SOURCE_SCHEMA: dict[str, pl.DataType] = {
    "classification_version": pl.String,
    "source_id": pl.String,
    "authority": pl.String,
    "release": pl.String,
    "citation": pl.String,
    "retrieved_at": pl.String,
    "evidence_url": pl.String,
    "evidence": pl.String,
}

NODE_SCHEMA: dict[str, pl.DataType] = {
    "classification_version": pl.String,
    "node_id": pl.String,
    "rank": pl.String,
    "scientific_name": pl.String,
    "source_id": pl.String,
    "source_release": pl.String,
    "citation": pl.String,
    "retrieved_at": pl.String,
    "evidence": pl.String,
    "reviewed": pl.Boolean,
    "review_status": pl.String,
    "reviewed_by": pl.String,
    "reviewed_at": pl.String,
    "enabled": pl.Boolean,
    "disabled_reason": pl.String,
}

EDGE_SCHEMA: dict[str, pl.DataType] = {
    "classification_version": pl.String,
    "parent_node_id": pl.String,
    "child_node_id": pl.String,
    "parent_rank": pl.String,
    "child_rank": pl.String,
    "edge_type": pl.String,
    "skipped_ranks": pl.List(pl.String),
    "skip_reason": pl.String,
    "source_id": pl.String,
    "source_release": pl.String,
    "citation": pl.String,
    "retrieved_at": pl.String,
    "evidence": pl.String,
    "reviewed": pl.Boolean,
    "review_status": pl.String,
    "reviewed_by": pl.String,
    "reviewed_at": pl.String,
    "enabled": pl.Boolean,
    "disabled_reason": pl.String,
}

GBIF_MAPPING_SCHEMA: dict[str, pl.DataType] = {
    "classification_version": pl.String,
    "accepted_taxon_key": pl.String,
    "gbif_species_key": pl.String,
    "accepted_scientific_name": pl.String,
    "species_node_id": pl.String,
    "taxonomic_status": pl.String,
    "source_id": pl.String,
    "source_release": pl.String,
    "citation": pl.String,
    "retrieved_at": pl.String,
    "evidence": pl.String,
    "reviewed": pl.Boolean,
    "review_status": pl.String,
    "reviewed_by": pl.String,
    "reviewed_at": pl.String,
    "enabled": pl.Boolean,
    "disabled_reason": pl.String,
}

LEAF_PATH_SCHEMA: dict[str, pl.DataType] = {
    "classification_version": pl.String,
    "accepted_taxon_key": pl.String,
    "gbif_species_key": pl.String,
    "family_node_id": pl.String,
    "family": pl.String,
    "subfamily_node_id": pl.String,
    "subfamily": pl.String,
    "tribe_node_id": pl.String,
    "tribe": pl.String,
    "subtribe_node_id": pl.String,
    "subtribe": pl.String,
    "genus_node_id": pl.String,
    "genus": pl.String,
    "species_node_id": pl.String,
    "species": pl.String,
    "rank_path": pl.List(pl.String),
    "rank_path_node_ids": pl.List(pl.String),
    "skipped_ranks": pl.List(pl.String),
    "path_completeness": pl.String,
    "hierarchy_hash": pl.String,
    "source_release": pl.String,
    "enabled": pl.Boolean,
    "disabled_reason": pl.String,
}

PROMPT_LABEL_SCHEMA: dict[str, pl.DataType] = {
    "classification_version": pl.String,
    "prompt_version": pl.String,
    "node_id": pl.String,
    "rank": pl.String,
    "scientific_name": pl.String,
    "label": pl.String,
    "prompt_template": pl.String,
    "sort_order": pl.Int64,
    "enabled": pl.Boolean,
}

QA_FINDING_SCHEMA: dict[str, pl.DataType] = {
    "severity": pl.String,
    "code": pl.String,
    "table": pl.String,
    "subject": pl.String,
    "message": pl.String,
    "details_json": pl.String,
}

RANK_PROMPT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "FAMILY": ("a photo of a butterfly in family {name}",),
    "SUBFAMILY": ("a photo of a butterfly in subfamily {name}",),
    "TRIBE": ("a photo of a butterfly in tribe {name}",),
    "SUBTRIBE": ("a photo of a butterfly in subtribe {name}",),
    "GENUS": ("a photo of a butterfly in genus {name}",),
    "SPECIES": ("a photo of the butterfly species {name}",),
}


@dataclass(frozen=True)
class ClassificationV3Frames:
    sources: pl.DataFrame
    nodes: pl.DataFrame
    edges: pl.DataFrame
    gbif_mappings: pl.DataFrame
    leaf_paths: pl.DataFrame
    prompt_labels: pl.DataFrame


def classification_v3_artifact_paths(root: str | Path) -> dict[str, Path]:
    base = Path(root)
    return {
        "sources": base / CLASSIFICATION_V3_SOURCES_FILE,
        "nodes": base / CLASSIFICATION_V3_NODES_FILE,
        "edges": base / CLASSIFICATION_V3_EDGES_FILE,
        "gbif_mappings": base / CLASSIFICATION_V3_GBIF_MAPPINGS_FILE,
        "leaf_paths": base / CLASSIFICATION_V3_LEAF_PATHS_FILE,
        "prompt_labels": base / CLASSIFICATION_V3_PROMPT_LABELS_FILE,
        "qa_findings": base / CLASSIFICATION_V3_QA_FINDINGS_FILE,
        "manifest": base / CLASSIFICATION_V3_MANIFEST_FILE,
    }


def classification_v3_artifact_uris(root: str) -> dict[str, str]:
    base = str(root).rstrip("/")
    return {
        "sources": join_uri(base, CLASSIFICATION_V3_SOURCES_FILE),
        "nodes": join_uri(base, CLASSIFICATION_V3_NODES_FILE),
        "edges": join_uri(base, CLASSIFICATION_V3_EDGES_FILE),
        "gbif_mappings": join_uri(base, CLASSIFICATION_V3_GBIF_MAPPINGS_FILE),
        "leaf_paths": join_uri(base, CLASSIFICATION_V3_LEAF_PATHS_FILE),
        "prompt_labels": join_uri(base, CLASSIFICATION_V3_PROMPT_LABELS_FILE),
        "qa_findings": join_uri(base, CLASSIFICATION_V3_QA_FINDINGS_FILE),
        "manifest": join_uri(base, CLASSIFICATION_V3_MANIFEST_FILE),
    }


__all__ = [
    "ALLOWED_RANK_TRANSITIONS",
    "ASSERTED_PARENT_EDGE",
    "CLASSIFICATION_RANKS",
    "CLASSIFICATION_V3_PROMPT_VERSION",
    "CLASSIFICATION_V3_VERSION",
    "ClassificationV3Frames",
    "DEFAULT_CLASSIFICATION_V3_SOURCE",
    "EDGE_SCHEMA",
    "GBIF_MAPPING_SCHEMA",
    "LEAF_PATH_SCHEMA",
    "MANDATORY_CLASSIFICATION_RANKS",
    "NODE_SCHEMA",
    "OPTIONAL_CLASSIFICATION_RANKS",
    "PROMPT_LABEL_SCHEMA",
    "QA_FINDING_SCHEMA",
    "RANK_PROMPT_TEMPLATES",
    "REVIEWED_RANK_SKIP_EDGE",
    "SOURCE_SCHEMA",
    "SUPPORTED_EDGE_TYPES",
    "classification_v3_artifact_paths",
    "classification_v3_artifact_uris",
]
