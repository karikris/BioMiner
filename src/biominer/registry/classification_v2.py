from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from typing import Any, Sequence

import polars as pl


CLASSIFICATION_V2_VERSION = "butterfly-classification-v2.0.0"
CLASSIFICATION_V2_PROMPT_VERSION = "butterfly-five-rank-prompts-v2"
CLASSIFICATION_RANKS = ("FAMILY", "SUBFAMILY", "TRIBE", "GENUS", "SPECIES")
ALLOWED_RANK_TRANSITIONS = tuple(zip(CLASSIFICATION_RANKS[:-1], CLASSIFICATION_RANKS[1:], strict=True))

CLASSIFICATION_V2_SOURCES_FILE = "classification_sources.parquet"
CLASSIFICATION_V2_NODES_FILE = "classification_nodes.parquet"
CLASSIFICATION_V2_EDGES_FILE = "classification_edges.parquet"
CLASSIFICATION_V2_GBIF_MAPPINGS_FILE = "species_gbif_mappings.parquet"
CLASSIFICATION_V2_LEAF_PATHS_FILE = "classification_leaf_paths.parquet"
CLASSIFICATION_V2_PROMPT_LABELS_FILE = "classification_prompt_labels.parquet"
CLASSIFICATION_V2_QA_FINDINGS_FILE = "classification_qa_findings.parquet"
CLASSIFICATION_V2_MANIFEST_FILE = "classification_manifest.json"

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
    "enabled": pl.Boolean,
    "disabled_reason": pl.String,
}

EDGE_SCHEMA: dict[str, pl.DataType] = {
    "classification_version": pl.String,
    "parent_node_id": pl.String,
    "child_node_id": pl.String,
    "parent_rank": pl.String,
    "child_rank": pl.String,
    "source_id": pl.String,
    "source_release": pl.String,
    "citation": pl.String,
    "retrieved_at": pl.String,
    "evidence": pl.String,
    "reviewed": pl.Boolean,
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
    "genus_node_id": pl.String,
    "genus": pl.String,
    "species_node_id": pl.String,
    "species": pl.String,
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
    "GENUS": ("a photo of a butterfly in genus {name}",),
    "SPECIES": ("a photo of the butterfly species {name}",),
}


@dataclass(frozen=True)
class ClassificationV2Frames:
    sources: pl.DataFrame
    nodes: pl.DataFrame
    edges: pl.DataFrame
    gbif_mappings: pl.DataFrame
    leaf_paths: pl.DataFrame
    prompt_labels: pl.DataFrame


def build_classification_v2_frames(
    taxa: pl.DataFrame,
    source: dict[str, Any],
) -> ClassificationV2Frames:
    version = _text(source.get("classification_version")) or CLASSIFICATION_V2_VERSION
    sources = _source_frame(source.get("sources") or (), version=version)
    source_by_id = {str(row["source_id"]): row for row in sources.iter_rows(named=True)}
    nodes = _node_frame(source.get("nodes") or (), version=version, source_by_id=source_by_id)
    node_by_id = {str(row["node_id"]): row for row in nodes.iter_rows(named=True)}
    edges = _edge_frame(source.get("edges") or (), version=version, source_by_id=source_by_id, node_by_id=node_by_id)
    mappings = _mapping_frame(
        source.get("species_mappings") or (),
        version=version,
        taxa=taxa,
        source_by_id=source_by_id,
        node_by_id=node_by_id,
    )
    leaf_paths = _leaf_path_frame(version=version, nodes=nodes, edges=edges, mappings=mappings)
    prompt_labels = _prompt_label_frame(version=version, nodes=nodes, leaf_paths=leaf_paths)
    return ClassificationV2Frames(
        sources=sources,
        nodes=nodes,
        edges=edges,
        gbif_mappings=mappings,
        leaf_paths=leaf_paths,
        prompt_labels=prompt_labels,
    )


def build_classification_v2_manifest(frames: ClassificationV2Frames, *, registry_version: str = "") -> dict[str, Any]:
    enabled_paths = frames.leaf_paths.filter(pl.col("enabled"))
    rank_counts = {
        rank: frames.nodes.filter((pl.col("rank") == rank) & pl.col("enabled")).height
        for rank in CLASSIFICATION_RANKS
    }
    return {
        "classification_version": _first_value(frames.nodes, "classification_version") or CLASSIFICATION_V2_VERSION,
        "prompt_version": CLASSIFICATION_V2_PROMPT_VERSION,
        "registry_version": registry_version,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "rank_order": list(CLASSIFICATION_RANKS),
        "source_count": frames.sources.height,
        "node_count": frames.nodes.height,
        "enabled_node_count": frames.nodes.filter(pl.col("enabled")).height,
        "enabled_node_counts_by_rank": rank_counts,
        "edge_count": frames.edges.height,
        "enabled_edge_count": frames.edges.filter(pl.col("enabled")).height,
        "gbif_mapping_count": frames.gbif_mappings.height,
        "enabled_gbif_mapping_count": frames.gbif_mappings.filter(pl.col("enabled")).height,
        "leaf_path_count": frames.leaf_paths.height,
        "enabled_leaf_path_count": enabled_paths.height,
        "prompt_label_count": frames.prompt_labels.filter(pl.col("enabled")).height,
    }


def classification_v2_fingerprint(frames: ClassificationV2Frames) -> str:
    payload = {
        "sources": frames.sources.to_dicts(),
        "nodes": frames.nodes.to_dicts(),
        "edges": frames.edges.to_dicts(),
        "mappings": frames.gbif_mappings.to_dicts(),
        "paths": frames.leaf_paths.to_dicts(),
        "prompts": frames.prompt_labels.to_dicts(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _source_frame(rows: Sequence[dict[str, Any]], *, version: str) -> pl.DataFrame:
    normalized = [
        {
            "classification_version": version,
            "source_id": _text(row.get("source_id")),
            "authority": _text(row.get("authority")),
            "release": _text(row.get("release")),
            "citation": _text(row.get("citation")),
            "retrieved_at": _text(row.get("retrieved_at")),
            "evidence_url": _text(row.get("evidence_url")),
            "evidence": _text(row.get("evidence")),
        }
        for row in rows
    ]
    return _frame(normalized, SOURCE_SCHEMA).sort("source_id")


def _node_frame(
    rows: Sequence[dict[str, Any]],
    *,
    version: str,
    source_by_id: dict[str, dict[str, Any]],
) -> pl.DataFrame:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        source_id = _text(row.get("source_id"))
        source = source_by_id.get(source_id, {})
        reviewed = _bool(row.get("reviewed"))
        reasons = []
        if not reviewed:
            reasons.append("unreviewed_node")
        if not source:
            reasons.append("unknown_source")
        normalized.append(
            {
                "classification_version": version,
                "node_id": _text(row.get("node_id")),
                "rank": _text(row.get("rank")).upper(),
                "scientific_name": _text(row.get("scientific_name")),
                "source_id": source_id,
                "source_release": _text(source.get("release")),
                "citation": _text(source.get("citation")),
                "retrieved_at": _text(source.get("retrieved_at")),
                "evidence": _text(row.get("evidence"), source.get("evidence")),
                "reviewed": reviewed,
                "enabled": _bool(row.get("enabled"), default=True) and not reasons,
                "disabled_reason": ",".join(reasons),
            }
        )
    return _frame(normalized, NODE_SCHEMA).sort(["rank", "scientific_name", "node_id"])


def _edge_frame(
    rows: Sequence[dict[str, Any]],
    *,
    version: str,
    source_by_id: dict[str, dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
) -> pl.DataFrame:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        parent_id = _text(row.get("parent_node_id"))
        child_id = _text(row.get("child_node_id"))
        parent = node_by_id.get(parent_id, {})
        child = node_by_id.get(child_id, {})
        source_id = _text(row.get("source_id"))
        source = source_by_id.get(source_id, {})
        reviewed = _bool(row.get("reviewed"))
        transition = (_text(parent.get("rank")), _text(child.get("rank")))
        reasons = []
        if not parent or not child:
            reasons.append("unknown_node")
        if transition not in ALLOWED_RANK_TRANSITIONS:
            reasons.append("invalid_rank_transition")
        if not reviewed:
            reasons.append("unreviewed_edge")
        if not source:
            reasons.append("unknown_source")
        if parent and not _bool(parent.get("enabled")):
            reasons.append("disabled_parent")
        if child and not _bool(child.get("enabled")):
            reasons.append("disabled_child")
        normalized.append(
            {
                "classification_version": version,
                "parent_node_id": parent_id,
                "child_node_id": child_id,
                "parent_rank": _text(parent.get("rank")),
                "child_rank": _text(child.get("rank")),
                "source_id": source_id,
                "source_release": _text(source.get("release")),
                "citation": _text(source.get("citation")),
                "retrieved_at": _text(source.get("retrieved_at")),
                "evidence": _text(row.get("evidence"), source.get("evidence")),
                "reviewed": reviewed,
                "enabled": _bool(row.get("enabled"), default=True) and not reasons,
                "disabled_reason": ",".join(reasons),
            }
        )
    return _frame(normalized, EDGE_SCHEMA).sort(["parent_rank", "parent_node_id", "child_node_id"])


def _mapping_frame(
    rows: Sequence[dict[str, Any]],
    *,
    version: str,
    taxa: pl.DataFrame,
    source_by_id: dict[str, dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
) -> pl.DataFrame:
    taxa_rows = list(taxa.iter_rows(named=True))
    normalized: list[dict[str, Any]] = []
    for row in rows:
        gbif_species_key = _bare_gbif_key(row.get("gbif_species_key"))
        expected_name = _text(row.get("accepted_scientific_name"))
        matches = [
            taxon
            for taxon in taxa_rows
            if _text(taxon.get("rank")).upper() == "SPECIES"
            and _bare_gbif_key(taxon.get("accepted_taxon_key"), taxon.get("species_key")) == gbif_species_key
        ]
        taxon = matches[0] if len(matches) == 1 else {}
        actual_name = _text(taxon.get("species"), taxon.get("scientific_name"))
        status = _text(taxon.get("taxonomic_status"), taxon.get("status")).upper()
        species_node_id = _text(row.get("species_node_id"))
        species_node = node_by_id.get(species_node_id, {})
        source_id = _text(row.get("source_id"))
        source = source_by_id.get(source_id, {})
        reviewed = _bool(row.get("reviewed"))
        reasons = []
        if len(matches) != 1:
            reasons.append("gbif_species_key_not_unique")
        if status != "ACCEPTED":
            reasons.append("gbif_species_not_accepted")
        if expected_name != actual_name:
            reasons.append("gbif_scientific_name_mismatch")
        if _text(species_node.get("rank")) != "SPECIES":
            reasons.append("invalid_species_node")
        if not reviewed:
            reasons.append("unreviewed_mapping")
        if not source:
            reasons.append("unknown_source")
        normalized.append(
            {
                "classification_version": version,
                "accepted_taxon_key": _text(taxon.get("accepted_taxon_key")),
                "gbif_species_key": gbif_species_key,
                "accepted_scientific_name": actual_name or expected_name,
                "species_node_id": species_node_id,
                "taxonomic_status": status,
                "source_id": source_id,
                "source_release": _text(source.get("release")),
                "citation": _text(source.get("citation")),
                "retrieved_at": _text(source.get("retrieved_at")),
                "evidence": _text(row.get("evidence"), source.get("evidence")),
                "reviewed": reviewed,
                "enabled": _bool(row.get("enabled"), default=True) and not reasons,
                "disabled_reason": ",".join(reasons),
            }
        )
    return _frame(normalized, GBIF_MAPPING_SCHEMA).sort(["accepted_scientific_name", "gbif_species_key"])


def _leaf_path_frame(
    *,
    version: str,
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    mappings: pl.DataFrame,
) -> pl.DataFrame:
    node_by_id = {str(row["node_id"]): row for row in nodes.iter_rows(named=True)}
    parent_by_child = {
        str(row["child_node_id"]): str(row["parent_node_id"])
        for row in edges.filter(pl.col("enabled")).iter_rows(named=True)
    }
    rows: list[dict[str, Any]] = []
    for mapping in mappings.iter_rows(named=True):
        current = str(mapping["species_node_id"])
        path: list[dict[str, Any]] = []
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            node = node_by_id.get(current)
            if node is None:
                break
            path.append(node)
            current = parent_by_child.get(current, "")
        path.reverse()
        path_by_rank = {str(node["rank"]): node for node in path}
        complete = tuple(path_by_rank) == CLASSIFICATION_RANKS and bool(mapping["enabled"])
        reasons = [] if complete else [str(mapping.get("disabled_reason") or "incomplete_enabled_path")]
        row: dict[str, Any] = {
            "classification_version": version,
            "accepted_taxon_key": mapping["accepted_taxon_key"],
            "gbif_species_key": mapping["gbif_species_key"],
            "source_release": mapping["source_release"],
            "enabled": complete,
            "disabled_reason": ",".join(reason for reason in reasons if reason),
        }
        for rank in CLASSIFICATION_RANKS:
            prefix = rank.casefold()
            node = path_by_rank.get(rank, {})
            row[f"{prefix}_node_id"] = _text(node.get("node_id"))
            row[prefix] = _text(node.get("scientific_name"))
        rows.append(row)
    return _frame(rows, LEAF_PATH_SCHEMA).sort(["family", "subfamily", "tribe", "genus", "species"])


def _prompt_label_frame(*, version: str, nodes: pl.DataFrame, leaf_paths: pl.DataFrame) -> pl.DataFrame:
    usable_node_ids: set[str] = set()
    for row in leaf_paths.filter(pl.col("enabled")).iter_rows(named=True):
        usable_node_ids.update(str(row[f"{rank.casefold()}_node_id"]) for rank in CLASSIFICATION_RANKS)
    rows = []
    for node in nodes.filter(pl.col("enabled")).iter_rows(named=True):
        node_id = str(node["node_id"])
        if node_id not in usable_node_ids:
            continue
        rank = str(node["rank"])
        name = str(node["scientific_name"])
        for sort_order, template in enumerate(RANK_PROMPT_TEMPLATES[rank], start=1):
            rows.append(
                {
                    "classification_version": version,
                    "prompt_version": CLASSIFICATION_V2_PROMPT_VERSION,
                    "node_id": node_id,
                    "rank": rank,
                    "scientific_name": name,
                    "label": template.format(name=name),
                    "prompt_template": template,
                    "sort_order": sort_order,
                    "enabled": True,
                }
            )
    return _frame(rows, PROMPT_LABEL_SCHEMA).sort(["rank", "scientific_name", "sort_order"])


def _frame(rows: Sequence[dict[str, Any]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    frame = pl.DataFrame(rows)
    expressions = [
        pl.col(name).cast(dtype).alias(name) if name in frame.columns else pl.lit(_default(dtype), dtype=dtype).alias(name)
        for name, dtype in schema.items()
    ]
    return frame.select(expressions)


def _default(dtype: pl.DataType) -> object:
    if dtype == pl.Boolean:
        return False
    if dtype in {pl.Int32, pl.Int64, pl.UInt32, pl.UInt64}:
        return 0
    return ""


def _text(*values: object) -> str:
    for value in values:
        text = " ".join(str(value or "").strip().split())
        if text:
            return text
    return ""


def _bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _bare_gbif_key(*values: object) -> str:
    text = _text(*values)
    return text.split(":", 1)[1] if text.casefold().startswith("gbif:") else text


def _first_value(frame: pl.DataFrame, column: str) -> str:
    if frame.is_empty() or column not in frame.columns:
        return ""
    return _text(frame[column][0])


__all__ = [
    "ALLOWED_RANK_TRANSITIONS",
    "CLASSIFICATION_RANKS",
    "CLASSIFICATION_V2_VERSION",
    "ClassificationV2Frames",
    "build_classification_v2_frames",
    "build_classification_v2_manifest",
    "classification_v2_fingerprint",
]
