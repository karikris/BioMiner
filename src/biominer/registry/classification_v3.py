from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from biominer.storage.parquet import write_parquet
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


def load_classification_v3_source(path: str | Path = DEFAULT_CLASSIFICATION_V3_SOURCE) -> dict[str, Any]:
    source_path = Path(path)
    if not source_path.exists() and source_path == DEFAULT_CLASSIFICATION_V3_SOURCE:
        source_path = Path(__file__).resolve().parents[3] / DEFAULT_CLASSIFICATION_V3_SOURCE
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"classification source must be a JSON object: {source_path}")
    return payload


def write_classification_v3_artifacts(
    registry_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    source_path: str | Path = DEFAULT_CLASSIFICATION_V3_SOURCE,
) -> dict[str, Any]:
    registry = Path(registry_dir)
    output = Path(output_dir) if output_dir is not None else registry
    taxa_path = registry / "taxa.parquet"
    if not taxa_path.exists():
        raise FileNotFoundError(f"missing required registry artifact: {taxa_path}")
    existing_manifest = _read_json_optional(output / CLASSIFICATION_V3_MANIFEST_FILE)
    existing_version = _text(existing_manifest.get("classification_version"))
    if existing_version and existing_version != CLASSIFICATION_V3_VERSION:
        raise ValueError(
            "classification artifact version conflict: "
            f"found {existing_version}, expected {CLASSIFICATION_V3_VERSION}; rebuild in a new versioned registry root"
        )
    registry_manifest = _read_json_optional(registry / "manifest.json")
    artifact_frames, manifest = compile_classification_v3_artifacts(
        pl.read_parquet(taxa_path),
        source_path=source_path,
        registry_version=_text(registry_manifest.get("registry_version")),
    )
    output.mkdir(parents=True, exist_ok=True)
    paths = classification_v3_artifact_paths(output)
    write_parquet(artifact_frames["qa_findings"], paths["qa_findings"])
    if manifest["fatal_finding_count"]:
        fatal_codes = artifact_frames["qa_findings"].filter(pl.col("severity") == "fatal")["code"].to_list()
        raise ValueError("classification-v3 fatal QA: " + ", ".join(str(code) for code in fatal_codes))
    for key in ("sources", "nodes", "edges", "gbif_mappings", "leaf_paths", "prompt_labels"):
        write_parquet(artifact_frames[key], paths[key])
    manifest["artifacts"] = {
        key: {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for key, path in paths.items()
        if key != "manifest" and path.exists()
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {**manifest, "outputs": {key: str(path) for key, path in paths.items()}}


def compile_classification_v3_artifacts(
    taxa: pl.DataFrame,
    *,
    source_path: str | Path = DEFAULT_CLASSIFICATION_V3_SOURCE,
    registry_version: str = "",
) -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    frames = build_classification_v3_frames(taxa, load_classification_v3_source(source_path))
    findings = validate_classification_v3(frames, taxa=taxa)
    manifest = build_classification_v3_manifest(frames, registry_version=registry_version)
    manifest["classification_fingerprint"] = classification_v3_fingerprint(frames)
    manifest["hierarchy_fingerprint"] = hierarchy_fingerprint(frames)
    manifest["fatal_finding_count"] = sum(1 for finding in findings if finding["severity"] == "fatal")
    manifest["warning_finding_count"] = sum(1 for finding in findings if finding["severity"] == "warning")
    manifest["qa_status"] = "failed" if manifest["fatal_finding_count"] else "passed"
    accepted_species = _accepted_species_rows(taxa)
    enabled_keys = set(frames.leaf_paths.filter(pl.col("enabled"))["accepted_taxon_key"].to_list())
    manifest["accepted_species_count"] = len(accepted_species)
    manifest["unmapped_accepted_species_count"] = sum(
        1 for row in accepted_species if _text(row.get("accepted_taxon_key")) not in enabled_keys
    )
    manifest["coverage_by_family"] = _coverage_by_family(accepted_species, enabled_keys)
    return (
        {
            "sources": frames.sources,
            "nodes": frames.nodes,
            "edges": frames.edges,
            "gbif_mappings": frames.gbif_mappings,
            "leaf_paths": frames.leaf_paths,
            "prompt_labels": frames.prompt_labels,
            "qa_findings": classification_v3_qa_frame(findings),
        },
        manifest,
    )


def build_classification_v3_frames(taxa: pl.DataFrame, source: dict[str, Any]) -> ClassificationV3Frames:
    version = _text(source.get("classification_version")) or CLASSIFICATION_V3_VERSION
    sources = _source_frame(source.get("sources") or (), version=version)
    source_by_id = {str(row["source_id"]): row for row in sources.iter_rows(named=True)}
    nodes = _node_frame(source.get("nodes") or (), version=version, source_by_id=source_by_id)
    node_by_id = {str(row["node_id"]): row for row in nodes.iter_rows(named=True)}
    edges = _edge_frame(
        source.get("edges") or (),
        version=version,
        source_by_id=source_by_id,
        node_by_id=node_by_id,
    )
    mappings = _mapping_frame(
        source.get("species_mappings") or (),
        version=version,
        taxa=taxa,
        source_by_id=source_by_id,
        node_by_id=node_by_id,
    )
    leaf_paths = _leaf_path_frame(version=version, nodes=nodes, edges=edges, mappings=mappings)
    prompt_labels = _prompt_label_frame(version=version, nodes=nodes, leaf_paths=leaf_paths)
    return ClassificationV3Frames(
        sources=sources,
        nodes=nodes,
        edges=edges,
        gbif_mappings=mappings,
        leaf_paths=leaf_paths,
        prompt_labels=prompt_labels,
    )


def build_classification_v3_manifest(frames: ClassificationV3Frames, *, registry_version: str = "") -> dict[str, Any]:
    enabled_paths = frames.leaf_paths.filter(pl.col("enabled"))
    rank_counts = {
        rank: frames.nodes.filter((pl.col("rank") == rank) & pl.col("enabled")).height
        for rank in CLASSIFICATION_RANKS
    }
    skip_count = frames.edges.filter(
        pl.col("enabled") & (pl.col("edge_type") == REVIEWED_RANK_SKIP_EDGE)
    ).height
    return {
        "classification_version": _first_value(frames.nodes, "classification_version") or CLASSIFICATION_V3_VERSION,
        "prompt_version": CLASSIFICATION_V3_PROMPT_VERSION,
        "registry_version": registry_version,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "rank_order": list(CLASSIFICATION_RANKS),
        "mandatory_ranks": list(MANDATORY_CLASSIFICATION_RANKS),
        "optional_ranks": list(OPTIONAL_CLASSIFICATION_RANKS),
        "source_count": frames.sources.height,
        "node_count": frames.nodes.height,
        "enabled_node_count": frames.nodes.filter(pl.col("enabled")).height,
        "enabled_node_counts_by_rank": rank_counts,
        "edge_count": frames.edges.height,
        "enabled_edge_count": frames.edges.filter(pl.col("enabled")).height,
        "reviewed_rank_skip_count": skip_count,
        "gbif_mapping_count": frames.gbif_mappings.height,
        "enabled_gbif_mapping_count": frames.gbif_mappings.filter(pl.col("enabled")).height,
        "leaf_path_count": frames.leaf_paths.height,
        "enabled_leaf_path_count": enabled_paths.height,
        "prompt_label_count": frames.prompt_labels.filter(pl.col("enabled")).height,
    }


def classification_v3_fingerprint(frames: ClassificationV3Frames) -> str:
    payload = {
        "classification_version": CLASSIFICATION_V3_VERSION,
        "rank_order": list(CLASSIFICATION_RANKS),
        "sources": _canonical_rows(frames.sources, ("source_id",)),
        "nodes": _canonical_rows(frames.nodes, ("rank", "scientific_name", "node_id")),
        "edges": _canonical_rows(frames.edges, ("parent_rank", "parent_node_id", "child_node_id")),
        "mappings": _canonical_rows(frames.gbif_mappings, ("accepted_scientific_name", "gbif_species_key")),
        "paths": _canonical_rows(frames.leaf_paths, ("accepted_taxon_key", "hierarchy_hash")),
        "prompts": _canonical_rows(frames.prompt_labels, ("rank", "scientific_name", "sort_order", "label")),
    }
    return _sha256_json(payload)


def hierarchy_fingerprint(frames: ClassificationV3Frames) -> str:
    payload = {
        "classification_version": CLASSIFICATION_V3_VERSION,
        "rank_order": list(CLASSIFICATION_RANKS),
        "nodes": _canonical_rows(frames.nodes, ("rank", "scientific_name", "node_id")),
        "edges": _canonical_rows(frames.edges, ("parent_rank", "parent_node_id", "child_node_id")),
        "mappings": _canonical_rows(frames.gbif_mappings, ("accepted_scientific_name", "gbif_species_key")),
        "paths": _canonical_rows(frames.leaf_paths, ("accepted_taxon_key", "hierarchy_hash")),
    }
    return _sha256_json(payload)


def classification_v3_qa_frame(findings: Sequence[dict[str, Any]]) -> pl.DataFrame:
    rows = [
        {
            "severity": _text(finding.get("severity")),
            "code": _text(finding.get("code")),
            "table": _text(finding.get("table")),
            "subject": _text(finding.get("subject")),
            "message": _text(finding.get("message")),
            "details_json": json.dumps(finding.get("details") or {}, sort_keys=True, default=str),
        }
        for finding in findings
    ]
    return _typed_frame(rows, QA_FINDING_SCHEMA).sort(["severity", "code", "subject"])


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
    return _typed_frame(normalized, SOURCE_SCHEMA).sort("source_id")


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
        review = _review_fields(row)
        reasons: list[str] = []
        if not review["reviewed"]:
            reasons.append("unreviewed_node")
        if not source:
            reasons.append("unknown_source")
        if _text(row.get("rank")).upper() not in CLASSIFICATION_RANKS:
            reasons.append("unknown_rank")
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
                **review,
                "enabled": _bool(row.get("enabled"), default=True) and not reasons,
                "disabled_reason": ",".join(reasons),
            }
        )
    return _sort_nodes(_typed_frame(normalized, NODE_SCHEMA))


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
        review = _review_fields(row)
        edge_type = _text(row.get("edge_type")) or ASSERTED_PARENT_EDGE
        skipped_ranks = [rank.upper() for rank in _string_list(row.get("skipped_ranks"))]
        skip_reason = _text(row.get("skip_reason"))
        parent_rank = _text(parent.get("rank"))
        child_rank = _text(child.get("rank"))
        reasons: list[str] = []
        if not parent or not child:
            reasons.append("unknown_node")
        if edge_type not in SUPPORTED_EDGE_TYPES:
            reasons.append("invalid_edge_type")
        if not _edge_transition_is_valid(
            parent_rank=parent_rank,
            child_rank=child_rank,
            edge_type=edge_type,
            skipped_ranks=skipped_ranks,
        ):
            reasons.append("invalid_rank_transition")
        if edge_type == REVIEWED_RANK_SKIP_EDGE and not skip_reason:
            reasons.append("missing_skip_reason")
        if not review["reviewed"]:
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
                "parent_rank": parent_rank,
                "child_rank": child_rank,
                "edge_type": edge_type,
                "skipped_ranks": skipped_ranks,
                "skip_reason": skip_reason,
                "source_id": source_id,
                "source_release": _text(source.get("release")),
                "citation": _text(source.get("citation")),
                "retrieved_at": _text(source.get("retrieved_at")),
                "evidence": _text(row.get("evidence"), source.get("evidence")),
                **review,
                "enabled": _bool(row.get("enabled"), default=True) and not reasons,
                "disabled_reason": ",".join(reasons),
            }
        )
    return _typed_frame(normalized, EDGE_SCHEMA).sort(
        ["parent_rank", "parent_node_id", "child_rank", "child_node_id", "edge_type"]
    )


def _mapping_frame(
    rows: Sequence[dict[str, Any]],
    *,
    version: str,
    taxa: pl.DataFrame,
    source_by_id: dict[str, dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
) -> pl.DataFrame:
    taxa_by_key: dict[str, list[dict[str, Any]]] = {}
    for taxon in taxa.iter_rows(named=True):
        if _text(taxon.get("rank")).upper() != "SPECIES":
            continue
        key = _bare_gbif_key(taxon.get("accepted_taxon_key"), taxon.get("species_key"))
        taxa_by_key.setdefault(key, []).append(dict(taxon))
    normalized: list[dict[str, Any]] = []
    for row in rows:
        gbif_species_key = _bare_gbif_key(row.get("gbif_species_key"))
        expected_name = _text(row.get("accepted_scientific_name"))
        matches = taxa_by_key.get(gbif_species_key, [])
        taxon = matches[0] if len(matches) == 1 else {}
        actual_name = _text(taxon.get("species"), taxon.get("scientific_name"))
        status = _text(taxon.get("taxonomic_status"), taxon.get("status")).upper()
        species_node_id = _text(row.get("species_node_id"))
        species_node = node_by_id.get(species_node_id, {})
        source_id = _text(row.get("source_id"))
        source = source_by_id.get(source_id, {})
        review = _review_fields(row)
        reasons: list[str] = []
        if len(matches) != 1:
            reasons.append("gbif_species_key_not_unique")
        if status != "ACCEPTED":
            reasons.append("gbif_species_not_accepted")
        if expected_name != actual_name:
            reasons.append("gbif_scientific_name_mismatch")
        if _text(species_node.get("rank")) != "SPECIES":
            reasons.append("invalid_species_node")
        if species_node and not _bool(species_node.get("enabled")):
            reasons.append("disabled_species_node")
        if not review["reviewed"]:
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
                **review,
                "enabled": _bool(row.get("enabled"), default=True) and not reasons,
                "disabled_reason": ",".join(reasons),
            }
        )
    return _typed_frame(normalized, GBIF_MAPPING_SCHEMA).sort(["accepted_scientific_name", "gbif_species_key"])


def _leaf_path_frame(
    *,
    version: str,
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    mappings: pl.DataFrame,
) -> pl.DataFrame:
    node_by_id = {str(row["node_id"]): dict(row) for row in nodes.iter_rows(named=True)}
    parents_by_child: dict[str, list[dict[str, Any]]] = {}
    for edge in edges.filter(pl.col("enabled")).iter_rows(named=True):
        edge_row = dict(edge)
        parents_by_child.setdefault(str(edge_row["child_node_id"]), []).append(edge_row)
    for parent_edges in parents_by_child.values():
        parent_edges.sort(key=lambda row: (str(row["parent_node_id"]), str(row["edge_type"])))
    rows: list[dict[str, Any]] = []
    for mapping in mappings.iter_rows(named=True):
        species_node_id = str(mapping["species_node_id"])
        chains = _parent_chains(species_node_id, node_by_id=node_by_id, parents_by_child=parents_by_child)
        if not chains:
            chains = [([species_node_id] if species_node_id in node_by_id else [], [])]
        for node_ids, chain_edges in chains:
            rows.append(
                _leaf_path_row(
                    version=version,
                    mapping=dict(mapping),
                    node_ids=node_ids,
                    chain_edges=chain_edges,
                    node_by_id=node_by_id,
                )
            )
    return _typed_frame(rows, LEAF_PATH_SCHEMA).sort(
        ["family", "subfamily", "tribe", "subtribe", "genus", "species", "hierarchy_hash"]
    )


def _parent_chains(
    species_node_id: str,
    *,
    node_by_id: dict[str, dict[str, Any]],
    parents_by_child: dict[str, list[dict[str, Any]]],
) -> list[tuple[list[str], list[dict[str, Any]]]]:
    chains: list[tuple[list[str], list[dict[str, Any]]]] = []

    def walk(current: str, reverse_nodes: list[str], reverse_edges: list[dict[str, Any]], active: set[str]) -> None:
        if current in active or current not in node_by_id:
            return
        next_nodes = [*reverse_nodes, current]
        if str(node_by_id[current].get("rank")) == "FAMILY":
            chains.append((list(reversed(next_nodes)), list(reversed(reverse_edges))))
            return
        for edge in parents_by_child.get(current, []):
            walk(str(edge["parent_node_id"]), next_nodes, [*reverse_edges, edge], {*active, current})

    walk(species_node_id, [], [], set())
    return sorted(chains, key=lambda item: (tuple(item[0]), _canonical_json(item[1])))


def _leaf_path_row(
    *,
    version: str,
    mapping: dict[str, Any],
    node_ids: list[str],
    chain_edges: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    path_nodes = [node_by_id[node_id] for node_id in node_ids if node_id in node_by_id]
    rank_path = [str(node.get("rank") or "") for node in path_nodes]
    rank_path_node_ids = [str(node.get("node_id") or "") for node in path_nodes]
    path_by_rank = {str(node.get("rank") or ""): node for node in path_nodes}
    skipped = {rank for edge in chain_edges for rank in _string_list(edge.get("skipped_ranks"))}
    skipped_ranks = [rank for rank in CLASSIFICATION_RANKS if rank in skipped]
    mandatory_present = all(rank in path_by_rank for rank in MANDATORY_CLASSIFICATION_RANKS)
    actual_subtribe = "SUBTRIBE" in path_by_rank
    reviewed_subtribe_skip = (
        not actual_subtribe
        and skipped_ranks == ["SUBTRIBE"]
        and any(
            str(edge.get("edge_type")) == REVIEWED_RANK_SKIP_EDGE
            and str(edge.get("parent_rank")) == "TRIBE"
            and str(edge.get("child_rank")) == "GENUS"
            and bool(edge.get("reviewed"))
            for edge in chain_edges
        )
    )
    all_nodes_enabled = all(bool(node.get("enabled")) for node in path_nodes)
    all_edges_enabled = all(bool(edge.get("enabled")) for edge in chain_edges)
    structurally_complete = (
        bool(path_nodes)
        and rank_path[0] == "FAMILY"
        and rank_path[-1] == "SPECIES"
        and len(rank_path) == len(set(rank_path))
        and mandatory_present
        and (actual_subtribe or reviewed_subtribe_skip)
    )
    enabled = bool(mapping.get("enabled")) and structurally_complete and all_nodes_enabled and all_edges_enabled
    path_completeness = (
        "complete"
        if enabled and actual_subtribe
        else "reviewed_optional_skip"
        if enabled and reviewed_subtribe_skip
        else "incomplete"
    )
    disabled_reasons: list[str] = []
    if not mapping.get("enabled"):
        disabled_reasons.extend(_split_reasons(mapping.get("disabled_reason")))
    if not structurally_complete:
        disabled_reasons.append("incomplete_path")
    if not all_nodes_enabled:
        disabled_reasons.append("disabled_node_in_path")
    if not all_edges_enabled:
        disabled_reasons.append("disabled_edge_in_path")
    identity = {
        "classification_version": version,
        "accepted_taxon_key": _text(mapping.get("accepted_taxon_key")),
        "rank_path": rank_path,
        "rank_path_node_ids": rank_path_node_ids,
        "skipped_ranks": skipped_ranks,
        "edges": [
            {
                "parent_node_id": _text(edge.get("parent_node_id")),
                "child_node_id": _text(edge.get("child_node_id")),
                "edge_type": _text(edge.get("edge_type")),
                "skipped_ranks": _string_list(edge.get("skipped_ranks")),
            }
            for edge in chain_edges
        ],
    }
    row: dict[str, Any] = {
        "classification_version": version,
        "accepted_taxon_key": _text(mapping.get("accepted_taxon_key")),
        "gbif_species_key": _text(mapping.get("gbif_species_key")),
        "rank_path": rank_path,
        "rank_path_node_ids": rank_path_node_ids,
        "skipped_ranks": skipped_ranks,
        "path_completeness": path_completeness,
        "hierarchy_hash": _sha256_json(identity),
        "source_release": _text(mapping.get("source_release")),
        "enabled": enabled,
        "disabled_reason": ",".join(dict.fromkeys(reason for reason in disabled_reasons if reason)),
    }
    for rank in CLASSIFICATION_RANKS:
        prefix = rank.casefold()
        node = path_by_rank.get(rank, {})
        row[f"{prefix}_node_id"] = _text(node.get("node_id"))
        row[prefix] = _text(node.get("scientific_name"))
    return row


def _prompt_label_frame(*, version: str, nodes: pl.DataFrame, leaf_paths: pl.DataFrame) -> pl.DataFrame:
    usable_node_ids: set[str] = set()
    for row in leaf_paths.filter(pl.col("enabled")).iter_rows(named=True):
        usable_node_ids.update(_string_list(row.get("rank_path_node_ids")))
    rows: list[dict[str, Any]] = []
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
                    "prompt_version": CLASSIFICATION_V3_PROMPT_VERSION,
                    "node_id": node_id,
                    "rank": rank,
                    "scientific_name": name,
                    "label": template.format(name=name),
                    "prompt_template": template,
                    "sort_order": sort_order,
                    "enabled": True,
                }
            )
    return _sort_prompts(_typed_frame(rows, PROMPT_LABEL_SCHEMA))


def validate_classification_v3(
    frames: ClassificationV3Frames,
    *,
    taxa: pl.DataFrame,
) -> list[dict[str, Any]]:
    """Return deterministic construction-level findings.

    Phase 1.5 extends these graph-construction guards with the complete production
    QA contract. Keeping the checks here makes every intermediate artifact build
    fail closed when a source tries to bypass the six-rank transition rules.
    """
    findings: list[dict[str, Any]] = []
    if frames.sources.is_empty():
        findings.append(
            _finding("fatal", "missing_source_catalog", "classification_sources", "", "source catalog is empty")
        )
    for source in frames.sources.iter_rows(named=True):
        missing = [
            field
            for field in ("source_id", "authority", "release", "citation", "retrieved_at", "evidence_url", "evidence")
            if not _text(source.get(field))
        ]
        if missing:
            findings.append(
                _finding(
                    "fatal",
                    "source_missing_provenance",
                    "classification_sources",
                    source.get("source_id"),
                    "classification source is missing required provenance",
                    {"missing_fields": missing},
                )
            )
    for node in frames.nodes.iter_rows(named=True):
        if _text(node.get("rank")) not in CLASSIFICATION_RANKS:
            findings.append(
                _finding(
                    "fatal",
                    "unknown_rank",
                    "classification_nodes",
                    node.get("node_id"),
                    "node rank is outside the six-rank classification contract",
                    {"rank": node.get("rank")},
                )
            )
    for edge in frames.edges.iter_rows(named=True):
        subject = f"{edge['parent_node_id']}->{edge['child_node_id']}"
        edge_type = _text(edge.get("edge_type"))
        skipped_ranks = _string_list(edge.get("skipped_ranks"))
        if not _edge_transition_is_valid(
            parent_rank=_text(edge.get("parent_rank")),
            child_rank=_text(edge.get("child_rank")),
            edge_type=edge_type,
            skipped_ranks=skipped_ranks,
        ):
            code = "invalid_rank_skip" if edge_type == REVIEWED_RANK_SKIP_EDGE else "invalid_rank_transition"
            findings.append(
                _finding(
                    "fatal",
                    code,
                    "classification_edges",
                    subject,
                    "edge violates the six-rank transition contract",
                    {
                        "parent_rank": edge.get("parent_rank"),
                        "child_rank": edge.get("child_rank"),
                        "edge_type": edge_type,
                        "skipped_ranks": skipped_ranks,
                    },
                )
            )
        if edge_type == REVIEWED_RANK_SKIP_EDGE and (
            not edge.get("reviewed")
            or not _text(edge.get("skip_reason"))
            or any(
                not _text(edge.get(field))
                for field in ("source_release", "citation", "retrieved_at", "evidence", "reviewed_by", "reviewed_at")
            )
        ):
            findings.append(
                _finding(
                    "fatal",
                    "unreviewed_rank_skip",
                    "classification_edges",
                    subject,
                    "rank skip lacks complete reviewed provenance",
                )
            )
    enabled_keys = set(frames.leaf_paths.filter(pl.col("enabled"))["accepted_taxon_key"].to_list())
    for row in _accepted_species_rows(taxa):
        key = _text(row.get("accepted_taxon_key"))
        if key not in enabled_keys:
            findings.append(
                _finding(
                    "warning",
                    "unmapped_accepted_species",
                    "classification_leaf_paths",
                    key,
                    "accepted GBIF species has no complete reviewed six-rank path",
                    {
                        "scientific_name": _text(row.get("species"), row.get("scientific_name")),
                        "family": _text(row.get("family")),
                    },
                )
            )
    return sorted(findings, key=lambda finding: (finding["severity"], finding["code"], finding["subject"]))


def _edge_transition_is_valid(
    *,
    parent_rank: str,
    child_rank: str,
    edge_type: str,
    skipped_ranks: Sequence[str],
) -> bool:
    transition = (parent_rank.upper(), child_rank.upper())
    normalized_skips = [rank.upper() for rank in skipped_ranks]
    if edge_type == ASSERTED_PARENT_EDGE:
        return transition in ALLOWED_RANK_TRANSITIONS and not normalized_skips
    return (
        edge_type == REVIEWED_RANK_SKIP_EDGE
        and transition == ("TRIBE", "GENUS")
        and normalized_skips == ["SUBTRIBE"]
    )


def _sort_nodes(frame: pl.DataFrame) -> pl.DataFrame:
    rank_order = {rank: index for index, rank in enumerate(CLASSIFICATION_RANKS)}
    rows = sorted(
        frame.iter_rows(named=True),
        key=lambda row: (
            rank_order.get(_text(row.get("rank")), len(rank_order)),
            _text(row.get("scientific_name")),
            _text(row.get("node_id")),
        ),
    )
    return _typed_frame(rows, NODE_SCHEMA)


def _sort_prompts(frame: pl.DataFrame) -> pl.DataFrame:
    rank_order = {rank: index for index, rank in enumerate(CLASSIFICATION_RANKS)}
    rows = sorted(
        frame.iter_rows(named=True),
        key=lambda row: (
            rank_order.get(_text(row.get("rank")), len(rank_order)),
            _text(row.get("scientific_name")),
            int(row.get("sort_order") or 0),
            _text(row.get("label")),
        ),
    )
    return _typed_frame(rows, PROMPT_LABEL_SCHEMA)


def _canonical_rows(frame: pl.DataFrame, sort_columns: Sequence[str]) -> list[dict[str, Any]]:
    rows = [_canonical_value(dict(row)) for row in frame.iter_rows(named=True)]
    return sorted(
        rows,
        key=lambda row: (
            *(_canonical_json(row.get(column)) for column in sort_columns),
            _canonical_json(row),
        ),
    )


def _canonical_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, set):
        return sorted((_canonical_value(item) for item in value), key=_canonical_json)
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _typed_frame(rows: Sequence[dict[str, Any]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    normalized = [
        {
            name: row.get(name) if row.get(name) is not None else _default_for_dtype(dtype)
            for name, dtype in schema.items()
        }
        for row in rows
    ]
    return pl.DataFrame(normalized, schema=schema, strict=False)


def _default_for_dtype(dtype: pl.DataType) -> object:
    if dtype == pl.Boolean:
        return False
    if dtype in {pl.Int32, pl.Int64, pl.UInt32, pl.UInt64}:
        return 0
    if isinstance(dtype, pl.List):
        return []
    return ""


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Sequence[object] = [value]
    elif isinstance(value, set):
        values = sorted(value, key=str)
    elif isinstance(value, Sequence):
        values = value
    else:
        values = [value]
    return [text for item in values if (text := _text(item))]


def _split_reasons(value: object) -> list[str]:
    return [reason.strip() for reason in _text(value).split(",") if reason.strip()]


def _accepted_species_rows(taxa: pl.DataFrame) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in taxa.iter_rows(named=True)
        if _text(row.get("rank")).upper() == "SPECIES"
        and _text(row.get("taxonomic_status"), row.get("status")).upper() == "ACCEPTED"
    ]


def _coverage_by_family(accepted_species: list[dict[str, Any]], enabled_keys: set[str]) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, int | str]] = {}
    for row in accepted_species:
        family = _text(row.get("family")) or "unknown"
        bucket = counts.setdefault(family, {"family": family, "accepted_species": 0, "mapped_species": 0})
        bucket["accepted_species"] = int(bucket["accepted_species"]) + 1
        if _text(row.get("accepted_taxon_key")) in enabled_keys:
            bucket["mapped_species"] = int(bucket["mapped_species"]) + 1
    return [counts[family] for family in sorted(counts)]


def _finding(
    severity: str,
    code: str,
    table: str,
    subject: object,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "table": table,
        "subject": _text(subject),
        "message": message,
        "details": dict(details or {}),
    }


def _text(*values: object) -> str:
    for value in values:
        text = " ".join(str(value if value is not None else "").strip().split())
        if text:
            return text
    return ""


def _bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _review_fields(row: dict[str, Any]) -> dict[str, object]:
    status = _text(row.get("review_status")) or ("reviewed" if _bool(row.get("reviewed")) else "candidate")
    reviewed_by = _text(row.get("reviewed_by"))
    reviewed_at = _text(row.get("reviewed_at"))
    reviewed = _bool(row.get("reviewed")) and status == "reviewed" and bool(reviewed_by) and bool(reviewed_at)
    return {
        "reviewed": reviewed,
        "review_status": status,
        "reviewed_by": reviewed_by,
        "reviewed_at": reviewed_at,
    }


def _bare_gbif_key(*values: object) -> str:
    text = _text(*values)
    return text.split(":", 1)[1] if text.casefold().startswith("gbif:") else text


def _first_value(frame: pl.DataFrame, column: str) -> str:
    if frame.is_empty() or column not in frame.columns:
        return ""
    return _text(frame[column][0])


def _read_json_optional(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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
    "build_classification_v3_frames",
    "build_classification_v3_manifest",
    "classification_v3_artifact_paths",
    "classification_v3_artifact_uris",
    "classification_v3_fingerprint",
    "classification_v3_qa_frame",
    "compile_classification_v3_artifacts",
    "hierarchy_fingerprint",
    "load_classification_v3_source",
    "validate_classification_v3",
    "write_classification_v3_artifacts",
]
