from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from biominer.storage.parquet import write_parquet


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
DEFAULT_CLASSIFICATION_V2_SOURCE = Path("config/taxonomy/papilionoidea_classification_v2.json")

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


def load_classification_v2_source(path: str | Path = DEFAULT_CLASSIFICATION_V2_SOURCE) -> dict[str, Any]:
    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"classification source must be a JSON object: {source_path}")
    return payload


def write_classification_v2_artifacts(
    registry_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    source_path: str | Path = DEFAULT_CLASSIFICATION_V2_SOURCE,
) -> dict[str, Any]:
    registry = Path(registry_dir)
    output = Path(output_dir) if output_dir is not None else registry
    taxa_path = registry / "taxa.parquet"
    if not taxa_path.exists():
        raise FileNotFoundError(f"missing required registry artifact: {taxa_path}")
    registry_manifest = _read_json_optional(registry / "manifest.json")
    taxa = pl.read_parquet(taxa_path)
    frames = build_classification_v2_frames(taxa, load_classification_v2_source(source_path))
    findings = validate_classification_v2(frames, taxa=taxa)
    qa_findings = classification_v2_qa_frame(findings)
    manifest = build_classification_v2_manifest(
        frames,
        registry_version=_text(registry_manifest.get("registry_version")),
    )
    manifest["classification_fingerprint"] = classification_v2_fingerprint(frames)
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
    output.mkdir(parents=True, exist_ok=True)
    paths = classification_v2_artifact_paths(output)
    write_parquet(qa_findings, paths["qa_findings"])
    if manifest["fatal_finding_count"]:
        raise ValueError(
            "classification-v2 fatal QA: "
            + ", ".join(str(finding["code"]) for finding in findings if finding["severity"] == "fatal")
        )
    for key, frame in (
        ("sources", frames.sources),
        ("nodes", frames.nodes),
        ("edges", frames.edges),
        ("gbif_mappings", frames.gbif_mappings),
        ("leaf_paths", frames.leaf_paths),
        ("prompt_labels", frames.prompt_labels),
    ):
        write_parquet(frame, paths[key])
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {**manifest, "outputs": {key: str(path) for key, path in paths.items()}}


def validate_classification_v2(
    frames: ClassificationV2Frames,
    *,
    taxa: pl.DataFrame,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    _validate_sources(frames.sources, findings)
    _validate_nodes(frames.nodes, findings)
    _validate_edges(frames.nodes, frames.edges, findings)
    _validate_mappings(frames.nodes, frames.gbif_mappings, findings)
    _validate_leaf_paths(frames.leaf_paths, findings)
    _append_registry_coverage_gaps(taxa, frames.leaf_paths, findings)
    return findings


def classification_v2_qa_frame(findings: Sequence[dict[str, Any]]) -> pl.DataFrame:
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
    return _frame(rows, QA_FINDING_SCHEMA).sort(["severity", "code", "subject"])


def classification_v2_artifact_paths(root: str | Path) -> dict[str, Path]:
    base = Path(root)
    return {
        "sources": base / CLASSIFICATION_V2_SOURCES_FILE,
        "nodes": base / CLASSIFICATION_V2_NODES_FILE,
        "edges": base / CLASSIFICATION_V2_EDGES_FILE,
        "gbif_mappings": base / CLASSIFICATION_V2_GBIF_MAPPINGS_FILE,
        "leaf_paths": base / CLASSIFICATION_V2_LEAF_PATHS_FILE,
        "prompt_labels": base / CLASSIFICATION_V2_PROMPT_LABELS_FILE,
        "qa_findings": base / CLASSIFICATION_V2_QA_FINDINGS_FILE,
        "manifest": base / CLASSIFICATION_V2_MANIFEST_FILE,
    }


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
        review = _review_fields(row)
        reviewed = bool(review["reviewed"])
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
                "review_status": review["review_status"],
                "reviewed_by": review["reviewed_by"],
                "reviewed_at": review["reviewed_at"],
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
        review = _review_fields(row)
        reviewed = bool(review["reviewed"])
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
                "review_status": review["review_status"],
                "reviewed_by": review["reviewed_by"],
                "reviewed_at": review["reviewed_at"],
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
        review = _review_fields(row)
        reviewed = bool(review["reviewed"])
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
                "review_status": review["review_status"],
                "reviewed_by": review["reviewed_by"],
                "reviewed_at": review["reviewed_at"],
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


def _validate_sources(frame: pl.DataFrame, findings: list[dict[str, Any]]) -> None:
    if frame.is_empty():
        findings.append(_finding("fatal", "missing_source_catalog", "classification_sources", "", "source catalog is empty"))
        return
    _append_duplicate_findings(frame, ("source_id",), "duplicate_source_id", "classification_sources", findings)
    for row in frame.iter_rows(named=True):
        missing = [
            field
            for field in ("source_id", "authority", "release", "citation", "retrieved_at", "evidence_url", "evidence")
            if not _text(row.get(field))
        ]
        if missing:
            findings.append(
                _finding(
                    "fatal",
                    "source_missing_provenance",
                    "classification_sources",
                    row.get("source_id"),
                    "classification source is missing required provenance",
                    {"missing_fields": missing},
                )
            )


def _validate_nodes(frame: pl.DataFrame, findings: list[dict[str, Any]]) -> None:
    _append_duplicate_findings(frame, ("node_id",), "duplicate_node_id", "classification_nodes", findings)
    invalid = frame.filter(~pl.col("rank").is_in(CLASSIFICATION_RANKS))
    for row in invalid.iter_rows(named=True):
        findings.append(
            _finding(
                "fatal",
                "invalid_node_rank",
                "classification_nodes",
                row.get("node_id"),
                "node rank is outside the five-rank classification contract",
                {"rank": row.get("rank")},
            )
        )
    for rank in CLASSIFICATION_RANKS:
        if frame.filter((pl.col("rank") == rank) & pl.col("enabled")).is_empty():
            findings.append(
                _finding(
                    "fatal",
                    "missing_enabled_rank",
                    "classification_nodes",
                    rank,
                    f"no enabled {rank} node exists",
                )
            )
    required = ("node_id", "scientific_name", "source_id", "source_release", "citation", "retrieved_at", "evidence", "reviewed_by", "reviewed_at")
    for row in frame.filter(pl.col("enabled")).iter_rows(named=True):
        missing = [field for field in required if not _text(row.get(field))]
        if not row.get("reviewed") or row.get("review_status") != "reviewed" or missing:
            findings.append(
                _finding(
                    "fatal",
                    "enabled_node_without_reviewed_provenance",
                    "classification_nodes",
                    row.get("node_id"),
                    "enabled node lacks reviewed provenance",
                    {"missing_fields": missing},
                )
            )


def _validate_edges(nodes: pl.DataFrame, edges: pl.DataFrame, findings: list[dict[str, Any]]) -> None:
    _append_duplicate_findings(
        edges,
        ("parent_node_id", "child_node_id"),
        "duplicate_edge",
        "classification_edges",
        findings,
    )
    known_nodes = set(nodes["node_id"].to_list())
    for row in edges.iter_rows(named=True):
        subject = f"{row['parent_node_id']}->{row['child_node_id']}"
        if row["parent_node_id"] not in known_nodes or row["child_node_id"] not in known_nodes:
            findings.append(_finding("fatal", "edge_unknown_node", "classification_edges", subject, "edge references an unknown node"))
        if (row["parent_rank"], row["child_rank"]) not in ALLOWED_RANK_TRANSITIONS:
            findings.append(
                _finding(
                    "fatal",
                    "invalid_edge_rank_transition",
                    "classification_edges",
                    subject,
                    "edge violates the allowed rank order",
                    {"parent_rank": row["parent_rank"], "child_rank": row["child_rank"]},
                )
            )
        if row["enabled"] and (
            not row["reviewed"]
            or row["review_status"] != "reviewed"
            or any(not _text(row.get(field)) for field in ("source_release", "citation", "retrieved_at", "evidence", "reviewed_by", "reviewed_at"))
        ):
            findings.append(
                _finding(
                    "fatal",
                    "enabled_edge_without_reviewed_provenance",
                    "classification_edges",
                    subject,
                    "enabled edge lacks reviewed provenance",
                )
            )
    enabled = edges.filter(pl.col("enabled"))
    parents = enabled.group_by("child_node_id").agg(pl.col("parent_node_id").n_unique().alias("parent_count"))
    for row in parents.filter(pl.col("parent_count") > 1).iter_rows(named=True):
        findings.append(
            _finding(
                "fatal",
                "enabled_node_has_multiple_parents",
                "classification_edges",
                row.get("child_node_id"),
                "enabled node has more than one enabled parent",
            )
        )
    cycle = _first_cycle(enabled)
    if cycle:
        findings.append(
            _finding(
                "fatal",
                "enabled_hierarchy_cycle",
                "classification_edges",
                cycle[0],
                "enabled classification edges contain a cycle",
                {"cycle": cycle},
            )
        )


def _validate_mappings(nodes: pl.DataFrame, mappings: pl.DataFrame, findings: list[dict[str, Any]]) -> None:
    known_species_nodes = set(nodes.filter(pl.col("rank") == "SPECIES")["node_id"].to_list())
    for row in mappings.iter_rows(named=True):
        subject = row.get("gbif_species_key")
        if row.get("species_node_id") not in known_species_nodes:
            findings.append(
                _finding("fatal", "mapping_unknown_species_node", "species_gbif_mappings", subject, "GBIF mapping references an unknown species node")
            )
        if row["enabled"] and (
            row["taxonomic_status"] != "ACCEPTED"
            or not row["reviewed"]
            or row["review_status"] != "reviewed"
            or any(not _text(row.get(field)) for field in ("accepted_taxon_key", "gbif_species_key", "accepted_scientific_name", "source_release", "citation", "retrieved_at", "evidence", "reviewed_by", "reviewed_at"))
        ):
            findings.append(
                _finding(
                    "fatal",
                    "enabled_mapping_invalid",
                    "species_gbif_mappings",
                    subject,
                    "enabled GBIF mapping is not accepted, reviewed, and fully sourced",
                )
            )
    enabled = mappings.filter(pl.col("enabled"))
    if enabled.is_empty():
        findings.append(_finding("fatal", "no_enabled_gbif_mapping", "species_gbif_mappings", "", "no enabled GBIF species mapping exists"))
    _append_duplicate_findings(
        enabled,
        ("gbif_species_key",),
        "duplicate_enabled_gbif_mapping",
        "species_gbif_mappings",
        findings,
    )
    _append_duplicate_findings(
        enabled,
        ("species_node_id",),
        "duplicate_enabled_species_node_mapping",
        "species_gbif_mappings",
        findings,
    )


def _validate_leaf_paths(paths: pl.DataFrame, findings: list[dict[str, Any]]) -> None:
    enabled = paths.filter(pl.col("enabled"))
    if enabled.is_empty():
        findings.append(_finding("fatal", "no_enabled_leaf_path", "classification_leaf_paths", "", "no complete enabled five-rank leaf path exists"))
        return
    _append_duplicate_findings(
        enabled,
        ("accepted_taxon_key",),
        "duplicate_enabled_leaf_path",
        "classification_leaf_paths",
        findings,
    )
    required = [
        item
        for rank in CLASSIFICATION_RANKS
        for item in (f"{rank.casefold()}_node_id", rank.casefold())
    ]
    for row in enabled.iter_rows(named=True):
        missing = [field for field in required if not _text(row.get(field))]
        if missing:
            findings.append(
                _finding(
                    "fatal",
                    "enabled_leaf_path_has_gap",
                    "classification_leaf_paths",
                    row.get("accepted_taxon_key"),
                    "enabled leaf path skips one or more required ranks",
                    {"missing_fields": missing},
                )
            )


def _append_registry_coverage_gaps(
    taxa: pl.DataFrame,
    leaf_paths: pl.DataFrame,
    findings: list[dict[str, Any]],
) -> None:
    enabled_keys = set(leaf_paths.filter(pl.col("enabled"))["accepted_taxon_key"].to_list())
    for row in _accepted_species_rows(taxa):
        key = _text(row.get("accepted_taxon_key"))
        if key not in enabled_keys:
            findings.append(
                _finding(
                    "warning",
                    "unmapped_accepted_species",
                    "classification_leaf_paths",
                    key,
                    "accepted GBIF species has no complete reviewed five-rank path",
                    {"scientific_name": _text(row.get("species"), row.get("scientific_name")), "family": _text(row.get("family"))},
                )
            )


def _append_duplicate_findings(
    frame: pl.DataFrame,
    keys: tuple[str, ...],
    code: str,
    table: str,
    findings: list[dict[str, Any]],
) -> None:
    if frame.is_empty():
        return
    duplicates = frame.group_by(list(keys)).len().filter(pl.col("len") > 1)
    for row in duplicates.iter_rows(named=True):
        subject = "|".join(_text(row.get(key)) for key in keys)
        findings.append(_finding("fatal", code, table, subject, f"duplicate rows exist for {', '.join(keys)}"))


def _first_cycle(edges: pl.DataFrame) -> list[str]:
    children: dict[str, list[str]] = {}
    for row in edges.iter_rows(named=True):
        children.setdefault(str(row["parent_node_id"]), []).append(str(row["child_node_id"]))
    visited: set[str] = set()
    active: list[str] = []

    def visit(node: str) -> list[str]:
        if node in active:
            start = active.index(node)
            return [*active[start:], node]
        if node in visited:
            return []
        active.append(node)
        for child in children.get(node, ()):
            cycle = visit(child)
            if cycle:
                return cycle
        active.pop()
        visited.add(node)
        return []

    for node in sorted(children):
        cycle = visit(node)
        if cycle:
            return cycle
    return []


def _accepted_species_rows(taxa: pl.DataFrame) -> list[dict[str, Any]]:
    return [
        row
        for row in taxa.iter_rows(named=True)
        if _text(row.get("rank")).upper() == "SPECIES"
        and (_text(row.get("taxonomic_status"), row.get("status")).upper() or "ACCEPTED") == "ACCEPTED"
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


__all__ = [
    "ALLOWED_RANK_TRANSITIONS",
    "CLASSIFICATION_RANKS",
    "CLASSIFICATION_V2_VERSION",
    "ClassificationV2Frames",
    "build_classification_v2_frames",
    "build_classification_v2_manifest",
    "classification_v2_artifact_paths",
    "classification_v2_fingerprint",
    "load_classification_v2_source",
    "write_classification_v2_artifacts",
]
