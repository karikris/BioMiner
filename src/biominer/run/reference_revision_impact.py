"""Transitive artifact impact analysis for a reference-bank revision."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.adaptive_bank_revision import (
    AdaptiveSupportBankRevision,
    validate_adaptive_support_bank_revision,
)
from biominer.storage.parquet import write_parquet


REFERENCE_REVISION_IMPACT_FILE = "reference_revision_impact.parquet"
REFERENCE_REVISION_IMPACT_SCHEMA_VERSION = (
    "reference-revision-impact-v1.0.0"
)
REFERENCE_ARTIFACT_CATALOG_SCHEMA_VERSION = (
    "reference-artifact-catalog-v1.0.0"
)
REFERENCE_ARTIFACT_DEPENDENCY_SCHEMA_VERSION = (
    "reference-artifact-dependency-v1.0.0"
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


REFERENCE_ARTIFACT_CATALOG_SCHEMA = {
    "schema_version": pl.String,
    "artifact_id": pl.String,
    "artifact_type": pl.String,
    "artifact_fingerprint": pl.String,
    "reference_bank_fingerprint": pl.String,
    "species": pl.List(pl.String),
    "routes": pl.List(pl.String),
    "region": pl.String,
    "record_ids": pl.List(pl.String),
}

REFERENCE_ARTIFACT_DEPENDENCY_SCHEMA = {
    "schema_version": pl.String,
    "upstream_artifact_id": pl.String,
    "downstream_artifact_id": pl.String,
    "dependency_kind": pl.String,
    "dependency_fingerprint": pl.String,
}


def reference_revision_impact_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "revision_fingerprint": pl.String,
        "artifact_id": pl.String,
        "artifact_type": pl.String,
        "artifact_fingerprint": pl.String,
        "impact_status": pl.String,
        "directly_affected": pl.Boolean,
        "impact_depth": pl.UInt32,
        "affected_reference_media_ids": pl.List(pl.String),
        "affected_species": pl.List(pl.String),
        "affected_routes": pl.List(pl.String),
        "affected_regions": pl.List(pl.String),
        "affected_record_ids": pl.List(pl.String),
        "impact_reasons": pl.List(pl.String),
        "expected_action": pl.String,
        "old_reference_bank_fingerprint": pl.String,
        "new_reference_bank_fingerprint": pl.String,
        "impact_fingerprint": pl.String,
    }


def reference_artifact_catalog_frame(
    rows: Sequence[Mapping[str, object]] | None = None,
) -> pl.DataFrame:
    normalized = []
    for source in rows or []:
        row = dict(source)
        row.setdefault("schema_version", REFERENCE_ARTIFACT_CATALOG_SCHEMA_VERSION)
        normalized.append(row)
    frame = pl.DataFrame(
        normalized,
        schema=REFERENCE_ARTIFACT_CATALOG_SCHEMA,
        orient="row",
        strict=True,
    ).sort("artifact_id")
    if frame["artifact_id"].n_unique() != frame.height:
        raise ValueError("reference artifact catalog repeats an artifact ID")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != REFERENCE_ARTIFACT_CATALOG_SCHEMA_VERSION:
            raise ValueError("unsupported reference artifact catalog version")
        _required_text(row["artifact_id"], field="artifact_id")
        artifact_type = _required_text(row["artifact_type"], field="artifact_type")
        _full_sha256(row["artifact_fingerprint"], field="artifact_fingerprint")
        _full_sha256(
            row["reference_bank_fingerprint"],
            field="reference_bank_fingerprint",
        )
        for field in ("species", "routes", "record_ids"):
            values = row[field]
            if values != sorted(set(values)):
                raise ValueError(f"artifact catalog {field} must be canonical")
            for value in values:
                _required_text(value, field=field)
        if row["region"] is not None:
            _required_text(row["region"], field="region")
        if artifact_type == "flickr_score_partition" and not row["record_ids"]:
            raise ValueError("Flickr score partitions require explicit record IDs")
        if artifact_type != "flickr_score_partition" and row["record_ids"]:
            raise ValueError("only Flickr score partitions may carry record IDs")
    return frame


def reference_artifact_dependencies_frame(
    rows: Sequence[Mapping[str, object]] | None = None,
) -> pl.DataFrame:
    normalized = []
    for source in rows or []:
        row = dict(source)
        row.setdefault(
            "schema_version",
            REFERENCE_ARTIFACT_DEPENDENCY_SCHEMA_VERSION,
        )
        if not row.get("dependency_fingerprint"):
            row["dependency_fingerprint"] = ""
            payload = dict(row)
            payload.pop("dependency_fingerprint")
            row["dependency_fingerprint"] = canonical_semantic_fingerprint(payload)
        normalized.append(row)
    frame = pl.DataFrame(
        normalized,
        schema=REFERENCE_ARTIFACT_DEPENDENCY_SCHEMA,
        orient="row",
        strict=True,
    ).sort("upstream_artifact_id", "downstream_artifact_id", "dependency_kind")
    identity = ["upstream_artifact_id", "downstream_artifact_id", "dependency_kind"]
    if frame.select(identity).unique().height != frame.height:
        raise ValueError("reference artifact dependencies repeat an edge")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != REFERENCE_ARTIFACT_DEPENDENCY_SCHEMA_VERSION:
            raise ValueError("unsupported artifact dependency schema version")
        upstream = _required_text(
            row["upstream_artifact_id"],
            field="upstream_artifact_id",
        )
        downstream = _required_text(
            row["downstream_artifact_id"],
            field="downstream_artifact_id",
        )
        if upstream == downstream:
            raise ValueError("an artifact cannot depend on itself")
        _required_text(row["dependency_kind"], field="dependency_kind")
        payload = dict(row)
        fingerprint = payload.pop("dependency_fingerprint")
        if fingerprint != canonical_semantic_fingerprint(payload):
            raise ValueError("artifact dependency fingerprint mismatch")
    return frame


def calculate_reference_revision_impact(
    revision: AdaptiveSupportBankRevision,
    artifact_catalog: pl.DataFrame,
    dependencies: pl.DataFrame,
) -> pl.DataFrame:
    """Propagate direct reference changes through a validated artifact DAG."""

    validate_adaptive_support_bank_revision(revision)
    catalog = reference_artifact_catalog_frame(artifact_catalog.to_dicts())
    edges = reference_artifact_dependencies_frame(dependencies.to_dicts())
    node_by_id = {
        str(row["artifact_id"]): row for row in catalog.iter_rows(named=True)
    }
    node_ids = set(node_by_id)
    for row in catalog.iter_rows(named=True):
        if row["reference_bank_fingerprint"] != (
            revision.old_reference_bank_fingerprint
        ):
            raise ValueError("artifact catalog reference-bank binding is stale")
    edge_ids = {
        str(value)
        for row in edges.iter_rows(named=True)
        for value in (row["upstream_artifact_id"], row["downstream_artifact_id"])
    }
    unknown_edge_ids = sorted(edge_ids - node_ids)
    if unknown_edge_ids:
        raise ValueError(
            "artifact dependency graph references unknown nodes: "
            + ", ".join(unknown_edge_ids)
        )
    invalidation_ids = set(revision.invalidation_manifest["artifact_id"])
    missing_direct_nodes = sorted(invalidation_ids - node_ids)
    if missing_direct_nodes:
        raise ValueError(
            "artifact catalog omits revision dependencies: "
            + ", ".join(missing_direct_nodes)
        )

    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in edges.iter_rows(named=True):
        upstream = str(edge["upstream_artifact_id"])
        downstream = str(edge["downstream_artifact_id"])
        adjacency[upstream].append(downstream)
        indegree[downstream] += 1
    for values in adjacency.values():
        values.sort()
    order = _topological_order(adjacency, indegree)

    changes_by_id = {
        str(row["reference_media_id"]): row
        for row in revision.change_manifest.iter_rows(named=True)
    }
    direct_by_id = {
        str(row["artifact_id"]): row
        for row in revision.invalidation_manifest.iter_rows(named=True)
        if row["invalidated"]
    }
    state: dict[str, dict[str, object]] = {}
    for artifact_id, invalidation in direct_by_id.items():
        reference_ids = sorted(invalidation["affected_reference_media_ids"])
        change_rows = [changes_by_id[reference_id] for reference_id in reference_ids]
        node = node_by_id[artifact_id]
        state[artifact_id] = {
            "depth": 0,
            "direct": True,
            "reference_ids": set(reference_ids),
            "species": {str(row["scientific_name"]) for row in change_rows},
            "routes": {
                str(route)
                for row in change_rows
                for route in (row["old_route"], row["new_route"])
                if route is not None
            },
            "regions": {str(node["region"])} if node["region"] else set(),
            "reasons": {
                f"direct_reference_change:{row['change_type']}"
                for row in change_rows
            },
        }

    for artifact_id in order:
        parent = state.get(artifact_id)
        if parent is None:
            continue
        for downstream_id in adjacency[artifact_id]:
            child = state.setdefault(
                downstream_id,
                {
                    "depth": int(parent["depth"]) + 1,
                    "direct": False,
                    "reference_ids": set(),
                    "species": set(),
                    "routes": set(),
                    "regions": set(),
                    "reasons": set(),
                },
            )
            child["depth"] = min(
                int(child["depth"]),
                int(parent["depth"]) + 1,
            )
            for field in ("reference_ids", "species", "routes", "regions"):
                child[field].update(parent[field])  # type: ignore[union-attr]
            if node_by_id[downstream_id]["region"]:
                child["regions"].add(  # type: ignore[union-attr]
                    str(node_by_id[downstream_id]["region"])
                )
            child["reasons"].add(  # type: ignore[union-attr]
                f"upstream_artifact_changed:{artifact_id}"
            )

    rows: list[dict[str, object]] = []
    for artifact_id in sorted(node_ids):
        node = node_by_id[artifact_id]
        impact = state.get(artifact_id)
        affected = impact is not None
        row = {
            "schema_version": REFERENCE_REVISION_IMPACT_SCHEMA_VERSION,
            "revision_fingerprint": revision.revision_fingerprint,
            "artifact_id": artifact_id,
            "artifact_type": node["artifact_type"],
            "artifact_fingerprint": node["artifact_fingerprint"],
            "impact_status": "affected" if affected else "reusable_as_is",
            "directly_affected": bool(impact and impact["direct"]),
            "impact_depth": int(impact["depth"]) if impact else None,
            "affected_reference_media_ids": (
                sorted(impact["reference_ids"]) if impact else []
            ),
            "affected_species": sorted(impact["species"]) if impact else [],
            "affected_routes": sorted(impact["routes"]) if impact else [],
            "affected_regions": sorted(impact["regions"]) if impact else [],
            "affected_record_ids": (
                list(node["record_ids"])
                if affected and node["artifact_type"] == "flickr_score_partition"
                else []
            ),
            "impact_reasons": sorted(impact["reasons"]) if impact else [],
            "expected_action": (
                _affected_action(str(node["artifact_type"]))
                if affected
                else "reuse_without_recomputation"
            ),
            "old_reference_bank_fingerprint": (
                revision.old_reference_bank_fingerprint
            ),
            "new_reference_bank_fingerprint": (
                revision.new_reference_bank_fingerprint
            ),
            "impact_fingerprint": "",
        }
        payload = dict(row)
        payload.pop("impact_fingerprint")
        row["impact_fingerprint"] = canonical_semantic_fingerprint(payload)
        rows.append(row)
    result = pl.DataFrame(
        rows,
        schema=reference_revision_impact_schema(),
        orient="row",
        strict=True,
    ).sort("artifact_id")
    validate_reference_revision_impact(result)
    return result


def validate_reference_revision_impact(frame: pl.DataFrame) -> None:
    if frame.schema != reference_revision_impact_schema():
        raise ValueError("reference revision impact schema mismatch")
    if frame["artifact_id"].n_unique() != frame.height:
        raise ValueError("reference revision impact repeats an artifact")
    if not frame.equals(frame.sort("artifact_id")):
        raise ValueError("reference revision impact is not deterministically sorted")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != REFERENCE_REVISION_IMPACT_SCHEMA_VERSION:
            raise ValueError("unsupported reference revision impact version")
        affected = row["impact_status"] == "affected"
        if row["impact_status"] not in {"affected", "reusable_as_is"}:
            raise ValueError("unsupported reference revision impact status")
        evidence_fields = (
            "affected_reference_media_ids",
            "affected_species",
            "affected_routes",
            "affected_regions",
            "impact_reasons",
        )
        if affected:
            if row["impact_depth"] is None or not row[
                "affected_reference_media_ids"
            ]:
                raise ValueError("affected artifact lacks propagated evidence")
        elif (
            row["impact_depth"] is not None
            or row["directly_affected"]
            or any(row[field] for field in evidence_fields)
            or row["affected_record_ids"]
            or row["expected_action"] != "reuse_without_recomputation"
        ):
            raise ValueError("reusable artifact contains impact evidence")
        if row["directly_affected"] and row["impact_depth"] != 0:
            raise ValueError("directly affected artifact must have impact depth zero")
        for field in (*evidence_fields, "affected_record_ids"):
            if row[field] != sorted(set(row[field])):
                raise ValueError(f"reference revision impact {field} is not canonical")
        payload = dict(row)
        fingerprint = payload.pop("impact_fingerprint")
        if fingerprint != canonical_semantic_fingerprint(payload):
            raise ValueError("reference revision impact fingerprint mismatch")


def write_reference_revision_impact(
    frame: pl.DataFrame,
    output: str | Path,
) -> Path:
    validate_reference_revision_impact(frame)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= REFERENCE_REVISION_IMPACT_FILE
    return write_parquet(frame, destination)


def _topological_order(
    adjacency: Mapping[str, Sequence[str]],
    indegree: Mapping[str, int],
) -> list[str]:
    remaining = dict(indegree)
    queue = deque(sorted(node for node, degree in remaining.items() if degree == 0))
    order: list[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for downstream in adjacency[node]:
            remaining[downstream] -= 1
            if remaining[downstream] == 0:
                queue.append(downstream)
        queue = deque(sorted(queue))
    if len(order) != len(remaining):
        raise ValueError("reference artifact dependency graph contains a cycle")
    return order


def _affected_action(artifact_type: str) -> str:
    return {
        "reference_embedding": "reuse_unchanged_vectors_filter_removed_references",
        "reference_prototype": "rebuild_affected_prototype",
        "regional_prototype": "rebuild_affected_regional_prototype",
        "classifier": "refresh_affected_classifier",
        "calibrator": "refresh_if_training_data_changed",
        "candidate_set": "rebuild_affected_candidate_set",
        "flickr_score_partition": "selectively_rescore_affected_records",
    }.get(artifact_type, "rebuild_affected_artifact")


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank")
    return text


def _full_sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full sha256 fingerprint")
    return text


__all__ = [
    "REFERENCE_ARTIFACT_CATALOG_SCHEMA",
    "REFERENCE_ARTIFACT_CATALOG_SCHEMA_VERSION",
    "REFERENCE_ARTIFACT_DEPENDENCY_SCHEMA",
    "REFERENCE_ARTIFACT_DEPENDENCY_SCHEMA_VERSION",
    "REFERENCE_REVISION_IMPACT_FILE",
    "REFERENCE_REVISION_IMPACT_SCHEMA_VERSION",
    "calculate_reference_revision_impact",
    "reference_artifact_catalog_frame",
    "reference_artifact_dependencies_frame",
    "reference_revision_impact_schema",
    "validate_reference_revision_impact",
    "write_reference_revision_impact",
]
