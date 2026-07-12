from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Self, Sequence

import polars as pl

from biominer.registry.classification_v3 import (
    CLASSIFICATION_PROMPT_STAGES,
    CLASSIFICATION_RANKS,
    CLASSIFICATION_V3_PROMPT_VERSION,
    CLASSIFICATION_V3_VERSION,
    EDGE_SCHEMA,
    GBIF_MAPPING_SCHEMA,
    LEAF_PATH_SCHEMA,
    NODE_SCHEMA,
    OPTIONAL_CLASSIFICATION_RANKS,
    PROMPT_LABEL_SCHEMA,
    QA_FINDING_SCHEMA,
    RANK_SCREEN_PROMPT_STAGE,
    SOURCE_SCHEMA,
    SPECIES_FIRST_PASS_PROMPT_STAGE,
    SPECIES_RERANK_PROMPT_STAGE,
    ClassificationV3Frames,
    classification_v3_artifact_paths,
    classification_v3_fingerprint,
    hierarchy_fingerprint,
)
from biominer.registry.unified import PATH_RANKS as BIOCLIP_SUPPORTED_RANKS, stable_identity


_CHECKSUM_ARTIFACTS = (
    "sources",
    "nodes",
    "edges",
    "gbif_mappings",
    "leaf_paths",
    "prompt_labels",
    "qa_findings",
)
_PATH_SORT_COLUMNS = (
    *(rank.casefold() for rank in CLASSIFICATION_RANKS),
    *(f"{rank.casefold()}_node_id" for rank in CLASSIFICATION_RANKS),
    "accepted_taxon_key",
    "hierarchy_hash",
)


@dataclass(frozen=True)
class PathTaxonomyStore:
    sources: pl.DataFrame
    nodes: pl.DataFrame
    edges: pl.DataFrame
    gbif_mappings: pl.DataFrame
    leaf_paths: pl.DataFrame
    prompt_labels: pl.DataFrame
    qa_findings: pl.DataFrame
    manifest: Mapping[str, object]
    _enabled_hierarchy_hashes: frozenset[str]

    @classmethod
    def from_frames(
        cls,
        *,
        sources: pl.DataFrame,
        nodes: pl.DataFrame,
        edges: pl.DataFrame,
        gbif_mappings: pl.DataFrame,
        leaf_paths: pl.DataFrame,
        prompt_labels: pl.DataFrame,
        qa_findings: pl.DataFrame,
        manifest: Mapping[str, object],
    ) -> Self:
        manifest_payload = dict(manifest)
        _validate_manifest_header(manifest_payload)
        normalized = {
            "sources": _require_schema(sources, SOURCE_SCHEMA, artifact="sources"),
            "nodes": _require_schema(nodes, NODE_SCHEMA, artifact="nodes"),
            "edges": _require_schema(edges, EDGE_SCHEMA, artifact="edges"),
            "gbif_mappings": _require_schema(
                gbif_mappings,
                GBIF_MAPPING_SCHEMA,
                artifact="gbif_mappings",
            ),
            "leaf_paths": _require_schema(
                leaf_paths,
                LEAF_PATH_SCHEMA,
                artifact="leaf_paths",
            ),
            "prompt_labels": _require_schema(
                prompt_labels,
                PROMPT_LABEL_SCHEMA,
                artifact="prompt_labels",
            ),
            "qa_findings": _require_schema(
                qa_findings,
                QA_FINDING_SCHEMA,
                artifact="qa_findings",
            ),
        }
        store = cls(
            **normalized,
            manifest=MappingProxyType(manifest_payload),
            _enabled_hierarchy_hashes=frozenset(
                str(value)
                for value in normalized["leaf_paths"]
                .filter(pl.col("enabled"))["hierarchy_hash"]
                .to_list()
            ),
        )
        store._validate_loaded_frames()
        return store

    @classmethod
    def read(cls, root: str | Path) -> Self:
        unified_paths = Path(root) / "species_paths.parquet"
        if unified_paths.exists():
            return _read_unified_registry_store(unified_paths)
        paths = classification_v3_artifact_paths(root)
        required = (*_CHECKSUM_ARTIFACTS, "manifest")
        missing = [str(paths[key]) for key in required if not paths[key].exists()]
        if missing:
            raise FileNotFoundError("missing classification-v3 artifacts: " + ", ".join(missing))
        manifest = _read_manifest(paths["manifest"])
        _validate_manifest_header(manifest)
        _validate_artifact_checksums(paths, manifest)
        sources = _scan(paths["sources"], SOURCE_SCHEMA, artifact="sources")
        nodes = _scan(paths["nodes"], NODE_SCHEMA, artifact="nodes")
        edges = _scan(paths["edges"], EDGE_SCHEMA, artifact="edges")
        gbif_mappings = _scan(paths["gbif_mappings"], GBIF_MAPPING_SCHEMA, artifact="gbif_mappings")
        leaf_paths = _scan(paths["leaf_paths"], LEAF_PATH_SCHEMA, artifact="leaf_paths")
        prompt_labels = _scan(paths["prompt_labels"], PROMPT_LABEL_SCHEMA, artifact="prompt_labels")
        qa_findings = _scan(paths["qa_findings"], QA_FINDING_SCHEMA, artifact="qa_findings")
        return cls.from_frames(
            sources=sources,
            nodes=nodes,
            edges=edges,
            gbif_mappings=gbif_mappings,
            leaf_paths=leaf_paths,
            prompt_labels=prompt_labels,
            qa_findings=qa_findings,
            manifest=manifest,
        )

    @classmethod
    def from_species_paths(cls, paths: pl.DataFrame) -> Self:
        return UnifiedPathTaxonomyStore(paths)  # type: ignore[return-value]

    @property
    def classification_version(self) -> str:
        return str(self.manifest.get("classification_version") or "")

    @property
    def prompt_version(self) -> str:
        return str(self.manifest.get("prompt_version") or "")

    @property
    def classification_fingerprint(self) -> str:
        return str(self.manifest.get("classification_fingerprint") or "")

    @property
    def hierarchy_fingerprint(self) -> str:
        return str(self.manifest.get("hierarchy_fingerprint") or "")

    def validation_findings(self) -> list[dict[str, object]]:
        return [dict(row) for row in self.qa_findings.iter_rows(named=True)]

    def enabled_paths(self) -> pl.DataFrame:
        return _sort_paths(self.leaf_paths.filter(pl.col("enabled")))

    def rank_candidates(self, rank: str) -> pl.DataFrame:
        return self.candidate_nodes_in_paths(self.enabled_paths(), rank)

    def candidate_nodes_in_paths(self, active_paths: pl.DataFrame, rank: str) -> pl.DataFrame:
        normalized_rank = _rank(rank)
        paths = self._active_paths(active_paths)
        node_ids = _rank_node_ids(paths, normalized_rank)
        if node_ids.is_empty():
            return pl.DataFrame(schema=NODE_SCHEMA)
        return (
            self.nodes.filter((pl.col("rank") == normalized_rank) & pl.col("enabled"))
            .join(node_ids, on="node_id", how="semi")
            .sort(["scientific_name", "node_id"])
        )

    def filter_paths_by_rank_nodes(
        self,
        active_paths: pl.DataFrame,
        rank: str,
        selected_node_ids: Sequence[str],
        *,
        carry_reviewed_skip_paths: bool = False,
    ) -> pl.DataFrame:
        normalized_rank = _rank(rank)
        paths = self._active_paths(active_paths)
        selected_ids = _node_ids(selected_node_ids)
        selected_paths = (
            paths.filter(pl.col(f"{normalized_rank.casefold()}_node_id").is_in(selected_ids))
            if selected_ids
            else pl.DataFrame(schema=LEAF_PATH_SCHEMA)
        )
        if not carry_reviewed_skip_paths:
            return _deduplicate_paths(selected_paths)
        skip_paths = self.reviewed_skip_paths(paths, normalized_rank)
        if selected_paths.is_empty() and skip_paths.is_empty():
            return pl.DataFrame(schema=LEAF_PATH_SCHEMA)
        return _deduplicate_paths(pl.concat([selected_paths, skip_paths]))

    def paths_with_asserted_rank(self, active_paths: pl.DataFrame, rank: str) -> pl.DataFrame:
        normalized_rank = _rank(rank)
        paths = self._active_paths(active_paths)
        return _deduplicate_paths(
            paths.filter(pl.col(f"{normalized_rank.casefold()}_node_id") != "")
        )

    def reviewed_skip_paths(self, active_paths: pl.DataFrame, rank: str) -> pl.DataFrame:
        normalized_rank = _rank(rank)
        paths = self._active_paths(active_paths)
        if normalized_rank not in OPTIONAL_CLASSIFICATION_RANKS:
            return pl.DataFrame(schema=LEAF_PATH_SCHEMA)
        return _deduplicate_paths(
            paths.filter(
                (pl.col(f"{normalized_rank.casefold()}_node_id") == "")
                & pl.col("skipped_ranks").list.contains(normalized_rank)
                & (pl.col("path_completeness") == "reviewed_optional_skip")
            )
        )

    def species_nodes_in_paths(self, active_paths: pl.DataFrame) -> pl.DataFrame:
        return self.candidate_nodes_in_paths(active_paths, "SPECIES")

    def paths_for_species_nodes(self, species_node_ids: Sequence[str]) -> pl.DataFrame:
        ids = _node_ids(species_node_ids)
        if not ids:
            return pl.DataFrame(schema=LEAF_PATH_SCHEMA)
        return _deduplicate_paths(
            self.enabled_paths().filter(pl.col("species_node_id").is_in(ids))
        )

    def path_for_species_node(self, species_node_id: str) -> dict[str, object]:
        normalized_id = str(species_node_id or "").strip()
        if not normalized_id:
            raise ValueError("species_node_id must be nonblank")
        paths = self.paths_for_species_nodes((normalized_id,))
        if paths.is_empty():
            raise KeyError(f"classification-v3 species path not found: {normalized_id}")
        if paths.height != 1:
            raise ValueError(f"classification-v3 species has multiple enabled paths: {normalized_id}")
        return dict(paths.row(0, named=True))

    def prompt_rows_for_nodes(
        self,
        node_ids: Sequence[str],
        prompt_stage: str,
    ) -> pl.DataFrame:
        stage = str(prompt_stage or "").strip().casefold()
        if stage not in CLASSIFICATION_PROMPT_STAGES:
            raise ValueError(
                f"classification-v3 prompt stage is unavailable: {prompt_stage}; "
                f"expected one of {', '.join(CLASSIFICATION_PROMPT_STAGES)}"
            )
        ids = _node_ids(node_ids)
        if not ids:
            return pl.DataFrame(schema=PROMPT_LABEL_SCHEMA)
        selected_nodes = self.nodes.filter(pl.col("enabled") & pl.col("node_id").is_in(ids))
        if set(selected_nodes["node_id"].to_list()) != set(ids):
            raise ValueError("classification-v3 prompts require enabled classification node IDs")
        ranks = set(selected_nodes["rank"].to_list())
        if stage == RANK_SCREEN_PROMPT_STAGE:
            valid = "SPECIES" not in ranks
            required = "enabled intermediate-rank"
        else:
            valid = ranks == {"SPECIES"}
            required = "enabled SPECIES"
        if not valid:
            raise ValueError(f"{stage} prompts require only {required} node IDs")
        return _sort_prompts(
            self.prompt_labels.filter(
                pl.col("enabled")
                & (pl.col("prompt_stage") == stage)
                & pl.col("node_id").is_in(ids)
            )
        )

    def mappings_for_species_nodes(self, node_ids: Sequence[str]) -> pl.DataFrame:
        ids = _node_ids(node_ids)
        if not ids:
            return pl.DataFrame(schema=GBIF_MAPPING_SCHEMA)
        return self.gbif_mappings.filter(
            pl.col("enabled") & pl.col("species_node_id").is_in(ids)
        ).sort(["accepted_scientific_name", "gbif_species_key", "species_node_id"])

    def _classification_frames(self) -> ClassificationV3Frames:
        return ClassificationV3Frames(
            sources=self.sources,
            nodes=self.nodes,
            edges=self.edges,
            gbif_mappings=self.gbif_mappings,
            leaf_paths=self.leaf_paths,
            prompt_labels=self.prompt_labels,
        )

    def _active_paths(self, active_paths: pl.DataFrame) -> pl.DataFrame:
        paths = _normalize_active_paths(active_paths)
        foreign_hashes = sorted(
            {
                str(value)
                for value in paths["hierarchy_hash"].to_list()
                if str(value) not in self._enabled_hierarchy_hashes
            }
        )
        if foreign_hashes:
            raise ValueError(
                "active classification paths do not belong to this taxonomy store: "
                + ", ".join(foreign_hashes)
            )
        return paths

    def _validate_loaded_frames(self) -> None:
        versioned_frames = {
            "sources": self.sources,
            "nodes": self.nodes,
            "edges": self.edges,
            "gbif_mappings": self.gbif_mappings,
            "leaf_paths": self.leaf_paths,
            "prompt_labels": self.prompt_labels,
        }
        for name, frame in versioned_frames.items():
            versions = set(frame["classification_version"].to_list())
            if versions and versions != {CLASSIFICATION_V3_VERSION}:
                raise ValueError(f"classification-v3 frame version mismatch: {name}")
        prompt_versions = set(self.prompt_labels["prompt_version"].to_list())
        if prompt_versions and prompt_versions != {CLASSIFICATION_V3_PROMPT_VERSION}:
            raise ValueError("classification-v3 prompt rows do not match the manifest prompt version")
        fatal_count = self.qa_findings.filter(pl.col("severity") == "fatal").height
        warning_count = self.qa_findings.filter(pl.col("severity") == "warning").height
        if fatal_count != int(self.manifest.get("fatal_finding_count") or 0):
            raise ValueError("classification-v3 fatal QA count mismatch")
        if warning_count != int(self.manifest.get("warning_finding_count") or 0):
            raise ValueError("classification-v3 warning QA count mismatch")
        frames = self._classification_frames()
        expected_hierarchy = self.hierarchy_fingerprint
        if not expected_hierarchy or hierarchy_fingerprint(frames) != expected_hierarchy:
            raise ValueError("classification-v3 hierarchy fingerprint mismatch")
        expected_classification = self.classification_fingerprint
        if not expected_classification or classification_v3_fingerprint(frames) != expected_classification:
            raise ValueError("classification-v3 classification fingerprint mismatch")


def _read_unified_registry_store(path: Path) -> PathTaxonomyStore:
    """Adapt the unified one-row-per-species path artifact for cascade callers."""

    return UnifiedPathTaxonomyStore(pl.read_parquet(path))  # type: ignore[return-value]


class UnifiedPathTaxonomyStore:
    """Taxonomy-store interface backed directly by seven-rank species paths."""

    def __init__(self, species_paths: pl.DataFrame) -> None:
        self.leaf_paths = species_paths
        self.rank_order = BIOCLIP_SUPPORTED_RANKS
        self._enabled_hashes = frozenset(
            str(value) for value in species_paths.filter(pl.col("enabled"))["path_fingerprint"].to_list()
        )
        nodes: dict[str, dict[str, object]] = {}
        prompts: list[dict[str, object]] = []
        mappings: list[dict[str, object]] = []
        for path_row in species_paths.iter_rows(named=True):
            row = dict(path_row)
            for rank in self.rank_order:
                prefix = rank.casefold()
                node_id = str(row.get(f"{prefix}_node_id") or "")
                if not node_id:
                    continue
                name = str(row.get(prefix) or "")
                kind = str(row.get(f"{prefix}_candidate_kind") or "observed_taxon")
                semantic_rank = str(row.get(f"{prefix}_semantic_rank") or rank)
                nodes.setdefault(
                    node_id,
                    {
                        "node_id": node_id,
                        "rank": rank,
                        "scientific_name": name,
                        "candidate_kind": kind,
                        "semantic_rank": semantic_rank,
                        "proxy_source_node_id": str(row.get(f"{prefix}_proxy_source_node_id") or ""),
                        "enabled": bool(row.get("enabled", True)),
                    },
                )
            species_id = str(row.get("species_node_id") or "")
            mappings.append(
                {
                    "accepted_taxon_key": str(row.get("accepted_taxon_key") or ""),
                    "gbif_species_key": str(row.get("accepted_taxon_key") or "").removeprefix("gbif:"),
                    "accepted_scientific_name": str(row.get("accepted_scientific_name") or row.get("species") or ""),
                    "species_node_id": species_id,
                    "taxonomic_status": "ACCEPTED",
                    "enabled": bool(row.get("enabled", True)),
                }
            )
        for node in nodes.values():
            stages = (RANK_SCREEN_PROMPT_STAGE,) if node["rank"] != "SPECIES" else (
                SPECIES_FIRST_PASS_PROMPT_STAGE,
                SPECIES_RERANK_PROMPT_STAGE,
            )
            for stage in stages:
                if stage == SPECIES_FIRST_PASS_PROMPT_STAGE:
                    label = f"a field photograph of the butterfly species {node['scientific_name']}"
                elif stage == SPECIES_RERANK_PROMPT_STAGE:
                    label = f"a detailed close-up photograph of the butterfly species {node['scientific_name']}"
                else:
                    label = f"a butterfly in the {str(node['semantic_rank']).casefold()} {node['scientific_name']}"
                if node["candidate_kind"] == "carry_forward_proxy":
                    label += "; routing proxy for a missing hierarchy rank"
                prompts.append(
                    {
                        "prompt_stage": stage,
                        "node_id": node["node_id"],
                        "rank": node["rank"],
                        "scientific_name": node["scientific_name"],
                        "label": label,
                        "prompt_template": "semantic_rank_routing_proxy" if node["candidate_kind"] == "carry_forward_proxy" else "observed_taxon",
                        "sort_order": 1,
                        "enabled": node["enabled"],
                    }
                )
        self.nodes = pl.DataFrame(list(nodes.values())).sort(["rank", "scientific_name", "node_id"])
        self.prompt_labels = pl.DataFrame(prompts).sort(["prompt_stage", "rank", "scientific_name", "node_id"])
        self.gbif_mappings = pl.DataFrame(mappings).sort(["accepted_scientific_name", "species_node_id"])
        fingerprint = stable_identity(
            "unified-taxonomy-store",
            *sorted(str(value) for value in species_paths["path_fingerprint"].to_list()),
        )
        self.manifest = MappingProxyType(
            {
                "classification_version": "butterfly-col-xr-seven-rank-v1.0.0",
                "prompt_version": "bioclip-supported-seven-rank-prompts-v1",
                "hierarchy_fingerprint": fingerprint,
                "classification_fingerprint": stable_identity("unified-prompts", fingerprint),
                "qa_status": "passed",
                "fatal_finding_count": 0,
                "warning_finding_count": 0,
                "rank_order": list(self.rank_order),
            }
        )
        self.qa_findings = pl.DataFrame(schema=QA_FINDING_SCHEMA)

    @property
    def classification_version(self) -> str:
        return str(self.manifest["classification_version"])

    @property
    def prompt_version(self) -> str:
        return str(self.manifest["prompt_version"])

    @property
    def hierarchy_fingerprint(self) -> str:
        return str(self.manifest["hierarchy_fingerprint"])

    @property
    def classification_fingerprint(self) -> str:
        return str(self.manifest["classification_fingerprint"])

    def validation_findings(self) -> list[dict[str, object]]:
        return []

    def enabled_paths(self) -> pl.DataFrame:
        return self.leaf_paths.filter(pl.col("enabled")).sort(["accepted_scientific_name", "accepted_taxon_key"])

    def rank_candidates(self, rank: str) -> pl.DataFrame:
        return self.candidate_nodes_in_paths(self.enabled_paths(), rank)

    def candidate_nodes_in_paths(self, active_paths: pl.DataFrame, rank: str) -> pl.DataFrame:
        normalized = str(rank).upper()
        if normalized not in self.rank_order:
            raise ValueError(f"unsupported BioCLIP rank: {rank}")
        ids = active_paths.select(pl.col(f"{normalized.casefold()}_node_id").alias("node_id")).unique()
        return self.nodes.join(ids, on="node_id", how="semi").sort(["scientific_name", "node_id"])

    def filter_paths_by_rank_nodes(
        self,
        active_paths: pl.DataFrame,
        rank: str,
        selected_node_ids: Sequence[str],
        *,
        carry_reviewed_skip_paths: bool = False,
    ) -> pl.DataFrame:
        del carry_reviewed_skip_paths
        return active_paths.filter(pl.col(f"{str(rank).casefold()}_node_id").is_in(_node_ids(selected_node_ids)))

    def paths_with_asserted_rank(self, active_paths: pl.DataFrame, rank: str) -> pl.DataFrame:
        return active_paths.filter(pl.col(f"{str(rank).casefold()}_node_id") != "")

    def reviewed_skip_paths(self, active_paths: pl.DataFrame, rank: str) -> pl.DataFrame:
        del rank
        return active_paths.head(0)

    def species_nodes_in_paths(self, active_paths: pl.DataFrame) -> pl.DataFrame:
        return self.candidate_nodes_in_paths(active_paths, "SPECIES")

    def paths_for_species_nodes(self, species_node_ids: Sequence[str]) -> pl.DataFrame:
        return self.enabled_paths().filter(pl.col("species_node_id").is_in(_node_ids(species_node_ids)))

    def path_for_species_node(self, species_node_id: str) -> dict[str, object]:
        frame = self.paths_for_species_nodes((species_node_id,))
        if frame.height != 1:
            raise KeyError(f"unified registry species path not singular: {species_node_id}")
        return dict(frame.row(0, named=True))

    def prompt_rows_for_nodes(self, node_ids: Sequence[str], prompt_stage: str) -> pl.DataFrame:
        return self.prompt_labels.filter(
            (pl.col("prompt_stage") == str(prompt_stage).casefold()) & pl.col("node_id").is_in(_node_ids(node_ids))
        ).sort(["scientific_name", "node_id", "sort_order"])

    def mappings_for_species_nodes(self, node_ids: Sequence[str]) -> pl.DataFrame:
        return self.gbif_mappings.filter(pl.col("species_node_id").is_in(_node_ids(node_ids)))


def _store_from_unified_paths(paths: pl.DataFrame) -> PathTaxonomyStore:
    required = {
        "accepted_taxon_key",
        "accepted_scientific_name",
        "path_fingerprint",
        "enabled",
        *(f"{rank.casefold()}_node_id" for rank in CLASSIFICATION_RANKS),
        *(rank.casefold() for rank in CLASSIFICATION_RANKS),
    }
    missing = sorted(required - set(paths.columns))
    if missing:
        raise ValueError("species_paths.parquet is missing columns: " + ", ".join(missing))

    source_release = str(paths["source_release"][0]) if paths.height and "source_release" in paths.columns else ""
    sources = pl.DataFrame(
        [
            {
                "classification_version": CLASSIFICATION_V3_VERSION,
                "source_id": "unified-registry",
                "authority": "Catalogue of Life XR",
                "release": source_release,
                "citation": "Catalogue of Life XR 315557; DOI 10.48580/dgy8b",
                "retrieved_at": "",
                "evidence_url": "https://www.catalogueoflife.org/",
                "evidence": "Unified registry species path",
            }
        ],
        schema=SOURCE_SCHEMA,
    )
    node_rows: dict[str, dict[str, object]] = {}
    edge_rows: dict[tuple[str, str], dict[str, object]] = {}
    mapping_rows: list[dict[str, object]] = []
    leaf_rows: list[dict[str, object]] = []
    prompt_rows: list[dict[str, object]] = []
    proxy_semantics: dict[str, tuple[str, str, str]] = {}
    for source_path in paths.iter_rows(named=True):
        row = dict(source_path)
        rank_node_ids: list[str] = []
        for rank in CLASSIFICATION_RANKS:
            prefix = rank.casefold()
            node_id = str(row.get(f"{prefix}_node_id") or "")
            name = str(row.get(prefix) or "")
            semantic_rank = str(row.get(f"{prefix}_semantic_rank") or rank)
            candidate_kind = str(row.get(f"{prefix}_candidate_kind") or "observed_taxon")
            if not node_id:
                continue
            rank_node_ids.append(node_id)
            node_rows.setdefault(
                node_id,
                {
                    "classification_version": CLASSIFICATION_V3_VERSION,
                    "node_id": node_id,
                    "rank": rank,
                    "scientific_name": name,
                    "source_id": "unified-registry",
                    "source_release": source_release,
                    "citation": "Catalogue of Life XR 315557; DOI 10.48580/dgy8b",
                    "retrieved_at": "",
                    "evidence": "routing proxy" if candidate_kind == "carry_forward_proxy" else "observed registry taxon",
                    "reviewed": True,
                    "review_status": "source_accepted",
                    "reviewed_by": "Catalogue of Life XR",
                    "reviewed_at": "",
                    "enabled": bool(row.get("enabled", True)),
                    "disabled_reason": str(row.get("disabled_reason") or ""),
                },
            )
            proxy_semantics[node_id] = (candidate_kind, semantic_rank, name)
        for parent_rank, child_rank in zip(CLASSIFICATION_RANKS[:-1], CLASSIFICATION_RANKS[1:], strict=True):
            parent_id = str(row.get(f"{parent_rank.casefold()}_node_id") or "")
            child_id = str(row.get(f"{child_rank.casefold()}_node_id") or "")
            if not parent_id or not child_id:
                continue
            edge_rows.setdefault(
                (parent_id, child_id),
                {
                    "classification_version": CLASSIFICATION_V3_VERSION,
                    "parent_node_id": parent_id,
                    "child_node_id": child_id,
                    "parent_rank": parent_rank,
                    "child_rank": child_rank,
                    "edge_type": "asserted_parent",
                    "skipped_ranks": [],
                    "skip_reason": "",
                    "source_id": "unified-registry",
                    "source_release": source_release,
                    "citation": "Catalogue of Life XR 315557; DOI 10.48580/dgy8b",
                    "retrieved_at": "",
                    "evidence": "species_paths routing edge",
                    "reviewed": True,
                    "review_status": "source_accepted",
                    "reviewed_by": "Catalogue of Life XR",
                    "reviewed_at": "",
                    "enabled": bool(row.get("enabled", True)),
                    "disabled_reason": str(row.get("disabled_reason") or ""),
                },
            )
        species_node_id = str(row.get("species_node_id") or "")
        accepted_key = str(row.get("accepted_taxon_key") or "")
        accepted_name = str(row.get("accepted_scientific_name") or row.get("species") or "")
        mapping_rows.append(
            {
                "classification_version": CLASSIFICATION_V3_VERSION,
                "accepted_taxon_key": accepted_key,
                "gbif_species_key": accepted_key.removeprefix("gbif:"),
                "accepted_scientific_name": accepted_name,
                "species_node_id": species_node_id,
                "taxonomic_status": "ACCEPTED",
                "source_id": "unified-registry",
                "source_release": source_release,
                "citation": "Catalogue of Life XR 315557; DOI 10.48580/dgy8b",
                "retrieved_at": "",
                "evidence": "unified registry identity",
                "reviewed": True,
                "review_status": "source_accepted",
                "reviewed_by": "Catalogue of Life XR",
                "reviewed_at": "",
                "enabled": bool(row.get("enabled", True)),
                "disabled_reason": str(row.get("disabled_reason") or ""),
            }
        )
        leaf: dict[str, object] = {
            "classification_version": CLASSIFICATION_V3_VERSION,
            "accepted_taxon_key": accepted_key,
            "gbif_species_key": accepted_key.removeprefix("gbif:"),
            "rank_path": list(CLASSIFICATION_RANKS),
            "rank_path_node_ids": rank_node_ids,
            "skipped_ranks": [],
            "path_completeness": "complete_with_routing_proxies" if any(
                str(row.get(f"{rank.casefold()}_candidate_kind") or "") == "carry_forward_proxy"
                for rank in CLASSIFICATION_RANKS
            ) else "complete",
            "hierarchy_hash": str(row.get("path_fingerprint") or ""),
            "source_release": source_release,
            "enabled": bool(row.get("enabled", True)),
            "disabled_reason": str(row.get("disabled_reason") or ""),
        }
        for rank in CLASSIFICATION_RANKS:
            prefix = rank.casefold()
            leaf[f"{prefix}_node_id"] = str(row.get(f"{prefix}_node_id") or "")
            leaf[prefix] = str(row.get(prefix) or "")
        leaf_rows.append(leaf)

    nodes = pl.DataFrame(list(node_rows.values()), schema=NODE_SCHEMA).sort(["rank", "scientific_name", "node_id"])
    edges = pl.DataFrame(list(edge_rows.values()), schema=EDGE_SCHEMA).sort(["parent_rank", "parent_node_id", "child_node_id"])
    mappings = pl.DataFrame(mapping_rows, schema=GBIF_MAPPING_SCHEMA).sort(["accepted_scientific_name", "gbif_species_key"])
    leaf_paths = pl.DataFrame(leaf_rows, schema=LEAF_PATH_SCHEMA).sort(["family", "genus", "species", "accepted_taxon_key"])
    for node in nodes.iter_rows(named=True):
        node_id = str(node["node_id"])
        rank = str(node["rank"])
        name = str(node["scientific_name"])
        candidate_kind, semantic_rank, _ = proxy_semantics[node_id]
        stages = (RANK_SCREEN_PROMPT_STAGE,) if rank != "SPECIES" else (
            SPECIES_FIRST_PASS_PROMPT_STAGE,
            SPECIES_RERANK_PROMPT_STAGE,
        )
        for stage in stages:
            semantic_label = f"a butterfly in the {semantic_rank.casefold()} {name}"
            if candidate_kind == "carry_forward_proxy":
                semantic_label += "; routing proxy for a missing hierarchy rank"
            prompt_rows.append(
                {
                    "classification_version": CLASSIFICATION_V3_VERSION,
                    "prompt_version": CLASSIFICATION_V3_PROMPT_VERSION,
                    "prompt_stage": stage,
                    "node_id": node_id,
                    "rank": rank,
                    "scientific_name": name,
                    "label": semantic_label,
                    "prompt_template": "semantic_rank_routing_proxy" if candidate_kind == "carry_forward_proxy" else "observed_taxon",
                    "sort_order": 1,
                    "enabled": bool(node["enabled"]),
                }
            )
    prompts = pl.DataFrame(prompt_rows, schema=PROMPT_LABEL_SCHEMA).sort(["prompt_stage", "rank", "scientific_name", "node_id"])
    qa = pl.DataFrame(schema=QA_FINDING_SCHEMA)
    frames = ClassificationV3Frames(
        sources=sources,
        nodes=nodes,
        edges=edges,
        gbif_mappings=mappings,
        leaf_paths=leaf_paths,
        prompt_labels=prompts,
    )
    manifest = {
        "classification_version": CLASSIFICATION_V3_VERSION,
        "prompt_version": CLASSIFICATION_V3_PROMPT_VERSION,
        "qa_status": "passed",
        "fatal_finding_count": 0,
        "warning_finding_count": 0,
        "hierarchy_fingerprint": hierarchy_fingerprint(frames),
        "classification_fingerprint": classification_v3_fingerprint(frames),
        "taxonomy_input": "species_paths.parquet",
    }
    return PathTaxonomyStore.from_frames(
        sources=sources,
        nodes=nodes,
        edges=edges,
        gbif_mappings=mappings,
        leaf_paths=leaf_paths,
        prompt_labels=prompts,
        qa_findings=qa,
        manifest=manifest,
    )


def _read_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("classification-v3 manifest must be a JSON object")
    return payload


def _validate_manifest_header(manifest: Mapping[str, object]) -> None:
    if manifest.get("classification_version") != CLASSIFICATION_V3_VERSION:
        raise ValueError("classification-v3 manifest version mismatch")
    if manifest.get("prompt_version") != CLASSIFICATION_V3_PROMPT_VERSION:
        raise ValueError("classification-v3 manifest prompt version mismatch")
    if int(manifest.get("fatal_finding_count") or 0) > 0 or manifest.get("qa_status") != "passed":
        raise ValueError("classification-v3 manifest did not pass fatal QA")


def _validate_artifact_checksums(paths: dict[str, Path], manifest: Mapping[str, object]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("classification-v3 manifest has no artifact checksums")
    for key in _CHECKSUM_ARTIFACTS:
        metadata = artifacts.get(key)
        if not isinstance(metadata, dict):
            raise ValueError(f"classification-v3 manifest is missing checksum for {key}")
        if metadata.get("file") != paths[key].name:
            raise ValueError(f"classification-v3 artifact filename mismatch: {key}")
        expected_bytes = metadata.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes != paths[key].stat().st_size:
            raise ValueError(f"classification-v3 artifact byte count mismatch: {key}")
        if str(metadata.get("sha256") or "") != _sha256_file(paths[key]):
            raise ValueError(f"classification-v3 artifact checksum mismatch: {key}")


def _scan(path: Path, schema: dict[str, pl.DataType], *, artifact: str) -> pl.DataFrame:
    lazy = pl.scan_parquet(path)
    if dict(lazy.collect_schema()) != schema:
        raise ValueError(f"classification-v3 artifact schema mismatch: {artifact}")
    return lazy.select(list(schema)).collect(engine="streaming")


def _require_schema(
    frame: pl.DataFrame,
    schema: dict[str, pl.DataType],
    *,
    artifact: str,
) -> pl.DataFrame:
    if frame.columns != list(schema) or dict(frame.schema) != schema:
        raise ValueError(f"classification-v3 artifact schema mismatch: {artifact}")
    return frame.select(list(schema))


def _rank(value: str) -> str:
    normalized = str(value or "").strip().upper()
    if normalized not in CLASSIFICATION_RANKS:
        raise ValueError(f"unsupported classification rank: {value}")
    return normalized


def _node_ids(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(value).strip()
            for value in values
            if value is not None and str(value).strip()
        )
    )


def _rank_node_ids(paths: pl.DataFrame, rank: str) -> pl.DataFrame:
    column = f"{rank.casefold()}_node_id"
    return (
        paths.select(pl.col(column).alias("node_id"))
        .filter(pl.col("node_id") != "")
        .unique()
        .sort("node_id")
    )


def _normalize_active_paths(paths: pl.DataFrame) -> pl.DataFrame:
    missing = [column for column in LEAF_PATH_SCHEMA if column not in paths.columns]
    if missing:
        raise ValueError("active classification paths are missing columns: " + ", ".join(missing))
    selected = paths.select(list(LEAF_PATH_SCHEMA))
    versions = set(selected["classification_version"].to_list())
    if versions and versions != {CLASSIFICATION_V3_VERSION}:
        raise ValueError("active classification paths have a version mismatch")
    if selected["enabled"].null_count() or not selected["enabled"].all():
        raise ValueError("active classification paths contain disabled rows")
    return _sort_paths(selected)


def _sort_paths(paths: pl.DataFrame) -> pl.DataFrame:
    return paths.sort(list(_PATH_SORT_COLUMNS))


def _deduplicate_paths(paths: pl.DataFrame) -> pl.DataFrame:
    if paths.is_empty():
        return pl.DataFrame(schema=LEAF_PATH_SCHEMA)
    return (
        _sort_paths(paths)
        .unique(subset=["hierarchy_hash"], keep="first", maintain_order=True)
        .sort(list(_PATH_SORT_COLUMNS))
    )


def _sort_prompts(prompts: pl.DataFrame) -> pl.DataFrame:
    rank_order = {rank: index for index, rank in enumerate(CLASSIFICATION_RANKS)}
    stage_order = {stage: index for index, stage in enumerate(CLASSIFICATION_PROMPT_STAGES)}
    rows = sorted(
        prompts.iter_rows(named=True),
        key=lambda row: (
            stage_order.get(str(row.get("prompt_stage") or ""), len(stage_order)),
            rank_order.get(str(row.get("rank") or ""), len(rank_order)),
            str(row.get("scientific_name") or ""),
            str(row.get("node_id") or ""),
            int(row.get("sort_order") or 0),
            str(row.get("label") or ""),
        ),
    )
    return pl.DataFrame(rows, schema=PROMPT_LABEL_SCHEMA) if rows else pl.DataFrame(schema=PROMPT_LABEL_SCHEMA)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "PathTaxonomyStore",
    "RANK_SCREEN_PROMPT_STAGE",
    "SPECIES_FIRST_PASS_PROMPT_STAGE",
    "SPECIES_RERANK_PROMPT_STAGE",
]
