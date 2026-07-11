from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import polars as pl

from biominer.registry.classification_v2 import (
    CLASSIFICATION_RANKS,
    CLASSIFICATION_V2_VERSION,
    EDGE_SCHEMA,
    GBIF_MAPPING_SCHEMA,
    LEAF_PATH_SCHEMA,
    NODE_SCHEMA,
    PROMPT_LABEL_SCHEMA,
    SOURCE_SCHEMA,
    ClassificationV2Frames,
    classification_v2_artifact_paths,
    classification_v2_fingerprint,
)


@dataclass(frozen=True)
class FiveRankTaxonomyStore:
    sources: pl.DataFrame
    nodes: pl.DataFrame
    edges: pl.DataFrame
    gbif_mappings: pl.DataFrame
    leaf_paths: pl.DataFrame
    prompt_labels: pl.DataFrame
    manifest: dict[str, object]

    @classmethod
    def read(
        cls,
        root: str | Path,
        *,
        ranks: Sequence[str] = CLASSIFICATION_RANKS,
    ) -> FiveRankTaxonomyStore:
        requested_ranks = tuple(dict.fromkeys(str(rank).upper() for rank in ranks))
        invalid = [rank for rank in requested_ranks if rank not in CLASSIFICATION_RANKS]
        if invalid:
            raise ValueError("unsupported classification ranks: " + ", ".join(invalid))
        paths = classification_v2_artifact_paths(root)
        required = ("sources", "nodes", "edges", "gbif_mappings", "leaf_paths", "prompt_labels", "manifest")
        missing = [str(paths[key]) for key in required if not paths[key].exists()]
        if missing:
            raise FileNotFoundError("missing classification-v2 artifacts: " + ", ".join(missing))
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest.get("classification_version") != CLASSIFICATION_V2_VERSION:
            raise ValueError("classification-v2 manifest version mismatch")
        if int(manifest.get("fatal_finding_count") or 0) > 0 or manifest.get("qa_status") != "passed":
            raise ValueError("classification-v2 manifest did not pass fatal QA")
        _validate_artifact_checksums(paths, manifest)
        nodes = _scan(paths["nodes"], NODE_SCHEMA).filter(pl.col("rank").is_in(requested_ranks))
        edges = _scan(paths["edges"], EDGE_SCHEMA).filter(
            pl.col("parent_rank").is_in(requested_ranks) & pl.col("child_rank").is_in(requested_ranks)
        )
        prompts = _scan(paths["prompt_labels"], PROMPT_LABEL_SCHEMA).filter(pl.col("rank").is_in(requested_ranks))
        store = cls(
            sources=_scan(paths["sources"], SOURCE_SCHEMA),
            nodes=nodes,
            edges=edges,
            gbif_mappings=_scan(paths["gbif_mappings"], GBIF_MAPPING_SCHEMA),
            leaf_paths=_scan(paths["leaf_paths"], LEAF_PATH_SCHEMA),
            prompt_labels=prompts,
            manifest=manifest,
        )
        if requested_ranks == CLASSIFICATION_RANKS:
            expected = str(manifest.get("classification_fingerprint") or "")
            if expected and store.taxonomy_fingerprint != expected:
                raise ValueError("classification-v2 taxonomy fingerprint mismatch")
        return store

    @property
    def taxonomy_fingerprint(self) -> str:
        return classification_v2_fingerprint(
            ClassificationV2Frames(
                sources=self.sources,
                nodes=self.nodes,
                edges=self.edges,
                gbif_mappings=self.gbif_mappings,
                leaf_paths=self.leaf_paths,
                prompt_labels=self.prompt_labels,
            )
        )

    @property
    def classification_version(self) -> str:
        return str(self.manifest.get("classification_version") or "")

    @property
    def prompt_version(self) -> str:
        return str(self.manifest.get("prompt_version") or "")

    def candidates(self, rank: str) -> pl.DataFrame:
        normalized = _rank(rank)
        return self.nodes.filter((pl.col("rank") == normalized) & pl.col("enabled")).sort(
            ["scientific_name", "node_id"]
        )

    def child_candidates(self, parent_node_ids: Sequence[str], *, child_rank: str) -> pl.DataFrame:
        parent_ids = tuple(dict.fromkeys(str(node_id) for node_id in parent_node_ids if str(node_id)))
        normalized_rank = _rank(child_rank)
        if not parent_ids:
            return pl.DataFrame(schema=NODE_SCHEMA)
        child_ids = (
            self.edges.filter(
                pl.col("enabled")
                & pl.col("parent_node_id").is_in(parent_ids)
                & (pl.col("child_rank") == normalized_rank)
            )["child_node_id"]
            .unique()
            .to_list()
        )
        return self.nodes.filter(pl.col("enabled") & pl.col("node_id").is_in(child_ids)).sort(
            ["scientific_name", "node_id"]
        )

    def prompt_rows_for_nodes(self, node_ids: Sequence[str]) -> pl.DataFrame:
        ids = tuple(dict.fromkeys(str(node_id) for node_id in node_ids if str(node_id)))
        if not ids:
            return pl.DataFrame(schema=PROMPT_LABEL_SCHEMA)
        return self.prompt_labels.filter(pl.col("enabled") & pl.col("node_id").is_in(ids)).sort(
            ["rank", "scientific_name", "sort_order", "label"]
        )

    def prompt_rows_for_rank(self, rank: str) -> pl.DataFrame:
        normalized = _rank(rank)
        return self.prompt_labels.filter((pl.col("rank") == normalized) & pl.col("enabled")).sort(
            ["scientific_name", "sort_order", "label"]
        )

    def leaf_paths_for_nodes(self, rank: str, node_ids: Sequence[str]) -> pl.DataFrame:
        normalized = _rank(rank)
        ids = tuple(dict.fromkeys(str(node_id) for node_id in node_ids if str(node_id)))
        if not ids:
            return pl.DataFrame(schema=LEAF_PATH_SCHEMA)
        return self.leaf_paths.filter(
            pl.col("enabled") & pl.col(f"{normalized.casefold()}_node_id").is_in(ids)
        ).sort(["family", "subfamily", "tribe", "genus", "species"])

    def species_candidates_for_genera(self, genus_node_ids: Sequence[str]) -> pl.DataFrame:
        paths = self.leaf_paths_for_nodes("GENUS", genus_node_ids)
        if paths.is_empty():
            return pl.DataFrame(schema=NODE_SCHEMA)
        return self.nodes.filter(
            pl.col("enabled") & pl.col("node_id").is_in(paths["species_node_id"].unique().to_list())
        ).sort(["scientific_name", "node_id"])

    def gbif_mapping_for_species_nodes(self, species_node_ids: Sequence[str]) -> pl.DataFrame:
        ids = tuple(dict.fromkeys(str(node_id) for node_id in species_node_ids if str(node_id)))
        if not ids:
            return pl.DataFrame(schema=GBIF_MAPPING_SCHEMA)
        return self.gbif_mappings.filter(
            pl.col("enabled") & pl.col("species_node_id").is_in(ids)
        ).sort(["accepted_scientific_name", "gbif_species_key"])


def _scan(path: Path, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.scan_parquet(path).select(list(schema)).collect(engine="streaming")


def _rank(value: str) -> str:
    normalized = str(value or "").strip().upper()
    if normalized not in CLASSIFICATION_RANKS:
        raise ValueError(f"unsupported classification rank: {value}")
    return normalized


def _validate_artifact_checksums(paths: dict[str, Path], manifest: dict[str, object]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("classification-v2 manifest has no artifact checksums")
    for key in ("sources", "nodes", "edges", "gbif_mappings", "leaf_paths", "prompt_labels"):
        metadata = artifacts.get(key)
        if not isinstance(metadata, dict):
            raise ValueError(f"classification-v2 manifest is missing checksum for {key}")
        expected = str(metadata.get("sha256") or "")
        actual = _sha256_file(paths[key])
        if expected != actual:
            raise ValueError(f"classification-v2 artifact checksum mismatch: {key}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = ["FiveRankTaxonomyStore"]
