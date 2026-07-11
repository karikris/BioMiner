from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Self, Sequence

import polars as pl

from biominer.registry.classification_v3 import (
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
    SOURCE_SCHEMA,
    ClassificationV3Frames,
    classification_v3_artifact_paths,
    classification_v3_fingerprint,
    hierarchy_fingerprint,
)


RANK_SCREEN_PROMPT_STAGE = "rank_screen"
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
    manifest: dict[str, object]

    @classmethod
    def read(cls, root: str | Path) -> Self:
        paths = classification_v3_artifact_paths(root)
        required = (*_CHECKSUM_ARTIFACTS, "manifest")
        missing = [str(paths[key]) for key in required if not paths[key].exists()]
        if missing:
            raise FileNotFoundError("missing classification-v3 artifacts: " + ", ".join(missing))
        manifest = _read_manifest(paths["manifest"])
        _validate_manifest_header(manifest)
        _validate_artifact_checksums(paths, manifest)
        store = cls(
            sources=_scan(paths["sources"], SOURCE_SCHEMA),
            nodes=_scan(paths["nodes"], NODE_SCHEMA),
            edges=_scan(paths["edges"], EDGE_SCHEMA),
            gbif_mappings=_scan(paths["gbif_mappings"], GBIF_MAPPING_SCHEMA),
            leaf_paths=_scan(paths["leaf_paths"], LEAF_PATH_SCHEMA),
            prompt_labels=_scan(paths["prompt_labels"], PROMPT_LABEL_SCHEMA),
            qa_findings=_scan(paths["qa_findings"], QA_FINDING_SCHEMA),
            manifest=manifest,
        )
        store._validate_loaded_frames()
        return store

    @property
    def classification_version(self) -> str:
        return str(self.manifest.get("classification_version") or "")

    @property
    def prompt_version(self) -> str:
        return str(self.manifest.get("prompt_version") or "")

    @property
    def classification_fingerprint(self) -> str:
        return classification_v3_fingerprint(self._classification_frames())

    @property
    def hierarchy_fingerprint(self) -> str:
        return hierarchy_fingerprint(self._classification_frames())

    def validation_findings(self) -> list[dict[str, object]]:
        return [dict(row) for row in self.qa_findings.iter_rows(named=True)]

    def enabled_paths(self) -> pl.DataFrame:
        return _sort_paths(self.leaf_paths.filter(pl.col("enabled")))

    def rank_candidates(self, rank: str) -> pl.DataFrame:
        return self.candidate_nodes_in_paths(self.enabled_paths(), rank)

    def candidate_nodes_in_paths(self, active_paths: pl.DataFrame, rank: str) -> pl.DataFrame:
        normalized_rank = _rank(rank)
        paths = _active_paths(active_paths)
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
        paths = _active_paths(active_paths)
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
        paths = _active_paths(active_paths)
        return _deduplicate_paths(
            paths.filter(pl.col(f"{normalized_rank.casefold()}_node_id") != "")
        )

    def reviewed_skip_paths(self, active_paths: pl.DataFrame, rank: str) -> pl.DataFrame:
        normalized_rank = _rank(rank)
        paths = _active_paths(active_paths)
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
        if stage != RANK_SCREEN_PROMPT_STAGE:
            raise ValueError(
                f"classification-v3 prompt stage is unavailable: {prompt_stage}; "
                f"expected {RANK_SCREEN_PROMPT_STAGE}"
            )
        ids = _node_ids(node_ids)
        if not ids:
            return pl.DataFrame(schema=PROMPT_LABEL_SCHEMA)
        return _sort_prompts(
            self.prompt_labels.filter(pl.col("enabled") & pl.col("node_id").is_in(ids))
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
        expected_hierarchy = str(self.manifest.get("hierarchy_fingerprint") or "")
        if not expected_hierarchy or self.hierarchy_fingerprint != expected_hierarchy:
            raise ValueError("classification-v3 hierarchy fingerprint mismatch")
        expected_classification = str(self.manifest.get("classification_fingerprint") or "")
        if not expected_classification or self.classification_fingerprint != expected_classification:
            raise ValueError("classification-v3 classification fingerprint mismatch")


def _read_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("classification-v3 manifest must be a JSON object")
    return payload


def _validate_manifest_header(manifest: dict[str, object]) -> None:
    if manifest.get("classification_version") != CLASSIFICATION_V3_VERSION:
        raise ValueError("classification-v3 manifest version mismatch")
    if manifest.get("prompt_version") != CLASSIFICATION_V3_PROMPT_VERSION:
        raise ValueError("classification-v3 manifest prompt version mismatch")
    if int(manifest.get("fatal_finding_count") or 0) > 0 or manifest.get("qa_status") != "passed":
        raise ValueError("classification-v3 manifest did not pass fatal QA")


def _validate_artifact_checksums(paths: dict[str, Path], manifest: dict[str, object]) -> None:
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


def _scan(path: Path, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.scan_parquet(path).select(list(schema)).collect(engine="streaming")


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


def _active_paths(paths: pl.DataFrame) -> pl.DataFrame:
    missing = [column for column in LEAF_PATH_SCHEMA if column not in paths.columns]
    if missing:
        raise ValueError("active classification paths are missing columns: " + ", ".join(missing))
    selected = paths.select(list(LEAF_PATH_SCHEMA))
    versions = set(selected["classification_version"].to_list())
    if versions and versions != {CLASSIFICATION_V3_VERSION}:
        raise ValueError("active classification paths have a version mismatch")
    if selected.filter(~pl.col("enabled")).height:
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
    rows = sorted(
        prompts.iter_rows(named=True),
        key=lambda row: (
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


__all__ = ["PathTaxonomyStore", "RANK_SCREEN_PROMPT_STAGE"]
