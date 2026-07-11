from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from biominer.registry.classification_v3 import CLASSIFICATION_RANKS
from biominer.storage.parquet import DEFAULT_PARQUET_COMPRESSION, write_parquet


PATH_CASCADE_OUTPUT_SCHEMA_VERSION = "butterfly-cascade-output-v1.0.0"
PATH_CASCADE_PRUNING_TRACE_VERSION = "global-rank-pruning-v1"

RANK_COUNT_DTYPE = pl.Struct({rank: pl.UInt32 for rank in CLASSIFICATION_RANKS})
_INTERMEDIATE_RANK_PREFIXES = tuple(rank.casefold() for rank in CLASSIFICATION_RANKS[:-1])


def _rank_output_schema() -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {}
    for prefix in _INTERMEDIATE_RANK_PREFIXES:
        schema.update(
            {
                f"{prefix}_top3": pl.List(pl.String),
                f"{prefix}_top3_node_ids": pl.List(pl.String),
                f"{prefix}_top3_scores": pl.List(pl.Float32),
                f"{prefix}_top1": pl.String,
                f"{prefix}_top1_node_id": pl.String,
                f"{prefix}_top1_score": pl.Float32,
                f"{prefix}_margin": pl.Float32,
                f"selected_{prefix}": pl.String,
                f"selected_{prefix}_node_id": pl.String,
                f"selected_{prefix}_score": pl.Float32,
            }
        )
    return schema


PATH_CASCADE_OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "classifier_schema_version": pl.String,
    "classification_version": pl.String,
    "prompt_version": pl.String,
    "hierarchy_fingerprint": pl.String,
    "classification_fingerprint": pl.String,
    "embedding_cache_fingerprint": pl.String,
    "beam_strategy": pl.String,
    "rank_beam_width": pl.UInt8,
    "species_first_pass_top_k": pl.UInt8,
    "species_rerank_top_k": pl.UInt8,
    "species_report_top_k": pl.UInt8,
    **_rank_output_schema(),
    "species_top20": pl.List(pl.String),
    "species_top20_node_ids": pl.List(pl.String),
    "species_top20_accepted_taxon_keys": pl.List(pl.String),
    "species_top20_first_pass_scores": pl.List(pl.Float32),
    "species_top5": pl.List(pl.String),
    "species_top5_node_ids": pl.List(pl.String),
    "species_top5_accepted_taxon_keys": pl.List(pl.String),
    "species_top5_rerank_scores": pl.List(pl.Float32),
    "species_top3": pl.List(pl.String),
    "species_top3_node_ids": pl.List(pl.String),
    "species_top3_accepted_taxon_keys": pl.List(pl.String),
    "species_top3_rerank_scores": pl.List(pl.Float32),
    "species_top1": pl.String,
    "species_top1_node_id": pl.String,
    "species_top1_accepted_taxon_key": pl.String,
    "species_top1_first_pass_score": pl.Float32,
    "species_top1_rerank_score": pl.Float32,
    "species_top1_margin": pl.Float32,
    "skipped_ranks": pl.List(pl.String),
    "fully_skipped_ranks": pl.List(pl.String),
    "candidate_counts_by_rank": RANK_COUNT_DTYPE,
    "retained_counts_by_rank": RANK_COUNT_DTYPE,
    "active_path_counts_before_by_rank": RANK_COUNT_DTYPE,
    "active_path_counts_after_by_rank": RANK_COUNT_DTYPE,
    "pruning_trace_version": pl.String,
    "pruning_trace_json": pl.String,
}


def empty_path_cascade_output_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=PATH_CASCADE_OUTPUT_SCHEMA)


def path_cascade_output_frame(
    rows: Sequence[Mapping[str, Any]],
) -> pl.DataFrame:
    if not rows:
        return empty_path_cascade_output_frame()
    normalized = [_normalize_row(row) for row in rows]
    frame = pl.DataFrame(normalized, schema=PATH_CASCADE_OUTPUT_SCHEMA, orient="row", strict=True)
    return validate_path_cascade_output_frame(frame)


def validate_path_cascade_output_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.columns != list(PATH_CASCADE_OUTPUT_SCHEMA):
        raise ValueError("path cascade output columns do not match the versioned schema")
    if dict(frame.schema) != PATH_CASCADE_OUTPUT_SCHEMA:
        raise ValueError("path cascade output physical schema mismatch")
    versions = set(frame["classifier_schema_version"].drop_nulls().to_list())
    if versions and versions != {PATH_CASCADE_OUTPUT_SCHEMA_VERSION}:
        raise ValueError("path cascade output classifier schema version mismatch")
    return frame.select(list(PATH_CASCADE_OUTPUT_SCHEMA))


def write_path_cascade_output(
    frame: pl.DataFrame,
    path: str | Path,
    *,
    compression: str = DEFAULT_PARQUET_COMPRESSION,
) -> Path:
    validated = validate_path_cascade_output_frame(frame)
    return write_parquet(validated, path, compression=compression)


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    provided_version = row.get("classifier_schema_version")
    if provided_version not in (None, "", PATH_CASCADE_OUTPUT_SCHEMA_VERSION):
        raise ValueError("path cascade output classifier schema version mismatch")
    normalized: dict[str, Any] = {}
    for name, dtype in PATH_CASCADE_OUTPUT_SCHEMA.items():
        value = row.get(name)
        if name == "classifier_schema_version":
            value = PATH_CASCADE_OUTPUT_SCHEMA_VERSION
        elif name == "pruning_trace_version" and value in (None, ""):
            value = PATH_CASCADE_PRUNING_TRACE_VERSION
        elif isinstance(dtype, pl.List) and value is None:
            value = []
        elif dtype == RANK_COUNT_DTYPE and value is None:
            value = {rank: 0 for rank in CLASSIFICATION_RANKS}
        normalized[name] = value
    return normalized


__all__ = [
    "PATH_CASCADE_OUTPUT_SCHEMA",
    "PATH_CASCADE_OUTPUT_SCHEMA_VERSION",
    "PATH_CASCADE_PRUNING_TRACE_VERSION",
    "RANK_COUNT_DTYPE",
    "empty_path_cascade_output_frame",
    "path_cascade_output_frame",
    "validate_path_cascade_output_frame",
    "write_path_cascade_output",
]
