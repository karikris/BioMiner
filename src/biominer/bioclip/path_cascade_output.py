from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from biominer.bioclip.path_cascade_classifier import (
    DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    DEFAULT_SPECIES_REPORT_TOP_K,
    DEFAULT_SPECIES_RERANK_TOP_K,
    PathCascadeResult,
    RankCandidateScore,
    RankStepResult,
)
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


def path_cascade_result_to_output_row(result: PathCascadeResult) -> dict[str, Any]:
    steps_by_rank = {step.rank: step for step in result.rank_steps}
    if set(steps_by_rank) != set(CLASSIFICATION_RANKS):
        missing = sorted(set(CLASSIFICATION_RANKS) - set(steps_by_rank))
        raise ValueError("path cascade result is missing rank steps: " + ", ".join(missing))
    selected_by_rank = {score.rank: score for score in result.final_winning_path}
    row: dict[str, Any] = {
        "classifier_schema_version": PATH_CASCADE_OUTPUT_SCHEMA_VERSION,
        "classification_version": result.classification_version,
        "prompt_version": result.prompt_version,
        "hierarchy_fingerprint": result.taxonomy_fingerprint,
        "classification_fingerprint": result.classification_fingerprint,
        "embedding_cache_fingerprint": result.embedding_cache_fingerprint,
        "beam_strategy": result.beam_strategy,
        "rank_beam_width": result.rank_beam_width,
        "species_first_pass_top_k": DEFAULT_SPECIES_FIRST_PASS_TOP_K,
        "species_rerank_top_k": DEFAULT_SPECIES_RERANK_TOP_K,
        "species_report_top_k": DEFAULT_SPECIES_REPORT_TOP_K,
        "skipped_ranks": list(result.skipped_ranks),
        "fully_skipped_ranks": [
            rank for rank in CLASSIFICATION_RANKS if steps_by_rank[rank].skipped
        ],
        "candidate_counts_by_rank": _rank_counts(
            steps_by_rank,
            attribute="candidate_count",
        ),
        "retained_counts_by_rank": _rank_counts(
            steps_by_rank,
            attribute="retained_count",
        ),
        "active_path_counts_before_by_rank": _rank_counts(
            steps_by_rank,
            attribute="active_path_count_before",
        ),
        "active_path_counts_after_by_rank": _rank_counts(
            steps_by_rank,
            attribute="active_path_count_after",
        ),
        "pruning_trace_version": PATH_CASCADE_PRUNING_TRACE_VERSION,
        "pruning_trace_json": "[]",
    }
    for rank in CLASSIFICATION_RANKS[:-1]:
        prefix = rank.casefold()
        step = steps_by_rank[rank]
        top3 = step.top_candidates[:3]
        top1 = top3[0] if top3 else None
        selected = selected_by_rank.get(rank)
        row.update(
            {
                f"{prefix}_top3": [score.scientific_name for score in top3],
                f"{prefix}_top3_node_ids": [score.node_id for score in top3],
                f"{prefix}_top3_scores": [score.raw_similarity for score in top3],
                f"{prefix}_top1": top1.scientific_name if top1 else None,
                f"{prefix}_top1_node_id": top1.node_id if top1 else None,
                f"{prefix}_top1_score": top1.raw_similarity if top1 else None,
                f"{prefix}_margin": step.top1_margin,
                f"selected_{prefix}": selected.scientific_name if selected else None,
                f"selected_{prefix}_node_id": selected.node_id if selected else None,
                f"selected_{prefix}_score": selected.raw_similarity if selected else None,
            }
        )
    row.update(_species_output_values(result))
    return path_cascade_output_frame([row]).row(0, named=True)


def _species_output_values(result: PathCascadeResult) -> dict[str, Any]:
    species_top1 = result.species_top1
    return {
        **_species_list_values(
            prefix="species_top20",
            scores=result.species_top20,
            score_field="first_pass",
        ),
        **_species_list_values(
            prefix="species_top5",
            scores=result.species_top5,
            score_field="rerank",
        ),
        **_species_list_values(
            prefix="species_top3",
            scores=result.species_top3,
            score_field="rerank",
        ),
        "species_top1": species_top1.scientific_name if species_top1 else None,
        "species_top1_node_id": species_top1.node_id if species_top1 else None,
        "species_top1_accepted_taxon_key": (
            species_top1.accepted_taxon_key if species_top1 else None
        ),
        "species_top1_first_pass_score": (
            species_top1.first_pass_raw_similarity if species_top1 else None
        ),
        "species_top1_rerank_score": (
            species_top1.rerank_raw_similarity if species_top1 else None
        ),
        "species_top1_margin": _raw_margin(result.species_reranked_top20),
    }


def _species_list_values(
    *,
    prefix: str,
    scores: Sequence[RankCandidateScore],
    score_field: str,
) -> dict[str, Any]:
    accepted_keys = [score.accepted_taxon_key for score in scores]
    if any(not key for key in accepted_keys):
        raise ValueError(f"{prefix} contains a species without an accepted GBIF key")
    values = (
        [score.first_pass_raw_similarity for score in scores]
        if score_field == "first_pass"
        else [score.rerank_raw_similarity for score in scores]
    )
    if any(value is None for value in values):
        raise ValueError(f"{prefix} is missing its {score_field} score")
    score_column = (
        "species_top20_first_pass_scores"
        if prefix == "species_top20"
        else f"{prefix}_rerank_scores"
    )
    return {
        prefix: [score.scientific_name for score in scores],
        f"{prefix}_node_ids": [score.node_id for score in scores],
        f"{prefix}_accepted_taxon_keys": accepted_keys,
        score_column: values,
    }


def _rank_counts(
    steps_by_rank: Mapping[str, RankStepResult],
    *,
    attribute: str,
) -> dict[str, int]:
    return {rank: int(getattr(steps_by_rank[rank], attribute)) for rank in CLASSIFICATION_RANKS}


def _raw_margin(scores: Sequence[RankCandidateScore]) -> float | None:
    return (
        scores[0].raw_similarity - scores[1].raw_similarity
        if len(scores) > 1
        else None
    )


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
    "path_cascade_result_to_output_row",
    "validate_path_cascade_output_frame",
    "write_path_cascade_output",
]
