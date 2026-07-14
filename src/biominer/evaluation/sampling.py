"""Deterministic target-evaluation sampling registers.

The register stores discovery and model evidence used to choose review items.
It deliberately contains no reviewed target-presence field: Flickr query hits
remain provenance, never ground truth.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.flickr_fetch.geographic_clustering import NO_GEO_CLUSTER_ID
from biominer.reports.flickr_fetch import current_git_sha
from biominer.storage.parquet import write_parquet


EVALUATION_SAMPLING_FRAME_SCHEMA_VERSION = "evaluation-sampling-frame-v1.0.0"
EVALUATION_SAMPLING_REPORT_SCHEMA_VERSION = "evaluation-sampling-report-v1.0.0"
PAPILIO_DEMOLEUS_EVALUATION_SAMPLING_FRAME_FILE = (
    "papilio_demoleus_evaluation_sampling_frame.parquet"
)

QUERY_TIERS = ("T1", "T2", "T3", "T4", "T5")
QUERY_TIER_SET = frozenset(QUERY_TIERS)

QUERY_PROVENANCE_SCHEMA: dict[str, pl.DataType] = {
    "query_definition_id": pl.String,
    "query_tier": pl.String,
    "query_term": pl.String,
    "query_field": pl.String,
    "query_priority": pl.Int64,
}

QUERY_DEFINITION_SCHEMA: dict[str, pl.DataType] = QUERY_PROVENANCE_SCHEMA.copy()

EVALUATION_SAMPLING_FRAME_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "sampling_unit_id": pl.String,
    "sampling_hash": pl.String,
    "sampling_rank": pl.UInt32,
    "source": pl.String,
    "flickr_photo_id": pl.String,
    "source_record_hash": pl.String,
    "photo_page_url": pl.String,
    "image_url": pl.String,
    "owner_id": pl.String,
    "owner_name": pl.String,
    "source_owner_group_id": pl.String,
    "year": pl.Int32,
    "year_source": pl.String,
    "year_stratum": pl.String,
    "geo_cluster_id": pl.String,
    "no_geo": pl.Boolean,
    "geo_stratum": pl.String,
    "primary_query_tier": pl.String,
    "query_tiers": pl.List(pl.String),
    "primary_query_term": pl.String,
    "primary_query_field": pl.String,
    "query_terms": pl.List(pl.String),
    "query_fields": pl.List(pl.String),
    "query_definition_ids": pl.List(pl.String),
    "query_hit_count": pl.UInt32,
    "query_provenance": pl.List(pl.Struct(QUERY_PROVENANCE_SCHEMA)),
    "metadata_target_text_evidence": pl.Boolean,
    "metadata_image_category": pl.String,
    "metadata_life_stage": pl.String,
    "initial_score_status": pl.String,
    "initial_target_score_id": pl.String,
    "initial_scoring_unit_id": pl.String,
    "initial_route": pl.String,
    "yoloe_route": pl.String,
    "yoloe_routes": pl.List(pl.String),
    "subject_area_ratio": pl.Float32,
    "subject_area_band": pl.String,
    "initial_visual_domain": pl.String,
    "visual_domain_source": pl.String,
    "initial_reference_score": pl.Float32,
    "initial_reference_score_band": pl.String,
    "initial_reference_score_tail": pl.String,
    "initial_competitor_margin": pl.Float32,
    "initial_competitor_margin_band": pl.String,
    "initial_competitor_margin_tail": pl.String,
    "best_competitor_accepted_taxon_key": pl.String,
    "best_competitor_scientific_name": pl.String,
    "current_false_positive_genus": pl.String,
    "false_positive_genus_stratum": pl.String,
    "visual_input_disagreement": pl.Float32,
    "visual_input_disagreement_band": pl.String,
    "text_image_reference_disagreement": pl.String,
}

_CANDIDATE_REQUIRED_COLUMNS = {
    "source",
    "flickr_photo_id",
    "source_record_hash",
    "photo_page_url",
    "image_url",
    "owner_id",
    "owner_name",
    "date_taken",
    "date_upload",
    "raw_title",
    "raw_description",
    "raw_tags",
    "machine_tags",
    "query_definition_ids",
    "query_hit_count",
    "image_category",
    "life_stage",
}
_GEO_REQUIRED_COLUMNS = {
    "source",
    "flickr_photo_id",
    "source_record_hash",
    "geo_cluster_id",
}
_SCORE_REQUIRED_COLUMNS = {
    "target_score_id",
    "source",
    "flickr_photo_id",
    "source_record_hash",
    "scoring_unit_id",
    "route",
    "target_reference_centroid_similarity",
    "target_competitor_margin",
    "best_competitor_accepted_taxon_key",
    "best_competitor_scientific_name",
    "yoloe_route",
    "subject_area_ratio",
    "visual_input_disagreement",
    "route_compatible",
}
_FORBIDDEN_OUTPUT_COLUMNS = frozenset(
    {
        "accepted_taxon_key",
        "all_query_labels",
        "is_target_positive",
        "label_level",
        "reviewed_label",
        "scientific_name",
        "target_present",
    }
)
_KEY = ["source", "flickr_photo_id"]
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluationSamplingConfig:
    """Versioned thresholds and deterministic rank seed."""

    target_text_terms: tuple[str, ...] = ()
    random_seed: int = 42
    low_tail_quantile: float = 0.10
    high_tail_quantile: float = 0.90
    near_tie_margin: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise TypeError("random_seed must be an integer")
        if not 0 <= self.random_seed <= 2**64 - 1:
            raise ValueError("random_seed must be between 0 and 2**64 - 1")
        for field in ("low_tail_quantile", "high_tail_quantile"):
            value = float(getattr(self, field))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be finite and between zero and one")
            object.__setattr__(self, field, value)
        if self.low_tail_quantile >= self.high_tail_quantile:
            raise ValueError("low_tail_quantile must be below high_tail_quantile")
        margin = float(self.near_tie_margin)
        if not math.isfinite(margin) or margin <= 0.0:
            raise ValueError("near_tie_margin must be finite and positive")
        object.__setattr__(self, "near_tie_margin", margin)
        normalized_terms = tuple(
            dict.fromkeys(
                term
                for term in (
                    _normalize_token_text(value) for value in self.target_text_terms
                )
                if term
            )
        )
        object.__setattr__(self, "target_text_terms", normalized_terms)

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": EVALUATION_SAMPLING_FRAME_SCHEMA_VERSION,
                "target_text_terms": list(self.target_text_terms),
                "random_seed": self.random_seed,
                "low_tail_quantile": self.low_tail_quantile,
                "high_tail_quantile": self.high_tail_quantile,
                "near_tie_margin": self.near_tie_margin,
            }
        )


@dataclass(frozen=True, slots=True)
class EvaluationSamplingPublication:
    frame_path: Path
    report_json_path: Path
    report_markdown_path: Path
    report: Mapping[str, Any]


def empty_evaluation_sampling_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=EVALUATION_SAMPLING_FRAME_SCHEMA)


def read_flickr_query_definitions(state_db: str | Path) -> pl.DataFrame:
    """Read one consistent definition per ID from a poller's SQLite state."""

    path = Path(state_db)
    if not path.is_file():
        raise FileNotFoundError(f"Flickr state database does not exist: {path}")
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT
                query_definition_id,
                trust_tier,
                term,
                search_field,
                query_priority
            FROM flickr_work_items
            WHERE query_definition_id IS NOT NULL
              AND query_definition_id != ''
            ORDER BY query_definition_id, page, work_item_id
            """
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise ValueError(
            f"could not read Flickr query definitions from {path}: {exc}"
        ) from exc
    finally:
        connection.close()
    definitions: dict[str, tuple[str, str, str, int]] = {}
    for definition_id, tier, term, field, priority in rows:
        normalized = (
            _required_text(tier, field="trust_tier"),
            _required_text(term, field="term"),
            _required_text(field, field="search_field"),
            _required_integer(priority, field="query_priority"),
        )
        identifier = _required_text(
            definition_id,
            field="query_definition_id",
        )
        previous = definitions.setdefault(identifier, normalized)
        if previous != normalized:
            raise ValueError(
                "Flickr state contains conflicting query-definition rows for "
                f"{identifier!r}"
            )
    frame = pl.DataFrame(
        [
            {
                "query_definition_id": identifier,
                "query_tier": values[0],
                "query_term": values[1],
                "query_field": values[2],
                "query_priority": values[3],
            }
            for identifier, values in sorted(definitions.items())
        ],
        schema=QUERY_DEFINITION_SCHEMA,
    )
    _validate_query_definitions(frame)
    return frame


def build_evaluation_sampling_frame(
    candidates: pl.DataFrame,
    geo_assignments: pl.DataFrame,
    query_definitions: pl.DataFrame,
    *,
    object_scores: pl.DataFrame | None = None,
    competitor_taxa: pl.DataFrame | None = None,
    config: EvaluationSamplingConfig | None = None,
) -> pl.DataFrame:
    """Build one deterministic, unlabelled sampling row per Flickr photo."""

    policy = config or EvaluationSamplingConfig()
    candidate_frame = _normalize_candidates(candidates)
    if candidate_frame.is_empty():
        return empty_evaluation_sampling_frame()
    geo_frame = _normalize_geo_assignments(geo_assignments, candidate_frame)
    definition_frame = _normalize_query_definitions(query_definitions)
    query_frame = _candidate_query_provenance(candidate_frame, definition_frame)
    score_frame = _primary_scores(
        object_scores,
        candidate_frame,
        competitor_taxa=competitor_taxa,
    )

    frame = (
        candidate_frame.join(geo_frame, on=_KEY, how="left", validate="1:1")
        .join(query_frame, on=_KEY, how="left", validate="1:1")
        .join(score_frame, on=_KEY, how="left", validate="1:1")
    )
    frame = _derive_candidate_strata(frame, config=policy)
    frame = _derive_score_strata(frame, config=policy)
    frame = _add_sampling_identities(frame, config=policy)
    output = frame.select(list(EVALUATION_SAMPLING_FRAME_SCHEMA)).cast(
        EVALUATION_SAMPLING_FRAME_SCHEMA
    )
    validate_evaluation_sampling_frame(output)
    return output.sort("sampling_rank")


def validate_evaluation_sampling_frame(frame: pl.DataFrame) -> None:
    missing = sorted(set(EVALUATION_SAMPLING_FRAME_SCHEMA) - set(frame.columns))
    if missing:
        raise ValueError(f"evaluation sampling frame is missing columns: {missing}")
    forbidden = sorted(_FORBIDDEN_OUTPUT_COLUMNS & set(frame.columns))
    if forbidden:
        raise ValueError(
            f"evaluation sampling frame contains ground-truth columns: {forbidden}"
        )
    mismatches = {
        column: (str(frame.schema[column]), str(dtype))
        for column, dtype in EVALUATION_SAMPLING_FRAME_SCHEMA.items()
        if frame.schema[column] != dtype
    }
    if mismatches:
        raise ValueError(
            f"evaluation sampling frame has incompatible column types: {mismatches}"
        )
    if frame.is_empty():
        return
    versions = set(frame["schema_version"].to_list())
    if versions != {EVALUATION_SAMPLING_FRAME_SCHEMA_VERSION}:
        raise ValueError(f"unsupported sampling schema versions: {versions}")
    _require_unique(frame, ["sampling_unit_id"], label="sampling unit")
    _require_unique(frame, _KEY, label="Flickr candidate")
    ranks = frame["sampling_rank"].sort().to_list()
    if ranks != list(range(1, frame.height + 1)):
        raise ValueError("sampling_rank must contain each rank exactly once")
    if frame.filter(~pl.col("primary_query_tier").is_in(QUERY_TIERS)).height:
        raise ValueError("sampling frame contains an invalid primary query tier")


def materialize_evaluation_sampling_frame(
    *,
    candidates_path: str | Path,
    geo_assignments_path: str | Path,
    query_state_db: str | Path,
    output_path: str | Path,
    object_scores_path: str | Path | None = None,
    competitor_taxa_path: str | Path | None = None,
    config: EvaluationSamplingConfig | None = None,
    run_id: str | None = None,
) -> EvaluationSamplingPublication:
    """Load, build, write and audit a local evaluation sampling register."""

    started_at = datetime.now(UTC)
    policy = config or EvaluationSamplingConfig()
    effective_run_id = _required_text(
        run_id
        or "evaluation-sampling-"
        + started_at.strftime("%Y%m%dT%H%M%S%fZ-")
        + uuid4().hex[:12],
        field="run_id",
    )
    inputs = {
        "candidates": _required_file(candidates_path),
        "geo_assignments": _required_file(geo_assignments_path),
        "query_state_db": _required_file(query_state_db),
    }
    if object_scores_path is not None:
        inputs["object_scores"] = _required_file(object_scores_path)
    if competitor_taxa_path is not None:
        inputs["competitor_taxa"] = _required_file(competitor_taxa_path)
    _log_event(
        "evaluation_sampling_build_started",
        command="evaluation.build_sampling_frame",
        run_id=effective_run_id,
        inputs={name: str(path) for name, path in inputs.items()},
        output=str(output_path),
        started_at=started_at.isoformat(),
    )
    candidates = pl.read_parquet(inputs["candidates"])
    geo_assignments = pl.read_parquet(inputs["geo_assignments"])
    query_definitions = read_flickr_query_definitions(inputs["query_state_db"])
    object_scores = (
        pl.read_parquet(inputs["object_scores"]) if "object_scores" in inputs else None
    )
    competitor_taxa = (
        pl.read_parquet(inputs["competitor_taxa"])
        if "competitor_taxa" in inputs
        else None
    )
    frame = build_evaluation_sampling_frame(
        candidates,
        geo_assignments,
        query_definitions,
        object_scores=object_scores,
        competitor_taxa=competitor_taxa,
        config=policy,
    )
    destination = Path(output_path)
    if destination.suffix.casefold() != ".parquet":
        raise ValueError("evaluation sampling output must use a .parquet suffix")
    frame_path = write_parquet(frame, destination)
    ended_at = datetime.now(UTC)
    report = _sampling_report(
        frame,
        frame_path=frame_path,
        input_paths=inputs,
        config=policy,
        run_id=effective_run_id,
        started_at=started_at,
        ended_at=ended_at,
    )
    report_json_path = destination.with_suffix(".report.json")
    report_markdown_path = destination.with_suffix(".report.md")
    _atomic_write_text(
        report_json_path,
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
    )
    _atomic_write_text(report_markdown_path, _sampling_report_markdown(report))
    _log_event(
        "evaluation_sampling_build_completed",
        command="evaluation.build_sampling_frame",
        run_id=effective_run_id,
        output=str(frame_path),
        row_count=frame.height,
        byte_count=frame_path.stat().st_size,
        ended_at=ended_at.isoformat(),
        elapsed_seconds=report["elapsed_seconds"],
    )
    return EvaluationSamplingPublication(
        frame_path=frame_path,
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
        report=report,
    )


def _normalize_candidates(frame: pl.DataFrame) -> pl.DataFrame:
    _require_frame(frame, label="candidates")
    _require_columns(frame, _CANDIDATE_REQUIRED_COLUMNS, label="candidates")
    candidates = frame.select(sorted(_CANDIDATE_REQUIRED_COLUMNS)).with_columns(
        pl.col("source").cast(pl.String),
        pl.col("flickr_photo_id").cast(pl.String),
        pl.col("source_record_hash").cast(pl.String),
        pl.col("photo_page_url").cast(pl.String),
        pl.col("image_url").cast(pl.String),
        pl.col("owner_id").cast(pl.String),
        pl.col("owner_name").cast(pl.String),
        pl.col("date_taken").cast(pl.String),
        pl.col("date_upload").cast(pl.String),
        pl.col("raw_title").cast(pl.String),
        pl.col("raw_description").cast(pl.String),
        pl.col("raw_tags").cast(pl.String),
        pl.col("machine_tags").cast(pl.String),
        pl.col("query_definition_ids").cast(pl.List(pl.String)),
        pl.col("query_hit_count").cast(pl.UInt32),
        pl.col("image_category").cast(pl.String),
        pl.col("life_stage").cast(pl.String),
    )
    _require_nonblank(candidates, _KEY + ["source_record_hash", "owner_id"])
    _require_unique(candidates, _KEY, label="Flickr candidate")
    if candidates.filter(pl.col("query_definition_ids").list.len() == 0).height:
        raise ValueError("every candidate must retain at least one query definition")
    return candidates.sort(_KEY)


def _normalize_geo_assignments(
    frame: pl.DataFrame,
    candidates: pl.DataFrame,
) -> pl.DataFrame:
    _require_frame(frame, label="geo assignments")
    _require_columns(frame, _GEO_REQUIRED_COLUMNS, label="geo assignments")
    geo = frame.select(sorted(_GEO_REQUIRED_COLUMNS)).cast(
        {column: pl.String for column in _GEO_REQUIRED_COLUMNS}
    )
    _require_nonblank(geo, _KEY + ["source_record_hash", "geo_cluster_id"])
    _require_unique(geo, _KEY, label="geo assignment")
    checked = candidates.select(_KEY + ["source_record_hash"]).join(
        geo,
        on=_KEY,
        how="full",
        suffix="_geo",
        validate="1:1",
    )
    missing = checked.filter(
        pl.col("source_record_hash").is_null()
        | pl.col("source_record_hash_geo").is_null()
    )
    if missing.height:
        raise ValueError(
            "geo assignments must contain exactly one row for every candidate"
        )
    mismatch = checked.filter(
        pl.col("source_record_hash") != pl.col("source_record_hash_geo")
    )
    if mismatch.height:
        raise ValueError("geo assignments do not match candidate record hashes")
    return geo.select(_KEY + ["geo_cluster_id"])


def _normalize_query_definitions(frame: pl.DataFrame) -> pl.DataFrame:
    _require_frame(frame, label="query definitions")
    _require_columns(frame, set(QUERY_DEFINITION_SCHEMA), label="query definitions")
    definitions = frame.select(list(QUERY_DEFINITION_SCHEMA)).cast(
        QUERY_DEFINITION_SCHEMA
    )
    _validate_query_definitions(definitions)
    return definitions


def _validate_query_definitions(frame: pl.DataFrame) -> None:
    _require_nonblank(
        frame,
        ["query_definition_id", "query_tier", "query_term", "query_field"],
    )
    _require_unique(frame, ["query_definition_id"], label="query definition")
    invalid_tiers = sorted(set(frame["query_tier"].to_list()) - QUERY_TIER_SET)
    if invalid_tiers:
        raise ValueError(f"unsupported query tiers: {invalid_tiers}")
    invalid_fields = sorted(set(frame["query_field"].to_list()) - {"tags", "text"})
    if invalid_fields:
        raise ValueError(f"unsupported Flickr search fields: {invalid_fields}")


def _candidate_query_provenance(
    candidates: pl.DataFrame,
    definitions: pl.DataFrame,
) -> pl.DataFrame:
    links = (
        candidates.select(_KEY + ["query_definition_ids"])
        .explode("query_definition_ids")
        .rename({"query_definition_ids": "query_definition_id"})
        .join(
            definitions,
            on="query_definition_id",
            how="left",
            validate="m:1",
        )
    )
    unresolved = links.filter(pl.col("query_tier").is_null())
    if unresolved.height:
        identifiers = sorted(
            set(unresolved["query_definition_id"].drop_nulls().to_list())
        )
        raise ValueError(
            "candidate query definitions are missing from Flickr state: "
            f"{identifiers[:10]}"
        )
    links = links.with_columns(
        pl.when(pl.col("query_field") == "tags")
        .then(pl.lit(0, dtype=pl.UInt8))
        .otherwise(pl.lit(1, dtype=pl.UInt8))
        .alias("_field_order")
    ).sort(
        _KEY
        + [
            "query_priority",
            "_field_order",
            "query_term",
            "query_definition_id",
        ]
    )
    return links.group_by(_KEY, maintain_order=True).agg(
        pl.col("query_tier").first().alias("primary_query_tier"),
        pl.col("query_tier").unique(maintain_order=True).alias("query_tiers"),
        pl.col("query_term").first().alias("primary_query_term"),
        pl.col("query_field").first().alias("primary_query_field"),
        pl.col("query_term").unique(maintain_order=True).alias("query_terms"),
        pl.col("query_field").unique(maintain_order=True).alias("query_fields"),
        pl.col("query_definition_id")
        .unique(maintain_order=True)
        .alias("query_definition_ids"),
        pl.struct(list(QUERY_PROVENANCE_SCHEMA)).alias("query_provenance"),
    )


def _primary_scores(
    object_scores: pl.DataFrame | None,
    candidates: pl.DataFrame,
    *,
    competitor_taxa: pl.DataFrame | None,
) -> pl.DataFrame:
    if object_scores is None or object_scores.is_empty():
        return _empty_primary_scores()
    _require_frame(object_scores, label="object scores")
    _require_columns(object_scores, _SCORE_REQUIRED_COLUMNS, label="object scores")
    scores = object_scores.select(sorted(_SCORE_REQUIRED_COLUMNS)).with_columns(
        pl.col("target_score_id").cast(pl.String),
        pl.col("source").cast(pl.String),
        pl.col("flickr_photo_id").cast(pl.String),
        pl.col("source_record_hash").cast(pl.String),
        pl.col("scoring_unit_id").cast(pl.String),
        pl.col("route").cast(pl.String),
        pl.col("target_reference_centroid_similarity").cast(pl.Float32),
        pl.col("target_competitor_margin").cast(pl.Float32),
        pl.col("best_competitor_accepted_taxon_key").cast(pl.String),
        pl.col("best_competitor_scientific_name").cast(pl.String),
        pl.col("yoloe_route").cast(pl.String),
        pl.col("subject_area_ratio").cast(pl.Float32),
        pl.col("visual_input_disagreement").cast(pl.Float32),
        pl.col("route_compatible").cast(pl.Boolean),
    )
    _require_nonblank(
        scores,
        _KEY + ["source_record_hash", "target_score_id", "scoring_unit_id"],
    )
    _require_unique(scores, ["target_score_id"], label="target score")
    _validate_score_numbers(scores)
    candidate_hashes = candidates.select(_KEY + ["source_record_hash"])
    checked = scores.join(
        candidate_hashes,
        on=_KEY,
        how="left",
        suffix="_candidate",
        validate="m:1",
    )
    if checked.filter(pl.col("source_record_hash_candidate").is_null()).height:
        raise ValueError("object scores contain photos outside the candidate stream")
    if checked.filter(
        pl.col("source_record_hash") != pl.col("source_record_hash_candidate")
    ).height:
        raise ValueError("object scores do not match candidate record hashes")
    scores = scores.with_columns(
        pl.col("target_reference_centroid_similarity")
        .is_not_null()
        .alias("_reference_available")
    ).sort(
        _KEY
        + [
            "route_compatible",
            "_reference_available",
            "target_reference_centroid_similarity",
            "target_competitor_margin",
            "scoring_unit_id",
        ],
        descending=[False, False, True, True, True, True, False],
        nulls_last=True,
    )
    primary = scores.group_by(_KEY, maintain_order=True).agg(
        pl.col("target_score_id").first().alias("initial_target_score_id"),
        pl.col("scoring_unit_id").first().alias("initial_scoring_unit_id"),
        pl.col("route").first().alias("initial_route"),
        pl.col("yoloe_route").first().alias("yoloe_route"),
        pl.col("yoloe_route")
        .drop_nulls()
        .unique(maintain_order=True)
        .alias("yoloe_routes"),
        pl.col("subject_area_ratio").first().alias("subject_area_ratio"),
        pl.col("target_reference_centroid_similarity")
        .first()
        .alias("initial_reference_score"),
        pl.col("target_competitor_margin").first().alias("initial_competitor_margin"),
        pl.col("best_competitor_accepted_taxon_key")
        .first()
        .alias("best_competitor_accepted_taxon_key"),
        pl.col("best_competitor_scientific_name")
        .first()
        .alias("best_competitor_scientific_name"),
        pl.col("visual_input_disagreement").first().alias("visual_input_disagreement"),
    )
    return _add_false_positive_genus(primary, competitor_taxa)


def _empty_primary_scores() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "source": pl.String,
            "flickr_photo_id": pl.String,
            "initial_target_score_id": pl.String,
            "initial_scoring_unit_id": pl.String,
            "initial_route": pl.String,
            "yoloe_route": pl.String,
            "yoloe_routes": pl.List(pl.String),
            "subject_area_ratio": pl.Float32,
            "initial_reference_score": pl.Float32,
            "initial_competitor_margin": pl.Float32,
            "best_competitor_accepted_taxon_key": pl.String,
            "best_competitor_scientific_name": pl.String,
            "visual_input_disagreement": pl.Float32,
            "current_false_positive_genus": pl.String,
        }
    )


def _add_false_positive_genus(
    primary: pl.DataFrame,
    competitor_taxa: pl.DataFrame | None,
) -> pl.DataFrame:
    losing = primary.filter(pl.col("initial_competitor_margin") < 0.0)
    if losing.is_empty():
        return primary.with_columns(
            pl.lit(None, dtype=pl.String).alias("current_false_positive_genus")
        )
    if losing.filter(pl.col("best_competitor_accepted_taxon_key").is_null()).height:
        raise ValueError("target-losing scores require a best competitor taxon key")
    if competitor_taxa is None:
        raise ValueError(
            "competitor_taxa is required to resolve current false-positive genera"
        )
    _require_frame(competitor_taxa, label="competitor taxa")
    key_column = (
        "candidate_accepted_taxon_key"
        if "candidate_accepted_taxon_key" in competitor_taxa.columns
        else "accepted_taxon_key"
    )
    _require_columns(
        competitor_taxa,
        {key_column, "genus"},
        label="competitor taxa",
    )
    taxa = (
        competitor_taxa.select(
            pl.col(key_column)
            .cast(pl.String)
            .alias("best_competitor_accepted_taxon_key"),
            pl.col("genus").cast(pl.String).alias("_competitor_genus"),
        )
        .drop_nulls()
        .unique()
    )
    conflicts = (
        taxa.group_by("best_competitor_accepted_taxon_key")
        .agg(pl.col("_competitor_genus").n_unique().alias("genus_count"))
        .filter(pl.col("genus_count") != 1)
    )
    if conflicts.height:
        raise ValueError("competitor taxa map one accepted key to multiple genera")
    taxa = taxa.unique(subset=["best_competitor_accepted_taxon_key"])
    enriched = primary.join(
        taxa,
        on="best_competitor_accepted_taxon_key",
        how="left",
        validate="m:1",
    )
    unresolved = enriched.filter(
        (pl.col("initial_competitor_margin") < 0.0)
        & pl.col("_competitor_genus").is_null()
    )
    if unresolved.height:
        keys = sorted(
            set(unresolved["best_competitor_accepted_taxon_key"].drop_nulls().to_list())
        )
        raise ValueError(f"competitor genera are unresolved for keys: {keys[:10]}")
    return enriched.with_columns(
        pl.when(pl.col("initial_competitor_margin") < 0.0)
        .then(pl.col("_competitor_genus"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("current_false_positive_genus")
    ).drop("_competitor_genus")


def _derive_candidate_strata(
    frame: pl.DataFrame,
    *,
    config: EvaluationSamplingConfig,
) -> pl.DataFrame:
    year_taken = _bounded_year(
        pl.col("date_taken").str.slice(0, 4).cast(pl.Int32, strict=False)
    )
    upload_iso_year = pl.when(pl.col("date_upload").str.contains(r"^\d{4}-")).then(
        pl.col("date_upload").str.slice(0, 4).cast(pl.Int32, strict=False)
    )
    upload_epoch_year = pl.from_epoch(
        pl.col("date_upload").cast(pl.Int64, strict=False),
        time_unit="s",
    ).dt.year()
    year_upload = _bounded_year(pl.coalesce(upload_iso_year, upload_epoch_year))
    year = pl.coalesce(year_taken, year_upload)
    target_text = _target_text_evidence_expression(config.target_text_terms)
    derived = frame.with_columns(
        year.alias("year"),
        pl.when(year_taken.is_not_null())
        .then(pl.lit("date_taken"))
        .when(year_upload.is_not_null())
        .then(pl.lit("date_upload"))
        .otherwise(pl.lit("unknown"))
        .alias("year_source"),
        pl.when(year.is_not_null())
        .then(pl.concat_str(pl.lit("year:"), year.cast(pl.String)))
        .otherwise(pl.lit("unknown_year"))
        .alias("year_stratum"),
        (pl.col("geo_cluster_id") == NO_GEO_CLUSTER_ID).alias("no_geo"),
        pl.when(pl.col("geo_cluster_id") == NO_GEO_CLUSTER_ID)
        .then(pl.lit("no_geo"))
        .otherwise(pl.col("geo_cluster_id"))
        .alias("geo_stratum"),
        target_text.alias("metadata_target_text_evidence"),
        pl.col("image_category").alias("metadata_image_category"),
        pl.col("life_stage").alias("metadata_life_stage"),
    )
    owner_ids = [
        "source-owner-group:"
        + canonical_semantic_fingerprint(
            {"source": source, "owner_id": owner_id}
        ).removeprefix("sha256:")
        for source, owner_id in derived.select("source", "owner_id").iter_rows()
    ]
    return derived.with_columns(
        pl.Series("source_owner_group_id", owner_ids, dtype=pl.String)
    )


def _derive_score_strata(
    frame: pl.DataFrame,
    *,
    config: EvaluationSamplingConfig,
) -> pl.DataFrame:
    has_score_row = pl.col("initial_target_score_id").is_not_null()
    has_reference = pl.col("initial_reference_score").is_not_null()
    has_margin = pl.col("initial_competitor_margin").is_not_null()
    visual_domain = (
        pl.when(pl.col("yoloe_route").is_in(["adult_field", "larval", "pupal", "egg"]))
        .then(pl.lit("live_field"))
        .when(pl.col("yoloe_route") == "pinned_specimen")
        .then(pl.lit("pinned_specimen"))
        .when(pl.col("image_category") == "artwork")
        .then(pl.lit("artwork"))
        .when(pl.col("image_category") == "logo_or_brand")
        .then(pl.lit("logo"))
        .when(pl.col("image_category") == "tattoo")
        .then(pl.lit("tattoo"))
        .when(pl.col("image_category") == "museum_specimen")
        .then(pl.lit("pinned_specimen"))
        .when(
            pl.col("image_category").is_in(
                ["object_or_product", "textile_or_pattern", "not_lepidoptera"]
            )
        )
        .then(pl.lit("unsuitable"))
        .when(pl.col("image_category") == "life_stage_non_adult")
        .then(pl.lit("live_field"))
        .otherwise(pl.lit("ambiguous"))
    )
    frame = frame.with_columns(
        pl.when(~has_score_row)
        .then(pl.lit("not_scored"))
        .when(~has_reference)
        .then(pl.lit("reference_score_missing"))
        .when(~has_margin)
        .then(pl.lit("competitor_margin_missing"))
        .otherwise(pl.lit("scored"))
        .alias("initial_score_status"),
        pl.col("initial_route").fill_null("not_scored"),
        pl.col("yoloe_route").fill_null("not_run"),
        pl.col("yoloe_routes").fill_null(pl.lit([], dtype=pl.List(pl.String))),
        _subject_area_band().alias("subject_area_band"),
        visual_domain.alias("initial_visual_domain"),
        pl.when(pl.col("yoloe_route").is_not_null())
        .then(pl.lit("yoloe_route"))
        .when(pl.col("image_category") != "unknown")
        .then(pl.lit("metadata_keyword_heuristic"))
        .otherwise(pl.lit("unresolved"))
        .alias("visual_domain_source"),
        _reference_score_band().alias("initial_reference_score_band"),
        _competitor_margin_band(config.near_tie_margin).alias(
            "initial_competitor_margin_band"
        ),
        pl.when(~has_score_row)
        .then(pl.lit("not_scored"))
        .when(pl.col("visual_input_disagreement").is_null())
        .then(pl.lit("not_available"))
        .when(pl.col("visual_input_disagreement") <= 0.05)
        .then(pl.lit("low"))
        .when(pl.col("visual_input_disagreement") <= 0.20)
        .then(pl.lit("moderate"))
        .otherwise(pl.lit("high"))
        .alias("visual_input_disagreement_band"),
        pl.when(~has_score_row)
        .then(pl.lit("not_scored"))
        .when(pl.col("current_false_positive_genus").is_not_null())
        .then(pl.col("current_false_positive_genus"))
        .when(~has_margin)
        .then(pl.lit("margin_unavailable"))
        .otherwise(pl.lit("target_not_outscored"))
        .alias("false_positive_genus_stratum"),
        _text_image_disagreement().alias("text_image_reference_disagreement"),
    )
    frame = _add_tail_stratum(
        frame,
        value_column="initial_reference_score",
        output_column="initial_reference_score_tail",
        config=config,
    )
    return _add_tail_stratum(
        frame,
        value_column="initial_competitor_margin",
        output_column="initial_competitor_margin_tail",
        config=config,
    )


def _add_tail_stratum(
    frame: pl.DataFrame,
    *,
    value_column: str,
    output_column: str,
    config: EvaluationSamplingConfig,
) -> pl.DataFrame:
    values = frame[value_column].drop_nulls()
    if values.is_empty():
        return frame.with_columns(pl.lit("not_scored").alias(output_column))
    low = float(values.quantile(config.low_tail_quantile, interpolation="nearest"))
    high = float(values.quantile(config.high_tail_quantile, interpolation="nearest"))
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
        return frame.with_columns(
            pl.when(pl.col(value_column).is_null())
            .then(pl.lit("not_scored"))
            .otherwise(pl.lit("flat_distribution"))
            .alias(output_column)
        )
    return frame.with_columns(
        pl.when(pl.col(value_column).is_null())
        .then(pl.lit("not_scored"))
        .when(pl.col(value_column) <= low)
        .then(pl.lit("low_tail"))
        .when(pl.col(value_column) >= high)
        .then(pl.lit("high_tail"))
        .otherwise(pl.lit("middle"))
        .alias(output_column)
    )


def _add_sampling_identities(
    frame: pl.DataFrame,
    *,
    config: EvaluationSamplingConfig,
) -> pl.DataFrame:
    rows = frame.select(_KEY).iter_rows()
    unit_ids: list[str] = []
    hashes: list[str] = []
    for source, flickr_photo_id in rows:
        identity = {"source": source, "flickr_photo_id": flickr_photo_id}
        unit_ids.append(
            "evaluation-sampling-unit:"
            + canonical_semantic_fingerprint(
                {
                    "version": EVALUATION_SAMPLING_FRAME_SCHEMA_VERSION,
                    **identity,
                }
            ).removeprefix("sha256:")
        )
        hashes.append(
            canonical_semantic_fingerprint(
                {"random_seed": config.random_seed, **identity}
            )
        )
    with_hashes = frame.with_columns(
        pl.lit(EVALUATION_SAMPLING_FRAME_SCHEMA_VERSION).alias("schema_version"),
        pl.Series("sampling_unit_id", unit_ids, dtype=pl.String),
        pl.Series("sampling_hash", hashes, dtype=pl.String),
    )
    ranks = (
        with_hashes.select(_KEY + ["sampling_hash"])
        .sort(["sampling_hash", *_KEY])
        .with_row_index("sampling_rank", offset=1)
        .with_columns(pl.col("sampling_rank").cast(pl.UInt32))
        .select(_KEY + ["sampling_rank"])
    )
    return with_hashes.join(ranks, on=_KEY, how="left", validate="1:1")


def _target_text_evidence_expression(
    terms: Sequence[str],
) -> pl.Expr:
    if not terms:
        return pl.lit(None, dtype=pl.Boolean)
    combined = pl.concat_str(
        [
            pl.col("raw_title").fill_null(""),
            pl.col("raw_description").fill_null(""),
            pl.col("raw_tags").fill_null(""),
            pl.col("machine_tags").fill_null(""),
        ],
        separator=" ",
    )
    normalized = (
        combined.str.to_lowercase()
        .str.replace_all(r"[^\p{L}\p{N}]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )
    padded = pl.concat_str(pl.lit(" "), normalized, pl.lit(" "))
    expression = pl.lit(False)
    for term in terms:
        expression = expression | padded.str.contains(f" {term} ", literal=True)
    return expression


def _bounded_year(expression: pl.Expr) -> pl.Expr:
    return pl.when(expression.is_between(1800, 2100)).then(expression)


def _subject_area_band() -> pl.Expr:
    value = pl.col("subject_area_ratio")
    return (
        pl.when(value.is_null())
        .then(pl.lit("not_measured"))
        .when(value <= 0.01)
        .then(pl.lit("very_small"))
        .when(value <= 0.05)
        .then(pl.lit("small"))
        .when(value <= 0.25)
        .then(pl.lit("medium"))
        .when(value <= 0.50)
        .then(pl.lit("large"))
        .otherwise(pl.lit("dominant"))
    )


def _reference_score_band() -> pl.Expr:
    value = pl.col("initial_reference_score")
    return (
        pl.when(value.is_null())
        .then(pl.lit("not_scored"))
        .when(value < 0.0)
        .then(pl.lit("negative"))
        .when(value < 0.25)
        .then(pl.lit("0.00_to_0.25"))
        .when(value < 0.50)
        .then(pl.lit("0.25_to_0.50"))
        .when(value < 0.75)
        .then(pl.lit("0.50_to_0.75"))
        .otherwise(pl.lit("0.75_to_1.00"))
    )


def _competitor_margin_band(near_tie_margin: float) -> pl.Expr:
    value = pl.col("initial_competitor_margin")
    return (
        pl.when(value.is_null())
        .then(pl.lit("not_scored"))
        .when(value < -near_tie_margin)
        .then(pl.lit("competitor_clear"))
        .when(value < 0.0)
        .then(pl.lit("competitor_narrow"))
        .when(value <= near_tie_margin)
        .then(pl.lit("near_tie"))
        .otherwise(pl.lit("target_clear"))
    )


def _text_image_disagreement() -> pl.Expr:
    text_evidence = pl.col("metadata_target_text_evidence")
    margin = pl.col("initial_competitor_margin")
    return (
        pl.when(text_evidence.is_null())
        .then(pl.lit("text_evidence_unavailable"))
        .when(margin.is_null())
        .then(pl.lit("reference_evidence_unavailable"))
        .when(margin == 0.0)
        .then(pl.lit("reference_tie"))
        .when(text_evidence & (margin > 0.0))
        .then(pl.lit("agreement_target"))
        .when(text_evidence & (margin < 0.0))
        .then(pl.lit("disagreement_text_target_reference_competitor"))
        .when(~text_evidence & (margin > 0.0))
        .then(pl.lit("disagreement_text_absent_reference_target"))
        .otherwise(pl.lit("agreement_no_target_text_reference_competitor"))
    )


def _validate_score_numbers(frame: pl.DataFrame) -> None:
    for column in (
        "target_reference_centroid_similarity",
        "target_competitor_margin",
        "subject_area_ratio",
        "visual_input_disagreement",
    ):
        invalid = frame.filter(
            pl.col(column).is_not_null() & ~pl.col(column).is_finite()
        )
        if invalid.height:
            raise ValueError(f"object score field {column} must be finite")
    for column in ("subject_area_ratio", "visual_input_disagreement"):
        invalid = frame.filter(
            pl.col(column).is_not_null()
            & ((pl.col(column) < 0.0) | (pl.col(column) > 1.0))
        )
        if invalid.height:
            raise ValueError(
                f"object score field {column} must be between zero and one"
            )
    invalid_similarity = frame.filter(
        pl.col("target_reference_centroid_similarity").is_not_null()
        & (
            (pl.col("target_reference_centroid_similarity") < -1.0)
            | (pl.col("target_reference_centroid_similarity") > 1.0)
        )
    )
    if invalid_similarity.height:
        raise ValueError("target reference similarities must be between -1 and 1")


def _sampling_report(
    frame: pl.DataFrame,
    *,
    frame_path: Path,
    input_paths: Mapping[str, Path],
    config: EvaluationSamplingConfig,
    run_id: str,
    started_at: datetime,
    ended_at: datetime,
) -> dict[str, Any]:
    return {
        "schema_version": EVALUATION_SAMPLING_REPORT_SCHEMA_VERSION,
        "sampling_schema_version": EVALUATION_SAMPLING_FRAME_SCHEMA_VERSION,
        "command": "evaluation.build_sampling_frame",
        "run_id": run_id,
        "pid": os.getpid(),
        "git_sha": current_git_sha(),
        "status": "complete",
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "elapsed_seconds": max(0.0, (ended_at - started_at).total_seconds()),
        "network_requests": 0,
        "rows_in": frame.height,
        "rows_out": frame.height,
        "duplicate_rows_removed": 0,
        "errors": 0,
        "config_fingerprint": config.fingerprint,
        "target_text_term_count": len(config.target_text_terms),
        "scored_count": frame.filter(pl.col("initial_score_status") == "scored").height,
        "unscored_count": frame.filter(
            pl.col("initial_score_status") == "not_scored"
        ).height,
        "no_geo_count": frame.filter(pl.col("no_geo")).height,
        "counts_by_query_tier": _counts_with_zeros(
            frame["primary_query_tier"],
            values=QUERY_TIERS,
        ),
        "counts_by_geo_stratum": _counts(frame["geo_stratum"]),
        "counts_by_yoloe_route": _counts(frame["yoloe_route"]),
        "counts_by_visual_domain": _counts(frame["initial_visual_domain"]),
        "counts_by_reference_score_tail": _counts(
            frame["initial_reference_score_tail"]
        ),
        "counts_by_disagreement": _counts(frame["text_image_reference_disagreement"]),
        "inputs": {
            name: _file_artifact(path) for name, path in sorted(input_paths.items())
        },
        "artifact": {
            **_file_artifact(frame_path),
            "row_count": frame.height,
        },
    }


def _sampling_report_markdown(report: Mapping[str, Any]) -> str:
    artifact = report["artifact"]
    return "\n".join(
        [
            "# Evaluation sampling frame",
            "",
            f"- Status: `{report['status']}`",
            f"- Run ID: `{report['run_id']}`",
            f"- Rows: `{report['rows_out']}`",
            f"- Scored: `{report['scored_count']}`",
            f"- Unscored: `{report['unscored_count']}`",
            f"- No geo: `{report['no_geo_count']}`",
            f"- Artifact: `{artifact['path']}`",
            f"- SHA-256: `{artifact['sha256']}`",
            "",
        ]
    )


def _counts(series: pl.Series) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in series.to_list()).items()))


def _counts_with_zeros(
    series: pl.Series,
    *,
    values: Sequence[str],
) -> dict[str, int]:
    counts = Counter(str(value) for value in series.to_list())
    return {value: counts.get(value, 0) for value in values}


def _file_artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "byte_count": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_token_text(value: object) -> str:
    return " ".join(
        "".join(
            character.casefold() if character.isalnum() else " "
            for character in str(value or "")
        ).split()
    )


def _required_file(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"input path does not exist: {path}")
    return path


def _require_frame(value: object, *, label: str) -> None:
    if not isinstance(value, pl.DataFrame):
        raise TypeError(f"{label} must be a Polars DataFrame")


def _require_columns(
    frame: pl.DataFrame,
    required: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} are missing required columns: {missing}")


def _require_nonblank(frame: pl.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if frame.filter(
            pl.col(column).is_null()
            | (pl.col(column).cast(pl.String).str.strip_chars() == "")
        ).height:
            raise ValueError(f"column {column} cannot contain blank values")


def _require_unique(
    frame: pl.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> None:
    if frame.select(columns).n_unique() != frame.height:
        raise ValueError(f"{label} rows must be unique by {list(columns)}")


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} cannot be blank")
    return text


def _required_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be an integer") from exc
    return integer


def _log_event(event: str, **fields: object) -> None:
    _LOGGER.info(
        json.dumps(
            {"event": event, **fields},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
    )


__all__ = [
    "EVALUATION_SAMPLING_FRAME_SCHEMA",
    "EVALUATION_SAMPLING_FRAME_SCHEMA_VERSION",
    "EVALUATION_SAMPLING_REPORT_SCHEMA_VERSION",
    "PAPILIO_DEMOLEUS_EVALUATION_SAMPLING_FRAME_FILE",
    "QUERY_DEFINITION_SCHEMA",
    "QUERY_PROVENANCE_SCHEMA",
    "QUERY_TIERS",
    "EvaluationSamplingConfig",
    "EvaluationSamplingPublication",
    "build_evaluation_sampling_frame",
    "empty_evaluation_sampling_frame",
    "materialize_evaluation_sampling_frame",
    "read_flickr_query_definitions",
    "validate_evaluation_sampling_frame",
]
