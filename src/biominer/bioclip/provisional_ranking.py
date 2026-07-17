"""Nonparametric provisional ranking over admitted reference evidence."""

from __future__ import annotations

from collections.abc import Sequence
from math import fsum, isfinite, sqrt
from pathlib import Path

import polars as pl

from biominer.bioclip.provisional_prototypes import (
    validate_robust_provisional_prototypes,
)
from biominer.bioclip.reference_embeddings import validate_reference_embeddings
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.storage.parquet import write_parquet


PROVISIONAL_REFERENCE_RANKING_SCHEMA_VERSION = (
    "provisional-reference-ranking-v1.0.0"
)
PROVISIONAL_REFERENCE_RANKING_FILE = "provisional_reference_ranking.parquet"
PROVISIONAL_SCORE_SEMANTICS = (
    "uncalibrated_reference_similarity_and_margin_not_probability"
)


def provisional_reference_ranking_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "query_id": pl.String,
        "candidate_rank": pl.UInt32,
        "accepted_taxon_key": pl.String,
        "species": pl.String,
        "route": pl.String,
        "visual_domain": pl.String,
        "prototype_method": pl.String,
        "prototype_similarity": pl.Float64,
        "nearest_reference_media_id": pl.String,
        "nearest_reference_similarity": pl.Float64,
        "top_reference_media_ids": pl.List(pl.String),
        "top_reference_similarities": pl.List(pl.Float64),
        "top_k_reference_mean": pl.Float64,
        "provisional_score": pl.Float64,
        "nearest_competing_taxon_key": pl.String,
        "nearest_competing_score": pl.Float64,
        "raw_competitor_margin": pl.Float64,
        "geography_compatible": pl.Boolean,
        "domain_compatible": pl.Boolean,
        "provisional_decision_state": pl.String,
        "probability_available": pl.Boolean,
        "calibrated_target_probability": pl.Float64,
        "required_human_review_state": pl.String,
        "score_semantics": pl.String,
        "reference_admission_mode": pl.String,
        "admission_policy_fingerprint": pl.String,
        "model_fingerprint": pl.String,
        "reference_embedding_fingerprint": pl.String,
        "support_manifest_fingerprint": pl.String,
        "ranking_fingerprint": pl.String,
    }


def provisional_reference_ranking(
    *,
    query_id: str,
    query_embedding: Sequence[float],
    query_route: str,
    query_visual_domain: str,
    reference_embeddings: pl.DataFrame,
    prototypes: pl.DataFrame,
    query_geo_cluster_id: str | None = None,
    prototype_method: str = "trimmed_mean",
    top_k: int = 3,
) -> pl.DataFrame:
    """Rank candidates with raw evidence and no probability claim."""

    query = str(query_id).strip()
    if not query:
        raise ValueError("query_id must be nonblank")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    validate_reference_embeddings(reference_embeddings)
    validate_robust_provisional_prototypes(prototypes)
    selected_prototypes = prototypes.filter(
        (pl.col("route") == query_route)
        & (pl.col("prototype_method") == prototype_method)
    )
    support = reference_embeddings.filter(
        (pl.col("support_split") == "support_train")
        & (pl.col("route") == query_route)
    )
    if selected_prototypes.is_empty() or support.is_empty():
        raise ValueError("provisional ranking lacks compatible prototype support")
    admission_mode = _single(selected_prototypes, "reference_admission_mode")
    if admission_mode != "adaptive_gbif_fast_start":
        raise ValueError("provisional ranking requires adaptive admission evidence")
    admission_fingerprint = _single(
        selected_prototypes,
        "admission_policy_fingerprint",
    )
    if set(support["admission_policy_fingerprint"].to_list()) != {
        admission_fingerprint
    }:
        raise ValueError("prototype and reference admission policies disagree")
    model_fingerprint = _single(selected_prototypes, "model_fingerprint")
    support_fingerprint = _single(
        selected_prototypes,
        "support_manifest_fingerprint",
    )
    embedding_fingerprint = _single(
        selected_prototypes,
        "reference_embedding_fingerprint",
    )
    vector = _unit(query_embedding)
    candidates: list[dict[str, object]] = []
    taxon_keys = sorted(set(str(value) for value in selected_prototypes["accepted_taxon_key"]))
    for taxon_key in taxon_keys:
        prototype_rows = selected_prototypes.filter(
            pl.col("accepted_taxon_key") == taxon_key
        ).to_dicts()
        reference_rows = support.filter(
            pl.col("accepted_taxon_key") == taxon_key
        ).to_dicts()
        if not reference_rows:
            continue
        prototype_similarity = max(
            _dot(vector, row["embedding"]) for row in prototype_rows
        )
        ranked_references = sorted(
            (
                (_dot(vector, row["embedding"]), str(row["reference_media_id"]), row)
                for row in reference_rows
            ),
            key=lambda item: (-item[0], item[1]),
        )[:top_k]
        top_mean = fsum(item[0] for item in ranked_references) / len(
            ranked_references
        )
        score = (prototype_similarity + top_mean) / 2
        geo_compatible = (
            None
            if query_geo_cluster_id is None
            else any(
                row["geo_cluster_id"] == query_geo_cluster_id
                for row in reference_rows
            )
        )
        candidates.append(
            {
                "accepted_taxon_key": taxon_key,
                "species": str(prototype_rows[0]["species"]),
                "prototype_similarity": prototype_similarity,
                "top_references": ranked_references,
                "top_k_reference_mean": top_mean,
                "provisional_score": score,
                "geography_compatible": geo_compatible,
            }
        )
    if len(candidates) < 2:
        raise ValueError("provisional ranking requires target and competitor support")
    candidates.sort(
        key=lambda item: (
            -float(item["provisional_score"]),
            str(item["accepted_taxon_key"]),
        )
    )
    output: list[dict[str, object]] = []
    for rank, candidate in enumerate(candidates, start=1):
        competitor = max(
            (
                item
                for item in candidates
                if item["accepted_taxon_key"] != candidate["accepted_taxon_key"]
            ),
            key=lambda item: float(item["provisional_score"]),
        )
        references = candidate["top_references"]
        assert isinstance(references, list)
        base: dict[str, object] = {
            "schema_version": PROVISIONAL_REFERENCE_RANKING_SCHEMA_VERSION,
            "query_id": query,
            "candidate_rank": rank,
            "accepted_taxon_key": candidate["accepted_taxon_key"],
            "species": candidate["species"],
            "route": query_route,
            "visual_domain": query_visual_domain,
            "prototype_method": prototype_method,
            "prototype_similarity": candidate["prototype_similarity"],
            "nearest_reference_media_id": references[0][1],
            "nearest_reference_similarity": references[0][0],
            "top_reference_media_ids": [item[1] for item in references],
            "top_reference_similarities": [item[0] for item in references],
            "top_k_reference_mean": candidate["top_k_reference_mean"],
            "provisional_score": candidate["provisional_score"],
            "nearest_competing_taxon_key": competitor["accepted_taxon_key"],
            "nearest_competing_score": competitor["provisional_score"],
            "raw_competitor_margin": float(candidate["provisional_score"])
            - float(competitor["provisional_score"]),
            "geography_compatible": candidate["geography_compatible"],
            "domain_compatible": all(
                row[2]["visual_domain"] == query_visual_domain
                for row in references
            ),
            "provisional_decision_state": (
                "provisional_lead_requires_human_review"
                if rank == 1
                else "provisional_alternative"
            ),
            "probability_available": False,
            "calibrated_target_probability": None,
            "required_human_review_state": "mandatory_before_final_inclusion",
            "score_semantics": PROVISIONAL_SCORE_SEMANTICS,
            "reference_admission_mode": admission_mode,
            "admission_policy_fingerprint": admission_fingerprint,
            "model_fingerprint": model_fingerprint,
            "reference_embedding_fingerprint": embedding_fingerprint,
            "support_manifest_fingerprint": support_fingerprint,
            "ranking_fingerprint": "",
        }
        payload = dict(base)
        payload.pop("ranking_fingerprint")
        base["ranking_fingerprint"] = canonical_semantic_fingerprint(payload)
        output.append(base)
    result = pl.DataFrame(
        output,
        schema=provisional_reference_ranking_schema(),
        orient="row",
        strict=True,
    ).sort("candidate_rank", "accepted_taxon_key")
    validate_provisional_reference_ranking(result)
    return result


def validate_provisional_reference_ranking(frame: pl.DataFrame) -> None:
    if not isinstance(frame, pl.DataFrame) or frame.schema != (
        provisional_reference_ranking_schema()
    ):
        raise ValueError("provisional reference ranking schema mismatch")
    if frame.is_empty() or frame["candidate_rank"].to_list() != list(
        range(1, frame.height + 1)
    ):
        raise ValueError("provisional candidate ranks are incomplete")
    for row in frame.iter_rows(named=True):
        if (
            row["schema_version"] != PROVISIONAL_REFERENCE_RANKING_SCHEMA_VERSION
            or row["probability_available"]
            or row["calibrated_target_probability"] is not None
            or row["score_semantics"] != PROVISIONAL_SCORE_SEMANTICS
            or row["required_human_review_state"]
            != "mandatory_before_final_inclusion"
        ):
            raise ValueError("provisional ranking scientific semantics are invalid")
        for field in (
            "prototype_similarity",
            "nearest_reference_similarity",
            "top_k_reference_mean",
            "provisional_score",
            "nearest_competing_score",
            "raw_competitor_margin",
        ):
            if not isfinite(float(row[field])):
                raise ValueError(f"provisional ranking {field} must be finite")
        payload = dict(row)
        fingerprint = payload.pop("ranking_fingerprint")
        if fingerprint != canonical_semantic_fingerprint(payload):
            raise ValueError("provisional ranking fingerprint mismatch")


def write_provisional_reference_ranking(
    frame: pl.DataFrame,
    output: str | Path,
) -> Path:
    validate_provisional_reference_ranking(frame)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= PROVISIONAL_REFERENCE_RANKING_FILE
    return write_parquet(frame, destination)


def _unit(values: Sequence[float]) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    norm = sqrt(fsum(value * value for value in vector))
    if not isfinite(norm) or norm <= 0:
        raise ValueError("query embedding must have non-zero finite norm")
    return tuple(value / norm for value in vector)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return fsum(
        float(a) * float(b) for a, b in zip(left, right, strict=True)
    )


def _single(frame: pl.DataFrame, field: str) -> str:
    values = frame[field].unique().to_list()
    if len(values) != 1 or not isinstance(values[0], str) or not values[0]:
        raise ValueError(f"{field} must have one nonblank value")
    return values[0]


__all__ = [
    "PROVISIONAL_REFERENCE_RANKING_FILE",
    "PROVISIONAL_REFERENCE_RANKING_SCHEMA_VERSION",
    "PROVISIONAL_SCORE_SEMANTICS",
    "provisional_reference_ranking",
    "provisional_reference_ranking_schema",
    "validate_provisional_reference_ranking",
    "write_provisional_reference_ranking",
]
