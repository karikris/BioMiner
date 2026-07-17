"""Descriptive embedding diagnostics for provisional reference support."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from pathlib import Path

import polars as pl

from biominer.bioclip.reference_embeddings import (
    reference_embeddings_artifact_fingerprint,
    validate_reference_embeddings,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.readiness import reference_route_dimensions
from biominer.storage.parquet import write_parquet
from biominer.vision.full_frame_attention import RAW_FULL_IMAGE_KIND


REFERENCE_QUALITY_DIAGNOSTICS_SCHEMA_VERSION = (
    "reference-quality-diagnostics-v1.0.0"
)
REFERENCE_QUALITY_DIAGNOSTICS_FILE = "reference_quality_diagnostics.parquet"
REFERENCE_OUTLIER_SCORE_VERSION = "reference-outlier-score-v1"


@dataclass(frozen=True, slots=True)
class ReferenceQualityDiagnosticPolicy:
    review_threshold: float = 0.35

    def __post_init__(self) -> None:
        if (
            isinstance(self.review_threshold, bool)
            or not isinstance(self.review_threshold, int | float)
            or not isfinite(self.review_threshold)
            or not 0 <= self.review_threshold <= 2
        ):
            raise ValueError("review_threshold must be finite and in [0, 2]")
        object.__setattr__(self, "review_threshold", float(self.review_threshold))

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": "reference-quality-diagnostic-policy-v1",
                "review_threshold": self.review_threshold,
                "outlier_score_version": REFERENCE_OUTLIER_SCORE_VERSION,
            }
        )


def reference_quality_diagnostics_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "reference_media_id": pl.String,
        "reference_observation_id": pl.String,
        "accepted_taxon_key": pl.String,
        "species": pl.String,
        "route": pl.String,
        "visual_domain": pl.String,
        "identity_evidence_basis": pl.String,
        "reference_admission_mode": pl.String,
        "admission_policy_fingerprint": pl.String,
        "similarity_to_class_centroid": pl.Float64,
        "leave_one_out_centroid_similarity": pl.Float64,
        "nearest_same_species_reference_media_id": pl.String,
        "nearest_same_species_similarity": pl.Float64,
        "nearest_competing_reference_media_id": pl.String,
        "nearest_competing_taxon_key": pl.String,
        "nearest_competing_species_similarity": pl.Float64,
        "same_minus_competitor_margin": pl.Float64,
        "prototype_influence": pl.Float64,
        "route_domain_mismatch": pl.Boolean,
        "embedding_outlier_score": pl.Float64,
        "outlier_score_version": pl.String,
        "review_threshold": pl.Float64,
        "diagnostic_state": pl.String,
        "taxon_misidentification_conclusion": pl.String,
        "policy_fingerprint": pl.String,
        "model_fingerprint": pl.String,
        "reference_embedding_fingerprint": pl.String,
        "support_manifest_fingerprint": pl.String,
        "diagnostic_fingerprint": pl.String,
    }


def build_reference_quality_diagnostics(
    reference_embeddings: pl.DataFrame,
    *,
    policy: ReferenceQualityDiagnosticPolicy | None = None,
) -> pl.DataFrame:
    """Calculate peer evidence without inferring taxonomic correctness."""

    validate_reference_embeddings(reference_embeddings)
    active = policy or ReferenceQualityDiagnosticPolicy()
    provisional = reference_embeddings.filter(
        pl.col("provisional_support")
        & (pl.col("support_split") == "support_train")
        & (pl.col("visual_input_kind") == RAW_FULL_IMAGE_KIND)
    )
    if provisional.is_empty():
        return pl.DataFrame(schema=reference_quality_diagnostics_schema())
    source_rows = provisional.to_dicts()
    by_class: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in source_rows:
        by_class.setdefault(
            (str(row["accepted_taxon_key"]), str(row["route"])), []
        ).append(row)
    model_fingerprint = _single(provisional, "model_fingerprint")
    support_fingerprint = _single(provisional, "support_manifest_fingerprint")
    embedding_fingerprint = reference_embeddings_artifact_fingerprint(
        reference_embeddings
    )
    output: list[dict[str, object]] = []
    for row in sorted(source_rows, key=lambda item: str(item["reference_media_id"])):
        vector = _vector(row)
        class_key = (str(row["accepted_taxon_key"]), str(row["route"]))
        peers = by_class[class_key]
        centroid = _unit(_mean([_vector(item) for item in peers]))
        same = [
            item
            for item in peers
            if item["reference_media_id"] != row["reference_media_id"]
        ]
        competitors = [
            item
            for item in source_rows
            if item["accepted_taxon_key"] != row["accepted_taxon_key"]
            and item["route"] == row["route"]
        ]
        nearest_same = _nearest(vector, same)
        nearest_competitor = _nearest(vector, competitors)
        loo_centroid = (
            _unit(_mean([_vector(item) for item in same])) if same else None
        )
        centroid_similarity = _dot(vector, centroid)
        loo_similarity = (
            _dot(vector, loo_centroid) if loo_centroid is not None else None
        )
        same_similarity = nearest_same[0] if nearest_same is not None else None
        competitor_similarity = (
            nearest_competitor[0] if nearest_competitor is not None else None
        )
        margin = (
            same_similarity - competitor_similarity
            if same_similarity is not None and competitor_similarity is not None
            else None
        )
        influence = (
            1 - _dot(centroid, loo_centroid)
            if loo_centroid is not None
            else None
        )
        expected_life_stage, expected_domain = reference_route_dimensions(
            str(row["route"])
        )
        mismatch = (
            row["life_stage"] != expected_life_stage
            or row["visual_domain"] != expected_domain
        )
        components = [
            1 - (loo_similarity if loo_similarity is not None else centroid_similarity),
            max(0.0, -(margin or 0.0)),
            influence or 0.0,
            1.0 if mismatch else 0.0,
        ]
        outlier_score = fsum(components) / len(components)
        enough_peers = same_similarity is not None and competitor_similarity is not None
        diagnostic_state = (
            "insufficient_peers"
            if not enough_peers
            else "reference_quality_review_candidate"
            if outlier_score >= active.review_threshold
            else "within_configured_dispersion"
        )
        base: dict[str, object] = {
            "schema_version": REFERENCE_QUALITY_DIAGNOSTICS_SCHEMA_VERSION,
            "reference_media_id": row["reference_media_id"],
            "reference_observation_id": row["reference_observation_id"],
            "accepted_taxon_key": row["accepted_taxon_key"],
            "species": row["scientific_name"],
            "route": row["route"],
            "visual_domain": row["visual_domain"],
            "identity_evidence_basis": row["identity_evidence_basis"],
            "reference_admission_mode": row["reference_admission_mode"],
            "admission_policy_fingerprint": row["admission_policy_fingerprint"],
            "similarity_to_class_centroid": centroid_similarity,
            "leave_one_out_centroid_similarity": loo_similarity,
            "nearest_same_species_reference_media_id": (
                nearest_same[1]["reference_media_id"]
                if nearest_same is not None
                else None
            ),
            "nearest_same_species_similarity": same_similarity,
            "nearest_competing_reference_media_id": (
                nearest_competitor[1]["reference_media_id"]
                if nearest_competitor is not None
                else None
            ),
            "nearest_competing_taxon_key": (
                nearest_competitor[1]["accepted_taxon_key"]
                if nearest_competitor is not None
                else None
            ),
            "nearest_competing_species_similarity": competitor_similarity,
            "same_minus_competitor_margin": margin,
            "prototype_influence": influence,
            "route_domain_mismatch": mismatch,
            "embedding_outlier_score": outlier_score,
            "outlier_score_version": REFERENCE_OUTLIER_SCORE_VERSION,
            "review_threshold": active.review_threshold,
            "diagnostic_state": diagnostic_state,
            "taxon_misidentification_conclusion": "not_assessed",
            "policy_fingerprint": active.fingerprint,
            "model_fingerprint": model_fingerprint,
            "reference_embedding_fingerprint": embedding_fingerprint,
            "support_manifest_fingerprint": support_fingerprint,
            "diagnostic_fingerprint": "",
        }
        payload = dict(base)
        payload.pop("diagnostic_fingerprint")
        base["diagnostic_fingerprint"] = canonical_semantic_fingerprint(payload)
        output.append(base)
    result = pl.DataFrame(
        output,
        schema=reference_quality_diagnostics_schema(),
        orient="row",
        strict=True,
    ).sort("accepted_taxon_key", "route", "reference_media_id")
    validate_reference_quality_diagnostics(result)
    return result


def validate_reference_quality_diagnostics(frame: pl.DataFrame) -> None:
    if not isinstance(frame, pl.DataFrame) or frame.schema != (
        reference_quality_diagnostics_schema()
    ):
        raise ValueError("reference quality diagnostics schema mismatch")
    if frame["reference_media_id"].n_unique() != frame.height:
        raise ValueError("reference diagnostics repeat a reference media ID")
    for row in frame.iter_rows(named=True):
        if (
            row["schema_version"]
            != REFERENCE_QUALITY_DIAGNOSTICS_SCHEMA_VERSION
            or row["identity_evidence_basis"] != "gbif_provider_asserted"
            or row["taxon_misidentification_conclusion"] != "not_assessed"
        ):
            raise ValueError("reference diagnostic evidence semantics are invalid")
        for field in (
            "similarity_to_class_centroid",
            "embedding_outlier_score",
            "review_threshold",
        ):
            if not isfinite(float(row[field])):
                raise ValueError(f"reference diagnostic {field} must be finite")
        if not 0 <= float(row["embedding_outlier_score"]) <= 2:
            raise ValueError("reference diagnostic outlier score is invalid")
        payload = dict(row)
        fingerprint = payload.pop("diagnostic_fingerprint")
        if fingerprint != canonical_semantic_fingerprint(payload):
            raise ValueError("reference diagnostic fingerprint mismatch")


def write_reference_quality_diagnostics(
    frame: pl.DataFrame,
    output: str | Path,
) -> Path:
    validate_reference_quality_diagnostics(frame)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= REFERENCE_QUALITY_DIAGNOSTICS_FILE
    return write_parquet(frame, destination)


def _nearest(
    vector: Sequence[float],
    rows: Sequence[Mapping[str, object]],
) -> tuple[float, Mapping[str, object]] | None:
    if not rows:
        return None
    return max(
        ((_dot(vector, _vector(row)), row) for row in rows),
        key=lambda item: (item[0], str(item[1]["reference_media_id"])),
    )


def _vector(row: Mapping[str, object]) -> tuple[float, ...]:
    return tuple(float(value) for value in row["embedding"])  # type: ignore[arg-type]


def _mean(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    return tuple(
        fsum(vector[index] for vector in vectors) / len(vectors)
        for index in range(len(vectors[0]))
    )


def _unit(vector: Sequence[float]) -> tuple[float, ...]:
    norm = sqrt(fsum(value * value for value in vector))
    if not isfinite(norm) or norm <= 0:
        raise ValueError("diagnostic centroid has zero or invalid norm")
    return tuple(value / norm for value in vector)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return fsum(a * b for a, b in zip(left, right, strict=True))


def _single(frame: pl.DataFrame, field: str) -> str:
    values = frame[field].unique().to_list()
    if len(values) != 1 or not isinstance(values[0], str) or not values[0]:
        raise ValueError(f"{field} must contain one nonblank value")
    return values[0]


__all__ = [
    "REFERENCE_OUTLIER_SCORE_VERSION",
    "REFERENCE_QUALITY_DIAGNOSTICS_FILE",
    "REFERENCE_QUALITY_DIAGNOSTICS_SCHEMA_VERSION",
    "ReferenceQualityDiagnosticPolicy",
    "build_reference_quality_diagnostics",
    "reference_quality_diagnostics_schema",
    "validate_reference_quality_diagnostics",
    "write_reference_quality_diagnostics",
]
