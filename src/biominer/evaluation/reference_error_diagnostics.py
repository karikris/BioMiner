"""Descriptive links between model errors and reference-bank diagnostics."""

from __future__ import annotations

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint


REFERENCE_ERROR_DIAGNOSTIC_SCHEMA = {
    "target_species": pl.String,
    "competitor_species": pl.String,
    "region": pl.String,
    "route": pl.String,
    "metric_status": pl.String,
    "model_error_rate": pl.Float64,
    "prototype_dispersion_mean": pl.Float64,
    "prototype_dispersion_max": pl.Float64,
    "reference_outlier_count": pl.UInt64,
    "high_influence_reference_count": pl.UInt64,
    "route_mismatch_reference_count": pl.UInt64,
    "route_imbalance_ratio": pl.Float64,
    "provider_dataset_count": pl.UInt64,
    "largest_provider_dataset_fraction": pl.Float64,
    "geographic_support_cluster_count": pl.UInt64,
    "geographic_support_gap": pl.Boolean,
    "target_reference_count": pl.UInt64,
    "competitor_reference_count": pl.UInt64,
    "target_to_competitor_reference_ratio": pl.Float64,
    "observer_count": pl.UInt64,
    "observer_diversity_ratio": pl.Float64,
    "prioritization_evidence": pl.Boolean,
    "reference_identity_conclusion": pl.String,
    "diagnostic_fingerprint": pl.String,
}


def relate_errors_to_reference_quality(
    performance: pl.DataFrame,
    reference_diagnostics: pl.DataFrame,
    prototypes: pl.DataFrame,
    support_manifest: pl.DataFrame,
    *,
    underperforming_species: tuple[str, ...],
    high_influence_threshold: float = 0.1,
) -> pl.DataFrame:
    """Describe reference-bank properties for explicitly underperforming species."""

    flagged = tuple(sorted({str(value).strip() for value in underperforming_species}))
    if not flagged or any(not value for value in flagged):
        return pl.DataFrame(schema=REFERENCE_ERROR_DIAGNOSTIC_SCHEMA)
    _require_columns(
        performance,
        {
            "target_species",
            "competitor_species",
            "region",
            "route",
            "metric_status",
            "false_positive_rate",
            "false_negative_rate",
        },
        artifact="performance",
    )
    _require_columns(
        reference_diagnostics,
        {
            "species",
            "route",
            "embedding_outlier_score",
            "review_threshold",
            "prototype_influence",
            "route_domain_mismatch",
            "taxon_misidentification_conclusion",
        },
        artifact="reference diagnostics",
    )
    _require_columns(
        prototypes,
        {"species", "route", "dispersion"},
        artifact="prototypes",
    )
    _require_columns(
        support_manifest,
        {
            "scientific_name",
            "route",
            "provider_dataset_key",
            "geo_cluster_id",
            "observer_id",
            "support_eligible",
        },
        artifact="support manifest",
    )
    if reference_diagnostics.filter(
        pl.col("taxon_misidentification_conclusion") != "not_assessed"
    ).height:
        raise ValueError("reference diagnostics must not assert misidentification")

    output: list[dict[str, object]] = []
    selected = performance.filter(pl.col("target_species").is_in(flagged))
    for performance_row in selected.iter_rows(named=True):
        species = str(performance_row["target_species"])
        route = str(performance_row["route"])
        competitor = str(performance_row["competitor_species"])
        diagnostics = reference_diagnostics.filter(
            (pl.col("species") == species) & (pl.col("route") == route)
        )
        prototype_rows = prototypes.filter(
            (pl.col("species") == species) & (pl.col("route") == route)
        )
        target_support = support_manifest.filter(
            pl.col("support_eligible")
            & (pl.col("scientific_name") == species)
            & (pl.col("route") == route)
        )
        competitor_support = support_manifest.filter(
            pl.col("support_eligible")
            & (pl.col("scientific_name") == competitor)
            & (pl.col("route") == route)
        )
        all_species_support = support_manifest.filter(
            pl.col("support_eligible") & (pl.col("scientific_name") == species)
        )
        route_counts = all_species_support.group_by("route").len()["len"].to_list()
        target_count = target_support.height
        competitor_count = competitor_support.height
        dataset_counts = target_support.group_by("provider_dataset_key").len()
        observer_count = target_support["observer_id"].drop_nulls().n_unique()
        geo_values = set(target_support["geo_cluster_id"].drop_nulls().to_list())
        region = str(performance_row["region"])
        fpr = performance_row["false_positive_rate"]
        fnr = performance_row["false_negative_rate"]
        base: dict[str, object] = {
            "target_species": species,
            "competitor_species": competitor,
            "region": region,
            "route": route,
            "metric_status": performance_row["metric_status"],
            "model_error_rate": _mean_optional(fpr, fnr),
            "prototype_dispersion_mean": _series_mean(prototype_rows, "dispersion"),
            "prototype_dispersion_max": _series_max(prototype_rows, "dispersion"),
            "reference_outlier_count": diagnostics.filter(
                pl.col("embedding_outlier_score") >= pl.col("review_threshold")
            ).height,
            "high_influence_reference_count": diagnostics.filter(
                pl.col("prototype_influence").is_not_null()
                & (pl.col("prototype_influence") >= high_influence_threshold)
            ).height,
            "route_mismatch_reference_count": diagnostics.filter(
                pl.col("route_domain_mismatch")
            ).height,
            "route_imbalance_ratio": (
                (max(route_counts) - min(route_counts)) / max(route_counts)
                if len(route_counts) > 1 and max(route_counts) > 0
                else 0.0
            ),
            "provider_dataset_count": dataset_counts.height,
            "largest_provider_dataset_fraction": (
                float(dataset_counts["len"].max()) / target_count
                if target_count
                else None
            ),
            "geographic_support_cluster_count": len(geo_values),
            "geographic_support_gap": region not in {"global", *geo_values},
            "target_reference_count": target_count,
            "competitor_reference_count": competitor_count,
            "target_to_competitor_reference_ratio": (
                target_count / competitor_count if competitor_count else None
            ),
            "observer_count": observer_count,
            "observer_diversity_ratio": (
                observer_count / target_count if target_count else None
            ),
            "prioritization_evidence": True,
            "reference_identity_conclusion": "not_assessed",
            "diagnostic_fingerprint": "",
        }
        payload = dict(base)
        payload.pop("diagnostic_fingerprint")
        base["diagnostic_fingerprint"] = canonical_semantic_fingerprint(payload)
        output.append(base)
    return pl.DataFrame(
        output,
        schema=REFERENCE_ERROR_DIAGNOSTIC_SCHEMA,
        orient="row",
        strict=True,
    ).sort("target_species", "region", "route", "competitor_species")


def _require_columns(
    frame: pl.DataFrame,
    required: set[str],
    *,
    artifact: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{artifact} missing columns: {missing}")


def _mean_optional(*values: object) -> float | None:
    available = [float(value) for value in values if value is not None]
    return sum(available) / len(available) if available else None


def _series_mean(frame: pl.DataFrame, column: str) -> float | None:
    value = frame[column].mean() if frame.height else None
    return float(value) if value is not None else None


def _series_max(frame: pl.DataFrame, column: str) -> float | None:
    value = frame[column].max() if frame.height else None
    return float(value) if value is not None else None


__all__ = [
    "REFERENCE_ERROR_DIAGNOSTIC_SCHEMA",
    "relate_errors_to_reference_quality",
]
