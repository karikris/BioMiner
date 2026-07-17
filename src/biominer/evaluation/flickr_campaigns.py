"""Provenance-preserving Flickr human-verification campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


CAMPAIGN_FILTERS: tuple[tuple[str, str], ...] = (
    ("representative_quality_audit", "representative_quality_audit_candidate"),
    ("high_confidence_candidate_review", "high_confidence_candidate"),
    ("failure_discovery", "failure_discovery_candidate"),
    ("conflict_adjudication", "conflict_adjudication_required"),
)

PROVENANCE_COLUMNS: tuple[str, ...] = (
    "source_record_id",
    "inclusion_probability",
    "sampling_stratum",
    "owner_group",
    "observation_group",
    "duplicate_group",
    "geographic_cluster",
    "query_tier",
    "score_band",
    "candidate_competitors",
)


@dataclass(frozen=True, slots=True)
class FlickrVerificationCampaigns:
    representative_quality_audit: pl.DataFrame
    high_confidence_candidate_review: pl.DataFrame
    failure_discovery: pl.DataFrame
    conflict_adjudication: pl.DataFrame

    def write_parquet(self, output_dir: str | Path) -> dict[str, Path]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}
        for campaign_name, _ in CAMPAIGN_FILTERS:
            output_path = root / f"{campaign_name}.parquet"
            getattr(self, campaign_name).write_parquet(output_path)
            outputs[campaign_name] = output_path
        return outputs


def build_flickr_verification_campaigns(
    candidates: pl.DataFrame,
) -> FlickrVerificationCampaigns:
    """Route candidates to one or more campaigns without losing sample design."""

    required = {
        *PROVENANCE_COLUMNS,
        *(flag for _, flag in CAMPAIGN_FILTERS),
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Flickr campaign candidates missing columns: {missing}")
    if candidates["source_record_id"].null_count() or candidates["source_record_id"].n_unique() != candidates.height:
        raise ValueError("source_record_id must be nonnull and unique")
    if candidates["inclusion_probability"].null_count() or candidates.filter(
        ~pl.col("inclusion_probability").is_between(0.0, 1.0, closed="right")
    ).height:
        raise ValueError("inclusion_probability must be within (0, 1]")
    if candidates["candidate_competitors"].dtype != pl.List(pl.String):
        raise ValueError("candidate_competitors must be List(String)")
    if candidates.filter(
        ~pl.any_horizontal(pl.col(flag) for _, flag in CAMPAIGN_FILTERS)
    ).height:
        raise ValueError("every candidate must belong to at least one campaign")

    campaigns: dict[str, pl.DataFrame] = {}
    for campaign_name, flag in CAMPAIGN_FILTERS:
        campaigns[campaign_name] = (
            candidates.filter(pl.col(flag))
            .with_columns(pl.lit(campaign_name).alias("verification_campaign"))
            .sort(
                ["sampling_stratum", "inclusion_probability", "source_record_id"],
                descending=[False, True, False],
            )
        )
    return FlickrVerificationCampaigns(**campaigns)


__all__ = [
    "CAMPAIGN_FILTERS",
    "PROVENANCE_COLUMNS",
    "FlickrVerificationCampaigns",
    "build_flickr_verification_campaigns",
]
