from __future__ import annotations

import polars as pl
import pytest

from biominer.evaluation.flickr_campaigns import (
    PROVENANCE_COLUMNS,
    build_flickr_verification_campaigns,
)


def _candidate_frame() -> pl.DataFrame:
    rows = []
    assignments = (
        ("representative", True, False, False, False),
        ("high", False, True, False, False),
        ("failure", False, False, True, False),
        ("conflict", False, False, False, True),
        ("overlap", True, False, True, False),
    )
    for index, (record_id, representative, high, failure, conflict) in enumerate(
        assignments
    ):
        rows.append(
            {
                "source_record_id": f"flickr:{record_id}",
                "inclusion_probability": 0.2 + index * 0.1,
                "sampling_stratum": f"stratum-{index % 2}",
                "owner_group": f"owner-{index % 2}",
                "observation_group": f"observation-{index}",
                "duplicate_group": f"duplicate-{index}",
                "geographic_cluster": f"geo-{index % 3}",
                "query_tier": f"tier-{index % 2}",
                "score_band": "high" if high else "other",
                "candidate_competitors": ["Papilio polytes", "Papilio cresphontes"],
                "representative_quality_audit_candidate": representative,
                "high_confidence_candidate": high,
                "failure_discovery_candidate": failure,
                "conflict_adjudication_required": conflict,
            }
        )
    return pl.DataFrame(rows)


def test_four_campaigns_preserve_sampling_and_group_provenance(tmp_path) -> None:
    campaigns = build_flickr_verification_campaigns(_candidate_frame())

    assert campaigns.representative_quality_audit.height == 2
    assert campaigns.high_confidence_candidate_review.height == 1
    assert campaigns.failure_discovery.height == 2
    assert campaigns.conflict_adjudication.height == 1
    for campaign_name in (
        "representative_quality_audit",
        "high_confidence_candidate_review",
        "failure_discovery",
        "conflict_adjudication",
    ):
        frame = getattr(campaigns, campaign_name)
        assert set(PROVENANCE_COLUMNS) <= set(frame.columns)
        assert set(frame["verification_campaign"]) == {campaign_name}
        assert frame["candidate_competitors"].dtype == pl.List(pl.String)

    outputs = campaigns.write_parquet(tmp_path)
    assert set(outputs) == {
        "representative_quality_audit",
        "high_confidence_candidate_review",
        "failure_discovery",
        "conflict_adjudication",
    }
    assert all(path.suffix == ".parquet" and path.exists() for path in outputs.values())


def test_campaign_membership_can_overlap_without_changing_inclusion_probability() -> None:
    campaigns = build_flickr_verification_campaigns(_candidate_frame())

    representative = campaigns.representative_quality_audit.filter(
        pl.col("source_record_id") == "flickr:overlap"
    ).row(0, named=True)
    failure = campaigns.failure_discovery.filter(
        pl.col("source_record_id") == "flickr:overlap"
    ).row(0, named=True)
    assert representative["inclusion_probability"] == failure["inclusion_probability"]
    assert representative["sampling_stratum"] == failure["sampling_stratum"]


def test_campaign_builder_rejects_missing_or_invalid_sample_design() -> None:
    frame = _candidate_frame()
    with pytest.raises(ValueError, match="missing columns"):
        build_flickr_verification_campaigns(frame.drop("owner_group"))
    with pytest.raises(ValueError, match="inclusion_probability"):
        build_flickr_verification_campaigns(
            frame.with_columns(pl.lit(0.0).alias("inclusion_probability"))
        )
    with pytest.raises(ValueError, match="at least one campaign"):
        build_flickr_verification_campaigns(
            frame.with_columns(
                *[
                    pl.lit(False).alias(name)
                    for name in (
                        "representative_quality_audit_candidate",
                        "high_confidence_candidate",
                        "failure_discovery_candidate",
                        "conflict_adjudication_required",
                    )
                ]
            )
        )
