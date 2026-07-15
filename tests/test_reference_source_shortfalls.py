from __future__ import annotations

import json

import polars as pl
import pytest

from biominer.references.source_shortfalls import (
    REFERENCE_SOURCE_SHORTFALL_FILE,
    REFERENCE_SOURCE_SHORTFALL_MARKDOWN_FILE,
    compile_reference_source_shortfalls,
    write_reference_source_shortfalls,
)


def _plan() -> dict[str, object]:
    return {
        "candidate_semantics": "source_taxon_match_not_human_verified_image_label",
        "queries": [
            {"accepted_taxon_key": "gbif:1"},
            {"accepted_taxon_key": "gbif:2"},
        ],
        "acquisition_quotas": {
            "target_adult": {
                "species": ["gbif:1"],
                "life_stage": "adult",
                "minimum_per_species": 2,
            },
            "selected_regional_competitors": {
                "species": ["gbif:2"],
                "life_stage": "adult",
                "minimum_per_species": 2,
            },
            "target_caterpillar": {
                "species": ["gbif:1"],
                "life_stage": "larva",
                "minimum_total": 1,
                "separate_from_adult_bank": True,
            },
            "other_insect_or_moth_negatives": {
                "minimum_total": 3,
                "status": "unresolved_requires_registry_linked_taxa",
            },
        },
    }


def _observations() -> pl.DataFrame:
    return pl.DataFrame(
        [
            _observation("adult-target", "gbif:1", "adult"),
            _observation("larva-target", "gbif:1", "larva"),
            _observation("adult-competitor", "gbif:2", "adult"),
            _observation("preserved", "gbif:1", "adult", preserved=True),
            _observation("conflict", "gbif:1", "adult", reconciliation="conflict"),
        ]
    )


def _observation(
    identity: str,
    key: str,
    life_stage: str,
    *,
    preserved: bool = False,
    reconciliation: str = "accepted_key_exact",
) -> dict[str, object]:
    return {
        "reference_observation_id": identity,
        "accepted_taxon_key": key,
        "taxon_reconciliation_status": reconciliation,
        "uncertain_taxon_match": False,
        "fossil": False,
        "occurrence_absent": False,
        "basis_of_record_suitable": True,
        "preserved_specimen": preserved,
        "life_stage": life_stage,
    }


def _media() -> pl.DataFrame:
    return pl.DataFrame(
        [
            _medium("media-adult-target", "adult-target", provider_accepted=True),
            _medium("media-larva-target", "larva-target"),
            _medium("media-adult-competitor", "adult-competitor"),
            _medium("media-preserved", "preserved"),
            _medium("media-conflict", "conflict"),
            _medium("media-denied", "adult-target", licence="denied"),
        ]
    )


def _medium(
    identity: str,
    observation: str,
    *,
    provider_accepted: bool = False,
    licence: str = "allowed",
) -> dict[str, object]:
    return {
        "reference_media_id": identity,
        "reference_observation_id": observation,
        "download_status": "pending",
        "verification_status": "accepted" if provider_accepted else "unreviewed",
        "licence_policy_status": licence,
    }


def test_compile_reports_stage_specific_candidate_and_review_shortfalls() -> None:
    report = compile_reference_source_shortfalls(
        query_plan=_plan(),
        observations=_observations(),
        media_candidates=_media(),
        created_at="2026-07-15T00:00:00Z",
    )
    groups = {row["group"]: row for row in report["groups"]}

    assert report["status"] == "awaiting_human_review_or_additional_sources"
    assert report["eligible_source_media_candidate_count"] == 3
    assert report["human_review_status"] == "not_available_at_metadata_stage"
    assert report["human_verified_source_media_count"] == 0
    assert groups["target_adult"]["source_candidate_media_count"] == 1
    assert groups["target_adult"]["source_candidate_shortfall"] == 1
    assert groups["target_adult"]["human_verified_shortfall"] == 2
    assert groups["target_caterpillar"]["source_candidate_shortfall"] == 0
    assert groups["target_caterpillar"]["human_verified_shortfall"] == 1
    assert groups["selected_regional_competitors"]["source_candidate_shortfall"] == 1
    assert groups["other_insect_or_moth_negatives"]["status"] == (
        "unresolved_requires_registry_linked_taxa"
    )
    assert groups["other_insect_or_moth_negatives"]["source_candidate_shortfall"] == 3


def test_compile_rejects_quota_for_unqueried_taxon() -> None:
    plan = _plan()
    plan["acquisition_quotas"]["target_adult"]["species"] = ["gbif:999"]

    with pytest.raises(ValueError, match="unqueried taxa"):
        compile_reference_source_shortfalls(
            query_plan=plan,
            observations=_observations(),
            media_candidates=_media(),
        )


def test_write_shortfall_json_and_markdown(tmp_path) -> None:
    query_plan = tmp_path / "queries.json"
    query_plan.write_text(json.dumps(_plan()), encoding="utf-8")
    report = compile_reference_source_shortfalls(
        query_plan=_plan(),
        observations=_observations(),
        media_candidates=_media(),
        query_plan_path=query_plan,
    )
    artifacts = write_reference_source_shortfalls(report, tmp_path / "output")

    assert artifacts["source_shortfalls"].name == REFERENCE_SOURCE_SHORTFALL_FILE
    assert (
        artifacts["source_shortfalls_markdown"].name
        == REFERENCE_SOURCE_SHORTFALL_MARKDOWN_FILE
    )
    restored = json.loads(
        artifacts["source_shortfalls"].read_text(encoding="utf-8")
    )
    assert restored["query_plan_sha256"].startswith("sha256:")
    markdown = artifacts["source_shortfalls_markdown"].read_text(encoding="utf-8")
    assert "Source taxon matches are acquisition candidates" in markdown
