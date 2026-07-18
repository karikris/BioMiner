"""Completeness and evidence-boundary checks for the final workflow report."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REPORT = ROOT / "reports/geo_dynamic_pooling/final_report.json"
SUMMARY = ROOT / "reports/geo_dynamic_pooling/final_report.md"
TECHNICAL = ROOT / "reports/geo_dynamic_pooling/release_technical_verification.json"
SEMANTICS = ROOT / "reports/geo_dynamic_pooling/release_scientific_semantics.json"
GITHITS = ROOT / "provenance/githits.jsonl"
TASK_PUSHES = ROOT / "provenance/task_pushes.jsonl"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_final_report_is_self_identifying_and_fail_closed() -> None:
    report = _json(REPORT)
    result = report["executive_result"]

    assert report["schema_version"] == "geo-dynamic-pooling-final-report-v1.0.0"
    assert report["goal_starting_sha"] == ("c7eaa9bf3696a25a0c8229837819dccec4fb9d66")
    assert report["pre_report_workflow_sha"] == (
        "98c64ec27e0aaa6aa3da333b3e4d37df3fc1c30b"
    )
    assert report["final_report_containing_commit"] == "self"
    assert result["production_selection_decision"] == "insufficient_evidence"
    assert result["eligible_variant_count"] == 0
    assert result["runtime_settings_changed"] is False
    assert result["selected_candidate_strategy"] is None
    assert result["selected_pool_variant"] is None
    assert result["selected_fusion_method"] is None
    assert result["occurrence_release_authorized"] is False


def test_all_minimum_artifacts_are_mapped_to_implemented_contracts() -> None:
    report = _json(REPORT)
    contracts = report["minimum_required_artifact_contracts"]
    groups = (
        "reference_indexing",
        "flickr_partitioning",
        "candidates_and_pools",
        "scores",
        "review_and_statistics",
        "outcome_lanes",
        "incremental_remediation",
        "product_handoffs",
    )
    rows = [row for group in groups for row in contracts[group]]

    assert len(rows) == 36
    assert all(row["requested"] and row["implemented"] for row in rows)
    requested = {row["requested"] for row in rows}
    assert {
        "reference_geography_index.parquet",
        "flickr_scoring_units.parquet",
        "dynamic_reference_pool_plans.parquet",
        "dynamic_pool_candidate_scores.parquet",
        "review_evidence_policy.json",
        "human_reviewed_release_candidates.parquet",
        "dynamic_pool_revision_impact.parquet",
        "taxalens_dynamic_pool_handoff.json",
        "butterflylens_dynamic_pool_handoff.json",
        "cross_repository_compatibility_report.json",
    } <= requested


def test_live_metrics_are_not_fabricated_from_fixture_evidence() -> None:
    metrics = _json(REPORT)["fixture_and_live_metrics"]

    assert metrics["reference_geography_index"]["live_reference_geography_rows"] is None
    assert metrics["flickr_and_embeddings"]["live_flickr_photos_processed"] is None
    assert metrics["pools"]["live_average_total_pool_size"] is None
    assert metrics["candidate_and_scoring"]["reviewed_species_candidate_recall"] is None
    assert metrics["candidate_and_scoring"]["live_dynamic_scoring_throughput"] is None
    assert metrics["candidate_and_scoring"]["live_peak_mps_memory_bytes"] is None
    assert (
        metrics["review_and_statistical_quality"]["estimated_reviewed_precision"]
        is None
    )
    assert (
        metrics["review_and_statistical_quality"]["decisive_human_reviews_completed"]
        == 0
    )
    assert metrics["outcomes"]["unreviewed_occurrence_exports"] == 0


def test_all_seventy_acceptance_criteria_are_grouped_with_maturity() -> None:
    acceptance = _json(REPORT)["acceptance_criteria"]
    groups = [value for key, value in acceptance.items() if key != "total"]

    assert acceptance["total"] == 70
    assert sum(len(group["criteria"]) for group in groups) == 70
    assert all(group["status"] for group in groups)
    assert "live_performance_unavailable" in acceptance["efficiency_11_20"]["status"]
    assert (
        "live_estimates_unavailable"
        in acceptance["statistical_quality_31_44"]["status"]
    )


def test_final_report_matches_release_receipts_and_provenance_ledgers() -> None:
    report = _json(REPORT)
    technical = _json(TECHNICAL)
    semantics = _json(SEMANTICS)
    records = [row for row in _jsonl(GITHITS) if row["task_id"].startswith("geo-pool-")]
    pushes = [
        row for row in _jsonl(TASK_PUSHES) if row["task_id"].startswith("geo-pool-")
    ]

    assert (
        report["release_verification"]["full_pytest"]["passed"]
        == technical["test_gates"][0]["passed"]
    )
    assert (
        report["release_verification"]["scientific_semantics"]["passed"]
        == semantics["gate"]["passed"]
    )
    assert report["githits_impact"]["records"] == len(records) == 139
    assert report["githits_impact"]["status_counts"] == dict(
        Counter(row["githits_status"] for row in records)
    )
    prior_pushes = [row for row in pushes if row["task_id"] != "geo-pool-16.2"]
    assert (
        len(prior_pushes)
        == report["workflow_scope"]["verified_task_pushes_before_final_task_push"]
        == 32
    )
    if len(pushes) > len(prior_pushes):
        assert (
            len(pushes)
            == report["workflow_scope"]["verified_task_pushes_after_task_closure"]
            == 33
        )
        final_push = next(row for row in pushes if row["task_id"] == "geo-pool-16.2")
        assert final_push["verified_remote_sha"] == (
            "ade7c17741decb8866ce885396b8f0142cdf7eea"
        )
    assert all(row["status"] == "verified" for row in pushes)


def test_markdown_summary_exposes_judge_and_operator_boundaries() -> None:
    summary = " ".join(SUMMARY.read_text(encoding="utf-8").split())

    required = (
        "software and fixture goal complete; live scientific work pending",
        "insufficient evidence",
        "Zero of 24 pilot variants are eligible",
        "Unreviewed occurrence exports | **0**",
        "All 70 requested acceptance criteria",
        "GitHits made no direct code contribution",
        "no further call was made",
        "Exact next action",
    )
    assert all(term in summary for term in required)
