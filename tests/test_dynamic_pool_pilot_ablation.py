"""Tests for the bounded pilot family/geography candidate ablation."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import polars as pl
import pytest

from biominer.bioclip.family_geo_candidates import (
    validate_family_geo_candidate_sets,
)
from biominer.evaluation.dynamic_pool_pilot_ablation import (
    DYNAMIC_POOL_PILOT_ABLATION_REPORT_FILE,
    build_dynamic_pool_pilot_candidate_ablation,
    build_dynamic_pool_pilot_candidate_ablation_report,
    build_pilot_family_geo_candidate_sets,
    validate_dynamic_pool_pilot_candidate_ablation,
    validate_dynamic_pool_pilot_candidate_ablation_report,
    write_dynamic_pool_pilot_candidate_ablation_report,
)
from biominer.evaluation.dynamic_pool_pilot_plan import (
    PILOT_CANDIDATE_STRATEGIES,
    load_dynamic_pool_pilot_plan,
)


ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "config/pilot/geography_conditioned_dynamic_pool_pilot_v1.json"
REPORT_PATH = ROOT / "reports/geo_dynamic_pooling/pilot/candidate_pooling_ablation.json"


def _plan() -> dict[str, object]:
    return load_dynamic_pool_pilot_plan(PLAN_PATH, repository_root=ROOT)


def test_candidate_fixture_uses_complete_production_union_contract() -> None:
    plan = _plan()
    frame = build_pilot_family_geo_candidate_sets(plan)

    validate_family_geo_candidate_sets(frame)
    assert frame.height == 35
    assert frame["candidate_set_id"].n_unique() == 7
    assert set(frame["candidate_accepted_taxon_key"]) == {
        taxon["accepted_taxon_key"] for taxon in plan["taxon_catalog"]
    }
    for (_set_id,), group in frame.group_by("candidate_set_id"):
        assert group.height == 5
        assert group["target_candidate"].sum() == 1
        assert all(group["target_preserved"])
        assert all(group["included_in_complete_union"])
        assert not any(group["family_changed_membership"])


def test_no_geo_case_has_no_fabricated_geographic_evidence() -> None:
    plan = _plan()
    no_geo = next(
        case
        for case in plan["cases"]
        if case["geographic_evidence_status"] == "missing_source_geography"
    )
    frame = build_pilot_family_geo_candidate_sets(plan).filter(
        pl.col("flickr_photo_id") == no_geo["fixture_media_id"]
    )

    assert set(frame["geographic_evidence_status"]) == {"unavailable"}
    assert set(frame["geographic_evidence_reason"]) == {
        "missing_source_geography_global_only"
    }
    assert frame["geographic_scopes"].to_list() == [[]] * frame.height
    assert frame["geographic_evidence_score"].null_count() == frame.height
    assert frame["occurrence_support"].sum() == 0


def test_all_strategies_preserve_target_and_identical_membership() -> None:
    plan = _plan()
    frame = build_dynamic_pool_pilot_candidate_ablation(plan)

    validate_dynamic_pool_pilot_candidate_ablation(frame, plan)
    assert frame.height == 21
    assert set(frame["strategy_name"]) == set(PILOT_CANDIDATE_STRATEGIES)
    assert frame["target_preserved"].sum() == 21
    assert frame["complete_union_preserved"].sum() == 21
    assert set(frame["candidate_set_size"]) == {5}
    assert set(frame["classification_accuracy_status"]) == {"unavailable_fixture_only"}
    assert set(frame["timing_status"]) == {"not_instrumented"}
    assert not any(frame["production_default_eligible"])
    for (_case_id,), group in frame.group_by("case_id"):
        assert group["membership_fingerprint"].n_unique() == 1
        assert group["candidate_set_id"].n_unique() == 1


def test_structural_metrics_expose_order_tradeoff_without_accuracy_claim() -> None:
    plan = _plan()
    frame = build_dynamic_pool_pilot_candidate_ablation(plan)
    report = build_dynamic_pool_pilot_candidate_ablation_report(plan, frame)
    metrics = {row["strategy_name"]: row for row in report["strategy_metrics"]}

    assert metrics["geography_first"]["target_candidate_recall_at_1"] == 1.0
    assert (
        metrics["parallel_family_geography_union"]["target_candidate_recall_at_1"]
        == 1.0
    )
    assert metrics["family_first_safe"]["target_candidate_recall_at_1"] == (
        pytest.approx(1 / 7)
    )
    assert metrics["family_first_safe"]["target_candidate_recall_at_3"] == 1.0
    assert metrics["family_first_safe"]["maximum_target_rank"] == 3
    assert all(row["target_candidate_recall_at_5"] == 1.0 for row in metrics.values())
    assert all(
        row["accuracy_interpretation"] == "unavailable_fixture_only"
        for row in metrics.values()
    )
    assert report["classification_accuracy"]["status"] == "unavailable"
    assert report["timing"]["status"] == "not_instrumented"
    assert report["selection"]["status"] == "insufficient_evidence"
    assert report["selection"]["selected_candidate_strategy"] is None
    assert report["selection"]["production_default_eligible"] is False


def test_ablation_is_deterministic_and_result_bound() -> None:
    plan = _plan()
    first = build_dynamic_pool_pilot_candidate_ablation(plan)
    second = build_dynamic_pool_pilot_candidate_ablation(plan)

    assert first.equals(second)
    assert first["result_fingerprint"].n_unique() == first.height
    tampered = first.with_columns(
        pl.when(pl.col("strategy_name") == "family_first_safe")
        .then(pl.lit(1.0))
        .otherwise(pl.col("target_candidate_recall_at_1"))
        .alias("target_candidate_recall_at_1")
    )
    with pytest.raises(ValueError, match="recall is inconsistent"):
        validate_dynamic_pool_pilot_candidate_ablation(tampered, plan)


def test_report_round_trip_and_committed_evidence_match(tmp_path: Path) -> None:
    plan = _plan()
    frame = build_dynamic_pool_pilot_candidate_ablation(plan)
    report = build_dynamic_pool_pilot_candidate_ablation_report(plan, frame)

    validate_dynamic_pool_pilot_candidate_ablation_report(report, plan, frame)
    output = write_dynamic_pool_pilot_candidate_ablation_report(
        report, plan, frame, tmp_path
    )
    assert output.name == DYNAMIC_POOL_PILOT_ABLATION_REPORT_FILE
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert json.loads(REPORT_PATH.read_text(encoding="utf-8")) == report


def test_report_rejects_metric_or_authority_tampering() -> None:
    plan = _plan()
    frame = build_dynamic_pool_pilot_candidate_ablation(plan)
    report = build_dynamic_pool_pilot_candidate_ablation_report(plan, frame)
    changes = (
        ("selection", "production_default_eligible", True),
        ("selection", "selected_candidate_strategy", "geography_first"),
        ("classification_accuracy", "status", "available"),
        ("scientific_claims", "occurrence_release_authorized", True),
    )
    for section, field, value in changes:
        tampered = deepcopy(report)
        tampered[section][field] = value
        with pytest.raises(ValueError):
            validate_dynamic_pool_pilot_candidate_ablation_report(tampered, plan, frame)
