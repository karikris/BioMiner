"""Tests for bounded dynamic-pool pilot review-work planning."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import polars as pl
import pytest

from biominer.evaluation.dynamic_pool_pilot_plan import (
    load_dynamic_pool_pilot_plan,
)
from biominer.evaluation.dynamic_pool_pilot_review import (
    DYNAMIC_POOL_PILOT_REVIEW_REPORT_FILE,
    build_dynamic_pool_pilot_review_plan,
    build_dynamic_pool_pilot_review_report,
    validate_dynamic_pool_pilot_review_plan,
    validate_dynamic_pool_pilot_review_report,
    write_dynamic_pool_pilot_review_report,
)
from biominer.evaluation.dynamic_pool_pilot_scoring import (
    execute_dynamic_pool_pilot_scoring,
)


ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "config/pilot/geography_conditioned_dynamic_pool_pilot_v1.json"
REPORT_PATH = ROOT / "reports/geo_dynamic_pooling/pilot/dynamic_pool_review_plan.json"


def _inputs():
    plan = load_dynamic_pool_pilot_plan(PLAN_PATH, repository_root=ROOT)
    scoring = execute_dynamic_pool_pilot_scoring(plan)
    return plan, scoring


def test_review_work_covers_fixture_without_creating_release_work() -> None:
    plan, scoring = _inputs()
    review = build_dynamic_pool_pilot_review_plan(plan, scoring)

    assert review.audit_frame.height == 7
    assert review.representative.population_count == 7
    assert review.representative.selected_count == 7
    assert review.targeted_queue.height == 7
    assert review.release_queue.height == 0
    assert set(review.representative.register["inclusion_probability"]) == {1.0}
    assert set(review.representative.register["sampling_weight"]) == {1.0}
    assert all(review.representative.register["representative_estimation_eligible"])


def test_targeted_queue_stays_outside_representative_and_release_evidence() -> None:
    plan, scoring = _inputs()
    queue = build_dynamic_pool_pilot_review_plan(plan, scoring).targeted_queue
    reasons = [reason for row in queue["priority_reasons"] for reason in row]

    assert reasons.count("high_pool_disagreement") == 6
    assert reasons.count("no_geo") == 1
    assert queue["inclusion_probability"].null_count() == 7
    assert queue["sampling_weight"].null_count() == 7
    assert not any(queue["representative_estimation_eligible"])
    assert not any(queue["release_authorized"])


def test_report_exposes_zero_outcomes_and_full_real_evidence_shortfall() -> None:
    plan, scoring = _inputs()
    review = build_dynamic_pool_pilot_review_plan(plan, scoring)
    report = build_dynamic_pool_pilot_review_report(plan, scoring, review)

    assert report["workload"] == {
        "unique_fixture_item_count": 7,
        "representative_and_targeted_overlap_count": 7,
        "purposes_merged": False,
        "reviewer_identity_count": 0,
        "assignment_count": 0,
        "completed_review_count": 0,
        "decisive_review_count": 0,
        "adjudication_count": 0,
    }
    assert report["production_evidence_gap"] == {
        "minimum_effective_reviewed_records": 86,
        "real_effective_reviewed_records": 0,
        "remaining_effective_review_shortfall": 86,
        "minimum_subgroup_independent_records": 30,
        "reviewed_precision_lower_bound": None,
        "reviewed_precision_status": "unavailable_no_completed_real_reviews",
        "statistical_support_status": "insufficient_evidence",
    }
    assert report["selection"]["status"] == "insufficient_evidence"
    assert report["selection"]["production_default_eligible"] is False
    assert report["release"]["release_authorized"] is False


def test_review_plan_is_deterministic_and_rejects_boundary_tampering() -> None:
    plan, scoring = _inputs()
    first = build_dynamic_pool_pilot_review_plan(plan, scoring)
    second = build_dynamic_pool_pilot_review_plan(plan, scoring)

    assert first.audit_frame.equals(second.audit_frame)
    assert first.representative.register.equals(second.representative.register)
    assert first.representative.sample.equals(second.representative.sample)
    assert first.targeted_queue.equals(second.targeted_queue)
    assert first.release_queue.equals(second.release_queue)
    assert first.review_plan_fingerprint == second.review_plan_fingerprint

    tampered = replace(
        first,
        targeted_queue=first.targeted_queue.with_columns(
            pl.lit(True).alias("representative_estimation_eligible")
        ),
    )
    with pytest.raises(ValueError, match="evidence boundary"):
        validate_dynamic_pool_pilot_review_plan(tampered, plan, scoring)


def test_report_round_trip_and_committed_evidence_match(tmp_path: Path) -> None:
    plan, scoring = _inputs()
    review = build_dynamic_pool_pilot_review_plan(plan, scoring)
    report = build_dynamic_pool_pilot_review_report(plan, scoring, review)

    validate_dynamic_pool_pilot_review_report(report, plan, scoring, review)
    output = write_dynamic_pool_pilot_review_report(
        report, plan, scoring, review, tmp_path
    )
    assert output.name == DYNAMIC_POOL_PILOT_REVIEW_REPORT_FILE
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert json.loads(REPORT_PATH.read_text(encoding="utf-8")) == report


def test_report_rejects_completed_review_or_release_claims() -> None:
    plan, scoring = _inputs()
    review = build_dynamic_pool_pilot_review_plan(plan, scoring)
    report = build_dynamic_pool_pilot_review_report(plan, scoring, review)

    for section, field, value in (
        ("workload", "completed_review_count", 7),
        ("selection", "production_default_eligible", True),
        ("release", "release_authorized", True),
    ):
        tampered = deepcopy(report)
        tampered[section][field] = value
        with pytest.raises(ValueError, match="differs from its plan"):
            validate_dynamic_pool_pilot_review_report(tampered, plan, scoring, review)
