"""Tests for the complete bounded-pilot production-selection table."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import polars as pl
import pytest

from biominer.evaluation.dynamic_pool_pilot_plan import (
    load_dynamic_pool_pilot_plan,
)
from biominer.evaluation.dynamic_pool_pilot_review import (
    build_dynamic_pool_pilot_review_plan,
)
from biominer.evaluation.dynamic_pool_pilot_scoring import (
    execute_dynamic_pool_pilot_scoring,
)
from biominer.evaluation.dynamic_pool_pilot_selection_ablation import (
    DYNAMIC_POOL_PILOT_SELECTION_ABLATION_REPORT_FILE,
    DYNAMIC_POOL_PILOT_SELECTION_ABLATION_TABLE_FILE,
    build_dynamic_pool_pilot_selection_ablation,
    build_dynamic_pool_pilot_selection_ablation_report,
    publish_dynamic_pool_pilot_selection_ablation_table,
    validate_dynamic_pool_pilot_selection_ablation,
    validate_dynamic_pool_pilot_selection_ablation_report,
    write_dynamic_pool_pilot_selection_ablation_report,
    write_dynamic_pool_pilot_selection_ablation_table,
)


ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "config/pilot/geography_conditioned_dynamic_pool_pilot_v1.json"
TABLE_PATH = (
    ROOT / "reports/geo_dynamic_pooling/pilot/production_selection_ablation.csv"
)


def _inputs():
    plan = load_dynamic_pool_pilot_plan(PLAN_PATH, repository_root=ROOT)
    scoring = execute_dynamic_pool_pilot_scoring(plan)
    review = build_dynamic_pool_pilot_review_plan(plan, scoring)
    return plan, scoring, review


def test_table_covers_every_candidate_pool_and_fusion_variant() -> None:
    plan, scoring, review = _inputs()
    frame = build_dynamic_pool_pilot_selection_ablation(plan, scoring, review)

    assert frame.height == 24
    assert (
        frame.select("candidate_strategy", "pool_variant", "fusion_method").n_unique()
        == 24
    )
    assert frame["candidate_strategy"].n_unique() == 3
    assert frame["pool_variant"].n_unique() == 2
    assert frame["fusion_method"].n_unique() == 4
    assert set(frame["case_count"]) == {7}
    assert set(frame["candidate_set_size"]) == {5}


def test_structural_recall_is_separate_from_unavailable_accuracy() -> None:
    plan, scoring, review = _inputs()
    frame = build_dynamic_pool_pilot_selection_ablation(plan, scoring, review)
    by_strategy = {
        strategy: group for (strategy,), group in frame.group_by("candidate_strategy")
    }

    assert set(by_strategy["geography_first"]["candidate_target_recall_at_1"]) == {1.0}
    assert set(
        by_strategy["parallel_family_geography_union"]["candidate_target_recall_at_1"]
    ) == {1.0}
    assert by_strategy["family_first_safe"]["candidate_target_recall_at_1"].item(
        0
    ) == pytest.approx(1 / 7)
    assert set(by_strategy["family_first_safe"]["candidate_target_recall_at_3"]) == {
        1.0
    }
    assert set(frame["classification_accuracy_status"]) == {"unavailable_fixture_only"}


def test_precision_subgroups_workload_and_mps_remain_denominator_explicit() -> None:
    plan, scoring, review = _inputs()
    frame = build_dynamic_pool_pilot_selection_ablation(plan, scoring, review)

    assert frame["reviewed_precision"].null_count() == 24
    assert frame["reviewed_precision_lower_bound"].null_count() == 24
    assert set(frame["effective_real_reviewed_records"]) == {0}
    assert set(frame["effective_review_shortfall"]) == {86}
    assert set(frame["minimum_subgroup_independent_records"]) == {30}
    assert set(frame["representative_fixture_work_count"]) == {7}
    assert set(frame["targeted_fixture_work_count"]) == {7}
    assert set(frame["completed_real_review_count"]) == {0}
    assert frame["mps_peak_memory_bytes"].null_count() == 24
    assert set(frame["mps_memory_status"]) == {
        "unavailable_cached_vector_fixture_not_mps"
    }


def test_observed_reuse_and_pruning_metrics_are_complete_not_guessed() -> None:
    plan, scoring, review = _inputs()
    frame = build_dynamic_pool_pilot_selection_ablation(plan, scoring, review)

    assert set(frame["distinct_score_work_count"]) == {7}
    assert set(frame["execution_query_embedding_reuse_event_count"]) == {7}
    assert set(frame["execution_within_batch_matrix_reuse_count"]) == {65}
    assert set(frame["execution_maximum_batch_pool_matrix_bytes"]) == {2240}
    assert set(
        frame.filter(pl.col("pool_variant") == "global_only_control")[
            "local_evidence_available_case_count"
        ]
    ) == {0}
    assert set(
        frame.filter(pl.col("pool_variant") == "dynamic_global_local")[
            "local_evidence_available_case_count"
        ]
    ) == {6}
    assert set(frame["target_pruning_regression_count"]) == {0}


def test_no_variant_is_eligible_and_report_covers_all_nine_criteria() -> None:
    plan, scoring, review = _inputs()
    frame = build_dynamic_pool_pilot_selection_ablation(plan, scoring, review)
    report = build_dynamic_pool_pilot_selection_ablation_report(
        plan, scoring, review, frame
    )

    assert not any(frame["production_default_eligible"])
    assert report["variant_count"] == 24
    assert len(report["selection_criteria"]) == 9
    assert report["selection"]["status"] == "insufficient_evidence"
    assert report["selection"]["eligible_variant_count"] == 0
    assert report["selection"]["selected_candidate_strategy"] is None
    assert report["selection"]["selected_pool_variant"] is None
    assert report["selection"]["selected_fusion_method"] is None
    assert report["selection"]["production_default_eligible"] is False


def test_table_and_report_are_deterministic_and_reject_authority_tampering() -> None:
    plan, scoring, review = _inputs()
    first = build_dynamic_pool_pilot_selection_ablation(plan, scoring, review)
    second = build_dynamic_pool_pilot_selection_ablation(plan, scoring, review)

    assert first.equals(second)
    tampered = first.with_columns(pl.lit(True).alias("production_default_eligible"))
    with pytest.raises(ValueError, match="authorized a default"):
        validate_dynamic_pool_pilot_selection_ablation(tampered, plan, scoring, review)

    report = build_dynamic_pool_pilot_selection_ablation_report(
        plan, scoring, review, first
    )
    tampered_report = deepcopy(report)
    tampered_report["selection"]["production_default_eligible"] = True
    with pytest.raises(ValueError, match="differs from its table"):
        validate_dynamic_pool_pilot_selection_ablation_report(
            tampered_report, plan, scoring, review, first
        )


def test_public_table_matches_committed_evidence(tmp_path: Path) -> None:
    plan, scoring, review = _inputs()
    frame = build_dynamic_pool_pilot_selection_ablation(plan, scoring, review)
    expected = publish_dynamic_pool_pilot_selection_ablation_table(
        frame, plan, scoring, review
    )

    output = write_dynamic_pool_pilot_selection_ablation_table(
        frame, plan, scoring, review, tmp_path
    )
    assert output.name == DYNAMIC_POOL_PILOT_SELECTION_ABLATION_TABLE_FILE
    assert pl.read_csv(output, schema_overrides=expected.schema).equals(expected)
    assert pl.read_csv(TABLE_PATH, schema_overrides=expected.schema).equals(expected)


def test_report_round_trip(tmp_path: Path) -> None:
    plan, scoring, review = _inputs()
    frame = build_dynamic_pool_pilot_selection_ablation(plan, scoring, review)
    report = build_dynamic_pool_pilot_selection_ablation_report(
        plan, scoring, review, frame
    )

    output = write_dynamic_pool_pilot_selection_ablation_report(
        report, plan, scoring, review, frame, tmp_path
    )
    assert output.name == DYNAMIC_POOL_PILOT_SELECTION_ABLATION_REPORT_FILE
    assert json.loads(output.read_text(encoding="utf-8")) == report
