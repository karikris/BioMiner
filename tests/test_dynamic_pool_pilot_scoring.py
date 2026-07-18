"""Tests for bounded cached-vector dynamic-pool pilot scoring."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import polars as pl
import pytest

from biominer.evaluation.dynamic_pool_pilot_plan import (
    PILOT_CANDIDATE_STRATEGIES,
    load_dynamic_pool_pilot_plan,
)
from biominer.evaluation.dynamic_pool_pilot_scoring import (
    DYNAMIC_POOL_PILOT_SCORING_REPORT_FILE,
    build_dynamic_pool_pilot_scoring_report,
    execute_dynamic_pool_pilot_scoring,
    validate_dynamic_pool_pilot_scoring_execution,
    validate_dynamic_pool_pilot_scoring_report,
    validate_dynamic_pool_pilot_scoring_results,
    write_dynamic_pool_pilot_scoring_report,
)


ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "config/pilot/geography_conditioned_dynamic_pool_pilot_v1.json"
REPORT_PATH = ROOT / "reports/geo_dynamic_pooling/pilot/dynamic_pool_scoring.json"


def _plan() -> dict[str, object]:
    return load_dynamic_pool_pilot_plan(PLAN_PATH, repository_root=ROOT)


def test_execution_covers_all_variants_without_encoder_or_images() -> None:
    plan = _plan()
    execution = execute_dynamic_pool_pilot_scoring(plan)

    validate_dynamic_pool_pilot_scoring_execution(execution, plan)
    assert len(execution.works) == 14
    assert execution.results.height == 168
    assert (
        execution.results.select(
            "case_id", "candidate_strategy", "pool_variant", "fusion_method"
        ).n_unique()
        == 168
    )
    assert execution.batch_result.metrics.work_items == 14
    assert execution.batch_result.metrics.execution_batches == 1
    assert execution.batch_result.metrics.encoder_invocations == 0
    assert execution.batch_result.metrics.image_materializations == 0
    assert set(execution.results["candidate_strategy"]) == set(
        PILOT_CANDIDATE_STRATEGIES
    )
    assert set(execution.results["pool_variant"]) == {
        "global_only_control",
        "dynamic_global_local",
    }
    assert execution.results["fixture_expected_target_at_1"].sum() == 168


def test_score_work_is_reused_across_candidate_schedules() -> None:
    plan = _plan()
    frame = execute_dynamic_pool_pilot_scoring(plan).results

    for (_case, _pool, _method), group in frame.group_by(
        "case_id", "pool_variant", "fusion_method"
    ):
        assert group.height == len(PILOT_CANDIDATE_STRATEGIES)
        assert group["source_work_fingerprint"].n_unique() == 1
        assert group["source_result_fingerprint"].n_unique() == 1
        assert group["query_embedding_fingerprint"].n_unique() == 1
        assert group["target_raw_fusion_score"].n_unique() == 1
        assert all(group["score_work_reused_across_candidate_strategies"])


def test_no_geo_uses_exact_global_fallback_and_located_cases_add_local() -> None:
    plan = _plan()
    frame = execute_dynamic_pool_pilot_scoring(plan).results
    no_geo = frame.filter(pl.col("no_geo"))
    located_dynamic = frame.filter(
        (~pl.col("no_geo")) & (pl.col("pool_variant") == "dynamic_global_local")
    )

    assert set(no_geo["local_evidence_status"]) == {"unavailable"}
    assert set(located_dynamic["local_evidence_status"]) == {"available"}
    global_rows = no_geo.filter(pl.col("pool_variant") == "global_only_control")
    dynamic_rows = no_geo.filter(pl.col("pool_variant") == "dynamic_global_local")
    pairs = global_rows.join(
        dynamic_rows,
        on=["case_id", "candidate_strategy", "fusion_method"],
        suffix="_dynamic",
        validate="1:1",
    )
    assert all(
        left == pytest.approx(right)
        for left, right in zip(
            pairs["target_raw_fusion_score"],
            pairs["target_raw_fusion_score_dynamic"],
            strict=True,
        )
    )
    assert (
        pairs["top_candidate_accepted_taxon_key"].to_list()
        == pairs["top_candidate_accepted_taxon_key_dynamic"].to_list()
    )


def test_observed_matrix_and_query_reuse_are_denominator_explicit() -> None:
    plan = _plan()
    execution = execute_dynamic_pool_pilot_scoring(plan)
    family = execution.family_cache_metrics
    candidate = execution.dynamic_cache_metrics.candidate
    pool = execution.dynamic_cache_metrics.pool
    batch = execution.batch_result.metrics

    assert (family.requests, family.hits, family.misses) == (14, 13, 1)
    assert (candidate.requests, candidate.hits, candidate.misses) == (14, 7, 7)
    assert (pool.requests, pool.hits, pool.misses) == (100, 65, 35)
    assert batch.pool_matrix_references == 100
    assert batch.unique_pool_matrices == 35
    assert batch.unique_pool_matrix_rows == 70
    assert batch.unique_pool_matrix_bytes == 2240
    assert batch.within_batch_matrix_reuses == 65
    assert execution.results["query_embedding_fingerprint"].n_unique() == 7
    assert (
        sum(
            result.cached_query_vectors_consumed
            for result in execution.batch_result.canonical_results
        )
        == 14
    )


def test_scores_remain_raw_and_all_authority_stays_unavailable() -> None:
    plan = _plan()
    frame = execute_dynamic_pool_pilot_scoring(plan).results

    assert not any(frame["raw_score_is_probability"])
    assert set(frame["probability_availability"]) == {
        "unavailable_fixture_uncalibrated"
    }
    assert set(frame["human_review_status"]) == {"unavailable_not_run"}
    assert not any(frame["production_default_eligible"])
    assert set(frame["expected_label_basis"]) == {
        "fixture_expected_taxon_not_human_review"
    }


def test_report_records_raw_comparison_and_fail_closed_selection() -> None:
    plan = _plan()
    execution = execute_dynamic_pool_pilot_scoring(plan)
    report = build_dynamic_pool_pilot_scoring_report(plan, execution)

    assert report["coverage"] == {
        "case_count": 7,
        "candidate_strategy_count": 3,
        "pool_variant_count": 2,
        "fusion_method_count": 4,
        "variant_count_per_case": 24,
        "result_row_count": 168,
        "score_work_item_count": 14,
    }
    assert report["model_execution"]["bioclip_image_encoder_run"] is False
    assert report["model_execution"]["synthetic_fixture_vectors"] is True
    assert report["embedding_reuse"]["unique_query_embedding_count"] == 7
    assert report["embedding_reuse"]["query_embedding_consumption_count"] == 14
    assert report["embedding_reuse"]["query_embedding_reuse_event_count"] == 7
    assert report["embedding_reuse"]["avoided_encoder_seconds"] is None
    comparison = report["global_local_comparison"]
    assert comparison["located_pair_count"] == 72
    assert comparison["located_target_raw_score_changed_count"] == 36
    assert comparison["located_top_candidate_changed_count"] == 0
    assert comparison["no_geo_pair_count"] == 12
    assert comparison["no_geo_global_fallback_parity_count"] == 12
    assert report["selection"]["status"] == "insufficient_evidence"
    assert report["selection"]["selected_candidate_strategy"] is None
    assert report["selection"]["selected_pool_variant"] is None
    assert report["selection"]["selected_fusion_method"] is None
    assert report["selection"]["production_default_eligible"] is False


def test_execution_and_report_are_deterministic_and_reject_drift() -> None:
    plan = _plan()
    first = execute_dynamic_pool_pilot_scoring(plan)
    second = execute_dynamic_pool_pilot_scoring(plan)

    assert first.works == second.works
    assert first.batch_result == second.batch_result
    assert first.family_cache_metrics == second.family_cache_metrics
    assert first.dynamic_cache_metrics == second.dynamic_cache_metrics
    assert first.results.equals(second.results)
    assert first.execution_fingerprint == second.execution_fingerprint
    tampered_results = first.results.with_columns(
        pl.lit(True).alias("raw_score_is_probability")
    )
    with pytest.raises(ValueError, match="promoted a raw score"):
        validate_dynamic_pool_pilot_scoring_results(tampered_results, plan)
    with pytest.raises(ValueError, match="execution fingerprint differs"):
        validate_dynamic_pool_pilot_scoring_execution(
            replace(first, execution_fingerprint="sha256:" + "f" * 64), plan
        )


def test_report_round_trip_and_committed_evidence_match(tmp_path: Path) -> None:
    plan = _plan()
    execution = execute_dynamic_pool_pilot_scoring(plan)
    report = build_dynamic_pool_pilot_scoring_report(plan, execution)

    validate_dynamic_pool_pilot_scoring_report(report, plan, execution)
    output = write_dynamic_pool_pilot_scoring_report(report, plan, execution, tmp_path)
    assert output.name == DYNAMIC_POOL_PILOT_SCORING_REPORT_FILE
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert json.loads(REPORT_PATH.read_text(encoding="utf-8")) == report


def test_report_rejects_authority_and_metric_tampering() -> None:
    plan = _plan()
    execution = execute_dynamic_pool_pilot_scoring(plan)
    report = build_dynamic_pool_pilot_scoring_report(plan, execution)
    changes = (
        ("model_execution", "bioclip_image_encoder_run", True),
        ("selection", "production_default_eligible", True),
        ("selection", "selected_fusion_method", "unweighted_component_mean"),
        ("scientific_claims", "raw_scores_are_probabilities", True),
        ("scientific_claims", "human_review_completed", True),
    )
    for section, field, value in changes:
        tampered = deepcopy(report)
        tampered[section][field] = value
        with pytest.raises(ValueError):
            validate_dynamic_pool_pilot_scoring_report(tampered, plan, execution)
