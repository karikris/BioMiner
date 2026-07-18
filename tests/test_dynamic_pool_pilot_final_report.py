"""Tests for the integrated geography-conditioned pooling pilot report."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from biominer.evaluation.dynamic_pool_pilot_ablation import (
    build_dynamic_pool_pilot_candidate_ablation,
)
from biominer.evaluation.dynamic_pool_pilot_final_report import (
    DYNAMIC_POOL_PILOT_FINAL_REPORT_FILE,
    DYNAMIC_POOL_PILOT_FINAL_REPORT_SUMMARY_FILE,
    build_dynamic_pool_pilot_final_report,
    dynamic_pool_pilot_final_report_markdown,
    validate_dynamic_pool_pilot_final_report,
    write_dynamic_pool_pilot_final_report,
)
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
    build_dynamic_pool_pilot_selection_ablation,
)
from biominer.run.dynamic_pool_config import DynamicPoolingSettings
from biominer.run.dynamic_pool_default_selection import (
    build_dynamic_pool_production_default_decision,
)


ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "config/pilot/geography_conditioned_dynamic_pool_pilot_v1.json"
REPORT_PATH = (
    ROOT / "reports/geo_dynamic_pooling/pilot/geography_conditioned_pooling_report.json"
)
SUMMARY_PATH = (
    ROOT / "reports/geo_dynamic_pooling/pilot/geography_conditioned_pooling_report.md"
)


def _inputs():
    plan = load_dynamic_pool_pilot_plan(PLAN_PATH, repository_root=ROOT)
    candidate = build_dynamic_pool_pilot_candidate_ablation(plan)
    scoring = execute_dynamic_pool_pilot_scoring(plan)
    review = build_dynamic_pool_pilot_review_plan(plan, scoring)
    selection = build_dynamic_pool_pilot_selection_ablation(plan, scoring, review)
    settings = DynamicPoolingSettings()
    decision = build_dynamic_pool_production_default_decision(
        plan, scoring, review, selection, settings
    )
    report = build_dynamic_pool_pilot_final_report(
        plan, candidate, scoring, review, selection, decision, settings
    )
    return plan, candidate, scoring, review, selection, decision, settings, report


def test_report_preserves_fixture_and_historical_evidence_boundary() -> None:
    *_inputs_without_report, report = _inputs()
    evidence = report["evidence_inventory"]

    assert evidence["current_execution_basis"] == "deterministic_fixture_vectors"
    assert evidence["current_fixture_case_count"] == 7
    assert evidence["current_source_bound_real_flickr_item_count"] == 0
    assert evidence["current_human_reviewed_label_count"] == 0
    assert evidence["historical_real_execution_manifest_count"] == 4
    assert evidence["historical_outputs_count_as_current_execution"] is False
    assert evidence["live_network_calls"] == 0
    assert evidence["live_bioclip_image_encoder_runs"] == 0


def test_report_has_complete_scope_ablation_and_reuse_denominators() -> None:
    *_inputs_without_report, report = _inputs()

    assert report["scope"]["taxon_count"] == 5
    assert report["scope"]["case_count"] == 7
    assert report["scope"]["located_case_count"] == 6
    assert report["scope"]["no_geo_case_count"] == 1
    assert report["candidate_ablation"]["strategy_count"] == 3
    assert report["candidate_ablation"]["target_pruning_regression_count"] == 0
    assert report["scoring_ablation"]["variant_count"] == 24
    assert report["scoring_ablation"]["case_variant_result_count"] == 168
    assert report["scoring_ablation"]["located_target_raw_score_changed_count"] == 36
    assert report["scoring_ablation"]["located_top_candidate_changed_count"] == 0
    assert report["scoring_ablation"]["no_geo_exact_global_fallback_parity_count"] == 12
    assert report["computation_and_reuse"]["query_embedding_reuse_event_count"] == 7
    assert report["computation_and_reuse"]["within_batch_matrix_reuses"] == 65


def test_report_preserves_zero_review_shortfall_and_insufficient_decision() -> None:
    *_inputs_without_report, report = _inputs()
    reviews = report["review_and_statistical_support"]
    result = report["executive_result"]

    assert reviews["representative_fixture_selected_count"] == 7
    assert reviews["targeted_fixture_selected_count"] == 7
    assert reviews["completed_real_review_count"] == 0
    assert reviews["effective_real_review_count"] == 0
    assert reviews["effective_real_review_shortfall"] == 86
    assert reviews["reviewed_precision"] is None
    assert reviews["reviewed_precision_lower_bound"] is None
    assert reviews["statistical_support_status"] == "insufficient_evidence"
    assert result["decision"] == "insufficient_evidence"
    assert result["eligible_variant_count"] == 0
    assert result["production_default_authorized"] is False
    assert result["occurrence_release_authorized"] is False


def test_report_records_all_selection_gates_and_unchanged_settings() -> None:
    *_inputs_without_report, report = _inputs()
    selection = report["production_selection"]

    assert len(selection["criterion_evaluations"]) == 9
    assert len(selection["blocking_criteria"]) == 6
    assert (
        selection["current_settings_fingerprint"]
        == selection["resulting_settings_fingerprint"]
    )
    assert selection["settings_fingerprint_changed"] is False
    assert selection["review_projection_is_selected_default"] is False


def test_report_is_deterministic_and_rejects_scientific_claim_tampering() -> None:
    inputs = _inputs()
    *arguments, first = inputs
    second = build_dynamic_pool_pilot_final_report(*arguments)

    assert first == second
    tampered = deepcopy(first)
    tampered["scientific_invariants"]["raw_scores_are_probabilities"] = True
    with pytest.raises(ValueError, match="differs from source evidence"):
        validate_dynamic_pool_pilot_final_report(tampered, *arguments)


def test_json_and_markdown_round_trip_and_match_committed_evidence(
    tmp_path: Path,
) -> None:
    inputs = _inputs()
    *arguments, report = inputs
    expected_markdown = dynamic_pool_pilot_final_report_markdown(report, *arguments)

    json_path, markdown_path = write_dynamic_pool_pilot_final_report(
        report, *arguments, tmp_path
    )
    assert json_path.name == DYNAMIC_POOL_PILOT_FINAL_REPORT_FILE
    assert markdown_path.name == DYNAMIC_POOL_PILOT_FINAL_REPORT_SUMMARY_FILE
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    assert markdown_path.read_text(encoding="utf-8") == expected_markdown
    assert json.loads(REPORT_PATH.read_text(encoding="utf-8")) == report
    assert SUMMARY_PATH.read_text(encoding="utf-8") == expected_markdown
