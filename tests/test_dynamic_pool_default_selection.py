"""Tests for the bounded pilot's fail-closed production-default decision."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

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
    build_dynamic_pool_pilot_selection_ablation,
)
from biominer.run.dynamic_pool_config import DynamicPoolingSettings
from biominer.run.dynamic_pool_default_selection import (
    DYNAMIC_POOL_PRODUCTION_DEFAULT_DECISION_FILE,
    build_dynamic_pool_production_default_decision,
    validate_dynamic_pool_production_default_decision,
    write_dynamic_pool_production_default_decision,
)


ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "config/pilot/geography_conditioned_dynamic_pool_pilot_v1.json"
DECISION_PATH = (
    ROOT / "reports/geo_dynamic_pooling/pilot/production_default_decision.json"
)


def _inputs():
    plan = load_dynamic_pool_pilot_plan(PLAN_PATH, repository_root=ROOT)
    scoring = execute_dynamic_pool_pilot_scoring(plan)
    review = build_dynamic_pool_pilot_review_plan(plan, scoring)
    ablation = build_dynamic_pool_pilot_selection_ablation(plan, scoring, review)
    return plan, scoring, review, ablation, DynamicPoolingSettings()


def test_fixture_policy_returns_insufficient_evidence_not_measured_rejection() -> None:
    plan, scoring, review, ablation, settings = _inputs()
    decision = build_dynamic_pool_production_default_decision(
        plan, scoring, review, ablation, settings
    )
    result = decision["decision"]

    assert result["outcome"] == "insufficient_evidence"
    assert result["outcome_is_rejection_of_measured_performance"] is False
    assert result["eligible_variant_count"] == 0
    assert result["selected_candidate_strategy"] is None
    assert result["selected_pool_variant"] is None
    assert result["selected_fusion_method"] is None
    assert result["selection_evidence_fingerprint"] is None
    assert result["production_default_authorized"] is False


def test_all_nine_criteria_are_evaluated_with_six_blockers() -> None:
    plan, scoring, review, ablation, settings = _inputs()
    decision = build_dynamic_pool_production_default_decision(
        plan, scoring, review, ablation, settings
    )
    criteria = decision["criterion_evaluations"]

    assert len(criteria) == 9
    assert {row["criterion"] for row in criteria} == {
        "target_candidate_recall",
        "reviewed_precision_and_confidence_bounds",
        "family_and_geographic_subgroup_behavior",
        "review_workload",
        "computation",
        "embedding_and_matrix_reuse",
        "mps_memory",
        "target_pruning_regressions",
        "unsupported_statistical_claims",
    }
    assert decision["decision"]["blocking_criteria"] == [
        "target_candidate_recall",
        "reviewed_precision_and_confidence_bounds",
        "family_and_geographic_subgroup_behavior",
        "review_workload",
        "computation",
        "mps_memory",
    ]


def test_unselected_runtime_defaults_remain_byte_semantically_unchanged() -> None:
    plan, scoring, review, ablation, settings = _inputs()
    decision = build_dynamic_pool_production_default_decision(
        plan, scoring, review, ablation, settings
    )

    assert settings.selection_status == "unselected"
    assert settings.candidate_strategy is None
    assert settings.fusion_method is None
    assert (
        decision["current_runtime_settings"] == decision["resulting_runtime_settings"]
    )
    assert decision["current_runtime_settings"]["pool_variant"] is None
    assert decision["default_change"] == {
        "candidate_strategy_changed": False,
        "pool_variant_changed": False,
        "fusion_method_changed": False,
        "reference_pool_policy_changed": False,
        "settings_fingerprint_changed": False,
    }
    assert decision["current_runtime_settings"]["settings_fingerprint"] == (
        "sha256:0fd197b2650a79d99970cada3dcbabe9980c5a265d9d71f929bbcf6f51e13e7d"
    )


def test_decision_requires_current_unselected_baseline() -> None:
    plan, scoring, review, ablation, _settings = _inputs()
    selected = DynamicPoolingSettings(
        candidate_strategy="parallel_family_geography_union",
        candidate_strategy_selection_fingerprint="sha256:" + "a" * 64,
        fusion_method="unweighted_component_mean",
        fusion_selection_fingerprint="sha256:" + "b" * 64,
    )

    with pytest.raises(ValueError, match="requires an unselected baseline"):
        build_dynamic_pool_production_default_decision(
            plan, scoring, review, ablation, selected
        )


def test_decision_is_deterministic_and_rejects_selection_tampering() -> None:
    plan, scoring, review, ablation, settings = _inputs()
    first = build_dynamic_pool_production_default_decision(
        plan, scoring, review, ablation, settings
    )
    second = build_dynamic_pool_production_default_decision(
        plan, scoring, review, ablation, settings
    )

    assert first == second
    tampered = deepcopy(first)
    tampered["decision"]["production_default_authorized"] = True
    tampered["decision"]["selected_candidate_strategy"] = (
        "parallel_family_geography_union"
    )
    with pytest.raises(ValueError, match="differs from policy evidence"):
        validate_dynamic_pool_production_default_decision(
            tampered, plan, scoring, review, ablation, settings
        )


def test_decision_round_trip_and_committed_evidence_match(tmp_path: Path) -> None:
    plan, scoring, review, ablation, settings = _inputs()
    decision = build_dynamic_pool_production_default_decision(
        plan, scoring, review, ablation, settings
    )

    output = write_dynamic_pool_production_default_decision(
        decision, plan, scoring, review, ablation, settings, tmp_path
    )
    assert output.name == DYNAMIC_POOL_PRODUCTION_DEFAULT_DECISION_FILE
    assert json.loads(output.read_text(encoding="utf-8")) == decision
    assert json.loads(DECISION_PATH.read_text(encoding="utf-8")) == decision
