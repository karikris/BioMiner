"""Fail-closed production-default decision for the bounded pilot."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.dynamic_pool_pilot_plan import (
    validate_dynamic_pool_pilot_plan,
)
from biominer.evaluation.dynamic_pool_pilot_review import (
    DynamicPoolPilotReviewPlan,
)
from biominer.evaluation.dynamic_pool_pilot_scoring import (
    DynamicPoolPilotScoringExecution,
)
from biominer.evaluation.dynamic_pool_pilot_selection_ablation import (
    validate_dynamic_pool_pilot_selection_ablation,
)
from biominer.run.dynamic_pool_config import DynamicPoolingSettings


DYNAMIC_POOL_PRODUCTION_DEFAULT_DECISION_VERSION = (
    "dynamic-pool-production-default-decision-v1.0.0"
)
DYNAMIC_POOL_PRODUCTION_DEFAULT_DECISION_FILE = "production_default_decision.json"


def build_dynamic_pool_production_default_decision(
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
    ablation: pl.DataFrame,
    current_settings: DynamicPoolingSettings,
) -> dict[str, object]:
    """Apply the frozen policy without moving an unselected runtime default."""

    _validate_inputs(plan, scoring, review, ablation, current_settings)
    decision = _production_default_decision_payload(
        plan, scoring, review, ablation, current_settings
    )
    decision["decision_fingerprint"] = canonical_semantic_fingerprint(decision)
    validate_dynamic_pool_production_default_decision(
        decision, plan, scoring, review, ablation, current_settings
    )
    return decision


def validate_dynamic_pool_production_default_decision(
    decision: Mapping[str, object],
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
    ablation: pl.DataFrame,
    current_settings: DynamicPoolingSettings,
) -> None:
    """Require the decision to equal a fresh fail-closed policy evaluation."""

    if (
        decision.get("schema_version")
        != DYNAMIC_POOL_PRODUCTION_DEFAULT_DECISION_VERSION
    ):
        raise ValueError("unsupported dynamic-pool default decision version")
    _validate_inputs(plan, scoring, review, ablation, current_settings)
    expected = _production_default_decision_payload(
        plan, scoring, review, ablation, current_settings
    )
    expected["decision_fingerprint"] = canonical_semantic_fingerprint(expected)
    if dict(decision) != expected:
        raise ValueError("dynamic-pool default decision differs from policy evidence")


def write_dynamic_pool_production_default_decision(
    decision: Mapping[str, object],
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
    ablation: pl.DataFrame,
    current_settings: DynamicPoolingSettings,
    output: str | Path,
) -> Path:
    """Atomically write one validated production-default decision."""

    validate_dynamic_pool_production_default_decision(
        decision, plan, scoring, review, ablation, current_settings
    )
    destination = Path(output)
    if destination.suffix.casefold() != ".json":
        destination /= DYNAMIC_POOL_PRODUCTION_DEFAULT_DECISION_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(decision), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _validate_inputs(
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
    ablation: pl.DataFrame,
    current_settings: DynamicPoolingSettings,
) -> None:
    validate_dynamic_pool_pilot_plan(plan)
    validate_dynamic_pool_pilot_selection_ablation(ablation, plan, scoring, review)
    if not isinstance(current_settings, DynamicPoolingSettings):
        raise TypeError("current_settings must be DynamicPoolingSettings")
    if current_settings.selection_status != "unselected":
        raise ValueError("pilot default decision requires an unselected baseline")


def _production_default_decision_payload(
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
    ablation: pl.DataFrame,
    current_settings: DynamicPoolingSettings,
) -> dict[str, object]:
    policy = plan["acceptance_policy"]
    table_fingerprint = canonical_semantic_fingerprint(
        ablation["row_fingerprint"].to_list()
    )
    settings = _runtime_settings_summary(current_settings)
    criteria = _criterion_evaluations(policy, scoring, review, ablation)
    blockers = [
        criterion["criterion"]
        for criterion in criteria
        if criterion["selection_gate_status"] != "passed_eligible_evidence"
    ]
    return {
        "schema_version": DYNAMIC_POOL_PRODUCTION_DEFAULT_DECISION_VERSION,
        "pilot_id": plan["pilot_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "acceptance_policy_version": policy["policy_version"],
        "source_ablation_table_fingerprint": table_fingerprint,
        "source_scoring_execution_fingerprint": scoring.execution_fingerprint,
        "source_review_plan_fingerprint": review.review_plan_fingerprint,
        "evidence_basis": "fixture_execution_with_zero_completed_real_reviews",
        "criterion_evaluations": criteria,
        "decision": {
            "outcome": policy["fixture_forced_decision"],
            "outcome_is_rejection_of_measured_performance": False,
            "eligible_variant_count": int(
                ablation["production_default_eligible"].sum()
            ),
            "selected_candidate_strategy": None,
            "selected_pool_variant": None,
            "selected_fusion_method": None,
            "selection_evidence_fingerprint": None,
            "production_default_authorized": False,
            "runtime_settings_changed": False,
            "blocking_criteria": blockers,
            "reason": (
                "fixture evidence and zero completed real reviews cannot satisfy "
                "the production acceptance policy"
            ),
        },
        "current_runtime_settings": settings,
        "resulting_runtime_settings": dict(settings),
        "default_change": {
            "candidate_strategy_changed": False,
            "pool_variant_changed": False,
            "fusion_method_changed": False,
            "reference_pool_policy_changed": False,
            "settings_fingerprint_changed": False,
        },
        "next_evidence_required": {
            "source_bound_human_review": True,
            "minimum_effective_reviewed_records": policy[
                "minimum_effective_reviewed_records"
            ],
            "remaining_effective_review_shortfall": policy[
                "minimum_effective_reviewed_records"
            ],
            "minimum_subgroup_independent_records": policy[
                "minimum_subgroup_independent_records"
            ],
            "minimum_reviewed_precision_lower_bound": policy[
                "minimum_reviewed_precision_lower_bound"
            ],
            "comparable_instrumented_computation": True,
            "mps_peak_memory_measurement": True,
        },
        "scientific_claims": {
            "fixture_metrics_select_a_default": False,
            "insufficient_evidence_is_measured_rejection": False,
            "raw_scores_are_probabilities": False,
            "missing_geography_is_biological_absence": False,
            "review_work_is_completed_review": False,
            "production_default_selected": False,
            "occurrence_release_authorized": False,
        },
    }


def _criterion_evaluations(
    policy: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
    ablation: pl.DataFrame,
) -> list[dict[str, object]]:
    metrics = scoring.batch_result.metrics
    return [
        {
            "criterion": "target_candidate_recall",
            "required": policy["minimum_target_candidate_recall"],
            "observed": float(ablation["candidate_target_recall_at_5"].min()),
            "evidence_status": "fixture_structural_only",
            "selection_gate_status": "insufficient_ineligible_evidence",
        },
        {
            "criterion": "reviewed_precision_and_confidence_bounds",
            "required_lower_bound": policy["minimum_reviewed_precision_lower_bound"],
            "observed_lower_bound": None,
            "evidence_status": "unavailable_no_completed_real_reviews",
            "selection_gate_status": "insufficient_unavailable",
        },
        {
            "criterion": "family_and_geographic_subgroup_behavior",
            "required_independent_records_per_subgroup": policy[
                "minimum_subgroup_independent_records"
            ],
            "observed_minimum_independent_records": 0,
            "evidence_status": "unavailable_no_completed_real_reviews",
            "selection_gate_status": "insufficient_unavailable",
        },
        {
            "criterion": "review_workload",
            "required_effective_real_reviews": policy[
                "minimum_effective_reviewed_records"
            ],
            "observed_effective_real_reviews": 0,
            "planned_representative_fixture_items": (
                review.representative.selected_count
            ),
            "planned_targeted_fixture_items": review.targeted_queue.height,
            "evidence_status": "planned_fixture_work_not_completed_reviews",
            "selection_gate_status": "insufficient_unmet",
        },
        {
            "criterion": "computation",
            "observed_score_work_items": len(scoring.works),
            "encoder_invocations": metrics.encoder_invocations,
            "instrumented_comparable_runtime_seconds": None,
            "evidence_status": "fixture_counts_available_timing_not_instrumented",
            "selection_gate_status": "insufficient_unavailable",
        },
        {
            "criterion": "embedding_and_matrix_reuse",
            "embedding_reuse_required": policy["embedding_reuse_required"],
            "matrix_reuse_required": policy["matrix_reuse_required"],
            "observed_query_embedding_reuse_events": len(scoring.works)
            - scoring.results["query_embedding_fingerprint"].n_unique(),
            "observed_within_batch_matrix_reuses": (metrics.within_batch_matrix_reuses),
            "evidence_status": "observed_complete_fixture_execution",
            "selection_gate_status": "passed_eligible_evidence",
        },
        {
            "criterion": "mps_memory",
            "required_maximum_bytes": policy["mps_memory_limit_bytes"],
            "observed_peak_bytes": None,
            "evidence_status": "unavailable_cached_vector_fixture_not_mps",
            "selection_gate_status": "insufficient_unavailable",
        },
        {
            "criterion": "target_pruning_regressions",
            "required_regression_count": 0,
            "observed_regression_count": int(
                ablation["target_pruning_regression_count"].max()
            ),
            "evidence_status": "observed_complete_fixture_union",
            "selection_gate_status": "passed_eligible_evidence",
        },
        {
            "criterion": "unsupported_statistical_claims",
            "allowed": policy["unsupported_statistical_claims_allowed"],
            "observed": any(ablation["unsupported_statistical_claims_present"]),
            "evidence_status": "validated_report_contract",
            "selection_gate_status": "passed_eligible_evidence",
        },
    ]


def _runtime_settings_summary(settings: DynamicPoolingSettings) -> dict[str, object]:
    return {
        "settings_schema_version": settings.schema_version,
        "settings_fingerprint": settings.fingerprint,
        "selection_status": settings.selection_status,
        "candidate_strategy": settings.candidate_strategy,
        "candidate_strategy_selection_fingerprint": (
            settings.candidate_strategy_selection_fingerprint
        ),
        "pool_variant": None,
        "fusion_method": settings.fusion_method,
        "fusion_selection_fingerprint": settings.fusion_selection_fingerprint,
        "reference_pool_policy_fingerprint": (
            settings.reference_pool_policy.fingerprint
        ),
        "release_requires_human_review": settings.release_requires_human_review,
        "missing_geography_is_biological_absence": (
            settings.missing_geography_is_biological_absence
        ),
        "raw_scores_are_probabilities": settings.raw_scores_are_probabilities,
    }


__all__ = [
    "DYNAMIC_POOL_PRODUCTION_DEFAULT_DECISION_FILE",
    "DYNAMIC_POOL_PRODUCTION_DEFAULT_DECISION_VERSION",
    "build_dynamic_pool_production_default_decision",
    "validate_dynamic_pool_production_default_decision",
    "write_dynamic_pool_production_default_decision",
]
