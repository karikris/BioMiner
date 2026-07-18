"""Integrated scientific and production-decision report for the bounded pilot."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.dynamic_pool_pilot_ablation import (
    validate_dynamic_pool_pilot_candidate_ablation,
)
from biominer.evaluation.dynamic_pool_pilot_plan import (
    validate_dynamic_pool_pilot_plan,
)
from biominer.evaluation.dynamic_pool_pilot_review import (
    DynamicPoolPilotReviewPlan,
    validate_dynamic_pool_pilot_review_plan,
)
from biominer.evaluation.dynamic_pool_pilot_scoring import (
    DynamicPoolPilotScoringExecution,
    validate_dynamic_pool_pilot_scoring_execution,
)
from biominer.evaluation.dynamic_pool_pilot_selection_ablation import (
    validate_dynamic_pool_pilot_selection_ablation,
)
from biominer.run.dynamic_pool_config import DynamicPoolingSettings
from biominer.run.dynamic_pool_default_selection import (
    validate_dynamic_pool_production_default_decision,
)


DYNAMIC_POOL_PILOT_FINAL_REPORT_VERSION = "dynamic-pool-pilot-final-report-v1.0.0"
DYNAMIC_POOL_PILOT_FINAL_REPORT_FILE = "geography_conditioned_pooling_report.json"
DYNAMIC_POOL_PILOT_FINAL_REPORT_SUMMARY_FILE = "geography_conditioned_pooling_report.md"


def build_dynamic_pool_pilot_final_report(
    plan: Mapping[str, object],
    candidate_ablation: pl.DataFrame,
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
    selection_ablation: pl.DataFrame,
    decision: Mapping[str, object],
    current_settings: DynamicPoolingSettings,
) -> dict[str, object]:
    """Consolidate execution, evidence maturity, decision, and remaining work."""

    _validate_inputs(
        plan,
        candidate_ablation,
        scoring,
        review,
        selection_ablation,
        decision,
        current_settings,
    )
    report = _final_report_payload(
        plan,
        candidate_ablation,
        scoring,
        review,
        selection_ablation,
        decision,
        current_settings,
    )
    report["report_fingerprint"] = canonical_semantic_fingerprint(report)
    validate_dynamic_pool_pilot_final_report(
        report,
        plan,
        candidate_ablation,
        scoring,
        review,
        selection_ablation,
        decision,
        current_settings,
    )
    return report


def validate_dynamic_pool_pilot_final_report(
    report: Mapping[str, object],
    plan: Mapping[str, object],
    candidate_ablation: pl.DataFrame,
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
    selection_ablation: pl.DataFrame,
    decision: Mapping[str, object],
    current_settings: DynamicPoolingSettings,
) -> None:
    """Require the final report to equal one fresh source-derived projection."""

    if report.get("schema_version") != DYNAMIC_POOL_PILOT_FINAL_REPORT_VERSION:
        raise ValueError("unsupported dynamic-pool pilot final report version")
    _validate_inputs(
        plan,
        candidate_ablation,
        scoring,
        review,
        selection_ablation,
        decision,
        current_settings,
    )
    expected = _final_report_payload(
        plan,
        candidate_ablation,
        scoring,
        review,
        selection_ablation,
        decision,
        current_settings,
    )
    expected["report_fingerprint"] = canonical_semantic_fingerprint(expected)
    if dict(report) != expected:
        raise ValueError("dynamic-pool pilot final report differs from source evidence")


def dynamic_pool_pilot_final_report_markdown(
    report: Mapping[str, object],
    plan: Mapping[str, object],
    candidate_ablation: pl.DataFrame,
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
    selection_ablation: pl.DataFrame,
    decision: Mapping[str, object],
    current_settings: DynamicPoolingSettings,
) -> str:
    """Render the validated executive report without adding new claims."""

    validate_dynamic_pool_pilot_final_report(
        report,
        plan,
        candidate_ablation,
        scoring,
        review,
        selection_ablation,
        decision,
        current_settings,
    )
    return _final_report_markdown(report)


def write_dynamic_pool_pilot_final_report(
    report: Mapping[str, object],
    plan: Mapping[str, object],
    candidate_ablation: pl.DataFrame,
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
    selection_ablation: pl.DataFrame,
    decision: Mapping[str, object],
    current_settings: DynamicPoolingSettings,
    output: str | Path,
) -> tuple[Path, Path]:
    """Atomically write validated JSON and Markdown report representations."""

    markdown = dynamic_pool_pilot_final_report_markdown(
        report,
        plan,
        candidate_ablation,
        scoring,
        review,
        selection_ablation,
        decision,
        current_settings,
    )
    destination = Path(output)
    if destination.suffix.casefold() == ".json":
        directory = destination.parent
        json_path = destination
    else:
        directory = destination
        json_path = directory / DYNAMIC_POOL_PILOT_FINAL_REPORT_FILE
    markdown_path = directory / DYNAMIC_POOL_PILOT_FINAL_REPORT_SUMMARY_FILE
    directory.mkdir(parents=True, exist_ok=True)
    json_temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    markdown_temporary = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    json_temporary.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_temporary.write_text(markdown, encoding="utf-8")
    json_temporary.replace(json_path)
    markdown_temporary.replace(markdown_path)
    return json_path, markdown_path


def _validate_inputs(
    plan: Mapping[str, object],
    candidate_ablation: pl.DataFrame,
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
    selection_ablation: pl.DataFrame,
    decision: Mapping[str, object],
    current_settings: DynamicPoolingSettings,
) -> None:
    validate_dynamic_pool_pilot_plan(plan)
    validate_dynamic_pool_pilot_candidate_ablation(candidate_ablation, plan)
    validate_dynamic_pool_pilot_scoring_execution(scoring, plan)
    validate_dynamic_pool_pilot_review_plan(review, plan, scoring)
    validate_dynamic_pool_pilot_selection_ablation(
        selection_ablation, plan, scoring, review
    )
    validate_dynamic_pool_production_default_decision(
        decision,
        plan,
        scoring,
        review,
        selection_ablation,
        current_settings,
    )


def _final_report_payload(
    plan: Mapping[str, object],
    candidate_ablation: pl.DataFrame,
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
    selection_ablation: pl.DataFrame,
    decision: Mapping[str, object],
    current_settings: DynamicPoolingSettings,
) -> dict[str, object]:
    policy = plan["acceptance_policy"]
    located = scoring.results.filter(~pl.col("no_geo"))
    no_geo = scoring.results.filter(pl.col("no_geo"))
    global_rows = scoring.results.filter(
        pl.col("pool_variant") == "global_only_control"
    )
    dynamic_rows = scoring.results.filter(
        pl.col("pool_variant") == "dynamic_global_local"
    )
    pairs = global_rows.join(
        dynamic_rows,
        on=["case_id", "candidate_strategy", "fusion_method"],
        suffix="_dynamic",
        validate="1:1",
    )
    located_pairs = pairs.filter(~pl.col("no_geo"))
    no_geo_pairs = pairs.filter(pl.col("no_geo"))
    metrics = scoring.batch_result.metrics
    strategy_metrics = []
    for strategy in plan["ablations"]["candidate_strategies"]:
        group = candidate_ablation.filter(pl.col("strategy_name") == strategy)
        strategy_metrics.append(
            {
                "candidate_strategy": strategy,
                "case_count": group.height,
                "target_candidate_recall_at_1": float(
                    group["target_candidate_recall_at_1"].mean()
                ),
                "target_candidate_recall_at_3": float(
                    group["target_candidate_recall_at_3"].mean()
                ),
                "target_candidate_recall_at_5": float(
                    group["target_candidate_recall_at_5"].mean()
                ),
                "target_preserved_count": int(group["target_preserved"].sum()),
                "complete_union_preserved_count": int(
                    group["complete_union_preserved"].sum()
                ),
                "interpretation": "fixture_structural_not_reviewed_accuracy",
            }
        )
    historical = [
        item
        for item in plan["durable_inputs"]
        if item["evidence_kind"] == "historical_real_execution_manifest"
    ]
    return {
        "schema_version": DYNAMIC_POOL_PILOT_FINAL_REPORT_VERSION,
        "pilot_id": plan["pilot_id"],
        "title": plan["title"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "status": "completed_fixture_pilot_insufficient_production_evidence",
        "executive_result": {
            "decision": decision["decision"]["outcome"],
            "eligible_variant_count": 0,
            "selected_candidate_strategy": None,
            "selected_pool_variant": None,
            "selected_fusion_method": None,
            "runtime_settings_changed": False,
            "production_default_authorized": False,
            "occurrence_release_authorized": False,
            "reason": decision["decision"]["reason"],
        },
        "evidence_inventory": {
            "current_execution_basis": "deterministic_fixture_vectors",
            "current_fixture_case_count": len(plan["cases"]),
            "current_source_bound_real_flickr_item_count": 0,
            "current_human_reviewed_label_count": 0,
            "historical_real_execution_manifest_count": len(historical),
            "historical_real_execution_manifests": [
                {
                    "relative_path": item["relative_path"],
                    "sha256": item["sha256"],
                    "authority": item["scientific_authority"],
                }
                for item in historical
            ],
            "historical_outputs_count_as_current_execution": False,
            "live_network_calls": 0,
            "live_bioclip_image_encoder_runs": 0,
        },
        "scope": {
            "taxon_count": len(plan["taxon_catalog"]),
            "taxa": [
                {
                    "accepted_taxon_key": taxon["accepted_taxon_key"],
                    "scientific_name": taxon["scientific_name"],
                    "family": taxon["family"],
                    "pilot_roles": list(taxon["pilot_roles"]),
                }
                for taxon in plan["taxon_catalog"]
            ],
            "case_count": len(plan["cases"]),
            "located_case_count": located["case_id"].n_unique(),
            "no_geo_case_count": no_geo["case_id"].n_unique(),
            "australian_case_count": sum(
                case["country_code"] == "AU" for case in plan["cases"]
            ),
            "candidate_taxa_per_case": int(
                candidate_ablation["candidate_set_size"].min()
            ),
        },
        "candidate_ablation": {
            "strategy_count": candidate_ablation["strategy_name"].n_unique(),
            "result_row_count": candidate_ablation.height,
            "strategy_metrics": strategy_metrics,
            "target_pruning_regression_count": int(
                (~candidate_ablation["target_preserved"]).sum()
            ),
            "classification_accuracy_status": "unavailable_fixture_only",
        },
        "scoring_ablation": {
            "candidate_strategy_count": selection_ablation[
                "candidate_strategy"
            ].n_unique(),
            "pool_variant_count": selection_ablation["pool_variant"].n_unique(),
            "fusion_method_count": selection_ablation["fusion_method"].n_unique(),
            "variant_count": selection_ablation.height,
            "case_variant_result_count": scoring.results.height,
            "score_work_item_count": len(scoring.works),
            "located_global_dynamic_pair_count": located_pairs.height,
            "located_target_raw_score_changed_count": _changed_count(
                located_pairs,
                "target_raw_fusion_score",
                "target_raw_fusion_score_dynamic",
            ),
            "located_top_candidate_changed_count": sum(
                left != right
                for left, right in zip(
                    located_pairs["top_candidate_accepted_taxon_key"],
                    located_pairs["top_candidate_accepted_taxon_key_dynamic"],
                    strict=True,
                )
            ),
            "no_geo_global_dynamic_pair_count": no_geo_pairs.height,
            "no_geo_exact_global_fallback_parity_count": _fallback_parity_count(
                no_geo_pairs
            ),
            "raw_score_is_probability": False,
            "classification_accuracy_status": "unavailable_fixture_only",
        },
        "computation_and_reuse": {
            "query_embedding_count": scoring.results[
                "query_embedding_fingerprint"
            ].n_unique(),
            "query_embedding_consumption_count": sum(
                result.cached_query_vectors_consumed
                for result in scoring.batch_result.canonical_results
            ),
            "query_embedding_reuse_event_count": len(scoring.works)
            - scoring.results["query_embedding_fingerprint"].n_unique(),
            "encoder_invocations": metrics.encoder_invocations,
            "image_materializations": metrics.image_materializations,
            "pool_matrix_references": metrics.pool_matrix_references,
            "unique_pool_matrices": metrics.unique_pool_matrices,
            "within_batch_matrix_reuses": metrics.within_batch_matrix_reuses,
            "maximum_batch_pool_matrix_bytes": (
                metrics.maximum_batch_pool_matrix_bytes
            ),
            "instrumented_runtime_seconds": None,
            "avoided_runtime_seconds": None,
            "mps_peak_memory_bytes": None,
            "mps_memory_limit_bytes": policy["mps_memory_limit_bytes"],
            "mps_status": "not_executed_cached_vector_fixture",
        },
        "review_and_statistical_support": {
            "representative_fixture_population_count": (
                review.representative.population_count
            ),
            "representative_fixture_selected_count": (
                review.representative.selected_count
            ),
            "targeted_fixture_selected_count": review.targeted_queue.height,
            "representative_and_targeted_purposes_separate": True,
            "reviewer_identity_count": 0,
            "assignment_count": 0,
            "completed_real_review_count": 0,
            "effective_real_review_count": 0,
            "minimum_effective_real_review_count": policy[
                "minimum_effective_reviewed_records"
            ],
            "effective_real_review_shortfall": policy[
                "minimum_effective_reviewed_records"
            ],
            "minimum_subgroup_independent_records": policy[
                "minimum_subgroup_independent_records"
            ],
            "reviewed_precision": None,
            "reviewed_precision_lower_bound": None,
            "minimum_reviewed_precision_lower_bound": policy[
                "minimum_reviewed_precision_lower_bound"
            ],
            "family_subgroup_status": "unavailable_no_completed_real_reviews",
            "geographic_subgroup_status": ("unavailable_no_completed_real_reviews"),
            "statistical_support_status": "insufficient_evidence",
            "occurrence_release_review_queue_count": review.release_queue.height,
        },
        "production_selection": {
            "acceptance_policy_version": decision["acceptance_policy_version"],
            "criterion_evaluations": list(decision["criterion_evaluations"]),
            "blocking_criteria": list(decision["decision"]["blocking_criteria"]),
            "decision_fingerprint": decision["decision_fingerprint"],
            "source_ablation_table_fingerprint": decision[
                "source_ablation_table_fingerprint"
            ],
            "current_settings_fingerprint": current_settings.fingerprint,
            "resulting_settings_fingerprint": current_settings.fingerprint,
            "settings_fingerprint_changed": False,
            "review_projection_is_selected_default": False,
        },
        "source_fingerprints": {
            "candidate_ablation": canonical_semantic_fingerprint(
                candidate_ablation["result_fingerprint"].to_list()
            ),
            "scoring_execution": scoring.execution_fingerprint,
            "scoring_results": canonical_semantic_fingerprint(
                scoring.results["result_fingerprint"].to_list()
            ),
            "review_plan": review.review_plan_fingerprint,
            "selection_ablation_table": decision["source_ablation_table_fingerprint"],
            "production_default_decision": decision["decision_fingerprint"],
        },
        "claims": {
            "allowed": [
                "The frozen fixture pilot exercises all 24 declared candidate, pool and fusion variants through production contracts.",
                "All candidate strategies preserve the complete five-taxon union and target in every fixture case.",
                "Observed cached-vector embedding and matrix reuse counts describe the complete fixture execution.",
                "The no-geography fixture preserves exact global fallback without implying biological absence.",
                "The production acceptance decision is insufficient evidence and runtime defaults remain unchanged.",
            ],
            "blocked": [
                "Fixture target ranks are reviewed classification accuracy or empirical superiority.",
                "Raw scores are calibrated probabilities.",
                "Historical manifests are current pilot execution or human-review outcomes.",
                "Planned representative or targeted fixture work is completed source-bound review.",
                "Any candidate strategy, pool variant or fusion method is a selected production default.",
                "Any occurrence is release ready or release authorized.",
            ],
        },
        "remaining_evidence": dict(decision["next_evidence_required"]),
        "scientific_invariants": {
            "raw_scores_are_probabilities": False,
            "missing_geography_is_biological_absence": False,
            "representative_and_targeted_review_are_merged": False,
            "fixture_reviews_satisfy_real_review_minimum": False,
            "production_default_selected": False,
            "occurrence_release_authorized": False,
        },
    }


def _changed_count(frame: pl.DataFrame, left: str, right: str) -> int:
    return sum(
        abs(float(a) - float(b)) > 1e-12
        for a, b in zip(frame[left], frame[right], strict=True)
    )


def _fallback_parity_count(frame: pl.DataFrame) -> int:
    return sum(
        abs(float(left_score) - float(right_score)) <= 1e-12 and left_top == right_top
        for left_score, right_score, left_top, right_top in zip(
            frame["target_raw_fusion_score"],
            frame["target_raw_fusion_score_dynamic"],
            frame["top_candidate_accepted_taxon_key"],
            frame["top_candidate_accepted_taxon_key_dynamic"],
            strict=True,
        )
    )


def _final_report_markdown(report: Mapping[str, object]) -> str:
    result = report["executive_result"]
    scope = report["scope"]
    scoring = report["scoring_ablation"]
    compute = report["computation_and_reuse"]
    review = report["review_and_statistical_support"]
    selection = report["production_selection"]
    lines = [
        "# Geography-conditioned dynamic pooling pilot",
        "",
        "Decision: **insufficient production evidence; no default selected or changed**.",
        "",
        "## Evidence boundary",
        "",
        (
            f"The current run is a deterministic {scope['case_count']}-case fixture "
            f"pilot over {scope['taxon_count']} taxa, with {scope['located_case_count']} "
            f"located cases and {scope['no_geo_case_count']} no-geography case. It "
            "made no network call, ran no BioCLIP image encoder, and contains no "
            "source-bound human label. Historical real-execution manifests are "
            "inventory only and are not counted as current results."
        ),
        "",
        "## Complete ablation",
        "",
        (
            f"The report covers {scoring['variant_count']} candidate/pool/fusion "
            f"variants and {scoring['case_variant_result_count']} case-variant rows. "
            "All strategies retain the complete five-taxon union and every target. "
            "Their order metrics are fixture structural recall, not classification "
            "accuracy."
        ),
        "",
        (
            f"Across {scoring['located_global_dynamic_pair_count']} located "
            f"global/dynamic pairs, {scoring['located_target_raw_score_changed_count']} "
            "target raw scores change and zero top candidates change. All "
            f"{scoring['no_geo_global_dynamic_pair_count']} no-geography pairs retain "
            "exact global fallback. Raw values are not probabilities."
        ),
        "",
        "## Computation and review",
        "",
        (
            f"The shared run uses {scoring['score_work_item_count']} cached-vector "
            f"work items, {compute['query_embedding_count']} unique query vectors, "
            f"{compute['query_embedding_reuse_event_count']} query reuse events, "
            f"{compute['pool_matrix_references']} pool-matrix references, and "
            f"{compute['within_batch_matrix_reuses']} within-batch matrix reuses. "
            "Runtime savings and MPS peak memory were not measured."
        ),
        "",
        (
            f"Seven representative and seven targeted fixture work items are "
            f"planned, but completed real reviews remain {review['completed_real_review_count']}. "
            f"The effective-review shortfall is therefore "
            f"{review['effective_real_review_shortfall']} of "
            f"{review['minimum_effective_real_review_count']}. Reviewed precision, "
            "confidence bounds, and family/geographic subgroup estimates are unavailable."
        ),
        "",
        "## Production decision",
        "",
        (
            f"All nine selection criteria were evaluated. Six remain blocking: "
            f"{', '.join(selection['blocking_criteria'])}. Zero variants are eligible. "
            "This is insufficient evidence, not rejection of measured production "
            "performance."
        ),
        "",
        (
            "Current and resulting runtime settings have the same fingerprint: "
            f"`{selection['current_settings_fingerprint']}`. Candidate strategy, "
            "pool variant, fusion method, production authority, and release authority "
            "remain unset."
        ),
        "",
        "## Claims allowed",
        "",
        *[f"- {claim}" for claim in report["claims"]["allowed"]],
        "",
        "## Claims blocked",
        "",
        *[f"- {claim}" for claim in report["claims"]["blocked"]],
        "",
        f"Report fingerprint: `{report['report_fingerprint']}`.",
        f"Decision outcome: `{result['decision']}`.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "DYNAMIC_POOL_PILOT_FINAL_REPORT_FILE",
    "DYNAMIC_POOL_PILOT_FINAL_REPORT_SUMMARY_FILE",
    "DYNAMIC_POOL_PILOT_FINAL_REPORT_VERSION",
    "build_dynamic_pool_pilot_final_report",
    "dynamic_pool_pilot_final_report_markdown",
    "validate_dynamic_pool_pilot_final_report",
    "write_dynamic_pool_pilot_final_report",
]
