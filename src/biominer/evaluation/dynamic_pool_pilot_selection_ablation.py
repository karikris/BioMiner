"""Complete production-selection ablation table for the bounded pilot."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re

import polars as pl

from biominer.bioclip.dynamic_pool_fusion import RAW_FUSION_METHODS
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.dynamic_pool_pilot_ablation import (
    build_dynamic_pool_pilot_candidate_ablation,
    validate_dynamic_pool_pilot_candidate_ablation,
)
from biominer.evaluation.dynamic_pool_pilot_plan import (
    PILOT_CANDIDATE_STRATEGIES,
    PILOT_POOL_VARIANTS,
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


DYNAMIC_POOL_PILOT_SELECTION_ABLATION_VERSION = (
    "dynamic-pool-pilot-selection-ablation-v1.0.0"
)
DYNAMIC_POOL_PILOT_SELECTION_ABLATION_REPORT_VERSION = (
    "dynamic-pool-pilot-selection-ablation-report-v1.0.0"
)
DYNAMIC_POOL_PILOT_SELECTION_ABLATION_REPORT_FILE = "production_selection_ablation.json"
DYNAMIC_POOL_PILOT_SELECTION_ABLATION_TABLE_FILE = "production_selection_ablation.csv"

_SORT = ("candidate_strategy", "pool_variant", "fusion_method")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SELECTION_BLOCKERS = [
    "fixture_evidence_ineligible",
    "reviewed_precision_unavailable",
    "family_subgroup_estimates_unavailable",
    "geographic_subgroup_estimates_unavailable",
    "effective_real_review_minimum_unmet",
    "mps_execution_not_run",
]


def dynamic_pool_pilot_selection_ablation_schema() -> dict[str, pl.DataType]:
    """Return one denominator-explicit row per frozen production variant."""

    return {
        "schema_version": pl.String,
        "pilot_id": pl.String,
        "plan_fingerprint": pl.String,
        "candidate_strategy": pl.String,
        "pool_variant": pl.String,
        "fusion_method": pl.String,
        "case_count": pl.UInt32,
        "candidate_set_size": pl.UInt32,
        "candidate_target_recall_at_1": pl.Float64,
        "candidate_target_recall_at_3": pl.Float64,
        "candidate_target_recall_at_5": pl.Float64,
        "fixture_scored_target_at_1_fraction": pl.Float64,
        "located_fixture_scored_target_at_1_fraction": pl.Float64,
        "no_geo_fixture_scored_target_at_1_fraction": pl.Float64,
        "reviewed_precision": pl.Float64,
        "reviewed_precision_lower_bound": pl.Float64,
        "effective_real_reviewed_records": pl.UInt32,
        "minimum_effective_reviewed_records": pl.UInt32,
        "effective_review_shortfall": pl.UInt32,
        "minimum_subgroup_independent_records": pl.UInt32,
        "family_subgroup_status": pl.String,
        "geographic_subgroup_status": pl.String,
        "representative_fixture_work_count": pl.UInt32,
        "targeted_fixture_work_count": pl.UInt32,
        "completed_real_review_count": pl.UInt32,
        "distinct_score_work_count": pl.UInt32,
        "local_evidence_available_case_count": pl.UInt32,
        "execution_query_embedding_count": pl.UInt32,
        "execution_query_embedding_consumption_count": pl.UInt32,
        "execution_query_embedding_reuse_event_count": pl.UInt32,
        "execution_pool_matrix_reference_count": pl.UInt32,
        "execution_unique_pool_matrix_count": pl.UInt32,
        "execution_within_batch_matrix_reuse_count": pl.UInt32,
        "execution_maximum_batch_pool_matrix_bytes": pl.UInt64,
        "mps_peak_memory_bytes": pl.UInt64,
        "mps_memory_limit_bytes": pl.UInt64,
        "mps_memory_status": pl.String,
        "target_preserved_case_count": pl.UInt32,
        "complete_union_preserved_case_count": pl.UInt32,
        "target_pruning_regression_count": pl.UInt32,
        "target_recall_status": pl.String,
        "reviewed_precision_status": pl.String,
        "subgroup_behavior_status": pl.String,
        "review_workload_status": pl.String,
        "computation_status": pl.String,
        "reuse_status": pl.String,
        "target_pruning_status": pl.String,
        "statistical_claim_status": pl.String,
        "classification_accuracy_status": pl.String,
        "raw_score_is_probability": pl.Boolean,
        "unsupported_statistical_claims_present": pl.Boolean,
        "production_default_eligible": pl.Boolean,
        "selection_blockers": pl.List(pl.String),
        "evidence_basis": pl.String,
        "row_fingerprint": pl.String,
    }


def build_dynamic_pool_pilot_selection_ablation(
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
) -> pl.DataFrame:
    """Join all nine selection criteria without manufacturing missing evidence."""

    validate_dynamic_pool_pilot_plan(plan)
    validate_dynamic_pool_pilot_scoring_execution(scoring, plan)
    validate_dynamic_pool_pilot_review_plan(review, plan, scoring)
    candidate = build_dynamic_pool_pilot_candidate_ablation(plan)
    validate_dynamic_pool_pilot_candidate_ablation(candidate, plan)
    rows: list[dict[str, object]] = []
    for strategy in PILOT_CANDIDATE_STRATEGIES:
        candidate_group = candidate.filter(pl.col("strategy_name") == strategy)
        for pool_variant in PILOT_POOL_VARIANTS:
            for fusion_method in RAW_FUSION_METHODS:
                score_group = scoring.results.filter(
                    (pl.col("candidate_strategy") == strategy)
                    & (pl.col("pool_variant") == pool_variant)
                    & (pl.col("fusion_method") == fusion_method)
                )
                base = _selection_ablation_row(
                    plan=plan,
                    scoring=scoring,
                    review=review,
                    candidate_group=candidate_group,
                    score_group=score_group,
                    strategy=strategy,
                    pool_variant=pool_variant,
                    fusion_method=fusion_method,
                )
                rows.append(
                    {
                        **base,
                        "row_fingerprint": canonical_semantic_fingerprint(base),
                    }
                )
    frame = pl.DataFrame(
        rows,
        schema=dynamic_pool_pilot_selection_ablation_schema(),
        orient="row",
        strict=True,
    ).sort(*_SORT)
    validate_dynamic_pool_pilot_selection_ablation(frame, plan, scoring, review)
    return frame


def validate_dynamic_pool_pilot_selection_ablation(
    frame: pl.DataFrame,
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
) -> None:
    """Require complete variants and fail-closed criterion semantics."""

    validate_dynamic_pool_pilot_plan(plan)
    validate_dynamic_pool_pilot_scoring_execution(scoring, plan)
    validate_dynamic_pool_pilot_review_plan(review, plan, scoring)
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("pilot selection ablation must be a Polars DataFrame")
    if frame.schema != dynamic_pool_pilot_selection_ablation_schema():
        raise ValueError("pilot selection ablation schema mismatch")
    if not frame.equals(frame.sort(*_SORT)):
        raise ValueError("pilot selection ablation is not canonically sorted")
    expected_count = (
        len(PILOT_CANDIDATE_STRATEGIES)
        * len(PILOT_POOL_VARIANTS)
        * len(RAW_FUSION_METHODS)
    )
    if (
        frame.height != expected_count
        or frame.select(*_SORT).n_unique() != expected_count
    ):
        raise ValueError("pilot selection ablation variant coverage differs")
    if set(frame["candidate_strategy"]) != set(PILOT_CANDIDATE_STRATEGIES):
        raise ValueError("pilot selection candidate strategies differ")
    if set(frame["pool_variant"]) != set(PILOT_POOL_VARIANTS):
        raise ValueError("pilot selection pool variants differ")
    if set(frame["fusion_method"]) != set(RAW_FUSION_METHODS):
        raise ValueError("pilot selection fusion methods differ")
    for row in frame.to_dicts():
        if row["schema_version"] != DYNAMIC_POOL_PILOT_SELECTION_ABLATION_VERSION:
            raise ValueError("unsupported pilot selection ablation version")
        if row["pilot_id"] != plan["pilot_id"]:
            raise ValueError("pilot selection ablation pilot identity differs")
        if row["plan_fingerprint"] != plan["plan_fingerprint"]:
            raise ValueError("pilot selection ablation plan identity differs")
        if row["reviewed_precision"] is not None:
            raise ValueError("pilot selection ablation fabricated reviewed precision")
        if row["reviewed_precision_lower_bound"] is not None:
            raise ValueError("pilot selection ablation fabricated a precision bound")
        if row["effective_real_reviewed_records"] != 0:
            raise ValueError("pilot selection ablation fabricated real reviews")
        if (
            row["effective_review_shortfall"]
            != row["minimum_effective_reviewed_records"]
        ):
            raise ValueError("pilot selection ablation review shortfall differs")
        if row["mps_peak_memory_bytes"] is not None:
            raise ValueError("pilot selection ablation fabricated MPS memory")
        if row["mps_memory_status"] != "unavailable_cached_vector_fixture_not_mps":
            raise ValueError("pilot selection ablation MPS status differs")
        if row["target_pruning_regression_count"] != 0:
            raise ValueError("pilot selection ablation contains target pruning")
        if row["raw_score_is_probability"] is not False:
            raise ValueError("pilot selection ablation promoted raw scores")
        if row["unsupported_statistical_claims_present"] is not False:
            raise ValueError("pilot selection ablation contains unsupported claims")
        if row["production_default_eligible"] is not False:
            raise ValueError("pilot selection ablation authorized a default")
        if row["selection_blockers"] != _SELECTION_BLOCKERS:
            raise ValueError("pilot selection ablation blockers differ")
        payload = {key: value for key, value in row.items() if key != "row_fingerprint"}
        if row["row_fingerprint"] != canonical_semantic_fingerprint(payload):
            raise ValueError("pilot selection ablation row fingerprint differs")
        if not _is_sha256(row["row_fingerprint"]):
            raise ValueError("pilot selection ablation row fingerprint is invalid")


def build_dynamic_pool_pilot_selection_ablation_report(
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
    frame: pl.DataFrame,
) -> dict[str, object]:
    """Publish all variants and an explicit criterion-availability register."""

    validate_dynamic_pool_pilot_selection_ablation(frame, plan, scoring, review)
    report = _selection_ablation_report_payload(plan, scoring, review, frame)
    report["report_fingerprint"] = canonical_semantic_fingerprint(report)
    validate_dynamic_pool_pilot_selection_ablation_report(
        report, plan, scoring, review, frame
    )
    return report


def validate_dynamic_pool_pilot_selection_ablation_report(
    report: Mapping[str, object],
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
    frame: pl.DataFrame,
) -> None:
    """Require the report to equal one fresh table-derived projection."""

    if (
        report.get("schema_version")
        != DYNAMIC_POOL_PILOT_SELECTION_ABLATION_REPORT_VERSION
    ):
        raise ValueError("unsupported pilot selection ablation report version")
    validate_dynamic_pool_pilot_selection_ablation(frame, plan, scoring, review)
    expected = _selection_ablation_report_payload(plan, scoring, review, frame)
    expected["report_fingerprint"] = canonical_semantic_fingerprint(expected)
    if dict(report) != expected:
        raise ValueError("pilot selection ablation report differs from its table")


def write_dynamic_pool_pilot_selection_ablation_report(
    report: Mapping[str, object],
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
    frame: pl.DataFrame,
    output: str | Path,
) -> Path:
    """Atomically write one validated selection-ablation report."""

    validate_dynamic_pool_pilot_selection_ablation_report(
        report, plan, scoring, review, frame
    )
    destination = Path(output)
    if destination.suffix.casefold() != ".json":
        destination /= DYNAMIC_POOL_PILOT_SELECTION_ABLATION_REPORT_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def publish_dynamic_pool_pilot_selection_ablation_table(
    frame: pl.DataFrame,
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
) -> pl.DataFrame:
    """Return the stable, machine-readable public selection table."""

    validate_dynamic_pool_pilot_selection_ablation(frame, plan, scoring, review)
    return frame.select(
        "candidate_strategy",
        "pool_variant",
        "fusion_method",
        "case_count",
        "candidate_set_size",
        "candidate_target_recall_at_1",
        "candidate_target_recall_at_3",
        "candidate_target_recall_at_5",
        "fixture_scored_target_at_1_fraction",
        "local_evidence_available_case_count",
        "reviewed_precision_lower_bound",
        "family_subgroup_status",
        "geographic_subgroup_status",
        "representative_fixture_work_count",
        "targeted_fixture_work_count",
        "completed_real_review_count",
        "effective_review_shortfall",
        "distinct_score_work_count",
        "execution_query_embedding_reuse_event_count",
        "execution_within_batch_matrix_reuse_count",
        "execution_maximum_batch_pool_matrix_bytes",
        "mps_peak_memory_bytes",
        "mps_memory_status",
        "target_pruning_regression_count",
        "classification_accuracy_status",
        "raw_score_is_probability",
        "unsupported_statistical_claims_present",
        "production_default_eligible",
        "evidence_basis",
        "row_fingerprint",
    )


def write_dynamic_pool_pilot_selection_ablation_table(
    frame: pl.DataFrame,
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
    output: str | Path,
) -> Path:
    """Atomically write the validated public table as CSV."""

    table = publish_dynamic_pool_pilot_selection_ablation_table(
        frame, plan, scoring, review
    )
    destination = Path(output)
    if destination.suffix.casefold() != ".csv":
        destination /= DYNAMIC_POOL_PILOT_SELECTION_ABLATION_TABLE_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    table.write_csv(temporary)
    temporary.replace(destination)
    return destination


def _selection_ablation_row(
    *,
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
    candidate_group: pl.DataFrame,
    score_group: pl.DataFrame,
    strategy: str,
    pool_variant: str,
    fusion_method: str,
) -> dict[str, object]:
    located = score_group.filter(~pl.col("no_geo"))
    no_geo = score_group.filter(pl.col("no_geo"))
    metrics = scoring.batch_result.metrics
    policy = plan["acceptance_policy"]
    return {
        "schema_version": DYNAMIC_POOL_PILOT_SELECTION_ABLATION_VERSION,
        "pilot_id": plan["pilot_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "candidate_strategy": strategy,
        "pool_variant": pool_variant,
        "fusion_method": fusion_method,
        "case_count": score_group.height,
        "candidate_set_size": int(candidate_group["candidate_set_size"].min()),
        "candidate_target_recall_at_1": float(
            candidate_group["target_candidate_recall_at_1"].mean()
        ),
        "candidate_target_recall_at_3": float(
            candidate_group["target_candidate_recall_at_3"].mean()
        ),
        "candidate_target_recall_at_5": float(
            candidate_group["target_candidate_recall_at_5"].mean()
        ),
        "fixture_scored_target_at_1_fraction": float(
            score_group["fixture_expected_target_at_1"].mean()
        ),
        "located_fixture_scored_target_at_1_fraction": float(
            located["fixture_expected_target_at_1"].mean()
        ),
        "no_geo_fixture_scored_target_at_1_fraction": float(
            no_geo["fixture_expected_target_at_1"].mean()
        ),
        "reviewed_precision": None,
        "reviewed_precision_lower_bound": None,
        "effective_real_reviewed_records": 0,
        "minimum_effective_reviewed_records": int(
            policy["minimum_effective_reviewed_records"]
        ),
        "effective_review_shortfall": int(policy["minimum_effective_reviewed_records"]),
        "minimum_subgroup_independent_records": int(
            policy["minimum_subgroup_independent_records"]
        ),
        "family_subgroup_status": "unavailable_no_source_bound_reviews",
        "geographic_subgroup_status": "unavailable_no_source_bound_reviews",
        "representative_fixture_work_count": review.representative.selected_count,
        "targeted_fixture_work_count": review.targeted_queue.height,
        "completed_real_review_count": 0,
        "distinct_score_work_count": score_group[
            "source_result_fingerprint"
        ].n_unique(),
        "local_evidence_available_case_count": score_group.filter(
            pl.col("local_evidence_status") == "available"
        ).height,
        "execution_query_embedding_count": scoring.results[
            "query_embedding_fingerprint"
        ].n_unique(),
        "execution_query_embedding_consumption_count": sum(
            result.cached_query_vectors_consumed
            for result in scoring.batch_result.canonical_results
        ),
        "execution_query_embedding_reuse_event_count": (
            len(scoring.works)
            - scoring.results["query_embedding_fingerprint"].n_unique()
        ),
        "execution_pool_matrix_reference_count": metrics.pool_matrix_references,
        "execution_unique_pool_matrix_count": metrics.unique_pool_matrices,
        "execution_within_batch_matrix_reuse_count": (
            metrics.within_batch_matrix_reuses
        ),
        "execution_maximum_batch_pool_matrix_bytes": (
            metrics.maximum_batch_pool_matrix_bytes
        ),
        "mps_peak_memory_bytes": None,
        "mps_memory_limit_bytes": int(policy["mps_memory_limit_bytes"]),
        "mps_memory_status": "unavailable_cached_vector_fixture_not_mps",
        "target_preserved_case_count": int(candidate_group["target_preserved"].sum()),
        "complete_union_preserved_case_count": int(
            candidate_group["complete_union_preserved"].sum()
        ),
        "target_pruning_regression_count": int(
            (~candidate_group["target_preserved"]).sum()
        ),
        "target_recall_status": "observed_fixture_structural_not_accuracy",
        "reviewed_precision_status": "unavailable_no_completed_real_reviews",
        "subgroup_behavior_status": "unavailable_no_completed_real_reviews",
        "review_workload_status": "planned_fixture_work_not_completed_reviews",
        "computation_status": "observed_cached_vector_fixture_execution",
        "reuse_status": "observed_complete_execution_shared_across_variants",
        "target_pruning_status": "passed_fixture_structural_contract",
        "statistical_claim_status": "passed_no_unsupported_claims",
        "classification_accuracy_status": "unavailable_fixture_only",
        "raw_score_is_probability": False,
        "unsupported_statistical_claims_present": False,
        "production_default_eligible": False,
        "selection_blockers": list(_SELECTION_BLOCKERS),
        "evidence_basis": "fixture_structural_and_cached_vector_execution_only",
    }


def _selection_ablation_report_payload(
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
    frame: pl.DataFrame,
) -> dict[str, object]:
    return {
        "schema_version": DYNAMIC_POOL_PILOT_SELECTION_ABLATION_REPORT_VERSION,
        "pilot_id": plan["pilot_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "scoring_execution_fingerprint": scoring.execution_fingerprint,
        "review_plan_fingerprint": review.review_plan_fingerprint,
        "evidence_basis": "fixture_ablation_not_production_selection_evidence",
        "variant_count": frame.height,
        "candidate_strategy_count": frame["candidate_strategy"].n_unique(),
        "pool_variant_count": frame["pool_variant"].n_unique(),
        "fusion_method_count": frame["fusion_method"].n_unique(),
        "selection_criteria": [
            {
                "criterion": "target_candidate_recall",
                "status": "available_fixture_structural_only",
                "production_selection_eligible": False,
            },
            {
                "criterion": "reviewed_precision_and_confidence_bounds",
                "status": "unavailable_no_completed_real_reviews",
                "production_selection_eligible": False,
            },
            {
                "criterion": "family_and_geographic_subgroup_behavior",
                "status": "unavailable_no_completed_real_reviews",
                "production_selection_eligible": False,
            },
            {
                "criterion": "review_workload",
                "status": "available_planned_fixture_work_only",
                "production_selection_eligible": False,
            },
            {
                "criterion": "computation",
                "status": "available_cached_vector_fixture_execution",
                "production_selection_eligible": False,
            },
            {
                "criterion": "embedding_and_matrix_reuse",
                "status": "available_observed_complete_execution",
                "production_selection_eligible": False,
            },
            {
                "criterion": "mps_memory",
                "status": "unavailable_cached_vector_fixture_not_mps",
                "production_selection_eligible": False,
            },
            {
                "criterion": "target_pruning_regressions",
                "status": "passed_fixture_structural_contract",
                "production_selection_eligible": False,
            },
            {
                "criterion": "unsupported_statistical_claims",
                "status": "passed_no_unsupported_claims",
                "production_selection_eligible": False,
            },
        ],
        "table_rows": frame.to_dicts(),
        "table_fingerprint": canonical_semantic_fingerprint(
            frame["row_fingerprint"].to_list()
        ),
        "selection": {
            "status": "insufficient_evidence",
            "eligible_variant_count": 0,
            "selected_candidate_strategy": None,
            "selected_pool_variant": None,
            "selected_fusion_method": None,
            "production_default_eligible": False,
            "blockers": list(_SELECTION_BLOCKERS),
        },
        "scientific_claims": {
            "fixture_target_recall_is_reviewed_accuracy": False,
            "raw_scores_are_probabilities": False,
            "missing_geography_is_biological_absence": False,
            "targeted_review_is_representative": False,
            "mps_memory_was_measured": False,
            "production_default_selected": False,
            "occurrence_release_authorized": False,
        },
    }


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


__all__ = [
    "DYNAMIC_POOL_PILOT_SELECTION_ABLATION_REPORT_FILE",
    "DYNAMIC_POOL_PILOT_SELECTION_ABLATION_REPORT_VERSION",
    "DYNAMIC_POOL_PILOT_SELECTION_ABLATION_TABLE_FILE",
    "DYNAMIC_POOL_PILOT_SELECTION_ABLATION_VERSION",
    "build_dynamic_pool_pilot_selection_ablation",
    "build_dynamic_pool_pilot_selection_ablation_report",
    "dynamic_pool_pilot_selection_ablation_schema",
    "publish_dynamic_pool_pilot_selection_ablation_table",
    "validate_dynamic_pool_pilot_selection_ablation",
    "validate_dynamic_pool_pilot_selection_ablation_report",
    "write_dynamic_pool_pilot_selection_ablation_report",
    "write_dynamic_pool_pilot_selection_ablation_table",
]
