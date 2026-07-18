"""Representative and targeted review-work plan for the bounded pilot."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.dynamic_pool_pilot_plan import (
    validate_dynamic_pool_pilot_plan,
)
from biominer.evaluation.dynamic_pool_pilot_scoring import (
    DynamicPoolPilotScoringExecution,
    validate_dynamic_pool_pilot_scoring_execution,
)
from biominer.evaluation.dynamic_pool_review import (
    FailureDiscoveryPolicy,
    OccurrenceReleaseReviewPolicy,
    ProbabilityAuditSamplingPolicy,
    ProbabilityAuditSelection,
    build_dynamic_pool_audit_frame,
    build_failure_discovery_queue,
    build_occurrence_release_review_queue,
    build_probability_audit_sample,
    validate_dynamic_pool_audit_frame,
    validate_failure_discovery_queue,
    validate_occurrence_release_review_queue,
    validate_probability_audit_selection,
)


DYNAMIC_POOL_PILOT_REVIEW_PLAN_VERSION = "dynamic-pool-pilot-review-plan-v1.0.0"
DYNAMIC_POOL_PILOT_REVIEW_REPORT_VERSION = "dynamic-pool-pilot-review-report-v1.0.0"
DYNAMIC_POOL_PILOT_REVIEW_REPORT_FILE = "dynamic_pool_review_plan.json"
PILOT_REVIEW_PROJECTION = {
    "candidate_strategy": "parallel_family_geography_union",
    "pool_variant": "dynamic_global_local",
    "fusion_method": "unweighted_component_mean",
    "authority": "projection_view_not_selected_production_configuration",
}

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class DynamicPoolPilotReviewPlan:
    """Production review contracts populated only with fixture work items."""

    source_scoring_execution_fingerprint: str
    audit_frame: pl.DataFrame
    representative: ProbabilityAuditSelection
    targeted_queue: pl.DataFrame
    release_queue: pl.DataFrame
    review_plan_fingerprint: str


def build_dynamic_pool_pilot_review_plan(
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
) -> DynamicPoolPilotReviewPlan:
    """Build review work while leaving assignments and outcomes unavailable."""

    validate_dynamic_pool_pilot_plan(plan)
    validate_dynamic_pool_pilot_scoring_execution(scoring, plan)
    audit_frame = build_dynamic_pool_audit_frame(_audit_candidates(plan, scoring))
    representative = build_probability_audit_sample(
        audit_frame,
        policy=ProbabilityAuditSamplingPolicy(
            review_budget=len(plan["cases"]),
            minimum_per_nonempty_stratum=1,
            random_seed=int(plan["execution_limits"]["random_seed"]),
        ),
    )
    targeted_queue = build_failure_discovery_queue(
        audit_frame,
        policy=FailureDiscoveryPolicy(
            near_margin_cutoff=0.05,
            high_disagreement_cutoff=1e-12,
            low_score_cutoff=0.50,
            small_subject_cutoff=0.10,
            max_queue_size=len(plan["cases"]),
        ),
    )
    release_queue = build_occurrence_release_review_queue(
        audit_frame,
        policy=OccurrenceReleaseReviewPolicy(
            require_second_review=True,
            adjudication_on_conflict=True,
        ),
    )
    provisional = DynamicPoolPilotReviewPlan(
        source_scoring_execution_fingerprint=scoring.execution_fingerprint,
        audit_frame=audit_frame,
        representative=representative,
        targeted_queue=targeted_queue,
        release_queue=release_queue,
        review_plan_fingerprint="",
    )
    result = DynamicPoolPilotReviewPlan(
        source_scoring_execution_fingerprint=(
            provisional.source_scoring_execution_fingerprint
        ),
        audit_frame=provisional.audit_frame,
        representative=provisional.representative,
        targeted_queue=provisional.targeted_queue,
        release_queue=provisional.release_queue,
        review_plan_fingerprint=canonical_semantic_fingerprint(
            _review_plan_identity(provisional)
        ),
    )
    validate_dynamic_pool_pilot_review_plan(result, plan, scoring)
    return result


def validate_dynamic_pool_pilot_review_plan(
    review: DynamicPoolPilotReviewPlan,
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
) -> None:
    """Validate design separation, zero outcomes, and fail-closed release state."""

    validate_dynamic_pool_pilot_plan(plan)
    validate_dynamic_pool_pilot_scoring_execution(scoring, plan)
    if not isinstance(review, DynamicPoolPilotReviewPlan):
        raise TypeError("pilot review plan has the wrong type")
    if review.source_scoring_execution_fingerprint != scoring.execution_fingerprint:
        raise ValueError("pilot review plan scoring identity differs")
    validate_dynamic_pool_audit_frame(review.audit_frame)
    validate_probability_audit_selection(
        review.representative.register,
        review.representative.sample,
    )
    validate_failure_discovery_queue(review.targeted_queue)
    validate_occurrence_release_review_queue(
        review.release_queue,
        source_frame=review.audit_frame,
    )
    case_count = len(plan["cases"])
    if review.audit_frame.height != case_count:
        raise ValueError("pilot review audit-frame coverage differs")
    if review.representative.population_count != case_count:
        raise ValueError("pilot representative population count differs")
    if review.representative.selected_count != case_count:
        raise ValueError("pilot representative fixture selection is incomplete")
    if review.targeted_queue.height != case_count:
        raise ValueError("pilot targeted fixture queue coverage differs")
    if review.release_queue.height:
        raise ValueError("pilot fixture items cannot enter occurrence release review")
    register = review.representative.register
    if set(register["inclusion_probability"]) != {1.0}:
        raise ValueError("pilot representative inclusion probabilities differ")
    if set(register["sampling_weight"]) != {1.0}:
        raise ValueError("pilot representative sampling weights differ")
    if not all(register["representative_estimation_eligible"]):
        raise ValueError("pilot probability design lost within-fixture eligibility")
    targeted = review.targeted_queue
    if targeted["inclusion_probability"].null_count() != targeted.height:
        raise ValueError("pilot targeted queue acquired inclusion probabilities")
    if targeted["sampling_weight"].null_count() != targeted.height:
        raise ValueError("pilot targeted queue acquired sampling weights")
    if any(targeted["representative_estimation_eligible"]):
        raise ValueError("pilot targeted queue entered representative estimation")
    if any(targeted["release_authorized"]):
        raise ValueError("pilot targeted queue acquired release authority")
    expected = canonical_semantic_fingerprint(_review_plan_identity(review))
    if review.review_plan_fingerprint != expected:
        raise ValueError("pilot review plan fingerprint differs")


def build_dynamic_pool_pilot_review_report(
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
) -> dict[str, object]:
    """Summarize review work and its real-evidence shortfall."""

    validate_dynamic_pool_pilot_review_plan(review, plan, scoring)
    report = _review_report_payload(plan, scoring, review)
    report["report_fingerprint"] = canonical_semantic_fingerprint(report)
    validate_dynamic_pool_pilot_review_report(report, plan, scoring, review)
    return report


def validate_dynamic_pool_pilot_review_report(
    report: Mapping[str, object],
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
) -> None:
    """Require the report to equal a fresh plan-derived summary."""

    if report.get("schema_version") != DYNAMIC_POOL_PILOT_REVIEW_REPORT_VERSION:
        raise ValueError("unsupported pilot review report version")
    validate_dynamic_pool_pilot_review_plan(review, plan, scoring)
    expected = _review_report_payload(plan, scoring, review)
    expected["report_fingerprint"] = canonical_semantic_fingerprint(expected)
    if dict(report) != expected:
        raise ValueError("pilot review report differs from its plan")


def write_dynamic_pool_pilot_review_report(
    report: Mapping[str, object],
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
    output: str | Path,
) -> Path:
    """Atomically write one validated review-work report."""

    validate_dynamic_pool_pilot_review_report(report, plan, scoring, review)
    destination = Path(output)
    if destination.suffix.casefold() != ".json":
        destination /= DYNAMIC_POOL_PILOT_REVIEW_REPORT_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _audit_candidates(
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
) -> list[dict[str, object]]:
    frame = scoring.results
    selected = frame.filter(
        (pl.col("candidate_strategy") == PILOT_REVIEW_PROJECTION["candidate_strategy"])
        & (pl.col("pool_variant") == PILOT_REVIEW_PROJECTION["pool_variant"])
        & (pl.col("fusion_method") == PILOT_REVIEW_PROJECTION["fusion_method"])
    ).sort("case_id")
    global_control = frame.filter(
        (pl.col("candidate_strategy") == PILOT_REVIEW_PROJECTION["candidate_strategy"])
        & (pl.col("pool_variant") == "global_only_control")
        & (pl.col("fusion_method") == PILOT_REVIEW_PROJECTION["fusion_method"])
    ).select("case_id", pl.col("target_raw_fusion_score").alias("global_score"))
    selected = selected.join(global_control, on="case_id", validate="1:1")
    catalog = {
        str(taxon["accepted_taxon_key"]): taxon for taxon in plan["taxon_catalog"]
    }
    source_artifact_fingerprint = canonical_semantic_fingerprint(
        frame["result_fingerprint"].to_list()
    )
    rows: list[dict[str, object]] = []
    for row in selected.to_dicts():
        taxon = catalog[str(row["target_accepted_taxon_key"])]
        case_id = str(row["case_id"])
        rows.append(
            {
                "sampling_unit_id": f"pilot-review-unit:{case_id}",
                "source_record_hash": canonical_semantic_fingerprint(
                    ["pilot-fixture-source-record", row["fixture_media_id"]]
                ),
                "source_artifact_fingerprint": source_artifact_fingerprint,
                "flickr_photo_id": row["fixture_media_id"],
                "organism_unit_id": f"pilot-organism:{case_id}",
                "candidate_family_accepted_taxon_key": taxon["family_key"],
                "candidate_family_scientific_name": taxon["family"],
                "candidate_genus_accepted_taxon_key": taxon["genus_key"],
                "candidate_genus_scientific_name": taxon["genus"],
                "candidate_species_accepted_taxon_key": taxon["accepted_taxon_key"],
                "candidate_species_scientific_name": taxon["scientific_name"],
                "geographic_cluster_id": row["region_id"],
                "no_geo": row["no_geo"],
                "primary_query_tier": "T1",
                "raw_fusion_score": row["target_raw_fusion_score"],
                "raw_competitor_margin": row["top_margin_raw"],
                "pool_disagreement": abs(
                    float(row["target_raw_fusion_score"]) - float(row["global_score"])
                ),
                "route": "adult_field",
                "visual_domain": "fixture_vector_no_image",
                "subject_area_ratio": 0.5,
                "owner_group_id": f"pilot-owner-group:{case_id}",
                "duplicate_group_id": f"pilot-duplicate-group:{case_id}",
                "observation_group_id": f"pilot-observation-group:{case_id}",
                "final_release_candidate": False,
            }
        )
    return rows


def _review_report_payload(
    plan: Mapping[str, object],
    scoring: DynamicPoolPilotScoringExecution,
    review: DynamicPoolPilotReviewPlan,
) -> dict[str, object]:
    representative_ids = set(review.representative.sample["sampling_unit_id"])
    targeted_ids = set(review.targeted_queue["sampling_unit_id"])
    reasons = Counter(
        reason
        for row_reasons in review.targeted_queue["priority_reasons"].to_list()
        for reason in row_reasons
    )
    required_effective = int(
        plan["acceptance_policy"]["minimum_effective_reviewed_records"]
    )
    return {
        "schema_version": DYNAMIC_POOL_PILOT_REVIEW_REPORT_VERSION,
        "pilot_id": plan["pilot_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "source_scoring_execution_fingerprint": scoring.execution_fingerprint,
        "review_plan_fingerprint": review.review_plan_fingerprint,
        "evidence_basis": "fixture_review_work_plan_not_human_outcomes",
        "projection_view": dict(PILOT_REVIEW_PROJECTION),
        "audit_frame": {
            "fixture_item_count": review.audit_frame.height,
            "independence_component_count": review.representative.population_count,
            "analysis_stratum_count": review.audit_frame[
                "analysis_stratum_id"
            ].n_unique(),
            "source_bound_real_flickr_item_count": 0,
            "subject_area_ratio": "fixture_control_value_not_measured",
            "raw_score_is_probability": False,
        },
        "representative_review": {
            "sampling_design": (
                "stratified_srs_without_replacement_of_connected_"
                "duplicate_observation_components"
            ),
            "population_count": review.representative.population_count,
            "selected_count": review.representative.selected_count,
            "register_fingerprint": review.representative.register_fingerprint,
            "inclusion_probability_minimum": float(
                review.representative.register["inclusion_probability"].min()
            ),
            "inclusion_probability_maximum": float(
                review.representative.register["inclusion_probability"].max()
            ),
            "sampling_weight_minimum": float(
                review.representative.register["sampling_weight"].min()
            ),
            "sampling_weight_maximum": float(
                review.representative.register["sampling_weight"].max()
            ),
            "within_fixture_design_estimation_eligible": True,
            "real_biological_estimation_eligible": False,
            "reason": "fixture_items_have_no_source_bound_human_labels",
        },
        "targeted_review": {
            "queue_kind": "targeted_failure_discovery",
            "selected_count": review.targeted_queue.height,
            "priority_reason_counts": dict(sorted(reasons.items())),
            "inclusion_probabilities_available": False,
            "sampling_weights_available": False,
            "representative_estimation_eligible": False,
            "release_authorized": False,
        },
        "workload": {
            "unique_fixture_item_count": len(representative_ids | targeted_ids),
            "representative_and_targeted_overlap_count": len(
                representative_ids & targeted_ids
            ),
            "purposes_merged": False,
            "reviewer_identity_count": 0,
            "assignment_count": 0,
            "completed_review_count": 0,
            "decisive_review_count": 0,
            "adjudication_count": 0,
        },
        "production_evidence_gap": {
            "minimum_effective_reviewed_records": required_effective,
            "real_effective_reviewed_records": 0,
            "remaining_effective_review_shortfall": required_effective,
            "minimum_subgroup_independent_records": plan["acceptance_policy"][
                "minimum_subgroup_independent_records"
            ],
            "reviewed_precision_lower_bound": None,
            "reviewed_precision_status": "unavailable_no_completed_real_reviews",
            "statistical_support_status": "insufficient_evidence",
        },
        "release": {
            "occurrence_release_review_queue_count": review.release_queue.height,
            "fixture_items_are_release_candidates": False,
            "release_ready_count": 0,
            "release_authorized": False,
        },
        "selection": {
            "status": "insufficient_evidence",
            "production_default_eligible": False,
            "reason": "review_work_is_not_completed_source_bound_evidence",
        },
        "scientific_claims": {
            "selected_review_work_is_completed_review": False,
            "targeted_review_is_representative": False,
            "fixture_reviews_satisfy_real_review_minimum": False,
            "raw_scores_are_probabilities": False,
            "missing_geography_is_biological_absence": False,
            "occurrence_release_authorized": False,
        },
    }


def _review_plan_identity(review: DynamicPoolPilotReviewPlan) -> dict[str, object]:
    return {
        "schema_version": DYNAMIC_POOL_PILOT_REVIEW_PLAN_VERSION,
        "source_scoring_execution_fingerprint": (
            review.source_scoring_execution_fingerprint
        ),
        "audit_frame_fingerprint": (
            review.audit_frame["frame_fingerprint"].item(0)
            if review.audit_frame.height
            else None
        ),
        "representative_register_fingerprint": (
            review.representative.register_fingerprint
        ),
        "representative_sample_unit_fingerprints": review.representative.sample[
            "audit_unit_fingerprint"
        ].to_list(),
        "targeted_queue_fingerprint": (
            review.targeted_queue["failure_queue_fingerprint"].item(0)
            if review.targeted_queue.height
            else None
        ),
        "release_queue_fingerprint": (
            review.release_queue["release_review_queue_fingerprint"].item(0)
            if review.release_queue.height
            else None
        ),
        "reviewer_identities": [],
        "assignments": [],
        "completed_reviews": [],
        "release_authorized": False,
    }


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


__all__ = [
    "DYNAMIC_POOL_PILOT_REVIEW_PLAN_VERSION",
    "DYNAMIC_POOL_PILOT_REVIEW_REPORT_FILE",
    "DYNAMIC_POOL_PILOT_REVIEW_REPORT_VERSION",
    "DynamicPoolPilotReviewPlan",
    "PILOT_REVIEW_PROJECTION",
    "build_dynamic_pool_pilot_review_plan",
    "build_dynamic_pool_pilot_review_report",
    "validate_dynamic_pool_pilot_review_plan",
    "validate_dynamic_pool_pilot_review_report",
    "write_dynamic_pool_pilot_review_report",
]
