"""Validated Papilio adaptive fast-start pilot contract."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.reference_escalation import ReferenceEscalationPolicy
from biominer.references.admission import default_reference_admission_policy


ADAPTIVE_PILOT_PLAN_SCHEMA_VERSION = "adaptive-pilot-plan-v1.0.0"
PAPILIO_DEMOLEUS_TAXON_KEY = "gbif:1938069"
PAPILIO_DEMOLEUS_SCIENTIFIC_NAME = "Papilio demoleus"
REQUIRED_AUTOMATED_GATES = (
    "accepted_taxon_reconciliation",
    "accepted_media_licence",
    "decoded_image_dimensions",
    "canonical_duplicate_resolution",
    "observation_and_observer_independence",
    "yoloe_route_compatibility",
    "minimum_subject_area",
)
REQUIRED_LEAKAGE_IDENTITIES = (
    "source_media_id",
    "content_sha256",
    "duplicate_group_id",
)


def load_adaptive_pilot_plan(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("adaptive pilot plan must contain a JSON object")
    validate_adaptive_pilot_plan(payload)
    return payload


def validate_adaptive_pilot_plan(plan: Mapping[str, object]) -> None:
    if plan.get("schema_version") != ADAPTIVE_PILOT_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported adaptive pilot plan schema version")
    target = _mapping(plan, "target")
    if target != {
        "accepted_taxon_key": PAPILIO_DEMOLEUS_TAXON_KEY,
        "scientific_name": PAPILIO_DEMOLEUS_SCIENTIFIC_NAME,
    }:
        raise ValueError("adaptive pilot target must be Papilio demoleus")

    workflow = _mapping(plan, "reference_workflow")
    admission = default_reference_admission_policy()
    if workflow.get("source") != "gbif":
        raise ValueError("adaptive pilot reference source must be GBIF")
    if workflow.get("admission_mode") != admission.mode:
        raise ValueError("adaptive pilot must use the default admission mode")
    if workflow.get("admission_policy_fingerprint") != admission.fingerprint:
        raise ValueError("adaptive pilot admission policy fingerprint mismatch")
    if tuple(workflow.get("automated_gates", ())) != REQUIRED_AUTOMATED_GATES:
        raise ValueError("adaptive pilot automated gates are incomplete or reordered")
    if workflow.get("support_maturity") != (
        "provider_asserted_provisional_support"
    ):
        raise ValueError("adaptive pilot must label GBIF support provisional")
    if workflow.get("human_review_required_before_first_scoring") is not False:
        raise ValueError("adaptive pilot first scoring must not await reference review")
    if workflow.get("statistical_audit_required") is not True:
        raise ValueError("adaptive pilot requires a statistical audit")

    evaluation = _mapping(plan, "flickr_evaluation")
    audit = _mapping(evaluation, "representative_audit_sample")
    if evaluation.get("source") != "flickr":
        raise ValueError("adaptive pilot evaluation source must be Flickr")
    if audit.get("label_maturity") != "human_reviewed_flickr_labels":
        raise ValueError("adaptive pilot audit requires human-reviewed Flickr labels")
    if audit.get("sampling_frame") != "representative":
        raise ValueError("initial Flickr audit sample must be representative")
    if audit.get("targeted_followup_is_separate") is not True:
        raise ValueError("representative and targeted audit samples must remain separate")
    if not _positive_integer(audit.get("minimum_reviewed_records")):
        raise ValueError("minimum reviewed Flickr records must be positive")
    if evaluation.get("final_release_requires_human_review") is not True:
        raise ValueError("final Flickr release must require human review")

    isolation = _mapping(plan, "holdout_isolation")
    if isolation.get("support_partition") != "support_train":
        raise ValueError("pilot support partition must be support_train")
    if isolation.get("final_test_partition") != "final_test":
        raise ValueError("pilot final-test partition must be final_test")
    if isolation.get("support_source") != "gbif":
        raise ValueError("pilot support isolation source must be GBIF")
    if isolation.get("final_test_source") != "flickr":
        raise ValueError("pilot final-test isolation source must be Flickr")
    if tuple(isolation.get("leakage_identities", ())) != REQUIRED_LEAKAGE_IDENTITIES:
        raise ValueError("pilot leakage identities are incomplete or reordered")
    if isolation.get("cross_source_duplicate_resolution_required") is not True:
        raise ValueError("cross-source duplicate resolution must be required")
    if isolation.get("final_test_may_enter_support") is not False:
        raise ValueError("final-test Flickr media must never enter support")

    escalation = _mapping(plan, "reference_escalation")
    policy = ReferenceEscalationPolicy()
    objectives = _mapping(plan, "quality_objectives")
    if objectives != {
        "precision_lower_bound_minimum": policy.minimum_precision_lower_bound,
        "false_positive_rate_maximum": policy.maximum_false_positive_rate,
        "target_recall_minimum": policy.minimum_target_recall,
        "competitor_confusion_rate_maximum": (
            policy.maximum_competitor_confusion_rate
        ),
    }:
        raise ValueError("pilot quality objectives mismatch production policy")
    if escalation.get("policy_version") != policy.policy_version:
        raise ValueError("pilot escalation policy version mismatch")
    if escalation.get("policy_fingerprint") != policy.fingerprint:
        raise ValueError("pilot escalation policy fingerprint mismatch")
    thresholds = _mapping(escalation, "thresholds")
    expected_thresholds = {
        field: getattr(policy, field)
        for field in policy.__dataclass_fields__
        if field not in {"schema_version", "policy_version"}
    }
    if thresholds != expected_thresholds:
        raise ValueError("pilot escalation thresholds mismatch production policy")
    if escalation.get("identity_conclusion") != "not_assessed":
        raise ValueError("statistical escalation must not claim reference identity")
    if escalation.get("review_scope") != "flagged_species_only":
        raise ValueError("adaptive pilot reference review must remain targeted")

    execution = _mapping(plan, "execution")
    if execution.get("initial_sequence") != [
        "admit_gbif_provisional_support",
        "embed_reference_images",
        "build_provisional_prototypes",
        "score_flickr_provisionally",
    ]:
        raise ValueError("adaptive pilot initial sequence mismatch")
    if execution.get("post_scoring_sequence") != [
        "human_review_representative_flickr_sample",
        "run_species_level_statistical_audit",
        "review_flagged_references_only",
        "revise_affected_reference_bank",
        "selectively_rescore_affected_flickr_records",
    ]:
        raise ValueError("adaptive pilot post-scoring sequence mismatch")

    expected_fingerprint = canonical_semantic_fingerprint(
        {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    )
    if plan.get("plan_fingerprint") != expected_fingerprint:
        raise ValueError("adaptive pilot plan fingerprint mismatch")


def validate_pilot_partition_isolation(assignments: pl.DataFrame) -> None:
    required = {
        "source_media_id",
        "source",
        "content_sha256",
        "duplicate_group_id",
        "partition",
    }
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"pilot partition assignments missing columns: {sorted(missing)}")
    if assignments.is_empty():
        raise ValueError("pilot partition assignments must not be empty")
    support = assignments.filter(pl.col("partition") == "support_train")
    final_test = assignments.filter(pl.col("partition") == "final_test")
    if support.is_empty() or final_test.is_empty():
        raise ValueError("pilot requires both support_train and final_test assignments")
    if support.filter(pl.col("source") != "gbif").height:
        raise ValueError("pilot support_train may contain only GBIF media")
    if final_test.filter(pl.col("source") != "flickr").height:
        raise ValueError("pilot final_test may contain only Flickr media")
    for identity in REQUIRED_LEAKAGE_IDENTITIES:
        support_values = set(support[identity].drop_nulls().to_list())
        final_values = set(final_test[identity].drop_nulls().to_list())
        overlap = sorted(support_values & final_values)
        if overlap:
            raise ValueError(
                f"pilot support/final-test leakage through {identity}: {overlap}"
            )


def _mapping(parent: Mapping[str, object], field: str) -> Mapping[str, object]:
    value = parent.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"adaptive pilot {field} must be an object")
    return value


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
