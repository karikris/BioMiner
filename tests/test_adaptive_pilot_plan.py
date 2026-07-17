from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import polars as pl
import pytest

from biominer.evaluation.reference_escalation import ReferenceEscalationPolicy
from biominer.references.admission import default_reference_admission_policy
from biominer.run.adaptive_pilot import (
    REQUIRED_AUTOMATED_GATES,
    load_adaptive_pilot_plan,
    validate_adaptive_pilot_plan,
    validate_pilot_partition_isolation,
)


PLAN_PATH = Path(
    "config/pilot/papilio_demoleus_adaptive_gbif_fast_start.json"
)


def test_papilio_plan_binds_production_policies_and_fast_start_sequence() -> None:
    plan = load_adaptive_pilot_plan(PLAN_PATH)
    workflow = plan["reference_workflow"]
    evaluation = plan["flickr_evaluation"]
    escalation = plan["reference_escalation"]

    assert plan["target"] == {
        "accepted_taxon_key": "gbif:1938069",
        "scientific_name": "Papilio demoleus",
    }
    assert workflow["source"] == "gbif"
    assert workflow["admission_mode"] == "adaptive_gbif_fast_start"
    assert tuple(workflow["automated_gates"]) == REQUIRED_AUTOMATED_GATES
    assert workflow["human_review_required_before_first_scoring"] is False
    assert workflow["statistical_audit_required"] is True
    assert workflow["admission_policy_fingerprint"] == (
        default_reference_admission_policy().fingerprint
    )
    assert evaluation["representative_audit_sample"][
        "label_maturity"
    ] == "human_reviewed_flickr_labels"
    assert evaluation["representative_audit_sample"][
        "targeted_followup_is_separate"
    ] is True
    assert escalation["policy_fingerprint"] == (
        ReferenceEscalationPolicy().fingerprint
    )
    assert escalation["identity_conclusion"] == "not_assessed"
    assert plan["execution"]["initial_sequence"][-1] == (
        "score_flickr_provisionally"
    )


def test_plan_fails_closed_when_scientific_boundaries_are_weakened() -> None:
    plan = load_adaptive_pilot_plan(PLAN_PATH)
    changes = (
        ("reference_workflow", "source", "flickr"),
        (
            "reference_workflow",
            "human_review_required_before_first_scoring",
            True,
        ),
        ("reference_workflow", "statistical_audit_required", False),
        ("flickr_evaluation", "final_release_requires_human_review", False),
        ("holdout_isolation", "final_test_may_enter_support", True),
        ("reference_escalation", "identity_conclusion", "misidentified"),
        ("reference_escalation", "review_scope", "all_species"),
    )
    for section, field, value in changes:
        tampered = deepcopy(plan)
        tampered[section][field] = value
        with pytest.raises(ValueError):
            validate_adaptive_pilot_plan(tampered)


def test_cross_source_mirrors_cannot_cross_support_and_final_test() -> None:
    valid = pl.DataFrame(
        [
            {
                "source_media_id": "gbif:media:1",
                "source": "gbif",
                "content_sha256": "sha256:" + "a" * 64,
                "duplicate_group_id": "duplicate:gbif:1",
                "partition": "support_train",
            },
            {
                "source_media_id": "flickr:photo:2",
                "source": "flickr",
                "content_sha256": "sha256:" + "b" * 64,
                "duplicate_group_id": "duplicate:flickr:2",
                "partition": "final_test",
            },
        ]
    )
    validate_pilot_partition_isolation(valid)

    mirrored = valid.with_columns(
        pl.when(pl.col("partition") == "final_test")
        .then(pl.lit("sha256:" + "a" * 64))
        .otherwise(pl.col("content_sha256"))
        .alias("content_sha256")
    )
    with pytest.raises(ValueError, match="content_sha256"):
        validate_pilot_partition_isolation(mirrored)
