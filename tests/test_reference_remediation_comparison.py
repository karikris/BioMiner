from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

import polars as pl
import pytest

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.leakage import (
    EVALUATION_IDENTITY_COMPONENT_SCHEMA,
    EVALUATION_IDENTITY_COMPONENT_SCHEMA_VERSION,
)
from biominer.evaluation.reference_escalation import (
    flag_species_for_reference_review,
)
from biominer.evaluation.remediation_comparison import (
    PAIRED_ACCURACY_CHANGE,
    REMEDIATION_COMPUTE_WORK_SCHEMA,
    REMEDIATION_METRIC_CHANGE_SCHEMA,
    REMEDIATION_PAIRED_ITEM_SCHEMA,
    compare_reference_remediation_results,
    remediation_compute_work_frame,
    remediation_pair_bindings_frame,
    remediation_review_effort_frame,
    validate_reference_remediation_comparison,
    write_reference_remediation_comparison,
)
from biominer.evaluation.target_metrics import (
    target_verification_evaluation_frame,
)
from biominer.evaluation.uncertainty import GroupedBootstrapConfig
from biominer.run.flickr_selective_rescore import calculate_flickr_rescore_plan
from test_flickr_selective_rescore import _revision_and_evidence
from test_reference_escalation import _performance, _reference_evidence
from test_target_verification_metrics import _natural_rows
from test_targeted_reference_review import SHA_A


def _comparison():  # noqa: ANN202
    before_rows = _natural_rows()
    after_rows = deepcopy(before_rows)
    after_rows[0].update(
        calibrated_target_probability=0.1,
        classification_decision="other_butterfly",
        target_competitor_margin=-0.5,
    )
    after_rows[2].update(
        calibrated_target_probability=0.1,
        classification_decision="other_butterfly",
        target_competitor_margin=-0.5,
    )
    before = target_verification_evaluation_frame(before_rows)
    after = target_verification_evaluation_frame(after_rows)
    revision, evidence = _revision_and_evidence()
    plan = calculate_flickr_rescore_plan(
        revision,
        evidence,
        margin_impact_band=0.1,
    )
    score_ids = (
        "score:target",
        "score:competitor",
        "score:candidate",
        "score:removed",
        "score:margin",
        "score:unrelated",
    )
    bindings = remediation_pair_bindings_frame(
        [
            {"evaluation_item_id": f"item-{index}", "target_score_id": score_id}
            for index, score_id in enumerate(score_ids, start=1)
        ]
    )
    components = pl.DataFrame(
        [
            {
                "schema_version": EVALUATION_IDENTITY_COMPONENT_SCHEMA_VERSION,
                "register_fingerprint": SHA_A,
                "partition": "natural_stream",
                "bootstrap_component_id": canonical_semantic_fingerprint(
                    {
                        "schema_version": EVALUATION_IDENTITY_COMPONENT_SCHEMA_VERSION,
                        "register_fingerprint": SHA_A,
                        "partition": "natural_stream",
                        "item_ids": [f"item-{index}"],
                    }
                ),
                "component_size": 1,
                "item_id": f"item-{index}",
            }
            for index in range(1, 7)
        ],
        schema=EVALUATION_IDENTITY_COMPONENT_SCHEMA,
        orient="row",
        strict=True,
    )
    compute = remediation_compute_work_frame(
        [
            {
                "work_kind": "target_scoring",
                "unit_name": "records",
                "full_rerun_units": 8.0,
                "incremental_units": 7.0,
                "evidence_basis": "fixture",
            },
            {
                "work_kind": "image_embedding",
                "unit_name": "embeddings",
                "full_rerun_units": 8.0,
                "incremental_units": 0.0,
                "evidence_basis": "fixture",
            },
        ]
    )
    effort = remediation_review_effort_frame(
        [
            {
                "review_kind": "targeted_reference_review",
                "reviewed_item_count": 3,
                "review_minutes": None,
                "evidence_basis": "fixture",
            }
        ]
    )
    escalations = flag_species_for_reference_review(
        _performance(precision_ci_lower=0.4),
        _reference_evidence(),
    )
    result = compare_reference_remediation_results(
        before,
        after,
        bindings,
        plan,
        components,
        escalations,
        compute_work=compute,
        review_effort=effort,
        bootstrap_config=GroupedBootstrapConfig(replicate_count=64),
        generated_at=datetime(2026, 7, 18, tzinfo=UTC),
    )
    return result, before, after, bindings, plan, components, escalations


def test_paired_report_quantifies_selection_outcomes_metrics_and_effort(
    tmp_path,
) -> None:
    result, *_ = _comparison()

    assert result.paired_items.schema == REMEDIATION_PAIRED_ITEM_SCHEMA
    assert result.metric_changes.schema == REMEDIATION_METRIC_CHANGE_SCHEMA
    totals = result.report["totals"]
    assert totals == {
        "records_rescored": 7,
        "records_reused": 1,
        "paired_human_reviewed_records": 6,
        "paired_records_rescored": 5,
        "paired_records_reused": 1,
        "decisions_changed": 2,
        "errors_corrected": 1,
        "new_errors": 1,
        "weighted_decisions_changed": 5.0,
        "weighted_errors_corrected": 3.0,
        "weighted_new_errors": 2.0,
    }
    interval = result.paired_uncertainty.intervals.row(0, named=True)
    assert interval["metric_name"] == PAIRED_ACCURACY_CHANGE
    assert interval["point_estimate"] == pytest.approx(1.0 / 12.0)
    assert interval["bootstrap_replicates"] == 64
    assert result.report["compute_avoided"]["rows"][1][  # type: ignore[index]
        "compute_avoided_units"
    ] in {1.0, 8.0}
    assert result.report["review_effort"]["rows"][0][  # type: ignore[index]
        "review_minutes"
    ] is None
    assert result.report["remaining_flagged_species"] == ["Papilio demoleus"]
    assert len(result.report["evidence_maturity"]["labels"]) == 6
    assert "Point-estimate deltas" in " ".join(result.report["limitations"])

    paths = write_reference_remediation_comparison(result, tmp_path)
    assert set(paths) == {
        "json",
        "markdown",
        "paired_items",
        "metric_changes",
        "paired_intervals",
        "paired_components",
        "compute_work",
        "review_effort",
    }
    assert pl.read_parquet(paths["paired_items"]).equals(result.paired_items)
    assert pl.read_parquet(paths["metric_changes"]).equals(result.metric_changes)
    assert (
        pl.read_parquet(paths["compute_work"]).schema
        == REMEDIATION_COMPUTE_WORK_SCHEMA
    )
    assert paths["markdown"].read_text().startswith(
        "# Reference remediation comparison"
    )


def test_reused_score_must_retain_exact_evaluation_row() -> None:
    _, before, after, bindings, plan, components, escalations = _comparison()
    tampered = after.with_columns(
        pl.when(pl.col("evaluation_item_id") == "item-6")
        .then(pl.lit(0.11))
        .otherwise(pl.col("calibrated_target_probability"))
        .alias("calibrated_target_probability")
    )

    with pytest.raises(ValueError, match="reused target score"):
        compare_reference_remediation_results(
            before,
            tampered,
            bindings,
            plan,
            components,
            escalations,
            bootstrap_config=GroupedBootstrapConfig(replicate_count=8),
        )


def test_pairing_rejects_static_label_drift_and_tampered_result() -> None:
    result, before, after, bindings, plan, components, escalations = _comparison()
    drifted = after.with_columns(
        pl.when(pl.col("evaluation_item_id") == "item-1")
        .then(pl.lit(False))
        .otherwise(pl.col("target_present"))
        .alias("target_present")
    )
    with pytest.raises(ValueError, match="changed static fields"):
        compare_reference_remediation_results(
            before,
            drifted,
            bindings,
            plan,
            components,
            escalations,
            bootstrap_config=GroupedBootstrapConfig(replicate_count=8),
        )

    result.paired_items[0, "after_decision"] = "abstain"
    with pytest.raises(ValueError, match="pair_fingerprint mismatch"):
        validate_reference_remediation_comparison(result)


def test_compute_evidence_rejects_incomparable_or_impossible_claims() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        remediation_compute_work_frame(
            [
                {
                    "work_kind": "scoring",
                    "unit_name": "records",
                    "full_rerun_units": 1.0,
                    "incremental_units": 2.0,
                    "evidence_basis": "measured",
                }
            ]
        )
    with pytest.raises(ValueError, match="evidence bases"):
        remediation_review_effort_frame(
            [
                {
                    "review_kind": "review",
                    "reviewed_item_count": 1,
                    "review_minutes": 2.0,
                    "evidence_basis": "assumed",
                }
            ]
        )
