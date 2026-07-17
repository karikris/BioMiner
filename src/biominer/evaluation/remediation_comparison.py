"""Paired evaluation of results after targeted reference remediation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from math import isfinite
from pathlib import Path
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.reference_escalation import (
    REFERENCE_ESCALATION_SCHEMA,
    validate_reference_escalations,
)
from biominer.evaluation.target_metrics import (
    TARGET_VERIFICATION_EVALUATION_SCHEMA,
    TargetVerificationMetricsConfig,
    compute_target_verification_metrics,
    validate_target_verification_evaluation_frame,
)
from biominer.evaluation.uncertainty import (
    GroupedBootstrapConfig,
    GroupedBootstrapResult,
    build_grouped_metric_confidence_intervals,
    validate_grouped_bootstrap_result,
)
from biominer.ml.nonmatch import TARGET_CONFIRMED
from biominer.reports.evidence_maturity import (
    evidence_maturity_payload,
    validate_evidence_maturity_payload,
)
from biominer.run.flickr_selective_rescore import validate_flickr_rescore_plan
from biominer.storage.parquet import write_parquet


REMEDIATION_COMPARISON_SCHEMA_VERSION = "reference-remediation-comparison-v1.0.0"
REMEDIATION_PAIR_BINDING_SCHEMA_VERSION = "remediation-pair-binding-v1.0.0"
REMEDIATION_PAIRED_ITEM_SCHEMA_VERSION = "remediation-paired-item-v1.0.0"
REMEDIATION_METRIC_CHANGE_SCHEMA_VERSION = "remediation-metric-change-v1.0.0"
REMEDIATION_COMPUTE_WORK_SCHEMA_VERSION = "remediation-compute-work-v1.0.0"
REMEDIATION_REVIEW_EFFORT_SCHEMA_VERSION = "remediation-review-effort-v1.0.0"

REMEDIATION_COMPARISON_JSON_FILE = "reference_remediation_comparison.json"
REMEDIATION_COMPARISON_MARKDOWN_FILE = "reference_remediation_comparison.md"
REMEDIATION_PAIRED_ITEMS_FILE = "reference_remediation_paired_items.parquet"
REMEDIATION_METRIC_CHANGES_FILE = "reference_remediation_metric_changes.parquet"
REMEDIATION_PAIRED_INTERVALS_FILE = "reference_remediation_paired_intervals.parquet"
REMEDIATION_PAIRED_COMPONENTS_FILE = (
    "reference_remediation_paired_components.parquet"
)
REMEDIATION_COMPUTE_WORK_FILE = "reference_remediation_compute_work.parquet"
REMEDIATION_REVIEW_EFFORT_FILE = "reference_remediation_review_effort.parquet"

EVIDENCE_BASES = frozenset({"measured", "estimated", "fixture"})
PAIRED_ACCURACY_CHANGE = "paired_accuracy_change"
POINT_ESTIMATE_POLICY = "after_minus_before_no_directional_improvement_claim"
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")

REMEDIATION_PAIR_BINDING_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "evaluation_item_id": pl.String,
    "target_score_id": pl.String,
    "binding_fingerprint": pl.String,
}

REMEDIATION_PAIRED_ITEM_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "evaluation_item_id": pl.String,
    "evaluation_set": pl.String,
    "sampling_weight": pl.Float64,
    "target_score_id": pl.String,
    "rescore_required": pl.Boolean,
    "rescore_action": pl.String,
    "before_decision": pl.String,
    "after_decision": pl.String,
    "before_abstained": pl.Boolean,
    "after_abstained": pl.Boolean,
    "before_target_predicted": pl.Boolean,
    "after_target_predicted": pl.Boolean,
    "target_present": pl.Boolean,
    "before_correct": pl.Boolean,
    "after_correct": pl.Boolean,
    "decision_changed": pl.Boolean,
    "error_corrected": pl.Boolean,
    "new_error": pl.Boolean,
    "outcome_transition": pl.String,
    "pair_fingerprint": pl.String,
}

REMEDIATION_METRIC_CHANGE_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "evaluation_set": pl.String,
    "scope": pl.String,
    "stratum_dimension": pl.String,
    "stratum_value": pl.String,
    "metric_family": pl.String,
    "metric_name": pl.String,
    "before_metric_value": pl.Float64,
    "after_metric_value": pl.Float64,
    "metric_value_delta": pl.Float64,
    "before_undefined_reason": pl.String,
    "after_undefined_reason": pl.String,
    "point_estimate_policy": pl.String,
    "metric_change_fingerprint": pl.String,
}

REMEDIATION_COMPUTE_WORK_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "work_kind": pl.String,
    "unit_name": pl.String,
    "full_rerun_units": pl.Float64,
    "incremental_units": pl.Float64,
    "compute_avoided_units": pl.Float64,
    "evidence_basis": pl.String,
    "work_fingerprint": pl.String,
}

REMEDIATION_REVIEW_EFFORT_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "review_kind": pl.String,
    "reviewed_item_count": pl.UInt64,
    "review_minutes": pl.Float64,
    "evidence_basis": pl.String,
    "effort_fingerprint": pl.String,
}

_STATIC_EVALUATION_FIELDS = tuple(
    field
    for field in TARGET_VERIFICATION_EVALUATION_SCHEMA
    if field
    not in {
        "calibrated_target_probability",
        "calibration_method",
        "calibration_split_fingerprint",
        "calibrator_fingerprint",
        "classification_decision",
        "abstained",
        "target_competitor_margin",
        "detector_gate_passed",
        "true_family_rank",
        "true_genus_rank",
        "true_species_rank",
        "old_classifier_target_pruned",
    }
)

_METRIC_KEYS = (
    "evaluation_set",
    "scope",
    "stratum_dimension",
    "stratum_value",
    "metric_family",
    "metric_name",
)


@dataclass(frozen=True, slots=True)
class ReferenceRemediationComparison:
    paired_items: pl.DataFrame
    metric_changes: pl.DataFrame
    paired_uncertainty: GroupedBootstrapResult
    compute_work: pl.DataFrame
    review_effort: pl.DataFrame
    report: dict[str, object]
    markdown: str


def remediation_pair_bindings_frame(
    rows: Sequence[Mapping[str, object]] | None = None,
) -> pl.DataFrame:
    normalized: list[dict[str, object]] = []
    for source in rows or ():
        row = dict(source)
        row.setdefault("schema_version", REMEDIATION_PAIR_BINDING_SCHEMA_VERSION)
        row["binding_fingerprint"] = ""
        row["binding_fingerprint"] = _row_fingerprint(
            row, fingerprint_field="binding_fingerprint"
        )
        normalized.append(row)
    frame = pl.DataFrame(
        normalized,
        schema=REMEDIATION_PAIR_BINDING_SCHEMA,
        orient="row",
        strict=True,
    ).sort("evaluation_item_id")
    _validate_fingerprinted_frame(
        frame,
        schema=REMEDIATION_PAIR_BINDING_SCHEMA,
        schema_version=REMEDIATION_PAIR_BINDING_SCHEMA_VERSION,
        key_fields=("evaluation_item_id",),
        fingerprint_field="binding_fingerprint",
        artifact="remediation pair bindings",
    )
    if frame["target_score_id"].n_unique() != frame.height:
        raise ValueError("remediation pair bindings repeat a target score")
    return frame


def remediation_compute_work_frame(
    rows: Sequence[Mapping[str, object]] | None = None,
) -> pl.DataFrame:
    normalized: list[dict[str, object]] = []
    for source in rows or ():
        row = dict(source)
        row.setdefault("schema_version", REMEDIATION_COMPUTE_WORK_SCHEMA_VERSION)
        full = _nonnegative_number(
            row.get("full_rerun_units"), field="full_rerun_units"
        )
        incremental = _nonnegative_number(
            row.get("incremental_units"), field="incremental_units"
        )
        if incremental > full:
            raise ValueError("incremental compute units cannot exceed a full rerun")
        row["full_rerun_units"] = full
        row["incremental_units"] = incremental
        row["compute_avoided_units"] = full - incremental
        row["work_fingerprint"] = ""
        row["work_fingerprint"] = _row_fingerprint(
            row, fingerprint_field="work_fingerprint"
        )
        normalized.append(row)
    frame = pl.DataFrame(
        normalized,
        schema=REMEDIATION_COMPUTE_WORK_SCHEMA,
        orient="row",
        strict=True,
    ).sort("work_kind", "unit_name")
    _validate_fingerprinted_frame(
        frame,
        schema=REMEDIATION_COMPUTE_WORK_SCHEMA,
        schema_version=REMEDIATION_COMPUTE_WORK_SCHEMA_VERSION,
        key_fields=("work_kind", "unit_name"),
        fingerprint_field="work_fingerprint",
        artifact="remediation compute work",
    )
    _validate_evidence_bases(frame)
    return frame


def remediation_review_effort_frame(
    rows: Sequence[Mapping[str, object]] | None = None,
) -> pl.DataFrame:
    normalized: list[dict[str, object]] = []
    for source in rows or ():
        row = dict(source)
        row.setdefault("schema_version", REMEDIATION_REVIEW_EFFORT_SCHEMA_VERSION)
        minutes = row.get("review_minutes")
        row["review_minutes"] = (
            None
            if minutes is None
            else _nonnegative_number(minutes, field="review_minutes")
        )
        count = row.get("reviewed_item_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("reviewed_item_count must be a non-negative integer")
        row["effort_fingerprint"] = ""
        row["effort_fingerprint"] = _row_fingerprint(
            row, fingerprint_field="effort_fingerprint"
        )
        normalized.append(row)
    frame = pl.DataFrame(
        normalized,
        schema=REMEDIATION_REVIEW_EFFORT_SCHEMA,
        orient="row",
        strict=True,
    ).sort("review_kind")
    _validate_fingerprinted_frame(
        frame,
        schema=REMEDIATION_REVIEW_EFFORT_SCHEMA,
        schema_version=REMEDIATION_REVIEW_EFFORT_SCHEMA_VERSION,
        key_fields=("review_kind",),
        fingerprint_field="effort_fingerprint",
        artifact="remediation review effort",
    )
    _validate_evidence_bases(frame)
    return frame


def compare_reference_remediation_results(
    before: pl.DataFrame,
    after: pl.DataFrame,
    bindings: pl.DataFrame,
    rescore_plan: pl.DataFrame,
    bootstrap_components: pl.DataFrame,
    remaining_escalations: pl.DataFrame,
    *,
    compute_work: pl.DataFrame | None = None,
    review_effort: pl.DataFrame | None = None,
    metrics_config: TargetVerificationMetricsConfig | None = None,
    bootstrap_config: GroupedBootstrapConfig | None = None,
    generated_at: datetime | None = None,
) -> ReferenceRemediationComparison:
    """Compare bound, human-labelled records and report incremental work."""

    validate_target_verification_evaluation_frame(before)
    validate_target_verification_evaluation_frame(after)
    validate_flickr_rescore_plan(rescore_plan)
    validate_reference_escalations(remaining_escalations)
    normalized_bindings = remediation_pair_bindings_frame(bindings.to_dicts())
    work = remediation_compute_work_frame(
        None if compute_work is None else compute_work.to_dicts()
    )
    effort = remediation_review_effort_frame(
        None if review_effort is None else review_effort.to_dicts()
    )
    paired = _pair_items(before, after, normalized_bindings, rescore_plan)

    active_metrics = metrics_config or TargetVerificationMetricsConfig()
    before_metrics = compute_target_verification_metrics(before, active_metrics)
    after_metrics = compute_target_verification_metrics(after, active_metrics)
    if (
        before_metrics.configuration_fingerprint
        != after_metrics.configuration_fingerprint
    ):
        raise ValueError("before and after metrics use different configurations")
    changes = _metric_changes(before_metrics.metrics, after_metrics.metrics)

    active_bootstrap = bootstrap_config or active_metrics.bootstrap_config
    point_estimates = {
        (evaluation_set, PAIRED_ACCURACY_CHANGE): _paired_accuracy_change(
            paired.filter(pl.col("evaluation_set") == evaluation_set)
        )
        for evaluation_set in sorted(set(paired["evaluation_set"]))
    }
    paired_input_fingerprint = _frame_fingerprint(paired)
    uncertainty = build_grouped_metric_confidence_intervals(
        paired,
        bootstrap_components,
        metric_names=(PAIRED_ACCURACY_CHANGE,),
        point_estimates=point_estimates,
        metric_evaluator=lambda sample: {
            PAIRED_ACCURACY_CHANGE: _paired_accuracy_change(sample)
        },
        input_fingerprint=paired_input_fingerprint,
        metric_configuration_fingerprint=canonical_semantic_fingerprint(
            {
                "metric": PAIRED_ACCURACY_CHANGE,
                "definition": "sampling_weighted_after_correct_minus_before_correct",
                "pair_schema": REMEDIATION_PAIRED_ITEM_SCHEMA_VERSION,
            }
        ),
        config=active_bootstrap,
    )

    timestamp = _utc_datetime(generated_at or datetime.now(UTC))
    report = _report_payload(
        paired,
        changes,
        uncertainty,
        rescore_plan,
        remaining_escalations,
        work,
        effort,
        before_input_fingerprint=before_metrics.input_fingerprint,
        after_input_fingerprint=after_metrics.input_fingerprint,
        metric_configuration_fingerprint=before_metrics.configuration_fingerprint,
        generated_at=timestamp,
    )
    result = ReferenceRemediationComparison(
        paired_items=paired,
        metric_changes=changes,
        paired_uncertainty=uncertainty,
        compute_work=work,
        review_effort=effort,
        report=report,
        markdown=_markdown(report),
    )
    validate_reference_remediation_comparison(result)
    return result


def validate_reference_remediation_comparison(
    result: ReferenceRemediationComparison,
) -> None:
    _validate_fingerprinted_frame(
        result.paired_items,
        schema=REMEDIATION_PAIRED_ITEM_SCHEMA,
        schema_version=REMEDIATION_PAIRED_ITEM_SCHEMA_VERSION,
        key_fields=("evaluation_set", "evaluation_item_id"),
        fingerprint_field="pair_fingerprint",
        artifact="remediation paired items",
    )
    _validate_paired_item_semantics(result.paired_items)
    _validate_fingerprinted_frame(
        result.metric_changes,
        schema=REMEDIATION_METRIC_CHANGE_SCHEMA,
        schema_version=REMEDIATION_METRIC_CHANGE_SCHEMA_VERSION,
        key_fields=_METRIC_KEYS,
        fingerprint_field="metric_change_fingerprint",
        artifact="remediation metric changes",
    )
    _validate_metric_change_semantics(result.metric_changes)
    normalized_work = remediation_compute_work_frame(result.compute_work.to_dicts())
    if not normalized_work.equals(result.compute_work):
        raise ValueError("remediation compute work does not match derived values")
    normalized_effort = remediation_review_effort_frame(
        result.review_effort.to_dicts()
    )
    if not normalized_effort.equals(result.review_effort):
        raise ValueError("remediation review effort does not match derived values")
    validate_grouped_bootstrap_result(result.paired_uncertainty)
    report = result.report
    if report.get("schema_version") != REMEDIATION_COMPARISON_SCHEMA_VERSION:
        raise ValueError("reference remediation comparison schema mismatch")
    validate_evidence_maturity_payload(report.get("evidence_maturity"))
    payload = dict(report)
    fingerprint = payload.pop("report_fingerprint", None)
    if fingerprint != canonical_semantic_fingerprint(payload):
        raise ValueError("reference remediation comparison fingerprint mismatch")
    if not result.markdown.startswith("# Reference remediation comparison"):
        raise ValueError("reference remediation comparison Markdown mismatch")


def write_reference_remediation_comparison(
    result: ReferenceRemediationComparison,
    output_dir: str | Path,
) -> dict[str, Path]:
    validate_reference_remediation_comparison(result)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": root / REMEDIATION_COMPARISON_JSON_FILE,
        "markdown": root / REMEDIATION_COMPARISON_MARKDOWN_FILE,
        "paired_items": root / REMEDIATION_PAIRED_ITEMS_FILE,
        "metric_changes": root / REMEDIATION_METRIC_CHANGES_FILE,
        "paired_intervals": root / REMEDIATION_PAIRED_INTERVALS_FILE,
        "paired_components": root / REMEDIATION_PAIRED_COMPONENTS_FILE,
        "compute_work": root / REMEDIATION_COMPUTE_WORK_FILE,
        "review_effort": root / REMEDIATION_REVIEW_EFFORT_FILE,
    }
    paths["json"].write_text(
        json.dumps(result.report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["markdown"].write_text(result.markdown, encoding="utf-8")
    write_parquet(result.paired_items, paths["paired_items"])
    write_parquet(result.metric_changes, paths["metric_changes"])
    write_parquet(result.paired_uncertainty.intervals, paths["paired_intervals"])
    write_parquet(
        result.paired_uncertainty.components,
        paths["paired_components"],
    )
    write_parquet(result.compute_work, paths["compute_work"])
    write_parquet(result.review_effort, paths["review_effort"])
    return paths


def _pair_items(
    before: pl.DataFrame,
    after: pl.DataFrame,
    bindings: pl.DataFrame,
    rescore_plan: pl.DataFrame,
) -> pl.DataFrame:
    before_ids = set(before["evaluation_item_id"])
    if set(after["evaluation_item_id"]) != before_ids:
        raise ValueError("before and after evaluation item coverage differs")
    if set(bindings["evaluation_item_id"]) != before_ids:
        raise ValueError("pair bindings do not exactly cover evaluation items")
    plan_by_score = {
        str(row["target_score_id"]): row
        for row in rescore_plan.iter_rows(named=True)
    }
    unknown = sorted(set(bindings["target_score_id"]) - set(plan_by_score))
    if unknown:
        raise ValueError(
            "pair bindings reference unknown target scores: " + ", ".join(unknown)
        )
    before_by_id = {
        str(row["evaluation_item_id"]): row
        for row in before.iter_rows(named=True)
    }
    after_by_id = {
        str(row["evaluation_item_id"]): row
        for row in after.iter_rows(named=True)
    }
    rows: list[dict[str, object]] = []
    for binding in bindings.iter_rows(named=True):
        item_id = str(binding["evaluation_item_id"])
        prior = before_by_id[item_id]
        current = after_by_id[item_id]
        drift = [
            field
            for field in _STATIC_EVALUATION_FIELDS
            if prior[field] != current[field]
        ]
        if drift:
            raise ValueError(
                f"paired evaluation item {item_id} changed static fields: {drift}"
            )
        plan = plan_by_score[str(binding["target_score_id"])]
        if not plan["rescore_required"] and prior != current:
            raise ValueError(
                f"reused target score for {item_id} changed its evaluation row"
            )
        before_target = prior["classification_decision"] == TARGET_CONFIRMED
        after_target = current["classification_decision"] == TARGET_CONFIRMED
        target_present = bool(prior["target_present"])
        before_correct = before_target == target_present
        after_correct = after_target == target_present
        if not before_correct and after_correct:
            transition = "error_corrected"
        elif before_correct and not after_correct:
            transition = "new_error"
        elif before_correct:
            transition = "unchanged_correct"
        else:
            transition = "unchanged_error"
        row: dict[str, object] = {
            "schema_version": REMEDIATION_PAIRED_ITEM_SCHEMA_VERSION,
            "evaluation_item_id": item_id,
            "evaluation_set": prior["evaluation_set"],
            "sampling_weight": prior["sampling_weight"],
            "target_score_id": binding["target_score_id"],
            "rescore_required": plan["rescore_required"],
            "rescore_action": plan["rescore_action"],
            "before_decision": prior["classification_decision"],
            "after_decision": current["classification_decision"],
            "before_abstained": prior["abstained"],
            "after_abstained": current["abstained"],
            "before_target_predicted": before_target,
            "after_target_predicted": after_target,
            "target_present": target_present,
            "before_correct": before_correct,
            "after_correct": after_correct,
            "decision_changed": (
                prior["classification_decision"]
                != current["classification_decision"]
            ),
            "error_corrected": transition == "error_corrected",
            "new_error": transition == "new_error",
            "outcome_transition": transition,
            "pair_fingerprint": "",
        }
        row["pair_fingerprint"] = _row_fingerprint(
            row, fingerprint_field="pair_fingerprint"
        )
        rows.append(row)
    paired = pl.DataFrame(
        rows,
        schema=REMEDIATION_PAIRED_ITEM_SCHEMA,
        orient="row",
        strict=True,
    ).sort("evaluation_set", "evaluation_item_id")
    _validate_fingerprinted_frame(
        paired,
        schema=REMEDIATION_PAIRED_ITEM_SCHEMA,
        schema_version=REMEDIATION_PAIRED_ITEM_SCHEMA_VERSION,
        key_fields=("evaluation_set", "evaluation_item_id"),
        fingerprint_field="pair_fingerprint",
        artifact="remediation paired items",
    )
    return paired


def _metric_changes(before: pl.DataFrame, after: pl.DataFrame) -> pl.DataFrame:
    before_by_key = {
        tuple(row[field] for field in _METRIC_KEYS): row
        for row in before.iter_rows(named=True)
    }
    after_by_key = {
        tuple(row[field] for field in _METRIC_KEYS): row
        for row in after.iter_rows(named=True)
    }
    if set(before_by_key) != set(after_by_key):
        raise ValueError("before and after metric strata do not match")
    rows: list[dict[str, object]] = []
    for key in sorted(before_by_key):
        prior = before_by_key[key]
        current = after_by_key[key]
        before_value = prior["metric_value"]
        after_value = current["metric_value"]
        row = {
            "schema_version": REMEDIATION_METRIC_CHANGE_SCHEMA_VERSION,
            **dict(zip(_METRIC_KEYS, key, strict=True)),
            "before_metric_value": before_value,
            "after_metric_value": after_value,
            "metric_value_delta": (
                None
                if before_value is None or after_value is None
                else float(after_value) - float(before_value)
            ),
            "before_undefined_reason": prior["undefined_reason"],
            "after_undefined_reason": current["undefined_reason"],
            "point_estimate_policy": POINT_ESTIMATE_POLICY,
            "metric_change_fingerprint": "",
        }
        row["metric_change_fingerprint"] = _row_fingerprint(
            row, fingerprint_field="metric_change_fingerprint"
        )
        rows.append(row)
    return pl.DataFrame(
        rows,
        schema=REMEDIATION_METRIC_CHANGE_SCHEMA,
        orient="row",
        strict=True,
    ).sort(*_METRIC_KEYS)


def _paired_accuracy_change(frame: pl.DataFrame) -> float:
    total_weight = float(frame["sampling_weight"].sum())
    deltas = frame.select(
        (
            pl.col("sampling_weight")
            * (
                pl.col("after_correct").cast(pl.Float64)
                - pl.col("before_correct").cast(pl.Float64)
            )
        ).sum()
    ).item()
    return float(deltas) / total_weight


def _report_payload(
    paired: pl.DataFrame,
    changes: pl.DataFrame,
    uncertainty: GroupedBootstrapResult,
    rescore_plan: pl.DataFrame,
    escalations: pl.DataFrame,
    work: pl.DataFrame,
    effort: pl.DataFrame,
    *,
    before_input_fingerprint: str,
    after_input_fingerprint: str,
    metric_configuration_fingerprint: str,
    generated_at: datetime,
) -> dict[str, object]:
    flagged = escalations.filter(pl.col("flagged_for_reference_review"))
    overall_changes = changes.filter(pl.col("scope") == "overall")
    totals = {
        "records_rescored": rescore_plan.filter(pl.col("rescore_required")).height,
        "records_reused": rescore_plan.filter(~pl.col("rescore_required")).height,
        "paired_human_reviewed_records": paired.height,
        "paired_records_rescored": paired.filter(pl.col("rescore_required")).height,
        "paired_records_reused": paired.filter(~pl.col("rescore_required")).height,
        "decisions_changed": int(paired["decision_changed"].sum()),
        "errors_corrected": int(paired["error_corrected"].sum()),
        "new_errors": int(paired["new_error"].sum()),
        "weighted_decisions_changed": _weighted_boolean_sum(
            paired, "decision_changed"
        ),
        "weighted_errors_corrected": _weighted_boolean_sum(
            paired, "error_corrected"
        ),
        "weighted_new_errors": _weighted_boolean_sum(paired, "new_error"),
    }
    intervals = uncertainty.intervals.to_dicts()
    report: dict[str, object] = {
        "schema_version": REMEDIATION_COMPARISON_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "status": "complete",
        "evidence_maturity": evidence_maturity_payload(),
        "totals": totals,
        "paired_accuracy_change": intervals,
        "overall_metric_changes": [
            {
                field: row[field]
                for field in (
                    "evaluation_set",
                    "metric_family",
                    "metric_name",
                    "before_metric_value",
                    "after_metric_value",
                    "metric_value_delta",
                    "before_undefined_reason",
                    "after_undefined_reason",
                    "point_estimate_policy",
                )
            }
            for row in overall_changes.iter_rows(named=True)
        ],
        "compute_avoided": {
            "availability": "unavailable" if work.is_empty() else "supplied",
            "rows": [
                {
                    field: row[field]
                    for field in (
                        "work_kind",
                        "unit_name",
                        "full_rerun_units",
                        "incremental_units",
                        "compute_avoided_units",
                        "evidence_basis",
                    )
                }
                for row in work.iter_rows(named=True)
            ],
            "aggregation_policy": "units_are_not_summed_across_different_unit_names",
        },
        "review_effort": {
            "availability": "unavailable" if effort.is_empty() else "supplied",
            "rows": [
                {
                    field: row[field]
                    for field in (
                        "review_kind",
                        "reviewed_item_count",
                        "review_minutes",
                        "evidence_basis",
                    )
                }
                for row in effort.iter_rows(named=True)
            ],
        },
        "remaining_flagged_species": sorted(set(flagged["target_species"])),
        "remaining_flagged_groups": [
            {
                "target_species": row["target_species"],
                "competitor_species": row["competitor_species"],
                "region": row["region"],
                "route": row["route"],
                "flag_reasons": row["flag_reasons"],
            }
            for row in flagged.iter_rows(named=True)
        ],
        "provenance": {
            "before_evaluation_fingerprint": before_input_fingerprint,
            "after_evaluation_fingerprint": after_input_fingerprint,
            "metric_configuration_fingerprint": metric_configuration_fingerprint,
            "rescore_plan_fingerprint": _frame_fingerprint(rescore_plan),
            "paired_items_fingerprint": _frame_fingerprint(paired),
            "metric_changes_fingerprint": _frame_fingerprint(changes),
            "paired_uncertainty_fingerprint": (
                uncertainty.uncertainty_fingerprint
            ),
            "compute_work_fingerprint": _frame_fingerprint(work),
            "review_effort_fingerprint": _frame_fingerprint(effort),
            "remaining_escalations_fingerprint": _frame_fingerprint(escalations),
        },
        "limitations": [
            "Only records with human-reviewed target-presence labels contribute to corrected-error and new-error counts.",
            "Point-estimate deltas are after minus before and do not alone establish improvement; metric directions differ.",
            "The paired bootstrap resamples complete identity components and quantifies uncertainty only for paired accuracy change.",
            "Compute and review quantities are reported only when supplied with an explicit measured, estimated, or fixture basis.",
            "Remaining statistical flags prioritize review and do not assert a taxonomic identity error.",
        ],
        "report_fingerprint": "",
    }
    report["report_fingerprint"] = _row_fingerprint(
        report, fingerprint_field="report_fingerprint"
    )
    return report


def _markdown(report: Mapping[str, object]) -> str:
    totals = report["totals"]
    compute = report["compute_avoided"]
    review = report["review_effort"]
    assert isinstance(totals, Mapping)
    assert isinstance(compute, Mapping)
    assert isinstance(review, Mapping)
    lines = [
        "# Reference remediation comparison",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Selection and paired outcomes",
        "",
        f"- All production-plan records rescored: {totals['records_rescored']}",
        f"- All production-plan records reused: {totals['records_reused']}",
        f"- Paired human-reviewed records: {totals['paired_human_reviewed_records']}",
        f"- Decisions changed: {totals['decisions_changed']}",
        f"- Errors corrected: {totals['errors_corrected']}",
        f"- New errors: {totals['new_errors']}",
        "",
        "## Compute and review evidence",
        "",
        f"- Compute avoided evidence: `{compute['availability']}`",
        f"- Review effort evidence: `{review['availability']}`",
        "- Unlike compute units are never summed.",
        "",
        "## Remaining review flags",
        "",
        f"- Species still flagged: {len(report['remaining_flagged_species'])}",  # type: ignore[arg-type]
        "",
        "## Interpretation boundaries",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])  # type: ignore[union-attr]
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Report fingerprint: `{report['report_fingerprint']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_fingerprinted_frame(
    frame: pl.DataFrame,
    *,
    schema: Mapping[str, pl.DataType],
    schema_version: str,
    key_fields: Sequence[str],
    fingerprint_field: str,
    artifact: str,
) -> None:
    if frame.schema != schema:
        raise ValueError(f"{artifact} schema mismatch")
    if not frame.equals(frame.sort(*key_fields)):
        raise ValueError(f"{artifact} is not sorted")
    if frame.select(list(key_fields)).n_unique() != frame.height:
        raise ValueError(f"{artifact} repeats its key")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != schema_version:
            raise ValueError(f"unsupported {artifact} schema version")
        for field, value in row.items():
            if field.endswith("_id") or field in {
                "work_kind", "unit_name", "review_kind", "evidence_basis"
            }:
                _required_text(value, field=field)
        fingerprint = str(row[fingerprint_field])
        _sha256(fingerprint, field=fingerprint_field)
        if fingerprint != _row_fingerprint(row, fingerprint_field=fingerprint_field):
            raise ValueError(f"{artifact} {fingerprint_field} mismatch")


def _validate_evidence_bases(frame: pl.DataFrame) -> None:
    invalid = sorted(set(frame["evidence_basis"]) - EVIDENCE_BASES)
    if invalid:
        raise ValueError(f"unsupported remediation evidence bases: {invalid}")


def _validate_paired_item_semantics(frame: pl.DataFrame) -> None:
    if frame.filter(
        pl.col("sampling_weight").is_null()
        | ~pl.col("sampling_weight").is_finite()
        | (pl.col("sampling_weight") <= 0.0)
    ).height:
        raise ValueError("paired item sampling weights must be positive and finite")
    for row in frame.iter_rows(named=True):
        before_target = row["before_decision"] == TARGET_CONFIRMED
        after_target = row["after_decision"] == TARGET_CONFIRMED
        before_correct = before_target == row["target_present"]
        after_correct = after_target == row["target_present"]
        if not before_correct and after_correct:
            transition = "error_corrected"
        elif before_correct and not after_correct:
            transition = "new_error"
        elif before_correct:
            transition = "unchanged_correct"
        else:
            transition = "unchanged_error"
        expected = {
            "before_target_predicted": before_target,
            "after_target_predicted": after_target,
            "before_correct": before_correct,
            "after_correct": after_correct,
            "decision_changed": (
                row["before_decision"] != row["after_decision"]
            ),
            "error_corrected": transition == "error_corrected",
            "new_error": transition == "new_error",
            "outcome_transition": transition,
            "rescore_action": (
                "selectively_rescore"
                if row["rescore_required"]
                else "reuse_prior_score"
            ),
        }
        if any(row[field] != value for field, value in expected.items()):
            raise ValueError("remediation paired item semantics are inconsistent")
        if not row["rescore_required"] and row["decision_changed"]:
            raise ValueError("reused remediation pair changed its decision")


def _validate_metric_change_semantics(frame: pl.DataFrame) -> None:
    for row in frame.iter_rows(named=True):
        before = row["before_metric_value"]
        after = row["after_metric_value"]
        expected_delta = (
            None
            if before is None or after is None
            else float(after) - float(before)
        )
        if (
            row["point_estimate_policy"] != POINT_ESTIMATE_POLICY
            or row["metric_value_delta"] != expected_delta
        ):
            raise ValueError("remediation metric-change semantics are inconsistent")


def _weighted_boolean_sum(frame: pl.DataFrame, field: str) -> float:
    return float(
        frame.select(
            (pl.col("sampling_weight") * pl.col(field).cast(pl.Float64)).sum()
        ).item()
    )


def _nonnegative_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    numeric = float(value)
    if not isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return numeric


def _frame_fingerprint(frame: pl.DataFrame) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema": [(name, str(dtype)) for name, dtype in frame.schema.items()],
            "rows": frame.to_dicts(),
        }
    )


def _row_fingerprint(
    row: Mapping[str, object], *, fingerprint_field: str
) -> str:
    payload = dict(row)
    payload.pop(fingerprint_field)
    return canonical_semantic_fingerprint(payload)


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "")
    if not text or text != text.strip():
        raise ValueError(f"{field} must be canonical nonblank text")
    return text


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full sha256 fingerprint")
    return text


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("generated_at must include a timezone")
    return value.astimezone(UTC)


__all__ = [
    "EVIDENCE_BASES",
    "PAIRED_ACCURACY_CHANGE",
    "REMEDIATION_COMPARISON_JSON_FILE",
    "REMEDIATION_COMPARISON_MARKDOWN_FILE",
    "REMEDIATION_COMPUTE_WORK_SCHEMA",
    "REMEDIATION_METRIC_CHANGE_SCHEMA",
    "REMEDIATION_PAIR_BINDING_SCHEMA",
    "REMEDIATION_PAIRED_COMPONENTS_FILE",
    "REMEDIATION_PAIRED_ITEM_SCHEMA",
    "REMEDIATION_REVIEW_EFFORT_SCHEMA",
    "ReferenceRemediationComparison",
    "compare_reference_remediation_results",
    "remediation_compute_work_frame",
    "remediation_pair_bindings_frame",
    "remediation_review_effort_frame",
    "validate_reference_remediation_comparison",
    "write_reference_remediation_comparison",
]
