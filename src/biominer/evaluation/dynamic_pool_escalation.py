"""Fail-closed remediation triggers for hierarchical dynamic-pool audits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isfinite

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.dynamic_pool_quality import (
    QUALITY_REPORT_SCHEMA,
    validate_dynamic_pool_quality_report,
)


DYNAMIC_POOL_ESCALATION_POLICY_VERSION = "dynamic-pool-escalation-policy-v1.0.0"
DYNAMIC_POOL_ESCALATION_REPORT_VERSION = "dynamic-pool-escalation-report-v1.0.0"

TRIGGER_RULE_SCHEMA = pl.Struct(
    {
        "reason": pl.String,
        "metric_name": pl.String,
        "comparison_basis": pl.String,
        "observed": pl.Float64,
        "operator": pl.String,
        "threshold": pl.Float64,
        "source_quality_row_fingerprint": pl.String,
        "recommended_action": pl.String,
    }
)

POOLING_ESCALATION_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "report_fingerprint": pl.String,
    "decision_fingerprint": pl.String,
    "policy_fingerprint": pl.String,
    "source_quality_report_fingerprint": pl.String,
    "source_quality_row_fingerprints": pl.List(pl.String),
    "hierarchy_level": pl.String,
    "geography_level": pl.String,
    "group_id": pl.String,
    "group_label": pl.String,
    "family_key": pl.String,
    "family_name": pl.String,
    "genus_key": pl.String,
    "genus_name": pl.String,
    "species_key": pl.String,
    "scientific_name": pl.String,
    "geography_value": pl.String,
    "no_geo": pl.Boolean,
    "quality_group_status": pl.String,
    "escalation_status": pl.String,
    "flagged_for_remediation": pl.Boolean,
    "trigger_reasons": pl.List(pl.String),
    "triggered_rules": pl.List(TRIGGER_RULE_SCHEMA),
    "recommended_actions": pl.List(pl.String),
    "reference_review_candidate": pl.Boolean,
    "additional_flickr_audit_candidate": pl.Boolean,
    "human_review_required": pl.Boolean,
    "automatic_reference_disposition": pl.Boolean,
    "taxon_misidentification_conclusion": pl.String,
    "authorizes_occurrence_release": pl.Boolean,
}


@dataclass(frozen=True, slots=True)
class DynamicPoolEscalationPolicy:
    """Versioned objectives that trigger review, never automatic mutation."""

    schema_version: str = DYNAMIC_POOL_ESCALATION_POLICY_VERSION
    minimum_precision_lower_bound: float = 0.95
    maximum_family_routing_error_rate: float = 0.10
    maximum_global_local_disagreement_rate: float = 0.20
    maximum_local_support_insufficiency_rate: float = 0.20
    maximum_reference_outlier_error_influence_rate: float = 0.20
    maximum_weighted_brier_score: float = 0.10
    maximum_weighted_ece: float = 0.10
    maximum_ood_false_positive_incidence: float = 0.05

    def __post_init__(self) -> None:
        if self.schema_version != DYNAMIC_POOL_ESCALATION_POLICY_VERSION:
            raise ValueError("unsupported dynamic-pool escalation policy version")
        for field in (
            "minimum_precision_lower_bound",
            "maximum_family_routing_error_rate",
            "maximum_global_local_disagreement_rate",
            "maximum_local_support_insufficiency_rate",
            "maximum_reference_outlier_error_influence_rate",
            "maximum_weighted_brier_score",
            "maximum_weighted_ece",
            "maximum_ood_false_positive_incidence",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{field} must be numeric")
            normalized = float(value)
            if not isfinite(normalized) or not 0.0 <= normalized <= 1.0:
                raise ValueError(f"{field} must be finite within [0, 1]")
            object.__setattr__(self, field, normalized)

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(asdict(self))


def define_pooling_escalations(
    quality_reports: Sequence[pl.DataFrame],
    *,
    policy: DynamicPoolEscalationPolicy | None = None,
) -> pl.DataFrame:
    """Convert audited quality failures into non-mutating remediation actions."""

    selected_policy = policy or DynamicPoolEscalationPolicy()
    reports = tuple(quality_reports)
    if not reports:
        raise ValueError("pooling escalation requires at least one quality report")
    for report in reports:
        validate_dynamic_pool_quality_report(report)
    combined = pl.concat(reports, how="vertical")
    if combined.schema != QUALITY_REPORT_SCHEMA:
        raise ValueError("pooling escalation quality schemas are inconsistent")
    if combined.select(["hierarchy_level", "group_id", "metric_name"]).n_unique() != (
        combined.height
    ):
        raise ValueError("pooling escalation quality groups overlap")

    semantic_rows = []
    for _, group in combined.group_by(
        ["hierarchy_level", "group_id"], maintain_order=False
    ):
        semantic_rows.append(_escalation_row(group, policy=selected_policy))
    semantic_rows.sort(
        key=lambda row: (str(row["hierarchy_level"]), str(row["group_id"]))
    )
    decision_fingerprints = [
        canonical_semantic_fingerprint(row) for row in semantic_rows
    ]
    report_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": DYNAMIC_POOL_ESCALATION_REPORT_VERSION,
            "policy_fingerprint": selected_policy.fingerprint,
            "source_quality_report_fingerprints": sorted(
                set(combined["report_fingerprint"].to_list())
            ),
            "decision_fingerprints": decision_fingerprints,
        }
    )
    rows = [
        {
            "schema_version": DYNAMIC_POOL_ESCALATION_REPORT_VERSION,
            "report_fingerprint": report_fingerprint,
            "decision_fingerprint": decision_fingerprint,
            **row,
        }
        for row, decision_fingerprint in zip(
            semantic_rows, decision_fingerprints, strict=True
        )
    ]
    table = pl.DataFrame(rows, schema=POOLING_ESCALATION_SCHEMA, strict=True).sort(
        ["hierarchy_level", "group_id"]
    )
    validate_pooling_escalations(table)
    return table


def validate_pooling_escalations(table: pl.DataFrame) -> None:
    """Reject automatic conclusions, authority drift and fingerprint changes."""

    if not isinstance(table, pl.DataFrame):
        raise TypeError("pooling escalation report must be a Polars DataFrame")
    if table.schema != POOLING_ESCALATION_SCHEMA:
        raise ValueError("pooling escalation schema does not match contract")
    if table.is_empty():
        raise ValueError("pooling escalation report must not be empty")
    if not table.equals(table.sort(["hierarchy_level", "group_id"])):
        raise ValueError("pooling escalation report is not canonically sorted")
    if table.select(["hierarchy_level", "group_id"]).n_unique() != table.height:
        raise ValueError("pooling escalation groups must be unique")
    if table.filter(
        (pl.col("schema_version") != DYNAMIC_POOL_ESCALATION_REPORT_VERSION)
        | pl.col("automatic_reference_disposition")
        | (pl.col("taxon_misidentification_conclusion") != "not_assessed")
        | pl.col("authorizes_occurrence_release")
        | ~pl.col("escalation_status").is_in(
            [
                "no_action",
                "evidence_collection_required",
                "remediation_review_required",
            ]
        )
    ).height:
        raise ValueError("pooling escalation crossed its authority contract")
    for row in table.iter_rows(named=True):
        rules = row["triggered_rules"]
        reasons = [str(rule["reason"]) for rule in rules]
        actions = sorted({str(rule["recommended_action"]) for rule in rules})
        flagged = bool(rules)
        if (
            row["trigger_reasons"] != reasons
            or row["recommended_actions"] != actions
            or bool(row["flagged_for_remediation"]) != flagged
            or bool(row["human_review_required"]) != flagged
        ):
            raise ValueError("pooling escalation decision semantics are invalid")
        if row["escalation_status"] == "no_action" and flagged:
            raise ValueError("no-action escalation contains triggers")
        if row["escalation_status"] != "no_action" and not flagged:
            raise ValueError("flagged escalation is missing triggers")
        base = {
            field: row[field]
            for field in POOLING_ESCALATION_SCHEMA
            if field
            not in {"schema_version", "report_fingerprint", "decision_fingerprint"}
        }
        if row["decision_fingerprint"] != canonical_semantic_fingerprint(base):
            raise ValueError("pooling escalation decision fingerprint mismatch")
    if table["report_fingerprint"].n_unique() != 1:
        raise ValueError("pooling escalation report has mixed fingerprints")


def _escalation_row(
    group: pl.DataFrame,
    *,
    policy: DynamicPoolEscalationPolicy,
) -> dict[str, object]:
    first = group.row(0, named=True)
    identity_fields = (
        "hierarchy_level",
        "geography_level",
        "group_id",
        "group_label",
        "family_key",
        "family_name",
        "genus_key",
        "genus_name",
        "species_key",
        "scientific_name",
        "geography_value",
        "no_geo",
        "group_status",
        "report_fingerprint",
    )
    for field in identity_fields:
        if group[field].n_unique() != 1:
            raise ValueError(f"quality group has mixed {field}")
    rows = {str(row["metric_name"]): row for row in group.iter_rows(named=True)}
    rules: list[dict[str, object]] = []
    if first["group_status"] != "complete":
        rules.append(
            _rule(
                reason="insufficient_representative_evidence",
                metric_name="group_sample",
                comparison_basis="group_status",
                observed=None,
                operator="!=",
                threshold=None,
                source_fingerprint=str(first["row_fingerprint"]),
                action="collect_additional_representative_flickr_reviews",
            )
        )
        status = "evidence_collection_required"
    else:
        _append_incomplete_metric_rules(rules, rows)
        precision = rows["selection_precision"]
        if precision["metric_status"] == "complete":
            reason = (
                "geography_precision_lower_bound_below_objective"
                if first["hierarchy_level"] == "geography"
                else "precision_lower_bound_below_objective"
            )
            action = (
                "review_geographic_group"
                if first["hierarchy_level"] == "geography"
                else "collect_additional_representative_flickr_reviews"
            )
            _append_if_below(
                rules,
                precision,
                reason=reason,
                comparison_basis="confidence_interval_lower",
                threshold=policy.minimum_precision_lower_bound,
                action=action,
            )
        for metric, reason, threshold, action in (
            (
                "family_routing_error_rate",
                "family_misrouting_above_objective",
                policy.maximum_family_routing_error_rate,
                "review_candidate_routing",
            ),
            (
                "global_local_disagreement_rate",
                "global_local_disagreement_above_objective",
                policy.maximum_global_local_disagreement_rate,
                "collect_disagreement_flickr_reviews",
            ),
            (
                "local_support_insufficiency_rate",
                "local_support_insufficiency_above_objective",
                policy.maximum_local_support_insufficiency_rate,
                "review_local_reference_coverage",
            ),
            (
                "reference_outlier_error_influence_rate",
                "reference_outlier_influence_above_objective",
                policy.maximum_reference_outlier_error_influence_rate,
                "review_reference_influence",
            ),
            (
                "weighted_brier_score",
                "weighted_brier_score_above_objective",
                policy.maximum_weighted_brier_score,
                "review_calibration",
            ),
            (
                "weighted_ece",
                "weighted_ece_above_objective",
                policy.maximum_weighted_ece,
                "review_calibration",
            ),
            (
                "ood_false_positive_incidence",
                "ood_false_positive_incidence_above_objective",
                policy.maximum_ood_false_positive_incidence,
                "review_ood_false_positives",
            ),
        ):
            row = rows[metric]
            if row["metric_status"] == "complete":
                _append_if_above(
                    rules,
                    row,
                    reason=reason,
                    comparison_basis="estimate",
                    threshold=threshold,
                    action=action,
                )
        status = (
            "evidence_collection_required"
            if rules
            and all(str(rule["reason"]).startswith("insufficient_") for rule in rules)
            else "remediation_review_required"
            if rules
            else "no_action"
        )
    reasons = [str(rule["reason"]) for rule in rules]
    actions = sorted({str(rule["recommended_action"]) for rule in rules})
    reference_actions = {
        "review_candidate_routing",
        "review_local_reference_coverage",
        "review_reference_influence",
    }
    flickr_actions = {
        "collect_additional_representative_flickr_reviews",
        "collect_disagreement_flickr_reviews",
        "review_geographic_group",
        "review_calibration",
        "review_ood_false_positives",
    }
    return {
        "policy_fingerprint": policy.fingerprint,
        "source_quality_report_fingerprint": first["report_fingerprint"],
        "source_quality_row_fingerprints": sorted(group["row_fingerprint"].to_list()),
        "hierarchy_level": first["hierarchy_level"],
        "geography_level": first["geography_level"],
        "group_id": first["group_id"],
        "group_label": first["group_label"],
        "family_key": first["family_key"],
        "family_name": first["family_name"],
        "genus_key": first["genus_key"],
        "genus_name": first["genus_name"],
        "species_key": first["species_key"],
        "scientific_name": first["scientific_name"],
        "geography_value": first["geography_value"],
        "no_geo": first["no_geo"],
        "quality_group_status": first["group_status"],
        "escalation_status": status,
        "flagged_for_remediation": bool(rules),
        "trigger_reasons": reasons,
        "triggered_rules": rules,
        "recommended_actions": actions,
        "reference_review_candidate": bool(reference_actions & set(actions)),
        "additional_flickr_audit_candidate": bool(flickr_actions & set(actions)),
        "human_review_required": bool(rules),
        "automatic_reference_disposition": False,
        "taxon_misidentification_conclusion": "not_assessed",
        "authorizes_occurrence_release": False,
    }


def _append_incomplete_metric_rules(
    rules: list[dict[str, object]],
    rows: dict[str, dict[str, object]],
) -> None:
    for metric_name in sorted(rows):
        row = rows[metric_name]
        if row["metric_status"] == "complete" or row["denominator_item_count"] == 0:
            continue
        rules.append(
            _rule(
                reason=f"insufficient_metric_evidence:{metric_name}",
                metric_name=metric_name,
                comparison_basis="metric_status",
                observed=None,
                operator="!=",
                threshold=None,
                source_fingerprint=str(row["row_fingerprint"]),
                action="collect_additional_representative_flickr_reviews",
            )
        )


def _append_if_below(
    rules: list[dict[str, object]],
    row: dict[str, object],
    *,
    reason: str,
    comparison_basis: str,
    threshold: float,
    action: str,
) -> None:
    observed = row[comparison_basis]
    if observed is not None and float(observed) < threshold:
        rules.append(
            _rule(
                reason=reason,
                metric_name=str(row["metric_name"]),
                comparison_basis=comparison_basis,
                observed=float(observed),
                operator="<",
                threshold=threshold,
                source_fingerprint=str(row["row_fingerprint"]),
                action=action,
            )
        )


def _append_if_above(
    rules: list[dict[str, object]],
    row: dict[str, object],
    *,
    reason: str,
    comparison_basis: str,
    threshold: float,
    action: str,
) -> None:
    observed = row[comparison_basis]
    if observed is not None and float(observed) > threshold:
        rules.append(
            _rule(
                reason=reason,
                metric_name=str(row["metric_name"]),
                comparison_basis=comparison_basis,
                observed=float(observed),
                operator=">",
                threshold=threshold,
                source_fingerprint=str(row["row_fingerprint"]),
                action=action,
            )
        )


def _rule(
    *,
    reason: str,
    metric_name: str,
    comparison_basis: str,
    observed: float | None,
    operator: str,
    threshold: float | None,
    source_fingerprint: str,
    action: str,
) -> dict[str, object]:
    return {
        "reason": reason,
        "metric_name": metric_name,
        "comparison_basis": comparison_basis,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "source_quality_row_fingerprint": source_fingerprint,
        "recommended_action": action,
    }


__all__ = [
    "DYNAMIC_POOL_ESCALATION_POLICY_VERSION",
    "DYNAMIC_POOL_ESCALATION_REPORT_VERSION",
    "POOLING_ESCALATION_SCHEMA",
    "TRIGGER_RULE_SCHEMA",
    "DynamicPoolEscalationPolicy",
    "define_pooling_escalations",
    "validate_pooling_escalations",
]
