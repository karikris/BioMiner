"""Versioned species-level escalation into targeted reference review."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint


ESCALATION_RULE_SCHEMA = pl.Struct(
    {
        "reason": pl.String,
        "observed": pl.Float64,
        "operator": pl.String,
        "threshold": pl.Float64,
    }
)
REFERENCE_ESCALATION_SCHEMA = {
    "target_species": pl.String,
    "competitor_species": pl.String,
    "region": pl.String,
    "route": pl.String,
    "metric_status": pl.String,
    "flagged_for_reference_review": pl.Boolean,
    "review_scope": pl.String,
    "flag_reasons": pl.List(pl.String),
    "triggered_rules": pl.List(ESCALATION_RULE_SCHEMA),
    "policy_version": pl.String,
    "policy_fingerprint": pl.String,
    "statistical_identity_conclusion": pl.String,
    "decision_fingerprint": pl.String,
}


@dataclass(frozen=True, slots=True)
class ReferenceEscalationPolicy:
    schema_version: str = "reference-escalation-policy-v1.0.0"
    policy_version: str = "adaptive-reference-escalation-v1"
    minimum_precision_lower_bound: float = 0.8
    maximum_false_positive_rate: float = 0.2
    minimum_target_recall: float = 0.75
    maximum_competitor_confusion_rate: float = 0.2
    maximum_prototype_dispersion: float = 0.35
    maximum_high_influence_outlier_rate: float = 0.2
    maximum_route_imbalance_ratio: float = 0.5
    minimum_reference_count: int = 5

    def __post_init__(self) -> None:
        for field in (
            "minimum_precision_lower_bound",
            "maximum_false_positive_rate",
            "minimum_target_recall",
            "maximum_competitor_confusion_rate",
            "maximum_prototype_dispersion",
            "maximum_high_influence_outlier_rate",
            "maximum_route_imbalance_ratio",
        ):
            value = getattr(self, field)
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{field} must be finite within [0, 1]")
        if self.minimum_reference_count < 1:
            raise ValueError("minimum_reference_count must be positive")

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                field: getattr(self, field)
                for field in self.__dataclass_fields__
            }
        )


def flag_species_for_reference_review(
    performance: pl.DataFrame,
    reference_evidence: pl.DataFrame,
    *,
    policy: ReferenceEscalationPolicy | None = None,
) -> pl.DataFrame:
    """Apply versioned thresholds and persist every escalation reason."""

    active = policy or ReferenceEscalationPolicy()
    keys = ["target_species", "competitor_species", "region", "route"]
    _require_columns(
        performance,
        {
            *keys,
            "metric_status",
            "reviewed_record_count",
            "precision_ci_lower",
            "recall",
            "false_positive_rate",
            "competitor_confusion_rate",
        },
        artifact="performance",
    )
    _require_columns(
        reference_evidence,
        {
            *keys,
            "prototype_dispersion_max",
            "high_influence_reference_count",
            "reference_outlier_count",
            "route_imbalance_ratio",
            "target_reference_count",
            "reference_identity_conclusion",
        },
        artifact="reference evidence",
    )
    if reference_evidence.filter(
        pl.col("reference_identity_conclusion") != "not_assessed"
    ).height:
        raise ValueError("reference evidence must not claim identity error")
    if performance.select(keys).n_unique() != performance.height:
        raise ValueError("performance escalation keys must be unique")
    if reference_evidence.select(keys).n_unique() != reference_evidence.height:
        raise ValueError("reference evidence escalation keys must be unique")
    joined = performance.join(
        reference_evidence,
        on=keys,
        how="left",
        validate="1:1",
        suffix="_reference",
    )
    if joined["target_reference_count"].null_count():
        raise ValueError("every performance group requires reference evidence")

    decisions: list[dict[str, object]] = []
    for row in joined.iter_rows(named=True):
        rules: list[dict[str, object]] = []
        if row["metric_status"] != "complete":
            rules.append(
                _rule(
                    "insufficient_human_audit_sample",
                    float(row["reviewed_record_count"]),
                    "metric_status !=",
                    None,
                )
            )
        else:
            _append_if_below(
                rules,
                "precision_lower_bound_below_objective",
                row["precision_ci_lower"],
                active.minimum_precision_lower_bound,
            )
            _append_if_above(
                rules,
                "false_positive_rate_above_objective",
                row["false_positive_rate"],
                active.maximum_false_positive_rate,
            )
            _append_if_below(
                rules,
                "target_recall_below_objective",
                row["recall"],
                active.minimum_target_recall,
            )
            _append_if_above(
                rules,
                "competitor_confusion_above_objective",
                row["competitor_confusion_rate"],
                active.maximum_competitor_confusion_rate,
            )
        _append_if_above(
            rules,
            "prototype_dispersion_above_objective",
            row["prototype_dispersion_max"],
            active.maximum_prototype_dispersion,
        )
        reference_count = int(row["target_reference_count"])
        outlier_count = max(
            int(row["high_influence_reference_count"]),
            int(row["reference_outlier_count"]),
        )
        outlier_rate = outlier_count / reference_count if reference_count else 1.0
        _append_if_above(
            rules,
            "high_influence_outlier_rate_above_objective",
            outlier_rate,
            active.maximum_high_influence_outlier_rate,
        )
        _append_if_above(
            rules,
            "route_imbalance_above_objective",
            row["route_imbalance_ratio"],
            active.maximum_route_imbalance_ratio,
        )
        if reference_count < active.minimum_reference_count:
            rules.append(
                _rule(
                    "reference_support_shortfall",
                    float(reference_count),
                    "<",
                    float(active.minimum_reference_count),
                )
            )
        reasons = tuple(str(rule["reason"]) for rule in rules)
        base: dict[str, object] = {
            **{key: row[key] for key in keys},
            "metric_status": row["metric_status"],
            "flagged_for_reference_review": bool(rules),
            "review_scope": "species_reference_group" if rules else "none",
            "flag_reasons": reasons,
            "triggered_rules": rules,
            "policy_version": active.policy_version,
            "policy_fingerprint": active.fingerprint,
            "statistical_identity_conclusion": "not_assessed",
            "decision_fingerprint": "",
        }
        payload = dict(base)
        payload.pop("decision_fingerprint")
        base["decision_fingerprint"] = canonical_semantic_fingerprint(payload)
        decisions.append(base)
    result = pl.DataFrame(
        decisions,
        schema=REFERENCE_ESCALATION_SCHEMA,
        orient="row",
        strict=True,
    ).sort(keys)
    validate_reference_escalations(result)
    return result


def validate_reference_escalations(frame: pl.DataFrame) -> None:
    if frame.schema != REFERENCE_ESCALATION_SCHEMA:
        raise ValueError("reference escalation schema mismatch")
    keys = ["target_species", "competitor_species", "region", "route"]
    if frame.select(keys).unique().height != frame.height:
        raise ValueError("reference escalation groups must be unique")
    for row in frame.iter_rows(named=True):
        rules = row["triggered_rules"]
        reasons = [str(rule["reason"]) for rule in rules]
        if row["statistical_identity_conclusion"] != "not_assessed":
            raise ValueError("reference escalation must not claim identity error")
        if (
            row["flag_reasons"] != reasons
            or bool(row["flagged_for_reference_review"]) != bool(rules)
            or row["review_scope"]
            != ("species_reference_group" if rules else "none")
        ):
            raise ValueError("reference escalation decision semantics are invalid")
        payload = dict(row)
        fingerprint = payload.pop("decision_fingerprint")
        if fingerprint != canonical_semantic_fingerprint(payload):
            raise ValueError("reference escalation decision fingerprint mismatch")


def _append_if_below(
    rules: list[dict[str, object]],
    reason: str,
    observed: object,
    threshold: float,
) -> None:
    if observed is not None and float(observed) < threshold:
        rules.append(_rule(reason, float(observed), "<", threshold))


def _append_if_above(
    rules: list[dict[str, object]],
    reason: str,
    observed: object,
    threshold: float,
) -> None:
    if observed is not None and float(observed) > threshold:
        rules.append(_rule(reason, float(observed), ">", threshold))


def _rule(
    reason: str,
    observed: float,
    operator: str,
    threshold: float | None,
) -> dict[str, object]:
    return {
        "reason": reason,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
    }


def _require_columns(
    frame: pl.DataFrame,
    required: set[str],
    *,
    artifact: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{artifact} missing columns: {missing}")


__all__ = [
    "ESCALATION_RULE_SCHEMA",
    "REFERENCE_ESCALATION_SCHEMA",
    "ReferenceEscalationPolicy",
    "flag_species_for_reference_review",
    "validate_reference_escalations",
]
