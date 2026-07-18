"""Fail-closed selection report for candidate-scheduling strategies."""

from __future__ import annotations

from collections.abc import Mapping
import json
from math import isfinite
from pathlib import Path

import polars as pl

from biominer.candidates.strategy_ablation import (
    CANDIDATE_STRATEGIES,
    PARALLEL_UNION_STRATEGY,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.candidate_strategies import (
    summarize_family_pruning_counterfactual,
    validate_candidate_strategy_metrics,
    validate_family_pruning_counterfactual,
)


CANDIDATE_STRATEGY_ABLATION_REPORT_SCHEMA_VERSION = (
    "candidate-strategy-ablation-report-v1.0.0"
)
CANDIDATE_STRATEGY_ABLATION_REPORT_FILE = "candidate_strategy_ablation_report.json"
CANDIDATE_STRATEGY_ABLATION_SUMMARY_FILE = "candidate_strategy_ablation_report.md"

_GATE_FIELDS = frozenset(
    {
        "selection_k",
        "minimum_evaluated_labels",
        "minimum_target_recall",
        "minimum_species_recall",
        "minimum_family_recall",
        "minimum_no_geo_species_recall",
        "minimum_wrong_family_species_recall",
        "maximum_recall_shortfall",
        "maximum_mean_dot_products",
        "maximum_mean_reference_members",
        "maximum_mean_elapsed_time_ms",
        "maximum_peak_memory_bytes",
        "minimum_cache_reuse_fraction",
        "minimum_family_pruning_eligible_labels",
        "require_non_fixture_evidence",
    }
)
_FIXTURE_MARKERS = ("fixture", "synthetic", "mock", "test-only")


def build_candidate_strategy_ablation_report(
    metrics: pl.DataFrame,
    family_pruning_counterfactual: pl.DataFrame,
    *,
    validation_gate: Mapping[str, object],
    intended_candidate: str = PARALLEL_UNION_STRATEGY,
) -> dict[str, object]:
    """Select the intended strategy only when every configured check passes."""

    validate_candidate_strategy_metrics(metrics)
    validate_family_pruning_counterfactual(family_pruning_counterfactual)
    intended = str(intended_candidate or "").strip().casefold()
    if intended not in CANDIDATE_STRATEGIES:
        raise ValueError(f"unsupported intended candidate strategy {intended!r}")
    gate = _normalize_gate(validation_gate)
    evaluation_run_id = _single_text(metrics, "evaluation_run_id")
    if _single_text(
        family_pruning_counterfactual, "evaluation_run_id"
    ) != evaluation_run_id:
        raise ValueError("metrics and family-pruning evidence use different runs")
    selected_k = int(gate["selection_k"])
    selected_rows = metrics.filter(pl.col("k") == selected_k)
    if selected_rows.is_empty():
        raise ValueError(f"strategy metrics do not contain configured k={selected_k}")
    strategies = sorted(set(str(value) for value in selected_rows["strategy_name"]))
    if intended not in strategies:
        raise ValueError("metrics do not contain the intended candidate strategy")
    _validate_comparable_strategy_rows(selected_rows)
    summaries = [
        _strategy_summary(
            selected_rows.filter(pl.col("strategy_name") == strategy),
            strategy=strategy,
        )
        for strategy in strategies
    ]
    summary_by_strategy = {item["strategy_name"]: item for item in summaries}
    intended_summary = summary_by_strategy[intended]
    pruning_summary = summarize_family_pruning_counterfactual(
        family_pruning_counterfactual
    )
    checks = _gate_checks(
        gate=gate,
        summaries=summary_by_strategy,
        intended=intended,
        intended_summary=intended_summary,
        selected_rows=selected_rows,
        pruning_summary=pruning_summary,
    )
    gate_passed = all(bool(check["passed"]) for check in checks)
    report: dict[str, object] = {
        "schema_version": CANDIDATE_STRATEGY_ABLATION_REPORT_SCHEMA_VERSION,
        "evaluation_run_id": evaluation_run_id,
        "selection_k": selected_k,
        "intended_candidate": intended,
        "evaluated_strategies": strategies,
        "validation_gate": gate,
        "validation_checks": checks,
        "validation_gate_passed": gate_passed,
        "selected_strategy": intended if gate_passed else None,
        "selection_status": (
            "selected_for_next_phase" if gate_passed else "validation_gate_failed"
        ),
        "strategy_summaries": summaries,
        "family_pruning_counterfactual": pruning_summary,
        "production_default_eligible": gate_passed,
        "production_default_changed": False,
        "superiority_claimed": False,
        "selection_basis": (
            "Configured validation thresholds and observed non-inferiority on the "
            "identified evaluation run; not universal or causal superiority."
        ),
        "blocked_claims": [
            "universal strategy superiority",
            "family or geography evidence proves taxonomic identity or absence",
            "selection alone changes the production default",
            "selection alone establishes calibration, statistical support, or release readiness",
        ],
    }
    report["report_fingerprint"] = canonical_semantic_fingerprint(report)
    validate_candidate_strategy_ablation_report(report)
    return report


def validate_candidate_strategy_ablation_report(
    report: Mapping[str, object],
) -> None:
    if report.get("schema_version") != CANDIDATE_STRATEGY_ABLATION_REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported candidate strategy ablation report schema")
    strategies = report.get("evaluated_strategies")
    if not isinstance(strategies, list) or strategies != sorted(set(strategies)):
        raise ValueError("evaluated strategies are not canonical")
    intended = report.get("intended_candidate")
    if intended not in CANDIDATE_STRATEGIES or intended not in strategies:
        raise ValueError("candidate strategy ablation intended candidate is invalid")
    checks = report.get("validation_checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("candidate strategy ablation checks are missing")
    check_names = [check.get("name") for check in checks if isinstance(check, Mapping)]
    if len(check_names) != len(checks) or len(check_names) != len(set(check_names)):
        raise ValueError("candidate strategy ablation checks are invalid")
    gate_passed = all(bool(check.get("passed")) for check in checks)
    if bool(report.get("validation_gate_passed")) != gate_passed:
        raise ValueError("candidate strategy validation-gate status is inconsistent")
    expected_selection = intended if gate_passed else None
    if report.get("selected_strategy") != expected_selection:
        raise ValueError("candidate strategy selection is inconsistent with the gate")
    if bool(report.get("production_default_eligible")) != gate_passed:
        raise ValueError("production-default eligibility is inconsistent with the gate")
    if report.get("production_default_changed") is not False:
        raise ValueError("ablation report must not mutate the production default")
    if report.get("superiority_claimed") is not False:
        raise ValueError("ablation report must not claim unsupported superiority")
    fingerprint = report.get("report_fingerprint")
    payload = dict(report)
    payload.pop("report_fingerprint", None)
    if fingerprint != canonical_semantic_fingerprint(payload):
        raise ValueError("candidate strategy ablation report fingerprint is invalid")


def candidate_strategy_ablation_markdown(report: Mapping[str, object]) -> str:
    validate_candidate_strategy_ablation_report(report)
    summaries = report["strategy_summaries"]
    lines = [
        "# Candidate Strategy Ablation",
        "",
        f"- Evaluation run: `{report['evaluation_run_id']}`",
        f"- Selection cutoff: k={report['selection_k']}",
        f"- Intended candidate: `{report['intended_candidate']}`",
        f"- Validation gate passed: `{str(report['validation_gate_passed']).lower()}`",
        f"- Selected strategy: `{report['selected_strategy'] or 'none'}`",
        "- Production default changed: `false`",
        "- Universal superiority claimed: `false`",
        "",
        "## Strategy metrics",
        "",
        "| Strategy | Labels | Target recall | Species recall | Family recall | No-geo species recall | Wrong-family species recall | Mean dot products | Mean elapsed ms | Peak memory bytes | Cache reuse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        display = {
            **summary,
            "no_geo_display": _format_optional(
                summary["no_geo"]["species_recall"]
            ),
            "wrong_family_display": _format_optional(
                summary["wrong_family"]["species_recall"]
            ),
        }
        lines.append(
            "| {strategy_name} | {evaluated_label_count} | {target_recall:.4f} | "
            "{species_recall:.4f} | {family_recall:.4f} | {no_geo_display} | "
            "{wrong_family_display} | {mean_dot_products:.2f} | {mean_elapsed_time_ms:.4f} | "
            "{peak_memory_bytes} | {cache_reuse_fraction:.4f} |".format(
                **display,
            )
        )
    lines.extend(
        [
            "",
            "## Validation checks",
            "",
            *[
                f"- {'PASS' if check['passed'] else 'FAIL'} — {check['name']}: "
                f"observed `{json.dumps(check['observed'], sort_keys=True)}`; "
                f"requirement `{json.dumps(check['requirement'], sort_keys=True)}`"
                for check in report["validation_checks"]
            ],
            "",
            "## Family-pruning counterfactual",
            "",
            f"- Eligible reviewed species: {report['family_pruning_counterfactual']['eligible_correct_species_count']}",
            f"- Correct species lost: {report['family_pruning_counterfactual']['correct_species_lost_count']}",
            f"- Loss rate: {_format_optional(report['family_pruning_counterfactual']['correct_species_lost_rate'])}",
            "- Production candidate membership changed: `false`",
            "",
            "## Interpretation boundary",
            "",
            f"{report['selection_basis']}",
            "",
            *[f"- Blocked: {claim}" for claim in report["blocked_claims"]],
            "",
        ]
    )
    return "\n".join(lines)


def write_candidate_strategy_ablation_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> dict[str, Path]:
    validate_candidate_strategy_ablation_report(report)
    destination = Path(output_path)
    json_path = (
        destination
        if destination.suffix.casefold() == ".json"
        else destination / CANDIDATE_STRATEGY_ABLATION_REPORT_FILE
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path = json_path.with_name(CANDIDATE_STRATEGY_ABLATION_SUMMARY_FILE)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(
        candidate_strategy_ablation_markdown(report), encoding="utf-8"
    )
    return {"json": json_path, "markdown": markdown_path}


def _strategy_summary(frame: pl.DataFrame, *, strategy: str) -> dict[str, object]:
    no_geo = frame.filter(pl.col("no_geo"))
    wrong_family = frame.filter(pl.col("wrong_family"))
    reference_members = int(frame["reference_member_count"].sum())
    cache_reused = int(frame["cache_reused_reference_members"].sum())
    return {
        "strategy_name": strategy,
        "evaluated_label_count": frame["source_candidate_set_id"].n_unique(),
        "target_recall": _mean(frame, "target_candidate_recall_at_k"),
        "species_recall": _mean(frame, "species_candidate_recall_at_k"),
        "family_recall": _mean(frame, "family_candidate_recall_at_k"),
        "mean_candidate_set_size": _mean(frame, "candidate_set_size"),
        "mean_evaluated_candidate_count": _mean(
            frame, "evaluated_candidate_count"
        ),
        "mean_dot_products": _mean(frame, "dot_product_count"),
        "mean_reference_members": _mean(frame, "reference_member_count"),
        "mean_elapsed_time_ms": _mean(frame, "elapsed_time_ms"),
        "peak_memory_bytes": int(frame["peak_memory_bytes"].max()),
        "cache_reuse_fraction": (
            cache_reused / reference_members if reference_members else 0.0
        ),
        "no_geo": _slice_summary(no_geo),
        "wrong_family": _slice_summary(wrong_family),
        "label_sources": sorted(set(str(value) for value in frame["label_source"])),
        "measurement_sources": sorted(
            set(str(value) for value in frame["measurement_source"])
        ),
    }


def _slice_summary(frame: pl.DataFrame) -> dict[str, object]:
    return {
        "evaluated_label_count": frame["source_candidate_set_id"].n_unique(),
        "species_recall": (
            _mean(frame, "species_candidate_recall_at_k")
            if not frame.is_empty()
            else None
        ),
        "family_recall": (
            _mean(frame, "family_candidate_recall_at_k")
            if not frame.is_empty()
            else None
        ),
    }


def _gate_checks(
    *,
    gate: Mapping[str, object],
    summaries: Mapping[str, Mapping[str, object]],
    intended: str,
    intended_summary: Mapping[str, object],
    selected_rows: pl.DataFrame,
    pruning_summary: Mapping[str, object],
) -> list[dict[str, object]]:
    required = sorted(CANDIDATE_STRATEGIES)
    best_recalls = {
        metric: max(float(summary[metric]) for summary in summaries.values())
        for metric in ("target_recall", "species_recall", "family_recall")
    }
    observed_shortfalls = {
        metric: best - float(intended_summary[metric])
        for metric, best in best_recalls.items()
    }
    source_values = [
        *selected_rows["label_source"].to_list(),
        *selected_rows["measurement_source"].to_list(),
    ]
    non_fixture = not any(
        marker in str(value).casefold()
        for value in source_values
        for marker in _FIXTURE_MARKERS
    )
    checks = [
        _check(
            "required_strategies_present",
            sorted(summaries) == required,
            sorted(summaries),
            required,
        ),
        _check(
            "minimum_evaluated_labels",
            int(intended_summary["evaluated_label_count"])
            >= int(gate["minimum_evaluated_labels"]),
            intended_summary["evaluated_label_count"],
            {"minimum": gate["minimum_evaluated_labels"]},
        ),
    ]
    for metric, threshold_name in (
        ("target_recall", "minimum_target_recall"),
        ("species_recall", "minimum_species_recall"),
        ("family_recall", "minimum_family_recall"),
    ):
        checks.append(
            _check(
                threshold_name,
                float(intended_summary[metric]) >= float(gate[threshold_name]),
                intended_summary[metric],
                {"minimum": gate[threshold_name]},
            )
        )
    checks.extend(
        [
            _minimum_slice_check(
                intended_summary,
                slice_name="no_geo",
                gate=gate,
                gate_name="minimum_no_geo_species_recall",
            ),
            _minimum_slice_check(
                intended_summary,
                slice_name="wrong_family",
                gate=gate,
                gate_name="minimum_wrong_family_species_recall",
            ),
            _check(
                "recall_noninferiority",
                all(
                    shortfall <= float(gate["maximum_recall_shortfall"])
                    for shortfall in observed_shortfalls.values()
                ),
                observed_shortfalls,
                {"maximum_each": gate["maximum_recall_shortfall"]},
            ),
            _maximum_check(
                intended_summary,
                metric="mean_dot_products",
                gate=gate,
                gate_name="maximum_mean_dot_products",
            ),
            _maximum_check(
                intended_summary,
                metric="mean_reference_members",
                gate=gate,
                gate_name="maximum_mean_reference_members",
            ),
            _maximum_check(
                intended_summary,
                metric="mean_elapsed_time_ms",
                gate=gate,
                gate_name="maximum_mean_elapsed_time_ms",
            ),
            _maximum_check(
                intended_summary,
                metric="peak_memory_bytes",
                gate=gate,
                gate_name="maximum_peak_memory_bytes",
            ),
            _check(
                "minimum_cache_reuse_fraction",
                float(intended_summary["cache_reuse_fraction"])
                >= float(gate["minimum_cache_reuse_fraction"]),
                intended_summary["cache_reuse_fraction"],
                {"minimum": gate["minimum_cache_reuse_fraction"]},
            ),
            _check(
                "minimum_family_pruning_eligible_labels",
                int(pruning_summary["eligible_correct_species_count"])
                >= int(gate["minimum_family_pruning_eligible_labels"]),
                pruning_summary["eligible_correct_species_count"],
                {"minimum": gate["minimum_family_pruning_eligible_labels"]},
            ),
            _check(
                "non_fixture_evidence",
                non_fixture or not bool(gate["require_non_fixture_evidence"]),
                {"non_fixture": non_fixture},
                {"required": gate["require_non_fixture_evidence"]},
            ),
        ]
    )
    return checks


def _minimum_slice_check(
    summary: Mapping[str, object],
    *,
    slice_name: str,
    gate: Mapping[str, object],
    gate_name: str,
) -> dict[str, object]:
    slice_summary = summary[slice_name]
    recall = slice_summary["species_recall"]
    passed = (
        int(slice_summary["evaluated_label_count"]) > 0
        and recall is not None
        and float(recall) >= float(gate[gate_name])
    )
    return _check(
        gate_name,
        passed,
        slice_summary,
        {"minimum_recall": gate[gate_name], "minimum_labels": 1},
    )


def _maximum_check(
    summary: Mapping[str, object],
    *,
    metric: str,
    gate: Mapping[str, object],
    gate_name: str,
) -> dict[str, object]:
    return _check(
        gate_name,
        float(summary[metric]) <= float(gate[gate_name]),
        summary[metric],
        {"maximum": gate[gate_name]},
    )


def _check(
    name: str,
    passed: bool,
    observed: object,
    requirement: object,
) -> dict[str, object]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "requirement": requirement,
    }


def _normalize_gate(gate: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(gate, Mapping) or set(gate) != _GATE_FIELDS:
        raise ValueError("candidate strategy validation-gate fields do not match")
    normalized: dict[str, object] = {
        "selection_k": _positive_int(gate["selection_k"], field="selection_k"),
        "minimum_evaluated_labels": _positive_int(
            gate["minimum_evaluated_labels"], field="minimum_evaluated_labels"
        ),
        "minimum_family_pruning_eligible_labels": _positive_int(
            gate["minimum_family_pruning_eligible_labels"],
            field="minimum_family_pruning_eligible_labels",
        ),
        "maximum_mean_dot_products": _nonnegative_float(
            gate["maximum_mean_dot_products"], field="maximum_mean_dot_products"
        ),
        "maximum_mean_reference_members": _nonnegative_float(
            gate["maximum_mean_reference_members"],
            field="maximum_mean_reference_members",
        ),
        "maximum_mean_elapsed_time_ms": _nonnegative_float(
            gate["maximum_mean_elapsed_time_ms"],
            field="maximum_mean_elapsed_time_ms",
        ),
        "maximum_peak_memory_bytes": _nonnegative_int(
            gate["maximum_peak_memory_bytes"], field="maximum_peak_memory_bytes"
        ),
    }
    for field in (
        "minimum_target_recall",
        "minimum_species_recall",
        "minimum_family_recall",
        "minimum_no_geo_species_recall",
        "minimum_wrong_family_species_recall",
        "maximum_recall_shortfall",
        "minimum_cache_reuse_fraction",
    ):
        normalized[field] = _unit_interval(gate[field], field=field)
    require_non_fixture = gate["require_non_fixture_evidence"]
    if not isinstance(require_non_fixture, bool):
        raise ValueError("require_non_fixture_evidence must be Boolean")
    normalized["require_non_fixture_evidence"] = require_non_fixture
    return normalized


def _validate_comparable_strategy_rows(frame: pl.DataFrame) -> None:
    expected_sets: set[str] | None = None
    for (_strategy,), group in frame.group_by("strategy_name"):
        candidate_sets = set(str(value) for value in group["source_candidate_set_id"])
        if group.height != len(candidate_sets):
            raise ValueError("strategy metrics contain duplicate labels at selection k")
        if expected_sets is None:
            expected_sets = candidate_sets
        elif candidate_sets != expected_sets:
            raise ValueError("strategy metrics do not cover comparable candidate sets")


def _single_text(frame: pl.DataFrame, field: str) -> str:
    values = set(str(value) for value in frame[field])
    if len(values) != 1:
        raise ValueError(f"{field} must contain exactly one value")
    return next(iter(values))


def _mean(frame: pl.DataFrame, field: str) -> float:
    value = frame[field].mean()
    if value is None:
        raise ValueError(f"cannot calculate mean {field}")
    return float(value)


def _unit_interval(value: object, *, field: str) -> float:
    number = _nonnegative_float(value, field=field)
    if number > 1:
        raise ValueError(f"{field} must be within [0, 1]")
    return number


def _positive_int(value: object, *, field: str) -> int:
    number = _nonnegative_int(value, field=field)
    if number <= 0:
        raise ValueError(f"{field} must be positive")
    return number


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative integer") from exc
    if number < 0 or number != value:
        raise ValueError(f"{field} must be a non-negative integer")
    return number


def _nonnegative_float(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite non-negative number") from exc
    if not isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return number


def _format_optional(value: object) -> str:
    return "undefined" if value is None else f"{float(value):.4f}"


__all__ = [
    "CANDIDATE_STRATEGY_ABLATION_REPORT_FILE",
    "CANDIDATE_STRATEGY_ABLATION_REPORT_SCHEMA_VERSION",
    "CANDIDATE_STRATEGY_ABLATION_SUMMARY_FILE",
    "build_candidate_strategy_ablation_report",
    "candidate_strategy_ablation_markdown",
    "validate_candidate_strategy_ablation_report",
    "write_candidate_strategy_ablation_report",
]
