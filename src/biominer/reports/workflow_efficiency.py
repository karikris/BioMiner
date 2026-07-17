"""Evidence-qualified efficiency reporting for the adaptive workflow."""

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
from biominer.storage.parquet import write_parquet


ADAPTIVE_WORKFLOW_EFFICIENCY_METRICS_FILE = (
    "adaptive_workflow_efficiency_metrics.parquet"
)
ADAPTIVE_WORKFLOW_EFFICIENCY_REPORT_FILE = (
    "adaptive_workflow_efficiency_report.json"
)
ADAPTIVE_WORKFLOW_EFFICIENCY_SUMMARY_FILE = (
    "adaptive_workflow_efficiency_report.md"
)
ADAPTIVE_WORKFLOW_EFFICIENCY_METRIC_SCHEMA_VERSION = (
    "adaptive-workflow-efficiency-metric-v1.0.0"
)
ADAPTIVE_WORKFLOW_EFFICIENCY_REPORT_SCHEMA_VERSION = (
    "adaptive-workflow-efficiency-report-v1.0.0"
)

EFFICIENCY_METRICS = (
    "time_to_first_score",
    "manual_reviews_avoided",
    "manual_reviews_triggered_later",
    "embeddings_reused",
    "yoloe_runs_reused",
    "prototypes_rebuilt",
    "records_selectively_rescored",
    "full_rerun_work_avoided",
    "peak_memory",
)
EFFICIENCY_EVIDENCE_STATUSES = frozenset(
    {"measured", "derived", "estimated", "fixture", "unavailable", "not_instrumented"}
)
_AVAILABLE_STATUSES = frozenset({"measured", "derived", "estimated", "fixture"})
_EXPECTED_UNITS = {
    "time_to_first_score": "seconds",
    "manual_reviews_avoided": "reviews",
    "manual_reviews_triggered_later": "reviews",
    "embeddings_reused": "embeddings",
    "yoloe_runs_reused": "runs",
    "prototypes_rebuilt": "prototypes",
    "records_selectively_rescored": "records",
    "peak_memory": "bytes",
}
_COUNTERFACTUAL_METRICS = frozenset(
    {"manual_reviews_avoided", "full_rerun_work_avoided"}
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")

ADAPTIVE_WORKFLOW_EFFICIENCY_METRIC_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "metric_name": pl.String,
    "unit_name": pl.String,
    "metric_value": pl.Float64,
    "evidence_status": pl.String,
    "source_artifact_id": pl.String,
    "source_artifact_fingerprint": pl.String,
    "derivation": pl.String,
    "unavailable_reason": pl.String,
    "metric_fingerprint": pl.String,
}


@dataclass(frozen=True, slots=True)
class AdaptiveWorkflowEfficiencyReport:
    metrics: pl.DataFrame
    report: dict[str, object]
    markdown: str


def adaptive_workflow_efficiency_metrics_frame(
    rows: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    normalized: list[dict[str, object]] = []
    for source in rows:
        row = dict(source)
        name = _required_text(row.get("metric_name"), field="metric_name")
        unit = _required_text(row.get("unit_name"), field="unit_name")
        status = _required_text(
            row.get("evidence_status"), field="evidence_status"
        )
        if name not in EFFICIENCY_METRICS:
            raise ValueError(f"unsupported adaptive efficiency metric: {name}")
        if status not in EFFICIENCY_EVIDENCE_STATUSES:
            raise ValueError(f"unsupported efficiency evidence status: {status}")
        expected_unit = _EXPECTED_UNITS.get(name)
        if expected_unit is not None and unit != expected_unit:
            raise ValueError(f"{name} must use {expected_unit}")
        value = row.get("metric_value")
        available = status in _AVAILABLE_STATUSES
        if available:
            numeric = _nonnegative_number(value, field="metric_value")
            reason = None
            derivation = _required_text(row.get("derivation"), field="derivation")
        else:
            if value is not None:
                raise ValueError("unavailable efficiency metrics cannot have a value")
            numeric = None
            reason = _required_text(
                row.get("unavailable_reason"), field="unavailable_reason"
            )
            derivation = None
        if name in _COUNTERFACTUAL_METRICS and status == "measured":
            raise ValueError(f"{name} is counterfactual and cannot be directly measured")
        item: dict[str, object] = {
            "schema_version": ADAPTIVE_WORKFLOW_EFFICIENCY_METRIC_SCHEMA_VERSION,
            "metric_name": name,
            "unit_name": unit,
            "metric_value": numeric,
            "evidence_status": status,
            "source_artifact_id": _required_text(
                row.get("source_artifact_id"), field="source_artifact_id"
            ),
            "source_artifact_fingerprint": _sha256(
                row.get("source_artifact_fingerprint"),
                field="source_artifact_fingerprint",
            ),
            "derivation": derivation,
            "unavailable_reason": reason,
            "metric_fingerprint": "",
        }
        item["metric_fingerprint"] = _fingerprint_without(
            item, "metric_fingerprint"
        )
        normalized.append(item)
    frame = pl.DataFrame(
        normalized,
        schema=ADAPTIVE_WORKFLOW_EFFICIENCY_METRIC_SCHEMA,
        orient="row",
        strict=True,
    ).sort("metric_name", "unit_name")
    validate_adaptive_workflow_efficiency_metrics(frame)
    return frame


def validate_adaptive_workflow_efficiency_metrics(frame: pl.DataFrame) -> None:
    if frame.schema != ADAPTIVE_WORKFLOW_EFFICIENCY_METRIC_SCHEMA:
        raise ValueError("adaptive workflow efficiency metric schema mismatch")
    if not frame.equals(frame.sort("metric_name", "unit_name")):
        raise ValueError("adaptive workflow efficiency metrics are not sorted")
    if frame.select("metric_name", "unit_name").n_unique() != frame.height:
        raise ValueError("adaptive workflow efficiency metrics repeat a unit")
    names = set(frame["metric_name"])
    if names != set(EFFICIENCY_METRICS):
        raise ValueError("adaptive workflow efficiency report must cover every metric")
    for name in set(EFFICIENCY_METRICS) - {"full_rerun_work_avoided"}:
        if frame.filter(pl.col("metric_name") == name).height != 1:
            raise ValueError(f"adaptive workflow efficiency metric {name} must be unique")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != ADAPTIVE_WORKFLOW_EFFICIENCY_METRIC_SCHEMA_VERSION:
            raise ValueError("unsupported adaptive efficiency metric version")
        status = row["evidence_status"]
        if status not in EFFICIENCY_EVIDENCE_STATUSES:
            raise ValueError("adaptive efficiency evidence status is invalid")
        expected_unit = _EXPECTED_UNITS.get(str(row["metric_name"]))
        if expected_unit is not None and row["unit_name"] != expected_unit:
            raise ValueError("adaptive efficiency metric unit mismatch")
        if status in _AVAILABLE_STATUSES:
            _nonnegative_number(row["metric_value"], field="metric_value")
            _required_text(row["derivation"], field="derivation")
            if row["unavailable_reason"] is not None:
                raise ValueError("available efficiency metric has unavailable reason")
        else:
            if row["metric_value"] is not None or row["derivation"] is not None:
                raise ValueError("unavailable efficiency metric claims a value")
            _required_text(row["unavailable_reason"], field="unavailable_reason")
        if row["metric_name"] in _COUNTERFACTUAL_METRICS and status == "measured":
            raise ValueError("counterfactual efficiency metric cannot be measured")
        _required_text(row["source_artifact_id"], field="source_artifact_id")
        _sha256(
            row["source_artifact_fingerprint"],
            field="source_artifact_fingerprint",
        )
        if row["metric_fingerprint"] != _fingerprint_without(
            row, "metric_fingerprint"
        ):
            raise ValueError("adaptive efficiency metric fingerprint mismatch")


def build_adaptive_workflow_efficiency_report(
    metrics: pl.DataFrame,
    *,
    generated_at: datetime | None = None,
) -> AdaptiveWorkflowEfficiencyReport:
    validate_adaptive_workflow_efficiency_metrics(metrics)
    timestamp = _utc_datetime(generated_at or datetime.now(UTC))
    metric_rows = [_report_metric(row) for row in metrics.iter_rows(named=True)]
    report: dict[str, object] = {
        "schema_version": ADAPTIVE_WORKFLOW_EFFICIENCY_REPORT_SCHEMA_VERSION,
        "generated_at": timestamp.isoformat(),
        "status": "complete",
        "metrics": metric_rows,
        "evidence_summary": {
            "available_metric_rows": sum(
                row["evidence_status"] in _AVAILABLE_STATUSES
                for row in metric_rows
            ),
            "unavailable_metric_rows": sum(
                row["evidence_status"] not in _AVAILABLE_STATUSES
                for row in metric_rows
            ),
        },
        "provenance": {
            "metrics_fingerprint": _frame_fingerprint(metrics),
            "aggregation_policy": "never_sum_different_unit_names",
        },
        "limitations": [
            "Unavailable and not-instrumented values are not zero.",
            "Avoided reviews and avoided full-rerun work are counterfactual quantities and require an explicit derivation or estimate.",
            "Efficiency rows with different units are never summed.",
            "Fixture values validate reporting behavior and are not production benchmark claims.",
            "Peak memory is meaningful only for the process boundary named by its source artifact.",
        ],
        "report_fingerprint": "",
    }
    report["report_fingerprint"] = _fingerprint_without(
        report, "report_fingerprint"
    )
    result = AdaptiveWorkflowEfficiencyReport(
        metrics=metrics,
        report=report,
        markdown=_markdown(report),
    )
    validate_adaptive_workflow_efficiency_report(result)
    return result


def validate_adaptive_workflow_efficiency_report(
    result: AdaptiveWorkflowEfficiencyReport,
) -> None:
    validate_adaptive_workflow_efficiency_metrics(result.metrics)
    report = result.report
    if report.get("schema_version") != ADAPTIVE_WORKFLOW_EFFICIENCY_REPORT_SCHEMA_VERSION:
        raise ValueError("adaptive workflow efficiency report schema mismatch")
    expected_metrics = [
        _report_metric(row) for row in result.metrics.iter_rows(named=True)
    ]
    if report.get("metrics") != expected_metrics:
        raise ValueError("adaptive workflow efficiency report metrics mismatch")
    expected_summary = {
        "available_metric_rows": sum(
            row["evidence_status"] in _AVAILABLE_STATUSES
            for row in expected_metrics
        ),
        "unavailable_metric_rows": sum(
            row["evidence_status"] not in _AVAILABLE_STATUSES
            for row in expected_metrics
        ),
    }
    if report.get("evidence_summary") != expected_summary:
        raise ValueError("adaptive workflow efficiency evidence summary mismatch")
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get(
        "metrics_fingerprint"
    ) != _frame_fingerprint(result.metrics):
        raise ValueError("adaptive workflow efficiency provenance mismatch")
    if provenance.get("aggregation_policy") != "never_sum_different_unit_names":
        raise ValueError("adaptive workflow efficiency aggregation policy mismatch")
    if report.get("report_fingerprint") != _fingerprint_without(
        report, "report_fingerprint"
    ):
        raise ValueError("adaptive workflow efficiency report fingerprint mismatch")
    if not result.markdown.startswith("# Adaptive workflow efficiency"):
        raise ValueError("adaptive workflow efficiency Markdown mismatch")


def write_adaptive_workflow_efficiency_report(
    result: AdaptiveWorkflowEfficiencyReport,
    output_dir: str | Path,
) -> dict[str, Path]:
    validate_adaptive_workflow_efficiency_report(result)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": root / ADAPTIVE_WORKFLOW_EFFICIENCY_METRICS_FILE,
        "json": root / ADAPTIVE_WORKFLOW_EFFICIENCY_REPORT_FILE,
        "markdown": root / ADAPTIVE_WORKFLOW_EFFICIENCY_SUMMARY_FILE,
    }
    write_parquet(result.metrics, paths["metrics"])
    paths["json"].write_text(
        json.dumps(result.report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["markdown"].write_text(result.markdown, encoding="utf-8")
    return paths


def _report_metric(row: Mapping[str, object]) -> dict[str, object]:
    return {
        field: row[field]
        for field in (
            "metric_name",
            "unit_name",
            "metric_value",
            "evidence_status",
            "source_artifact_id",
            "source_artifact_fingerprint",
            "derivation",
            "unavailable_reason",
            "metric_fingerprint",
        )
    }


def _markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Adaptive workflow efficiency",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "| Metric | Value | Unit | Evidence | Derivation or reason |",
        "|---|---:|---|---|---|",
    ]
    for row in report["metrics"]:  # type: ignore[union-attr]
        value = "unavailable" if row["metric_value"] is None else row["metric_value"]
        basis = row["derivation"] or row["unavailable_reason"]
        lines.append(
            f"| {row['metric_name']} | {value} | {row['unit_name']} | "
            f"{row['evidence_status']} | {basis} |"
        )
    lines.extend(["", "## Interpretation boundaries", ""])
    lines.extend(f"- {item}" for item in report["limitations"])  # type: ignore[union-attr]
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Metrics fingerprint: `{report['provenance']['metrics_fingerprint']}`",  # type: ignore[index]
            f"- Report fingerprint: `{report['report_fingerprint']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _frame_fingerprint(frame: pl.DataFrame) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema": [(name, str(dtype)) for name, dtype in frame.schema.items()],
            "rows": frame.to_dicts(),
        }
    )


def _fingerprint_without(row: Mapping[str, object], field: str) -> str:
    payload = dict(row)
    payload.pop(field)
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


def _nonnegative_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("generated_at must include a timezone")
    return value.astimezone(UTC)


__all__ = [
    "ADAPTIVE_WORKFLOW_EFFICIENCY_METRIC_SCHEMA",
    "ADAPTIVE_WORKFLOW_EFFICIENCY_METRICS_FILE",
    "ADAPTIVE_WORKFLOW_EFFICIENCY_REPORT_FILE",
    "ADAPTIVE_WORKFLOW_EFFICIENCY_SUMMARY_FILE",
    "AdaptiveWorkflowEfficiencyReport",
    "EFFICIENCY_EVIDENCE_STATUSES",
    "EFFICIENCY_METRICS",
    "adaptive_workflow_efficiency_metrics_frame",
    "build_adaptive_workflow_efficiency_report",
    "validate_adaptive_workflow_efficiency_metrics",
    "validate_adaptive_workflow_efficiency_report",
    "write_adaptive_workflow_efficiency_report",
]
