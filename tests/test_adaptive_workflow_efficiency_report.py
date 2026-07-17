from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json

import polars as pl
import pytest

from biominer.reports.workflow_efficiency import (
    ADAPTIVE_WORKFLOW_EFFICIENCY_METRIC_SCHEMA,
    adaptive_workflow_efficiency_metrics_frame,
    build_adaptive_workflow_efficiency_report,
    validate_adaptive_workflow_efficiency_metrics,
    write_adaptive_workflow_efficiency_report,
)


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _row(
    name: str,
    unit: str,
    value: float | None,
    *,
    status: str = "fixture",
    reason: str | None = None,
) -> dict[str, object]:
    return {
        "metric_name": name,
        "unit_name": unit,
        "metric_value": value,
        "evidence_status": status,
        "source_artifact_id": f"fixture:{name}:{unit}",
        "source_artifact_fingerprint": _sha(f"fixture:{name}:{unit}"),
        "derivation": None if value is None else "fixture observation",
        "unavailable_reason": reason,
    }


def _rows() -> list[dict[str, object]]:
    return [
        _row("time_to_first_score", "seconds", 12.5),
        _row("manual_reviews_avoided", "reviews", 80.0, status="derived"),
        _row("manual_reviews_triggered_later", "reviews", 4.0),
        _row("embeddings_reused", "embeddings", 100.0),
        _row("yoloe_runs_reused", "runs", 10.0),
        _row("prototypes_rebuilt", "prototypes", 2.0),
        _row("records_selectively_rescored", "records", 25.0),
        _row(
            "full_rerun_work_avoided",
            "records",
            75.0,
            status="derived",
        ),
        _row(
            "full_rerun_work_avoided",
            "embeddings",
            100.0,
            status="derived",
        ),
        _row("peak_memory", "bytes", 1024.0),
    ]


def test_efficiency_report_covers_required_metrics_without_mixing_units(
    tmp_path,
) -> None:
    metrics = adaptive_workflow_efficiency_metrics_frame(_rows())
    result = build_adaptive_workflow_efficiency_report(
        metrics,
        generated_at=datetime(2026, 7, 18, tzinfo=UTC),
    )

    assert metrics.schema == ADAPTIVE_WORKFLOW_EFFICIENCY_METRIC_SCHEMA
    avoided = metrics.filter(
        pl.col("metric_name") == "full_rerun_work_avoided"
    )
    assert avoided["unit_name"].to_list() == ["embeddings", "records"]
    assert result.report["evidence_summary"] == {
        "available_metric_rows": 10,
        "unavailable_metric_rows": 0,
    }
    assert result.report["provenance"][  # type: ignore[index]
        "aggregation_policy"
    ] == "never_sum_different_unit_names"

    paths = write_adaptive_workflow_efficiency_report(result, tmp_path)
    assert pl.read_parquet(paths["metrics"]).equals(metrics)
    assert json.loads(paths["json"].read_text()) == result.report
    assert paths["markdown"].read_text().startswith(
        "# Adaptive workflow efficiency"
    )


def test_unavailable_metric_is_not_zero_and_requires_reason() -> None:
    rows = _rows()
    rows[0] = _row(
        "time_to_first_score",
        "seconds",
        None,
        status="not_instrumented",
        reason="start timestamp was not recorded",
    )
    metrics = adaptive_workflow_efficiency_metrics_frame(rows)
    value = metrics.filter(pl.col("metric_name") == "time_to_first_score")[
        "metric_value"
    ].item()
    assert value is None

    rows[0]["unavailable_reason"] = None
    with pytest.raises(ValueError, match="unavailable_reason"):
        adaptive_workflow_efficiency_metrics_frame(rows)


def test_counterfactual_savings_cannot_be_labelled_directly_measured() -> None:
    rows = _rows()
    rows[1]["evidence_status"] = "measured"

    with pytest.raises(ValueError, match="counterfactual"):
        adaptive_workflow_efficiency_metrics_frame(rows)


def test_efficiency_validator_rejects_tampering_and_missing_metric() -> None:
    metrics = adaptive_workflow_efficiency_metrics_frame(_rows())
    tampered = metrics.with_columns(
        pl.when(pl.col("metric_name") == "peak_memory")
        .then(pl.lit(2048.0))
        .otherwise(pl.col("metric_value"))
        .alias("metric_value")
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_adaptive_workflow_efficiency_metrics(tampered)

    missing = metrics.filter(pl.col("metric_name") != "yoloe_runs_reused")
    with pytest.raises(ValueError, match="cover every metric"):
        validate_adaptive_workflow_efficiency_metrics(missing)
