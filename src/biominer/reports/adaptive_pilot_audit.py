"""Fail-closed validation for the Papilio statistical-audit status."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.holdouts import load_natural_stream_selection


PILOT_AUDIT_SCHEMA_VERSION = "adaptive-pilot-statistical-audit-v1.0.0"
QUALITY_METRICS = (
    "precision_ci_lower",
    "recall",
    "false_positive_rate",
    "competitor_confusion_rate",
)


def load_pilot_audit(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pilot audit must contain an object")
    validate_pilot_audit(payload)
    return payload


def validate_pilot_audit(report: Mapping[str, object]) -> None:
    if report.get("schema_version") != PILOT_AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported pilot audit schema")
    if report.get("metric_status") != "insufficient_sample":
        raise ValueError("pilot audit must disclose insufficient sample")
    if report.get("reviewed_flickr_label_count") != 0:
        raise ValueError("pilot audit reviewed-label count mismatch")
    metrics = report.get("quality_metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(QUALITY_METRICS):
        raise ValueError("pilot audit metric fields are incomplete")
    if any(metrics[name] is not None for name in QUALITY_METRICS):
        raise ValueError("pilot audit cannot fabricate unavailable metrics")
    queue = report.get("required_human_review_queue")
    if not isinstance(queue, Mapping):
        raise ValueError("pilot audit review queue evidence is missing")
    selection = load_natural_stream_selection(str(queue.get("path")))
    if queue.get("row_count") != selection.height or selection.height != 50:
        raise ValueError("pilot audit review queue count mismatch")
    if queue.get("selection_fingerprint") != selection[
        "selection_fingerprint"
    ].item(0):
        raise ValueError("pilot audit queue fingerprint mismatch")
    if queue.get("sampling_frame_fingerprint") != selection[
        "sampling_frame_fingerprint"
    ].item(0):
        raise ValueError("pilot audit sampling fingerprint mismatch")
    if report.get("reference_escalation_status") != (
        "deferred_pending_human_flickr_labels"
    ):
        raise ValueError("pilot reference escalation must await human labels")
    semantics = report.get("semantics")
    if semantics != {
        "sampling_selection_used_outcomes": False,
        "missing_metrics_interpreted_as_zero": False,
        "statistical_audit_proves_reference_identity": False,
    }:
        raise ValueError("pilot audit semantics were weakened")
    expected = canonical_semantic_fingerprint(
        {key: value for key, value in report.items() if key != "report_fingerprint"}
    )
    if report.get("report_fingerprint") != expected:
        raise ValueError("pilot audit report fingerprint mismatch")
