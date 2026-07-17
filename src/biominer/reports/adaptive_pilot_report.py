"""Cross-artifact validation for the Papilio adaptive pilot report."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.reports.adaptive_pilot_audit import load_pilot_audit
from biominer.reports.adaptive_pilot_initial import load_initial_pilot_report
from biominer.reports.adaptive_pilot_remediation import load_pilot_remediation


PILOT_REPORT_SCHEMA_VERSION = "adaptive-pilot-report-v1.0.0"
REQUIRED_METRICS = (
    "time_to_first_score_ms",
    "references_admitted",
    "reference_reviews_before_first_score",
    "flickr_labels_reviewed",
    "species_flagged",
    "reference_images_later_reviewed",
    "embeddings_reused",
    "records_selectively_rescored",
)


def load_adaptive_pilot_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("adaptive pilot report must contain an object")
    validate_adaptive_pilot_report(payload)
    return payload


def validate_adaptive_pilot_report(report: Mapping[str, object]) -> None:
    if report.get("schema_version") != PILOT_REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported adaptive pilot report schema")
    sources = report.get("source_reports")
    if not isinstance(sources, Mapping):
        raise ValueError("adaptive pilot source reports are missing")
    initial = load_initial_pilot_report(str(sources.get("initial_scoring")))
    audit = load_pilot_audit(str(sources.get("statistical_audit")))
    remediation = load_pilot_remediation(str(sources.get("remediation")))
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping) or tuple(metrics) != REQUIRED_METRICS:
        raise ValueError("adaptive pilot metric set is incomplete or reordered")
    expected = {
        "time_to_first_score_ms": (502.480375, "measured_fixture"),
        "references_admitted": (1, "fixture"),
        "reference_reviews_before_first_score": (0, "fixture"),
        "flickr_labels_reviewed": (0, "observed_workspace"),
        "species_flagged": (0, "unavailable_pending_labels"),
        "reference_images_later_reviewed": (0, "unavailable_pending_labels"),
        "embeddings_reused": (2, "fixture"),
        "records_selectively_rescored": (1, "fixture"),
    }
    for name, (value, evidence_status) in expected.items():
        row = metrics.get(name)
        if not isinstance(row, Mapping) or row.get("value") != value:
            raise ValueError(f"adaptive pilot metric mismatch: {name}")
        if row.get("evidence_status") != evidence_status:
            raise ValueError(f"adaptive pilot evidence status mismatch: {name}")
    if metrics["time_to_first_score_ms"]["value"] != initial[
        "current_execution"
    ]["metrics"]["time_to_first_provisional_scoring_ms"]:
        raise ValueError("adaptive pilot first-score source mismatch")
    if metrics["flickr_labels_reviewed"]["value"] != audit[
        "reviewed_flickr_label_count"
    ]:
        raise ValueError("adaptive pilot Flickr review source mismatch")
    if remediation["live_remediation"]["species_legitimately_flagged"] != 0:
        raise ValueError("adaptive pilot live flag count mismatch")
    quality = report.get("quality_metrics")
    if not isinstance(quality, Mapping) or any(
        value is not None for value in quality.values()
    ):
        raise ValueError("adaptive pilot unavailable quality metrics must be null")
    strict = report.get("strict_mode_comparison")
    if not isinstance(strict, Mapping) or strict.get(
        "provider_only_support_eligible"
    ) is not False:
        raise ValueError("strict pilot comparison was weakened")
    if strict.get("final_flickr_human_review_required") is not True:
        raise ValueError("strict comparison must retain Flickr human review")
    if not isinstance(report.get("limitations"), list) or len(
        report["limitations"]
    ) < 5:
        raise ValueError("adaptive pilot limitations are incomplete")
    fingerprint = canonical_semantic_fingerprint(
        {key: value for key, value in report.items() if key != "report_fingerprint"}
    )
    if report.get("report_fingerprint") != fingerprint:
        raise ValueError("adaptive pilot report fingerprint mismatch")
