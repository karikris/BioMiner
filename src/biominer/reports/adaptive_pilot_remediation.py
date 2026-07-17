"""Validation for fixture-backed Papilio remediation evidence."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.reports.adaptive_pilot_audit import load_pilot_audit


PILOT_REMEDIATION_SCHEMA_VERSION = "adaptive-pilot-remediation-v1.0.0"


def load_pilot_remediation(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pilot remediation report must contain an object")
    validate_pilot_remediation(payload)
    return payload


def validate_pilot_remediation(report: Mapping[str, object]) -> None:
    if report.get("schema_version") != PILOT_REMEDIATION_SCHEMA_VERSION:
        raise ValueError("unsupported pilot remediation schema")
    audit_path = report.get("audit_report_path")
    if not isinstance(audit_path, str):
        raise ValueError("pilot remediation audit path is missing")
    audit = load_pilot_audit(audit_path)
    live = _mapping(report, "live_remediation")
    if live.get("status") != "blocked_pending_human_flickr_review":
        raise ValueError("live remediation blocker is not explicit")
    if live.get("pending_human_review_count") != audit[
        "required_human_review_queue"
    ]["row_count"]:
        raise ValueError("live remediation review count mismatch")
    if live.get("species_legitimately_flagged") != 0:
        raise ValueError("unavailable audit cannot legitimately flag a species")
    fixture = _mapping(report, "fixture_demonstration")
    expected_counts = {
        "flagged_species": 1,
        "unaffected_species": 1,
        "targeted_reference_rows": 2,
        "verified_reference_rows": 1,
        "excluded_reference_rows": 1,
        "embedding_rows_reused": 2,
        "flickr_rows_selectively_rescored": 1,
        "flickr_rows_reused": 1,
    }
    if fixture.get("status") != "passed" or fixture.get("counts") != expected_counts:
        raise ValueError("pilot remediation fixture evidence mismatch")
    if fixture.get("quality_comparison_status") != "unavailable_no_human_labels":
        raise ValueError("pilot remediation cannot fabricate quality comparison")
    semantics = _mapping(report, "semantics")
    if semantics != {
        "fixture_remediation_is_live_remediation": False,
        "statistical_flag_proves_reference_identity": False,
        "unaffected_species_requires_review": False,
        "quality_improvement_claimed": False,
    }:
        raise ValueError("pilot remediation semantics were weakened")
    expected = canonical_semantic_fingerprint(
        {key: value for key, value in report.items() if key != "report_fingerprint"}
    )
    if report.get("report_fingerprint") != expected:
        raise ValueError("pilot remediation report fingerprint mismatch")


def _mapping(parent: Mapping[str, object], field: str) -> Mapping[str, object]:
    value = parent.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"pilot remediation {field} must be an object")
    return value
