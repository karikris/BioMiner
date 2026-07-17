from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from biominer.reports.adaptive_pilot_audit import (
    QUALITY_METRICS,
    load_pilot_audit,
    validate_pilot_audit,
)


REPORT = Path("reports/gbif_fast_start/papilio_demoleus_statistical_audit.json")


def test_papilio_audit_is_unavailable_and_emits_real_review_queue() -> None:
    report = load_pilot_audit(REPORT)
    assert report["metric_status"] == "insufficient_sample"
    assert report["reviewed_flickr_label_count"] == 0
    assert all(report["quality_metrics"][name] is None for name in QUALITY_METRICS)
    assert report["required_human_review_queue"]["row_count"] == 50
    assert report["reference_escalation_status"].startswith("deferred_")


def test_papilio_audit_rejects_fabricated_metric_and_premature_escalation() -> None:
    report = load_pilot_audit(REPORT)
    fabricated = deepcopy(report)
    fabricated["quality_metrics"]["recall"] = 0.9
    with pytest.raises(ValueError, match="fabricate"):
        validate_pilot_audit(fabricated)
    premature = deepcopy(report)
    premature["reference_escalation_status"] = "flagged"
    with pytest.raises(ValueError, match="await human labels"):
        validate_pilot_audit(premature)
