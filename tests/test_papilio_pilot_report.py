from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from biominer.reports.adaptive_pilot_report import (
    load_adaptive_pilot_report,
    validate_adaptive_pilot_report,
)


REPORT = Path("reports/gbif_fast_start/papilio_demoleus_pilot_report.json")


def test_papilio_pilot_report_preserves_metric_evidence_status() -> None:
    report = load_adaptive_pilot_report(REPORT)
    assert report["metrics"]["reference_reviews_before_first_score"]["value"] == 0
    assert report["metrics"]["flickr_labels_reviewed"]["value"] == 0
    assert all(value is None for value in report["quality_metrics"].values())
    assert report["historical_context"]["counted_as_current_pilot"] is False
    assert report["strict_mode_comparison"]["provider_only_support_eligible"] is False


def test_papilio_pilot_report_rejects_metric_scope_tampering() -> None:
    report = load_adaptive_pilot_report(REPORT)
    tampered = deepcopy(report)
    tampered["metrics"]["embeddings_reused"]["evidence_status"] = "live"
    with pytest.raises(ValueError, match="evidence status"):
        validate_adaptive_pilot_report(tampered)
    tampered = deepcopy(report)
    tampered["quality_metrics"]["recall"] = 0.9
    with pytest.raises(ValueError, match="must be null"):
        validate_adaptive_pilot_report(tampered)
