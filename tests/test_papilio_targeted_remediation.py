from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from biominer.reports.adaptive_pilot_remediation import (
    load_pilot_remediation,
    validate_pilot_remediation,
)


REPORT = Path(
    "reports/gbif_fast_start/papilio_demoleus_targeted_remediation.json"
)


def test_papilio_remediation_discloses_blocker_and_fixture_scope() -> None:
    report = load_pilot_remediation(REPORT)
    assert report["live_remediation"]["pending_human_review_count"] == 50
    assert report["live_remediation"]["species_legitimately_flagged"] == 0
    assert report["fixture_demonstration"]["counts"][
        "excluded_reference_rows"
    ] == 1
    assert report["fixture_demonstration"]["counts"]["flickr_rows_reused"] == 1


def test_papilio_remediation_rejects_overclaiming() -> None:
    report = load_pilot_remediation(REPORT)
    for field in (
        "fixture_remediation_is_live_remediation",
        "statistical_flag_proves_reference_identity",
        "unaffected_species_requires_review",
        "quality_improvement_claimed",
    ):
        tampered = deepcopy(report)
        tampered["semantics"][field] = True
        with pytest.raises(ValueError, match="semantics"):
            validate_pilot_remediation(tampered)
