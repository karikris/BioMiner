from __future__ import annotations

import pytest

from biominer.gbif_quality.assertions import assertion_table, build_assertion


def test_assertion_has_complete_evidence_and_stable_identity() -> None:
    values = {
        "source_snapshot_version": "sha256:test",
        "source_row_id": "source:1",
        "gbif_id": "1",
        "target_field": "derived_year",
        "original_value": None,
        "derived_value": 2020,
        "evidence_source": "eventDate",
        "derivation_method": "iso_date_component",
        "derivation_rule_version": "v1",
        "confidence_class": "DETERMINISTIC_DERIVATION",
        "validation_status": "PASS",
        "conflict_status": "PASS",
        "retrieval_timestamp": "2026-07-22T00:00:00Z",
    }
    first = build_assertion(**values)
    second = build_assertion(**values)

    assert first.assertion_id == second.assertion_id
    assert first.to_row()["derived_value"] == "2020"
    assert assertion_table([first]).num_rows == 1


def test_assertion_rejects_uncalibrated_confidence_label() -> None:
    with pytest.raises(ValueError, match="confidence class"):
        build_assertion(
            source_snapshot_version="s",
            source_row_id="r",
            gbif_id="1",
            target_field="x",
            original_value=None,
            derived_value="y",
            evidence_source="e",
            derivation_method="m",
            derivation_rule_version="v1",
            confidence_class="0.99",
            validation_status="PASS",
            conflict_status="PASS",
            retrieval_timestamp="t",
        )
