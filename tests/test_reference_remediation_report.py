from __future__ import annotations

from datetime import UTC, datetime
import json

import pytest

from biominer.references.adaptive_bank_revision import revise_adaptive_support_bank
from biominer.references.remediation_report import (
    build_reference_remediation_report,
    reference_remediation_impact_estimates_frame,
    write_reference_remediation_report,
)
from test_adaptive_bank_revision import (
    _dependencies,
    _review_with_verify_and_exclude,
    _support_manifest,
)
from test_targeted_reference_review import SHA_A, SHA_B


GENERATED_AT = datetime(2026, 7, 17, 15, 0, tzinfo=UTC)


def _revision_inputs():  # noqa: ANN202
    inputs, review = _review_with_verify_and_exclude()
    revision = revise_adaptive_support_bank(
        _support_manifest(inputs[0]),
        review,
        _dependencies(inputs[0]),
    )
    return review, revision


def test_report_contains_measured_remediation_and_qualified_impacts(tmp_path) -> None:
    review, revision = _revision_inputs()
    impacts = reference_remediation_impact_estimates_frame(
        [
            {
                "impact_scope_id": "flickr-partition:qld",
                "invalidated_artifact_id": "prototype:target",
                "species": "Papilio demoleus",
                "route": "adult_field",
                "expected_impacted_record_count": 17,
                "estimate_basis": "exact_dependency_index",
                "source_fingerprint": SHA_A,
            },
            {
                "impact_scope_id": "flickr-partition:nsw",
                "invalidated_artifact_id": "embedding:a",
                "species": "Papilio demoleus",
                "route": "adult_field",
                "expected_impacted_record_count": 5,
                "estimate_basis": "upper_bound",
                "source_fingerprint": SHA_B,
            },
        ]
    )

    result = build_reference_remediation_report(
        review,
        revision,
        impacts,
        generated_at=GENERATED_AT,
    )

    counts = result.report["counts"]
    assert counts == {
        "species_flagged": 1,
        "references_targeted": 2,
        "references_reviewed": 2,
        "references_verified": 1,
        "references_excluded": 1,
        "unchanged_provisional_references": 1,
        "flagged_review_pending": 0,
        "prototype_artifacts_invalidated": 1,
    }
    expected = result.report["expected_impacted_records"]
    assert expected["availability"] == "complete"
    assert expected["total_expected_impacted_records"] == 22
    prototypes = result.report["prototype_changes"]
    assert prototypes["observed_change_count"] is None
    assert prototypes["observation_status"] == "not_measured_rebuild_required"
    paths = write_reference_remediation_report(result, tmp_path)
    assert json.loads(paths["json"].read_text())["counts"] == counts
    assert "No observed before/after prototype change is claimed" in paths[
        "markdown"
    ].read_text()


def test_missing_impact_estimates_are_unavailable_not_zero() -> None:
    review, revision = _revision_inputs()

    result = build_reference_remediation_report(
        review,
        revision,
        reference_remediation_impact_estimates_frame([]),
        generated_at=GENERATED_AT,
    )

    expected = result.report["expected_impacted_records"]
    assert expected["availability"] == "unavailable"
    assert expected["total_expected_impacted_records"] is None
    assert expected["available_estimate_sum"] is None


def test_impact_estimate_cannot_claim_an_unaffected_artifact() -> None:
    review, revision = _revision_inputs()
    impacts = reference_remediation_impact_estimates_frame(
        [
            {
                "impact_scope_id": "unaffected",
                "invalidated_artifact_id": "embedding:unflagged",
                "species": "Papilio machaon",
                "route": "adult_field",
                "expected_impacted_record_count": 1,
                "estimate_basis": "upper_bound",
                "source_fingerprint": SHA_A,
            }
        ]
    )

    with pytest.raises(ValueError, match="not invalidated"):
        build_reference_remediation_report(
            review,
            revision,
            impacts,
            generated_at=GENERATED_AT,
        )


def test_impact_scopes_must_be_unique() -> None:
    row = {
        "impact_scope_id": "duplicate",
        "invalidated_artifact_id": "prototype:target",
        "species": "Papilio demoleus",
        "route": "adult_field",
        "expected_impacted_record_count": 1,
        "estimate_basis": "upper_bound",
        "source_fingerprint": SHA_A,
    }

    with pytest.raises(ValueError, match="repeat an impact scope"):
        reference_remediation_impact_estimates_frame([row, row])
