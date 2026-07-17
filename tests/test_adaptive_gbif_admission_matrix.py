from __future__ import annotations

import pytest

from biominer.references.admission import default_reference_admission_policy
from biominer.references.admission_eligibility import (
    EligibilityDecision,
    evaluate_gbif_provisional_eligibility,
)
from biominer.references.adaptive_bank_revision import revise_adaptive_support_bank
from test_adaptive_bank_revision import (
    _dependencies,
    _review_with_verify_and_exclude,
    _support_manifest,
)
from test_gbif_admission_eligibility import _eligible_evidence


@pytest.mark.parametrize(
    ("case", "changes", "decision", "reason"),
    [
        pytest.param(
            "valid_unreviewed_gbif_adult_field",
            {},
            EligibilityDecision.ADMITTED,
            None,
            id="valid-unreviewed-gbif-adult-field",
        ),
        pytest.param(
            "non_gbif_unreviewed",
            {"source": "inaturalist"},
            EligibilityDecision.EXCLUDED,
            "provider_source_not_allowed",
            id="non-gbif-unreviewed",
        ),
        pytest.param(
            "wrong_taxon",
            {"provider_identity_matches_candidate_taxon": False},
            EligibilityDecision.EXCLUDED,
            "provider_identity_mismatch",
            id="wrong-taxon",
        ),
        pytest.param(
            "ambiguous_taxon",
            {"uncertain_taxon_match": True},
            EligibilityDecision.EXCLUDED,
            "uncertain_taxon_match",
            id="ambiguous-taxon",
        ),
        pytest.param(
            "fossil",
            {"fossil": True},
            EligibilityDecision.EXCLUDED,
            "fossil_record",
            id="fossil",
        ),
        pytest.param(
            "preserved_specimen_in_adult_bank",
            {"yoloe_route": "pinned_specimen"},
            EligibilityDecision.EXCLUDED,
            "route_incompatible_with_requested_bank",
            id="preserved-specimen",
        ),
        pytest.param(
            "invalid_licence",
            {"licence_policy_status": "research_only"},
            EligibilityDecision.EXCLUDED,
            "licence_not_accepted_for_configured_use",
            id="invalid-licence",
        ),
        pytest.param(
            "failed_decode",
            {"decode_status": "decode_failed"},
            EligibilityDecision.EXCLUDED,
            "image_decode_not_valid",
            id="failed-decode",
        ),
        pytest.param(
            "duplicate_conflict",
            {"duplicate_resolution_status": "conflict"},
            EligibilityDecision.EXCLUDED,
            "unresolved_duplicate_conflict",
            id="duplicate-conflict",
        ),
        pytest.param(
            "artifact_route",
            {"visual_domain": "logo"},
            EligibilityDecision.EXCLUDED,
            "artifact_or_unsuitable_domain",
            id="artifact-route",
        ),
        pytest.param(
            "tiny_subject",
            {"subject_area_ratio": 0.01},
            EligibilityDecision.EXCLUDED,
            "subject_area_below_policy",
            id="tiny-subject",
        ),
    ],
)
def test_adaptive_gbif_admission_policy_matrix(
    case: str,
    changes: dict[str, object],
    decision: EligibilityDecision,
    reason: str | None,
) -> None:
    result = evaluate_gbif_provisional_eligibility(
        _eligible_evidence(**changes),
        default_reference_admission_policy(),
    )

    assert case
    assert result.decision is decision
    assert result.human_verified is False
    if reason is None:
        assert result.reason_codes == ()
    else:
        assert reason in result.reason_codes


def test_human_rejection_overrides_and_verification_promotes_provisional_support() -> (
    None
):
    inputs, review = _review_with_verify_and_exclude()
    revision = revise_adaptive_support_bank(
        _support_manifest(inputs[0]),
        review,
        _dependencies(inputs[0]),
    )
    revised = {
        row["reference_media_id"]: row
        for row in revision.revised_support_manifest.iter_rows(named=True)
    }
    verified_id = review.workflow.verified["reference_media_id"].item()
    excluded_id = review.workflow.excluded["reference_media_id"].item()

    assert revised[verified_id]["identity_evidence_basis"] == "human_verified"
    assert revised[verified_id]["human_verified_identity"] is True
    assert revised[verified_id]["provisional_support"] is False
    assert revised[verified_id]["support_eligible"] is True
    assert revised[excluded_id]["identity_evidence_basis"] == "none"
    assert revised[excluded_id]["human_verified_identity"] is False
    assert revised[excluded_id]["provisional_support"] is False
    assert revised[excluded_id]["support_eligible"] is False
