from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from biominer.references.admission import default_reference_admission_policy
from biominer.references.admission_eligibility import (
    ELIGIBILITY_GATE_IDS,
    GBIF_PROVIDER_ASSERTION_IDENTITY_BASIS,
    EligibilityDecision,
    GBIFEligibilityEvidence,
    GateDisposition,
    evaluate_gbif_provisional_eligibility,
)


def _eligible_evidence(**changes: object) -> GBIFEligibilityEvidence:
    values: dict[str, object] = {
        "source": "GBIF",
        "taxon_reconciliation_status": "accepted_key_exact",
        "resolves_to_candidate_accepted_taxon_key": True,
        "uncertain_taxon_match": False,
        "occurrence_absent": False,
        "fossil": False,
        "media_type": "StillImage",
        "download_status": "complete",
        "content_type": "image/jpeg",
        "decode_status": "valid",
        "decoded_width": 1024,
        "decoded_height": 768,
        "image_sha256": "a" * 64,
        "licence_policy_status": "allowed",
        "creator": "Example observer",
        "source_url": "https://example.test/occurrence/1",
        "attribution": "Example observer / CC BY 4.0",
        "duplicate_processing_completed": True,
        "canonical_media": True,
        "duplicate_resolution_status": "resolved",
        "duplicate_conflict_targeted_review": False,
        "provider_identity_matches_candidate_taxon": True,
        "independence_processing_completed": True,
        "selected_images_from_observation": 1,
        "observer_identity_available": True,
        "observer_image_ordinal_before_reuse": 1,
        "observer_reuse_justified": False,
        "near_identical_view": False,
        "distinct_additional_view_justified": False,
        "yoloe_routing_completed": True,
        "yoloe_route": "adult_field",
        "requested_bank_route": "adult_field",
        "visual_domain": "live_field",
        "subject_present": True,
        "ambiguous_domain_targeted_review": False,
        "subject_area_ratio": 0.25,
        "full_frame_input_generation_succeeded": True,
        "prototype_scope": "regional",
        "usable_geography": True,
    }
    values.update(changes)
    return GBIFEligibilityEvidence(**values)  # type: ignore[arg-type]


def test_all_non_negotiable_gates_admit_provider_asserted_evidence() -> None:
    evidence = _eligible_evidence()

    result = evaluate_gbif_provisional_eligibility(
        evidence, default_reference_admission_policy()
    )

    assert result.decision is EligibilityDecision.ADMITTED
    assert tuple(gate.gate_id for gate in result.gate_results) == ELIGIBILITY_GATE_IDS
    assert len(result.gate_results) == 24
    assert {gate.disposition for gate in result.gate_results} == {
        GateDisposition.PASSED
    }
    assert result.reason_codes == ()
    assert result.identity_basis == GBIF_PROVIDER_ASSERTION_IDENTITY_BASIS
    assert result.human_verified is False
    assert result.geographic_prototype_eligible is True


@pytest.mark.parametrize(
    ("changes", "reason_code"),
    [
        ({"source": "inaturalist"}, "provider_source_not_allowed"),
        (
            {"taxon_reconciliation_status": "unresolved"},
            "provider_taxon_not_reconciled_to_candidate_accepted_key",
        ),
        (
            {"resolves_to_candidate_accepted_taxon_key": False},
            "provider_taxon_not_reconciled_to_candidate_accepted_key",
        ),
        ({"uncertain_taxon_match": True}, "uncertain_taxon_match"),
        ({"occurrence_absent": True}, "occurrence_is_absent"),
        ({"fossil": True}, "fossil_record"),
        ({"media_type": "MovingImage"}, "unsupported_media_type"),
        ({"download_status": "failed"}, "download_unavailable"),
        ({"content_type": "text/html"}, "raster_content_type_invalid_or_missing"),
        ({"decode_status": "decode_failed"}, "image_decode_not_valid"),
        ({"decoded_width": 511}, "decoded_dimensions_below_policy"),
        ({"image_sha256": "not-a-hash"}, "image_sha256_missing_or_invalid"),
        (
            {"licence_policy_status": "research_only"},
            "licence_not_accepted_for_configured_use",
        ),
        ({"creator": None}, "creator_source_or_attribution_missing"),
        ({"canonical_media": False}, "noncanonical_media"),
        (
            {"duplicate_resolution_status": "conflict"},
            "unresolved_duplicate_conflict",
        ),
        (
            {"provider_identity_matches_candidate_taxon": False},
            "provider_identity_mismatch",
        ),
        ({"selected_images_from_observation": 2}, "observation_quota_exceeded"),
        ({"near_identical_view": True}, "near_identical_view_reuse"),
        ({"yoloe_route": "larval"}, "route_incompatible_with_requested_bank"),
        ({"visual_domain": "logo"}, "artifact_or_unsuitable_domain"),
        ({"subject_present": False}, "no_organism_detected"),
        (
            {"visual_domain": "ambiguous"},
            "ambiguous_domain_not_admissible",
        ),
        ({"subject_area_ratio": 0.049}, "subject_area_below_policy"),
        (
            {"prototype_scope": "local", "usable_geography": False},
            "geographic_scope_missing_usable_geography",
        ),
    ],
)
def test_hard_gate_failures_are_excluded(
    changes: dict[str, object], reason_code: str
) -> None:
    result = evaluate_gbif_provisional_eligibility(
        _eligible_evidence(**changes), default_reference_admission_policy()
    )

    assert result.decision is EligibilityDecision.EXCLUDED
    assert reason_code in result.reason_codes
    assert len(result.gate_results) == 24


@pytest.mark.parametrize(
    ("changes", "reason_code"),
    [
        ({"download_status": "pending"}, "download_not_completed"),
        ({"decoded_width": None}, "decoded_dimensions_missing"),
        (
            {
                "duplicate_processing_completed": False,
                "duplicate_resolution_status": None,
            },
            "duplicate_processing_incomplete",
        ),
        (
            {"duplicate_resolution_status": "review_required"},
            "duplicate_conflict_targeted_review",
        ),
        (
            {
                "duplicate_resolution_status": "conflict",
                "duplicate_conflict_targeted_review": True,
            },
            "duplicate_conflict_targeted_review",
        ),
        (
            {"independence_processing_completed": False},
            "independence_processing_incomplete",
        ),
        ({"observer_identity_available": False}, "observer_identity_missing"),
        (
            {"yoloe_routing_completed": False, "yoloe_route": None},
            "yoloe_routing_incomplete",
        ),
        (
            {
                "yoloe_route": "larval",
                "requested_bank_route": "larval",
                "visual_domain": "live_field",
            },
            "route_requires_human_review",
        ),
        ({"subject_present": None}, "subject_presence_unresolved"),
        (
            {
                "visual_domain": "ambiguous",
                "ambiguous_domain_targeted_review": True,
            },
            "ambiguous_domain_targeted_review",
        ),
        ({"subject_area_ratio": None}, "subject_area_not_measured"),
        (
            {"full_frame_input_generation_succeeded": False},
            "full_frame_input_generation_failed",
        ),
    ],
)
def test_incomplete_or_targeted_evidence_requires_review(
    changes: dict[str, object], reason_code: str
) -> None:
    result = evaluate_gbif_provisional_eligibility(
        _eligible_evidence(**changes), default_reference_admission_policy()
    )

    assert result.decision is EligibilityDecision.REVIEW_REQUIRED
    assert reason_code in result.reason_codes


def test_exclusion_takes_precedence_over_review_and_all_reasons_are_retained() -> None:
    result = evaluate_gbif_provisional_eligibility(
        _eligible_evidence(
            occurrence_absent=True,
            download_status="pending",
            subject_area_ratio=None,
        ),
        default_reference_admission_policy(),
    )

    assert result.decision is EligibilityDecision.EXCLUDED
    assert result.reason_codes == (
        "occurrence_is_absent",
        "download_not_completed",
        "subject_area_not_measured",
    )


def test_global_candidate_can_pass_without_geography_but_is_not_geo_eligible() -> None:
    result = evaluate_gbif_provisional_eligibility(
        _eligible_evidence(prototype_scope="global", usable_geography=False),
        default_reference_admission_policy(),
    )

    assert result.decision is EligibilityDecision.ADMITTED
    assert result.geographic_prototype_eligible is False
    assert (
        result.gate_results[-1].reason_code
        == "global_scope_allows_missing_geography"
    )


def test_observer_reuse_is_allowed_only_after_explicit_justification() -> None:
    policy = default_reference_admission_policy()
    excluded = evaluate_gbif_provisional_eligibility(
        _eligible_evidence(observer_image_ordinal_before_reuse=2), policy
    )
    admitted = evaluate_gbif_provisional_eligibility(
        _eligible_evidence(
            observer_image_ordinal_before_reuse=2, observer_reuse_justified=True
        ),
        policy,
    )

    assert excluded.decision is EligibilityDecision.EXCLUDED
    assert "observer_reused_too_early" in excluded.reason_codes
    assert admitted.decision is EligibilityDecision.ADMITTED


def test_second_observation_image_requires_distinct_view_justification() -> None:
    policy = default_reference_admission_policy()
    excluded = evaluate_gbif_provisional_eligibility(
        _eligible_evidence(selected_images_from_observation=2), policy
    )
    admitted = evaluate_gbif_provisional_eligibility(
        _eligible_evidence(
            selected_images_from_observation=2,
            distinct_additional_view_justified=True,
        ),
        policy,
    )

    assert excluded.decision is EligibilityDecision.EXCLUDED
    assert "observation_quota_exceeded" in excluded.reason_codes
    assert admitted.decision is EligibilityDecision.ADMITTED


def test_evaluator_is_deterministic_and_does_not_mutate_evidence() -> None:
    evidence = _eligible_evidence()
    policy = default_reference_admission_policy()
    before = evidence.identity_payload()

    first = evaluate_gbif_provisional_eligibility(evidence, policy)
    second = evaluate_gbif_provisional_eligibility(evidence, policy)

    assert evidence.identity_payload() == before
    assert first == second
    assert first.fingerprint == second.fingerprint
    with pytest.raises(FrozenInstanceError):
        evidence.source = "other"  # type: ignore[misc]


def test_provider_assertion_result_cannot_be_relabelled_as_human_verified() -> None:
    result = evaluate_gbif_provisional_eligibility(
        _eligible_evidence(), default_reference_admission_policy()
    )

    with pytest.raises(ValueError, match="cannot be human verified"):
        replace(result, human_verified=True)


def test_yoloe_route_is_only_domain_routing_not_species_identity() -> None:
    evidence_fields = set(GBIFEligibilityEvidence.__dataclass_fields__)

    assert "yoloe_route" in evidence_fields
    assert not any(
        "species" in field or "scientific_name" in field
        for field in evidence_fields
        if field.startswith("yoloe")
    )
    result = evaluate_gbif_provisional_eligibility(
        _eligible_evidence(provider_identity_matches_candidate_taxon=False),
        default_reference_admission_policy(),
    )
    assert result.decision is EligibilityDecision.EXCLUDED
    assert "provider_identity_mismatch" in result.reason_codes
