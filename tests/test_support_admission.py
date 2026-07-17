from __future__ import annotations

from dataclasses import replace

from biominer.references.admission import (
    default_reference_admission_policy,
    strict_reference_admission_policy,
)
from biominer.references.support_admission import (
    SupportAdmissionEvidence,
    evaluate_support_admission,
)


def _evidence(**changes: object) -> SupportAdmissionEvidence:
    values: dict[str, object] = {
        "source": "gbif",
        "review_status": "not_requested",
        "verification_status": "unreviewed",
        "target_identity_verified": False,
        "human_rejected": False,
        "human_rejection_reasons": (),
        "provider_assertion_passed": True,
        "automated_admission_gates_passed": True,
        "route_compatible": True,
        "canonical_media": True,
        "deduplication_completed": True,
        "provisional_support_declared": True,
        "policy_permits_human_verified_support": True,
        "policy_permits_provisional_support": True,
        "exclusion_reasons": (),
    }
    values.update(changes)
    return SupportAdmissionEvidence(**values)  # type: ignore[arg-type]


def test_strict_human_verified_path_is_eligible() -> None:
    result = evaluate_support_admission(
        _evidence(
            review_status="completed",
            verification_status="verified",
            target_identity_verified=True,
            provider_assertion_passed=False,
            automated_admission_gates_passed=False,
            provisional_support_declared=False,
            policy_permits_provisional_support=False,
        ),
        strict_reference_admission_policy(),
    )

    assert result.eligible is True
    assert result.evidence_path == "human_verified"
    assert result.provisional is False
    assert result.reasons == ("strict_human_review_verified",)


def test_adaptive_gbif_provider_path_is_eligible_but_provisional() -> None:
    result = evaluate_support_admission(
        _evidence(), default_reference_admission_policy()
    )

    assert result.eligible is True
    assert result.evidence_path == "gbif_provider_asserted"
    assert result.provisional is True
    assert result.reasons == ("automated_gbif_quality_gates_passed",)


def test_strict_policy_never_admits_provider_assertion_only() -> None:
    result = evaluate_support_admission(
        _evidence(), strict_reference_admission_policy()
    )

    assert result.eligible is False
    assert result.evidence_path == "none"
    assert "policy_does_not_enable_provisional_support" in result.reasons


def test_human_rejection_overrides_complete_provider_and_automated_evidence() -> None:
    result = evaluate_support_admission(
        _evidence(
            human_rejected=True,
            human_rejection_reasons=("wrong species",),
        ),
        default_reference_admission_policy(),
    )

    assert result.eligible is False
    assert result.evidence_path == "none"
    assert result.reasons == (
        "human_rejection:wrong species",
        "human_rejection_override",
    )


def test_every_adaptive_gate_is_required() -> None:
    checks = {
        "provider_assertion_passed": "provider_assertion_failed",
        "automated_admission_gates_passed": "automated_admission_gates_failed",
        "route_compatible": "reference_route_incompatible",
        "canonical_media": "noncanonical_media",
        "deduplication_completed": "deduplication_incomplete",
        "provisional_support_declared": "provisional_support_not_declared",
        "policy_permits_provisional_support": "provisional_support_not_permitted",
    }
    for field, reason in checks.items():
        result = evaluate_support_admission(
            replace(_evidence(), **{field: False}),
            default_reference_admission_policy(),
        )
        assert result.eligible is False
        assert reason in result.reasons


def test_explicit_exclusion_precedes_both_admission_paths() -> None:
    result = evaluate_support_admission(
        _evidence(
            review_status="completed",
            verification_status="verified",
            target_identity_verified=True,
            exclusion_reasons=("licence_not_accepted",),
        ),
        default_reference_admission_policy(),
    )

    assert result.eligible is False
    assert result.reasons == ("licence_not_accepted",)
