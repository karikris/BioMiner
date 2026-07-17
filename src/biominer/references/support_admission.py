"""Policy-driven eligibility for strict and provisional support rows."""

from __future__ import annotations

from dataclasses import dataclass

from biominer.references.admission import (
    DEFAULT_REFERENCE_ADMISSION_MODE,
    ReferenceAdmissionPolicy,
)


SUPPORT_EVIDENCE_PATHS = frozenset(
    {"human_verified", "gbif_provider_asserted", "none"}
)


@dataclass(frozen=True, slots=True)
class SupportAdmissionEvidence:
    """Facts required to choose one support-admission evidence path."""

    source: str
    review_status: str
    verification_status: str
    target_identity_verified: bool
    human_rejected: bool
    human_rejection_reasons: tuple[str, ...]
    provider_assertion_passed: bool
    automated_admission_gates_passed: bool
    route_compatible: bool
    canonical_media: bool
    deduplication_completed: bool
    provisional_support_declared: bool
    policy_permits_human_verified_support: bool
    policy_permits_provisional_support: bool
    exclusion_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in ("source", "review_status", "verification_status"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be nonblank text")
            object.__setattr__(self, field, value.strip().casefold())
        for field in (
            "target_identity_verified",
            "human_rejected",
            "provider_assertion_passed",
            "automated_admission_gates_passed",
            "route_compatible",
            "canonical_media",
            "deduplication_completed",
            "provisional_support_declared",
            "policy_permits_human_verified_support",
            "policy_permits_provisional_support",
        ):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be Boolean")
        for field in ("human_rejection_reasons", "exclusion_reasons"):
            values = getattr(self, field)
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"{field} must be a tuple of nonblank strings")
            object.__setattr__(
                self,
                field,
                tuple(sorted({value.strip().casefold() for value in values})),
            )
        if self.human_rejected and not self.human_rejection_reasons:
            raise ValueError("human rejection requires at least one reason")
        if not self.human_rejected and self.human_rejection_reasons:
            raise ValueError("human rejection reasons require human_rejected")


@dataclass(frozen=True, slots=True)
class SupportAdmissionEvaluation:
    """Fail-closed support decision and the accepted evidence path."""

    eligible: bool
    admission_status: str
    evidence_path: str
    provisional: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.admission_status not in {"admitted", "excluded"}:
            raise ValueError("unsupported support admission status")
        if self.evidence_path not in SUPPORT_EVIDENCE_PATHS:
            raise ValueError("unsupported support evidence path")
        if self.eligible != (self.admission_status == "admitted"):
            raise ValueError("support eligibility and admission status disagree")
        if self.eligible and self.evidence_path == "none":
            raise ValueError("eligible support requires an evidence path")
        if self.provisional != (self.evidence_path == "gbif_provider_asserted"):
            raise ValueError("provisional state and evidence path disagree")
        if not self.reasons or self.reasons != tuple(sorted(set(self.reasons))):
            raise ValueError("support admission reasons must be canonical")


def evaluate_support_admission(
    evidence: SupportAdmissionEvidence,
    policy: ReferenceAdmissionPolicy,
) -> SupportAdmissionEvaluation:
    """Evaluate human rejection first, then strict and provisional paths."""

    if evidence.human_rejected:
        return _excluded(
            "human_rejection_override",
            *(
                f"human_rejection:{reason}"
                for reason in evidence.human_rejection_reasons
            ),
        )
    if evidence.exclusion_reasons:
        return _excluded(*evidence.exclusion_reasons)

    human_path = (
        evidence.policy_permits_human_verified_support
        and evidence.review_status == "completed"
        and evidence.verification_status == "verified"
        and evidence.target_identity_verified
    )
    if human_path:
        return SupportAdmissionEvaluation(
            eligible=True,
            admission_status="admitted",
            evidence_path="human_verified",
            provisional=False,
            reasons=("strict_human_review_verified",),
        )

    provisional_path = (
        policy.mode == DEFAULT_REFERENCE_ADMISSION_MODE
        and evidence.policy_permits_provisional_support
        and evidence.source == "gbif"
        and evidence.provider_assertion_passed
        and evidence.automated_admission_gates_passed
        and evidence.route_compatible
        and evidence.canonical_media
        and evidence.deduplication_completed
        and evidence.provisional_support_declared
    )
    if provisional_path:
        return SupportAdmissionEvaluation(
            eligible=True,
            admission_status="admitted",
            evidence_path="gbif_provider_asserted",
            provisional=True,
            reasons=("automated_gbif_quality_gates_passed",),
        )

    missing: list[str] = []
    if evidence.review_status != "completed":
        missing.append("human_review_incomplete")
    if evidence.verification_status != "verified":
        missing.append("human_identity_not_verified")
    if not evidence.target_identity_verified:
        missing.append("target_identity_not_verified")
    if policy.mode != DEFAULT_REFERENCE_ADMISSION_MODE:
        missing.append("policy_does_not_enable_provisional_support")
    else:
        for passed, reason in (
            (evidence.policy_permits_provisional_support, "provisional_support_not_permitted"),
            (evidence.source == "gbif", "provider_source_not_gbif"),
            (evidence.provider_assertion_passed, "provider_assertion_failed"),
            (evidence.automated_admission_gates_passed, "automated_admission_gates_failed"),
            (evidence.route_compatible, "reference_route_incompatible"),
            (evidence.canonical_media, "noncanonical_media"),
            (evidence.deduplication_completed, "deduplication_incomplete"),
            (evidence.provisional_support_declared, "provisional_support_not_declared"),
        ):
            if not passed:
                missing.append(reason)
    return _excluded(*missing)


def _excluded(*reasons: str) -> SupportAdmissionEvaluation:
    canonical = tuple(sorted(set(reasons))) or ("no_accepted_support_path",)
    return SupportAdmissionEvaluation(
        eligible=False,
        admission_status="excluded",
        evidence_path="none",
        provisional=False,
        reasons=canonical,
    )


__all__ = [
    "SUPPORT_EVIDENCE_PATHS",
    "SupportAdmissionEvaluation",
    "SupportAdmissionEvidence",
    "evaluate_support_admission",
]
