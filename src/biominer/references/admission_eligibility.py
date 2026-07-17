"""Pure GBIF provider-assertion eligibility evaluation.

This module decides whether one media candidate may enter a provisional reference
bank.  It consumes evidence produced by acquisition, deduplication, routing, and
selection; it does not infer species identity and does not mutate source rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.admission import (
    DEFAULT_REFERENCE_ADMISSION_MODE,
    ReferenceAdmissionPolicy,
)
from biominer.references.schemas import (
    DECODE_STATUSES,
    DOWNLOAD_STATUSES,
    DUPLICATE_RESOLUTION_STATUSES,
    LICENCE_POLICY_STATUSES,
    REFERENCE_MEDIA_RASTER_CONTENT_TYPES,
    REFERENCE_ROUTES,
    REFERENCE_VISUAL_DOMAINS,
    TAXON_RECONCILIATION_STATUSES,
)


GBIF_ELIGIBILITY_SCHEMA_VERSION = "gbif-provider-eligibility-v1.0.0"
GBIF_PROVIDER_ASSERTION_IDENTITY_BASIS = "gbif_provider_asserted"
PROTOTYPE_SCOPES = frozenset({"global", "local", "regional"})

ELIGIBILITY_GATE_IDS = (
    "01_provider_source",
    "02_taxon_reconciliation",
    "03_certain_taxon_match",
    "04_occurrence_present",
    "05_non_fossil",
    "06_supported_still_image",
    "07_download_complete",
    "08_content_type_valid",
    "09_decode_succeeded",
    "10_image_sha256_present",
    "11_licence_accepted",
    "12_attribution_complete",
    "13_duplicate_processing_complete",
    "14_canonical_media",
    "15_duplicate_conflicts_resolved",
    "16_provider_identity_match",
    "17_observation_independence",
    "18_yoloe_routing_complete",
    "19_route_compatible",
    "20_biological_visual_domain",
    "21_unambiguous_visual_domain",
    "22_subject_area_threshold",
    "23_full_frame_input_available",
    "24_geography_scope_eligible",
)

_SHA256_PATTERN = re.compile(r"(?:sha256:)?[0-9a-f]{64}\Z")
_ARTIFACT_DOMAINS = frozenset({"artwork", "logo", "tattoo", "unsuitable"})
_AMBIGUOUS_DOMAINS = frozenset(
    {"ambiguous", "partial_wing", "dead_or_damaged_specimen"}
)


class EligibilityDecision(str, Enum):
    """Terminal admission decision, ordered by fail-closed precedence."""

    ADMITTED = "admitted"
    EXCLUDED = "excluded"
    REVIEW_REQUIRED = "review_required"


class GateDisposition(str, Enum):
    """Outcome of one non-negotiable admission gate."""

    PASSED = "passed"
    EXCLUDED = "excluded"
    REVIEW_REQUIRED = "review_required"


@dataclass(frozen=True, slots=True)
class GBIFEligibilityEvidence:
    """Normalized evidence needed to evaluate all provisional-admission gates."""

    source: str
    taxon_reconciliation_status: str
    resolves_to_candidate_accepted_taxon_key: bool
    uncertain_taxon_match: bool
    occurrence_absent: bool
    fossil: bool
    media_type: str
    download_status: str
    content_type: str | None
    decode_status: str
    decoded_width: int | None
    decoded_height: int | None
    image_sha256: str | None
    licence_policy_status: str
    creator: str | None
    source_url: str | None
    attribution: str | None
    duplicate_processing_completed: bool
    canonical_media: bool
    duplicate_resolution_status: str | None
    duplicate_conflict_targeted_review: bool
    provider_identity_matches_candidate_taxon: bool
    independence_processing_completed: bool
    selected_images_from_observation: int
    observer_identity_available: bool
    observer_image_ordinal_before_reuse: int
    observer_reuse_justified: bool
    near_identical_view: bool
    yoloe_routing_completed: bool
    yoloe_route: str | None
    requested_bank_route: str
    visual_domain: str
    subject_present: bool | None
    ambiguous_domain_targeted_review: bool
    subject_area_ratio: float | None
    full_frame_input_generation_succeeded: bool
    prototype_scope: str
    usable_geography: bool

    def __post_init__(self) -> None:
        for field in (
            "source",
            "taxon_reconciliation_status",
            "media_type",
            "download_status",
            "decode_status",
            "licence_policy_status",
            "requested_bank_route",
            "visual_domain",
            "prototype_scope",
        ):
            object.__setattr__(
                self, field, _normalized_text(getattr(self, field), field)
            )
        for field in (
            "content_type",
            "image_sha256",
            "duplicate_resolution_status",
            "yoloe_route",
        ):
            object.__setattr__(
                self, field, _optional_normalized_text(getattr(self, field))
            )
        for field in ("creator", "source_url", "attribution"):
            object.__setattr__(self, field, _optional_text(getattr(self, field)))

        if self.taxon_reconciliation_status not in TAXON_RECONCILIATION_STATUSES:
            raise ValueError("unsupported taxon_reconciliation_status")
        if self.download_status not in DOWNLOAD_STATUSES:
            raise ValueError("unsupported download_status")
        if self.decode_status not in DECODE_STATUSES:
            raise ValueError("unsupported decode_status")
        if self.licence_policy_status not in LICENCE_POLICY_STATUSES:
            raise ValueError("unsupported licence_policy_status")
        if (
            self.duplicate_resolution_status is not None
            and self.duplicate_resolution_status not in DUPLICATE_RESOLUTION_STATUSES
        ):
            raise ValueError("unsupported duplicate_resolution_status")
        if self.requested_bank_route not in REFERENCE_ROUTES:
            raise ValueError("unsupported requested_bank_route")
        if self.visual_domain not in REFERENCE_VISUAL_DOMAINS:
            raise ValueError("unsupported visual_domain")
        if self.prototype_scope not in PROTOTYPE_SCOPES:
            raise ValueError("unsupported prototype_scope")

        for field in ("decoded_width", "decoded_height"):
            value = getattr(self, field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{field} must be a positive integer or None")
        for field in (
            "selected_images_from_observation",
            "observer_image_ordinal_before_reuse",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.subject_area_ratio is not None:
            value = self.subject_area_ratio
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("subject_area_ratio must be finite and in [0, 1]")
            object.__setattr__(self, "subject_area_ratio", float(value))

        for field in (
            "resolves_to_candidate_accepted_taxon_key",
            "uncertain_taxon_match",
            "occurrence_absent",
            "fossil",
            "duplicate_processing_completed",
            "canonical_media",
            "duplicate_conflict_targeted_review",
            "provider_identity_matches_candidate_taxon",
            "independence_processing_completed",
            "observer_identity_available",
            "observer_reuse_justified",
            "near_identical_view",
            "yoloe_routing_completed",
            "ambiguous_domain_targeted_review",
            "full_frame_input_generation_succeeded",
            "usable_geography",
        ):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be Boolean")
        if self.subject_present is not None and not isinstance(
            self.subject_present, bool
        ):
            raise TypeError("subject_present must be Boolean or None")

    def identity_payload(self) -> dict[str, object]:
        """Return a canonical projection used for deterministic result identity."""

        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class EligibilityGateResult:
    """Auditable result for one admission gate."""

    gate_id: str
    disposition: GateDisposition
    reason_code: str

    def to_dict(self) -> dict[str, str]:
        return {
            "gate_id": self.gate_id,
            "disposition": self.disposition.value,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True)
class GBIFEligibilityResult:
    """Complete admission result with all 24 gate outcomes."""

    schema_version: str
    decision: EligibilityDecision
    identity_basis: str
    human_verified: bool
    geographic_prototype_eligible: bool
    policy_fingerprint: str
    evidence_fingerprint: str
    gate_results: tuple[EligibilityGateResult, ...]

    def __post_init__(self) -> None:
        if self.schema_version != GBIF_ELIGIBILITY_SCHEMA_VERSION:
            raise ValueError("unsupported GBIF eligibility schema version")
        if self.identity_basis != GBIF_PROVIDER_ASSERTION_IDENTITY_BASIS:
            raise ValueError(
                "GBIF eligibility identity_basis must be provider asserted"
            )
        if self.human_verified:
            raise ValueError("GBIF provider assertion cannot be human verified")
        gate_ids = tuple(result.gate_id for result in self.gate_results)
        if gate_ids != ELIGIBILITY_GATE_IDS:
            raise ValueError(
                "eligibility result must contain all gates in canonical order"
            )
        if self.decision is not _decision_from_gates(self.gate_results):
            raise ValueError("eligibility decision disagrees with gate results")

    @property
    def reason_codes(self) -> tuple[str, ...]:
        """Return every non-passing reason in canonical gate order."""

        return tuple(
            result.reason_code
            for result in self.gate_results
            if result.disposition is not GateDisposition.PASSED
        )

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision": self.decision.value,
            "identity_basis": self.identity_basis,
            "human_verified": self.human_verified,
            "geographic_prototype_eligible": self.geographic_prototype_eligible,
            "policy_fingerprint": self.policy_fingerprint,
            "evidence_fingerprint": self.evidence_fingerprint,
            "gate_results": [result.to_dict() for result in self.gate_results],
        }


def evaluate_gbif_provisional_eligibility(
    evidence: GBIFEligibilityEvidence,
    policy: ReferenceAdmissionPolicy,
) -> GBIFEligibilityResult:
    """Evaluate all gates without changing evidence, policy, or external state."""

    if policy.mode != DEFAULT_REFERENCE_ADMISSION_MODE:
        raise ValueError("GBIF provisional eligibility requires the adaptive policy")

    gate_results = (
        _provider_source_gate(evidence, policy),
        _taxon_reconciliation_gate(evidence, policy),
        _boolean_gate(
            "03_certain_taxon_match",
            not evidence.uncertain_taxon_match,
            "taxon_match_is_certain",
            "uncertain_taxon_match",
        ),
        _boolean_gate(
            "04_occurrence_present",
            not evidence.occurrence_absent,
            "occurrence_is_present",
            "occurrence_is_absent",
        ),
        _boolean_gate(
            "05_non_fossil",
            not evidence.fossil,
            "record_is_not_fossil",
            "fossil_record",
        ),
        _still_image_gate(evidence),
        _download_gate(evidence),
        _content_type_gate(evidence),
        _decode_gate(evidence, policy),
        _sha256_gate(evidence),
        _licence_gate(evidence, policy),
        _attribution_gate(evidence),
        _duplicate_processing_gate(evidence),
        _canonical_media_gate(evidence, policy),
        _duplicate_resolution_gate(evidence),
        _boolean_gate(
            "16_provider_identity_match",
            evidence.provider_identity_matches_candidate_taxon,
            "provider_identity_matches_candidate_taxon",
            "provider_identity_mismatch",
        ),
        _independence_gate(evidence, policy),
        _routing_complete_gate(evidence, policy),
        _route_compatibility_gate(evidence, policy),
        _biological_domain_gate(evidence),
        _ambiguous_domain_gate(evidence),
        _subject_area_gate(evidence, policy),
        _full_frame_gate(evidence),
        _geography_gate(evidence),
    )
    return GBIFEligibilityResult(
        schema_version=GBIF_ELIGIBILITY_SCHEMA_VERSION,
        decision=_decision_from_gates(gate_results),
        identity_basis=GBIF_PROVIDER_ASSERTION_IDENTITY_BASIS,
        human_verified=False,
        geographic_prototype_eligible=evidence.usable_geography,
        policy_fingerprint=policy.fingerprint,
        evidence_fingerprint=canonical_semantic_fingerprint(
            evidence.identity_payload()
        ),
        gate_results=gate_results,
    )


def _provider_source_gate(
    evidence: GBIFEligibilityEvidence, policy: ReferenceAdmissionPolicy
) -> EligibilityGateResult:
    return _boolean_gate(
        "01_provider_source",
        evidence.source in policy.allowed_provider_sources,
        "provider_source_allowed",
        "provider_source_not_allowed",
    )


def _taxon_reconciliation_gate(
    evidence: GBIFEligibilityEvidence, policy: ReferenceAdmissionPolicy
) -> EligibilityGateResult:
    passed = (
        evidence.taxon_reconciliation_status
        in policy.accepted_taxon_reconciliation_statuses
        and evidence.resolves_to_candidate_accepted_taxon_key
    )
    return _boolean_gate(
        "02_taxon_reconciliation",
        passed,
        "provider_taxon_resolves_to_candidate_accepted_key",
        "provider_taxon_not_reconciled_to_candidate_accepted_key",
    )


def _still_image_gate(evidence: GBIFEligibilityEvidence) -> EligibilityGateResult:
    media_type = evidence.media_type.replace("_", "").replace(" ", "")
    return _boolean_gate(
        "06_supported_still_image",
        media_type in {"stillimage", "image"},
        "supported_still_image",
        "unsupported_media_type",
    )


def _download_gate(evidence: GBIFEligibilityEvidence) -> EligibilityGateResult:
    if evidence.download_status == "complete":
        return _passed("07_download_complete", "download_complete")
    if evidence.download_status == "pending":
        return _review("07_download_complete", "download_not_completed")
    return _excluded("07_download_complete", "download_unavailable")


def _content_type_gate(evidence: GBIFEligibilityEvidence) -> EligibilityGateResult:
    return _boolean_gate(
        "08_content_type_valid",
        evidence.content_type in REFERENCE_MEDIA_RASTER_CONTENT_TYPES,
        "raster_content_type_valid",
        "raster_content_type_invalid_or_missing",
    )


def _decode_gate(
    evidence: GBIFEligibilityEvidence, policy: ReferenceAdmissionPolicy
) -> EligibilityGateResult:
    if evidence.decode_status != "valid":
        return _excluded("09_decode_succeeded", "image_decode_not_valid")
    if evidence.decoded_width is None or evidence.decoded_height is None:
        return _review("09_decode_succeeded", "decoded_dimensions_missing")
    if (
        evidence.decoded_width < policy.minimum_decoded_width
        or evidence.decoded_height < policy.minimum_decoded_height
    ):
        return _excluded("09_decode_succeeded", "decoded_dimensions_below_policy")
    return _passed("09_decode_succeeded", "image_decode_and_dimensions_valid")


def _sha256_gate(evidence: GBIFEligibilityEvidence) -> EligibilityGateResult:
    valid = evidence.image_sha256 is not None and bool(
        _SHA256_PATTERN.fullmatch(evidence.image_sha256)
    )
    return _boolean_gate(
        "10_image_sha256_present",
        valid,
        "image_sha256_valid",
        "image_sha256_missing_or_invalid",
    )


def _licence_gate(
    evidence: GBIFEligibilityEvidence, policy: ReferenceAdmissionPolicy
) -> EligibilityGateResult:
    return _boolean_gate(
        "11_licence_accepted",
        evidence.licence_policy_status in policy.accepted_licence_policy_statuses,
        "licence_accepted_for_configured_use",
        "licence_not_accepted_for_configured_use",
    )


def _attribution_gate(evidence: GBIFEligibilityEvidence) -> EligibilityGateResult:
    complete = all((evidence.creator, evidence.source_url, evidence.attribution))
    return _boolean_gate(
        "12_attribution_complete",
        complete,
        "creator_source_and_attribution_present",
        "creator_source_or_attribution_missing",
    )


def _duplicate_processing_gate(
    evidence: GBIFEligibilityEvidence,
) -> EligibilityGateResult:
    if not evidence.duplicate_processing_completed:
        return _review(
            "13_duplicate_processing_complete", "duplicate_processing_incomplete"
        )
    if evidence.duplicate_resolution_status is None:
        return _review(
            "13_duplicate_processing_complete", "duplicate_resolution_status_missing"
        )
    return _passed(
        "13_duplicate_processing_complete", "exact_and_perceptual_processing_complete"
    )


def _canonical_media_gate(
    evidence: GBIFEligibilityEvidence, policy: ReferenceAdmissionPolicy
) -> EligibilityGateResult:
    passed = evidence.canonical_media or not policy.require_canonical_media
    return _boolean_gate(
        "14_canonical_media", passed, "canonical_media", "noncanonical_media"
    )


def _duplicate_resolution_gate(
    evidence: GBIFEligibilityEvidence,
) -> EligibilityGateResult:
    status = evidence.duplicate_resolution_status
    if not evidence.duplicate_processing_completed or status is None:
        return _review(
            "15_duplicate_conflicts_resolved", "duplicate_resolution_pending"
        )
    if status == "resolved":
        return _passed("15_duplicate_conflicts_resolved", "duplicate_evidence_resolved")
    if status == "review_required" or (
        status == "conflict" and evidence.duplicate_conflict_targeted_review
    ):
        return _review(
            "15_duplicate_conflicts_resolved", "duplicate_conflict_targeted_review"
        )
    return _excluded(
        "15_duplicate_conflicts_resolved", "unresolved_duplicate_conflict"
    )


def _independence_gate(
    evidence: GBIFEligibilityEvidence, policy: ReferenceAdmissionPolicy
) -> EligibilityGateResult:
    if not evidence.independence_processing_completed:
        return _review(
            "17_observation_independence", "independence_processing_incomplete"
        )
    if not evidence.observer_identity_available:
        return _review("17_observation_independence", "observer_identity_missing")
    if evidence.near_identical_view:
        return _excluded("17_observation_independence", "near_identical_view_reuse")
    if (
        evidence.selected_images_from_observation
        > policy.maximum_images_per_observation
    ):
        return _excluded("17_observation_independence", "observation_quota_exceeded")
    if (
        evidence.observer_image_ordinal_before_reuse
        > policy.maximum_images_per_observer_before_reuse
        and not evidence.observer_reuse_justified
    ):
        return _excluded("17_observation_independence", "observer_reused_too_early")
    return _passed("17_observation_independence", "observation_independence_satisfied")


def _routing_complete_gate(
    evidence: GBIFEligibilityEvidence, policy: ReferenceAdmissionPolicy
) -> EligibilityGateResult:
    if not policy.require_yoloe_route:
        return _passed("18_yoloe_routing_complete", "routing_not_required_by_policy")
    if not evidence.yoloe_routing_completed or evidence.yoloe_route is None:
        return _review("18_yoloe_routing_complete", "yoloe_routing_incomplete")
    return _passed("18_yoloe_routing_complete", "yoloe_routing_complete")


def _route_compatibility_gate(
    evidence: GBIFEligibilityEvidence, policy: ReferenceAdmissionPolicy
) -> EligibilityGateResult:
    if not evidence.yoloe_routing_completed or evidence.yoloe_route is None:
        return _review("19_route_compatible", "route_compatibility_pending")
    if evidence.yoloe_route not in REFERENCE_ROUTES:
        return _excluded("19_route_compatible", "unsupported_yoloe_route")
    if evidence.yoloe_route != evidence.requested_bank_route:
        return _excluded(
            "19_route_compatible", "route_incompatible_with_requested_bank"
        )
    if evidence.yoloe_route not in policy.allowed_unreviewed_routes:
        return _review("19_route_compatible", "route_requires_human_review")
    return _passed("19_route_compatible", "route_compatible_with_requested_bank")


def _biological_domain_gate(evidence: GBIFEligibilityEvidence) -> EligibilityGateResult:
    if evidence.visual_domain in _ARTIFACT_DOMAINS:
        return _excluded("20_biological_visual_domain", "artifact_or_unsuitable_domain")
    if evidence.subject_present is False:
        return _excluded("20_biological_visual_domain", "no_organism_detected")
    if evidence.subject_present is None:
        return _review("20_biological_visual_domain", "subject_presence_unresolved")
    return _passed("20_biological_visual_domain", "biological_subject_present")


def _ambiguous_domain_gate(evidence: GBIFEligibilityEvidence) -> EligibilityGateResult:
    if evidence.visual_domain not in _AMBIGUOUS_DOMAINS:
        return _passed("21_unambiguous_visual_domain", "visual_domain_unambiguous")
    if evidence.ambiguous_domain_targeted_review:
        return _review(
            "21_unambiguous_visual_domain", "ambiguous_domain_targeted_review"
        )
    return _excluded("21_unambiguous_visual_domain", "ambiguous_domain_not_admissible")


def _subject_area_gate(
    evidence: GBIFEligibilityEvidence, policy: ReferenceAdmissionPolicy
) -> EligibilityGateResult:
    if evidence.subject_area_ratio is None:
        return _review("22_subject_area_threshold", "subject_area_not_measured")
    if evidence.subject_area_ratio < policy.minimum_subject_area_ratio:
        return _excluded("22_subject_area_threshold", "subject_area_below_policy")
    return _passed("22_subject_area_threshold", "subject_area_meets_policy")


def _full_frame_gate(evidence: GBIFEligibilityEvidence) -> EligibilityGateResult:
    if evidence.full_frame_input_generation_succeeded:
        return _passed("23_full_frame_input_available", "full_frame_input_available")
    return _review(
        "23_full_frame_input_available", "full_frame_input_generation_failed"
    )


def _geography_gate(evidence: GBIFEligibilityEvidence) -> EligibilityGateResult:
    if evidence.prototype_scope == "global":
        reason = (
            "global_scope_with_usable_geography"
            if evidence.usable_geography
            else "global_scope_allows_missing_geography"
        )
        return _passed("24_geography_scope_eligible", reason)
    if evidence.usable_geography:
        return _passed(
            "24_geography_scope_eligible", "geographic_scope_has_usable_geography"
        )
    return _excluded(
        "24_geography_scope_eligible", "geographic_scope_missing_usable_geography"
    )


def _boolean_gate(
    gate_id: str,
    passed: bool,
    passed_reason: str,
    excluded_reason: str,
) -> EligibilityGateResult:
    if passed:
        return _passed(gate_id, passed_reason)
    return _excluded(gate_id, excluded_reason)


def _passed(gate_id: str, reason: str) -> EligibilityGateResult:
    return EligibilityGateResult(gate_id, GateDisposition.PASSED, reason)


def _excluded(gate_id: str, reason: str) -> EligibilityGateResult:
    return EligibilityGateResult(gate_id, GateDisposition.EXCLUDED, reason)


def _review(gate_id: str, reason: str) -> EligibilityGateResult:
    return EligibilityGateResult(gate_id, GateDisposition.REVIEW_REQUIRED, reason)


def _decision_from_gates(
    gate_results: tuple[EligibilityGateResult, ...],
) -> EligibilityDecision:
    dispositions = {result.disposition for result in gate_results}
    if GateDisposition.EXCLUDED in dispositions:
        return EligibilityDecision.EXCLUDED
    if GateDisposition.REVIEW_REQUIRED in dispositions:
        return EligibilityDecision.REVIEW_REQUIRED
    return EligibilityDecision.ADMITTED


def _normalized_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonblank text")
    return value.strip().casefold()


def _optional_normalized_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional text evidence must be text or None")
    normalized = value.strip()
    return normalized.casefold() if normalized else None


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional text evidence must be text or None")
    normalized = value.strip()
    return normalized if normalized else None


__all__ = [
    "ELIGIBILITY_GATE_IDS",
    "GBIF_ELIGIBILITY_SCHEMA_VERSION",
    "GBIF_PROVIDER_ASSERTION_IDENTITY_BASIS",
    "EligibilityDecision",
    "EligibilityGateResult",
    "GBIFEligibilityEvidence",
    "GBIFEligibilityResult",
    "GateDisposition",
    "evaluate_gbif_provisional_eligibility",
]
