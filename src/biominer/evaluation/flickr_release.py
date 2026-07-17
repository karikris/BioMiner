"""Fail-closed release policy for human-verified Flickr records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


DECISIVE_REVIEW_DECISIONS = frozenset({"include", "exclude"})


class FlickrReleaseState(StrEnum):
    ELIGIBLE = "eligible"
    EXCLUDED = "excluded"


class FlickrReleaseReason(StrEnum):
    HUMAN_REVIEW_MISSING = "human_review_missing"
    REVIEW_NOT_DECISIVE = "review_not_decisive"
    REVIEW_EXCLUDES_RECORD = "review_excludes_record"
    REVIEW_SOURCE_HASH_MISMATCH = "review_source_hash_mismatch"
    DUPLICATE_GROUP_UNRESOLVED = "duplicate_group_unresolved"
    TARGET_IDENTITY_UNSUPPORTED = "target_identity_unsupported"
    VISUAL_DOMAIN_UNSUITABLE = "visual_domain_unsuitable"
    LIFE_STAGE_UNSUITABLE = "life_stage_unsuitable"
    COORDINATE_REQUIREMENTS_FAILED = "coordinate_requirements_failed"
    DATE_REQUIREMENTS_FAILED = "date_requirements_failed"
    SECOND_REVIEW_INCOMPLETE = "second_review_incomplete"
    ADJUDICATION_INCOMPLETE = "adjudication_incomplete"
    RELEASE_POLICY_DENIED = "release_policy_denied"


@dataclass(frozen=True, slots=True)
class FlickrReleaseEvidence:
    source_record_id: str
    source_image_sha256: str
    review_decision: str | None = None
    review_source_image_sha256: str | None = None
    duplicate_group_resolved: bool = False
    target_identity_supported: bool = False
    visual_domain_suitable: bool = False
    life_stage_suitable: bool = False
    coordinate_requirements_pass: bool = False
    date_requirements_pass: bool = False
    second_review_required: bool = False
    second_review_complete: bool = False
    adjudication_required: bool = False
    adjudication_complete: bool = False
    release_policy_permits: bool = False

    def __post_init__(self) -> None:
        record_id = str(self.source_record_id).strip()
        if not record_id:
            raise ValueError("source_record_id must be nonblank")
        _validate_sha256(self.source_image_sha256, field="source_image_sha256")
        decision = (
            str(self.review_decision).strip().casefold()
            if self.review_decision is not None
            else None
        )
        if decision not in {None, "include", "exclude", "uncertain"}:
            raise ValueError(f"unsupported review_decision: {self.review_decision!r}")
        if self.review_source_image_sha256 is not None:
            _validate_sha256(
                self.review_source_image_sha256,
                field="review_source_image_sha256",
            )
        object.__setattr__(self, "source_record_id", record_id)
        object.__setattr__(self, "review_decision", decision)


@dataclass(frozen=True, slots=True)
class FlickrReleaseDecision:
    source_record_id: str
    state: FlickrReleaseState
    eligible_for_final_occurrence_dataset: bool
    reasons: tuple[FlickrReleaseReason, ...]
    source_image_sha256: str
    review_source_image_sha256: str | None


def decide_flickr_release(evidence: FlickrReleaseEvidence) -> FlickrReleaseDecision:
    """Apply every mandatory release prerequisite without score-based shortcuts."""

    reasons: list[FlickrReleaseReason] = []
    if evidence.review_decision is None:
        reasons.append(FlickrReleaseReason.HUMAN_REVIEW_MISSING)
    elif evidence.review_decision not in DECISIVE_REVIEW_DECISIONS:
        reasons.append(FlickrReleaseReason.REVIEW_NOT_DECISIVE)
    elif evidence.review_decision == "exclude":
        reasons.append(FlickrReleaseReason.REVIEW_EXCLUDES_RECORD)
    if evidence.review_source_image_sha256 != evidence.source_image_sha256:
        reasons.append(FlickrReleaseReason.REVIEW_SOURCE_HASH_MISMATCH)
    if not evidence.duplicate_group_resolved:
        reasons.append(FlickrReleaseReason.DUPLICATE_GROUP_UNRESOLVED)
    if not evidence.target_identity_supported:
        reasons.append(FlickrReleaseReason.TARGET_IDENTITY_UNSUPPORTED)
    if not evidence.visual_domain_suitable:
        reasons.append(FlickrReleaseReason.VISUAL_DOMAIN_UNSUITABLE)
    if not evidence.life_stage_suitable:
        reasons.append(FlickrReleaseReason.LIFE_STAGE_UNSUITABLE)
    if not evidence.coordinate_requirements_pass:
        reasons.append(FlickrReleaseReason.COORDINATE_REQUIREMENTS_FAILED)
    if not evidence.date_requirements_pass:
        reasons.append(FlickrReleaseReason.DATE_REQUIREMENTS_FAILED)
    if evidence.second_review_required and not evidence.second_review_complete:
        reasons.append(FlickrReleaseReason.SECOND_REVIEW_INCOMPLETE)
    if evidence.adjudication_required and not evidence.adjudication_complete:
        reasons.append(FlickrReleaseReason.ADJUDICATION_INCOMPLETE)
    if not evidence.release_policy_permits:
        reasons.append(FlickrReleaseReason.RELEASE_POLICY_DENIED)

    eligible = not reasons
    return FlickrReleaseDecision(
        source_record_id=evidence.source_record_id,
        state=(FlickrReleaseState.ELIGIBLE if eligible else FlickrReleaseState.EXCLUDED),
        eligible_for_final_occurrence_dataset=eligible,
        reasons=tuple(reasons),
        source_image_sha256=evidence.source_image_sha256,
        review_source_image_sha256=evidence.review_source_image_sha256,
    )


def _validate_sha256(value: str, *, field: str) -> None:
    normalized = str(value).strip().casefold()
    if not normalized.startswith("sha256:") or len(normalized) != 71:
        raise ValueError(f"{field} must be a sha256: digest")
    try:
        int(normalized.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a sha256: digest") from exc


__all__ = [
    "DECISIVE_REVIEW_DECISIONS",
    "FlickrReleaseDecision",
    "FlickrReleaseEvidence",
    "FlickrReleaseReason",
    "FlickrReleaseState",
    "decide_flickr_release",
]
