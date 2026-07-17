"""Typed defaults for adaptive GBIF reference admission."""

from __future__ import annotations

from dataclasses import dataclass


HUMAN_VERIFIED_STRICT = "human_verified_strict"
ADAPTIVE_GBIF_FAST_START = "adaptive_gbif_fast_start"
HUMAN_VERIFIED_FLAGGED_ONLY = "human_verified_flagged_only"

REFERENCE_ADMISSION_MODES: tuple[str, ...] = (
    HUMAN_VERIFIED_STRICT,
    ADAPTIVE_GBIF_FAST_START,
    HUMAN_VERIFIED_FLAGGED_ONLY,
)
DEFAULT_REFERENCE_ADMISSION_MODE = ADAPTIVE_GBIF_FAST_START
DEFAULT_REFERENCE_SOURCE = "gbif"
DEFAULT_INITIAL_SCORING_MODE = "provisional_reference_ranking"


@dataclass(frozen=True, slots=True)
class AdaptiveReferenceSettings:
    """Production reference defaults recorded in every run manifest."""

    reference_admission_mode: str = DEFAULT_REFERENCE_ADMISSION_MODE
    reference_source: str = DEFAULT_REFERENCE_SOURCE
    initial_scoring_mode: str = DEFAULT_INITIAL_SCORING_MODE
    flickr_release_requires_human_review: bool = True
    statistical_reference_audit: bool = True

    def __post_init__(self) -> None:
        admission_mode = str(self.reference_admission_mode).strip().casefold()
        if admission_mode not in REFERENCE_ADMISSION_MODES:
            raise ValueError(
                "unsupported reference_admission_mode: "
                f"{self.reference_admission_mode!r}"
            )
        source = str(self.reference_source).strip().casefold()
        if not source:
            raise ValueError("reference_source must be nonblank")
        scoring_mode = str(self.initial_scoring_mode).strip().casefold()
        if not scoring_mode:
            raise ValueError("initial_scoring_mode must be nonblank")
        object.__setattr__(self, "reference_admission_mode", admission_mode)
        object.__setattr__(self, "reference_source", source)
        object.__setattr__(self, "initial_scoring_mode", scoring_mode)


@dataclass(frozen=True, slots=True)
class AdaptiveReferenceValidationContext:
    """Evidence-use claims that constrain otherwise valid reference settings."""

    strict_readiness_claim: bool = False
    reference_split_uses: tuple[str, ...] = ()
    final_flickr_export_requested: bool = True
    calibrator_available: bool = False
    supported_unreviewed_sources: tuple[str, ...] = (DEFAULT_REFERENCE_SOURCE,)

    def __post_init__(self) -> None:
        split_uses = _canonical_values(
            self.reference_split_uses,
            field="reference_split_uses",
        )
        supported_sources = _canonical_values(
            self.supported_unreviewed_sources,
            field="supported_unreviewed_sources",
        )
        object.__setattr__(self, "reference_split_uses", split_uses)
        object.__setattr__(self, "supported_unreviewed_sources", supported_sources)


def validate_adaptive_reference_settings(
    settings: AdaptiveReferenceSettings,
    *,
    context: AdaptiveReferenceValidationContext | None = None,
) -> AdaptiveReferenceSettings:
    """Reject unsafe cross-field combinations and return validated settings."""

    usage = context or AdaptiveReferenceValidationContext()
    unreviewed_references = (
        settings.reference_admission_mode != HUMAN_VERIFIED_STRICT
    )
    provisional_scoring = (
        settings.initial_scoring_mode == DEFAULT_INITIAL_SCORING_MODE
    )
    restricted_splits = {"calibration", "final_test"}.intersection(
        usage.reference_split_uses
    )

    if provisional_scoring and usage.strict_readiness_claim:
        raise ValueError(
            "provisional references cannot claim strict reference readiness"
        )
    if unreviewed_references and restricted_splits:
        raise ValueError(
            "unreviewed references cannot enter calibration or final_test splits: "
            f"{sorted(restricted_splits)}"
        )
    if (
        usage.final_flickr_export_requested
        and not settings.flickr_release_requires_human_review
    ):
        raise ValueError("final Flickr export requires human review")
    if (
        settings.initial_scoring_mode == "calibrated_probability"
        and not usage.calibrator_available
    ):
        raise ValueError("calibrated probability requires a valid calibrator")
    if (
        settings.reference_admission_mode == ADAPTIVE_GBIF_FAST_START
        and not settings.statistical_reference_audit
    ):
        raise ValueError(
            "adaptive_gbif_fast_start requires a statistical reference audit policy"
        )
    if (
        unreviewed_references
        and settings.reference_source not in usage.supported_unreviewed_sources
    ):
        raise ValueError(
            "unreviewed references are unsupported for source: "
            f"{settings.reference_source}"
        )
    return settings


def _canonical_values(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field} must be a tuple")
    normalized = tuple(sorted({str(value).strip().casefold() for value in values}))
    if any(not value for value in normalized):
        raise ValueError(f"{field} contains blank values")
    return normalized


__all__ = [
    "ADAPTIVE_GBIF_FAST_START",
    "AdaptiveReferenceValidationContext",
    "AdaptiveReferenceSettings",
    "DEFAULT_INITIAL_SCORING_MODE",
    "DEFAULT_REFERENCE_ADMISSION_MODE",
    "DEFAULT_REFERENCE_SOURCE",
    "HUMAN_VERIFIED_FLAGGED_ONLY",
    "HUMAN_VERIFIED_STRICT",
    "REFERENCE_ADMISSION_MODES",
    "validate_adaptive_reference_settings",
]
