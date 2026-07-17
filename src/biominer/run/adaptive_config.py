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


__all__ = [
    "ADAPTIVE_GBIF_FAST_START",
    "AdaptiveReferenceSettings",
    "DEFAULT_INITIAL_SCORING_MODE",
    "DEFAULT_REFERENCE_ADMISSION_MODE",
    "DEFAULT_REFERENCE_SOURCE",
    "HUMAN_VERIFIED_FLAGGED_ONLY",
    "HUMAN_VERIFIED_STRICT",
    "REFERENCE_ADMISSION_MODES",
]
