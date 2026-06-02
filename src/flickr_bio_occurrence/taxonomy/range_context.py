from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RangeContext:
    range_context_status: str
    range_context_source: str
    range_context_notes: str
    range_extension_candidate: bool
    known_range_match: bool
    known_range_distance_km: float | None
    analysis_outlier_candidate: bool
    review_status: str


def annotate_range_context(*, known_range_match: bool, known_range_distance_km: float | None) -> RangeContext:
    if known_range_match:
        status = "inside_known_range"
        range_extension = False
        review_status = "machine_suggested"
        notes = "Location matches configured known range context."
    else:
        status = "range_extension_candidate"
        range_extension = True
        review_status = "needs_review"
        notes = "Outside known range context; retained as a discovery or range-extension candidate."
    return RangeContext(
        range_context_status=status,
        range_context_source="configured_seed_context",
        range_context_notes=notes,
        range_extension_candidate=range_extension,
        known_range_match=known_range_match,
        known_range_distance_km=known_range_distance_km,
        analysis_outlier_candidate=not known_range_match,
        review_status=review_status,
    )
