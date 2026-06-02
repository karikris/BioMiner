from __future__ import annotations


def review_status_for_candidate(*, species_agreement_status: str, range_extension_candidate: bool) -> str:
    if species_agreement_status == "text_vision_conflict":
        return "needs_review"
    if range_extension_candidate:
        return "needs_review"
    if species_agreement_status in {"exact_species_agreement", "same_genus_agreement"}:
        return "machine_suggested"
    return "needs_review"
