from __future__ import annotations


REVIEW_QUEUE_BUCKETS = {"bronze", "in_review"}
ACTIONABLE_IN_REVIEW_REASONS = {
    "ambiguous_species_margin",
    "species_conflict",
    "taxonomy_inconsistent",
    "missing_event_date",
    "missing_geo",
    "unknown_life_stage",
    "low_confidence",
    "text_vision_conflict",
    "detected_object_without_bioclip_score",
}


def comment_review_is_actionable(*, bucket: str, reason: str) -> bool:
    if bucket == "bronze":
        return True
    if bucket == "in_review":
        return reason in ACTIONABLE_IN_REVIEW_REASONS
    return False


__all__ = ["ACTIONABLE_IN_REVIEW_REASONS", "REVIEW_QUEUE_BUCKETS", "comment_review_is_actionable"]
