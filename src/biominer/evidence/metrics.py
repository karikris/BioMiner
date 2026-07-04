from __future__ import annotations

from typing import Any

import polars as pl


REVIEW_QUEUE_SCHEMA: dict[str, pl.DataType] = {
    "source": pl.String,
    "flickr_photo_id": pl.String,
    "review_bucket": pl.String,
    "review_priority": pl.Int64,
    "review_reason": pl.String,
    "best_detection_id": pl.String,
    "detection_count": pl.Int64,
    "best_object_occurrence_bin": pl.String,
    "best_object_species_top1": pl.String,
    "best_object_score": pl.Float64,
    "all_detection_ids": pl.List(pl.String),
    "all_candidate_species": pl.List(pl.String),
}
REVIEW_QUEUE_BUCKETS = {"bronze", "in_review"}


def evidence_count_metrics(joined: pl.DataFrame, photo_summary: pl.DataFrame | None = None) -> dict[str, Any]:
    """Return small deterministic metrics for joined object evidence outputs."""

    metrics: dict[str, Any] = {
        "object_evidence_rows": joined.height,
        "photo_summary_rows": photo_summary.height if photo_summary is not None else None,
        "object_occurrence_bin_counts": _value_counts(joined, "occurrence_bin"),
    }
    if photo_summary is not None:
        metrics["photo_occurrence_bin_counts"] = _value_counts(photo_summary, "photo_occurrence_bin")
    return metrics


def build_review_queue(photo_summary: pl.DataFrame) -> pl.DataFrame:
    """Return review-ready photo rows from production photo evidence summaries."""

    if photo_summary.is_empty() or "photo_occurrence_bin" not in photo_summary.columns:
        return pl.DataFrame(schema=REVIEW_QUEUE_SCHEMA)
    rows: list[dict[str, Any]] = []
    for row in photo_summary.to_dicts():
        bucket = str(row.get("photo_occurrence_bin") or "")
        if bucket not in REVIEW_QUEUE_BUCKETS:
            continue
        rows.append(
            {
                "source": str(row.get("source") or ""),
                "flickr_photo_id": str(row.get("flickr_photo_id") or ""),
                "review_bucket": bucket,
                "review_priority": _review_priority(bucket),
                "review_reason": str(row.get("photo_bin_reason") or bucket),
                "best_detection_id": str(row.get("best_detection_id") or ""),
                "detection_count": int(row.get("detection_count") or 0),
                "best_object_occurrence_bin": str(row.get("best_object_occurrence_bin") or ""),
                "best_object_species_top1": str(row.get("best_object_species_top1") or ""),
                "best_object_score": row.get("best_object_score"),
                "all_detection_ids": _string_list(row.get("all_detection_ids")),
                "all_candidate_species": _string_list(row.get("all_candidate_species")),
            }
        )
    if not rows:
        return pl.DataFrame(schema=REVIEW_QUEUE_SCHEMA)
    return pl.DataFrame(rows, schema=REVIEW_QUEUE_SCHEMA).sort(["review_priority", "source", "flickr_photo_id"])


def _value_counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    if frame.is_empty() or column not in frame.columns:
        return {}
    counts = frame.group_by(column).len(name="count").sort(column).to_dicts()
    return {str(row[column] or ""): int(row["count"]) for row in counts}


def _review_priority(bucket: str) -> int:
    return 10 if bucket == "in_review" else 20


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if isinstance(value, tuple):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if str(value) else []


__all__ = ["REVIEW_QUEUE_SCHEMA", "build_review_queue", "evidence_count_metrics"]
