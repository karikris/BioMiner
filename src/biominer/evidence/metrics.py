from __future__ import annotations

from typing import Any

import polars as pl


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


def _value_counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    if frame.is_empty() or column not in frame.columns:
        return {}
    counts = frame.group_by(column).len(name="count").sort(column).to_dicts()
    return {str(row[column] or ""): int(row["count"]) for row in counts}


__all__ = ["evidence_count_metrics"]
