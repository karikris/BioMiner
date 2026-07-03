from __future__ import annotations

from typing import Any, Iterable

import polars as pl

from biominer.filter.rules import classify_evidence_frame, classify_evidence_row, classify_evidence_rows, review_reasons_for_evidence


def bucket_evidence_frame(evidence: pl.DataFrame) -> pl.DataFrame:
    """Classify evidence rows through the production evidence package boundary."""

    return classify_evidence_frame(evidence)


def bucket_evidence_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return classify_evidence_rows(rows)


__all__ = [
    "bucket_evidence_frame",
    "bucket_evidence_rows",
    "classify_evidence_frame",
    "classify_evidence_row",
    "classify_evidence_rows",
    "review_reasons_for_evidence",
]
