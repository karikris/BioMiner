"""Fail-closed final Parquet export for verified Flickr occurrences."""

from __future__ import annotations

from pathlib import Path

import polars as pl


REQUIRED_FLICKR_EXPORT_COLUMNS = frozenset(
    {
        "source_record_id",
        "human_review_decision",
        "source_image_sha256",
        "review_source_image_sha256",
        "conflict_status",
        "occurrence_claim_supported",
        "eligible_for_final_occurrence_dataset",
        "release_state",
    }
)


class FlickrExportValidationError(ValueError):
    def __init__(self, blocked_records: dict[str, tuple[str, ...]]) -> None:
        self.blocked_records = blocked_records
        summary = "; ".join(
            f"{record_id}={list(reasons)}"
            for record_id, reasons in sorted(blocked_records.items())
        )
        super().__init__(f"Flickr export rejected (fail-closed): {summary}")


def validate_verified_flickr_export(records: pl.DataFrame) -> pl.DataFrame:
    """Return the unchanged frame only when every row is release eligible."""

    missing = sorted(REQUIRED_FLICKR_EXPORT_COLUMNS - set(records.columns))
    if missing:
        raise ValueError(f"verified Flickr export missing columns: {missing}")
    if records.is_empty():
        raise ValueError("verified Flickr export must not be empty")
    if (
        records["source_record_id"].null_count()
        or records["source_record_id"].n_unique() != records.height
    ):
        raise ValueError("source_record_id must be nonnull and unique")

    blocked: dict[str, tuple[str, ...]] = {}
    for row in records.iter_rows(named=True):
        reasons = _row_blocking_reasons(row)
        if reasons:
            blocked[str(row["source_record_id"])] = reasons
    if blocked:
        raise FlickrExportValidationError(blocked)
    return records


def write_verified_flickr_export(
    records: pl.DataFrame,
    output_path: str | Path,
) -> Path:
    """Validate the complete batch before atomically publishing Parquet."""

    verified = validate_verified_flickr_export(records)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    verified.write_parquet(temporary)
    temporary.replace(destination)
    return destination


def _row_blocking_reasons(row: dict[str, object]) -> tuple[str, ...]:
    reasons: list[str] = []
    decision = _normalize_label(row.get("human_review_decision"))
    if decision in {"", "unreviewed"}:
        reasons.append("unreviewed")
    elif decision == "skip":
        reasons.append("skip")
    elif decision in {"can't view", "cant view"}:
        reasons.append("cant_view")
    elif decision == "uncertain":
        reasons.append("uncertain_label")
    elif decision != "include":
        reasons.append("review_not_include")
    conflict_status = _normalize_label(row.get("conflict_status"))
    if conflict_status not in {"resolved", "not_required"}:
        reasons.append("unresolved_conflict")
    if row.get("review_source_image_sha256") != row.get("source_image_sha256"):
        reasons.append("stale_source_hash")
    if row.get("occurrence_claim_supported") is not True:
        reasons.append("unsupported_occurrence_claim")
    if (
        row.get("eligible_for_final_occurrence_dataset") is not True
        or _normalize_label(row.get("release_state")) != "eligible"
    ):
        reasons.append("release_not_eligible")
    return tuple(reasons)


def _normalize_label(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().casefold().replace("’", "'")


__all__ = [
    "FlickrExportValidationError",
    "REQUIRED_FLICKR_EXPORT_COLUMNS",
    "validate_verified_flickr_export",
    "write_verified_flickr_export",
]
