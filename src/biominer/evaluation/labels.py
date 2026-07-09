from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


REVIEWED_LABEL_SCHEMA_VERSION = "reviewed-labels-v1"

LABEL_LEVELS = frozenset({"photo", "object", "family", "species", "negative"})
REVIEW_CONFIDENCE_VALUES = frozenset({"high", "medium", "low", "unknown"})

REVIEWED_LABEL_SCHEMA: dict[str, pl.DataType] = {
    "source": pl.String,
    "flickr_photo_id": pl.String,
    "detection_id": pl.String,
    "crop_hash": pl.String,
    "label_level": pl.String,
    "is_butterfly": pl.Boolean,
    "accepted_taxon_key": pl.String,
    "scientific_name": pl.String,
    "family_key": pl.String,
    "family": pl.String,
    "genus_key": pl.String,
    "genus": pl.String,
    "label_source": pl.String,
    "reviewer_id": pl.String,
    "reviewed_at": pl.String,
    "review_confidence": pl.String,
    "review_notes": pl.String,
}


def empty_reviewed_label_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=REVIEWED_LABEL_SCHEMA)


def validate_reviewed_label_frame(frame: pl.DataFrame) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    missing = [column for column in REVIEWED_LABEL_SCHEMA if column not in frame.columns]
    if missing:
        findings.append(
            _finding(
                "fatal",
                "missing_required_columns",
                "reviewed labels are missing required columns",
                {"columns": missing},
            )
        )
        return findings

    rows = _rows(frame)
    _append_invalid_label_level_findings(findings, rows)
    _append_invalid_confidence_findings(findings, rows)
    _append_missing_butterfly_taxonomy_findings(findings, rows)
    _append_duplicate_species_conflict_findings(findings, rows)
    _append_review_metadata_warnings(findings, rows)
    _append_photo_without_object_warning(findings, rows)
    return findings


def read_reviewed_labels(path: str | Path) -> pl.DataFrame:
    source = Path(path)
    suffix = source.suffix.casefold()
    if suffix == ".parquet":
        return pl.read_parquet(source)
    if suffix in {".jsonl", ".ndjson"}:
        return pl.read_ndjson(source)
    if suffix == ".json":
        return pl.read_json(source)
    raise ValueError(f"unsupported reviewed-label format: {suffix or '<none>'}")


def _rows(frame: pl.DataFrame) -> list[dict[str, Any]]:
    if frame.is_empty():
        return []
    return frame.select(list(REVIEWED_LABEL_SCHEMA)).to_dicts()


def _append_invalid_label_level_findings(findings: list[dict[str, object]], rows: list[dict[str, Any]]) -> None:
    invalid = sorted(
        {
            str(row.get("label_level") or "").strip()
            for row in rows
            if str(row.get("label_level") or "").strip() not in LABEL_LEVELS
        }
    )
    if invalid:
        findings.append(
            _finding(
                "fatal",
                "invalid_label_level",
                "reviewed labels contain invalid label_level values",
                {"values": invalid, "allowed": sorted(LABEL_LEVELS)},
            )
        )


def _append_invalid_confidence_findings(findings: list[dict[str, object]], rows: list[dict[str, Any]]) -> None:
    invalid = sorted(
        {
            str(row.get("review_confidence") or "").strip()
            for row in rows
            if str(row.get("review_confidence") or "").strip() not in REVIEW_CONFIDENCE_VALUES
        }
    )
    if invalid:
        findings.append(
            _finding(
                "fatal",
                "invalid_review_confidence",
                "reviewed labels contain invalid review_confidence values",
                {"values": invalid, "allowed": sorted(REVIEW_CONFIDENCE_VALUES)},
            )
        )


def _append_missing_butterfly_taxonomy_findings(findings: list[dict[str, object]], rows: list[dict[str, Any]]) -> None:
    bad_rows = [
        _row_ref(index, row)
        for index, row in enumerate(rows)
        if bool(row.get("is_butterfly")) and (not _text(row.get("family")) or not _text(row.get("scientific_name")))
    ]
    if bad_rows:
        findings.append(
            _finding(
                "fatal",
                "butterfly_label_missing_taxonomy",
                "butterfly-positive reviewed labels require family and scientific_name",
                {"rows": bad_rows},
            )
        )


def _append_duplicate_species_conflict_findings(findings: list[dict[str, object]], rows: list[dict[str, Any]]) -> None:
    labels_by_object: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
    for row in rows:
        detection_id = _text(row.get("detection_id"))
        if not detection_id or _text(row.get("label_level")) not in {"object", "family", "species"}:
            continue
        if not bool(row.get("is_butterfly")):
            continue
        signature = (_text(row.get("accepted_taxon_key")), _text(row.get("scientific_name")))
        if not any(signature):
            continue
        key = (_text(row.get("source")), _text(row.get("flickr_photo_id")), detection_id)
        labels_by_object.setdefault(key, set()).add(signature)

    conflicts = [
        {"source": key[0], "flickr_photo_id": key[1], "detection_id": key[2], "labels": sorted(values)}
        for key, values in sorted(labels_by_object.items())
        if len(values) > 1
    ]
    if conflicts:
        findings.append(
            _finding(
                "fatal",
                "duplicate_object_conflicting_species_labels",
                "duplicate object-level reviewed labels contain conflicting species labels",
                {"conflicts": conflicts},
            )
        )


def _append_review_metadata_warnings(findings: list[dict[str, object]], rows: list[dict[str, Any]]) -> None:
    missing_reviewer = [_row_ref(index, row) for index, row in enumerate(rows) if not _text(row.get("reviewer_id"))]
    if missing_reviewer:
        findings.append(
            _finding(
                "warning",
                "missing_reviewer_id",
                "one or more reviewed labels is missing reviewer_id",
                {"rows": missing_reviewer},
            )
        )

    missing_reviewed_at = [_row_ref(index, row) for index, row in enumerate(rows) if not _text(row.get("reviewed_at"))]
    if missing_reviewed_at:
        findings.append(
            _finding(
                "warning",
                "missing_reviewed_at",
                "one or more reviewed labels is missing reviewed_at",
                {"rows": missing_reviewed_at},
            )
        )

    low_confidence = [
        _row_ref(index, row)
        for index, row in enumerate(rows)
        if _text(row.get("review_confidence")) == "low"
    ]
    if low_confidence:
        findings.append(
            _finding(
                "warning",
                "low_confidence_labels",
                "one or more reviewed labels has low confidence",
                {"rows": low_confidence},
            )
        )


def _append_photo_without_object_warning(findings: list[dict[str, object]], rows: list[dict[str, Any]]) -> None:
    object_keys = {
        (_text(row.get("source")), _text(row.get("flickr_photo_id")))
        for row in rows
        if (
            _text(row.get("label_level")) in {"object", "family", "species", "negative"}
            and _text(row.get("detection_id"))
        )
    }
    photo_only = [
        _row_ref(index, row)
        for index, row in enumerate(rows)
        if (
            _text(row.get("label_level")) == "photo"
            and (_text(row.get("source")), _text(row.get("flickr_photo_id"))) not in object_keys
        )
    ]
    if photo_only:
        findings.append(
            _finding(
                "warning",
                "photo_label_without_object_label",
                "one or more photo-level labels has no matching object-level label",
                {"rows": photo_only},
            )
        )


def _row_ref(index: int, row: dict[str, Any]) -> dict[str, object]:
    return {
        "row_index": index,
        "source": _text(row.get("source")),
        "flickr_photo_id": _text(row.get("flickr_photo_id")),
        "detection_id": _text(row.get("detection_id")),
    }


def _finding(severity: str, code: str, message: str, details: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "details": dict(details or {}),
    }


def _text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


__all__ = [
    "LABEL_LEVELS",
    "REVIEWED_LABEL_SCHEMA",
    "REVIEWED_LABEL_SCHEMA_VERSION",
    "REVIEW_CONFIDENCE_VALUES",
    "empty_reviewed_label_frame",
    "read_reviewed_labels",
    "validate_reviewed_label_frame",
]
