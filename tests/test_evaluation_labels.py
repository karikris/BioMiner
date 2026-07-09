from __future__ import annotations

from pathlib import Path

import polars as pl

from biominer.evaluation.labels import (
    REVIEWED_LABEL_SCHEMA,
    empty_reviewed_label_frame,
    read_reviewed_labels,
    validate_reviewed_label_frame,
)


FIXTURE_DIR = Path("tests/fixtures/evaluation")


def test_empty_reviewed_label_frame_has_correct_schema() -> None:
    frame = empty_reviewed_label_frame()

    assert frame.is_empty()
    assert dict(frame.schema) == REVIEWED_LABEL_SCHEMA


def test_valid_reviewed_label_fixture_passes() -> None:
    frame = read_reviewed_labels(FIXTURE_DIR / "reviewed_labels_valid.jsonl")

    assert validate_reviewed_label_frame(frame) == []


def test_read_reviewed_labels_supports_parquet(tmp_path) -> None:
    source = read_reviewed_labels(FIXTURE_DIR / "reviewed_labels_valid.jsonl")
    path = tmp_path / "reviewed_labels.parquet"
    source.write_parquet(path)

    frame = read_reviewed_labels(path)

    assert frame.to_dicts() == source.to_dicts()
    assert validate_reviewed_label_frame(frame) == []


def test_duplicate_conflict_produces_fatal_finding() -> None:
    frame = read_reviewed_labels(FIXTURE_DIR / "reviewed_labels_duplicate_conflict.jsonl")

    findings = validate_reviewed_label_frame(frame)

    assert _finding_codes(findings, severity="fatal") == ["duplicate_object_conflicting_species_labels"]


def test_missing_family_for_butterfly_produces_fatal_finding() -> None:
    frame = pl.DataFrame([{**_valid_species_row(), "family": ""}])

    findings = validate_reviewed_label_frame(frame)

    assert _finding_codes(findings, severity="fatal") == ["butterfly_label_missing_taxonomy"]


def test_invalid_confidence_produces_fatal_finding() -> None:
    frame = pl.DataFrame([{**_valid_species_row(), "review_confidence": "certain"}])

    findings = validate_reviewed_label_frame(frame)

    assert _finding_codes(findings, severity="fatal") == ["invalid_review_confidence"]


def test_missing_required_columns_produces_fatal_finding() -> None:
    frame = pl.DataFrame([{"source": "flickr", "flickr_photo_id": "1"}])

    findings = validate_reviewed_label_frame(frame)

    assert _finding_codes(findings, severity="fatal") == ["missing_required_columns"]


def _finding_codes(findings: list[dict[str, object]], *, severity: str) -> list[str]:
    return sorted(str(finding["code"]) for finding in findings if finding.get("severity") == severity)


def _valid_species_row() -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": "300",
        "detection_id": "d300-1",
        "crop_hash": "sha256:crop300",
        "label_level": "species",
        "is_butterfly": True,
        "accepted_taxon_key": "gbif:100",
        "scientific_name": "Papilio demoleus",
        "family_key": "gbif:9417",
        "family": "Papilionidae",
        "genus_key": "gbif:90",
        "genus": "Papilio",
        "label_source": "fixture",
        "reviewer_id": "reviewer-a",
        "reviewed_at": "2026-07-10T00:00:00Z",
        "review_confidence": "high",
        "review_notes": "synthetic reviewed label",
    }
