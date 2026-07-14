from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from biominer.evaluation.labels import (
    REVIEWED_LABEL_SCHEMA,
    REVIEWED_LABEL_SCHEMA_VERSION,
    REVIEWED_LABEL_V1_SCHEMA,
    REVIEWED_LABEL_V1_SCHEMA_VERSION,
    empty_reviewed_label_frame,
    normalize_reviewed_label_frame,
    read_reviewed_labels,
    validate_reviewed_label_frame,
)


FIXTURE_DIR = Path("tests/fixtures/evaluation")


def test_empty_reviewed_label_frame_has_correct_schema() -> None:
    frame = empty_reviewed_label_frame()

    assert frame.is_empty()
    assert dict(frame.schema) == REVIEWED_LABEL_SCHEMA
    assert REVIEWED_LABEL_SCHEMA_VERSION == "reviewed-labels-v2"
    assert REVIEWED_LABEL_V1_SCHEMA_VERSION == "reviewed-labels-v1"


def test_valid_reviewed_label_fixture_passes() -> None:
    frame = read_reviewed_labels(FIXTURE_DIR / "reviewed_labels_valid.jsonl")

    assert frame["schema_version"].unique().to_list() == [
        REVIEWED_LABEL_SCHEMA_VERSION
    ]
    assert frame["target_present"].to_list() == [None, False]
    assert frame["label_certainty"].to_list() == ["high", "medium"]
    assert frame["route"].to_list() == [None, None]
    assert validate_reviewed_label_frame(frame) == []


def test_v1_reader_resolves_target_presence_only_with_explicit_target_key() -> None:
    frame = read_reviewed_labels(
        FIXTURE_DIR / "reviewed_labels_valid.jsonl",
        target_accepted_taxon_key="gbif:100",
    )

    assert frame["target_present"].to_list() == [True, False]
    assert frame["unsuitable_for_species_identification"].to_list() == [
        False,
        None,
    ]
    assert frame["ambiguity_reason"].to_list() == [
        "legacy_v1_missing_target_aware_fields",
        "legacy_v1_missing_target_aware_fields",
    ]


def test_declared_v1_and_sampling_columns_survive_migration() -> None:
    source = pl.DataFrame(
        [
            {
                **_valid_v1_species_row(),
                "schema_version": REVIEWED_LABEL_V1_SCHEMA_VERSION,
                "sampling_weight": 2.5,
            }
        ]
    )

    frame = normalize_reviewed_label_frame(
        source,
        target_accepted_taxon_key="gbif:100",
    )

    assert frame["schema_version"].item() == REVIEWED_LABEL_SCHEMA_VERSION
    assert frame["target_present"].item() is True
    assert frame["sampling_weight"].item() == 2.5
    assert frame.columns[-1] == "sampling_weight"


def test_v1_negative_with_all_null_taxonomy_columns_migrates() -> None:
    row = _valid_v1_species_row()
    row.update(
        {
            "label_level": "negative",
            "is_butterfly": False,
            "accepted_taxon_key": None,
            "scientific_name": None,
            "family_key": None,
            "family": None,
            "genus_key": None,
            "genus": None,
        }
    )

    frame = normalize_reviewed_label_frame(pl.DataFrame([row]))

    assert frame["target_present"].item() is False
    assert frame["accepted_taxon_key"].dtype == pl.Null
    assert validate_reviewed_label_frame(frame) == []


def test_native_v2_frame_round_trips_without_migration() -> None:
    source = pl.DataFrame([_valid_species_row()])

    frame = normalize_reviewed_label_frame(source)

    assert frame.to_dicts() == source.select(frame.columns).to_dicts()
    assert validate_reviewed_label_frame(frame) == []


def test_partial_v2_frame_is_not_silently_downgraded_to_v1() -> None:
    partial = pl.DataFrame(
        [
            {
                **_valid_v1_species_row(),
                "schema_version": REVIEWED_LABEL_SCHEMA_VERSION,
                "target_present": True,
            }
        ]
    )

    with pytest.raises(ValueError, match="incomplete reviewed-label v2"):
        normalize_reviewed_label_frame(partial)


def test_reader_rejects_unsupported_declared_schema_version(tmp_path) -> None:
    path = tmp_path / "labels.parquet"
    pl.DataFrame(
        [{**_valid_species_row(), "schema_version": "reviewed-labels-v3"}]
    ).write_parquet(path)

    with pytest.raises(ValueError, match="unsupported reviewed-label schema_version"):
        read_reviewed_labels(path)


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


def test_route_must_match_reviewed_life_stage_and_visual_domain() -> None:
    frame = pl.DataFrame([{**_valid_species_row(), "route": "larval"}])

    findings = validate_reviewed_label_frame(frame)

    assert _finding_codes(findings, severity="fatal") == [
        "route_dimension_conflict"
    ]


def test_missing_required_columns_produces_fatal_finding() -> None:
    frame = pl.DataFrame([{"source": "flickr", "flickr_photo_id": "1"}])

    findings = validate_reviewed_label_frame(frame)

    assert _finding_codes(findings, severity="fatal") == ["missing_required_columns"]


def _finding_codes(findings: list[dict[str, object]], *, severity: str) -> list[str]:
    return sorted(str(finding["code"]) for finding in findings if finding.get("severity") == severity)


def _valid_species_row() -> dict[str, object]:
    return {
        "schema_version": REVIEWED_LABEL_SCHEMA_VERSION,
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
        "target_present": True,
        "label_certainty": "high",
        "life_stage": "adult",
        "visual_domain": "live_field",
        "view": "dorsal",
        "route": "adult_field",
        "geo_cluster_id": "h3:regional:300",
        "source_query_tier": "T1",
        "source_query_term": "Papilio demoleus",
        "duplicate_group_id": "duplicate:300",
        "observer_owner_group_id": "flickr-owner:owner-a",
        "dataset_split": "final_test",
        "second_review_status": "completed",
        "ambiguity_reason": "",
        "unsuitable_for_species_identification": False,
    }


def _valid_v1_species_row() -> dict[str, object]:
    row = _valid_species_row()
    return {key: row[key] for key in REVIEWED_LABEL_V1_SCHEMA}
