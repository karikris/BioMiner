from __future__ import annotations

import polars as pl
import pytest

from biominer.evaluation.flickr_export import (
    FlickrExportValidationError,
    validate_verified_flickr_export,
    write_verified_flickr_export,
)


def _valid_row() -> dict[str, object]:
    digest = "sha256:" + "d" * 64
    return {
        "source_record_id": "flickr:verified",
        "human_review_decision": "include",
        "source_image_sha256": digest,
        "review_source_image_sha256": digest,
        "conflict_status": "not_required",
        "occurrence_claim_supported": True,
        "eligible_for_final_occurrence_dataset": True,
        "release_state": "eligible",
        "scientific_name": "Papilio demoleus",
    }


def test_verified_flickr_export_writes_parquet_without_changing_rows(tmp_path) -> None:
    frame = pl.DataFrame([_valid_row()])
    output = write_verified_flickr_export(frame, tmp_path / "occurrences.parquet")

    assert output.exists()
    assert pl.read_parquet(output).to_dicts() == frame.to_dicts()
    assert validate_verified_flickr_export(frame) is frame


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"human_review_decision": None}, "unreviewed"),
        ({"human_review_decision": "Skip"}, "skip"),
        ({"human_review_decision": "Can’t view"}, "cant_view"),
        ({"human_review_decision": "uncertain"}, "uncertain_label"),
        ({"conflict_status": "required"}, "unresolved_conflict"),
        ({"review_source_image_sha256": "sha256:" + "e" * 64}, "stale_source_hash"),
        ({"occurrence_claim_supported": False}, "unsupported_occurrence_claim"),
        ({"eligible_for_final_occurrence_dataset": False}, "release_not_eligible"),
    ],
)
def test_every_unverified_state_blocks_the_entire_export(
    tmp_path,
    changes: dict[str, object],
    reason: str,
) -> None:
    row = {**_valid_row(), **changes}
    destination = tmp_path / "must_not_exist.parquet"

    with pytest.raises(FlickrExportValidationError) as error:
        write_verified_flickr_export(pl.DataFrame([row]), destination)

    assert reason in error.value.blocked_records["flickr:verified"]
    assert not destination.exists()


def test_one_bad_record_prevents_accidental_partial_release(tmp_path) -> None:
    valid = _valid_row()
    invalid = {
        **_valid_row(),
        "source_record_id": "flickr:unreviewed",
        "human_review_decision": "unreviewed",
        "eligible_for_final_occurrence_dataset": False,
        "release_state": "excluded",
    }
    destination = tmp_path / "mixed.parquet"

    with pytest.raises(FlickrExportValidationError, match="flickr:unreviewed"):
        write_verified_flickr_export(pl.DataFrame([valid, invalid]), destination)

    assert not destination.exists()


def test_export_rejects_missing_contract_columns_and_empty_frames() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        validate_verified_flickr_export(pl.DataFrame([{"source_record_id": "x"}]))
    with pytest.raises(ValueError, match="must not be empty"):
        validate_verified_flickr_export(
            pl.DataFrame(schema={name: pl.String for name in _valid_row()})
        )
