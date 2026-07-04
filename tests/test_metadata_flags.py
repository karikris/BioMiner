from __future__ import annotations

import polars as pl

import biominer.filter as filter_api
import biominer.filter.metadata_flags as metadata_flags
from biominer.filter.metadata_flags import flag_metadata_records


def test_metadata_keyword_path_helpers_are_removed_from_public_api() -> None:
    assert not hasattr(metadata_flags, "load_metadata_keyword_groups")
    assert not hasattr(metadata_flags, "flag_metadata_parquet")
    assert not hasattr(filter_api, "load_metadata_keyword_groups")
    assert not hasattr(filter_api, "flag_metadata_parquet")


def _groups() -> dict[str, tuple[str, ...]]:
    return {
        "artwork": ("illustration", "painting"),
        "museum_specimen": ("pinned specimen",),
        "object_or_product": ("toy", "sticker"),
        "not_lepidoptera": ("beetle",),
    }


def test_metadata_keyword_flags_do_not_drop_non_biodiversity_hints() -> None:
    frame = pl.DataFrame(
        [
            {"flickr_photo_id": "1", "raw_title": "Papilio demoleus butterfly in garden"},
            {"flickr_photo_id": "2", "raw_title": "Papilio demoleus illustration plate"},
            {"flickr_photo_id": "3", "raw_title": "Butterfly toy sticker"},
        ]
    )

    flagged = flag_metadata_records(frame, keyword_groups=_groups())

    assert flagged["flickr_photo_id"].to_list() == ["1", "2", "3"]
    assert flagged["filter_decision"].to_list() == ["flag", "flag", "flag"]
    row_by_id = {row["flickr_photo_id"]: row for row in flagged.to_dicts()}
    assert row_by_id["1"]["hard_negative_text_hint"] is False
    assert row_by_id["2"]["artwork_hint"] is True
    assert row_by_id["2"]["matched_keyword_groups"] == ["artwork"]
    assert row_by_id["3"]["object_or_product_hint"] is True
    assert row_by_id["3"]["matched_keywords"] == ["toy", "sticker"]


def test_metadata_flags_preserve_life_stage_hints() -> None:
    frame = pl.DataFrame([{"flickr_photo_id": "1", "raw_title": "Papilio demoleus caterpillar on citrus"}])

    row = flag_metadata_records(frame, keyword_groups=_groups()).to_dicts()[0]

    assert row["metadata_image_category_hint"] == "life_stage_non_adult"
    assert row["metadata_life_stage_hint"] == "caterpillar"
    assert row["hard_negative_text_hint"] is False
