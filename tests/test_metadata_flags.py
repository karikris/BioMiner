from __future__ import annotations

import polars as pl

from biominer.filter.category_model import infer_category_from_record
from biominer.filter.metadata_flags import flag_metadata_records


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
    assert "filter_decision" not in flagged.columns
    assert "filter_reason" not in flagged.columns
    row_by_id = {row["flickr_photo_id"]: row for row in flagged.to_dicts()}
    assert row_by_id["1"]["hard_negative_text_hint"] is False
    assert row_by_id["1"]["metadata_image_category_hint"] == "unknown"
    assert row_by_id["1"]["metadata_life_stage_hint"] == "unknown"
    assert row_by_id["2"]["artwork_hint"] is True
    assert row_by_id["2"]["matched_keyword_groups"] == ["artwork"]
    assert row_by_id["3"]["object_or_product_hint"] is True
    assert row_by_id["3"]["matched_keywords"] == ["toy", "sticker"]


def test_species_bioclip_metadata_fields_do_not_infer_adult_visual_category() -> None:
    category = infer_category_from_record(
        {
            "raw_title": "Papilio demoleus from Flickr metadata",
            "bioclip_top1_label": "a photo of Papilio demoleus",
            "bioclip_species_agreement_status": "exact_species_agreement",
            "is_target_positive": True,
        }
    )

    assert category["image_category"] == "unknown"
    assert category["life_stage"] == "unknown"
    assert category["negative_filter_reason"] is None


def test_metadata_flags_preserve_life_stage_hints() -> None:
    frame = pl.DataFrame([{"flickr_photo_id": "1", "raw_title": "Papilio demoleus caterpillar on citrus"}])

    row = flag_metadata_records(frame, keyword_groups=_groups()).to_dicts()[0]

    assert row["metadata_image_category_hint"] == "life_stage_non_adult"
    assert row["metadata_life_stage_hint"] == "caterpillar"
    assert row["hard_negative_text_hint"] is False
