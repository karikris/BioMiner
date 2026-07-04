from __future__ import annotations

import importlib.util
import json

import polars as pl

from biominer.filter.metadata_flags import flag_metadata_parquet, flag_metadata_records, load_metadata_keyword_groups


def test_legacy_anti_keyword_module_path_is_removed() -> None:
    assert importlib.util.find_spec("biominer.filter.anti_keywords") is None


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


def test_flag_metadata_parquet_writes_all_input_rows(tmp_path) -> None:
    input_path = tmp_path / "input.parquet"
    keyword_path = tmp_path / "metadata_keywords.json"
    output_path = tmp_path / "flagged.parquet"
    pl.DataFrame(
        [
            {"flickr_photo_id": "1", "raw_title": "adult butterfly"},
            {"flickr_photo_id": "2", "raw_title": "pinned specimen"},
        ]
    ).write_parquet(input_path)
    keyword_path.write_text(json.dumps({"metadata_keywords": {"museum_specimen": ["pinned specimen"]}}), encoding="utf-8")

    payload = flag_metadata_parquet(input_path=input_path, keywords_json=keyword_path, output_path=output_path)

    assert payload["flagged_rows"] == 2
    assert payload["dropped_rows"] == 0
    assert pl.read_parquet(output_path).height == 2


def test_load_metadata_keyword_groups_accepts_grouped_json(tmp_path) -> None:
    path = tmp_path / "metadata_keywords.json"
    path.write_text(json.dumps({"artwork": ["plate"], "tattoo": ["tattoo"]}), encoding="utf-8")

    assert load_metadata_keyword_groups(path) == {"artwork": ("plate",), "tattoo": ("tattoo",)}
