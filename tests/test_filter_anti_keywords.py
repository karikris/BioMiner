from __future__ import annotations

import json

import polars as pl

from biominer.filter.anti_keywords import filter_biodiversity_parquet, filter_biodiversity_records, load_anti_keyword_groups


def _groups() -> dict[str, tuple[str, ...]]:
    return {
        "artwork": ("illustration", "painting"),
        "museum_specimen": ("pinned specimen",),
        "object_or_product": ("toy", "sticker"),
        "not_lepidoptera": ("beetle",),
    }


def test_anti_keyword_filter_drops_non_biodiversity_records() -> None:
    frame = pl.DataFrame(
        [
            {"flickr_photo_id": "1", "raw_title": "Papilio demoleus butterfly in garden"},
            {"flickr_photo_id": "2", "raw_title": "Papilio demoleus illustration plate"},
            {"flickr_photo_id": "3", "raw_title": "Butterfly toy sticker"},
        ]
    )

    kept, dropped = filter_biodiversity_records(frame, anti_keyword_groups=_groups())

    assert kept["flickr_photo_id"].to_list() == ["1"]
    assert dropped["flickr_photo_id"].to_list() == ["2", "3"]
    assert set(dropped["filter_decision"].to_list()) == {"drop"}


def test_anti_keyword_filter_keeps_life_stage_biodiversity_records() -> None:
    frame = pl.DataFrame([{"flickr_photo_id": "1", "raw_title": "Papilio demoleus caterpillar on citrus"}])

    kept, dropped = filter_biodiversity_records(frame, anti_keyword_groups=_groups())

    row = kept.to_dicts()[0]
    assert dropped.height == 0
    assert row["filter_decision"] == "keep"
    assert row["image_category"] == "life_stage_non_adult"
    assert row["life_stage"] == "caterpillar"


def test_filter_biodiversity_parquet_writes_kept_and_dropped_outputs(tmp_path) -> None:
    input_path = tmp_path / "input.parquet"
    anti_path = tmp_path / "anti.json"
    kept_path = tmp_path / "kept.parquet"
    dropped_path = tmp_path / "dropped.parquet"
    pl.DataFrame(
        [
            {"flickr_photo_id": "1", "raw_title": "adult butterfly"},
            {"flickr_photo_id": "2", "raw_title": "pinned specimen"},
        ]
    ).write_parquet(input_path)
    anti_path.write_text(json.dumps({"anti_keywords": {"museum_specimen": ["pinned specimen"]}}), encoding="utf-8")

    payload = filter_biodiversity_parquet(
        input_path=input_path,
        anti_keywords_json=anti_path,
        output_path=kept_path,
        dropped_output_path=dropped_path,
    )

    assert payload["kept_rows"] == 1
    assert payload["dropped_rows"] == 1
    assert pl.read_parquet(kept_path).height == 1
    assert pl.read_parquet(dropped_path).height == 1


def test_load_anti_keyword_groups_accepts_grouped_json(tmp_path) -> None:
    path = tmp_path / "anti.json"
    path.write_text(json.dumps({"artwork": ["plate"], "tattoo": ["tattoo"]}), encoding="utf-8")

    assert load_anti_keyword_groups(path) == {"artwork": ("plate",), "tattoo": ("tattoo",)}
