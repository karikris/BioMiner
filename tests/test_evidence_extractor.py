from __future__ import annotations

import polars as pl

from flickr_bio_occurrence.evidence.extractor import build_evidence_frame, write_staging_evidence


def test_extracts_text_comment_verification_and_species_evidence() -> None:
    frame = build_evidence_frame(
        [
            {
                "photos": {
                    "photo": [
                        {
                            "id": "1",
                            "owner": "owner-1",
                            "ownername": "Owner Name",
                            "title": "Papilio demoleus in garden",
                            "description": {"_content": "verified by local expert as species: Papilio demoleus"},
                            "tags": "lime butterfly determined by observer",
                            "machine_tags": "taxon:Papilio_demoleus",
                            "comments": {
                                "comment": [
                                    {"_content": "confirmed by reviewer"},
                                    {"_content": "ID by museum volunteer"},
                                ]
                            },
                            "url_l": "https://live.staticflickr.com/large.jpg",
                            "url_m": "https://live.staticflickr.com/medium.jpg",
                            "datetaken": "2024-01-01 10:00:00",
                            "dateupload": "1700000000",
                            "latitude": "-27.0",
                            "longitude": "153.0",
                            "accuracy": "16",
                            "license": "4",
                        }
                    ]
                }
            }
        ],
        species_query="Papilio demoleus",
    )
    row = frame.to_dicts()[0]

    assert row["flickr_photo_id"] == "1"
    assert row["image_url"] == "https://live.staticflickr.com/large.jpg"
    assert row["image_url_kind"] == "url_l"
    assert row["comments_count"] == 2
    assert "confirmed by reviewer" in row["comments_text"]
    assert row["species_text_match"] is True
    assert set(row["species_text_source"]) == {"title", "description"}
    assert "Papilio demoleus" in row["scientific_names_detected"]
    assert row["human_verification_detected"] is True
    assert {"verified by", "determined by", "confirmed by", "ID by", "expert", "museum", "species:", "taxon:"}.issubset(
        set(row["human_verification_terms"])
    )
    assert row["human_verification_confidence"] == 1.0
    assert "human_verification_phrase" in row["review_flags"]
    assert row["occurrence_bin"] == "in_review"
    assert row["bin_reason"] == "unclassified_evidence"
    assert row["image_category"] == "adult_butterfly"
    assert row["life_stage"] == "adult_butterfly"
    assert row["negative_filter_reason"] is None


def test_extracts_museum_artwork_specimen_collection_captive_and_non_target_indicators() -> None:
    frame = build_evidence_frame(
        [
            {
                "photos": {
                    "photo": [
                        {
                            "id": "2",
                            "title": "Museum specimen plate",
                            "description": {"_content": "Pinned specimen from collection; artwork illustration; captive in butterfly house; moth nearby"},
                            "url_m": "https://live.staticflickr.com/medium.jpg",
                        }
                    ]
                }
            }
        ],
        species_query="Papilio demoleus",
    )
    row = frame.to_dicts()[0]

    assert row["image_url_kind"] == "url_m"
    assert row["museum_detected"] is True
    assert row["artwork_detected"] is True
    assert row["specimen_detected"] is True
    assert row["image_category"] == "museum_specimen"
    assert row["life_stage"] == "adult_butterfly"
    assert row["negative_filter_reason"] == "museum_specimen"
    assert row["collection_detected"] is True
    assert row["captive_detected"] is True
    assert row["non_target_order_detected"] is True
    assert set(row["review_flags"]) >= {
        "museum_context",
        "artwork_context",
        "specimen_context",
        "collection_context",
        "captive_context",
        "non_target_order_context",
        "no_species_text_match",
    }


def test_missing_image_url_and_comments_are_represented_explicitly() -> None:
    frame = build_evidence_frame(
        [{"photos": {"photo": [{"id": "3", "title": "Papilio demoleus"}]}}],
        species_query="Papilio demoleus",
    )
    row = frame.to_dicts()[0]

    assert row["image_url"] is None
    assert row["image_url_kind"] is None
    assert row["comments_text"] is None
    assert row["comments_count"] == 0
    assert "missing_image_url" in row["review_flags"]


def test_evidence_schema_defaults_to_adult_butterfly_category() -> None:
    frame = build_evidence_frame(
        [{"photos": {"photo": [{"id": "5", "title": "Papilio demoleus", "url_l": "https://live.staticflickr.com/large.jpg"}]}}],
        species_query="Papilio demoleus",
    )
    row = frame.to_dicts()[0]

    assert row["image_category"] == "adult_butterfly"
    assert row["life_stage"] == "adult_butterfly"
    assert row["negative_filter_reason"] is None


def test_evidence_schema_maps_tattoo_and_life_stage_to_image_category() -> None:
    tattoo = build_evidence_frame(
        [{"photos": {"photo": [{"id": "6", "title": "Papilio demoleus tattoo", "url_l": "https://live.staticflickr.com/large.jpg"}]}}],
        species_query="Papilio demoleus",
    ).to_dicts()[0]
    caterpillar = build_evidence_frame(
        [{"photos": {"photo": [{"id": "7", "title": "Papilio demoleus caterpillar", "url_l": "https://live.staticflickr.com/large.jpg"}]}}],
        species_query="Papilio demoleus",
    ).to_dicts()[0]

    assert tattoo["tattoo_detected"] is True
    assert tattoo["image_category"] == "tattoo"
    assert tattoo["life_stage"] == "adult_butterfly"
    assert caterpillar["image_category"] == "life_stage_non_adult"
    assert caterpillar["life_stage"] == "caterpillar"


def test_write_staging_evidence_writes_parquet(tmp_path) -> None:
    output_path = write_staging_evidence(
        [{"photos": {"photo": [{"id": "4", "title": "Papilio demoleus", "url_l": "https://live.staticflickr.com/large.jpg"}]}}],
        species_query="Papilio demoleus",
        output_path=tmp_path / "staging" / "evidence" / "staging_evidence.parquet",
    )

    frame = pl.read_parquet(output_path)
    assert output_path.name == "staging_evidence.parquet"
    assert frame["flickr_photo_id"][0] == "4"
    assert frame["species_text_match"][0] is True
