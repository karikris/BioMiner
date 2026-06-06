from __future__ import annotations

from flickr_bio_occurrence.pipeline.transforms import build_dwc_rows, build_silver_candidates, flatten_search_payloads


def test_flatten_search_payloads_preserves_raw_photo_metadata_fields() -> None:
    bronze = flatten_search_payloads(
        [
            {
                "photos": {
                    "photo": [
                        {
                            "id": "1",
                            "owner": "owner-1",
                            "ownername": "Owner Name",
                            "title": "Papilio demoleus",
                            "latitude": "-27.4698",
                            "longitude": "153.0251",
                            "datetaken": "2024-01-15 10:00:00",
                            "tags": "Papilio demoleus lime butterfly",
                            "url_m": "https://live.staticflickr.com/example.jpg",
                            "license": "4",
                        }
                    ]
                }
            }
        ],
        species_name="Papilio demoleus",
        region_id="AU_QLD",
        work_item_id="work-1",
    )

    assert bronze.shape[0] == 1
    assert bronze["flickr_photo_id"][0] == "1"
    assert bronze["raw_title"][0] == "Papilio demoleus"
    assert bronze["decimalLatitude"][0] == -27.4698


def test_flatten_search_payloads_prefers_large_image_url_over_medium() -> None:
    bronze = flatten_search_payloads(
        [
            {
                "photos": {
                    "photo": [
                        {
                            "id": "1",
                            "url_m": "https://live.staticflickr.com/medium.jpg",
                            "url_l": "https://live.staticflickr.com/large.jpg",
                            "url_o": "https://live.staticflickr.com/original.jpg",
                        }
                    ]
                }
            }
        ],
        species_name="Papilio demoleus",
        region_id="WORLD",
        work_item_id="work-1",
    )

    assert bronze["image_url"][0] == "https://live.staticflickr.com/large.jpg"


def test_flatten_search_payloads_falls_back_to_medium_image_url() -> None:
    bronze = flatten_search_payloads(
        [
            {
                "photos": {
                    "photo": [
                        {
                            "id": "1",
                            "url_m": "https://live.staticflickr.com/medium.jpg",
                            "url_o": "https://live.staticflickr.com/original.jpg",
                        }
                    ]
                }
            }
        ],
        species_name="Papilio demoleus",
        region_id="WORLD",
        work_item_id="work-1",
    )

    assert bronze["image_url"][0] == "https://live.staticflickr.com/medium.jpg"


def test_build_silver_candidates_keeps_exact_coordinates_and_needs_review() -> None:
    bronze = flatten_search_payloads(
        [{"photos": {"photo": [{"id": "1", "title": "Papilio demoleus", "latitude": "-27.0", "longitude": "153.0"}]}}],
        species_name="Papilio demoleus",
        region_id="AU_QLD",
        work_item_id="work-1",
    )

    silver = build_silver_candidates(bronze)

    assert silver["exact_decimalLatitude_internal"][0] == -27.0
    assert silver["exact_decimalLongitude_internal"][0] == 153.0
    assert silver["review_status"][0] == "needs_review"
    assert silver["range_extension_candidate"][0] is False


def test_build_dwc_rows_outputs_required_dwc_fields() -> None:
    bronze = flatten_search_payloads(
        [{"photos": {"photo": [{"id": "1", "title": "Papilio demoleus", "latitude": "-27.0", "longitude": "153.0"}]}}],
        species_name="Papilio demoleus",
        region_id="AU_QLD",
        work_item_id="work-1",
    )
    silver = build_silver_candidates(bronze)

    gold = build_dwc_rows(silver)

    assert gold["scientificName"][0] == "Papilio demoleus"
    assert gold["basisOfRecord"][0] == "HumanObservation"
    assert gold["occurrenceID"][0]
