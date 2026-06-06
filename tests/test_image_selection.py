from __future__ import annotations

import pytest

from flickr_bio_occurrence.vision.image_selection import ImageSelectionPolicy, OriginalImageTooLarge, select_flickr_image_url


def test_default_image_selection_prefers_large_url() -> None:
    selected = select_flickr_image_url(
        {
            "url_m": "https://live.staticflickr.com/medium.jpg",
            "url_l": "https://live.staticflickr.com/large.jpg",
            "url_o": "https://live.staticflickr.com/original.jpg",
        }
    )

    assert selected.url == "https://live.staticflickr.com/large.jpg"
    assert selected.kind == "url_l"


def test_default_image_selection_falls_back_to_medium_url() -> None:
    selected = select_flickr_image_url(
        {
            "url_m": "https://live.staticflickr.com/medium.jpg",
            "url_o": "https://live.staticflickr.com/original.jpg",
        }
    )

    assert selected.url == "https://live.staticflickr.com/medium.jpg"
    assert selected.kind == "url_m"


def test_original_url_requires_explicit_diagnostic_mode() -> None:
    photo = {
        "url_m": "https://live.staticflickr.com/medium.jpg",
        "url_l": "https://live.staticflickr.com/large.jpg",
        "url_o": "https://live.staticflickr.com/original.jpg",
        "o_width": "1000",
        "o_height": "1000",
    }

    assert select_flickr_image_url(photo).kind == "url_l"
    assert select_flickr_image_url(photo, ImageSelectionPolicy(mode="original_diagnostic")).kind == "url_o"


def test_original_diagnostic_mode_enforces_pixel_guardrail() -> None:
    with pytest.raises(OriginalImageTooLarge):
        select_flickr_image_url(
            {
                "url_o": "https://live.staticflickr.com/original.jpg",
                "o_width": "10000",
                "o_height": "10000",
            },
            ImageSelectionPolicy(mode="original_diagnostic", original_max_pixels=1_000_000),
        )
