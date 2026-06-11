from __future__ import annotations

from biominer.flickr_fetch.endpoints import ALLOWED_FLICKR_METHODS, FLICKR_REST_BASE_URL


def test_no_private_or_scraping_endpoints_exist() -> None:
    assert FLICKR_REST_BASE_URL == "https://www.flickr.com/services/rest/"
    assert ALLOWED_FLICKR_METHODS == {
        "flickr.photos.search",
        "flickr.photos.getInfo",
        "flickr.photos.getExif",
        "flickr.photos.geo.getLocation",
        "flickr.photos.comments.getList",
    }
    for method in ALLOWED_FLICKR_METHODS:
        assert "private" not in method.lower()
        assert "scrape" not in method.lower()
