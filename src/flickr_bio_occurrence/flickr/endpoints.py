from __future__ import annotations


FLICKR_REST_BASE_URL = "https://www.flickr.com/services/rest/"

ALLOWED_FLICKR_METHODS = {
    "flickr.photos.search",
    "flickr.photos.getInfo",
    "flickr.photos.getExif",
    "flickr.photos.geo.getLocation",
    "flickr.photos.comments.getList",
}

SEARCH_METHOD = "flickr.photos.search"
ENRICHMENT_METHODS = ALLOWED_FLICKR_METHODS - {SEARCH_METHOD}
