from __future__ import annotations

from flickr_bio_occurrence.geo.area_flags import classify_georef_area
from flickr_bio_occurrence.geo.georeference import from_flickr_coordinates


def test_geo_area_over_100km2_is_flagged() -> None:
    result = classify_georef_area(101)

    assert result.georef_area_over_100km2 is True
    assert result.georef_review_required is True
    assert result.georef_precision_class == "area_over_100km2"


def test_exact_flickr_coordinates_are_stored_internal() -> None:
    result = from_flickr_coordinates(latitude=-27.4698, longitude=153.0251, accuracy=16)

    assert result.exact_decimalLatitude_internal == -27.4698
    assert result.exact_decimalLongitude_internal == 153.0251
    assert result.exact_coordinate_source == "flickr_explicit_geotag"
    assert result.decimalLatitude == -27.4698
    assert result.decimalLongitude == 153.0251
    assert result.publication_generalisation_required is False
    assert result.publish_decimalLatitude is None
