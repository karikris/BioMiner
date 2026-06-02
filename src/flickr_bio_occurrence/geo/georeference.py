from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Georeference:
    exact_decimalLatitude_internal: float
    exact_decimalLongitude_internal: float
    exact_coordinate_source: str
    decimalLatitude: float
    decimalLongitude: float
    coordinateUncertaintyInMeters: int | None
    verbatimLocality: str | None
    georeferenceSources: str
    georeferenceRemarks: str
    publish_decimalLatitude: float | None
    publish_decimalLongitude: float | None
    publish_coordinateUncertaintyInMeters: int | None
    publication_generalisation_required: bool
    publication_generalisation_reason: str | None


def from_flickr_coordinates(*, latitude: float, longitude: float, accuracy: int | None = None) -> Georeference:
    return Georeference(
        exact_decimalLatitude_internal=latitude,
        exact_decimalLongitude_internal=longitude,
        exact_coordinate_source="flickr_explicit_geotag",
        decimalLatitude=latitude,
        decimalLongitude=longitude,
        coordinateUncertaintyInMeters=None,
        verbatimLocality=None,
        georeferenceSources="Flickr explicit geotag",
        georeferenceRemarks=f"Flickr accuracy code: {accuracy}" if accuracy is not None else "Flickr explicit geotag",
        publish_decimalLatitude=None,
        publish_decimalLongitude=None,
        publish_coordinateUncertaintyInMeters=None,
        publication_generalisation_required=False,
        publication_generalisation_reason=None,
    )
