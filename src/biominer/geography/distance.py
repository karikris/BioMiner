from __future__ import annotations

import math

from biominer.geography.schemas import GeographicCoordinate


# The authalic radius used by the H3 core great-circle implementation.
EARTH_RADIUS_KM = 6371.007180918475


def great_circle_distance_km(
    first: GeographicCoordinate,
    second: GeographicCoordinate,
) -> float:
    if not isinstance(first, GeographicCoordinate) or not isinstance(second, GeographicCoordinate):
        raise TypeError("great-circle distance requires GeographicCoordinate values")

    latitude_1 = math.radians(float(first.latitude))
    latitude_2 = math.radians(float(second.latitude))
    latitude_delta = latitude_2 - latitude_1
    longitude_delta = math.radians(float(second.longitude) - float(first.longitude))
    sin_latitude = math.sin(latitude_delta / 2.0)
    sin_longitude = math.sin(longitude_delta / 2.0)
    haversine = (
        sin_latitude * sin_latitude
        + math.cos(latitude_1) * math.cos(latitude_2) * sin_longitude * sin_longitude
    )
    central_angle = 2.0 * math.atan2(math.sqrt(haversine), math.sqrt(max(0.0, 1.0 - haversine)))
    return EARTH_RADIUS_KM * central_angle


__all__ = ["EARTH_RADIUS_KM", "great_circle_distance_km"]
