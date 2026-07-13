from __future__ import annotations

import math
from numbers import Integral, Real


MIN_GRID_RESOLUTION = 0
MAX_GRID_RESOLUTION = 15


def validate_latitude(value: Real, *, field_name: str = "latitude") -> float:
    latitude = _finite_real(value, field_name=field_name)
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f"{field_name} must be between -90 and 90 degrees")
    return latitude


def normalize_longitude(value: Real, *, field_name: str = "longitude") -> float:
    longitude = _finite_real(value, field_name=field_name)
    normalized = (longitude + 180.0) % 360.0 - 180.0
    return 0.0 if normalized == 0.0 else normalized


def validate_coordinate_uncertainty(
    value: Real | None,
    *,
    field_name: str = "coordinate_uncertainty_m",
) -> float | None:
    if value is None:
        return None
    uncertainty = _finite_real(value, field_name=field_name)
    if uncertainty < 0.0:
        raise ValueError(f"{field_name} must be greater than or equal to zero")
    return uncertainty


def validate_resolution(value: Integral, *, field_name: str = "resolution") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} must be an integer resolution")
    resolution = int(value)
    if not MIN_GRID_RESOLUTION <= resolution <= MAX_GRID_RESOLUTION:
        raise ValueError(
            f"{field_name} must be between {MIN_GRID_RESOLUTION} and {MAX_GRID_RESOLUTION}"
        )
    return resolution


def validate_grid_distance(value: Integral, *, field_name: str = "grid_distance") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} must be a non-negative integer")
    distance = int(value)
    if distance < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return distance


def _finite_real(value: Real, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite real number")
    return normalized


__all__ = [
    "MAX_GRID_RESOLUTION",
    "MIN_GRID_RESOLUTION",
    "normalize_longitude",
    "validate_coordinate_uncertainty",
    "validate_grid_distance",
    "validate_latitude",
    "validate_resolution",
]
