from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeorefAreaFlags:
    georef_area_km2: float | None
    georef_area_over_100km2: bool
    georef_precision_class: str
    georef_review_required: bool


def classify_georef_area(georef_area_km2: float | None, *, threshold_km2: float = 100) -> GeorefAreaFlags:
    if georef_area_km2 is not None and georef_area_km2 > threshold_km2:
        return GeorefAreaFlags(georef_area_km2, True, "area_over_100km2", True)
    return GeorefAreaFlags(georef_area_km2, False, "exact_gps", False)
