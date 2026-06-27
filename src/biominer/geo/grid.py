from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Iterable

import polars as pl


GEO_GRID_LEVELS: dict[str, float | None] = {
    "G0_world": None,
    "G2_20deg": 20.0,
    "G3_10deg": 10.0,
    "G4_5deg": 5.0,
    "G5_2deg": 2.0,
    "G6_1deg": 1.0,
}

_FALLBACK_ORDER = ("G6_1deg", "G5_2deg", "G4_5deg", "G3_10deg", "G2_20deg", "G0_world")


@dataclass(frozen=True)
class GeoCandidateLookup:
    candidates: pl.DataFrame
    requested_grid_level: str
    selected_grid_level: str
    geocell_id: str
    fallback_reason: str | None


def geocell_id(grid_level: str, latitude: float, longitude: float) -> str:
    cell_size = GEO_GRID_LEVELS[grid_level]
    if cell_size is None:
        return "G0_world:global"
    lat_min = _cell_floor(latitude, origin=-90.0, cell_size=cell_size)
    lon_min = _cell_floor(longitude, origin=-180.0, cell_size=cell_size)
    lat_max = min(90.0, lat_min + cell_size)
    lon_max = min(180.0, lon_min + cell_size)
    return f"{grid_level}:lat_{lat_min:+06.1f}_{lat_max:+06.1f}:lon_{lon_min:+07.1f}_{lon_max:+07.1f}"


def neighbour_geocell_ids(grid_level: str, latitude: float, longitude: float) -> tuple[str, ...]:
    cell_size = GEO_GRID_LEVELS[grid_level]
    if cell_size is None:
        return ("G0_world:global",)
    lat_min = _cell_floor(latitude, origin=-90.0, cell_size=cell_size)
    lon_min = _cell_floor(longitude, origin=-180.0, cell_size=cell_size)
    cells: set[str] = set()
    for lat_offset in (-cell_size, 0.0, cell_size):
        for lon_offset in (-cell_size, 0.0, cell_size):
            lat = min(89.999999, max(-90.0, lat_min + lat_offset + 0.000001))
            lon = _wrap_longitude(lon_min + lon_offset + 0.000001)
            cells.add(geocell_id(grid_level, lat, lon))
    return tuple(sorted(cells))


def candidate_set_for_point(
    geo_species_index: pl.DataFrame,
    *,
    latitude: float,
    longitude: float,
    preferred_grid_level: str = "G4_5deg",
    min_species_per_cell: int = 5,
    include_neighbours: bool = False,
) -> GeoCandidateLookup:
    for level in _fallback_levels(preferred_grid_level):
        cell_ids = (
            neighbour_geocell_ids(level, latitude, longitude)
            if include_neighbours and level != "G0_world"
            else (geocell_id(level, latitude, longitude),)
        )
        candidates = geo_species_index.filter(
            (pl.col("grid_level") == level) & pl.col("geocell_id").is_in(cell_ids)
        )
        species_count = candidates.select(pl.col("species_key").n_unique()).item() if candidates.height else 0
        if species_count >= min_species_per_cell or level == "G0_world":
            selected_cell = cell_ids[0] if len(cell_ids) == 1 else "|".join(cell_ids)
            return GeoCandidateLookup(
                candidates=candidates.sort(["candidate_rank_prior", "scientific_name"], descending=[True, False]) if candidates.height else candidates,
                requested_grid_level=preferred_grid_level,
                selected_grid_level=level,
                geocell_id=selected_cell,
                fallback_reason=None if level == preferred_grid_level else "local_cell_below_min_species",
            )
    raise ValueError(f"Unsupported preferred grid level {preferred_grid_level!r}")


def _fallback_levels(preferred_grid_level: str) -> Iterable[str]:
    if preferred_grid_level not in _FALLBACK_ORDER:
        raise ValueError(f"Unsupported grid level {preferred_grid_level!r}")
    start = _FALLBACK_ORDER.index(preferred_grid_level)
    return _FALLBACK_ORDER[start:]


def _cell_floor(value: float, *, origin: float, cell_size: float) -> float:
    return origin + floor((value - origin) / cell_size) * cell_size


def _wrap_longitude(value: float) -> float:
    while value < -180.0:
        value += 360.0
    while value >= 180.0:
        value -= 360.0
    return value
