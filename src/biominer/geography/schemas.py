from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

from biominer.geography.validation import (
    normalize_longitude,
    validate_coordinate_uncertainty,
    validate_latitude,
    validate_resolution,
)


@dataclass(frozen=True, slots=True)
class GeographicCoordinate:
    latitude: Real
    longitude: Real
    coordinate_uncertainty_m: Real | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "latitude", validate_latitude(self.latitude))
        object.__setattr__(self, "longitude", normalize_longitude(self.longitude))
        object.__setattr__(
            self,
            "coordinate_uncertainty_m",
            validate_coordinate_uncertainty(self.coordinate_uncertainty_m),
        )


@dataclass(frozen=True, slots=True)
class GeographicResolutions:
    coarse: Integral
    regional: Integral
    local: Integral

    def __post_init__(self) -> None:
        coarse = validate_resolution(self.coarse, field_name="coarse resolution")
        regional = validate_resolution(self.regional, field_name="regional resolution")
        local = validate_resolution(self.local, field_name="local resolution")
        if not coarse < regional < local:
            raise ValueError("geographic resolutions must be strictly ordered coarse < regional < local")
        object.__setattr__(self, "coarse", coarse)
        object.__setattr__(self, "regional", regional)
        object.__setattr__(self, "local", local)

    @property
    def values(self) -> tuple[int, int, int]:
        return (int(self.coarse), int(self.regional), int(self.local))


@dataclass(frozen=True, slots=True)
class CellAtResolution:
    resolution: Integral
    cell_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolution", validate_resolution(self.resolution))
        if not isinstance(self.cell_id, str) or not self.cell_id.strip():
            raise ValueError("cell_id must be a nonblank string")


@dataclass(frozen=True, slots=True)
class CellProjection:
    coordinate: GeographicCoordinate
    cells: tuple[CellAtResolution, ...]
    grid_name: str
    grid_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate, GeographicCoordinate):
            raise TypeError("coordinate must be a GeographicCoordinate")
        if not self.cells:
            raise ValueError("cells must contain at least one resolution")
        resolutions = tuple(int(item.resolution) for item in self.cells)
        if resolutions != tuple(sorted(set(resolutions))):
            raise ValueError("cells must use unique resolutions in ascending order")
        if not self.grid_name.strip() or not self.grid_version.strip():
            raise ValueError("grid_name and grid_version must be nonblank")

    def cell_at(self, resolution: Integral) -> str:
        target = validate_resolution(resolution)
        for item in self.cells:
            if item.resolution == target:
                return item.cell_id
        raise KeyError(f"projection does not contain resolution {target}")


__all__ = [
    "CellAtResolution",
    "CellProjection",
    "GeographicCoordinate",
    "GeographicResolutions",
]
