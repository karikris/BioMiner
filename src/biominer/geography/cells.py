from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from numbers import Integral, Real
from types import ModuleType
from typing import Protocol, runtime_checkable

from biominer.geography.schemas import (
    CellAtResolution,
    CellProjection,
    GeographicCoordinate,
    GeographicResolutions,
)
from biominer.geography.validation import validate_grid_distance, validate_resolution


class CellGridError(ValueError):
    """A backend-neutral cell operation could not be completed."""


class CellGridDependencyError(RuntimeError):
    """The optional cell-grid backend is not installed."""


@runtime_checkable
class CellGrid(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def coordinate_to_cell(self, coordinate: GeographicCoordinate, *, resolution: int) -> str: ...

    def parent(self, cell_id: str, *, resolution: int | None = None) -> str: ...

    def neighbours(
        self,
        cell_id: str,
        *,
        grid_distance: int = 1,
        include_origin: bool = False,
    ) -> tuple[str, ...]: ...

    def center(self, cell_id: str) -> GeographicCoordinate: ...

    def is_valid(self, cell_id: object) -> bool: ...


class _H3CellGrid:
    name = "hierarchical_global_grid"

    def __init__(self, api: ModuleType) -> None:
        self._api = api

    @property
    def version(self) -> str:
        return f"h3:{self._api.__version__}"

    def coordinate_to_cell(self, coordinate: GeographicCoordinate, *, resolution: int) -> str:
        normalized_resolution = validate_resolution(resolution)
        try:
            return str(
                self._api.latlng_to_cell(
                    float(coordinate.latitude),
                    float(coordinate.longitude),
                    normalized_resolution,
                )
            )
        except (TypeError, ValueError) as exc:
            raise CellGridError(f"coordinate could not be assigned at resolution {normalized_resolution}") from exc

    def parent(self, cell_id: str, *, resolution: int | None = None) -> str:
        self._require_valid_cell(cell_id)
        current_resolution = int(self._api.get_resolution(cell_id))
        if current_resolution == 0:
            raise CellGridError("cell at resolution 0 has no parent")
        parent_resolution = current_resolution - 1 if resolution is None else validate_resolution(resolution)
        if parent_resolution >= current_resolution:
            raise ValueError(
                f"parent resolution must be coarser than cell resolution {current_resolution}"
            )
        try:
            return str(self._api.cell_to_parent(cell_id, parent_resolution))
        except (TypeError, ValueError) as exc:
            raise CellGridError(
                f"cell parent could not be computed at resolution {parent_resolution}"
            ) from exc

    def neighbours(
        self,
        cell_id: str,
        *,
        grid_distance: int = 1,
        include_origin: bool = False,
    ) -> tuple[str, ...]:
        self._require_valid_cell(cell_id)
        distance = validate_grid_distance(grid_distance)
        try:
            cells = {str(value) for value in self._api.grid_disk(cell_id, distance)}
        except (TypeError, ValueError) as exc:
            raise CellGridError(f"cell neighbours could not be computed at distance {distance}") from exc
        if not include_origin:
            cells.discard(cell_id)
        return tuple(sorted(cells))

    def center(self, cell_id: str) -> GeographicCoordinate:
        self._require_valid_cell(cell_id)
        try:
            latitude, longitude = self._api.cell_to_latlng(cell_id)
        except (TypeError, ValueError) as exc:
            raise CellGridError("cell center could not be computed") from exc
        return GeographicCoordinate(latitude=latitude, longitude=longitude)

    def is_valid(self, cell_id: object) -> bool:
        return isinstance(cell_id, str) and bool(self._api.is_valid_cell(cell_id))

    def _require_valid_cell(self, cell_id: object) -> None:
        if not self.is_valid(cell_id):
            raise CellGridError(f"invalid cell identifier: {cell_id!r}")


@lru_cache(maxsize=1)
def default_cell_grid() -> CellGrid:
    return _H3CellGrid(_load_h3())


def coordinate_to_cell(
    latitude: Real,
    longitude: Real,
    *,
    resolution: Integral,
    coordinate_uncertainty_m: Real | None = None,
    grid: CellGrid | None = None,
) -> str:
    coordinate = GeographicCoordinate(
        latitude=latitude,
        longitude=longitude,
        coordinate_uncertainty_m=coordinate_uncertainty_m,
    )
    return (grid or default_cell_grid()).coordinate_to_cell(
        coordinate,
        resolution=validate_resolution(resolution),
    )


def project_coordinate(
    coordinate: GeographicCoordinate,
    *,
    resolutions: GeographicResolutions,
    grid: CellGrid | None = None,
) -> CellProjection:
    if not isinstance(coordinate, GeographicCoordinate):
        raise TypeError("coordinate must be a GeographicCoordinate")
    if not isinstance(resolutions, GeographicResolutions):
        raise TypeError("resolutions must be GeographicResolutions")
    backend = grid or default_cell_grid()
    cells = tuple(
        CellAtResolution(
            resolution=resolution,
            cell_id=backend.coordinate_to_cell(coordinate, resolution=resolution),
        )
        for resolution in resolutions.values
    )
    return CellProjection(
        coordinate=coordinate,
        cells=cells,
        grid_name=backend.name,
        grid_version=backend.version,
    )


def cell_parent(
    cell_id: str,
    *,
    resolution: Integral | None = None,
    grid: CellGrid | None = None,
) -> str:
    target_resolution = None if resolution is None else validate_resolution(resolution)
    return (grid or default_cell_grid()).parent(cell_id, resolution=target_resolution)


def neighbour_cells(
    cell_id: str,
    *,
    grid_distance: Integral = 1,
    include_origin: bool = False,
    grid: CellGrid | None = None,
) -> tuple[str, ...]:
    return (grid or default_cell_grid()).neighbours(
        cell_id,
        grid_distance=validate_grid_distance(grid_distance),
        include_origin=include_origin,
    )


def cell_center(cell_id: str, *, grid: CellGrid | None = None) -> GeographicCoordinate:
    return (grid or default_cell_grid()).center(cell_id)


def is_valid_cell(cell_id: object, *, grid: CellGrid | None = None) -> bool:
    return (grid or default_cell_grid()).is_valid(cell_id)


def _load_h3() -> ModuleType:
    try:
        return import_module("h3")
    except ModuleNotFoundError as exc:
        if exc.name != "h3":
            raise
        raise CellGridDependencyError(
            "hierarchical cell operations require the optional geo dependency"
        ) from exc


__all__ = [
    "CellGrid",
    "CellGridDependencyError",
    "CellGridError",
    "cell_center",
    "cell_parent",
    "coordinate_to_cell",
    "default_cell_grid",
    "is_valid_cell",
    "neighbour_cells",
    "project_coordinate",
]
