"""Backend-neutral geographic cell and distance primitives."""

from biominer.geography.cells import (
    CellGrid,
    CellGridDependencyError,
    CellGridError,
    cell_center,
    cell_parent,
    coordinate_to_cell,
    default_cell_grid,
    is_valid_cell,
    neighbour_cells,
    project_coordinate,
)
from biominer.geography.distance import great_circle_distance_km
from biominer.geography.schemas import (
    CellAtResolution,
    CellProjection,
    GeographicCoordinate,
    GeographicResolutions,
)

__all__ = [
    "CellAtResolution",
    "CellGrid",
    "CellGridDependencyError",
    "CellGridError",
    "CellProjection",
    "GeographicCoordinate",
    "GeographicResolutions",
    "cell_center",
    "cell_parent",
    "coordinate_to_cell",
    "default_cell_grid",
    "great_circle_distance_km",
    "is_valid_cell",
    "neighbour_cells",
    "project_coordinate",
]
