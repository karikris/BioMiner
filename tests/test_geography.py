from __future__ import annotations

import math

import pytest

from biominer.geography import (
    CellGridDependencyError,
    CellGridError,
    GeographicCoordinate,
    GeographicResolutions,
    cell_center,
    cell_parent,
    coordinate_to_cell,
    great_circle_distance_km,
    is_valid_cell,
    neighbour_cells,
    project_coordinate,
)
from biominer.geography import cells as cell_module


def test_coordinate_validation_rejects_invalid_values() -> None:
    for latitude in (-90.0001, 90.0001, math.nan, math.inf, -math.inf, True):
        with pytest.raises(ValueError, match="latitude"):
            GeographicCoordinate(latitude=latitude, longitude=0.0)

    for longitude in (math.nan, math.inf, -math.inf, False):
        with pytest.raises(ValueError, match="longitude"):
            GeographicCoordinate(latitude=0.0, longitude=longitude)

    for uncertainty in (-0.1, math.nan, math.inf, True):
        with pytest.raises(ValueError, match="coordinate_uncertainty_m"):
            GeographicCoordinate(
                latitude=0.0,
                longitude=0.0,
                coordinate_uncertainty_m=uncertainty,
            )


@pytest.mark.parametrize(
    ("longitude", "expected"),
    [
        (-540.0, -180.0),
        (-181.0, 179.0),
        (-180.0, -180.0),
        (180.0, -180.0),
        (181.0, -179.0),
        (540.0, -180.0),
    ],
)
def test_coordinate_normalizes_longitude(longitude: float, expected: float) -> None:
    coordinate = GeographicCoordinate(latitude=0.0, longitude=longitude)
    assert coordinate.longitude == expected


def test_resolution_configuration_requires_strict_coarse_to_local_order() -> None:
    assert GeographicResolutions(coarse=3, regional=5, local=7).values == (3, 5, 7)

    for values in ((5, 5, 7), (7, 5, 3), (-1, 5, 7), (3, 5, 16), (True, 5, 7)):
        with pytest.raises(ValueError, match="resolution"):
            GeographicResolutions(coarse=values[0], regional=values[1], local=values[2])


def test_coordinate_to_cell_matches_upstream_basic_string_api() -> None:
    cell = coordinate_to_cell(37.7752702151959, -122.418307270836, resolution=9)
    assert cell == "8928308280fffff"
    assert is_valid_cell(cell)
    assert coordinate_to_cell(0.0, 181.0, resolution=6) == coordinate_to_cell(
        0.0,
        -179.0,
        resolution=6,
    )


def test_projection_retains_uncertainty_and_hierarchy() -> None:
    coordinate = GeographicCoordinate(
        latitude=-27.4705,
        longitude=153.0260,
        coordinate_uncertainty_m=125.0,
    )
    resolutions = GeographicResolutions(coarse=3, regional=5, local=7)

    projection = project_coordinate(coordinate, resolutions=resolutions)

    assert projection.coordinate == coordinate
    assert projection.coordinate.coordinate_uncertainty_m == 125.0
    assert tuple(item.resolution for item in projection.cells) == (3, 5, 7)
    assert projection.cell_at(3) == cell_parent(projection.cell_at(7), resolution=3)
    assert projection.cell_at(5) == cell_parent(projection.cell_at(7), resolution=5)
    assert projection.grid_name == "hierarchical_global_grid"
    assert projection.grid_version.startswith("h3:")


def test_parent_rejects_root_same_and_finer_resolutions() -> None:
    root = coordinate_to_cell(0.0, 0.0, resolution=0)
    child = coordinate_to_cell(0.0, 0.0, resolution=7)

    with pytest.raises(CellGridError, match="resolution 0"):
        cell_parent(root)
    with pytest.raises(ValueError, match="coarser"):
        cell_parent(child, resolution=7)
    with pytest.raises(ValueError, match="coarser"):
        cell_parent(child, resolution=8)


def test_neighbours_are_sorted_deterministic_and_exclude_origin() -> None:
    cell = coordinate_to_cell(37.7752702151959, -122.418307270836, resolution=9)

    first = neighbour_cells(cell)
    second = neighbour_cells(cell)

    assert first == second == tuple(sorted(first))
    assert len(first) == 6
    assert cell not in first
    assert neighbour_cells(cell, grid_distance=0) == ()
    assert neighbour_cells(cell, grid_distance=0, include_origin=True) == (cell,)


def test_pentagon_neighbours_do_not_assume_six_adjacent_cells() -> None:
    pentagon = "821c07fffffffff"
    assert len(neighbour_cells(pentagon)) == 5


def test_cell_center_and_invalid_cell_behavior() -> None:
    cell = coordinate_to_cell(-27.4705, 153.0260, resolution=7)
    center = cell_center(cell)

    assert -90.0 <= center.latitude <= 90.0
    assert -180.0 <= center.longitude < 180.0
    assert center.coordinate_uncertainty_m is None
    assert not is_valid_cell("not-a-cell")
    with pytest.raises(CellGridError, match="invalid cell"):
        cell_center("not-a-cell")


def test_great_circle_distance_handles_identity_symmetry_and_dateline() -> None:
    lyon = GeographicCoordinate(45.7597, 4.8422)
    paris = GeographicCoordinate(48.8567, 2.3508)
    east = GeographicCoordinate(0.0, 179.7)
    west = GeographicCoordinate(0.0, -179.7)

    assert great_circle_distance_km(lyon, lyon) == 0.0
    assert great_circle_distance_km(lyon, paris) == pytest.approx(392.217, rel=2e-5)
    assert great_circle_distance_km(lyon, paris) == pytest.approx(
        great_circle_distance_km(paris, lyon)
    )
    assert great_circle_distance_km(east, west) == pytest.approx(66.717, rel=2e-5)


def test_missing_optional_dependency_fails_only_when_grid_is_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = cell_module.import_module

    def blocked_import(name: str):
        if name == "h3":
            error = ModuleNotFoundError("No module named 'h3'")
            error.name = "h3"
            raise error
        return real_import_module(name)

    monkeypatch.setattr(cell_module, "import_module", blocked_import)
    cell_module.default_cell_grid.cache_clear()
    try:
        with pytest.raises(CellGridDependencyError, match="optional geo dependency"):
            coordinate_to_cell(0.0, 0.0, resolution=3)
    finally:
        cell_module.default_cell_grid.cache_clear()
