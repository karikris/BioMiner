from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
from statistics import median
from typing import Any, Iterable

import polars as pl

from biominer.geo.grid import GEO_GRID_LEVELS, geocell_id
from biominer.storage.parquet import write_parquet


def build_geo_candidate_tables(
    occurrences: pl.DataFrame,
    *,
    output_dir: str | Path,
    geo_version: str,
    grid_levels: Iterable[str] = ("G2_20deg", "G3_10deg", "G4_5deg", "G5_2deg", "G6_1deg"),
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference = _occurrence_reference(occurrences, geo_version=geo_version)
    grid_cells, species_index = _species_index(reference, geo_version=geo_version, grid_levels=tuple(grid_levels))
    candidate_sets = species_index.select(
        [
            "geo_version",
            "geocell_id",
            "grid_level",
            "species_key",
            "scientific_name",
            "family",
            "genus",
            "occurrence_count",
            "candidate_rank_prior",
            "provenance_json",
        ]
    )
    manifest = {
        "geo_version": geo_version,
        "grid_levels": list(grid_levels),
        "occurrence_rows": reference.height,
        "species_index_rows": species_index.height,
    }
    outputs = {
        "gbif_occurrence_reference": output / "gbif_occurrence_reference.parquet",
        "geo_grid_cells": output / "geo_grid_cells.parquet",
        "geo_species_index": output / "geo_species_index.parquet",
        "geo_candidate_sets": output / "geo_candidate_sets.parquet",
        "manifest": output / "manifest.json",
    }
    write_parquet(reference, outputs["gbif_occurrence_reference"])
    write_parquet(grid_cells, outputs["geo_grid_cells"])
    write_parquet(species_index, outputs["geo_species_index"])
    write_parquet(candidate_sets, outputs["geo_candidate_sets"])
    outputs["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return outputs


def _occurrence_reference(frame: pl.DataFrame, *, geo_version: str) -> pl.DataFrame:
    rows = []
    for row in frame.to_dicts():
        latitude = _float_value(row, "decimalLatitude", "latitude")
        longitude = _float_value(row, "decimalLongitude", "longitude")
        if latitude is None or longitude is None:
            continue
        rows.append(
            {
                "geo_version": geo_version,
                "species_key": str(_first_value(row, "species_key", "speciesKey", "taxonKey") or ""),
                "scientific_name": str(_first_value(row, "scientific_name", "scientificName", "species") or ""),
                "family": _text_value(row, "family"),
                "genus": _text_value(row, "genus"),
                "decimal_latitude": latitude,
                "decimal_longitude": longitude,
                "coordinate_uncertainty_meters": _float_value(row, "coordinateUncertaintyInMeters", "coordinate_uncertainty_meters"),
                "basis_of_record": _text_value(row, "basisOfRecord", "basis_of_record"),
                "dataset_key": _text_value(row, "datasetKey", "dataset_key"),
                "year": _int_value(row, "year"),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "geo_version": pl.Utf8,
            "species_key": pl.Utf8,
            "scientific_name": pl.Utf8,
            "family": pl.Utf8,
            "genus": pl.Utf8,
            "decimal_latitude": pl.Float64,
            "decimal_longitude": pl.Float64,
            "coordinate_uncertainty_meters": pl.Float64,
            "basis_of_record": pl.Utf8,
            "dataset_key": pl.Utf8,
            "year": pl.Int64,
        }
    )


def _species_index(reference: pl.DataFrame, *, geo_version: str, grid_levels: tuple[str, ...]) -> tuple[pl.DataFrame, pl.DataFrame]:
    aggregates: dict[tuple[str, str, str], dict[str, Any]] = {}
    grid_cell_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in reference.to_dicts():
        species_key = str(row["species_key"])
        if not species_key:
            continue
        for level in ("G0_world", *grid_levels):
            cell_id = geocell_id(level, float(row["decimal_latitude"]), float(row["decimal_longitude"]))
            grid_cell_rows[(level, cell_id)] = {"geo_version": geo_version, "grid_level": level, "geocell_id": cell_id}
            key = (level, cell_id, species_key)
            aggregate = aggregates.setdefault(
                key,
                {
                    "geo_version": geo_version,
                    "geocell_id": cell_id,
                    "grid_level": level,
                    "species_key": species_key,
                    "scientific_name": row["scientific_name"],
                    "family": row["family"],
                    "genus": row["genus"],
                    "occurrence_count": 0,
                    "record_count_weighted": 0.0,
                    "years": [],
                    "basis": Counter(),
                    "uncertainties": [],
                    "datasets": set(),
                },
            )
            aggregate["occurrence_count"] += 1
            aggregate["record_count_weighted"] += 1.0
            if row.get("year") is not None:
                aggregate["years"].append(int(row["year"]))
            if row.get("basis_of_record"):
                aggregate["basis"][str(row["basis_of_record"])] += 1
            if row.get("coordinate_uncertainty_meters") is not None:
                aggregate["uncertainties"].append(float(row["coordinate_uncertainty_meters"]))
            if row.get("dataset_key"):
                aggregate["datasets"].add(str(row["dataset_key"]))
    totals_by_cell: dict[tuple[str, str], int] = defaultdict(int)
    for (level, cell_id, _species_key), aggregate in aggregates.items():
        totals_by_cell[(level, cell_id)] += int(aggregate["occurrence_count"])
    species_rows = []
    for (level, cell_id, _species_key), aggregate in aggregates.items():
        total = totals_by_cell[(level, cell_id)]
        years = aggregate.pop("years")
        basis = aggregate.pop("basis")
        uncertainties = aggregate.pop("uncertainties")
        datasets = aggregate.pop("datasets")
        species_rows.append(
            {
                **aggregate,
                "first_year": min(years) if years else None,
                "last_year": max(years) if years else None,
                "basis_of_record_counts": json.dumps(dict(sorted(basis.items())), sort_keys=True),
                "coordinate_uncertainty_summary": json.dumps(_uncertainty_summary(uncertainties), sort_keys=True),
                "source_dataset_count": len(datasets),
                "candidate_rank_prior": float(aggregate["occurrence_count"]) / float(total) if total else 0.0,
                "provenance_json": json.dumps({"source": "gbif_occurrence_reference", "geo_version": geo_version}, sort_keys=True),
            }
        )
    grid_cells = pl.DataFrame(sorted(grid_cell_rows.values(), key=lambda row: (row["grid_level"], row["geocell_id"])))
    species_index = pl.DataFrame(species_rows).sort(["grid_level", "geocell_id", "scientific_name"]) if species_rows else _empty_species_index()
    return grid_cells, species_index


def _empty_species_index() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "geo_version": pl.Utf8,
            "geocell_id": pl.Utf8,
            "grid_level": pl.Utf8,
            "species_key": pl.Utf8,
            "scientific_name": pl.Utf8,
            "family": pl.Utf8,
            "genus": pl.Utf8,
            "occurrence_count": pl.Int64,
            "record_count_weighted": pl.Float64,
            "first_year": pl.Int64,
            "last_year": pl.Int64,
            "basis_of_record_counts": pl.Utf8,
            "coordinate_uncertainty_summary": pl.Utf8,
            "source_dataset_count": pl.Int64,
            "candidate_rank_prior": pl.Float64,
            "provenance_json": pl.Utf8,
        }
    )


def _uncertainty_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "median": median(values) if values else None,
        "max": max(values) if values else None,
    }


def _first_value(row: dict[str, Any], *keys: str) -> object | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _text_value(row: dict[str, Any], *keys: str) -> str | None:
    value = _first_value(row, *keys)
    return str(value) if value not in (None, "") else None


def _float_value(row: dict[str, Any], *keys: str) -> float | None:
    value = _first_value(row, *keys)
    return float(value) if value not in (None, "") else None


def _int_value(row: dict[str, Any], *keys: str) -> int | None:
    value = _first_value(row, *keys)
    return int(value) if value not in (None, "") else None
