from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
import math
from numbers import Integral
from pathlib import Path
from typing import Any

import polars as pl

from biominer.flickr_fetch.geographic_clustering import (
    GLOBAL_FALLBACK_CLUSTER_IDS,
)
from biominer.geography import CellGrid, default_cell_grid
from biominer.storage.parquet import write_parquet


REGIONAL_TAXON_OCCURRENCE_SCHEMA_VERSION = "regional-taxon-occurrence-v1.0.0"
REGIONAL_SCOPE_MEMBERSHIP_SCHEMA_VERSION = "regional-scope-membership-v1.1.0"
COORDINATE_CONFIDENCE_POLICY_VERSION = "inverse-uncertainty-100km-v1.0.0"
REGIONAL_TAXON_OCCURRENCE_FILE = "regional_taxon_occurrence.parquet"
DEFAULT_SPATIAL_RESOLUTION = 5
DEFAULT_COORDINATE_CONFIDENCE_SCALE_M = 100_000.0

_SCOPE_TYPES = frozenset(
    {"geo_cluster", "spatial_cell", "country", "bioregion", "global"}
)
_OVERLAP_PRIORITY = {
    "exact": 0,
    "buffer": 1,
    "country": 2,
    "bioregion": 3,
    "global": 4,
}
_OVERLAP_BY_PRIORITY = {priority: name for name, priority in _OVERLAP_PRIORITY.items()}


@dataclass(frozen=True, slots=True)
class _Taxon:
    accepted_taxon_key: str
    scientific_name: str
    family: str
    genus: str
    subfamily: str | None = None
    tribe: str | None = None


@dataclass(frozen=True, slots=True)
class _Occurrence:
    identity: tuple[str, str, str]
    source: str
    source_dataset_key: str
    source_record_id: str
    accepted_taxon_key: str
    spatial_cell_id: str
    spatial_resolution: int
    country_code: str | None
    bioregion: str | None
    event_date: date | None
    coordinate_uncertainty_m: float | None


@dataclass(frozen=True, slots=True)
class _ScopeMembership:
    regional_scope_id: str
    regional_scope_type: str
    overlap_type: str
    spatial_cell_id: str | None
    spatial_resolution: int | None
    country_code: str | None
    bioregion: str | None

    @property
    def priority(self) -> int:
        return _OVERLAP_PRIORITY[self.overlap_type]


def regional_taxon_occurrence_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "regional_scope_id": pl.String,
        "regional_scope_type": pl.String,
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "family": pl.String,
        "subfamily": pl.String,
        "tribe": pl.String,
        "genus": pl.String,
        "occurrence_count": pl.UInt64,
        "independent_dataset_count": pl.UInt64,
        "earliest_occurrence_date": pl.Date,
        "latest_occurrence_date": pl.Date,
        "coordinate_confidence": pl.Float32,
        "overlap_type": pl.String,
        "source": pl.String,
        "source_dataset_keys": pl.List(pl.String),
        "evidence_version": pl.String,
        "registry_version": pl.String,
    }


def regional_scope_membership_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "regional_scope_id": pl.String,
        "regional_scope_type": pl.String,
        "overlap_type": pl.String,
        "spatial_cell_id": pl.String,
        "spatial_resolution": pl.UInt8,
        "country_code": pl.String,
        "bioregion": pl.String,
    }


def build_flickr_cluster_scope_memberships(
    clusters: pl.DataFrame,
    *,
    buffer_grid_distance: int = 1,
    include_country_fallback: bool = True,
    include_global_fallback: bool = True,
    grid: CellGrid | None = None,
) -> pl.DataFrame:
    if not isinstance(clusters, pl.DataFrame):
        raise TypeError("clusters must be a Polars DataFrame")
    _require_columns(
        clusters,
        artifact="flickr_geo_clusters",
        required={
            "geo_cluster_id",
            "member_cell_ids",
            "source_resolution",
            "countries",
            "candidate_distribution_only",
        },
    )
    if isinstance(buffer_grid_distance, bool) or not isinstance(buffer_grid_distance, int):
        raise TypeError("buffer_grid_distance must be an integer")
    if buffer_grid_distance < 0:
        raise ValueError("buffer_grid_distance must be non-negative")
    backend = grid or default_cell_grid()
    rows: list[dict[str, object]] = []
    seen_cluster_ids: set[str] = set()

    for row in clusters.sort("geo_cluster_id").iter_rows(named=True):
        cluster_id = _required_text(row.get("geo_cluster_id"), field="geo_cluster_id")
        if cluster_id in seen_cluster_ids:
            raise ValueError(f"duplicate Flickr geo cluster ID: {cluster_id}")
        seen_cluster_ids.add(cluster_id)
        if row.get("candidate_distribution_only") is not True:
            raise ValueError(
                f"Flickr geo cluster {cluster_id!r} is not marked candidate_distribution_only"
            )
        resolution = _resolution(row.get("source_resolution"), field="source_resolution")
        member_cells = sorted(
            {
                _required_text(value, field="member_cell_ids")
                for value in _sequence(row.get("member_cell_ids"), field="member_cell_ids")
            }
        )
        for cell_id in member_cells:
            if not backend.is_valid(cell_id):
                raise ValueError(f"Flickr geo cluster {cluster_id!r} has invalid cell {cell_id!r}")

        if cluster_id in GLOBAL_FALLBACK_CLUSTER_IDS:
            if member_cells:
                raise ValueError("fallback clusters cannot contain spatial cells")
            if include_global_fallback:
                rows.append(_scope_row(cluster_id, "geo_cluster", "global"))
            continue
        if not member_cells:
            raise ValueError(f"located Flickr geo cluster {cluster_id!r} has no member cells")

        for cell_id in member_cells:
            rows.append(
                _scope_row(
                    cluster_id,
                    "geo_cluster",
                    "exact",
                    spatial_cell_id=cell_id,
                    spatial_resolution=resolution,
                )
            )
        if buffer_grid_distance:
            buffer_cells = {
                neighbour
                for cell_id in member_cells
                for neighbour in backend.neighbours(
                    cell_id,
                    grid_distance=buffer_grid_distance,
                )
            } - set(member_cells)
            for cell_id in sorted(buffer_cells):
                rows.append(
                    _scope_row(
                        cluster_id,
                        "geo_cluster",
                        "buffer",
                        spatial_cell_id=cell_id,
                        spatial_resolution=resolution,
                    )
                )
        if include_country_fallback:
            for country_code in sorted(
                {
                    _country_code(value)
                    for value in _sequence(row.get("countries"), field="countries")
                }
                - {None}
            ):
                rows.append(
                    _scope_row(
                        cluster_id,
                        "geo_cluster",
                        "country",
                        country_code=country_code,
                    )
                )

    return _membership_frame(rows)


def build_regional_taxon_occurrence_index(
    occurrence_evidence: pl.DataFrame,
    taxa: pl.DataFrame,
    *,
    evidence_version: str,
    registry_version: str,
    spatial_resolution: int = DEFAULT_SPATIAL_RESOLUTION,
    scope_memberships: pl.DataFrame | None = None,
    classification_paths: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if not isinstance(occurrence_evidence, pl.DataFrame):
        raise TypeError("occurrence_evidence must be a Polars DataFrame")
    if not isinstance(taxa, pl.DataFrame):
        raise TypeError("taxa must be a Polars DataFrame")
    source_version = _required_text(evidence_version, field="evidence_version")
    version = f"{source_version}+{COORDINATE_CONFIDENCE_POLICY_VERSION}"
    registry = _required_text(registry_version, field="registry_version")
    resolution = _resolution(spatial_resolution, field="spatial_resolution")
    taxonomy = _accepted_species_taxonomy(taxa, classification_paths=classification_paths)
    occurrences = _credible_occurrences(
        occurrence_evidence,
        spatial_resolution=resolution,
    )
    unknown_keys = sorted(
        {occurrence.accepted_taxon_key for occurrence in occurrences} - set(taxonomy)
    )
    if unknown_keys:
        raise ValueError(
            "eligible occurrence keys are not accepted species in the registry: "
            + ", ".join(unknown_keys)
        )

    if scope_memberships is None:
        memberships = tuple(
            _ScopeMembership(
                regional_scope_id=cell_id,
                regional_scope_type="spatial_cell",
                overlap_type="exact",
                spatial_cell_id=cell_id,
                spatial_resolution=resolution,
                country_code=None,
                bioregion=None,
            )
            for cell_id in sorted({item.spatial_cell_id for item in occurrences})
        )
    else:
        memberships = _validated_scope_memberships(scope_memberships)

    matched = _match_occurrences_to_scopes(occurrences, memberships)
    output_rows: list[dict[str, object]] = []
    for group_key, records_by_priority in sorted(matched.items()):
        scope_id, scope_type, taxon_key, source = group_key
        strongest_priority = min(records_by_priority)
        selected = tuple(records_by_priority[strongest_priority].values())
        taxon = taxonomy[taxon_key]
        uncertainties = [
            item.coordinate_uncertainty_m
            for item in selected
            if item.coordinate_uncertainty_m is not None
        ]
        dates = [item.event_date for item in selected if item.event_date is not None]
        datasets = sorted({item.source_dataset_key for item in selected})
        confidence = (
            sum(
                1.0
                / (1.0 + uncertainty / DEFAULT_COORDINATE_CONFIDENCE_SCALE_M)
                for uncertainty in uncertainties
            )
            / len(uncertainties)
            if uncertainties and len(uncertainties) == len(selected)
            else None
        )
        output_rows.append(
            {
                "schema_version": REGIONAL_TAXON_OCCURRENCE_SCHEMA_VERSION,
                "regional_scope_id": scope_id,
                "regional_scope_type": scope_type,
                "accepted_taxon_key": taxon.accepted_taxon_key,
                "scientific_name": taxon.scientific_name,
                "family": taxon.family,
                "subfamily": taxon.subfamily,
                "tribe": taxon.tribe,
                "genus": taxon.genus,
                "occurrence_count": len(selected),
                "independent_dataset_count": len(datasets),
                "earliest_occurrence_date": min(dates) if dates else None,
                "latest_occurrence_date": max(dates) if dates else None,
                "coordinate_confidence": confidence,
                "overlap_type": _OVERLAP_BY_PRIORITY[strongest_priority],
                "source": source,
                "source_dataset_keys": datasets,
                "evidence_version": version,
                "registry_version": registry,
            }
        )
    return _occurrence_index_frame(output_rows)


def write_regional_taxon_occurrence(
    frame: pl.DataFrame,
    output: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a Polars DataFrame")
    schema = regional_taxon_occurrence_schema()
    if frame.schema != schema:
        raise ValueError("regional taxon occurrence frame does not match the physical schema")
    expected = frame.sort(
        ["regional_scope_id", "accepted_taxon_key", "source", "evidence_version"]
    )
    if not frame.equals(expected):
        raise ValueError("regional taxon occurrence frame is not in deterministic sort order")
    duplicates = frame.group_by(
        ["regional_scope_id", "accepted_taxon_key", "source", "evidence_version"]
    ).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise ValueError("regional taxon occurrence frame contains duplicate primary keys")
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= REGIONAL_TAXON_OCCURRENCE_FILE
    return write_parquet(frame, destination, overwrite=overwrite)


def _credible_occurrences(
    evidence: pl.DataFrame,
    *,
    spatial_resolution: int,
) -> tuple[_Occurrence, ...]:
    record_id_field = (
        "source_record_id"
        if "source_record_id" in evidence.columns
        else "gbif_id"
        if "gbif_id" in evidence.columns
        else None
    )
    if record_id_field is None:
        raise ValueError("occurrence evidence requires source_record_id or gbif_id")
    _require_columns(
        evidence,
        artifact="occurrence_evidence",
        required={
            "source",
            "source_dataset_key",
            "accepted_taxon_key",
            "spatial_cell_id",
            "spatial_resolution",
            "country_code",
            "bioregion",
            "coordinate_uncertainty_m",
            "event_date",
            "occurrence_status",
            "has_geospatial_issue",
            "preserved_specimen",
            "fossil",
            "range_inference_eligible",
            "taxon_key_match",
            "coordinate_valid",
        },
    )
    by_identity: dict[tuple[str, str, str], _Occurrence] = {}
    for row in evidence.iter_rows(named=True):
        if _resolution(row.get("spatial_resolution"), field="spatial_resolution") != spatial_resolution:
            continue
        eligible = _boolean(row.get("range_inference_eligible"), field="range_inference_eligible")
        if not eligible:
            continue
        contradictions: list[str] = []
        if not _boolean(row.get("taxon_key_match"), field="taxon_key_match"):
            contradictions.append("taxon_key_match=false")
        if not _boolean(row.get("coordinate_valid"), field="coordinate_valid"):
            contradictions.append("coordinate_valid=false")
        if str(row.get("occurrence_status") or "").strip().upper() != "PRESENT":
            contradictions.append("occurrence_status!=PRESENT")
        for field in ("has_geospatial_issue", "preserved_specimen", "fossil"):
            if _boolean(row.get(field), field=field):
                contradictions.append(f"{field}=true")
        if contradictions:
            record_id = str(row.get(record_id_field) or "<missing>")
            raise ValueError(
                f"occurrence {record_id!r} contradicts range_inference_eligible: "
                + ", ".join(contradictions)
            )

        source = _required_text(row.get("source"), field="source")
        dataset = _required_text(
            row.get("source_dataset_key"),
            field="source_dataset_key",
        )
        record_id = _required_text(row.get(record_id_field), field=record_id_field)
        cell_id = _required_text(row.get("spatial_cell_id"), field="spatial_cell_id")
        uncertainty = _optional_nonnegative_float(
            row.get("coordinate_uncertainty_m"),
            field="coordinate_uncertainty_m",
        )
        occurrence = _Occurrence(
            identity=(source.casefold(), dataset, record_id),
            source=source,
            source_dataset_key=dataset,
            source_record_id=record_id,
            accepted_taxon_key=_required_text(
                row.get("accepted_taxon_key"),
                field="accepted_taxon_key",
            ),
            spatial_cell_id=cell_id,
            spatial_resolution=spatial_resolution,
            country_code=_country_code(row.get("country_code")),
            bioregion=_optional_text(row.get("bioregion")),
            event_date=_optional_date(row.get("event_date")),
            coordinate_uncertainty_m=uncertainty,
        )
        previous = by_identity.get(occurrence.identity)
        if previous is not None and previous != occurrence:
            raise ValueError(
                "conflicting occurrence evidence for "
                f"{source}:{dataset}:{record_id}"
            )
        by_identity[occurrence.identity] = occurrence
    return tuple(
        sorted(
            by_identity.values(),
            key=lambda item: (
                item.source.casefold(),
                item.source_dataset_key,
                item.source_record_id,
            ),
        )
    )


def _accepted_species_taxonomy(
    taxa: pl.DataFrame,
    *,
    classification_paths: pl.DataFrame | None,
) -> dict[str, _Taxon]:
    _require_columns(
        taxa,
        artifact="taxa",
        required={
            "accepted_taxon_key",
            "scientific_name",
            "rank",
            "taxonomic_status",
            "family",
            "genus",
        },
    )
    reviewed = _reviewed_ranks(classification_paths)
    taxonomy: dict[str, _Taxon] = {}
    for row in taxa.iter_rows(named=True):
        if str(row.get("rank") or "").strip().upper() != "SPECIES":
            continue
        if str(row.get("taxonomic_status") or "").strip().upper() != "ACCEPTED":
            continue
        if "in_scope" in taxa.columns and row.get("in_scope") is not True:
            continue
        key = _required_text(row.get("accepted_taxon_key"), field="accepted_taxon_key")
        if key in taxonomy:
            raise ValueError(f"duplicate accepted species in taxa: {key}")
        scientific_name = _required_text(row.get("scientific_name"), field="scientific_name")
        family = _required_text(row.get("family"), field="family")
        genus = _required_text(row.get("genus"), field="genus")
        reviewed_row = reviewed.get(key)
        if reviewed_row is not None:
            for field, expected in (
                ("species", scientific_name),
                ("family", family),
                ("genus", genus),
            ):
                actual = _required_text(reviewed_row.get(field), field=field)
                if actual != expected:
                    raise ValueError(
                        f"classification path {key} {field} {actual!r} conflicts with "
                        f"accepted registry value {expected!r}"
                    )
        taxonomy[key] = _Taxon(
            accepted_taxon_key=key,
            scientific_name=scientific_name,
            family=family,
            genus=genus,
            subfamily=(
                _optional_text(reviewed_row.get("subfamily"))
                if reviewed_row is not None
                else None
            ),
            tribe=(
                _optional_text(reviewed_row.get("tribe"))
                if reviewed_row is not None
                else None
            ),
        )
    return taxonomy


def _reviewed_ranks(
    classification_paths: pl.DataFrame | None,
) -> dict[str, dict[str, Any]]:
    if classification_paths is None:
        return {}
    if not isinstance(classification_paths, pl.DataFrame):
        raise TypeError("classification_paths must be a Polars DataFrame or None")
    _require_columns(
        classification_paths,
        artifact="classification_paths",
        required={
            "accepted_taxon_key",
            "species",
            "family",
            "subfamily",
            "tribe",
            "genus",
            "enabled",
        },
    )
    rows: dict[str, dict[str, Any]] = {}
    for row in classification_paths.filter(pl.col("enabled")).iter_rows(named=True):
        key = _required_text(row.get("accepted_taxon_key"), field="accepted_taxon_key")
        if key in rows:
            raise ValueError(f"duplicate enabled classification path for {key}")
        rows[key] = row
    return rows


def _validated_scope_memberships(
    frame: pl.DataFrame,
) -> tuple[_ScopeMembership, ...]:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("scope_memberships must be a Polars DataFrame or None")
    _require_columns(
        frame,
        artifact="scope_memberships",
        required={
            "regional_scope_id",
            "regional_scope_type",
            "overlap_type",
            "spatial_cell_id",
            "spatial_resolution",
            "country_code",
            "bioregion",
        },
    )
    if "schema_version" in frame.columns:
        versions = sorted(
            {
                _required_text(value, field="schema_version")
                for value in frame["schema_version"].to_list()
            }
        )
        if versions != [REGIONAL_SCOPE_MEMBERSHIP_SCHEMA_VERSION]:
            raise ValueError(f"unsupported regional scope membership schema versions: {versions}")
    memberships: dict[tuple[object, ...], _ScopeMembership] = {}
    scope_types: dict[str, str] = {}
    for row in frame.iter_rows(named=True):
        scope_id = _required_text(row.get("regional_scope_id"), field="regional_scope_id")
        scope_type = _required_text(
            row.get("regional_scope_type"),
            field="regional_scope_type",
        )
        if scope_type not in _SCOPE_TYPES:
            raise ValueError(f"unsupported regional_scope_type: {scope_type}")
        previous_type = scope_types.setdefault(scope_id, scope_type)
        if previous_type != scope_type:
            raise ValueError(f"regional scope {scope_id!r} has conflicting scope types")
        overlap = _required_text(row.get("overlap_type"), field="overlap_type")
        if overlap not in _OVERLAP_PRIORITY:
            raise ValueError(f"unsupported overlap_type: {overlap}")
        cell_id = _optional_text(row.get("spatial_cell_id"))
        raw_resolution = row.get("spatial_resolution")
        country = _country_code(row.get("country_code"))
        bioregion = _optional_text(row.get("bioregion"))
        resolution = (
            _resolution(raw_resolution, field="spatial_resolution")
            if raw_resolution is not None
            else None
        )
        if overlap in {"exact", "buffer"}:
            if cell_id is None or resolution is None or country or bioregion:
                raise ValueError(
                    f"{overlap} membership requires only a spatial cell and resolution"
                )
        elif overlap == "country":
            if country is None or cell_id or resolution is not None or bioregion:
                raise ValueError("country membership requires only country_code")
        elif overlap == "bioregion":
            if bioregion is None or cell_id or resolution is not None or country:
                raise ValueError("bioregion membership requires only bioregion")
        elif cell_id or resolution is not None or country or bioregion:
            raise ValueError("global membership cannot contain a spatial selector")
        membership = _ScopeMembership(
            regional_scope_id=scope_id,
            regional_scope_type=scope_type,
            overlap_type=overlap,
            spatial_cell_id=cell_id,
            spatial_resolution=resolution,
            country_code=country,
            bioregion=bioregion,
        )
        key = (
            scope_id,
            scope_type,
            overlap,
            cell_id,
            resolution,
            country,
            bioregion,
        )
        memberships[key] = membership
    return tuple(
        sorted(
            memberships.values(),
            key=lambda item: (
                item.regional_scope_id,
                item.priority,
                item.spatial_resolution if item.spatial_resolution is not None else -1,
                item.spatial_cell_id or "",
                item.country_code or "",
                item.bioregion or "",
            ),
        )
    )


def _match_occurrences_to_scopes(
    occurrences: Sequence[_Occurrence],
    memberships: Sequence[_ScopeMembership],
) -> dict[
    tuple[str, str, str, str],
    dict[int, dict[tuple[str, str, str], _Occurrence]],
]:
    by_cell: dict[tuple[int, str], list[_ScopeMembership]] = defaultdict(list)
    by_country: dict[str, list[_ScopeMembership]] = defaultdict(list)
    by_bioregion: dict[str, list[_ScopeMembership]] = defaultdict(list)
    global_memberships: list[_ScopeMembership] = []
    for membership in memberships:
        if membership.overlap_type in {"exact", "buffer"}:
            by_cell[(int(membership.spatial_resolution), str(membership.spatial_cell_id))].append(
                membership
            )
        elif membership.overlap_type == "country":
            by_country[str(membership.country_code)].append(membership)
        elif membership.overlap_type == "bioregion":
            by_bioregion[str(membership.bioregion)].append(membership)
        else:
            global_memberships.append(membership)

    matched: dict[
        tuple[str, str, str, str],
        dict[int, dict[tuple[str, str, str], _Occurrence]],
    ] = defaultdict(lambda: defaultdict(dict))
    for occurrence in occurrences:
        candidate_memberships = [
            *by_cell[(occurrence.spatial_resolution, occurrence.spatial_cell_id)],
            *by_country.get(occurrence.country_code or "", []),
            *by_bioregion.get(occurrence.bioregion or "", []),
            *global_memberships,
        ]
        strongest_by_scope: dict[tuple[str, str], _ScopeMembership] = {}
        for membership in candidate_memberships:
            scope_key = (membership.regional_scope_id, membership.regional_scope_type)
            previous = strongest_by_scope.get(scope_key)
            if previous is None or membership.priority < previous.priority:
                strongest_by_scope[scope_key] = membership
        for membership in strongest_by_scope.values():
            group_key = (
                membership.regional_scope_id,
                membership.regional_scope_type,
                occurrence.accepted_taxon_key,
                occurrence.source,
            )
            matched[group_key][membership.priority][occurrence.identity] = occurrence
    return matched


def _scope_row(
    scope_id: str,
    scope_type: str,
    overlap_type: str,
    *,
    spatial_cell_id: str | None = None,
    spatial_resolution: int | None = None,
    country_code: str | None = None,
    bioregion: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": REGIONAL_SCOPE_MEMBERSHIP_SCHEMA_VERSION,
        "regional_scope_id": scope_id,
        "regional_scope_type": scope_type,
        "overlap_type": overlap_type,
        "spatial_cell_id": spatial_cell_id,
        "spatial_resolution": spatial_resolution,
        "country_code": country_code,
        "bioregion": bioregion,
    }


def _membership_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = regional_scope_membership_schema()
    if not rows:
        return pl.DataFrame(schema=schema)
    return (
        pl.DataFrame(rows, schema=schema)
        .unique(maintain_order=False)
        .sort(
            [
                "regional_scope_id",
                "overlap_type",
                "spatial_resolution",
                "spatial_cell_id",
                "country_code",
                "bioregion",
            ],
            nulls_last=True,
        )
    )


def _occurrence_index_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = regional_taxon_occurrence_schema()
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema).sort(
        ["regional_scope_id", "accepted_taxon_key", "source", "evidence_version"]
    )


def _require_columns(
    frame: pl.DataFrame,
    *,
    artifact: str,
    required: set[str],
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{artifact} is missing required columns: {missing}")


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank")
    return text


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _country_code(value: object) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    code = text.upper()
    if len(code) != 2 or not code.isalpha():
        raise ValueError(f"country_code must be ISO alpha-2 or null, got {value!r}")
    return code


def _resolution(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    resolution = int(value)
    if resolution < 0 or resolution > 15:
        raise ValueError(f"{field} must be between 0 and 15")
    return resolution


def _optional_nonnegative_float(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be numeric or null") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be boolean")
    return value


def _optional_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError as exc:
            raise ValueError(f"event_date must be ISO 8601 or null, got {value!r}") from exc
    raise TypeError("event_date must be a date, datetime, ISO string, or null")


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise TypeError(f"{field} must be a sequence")


__all__ = [
    "COORDINATE_CONFIDENCE_POLICY_VERSION",
    "DEFAULT_COORDINATE_CONFIDENCE_SCALE_M",
    "DEFAULT_SPATIAL_RESOLUTION",
    "REGIONAL_SCOPE_MEMBERSHIP_SCHEMA_VERSION",
    "REGIONAL_TAXON_OCCURRENCE_FILE",
    "REGIONAL_TAXON_OCCURRENCE_SCHEMA_VERSION",
    "build_flickr_cluster_scope_memberships",
    "build_regional_taxon_occurrence_index",
    "regional_scope_membership_schema",
    "regional_taxon_occurrence_schema",
    "write_regional_taxon_occurrence",
]
