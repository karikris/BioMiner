from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import polars as pl

from biominer.geography import CellGrid, default_cell_grid
from biominer.geography.validation import validate_grid_distance, validate_resolution
from biominer.registry.geographic_spread import (
    geographic_occurrence_evidence_schema,
    geographic_spread_schema,
)
from biominer.storage.parquet import write_parquet


GEOGRAPHIC_SUMMARY_SCHEMA_VERSION = "taxon-geographic-summary-v1.0.0"
GEOGRAPHIC_SUMMARY_POLICY_VERSION = "geographic-summary-policy-v1.0.0"
GEOGRAPHIC_SUMMARY_MANIFEST_SCHEMA_VERSION = "geographic-summary-build-v1.0.0"
TAXON_GEOGRAPHIC_SUMMARY_FILE = "taxon_geographic_summary.parquet"
GEOGRAPHIC_QA_FINDINGS_FILE = "geographic_qa_findings.parquet"
GEOGRAPHIC_SUMMARY_MANIFEST_FILE = "geographic_summary_manifest.json"

_SPREAD_PRIMARY_KEY = (
    "accepted_taxon_key",
    "source",
    "source_dataset_key",
    "spatial_resolution",
    "spatial_cell_id",
    "known_range_role",
    "source_snapshot_version",
)
_KNOWN_RANGE_ROLES = frozenset({"native", "introduced", "vagrant", "uncertain", "unknown"})


@dataclass(frozen=True, slots=True)
class GeographicSummaryPolicy:
    component_resolution: int | None = None
    min_eligible_occurrences: int = 5
    min_occupied_cells: int = 2
    outlier_max_eligible_occurrences: int = 1
    outlier_neighbour_distance: int = 3
    current_window_years: int = 20

    def __post_init__(self) -> None:
        if self.component_resolution is not None:
            object.__setattr__(
                self,
                "component_resolution",
                validate_resolution(self.component_resolution),
            )
        for name in ("min_eligible_occurrences", "min_occupied_cells"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.outlier_max_eligible_occurrences, bool)
            or not isinstance(self.outlier_max_eligible_occurrences, int)
            or self.outlier_max_eligible_occurrences < 0
        ):
            raise ValueError("outlier_max_eligible_occurrences must be a non-negative integer")
        object.__setattr__(
            self,
            "outlier_neighbour_distance",
            validate_grid_distance(self.outlier_neighbour_distance),
        )
        if (
            isinstance(self.current_window_years, bool)
            or not isinstance(self.current_window_years, int)
            or self.current_window_years < 1
        ):
            raise ValueError("current_window_years must be a positive integer")

    def manifest(self) -> dict[str, object]:
        return {"policy_version": GEOGRAPHIC_SUMMARY_POLICY_VERSION, **asdict(self)}


@dataclass(frozen=True, slots=True)
class GeographicSummaryBuildResult:
    summary: pl.DataFrame
    qa: pl.DataFrame
    manifest: dict[str, object]


def geographic_summary_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "registry_version": pl.String,
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "geographic_evidence_version": pl.String,
        "cell_counts_by_resolution": pl.List(
            pl.Struct({"resolution": pl.UInt8, "count": pl.UInt64})
        ),
        "countries": pl.List(pl.String),
        "admin_regions": pl.List(pl.String),
        "occupied_envelope": pl.Struct(
            {
                "south": pl.Float64,
                "north": pl.Float64,
                "west": pl.Float64,
                "east": pl.Float64,
                "crosses_dateline": pl.Boolean,
            }
        ),
        "disconnected_range_component_count": pl.UInt32,
        "occurrence_density_summary": pl.Struct(
            {
                "min": pl.Float64,
                "p50": pl.Float64,
                "p95": pl.Float64,
                "max": pl.Float64,
            }
        ),
        "data_deficient": pl.Boolean,
        "data_deficient_reasons": pl.List(pl.String),
        "suspicious_outlier_cell_count": pl.UInt64,
        "range_source_coverage": pl.List(
            pl.Struct(
                {
                    "source": pl.String,
                    "dataset_count": pl.UInt64,
                    "eligible_occurrence_count": pl.UInt64,
                }
            )
        ),
        "known_introduced_regions": pl.List(pl.String),
        "current_evidence_count": pl.UInt64,
        "historical_evidence_count": pl.UInt64,
        "spread_fingerprint": pl.String,
        "created_at": pl.Datetime("us", "UTC"),
    }


def geographic_qa_schema() -> dict[str, pl.DataType]:
    return {"severity": pl.String, "code": pl.String, "subject": pl.String}


def build_geographic_summary(
    *,
    spread: pl.DataFrame,
    occurrence_evidence: pl.DataFrame,
    taxa: pl.DataFrame,
    registry_version: str,
    policy: GeographicSummaryPolicy,
    output_dir: str | Path,
    created_at: str | datetime,
    grid: CellGrid | None = None,
) -> GeographicSummaryBuildResult:
    _require_schema(spread, geographic_spread_schema(), name="geographic spread")
    _require_schema(
        occurrence_evidence,
        geographic_occurrence_evidence_schema(),
        name="geographic occurrence evidence",
    )
    if not isinstance(policy, GeographicSummaryPolicy):
        raise TypeError("policy must be GeographicSummaryPolicy")
    registry = _required_text(registry_version, field_name="registry_version")
    created = _utc_datetime(created_at)
    backend = grid or default_cell_grid()
    identities = _taxon_identities(
        taxa=taxa,
        spread=spread,
        occurrence_evidence=occurrence_evidence,
        registry_version=registry,
    )
    spread_rows, structural_findings = _validated_spread_rows(
        spread,
        registry_version=registry,
        grid=backend,
    )
    evidence_rows = _validated_evidence_rows(
        occurrence_evidence,
        registry_version=registry,
    )
    spread_by_taxon = _rows_by_taxon(spread_rows)
    evidence_by_taxon = _rows_by_taxon(evidence_rows)

    findings = list(structural_findings)
    summary_rows: list[dict[str, object]] = []
    for taxon_key in sorted(identities):
        scientific_name = identities[taxon_key]
        taxon_spread = spread_by_taxon.get(taxon_key, [])
        taxon_evidence = evidence_by_taxon.get(taxon_key, [])
        row, taxon_findings = _summarize_taxon(
            accepted_taxon_key=taxon_key,
            scientific_name=scientific_name,
            registry_version=registry,
            spread_rows=taxon_spread,
            evidence_rows=taxon_evidence,
            policy=policy,
            created_at=created,
            grid=backend,
        )
        summary_rows.append(row)
        findings.extend(taxon_findings)

    summary = _typed_frame(summary_rows, geographic_summary_schema()).sort(
        ["accepted_taxon_key", "geographic_evidence_version"]
    )
    qa = _qa_frame(findings)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = write_parquet(summary, output / TAXON_GEOGRAPHIC_SUMMARY_FILE)
    qa_path = write_parquet(qa, output / GEOGRAPHIC_QA_FINDINGS_FILE)
    manifest = _manifest(
        summary=summary,
        qa=qa,
        summary_path=summary_path,
        qa_path=qa_path,
        registry_version=registry,
        policy=policy,
        created_at=created,
        grid=backend,
    )
    _write_json_atomic(manifest, output / GEOGRAPHIC_SUMMARY_MANIFEST_FILE)
    return GeographicSummaryBuildResult(summary=summary, qa=qa, manifest=manifest)


def _summarize_taxon(
    *,
    accepted_taxon_key: str,
    scientific_name: str,
    registry_version: str,
    spread_rows: list[dict[str, object]],
    evidence_rows: list[dict[str, object]],
    policy: GeographicSummaryPolicy,
    created_at: datetime,
    grid: CellGrid,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    eligible_rows = [
        row for row in spread_rows if int(row.get("range_inference_eligible_count") or 0) > 0
    ]
    resolutions = sorted({int(row["spatial_resolution"]) for row in eligible_rows})
    component_resolution = _component_resolution(resolutions, policy.component_resolution)
    component_rows = [
        row
        for row in eligible_rows
        if component_resolution is not None
        and int(row["spatial_resolution"]) == component_resolution
    ]
    metric_resolution = (
        component_resolution
        if component_resolution is not None
        else resolutions[len(resolutions) // 2]
        if resolutions
        else None
    )
    metric_rows = [
        row
        for row in eligible_rows
        if metric_resolution is not None and int(row["spatial_resolution"]) == metric_resolution
    ]
    densities, roles_by_cell = _cell_evidence(component_rows)
    cells = set(densities)
    components = _connected_components(cells, grid=grid)
    outlier_cells = _outlier_cells(
        cells=cells,
        densities=densities,
        roles_by_cell=roles_by_cell,
        components=components,
        policy=policy,
        grid=grid,
    )
    cutoff = _year_cutoff(created_at.date(), policy.current_window_years)
    current_count, historical_count = _temporal_counts(
        evidence_rows,
        cutoff=cutoff,
    )
    georeferenced_count = _georeferenced_count(evidence_rows, metric_rows)
    eligible_count = sum(
        int(row.get("range_inference_eligible_count") or 0) for row in metric_rows
    )
    data_deficient_reasons: list[str] = []
    if georeferenced_count == 0:
        data_deficient_reasons.append("no_georeferenced_evidence")
    if eligible_count == 0:
        data_deficient_reasons.append("no_range_inference_eligible_occurrences")
    elif eligible_count < policy.min_eligible_occurrences:
        data_deficient_reasons.append("insufficient_range_inference_eligible_occurrences")
    if cells and len(cells) < policy.min_occupied_cells:
        data_deficient_reasons.append("insufficient_occupied_cells")
    if eligible_rows and component_resolution is None:
        data_deficient_reasons.append("missing_component_resolution")

    spread_fingerprint = _fingerprint_rows(spread_rows)
    evidence_fingerprint = _fingerprint_rows(evidence_rows)
    evidence_version = _sha256_json(
        {
            "summary_schema_version": GEOGRAPHIC_SUMMARY_SCHEMA_VERSION,
            "spread_fingerprint": spread_fingerprint,
            "occurrence_evidence_fingerprint": evidence_fingerprint,
            "policy": policy.manifest(),
            "resolved_component_resolution": component_resolution,
            "current_evidence_cutoff": cutoff,
            "grid_name": grid.name,
            "grid_version": grid.version,
        }
    )
    row = {
        "schema_version": GEOGRAPHIC_SUMMARY_SCHEMA_VERSION,
        "registry_version": registry_version,
        "accepted_taxon_key": accepted_taxon_key,
        "scientific_name": scientific_name,
        "geographic_evidence_version": evidence_version,
        "cell_counts_by_resolution": _cell_counts(eligible_rows),
        "countries": sorted(
            {str(row["country_code"]) for row in eligible_rows if row.get("country_code")}
        ),
        "admin_regions": _admin_regions(eligible_rows),
        "occupied_envelope": _occupied_envelope(cells, grid=grid),
        "disconnected_range_component_count": len(components),
        "occurrence_density_summary": _density_summary(tuple(densities.values())),
        "data_deficient": bool(data_deficient_reasons),
        "data_deficient_reasons": sorted(data_deficient_reasons),
        "suspicious_outlier_cell_count": len(outlier_cells),
        "range_source_coverage": _source_coverage(metric_rows),
        "known_introduced_regions": _introduced_regions(metric_rows),
        "current_evidence_count": current_count,
        "historical_evidence_count": historical_count,
        "spread_fingerprint": spread_fingerprint,
        "created_at": created_at,
    }
    findings = _taxon_qa_findings(
        accepted_taxon_key=accepted_taxon_key,
        spread_rows=spread_rows,
        evidence_rows=evidence_rows,
        outlier_count=len(outlier_cells),
        georeferenced_count=georeferenced_count,
    )
    return row, findings


def _validated_spread_rows(
    spread: pl.DataFrame,
    *,
    registry_version: str,
    grid: CellGrid,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    findings: list[dict[str, str]] = []
    rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    impossible_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    taxon_mismatch_counts: Counter[str] = Counter()
    for row in sorted(spread.to_dicts(), key=_canonical_json):
        if row.get("registry_version") != registry_version:
            raise ValueError("geographic spread registry_version mismatch")
        taxon_key = str(row.get("accepted_taxon_key") or "")
        key = tuple(row.get(field_name) for field_name in _SPREAD_PRIMARY_KEY)
        if key in seen:
            duplicate_counts[taxon_key] += 1
            continue
        seen.add(key)
        role = str(row.get("known_range_role") or "")
        if role not in _KNOWN_RANGE_ROLES:
            findings.append(_finding("fatal", "geographic_invalid_range_role", taxon_key))
            continue
        expected_key = f"gbif:{int(row['gbif_species_key'])}"
        if taxon_key != expected_key:
            taxon_mismatch_counts[taxon_key] += 1
            continue
        if not _valid_cell_row(row, grid=grid):
            impossible_counts[taxon_key] += 1
            continue
        rows.append(row)
    findings.extend(
        _count_findings("fatal", "geographic_duplicate_spread_primary_key", duplicate_counts)
    )
    findings.extend(
        _count_findings("fatal", "geographic_spread_taxon_key_mismatch", taxon_mismatch_counts)
    )
    findings.extend(
        _count_findings("fatal", "geographic_impossible_cell_identifier", impossible_counts)
    )
    return rows, findings


def _validated_evidence_rows(
    evidence: pl.DataFrame,
    *,
    registry_version: str,
) -> list[dict[str, object]]:
    rows = evidence.to_dicts()
    if any(row.get("registry_version") != registry_version for row in rows):
        raise ValueError("geographic occurrence evidence registry_version mismatch")
    return rows


def _taxon_identities(
    *,
    taxa: pl.DataFrame,
    spread: pl.DataFrame,
    occurrence_evidence: pl.DataFrame,
    registry_version: str,
) -> dict[str, str]:
    required = {"accepted_taxon_key", "scientific_name"}
    if not required <= set(taxa.columns):
        raise ValueError("taxa must contain accepted_taxon_key and scientific_name")
    identities: dict[str, str] = {}
    for row in taxa.iter_rows(named=True):
        if "rank" in taxa.columns and str(row.get("rank") or "").upper() != "SPECIES":
            continue
        _add_identity(identities, row)
    for frame in (spread, occurrence_evidence):
        for row in frame.select(["accepted_taxon_key", "scientific_name"]).unique().to_dicts():
            _add_identity(identities, row)
    if not registry_version:
        raise ValueError("registry_version must be nonblank")
    return identities


def _add_identity(identities: dict[str, str], row: Mapping[str, object]) -> None:
    key = _required_text(row.get("accepted_taxon_key"), field_name="accepted_taxon_key")
    name = _required_text(row.get("scientific_name"), field_name="scientific_name")
    previous = identities.get(key)
    if previous is not None and previous != name:
        raise ValueError(f"conflicting scientific names for {key}: {previous!r} and {name!r}")
    identities[key] = name


def _rows_by_taxon(rows: Sequence[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("accepted_taxon_key") or "")].append(row)
    return dict(grouped)


def _valid_cell_row(row: Mapping[str, object], *, grid: CellGrid) -> bool:
    cell_id = row.get("spatial_cell_id")
    if not grid.is_valid(cell_id):
        return False
    try:
        resolution = validate_resolution(row.get("spatial_resolution"))
        center = grid.center(str(cell_id))
        if grid.coordinate_to_cell(center, resolution=resolution) != cell_id:
            return False
        latitude = float(row["centroid_latitude"])
        longitude = float(row["centroid_longitude"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and abs(latitude - float(center.latitude)) <= 1e-7
        and abs(_longitude_delta(longitude, float(center.longitude))) <= 1e-7
    )


def _component_resolution(resolutions: list[int], configured: int | None) -> int | None:
    if configured is not None:
        return configured if configured in resolutions else None
    if not resolutions:
        return None
    return resolutions[len(resolutions) // 2]


def _cell_evidence(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, int], dict[str, set[str]]]:
    densities: Counter[str] = Counter()
    roles: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        cell_id = str(row["spatial_cell_id"])
        densities[cell_id] += int(row.get("range_inference_eligible_count") or 0)
        roles[cell_id].add(str(row.get("known_range_role") or "unknown"))
    return dict(densities), dict(roles)


def _connected_components(cells: set[str], *, grid: CellGrid) -> list[tuple[str, ...]]:
    unseen = set(cells)
    components: list[tuple[str, ...]] = []
    while unseen:
        origin = min(unseen)
        unseen.remove(origin)
        queue = deque([origin])
        component = {origin}
        while queue:
            current = queue.popleft()
            adjacent = set(grid.neighbours(current)) & unseen
            for neighbour in sorted(adjacent):
                unseen.remove(neighbour)
                component.add(neighbour)
                queue.append(neighbour)
        components.append(tuple(sorted(component)))
    return sorted(components, key=lambda values: (-len(values), values))


def _outlier_cells(
    *,
    cells: set[str],
    densities: Mapping[str, int],
    roles_by_cell: Mapping[str, set[str]],
    components: Sequence[tuple[str, ...]],
    policy: GeographicSummaryPolicy,
    grid: CellGrid,
) -> set[str]:
    outliers: set[str] = set()
    for component in components:
        if len(component) != 1:
            continue
        cell_id = component[0]
        if densities.get(cell_id, 0) > policy.outlier_max_eligible_occurrences:
            continue
        if roles_by_cell.get(cell_id, set()) & {"introduced", "vagrant"}:
            continue
        nearby = set(
            grid.neighbours(
                cell_id,
                grid_distance=policy.outlier_neighbour_distance,
            )
        )
        if not nearby.intersection(cells):
            outliers.add(cell_id)
    return outliers


def _cell_counts(rows: Sequence[Mapping[str, object]]) -> list[dict[str, int]]:
    cells_by_resolution: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        cells_by_resolution[int(row["spatial_resolution"])].add(str(row["spatial_cell_id"]))
    return [
        {"resolution": resolution, "count": len(cells_by_resolution[resolution])}
        for resolution in sorted(cells_by_resolution)
    ]


def _occupied_envelope(cells: set[str], *, grid: CellGrid) -> dict[str, object] | None:
    if not cells:
        return None
    centers = [grid.center(cell_id) for cell_id in sorted(cells)]
    latitudes = [float(center.latitude) for center in centers]
    west, east, crosses_dateline = _longitude_envelope(
        [float(center.longitude) for center in centers]
    )
    return {
        "south": min(latitudes),
        "north": max(latitudes),
        "west": west,
        "east": east,
        "crosses_dateline": crosses_dateline,
    }


def _longitude_envelope(longitudes: Sequence[float]) -> tuple[float, float, bool]:
    circular = sorted(longitude % 360.0 for longitude in longitudes)
    if len(circular) == 1:
        longitude = _wrapped_longitude(circular[0])
        return longitude, longitude, False
    gaps = [circular[index + 1] - circular[index] for index in range(len(circular) - 1)]
    gaps.append(circular[0] + 360.0 - circular[-1])
    gap_index = max(range(len(gaps)), key=gaps.__getitem__)
    west = _wrapped_longitude(circular[(gap_index + 1) % len(circular)])
    east = _wrapped_longitude(circular[gap_index])
    return west, east, west > east


def _density_summary(values: Sequence[int]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(float(value) for value in values)
    return {
        "min": ordered[0],
        "p50": _nearest_rank(ordered, 0.50),
        "p95": _nearest_rank(ordered, 0.95),
        "max": ordered[-1],
    }


def _source_coverage(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    datasets: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for row in rows:
        source = str(row.get("source") or "")
        dataset = str(row.get("source_dataset_key") or "")
        if dataset:
            datasets[source].add(dataset)
        counts[source] += int(row.get("range_inference_eligible_count") or 0)
    return [
        {
            "source": source,
            "dataset_count": len(datasets[source]),
            "eligible_occurrence_count": counts[source],
        }
        for source in sorted(counts)
    ]


def _admin_regions(rows: Sequence[Mapping[str, object]]) -> list[str]:
    regions: set[str] = set()
    for row in rows:
        admin1 = str(row.get("admin1") or "")
        if not admin1:
            continue
        country = str(row.get("country_code") or "")
        regions.add(f"{country}:{admin1}" if country else admin1)
    return sorted(regions)


def _introduced_regions(rows: Sequence[Mapping[str, object]]) -> list[str]:
    regions: set[str] = set()
    for row in rows:
        if row.get("known_range_role") != "introduced":
            continue
        country = str(row.get("country_code") or "")
        admin1 = str(row.get("admin1") or "")
        if admin1:
            regions.add(f"admin1:{country}:{admin1}" if country else f"admin1:{admin1}")
        elif country:
            regions.add(f"country:{country}")
        else:
            regions.add(
                f"cell:{int(row['spatial_resolution'])}:{row['spatial_cell_id']}"
            )
    return sorted(regions)


def _temporal_counts(
    evidence_rows: Sequence[dict[str, object]],
    *,
    cutoff: date,
) -> tuple[int, int]:
    current = 0
    historical = 0
    for row in _one_row_per_occurrence(evidence_rows):
        event_date = row.get("event_date")
        if not isinstance(event_date, date):
            continue
        valid = bool(row.get("coordinate_valid")) and bool(row.get("taxon_key_match"))
        if not valid or bool(row.get("has_geospatial_issue")) or bool(row.get("fossil")):
            continue
        if bool(row.get("range_inference_eligible")) and event_date >= cutoff:
            current += 1
        elif event_date < cutoff:
            historical += 1
    return current, historical


def _georeferenced_count(
    evidence_rows: Sequence[dict[str, object]],
    component_rows: Sequence[Mapping[str, object]],
) -> int:
    if evidence_rows:
        return sum(
            bool(row.get("coordinate_valid")) and bool(row.get("taxon_key_match"))
            for row in _one_row_per_occurrence(evidence_rows)
        )
    return sum(int(row.get("georeferenced_occurrence_count") or 0) for row in component_rows)


def _one_row_per_occurrence(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    by_id: dict[str, dict[str, object]] = {}
    for row in sorted(rows, key=_canonical_json):
        gbif_id = str(row.get("gbif_id") or "")
        previous = by_id.get(gbif_id)
        if previous is None or (
            not previous.get("coordinate_valid") and row.get("coordinate_valid")
        ):
            by_id[gbif_id] = row
    return [by_id[key] for key in sorted(by_id)]


def _taxon_qa_findings(
    *,
    accepted_taxon_key: str,
    spread_rows: Sequence[dict[str, object]],
    evidence_rows: Sequence[dict[str, object]],
    outlier_count: int,
    georeferenced_count: int,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    occurrences = _one_row_per_occurrence(evidence_rows)
    invalid_count = sum(row.get("exclusion_reason") == "invalid_coordinate" for row in occurrences)
    mismatch_count = sum(row.get("exclusion_reason") == "taxon_key_mismatch" for row in occurrences)
    if invalid_count:
        findings.append(
            _finding(
                "warning",
                "geographic_invalid_coordinate",
                f"{accepted_taxon_key}:{invalid_count}",
            )
        )
    if mismatch_count:
        findings.append(
            _finding(
                "warning",
                "geographic_taxon_key_mismatch",
                f"{accepted_taxon_key}:{mismatch_count}",
            )
        )
    valid_occurrences = [
        row
        for row in occurrences
        if row.get("coordinate_valid") and row.get("taxon_key_match")
    ]
    if valid_occurrences and all(bool(row.get("preserved_specimen")) for row in valid_occurrences):
        findings.append(
            _finding(
                "warning",
                "geographic_occurrences_only_from_preserved_specimens",
                accepted_taxon_key,
            )
        )
    if georeferenced_count == 0:
        findings.append(
            _finding("warning", "geographic_no_georeferenced_evidence", accepted_taxon_key)
        )
    conflict_count = _range_role_conflict_count(spread_rows)
    if conflict_count:
        findings.append(
            _finding(
                "warning",
                "geographic_conflicting_native_introduced_evidence",
                f"{accepted_taxon_key}:{conflict_count}",
            )
        )
    if outlier_count:
        findings.append(
            _finding(
                "warning",
                "geographic_extreme_isolated_outlier",
                f"{accepted_taxon_key}:{outlier_count}",
            )
        )
    return findings


def _range_role_conflict_count(rows: Sequence[Mapping[str, object]]) -> int:
    roles: dict[tuple[int, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (int(row["spatial_resolution"]), str(row["spatial_cell_id"]))
        roles[key].add(str(row.get("known_range_role") or "unknown"))
    return sum({"native", "introduced"} <= values for values in roles.values())


def _qa_frame(findings: Sequence[Mapping[str, object]]) -> pl.DataFrame:
    unique = {
        (
            str(finding.get("severity") or ""),
            str(finding.get("code") or ""),
            str(finding.get("subject") or ""),
        )
        for finding in findings
    }
    rows = [
        {"severity": severity, "code": code, "subject": subject}
        for severity, code, subject in sorted(unique)
    ]
    return _typed_frame(rows, geographic_qa_schema()).sort(["severity", "code", "subject"])


def _count_findings(
    severity: str,
    code: str,
    counts: Mapping[str, int],
) -> list[dict[str, str]]:
    return [
        _finding(severity, code, f"{subject}:{counts[subject]}")
        for subject in sorted(counts)
        if counts[subject]
    ]


def _finding(severity: str, code: str, subject: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "subject": subject}


def _manifest(
    *,
    summary: pl.DataFrame,
    qa: pl.DataFrame,
    summary_path: Path,
    qa_path: Path,
    registry_version: str,
    policy: GeographicSummaryPolicy,
    created_at: datetime,
    grid: CellGrid,
) -> dict[str, object]:
    fatal_count = qa.filter(pl.col("severity") == "fatal").height
    warning_count = qa.filter(pl.col("severity") == "warning").height
    return {
        "schema_version": GEOGRAPHIC_SUMMARY_MANIFEST_SCHEMA_VERSION,
        "registry_version": registry_version,
        "status": "complete",
        "qa_status": "failed" if fatal_count else "passed",
        "qa_fatal_count": fatal_count,
        "qa_warning_count": warning_count,
        "summary_row_count": summary.height,
        "qa_row_count": qa.height,
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "grid_name": grid.name,
        "grid_version": grid.version,
        "policy": policy.manifest(),
        "files": {
            "taxon_geographic_summary": _artifact(summary_path, summary.height),
            "geographic_qa_findings": _artifact(qa_path, qa.height),
        },
    }


def _artifact(path: Path, row_count: int) -> dict[str, object]:
    return {
        "file": path.name,
        "row_count": row_count,
        "byte_count": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _fingerprint_rows(rows: Sequence[Mapping[str, object]]) -> str:
    return _sha256_json(sorted((_canonical(row) for row in rows), key=_canonical_json))


def _canonical(value: object) -> object:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("geographic artifact contains a non-finite float")
        return 0.0 if value == 0.0 else value
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _year_cutoff(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    return values[max(0, math.ceil(quantile * len(values)) - 1)]


def _wrapped_longitude(value: float) -> float:
    wrapped = (value + 180.0) % 360.0 - 180.0
    return 0.0 if wrapped == 0.0 else wrapped


def _longitude_delta(left: float, right: float) -> float:
    return _wrapped_longitude(left - right)


def _require_schema(
    frame: pl.DataFrame,
    expected: dict[str, pl.DataType],
    *,
    name: str,
) -> None:
    if frame.schema != expected:
        raise ValueError(f"{name} schema mismatch")


def _typed_frame(
    rows: Sequence[Mapping[str, object]],
    schema: dict[str, pl.DataType],
) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    normalized = [{field_name: row.get(field_name) for field_name in schema} for row in rows]
    return pl.DataFrame(normalized, schema=schema, orient="row", strict=False)


def _required_text(value: object, *, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{field_name} must be nonblank")
    return text


def _utc_datetime(value: str | datetime) -> datetime:
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    if parsed.tzinfo is None:
        raise ValueError("created_at must include a timezone")
    return parsed.astimezone(UTC)


def _write_json_atomic(payload: Mapping[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "GEOGRAPHIC_QA_FINDINGS_FILE",
    "GEOGRAPHIC_SUMMARY_POLICY_VERSION",
    "GEOGRAPHIC_SUMMARY_SCHEMA_VERSION",
    "GeographicSummaryBuildResult",
    "GeographicSummaryPolicy",
    "TAXON_GEOGRAPHIC_SUMMARY_FILE",
    "build_geographic_summary",
    "geographic_qa_schema",
    "geographic_summary_schema",
]
