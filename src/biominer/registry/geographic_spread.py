from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from biominer.geography import (
    CellGrid,
    GeographicCoordinate,
    GeographicResolutions,
    default_cell_grid,
    project_coordinate,
)
from biominer.registry.gbif_production import RetryingHTTPGet
from biominer.storage.parquet import iter_parquet_batches, write_parquet


logger = logging.getLogger(__name__)

GEOGRAPHIC_SPREAD_SCHEMA_VERSION = "taxon-geographic-spread-v1.0.0"
GEOGRAPHIC_EVIDENCE_SCHEMA_VERSION = "geographic-occurrence-evidence-v1.0.0"
GEOGRAPHIC_CHECKPOINT_SCHEMA_VERSION = "geographic-spread-checkpoint-v1.0.0"
GEOGRAPHIC_BUILD_MANIFEST_SCHEMA_VERSION = "geographic-spread-build-v1.0.0"
TAXON_GEOGRAPHIC_SPREAD_FILE = "taxon_geographic_spread.parquet"
GEOGRAPHIC_OCCURRENCE_EVIDENCE_FILE = "geographic_occurrence_evidence.parquet"
GEOGRAPHIC_SPREAD_MANIFEST_FILE = "geographic_spread_manifest.json"
GEOGRAPHIC_CHECKPOINT_STATE_FILE = "state.json"
GBIF_SEARCH_PAGE_LIMIT = 300
GBIF_SEARCH_MAX_RECORDS = 100_000

HTTPGet = Callable[[str, dict[str, object]], dict[str, Any]]

_OBSERVATION_BASIS = frozenset({"HUMAN_OBSERVATION", "MACHINE_OBSERVATION", "OBSERVATION"})
_GEOSPATIAL_ISSUES = frozenset(
    {
        "CONTINENT_COUNTRY_MISMATCH",
        "CONTINENT_DERIVED_FROM_COORDINATES",
        "COORDINATE_INVALID",
        "COORDINATE_OUT_OF_RANGE",
        "COORDINATE_REPROJECTED",
        "COORDINATE_REPROJECTION_FAILED",
        "COORDINATE_ROUNDED",
        "COUNTRY_COORDINATE_MISMATCH",
        "COUNTRY_DERIVED_FROM_COORDINATES",
        "GEODETIC_DATUM_ASSUMED_WGS84",
        "GEODETIC_DATUM_INVALID",
        "PRESUMED_NEGATED_LATITUDE",
        "PRESUMED_NEGATED_LONGITUDE",
        "PRESUMED_SWAPPED_COORDINATE",
        "ZERO_COORDINATE",
    }
)


class BulkDownloadRequired(RuntimeError):
    def __init__(self, *, total_records: int, request_payload: dict[str, object]) -> None:
        self.total_records = total_records
        self.request_payload = request_payload
        super().__init__(
            f"GBIF occurrence search returned {total_records} records; the documented "
            f"search ceiling is {GBIF_SEARCH_MAX_RECORDS}, so an authenticated "
            "SIMPLE_PARQUET occurrence download is required"
        )


@dataclass(frozen=True, slots=True)
class OccurrenceBatch:
    cursor: int
    next_cursor: int
    records: tuple[dict[str, object], ...]
    end_of_records: bool
    total_records: int | None

    def __post_init__(self) -> None:
        if self.cursor < 0 or self.next_cursor < self.cursor:
            raise ValueError("occurrence batch cursors must be non-negative and monotonic")
        if self.next_cursor - self.cursor != len(self.records):
            raise ValueError("occurrence batch cursor span must equal record count")
        if self.total_records is not None and self.total_records < self.next_cursor:
            raise ValueError("occurrence batch total_records cannot be below next_cursor")


class OccurrenceBatchSource(Protocol):
    source: str
    source_query_hash: str
    source_snapshot_version: str

    def iter_batches(self, *, start_cursor: int = 0) -> Iterator[OccurrenceBatch]: ...


@dataclass(frozen=True, slots=True)
class GeographicSpreadBuildResult:
    spread: pl.DataFrame
    evidence_path: Path
    manifest: dict[str, object]
    resumed: bool


class GBIFOccurrenceSearchSource:
    source = "GBIF"

    def __init__(
        self,
        *,
        accepted_taxon_key: str,
        source_snapshot_version: str,
        http_get: HTTPGet | None = None,
        page_size: int = GBIF_SEARCH_PAGE_LIMIT,
        max_retries: int = 5,
    ) -> None:
        self.accepted_taxon_key = _accepted_taxon_key(accepted_taxon_key)
        self.source_snapshot_version = _required_text(
            source_snapshot_version,
            field_name="source_snapshot_version",
        )
        if not 1 <= page_size <= GBIF_SEARCH_PAGE_LIMIT:
            raise ValueError(f"page_size must be between 1 and {GBIF_SEARCH_PAGE_LIMIT}")
        self.page_size = page_size
        self.source_query_hash = _source_query_hash(self.accepted_taxon_key)
        self._transport = None if http_get is not None else RetryingHTTPGet(max_retries=max_retries)
        self._http_get = http_get or self._transport

    def iter_batches(self, *, start_cursor: int = 0) -> Iterator[OccurrenceBatch]:
        if start_cursor < 0 or start_cursor > GBIF_SEARCH_MAX_RECORDS:
            raise ValueError(
                f"start_cursor must be between 0 and {GBIF_SEARCH_MAX_RECORDS}"
            )
        cursor = start_cursor
        while True:
            request_limit = min(self.page_size, GBIF_SEARCH_MAX_RECORDS - cursor)
            if request_limit < 1:
                raise ValueError("start_cursor is at the GBIF occurrence search ceiling")
            payload = self._http_get(
                "/occurrence/search",
                {
                    "taxonKey": _bare_gbif_key(self.accepted_taxon_key),
                    "hasCoordinate": "true",
                    "limit": request_limit,
                    "offset": cursor,
                },
            )
            if not isinstance(payload, dict):
                raise ValueError("GBIF occurrence search response must be a JSON object")
            total = _nonnegative_int(payload.get("count"), field_name="count")
            if total > GBIF_SEARCH_MAX_RECORDS:
                raise BulkDownloadRequired(
                    total_records=total,
                    request_payload=build_gbif_bulk_download_request(
                        accepted_taxon_key=self.accepted_taxon_key
                    ),
                )
            response_offset = _nonnegative_int(
                payload.get("offset", cursor),
                field_name="offset",
            )
            if response_offset != cursor:
                raise ValueError(
                    f"GBIF occurrence page offset {response_offset} did not match request {cursor}"
                )
            raw_results = payload.get("results") or []
            if not isinstance(raw_results, list) or not all(
                isinstance(row, dict) for row in raw_results
            ):
                raise ValueError("GBIF occurrence search results must be an array of objects")
            records = tuple(dict(row) for row in raw_results)
            next_cursor = cursor + len(records)
            end_of_records = bool(payload.get("endOfRecords")) or next_cursor >= total
            if not records and not end_of_records:
                raise ValueError("GBIF occurrence search returned an empty non-terminal page")
            yield OccurrenceBatch(
                cursor=cursor,
                next_cursor=next_cursor,
                records=records,
                end_of_records=end_of_records,
                total_records=total,
            )
            if end_of_records:
                return
            cursor = next_cursor

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()

    def __enter__(self) -> GBIFOccurrenceSearchSource:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class GBIFParquetOccurrenceSource:
    source = "GBIF"

    def __init__(
        self,
        path: str | Path,
        *,
        accepted_taxon_key: str,
        source_snapshot_version: str,
        batch_size: int = 10_000,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.accepted_taxon_key = _accepted_taxon_key(accepted_taxon_key)
        self.source_snapshot_version = _required_text(
            source_snapshot_version,
            field_name="source_snapshot_version",
        )
        self.source_query_hash = _source_query_hash(self.accepted_taxon_key)

    def iter_batches(self, *, start_cursor: int = 0) -> Iterator[OccurrenceBatch]:
        if start_cursor < 0:
            raise ValueError("start_cursor must be non-negative")
        total = _parquet_row_count(self.path)
        offset = 0
        for frame in iter_parquet_batches(self.path, batch_size=self.batch_size):
            batch_end = offset + frame.height
            if batch_end <= start_cursor:
                offset = batch_end
                continue
            start_in_batch = max(0, start_cursor - offset)
            selected = frame.slice(start_in_batch)
            cursor = offset + start_in_batch
            records = tuple(dict(row) for row in selected.iter_rows(named=True))
            next_cursor = cursor + len(records)
            yield OccurrenceBatch(
                cursor=cursor,
                next_cursor=next_cursor,
                records=records,
                end_of_records=next_cursor >= total,
                total_records=total,
            )
            offset = batch_end
        if total == 0 and start_cursor == 0:
            yield OccurrenceBatch(
                cursor=0,
                next_cursor=0,
                records=(),
                end_of_records=True,
                total_records=0,
            )


def build_gbif_bulk_download_request(
    *,
    accepted_taxon_key: str,
    notification_addresses: Sequence[str] = (),
) -> dict[str, object]:
    key = _accepted_taxon_key(accepted_taxon_key)
    payload: dict[str, object] = {
        "format": "SIMPLE_PARQUET",
        "predicate": {
            "type": "and",
            "predicates": [
                {"type": "equals", "key": "TAXON_KEY", "value": _bare_gbif_key(key)},
                {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
            ],
        },
    }
    addresses = sorted(
        {
            str(value).strip()
            for value in notification_addresses
            if str(value).strip()
        }
    )
    if addresses:
        payload["notificationAddresses"] = addresses
        payload["sendNotification"] = True
    return payload


def geographic_spread_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "registry_version": pl.String,
        "accepted_taxon_key": pl.String,
        "gbif_species_key": pl.UInt64,
        "scientific_name": pl.String,
        "source": pl.String,
        "source_dataset_key": pl.String,
        "source_dataset_citation": pl.String,
        "source_query_hash": pl.String,
        "spatial_cell_id": pl.String,
        "spatial_resolution": pl.UInt8,
        "country_code": pl.String,
        "admin1": pl.String,
        "bioregion": pl.String,
        "centroid_latitude": pl.Float64,
        "centroid_longitude": pl.Float64,
        "occurrence_count": pl.UInt64,
        "georeferenced_occurrence_count": pl.UInt64,
        "range_inference_eligible_count": pl.UInt64,
        "preserved_specimen_count": pl.UInt64,
        "fossil_count": pl.UInt64,
        "geospatial_issue_count": pl.UInt64,
        "coordinate_uncertainty_summary": pl.Struct(
            {
                "count": pl.UInt64,
                "min_m": pl.Float64,
                "p50_m": pl.Float64,
                "p95_m": pl.Float64,
                "max_m": pl.Float64,
            }
        ),
        "earliest_occurrence_date": pl.Date,
        "latest_occurrence_date": pl.Date,
        "basis_of_record_counts": pl.List(
            pl.Struct({"value": pl.String, "count": pl.UInt64})
        ),
        "establishment_means": pl.List(pl.String),
        "occurrence_status": pl.String,
        "known_range_role": pl.String,
        "evidence_confidence": pl.Float32,
        "retrieved_at": pl.Datetime("us", "UTC"),
        "source_snapshot_version": pl.String,
    }


def geographic_occurrence_evidence_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "gbif_id": pl.String,
        "registry_version": pl.String,
        "accepted_taxon_key": pl.String,
        "gbif_species_key": pl.UInt64,
        "scientific_name": pl.String,
        "source": pl.String,
        "source_dataset_key": pl.String,
        "source_dataset_citation": pl.String,
        "source_query_hash": pl.String,
        "spatial_cell_id": pl.String,
        "spatial_resolution": pl.UInt8,
        "country_code": pl.String,
        "admin1": pl.String,
        "bioregion": pl.String,
        "centroid_latitude": pl.Float64,
        "centroid_longitude": pl.Float64,
        "coordinate_uncertainty_m": pl.Float64,
        "event_date": pl.Date,
        "basis_of_record": pl.String,
        "establishment_means": pl.String,
        "occurrence_status": pl.String,
        "known_range_role": pl.String,
        "has_geospatial_issue": pl.Boolean,
        "preserved_specimen": pl.Boolean,
        "fossil": pl.Boolean,
        "range_inference_eligible": pl.Boolean,
        "taxon_key_match": pl.Boolean,
        "coordinate_valid": pl.Boolean,
        "exclusion_reason": pl.String,
        "retrieved_at": pl.Datetime("us", "UTC"),
        "source_snapshot_version": pl.String,
    }


def build_taxon_geographic_spread(
    *,
    accepted_taxon_key: str,
    scientific_name: str,
    registry_version: str,
    source: OccurrenceBatchSource,
    resolutions: GeographicResolutions,
    output_dir: str | Path,
    checkpoint_dir: str | Path,
    retrieved_at: str | datetime,
    grid: CellGrid | None = None,
) -> GeographicSpreadBuildResult:
    accepted_key = _accepted_taxon_key(accepted_taxon_key)
    name = _required_text(scientific_name, field_name="scientific_name")
    registry = _required_text(registry_version, field_name="registry_version")
    if not isinstance(resolutions, GeographicResolutions):
        raise TypeError("resolutions must be GeographicResolutions")
    retrieved = _utc_datetime(retrieved_at)
    backend = grid or default_cell_grid()
    output = Path(output_dir)
    checkpoint = Path(checkpoint_dir)
    parts_dir = checkpoint / "parts"
    output.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    identity = {
        "schema_version": GEOGRAPHIC_CHECKPOINT_SCHEMA_VERSION,
        "registry_version": registry,
        "accepted_taxon_key": accepted_key,
        "scientific_name": name,
        "source": _required_text(source.source, field_name="source"),
        "source_query_hash": _required_hash(
            source.source_query_hash,
            field_name="source_query_hash",
        ),
        "source_snapshot_version": _required_text(
            source.source_snapshot_version,
            field_name="source_snapshot_version",
        ),
        "resolutions": list(resolutions.values),
        "grid_name": backend.name,
        "grid_version": backend.version,
        "retrieved_at": retrieved.isoformat().replace("+00:00", "Z"),
    }
    state_path = checkpoint / GEOGRAPHIC_CHECKPOINT_STATE_FILE
    state, resumed = _load_or_initialize_state(state_path, identity=identity, parts_dir=parts_dir)
    next_cursor = int(state["next_cursor"])
    ended = str(state.get("status") or "") == "complete"

    if not ended:
        for batch in source.iter_batches(start_cursor=next_cursor):
            if batch.cursor != next_cursor:
                raise ValueError(
                    f"occurrence source yielded cursor {batch.cursor}, expected {next_cursor}"
                )
            previous_total = state.get("total_records")
            if (
                previous_total is not None
                and batch.total_records is not None
                and int(previous_total) != batch.total_records
            ):
                raise ValueError(
                    "occurrence source total_records changed within one source snapshot: "
                    f"found {batch.total_records}, expected {previous_total}"
                )
            evidence = _evidence_frame(
                batch.records,
                accepted_taxon_key=accepted_key,
                scientific_name=name,
                registry_version=registry,
                source_name=identity["source"],
                source_query_hash=identity["source_query_hash"],
                source_snapshot_version=identity["source_snapshot_version"],
                resolutions=resolutions,
                retrieved_at=retrieved,
                grid=backend,
            )
            part_path = parts_dir / f"part-{batch.cursor:012d}.parquet"
            _write_or_validate_checkpoint_part(evidence, part_path)
            part = {
                "file": part_path.name,
                "cursor": batch.cursor,
                "next_cursor": batch.next_cursor,
                "row_count": evidence.height,
                "byte_count": part_path.stat().st_size,
                "sha256": _sha256_file(part_path),
            }
            state["parts"] = [*list(state.get("parts") or []), part]
            state["next_cursor"] = batch.next_cursor
            state["total_records"] = batch.total_records
            state["status"] = "complete" if batch.end_of_records else "running"
            _write_json_atomic(state, state_path)
            next_cursor = batch.next_cursor
            ended = batch.end_of_records
            logger.info(
                "registry.geographic_spread.checkpoint accepted_taxon_key=%s cursor=%d "
                "next_cursor=%d evidence_rows=%d bytes=%d status=%s",
                accepted_key,
                batch.cursor,
                batch.next_cursor,
                evidence.height,
                part_path.stat().st_size,
                state["status"],
            )
        if not ended:
            raise ValueError("occurrence source ended without a terminal batch")

    evidence = _compact_evidence(parts_dir, parts=list(state.get("parts") or []))
    evidence_path = write_parquet(evidence, output / GEOGRAPHIC_OCCURRENCE_EVIDENCE_FILE)
    spread = _aggregate_spread(evidence)
    spread_path = write_parquet(spread, output / TAXON_GEOGRAPHIC_SPREAD_FILE)
    manifest = _build_manifest(
        identity=identity,
        state=state,
        evidence=evidence,
        spread=spread,
        evidence_path=evidence_path,
        spread_path=spread_path,
        resumed=resumed,
        retrieved_at=retrieved,
    )
    _write_json_atomic(manifest, output / GEOGRAPHIC_SPREAD_MANIFEST_FILE)
    logger.info(
        "registry.geographic_spread.complete accepted_taxon_key=%s occurrences=%d "
        "spread_rows=%d eligible=%d resumed=%s",
        accepted_key,
        manifest["completed_occurrence_count"],
        spread.height,
        manifest["range_inference_eligible_occurrence_count"],
        resumed,
    )
    return GeographicSpreadBuildResult(
        spread=spread,
        evidence_path=evidence_path,
        manifest=manifest,
        resumed=resumed,
    )


def _evidence_frame(
    records: Sequence[Mapping[str, object]],
    *,
    accepted_taxon_key: str,
    scientific_name: str,
    registry_version: str,
    source_name: object,
    source_query_hash: object,
    source_snapshot_version: object,
    resolutions: GeographicResolutions,
    retrieved_at: datetime,
    grid: CellGrid,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        base = _evidence_base_row(
            record,
            accepted_taxon_key=accepted_taxon_key,
            scientific_name=scientific_name,
            registry_version=registry_version,
            source_name=str(source_name),
            source_query_hash=str(source_query_hash),
            source_snapshot_version=str(source_snapshot_version),
            retrieved_at=retrieved_at,
        )
        if not base["taxon_key_match"]:
            rows.append({**base, "exclusion_reason": "taxon_key_mismatch"})
            continue
        if not base["source_dataset_key"]:
            rows.append({**base, "exclusion_reason": "missing_dataset_key"})
            continue
        try:
            coordinate = GeographicCoordinate(
                latitude=_value(record, "decimalLatitude", "decimal_latitude"),
                longitude=_value(record, "decimalLongitude", "decimal_longitude"),
                coordinate_uncertainty_m=_optional_float(
                    _value(
                        record,
                        "coordinateUncertaintyInMeters",
                        "coordinate_uncertainty_in_meters",
                    )
                ),
            )
            projection = project_coordinate(coordinate, resolutions=resolutions, grid=grid)
        except (TypeError, ValueError):
            rows.append({**base, "exclusion_reason": "invalid_coordinate"})
            continue
        for cell in projection.cells:
            center = grid.center(cell.cell_id)
            rows.append(
                {
                    **base,
                    "spatial_cell_id": cell.cell_id,
                    "spatial_resolution": cell.resolution,
                    "centroid_latitude": center.latitude,
                    "centroid_longitude": center.longitude,
                    "coordinate_uncertainty_m": coordinate.coordinate_uncertainty_m,
                    "coordinate_valid": True,
                    "exclusion_reason": None,
                }
            )
    return _typed_frame(rows, geographic_occurrence_evidence_schema()).sort(
        ["gbif_id", "spatial_resolution"]
    )


def _evidence_base_row(
    record: Mapping[str, object],
    *,
    accepted_taxon_key: str,
    scientific_name: str,
    registry_version: str,
    source_name: str,
    source_query_hash: str,
    source_snapshot_version: str,
    retrieved_at: datetime,
) -> dict[str, object]:
    gbif_species_key = int(_bare_gbif_key(accepted_taxon_key))
    record_species_value = _value(record, "speciesKey", "species_key")
    record_species_key = _optional_int(record_species_value)
    record_accepted_key = _optional_int(
        _value(
            record,
            "acceptedTaxonKey",
            "accepted_taxon_key",
            "taxonKey",
            "taxon_key",
        )
    )
    basis = _normalized_token(_value(record, "basisOfRecord", "basis_of_record"))
    establishment = _normalized_token(
        _value(record, "establishmentMeans", "establishment_means")
    )
    occurrence_status = _normalized_token(
        _value(record, "occurrenceStatus", "occurrence_status")
    ) or "UNKNOWN"
    issues = _issue_values(_value(record, "issues", "issue"))
    has_geospatial_issue = bool(
        _bool_value(_value(record, "hasGeospatialIssues", "has_geospatial_issues"))
    ) or bool(issues & _GEOSPATIAL_ISSUES)
    preserved = basis == "PRESERVED_SPECIMEN"
    fossil = basis == "FOSSIL_SPECIMEN"
    eligible = (
        basis in _OBSERVATION_BASIS
        and occurrence_status != "ABSENT"
        and not has_geospatial_issue
        and not preserved
        and not fossil
    )
    return {
        "schema_version": GEOGRAPHIC_EVIDENCE_SCHEMA_VERSION,
        "gbif_id": _gbif_id(record),
        "registry_version": registry_version,
        "accepted_taxon_key": accepted_taxon_key,
        "gbif_species_key": gbif_species_key,
        "scientific_name": scientific_name,
        "source": source_name,
        "source_dataset_key": _optional_text(
            _value(record, "datasetKey", "dataset_key")
        ),
        "source_dataset_citation": _optional_text(
            _value(record, "datasetCitation", "dataset_citation", "datasetTitle", "dataset_title")
        ),
        "source_query_hash": source_query_hash,
        "spatial_cell_id": None,
        "spatial_resolution": None,
        "country_code": _country_code(_value(record, "countryCode", "country_code")),
        "admin1": _optional_text(_value(record, "stateProvince", "state_province", "admin1")),
        "bioregion": _optional_text(record.get("bioregion")),
        "centroid_latitude": None,
        "centroid_longitude": None,
        "coordinate_uncertainty_m": _optional_float(
            _value(
                record,
                "coordinateUncertaintyInMeters",
                "coordinate_uncertainty_in_meters",
            )
        ),
        "event_date": _event_date(record),
        "basis_of_record": basis,
        "establishment_means": establishment,
        "occurrence_status": occurrence_status,
        "known_range_role": _known_range_role(establishment),
        "has_geospatial_issue": has_geospatial_issue,
        "preserved_specimen": preserved,
        "fossil": fossil,
        "range_inference_eligible": eligible,
        "taxon_key_match": (
            record_species_key == gbif_species_key
            if record_species_value not in (None, "")
            else record_accepted_key == gbif_species_key
        ),
        "coordinate_valid": False,
        "exclusion_reason": None,
        "retrieved_at": retrieved_at,
        "source_snapshot_version": source_snapshot_version,
    }


@dataclass(slots=True)
class _SpreadAccumulator:
    template: dict[str, object]
    records: dict[str, dict[str, object]] = field(default_factory=dict)
    uncertainties: list[float] = field(default_factory=list)
    dates: list[date] = field(default_factory=list)
    basis_counts: Counter[str] = field(default_factory=Counter)
    establishment_means: set[str] = field(default_factory=set)
    statuses: set[str] = field(default_factory=set)
    citations: set[str] = field(default_factory=set)
    country_codes: set[str] = field(default_factory=set)
    admin1_values: set[str] = field(default_factory=set)
    bioregions: set[str] = field(default_factory=set)

    def add(self, row: dict[str, object]) -> None:
        gbif_id = str(row["gbif_id"])
        previous = self.records.get(gbif_id)
        if previous is not None:
            if previous != row:
                raise ValueError(f"conflicting geographic evidence for GBIF occurrence {gbif_id}")
            return
        self.records[gbif_id] = row
        uncertainty = row.get("coordinate_uncertainty_m")
        if uncertainty is not None:
            self.uncertainties.append(float(uncertainty))
        event_date = row.get("event_date")
        if isinstance(event_date, date):
            self.dates.append(event_date)
        basis = str(row.get("basis_of_record") or "")
        if basis:
            self.basis_counts[basis] += 1
        establishment = str(row.get("establishment_means") or "")
        if establishment:
            self.establishment_means.add(establishment)
        self.statuses.add(str(row.get("occurrence_status") or "UNKNOWN"))
        citation = str(row.get("source_dataset_citation") or "")
        if citation:
            self.citations.add(citation)
        country_code = str(row.get("country_code") or "")
        if country_code:
            self.country_codes.add(country_code)
        admin1 = str(row.get("admin1") or "")
        if admin1:
            self.admin1_values.add(admin1)
        bioregion = str(row.get("bioregion") or "")
        if bioregion:
            self.bioregions.add(bioregion)

    def output_row(self) -> dict[str, object]:
        rows = tuple(self.records.values())
        count = len(rows)
        eligible = sum(bool(row["range_inference_eligible"]) for row in rows)
        statuses = sorted(self.statuses)
        return {
            **self.template,
            "source_dataset_citation": min(self.citations) if self.citations else None,
            "country_code": _single_or_none(self.country_codes),
            "admin1": _single_or_none(self.admin1_values),
            "bioregion": _single_or_none(self.bioregions),
            "occurrence_count": count,
            "georeferenced_occurrence_count": count,
            "range_inference_eligible_count": eligible,
            "preserved_specimen_count": sum(bool(row["preserved_specimen"]) for row in rows),
            "fossil_count": sum(bool(row["fossil"]) for row in rows),
            "geospatial_issue_count": sum(bool(row["has_geospatial_issue"]) for row in rows),
            "coordinate_uncertainty_summary": _uncertainty_summary(self.uncertainties),
            "earliest_occurrence_date": min(self.dates) if self.dates else None,
            "latest_occurrence_date": max(self.dates) if self.dates else None,
            "basis_of_record_counts": [
                {"value": value, "count": self.basis_counts[value]}
                for value in sorted(self.basis_counts)
            ],
            "establishment_means": sorted(self.establishment_means),
            "occurrence_status": statuses[0] if len(statuses) == 1 else "MIXED",
            "evidence_confidence": eligible / count if count else None,
        }


def _aggregate_spread(evidence: pl.DataFrame) -> pl.DataFrame:
    groups: dict[tuple[object, ...], _SpreadAccumulator] = {}
    for row in evidence.iter_rows(named=True):
        if not row.get("coordinate_valid") or not row.get("taxon_key_match"):
            continue
        if not row.get("spatial_cell_id") or not row.get("source_dataset_key"):
            continue
        key = (
            row["accepted_taxon_key"],
            row["source"],
            row["source_dataset_key"],
            row["spatial_resolution"],
            row["spatial_cell_id"],
            row["known_range_role"],
            row["source_snapshot_version"],
        )
        accumulator = groups.get(key)
        if accumulator is None:
            accumulator = _SpreadAccumulator(
                template={
                    "schema_version": GEOGRAPHIC_SPREAD_SCHEMA_VERSION,
                    "registry_version": row["registry_version"],
                    "accepted_taxon_key": row["accepted_taxon_key"],
                    "gbif_species_key": row["gbif_species_key"],
                    "scientific_name": row["scientific_name"],
                    "source": row["source"],
                    "source_dataset_key": row["source_dataset_key"],
                    "source_dataset_citation": None,
                    "source_query_hash": row["source_query_hash"],
                    "spatial_cell_id": row["spatial_cell_id"],
                    "spatial_resolution": row["spatial_resolution"],
                    "country_code": None,
                    "admin1": None,
                    "bioregion": None,
                    "centroid_latitude": row["centroid_latitude"],
                    "centroid_longitude": row["centroid_longitude"],
                    "occurrence_count": 0,
                    "georeferenced_occurrence_count": 0,
                    "range_inference_eligible_count": 0,
                    "preserved_specimen_count": 0,
                    "fossil_count": 0,
                    "geospatial_issue_count": 0,
                    "coordinate_uncertainty_summary": None,
                    "earliest_occurrence_date": None,
                    "latest_occurrence_date": None,
                    "basis_of_record_counts": [],
                    "establishment_means": [],
                    "occurrence_status": "",
                    "known_range_role": row["known_range_role"],
                    "evidence_confidence": None,
                    "retrieved_at": row["retrieved_at"],
                    "source_snapshot_version": row["source_snapshot_version"],
                }
            )
            groups[key] = accumulator
        accumulator.add(row)
    rows = [groups[key].output_row() for key in sorted(groups, key=_sort_key)]
    return _typed_frame(rows, geographic_spread_schema()).sort(
        [
            "accepted_taxon_key",
            "source",
            "source_dataset_key",
            "spatial_resolution",
            "spatial_cell_id",
            "known_range_role",
        ]
    )


def _load_or_initialize_state(
    path: Path,
    *,
    identity: dict[str, object],
    parts_dir: Path,
) -> tuple[dict[str, object], bool]:
    if not path.exists():
        has_atomic_parts = any(parts_dir.glob("*.parquet"))
        return (
            {**identity, "status": "running", "next_cursor": 0, "parts": []},
            has_atomic_parts,
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("geographic spread checkpoint state must be a JSON object")
    for field_name, expected in identity.items():
        if payload.get(field_name) != expected:
            raise ValueError(
                f"geographic spread checkpoint {field_name} mismatch: "
                f"found {payload.get(field_name)!r}, expected {expected!r}"
            )
    status = str(payload.get("status") or "")
    if status not in {"running", "complete"}:
        raise ValueError(f"invalid geographic spread checkpoint status: {status!r}")
    validated_cursor = _validate_checkpoint_parts(
        payload.get("parts"),
        parts_dir=parts_dir,
    )
    state_cursor = _nonnegative_int(payload.get("next_cursor"), field_name="next_cursor")
    if state_cursor != validated_cursor:
        raise ValueError(
            "geographic spread checkpoint next_cursor does not match its parts: "
            f"found {state_cursor}, expected {validated_cursor}"
        )
    recorded_files = {
        str(item["file"])
        for item in payload.get("parts", [])
        if isinstance(item, dict) and item.get("file")
    }
    orphan_files = sorted(
        path.name for path in parts_dir.glob("*.parquet") if path.name not in recorded_files
    )
    expected_orphan = f"part-{state_cursor:012d}.parquet"
    if orphan_files and (status == "complete" or orphan_files != [expected_orphan]):
        raise ValueError(
            "geographic spread checkpoint contains unexpected unrecorded parts: "
            + ", ".join(orphan_files)
        )
    return payload, True


def _validate_checkpoint_parts(value: object, *, parts_dir: Path) -> int:
    if not isinstance(value, list):
        raise ValueError("geographic spread checkpoint parts must be an array")
    expected_cursor = 0
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("geographic spread checkpoint part must be an object")
        if int(item.get("cursor", -1)) != expected_cursor:
            raise ValueError("geographic spread checkpoint parts are not cursor-contiguous")
        part = parts_dir / str(item.get("file") or "")
        if not part.is_file():
            raise ValueError(f"geographic spread checkpoint part is missing: {part}")
        if _sha256_file(part) != item.get("sha256"):
            raise ValueError(f"geographic spread checkpoint checksum mismatch: {part}")
        if part.stat().st_size != int(item.get("byte_count", -1)):
            raise ValueError(f"geographic spread checkpoint byte-count mismatch: {part}")
        frame = pl.read_parquet(part)
        if frame.schema != geographic_occurrence_evidence_schema():
            raise ValueError(f"geographic spread checkpoint schema mismatch: {part}")
        if frame.height != int(item.get("row_count", -1)):
            raise ValueError(f"geographic spread checkpoint row-count mismatch: {part}")
        expected_cursor = int(item.get("next_cursor", -1))
    return expected_cursor


def _write_or_validate_checkpoint_part(frame: pl.DataFrame, path: Path) -> None:
    if not path.exists():
        write_parquet(frame, path, overwrite=False)
        return
    existing = pl.read_parquet(path)
    if existing.schema != geographic_occurrence_evidence_schema():
        raise ValueError(f"unrecorded geographic checkpoint part has wrong schema: {path}")
    if not existing.equals(frame):
        raise ValueError(f"unrecorded geographic checkpoint part conflicts with source: {path}")


def _compact_evidence(parts_dir: Path, *, parts: list[object]) -> pl.DataFrame:
    if not parts:
        return pl.DataFrame(schema=geographic_occurrence_evidence_schema())
    rows_by_key: dict[tuple[str, int | None], dict[str, object]] = {}
    for item in parts:
        assert isinstance(item, dict)
        frame = pl.read_parquet(parts_dir / str(item["file"]))
        for row in frame.iter_rows(named=True):
            key = (str(row["gbif_id"]), row.get("spatial_resolution"))
            previous = rows_by_key.get(key)
            if previous is not None and previous != row:
                raise ValueError(
                    f"conflicting checkpoint rows for occurrence {key[0]} resolution {key[1]}"
                )
            rows_by_key[key] = row
    return _typed_frame(
        [rows_by_key[key] for key in sorted(rows_by_key, key=_sort_key)],
        geographic_occurrence_evidence_schema(),
    ).sort(["gbif_id", "spatial_resolution"])


def _build_manifest(
    *,
    identity: dict[str, object],
    state: dict[str, object],
    evidence: pl.DataFrame,
    spread: pl.DataFrame,
    evidence_path: Path,
    spread_path: Path,
    resumed: bool,
    retrieved_at: datetime,
) -> dict[str, object]:
    occurrence_rows = _one_row_per_occurrence(evidence)
    return {
        "schema_version": GEOGRAPHIC_BUILD_MANIFEST_SCHEMA_VERSION,
        **{key: value for key, value in identity.items() if key != "schema_version"},
        "status": "complete",
        "retrieved_at": retrieved_at.isoformat().replace("+00:00", "Z"),
        "resumed": resumed,
        "completed_occurrence_count": len(occurrence_rows),
        "invalid_coordinate_count": sum(
            row.get("exclusion_reason") == "invalid_coordinate" for row in occurrence_rows
        ),
        "taxon_key_mismatch_count": sum(
            row.get("exclusion_reason") == "taxon_key_mismatch" for row in occurrence_rows
        ),
        "range_inference_eligible_occurrence_count": sum(
            bool(row.get("range_inference_eligible")) and bool(row.get("coordinate_valid"))
            for row in occurrence_rows
        ),
        "evidence_row_count": evidence.height,
        "evidence_confidence_method": "eligible-occurrence-fraction-v1",
        "spread_row_count": spread.height,
        "checkpoint_part_count": len(state.get("parts") or []),
        "source_total_records": state.get("total_records"),
        "files": {
            "geographic_occurrence_evidence": _artifact_entry(evidence_path, evidence.height),
            "taxon_geographic_spread": _artifact_entry(spread_path, spread.height),
        },
    }


def _one_row_per_occurrence(evidence: pl.DataFrame) -> list[dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for row in evidence.iter_rows(named=True):
        gbif_id = str(row["gbif_id"])
        previous = rows.get(gbif_id)
        if previous is None or (
            not previous.get("coordinate_valid") and row.get("coordinate_valid")
        ):
            rows[gbif_id] = row
    return [rows[key] for key in sorted(rows)]


def _uncertainty_summary(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {"count": 0, "min_m": None, "p50_m": None, "p95_m": None, "max_m": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min_m": ordered[0],
        "p50_m": _nearest_rank(ordered, 0.50),
        "p95_m": _nearest_rank(ordered, 0.95),
        "max_m": ordered[-1],
    }


def _single_or_none(values: set[str]) -> str | None:
    return next(iter(values)) if len(values) == 1 else None


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    index = max(0, math.ceil(quantile * len(values)) - 1)
    return values[index]


def _typed_frame(
    rows: Sequence[Mapping[str, object]],
    schema: dict[str, pl.DataType],
) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    normalized = [{name: row.get(name) for name in schema} for row in rows]
    return pl.DataFrame(normalized, schema=schema, orient="row", strict=False)


def _event_date(record: Mapping[str, object]) -> date | None:
    text = _optional_text(_value(record, "eventDate", "event_date"))
    if text:
        candidate = text.split("/", 1)[0][:10]
        try:
            return date.fromisoformat(candidate)
        except ValueError:
            pass
    year = _optional_int(record.get("year"))
    month = _optional_int(record.get("month")) or 1
    day = _optional_int(record.get("day")) or 1
    if year is None:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _known_range_role(establishment_means: str) -> str:
    normalized = establishment_means.casefold()
    introduced_tokens = ("introduced", "invasive", "naturalised", "naturalized")
    if any(token in normalized for token in introduced_tokens):
        return "introduced"
    if "native" in normalized:
        return "native"
    if "vagrant" in normalized:
        return "vagrant"
    if any(token in normalized for token in ("uncertain", "unknown")):
        return "uncertain"
    return "unknown"


def _issue_values(value: object) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = str(value or "").replace(",", ";").split(";")
    return {_normalized_token(item) for item in values if _normalized_token(item)}


def _source_query_hash(accepted_taxon_key: str) -> str:
    return _sha256_json(
        {
            "source": "GBIF",
            "endpoint": "/occurrence/search|/occurrence/download/request",
            "accepted_taxon_key": accepted_taxon_key,
            "has_coordinate": True,
        }
    )


def _gbif_id(record: Mapping[str, object]) -> str:
    value = _optional_text(_value(record, "key", "gbifID", "gbif_id"))
    return value or "missing:" + _sha256_json(dict(record)).removeprefix("sha256:")


def _accepted_taxon_key(value: object) -> str:
    text = str(value or "").strip()
    bare = _bare_gbif_key(text)
    if not bare.isdigit() or int(bare) <= 0:
        raise ValueError("accepted_taxon_key must be a positive source-qualified GBIF key")
    return f"gbif:{int(bare)}"


def _bare_gbif_key(value: object) -> str:
    return str(value or "").strip().removeprefix("gbif:")


def _country_code(value: object) -> str | None:
    code = str(value or "").strip().upper()
    return code if len(code) == 2 and code.isalpha() else None


def _normalized_token(value: object) -> str:
    return str(value or "").strip().upper().replace(" ", "_")


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _required_text(value: object, *, field_name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{field_name} must be nonblank")
    return text


def _required_hash(value: object, *, field_name: str) -> str:
    text = _required_text(value, field_name=field_name)
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValueError(f"{field_name} must be a full sha256: digest")
    return text


def _optional_int(value: object) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _nonnegative_int(value: object, *, field_name: str) -> int:
    parsed = _optional_int(value)
    if parsed is None or parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return parsed


def _optional_float(value: object) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes"}


def _value(record: Mapping[str, object], *names: str) -> object:
    for name in names:
        if name in record:
            return record[name]
    return None


def _utc_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("retrieved_at must include a timezone")
    return parsed.astimezone(UTC)


def _parquet_row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    try:
        return int(parquet.metadata.num_rows)
    finally:
        parquet.close()


def _artifact_entry(path: Path, row_count: int) -> dict[str, object]:
    return {
        "file": path.name,
        "byte_count": path.stat().st_size,
        "row_count": row_count,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json_atomic(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sort_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


__all__ = [
    "GBIF_SEARCH_MAX_RECORDS",
    "GBIF_SEARCH_PAGE_LIMIT",
    "GEOGRAPHIC_OCCURRENCE_EVIDENCE_FILE",
    "GEOGRAPHIC_SPREAD_SCHEMA_VERSION",
    "GeographicSpreadBuildResult",
    "BulkDownloadRequired",
    "GBIFOccurrenceSearchSource",
    "GBIFParquetOccurrenceSource",
    "OccurrenceBatch",
    "OccurrenceBatchSource",
    "TAXON_GEOGRAPHIC_SPREAD_FILE",
    "build_gbif_bulk_download_request",
    "build_taxon_geographic_spread",
    "geographic_occurrence_evidence_schema",
    "geographic_spread_schema",
]
