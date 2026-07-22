from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.registry import CHECK_REGISTRY_VERSION


OCCURRENCE_QUALITY_VERSION = "biominer-gbif-occurrence-quality/v1"
OCCURRENCE_QUALITY_SCHEMA = pa.schema(
    [
        ("quality_version", pa.string()),
        ("check_registry_version", pa.string()),
        ("source_snapshot_id", pa.string()),
        ("occurrence_quality_id", pa.string()),
        ("gbifID", pa.string()),
        ("media_assertion_count", pa.int64()),
        ("gbif_issue_flags", pa.list_(pa.string())),
        ("gbif_id_status", pa.string()),
        ("dataset_key_status", pa.string()),
        ("occurrence_id_status", pa.string()),
        ("occurrence_identity_conflict_status", pa.string()),
        ("basis_of_record_status", pa.string()),
        ("occurrence_status_vocabulary_status", pa.string()),
        ("sex_vocabulary_status", pa.string()),
        ("event_date_status", pa.string()),
        ("temporal_component_conflict_status", pa.string()),
        ("coordinate_pair_status", pa.string()),
        ("zero_coordinate_status", pa.string()),
        ("coordinate_uncertainty_status", pa.string()),
        ("rank_name_consistency_status", pa.string()),
        ("accepted_taxon_key_status", pa.string()),
        ("taxonomic_hierarchy_status", pa.string()),
        ("identified_by_status", pa.string()),
        ("verification_source_evidence_status", pa.string()),
        ("occurrence_count_consistency_status", pa.string()),
        ("overall_occurrence_quality_status", pa.string()),
    ]
)
ISSUE_SUMMARY_SCHEMA = pa.schema(
    [
        ("quality_version", pa.string()),
        ("gbif_issue_flag", pa.string()),
        ("occurrence_count", pa.int64()),
        ("media_row_count", pa.int64()),
    ]
)
STATUS_SUMMARY_SCHEMA = pa.schema(
    [
        ("quality_version", pa.string()),
        ("check_output_field", pa.string()),
        ("status", pa.string()),
        ("occurrence_count", pa.int64()),
        ("media_row_count", pa.int64()),
    ]
)
_SOURCE_FIELDS = (
    "datasetKey",
    "occurrenceID",
    "basisOfRecord",
    "occurrenceStatus",
    "sex",
    "eventDate",
    "year",
    "month",
    "day",
    "decimalLatitude",
    "decimalLongitude",
    "coordinateUncertaintyInMeters",
    "informationWithheld",
    "dataGeneralizations",
    "taxonRank",
    "species",
    "specificEpithet",
    "genus",
    "taxonKey",
    "acceptedTaxonKey",
    "taxonomicStatus",
    "identifiedBy",
    "identificationVerificationStatus",
    "individualCount",
)
_STATUS_FIELDS = tuple(
    field.name
    for field in OCCURRENCE_QUALITY_SCHEMA
    if field.name.endswith("_status") and not field.name.startswith("overall_")
)
_PROVISIONAL_SCHEMA = OCCURRENCE_QUALITY_SCHEMA.append(
    pa.field("_dataset_key", pa.string())
).append(pa.field("_source_occurrence_id", pa.string()))


@dataclass(frozen=True, slots=True)
class OccurrenceQualityResult:
    output_directory: Path
    quality_path: Path
    manifest: dict[str, object]


@dataclass(slots=True)
class _ValueState:
    value: str | None = None
    saw_missing: bool = False
    conflict: bool = False

    def add(self, raw: object | None) -> None:
        value = _trimmed(raw)
        if value is None:
            if self.value is not None:
                self.conflict = True
            self.saw_missing = True
            return
        if self.value is None:
            self.value = value
            if self.saw_missing:
                self.conflict = True
        elif value != self.value:
            self.conflict = True


@dataclass(slots=True)
class _OccurrenceAccumulator:
    gbif_id: str
    fields: dict[str, _ValueState]
    issues: set[str]
    media_count: int = 0

    @classmethod
    def create(cls, gbif_id: str) -> _OccurrenceAccumulator:
        return cls(
            gbif_id=gbif_id,
            fields={name: _ValueState() for name in _SOURCE_FIELDS},
            issues=set(),
        )

    def add(self, values: dict[str, object | None]) -> None:
        self.media_count += 1
        for name, state in self.fields.items():
            state.add(values[name])
        issue = _trimmed(values["issue"])
        if issue:
            self.issues.update(part.strip() for part in issue.split(";") if part.strip())


def publish_occurrence_quality(
    *,
    v3_parquet: str | Path,
    output_directory: str | Path,
    source_snapshot_id: str,
    expected_media_rows: int,
    expected_occurrences: int,
    code_commit: str,
    memory_limit: str = "4GB",
    threads: int = 4,
    temp_directory: str | Path | None = None,
    batch_rows: int = 50_000,
) -> OccurrenceQualityResult:
    """Publish exact request-free occurrence checks at one row per gbifID."""

    v3 = Path(v3_parquet).resolve()
    destination = Path(output_directory).resolve()
    if not v3.is_file():
        raise FileNotFoundError(v3)
    if destination.exists():
        raise FileExistsError(destination)
    if threads < 1:
        raise ValueError("threads must be positive")
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    parquet = pq.ParquetFile(v3)
    missing = {"gbifID", "issue", *_SOURCE_FIELDS} - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"v3 lacks occurrence check fields: {sorted(missing)}")
    if parquet.metadata.num_rows != expected_media_rows:
        raise ValueError("v3 media row count differs from expected")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    temporary = (
        Path(temp_directory).resolve()
        if temp_directory is not None
        else staging / "duckdb_tmp"
    )
    temporary.mkdir(parents=True, exist_ok=True)
    quality = staging / "occurrence_quality.parquet"
    provisional = staging / "occurrence_quality_provisional.parquet"
    identity_conflicts = staging / "occurrence_identity_conflicts.parquet"
    issue_summary = staging / "gbif_issue_summary.parquet"
    status_summary = staging / "occurrence_check_status_summary.parquet"
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads = {threads}")
        connection.execute(f"SET memory_limit = {_literal(memory_limit)}")
        connection.execute(f"SET temp_directory = {_literal(str(temporary))}")
        connection.execute("SET preserve_insertion_order = false")
        provisional_counts = _write_occurrences_streaming(
            v3=v3,
            output=provisional,
            source_snapshot_id=source_snapshot_id,
            batch_rows=batch_rows,
        )
        connection.execute(
            f"COPY ({_identity_conflict_query(provisional)}) TO {_literal(str(identity_conflicts))} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        connection.execute(
            f"COPY ({_final_quality_query(provisional, identity_conflicts)}) TO {_literal(str(quality))} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"
        )
        connection.execute(
            f"COPY ({_issue_summary_query(quality)}) TO {_literal(str(issue_summary))} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        connection.execute(
            f"COPY ({_status_summary_query(quality)}) TO {_literal(str(status_summary))} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        counts = _counts(connection, quality)
        counts.update(provisional_counts)
    except BaseException:
        connection.close()
        shutil.rmtree(staging, ignore_errors=True)
        if temp_directory is None:
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        try:
            connection.close()
        except Exception:
            pass
    if temp_directory is None:
        shutil.rmtree(temporary, ignore_errors=True)
    validation = {
        "one_row_per_occurrence": counts["rows"]
        == counts["distinct_gbif_ids"]
        == expected_occurrences,
        "media_denominator_reconciles": counts["media_rows"] == expected_media_rows,
        "all_check_statuses_present": counts["rows_with_all_statuses"]
        == expected_occurrences,
        "quality_schema_matches": pq.ParquetFile(quality).schema_arrow.equals(
            OCCURRENCE_QUALITY_SCHEMA
        ),
        "issue_summary_schema_matches": pq.ParquetFile(
            issue_summary
        ).schema_arrow.equals(ISSUE_SUMMARY_SCHEMA),
        "status_summary_schema_matches": pq.ParquetFile(
            status_summary
        ).schema_arrow.equals(STATUS_SUMMARY_SCHEMA),
    }
    if not all(validation.values()):
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(f"occurrence quality validation failed: {validation}")
    provisional.unlink()
    artifacts = [
        _artifact(path)
        for path in (quality, identity_conflicts, issue_summary, status_summary)
    ]
    manifest = {
        "schema_version": OCCURRENCE_QUALITY_VERSION,
        "check_registry_version": CHECK_REGISTRY_VERSION,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "code_commit": code_commit,
        "source_snapshot_id": source_snapshot_id,
        "inputs": {"v3_parquet": str(v3)},
        "counts": counts,
        "validation": validation,
        "artifacts": artifacts,
        "network_requests": 0,
        "manifest_policy": {"written_last": True},
    }
    _write_json(staging / "manifest.json", manifest)
    for artifact in artifacts:
        _verify(staging, artifact)
    os.replace(staging, destination)
    return OccurrenceQualityResult(
        output_directory=destination,
        quality_path=destination / quality.name,
        manifest=manifest,
    )


def _write_occurrences_streaming(
    *,
    v3: Path,
    output: Path,
    source_snapshot_id: str,
    batch_rows: int,
) -> dict[str, int]:
    columns = ("gbifID", "issue", *_SOURCE_FIELDS)
    parquet = pq.ParquetFile(v3)
    writer = pq.ParquetWriter(
        output,
        _PROVISIONAL_SCHEMA,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    buffers = {field.name: [] for field in _PROVISIONAL_SCHEMA}
    current: _OccurrenceAccumulator | None = None
    media_rows = 0
    occurrence_rows = 0
    previous_gbif: str | None = None

    def emit(accumulator: _OccurrenceAccumulator) -> None:
        nonlocal occurrence_rows
        row = _occurrence_row(accumulator, source_snapshot_id)
        for name in buffers:
            buffers[name].append(row[name])
        occurrence_rows += 1
        if len(buffers["gbifID"]) >= batch_rows:
            writer.write_table(
                pa.Table.from_pydict(buffers, schema=_PROVISIONAL_SCHEMA),
                row_group_size=batch_rows,
            )
            for values in buffers.values():
                values.clear()

    try:
        for batch in parquet.iter_batches(
            batch_size=batch_rows, columns=list(columns), use_threads=True
        ):
            values = {
                name: batch.column(index).to_pylist()
                for index, name in enumerate(columns)
            }
            for index in range(batch.num_rows):
                gbif_id = _trimmed(values["gbifID"][index])
                if gbif_id is None:
                    raise ValueError("v3 contains a blank gbifID")
                if current is None or gbif_id != current.gbif_id:
                    if current is not None:
                        emit(current)
                    if previous_gbif is not None and gbif_id < previous_gbif:
                        raise ValueError("v3 is not sorted by gbifID as its lineage contract states")
                    previous_gbif = gbif_id
                    current = _OccurrenceAccumulator.create(gbif_id)
                current.add({name: values[name][index] for name in ("issue", *_SOURCE_FIELDS)})
                media_rows += 1
        if current is not None:
            emit(current)
        if buffers["gbifID"]:
            writer.write_table(
                pa.Table.from_pydict(buffers, schema=_PROVISIONAL_SCHEMA),
                row_group_size=batch_rows,
            )
    finally:
        writer.close()
    return {
        "streamed_media_rows": media_rows,
        "streamed_occurrence_rows": occurrence_rows,
    }


def _occurrence_row(
    accumulator: _OccurrenceAccumulator, source_snapshot_id: str
) -> dict[str, object]:
    field = lambda name: accumulator.fields[name].value
    conflict = lambda name: accumulator.fields[name].conflict
    gbif_status = "PASS" if accumulator.gbif_id.isascii() and accumulator.gbif_id.isdigit() else "FAIL"
    dataset_key = field("datasetKey")
    dataset_status = _value_status(
        dataset_key,
        conflict("datasetKey"),
        valid=bool(
            dataset_key
            and re.fullmatch(
                r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}",
                dataset_key,
            )
        ),
    )
    source_occurrence_id = field("occurrenceID")
    occurrence_id_status = _presence_status(
        source_occurrence_id, conflict("occurrenceID")
    )
    basis = field("basisOfRecord")
    basis_status = _vocabulary_status(
        basis,
        conflict("basisOfRecord"),
        {
            "HUMAN_OBSERVATION",
            "MACHINE_OBSERVATION",
            "PRESERVED_SPECIMEN",
            "MATERIAL_SAMPLE",
            "LIVING_SPECIMEN",
            "FOSSIL_SPECIMEN",
            "MATERIAL_CITATION",
            "OBSERVATION",
            "OCCURRENCE",
        },
    )
    occurrence_status = field("occurrenceStatus")
    occurrence_vocab = _vocabulary_status(
        occurrence_status,
        conflict("occurrenceStatus"),
        {"PRESENT", "ABSENT"},
    )
    sex = field("sex")
    sex_status = (
        "NOT_APPLICABLE"
        if sex is None
        else _vocabulary_status(
            sex,
            conflict("sex"),
            {"MALE", "FEMALE", "HERMAPHRODITE", "INDETERMINATE", "MIXED", "OTHER"},
        )
    )
    event_date = field("eventDate")
    event_status = (
        "UNKNOWN"
        if event_date is None
        else "CONFLICT"
        if conflict("eventDate")
        else "PASS"
        if _parse_event(event_date) is not None
        else "FAIL"
    )
    temporal_status = _temporal_status(
        event_date,
        field("year"),
        field("month"),
        field("day"),
        any(conflict(name) for name in ("year", "month", "day")),
    )
    coordinate_status = _coordinate_status(
        latitude=field("decimalLatitude"),
        longitude=field("decimalLongitude"),
        conflict=conflict("decimalLatitude") or conflict("decimalLongitude"),
        withheld=field("informationWithheld") is not None,
        generalized=field("dataGeneralizations") is not None,
    )
    latitude = _finite_float(field("decimalLatitude"))
    longitude = _finite_float(field("decimalLongitude"))
    zero_status = (
        "NOT_APPLICABLE"
        if coordinate_status != "PASS"
        else "FAIL"
        if latitude == 0 and longitude == 0
        else "PASS"
    )
    uncertainty = _finite_float(field("coordinateUncertaintyInMeters"))
    uncertainty_status = (
        "NOT_APPLICABLE"
        if coordinate_status != "PASS"
        else "UNKNOWN"
        if field("coordinateUncertaintyInMeters") is None
        else "FAIL"
        if uncertainty is None or uncertainty < 0
        else "PASS"
    )
    rank = (field("taxonRank") or "").upper()
    rank_status = (
        "UNKNOWN"
        if not rank
        else "CONFLICT"
        if conflict("taxonRank")
        else "FAIL"
        if rank in {"SPECIES", "SUBSPECIES", "VARIETY", "FORM", "INFRASPECIFIC_NAME", "ABERRATION"}
        and field("species") is None
        else "PASS"
    )
    taxon_key = field("taxonKey")
    accepted_key = field("acceptedTaxonKey")
    accepted_status = (
        "NOT_APPLICABLE"
        if taxon_key is None
        else "UNKNOWN"
        if accepted_key is None
        else "PASS"
        if re.fullmatch(r"[A-Za-z0-9]+", taxon_key)
        and re.fullmatch(r"[A-Za-z0-9]+", accepted_key)
        else "FAIL"
    )
    species = field("species")
    genus = field("genus")
    hierarchy_status = (
        "UNKNOWN"
        if species is None and genus is None
        else "CONFLICT"
        if species is not None
        and genus is not None
        and not species.casefold().startswith(genus.casefold() + " ")
        else "PASS"
    )
    identified_status = _presence_status(field("identifiedBy"), conflict("identifiedBy"))
    count_status = _count_status(occurrence_status, field("individualCount"))
    identity_status = (
        "NOT_APPLICABLE"
        if dataset_key is None or source_occurrence_id is None
        else "PASS"
    )
    statuses = {
        "gbif_id_status": gbif_status,
        "dataset_key_status": dataset_status,
        "occurrence_id_status": occurrence_id_status,
        "occurrence_identity_conflict_status": identity_status,
        "basis_of_record_status": basis_status,
        "occurrence_status_vocabulary_status": occurrence_vocab,
        "sex_vocabulary_status": sex_status,
        "event_date_status": event_status,
        "temporal_component_conflict_status": temporal_status,
        "coordinate_pair_status": coordinate_status,
        "zero_coordinate_status": zero_status,
        "coordinate_uncertainty_status": uncertainty_status,
        "rank_name_consistency_status": rank_status,
        "accepted_taxon_key_status": accepted_status,
        "taxonomic_hierarchy_status": hierarchy_status,
        "identified_by_status": identified_status,
        "verification_source_evidence_status": "UNKNOWN",
        "occurrence_count_consistency_status": count_status,
    }
    return {
        "quality_version": OCCURRENCE_QUALITY_VERSION,
        "check_registry_version": CHECK_REGISTRY_VERSION,
        "source_snapshot_id": source_snapshot_id,
        "occurrence_quality_id": "sha256:"
        + hashlib.sha256(
            f"{source_snapshot_id}|occurrence|{accumulator.gbif_id}".encode()
        ).hexdigest(),
        "gbifID": accumulator.gbif_id,
        "media_assertion_count": accumulator.media_count,
        "gbif_issue_flags": sorted(accumulator.issues),
        **statuses,
        "overall_occurrence_quality_status": _overall(tuple(statuses.values())),
        "_dataset_key": dataset_key,
        "_source_occurrence_id": source_occurrence_id,
    }


def _identity_conflict_query(provisional: Path) -> str:
    return f"""
        SELECT {_literal(OCCURRENCE_QUALITY_VERSION)} AS quality_version,
               _dataset_key AS dataset_key,
               _source_occurrence_id AS occurrence_id,
               count(*)::BIGINT AS occurrence_count
        FROM read_parquet({_literal(str(provisional))})
        WHERE _dataset_key IS NOT NULL AND _source_occurrence_id IS NOT NULL
        GROUP BY _dataset_key, _source_occurrence_id
        HAVING count(*) > 1
        ORDER BY occurrence_count DESC, dataset_key, occurrence_id
    """


def _final_quality_query(provisional: Path, conflicts: Path) -> str:
    base_columns = [
        field.name
        for field in OCCURRENCE_QUALITY_SCHEMA
        if field.name not in _STATUS_FIELDS
        and field.name != "overall_occurrence_quality_status"
    ]
    projected = ", ".join(f"p.{name}" for name in base_columns)
    final_statuses = [
        (
            "CASE WHEN p.occurrence_identity_conflict_status = 'NOT_APPLICABLE' "
            "THEN 'NOT_APPLICABLE' WHEN c.occurrence_count IS NOT NULL "
            "THEN 'CONFLICT' ELSE 'PASS' END"
            if name == "occurrence_identity_conflict_status"
            else f"p.{name}"
        )
        for name in _STATUS_FIELDS
    ]
    checked_columns = ", ".join(
        f"{expression} AS {name}"
        for name, expression in zip(_STATUS_FIELDS, final_statuses, strict=True)
    )
    conflict_any = " OR ".join(f"{name} = 'CONFLICT'" for name in _STATUS_FIELDS)
    fail_any = " OR ".join(f"{name} = 'FAIL'" for name in _STATUS_FIELDS)
    unknown_any = " OR ".join(
        f"{name} IN ('UNKNOWN','NOT_TESTED')" for name in _STATUS_FIELDS
    )
    final_columns = ", ".join(
        field.name for field in list(OCCURRENCE_QUALITY_SCHEMA)[:-1]
    )
    return f"""
        WITH checked AS (
          SELECT {projected}, {checked_columns}
          FROM read_parquet({_literal(str(provisional))}) p
          LEFT JOIN read_parquet({_literal(str(conflicts))}) c
            ON p._dataset_key = c.dataset_key
           AND p._source_occurrence_id = c.occurrence_id
        )
        SELECT {final_columns},
               CASE WHEN {conflict_any} THEN 'CONFLICT'
                    WHEN {fail_any} THEN 'FAIL'
                    WHEN {unknown_any} THEN 'UNKNOWN'
                    ELSE 'PASS' END AS overall_occurrence_quality_status
        FROM checked ORDER BY gbifID
    """


def _value_status(value: str | None, conflict: bool, *, valid: bool) -> str:
    if value is None:
        return "UNKNOWN"
    if conflict:
        return "CONFLICT"
    return "PASS" if valid else "FAIL"


def _presence_status(value: str | None, conflict: bool) -> str:
    if value is None:
        return "UNKNOWN"
    return "CONFLICT" if conflict else "PASS"


def _vocabulary_status(
    value: str | None, conflict: bool, allowed: set[str]
) -> str:
    return _value_status(
        value, conflict, valid=bool(value and value.upper() in allowed)
    )


def _parse_event(value: str) -> tuple[date, date] | None:
    endpoints = value.split("/")
    if len(endpoints) > 2:
        return None
    parsed = [_parse_endpoint(endpoint.strip()) for endpoint in endpoints]
    if any(item is None for item in parsed):
        return None
    start = parsed[0]
    end = parsed[-1]
    assert start is not None and end is not None
    return (start, end) if start <= end else None


def _parse_endpoint(value: str) -> date | None:
    try:
        if re.fullmatch(r"[0-9]{4}", value):
            return date(int(value), 1, 1)
        if re.fullmatch(r"[0-9]{4}-[0-9]{2}", value):
            return date.fromisoformat(value + "-01")
        if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            return date.fromisoformat(value)
        if re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}(?::[0-9]{2}(?:\.[0-9]{1,6})?)?(?:Z|[+-][0-9]{2}:[0-9]{2})?",
            value,
        ):
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None
    return None


def _temporal_status(
    event_date: str | None,
    year: str | None,
    month: str | None,
    day: str | None,
    component_conflict: bool,
) -> str:
    if event_date is None or (year is None and month is None and day is None):
        return "NOT_APPLICABLE"
    if component_conflict:
        return "CONFLICT"
    parsed = _parse_event(event_date)
    if parsed is None:
        return "UNKNOWN"
    start = parsed[0]
    for raw, expected in ((year, start.year), (month, start.month), (day, start.day)):
        if raw is None:
            continue
        try:
            if int(raw) != expected:
                return "CONFLICT"
        except ValueError:
            return "CONFLICT"
    return "PASS"


def _coordinate_status(
    *,
    latitude: str | None,
    longitude: str | None,
    conflict: bool,
    withheld: bool,
    generalized: bool,
) -> str:
    if conflict:
        return "CONFLICT"
    if latitude is None and longitude is None:
        return "WITHHELD" if withheld else "GENERALIZED" if generalized else "UNKNOWN"
    if latitude is None or longitude is None:
        return "FAIL"
    lat = _finite_float(latitude)
    lon = _finite_float(longitude)
    if lat is None or lon is None or not -90 <= lat <= 90 or not -180 <= lon <= 180:
        return "FAIL"
    return "PASS"


def _count_status(occurrence_status: str | None, count: str | None) -> str:
    if occurrence_status is None:
        return "UNKNOWN"
    if count is None:
        return "PASS"
    value = _finite_float(count)
    if value is None or value < 0:
        return "FAIL"
    if occurrence_status.upper() == "ABSENT" and value > 0:
        return "CONFLICT"
    return "PASS"


def _finite_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _overall(statuses: tuple[str, ...]) -> str:
    if "CONFLICT" in statuses:
        return "CONFLICT"
    if "FAIL" in statuses:
        return "FAIL"
    if "UNKNOWN" in statuses or "NOT_TESTED" in statuses:
        return "UNKNOWN"
    return "PASS"


def _trimmed(value: object | None) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _quality_query(v3: Path, snapshot: str) -> str:
    aggregates = []
    for field in _SOURCE_FIELDS:
        value = _nonblank(field)
        alias = _snake(field)
        aggregates.extend(
            [
                f"min({value}) AS {alias}",
                f"max({value}) AS {alias}_max",
                f"count(*) FILTER (WHERE {value} IS NULL)::BIGINT AS {alias}_missing",
            ]
        )
    issues = (
        "list_sort(list_distinct(list_filter(string_split("
        "coalesce(string_agg(DISTINCT issue, ';'), ''), ';'), x -> x <> '')))"
    )
    base = f"""
        SELECT gbifID,
               count(*)::BIGINT AS media_assertion_count,
               {issues} AS gbif_issue_flags,
               {', '.join(aggregates)}
        FROM read_parquet({_literal(str(v3))})
        GROUP BY gbifID
    """
    conflict = lambda name: (
        f"({name} IS DISTINCT FROM {name}_max OR "
        f"({name} IS NOT NULL AND {name}_missing > 0))"
    )
    dataset_conflict = conflict("dataset_key")
    occurrence_conflict = conflict("occurrence_id")
    basis_conflict = conflict("basis_of_record")
    occurrence_status_conflict = conflict("occurrence_status")
    sex_conflict = conflict("sex")
    event_conflict = conflict("event_date")
    coordinate_conflict = (
        f"{conflict('decimal_latitude')} OR {conflict('decimal_longitude')}"
    )
    rank_conflict = conflict("taxon_rank")
    identified_conflict = conflict("identified_by")
    valid_event = _valid_event("event_date")
    lat = "try_cast(decimal_latitude AS DOUBLE)"
    lon = "try_cast(decimal_longitude AS DOUBLE)"
    coordinate_status = f"""CASE
        WHEN {coordinate_conflict} THEN 'CONFLICT'
        WHEN decimal_latitude IS NULL AND decimal_longitude IS NULL
          AND information_withheld IS NOT NULL THEN 'WITHHELD'
        WHEN decimal_latitude IS NULL AND decimal_longitude IS NULL
          AND data_generalizations IS NOT NULL THEN 'GENERALIZED'
        WHEN decimal_latitude IS NULL AND decimal_longitude IS NULL THEN 'UNKNOWN'
        WHEN decimal_latitude IS NULL OR decimal_longitude IS NULL THEN 'FAIL'
        WHEN {lat} IS NULL OR {lon} IS NULL OR {lat} NOT BETWEEN -90 AND 90
          OR {lon} NOT BETWEEN -180 AND 180 THEN 'FAIL'
        ELSE 'PASS' END"""
    status_select = f"""
        SELECT *,
          CASE WHEN regexp_matches(gbifID, '^[0-9]+$') THEN 'PASS' ELSE 'FAIL' END AS gbif_id_status,
          CASE WHEN dataset_key IS NULL THEN 'UNKNOWN'
               WHEN {dataset_conflict} THEN 'CONFLICT'
               WHEN regexp_matches(dataset_key, '^[0-9a-fA-F]{{8}}-[0-9a-fA-F]{{4}}-[1-5][0-9a-fA-F]{{3}}-[89abAB][0-9a-fA-F]{{3}}-[0-9a-fA-F]{{12}}$') THEN 'PASS'
               ELSE 'FAIL' END AS dataset_key_status,
          CASE WHEN occurrence_id IS NULL THEN 'UNKNOWN'
               WHEN {occurrence_conflict} THEN 'CONFLICT' ELSE 'PASS' END AS occurrence_id_status,
          CASE WHEN dataset_key IS NULL OR occurrence_id IS NULL THEN 'NOT_APPLICABLE'
               WHEN coalesce(pair_count, 0) > 1 THEN 'CONFLICT' ELSE 'PASS' END AS occurrence_identity_conflict_status,
          CASE WHEN basis_of_record IS NULL THEN 'UNKNOWN'
               WHEN {basis_conflict} THEN 'CONFLICT'
               WHEN upper(basis_of_record) IN ('HUMAN_OBSERVATION','MACHINE_OBSERVATION','PRESERVED_SPECIMEN','MATERIAL_SAMPLE','LIVING_SPECIMEN','FOSSIL_SPECIMEN','MATERIAL_CITATION','OBSERVATION','OCCURRENCE') THEN 'PASS'
               ELSE 'FAIL' END AS basis_of_record_status,
          CASE WHEN occurrence_status IS NULL THEN 'UNKNOWN'
               WHEN {occurrence_status_conflict} THEN 'CONFLICT'
               WHEN upper(occurrence_status) IN ('PRESENT','ABSENT') THEN 'PASS' ELSE 'FAIL' END AS occurrence_status_vocabulary_status,
          CASE WHEN sex IS NULL THEN 'NOT_APPLICABLE'
               WHEN {sex_conflict} THEN 'CONFLICT'
               WHEN lower(sex) IN ('male','female','hermaphrodite','indeterminate','mixed','other') THEN 'PASS' ELSE 'FAIL' END AS sex_vocabulary_status,
          CASE WHEN event_date IS NULL THEN 'UNKNOWN'
               WHEN {event_conflict} THEN 'CONFLICT'
               WHEN {valid_event} THEN 'PASS' ELSE 'FAIL' END AS event_date_status,
          {_temporal_conflict_status(valid_event)} AS temporal_component_conflict_status,
          {coordinate_status} AS coordinate_pair_status,
          CASE WHEN coordinate_pair_status <> 'PASS' THEN 'NOT_APPLICABLE'
               WHEN {lat} = 0 AND {lon} = 0 THEN 'FAIL' ELSE 'PASS' END AS zero_coordinate_status,
          CASE WHEN coordinate_pair_status <> 'PASS' THEN 'NOT_APPLICABLE'
               WHEN coordinate_uncertainty_in_meters IS NULL THEN 'UNKNOWN'
               WHEN try_cast(coordinate_uncertainty_in_meters AS DOUBLE) IS NULL
                 OR NOT isfinite(try_cast(coordinate_uncertainty_in_meters AS DOUBLE))
                 OR try_cast(coordinate_uncertainty_in_meters AS DOUBLE) < 0 THEN 'FAIL'
               ELSE 'PASS' END AS coordinate_uncertainty_status,
          CASE WHEN taxon_rank IS NULL THEN 'UNKNOWN'
               WHEN {rank_conflict} THEN 'CONFLICT'
               WHEN upper(taxon_rank) IN ('SPECIES','SUBSPECIES','VARIETY','FORM','INFRASPECIFIC_NAME','ABERRATION')
                 AND species IS NULL THEN 'FAIL' ELSE 'PASS' END AS rank_name_consistency_status,
          CASE WHEN taxon_key IS NULL THEN 'NOT_APPLICABLE'
               WHEN accepted_taxon_key IS NULL THEN 'UNKNOWN'
               WHEN try_cast(taxon_key AS BIGINT) IS NULL OR try_cast(accepted_taxon_key AS BIGINT) IS NULL THEN 'FAIL'
               ELSE 'PASS' END AS accepted_taxon_key_status,
          CASE WHEN species IS NULL AND genus IS NULL THEN 'UNKNOWN'
               WHEN species IS NOT NULL AND genus IS NOT NULL
                 AND NOT starts_with(lower(species), lower(genus) || ' ') THEN 'CONFLICT'
               ELSE 'PASS' END AS taxonomic_hierarchy_status,
          CASE WHEN identified_by IS NULL THEN 'UNKNOWN'
               WHEN {identified_conflict} THEN 'CONFLICT' ELSE 'PASS' END AS identified_by_status,
          'UNKNOWN' AS verification_source_evidence_status,
          CASE WHEN occurrence_status IS NULL THEN 'UNKNOWN'
               WHEN individual_count IS NOT NULL AND (try_cast(individual_count AS DOUBLE) IS NULL OR try_cast(individual_count AS DOUBLE) < 0) THEN 'FAIL'
               WHEN upper(occurrence_status) = 'ABSENT' AND try_cast(individual_count AS DOUBLE) > 0 THEN 'CONFLICT'
               ELSE 'PASS' END AS occurrence_count_consistency_status
        FROM paired
    """
    status_names = ", ".join(_STATUS_FIELDS)
    conflict_any = " OR ".join(f"{name} = 'CONFLICT'" for name in _STATUS_FIELDS)
    fail_any = " OR ".join(f"{name} = 'FAIL'" for name in _STATUS_FIELDS)
    unknown_any = " OR ".join(f"{name} IN ('UNKNOWN','NOT_TESTED')" for name in _STATUS_FIELDS)
    return f"""
        WITH base AS ({base}),
        pair_counts AS (
          SELECT dataset_key, occurrence_id, count(*)::BIGINT AS pair_count
          FROM base WHERE dataset_key IS NOT NULL AND occurrence_id IS NOT NULL
          GROUP BY dataset_key, occurrence_id
        ), paired AS (
          SELECT base.*, pair_counts.pair_count
          FROM base LEFT JOIN pair_counts USING (dataset_key, occurrence_id)
        ), statuses AS ({status_select})
        SELECT {_literal(OCCURRENCE_QUALITY_VERSION)} AS quality_version,
               {_literal(CHECK_REGISTRY_VERSION)} AS check_registry_version,
               {_literal(snapshot)} AS source_snapshot_id,
               'sha256:' || sha256({_literal(snapshot)} || '|occurrence|' || gbifID) AS occurrence_quality_id,
               gbifID,
               media_assertion_count,
               gbif_issue_flags,
               {status_names},
               CASE WHEN {conflict_any} THEN 'CONFLICT'
                    WHEN {fail_any} THEN 'FAIL'
                    WHEN {unknown_any} THEN 'UNKNOWN'
                    ELSE 'PASS' END AS overall_occurrence_quality_status
        FROM statuses
        ORDER BY gbifID
    """


def _temporal_conflict_status(valid_event: str) -> str:
    event_year = "try_cast(substr(event_date, 1, 4) AS INTEGER)"
    event_month = "try_cast(substr(event_date, 6, 2) AS INTEGER)"
    event_day = "try_cast(substr(event_date, 9, 2) AS INTEGER)"
    conflict = (
        f"(year IS NOT NULL AND try_cast(year AS INTEGER) IS DISTINCT FROM {event_year}) OR "
        f"(month IS NOT NULL AND length(event_date) >= 7 AND try_cast(month AS INTEGER) IS DISTINCT FROM {event_month}) OR "
        f"(day IS NOT NULL AND length(event_date) >= 10 AND try_cast(day AS INTEGER) IS DISTINCT FROM {event_day})"
    )
    component_variants = " OR ".join(
        f"({name} IS DISTINCT FROM {name}_max OR ({name} IS NOT NULL AND {name}_missing > 0))"
        for name in ("year", "month", "day")
    )
    return f"""CASE
        WHEN event_date IS NULL OR (year IS NULL AND month IS NULL AND day IS NULL) THEN 'NOT_APPLICABLE'
        WHEN {component_variants} THEN 'CONFLICT'
        WHEN NOT ({valid_event}) THEN 'UNKNOWN'
        WHEN {conflict} THEN 'CONFLICT' ELSE 'PASS' END"""


def _valid_event(field: str) -> str:
    endpoint = lambda value: (
        f"(regexp_matches({value}, '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}(?:T.*)?$') "
        f"AND try_cast(substr({value}, 1, 10) AS DATE) IS NOT NULL)"
    )
    return f"""(
      (regexp_matches({field}, '^[0-9]{{4}}$') AND try_cast({field} AS INTEGER) BETWEEN 1 AND 9999)
      OR (regexp_matches({field}, '^[0-9]{{4}}-[0-9]{{2}}$') AND try_cast(substr({field}, 6, 2) AS INTEGER) BETWEEN 1 AND 12)
      OR ({endpoint(field)})
      OR (regexp_matches({field}, '^[^/]+/[^/]+$')
          AND {endpoint(f"split_part({field}, '/', 1)")}
          AND {endpoint(f"split_part({field}, '/', 2)")}
          AND try_cast(substr(split_part({field}, '/', 1), 1, 10) AS DATE)
              <= try_cast(substr(split_part({field}, '/', 2), 1, 10) AS DATE))
    )"""


def _issue_summary_query(quality: Path) -> str:
    return f"""
      SELECT {_literal(OCCURRENCE_QUALITY_VERSION)} AS quality_version,
             flag AS gbif_issue_flag,
             count(*)::BIGINT AS occurrence_count,
             sum(media_assertion_count)::BIGINT AS media_row_count
      FROM read_parquet({_literal(str(quality))}), unnest(gbif_issue_flags) AS flags(flag)
      GROUP BY flag ORDER BY occurrence_count DESC, flag
    """


def _status_summary_query(quality: Path) -> str:
    parts = [
        f"SELECT {_literal(OCCURRENCE_QUALITY_VERSION)} AS quality_version, "
        f"{_literal(field)} AS check_output_field, {field} AS status, "
        "count(*)::BIGINT AS occurrence_count, sum(media_assertion_count)::BIGINT AS media_row_count "
        f"FROM read_parquet({_literal(str(quality))}) GROUP BY {field}"
        for field in (*_STATUS_FIELDS, "overall_occurrence_quality_status")
    ]
    return " UNION ALL ".join(parts)


def _counts(connection: duckdb.DuckDBPyConnection, quality: Path) -> dict[str, int]:
    all_present = " AND ".join(f"{field} IS NOT NULL" for field in _STATUS_FIELDS)
    row = connection.execute(
        f"""SELECT count(*)::BIGINT, count(DISTINCT gbifID)::BIGINT,
                   sum(media_assertion_count)::BIGINT,
                   count(*) FILTER (WHERE {all_present})::BIGINT
            FROM read_parquet({_literal(str(quality))})"""
    ).fetchone()
    assert row is not None
    return dict(
        zip(
            ("rows", "distinct_gbif_ids", "media_rows", "rows_with_all_statuses"),
            map(int, row),
            strict=True,
        )
    )


def _nonblank(field: str) -> str:
    return f"nullif(trim(cast(\"{field}\" AS VARCHAR)), '')"


def _snake(value: str) -> str:
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first).lower()


def _artifact(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    return {
        "path": path.name,
        "physical_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "row_count": parquet.metadata.num_rows,
        "column_count": len(parquet.schema_arrow),
        "row_group_count": parquet.metadata.num_row_groups,
    }


def _verify(root: Path, artifact: dict[str, object]) -> None:
    path = root / str(artifact["path"])
    if _sha256(path) != artifact["sha256"]:
        raise ValueError(f"occurrence quality checksum mismatch: {path.name}")
    if pq.ParquetFile(path).metadata.num_rows != artifact["row_count"]:
        raise ValueError(f"occurrence quality row count mismatch: {path.name}")


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


__all__ = [
    "OCCURRENCE_QUALITY_SCHEMA",
    "OCCURRENCE_QUALITY_VERSION",
    "OccurrenceQualityResult",
    "publish_occurrence_quality",
]
