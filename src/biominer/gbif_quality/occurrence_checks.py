from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
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


@dataclass(frozen=True, slots=True)
class OccurrenceQualityResult:
    output_directory: Path
    quality_path: Path
    manifest: dict[str, object]


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
    issue_summary = staging / "gbif_issue_summary.parquet"
    status_summary = staging / "occurrence_check_status_summary.parquet"
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads = {threads}")
        connection.execute(f"SET memory_limit = {_literal(memory_limit)}")
        connection.execute(f"SET temp_directory = {_literal(str(temporary))}")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute(
            f"COPY ({_quality_query(v3, source_snapshot_id)}) TO {_literal(str(quality))} "
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
    artifacts = [_artifact(path) for path in (quality, issue_summary, status_summary)]
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
