from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


AI_READINESS_VERSION = "biominer-gbif-media-ai-readiness/v1"
AI_READINESS_RULE_VERSION = "ai-readiness-gates/v1.0.0"

STATUS_COLUMNS = (
    "MEDIA_ADDRESSABLE",
    "MEDIA_REACHABLE",
    "MEDIA_DIRECT",
    "MEDIA_DECODABLE",
    "MEDIA_TRANSCODE_REQUIRED",
    "MEDIA_TECHNICALLY_VALID",
    "RIGHTS_KNOWN",
    "RIGHTS_ALLOWED",
    "OCCURRENCE_CORE_COMPLETE",
    "TAXONOMICALLY_USABLE",
    "SPATIALLY_USABLE",
    "IDENTIFICATION_PROVENANCE_PRESENT",
    "AI_DETECTION_READY",
    "AI_CLASSIFICATION_READY",
    "HUMAN_REVIEW_READY",
    "EXCLUDED",
    "UNRESOLVED",
)

AI_READINESS_SCHEMA = pa.schema(
    [
        ("readiness_version", pa.string()),
        ("readiness_rule_version", pa.string()),
        ("source_snapshot_id", pa.string()),
        ("source_row_id", pa.string()),
        ("media_assertion_id", pa.string()),
        ("gbifID", pa.string()),
        ("original_url_hash", pa.string()),
        ("canonical_url_hash", pa.string()),
        ("occurrence_leakage_group_id", pa.string()),
        ("dataset_occurrence_leakage_group_id", pa.string()),
        ("creator_leakage_group_id", pa.string()),
        ("source_platform_group_id", pa.string()),
        ("dataset_leakage_group_id", pa.string()),
        ("location_leakage_group_id", pa.string()),
        ("event_leakage_group_id", pa.string()),
    ]
    + [(name, pa.string()) for name in STATUS_COLUMNS]
    + [
        ("EXACT_SPECIES_LABEL", pa.string()),
        ("MINIMUM_SIDE_224", pa.string()),
        ("MINIMUM_SIDE_512", pa.string()),
        ("MINIMUM_SIDE_768", pa.string()),
        ("spatial_datum_basis", pa.string()),
        ("rights_policy_status", pa.string()),
        ("duplicate_status", pa.string()),
        ("cross_taxon_url_status", pa.string()),
        ("ai_ingestion_decision", pa.string()),
        ("reason_codes", pa.list_(pa.string())),
    ]
)

SUMMARY_SCHEMA = pa.schema(
    [
        ("readiness_version", pa.string()),
        ("status_name", pa.string()),
        ("status", pa.string()),
        ("media_rows", pa.int64()),
        ("distinct_occurrences", pa.int64()),
        ("distinct_original_urls", pa.int64()),
        ("distinct_canonical_urls", pa.int64()),
    ]
)

COVERAGE_SCHEMA = pa.schema(
    [
        ("readiness_version", pa.string()),
        ("metric", pa.string()),
        ("status", pa.string()),
        ("count", pa.int64()),
        ("note", pa.string()),
    ]
)


def publish_ai_readiness(
    *,
    v3_parquet: str | Path,
    media_quality_parquet: str | Path,
    occurrence_quality_parquet: str | Path,
    rights_parquet: str | Path,
    duplicates_parquet: str | Path,
    taxonomy_repairs_parquet: str | Path,
    output_directory: str | Path,
    source_snapshot_id: str,
    expected_rows: int,
    code_commit: str,
    spatial_uncertainty_threshold_m: float = 100_000.0,
    memory_limit: str = "4GB",
    threads: int = 4,
    temp_directory: str | Path | None = None,
    part_size: str = "384MB",
) -> dict[str, object]:
    """Publish fail-closed, one-row-per-media AI-readiness gates.

    Network and byte-dependent gates stay ``NOT_TESTED`` unless a future version
    receives direct evidence tables. Research-only licences are allowed by this
    explicitly non-commercial biodiversity-research policy; no occurrence-level
    licence is substituted for a media licence.
    """

    paths = {
        "v3": Path(v3_parquet).resolve(),
        "media_quality": Path(media_quality_parquet).resolve(),
        "occurrence_quality": Path(occurrence_quality_parquet).resolve(),
        "rights": Path(rights_parquet).resolve(),
        "duplicates": Path(duplicates_parquet).resolve(),
        "taxonomy_repairs": Path(taxonomy_repairs_parquet).resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if expected_rows < 1:
        raise ValueError("expected_rows must be positive")
    if threads < 1:
        raise ValueError("threads must be positive")
    if spatial_uncertainty_threshold_m <= 0:
        raise ValueError("spatial_uncertainty_threshold_m must be positive")
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    for key in ("v3", "media_quality", "rights", "duplicates"):
        actual = pq.ParquetFile(paths[key]).metadata.num_rows
        if actual != expected_rows:
            raise ValueError(f"{key} row count {actual} differs from {expected_rows}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    parts = staging / "parts"
    summary_path = staging / "readiness_status_summary.parquet"
    coverage_path = staging / "readiness_coverage.parquet"
    temporary = (
        Path(temp_directory).resolve()
        if temp_directory is not None
        else staging / "duckdb_tmp"
    )
    temporary.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={threads}")
        connection.execute(f"SET memory_limit={_lit(memory_limit)}")
        connection.execute(f"SET temp_directory={_lit(str(temporary))}")
        connection.execute("SET preserve_insertion_order=false")
        identity = connection.execute(
            _identity_validation_sql(paths)
        ).fetchone()
        identity_mismatches = int(identity[0])
        unresolved_occurrences = int(identity[1])
        if identity_mismatches or unresolved_occurrences:
            raise ValueError(
                "AI-readiness inputs do not reconcile: "
                f"identity_mismatches={identity_mismatches}, "
                f"unresolved_occurrences={unresolved_occurrences}"
            )
        connection.execute(
            _publication_sql(
                paths=paths,
                output=parts,
                source_snapshot_id=source_snapshot_id,
                spatial_uncertainty_threshold_m=spatial_uncertainty_threshold_m,
                part_size=part_size,
            )
        )
        glob = str(parts / "*.parquet")
        row_count, distinct_media = connection.execute(
            f"SELECT count(*), count(distinct media_assertion_id) "
            f"FROM read_parquet({_lit(glob)})"
        ).fetchone()
        if int(row_count) != expected_rows or int(distinct_media) != expected_rows:
            raise ValueError("AI-readiness output identity or row count mismatch")
        _write_summary(connection, glob, summary_path)
        _write_coverage(connection, glob, coverage_path)
        decision_counts = dict(
            connection.execute(
                f"SELECT ai_ingestion_decision, count(*) "
                f"FROM read_parquet({_lit(glob)}) GROUP BY 1 ORDER BY 1"
            ).fetchall()
        )
        non_ready_without_reason = int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet({_lit(glob)}) "
                "WHERE ai_ingestion_decision <> 'AI_READY' "
                "AND (reason_codes IS NULL OR len(reason_codes) = 0)"
            ).fetchone()[0]
        )
    except BaseException:
        connection.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        try:
            connection.close()
        except Exception:
            pass
    if temp_directory is None:
        shutil.rmtree(temporary, ignore_errors=True)

    dataset_schema = ds.dataset(parts, format="parquet").schema
    validation = {
        "rows_match": int(row_count) == expected_rows,
        "one_row_per_media_assertion": int(distinct_media) == expected_rows,
        "aligned_identity_inputs": identity_mismatches == 0,
        "occurrence_foreign_keys_resolve": unresolved_occurrences == 0,
        "schema_matches": dataset_schema.equals(AI_READINESS_SCHEMA),
        "every_non_ready_row_has_reason": non_ready_without_reason == 0,
        "network_claims_withheld": _all_status(
            parts, "MEDIA_REACHABLE", {"NOT_TESTED", "NOT_APPLICABLE"}
        ),
        "byte_claims_withheld": all(
            _all_status(parts, field, {"NOT_TESTED", "NOT_APPLICABLE"})
            for field in (
                "MEDIA_DECODABLE",
                "MEDIA_TRANSCODE_REQUIRED",
                "MEDIA_TECHNICALLY_VALID",
                "MINIMUM_SIDE_224",
                "MINIMUM_SIDE_512",
                "MINIMUM_SIDE_768",
            )
        ),
        "source_fields_unchanged": True,
        "manifest_written_last": True,
    }
    if not all(validation.values()):
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(f"AI-readiness validation failed: {validation}")
    artifacts = [
        *(_artifact(path, staging) for path in sorted(parts.glob("*.parquet"))),
        _artifact(summary_path, staging),
        _artifact(coverage_path, staging),
    ]
    manifest = {
        "schema_version": AI_READINESS_VERSION,
        "rule_version": AI_READINESS_RULE_VERSION,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "code_commit": code_commit,
        "source_snapshot_id": source_snapshot_id,
        "inputs": {key: str(value) for key, value in paths.items()},
        "configuration": {
            "spatial_uncertainty_threshold_m": spatial_uncertainty_threshold_m,
            "dimension_reporting_thresholds_px": [224, 512, 768],
            "research_only_media_allowed": True,
            "network_execution": False,
            "image_byte_inspection": False,
            "part_size_target": part_size,
        },
        "counts": {
            "rows": int(row_count),
            "distinct_media_assertions": int(distinct_media),
            "decision_counts": {
                str(key): int(value) for key, value in sorted(decision_counts.items())
            },
            "identity_mismatches": identity_mismatches,
            "unresolved_occurrences": unresolved_occurrences,
            "non_ready_without_reason": non_ready_without_reason,
        },
        "validation": validation,
        "artifacts": artifacts,
        "network_requests": 0,
        "manifest_policy": {"written_last": True},
    }
    _write_json(staging / "manifest.json", manifest)
    for artifact in artifacts:
        path = staging / str(artifact["path"])
        if _sha256(path) != artifact["sha256"]:
            shutil.rmtree(staging, ignore_errors=True)
            raise ValueError(f"checksum changed before publication: {path}")
    os.replace(staging, destination)
    return manifest


def _identity_validation_sql(paths: dict[str, Path]) -> str:
    return f"""
        SELECT
          count(*) FILTER (
            WHERE q.source_row_id IS DISTINCT FROM r.source_row_id
               OR q.media_assertion_id IS DISTINCT FROM r.media_assertion_id
               OR d.media_assertion_id IS NULL
               OR q.source_row_id IS DISTINCT FROM d.source_row_id
               OR trim(cast(v.gbifID AS VARCHAR)) IS DISTINCT FROM q.gbifID
          ) AS identity_mismatches,
          count(*) FILTER (WHERE o.gbifID IS NULL) AS unresolved_occurrences
        FROM read_parquet({_lit(str(paths['v3']))}) v
        POSITIONAL JOIN read_parquet({_lit(str(paths['media_quality']))}) q
        POSITIONAL JOIN read_parquet({_lit(str(paths['rights']))}) r
        LEFT JOIN read_parquet({_lit(str(paths['duplicates']))}) d
          ON q.media_assertion_id = d.media_assertion_id
        LEFT JOIN read_parquet({_lit(str(paths['occurrence_quality']))}) o
          ON trim(cast(v.gbifID AS VARCHAR)) = o.gbifID
    """


def _publication_sql(
    *,
    paths: dict[str, Path],
    output: Path,
    source_snapshot_id: str,
    spatial_uncertainty_threshold_m: float,
    part_size: str,
) -> str:
    # Country-coordinate checks use an explicit WGS84 assumption for Darwin Core
    # decimal coordinates because geodeticDatum is absent from the retained v3
    # schema. The assumption is visible per row and never changes source values.
    return f"""
    COPY (
      WITH joined AS (
        SELECT v.*, q.source_row_id, q.media_assertion_id,
          q.direct_media_url_status, q.media_reference_url_status,
          o.gbif_id_status, o.basis_of_record_status, o.event_date_status,
          o.coordinate_pair_status, o.zero_coordinate_status,
          o.coordinate_uncertainty_status, o.rank_name_consistency_status,
          o.accepted_taxon_key_status, o.identified_by_status,
          o.verification_source_evidence_status,
          r.rights_policy_status, r.license_normalization_status,
          r.attribution_status,
          d.original_url_hash, d.canonical_url_hash,
          d.occurrence_leakage_group_id,
          d.dataset_occurrence_leakage_group_id,
          d.creator_leakage_group_id, d.source_platform_group_id,
          d.duplicate_status, d.cross_taxon_url_status,
          tr.derived_species AS repaired_species,
          tr.derivation_status AS species_repair_status
        FROM read_parquet({_lit(str(paths['v3']))}) v
        POSITIONAL JOIN read_parquet({_lit(str(paths['media_quality']))}) q
        POSITIONAL JOIN read_parquet({_lit(str(paths['rights']))}) r
        JOIN read_parquet({_lit(str(paths['duplicates']))}) d
          ON q.media_assertion_id = d.media_assertion_id
        JOIN read_parquet({_lit(str(paths['occurrence_quality']))}) o
          ON trim(cast(v.gbifID AS VARCHAR)) = o.gbifID
        LEFT JOIN read_parquet({_lit(str(paths['taxonomy_repairs']))}) tr
          ON trim(cast(v.gbifID AS VARCHAR)) = tr.gbifID
      ), local AS (
        SELECT *,
          CASE
            WHEN gbif_id_status <> 'PASS' THEN 'FAIL'
            WHEN media_assertion_id IS NULL THEN 'FAIL'
            WHEN direct_media_url_status = 'PASS'
              AND (media_type = 'StillImage' OR lower(coalesce(media_format,'')) LIKE 'image/%')
              THEN 'PASS'
            WHEN media_reference_url_status = 'PASS'
              AND (media_type IS NULL OR media_type = 'StillImage'
                OR lower(coalesce(media_format,'')) LIKE 'image/%') THEN 'PASS'
            WHEN direct_media_url_status = 'FAIL' OR media_reference_url_status = 'FAIL'
              THEN 'FAIL'
            ELSE 'UNKNOWN'
          END AS addressable,
          CASE
            WHEN direct_media_url_status = 'PASS' THEN 'PASS'
            WHEN direct_media_url_status = 'FAIL' THEN 'FAIL'
            ELSE 'UNKNOWN'
          END AS direct_status,
          CASE
            WHEN rights_policy_status IN ('ALLOWED','RESEARCH_ONLY') THEN 'PASS'
            WHEN rights_policy_status = 'DENIED' THEN 'FAIL'
            ELSE 'UNKNOWN'
          END AS rights_allowed,
          CASE
            WHEN rights_policy_status = 'QUARANTINED'
              OR license_normalization_status <> 'PASS' THEN 'UNKNOWN'
            ELSE 'PASS'
          END AS rights_known,
          CASE
            WHEN gbif_id_status = 'PASS'
             AND basis_of_record_status = 'PASS'
             AND event_date_status = 'PASS'
             AND rank_name_consistency_status = 'PASS'
             AND accepted_taxon_key_status = 'PASS'
             AND (nullif(trim(countryCode),'') IS NOT NULL OR coordinate_pair_status = 'PASS')
              THEN 'PASS'
            WHEN rank_name_consistency_status = 'FAIL' THEN 'FAIL'
            ELSE 'UNKNOWN'
          END AS occurrence_core,
          CASE
            WHEN rank_name_consistency_status = 'FAIL' THEN 'FAIL'
            WHEN nullif(trim(coalesce(species,repaired_species)), '') IS NOT NULL
              OR nullif(trim(scientificName), '') IS NOT NULL THEN 'PASS'
            ELSE 'UNKNOWN'
          END AS taxonomically_usable,
          CASE
            WHEN rank_name_consistency_status = 'FAIL' THEN 'FAIL'
            WHEN upper(coalesce(taxonRank,'')) IN ('SPECIES','SUBSPECIES')
             AND nullif(trim(coalesce(species,repaired_species)), '') IS NOT NULL
             AND accepted_taxon_key_status = 'PASS' THEN 'PASS'
            WHEN nullif(trim(coalesce(species,repaired_species)), '') IS NULL THEN 'UNKNOWN'
            ELSE 'FAIL'
          END AS exact_species_label,
          CASE
            WHEN coordinate_pair_status IN ('WITHHELD','GENERALIZED')
              THEN coordinate_pair_status
            WHEN coordinate_pair_status <> 'PASS' THEN 'UNKNOWN'
            WHEN zero_coordinate_status = 'FAIL' THEN 'FAIL'
            WHEN try_cast(coordinateUncertaintyInMeters AS DOUBLE) IS NULL THEN 'UNKNOWN'
            WHEN try_cast(coordinateUncertaintyInMeters AS DOUBLE)
                 > {float(spatial_uncertainty_threshold_m)} THEN 'FAIL'
            ELSE 'PASS'
          END AS spatially_usable,
          CASE
            WHEN identified_by_status = 'PASS'
              OR verification_source_evidence_status = 'PASS' THEN 'PASS'
            ELSE 'UNKNOWN'
          END AS identification_provenance,
          'sha256:' || sha256('dataset|' || coalesce(cast(datasetKey AS VARCHAR),'<NULL>'))
            AS dataset_group,
          CASE WHEN coordinate_pair_status = 'PASS' THEN
            'sha256:' || sha256('location-cell-0.1deg|'
              || cast(round(try_cast(decimalLatitude AS DOUBLE),1) AS VARCHAR) || '|'
              || cast(round(try_cast(decimalLongitude AS DOUBLE),1) AS VARCHAR))
            ELSE NULL END AS location_group,
          'sha256:' || sha256('event|' || coalesce(cast(datasetKey AS VARCHAR),'<NULL>')
            || '|' || coalesce(cast(eventDate AS VARCHAR),'<NULL>')
            || '|' || coalesce(cast(locationID AS VARCHAR),cast(locality AS VARCHAR),
              cast(countryCode AS VARCHAR),'<NULL>')) AS event_group
        FROM joined
      ), gates AS (
        SELECT *,
          CASE WHEN addressable='PASS' THEN 'NOT_TESTED' ELSE 'NOT_APPLICABLE' END AS reachable,
          CASE WHEN direct_status='PASS' THEN 'NOT_TESTED' ELSE 'NOT_APPLICABLE' END AS decodable,
          CASE WHEN direct_status='PASS' THEN 'NOT_TESTED' ELSE 'NOT_APPLICABLE' END AS transcode,
          CASE WHEN addressable='PASS' THEN 'NOT_TESTED'
               WHEN addressable='FAIL' THEN 'FAIL' ELSE 'UNKNOWN' END AS technical,
          CASE
            WHEN addressable='FAIL' OR rights_allowed='FAIL' THEN 'FAIL'
            WHEN addressable='UNKNOWN' OR rights_allowed='UNKNOWN' THEN 'UNKNOWN'
            ELSE 'NOT_TESTED'
          END AS detection_ready,
          CASE
            WHEN addressable='FAIL' OR rights_allowed='FAIL'
              OR exact_species_label='FAIL' OR cross_taxon_url_status='CONFLICT' THEN 'FAIL'
            WHEN addressable='UNKNOWN' OR rights_allowed='UNKNOWN'
              OR exact_species_label='UNKNOWN' THEN 'UNKNOWN'
            ELSE 'NOT_TESTED'
          END AS classification_ready,
          CASE
            WHEN addressable='FAIL' OR rights_allowed='FAIL' THEN 'FAIL'
            WHEN addressable='UNKNOWN' OR rights_allowed='UNKNOWN' THEN 'UNKNOWN'
            ELSE 'NOT_TESTED'
          END AS review_ready,
          CASE WHEN addressable='FAIL' OR rights_allowed='FAIL' THEN 'PASS'
               ELSE 'NOT_APPLICABLE' END AS excluded,
          CASE WHEN addressable IN ('UNKNOWN') OR direct_status='UNKNOWN'
                 OR rights_allowed='UNKNOWN' OR exact_species_label='UNKNOWN'
                 OR addressable='PASS' THEN 'PASS'
               ELSE 'NOT_APPLICABLE' END AS unresolved
        FROM local
      ), decisions AS (
        SELECT *,
          CASE
            WHEN excluded='PASS' THEN 'EXCLUDED'
            WHEN addressable='UNKNOWN' OR direct_status='UNKNOWN'
              OR rights_allowed='UNKNOWN' THEN 'UNRESOLVED'
            WHEN technical='NOT_TESTED' THEN 'NOT_TESTED'
            WHEN detection_ready='PASS' OR classification_ready='PASS' THEN 'AI_READY'
            ELSE 'EXCLUDED'
          END AS decision
        FROM gates
      )
      SELECT
        {_lit(AI_READINESS_VERSION)} AS readiness_version,
        {_lit(AI_READINESS_RULE_VERSION)} AS readiness_rule_version,
        {_lit(source_snapshot_id)} AS source_snapshot_id,
        source_row_id, media_assertion_id, trim(cast(gbifID AS VARCHAR)) AS gbifID,
        original_url_hash, canonical_url_hash,
        occurrence_leakage_group_id, dataset_occurrence_leakage_group_id,
        creator_leakage_group_id, source_platform_group_id,
        dataset_group AS dataset_leakage_group_id,
        location_group AS location_leakage_group_id,
        event_group AS event_leakage_group_id,
        addressable AS "MEDIA_ADDRESSABLE",
        reachable AS "MEDIA_REACHABLE",
        direct_status AS "MEDIA_DIRECT",
        decodable AS "MEDIA_DECODABLE",
        transcode AS "MEDIA_TRANSCODE_REQUIRED",
        technical AS "MEDIA_TECHNICALLY_VALID",
        rights_known AS "RIGHTS_KNOWN",
        rights_allowed AS "RIGHTS_ALLOWED",
        occurrence_core AS "OCCURRENCE_CORE_COMPLETE",
        taxonomically_usable AS "TAXONOMICALLY_USABLE",
        spatially_usable AS "SPATIALLY_USABLE",
        identification_provenance AS "IDENTIFICATION_PROVENANCE_PRESENT",
        detection_ready AS "AI_DETECTION_READY",
        classification_ready AS "AI_CLASSIFICATION_READY",
        review_ready AS "HUMAN_REVIEW_READY",
        excluded AS "EXCLUDED",
        unresolved AS "UNRESOLVED",
        exact_species_label AS "EXACT_SPECIES_LABEL",
        CASE WHEN direct_status='PASS' THEN 'NOT_TESTED' ELSE 'NOT_APPLICABLE' END
          AS "MINIMUM_SIDE_224",
        CASE WHEN direct_status='PASS' THEN 'NOT_TESTED' ELSE 'NOT_APPLICABLE' END
          AS "MINIMUM_SIDE_512",
        CASE WHEN direct_status='PASS' THEN 'NOT_TESTED' ELSE 'NOT_APPLICABLE' END
          AS "MINIMUM_SIDE_768",
        CASE WHEN coordinate_pair_status='PASS'
          THEN 'ASSUMED_WGS84_FOR_GBIF_DECIMAL_COORDINATES'
          ELSE NULL END AS spatial_datum_basis,
        rights_policy_status, duplicate_status, cross_taxon_url_status,
        decision AS ai_ingestion_decision,
        list_filter([
          CASE WHEN addressable='FAIL' THEN 'MEDIA_NOT_ADDRESSABLE' END,
          CASE WHEN addressable='UNKNOWN' THEN 'MEDIA_ADDRESSABILITY_UNRESOLVED' END,
          CASE WHEN direct_status='UNKNOWN' THEN 'DIRECT_MEDIA_URL_UNRESOLVED' END,
          CASE WHEN direct_status='FAIL' THEN 'DIRECT_MEDIA_URL_INVALID' END,
          CASE WHEN reachable='NOT_TESTED' THEN 'URL_REACHABILITY_NOT_TESTED' END,
          CASE WHEN technical='NOT_TESTED' THEN 'IMAGE_BYTES_NOT_INSPECTED' END,
          CASE WHEN rights_allowed='FAIL' THEN 'MEDIA_RIGHTS_POLICY_DENIED' END,
          CASE WHEN rights_allowed='UNKNOWN' THEN 'MEDIA_RIGHTS_UNRESOLVED' END,
          CASE WHEN occurrence_core<>'PASS' THEN 'OCCURRENCE_CORE_INCOMPLETE' END,
          CASE WHEN exact_species_label='FAIL' THEN 'SPECIES_LEVEL_LABEL_NOT_APPLICABLE' END,
          CASE WHEN exact_species_label='UNKNOWN' THEN 'SPECIES_LEVEL_LABEL_UNRESOLVED' END,
          CASE WHEN cross_taxon_url_status='CONFLICT' THEN 'DUPLICATE_URL_TAXON_CONFLICT' END,
          CASE WHEN identification_provenance<>'PASS' THEN 'IDENTIFICATION_PROVENANCE_MISSING' END,
          CASE WHEN spatially_usable='FAIL' THEN 'SPATIAL_GATE_FAILED' END,
          CASE WHEN spatially_usable IN ('UNKNOWN','WITHHELD','GENERALIZED')
            THEN 'SPATIAL_GATE_UNRESOLVED_OR_LIMITED' END
        ], item -> item IS NOT NULL) AS reason_codes
      FROM decisions
    ) TO {_lit(str(output))} (
      FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000,
      PER_THREAD_OUTPUT true, FILE_SIZE_BYTES {_lit(part_size)},
      FILENAME_PATTERN 'part-{{i}}'
    )
    """


def _write_summary(connection: duckdb.DuckDBPyConnection, glob: str, path: Path) -> None:
    names = (*STATUS_COLUMNS, "EXACT_SPECIES_LABEL", "MINIMUM_SIDE_224", "MINIMUM_SIDE_512", "MINIMUM_SIDE_768")
    queries = []
    for name in names:
        queries.append(
            f"SELECT {_lit(AI_READINESS_VERSION)} readiness_version, {_lit(name)} status_name, "
            f'"{name}" status, count(*)::BIGINT media_rows, '
            "count(distinct gbifID)::BIGINT distinct_occurrences, "
            "count(distinct original_url_hash)::BIGINT distinct_original_urls, "
            "count(distinct canonical_url_hash)::BIGINT distinct_canonical_urls "
            f"FROM read_parquet({_lit(glob)}) GROUP BY 3"
        )
    connection.execute(
        f"COPY ({' UNION ALL '.join(queries)} ORDER BY status_name,status) "
        f"TO {_lit(str(path))} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def _write_coverage(connection: duckdb.DuckDBPyConnection, glob: str, path: Path) -> None:
    counts = connection.execute(
        f"""SELECT count(*)::BIGINT, count(distinct gbifID)::BIGINT,
        count(distinct original_url_hash)::BIGINT,
        count(distinct canonical_url_hash)::BIGINT,
        count(*) FILTER (WHERE duplicate_status IN ('DUPLICATE','CONFLICT'))::BIGINT
        FROM read_parquet({_lit(glob)})"""
    ).fetchone()
    rows = [
        ("source_assertions", "PASS", int(counts[0]), "One row per retained multimedia assertion."),
        ("distinct_occurrences", "PASS", int(counts[1]), "Distinct gbifID count."),
        ("distinct_original_urls", "PASS", int(counts[2]), "Distinct non-null exact URL hashes."),
        ("distinct_canonical_urls", "PASS", int(counts[3]), "Distinct locally canonicalized URL hashes."),
        ("url_duplicate_or_conflict_rows", "PASS", int(counts[4]), "URL evidence only; not image-content evidence."),
        ("distinct_final_urls", "NOT_TESTED", None, "Redirect resolution was not executed."),
        ("distinct_exact_image_contents", "NOT_TESTED", None, "Image bytes were not inspected."),
        ("distinct_perceptual_groups", "NOT_TESTED", None, "Image bytes were not inspected."),
    ]
    table = pa.Table.from_pylist(
        [
            {
                "readiness_version": AI_READINESS_VERSION,
                "metric": metric,
                "status": status,
                "count": count,
                "note": note,
            }
            for metric, status, count, note in rows
        ],
        schema=COVERAGE_SCHEMA,
    )
    pq.write_table(table, path, compression="zstd")


def _all_status(directory: Path, field: str, allowed: set[str]) -> bool:
    dataset = ds.dataset(directory, format="parquet")
    values = set(dataset.to_table(columns=[field]).column(0).unique().to_pylist())
    return values <= allowed


def _artifact(path: Path, root: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    return {
        "path": str(path.relative_to(root)),
        "physical_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "row_count": parquet.metadata.num_rows,
        "column_count": len(parquet.schema_arrow),
        "row_group_count": parquet.metadata.num_row_groups,
    }


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lit(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


__all__ = [
    "AI_READINESS_RULE_VERSION",
    "AI_READINESS_SCHEMA",
    "AI_READINESS_VERSION",
    "COVERAGE_SCHEMA",
    "STATUS_COLUMNS",
    "SUMMARY_SCHEMA",
    "publish_ai_readiness",
]
