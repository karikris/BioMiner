from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.assertions import assertion_table, build_assertion


GEOGRAPHY_VERSION = "biominer-gbif-geographic-enrichment/v1"
GEOGRAPHY_RULE_VERSION = "pinned-snapshot-country-consensus/v1.0.0"
MIN_MAPPING_CONFIDENCE = 0.99
COUNTRY_REFERENCE_SCHEMA = pa.schema(
    [
        ("countryCode", pa.string()),
        ("mapped_continent", pa.string()),
        ("continent_support_rows", pa.int64()),
        ("continent_observed_rows", pa.int64()),
        ("continent_confidence", pa.float64()),
        ("mapped_gbifRegion", pa.string()),
        ("gbif_region_support_rows", pa.int64()),
        ("gbif_region_observed_rows", pa.int64()),
        ("gbif_region_confidence", pa.float64()),
        ("mapping_source_snapshot", pa.string()),
        ("mapping_rule_version", pa.string()),
    ]
)
GEOGRAPHIC_OUTCOME_SCHEMA = pa.schema(
    [
        ("geography_version", pa.string()),
        ("source_snapshot_id", pa.string()),
        ("source_row_id", pa.string()),
        ("gbifID", pa.string()),
        ("affected_media_rows", pa.int64()),
        ("decimalLatitude", pa.string()),
        ("decimalLongitude", pa.string()),
        ("source_countryCode", pa.string()),
        ("source_continent", pa.string()),
        ("source_gbifRegion", pa.string()),
        ("derived_countryCode", pa.string()),
        ("derived_continent", pa.string()),
        ("derived_gbifRegion", pa.string()),
        ("country_derivation_status", pa.string()),
        ("country_derivation_reason", pa.string()),
        ("continent_derivation_status", pa.string()),
        ("continent_derivation_reason", pa.string()),
        ("gbif_region_derivation_status", pa.string()),
        ("gbif_region_derivation_reason", pa.string()),
        ("geographic_conflict_status", pa.string()),
        ("geographic_conflict_fields", pa.list_(pa.string())),
        ("boundary_dataset", pa.string()),
        ("boundary_version", pa.string()),
        ("boundary_confidence", pa.string()),
        ("border_ambiguity_status", pa.string()),
    ]
)


@dataclass(frozen=True, slots=True)
class GeographyResult:
    output_directory: Path
    outcome_path: Path
    assertion_path: Path
    manifest: dict[str, object]


def publish_geographic_enrichment(
    *,
    v3_parquet: str | Path,
    output_directory: str | Path,
    source_snapshot_id: str,
    expected_coordinate_country_media_rows: int,
    expected_coordinate_country_occurrences: int,
    expected_missing_continent_media_rows: int,
    expected_missing_continent_occurrences: int,
    expected_missing_region_media_rows: int,
    expected_missing_region_occurrences: int,
    code_commit: str,
    memory_limit: str = "4GB",
    threads: int = 4,
    minimum_mapping_confidence: float = MIN_MAPPING_CONFIDENCE,
) -> GeographyResult:
    source = Path(v3_parquet).resolve()
    destination = Path(output_directory).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    if not 0 < minimum_mapping_confidence <= 1:
        raise ValueError("minimum_mapping_confidence must be in (0, 1]")
    required = {
        "gbifID", "decimalLatitude", "decimalLongitude", "countryCode",
        "continent", "gbifRegion",
    }
    missing = required - set(pq.ParquetFile(source).schema_arrow.names)
    if missing:
        raise ValueError(f"v3 lacks geographic fields: {sorted(missing)}")
    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={threads}")
        connection.execute(f"SET memory_limit={_literal(memory_limit)}")
        mappings = _build_mappings(connection, source, source_snapshot_id)
        mapping_table = pa.Table.from_pylist(mappings, schema=COUNTRY_REFERENCE_SCHEMA)
        connection.register("country_reference", mapping_table)
        candidates = _candidate_rows(
            connection, source, minimum_mapping_confidence=minimum_mapping_confidence
        )
    finally:
        connection.close()
    assertions = []
    outcomes = []
    counts = defaultdict(int)
    for row in candidates:
        country_code = _trimmed(row["source_countryCode"])
        mapped_continent = _trimmed(row["mapped_continent"])
        mapped_region = _trimmed(row["mapped_gbifRegion"])
        continent_safe = (
            mapped_continent is not None
            and float(row["continent_confidence"] or 0) >= minimum_mapping_confidence
        )
        region_safe = (
            mapped_region is not None
            and float(row["gbif_region_confidence"] or 0) >= minimum_mapping_confidence
        )
        valid_coordinates = _valid_coordinates(
            row["decimalLatitude"], row["decimalLongitude"]
        )
        missing_country = country_code is None and valid_coordinates
        missing_continent = row["source_continent"] is None and country_code is not None
        missing_region = row["source_gbifRegion"] is None and country_code is not None
        conflicts = []
        for field, count_field in (
            ("countryCode", "country_code_value_count"),
            ("continent", "continent_value_count"),
            ("gbifRegion", "gbif_region_value_count"),
        ):
            if int(row[count_field]) > 1:
                conflicts.append(field)
        if (
            row["source_continent"] is not None
            and continent_safe
            and str(row["source_continent"]) != mapped_continent
        ):
            conflicts.append("continent")
        if (
            row["source_gbifRegion"] is not None
            and region_safe
            and str(row["source_gbifRegion"]) != mapped_region
        ):
            conflicts.append("gbifRegion")
        derived_continent = mapped_continent if missing_continent and continent_safe else None
        derived_region = mapped_region if missing_region and region_safe else None
        source_row_id = _source_row_id(source_snapshot_id, str(row["gbifID"]))
        country_status = "NOT_TESTED" if missing_country else "NOT_APPLICABLE"
        country_reason = (
            "pinned_coordinate_boundary_dataset_unavailable"
            if missing_country
            else "country_code_present_or_coordinates_not_eligible"
        )
        continent_status, continent_reason = _mapping_status(
            missing_continent, derived_continent
        )
        region_status, region_reason = _mapping_status(missing_region, derived_region)
        outcome = {
            "geography_version": GEOGRAPHY_VERSION,
            "source_snapshot_id": source_snapshot_id,
            "source_row_id": source_row_id,
            **{name: row[name] for name in (
                "gbifID", "affected_media_rows", "decimalLatitude", "decimalLongitude",
                "source_countryCode", "source_continent", "source_gbifRegion",
            )},
            "derived_countryCode": None,
            "derived_continent": derived_continent,
            "derived_gbifRegion": derived_region,
            "country_derivation_status": country_status,
            "country_derivation_reason": country_reason,
            "continent_derivation_status": continent_status,
            "continent_derivation_reason": continent_reason,
            "gbif_region_derivation_status": region_status,
            "gbif_region_derivation_reason": region_reason,
            "geographic_conflict_status": "CONFLICT" if conflicts else "PASS",
            "geographic_conflict_fields": conflicts,
            "boundary_dataset": None,
            "boundary_version": None,
            "boundary_confidence": None,
            "border_ambiguity_status": "NOT_TESTED" if missing_country else "NOT_APPLICABLE",
        }
        outcomes.append(outcome)
        counts["affected_occurrences"] += 1
        counts["affected_media_rows"] += int(row["affected_media_rows"])
        counts["coordinate_country_candidate_occurrences"] += int(missing_country)
        counts["coordinate_country_candidate_media_rows"] += int(missing_country) * int(row["affected_media_rows"])
        counts["missing_continent_occurrences"] += int(missing_continent)
        counts["missing_continent_media_rows"] += int(missing_continent) * int(row["affected_media_rows"])
        counts["missing_region_occurrences"] += int(missing_region)
        counts["missing_region_media_rows"] += int(missing_region) * int(row["affected_media_rows"])
        counts["derived_continent_occurrences"] += int(derived_continent is not None)
        counts["derived_continent_media_rows"] += int(derived_continent is not None) * int(row["affected_media_rows"])
        counts["derived_region_occurrences"] += int(derived_region is not None)
        counts["derived_region_media_rows"] += int(derived_region is not None) * int(row["affected_media_rows"])
        counts["conflict_occurrences"] += int(bool(conflicts))
        for target, value, evidence in (
            ("derived_continent", derived_continent, "countryCode"),
            ("derived_gbifRegion", derived_region, "countryCode"),
        ):
            if value is None:
                continue
            assertions.append(build_assertion(
                source_snapshot_version=source_snapshot_id,
                source_row_id=source_row_id,
                gbif_id=str(row["gbifID"]),
                target_field=target,
                original_value=None,
                derived_value=value,
                evidence_source=evidence,
                source_url_or_record_identifier=f"gbifID:{row['gbifID']}",
                retrieval_timestamp=generated_at,
                derivation_method="pinned_snapshot_country_code_consensus",
                derivation_rule_version=GEOGRAPHY_RULE_VERSION,
                confidence_class="DETERMINISTIC_DERIVATION",
                validation_status="PASS",
                conflict_status="PASS",
                reviewer_status="NOT_REQUIRED",
            ))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    reference_path = staging / "country_reference.parquet"
    outcome_path = staging / "geographic_outcomes.parquet"
    assertion_path = staging / "geographic_assertions.parquet"
    try:
        pq.write_table(mapping_table, reference_path, compression="zstd")
        pq.write_table(pa.Table.from_pylist(outcomes, schema=GEOGRAPHIC_OUTCOME_SCHEMA), outcome_path, compression="zstd")
        pq.write_table(assertion_table(assertions), assertion_path, compression="zstd")
        expected = {
            "coordinate_country_candidate_media_rows": expected_coordinate_country_media_rows,
            "coordinate_country_candidate_occurrences": expected_coordinate_country_occurrences,
            "missing_continent_media_rows": expected_missing_continent_media_rows,
            "missing_continent_occurrences": expected_missing_continent_occurrences,
            "missing_region_media_rows": expected_missing_region_media_rows,
            "missing_region_occurrences": expected_missing_region_occurrences,
        }
        validation = {f"{name}_match": counts[name] == value for name, value in expected.items()}
        validation.update({
            "all_coordinate_country_candidates_retained": counts["coordinate_country_candidate_occurrences"] <= len(outcomes),
            "coordinate_country_not_fabricated": counts["coordinate_country_candidate_occurrences"] > 0 and all(row["derived_countryCode"] is None for row in outcomes),
            "assertions_reconcile": len(assertions) == counts["derived_continent_occurrences"] + counts["derived_region_occurrences"],
            "outcome_schema_matches": pq.ParquetFile(outcome_path).schema_arrow.equals(GEOGRAPHIC_OUTCOME_SCHEMA),
        })
        if not all(validation.values()):
            raise ValueError(f"geographic enrichment validation failed: {validation}; {dict(counts)}")
        artifacts = [_artifact(path) for path in (reference_path, outcome_path, assertion_path)]
        manifest = {
            "schema_version": GEOGRAPHY_VERSION,
            "rule_version": GEOGRAPHY_RULE_VERSION,
            "generated_at": generated_at,
            "code_commit": code_commit,
            "source_snapshot_id": source_snapshot_id,
            "input": str(source),
            "minimum_mapping_confidence": minimum_mapping_confidence,
            "counts": dict(sorted(counts.items())),
            "validation": validation,
            "artifacts": artifacts,
            "policy": {
                "source_fields_unchanged": True,
                "coordinate_to_country": "NOT_TESTED_NO_PINNED_BOUNDARY",
                "country_code_mapping": "PINNED_SOURCE_SNAPSHOT_CONSENSUS",
                "unsafe_mappings_retained_unresolved": True,
            },
            "network_requests": 0,
            "manifest_policy": {"written_last": True},
        }
        _write_json(staging / "manifest.json", manifest)
        for artifact in artifacts:
            _verify(staging, artifact)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return GeographyResult(destination, destination / outcome_path.name, destination / assertion_path.name, manifest)


def _build_mappings(connection: duckdb.DuckDBPyConnection, source: Path, snapshot: str) -> list[dict[str, object]]:
    cursor = connection.execute(f"""
        SELECT trim(countryCode) countryCode, continent, gbifRegion, count(*)::BIGINT n
        FROM read_parquet({_literal(str(source))})
        WHERE countryCode IS NOT NULL
        GROUP BY ALL ORDER BY countryCode, n DESC, continent, gbifRegion
    """)
    grouped: dict[str, list[tuple[str | None, str | None, int]]] = defaultdict(list)
    for code, continent, region, count in cursor.fetchall():
        grouped[str(code)].append((continent, region, int(count)))
    result = []
    for code, rows in sorted(grouped.items()):
        continent_counts: dict[str, int] = defaultdict(int)
        region_counts: dict[str, int] = defaultdict(int)
        for continent, region, count in rows:
            if continent is not None: continent_counts[str(continent)] += count
            if region is not None: region_counts[str(region)] += count
        continent, cs, co = _mode(continent_counts)
        region, rs, ro = _mode(region_counts)
        result.append({
            "countryCode": code,
            "mapped_continent": continent,
            "continent_support_rows": cs,
            "continent_observed_rows": co,
            "continent_confidence": cs / co if co else 0.0,
            "mapped_gbifRegion": region,
            "gbif_region_support_rows": rs,
            "gbif_region_observed_rows": ro,
            "gbif_region_confidence": rs / ro if ro else 0.0,
            "mapping_source_snapshot": snapshot,
            "mapping_rule_version": GEOGRAPHY_RULE_VERSION,
        })
    return result


def _mode(counts: dict[str, int]) -> tuple[str | None, int, int]:
    observed = sum(counts.values())
    if not counts: return None, 0, 0
    value, support = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return value, support, observed


def _candidate_rows(
    connection: duckdb.DuckDBPyConnection,
    source: Path,
    *,
    minimum_mapping_confidence: float,
) -> list[dict[str, object]]:
    cursor = connection.execute(f"""
        WITH occurrence AS (
          SELECT trim(cast(gbifID AS VARCHAR)) gbifID, count(*)::BIGINT affected_media_rows,
                 min(cast(decimalLatitude AS VARCHAR)) decimalLatitude,
                 min(cast(decimalLongitude AS VARCHAR)) decimalLongitude,
                 min(countryCode) source_countryCode, min(continent) source_continent,
                 min(gbifRegion) source_gbifRegion,
                 count(distinct coalesce(countryCode, '<NULL>')) country_code_value_count,
                 count(distinct coalesce(continent, '<NULL>')) continent_value_count,
                 count(distinct coalesce(gbifRegion, '<NULL>')) gbif_region_value_count
          FROM read_parquet({_literal(str(source))}) GROUP BY gbifID
        ), enriched AS (
          SELECT o.*, r.mapped_continent, r.continent_confidence,
                 r.mapped_gbifRegion, r.gbif_region_confidence
          FROM occurrence o LEFT JOIN country_reference r ON o.source_countryCode=r.countryCode
        )
        SELECT * FROM enriched
        WHERE (source_countryCode IS NULL AND try_cast(decimalLatitude AS DOUBLE) BETWEEN -90 AND 90
               AND try_cast(decimalLongitude AS DOUBLE) BETWEEN -180 AND 180)
           OR (source_countryCode IS NOT NULL AND source_continent IS NULL)
           OR (source_countryCode IS NOT NULL AND source_gbifRegion IS NULL)
           OR country_code_value_count > 1 OR continent_value_count > 1 OR gbif_region_value_count > 1
           OR (source_continent IS NOT NULL AND continent_confidence >= {minimum_mapping_confidence}
               AND source_continent <> mapped_continent)
           OR (source_gbifRegion IS NOT NULL AND gbif_region_confidence >= {minimum_mapping_confidence}
               AND source_gbifRegion <> mapped_gbifRegion)
        ORDER BY gbifID
    """)
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _mapping_status(missing: bool, derived: str | None) -> tuple[str, str]:
    if not missing: return "NOT_APPLICABLE", "source_value_present"
    if derived is not None: return "PASS", "safe_pinned_snapshot_country_code_consensus"
    return "UNKNOWN", "country_code_mapping_below_confidence_threshold"


def _valid_coordinates(latitude: object | None, longitude: object | None) -> bool:
    try:
        lat, lon = float(str(latitude)), float(str(longitude))
    except (TypeError, ValueError):
        return False
    return -90 <= lat <= 90 and -180 <= lon <= 180


def _source_row_id(snapshot: str, gbif_id: str) -> str:
    return "sha256:" + hashlib.sha256(f"{snapshot}|occurrence.txt|gbifID={gbif_id}".encode()).hexdigest()


def _trimmed(value: object | None) -> str | None:
    if value is None: return None
    text = str(value).strip()
    return text or None


def _literal(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _artifact(path: Path) -> dict[str, object]:
    p = pq.ParquetFile(path)
    return {"path": path.name, "physical_bytes": path.stat().st_size, "sha256": _sha256(path), "row_count": p.metadata.num_rows, "column_count": len(p.schema_arrow), "row_group_count": p.metadata.num_row_groups}


def _verify(root: Path, artifact: dict[str, object]) -> None:
    if _sha256(root / str(artifact["path"])) != artifact["sha256"]: raise ValueError(f"geography checksum mismatch: {artifact['path']}")


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


__all__ = ["COUNTRY_REFERENCE_SCHEMA", "GEOGRAPHIC_OUTCOME_SCHEMA", "GEOGRAPHY_VERSION", "GeographyResult", "publish_geographic_enrichment"]
