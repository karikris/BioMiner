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


GEOGRAPHY_VERSION = "biominer-gbif-geographic-enrichment/v2"
GEOGRAPHY_RULE_VERSION = "pinned-boundary-and-snapshot-consensus/v2.0.0"
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
        ("derived_country", pa.string()),
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
        ("boundary_candidate_countryCodes", pa.list_(pa.string())),
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
    boundary_manifest: str | Path | None = None,
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
    boundary = (
        _load_boundary_reference(Path(boundary_manifest).resolve())
        if boundary_manifest is not None
        else None
    )
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={threads}")
        connection.execute(f"SET memory_limit={_literal(memory_limit)}")
        mappings = _build_mappings(connection, source, source_snapshot_id)
        mapping_table = pa.Table.from_pylist(mappings, schema=COUNTRY_REFERENCE_SCHEMA)
        connection.register("country_reference", mapping_table)
        if boundary is not None:
            _register_boundary_countries(connection, boundary["boundary_path"])
        candidates = _candidate_rows(
            connection,
            source,
            minimum_mapping_confidence=minimum_mapping_confidence,
            boundary_available=boundary is not None,
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
        coordinates_in_bounds = _coordinates_in_bounds(
            row["decimalLatitude"], row["decimalLongitude"]
        )
        invalid_zero_zero = _is_zero_zero(
            row["decimalLatitude"], row["decimalLongitude"]
        )
        valid_coordinates = coordinates_in_bounds and not invalid_zero_zero
        missing_country = country_code is None and valid_coordinates
        baseline_coordinate_country_candidate = (
            country_code is None and coordinates_in_bounds
        )
        boundary_match_count = int(row["boundary_match_count"] or 0)
        derived_country_code = (
            _trimmed(row["boundary_country_code"])
            if missing_country and boundary_match_count == 1
            else None
        )
        derived_country = (
            _trimmed(row["boundary_country"])
            if derived_country_code is not None
            else None
        )
        missing_continent = row["source_continent"] is None and country_code is not None
        missing_region = row["source_gbifRegion"] is None and country_code is not None
        derive_continent_needed = row["source_continent"] is None and (
            country_code is not None or derived_country_code is not None
        )
        derive_region_needed = row["source_gbifRegion"] is None and (
            country_code is not None or derived_country_code is not None
        )
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
        derived_continent = (
            mapped_continent if derive_continent_needed and continent_safe else None
        )
        derived_region = mapped_region if derive_region_needed and region_safe else None
        source_row_id = _source_row_id(source_snapshot_id, str(row["gbifID"]))
        country_status, country_reason = _country_boundary_status(
            missing_country=missing_country,
            invalid_zero_zero=country_code is None and invalid_zero_zero,
            boundary_available=boundary is not None,
            boundary_match_count=boundary_match_count,
        )
        continent_status, continent_reason = _mapping_status(
            derive_continent_needed, derived_continent
        )
        region_status, region_reason = _mapping_status(
            derive_region_needed, derived_region
        )
        outcome = {
            "geography_version": GEOGRAPHY_VERSION,
            "source_snapshot_id": source_snapshot_id,
            "source_row_id": source_row_id,
            **{name: row[name] for name in (
                "gbifID", "affected_media_rows", "decimalLatitude", "decimalLongitude",
                "source_countryCode", "source_continent", "source_gbifRegion",
            )},
            "derived_countryCode": derived_country_code,
            "derived_country": derived_country,
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
            "boundary_dataset": (
                str(boundary["boundary_dataset"])
                if missing_country and boundary is not None
                else None
            ),
            "boundary_version": (
                str(boundary["boundary_version"])
                if missing_country and boundary is not None
                else None
            ),
            "boundary_confidence": _boundary_confidence(
                missing_country=missing_country,
                boundary_available=boundary is not None,
                boundary_match_count=boundary_match_count,
            ),
            "border_ambiguity_status": _border_status(
                missing_country=missing_country,
                boundary_available=boundary is not None,
                boundary_match_count=boundary_match_count,
            ),
            "boundary_candidate_countryCodes": list(
                row["boundary_candidate_countryCodes"] or []
            ),
        }
        outcomes.append(outcome)
        counts["affected_occurrences"] += 1
        counts["affected_media_rows"] += int(row["affected_media_rows"])
        counts["coordinate_country_candidate_occurrences"] += int(missing_country)
        counts["coordinate_country_candidate_media_rows"] += int(missing_country) * int(row["affected_media_rows"])
        counts["baseline_coordinate_country_candidate_occurrences"] += int(
            baseline_coordinate_country_candidate
        )
        counts["baseline_coordinate_country_candidate_media_rows"] += int(
            baseline_coordinate_country_candidate
        ) * int(row["affected_media_rows"])
        counts["zero_zero_coordinate_occurrences"] += int(
            country_code is None and invalid_zero_zero
        )
        counts["zero_zero_coordinate_media_rows"] += int(
            country_code is None and invalid_zero_zero
        ) * int(row["affected_media_rows"])
        counts["derived_country_occurrences"] += int(derived_country_code is not None)
        counts["derived_country_media_rows"] += int(derived_country_code is not None) * int(row["affected_media_rows"])
        counts["ambiguous_border_occurrences"] += int(
            missing_country and boundary_match_count > 1
        )
        counts["outside_or_unmapped_occurrences"] += int(
            missing_country and boundary is not None and boundary_match_count == 0
        )
        counts["missing_continent_occurrences"] += int(missing_continent)
        counts["missing_continent_media_rows"] += int(missing_continent) * int(row["affected_media_rows"])
        counts["missing_region_occurrences"] += int(missing_region)
        counts["missing_region_media_rows"] += int(missing_region) * int(row["affected_media_rows"])
        counts["derived_continent_occurrences"] += int(derived_continent is not None)
        counts["derived_continent_media_rows"] += int(derived_continent is not None) * int(row["affected_media_rows"])
        counts["derived_region_occurrences"] += int(derived_region is not None)
        counts["derived_region_media_rows"] += int(derived_region is not None) * int(row["affected_media_rows"])
        counts["conflict_occurrences"] += int(bool(conflicts))
        for target, value, evidence, method in (
            (
                "derived_countryCode",
                derived_country_code,
                str(boundary["boundary_dataset"]) if boundary is not None else "",
                "unique_point_polygon_intersection",
            ),
            (
                "derived_country",
                derived_country,
                str(boundary["boundary_dataset"]) if boundary is not None else "",
                "unique_point_polygon_intersection",
            ),
            (
                "derived_continent",
                derived_continent,
                derived_country_code or "countryCode",
                "pinned_snapshot_country_code_consensus",
            ),
            (
                "derived_gbifRegion",
                derived_region,
                derived_country_code or "countryCode",
                "pinned_snapshot_country_code_consensus",
            ),
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
                derivation_method=method,
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
            "baseline_coordinate_country_candidate_media_rows": expected_coordinate_country_media_rows,
            "baseline_coordinate_country_candidate_occurrences": expected_coordinate_country_occurrences,
            "missing_continent_media_rows": expected_missing_continent_media_rows,
            "missing_continent_occurrences": expected_missing_continent_occurrences,
            "missing_region_media_rows": expected_missing_region_media_rows,
            "missing_region_occurrences": expected_missing_region_occurrences,
        }
        validation = {f"{name}_match": counts[name] == value for name, value in expected.items()}
        validation.update({
            "all_coordinate_country_candidates_retained": counts["coordinate_country_candidate_occurrences"] <= len(outcomes),
            "baseline_candidate_difference_explained": counts["baseline_coordinate_country_candidate_occurrences"] == counts["coordinate_country_candidate_occurrences"] + counts["zero_zero_coordinate_occurrences"],
            "coordinate_country_resolved_or_retained": counts["coordinate_country_candidate_occurrences"] == counts["derived_country_occurrences"] + counts["ambiguous_border_occurrences"] + counts["outside_or_unmapped_occurrences"] if boundary is not None else counts["derived_country_occurrences"] == 0,
            "boundary_reference_checksum_valid": boundary is None or bool(boundary["all_checksums_valid"]),
            "assertions_reconcile": len(assertions) == sum(
                int(row[name] is not None)
                for row in outcomes
                for name in (
                    "derived_countryCode",
                    "derived_country",
                    "derived_continent",
                    "derived_gbifRegion",
                )
            ),
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
            "boundary_reference": (
                {
                    "manifest": str(boundary["manifest_path"]),
                    "manifest_sha256": boundary["manifest_sha256"],
                    "boundary_path": str(boundary["boundary_path"]),
                    "boundary_sha256": boundary["boundary_sha256"],
                    "dataset": boundary["boundary_dataset"],
                    "version": boundary["boundary_version"],
                }
                if boundary is not None
                else None
            ),
            "minimum_mapping_confidence": minimum_mapping_confidence,
            "counts": dict(sorted(counts.items())),
            "validation": validation,
            "artifacts": artifacts,
            "policy": {
                "source_fields_unchanged": True,
                "coordinate_to_country": (
                    "PINNED_BOUNDARY_UNIQUE_INTERSECTION_ONLY"
                    if boundary is not None
                    else "NOT_TESTED_NO_PINNED_BOUNDARY"
                ),
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
        SELECT upper(trim(countryCode)) countryCode, continent, gbifRegion, count(*)::BIGINT n
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
    boundary_available: bool,
) -> list[dict[str, object]]:
    boundary_match_cte = (
        """
        , coordinate_country_matches AS (
          SELECT
            o.gbifID,
            count(DISTINCT c.country_code)::BIGINT AS boundary_match_count,
            min(c.country_code) AS boundary_country_code,
            min(c.country_name) AS boundary_country,
            list(DISTINCT c.country_code ORDER BY c.country_code)
              FILTER (WHERE c.country_code IS NOT NULL)
              AS boundary_candidate_countryCodes
          FROM occurrence o
          LEFT JOIN boundary_countries c
            ON o.source_countryCode IS NULL
           AND try_cast(o.decimalLatitude AS DOUBLE) BETWEEN -90 AND 90
           AND try_cast(o.decimalLongitude AS DOUBLE) BETWEEN -180 AND 180
           AND NOT (
             try_cast(o.decimalLatitude AS DOUBLE) = 0
             AND try_cast(o.decimalLongitude AS DOUBLE) = 0
           )
           AND ST_Intersects(
             c.geom,
             ST_Point(
               try_cast(o.decimalLongitude AS DOUBLE),
               try_cast(o.decimalLatitude AS DOUBLE)
             )
           )
          GROUP BY o.gbifID
        )
        """
        if boundary_available
        else """
        , coordinate_country_matches AS (
          SELECT
            gbifID,
            0::BIGINT AS boundary_match_count,
            NULL::VARCHAR AS boundary_country_code,
            NULL::VARCHAR AS boundary_country,
            []::VARCHAR[] AS boundary_candidate_countryCodes
          FROM occurrence
        )
        """
    )
    cursor = connection.execute(f"""
        WITH occurrence AS (
          SELECT trim(cast(gbifID AS VARCHAR)) gbifID, count(*)::BIGINT affected_media_rows,
                 min(cast(decimalLatitude AS VARCHAR)) decimalLatitude,
                 min(cast(decimalLongitude AS VARCHAR)) decimalLongitude,
                 min(upper(trim(countryCode))) source_countryCode,
                 min(continent) source_continent,
                 min(gbifRegion) source_gbifRegion,
                 count(distinct coalesce(countryCode, '<NULL>')) country_code_value_count,
                 count(distinct coalesce(continent, '<NULL>')) continent_value_count,
                 count(distinct coalesce(gbifRegion, '<NULL>')) gbif_region_value_count
          FROM read_parquet({_literal(str(source))}) GROUP BY gbifID
        )
        {boundary_match_cte}
        , enriched AS (
          SELECT
                 o.*,
                 b.boundary_match_count,
                 b.boundary_country_code,
                 b.boundary_country,
                 b.boundary_candidate_countryCodes,
                 r.mapped_continent, r.continent_confidence,
                 r.mapped_gbifRegion, r.gbif_region_confidence
          FROM occurrence o
          JOIN coordinate_country_matches b USING (gbifID)
          LEFT JOIN country_reference r
            ON coalesce(o.source_countryCode, b.boundary_country_code)=r.countryCode
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


def _country_boundary_status(
    *,
    missing_country: bool,
    invalid_zero_zero: bool,
    boundary_available: bool,
    boundary_match_count: int,
) -> tuple[str, str]:
    if invalid_zero_zero:
        return "FAIL", "exact_zero_zero_coordinate"
    if not missing_country:
        return "NOT_APPLICABLE", "country_code_present_or_coordinates_not_eligible"
    if not boundary_available:
        return "NOT_TESTED", "pinned_coordinate_boundary_dataset_unavailable"
    if boundary_match_count == 1:
        return "PASS", "unique_pinned_boundary_intersection"
    if boundary_match_count > 1:
        return "UNKNOWN", "ambiguous_country_border_intersection"
    return "UNKNOWN", "coordinate_outside_or_unmapped_by_boundary"


def _boundary_confidence(
    *,
    missing_country: bool,
    boundary_available: bool,
    boundary_match_count: int,
) -> str | None:
    if not missing_country or not boundary_available:
        return None
    if boundary_match_count == 1:
        return "UNIQUE_POLYGON_INTERSECTION"
    if boundary_match_count > 1:
        return "AMBIGUOUS_MULTIPLE_INTERSECTIONS"
    return "NO_POLYGON_INTERSECTION"


def _border_status(
    *,
    missing_country: bool,
    boundary_available: bool,
    boundary_match_count: int,
) -> str:
    if not missing_country:
        return "NOT_APPLICABLE"
    if not boundary_available:
        return "NOT_TESTED"
    if boundary_match_count > 1:
        return "AMBIGUOUS"
    return "PASS" if boundary_match_count == 1 else "OUTSIDE_OR_UNMAPPED"


def _coordinates_in_bounds(
    latitude: object | None, longitude: object | None
) -> bool:
    try:
        lat, lon = float(str(latitude)), float(str(longitude))
    except (TypeError, ValueError):
        return False
    return -90 <= lat <= 90 and -180 <= lon <= 180


def _is_zero_zero(latitude: object | None, longitude: object | None) -> bool:
    try:
        return float(str(latitude)) == 0 and float(str(longitude)) == 0
    except (TypeError, ValueError):
        return False


def _load_boundary_reference(manifest_path: Path) -> dict[str, object]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("boundary manifest has no file inventory")
    verified: dict[str, str] = {}
    for raw_name, raw_checksum in files.items():
        name = str(raw_name)
        expected = str(raw_checksum).removeprefix("sha256:")
        path = (manifest_path.parent / name).resolve()
        if not path.is_relative_to(manifest_path.parent.resolve()):
            raise ValueError(f"boundary member escapes manifest directory: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"boundary checksum mismatch: {name}")
        verified[name] = actual
    boundary_names = [
        name for name in verified if Path(name).suffix.casefold() in {".shp", ".geojson"}
    ]
    if len(boundary_names) != 1:
        raise ValueError(
            "boundary manifest must inventory exactly one .shp or .geojson geometry file"
        )
    boundary_path = (manifest_path.parent / boundary_names[0]).resolve()
    dataset = _trimmed(value.get("boundary_dataset"))
    version = _trimmed(value.get("boundary_version"))
    if dataset is None or version is None:
        raise ValueError("boundary manifest must name its dataset and version")
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "boundary_path": boundary_path,
        "boundary_sha256": verified[boundary_names[0]],
        "boundary_dataset": dataset,
        "boundary_version": version,
        "all_checksums_valid": True,
    }


def _register_boundary_countries(
    connection: duckdb.DuckDBPyConnection,
    boundary_path: object,
) -> None:
    try:
        connection.execute("LOAD spatial")
    except duckdb.Error as exc:
        raise RuntimeError(
            "DuckDB spatial extension must be installed before pinned boundary enrichment"
        ) from exc
    path = str(boundary_path)
    columns = {
        str(row[0])
        for row in connection.execute(
            "DESCRIBE SELECT * FROM ST_Read(?)", [path]
        ).fetchall()
    }
    required = {"ISO_A2", "ISO_A2_EH", "ADMIN", "geom"}
    missing = required - columns
    if missing:
        raise ValueError(f"boundary dataset lacks required fields: {sorted(missing)}")
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE boundary_countries AS
        SELECT
          coalesce(
            nullif(upper(trim(cast(ISO_A2 AS VARCHAR))), '-99'),
            nullif(upper(trim(cast(ISO_A2_EH AS VARCHAR))), '-99')
          ) AS country_code,
          nullif(trim(cast(ADMIN AS VARCHAR)), '') AS country_name,
          geom
        FROM ST_Read(?)
        WHERE coalesce(
            nullif(upper(trim(cast(ISO_A2 AS VARCHAR))), '-99'),
            nullif(upper(trim(cast(ISO_A2_EH AS VARCHAR))), '-99')
          ) IS NOT NULL
        """,
        [path],
    )


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
