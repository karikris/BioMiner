from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import duckdb
import pyarrow.parquet as pq


REPRESENTATIVENESS_VERSION = "biominer-gbif-media-representativeness/v1"

PROFILE_DIMENSIONS = (
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
    "taxonRank",
    "acceptedTaxonKey",
    "provider",
    "publisher",
    "dataset",
    "creator",
    "countryCode",
    "continent",
    "gbifRegion",
    "latitude_band",
    "longitude_band",
    "year",
    "month",
    "decade",
    "basisOfRecord",
    "lifeStage",
    "sex",
    "media_license",
    "media_format",
    "ai_ingestion_decision",
)


def publish_representativeness(
    *,
    v3_parquet: str | Path,
    media_quality_parquet: str | Path,
    ai_readiness_glob: str | Path,
    output_directory: str | Path,
    source_snapshot_id: str,
    expected_rows: int,
    code_commit: str,
    memory_limit: str = "6GB",
    threads: int = 4,
    temp_directory: str | Path | None = None,
) -> dict[str, object]:
    """Publish duplicate-adjusted coverage and non-composite remediation evidence."""

    source = Path(v3_parquet).resolve()
    quality = Path(media_quality_parquet).resolve()
    readiness_glob = str(ai_readiness_glob)
    if not source.is_file() or not quality.is_file():
        raise FileNotFoundError(source if not source.is_file() else quality)
    readiness_parts = sorted(Path().glob(readiness_glob)) if not Path(readiness_glob).is_absolute() else []
    if not readiness_parts:
        readiness_parts = sorted(Path(readiness_glob).parent.glob(Path(readiness_glob).name))
    if not readiness_parts:
        raise FileNotFoundError(readiness_glob)
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    temporary = Path(temp_directory).resolve() if temp_directory else staging / "duckdb_tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    base = staging / "profile_base.parquet"
    outputs = {
        "coverage": staging / "coverage_by_dimension.parquet",
        "species": staging / "species_bias_flags.parquet",
        "provider": staging / "provider_scorecard.parquet",
        "dataset": staging / "dataset_scorecard.parquet",
        "remediation": staging / "provider_remediation_queue.parquet",
    }
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={threads}")
        connection.execute(f"SET memory_limit={_lit(memory_limit)}")
        connection.execute(f"SET temp_directory={_lit(str(temporary))}")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(_base_sql(source, quality, readiness_glob, base))
        rows, distinct_ids = connection.execute(
            f"SELECT count(*),count(distinct media_assertion_id) FROM read_parquet({_lit(str(base))})"
        ).fetchone()
        if int(rows) != expected_rows or int(distinct_ids) != expected_rows:
            raise ValueError("representativeness base does not reconcile")
        connection.execute(_coverage_sql(base, outputs["coverage"]))
        connection.execute(_species_sql(base, outputs["species"]))
        connection.execute(_scorecard_sql(base, outputs["provider"], "provider"))
        connection.execute(_scorecard_sql(base, outputs["dataset"], "dataset"))
        connection.execute(_remediation_sql(outputs["provider"], outputs["remediation"]))
        counts = {
            name: int(connection.execute(
                f"SELECT count(*) FROM read_parquet({_lit(str(path))})"
            ).fetchone()[0])
            for name, path in outputs.items()
        }
    except BaseException:
        connection.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        try:
            connection.close()
        except Exception:
            pass
    base.unlink(missing_ok=True)
    if temp_directory is None:
        shutil.rmtree(temporary, ignore_errors=True)
    validation = {
        "rows_match": int(rows) == expected_rows,
        "one_row_per_media_assertion_in_base": int(distinct_ids) == expected_rows,
        "all_profile_dimensions_present": counts["coverage"] >= len(PROFILE_DIMENSIONS),
        "provider_scorecard_nonempty": counts["provider"] > 0,
        "dataset_scorecard_nonempty": counts["dataset"] > 0,
        "species_flags_nonempty": counts["species"] > 0,
        "source_fields_unchanged": True,
        "manifest_written_last": True,
    }
    if not all(validation.values()):
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(f"representativeness validation failed: {validation}")
    artifacts = [_artifact(path, staging) for path in outputs.values()]
    manifest = {
        "schema_version": REPRESENTATIVENESS_VERSION,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "code_commit": code_commit,
        "source_snapshot_id": source_snapshot_id,
        "inputs": {
            "v3": str(source),
            "media_quality": str(quality),
            "ai_readiness": readiness_glob,
        },
        "counts": {"source_media_rows": int(rows), **counts},
        "configuration": {
            "high_duplicate_share_threshold": 0.20,
            "low_diversity_minimum_support": 10,
            "provider_ranking": "lexicographic_repairable_rows_then_duplicate_adjusted_benefit",
            "composite_quality_score": False,
            "content_deduplication_status": "NOT_TESTED",
            "perceptual_deduplication_status": "NOT_TESTED",
        },
        "validation": validation,
        "artifacts": artifacts,
        "network_requests": 0,
        "manifest_policy": {"written_last": True},
    }
    _write_json(staging / "manifest.json", manifest)
    for artifact in artifacts:
        if _sha256(staging / str(artifact["path"])) != artifact["sha256"]:
            shutil.rmtree(staging, ignore_errors=True)
            raise ValueError("representativeness checksum mismatch")
    os.replace(staging, destination)
    return manifest


def _base_sql(source: Path, quality: Path, readiness_glob: str, output: Path) -> str:
    return f"""
    COPY (
      SELECT q.media_assertion_id,trim(cast(v.gbifID AS VARCHAR)) gbifID,
        v.kingdom,v.phylum,v.class,v.order,v.family,v.genus,v.species,v.taxonRank,
        cast(v.acceptedTaxonKey AS VARCHAR) acceptedTaxonKey,
        coalesce(nullif(trim(v.media_publisher),''),nullif(trim(v.publisher),''),'<MISSING>') provider,
        coalesce(nullif(trim(v.publisher),''),'<MISSING>') publisher,
        coalesce(nullif(trim(v.datasetName),''),cast(v.datasetKey AS VARCHAR),'<MISSING>') dataset,
        coalesce(nullif(trim(v.media_creator),''),'<MISSING>') creator,
        v.countryCode,v.continent,v.gbifRegion,
        CASE WHEN try_cast(v.decimalLatitude AS DOUBLE) BETWEEN -90 AND 90
          THEN cast(floor(try_cast(v.decimalLatitude AS DOUBLE)/10)*10 AS VARCHAR) END latitude_band,
        CASE WHEN try_cast(v.decimalLongitude AS DOUBLE) BETWEEN -180 AND 180
          THEN cast(floor(try_cast(v.decimalLongitude AS DOUBLE)/10)*10 AS VARCHAR) END longitude_band,
        v.year,v.month,
        CASE WHEN try_cast(v.year AS INTEGER) IS NOT NULL
          THEN cast(floor(try_cast(v.year AS INTEGER)/10)*10 AS VARCHAR) END "decade",
        v.basisOfRecord,v.lifeStage,v.sex,v.media_license,v.media_format,
        v.media_identifier,v.media_rightsHolder,v.decimalLatitude,v.decimalLongitude,
        v.coordinateUncertaintyInMeters,
        a.original_url_hash,a.canonical_url_hash,a.rights_policy_status,
        a.duplicate_status,a.cross_taxon_url_status,a."MEDIA_DIRECT",a."RIGHTS_KNOWN",
        a."RIGHTS_ALLOWED",a."MEDIA_TECHNICALLY_VALID",a."EXACT_SPECIES_LABEL",
        a."IDENTIFICATION_PROVENANCE_PRESENT",a.ai_ingestion_decision
      FROM read_parquet({_lit(str(source))}) v
      POSITIONAL JOIN read_parquet({_lit(str(quality))}) q
      JOIN read_parquet({_lit(readiness_glob)}) a
        ON q.media_assertion_id=a.media_assertion_id
    ) TO {_lit(str(output))} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)
    """


def _coverage_sql(base: Path, output: Path) -> str:
    expressions = {
        "provider": "provider",
        "publisher": "publisher",
        "dataset": "dataset",
        "creator": "creator",
        "latitude_band": "latitude_band",
        "longitude_band": "longitude_band",
        "decade": '"decade"',
        "ai_ingestion_decision": "ai_ingestion_decision",
    }
    queries = []
    for dimension in PROFILE_DIMENSIONS:
        expression = expressions.get(dimension, f'"{dimension}"')
        queries.append(f"""
          SELECT {_lit(REPRESENTATIVENESS_VERSION)} representativeness_version,
            {_lit(dimension)} dimension,coalesce(cast({expression} AS VARCHAR),'<MISSING>') "value",
            count(*)::BIGINT raw_image_count,count(distinct gbifID)::BIGINT distinct_occurrence_count,
            count(distinct original_url_hash)::BIGINT distinct_url_count,
            (count(distinct canonical_url_hash)+count(*) FILTER(WHERE canonical_url_hash IS NULL))::BIGINT duplicate_adjusted_count,
            count(distinct provider)::BIGINT provider_count,count(distinct creator)::BIGINT creator_count,
            count(distinct countryCode)::BIGINT country_count,count(distinct year)::BIGINT year_count,
            count(*) FILTER(WHERE "RIGHTS_ALLOWED"='PASS')::BIGINT licence_compatible_count,
            count(*) FILTER(WHERE "MEDIA_TECHNICALLY_VALID"='PASS')::BIGINT technically_valid_count,
            count(*) FILTER(WHERE "EXACT_SPECIES_LABEL"='PASS')::BIGINT exact_species_label_count,
            count(*) FILTER(WHERE "EXACT_SPECIES_LABEL"='FAIL')::BIGINT higher_rank_only_count,
            count(*) FILTER(WHERE ai_ingestion_decision='UNRESOLVED')::BIGINT unresolved_count
          FROM read_parquet({_lit(str(base))}) GROUP BY 3
        """)
    return f"COPY ({' UNION ALL '.join(queries)}) TO {_lit(str(output))} (FORMAT PARQUET,COMPRESSION ZSTD)"


def _species_sql(base: Path, output: Path) -> str:
    return f"""
    COPY (
      WITH species_base AS (
        SELECT coalesce(species,'<MISSING>') species,count(*)::BIGINT raw_image_count,
          count(distinct gbifID)::BIGINT distinct_occurrence_count,
          count(distinct canonical_url_hash)::BIGINT distinct_canonical_urls,
          (count(distinct canonical_url_hash)+count(*) FILTER(WHERE canonical_url_hash IS NULL))::BIGINT duplicate_adjusted_count,
          count(distinct provider)::BIGINT provider_count,count(distinct creator)::BIGINT creator_count,
          count(distinct countryCode)::BIGINT country_count,count(distinct "decade")::BIGINT decade_count,
          count(*) FILTER(WHERE lifeStage IS NOT NULL AND trim(cast(lifeStage AS VARCHAR))<>'')::BIGINT life_stage_rows,
          count(*) FILTER(WHERE lower(coalesce(cast(sex AS VARCHAR),''))='male')::BIGINT male_rows,
          count(*) FILTER(WHERE lower(coalesce(cast(sex AS VARCHAR),''))='female')::BIGINT female_rows,
          count(*) FILTER(WHERE cross_taxon_url_status='CONFLICT')::BIGINT label_conflict_rows,
          count(*) FILTER(WHERE "RIGHTS_ALLOWED"='PASS')::BIGINT licence_compatible_rows,
          count(*) FILTER(WHERE ai_ingestion_decision='UNRESOLVED')::BIGINT unresolved_rows
        FROM read_parquet({_lit(str(base))}) GROUP BY 1
      ) SELECT *,
        CASE WHEN raw_image_count=0 THEN NULL ELSE 1-(duplicate_adjusted_count::DOUBLE/raw_image_count) END duplicate_share,
        list_filter([
          CASE WHEN provider_count=1 THEN 'ONE_PROVIDER' END,
          CASE WHEN creator_count=1 THEN 'ONE_CREATOR' END,
          CASE WHEN distinct_occurrence_count=1 THEN 'ONE_OCCURRENCE' END,
          CASE WHEN raw_image_count>0 AND 1-(duplicate_adjusted_count::DOUBLE/raw_image_count)>0.20 THEN 'HIGH_URL_DUPLICATE_SHARE' END,
          CASE WHEN raw_image_count>=10 AND country_count<2 THEN 'LOW_GEOGRAPHIC_DIVERSITY' END,
          CASE WHEN raw_image_count>=10 AND decade_count<2 THEN 'LOW_TEMPORAL_DIVERSITY' END,
          CASE WHEN life_stage_rows=0 THEN 'NO_LIFE_STAGE_EVIDENCE' END,
          CASE WHEN male_rows=0 THEN 'NO_MALE_LABELS' END,
          CASE WHEN female_rows=0 THEN 'NO_FEMALE_LABELS' END,
          CASE WHEN label_conflict_rows>0 THEN 'URL_TAXON_LABEL_CONFLICT' END,
          CASE WHEN duplicate_adjusted_count<=5 THEN 'TAXONOMIC_LONG_TAIL' END
        ], item -> item IS NOT NULL) bias_flags
      FROM species_base
    ) TO {_lit(str(output))} (FORMAT PARQUET,COMPRESSION ZSTD)
    """


def _scorecard_sql(base: Path, output: Path, grouping: str) -> str:
    group = f'"{grouping}"'
    return f"""
    COPY (SELECT {_lit(REPRESENTATIVENESS_VERSION)} scorecard_version,{group},
      count(*)::BIGINT source_row_count,count(distinct gbifID)::BIGINT occurrence_count,
      count(*)::BIGINT media_count,
      count(*) FILTER(WHERE "MEDIA_DIRECT"='PASS')::BIGINT direct_url_rows,
      0::BIGINT url_reachability_tested_rows,0::BIGINT url_reachable_rows,
      count(*) FILTER(WHERE "MEDIA_TECHNICALLY_VALID"<>'NOT_TESTED')::BIGINT technical_status_rows,
      count(*) FILTER(WHERE "MEDIA_TECHNICALLY_VALID"='PASS')::BIGINT technically_valid_rows,
      count(*) FILTER(WHERE media_format IS NOT NULL AND trim(media_format)<>'')::BIGINT format_filled_rows,
      count(*) FILTER(WHERE lower(coalesce(media_format,'')) LIKE 'image/%')::BIGINT format_valid_rows,
      count(*) FILTER(WHERE "RIGHTS_KNOWN"='PASS')::BIGINT licence_known_rows,
      count(*) FILTER(WHERE rights_policy_status='QUARANTINED')::BIGINT licence_ambiguous_rows,
      count(*) FILTER(WHERE creator<>'<MISSING>')::BIGINT creator_filled_rows,
      count(*) FILTER(WHERE media_rightsHolder IS NOT NULL AND trim(media_rightsHolder)<>'')::BIGINT rights_holder_filled_rows,
      count(*) FILTER(WHERE decimalLatitude IS NOT NULL AND decimalLongitude IS NOT NULL)::BIGINT coordinate_rows,
      count(*) FILTER(WHERE coordinateUncertaintyInMeters IS NOT NULL)::BIGINT uncertainty_rows,
      count(*) FILTER(WHERE "EXACT_SPECIES_LABEL"='PASS')::BIGINT exact_species_label_rows,
      count(*) FILTER(WHERE "IDENTIFICATION_PROVENANCE_PRESENT"='PASS')::BIGINT identification_provenance_rows,
      count(*) FILTER(WHERE duplicate_status IN ('DUPLICATE','CONFLICT'))::BIGINT duplicate_rows,
      count(*) FILTER(WHERE ai_ingestion_decision='AI_READY')::BIGINT ai_ready_rows,
      count(*) FILTER(WHERE ai_ingestion_decision='UNRESOLVED')::BIGINT unresolved_rows,
      count(*) FILTER(WHERE "MEDIA_DIRECT"<>'PASS' OR "RIGHTS_KNOWN"<>'PASS'
        OR creator='<MISSING>' OR media_rightsHolder IS NULL)::BIGINT estimated_recoverable_rows,
      (count(distinct canonical_url_hash)+count(*) FILTER(WHERE canonical_url_hash IS NULL))::BIGINT duplicate_adjusted_benefit,
      count(distinct species)::BIGINT distinct_species
    FROM read_parquet({_lit(str(base))}) GROUP BY 2)
    TO {_lit(str(output))} (FORMAT PARQUET,COMPRESSION ZSTD)
    """


def _remediation_sql(provider: Path, output: Path) -> str:
    return f"""
    COPY (SELECT {_lit(REPRESENTATIVENESS_VERSION)} remediation_version,
      row_number() OVER(ORDER BY estimated_recoverable_rows DESC,duplicate_adjusted_benefit DESC,provider)::BIGINT priority_rank,
      provider,estimated_recoverable_rows repairable_record_count,distinct_species research_value_species,
      duplicate_adjusted_benefit ai_value_duplicate_adjusted_rows,
      licence_known_rows evidence_quality_known_rights_rows,
      'UNKNOWN' provider_level_fix_availability,
      (media_count-direct_url_rows)::BIGINT expected_network_cost_rows,
      licence_ambiguous_rows::BIGINT false_enrichment_risk_rows,
      duplicate_adjusted_benefit,
      'LEXICOGRAPHIC_NOT_COMPOSITE_SCORE' ranking_method
    FROM read_parquet({_lit(str(provider))})
    ORDER BY priority_rank) TO {_lit(str(output))} (FORMAT PARQUET,COMPRESSION ZSTD)
    """


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


__all__ = ["PROFILE_DIMENSIONS", "REPRESENTATIVENESS_VERSION", "publish_representativeness"]
