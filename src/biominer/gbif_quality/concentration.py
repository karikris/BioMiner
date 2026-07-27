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


CONCENTRATION_VERSION = "biominer-gbif-representativeness-concentration/v1"
DIMENSIONS = {
    "provider": "provider",
    "creator": "creator",
    "region": "region",
    "decade": "decade",
}


def publish_concentration_metrics(
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
    """Publish explicit provider, creator, regional, and temporal concentration."""

    source = Path(v3_parquet).resolve()
    quality = Path(media_quality_parquet).resolve()
    ai_glob = str(ai_readiness_glob)
    destination = Path(output_directory).resolve()
    for path in (source, quality):
        if not path.is_file():
            raise FileNotFoundError(path)
        if pq.ParquetFile(path).metadata.num_rows != expected_rows:
            raise ValueError(f"input row count mismatch: {path}")
    if not list(Path(ai_glob).parent.glob(Path(ai_glob).name)):
        raise FileNotFoundError(ai_glob)
    if destination.exists():
        raise FileExistsError(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.staging"
    staging.mkdir()
    temporary = (
        Path(temp_directory).resolve()
        if temp_directory is not None
        else staging / "duckdb_tmp"
    )
    temporary.mkdir(parents=True, exist_ok=True)
    base = staging / "concentration_base.parquet"
    output = staging / "concentration_metrics.parquet"
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={threads}")
        connection.execute(f"SET memory_limit={_lit(memory_limit)}")
        connection.execute(f"SET temp_directory={_lit(str(temporary))}")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(_base_sql(source, quality, ai_glob, base))
        base_rows = int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet({_lit(str(base))})"
            ).fetchone()[0]
        )
        if base_rows != expected_rows:
            raise ValueError("concentration base row count mismatch")
        connection.execute(_metrics_sql(base, output))
        observed = connection.execute(
            f"""
            SELECT count(*), count(distinct concentration_dimension),
              count(*) FILTER (
                WHERE hhi < 0 OR hhi > 1 OR max_value_share < 0 OR max_value_share > 1
              ),
              count(*) FILTER (WHERE distinct_values < 1 OR media_rows < 1)
            FROM read_parquet({_lit(str(output))})
            """
        ).fetchone()
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
        "source_rows_match": base_rows == expected_rows,
        "all_concentration_dimensions_present": int(observed[1]) == len(DIMENSIONS),
        "metrics_within_bounds": int(observed[2]) == 0,
        "all_groups_nonempty": int(observed[3]) == 0,
        "technical_usability_not_inferred": True,
        "manifest_written_last": True,
    }
    if not all(validation.values()):
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(f"concentration validation failed: {validation}")
    artifact = _artifact(output)
    manifest = {
        "schema_version": CONCENTRATION_VERSION,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "code_commit": code_commit,
        "source_snapshot_id": source_snapshot_id,
        "inputs": {
            "v3": str(source),
            "media_quality": str(quality),
            "ai_readiness": ai_glob,
        },
        "counts": {
            "source_rows": base_rows,
            "metric_rows": int(observed[0]),
            "dimensions": int(observed[1]),
        },
        "configuration": {
            "cohorts": ["ALL_MEDIA", "RIGHTS_QUALIFIED"],
            "dimensions": list(DIMENSIONS),
            "technically_usable_cohort_status": "NOT_TESTED",
        },
        "validation": validation,
        "artifacts": [artifact],
        "network_requests": 0,
        "manifest_policy": {"written_last": True},
    }
    _write_json(staging / "manifest.json", manifest)
    if _sha256(output) != artifact["sha256"]:
        raise ValueError("concentration artifact checksum mismatch")
    os.replace(staging, destination)
    return manifest


def _base_sql(source: Path, quality: Path, ai_glob: str, output: Path) -> str:
    return f"""
    COPY (
      SELECT q.media_assertion_id,
        trim(cast(v.gbifID AS VARCHAR)) AS gbifID,
        coalesce(nullif(trim(v.species), ''), '<MISSING>') AS species,
        coalesce(nullif(trim(v.media_publisher), ''), nullif(trim(v.publisher), ''),
          '<MISSING>') AS provider,
        coalesce(nullif(trim(v.media_creator), ''), '<MISSING>') AS creator,
        coalesce(nullif(trim(v.gbifRegion), ''), nullif(trim(v.continent), ''),
          nullif(trim(v.countryCode), ''), '<MISSING>') AS region,
        coalesce(
          cast(floor(try_cast(v.year AS DOUBLE) / 10) * 10 AS VARCHAR), '<MISSING>'
        ) AS decade,
        a.RIGHTS_ALLOWED
      FROM read_parquet({_lit(str(source))}) v
      POSITIONAL JOIN read_parquet({_lit(str(quality))}) q
      JOIN read_parquet({_lit(ai_glob)}) a
        ON q.media_assertion_id = a.media_assertion_id
    ) TO {_lit(str(output))} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
    """


def _metrics_sql(base: Path, output: Path) -> str:
    unions: list[str] = []
    for dimension, field in DIMENSIONS.items():
        for cohort, condition in (
            ("ALL_MEDIA", "TRUE"),
            ("RIGHTS_QUALIFIED", "RIGHTS_ALLOWED = 'PASS'"),
        ):
            unions.append(
                f"""
                WITH grouped AS (
                  SELECT species, {field} AS dimension_value, count(*)::BIGINT value_rows,
                    count(distinct gbifID)::BIGINT value_occurrences
                  FROM read_parquet({_lit(str(base))})
                  WHERE {condition}
                  GROUP BY species, {field}
                ), shares AS (
                  SELECT *, sum(value_rows) OVER (PARTITION BY species) AS species_rows
                  FROM grouped
                )
                SELECT {_lit(CONCENTRATION_VERSION)} AS concentration_version,
                  {_lit(cohort)} AS cohort,
                  species,
                  {_lit(dimension)} AS concentration_dimension,
                  sum(value_rows)::BIGINT AS media_rows,
                  sum(value_occurrences)::BIGINT AS distinct_occurrence_value_sum,
                  count(*)::BIGINT AS distinct_values,
                  max(value_rows)::BIGINT AS largest_value_rows,
                  max(value_rows)::DOUBLE / sum(value_rows) AS max_value_share,
                  sum(power(value_rows::DOUBLE / species_rows, 2)) AS hhi,
                  1.0 / sum(power(value_rows::DOUBLE / species_rows, 2)) AS effective_value_count,
                  'PASS' AS evidence_status
                FROM shares
                GROUP BY species
                """
            )
    combined = " UNION ALL ".join(f"({query})" for query in unions)
    return f"COPY ({combined}) TO {_lit(str(output))} (FORMAT PARQUET, COMPRESSION ZSTD)"


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


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lit(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


__all__ = ["CONCENTRATION_VERSION", "DIMENSIONS", "publish_concentration_metrics"]
