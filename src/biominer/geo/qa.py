from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import duckdb


def build_geo_qa_report(
    *,
    classified_path: str | Path,
    geo_candidates_path: str | Path,
    output_dir: str | Path = "reports",
    benchmark_json: str | Path | None = None,
    report_name: str = "geo_qa",
) -> dict[str, object]:
    classified = Path(classified_path)
    geo_candidates = Path(geo_candidates_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(database=":memory:")
    try:
        _create_parquet_view(conn, "classified", classified)
        _create_parquet_view(conn, "geo_candidates", geo_candidates)
        classified_columns = _columns(conn, "classified")
        geo_columns = _columns(conn, "geo_candidates")
        report = {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "classified_path": str(classified),
            "geo_candidates_path": str(geo_candidates),
            "benchmark_json": str(benchmark_json) if benchmark_json else None,
            "classified_rows": _scalar(conn, "SELECT COUNT(*) FROM classified"),
            "geo_candidate_index_rows": _scalar(conn, "SELECT COUNT(*) FROM geo_candidates"),
            "geo_candidate_coverage": _geo_candidate_coverage(conn, classified_columns),
            "geo_candidate_fallback_counts": _fallback_counts(conn, classified_columns),
            "candidate_set_count_distribution": _candidate_set_distribution(conn, classified_columns),
            "candidate_set_count": _candidate_set_count(conn, classified_columns),
            "avg_records_per_candidate_set": _records_per_candidate_set(conn, classified_columns, "avg"),
            "max_records_per_candidate_set": _records_per_candidate_set(conn, classified_columns, "max"),
            "top_large_candidate_sets": _top_large_candidate_sets(conn, classified_columns),
            "empty_suspicious_geo_cells": _empty_suspicious_geo_cells(conn, classified_columns),
            "bucket_counts_by_geo_availability": _bucket_counts_by_geo_availability(conn, classified_columns),
            "geo_index_cell_distribution": _geo_index_cell_distribution(conn, geo_columns),
            "benchmark_summary": _benchmark_summary(benchmark_json),
        }
    finally:
        conn.close()
    json_path = output / f"{report_name}.json"
    markdown_path = output / f"{report_name}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return {**report, "json_report": str(json_path), "markdown_report": str(markdown_path)}


def _create_parquet_view(conn: duckdb.DuckDBPyConnection, view_name: str, path: Path) -> None:
    escaped = str(path).replace("'", "''")
    conn.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{escaped}')")


def _columns(conn: duckdb.DuckDBPyConnection, view_name: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info('{view_name}')").fetchall()}


def _scalar(conn: duckdb.DuckDBPyConnection, sql: str) -> int | float | None:
    value = conn.execute(sql).fetchone()[0]
    return value


def _geo_candidate_coverage(conn: duckdb.DuckDBPyConnection, columns: set[str]) -> dict[str, object]:
    total = int(_scalar(conn, "SELECT COUNT(*) FROM classified") or 0)
    latlon_expr = _latlon_available_expr(columns)
    cell_expr = "geo_candidate_cell_id IS NOT NULL AND geo_candidate_cell_id <> ''" if "geo_candidate_cell_id" in columns else "false"
    candidate_expr = "species_candidate_count IS NOT NULL AND species_candidate_count > 0" if "species_candidate_count" in columns else "false"
    row = conn.execute(
        f"""
        SELECT
          SUM(CASE WHEN {latlon_expr} THEN 1 ELSE 0 END) AS rows_with_geo,
          SUM(CASE WHEN {cell_expr} THEN 1 ELSE 0 END) AS rows_with_geo_candidate_cell,
          SUM(CASE WHEN {candidate_expr} THEN 1 ELSE 0 END) AS rows_with_species_candidates
        FROM classified
        """
    ).fetchone()
    rows_with_geo = int(row[0] or 0)
    rows_with_cell = int(row[1] or 0)
    return {
        "rows": total,
        "rows_with_geo": rows_with_geo,
        "rows_with_geo_candidate_cell": rows_with_cell,
        "rows_with_species_candidates": int(row[2] or 0),
        "geo_candidate_coverage": rows_with_cell / rows_with_geo if rows_with_geo else None,
    }


def _fallback_counts(conn: duckdb.DuckDBPyConnection, columns: set[str]) -> list[dict[str, object]] | str:
    if "geo_candidate_grid_level" not in columns:
        return "not_instrumented"
    fallback = "geo_candidate_fallback_level" if "geo_candidate_fallback_level" in columns else "NULL"
    return _rows(
        conn,
        f"""
        SELECT
          COALESCE(geo_candidate_grid_level, 'missing') AS grid_level,
          COALESCE({fallback}, 'none') AS fallback_level,
          COUNT(*) AS records
        FROM classified
        GROUP BY 1, 2
        ORDER BY records DESC, grid_level, fallback_level
        """,
    )


def _candidate_set_distribution(conn: duckdb.DuckDBPyConnection, columns: set[str]) -> dict[str, object] | str:
    if "species_candidate_count" not in columns:
        return "not_instrumented"
    row = conn.execute(
        """
        SELECT
          COUNT(species_candidate_count) AS count,
          MIN(species_candidate_count) AS min,
          AVG(species_candidate_count) AS avg,
          QUANTILE_CONT(species_candidate_count, 0.5) AS p50,
          QUANTILE_CONT(species_candidate_count, 0.95) AS p95,
          MAX(species_candidate_count) AS max
        FROM classified
        WHERE species_candidate_count IS NOT NULL
        """
    ).fetchone()
    return {
        "count": int(row[0] or 0),
        "min": int(row[1]) if row[1] is not None else None,
        "avg": float(row[2]) if row[2] is not None else None,
        "p50": float(row[3]) if row[3] is not None else None,
        "p95": float(row[4]) if row[4] is not None else None,
        "max": int(row[5]) if row[5] is not None else None,
    }


def _candidate_set_count(conn: duckdb.DuckDBPyConnection, columns: set[str]) -> int | str:
    if "candidate_set_signature" not in columns:
        return "not_instrumented"
    return int(_scalar(conn, "SELECT COUNT(DISTINCT candidate_set_signature) FROM classified") or 0)


def _records_per_candidate_set(conn: duckdb.DuckDBPyConnection, columns: set[str], metric: str) -> float | int | str:
    if "candidate_set_signature" not in columns:
        return "not_instrumented"
    aggregate = "AVG(records)" if metric == "avg" else "MAX(records)"
    value = _scalar(
        conn,
        f"""
        SELECT {aggregate}
        FROM (
          SELECT candidate_set_signature, COUNT(*) AS records
          FROM classified
          GROUP BY candidate_set_signature
        )
        """,
    )
    if value is None:
        return 0.0 if metric == "avg" else 0
    return float(value) if metric == "avg" else int(value)


def _top_large_candidate_sets(conn: duckdb.DuckDBPyConnection, columns: set[str]) -> list[dict[str, object]] | str:
    if "candidate_set_signature" not in columns:
        return "not_instrumented"
    species_count = "MAX(species_candidate_count) AS species_candidate_count" if "species_candidate_count" in columns else "NULL AS species_candidate_count"
    grid = "MAX(geo_candidate_grid_level) AS geo_candidate_grid_level" if "geo_candidate_grid_level" in columns else "NULL AS geo_candidate_grid_level"
    return _rows(
        conn,
        f"""
        SELECT candidate_set_signature, COUNT(*) AS records, {species_count}, {grid}
        FROM classified
        GROUP BY candidate_set_signature
        ORDER BY records DESC, candidate_set_signature
        LIMIT 10
        """,
    )


def _empty_suspicious_geo_cells(conn: duckdb.DuckDBPyConnection, columns: set[str]) -> list[dict[str, object]] | str:
    if "species_candidate_count" not in columns and "geo_candidate_cell_id" not in columns:
        return "not_instrumented"
    latlon_expr = _latlon_available_expr(columns)
    candidate_expr = "species_candidate_count IS NULL OR species_candidate_count = 0" if "species_candidate_count" in columns else "true"
    cell = "geo_candidate_cell_id" if "geo_candidate_cell_id" in columns else "NULL"
    grid = "geo_candidate_grid_level" if "geo_candidate_grid_level" in columns else "NULL"
    return _rows(
        conn,
        f"""
        SELECT COALESCE({grid}, 'missing') AS grid_level, COALESCE({cell}, 'missing') AS geocell_id, COUNT(*) AS records
        FROM classified
        WHERE {latlon_expr} AND ({candidate_expr})
        GROUP BY 1, 2
        ORDER BY records DESC, grid_level, geocell_id
        LIMIT 20
        """,
    )


def _bucket_counts_by_geo_availability(conn: duckdb.DuckDBPyConnection, columns: set[str]) -> list[dict[str, object]] | str:
    if "occurrence_bin" not in columns:
        return "not_instrumented"
    latlon_expr = _latlon_available_expr(columns)
    return _rows(
        conn,
        f"""
        SELECT occurrence_bin, {latlon_expr} AS has_geo, COUNT(*) AS records
        FROM classified
        GROUP BY 1, 2
        ORDER BY occurrence_bin, has_geo
        """,
    )


def _geo_index_cell_distribution(conn: duckdb.DuckDBPyConnection, columns: set[str]) -> dict[str, object] | str:
    if "grid_level" not in columns or "geocell_id" not in columns:
        return "not_instrumented"
    row = conn.execute(
        """
        WITH cell_counts AS (
          SELECT grid_level, geocell_id, COUNT(DISTINCT species_key) AS species_per_cell
          FROM geo_candidates
          GROUP BY 1, 2
        )
        SELECT
          (SELECT COUNT(*) FROM geo_candidates) AS rows,
          (SELECT COUNT(DISTINCT geocell_id) FROM geo_candidates) AS cells,
          (SELECT COUNT(DISTINCT species_key) FROM geo_candidates) AS species,
          AVG(species_per_cell) AS avg_species_per_cell,
          MAX(species_per_cell) AS max_species_per_cell
        FROM cell_counts
        """
    ).fetchone()
    return {
        "rows": int(row[0] or 0),
        "cells": int(row[1] or 0),
        "species": int(row[2] or 0),
        "avg_species_per_cell": float(row[3]) if row[3] is not None else None,
        "max_species_per_cell": int(row[4]) if row[4] is not None else None,
    }


def _benchmark_summary(path: str | Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    source = Path(path)
    if not source.exists():
        return {"error": "benchmark_json_missing", "path": str(source)}
    payload = json.loads(source.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    return {
        "path": str(source),
        "run_id": payload.get("run_id"),
        "configurations": len(runs) if isinstance(runs, list) else 0,
    }


def _latlon_available_expr(columns: set[str]) -> str:
    if {"latitude", "longitude"}.issubset(columns):
        return "latitude IS NOT NULL AND longitude IS NOT NULL"
    if {"decimalLatitude", "decimalLongitude"}.issubset(columns):
        return "decimalLatitude IS NOT NULL AND decimalLongitude IS NOT NULL"
    return "false"


def _rows(conn: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, object]]:
    columns = [description[0] for description in conn.execute(sql).description]
    return [dict(zip(columns, row, strict=True)) for row in conn.fetchall()]


def _markdown_report(report: dict[str, object]) -> str:
    coverage = report.get("geo_candidate_coverage")
    distribution = report.get("candidate_set_count_distribution")
    lines = [
        "# Geo QA Report",
        "",
        f"- Classified rows: {report.get('classified_rows')}",
        f"- Geo candidate index rows: {report.get('geo_candidate_index_rows')}",
        f"- Candidate set count: {report.get('candidate_set_count')}",
        f"- Avg records per candidate set: {report.get('avg_records_per_candidate_set')}",
        f"- Max records per candidate set: {report.get('max_records_per_candidate_set')}",
    ]
    if isinstance(coverage, dict):
        lines.append(f"- Geo candidate coverage: {coverage.get('geo_candidate_coverage')}")
    if isinstance(distribution, dict):
        lines.append(f"- Candidate count p95: {distribution.get('p95')}")
    lines.extend(["", "## Fallback Counts", ""])
    fallback_counts = report.get("geo_candidate_fallback_counts")
    if isinstance(fallback_counts, list):
        for row in fallback_counts[:10]:
            lines.append(f"- {row.get('grid_level')} / {row.get('fallback_level')}: {row.get('records')}")
    else:
        lines.append(f"- {fallback_counts}")
    lines.extend(["", "## Top Large Candidate Sets", ""])
    large_sets = report.get("top_large_candidate_sets")
    if isinstance(large_sets, list):
        for row in large_sets[:10]:
            lines.append(f"- {row.get('candidate_set_signature')}: {row.get('records')} records")
    else:
        lines.append(f"- {large_sets}")
    lines.append("")
    return "\n".join(lines)
