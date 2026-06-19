from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb


def audit_registry(registry_dir: str | Path) -> dict[str, Any]:
    base = Path(registry_dir)
    with duckdb.connect(":memory:") as conn:
        return {
            "registry_dir": str(base),
            "taxa_by_rank": _count_map(conn, base / "taxa.parquet", "rank", where="rank <> ''"),
            "taxa_by_family": _count_map(conn, base / "taxa.parquet", "family", where="family <> ''"),
            "enabled_names_by_class": _count_map(conn, base / "names.parquet", "name_class", where="enabled = true"),
            "names_by_source": _count_map(conn, base / "names.parquet", "source"),
            "names_by_language": _count_map(conn, base / "names.parquet", "language", where="language <> ''"),
            "flickr_queries_by_field": _count_map(
                conn,
                base / "flickr_query_definitions.parquet",
                "search_field",
                where="enabled = true",
            ),
            "qa_by_severity": _count_map(conn, base / "qa_findings.parquet", "severity"),
        }


def _count_map(conn: duckdb.DuckDBPyConnection, parquet_path: Path, column: str, *, where: str = "true") -> dict[str, int]:
    rows = conn.execute(
        f"""
        SELECT {column} AS key, count(*) AS count
        FROM read_parquet(?)
        WHERE {where}
        GROUP BY {column}
        ORDER BY {column}
        """,
        [str(parquet_path)],
    ).fetchall()
    return {str(key): int(count) for key, count in rows}
