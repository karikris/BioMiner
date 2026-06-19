from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import resource
import sqlite3
import subprocess
from typing import Any

import polars as pl

from biominer.flickr_fetch.metadata_poller import PollOnceResult


def write_step1_manifest(
    path: str | Path,
    *,
    run_id: str,
    command: list[str],
    expected_outputs: dict[str, str],
    expected_pages: int,
    status: str,
    pid: int | None = None,
    started_at: str | None = None,
    git_sha: str | None = None,
) -> dict[str, Any]:
    payload = {
        "run_id": run_id,
        "command": command,
        "git_sha": git_sha or current_git_sha(),
        "pid": pid or os.getpid(),
        "status": status,
        "start_time": started_at or datetime.now(UTC).isoformat(),
        "end_time": None,
        "expected_pages": expected_pages,
        "expected_outputs": expected_outputs,
        "environment": environment_summary(),
    }
    _write_json(path, payload)
    return payload


def build_step1_fetch_report(
    *,
    run_id: str,
    command: list[str],
    result: PollOnceResult,
    raw_root: str | Path,
    evidence_output: str | Path,
    query_provenance: pl.DataFrame | None = None,
    started_at: datetime,
    ended_at: datetime,
    workers: int,
    expected_pages: int,
    status: str,
    pid: int | None = None,
    git_sha: str | None = None,
) -> dict[str, Any]:
    total_sec = max(0.0, (ended_at - started_at).total_seconds())
    raw_bytes = _tree_bytes(Path(raw_root))
    evidence_bytes = _file_bytes(Path(evidence_output))
    state_bytes = _file_bytes(result.state_db)
    state_work = _state_work_summary(result.state_db)
    call_timings = _api_call_timing_summary(result.state_db)
    return {
        "run_id": run_id,
        "command": command,
        "git_sha": git_sha or current_git_sha(),
        "pid": pid or os.getpid(),
        "status": status,
        "start_time": started_at.isoformat(),
        "end_time": ended_at.isoformat(),
        "expected_pages": expected_pages,
        "workers": workers,
        "timings": {
            "total_sec": total_sec,
            "avg_sec_per_call": call_timings["avg_sec_per_call"],
            "p50_call_sec": call_timings["p50_call_sec"],
            "p95_call_sec": call_timings["p95_call_sec"],
        },
        "api_budget": {
            "api_calls_used": result.api_calls_made,
            "remaining_soft_budget": result.remaining_soft_budget,
            "remaining_hard_budget": result.remaining_hard_budget,
            "calls_per_hour": (result.api_calls_made / total_sec * 3600) if total_sec else None,
            "budget_limited_exit": result.remaining_soft_budget == 0 and bool(state_work.get("has_pending_work")),
        },
        "work": {
            "pages_claimed": result.work_items_claimed,
            "raw_responses_written": result.raw_responses_written,
            "stale_claims_requeued": result.stale_claims_requeued,
            "pages_or_probes": result.work_items_claimed,
            "splits": state_work["split_probes_enqueued_by_reason"],
            **state_work,
        },
        "rows": {
            "records_fetched": result.evidence_rows_written,
            "records_inserted": result.source_records_inserted,
            "source_records_inserted": result.source_records_inserted,
            "duplicate_records_skipped": result.duplicate_records_skipped,
            "query_hits_inserted": result.query_hits_inserted,
            "duplicate_query_hits_skipped": result.duplicate_query_hits_skipped,
            "image_urls_queued": result.image_urls_queued,
            "parquet_rows": result.evidence_rows_total,
        },
        "query_provenance": _query_provenance_summary(query_provenance),
        "throughput": {
            "records_per_call": (result.evidence_rows_written / result.api_calls_made) if result.api_calls_made else None,
            "records_per_page": (result.evidence_rows_written / result.raw_responses_written) if result.raw_responses_written else None,
        },
        "storage_bytes": {
            "raw_json_bytes": raw_bytes,
            "evidence_parquet_bytes": evidence_bytes,
            "state_db_bytes": state_bytes,
            "total_artifact_bytes": raw_bytes + evidence_bytes + state_bytes,
        },
        "distributions": {
            "bucket_counts": "not_instrumented",
            "category_counts": "not_instrumented",
            "life_stage_counts": "not_instrumented",
        },
        "memory": {
            "rss_kb": _max_rss_kb(),
            "max_rss_kb": _max_rss_kb(),
            "peak_memory": "not_instrumented",
            "peak_traced_bytes": "not_instrumented",
        },
        "gpu_memory": "not_instrumented",
        "failures": {
            "failed_pages": max(0, result.work_items_claimed - result.raw_responses_written),
        },
        "environment": environment_summary(),
    }


def write_step1_fetch_report(path: str | Path, report: dict[str, Any]) -> None:
    _write_json(path, report)


def _query_provenance_summary(frame: pl.DataFrame | None) -> dict[str, Any]:
    if frame is None or frame.is_empty():
        return {
            "unique_query_labels_with_records": "not_instrumented",
            "duplicate_records_with_additional_query_hits": "not_instrumented",
            "query_hit_count_distribution": "not_instrumented",
            "top_query_labels_by_records": "not_instrumented",
        }
    exploded = frame.select("all_query_labels").explode("all_query_labels").drop_nulls()
    top_labels = (
        exploded.group_by("all_query_labels")
        .len(name="records")
        .sort(["records", "all_query_labels"], descending=[True, False])
        .rename({"all_query_labels": "query_label"})
        .to_dicts()
    )
    distribution = (
        frame.group_by("query_hit_count")
        .len(name="records")
        .sort("query_hit_count")
        .to_dicts()
    )
    return {
        "unique_query_labels_with_records": exploded["all_query_labels"].n_unique() if exploded.height else 0,
        "duplicate_records_with_additional_query_hits": frame.filter(pl.col("query_hit_count") > 1).height,
        "query_hit_count_distribution": distribution,
        "top_query_labels_by_records": top_labels[:20],
    }


def _state_work_summary(path: Path) -> dict[str, Any]:
    fallback = {
        "count_probes_completed": "not_instrumented",
        "page_fetches_completed": "not_instrumented",
        "split_probes_enqueued_by_reason": "not_instrumented",
        "pending_count_probes": "not_instrumented",
        "pending_page_fetches": "not_instrumented",
        "completed_date_slices": "not_instrumented",
        "pending_date_slices": "not_instrumented",
        "last_completed_date_range": None,
        "next_pending_date_range": None,
        "has_pending_work": None,
        "saturated_slices": "not_instrumented",
        "saturated_slice_count": "not_instrumented",
        "slice_page1_completed": "not_instrumented",
        "remaining_pages_enqueued_from_page1": "not_instrumented",
        "empty_or_single_page_slices": "not_instrumented",
        "page_calls_avoided_estimate": "not_instrumented",
        "reported_over_window_slices": "not_instrumented",
        "saturated_remediation_pending": "not_instrumented",
        "saturated_remediation_enqueued": "not_instrumented",
    }
    if not path.exists():
        return fallback
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}
            if "flickr_work_items" not in tables:
                return fallback
            saturated_slices = _saturated_slices(conn)
            page1_completed = _slice_page1_completed(conn)
            remaining_pages = _remaining_pages_enqueued_from_page1(conn)
            return {
                "count_probes_completed": _one(conn, "SELECT count(*) FROM flickr_work_items WHERE lane = 'count_probe' AND status = 'completed'"),
                "page_fetches_completed": _one(conn, "SELECT count(*) FROM flickr_work_items WHERE lane IN ('normal_page', 'bbox_page') AND status = 'completed'"),
                "split_probes_enqueued_by_reason": _split_counts(conn),
                "pending_count_probes": _one(conn, "SELECT count(*) FROM flickr_work_items WHERE lane = 'count_probe' AND status = 'pending'"),
                "pending_page_fetches": _one(conn, "SELECT count(*) FROM flickr_work_items WHERE lane IN ('normal_page', 'bbox_page') AND status = 'pending'"),
                "completed_date_slices": _one(conn, "SELECT count(*) FROM flickr_work_items WHERE status = 'completed' AND COALESCE(date_kind, '') != ''"),
                "pending_date_slices": _one(conn, "SELECT count(*) FROM flickr_work_items WHERE status = 'pending' AND COALESCE(date_kind, '') != ''"),
                "last_completed_date_range": _date_range(conn, status="completed", descending=True),
                "next_pending_date_range": _date_range(conn, status="pending", descending=False),
                "has_pending_work": bool(_one(conn, "SELECT count(*) FROM flickr_work_items WHERE status = 'pending'")),
                "saturated_slices": saturated_slices,
                "saturated_slice_count": len(saturated_slices),
                "slice_page1_completed": page1_completed,
                "remaining_pages_enqueued_from_page1": remaining_pages,
                "empty_or_single_page_slices": _empty_or_single_page_slices(conn),
                "page_calls_avoided_estimate": max(0, _remaining_page_capacity_from_page1(conn) - remaining_pages),
                "reported_over_window_slices": _reported_over_window_slices(conn),
                "saturated_remediation_pending": len(saturated_slices),
                "saturated_remediation_enqueued": _saturated_remediation_enqueued(conn),
            }
    except sqlite3.DatabaseError:
        return fallback


def _api_call_timing_summary(path: Path) -> dict[str, float | None | str]:
    fallback: dict[str, float | None | str] = {
        "avg_sec_per_call": None,
        "p50_call_sec": "not_instrumented",
        "p95_call_sec": "not_instrumented",
    }
    if not path.exists():
        return fallback
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}
            if "api_call_ledger" not in tables or not _has_api_columns(conn, "duration_sec"):
                return fallback
            durations = [
                float(row["duration_sec"])
                for row in conn.execute(
                    "SELECT duration_sec FROM api_call_ledger WHERE duration_sec IS NOT NULL ORDER BY duration_sec"
                ).fetchall()
            ]
    except sqlite3.DatabaseError:
        return fallback
    if not durations:
        return fallback
    return {
        "avg_sec_per_call": sum(durations) / len(durations),
        "p50_call_sec": _percentile(durations, 50),
        "p95_call_sec": _percentile(durations, 95),
    }


def _has_api_columns(conn: sqlite3.Connection, *names: str) -> bool:
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(api_call_ledger)").fetchall()}
    return all(name in existing for name in names)


def _percentile(sorted_values: list[float], percentile: int) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    index = max(0, min(len(sorted_values) - 1, int(((percentile / 100) * len(sorted_values) + 0.999999) - 1)))
    return sorted_values[index]


def _one(conn: sqlite3.Connection, sql: str) -> int:
    return int(conn.execute(sql).fetchone()[0])


def _split_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT split_reason, count(*)
        FROM flickr_work_items
        WHERE lane = 'count_probe' AND split_reason IS NOT NULL
        GROUP BY split_reason
        ORDER BY split_reason
        """
    ).fetchall()
    return {str(row[0]): int(row[1]) for row in rows}


def _date_range(conn: sqlite3.Connection, *, status: str, descending: bool) -> dict[str, str] | None:
    order = "DESC" if descending else "ASC"
    row = conn.execute(
        f"""
        SELECT date_kind, min_date, max_date
        FROM flickr_work_items
        WHERE status = ? AND COALESCE(date_kind, '') != ''
        ORDER BY min_date {order}, max_date {order}
        LIMIT 1
        """,
        (status,),
    ).fetchone()
    if row is None:
        return None
    return {"date_kind": str(row["date_kind"]), "min_date": str(row["min_date"]), "max_date": str(row["max_date"])}


def _saturated_slices(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(flickr_work_items)").fetchall()}
    if "records_returned" not in existing:
        return []
    rows = conn.execute(
        """
        SELECT date_kind, min_date, max_date, page, records_returned
        FROM flickr_work_items
        WHERE status = 'completed'
          AND lane = 'normal_page'
          AND (
            (page = 8 AND per_page = 500 AND records_returned = 500)
            OR (page = 16 AND per_page = 250 AND records_returned = 250)
          )
          AND COALESCE(date_kind, '') != ''
        ORDER BY min_date, max_date, page
        """
    ).fetchall()
    return [
        {
            "date_kind": str(row["date_kind"]),
            "min_date": str(row["min_date"]),
            "max_date": str(row["max_date"]),
            "page": int(row["page"]),
            "records_returned": int(row["records_returned"]),
        }
        for row in rows
    ]


def _has_columns(conn: sqlite3.Connection, *names: str) -> bool:
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(flickr_work_items)").fetchall()}
    return all(name in existing for name in names)


def _slice_page1_completed(conn: sqlite3.Connection) -> int:
    return _one(
        conn,
        """
        SELECT count(*)
        FROM flickr_work_items
        WHERE status = 'completed'
          AND lane = 'normal_page'
          AND page = 1
          AND split_reason = 'upload_date'
          AND COALESCE(date_kind, '') != ''
        """,
    )


def _remaining_pages_enqueued_from_page1(conn: sqlite3.Connection) -> int:
    return _one(
        conn,
        """
        SELECT count(*)
        FROM flickr_work_items
        WHERE lane = 'normal_page'
          AND page BETWEEN 2 AND 16
          AND split_reason = 'upload_date'
          AND COALESCE(date_kind, '') != ''
        """,
    )


def _remaining_page_capacity_from_page1(conn: sqlite3.Connection) -> int:
    if not _has_columns(conn, "response_perpage"):
        return _slice_page1_completed(conn) * 7
    rows = conn.execute(
        """
        SELECT COALESCE(response_perpage, per_page) AS perpage
        FROM flickr_work_items
        WHERE status = 'completed'
          AND lane = 'normal_page'
          AND page = 1
          AND split_reason = 'upload_date'
          AND COALESCE(date_kind, '') != ''
        """
    ).fetchall()
    capacity = 0
    for row in rows:
        perpage = int(row["perpage"] or 500)
        capacity += max(0, (4000 // perpage) - 1) if perpage > 0 else 0
    return capacity


def _empty_or_single_page_slices(conn: sqlite3.Connection) -> int:
    if not _has_columns(conn, "response_pages"):
        return 0
    return _one(
        conn,
        """
        SELECT count(*)
        FROM flickr_work_items
        WHERE status = 'completed'
          AND lane = 'normal_page'
          AND page = 1
          AND split_reason = 'upload_date'
          AND COALESCE(response_pages, 0) <= 1
        """,
    )


def _reported_over_window_slices(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if not _has_columns(conn, "response_total", "response_pages"):
        return []
    rows = conn.execute(
        """
        SELECT date_kind, min_date, max_date, response_total, response_pages
        FROM flickr_work_items
        WHERE status = 'completed'
          AND lane = 'normal_page'
          AND page = 1
          AND split_reason = 'upload_date'
          AND (
            COALESCE(response_total, 0) > 4000
            OR COALESCE(response_pages, 0) > (4000 / COALESCE(NULLIF(response_perpage, 0), per_page, 500))
          )
        ORDER BY min_date, max_date
        """
    ).fetchall()
    return [
        {
            "date_kind": str(row["date_kind"]),
            "min_date": str(row["min_date"]),
            "max_date": str(row["max_date"]),
            "response_total": int(row["response_total"]),
            "response_pages": int(row["response_pages"]),
        }
        for row in rows
    ]


def _saturated_remediation_enqueued(conn: sqlite3.Connection) -> int:
    if not _has_columns(conn, "parent_query_hash"):
        return 0
    return _one(
        conn,
        """
        SELECT count(*)
        FROM flickr_work_items
        WHERE lane = 'normal_page'
          AND page = 1
          AND split_reason = 'upload_date'
          AND parent_query_hash IS NOT NULL
          AND COALESCE(date_kind, '') != ''
        """,
    )


def environment_summary() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
    }


def current_git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _file_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _max_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
