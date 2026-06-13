from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import resource
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
            "avg_sec_per_call": (total_sec / result.api_calls_made) if result.api_calls_made else None,
            "p50_call_sec": "not_instrumented",
            "p95_call_sec": "not_instrumented",
        },
        "api_budget": {
            "api_calls_used": result.api_calls_made,
            "remaining_soft_budget": result.remaining_soft_budget,
            "remaining_hard_budget": result.remaining_hard_budget,
            "calls_per_hour": (result.api_calls_made / total_sec * 3600) if total_sec else None,
        },
        "work": {
            "pages_claimed": result.work_items_claimed,
            "raw_responses_written": result.raw_responses_written,
            "stale_claims_requeued": result.stale_claims_requeued,
            "pages_or_probes": result.work_items_claimed,
            "splits": "not_instrumented",
        },
        "rows": {
            "records_fetched": result.evidence_rows_written,
            "source_records_inserted": result.source_records_inserted,
            "duplicate_records_skipped": result.duplicate_records_skipped,
            "query_hits_inserted": result.query_hits_inserted,
            "duplicate_query_hits_skipped": result.duplicate_query_hits_skipped,
            "image_urls_queued": result.image_urls_queued,
            "parquet_rows": result.evidence_rows_written,
        },
        "query_provenance": _query_provenance_summary(query_provenance),
        "throughput": {
            "records_per_call": (result.evidence_rows_written / result.api_calls_made) if result.api_calls_made else None,
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
