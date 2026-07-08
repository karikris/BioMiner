from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any

from biominer.flickr_fetch.metadata_poller import MetadataPollState, poll_once
from biominer.flickr_fetch.query_planner import FLICKR_SEARCH_RESULT_WINDOW, FlickrQuery, NORMAL_PAGE_SIZE, SearchField, query_hash
from biominer.reports.flickr_fetch import current_git_sha, write_step1_manifest
from biominer.secrets_loader import load_runtime_secrets_env


DEFAULT_START_DATE = "2004-02-10"
DEFAULT_END_DATE = "2026-07-08"
DEFAULT_REGISTRY_VERSION = "ad-hoc-papilio-demoleus-ranked-slices-20260708"

SIX_YEAR_SOURCE_ROWS: tuple[tuple[int, str, SearchField], ...] = (
    (16, "Chequered Swallowtail", "text"),
    (17, "Lime Butterfly", "tags"),
    (18, "na citrus", "text"),
    (19, "達摩鳳蝶", "text"),
    (20, "Lemon Swallowtail", "text"),
    (21, "Papilio malayanus", "text"),
    (22, "Checkered Swallowtail", "text"),
    (23, "花鳳蝶", "text"),
    (24, "Citrus Swallowtail", "tags"),
    (25, "Small Citrus Butterfly", "text"),
)

ONE_YEAR_SOURCE_ROWS: tuple[tuple[int, str, SearchField], ...] = (
    (1, "Papilio demoleus", "text"),
    (2, "Lemon Butterfly", "text"),
    (3, "Lime Butterfly", "text"),
    (4, "無尾鳳蝶", "text"),
    (5, "Citrus Swallowtail", "text"),
    (6, "Lemon Butterfly", "text"),
    (7, "Lime Butterfly", "text"),
    (8, "Lime Butterfly", "text"),
    (9, "Lemon Butterfly", "text"),
    (10, "Papilio demoleus", "tags"),
    (11, "Lime Swallowtail", "text"),
    (12, "無尾鳳蝶", "tags"),
    (13, "Common Lime Butterfly", "text"),
    (14, "Lime Butterfly", "text"),
    (15, "Lemon Butterfly", "text"),
)

LANGUAGE_BY_TERM = {
    "Papilio demoleus": ("lat", "la"),
    "Papilio malayanus": ("lat", "la"),
    "無尾鳳蝶": ("zho", "zh"),
    "達摩鳳蝶": ("zho", "zh"),
    "花鳳蝶": ("zho", "zh"),
    "na citrus": ("hau", "ha"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Papilio demoleus ranked Flickr follow-up queries over annual and six-year upload slices.")
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--state-db")
    parser.add_argument("--raw-root")
    parser.add_argument("--evidence-output")
    parser.add_argument("--manifest")
    parser.add_argument("--report-json")
    parser.add_argument("--report-md")
    parser.add_argument("--plan-json")
    parser.add_argument("--plan-md")
    parser.add_argument("--progress-log")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--max-api-calls", type=int, default=3500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--api-key-env", default="FLICKR_API_KEY")
    parser.add_argument("--secrets-env")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def six_year_query_specs() -> tuple[tuple[int, str, SearchField], ...]:
    return _dedupe_specs(SIX_YEAR_SOURCE_ROWS)


def one_year_query_specs() -> tuple[tuple[int, str, SearchField], ...]:
    return _dedupe_specs(ONE_YEAR_SOURCE_ROWS)


def calendar_year_upload_ranges(*, start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE) -> tuple[tuple[str, str], ...]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    current = start
    ranges: list[tuple[str, str]] = []
    while current <= end:
        range_end = min(date(current.year, 12, 31), end)
        ranges.append((current.isoformat(), range_end.isoformat()))
        current = date(current.year + 1, 1, 1)
    return tuple(ranges)


def six_year_upload_ranges(*, start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE) -> tuple[tuple[str, str], ...]:
    requested_start = date.fromisoformat(start_date)
    requested_end = date.fromisoformat(end_date)
    if requested_end < requested_start:
        raise ValueError("end_date must be on or after start_date")
    ranges = (
        (date(2004, 2, 10), date(2009, 12, 31)),
        (date(2010, 1, 1), date(2015, 12, 31)),
        (date(2016, 1, 1), date(2021, 12, 31)),
        (date(2022, 1, 1), requested_end),
    )
    output: list[tuple[str, str]] = []
    for start, end in ranges:
        clipped_start = max(start, requested_start)
        clipped_end = min(end, requested_end)
        if clipped_start <= clipped_end:
            output.append((clipped_start.isoformat(), clipped_end.isoformat()))
    return tuple(output)


def build_ranked_slice_queries(*, start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE) -> tuple[FlickrQuery, ...]:
    queries: list[FlickrQuery] = []
    for source_rank, term, search_field in six_year_query_specs():
        for slice_index, (start, end) in enumerate(six_year_upload_ranges(start_date=start_date, end_date=end_date)):
            queries.append(_query(source_rank, term, search_field, start, end, slice_group="six_year", slice_index=slice_index))
    annual_offset = len(queries)
    for source_rank, term, search_field in one_year_query_specs():
        for slice_index, (start, end) in enumerate(calendar_year_upload_ranges(start_date=start_date, end_date=end_date)):
            queries.append(
                _query(
                    source_rank,
                    term,
                    search_field,
                    start,
                    end,
                    slice_group="one_year",
                    slice_index=annual_offset + slice_index,
                )
            )
    return tuple(sorted(queries, key=lambda item: (item.query_priority, item.min_upload_date or "", item.max_upload_date or "", item.term, item.search_field)))


def seed_ranked_slice_work(
    *,
    state_db: str | Path,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> int:
    state = MetadataPollState(state_db)
    return sum(state.enqueue_work_item(query) for query in build_ranked_slice_queries(start_date=start_date, end_date=end_date))


def main() -> int:
    args = build_parser().parse_args()
    started = datetime.now(UTC)
    started_monotonic = time.perf_counter()
    run_id = args.run_id or f"papilio_demoleus_ranked_slices_{started.strftime('%Y%m%dT%H%M%SZ')}"
    paths = _resolve_paths(args, run_id)
    queries = build_ranked_slice_queries(start_date=args.start_date, end_date=args.end_date)
    plan = _build_plan(run_id=run_id, queries=queries, args=args, paths=paths)
    _write_plan_reports(plan, paths["plan_json"], paths["plan_md"])
    if args.plan_only:
        print(json.dumps({"run_id": run_id, "status": "planned", "query_count": len(queries), "plan_json": str(paths["plan_json"]), "plan_md": str(paths["plan_md"])}, ensure_ascii=False))
        return 0

    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["report_dir"].mkdir(parents=True, exist_ok=True)
    _log_event({"event": "run_started", "run_id": run_id, "query_count": len(queries)}, paths["progress_log"])
    manifest = write_step1_manifest(
        paths["manifest"],
        run_id=run_id,
        command=["scripts/run_papilio_demoleus_ranked_flickr_slices.py", *sys.argv[1:]],
        expected_outputs={
            "state_db": str(paths["state_db"]),
            "raw_root": str(paths["raw_root"]),
            "evidence_output": str(paths["evidence_output"]),
            "report_json": str(paths["report_json"]),
            "report_md": str(paths["report_md"]),
            "plan_json": str(paths["plan_json"]),
            "plan_md": str(paths["plan_md"]),
            "progress_log": str(paths["progress_log"]),
        },
        expected_pages=len(queries),
        status="running",
        started_at=started.isoformat(),
        git_sha=current_git_sha(),
    )
    state = MetadataPollState(paths["state_db"])
    inserted = sum(state.enqueue_work_item(query) for query in queries)
    _log_event({"event": "work_seeded", "inserted": inserted, "work_item_count": state.work_item_count()}, paths["progress_log"])
    result = poll_once(
        state_db=paths["state_db"],
        raw_root=paths["raw_root"],
        evidence_output=paths["evidence_output"],
        max_api_calls=args.max_api_calls,
        api_key=_load_api_key(api_key_env=args.api_key_env, secrets_env=args.secrets_env),
        workers=args.workers,
        progress_callback=lambda event: _log_event(event, paths["progress_log"]),
        run_id=run_id,
        worker_id=os.environ.get("BIOMINER_WORKER_ID") or "ranked-slice-runner",
        compact_after_run=True,
    )
    ended = datetime.now(UTC)
    status_counts = _status_counts(paths["state_db"])
    status = "complete" if status_counts.get("pending", 0) == 0 and status_counts.get("claimed", 0) == 0 and status_counts.get("failed", 0) == 0 else "incomplete"
    report = {
        "run_id": run_id,
        "status": status,
        "git_sha": current_git_sha(),
        "started_at": started.isoformat(),
        "finished_at": ended.isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started_monotonic, 3),
        "command": ["scripts/run_papilio_demoleus_ranked_flickr_slices.py", *sys.argv[1:]],
        "plan": plan,
        "status_counts": status_counts,
        "poll_result": {**result.__dict__, "state_db": str(result.state_db)},
        "db_summary": _db_summary(paths["state_db"]),
        "slice_results": _slice_rows(paths["state_db"]),
        "artifacts": {key: str(value) for key, value in paths.items() if key not in {"report_dir", "run_dir"}},
    }
    _write_run_reports(report, paths["report_json"], paths["report_md"])
    manifest["status"] = status
    manifest["end_time"] = ended.isoformat()
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _log_event({"event": "run_completed", "run_id": run_id, "status": status, "api_calls_made": result.api_calls_made}, paths["progress_log"])
    print(json.dumps({"run_id": run_id, "status": status, "report_json": str(paths["report_json"]), "report_md": str(paths["report_md"])}, ensure_ascii=False))
    return 0


def _dedupe_specs(rows: tuple[tuple[int, str, SearchField], ...]) -> tuple[tuple[int, str, SearchField], ...]:
    seen: set[tuple[str, SearchField]] = set()
    output: list[tuple[int, str, SearchField]] = []
    for source_rank, term, search_field in rows:
        key = (term, search_field)
        if key in seen:
            continue
        seen.add(key)
        output.append((source_rank, term, search_field))
    return tuple(output)


def _query(
    source_rank: int,
    term: str,
    search_field: SearchField,
    start: str,
    end: str,
    *,
    slice_group: str,
    slice_index: int,
) -> FlickrQuery:
    language, bcp47 = LANGUAGE_BY_TERM.get(term, ("eng", "en"))
    is_scientific = language == "lat"
    slug = _slug(term)
    return FlickrQuery(
        term=term,
        language=language,
        bcp47=bcp47,
        search_field=search_field,
        lane="normal_page",
        page=1,
        per_page=NORMAL_PAGE_SIZE,
        has_geo=0,
        min_upload_date=start,
        max_upload_date=end,
        split_reason="upload_date",
        split_depth=1,
        slice_index=slice_index,
        registry_version=DEFAULT_REGISTRY_VERSION,
        query_definition_id=f"ranked-slice:{slice_group}:{search_field}:{slug}:{start}:{end}",
        accepted_scientific_name="Papilio demoleus",
        term_type="scientific_name" if is_scientific else "common_name",
        term_confidence="ranked_followup",
        trust_tier="T5",
        query_priority=(1000 if slice_group == "one_year" else 2000) + source_rank,
    )


def _slug(value: str) -> str:
    lowered = value.casefold()
    ascii_slug = "".join(character if character.isascii() and character.isalnum() else "_" for character in lowered).strip("_")
    ascii_slug = "_".join(part for part in ascii_slug.split("_") if part)
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{ascii_slug or 'term'}_{digest}"


def _resolve_paths(args: argparse.Namespace, run_id: str) -> dict[str, Path]:
    output_root = Path(args.output_root)
    report_dir = Path(args.report_dir)
    run_dir = output_root / run_id
    return {
        "run_dir": run_dir,
        "report_dir": report_dir,
        "state_db": Path(args.state_db) if args.state_db else run_dir / "flickr_poller.sqlite",
        "raw_root": Path(args.raw_root) if args.raw_root else run_dir / "raw",
        "evidence_output": Path(args.evidence_output) if args.evidence_output else run_dir / "evidence.parquet",
        "manifest": Path(args.manifest) if args.manifest else run_dir / "manifest.json",
        "report_json": Path(args.report_json) if args.report_json else report_dir / f"{run_id}.json",
        "report_md": Path(args.report_md) if args.report_md else report_dir / f"{run_id}.md",
        "plan_json": Path(args.plan_json) if args.plan_json else report_dir / f"{run_id}_query_plan.json",
        "plan_md": Path(args.plan_md) if args.plan_md else report_dir / f"{run_id}_query_plan.md",
        "progress_log": Path(args.progress_log) if args.progress_log else run_dir / "progress.jsonl",
    }


def _build_plan(*, run_id: str, queries: tuple[FlickrQuery, ...], args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    rows = []
    for query in queries:
        rows.append(
            {
                "term": query.term,
                "search_field": query.search_field,
                "language": query.language,
                "bcp47": query.bcp47,
                "slice_group": "one_year" if (query.query_priority or 0) < 2000 else "six_year",
                "min_upload_date": query.min_upload_date,
                "max_upload_date": query.max_upload_date,
                "page": query.page,
                "per_page": query.per_page,
                "query_priority": query.query_priority,
                "query_definition_id": query.query_definition_id,
                "query_hash": query_hash(query),
            }
        )
    return {
        "run_id": run_id,
        "git_sha": current_git_sha(),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "query_count": len(queries),
        "initial_page1_queries": len(queries),
        "six_year_specs": [{"rank": rank, "term": term, "search_field": field} for rank, term, field in six_year_query_specs()],
        "one_year_specs": [{"rank": rank, "term": term, "search_field": field} for rank, term, field in one_year_query_specs()],
        "six_year_ranges": [{"min_upload_date": start, "max_upload_date": end} for start, end in six_year_upload_ranges(start_date=args.start_date, end_date=args.end_date)],
        "one_year_ranges": [{"min_upload_date": start, "max_upload_date": end} for start, end in calendar_year_upload_ranges(start_date=args.start_date, end_date=args.end_date)],
        "max_api_calls": args.max_api_calls,
        "workers": args.workers,
        "paths": {key: str(value) for key, value in paths.items() if key not in {"report_dir", "run_dir"}},
        "queries": rows,
    }


def _write_plan_reports(plan: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# {plan['run_id']} Query Plan",
        "",
        f"- start_date: {plan['start_date']}",
        f"- end_date: {plan['end_date']}",
        f"- initial_page1_queries: {plan['initial_page1_queries']}",
        f"- six_year_terms: {len(plan['six_year_specs'])}",
        f"- one_year_terms_after_dedupe: {len(plan['one_year_specs'])}",
        f"- six_year_ranges: {len(plan['six_year_ranges'])}",
        f"- one_year_ranges: {len(plan['one_year_ranges'])}",
        "",
        "| term | field | group | min upload | max upload | page | per page |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in plan["queries"]:
        term = str(row["term"]).replace("|", "\\|")
        lines.append(f"| {term} | {row['search_field']} | {row['slice_group']} | {row['min_upload_date']} | {row['max_upload_date']} | {row['page']} | {row['per_page']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_api_key(*, api_key_env: str, secrets_env: str | None) -> str:
    default_agent_env = Path.home() / ".config" / "agent-env" / "secrets.env"
    load_runtime_secrets_env(secrets_env or default_agent_env)
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is required; set it in the environment or pass --secrets-env")
    return api_key


def _status_counts(db_path: Path) -> dict[str, int]:
    with sqlite3.connect(db_path) as conn:
        return {str(row[0]): int(row[1]) for row in conn.execute("SELECT status, count(*) FROM flickr_work_items GROUP BY status")}


def _db_summary(db_path: Path) -> dict[str, object]:
    with sqlite3.connect(db_path) as conn:
        source_records = int(conn.execute("SELECT count(*) FROM source_records").fetchone()[0])
        unique_records = int(conn.execute("SELECT count(DISTINCT flickr_photo_id) FROM source_records").fetchone()[0])
        image_urls = int(conn.execute("SELECT count(*) FROM source_record_image_urls").fetchone()[0])
        api_rows = conn.execute("SELECT status, count(*) FROM api_call_ledger GROUP BY status").fetchall()
        hit_row = conn.execute("SELECT sum(query_hit_count), sum(duplicate_query_hit_count) FROM source_records").fetchone()
    return {
        "source_records": source_records,
        "unique_flickr_photo_ids": unique_records,
        "image_urls": image_urls,
        "api_call_ledger_status_counts": {str(status): int(count) for status, count in api_rows},
        "query_hit_count": int(hit_row[0] or 0),
        "duplicate_query_hit_count": int(hit_row[1] or 0),
    }


def _slice_rows(db_path: Path) -> list[dict[str, object]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                term,
                search_field,
                min_date,
                max_date,
                count(*) AS work_items,
                sum(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_pages,
                sum(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_pages,
                sum(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending_pages,
                sum(COALESCE(records_returned, 0)) AS records_returned,
                max(COALESCE(response_total, 0)) AS reported_total,
                max(COALESCE(response_pages, 0)) AS reported_pages,
                max(COALESCE(response_perpage, 0)) AS response_perpage
            FROM flickr_work_items
            GROUP BY term, search_field, min_date, max_date
            ORDER BY records_returned DESC, reported_total DESC, term, search_field, min_date
            """
        ).fetchall()
    output: list[dict[str, object]] = []
    for row in rows:
        response_perpage = int(row["response_perpage"] or NORMAL_PAGE_SIZE)
        accessible_pages = max(1, FLICKR_SEARCH_RESULT_WINDOW // response_perpage)
        reported_pages = int(row["reported_pages"] or 0)
        target_pages = min(reported_pages, accessible_pages) if reported_pages else 1
        output.append(
            {
                "term": row["term"],
                "search_field": row["search_field"],
                "min_upload_date": row["min_date"],
                "max_upload_date": row["max_date"],
                "work_items": int(row["work_items"] or 0),
                "completed_pages": int(row["completed_pages"] or 0),
                "failed_pages": int(row["failed_pages"] or 0),
                "pending_pages": int(row["pending_pages"] or 0),
                "records_returned": int(row["records_returned"] or 0),
                "reported_total": int(row["reported_total"] or 0),
                "reported_pages": reported_pages,
                "response_perpage": response_perpage,
                "accessible_pages": accessible_pages,
                "target_pages": target_pages,
                "complete": int(row["completed_pages"] or 0) == target_pages and int(row["failed_pages"] or 0) == 0 and int(row["pending_pages"] or 0) == 0,
                "capped_by_accessible_window": reported_pages > accessible_pages,
            }
        )
    return output


def _write_run_reports(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# {report['run_id']}",
        "",
        f"- status: {report['status']}",
        f"- query_count: {report['plan']['query_count']}",
        f"- api_calls_made: {report['poll_result']['api_calls_made']}",
        f"- status_counts: {report['status_counts']}",
        f"- unique_flickr_photo_ids: {report['db_summary']['unique_flickr_photo_ids']}",
        f"- source_records_with_usable_urls: {report['db_summary']['source_records']}",
        f"- report_json: {json_path}",
        f"- plan_json: {report['artifacts']['plan_json']}",
        "",
        "## Slice Results",
        "",
        "| term | field | upload window | completed pages | target pages | reported total | records returned | capped |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["slice_results"]:
        lines.append(
            "| {term} | {search_field} | {min_upload_date}..{max_upload_date} | {completed_pages} | {target_pages} | {reported_total} | {records_returned} | {capped_by_accessible_window} |".format(
                **row
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log_event(event: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(UTC).isoformat(), **event}
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
