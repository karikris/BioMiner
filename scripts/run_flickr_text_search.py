from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path

from biominer.flickr_fetch.metadata_poller import MetadataPollState, poll_once
from biominer.flickr_fetch.query_planner import (
    COUNT_PROBE_PAGE_SIZE,
    DEFAULT_COARSE_SLICE_DAYS,
    DEFAULT_COARSE_SLICE_END_DATE,
    DEFAULT_FIXED_SLICE_DAYS,
    DEFAULT_FIXED_SLICE_END_DATE,
    DEFAULT_FIXED_SLICE_PAGES,
    DEFAULT_FIXED_SLICE_START_DATE,
    GEO_PAGE_SIZE,
    NORMAL_PAGE_SIZE,
    STABLE_RESULT_THRESHOLD,
    FlickrQuery,
    SearchField,
    plan_fixed_upload_slice_pages,
)
from biominer.reports.flickr_fetch import (
    build_step1_fetch_report,
    current_git_sha,
    write_step1_fetch_report,
    write_step1_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded Flickr text or tag search over fixed upload-date slices.")
    parser.add_argument("--term", required=True)
    parser.add_argument("--search-field", choices=("text", "tags"), default="text")
    parser.add_argument("--pages", type=int)
    parser.add_argument("--allow-direct-pages", action="store_true")
    parser.add_argument("--unsafe-direct-pages", action="store_true")
    parser.add_argument("--state-db", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--evidence-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--max-api-calls", type=int, default=3500)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--api-key-env", default="FLICKR_API_KEY")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start-date", default=DEFAULT_FIXED_SLICE_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_FIXED_SLICE_END_DATE)
    parser.add_argument("--slice-days", type=int, default=DEFAULT_FIXED_SLICE_DAYS)
    parser.add_argument("--coarse-end-date", default=DEFAULT_COARSE_SLICE_END_DATE)
    parser.add_argument("--coarse-slice-days", type=int, default=DEFAULT_COARSE_SLICE_DAYS)
    parser.add_argument("--pages-per-slice", type=int, default=DEFAULT_FIXED_SLICE_PAGES)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is required")
    command = [
        "scripts/run_flickr_text_search.py",
        "--term",
        args.term,
        "--search-field",
        args.search_field,
        "--state-db",
        args.state_db,
        "--raw-root",
        args.raw_root,
        "--evidence-output",
        args.evidence_output,
        "--manifest",
        args.manifest,
        "--report",
        args.report,
        "--log-path",
        args.log_path,
        "--max-api-calls",
        str(args.max_api_calls),
        "--workers",
        str(args.workers),
        "--run-id",
        args.run_id,
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--slice-days",
        str(args.slice_days),
        "--pages-per-slice",
        str(args.pages_per_slice),
    ]
    if args.coarse_end_date:
        command.extend(["--coarse-end-date", args.coarse_end_date])
    if args.coarse_slice_days:
        command.extend(["--coarse-slice-days", str(args.coarse_slice_days)])
    if args.pages is not None:
        command.extend(["--pages", str(args.pages)])
    if args.allow_direct_pages:
        command.append("--allow-direct-pages")
    if args.unsafe_direct_pages:
        command.append("--unsafe-direct-pages")
    expected_outputs = {
        "state_db": args.state_db,
        "raw_root": args.raw_root,
        "evidence_output": args.evidence_output,
        "manifest": args.manifest,
        "report": args.report,
        "log_path": args.log_path,
    }
    started = datetime.now(UTC)
    git_sha = current_git_sha()
    write_step1_manifest(
        args.manifest,
        run_id=args.run_id,
        command=command,
        expected_outputs=expected_outputs,
        expected_pages=args.pages or 1,
        status="running",
        started_at=started.isoformat(),
        git_sha=git_sha,
    )
    state = MetadataPollState(args.state_db)
    if args.allow_direct_pages:
        if args.pages is None:
            raise ValueError("--pages is required with --allow-direct-pages")
        inserted = _enqueue_direct_pages(
            state,
            term=args.term,
            pages=args.pages,
            search_field=args.search_field,
            has_geo=1,
            unsafe=args.unsafe_direct_pages,
        )
        event = {"event": "work_enqueued", "inserted": inserted, "pages": args.pages, "search_field": args.search_field, "mode": "direct_pages"}
    else:
        inserted = _enqueue_fixed_upload_slice_pages(
            state,
            term=args.term,
            search_field=args.search_field,
            start_date=args.start_date,
            end_date=args.end_date,
            slice_days=args.slice_days,
            coarse_end_date=args.coarse_end_date,
            coarse_slice_days=args.coarse_slice_days,
            pages_per_slice=args.pages_per_slice,
        )
        event = {
            "event": "work_enqueued",
            "inserted": inserted,
            "search_field": args.search_field,
            "mode": "fixed_upload_slices",
            "start_date": args.start_date,
            "end_date": args.end_date,
            "pages_per_slice": args.pages_per_slice,
        }
    print(json.dumps(event, sort_keys=True), flush=True)
    result = poll_once(
        state_db=args.state_db,
        raw_root=args.raw_root,
        evidence_output=args.evidence_output,
        max_api_calls=args.max_api_calls,
        api_key=api_key,
        workers=args.workers,
    )
    ended = datetime.now(UTC)
    report = build_step1_fetch_report(
        run_id=args.run_id,
        command=command,
        result=result,
        raw_root=args.raw_root,
        evidence_output=args.evidence_output,
        query_provenance=state.source_records_with_query_provenance(),
        started_at=started,
        ended_at=ended,
        workers=args.workers,
        expected_pages=args.pages or 1,
        status="completed",
        git_sha=git_sha,
    )
    write_step1_fetch_report(args.report, report)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest["status"] = "completed"
    manifest["end_time"] = ended.isoformat()
    Path(args.manifest).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "run_completed", **result.__dict__, "state_db": str(result.state_db)}, default=str, sort_keys=True), flush=True)


def _enqueue_count_probe(state: MetadataPollState, *, term: str, search_field: SearchField) -> int:
    return state.enqueue_work_item(
        FlickrQuery(
            term=term,
            language="en",
            search_field=search_field,
            lane="count_probe",
            page=1,
            per_page=COUNT_PROBE_PAGE_SIZE,
            has_geo=0,
        )
    )


def _enqueue_fixed_upload_slice_pages(
    state: MetadataPollState,
    *,
    term: str,
    search_field: SearchField,
    start_date: str,
    end_date: str,
    slice_days: int,
    coarse_end_date: str | None,
    coarse_slice_days: int | None,
    pages_per_slice: int,
) -> int:
    return sum(
        state.enqueue_work_item(query)
        for query in plan_fixed_upload_slice_pages(
            term=term,
            search_field=search_field,
            start_date=start_date,
            end_date=end_date,
            slice_days=slice_days,
            coarse_end_date=coarse_end_date,
            coarse_slice_days=coarse_slice_days,
            pages_per_slice=pages_per_slice,
        )
    )


def _enqueue_direct_pages(state: MetadataPollState, *, term: str, pages: int, search_field: SearchField, has_geo: int, unsafe: bool) -> int:
    per_page = GEO_PAGE_SIZE if has_geo else NORMAL_PAGE_SIZE
    if not unsafe and pages * per_page > STABLE_RESULT_THRESHOLD:
        raise ValueError(f"Direct page mode is limited to {STABLE_RESULT_THRESHOLD} estimated records; use count-probe mode for broader searches.")
    return sum(
        state.enqueue_work_item(
            FlickrQuery(
                term=term,
                language="en",
                search_field=search_field,
                lane="normal_page",
                page=page,
                per_page=per_page,
                has_geo=has_geo,
            )
        )
        for page in range(1, pages + 1)
    )


if __name__ == "__main__":
    main()
