from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path

from biominer.flickr_fetch.metadata_poller import MetadataPollState, poll_once
from biominer.flickr_fetch.query_planner import FlickrQuery, GEO_PAGE_SIZE, SearchField
from biominer.reports.flickr_fetch import (
    build_step1_fetch_report,
    current_git_sha,
    write_step1_fetch_report,
    write_step1_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded Flickr text or tag search page fetch.")
    parser.add_argument("--term", required=True)
    parser.add_argument("--search-field", choices=("text", "tags"), default="text")
    parser.add_argument("--pages", type=int, required=True)
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
        "--pages",
        str(args.pages),
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
    ]
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
        expected_pages=args.pages,
        status="running",
        started_at=started.isoformat(),
        git_sha=git_sha,
    )
    state = MetadataPollState(args.state_db)
    inserted = _enqueue_pages(state, term=args.term, pages=args.pages, search_field=args.search_field)
    print(json.dumps({"event": "work_enqueued", "inserted": inserted, "pages": args.pages, "search_field": args.search_field}, sort_keys=True), flush=True)
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
        expected_pages=args.pages,
        status="completed",
        git_sha=git_sha,
    )
    write_step1_fetch_report(args.report, report)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest["status"] = "completed"
    manifest["end_time"] = ended.isoformat()
    Path(args.manifest).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "run_completed", **result.__dict__, "state_db": str(result.state_db)}, default=str, sort_keys=True), flush=True)


def _enqueue_pages(state: MetadataPollState, *, term: str, pages: int, search_field: SearchField) -> int:
    return sum(
        state.enqueue_work_item(
            FlickrQuery(
                term=term,
                language="en",
                search_field=search_field,
                lane="normal_page",
                page=page,
                per_page=GEO_PAGE_SIZE,
                has_geo=1,
            )
        )
        for page in range(1, pages + 1)
    )


if __name__ == "__main__":
    main()
