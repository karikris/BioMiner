from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from biominer.flickr_fetch.metadata_poller import SOFT_API_CALLS_PER_HOUR, MetadataPollState, poll_once
from biominer.flickr_fetch.query_planner import SearchField, plan_fixed_upload_slice_pages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch Flickr metadata for a keyword over fixed upload-date chunks.")
    field = parser.add_mutually_exclusive_group(required=True)
    field.add_argument("--text", dest="search_field", action="store_const", const="text")
    field.add_argument("--tag", "--tags", dest="search_field", action="store_const", const="tags")
    parser.add_argument("--keyword", required=True)
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2024-11-25")
    parser.add_argument("--state-db", default="data/state/flicker_miner.sqlite")
    parser.add_argument("--raw-root", default="data/raw/flicker_miner")
    parser.add_argument("--evidence-output", default="staging/evidence/flicker_miner.parquet")
    parser.add_argument("--max-api-calls", type=int, default=SOFT_API_CALLS_PER_HOUR)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--api-key-env", default="FLICKR_API_KEY")
    parser.add_argument("--seed-only", action="store_true")
    return parser


def seed_time_chunks(
    *,
    state_db: str | Path,
    keyword: str,
    search_field: SearchField,
    days: int,
    start_date: str,
    end_date: str,
) -> int:
    state = MetadataPollState(state_db)
    return sum(
        state.enqueue_work_item(query)
        for query in plan_fixed_upload_slice_pages(
            term=keyword,
            search_field=search_field,
            start_date=start_date,
            end_date=end_date,
            slice_days=days,
            coarse_end_date=None,
            coarse_slice_days=None,
        )
    )


def main() -> int:
    args = build_parser().parse_args()
    inserted = seed_time_chunks(
        state_db=args.state_db,
        keyword=args.keyword,
        search_field=args.search_field,
        days=args.days,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if args.seed_only:
        print(json.dumps({"seeded": inserted, "state_db": args.state_db}, indent=2, sort_keys=True))
        return 0
    result = poll_once(
        state_db=args.state_db,
        raw_root=args.raw_root,
        evidence_output=args.evidence_output,
        max_api_calls=args.max_api_calls,
        api_key=os.environ.get(args.api_key_env),
        workers=args.workers,
    )
    print(json.dumps({"seeded": inserted, **result.__dict__, "state_db": str(result.state_db)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
