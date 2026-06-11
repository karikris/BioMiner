from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import polars as pl

from biominer.flickr_fetch.query_planner import GEO_PAGE_SIZE, NORMAL_PAGE_SIZE, build_papilio_demoleus_count_probes_from_json
from biominer.flickr_fetch.rate_limiter import DEFAULT_RATE_LIMIT_LEDGER_PATH, FlickrRateLimiter
from biominer.flickr_comments.comment_review import (
    apply_comment_review_decisions_to_parquet,
    build_comment_review_queue_from_parquet,
    review_comments_once,
)
from biominer.flickr_comments.comments_enrichment import CommentsEnrichmentState, fetch_flickr_comments
from biominer.filter.anti_keywords import filter_biodiversity_parquet
from biominer.filter.rules import classify_evidence_frame
from biominer.flickr_fetch.metadata_poller import SOFT_API_CALLS_PER_HOUR, MetadataPollState, poll_once


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biominer")
    parser.add_argument("--version", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    fetch_comments = subparsers.add_parser("fetch-comments")
    fetch_comments.add_argument("--photo-id", action="append", default=[])
    fetch_comments.add_argument("--state-db", default="data/state/flickr_poller.sqlite")
    fetch_comments.add_argument("--limit", type=int, default=0)
    fetch_comments.add_argument("--dry-run", action="store_true")
    fetch_comments.add_argument("--selected-for-qa", action="store_true")
    fetch_comments.add_argument("--api-key-env", default="FLICKR_API_KEY")
    fetch_comments.add_argument("--min-photos", type=int, default=2)
    fetch_comments.add_argument("--min-users", type=int, default=2)
    papilio_plan = subparsers.add_parser("build-papilio-demoleus-query-plan")
    papilio_plan.add_argument("--keywords-json", required=True)
    papilio_plan.add_argument("--state-db", default="data/state/flickr_poller.sqlite")
    build_comment_queue = subparsers.add_parser("build-comment-review-queue")
    build_comment_queue.add_argument("--input", required=True)
    build_comment_queue.add_argument("--state-db", default="data/state/comment_review.sqlite")
    review_comments = subparsers.add_parser("review-comments-once")
    review_comments.add_argument("--state-db", default="data/state/comment_review.sqlite")
    review_comments.add_argument("--max-api-calls", type=int, default=300)
    review_comments.add_argument("--api-key-env", default="FLICKR_API_KEY")
    apply_comment_decisions = subparsers.add_parser("apply-comment-review-decisions")
    apply_comment_decisions.add_argument("--input", required=True)
    apply_comment_decisions.add_argument("--output", required=True)
    apply_comment_decisions.add_argument("--state-db", default="data/state/comment_review.sqlite")
    poll_once_parser = subparsers.add_parser("poll-once")
    poll_once_parser.add_argument("--max-api-calls", type=int, default=SOFT_API_CALLS_PER_HOUR)
    poll_once_parser.add_argument("--workers", type=int, default=1)
    poll_once_parser.add_argument("--stale-claim-seconds", type=int, default=3600)
    poll_once_parser.add_argument("--state-db", default="data/state/flickr_poller.sqlite")
    poll_once_parser.add_argument("--raw-root", default="data/raw")
    poll_once_parser.add_argument("--evidence-output", default="staging/evidence/poll_once_evidence.parquet")
    poll_once_parser.add_argument("--api-key-env", default="FLICKR_API_KEY")
    apply_rules = subparsers.add_parser("apply-rules")
    apply_rules.add_argument("--evidence", required=True)
    apply_rules.add_argument("--output", required=True)
    filter_parser = subparsers.add_parser("filter")
    filter_parser.add_argument("--input", required=True)
    filter_parser.add_argument("--anti-keywords-json", required=True)
    filter_parser.add_argument("--output", required=True)
    filter_parser.add_argument("--dropped-output", required=True)
    gc_cache = subparsers.add_parser("gc-cache")
    gc_cache.add_argument("--cache-root", required=True)
    gc_cache.add_argument("--delete", action="store_true")
    compact_parquet = subparsers.add_parser("compact-parquet")
    compact_parquet.add_argument("--input-root", required=True)
    compact_parquet.add_argument("--output", required=True)
    qa_rate_limit = subparsers.add_parser("qa-rate-limit")
    qa_rate_limit.add_argument("--ledger-path", default=str(DEFAULT_RATE_LIMIT_LEDGER_PATH))
    qa_summary = subparsers.add_parser("qa-summary")
    qa_summary.add_argument("--report", required=True)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.version:
        print("biominer 0.1.0")
        return 0
    if args.command == "fetch-comments":
        state = CommentsEnrichmentState(args.state_db)
        queued = state.queue_candidates(
            (
                {
                    "source": "flickr",
                    "flickr_photo_id": photo_id,
                    "triage_bin": "in_review",
                    "triage_reason": "selected_candidate",
                }
                for photo_id in args.photo_id
            ),
            selected_for_qa=args.selected_for_qa,
        )
        processed = {"comment_records_processed": 0, "comment_records_failed": 0, "term_observations_inserted": 0}
        if args.limit > 0 and not args.dry_run:
            api_key = os.environ.get(args.api_key_env)
            if not api_key:
                print(
                    json.dumps(
                        {"error": f"{args.api_key_env} is required unless --dry-run or --limit 0 is used"},
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 2
            processed = state.process_pending(fetch_comments=fetch_flickr_comments(api_key=api_key), limit=args.limit)
        promoted = state.promote_supported_terms(min_photos=args.min_photos, min_users=args.min_users)
        payload = {
            "implemented": True,
            "comment_fetch_scope": "selected_candidate_records_only",
            "photo_ids_requested": args.photo_id,
            "queued_comment_candidates_added": queued,
            **processed,
            "promoted_terms_added": len(promoted),
            "promoted_terms": [term.__dict__ for term in promoted],
            **state.summary(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "build-papilio-demoleus-query-plan":
        queries = build_papilio_demoleus_count_probes_from_json(args.keywords_json)
        state = MetadataPollState(args.state_db)
        inserted = sum(state.enqueue_work_item(query) for query in queries)
        print(
            json.dumps(
                {
                    "state_db": args.state_db,
                    "keywords_json": args.keywords_json,
                    "count_probes_seen": len(queries),
                    "count_probes_inserted": inserted,
                    "soft_api_calls_per_hour": SOFT_API_CALLS_PER_HOUR,
                    "per_page_for_final_fetches": GEO_PAGE_SIZE,
                    "per_page_for_non_geo_fetches": NORMAL_PAGE_SIZE,
                    "max_result_pages_per_query": 3999,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "build-comment-review-queue":
        payload = build_comment_review_queue_from_parquet(input_path=args.input, state_db=args.state_db)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "review-comments-once":
        try:
            payload = review_comments_once(
                state_db=args.state_db,
                max_api_calls=args.max_api_calls,
                api_key=os.environ.get(args.api_key_env),
            )
        except RuntimeError as exc:
            print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
            return 2
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "apply-comment-review-decisions":
        payload = apply_comment_review_decisions_to_parquet(input_path=args.input, output_path=args.output, state_db=args.state_db)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "poll-once":
        result = poll_once(
            state_db=args.state_db,
            raw_root=args.raw_root,
            evidence_output=args.evidence_output,
            max_api_calls=args.max_api_calls,
            api_key=os.environ.get(args.api_key_env),
            workers=args.workers,
            stale_claim_seconds=args.stale_claim_seconds,
        )
        print(json.dumps({**result.__dict__, "state_db": str(result.state_db)}, indent=2, sort_keys=True))
        return 0
    if args.command == "apply-rules":
        classified = classify_evidence_frame(pl.read_parquet(args.evidence))
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        classified.write_parquet(output_path)
        print(json.dumps(_publication_state_summary(classified, output_path), indent=2, sort_keys=True))
        return 0
    if args.command == "filter":
        payload = filter_biodiversity_parquet(
            input_path=args.input,
            anti_keywords_json=args.anti_keywords_json,
            output_path=args.output,
            dropped_output_path=args.dropped_output,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "gc-cache":
        print(json.dumps(_cache_gc_summary(Path(args.cache_root), delete=args.delete), indent=2, sort_keys=True))
        return 0
    if args.command == "compact-parquet":
        input_paths = sorted(Path(args.input_root).rglob("*.parquet"))
        frame = pl.read_parquet(input_paths) if input_paths else pl.DataFrame()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(output_path)
        print(
            json.dumps(
                {
                    "input_parquet_files": len(input_paths),
                    "output": str(output_path),
                    "rows": frame.height,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "qa-rate-limit":
        limiter = FlickrRateLimiter(args.ledger_path)
        payload = {
            "ledger_path": str(limiter.ledger_path),
            "api_calls_in_window": limiter.api_calls_in_window(),
            "photo_records_in_window": limiter.photo_records_in_window(),
            "soft_api_calls_per_hour": limiter.soft_api_calls_per_hour,
            "hard_api_calls_per_hour": limiter.hard_api_calls_per_hour,
            "hard_photo_records_per_hour": limiter.hard_photo_records_per_hour,
            "window_seconds": limiter.window_seconds,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "qa-summary":
        print(json.dumps(_summarize_report(Path(args.report)), indent=2, sort_keys=True))
        return 0
    return 2


def _summarize_report(report_path: Path) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    storage = report.get("storage_artifacts", {})
    memory = report.get("memory_artifacts", {})
    compute = report.get("compute_artifacts", {})
    return {
        "report": str(report_path),
        "species": report.get("species"),
        "region": report.get("region"),
        "target_record_count": report.get("target_record_count"),
        "actual_unique_records": report.get("actual_unique_records"),
        "api_calls_made": report.get("api_calls_made", report.get("work_items_called")),
        "step_timings_seconds": report.get("step_timings_seconds", {}),
        "total_artifact_bytes": storage.get("total_artifact_bytes"),
        "peak_traced_bytes": memory.get("peak_traced_bytes"),
        "max_rss_kb": memory.get("max_rss_kb"),
        "vision_model_loaded": compute.get("vision_model_loaded"),
    }


def _publication_state_summary(frame: pl.DataFrame, output_path: Path) -> dict[str, object]:
    state_counts = {
        str(row["publication_state"]): int(row["len"])
        for row in frame.group_by("publication_state").len().to_dicts()
    } if frame.height else {}
    in_review_without_reason = 0
    if frame.height and "review_reason" in frame.columns:
        in_review_without_reason = frame.filter(
            (pl.col("publication_state") == "in_review") & (pl.col("review_reason").list.len() == 0)
        ).height
    return {
        "output": str(output_path),
        "rows": frame.height,
        "publication_state_counts": state_counts,
        "in_review_without_reason": in_review_without_reason,
    }


def _cache_gc_summary(cache_root: Path, *, delete: bool) -> dict[str, object]:
    files = [path for path in cache_root.rglob("*") if path.is_file()] if cache_root.exists() else []
    deleted = 0
    if delete:
        for path in files:
            path.unlink()
            deleted += 1
    return {
        "cache_root": str(cache_root),
        "files_seen": len(files),
        "bytes_seen": sum(path.stat().st_size for path in files if path.exists()),
        "deleted_files": deleted,
    }


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
