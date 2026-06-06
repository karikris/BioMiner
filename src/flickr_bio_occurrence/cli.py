from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import polars as pl

from flickr_bio_occurrence.benchmark.estimates import estimate_combined_production, estimate_production_from_report
from flickr_bio_occurrence.benchmark.offline_run import run_existing_payload_benchmark
from flickr_bio_occurrence.evidence.extractor import write_staging_evidence
from flickr_bio_occurrence.evidence.rules import classify_evidence_frame
from flickr_bio_occurrence.flickr.rate_limiter import DEFAULT_RATE_LIMIT_LEDGER_PATH, FlickrRateLimiter
from flickr_bio_occurrence.pipeline.comments_enrichment import CommentsEnrichmentState, fetch_flickr_comments
from flickr_bio_occurrence.pipeline.job_queue import ClassificationJobQueue
from flickr_bio_occurrence.pipeline.dry_run import build_dry_run_summary
from flickr_bio_occurrence.pipeline.metadata_poller import SOFT_API_CALLS_PER_HOUR, poll_once
from flickr_bio_occurrence.vision.service import BioClipJobService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flickr-bio")
    parser.add_argument("--version", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("--species", required=True)
    fetch.add_argument("--region", required=True)
    fetch.add_argument("--year", type=int, required=True)
    fetch.add_argument("--month", type=int, required=True)
    fetch.add_argument("--dry-run", action="store_true")
    fetch_live = subparsers.add_parser("fetch-live")
    fetch_live.add_argument("--species", required=True)
    fetch_live.add_argument("--region", required=True)
    fetch_live.add_argument("--year", type=int, required=True)
    fetch_live.add_argument("--month", type=int, required=True)
    fetch_live.add_argument("--dry-run", action="store_true")
    fetch_comments = subparsers.add_parser("fetch-comments")
    fetch_comments.add_argument("--photo-id", action="append", default=[])
    fetch_comments.add_argument("--state-db", default="data/state/flickr_poller.sqlite")
    fetch_comments.add_argument("--limit", type=int, default=0)
    fetch_comments.add_argument("--dry-run", action="store_true")
    fetch_comments.add_argument("--selected-for-qa", action="store_true")
    fetch_comments.add_argument("--api-key-env", default="FLICKR_API_KEY")
    fetch_comments.add_argument("--min-photos", type=int, default=2)
    fetch_comments.add_argument("--min-users", type=int, default=2)
    poll_once_parser = subparsers.add_parser("poll-once")
    poll_once_parser.add_argument("--max-api-calls", type=int, default=SOFT_API_CALLS_PER_HOUR)
    poll_once_parser.add_argument("--state-db", default="data/state/flickr_poller.sqlite")
    poll_once_parser.add_argument("--raw-root", default="data/raw")
    poll_once_parser.add_argument("--evidence-output", default="staging/evidence/poll_once_evidence.parquet")
    poll_once_parser.add_argument("--api-key-env", default="FLICKR_API_KEY")
    build_evidence = subparsers.add_parser("build-evidence")
    build_evidence.add_argument("--raw-root", required=True)
    build_evidence.add_argument("--output", required=True)
    build_evidence.add_argument("--species", default="Papilio demoleus")
    build_evidence.add_argument("--queue-path")
    build_evidence.add_argument("--model-version", default="bioclip2_5_huge")
    classify_once = subparsers.add_parser("classify-once")
    classify_once.add_argument("--queue-path", required=True)
    classify_once.add_argument("--prediction-output-dir", required=True)
    classify_once.add_argument("--fake-classifier", action="store_true")
    classify_watch = subparsers.add_parser("classify-watch")
    classify_watch.add_argument("--queue-path", required=True)
    classify_watch.add_argument("--prediction-output-dir", required=True)
    classify_watch.add_argument("--limit", type=int, default=1)
    classify_watch.add_argument("--fake-classifier", action="store_true")
    apply_rules = subparsers.add_parser("apply-rules")
    apply_rules.add_argument("--evidence", required=True)
    apply_rules.add_argument("--output", required=True)
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
    qa_estimate = subparsers.add_parser("qa-estimate")
    qa_estimate.add_argument("--report", required=True)
    qa_estimate.add_argument("--target-records", type=int, default=3200)
    qa_estimate.add_argument("--api-call-target", type=int, default=3200)
    qa_estimate.add_argument("--soft-api-calls-per-hour", type=int, default=3200)
    qa_estimate_combined = subparsers.add_parser("qa-estimate-combined")
    qa_estimate_combined.add_argument("--metadata-report", required=True)
    qa_estimate_combined.add_argument("--vision-report", required=True)
    qa_estimate_combined.add_argument("--target-records", type=int, default=3200)
    qa_estimate_combined.add_argument("--api-call-target", type=int, default=3200)
    qa_estimate_combined.add_argument("--soft-api-calls-per-hour", type=int, default=3200)
    offline = subparsers.add_parser("benchmark-existing-payloads")
    offline.add_argument("--raw-root", required=True)
    offline.add_argument("--output-dir", required=True)
    offline.add_argument("--species", default="Papilio demoleus")
    offline.add_argument("--region-id", default="AU_ALL")
    offline.add_argument("--target-records", type=int, default=1000)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.version:
        print("flickr-bio-occurrence 0.1.0")
        return 0
    if args.command in {"fetch", "fetch-live"} and args.dry_run:
        summary = build_dry_run_summary(
            species=args.species,
            region=args.region,
            year=args.year,
            month=args.month,
            config_path="config/pipeline.toml",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
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
    if args.command == "poll-once":
        result = poll_once(
            state_db=args.state_db,
            raw_root=args.raw_root,
            evidence_output=args.evidence_output,
            max_api_calls=args.max_api_calls,
            api_key=os.environ.get(args.api_key_env),
        )
        print(json.dumps({**result.__dict__, "state_db": str(result.state_db)}, indent=2, sort_keys=True))
        return 0
    if args.command == "build-evidence":
        payloads = _read_json_payloads(Path(args.raw_root))
        output_path = write_staging_evidence(payloads, species_query=args.species, output_path=args.output)
        evidence = pl.read_parquet(output_path)
        enqueued_job = None
        if args.queue_path:
            queue = ClassificationJobQueue(args.queue_path)
            enqueued_job = queue.enqueue_evidence_shard(output_path, model_version=args.model_version).job_id
        print(
            json.dumps(
                {
                    "evidence_parquet_path": str(output_path),
                    "evidence_rows": evidence.height,
                    "classification_job_id": enqueued_job,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command in {"classify-once", "classify-watch"}:
        if not args.fake_classifier:
            print(json.dumps({"error": "classify CLI requires --fake-classifier in this runtime"}, indent=2, sort_keys=True))
            return 2
        queue = ClassificationJobQueue(args.queue_path)
        service = BioClipJobService(
            queue=queue,
            classifier=_FakeEvidenceClassifier(),
            prediction_output_dir=args.prediction_output_dir,
        )
        if args.command == "classify-once":
            result = service.process_next_job()
            processed = [] if result is None else [result]
        else:
            processed = service.process_pending_jobs(limit=args.limit)
        print(
            json.dumps(
                {
                    "processed_jobs": len(processed),
                    "prediction_rows": sum(item.prediction_rows for item in processed),
                    "prediction_parquet_paths": [str(path) for item in processed for path in item.prediction_parquet_paths],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "apply-rules":
        classified = classify_evidence_frame(pl.read_parquet(args.evidence))
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        classified.write_parquet(output_path)
        print(json.dumps(_publication_state_summary(classified, output_path), indent=2, sort_keys=True))
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
    if args.command == "qa-estimate":
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        estimate = estimate_production_from_report(
            report,
            target_records_with_images=args.target_records,
            api_call_target=args.api_call_target,
            soft_api_calls_per_hour=args.soft_api_calls_per_hour,
        )
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return 0
    if args.command == "qa-estimate-combined":
        metadata_report = json.loads(Path(args.metadata_report).read_text(encoding="utf-8"))
        vision_report = json.loads(Path(args.vision_report).read_text(encoding="utf-8"))
        estimate = estimate_combined_production(
            metadata_report,
            vision_report,
            target_records_with_images=args.target_records,
            api_call_target=args.api_call_target,
            soft_api_calls_per_hour=args.soft_api_calls_per_hour,
        )
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return 0
    if args.command == "benchmark-existing-payloads":
        payload_paths = sorted(Path(args.raw_root).rglob("*.json"))
        report_path = run_existing_payload_benchmark(
            payload_paths=payload_paths,
            output_dir=args.output_dir,
            species_name=args.species,
            region_id=args.region_id,
            target_records=args.target_records,
        )
        summary = _summarize_report(report_path)
        summary["raw_payload_files"] = len(payload_paths)
        print(json.dumps(summary, indent=2, sort_keys=True))
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


class _FakeEvidenceClassifier:
    def classify_evidence_rows(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        return [self(row) for row in rows]

    def __call__(self, row: dict[str, object]) -> dict[str, object]:
        return {
            "flickr_photo_id": row["flickr_photo_id"],
            "model_version": "fake_bioclip",
            "model_checkpoint": "fake",
            "image_hash": f"sha256:{row['flickr_photo_id']}",
            "image_url_used": row.get("image_url"),
            "top1_label": "a photo of Papilio demoleus",
            "top1_score": 0.9,
            "species_agreement_status": "exact_species_agreement",
        }


def _read_json_payloads(raw_root: Path) -> list[dict[str, object]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(raw_root.rglob("*.json"))]


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
