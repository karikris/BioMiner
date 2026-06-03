from __future__ import annotations

import argparse
import json
from pathlib import Path

from flickr_bio_occurrence.benchmark.offline_run import run_existing_payload_benchmark
from flickr_bio_occurrence.flickr.rate_limiter import DEFAULT_RATE_LIMIT_LEDGER_PATH, FlickrRateLimiter
from flickr_bio_occurrence.pipeline.dry_run import build_dry_run_summary


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
    qa_rate_limit = subparsers.add_parser("qa-rate-limit")
    qa_rate_limit.add_argument("--ledger-path", default=str(DEFAULT_RATE_LIMIT_LEDGER_PATH))
    qa_summary = subparsers.add_parser("qa-summary")
    qa_summary.add_argument("--report", required=True)
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
    if args.command == "fetch" and args.dry_run:
        summary = build_dry_run_summary(
            species=args.species,
            region=args.region,
            year=args.year,
            month=args.month,
            config_path="config/pipeline.toml",
            model_registry_path="config/model_registry.toml",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
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


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
