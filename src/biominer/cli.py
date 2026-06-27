from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import subprocess

import polars as pl

from biominer.bioclip.benchmark import build_benchmark_report, parse_csv_candidate_modes, parse_csv_ints
from biominer.bioclip.bioclip import BioClipClassifier, PersistentBioClipScorer
from biominer.bioclip.candidate_sets import CandidateMode, CandidateStrategy
from biominer.bioclip.model_registry import BioClipRuntime, ModelConfig
from biominer.bioclip.register_runner import process_records_with_registers
from biominer.bioclip.species_candidates import DEFAULT_SPECIES_CANDIDATE_LIMIT, load_species_candidates
from biominer.flickr_fetch.query_planner import (
    FLICKR_SEARCH_RESULT_WINDOW,
    GEO_PAGE_SIZE,
    NORMAL_PAGE_SIZE,
    STABLE_RESULT_THRESHOLD,
    build_papilio_demoleus_count_probes_from_json,
    load_registry_flickr_queries,
)
from biominer.flickr_comments.comment_review import (
    apply_comment_review_decisions_to_parquet,
    build_comment_review_queue_from_parquet,
    review_comments_once,
)
from biominer.flickr_comments.comments_enrichment import CommentsEnrichmentState, fetch_flickr_comments
from biominer.filter.anti_keywords import filter_biodiversity_parquet
from biominer.filter.rules import classify_evidence_frame
from biominer.flickr_fetch.metadata_poller import SOFT_API_CALLS_PER_HOUR, MetadataPollState, poll_once
from biominer.registry.audit import audit_registry
from biominer.registry.build import build_registry
from biominer.registry.compiler import compile_registry_fixture
from biominer.registry.enrichment import INATURALIST_DAILY_REQUEST_LIMIT, build_enrichment_sources_from_registry, compile_enriched_registry
from biominer.registry.gbif import GBIFClient
from biominer.registry.gbif_source import build_gbif_source_snapshot
from biominer.registry.scope import load_scope
from biominer.reports.buckets import export_bucket_views
from biominer.reports.name_evidence import build_name_evidence_report, write_name_evidence_report


BIOCLIP_25_HUGE_REPO_ID = "imageomics/bioclip-2.5-vith14"
BIOCLIP_25_HUGE_REVISION = "191d741545e4c741cdef4b22c6eb69c945c1e592"
BIOCLIP_RUNTIME_PYTHON = ".venv-bioclip-py312/bin/python"
BIOCLIP_HF_CACHE_DIR = "data/cache/huggingface"
BIOCLIP_PREFETCH_ALLOW_PATTERNS = (
    "open_clip_config.json",
    "open_clip_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
BIOCLIP_PREFETCH_IGNORE_PATTERNS = (
    "*.bin",
    "*.pt",
    "pytorch_model*",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biominer")
    parser.add_argument("--version", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    bioclip = subparsers.add_parser("bioclip")
    bioclip_subparsers = bioclip.add_subparsers(dest="bioclip_command")
    bioclip_runtime = bioclip_subparsers.add_parser("runtime-check")
    bioclip_runtime.add_argument("--runtime-python", default=BIOCLIP_RUNTIME_PYTHON)
    bioclip_runtime.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    bioclip_runtime.add_argument("--hf-cache-dir", default=BIOCLIP_HF_CACHE_DIR)
    bioclip_prefetch = bioclip_subparsers.add_parser("prefetch-model")
    bioclip_prefetch.add_argument("--runtime-python", default=BIOCLIP_RUNTIME_PYTHON)
    bioclip_prefetch.add_argument("--hf-cache-dir", default=BIOCLIP_HF_CACHE_DIR)
    bioclip_prefetch.add_argument("--model-name", default=BIOCLIP_25_HUGE_REPO_ID)
    bioclip_prefetch.add_argument("--revision", default=BIOCLIP_25_HUGE_REVISION)
    bioclip_prefetch.add_argument("--max-workers", type=int, default=8)
    bioclip_screen = bioclip_subparsers.add_parser("screen")
    bioclip_screen.add_argument("--input", required=True)
    bioclip_screen.add_argument("--species-candidates", required=True)
    bioclip_screen.add_argument("--output", required=True)
    bioclip_screen.add_argument("--runtime-python", default=BIOCLIP_RUNTIME_PYTHON)
    bioclip_screen.add_argument("--hf-cache-dir", default=BIOCLIP_HF_CACHE_DIR)
    bioclip_screen.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    bioclip_screen.add_argument("--cache-root", default="data/cache/images")
    bioclip_screen.add_argument("--register-count", type=int, default=2)
    bioclip_screen.add_argument("--register-size", type=int, default=4)
    bioclip_screen.add_argument("--download-workers", type=int, default=4)
    bioclip_screen.add_argument("--candidate-limit", type=int, default=DEFAULT_SPECIES_CANDIDATE_LIMIT)
    bioclip_screen.add_argument("--candidate-mode", choices=[mode.value for mode in CandidateMode], default=CandidateMode.HYBRID.value)
    bioclip_screen.add_argument("--candidate-strategy", choices=[strategy.value for strategy in CandidateStrategy], default=CandidateStrategy.ALL.value)
    bioclip_screen.add_argument("--target-species")
    bioclip_screen.add_argument("--emit-image-embeddings", action="store_true")
    bioclip_screen.add_argument("--embedding-output")
    bioclip_screen.add_argument("--bucket-views-dir")
    bioclip_benchmark = bioclip_subparsers.add_parser("benchmark")
    bioclip_benchmark.add_argument("--input", required=True)
    bioclip_benchmark.add_argument("--species-candidates", required=True)
    bioclip_benchmark.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    bioclip_benchmark.add_argument("--register-sizes", default="8,16,32,64")
    bioclip_benchmark.add_argument("--register-counts", default="2,4,6")
    bioclip_benchmark.add_argument("--candidate-limits", default="500,2000,18000")
    bioclip_benchmark.add_argument("--classification-modes", default="triage,hybrid,rescue_full_species")
    bioclip_benchmark.add_argument("--candidate-strategy", choices=[strategy.value for strategy in CandidateStrategy], default=CandidateStrategy.ALL.value)
    bioclip_benchmark.add_argument("--download-workers", type=int, default=4)
    bioclip_benchmark.add_argument("--output", required=True)
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
    registry = subparsers.add_parser("registry")
    registry_subparsers = registry.add_subparsers(dest="registry_command")
    registry_compile = registry_subparsers.add_parser("compile-fixture")
    registry_compile.add_argument("--source-json", required=True)
    registry_compile.add_argument("--output-dir", required=True)
    registry_compile.add_argument("--registry-version", required=True)
    registry_compile.add_argument("--scope-json", default="config/butterfly_scope.json")
    registry_compile_enriched = registry_subparsers.add_parser("compile-enriched")
    registry_compile_enriched.add_argument("--registry-dir", required=True)
    registry_compile_enriched.add_argument("--registry-version", required=True)
    registry_compile_enriched.add_argument("--scope-json", default="config/butterfly_scope.json")
    registry_fetch_taxonomy = registry_subparsers.add_parser("fetch-taxonomy")
    registry_fetch_taxonomy.add_argument("--output-json", required=True)
    registry_fetch_taxonomy.add_argument("--scope-json", default="config/butterfly_scope.json")
    registry_fetch_taxonomy.add_argument("--retrieved-at")
    registry_enrich_sources = registry_subparsers.add_parser("enrich-sources")
    registry_enrich_sources.add_argument("--registry-dir", required=True)
    registry_enrich_sources.add_argument("--sources", default="col,inaturalist,tmd_de,itis")
    registry_enrich_sources.add_argument("--workers", type=int, default=8)
    registry_enrich_sources.add_argument("--progress-every", type=int, default=100)
    registry_enrich_sources.add_argument("--checkpoint-every", type=int, default=500)
    registry_enrich_sources.add_argument("--max-retries", type=int, default=5)
    registry_enrich_sources.add_argument("--inaturalist-daily-request-limit", type=int, default=INATURALIST_DAILY_REQUEST_LIMIT)
    registry_enrich_sources.add_argument("--limit", type=int, default=0)
    registry_enrich_sources.add_argument("--report-dir", default="reports")
    registry_build = registry_subparsers.add_parser("build")
    registry_build.add_argument("--output-dir", required=True)
    registry_build.add_argument("--registry-version", required=True)
    registry_build.add_argument("--scope-json", default="config/butterfly_scope.json")
    registry_build.add_argument("--source-json")
    registry_build.add_argument("--reuse-source-json", action="store_true")
    registry_build.add_argument("--report-dir", default="reports")
    registry_build.add_argument("--retrieved-at")
    registry_build.add_argument("--workers", type=int, default=8)
    registry_build.add_argument("--progress-every", type=int, default=100)
    registry_build.add_argument("--checkpoint-every", type=int, default=500)
    registry_build.add_argument("--max-retries", type=int, default=5)
    registry_build.add_argument("--enrichment-sources", default="col,inaturalist,tmd_de,itis")
    registry_build.add_argument("--inaturalist-daily-request-limit", type=int, default=INATURALIST_DAILY_REQUEST_LIMIT)
    registry_build.add_argument("--skip-enrichment", action="store_true")
    registry_seed = registry_subparsers.add_parser("seed-flickr-queries")
    registry_seed.add_argument("--query-definitions", required=True)
    registry_seed.add_argument("--state-db", default="data/state/flickr_poller.sqlite")
    registry_seed.add_argument("--start-date", default="2004-02-10")
    registry_seed.add_argument("--end-date", default=datetime.now(UTC).date().isoformat())
    registry_seed.add_argument("--slice-days", type=int, default=5)
    registry_audit = registry_subparsers.add_parser("audit")
    registry_audit.add_argument("--registry-dir", required=True)
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
    qa_rate_limit.add_argument("--state-db", default="data/state/flickr_poller.sqlite")
    qa_rate_limit.add_argument("--ledger-path", dest="state_db")
    qa_summary = subparsers.add_parser("qa-summary")
    qa_summary.add_argument("--report", required=True)
    export_views = subparsers.add_parser("export-bucket-views")
    export_views.add_argument("--input", required=True)
    export_views.add_argument("--output-dir", required=True)
    name_evidence = subparsers.add_parser("report-name-evidence")
    name_evidence.add_argument("--metadata-output", required=True)
    name_evidence.add_argument("--bioclip-output", required=True)
    name_evidence.add_argument("--keywords-json", required=True)
    name_evidence.add_argument("--target-species", required=True)
    name_evidence.add_argument("--score-threshold", type=float, default=0.9)
    name_evidence.add_argument("--output", required=True)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.version:
        print("biominer 0.1.0")
        return 0
    if args.command == "bioclip":
        if args.bioclip_command == "runtime-check":
            return _run_bioclip_runtime_check(args)
        if args.bioclip_command == "prefetch-model":
            return _run_bioclip_prefetch_model(args)
        if args.bioclip_command == "screen":
            return _run_bioclip_screen(args)
        if args.bioclip_command == "benchmark":
            return _run_bioclip_benchmark(args)
        return 2
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
                    "flickr_search_result_window": FLICKR_SEARCH_RESULT_WINDOW,
                    "stable_result_threshold": STABLE_RESULT_THRESHOLD,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "registry":
        if args.registry_command == "fetch-taxonomy":
            retrieved_at = args.retrieved_at or datetime.now(UTC).isoformat()
            snapshot = build_gbif_source_snapshot(
                GBIFClient(),
                load_scope(args.scope_json),
                retrieved_at=retrieved_at,
            )
            output = Path(args.output_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "output_json": str(output),
                        "source": snapshot.get("source"),
                        "taxa_rows": len(snapshot.get("taxa", [])),
                        "name_rows": len(snapshot.get("names", [])),
                        "source_assertion_rows": len(snapshot.get("source_assertions", [])),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.registry_command == "compile-fixture":
            payload = compile_registry_fixture(
                args.source_json,
                args.output_dir,
                registry_version=args.registry_version,
                scope_path=args.scope_json,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.registry_command == "compile-enriched":
            payload = compile_enriched_registry(
                registry_dir=args.registry_dir,
                registry_version=args.registry_version,
                scope_path=args.scope_json,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.registry_command == "enrich-sources":
            logging.basicConfig(
                level=getattr(logging, os.environ.get("BIOMINER_LOG_LEVEL", "INFO").upper(), logging.INFO),
                format="%(asctime)s %(levelname)s %(name)s %(message)s",
                force=True,
            )
            payload = build_enrichment_sources_from_registry(
                registry_dir=args.registry_dir,
                sources=tuple(part.strip() for part in args.sources.split(",") if part.strip()),
                workers=args.workers,
                progress_every=args.progress_every,
                checkpoint_every=args.checkpoint_every,
                max_retries=args.max_retries,
                inaturalist_daily_request_limit=args.inaturalist_daily_request_limit,
                limit=args.limit,
                report_dir=args.report_dir,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.registry_command == "build":
            logging.basicConfig(
                level=getattr(logging, os.environ.get("BIOMINER_LOG_LEVEL", "INFO").upper(), logging.INFO),
                format="%(asctime)s %(levelname)s %(name)s %(message)s",
                force=True,
            )
            try:
                payload = build_registry(
                    output_dir=args.output_dir,
                    registry_version=args.registry_version,
                    scope_path=args.scope_json,
                    source_json=args.source_json,
                    reuse_source_json=args.reuse_source_json,
                    report_dir=args.report_dir,
                    retrieved_at=args.retrieved_at,
                    workers=args.workers,
                    progress_every=args.progress_every,
                    checkpoint_every=args.checkpoint_every,
                    max_retries=args.max_retries,
                    enrichment_sources=tuple(part.strip() for part in args.enrichment_sources.split(",") if part.strip()),
                    inaturalist_daily_request_limit=args.inaturalist_daily_request_limit,
                    skip_enrichment=args.skip_enrichment,
                )
            except FileNotFoundError as exc:
                print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
                return 2
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.registry_command == "seed-flickr-queries":
            queries = load_registry_flickr_queries(
                args.query_definitions,
                start_date=args.start_date,
                end_date=args.end_date,
                slice_days=args.slice_days,
            )
            state = MetadataPollState(args.state_db)
            inserted = sum(state.enqueue_work_item(query) for query in queries)
            print(
                json.dumps(
                    {
                        "query_definitions": args.query_definitions,
                        "state_db": args.state_db,
                        "work_items_seen": len(queries),
                        "work_items_inserted": inserted,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.registry_command == "audit":
            print(json.dumps(audit_registry(args.registry_dir), indent=2, sort_keys=True))
            return 0
        return 2
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
        print(json.dumps(MetadataPollState(args.state_db).api_budget_summary(), indent=2, sort_keys=True))
        return 0
    if args.command == "qa-summary":
        print(json.dumps(_summarize_report(Path(args.report)), indent=2, sort_keys=True))
        return 0
    if args.command == "export-bucket-views":
        print(json.dumps(export_bucket_views(args.input, args.output_dir), indent=2, sort_keys=True))
        return 0
    if args.command == "report-name-evidence":
        report = build_name_evidence_report(
            metadata_path=args.metadata_output,
            bioclip_output_path=args.bioclip_output,
            keywords_json=args.keywords_json,
            target_species=args.target_species,
            score_threshold=args.score_threshold,
        )
        write_name_evidence_report(args.output, report)
        print(json.dumps({"output": args.output, **report}, indent=2, sort_keys=True))
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


def _run_bioclip_runtime_check(args: argparse.Namespace) -> int:
    runtime_python = Path(args.runtime_python)
    if not runtime_python.exists():
        print(json.dumps({"error": f"BioCLIP runtime Python not found: {runtime_python}"}, indent=2, sort_keys=True))
        return 2
    env = _bioclip_worker_env(args.hf_cache_dir)
    result = subprocess.run(
        [
            str(runtime_python),
            "-c",
            _BIOCLIP_RUNTIME_CHECK_SCRIPT,
            args.device,
        ],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )
    if result.returncode != 0:
        print(json.dumps({"error": result.stderr.strip() or result.stdout.strip()}, indent=2, sort_keys=True))
        return 2
    print(result.stdout.strip())
    return 0


def _run_bioclip_prefetch_model(args: argparse.Namespace) -> int:
    runtime_python = Path(args.runtime_python)
    if not runtime_python.exists():
        print(json.dumps({"error": f"BioCLIP runtime Python not found: {runtime_python}"}, indent=2, sort_keys=True))
        return 2
    env = _bioclip_worker_env(args.hf_cache_dir)
    result = subprocess.run(
        [
            str(runtime_python),
            "-c",
            _BIOCLIP_PREFETCH_SCRIPT,
            args.model_name,
            args.revision,
            str(args.max_workers),
        ],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )
    if result.returncode != 0:
        print(json.dumps({"error": result.stderr.strip() or result.stdout.strip()}, indent=2, sort_keys=True))
        return 2
    print(result.stdout.strip())
    return 0


def _run_bioclip_screen(args: argparse.Namespace) -> int:
    runtime_python = Path(args.runtime_python)
    if not runtime_python.exists():
        print(json.dumps({"error": f"BioCLIP runtime Python not found: {runtime_python}"}, indent=2, sort_keys=True))
        return 2

    runtime = _bioclip_runtime(runtime_python=runtime_python)
    scorer = PersistentBioClipScorer(runtime=runtime, hf_cache_dir=args.hf_cache_dir, device=args.device)
    try:
        classifier = BioClipClassifier(runtime=runtime, scorer=scorer)
        records = pl.read_parquet(args.input).to_dicts()
        species_candidates = load_species_candidates(
            args.species_candidates,
            limit=args.candidate_limit,
            target_species=args.target_species,
        )
        result = process_records_with_registers(
            records,
            classifier=classifier,
            species_candidates=species_candidates,
            output_path=args.output,
            cache_root=args.cache_root,
            register_count=args.register_count,
            register_size=args.register_size,
            download_workers=args.download_workers,
            classification_mode=args.candidate_mode,
            candidate_strategy=args.candidate_strategy,
            candidate_limit=args.candidate_limit,
            target_species=args.target_species,
            emit_image_embeddings=args.emit_image_embeddings,
            embedding_output=args.embedding_output,
            model_id="bioclip2_5",
            model_version="bioclip2_5_huge",
            model_checkpoint=BIOCLIP_25_HUGE_REVISION,
            bucket_views_dir=args.bucket_views_dir,
        )
    finally:
        scorer.close()

    print(
        json.dumps(
            {
                "output": str(result.output_path),
                "rows": result.frame.height,
                "records_seen": result.records_seen,
                "records_classified": result.records_classified,
                "records_skipped_existing": result.records_skipped_existing,
                "download_failures": result.download_failures,
                "bioclip_failures": result.bioclip_failures,
                "images_deleted_after_classification": result.images_deleted_after_classification,
                "max_staged_images": result.max_staged_images,
                "register_count": result.register_count,
                "register_size": result.register_size,
                "candidate_set_count": getattr(result, "candidate_set_count", None),
                "avg_records_per_candidate_set": getattr(result, "avg_records_per_candidate_set", None),
                "max_records_per_candidate_set": getattr(result, "max_records_per_candidate_set", None),
                "text_embedding_cache_hit_proxy": getattr(result, "text_embedding_cache_hit_proxy", None),
                "classification_mode": args.candidate_mode,
                "candidate_strategy": args.candidate_strategy,
                "embedding_output": str(getattr(result, "embedding_output_path", None)) if getattr(result, "embedding_output_path", None) else None,
                "embeddings_written": getattr(result, "embeddings_written", 0),
                "model_name": BIOCLIP_25_HUGE_REPO_ID,
                "model_revision": BIOCLIP_25_HUGE_REVISION,
                "device": args.device,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_bioclip_benchmark(args: argparse.Namespace) -> int:
    report = build_benchmark_report(
        input_path=args.input,
        species_candidates_path=args.species_candidates,
        output_path=args.output,
        device=args.device,
        register_sizes=parse_csv_ints(args.register_sizes),
        register_counts=parse_csv_ints(args.register_counts),
        candidate_limits=parse_csv_ints(args.candidate_limits),
        classification_modes=parse_csv_candidate_modes(args.classification_modes),
        candidate_strategy=args.candidate_strategy,
        download_workers=args.download_workers,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "run_id": report["run_id"],
                "configurations": len(report["runs"]) if isinstance(report.get("runs"), list) else 0,
                "status": "benchmark_skeleton_written",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _bioclip_runtime(*, runtime_python: Path) -> BioClipRuntime:
    model = ModelConfig(
        model_id="bioclip2_5_huge",
        display_name="BioCLIP 2.5 Huge",
        role="preferred",
        status="use_if_available",
        task="biology image-text classification and embedding",
        model_name=BIOCLIP_25_HUGE_REPO_ID,
        checkpoint=BIOCLIP_25_HUGE_REVISION,
        package_name="open_clip_torch",
        package_version="3.3.0",
        model_hash=f"hf-revision:{BIOCLIP_25_HUGE_REVISION}",
    )
    return BioClipRuntime(
        model=model,
        home=runtime_python.parent.parent,
        venv_python=runtime_python,
        package_version="3.3.0",
        available=True,
    )


def _bioclip_worker_env(hf_cache_dir: str | Path) -> dict[str, str]:
    env = os.environ.copy()
    cache_path = Path(hf_cache_dir).resolve()
    hub_path = cache_path / "hub"
    hub_path.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(cache_path)
    env["HUGGINGFACE_HUB_CACHE"] = str(hub_path)
    return env


_BIOCLIP_RUNTIME_CHECK_SCRIPT = r"""
from __future__ import annotations

import importlib.metadata
import json
import sys

import open_clip
import torch

requested = sys.argv[1]
if requested == "auto":
    if torch.cuda.is_available():
        resolved = "cuda"
    elif torch.backends.mps.is_available():
        resolved = "mps"
    else:
        resolved = "cpu"
elif requested == "cuda" and not torch.cuda.is_available():
    raise SystemExit("CUDA was requested but is not available")
elif requested == "mps" and not torch.backends.mps.is_available():
    raise SystemExit("MPS was requested but is not available")
else:
    resolved = requested

print(json.dumps({
    "device_requested": requested,
    "device_resolved": resolved,
    "cuda_available": torch.cuda.is_available(),
    "mps_available": torch.backends.mps.is_available(),
    "open_clip_version": importlib.metadata.version("open_clip_torch"),
    "torch_version": torch.__version__,
}, sort_keys=True))
"""


_BIOCLIP_PREFETCH_SCRIPT = rf"""
from __future__ import annotations

import json
import sys

from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=sys.argv[1],
    repo_type="model",
    revision=sys.argv[2],
    allow_patterns={list(BIOCLIP_PREFETCH_ALLOW_PATTERNS)!r},
    ignore_patterns={list(BIOCLIP_PREFETCH_IGNORE_PATTERNS)!r},
    max_workers=int(sys.argv[3]),
)
print(json.dumps({{"snapshot_path": path, "model_name": sys.argv[1], "revision": sys.argv[2]}}, sort_keys=True))
"""


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
