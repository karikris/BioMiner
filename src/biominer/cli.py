from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import subprocess

import polars as pl

from biominer.bioclip.bioclip import BioClipClassifier, PersistentBioClipScorer
from biominer.bioclip.ablation import run_object_ablations
from biominer.bioclip.candidate_sets import build_candidate_set
from biominer.bioclip.model_registry import BioClipRuntime, ModelConfig
from biominer.bioclip.object_runner import EphemeralCropBioClipScorer, screen_object_detections, write_object_evidence_outputs
from biominer.bioclip.register_runner import process_records_with_registers
from biominer.bioclip.species_candidates import DEFAULT_SPECIES_CANDIDATE_LIMIT, load_species_candidates
from biominer.detection.detector_base import DecodedImage, DetectionCandidate, FakeObjectDetector
from biominer.detection.evaluate import evaluate_xie_style
from biominer.detection.image_io import load_decoded_image_from_record
from biominer.detection.pipeline import run_detection_pipeline
from biominer.flickr_fetch.query_planner import load_registry_flickr_queries
from biominer.flickr_comments.comment_review import (
    CommentReviewState,
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
from biominer.species.context import SpeciesContext
from biominer.species.registry_refresh import resolve_species_context, write_species_registry_outputs
from biominer.species.query_compile import write_species_flickr_queries
from biominer.species.workflow import (
    build_species_comment_queue,
    fetch_species_flickr,
    run_species_workflow,
    species_candidates_from_context,
)
from biominer.storage.compaction import compact_parquet_shards
from biominer.config import StorageConfig, load_biominer_config
from biominer.storage.factory import create_storage_backend
from biominer.workstore.sqlite import SQLiteWorkStore


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
    parser.add_argument("--config")
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
    bioclip_screen.add_argument("--target-species")
    bioclip_screen.add_argument("--bucket-views-dir")
    bioclip_screen_objects = bioclip_subparsers.add_parser("screen-objects")
    bioclip_screen_objects.add_argument("--input", required=True)
    bioclip_screen_objects.add_argument("--detections", required=True)
    bioclip_screen_objects.add_argument("--species-context", required=True)
    bioclip_screen_objects.add_argument("--species-candidates")
    bioclip_screen_objects.add_argument("--geo-prior-table")
    bioclip_screen_objects.add_argument("--output", required=True)
    bioclip_screen_objects.add_argument("--ablation-mode", choices=("whole_image", "detector_crop", "detector_crop_segmentation"), default="detector_crop")
    bioclip_screen_objects.add_argument("--runtime-python", default=BIOCLIP_RUNTIME_PYTHON)
    bioclip_screen_objects.add_argument("--hf-cache-dir", default=BIOCLIP_HF_CACHE_DIR)
    bioclip_screen_objects.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    bioclip_screen_objects.add_argument("--cache-root", default="data/cache/images")
    bioclip_screen_objects.add_argument("--crop-temp-dir", default="data/cache/object_crops")
    bioclip_screen_objects.add_argument("--crop-target-px", type=int, default=336)
    bioclip_screen_objects.add_argument("--crop-padding-ratio", type=float, default=0.12)
    bioclip_screen_objects.add_argument("--retain-debug-crops", action="store_true")
    bioclip_ablate_objects = bioclip_subparsers.add_parser("ablate-objects")
    bioclip_ablate_objects.add_argument("--input", required=True)
    bioclip_ablate_objects.add_argument("--detections", required=True)
    bioclip_ablate_objects.add_argument("--species-context", required=True)
    bioclip_ablate_objects.add_argument("--species-candidates")
    bioclip_ablate_objects.add_argument("--geo-prior-table")
    bioclip_ablate_objects.add_argument("--output-dir", required=True)
    bioclip_ablate_objects.add_argument("--modes", default="whole_image,detector_crop,detector_crop_segmentation")
    bioclip_ablate_objects.add_argument("--runtime-python", default=BIOCLIP_RUNTIME_PYTHON)
    bioclip_ablate_objects.add_argument("--hf-cache-dir", default=BIOCLIP_HF_CACHE_DIR)
    bioclip_ablate_objects.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    bioclip_ablate_objects.add_argument("--cache-root", default="data/cache/images")
    bioclip_ablate_objects.add_argument("--crop-temp-dir", default="data/cache/object_crops")
    bioclip_ablate_objects.add_argument("--crop-target-px", type=int, default=336)
    bioclip_ablate_objects.add_argument("--crop-padding-ratio", type=float, default=0.12)
    bioclip_ablate_objects.add_argument("--retain-debug-crops", action="store_true")
    detect = subparsers.add_parser("detect")
    detect_subparsers = detect.add_subparsers(dest="detect_command")
    detect_boxes = detect_subparsers.add_parser("boxes")
    detect_boxes.add_argument("--input", required=True)
    detect_boxes.add_argument("--output", required=True)
    detect_boxes.add_argument("--backend", default="yolo", choices=("yolo", "fake"))
    detect_boxes.add_argument("--runtime-python", default=".venv-vision-py312/bin/python")
    detect_boxes.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    detect_crop_preview = detect_subparsers.add_parser("crop-preview")
    detect_crop_preview.add_argument("--detections", required=True)
    detect_crop_preview.add_argument("--output", required=True)
    detect_eval = detect_subparsers.add_parser("eval")
    detect_eval.add_argument("--predictions", required=True)
    detect_eval.add_argument("--ground-truth")
    detect_eval.add_argument("--output", required=True)
    fetch_comments = subparsers.add_parser("fetch-comments")
    fetch_comments.add_argument("--photo-id", action="append", default=[])
    fetch_comments.add_argument("--state-db", default="data/state/flickr_poller.sqlite")
    fetch_comments.add_argument("--limit", type=int, default=0)
    fetch_comments.add_argument("--dry-run", action="store_true")
    fetch_comments.add_argument("--selected-for-qa", action="store_true")
    fetch_comments.add_argument("--api-key-env", default="FLICKR_API_KEY")
    fetch_comments.add_argument("--min-photos", type=int, default=2)
    fetch_comments.add_argument("--min-users", type=int, default=2)
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
    species = subparsers.add_parser("species")
    species_subparsers = species.add_subparsers(dest="species_command")
    species_resolve = species_subparsers.add_parser("resolve")
    _add_species_context_args(species_resolve)
    species_refresh = species_subparsers.add_parser("refresh-registry")
    _add_species_context_args(species_refresh)
    species_compile = species_subparsers.add_parser("compile-flickr-queries")
    _add_species_context_args(species_compile)
    species_fetch = species_subparsers.add_parser("fetch-flickr")
    species_fetch.add_argument("--state-db", required=True)
    species_fetch.add_argument("--output-root", required=True)
    species_fetch.add_argument("--workers", type=int, default=8)
    species_fetch.add_argument("--max-api-calls", type=int, default=SOFT_API_CALLS_PER_HOUR)
    species_fetch.add_argument("--api-key-env", default="FLICKR_API_KEY")
    species_bioclip = species_subparsers.add_parser("bioclip-funnel")
    species_bioclip.add_argument("--context-json", required=True)
    species_bioclip.add_argument("--input", required=True)
    species_bioclip.add_argument("--species-candidates")
    species_bioclip.add_argument("--output", required=True)
    species_bioclip.add_argument("--runtime-python", default=BIOCLIP_RUNTIME_PYTHON)
    species_bioclip.add_argument("--hf-cache-dir", default=BIOCLIP_HF_CACHE_DIR)
    species_bioclip.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    species_bioclip.add_argument("--cache-root", default="data/cache/images")
    species_bioclip.add_argument("--register-count", type=int, default=4)
    species_bioclip.add_argument("--register-size", type=int, default=20)
    species_bioclip.add_argument("--download-workers", type=int, default=4)
    species_detect = species_subparsers.add_parser("detect")
    species_detect.add_argument("--input", required=True)
    species_detect.add_argument("--output", required=True)
    species_detect.add_argument("--backend", default="yolo", choices=("yolo", "fake"))
    species_detect.add_argument("--runtime-python", default=".venv-vision-py312/bin/python")
    species_detect.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    species_bioclip_objects = species_subparsers.add_parser("bioclip-objects")
    species_bioclip_objects.add_argument("--context-json", required=True)
    species_bioclip_objects.add_argument("--input", required=True)
    species_bioclip_objects.add_argument("--detections", required=True)
    species_bioclip_objects.add_argument("--species-candidates")
    species_bioclip_objects.add_argument("--output", required=True)
    species_bioclip_objects.add_argument("--ablation-mode", choices=("whole_image", "detector_crop", "detector_crop_segmentation"), default="detector_crop")
    species_bioclip_objects.add_argument("--runtime-python", default=BIOCLIP_RUNTIME_PYTHON)
    species_bioclip_objects.add_argument("--hf-cache-dir", default=BIOCLIP_HF_CACHE_DIR)
    species_bioclip_objects.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    species_bioclip_objects.add_argument("--cache-root", default="data/cache/images")
    species_bioclip_objects.add_argument("--crop-temp-dir", default="data/cache/object_crops")
    species_bioclip_objects.add_argument("--crop-target-px", type=int, default=336)
    species_bioclip_objects.add_argument("--crop-padding-ratio", type=float, default=0.12)
    species_bioclip_objects.add_argument("--retain-debug-crops", action="store_true")
    species_ablate_objects = species_subparsers.add_parser("ablate-objects")
    species_ablate_objects.add_argument("--context-json", required=True)
    species_ablate_objects.add_argument("--input", required=True)
    species_ablate_objects.add_argument("--detections", required=True)
    species_ablate_objects.add_argument("--species-candidates")
    species_ablate_objects.add_argument("--output-dir", required=True)
    species_ablate_objects.add_argument("--modes", default="whole_image,detector_crop,detector_crop_segmentation")
    species_ablate_objects.add_argument("--runtime-python", default=BIOCLIP_RUNTIME_PYTHON)
    species_ablate_objects.add_argument("--hf-cache-dir", default=BIOCLIP_HF_CACHE_DIR)
    species_ablate_objects.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    species_ablate_objects.add_argument("--cache-root", default="data/cache/images")
    species_ablate_objects.add_argument("--crop-temp-dir", default="data/cache/object_crops")
    species_ablate_objects.add_argument("--crop-target-px", type=int, default=336)
    species_ablate_objects.add_argument("--crop-padding-ratio", type=float, default=0.12)
    species_ablate_objects.add_argument("--retain-debug-crops", action="store_true")
    species_review = species_subparsers.add_parser("review-comments")
    species_review.add_argument("--context-json", required=True)
    species_review.add_argument("--input")
    species_review.add_argument("--state-db", required=True)
    species_review.add_argument("--max-api-calls", type=int, default=300)
    species_review.add_argument("--api-key-env", default="FLICKR_API_KEY")
    species_run = species_subparsers.add_parser("run")
    species_run.add_argument("--scientific-name", required=True)
    species_run.add_argument("--registry-dir", required=True)
    species_run.add_argument("--output-root", required=True)
    species_run.add_argument("--workers", type=int, default=8)
    species_run.add_argument("--max-api-calls", type=int, default=SOFT_API_CALLS_PER_HOUR)
    species_run.add_argument("--download-workers", type=int, default=4)
    species_run.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    species_run.add_argument("--api-key-env", default="FLICKR_API_KEY")
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
    poll_once_parser.add_argument("--run-id")
    poll_once_parser.add_argument("--worker-id")
    poll_once_parser.add_argument("--storage-backend", choices=("local", "s3"), default="local")
    poll_once_parser.add_argument("--storage-prefix")
    poll_once_parser.add_argument("--evidence-stage", default="poll_once")
    poll_once_parser.add_argument("--no-compact", action="store_true")
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
    compact_parquet.add_argument("--input-root")
    compact_parquet.add_argument("--output")
    compact_parquet.add_argument("--input-prefix")
    compact_parquet.add_argument("--output-prefix")
    compact_parquet.add_argument("--source-stage", default="poll_once")
    compact_parquet.add_argument("--output-stage")
    compact_parquet.add_argument("--registry-version")
    compact_parquet.add_argument("--run-id")
    compact_parquet.add_argument("--compaction-run-id")
    compact_parquet.add_argument("--target-file-mb", type=int, default=256)
    compact_parquet.add_argument("--max-file-mb", type=int, default=512)
    compact_parquet.add_argument("--dedupe-key", action="append", default=[])
    compact_parquet.add_argument("--schema-mode", choices=("strict", "diagonal_relaxed"), default="strict")
    compact_parquet.add_argument("--dry-run", action="store_true")
    compact_parquet.add_argument("--storage-backend", choices=("local", "s3"))
    compact_parquet.add_argument("--workstore-sqlite-path")
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


def _add_species_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scientific-name")
    parser.add_argument("--accepted-taxon-key")
    parser.add_argument("--registry-dir", required=True)
    parser.add_argument("--output-root", required=True)


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
        if args.bioclip_command == "screen-objects":
            return _run_bioclip_screen_objects(args)
        if args.bioclip_command == "ablate-objects":
            return _run_bioclip_ablate_objects(args)
        return 2
    if args.command == "detect":
        if args.detect_command == "boxes":
            return _run_detect_boxes(args)
        if args.detect_command == "crop-preview":
            return _run_detect_crop_preview(args)
        if args.detect_command == "eval":
            return _run_detect_eval(args)
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
    if args.command == "species":
        return _run_species_command(args)
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
            run_id=args.run_id,
            worker_id=args.worker_id,
            storage_backend=args.storage_backend,
            storage_prefix=args.storage_prefix,
            evidence_stage=args.evidence_stage,
            compact_after_run=not args.no_compact,
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
        if args.input_prefix or args.output_prefix:
            if not args.input_prefix or not args.output_prefix:
                print(json.dumps({"error": "--input-prefix and --output-prefix must be provided together"}, indent=2, sort_keys=True))
                return 2
            biominer_config = load_biominer_config(args.config)
            storage_config = StorageConfig(
                **{
                    **biominer_config.storage.__dict__,
                    "backend": args.storage_backend or biominer_config.storage.backend,
                }
            )
            storage = create_storage_backend(storage_config)
            workstore = SQLiteWorkStore(args.workstore_sqlite_path) if args.workstore_sqlite_path else None
            result = compact_parquet_shards(
                storage=storage,
                workstore=workstore,
                input_prefix=args.input_prefix,
                output_prefix=args.output_prefix,
                job_name="flickr_poll_once",
                source_stage=args.source_stage,
                output_stage=args.output_stage,
                registry_version=args.registry_version,
                run_id=args.run_id,
                compaction_run_id=args.compaction_run_id,
                target_file_mb=args.target_file_mb,
                max_file_mb=args.max_file_mb,
                dedupe_keys=args.dedupe_key or None,
                schema_mode=args.schema_mode,
                dry_run=args.dry_run,
            )
            print(json.dumps(result.__dict__, indent=2, sort_keys=True))
            return 0
        if not args.input_root or not args.output:
            print(json.dumps({"error": "--input-root and --output are required for legacy compaction"}, indent=2, sort_keys=True))
            return 2
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


def _run_species_command(args: argparse.Namespace) -> int:
    if args.species_command in {"resolve", "refresh-registry", "compile-flickr-queries"}:
        try:
            context = resolve_species_context(
                scientific_name=args.scientific_name,
                accepted_taxon_key=args.accepted_taxon_key,
                registry_dir=args.registry_dir,
            )
        except ValueError as exc:
            print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
            return 2
        report = write_species_registry_outputs(context=context, registry_dir=args.registry_dir, output_root=args.output_root)
        payload: dict[str, object] = {
            "scientific_name": context.scientific_name,
            "accepted_taxon_key": context.accepted_taxon_key,
            "registry_version": context.registry_version,
            "output_root": args.output_root,
            "species_context": str(Path(args.output_root) / "species_context.json"),
            "registry_refresh_report": report.get("report"),
        }
        if args.species_command == "compile-flickr-queries":
            query_result = write_species_flickr_queries(context, Path(args.output_root) / "flickr_query_definitions.parquet")
            payload.update({"query_definitions": str(query_result.output_path), "query_definition_rows": query_result.rows})
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.species_command == "fetch-flickr":
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            print(json.dumps({"error": f"{args.api_key_env} is required for species fetch-flickr"}, indent=2, sort_keys=True))
            return 2
        result = fetch_species_flickr(
            state_db=args.state_db,
            output_root=args.output_root,
            max_api_calls=args.max_api_calls,
            api_key=api_key,
            workers=args.workers,
        )
        print(json.dumps({**result.__dict__, "state_db": str(result.state_db)}, indent=2, sort_keys=True))
        return 0
    if args.species_command == "bioclip-funnel":
        context = SpeciesContext.read_json(args.context_json)
        runtime_python = Path(args.runtime_python)
        if not runtime_python.exists():
            print(json.dumps({"error": f"BioCLIP runtime Python not found: {runtime_python}"}, indent=2, sort_keys=True))
            return 2
        runtime = _bioclip_runtime(runtime_python=runtime_python)
        scorer = PersistentBioClipScorer(runtime=runtime, hf_cache_dir=args.hf_cache_dir, device=args.device)
        try:
            classifier = BioClipClassifier(runtime=runtime, scorer=scorer)
            records = pl.read_parquet(args.input).to_dicts()
            species_candidates = (
                load_species_candidates(args.species_candidates, target_species=context.scientific_name)
                if args.species_candidates
                else species_candidates_from_context(context)
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
                model_id="bioclip2_5",
                model_version="bioclip2_5_huge",
                model_checkpoint=BIOCLIP_25_HUGE_REVISION,
            )
        finally:
            scorer.close()
        print(json.dumps({"output": str(result.output_path), "rows": result.frame.height}, indent=2, sort_keys=True))
        return 0
    if args.species_command == "detect":
        return _run_detect_boxes(args)
    if args.species_command == "bioclip-objects":
        args.species_context = args.context_json
        return _run_bioclip_screen_objects(args)
    if args.species_command == "ablate-objects":
        args.species_context = args.context_json
        return _run_bioclip_ablate_objects(args)
    if args.species_command == "review-comments":
        context = SpeciesContext.read_json(args.context_json)
        if args.input:
            payload = build_species_comment_queue(context=context, input_path=args.input, state_db=args.state_db)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            print(json.dumps({"error": f"{args.api_key_env} is required for species review-comments without --input"}, indent=2, sort_keys=True))
            return 2
        state = CommentReviewState(args.state_db, species_context=context)
        result = state.process_pending(fetch_comments=fetch_flickr_comments(api_key=api_key), max_api_calls=args.max_api_calls)
        print(json.dumps({**state.summary(), **result}, indent=2, sort_keys=True))
        return 0
    if args.species_command == "run":
        api_key = os.environ.get(args.api_key_env)
        try:
            result = run_species_workflow(
                scientific_name=args.scientific_name,
                registry_dir=args.registry_dir,
                output_root=args.output_root,
                workers=args.workers,
                max_api_calls=args.max_api_calls,
                api_key=api_key,
                fetch=bool(api_key),
            )
        except ValueError as exc:
            print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
            return 2
        print(
            json.dumps(
                {
                    "scientific_name": result.context.scientific_name,
                    "accepted_taxon_key": result.context.accepted_taxon_key,
                    "output_root": str(result.output_root),
                    "species_context": str(result.output_root / "species_context.json"),
                    "query_definitions": str(result.query_definitions),
                    "state_db": str(result.state_db),
                    "evidence_output": str(result.evidence_output),
                    "fetch_status": "completed" if result.poll_result else "skipped_missing_api_key",
                    "poll_result": None
                    if result.poll_result is None
                    else {**result.poll_result.__dict__, "state_db": str(result.poll_result.state_db)},
                    "download_workers": args.download_workers,
                    "device": args.device,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return 2


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
                "model_name": BIOCLIP_25_HUGE_REPO_ID,
                "model_revision": BIOCLIP_25_HUGE_REVISION,
                "device": args.device,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_detect_boxes(args: argparse.Namespace) -> int:
    records = pl.read_parquet(args.input).to_dicts()
    try:
        detector, image_loader = _detect_boxes_backend(args, records)
    except RuntimeError as exc:
        print(json.dumps({"error": str(exc), "backend": args.backend, "runtime_python": args.runtime_python}, indent=2, sort_keys=True))
        return 2
    result = run_detection_pipeline(
        records=records,
        detector=detector,
        output_path=args.output,
        image_loader=image_loader,
    )
    print(
        json.dumps(
            {
                "output": str(result.output_path),
                "rows": result.frame.height,
                "backend": detector.backend,
                "records_seen": result.records_seen,
                "images_loaded": result.images_loaded,
                "detections_written": result.detections_written,
                "crops_created": result.crops_created,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _detect_boxes_backend(args: argparse.Namespace, records: list[dict[str, object]]):
    if args.backend == "fake":
        return FakeObjectDetector([_fake_detections_for_record(record) for record in records]), _blank_decoded_image
    if args.backend == "yolo":
        from biominer.detection.yolo_detector import YoloObjectDetector

        return YoloObjectDetector(device=args.device), load_decoded_image_from_record
    raise RuntimeError(f"unsupported detection backend: {args.backend}")


def _run_detect_crop_preview(args: argparse.Namespace) -> int:
    detections = pl.read_parquet(args.detections)
    preview = {
        "detections": args.detections,
        "output": args.output,
        "rows_seen": detections.height,
        "note": "crop-preview records geometry only in the core environment; debug crop files require an explicit vision sidecar",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(preview, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(preview, indent=2, sort_keys=True))
    return 0


def _run_detect_eval(args: argparse.Namespace) -> int:
    predictions = pl.read_parquet(args.predictions).to_dicts()
    truth = pl.read_parquet(args.ground_truth).to_dicts() if args.ground_truth else None
    report = evaluate_xie_style(predictions=predictions, ground_truth=truth)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _run_bioclip_screen_objects(args: argparse.Namespace) -> int:
    runtime_python = Path(args.runtime_python)
    if not runtime_python.exists():
        print(json.dumps({"error": f"BioCLIP runtime Python not found: {runtime_python}"}, indent=2, sort_keys=True))
        return 2
    context = SpeciesContext.read_json(args.species_context)
    records = pl.read_parquet(args.input)
    detections = pl.read_parquet(args.detections)
    geo_prior_table = _optional_parquet(getattr(args, "geo_prior_table", None))
    candidate_set = build_candidate_set(context, species_candidate_path=args.species_candidates if getattr(args, "species_candidates", None) else None)
    runtime = _bioclip_runtime(runtime_python=runtime_python)
    scorer = PersistentBioClipScorer(runtime=runtime, hf_cache_dir=args.hf_cache_dir, device=args.device)
    object_scorer = EphemeralCropBioClipScorer(
        scorer=scorer,
        image_loader=lambda item: load_decoded_image_from_record(item, cache_root=args.cache_root),
        temp_dir=args.crop_temp_dir,
        crop_padding_ratio=args.crop_padding_ratio,
        crop_target_px=args.crop_target_px,
        model_id="bioclip2_5",
        model_version="bioclip2_5_huge",
        model_checkpoint=BIOCLIP_25_HUGE_REVISION,
        retain_debug_crops=args.retain_debug_crops,
    )
    try:
        result = screen_object_detections(
            canonical_records=records,
            detections=detections,
            species_context=context,
            candidate_set=candidate_set,
            scorer=object_scorer,
            output_path=args.output,
            ablation_mode=args.ablation_mode,
            geo_prior_table=geo_prior_table,
        )
    finally:
        scorer.close()
    print(
        json.dumps(
            {
                "output": str(result.output_path),
                "rows": result.frame.height,
                "records_seen": result.records_seen,
                "detections_seen": result.detections_seen,
                "crops_scored": result.crops_scored,
                "candidate_set_id": candidate_set.candidate_set_id,
                "scorer": "ephemeral_crop_bioclip",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_bioclip_ablate_objects(args: argparse.Namespace) -> int:
    runtime_python = Path(args.runtime_python)
    if not runtime_python.exists():
        print(json.dumps({"error": f"BioCLIP runtime Python not found: {runtime_python}"}, indent=2, sort_keys=True))
        return 2
    context = SpeciesContext.read_json(args.species_context)
    records = pl.read_parquet(args.input)
    detections = pl.read_parquet(args.detections)
    geo_prior_table = _optional_parquet(getattr(args, "geo_prior_table", None))
    candidate_set = build_candidate_set(context, species_candidate_path=args.species_candidates if getattr(args, "species_candidates", None) else None)
    modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())
    runtime = _bioclip_runtime(runtime_python=runtime_python)
    scorer = PersistentBioClipScorer(runtime=runtime, hf_cache_dir=args.hf_cache_dir, device=args.device)
    object_scorer = EphemeralCropBioClipScorer(
        scorer=scorer,
        image_loader=lambda item: load_decoded_image_from_record(item, cache_root=args.cache_root),
        temp_dir=args.crop_temp_dir,
        crop_padding_ratio=args.crop_padding_ratio,
        crop_target_px=args.crop_target_px,
        model_id="bioclip2_5",
        model_version="bioclip2_5_huge",
        model_checkpoint=BIOCLIP_25_HUGE_REVISION,
        retain_debug_crops=args.retain_debug_crops,
    )
    try:
        report = run_object_ablations(
            canonical_records=records,
            detections=detections,
            species_context=context,
            candidate_set=candidate_set,
            scorer=object_scorer,
            output_dir=args.output_dir,
            modes=modes,  # type: ignore[arg-type]
            geo_prior_table=geo_prior_table,
        )
    finally:
        scorer.close()
    print(json.dumps({"output_dir": str(report.output_dir), **report.report}, indent=2, sort_keys=True))
    return 0


def _optional_parquet(path: str | Path | None) -> pl.DataFrame | None:
    if not path:
        return None
    return pl.read_parquet(path)


def _fake_detections_for_record(record: dict[str, object]) -> list[DetectionCandidate]:
    bbox = record.get("bbox_xyxy")
    width = int(record.get("image_width") or record.get("width") or 1)
    height = int(record.get("image_height") or record.get("height") or 1)
    if isinstance(bbox, list | tuple) and len(bbox) == 4:
        values = tuple(float(value) for value in bbox)
    else:
        values = (0.0, 0.0, float(width), float(height))
    return [DetectionCandidate(label="butterfly_like", score=1.0, bbox_xyxy=values, objectness_score=1.0)]


def _blank_decoded_image(record: dict[str, object]) -> DecodedImage:
    width = max(1, int(record.get("image_width") or record.get("width") or 1))
    height = max(1, int(record.get("image_height") or record.get("height") or 1))
    return DecodedImage(width=width, height=height, mode="RGB", data=b"\x00\x00\x00" * width * height)


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
