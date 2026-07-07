from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
from html import escape
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys

import polars as pl

from biominer.bioclip.bioclip import PersistentBioClipScorer
from biominer.bioclip.ablation import run_object_ablations
from biominer.bioclip.candidate_sets import build_candidate_set
from biominer.bioclip.embedding_cache import read_embedding_cache, prepare_candidate_text_embedding_cache, prepare_object_image_embedding_cache
from biominer.bioclip.model_registry import BioClipRuntime, ModelConfig
from biominer.bioclip.object_runner import (
    CachedObjectEmbeddingScorer,
    EphemeralCropBioClipScorer,
    PRIMARY_VISUAL_CLASSIFIER,
    materialize_detector_crop_inputs,
    screen_object_detections,
    write_object_evidence_outputs,
)
from biominer.detection.detector_base import DecodedImage, DetectionCandidate, FakeObjectDetector
from biominer.detection.evaluate import evaluate_xie_style
from biominer.detection.image_io import load_decoded_image_from_record
from biominer.detection.pipeline import run_detection_pipeline
from biominer.detection.policy import DetectionPolicy, DetectionRunPolicy, runtime_profile
from biominer.detection.segmentation import make_segmenter
from biominer.flickr_fetch.query_planner import load_registry_flickr_queries
from biominer.flickr_comments.comment_review import (
    CommentReviewState,
    apply_comment_review_decisions_to_parquet,
    build_comment_review_queue_from_parquet,
    review_comments_once,
)
from biominer.flickr_comments.comments_enrichment import CommentsEnrichmentState, fetch_flickr_comments
from biominer.flickr_fetch.metadata_poller import SOFT_API_CALLS_PER_HOUR, MetadataPollState, poll_once
from biominer.registry.audit import audit_registry
from biominer.registry.build import build_registry
from biominer.registry.compiler import compile_registry_fixture
from biominer.registry.enrichment import DEFAULT_ENRICHMENT_SOURCES, INATURALIST_DAILY_REQUEST_LIMIT, build_enrichment_sources_from_registry, compile_enriched_registry
from biominer.registry.gbif import GBIFClient
from biominer.registry.gbif_source import build_gbif_source_snapshot
from biominer.registry.scope import load_scope
from biominer.registry.translation_sources import DEFAULT_TRANSLATION_SOURCES, DEFAULT_TRANSLATION_TARGET_LOCALES_JSON
from biominer.runtime_paths import BASE_PATH, BIOCLIP25_DIR, YOLOE26_DIR
from biominer.run import ProductionRunOrchestrator, ProductionRunRequest, RunStage
from biominer.run.stages import DEFAULT_PRODUCTION_STAGES
from biominer.secrets_loader import load_runtime_secrets_env
from biominer.species.context import SpeciesContext
from biominer.config import ConfigError, create_workstore, load_biominer_config, redact_config, redact_text, validate_config
from biominer.storage.factory import create_storage_backend
from biominer.storage.uri import is_cloud_uri, join_uri


BIOCLIP_25_HUGE_REPO_ID = "imageomics/bioclip-2.5-vith14"
BIOCLIP_25_HUGE_REVISION = "191d741545e4c741cdef4b22c6eb69c945c1e592"
BIOMINER_BASE_PATH = BASE_PATH
YOLOE26_RUNTIME_ROOT = YOLOE26_DIR
YOLOE26_RUNTIME_PYTHON = str(YOLOE26_RUNTIME_ROOT / "venv" / "bin" / "python")
YOLOE26_MODEL_DIR = str(YOLOE26_RUNTIME_ROOT / "models")
YOLOE26_CACHE_ROOT = str(YOLOE26_RUNTIME_ROOT / "cache")
BIOCLIP25_RUNTIME_ROOT = BIOCLIP25_DIR
BIOCLIP_RUNTIME_PYTHON = str(BIOCLIP25_RUNTIME_ROOT / "venv" / "bin" / "python")
BIOCLIP_HF_CACHE_DIR = str(BIOCLIP25_RUNTIME_ROOT / "cache" / "huggingface")
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

RUN_STAGE_ALIASES = {
    "resolve": RunStage.RESOLVE_TAXON_SCOPE,
    "resolve_taxon_scope": RunStage.RESOLVE_TAXON_SCOPE,
    "registry": RunStage.BUILD_REGISTRY,
    "build_registry": RunStage.BUILD_REGISTRY,
    "queries": RunStage.COMPILE_QUERIES,
    "compile_queries": RunStage.COMPILE_QUERIES,
    "enqueue": RunStage.ENQUEUE_FLICKR_WORK,
    "enqueue_flickr_work": RunStage.ENQUEUE_FLICKR_WORK,
    "fetch": RunStage.POLL_FLICKR,
    "poll": RunStage.POLL_FLICKR,
    "poll_flickr": RunStage.POLL_FLICKR,
    "detect": RunStage.DETECT_OBJECTS,
    "detect_objects": RunStage.DETECT_OBJECTS,
    "score": RunStage.SCORE_BIOCLIP,
    "score_bioclip": RunStage.SCORE_BIOCLIP,
    "join": RunStage.JOIN_EVIDENCE,
    "join_evidence": RunStage.JOIN_EVIDENCE,
    "summarize": RunStage.SUMMARIZE,
    "summary": RunStage.SUMMARIZE,
    "queue_comments": RunStage.QUEUE_COMMENT_REVIEW,
    "queue_comment_review": RunStage.QUEUE_COMMENT_REVIEW,
    "review_comments": RunStage.REVIEW_COMMENTS,
    "apply_comments": RunStage.APPLY_COMMENT_REVIEW,
    "apply_comment_review": RunStage.APPLY_COMMENT_REVIEW,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biominer")
    parser.add_argument("--config")
    parser.add_argument("--version", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    vision = subparsers.add_parser("vision")
    vision_subparsers = vision.add_subparsers(dest="vision_command")
    vision_detect = vision_subparsers.add_parser("detect")
    vision_detect.add_argument("--input", required=True)
    vision_detect.add_argument("--output", required=True)
    vision_detect.add_argument("--backend", default="yoloe26", choices=("yoloe26", "yolo26", "fake"))
    vision_detect.add_argument("--runtime-python", default=YOLOE26_RUNTIME_PYTHON)
    vision_detect.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    vision_detect.add_argument("--checkpoint", default="yoloe-26s-seg.pt")
    vision_detect.add_argument("--imgsz", type=int, default=640)
    vision_detect.add_argument("--conf", type=float, default=0.20)
    vision_detect.add_argument("--iou", type=float, default=0.50)
    vision_detect.add_argument("--max-det", type=int, default=8)
    vision_detect.add_argument("--prompt-class", action="append", default=[])
    vision_detect.add_argument("--include-hard-negative-prompts", action=argparse.BooleanOptionalAction, default=True)
    _add_detection_policy_args(vision_detect)
    vision_score = vision_subparsers.add_parser("score")
    vision_score.add_argument("--input", required=True)
    vision_score.add_argument("--detections", required=True)
    vision_score.add_argument("--species-context", required=True)
    vision_score.add_argument("--species-candidates")
    vision_score.add_argument("--geo-prior-table")
    vision_score.add_argument("--output", required=True)
    vision_score.add_argument("--ablation-mode", choices=("whole_image", "detector_crop", "detector_crop_segmentation"), default="detector_crop")
    vision_score.add_argument("--runtime-python", default=BIOCLIP_RUNTIME_PYTHON)
    vision_score.add_argument("--hf-cache-dir", default=BIOCLIP_HF_CACHE_DIR)
    vision_score.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    vision_score.add_argument("--cache-root", default="data/cache/images")
    vision_score.add_argument("--crop-temp-dir", default="data/cache/object_crops")
    vision_score.add_argument("--crop-target-px", type=int, default=336)
    vision_score.add_argument("--crop-padding-ratio", type=float, default=0.12)
    vision_score.add_argument("--parquet-batch-rows", type=int, default=10000)
    vision_score.add_argument("--retain-debug-crops", action="store_true")
    vision_score.add_argument("--text-embedding-batch-size", type=int, default=256)
    vision_score.add_argument("--candidate-text-embedding-cache")
    vision_score.add_argument("--object-image-embedding-cache")
    vision_score.add_argument("--segmenter", default="none", choices=("none", "sam", "sam2"))
    vision_ablate = vision_subparsers.add_parser("ablate")
    vision_ablate.add_argument("--input", required=True)
    vision_ablate.add_argument("--detections", required=True)
    vision_ablate.add_argument("--species-context", required=True)
    vision_ablate.add_argument("--species-candidates")
    vision_ablate.add_argument("--geo-prior-table")
    vision_ablate.add_argument("--output-dir", required=True)
    vision_ablate.add_argument("--modes", default="whole_image,detector_crop,detector_crop_segmentation")
    vision_ablate.add_argument("--runtime-python", default=BIOCLIP_RUNTIME_PYTHON)
    vision_ablate.add_argument("--hf-cache-dir", default=BIOCLIP_HF_CACHE_DIR)
    vision_ablate.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    vision_ablate.add_argument("--cache-root", default="data/cache/images")
    vision_ablate.add_argument("--crop-temp-dir", default="data/cache/object_crops")
    vision_ablate.add_argument("--crop-target-px", type=int, default=336)
    vision_ablate.add_argument("--crop-padding-ratio", type=float, default=0.12)
    vision_ablate.add_argument("--parquet-batch-rows", type=int, default=10000)
    vision_ablate.add_argument("--retain-debug-crops", action="store_true")
    vision_ablate.add_argument("--text-embedding-batch-size", type=int, default=256)
    vision_ablate.add_argument("--candidate-text-embedding-cache")
    vision_ablate.add_argument("--object-image-embedding-cache")
    vision_ablate.add_argument("--segmenter", default="none", choices=("none", "sam", "sam2"))
    evidence = subparsers.add_parser("evidence")
    evidence_subparsers = evidence.add_subparsers(dest="evidence_command")
    evidence_join = evidence_subparsers.add_parser("join")
    _add_object_evidence_join_args(evidence_join)
    evidence_join.add_argument("--species-context")
    registry = subparsers.add_parser("registry")
    registry_subparsers = registry.add_subparsers(dest="registry_command")
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
    registry_build.add_argument("--enrichment-sources", default=",".join(DEFAULT_ENRICHMENT_SOURCES))
    registry_build.add_argument("--translation-sources", default=",".join(DEFAULT_TRANSLATION_SOURCES))
    registry_build.add_argument("--translation-target-locales-json", default=DEFAULT_TRANSLATION_TARGET_LOCALES_JSON)
    registry_build.add_argument("--skip-translations", action="store_true")
    registry_build.add_argument("--translation-daily-request-limit", type=int, default=10000)
    registry_build.add_argument(
        "--max-translation-candidates-per-name",
        type=int,
        default=0,
        help="Maximum MyMemory candidates to keep per source name and target language; 0 keeps all returned candidates.",
    )
    registry_build.add_argument("--mymemory-email")
    registry_build.add_argument("--mymemory-key")
    registry_build.add_argument("--mymemory-allow-machine-translation", action="store_true")
    registry_build.add_argument("--translation-workers", type=int, default=1)
    registry_build.add_argument("--translation-checkpoint-every", type=int, default=100)
    registry_build.add_argument("--translation-checkpoint-seconds", type=float, default=60.0)
    registry_build.add_argument("--translation-language-shards", type=int, default=0)
    registry_build.add_argument("--query-curation-json")
    registry_build.add_argument("--inaturalist-daily-request-limit", type=int, default=INATURALIST_DAILY_REQUEST_LIMIT)
    registry_build.add_argument("--range-discovery-source", default="gbif", choices=("gbif",))
    registry_build.add_argument("--range-seed-json")
    registry_build.add_argument("--language-targets-json")
    registry_build.add_argument("--curated-static-source-config-dir", default="config/vernacular_sources")
    registry_build.add_argument("--curated-static-source-snapshot-dir", default="data/source_snapshots")
    registry_build.add_argument("--skip-range-discovery", action="store_true")
    registry_build.add_argument("--skip-language-targets", action="store_true")
    registry_build.add_argument("--skip-curated-static-sources", action="store_true")
    registry_build.add_argument("--skip-enrichment", action="store_true")
    registry_audit = registry_subparsers.add_parser("audit")
    registry_audit.add_argument("--registry-dir", required=True)
    registry_audit.add_argument("--report-dir", default="reports")
    dev = subparsers.add_parser("dev")
    dev_subparsers = dev.add_subparsers(dest="dev_command")
    dev_vision = dev_subparsers.add_parser("vision")
    dev_vision_subparsers = dev_vision.add_subparsers(dest="vision_command")
    _add_dev_vision_commands(dev_vision_subparsers)
    dev_registry = dev_subparsers.add_parser("registry")
    dev_registry_subparsers = dev_registry.add_subparsers(dest="registry_command")
    registry_compile = dev_registry_subparsers.add_parser("compile-fixture")
    registry_compile.add_argument("--source-json", required=True)
    registry_compile.add_argument("--output-dir", required=True)
    registry_compile.add_argument("--registry-version", required=True)
    registry_compile.add_argument("--scope-json", default="config/butterfly_scope.json")
    registry_compile_enriched = dev_registry_subparsers.add_parser("compile-enriched")
    registry_compile_enriched.add_argument("--registry-dir", required=True)
    registry_compile_enriched.add_argument("--registry-version", required=True)
    registry_compile_enriched.add_argument("--scope-json", default="config/butterfly_scope.json")
    registry_compile_enriched.add_argument("--query-curation-json")
    registry_fetch_taxonomy = dev_registry_subparsers.add_parser("fetch-taxonomy")
    registry_fetch_taxonomy.add_argument("--output-json", required=True)
    registry_fetch_taxonomy.add_argument("--scope-json", default="config/butterfly_scope.json")
    registry_fetch_taxonomy.add_argument("--retrieved-at")
    registry_enrich_sources = dev_registry_subparsers.add_parser("enrich-sources")
    registry_enrich_sources.add_argument("--registry-dir", required=True)
    registry_enrich_sources.add_argument("--sources", default=",".join(DEFAULT_ENRICHMENT_SOURCES))
    registry_enrich_sources.add_argument("--workers", type=int, default=8)
    registry_enrich_sources.add_argument("--progress-every", type=int, default=100)
    registry_enrich_sources.add_argument("--checkpoint-every", type=int, default=500)
    registry_enrich_sources.add_argument("--max-retries", type=int, default=5)
    registry_enrich_sources.add_argument("--inaturalist-daily-request-limit", type=int, default=INATURALIST_DAILY_REQUEST_LIMIT)
    registry_enrich_sources.add_argument("--limit", type=int, default=0)
    registry_enrich_sources.add_argument("--report-dir", default="reports")
    registry_seed = dev_registry_subparsers.add_parser("seed-flickr-queries")
    registry_seed.add_argument("--query-definitions", required=True)
    registry_seed.add_argument("--state-db", default="data/state/flickr_poller.sqlite")
    dev_comments = dev_subparsers.add_parser("comments")
    dev_comments_subparsers = dev_comments.add_subparsers(dest="comments_command")
    comments_fetch = dev_comments_subparsers.add_parser("fetch")
    comments_fetch.add_argument("--photo-id", action="append", default=[])
    comments_fetch.add_argument("--state-db", default="data/state/flickr_poller.sqlite")
    comments_fetch.add_argument("--limit", type=int, default=0)
    comments_fetch.add_argument("--dry-run", action="store_true")
    comments_fetch.add_argument("--selected-for-qa", action="store_true")
    comments_fetch.add_argument("--api-key-env", default="FLICKR_API_KEY")
    comments_fetch.add_argument("--min-photos", type=int, default=2)
    comments_fetch.add_argument("--min-users", type=int, default=2)
    comments_queue = dev_comments_subparsers.add_parser("queue")
    comments_queue.add_argument("--input", required=True)
    comments_queue.add_argument("--state-db", default="data/state/comment_review.sqlite")
    comments_review = dev_comments_subparsers.add_parser("review-once")
    comments_review.add_argument("--state-db", default="data/state/comment_review.sqlite")
    comments_review.add_argument("--max-api-calls", type=int, default=300)
    comments_review.add_argument("--api-key-env", default="FLICKR_API_KEY")
    comments_apply = dev_comments_subparsers.add_parser("apply-decisions")
    comments_apply.add_argument("--input", required=True)
    comments_apply.add_argument("--output", required=True)
    comments_apply.add_argument("--state-db", default="data/state/comment_review.sqlite")
    dev_flickr = dev_subparsers.add_parser("flickr")
    dev_flickr_subparsers = dev_flickr.add_subparsers(dest="flickr_command")
    dev_poll_once = dev_flickr_subparsers.add_parser("poll-once")
    _add_poll_once_args(dev_poll_once)
    storage = subparsers.add_parser("storage")
    storage_subparsers = storage.add_subparsers(dest="storage_command")
    storage_doctor = storage_subparsers.add_parser("doctor")
    storage_doctor.add_argument("--config")
    workstore = subparsers.add_parser("workstore")
    workstore_subparsers = workstore.add_subparsers(dest="workstore_command")
    workstore_doctor = workstore_subparsers.add_parser("doctor")
    workstore_doctor.add_argument("--config")
    production_run = subparsers.add_parser("run")
    production_run.add_argument("--taxon", required=True)
    production_run.add_argument("--rank", default="auto", choices=("auto", "family", "genus", "species"))
    production_run.add_argument("--registry-dir", required=True)
    production_run.add_argument("--output-prefix", required=True)
    production_run.add_argument("--storage-backend", default="s3", choices=("s3", "local"))
    production_run.add_argument("--workstore-backend", default="postgres", choices=("postgres", "sqlite"))
    production_run.add_argument("--vision-backend", default="yoloe26")
    production_run.add_argument("--bioclip-model", default=BIOCLIP_25_HUGE_REPO_ID)
    production_run.add_argument("--stages")
    production_run.add_argument("--dry-run", action="store_true")
    production_run.add_argument("--build-registry-if-missing", action="store_true")
    production_run.add_argument("--limit-species", type=int, default=0)
    production_run.add_argument("--limit-records", type=int, default=0)
    production_run.add_argument("--comments-max-api-calls", type=int, default=300)
    return parser


def _add_poll_once_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-api-calls", type=int, default=SOFT_API_CALLS_PER_HOUR)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--stale-claim-seconds", type=int, default=3600)
    parser.add_argument("--state-db", default="data/state/flickr_poller.sqlite")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--evidence-output", default="staging/evidence/poll_once_evidence.parquet")
    parser.add_argument("--api-key-env", default="FLICKR_API_KEY")
    parser.add_argument("--run-id")
    parser.add_argument("--worker-id")
    parser.add_argument("--storage-backend", choices=("local", "s3"), default="local")
    parser.add_argument("--storage-prefix")
    parser.add_argument("--evidence-stage", default="poll_once")
    parser.add_argument("--no-compact", action="store_true")
    parser.add_argument("--config")


def _add_object_evidence_join_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--joined-output", required=True)
    parser.add_argument("--photo-summary-output", required=True)


def _add_detection_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=("mac_m5pro_64gb",))
    parser.add_argument("--box-score-threshold", type=float)
    parser.add_argument("--nms-iou-threshold", type=float)
    parser.add_argument("--min-box-area-ratio", type=float)
    parser.add_argument("--max-boxes-per-image", type=int)
    parser.add_argument("--crop-padding-ratio", type=float)
    parser.add_argument("--image-max-side-px", type=int)
    parser.add_argument("--crop-target-px", type=int)
    parser.add_argument("--retain-debug-crops", action="store_true")
    parser.add_argument("--debug-crop-limit", type=int)
    parser.add_argument("--download-workers", type=int)
    parser.add_argument("--decode-workers", type=int)
    parser.add_argument("--detector-workers", type=int)
    parser.add_argument("--max-inflight-images", type=int)
    parser.add_argument("--max-inflight-crops", type=int)
    parser.add_argument("--detector-batch-size", type=int)
    parser.add_argument("--crop-batch-size", type=int)
    parser.add_argument("--parquet-batch-rows", type=int)


def _add_dev_vision_commands(subparsers: Any) -> None:
    bioclip_runtime = subparsers.add_parser("bioclip-runtime-check")
    bioclip_runtime.add_argument("--runtime-python", default=BIOCLIP_RUNTIME_PYTHON)
    bioclip_runtime.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    bioclip_runtime.add_argument("--hf-cache-dir", default=BIOCLIP_HF_CACHE_DIR)
    bioclip_prefetch = subparsers.add_parser("bioclip-prefetch-model")
    bioclip_prefetch.add_argument("--runtime-python", default=BIOCLIP_RUNTIME_PYTHON)
    bioclip_prefetch.add_argument("--hf-cache-dir", default=BIOCLIP_HF_CACHE_DIR)
    bioclip_prefetch.add_argument("--model-name", default=BIOCLIP_25_HUGE_REPO_ID)
    bioclip_prefetch.add_argument("--revision", default=BIOCLIP_25_HUGE_REVISION)
    bioclip_prefetch.add_argument("--max-workers", type=int, default=8)
    yoloe26_runtime = subparsers.add_parser("yoloe26-runtime-check")
    yoloe26_runtime.add_argument("--runtime-python", default=YOLOE26_RUNTIME_PYTHON)
    yoloe26_runtime.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    yoloe26_runtime.add_argument("--checkpoint", default="yoloe-26s-seg.pt")
    yoloe26_prefetch = subparsers.add_parser("yoloe26-prefetch")
    yoloe26_prefetch.add_argument("--runtime-python", default=YOLOE26_RUNTIME_PYTHON)
    yoloe26_prefetch.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    yoloe26_prefetch.add_argument("--checkpoint", default="yoloe-26s-seg.pt")
    yoloe26_smoke = subparsers.add_parser("yoloe26-smoke")
    yoloe26_smoke.add_argument("--runtime-python", default=YOLOE26_RUNTIME_PYTHON)
    yoloe26_smoke.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    yoloe26_smoke.add_argument("--checkpoint", default="yoloe-26s-seg.pt")
    yoloe26_smoke.add_argument("--image")
    yoloe26_smoke.add_argument("--output-dir", default="reports/yoloe26_smoke")
    yoloe26_prototype = subparsers.add_parser("yoloe26-prototype-run")
    yoloe26_prototype.add_argument("--input", required=True)
    yoloe26_prototype.add_argument("--species-context", required=True)
    yoloe26_prototype.add_argument("--species-candidates")
    yoloe26_prototype.add_argument("--output-dir", required=True)
    yoloe26_prototype.add_argument("--vision-runtime-python", default=YOLOE26_RUNTIME_PYTHON)
    yoloe26_prototype.add_argument("--bioclip-runtime-python", default=BIOCLIP_RUNTIME_PYTHON)
    yoloe26_prototype.add_argument("--hf-cache-dir", default=BIOCLIP_HF_CACHE_DIR)
    yoloe26_prototype.add_argument("--cache-root", default=str(YOLOE26_RUNTIME_ROOT / "cache" / "images"))
    yoloe26_prototype.add_argument("--crop-temp-dir", default=str(YOLOE26_RUNTIME_ROOT / "cache" / "object_crops"))
    yoloe26_prototype.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    yoloe26_prototype.add_argument("--checkpoint", default="yoloe-26s-seg.pt")
    yoloe26_prototype.add_argument("--limit", type=int)
    yoloe26_prototype.add_argument("--retain-debug-crops", action="store_true")
    yoloe26_prototype.add_argument("--ablation-mode", choices=("whole_image", "detector_crop", "detector_crop_segmentation"), default="detector_crop")
    yoloe26_prototype.add_argument("--also-whole-image", action="store_true")
    yoloe26_prototype.add_argument("--parquet-batch-rows", type=int, default=10000)
    yoloe26_prototype.add_argument("--imgsz", type=int, default=640)
    yoloe26_prototype.add_argument("--conf", type=float, default=0.20)
    yoloe26_prototype.add_argument("--iou", type=float, default=0.50)
    yoloe26_prototype.add_argument("--max-det", type=int, default=8)
    yoloe26_prototype.add_argument("--prompt-class", action="append", default=[])
    yoloe26_prototype.add_argument("--include-hard-negative-prompts", action=argparse.BooleanOptionalAction, default=True)
    detect_crop_preview = subparsers.add_parser("crop-preview")
    detect_crop_preview.add_argument("--detections", required=True)
    detect_crop_preview.add_argument("--output", required=True)
    detect_crop_preview.add_argument("--limit", type=int, default=200)
    detect_eval = subparsers.add_parser("eval")
    detect_eval.add_argument("--predictions", required=True)
    detect_eval.add_argument("--ground-truth")
    detect_eval.add_argument("--output", required=True)
    detect_eval.add_argument("--iou-threshold", type=float, default=0.5)
    detect_eval.add_argument("--score-threshold", type=float, default=0.35)


def run(args: argparse.Namespace) -> int:
    if args.version:
        print("biominer 0.1.0")
        return 0
    if args.command == "vision" or (args.command == "dev" and args.dev_command == "vision"):
        if args.vision_command == "detect":
            return _run_detect_boxes(args)
        if args.vision_command == "score":
            return _run_bioclip_screen_objects(args)
        if args.vision_command == "ablate":
            return _run_bioclip_ablate_objects(args)
        if args.vision_command == "bioclip-runtime-check":
            return _run_bioclip_runtime_check(args)
        if args.vision_command == "bioclip-prefetch-model":
            return _run_bioclip_prefetch_model(args)
        if args.vision_command == "yoloe26-runtime-check":
            return _run_yoloe26_runtime_check(args)
        if args.vision_command == "yoloe26-prefetch":
            return _run_yoloe26_prefetch(args)
        if args.vision_command == "yoloe26-smoke":
            return _run_yoloe26_smoke(args)
        if args.vision_command == "yoloe26-prototype-run":
            return _run_yoloe26_prototype_run(args)
        if args.vision_command == "crop-preview":
            return _run_detect_crop_preview(args)
        if args.vision_command == "eval":
            return _run_detect_eval(args)
        return 2
    if args.command == "evidence":
        if args.evidence_command == "join":
            return _run_bioclip_join_object_evidence(args)
        return 2
    if args.command == "storage":
        return _run_storage_command(args)
    if args.command == "workstore":
        return _run_workstore_command(args)
    if args.command == "run":
        return _run_production_command(args)
    if args.command == "dev" and args.dev_command == "comments":
        return _run_dev_comments_command(args)
    if args.command == "dev" and args.dev_command == "flickr":
        return _run_dev_flickr_command(args)
    if args.command == "registry" or (args.command == "dev" and args.dev_command == "registry"):
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
                query_curation_json=args.query_curation_json,
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
                    translation_sources=tuple(part.strip() for part in args.translation_sources.split(",") if part.strip()),
                    translation_target_locales_json=args.translation_target_locales_json,
                    skip_translations=args.skip_translations,
                    translation_daily_request_limit=args.translation_daily_request_limit,
                    max_translation_candidates_per_name=args.max_translation_candidates_per_name,
                    mymemory_email=args.mymemory_email,
                    mymemory_key=args.mymemory_key,
                    mymemory_allow_machine_translation=args.mymemory_allow_machine_translation,
                    translation_workers=args.translation_workers,
                    translation_checkpoint_every=args.translation_checkpoint_every,
                    translation_checkpoint_seconds=args.translation_checkpoint_seconds,
                    translation_language_shards=args.translation_language_shards,
                    query_curation_json=args.query_curation_json,
                    inaturalist_daily_request_limit=args.inaturalist_daily_request_limit,
                    range_discovery_source=args.range_discovery_source,
                    range_seed_json=args.range_seed_json,
                    language_targets_json=args.language_targets_json,
                    curated_static_source_config_dir=args.curated_static_source_config_dir,
                    curated_static_source_snapshot_dir=args.curated_static_source_snapshot_dir,
                    skip_range_discovery=args.skip_range_discovery,
                    skip_language_targets=args.skip_language_targets,
                    skip_curated_static_sources=args.skip_curated_static_sources,
                    skip_enrichment=args.skip_enrichment,
                )
            except FileNotFoundError as exc:
                print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
                return 2
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.registry_command == "seed-flickr-queries":
            queries = load_registry_flickr_queries(args.query_definitions)
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
            print(json.dumps(audit_registry(args.registry_dir, report_dir=args.report_dir), indent=2, sort_keys=True))
            return 0
        return 2
    return 2


def _run_dev_comments_command(args: argparse.Namespace) -> int:
    if args.comments_command == "fetch":
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
    if args.comments_command == "queue":
        payload = build_comment_review_queue_from_parquet(input_path=args.input, state_db=args.state_db)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.comments_command == "review-once":
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
    if args.comments_command == "apply-decisions":
        payload = apply_comment_review_decisions_to_parquet(input_path=args.input, output_path=args.output, state_db=args.state_db)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    return 2


def _run_dev_flickr_command(args: argparse.Namespace) -> int:
    if args.flickr_command != "poll-once":
        return 2
    work_store = None
    if args.no_compact and args.storage_backend != "local":
        biominer_config = load_biominer_config(args.config)
        work_store = create_workstore(biominer_config.workstore)
        _init_workstore_schema(work_store)
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
        work_store=work_store,
    )
    print(json.dumps({**result.__dict__, "state_db": str(result.state_db)}, indent=2, sort_keys=True))
    return 0


def _run_storage_command(args: argparse.Namespace) -> int:
    if args.storage_command != "doctor":
        return 2
    try:
        payload = _run_storage_doctor(args)
    except Exception as exc:  # pragma: no cover - exercised by live doctor runs.
        print(json.dumps({"status": "error", "error": _redact_cloud_error(str(exc), args)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "ok" else 2


def _run_workstore_command(args: argparse.Namespace) -> int:
    if args.workstore_command != "doctor":
        return 2
    try:
        payload = _run_workstore_doctor(args)
    except Exception as exc:  # pragma: no cover - exercised by live doctor runs.
        print(json.dumps({"status": "error", "error": _redact_cloud_error(str(exc), args)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "ok" else 2


def _run_storage_doctor(args: argparse.Namespace) -> dict[str, object]:
    config = load_biominer_config(args.config)
    storage = create_storage_backend(config.storage)
    run_id = f"storage-doctor-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    base_uri = _storage_base_uri(storage=storage, config=config)
    doctor_prefix = join_uri(base_uri, "doctor", f"run_id={run_id}")

    json_uri = join_uri(doctor_prefix, "probe.json")
    json_payload = {"run_id": run_id, "probe": "json"}
    storage.write_json(json_uri, json_payload)
    json_roundtrip = storage.read_json(json_uri) == json_payload
    json_deleted = storage.delete(json_uri)

    parquet_uri = join_uri(doctor_prefix, "probe.parquet")
    parquet_frame = pl.DataFrame({"probe": ["a", "b"], "value": [1, 2]})
    storage.write_parquet_shard(parquet_uri, parquet_frame)
    parquet_rows = storage.scan_parquet(parquet_uri).collect().height
    return {
        "status": "ok",
        "command": "storage doctor",
        "config": redact_config(config),
        "storage": {
            "backend": config.storage.backend,
            "json_roundtrip": json_roundtrip,
            "json_deleted": json_deleted,
            "json_uri": json_uri,
            "parquet_uri": parquet_uri,
            "parquet_rows": parquet_rows,
        },
    }


def _run_workstore_doctor(args: argparse.Namespace) -> dict[str, object]:
    config = load_biominer_config(args.config)
    workstore = create_workstore(config.workstore)
    run_id = f"workstore-doctor-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    job_name = "workstore_doctor"
    stage = "doctor"
    work_key = f"workstore-doctor-work:{run_id}"
    parquet_uri = f"workstore-doctor://{run_id}/probe.parquet"
    parquet_rows = 0
    payload: dict[str, object] = {
        "status": "ok",
        "command": "workstore doctor",
        "config": redact_config(config),
        "workstore": {
            "backend": config.workstore.backend,
            "schema_initialized": False,
            "work_items_inserted": 0,
            "claimed_work_key": None,
            "completed_keys": [],
            "registered_shards": 0,
        },
    }
    try:
        _init_workstore_schema(workstore)
        workstore.get_or_create_run(
            job_name=job_name,
            stage=stage,
            run_id=run_id,
            registry_version=None,
            config={"command": "workstore doctor"},
        )
        inserted = workstore.enqueue_work(job_name, None, [{"work_key": work_key, "probe": "workstore"}], stage=stage)
        claimed = workstore.claim_next_batch(config.runtime.worker_id, 1, job_name=job_name, stage=stage, registry_version=None)
        claimed_work_key = str(claimed[0]["work_key"]) if claimed else None
        if claimed_work_key:
            workstore.mark_completed(claimed_work_key, parquet_uri, None, parquet_rows)
        workstore.register_shard(
            shard_id=f"{run_id}-probe",
            job_name=job_name,
            registry_version=None,
            stage=stage,
            run_id=run_id,
            worker_id=config.runtime.worker_id,
            uri=parquet_uri,
            checksum=None,
            row_count=parquet_rows,
            byte_count=None,
            metadata={"kind": "workstore_doctor"},
        )
        shards = workstore.list_committed_shards(job_name=job_name, stage=stage, registry_version=None, run_id=run_id)
        payload["workstore"] = {
            "backend": config.workstore.backend,
            "schema_initialized": True,
            "work_items_inserted": inserted,
            "claimed_work_key": claimed_work_key,
            "completed_keys": [work_key] if work_key in workstore.completed_keys(job_name, None, stage=stage) else [],
            "registered_shards": len(shards),
        }
    except Exception as exc:  # noqa: BLE001 - doctor reports partial diagnostics.
        payload["status"] = "error"
        workstore_payload = dict(payload["workstore"]) if isinstance(payload["workstore"], dict) else {}
        payload["workstore"] = {**workstore_payload, "error": redact_text(str(exc), config)}
    return payload


def _init_workstore_schema(workstore: object) -> None:
    init_schema = getattr(workstore, "init_schema", None)
    if not callable(init_schema):
        raise RuntimeError("configured workstore does not support schema initialization")
    init_schema()


def _init_workstore_schema_if_supported(workstore: object) -> None:
    init_schema = getattr(workstore, "init_schema", None)
    if callable(init_schema):
        init_schema()


def _redact_cloud_error(error: str, args: argparse.Namespace) -> str:
    try:
        config = load_biominer_config(args.config)
    except Exception:  # noqa: BLE001 - best-effort fallback for config-load failures.
        return error
    return redact_text(error, config)


def _storage_base_uri(*, storage: object, config: object) -> str:
    base_uri = getattr(storage, "base_uri", None)
    if base_uri:
        return str(base_uri)
    storage_config = getattr(config, "storage")
    if getattr(storage_config, "backend") == "s3" and getattr(storage_config, "bucket"):
        return join_uri(f"s3://{storage_config.bucket}", str(getattr(storage_config, "prefix", "") or ""))
    return str(getattr(storage_config, "prefix", "."))


def _run_production_command(args: argparse.Namespace) -> int:
    config = None
    try:
        config = load_biominer_config(args.config)
        config = replace(
            config,
            storage=replace(config.storage, backend=args.storage_backend),
            workstore=replace(config.workstore, backend=args.workstore_backend),
        )
        allow_local = args.storage_backend == "local" and args.workstore_backend == "sqlite"
        if (args.storage_backend == "local") != (args.workstore_backend == "sqlite"):
            raise ConfigError("local dev mode requires --storage-backend local --workstore-backend sqlite")
        validate_config(config, require_cloud_credentials=not allow_local, allow_local_backends=allow_local)
        stages = _parse_run_stages(args.stages)
        storage = None
        registry_dir_is_cloud = is_cloud_uri(args.registry_dir)
        if args.storage_backend != "local" and (not args.dry_run or registry_dir_is_cloud):
            storage = create_storage_backend(config.storage)
        workstore = None
        if not args.dry_run and RunStage.ENQUEUE_FLICKR_WORK in stages:
            workstore = create_workstore(config.workstore)
            _init_workstore_schema_if_supported(workstore)
        limits = {
            key: value
            for key, value in {
                "species": args.limit_species,
                "records": args.limit_records,
                "comment_api_calls": args.comments_max_api_calls,
            }.items()
            if value and value > 0
        }
        request = ProductionRunRequest(
            taxon=args.taxon,
            rank=args.rank,
            registry_dir=args.registry_dir,
            output_root=args.output_prefix,
            storage_backend=args.storage_backend,
            workstore_backend=args.workstore_backend,
            vision_backend=args.vision_backend,
            bioclip_model=args.bioclip_model,
            worker_id=config.runtime.worker_id or ("local" if allow_local else ""),
            stages=stages,
            dry_run=args.dry_run,
            build_registry_if_missing=args.build_registry_if_missing,
            limits=limits,
        )
        plan = ProductionRunOrchestrator(request, storage=storage, workstore=workstore).run()
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        payload: dict[str, object] = {"error": redact_text(str(exc), config) if config else str(exc)}
        if config is not None:
            payload["config"] = redact_config(config)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    return 0


def _parse_run_stages(value: str | None) -> tuple[RunStage, ...]:
    if not value:
        return DEFAULT_PRODUCTION_STAGES
    stages: list[RunStage] = []
    for raw_part in value.split(","):
        part = raw_part.strip().casefold()
        if not part:
            continue
        if part == "all":
            return DEFAULT_PRODUCTION_STAGES
        stage = RUN_STAGE_ALIASES.get(part)
        if stage is None:
            try:
                stage = RunStage(part)
            except ValueError as exc:
                allowed = ", ".join(sorted(RUN_STAGE_ALIASES))
                raise ValueError(f"unknown run stage {raw_part!r}; expected one of: {allowed}") from exc
        if stage not in stages:
            stages.append(stage)
    return tuple(stages) or DEFAULT_PRODUCTION_STAGES


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


def _run_detect_boxes(args: argparse.Namespace) -> int:
    records = pl.read_parquet(args.input).to_dicts()
    try:
        detector, image_loader = _detect_boxes_backend(args, records)
    except (RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc), "backend": args.backend, "runtime_python": args.runtime_python}, indent=2, sort_keys=True))
        return 2
    detection_policy = _detection_policy_from_args(args)
    run_policy = _detection_run_policy_from_args(args)
    result = run_detection_pipeline(
        records=records,
        detector=detector,
        output_path=args.output,
        image_loader=image_loader,
        detection_policy=detection_policy,
        run_policy=run_policy,
    )
    print(
        json.dumps(
            {
                "output": str(result.output_path),
                "rows": result.frame.height,
                "profile": args.profile,
                "backend": detector.backend,
                "records_seen": result.records_seen,
                "images_loaded": result.images_loaded,
                "detections_written": result.detections_written,
                "crops_created": result.crops_created,
                "parquet_batches_written": result.parquet_batches_written,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_yoloe26_runtime_check(args: argparse.Namespace) -> int:
    runtime_python = Path(args.runtime_python).expanduser()
    if not runtime_python.exists():
        print(json.dumps({"error": f"YOLOE-26 runtime Python not found: {runtime_python}"}, indent=2, sort_keys=True))
        return 2
    result = subprocess.run(
        [
            str(runtime_python),
            "-c",
            _YOLOE26_RUNTIME_CHECK_SCRIPT,
            args.device,
            args.checkpoint,
        ],
        capture_output=True,
        check=False,
        cwd=str(_yoloe26_model_dir(runtime_python)),
        env=_yoloe26_worker_env(runtime_python),
        text=True,
    )
    if result.returncode != 0:
        print(json.dumps({"error": result.stderr.strip() or result.stdout.strip()}, indent=2, sort_keys=True))
        return 2
    print(result.stdout.strip())
    return 0


def _run_yoloe26_prefetch(args: argparse.Namespace) -> int:
    runtime_python = Path(args.runtime_python).expanduser()
    if not runtime_python.exists():
        print(json.dumps({"error": f"YOLOE-26 runtime Python not found: {runtime_python}"}, indent=2, sort_keys=True))
        return 2
    result = subprocess.run(
        [
            str(runtime_python),
            "-c",
            _YOLOE26_PREFETCH_SCRIPT,
            args.device,
            args.checkpoint,
            json.dumps(list(_default_yoloe26_prompts())),
        ],
        capture_output=True,
        check=False,
        cwd=str(_yoloe26_model_dir(runtime_python)),
        env=_yoloe26_worker_env(runtime_python),
        text=True,
    )
    if result.returncode != 0:
        print(json.dumps({"error": result.stderr.strip() or result.stdout.strip()}, indent=2, sort_keys=True))
        return 2
    print(result.stdout.strip())
    return 0


def _run_yoloe26_smoke(args: argparse.Namespace) -> int:
    runtime_python = Path(args.runtime_python).expanduser()
    if not runtime_python.exists():
        print(json.dumps({"error": f"YOLOE-26 runtime Python not found: {runtime_python}"}, indent=2, sort_keys=True))
        return 2
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = str(Path(args.image).expanduser().resolve()) if args.image else ""
    result = subprocess.run(
        [
            str(runtime_python),
            "-c",
            _YOLOE26_SMOKE_SCRIPT,
            args.device,
            args.checkpoint,
            str(output_dir),
            image_path,
            json.dumps(list(_default_yoloe26_prompts())),
        ],
        capture_output=True,
        check=False,
        cwd=str(_yoloe26_model_dir(runtime_python)),
        env=_yoloe26_worker_env(runtime_python),
        text=True,
    )
    if result.returncode != 0:
        print(json.dumps({"error": result.stderr.strip() or result.stdout.strip()}, indent=2, sort_keys=True))
        return 2
    print(result.stdout.strip())
    return 0


def _run_yoloe26_prototype_run(args: argparse.Namespace) -> int:
    vision_python = Path(args.vision_runtime_python).expanduser()
    bioclip_python = Path(args.bioclip_runtime_python).expanduser()
    if not vision_python.exists():
        print(json.dumps({"error": f"YOLOE-26 runtime Python not found: {vision_python}"}, indent=2, sort_keys=True))
        return 2
    if not bioclip_python.exists():
        print(json.dumps({"error": f"BioCLIP runtime Python not found: {bioclip_python}"}, indent=2, sort_keys=True))
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = pl.read_parquet(args.input)
    if args.limit is not None and args.limit > 0:
        records = records.head(args.limit)
        canonical_records_path = output_dir / "canonical_records_limited_yoloe26.parquet"
        records.write_parquet(canonical_records_path)
    else:
        canonical_records_path = Path(args.input)

    detections_path = output_dir / "object_detections_yoloe26.parquet"
    scores_path = output_dir / "object_bioclip_scores_yoloe26.parquet"
    joined_path = output_dir / "object_evidence_joined_yoloe26.parquet"
    summary_path = output_dir / "photo_evidence_summary_yoloe26.parquet"
    metrics_path = output_dir / "yoloe26_metrics.json"
    manifest_path = output_dir / "yoloe26_run_manifest.json"
    markdown_path = output_dir / "yoloe26_summary.md"

    from biominer.detection.yoloe26_detector import YoloE26SidecarObjectDetector

    detector = YoloE26SidecarObjectDetector(
        runtime_python=str(vision_python),
        checkpoint=args.checkpoint,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        prompt_classes=_yoloe26_prompt_classes(args),
    )
    detection_result = run_detection_pipeline(
        records=records.to_dicts(),
        detector=detector,
        output_path=detections_path,
        image_loader=lambda record: load_decoded_image_from_record(record, cache_root=args.cache_root),
        detection_policy=DetectionPolicy(
            backend="yoloe26",
            box_score_threshold=args.conf,
            nms_iou_threshold=args.iou,
            max_boxes_per_image=args.max_det,
            retain_debug_crops=args.retain_debug_crops,
        ),
        run_policy=DetectionRunPolicy(parquet_batch_rows=args.parquet_batch_rows),
    )

    context = SpeciesContext.read_json(args.species_context)
    candidate_set = _build_candidate_set_for_cli(
        context,
        command="dev vision yoloe26-prototype-run",
        species_candidate_path=args.species_candidates if getattr(args, "species_candidates", None) else None,
        records=records.to_dicts(),
    )
    if isinstance(candidate_set, int):
        return candidate_set
    runtime = _bioclip_runtime(runtime_python=bioclip_python)
    scorer = PersistentBioClipScorer(runtime=runtime, hf_cache_dir=args.hf_cache_dir, device=args.device)
    try:
        object_scorer = EphemeralCropBioClipScorer(
            scorer=scorer,
            image_loader=lambda item: load_decoded_image_from_record(item, cache_root=args.cache_root),
            temp_dir=args.crop_temp_dir,
            crop_padding_ratio=0.12,
            crop_target_px=336,
            model_id="bioclip2_5",
            model_version="bioclip2_5_huge",
            model_checkpoint=BIOCLIP_25_HUGE_REVISION,
            retain_debug_crops=args.retain_debug_crops,
            segmenter=make_segmenter("none"),
        )
        score_result = screen_object_detections(
            canonical_records=records,
            detections=detection_result.frame,
            species_context=context,
            candidate_set=candidate_set,
            scorer=object_scorer,
            output_path=scores_path,
            ablation_mode=args.ablation_mode,
            parquet_batch_rows=args.parquet_batch_rows,
        )
    finally:
        scorer.close()

    evidence_outputs = write_object_evidence_outputs(
        canonical_records_path=canonical_records_path,
        detections_path=detections_path,
        scores_path=scores_path,
        joined_output_path=joined_path,
        photo_summary_output_path=summary_path,
        species_context=context,
    )
    photo_summary = pl.read_parquet(evidence_outputs.photo_evidence_summary)
    metrics = _yoloe26_metrics(
        detection_result=detection_result,
        score_frame=score_result.frame,
        photo_summary=photo_summary,
        checkpoint=args.checkpoint,
        prompt_classes=_yoloe26_prompt_classes(args),
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "input": str(args.input),
        "canonical_records_path": str(canonical_records_path),
        "species_context": str(args.species_context),
        "species_candidates": args.species_candidates,
        "vision_runtime_python": str(vision_python),
        "bioclip_runtime_python": str(bioclip_python),
        "outputs": {
            "object_detections": str(detections_path),
            "object_bioclip_scores": str(scores_path),
            "object_evidence_joined": str(evidence_outputs.object_evidence_joined),
            "photo_evidence_summary": str(evidence_outputs.photo_evidence_summary),
            "metrics": str(metrics_path),
            "summary": str(markdown_path),
        },
        "also_whole_image_requested": bool(args.also_whole_image),
        "also_whole_image_status": "not_instrumented",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_yoloe26_summary_markdown(metrics=metrics, manifest=manifest), encoding="utf-8")
    print(json.dumps({"metrics": str(metrics_path), "manifest": str(manifest_path), **manifest["outputs"]}, indent=2, sort_keys=True))
    return 0


def _detect_boxes_backend(args: argparse.Namespace, records: list[dict[str, object]]):
    if args.backend == "fake":
        return FakeObjectDetector([_fake_detections_for_record(record) for record in records]), _blank_decoded_image
    if args.backend == "yoloe26":
        from biominer.detection.yoloe26_detector import YoloE26ObjectDetector, YoloE26SidecarObjectDetector

        prompts = _yoloe26_prompt_classes(args)
        kwargs = {
            "checkpoint": args.checkpoint,
            "device": args.device,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "prompt_classes": prompts,
        }
        if _use_vision_sidecar(args.runtime_python):
            return YoloE26SidecarObjectDetector(runtime_python=args.runtime_python, **kwargs), load_decoded_image_from_record
        return YoloE26ObjectDetector(**kwargs), load_decoded_image_from_record
    if args.backend == "yolo26":
        from biominer.detection.yolo26_detector import Yolo26ObjectDetector, Yolo26SidecarObjectDetector

        kwargs = {
            "checkpoint": _explicit_yolo26_checkpoint(args),
            "device": args.device,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
        }
        if _use_vision_sidecar(args.runtime_python):
            return Yolo26SidecarObjectDetector(runtime_python=args.runtime_python, **kwargs), load_decoded_image_from_record
        return Yolo26ObjectDetector(**kwargs), load_decoded_image_from_record
    raise RuntimeError(f"unsupported detection backend: {args.backend}")


def _explicit_yolo26_checkpoint(args: argparse.Namespace) -> str:
    checkpoint = str(getattr(args, "checkpoint", "") or "").strip()
    if not checkpoint or checkpoint == "yoloe-26s-seg.pt":
        raise ValueError("YOLO26 inference requires --checkpoint pointing to a user-provided coarse object checkpoint")
    return checkpoint


def _yoloe26_prompt_classes(args: argparse.Namespace) -> tuple[str, ...]:
    prompts = tuple(str(value) for value in getattr(args, "prompt_class", []) if str(value).strip())
    if prompts:
        return prompts
    return _default_yoloe26_prompts(include_hard_negative_prompts=bool(getattr(args, "include_hard_negative_prompts", True)))


def _default_yoloe26_prompts(*, include_hard_negative_prompts: bool = True) -> tuple[str, ...]:
    from biominer.detection.yoloe26_detector import default_yoloe26_prompts

    return default_yoloe26_prompts(include_hard_negative_prompts=include_hard_negative_prompts)


def _yoloe26_worker_env(runtime_python: str | Path) -> dict[str, str]:
    env = os.environ.copy()
    runtime_root = _runtime_root_from_python(runtime_python, fallback=YOLOE26_RUNTIME_ROOT)
    cache_root = runtime_root / "cache"
    defaults = {
        "HF_HOME": cache_root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache_root / "huggingface" / "hub",
        "TORCH_HOME": cache_root / "torch",
        "YOLO_CONFIG_DIR": cache_root / "ultralytics",
        "BIOMINER_YOLO26_MODEL_DIR": runtime_root / "models",
    }
    source_path = str(Path.cwd() / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source_path if not current else f"{source_path}{os.pathsep}{current}"
    for key, value in defaults.items():
        env.setdefault(key, str(value))
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def _yoloe26_model_dir(runtime_python: str | Path) -> Path:
    env = _yoloe26_worker_env(runtime_python)
    model_dir = Path(env["BIOMINER_YOLO26_MODEL_DIR"])
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _runtime_root_from_python(runtime_python: str | Path, *, fallback: Path) -> Path:
    path = Path(runtime_python).expanduser()
    if len(path.parents) >= 3 and path.parent.name == "bin":
        return path.parents[2]
    return fallback


def _yoloe26_metrics(
    *,
    detection_result: object,
    score_frame: pl.DataFrame,
    photo_summary: pl.DataFrame,
    checkpoint: str,
    prompt_classes: tuple[str, ...],
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
) -> dict[str, object]:
    detections = detection_result.frame
    detected = detections.filter(pl.col("detection_status") == "detected") if "detection_status" in detections.columns else detections
    species_scores = _numeric_values(score_frame, "species_top1_score")
    species_margins = _numeric_values(score_frame, "species_top1_margin")
    detector_scores = _numeric_values(detected, "detector_score")
    return {
        "metrics_kind": "heuristic_without_ground_truth",
        "records_seen": int(detection_result.records_seen),
        "images_loaded": int(detection_result.images_loaded),
        "image_failures": int(detection_result.image_failures),
        "detections_written": int(detection_result.detections_written),
        "no_detection_count": _count_equals(detections, "detection_status", "no_detection"),
        "crops_created": int(detection_result.crops_created),
        "crops_scored": int(score_frame.height),
        "detections_by_detector_label": _value_counts(detected, "detector_label"),
        "hard_negative_count": _count_equals(detected, "detector_label", "hard_negative"),
        "occurrence_bin_counts": _value_counts(score_frame, "occurrence_bin"),
        "photo_occurrence_bin_counts": _value_counts(photo_summary, "photo_occurrence_bin"),
        "mean_detector_score": _mean(detector_scores),
        "median_detector_score": _median(detector_scores),
        "mean_species_top1_score": _mean(species_scores),
        "median_species_top1_score": _median(species_scores),
        "mean_species_margin": _mean(species_margins),
        "median_species_margin": _median(species_margins),
        "top20_bioclip_top1_species": _top_counts(score_frame, "species_top1_scientific_name", limit=20),
        "top20_detector_labels": _top_counts(detected, "detector_label", limit=20),
        "checkpoint": checkpoint,
        "prompt_classes": list(prompt_classes),
        "device": device,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
    }


def _value_counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    if column not in frame.columns:
        return counts
    for value in frame.get_column(column).to_list():
        text = str(value or "")
        if not text:
            continue
        counts[text] = counts.get(text, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _top_counts(frame: pl.DataFrame, column: str, *, limit: int) -> list[dict[str, object]]:
    return [{"value": key, "count": count} for key, count in list(_value_counts(frame, column).items())[:limit]]


def _count_equals(frame: pl.DataFrame, column: str, expected: str) -> int:
    if column not in frame.columns:
        return 0
    return sum(1 for value in frame.get_column(column).to_list() if str(value or "") == expected)


def _numeric_values(frame: pl.DataFrame, column: str) -> list[float]:
    if column not in frame.columns:
        return []
    values: list[float] = []
    for value in frame.get_column(column).to_list():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _yoloe26_summary_markdown(*, metrics: dict[str, object], manifest: dict[str, object]) -> str:
    lines = [
        "# YOLOE-26 Prototype Summary",
        "",
        "Metrics are heuristic because no reviewed ground-truth boxes or species labels were supplied.",
        "",
        f"- Records seen: {metrics['records_seen']}",
        f"- Images loaded: {metrics['images_loaded']}",
        f"- Image failures: {metrics['image_failures']}",
        f"- Detections written: {metrics['detections_written']}",
        f"- Crops scored: {metrics['crops_scored']}",
        f"- Mean detector score: {metrics['mean_detector_score']}",
        f"- Mean BioCLIP top1 species score: {metrics['mean_species_top1_score']}",
        f"- Mean BioCLIP species margin: {metrics['mean_species_margin']}",
        "",
        "## Outputs",
        "",
    ]
    outputs = manifest.get("outputs", {})
    if isinstance(outputs, dict):
        for key, value in outputs.items():
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Detector Labels", "", json.dumps(metrics["detections_by_detector_label"], indent=2, sort_keys=True)])
    lines.extend(["", "## Occurrence Bins", "", json.dumps(metrics["occurrence_bin_counts"], indent=2, sort_keys=True)])
    return "\n".join(lines) + "\n"


def _detection_policy_from_args(args: argparse.Namespace) -> DetectionPolicy:
    profile = runtime_profile(args.profile).detection_policy if getattr(args, "profile", None) else DetectionPolicy()
    return DetectionPolicy(
        backend=args.backend,
        box_score_threshold=args.box_score_threshold if args.box_score_threshold is not None else profile.box_score_threshold,
        nms_iou_threshold=args.nms_iou_threshold if args.nms_iou_threshold is not None else profile.nms_iou_threshold,
        min_box_area_ratio=args.min_box_area_ratio if args.min_box_area_ratio is not None else profile.min_box_area_ratio,
        max_boxes_per_image=args.max_boxes_per_image if args.max_boxes_per_image is not None else profile.max_boxes_per_image,
        crop_padding_ratio=args.crop_padding_ratio if args.crop_padding_ratio is not None else profile.crop_padding_ratio,
        image_max_side_px=args.image_max_side_px if args.image_max_side_px is not None else profile.image_max_side_px,
        crop_target_px=args.crop_target_px if args.crop_target_px is not None else profile.crop_target_px,
        retain_debug_crops=args.retain_debug_crops or profile.retain_debug_crops,
        debug_crop_limit=args.debug_crop_limit if args.debug_crop_limit is not None else profile.debug_crop_limit,
    )


def _detection_run_policy_from_args(args: argparse.Namespace) -> DetectionRunPolicy:
    profile = runtime_profile(args.profile).run_policy if getattr(args, "profile", None) else DetectionRunPolicy()
    return DetectionRunPolicy(
        download_workers=args.download_workers if args.download_workers is not None else profile.download_workers,
        decode_workers=args.decode_workers if args.decode_workers is not None else profile.decode_workers,
        detector_workers=args.detector_workers if args.detector_workers is not None else profile.detector_workers,
        max_inflight_images=args.max_inflight_images if args.max_inflight_images is not None else profile.max_inflight_images,
        max_inflight_crops=args.max_inflight_crops if args.max_inflight_crops is not None else profile.max_inflight_crops,
        detector_batch_size=args.detector_batch_size if args.detector_batch_size is not None else profile.detector_batch_size,
        crop_batch_size=args.crop_batch_size if args.crop_batch_size is not None else profile.crop_batch_size,
        parquet_batch_rows=args.parquet_batch_rows if args.parquet_batch_rows is not None else profile.parquet_batch_rows,
    )


def _use_vision_sidecar(runtime_python: str) -> bool:
    path = Path(runtime_python)
    if not path.exists():
        return False
    return path.resolve() != Path(sys.executable).resolve()


def _run_detect_crop_preview(args: argparse.Namespace) -> int:
    detections = pl.read_parquet(args.detections)
    rows = _crop_preview_rows(detections, limit=max(1, args.limit))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".html", ".htm"}:
        output.write_text(_crop_preview_html(rows), encoding="utf-8")
        output_format = "html"
    else:
        output.write_text(json.dumps({"preview_rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
        output_format = "json"
    preview = {
        "detections": args.detections,
        "output": args.output,
        "format": output_format,
        "rows_seen": detections.height,
        "preview_rows": len(rows),
        "skipped_rows": detections.height - len(rows),
        "storage_policy": "remote_image_references_only",
    }
    print(json.dumps(preview, indent=2, sort_keys=True))
    return 0


def _crop_preview_rows(detections: pl.DataFrame, *, limit: int) -> list[dict[str, object]]:
    if detections.is_empty():
        return []
    rows: list[dict[str, object]] = []
    for row in detections.to_dicts():
        if len(rows) >= limit:
            break
        if str(row.get("detection_status") or "") != "detected":
            continue
        image_url = str(row.get("image_url") or "")
        bbox = _normalised_bbox(row.get("bbox_xyxyn"))
        if not image_url or bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        rows.append(
            {
                "source": str(row.get("source") or ""),
                "flickr_photo_id": str(row.get("flickr_photo_id") or ""),
                "image_url": image_url,
                "detection_id": str(row.get("detection_id") or ""),
                "crop_hash": str(row.get("crop_hash") or ""),
                "detector_label": str(row.get("detector_label") or ""),
                "detector_score": row.get("detector_score"),
                "bbox_xyxyn": [x1, y1, x2, y2],
                "left_pct": _percent(x1),
                "top_pct": _percent(y1),
                "width_pct": _percent(max(0.0, x2 - x1)),
                "height_pct": _percent(max(0.0, y2 - y1)),
            }
        )
    return rows


def _normalised_bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (max(0.0, min(1.0, float(item))) for item in value)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _percent(value: float) -> str:
    return f"{value * 100:.4f}%"


def _crop_preview_html(rows: list[dict[str, object]]) -> str:
    cards = "\n".join(_crop_preview_card(row) for row in rows)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BioMiner Crop Preview</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2933; background: #f7f8fa; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }}
.card {{ background: #fff; border: 1px solid #d7dce2; border-radius: 6px; padding: 12px; }}
.image-wrap {{ position: relative; background: #111; overflow: hidden; }}
.image-wrap img {{ display: block; width: 100%; height: auto; }}
.bbox {{ position: absolute; border: 2px solid #f97316; box-sizing: border-box; }}
.meta {{ font-size: 12px; line-height: 1.45; overflow-wrap: anywhere; }}
</style>
</head>
<body>
<h1>BioMiner Crop Preview</h1>
<p>{len(rows)} detected crop previews. Images are referenced remotely; no local image archive is created.</p>
<div class="grid">
{cards}
</div>
</body>
</html>
"""


def _crop_preview_card(row: dict[str, object]) -> str:
    image_url = escape(str(row["image_url"]), quote=True)
    title = escape(f"{row['source']}:{row['flickr_photo_id']}", quote=False)
    detection_id = escape(str(row["detection_id"]), quote=False)
    crop_hash = escape(str(row["crop_hash"]), quote=False)
    label = escape(str(row["detector_label"]), quote=False)
    score = row.get("detector_score")
    score_text = "" if score is None else f"{float(score):.4f}"
    return f"""<article class="card">
<div class="image-wrap">
<img src="{image_url}" alt="{title}">
<div class="bbox" style="left: {row['left_pct']}; top: {row['top_pct']}; width: {row['width_pct']}; height: {row['height_pct']};"></div>
</div>
<div class="meta">
<strong>{title}</strong><br>
detection_id: {detection_id}<br>
crop_hash: {crop_hash}<br>
label: {label} score: {score_text}
</div>
</article>"""


def _run_detect_eval(args: argparse.Namespace) -> int:
    predictions = pl.read_parquet(args.predictions).to_dicts()
    truth = pl.read_parquet(args.ground_truth).to_dicts() if args.ground_truth else None
    report = evaluate_xie_style(
        predictions=predictions,
        ground_truth=truth,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
    )
    report = {
        **report,
        "iou_threshold": args.iou_threshold,
        "score_threshold": args.score_threshold,
    }
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
    candidate_set = _build_candidate_set_for_cli(
        context,
        command="vision score",
        species_candidate_path=args.species_candidates if getattr(args, "species_candidates", None) else None,
        records=records.to_dicts(),
        geospatial_scope=str(args.geo_prior_table) if getattr(args, "geo_prior_table", None) else None,
        geo_prior_table=geo_prior_table,
    )
    if isinstance(candidate_set, int):
        return candidate_set
    runtime = _bioclip_runtime(runtime_python=runtime_python)
    scorer = PersistentBioClipScorer(runtime=runtime, hf_cache_dir=args.hf_cache_dir, device=args.device)
    try:
        cache_error = _validate_object_image_cache_args(args=args, modes=(args.ablation_mode,))
        if cache_error is not None:
            print(json.dumps({"error": cache_error}, indent=2, sort_keys=True))
            return 2
        text_cache_payload = _prepare_candidate_text_embedding_cache_if_requested(args=args, candidate_set=candidate_set, scorer=scorer)
        object_cache_payload = _prepare_object_image_embedding_cache_if_requested(
            args=args,
            records=records,
            detections=detections,
            scorer=scorer,
        )
        object_scorer = _object_scorer_for_args(
            args=args,
            scorer=scorer,
            candidate_set_id=candidate_set.candidate_set_id,
        )
        result = screen_object_detections(
            canonical_records=records,
            detections=detections,
            species_context=context,
            candidate_set=candidate_set,
            scorer=object_scorer,
            output_path=args.output,
            ablation_mode=args.ablation_mode,
            geo_prior_table=geo_prior_table,
            parquet_batch_rows=args.parquet_batch_rows,
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
                "score_batches_written": result.score_batches_written,
                "primary_visual_classifier": result.visual_classifier,
                "visual_mode": result.visual_mode,
                "visual_mode_status": result.visual_mode_status,
                "segmentation_unavailable_count": result.segmentation_unavailable_count,
                "segmentation_unavailable_reason": result.segmentation_unavailable_reason,
                "candidate_set_id": candidate_set.candidate_set_id,
                "scorer": "cached_object_embedding" if object_cache_payload is not None else "ephemeral_crop_bioclip",
                **({"candidate_text_embedding_cache": text_cache_payload} if text_cache_payload is not None else {}),
                **({"object_image_embedding_cache": object_cache_payload} if object_cache_payload is not None else {}),
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
    candidate_set = _build_candidate_set_for_cli(
        context,
        command="vision ablate",
        species_candidate_path=args.species_candidates if getattr(args, "species_candidates", None) else None,
        records=records.to_dicts(),
        geospatial_scope=str(args.geo_prior_table) if getattr(args, "geo_prior_table", None) else None,
        geo_prior_table=geo_prior_table,
    )
    if isinstance(candidate_set, int):
        return candidate_set
    modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())
    runtime = _bioclip_runtime(runtime_python=runtime_python)
    scorer = PersistentBioClipScorer(runtime=runtime, hf_cache_dir=args.hf_cache_dir, device=args.device)
    try:
        cache_error = _validate_object_image_cache_args(args=args, modes=modes)
        if cache_error is not None:
            print(json.dumps({"error": cache_error}, indent=2, sort_keys=True))
            return 2
        text_cache_payload = _prepare_candidate_text_embedding_cache_if_requested(args=args, candidate_set=candidate_set, scorer=scorer)
        object_cache_payload = _prepare_object_image_embedding_cache_if_requested(
            args=args,
            records=records,
            detections=detections,
            scorer=scorer,
        )
        object_scorer = _object_scorer_for_args(
            args=args,
            scorer=scorer,
            candidate_set_id=candidate_set.candidate_set_id,
        )
        report = run_object_ablations(
            canonical_records=records,
            detections=detections,
            species_context=context,
            candidate_set=candidate_set,
            scorer=object_scorer,
            output_dir=args.output_dir,
            modes=modes,  # type: ignore[arg-type]
            geo_prior_table=geo_prior_table,
            parquet_batch_rows=args.parquet_batch_rows,
        )
    finally:
        scorer.close()
    print(
        json.dumps(
            {
                "output_dir": str(report.output_dir),
                "primary_visual_classifier": PRIMARY_VISUAL_CLASSIFIER,
                **report.report,
                **({"candidate_text_embedding_cache": text_cache_payload} if text_cache_payload is not None else {}),
                **({"object_image_embedding_cache": object_cache_payload} if object_cache_payload is not None else {}),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_bioclip_join_object_evidence(args: argparse.Namespace) -> int:
    context = SpeciesContext.read_json(args.species_context) if getattr(args, "species_context", None) else None
    outputs = write_object_evidence_outputs(
        canonical_records_path=args.input,
        detections_path=args.detections,
        scores_path=args.scores,
        joined_output_path=args.joined_output,
        photo_summary_output_path=args.photo_summary_output,
        species_context=context,
    )
    print(
        json.dumps(
            {
                "object_evidence_joined": str(outputs.object_evidence_joined),
                "photo_evidence_summary": str(outputs.photo_evidence_summary),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _build_candidate_set_for_cli(
    context: SpeciesContext,
    *,
    command: str,
    species_candidate_path: str | Path | None = None,
    records: list[dict[str, Any]] | None = None,
    geospatial_scope: str | None = None,
    geo_prior_table: pl.DataFrame | None = None,
):
    try:
        return build_candidate_set(
            context,
            species_candidate_path=species_candidate_path,
            records=records,
            geospatial_scope=geospatial_scope,
            geo_prior_table=geo_prior_table,
        )
    except ValueError as exc:
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "command": command,
                    "hint": "Provide --species-candidates with same-genus/same-family registry candidates, "
                    "or include query/geospatial provenance that expands beyond the target species.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2


def _optional_parquet(path: str | Path | None) -> pl.DataFrame | None:
    if not path:
        return None
    return pl.read_parquet(path)


def _prepare_candidate_text_embedding_cache_if_requested(
    *,
    args: argparse.Namespace,
    candidate_set: object,
    scorer: PersistentBioClipScorer,
) -> dict[str, object] | None:
    cache_path = getattr(args, "candidate_text_embedding_cache", None)
    if not cache_path:
        return None
    update = prepare_candidate_text_embedding_cache(
        candidate_set,  # type: ignore[arg-type]
        cache_path,
        model_id="bioclip2_5",
        model_checkpoint=BIOCLIP_25_HUGE_REVISION,
        embed_labels=scorer.embed_text_labels,
        batch_size=args.text_embedding_batch_size,
    )
    return {
        "output_path": str(update.output_path),
        "rows_total": update.rows_total,
        "rows_added": update.rows_added,
        "rows_reused": update.rows_reused,
        "embeddings_computed": update.embeddings_computed,
        "text_embedding_batch_size": args.text_embedding_batch_size,
    }


def _validate_object_image_cache_args(*, args: argparse.Namespace, modes: tuple[str, ...]) -> str | None:
    if not getattr(args, "object_image_embedding_cache", None):
        return None
    if not getattr(args, "candidate_text_embedding_cache", None):
        return "--object-image-embedding-cache requires --candidate-text-embedding-cache for cached dot-product scoring"
    unsupported = sorted(set(modes) - {"detector_crop"})
    if unsupported:
        return "--object-image-embedding-cache is only valid for detector_crop mode; unsupported modes: " + ",".join(unsupported)
    return None


def _prepare_object_image_embedding_cache_if_requested(
    *,
    args: argparse.Namespace,
    records: pl.DataFrame,
    detections: pl.DataFrame,
    scorer: PersistentBioClipScorer,
) -> dict[str, object] | None:
    cache_path = getattr(args, "object_image_embedding_cache", None)
    if not cache_path:
        return None
    materialized = materialize_detector_crop_inputs(
        canonical_records=records,
        detections=detections,
        image_loader=lambda item: load_decoded_image_from_record(item, cache_root=args.cache_root),
        temp_dir=args.crop_temp_dir,
        crop_padding_ratio=args.crop_padding_ratio,
        crop_target_px=args.crop_target_px,
    )
    try:
        update = prepare_object_image_embedding_cache(
            materialized.rows,
            cache_path,
            model_id="bioclip2_5",
            model_checkpoint=BIOCLIP_25_HUGE_REVISION,
            crop_path_by_hash=materialized.crop_path_by_hash,
            embed_image_paths=scorer.embed_image_paths,
        )
    finally:
        materialized.cleanup()
    return {
        "output_path": str(update.output_path),
        "rows_total": update.rows_total,
        "rows_added": update.rows_added,
        "rows_reused": update.rows_reused,
        "embeddings_computed": update.embeddings_computed,
    }


def _object_scorer_for_args(
    *,
    args: argparse.Namespace,
    scorer: PersistentBioClipScorer,
    candidate_set_id: str,
) -> object:
    if getattr(args, "object_image_embedding_cache", None):
        return CachedObjectEmbeddingScorer(
            text_embeddings=read_embedding_cache(args.candidate_text_embedding_cache),
            image_embeddings=read_embedding_cache(args.object_image_embedding_cache),
            candidate_set_id=candidate_set_id,
            model_id="bioclip2_5",
            model_version="bioclip2_5_huge",
            model_checkpoint=BIOCLIP_25_HUGE_REVISION,
        )
    return EphemeralCropBioClipScorer(
        scorer=scorer,
        image_loader=lambda item: load_decoded_image_from_record(item, cache_root=args.cache_root),
        temp_dir=args.crop_temp_dir,
        crop_padding_ratio=args.crop_padding_ratio,
        crop_target_px=args.crop_target_px,
        model_id="bioclip2_5",
        model_version="bioclip2_5_huge",
        model_checkpoint=BIOCLIP_25_HUGE_REVISION,
        retain_debug_crops=args.retain_debug_crops,
        segmenter=make_segmenter(getattr(args, "segmenter", "none")),
    )


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


_YOLOE26_RUNTIME_CHECK_SCRIPT = r"""
from __future__ import annotations

import importlib.metadata
import json
import sys

import torch
from ultralytics import YOLOE

requested = sys.argv[1]
checkpoint = sys.argv[2]
if requested == "auto":
    if torch.cuda.is_available():
        resolved = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        resolved = "mps"
    else:
        resolved = "cpu"
elif requested == "cuda" and not torch.cuda.is_available():
    raise SystemExit("CUDA was requested but is not available")
elif requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    raise SystemExit("MPS was requested but is not available")
else:
    resolved = requested

model = YOLOE(checkpoint)
model.set_classes(["butterfly", "moth", "caterpillar", "pupa"])
cuda_device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print(json.dumps({
    "runtime_python": sys.executable,
    "checkpoint": checkpoint,
    "device_requested": requested,
    "device_resolved": resolved,
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_name": cuda_device_name,
    "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    "ultralytics_version": importlib.metadata.version("ultralytics"),
    "yoloe_import": True,
    "checkpoint_load": True,
    "set_classes": True,
}, sort_keys=True))
"""


_YOLOE26_PREFETCH_SCRIPT = r"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import sys

import torch
from ultralytics import YOLOE

device = sys.argv[1]
checkpoint = sys.argv[2]
prompts = json.loads(sys.argv[3])
model = YOLOE(checkpoint)
model.set_classes(prompts)
model_dir = Path(os.environ.get("BIOMINER_YOLO26_MODEL_DIR") or Path.cwd())
checkpoint_path = model_dir / checkpoint
print(json.dumps({
    "runtime_python": sys.executable,
    "checkpoint": checkpoint,
    "checkpoint_path": str(checkpoint_path if checkpoint_path.exists() else checkpoint),
    "model_cache_dir": str(model_dir),
    "device_requested": device,
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    "ultralytics_version": importlib.metadata.version("ultralytics"),
    "prompt_class_count": len(prompts),
}, sort_keys=True))
"""


_YOLOE26_SMOKE_SCRIPT = r"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

from biominer.detection.detector_base import DecodedImage
from biominer.detection.yoloe26_detector import YoloE26ObjectDetector

device = sys.argv[1]
checkpoint = sys.argv[2]
output_dir = Path(sys.argv[3])
image_path = Path(sys.argv[4]) if sys.argv[4] else None
prompts = tuple(json.loads(sys.argv[5]))
output_dir.mkdir(parents=True, exist_ok=True)

if image_path is not None:
    image = Image.open(image_path).convert("RGB")
    source_uri = str(image_path)
    synthetic = False
else:
    image = Image.new("RGB", (32, 32), (240, 240, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 24, 24), outline=(80, 80, 80), width=2)
    source_uri = "synthetic://blank-smoke-image"
    synthetic = True

decoded = DecodedImage(width=image.width, height=image.height, mode="RGB", data=image.tobytes(), source_uri=source_uri)
detector = YoloE26ObjectDetector(checkpoint=checkpoint, device=device, prompt_classes=prompts)
detections = detector.detect_batch([decoded])[0]
preview_rows = [
    {
        "label": item.label,
        "score": item.score,
        "bbox_xyxy": list(item.bbox_xyxy),
        "objectness_score": item.objectness_score,
    }
    for item in detections
]
preview_path = output_dir / "detections_preview.json"
preview_path.write_text(json.dumps({"synthetic_image": synthetic, "detections": preview_rows}, indent=2, sort_keys=True), encoding="utf-8")
if detections:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for item in detections:
        draw.rectangle(item.bbox_xyxy, outline=(255, 100, 0), width=2)
    annotated.save(output_dir / "annotated_preview.jpg")
print(json.dumps({
    "output_dir": str(output_dir),
    "detections_preview": str(preview_path),
    "detections": len(detections),
    "synthetic_image": synthetic,
    "checkpoint": checkpoint,
}, sort_keys=True))
"""


def main() -> None:
    load_runtime_secrets_env()
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
