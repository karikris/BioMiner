from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
from html import escape
import io
import json
import logging
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any

import polars as pl

from biominer.bioclip.classification_modes import (
    DEFAULT_CLASSIFICATION_MODE,
    DEFAULT_RANK_BEAM_WIDTH,
    DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    DEFAULT_SPECIES_RERANK_TOP_K,
    HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    SUPPORTED_CLASSIFICATION_MODES,
    is_build_week_prototype_classification,
    normalize_classification_mode,
)
from biominer.bioclip.model_registry import BioClipRuntime, ModelConfig
from biominer.bioclip.path_taxonomy_store import PathTaxonomyStore
from biominer.bioclip.object_runner import (
    write_object_evidence_outputs,
)
from biominer.benchmarks.path_cascade import run_path_cascade_benchmark
from biominer.benchmarks.vision_live import (
    LiveM5ProBenchmarkRequest,
    run_live_m5pro_benchmark,
    validate_live_m5pro_benchmark_request,
)
from biominer.benchmarks.vision_plumbing import run_rolling_worker_benchmark_matrix, run_vision_plumbing_benchmark
from biominer.detection.evaluate import evaluate_xie_style
from biominer.detection.policy import (
    VisionRuntimeSettings,
    vision_runtime_settings,
)
from biominer.evaluation.labels import (
    normalize_reviewed_label_frame,
    read_reviewed_labels,
    validate_reviewed_label_frame,
)
from biominer.evaluation.review_queue import build_hierarchical_review_queue
from biominer.evaluation.reports import write_evaluation_report, write_evaluation_report_to_storage
from biominer.evaluation.sampling import (
    EvaluationSamplingConfig,
    materialize_evaluation_sampling_frame,
)
from biominer.evaluation.xie_style import EVALUATION_PROFILE as XIE_STYLE_EVALUATION_PROFILE
from biominer.evaluation.xie_style import evaluate_xie_style_hierarchical
from biominer.flickr_fetch.query_planner import load_registry_flickr_queries
from biominer.flickr_fetch.australia import (
    AUSTRALIA_FLICKR_BBOX,
    build_australia_presence,
    compile_australia_query_plan,
)
from biominer.flickr_fetch.metadata_poller import SOFT_API_CALLS_PER_HOUR, MetadataPollState, poll_once
from biominer.registry.audit import audit_registry
from biominer.registry.build import build_registry
from biominer.registry.compiler import compile_registry_fixture, compile_registry_parquet_source
from biominer.registry.checklistbank import harvest_col_xr_names, harvest_col_xr_taxonomy
from biominer.registry.col_xr import extract_col_xr_snapshot
from biominer.registry.enrichment import DEFAULT_ENRICHMENT_SOURCES, INATURALIST_DAILY_REQUEST_LIMIT, build_enrichment_sources_from_registry, compile_enriched_registry
from biominer.registry.gbif import GBIFClient
from biominer.registry.gbif_source import build_gbif_source_snapshot
from biominer.registry.hierarchy_enrichment import (
    harvest_gbif_genus_evidence,
    harvest_open_tree_genus_evidence,
)
from biominer.registry.scope import load_scope
from biominer.registry.publish import publish_registry
from biominer.registry.translation_harvester import (
    MYMEMORY_MONTHLY_BANDWIDTH_MB_LIMIT,
    MYMEMORY_MONTHLY_INPUT_WORD_LIMIT,
    MYMEMORY_MONTHLY_REQUEST_LIMIT,
    MYMEMORY_RESPONSE_BYTE_RESERVATION,
)
from biominer.registry.translation_sources import DEFAULT_TRANSLATION_SOURCES, DEFAULT_TRANSLATION_TARGET_LOCALES_JSON
from biominer.references.negative_manifest import (
    publish_curated_visual_domain_negative_manifest,
)
from biominer.references.readiness import (
    ReferenceBankReadinessPolicy,
    ReferenceModelInputIdentity,
    build_reference_bank_readiness,
    publish_reference_bank_readiness,
    reference_readiness_allows_vision,
)
from biominer.references.review import (
    advance_reference_review_history_head,
    build_reference_review_queue,
    import_reference_review_decisions,
    initialize_reference_review_history_head,
    validate_reference_review_history_head,
    validate_reference_review_history_head_destination,
    validate_reference_review_packet_artifact,
    write_reference_review_export,
    write_reference_review_import,
)
from biominer.reference_workflow_cli import (
    ReferenceWorkflowRuntimeDefaults,
    add_reference_workflow_parsers,
    is_reference_workflow_command,
    run_reference_workflow_command,
)
from biominer.runtime_paths import BASE_PATH, BIOCLIP25_DIR, YOLOE26_DIR
from biominer.run import (
    ADAPTIVE_REFERENCE_PRODUCTION_STAGES,
    REFERENCE_FIRST_PRODUCTION_STAGES,
    ProductionRunOrchestrator,
    ProductionRunRequest,
    RunStage,
)
from biominer.run.adaptive_config import (
    DEFAULT_INITIAL_SCORING_MODE,
    DEFAULT_REFERENCE_ADMISSION_MODE,
    DEFAULT_REFERENCE_SOURCE,
    REFERENCE_ADMISSION_MODES,
)
from biominer.run.dynamic_pool_cli import (
    add_dynamic_pooling_parsers,
    run_dynamic_pooling_command,
)
from biominer.run.stages import DEFAULT_PRODUCTION_STAGES
from biominer.secrets_loader import load_runtime_secrets_env
from biominer.species.context import SpeciesContext
from biominer.config import ConfigError, create_storage_backend, create_workstore, load_biominer_config, redact_config, redact_text, validate_config
from biominer.storage.handoff import (
    build_handoff_bundle,
    receive_handoff_bundle,
    upload_handoff_bundle,
)
from biominer.storage.parquet import write_parquet
from biominer.storage.uri import is_cloud_uri, join_uri, normalize_local_uri


STANDARD_EVALUATION_PROFILE = "standard"
XIE_STYLE_METRICS_FILE = "xie_style_metrics.json"
BIOCLIP_25_HUGE_REPO_ID = "imageomics/bioclip-2.5-vith14"
BIOCLIP_25_HUGE_REVISION = "191d741545e4c741cdef4b22c6eb69c945c1e592"
BIOCLIP_OPENCLIP_VERSION = "3.3.0"
BIOMINER_BASE_PATH = BASE_PATH
YOLOE26_RUNTIME_ROOT = YOLOE26_DIR
YOLOE26_RUNTIME_PYTHON = str(YOLOE26_RUNTIME_ROOT / "venv" / "bin" / "python")
YOLOE26_MODEL_DIR = str(YOLOE26_RUNTIME_ROOT / "models")
YOLOE26_CACHE_ROOT = str(YOLOE26_RUNTIME_ROOT / "cache")
BIOCLIP25_RUNTIME_ROOT = BIOCLIP25_DIR
BIOCLIP_RUNTIME_PYTHON = str(BIOCLIP25_RUNTIME_ROOT / "venv" / "bin" / "python")
BIOCLIP_HF_CACHE_DIR = str(BIOCLIP25_RUNTIME_ROOT / "cache" / "huggingface")
BIOCLIP_WORKER_SCRIPT = str(
    Path(__file__).with_name("bioclip") / "bioclip_worker.py"
)
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
    "geographic_spread": RunStage.GEOGRAPHIC_SPREAD,
    "flickr_geo_clustering": RunStage.FLICKR_GEO_CLUSTERING,
    "regional_candidates": RunStage.REGIONAL_CANDIDATE_GENERATION,
    "regional_candidate_generation": RunStage.REGIONAL_CANDIDATE_GENERATION,
    "reference_metadata": RunStage.REFERENCE_METADATA,
    "reference_media": RunStage.REFERENCE_MEDIA,
    "reference_review": RunStage.REFERENCE_REVIEW,
    "reference_embeddings": RunStage.REFERENCE_EMBEDDINGS,
    "reference_prototypes": RunStage.REFERENCE_PROTOTYPES,
    "classifier_training": RunStage.CLASSIFIER_TRAINING,
    "classifier_calibration": RunStage.CLASSIFIER_CALIBRATION,
    "reference_readiness": RunStage.REFERENCE_READINESS,
    "flickr_detection": RunStage.FLICKR_DETECTION,
    "flickr_embedding": RunStage.FLICKR_EMBEDDING,
    "target_aware_scoring": RunStage.TARGET_AWARE_SCORING,
    "evidence": RunStage.EVIDENCE,
    "evaluation": RunStage.EVALUATION,
    "detect": RunStage.DETECT_OBJECTS,
    "detect_objects": RunStage.DETECT_OBJECTS,
    "score": RunStage.SCORE_BIOCLIP,
    "score_bioclip": RunStage.SCORE_BIOCLIP,
    "join": RunStage.JOIN_EVIDENCE,
    "join_evidence": RunStage.JOIN_EVIDENCE,
    "summarize": RunStage.SUMMARIZE,
    "summary": RunStage.SUMMARIZE,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biominer")
    parser.add_argument("--config")
    parser.add_argument("--version", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    dynamic_pooling = subparsers.add_parser("dynamic-pooling")
    dynamic_pooling_subparsers = dynamic_pooling.add_subparsers(
        dest="dynamic_pooling_command"
    )
    add_dynamic_pooling_parsers(dynamic_pooling_subparsers)
    evidence = subparsers.add_parser("evidence")
    evidence_subparsers = evidence.add_subparsers(dest="evidence_command")
    evidence_join = evidence_subparsers.add_parser("join")
    _add_object_evidence_join_args(evidence_join)
    evidence_join.add_argument("--species-context")
    evaluation = subparsers.add_parser("evaluation")
    evaluation_subparsers = evaluation.add_subparsers(dest="evaluation_command")
    evaluation_classify = evaluation_subparsers.add_parser("classify")
    evaluation_input = evaluation_classify.add_mutually_exclusive_group(required=True)
    evaluation_input.add_argument("--object-scores")
    evaluation_input.add_argument("--object-evidence")
    evaluation_classify.add_argument("--reviewed-labels", required=True)
    evaluation_classify.add_argument("--output-dir", required=True)
    evaluation_classify.add_argument("--storage-backend", choices=("local", "s3"), default="local")
    evaluation_classify.add_argument("--config")
    evaluation_classify.add_argument("--write-charts", action="store_true")
    evaluation_classify.add_argument(
        "--evaluation-profile",
        choices=(STANDARD_EVALUATION_PROFILE, XIE_STYLE_EVALUATION_PROFILE),
        default=STANDARD_EVALUATION_PROFILE,
    )
    evaluation_review_queue = evaluation_subparsers.add_parser("review-queue")
    evaluation_review_queue.add_argument("--object-evidence", required=True)
    evaluation_review_queue.add_argument("--photo-summary")
    evaluation_review_queue.add_argument("--output", required=True)
    evaluation_review_queue.add_argument("--max-rows", type=int)
    evaluation_sampling = evaluation_subparsers.add_parser("build-sampling-frame")
    evaluation_sampling.add_argument("--candidates", required=True)
    evaluation_sampling.add_argument("--geo-assignments", required=True)
    evaluation_sampling.add_argument("--query-state-db", required=True)
    evaluation_sampling.add_argument("--object-scores")
    evaluation_sampling.add_argument("--competitor-taxa")
    evaluation_sampling.add_argument("--target-text-term", action="append", default=[])
    evaluation_sampling.add_argument("--random-seed", type=int, default=42)
    evaluation_sampling.add_argument("--run-id")
    evaluation_sampling.add_argument("--output", required=True)
    references = subparsers.add_parser("references")
    references_subparsers = references.add_subparsers(dest="references_command")
    add_reference_workflow_parsers(
        references_subparsers,
        runtime_defaults=ReferenceWorkflowRuntimeDefaults(
            runtime_python=BIOCLIP_RUNTIME_PYTHON,
            hf_cache_dir=BIOCLIP_HF_CACHE_DIR,
        ),
    )
    references_export = references_subparsers.add_parser("export-review-queue")
    references_export.add_argument("--acquisition-selections", required=True)
    references_export.add_argument("--observations", required=True)
    references_export.add_argument("--media-candidates", required=True)
    references_export.add_argument("--media-objects", required=True)
    references_export.add_argument("--duplicate-relationships", required=True)
    references_export.add_argument("--deduplication-report", required=True)
    references_export.add_argument("--reference-bank-version", required=True)
    references_export.add_argument("--output-dir", required=True)
    references_export.add_argument("--history-head", required=True)
    references_export.add_argument("--run-id")
    references_export.add_argument("--include-research-only", action="store_true")
    references_import = references_subparsers.add_parser("import-review-decisions")
    references_import.add_argument("--review-queue", required=True)
    references_import.add_argument("--queue-provenance", required=True)
    references_import.add_argument("--decisions", required=True)
    references_import.add_argument("--existing-decisions", required=True)
    references_import.add_argument("--prior-review-report", required=True)
    references_import.add_argument("--history-head", required=True)
    references_import.add_argument("--output-dir", required=True)
    references_import.add_argument("--run-id")
    references_negatives = references_subparsers.add_parser(
        "compile-visual-domain-negatives"
    )
    references_negatives.add_argument("--source-manifest", required=True)
    references_negatives.add_argument("--output-dir", required=True)
    references_negatives.add_argument("--run-id")
    references_readiness = references_subparsers.add_parser("validate-readiness")
    references_readiness.add_argument("--candidate-species", required=True)
    references_readiness.add_argument("--acquisition-plan", required=True)
    references_readiness.add_argument("--acquisition-selections", required=True)
    references_readiness.add_argument("--observations", required=True)
    references_readiness.add_argument("--media-candidates", required=True)
    references_readiness.add_argument("--media-objects", required=True)
    references_readiness.add_argument("--duplicate-relationships", required=True)
    references_readiness.add_argument("--deduplication-report", required=True)
    references_readiness.add_argument("--review-queue", required=True)
    references_readiness.add_argument("--queue-provenance", required=True)
    references_readiness.add_argument("--review-decisions", required=True)
    references_readiness.add_argument("--split-assignments", required=True)
    references_readiness.add_argument("--readiness-policy", required=True)
    references_readiness.add_argument("--model-identity", required=True)
    references_readiness.add_argument("--registry-version", required=True)
    references_readiness.add_argument("--reference-bank-version", required=True)
    references_readiness.add_argument("--output-dir", required=True)
    references_readiness.add_argument("--run-id")
    bioclip = subparsers.add_parser("bioclip")
    bioclip_subparsers = bioclip.add_subparsers(dest="bioclip_command")
    bioclip_screen = bioclip_subparsers.add_parser("screen")
    bioclip_screen.add_argument("--input", required=True)
    bioclip_screen.add_argument("--output-dir", required=True)
    bioclip_screen.add_argument("--registry-dir", required=True, help="Unified registry containing species_paths.parquet")
    bioclip_screen.add_argument("--runtime-python", default=BIOCLIP_RUNTIME_PYTHON)
    bioclip_screen.add_argument("--taxonomy-text-embedding-cache")
    bioclip_screen.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    bioclip_screen.add_argument("--dry-run", action="store_true")
    bioclip_evidence = bioclip_subparsers.add_parser("prototype-evidence")
    bioclip_evidence.add_argument("--config", required=True)
    registry = subparsers.add_parser("registry")
    registry_subparsers = registry.add_subparsers(dest="registry_command")
    registry_build = registry_subparsers.add_parser("build")
    registry_build.add_argument("--output-dir", required=True)
    registry_build.add_argument("--registry-version", required=True)
    registry_build.add_argument("--scope-json", default="config/butterfly_scope.json")
    registry_build.add_argument("--source-json")
    registry_build.add_argument("--col-xr-archive", help="Pinned CoL XR Darwin Core archive for dataset 315557")
    registry_build.add_argument(
        "--col-xr-parquet-source",
        help="Parquet source directory produced by registry harvest-col-xr",
    )
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
    registry_build.add_argument("--mymemory-monthly-request-limit", type=int, default=MYMEMORY_MONTHLY_REQUEST_LIMIT)
    registry_build.add_argument("--mymemory-monthly-input-word-limit", type=int, default=MYMEMORY_MONTHLY_INPUT_WORD_LIMIT)
    registry_build.add_argument("--mymemory-monthly-bandwidth-mb-limit", type=int, default=MYMEMORY_MONTHLY_BANDWIDTH_MB_LIMIT)
    registry_build.add_argument("--mymemory-response-byte-reservation", type=int, default=MYMEMORY_RESPONSE_BYTE_RESERVATION)
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
    registry_harvest_col = registry_subparsers.add_parser("harvest-col-xr")
    registry_harvest_col.add_argument("--output-dir", default="data/sources/col/COL26.6-XR")
    registry_harvest_col.add_argument("--scope-json", default="config/butterfly_scope.json")
    registry_harvest_col.add_argument("--workers", type=int, default=32)
    registry_harvest_col.add_argument("--max-retries", type=int, default=4)
    registry_harvest_col.add_argument("--skip-names", action="store_true")
    registry_harvest_col.add_argument("--name-limit", type=int, default=0)
    registry_enrich_hierarchy = registry_subparsers.add_parser("enrich-hierarchy")
    registry_enrich_hierarchy.add_argument("--registry-dir", required=True)
    registry_enrich_hierarchy.add_argument("--source-dir", required=True)
    registry_enrich_hierarchy.add_argument("--sources", default="gbif,open_tree")
    registry_enrich_hierarchy.add_argument("--workers", type=int, default=8)
    registry_enrich_hierarchy.add_argument("--max-retries", type=int, default=4)
    registry_audit = registry_subparsers.add_parser("audit")
    registry_audit.add_argument("--registry-dir", required=True)
    registry_audit.add_argument("--report-dir", default="reports")
    registry_publish = registry_subparsers.add_parser("publish")
    registry_publish.add_argument("--registry-dir", required=True)
    registry_publish.add_argument("--output-dir", default="data/registry/current")
    registry_publish.add_argument("--replace-existing", action="store_true")
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
    dev_flickr = dev_subparsers.add_parser("flickr")
    dev_flickr_subparsers = dev_flickr.add_subparsers(dest="flickr_command")
    dev_poll_once = dev_flickr_subparsers.add_parser("poll-once")
    _add_poll_once_args(dev_poll_once)
    australia_live = dev_flickr_subparsers.add_parser("australia-live")
    australia_live.add_argument("--registry-dir", default="data/registry/current")
    australia_live.add_argument("--output-dir", default="data/registry/current")
    australia_live.add_argument("--presence-state-db", default="data/state/gbif_australia_presence.sqlite")
    australia_live.add_argument("--state-db", default="data/state/flickr_australia.sqlite")
    australia_live.add_argument("--raw-root", default="staging/flickr/australia/raw")
    australia_live.add_argument("--evidence-output", default="staging/flickr/australia/evidence.parquet")
    australia_live.add_argument("--api-key-env", default="FLICKR_API_KEY")
    australia_live.add_argument("--place-id")
    australia_live.add_argument("--woe-id")
    australia_live.add_argument("--bbox", default=AUSTRALIA_FLICKR_BBOX)
    australia_live.add_argument("--gbif-workers", type=int, default=1, help="must remain 1; GBIF requests are globally paced")
    australia_live.add_argument("--max-api-calls", type=int, default=3400)
    australia_live.add_argument("--run-id")
    storage = subparsers.add_parser("storage")
    storage_subparsers = storage.add_subparsers(dest="storage_command")
    storage_doctor = storage_subparsers.add_parser("doctor")
    storage_doctor.add_argument("--config")
    storage_handoff_build = storage_subparsers.add_parser("handoff-build")
    storage_handoff_build.add_argument("--root", default=".")
    storage_handoff_build.add_argument("--source", action="append", required=True)
    storage_handoff_build.add_argument("--output-dir", required=True)
    storage_handoff_build.add_argument("--name", required=True)
    storage_handoff_build.add_argument("--source-git-sha", required=True)
    storage_handoff_upload = storage_subparsers.add_parser("handoff-upload")
    storage_handoff_upload.add_argument("--archive", required=True)
    storage_handoff_upload.add_argument("--sha256", required=True)
    storage_handoff_upload.add_argument("--destination-prefix", required=True)
    storage_handoff_upload.add_argument("--receipt", required=True)
    storage_handoff_upload.add_argument("--config")
    storage_handoff_receive = storage_subparsers.add_parser("handoff-receive")
    storage_handoff_receive.add_argument("--uri", required=True)
    storage_handoff_receive.add_argument("--sha256", required=True)
    storage_handoff_receive.add_argument("--cache-dir", required=True)
    storage_handoff_receive.add_argument("--destination", required=True)
    storage_handoff_receive.add_argument("--receipt", required=True)
    storage_handoff_receive.add_argument("--config")
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
    production_run.add_argument("--vision-profile", choices=("mac_m5pro_64gb",))
    production_run.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"))
    production_run.add_argument("--yolo-checkpoint")
    production_run.add_argument("--yolo-sidecar-transport", choices=("json_b64", "image_path"))
    production_run.add_argument("--yolo-imgsz", type=int)
    production_run.add_argument("--yolo-batch", type=int)
    production_run.add_argument(
        "--possible-adult-route",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    production_run.add_argument("--possible-adult-route-threshold", type=float)
    production_run.add_argument(
        "--ambiguous-insect-review",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    production_run.add_argument("--ambiguous-insect-review-threshold", type=float)
    production_run.add_argument("--bioclip-batch", type=int)
    production_run.add_argument("--adaptive-batching", action="store_true")
    production_run.add_argument("--bioclip-top-k", type=int)
    production_run.add_argument(
        "--classification-mode",
        type=_classification_mode_arg,
        choices=SUPPORTED_CLASSIFICATION_MODES,
        default=DEFAULT_CLASSIFICATION_MODE,
    )
    production_run.add_argument(
        "--classification-config",
        help=(
            "explicit local configuration for an opt-in classification mode; "
            "required by build_week_target_aware_prototype"
        ),
    )
    production_run.add_argument("--taxonomy-text-embedding-cache")
    production_run.add_argument("--regional-candidates")
    production_run.add_argument("--reference-embeddings")
    production_run.add_argument("--classifier-artifact")
    production_run.add_argument("--calibrator-artifact")
    production_run.add_argument(
        "--reference-bank-readiness",
        help="immutable reference-bank readiness artifact directory",
    )
    production_run.add_argument(
        "--reference-bank-readiness-sha256",
        help="trusted sha256: digest pin for reference_bank_readiness.json",
    )
    production_run.add_argument(
        "--reference-admission-mode",
        choices=REFERENCE_ADMISSION_MODES,
        default=DEFAULT_REFERENCE_ADMISSION_MODE,
        help=(
            "reference admission policy: adaptive GBIF fast-start is the "
            "production default; strict blocks on review; flagged-only "
            "escalates statistically flagged species"
        ),
    )
    production_run.add_argument(
        "--reference-source",
        default=DEFAULT_REFERENCE_SOURCE,
        help="reference observation source (production default: gbif)",
    )
    production_run.add_argument(
        "--initial-scoring-mode",
        default=DEFAULT_INITIAL_SCORING_MODE,
        help=(
            "first-pass scoring contract; provisional_reference_ranking "
            "never claims calibrated probability"
        ),
    )
    production_run.add_argument(
        "--flickr-release-requires-human-review",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require decisive human verification before final Flickr release",
    )
    production_run.add_argument(
        "--statistical-reference-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require species-level statistical reference audit before approval",
    )
    production_run.add_argument(
        "--strict-reference-readiness-claim",
        action="store_true",
        help=(
            "declare that all admitted references satisfy the strict human-"
            "verified readiness contract"
        ),
    )
    production_run.add_argument(
        "--reference-split-use",
        action="append",
        choices=("screening", "training", "calibration", "final_test"),
        default=[],
        help="declare each downstream split that may consume admitted references",
    )
    production_run.add_argument("--crop-padding-ratio", type=float)
    production_run.add_argument("--parquet-compression")
    production_run.add_argument("--delete-images-after-commit", action=argparse.BooleanOptionalAction, default=None)
    production_run.add_argument("--stages")
    production_run.add_argument(
        "--workflow",
        choices=("adaptive", "legacy", "reference-first"),
        default="adaptive",
        help="production stage contract (default: adaptive GBIF fast-start)",
    )
    production_run.add_argument("--dry-run", action="store_true")
    production_run.add_argument("--build-registry-if-missing", action="store_true")
    production_run.add_argument("--limit-species", type=int, default=0)
    production_run.add_argument("--limit-records", type=int, default=0)
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


def _add_dev_vision_commands(subparsers: Any) -> None:
    bioclip_runtime = subparsers.add_parser("bioclip-runtime-check")
    bioclip_runtime.add_argument("--runtime-python", default=BIOCLIP_RUNTIME_PYTHON)
    bioclip_runtime.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    bioclip_runtime.add_argument("--hf-cache-dir", default=BIOCLIP_HF_CACHE_DIR)
    bioclip_runtime.add_argument("--model-name", default=BIOCLIP_25_HUGE_REPO_ID)
    bioclip_runtime.add_argument("--revision", default=BIOCLIP_25_HUGE_REVISION)
    bioclip_prefetch = subparsers.add_parser("bioclip-prefetch-model")
    bioclip_prefetch.add_argument("--runtime-python", default=BIOCLIP_RUNTIME_PYTHON)
    bioclip_prefetch.add_argument("--hf-cache-dir", default=BIOCLIP_HF_CACHE_DIR)
    bioclip_prefetch.add_argument("--model-name", default=BIOCLIP_25_HUGE_REPO_ID)
    bioclip_prefetch.add_argument("--revision", default=BIOCLIP_25_HUGE_REVISION)
    bioclip_prefetch.add_argument("--max-workers", type=int, default=8)
    text_embedding_cache = subparsers.add_parser("build-text-embedding-cache")
    text_embedding_cache.add_argument("--registry-dir", required=True)
    text_embedding_cache.add_argument("--output", required=True)
    text_embedding_cache.add_argument("--runtime-python", default=BIOCLIP_RUNTIME_PYTHON)
    text_embedding_cache.add_argument("--hf-cache-dir", default=BIOCLIP_HF_CACHE_DIR)
    text_embedding_cache.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    text_embedding_cache.add_argument("--batch-size", type=int, default=256)
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
    prototype_smoke = subparsers.add_parser(
        "prototype-smoke",
        aliases=("prototype-smoke-five",),
        help="run local YOLOE and BioCLIP smoke evidence over any non-empty image set",
    )
    prototype_smoke.add_argument("--config", required=True)
    prototype_embeddings = subparsers.add_parser("prototype-build-embeddings")
    prototype_embeddings.add_argument("--config", required=True)
    prototype_staged = subparsers.add_parser("prototype-staged-flickr")
    prototype_staged.add_argument("--config", required=True)
    prototype_benchmark = subparsers.add_parser("prototype-benchmark-matrix")
    prototype_benchmark.add_argument("--config", required=True)
    prototype_policy = subparsers.add_parser("prototype-select-policy")
    prototype_policy.add_argument("--config", required=True)
    benchmark = subparsers.add_parser("benchmark-plumbing")
    benchmark.add_argument("--records", type=int, default=1000)
    benchmark.add_argument("--butterfly-rate", type=float, default=0.25)
    benchmark.add_argument("--detections-per-butterfly", type=int, default=1)
    benchmark.add_argument(
        "--classification-mode",
        type=_classification_mode_arg,
        choices=SUPPORTED_CLASSIFICATION_MODES,
        default=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    )
    benchmark.add_argument("--registry-dir")
    benchmark.add_argument("--rank-beam-width", type=int, default=DEFAULT_RANK_BEAM_WIDTH)
    benchmark.add_argument("--species-first-pass-top-k", type=int, default=DEFAULT_SPECIES_FIRST_PASS_TOP_K)
    benchmark.add_argument("--species-rerank-top-k", type=int, default=DEFAULT_SPECIES_RERANK_TOP_K)
    benchmark.add_argument("--output-dir", required=True)
    rolling_benchmark = subparsers.add_parser("benchmark-rolling-matrix")
    rolling_benchmark.add_argument("--records", type=int, default=1000)
    rolling_benchmark.add_argument("--output-dir", required=True)
    cascade_benchmark = subparsers.add_parser("benchmark-cascade")
    cascade_benchmark.add_argument("--output-dir", required=True)
    live_benchmark = subparsers.add_parser("benchmark-live-m5pro")
    live_benchmark.add_argument("--input", required=True)
    live_benchmark.add_argument("--registry-dir", required=True)
    live_benchmark.add_argument("--taxonomy-text-embedding-cache", required=True)
    live_benchmark.add_argument("--vision-runtime-python", default=YOLOE26_RUNTIME_PYTHON)
    live_benchmark.add_argument("--bioclip-runtime-python", default=BIOCLIP_RUNTIME_PYTHON)
    live_benchmark.add_argument("--hf-cache-dir", default=BIOCLIP_HF_CACHE_DIR)
    live_benchmark.add_argument("--checkpoint", default="yoloe-26s-seg.pt")
    live_benchmark.add_argument("--yolo-sidecar-transport", default="json_b64", choices=("json_b64", "image_path"))
    live_benchmark.add_argument("--device", default="mps", choices=("auto", "cuda", "mps", "cpu"))
    live_benchmark.add_argument("--limit", type=int, default=100)
    live_benchmark.add_argument("--output-dir", required=True)
    live_benchmark.add_argument("--cache-root", default="data/cache/images")
    live_benchmark.add_argument("--crop-temp-dir", default="data/cache/object_crops")
    live_benchmark.add_argument("--imgsz", type=int, default=768)
    live_benchmark.add_argument("--conf", type=float, default=0.20)
    live_benchmark.add_argument("--iou", type=float, default=0.50)
    live_benchmark.add_argument("--max-det", type=int, default=8)
    live_benchmark.add_argument("--yolo-batch", type=int, default=16)
    live_benchmark.add_argument("--bioclip-batch", type=int, default=24)
    live_benchmark.add_argument("--crop-padding-ratio", type=float, default=0.08)
    live_benchmark.add_argument("--crop-target-px", type=int, default=336)
    live_benchmark.add_argument("--parquet-batch-rows", type=int, default=10000)
    live_benchmark.add_argument("--prompt-class", action="append", default=[])
    live_benchmark.add_argument("--include-hard-negative-prompts", action=argparse.BooleanOptionalAction, default=True)
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
    if args.command == "dev" and args.dev_command == "vision":
        if args.vision_command == "bioclip-runtime-check":
            return _run_bioclip_runtime_check(args)
        if args.vision_command == "bioclip-prefetch-model":
            return _run_bioclip_prefetch_model(args)
        if args.vision_command == "build-text-embedding-cache":
            return _run_build_text_embedding_cache(args)
        if args.vision_command == "yoloe26-runtime-check":
            return _run_yoloe26_runtime_check(args)
        if args.vision_command == "yoloe26-prefetch":
            return _run_yoloe26_prefetch(args)
        if args.vision_command == "yoloe26-smoke":
            return _run_yoloe26_smoke(args)
        if args.vision_command in {"prototype-smoke", "prototype-smoke-five"}:
            return _run_prototype_vision_smoke(args)
        if args.vision_command == "prototype-build-embeddings":
            return _run_prototype_support_embeddings(args)
        if args.vision_command == "prototype-staged-flickr":
            return _run_prototype_staged_flickr(args)
        if args.vision_command == "prototype-benchmark-matrix":
            return _run_prototype_benchmark_matrix(args)
        if args.vision_command == "prototype-select-policy":
            return _run_prototype_policy_selection(args)
        if args.vision_command == "benchmark-plumbing":
            return _run_vision_benchmark_plumbing(args)
        if args.vision_command == "benchmark-rolling-matrix":
            return _run_vision_benchmark_rolling_matrix(args)
        if args.vision_command == "benchmark-cascade":
            return _run_path_cascade_benchmark(args)
        if args.vision_command == "benchmark-live-m5pro":
            return _run_vision_benchmark_live_m5pro(args)
        if args.vision_command == "crop-preview":
            return _run_detect_crop_preview(args)
        if args.vision_command == "eval":
            return _run_detect_eval(args)
        return 2
    if args.command == "bioclip":
        if args.bioclip_command == "prototype-evidence":
            from biominer.reports.prototype_evidence import (
                PrototypeEvidenceConfig,
                build_prototype_evidence_outputs,
            )

            try:
                result = build_prototype_evidence_outputs(
                    PrototypeEvidenceConfig.read_json(args.config)
                )
            except (FileNotFoundError, TypeError, ValueError) as exc:
                print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
                return 2
            print(
                json.dumps(
                    {
                        "dashboard": str(result.dashboard_path),
                        "regional_competitors": str(result.competitors_path),
                        "nearest_references": str(result.references_path),
                        "report": str(result.report_path),
                        "summary": str(result.summary_path),
                        "status": result.report["status"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.bioclip_command != "screen":
            return 2
        registry = Path(args.registry_dir)
        try:
            store = PathTaxonomyStore.read(registry)
        except (FileNotFoundError, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
            return 2
        payload = {
            "status": "validated" if args.dry_run else "ready",
            "input": args.input,
            "output_dir": args.output_dir,
            "registry_dir": str(registry),
            "rank_order": list(getattr(store, "rank_order", ("FAMILY", "SUBFAMILY", "TRIBE", "SUBTRIBE", "GENUS", "SPECIES"))),
            "classification_fingerprint": store.classification_fingerprint,
            "dry_run": bool(args.dry_run),
        }
        if not args.dry_run:
            payload["next_command"] = "biominer run (screen execution remains orchestrated through the production run command)"
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "evidence":
        if args.evidence_command == "join":
            return _run_bioclip_join_object_evidence(args)
        return 2
    if args.command == "evaluation":
        return _run_evaluation_command(args)
    if args.command == "dynamic-pooling":
        return run_dynamic_pooling_command(args)
    if args.command == "references":
        return _run_references_command(args)
    if args.command == "storage":
        return _run_storage_command(args)
    if args.command == "workstore":
        return _run_workstore_command(args)
    if args.command == "run":
        return _run_production_command(args)
    if args.command == "dev" and args.dev_command == "flickr":
        return _run_dev_flickr_command(args)
    if args.command == "registry" or (args.command == "dev" and args.dev_command == "registry"):
        if args.registry_command == "harvest-col-xr":
            taxonomy = asyncio.run(
                harvest_col_xr_taxonomy(
                    args.output_dir,
                    scope_path=args.scope_json,
                    workers=args.workers,
                    max_retries=args.max_retries,
                )
            )
            names = None
            if not args.skip_names:
                names = asyncio.run(
                    harvest_col_xr_names(
                        args.output_dir,
                        workers=args.workers,
                        max_retries=args.max_retries,
                        limit=args.name_limit,
                    )
                )
            print(json.dumps({"taxonomy": taxonomy, "names": names}, indent=2, sort_keys=True))
            return 0
        if args.registry_command == "enrich-hierarchy":
            requested = tuple(part.strip().casefold() for part in args.sources.split(",") if part.strip())
            unknown = set(requested) - {"gbif", "open_tree"}
            if unknown:
                print(json.dumps({"error": f"unknown hierarchy sources: {sorted(unknown)}"}, indent=2))
                return 2
            result: dict[str, Any] = {}
            if "gbif" in requested:
                result["gbif"] = asyncio.run(
                    harvest_gbif_genus_evidence(
                        args.registry_dir,
                        args.source_dir,
                        workers=args.workers,
                        max_retries=args.max_retries,
                    )
                )
            if "open_tree" in requested:
                result["open_tree"] = asyncio.run(
                    harvest_open_tree_genus_evidence(
                        args.registry_dir,
                        args.source_dir,
                        workers=min(args.workers, 4),
                        max_retries=args.max_retries,
                    )
                )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
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
                if args.col_xr_parquet_source:
                    if args.source_json or args.col_xr_archive:
                        raise ValueError(
                            "--col-xr-parquet-source cannot be combined with --source-json or --col-xr-archive"
                        )
                    payload = compile_registry_parquet_source(
                        args.col_xr_parquet_source,
                        args.output_dir,
                        registry_version=args.registry_version,
                        scope_path=args.scope_json,
                        query_curation_json=args.query_curation_json,
                    )
                    print(json.dumps(payload, indent=2, sort_keys=True))
                    return 0
                source_json = args.source_json
                reuse_source_json = args.reuse_source_json
                if args.col_xr_archive:
                    if is_cloud_uri(str(args.output_dir)):
                        raise ValueError("--col-xr-archive currently requires a local build directory")
                    source_snapshot = extract_col_xr_snapshot(
                        args.col_xr_archive,
                        scope_path=args.scope_json,
                        retrieved_at=args.retrieved_at,
                    )
                    source_path = Path(args.output_dir) / "col_xr_source_snapshot.json"
                    source_path.parent.mkdir(parents=True, exist_ok=True)
                    source_path.write_text(json.dumps(source_snapshot, indent=2, sort_keys=True), encoding="utf-8")
                    source_json = str(source_path)
                    reuse_source_json = True
                payload = build_registry(
                    output_dir=args.output_dir,
                    registry_version=args.registry_version,
                    scope_path=args.scope_json,
                    source_json=source_json,
                    reuse_source_json=reuse_source_json,
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
                    mymemory_monthly_request_limit=args.mymemory_monthly_request_limit,
                    mymemory_monthly_input_word_limit=args.mymemory_monthly_input_word_limit,
                    mymemory_monthly_bandwidth_mb_limit=args.mymemory_monthly_bandwidth_mb_limit,
                    mymemory_response_byte_reservation=args.mymemory_response_byte_reservation,
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
            except (FileNotFoundError, ValueError) as exc:
                print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
                return 2
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.registry_command == "seed-flickr-queries":
            queries = load_registry_flickr_queries(args.query_definitions)
            state = MetadataPollState(args.state_db)
            registration = state.register_query_definitions(pl.read_parquet(args.query_definitions))
            inserted = sum(state.enqueue_work_item(query) for query in queries)
            print(
                json.dumps(
                    {
                        "query_definitions": args.query_definitions,
                        "state_db": args.state_db,
                        "work_items_seen": len(queries),
                        "work_items_inserted": inserted,
                        **registration,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.registry_command == "audit":
            print(json.dumps(audit_registry(args.registry_dir, report_dir=args.report_dir), indent=2, sort_keys=True))
            return 0
        if args.registry_command == "publish":
            try:
                payload = publish_registry(
                    args.registry_dir,
                    output_dir=args.output_dir,
                    replace_existing=args.replace_existing,
                )
            except (FileExistsError, FileNotFoundError, ValueError) as exc:
                print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
                return 2
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        return 2
    return 2


def _run_dev_flickr_command(args: argparse.Namespace) -> int:
    if args.flickr_command == "australia-live":
        return _run_australia_live(args)
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


def _run_australia_live(args: argparse.Namespace) -> int:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(json.dumps({"error": f"missing Flickr API key in {args.api_key_env}"}, indent=2, sort_keys=True))
        return 2
    cutoff = datetime.now(UTC).date().isoformat()
    output_dir = Path(args.output_dir)
    presence = build_australia_presence(
        registry_dir=args.registry_dir,
        state_db=args.presence_state_db,
        output_path=output_dir / "australia_species_presence.parquet",
        workers=args.gbif_workers,
    )
    failed = int(presence.filter(pl.col("status") != "complete").height)
    if failed:
        print(json.dumps({"error": "GBIF Australia prescan has retryable failures; rerun after they clear", "failed_species": failed}, indent=2, sort_keys=True))
        return 2
    place_id = args.place_id
    definitions, associations, queries = compile_australia_query_plan(
        registry_dir=args.registry_dir,
        presence=presence,
        output_dir=output_dir,
        place_id=place_id,
        woe_id=args.woe_id,
        bbox=args.bbox,
        cutoff=cutoff,
    )
    state = MetadataPollState(args.state_db)
    registration = state.register_query_definitions(definitions, keyword_associations=associations)
    enqueued = state.enqueue_initial_work_items(queries)
    result = poll_once(
        state_db=args.state_db,
        raw_root=args.raw_root,
        evidence_output=args.evidence_output,
        max_api_calls=args.max_api_calls,
        api_key=api_key,
        workers=1,
        run_id=args.run_id,
        evidence_stage="flickr_australia_metadata",
        min_call_interval_seconds=3600 / 3400,
    )
    payload = {
        "scope": "Australia public geotagged Flickr photos",
        "place_id": place_id,
        "woe_id": args.woe_id,
        "bbox": args.bbox,
        "cutoff": cutoff,
        "gbif_species": presence.height,
        "gbif_local_species": int(presence.filter(pl.col("gbif_au_occurrence_count") > 0).height),
        "query_definitions": definitions.height,
        "query_associations": associations.height,
        "work_items_enqueued": enqueued,
        "registration": registration,
        "poll": {**result.__dict__, "state_db": str(result.state_db)},
    }
    report = Path("reports") / f"flickr_australia_{args.run_id or datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "report": str(report)}, indent=2, sort_keys=True, default=str))
    return 0


def _run_evaluation_command(args: argparse.Namespace) -> int:
    if args.evaluation_command == "classify":
        return _run_evaluation_classify(args)
    if args.evaluation_command == "review-queue":
        return _run_evaluation_review_queue(args)
    if args.evaluation_command == "build-sampling-frame":
        return _run_evaluation_sampling_frame(args)
    return 2


def _run_references_command(args: argparse.Namespace) -> int:
    if is_reference_workflow_command(args.references_command):
        return run_reference_workflow_command(args)
    readiness_status: str | None = None
    readiness_sha256: str | None = None
    vision_permitted: bool | None = None
    try:
        if args.references_command == "export-review-queue":
            validate_reference_review_history_head_destination(
                args.history_head,
                args.output_dir,
            )
            result = build_reference_review_queue(
                _read_reference_parquet(
                    args.acquisition_selections,
                    artifact="acquisition selections",
                ),
                _read_reference_parquet(args.media_objects, artifact="media objects"),
                _read_reference_parquet(
                    args.media_candidates,
                    artifact="media candidates",
                ),
                _read_reference_parquet(args.observations, artifact="observations"),
                _read_reference_parquet(
                    args.duplicate_relationships,
                    artifact="duplicate relationships",
                ),
                deduplication_report=_read_reference_json(
                    args.deduplication_report,
                    artifact="deduplication report",
                ),
                reference_bank_version=args.reference_bank_version,
                include_research_only=args.include_research_only,
            )
            artifacts = write_reference_review_export(
                result,
                Path(args.output_dir),
                run_id=args.run_id,
            )
            initialize_reference_review_history_head(
                args.history_head,
                artifacts["report"],
            )
        elif args.references_command == "import-review-decisions":
            prior_report, prior_report_sha256 = validate_reference_review_history_head(
                args.history_head,
                args.prior_review_report,
            )
            for logical_name, path in (
                ("queue", args.review_queue),
                ("queue_provenance", args.queue_provenance),
                ("decisions", args.existing_decisions),
            ):
                validate_reference_review_packet_artifact(
                    prior_report,
                    logical_name,
                    path,
                )
            existing_decisions = _read_reference_parquet(
                args.existing_decisions,
                artifact="existing decisions",
            )
            result = import_reference_review_decisions(
                _read_reference_parquet(args.decisions, artifact="decisions"),
                queue=_read_reference_parquet(
                    args.review_queue,
                    artifact="review queue",
                ),
                queue_provenance=_read_reference_parquet(
                    args.queue_provenance,
                    artifact="review queue provenance",
                ),
                existing_decisions=existing_decisions,
                prior_report=prior_report,
                prior_report_sha256=prior_report_sha256,
            )
            artifacts = write_reference_review_import(
                result,
                Path(args.output_dir),
                run_id=args.run_id,
            )
            advance_reference_review_history_head(
                args.history_head,
                prior_report_path=args.prior_review_report,
                next_report_path=artifacts["report"],
            )
        elif args.references_command == "compile-visual-domain-negatives":
            try:
                artifacts = publish_curated_visual_domain_negative_manifest(
                    args.source_manifest,
                    Path(args.output_dir),
                    run_id=args.run_id,
                )
            except TypeError as exc:
                raise ValueError(str(exc)) from exc
        elif args.references_command == "validate-readiness":
            policy = ReferenceBankReadinessPolicy.from_mapping(
                _read_reference_json(
                    args.readiness_policy,
                    artifact="readiness policy",
                )
            )
            model_identity = ReferenceModelInputIdentity.from_mapping(
                _read_reference_json(
                    args.model_identity,
                    artifact="model identity",
                )
            )
            result = build_reference_bank_readiness(
                candidate_species=_read_reference_parquet(
                    args.candidate_species,
                    artifact="candidate species",
                ),
                acquisition_plan=_read_reference_parquet(
                    args.acquisition_plan,
                    artifact="acquisition plan",
                ),
                acquisition_selections=_read_reference_parquet(
                    args.acquisition_selections,
                    artifact="acquisition selections",
                ),
                observations=_read_reference_parquet(
                    args.observations,
                    artifact="observations",
                ),
                media_candidates=_read_reference_parquet(
                    args.media_candidates,
                    artifact="media candidates",
                ),
                media_objects=_read_reference_parquet(
                    args.media_objects,
                    artifact="media objects",
                ),
                duplicate_relationships=_read_reference_parquet(
                    args.duplicate_relationships,
                    artifact="duplicate relationships",
                ),
                deduplication_report=_read_reference_json(
                    args.deduplication_report,
                    artifact="deduplication report",
                ),
                review_queue=_read_reference_parquet(
                    args.review_queue,
                    artifact="review queue",
                ),
                queue_provenance=_read_reference_parquet(
                    args.queue_provenance,
                    artifact="review queue provenance",
                ),
                review_decisions=_read_reference_parquet(
                    args.review_decisions,
                    artifact="review decisions",
                ),
                split_assignments=_read_reference_parquet(
                    args.split_assignments,
                    artifact="split assignments",
                ),
                policy=policy,
                registry_version=args.registry_version,
                reference_bank_version=args.reference_bank_version,
                model_identity=model_identity,
            )
            artifacts = publish_reference_bank_readiness(
                result,
                Path(args.output_dir),
                run_id=args.run_id,
            )
            readiness_status = str(result.readiness["status"])
            readiness_sha256 = _sha256_file_path(artifacts["readiness"])
            vision_permitted = reference_readiness_allows_vision(result.readiness)
        else:
            return 2
    except (OSError, TypeError, ValueError, pl.exceptions.PolarsError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 2

    payload = {
        "artifacts": {str(name): str(path) for name, path in sorted(artifacts.items())},
        "command": f"references {args.references_command}",
        "output_dir": str(args.output_dir),
        "status": "complete",
    }
    if readiness_status is not None:
        payload["readiness"] = readiness_status
        payload["readiness_sha256"] = readiness_sha256
        payload["vision_permitted"] = vision_permitted
    print(json.dumps(payload, sort_keys=True))
    return 0 if vision_permitted is not False else 2


def _read_reference_parquet(path: str | Path, *, artifact: str) -> pl.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"{artifact} path does not exist: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"{artifact} path is not a file: {input_path}")
    return pl.read_parquet(input_path)


def _sha256_file_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_reference_json(path: str | Path, *, artifact: str) -> dict[str, object]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"{artifact} path does not exist: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"{artifact} path is not a file: {input_path}")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact} must contain a JSON object: {input_path}")
    return payload


def _run_evaluation_classify(args: argparse.Namespace) -> int:
    input_uri = str(args.object_scores or args.object_evidence)
    input_kind = "object_scores" if args.object_scores else "object_evidence"
    labels_uri = str(args.reviewed_labels)
    storage_backend = str(getattr(args, "storage_backend", "local") or "local")
    try:
        storage = None
        if storage_backend == "local":
            object_scores = _read_local_evaluation_parquet(input_uri, input_kind)
            reviewed_labels = _read_local_reviewed_labels(labels_uri)
        elif storage_backend == "s3":
            if bool(getattr(args, "write_charts", False)):
                raise ValueError("--write-charts is currently supported only for local evaluation outputs")
            storage = _evaluation_storage_from_config(args)
            object_scores = _read_storage_evaluation_parquet(storage, input_uri, input_kind)
            reviewed_labels = _read_storage_reviewed_labels(storage, labels_uri)
            _require_s3_uri("output-dir", args.output_dir)
        else:
            raise ValueError(f"unsupported evaluation storage backend: {storage_backend}")
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError, pl.exceptions.PolarsError) as exc:
        print(json.dumps({"error": _redact_cloud_error(str(exc), args)}, indent=2, sort_keys=True))
        return 2

    label_findings = validate_reviewed_label_frame(reviewed_labels)
    fatal_findings = [finding for finding in label_findings if finding.get("severity") == "fatal"]
    if fatal_findings:
        print(
            json.dumps(
                {
                    "error": "reviewed labels failed validation",
                    "fatal_findings": fatal_findings,
                    "finding_count": len(label_findings),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    if storage is None:
        output_dir = _local_evaluation_output_dir(args.output_dir)
        paths = write_evaluation_report(
            object_scores=object_scores,
            reviewed_labels=reviewed_labels,
            output_dir=output_dir,
            write_charts=bool(getattr(args, "write_charts", False)),
        )
        metrics = json.loads(Path(paths["metrics"]).read_text(encoding="utf-8"))
    else:
        paths = write_evaluation_report_to_storage(
            object_scores=object_scores,
            reviewed_labels=reviewed_labels,
            output_dir=args.output_dir,
            storage=storage,
            write_charts=bool(getattr(args, "write_charts", False)),
        )
        metrics = storage.read_json(paths["metrics"])
    evaluation_profile = str(getattr(args, "evaluation_profile", STANDARD_EVALUATION_PROFILE))
    if evaluation_profile == XIE_STYLE_EVALUATION_PROFILE:
        xie_metrics = evaluate_xie_style_hierarchical(
            object_scores=object_scores,
            reviewed_labels=reviewed_labels,
        )
        if storage is None:
            xie_path = str(output_dir / XIE_STYLE_METRICS_FILE)
            xie_output = Path(xie_path)
            xie_output.parent.mkdir(parents=True, exist_ok=True)
            xie_output.write_text(json.dumps(xie_metrics, indent=2, sort_keys=True), encoding="utf-8")
        else:
            xie_path = join_uri(args.output_dir, XIE_STYLE_METRICS_FILE)
            storage.write_json(xie_path, xie_metrics)
        paths["xie_style_metrics"] = xie_path
    payload = {
        "status": "complete",
        "storage_backend": storage_backend,
        "input_kind": input_kind,
        "input_path": input_uri,
        "reviewed_labels": labels_uri,
        "output_dir": str(args.output_dir),
        "evaluation_profile": evaluation_profile,
        "write_charts": bool(getattr(args, "write_charts", False)),
        "paths": paths,
        "metrics": {
            "evaluated_objects": metrics["metrics"].get("evaluated_objects"),
            "family_top1_accuracy": metrics["metrics"].get("family_top1_accuracy"),
            "family_top3_recall": metrics["metrics"].get("family_top3_recall"),
            "species_top1_accuracy": metrics["metrics"].get("species_top1_accuracy"),
            "species_top5_recall": metrics["metrics"].get("species_top5_recall"),
            "species_top20_recall": metrics["metrics"].get("species_top20_recall"),
            "species_mrr": metrics["metrics"].get("species_mrr"),
        },
        "label_validation": {
            "finding_count": len(label_findings),
            "warning_count": sum(1 for finding in label_findings if finding.get("severity") == "warning"),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_evaluation_review_queue(args: argparse.Namespace) -> int:
    try:
        if args.max_rows is not None and args.max_rows < 0:
            raise ValueError("--max-rows must be non-negative")
        object_evidence = _read_local_evaluation_parquet(str(args.object_evidence), "object-evidence")
        photo_summary = None
        if args.photo_summary:
            photo_summary = _read_local_evaluation_parquet(str(args.photo_summary), "photo-summary")
        _raise_if_cloud_uri_for_local_backend("output", str(args.output))
        output = normalize_local_uri(args.output)
        queue = build_hierarchical_review_queue(
            object_evidence=object_evidence,
            photo_summary=photo_summary,
            max_rows=args.max_rows,
        )
        write_parquet(queue, output)
    except (FileNotFoundError, RuntimeError, ValueError, pl.exceptions.PolarsError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
        return 2

    payload = {
        "status": "complete",
        "input_path": str(args.object_evidence),
        "photo_summary": str(args.photo_summary) if args.photo_summary else None,
        "output": str(output),
        "review_queue_rows": queue.height,
        "review_priority_counts": _value_counts(queue, "review_priority"),
        "review_reason_counts": _value_counts(queue, "review_reason"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_evaluation_sampling_frame(args: argparse.Namespace) -> int:
    try:
        if args.object_scores and not args.competitor_taxa:
            raise ValueError(
                "--competitor-taxa is required when --object-scores is provided"
            )
        publication = materialize_evaluation_sampling_frame(
            candidates_path=args.candidates,
            geo_assignments_path=args.geo_assignments,
            query_state_db=args.query_state_db,
            object_scores_path=args.object_scores,
            competitor_taxa_path=args.competitor_taxa,
            output_path=args.output,
            config=EvaluationSamplingConfig(
                target_text_terms=tuple(args.target_text_term),
                random_seed=args.random_seed,
            ),
            run_id=args.run_id,
        )
    except (
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
        pl.exceptions.PolarsError,
    ) as exc:
        print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "complete",
                "frame": str(publication.frame_path),
                "report_json": str(publication.report_json_path),
                "report_markdown": str(publication.report_markdown_path),
                "metrics": {
                    "rows": publication.report["rows_out"],
                    "scored": publication.report["scored_count"],
                    "unscored": publication.report["unscored_count"],
                    "no_geo": publication.report["no_geo_count"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _read_local_evaluation_parquet(uri: str, input_kind: str) -> pl.DataFrame:
    _raise_if_cloud_uri_for_local_backend(input_kind, uri)
    path = normalize_local_uri(uri)
    if not path.exists():
        raise FileNotFoundError(f"{input_kind} path does not exist: {path}")
    return pl.read_parquet(path)


def _read_local_reviewed_labels(uri: str) -> pl.DataFrame:
    _raise_if_cloud_uri_for_local_backend("reviewed-labels", uri)
    path = normalize_local_uri(uri)
    if not path.exists():
        raise FileNotFoundError(f"reviewed-labels path does not exist: {path}")
    return read_reviewed_labels(path)


def _local_evaluation_output_dir(uri: str) -> Path:
    _raise_if_cloud_uri_for_local_backend("output-dir", uri)
    return normalize_local_uri(uri)


def _read_storage_evaluation_parquet(storage: object, uri: str, input_kind: str) -> pl.DataFrame:
    _require_s3_uri(input_kind, uri)
    if not storage.exists(uri):
        raise FileNotFoundError(f"{input_kind} path does not exist: {uri}")
    return storage.read_parquet(uri)


def _read_storage_reviewed_labels(storage: object, uri: str) -> pl.DataFrame:
    _require_s3_uri("reviewed-labels", uri)
    if not storage.exists(uri):
        raise FileNotFoundError(f"reviewed-labels path does not exist: {uri}")
    suffix = Path(uri).suffix.casefold()
    if suffix == ".parquet":
        frame = storage.read_parquet(uri)
    elif suffix in {".jsonl", ".ndjson"}:
        frame = pl.read_ndjson(
            io.BytesIO(storage.read_text(uri).encode("utf-8"))
        )
    elif suffix == ".json":
        frame = pl.read_json(io.BytesIO(storage.read_text(uri).encode("utf-8")))
    else:
        raise ValueError(
            f"unsupported reviewed-label format: {suffix or '<none>'}"
        )
    return normalize_reviewed_label_frame(frame)


def _evaluation_storage_from_config(args: argparse.Namespace) -> object:
    config = load_biominer_config(args.config)
    config = replace(config, storage=replace(config.storage, backend="s3"))
    return create_storage_backend(config.storage)


def _raise_if_cloud_uri_for_local_backend(name: str, uri: str) -> None:
    if is_cloud_uri(uri):
        raise ValueError(f"{name} is a cloud URI; use --storage-backend s3: {uri}")


def _require_s3_uri(name: str, uri: str) -> None:
    if not is_cloud_uri(uri):
        raise ValueError(f"{name} must be an s3:// URI when --storage-backend s3 is used: {uri}")


def _run_storage_command(args: argparse.Namespace) -> int:
    handlers = {
        "doctor": _run_storage_doctor,
        "handoff-build": _run_storage_handoff_build,
        "handoff-upload": _run_storage_handoff_upload,
        "handoff-receive": _run_storage_handoff_receive,
    }
    handler = handlers.get(args.storage_command)
    if handler is None:
        return 2
    try:
        payload = handler(args)
    except Exception as exc:  # pragma: no cover - exercised by live storage runs.
        print(json.dumps({"status": "error", "error": _redact_cloud_error(str(exc), args)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "ok" else 2


def _run_storage_handoff_build(args: argparse.Namespace) -> dict[str, object]:
    bundle = build_handoff_bundle(
        root=args.root,
        sources=args.source,
        output_dir=args.output_dir,
        name=args.name,
        source_git_sha=args.source_git_sha,
    )
    return {
        "status": "ok",
        "command": "storage handoff-build",
        "bundle": {
            "archive_path": str(bundle.archive_path),
            "sha256": bundle.sha256,
            "archive_byte_count": bundle.byte_count,
            "file_count": bundle.file_count,
            "source_byte_count": bundle.source_byte_count,
            "source_git_sha": bundle.source_git_sha,
            "source_roots": list(bundle.source_roots),
            "local_integrity": "archive_and_embedded_inventory_verified",
        },
    }


def _run_storage_handoff_upload(args: argparse.Namespace) -> dict[str, object]:
    config = load_biominer_config(args.config)
    if config.storage.backend != "s3":
        raise ValueError("storage handoff-upload requires the s3 storage backend")
    storage = create_storage_backend(config.storage)
    receipt = upload_handoff_bundle(
        storage=storage,
        archive=args.archive,
        expected_sha256=args.sha256,
        destination_prefix=args.destination_prefix,
        receipt_path=args.receipt,
    )
    return {
        "status": "ok",
        "command": "storage handoff-upload",
        "receipt": receipt,
    }


def _run_storage_handoff_receive(args: argparse.Namespace) -> dict[str, object]:
    config = load_biominer_config(args.config)
    if config.storage.backend != "s3":
        raise ValueError("storage handoff-receive requires the s3 storage backend")
    storage = create_storage_backend(config.storage)
    receipt = receive_handoff_bundle(
        storage=storage,
        uri=args.uri,
        expected_sha256=args.sha256,
        cache_dir=args.cache_dir,
        destination=args.destination,
        receipt_path=args.receipt,
    )
    return {
        "status": "ok",
        "command": "storage handoff-receive",
        "receipt": receipt,
    }


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


def _production_vision_settings_from_args(args: argparse.Namespace) -> VisionRuntimeSettings:
    settings = (
        vision_runtime_settings(args.vision_profile)
        if getattr(args, "vision_profile", None)
        else VisionRuntimeSettings(bioclip_model=BIOCLIP_25_HUGE_REPO_ID)
    )
    overrides: dict[str, object] = {}
    if getattr(args, "device", None) is not None:
        overrides["device"] = args.device
    if getattr(args, "yolo_checkpoint", None) is not None:
        overrides["yolo_checkpoint"] = args.yolo_checkpoint
    if getattr(args, "yolo_sidecar_transport", None) is not None:
        overrides["yolo_sidecar_transport"] = args.yolo_sidecar_transport
    if getattr(args, "yolo_imgsz", None) is not None:
        overrides["yolo_imgsz"] = args.yolo_imgsz
    if getattr(args, "yolo_batch", None) is not None:
        overrides["detector_batch_size"] = args.yolo_batch
    if getattr(args, "possible_adult_route", None) is not None:
        overrides["possible_adult_route_enabled"] = args.possible_adult_route
    if getattr(args, "possible_adult_route_threshold", None) is not None:
        overrides["possible_adult_route_threshold"] = (
            args.possible_adult_route_threshold
        )
    if getattr(args, "ambiguous_insect_review", None) is not None:
        overrides["ambiguous_insect_review_enabled"] = (
            args.ambiguous_insect_review
        )
    if getattr(args, "ambiguous_insect_review_threshold", None) is not None:
        overrides["ambiguous_insect_review_threshold"] = (
            args.ambiguous_insect_review_threshold
        )
    if getattr(args, "bioclip_batch", None) is not None:
        overrides["crop_batch_size"] = args.bioclip_batch
    if getattr(args, "bioclip_top_k", None) is not None:
        overrides["bioclip_top_k"] = args.bioclip_top_k
    if getattr(args, "crop_padding_ratio", None) is not None:
        overrides["crop_padding_ratio"] = args.crop_padding_ratio
    if getattr(args, "parquet_compression", None) is not None:
        overrides["parquet_compression"] = args.parquet_compression
    if getattr(args, "delete_images_after_commit", None) is not None:
        overrides["delete_images_after_commit"] = args.delete_images_after_commit
    if getattr(args, "adaptive_batching", False):
        overrides["adaptive_batching"] = True
    return settings.with_overrides(**overrides) if overrides else settings.with_overrides()


def _classification_mode_arg(value: str) -> str:
    try:
        return normalize_classification_mode(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _run_production_command(args: argparse.Namespace) -> int:
    config = None
    try:
        stages = _parse_run_stages(args.stages, workflow=args.workflow)
        prototype_config = None
        if args.classification_config:
            from biominer.bioclip.prototype_mode import BuildWeekPrototypeConfig

            prototype_config = BuildWeekPrototypeConfig.read_json(
                args.classification_config
            )
        if is_build_week_prototype_classification(args.classification_mode):
            if args.workflow != "reference-first":
                raise ValueError(
                    "build_week_target_aware_prototype requires "
                    "--workflow reference-first"
                )
            if prototype_config is None:
                raise ValueError(
                    "build_week_target_aware_prototype requires "
                    "--classification-config"
                )
        elif prototype_config is not None:
            raise ValueError(
                "--classification-config is only valid with "
                "build_week_target_aware_prototype"
            )
        if (
            not args.dry_run
            and any(stage in {RunStage.DETECT_OBJECTS, RunStage.SCORE_BIOCLIP} for stage in stages)
            and not args.reference_bank_readiness
        ):
            raise ValueError(
                "--reference-bank-readiness is required for non-dry detect_objects or score_bioclip stages"
            )
        if (
            not args.dry_run
            and any(stage in {RunStage.DETECT_OBJECTS, RunStage.SCORE_BIOCLIP} for stage in stages)
            and not args.reference_bank_readiness_sha256
        ):
            raise ValueError(
                "--reference-bank-readiness-sha256 is required for non-dry "
                "detect_objects or score_bioclip stages"
            )
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
        storage = None
        registry_dir_is_cloud = is_cloud_uri(args.registry_dir)
        if args.storage_backend != "local" and (not args.dry_run or registry_dir_is_cloud):
            storage = create_storage_backend(config.storage)
        workstore = None
        if not args.dry_run:
            workstore = create_workstore(config.workstore)
            _init_workstore_schema_if_supported(workstore)
        limits = {
            key: value
            for key, value in {
                "species": args.limit_species,
                "records": args.limit_records,
            }.items()
            if value and value > 0
        }
        vision_settings = _production_vision_settings_from_args(args)
        request = ProductionRunRequest(
            taxon=args.taxon,
            rank=args.rank,
            registry_dir=args.registry_dir,
            output_root=args.output_prefix,
            storage_backend=args.storage_backend,
            workstore_backend=args.workstore_backend,
            bioclip_model=vision_settings.bioclip_model,
            vision_profile=args.vision_profile,
            vision_settings=vision_settings,
            classification_mode=args.classification_mode,
            classification_config_path=args.classification_config,
            build_week_prototype_config=prototype_config,
            taxonomy_text_embedding_cache=args.taxonomy_text_embedding_cache,
            reference_bank_readiness=args.reference_bank_readiness,
            reference_bank_readiness_sha256=(
                args.reference_bank_readiness_sha256
            ),
            regional_candidates=args.regional_candidates,
            reference_embeddings=args.reference_embeddings,
            classifier_artifact=args.classifier_artifact,
            calibrator_artifact=args.calibrator_artifact,
            reference_admission_mode=args.reference_admission_mode,
            reference_source=args.reference_source,
            initial_scoring_mode=args.initial_scoring_mode,
            flickr_release_requires_human_review=(
                args.flickr_release_requires_human_review
            ),
            statistical_reference_audit=args.statistical_reference_audit,
            strict_reference_readiness_claim=(
                args.strict_reference_readiness_claim
            ),
            reference_split_uses=tuple(args.reference_split_use),
            worker_id="local" if allow_local and args.dry_run else config.runtime.worker_id or ("local" if allow_local else ""),
            stages=stages,
            dry_run=args.dry_run,
            build_registry_if_missing=args.build_registry_if_missing,
            limits=limits,
        )
        def create_vision_runtime() -> tuple[Any, Any, Any, list[Any]]:
            return _create_production_vision_runtime(
                vision_settings,
                classification_mode=args.classification_mode,
            )

        vision_runtime_factory = (
            create_vision_runtime
            if not args.dry_run
            and any(
                stage in {RunStage.DETECT_OBJECTS, RunStage.SCORE_BIOCLIP}
                for stage in stages
            )
            else None
        )
        plan = ProductionRunOrchestrator(
            request,
            storage=storage,
            workstore=workstore,
            vision_runtime_factory=vision_runtime_factory,
            flickr_api_key=os.environ.get("FLICKR_API_KEY"),
        ).run()
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        payload: dict[str, object] = {"error": redact_text(str(exc), config) if config else str(exc)}
        if config is not None:
            payload["config"] = redact_config(config)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    return 0


def _create_production_vision_runtime(
    vision_settings: VisionRuntimeSettings,
    *,
    classification_mode: str = DEFAULT_CLASSIFICATION_MODE,
) -> tuple[Any, Any, Any, list[Any]]:
    from biominer.bioclip.bioclip import PersistentBioClipScorer
    from biominer.bioclip.object_runner import EphemeralCropBioClipScorer
    from biominer.detection.image_io import load_decoded_image_from_record
    from biominer.detection.yoloe26_detector import YoloE26SidecarObjectDetector
    from biominer.vision.full_frame_attention import (
        TARGET_FULL_FRAME_IMAGE_RESIZE_MODE,
    )

    detector = YoloE26SidecarObjectDetector(
        runtime_python=YOLOE26_RUNTIME_PYTHON,
        checkpoint=vision_settings.yolo_checkpoint,
        device=vision_settings.device,
        imgsz=vision_settings.yolo_imgsz,
        conf=vision_settings.yolo_conf,
        iou=vision_settings.yolo_iou,
        max_det=vision_settings.yolo_max_det,
        transport=vision_settings.yolo_sidecar_transport,
        temp_dir=Path("/tmp") / "biominer_yoloe26",
    )
    runtime = _bioclip_runtime(
        runtime_python=Path(BIOCLIP_RUNTIME_PYTHON),
        model_name=vision_settings.bioclip_model,
    )
    image_resize_mode = (
        None
        if normalize_classification_mode(classification_mode)
        == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
        else TARGET_FULL_FRAME_IMAGE_RESIZE_MODE
    )
    persistent = PersistentBioClipScorer(
        runtime=runtime,
        hf_cache_dir=BIOCLIP_HF_CACHE_DIR,
        device=vision_settings.device,
        image_resize_mode=image_resize_mode,
        preprocess_workers=vision_settings.bioclip_preprocess_workers,
    )
    def image_loader(record: dict[str, Any]) -> Any:
        return load_decoded_image_from_record(record, cache_root="data/cache/images")
    scorer = EphemeralCropBioClipScorer(
        scorer=persistent,
        image_loader=image_loader,
        temp_dir=Path("/tmp") / "biominer_bioclip_crops",
        crop_padding_ratio=vision_settings.crop_padding_ratio,
        crop_target_px=vision_settings.crop_target_px,
        model_id=runtime.model.model_name.removeprefix("hf-hub:"),
        model_version=runtime.package_version,
        model_checkpoint=runtime.model.checkpoint,
    )
    return detector, image_loader, scorer, [persistent, detector]


def _parse_run_stages(
    value: str | None,
    *,
    workflow: str = "adaptive",
) -> tuple[RunStage, ...]:
    workflows = {
        "adaptive": ADAPTIVE_REFERENCE_PRODUCTION_STAGES,
        "legacy": DEFAULT_PRODUCTION_STAGES,
        "reference-first": REFERENCE_FIRST_PRODUCTION_STAGES,
    }
    if workflow not in workflows:
        raise ValueError("workflow must be adaptive, legacy, or reference-first")
    workflow_stages = workflows[workflow]
    if not value:
        return workflow_stages
    stages: list[RunStage] = []
    for raw_part in value.split(","):
        part = raw_part.strip().casefold()
        if not part:
            continue
        if part == "all":
            return workflow_stages
        stage = RUN_STAGE_ALIASES.get(part)
        if stage is None:
            try:
                stage = RunStage(part)
            except ValueError as exc:
                allowed = ", ".join(sorted(RUN_STAGE_ALIASES))
                raise ValueError(f"unknown run stage {raw_part!r}; expected one of: {allowed}") from exc
        if stage not in stages:
            stages.append(stage)
    return tuple(stages) or workflow_stages


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
            args.model_name,
            args.revision,
            BIOCLIP_OPENCLIP_VERSION,
            BIOCLIP_WORKER_SCRIPT,
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


def _run_build_text_embedding_cache(args: argparse.Namespace) -> int:
    from biominer.bioclip.bioclip import PersistentBioClipScorer
    from biominer.bioclip.path_taxonomy_store import PathTaxonomyStore
    from biominer.bioclip.taxonomy_embedding_cache import (
        build_taxonomy_text_embedding_cache,
    )

    runtime_python = Path(args.runtime_python).expanduser()
    if not runtime_python.exists():
        print(json.dumps({"error": f"BioCLIP runtime Python not found: {runtime_python}"}, indent=2, sort_keys=True))
        return 2
    scorer = PersistentBioClipScorer(
        runtime=_bioclip_runtime(runtime_python=runtime_python),
        hf_cache_dir=args.hf_cache_dir,
        device=args.device,
    )
    try:
        store = PathTaxonomyStore.read(args.registry_dir)
        frame = build_taxonomy_text_embedding_cache(
            store,
            model_id=scorer.runtime.model.model_name,
            model_checkpoint=scorer.runtime.model.checkpoint,
            embed_labels=scorer.embed_text_labels,
            batch_size=args.batch_size,
        )
        output = write_parquet(frame, args.output)
    finally:
        scorer.close()
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "rows": frame.height,
                "classification_fingerprint": store.classification_fingerprint,
                "hierarchy_fingerprint": store.hierarchy_fingerprint,
                "embedding_cache_fingerprint": frame["embedding_cache_fingerprint"][0] if frame.height else None,
                "model_id": scorer.runtime.model.model_name,
                "model_checkpoint": scorer.runtime.model.checkpoint,
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


def _run_prototype_vision_smoke(args: argparse.Namespace) -> int:
    from biominer.benchmarks.prototype_vision_smoke import (
        PrototypeVisionSmokeConfig,
        run_prototype_vision_smoke,
    )

    try:
        result = run_prototype_vision_smoke(
            PrototypeVisionSmokeConfig.read_json(args.config)
        )
    except (OSError, TypeError, ValueError, RuntimeError, pl.exceptions.PolarsError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": result.report["status"],
                "image_count": result.report["image_count"],
                "report": str(result.report_path),
                "summary": str(result.summary_path),
                "report_fingerprint": result.report["report_fingerprint"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_prototype_support_embeddings(args: argparse.Namespace) -> int:
    from biominer.benchmarks.prototype_support_embeddings import (
        PrototypeSupportEmbeddingConfig,
        run_prototype_support_embedding_job,
    )

    try:
        result = run_prototype_support_embedding_job(
            PrototypeSupportEmbeddingConfig.read_json(args.config)
        )
    except (OSError, TypeError, ValueError, RuntimeError, pl.exceptions.PolarsError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": result.report["status"],
                "embeddings": str(result.embeddings_path),
                "prototypes": str(result.prototypes_path),
                "visual_neighbours": (
                    str(result.visual_neighbours_path)
                    if result.visual_neighbours_path is not None
                    else None
                ),
                "failures": (
                    str(result.failures_path)
                    if result.failures_path is not None
                    else None
                ),
                "report": str(result.report_path),
                "report_fingerprint": result.report["report_fingerprint"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_prototype_benchmark_matrix(args: argparse.Namespace) -> int:
    from biominer.benchmarks.prototype_benchmark_matrix import (
        PrototypeBenchmarkConfig,
        run_prototype_benchmark_matrix,
    )

    try:
        result = run_prototype_benchmark_matrix(
            PrototypeBenchmarkConfig.read_json(args.config)
        )
    except (OSError, TypeError, ValueError, RuntimeError, pl.exceptions.PolarsError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": result.report["status"],
                "records_scored": result.report["counts"]["records_scored"],
                "records_skipped": result.report["counts"]["records_skipped"],
                "predictions": str(result.predictions_path),
                "experiment_summary": str(result.experiment_summary_path),
                "report": str(result.report_path),
                "report_fingerprint": result.report["report_fingerprint"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_prototype_policy_selection(args: argparse.Namespace) -> int:
    from biominer.benchmarks.prototype_policy_selection import (
        PrototypePolicySelectionConfig,
        select_prototype_policy,
    )

    try:
        result = select_prototype_policy(
            PrototypePolicySelectionConfig.read_json(args.config)
        )
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        pl.exceptions.PolarsError,
    ) as exc:
        print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": result.report["status"],
                "policy_status": result.policy["policy_status"],
                "selected_experiment_id": result.policy["selected_policy"][
                    "experiment_id"
                ],
                "policy": str(result.policy_path),
                "report": str(result.report_path),
                "policy_fingerprint": result.policy["policy_fingerprint"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_prototype_staged_flickr(args: argparse.Namespace) -> int:
    from biominer.benchmarks.prototype_staged_flickr import (
        PrototypeStagedFlickrConfig,
        run_prototype_staged_flickr,
    )

    try:
        result = run_prototype_staged_flickr(
            PrototypeStagedFlickrConfig.read_json(args.config)
        )
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        sqlite3.Error,
        pl.exceptions.PolarsError,
    ) as exc:
        print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": result.report["status"],
                "counts": result.report["counts"],
                "stages": result.report["stages"],
                "results": str(result.results_path),
                "candidates": str(result.candidates_path),
                "failures": (
                    str(result.failures_path)
                    if result.failures_path is not None
                    else None
                ),
                "report": str(result.report_path),
                "report_fingerprint": result.report["report_fingerprint"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_vision_benchmark_plumbing(args: argparse.Namespace) -> int:
    try:
        result = run_vision_plumbing_benchmark(
            records=args.records,
            butterfly_rate=args.butterfly_rate,
            detections_per_butterfly=args.detections_per_butterfly,
            classification_mode=args.classification_mode,
            registry_dir=args.registry_dir,
            output_dir=args.output_dir,
            rank_beam_width=args.rank_beam_width,
            species_first_pass_top_k=args.species_first_pass_top_k,
            species_rerank_top_k=args.species_rerank_top_k,
        )
    except Exception as exc:  # noqa: BLE001 - dev command reports compact failures.
        print(f"benchmark-plumbing failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "benchmark_metrics": str(result.metrics_path),
                "benchmark_summary": str(result.summary_path),
                "output_dir": str(result.output_dir),
                "records": result.metrics["records"],
                "crops_scored": result.metrics["crops_scored"],
                "elapsed_seconds": result.metrics["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_vision_benchmark_rolling_matrix(args: argparse.Namespace) -> int:
    try:
        result = run_rolling_worker_benchmark_matrix(
            records=args.records,
            output_dir=args.output_dir,
        )
    except Exception as exc:  # noqa: BLE001 - dev command reports compact failures.
        print(f"benchmark-rolling-matrix failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "benchmark_metrics": str(result.metrics_path),
                "benchmark_summary": str(result.summary_path),
                "output_dir": str(result.output_dir),
                "records": result.metrics["records"],
                "variant_count": result.metrics["variant_count"],
                "elapsed_seconds": result.metrics["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_path_cascade_benchmark(args: argparse.Namespace) -> int:
    try:
        result = run_path_cascade_benchmark(output_dir=args.output_dir)
    except Exception as exc:  # noqa: BLE001 - dev command reports compact failures.
        print(f"benchmark-cascade failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "benchmark_metrics": str(result.metrics_path),
                "benchmark_summary": str(result.summary_path),
                "output_dir": str(result.output_dir),
                "family_candidate_count": result.metrics["family_candidate_count"],
                "species_candidates_beneath_genus_top3": result.metrics[
                    "species_candidates_beneath_genus_top3"
                ],
                "elapsed_seconds": result.metrics["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_vision_benchmark_live_m5pro(args: argparse.Namespace) -> int:
    request = LiveM5ProBenchmarkRequest(
        input_path=Path(args.input).expanduser(),
        registry_dir=Path(args.registry_dir).expanduser(),
        taxonomy_text_embedding_cache=Path(args.taxonomy_text_embedding_cache).expanduser(),
        vision_runtime_python=Path(args.vision_runtime_python).expanduser(),
        bioclip_runtime_python=Path(args.bioclip_runtime_python).expanduser(),
        hf_cache_dir=Path(args.hf_cache_dir).expanduser(),
        checkpoint=args.checkpoint,
        yolo_sidecar_transport=args.yolo_sidecar_transport,
        device=args.device,
        limit=args.limit,
        output_dir=Path(args.output_dir).expanduser(),
        cache_root=Path(args.cache_root).expanduser(),
        crop_temp_dir=Path(args.crop_temp_dir).expanduser(),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        yolo_batch=args.yolo_batch,
        bioclip_batch=args.bioclip_batch,
        crop_padding_ratio=args.crop_padding_ratio,
        crop_target_px=args.crop_target_px,
        parquet_batch_rows=args.parquet_batch_rows,
        prompt_classes=_yoloe26_prompt_classes(args),
    )
    validation = validate_live_m5pro_benchmark_request(request)
    if validation is not None:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 2
    try:
        result = run_live_m5pro_benchmark(
            request=request,
            bioclip_runtime=_bioclip_runtime(runtime_python=request.bioclip_runtime_python),
        )
    except Exception as exc:  # noqa: BLE001 - live command should fail compactly.
        print(
            json.dumps(
                {
                    "benchmark_kind": "vision_live_m5pro",
                    "error": "benchmark_live_m5pro_failed",
                    "message": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "benchmark_metrics": str(result.metrics_path),
                "benchmark_summary": str(result.summary_path),
                "output_dir": str(result.output_dir),
                "records_loaded": result.metrics["records_loaded"],
                "elapsed_seconds": result.metrics["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


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
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
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




def _bioclip_runtime(
    *,
    runtime_python: Path,
    model_name: str = BIOCLIP_25_HUGE_REPO_ID,
) -> BioClipRuntime:
    model = ModelConfig(
        model_id="bioclip2_5_huge",
        display_name="BioCLIP 2.5 Huge",
        role="preferred",
        status="use_if_available",
        task="biology image-text classification and embedding",
        model_name=str(model_name).strip(),
        checkpoint=BIOCLIP_25_HUGE_REVISION,
        package_name="open_clip_torch",
        package_version=BIOCLIP_OPENCLIP_VERSION,
        model_hash=f"hf-revision:{BIOCLIP_25_HUGE_REVISION}",
    )
    return BioClipRuntime(
        model=model,
        home=runtime_python.parent.parent,
        venv_python=runtime_python,
        package_version=BIOCLIP_OPENCLIP_VERSION,
        available=True,
    )


def _bioclip_worker_env(hf_cache_dir: str | Path) -> dict[str, str]:
    env = os.environ.copy()
    source_path = str(Path(__file__).resolve().parents[1])
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        source_path if not current else f"{source_path}{os.pathsep}{current}"
    )
    cache_path = Path(hf_cache_dir).resolve()
    hub_path = cache_path / "hub"
    hub_path.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(cache_path)
    env["HUGGINGFACE_HUB_CACHE"] = str(hub_path)
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return env


_BIOCLIP_RUNTIME_CHECK_SCRIPT = r"""
from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import sys

import torch

requested = sys.argv[1]
model_id = sys.argv[2]
model_revision = sys.argv[3]
expected_open_clip_version = sys.argv[4]
worker_path = Path(sys.argv[5]).resolve(strict=True)
spec = importlib.util.spec_from_file_location("_biominer_bioclip_runtime_worker", worker_path)
if spec is None or spec.loader is None:
    raise SystemExit("BioCLIP worker module could not be loaded")
worker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = worker
spec.loader.exec_module(worker)
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
loaded = worker._LoadedBioClipModel.load(
    model_name=model_id,
    checkpoint=model_revision,
    device=requested,
    image_resize_mode="longest",
)
try:
    if loaded.open_clip_version != expected_open_clip_version:
        raise SystemExit(
            "OpenCLIP version mismatch: "
            f"expected {expected_open_clip_version}, got {loaded.open_clip_version}"
        )
    if loaded.image_resize_mode != "longest":
        raise SystemExit("BioCLIP runtime did not apply longest-side preprocessing")
    mps_fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    print(json.dumps({
        "runtime_python": sys.executable,
        "device_requested": requested,
        "device_resolved": loaded.device,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": mps_available,
        "model_load": True,
        "tokenizer_load": loaded.tokenizer is not None,
        **loaded.worker_metadata,
        "pytorch_mps_fallback_env": mps_fallback,
        "pytorch_mps_fallback_enabled": mps_fallback == "1",
        "pytorch_mps_fallback_recommendation": "set PYTORCH_ENABLE_MPS_FALLBACK=1 for Apple MPS sidecar runs",
        "torch_version": torch.__version__,
    }, sort_keys=True))
finally:
    loaded.close()
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
import os
from pathlib import Path
import sys

import torch
from ultralytics import YOLOE

requested = sys.argv[1]
checkpoint = sys.argv[2]
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
if requested == "auto":
    if torch.cuda.is_available():
        resolved = "cuda"
    elif mps_available:
        resolved = "mps"
    else:
        resolved = "cpu"
elif requested == "cuda" and not torch.cuda.is_available():
    raise SystemExit("CUDA was requested but is not available")
elif requested == "mps" and not mps_available:
    raise SystemExit("MPS was requested but is not available")
else:
    resolved = requested

model = YOLOE(checkpoint)
model.set_classes(["butterfly", "moth", "caterpillar", "pupa"])
model_dir = Path(os.environ.get("BIOMINER_YOLO26_MODEL_DIR") or Path.cwd())
checkpoint_path = model_dir / checkpoint
cuda_device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
mps_fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
print(json.dumps({
    "runtime_python": sys.executable,
    "checkpoint": checkpoint,
    "checkpoint_path": str(checkpoint_path if checkpoint_path.exists() else checkpoint),
    "checkpoint_resolved": True,
    "device_requested": requested,
    "device_resolved": resolved,
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_name": cuda_device_name,
    "mps_available": mps_available,
    "pytorch_mps_fallback_env": mps_fallback,
    "pytorch_mps_fallback_enabled": mps_fallback == "1",
    "pytorch_mps_fallback_recommendation": "set PYTORCH_ENABLE_MPS_FALLBACK=1 for Apple MPS sidecar runs",
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
