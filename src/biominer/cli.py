from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
from typing import Any

import polars as pl

from biominer.bioclip.model_registry import BioClipRuntime, ModelConfig
from biominer.detection.evaluate import evaluate_xie_style
from biominer.evaluation.sampling import (
    EvaluationSamplingConfig,
    materialize_evaluation_sampling_frame,
)
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
from biominer.secrets_loader import load_runtime_secrets_env
from biominer.config import ConfigError, create_storage_backend, create_workstore, load_biominer_config, redact_config, redact_text, validate_config
from biominer.storage.handoff import (
    build_handoff_bundle,
    receive_handoff_bundle,
    upload_handoff_bundle,
)
from biominer.storage.uri import is_cloud_uri, join_uri


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
    "reference_embeddings": RunStage.REFERENCE_EMBEDDINGS,
    "reference_prototypes": RunStage.REFERENCE_PROTOTYPES,
    "flickr_detection": RunStage.FLICKR_DETECTION,
    "flickr_embedding": RunStage.FLICKR_EMBEDDING,
    "reference_deduplication": RunStage.REFERENCE_DEDUPLICATION,
    "reference_quality_routing": RunStage.REFERENCE_QUALITY_ROUTING,
    "reference_admission": RunStage.REFERENCE_ADMISSION,
    "reference_geography_index": RunStage.REFERENCE_GEOGRAPHY_INDEX,
    "flickr_geo_taxon_partitioning": RunStage.FLICKR_GEO_TAXON_PARTITIONING,
    "family_routing": RunStage.FAMILY_ROUTING,
    "dynamic_pool_planning": RunStage.DYNAMIC_POOL_PLANNING,
    "dynamic_pool_scoring": RunStage.DYNAMIC_POOL_SCORING,
    "provisional_flickr_scoring": RunStage.PROVISIONAL_FLICKR_SCORING,
    "review_sample_planning": RunStage.REVIEW_SAMPLE_PLANNING,
    "flickr_human_verification": RunStage.FLICKR_HUMAN_VERIFICATION,
    "risk_controlled_audit": RunStage.RISK_CONTROLLED_AUDIT,
    "statistical_reference_audit": RunStage.STATISTICAL_REFERENCE_AUDIT,
    "targeted_reference_review": RunStage.TARGETED_REFERENCE_REVIEW,
    "affected_reference_rebuild": RunStage.AFFECTED_REFERENCE_REBUILD,
    "affected_record_rescore": RunStage.AFFECTED_RECORD_RESCORE,
    "final_quality_gate": RunStage.FINAL_QUALITY_GATE,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biominer")
    parser.add_argument("--config")
    parser.add_argument("--version", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    evaluation = subparsers.add_parser("evaluation")
    evaluation_subparsers = evaluation.add_subparsers(dest="evaluation_command")
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
    production_run.add_argument("--stages")
    production_run.add_argument("--dry-run", action="store_true")
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
        if args.vision_command == "yoloe26-runtime-check":
            return _run_yoloe26_runtime_check(args)
        if args.vision_command == "yoloe26-prefetch":
            return _run_yoloe26_prefetch(args)
        if args.vision_command == "yoloe26-smoke":
            return _run_yoloe26_smoke(args)
        if args.vision_command == "eval":
            return _run_detect_eval(args)
        return 2
    if args.command == "evaluation":
        return _run_evaluation_command(args)
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


def _run_production_command(args: argparse.Namespace) -> int:
    config = None
    try:
        stages = _parse_run_stages(args.stages)
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
        limits = {
            key: value
            for key, value in {
                "species": args.limit_species,
                "records": args.limit_records,
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
            limits=limits,
        )
        plan = ProductionRunOrchestrator(
            request,
            storage=storage,
        ).run()
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        payload: dict[str, object] = {"error": redact_text(str(exc), config) if config else str(exc)}
        if config is not None:
            payload["config"] = redact_config(config)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    return 0


def _parse_run_stages(
    value: str | None,
) -> tuple[RunStage, ...]:
    if not value:
        return ADAPTIVE_REFERENCE_PRODUCTION_STAGES
    stages: list[RunStage] = []
    for raw_part in value.split(","):
        part = raw_part.strip().casefold()
        if not part:
            continue
        if part == "all":
            return ADAPTIVE_REFERENCE_PRODUCTION_STAGES
        stage = RUN_STAGE_ALIASES.get(part)
        if stage is None:
            try:
                stage = RunStage(part)
            except ValueError as exc:
                allowed = ", ".join(sorted(RUN_STAGE_ALIASES))
                raise ValueError(f"unknown run stage {raw_part!r}; expected one of: {allowed}") from exc
        if stage not in stages:
            stages.append(stage)
    return tuple(stages) or ADAPTIVE_REFERENCE_PRODUCTION_STAGES


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
