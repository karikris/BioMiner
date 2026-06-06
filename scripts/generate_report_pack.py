from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


GENERATED_AT = datetime.now(UTC).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_report_pack(output_dir)


def write_report_pack(output_dir: Path) -> None:
    reports = build_reports()
    for name, payload in reports.items():
        if name.endswith(".json"):
            (output_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        else:
            (output_dir / name).write_text(str(payload), encoding="utf-8")


def build_reports() -> dict[str, Any]:
    summary = build_bioclip_run_summary()
    return {
        "bioclip_run_summary.json": summary,
        "bioclip_run_summary.md": build_bioclip_run_summary_markdown(summary),
        "storage_profile.json": build_storage_profile(),
        "cache_profile.json": build_cache_profile(),
        "gpu_profile.json": build_gpu_profile(),
        "quality_profile.json": build_quality_profile(),
        "idempotency_profile.json": build_idempotency_profile(),
        "code_cleanup_report.md": build_code_cleanup_report_markdown(),
        "agents_update_recommendations.json": build_agents_update_recommendations(),
    }


def base_payload() -> dict[str, Any]:
    return {
        "generated_at": GENERATED_AT,
        "phase": "Phase 6: CLI cleanup and final integration",
        "source": "static code audit; no network, CUDA, model weights, Flickr downloads, or large data files required",
    }


def build_bioclip_run_summary() -> dict[str, Any]:
    payload = base_payload()
    payload.update(
        {
            "python_target": ">=3.14",
            "bioclipminer_local": {
                "exists": Path("/mnt/c/WSL-Shared/Projects/BioCLIPMiner").exists(),
                "path": "/mnt/c/WSL-Shared/Projects/BioCLIPMiner",
                "inspected_files": [
                    "/mnt/c/WSL-Shared/Projects/BioCLIPMiner/requirements.txt",
                    "/mnt/c/WSL-Shared/Projects/BioCLIPMiner/smoke_bioclip.py",
                ],
                "notes": "Local BioCLIPMiner is minimal: requirements plus a smoke script that loads local BioCLIP 2.5 weights.",
            },
            "comment_handling": {
                "comments_endpoint_allowed": {
                    "value": True,
                    "file": "src/flickr_bio_occurrence/flickr/endpoints.py",
                    "symbol": "ALLOWED_FLICKR_METHODS",
                },
                "comments_currently_fetched": {
                    "value": False,
                    "file": "src/flickr_bio_occurrence/cli.py",
                    "symbol": "fetch-comments",
                    "notes": "Comment fetching is exposed as an audited unavailable command; no network comments fetch is implemented.",
                },
                "comments_stored_in_raw_payloads": {
                    "value": False,
                    "file": "src/flickr_bio_occurrence/flickr/client.py",
                    "symbol": "FlickrClient._write_raw_response",
                    "notes": "Raw payloads are photos_search responses only; comments are not fetched before writing.",
                },
                "comments_transformed_to_parquet": {
                    "value": False,
                    "file": "src/flickr_bio_occurrence/pipeline/transforms.py",
                    "symbol": "flatten_search_payloads",
                    "notes": "Transform reads payload['photos']['photo']; no comment fields are emitted.",
                },
                "comments_scanned_for_scientific_names_or_verification_phrases": {
                    "value": False,
                    "files_checked": [
                        "src/flickr_bio_occurrence",
                        "scripts",
                        "tests",
                    ],
                    "notes": "Existing raw payload comments are parsed by the evidence extractor when present, but no dedicated comments API fetch is implemented.",
                },
            },
            "image_selection": {
                "order": ["url_l", "url_m"],
                "default": "large-first, medium fallback only",
                "file": "src/flickr_bio_occurrence/vision/image_selection.py",
                "symbol": "select_flickr_image_url",
                "notes": "Original url_o is requested in Flickr extras but not selected by default.",
            },
            "cli_commands": {
                "file": "src/flickr_bio_occurrence/cli.py",
                "commands": [
                    "fetch-live",
                    "fetch-comments",
                    "build-evidence",
                    "classify-once",
                    "classify-watch",
                    "apply-rules",
                    "gc-cache",
                    "compact-parquet",
                    "qa-summary",
                ],
                "existing_compatibility_preserved": ["fetch", "qa-rate-limit", "qa-estimate", "qa-estimate-combined", "benchmark-existing-payloads"],
            },
            "evidence_first_pipeline": {
                "evidence_extractor": "src/flickr_bio_occurrence/evidence/extractor.py",
                "rule_engine": "src/flickr_bio_occurrence/evidence/rules.py",
                "job_queue": "src/flickr_bio_occurrence/pipeline/job_queue.py",
                "sharded_fetch": "src/flickr_bio_occurrence/pipeline/sharded_fetch.py",
                "vision_service": "src/flickr_bio_occurrence/vision/service.py",
                "one_evidence_row_per_photo_record": True,
                "one_publication_state_per_record": True,
                "in_review_requires_review_reason": True,
            },
            "bioclip_labels_and_agreement_rules": {
                "labels_file": "src/flickr_bio_occurrence/vision/bioclip.py",
                "labels_symbol": "DEFAULT_BIOCLIP_LABELS",
                "agreement_symbols": [
                    "PAPILIO_DEMOLEUS_VISUAL_LABELS",
                    "SWALLOWTAIL_VISUAL_LABELS",
                    "BUTTERFLY_VISUAL_LABELS",
                    "NON_WILD_OR_CONFLICT_LABELS",
                    "classify_species_agreement",
                ],
                "screening_evidence_only": True,
                "auto_validates_dwc": False,
            },
            "bronze_silver_gold_semantics": {
                "currently_implied_by_storage_layout": False,
                "files_and_symbols": [
                    {
                        "file": "src/flickr_bio_occurrence/pipeline/transforms.py",
                        "symbols": ["flatten_search_payloads", "build_silver_candidates", "build_dwc_rows"],
                    },
                    {
                        "file": "src/flickr_bio_occurrence/benchmark/offline_run.py",
                        "symbols": ["_write_outputs"],
                    },
                    {
                        "file": "src/flickr_bio_occurrence/storage/duckdb_index.py",
                        "symbols": ["create_qa_views"],
                    },
                ],
                "publication_state_field_present": True,
                "review_reason_field_present": True,
                "review_status_present": True,
                "notes": "Gold/Silver/Bronze/In Review are represented as explicit publication_state values by the evidence rules.",
            },
            "prediction_checkpoints": {
                "file": "src/flickr_bio_occurrence/benchmark/vision_checkpoint.py",
                "symbols": ["build_checkpointed_vision_predictions"],
                "layout": "silver/silver_vision_prediction/model_version=<model_id>/run_id=<run_id>/shard_id=<shard_id>/part-00000.parquet",
                "idempotency": "Existing predictions are skipped by flickr_photo_id, image_hash, model_version, and model_checkpoint.",
            },
            "cache_cleanup": {
                "handled": True,
                "image_cache_file": "src/flickr_bio_occurrence/vision/pipeline.py",
                "image_cache_symbol": "classify_bronze_photo_row",
                "notes": "Successful images are deleted by default after prediction; failed images are deleted by default unless keep_failed_images is enabled.",
            },
            "tests": {
                "network_required": False,
                "cuda_required": False,
                "model_weights_required": False,
                "phase_6_focused_tests": "tests/test_cli_dry_run.py, tests/test_report_pack.py, tests/test_streaming_jobs.py, evidence and BioCLIP focused tests",
            },
        }
    )
    return payload


def build_storage_profile() -> dict[str, Any]:
    payload = base_payload()
    payload.update(
        {
            "raw_flickr_payloads": {
                "photos_search_path": "data/raw/flickr/photos_search/",
                "comments_path": None,
                "comments_stored": False,
            },
            "parquet_outputs": {
                "evidence": "staging/evidence/",
                "bronze": "bronze/bronze_flickr_photo",
                "silver_occurrence_candidate": "silver/silver_occurrence_candidate",
                "silver_vision_prediction": "silver/silver_vision_prediction",
                "gold_dwc_occurrence": "gold/dwc_occurrence",
            },
            "duckdb": {
                "offline_benchmark_path": "existing_payload_benchmark.duckdb",
                "views_file": "src/flickr_bio_occurrence/storage/duckdb_index.py",
            },
            "classification_queue": {
                "default_path": "classification_jobs.sqlite",
                "fields": ["job_id", "evidence_parquet_path", "status", "model_version", "created_at", "claimed_at", "completed_at", "attempts", "error"],
            },
            "measured_bytes": "not_instrumented",
        }
    )
    return payload


def build_cache_profile() -> dict[str, Any]:
    payload = base_payload()
    payload.update(
        {
            "image_cache": {
                "default_root": "data/cache/images",
                "file": "src/flickr_bio_occurrence/vision/image_cache.py",
                "symbol": "cache_image_from_url",
                "layout": "<sha256[0:2]>/<sha256[2:4]>/<sha256>.<extension>",
                "cleanup": "not_instrumented",
                "successful_images_deleted_by_default": True,
                "keep_failed_images_default": False,
            },
            "huggingface_cache": {
                "default_root": "data/cache/huggingface",
                "file": "src/flickr_bio_occurrence/vision/bioclip_worker.py",
                "symbol": "configure_hf_cache_env",
                "separate_from_flickr_image_cache": True,
            },
        }
    )
    return payload


def build_gpu_profile() -> dict[str, Any]:
    payload = base_payload()
    payload.update(
        {
            "requires_cuda_for_tests": False,
            "runtime_detection": {
                "file": "src/flickr_bio_occurrence/vision/bioclip_worker.py",
                "symbol": "score_image",
                "device_expression": "requires CUDA by default; CPU fallback is not used when require_cuda is true",
            },
            "long_lived_service": {
                "file": "src/flickr_bio_occurrence/vision/service.py",
                "symbol": "BioClipJobService",
                "model_lifecycle": "externally owned classifier can process multiple claimed jobs without reinitialization",
            },
            "benchmark_reporting": {
                "file": "src/flickr_bio_occurrence/benchmark/vision_checkpoint.py",
                "symbol": "build_checkpointed_vision_predictions",
                "gpu_used": "reported when classifier scorer exposes device == 'cuda'",
                "gpu_name": "reported when classifier scorer exposes gpu_name",
            },
            "measured_gpu_name": "not_instrumented",
            "measured_cuda_available": "not_instrumented",
        }
    )
    return payload


def build_quality_profile() -> dict[str, Any]:
    payload = base_payload()
    payload.update(
        {
            "bioclip_role": "screening_evidence",
            "taxonomic_proof": False,
            "auto_validates_dwc_records": False,
            "agreement_rules_file": "src/flickr_bio_occurrence/vision/bioclip.py",
            "agreement_function": "classify_species_agreement",
            "default_review_status": {
                "file": "src/flickr_bio_occurrence/evidence/rules.py",
                "symbol": "classify_evidence_row",
                "value": "publication_state plus review_reason",
            },
            "manual_ground_truth_accuracy": None,
            "quality_metrics": "not_instrumented",
        }
    )
    return payload


def build_idempotency_profile() -> dict[str, Any]:
    payload = base_payload()
    payload.update(
        {
            "flickr_fetch_state": {
                "file": "scripts/run_butterfly_flickr_fetch.py",
                "symbol": "FetchState",
                "state_db": "fetch_state.sqlite",
                "behavior": "completed work_item_id rows are skipped; exhausted terms are skipped",
            },
            "rate_limit_ledger": {
                "file": "src/flickr_bio_occurrence/flickr/rate_limiter.py",
                "symbol": "FlickrRateLimiter",
                "api_call_table": "api_call_ledger",
                "photo_record_table": "photo_record_ledger",
            },
            "vision_checkpoints": {
                "file": "src/flickr_bio_occurrence/benchmark/vision_checkpoint.py",
                "symbol": "build_checkpointed_vision_predictions",
                "behavior": "existing prediction keys are skipped and new predictions are written in partitioned shards",
            },
            "classification_jobs": {
                "file": "src/flickr_bio_occurrence/pipeline/job_queue.py",
                "symbol": "ClassificationJobQueue",
                "behavior": "completed jobs are skipped on rerun; stale claimed jobs can be retried or failed",
            },
        }
    )
    return payload


def build_agents_update_recommendations() -> dict[str, Any]:
    payload = base_payload()
    payload.update(
        {
            "recommendations": [
                {
                    "topic": "comments",
                    "recommendation": "Dedicated Flickr comment fetching remains unavailable; keep the CLI audit explicit until API-backed comment tests are added.",
                },
                {
                    "topic": "publication_state",
                    "recommendation": "Keep publication_state and review_reason as explicit evidence-rule outputs; avoid reintroducing directory-only state semantics.",
                },
                {
                    "topic": "review_reason",
                    "recommendation": "Maintain tests that every in_review row has at least one review_reason.",
                },
                {
                    "topic": "BioCLIP runtime",
                    "recommendation": "Use BioClipJobService or PersistentBioClipScorer for production classification so the model remains loaded across jobs.",
                },
                {
                    "topic": "cache cleanup",
                    "recommendation": "Keep successful image deletion as the default; require explicit diagnostic config for originals or failed-image retention.",
                },
            ],
            "remaining_unavailable_features": [
                "dedicated Flickr comments API fetching",
                "multi-GPU classification",
                "dashboard",
            ],
        }
    )
    return payload


def build_bioclip_run_summary_markdown(summary: dict[str, Any]) -> str:
    comments = summary["comment_handling"]
    return "\n".join(
        [
            "# BioCLIP Run Summary",
            "",
            f"Generated: {summary['generated_at']}",
            "",
            "## Final Integration Findings",
            "",
            f"- BioCLIPMiner local path exists: `{summary['bioclipminer_local']['exists']}`.",
            f"- Flickr comments fetched: `{comments['comments_currently_fetched']['value']}`.",
            f"- Comments stored in raw payloads: `{comments['comments_stored_in_raw_payloads']['value']}`.",
            f"- Comments transformed to parquet: `{comments['comments_transformed_to_parquet']['value']}`.",
            f"- Comments searched for scientific names/verification phrases: `{comments['comments_scanned_for_scientific_names_or_verification_phrases']['value']}`.",
            f"- Image selection order: `{summary['image_selection']['order']}`.",
            f"- CLI commands: `{summary['cli_commands']['commands']}`.",
            f"- Evidence-first pipeline present: `{bool(summary['evidence_first_pipeline'])}`.",
            f"- One publication_state per record: `{summary['evidence_first_pipeline']['one_publication_state_per_record']}`.",
            f"- Prediction checkpoint layout: `{summary['prediction_checkpoints']['layout']}`.",
            f"- Cache cleanup handled: `{summary['cache_cleanup']['handled']}`.",
            "",
            "## Notes",
            "",
            "- BioCLIP output is screening evidence only, not taxonomic proof.",
            "- Dedicated comments API fetching remains explicitly unavailable.",
            "- No network, CUDA, real BioCLIP weights, or real Flickr downloads are required to generate this report pack.",
        ]
    ) + "\n"


def build_code_cleanup_report_markdown() -> str:
    return "\n".join(
        [
            "# Code Cleanup Report",
            "",
            f"Generated: {GENERATED_AT}",
            "",
            "## Final Integration Findings",
            "",
            "- Evidence extraction, evidence rules, streaming job queue, and long-lived classification service are present.",
            "- CLI commands expose fetch-live, fetch-comments audit, build-evidence, classify-once/watch, apply-rules, gc-cache, compact-parquet, and qa-summary.",
            "- Image selection defaults to `url_l -> url_m`; originals are not selected by default.",
            "- BioCLIP batching and persistent worker paths avoid model restart per image/job.",
            "- Prediction outputs use partitioned batch parquet instead of one file per photo.",
            "- Successful cached images are deleted by default after prediction writes.",
            "",
            "## Remaining Explicit Gaps",
            "",
            "- Dedicated Flickr comment API fetching remains unavailable and reported through `fetch-comments`.",
            "- Multi-GPU and dashboard workflows remain out of scope.",
        ]
    ) + "\n"


if __name__ == "__main__":
    main()
