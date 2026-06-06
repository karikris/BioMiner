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
        "image_triage_profile.json": build_image_triage_profile(),
        "cache_profile.json": build_cache_profile(),
        "gpu_profile.json": build_gpu_profile(),
        "idempotency_profile.json": build_idempotency_profile(),
        "code_cleanup_report.md": build_code_cleanup_report_markdown(),
        "agents_update_recommendations.json": build_agents_update_recommendations(),
    }


def base_payload() -> dict[str, Any]:
    return {
        "generated_at": GENERATED_AT,
        "phase": "Lean image triage pipeline",
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
            "image_triage_pipeline": {
                "module": "src/flickr_bio_occurrence/vision/triage.py",
                "output": "image_triage.parquet",
                "stores_downloaded_images": False,
                "keeps_metadata_urls_hashes_scores": True,
                "geo_time_fields": ["latitude", "longitude", "date_taken", "date_upload", "captured_at", "year", "month"],
                "triage_bins": ["gold", "silver", "bronze", "in_review"],
                "classification_statuses": ["success", "skipped_existing", "failed_download", "failed_bioclip", "invalid_record"],
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
            "darwin_core_scope": {
                "active_image_triage_dependency": False,
                "compatibility_only": True,
                "notes": "Darwin Core export and occurrence publication logic are retained only for existing compatibility tests.",
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
                "lean_triage_tests": "tests/test_image_triage.py and tests/test_report_pack.py",
            },
        }
    )
    return payload


def build_image_triage_profile() -> dict[str, Any]:
    payload = base_payload()
    payload.update(
        {
            "output_dataset": "image_triage.parquet",
            "required_fields": [
                "source",
                "source_record_id",
                "flickr_photo_id",
                "photo_page_url",
                "image_url",
                "image_url_kind",
                "latitude",
                "longitude",
                "date_taken",
                "date_upload",
                "captured_at",
                "year",
                "month",
                "source_record_hash",
                "image_hash",
                "image_downloaded",
                "image_deleted_after_classification",
                "classification_status",
                "classification_error",
                "model_id",
                "model_version",
                "model_checkpoint",
                "classified_at",
                "bioclip_top1_label",
                "bioclip_top1_score",
                "bioclip_topk_json",
                "triage_bin",
                "triage_reason",
                "is_target_positive",
                "is_negative_material",
            ],
            "records_seen": None,
            "records_classified": None,
            "records_skipped_existing": None,
            "download_failures": None,
            "bioclip_failures": None,
            "gold_count": None,
            "silver_count": None,
            "bronze_count": None,
            "in_review_count": None,
            "bronze_reason_counts": "not_instrumented",
            "image_cache_bytes_before": "not_instrumented",
            "image_cache_bytes_after": "not_instrumented",
            "images_deleted_after_classification": None,
            "duplicate_skipped_count": None,
            "model_load_count": "not_instrumented",
            "images_per_second": None,
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
                "image_cache_bytes_before": "not_instrumented",
                "image_cache_bytes_after": "not_instrumented",
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
                "file": "src/flickr_bio_occurrence/vision/triage.py",
                "symbol": "process_image_triage_records",
                "model_lifecycle": "classifier is externally supplied; model_load_count is not instrumented by this report pack",
            },
            "model_load_count": "not_instrumented",
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
            "image_triage_deduplication": {
                "file": "src/flickr_bio_occurrence/vision/triage.py",
                "symbol": "process_image_triage_records",
                "behavior": "successful source/photo/image_url/model combinations are skipped on rerun",
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
                    "recommendation": "Do not expand occurrence publication logic in the image-triage phase; use triage_bin only.",
                },
                {
                    "topic": "review_reason",
                    "recommendation": "Use triage_reason for image triage; keep review_reason only in legacy compatibility paths.",
                },
                {
                    "topic": "BioCLIP runtime",
                    "recommendation": "Use an externally owned BioCLIP 2.5 classifier with process_image_triage_records; do not restart the model per image.",
                },
                {
                    "topic": "cache cleanup",
                    "recommendation": "Keep successful image deletion as the default; require explicit diagnostic config for originals or failed-image retention.",
                },
            ],
            "remaining_unavailable_features": [
                "dedicated Flickr comments API fetching",
                "validated Darwin Core occurrence publication",
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
            f"- Image triage output: `{summary['image_triage_pipeline']['output']}`.",
            f"- Triage bins: `{summary['image_triage_pipeline']['triage_bins']}`.",
            f"- Geo/time fields retained: `{summary['image_triage_pipeline']['geo_time_fields']}`.",
            f"- Cache cleanup handled: `{summary['cache_cleanup']['handled']}`.",
            "",
            "## Notes",
            "",
            "- BioCLIP output is screening evidence only, not taxonomic proof.",
            "- Dedicated comments API fetching and validated Darwin Core publication remain explicitly unavailable in this phase.",
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
            "- The active flow is now a lean image-triage pipeline centered on `image_triage.parquet`.",
            "- Image selection defaults to `url_l -> url_m`; originals are not selected by default.",
            "- BioCLIP output is stored as model evidence only, not taxonomic validation.",
            "- Successful cached images are deleted by default after prediction writes.",
            "- Darwin Core export remains compatibility-only and is not expanded in the active triage flow.",
            "",
            "## Remaining Explicit Gaps",
            "",
            "- Dedicated Flickr comment API fetching remains unavailable and reported through `fetch-comments`.",
            "- Validated Darwin Core occurrence publication, multi-GPU, and dashboard workflows remain out of scope.",
        ]
    ) + "\n"


if __name__ == "__main__":
    main()
