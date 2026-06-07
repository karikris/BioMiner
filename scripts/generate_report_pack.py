from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


GENERATED_AT = datetime.now(UTC).isoformat()
PHASE = "Comment review for mismatches and missing-data recovery"
UNINSTRUMENTED = "not_instrumented"

REQUIRED_METRICS = {
    "gold_count": None,
    "silver_count": None,
    "bronze_count": None,
    "in_review_no_geo_count": None,
    "adult_butterfly_count": None,
    "egg_count": None,
    "caterpillar_count": None,
    "larva_count": None,
    "pupa_count": None,
    "chrysalis_count": None,
    "museum_specimen_count": None,
    "artwork_count": None,
    "tattoo_count": None,
    "ai_generated_count": None,
    "other_insect_count": None,
    "downloaded_images_deleted_count": None,
    "duplicate_skipped_count": None,
    "api_calls_used": None,
}
COMMENT_REVIEW_METRICS = {
    "comment_review_queue_created": None,
    "comment_calls_used": None,
    "comments_fetched": None,
    "species_conflicts_reviewed": None,
    "species_conflicts_resolved": None,
    "records_moved_to_gold": None,
    "records_kept_in_review_no_geo": None,
    "missing_geo_requests_created": None,
    "missing_date_requests_created": None,
    "comment_derived_terms_created": None,
    "comment_review_failures": None,
}

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
    reports: dict[str, Any] = {
        "query_term_totals.json": build_query_term_totals(),
        "bbox_coverage_profile.json": build_bbox_coverage_profile(),
        "occurrence_bin_profile.json": build_occurrence_bin_profile(),
        "life_stage_profile.json": build_life_stage_profile(),
        "no_geo_profile.json": build_no_geo_profile(),
        "comment_expansion_profile.json": build_comment_expansion_profile(),
        "comment_review_profile.json": build_comment_review_profile(),
        "missing_data_requests.json": build_missing_data_requests_profile(),
        "comment_species_resolution.json": build_comment_species_resolution_profile(),
        "api_budget_profile.json": build_api_budget_profile(),
        "code_cleanup_report.md": build_code_cleanup_report_markdown(),
        "agents_update_recommendations.json": build_agents_update_recommendations(),
    }
    reports["query_term_totals.md"] = build_query_term_totals_markdown(reports["query_term_totals.json"])
    reports["bbox_coverage_profile.md"] = build_bbox_coverage_profile_markdown(reports["bbox_coverage_profile.json"])
    reports["occurrence_bin_profile.md"] = build_occurrence_bin_profile_markdown(reports["occurrence_bin_profile.json"])
    reports["comment_review_profile.md"] = build_comment_review_profile_markdown(reports["comment_review_profile.json"])
    return reports


def base_payload() -> dict[str, Any]:
    return {
        "generated_at": GENERATED_AT,
        "phase": PHASE,
        "source": "static code audit; no network, CUDA, model weights, Flickr downloads, parquet, or DuckDB files required",
        "unsupported_metrics_policy": "Unsupported metrics are null or not_instrumented, never guessed.",
    }


def with_required_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    return {**base_payload(), **REQUIRED_METRICS, **COMMENT_REVIEW_METRICS, **payload}


def build_query_term_totals() -> dict[str, Any]:
    return with_required_metrics(
        {
            "report": "query_term_totals",
            "state_table": "flickr_work_items",
            "query_source": "Flickr search terms plus thresholded comments-derived terms",
            "query_term_counts": UNINSTRUMENTED,
            "comments_promoted_term_counts": UNINSTRUMENTED,
            "files": [
                "src/flickr_bio_occurrence/flickr/query_planner.py",
                "src/flickr_bio_occurrence/pipeline/comments_enrichment.py",
            ],
        }
    )


def build_bbox_coverage_profile() -> dict[str, Any]:
    return with_required_metrics(
        {
            "report": "bbox_coverage_profile",
            "bbox_queries_supported": True,
            "bbox_query_lane": "bbox_page",
            "bbox_counts": UNINSTRUMENTED,
            "coverage_notes": "Worldwide high-confidence queries are retained for discovery; broad queries can split into known-region bbox work items, and geotagged records outside configured regions are retained as review candidates.",
            "files": ["src/flickr_bio_occurrence/flickr/query_planner.py"],
        }
    )


def build_occurrence_bin_profile() -> dict[str, Any]:
    return with_required_metrics(
        {
            "report": "occurrence_bin_profile",
            "screening_evidence_only": True,
            "occurrence_bins": ["gold", "silver", "bronze", "in_review", "in_review/no_geo"],
            "semantics": {
                "gold": "BioCLIP target/butterfly-positive score >= 0.50",
                "silver": "BioCLIP target/butterfly-positive score < 0.50",
                "bronze": "negative or unusable butterfly-occurrence material",
                "in_review": "operational failure, invalid, missing, or unresolved record",
                "in_review/no_geo": "target-positive image evidence with missing geolocation",
            },
            "required_fields": [
                "occurrence_bin",
                "triage_bin",
                "bin_reason",
                "triage_reason",
                "image_category",
                "life_stage",
                "bioclip_top1_label",
                "bioclip_top1_score",
                "image_deleted_after_classification",
            ],
            "bin_reason_counts": UNINSTRUMENTED,
            "darwin_core_dependency": False,
        }
    )


def build_life_stage_profile() -> dict[str, Any]:
    return with_required_metrics(
        {
            "report": "life_stage_profile",
            "life_stage_values": ["adult_butterfly", "egg", "caterpillar", "larva", "pupa", "chrysalis", "unknown"],
            "image_category_values": [
                "adult_butterfly",
                "life_stage_non_adult",
                "museum_specimen",
                "artwork",
                "tattoo",
                "ai_generated",
                "logo_or_brand",
                "object_or_product",
                "textile_or_pattern",
                "other_insect",
                "not_lepidoptera",
                "unknown",
            ],
            "life_stage_source": "BioCLIP top-1 labels and source/comment text terms are screening evidence only.",
            "files": [
                "src/flickr_bio_occurrence/evidence/category_model.py",
                "src/flickr_bio_occurrence/vision/triage.py",
                "src/flickr_bio_occurrence/pipeline/comments_enrichment.py",
            ],
        }
    )


def build_no_geo_profile() -> dict[str, Any]:
    return with_required_metrics(
        {
            "report": "no_geo_profile",
            "no_geo_bin": "in_review/no_geo",
            "geo_fields_retained": ["latitude", "longitude", "accuracy"],
            "time_fields_retained": ["date_taken", "date_upload", "captured_at", "year", "month"],
            "no_geo_reason_counts": UNINSTRUMENTED,
        }
    )


def build_comment_expansion_profile() -> dict[str, Any]:
    return with_required_metrics(
        {
            "report": "comment_expansion_profile",
            "comments_endpoint_allowed": True,
            "comments_fetch_scope": "selected candidate records only",
            "hard_negative_bronze_skipped_unless_selected_for_qa": True,
            "term_kinds": ["scientific_name", "common_name", "life_stage"],
            "promotion_threshold": {
                "min_distinct_photos_default": 2,
                "min_distinct_users_default": 2,
            },
            "comments_do_not_override_triage": True,
            "queue_table": "comments_enrichment_queue",
            "observations_table": "comments_term_observations",
            "promoted_terms_table": "comment_promoted_terms",
            "queued_comment_candidates": UNINSTRUMENTED,
            "promoted_comment_terms": UNINSTRUMENTED,
            "files": ["src/flickr_bio_occurrence/pipeline/comments_enrichment.py"],
        }
    )


def build_comment_review_profile() -> dict[str, Any]:
    return with_required_metrics(
        {
            "report": "comment_review_profile",
            "review_scope": "targeted post-BioCLIP review only; comments are not fetched for every record",
            "default_max_comment_calls_per_hour": 300,
            "queue_table": "comment_review_queue",
            "results_table": "comment_review_results",
            "derived_terms_table": "comment_derived_terms",
            "missing_data_requests_table": "missing_data_requests",
            "eligible_reasons": [
                "species_conflict",
                "in_review_no_geo",
                "missing_event_date",
                "missing_geo",
                "unknown_image_category",
                "unknown_life_stage",
                "low_bioclip_score",
            ],
            "allowed_decisions": [
                "move_to_gold",
                "keep_silver",
                "keep_bronze",
                "keep_in_review_no_geo",
                "request_missing_geo",
                "request_missing_date",
                "request_species_review",
                "request_life_stage_review",
                "no_action",
            ],
            "gold_promotion_rules": [
                "BioCLIP Lepidoptera-positive",
                "image_url present",
                "event date present or normalized from comments",
                "geo present or structured coordinates recovered from comments",
                "comments support species/life-stage or resolve mismatch",
                "no hard-negative visual flag",
            ],
            "flickr_comments_are_separate_api_calls": True,
            "files": ["src/flickr_bio_occurrence/pipeline/comment_review.py"],
        }
    )


def build_missing_data_requests_profile() -> dict[str, Any]:
    return with_required_metrics(
        {
            "report": "missing_data_requests",
            "request_table": "missing_data_requests",
            "allowed_request_types": ["missing_geo", "missing_date", "ambiguous_species", "ambiguous_life_stage"],
            "free_text_location_policy": "Place names from comments create missing_geo requests unless structured geo is safely available.",
            "vague_date_policy": "Vague date text creates missing_date requests unless it can be normalized safely.",
            "pending_requests": UNINSTRUMENTED,
            "files": ["src/flickr_bio_occurrence/pipeline/comment_review.py"],
        }
    )


def build_comment_species_resolution_profile() -> dict[str, Any]:
    return with_required_metrics(
        {
            "report": "comment_species_resolution",
            "explicit_fields": [
                "flickr_text_species_candidate",
                "bioclip_species_candidate",
                "bioclip_tag_conflict",
                "comment_species_candidate",
                "comment_resolves_conflict",
            ],
            "comments_as_evidence_layer": True,
            "comments_do_not_replace_bioclip_output": True,
            "species_resolution_counts": UNINSTRUMENTED,
            "files": ["src/flickr_bio_occurrence/pipeline/comment_review.py"],
        }
    )


def build_api_budget_profile() -> dict[str, Any]:
    return with_required_metrics(
        {
            "report": "api_budget_profile",
            "soft_api_calls_per_hour": 3450,
            "hard_api_calls_per_hour": 3600,
            "api_call_ledger_table": "api_call_ledger",
            "rate_limit_file": "src/flickr_bio_occurrence/flickr/rate_limiter.py",
            "metadata_poller_file": "src/flickr_bio_occurrence/pipeline/metadata_poller.py",
            "api_calls_in_window": UNINSTRUMENTED,
            "remaining_soft_budget": UNINSTRUMENTED,
            "remaining_hard_budget": UNINSTRUMENTED,
        }
    )


def build_agents_update_recommendations() -> dict[str, Any]:
    return with_required_metrics(
        {
            "recommendations": [
                {
                    "topic": "reports",
                    "recommendation": "Keep reports centered on image triage, life stages, comments, API budget, no-geo bins, and outside-known-region discovery candidates.",
                },
                {
                    "topic": "range discovery",
                    "recommendation": "Preserve global high-confidence Papilio demoleus searches and flag geotagged hits outside configured known regions for researcher review, never automatic rejection.",
                },
                {
                    "topic": "comments",
                    "recommendation": "Keep term expansion and comment review as separate targeted phases; do not fetch comments for every Flickr search record.",
                },
                {
                    "topic": "comment review",
                    "recommendation": "Use comment review for BioCLIP/Flickr mismatches, no-geo rows, missing dates, low scores, and unknown category/life-stage records.",
                },
                {
                    "topic": "BioCLIP",
                    "recommendation": "Treat BioCLIP 2.5 labels and scores as screening evidence only; do not taxonomically validate records automatically.",
                },
                {
                    "topic": "Darwin Core",
                    "recommendation": "Keep Darwin Core mapper/exporter compatibility-only until existing tests and public API compatibility requirements are retired.",
                },
            ],
            "do_not_prioritize": [
                "validated Darwin Core occurrence publication",
                "identificationVerificationStatus expansion",
                "human-verification parsing as a gold/silver gate",
            ],
        }
    )


def build_query_term_totals_markdown(report: dict[str, Any]) -> str:
    return _markdown_report(
        "Query Term Totals",
        [
            ("Query source", report["query_source"]),
            ("Query term counts", report["query_term_counts"]),
            ("Comments promoted term counts", report["comments_promoted_term_counts"]),
            ("API calls used", report["api_calls_used"]),
        ],
    )


def build_bbox_coverage_profile_markdown(report: dict[str, Any]) -> str:
    return _markdown_report(
        "BBox Coverage Profile",
        [
            ("BBox queries supported", report["bbox_queries_supported"]),
            ("BBox query lane", report["bbox_query_lane"]),
            ("BBox counts", report["bbox_counts"]),
            ("Coverage notes", report["coverage_notes"]),
        ],
    )


def build_occurrence_bin_profile_markdown(report: dict[str, Any]) -> str:
    return _markdown_report(
        "Occurrence Bin Profile",
        [
            ("Gold count", report["gold_count"]),
            ("Silver count", report["silver_count"]),
            ("Bronze count", report["bronze_count"]),
            ("In review/no geo count", report["in_review_no_geo_count"]),
            ("Screening evidence only", report["screening_evidence_only"]),
            ("Darwin Core dependency", report["darwin_core_dependency"]),
        ],
    )


def build_comment_review_profile_markdown(report: dict[str, Any]) -> str:
    return _markdown_report(
        "Comment Review Profile",
        [
            ("Review scope", report["review_scope"]),
            ("Default max comment calls per hour", report["default_max_comment_calls_per_hour"]),
            ("Queue created", report["comment_review_queue_created"]),
            ("Comment calls used", report["comment_calls_used"]),
            ("Records moved to Gold", report["records_moved_to_gold"]),
            ("Missing geo requests", report["missing_geo_requests_created"]),
            ("Missing date requests", report["missing_date_requests_created"]),
        ],
    )


def build_code_cleanup_report_markdown() -> str:
    return "\n".join(
        [
            "# Code Cleanup Report",
            "",
            f"Generated: {GENERATED_AT}",
            "",
            "## Comment Review Scope",
            "",
            "- Added a separate targeted comment-review phase after BioCLIP triage.",
            "- Comment review creates its own queue, results table, derived terms, and missing-data request table.",
            "- Comment review is not part of initial Flickr `photos.search` and must not run for every record by default.",
            "- Retained BioCLIP as screening evidence only; no report claims taxonomic validation.",
            "- Required metrics are present in JSON reports as null or `not_instrumented` when no bounded run data is available.",
            "",
            "## Removed Or Superseded Report Paths",
            "",
            "- Superseded `bioclip_run_summary.*`, `quality_profile.json`, `image_triage_profile.json`, `cache_profile.json`, `gpu_profile.json`, and `idempotency_profile.json` in the active report pack.",
            "- Removed stale report text that described dedicated comments fetching as unavailable; comments enrichment now exists for selected candidates.",
            "",
            "## Compatibility-Only Code",
            "",
            "- `src/flickr_bio_occurrence/dwc/mapper.py` and `src/flickr_bio_occurrence/dwc/exporter.py` are retained only for existing tested public API compatibility.",
            "- Darwin Core compatibility code must not be used as the active image-triage or occurrence-publication path in this phase.",
            "- Removal condition: retire these shims when `tests/test_dwc_mapper.py` and any downstream public API expectations are removed or replaced.",
            "",
            "## Still Out Of Scope",
            "",
            "- Validated Darwin Core occurrence publication.",
            "- `identificationVerificationStatus` expansion.",
            "- Human verification as a gate for Gold/Silver.",
            "- Network/CUDA/BioCLIP/Flickr-dependent tests.",
        ]
    ) + "\n"


def _markdown_report(title: str, rows: list[tuple[str, object]]) -> str:
    lines = [
        f"# {title}",
        "",
        f"Generated: {GENERATED_AT}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    lines.extend(f"| {label} | `{value}` |" for label, value in rows)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
