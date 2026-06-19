from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from biominer.cli import build_parser, run


def test_cli_exposes_only_lean_pipeline_commands() -> None:
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices  # noqa: SLF001 - parser surface regression test.

    assert "poll-once" in commands
    assert "build-papilio-demoleus-query-plan" in commands
    assert "fetch" not in commands
    assert "fetch-live" not in commands
    assert "benchmark-existing-payloads" not in commands

def test_poll_once_cli_accepts_bounded_cycle_arguments() -> None:
    parser = build_parser()
    args = parser.parse_args(["poll-once", "--max-api-calls", "3500"])

    assert args.command == "poll-once"
    assert args.max_api_calls == 3500


def test_build_papilio_demoleus_query_plan_cli_reads_keyword_json(tmp_path, capsys) -> None:
    keywords = tmp_path / "keywords.json"
    keywords.write_text(
        json.dumps(
            {
                "dictionary_groups": {
                    "scientific_taxonomic": [
                        {
                            "term": "Papilio demoleus",
                            "language": "la",
                            "term_type": "scientific_name",
                            "confidence": "high",
                            "use_for_flickr": True,
                            "precision_tier": "high",
                        }
                    ],
                    "multilingual_common_name_expansion": [
                        {
                            "term": "butterfly",
                            "language": "en",
                            "term_type": "broad_butterfly",
                            "confidence": "medium",
                            "use_for_flickr": True,
                            "precision_tier": "low",
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "build-papilio-demoleus-query-plan",
            "--keywords-json",
            str(keywords),
            "--state-db",
            str(tmp_path / "poller.sqlite"),
        ]
    )

    assert run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count_probes_seen"] == 4
    assert payload["count_probes_inserted"] == 4
    assert payload["soft_api_calls_per_hour"] == 3500
    assert payload["per_page_for_final_fetches"] == 250
    assert payload["per_page_for_non_geo_fetches"] == 500
    assert payload["flickr_search_result_window"] == 4000
    assert payload["stable_result_threshold"] == 4000


def test_registry_compile_fixture_cli_writes_registry_outputs(tmp_path, capsys) -> None:
    source = tmp_path / "registry_source.json"
    source.write_text(
        json.dumps(
            {
                "source": "fixture",
                "source_version": "2026-06-20",
                "retrieved_at": "2026-06-20T00:00:00+00:00",
                "taxa": [
                    {
                        "accepted_taxon_key": "gbif:1",
                        "scientific_name": "Papilionoidea",
                        "rank": "SUPERFAMILY",
                    }
                ],
                "names": [
                    {
                        "accepted_taxon_key": "gbif:1",
                        "verbatim_name": "Papilionoidea",
                        "display_name": "Papilionoidea",
                        "language": "la",
                        "script": "Latn",
                        "name_class": "accepted_scientific",
                        "source": "GBIF",
                        "source_record_id": "gbif:1",
                        "trust_tier": "T1",
                        "precision_tier": "high",
                        "confidence": "high",
                        "enabled": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "registry"
    parser = build_parser()
    args = parser.parse_args(
        [
            "registry",
            "compile-fixture",
            "--source-json",
            str(source),
            "--output-dir",
            str(output),
            "--registry-version",
            "test-registry",
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["registry_version"] == "test-registry"
    assert payload["query_definition_rows"] == 2
    assert (output / "manifest.json").exists()
    assert (output / "flickr_query_definitions.parquet").exists()


def test_cli_help_does_not_describe_old_gold_silver_bronze_logic(capsys) -> None:
    parser = build_parser()

    parser.print_help()
    help_text = capsys.readouterr().out

    assert "human_verified_bioclip_positive" not in help_text
    assert "human verification" not in help_text.casefold()
    assert "bioclip_positive_without_human_verification" not in help_text


def test_qa_rate_limit_outputs_limiter_status_json(tmp_path, capsys) -> None:
    state = tmp_path / "poller.sqlite"
    from biominer.flickr_fetch.metadata_poller import MetadataPollState

    poll_state = MetadataPollState(state)
    poll_state.log_api_call(work_item_id="work-1", endpoint="flickr.photos.search", status="ok")
    parser = build_parser()
    args = parser.parse_args(["qa-rate-limit", "--state-db", str(state)])

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["api_calls_in_window"] == 1
    assert payload["photo_records_in_window"] == "not_instrumented"
    assert payload["soft_api_calls_per_hour"] == 3500
    assert payload["hard_api_calls_per_hour"] == 3600


def test_qa_summary_outputs_report_summary(tmp_path, capsys) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "species": "Papilio demoleus",
                "actual_unique_records": 16,
                "api_calls_made": 0,
                "step_timings_seconds": {"vision_classification": 84.9},
                "storage_artifacts": {"total_artifact_bytes": 1234},
                "memory_artifacts": {"peak_traced_bytes": 4567},
                "compute_artifacts": {"vision_model_loaded": True},
            }
        ),
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(["qa-summary", "--report", str(report_path)])

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["species"] == "Papilio demoleus"
    assert payload["actual_unique_records"] == 16
    assert payload["vision_model_loaded"] is True
    assert payload["total_artifact_bytes"] == 1234


def test_apply_rules_compact_and_gc_cache_cli(tmp_path, capsys) -> None:
    parser = build_parser()
    evidence_path = tmp_path / "evidence.parquet"
    pl.DataFrame(
        {
            "flickr_photo_id": ["1"],
            "image_url": ["https://live.staticflickr.com/large.jpg"],
            "bioclip_top1_label": ["a photo of Papilio demoleus"],
            "bioclip_top1_score": [0.9],
            "bioclip_species_agreement_status": ["exact_species_agreement"],
        }
    ).write_parquet(evidence_path)

    classified_path = tmp_path / "classified.parquet"
    args = parser.parse_args(["apply-rules", "--evidence", str(evidence_path), "--output", str(classified_path)])
    assert run(args) == 0
    rules_payload = json.loads(capsys.readouterr().out)
    assert rules_payload["rows"] == 1
    assert sum(rules_payload["publication_state_counts"].values()) == 1
    assert rules_payload["in_review_without_reason"] == 0

    predictions = tmp_path / "predictions"
    predictions.mkdir()
    pl.DataFrame({"flickr_photo_id": ["1"]}).write_parquet(predictions / "part.parquet")
    compacted_path = tmp_path / "compacted.parquet"
    args = parser.parse_args(["compact-parquet", "--input-root", str(predictions), "--output", str(compacted_path)])
    assert run(args) == 0
    compact_payload = json.loads(capsys.readouterr().out)
    assert compact_payload["input_parquet_files"] == 1
    assert compact_payload["rows"] == 1
    assert compacted_path.exists()


def test_comments_enrichment_cli(tmp_path, capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(["fetch-comments", "--photo-id", "1", "--state-db", str(tmp_path / "comments.sqlite"), "--dry-run"])
    assert run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["implemented"] is True
    assert payload["comment_fetch_scope"] == "selected_candidate_records_only"
    assert payload["photo_ids_requested"] == ["1"]
    assert payload["queued_comment_candidates_added"] == 1


def test_gc_cache_reports_deleted_files(tmp_path, capsys) -> None:
    parser = build_parser()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "image.jpg").write_bytes(b"abc")
    args = parser.parse_args(["gc-cache", "--cache-root", str(cache_root), "--delete"])
    assert run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["files_seen"] == 1
    assert payload["deleted_files"] == 1


def test_export_bucket_views_cli_writes_derived_parquet_files(tmp_path, capsys) -> None:
    input_path = tmp_path / "bucketed_records.parquet"
    output_dir = tmp_path / "views"
    pl.DataFrame(
        [
            {"flickr_photo_id": "1", "occurrence_bin": "gold"},
            {"flickr_photo_id": "2", "occurrence_bin": "silver"},
            {"flickr_photo_id": "3", "occurrence_bin": "bronze"},
            {"flickr_photo_id": "4", "occurrence_bin": "bin"},
        ]
    ).write_parquet(input_path)

    assert run(build_parser().parse_args(["export-bucket-views", "--input", str(input_path), "--output-dir", str(output_dir)])) == 0
    payload = json.loads(capsys.readouterr().out)

    assert set(payload) == {"gold", "silver", "bronze", "bin"}
    assert (output_dir / "gold_records.parquet").exists()
    assert (output_dir / "silver_records.parquet").exists()
    assert (output_dir / "bronze_records.parquet").exists()
    assert (output_dir / "bin_records.parquet").exists()
