from __future__ import annotations

from biominer.cli import build_parser
from biominer.gbif_quality.cli import COMMAND


def test_quality_baseline_cli_has_bounded_defaults() -> None:
    args = build_parser().parse_args([COMMAND, "baseline"])

    assert args.gbif_quality_command == "baseline"
    assert args.repository_root == "."
    assert args.data_output == "data/derived/gbif_media_database/v4"
    assert args.report_output == "reports/gbif_media_database/v4"
    assert args.memory_limit == "4GB"
    assert args.occurrence_batch_size == 8
    assert args.temp_directory is None


def test_quality_local_checks_cli_is_resumable_and_network_free() -> None:
    args = build_parser().parse_args([COMMAND, "local-checks"])

    assert args.gbif_quality_command == "local-checks"
    assert args.data_root == "data/derived/gbif_media_database/v4"
    assert args.memory_limit == "4GB"
    assert args.threads == 4
    assert args.batch_rows == 100_000


def test_quality_enrichment_cli_is_resumable_and_network_free() -> None:
    args = build_parser().parse_args([COMMAND, "enrich"])

    assert args.gbif_quality_command == "enrich"
    assert args.data_root == "data/derived/gbif_media_database/v4"
    assert args.memory_limit == "4GB"
    assert args.threads == 4
    assert args.batch_rows == 50_000


def test_quality_offline_publication_commands_have_bounded_defaults() -> None:
    for command in (
        "rights",
        "duplicates",
        "ai-readiness",
        "representativeness",
        "concentration",
        "media-resources",
        "gates",
        "review-capsules",
        "incremental",
        "freshness",
    ):
        args = build_parser().parse_args([COMMAND, command])

        assert args.gbif_quality_command == command
        assert args.data_root == "data/derived/gbif_media_database/v4"
        assert args.expected_rows == 16_612_063
        assert args.memory_limit == "6GB"
        assert args.threads == 4
        assert args.v3 is None
        assert args.code_commit is None


def test_quality_acceptance_defaults_pin_v3_checksum() -> None:
    args = build_parser().parse_args([COMMAND, "acceptance"])

    assert args.report_root == "reports/gbif_media_database/v4"
    assert args.output_directory.endswith("quality_results/global_acceptance")
    assert args.test_receipt.endswith("quality_results/test_receipt.json")
    assert args.expected_v3_sha256 == (
        "c96505f410723da57db4bd11bcffdc4e72be59ee59ecbaad8f4af8677229e57f"
    )


def test_quality_incremental_accepts_previous_state_for_diffing() -> None:
    args = build_parser().parse_args(
        [COMMAND, "incremental", "--previous-state-glob", "previous/**/*.parquet"]
    )

    assert args.previous_state_glob == "previous/**/*.parquet"
    assert args.partitions == 16


def test_quality_source_lineage_defaults_cover_raw_multimedia_rows() -> None:
    args = build_parser().parse_args([COMMAND, "source-lineage"])

    assert args.expected_rows == 18_680_565
    assert args.partition_rows == 1_000_000
    assert args.output_directory.endswith("source_lineage/identity_v2")
    assert args.multimedia_parquet is None


def test_quality_freshness_has_explicit_ttls() -> None:
    args = build_parser().parse_args([COMMAND, "freshness"])

    assert args.provider_stale_days == 365
    assert args.derived_stale_days == 30


def test_quality_provider_registry_is_offline_by_construction() -> None:
    args = build_parser().parse_args([COMMAND, "provider-registry"])

    assert args.output_directory.endswith("provider_enrichment")
    assert not hasattr(args, "execute_network")
