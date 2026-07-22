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
