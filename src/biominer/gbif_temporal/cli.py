from __future__ import annotations

import argparse
import json

from biominer.gbif_temporal.pipeline import publish_temporal_enrichment


COMMAND = "gbif-temporal-enrich"


def add_gbif_temporal_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    command = subparsers.add_parser(
        COMMAND,
        help="derive audited temporal components from GBIF eventDate values",
    )
    command.add_argument("--source", required=True)
    command.add_argument("--source-manifest", required=True)
    command.add_argument("--output-directory", required=True)
    command.add_argument(
        "--expected-source-sha256",
        default="c96505f410723da57db4bd11bcffdc4e72be59ee59ecbaad8f4af8677229e57f",
    )
    command.add_argument("--expected-source-rows", type=int, default=16_612_063)
    command.add_argument("--expected-derived-year-rows", type=int, default=2_360)
    command.add_argument("--expected-derived-month-rows", type=int, default=4_941)
    command.add_argument("--expected-derived-day-rows", type=int, default=18_741)
    command.add_argument(
        "--expected-pre-1960-excluded-rows",
        type=int,
        default=2_236,
    )
    command.add_argument("--batch-rows", type=int, default=50_000)
    command.add_argument("--duckdb-memory-limit", default="24GB")
    command.add_argument("--duckdb-threads", type=int, default=8)


def run_gbif_temporal_command(args: argparse.Namespace) -> int:
    result = publish_temporal_enrichment(
        source=args.source,
        source_manifest=args.source_manifest,
        output_directory=args.output_directory,
        expected_source_sha256=args.expected_source_sha256,
        expected_source_rows=args.expected_source_rows,
        expected_derived_year_rows=args.expected_derived_year_rows,
        expected_derived_month_rows=args.expected_derived_month_rows,
        expected_derived_day_rows=args.expected_derived_day_rows,
        expected_pre_1960_excluded_rows=args.expected_pre_1960_excluded_rows,
        batch_rows=args.batch_rows,
        duckdb_memory_limit=args.duckdb_memory_limit,
        duckdb_threads=args.duckdb_threads,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = ["COMMAND", "add_gbif_temporal_parser", "run_gbif_temporal_command"]
