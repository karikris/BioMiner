from __future__ import annotations

import argparse
import json

from biominer.gbif_quality.pipeline import (
    Phase1Config,
    Phase2Config,
    run_phase1_baseline,
    run_phase2_local_checks,
)


COMMAND = "gbif-media-quality"


def add_gbif_quality_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    command = subparsers.add_parser(
        COMMAND,
        help="audit and enrich the pinned GBIF media database",
    )
    stages = command.add_subparsers(dest="gbif_quality_command")
    baseline = stages.add_parser(
        "baseline", help="publish the reconciled Phase 1 quality baseline"
    )
    baseline.add_argument("--repository-root", default=".")
    baseline.add_argument(
        "--data-output", default="data/derived/gbif_media_database/v4"
    )
    baseline.add_argument(
        "--report-output", default="reports/gbif_media_database/v4"
    )
    baseline.add_argument("--temp-directory")
    baseline.add_argument("--memory-limit", default="4GB")
    baseline.add_argument("--occurrence-batch-size", type=int, default=8)
    local = stages.add_parser(
        "local-checks", help="run or resume the request-free Phase 2 checks"
    )
    local.add_argument("--repository-root", default=".")
    local.add_argument("--data-root", default="data/derived/gbif_media_database/v4")
    local.add_argument("--temp-directory")
    local.add_argument("--memory-limit", default="4GB")
    local.add_argument("--threads", type=int, default=4)
    local.add_argument("--batch-rows", type=int, default=100_000)


def run_gbif_quality_command(args: argparse.Namespace) -> int:
    if args.gbif_quality_command == "local-checks":
        result = run_phase2_local_checks(
            Phase2Config(
                repository_root=args.repository_root,
                data_root=args.data_root,
                temp_directory=args.temp_directory,
                memory_limit=args.memory_limit,
                threads=args.threads,
                batch_rows=args.batch_rows,
            )
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.gbif_quality_command != "baseline":
        return 2
    result = run_phase1_baseline(
        Phase1Config(
            repository_root=args.repository_root,
            data_output=args.data_output,
            report_output=args.report_output,
            temp_directory=args.temp_directory,
            memory_limit=args.memory_limit,
            occurrence_batch_size=args.occurrence_batch_size,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = ["COMMAND", "add_gbif_quality_parser", "run_gbif_quality_command"]
