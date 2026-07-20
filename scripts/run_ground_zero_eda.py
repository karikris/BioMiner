"""Run the aggregate-only Ground Zero EDA report."""

from __future__ import annotations

import argparse
import json
from json import JSONDecodeError
from pathlib import Path
import sys

import duckdb

from biominer.reports.ground_zero_eda import build_ground_zero_eda_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Occurrence Parquet source file.")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Optional JSON manifest for the occurrence Parquet source.",
    )
    parser.add_argument("--output", required=True, type=Path, help="New report output directory.")
    parser.add_argument("--top-n", type=int, default=20, help="Maximum categories per frequency table.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.source_manifest is None:
            result = build_ground_zero_eda_run(args.source, args.output, top_n=args.top_n)
        else:
            result = build_ground_zero_eda_run(
                args.source,
                args.output,
                source_manifest_path=args.source_manifest,
                top_n=args.top_n,
            )
    except (FileExistsError, FileNotFoundError, ValueError, JSONDecodeError, duckdb.Error) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2

    payload = {
        "manifest": result["manifest"],
        "manifest_path": str(args.output / "manifest.json"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
