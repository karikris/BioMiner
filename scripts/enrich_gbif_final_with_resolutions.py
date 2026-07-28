from __future__ import annotations

import argparse
import json
from pathlib import Path

from biominer.gbif_final.resolution_enrichment import (
    DEFAULT_EXPECTED_RESOLUTION_ROWS,
    enrich_final_with_resolutions,
)


def main() -> int:
    args = _parser().parse_args()
    manifest = enrich_final_with_resolutions(
        base_publication_directory=args.base_publication_directory,
        resolution_directory=args.resolution_directory,
        output_directory=args.output_directory,
        repository_root=args.repository_root,
        producer_git_sha=args.producer_git_sha,
        expected_resolution_rows=args.expected_resolution_rows,
        batch_rows=args.batch_rows,
        row_group_rows=args.row_group_rows,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Append terminal GBIF media URL-resolution evidence to the "
            "final enriched Parquet without dropping source rows."
        )
    )
    parser.add_argument(
        "--base-publication-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--resolution-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--producer-git-sha", required=True)
    parser.add_argument(
        "--expected-resolution-rows",
        type=int,
        default=DEFAULT_EXPECTED_RESOLUTION_ROWS,
    )
    parser.add_argument("--batch-rows", type=int, default=50_000)
    parser.add_argument("--row-group-rows", type=int, default=100_000)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
