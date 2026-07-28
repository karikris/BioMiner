from __future__ import annotations

import argparse
import json
from pathlib import Path

from biominer.gbif_final.publication_audit import (
    audit_final_publication,
)


def main() -> int:
    args = _parser().parse_args()
    manifest = audit_final_publication(
        publication_directory=args.publication_directory,
        temporal_parquet=args.temporal_parquet,
        pre_temporal_parquet=args.pre_temporal_parquet,
        registry_directory=args.registry_directory,
        source_assertions=args.source_assertions,
        quality_directory=args.quality_directory,
        output_directory=args.output_directory,
        repository_root=args.repository_root,
        expected_producer_git_sha=args.expected_producer_git_sha,
        memory_limit=args.memory_limit,
        threads=args.threads,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independently checksum-bind and validate a completed legacy "
            "or bounded GBIF final enriched publication."
        )
    )
    parser.add_argument(
        "--publication-directory",
        type=Path,
        required=True,
    )
    parser.add_argument("--temporal-parquet", type=Path, required=True)
    parser.add_argument("--pre-temporal-parquet", type=Path, required=True)
    parser.add_argument(
        "--registry-directory",
        type=Path,
        required=True,
    )
    parser.add_argument("--source-assertions", type=Path)
    parser.add_argument(
        "--quality-directory",
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
        default=Path("."),
    )
    parser.add_argument(
        "--expected-producer-git-sha",
        required=True,
    )
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=4)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
