from __future__ import annotations

import argparse
import json
from pathlib import Path

from biominer.gbif_final.locator_index import (
    build_final_locator_index,
)


def main() -> int:
    args = _parser().parse_args()
    manifest = build_final_locator_index(
        publication_directory=args.publication_directory,
        publication_audit_directory=args.publication_audit_directory,
        output_directory=args.output_directory,
        repository_root=args.repository_root,
        memory_limit=args.memory_limit,
        threads=args.threads,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a slim indexed URL/GBIF/species locator for the validated "
            "final enriched Parquet."
        )
    )
    parser.add_argument(
        "--publication-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--publication-audit-directory",
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
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=4)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
