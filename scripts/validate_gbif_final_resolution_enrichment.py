from __future__ import annotations

import argparse
import json
from pathlib import Path

from biominer.gbif_final.resolution_enrichment import (
    validate_resolution_enriched_publication,
)


def main() -> int:
    args = _parser().parse_args()
    manifest = validate_resolution_enriched_publication(
        args.output_directory,
        base_publication_directory=args.base_publication_directory,
        resolution_directory=args.resolution_directory,
        repository_root=args.repository_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independently revalidate a terminal resolver-integrated "
            "GBIF final publication."
        )
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--base-publication-directory",
        type=Path,
    )
    parser.add_argument(
        "--resolution-directory",
        type=Path,
    )
    parser.add_argument("--repository-root", type=Path)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
