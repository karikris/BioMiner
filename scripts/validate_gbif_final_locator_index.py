from __future__ import annotations

import argparse
import json
from pathlib import Path

from biominer.gbif_final.locator_index import (
    validate_final_locator_index,
)


def main() -> int:
    args = _parser().parse_args()
    manifest = validate_final_locator_index(
        index_directory=args.index_directory,
        publication_audit_directory=args.publication_audit_directory,
        repository_root=args.repository_root,
        publication_directory=args.publication_directory,
        require_dependencies=not args.allow_cleaned_dependencies,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Revalidate a local or transferred GBIF final locator index."
        )
    )
    parser.add_argument(
        "--index-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--publication-audit-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--publication-directory",
        type=Path,
        help="relocated final publication directory",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--allow-cleaned-dependencies",
        action="store_true",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
