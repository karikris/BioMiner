from __future__ import annotations

import argparse
import json
from pathlib import Path

from biominer.gbif_final.publication_audit import (
    validate_publication_audit,
)


def main() -> int:
    args = _parser().parse_args()
    manifest = validate_publication_audit(
        args.audit_directory,
        repository_root=args.repository_root,
        require_dependencies=not args.allow_cleaned_dependencies,
        primary_publication_directory=(
            args.primary_publication_directory
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independently revalidate a sealed GBIF final-publication audit, "
            "including a publication copied to a new filesystem location."
        )
    )
    parser.add_argument(
        "--audit-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--primary-publication-directory",
        type=Path,
        help=(
            "relocated directory containing the audited final Parquet and "
            "primary manifest"
        ),
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--allow-cleaned-dependencies",
        action="store_true",
        help=(
            "validate the final and sealed audit after checksum-recorded "
            "upstream intermediates were intentionally cleaned"
        ),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
