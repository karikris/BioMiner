from __future__ import annotations

import argparse
import json
from pathlib import Path

from biominer.gbif_final.superseded_cleanup import (
    execute_superseded_cleanup,
    plan_superseded_cleanup,
    prepare_superseded_cleanup,
)


def main() -> int:
    args = _parser().parse_args()
    arguments = {
        "repository_root": args.repository_root,
        "publication_audit_directory": args.publication_audit_directory,
        "state_directory": args.state_directory,
    }
    if not args.execute:
        result = plan_superseded_cleanup(**arguments)
        result["execution"] = {
            "mode": "dry-run",
            "filesystem_objects_deleted": 0,
        }
    else:
        if not args.state_directory.exists():
            prepare_superseded_cleanup(**arguments)
        result = execute_superseded_cleanup(**arguments)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and remove only the exact superseded GBIF artifacts "
            "after the final publication audit passes. Defaults to dry-run."
        )
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--publication-audit-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--state-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "write immutable intent, delete the exact allowlist, and seal "
            "the completion manifest"
        ),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
