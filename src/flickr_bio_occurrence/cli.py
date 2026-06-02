from __future__ import annotations

import argparse
import json

from flickr_bio_occurrence.pipeline.dry_run import build_dry_run_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flickr-bio")
    parser.add_argument("--version", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("--species", required=True)
    fetch.add_argument("--region", required=True)
    fetch.add_argument("--year", type=int, required=True)
    fetch.add_argument("--month", type=int, required=True)
    fetch.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.version:
        print("flickr-bio-occurrence 0.1.0")
        return 0
    if args.command == "fetch" and args.dry_run:
        summary = build_dry_run_summary(
            species=args.species,
            region=args.region,
            year=args.year,
            month=args.month,
            config_path="config/pipeline.toml",
            model_registry_path="config/model_registry.toml",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    return 2


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
