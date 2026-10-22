#!/usr/bin/env python3
"""Collect image-bearing butterfly species, then resume source-bound name enrichment."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biominer.registry.image_keyword_enrichment import SOURCE_ORDER, _enrich_locked, run_lock
from biominer.registry.image_species import collect_image_species


def load_selected_credentials(path: Path | None) -> None:
    """Read selected dotenv values without evaluating shell syntax or logging secrets."""
    if path is None:
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator or key.strip() not in {"MYMEMORY_API_KEY", "MYMEMORY_EMAIL"}:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key.strip()] = value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--local", action="store_true", required=True,
                        help="Explicit local filesystem/SQLite job; does not change production storage defaults")
    parser.add_argument("--database", type=Path, default=ROOT / "data/derived/gbif_media_database/v3/gbif_media.duckdb")
    parser.add_argument("--database-manifest", type=Path, default=ROOT / "data/derived/gbif_media_database/v3/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--secrets-file", type=Path)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--sources", nargs="+", choices=SOURCE_ORDER, default=SOURCE_ORDER)
    parser.add_argument("--max-species", type=int, default=0, help="0 means the complete collected cohort")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--max-runtime-hours", type=float, default=23.5)
    parser.add_argument("--max-http-requests", type=int, default=250000)
    parser.add_argument("--request-interval", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="2GB")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    log = logging.getLogger("biominer.image_keywords")
    output = args.output.resolve()

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    load_selected_credentials(args.secrets_file)
    try:
        with run_lock(output):
            collection_log = logging.FileHandler(output / "collection.log", encoding="utf-8")
            collection_log.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            collector = logging.getLogger("biominer.image_collection")
            collector.addHandler(collection_log)
            try:
                result = collect_image_species(args.database, args.database_manifest, output / "collection",
                    threads=args.threads, memory_limit=args.memory_limit,
                    logger=lambda event: collector.info("%s", json.dumps(event, sort_keys=True)))
            finally:
                collector.removeHandler(collection_log); collection_log.close()
            log.info("collection ready | %d species | %d sources | %d query-eligible species",
                result["species_count"], len(args.sources), result["query_eligible_species_count"])
            if args.collect_only:
                return 0
            result = _enrich_locked(output / "collection", output, sources=tuple(args.sources),
                checkpoint_every=args.checkpoint_every, max_species=args.max_species,
                max_seconds=args.max_runtime_hours * 3600, max_requests=args.max_http_requests,
                min_interval=args.request_interval, max_retries=args.max_retries)
            log.info("run ended | %s", json.dumps(result, sort_keys=True))
            return 0
    except KeyboardInterrupt:
        log.info("interrupted; committed species/source results remain resumable")
        return 130
    except Exception as exc:
        # Provider URLs and credentials must not reach exception tracebacks in logs.
        log.error("run failed | error_class=%s", type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
