from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import json
import logging
from pathlib import Path
import time

from biominer.registry.enrichment import (
    DEFAULT_ENRICHMENT_SOURCES,
    build_enrichment_sources_from_registry,
    compile_enriched_registry,
)
from biominer.registry.translation_harvester import build_translation_candidates_from_registry


LOGGER = logging.getLogger("biominer.gbif_flickr_keyword_recovery")


def _seconds_until_next_utc_day(now: datetime | None = None) -> float:
    current = now or datetime.now(UTC)
    next_day = (current + timedelta(days=1)).date()
    resume_at = datetime.combine(next_day, datetime.min.time(), tzinfo=UTC) + timedelta(minutes=2)
    return max(60.0, (resume_at - current).total_seconds())


def run(args: argparse.Namespace) -> dict[str, object]:
    base_registry = Path(args.base_registry_dir)
    enrichment_dir = Path(args.enrichment_dir)
    output_dir = Path(args.output_dir)
    enrichment_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("recovery.wikidata.start species_registry=%s output=%s", base_registry, enrichment_dir)
    wikidata_manifest = build_enrichment_sources_from_registry(
        registry_dir=base_registry,
        enrichment_dir=enrichment_dir,
        sources=("wikidata",),
        workers=args.workers,
        progress_every=args.progress_every,
        checkpoint_every=args.checkpoint_every,
        max_retries=args.max_retries,
        report_dir=args.report_dir,
    )
    LOGGER.info(
        "recovery.wikidata.complete status=%s enriched_species=%s assertions=%s links=%s",
        wikidata_manifest.get("status"),
        (wikidata_manifest.get("coverage") or {}).get("enriched_species"),
        wikidata_manifest.get("name_assertion_rows"),
        wikidata_manifest.get("external_taxon_link_rows"),
    )

    translation_manifest: dict[str, object] = {}
    while True:
        LOGGER.info("recovery.wikimedia.start enrichment=%s", enrichment_dir)
        translation_manifest = build_translation_candidates_from_registry(
            registry_dir=base_registry,
            enrichment_dir=enrichment_dir,
            translation_sources=("wikimedia",),
            target_locales_json=args.target_locales_json,
            max_retries=args.max_retries,
            daily_request_limit=args.translation_daily_request_limit,
            translation_workers=1,
            translation_checkpoint_every=args.translation_checkpoint_every,
            translation_checkpoint_seconds=args.translation_checkpoint_seconds,
        )
        status = str(translation_manifest.get("translation_status") or "")
        LOGGER.info(
            "recovery.wikimedia.checkpoint status=%s assertions=%s requests=%s",
            status,
            translation_manifest.get("wikimedia_assertion_rows"),
            translation_manifest.get("translation_request_rows"),
        )
        if status in {"complete", "complete_with_errors"}:
            break
        if status != "budget_exhausted":
            raise RuntimeError(f"unexpected Wikimedia translation status: {status}")
        delay = _seconds_until_next_utc_day()
        LOGGER.info("recovery.wikimedia.daily_budget_wait seconds=%.0f", delay)
        time.sleep(delay)

    LOGGER.info("recovery.compile.start output=%s", output_dir)
    canonical_manifest = compile_enriched_registry(
        base_registry_dir=base_registry,
        enrichment_dir=enrichment_dir,
        output_dir=output_dir,
        registry_version=args.registry_version,
        requested_sources=DEFAULT_ENRICHMENT_SOURCES,
        requested_translation_sources=("wikimedia",),
    )
    if canonical_manifest.get("qa_status") != "passed":
        raise RuntimeError(f"recovery registry failed QA: {canonical_manifest.get('qa_status')}")

    run_manifest = {
        "schema_version": "biominer-gbif-flickr-keyword-recovery/v1",
        "completed_at": datetime.now(UTC).isoformat(),
        "registry_version": args.registry_version,
        "base_registry_dir": str(base_registry),
        "enrichment_dir": str(enrichment_dir),
        "output_dir": str(output_dir),
        "wikidata_status": wikidata_manifest.get("status"),
        "wikimedia_status": translation_manifest.get("translation_status"),
        "qa_status": canonical_manifest.get("qa_status"),
        "query_definition_rows": canonical_manifest.get("query_definition_rows"),
        "name_rows": canonical_manifest.get("name_rows"),
        "translation_sources": ["wikimedia"],
        "machine_translation_enabled": False,
    }
    manifest_path = output_dir / "recovery_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("recovery.complete manifest=%s", manifest_path)
    return run_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-registry-dir", required=True)
    parser.add_argument("--enrichment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--registry-version", required=True)
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--target-locales-json", default="config/name_translation_target_locales.json")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--translation-daily-request-limit", type=int, default=10000)
    parser.add_argument("--translation-checkpoint-every", type=int, default=100)
    parser.add_argument("--translation-checkpoint-seconds", type=float, default=60.0)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
