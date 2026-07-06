from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import threading
from time import monotonic
from typing import Any

import polars as pl

from biominer.registry.compiler import compile_registry_fixture
from biominer.registry.normalize import normalize_language_code, normalize_name_key
from biominer.registry.translation_sources import DEFAULT_TRANSLATION_SOURCES, TRANSLATION_CANDIDATES_FILE, translation_candidate_schema, translation_source_display_names
from biominer.registry.trust_policy import decide_name_trust


ENRICHMENT_SCHEMA_VERSION = "registry-enrichment-v1"
SOURCE_ASSERTIONS_FILE = "source_name_assertions.parquet"
EXTERNAL_LINKS_FILE = "external_taxon_links.parquet"
ENRICHMENT_SOURCE_SNAPSHOTS_FILE = "enrichment_source_snapshots.parquet"
SOURCE_ERRORS_FILE = "source_error_records.parquet"
SOURCE_WORK_LEDGER_FILE = "source_work_ledger.parquet"
FINAL_SOURCE_SNAPSHOTS_FILE = "source_snapshots.parquet"
NAME_CANDIDATES_FILE = "name_candidates.parquet"
ENRICHMENT_MANIFEST_FILE = "enrichment_manifest.json"
TRANSLATION_WORK_LEDGER_FILE = "translation_work_ledger.parquet"
DEFAULT_ENRICHMENT_SOURCES = ("col", "inaturalist", "itis", "tmd_de", "wikidata")
BULK_ENRICHMENT_SOURCES = frozenset({"tmd_de"})
BULK_REGISTRY_WORK_KEY = "__registry__"
INATURALIST_DAILY_REQUEST_LIMIT = 10000
INATURALIST_WORKER_LIMIT = 1
INATURALIST_REQUESTS_PER_SPECIES = 1
WIKIDATA_WORKER_LIMIT = 1


logger = logging.getLogger(__name__)
ClientBundleFactory = Callable[[], dict[str, Any]]
_worker_local = threading.local()
_source_semaphores = {
    "inaturalist": threading.BoundedSemaphore(INATURALIST_WORKER_LIMIT),
    "wikidata": threading.BoundedSemaphore(WIKIDATA_WORKER_LIMIT),
}
SOURCE_WORKER_LIMITS = {
    "inaturalist": INATURALIST_WORKER_LIMIT,
    "tmd_de": 1,
    "wikidata": WIKIDATA_WORKER_LIMIT,
}


@dataclass(frozen=True)
class SpeciesContext:
    accepted_taxon_key: str
    accepted_scientific_name: str
    family_key: str
    family: str
    genus_key: str
    genus: str
    current_names: tuple[str, ...]


@dataclass(frozen=True)
class SpeciesEnrichmentResult:
    accepted_taxon_key: str
    name_assertions: tuple[dict[str, Any], ...]
    external_links: tuple[dict[str, Any], ...]
    source_snapshots: tuple[dict[str, Any], ...]
    errors: tuple[dict[str, str], ...]
    work_records: tuple[dict[str, Any], ...]


class DailyRequestBudget:
    def __init__(self, *, source: str, daily_limit: int, existing_rows: list[dict[str, Any]], day: str | None = None) -> None:
        self.source = source
        self.daily_limit = daily_limit
        self.day = day or datetime.now(UTC).date().isoformat()
        self._lock = threading.Lock()
        self.used = sum(
            int(row.get("request_count") or 0)
            for row in existing_rows
            if row.get("source") == source and row.get("request_day") == self.day
        )
        self.exhausted = False

    def reserve(self, count: int = 1) -> bool:
        if self.daily_limit <= 0:
            return True
        with self._lock:
            if self.used + count > self.daily_limit:
                self.exhausted = True
                return False
            self.used += count
            return True


def build_enrichment_sources_from_registry(
    *,
    registry_dir: str | Path,
    enrichment_dir: str | Path | None = None,
    sources: tuple[str, ...] = DEFAULT_ENRICHMENT_SOURCES,
    clients: dict[str, Any] | None = None,
    client_factory: ClientBundleFactory | None = None,
    workers: int = 8,
    progress_every: int = 100,
    checkpoint_every: int = 500,
    max_retries: int = 5,
    limit: int = 0,
    inaturalist_daily_request_limit: int = INATURALIST_DAILY_REQUEST_LIMIT,
    report_dir: str | Path = "reports",
) -> dict[str, Any]:
    _validate_runtime_options(workers=workers, progress_every=progress_every, checkpoint_every=checkpoint_every, max_retries=max_retries, limit=limit)
    started = monotonic()
    registry = Path(registry_dir)
    output = Path(enrichment_dir) if enrichment_dir is not None else registry
    taxa = pl.read_parquet(registry / "taxa.parquet")
    names = pl.read_parquet(registry / "names.parquet")
    species_rows = taxa.filter(pl.col("rank") == "SPECIES").sort(["family", "genus", "scientific_name"]).to_dicts()
    if limit:
        species_rows = species_rows[:limit]
    names_by_taxon: dict[str, list[str]] = {}
    for row in names.sort(["accepted_taxon_key", "display_name"]).to_dicts():
        names_by_taxon.setdefault(str(row.get("accepted_taxon_key") or ""), []).append(str(row.get("display_name") or ""))

    source_order = tuple(sources)
    allowed_source_names = set(_source_display_names(source_order))
    allowed_error_sources = set((*source_order, *allowed_source_names))
    name_assertions: list[dict[str, Any]] = [
        row for row in _read_or_empty(output / SOURCE_ASSERTIONS_FILE, _name_assertion_schema()).to_dicts()
        if str(row.get("source") or "") in allowed_source_names
    ]
    external_links: list[dict[str, Any]] = [
        row for row in _read_or_empty(output / EXTERNAL_LINKS_FILE, _external_link_schema()).to_dicts()
        if str(row.get("source") or "") in allowed_source_names
    ]
    source_snapshots: list[dict[str, Any]] = [
        row for row in _read_or_empty(output / ENRICHMENT_SOURCE_SNAPSHOTS_FILE, _source_snapshot_schema()).to_dicts()
        if str(row.get("source") or "") in allowed_source_names
    ]
    errors: list[dict[str, str]] = [
        row for row in _read_or_empty(output / SOURCE_ERRORS_FILE, _source_error_schema()).to_dicts()
        if str(row.get("source") or "") in allowed_error_sources
    ]
    work_ledger: list[dict[str, Any]] = [
        row for row in _read_or_empty(output / SOURCE_WORK_LEDGER_FILE, _source_work_schema()).to_dicts()
        if str(row.get("source") or "") in source_order
    ]
    completed_work = {
        (str(row.get("source") or ""), str(row.get("accepted_taxon_key") or ""))
        for row in work_ledger
        if str(row.get("status") or "") == "complete"
    }
    budgets = {
        "inaturalist": DailyRequestBudget(
            source="inaturalist",
            daily_limit=inaturalist_daily_request_limit,
            existing_rows=work_ledger,
        )
    }
    contexts = [_species_context(row, names_by_taxon) for row in species_rows]
    registry_version = _registry_version(registry)
    logger.info(
        "registry.enrichment.start registry=%s species=%d sources=%s workers=%d source_worker_limits=%s progress_every=%d checkpoint_every=%d max_retries=%d limit=%d",
        registry,
        len(contexts),
        ",".join(source_order),
        workers,
        _source_worker_limits(source_order, workers),
        progress_every,
        checkpoint_every,
        max_retries,
        limit,
    )

    bulk_coverage = _run_bulk_enrichment_sources(
        source_order=source_order,
        clients=clients,
        client_factory=client_factory,
        max_retries=max_retries,
        completed_work=completed_work,
        taxa_rows=taxa.to_dicts(),
        name_rows=names.to_dicts(),
        name_assertions=name_assertions,
        external_links=external_links,
        source_snapshots=source_snapshots,
        errors=errors,
        work_ledger=work_ledger,
    )
    per_species_sources = tuple(source for source in source_order if source not in BULK_ENRICHMENT_SOURCES)
    completed = 0
    for result in _enrichment_iterator(
        contexts=contexts,
        sources=per_species_sources,
        clients=clients,
        client_factory=client_factory,
        workers=workers,
        max_retries=max_retries,
        completed_work=completed_work,
        budgets=budgets,
    ):
        completed += 1
        name_assertions.extend(result.name_assertions)
        external_links.extend(result.external_links)
        source_snapshots.extend(result.source_snapshots)
        errors.extend(result.errors)
        work_ledger.extend(result.work_records)
        if completed % progress_every == 0 or completed == len(contexts):
            logger.info(
                "registry.enrichment.progress completed=%d/%d name_assertion_rows=%d external_taxon_link_rows=%d errors=%d elapsed_seconds=%.1f",
                completed,
                len(contexts),
                len(name_assertions),
                len(external_links),
                len(errors),
                monotonic() - started,
            )
        if completed % checkpoint_every == 0 or completed == len(contexts):
            _write_enrichment_checkpoint(
                output,
                name_assertions=name_assertions,
                external_links=external_links,
                source_snapshots=source_snapshots,
                errors=errors,
                work_ledger=work_ledger,
                completed=completed,
                total=len(contexts),
                source_order=source_order,
                workers=workers,
                source_worker_limits=_source_worker_limits(source_order, workers),
                progress_every=progress_every,
                checkpoint_every=checkpoint_every,
                max_retries=max_retries,
                limit=limit,
                started=started,
                base_registry_dir=registry,
                status=_run_status(completed=completed, total=len(contexts), budgets=budgets),
                bulk_coverage=bulk_coverage,
            )

    if not contexts or not per_species_sources:
        completed_without_species_work = 0 if not contexts else len(contexts)
        _write_enrichment_checkpoint(
            output,
            name_assertions=name_assertions,
            external_links=external_links,
            source_snapshots=source_snapshots,
            errors=errors,
            work_ledger=work_ledger,
            completed=completed_without_species_work,
            total=len(contexts),
            source_order=source_order,
            workers=workers,
            source_worker_limits=_source_worker_limits(source_order, workers),
            progress_every=progress_every,
            checkpoint_every=checkpoint_every,
            max_retries=max_retries,
            limit=limit,
            started=started,
            base_registry_dir=registry,
            status="complete",
            bulk_coverage=bulk_coverage,
        )

    manifest = json.loads((output / ENRICHMENT_MANIFEST_FILE).read_text(encoding="utf-8"))
    report_paths = _write_enrichment_reports(
        report=_enrichment_report(
            registry=registry,
            registry_version=registry_version,
            manifest=manifest,
            errors=errors,
            source_order=source_order,
            workers=workers,
            source_worker_limits=_source_worker_limits(source_order, workers),
            progress_every=progress_every,
            checkpoint_every=checkpoint_every,
            max_retries=max_retries,
            limit=limit,
            elapsed_seconds=monotonic() - started,
        ),
        report_dir=Path(report_dir),
        registry_version=registry_version,
    )
    manifest.update(report_paths)
    (output / ENRICHMENT_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("registry.enrichment.complete registry=%s enrichment_dir=%s completed=%d status=%s", registry, output, completed, manifest.get("status"))
    return manifest


def default_enrichment_clients(*, max_retries: int = 5) -> dict[str, Any]:
    from biominer.registry.enrichment_sources import CatalogueOfLifeClient, INaturalistClient, ITISClient, TMDGermanClient, WikidataClient

    return {
        "col": CatalogueOfLifeClient(max_retries=max_retries),
        "inaturalist": INaturalistClient(max_retries=max_retries),
        "itis": ITISClient(max_retries=max_retries),
        "tmd_de": TMDGermanClient(max_retries=max_retries),
        "wikidata": WikidataClient(max_retries=max_retries),
    }


def write_enrichment_sources(
    output_dir: str | Path,
    *,
    name_assertions: list[dict[str, Any]] | None = None,
    external_links: list[dict[str, Any]] | None = None,
    source_snapshots: list[dict[str, Any]] | None = None,
    source_errors: list[dict[str, Any]] | None = None,
    source_work: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    assertions = _name_assertions_frame(name_assertions or [])
    links = _external_links_frame(external_links or [])
    snapshots = _source_snapshots_frame(source_snapshots or [])
    errors = _source_errors_frame(source_errors or [])
    work = _source_work_frame(source_work or [])
    assertions.write_parquet(output / SOURCE_ASSERTIONS_FILE)
    links.write_parquet(output / EXTERNAL_LINKS_FILE)
    snapshots.write_parquet(output / ENRICHMENT_SOURCE_SNAPSHOTS_FILE)
    errors.write_parquet(output / SOURCE_ERRORS_FILE)
    work.write_parquet(output / SOURCE_WORK_LEDGER_FILE)
    manifest = {
        "schema_version": ENRICHMENT_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "name_assertion_rows": assertions.height,
        "external_taxon_link_rows": links.height,
        "source_snapshot_rows": snapshots.height,
        "source_error_rows": errors.height,
        "source_work_rows": work.height,
        "files": {
            "source_name_assertions": SOURCE_ASSERTIONS_FILE,
            "external_taxon_links": EXTERNAL_LINKS_FILE,
            "enrichment_source_snapshots": ENRICHMENT_SOURCE_SNAPSHOTS_FILE,
            "source_error_records": SOURCE_ERRORS_FILE,
            "source_work_ledger": SOURCE_WORK_LEDGER_FILE,
        },
    }
    (output / ENRICHMENT_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _species_context(species: dict[str, Any], names_by_taxon: dict[str, list[str]]) -> SpeciesContext:
    key = str(species.get("accepted_taxon_key") or "")
    return SpeciesContext(
        accepted_taxon_key=key,
        accepted_scientific_name=str(species.get("scientific_name") or ""),
        family_key=str(species.get("family_key") or ""),
        family=str(species.get("family") or ""),
        genus_key=str(species.get("genus_key") or ""),
        genus=str(species.get("genus") or ""),
        current_names=tuple(names_by_taxon.get(key, [])),
    )


def _enrichment_iterator(
    *,
    contexts: list[SpeciesContext],
    sources: tuple[str, ...],
    clients: dict[str, Any] | None,
    client_factory: ClientBundleFactory | None,
    workers: int,
    max_retries: int,
    completed_work: set[tuple[str, str]],
    budgets: dict[str, DailyRequestBudget],
) -> Iterator[SpeciesEnrichmentResult]:
    if not contexts:
        return iter(())
    if workers == 1:
        bundle = clients or (client_factory() if client_factory else default_enrichment_clients(max_retries=max_retries))
        return (
            _enrich_species_context(
                context,
                sources=sources,
                clients=bundle,
                completed_work=completed_work,
                budgets=budgets,
            )
            for context in contexts
        )

    def initializer() -> None:
        _worker_local.clients = clients or (client_factory() if client_factory else default_enrichment_clients(max_retries=max_retries))

    def task(context: SpeciesContext) -> SpeciesEnrichmentResult:
        bundle = getattr(_worker_local, "clients", None)
        if bundle is None:
            raise RuntimeError("Enrichment worker clients were not initialized")
        return _enrich_species_context(
            context,
            sources=sources,
            clients=bundle,
            completed_work=completed_work,
            budgets=budgets,
        )

    def generator() -> Iterator[SpeciesEnrichmentResult]:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="registry-enrich", initializer=initializer) as executor:
            yield from executor.map(task, contexts, buffersize=workers * 4)

    return generator()


def _enrich_species_context(
    context: SpeciesContext,
    *,
    sources: tuple[str, ...],
    clients: dict[str, Any],
    completed_work: set[tuple[str, str]],
    budgets: dict[str, DailyRequestBudget],
) -> SpeciesEnrichmentResult:
    name_assertions: list[dict[str, Any]] = []
    external_links: list[dict[str, Any]] = []
    source_snapshots: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    work_records: list[dict[str, Any]] = []
    for source in sources:
        if (source, context.accepted_taxon_key) in completed_work:
            continue
        client = clients.get(source)
        if client is None:
            errors.append({"source": source, "accepted_taxon_key": context.accepted_taxon_key, "error": "missing_client"})
            work_records.append(_source_work_record(source=source, context=context, status="error", error_class="missing_client", request_count=0))
            continue
        request_count = _request_count_for_source(source)
        budget = budgets.get(source)
        if budget is not None and not budget.reserve(request_count):
            work_records.append(_source_work_record(source=source, context=context, status="budget_exhausted", request_count=0))
            continue
        try:
            with _source_query_limit(source):
                result = client.enrich_species(context)
        except Exception as exc:  # noqa: BLE001 - source staging records and continues per species.
            errors.append(
                {
                    "source": source,
                    "accepted_taxon_key": context.accepted_taxon_key,
                    "error": type(exc).__name__,
                    "error_class": type(exc).__name__,
                    "endpoint": "",
                    "attempts": "1",
                    "retryable": "false",
                    "disposition": "quarantined",
                }
            )
            logger.info(
                "registry.enrichment.source_error source=%s accepted_taxon_key=%s error=%s",
                source,
                context.accepted_taxon_key,
                type(exc).__name__,
            )
            work_records.append(
                _source_work_record(
                    source=source,
                    context=context,
                    status="error",
                    error_class=type(exc).__name__,
                    request_count=request_count,
                )
            )
            continue
        name_assertions.extend(result.get("name_assertions", []))
        external_links.extend(result.get("external_links", []))
        source_snapshots.extend(result.get("source_snapshots", []))
        work_records.append(_source_work_record(source=source, context=context, status="complete", request_count=request_count))
    return SpeciesEnrichmentResult(
        accepted_taxon_key=context.accepted_taxon_key,
        name_assertions=tuple(name_assertions),
        external_links=tuple(external_links),
        source_snapshots=tuple(source_snapshots),
        errors=tuple(errors),
        work_records=tuple(work_records),
    )


def _run_bulk_enrichment_sources(
    *,
    source_order: tuple[str, ...],
    clients: dict[str, Any] | None,
    client_factory: ClientBundleFactory | None,
    max_retries: int,
    completed_work: set[tuple[str, str]],
    taxa_rows: list[dict[str, Any]],
    name_rows: list[dict[str, Any]],
    name_assertions: list[dict[str, Any]],
    external_links: list[dict[str, Any]],
    source_snapshots: list[dict[str, Any]],
    errors: list[dict[str, str]],
    work_ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    bulk_coverage: dict[str, Any] = {}
    bulk_sources = tuple(source for source in source_order if source in BULK_ENRICHMENT_SOURCES)
    if not bulk_sources:
        return bulk_coverage
    bundle = clients or (client_factory() if client_factory else default_enrichment_clients(max_retries=max_retries))
    for source in bulk_sources:
        display_source = _source_display_names((source,))[0]
        if (source, BULK_REGISTRY_WORK_KEY) in completed_work:
            continue
        client = bundle.get(source)
        if client is None:
            error_class = "missing_client"
            errors.append(_bulk_source_error(source=source, error_class=error_class))
            work_ledger.append(_bulk_source_work_record(source=source, status="error", error_class=error_class, request_count=0))
            continue
        try:
            result = client.enrich_registry(taxa_rows=taxa_rows, name_rows=name_rows)
        except Exception as exc:  # noqa: BLE001 - bulk source errors are staged and quarantined.
            error_class = type(exc).__name__
            errors.append(_bulk_source_error(source=source, error_class=error_class))
            logger.info("registry.enrichment.bulk_source_error source=%s error=%s", source, error_class)
            work_ledger.append(_bulk_source_work_record(source=source, status="error", error_class=error_class, request_count=0))
            continue
        name_assertions.extend(result.get("name_assertions", []))
        external_links.extend(result.get("external_links", []))
        source_snapshots.extend(result.get("source_snapshots", []))
        coverage = result.get("coverage", {})
        if isinstance(coverage, dict):
            bulk_coverage[display_source] = coverage
        request_count = int(coverage.get("request_count") or 1) if isinstance(coverage, dict) else 1
        work_ledger.append(_bulk_source_work_record(source=source, status="complete", request_count=request_count))
    return bulk_coverage


def _source_worker_limits(sources: tuple[str, ...], workers: int) -> dict[str, int]:
    return {source: SOURCE_WORKER_LIMITS.get(source, workers) for source in sources}


def _source_query_limit(source: str):
    semaphore = _source_semaphores.get(source)
    return semaphore if semaphore is not None else nullcontext()


def _request_count_for_source(source: str) -> int:
    return INATURALIST_REQUESTS_PER_SPECIES if source == "inaturalist" else 1


def _source_work_record(
    *,
    source: str,
    context: SpeciesContext,
    status: str,
    error_class: str = "",
    request_count: int = 1,
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "source": source,
        "accepted_taxon_key": context.accepted_taxon_key,
        "accepted_scientific_name": context.accepted_scientific_name,
        "status": status,
        "attempts": 1,
        "started_at": now,
        "finished_at": now,
        "request_count": request_count,
        "error_class": error_class,
        "request_day": datetime.now(UTC).date().isoformat(),
    }


def _bulk_source_work_record(
    *,
    source: str,
    status: str,
    error_class: str = "",
    request_count: int = 1,
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "source": source,
        "accepted_taxon_key": BULK_REGISTRY_WORK_KEY,
        "accepted_scientific_name": "registry",
        "status": status,
        "attempts": 1,
        "started_at": now,
        "finished_at": now,
        "request_count": request_count,
        "error_class": error_class,
        "request_day": datetime.now(UTC).date().isoformat(),
    }


def _bulk_source_error(*, source: str, error_class: str) -> dict[str, str]:
    return {
        "source": source,
        "accepted_taxon_key": BULK_REGISTRY_WORK_KEY,
        "error": error_class,
        "error_class": error_class,
        "endpoint": "",
        "attempts": "1",
        "retryable": "false",
        "disposition": "quarantined",
    }


def _run_status(*, completed: int, total: int, budgets: dict[str, DailyRequestBudget]) -> str:
    if any(budget.exhausted for budget in budgets.values()):
        return "budget_exhausted"
    return "partial" if completed < total else "complete"


def _write_enrichment_checkpoint(
    registry: Path,
    *,
    name_assertions: list[dict[str, Any]],
    external_links: list[dict[str, Any]],
    source_snapshots: list[dict[str, Any]],
    errors: list[dict[str, str]],
    work_ledger: list[dict[str, Any]],
    completed: int,
    total: int,
    source_order: tuple[str, ...],
    workers: int,
    source_worker_limits: dict[str, int],
    progress_every: int,
    checkpoint_every: int,
    max_retries: int,
    limit: int,
    started: float,
    base_registry_dir: Path,
    status: str,
    bulk_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    deduplicated_work = _deduplicate_latest_dicts(work_ledger, keys=("source", "accepted_taxon_key"))
    manifest = write_enrichment_sources(
        registry,
        name_assertions=_deduplicate_dicts(name_assertions, keys=("assertion_id", "accepted_taxon_key", "source", "source_record_id", "display_name")),
        external_links=_deduplicate_dicts(external_links, keys=("accepted_taxon_key", "source", "source_taxon_id", "match_method")),
        source_snapshots=_deduplicate_dicts(source_snapshots, keys=("source", "source_version", "source_path", "source_response_hash")),
        source_errors=errors,
        source_work=deduplicated_work,
    )
    coverage = _enrichment_coverage(name_assertions=name_assertions, source_order=source_order, total_species=total, bulk_coverage=bulk_coverage)
    _write_coverage_report(registry, coverage)
    manifest.update(
        {
            "registry_dir": str(registry),
            "source_order": list(source_order),
            "source_config_hash": _stable_id("sources", *source_order),
            "base_registry_dir": str(base_registry_dir),
            "base_registry_hash": _registry_artifact_hash(base_registry_dir),
            "source_worker_limits": source_worker_limits,
            "source_budget_exhausted": sorted(
                {
                    str(row.get("source") or "")
                    for row in deduplicated_work
                    if str(row.get("status") or "") == "budget_exhausted"
                }
            ),
            "species_seen": total,
            "completed_species": completed,
            "status": status,
            "workers": workers,
            "progress_every": progress_every,
            "checkpoint_every": checkpoint_every,
            "max_retries": max_retries,
            "limit": limit,
            "coverage": coverage,
            "errors": errors,
            "error_counts_by_source": dict(sorted(Counter(error["source"] for error in errors).items())),
            "elapsed_seconds": round(monotonic() - started, 6),
            "species_per_second": round(completed / max(monotonic() - started, 0.000001), 6),
            "artifact_bytes": _artifact_bytes(registry),
        }
    )
    (registry / ENRICHMENT_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(
        "registry.enrichment.checkpoint_write completed=%d/%d status=%s name_assertion_rows=%d external_taxon_link_rows=%d source_snapshot_rows=%d files=%s artifact_bytes=%s elapsed_seconds=%.1f",
        completed,
        total,
        status,
        manifest["name_assertion_rows"],
        manifest["external_taxon_link_rows"],
        manifest["source_snapshot_rows"],
        ",".join(manifest["files"].values()),
        manifest["artifact_bytes"],
        manifest["elapsed_seconds"],
    )
    return manifest


def compile_enriched_registry(
    *,
    registry_dir: str | Path | None = None,
    registry_version: str,
    scope_path: str | Path = "config/butterfly_scope.json",
    base_registry_dir: str | Path | None = None,
    enrichment_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    requested_sources: tuple[str, ...] | None = None,
    requested_translation_sources: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    registry = Path(registry_dir) if registry_dir is not None else Path(base_registry_dir or "")
    base = Path(base_registry_dir) if base_registry_dir is not None else registry
    enrichment = Path(enrichment_dir) if enrichment_dir is not None else registry
    output = Path(output_dir) if output_dir is not None else registry
    output.mkdir(parents=True, exist_ok=True)

    taxa = pl.read_parquet(base / "taxa.parquet")
    base_names = pl.read_parquet(base / "names.parquet")
    base_snapshots = pl.read_parquet(base / "source_snapshots.parquet") if (base / "source_snapshots.parquet").exists() else pl.DataFrame(schema=_source_snapshot_schema())
    assertions = _read_or_empty(enrichment / SOURCE_ASSERTIONS_FILE, _name_assertion_schema())
    candidate_translation_source_names = None if requested_translation_sources is None else set(translation_source_display_names(requested_translation_sources))
    translation_source_names = set(translation_source_display_names(requested_translation_sources if requested_translation_sources is not None else DEFAULT_TRANSLATION_SOURCES))
    translation_candidates = _read_or_empty(enrichment / TRANSLATION_CANDIDATES_FILE, translation_candidate_schema())
    if candidate_translation_source_names is None:
        pass
    elif candidate_translation_source_names:
        translation_candidates = translation_candidates.filter(pl.col("source").is_in(candidate_translation_source_names))
    else:
        translation_candidates = pl.DataFrame(schema=translation_candidate_schema())
    translation_assertions = _translation_candidate_assertions(enrichment / TRANSLATION_CANDIDATES_FILE, allowed_sources=candidate_translation_source_names)
    external_links = _read_or_empty(enrichment / EXTERNAL_LINKS_FILE, _external_link_schema())
    enrichment_snapshots = _read_or_empty(enrichment / ENRICHMENT_SOURCE_SNAPSHOTS_FILE, _source_snapshot_schema())
    source_errors = _read_or_empty(enrichment / SOURCE_ERRORS_FILE, _source_error_schema())
    source_work = _read_or_empty(enrichment / SOURCE_WORK_LEDGER_FILE, _source_work_schema())
    effective_sources = requested_sources or DEFAULT_ENRICHMENT_SOURCES
    effective_translation_sources = requested_translation_sources if requested_translation_sources is not None else DEFAULT_TRANSLATION_SOURCES
    allowed_sources = tuple(dict.fromkeys((*_source_display_names(effective_sources), *translation_source_names)))
    allowed_error_sources = tuple(dict.fromkeys((*effective_sources, *effective_translation_sources, *allowed_sources)))
    assertions = assertions.filter(pl.col("source").is_in(allowed_sources))
    if not translation_assertions.is_empty():
        assertions = pl.concat([assertions, translation_assertions], how="vertical")
    external_links = external_links.filter(pl.col("source").is_in(allowed_sources))
    enrichment_snapshots = enrichment_snapshots.filter(pl.col("source").is_in(allowed_sources))
    source_errors = source_errors.filter(pl.col("source").is_in(allowed_error_sources))
    source_work = source_work.filter(pl.col("source").is_in(tuple(dict.fromkeys((*effective_sources, *effective_translation_sources)))))
    assertions = _name_assertions_frame(assertions.to_dicts())

    accepted_keys = set(taxa["accepted_taxon_key"].to_list())
    candidates = _candidate_frame(assertions, accepted_keys)
    enabled_enrichment = candidates.filter(pl.col("enabled") & (pl.col("disabled_reason") == ""))
    combined_names = _combine_names(base_names, enabled_enrichment)
    source_payload = _source_payload(taxa, combined_names, base, enrichment)
    source_json = output / "combined_source_snapshot.json"
    source_json.write_text(json.dumps(source_payload, indent=2, sort_keys=True), encoding="utf-8")

    manifest = compile_registry_fixture(
        source_json,
        output,
        registry_version=registry_version,
        scope_path=scope_path,
    )
    candidate_output = candidates.filter(~(pl.col("enabled") & (pl.col("disabled_reason") == "")))
    candidate_output.write_parquet(output / NAME_CANDIDATES_FILE)
    translation_candidates.write_parquet(output / TRANSLATION_CANDIDATES_FILE)
    translation_work_source = enrichment / TRANSLATION_WORK_LEDGER_FILE
    if translation_work_source.exists() and effective_translation_sources:
        shutil.copy2(translation_work_source, output / TRANSLATION_WORK_LEDGER_FILE)
    assertions.write_parquet(output / SOURCE_ASSERTIONS_FILE)
    external_links.write_parquet(output / EXTERNAL_LINKS_FILE)
    source_errors.write_parquet(output / SOURCE_ERRORS_FILE)
    source_work.write_parquet(output / SOURCE_WORK_LEDGER_FILE)
    _merged_source_snapshots(base_snapshots, enrichment_snapshots).write_parquet(output / FINAL_SOURCE_SNAPSHOTS_FILE)
    _write_enriched_evidence(output, registry_version=registry_version, source_payload=source_payload, assertions=assertions)

    extra_qa = _enrichment_qa(assertions, accepted_keys, source_errors=source_errors)
    if extra_qa:
        qa = pl.read_parquet(output / "qa_findings.parquet")
        qa = pl.concat([qa, pl.DataFrame(extra_qa, schema={"severity": pl.String, "code": pl.String, "subject": pl.String})], how="vertical")
        qa.write_parquet(output / "qa_findings.parquet")
        fatal_count = qa.filter(pl.col("severity") == "fatal").height
        manifest["qa_finding_rows"] = qa.height
        manifest["qa_fatal_count"] = fatal_count
        manifest["qa_warning_count"] = qa.filter(pl.col("severity") == "warning").height
        manifest["qa_status"] = "failed" if fatal_count else "passed"

    final_names = pl.read_parquet(output / "names.parquet")
    query_definitions = pl.read_parquet(output / "flickr_query_definitions.parquet")
    enabled_t5_name_rows = _enabled_t5_name_count(final_names)
    t5_query_definition_rows = _t5_query_definition_count(query_definitions)

    manifest.update(
        {
            "base_registry_dir": str(base),
            "registry_dir": str(output),
            "enrichment_dir": str(enrichment),
            "enrichment_schema_version": ENRICHMENT_SCHEMA_VERSION,
            "enrichment_name_assertion_rows": assertions.height,
            "translation_candidate_rows": translation_assertions.height,
            "enabled_enrichment_name_rows": enabled_enrichment.height,
            "enabled_t5_name_rows": enabled_t5_name_rows,
            "name_candidate_rows": candidate_output.height,
            "t5_retrieval_query_definition_rows": 0,
            "t5_query_definition_rows": t5_query_definition_rows,
            "query_definition_rows": int(query_definitions.height),
            "external_taxon_link_rows": external_links.height,
            "source_error_rows": source_errors.height,
            "source_work_rows": source_work.height,
            "enrichment_sources": list(effective_sources),
            "translation_sources": list(effective_translation_sources),
        }
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _enabled_t5_name_count(names: pl.DataFrame) -> int:
    if names.is_empty() or "trust_tier" not in names.columns or "enabled" not in names.columns:
        return 0
    return names.filter((pl.col("trust_tier") == "T5") & pl.col("enabled")).height


def _t5_query_definition_count(queries: pl.DataFrame) -> int:
    if queries.is_empty() or "trust_tier" not in queries.columns:
        return 0
    return queries.filter(pl.col("trust_tier") == "T5").height


def _read_or_empty(path: Path, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=schema)


def _validate_runtime_options(*, workers: int, progress_every: int, checkpoint_every: int, max_retries: int, limit: int) -> None:
    if workers < 1 or workers > 32:
        raise ValueError("workers must be between 1 and 32")
    if progress_every < 1:
        raise ValueError("progress_every must be >= 1")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if limit < 0:
        raise ValueError("limit must be >= 0")


def _registry_version(registry: Path) -> str:
    manifest_path = registry / "manifest.json"
    if not manifest_path.exists():
        return "unknown"
    try:
        return str(json.loads(manifest_path.read_text(encoding="utf-8")).get("registry_version") or "unknown")
    except json.JSONDecodeError:
        return "unknown"


def _artifact_bytes(registry: Path) -> dict[str, int]:
    return {
        file_name: ((registry / file_name).stat().st_size if (registry / file_name).exists() else 0)
        for file_name in (
            SOURCE_ASSERTIONS_FILE,
            EXTERNAL_LINKS_FILE,
            ENRICHMENT_SOURCE_SNAPSHOTS_FILE,
            SOURCE_ERRORS_FILE,
            SOURCE_WORK_LEDGER_FILE,
            ENRICHMENT_MANIFEST_FILE,
            "enrichment_coverage.json",
            "enrichment_coverage.md",
        )
    }


def _enrichment_report(
    *,
    registry: Path,
    registry_version: str,
    manifest: dict[str, Any],
    errors: list[dict[str, str]],
    source_order: tuple[str, ...],
    workers: int,
    source_worker_limits: dict[str, int],
    progress_every: int,
    checkpoint_every: int,
    max_retries: int,
    limit: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "command": "biominer registry enrich-sources",
        "git_sha": _git_sha(),
        "pid": os.getpid(),
        "registry_version": registry_version,
        "registry_dir": str(registry),
        "status": manifest.get("status"),
        "source_order": list(source_order),
        "source_worker_limits": source_worker_limits,
        "workers": workers,
        "progress_every": progress_every,
        "checkpoint_every": checkpoint_every,
        "max_retries": max_retries,
        "limit": limit,
        "species_seen": manifest.get("species_seen"),
        "completed_species": manifest.get("completed_species"),
        "name_assertion_rows": manifest.get("name_assertion_rows"),
        "external_taxon_link_rows": manifest.get("external_taxon_link_rows"),
        "source_snapshot_rows": manifest.get("source_snapshot_rows"),
        "source_error_rows": manifest.get("source_error_rows"),
        "error_count": len(errors),
        "error_counts_by_source": dict(sorted(Counter(error["source"] for error in errors).items())),
        "artifact_bytes": manifest.get("artifact_bytes"),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "unsupported_metrics": {
            "rss_peak_memory": "not_instrumented",
            "gpu_memory": "not_applicable",
        },
    }


def _write_enrichment_reports(report: dict[str, Any], *, report_dir: Path, registry_version: str) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"registry_enrichment_{registry_version}"
    json_path = report_dir / f"{stem}.json"
    md_path = report_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_enrichment_report_markdown(report), encoding="utf-8")
    return {"report_json": str(json_path), "report_md": str(md_path)}


def _enrichment_report_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# Registry Enrichment {report['registry_version']}",
            "",
            f"- Status: {report['status']}",
            f"- Registry: {report['registry_dir']}",
            f"- Sources: {', '.join(report['source_order'])}",
            f"- Completed species: {report['completed_species']}/{report['species_seen']}",
            f"- Name assertions: {report['name_assertion_rows']}",
            f"- External links: {report['external_taxon_link_rows']}",
            f"- Source snapshots: {report['source_snapshot_rows']}",
            f"- Source errors: {report['source_error_rows']}",
            f"- Errors: {report['error_count']}",
            "",
        ]
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "not_instrumented"


def _name_assertions_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    normalized = [_normalize_assertion(row) for row in rows]
    return pl.DataFrame(normalized, schema=_name_assertion_schema()) if normalized else pl.DataFrame(schema=_name_assertion_schema())


def _translation_candidate_assertions(path: Path, *, allowed_sources: set[str] | None = None) -> pl.DataFrame:
    candidates = _read_or_empty(path, translation_candidate_schema())
    if candidates.is_empty():
        return pl.DataFrame(schema=_name_assertion_schema())
    rows: list[dict[str, Any]] = []
    for row in candidates.to_dicts():
        translated_name = str(row.get("translated_name") or "").strip()
        accepted_taxon_key = str(row.get("accepted_taxon_key") or "").strip()
        if not translated_name or not accepted_taxon_key:
            continue
        source = str(row.get("source") or "Translation").strip() or "Translation"
        if allowed_sources is not None and source not in allowed_sources:
            continue
        candidate_id = str(row.get("candidate_id") or row.get("source_record_id") or "").strip()
        source_record_id = str(row.get("source_record_id") or candidate_id).strip() or _stable_id(
            "translation",
            source,
            accepted_taxon_key,
            row.get("target_language"),
            translated_name,
        )
        rows.append(
            {
                "assertion_id": _stable_id("translation-assertion", source, source_record_id, accepted_taxon_key, translated_name),
                "accepted_taxon_key": accepted_taxon_key,
                "verbatim_name": translated_name,
                "display_name": translated_name,
                "normalized_match_key": normalize_name_key(translated_name),
                "language": normalize_language_code(row.get("target_language")),
                "script": "",
                "region": "",
                "bbox": "",
                "name_class": "generated_translation",
                "source": source,
                "source_record_id": source_record_id,
                "source_taxon_id": "",
                "trust_tier": str(row.get("trust_tier") or "T5"),
                "precision_tier": str(row.get("precision_tier") or "low"),
                "confidence": str(row.get("confidence") or "low"),
                "enabled": _boolish(row.get("enabled", True)),
                "review_state": str(row.get("review_state") or "accepted"),
                "disabled_reason": str(row.get("disabled_reason") or ""),
                "retrieved_at": "",
                "licence": "",
            }
        )
    return pl.DataFrame(rows, schema=_name_assertion_schema()) if rows else pl.DataFrame(schema=_name_assertion_schema())


def _normalize_assertion(row: dict[str, Any]) -> dict[str, Any]:
    display_name = str(row.get("display_name") or row.get("verbatim_name") or "")
    source = str(row.get("source") or "")
    source_record_id = str(row.get("source_record_id") or "")
    language = normalize_language_code(row.get("language"))
    decision = decide_name_trust(
        source=source,
        name_class=str(row.get("name_class") or "vernacular"),
        trust_tier=str(row.get("trust_tier") or ""),
        confidence=str(row.get("confidence") or ""),
        collision_status=str(row.get("collision_status") or ""),
        review_state=str(row.get("review_state") or ""),
        external_taxon_link_confident=_external_link_confident(row),
        corroborated=bool(row.get("corroborated", False)),
    )
    source_enabled = bool(row.get("enabled", True))
    enabled = bool(source_enabled and decision.enabled)
    disabled_reason = str(row.get("disabled_reason") or decision.disabled_reason)
    return {
        "assertion_id": str(row.get("assertion_id") or _stable_id("assertion", source, source_record_id, row.get("accepted_taxon_key"), display_name)),
        "accepted_taxon_key": str(row.get("accepted_taxon_key") or ""),
        "verbatim_name": str(row.get("verbatim_name") or display_name),
        "display_name": display_name,
        "normalized_match_key": normalize_name_key(display_name),
        "language": language,
        "script": str(row.get("script") or ""),
        "region": str(row.get("region") or ""),
        "bbox": str(row.get("bbox") or ""),
        "name_class": str(row.get("name_class") or "vernacular"),
        "source": source,
        "source_record_id": source_record_id,
        "source_taxon_id": str(row.get("source_taxon_id") or ""),
        "trust_tier": decision.trust_tier.value,
        "precision_tier": str(row.get("precision_tier") or ""),
        "confidence": str(row.get("confidence") or ""),
        "enabled": enabled,
        "review_state": str(row.get("review_state") or ("accepted" if enabled else "candidate")),
        "disabled_reason": "" if enabled else disabled_reason,
        "retrieved_at": str(row.get("retrieved_at") or ""),
        "licence": str(row.get("licence") or ""),
    }


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "accepted", "enabled"}


def _candidate_frame(assertions: pl.DataFrame, accepted_keys: set[str]) -> pl.DataFrame:
    if assertions.is_empty():
        return pl.DataFrame(schema=_candidate_schema())
    collision_keys = _collision_keys(assertions)
    rows = []
    for row in assertions.to_dicts():
        disabled_reason = str(row.get("disabled_reason") or "")
        enabled = bool(row.get("enabled"))
        if str(row.get("accepted_taxon_key") or "") not in accepted_keys:
            enabled = False
            disabled_reason = disabled_reason or "unknown_accepted_taxon_key"
        elif _candidate_collision_key(row) in collision_keys:
            decision = decide_name_trust(
                source=str(row.get("source") or ""),
                name_class=str(row.get("name_class") or ""),
                trust_tier=str(row.get("trust_tier") or ""),
                confidence=str(row.get("confidence") or ""),
                collision_status="collision",
                review_state=str(row.get("review_state") or ""),
                external_taxon_link_confident=True,
            )
            enabled = bool(enabled and decision.enabled)
            disabled_reason = "" if enabled else disabled_reason or decision.disabled_reason
        rows.append({**row, "enabled": enabled, "disabled_reason": disabled_reason})
    return pl.DataFrame(rows, schema=_candidate_schema())


def _external_link_confident(row: dict[str, Any]) -> bool:
    confidence = str(row.get("match_confidence") or row.get("external_taxon_link_confidence") or "").casefold()
    lineage = str(row.get("lineage_check") or "").casefold()
    source_taxon_id = str(row.get("source_taxon_id") or "")
    return confidence in {"high", "confident", "accepted"} or lineage in {"accepted_taxon_key", "confident"} or bool(source_taxon_id)


def _collision_keys(assertions: pl.DataFrame) -> set[tuple[str, str, str]]:
    grouped: dict[tuple[str, str, str], set[str]] = {}
    for row in assertions.to_dicts():
        key = _candidate_collision_key(row)
        if not key[0]:
            continue
        grouped.setdefault(key, set()).add(str(row.get("accepted_taxon_key") or ""))
    return {key for key, taxa in grouped.items() if len(taxa) > 1}


def _candidate_collision_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("normalized_match_key") or normalize_name_key(row.get("display_name") or row.get("verbatim_name"))),
        str(row.get("language") or ""),
        str(row.get("name_class") or ""),
    )


def _combine_names(base_names: pl.DataFrame, enabled_enrichment: pl.DataFrame) -> pl.DataFrame:
    base_rows = [_source_name_row(row) for row in base_names.to_dicts()]
    enrichment_rows = [_source_name_row(row) for row in enabled_enrichment.to_dicts()]
    unique: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in [*base_rows, *enrichment_rows]:
        key = (
            str(row["accepted_taxon_key"]),
            normalize_name_key(row["display_name"]),
            str(row["language"]),
            str(row["region"]),
            str(row["name_class"]),
        )
        if key not in unique or _source_rank(row["source"]) < _source_rank(unique[key]["source"]):
            unique[key] = row
    return pl.DataFrame(list(unique.values()))


def _source_rank(source: str) -> int:
    return {"GBIF": 0, "CoL": 1, "TMD": 2, "iNaturalist": 3, "ITIS": 4}.get(source, 9)


def _source_name_row(row: dict[str, Any]) -> dict[str, Any]:
    display_name = str(row.get("display_name") or row.get("verbatim_name") or "")
    language = normalize_language_code(row.get("language"))
    enabled = bool(row.get("enabled", True))
    return {
        "accepted_taxon_key": str(row.get("accepted_taxon_key") or ""),
        "verbatim_name": str(row.get("verbatim_name") or display_name),
        "display_name": display_name,
        "language": language,
        "script": str(row.get("script") or ""),
        "region": str(row.get("region") or ""),
        "bbox": str(row.get("bbox") or ""),
        "name_class": str(row.get("name_class") or ""),
        "source": str(row.get("source") or ""),
        "source_record_id": str(row.get("source_record_id") or ""),
        "trust_tier": str(row.get("trust_tier") or ""),
        "precision_tier": str(row.get("precision_tier") or ""),
        "confidence": str(row.get("confidence") or ""),
        "enabled": enabled,
        "disabled_reason": str(row.get("disabled_reason") or ""),
        "review_state": str(row.get("review_state") or ("accepted" if enabled else "disabled")),
        "corroborated": bool(row.get("corroborated", False)),
        "licence": str(row.get("licence") or ""),
    }


def _source_payload(taxa: pl.DataFrame, names: pl.DataFrame, base: Path, enrichment: Path) -> dict[str, Any]:
    return {
        "source": "GBIF+enrichment",
        "source_version": ENRICHMENT_SCHEMA_VERSION,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "base_registry_dir": str(base),
        "enrichment_dir": str(enrichment),
        "taxa": [_source_taxon_row(row) for row in taxa.to_dicts()],
        "names": [_source_name_row(row) for row in names.to_dicts()],
    }


def _source_taxon_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted_taxon_key": str(row.get("accepted_taxon_key") or ""),
        "scientific_name": str(row.get("scientific_name") or ""),
        "rank": str(row.get("rank") or ""),
        "parent_key": str(row.get("parent_key") or ""),
        "family_key": str(row.get("family_key") or ""),
        "family": str(row.get("family") or ""),
        "genus_key": str(row.get("genus_key") or ""),
        "genus": str(row.get("genus") or ""),
        "species_key": str(row.get("species_key") or ""),
        "species": str(row.get("species") or ""),
    }


def _merged_source_snapshots(base_snapshots: pl.DataFrame, enrichment_snapshots: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([base_snapshots, enrichment_snapshots], how="vertical_relaxed")


def _write_enriched_evidence(output: Path, *, registry_version: str, source_payload: dict[str, Any], assertions: pl.DataFrame) -> None:
    base_evidence = pl.read_parquet(output / "name_evidence.parquet")
    rows = base_evidence.to_dicts()
    seen_evidence_ids = {str(row.get("evidence_id") or "") for row in rows}
    source_hash = _payload_hash(source_payload)
    for row in assertions.to_dicts():
        display_name = str(row.get("display_name") or "")
        language = normalize_language_code(row.get("language"))
        name_id = _stable_id("name", registry_version, row.get("accepted_taxon_key"), display_name, language, row.get("region"))
        evidence_id = _stable_id("evidence", registry_version, row.get("accepted_taxon_key"), display_name, row.get("source"), row.get("source_record_id"))
        if evidence_id in seen_evidence_ids:
            continue
        seen_evidence_ids.add(evidence_id)
        rows.append(
            {
                "evidence_id": evidence_id,
                "name_id": name_id,
                "registry_version": registry_version,
                "accepted_taxon_key": str(row.get("accepted_taxon_key") or ""),
                "source": str(row.get("source") or ""),
                "source_record_id": str(row.get("source_record_id") or ""),
                "source_response_hash": source_hash,
                "retrieved_at": str(row.get("retrieved_at") or source_payload.get("retrieved_at") or ""),
                "licence": str(row.get("licence") or ""),
                "trust_tier": str(row.get("trust_tier") or ""),
                "review_state": str(row.get("review_state") or ("accepted" if row.get("enabled") else "candidate")),
            }
        )
    pl.DataFrame(rows, schema=base_evidence.schema).write_parquet(output / "name_evidence.parquet")


def _enrichment_qa(assertions: pl.DataFrame, accepted_keys: set[str], *, source_errors: pl.DataFrame | None = None) -> list[dict[str, str]]:
    findings = []
    for row in assertions.to_dicts():
        if str(row.get("accepted_taxon_key") or "") not in accepted_keys:
            findings.append({"severity": "warning", "code": "enrichment_name_without_base_taxon", "subject": str(row.get("source_record_id") or "")})
    if source_errors is not None:
        for row in source_errors.to_dicts():
            subject = f"{row.get('source')}:{row.get('accepted_taxon_key')}:{row.get('error_class')}"
            findings.append({"severity": "warning", "code": "source_enrichment_error", "subject": subject})
    return findings


def _deduplicate_dicts(rows: list[dict[str, Any]], *, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(column) or "") for column in keys)
        if key not in unique:
            unique[key] = row
    return list(unique.values())


def _deduplicate_latest_dicts(rows: list[dict[str, Any]], *, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(column) or "") for column in keys)
        unique[key] = row
    return list(unique.values())


def _external_links_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    normalized = [
        {
            "accepted_taxon_key": str(row.get("accepted_taxon_key") or ""),
            "source": str(row.get("source") or ""),
            "source_taxon_id": str(row.get("source_taxon_id") or ""),
            "match_method": str(row.get("match_method") or ""),
            "match_confidence": str(row.get("match_confidence") or ""),
            "lineage_check": str(row.get("lineage_check") or ""),
            "retrieved_at": str(row.get("retrieved_at") or ""),
        }
        for row in rows
    ]
    return pl.DataFrame(normalized, schema=_external_link_schema()) if normalized else pl.DataFrame(schema=_external_link_schema())


def _source_snapshots_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    normalized = [
        {
            "source": str(row.get("source") or ""),
            "source_version": str(row.get("source_version") or ""),
            "retrieved_at": str(row.get("retrieved_at") or ""),
            "source_path": str(row.get("source_path") or ""),
            "source_response_hash": str(row.get("source_response_hash") or ""),
            "licence": str(row.get("licence") or ""),
        }
        for row in rows
    ]
    return pl.DataFrame(normalized, schema=_source_snapshot_schema()) if normalized else pl.DataFrame(schema=_source_snapshot_schema())


def _source_errors_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    normalized = [
        {
            "accepted_taxon_key": str(row.get("accepted_taxon_key") or ""),
            "source": str(row.get("source") or ""),
            "endpoint": str(row.get("endpoint") or ""),
            "error_class": str(row.get("error_class") or row.get("error") or ""),
            "attempts": int(row.get("attempts") or 1),
            "retryable": str(row.get("retryable") or "false"),
            "disposition": str(row.get("disposition") or "quarantined"),
        }
        for row in rows
    ]
    return pl.DataFrame(normalized, schema=_source_error_schema()) if normalized else pl.DataFrame(schema=_source_error_schema())


def _source_work_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    normalized = [
        {
            "source": str(row.get("source") or ""),
            "accepted_taxon_key": str(row.get("accepted_taxon_key") or ""),
            "accepted_scientific_name": str(row.get("accepted_scientific_name") or ""),
            "status": str(row.get("status") or ""),
            "attempts": int(row.get("attempts") or 0),
            "started_at": str(row.get("started_at") or ""),
            "finished_at": str(row.get("finished_at") or ""),
            "request_count": int(row.get("request_count") or 0),
            "error_class": str(row.get("error_class") or ""),
            "request_day": str(row.get("request_day") or ""),
        }
        for row in rows
    ]
    return pl.DataFrame(normalized, schema=_source_work_schema()) if normalized else pl.DataFrame(schema=_source_work_schema())


def _enrichment_coverage(
    *,
    name_assertions: list[dict[str, Any]],
    source_order: tuple[str, ...],
    total_species: int,
    bulk_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_names = _source_display_names(source_order)
    rows = [row for row in name_assertions if str(row.get("source") or "") in source_names]
    taxa_by_source: dict[str, set[str]] = {source: set() for source in source_names}
    names_by_source: dict[str, int] = {source: 0 for source in source_names}
    for row in rows:
        source = str(row.get("source") or "")
        names_by_source[source] = names_by_source.get(source, 0) + 1
        taxa_by_source.setdefault(source, set()).add(str(row.get("accepted_taxon_key") or ""))
    enriched_taxa = set().union(*taxa_by_source.values()) if taxa_by_source else set()
    coverage = {
        "total_species": total_species,
        "enriched_species": len(enriched_taxa),
        "gbif_only_species": max(total_species - len(enriched_taxa), 0),
        "name_assertion_rows_by_source": dict(sorted(names_by_source.items())),
        "enriched_species_by_source": {
            source: len(keys)
            for source, keys in sorted(taxa_by_source.items())
        },
    }
    if bulk_coverage:
        coverage["bulk_sources"] = bulk_coverage
    return coverage


def _write_coverage_report(output: Path, coverage: dict[str, Any]) -> None:
    (output / "enrichment_coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Registry Enrichment Coverage",
        "",
        f"- Total species: {coverage['total_species']}",
        f"- Enriched species: {coverage['enriched_species']}",
        f"- GBIF-only species: {coverage['gbif_only_species']}",
        "",
        "## Name assertions by source",
        "",
    ]
    lines.extend(f"- {source}: {count}" for source, count in coverage["name_assertion_rows_by_source"].items())
    (output / "enrichment_coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _registry_artifact_hash(registry: Path) -> str:
    digest = hashlib.sha256()
    for file_name in ("taxa.parquet", "names.parquet", "manifest.json"):
        path = registry / file_name
        if path.exists():
            digest.update(file_name.encode("utf-8"))
            digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _source_display_names(sources: tuple[str, ...]) -> tuple[str, ...]:
    mapping = {
        "col": "CoL",
        "itis": "ITIS",
        "inaturalist": "iNaturalist",
        "mymemory": "MyMemory",
        "tmd_de": "TMD",
        "wikidata": "Wikidata",
        "wikimedia": "Wikimedia",
        "translation": "Translation",
    }
    return tuple(mapping.get(source, source) for source in sources)


def _name_assertion_schema() -> dict[str, pl.DataType]:
    return {
        "assertion_id": pl.String,
        "accepted_taxon_key": pl.String,
        "verbatim_name": pl.String,
        "display_name": pl.String,
        "normalized_match_key": pl.String,
        "language": pl.String,
        "script": pl.String,
        "region": pl.String,
        "bbox": pl.String,
        "name_class": pl.String,
        "source": pl.String,
        "source_record_id": pl.String,
        "source_taxon_id": pl.String,
        "trust_tier": pl.String,
        "precision_tier": pl.String,
        "confidence": pl.String,
        "enabled": pl.Boolean,
        "review_state": pl.String,
        "disabled_reason": pl.String,
        "retrieved_at": pl.String,
        "licence": pl.String,
    }


def _candidate_schema() -> dict[str, pl.DataType]:
    return _name_assertion_schema()


def _external_link_schema() -> dict[str, pl.DataType]:
    return {
        "accepted_taxon_key": pl.String,
        "source": pl.String,
        "source_taxon_id": pl.String,
        "match_method": pl.String,
        "match_confidence": pl.String,
        "lineage_check": pl.String,
        "retrieved_at": pl.String,
    }


def _source_snapshot_schema() -> dict[str, pl.DataType]:
    return {
        "source": pl.String,
        "source_version": pl.String,
        "retrieved_at": pl.String,
        "source_path": pl.String,
        "source_response_hash": pl.String,
        "licence": pl.String,
    }


def _source_error_schema() -> dict[str, pl.DataType]:
    return {
        "accepted_taxon_key": pl.String,
        "source": pl.String,
        "endpoint": pl.String,
        "error_class": pl.String,
        "attempts": pl.Int64,
        "retryable": pl.String,
        "disposition": pl.String,
    }


def _source_work_schema() -> dict[str, pl.DataType]:
    return {
        "source": pl.String,
        "accepted_taxon_key": pl.String,
        "accepted_scientific_name": pl.String,
        "status": pl.String,
        "attempts": pl.Int64,
        "started_at": pl.String,
        "finished_at": pl.String,
        "request_count": pl.Int64,
        "error_class": pl.String,
        "request_day": pl.String,
    }


def _stable_id(*parts: object) -> str:
    payload = json.dumps([str(part or "") for part in parts], ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _payload_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()
