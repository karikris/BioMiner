from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import threading
from time import monotonic
from typing import Any

import polars as pl

from biominer.common.artifacts import write_json_artifact, write_parquet_artifact
from biominer.common.concurrency import bounded_map_ordered
from biominer.registry.compiler import compile_registry_fixture
from biominer.registry.normalize import normalize_name_key


ENRICHMENT_SCHEMA_VERSION = "registry-enrichment-v1"
SOURCE_ASSERTIONS_FILE = "source_name_assertions.parquet"
EXTERNAL_LINKS_FILE = "external_taxon_links.parquet"
ENRICHMENT_SOURCE_SNAPSHOTS_FILE = "enrichment_source_snapshots.parquet"
FINAL_SOURCE_SNAPSHOTS_FILE = "source_snapshots.parquet"
NAME_CANDIDATES_FILE = "name_candidates.parquet"
ENRICHMENT_MANIFEST_FILE = "enrichment_manifest.json"


logger = logging.getLogger(__name__)
ClientBundleFactory = Callable[[], dict[str, Any]]
_worker_local = threading.local()
_source_semaphores = {"wikidata": threading.BoundedSemaphore(1)}
_SOURCE_LABELS = {"col": "CoL", "wikidata": "Wikidata", "itis": "ITIS"}


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


class FatalSourceError(RuntimeError):
    """Raised when continuing a source run would harm the source or corrupt results."""


class SourceRateLimitError(FatalSourceError):
    """Raised when a source indicates the job should stop instead of retrying."""


def build_enrichment_sources_from_registry(
    *,
    registry_dir: str | Path,
    sources: tuple[str, ...] = ("col", "wikidata", "itis"),
    clients: dict[str, Any] | None = None,
    client_factory: ClientBundleFactory | None = None,
    workers: int = 8,
    progress_every: int = 100,
    checkpoint_every: int = 500,
    max_retries: int = 5,
    limit: int = 0,
    report_dir: str | Path = "reports",
) -> dict[str, Any]:
    _validate_runtime_options(workers=workers, progress_every=progress_every, checkpoint_every=checkpoint_every, max_retries=max_retries, limit=limit)
    started = monotonic()
    registry = Path(registry_dir)
    taxa = pl.read_parquet(registry / "taxa.parquet")
    names = pl.read_parquet(registry / "names.parquet")
    species_rows = taxa.filter(pl.col("rank") == "SPECIES").sort(["family", "genus", "scientific_name"]).to_dicts()
    if limit:
        species_rows = species_rows[:limit]
    names_by_taxon: dict[str, list[str]] = {}
    for row in names.sort(["accepted_taxon_key", "display_name"]).to_dicts():
        names_by_taxon.setdefault(str(row.get("accepted_taxon_key") or ""), []).append(str(row.get("display_name") or ""))

    source_order = tuple(sources)
    name_assertions: list[dict[str, Any]] = []
    external_links: list[dict[str, Any]] = []
    source_snapshots: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    completed_taxon_keys: list[str] = []
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

    completed = 0
    for result in _enrichment_iterator(
        contexts=contexts,
        sources=source_order,
        clients=clients,
        client_factory=client_factory,
        workers=workers,
        max_retries=max_retries,
    ):
        completed += 1
        completed_taxon_keys.append(result.accepted_taxon_key)
        name_assertions.extend(result.name_assertions)
        external_links.extend(result.external_links)
        source_snapshots.extend(result.source_snapshots)
        errors.extend(result.errors)
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
                registry,
                name_assertions=name_assertions,
                external_links=external_links,
                source_snapshots=source_snapshots,
                errors=errors,
                completed_taxon_keys=completed_taxon_keys,
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
                status="partial" if completed < len(contexts) else "complete",
            )

    if not contexts:
        _write_enrichment_checkpoint(
            registry,
            name_assertions=[],
            external_links=[],
            source_snapshots=[],
            errors=[],
            completed_taxon_keys=[],
            completed=0,
            total=0,
            source_order=source_order,
            workers=workers,
            source_worker_limits=_source_worker_limits(source_order, workers),
            progress_every=progress_every,
            checkpoint_every=checkpoint_every,
            max_retries=max_retries,
            limit=limit,
            started=started,
            status="complete",
        )

    manifest = json.loads((registry / ENRICHMENT_MANIFEST_FILE).read_text(encoding="utf-8"))
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
    (registry / ENRICHMENT_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("registry.enrichment.complete registry=%s completed=%d status=%s", registry, completed, manifest.get("status"))
    return manifest


def default_enrichment_clients(*, max_retries: int = 5) -> dict[str, Any]:
    from biominer.registry.enrichment_sources import CatalogueOfLifeClient, ITISClient, WikidataClient

    return {
        "col": CatalogueOfLifeClient(max_retries=max_retries),
        "wikidata": WikidataClient(max_retries=max_retries),
        "itis": ITISClient(max_retries=max_retries),
    }


def write_enrichment_sources(
    output_dir: str | Path,
    *,
    name_assertions: list[dict[str, Any]] | None = None,
    external_links: list[dict[str, Any]] | None = None,
    source_snapshots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    assertions = _name_assertions_frame(name_assertions or [])
    links = _external_links_frame(external_links or [])
    snapshots = _source_snapshots_frame(source_snapshots or [])
    assertion_artifact = write_parquet_artifact(output / SOURCE_ASSERTIONS_FILE, assertions)
    link_artifact = write_parquet_artifact(output / EXTERNAL_LINKS_FILE, links)
    snapshot_artifact = write_parquet_artifact(output / ENRICHMENT_SOURCE_SNAPSHOTS_FILE, snapshots)
    manifest = {
        "schema_version": ENRICHMENT_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "name_assertion_rows": assertions.height,
        "external_taxon_link_rows": links.height,
        "source_snapshot_rows": snapshots.height,
        "files": {
            "source_name_assertions": SOURCE_ASSERTIONS_FILE,
            "external_taxon_links": EXTERNAL_LINKS_FILE,
            "enrichment_source_snapshots": ENRICHMENT_SOURCE_SNAPSHOTS_FILE,
        },
        "artifacts": {
            SOURCE_ASSERTIONS_FILE: assertion_artifact.to_dict(),
            EXTERNAL_LINKS_FILE: link_artifact.to_dict(),
            ENRICHMENT_SOURCE_SNAPSHOTS_FILE: snapshot_artifact.to_dict(),
        },
    }
    manifest_artifact = write_json_artifact(output / ENRICHMENT_MANIFEST_FILE, manifest)
    manifest["artifacts"][ENRICHMENT_MANIFEST_FILE] = manifest_artifact.to_dict()
    write_json_artifact(output / ENRICHMENT_MANIFEST_FILE, manifest)
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
) -> Iterator[SpeciesEnrichmentResult]:
    if not contexts:
        return iter(())
    if workers == 1:
        bundle = clients or (client_factory() if client_factory else default_enrichment_clients(max_retries=max_retries))
        return (_enrich_species_context(context, sources=sources, clients=bundle) for context in contexts)

    def initializer() -> None:
        _worker_local.clients = clients or (client_factory() if client_factory else default_enrichment_clients(max_retries=max_retries))

    def task(context: SpeciesContext) -> SpeciesEnrichmentResult:
        bundle = getattr(_worker_local, "clients", None)
        if bundle is None:
            raise RuntimeError("Enrichment worker clients were not initialized")
        return _enrich_species_context(context, sources=sources, clients=bundle)

    def generator() -> Iterator[SpeciesEnrichmentResult]:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="registry-enrich", initializer=initializer) as executor:
            yield from bounded_map_ordered(executor, task, contexts, buffersize=workers * 4)

    return generator()


def _enrich_species_context(context: SpeciesContext, *, sources: tuple[str, ...], clients: dict[str, Any]) -> SpeciesEnrichmentResult:
    name_assertions: list[dict[str, Any]] = []
    external_links: list[dict[str, Any]] = []
    source_snapshots: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for source in sources:
        client = clients.get(source)
        if client is None:
            errors.append({"source": source, "accepted_taxon_key": context.accepted_taxon_key, "error": "missing_client"})
            continue
        try:
            with _source_query_limit(source):
                result = client.enrich_species(context)
        except FatalSourceError:
            logger.info(
                "registry.enrichment.fatal_source_error source=%s accepted_taxon_key=%s",
                source,
                context.accepted_taxon_key,
            )
            raise
        except Exception as exc:  # noqa: BLE001 - source staging records and continues per species.
            errors.append({"source": source, "accepted_taxon_key": context.accepted_taxon_key, "error": type(exc).__name__})
            logger.info(
                "registry.enrichment.source_error source=%s accepted_taxon_key=%s error=%s",
                source,
                context.accepted_taxon_key,
                type(exc).__name__,
            )
            continue
        name_assertions.extend(result.get("name_assertions", []))
        external_links.extend(result.get("external_links", []))
        source_snapshots.extend(result.get("source_snapshots", []))
    return SpeciesEnrichmentResult(
        accepted_taxon_key=context.accepted_taxon_key,
        name_assertions=tuple(name_assertions),
        external_links=tuple(external_links),
        source_snapshots=tuple(source_snapshots),
        errors=tuple(errors),
    )


def _source_worker_limits(sources: tuple[str, ...], workers: int) -> dict[str, int]:
    return {source: (1 if source == "wikidata" else workers) for source in sources}


def _source_query_limit(source: str):
    semaphore = _source_semaphores.get(source)
    return semaphore if semaphore is not None else nullcontext()


def _source_labels(sources: tuple[str, ...]) -> set[str]:
    return {_SOURCE_LABELS.get(source, source) for source in sources}


@contextmanager
def _registry_enrichment_lock(registry: Path) -> Iterator[None]:
    registry.mkdir(parents=True, exist_ok=True)
    lock_path = registry / ".enrichment_write.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _existing_rows_to_preserve(path: Path, *, source_labels: set[str], completed_taxon_keys: set[str] | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    preserved = []
    for row in pl.read_parquet(path).to_dicts():
        source = str(row.get("source") or "")
        accepted_taxon_key = str(row.get("accepted_taxon_key") or "")
        if source not in source_labels or (completed_taxon_keys is not None and accepted_taxon_key not in completed_taxon_keys):
            preserved.append(row)
    return preserved


def _write_enrichment_checkpoint(
    registry: Path,
    *,
    name_assertions: list[dict[str, Any]],
    external_links: list[dict[str, Any]],
    source_snapshots: list[dict[str, Any]],
    errors: list[dict[str, str]],
    completed_taxon_keys: list[str],
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
    status: str,
) -> dict[str, Any]:
    with _registry_enrichment_lock(registry):
        current_source_labels = _source_labels(source_order)
        completed_taxon_key_set = {str(key) for key in completed_taxon_keys if str(key)}
        staged_assertions = [
            *_existing_rows_to_preserve(registry / SOURCE_ASSERTIONS_FILE, source_labels=current_source_labels, completed_taxon_keys=completed_taxon_key_set),
            *_deduplicate_dicts(name_assertions, keys=("assertion_id", "accepted_taxon_key", "source", "source_record_id", "display_name")),
        ]
        staged_links = [
            *_existing_rows_to_preserve(registry / EXTERNAL_LINKS_FILE, source_labels=current_source_labels, completed_taxon_keys=completed_taxon_key_set),
            *_deduplicate_dicts(external_links, keys=("accepted_taxon_key", "source", "source_taxon_id", "match_method")),
        ]
        staged_snapshots = [
            *_existing_rows_to_preserve(registry / ENRICHMENT_SOURCE_SNAPSHOTS_FILE, source_labels=current_source_labels),
            *_deduplicate_dicts(source_snapshots, keys=("source", "source_version", "source_path", "source_response_hash")),
        ]
        manifest = write_enrichment_sources(
            registry,
            name_assertions=_deduplicate_dicts(staged_assertions, keys=("assertion_id", "accepted_taxon_key", "source", "source_record_id", "display_name")),
            external_links=_deduplicate_dicts(staged_links, keys=("accepted_taxon_key", "source", "source_taxon_id", "match_method")),
            source_snapshots=_deduplicate_dicts(staged_snapshots, keys=("source", "source_version", "source_path", "source_response_hash")),
        )
        manifest.update(
            {
                "registry_dir": str(registry),
                "source_order": list(source_order),
                "source_worker_limits": source_worker_limits,
                "species_seen": total,
                "completed_species": completed,
                "status": status,
                "workers": workers,
                "progress_every": progress_every,
                "checkpoint_every": checkpoint_every,
                "max_retries": max_retries,
                "limit": limit,
                "errors": errors,
                "error_counts_by_source": dict(sorted(Counter(error["source"] for error in errors).items())),
                "elapsed_seconds": round(monotonic() - started, 6),
                "species_per_second": round(completed / max(monotonic() - started, 0.000001), 6),
                "artifact_bytes": _artifact_bytes(registry),
            }
        )
        manifest_artifact = write_json_artifact(registry / ENRICHMENT_MANIFEST_FILE, manifest)
        manifest.setdefault("artifacts", {})[ENRICHMENT_MANIFEST_FILE] = manifest_artifact.to_dict()
        write_json_artifact(registry / ENRICHMENT_MANIFEST_FILE, manifest)
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
    external_links = _read_or_empty(enrichment / EXTERNAL_LINKS_FILE, _external_link_schema())
    enrichment_snapshots = _read_or_empty(enrichment / ENRICHMENT_SOURCE_SNAPSHOTS_FILE, _source_snapshot_schema())

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
    assertions.write_parquet(output / SOURCE_ASSERTIONS_FILE)
    external_links.write_parquet(output / EXTERNAL_LINKS_FILE)
    _merged_source_snapshots(base_snapshots, enrichment_snapshots).write_parquet(output / FINAL_SOURCE_SNAPSHOTS_FILE)
    _write_enriched_evidence(output, registry_version=registry_version, source_payload=source_payload, assertions=assertions)

    extra_qa = _enrichment_qa(assertions, accepted_keys)
    if extra_qa:
        qa = pl.read_parquet(output / "qa_findings.parquet")
        qa = pl.concat([qa, pl.DataFrame(extra_qa, schema={"severity": pl.String, "code": pl.String, "subject": pl.String})], how="vertical")
        qa.write_parquet(output / "qa_findings.parquet")
        fatal_count = qa.filter(pl.col("severity") == "fatal").height
        manifest["qa_finding_rows"] = qa.height
        manifest["qa_fatal_count"] = fatal_count
        manifest["qa_warning_count"] = qa.filter(pl.col("severity") == "warning").height
        manifest["qa_status"] = "failed" if fatal_count else "passed"

    manifest.update(
        {
            "base_registry_dir": str(base),
            "registry_dir": str(output),
            "enrichment_dir": str(enrichment),
            "enrichment_schema_version": ENRICHMENT_SCHEMA_VERSION,
            "enrichment_name_assertion_rows": assertions.height,
            "enabled_enrichment_name_rows": enabled_enrichment.height,
            "name_candidate_rows": candidate_output.height,
            "external_taxon_link_rows": external_links.height,
        }
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


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
            ENRICHMENT_MANIFEST_FILE,
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


def _normalize_assertion(row: dict[str, Any]) -> dict[str, Any]:
    display_name = str(row.get("display_name") or row.get("verbatim_name") or "")
    source = str(row.get("source") or "")
    source_record_id = str(row.get("source_record_id") or "")
    return {
        "assertion_id": str(row.get("assertion_id") or _stable_id("assertion", source, source_record_id, row.get("accepted_taxon_key"), display_name)),
        "accepted_taxon_key": str(row.get("accepted_taxon_key") or ""),
        "verbatim_name": str(row.get("verbatim_name") or display_name),
        "display_name": display_name,
        "normalized_match_key": normalize_name_key(display_name),
        "language": str(row.get("language") or ""),
        "script": str(row.get("script") or ""),
        "region": str(row.get("region") or ""),
        "bbox": str(row.get("bbox") or ""),
        "name_class": str(row.get("name_class") or "vernacular"),
        "source": source,
        "source_record_id": source_record_id,
        "source_taxon_id": str(row.get("source_taxon_id") or ""),
        "trust_tier": str(row.get("trust_tier") or ""),
        "precision_tier": str(row.get("precision_tier") or ""),
        "confidence": str(row.get("confidence") or ""),
        "enabled": bool(row.get("enabled", True)),
        "review_state": str(row.get("review_state") or ("accepted" if row.get("enabled", True) else "candidate")),
        "disabled_reason": str(row.get("disabled_reason") or ""),
        "retrieved_at": str(row.get("retrieved_at") or ""),
        "licence": str(row.get("licence") or ""),
    }


def _candidate_frame(assertions: pl.DataFrame, accepted_keys: set[str]) -> pl.DataFrame:
    if assertions.is_empty():
        return pl.DataFrame(schema=_candidate_schema())
    rows = []
    for row in assertions.to_dicts():
        disabled_reason = str(row.get("disabled_reason") or "")
        enabled = bool(row.get("enabled"))
        if str(row.get("accepted_taxon_key") or "") not in accepted_keys:
            enabled = False
            disabled_reason = disabled_reason or "unknown_accepted_taxon_key"
        rows.append({**row, "enabled": enabled, "disabled_reason": disabled_reason})
    return pl.DataFrame(rows, schema=_candidate_schema())


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
    return {"GBIF": 0, "CoL": 1, "ITIS": 2, "Wikidata": 3}.get(source, 9)


def _source_name_row(row: dict[str, Any]) -> dict[str, Any]:
    display_name = str(row.get("display_name") or row.get("verbatim_name") or "")
    return {
        "accepted_taxon_key": str(row.get("accepted_taxon_key") or ""),
        "verbatim_name": str(row.get("verbatim_name") or display_name),
        "display_name": display_name,
        "language": str(row.get("language") or ""),
        "script": str(row.get("script") or ""),
        "region": str(row.get("region") or ""),
        "bbox": str(row.get("bbox") or ""),
        "name_class": str(row.get("name_class") or ""),
        "source": str(row.get("source") or ""),
        "source_record_id": str(row.get("source_record_id") or ""),
        "trust_tier": str(row.get("trust_tier") or ""),
        "precision_tier": str(row.get("precision_tier") or ""),
        "confidence": str(row.get("confidence") or ""),
        "enabled": bool(row.get("enabled", True)),
        "disabled_reason": str(row.get("disabled_reason") or ""),
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
        name_id = _stable_id("name", registry_version, row.get("accepted_taxon_key"), display_name, row.get("language"), row.get("region"))
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


def _enrichment_qa(assertions: pl.DataFrame, accepted_keys: set[str]) -> list[dict[str, str]]:
    findings = []
    for row in assertions.to_dicts():
        if str(row.get("accepted_taxon_key") or "") not in accepted_keys:
            findings.append({"severity": "warning", "code": "enrichment_name_without_base_taxon", "subject": str(row.get("source_record_id") or "")})
    return findings


def _deduplicate_dicts(rows: list[dict[str, Any]], *, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(column) or "") for column in keys)
        if key not in unique:
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


def _stable_id(*parts: object) -> str:
    payload = json.dumps([str(part or "") for part in parts], ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _payload_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()
