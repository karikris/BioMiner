from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import threading
from time import monotonic
from typing import Any

import polars as pl

from biominer.registry.gbif import GBIFClient, resolve_family
from biominer.registry.gbif_production import ProductionGBIFClient
from biominer.registry.scope import ButterflyScope


logger = logging.getLogger(__name__)
GBIFClientFactory = Callable[[], GBIFClient]
_worker_local = threading.local()


@dataclass(frozen=True)
class SpeciesEnrichment:
    species_key: str
    names: tuple[dict[str, object], ...]
    logical_calls: int
    request_attempts: int
    retries: int


@dataclass
class FamilyCheckpoint:
    completed_species_keys: set[str]
    enrichment_names: list[dict[str, object]]
    logical_calls: int = 0
    request_attempts: int = 0
    retries: int = 0


def build_gbif_source_snapshot(
    client: GBIFClient,
    scope: ButterflyScope,
    *,
    retrieved_at: str,
    page_limit: int = 1000,
    checkpoint_dir: str | Path | None = None,
    workers: int = 1,
    progress_every: int = 100,
    checkpoint_every: int = 500,
    max_retries: int = 5,
    client_factory: GBIFClientFactory | None = None,
) -> dict[str, Any]:
    _validate_runtime_options(
        workers=workers,
        progress_every=progress_every,
        checkpoint_every=checkpoint_every,
        max_retries=max_retries,
    )
    started = monotonic()
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir else None
    if checkpoint_root:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    root_key = scope.root_taxon_key
    if not root_key:
        root_match = client.match_name(
            scope.root_scientific_name,
            rank=scope.root_rank,
        )
        root_key = root_match.get("acceptedUsageKey") or root_match.get("usageKey")
        if root_key is None:
            raise ValueError(
                f"GBIF did not resolve scope root {scope.root_scientific_name!r}"
            )

    root_usage = client.usage(root_key)
    resolved_root_name = _scientific_name(root_usage)
    resolved_root_rank = str(root_usage.get("rank") or "")

    if resolved_root_name != scope.root_scientific_name or resolved_root_rank != scope.root_rank:
        raise ValueError(
            f"Configured GBIF root {root_key!r} resolved to "
            f"{resolved_root_name!r} rank {resolved_root_rank!r}; expected "
            f"{scope.root_scientific_name!r} rank {scope.root_rank!r}"
        )
    taxa = [_taxon_row(root_usage, parent_key="", family={})]
    names = [_scientific_name_row(root_usage, name_class="accepted_scientific")]
    assertions: list[dict[str, Any]] = []
    total_worker_calls = 0
    total_worker_attempts = 0
    total_worker_retries = 0
    total_resumed_species = 0

    for family_name in scope.included_families:
        family_started = monotonic()
        resolution = resolve_family(
            client,
            family_name,
            root_name=scope.root_scientific_name,
            expected_taxon_key=scope.family_taxon_keys.get(family_name),
        )
        family_key = _bare_key(resolution.accepted_taxon_key)
        family_usage = client.usage(family_key)
        family_taxa = [_taxon_row(family_usage, parent_key=f"gbif:{root_key}", family=family_usage)]
        family_base_names = [_scientific_name_row(family_usage, name_class="accepted_scientific")]
        assertions.append(
            {
                "configured_name": family_name,
                "accepted_taxon_key": resolution.accepted_taxon_key,
                "matched_usage_key": resolution.matched_usage_key,
                "match_type": resolution.match_type,
                "confidence": resolution.confidence,
                "lineage_names": list(resolution.lineage_names),
                "root_lineage_verified": (
                    scope.root_scientific_name in resolution.lineage_names
                ),
                "lineage_validation": (
                    "explicit_root"
                    if scope.root_scientific_name in resolution.lineage_names
                    else "configured_family_lepidoptera_fallback"
                ),
            }
        )
        logger.info("registry.gbif.family.start family=%s key=%s", family_name, family_key)

        species_rows: list[dict[str, Any]] = []
        for genus in client.children(family_key, rank="GENUS", limit=page_limit):
            genus_key = genus.get("key")
            family_taxa.append(_taxon_row(genus, parent_key=f"gbif:{family_key}", family=family_usage, genus=genus))
            family_base_names.append(_scientific_name_row(genus, name_class="accepted_scientific"))
            for species in client.children(genus_key, rank="SPECIES", limit=page_limit):
                species_key = species.get("key")
                if species_key is None:
                    continue
                species_rows.append(species)
                family_taxa.append(
                    _taxon_row(
                        species,
                        parent_key=f"gbif:{genus_key}",
                        family=family_usage,
                        genus=genus,
                        species=species,
                    )
                )
                family_base_names.append(_scientific_name_row(species, name_class="accepted_scientific"))

        species_rows.sort(key=lambda row: (_scientific_name(row), str(row.get("key") or "")))
        checkpoint = _load_checkpoint(checkpoint_root, family_name, family_key) if checkpoint_root else FamilyCheckpoint(set(), [])
        total_resumed_species += len(checkpoint.completed_species_keys)
        pending_species = [row for row in species_rows if str(row.get("key") or "") not in checkpoint.completed_species_keys]
        logger.info(
            "registry.gbif.family.enrichment family=%s species=%d resumed=%d pending=%d workers=%d",
            family_name,
            len(species_rows),
            len(checkpoint.completed_species_keys),
            len(pending_species),
            workers,
        )
        completed_this_run = 0
        try:
            for result in _enrichment_iterator(
                client=client,
                species_rows=pending_species,
                workers=workers,
                page_limit=page_limit,
                max_retries=max_retries,
                client_factory=client_factory,
            ):
                checkpoint.enrichment_names.extend(result.names)
                checkpoint.completed_species_keys.add(result.species_key)
                checkpoint.logical_calls += result.logical_calls
                checkpoint.request_attempts += result.request_attempts
                checkpoint.retries += result.retries
                completed_this_run += 1
                total_done = len(checkpoint.completed_species_keys)
                if completed_this_run % progress_every == 0 or total_done == len(species_rows):
                    logger.info(
                        "registry.gbif.family.progress family=%s completed=%d/%d names=%d",
                        family_name,
                        total_done,
                        len(species_rows),
                        len(checkpoint.enrichment_names),
                    )
                if checkpoint_root and completed_this_run % checkpoint_every == 0:
                    _write_checkpoint(checkpoint_root, family_name, family_key, checkpoint, status="partial")
        except BaseException:
            if checkpoint_root:
                _write_checkpoint(checkpoint_root, family_name, family_key, checkpoint, status="partial")
            raise

        if checkpoint_root:
            _write_checkpoint(checkpoint_root, family_name, family_key, checkpoint, status="complete")
        family_enrichment_names = _deduplicate_name_rows(checkpoint.enrichment_names)
        family_names = _interleave_enrichment_names(family_base_names, family_enrichment_names)
        taxa.extend(family_taxa)
        names.extend(family_names)
        total_worker_calls += checkpoint.logical_calls
        total_worker_attempts += checkpoint.request_attempts
        total_worker_retries += checkpoint.retries
        logger.info(
            "registry.gbif.family.complete family=%s species=%d enrichment_names=%d elapsed_seconds=%.1f",
            family_name,
            len(species_rows),
            len(family_enrichment_names),
            monotonic() - family_started,
        )

    taxa = _deduplicate_taxa_rows(taxa)
    names = _deduplicate_name_rows(names)

    return {
        "source": "GBIF",
        "source_version": "gbif-species-api",
        "retrieved_at": retrieved_at,
        "taxa": taxa,
        "names": names,
        "source_assertions": assertions,
        "metrics": _metrics(
            taxa,
            names,
            assertions,
            gbif_calls=client.call_count + (total_worker_calls if workers > 1 else 0),
            request_attempts=int(getattr(client, "request_attempt_count", client.call_count)) + (total_worker_attempts if workers > 1 else 0),
            retries=int(getattr(client, "retry_count", 0)) + (total_worker_retries if workers > 1 else 0),
            elapsed_seconds=monotonic() - started,
            workers=workers,
            progress_every=progress_every,
            checkpoint_every=checkpoint_every,
            resumed_species=total_resumed_species,
        ),
    }


def _enrichment_iterator(
    *,
    client: GBIFClient,
    species_rows: list[dict[str, Any]],
    workers: int,
    page_limit: int,
    max_retries: int,
    client_factory: GBIFClientFactory | None,
) -> Iterator[SpeciesEnrichment]:
    if not species_rows:
        return iter(())
    if workers == 1:
        return (_enrich_species(client, species, page_limit=page_limit) for species in species_rows)
    factory = client_factory or (lambda: ProductionGBIFClient(max_retries=max_retries, max_connections=workers))
    worker_clients: list[GBIFClient] = []
    worker_clients_lock = threading.Lock()

    def initializer() -> None:
        worker_client = factory()
        _worker_local.client = worker_client
        with worker_clients_lock:
            worker_clients.append(worker_client)

    def task(species: dict[str, Any]) -> SpeciesEnrichment:
        worker_client = getattr(_worker_local, "client", None)
        if worker_client is None:
            raise RuntimeError("GBIF worker client was not initialized")
        return _enrich_species(worker_client, species, page_limit=page_limit)

    def generator() -> Iterator[SpeciesEnrichment]:
        try:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gbif-enrich", initializer=initializer) as executor:
                yield from executor.map(task, species_rows)
        finally:
            for worker_client in worker_clients:
                close = getattr(worker_client, "close", None)
                if callable(close):
                    close()

    return generator()


def _enrich_species(client: GBIFClient, species: dict[str, Any], *, page_limit: int) -> SpeciesEnrichment:
    species_key = species.get("key")
    if species_key is None:
        raise ValueError("Species row is missing GBIF key")
    before_calls = client.call_count
    before_attempts = int(getattr(client, "request_attempt_count", before_calls))
    before_retries = int(getattr(client, "retry_count", 0))
    rows: list[dict[str, object]] = []
    for synonym in client.synonyms(species_key, limit=page_limit):
        rows.append(_scientific_name_row(synonym, name_class="scientific_synonym", accepted_key=species_key))
    for vernacular in client.vernacular_names(species_key, limit=page_limit):
        rows.append(_vernacular_name_row(vernacular, accepted_key=species_key))
    return SpeciesEnrichment(
        species_key=str(species_key),
        names=tuple(rows),
        logical_calls=client.call_count - before_calls,
        request_attempts=int(getattr(client, "request_attempt_count", client.call_count)) - before_attempts,
        retries=int(getattr(client, "retry_count", 0)) - before_retries,
    )


def _load_checkpoint(checkpoint_root: Path | None, family_name: str, family_key: str) -> FamilyCheckpoint:
    if checkpoint_root is None:
        return FamilyCheckpoint(set(), [])
    family_dir = _checkpoint_family_dir(checkpoint_root, family_name)
    state_path = family_dir / "state.json"
    names_path = family_dir / "enrichment_names.parquet"
    if not state_path.exists():
        return FamilyCheckpoint(set(), [])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if str(state.get("family_key") or "") != family_key:
        raise ValueError(f"Checkpoint for {family_name} belongs to GBIF key {state.get('family_key')!r}, expected {family_key!r}")
    enrichment_names = pl.read_parquet(names_path).to_dicts() if names_path.exists() else []
    return FamilyCheckpoint(
        completed_species_keys={str(value) for value in state.get("completed_species_keys", [])},
        enrichment_names=enrichment_names,
        logical_calls=int(state.get("logical_calls") or 0),
        request_attempts=int(state.get("request_attempts") or 0),
        retries=int(state.get("retries") or 0),
    )


def _write_checkpoint(
    checkpoint_root: Path,
    family_name: str,
    family_key: str,
    checkpoint: FamilyCheckpoint,
    *,
    status: str,
) -> None:
    family_dir = _checkpoint_family_dir(checkpoint_root, family_name)
    family_dir.mkdir(parents=True, exist_ok=True)
    names_path = family_dir / "enrichment_names.parquet"
    names_tmp = family_dir / "enrichment_names.parquet.tmp"
    state_path = family_dir / "state.json"
    state_tmp = family_dir / "state.json.tmp"
    rows = _deduplicate_name_rows(checkpoint.enrichment_names)
    frame = pl.DataFrame(rows) if rows else pl.DataFrame(schema=_name_schema())
    frame.write_parquet(names_tmp)
    os.replace(names_tmp, names_path)
    state = {
        "schema_version": 1,
        "family": family_name,
        "family_key": family_key,
        "status": status,
        "completed_species_keys": sorted(checkpoint.completed_species_keys, key=_species_key_sort),
        "logical_calls": checkpoint.logical_calls,
        "request_attempts": checkpoint.request_attempts,
        "retries": checkpoint.retries,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    state_tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(state_tmp, state_path)


def _checkpoint_family_dir(checkpoint_root: Path, family_name: str) -> Path:
    safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in family_name)
    return checkpoint_root / safe_name


def _species_key_sort(value: str) -> tuple[int, object]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _validate_runtime_options(*, workers: int, progress_every: int, checkpoint_every: int, max_retries: int) -> None:
    if workers < 1 or workers > 32:
        raise ValueError("workers must be between 1 and 32")
    if progress_every < 1:
        raise ValueError("progress_every must be >= 1")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")


def _metrics(
    taxa: list[dict[str, str]],
    names: list[dict[str, object]],
    assertions: list[dict[str, Any]],
    *,
    gbif_calls: int,
    request_attempts: int,
    retries: int,
    elapsed_seconds: float,
    workers: int,
    progress_every: int,
    checkpoint_every: int,
    resumed_species: int,
) -> dict[str, Any]:
    taxa_by_rank = Counter(row["rank"] for row in taxa)
    name_classes = Counter(str(row["name_class"]) for row in names)
    return {
        "gbif_calls": gbif_calls,
        "gbif_request_attempts": request_attempts,
        "gbif_retries": retries,
        "taxa_rows": len(taxa),
        "taxa_by_rank": dict(sorted(taxa_by_rank.items())),
        "name_rows": len(names),
        "synonym_rows": name_classes.get("scientific_synonym", 0),
        "vernacular_rows": name_classes.get("vernacular", 0),
        "source_assertion_rows": len(assertions),
        "workers": workers,
        "progress_every": progress_every,
        "checkpoint_every": checkpoint_every,
        "resumed_species": resumed_species,
        "elapsed_seconds": round(elapsed_seconds, 6),
    }


def _deduplicate_taxa_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        key = str(row.get("accepted_taxon_key") or "")
        if key not in unique:
            unique[key] = row
    return list(unique.values())


def _deduplicate_name_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    unique: dict[tuple[str, ...], dict[str, object]] = {}
    for row in rows:
        key = (
            str(row.get("accepted_taxon_key") or ""),
            str(row.get("name_class") or ""),
            str(row.get("language") or ""),
            str(row.get("display_name") or ""),
            str(row.get("source_record_id") or ""),
        )
        if key not in unique:
            unique[key] = row
    return list(unique.values())


def _interleave_enrichment_names(
    base_names: list[dict[str, object]],
    enrichment_names: list[dict[str, object]],
) -> list[dict[str, object]]:
    enrichment_by_taxon: dict[str, list[dict[str, object]]] = {}
    for row in enrichment_names:
        enrichment_by_taxon.setdefault(str(row.get("accepted_taxon_key") or ""), []).append(row)
    rows: list[dict[str, object]] = []
    emitted_species: set[str] = set()
    for row in base_names:
        rows.append(row)
        key = str(row.get("accepted_taxon_key") or "")
        if str(row.get("name_class") or "") == "accepted_scientific" and key in enrichment_by_taxon and key not in emitted_species:
            rows.extend(enrichment_by_taxon[key])
            emitted_species.add(key)
    for key, values in enrichment_by_taxon.items():
        if key not in emitted_species:
            rows.extend(values)
    return rows


def _name_schema() -> dict[str, pl.DataType]:
    return {
        "accepted_taxon_key": pl.String,
        "verbatim_name": pl.String,
        "display_name": pl.String,
        "language": pl.String,
        "script": pl.String,
        "region": pl.String,
        "bbox": pl.String,
        "name_class": pl.String,
        "source": pl.String,
        "source_record_id": pl.String,
        "trust_tier": pl.String,
        "precision_tier": pl.String,
        "confidence": pl.String,
        "enabled": pl.Boolean,
    }


def _taxon_row(
    usage: dict[str, Any],
    *,
    parent_key: str,
    family: dict[str, Any],
    genus: dict[str, Any] | None = None,
    species: dict[str, Any] | None = None,
) -> dict[str, str]:
    key = usage.get("key")
    rank = str(usage.get("rank") or "")
    return {
        "accepted_taxon_key": f"gbif:{key}",
        "scientific_name": _scientific_name(usage),
        "rank": rank,
        "parent_key": parent_key,
        "family_key": f"gbif:{family.get('key')}" if family.get("key") else "",
        "family": _scientific_name(family),
        "genus_key": f"gbif:{(genus or {}).get('key')}" if (genus or {}).get("key") else "",
        "genus": _scientific_name(genus or {}),
        "species_key": f"gbif:{(species or {}).get('key')}" if (species or {}).get("key") else "",
        "species": _scientific_name(species or {}) if rank == "SPECIES" else "",
    }


def _scientific_name_row(usage: dict[str, Any], *, name_class: str, accepted_key: object | None = None) -> dict[str, object]:
    key = accepted_key or usage.get("key")
    name = _scientific_name(usage)
    return {
        "accepted_taxon_key": f"gbif:{key}",
        "verbatim_name": name,
        "display_name": name,
        "language": "la",
        "script": "Latn",
        "region": "",
        "bbox": "",
        "name_class": name_class,
        "source": "GBIF",
        "source_record_id": f"gbif:{usage.get('key')}",
        "trust_tier": "T1",
        "precision_tier": "high",
        "confidence": "high",
        "enabled": True,
    }


def _vernacular_name_row(usage: dict[str, Any], *, accepted_key: object) -> dict[str, object]:
    name = str(usage.get("vernacularName") or "")
    return {
        "accepted_taxon_key": f"gbif:{accepted_key}",
        "verbatim_name": name,
        "display_name": name,
        "language": str(usage.get("language") or ""),
        "script": "",
        "region": str(usage.get("country") or ""),
        "bbox": "",
        "name_class": "vernacular",
        "source": "GBIF",
        "source_record_id": f"gbif:vernacular:{accepted_key}:{name}",
        "trust_tier": "T2",
        "precision_tier": "medium",
        "confidence": "medium",
        "enabled": True,
    }


def _scientific_name(usage: dict[str, Any]) -> str:
    return str(usage.get("canonicalName") or usage.get("scientificName") or "")


def _bare_key(value: str) -> str:
    return value.removeprefix("gbif:")
