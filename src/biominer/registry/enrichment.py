from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from biominer.registry.compiler import compile_registry_fixture
from biominer.registry.normalize import normalize_name_key


ENRICHMENT_SCHEMA_VERSION = "registry-enrichment-v1"
SOURCE_ASSERTIONS_FILE = "source_name_assertions.parquet"
EXTERNAL_LINKS_FILE = "external_taxon_links.parquet"
ENRICHMENT_SOURCE_SNAPSHOTS_FILE = "enrichment_source_snapshots.parquet"
FINAL_SOURCE_SNAPSHOTS_FILE = "source_snapshots.parquet"
NAME_CANDIDATES_FILE = "name_candidates.parquet"
ENRICHMENT_MANIFEST_FILE = "enrichment_manifest.json"


@dataclass(frozen=True)
class SpeciesContext:
    accepted_taxon_key: str
    accepted_scientific_name: str
    family_key: str
    family: str
    genus_key: str
    genus: str
    current_names: tuple[str, ...]


def build_enrichment_sources_from_registry(
    *,
    registry_dir: str | Path,
    sources: tuple[str, ...] = ("col", "wikidata", "itis"),
    clients: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry = Path(registry_dir)
    clients = clients or default_enrichment_clients()
    taxa = pl.read_parquet(registry / "taxa.parquet")
    names = pl.read_parquet(registry / "names.parquet")
    species_rows = taxa.filter(pl.col("rank") == "SPECIES").sort(["family", "genus", "scientific_name"]).to_dicts()
    names_by_taxon: dict[str, list[str]] = {}
    for row in names.sort(["accepted_taxon_key", "display_name"]).to_dicts():
        names_by_taxon.setdefault(str(row.get("accepted_taxon_key") or ""), []).append(str(row.get("display_name") or ""))

    name_assertions: list[dict[str, Any]] = []
    external_links: list[dict[str, Any]] = []
    source_snapshots: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for species in species_rows:
        context = SpeciesContext(
            accepted_taxon_key=str(species.get("accepted_taxon_key") or ""),
            accepted_scientific_name=str(species.get("scientific_name") or ""),
            family_key=str(species.get("family_key") or ""),
            family=str(species.get("family") or ""),
            genus_key=str(species.get("genus_key") or ""),
            genus=str(species.get("genus") or ""),
            current_names=tuple(names_by_taxon.get(str(species.get("accepted_taxon_key") or ""), [])),
        )
        for source in sources:
            client = clients.get(source)
            if client is None:
                errors.append({"source": source, "accepted_taxon_key": context.accepted_taxon_key, "error": "missing_client"})
                continue
            try:
                result = client.enrich_species(context)
            except Exception as exc:  # noqa: BLE001 - source staging records and continues per species.
                errors.append({"source": source, "accepted_taxon_key": context.accepted_taxon_key, "error": type(exc).__name__})
                continue
            name_assertions.extend(result.get("name_assertions", []))
            external_links.extend(result.get("external_links", []))
            source_snapshots.extend(result.get("source_snapshots", []))

    manifest = write_enrichment_sources(
        registry,
        name_assertions=name_assertions,
        external_links=external_links,
        source_snapshots=_deduplicate_dicts(source_snapshots, keys=("source", "source_version", "source_path", "source_response_hash")),
    )
    manifest.update(
        {
            "registry_dir": str(registry),
            "source_order": list(sources),
            "species_seen": len(species_rows),
            "errors": errors,
        }
    )
    (registry / ENRICHMENT_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def default_enrichment_clients() -> dict[str, Any]:
    from biominer.registry.enrichment_sources import CatalogueOfLifeClient, ITISClient, WikidataClient

    return {
        "col": CatalogueOfLifeClient(),
        "wikidata": WikidataClient(),
        "itis": ITISClient(),
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
    assertions.write_parquet(output / SOURCE_ASSERTIONS_FILE)
    links.write_parquet(output / EXTERNAL_LINKS_FILE)
    snapshots.write_parquet(output / ENRICHMENT_SOURCE_SNAPSHOTS_FILE)
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
    }
    (output / ENRICHMENT_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
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
