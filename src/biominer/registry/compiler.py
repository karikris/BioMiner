from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from biominer.registry.normalize import normalize_name_key
from biominer.registry.scope import load_scope
from biominer.storage.parquet import write_parquet


REGISTRY_SCHEMA_VERSION = "registry-foundation-v1"
COMPILER_VERSION = "registry-compiler-v1"


def compile_registry_fixture(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    registry_version: str,
    scope_path: str | Path = "config/butterfly_scope.json",
) -> dict[str, Any]:
    source = Path(source_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    scope = load_scope(scope_path)

    taxa = _taxa_frame(payload.get("taxa", []), scope_id=scope.scope_id)
    names = _names_frame(payload.get("names", []), registry_version=registry_version)
    evidence = _name_evidence_frame(payload.get("names", []), registry_version=registry_version, source_payload=payload)
    snapshots = _source_snapshots_frame(payload, source_path=source)
    queries = _query_definitions_frame(names, taxa, registry_version=registry_version)
    qa_findings = _qa_findings(taxa, names, scope)
    qa = pl.DataFrame(qa_findings) if qa_findings else _empty_qa_frame()
    manifest = _manifest(
        registry_version=registry_version,
        scope_id=scope.scope_id,
        taxa=taxa,
        names=names,
        queries=queries,
        qa=qa,
        source_path=source,
        output_dir=output,
    )

    write_parquet(taxa, output / "taxa.parquet")
    write_parquet(_taxon_relations_frame(taxa), output / "taxon_relations.parquet")
    write_parquet(names, output / "names.parquet")
    write_parquet(evidence, output / "name_evidence.parquet")
    write_parquet(snapshots, output / "source_snapshots.parquet")
    write_parquet(queries, output / "flickr_query_definitions.parquet")
    write_parquet(qa, output / "qa_findings.parquet")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _taxa_frame(rows: list[dict[str, Any]], *, scope_id: str) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "registry_schema_version": REGISTRY_SCHEMA_VERSION,
                "scope_id": scope_id,
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
                "in_scope": True,
            }
            for row in rows
        ],
        schema={
            "registry_schema_version": pl.String,
            "scope_id": pl.String,
            "accepted_taxon_key": pl.String,
            "scientific_name": pl.String,
            "rank": pl.String,
            "parent_key": pl.String,
            "family_key": pl.String,
            "family": pl.String,
            "genus_key": pl.String,
            "genus": pl.String,
            "species_key": pl.String,
            "species": pl.String,
            "in_scope": pl.Boolean,
        },
    )


def _names_frame(rows: list[dict[str, Any]], *, registry_version: str) -> pl.DataFrame:
    normalized_rows = []
    for row in rows:
        display_name = str(row.get("display_name") or row.get("verbatim_name") or "")
        accepted_taxon_key = str(row.get("accepted_taxon_key") or "")
        normalized_rows.append(
            {
                "name_id": _stable_id("name", registry_version, accepted_taxon_key, display_name, row.get("language"), row.get("region")),
                "registry_version": registry_version,
                "accepted_taxon_key": accepted_taxon_key,
                "verbatim_name": str(row.get("verbatim_name") or display_name),
                "display_name": display_name,
                "normalized_match_key": normalize_name_key(display_name),
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
            }
        )
    return pl.DataFrame(normalized_rows, schema=_names_schema())


def _name_evidence_frame(rows: list[dict[str, Any]], *, registry_version: str, source_payload: dict[str, Any]) -> pl.DataFrame:
    evidence_rows = []
    source_hash = _payload_hash(source_payload)
    for row in rows:
        display_name = str(row.get("display_name") or row.get("verbatim_name") or "")
        accepted_taxon_key = str(row.get("accepted_taxon_key") or "")
        evidence_rows.append(
            {
                "evidence_id": _stable_id("evidence", registry_version, accepted_taxon_key, display_name, row.get("source"), row.get("source_record_id")),
                "name_id": _stable_id("name", registry_version, accepted_taxon_key, display_name, row.get("language"), row.get("region")),
                "registry_version": registry_version,
                "accepted_taxon_key": accepted_taxon_key,
                "source": str(row.get("source") or ""),
                "source_record_id": str(row.get("source_record_id") or ""),
                "source_response_hash": source_hash,
                "retrieved_at": str(source_payload.get("retrieved_at") or ""),
                "licence": str(row.get("licence") or ""),
                "trust_tier": str(row.get("trust_tier") or ""),
                "review_state": "accepted" if row.get("enabled", True) else "disabled",
            }
        )
    return pl.DataFrame(evidence_rows, schema=_evidence_schema())


def _source_snapshots_frame(payload: dict[str, Any], *, source_path: Path) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "source": str(payload.get("source") or ""),
                "source_version": str(payload.get("source_version") or ""),
                "retrieved_at": str(payload.get("retrieved_at") or ""),
                "source_path": str(source_path),
                "source_response_hash": _payload_hash(payload),
                "licence": str(payload.get("licence") or ""),
            }
        ],
        schema={
            "source": pl.String,
            "source_version": pl.String,
            "retrieved_at": pl.String,
            "source_path": pl.String,
            "source_response_hash": pl.String,
            "licence": pl.String,
        },
    )


def _query_definitions_frame(names: pl.DataFrame, taxa: pl.DataFrame, *, registry_version: str) -> pl.DataFrame:
    if names.is_empty():
        return pl.DataFrame([], schema=_query_schema())
    taxa_lookup = taxa.select(
        "accepted_taxon_key",
        "scientific_name",
        "rank",
        "family_key",
        "family",
        "genus_key",
        "genus",
        "species_key",
        "species",
    )
    enabled_names = names.filter(pl.col("enabled"))
    rows: list[dict[str, Any]] = []
    joined = enabled_names.join(taxa_lookup, on="accepted_taxon_key", how="left").to_dicts()
    for item in joined:
        for field in ("tags", "text"):
            priority = _search_priority(str(item["name_class"]), field)
            rows.append(
                {
                    "query_definition_id": _stable_id(
                        "flickr-query",
                        registry_version,
                        item["name_id"],
                        field,
                        item["region"],
                        item["bbox"],
                    ),
                    "registry_schema_version": REGISTRY_SCHEMA_VERSION,
                    "compiler_version": COMPILER_VERSION,
                    "registry_version": registry_version,
                    "accepted_taxon_key": item["accepted_taxon_key"],
                    "accepted_scientific_name": item.get("scientific_name") or "",
                    "accepted_rank": item.get("rank") or "",
                    "family_key": item.get("family_key") or "",
                    "family": item.get("family") or "",
                    "genus_key": item.get("genus_key") or "",
                    "genus": item.get("genus") or "",
                    "species_key": item.get("species_key") or "",
                    "species": item.get("species") or "",
                    "name_id": item["name_id"],
                    "source_term": item["display_name"],
                    "normalized_query_term": item["display_name"],
                    "normalized_match_key": item["normalized_match_key"],
                    "language": item["language"],
                    "script": item["script"],
                    "region": item["region"],
                    "bbox": item["bbox"],
                    "name_class": item["name_class"],
                    "source": item["source"],
                    "trust_tier": item["trust_tier"],
                    "confidence": item["confidence"],
                    "precision_tier": item["precision_tier"],
                    "search_field": field,
                    "search_priority": priority,
                    "enabled": True,
                    "disabled_reason": "",
                }
            )
    return pl.DataFrame(rows, schema=_query_schema()).sort(["search_priority", "normalized_match_key", "query_definition_id"])


def _taxon_relations_frame(taxa: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for item in taxa.to_dicts():
        parent = str(item.get("parent_key") or "")
        if parent:
            rows.append(
                {
                    "accepted_taxon_key": item["accepted_taxon_key"],
                    "related_taxon_key": parent,
                    "relation_type": "parent",
                }
            )
    return pl.DataFrame(rows, schema={"accepted_taxon_key": pl.String, "related_taxon_key": pl.String, "relation_type": pl.String})


def _qa_findings(taxa: pl.DataFrame, names: pl.DataFrame, scope) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if taxa.filter((pl.col("scientific_name") == scope.root_scientific_name) & (pl.col("rank") == scope.root_rank)).is_empty():
        findings.append(_finding("fatal", "missing_scope_root", scope.root_scientific_name))
    families = set(taxa.filter(pl.col("rank") == "FAMILY").select("scientific_name").to_series().to_list())
    missing = [family for family in scope.included_families if family not in families]
    if missing:
        findings.append(_finding("warning", "configured_family_not_in_source", ",".join(missing)))
    if names.filter(pl.col("normalized_match_key") == "").height:
        findings.append(_finding("fatal", "empty_normalized_name", "names"))
    return findings


def _finding(severity: str, code: str, subject: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "subject": subject}


def _empty_qa_frame() -> pl.DataFrame:
    return pl.DataFrame([], schema={"severity": pl.String, "code": pl.String, "subject": pl.String})


def _manifest(
    *,
    registry_version: str,
    scope_id: str,
    taxa: pl.DataFrame,
    names: pl.DataFrame,
    queries: pl.DataFrame,
    qa: pl.DataFrame,
    source_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    fatal_count = qa.filter(pl.col("severity") == "fatal").height if not qa.is_empty() else 0
    return {
        "registry_version": registry_version,
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "compiler_version": COMPILER_VERSION,
        "scope_id": scope_id,
        "build_time": datetime.now(UTC).isoformat(),
        "source_path": str(source_path),
        "output_dir": str(output_dir),
        "taxa_rows": taxa.height,
        "name_rows": names.height,
        "query_definition_rows": queries.height,
        "qa_finding_rows": qa.height,
        "qa_status": "failed" if fatal_count else "passed",
        "source_hash": _file_hash(source_path),
    }


def _search_priority(name_class: str, field: str) -> int:
    field_offset = 0 if field == "tags" else 40
    if name_class in {"accepted_scientific", "canonical_scientific", "scientific_synonym"}:
        return 10 + field_offset
    if name_class in {"vernacular", "vernacular_alias"}:
        return 20 + field_offset
    return 100 + field_offset


def _stable_id(*parts: object) -> str:
    payload = json.dumps([str(part or "") for part in parts], ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _payload_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _names_schema() -> dict[str, pl.DataType]:
    return {
        "name_id": pl.String,
        "registry_version": pl.String,
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
        "trust_tier": pl.String,
        "precision_tier": pl.String,
        "confidence": pl.String,
        "enabled": pl.Boolean,
        "disabled_reason": pl.String,
    }


def _evidence_schema() -> dict[str, pl.DataType]:
    return {
        "evidence_id": pl.String,
        "name_id": pl.String,
        "registry_version": pl.String,
        "accepted_taxon_key": pl.String,
        "source": pl.String,
        "source_record_id": pl.String,
        "source_response_hash": pl.String,
        "retrieved_at": pl.String,
        "licence": pl.String,
        "trust_tier": pl.String,
        "review_state": pl.String,
    }


def _query_schema() -> dict[str, pl.DataType]:
    return {
        "query_definition_id": pl.String,
        "registry_schema_version": pl.String,
        "compiler_version": pl.String,
        "registry_version": pl.String,
        "accepted_taxon_key": pl.String,
        "accepted_scientific_name": pl.String,
        "accepted_rank": pl.String,
        "family_key": pl.String,
        "family": pl.String,
        "genus_key": pl.String,
        "genus": pl.String,
        "species_key": pl.String,
        "species": pl.String,
        "name_id": pl.String,
        "source_term": pl.String,
        "normalized_query_term": pl.String,
        "normalized_match_key": pl.String,
        "language": pl.String,
        "script": pl.String,
        "region": pl.String,
        "bbox": pl.String,
        "name_class": pl.String,
        "source": pl.String,
        "trust_tier": pl.String,
        "confidence": pl.String,
        "precision_tier": pl.String,
        "search_field": pl.String,
        "search_priority": pl.Int64,
        "enabled": pl.Boolean,
        "disabled_reason": pl.String,
    }
