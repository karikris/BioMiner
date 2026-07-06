from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from biominer.registry.normalize import parse_language_tag, normalize_language_code, normalize_name_key
from biominer.registry.query_eligibility import SCIENTIFIC_NAME_CLASSES, assess_name_query_eligibility
from biominer.registry.scope import load_scope
from biominer.storage.parquet import write_parquet


REGISTRY_SCHEMA_VERSION = "registry-foundation-v1"
COMPILER_VERSION = "registry-compiler-v1"
COLLISION_REVIEW_STATES = {"reviewed", "curator_reviewed", "manual_reviewed", "query_approved"}


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
    frames, manifest = compile_registry_frames(
        payload,
        source_ref=source,
        output_ref=output,
        registry_version=registry_version,
        scope_path=scope_path,
    )

    for filename, frame in frames.items():
        write_parquet(frame, output / filename)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def compile_registry_frames(
    source_payload: dict[str, Any],
    *,
    source_ref: str | Path,
    output_ref: str | Path,
    registry_version: str,
    scope_path: str | Path = "config/butterfly_scope.json",
) -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    scope = load_scope(scope_path)
    taxa = _taxa_frame(source_payload.get("taxa", []), scope_id=scope.scope_id)
    names = _names_frame(source_payload.get("names", []), registry_version=registry_version)
    name_collision_ledger = _name_collision_ledger_frame(names, registry_version=registry_version)
    names = _apply_name_collision_policy(names, name_collision_ledger)
    evidence = _name_evidence_frame(source_payload.get("names", []), registry_version=registry_version, source_payload=source_payload)
    snapshots = _source_snapshots_frame(source_payload, source_ref=source_ref)
    queries = _query_definitions_frame(names, taxa, registry_version=registry_version)
    qa_findings = _qa_findings(taxa, names, queries, scope)
    qa = pl.DataFrame(qa_findings) if qa_findings else _empty_qa_frame()
    frames = {
        "taxa.parquet": taxa,
        "taxon_relations.parquet": _taxon_relations_frame(taxa),
        "names.parquet": names,
        "name_collision_ledger.parquet": name_collision_ledger,
        "name_evidence.parquet": evidence,
        "source_snapshots.parquet": snapshots,
        "flickr_query_definitions.parquet": queries,
        "qa_findings.parquet": qa,
    }
    manifest = _manifest(
        registry_version=registry_version,
        scope_id=scope.scope_id,
        taxa=taxa,
        names=names,
        name_collision_ledger=name_collision_ledger,
        queries=queries,
        qa=qa,
        source_ref=source_ref,
        output_ref=output_ref,
        source_hash=_source_hash(source_payload, source_ref),
    )
    return frames, manifest


def query_definitions_from_names(names: pl.DataFrame, taxa: pl.DataFrame, *, registry_version: str) -> pl.DataFrame:
    """Build Flickr query definitions from names-shaped rows.

    This public wrapper lets enrichment append retrieval-only query rows for
    candidate terms without duplicating the registry query schema.
    """

    return _query_definitions_frame(names, taxa, registry_version=registry_version)


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
    normalized_rows = {}
    for row in rows:
        display_name = str(row.get("display_name") or row.get("verbatim_name") or "")
        accepted_taxon_key = str(row.get("accepted_taxon_key") or "")
        language_tag = parse_language_tag(row.get("language"))
        language = language_tag.language
        script = str(row.get("script") or language_tag.script)
        region = str(row.get("region") or language_tag.region)
        name_id = _stable_id("name", registry_version, accepted_taxon_key, display_name, language, script, region)
        enabled = bool(row.get("enabled", True))
        name_row = {
            "name_id": name_id,
            "registry_version": registry_version,
            "accepted_taxon_key": accepted_taxon_key,
            "verbatim_name": str(row.get("verbatim_name") or display_name),
            "display_name": display_name,
            "normalized_match_key": normalize_name_key(display_name),
            "language": language,
            "script": script,
            "region": region,
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
            "corroborated": _boolish(row.get("corroborated", False)),
        }
        query_decision = assess_name_query_eligibility(name_row)
        normalized_rows.setdefault(
            name_id,
            {
                **name_row,
                "query_eligible": query_decision.query_eligible,
                "query_disabled_reason": query_decision.query_disabled_reason,
                "species_specificity_score": query_decision.species_specificity_score,
            },
        )
    return pl.DataFrame(list(normalized_rows.values()), schema=_names_schema())


def _name_evidence_frame(rows: list[dict[str, Any]], *, registry_version: str, source_payload: dict[str, Any]) -> pl.DataFrame:
    evidence_rows = []
    source_hash = _payload_hash(source_payload)
    for row in rows:
        display_name = str(row.get("display_name") or row.get("verbatim_name") or "")
        accepted_taxon_key = str(row.get("accepted_taxon_key") or "")
        language_tag = parse_language_tag(row.get("language"))
        language = language_tag.language
        script = str(row.get("script") or language_tag.script)
        region = str(row.get("region") or language_tag.region)
        evidence_rows.append(
            {
                "evidence_id": _stable_id("evidence", registry_version, accepted_taxon_key, display_name, row.get("source"), row.get("source_record_id")),
                "name_id": _stable_id("name", registry_version, accepted_taxon_key, display_name, language, script, region),
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


def _source_snapshots_frame(payload: dict[str, Any], *, source_ref: str | Path) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "source": str(payload.get("source") or ""),
                "source_version": str(payload.get("source_version") or ""),
                "retrieved_at": str(payload.get("retrieved_at") or ""),
                "source_path": str(source_ref),
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
    names = _ensure_query_eligibility_columns(names)
    names = _apply_name_collision_policy(names, _name_collision_ledger_frame(names, registry_version=registry_version))
    enabled_names = names.filter(pl.col("enabled") & pl.col("query_eligible"))
    rows: list[dict[str, Any]] = []
    joined = enabled_names.join(taxa_lookup, on="accepted_taxon_key", how="left").to_dicts()
    for item in joined:
        for field in ("tags", "text"):
            priority = _search_priority(item, field)
            rows.append(
                {
                    "query_definition_id": _stable_id(
                        "flickr-query",
                        registry_version,
                        item["accepted_taxon_key"],
                        item["name_id"],
                        field,
                        item["normalized_match_key"],
                        item["language"],
                        item["region"],
                        item["bbox"],
                        item["name_class"],
                        item["source"],
                        item["source_record_id"],
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
                    "query_eligible": item["query_eligible"],
                    "query_disabled_reason": item["query_disabled_reason"],
                    "species_specificity_score": item["species_specificity_score"],
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


def _name_collision_ledger_frame(names: pl.DataFrame, *, registry_version: str) -> pl.DataFrame:
    if names.is_empty():
        return pl.DataFrame([], schema=_name_collision_ledger_schema())
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in names.filter((pl.col("enabled")) & (pl.col("normalized_match_key") != "")).to_dicts():
        key = (str(row.get("normalized_match_key") or ""), str(row.get("language") or ""))
        buckets.setdefault(key, []).append(row)
    rows: list[dict[str, Any]] = []
    for (normalized_match_key, language), bucket in sorted(buckets.items()):
        accepted_taxon_keys = sorted({str(row.get("accepted_taxon_key") or "") for row in bucket if row.get("accepted_taxon_key")})
        if len(accepted_taxon_keys) <= 1:
            continue
        query_blocking = [row for row in bucket if _name_collision_blocks_query(row)]
        rows.append(
            {
                "registry_version": registry_version,
                "normalized_match_key": normalized_match_key,
                "language": language,
                "taxon_count": len(accepted_taxon_keys),
                "enabled_name_count": len(bucket),
                "query_blocking_name_count": len(query_blocking),
                "accepted_taxon_keys": accepted_taxon_keys,
                "name_ids": sorted({str(row.get("name_id") or "") for row in bucket if row.get("name_id")}),
                "display_names": sorted({str(row.get("display_name") or "") for row in bucket if row.get("display_name")}),
                "name_classes": sorted({str(row.get("name_class") or "") for row in bucket if row.get("name_class")}),
                "sources": sorted({str(row.get("source") or "") for row in bucket if row.get("source")}),
                "collision_status": "query_blocking" if query_blocking else "reviewed_or_scientific",
                "query_disabled_reason": "normalized_name_language_collision" if query_blocking else "",
            }
        )
    return pl.DataFrame(rows, schema=_name_collision_ledger_schema())


def _apply_name_collision_policy(names: pl.DataFrame, collision_ledger: pl.DataFrame) -> pl.DataFrame:
    if names.is_empty() or collision_ledger.is_empty():
        return names
    blocking_keys = {
        (str(row["normalized_match_key"]), str(row["language"]))
        for row in collision_ledger.filter(pl.col("collision_status") == "query_blocking").to_dicts()
    }
    if not blocking_keys:
        return names
    rows: list[dict[str, Any]] = []
    for row in names.to_dicts():
        key = (str(row.get("normalized_match_key") or ""), str(row.get("language") or ""))
        if key in blocking_keys and _name_collision_blocks_query(row):
            row = {
                **row,
                "query_eligible": False,
                "query_disabled_reason": "normalized_name_language_collision",
                "species_specificity_score": min(float(row.get("species_specificity_score") or 0.0), 0.45),
            }
        rows.append(row)
    return pl.DataFrame(rows, schema=names.schema)


def _name_collision_blocks_query(row: dict[str, Any]) -> bool:
    if not bool(row.get("enabled")) or not bool(row.get("query_eligible")):
        return False
    if str(row.get("name_class") or "").casefold() in SCIENTIFIC_NAME_CLASSES:
        return False
    review_state = "_".join(str(row.get("review_state") or "").casefold().split())
    precision_tier = str(row.get("precision_tier") or "").casefold()
    if review_state == "query_approved" and precision_tier != "broad":
        return False
    if review_state in COLLISION_REVIEW_STATES and precision_tier == "high":
        return False
    return True


def _qa_findings(taxa: pl.DataFrame, names: pl.DataFrame, queries: pl.DataFrame, scope) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if taxa.filter((pl.col("scientific_name") == scope.root_scientific_name) & (pl.col("rank") == scope.root_rank)).is_empty():
        findings.append(_finding("fatal", "missing_scope_root", scope.root_scientific_name))
    families = set(taxa.filter(pl.col("rank") == "FAMILY").select("scientific_name").to_series().to_list())
    missing = [family for family in scope.included_families if family not in families]
    if missing:
        findings.extend(_finding("fatal", "configured_family_not_in_source", family) for family in missing)
    if names.filter(pl.col("normalized_match_key") == "").height:
        findings.append(_finding("fatal", "empty_normalized_name", "names"))
    if not queries.is_empty():
        duplicate_query_rows = queries.filter(pl.col("query_definition_id").is_duplicated()).height
        if duplicate_query_rows:
            findings.append(_finding("fatal", "duplicate_query_definition_id", str(duplicate_query_rows)))
        missing_lineage = queries.filter(pl.col("accepted_scientific_name") == "").height
        if missing_lineage:
            findings.append(_finding("fatal", "query_without_accepted_taxon_lineage", str(missing_lineage)))
    if not names.is_empty():
        enabled = names.filter(pl.col("enabled"))
        query_ineligible = enabled.filter(~pl.col("query_eligible")) if "query_eligible" in enabled.columns else pl.DataFrame()
        collisions = (
            enabled.group_by("normalized_match_key")
            .agg(pl.col("accepted_taxon_key").n_unique().alias("taxon_count"))
            .filter((pl.col("normalized_match_key") != "") & (pl.col("taxon_count") > 1))
            .select("normalized_match_key")
            .to_series()
            .to_list()
        )
        findings.extend(_finding("warning", "normalized_name_collision", str(key)) for key in sorted(collisions))
        weak_metadata = names.filter(
            (pl.col("enabled"))
            & (pl.col("name_class").is_in(["vernacular", "vernacular_alias"]))
            & ((pl.col("language") == "") | (pl.col("script") == ""))
        )
        findings.extend(
            _finding("warning", "weak_language_or_script_metadata", str(name))
            for name in sorted(set(weak_metadata.select("display_name").to_series().to_list()))
        )
        missing_source = names.filter((pl.col("enabled")) & ((pl.col("source") == "") | (pl.col("source_record_id") == "")))
        findings.extend(
            _finding("warning", "missing_name_source_evidence", str(name))
            for name in sorted(set(missing_source.select("display_name").to_series().to_list()))
        )
        disabled_count = names.filter(~pl.col("enabled")).height
        if disabled_count:
            findings.append(_finding("warning", "disabled_names_excluded_from_queries", str(disabled_count)))
        if not query_ineligible.is_empty():
            findings.append(_finding("warning", "query_ineligible_names_excluded_from_queries", str(query_ineligible.height)))
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
    name_collision_ledger: pl.DataFrame,
    queries: pl.DataFrame,
    qa: pl.DataFrame,
    source_ref: str | Path,
    output_ref: str | Path,
    source_hash: str,
) -> dict[str, Any]:
    fatal_count = qa.filter(pl.col("severity") == "fatal").height if not qa.is_empty() else 0
    warning_count = qa.filter(pl.col("severity") == "warning").height if not qa.is_empty() else 0
    return {
        "registry_version": registry_version,
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "compiler_version": COMPILER_VERSION,
        "scope_id": scope_id,
        "build_time": datetime.now(UTC).isoformat(),
        "source_path": str(source_ref),
        "output_dir": str(output_ref),
        "taxa_rows": taxa.height,
        "name_rows": names.height,
        "query_eligible_name_rows": names.filter(pl.col("query_eligible")).height if "query_eligible" in names.columns else None,
        "query_ineligible_name_rows": names.filter(pl.col("enabled") & ~pl.col("query_eligible")).height if "query_eligible" in names.columns else None,
        "name_collision_ledger_rows": name_collision_ledger.height,
        "query_blocking_name_collision_rows": (
            name_collision_ledger.filter(pl.col("collision_status") == "query_blocking").height if "collision_status" in name_collision_ledger.columns else 0
        ),
        "query_definition_rows": queries.height,
        "qa_finding_rows": qa.height,
        "qa_fatal_count": fatal_count,
        "qa_warning_count": warning_count,
        "qa_status": "failed" if fatal_count else "passed",
        "source_hash": source_hash,
    }


def _source_hash(payload: dict[str, Any], source_ref: str | Path) -> str:
    path = Path(source_ref)
    if path.exists():
        return _file_hash(path)
    return _payload_hash(payload)


def _search_priority(name: dict[str, Any], field: str) -> int:
    field_offset = 0 if field == "tags" else 1
    name_class = str(name.get("name_class") or "")
    language = str(name.get("language") or "").casefold()
    trust_tier = str(name.get("trust_tier") or "").casefold()
    if name_class in {"accepted_scientific", "canonical_scientific"}:
        return 10 + field_offset
    if name_class in {"vernacular", "vernacular_alias"} and language in {"en", "eng"} and trust_tier in {"t1", "t2", "t3"}:
        return 20 + field_offset
    if name_class == "scientific_synonym":
        return 30 + field_offset
    if name_class in {"vernacular", "vernacular_alias"}:
        return 40 + field_offset
    if name_class == "generated_translation" or trust_tier == "t5":
        return 80 + field_offset
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
        "review_state": pl.String,
        "corroborated": pl.Boolean,
        "query_eligible": pl.Boolean,
        "query_disabled_reason": pl.String,
        "species_specificity_score": pl.Float64,
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


def _name_collision_ledger_schema() -> dict[str, pl.DataType]:
    return {
        "registry_version": pl.String,
        "normalized_match_key": pl.String,
        "language": pl.String,
        "taxon_count": pl.Int64,
        "enabled_name_count": pl.Int64,
        "query_blocking_name_count": pl.Int64,
        "accepted_taxon_keys": pl.List(pl.String),
        "name_ids": pl.List(pl.String),
        "display_names": pl.List(pl.String),
        "name_classes": pl.List(pl.String),
        "sources": pl.List(pl.String),
        "collision_status": pl.String,
        "query_disabled_reason": pl.String,
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
        "query_eligible": pl.Boolean,
        "query_disabled_reason": pl.String,
        "species_specificity_score": pl.Float64,
    }


def _ensure_query_eligibility_columns(names: pl.DataFrame) -> pl.DataFrame:
    if names.is_empty():
        return names
    rows: list[dict[str, Any]] = []
    needs_rebuild = not {"query_eligible", "query_disabled_reason", "species_specificity_score"}.issubset(names.columns)
    if not needs_rebuild:
        return names
    for row in names.to_dicts():
        decision = assess_name_query_eligibility(row)
        rows.append(
            {
                **row,
                "query_eligible": decision.query_eligible,
                "query_disabled_reason": decision.query_disabled_reason,
                "species_specificity_score": decision.species_specificity_score,
            }
        )
    return pl.DataFrame(rows)


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "accepted", "enabled", "reviewed", "corroborated"}
