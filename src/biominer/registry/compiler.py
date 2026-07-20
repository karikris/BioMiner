from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any
import unicodedata

import polars as pl

from biominer.registry.normalize import parse_language_tag, normalize_language_code, normalize_name_key
from biominer.registry.query_curation import QueryCurationRule, apply_query_curation, load_query_curation_rules
from biominer.registry.query_eligibility import SCIENTIFIC_NAME_CLASSES, assess_name_query_eligibility
from biominer.registry.scope import load_scope
from biominer.registry.unified import (
    COL_XR_DATASET_KEY,
    COL_XR_DOI,
    COL_XR_RELEASE,
    canonical_query_rows,
    canonicalize_keyword_rows,
    collision_metrics,
    compile_species_paths,
)
from biominer.storage.parquet import write_parquet


REGISTRY_SCHEMA_VERSION = "unified-butterfly-registry-v1"
COMPILER_VERSION = "unified-registry-compiler-v1"
COLLISION_REVIEW_STATES = {"reviewed", "curator_reviewed", "manual_reviewed", "query_approved"}


def compile_registry_fixture(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    registry_version: str,
    scope_path: str | Path = "config/butterfly_scope.json",
    global_names_for_collision: pl.DataFrame | None = None,
    query_curation_json: str | Path | None = None,
    query_curation_rules: tuple[QueryCurationRule, ...] = (),
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
        global_names_for_collision=global_names_for_collision,
        query_curation_json=query_curation_json,
        query_curation_rules=query_curation_rules,
    )

    for filename, frame in frames.items():
        write_parquet(frame, output / filename)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def compile_registry_parquet_source(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    registry_version: str,
    scope_path: str | Path = "config/butterfly_scope.json",
    query_curation_json: str | Path | None = None,
) -> dict[str, Any]:
    """Compile a registry from the durable CoL XR Parquet source snapshot."""

    from biominer.registry.checklistbank import col_xr_payload_from_parquet

    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = col_xr_payload_from_parquet(source)
    frames, manifest = compile_registry_frames(
        payload,
        source_ref=source,
        output_ref=output,
        registry_version=registry_version,
        scope_path=scope_path,
        query_curation_json=query_curation_json,
    )
    for filename, frame in frames.items():
        write_parquet(frame, output / filename)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def compile_registry_frames(
    source_payload: dict[str, Any],
    *,
    source_ref: str | Path,
    output_ref: str | Path,
    registry_version: str,
    scope_path: str | Path = "config/butterfly_scope.json",
    global_names_for_collision: pl.DataFrame | None = None,
    query_curation_json: str | Path | None = None,
    query_curation_rules: tuple[QueryCurationRule, ...] = (),
) -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    scope = load_scope(scope_path)
    source_taxa = source_payload.get("taxa", [])
    lineage_taxa = _taxa_frame(source_taxa, scope_id=scope.scope_id, supported_only=False)
    taxa = _taxa_frame(source_taxa, scope_id=scope.scope_id)
    retained_taxon_keys = set(taxa["accepted_taxon_key"].to_list())
    retained_name_rows = [
        row
        for row in source_payload.get("names", [])
        if str(row.get("accepted_taxon_key") or "") in retained_taxon_keys
    ]
    names = _names_frame(retained_name_rows, registry_version=registry_version)
    snapshots = _source_snapshots_frame(source_payload, source_ref=source_ref)
    query_curation = query_curation_rules or load_query_curation_rules(query_curation_json)
    collision_names = _ensure_query_eligibility_columns(global_names_for_collision) if global_names_for_collision is not None else names
    name_collision_ledger = _name_collision_ledger_frame(collision_names, registry_version=registry_version)
    names = apply_query_curation(names, query_curation)
    names = canonicalize_keyword_rows(names)
    evidence = _name_evidence_frame(retained_name_rows, registry_version=registry_version, source_payload=source_payload)
    queries = _query_definitions_frame(names, taxa, registry_version=registry_version, query_curation_rules=query_curation)
    species_paths = compile_species_paths(
        lineage_taxa,
        registry_version=registry_version,
        source_release=str(source_payload.get("source_version") or COL_XR_RELEASE),
    )
    qa_findings = [*_qa_findings(taxa, names, queries, scope), *_species_path_qa(taxa, species_paths)]
    qa = pl.DataFrame(qa_findings) if qa_findings else _empty_qa_frame()
    frames = {
        "taxa.parquet": taxa,
        "species_paths.parquet": species_paths,
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
        species_paths=species_paths,
        names=names,
        name_collision_ledger=name_collision_ledger,
        queries=queries,
        qa=qa,
        source_ref=source_ref,
        output_ref=output_ref,
        source_hash=_source_hash(source_payload, source_ref),
        query_curation_rule_count=len(query_curation),
    )
    return frames, manifest


def _taxa_frame(
    rows: list[dict[str, Any]],
    *,
    scope_id: str,
    supported_only: bool = True,
) -> pl.DataFrame:
    supported_ranks = {"KINGDOM", "PHYLUM", "CLASS", "ORDER", "FAMILY", "GENUS", "SPECIES"}
    normalized_rows = [
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
                "genus_source_release": str(row.get("genus_source_release") or ""),
                "genus_evidence_ids": [str(value) for value in (row.get("genus_evidence_ids") or [])],
                "genus_supersedes_node_id": str(row.get("genus_supersedes_node_id") or ""),
                "species_key": str(row.get("species_key") or ""),
                "species": str(row.get("species") or ""),
                "taxonomic_status": str(row.get("taxonomic_status") or row.get("status") or "ACCEPTED").upper(),
                "source_taxon_id": str(row.get("source_taxon_id") or ""),
                "scientific_name_authorship": str(row.get("scientific_name_authorship") or ""),
                "source_dataset_key": str(row.get("source_dataset_key") or ""),
                "source_release": str(row.get("source_release") or ""),
                "in_scope": True,
            }
            for row in rows
            if not supported_only or str(row.get("rank") or "").upper() in supported_ranks
        ]
    return pl.DataFrame(
        normalized_rows,
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
            "genus_source_release": pl.String,
            "genus_evidence_ids": pl.List(pl.String),
            "genus_supersedes_node_id": pl.String,
            "species_key": pl.String,
            "species": pl.String,
            "taxonomic_status": pl.String,
            "source_taxon_id": pl.String,
            "scientific_name_authorship": pl.String,
            "source_dataset_key": pl.String,
            "source_release": pl.String,
            "in_scope": pl.Boolean,
        },
    )


def _names_frame(rows: list[dict[str, Any]], *, registry_version: str) -> pl.DataFrame:
    normalized_rows = {}
    for row in rows:
        display_name = str(row.get("display_name") or row.get("verbatim_name") or "")
        accepted_taxon_key = str(row.get("accepted_taxon_key") or "")
        language, api_language_code, script, region, bcp47 = _language_fields(row)
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
            "api_language_code": api_language_code,
            "script": script,
            "region": region,
            "bcp47": bcp47,
            "bbox": str(row.get("bbox") or ""),
            "name_class": str(row.get("name_class") or ""),
            "source": str(row.get("source") or ""),
            "source_record_id": str(row.get("source_record_id") or ""),
            "source_taxon_id": str(row.get("source_taxon_id") or ""),
            "lineage_check": str(row.get("lineage_check") or ""),
            "trust_tier": str(row.get("trust_tier") or ""),
            "precision_tier": str(row.get("precision_tier") or ""),
            "confidence": str(row.get("confidence") or ""),
            "enabled": enabled,
            "disabled_reason": str(row.get("disabled_reason") or ""),
            "review_state": str(row.get("review_state") or ("accepted" if enabled else "disabled")),
            "corroborated": _boolish(row.get("corroborated", False)),
            "keyword_id": "",
            "canonical_keyword_id": "",
            "original_trust_tier": "",
            "effective_trust_tier": "",
            "is_canonical_keyword": False,
            "suppressed_duplicate": False,
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
        language, _, script, region, _ = _language_fields(row)
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
                "source_url": str(payload.get("source_url") or ""),
                "citation": str(payload.get("citation") or ""),
            }
        ],
        schema={
            "source": pl.String,
            "source_version": pl.String,
            "retrieved_at": pl.String,
            "source_path": pl.String,
            "source_response_hash": pl.String,
            "licence": pl.String,
            "source_url": pl.String,
            "citation": pl.String,
        },
    )


def _query_definitions_frame(
    names: pl.DataFrame,
    taxa: pl.DataFrame,
    *,
    registry_version: str,
    global_names_for_collision: pl.DataFrame | None = None,
    query_curation_rules: tuple[QueryCurationRule, ...] = (),
) -> pl.DataFrame:
    if names.is_empty():
        return pl.DataFrame([], schema=_query_schema())
    names = _ensure_query_eligibility_columns(names)
    names = apply_query_curation(names, query_curation_rules)
    if "canonical_keyword_id" not in names.columns or names.filter(pl.col("canonical_keyword_id") != "").is_empty():
        names = canonicalize_keyword_rows(names)
    rows = canonical_query_rows(
        names,
        taxa,
        registry_version=registry_version,
        registry_schema_version=REGISTRY_SCHEMA_VERSION,
        compiler_version=COMPILER_VERSION,
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
        query_blocking = [row for row in bucket if _name_collision_blocks_query(row) or _row_has_blocking_collision_reason(row)]
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


def _row_has_blocking_collision_reason(row: dict[str, Any]) -> bool:
    return str(row.get("query_disabled_reason") or "") == "normalized_name_language_collision"


def _qa_findings(taxa: pl.DataFrame, names: pl.DataFrame, queries: pl.DataFrame, scope) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    stored_ranks = {"KINGDOM", "PHYLUM", "CLASS", "ORDER", "FAMILY", "GENUS", "SPECIES"}
    if scope.root_rank in stored_ranks and taxa.filter((pl.col("scientific_name") == scope.root_scientific_name) & (pl.col("rank") == scope.root_rank)).is_empty():
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
            enabled.group_by(["normalized_match_key", "language"])
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
        language_script_mismatch = [
            str(row["display_name"])
            for row in enabled.filter(pl.col("name_class").is_in(["vernacular", "vernacular_alias"])).select(["display_name", "language", "script"]).to_dicts()
            if _has_language_script_mismatch(str(row["display_name"]), str(row["language"] or ""), str(row["script"] or ""))
        ]
        findings.extend(_finding("warning", "language_script_mismatch", str(name)) for name in sorted(set(language_script_mismatch)))
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


def _species_path_qa(taxa: pl.DataFrame, paths: pl.DataFrame) -> list[dict[str, str]]:
    accepted_species = taxa.filter(
        (pl.col("rank") == "SPECIES") & (pl.col("taxonomic_status") == "ACCEPTED")
    )
    findings: list[dict[str, str]] = []
    if paths.height != accepted_species.height or paths["accepted_taxon_key"].n_unique() != paths.height:
        findings.append(_finding("fatal", "species_path_cardinality_mismatch", str(paths.height)))
    incomplete = paths.filter(~pl.col("enabled"))
    if not incomplete.is_empty():
        findings.append(_finding("fatal", "structurally_incomplete_species_paths", str(incomplete.height)))
    return findings


def _finding(severity: str, code: str, subject: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "subject": subject}


CYRILLIC_LANGUAGE_CODES = {"bel", "bul", "kaz", "kir", "mkd", "mon", "rus", "srp", "tgk", "ukr"}
ARABIC_SCRIPT_LANGUAGE_CODES = {"ara", "fas", "pus", "snd", "urd"}
DEVANAGARI_LANGUAGE_CODES = {"hin", "mar", "nep", "san"}
SCRIPT_LANGUAGE_REQUIREMENTS = {
    "Arab": ARABIC_SCRIPT_LANGUAGE_CODES,
    "Beng": {"ben"},
    "Cyrl": CYRILLIC_LANGUAGE_CODES,
    "Deva": DEVANAGARI_LANGUAGE_CODES,
    "Hani": {"jpn", "kor", "lzh", "zho"},
    "Jpan": {"jpn"},
    "Kana": {"jpn"},
    "Taml": {"tam"},
    "Telu": {"tel"},
}


def _has_language_script_mismatch(display_name: str, language: str, script: str) -> bool:
    implied_script = _implied_script(display_name)
    if not implied_script:
        return False
    normalized_language = normalize_language_code(language)
    allowed_languages = SCRIPT_LANGUAGE_REQUIREMENTS.get(implied_script)
    if allowed_languages and normalized_language not in allowed_languages:
        return True
    if not script or implied_script == "Kana":
        return False
    expected_script = implied_script
    return script != expected_script and normalized_language not in (allowed_languages or set())


def _implied_script(value: str) -> str:
    scripts: dict[str, int] = {}
    for character in value:
        if character.isspace() or character in "-'()":
            continue
        script = _script_from_unicode_name(unicodedata.name(character, ""))
        if script:
            scripts[script] = scripts.get(script, 0) + 1
    if not scripts:
        return ""
    if scripts.get("Kana"):
        return "Kana"
    return max(scripts, key=scripts.get)


def _script_from_unicode_name(character_name: str) -> str:
    if "HIRAGANA" in character_name or "KATAKANA" in character_name:
        return "Kana"
    if "CJK UNIFIED" in character_name or "CJK COMPATIBILITY" in character_name:
        return "Hani"
    if "CYRILLIC" in character_name:
        return "Cyrl"
    if "ARABIC" in character_name:
        return "Arab"
    if "BENGALI" in character_name:
        return "Beng"
    if "TAMIL" in character_name:
        return "Taml"
    if "TELUGU" in character_name:
        return "Telu"
    if "DEVANAGARI" in character_name:
        return "Deva"
    return ""


def _empty_qa_frame() -> pl.DataFrame:
    return pl.DataFrame([], schema={"severity": pl.String, "code": pl.String, "subject": pl.String})


def _manifest(
    *,
    registry_version: str,
    scope_id: str,
    taxa: pl.DataFrame,
    species_paths: pl.DataFrame,
    names: pl.DataFrame,
    name_collision_ledger: pl.DataFrame,
    queries: pl.DataFrame,
    qa: pl.DataFrame,
    source_ref: str | Path,
    output_ref: str | Path,
    source_hash: str,
    query_curation_rule_count: int = 0,
) -> dict[str, Any]:
    fatal_count = qa.filter(pl.col("severity") == "fatal").height if not qa.is_empty() else 0
    warning_count = qa.filter(pl.col("severity") == "warning").height if not qa.is_empty() else 0
    keyword_metrics = collision_metrics(names)
    return {
        "registry_version": registry_version,
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "compiler_version": COMPILER_VERSION,
        "scope_id": scope_id,
        "build_time": datetime.now(UTC).isoformat(),
        "source_path": str(source_ref),
        "output_dir": str(output_ref),
        "taxa_rows": taxa.height,
        "species_path_rows": species_paths.height,
        "species_paths_with_proxies": species_paths.filter(
            pl.any_horizontal(
                *(pl.col(f"{rank}_candidate_kind") == "carry_forward_proxy" for rank in ("kingdom", "phylum", "class", "order", "family", "genus", "species"))
            )
        ).height if not species_paths.is_empty() else 0,
        "name_rows": names.height,
        "query_eligible_name_rows": names.filter(pl.col("query_eligible")).height if "query_eligible" in names.columns else None,
        "query_ineligible_name_rows": names.filter(pl.col("enabled") & ~pl.col("query_eligible")).height if "query_eligible" in names.columns else None,
        "name_collision_ledger_rows": name_collision_ledger.height,
        "query_blocking_name_collision_rows": (
            name_collision_ledger.filter(pl.col("collision_status") == "query_blocking").height if "collision_status" in name_collision_ledger.columns else 0
        ),
        "query_definition_rows": queries.height,
        "query_definition_unique_term_count": _query_unique_term_count(queries),
        "query_definition_rows_by_source": _query_rows_by_source(queries),
        "query_definition_unique_term_counts_by_source": _query_unique_term_counts_by_source(queries),
        "query_curation_rule_count": query_curation_rule_count,
        "qa_finding_rows": qa.height,
        "qa_fatal_count": fatal_count,
        "qa_warning_count": warning_count,
        "qa_status": "failed" if fatal_count else "passed",
        "source_hash": source_hash,
        "identity_source": {
            "authority": "Catalogue of Life XR",
            "dataset_key": COL_XR_DATASET_KEY,
            "release": COL_XR_RELEASE,
            "doi": COL_XR_DOI,
        },
        **keyword_metrics,
    }


def _query_unique_term_count(queries: pl.DataFrame) -> int:
    if queries.is_empty() or "normalized_query_term" not in queries.columns:
        return 0
    return int(queries.select("normalized_query_term").n_unique())


def _query_rows_by_source(queries: pl.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if queries.is_empty() or "source" not in queries.columns:
        return counts
    for row in queries.select("source").to_dicts():
        source = str(row.get("source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return {source: counts[source] for source in sorted(counts)}


def _query_unique_term_counts_by_source(queries: pl.DataFrame) -> dict[str, int]:
    buckets: dict[str, set[str]] = {}
    if queries.is_empty() or not {"source", "normalized_query_term"}.issubset(queries.columns):
        return {}
    for row in queries.select(["source", "normalized_query_term"]).to_dicts():
        source = str(row.get("source") or "unknown")
        buckets.setdefault(source, set()).add(str(row.get("normalized_query_term") or ""))
    return {source: len(buckets[source]) for source in sorted(buckets)}


def _source_hash(payload: dict[str, Any], source_ref: str | Path) -> str:
    path = Path(source_ref)
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(path.glob("*.parquet"), key=lambda item: item.name):
            digest.update(child.name.encode("utf-8"))
            with child.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    if path.exists():
        return _file_hash(path)
    return _payload_hash(payload)


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
        "api_language_code": pl.String,
        "script": pl.String,
        "region": pl.String,
        "bcp47": pl.String,
        "bbox": pl.String,
        "name_class": pl.String,
        "source": pl.String,
        "source_record_id": pl.String,
        "source_taxon_id": pl.String,
        "lineage_check": pl.String,
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
        "keyword_id": pl.String,
        "canonical_keyword_id": pl.String,
        "original_trust_tier": pl.String,
        "effective_trust_tier": pl.String,
        "is_canonical_keyword": pl.Boolean,
        "suppressed_duplicate": pl.Boolean,
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
        "logical_query_id": pl.String,
        "canonical_keyword_id": pl.String,
        "keyword_id": pl.String,
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
        "api_language_code": pl.String,
        "script": pl.String,
        "region": pl.String,
        "bcp47": pl.String,
        "bbox": pl.String,
        "name_class": pl.String,
        "source": pl.String,
        "source_taxon_id": pl.String,
        "lineage_check": pl.String,
        "trust_tier": pl.String,
        "original_trust_tier": pl.String,
        "effective_trust_tier": pl.String,
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
    for row in names.to_dicts():
        if _row_has_blocking_collision_reason(row):
            rows.append(row)
            continue
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


def _language_fields(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    language_tag = parse_language_tag(row.get("bcp47") or row.get("language"))
    language = language_tag.language
    api_language_code = str(row.get("api_language_code") or language_tag.api_language_code)
    script = str(row.get("script") or language_tag.script)
    region = str(row.get("region") or language_tag.region)
    bcp47 = str(row.get("bcp47") or _format_bcp47(api_language_code, script, region))
    return language, api_language_code, script, region, bcp47


def _format_bcp47(api_language_code: str, script: str, region: str) -> str:
    if not api_language_code:
        return ""
    default_script = parse_language_tag(api_language_code).script
    parts = [api_language_code]
    if script and script != default_script:
        parts.append(script)
    if region:
        parts.append(region)
    return "-".join(parts)


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "accepted", "enabled", "reviewed", "corroborated"}
