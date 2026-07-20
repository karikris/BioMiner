from __future__ import annotations

import hashlib
import json
from typing import Any

import polars as pl


COL_XR_DATASET_KEY = "315557"
COL_XR_RELEASE = "COL26.6 XR"
COL_XR_DOI = "10.48580/dgy8b"

TRUST_TIERS = ("T1", "T2", "T3", "T4", "T5")
PATH_RANKS = ("KINGDOM", "PHYLUM", "CLASS", "ORDER", "FAMILY", "GENUS", "SPECIES")

_SOURCE_PRECEDENCE = {
    "col xr": 0,
    "col": 1,
    "gbif": 2,
    "ncbi": 10,
    "open tree": 11,
    "itis": 12,
    "eol": 13,
    "specialist authority": 14,
    "reviewed curated": 15,
    "wikisspecies": 20,
    "wikispecies": 20,
    "wikidata": 21,
    "inaturalist": 22,
    "bold": 23,
    "corroborated checklist": 24,
    "community": 30,
    "dictionary": 40,
    "machine translation": 41,
}
_NAME_CLASS_PRECEDENCE = {
    "accepted_scientific": 0,
    "canonical_scientific": 0,
    "scientific_synonym": 1,
    "vernacular": 2,
    "reviewed_vernacular": 2,
    "vernacular_alias": 3,
    "alias": 3,
    "homonym": 4,
    "spelling_variant": 4,
    "generated_translation": 5,
}


def stable_identity(namespace: str, *parts: object) -> str:
    payload = json.dumps([namespace, *(str(part or "") for part in parts)], ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_trust_tier(value: object, *, source: object = "", name_class: object = "") -> str:
    tier = str(value or "").strip().upper()
    if tier in TRUST_TIERS:
        return tier
    source_key = str(source or "").strip().casefold()
    class_key = str(name_class or "").strip().casefold()
    if class_key == "generated_translation" or any(token in source_key for token in ("translation", "dictionary", "mymemory")):
        return "T5"
    if source_key in {"col", "col xr", "gbif"}:
        return "T1"
    if source_key in {
        "ncbi",
        "open tree",
        "opentree",
        "itis",
        "eol",
        "specialist authority",
        "reviewed curated",
    }:
        return "T2"
    if source_key in {
        "wikispecies",
        "wikidata",
        "inaturalist",
        "bold",
        "corroborated checklist",
    }:
        return "T3"
    return "T4"


def tier_number(value: object) -> int:
    tier = normalize_trust_tier(value)
    return int(tier[1:])


def canonicalize_keyword_rows(names: pl.DataFrame) -> pl.DataFrame:
    """Retain every name association while selecting one executable term identity.

    Canonical identity is deliberately independent of registry version, source,
    language, and taxon. Those are association attributes, not Flickr request
    identity.
    """

    if names.is_empty():
        schema = dict(names.schema)
        schema.update(_keyword_columns_schema())
        return pl.DataFrame(schema=schema)

    rows: list[dict[str, Any]] = []
    for source_row in names.iter_rows(named=True):
        row = dict(source_row)
        normalized = str(row.get("normalized_match_key") or "").strip()
        original_tier = normalize_trust_tier(
            row.get("trust_tier"), source=row.get("source"), name_class=row.get("name_class")
        )
        keyword_id = stable_identity(
            "keyword-association",
            row.get("accepted_taxon_key"),
            normalized,
            row.get("language"),
            row.get("name_class"),
            row.get("source"),
            row.get("source_record_id"),
        )
        rows.append(
            {
                **row,
                "keyword_id": keyword_id,
                "canonical_keyword_id": stable_identity("canonical-keyword", normalized),
                "original_trust_tier": original_tier,
                "effective_trust_tier": original_tier,
                "is_canonical_keyword": False,
                "suppressed_duplicate": False,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("normalized_match_key") or ""), []).append(row)
    for normalized, bucket in grouped.items():
        actionable = [
            row
            for row in bucket
            if normalized and bool(row.get("enabled", True)) and bool(row.get("query_eligible", True))
        ]
        tier_source = actionable or [row for row in bucket if normalized and bool(row.get("enabled", True))] or bucket
        effective = min((str(row["original_trust_tier"]) for row in tier_source), key=tier_number)
        for row in bucket:
            row["effective_trust_tier"] = effective
        if actionable:
            canonical = min(actionable, key=_canonical_keyword_sort_key)
            canonical["is_canonical_keyword"] = True
            for row in bucket:
                row["suppressed_duplicate"] = row is not canonical
        elif bucket:
            for row in bucket:
                row["suppressed_duplicate"] = len(bucket) > 1
    return pl.DataFrame(rows).cast({**dict(names.schema), **_keyword_columns_schema()})


def canonical_query_rows(
    names: pl.DataFrame,
    taxa: pl.DataFrame,
    *,
    registry_version: str,
    registry_schema_version: str,
    compiler_version: str,
) -> list[dict[str, Any]]:
    if names.is_empty():
        return []
    taxa_by_key = {str(row["accepted_taxon_key"]): dict(row) for row in taxa.iter_rows(named=True)}
    rows: list[dict[str, Any]] = []
    canonical = names.filter(pl.col("is_canonical_keyword") & pl.col("enabled") & pl.col("query_eligible"))
    for keyword in canonical.iter_rows(named=True):
        item = dict(keyword)
        taxon = taxa_by_key.get(str(item.get("accepted_taxon_key") or ""), {})
        for field in ("tags", "text"):
            logical_query_id = stable_identity("flickr-logical-query", item["normalized_match_key"], field)
            rows.append(
                {
                    "query_definition_id": logical_query_id,
                    "logical_query_id": logical_query_id,
                    "canonical_keyword_id": item["canonical_keyword_id"],
                    "keyword_id": item["keyword_id"],
                    "registry_schema_version": registry_schema_version,
                    "compiler_version": compiler_version,
                    "registry_version": registry_version,
                    "accepted_taxon_key": item.get("accepted_taxon_key") or "",
                    "accepted_scientific_name": taxon.get("scientific_name") or "",
                    "accepted_rank": taxon.get("rank") or "",
                    "family_key": taxon.get("family_key") or "",
                    "family": taxon.get("family") or "",
                    "genus_key": taxon.get("genus_key") or "",
                    "genus": taxon.get("genus") or "",
                    "species_key": taxon.get("species_key") or "",
                    "species": taxon.get("species") or "",
                    "name_id": item.get("name_id") or "",
                    "source_term": item.get("display_name") or "",
                    "normalized_query_term": item.get("normalized_match_key") or "",
                    "normalized_match_key": item.get("normalized_match_key") or "",
                    "language": item.get("language") or "",
                    "api_language_code": item.get("api_language_code") or "",
                    "script": item.get("script") or "",
                    "region": item.get("region") or "",
                    "bcp47": item.get("bcp47") or "",
                    "bbox": item.get("bbox") or "",
                    "name_class": item.get("name_class") or "",
                    "source": item.get("source") or "",
                    "source_taxon_id": item.get("source_taxon_id") or "",
                    "lineage_check": item.get("lineage_check") or "",
                    "trust_tier": item["effective_trust_tier"],
                    "original_trust_tier": item["original_trust_tier"],
                    "effective_trust_tier": item["effective_trust_tier"],
                    "confidence": item.get("confidence") or "",
                    "precision_tier": item.get("precision_tier") or "",
                    "search_field": field,
                    "search_priority": global_query_priority(
                        item["effective_trust_tier"], item.get("name_class"), field
                    ),
                    "enabled": True,
                    "disabled_reason": "",
                    "query_eligible": True,
                    "query_disabled_reason": "",
                    "species_specificity_score": float(item.get("species_specificity_score") or 0.0),
                }
            )
    return sorted(rows, key=lambda row: (row["search_priority"], row["normalized_match_key"], row["logical_query_id"]))


def global_query_priority(tier: object, name_class: object, field: object) -> int:
    # Tier is the dominant component: no lower tier can sort ahead of a higher one.
    tier_offset = (tier_number(tier) - 1) * 100
    class_offset = _NAME_CLASS_PRECEDENCE.get(str(name_class or "").casefold(), 5) * 10
    field_offset = 0 if str(field) == "tags" else 1
    return tier_offset + class_offset + field_offset


def compile_species_paths(
    taxa: pl.DataFrame,
    *,
    registry_version: str,
    source_release: str = COL_XR_RELEASE,
) -> pl.DataFrame:
    """Compile exactly one complete routing path for every accepted species.

    Missing intermediate ranks are represented by parent carry-forward proxy
    nodes. A proxy is routing evidence and retains its observed semantic rank.
    """

    schema = species_path_schema()
    if taxa.is_empty():
        return pl.DataFrame(schema=schema)
    by_id = {str(row["accepted_taxon_key"]): dict(row) for row in taxa.iter_rows(named=True)}
    species_rows = [
        dict(row)
        for row in taxa.iter_rows(named=True)
        if str(row.get("rank") or "").upper() == "SPECIES"
        and str(row.get("taxonomic_status") or "ACCEPTED").upper() == "ACCEPTED"
    ]
    compiled: list[dict[str, Any]] = []
    for species in species_rows:
        observed = _observed_lineage(species, by_id)
        previous: dict[str, str] | None = None
        row: dict[str, Any] = {
            "registry_schema_version": "unified-butterfly-registry-v1",
            "registry_version": registry_version,
            "accepted_taxon_id": str(species.get("accepted_taxon_key") or ""),
            "accepted_taxon_key": str(species.get("accepted_taxon_key") or ""),
            "accepted_scientific_name": str(species.get("scientific_name") or species.get("species") or ""),
            "source_release": source_release,
            "enabled": True,
            "disabled_reason": "",
        }
        for rank in PATH_RANKS:
            prefix = rank.casefold()
            node = observed.get(rank)
            if node:
                node_id = str(node["node_id"])
                name = str(node["name"])
                semantic_rank = rank
                candidate_kind = "observed_taxon"
                proxy_source = ""
                previous = node
            elif previous:
                node_id = stable_identity("carry-forward-proxy", previous["node_id"], rank)
                name = str(previous["name"])
                semantic_rank = str(previous["rank"])
                candidate_kind = "carry_forward_proxy"
                proxy_source = str(previous["node_id"])
            else:
                node_id = ""
                name = ""
                semantic_rank = ""
                candidate_kind = "missing"
                proxy_source = ""
                row["enabled"] = False
                row["disabled_reason"] = "missing_supported_root_ancestor"
            row[f"{prefix}_node_id"] = node_id
            row[prefix] = name
            row[f"{prefix}_semantic_rank"] = semantic_rank
            row[f"{prefix}_candidate_kind"] = candidate_kind
            row[f"{prefix}_proxy_source_node_id"] = proxy_source
            row[f"{prefix}_supersedes_node_id"] = (
                str(node.get("supersedes_node_id") or "") if node else ""
            )
            row[f"{prefix}_source_release"] = (
                str(node.get("source_release") or source_release) if node else source_release
            )
            row[f"{prefix}_evidence_ids"] = (
                [str(value) for value in (node.get("evidence_ids") or [node_id])]
                if node
                else [proxy_source]
            )
        identity = {key: row[key] for key in row if key.endswith("_node_id") or key in {"accepted_taxon_key", "source_release"}}
        row["path_fingerprint"] = stable_identity("species-path", json.dumps(identity, sort_keys=True))
        compiled.append(row)
    return pl.DataFrame(compiled, schema=schema).sort(["accepted_scientific_name", "accepted_taxon_key"])


def species_path_schema() -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        "registry_schema_version": pl.String,
        "registry_version": pl.String,
        "accepted_taxon_id": pl.String,
        "accepted_taxon_key": pl.String,
        "accepted_scientific_name": pl.String,
        "source_release": pl.String,
        "path_fingerprint": pl.String,
        "enabled": pl.Boolean,
        "disabled_reason": pl.String,
    }
    for rank in PATH_RANKS:
        prefix = rank.casefold()
        schema.update(
            {
                f"{prefix}_node_id": pl.String,
                prefix: pl.String,
                f"{prefix}_semantic_rank": pl.String,
                f"{prefix}_candidate_kind": pl.String,
                f"{prefix}_proxy_source_node_id": pl.String,
                f"{prefix}_supersedes_node_id": pl.String,
                f"{prefix}_source_release": pl.String,
                f"{prefix}_evidence_ids": pl.List(pl.String),
            }
        )
    return schema


def _observed_lineage(species: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    lineage: list[dict[str, Any]] = []
    current = species
    visited: set[str] = set()
    while current:
        key = str(current.get("accepted_taxon_key") or "")
        if not key or key in visited:
            break
        visited.add(key)
        lineage.append(current)
        current = by_id.get(str(current.get("parent_key") or ""), {})
    observed: dict[str, dict[str, str]] = {}
    for taxon in reversed(lineage):
        rank = str(taxon.get("rank") or "").upper()
        if rank in PATH_RANKS:
            observed[rank] = {
                "node_id": str(taxon.get("accepted_taxon_key") or ""),
                "name": str(taxon.get("scientific_name") or taxon.get(rank.casefold()) or ""),
                "rank": rank,
            }
    if not any(rank in observed for rank in ("KINGDOM", "PHYLUM", "CLASS", "ORDER")):
        # Butterfly-scope snapshots can begin at Papilionoidea. The four stable
        # enclosing ranks are explicit in species_paths when the pinned CoL XR
        # archive does not supply their source nodes directly.
        observed.update(
            {
                "KINGDOM": {"node_id": "col-xr:animalia", "name": "Animalia", "rank": "KINGDOM"},
                "PHYLUM": {"node_id": "col-xr:arthropoda", "name": "Arthropoda", "rank": "PHYLUM"},
                "CLASS": {"node_id": "col-xr:insecta", "name": "Insecta", "rank": "CLASS"},
                "ORDER": {"node_id": "col-xr:lepidoptera", "name": "Lepidoptera", "rank": "ORDER"},
            }
        )
    # Source snapshots often flatten family/genus fields without emitting each
    # ancestor row. Preserve those observed identities instead of proxying them.
    for rank in ("FAMILY", "GENUS", "SPECIES"):
        prefix = rank.casefold()
        key = str(species.get(f"{prefix}_key") or (species.get("accepted_taxon_key") if rank == "SPECIES" else ""))
        name = str(species.get(prefix) or (species.get("scientific_name") if rank == "SPECIES" else ""))
        if key and name:
            observed.setdefault(
                rank,
                {
                    "node_id": key,
                    "name": name,
                    "rank": rank,
                    "source_release": str(species.get(f"{prefix}_source_release") or ""),
                    "evidence_ids": [
                        str(value) for value in (species.get(f"{prefix}_evidence_ids") or [key])
                    ],
                    "supersedes_node_id": str(species.get(f"{prefix}_supersedes_node_id") or ""),
                },
            )
    return observed


def _canonical_keyword_sort_key(row: dict[str, Any]) -> tuple[object, ...]:
    return (
        tier_number(row.get("original_trust_tier")),
        _SOURCE_PRECEDENCE.get(str(row.get("source") or "").casefold(), 99),
        _NAME_CLASS_PRECEDENCE.get(str(row.get("name_class") or "").casefold(), 99),
        str(row.get("keyword_id") or ""),
    )


def _keyword_columns_schema() -> dict[str, pl.DataType]:
    return {
        "keyword_id": pl.String,
        "canonical_keyword_id": pl.String,
        "original_trust_tier": pl.String,
        "effective_trust_tier": pl.String,
        "is_canonical_keyword": pl.Boolean,
        "suppressed_duplicate": pl.Boolean,
    }


def collision_metrics(names: pl.DataFrame) -> dict[str, int | dict[str, int]]:
    if names.is_empty() or "canonical_keyword_id" not in names.columns:
        return {
            "duplicate_keyword_rows_suppressed": 0,
            "cross_species_collisions": 0,
            "cross_tier_collisions": 0,
            "unique_normalized_terms_by_tier": {},
        }
    enabled = names.filter(pl.col("enabled"))
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in enabled.iter_rows(named=True):
        buckets.setdefault(str(row["canonical_keyword_id"]), []).append(dict(row))
    by_tier: dict[str, set[str]] = {tier: set() for tier in TRUST_TIERS}
    cross_species = 0
    cross_tier = 0
    for canonical_id, bucket in buckets.items():
        effective = str(bucket[0]["effective_trust_tier"])
        by_tier[effective].add(canonical_id)
        if len({str(row.get("accepted_taxon_key") or "") for row in bucket}) > 1:
            cross_species += 1
        if len({str(row.get("original_trust_tier") or "") for row in bucket}) > 1:
            cross_tier += 1
    return {
        "duplicate_keyword_rows_suppressed": names.filter(pl.col("suppressed_duplicate")).height,
        "cross_species_collisions": cross_species,
        "cross_tier_collisions": cross_tier,
        "unique_normalized_terms_by_tier": {tier: len(by_tier[tier]) for tier in TRUST_TIERS},
    }
