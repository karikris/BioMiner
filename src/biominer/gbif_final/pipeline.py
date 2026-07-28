from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import duckdb
import polars as pl
import pyarrow.parquet as pq

from biominer.registry.compiler import COMPILER_VERSION, REGISTRY_SCHEMA_VERSION
from biominer.registry.unified import canonical_query_rows, canonicalize_keyword_rows


FINAL_SCHEMA_VERSION = "gbif-media-final-enriched-v1"
FINAL_FILENAME = "gbif_media_final_enriched.parquet"
MANIFEST_FILENAME = "manifest.json"

_EVIDENCE_FIELDS = (
    "name_id",
    "accepted_taxon_key",
    "display_name",
    "normalized_match_key",
    "language",
    "bcp47",
    "name_class",
    "source",
    "source_record_id",
    "source_taxon_id",
    "lineage_check",
    "original_trust_tier",
    "effective_trust_tier",
    "confidence",
    "enabled",
    "disabled_reason",
    "review_state",
    "query_eligible",
    "query_disabled_reason",
    "is_canonical_keyword",
    "suppressed_duplicate",
    "keyword_owner_taxon_key",
    "keyword_owner_rank",
    "keyword_ownership_basis",
)
_ASSERTION_FIELDS = (
    "assertion_id",
    "accepted_taxon_key",
    "display_name",
    "normalized_match_key",
    "language",
    "bcp47",
    "name_class",
    "source",
    "source_record_id",
    "source_taxon_id",
    "lineage_check",
    "trust_tier",
    "confidence",
    "enabled",
    "review_state",
    "disabled_reason",
    "retrieved_at",
    "licence",
)
_QUERY_FIELDS = (
    "query_definition_id",
    "normalized_query_term",
    "source_term",
    "language",
    "bcp47",
    "name_class",
    "source",
    "source_taxon_id",
    "trust_tier",
    "search_field",
    "search_priority",
    "keyword_owner_taxon_key",
    "keyword_owner_rank",
    "keyword_ownership_basis",
    "query_stage",
    "query_stage_order",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _normalized(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _records_by_key(frame: pl.DataFrame, key: str, fields: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    available = [field for field in fields if field in frame.columns and field != key]
    result: dict[str, list[dict[str, Any]]] = {}
    for row in frame.select([key, *available]).sort([key, *([available[0]] if available else [])]).to_dicts():
        result.setdefault(str(row[key]), []).append({field: row.get(field) for field in available})
    return result


def _augment_taxa(taxa: pl.DataFrame) -> pl.DataFrame:
    rows = taxa.to_dicts()
    by_name = {_normalized(row.get("scientific_name")): row for row in rows}
    order = by_name.get("lepidoptera")
    if order is None:
        order = {column: None for column in taxa.columns}
        order.update(
            accepted_taxon_key="scope:lepidoptera",
            scientific_name="Lepidoptera",
            rank="ORDER",
            parent_key="",
        )
        rows.append(order)
    if "papilionoidea" not in by_name:
        superfamily = {column: None for column in taxa.columns}
        superfamily.update(
            accepted_taxon_key="scope:papilionoidea",
            scientific_name="Papilionoidea",
            rank="SUPERFAMILY",
            parent_key=order["accepted_taxon_key"],
        )
        rows.append(superfamily)
    return pl.DataFrame(rows, schema=taxa.schema)


def _augment_names(names: pl.DataFrame, taxa: pl.DataFrame) -> pl.DataFrame:
    additions: list[dict[str, Any]] = []
    taxa_by_name = {
        _normalized(row["scientific_name"]): str(row["accepted_taxon_key"])
        for row in taxa.iter_rows(named=True)
    }
    for display_name, normalized, name_class, language, key in (
        ("Lepidoptera", "lepidoptera", "accepted_scientific", "la", taxa_by_name["lepidoptera"]),
        ("Papilionoidea", "papilionoidea", "accepted_scientific", "la", taxa_by_name["papilionoidea"]),
        ("butterfly", "butterfly", "vernacular", "en", taxa_by_name["papilionoidea"]),
    ):
        if names.filter(
            (pl.col("accepted_taxon_key") == key)
            & (pl.col("normalized_match_key") == normalized)
        ).height:
            continue
        row = {column: None for column in names.columns}
        row.update(
            name_id=f"scope-policy:{normalized}",
            registry_version="final-consolidation-v1",
            accepted_taxon_key=key,
            verbatim_name=display_name,
            display_name=display_name,
            normalized_match_key=normalized,
            language=language,
            api_language_code=language,
            script="Latn",
            region="",
            bcp47=language,
            bbox="",
            name_class=name_class,
            source="butterfly_scope_policy",
            source_record_id=f"butterfly_scope_policy:{normalized}",
            source_taxon_id=key,
            lineage_check="scope_policy",
            trust_tier="T1",
            precision_tier="scope",
            confidence="high",
            enabled=True,
            disabled_reason="",
            review_state="policy",
            corroborated=True,
            query_eligible=True,
            query_disabled_reason="",
            species_specificity_score=0.0,
        )
        additions.append(row)
    return pl.concat([names, pl.DataFrame(additions, schema=names.schema)], how="vertical") if additions else names


def build_species_enrichments(
    *,
    source_parquet: str | Path,
    registry_dir: str | Path,
    output_path: str | Path,
    source_assertions_path: str | Path | None,
) -> pl.DataFrame:
    """Build one deterministic nested enrichment row per dataset species."""

    registry = Path(registry_dir)
    taxa = _augment_taxa(pl.read_parquet(registry / "taxa.parquet"))
    names = _augment_names(pl.read_parquet(registry / "names.parquet"), taxa)
    names = canonicalize_keyword_rows(names, taxa)
    queries = pl.DataFrame(
        canonical_query_rows(
            names,
            taxa,
            registry_version="final-consolidation-v1",
            registry_schema_version=REGISTRY_SCHEMA_VERSION,
            compiler_version=COMPILER_VERSION,
        )
    )
    paths = pl.read_parquet(registry / "species_paths.parquet")
    source_species = (
        pl.scan_parquet(source_parquet)
        .select("speciesKey", "species")
        .unique()
        .collect()
        .sort(["speciesKey", "species"])
    )
    taxa_species = taxa.filter(pl.col("rank") == "SPECIES")
    exact = {str(row["accepted_taxon_key"]).removeprefix("gbif:"): row for row in taxa_species.to_dicts()}
    name_buckets: dict[str, list[dict[str, Any]]] = {}
    for row in taxa_species.to_dicts():
        name_buckets.setdefault(_normalized(row["scientific_name"]), []).append(row)
    path_by_species = {str(row["accepted_taxon_key"]): row for row in paths.to_dicts()}
    evidence_by_taxon = _records_by_key(names, "accepted_taxon_key", _EVIDENCE_FIELDS)
    assertions = (
        pl.read_parquet(source_assertions_path)
        if source_assertions_path is not None and Path(source_assertions_path).exists()
        else pl.DataFrame()
    )
    assertions_by_taxon = (
        _records_by_key(assertions, "accepted_taxon_key", _ASSERTION_FIELDS)
        if not assertions.is_empty()
        else {}
    )
    queries_by_owner = (
        _records_by_key(queries, "accepted_taxon_key", _QUERY_FIELDS)
        if not queries.is_empty()
        else {}
    )

    rows: list[dict[str, Any]] = []
    for item in source_species.iter_rows(named=True):
        species_key = str(item.get("speciesKey") or "")
        matched = exact.get(species_key)
        method = "exact_gbif_species_key" if matched is not None else ""
        if matched is None:
            candidates = name_buckets.get(_normalized(item.get("species")), [])
            if len(candidates) == 1:
                matched = candidates[0]
                method = "unique_normalized_species_name"
        if matched is None:
            rows.append(
                {
                    "dataset_species_key": species_key,
                    "dataset_species": item.get("species"),
                    "registry_match_status": "unmatched",
                    "registry_match_method": None,
                    "registry_taxon_key": None,
                    "keyword_evidence": [],
                    "keyword_source_assertions": [],
                    "flickr_query_terms": [],
                }
            )
            continue
        taxon_key = str(matched["accepted_taxon_key"])
        path = path_by_species.get(taxon_key, {})
        ancestor_keys = [
            str(path.get(field) or "")
            for field in ("species_node_id", "genus_node_id", "family_node_id")
        ]
        ancestor_keys.extend(["scope:papilionoidea", "scope:lepidoptera"])
        evidence = [
            evidence_row
            for key in ancestor_keys
            for evidence_row in evidence_by_taxon.get(key, [])
        ]
        assertion_rows = [
            assertion
            for key in ancestor_keys
            for assertion in assertions_by_taxon.get(key, [])
        ]
        query_rows = [
            query
            for key in ancestor_keys
            for query in queries_by_owner.get(key, [])
        ]
        query_rows.sort(
            key=lambda row: (
                int(row.get("query_stage_order") or 99),
                int(row.get("search_priority") or 999999),
                str(row.get("normalized_query_term") or ""),
                str(row.get("search_field") or ""),
            )
        )
        rows.append(
            {
                "dataset_species_key": species_key,
                "dataset_species": item.get("species"),
                "registry_match_status": "matched",
                "registry_match_method": method,
                "registry_taxon_key": taxon_key,
                "keyword_evidence": evidence,
                "keyword_source_assertions": assertion_rows,
                "flickr_query_terms": query_rows,
            }
        )
    result = pl.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output, compression="zstd")
    return result


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _struct_expression(
    con: duckdb.DuckDBPyConnection,
    path: str | Path,
    alias: str,
    excluded: set[str],
) -> str:
    fields = [
        str(row[0])
        for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()
        if str(row[0]) not in excluded
    ]
    return "struct_pack(" + ",".join(
        f"{_quoted(field)} := {alias}.{_quoted(field)}" for field in fields
    ) + ")"


def build_final_parquet(
    *,
    temporal_parquet: str | Path,
    pre_temporal_parquet: str | Path,
    registry_dir: str | Path,
    source_assertions_path: str | Path | None,
    quality_dir: str | Path,
    output_dir: str | Path,
    producer_git_sha: str,
) -> dict[str, Any]:
    """Stage, validate, and atomically publish the final enriched Parquet."""

    temporal = Path(temporal_parquet).resolve()
    pre_temporal = Path(pre_temporal_parquet).resolve()
    quality = Path(quality_dir).resolve()
    output = Path(output_dir).resolve()
    staging = output.with_name(f".{output.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    enrichment_path = staging / "species_enrichments.parquet"
    build_species_enrichments(
        source_parquet=temporal,
        registry_dir=registry_dir,
        output_path=enrichment_path,
        source_assertions_path=source_assertions_path,
    )

    final_path = staging / FINAL_FILENAME
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='52GB'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET temp_directory=?", [str(staging / ".duckdb_tmp")])
    media_quality = quality / "media_assertion_quality/media_assertion_quality.parquet"
    occurrence_quality = quality / "occurrence_quality/occurrence_quality.parquet"
    rights = quality / "rights_and_attribution/media_rights.parquet"
    duplicates = quality / "duplicates/duplicate_membership.parquet"
    readiness = quality / "ai_readiness/parts/*.parquet"
    assertions = quality / "quality_results/phase3/derived_assertions.parquet"
    mq_struct = _struct_expression(con, media_quality, "mq", {"source_row_id", "media_assertion_id", "gbifID", "source_sort_position"})
    oq_struct = _struct_expression(con, occurrence_quality, "oq", {"gbifID"})
    rights_struct = _struct_expression(con, rights, "rq", {"source_row_id", "media_assertion_id", "gbifID", "media_identifier"})
    duplicate_struct = _struct_expression(con, duplicates, "dq", {"source_row_id", "media_assertion_id", "gbifID"})
    readiness_struct = _struct_expression(con, readiness, "ar", {"source_row_id", "media_assertion_id", "gbifID"})
    source_count = int(con.execute("SELECT count(*) FROM read_parquet(?)", [str(temporal)]).fetchone()[0])
    query = f"""
        COPY (
          WITH source_rows AS (
            SELECT row_number() OVER () AS source_ordinal, *
            FROM read_parquet(?)
          ),
          quality_rows AS (
            SELECT row_number() OVER (ORDER BY source_sort_position) AS source_ordinal,
                   source_row_id, media_assertion_id, gbifID
            FROM read_parquet(?)
          ),
          identities AS (
            SELECT s.gbifID, s.media_identifier, s.media_references,
                   q.source_row_id, q.media_assertion_id
            FROM source_rows s JOIN quality_rows q USING (source_ordinal)
          ),
          derived_by_occurrence AS (
            SELECT gbifID,
              list(struct_pack(
                assertion_id := assertion_id,
                target_field := target_field,
                original_value := original_value,
                derived_value := derived_value,
                evidence_source := evidence_source,
                derivation_method := derivation_method,
                derivation_rule_version := derivation_rule_version,
                confidence_class := confidence_class,
                validation_status := validation_status,
                conflict_status := conflict_status,
                reviewer_status := reviewer_status
              ) ORDER BY target_field, assertion_id) AS derived_quality_assertions
            FROM read_parquet(?) GROUP BY gbifID
          )
          SELECT
            t.*,
            i.source_row_id,
            i.media_assertion_id,
            {oq_struct} AS occurrence_quality,
            {mq_struct} AS media_quality,
            {rights_struct} AS rights_quality,
            {duplicate_struct} AS duplicate_quality,
            {readiness_struct} AS ai_readiness,
            da.derived_quality_assertions,
            se.registry_match_status,
            se.registry_match_method,
            se.registry_taxon_key,
            se.keyword_evidence,
            se.keyword_source_assertions,
            se.flickr_query_terms
          FROM read_parquet(?) t
          JOIN identities i
            ON t.gbifID IS NOT DISTINCT FROM i.gbifID
           AND t.media_identifier IS NOT DISTINCT FROM i.media_identifier
           AND t.media_references IS NOT DISTINCT FROM i.media_references
          LEFT JOIN read_parquet(?) oq ON t.gbifID = oq.gbifID
          LEFT JOIN read_parquet(?) mq ON i.media_assertion_id = mq.media_assertion_id
          LEFT JOIN read_parquet(?) rq ON i.media_assertion_id = rq.media_assertion_id
          LEFT JOIN read_parquet(?) dq ON i.media_assertion_id = dq.media_assertion_id
          LEFT JOIN read_parquet(?) ar ON i.media_assertion_id = ar.media_assertion_id
          LEFT JOIN derived_by_occurrence da ON t.gbifID = da.gbifID
          LEFT JOIN read_parquet(?) se
            ON coalesce(t.speciesKey, '') = coalesce(se.dataset_species_key, '')
           AND coalesce(t.species, '') = coalesce(se.dataset_species, '')
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """
    params = [
        # DuckDB binds the outer COPY target before placeholders in its SELECT.
        str(final_path),
        str(pre_temporal),
        str(media_quality),
        str(assertions),
        str(temporal),
        str(occurrence_quality),
        str(media_quality),
        str(rights),
        str(duplicates),
        str(readiness),
        str(enrichment_path),
    ]
    try:
        con.execute(query, params)
        output_count = int(
            con.execute("SELECT count(*) FROM read_parquet(?)", [str(final_path)]).fetchone()[0]
        )
        missing_identity = int(
            con.execute(
                "SELECT count(*) FROM read_parquet(?) WHERE source_row_id IS NULL OR media_assertion_id IS NULL",
                [str(final_path)],
            ).fetchone()[0]
        )
        if output_count != source_count or missing_identity:
            raise ValueError(
                f"acceptance failure: source_rows={source_count}, output_rows={output_count}, "
                f"missing_identity={missing_identity}"
            )
    finally:
        con.close()

    metadata = pq.ParquetFile(final_path).metadata
    row_group_rows = [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]
    if sum(row_group_rows) != source_count or any(rows <= 0 for rows in row_group_rows):
        raise ValueError("Parquet row groups are incomplete")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite published directory: {output}")
    enrichment_path.unlink()
    os.replace(staging, output)
    published = output / FINAL_FILENAME
    manifest = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "publication_role": "user_authorized_legacy_consolidation_source_of_truth",
        "ground_zero_production_lineage": False,
        "created_at": datetime.now(UTC).isoformat(),
        "producer_git_sha": producer_git_sha,
        "artifact": {
            "path": FINAL_FILENAME,
            "rows": source_count,
            "columns": metadata.num_columns,
            "row_groups": metadata.num_row_groups,
            "row_group_rows": row_group_rows,
            "bytes": published.stat().st_size,
            "sha256": _sha256(published),
        },
        "inputs": {
            "temporal_parquet": {"path": str(temporal), "sha256": _sha256(temporal)},
            "pre_temporal_parquet": {
                "path": str(pre_temporal),
                "sha256": _sha256(pre_temporal),
            },
            "registry_dir": str(Path(registry_dir).resolve()),
            "quality_dir": str(quality),
        },
        "acceptance_gate": {
            "row_count_preserved": True,
            "stable_media_identity_complete": True,
            "row_groups_complete": True,
            "manifest_written_last": True,
        },
    }
    (output / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
