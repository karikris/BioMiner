from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import duckdb
import polars as pl


def audit_registry(registry_dir: str | Path, *, report_dir: str | Path = "reports") -> dict[str, Any]:
    base = Path(registry_dir)
    with duckdb.connect(":memory:") as conn:
        summary = {
            "registry_dir": str(base),
            "taxa_by_rank": _count_map(conn, base / "taxa.parquet", "rank", where="rank <> ''"),
            "taxa_by_family": _count_map(conn, base / "taxa.parquet", "family", where="family <> ''"),
            "enabled_names_by_class": _count_map(conn, base / "names.parquet", "name_class", where="enabled = true"),
            "names_by_source": _count_map(conn, base / "names.parquet", "source"),
            "names_by_language": _count_map(conn, base / "names.parquet", "language", where="language <> ''"),
            "flickr_queries_by_field": _count_map(
                conn,
                base / "flickr_query_definitions.parquet",
                "search_field",
                where="enabled = true",
            ),
            "qa_by_severity": _count_map(conn, base / "qa_findings.parquet", "severity"),
        }
    registry_version = _registry_version(base)
    language_report = _language_target_coverage_report(base, registry_version=registry_version)
    names = pl.read_parquet(base / "names.parquet")
    keyword_metrics = _keyword_metrics(names)
    gap_report = _curated_gap_report(language_report)
    report_paths = _write_language_reports(
        report_dir=Path(report_dir),
        registry_version=registry_version,
        language_report=language_report,
        gap_report=gap_report,
    )
    return {**summary, **keyword_metrics, **report_paths}


def _keyword_metrics(names: pl.DataFrame) -> dict[str, Any]:
    if names.is_empty() or "canonical_keyword_id" not in names.columns:
        return {}
    enabled = names.filter(pl.col("enabled"))
    tiers = (
        enabled.filter(pl.col("is_canonical_keyword"))
        .group_by("effective_trust_tier")
        .agg(pl.col("canonical_keyword_id").n_unique().alias("count"))
        .sort("effective_trust_tier")
        .to_dicts()
    )
    collisions = enabled.group_by("canonical_keyword_id").agg(
        pl.col("accepted_taxon_key").n_unique().alias("taxa"),
        pl.col("original_trust_tier").n_unique().alias("tiers"),
    )
    return {
        "unique_normalized_terms_by_tier": {str(row["effective_trust_tier"]): int(row["count"]) for row in tiers},
        "duplicate_keyword_rows_suppressed": names.filter(pl.col("suppressed_duplicate")).height,
        "cross_species_collisions": collisions.filter(pl.col("taxa") > 1).height,
        "cross_tier_collisions": collisions.filter(pl.col("tiers") > 1).height,
    }


def _count_map(conn: duckdb.DuckDBPyConnection, parquet_path: Path, column: str, *, where: str = "true") -> dict[str, int]:
    rows = conn.execute(
        f"""
        SELECT {column} AS key, count(*) AS count
        FROM read_parquet(?)
        WHERE {where}
        GROUP BY {column}
        ORDER BY {column}
        """,
        [str(parquet_path)],
    ).fetchall()
    return {str(key): int(count) for key, count in rows}


def _language_target_coverage_report(base: Path, *, registry_version: str) -> dict[str, Any]:
    taxa = _read_optional_parquet(base / "taxa.parquet")
    range_countries = _read_optional_parquet(base / "range_countries.parquet")
    language_targets = _read_optional_parquet(base / "country_language_targets.parquet")
    assertions = _read_optional_parquet(base / "source_name_assertions.parquet")
    name_candidates = _read_optional_parquet(base / "name_candidates.parquet")
    translation_candidates = _read_optional_parquet(base / "translation_candidates.parquet")
    curated_assertions = _curated_assertions(assertions)
    generated_candidates = _generated_candidates(translation_candidates)
    target_rows = language_targets.to_dicts()
    curated_rows = curated_assertions.to_dicts()
    generated_rows = generated_candidates.to_dicts()
    gaps = [
        _language_gap_row(row)
        for row in target_rows
        if not _has_curated_source_for_target(row, curated_rows)
    ]
    only_generated = _languages_with_only_generated_candidates(gaps, generated_rows)
    species_summaries, dynamic_species_summaries = _species_regional_coverage_summaries(
        taxa=taxa,
        target_rows=target_rows,
        curated_rows=curated_rows,
        generated_rows=generated_rows,
    )
    return {
        "registry_dir": str(base),
        "registry_version": registry_version,
        "occurrence_countries_by_range_status": _count_by(range_countries, "range_status"),
        "language_targets_by_region": _language_targets_by_region(target_rows),
        "curated_names_by_source_language_region": _curated_names_by_source_language_region(curated_rows),
        "languages_with_no_curated_source_found": gaps,
        "languages_with_only_generated_candidates": only_generated,
        "translation_candidates_by_source_language": _translation_candidates_by_source_language(generated_rows),
        "names_disabled_due_to_ambiguity": _disabled_name_rows([assertions, name_candidates], reason_terms=("ambig", "collision")),
        "names_disabled_due_to_taxonomic_caution": _disabled_name_rows([assertions, name_candidates], reason_terms=("taxonomic_caution",)),
        "species_regional_coverage_summary": species_summaries,
        **dynamic_species_summaries,
    }


def _curated_gap_report(language_report: dict[str, Any]) -> dict[str, Any]:
    gaps = list(language_report["languages_with_no_curated_source_found"])
    return {
        "registry_dir": language_report["registry_dir"],
        "registry_version": language_report["registry_version"],
        "gap_count": len(gaps),
        "curated_vernacular_gaps": gaps,
        "languages_with_only_generated_candidates": language_report["languages_with_only_generated_candidates"],
        "names_disabled_due_to_ambiguity": language_report["names_disabled_due_to_ambiguity"],
        "names_disabled_due_to_taxonomic_caution": language_report["names_disabled_due_to_taxonomic_caution"],
        "species_regional_coverage_summary": language_report["species_regional_coverage_summary"],
    }


def _write_language_reports(
    *,
    report_dir: Path,
    registry_version: str,
    language_report: dict[str, Any],
    gap_report: dict[str, Any],
) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    safe_version = _safe_filename(registry_version)
    language_json = report_dir / f"language_target_coverage_{safe_version}.json"
    language_md = report_dir / f"language_target_coverage_{safe_version}.md"
    gap_json = report_dir / f"curated_vernacular_gap_report_{safe_version}.json"
    gap_md = report_dir / f"curated_vernacular_gap_report_{safe_version}.md"
    language_json.write_text(json.dumps(language_report, indent=2, sort_keys=True), encoding="utf-8")
    gap_json.write_text(json.dumps(gap_report, indent=2, sort_keys=True), encoding="utf-8")
    language_md.write_text(_language_report_markdown(language_report), encoding="utf-8")
    gap_md.write_text(_gap_report_markdown(gap_report), encoding="utf-8")
    return {
        "language_target_coverage_report": str(language_json),
        "language_target_coverage_markdown": str(language_md),
        "curated_vernacular_gap_report": str(gap_json),
        "curated_vernacular_gap_markdown": str(gap_md),
    }


def _read_optional_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


def _registry_version(base: Path) -> str:
    manifest = base / "manifest.json"
    if not manifest.exists():
        return "unknown"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "unknown"
    return str(payload.get("registry_version") or payload.get("version") or "unknown")


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "unknown").strip("_") or "unknown"


def _count_by(frame: pl.DataFrame, column: str) -> dict[str, int]:
    if frame.is_empty() or column not in frame.columns:
        return {}
    rows = frame.group_by(column).len().sort(column).to_dicts()
    return {str(row[column]): int(row["len"]) for row in rows}


def _curated_assertions(assertions: pl.DataFrame) -> pl.DataFrame:
    if assertions.is_empty():
        return assertions
    rows = []
    for row in assertions.to_dicts():
        if str(row.get("trust_tier") or "") == "T5":
            continue
        if str(row.get("name_class") or "") == "generated_translation":
            continue
        if not bool(row.get("enabled")):
            continue
        if str(row.get("disabled_reason") or ""):
            continue
        rows.append(row)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _generated_candidates(candidates: pl.DataFrame) -> pl.DataFrame:
    if candidates.is_empty():
        return candidates
    rows = [row for row in candidates.to_dicts() if str(row.get("trust_tier") or "") == "T5"]
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _language_targets_by_region(target_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in target_rows:
        key = (str(row.get("region") or ""), str(row.get("language_code") or ""))
        entry = grouped.setdefault(
            key,
            {"region": key[0], "language_code": key[1], "total_targets": 0, "enabled_targets": 0, "disabled_targets": 0},
        )
        entry["total_targets"] += 1
        if bool(row.get("enabled")):
            entry["enabled_targets"] += 1
        else:
            entry["disabled_targets"] += 1
    return [grouped[key] for key in sorted(grouped)]


def _curated_names_by_source_language_region(curated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], int] = {}
    for row in curated_rows:
        key = (str(row.get("source") or ""), str(row.get("language") or ""), str(row.get("region") or ""))
        grouped[key] = grouped.get(key, 0) + 1
    return [
        {"source": source, "language": language, "region": region, "count": grouped[(source, language, region)]}
        for source, language, region in sorted(grouped)
    ]


def _has_curated_source_for_target(target: dict[str, Any], curated_rows: list[dict[str, Any]]) -> bool:
    accepted_key = str(target.get("accepted_taxon_key") or "")
    language = str(target.get("language_code") or "")
    valid_regions = {
        str(target.get("region") or ""),
        str(target.get("country_code") or ""),
        str(target.get("admin1_code") or ""),
    }
    valid_regions.discard("")
    return any(
        str(row.get("accepted_taxon_key") or "") == accepted_key
        and str(row.get("language") or "") == language
        and str(row.get("region") or "") in valid_regions
        for row in curated_rows
    )


def _language_gap_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted_taxon_key": str(row.get("accepted_taxon_key") or ""),
        "scientific_name": str(row.get("scientific_name") or ""),
        "country_code": str(row.get("country_code") or ""),
        "country_name": str(row.get("country_name") or ""),
        "admin1_code": str(row.get("admin1_code") or ""),
        "admin1_name": str(row.get("admin1_name") or ""),
        "language_code": str(row.get("language_code") or ""),
        "language_name": str(row.get("language_name") or ""),
        "region": str(row.get("region") or ""),
        "target_enabled": bool(row.get("enabled")),
        "target_disabled_reason": str(row.get("disabled_reason") or ""),
    }


def _languages_with_only_generated_candidates(gaps: list[dict[str, Any]], generated_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    generated_keys = {
        (str(row.get("accepted_taxon_key") or ""), str(row.get("target_language") or ""))
        for row in generated_rows
    }
    rows = {
        (gap["accepted_taxon_key"], gap["language_code"], gap["region"])
        for gap in gaps
        if (gap["accepted_taxon_key"], gap["language_code"]) in generated_keys
    }
    return [
        {"accepted_taxon_key": accepted_key, "language_code": language, "region": region}
        for accepted_key, language, region in sorted(rows)
    ]


def _translation_candidates_by_source_language(generated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], int] = {}
    for row in generated_rows:
        key = (str(row.get("source") or ""), str(row.get("target_language") or ""))
        grouped[key] = grouped.get(key, 0) + 1
    return [
        {"source": source, "language_code": language, "count": grouped[(source, language)]}
        for source, language in sorted(grouped)
    ]


def _disabled_name_rows(frames: list[pl.DataFrame], *, reason_terms: tuple[str, ...]) -> list[dict[str, str]]:
    rows: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for frame in frames:
        if frame.is_empty():
            continue
        for row in frame.to_dicts():
            reason = str(row.get("disabled_reason") or "").casefold()
            if not any(term in reason for term in reason_terms):
                continue
            key = (
                str(row.get("accepted_taxon_key") or ""),
                str(row.get("display_name") or ""),
                str(row.get("language") or ""),
                str(row.get("region") or ""),
                str(row.get("source") or ""),
            )
            rows[key] = {
                "accepted_taxon_key": key[0],
                "display_name": key[1],
                "language": key[2],
                "region": key[3],
                "source": key[4],
            }
    return [rows[key] for key in sorted(rows)]


def _species_regional_coverage_summaries(
    *,
    taxa: pl.DataFrame,
    target_rows: list[dict[str, Any]],
    curated_rows: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    species_by_key = {
        str(row.get("accepted_taxon_key") or ""): str(row.get("scientific_name") or "")
        for row in taxa.to_dicts()
        if str(row.get("rank") or "") == "SPECIES" and row.get("accepted_taxon_key") and row.get("scientific_name")
    }
    summaries: dict[str, Any] = {}
    dynamic_keys: dict[str, Any] = {}
    for accepted_key, scientific_name in sorted(species_by_key.items(), key=lambda item: item[1]):
        summary = _single_species_regional_coverage_summary(
            accepted_key=accepted_key,
            target_rows=target_rows,
            curated_rows=curated_rows,
            generated_rows=generated_rows,
        )
        summaries[scientific_name] = summary
        dynamic_keys[f"{_safe_report_key(scientific_name)}_regional_coverage_summary"] = summary
    return summaries, dynamic_keys


def _single_species_regional_coverage_summary(
    *,
    accepted_key: str,
    target_rows: list[dict[str, Any]],
    curated_rows: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, dict[str, set[str]]] = {}
    for row in target_rows:
        if str(row.get("accepted_taxon_key") or "") != accepted_key:
            continue
        region = str(row.get("region") or "")
        entry = summary.setdefault(
            region,
            {
                "countries": set(),
                "language_targets": set(),
                "curated_languages": set(),
                "generated_candidate_languages": set(),
                "missing_curated_languages": set(),
            },
        )
        if row.get("country_code"):
            entry["countries"].add(str(row.get("country_code")))
        language = str(row.get("language_code") or "")
        if language:
            entry["language_targets"].add(language)
        has_curated = _has_curated_source_for_target(row, curated_rows)
        has_generated = any(
            str(candidate.get("accepted_taxon_key") or "") == str(row.get("accepted_taxon_key") or "")
            and str(candidate.get("target_language") or "") == language
            for candidate in generated_rows
        )
        if has_curated:
            entry["curated_languages"].add(language)
        else:
            entry["missing_curated_languages"].add(language)
        if has_generated:
            entry["generated_candidate_languages"].add(language)
    return {
        region: {field: sorted(values) for field, values in fields.items()}
        for region, fields in sorted(summary.items())
    }


def _safe_report_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or "species"


def _language_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Language Target Coverage",
        "",
        f"- Registry version: {report['registry_version']}",
        f"- Registry dir: {report['registry_dir']}",
        "",
        "## Occurrence Countries By Range Status",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in report["occurrence_countries_by_range_status"].items())
    lines.extend(["", "## Species Regional Coverage Summary", ""])
    for species, species_summary in report["species_regional_coverage_summary"].items():
        for region, summary in species_summary.items():
            lines.append(f"- {species} / {region}: targets={', '.join(summary['language_targets'])}; missing={', '.join(summary['missing_curated_languages'])}")
    lines.extend(["", "## Languages With No Curated Source Found", ""])
    lines.extend(f"- {row['region']} {row['language_code']} ({row['country_code']})" for row in report["languages_with_no_curated_source_found"])
    lines.extend(["", "## Languages With Only Generated Candidates", ""])
    lines.extend(f"- {row['region']} {row['language_code']}" for row in report["languages_with_only_generated_candidates"])
    return "\n".join(lines) + "\n"


def _gap_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Curated Vernacular Gap Report",
        "",
        f"- Registry version: {report['registry_version']}",
        f"- Gap count: {report['gap_count']}",
        "",
        "## Curated Vernacular Gaps",
        "",
    ]
    lines.extend(f"- {row['scientific_name']} {row['region']} {row['language_code']} ({row['country_code']})" for row in report["curated_vernacular_gaps"])
    lines.extend(["", "## Languages With Only Generated Candidates", ""])
    lines.extend(f"- {row['region']} {row['language_code']}" for row in report["languages_with_only_generated_candidates"])
    lines.extend(["", "## Disabled Due To Taxonomic Caution", ""])
    lines.extend(f"- {row['display_name']} {row['region']} {row['source']}" for row in report["names_disabled_due_to_taxonomic_caution"])
    return "\n".join(lines) + "\n"
