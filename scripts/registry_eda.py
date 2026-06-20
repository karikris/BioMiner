from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/biominer-matplotlib")

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import polars as pl


NAME_CLASS_GROUPS = {
    "scientific": ["accepted_scientific"],
    "synonym": ["scientific_synonym"],
    "common": ["vernacular", "vernacular_alias"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build exploratory EDA reports for a BioMiner Step 0 registry.")
    parser.add_argument("--registry-dir", default="data/registry/butterflies-v1")
    parser.add_argument("--output-dir", default="reports/registry_eda_butterflies-v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry_dir = Path(args.registry_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_registry(registry_dir)
    summaries = build_summaries(data)
    write_tables(output_dir, summaries)
    write_json(output_dir / "summary_metrics.json", build_metrics_payload(registry_dir, output_dir, data, summaries))
    write_markdown(output_dir / "registry_eda_report.md", registry_dir, data, summaries)
    write_deck(output_dir / "registry_eda_deck.pdf", registry_dir, data, summaries)
    print(json.dumps({"output_dir": str(output_dir), "files": sorted(p.name for p in output_dir.iterdir())}, indent=2))
    return 0


def load_registry(registry_dir: Path) -> dict[str, Any]:
    required = {
        "taxa": "taxa.parquet",
        "names": "names.parquet",
        "evidence": "name_evidence.parquet",
        "queries": "flickr_query_definitions.parquet",
        "qa": "qa_findings.parquet",
        "relations": "taxon_relations.parquet",
        "snapshots": "source_snapshots.parquet",
    }
    frames = {key: pl.read_parquet(registry_dir / filename) for key, filename in required.items()}
    manifest_path = registry_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    frames["manifest"] = manifest
    return frames


def build_summaries(data: dict[str, Any]) -> dict[str, pl.DataFrame]:
    taxa = data["taxa"]
    names = data["names"]
    evidence = data["evidence"]
    queries = data["queries"]
    qa = data["qa"]

    family_taxa = taxa.filter(pl.col("rank") == "FAMILY").select(
        "family_key",
        "family",
        pl.col("accepted_taxon_key").alias("family_taxon_key"),
    )
    species = taxa.filter(pl.col("rank") == "SPECIES")
    genera = taxa.filter(pl.col("rank") == "GENUS")

    names_with_taxa = names.join(
        taxa.select("accepted_taxon_key", "rank", "family_key", "family", "genus_key", "genus", "species_key", "species"),
        on="accepted_taxon_key",
        how="left",
    )
    queries_with_taxa = queries.join(
        taxa.select("accepted_taxon_key", "rank"),
        on="accepted_taxon_key",
        how="left",
        suffix="_taxa",
    )

    species_name_counts = (
        names_with_taxa.filter(pl.col("rank") == "SPECIES")
        .group_by("accepted_taxon_key")
        .agg(name_count_exprs())
    )
    species_query_counts = (
        queries_with_taxa.filter(pl.col("rank") == "SPECIES")
        .group_by("accepted_taxon_key")
        .agg(
            pl.len().alias("query_definition_rows"),
            pl.col("query_definition_id").n_unique().alias("distinct_query_definitions"),
            (pl.col("search_field") == "tags").sum().alias("tag_query_rows"),
            (pl.col("search_field") == "text").sum().alias("text_query_rows"),
        )
    )
    species_summary_base = (
        species.select(
            "accepted_taxon_key",
            "scientific_name",
            "family_key",
            "family",
            "genus_key",
            "genus",
            "species_key",
            "species",
        )
        .join(species_name_counts, on="accepted_taxon_key", how="left")
        .join(species_query_counts, on="accepted_taxon_key", how="left")
    )
    species_summary = (
        species_summary_base.with_columns(fill_count_columns(species_summary_base))
        .with_columns(
            (pl.col("common_name_rows") > 0).alias("has_common_name"),
            (pl.col("synonym_rows") > 0).alias("has_synonym"),
            (pl.col("language_count") > 0).alias("has_language_metadata"),
        )
        .sort(["family", "genus", "scientific_name"])
    )

    species_per_genus = (
        species.group_by("family_key", "family", "genus_key", "genus")
        .agg(pl.len().alias("species_count"))
        .sort(["family", "species_count", "genus"], descending=[False, True, False])
    )

    family_taxon_counts_base = (
        family_taxa.join(
            species.group_by("family_key").agg(pl.len().alias("species_count")),
            on="family_key",
            how="left",
        )
        .join(genera.group_by("family_key").agg(pl.len().alias("genus_count")), on="family_key", how="left")
        .join(taxa.group_by("family_key").agg(pl.len().alias("taxa_rows")), on="family_key", how="left")
    )
    family_taxon_counts = family_taxon_counts_base.with_columns(fill_count_columns(family_taxon_counts_base))
    family_name_counts = (
        names_with_taxa.group_by("family_key", "family")
        .agg(name_count_exprs())
        .sort("family")
    )
    family_query_counts = (
        queries.group_by("family_key", "family")
        .agg(
            pl.len().alias("query_definition_rows"),
            (pl.col("search_field") == "tags").sum().alias("tag_query_rows"),
            (pl.col("search_field") == "text").sum().alias("text_query_rows"),
            pl.col("normalized_query_term").n_unique().alias("distinct_query_terms"),
        )
    )
    family_species_coverage = (
        species_summary.group_by("family_key", "family")
        .agg(
            (pl.col("has_common_name")).sum().alias("species_with_common_name"),
            (pl.col("has_synonym")).sum().alias("species_with_synonym"),
            (pl.col("has_language_metadata")).sum().alias("species_with_language_metadata"),
            pl.col("common_name_rows").mean().alias("mean_common_names_per_species"),
            pl.col("common_name_rows").median().alias("median_common_names_per_species"),
            pl.col("common_name_rows").std().alias("std_common_names_per_species"),
            pl.col("synonym_rows").mean().alias("mean_synonyms_per_species"),
            pl.col("synonym_rows").median().alias("median_synonyms_per_species"),
            pl.col("synonym_rows").std().alias("std_synonyms_per_species"),
            pl.col("language_count").mean().alias("mean_languages_per_species"),
            pl.col("language_count").max().alias("max_languages_per_species"),
            pl.col("query_definition_rows").mean().alias("mean_queries_per_species"),
        )
    )
    family_genus_variation = (
        species_per_genus.group_by("family_key", "family")
        .agg(
            pl.col("species_count").mean().alias("mean_species_per_genus"),
            pl.col("species_count").median().alias("median_species_per_genus"),
            pl.col("species_count").std().alias("std_species_per_genus"),
            pl.col("species_count").min().alias("min_species_per_genus"),
            pl.col("species_count").quantile(0.25).alias("p25_species_per_genus"),
            pl.col("species_count").quantile(0.75).alias("p75_species_per_genus"),
            pl.col("species_count").max().alias("max_species_per_genus"),
        )
    )
    family_summary_base = (
        family_taxon_counts.join(family_name_counts, on=["family_key", "family"], how="left")
        .join(family_query_counts, on=["family_key", "family"], how="left")
        .join(family_species_coverage, on=["family_key", "family"], how="left")
        .join(family_genus_variation, on=["family_key", "family"], how="left")
    )
    family_summary = (
        family_summary_base.with_columns(fill_count_columns(family_summary_base))
        .with_columns(
            (pl.col("species_with_common_name") / pl.col("species_count") * 100).alias("pct_species_with_common_name"),
            (pl.col("species_with_synonym") / pl.col("species_count") * 100).alias("pct_species_with_synonym"),
            (pl.col("species_count") / pl.col("genus_count")).alias("species_per_genus_ratio"),
        )
        .sort("species_count", descending=True)
    )

    language_coverage = (
        names_with_taxa.filter(pl.col("language") != "")
        .group_by("language")
        .agg(
            pl.len().alias("name_rows"),
            pl.col("normalized_match_key").n_unique().alias("distinct_normalized_names"),
            pl.col("accepted_taxon_key").n_unique().alias("taxa_covered"),
            pl.col("family_key").filter(pl.col("family_key") != "").n_unique().alias("families_covered"),
            pl.col("source").n_unique().alias("sources"),
            (pl.col("enabled")).sum().alias("enabled_name_rows"),
            pl.col("name_class").filter(pl.col("name_class").is_in(NAME_CLASS_GROUPS["common"])).count().alias("common_name_rows"),
        )
        .sort("name_rows", descending=True)
    )
    family_language = (
        names_with_taxa.filter(pl.col("language") != "")
        .group_by("family_key", "family", "language")
        .agg(
            pl.len().alias("name_rows"),
            pl.col("accepted_taxon_key").n_unique().alias("taxa_covered"),
            pl.col("normalized_match_key").n_unique().alias("distinct_normalized_names"),
        )
        .sort(["family", "name_rows"], descending=[False, True])
    )

    source_summary = (
        names.group_by("source", "name_class")
        .agg(
            pl.len().alias("name_rows"),
            pl.col("accepted_taxon_key").n_unique().alias("taxa_covered"),
            pl.col("language").filter(pl.col("language") != "").n_unique().alias("language_count"),
            (pl.col("enabled")).sum().alias("enabled_name_rows"),
        )
        .sort("name_rows", descending=True)
    )
    trust_precision = (
        names.group_by("trust_tier", "precision_tier", "confidence")
        .agg(pl.len().alias("name_rows"), pl.col("accepted_taxon_key").n_unique().alias("taxa_covered"))
        .sort("name_rows", descending=True)
    )
    name_class_summary = (
        names.group_by("name_class")
        .agg(
            pl.len().alias("name_rows"),
            pl.col("accepted_taxon_key").n_unique().alias("taxa_covered"),
            pl.col("language").filter(pl.col("language") != "").n_unique().alias("language_count"),
            (pl.col("enabled")).sum().alias("enabled_name_rows"),
            (~pl.col("enabled")).sum().alias("disabled_name_rows"),
        )
        .sort("name_rows", descending=True)
    )
    query_summary = (
        queries.group_by("search_field", "search_priority", "name_class")
        .agg(
            pl.len().alias("query_definition_rows"),
            pl.col("accepted_taxon_key").n_unique().alias("taxa_covered"),
            pl.col("normalized_query_term").n_unique().alias("distinct_query_terms"),
        )
        .sort(["search_priority", "search_field", "name_class"])
    )
    qa_summary = (
        qa.group_by("severity", "code")
        .agg(pl.len().alias("finding_rows"), pl.col("subject").n_unique().alias("distinct_subjects"))
        .sort(["severity", "finding_rows"], descending=[False, True])
    )
    evidence_summary = (
        evidence.group_by("source", "trust_tier", "review_state")
        .agg(
            pl.len().alias("evidence_rows"),
            pl.col("name_id").n_unique().alias("distinct_names"),
            pl.col("accepted_taxon_key").n_unique().alias("taxa_covered"),
        )
        .sort("evidence_rows", descending=True)
    )
    null_summary = pl.concat(
        [missing_profile(table_name, frame) for table_name, frame in data.items() if isinstance(frame, pl.DataFrame)],
        how="vertical",
    ).sort(["missing_pct", "table", "column"], descending=[True, False, False])
    duplicate_summary = build_duplicate_summary(taxa, names, queries)

    return {
        "family_summary": family_summary,
        "species_summary": species_summary,
        "species_per_genus": species_per_genus,
        "language_coverage": language_coverage,
        "family_language": family_language,
        "source_summary": source_summary,
        "trust_precision_summary": trust_precision,
        "name_class_summary": name_class_summary,
        "query_summary": query_summary,
        "qa_summary": qa_summary,
        "evidence_summary": evidence_summary,
        "null_summary": null_summary,
        "duplicate_summary": duplicate_summary,
    }


def name_count_exprs() -> list[pl.Expr]:
    return [
        pl.len().alias("name_rows"),
        pl.col("name_id").n_unique().alias("distinct_name_ids"),
        pl.col("normalized_match_key").n_unique().alias("distinct_normalized_names"),
        pl.col("name_class").filter(pl.col("name_class").is_in(NAME_CLASS_GROUPS["scientific"])).count().alias("scientific_name_rows"),
        pl.col("name_class").filter(pl.col("name_class").is_in(NAME_CLASS_GROUPS["synonym"])).count().alias("synonym_rows"),
        pl.col("name_class").filter(pl.col("name_class").is_in(NAME_CLASS_GROUPS["common"])).count().alias("common_name_rows"),
        pl.col("language").filter(pl.col("language") != "").n_unique().alias("language_count"),
        pl.col("source").filter(pl.col("source") != "").n_unique().alias("source_count"),
        (pl.col("enabled")).sum().alias("enabled_name_rows"),
        (~pl.col("enabled")).sum().alias("disabled_name_rows"),
    ]


def fill_count_columns(frame: pl.DataFrame) -> list[pl.Expr]:
    return [
        pl.when(pl.col(column).is_null()).then(0).otherwise(pl.col(column)).alias(column)
        for column in [
            "species_count",
            "genus_count",
            "taxa_rows",
            "name_rows",
            "distinct_name_ids",
            "distinct_normalized_names",
            "scientific_name_rows",
            "synonym_rows",
            "common_name_rows",
            "language_count",
            "source_count",
            "enabled_name_rows",
            "disabled_name_rows",
            "query_definition_rows",
            "distinct_query_definitions",
            "tag_query_rows",
            "text_query_rows",
            "distinct_query_terms",
            "species_with_common_name",
            "species_with_synonym",
            "species_with_language_metadata",
        ]
        if column in frame.columns
    ]


def missing_profile(table_name: str, frame: pl.DataFrame) -> pl.DataFrame:
    rows = []
    total = frame.height
    for column in frame.columns:
        series = frame[column]
        nulls = int(series.null_count())
        blanks = int((series.cast(pl.String) == "").sum()) if series.dtype == pl.String else 0
        missing = nulls + blanks
        rows.append(
            {
                "table": table_name,
                "column": column,
                "dtype": str(series.dtype),
                "rows": total,
                "null_count": nulls,
                "blank_string_count": blanks,
                "missing_count": missing,
                "missing_pct": (missing / total * 100) if total else 0.0,
                "distinct_count": int(series.n_unique()),
            }
        )
    return pl.DataFrame(rows)


def build_duplicate_summary(taxa: pl.DataFrame, names: pl.DataFrame, queries: pl.DataFrame) -> pl.DataFrame:
    rows = []
    checks = [
        ("taxa", "accepted_taxon_key", taxa["accepted_taxon_key"].is_duplicated().sum()),
        ("names", "name_id", names["name_id"].is_duplicated().sum()),
        ("names", "normalized_match_key", names["normalized_match_key"].is_duplicated().sum()),
        ("queries", "query_definition_id", queries["query_definition_id"].is_duplicated().sum()),
        ("queries", "normalized_query_term", queries["normalized_query_term"].is_duplicated().sum()),
    ]
    for table, column, duplicated_rows in checks:
        rows.append({"table": table, "column": column, "duplicated_rows": int(duplicated_rows)})
    return pl.DataFrame(rows)


def write_tables(output_dir: Path, summaries: dict[str, pl.DataFrame]) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in summaries.items():
        frame.write_parquet(tables_dir / f"{name}.parquet")


def build_metrics_payload(registry_dir: Path, output_dir: Path, data: dict[str, Any], summaries: dict[str, pl.DataFrame]) -> dict[str, Any]:
    manifest = data["manifest"]
    family = summaries["family_summary"]
    language = summaries["language_coverage"]
    qa = summaries["qa_summary"]
    nulls = summaries["null_summary"]
    species = summaries["species_summary"]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "registry_dir": str(registry_dir),
        "output_dir": str(output_dir),
        "manifest": manifest,
        "headline_counts": {
            "taxa_rows": data["taxa"].height,
            "species_rows": data["taxa"].filter(pl.col("rank") == "SPECIES").height,
            "genus_rows": data["taxa"].filter(pl.col("rank") == "GENUS").height,
            "family_rows": data["taxa"].filter(pl.col("rank") == "FAMILY").height,
            "name_rows": data["names"].height,
            "common_name_rows": int(data["names"].filter(pl.col("name_class").is_in(NAME_CLASS_GROUPS["common"])).height),
            "synonym_rows": int(data["names"].filter(pl.col("name_class").is_in(NAME_CLASS_GROUPS["synonym"])).height),
            "languages_with_nonblank_code": language.height,
            "query_definition_rows": data["queries"].height,
            "qa_finding_rows": data["qa"].height,
        },
        "family_species_counts": records(family.select("family", "species_count", "genus_count", "common_name_rows", "synonym_rows", "language_count")),
        "top_languages": records(language.head(25)),
        "qa_by_code": records(qa),
        "highest_missing_columns": records(nulls.head(30)),
        "species_coverage": {
            "species_with_common_name": int(species.filter(pl.col("has_common_name")).height),
            "species_with_synonym": int(species.filter(pl.col("has_synonym")).height),
            "species_without_common_name": int(species.filter(~pl.col("has_common_name")).height),
            "species_without_synonym": int(species.filter(~pl.col("has_synonym")).height),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown(path: Path, registry_dir: Path, data: dict[str, Any], summaries: dict[str, pl.DataFrame]) -> None:
    manifest = data["manifest"]
    family = summaries["family_summary"]
    language = summaries["language_coverage"]
    name_class = summaries["name_class_summary"]
    query = summaries["query_summary"]
    qa = summaries["qa_summary"]
    nulls = summaries["null_summary"]
    duplicates = summaries["duplicate_summary"]
    species = summaries["species_summary"]

    lines = [
        "# BioMiner Step 0 Registry EDA",
        "",
        f"Registry directory: `{registry_dir}`",
        f"Generated at: `{datetime.now(UTC).isoformat()}`",
        "",
        "## Manifest",
        "",
        markdown_table([{"key": key, "value": value} for key, value in sorted(manifest.items())]),
        "",
        "## Headline Counts",
        "",
        markdown_table(
            [
                {"metric": "taxa rows", "value": data["taxa"].height},
                {"metric": "family rows", "value": data["taxa"].filter(pl.col("rank") == "FAMILY").height},
                {"metric": "genus rows", "value": data["taxa"].filter(pl.col("rank") == "GENUS").height},
                {"metric": "species rows", "value": data["taxa"].filter(pl.col("rank") == "SPECIES").height},
                {"metric": "name rows", "value": data["names"].height},
                {"metric": "common name rows", "value": data["names"].filter(pl.col("name_class").is_in(NAME_CLASS_GROUPS["common"])).height},
                {"metric": "synonym rows", "value": data["names"].filter(pl.col("name_class").is_in(NAME_CLASS_GROUPS["synonym"])).height},
                {"metric": "nonblank languages", "value": language.height},
                {"metric": "query definition rows", "value": data["queries"].height},
                {"metric": "QA finding rows", "value": data["qa"].height},
            ]
        ),
        "",
        "## Family Summary",
        "",
        markdown_table(records(family.select(
            "family",
            "genus_count",
            "species_count",
            "species_per_genus_ratio",
            "name_rows",
            "common_name_rows",
            "synonym_rows",
            "language_count",
            "query_definition_rows",
            "pct_species_with_common_name",
            "pct_species_with_synonym",
        ))),
        "",
        "## Species Coverage",
        "",
        markdown_table(
            [
                {"metric": "species with common name", "value": species.filter(pl.col("has_common_name")).height},
                {"metric": "species without common name", "value": species.filter(~pl.col("has_common_name")).height},
                {"metric": "species with synonym", "value": species.filter(pl.col("has_synonym")).height},
                {"metric": "species without synonym", "value": species.filter(~pl.col("has_synonym")).height},
                {"metric": "mean common names per species", "value": round_float(species["common_name_rows"].mean())},
                {"metric": "std common names per species", "value": round_float(species["common_name_rows"].std())},
                {"metric": "mean synonyms per species", "value": round_float(species["synonym_rows"].mean())},
                {"metric": "std synonyms per species", "value": round_float(species["synonym_rows"].std())},
                {"metric": "max languages per species", "value": species["language_count"].max()},
            ]
        ),
        "",
        "## Name Classes",
        "",
        markdown_table(records(name_class)),
        "",
        "## Language Coverage",
        "",
        markdown_table(records(language.head(30))),
        "",
        "## Query Definitions",
        "",
        markdown_table(records(query)),
        "",
        "## QA Findings",
        "",
        markdown_table(records(qa.head(60))),
        "",
        "## Highest Missing/Blank Columns",
        "",
        markdown_table(records(nulls.head(60))),
        "",
        "## Duplicate Diagnostics",
        "",
        markdown_table(records(duplicates)),
        "",
        "## Generated Tables",
        "",
        "All drill-down tables are in `tables/*.parquet` beside this report.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_deck(path: Path, registry_dir: Path, data: dict[str, Any], summaries: dict[str, pl.DataFrame]) -> None:
    family = summaries["family_summary"]
    language = summaries["language_coverage"]
    name_class = summaries["name_class_summary"]
    source = summaries["source_summary"]
    query = summaries["query_summary"]
    qa = summaries["qa_summary"]
    nulls = summaries["null_summary"]
    species_per_genus = summaries["species_per_genus"]
    species = summaries["species_summary"]

    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        manifest = data["manifest"]
        text = "\n".join(
            [
                "BioMiner Step 0 Butterfly Registry EDA",
                f"Registry: {registry_dir}",
                f"Version: {manifest.get('registry_version', '')}",
                f"Build time: {manifest.get('build_time', '')}",
                f"QA status: {manifest.get('qa_status', '')}",
                "",
                f"Taxa rows: {data['taxa'].height:,}",
                f"Species rows: {data['taxa'].filter(pl.col('rank') == 'SPECIES').height:,}",
                f"Name rows: {data['names'].height:,}",
                f"Query definitions: {data['queries'].height:,}",
                f"Languages with nonblank code: {language.height:,}",
            ]
        )
        ax.text(0.05, 0.92, text, va="top", ha="left", fontsize=16, linespacing=1.45)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        barh_slide(pdf, family.sort("species_count"), "Species Count by Family", "family", "species_count", "species")
        grouped_bar_slide(
            pdf,
            family.sort("species_count"),
            "Taxonomic Structure by Family",
            "family",
            ["genus_count", "species_count"],
            ["genera", "species"],
        )
        grouped_bar_slide(
            pdf,
            family.sort("name_rows"),
            "Names by Family and Class",
            "family",
            ["scientific_name_rows", "synonym_rows", "common_name_rows"],
            ["scientific", "synonym", "common"],
        )
        grouped_bar_slide(
            pdf,
            family.sort("species_count"),
            "Species Coverage by Family",
            "family",
            ["pct_species_with_common_name", "pct_species_with_synonym"],
            ["with common name %", "with synonym %"],
        )
        barh_slide(pdf, language.head(25).sort("name_rows"), "Top Languages by Name Rows", "language", "name_rows", "name rows")
        heat_table_slide(
            pdf,
            summaries["family_language"],
            "Family x Language Coverage",
            row_col="family",
            col_col="language",
            value_col="name_rows",
            max_columns=16,
        )
        grouped_bar_slide(
            pdf,
            family.sort("query_definition_rows"),
            "Query Definitions by Family",
            "family",
            ["tag_query_rows", "text_query_rows"],
            ["tags", "text"],
        )
        barh_slide(pdf, name_class.sort("name_rows"), "Name Rows by Name Class", "name_class", "name_rows", "name rows")
        barh_slide(
            pdf,
            source.head(20).sort("name_rows"),
            "Top Source and Name-Class Rows",
            "source_name_class",
            "name_rows",
            "name rows",
            transform=lambda df: df.with_columns((pl.col("source") + "." + pl.col("name_class")).alias("source_name_class")),
        )
        barh_slide(pdf, qa.head(20).sort("finding_rows"), "Top QA Finding Codes", "code", "finding_rows", "finding rows")
        barh_slide(pdf, nulls.head(25).sort("missing_pct"), "Highest Null or Blank String Rates", "table_column", "missing_pct", "missing %", transform=lambda df: df.with_columns((pl.col("table") + "." + pl.col("column")).alias("table_column")))
        box_slide(pdf, species_per_genus, "Species per Genus Distribution by Family", "family", "species_count", "species per genus")
        hist_slide(pdf, species, "Common Names per Species", "common_name_rows", "species")
        hist_slide(pdf, species, "Synonyms per Species", "synonym_rows", "species")


def barh_slide(
    pdf: PdfPages,
    frame: pl.DataFrame,
    title: str,
    label_col: str,
    value_col: str,
    x_label: str,
    transform: Any | None = None,
) -> None:
    if transform is not None:
        frame = transform(frame)
    labels = [str(x) for x in frame[label_col].to_list()]
    values = [float(x or 0) for x in frame[value_col].to_list()]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.barh(labels, values, color="#3b6ea8")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label)
    ax.grid(axis="x", alpha=0.25)
    for index, value in enumerate(values):
        ax.text(value, index, f" {value:,.1f}" if value % 1 else f" {int(value):,}", va="center", fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def grouped_bar_slide(pdf: PdfPages, frame: pl.DataFrame, title: str, label_col: str, value_cols: list[str], legends: list[str]) -> None:
    labels = [str(x) for x in frame[label_col].to_list()]
    x_positions = list(range(len(labels)))
    width = 0.8 / len(value_cols)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    for offset, (column, legend) in enumerate(zip(value_cols, legends, strict=True)):
        values = [float(x or 0) for x in frame[column].to_list()]
        positions = [x + (offset - (len(value_cols) - 1) / 2) * width for x in x_positions]
        ax.bar(positions, values, width=width, label=legend)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def heat_table_slide(pdf: PdfPages, frame: pl.DataFrame, title: str, row_col: str, col_col: str, value_col: str, max_columns: int) -> None:
    top_cols = frame.group_by(col_col).agg(pl.col(value_col).sum().alias("total")).sort("total", descending=True).head(max_columns)[col_col].to_list()
    pivot = (
        frame.filter(pl.col(col_col).is_in(top_cols))
        .pivot(index=row_col, on=col_col, values=value_col, aggregate_function="sum")
        .fill_null(0)
        .sort(row_col)
    )
    rows = [str(x) for x in pivot[row_col].to_list()]
    cols = [c for c in pivot.columns if c != row_col]
    matrix = [[float(pivot[c][i] or 0) for c in cols] for i in range(pivot.height)]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_title(title, fontsize=16)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    fig.colorbar(image, ax=ax, label=value_col)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def box_slide(pdf: PdfPages, frame: pl.DataFrame, title: str, group_col: str, value_col: str, y_label: str) -> None:
    groups = sorted(str(x) for x in frame[group_col].unique().to_list())
    values = [frame.filter(pl.col(group_col) == group)[value_col].to_list() for group in groups]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.boxplot(values, tick_labels=groups, showfliers=False)
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def hist_slide(pdf: PdfPages, frame: pl.DataFrame, title: str, value_col: str, y_label: str) -> None:
    values = [float(x or 0) for x in frame[value_col].to_list()]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.hist(values, bins=40, color="#568f3f", edgecolor="white")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(value_col)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [format_markdown_value(row.get(column)) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def records(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return [{key: convert_value(value) for key, value in row.items()} for row in frame.to_dicts()]


def convert_value(value: Any) -> Any:
    if isinstance(value, float):
        return round_float(value)
    return value


def round_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def format_markdown_value(value: Any) -> str:
    value = convert_value(value)
    if value is None:
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
