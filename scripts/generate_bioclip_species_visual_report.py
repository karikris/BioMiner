from __future__ import annotations

import argparse
import html
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/biominer-matplotlib")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import polars as pl
import seaborn as sns


DEFAULT_RUN_DIR = Path(
    "data/live_runs/papilio_demoleus_global_multilingual_20260609_071759/"
    "bioclip25_species_all_global_2000_candidates_7families_batched"
)
DEFAULT_PREDICTIONS = DEFAULT_RUN_DIR / "image_triage_species_all.parquet"
DEFAULT_CANDIDATES = (
    DEFAULT_RUN_DIR.parent
    / "bioclip25_species_sample_1000_2000_candidates_7families"
    / "butterfly_species_candidates_2000.parquet"
)

SPECIES_COUNT_THRESHOLDS = (0.01, 0.05, 0.10)
SCORE_BINS = [-0.001, 0.50, 0.75, 0.90, 0.98, 1.001]
SCORE_LABELS = ["<=0.50", "0.50-0.75", "0.75-0.90", "0.90-0.98", "0.98-1.00"]

FAMILY_FALLBACK_BY_GENUS = {
    "Acrodipsas": "Lycaenidae",
    "Candalides": "Lycaenidae",
    "Hesperilla": "Hesperiidae",
    "Hypochrysops": "Lycaenidae",
    "Jalmenus": "Lycaenidae",
    "Ocybadistes": "Hesperiidae",
    "Ogyris": "Lycaenidae",
    "Oreixenica": "Nymphalidae",
    "Ornithoptera": "Papilionidae",
    "Pasma": "Hesperiidae",
    "Paralucia": "Lycaenidae",
    "Telicota": "Hesperiidae",
}

PALETTE = {
    "Papilionidae": "#0072B2",
    "Nymphalidae": "#D55E00",
    "Lycaenidae": "#009E73",
    "Hesperiidae": "#CC79A7",
    "Pieridae": "#E69F00",
    "Riodinidae": "#56B4E9",
    "Unresolved": "#777777",
    "gold": "#C49A00",
    "silver": "#8B8B8B",
    "bronze": "#8C564B",
    "in_review/no_geo": "#9467BD",
    "in_review/error": "#D62728",
}


@dataclass(frozen=True)
class FilterResult:
    rows_before: int
    rows_after: int
    species: str | None
    excluded_image_categories: tuple[str, ...]

    @property
    def rows_dropped(self) -> int:
        return self.rows_before - self.rows_after


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a visual analysis deck for the Papilio demoleus BioCLIP species parquet."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--species", help="Keep only records whose top-1 scientific name exactly matches this species.")
    parser.add_argument(
        "--exclude-image-categories",
        nargs="*",
        default=[],
        help="Drop records with these image_category values before reporting.",
    )
    parser.add_argument(
        "--write-filtered-parquet",
        action="store_true",
        help="Write the filtered dataframe used for the report to the output directory.",
    )
    parser.add_argument("--filtered-parquet", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or args.predictions.parent / "visual_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(
        context="talk",
        style="whitegrid",
        rc={
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        },
    )

    predictions = load_predictions(args.predictions, args.candidates)
    predictions, filter_result = apply_filters(
        predictions,
        species=args.species,
        excluded_image_categories=tuple(args.exclude_image_categories),
    )
    if predictions.is_empty():
        raise SystemExit("No records remain after applying report filters.")
    if args.write_filtered_parquet:
        filtered_parquet = args.filtered_parquet or output_dir / "filtered_report_records.parquet"
        predictions.write_parquet(filtered_parquet)

    summary = write_tables(predictions, output_dir, filter_result)
    figures = write_figures(predictions, output_dir)
    write_pdf_deck(figures, output_dir)
    write_html_report(predictions, summary, figures, args.predictions, args.candidates, output_dir)

    print(f"Wrote visual report to {output_dir / 'bioclip25_species_visual_report.html'}")
    print(f"Wrote PDF deck to {output_dir / 'bioclip25_species_visual_report_deck.pdf'}")


def load_predictions(predictions_path: Path, candidates_path: Path) -> pl.DataFrame:
    df = pl.read_parquet(predictions_path)
    df = normalize_reason_labels(df)
    if candidates_path.exists():
        candidates = pl.read_parquet(candidates_path)
        candidates = candidates.select(["scientific_name", "family", "genus"]).unique(subset=["scientific_name"])
        df = df.join(
            candidates,
            how="left",
            left_on="species_top1_scientific_name",
            right_on="scientific_name",
        )
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("family"), pl.lit(None, dtype=pl.Utf8).alias("genus"))

    fallback_family = pl.col("genus").replace_strict(FAMILY_FALLBACK_BY_GENUS, default=None)
    df = df.with_columns(
        pl.coalesce(
            pl.col("genus"),
            pl.col("species_top1_scientific_name").str.split(" ").list.first(),
        ).alias("genus")
    ).with_columns(
        pl.coalesce(pl.col("family"), fallback_family, pl.lit("Unresolved")).alias("family_resolved"),
        score_band_expr("species_top1_score").alias("score_band"),
        pl.format("{}-{}-01", pl.col("year"), pl.col("month")).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("capture_month"),
    )

    topk_metrics = pl.DataFrame([topk_summary(topk) for topk in df["species_topk_json"].to_list()])
    df = df.hstack(topk_metrics) if topk_metrics.height else df
    return df


def normalize_reason_labels(df: pl.DataFrame) -> pl.DataFrame:
    expressions = [
        pl.col(column).replace({"not_target_species": "below_50"}).alias(column)
        for column in ("bin_reason", "triage_reason", "publication_state_reason")
        if column in df.columns
    ]
    return df.with_columns(expressions) if expressions else df


def apply_filters(
    df: pl.DataFrame,
    *,
    species: str | None,
    excluded_image_categories: tuple[str, ...],
) -> tuple[pl.DataFrame, FilterResult]:
    rows_before = len(df)
    filtered = df
    if species:
        filtered = filtered.filter(pl.col("species_top1_scientific_name") == species)
    if excluded_image_categories:
        filtered = filtered.filter(~pl.col("image_category").is_in(excluded_image_categories))
    return filtered, FilterResult(
        rows_before=rows_before,
        rows_after=len(filtered),
        species=species,
        excluded_image_categories=excluded_image_categories,
    )


def topk_summary(topk: Any) -> dict[str, float | int | None]:
    rows = list(iter_topk_rows(topk))
    scores = [score for _, score in rows if score is not None]
    species = {species for species, score in rows if species and score is not None}
    result: dict[str, float | int | None] = {
        "species_topk_count": len(species),
        "species_top2_score": scores[1] if len(scores) > 1 else None,
        "species_top1_top2_margin": (scores[0] - scores[1]) if len(scores) > 1 else None,
    }
    for threshold in SPECIES_COUNT_THRESHOLDS:
        result[f"species_count_ge_{threshold:.2f}"] = len(
            {species for species, score in rows if species and score is not None and score >= threshold}
        )
    return result


def iter_topk_rows(topk: Any) -> Iterable[tuple[str | None, float | None]]:
    if topk is None:
        return
    for row in topk:
        if not isinstance(row, dict):
            continue
        label = row.get("label")
        score = row.get("score")
        species = clean_species_label(label)
        yield species, float(score) if score is not None else None


def clean_species_label(label: Any) -> str | None:
    if not isinstance(label, str) or not label:
        return None
    prefix = "a photo of "
    return label[len(prefix) :] if label.startswith(prefix) else label


def score_band_expr(column: str) -> pl.Expr:
    expr = pl.lit(None, dtype=pl.Utf8)
    for lower, upper, label in reversed(list(zip(SCORE_BINS[:-1], SCORE_BINS[1:], SCORE_LABELS, strict=True))):
        expr = pl.when((pl.col(column) > lower) & (pl.col(column) <= upper)).then(pl.lit(label)).otherwise(expr)
    return expr


def write_tables(df: pl.DataFrame, output_dir: Path, filter_result: FilterResult | None = None) -> dict[str, object]:
    top_species = value_count_table(df, "species_top1_scientific_name", "species")
    top_families = value_count_table(df, "family_resolved", "family")
    occurrence_bins = value_count_table(df, "occurrence_bin", "occurrence_bin")
    image_categories = value_count_table(df, "image_category", "image_category")
    life_stages = value_count_table(df, "life_stage", "life_stage")
    bin_reasons = value_count_table(df, "bin_reason", "bin_reason")
    score_bands = value_count_table(df, "score_band", "score_band")
    species_per_image = species_per_image_table(df)
    family_bin = (
        df.group_by(["family_resolved", "occurrence_bin"])
        .len(name="records")
        .sort(["family_resolved", "occurrence_bin"])
    )
    monthly = (
        df.group_by(["year", "month"])
        .len(name="records")
        .sort(["year", "month"])
    )

    table_map = {
        "top_species_counts.csv": top_species,
        "family_counts.csv": top_families,
        "occurrence_bin_counts.csv": occurrence_bins,
        "image_category_counts.csv": image_categories,
        "life_stage_counts.csv": life_stages,
        "bin_reason_counts.csv": bin_reasons,
        "score_band_counts.csv": score_bands,
        "species_per_image_counts.csv": species_per_image,
        "family_by_occurrence_bin.csv": family_bin,
        "monthly_counts.csv": monthly,
    }
    if "triage_group_top" in df.columns:
        table_map["triage_group_counts.csv"] = value_count_table(df, "triage_group_top", "triage_group_top")
    for filename, table in table_map.items():
        table.write_csv(output_dir / filename)

    summary = {
        "records": int(df.height),
        "unique_top1_species": int(df["species_top1_scientific_name"].n_unique()),
        "families": int(df["family_resolved"].n_unique()),
        "records_with_geo": int(df.filter(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null()).height),
        "records_with_event_date": int(df.filter(pl.col("captured_at").is_not_null()).height),
        "downloaded_images_deleted": int(df.select(pl.col("image_deleted_after_classification").fill_null(False).sum()).item()),
        "classification_status": value_counts_dict(df, "classification_status"),
        "occurrence_bins": value_counts_dict(df, "occurrence_bin"),
        "image_categories": value_counts_dict(df, "image_category"),
        "life_stages": value_counts_dict(df, "life_stage"),
        "bin_reasons": value_counts_dict(df, "bin_reason"),
        "score_stats": numeric_summary(df["species_top1_score"]),
        "bioclip_score_stats": numeric_summary(df["bioclip_top1_score"]),
        "triage_score_stats": numeric_summary(df["triage_top1_score"]),
        "top1_top2_margin_stats": numeric_summary(df["species_top1_top2_margin"]),
        "species_per_image_stats": {
            "topk_slots": numeric_summary(df["species_topk_count"]),
            **{
                f"score_ge_{threshold:.2f}": numeric_summary(df[f"species_count_ge_{threshold:.2f}"])
                for threshold in SPECIES_COUNT_THRESHOLDS
            },
        },
        "top_10_species": top_species.head(10).to_dicts(),
        "top_10_families": top_families.head(10).to_dicts(),
        "unresolved_family_records": int(df.filter(pl.col("family_resolved") == "Unresolved").height),
    }
    if "species_top1_top2_margin" in df.columns:
        summary["species_margin_stats"] = numeric_summary(df["species_top1_top2_margin"])
    if "species_topk_entropy" in df.columns:
        summary["species_entropy_stats"] = numeric_summary(df["species_topk_entropy"])
    if "triage_group_top" in df.columns:
        summary["triage_groups"] = value_counts_dict(df, "triage_group_top")
    if filter_result is not None:
        summary["filters"] = {
            "source_rows_before_filter": filter_result.rows_before,
            "rows_after_filter": filter_result.rows_after,
            "rows_dropped_by_filter": filter_result.rows_dropped,
            "species_top1_scientific_name": filter_result.species,
            "excluded_image_categories": list(filter_result.excluded_image_categories),
        }
    (output_dir / "report_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def value_count_table(df: pl.DataFrame, column: str, label: str) -> pl.DataFrame:
    return (
        df.group_by(column, maintain_order=True)
        .len(name="records")
        .rename({column: label})
        .with_columns(pl.col(label).cast(pl.Utf8).fill_null("None"))
        .sort("records", descending=True)
    )


def species_per_image_table(df: pl.DataFrame) -> pl.DataFrame:
    frames = []
    for threshold in SPECIES_COUNT_THRESHOLDS:
        column = f"species_count_ge_{threshold:.2f}"
        data = (
            df.group_by(column)
            .len(name="records")
            .rename({column: "species_count"})
            .sort("species_count")
            .with_columns(pl.lit(threshold).alias("score_threshold"))
            .select(["score_threshold", "species_count", "records"])
        )
        frames.append(data)
    return pl.concat(frames)


def numeric_summary(series: pl.Series) -> dict[str, float | int | None]:
    clean = series.cast(pl.Float64, strict=False).drop_nulls()
    if clean.is_empty():
        return {"count": 0, "min": None, "p25": None, "median": None, "mean": None, "p75": None, "p90": None, "max": None}
    return {
        "count": int(clean.len()),
        "min": float(clean.min()),
        "p25": float(clean.quantile(0.25, interpolation="linear")),
        "median": float(clean.median()),
        "mean": float(clean.mean()),
        "p75": float(clean.quantile(0.75, interpolation="linear")),
        "p90": float(clean.quantile(0.90, interpolation="linear")),
        "max": float(clean.max()),
    }


def value_counts_dict(df: pl.DataFrame, column: str) -> dict[str, int]:
    return {str(row[column]): int(row["records"]) for row in value_count_table(df, column, column).to_dicts()}


def plot_data(df: pl.DataFrame) -> dict[str, list[object]]:
    return df.to_dict(as_series=False)


def write_figures(df: pl.DataFrame, output_dir: Path) -> list[tuple[str, str]]:
    figures = [
        ("Overview", overview_dashboard(df, output_dir)),
        ("Top 10 Predicted Species", top_species_plot(df, output_dir)),
        ("Top Families", top_families_plot(df, output_dir)),
        ("Occurrence Bins", occurrence_bin_plot(df, output_dir)),
        ("Bin Reasons", bin_reason_plot(df, output_dir)),
        ("Family By Occurrence Bin", family_occurrence_plot(df, output_dir)),
        ("Species Score Distribution", score_distribution_plot(df, output_dir)),
        ("Top-1 Versus Top-2 Margin", margin_distribution_plot(df, output_dir)),
        ("Species Count Per Image", species_per_image_plot(df, output_dir)),
        ("Top Species Score Spread", top_species_score_boxplot(df, output_dir)),
        ("Image Triage Categories", category_plot(df, output_dir)),
        ("Life Stage Mix", life_stage_plot(df, output_dir)),
        ("Records By Capture Month", month_heatmap(df, output_dir)),
        ("Global Prediction Footprint", map_scatter(df, output_dir)),
        ("Species Dominance Curve", dominance_curve(df, output_dir)),
        ("Pipeline Health", pipeline_health_plot(df, output_dir)),
    ]
    return figures


def overview_dashboard(df: pl.DataFrame, output_dir: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Papilio demoleus BioCLIP 2.5 Classification Overview", fontsize=22, y=0.98)

    metrics = [
        ("Records", f"{df.height:,}"),
        ("Top-1 species", f"{df['species_top1_scientific_name'].n_unique():,}"),
        ("Median species score", f"{df['species_top1_score'].median():.3f}"),
        (
            "Geo-tagged records",
            f"{df.filter(pl.col('latitude').is_not_null() & pl.col('longitude').is_not_null()).height:,}",
        ),
        (
            "Deleted temp images",
            f"{int(df.select(pl.col('image_deleted_after_classification').fill_null(False).sum()).item()):,}",
        ),
        ("Gold records", f"{df.filter(pl.col('occurrence_bin') == 'gold').height:,}"),
    ]
    axes[0, 0].axis("off")
    for index, (label, value) in enumerate(metrics):
        row, col = divmod(index, 2)
        axes[0, 0].text(0.05 + col * 0.48, 0.82 - row * 0.28, value, fontsize=26, weight="bold")
        axes[0, 0].text(0.05 + col * 0.48, 0.72 - row * 0.28, label, fontsize=13)

    top_species = value_count_table(df, "species_top1_scientific_name", "species").head(8).sort("records")
    sns.barplot(data=plot_data(top_species), y="species", x="records", ax=axes[0, 1], color="#0072B2")
    axes[0, 1].set_title("Leading predicted species")
    axes[0, 1].set_xlabel("Records")
    axes[0, 1].set_ylabel("")

    sns.histplot(x=df["species_top1_score"].to_list(), bins=35, ax=axes[1, 0], color="#009E73")
    axes[1, 0].set_title("Species certainty scores")
    axes[1, 0].set_xlabel("Top-1 score")
    axes[1, 0].set_ylabel("Records")

    bins = value_count_table(df, "occurrence_bin", "occurrence_bin").sort("records")
    sns.barplot(data=plot_data(bins), y="occurrence_bin", x="records", ax=axes[1, 1], color="#D55E00")
    axes[1, 1].set_title("Occurrence bins")
    axes[1, 1].set_xlabel("Records")
    axes[1, 1].set_ylabel("")

    return savefig(fig, output_dir / "overview_dashboard.png")


def top_species_plot(df: pl.DataFrame, output_dir: Path) -> str:
    data = value_count_table(df, "species_top1_scientific_name", "species").head(10).sort("records")
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(data=plot_data(data), y="species", x="records", ax=ax, color="#0072B2")
    ax.set_title("Top 10 Butterfly Species By Prediction Count")
    ax.set_xlabel("Predictions")
    ax.set_ylabel("")
    add_bar_labels(ax)
    return savefig(fig, output_dir / "top_10_species_predictions.png")


def top_families_plot(df: pl.DataFrame, output_dir: Path) -> str:
    data = value_count_table(df, "family_resolved", "family").head(10).sort("records")
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    colors = [PALETTE.get(family, "#666666") for family in data["family"].to_list()]
    sns.barplot(data=plot_data(data), y="family", x="records", ax=ax, palette=colors, hue="family", legend=False)
    ax.set_title("Top Families By Record Count")
    ax.set_xlabel("Records")
    ax.set_ylabel("")
    add_bar_labels(ax)
    return savefig(fig, output_dir / "top_family_counts.png")


def occurrence_bin_plot(df: pl.DataFrame, output_dir: Path) -> str:
    data = value_count_table(df, "occurrence_bin", "occurrence_bin").sort("records")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    colors = [PALETTE.get(bin_name, "#666666") for bin_name in data["occurrence_bin"].to_list()]
    sns.barplot(data=plot_data(data), y="occurrence_bin", x="records", ax=ax, palette=colors, hue="occurrence_bin", legend=False)
    ax.set_title("Occurrence Bin Mix")
    ax.set_xlabel("Records")
    ax.set_ylabel("")
    add_bar_labels(ax)
    return savefig(fig, output_dir / "occurrence_bin_mix.png")


def bin_reason_plot(df: pl.DataFrame, output_dir: Path) -> str:
    data = value_count_table(df, "bin_reason", "bin_reason").head(12).sort("records")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.barplot(data=plot_data(data), y="bin_reason", x="records", ax=ax, color="#56B4E9")
    ax.set_title("Top Bin Reasons")
    ax.set_xlabel("Records")
    ax.set_ylabel("")
    add_bar_labels(ax)
    return savefig(fig, output_dir / "bin_reason_counts.png")


def family_occurrence_plot(df: pl.DataFrame, output_dir: Path) -> str:
    top_families = value_count_table(df, "family_resolved", "family").head(8)["family"].to_list()
    grouped = (
        df.filter(pl.col("family_resolved").is_in(top_families))
        .group_by(["family_resolved", "occurrence_bin"])
        .len(name="records")
    )
    records_by_pair = {
        (row["family_resolved"], row["occurrence_bin"]): int(row["records"])
        for row in grouped.to_dicts()
    }
    bin_names = grouped["occurrence_bin"].drop_nulls().unique().sort().to_list()
    fig, ax = plt.subplots(figsize=(11, 6))
    left = [0] * len(top_families)
    for bin_name in bin_names:
        values = [records_by_pair.get((family, bin_name), 0) for family in top_families]
        ax.barh(
            top_families,
            values,
            left=left,
            label=bin_name,
            color=PALETTE.get(bin_name, "#666666"),
        )
        left = [current + value for current, value in zip(left, values, strict=True)]
    ax.invert_yaxis()
    ax.set_title("Top Families Split By Occurrence Bin")
    ax.set_xlabel("Records")
    ax.set_ylabel("")
    ax.legend(title="Occurrence bin", loc="lower right")
    return savefig(fig, output_dir / "family_by_occurrence_bin.png")


def score_distribution_plot(df: pl.DataFrame, output_dir: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    data = df.select(["species_top1_score", "occurrence_bin"])
    bin_palette = {k: PALETTE.get(k, "#666666") for k in data["occurrence_bin"].drop_nulls().unique().to_list()}
    sns.histplot(
        data=plot_data(data),
        x="species_top1_score",
        hue="occurrence_bin",
        bins=40,
        multiple="stack",
        palette=bin_palette,
        ax=axes[0],
    )
    axes[0].set_title("Species Top-1 Score Distribution")
    axes[0].set_xlabel("BioCLIP species top-1 score")
    axes[0].set_ylabel("Records")

    sns.ecdfplot(data=plot_data(data), x="species_top1_score", hue="occurrence_bin", ax=axes[1])
    axes[1].set_title("Cumulative Certainty By Bin")
    axes[1].set_xlabel("BioCLIP species top-1 score")
    axes[1].set_ylabel("Cumulative share")
    return savefig(fig, output_dir / "species_score_distribution.png")


def margin_distribution_plot(df: pl.DataFrame, output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.histplot(x=df["species_top1_top2_margin"].to_list(), bins=50, color="#56B4E9", ax=ax)
    ax.axvline(df["species_top1_top2_margin"].median(), color="#D55E00", linewidth=2, label="median")
    ax.set_title("Ambiguity: Top-1 Score Minus Top-2 Score")
    ax.set_xlabel("Top-1 minus top-2 species score")
    ax.set_ylabel("Records")
    ax.legend()
    return savefig(fig, output_dir / "top1_top2_margin_distribution.png")


def species_per_image_plot(df: pl.DataFrame, output_dir: Path) -> str:
    data = species_per_image_table(df).with_columns(
        pl.col("score_threshold")
        .map_elements(lambda value: f">= {value:.2f}", return_dtype=pl.Utf8)
        .alias("score_threshold")
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=plot_data(data), x="species_count", y="records", hue="score_threshold", ax=ax)
    ax.set_title("How Many Species Candidates Appear In One Image?")
    ax.set_xlabel("Species candidates in the top-k list at score threshold")
    ax.set_ylabel("Records")
    return savefig(fig, output_dir / "species_candidates_per_image.png")


def top_species_score_boxplot(df: pl.DataFrame, output_dir: Path) -> str:
    top_species = value_count_table(df, "species_top1_scientific_name", "species").head(12)["species"].to_list()
    data = df.filter(pl.col("species_top1_scientific_name").is_in(top_species)).select(
        ["species_top1_scientific_name", "species_top1_score", "occurrence_bin"]
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(
        data=plot_data(data),
        y="species_top1_scientific_name",
        x="species_top1_score",
        hue="occurrence_bin",
        order=top_species,
        fliersize=1.5,
        ax=ax,
    )
    ax.set_title("Confidence Spread For Top Predicted Species")
    ax.set_xlabel("Species top-1 score")
    ax.set_ylabel("")
    ax.legend(title="Occurrence bin", loc="lower right")
    return savefig(fig, output_dir / "top_species_score_spread.png")


def category_plot(df: pl.DataFrame, output_dir: Path) -> str:
    data = value_count_table(df, "image_category", "image_category").sort("records")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    sns.barplot(data=plot_data(data), y="image_category", x="records", ax=ax, color="#009E73")
    ax.set_title("Image Category Mix")
    ax.set_xlabel("Records")
    ax.set_ylabel("")
    add_bar_labels(ax)
    return savefig(fig, output_dir / "image_category_mix.png")


def life_stage_plot(df: pl.DataFrame, output_dir: Path) -> str:
    data = value_count_table(df, "life_stage", "life_stage").sort("records")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    sns.barplot(data=plot_data(data), y="life_stage", x="records", ax=ax, color="#CC79A7")
    ax.set_title("Life Stage Mix")
    ax.set_xlabel("Records")
    ax.set_ylabel("")
    add_bar_labels(ax)
    return savefig(fig, output_dir / "life_stage_mix.png")


def month_heatmap(df: pl.DataFrame, output_dir: Path) -> str:
    data = (
        df.group_by(["year", "month"])
        .len(name="records")
        .pivot(index="year", on="month", values="records")
        .fill_null(0)
        .sort("year")
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    month_columns = [column for column in data.columns if column != "year"]
    sns.heatmap(
        data.select(month_columns).to_numpy(),
        xticklabels=month_columns,
        yticklabels=data["year"].to_list(),
        cmap="YlGnBu",
        linewidths=0.2,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Predicted Records By Capture Year And Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    return savefig(fig, output_dir / "capture_year_month_heatmap.png")


def map_scatter(df: pl.DataFrame, output_dir: Path) -> str:
    geo = df.filter(pl.col("longitude").is_not_null() & pl.col("latitude").is_not_null())
    fig, ax = plt.subplots(figsize=(12, 6.5))
    if geo.is_empty():
        ax.text(0.5, 0.5, "No geotagged records", ha="center", va="center", fontsize=18)
    else:
        sns.scatterplot(
            data=plot_data(geo.select(["longitude", "latitude", "family_resolved", "species_top1_score"])),
            x="longitude",
            y="latitude",
            hue="family_resolved",
            size="species_top1_score",
            sizes=(8, 38),
            alpha=0.45,
            linewidth=0,
            palette={family: PALETTE.get(family, "#666666") for family in geo["family_resolved"].unique().to_list()},
            ax=ax,
        )
    ax.set_title("Global Geotag Footprint By Predicted Family")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    return savefig(fig, output_dir / "global_prediction_footprint.png")


def dominance_curve(df: pl.DataFrame, output_dir: Path) -> str:
    counts = value_count_table(df, "species_top1_scientific_name", "species")
    total = counts["records"].sum()
    counts = counts.with_columns(
        pl.int_range(1, pl.len() + 1).alias("rank"),
        (pl.col("records").cum_sum() / total).alias("cumulative_share"),
    )
    fig, ax = plt.subplots(figsize=(10, 5.8))
    sns.lineplot(data=plot_data(counts), x="rank", y="cumulative_share", ax=ax, color="#D55E00")
    ax.axhline(0.8, color="#555555", linestyle="--", linewidth=1)
    ax.set_title("Species Dominance Curve")
    ax.set_xlabel("Species rank by prediction count")
    ax.set_ylabel("Cumulative share of records")
    ax.set_ylim(0, 1.01)
    return savefig(fig, output_dir / "species_dominance_curve.png")


def pipeline_health_plot(df: pl.DataFrame, output_dir: Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    panels = [
        ("Classification status", "classification_status"),
        ("Image downloaded", "image_downloaded"),
        ("Temp image deleted", "image_deleted_after_classification"),
    ]
    for ax, (title, column) in zip(axes, panels, strict=True):
        data = value_count_table(df, column, column).sort("records")
        sns.barplot(data=plot_data(data), y=column, x="records", ax=ax, color="#0072B2")
        ax.set_title(title)
        ax.set_xlabel("Records")
        ax.set_ylabel("")
        add_bar_labels(ax)
    return savefig(fig, output_dir / "pipeline_health.png")


def write_pdf_deck(figures: list[tuple[str, str]], output_dir: Path) -> None:
    pdf_path = output_dir / "bioclip25_species_visual_report_deck.pdf"
    with PdfPages(pdf_path) as pdf:
        for title, filename in figures:
            fig, ax = plt.subplots(figsize=(13.33, 7.5))
            image = plt.imread(output_dir / filename)
            ax.imshow(image)
            ax.set_title(title, fontsize=18, pad=12)
            ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def add_bar_labels(ax: plt.Axes) -> None:
    for container in ax.containers:
        ax.bar_label(container, padding=4, fontsize=10)


def savefig(fig: plt.Figure, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


def write_html_report(
    df: pl.DataFrame,
    summary: dict[str, object],
    figures: list[tuple[str, str]],
    predictions_path: Path,
    candidates_path: Path,
    output_dir: Path,
) -> None:
    top_species_rows = table_rows(summary["top_10_species"], ["species", "records"])
    top_family_rows = table_rows(summary["top_10_families"], ["family", "records"])
    score_stats = summary["score_stats"]
    assert isinstance(score_stats, dict)
    figure_html = "\n".join(
        f"<section><h2>{html.escape(title)}</h2><img src=\"{html.escape(filename)}\" alt=\"{html.escape(title)}\"></section>"
        for title, filename in figures
    )
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>BioCLIP 2.5 Species Visual Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #202020; line-height: 1.45; }}
    header {{ max-width: 1160px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; margin: 22px 0; max-width: 1160px; }}
    .metric {{ border: 1px solid #ddd; border-radius: 6px; padding: 12px; background: #fafafa; }}
    .metric b {{ display: block; font-size: 25px; margin-bottom: 4px; }}
    .tables {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 24px; max-width: 1160px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f2f2f2; }}
    section {{ margin: 30px 0 42px; max-width: 1240px; }}
    img {{ width: 100%; max-width: 1200px; height: auto; border: 1px solid #ddd; }}
    code {{ background: #f5f5f5; padding: 2px 4px; border-radius: 3px; }}
    .note {{ max-width: 1160px; color: #444; }}
  </style>
</head>
<body>
  <header>
    <h1>BioCLIP 2.5 Species Visual Report</h1>
    <p class="note">Source predictions: <code>{html.escape(str(predictions_path))}</code></p>
    <p class="note">Family lookup: <code>{html.escape(str(candidates_path))}</code>. BioCLIP species predictions are screening evidence, not taxonomic validation.</p>
    <div class="metrics">
      <div class="metric"><b>{summary["records"]:,}</b>records</div>
      <div class="metric"><b>{summary["unique_top1_species"]:,}</b>top-1 species</div>
      <div class="metric"><b>{summary["families"]:,}</b>family groups</div>
      <div class="metric"><b>{summary["records_with_geo"]:,}</b>records with geo</div>
      <div class="metric"><b>{summary["records_with_event_date"]:,}</b>records with event date</div>
      <div class="metric"><b>{summary["downloaded_images_deleted"]:,}</b>temp images deleted</div>
      <div class="metric"><b>{score_stats["median"]:.3f}</b>median species score</div>
      <div class="metric"><b>{summary["unresolved_family_records"]:,}</b>unresolved-family records</div>
    </div>
  </header>
  <div class="tables">
    <div>
      <h2>Top 10 Species</h2>
      <table><thead><tr><th>Species</th><th>Records</th></tr></thead><tbody>{top_species_rows}</tbody></table>
    </div>
    <div>
      <h2>Top Families</h2>
      <table><thead><tr><th>Family</th><th>Records</th></tr></thead><tbody>{top_family_rows}</tbody></table>
    </div>
  </div>
  {figure_html}
</body>
</html>
"""
    (output_dir / "bioclip25_species_visual_report.html").write_text(body, encoding="utf-8")


def table_rows(rows: object, columns: list[str]) -> str:
    assert isinstance(rows, list)
    rendered = []
    for row in rows:
        assert isinstance(row, dict)
        cells = "".join(f"<td>{html.escape(str(row[column]))}</td>" for column in columns)
        rendered.append(f"<tr>{cells}</tr>")
    return "\n".join(rendered)


if __name__ == "__main__":
    main()
