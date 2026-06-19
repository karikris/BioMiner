from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def evaluate_validation(*, predictions_path: Path, reviewed_path: Path, output_path: Path) -> dict[str, object]:
    predictions = pl.read_parquet(predictions_path)
    reviewed = pl.read_parquet(reviewed_path)
    joined = predictions.join(reviewed, on="flickr_photo_id", how="inner").with_columns(
        (pl.col("species_top1_scientific_name") == pl.col("reviewed_species")).alias("species_correct")
    )
    bucket_metrics = {}
    for row in (
        joined.group_by("occurrence_bin")
        .agg(
            pl.len().alias("rows"),
            pl.col("species_correct").mean().alias("species_precision"),
            pl.col("species_top1_score").mean().alias("mean_species_score"),
        )
        .to_dicts()
    ):
        bucket_metrics[str(row["occurrence_bin"])] = {
            "rows": int(row["rows"]),
            "species_precision": float(row["species_precision"]),
            "mean_species_score": float(row["mean_species_score"]) if row["mean_species_score"] is not None else None,
        }
    metrics = {
        "rows_evaluated": joined.height,
        "bucket_metrics": bucket_metrics,
    }
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BioCLIP species predictions against reviewed labels.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--reviewed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    metrics = evaluate_validation(predictions_path=args.predictions, reviewed_path=args.reviewed, output_path=args.output)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
