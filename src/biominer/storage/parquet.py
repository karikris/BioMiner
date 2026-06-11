from __future__ import annotations

from pathlib import Path

import polars as pl


CANONICAL_BUCKETED_RECORDS = "bucketed_records.parquet"
BUCKET_VIEW_FILES = {
    "gold": "gold_records.parquet",
    "silver": "silver_records.parquet",
    "bronze": "bronze_records.parquet",
    "bin": "bin_records.parquet",
}


def write_parquet(frame: pl.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    frame.write_parquet(tmp)
    tmp.replace(output)
    return output


def write_bucket_views(frame: pl.DataFrame, output_dir: str | Path) -> dict[str, str]:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for bucket, filename in BUCKET_VIEW_FILES.items():
        path = base / filename
        view = frame.filter(pl.col("occurrence_bin") == bucket) if "occurrence_bin" in frame.columns else pl.DataFrame()
        write_parquet(view, path)
        outputs[bucket] = str(path)
    return outputs
