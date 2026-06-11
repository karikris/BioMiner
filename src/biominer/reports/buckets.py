from __future__ import annotations

from pathlib import Path

import polars as pl

from biominer.storage.parquet import write_bucket_views


def export_bucket_views(bucketed_records: str | Path, output_dir: str | Path) -> dict[str, str]:
    return write_bucket_views(pl.read_parquet(bucketed_records), output_dir)
