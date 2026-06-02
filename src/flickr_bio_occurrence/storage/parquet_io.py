from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl


def write_parquet_dataset(
    frame: pl.DataFrame,
    output_dir: str | Path,
    *,
    partition_by: Iterable[str] | None = None,
    file_name: str = "part-00000.parquet",
) -> list[Path]:
    output_path = Path(output_dir)
    partition_columns = list(partition_by or [])
    written: list[Path] = []
    if not partition_columns:
        output_path.mkdir(parents=True, exist_ok=True)
        target = output_path / file_name
        frame.write_parquet(target)
        return [target]

    for keys, group in frame.group_by(partition_columns, maintain_order=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        target_dir = output_path
        for column, value in zip(partition_columns, key_values, strict=True):
            target_dir = target_dir / f"{column}={value}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / file_name
        group.write_parquet(target)
        written.append(target)
    return written
