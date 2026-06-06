from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from flickr_bio_occurrence.storage.parquet_io import write_parquet_dataset


EXPORT_CSV_BY_DEFAULT = False


@dataclass(frozen=True)
class DwcExportOutputs:
    parquet_paths: list[Path]
    csv_path: Path | None


def export_dwc_records(
    frame: pl.DataFrame,
    output_dir: str | Path,
    *,
    output_csv: bool = EXPORT_CSV_BY_DEFAULT,
) -> DwcExportOutputs:
    """Export retained compatibility Darwin Core rows.

    Removal condition: delete this shim when Darwin Core compatibility tests
    and downstream public API expectations are retired. The active Phase 7
    image-triage pipeline does not publish validated Darwin Core records.
    """
    parquet_paths = write_parquet_dataset(frame, Path(output_dir) / "dwc_occurrence")
    csv_path = None
    if output_csv:
        csv_path = Path(output_dir) / "dwc_occurrence.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_csv(csv_path)
    return DwcExportOutputs(parquet_paths=parquet_paths, csv_path=csv_path)
