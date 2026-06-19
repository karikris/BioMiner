from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl


def load_validation_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_bioclip_species_validation.py"
    spec = importlib.util.spec_from_file_location("evaluate_bioclip_species_validation", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validation_report_counts_precision_by_bucket(tmp_path) -> None:
    report = load_validation_module()
    predictions = tmp_path / "predictions.parquet"
    reviewed = tmp_path / "reviewed.parquet"
    output = tmp_path / "metrics.json"

    pl.DataFrame(
        [
            {
                "flickr_photo_id": "1",
                "occurrence_bin": "gold",
                "species_top1_scientific_name": "Papilio demoleus",
                "species_top1_score": 0.91,
            },
            {
                "flickr_photo_id": "2",
                "occurrence_bin": "gold",
                "species_top1_scientific_name": "Papilio machaon",
                "species_top1_score": 0.88,
            },
            {
                "flickr_photo_id": "3",
                "occurrence_bin": "bronze",
                "species_top1_scientific_name": "Papilio demoleus",
                "species_top1_score": 0.42,
            },
        ]
    ).write_parquet(predictions)
    pl.DataFrame(
        [
            {"flickr_photo_id": "1", "reviewed_species": "Papilio demoleus"},
            {"flickr_photo_id": "2", "reviewed_species": "Papilio demoleus"},
            {"flickr_photo_id": "3", "reviewed_species": "Papilio demoleus"},
        ]
    ).write_parquet(reviewed)

    metrics = report.evaluate_validation(predictions_path=predictions, reviewed_path=reviewed, output_path=output)

    assert metrics["rows_evaluated"] == 3
    assert metrics["bucket_metrics"]["gold"]["rows"] == 2
    assert metrics["bucket_metrics"]["gold"]["species_precision"] == 0.5
    assert output.exists()
