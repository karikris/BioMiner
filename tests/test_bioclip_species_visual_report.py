from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest


def load_report_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_bioclip_species_visual_report.py"
    spec = importlib.util.spec_from_file_location("generate_bioclip_species_visual_report", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_topk_summary_counts_thresholded_species_per_image() -> None:
    report = load_report_module()
    topk = [
        {"label": "a photo of Papilio demoleus", "score": 0.61},
        {"label": "a photo of Papilio polytes", "score": 0.22},
        {"label": "a photo of Idea leuconoe", "score": 0.04},
        {"label": "a photo of Papilio demoleus", "score": 0.02},
    ]

    summary = report.topk_summary(topk)

    assert summary["species_topk_count"] == 3
    assert summary["species_top2_score"] == 0.22
    assert summary["species_top1_top2_margin"] == 0.39
    assert summary["species_count_ge_0.01"] == 3
    assert summary["species_count_ge_0.05"] == 2
    assert summary["species_count_ge_0.10"] == 2


def test_report_module_does_not_import_pandas() -> None:
    report = load_report_module()

    assert not hasattr(report, "pd")


def test_numeric_summary_reports_distribution_points() -> None:
    report = load_report_module()

    summary = report.numeric_summary(pl.Series([0.1, 0.4, 0.8, 1.0]))

    assert summary["count"] == 4
    assert summary["min"] == 0.1
    assert summary["median"] == pytest.approx(0.6)
    assert summary["max"] == 1.0


def test_apply_filters_keeps_species_and_drops_excluded_categories() -> None:
    report = load_report_module()
    df = pl.DataFrame(
        [
            {"species_top1_scientific_name": "Papilio demoleus", "image_category": "adult_butterfly"},
            {"species_top1_scientific_name": "Papilio demoleus", "image_category": "artwork"},
            {"species_top1_scientific_name": "Papilio polytes", "image_category": "adult_butterfly"},
        ]
    )

    filtered, result = report.apply_filters(
        df,
        species="Papilio demoleus",
        excluded_image_categories=("artwork", "museum_specimen", "object_or_product"),
    )

    assert filtered.height == 1
    assert filtered.row(0, named=True)["image_category"] == "adult_butterfly"
    assert result.rows_before == 3
    assert result.rows_after == 1
    assert result.rows_dropped == 2


def test_normalize_reason_labels_renames_legacy_not_target_species() -> None:
    report = load_report_module()
    df = pl.DataFrame(
        [
            {
                "bin_reason": "not_target_species",
                "triage_reason": "not_target_species",
                "publication_state_reason": "not_target_species",
            }
        ]
    )

    normalized = report.normalize_reason_labels(df)

    row = normalized.row(0, named=True)
    assert row["bin_reason"] == "below_50"
    assert row["triage_reason"] == "below_50"
    assert row["publication_state_reason"] == "below_50"
