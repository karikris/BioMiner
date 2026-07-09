from __future__ import annotations

import json
from pathlib import Path

import pytest
import polars as pl

from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION
from biominer.evaluation.charts import (
    CALIBRATION_RELIABILITY_CHART_FILE,
    FAMILY_CONFUSION_CHART_FILE,
    REVIEW_REASON_COUNTS_CHART_FILE,
    SPECIES_ACCURACY_BY_FAMILY_CHART_FILE,
)
from biominer.evaluation.reports import (
    CALIBRATION_BINS_FILE,
    EVALUATION_METRICS_FILE,
    EVALUATION_SUMMARY_FILE,
    FAMILY_CONFUSION_FILE,
    REVIEW_ERROR_EXAMPLES_FILE,
    SPECIES_CONFUSION_FILE,
    write_evaluation_report,
    write_evaluation_report_to_storage,
)


def test_write_evaluation_report_writes_json_parquet_and_markdown(tmp_path) -> None:
    paths = write_evaluation_report(
        object_scores=pl.DataFrame([_prediction()]),
        reviewed_labels=pl.DataFrame([_label()]),
        output_dir=tmp_path,
        run_manifest={"run_id": "run-1"},
    )

    assert sorted(paths) == [
        "calibration_bins",
        "family_confusion_matrix",
        "metrics",
        "review_error_examples",
        "species_confusion_matrix",
        "summary",
    ]
    for filename in (
        EVALUATION_METRICS_FILE,
        FAMILY_CONFUSION_FILE,
        SPECIES_CONFUSION_FILE,
        EVALUATION_SUMMARY_FILE,
        CALIBRATION_BINS_FILE,
        REVIEW_ERROR_EXAMPLES_FILE,
    ):
        assert (tmp_path / filename).exists()

    metrics = json.loads((tmp_path / EVALUATION_METRICS_FILE).read_text(encoding="utf-8"))
    family_confusion = pl.read_parquet(tmp_path / FAMILY_CONFUSION_FILE)
    species_confusion = pl.read_parquet(tmp_path / SPECIES_CONFUSION_FILE)
    calibration_bins = pl.read_parquet(tmp_path / CALIBRATION_BINS_FILE)
    review_errors = pl.read_parquet(tmp_path / REVIEW_ERROR_EXAMPLES_FILE)
    markdown = (tmp_path / EVALUATION_SUMMARY_FILE).read_text(encoding="utf-8")

    assert metrics["schema_version"] == "evaluation_metrics_v1"
    assert metrics["run"]["run_id"] == "run-1"
    assert metrics["metrics"]["species_top1_accuracy"] == 1.0
    assert metrics["metrics"]["species_top20_recall"] == 1.0
    assert metrics["calibration"]["sample_count"] == 1
    assert family_confusion.to_dicts()[0]["count"] == 1
    assert species_confusion.to_dicts()[0]["predicted_name"] == "Papilio demoleus"
    assert calibration_bins.height == 10
    assert review_errors.is_empty()
    assert "Family top1 accuracy" in markdown
    assert "Species top20 recall" in markdown
    assert "Species MRR" in markdown


def test_write_evaluation_report_can_write_optional_charts(tmp_path) -> None:
    pytest.importorskip("matplotlib")

    paths = write_evaluation_report(
        object_scores=pl.DataFrame([_prediction(), _prediction(source_id="2", detection_id="d2", score=0.25)]),
        reviewed_labels=pl.DataFrame([_label(), _label(source_id="2", detection_id="d2", taxon_key="gbif:200")]),
        output_dir=tmp_path,
        write_charts=True,
    )

    assert paths["family_confusion_chart"] == str(tmp_path / FAMILY_CONFUSION_CHART_FILE)
    assert paths["species_accuracy_by_family_chart"] == str(tmp_path / SPECIES_ACCURACY_BY_FAMILY_CHART_FILE)
    assert paths["calibration_reliability_chart"] == str(tmp_path / CALIBRATION_RELIABILITY_CHART_FILE)
    assert paths["review_reason_counts_chart"] == str(tmp_path / REVIEW_REASON_COUNTS_CHART_FILE)
    metrics = json.loads((tmp_path / EVALUATION_METRICS_FILE).read_text(encoding="utf-8"))
    assert metrics["artifacts"]["species_accuracy_by_family_chart"] == paths["species_accuracy_by_family_chart"]
    for key in (
        "family_confusion_chart",
        "species_accuracy_by_family_chart",
        "calibration_reliability_chart",
        "review_reason_counts_chart",
    ):
        assert Path(paths[key]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_write_evaluation_report_handles_empty_inputs(tmp_path) -> None:
    write_evaluation_report(
        object_scores=pl.DataFrame(),
        reviewed_labels=pl.DataFrame(),
        output_dir=tmp_path,
    )

    metrics = json.loads((tmp_path / EVALUATION_METRICS_FILE).read_text(encoding="utf-8"))
    markdown = (tmp_path / EVALUATION_SUMMARY_FILE).read_text(encoding="utf-8")

    assert metrics["metrics"]["evaluated_objects"] == 0
    assert metrics["metrics"]["missing_prediction_count"] == 0
    assert pl.read_parquet(tmp_path / FAMILY_CONFUSION_FILE).is_empty()
    assert pl.read_parquet(tmp_path / SPECIES_CONFUSION_FILE).is_empty()
    assert "No reviewed object labels were available for evaluation." in markdown


def test_write_evaluation_report_to_storage_writes_all_artifacts() -> None:
    storage = _MemoryStorage()
    output_dir = "s3://biominer/reports/evaluation/run-1"

    paths = write_evaluation_report_to_storage(
        object_scores=pl.DataFrame([_prediction()]),
        reviewed_labels=pl.DataFrame([_label()]),
        output_dir=output_dir,
        storage=storage,
        run_manifest={"run_id": "run-1"},
    )

    assert paths["metrics"] == f"{output_dir}/{EVALUATION_METRICS_FILE}"
    assert paths["summary"] == f"{output_dir}/{EVALUATION_SUMMARY_FILE}"
    assert storage.json_payloads[paths["metrics"]]["metrics"]["species_top1_accuracy"] == 1.0
    assert "Family top1 accuracy" in storage.text_payloads[paths["summary"]]
    assert storage.parquet_payloads[paths["family_confusion_matrix"]].height == 1
    assert storage.parquet_payloads[paths["species_confusion_matrix"]].height == 1
    assert storage.parquet_payloads[paths["calibration_bins"]].height == 10
    assert storage.parquet_payloads[paths["review_error_examples"]].is_empty()


def test_write_evaluation_report_to_storage_rejects_charts() -> None:
    with pytest.raises(RuntimeError, match="--write-charts"):
        write_evaluation_report_to_storage(
            object_scores=pl.DataFrame([_prediction()]),
            reviewed_labels=pl.DataFrame([_label()]),
            output_dir="s3://biominer/reports/evaluation/run-1",
            storage=_MemoryStorage(),
            write_charts=True,
        )


def _prediction(*, source_id: str = "1", detection_id: str = "d1", score: float = 0.91) -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": source_id,
        "detection_id": detection_id,
        "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        "taxonomy_table_version": "taxonomy-v1",
        "model_id": "bioclip",
        "model_checkpoint": "checkpoint-a",
        "family_top3": ["Papilionidae", "Nymphalidae", "Pieridae"],
        "family_top3_accepted_taxon_keys": ["gbif:9417", "gbif:7017", "gbif:5481"],
        "selected_family": "Papilionidae",
        "selected_family_key": "gbif:9417",
        "species_top1_scientific_name": "Papilio demoleus",
        "species_top1_accepted_taxon_key": "gbif:100",
        "accepted_taxon_key": "gbif:100",
        "species_top1_score": score,
        "species_top5": ["Papilio demoleus", "Papilio machaon"],
        "species_top5_accepted_taxon_keys": ["gbif:100", "gbif:200"],
        "species_top20": ["Papilio demoleus", "Papilio machaon"],
        "species_top20_accepted_taxon_keys": ["gbif:100", "gbif:200"],
    }


def _label(
    *,
    source_id: str = "1",
    detection_id: str = "d1",
    taxon_key: str = "gbif:100",
    scientific_name: str = "Papilio demoleus",
) -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": source_id,
        "detection_id": detection_id,
        "crop_hash": f"sha256:{detection_id}",
        "label_level": "species",
        "is_butterfly": True,
        "accepted_taxon_key": taxon_key,
        "scientific_name": scientific_name,
        "family_key": "gbif:9417",
        "family": "Papilionidae",
        "genus_key": "gbif:90",
        "genus": "Papilio",
        "label_source": "fixture",
        "reviewer_id": "reviewer-a",
        "reviewed_at": "2026-07-10T00:00:00Z",
        "review_confidence": "high",
        "review_notes": "synthetic",
    }


class _MemoryStorage:
    def __init__(self) -> None:
        self.json_payloads: dict[str, dict[str, object]] = {}
        self.parquet_payloads: dict[str, pl.DataFrame] = {}
        self.text_payloads: dict[str, str] = {}

    def write_json(self, uri: str, payload: dict[str, object]) -> str:
        self.json_payloads[uri] = payload
        return uri

    def read_json(self, uri: str) -> dict[str, object]:
        return self.json_payloads[uri]

    def write_parquet_shard(self, uri: str, frame: pl.DataFrame) -> str:
        self.parquet_payloads[uri] = frame
        return uri

    def read_parquet(self, uri: str) -> pl.DataFrame:
        return self.parquet_payloads[uri]

    def write_text(self, uri: str, text: str, *, encoding: str = "utf-8") -> str:
        self.text_payloads[uri] = text
        return uri

    def read_text(self, uri: str, *, encoding: str = "utf-8") -> str:
        return self.text_payloads[uri]

    def exists(self, uri: str) -> bool:
        return uri in self.json_payloads or uri in self.parquet_payloads or uri in self.text_payloads
