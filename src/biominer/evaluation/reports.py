from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION
from biominer.evaluation.calibration import expected_calibration_error
from biominer.evaluation.metrics import (
    evaluate_hierarchical_predictions,
    family_confusion_matrix,
    species_confusion_matrix,
)
from biominer.storage.cloud import CloudStorage
from biominer.storage.parquet import write_parquet
from biominer.storage.uri import join_uri


EVALUATION_METRICS_FILE = "evaluation_metrics.json"
FAMILY_CONFUSION_FILE = "family_confusion_matrix.parquet"
SPECIES_CONFUSION_FILE = "species_confusion_matrix.parquet"
EVALUATION_SUMMARY_FILE = "evaluation_summary.md"
CALIBRATION_BINS_FILE = "calibration_bins.parquet"
REVIEW_ERROR_EXAMPLES_FILE = "review_error_examples.parquet"

CALIBRATION_BIN_SCHEMA: dict[str, pl.DataType] = {
    "bin_index": pl.Int64,
    "lower": pl.Float64,
    "upper": pl.Float64,
    "count": pl.Int64,
    "avg_confidence": pl.Float64,
    "accuracy": pl.Float64,
    "gap": pl.Float64,
    "weight": pl.Float64,
}

REVIEW_ERROR_EXAMPLE_SCHEMA: dict[str, pl.DataType] = {
    "source": pl.String,
    "flickr_photo_id": pl.String,
    "detection_id": pl.String,
    "expected_key": pl.String,
    "expected_name": pl.String,
    "predicted_key": pl.String,
    "predicted_name": pl.String,
    "error_type": pl.String,
}


@dataclass(frozen=True)
class EvaluationReportArtifacts:
    paths: dict[str, str]
    metrics_payload: dict[str, object]
    family_confusion: pl.DataFrame
    species_confusion: pl.DataFrame
    calibration_bins: pl.DataFrame
    review_error_examples: pl.DataFrame
    summary_markdown: str


def write_evaluation_report(
    *,
    object_scores: pl.DataFrame,
    reviewed_labels: pl.DataFrame,
    output_dir: str | Path,
    run_manifest: dict[str, object] | None = None,
) -> dict[str, str]:
    artifacts = build_evaluation_report_artifacts(
        object_scores=object_scores,
        reviewed_labels=reviewed_labels,
        output_dir=output_dir,
        run_manifest=run_manifest,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    paths = {key: Path(path) for key, path in artifacts.paths.items()}
    write_parquet(artifacts.family_confusion, paths["family_confusion_matrix"])
    write_parquet(artifacts.species_confusion, paths["species_confusion_matrix"])
    write_parquet(artifacts.calibration_bins, paths["calibration_bins"])
    write_parquet(artifacts.review_error_examples, paths["review_error_examples"])
    paths["metrics"].write_text(json.dumps(artifacts.metrics_payload, indent=2, sort_keys=True), encoding="utf-8")
    paths["summary"].write_text(artifacts.summary_markdown, encoding="utf-8")
    return artifacts.paths


def write_evaluation_report_to_storage(
    *,
    object_scores: pl.DataFrame,
    reviewed_labels: pl.DataFrame,
    output_dir: str | Path,
    storage: CloudStorage,
    run_manifest: dict[str, object] | None = None,
) -> dict[str, str]:
    artifacts = build_evaluation_report_artifacts(
        object_scores=object_scores,
        reviewed_labels=reviewed_labels,
        output_dir=output_dir,
        run_manifest=run_manifest,
    )
    storage.write_parquet_shard(artifacts.paths["family_confusion_matrix"], artifacts.family_confusion)
    storage.write_parquet_shard(artifacts.paths["species_confusion_matrix"], artifacts.species_confusion)
    storage.write_parquet_shard(artifacts.paths["calibration_bins"], artifacts.calibration_bins)
    storage.write_parquet_shard(artifacts.paths["review_error_examples"], artifacts.review_error_examples)
    storage.write_json(artifacts.paths["metrics"], artifacts.metrics_payload)
    storage.write_text(artifacts.paths["summary"], artifacts.summary_markdown)
    return artifacts.paths


def build_evaluation_report_artifacts(
    *,
    object_scores: pl.DataFrame,
    reviewed_labels: pl.DataFrame,
    output_dir: str | Path,
    run_manifest: dict[str, object] | None = None,
) -> EvaluationReportArtifacts:
    metrics = evaluate_hierarchical_predictions(object_scores=object_scores, reviewed_labels=reviewed_labels)
    family_confusion = family_confusion_matrix(object_scores=object_scores, reviewed_labels=reviewed_labels)
    species_confusion = species_confusion_matrix(object_scores=object_scores, reviewed_labels=reviewed_labels)
    scored_for_calibration = _with_species_correct_column(object_scores, reviewed_labels)
    calibration = expected_calibration_error(
        predictions=scored_for_calibration,
        labels=reviewed_labels,
        score_column="species_top1_score",
        correct_column="species_top1_correct",
        bins=10,
    )
    review_errors = _review_error_examples(scored_for_calibration)

    paths: dict[str, str] = {
        "metrics": join_uri(output_dir, EVALUATION_METRICS_FILE),
        "family_confusion_matrix": join_uri(output_dir, FAMILY_CONFUSION_FILE),
        "species_confusion_matrix": join_uri(output_dir, SPECIES_CONFUSION_FILE),
        "summary": join_uri(output_dir, EVALUATION_SUMMARY_FILE),
        "calibration_bins": join_uri(output_dir, CALIBRATION_BINS_FILE),
        "review_error_examples": join_uri(output_dir, REVIEW_ERROR_EXAMPLES_FILE),
    }
    metrics_payload = {
        "schema_version": "evaluation_metrics_v1",
        "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        "run": _run_metadata(run_manifest or {}, object_scores),
        "metrics": metrics,
        "calibration": {key: value for key, value in calibration.items() if key != "bins"},
        "warnings": _warnings(metrics=metrics, calibration=calibration),
        "artifacts": paths,
    }

    return EvaluationReportArtifacts(
        paths=paths,
        metrics_payload=metrics_payload,
        family_confusion=family_confusion,
        species_confusion=species_confusion,
        calibration_bins=_calibration_bins_frame(calibration),
        review_error_examples=review_errors,
        summary_markdown=evaluation_summary_markdown(metrics_payload, family_confusion, species_confusion),
    )


def evaluation_summary_markdown(
    metrics_payload: Mapping[str, Any],
    family_confusion: pl.DataFrame,
    species_confusion: pl.DataFrame,
) -> str:
    metrics = _dict(metrics_payload.get("metrics"))
    calibration = _dict(metrics_payload.get("calibration"))
    run = _dict(metrics_payload.get("run"))
    warnings = [str(item) for item in metrics_payload.get("warnings") or []]
    warning_lines = [f"- {warning}" for warning in warnings] if warnings else ["- none"]
    lines = [
        "# Hierarchical Classification Evaluation",
        "",
        "## Run",
        "",
        f"- Run id: {_display(run.get('run_id'))}",
        f"- Classification mode: {_display(metrics_payload.get('classification_mode'))}",
        f"- Taxonomy table version: {_display(run.get('taxonomy_table_version'))}",
        "",
        "## Core Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Evaluated objects | {_display(metrics.get('evaluated_objects'))} |",
        f"| Evaluated photos | {_display(metrics.get('evaluated_photos'))} |",
        f"| Family top1 accuracy | {_display(metrics.get('family_top1_accuracy'))} |",
        f"| Family top3 recall | {_display(metrics.get('family_top3_recall'))} |",
        f"| Selected-family accuracy | {_display(metrics.get('selected_family_accuracy'))} |",
        f"| Species top1 accuracy | {_display(metrics.get('species_top1_accuracy'))} |",
        f"| Species top5 recall | {_display(metrics.get('species_top5_recall'))} |",
        f"| Species top20 recall | {_display(metrics.get('species_top20_recall'))} |",
        f"| Species MRR | {_display(metrics.get('species_mrr'))} |",
        f"| Missing predictions | {_display(metrics.get('missing_prediction_count'))} |",
        f"| False positive butterfly | {_display(metrics.get('false_positive_butterfly_count'))} |",
        f"| False negative butterfly | {_display(metrics.get('false_negative_butterfly_count'))} |",
        "",
        "## Calibration",
        "",
        f"- Mode: {_display(calibration.get('calibration_mode'))}",
        f"- ECE: {_display(calibration.get('ece'))}",
        f"- Samples: {_display(calibration.get('sample_count'))}",
        f"- Limitation: {_display(calibration.get('limitations'))}",
        "",
        "## Top Family Confusions",
        "",
        *_confusion_lines(family_confusion),
        "",
        "## Top Species Confusions",
        "",
        *_confusion_lines(species_confusion),
        "",
        "## Warnings",
        "",
        *warning_lines,
        "",
    ]
    if int(metrics.get("evaluated_objects") or 0) == 0:
        lines.extend(["No reviewed object labels were available for evaluation.", ""])
    return "\n".join(lines)


def _with_species_correct_column(object_scores: pl.DataFrame, reviewed_labels: pl.DataFrame) -> pl.DataFrame:
    labels_by_key = {
        _object_key(row): row
        for row in reviewed_labels.to_dicts()
        if _text(row.get("detection_id"))
    }
    rows = []
    for row in object_scores.to_dicts():
        if _text(row.get("classification_mode")) != HIERARCHICAL_BUTTERFLY_CLASSIFICATION:
            continue
        label = labels_by_key.get(_object_key(row))
        rows.append(
            {
                **row,
                "species_top1_correct": _species_top1_correct(row, label),
                "expected_key": _text(label.get("accepted_taxon_key")) if label else "",
                "expected_name": _text(label.get("scientific_name")) if label else "",
                "predicted_key": _text(row.get("species_top1_accepted_taxon_key") or row.get("accepted_taxon_key")),
                "predicted_name": _text(row.get("species_top1_scientific_name") or row.get("species_top1")),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _review_error_examples(frame: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for row in frame.to_dicts():
        expected = _text(row.get("expected_key")) or _text(row.get("expected_name"))
        predicted = _text(row.get("predicted_key")) or _text(row.get("predicted_name"))
        if not expected or bool(row.get("species_top1_correct")):
            continue
        rows.append(
            {
                "source": _text(row.get("source")),
                "flickr_photo_id": _text(row.get("flickr_photo_id")),
                "detection_id": _text(row.get("detection_id")),
                "expected_key": _text(row.get("expected_key")),
                "expected_name": _text(row.get("expected_name")),
                "predicted_key": _text(row.get("predicted_key")),
                "predicted_name": _text(row.get("predicted_name")),
                "error_type": "species_top1_mismatch" if predicted else "missing_species_prediction",
            }
        )
    return _ensure_review_error_schema(pl.DataFrame(rows))


def _calibration_bins_frame(calibration: Mapping[str, Any]) -> pl.DataFrame:
    return _ensure_calibration_schema(pl.DataFrame(calibration.get("bins") or []))


def _run_metadata(run_manifest: Mapping[str, Any], object_scores: pl.DataFrame) -> dict[str, object]:
    return {
        "run_id": run_manifest.get("run_id"),
        "taxonomy_table_version": run_manifest.get("taxonomy_table_version")
        or _first_value(object_scores, "taxonomy_table_version"),
        "taxonomy_prompt_variant_version": run_manifest.get("taxonomy_prompt_variant_version")
        or _first_value(object_scores, "taxonomy_prompt_variant_version"),
        "model_id": run_manifest.get("model_id") or _first_value(object_scores, "model_id"),
        "model_checkpoint": run_manifest.get("model_checkpoint") or _first_value(object_scores, "model_checkpoint"),
    }


def _warnings(*, metrics: Mapping[str, Any], calibration: Mapping[str, Any]) -> list[str]:
    warnings = [str(calibration.get("limitations"))]
    if int(metrics.get("missing_prediction_count") or 0) > 0:
        warnings.append("missing_predictions_present")
    if int(metrics.get("false_positive_butterfly_count") or 0) > 0:
        warnings.append("false_positive_butterfly_predictions_present")
    if int(metrics.get("false_negative_butterfly_count") or 0) > 0:
        warnings.append("false_negative_butterfly_predictions_present")
    return [warning for warning in warnings if warning]


def _species_top1_correct(prediction: Mapping[str, Any], label: Mapping[str, Any] | None) -> bool | None:
    if label is None or not bool(label.get("is_butterfly")):
        return None
    expected_key = _text(label.get("accepted_taxon_key"))
    expected_name = _text(label.get("scientific_name")).casefold()
    predicted_key = _text(prediction.get("species_top1_accepted_taxon_key") or prediction.get("accepted_taxon_key"))
    predicted_name = _text(prediction.get("species_top1_scientific_name") or prediction.get("species_top1")).casefold()
    return bool((expected_key and expected_key == predicted_key) or (expected_name and expected_name == predicted_name))


def _confusion_lines(frame: pl.DataFrame, *, limit: int = 10) -> list[str]:
    if frame.is_empty():
        return ["- none"]
    lines = []
    for row in frame.head(limit).to_dicts():
        lines.append(
            f"- {row['true_name']} -> {row['predicted_name']}: {row['count']}"
        )
    return lines


def _ensure_calibration_schema(frame: pl.DataFrame) -> pl.DataFrame:
    return _ensure_schema(frame, CALIBRATION_BIN_SCHEMA)


def _ensure_review_error_schema(frame: pl.DataFrame) -> pl.DataFrame:
    return _ensure_schema(frame, REVIEW_ERROR_EXAMPLE_SCHEMA)


def _ensure_schema(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if frame.is_empty() and not frame.columns:
        return pl.DataFrame(schema=schema)
    expressions = []
    for column, dtype in schema.items():
        if column in frame.columns:
            expressions.append(pl.col(column).cast(dtype).alias(column))
        else:
            expressions.append(pl.lit(_default_for_dtype(dtype)).cast(dtype).alias(column))
    return frame.with_columns(expressions).select(list(schema))


def _first_value(frame: pl.DataFrame, column: str) -> object:
    if frame.is_empty() or column not in frame.columns:
        return None
    values = frame.select(column).to_series().drop_nulls().to_list()
    return values[0] if values else None


def _object_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (_text(row.get("source")), _text(row.get("flickr_photo_id")), _text(row.get("detection_id")))


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _display(value: object) -> str:
    if value is None:
        return "not_instrumented"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _default_for_dtype(dtype: pl.DataType) -> object:
    if dtype == pl.Boolean:
        return False
    if dtype in {pl.Int64, pl.Int32, pl.UInt64, pl.UInt32}:
        return 0
    if dtype in {pl.Float64, pl.Float32}:
        return 0.0
    return ""


def _text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


__all__ = [
    "CALIBRATION_BINS_FILE",
    "EVALUATION_METRICS_FILE",
    "EVALUATION_SUMMARY_FILE",
    "FAMILY_CONFUSION_FILE",
    "REVIEW_ERROR_EXAMPLES_FILE",
    "SPECIES_CONFUSION_FILE",
    "EvaluationReportArtifacts",
    "build_evaluation_report_artifacts",
    "evaluation_summary_markdown",
    "write_evaluation_report",
    "write_evaluation_report_to_storage",
]
