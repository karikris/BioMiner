from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

import polars as pl

from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION
from biominer.evaluation.metrics import (
    evaluate_hierarchical_predictions,
    family_confusion_matrix,
    species_confusion_matrix,
)


EVALUATION_PROFILE = "xie_style_metrics_only"
ARCHITECTURE = "biominer_yoloe26_bioclip25_hierarchical"


def evaluate_xie_style_hierarchical(
    *,
    object_scores: pl.DataFrame,
    reviewed_labels: pl.DataFrame,
) -> dict[str, object]:
    """Return Xie-style benchmark metrics without changing BioMiner architecture."""

    base_metrics = evaluate_hierarchical_predictions(
        object_scores=object_scores,
        reviewed_labels=reviewed_labels,
    )
    family_rows = _per_family_rows(object_scores=object_scores, reviewed_labels=reviewed_labels)
    family_top1_values = [float(row["species_top1_accuracy"]) for row in family_rows]
    family_top5_values = [float(row["species_top5_recall"]) for row in family_rows]
    return {
        "evaluation_profile": EVALUATION_PROFILE,
        "architecture": ARCHITECTURE,
        "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        "sample_count": int(base_metrics["butterfly_positive_labels"]),
        "matched_sample_count": int(base_metrics["matched_hierarchical_objects"]),
        "family_accuracy": float(base_metrics["selected_family_accuracy"]),
        "family_top1_accuracy": float(base_metrics["family_top1_accuracy"]),
        "family_top3_recall": float(base_metrics["family_top3_recall"]),
        "species_top1_accuracy": float(base_metrics["species_top1_accuracy"]),
        "species_top5_accuracy": float(base_metrics["species_top5_recall"]),
        "species_top5_recall": float(base_metrics["species_top5_recall"]),
        "macro_average_by_family": {
            "family_count": len(family_rows),
            "species_top1_accuracy": _mean(family_top1_values),
            "species_top5_recall": _mean(family_top5_values),
        },
        "micro_average": {
            "species_top1_accuracy": float(base_metrics["species_top1_accuracy"]),
            "species_top5_recall": float(base_metrics["species_top5_recall"]),
        },
        "per_family_species_accuracy": family_rows,
        "confusion_summary": {
            "family": family_confusion_matrix(
                object_scores=object_scores,
                reviewed_labels=reviewed_labels,
            ).to_dicts(),
            "species": species_confusion_matrix(
                object_scores=object_scores,
                reviewed_labels=reviewed_labels,
                limit=20,
            ).to_dicts(),
        },
        "source_metrics": base_metrics,
    }


def _per_family_rows(*, object_scores: pl.DataFrame, reviewed_labels: pl.DataFrame) -> list[dict[str, object]]:
    predictions = [
        row
        for row in object_scores.to_dicts()
        if _text(row.get("classification_mode")) == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
    ]
    prediction_by_key = {_object_key(row): row for row in predictions if _has_object_key(row)}
    predictions_by_photo: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in predictions:
        predictions_by_photo.setdefault(_photo_key(row), []).append(row)

    groups: dict[tuple[str, str], dict[str, object]] = {}
    for label in reviewed_labels.to_dicts():
        if not bool(label.get("is_butterfly")):
            continue
        family_key = _text(label.get("family_key"))
        family = _text(label.get("family"))
        key = (family_key, family)
        group = groups.setdefault(
            key,
            {
                "family_key": family_key,
                "family": family,
                "sample_count": 0,
                "species_top1_correct": 0,
                "species_top5_correct": 0,
            },
        )
        group["sample_count"] = int(group["sample_count"]) + 1
        prediction = _prediction_for_label(label, prediction_by_key, predictions_by_photo)
        if prediction is None:
            continue
        if _species_top1_correct(label, prediction):
            group["species_top1_correct"] = int(group["species_top1_correct"]) + 1
        if _species_top5_correct(label, prediction):
            group["species_top5_correct"] = int(group["species_top5_correct"]) + 1

    rows = []
    for group in groups.values():
        sample_count = int(group["sample_count"])
        top1_correct = int(group["species_top1_correct"])
        top5_correct = int(group["species_top5_correct"])
        rows.append(
            {
                **group,
                "species_top1_accuracy": _ratio(top1_correct, sample_count),
                "species_top5_recall": _ratio(top5_correct, sample_count),
            }
        )
    return sorted(rows, key=lambda row: (str(row["family"]), str(row["family_key"])))


def _prediction_for_label(
    label: Mapping[str, Any],
    prediction_by_key: Mapping[tuple[str, str, str], dict[str, Any]],
    predictions_by_photo: Mapping[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any] | None:
    detection_id = _text(label.get("detection_id"))
    if detection_id:
        return prediction_by_key.get(_object_key(label))
    photo_predictions = predictions_by_photo.get(_photo_key(label), [])
    return photo_predictions[0] if len(photo_predictions) == 1 else None


def _species_top1_correct(label: Mapping[str, Any], prediction: Mapping[str, Any]) -> bool:
    return _matches_taxon(
        label_key=label.get("accepted_taxon_key"),
        label_name=label.get("scientific_name"),
        prediction_key=prediction.get("species_top1_accepted_taxon_key") or prediction.get("accepted_taxon_key"),
        prediction_name=prediction.get("species_top1_scientific_name") or prediction.get("species_top1"),
    )


def _species_top5_correct(label: Mapping[str, Any], prediction: Mapping[str, Any]) -> bool:
    return _matches_any_taxon(
        label_key=label.get("accepted_taxon_key"),
        label_name=label.get("scientific_name"),
        prediction_keys=_as_list(prediction.get("species_top5_accepted_taxon_keys")),
        prediction_names=_as_list(prediction.get("species_top5")),
    )


def _matches_any_taxon(
    *,
    label_key: object,
    label_name: object,
    prediction_keys: Iterable[object],
    prediction_names: Iterable[object],
) -> bool:
    label_key_text = _text(label_key)
    label_name_text = _text(label_name).casefold()
    for predicted_key in prediction_keys:
        if label_key_text and _text(predicted_key) == label_key_text:
            return True
    for predicted_name in prediction_names:
        if label_name_text and _text(predicted_name).casefold() == label_name_text:
            return True
    return False


def _matches_taxon(
    *,
    label_key: object,
    label_name: object,
    prediction_key: object,
    prediction_name: object,
) -> bool:
    label_key_text = _text(label_key)
    prediction_key_text = _text(prediction_key)
    if label_key_text and prediction_key_text:
        return label_key_text == prediction_key_text
    label_name_text = _text(label_name).casefold()
    prediction_name_text = _text(prediction_name).casefold()
    return bool(label_name_text and prediction_name_text and label_name_text == prediction_name_text)


def _object_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (_text(row.get("source")), _text(row.get("flickr_photo_id")), _text(row.get("detection_id")))


def _photo_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (_text(row.get("source")), _text(row.get("flickr_photo_id")))


def _has_object_key(row: Mapping[str, Any]) -> bool:
    return all(_object_key(row))


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return list(value)
    if isinstance(value, pl.Series):
        return value.to_list()
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return parsed
    return [value]


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


__all__ = [
    "ARCHITECTURE",
    "EVALUATION_PROFILE",
    "evaluate_xie_style_hierarchical",
]
