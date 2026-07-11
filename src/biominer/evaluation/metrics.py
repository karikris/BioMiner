from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

import polars as pl

from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION, TARGET_SCOPE_OBJECT_SCREENING


CONFUSION_MATRIX_SCHEMA: dict[str, pl.DataType] = {
    "true_key": pl.String,
    "true_name": pl.String,
    "predicted_key": pl.String,
    "predicted_name": pl.String,
    "count": pl.Int64,
    "classification_mode": pl.String,
}


def evaluate_hierarchical_predictions(
    *,
    object_scores: pl.DataFrame,
    reviewed_labels: pl.DataFrame,
) -> dict[str, object]:
    labels = reviewed_labels.to_dicts()
    predictions = object_scores.to_dicts()
    hierarchical = [
        row
        for row in predictions
        if _text(row.get("classification_mode")) == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
    ]
    non_hierarchical = [
        row
        for row in predictions
        if (
            _text(row.get("classification_mode"))
            and _text(row.get("classification_mode")) != HIERARCHICAL_BUTTERFLY_CLASSIFICATION
        )
    ]
    prediction_by_key = {_object_key(row): row for row in hierarchical if _has_object_key(row)}
    predictions_by_photo: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in hierarchical:
        predictions_by_photo.setdefault(_photo_key(row), []).append(row)

    object_label_keys = {_object_key(row) for row in labels if _text(row.get("detection_id"))}
    positive_labels = [row for row in labels if bool(row.get("is_butterfly"))]
    negative_labels = [row for row in labels if not bool(row.get("is_butterfly"))]
    matched_positive: list[tuple[dict[str, Any], dict[str, Any]]] = []
    matched_negative: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    missing_predictions = 0
    false_negative_butterfly_count = 0
    false_positive_butterfly_count = 0
    negative_correct_count = 0

    for label in labels:
        prediction = _prediction_for_label(label, prediction_by_key, predictions_by_photo)
        if prediction is None:
            missing_predictions += 1
            if bool(label.get("is_butterfly")):
                false_negative_butterfly_count += 1
            else:
                negative_correct_count += 1
            continue
        if bool(label.get("is_butterfly")):
            matched_positive.append((label, prediction))
        else:
            matched_negative.append((label, prediction))
            if _has_butterfly_prediction(prediction):
                false_positive_butterfly_count += 1
            else:
                negative_correct_count += 1

    family_top1_correct = sum(1 for label, prediction in matched_positive if _family_top1_correct(label, prediction))
    family_top3_correct = sum(1 for label, prediction in matched_positive if _family_in_top3(label, prediction))
    selected_family_correct = sum(
        1 for label, prediction in matched_positive if _selected_family_correct(label, prediction)
    )
    species_top1_correct = sum(1 for label, prediction in matched_positive if _species_top1_correct(label, prediction))
    species_top5_correct = sum(
        1
        for label, prediction in matched_positive
        if _species_in_topk(
            label,
            prediction,
            names_column="species_top5",
            keys_column="species_top5_accepted_taxon_keys",
        )
    )
    species_top20_correct = sum(
        1
        for label, prediction in matched_positive
        if _species_in_topk(
            label,
            prediction,
            names_column="species_top20",
            keys_column="species_top20_accepted_taxon_keys",
        )
    )
    reciprocal_ranks = [_species_reciprocal_rank(label, prediction) for label, prediction in matched_positive]
    hierarchical_keys = {_object_key(row) for row in hierarchical if _has_object_key(row)}

    denominator = len(matched_positive)
    return {
        "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        "hierarchical_prediction_count": len(hierarchical),
        "non_hierarchical_prediction_count": len(non_hierarchical),
        "target_scope_prediction_count": sum(
            1 for row in non_hierarchical if _text(row.get("classification_mode")) == TARGET_SCOPE_OBJECT_SCREENING
        ),
        "evaluated_objects": sum(1 for row in labels if _text(row.get("detection_id"))),
        "evaluated_photos": len({_photo_key(row) for row in labels}),
        "matched_hierarchical_objects": len(matched_positive) + len(matched_negative),
        "butterfly_positive_labels": len(positive_labels),
        "negative_labels": len(negative_labels),
        "family_top1_correct": family_top1_correct,
        "family_top1_accuracy": _ratio(family_top1_correct, denominator),
        "family_top3_recall": _ratio(family_top3_correct, denominator),
        "selected_family_accuracy": _ratio(selected_family_correct, denominator),
        "species_top1_correct": species_top1_correct,
        "species_top1_accuracy": _ratio(species_top1_correct, denominator),
        "species_top5_recall": _ratio(species_top5_correct, denominator),
        "species_top20_recall": _ratio(species_top20_correct, denominator),
        "species_mrr": _mean(reciprocal_ranks),
        "negative_correct_count": negative_correct_count,
        "false_positive_butterfly_count": false_positive_butterfly_count,
        "false_negative_butterfly_count": false_negative_butterfly_count,
        "missing_prediction_count": missing_predictions,
        "missing_label_count": len(hierarchical_keys - object_label_keys),
    }


def family_confusion_matrix(
    *,
    object_scores: pl.DataFrame,
    reviewed_labels: pl.DataFrame,
) -> pl.DataFrame:
    return _confusion_matrix(object_scores=object_scores, reviewed_labels=reviewed_labels, level="family")


def species_confusion_matrix(
    *,
    object_scores: pl.DataFrame,
    reviewed_labels: pl.DataFrame,
    limit: int | None = None,
) -> pl.DataFrame:
    return _confusion_matrix(
        object_scores=object_scores,
        reviewed_labels=reviewed_labels,
        level="species",
        limit=limit,
    )


def _confusion_matrix(
    *,
    object_scores: pl.DataFrame,
    reviewed_labels: pl.DataFrame,
    level: str,
    limit: int | None = None,
) -> pl.DataFrame:
    predictions = [
        row
        for row in object_scores.to_dicts()
        if _text(row.get("classification_mode")) == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
    ]
    prediction_by_key = {_object_key(row): row for row in predictions if _has_object_key(row)}
    predictions_by_photo: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in predictions:
        predictions_by_photo.setdefault(_photo_key(row), []).append(row)

    rows = []
    for label in reviewed_labels.to_dicts():
        prediction = _prediction_for_label(label, prediction_by_key, predictions_by_photo)
        true_key, true_name = _true_confusion_taxon(label, level=level)
        predicted_key, predicted_name = _predicted_confusion_taxon(prediction, level=level)
        rows.append(
            {
                "true_key": true_key,
                "true_name": true_name,
                "predicted_key": predicted_key,
                "predicted_name": predicted_name,
                "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
            }
        )

    if not rows:
        return pl.DataFrame(schema=CONFUSION_MATRIX_SCHEMA)
    grouped = (
        pl.DataFrame(rows)
        .group_by(["true_key", "true_name", "predicted_key", "predicted_name", "classification_mode"])
        .len()
        .rename({"len": "count"})
        .sort(["count", "true_name", "predicted_name"], descending=[True, False, False])
    )
    if limit is not None:
        grouped = grouped.head(max(0, int(limit)))
    return _ensure_confusion_schema(grouped)


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


def _true_confusion_taxon(label: Mapping[str, Any], *, level: str) -> tuple[str, str]:
    if not bool(label.get("is_butterfly")) or _text(label.get("label_level")) == "negative":
        return "not_butterfly", "not_butterfly"
    if level == "family":
        return _taxon_or_missing(label.get("family_key"), label.get("family"), missing="missing_family_label")
    return _taxon_or_missing(
        label.get("accepted_taxon_key"),
        label.get("scientific_name"),
        missing="missing_species_label",
    )


def _predicted_confusion_taxon(prediction: Mapping[str, Any] | None, *, level: str) -> tuple[str, str]:
    if prediction is None:
        return "missing_prediction", "missing_prediction"
    if level == "family":
        is_path_cascade = _is_path_cascade_prediction(prediction)
        return _taxon_or_missing(
            None if is_path_cascade else prediction.get("selected_family_key"),
            prediction.get("selected_family")
            or (None if is_path_cascade else prediction.get("family_top1")),
            missing="missing_prediction",
        )
    return _taxon_or_missing(
        prediction.get("species_top1_accepted_taxon_key") or prediction.get("accepted_taxon_key"),
        prediction.get("species_top1_scientific_name") or prediction.get("species_top1"),
        missing="missing_prediction",
    )


def _family_top1_correct(label: Mapping[str, Any], prediction: Mapping[str, Any]) -> bool:
    if _is_path_cascade_prediction(prediction):
        return _matches_taxon(
            label_key=label.get("family_key"),
            label_name=label.get("family"),
            prediction_key=None,
            prediction_name=prediction.get("family_top1"),
        )
    top_keys = _as_list(prediction.get("family_top3_accepted_taxon_keys"))
    top_names = _as_list(prediction.get("family_top3"))
    selected_key = _text(prediction.get("selected_family_key"))
    selected_name = _text(prediction.get("selected_family") or prediction.get("family_top1"))
    first_key = _text(top_keys[0]) if top_keys else selected_key
    first_name = _text(top_names[0]) if top_names else selected_name
    return _matches_taxon(
        label_key=label.get("family_key"),
        label_name=label.get("family"),
        prediction_key=first_key,
        prediction_name=first_name,
    )


def _family_in_top3(label: Mapping[str, Any], prediction: Mapping[str, Any]) -> bool:
    prediction_keys = (
        []
        if _is_path_cascade_prediction(prediction)
        else _as_list(prediction.get("family_top3_accepted_taxon_keys"))
    )
    return _matches_any_taxon(
        label_key=label.get("family_key"),
        label_name=label.get("family"),
        prediction_keys=prediction_keys,
        prediction_names=_as_list(prediction.get("family_top3")),
    )


def _selected_family_correct(label: Mapping[str, Any], prediction: Mapping[str, Any]) -> bool:
    is_path_cascade = _is_path_cascade_prediction(prediction)
    return _matches_taxon(
        label_key=label.get("family_key"),
        label_name=label.get("family"),
        prediction_key=None if is_path_cascade else prediction.get("selected_family_key"),
        prediction_name=prediction.get("selected_family")
        or (None if is_path_cascade else prediction.get("family_top1")),
    )


def _is_path_cascade_prediction(prediction: Mapping[str, Any]) -> bool:
    return _text(prediction.get("classifier_schema_version")).startswith(
        "butterfly-cascade-output-"
    )


def _species_top1_correct(label: Mapping[str, Any], prediction: Mapping[str, Any]) -> bool:
    return _matches_taxon(
        label_key=label.get("accepted_taxon_key"),
        label_name=label.get("scientific_name"),
        prediction_key=prediction.get("species_top1_accepted_taxon_key") or prediction.get("accepted_taxon_key"),
        prediction_name=prediction.get("species_top1_scientific_name") or prediction.get("species_top1"),
    )


def _species_in_topk(
    label: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    names_column: str,
    keys_column: str,
) -> bool:
    return _matches_any_taxon(
        label_key=label.get("accepted_taxon_key"),
        label_name=label.get("scientific_name"),
        prediction_keys=_as_list(prediction.get(keys_column)),
        prediction_names=_as_list(prediction.get(names_column)),
    )


def _species_reciprocal_rank(label: Mapping[str, Any], prediction: Mapping[str, Any]) -> float:
    keys = _as_list(prediction.get("species_top20_accepted_taxon_keys"))
    names = _as_list(prediction.get("species_top20"))
    if not keys and not names:
        keys = _as_list(prediction.get("species_top5_accepted_taxon_keys"))
        names = _as_list(prediction.get("species_top5"))
    label_key = _text(label.get("accepted_taxon_key"))
    label_name = _text(label.get("scientific_name")).casefold()
    width = max(len(keys), len(names))
    for index in range(width):
        predicted_key = _text(keys[index]) if index < len(keys) else ""
        predicted_name = _text(names[index]).casefold() if index < len(names) else ""
        if (label_key and predicted_key == label_key) or (label_name and predicted_name == label_name):
            return 1.0 / float(index + 1)
    return 0.0


def _matches_any_taxon(
    *,
    label_key: object,
    label_name: object,
    prediction_keys: Iterable[object],
    prediction_names: Iterable[object],
) -> bool:
    keys = {_text(value) for value in prediction_keys if _text(value)}
    names = {_text(value).casefold() for value in prediction_names if _text(value)}
    expected_key = _text(label_key)
    expected_name = _text(label_name).casefold()
    return bool((expected_key and expected_key in keys) or (expected_name and expected_name in names))


def _matches_taxon(*, label_key: object, label_name: object, prediction_key: object, prediction_name: object) -> bool:
    expected_key = _text(label_key)
    expected_name = _text(label_name).casefold()
    actual_key = _text(prediction_key)
    actual_name = _text(prediction_name).casefold()
    return bool((expected_key and actual_key == expected_key) or (expected_name and actual_name == expected_name))


def _has_butterfly_prediction(prediction: Mapping[str, Any]) -> bool:
    return bool(
        _text(prediction.get("selected_family_key"))
        or _text(prediction.get("selected_family"))
        or _text(prediction.get("species_top1_accepted_taxon_key"))
        or _text(prediction.get("species_top1_scientific_name"))
        or _as_list(prediction.get("species_top20"))
    )


def _taxon_or_missing(key: object, name: object, *, missing: str) -> tuple[str, str]:
    taxon_key = _text(key)
    taxon_name = _text(name)
    if not taxon_key and not taxon_name:
        return missing, missing
    return taxon_key or taxon_name, taxon_name or taxon_key


def _ensure_confusion_schema(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty() and not frame.columns:
        return pl.DataFrame(schema=CONFUSION_MATRIX_SCHEMA)
    expressions = []
    for column, dtype in CONFUSION_MATRIX_SCHEMA.items():
        expressions.append(pl.col(column).cast(dtype).alias(column))
    return frame.with_columns(expressions).select(list(CONFUSION_MATRIX_SCHEMA))


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
            return [text]
        return list(parsed) if isinstance(parsed, list) else [parsed]
    return [text]


def _has_object_key(row: Mapping[str, Any]) -> bool:
    return bool(_text(row.get("source")) and _text(row.get("flickr_photo_id")) and _text(row.get("detection_id")))


def _object_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (_text(row.get("source")), _text(row.get("flickr_photo_id")), _text(row.get("detection_id")))


def _photo_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (_text(row.get("source")), _text(row.get("flickr_photo_id")))


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


__all__ = [
    "CONFUSION_MATRIX_SCHEMA",
    "evaluate_hierarchical_predictions",
    "family_confusion_matrix",
    "species_confusion_matrix",
]
