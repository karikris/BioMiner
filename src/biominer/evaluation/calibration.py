from __future__ import annotations

import json
from math import log
from typing import Any, Mapping, Sequence

import polars as pl


CALIBRATION_MODE = "candidate_set_relative_heuristic"


def score_margin(row: dict[str, object], score_column: str = "species_top5_scores") -> float | None:
    scores = _float_list(row.get(score_column))
    if len(scores) < 2:
        return None
    return float(scores[0] - scores[1])


def topk_entropy(scores: Sequence[float] | object) -> float | None:
    values = [max(0.0, value) for value in _float_list(scores)]
    total = sum(values)
    if total <= 0:
        return None
    probabilities = [value / total for value in values if value > 0]
    return float(-sum(probability * log(probability) for probability in probabilities))


def expected_calibration_error(
    *,
    predictions: pl.DataFrame,
    labels: pl.DataFrame,
    score_column: str,
    correct_column: str,
    bins: int = 10,
) -> dict[str, object]:
    if bins <= 0:
        raise ValueError("bins must be positive")

    samples = []
    for row in predictions.to_dicts():
        confidence = _confidence(row.get(score_column))
        correct = _correct(row.get(correct_column))
        if confidence is None or correct is None:
            continue
        samples.append((confidence, correct))

    bin_rows = [
        {
            "bin_index": index,
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "count": 0,
            "avg_confidence": 0.0,
            "accuracy": 0.0,
            "gap": 0.0,
            "weight": 0.0,
        }
        for index in range(bins)
    ]
    if not samples:
        return {
            "calibration_mode": CALIBRATION_MODE,
            "score_column": score_column,
            "correct_column": correct_column,
            "bin_count": bins,
            "sample_count": 0,
            "label_rows": labels.height,
            "ece": 0.0,
            "bins": bin_rows,
            "limitations": "BioCLIP scores are candidate-set-relative, so ECE is heuristic.",
        }

    grouped: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for confidence, correct in samples:
        index = min(int(confidence * bins), bins - 1)
        grouped[index].append((confidence, correct))

    ece = 0.0
    sample_count = len(samples)
    for index, values in enumerate(grouped):
        if not values:
            continue
        avg_confidence = sum(confidence for confidence, _correct_value in values) / len(values)
        accuracy = sum(1.0 for _confidence_value, correct in values if correct) / len(values)
        gap = abs(avg_confidence - accuracy)
        weight = len(values) / sample_count
        ece += gap * weight
        bin_rows[index] = {
            **bin_rows[index],
            "count": len(values),
            "avg_confidence": float(avg_confidence),
            "accuracy": float(accuracy),
            "gap": float(gap),
            "weight": float(weight),
        }

    return {
        "calibration_mode": CALIBRATION_MODE,
        "score_column": score_column,
        "correct_column": correct_column,
        "bin_count": bins,
        "sample_count": sample_count,
        "label_rows": labels.height,
        "ece": float(ece),
        "bins": bin_rows,
        "limitations": "BioCLIP scores are candidate-set-relative, so ECE is heuristic.",
    }


def add_uncertainty_fields(
    frame: pl.DataFrame,
    *,
    low_margin_threshold: float = 0.05,
) -> pl.DataFrame:
    rows = []
    for row in frame.to_dicts():
        species_margin = _optional_float(row.get("species_top1_margin"))
        if species_margin is None:
            species_margin = score_margin(row, "species_top5_scores")
        family_margin = _optional_float(row.get("family_margin"))
        if family_margin is None:
            family_margin = score_margin(row, "family_top3_scores")
        entropy = topk_entropy(row.get("species_top5_scores"))
        low_margin = any(
            margin is not None and margin <= low_margin_threshold
            for margin in (species_margin, family_margin)
        )
        rows.append(
            {
                **row,
                "species_top1_margin": species_margin,
                "family_margin": family_margin,
                "species_top5_entropy": entropy,
                "low_margin_flag": bool(low_margin),
                "family_species_conflict_flag": _family_species_conflict(row),
            }
        )
    return pl.DataFrame(rows) if rows else frame.with_columns(
        [
            pl.lit(None).cast(pl.Float64).alias("species_top5_entropy"),
            pl.lit(False).cast(pl.Boolean).alias("low_margin_flag"),
            pl.lit(False).cast(pl.Boolean).alias("family_species_conflict_flag"),
        ]
    )


def _family_species_conflict(row: Mapping[str, Any]) -> bool:
    selected_key = _text(row.get("selected_family_key"))
    species_family_key = _text(row.get("species_candidate_family_key"))
    if selected_key and species_family_key and selected_key != species_family_key:
        return True
    selected_name = _text(row.get("selected_family")).casefold()
    species_family = _text(row.get("species_candidate_family")).casefold()
    return bool(selected_name and species_family and selected_name != species_family)


def _confidence(value: object) -> float | None:
    values = _float_list(value)
    if not values:
        return None
    return min(1.0, max(0.0, float(values[0] if len(values) == 1 else max(values))))


def _correct(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _float_list(value: object) -> list[float]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return [_float for item in value if (_float := _optional_float(item)) is not None]
    if isinstance(value, pl.Series):
        return [_float for item in value.to_list() if (_float := _optional_float(item)) is not None]
    if isinstance(value, int | float):
        return [float(value)]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [_float for item in parsed if (_float := _optional_float(item)) is not None]
    numeric = _optional_float(text)
    return [] if numeric is None else [numeric]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


__all__ = [
    "CALIBRATION_MODE",
    "add_uncertainty_fields",
    "expected_calibration_error",
    "score_margin",
    "topk_entropy",
]
