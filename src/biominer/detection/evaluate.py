from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable


def iou_xyxy(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]
    ix1 = max(min(ax1, ax2), min(bx1, bx2))
    iy1 = max(min(ay1, ay2), min(by1, by2))
    ix2 = min(max(ax1, ax2), max(bx1, bx2))
    iy2 = min(max(ay1, ay2), max(by1, by2))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _area((ax1, ay1, ax2, ay2)) + _area((bx1, by1, bx2, by2)) - intersection
    return intersection / union if union > 0 else 0.0


def joint_detection_species_correct(
    *,
    prediction: dict[str, Any],
    truth: dict[str, Any],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.35,
) -> bool:
    score = _optional_float(prediction.get("species_top1_score")) or 0.0
    if score < score_threshold:
        return False
    if iou_xyxy(prediction.get("bbox_xyxy") or (), truth.get("bbox_xyxy") or ()) < iou_threshold:
        return False
    predicted_key = _norm(prediction.get("accepted_taxon_key") or prediction.get("species_top1_accepted_taxon_key"))
    truth_key = _norm(truth.get("accepted_taxon_key"))
    if predicted_key and truth_key:
        return predicted_key == truth_key
    return _norm(prediction.get("species_top1_scientific_name")) == _norm(truth.get("scientific_name"))


def evaluate_xie_style(
    *,
    predictions: Iterable[dict[str, Any]],
    ground_truth: Iterable[dict[str, Any]] | None = None,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.35,
) -> dict[str, Any]:
    truth_rows = list(ground_truth or [])
    prediction_rows = list(predictions)
    if not truth_rows:
        return {
            "ground_truth_available": False,
            "iou_threshold": iou_threshold,
            "score_threshold": score_threshold,
            "predictions_seen": len(prediction_rows),
            "detector_ap50": None,
            "detector_ap50_95": None,
            "species_top1_accuracy": None,
            "species_top5_accuracy": None,
            "family_top3_accuracy": None,
            "genus_top8_accuracy": None,
            "joint_map50": None,
            "joint_top5_map50": None,
        }
    matches = _best_matches(prediction_rows, truth_rows, iou_threshold=iou_threshold, score_fn=_species_score)
    matched_truth = [truth for _prediction, truth, iou in matches if truth is not None and iou >= iou_threshold]
    species_top1 = _accuracy(matches, _species_top1_correct, truth_count=len(truth_rows))
    species_top5 = _accuracy(matches, _species_top5_correct, truth_count=len(truth_rows))
    family_top3 = _accuracy(
        matches,
        lambda prediction, truth: _norm(truth.get("family")) in {_norm(value) for value in prediction.get("family_top3", [])},
        truth_count=len(truth_rows),
    )
    genus_top8 = _accuracy(
        matches,
        lambda prediction, truth: _norm(truth.get("genus")) in {_norm(value) for value in prediction.get("genus_top8", [])},
        truth_count=len(truth_rows),
    )
    joint_results = [
        (
            truth is not None
            and
            joint_detection_species_correct(
                prediction=prediction,
                truth=truth,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
            ),
            _optional_float(prediction.get("species_top1_score")) or 0.0,
        )
        for prediction, truth, _iou in matches
    ]
    return {
        "ground_truth_available": True,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "predictions_seen": len(prediction_rows),
        "ground_truth_seen": len(truth_rows),
        "matched_ground_truth": len(matched_truth),
        "detector_ap50": _detector_ap(prediction_rows, truth_rows, iou_threshold=iou_threshold),
        "detector_ap50_95": _detector_ap50_95(prediction_rows, truth_rows),
        "species_top1_accuracy": species_top1,
        "species_top5_accuracy": species_top5,
        "family_top3_accuracy": family_top3,
        "genus_top8_accuracy": genus_top8,
        "joint_map50": _ap(joint_results, len(truth_rows)),
        "joint_top5_map50": _ap(
            [
                (
                    truth is not None
                    and iou >= iou_threshold
                    and _species_top5_correct(prediction, truth),
                    _optional_float(prediction.get("species_top1_score")) or 0.0,
                )
                for prediction, truth, iou in matches
            ],
            len(truth_rows),
        ),
    }


def _best_matches(
    predictions: list[dict[str, Any]],
    truths: list[dict[str, Any]],
    *,
    iou_threshold: float,
    score_fn: Callable[[dict[str, Any]], float],
) -> list[tuple[dict[str, Any], dict[str, Any] | None, float]]:
    truth_by_photo: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for truth in truths:
        truth_by_photo[(str(truth.get("source") or ""), str(truth.get("flickr_photo_id") or ""))].append(truth)
    used: set[int] = set()
    matches: list[tuple[dict[str, Any], dict[str, Any] | None, float]] = []
    for prediction in sorted(predictions, key=score_fn, reverse=True):
        key = (str(prediction.get("source") or ""), str(prediction.get("flickr_photo_id") or ""))
        best_index = -1
        best_truth: dict[str, Any] | None = None
        best_iou = 0.0
        for truth in truth_by_photo.get(key, []):
            truth_index = id(truth)
            if truth_index in used:
                continue
            current_iou = iou_xyxy(prediction.get("bbox_xyxy") or (), truth.get("bbox_xyxy") or ())
            if current_iou > best_iou:
                best_index = truth_index
                best_truth = truth
                best_iou = current_iou
        if best_truth is not None and best_iou >= iou_threshold:
            used.add(best_index)
            matches.append((prediction, best_truth, best_iou))
        else:
            matches.append((prediction, None, best_iou))
    return matches


def _detector_ap(predictions: list[dict[str, Any]], truths: list[dict[str, Any]], *, iou_threshold: float) -> float | None:
    matches = _best_matches(predictions, truths, iou_threshold=iou_threshold, score_fn=_detector_score)
    return _ap([(truth is not None and iou >= iou_threshold, _detector_score(prediction)) for prediction, truth, iou in matches], len(truths))


def _detector_ap50_95(predictions: list[dict[str, Any]], truths: list[dict[str, Any]]) -> float | None:
    values = [
        _detector_ap(predictions, truths, iou_threshold=round(0.50 + index * 0.05, 2))
        for index in range(10)
    ]
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _species_top1_correct(prediction: dict[str, Any], truth: dict[str, Any]) -> bool:
    predicted_key = _norm(prediction.get("accepted_taxon_key") or prediction.get("species_top1_accepted_taxon_key"))
    truth_key = _norm(truth.get("accepted_taxon_key"))
    if predicted_key and truth_key:
        return predicted_key == truth_key
    return _norm(prediction.get("species_top1_scientific_name")) == _norm(truth.get("scientific_name"))


def _species_top5_correct(prediction: dict[str, Any], truth: dict[str, Any]) -> bool:
    truth_key = _norm(truth.get("accepted_taxon_key"))
    predicted_keys = {_norm(value) for value in prediction.get("species_top5_accepted_taxon_keys", []) if value}
    if truth_key and predicted_keys:
        return truth_key in predicted_keys
    return _norm(truth.get("scientific_name")) in {_norm(value) for value in prediction.get("species_top5", [])}


def _accuracy(
    matches: list[tuple[dict[str, Any], dict[str, Any] | None, float]],
    predicate: Any,
    *,
    truth_count: int,
) -> float:
    if truth_count <= 0:
        return 0.0
    valid = [(prediction, truth) for prediction, truth, _iou in matches if truth is not None]
    return sum(1 for prediction, truth in valid if predicate(prediction, truth)) / truth_count


def _ap(results: list[tuple[bool, float]], truth_count: int) -> float | None:
    if truth_count <= 0:
        return None
    if not results:
        return 0.0
    ordered = sorted(results, key=lambda item: item[1], reverse=True)
    precisions: list[float] = []
    recalls: list[float] = []
    true_positive = 0
    false_positive = 0
    for is_correct, _score in ordered:
        if is_correct:
            true_positive += 1
        else:
            false_positive += 1
        precisions.append(true_positive / max(1, true_positive + false_positive))
        recalls.append(true_positive / truth_count)
    return _integrated_ap(recalls, precisions)


def _integrated_ap(recalls: list[float], precisions: list[float]) -> float:
    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]
    for index in range(len(mpre) - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]
    return ap


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, max(box[0], box[2]) - min(box[0], box[2])) * max(0.0, max(box[1], box[3]) - min(box[1], box[3]))


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _detector_score(row: dict[str, Any]) -> float:
    for key in ("detector_score", "objectness_score", "box_score", "score"):
        value = _optional_float(row.get(key))
        if value is not None:
            return value
    return _species_score(row)


def _species_score(row: dict[str, Any]) -> float:
    return _optional_float(row.get("species_top1_score")) or 0.0


def _norm(value: object) -> str:
    return " ".join(str(value or "").casefold().split())
