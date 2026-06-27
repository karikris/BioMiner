from __future__ import annotations

from math import log
import sys
from typing import Mapping, Sequence


TRIAGE_LABEL_GROUPS = {
    "adult_butterfly": {
        "a photo of an adult butterfly",
        "a photo of a swallowtail butterfly",
        "a photo of a butterfly",
    },
    "life_stage": {
        "a photo of an egg",
        "a photo of a caterpillar",
        "a photo of a larva",
        "a photo of a pupa",
        "a photo of a chrysalis",
    },
    "hard_negative": {
        "a photo of a moth",
        "a photo of a pinned museum specimen",
        "a photo of artwork or illustration",
        "a photo of a tattoo",
        "an ai generated image",
        "a photo of a logo or brand",
        "a photo of an object",
        "a photo of a textile or pattern",
        "a photo of an insect that is not a butterfly or moth",
        "a photo that is not a lepidoptera",
    },
}


def topk_margin(topk: Sequence[Mapping[str, object]]) -> float | None:
    scores = [float(row["score"]) for row in topk if row.get("score") is not None]
    if len(scores) < 2:
        return None
    return scores[0] - scores[1]


def probability_entropy(scores: Sequence[float]) -> float:
    total = sum(max(0.0, float(score)) for score in scores)
    if total <= 0:
        return 0.0
    probabilities = [max(0.0, float(score)) / total for score in scores]
    return -sum(probability * log(probability) for probability in probabilities if probability > 0)


def grouped_probability_summary(
    *,
    scores: Mapping[str, float],
    groups: Mapping[str, set[str]],
) -> dict[str, object]:
    group_scores = {
        group_name: sum(float(scores.get(label, 0.0)) for label in labels)
        for group_name, labels in groups.items()
    }
    top_group = max(group_scores.items(), key=lambda item: item[1])[0] if group_scores else None
    return {"top_group": top_group, "group_scores": group_scores}


def mps_memory_metrics(torch_module: object | None = None) -> dict[str, object]:
    torch = torch_module
    if torch is None:
        try:
            import torch as imported_torch
        except Exception:  # noqa: BLE001 - PyTorch is optional in the main runtime.
            return _mps_metrics(
                available=False,
                current=None,
                driver=None,
                recommended=None,
                note="torch_unavailable",
            )
        torch = imported_torch
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    available = bool(mps is not None and mps.is_available())
    if not available:
        return _mps_metrics(
            available=False,
            current=None,
            driver=None,
            recommended=None,
            note="mps_unavailable",
        )
    return _mps_metrics(
        available=True,
        current=_call_torch_mps_metric(torch, "current_allocated_memory"),
        driver=_call_torch_mps_metric(torch, "driver_allocated_memory"),
        recommended=_call_torch_mps_metric(torch, "recommended_max_memory"),
        note=None,
    )


def _mps_metrics(
    *,
    available: bool,
    current: int | None,
    driver: int | None,
    recommended: int | None,
    note: str | None,
) -> dict[str, object]:
    return {
        "mps_available": available,
        "mps_current_allocated_memory_bytes": current,
        "mps_driver_allocated_memory_bytes": driver,
        "mps_recommended_max_memory_bytes": recommended,
        "mps_metrics_note": note,
        "platform": sys.platform,
    }


def _call_torch_mps_metric(torch: object, name: str) -> int | None:
    mps_module = getattr(torch, "mps", None)
    metric = getattr(mps_module, name, None)
    if metric is None:
        return None
    try:
        value = metric()
    except Exception:  # noqa: BLE001 - unsupported metric stays null, never guessed.
        return None
    return int(value) if value is not None else None
