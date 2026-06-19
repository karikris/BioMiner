from __future__ import annotations

from math import log
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
