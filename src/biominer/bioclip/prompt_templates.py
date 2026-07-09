from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PromptVariant:
    label: str
    taxon_key: str
    prompt_kind: str


SPECIES_PROMPT_TEMPLATES = (
    ("scientific", "a photo of {name}"),
    ("field_adult", "a field photo of {name} adult butterfly"),
    ("close_adult", "a close-up photo of {name} butterfly"),
)

COMMON_NAME_PROMPT_TEMPLATES = (
    ("common", "a photo of {name}"),
    ("common_adult", "a field photo of {name} adult butterfly"),
)

SPECIES_PROMPT_AGGREGATION_DEFAULT = "mean"


def build_species_prompt_variants(
    *,
    scientific_name: str,
    common_names: Sequence[str] = (),
) -> list[PromptVariant]:
    variants: list[PromptVariant] = []
    seen: set[str] = set()
    for kind, template in SPECIES_PROMPT_TEMPLATES:
        label = template.format(name=scientific_name)
        if label not in seen:
            variants.append(PromptVariant(label=label, taxon_key=scientific_name, prompt_kind=kind))
            seen.add(label)
    for common_name in common_names:
        for kind, template in COMMON_NAME_PROMPT_TEMPLATES:
            label = template.format(name=common_name)
            if label not in seen:
                variants.append(PromptVariant(label=label, taxon_key=scientific_name, prompt_kind=kind))
                seen.add(label)
    return variants


def aggregate_prompt_scores(
    *,
    scores: Mapping[str, float],
    variants: Sequence[PromptVariant],
    top_k: int,
    aggregation: str = SPECIES_PROMPT_AGGREGATION_DEFAULT,
) -> list[dict[str, object]]:
    if aggregation not in {"max", "mean"}:
        raise ValueError("aggregation must be one of: max, mean")
    grouped: dict[str, dict[str, object]] = {}
    for variant in variants:
        score = float(scores.get(variant.label, 0.0))
        current = grouped.setdefault(
            variant.taxon_key,
            {"taxon_key": variant.taxon_key, "score": 0.0, "best_label": None, "prompt_scores": {}},
        )
        prompt_scores = current["prompt_scores"]
        assert isinstance(prompt_scores, dict)
        prompt_scores[variant.label] = score
        if current["best_label"] is None or score >= float(scores.get(str(current["best_label"]), 0.0)):
            current["best_label"] = variant.label
    rows = []
    for current in grouped.values():
        prompt_scores = current["prompt_scores"]
        assert isinstance(prompt_scores, dict)
        values = [float(value) for value in prompt_scores.values()]
        current["score"] = _aggregate_values(values, aggregation=aggregation)
        rows.append(current)
    return sorted(rows, key=lambda row: (-float(row["score"]), str(row["taxon_key"])))[:top_k]


def _aggregate_values(values: Sequence[float], *, aggregation: str) -> float:
    if not values:
        return 0.0
    if aggregation == "max":
        return max(values)
    return sum(values) / len(values)
