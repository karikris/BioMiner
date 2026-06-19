from __future__ import annotations

import pytest

from biominer.bioclip.diagnostics import (
    grouped_probability_summary,
    probability_entropy,
    topk_margin,
)


def test_topk_margin_returns_difference_between_top_two_scores() -> None:
    assert topk_margin([{"label": "a", "score": 0.8}, {"label": "b", "score": 0.3}]) == pytest.approx(0.5)


def test_probability_entropy_is_low_for_confident_prediction() -> None:
    assert probability_entropy([0.98, 0.01, 0.01]) < 0.12


def test_grouped_probability_summary_sums_groups_and_keeps_top_group() -> None:
    summary = grouped_probability_summary(
        scores={
            "a photo of an adult butterfly": 0.55,
            "a photo of a swallowtail butterfly": 0.25,
            "a photo of a moth": 0.10,
            "a photo of artwork or illustration": 0.10,
        },
        groups={
            "adult_butterfly": {"a photo of an adult butterfly", "a photo of a swallowtail butterfly"},
            "hard_negative": {"a photo of a moth", "a photo of artwork or illustration"},
        },
    )

    assert summary["top_group"] == "adult_butterfly"
    assert summary["group_scores"]["adult_butterfly"] == pytest.approx(0.80)
    assert summary["group_scores"]["hard_negative"] == pytest.approx(0.20)
