from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from biominer.bioclip.prompt_templates import PromptVariant, aggregate_prompt_scores


def _prompt_variant(label: str, taxon_key: str, prompt_kind: str) -> PromptVariant:
    return PromptVariant(
        label=label,
        taxon_key=taxon_key,
        prompt_kind=prompt_kind,
        prompt_version="test-prompt-v1",
        template_id=f"test-{prompt_kind}-v1",
        evidence_kind="test_fixture",
    )


def test_phase4_species_prompt_variants_rank_by_best_two_not_best_single_prompt() -> (
    None
):
    rows = aggregate_prompt_scores(
        scores={
            "a photo of Papilio demoleus": 0.99,
            "a field photo of Papilio demoleus adult butterfly": 0.0,
            "a close-up photo of Papilio demoleus butterfly": 0.0,
            "a photo of Papilio machaon": 0.50,
            "a field photo of Papilio machaon adult butterfly": 0.50,
            "a close-up photo of Papilio machaon butterfly": 0.50,
        },
        variants=[
            _prompt_variant(
                "a photo of Papilio demoleus", "Papilio demoleus", "scientific"
            ),
            _prompt_variant(
                "a field photo of Papilio demoleus adult butterfly",
                "Papilio demoleus",
                "field_adult",
            ),
            _prompt_variant(
                "a close-up photo of Papilio demoleus butterfly",
                "Papilio demoleus",
                "close_adult",
            ),
            _prompt_variant(
                "a photo of Papilio machaon", "Papilio machaon", "scientific"
            ),
            _prompt_variant(
                "a field photo of Papilio machaon adult butterfly",
                "Papilio machaon",
                "field_adult",
            ),
            _prompt_variant(
                "a close-up photo of Papilio machaon butterfly",
                "Papilio machaon",
                "close_adult",
            ),
        ],
        top_k=2,
    )

    assert [row["taxon_key"] for row in rows] == ["Papilio machaon", "Papilio demoleus"]
    assert rows[0]["score"] == pytest.approx(0.50)
    assert rows[1]["score"] == pytest.approx(0.495)
    assert rows[1]["best_label"] == "a photo of Papilio demoleus"


def test_phase4_heavy_vision_runtimes_are_not_required_dependencies() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = {
        _dependency_name(value) for value in pyproject["project"]["dependencies"]
    }

    assert "torch" not in dependencies
    assert "open-clip-torch" not in dependencies
    assert "open_clip_torch" not in dependencies
    assert "ultralytics" not in dependencies


def _dependency_name(value: object) -> str:
    package = str(value).split(";", maxsplit=1)[0]
    package = package.split("[", maxsplit=1)[0]
    return package.split(">=", maxsplit=1)[0].casefold()
