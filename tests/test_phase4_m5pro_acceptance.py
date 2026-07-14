from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from biominer.bioclip.classification_modes import (
    DEFAULT_CLASSIFICATION_MODE,
    DEFAULT_RANK_BEAM_WIDTH,
    DEFAULT_SPECIES_FIRST_PASS_TOP_K,
    DEFAULT_SPECIES_RERANK_TOP_K,
    HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
    TARGET_SCOPE_OBJECT_SCREENING,
)
from biominer.bioclip.prompt_templates import PromptVariant, aggregate_prompt_scores
from biominer.cli import build_parser
from biominer.detection.policy import vision_runtime_settings


def test_phase4_m5pro_profile_and_classifier_defaults_remain_acceptance_values() -> (
    None
):
    settings = vision_runtime_settings("mac_m5pro_64gb")

    assert settings.device == "mps"
    assert settings.yolo_checkpoint == "yoloe-26s-seg.pt"
    assert settings.yolo_imgsz == 768
    assert settings.detector_batch_size == 16
    assert settings.crop_batch_size == 24
    assert settings.crop_target_px == 336
    assert settings.crop_padding_ratio == 0.08
    assert settings.parquet_compression == "zstd"
    assert settings.delete_images_after_commit is True
    assert settings.adaptive_batching is False
    assert DEFAULT_CLASSIFICATION_MODE == TARGET_SCOPE_OBJECT_SCREENING
    assert (
        HIERARCHICAL_BUTTERFLY_CLASSIFICATION == "hierarchical_butterfly_classification"
    )
    assert DEFAULT_RANK_BEAM_WIDTH == 3
    assert DEFAULT_SPECIES_FIRST_PASS_TOP_K == 20
    assert DEFAULT_SPECIES_RERANK_TOP_K == 5


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
            PromptVariant(
                "a photo of Papilio demoleus", "Papilio demoleus", "scientific"
            ),
            PromptVariant(
                "a field photo of Papilio demoleus adult butterfly",
                "Papilio demoleus",
                "field_adult",
            ),
            PromptVariant(
                "a close-up photo of Papilio demoleus butterfly",
                "Papilio demoleus",
                "close_adult",
            ),
            PromptVariant(
                "a photo of Papilio machaon", "Papilio machaon", "scientific"
            ),
            PromptVariant(
                "a field photo of Papilio machaon adult butterfly",
                "Papilio machaon",
                "field_adult",
            ),
            PromptVariant(
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


def test_phase4_docs_and_cli_expose_supported_developer_benchmarks() -> None:
    parser = build_parser()
    live_args = parser.parse_args(
        [
            "dev",
            "vision",
            "benchmark-live-m5pro",
            "--input",
            "runs/local_debug/papilio_demoleus/canonical_source_records.parquet",
            "--taxonomy-candidate-table",
            "data/registry/current",
            "--taxonomy-text-embedding-cache",
            "data/registry/current/taxonomy_text_embeddings.parquet",
            "--output-dir",
            "reports/vision_benchmarks/m5pro_live",
        ]
    )
    plumbing_args = parser.parse_args(
        [
            "dev",
            "vision",
            "benchmark-plumbing",
            "--output-dir",
            "reports/vision_benchmarks/plumbing",
        ]
    )
    vision_doc = Path("docs/vision.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")

    assert live_args.vision_command == "benchmark-live-m5pro"
    assert plumbing_args.vision_command == "benchmark-plumbing"
    assert "uv run biominer dev vision benchmark-plumbing" in vision_doc
    assert "uv run biominer dev vision benchmark-rolling-matrix" in vision_doc
    assert "persistent worker" in vision_doc
    assert "family top 1" in readme


def _dependency_name(value: object) -> str:
    package = str(value).split(";", maxsplit=1)[0]
    package = package.split("[", maxsplit=1)[0]
    return package.split(">=", maxsplit=1)[0].casefold()
