from __future__ import annotations

from pathlib import Path

import pytest

from biominer.cli import build_parser


def test_authoritative_unified_registry_docs_exist() -> None:
    for path in (
        Path("README.md"),
        Path("docs/registry.md"),
        Path("docs/production.md"),
        Path("docs/vision.md"),
        Path("docs/migrations/unified-registry.md"),
        Path("config/vision_profiles/mac_m5pro_64gb.json"),
    ):
        assert path.exists(), path


def test_authoritative_examples_separate_base_overlay_cache_and_run_roots() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (Path("README.md"), Path("docs/registry.md"), Path("docs/production.md"))
    )

    for required in (
        "--output-dir data/registry/current",
        "--registry-dir data/registry/current",
        "--output data/cache/taxonomy/current/classification_text_embeddings.parquet",
        "--taxonomy-text-embedding-cache s3://biominer/cache/taxonomy/current/classification_text_embeddings.parquet",
    ):
        assert required in text
    for forbidden in (
        "--taxonomy-candidate-table data/registry/butterflies-v2",
        "--taxonomy-text-embedding-cache data/registry/butterflies-v2",
        "--taxonomy-candidate-table s3://biominer/registry/butterflies-v2",
        "--taxonomy-text-embedding-cache s3://biominer/registry/butterflies-v2",
    ):
        assert forbidden not in text


def test_superseded_classification_v2_assets_are_removed() -> None:
    for path in (
        Path("src/biominer/registry/classification_v2.py"),
        Path("src/biominer/bioclip/five_rank_classifier.py"),
        Path("src/biominer/bioclip/five_rank_store.py"),
        Path("src/biominer/bioclip/five_rank_embedding_cache.py"),
        Path("config/taxonomy/papilionoidea_classification_v2.json"),
    ):
        assert not path.exists(), path


def test_removed_visual_commands_have_no_parser_aliases() -> None:
    parser = build_parser()
    for command in (
        ["vision", "detect"],
        ["vision", "screen"],
        ["vision", "rolling-screen"],
        ["vision", "score"],
        ["vision", "ablate"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(command)


def test_readme_and_authoritative_docs_do_not_reference_removed_commands() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (Path("README.md"), Path("docs/registry.md"), Path("docs/production.md"), Path("docs/vision.md"))
    )
    for removed in (
        "biominer vision detect",
        "biominer vision screen",
        "biominer vision rolling-screen",
        "biominer vision score",
        "biominer vision ablate",
        "build-classification-table",
    ):
        assert removed not in text


def test_production_run_defaults_to_rolling_worker_without_public_switch() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--taxon",
            "Papilionoidea",
            "--rank",
            "family",
            "--registry-dir",
            "registry",
            "--output-prefix",
            "runs",
        ]
    )

    assert args.command == "run"
    assert not hasattr(args, "vision_worker")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--taxon",
                "Papilionoidea",
                "--registry-dir",
                "registry",
                "--output-prefix",
                "runs",
                "--vision-worker",
                "serial",
            ]
        )
