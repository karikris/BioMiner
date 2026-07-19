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


def test_authoritative_examples_separate_registry_and_run_roots() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (Path("README.md"), Path("docs/registry.md"), Path("docs/production.md"))
    )

    for required in (
        "--output-dir data/registry/current",
        "--registry-dir data/registry/build",
        "--output-prefix s3://biominer/runs/current",
    ):
        assert required in text
    for forbidden in (
        "build-text-embedding-cache",
        "classification_text_embeddings.parquet",
        "--taxonomy-text-embedding-cache data/registry/butterflies-v2",
        "--taxonomy-text-embedding-cache s3://biominer/registry/butterflies-v2",
    ):
        assert forbidden not in text


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
