from __future__ import annotations

import json
from pathlib import Path

import pytest

from biominer.cli import build_parser


def test_authoritative_docs_and_classification_source_exist() -> None:
    for path in (
        Path("README.md"),
        Path("docs/registry.md"),
        Path("docs/production.md"),
        Path("docs/vision.md"),
        Path("config/taxonomy/papilionoidea_classification_v2.json"),
        Path("config/taxonomy/papilionoidea_classification_v3.json"),
        Path("config/vision_profiles/mac_m5pro_64gb.json"),
    ):
        assert path.exists(), path


def test_classification_source_has_reviewed_five_rank_path() -> None:
    payload = json.loads(Path("config/taxonomy/papilionoidea_classification_v2.json").read_text(encoding="utf-8"))

    assert [node["rank"] for node in payload["nodes"]] == ["FAMILY", "SUBFAMILY", "TRIBE", "GENUS", "SPECIES"]
    assert [node["scientific_name"] for node in payload["nodes"]] == [
        "Papilionidae",
        "Papilioninae",
        "Papilionini",
        "Papilio",
        "Papilio demoleus",
    ]
    assert payload["species_mappings"][0]["gbif_species_key"] == "1938069"
    assert all(row["review_status"] == "reviewed" for row in [*payload["nodes"], *payload["edges"], *payload["species_mappings"]])


def test_removed_visual_commands_have_no_parser_aliases() -> None:
    parser = build_parser()
    for command in (
        ["vision", "detect"],
        ["vision", "screen"],
        ["vision", "rolling-screen"],
        ["vision", "score"],
        ["vision", "ablate"],
        ["bioclip", "screen"],
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
        "biominer bioclip screen",
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
