from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from biominer.cli import _parse_run_stages, build_parser, run
from biominer.evaluation.target_metrics import (
    TARGET_MARGIN_DISTRIBUTION_FILE,
    TARGET_VERIFICATION_METRICS_FILE,
    TARGET_VERIFICATION_REPORT_FILE,
    TARGET_VERIFICATION_REPORT_MARKDOWN_FILE,
    target_verification_evaluation_frame,
)
from biominer.flickr_fetch.geography import build_flickr_geography_frame
from biominer.run import REFERENCE_FIRST_PRODUCTION_STAGES
from biominer.reference_workflow_cli import (
    _reference_source_queries,
    resolve_reference_workflow_options,
)
from test_evaluation_holdouts import _frozen_holdout_pair, _leakage_register
from test_target_verification_metrics import _rows_for_frozen_holdouts


@pytest.mark.parametrize(
    "command",
    (
        "build-geographic-spread",
        "cluster-flickr-metadata",
        "plan",
        "fetch-metadata",
        "download",
        "build-support-embeddings",
        "build-prototypes",
        "train-classifier",
        "calibrate-classifier",
        "score-target-aware",
        "evaluate-target-verifier",
    ),
)
def test_reference_workflow_example_settings_cover_every_new_command(
    command: str,
) -> None:
    settings = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "reference_workflow.example.json"
    )
    args = build_parser().parse_args(
        [
            "references",
            command,
            "--settings-file",
            str(settings),
            "--dry-run",
        ]
    )

    resolved = resolve_reference_workflow_options(args)

    assert resolved.command == command


def test_reference_source_query_example_uses_source_specific_page_sizes() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "reference_source_queries.example.json"
    )

    queries = _reference_source_queries(path)

    assert [(source, query.page_size) for source, query in queries] == [
        ("GBIF", 300),
        ("iNaturalist", 200),
    ]


def test_reference_workflow_commands_are_nested_and_species_agnostic() -> None:
    parser = build_parser()
    top_level = parser._subparsers._group_actions[0].choices  # noqa: SLF001
    references = top_level["references"]
    commands = references._subparsers._group_actions[0].choices  # noqa: SLF001

    assert {
        "build-geographic-spread",
        "cluster-flickr-metadata",
        "plan",
        "fetch-metadata",
        "download",
        "export-review-queue",
        "import-review-decisions",
        "validate-readiness",
        "build-support-embeddings",
        "build-prototypes",
        "train-classifier",
        "calibrate-classifier",
        "score-target-aware",
        "evaluate-target-verifier",
    } <= set(commands)
    assert not any("papilio" in command for command in top_level)
    assert not any("demoleus" in command for command in commands)


def test_workflow_settings_are_typed_and_cli_values_win(tmp_path: Path) -> None:
    settings = tmp_path / "workflow.json"
    settings.write_text(
        json.dumps(
            {
                "schema_version": "target-aware-reference-cli-settings-v1",
                "commands": {
                    "cluster-flickr-metadata": {
                        "geography": "from-config.parquet",
                        "target_accepted_taxon_key": "gbif:1938069",
                        "output_dir": "from-config",
                        "minimum_cluster_images": 7,
                        "maximum_assignment_distance_km": 125.5,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "references",
            "cluster-flickr-metadata",
            "--settings-file",
            str(settings),
            "--minimum-cluster-images",
            "11",
            "--dry-run",
        ]
    )

    resolved = resolve_reference_workflow_options(args)

    assert resolved.values["geography"] == "from-config.parquet"
    assert resolved.values["minimum_cluster_images"] == 11
    assert resolved.values["maximum_assignment_distance_km"] == 125.5
    assert resolved.settings_fingerprint.startswith("sha256:")


def test_reference_first_workflow_selects_authoritative_stage_order() -> None:
    assert _parse_run_stages(None, workflow="reference-first") == (
        REFERENCE_FIRST_PRODUCTION_STAGES
    )
    assert _parse_run_stages("all", workflow="reference-first") == (
        REFERENCE_FIRST_PRODUCTION_STAGES
    )


def test_reference_first_run_parser_exposes_versioned_support_dependencies() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--taxon",
            "Example species",
            "--registry-dir",
            "registry",
            "--output-prefix",
            "runs",
            "--workflow",
            "reference-first",
            "--regional-candidates",
            "regional.parquet",
            "--reference-embeddings",
            "embeddings.parquet",
            "--classifier-artifact",
            "classifier",
            "--calibrator-artifact",
            "calibrator",
            "--dry-run",
        ]
    )

    assert args.workflow == "reference-first"
    assert args.regional_candidates == "regional.parquet"
    assert args.reference_embeddings == "embeddings.parquet"
    assert args.classifier_artifact == "classifier"
    assert args.calibrator_artifact == "calibrator"


def test_cluster_flickr_metadata_command_writes_real_artifacts(
    tmp_path: Path,
    capsys,
) -> None:  # noqa: ANN001 - pytest fixture.
    geography = build_flickr_geography_frame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "brisbane-1",
                "source_record_hash": "sha256:brisbane-1",
                "latitude": -27.4705,
                "longitude": 153.026,
                "accuracy": 16,
                "country_code": "AU",
                "admin1": "Queensland",
            },
            {
                "source": "flickr",
                "flickr_photo_id": "brisbane-2",
                "source_record_hash": "sha256:brisbane-2",
                "latitude": -27.471,
                "longitude": 153.027,
                "accuracy": 16,
                "country_code": "AU",
                "admin1": "Queensland",
            },
        ]
    )
    geography_path = tmp_path / "geography.parquet"
    geography.write_parquet(geography_path)
    output = tmp_path / "clusters"

    rc = run(
        build_parser().parse_args(
            [
                "references",
                "cluster-flickr-metadata",
                "--geography",
                str(geography_path),
                "--target-accepted-taxon-key",
                "gbif:1938069",
                "--output-dir",
                str(output),
                "--created-at",
                "2026-07-14T00:00:00Z",
            ]
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cluster_count"] == 1
    assert payload["assignment_count"] == 2
    assert (output / "flickr_geo_clusters.parquet").is_file()
    assert (output / "flickr_geo_assignments.parquet").is_file()


def test_geographic_spread_command_builds_from_pinned_parquet(
    tmp_path: Path,
    capsys,
) -> None:  # noqa: ANN001 - pytest fixture.
    occurrences = tmp_path / "occurrences.parquet"
    pl.DataFrame(
        [
            {
                "key": "occurrence-1",
                "acceptedTaxonKey": 1938069,
                "acceptedScientificName": "Papilio demoleus",
                "decimalLatitude": -27.4705,
                "decimalLongitude": 153.026,
                "coordinateUncertaintyInMeters": 25.0,
                "countryCode": "AU",
                "stateProvince": "Queensland",
                "datasetKey": "dataset-1",
                "datasetTitle": "Pinned fixture",
                "basisOfRecord": "HUMAN_OBSERVATION",
                "establishmentMeans": "NATIVE",
                "occurrenceStatus": "PRESENT",
                "eventDate": "2025-01-01",
                "issues": [],
            }
        ]
    ).write_parquet(occurrences)
    output = tmp_path / "spread"

    rc = run(
        build_parser().parse_args(
            [
                "references",
                "build-geographic-spread",
                "--accepted-taxon-key",
                "gbif:1938069",
                "--scientific-name",
                "Papilio demoleus",
                "--registry-version",
                "registry-v1",
                "--source-snapshot-version",
                "gbif-pinned-2026-07-14",
                "--occurrences",
                str(occurrences),
                "--output-dir",
                str(output),
                "--checkpoint-dir",
                str(tmp_path / "checkpoint"),
                "--retrieved-at",
                "2026-07-14T00:00:00Z",
            ]
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["row_count"] == 3
    assert (output / "taxon_geographic_spread.parquet").is_file()
    assert (output / "geographic_occurrence_evidence.parquet").is_file()
    assert (output / "geographic_spread_manifest.json").is_file()


def test_target_aware_dry_run_does_not_read_heavy_inputs(
    capsys,
) -> None:  # noqa: ANN001 - pytest fixture.
    args = build_parser().parse_args(
        [
            "references",
            "score-target-aware",
            "--candidate-set",
            "missing-candidate-set.json",
            "--known-negative-classes",
            "missing-negatives.json",
            "--visual-domain-classes",
            "missing-domains.json",
            "--score-map",
            "missing-scores.json",
            "--output",
            "scores.json",
            "--dry-run",
        ]
    )

    assert run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "references score-target-aware"
    assert payload["stage"] == "target_aware_scoring"
    assert payload["status"] == "planned"
    assert payload["dry_run"] is True


def test_target_aware_scoring_commits_complete_regional_union(
    tmp_path: Path,
    capsys,
) -> None:  # noqa: ANN001 - pytest fixture.
    candidate_set = _write_json(tmp_path / "candidates.json", _candidate_set())
    negatives = _write_json(
        tmp_path / "negatives.json",
        {
            "classes": [
                {
                    "class_id": "other_insect",
                    "display_name": "other insect",
                    "source_versions": ["negative-v1"],
                }
            ]
        },
    )
    domains = _write_json(
        tmp_path / "domains.json",
        {
            "classes": [
                {
                    "class_id": "adult_field",
                    "display_name": "adult field butterfly",
                    "source_versions": ["domain-v1"],
                }
            ]
        },
    )
    score_map = _write_json(
        tmp_path / "scores.json",
        {
            "scores": {
                "species:gbif:1938069": 0.1,
                "species:gbif:competitor": 0.9,
                "known_negative:other_insect": 0.2,
                "visual_domain:adult_field": 0.8,
                "family_diagnostic:Papilionidae": 0.6,
                "genus_diagnostic:Papilio": 0.5,
            }
        },
    )
    output = tmp_path / "target-aware-result.json"

    rc = run(
        build_parser().parse_args(
            [
                "references",
                "score-target-aware",
                "--candidate-set",
                str(candidate_set),
                "--known-negative-classes",
                str(negatives),
                "--visual-domain-classes",
                str(domains),
                "--score-map",
                str(score_map),
                "--output",
                str(output),
            ]
        )
    )

    assert rc == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    stdout = json.loads(capsys.readouterr().out)
    assert result["target_regional_rank"] == 2
    assert result["hierarchy_pruning_applied"] is False
    assert result["hierarchy_rankings_diagnostic_only"] is True
    assert {
        row["accepted_taxon_key"]
        for row in result["scored_classes"]
        if row["class_kind"] == "species"
    } == {
        "gbif:1938069",
        "gbif:competitor",
    }
    assert stdout["artifacts"]["complete_set_scores"] == str(output)


def test_target_verifier_evaluation_publishes_frozen_holdout_metrics(
    tmp_path: Path,
    capsys,
) -> None:  # noqa: ANN001 - pytest fixture.
    evaluation_frame_path = tmp_path / "evaluation-frame.parquet"
    balanced_holdout_path = tmp_path / "balanced-holdout.parquet"
    natural_holdout_path = tmp_path / "natural-holdout.parquet"
    leakage_register_path = tmp_path / "leakage-register.parquet"
    output = tmp_path / "evaluation"
    balanced_holdout, natural_holdout = _frozen_holdout_pair()
    leakage_register = _leakage_register(balanced_holdout, natural_holdout)
    evaluation_frame = target_verification_evaluation_frame(
        _rows_for_frozen_holdouts(balanced_holdout, natural_holdout)
    )
    evaluation_frame.write_parquet(evaluation_frame_path)
    balanced_holdout.write_parquet(balanced_holdout_path)
    natural_holdout.write_parquet(natural_holdout_path)
    leakage_register.write_parquet(leakage_register_path)

    rc = run(
        build_parser().parse_args(
            [
                "references",
                "evaluate-target-verifier",
                "--evaluation-frame",
                str(evaluation_frame_path),
                "--balanced-holdout",
                str(balanced_holdout_path),
                "--natural-holdout",
                str(natural_holdout_path),
                "--leakage-register",
                str(leakage_register_path),
                "--output-dir",
                str(output),
                "--ece-bin-count",
                "7",
            ]
        )
    )

    assert rc == 0
    report = json.loads(
        (output / TARGET_VERIFICATION_REPORT_FILE).read_text(encoding="utf-8")
    )
    stdout = json.loads(capsys.readouterr().out)
    assert report["status"] == "complete"
    assert report["report_fingerprint"] == stdout["report_fingerprint"]
    assert stdout["sample_count"] == evaluation_frame.height
    assert stdout["status"] == "complete"
    assert stdout["artifacts"] == {
        "margin_distribution": str(output / TARGET_MARGIN_DISTRIBUTION_FILE),
        "metrics": str(output / TARGET_VERIFICATION_METRICS_FILE),
        "report": str(output / TARGET_VERIFICATION_REPORT_FILE),
        "summary": str(output / TARGET_VERIFICATION_REPORT_MARKDOWN_FILE),
    }
    assert (output / TARGET_VERIFICATION_METRICS_FILE).is_file()
    assert (output / TARGET_MARGIN_DISTRIBUTION_FILE).is_file()
    assert (output / TARGET_VERIFICATION_REPORT_MARKDOWN_FILE).is_file()


def test_target_verifier_evaluation_requires_leakage_register(capsys) -> None:  # noqa: ANN001
    rc = run(
        build_parser().parse_args(
            [
                "references",
                "evaluate-target-verifier",
                "--evaluation-frame",
                "evaluation-frame.parquet",
                "--balanced-holdout",
                "balanced-holdout.parquet",
                "--natural-holdout",
                "natural-holdout.parquet",
                "--output-dir",
                "evaluation",
                "--dry-run",
            ]
        )
    )

    assert rc == 2
    assert "--leakage-register" in json.loads(capsys.readouterr().out)["error"]


def _candidate_set() -> dict[str, object]:
    target = {
        "scientific_name": "Papilio demoleus",
        "accepted_taxon_key": "gbif:1938069",
        "rank": "species",
        "family": "Papilionidae",
        "genus": "Papilio",
        "common_names": [],
        "candidate_reasons": ["target"],
        "source_versions": ["regional-v1"],
        "target_candidate": True,
        "candidate_priority": 0,
    }
    competitor = {
        "scientific_name": "Papilio competitor",
        "accepted_taxon_key": "gbif:competitor",
        "rank": "species",
        "family": "Papilionidae",
        "genus": "Papilio",
        "common_names": [],
        "candidate_reasons": ["regional_same_genus"],
        "source_versions": ["regional-v1"],
        "target_candidate": False,
        "candidate_priority": 1,
    }
    return {
        "candidate_set_id": "regional:au-1",
        "registry_version": "registry-v1",
        "target_accepted_taxon_key": "gbif:1938069",
        "target_scientific_name": "Papilio demoleus",
        "family_candidates": [target, competitor],
        "genus_candidates": [target, competitor],
        "species_candidates": [target, competitor],
        "prompt_variant_version": "object-bioclip-prompts-v1",
        "geospatial_scope": "au-1",
        "source_evidence": ["regional_candidate_set:regional:au-1"],
        "candidate_set_fingerprint": "sha256:" + "a" * 64,
    }


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
