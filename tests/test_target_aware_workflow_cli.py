from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from biominer.cli import _parse_run_stages, build_parser, run
from biominer.evaluation.target_metrics import (
    EVALUATION_BOOTSTRAP_COMPONENT_FILE,
    TARGET_CALIBRATION_RELIABILITY_FILE,
    TARGET_MARGIN_DISTRIBUTION_FILE,
    TARGET_METRIC_CONFIDENCE_INTERVAL_FILE,
    TARGET_THRESHOLD_OPERATING_POINTS_FILE,
    TARGET_VERIFICATION_METRICS_FILE,
    TARGET_VERIFICATION_REPORT_FILE,
    TARGET_VERIFICATION_REPORT_MARKDOWN_FILE,
    target_verification_evaluation_frame,
)
from biominer.flickr_fetch.geography import build_flickr_geography_frame
from biominer.run import REFERENCE_FIRST_PRODUCTION_STAGES
from biominer.reference_workflow_cli import (
    _prototype_duplicate_config,
    _reference_download_config,
    _reference_download_output_prefix,
    _reference_licence_policy,
    _reference_source_queries,
    resolve_reference_workflow_options,
)
from test_evaluation_holdouts import _frozen_holdout_pair, _leakage_register
from test_target_verification_metrics import _rows_for_frozen_holdouts


@pytest.mark.parametrize(
    "command",
    (
        "build-geographic-spread",
        "build-regional-competitor-evidence",
        "cluster-flickr-metadata",
        "materialize-flickr-workload",
        "plan",
        "fetch-metadata",
        "finalize-prototype-acquisition",
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


def test_papilio_pilot_config_caps_local_vision_verification() -> None:
    settings = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "pilot"
        / "papilio_demoleus_geographic_workload.json"
    )
    payload = json.loads(settings.read_text(encoding="utf-8"))
    constraints = payload["pilot"]["execution_constraints"]

    assert constraints == {
        "large_bioclip_runs": "different_computer_required",
        "large_yoloe_runs": "different_computer_required",
        "local_build_verification_max_images": 5,
    }
    resolved = resolve_reference_workflow_options(
        build_parser().parse_args(
            [
                "references",
                "materialize-flickr-workload",
                "--settings-file",
                str(settings),
                "--dry-run",
            ]
        )
    )
    assert resolved.values["target_accepted_taxon_key"] == "gbif:1938069"
    assert resolved.values["candidate_metadata_sha256"] == (
        "sha256:fe42f248cab68f6c3f67351800718fb9"
        "888b54a44f3b4a651d0f8bfa428c015d"
    )


def test_papilio_prototype_download_config_is_s3_portable_and_fail_closed() -> None:
    settings = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "pilot"
        / "papilio_demoleus_prototype_download.json"
    )
    resolved = resolve_reference_workflow_options(
        build_parser().parse_args(
            [
                "references",
                "download",
                "--settings-file",
                str(settings),
                "--dry-run",
            ]
        )
    )

    config = _reference_download_config(resolved.values)
    policy = _reference_licence_policy(resolved.values)

    assert resolved.values["storage_backend"] == "s3"
    assert resolved.values["storage_bucket"] == "biominer"
    assert resolved.values["output_prefix"] == (
        "reference-media/papilio-demoleus/prototype-20260715"
    )
    assert {
        item.source: item.allowed_hosts for item in config.provider_policies
    } == {
        "GBIF": ("observation.org",),
        "Wikimedia Commons": ("upload.wikimedia.org",),
    }
    assert config.max_concurrent_decodes == 1
    assert policy.version == "prototype-reference-licences-v1"
    assert "public-domain" in policy.broadly_reusable
    assert "public-domain" in policy.attribution_required


def test_reference_download_resolves_relative_prefix_against_s3_storage() -> None:
    class _Storage:
        base_uri = "s3://prototype-bucket/biominer"

    assert _reference_download_output_prefix(
        "reference-media/papilio-demoleus/prototype-20260715/",
        storage=_Storage(),
    ) == (
        "s3://prototype-bucket/biominer/reference-media/"
        "papilio-demoleus/prototype-20260715"
    )
    assert _reference_download_output_prefix(
        "s3://explicit-bucket/reference-media",
        storage=_Storage(),
    ) == "s3://explicit-bucket/reference-media"


def test_papilio_prototype_duplicate_config_freezes_auditable_thresholds() -> None:
    settings = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "pilot"
        / "papilio_demoleus_prototype_duplicates.json"
    )
    resolved = resolve_reference_workflow_options(
        build_parser().parse_args(
            [
                "references",
                "resolve-prototype-duplicates",
                "--settings-file",
                str(settings),
                "--dry-run",
            ]
        )
    )

    config = _prototype_duplicate_config(resolved.values)

    assert resolved.values["storage_backend"] == "local"
    assert "storage_bucket" not in resolved.values
    assert len(resolved.values["biological_observations"]) == 2
    assert config.same_observation_distance_threshold == 8
    assert config.cross_observation_distance_threshold == 4
    assert config.minimum_informative_bits == 8
    assert config.policy_version == "prototype-duplicate-resolution-policy-v1"


def test_papilio_pilot_reference_source_plan_freezes_phase14_quotas() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "pilot"
        / "papilio_demoleus_reference_source_queries.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload["queries"]
    query_keys = [query["accepted_taxon_key"] for query in queries]
    quotas = payload["acquisition_quotas"]

    assert payload["candidate_semantics"] == (
        "source_taxon_match_not_human_verified_image_label"
    )
    assert len(query_keys) == len(set(query_keys)) == 22
    assert set(query["source"] for query in queries) == {"GBIF"}
    machaon = next(
        query for query in queries if query["accepted_taxon_key"] == "gbif:8225376"
    )
    assert all(
        query["fallback_level"] == 3
        for query in queries
        if query is not machaon
    )
    assert machaon["fallback_level"] == 2
    assert len(machaon["country_codes"]) == 21
    assert payload["query_scope_policy"]["papilio_machaon"][
        "verified_scoped_count"
    ] == 638
    assert all(query["page_size"] == 300 for query in queries)
    assert quotas["target_adult"] == {
        "species": ["gbif:1938069"],
        "life_stage": "adult",
        "minimum_per_species": 50,
        "maximum_per_species": 100,
    }
    assert len(quotas["selected_regional_competitors"]["species"]) == 5
    assert len(quotas["reviewed_false_winner_genera"]["species"]) == 5
    assert quotas["historical_false_winner_species"]["species"] == [
        "gbif:1937474"
    ]
    broad = quotas["broader_papilionidae"]
    assert len(broad["species"]) * broad["planned_per_species"] == 200
    assert broad["minimum_total"] == 100
    assert broad["maximum_total"] == 300
    assert quotas["target_caterpillar"]["separate_from_adult_bank"] is True
    assert quotas["target_caterpillar"]["life_stage"] == "larva"
    assert quotas["other_insect_or_moth_negatives"]["status"].startswith(
        "unresolved"
    )
    assert quotas["domain_negatives"]["status"].startswith("unresolved")
    assert set(query_keys) == {
        key
        for name, group in quotas.items()
        if name not in {"other_insect_or_moth_negatives", "domain_negatives"}
        for key in group["species"]
    }


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


def test_reference_source_queries_apply_bounded_shared_defaults(tmp_path: Path) -> None:
    path = tmp_path / "queries.json"
    path.write_text(
        json.dumps(
            {
                "query_defaults": {
                    "geo_cluster_id": "unassigned_geo",
                    "fallback_level": 2,
                    "country_codes": ["AU", "IN"],
                    "page_size": 300,
                    "maximum_records": 3000,
                    "source_snapshot_version": "regional-snapshot-v1",
                },
                "queries": [
                    {
                        "source": "GBIF",
                        "accepted_taxon_key": "gbif:1",
                        "scientific_name": "Example species",
                        "source_taxon_id": "1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    ((source, query),) = _reference_source_queries(path)

    assert source == "GBIF"
    assert query.fallback_level == 2
    assert query.country_codes == ("AU", "IN")
    assert query.maximum_records == 3000
    assert query.source_snapshot_version == "regional-snapshot-v1"


def test_reference_workflow_commands_are_nested_and_species_agnostic() -> None:
    parser = build_parser()
    top_level = parser._subparsers._group_actions[0].choices  # noqa: SLF001
    references = top_level["references"]
    commands = references._subparsers._group_actions[0].choices  # noqa: SLF001

    assert {
        "build-geographic-spread",
        "build-regional-competitor-evidence",
        "cluster-flickr-metadata",
        "materialize-flickr-workload",
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


def test_settings_fingerprint_excludes_overwrite_execution_control(
    tmp_path: Path,
) -> None:
    common = [
        "references",
        "cluster-flickr-metadata",
        "--geography",
        str(tmp_path / "geography.parquet"),
        "--target-accepted-taxon-key",
        "gbif:1938069",
        "--output-dir",
        str(tmp_path / "output"),
        "--dry-run",
    ]

    default = resolve_reference_workflow_options(build_parser().parse_args(common))
    overwrite = resolve_reference_workflow_options(
        build_parser().parse_args([*common, "--overwrite"])
    )

    assert default.values["overwrite"] is False
    assert overwrite.values["overwrite"] is True
    assert default.settings_fingerprint == overwrite.settings_fingerprint


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


def test_materialize_flickr_workload_pins_input_and_retains_query_hits(
    tmp_path: Path,
    capsys,
) -> None:  # noqa: ANN001 - pytest fixture.
    source = tmp_path / "flickr-hits.ndjson"
    rows = [
        {
            "accuracy": 16,
            "fetched_at": "2026-07-14T00:00:00Z",
            "flickr_photo_id": "photo-1",
            "latitude": -27.4705,
            "longitude": 153.026,
            "query_field": "tags",
            "query_hash": "query-a",
            "query_term": "Papilio demoleus",
            "query_term_confidence": "high",
            "query_term_type": "scientific_name",
            "raw_photo_json": '{"id":"photo-1","revision":1}',
        },
        {
            "accuracy": 16,
            "fetched_at": "2026-07-13T00:00:00Z",
            "flickr_photo_id": "photo-1",
            "latitude": -27.4705,
            "longitude": 153.026,
            "query_field": "text",
            "query_hash": "query-b",
            "query_term": "lime butterfly",
            "query_term_confidence": "medium",
            "query_term_type": "common_name",
            "raw_photo_json": '{"id":"photo-1","revision":0}',
        },
        {
            "accuracy": 16,
            "fetched_at": "2026-07-14T00:00:00Z",
            "flickr_photo_id": "photo-2",
            "latitude": -27.471,
            "longitude": 153.027,
            "query_field": "tags",
            "query_hash": "query-c",
            "query_term": "Papilio demoleus",
            "query_term_confidence": "high",
            "query_term_type": "scientific_name",
            "raw_photo_json": '{"id":"photo-2"}',
        },
    ]
    source.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    output = tmp_path / "workload"
    reports = tmp_path / "reports"
    source_sha256 = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()

    rc = run(
        build_parser().parse_args(
            [
                "references",
                "materialize-flickr-workload",
                "--candidate-metadata",
                str(source),
                "--candidate-metadata-byte-count",
                str(source.stat().st_size),
                "--candidate-metadata-sha256",
                source_sha256,
                "--target-accepted-taxon-key",
                "gbif:1938069",
                "--output-dir",
                str(output),
                "--report-dir",
                str(reports),
                "--created-at",
                "2026-07-14T00:00:00Z",
                "--source-cell-field",
                "coarse_cell_id",
                "--source-resolution",
                "3",
                "--minimum-cluster-images",
                "1",
            ]
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["canonical_photo_count"] == 2
    assert payload["query_hit_count"] == 3
    assert payload["cluster_count"] == 1
    assert pl.read_parquet(output / "flickr_query_hits.parquet").height == 3
    manifest = json.loads(
        (output / "flickr_workload_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["candidate_semantics"] == (
        "flickr_search_candidate_not_taxonomic_label"
    )
    assert manifest["schema_version"] == "flickr-workload-manifest-v1.1.0"
    assert manifest["git_sha"]
    assert manifest["elapsed_seconds"] >= 0.0
    assert manifest["artifact_schema_versions"] == {
        "assignments": "flickr-geo-assignments-v1.1.0",
        "clusters": "flickr-geo-clusters-v1.1.0",
        "geography": "flickr-geography-v1.0.0",
        "input_projection": "flickr-workload-input-v1.0.0",
        "workload_report": "flickr-geographic-workload-report-v1.1.0",
    }
    assert manifest["source"]["sha256"] == source_sha256
    assert (output / "flickr_geography.parquet").is_file()
    assert (output / "flickr_geo_clusters.parquet").is_file()
    assert (output / "flickr_geo_assignments.parquet").is_file()
    assert (reports / "flickr_geographic_workload.json").is_file()
    assert (reports / "flickr_geographic_workload.md").is_file()


def test_materialize_flickr_workload_rejects_changed_snapshot(
    tmp_path: Path,
    capsys,
) -> None:  # noqa: ANN001 - pytest fixture.
    source = tmp_path / "flickr-hits.ndjson"
    source.write_text("{}\n", encoding="utf-8")

    rc = run(
        build_parser().parse_args(
            [
                "references",
                "materialize-flickr-workload",
                "--candidate-metadata",
                str(source),
                "--candidate-metadata-byte-count",
                str(source.stat().st_size + 1),
                "--target-accepted-taxon-key",
                "gbif:1938069",
                "--output-dir",
                str(tmp_path / "workload"),
                "--created-at",
                "2026-07-14T00:00:00Z",
            ]
        )
    )

    assert rc == 2
    assert "byte count does not match" in json.loads(capsys.readouterr().out)["error"]


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
                "--bootstrap-replicate-count",
                "64",
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
        "bootstrap_components": str(
            output / EVALUATION_BOOTSTRAP_COMPONENT_FILE
        ),
        "calibration_reliability": str(
            output / TARGET_CALIBRATION_RELIABILITY_FILE
        ),
        "margin_distribution": str(output / TARGET_MARGIN_DISTRIBUTION_FILE),
        "metrics": str(output / TARGET_VERIFICATION_METRICS_FILE),
        "confidence_intervals": str(
            output / TARGET_METRIC_CONFIDENCE_INTERVAL_FILE
        ),
        "report": str(output / TARGET_VERIFICATION_REPORT_FILE),
        "summary": str(output / TARGET_VERIFICATION_REPORT_MARKDOWN_FILE),
        "threshold_operating_points": str(
            output / TARGET_THRESHOLD_OPERATING_POINTS_FILE
        ),
    }
    assert (output / TARGET_VERIFICATION_METRICS_FILE).is_file()
    assert (output / TARGET_MARGIN_DISTRIBUTION_FILE).is_file()
    assert (output / TARGET_CALIBRATION_RELIABILITY_FILE).is_file()
    assert (output / TARGET_THRESHOLD_OPERATING_POINTS_FILE).is_file()
    assert (output / TARGET_METRIC_CONFIDENCE_INTERVAL_FILE).is_file()
    assert (output / EVALUATION_BOOTSTRAP_COMPONENT_FILE).is_file()
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
