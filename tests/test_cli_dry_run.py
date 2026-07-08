from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from biominer.config import BioMinerConfig, RuntimeConfig, StorageConfig, WorkStoreConfig
from biominer.cli import _detect_boxes_backend, _production_vision_settings_from_args, _yoloe26_metrics, build_parser, load_decoded_image_from_record, run
from biominer.detection.detector_base import DecodedImage, DetectionCandidate
from biominer.detection.policy import DetectionPolicy, DetectionRunPolicy
from biominer.registry.enrichment import DEFAULT_ENRICHMENT_SOURCES
from biominer.registry.translation_harvester import (
    MYMEMORY_MONTHLY_BANDWIDTH_MB_LIMIT,
    MYMEMORY_MONTHLY_INPUT_WORD_LIMIT,
    MYMEMORY_MONTHLY_REQUEST_LIMIT,
    MYMEMORY_RESPONSE_BYTE_RESERVATION,
)
from biominer.registry.translation_sources import DEFAULT_TRANSLATION_SOURCES


def test_cli_exposes_only_lean_pipeline_commands() -> None:
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices  # noqa: SLF001 - parser surface regression test.
    poll_once = parser.parse_args(
        ["dev", "flickr", "poll-once", "--max-api-calls", "3500", "--run-id", "run-1", "--worker-id", "worker-001"]
    )

    assert "run" in commands
    assert "vision" in commands
    assert "evidence" in commands
    assert "storage" in commands
    assert "workstore" in commands
    assert "bioclip" not in commands
    assert "detect" not in commands
    assert "species" not in commands
    assert "dev" in commands
    assert "cloud" not in commands
    assert "fetch-comments" not in commands
    assert "build-comment-review-queue" not in commands
    assert "review-comments-once" not in commands
    assert "apply-comment-review-decisions" not in commands
    assert "poll-once" not in commands
    assert "filter" not in commands
    assert "apply-rules" not in commands
    assert "compact-parquet" not in commands
    assert "gc-cache" not in commands
    assert "qa-rate-limit" not in commands
    assert "qa-summary" not in commands
    assert "export-bucket-views" not in commands
    assert "report-name-evidence" not in commands
    assert "build-papilio-demoleus-query-plan" not in commands
    assert "fetch" not in commands
    assert "fetch-live" not in commands
    assert "benchmark-existing-payloads" not in commands
    assert poll_once.command == "dev"
    assert poll_once.dev_command == "flickr"
    assert poll_once.flickr_command == "poll-once"
    assert poll_once.max_api_calls == 3500
    assert poll_once.run_id == "run-1"
    assert poll_once.worker_id == "worker-001"


def test_registry_public_cli_exposes_only_build_and_audit() -> None:
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices  # noqa: SLF001 - parser surface regression test.
    registry_choices = commands["registry"]._subparsers._group_actions[0].choices  # noqa: SLF001
    dev_choices = commands["dev"]._subparsers._group_actions[0].choices  # noqa: SLF001
    dev_registry_choices = dev_choices["registry"]._subparsers._group_actions[0].choices  # noqa: SLF001

    assert set(registry_choices) == {"build", "audit"}
    for internal in {"fetch-taxonomy", "compile-fixture", "compile-enriched", "enrich-sources", "seed-flickr-queries"}:
        assert internal not in registry_choices
        assert internal in dev_registry_choices


def test_registry_build_defaults_to_production_enrichment_sources() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "registry",
            "build",
            "--output-dir",
            "data/registry/v1",
            "--registry-version",
            "v1",
        ]
    )

    assert args.enrichment_sources == ",".join(DEFAULT_ENRICHMENT_SOURCES)
    assert args.translation_sources == ",".join(DEFAULT_TRANSLATION_SOURCES)
    assert args.skip_translations is False
    assert args.range_discovery_source == "gbif"
    assert args.range_seed_json is None
    assert args.language_targets_json is None
    assert args.curated_static_source_config_dir == "config/vernacular_sources"
    assert args.curated_static_source_snapshot_dir == "data/source_snapshots"
    assert args.skip_range_discovery is False
    assert args.skip_language_targets is False
    assert args.skip_curated_static_sources is False
    assert args.mymemory_monthly_request_limit == MYMEMORY_MONTHLY_REQUEST_LIMIT
    assert args.mymemory_monthly_input_word_limit == MYMEMORY_MONTHLY_INPUT_WORD_LIMIT
    assert args.mymemory_monthly_bandwidth_mb_limit == MYMEMORY_MONTHLY_BANDWIDTH_MB_LIMIT
    assert args.mymemory_response_byte_reservation == MYMEMORY_RESPONSE_BYTE_RESERVATION


def test_registry_build_source_defaults_exclude_blocked_providers() -> None:
    source_names = {source.casefold() for source in DEFAULT_ENRICHMENT_SOURCES}

    assert "ala" not in source_names
    assert "atlas_of_living_australia" not in source_names
    assert "slu" not in source_names
    assert "artdatabanken" not in source_names
    assert "swedish" not in source_names


def test_run_cli_vision_profile_populates_m5pro_defaults_and_overrides() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--taxon",
            "Papilio demoleus",
            "--registry-dir",
            "s3://biominer/biominer/registry/current",
            "--output-prefix",
            "s3://biominer/biominer/runs/papilio_demoleus",
            "--vision-profile",
            "mac_m5pro_64gb",
        ]
    )

    settings = _production_vision_settings_from_args(args)

    assert settings.profile_name == "mac_m5pro_64gb"
    assert settings.device == "mps"
    assert settings.yolo_checkpoint == "yoloe-26s-seg.pt"
    assert settings.yolo_imgsz == 768
    assert settings.detector_batch_size == 16
    assert settings.bioclip_model == "hf-hub:imageomics/bioclip-2.5-vith14"
    assert settings.crop_batch_size == 24
    assert settings.bioclip_top_k == 10
    assert settings.crop_padding_ratio == 0.08
    assert settings.parquet_compression == "zstd"
    assert settings.delete_images_after_commit is True

    overridden = parser.parse_args(
        [
            "run",
            "--taxon",
            "Papilio demoleus",
            "--registry-dir",
            "s3://biominer/biominer/registry/current",
            "--output-prefix",
            "s3://biominer/biominer/runs/papilio_demoleus",
            "--vision-profile",
            "mac_m5pro_64gb",
            "--device",
            "cpu",
            "--yolo-batch",
            "7",
            "--bioclip-model",
            "custom-bioclip",
            "--no-delete-images-after-commit",
        ]
    )

    overridden_settings = _production_vision_settings_from_args(overridden)

    assert overridden_settings.device == "cpu"
    assert overridden_settings.detector_batch_size == 7
    assert overridden_settings.bioclip_model == "custom-bioclip"
    assert overridden_settings.delete_images_after_commit is False
    assert overridden_settings.yolo_imgsz == 768
    assert overridden_settings.crop_padding_ratio == 0.08


def test_vision_ablate_command_still_allows_all_visual_modes() -> None:
    args = build_parser().parse_args(
        [
            "vision",
            "ablate",
            "--input",
            "records.parquet",
            "--detections",
            "detections.parquet",
            "--species-context",
            "species_context.json",
            "--output-dir",
            "reports/ablation",
            "--modes",
            "whole_image,detector_crop,detector_crop_segmentation",
        ]
    )

    assert args.command == "vision"
    assert args.vision_command == "ablate"
    assert args.modes == "whole_image,detector_crop,detector_crop_segmentation"


def test_registry_build_parses_regional_and_static_source_options() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "registry",
            "build",
            "--output-dir",
            "data/registry/papilio",
            "--registry-version",
            "papilio-v1",
            "--range-discovery-source",
            "gbif",
            "--range-seed-json",
            "config/range_seed/papilio_demoleus.json",
            "--language-targets-json",
            "config/language_targets/papilio_demoleus_region_language_targets.json",
            "--curated-static-source-config-dir",
            "config/vernacular_sources",
            "--curated-static-source-snapshot-dir",
            "data/source_snapshots",
            "--mymemory-monthly-request-limit",
            "321",
            "--mymemory-monthly-input-word-limit",
            "654",
            "--mymemory-monthly-bandwidth-mb-limit",
            "987",
            "--mymemory-response-byte-reservation",
            "12345",
            "--skip-range-discovery",
            "--skip-language-targets",
            "--skip-curated-static-sources",
        ]
    )

    assert args.range_discovery_source == "gbif"
    assert args.range_seed_json == "config/range_seed/papilio_demoleus.json"
    assert args.language_targets_json == "config/language_targets/papilio_demoleus_region_language_targets.json"
    assert args.curated_static_source_config_dir == "config/vernacular_sources"
    assert args.curated_static_source_snapshot_dir == "data/source_snapshots"
    assert args.skip_range_discovery is True
    assert args.skip_language_targets is True
    assert args.skip_curated_static_sources is True


def test_registry_build_cli_forwards_regional_and_static_source_options(monkeypatch, tmp_path, capsys) -> None:
    recorded: dict[str, Any] = {}

    def fake_build_registry(**kwargs: Any) -> dict[str, Any]:
        recorded.update(kwargs)
        return {"registry_version": kwargs["registry_version"], "manifest": {"qa_status": "passed"}}

    monkeypatch.setattr("biominer.cli.build_registry", fake_build_registry)
    parser = build_parser()
    args = parser.parse_args(
        [
            "registry",
            "build",
            "--output-dir",
            str(tmp_path / "registry"),
            "--registry-version",
            "regional-v1",
            "--range-discovery-source",
            "gbif",
            "--range-seed-json",
            "config/range_seed/papilio_demoleus.json",
            "--language-targets-json",
            "config/language_targets/papilio_demoleus_region_language_targets.json",
            "--curated-static-source-config-dir",
            "config/vernacular_sources",
            "--curated-static-source-snapshot-dir",
            "data/source_snapshots",
            "--mymemory-monthly-request-limit",
            "321",
            "--mymemory-monthly-input-word-limit",
            "654",
            "--mymemory-monthly-bandwidth-mb-limit",
            "987",
            "--mymemory-response-byte-reservation",
            "12345",
            "--skip-range-discovery",
            "--skip-language-targets",
            "--skip-curated-static-sources",
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["registry_version"] == "regional-v1"
    assert recorded["range_discovery_source"] == "gbif"
    assert recorded["range_seed_json"] == "config/range_seed/papilio_demoleus.json"
    assert recorded["language_targets_json"] == "config/language_targets/papilio_demoleus_region_language_targets.json"
    assert recorded["curated_static_source_config_dir"] == "config/vernacular_sources"
    assert recorded["curated_static_source_snapshot_dir"] == "data/source_snapshots"
    assert recorded["mymemory_monthly_request_limit"] == 321
    assert recorded["mymemory_monthly_input_word_limit"] == 654
    assert recorded["mymemory_monthly_bandwidth_mb_limit"] == 987
    assert recorded["mymemory_response_byte_reservation"] == 12345
    assert recorded["skip_range_discovery"] is True
    assert recorded["skip_language_targets"] is True
    assert recorded["skip_curated_static_sources"] is True


def test_registry_build_parses_translation_worker_checkpoint_controls() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "registry",
            "build",
            "--output-dir",
            "data/registry/v1",
            "--registry-version",
            "v1",
            "--translation-workers",
            "4",
            "--translation-checkpoint-every",
            "25",
            "--translation-checkpoint-seconds",
            "2.5",
            "--translation-language-shards",
            "3",
        ]
    )

    assert args.translation_workers == 4
    assert args.translation_checkpoint_every == 25
    assert args.translation_checkpoint_seconds == 2.5
    assert args.translation_language_shards == 3


def test_registry_build_parses_mymemory_monthly_budget_controls() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "registry",
            "build",
            "--output-dir",
            "data/registry/v1",
            "--registry-version",
            "v1",
            "--mymemory-monthly-request-limit",
            "10000",
            "--mymemory-monthly-input-word-limit",
            "10000",
            "--mymemory-monthly-bandwidth-mb-limit",
            "10240",
            "--mymemory-response-byte-reservation",
            "1048576",
        ]
    )

    assert args.mymemory_monthly_request_limit == 10000
    assert args.mymemory_monthly_input_word_limit == 10000
    assert args.mymemory_monthly_bandwidth_mb_limit == 10240
    assert args.mymemory_response_byte_reservation == 1048576


def test_registry_commands_parse_query_curation_json() -> None:
    parser = build_parser()

    build_args = parser.parse_args(
        [
            "registry",
            "build",
            "--output-dir",
            "data/registry/v1",
            "--registry-version",
            "v1",
            "--query-curation-json",
            "examples/species/papilio_demoleus/query_curation.json",
        ]
    )
    compile_args = parser.parse_args(
        [
            "dev",
            "registry",
            "compile-enriched",
            "--registry-dir",
            "data/registry/v1",
            "--registry-version",
            "v1",
            "--query-curation-json",
            "examples/species/papilio_demoleus/query_curation.json",
        ]
    )

    assert build_args.query_curation_json == "examples/species/papilio_demoleus/query_curation.json"
    assert compile_args.query_curation_json == "examples/species/papilio_demoleus/query_curation.json"


def test_run_cli_exposes_comment_and_registry_build_controls() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "run",
            "--taxon",
            "Danaus plexippus",
            "--registry-dir",
            "data/registry/current",
            "--output-prefix",
            "runs/danaus",
            "--storage-backend",
            "local",
            "--workstore-backend",
            "sqlite",
            "--build-registry-if-missing",
            "--comments-max-api-calls",
            "17",
        ]
    )

    assert args.build_registry_if_missing is True
    assert args.comments_max_api_calls == 17


def test_species_cli_removed_from_public_surface() -> None:
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices  # noqa: SLF001

    assert "species" not in commands
    assert "run" in commands
    for removed in {
        "resolve",
        "refresh-registry",
        "compile-flickr-queries",
        "fetch-flickr",
        "review-comments",
        "detect",
        "bioclip-funnel",
        "bioclip-objects",
        "ablate-objects",
        "join-object-evidence",
    }:
        with pytest.raises(SystemExit):
            parser.parse_args(["species", removed])


def test_cloud_cli_removed_from_public_surface() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["cloud", "doctor"])


def test_storage_and_workstore_doctor_commands_parse() -> None:
    parser = build_parser()
    storage_args = parser.parse_args(["storage", "doctor"])
    workstore_args = parser.parse_args(["workstore", "doctor"])

    assert storage_args.command == "storage"
    assert storage_args.storage_command == "doctor"
    assert workstore_args.command == "workstore"
    assert workstore_args.workstore_command == "doctor"


def test_storage_doctor_exercises_storage_without_workstore(capsys, monkeypatch) -> None:
    fake_storage = _FakeCloudStorage()
    config = _fake_cloud_config()
    calls: dict[str, int] = {"workstore": 0}
    monkeypatch.setattr("biominer.cli.load_biominer_config", lambda path: config)
    monkeypatch.setattr("biominer.cli.create_storage_backend", lambda storage_config: fake_storage)

    def fail_create_workstore(_workstore_config):  # noqa: ANN001, ANN202
        calls["workstore"] += 1
        raise AssertionError("storage doctor must not open the workstore")

    monkeypatch.setattr("biominer.cli.create_workstore", fail_create_workstore)

    rc = run(build_parser().parse_args(["storage", "doctor"]))
    rendered = capsys.readouterr().out
    output = json.loads(rendered)

    assert rc == 0
    assert calls["workstore"] == 0
    assert output["command"] == "storage doctor"
    assert output["status"] == "ok"
    assert output["storage"]["json_roundtrip"] is True
    assert output["storage"]["json_deleted"] is True
    assert output["storage"]["parquet_rows"] == 2
    assert "password" not in rendered
    assert "secret-value" not in rendered


def test_workstore_doctor_exercises_workstore_without_storage(capsys, monkeypatch) -> None:
    fake_store = _FakeCloudWorkStore()
    config = _fake_cloud_config()
    calls: dict[str, int] = {"storage": 0}
    monkeypatch.setattr("biominer.cli.load_biominer_config", lambda path: config)
    monkeypatch.setattr("biominer.cli.create_workstore", lambda workstore_config: fake_store)

    def fail_create_storage(_storage_config):  # noqa: ANN001, ANN202
        calls["storage"] += 1
        raise AssertionError("workstore doctor must not open storage")

    monkeypatch.setattr("biominer.cli.create_storage_backend", fail_create_storage)

    rc = run(build_parser().parse_args(["workstore", "doctor"]))
    rendered = capsys.readouterr().out
    output = json.loads(rendered)

    assert rc == 0
    assert calls["storage"] == 0
    assert output["command"] == "workstore doctor"
    assert output["status"] == "ok"
    assert output["workstore"]["schema_initialized"] is True
    assert output["workstore"]["claimed_work_key"].startswith("workstore-doctor-work:")
    assert output["workstore"]["registered_shards"] == 1
    assert "password" not in rendered
    assert "secret-value" not in rendered


def test_poll_once_cloud_no_compact_passes_workstore(monkeypatch, capsys) -> None:
    fake_store = _FakeCloudWorkStore()
    config = _fake_cloud_config()
    captured: dict[str, Any] = {}
    monkeypatch.setattr("biominer.cli.load_biominer_config", lambda path: config)
    monkeypatch.setattr("biominer.cli.create_workstore", lambda workstore_config: fake_store)

    def fake_poll_once(**kwargs) -> SimpleNamespace:  # noqa: ANN003
        captured.update(kwargs)
        return SimpleNamespace(state_db=Path("state.sqlite"), evidence_rows_written=0)

    monkeypatch.setattr("biominer.cli.poll_once", fake_poll_once)

    rc = run(
        build_parser().parse_args(
            [
                "dev",
                "flickr",
                "poll-once",
                "--storage-backend",
                "s3",
                "--storage-prefix",
                "s3://biominer/biominer",
                "--no-compact",
            ]
        )
    )
    output = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert output["evidence_rows_written"] == 0
    assert fake_store.schema_initialized
    assert captured["work_store"] is fake_store


def test_detect_boxes_cli_accepts_object_detection_arguments() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "vision",
            "detect",
            "--input",
            "filtered.parquet",
            "--output",
            "object_detections.parquet",
            "--runtime-python",
            ".venv-vision-py312/bin/python",
            "--device",
            "mps",
            "--download-workers",
            "2",
            "--max-inflight-images",
            "7",
            "--detector-batch-size",
            "3",
            "--parquet-batch-rows",
            "5",
            "--crop-target-px",
            "224",
            "--image-max-side-px",
            "960",
            "--crop-padding-ratio",
            "0.2",
        ]
    )
    vision_args = parser.parse_args(
        [
            "vision",
            "detect",
            "--input",
            "filtered.parquet",
            "--output",
            "object_detections.parquet",
            "--profile",
            "mac_m5pro_64gb",
            "--parquet-batch-rows",
            "6",
            "--image-max-side-px",
            "1024",
        ]
    )

    assert args.command == "vision"
    assert args.vision_command == "detect"
    assert args.input == "filtered.parquet"
    assert args.output == "object_detections.parquet"
    assert args.backend == "yoloe26"
    assert args.runtime_python == ".venv-vision-py312/bin/python"
    assert args.device == "mps"
    assert args.download_workers == 2
    assert args.max_inflight_images == 7
    assert args.detector_batch_size == 3
    assert args.parquet_batch_rows == 5
    assert args.crop_target_px == 224
    assert args.image_max_side_px == 960
    assert args.crop_padding_ratio == 0.2
    assert vision_args.command == "vision"
    assert vision_args.vision_command == "detect"
    assert vision_args.backend == "yoloe26"
    assert vision_args.profile == "mac_m5pro_64gb"
    assert vision_args.parquet_batch_rows == 6
    assert vision_args.image_max_side_px == 1024


def test_detect_boxes_cli_accepts_yoloe26_arguments() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "vision",
            "detect",
            "--input",
            "filtered.parquet",
            "--output",
            "object_detections.parquet",
            "--backend",
            "yoloe26",
            "--runtime-python",
            "/runtime-base/YOLO26/venv/bin/python",
            "--checkpoint",
            "yoloe-26m-seg.pt",
            "--device",
            "mps",
            "--imgsz",
            "768",
            "--conf",
            "0.15",
            "--iou",
            "0.55",
            "--max-det",
            "12",
            "--prompt-class",
            "butterfly",
            "--prompt-class",
            "museum label",
            "--no-include-hard-negative-prompts",
        ]
    )

    assert args.backend == "yoloe26"
    assert args.runtime_python == "/runtime-base/YOLO26/venv/bin/python"
    assert args.checkpoint == "yoloe-26m-seg.pt"
    assert args.device == "mps"
    assert args.imgsz == 768
    assert args.conf == 0.15
    assert args.iou == 0.55
    assert args.max_det == 12
    assert args.prompt_class == ["butterfly", "museum label"]
    assert args.include_hard_negative_prompts is False


def test_detect_boxes_cli_accepts_explicit_yolo26_checkpoint() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "vision",
            "detect",
            "--input",
            "filtered.parquet",
            "--output",
            "object_detections.parquet",
            "--backend",
            "yolo26",
            "--runtime-python",
            "/runtime-base/YOLO26/venv/bin/python",
            "--checkpoint",
            "coarse-objects.pt",
            "--device",
            "mps",
            "--imgsz",
            "768",
            "--conf",
            "0.15",
            "--iou",
            "0.55",
            "--max-det",
            "12",
        ]
    )

    assert args.backend == "yolo26"
    assert args.runtime_python == "/runtime-base/YOLO26/venv/bin/python"
    assert args.checkpoint == "coarse-objects.pt"
    assert args.device == "mps"
    assert args.imgsz == 768
    assert args.conf == 0.15
    assert args.iou == 0.55
    assert args.max_det == 12


def test_detect_boxes_cli_forwards_detection_and_run_policies(tmp_path, capsys, monkeypatch) -> None:
    input_path = tmp_path / "filtered.parquet"
    output_path = tmp_path / "object_detections.parquet"
    pl.DataFrame([{"source": "flickr", "flickr_photo_id": "photo-1", "image_url": "memory://photo-1"}]).write_parquet(input_path)
    calls: dict[str, object] = {}

    def fake_backend(args, records):  # noqa: ANN001, ANN202 - mirrors _detect_boxes_backend.
        calls["backend_args"] = args
        calls["records"] = records
        return SimpleNamespace(backend="fake"), lambda record: None

    def fake_pipeline(**kwargs):  # noqa: ANN003, ANN202 - mirrors run_detection_pipeline.
        calls["pipeline"] = kwargs
        return SimpleNamespace(
            frame=pl.DataFrame([{"detection_status": "no_detection"}]),
            output_path=Path(kwargs["output_path"]),
            records_seen=1,
            images_loaded=1,
            detections_written=0,
            crops_created=0,
            parquet_batches_written=2,
        )

    monkeypatch.setattr("biominer.cli._detect_boxes_backend", fake_backend)
    monkeypatch.setattr("biominer.cli.run_detection_pipeline", fake_pipeline)
    parser = build_parser()
    args = parser.parse_args(
        [
            "vision",
            "detect",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--backend",
            "fake",
            "--download-workers",
            "2",
            "--max-inflight-images",
            "7",
            "--detector-batch-size",
            "3",
            "--parquet-batch-rows",
            "5",
            "--crop-target-px",
            "224",
            "--image-max-side-px",
            "960",
            "--crop-padding-ratio",
            "0.2",
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    pipeline = calls["pipeline"]
    assert payload["parquet_batches_written"] == 2
    assert calls["records"][0]["flickr_photo_id"] == "photo-1"
    assert pipeline["detection_policy"].backend == "fake"
    assert pipeline["detection_policy"].crop_target_px == 224
    assert pipeline["detection_policy"].image_max_side_px == 960
    assert pipeline["detection_policy"].crop_padding_ratio == 0.2
    assert pipeline["run_policy"].download_workers == 2
    assert pipeline["run_policy"].max_inflight_images == 7
    assert pipeline["run_policy"].detector_batch_size == 3
    assert pipeline["run_policy"].parquet_batch_rows == 5


def test_detect_boxes_cli_applies_runtime_profile_with_explicit_overrides(tmp_path, capsys, monkeypatch) -> None:
    input_path = tmp_path / "filtered.parquet"
    output_path = tmp_path / "object_detections.parquet"
    pl.DataFrame([{"source": "flickr", "flickr_photo_id": "photo-1", "image_url": "memory://photo-1"}]).write_parquet(input_path)
    calls: dict[str, object] = {}

    def fake_backend(args, records):  # noqa: ANN001, ANN202 - mirrors _detect_boxes_backend.
        calls["backend_args"] = args
        calls["records"] = records
        return SimpleNamespace(backend="fake"), lambda record: None

    def fake_pipeline(**kwargs):  # noqa: ANN003, ANN202 - mirrors run_detection_pipeline.
        calls["pipeline"] = kwargs
        return SimpleNamespace(
            frame=pl.DataFrame([{"detection_status": "detected"}]),
            output_path=Path(kwargs["output_path"]),
            records_seen=1,
            images_loaded=1,
            detections_written=1,
            crops_created=1,
            parquet_batches_written=1,
        )

    monkeypatch.setattr("biominer.cli._detect_boxes_backend", fake_backend)
    monkeypatch.setattr("biominer.cli.run_detection_pipeline", fake_pipeline)
    parser = build_parser()
    args = parser.parse_args(
        [
            "vision",
            "detect",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--backend",
            "fake",
            "--profile",
            "mac_m5pro_64gb",
            "--crop-target-px",
            "448",
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    pipeline = calls["pipeline"]
    assert payload["profile"] == "mac_m5pro_64gb"
    assert pipeline["detection_policy"].image_max_side_px == 1280
    assert pipeline["detection_policy"].crop_target_px == 448
    assert pipeline["detection_policy"].retain_debug_crops is False
    assert pipeline["run_policy"].download_workers == 4
    assert pipeline["run_policy"].decode_workers == 4
    assert pipeline["run_policy"].detector_workers == 1
    assert pipeline["run_policy"].max_inflight_images == 32
    assert pipeline["run_policy"].max_inflight_crops == 96
    assert pipeline["run_policy"].detector_batch_size == 16
    assert pipeline["run_policy"].crop_batch_size == 24


def test_detect_boxes_cli_uses_runtime_profile_defaults_when_not_overridden(tmp_path, capsys, monkeypatch) -> None:
    input_path = tmp_path / "filtered.parquet"
    output_path = tmp_path / "object_detections.parquet"
    pl.DataFrame([{"source": "flickr", "flickr_photo_id": "photo-1", "image_url": "memory://photo-1"}]).write_parquet(input_path)
    calls: dict[str, object] = {}

    def fake_backend(args, records):  # noqa: ANN001, ANN202 - mirrors _detect_boxes_backend.
        return SimpleNamespace(backend="fake"), lambda record: None

    def fake_pipeline(**kwargs):  # noqa: ANN003, ANN202 - mirrors run_detection_pipeline.
        calls["pipeline"] = kwargs
        return SimpleNamespace(
            frame=pl.DataFrame([{"detection_status": "detected"}]),
            output_path=Path(kwargs["output_path"]),
            records_seen=1,
            images_loaded=1,
            detections_written=1,
            crops_created=1,
            parquet_batches_written=1,
        )

    custom_profile = SimpleNamespace(
        detection_policy=DetectionPolicy(image_max_side_px=640, crop_target_px=224, max_boxes_per_image=3),
        run_policy=DetectionRunPolicy(download_workers=2, max_inflight_images=5, detector_batch_size=6),
    )

    monkeypatch.setattr("biominer.cli._detect_boxes_backend", fake_backend)
    monkeypatch.setattr("biominer.cli.run_detection_pipeline", fake_pipeline)
    monkeypatch.setattr("biominer.cli.runtime_profile", lambda name: custom_profile)
    parser = build_parser()
    args = parser.parse_args(
        [
            "vision",
            "detect",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--backend",
            "fake",
            "--profile",
            "mac_m5pro_64gb",
        ]
    )

    assert run(args) == 0

    capsys.readouterr()
    pipeline = calls["pipeline"]
    assert pipeline["detection_policy"].image_max_side_px == 640
    assert pipeline["detection_policy"].crop_target_px == 224
    assert pipeline["detection_policy"].max_boxes_per_image == 3
    assert pipeline["run_policy"].download_workers == 2
    assert pipeline["run_policy"].max_inflight_images == 5
    assert pipeline["run_policy"].detector_batch_size == 6


def test_detect_eval_cli_forwards_xie_thresholds(tmp_path, capsys, monkeypatch) -> None:
    predictions = tmp_path / "predictions.parquet"
    truth = tmp_path / "truth.parquet"
    output = tmp_path / "eval_report.json"
    pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                "species_top1_scientific_name": "Danaus plexippus",
                "species_top1_score": 0.71,
            }
        ]
    ).write_parquet(predictions)
    pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                "scientific_name": "Danaus plexippus",
            }
        ]
    ).write_parquet(truth)
    calls: dict[str, object] = {}

    def fake_evaluate(**kwargs):  # noqa: ANN003, ANN202 - mirrors evaluate_xie_style.
        calls["evaluate"] = kwargs
        return {
            "ground_truth_available": True,
            "detector_ap50": 1.0,
            "joint_map50": 1.0,
        }

    monkeypatch.setattr("biominer.cli.evaluate_xie_style", fake_evaluate)
    parser = build_parser()
    args = parser.parse_args(
        [
            "dev",
            "vision",
            "eval",
            "--predictions",
            str(predictions),
            "--ground-truth",
            str(truth),
            "--output",
            str(output),
            "--iou-threshold",
            "0.6",
            "--score-threshold",
            "0.45",
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert payload["iou_threshold"] == 0.6
    assert payload["score_threshold"] == 0.45
    assert report["iou_threshold"] == 0.6
    assert report["score_threshold"] == 0.45
    assert calls["evaluate"]["iou_threshold"] == 0.6
    assert calls["evaluate"]["score_threshold"] == 0.45


def test_detect_crop_preview_writes_html_artifact_without_image_archive(tmp_path, capsys) -> None:
    detections = tmp_path / "object_detections.parquet"
    output = tmp_path / "crop_preview.html"
    pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "image_url": "https://live.staticflickr.com/photo-1.jpg",
                "detection_id": "det-1",
                "crop_hash": "sha256:crop-1",
                "bbox_xyxyn": [0.1, 0.2, 0.6, 0.8],
                "bbox_xyxy": [10.0, 20.0, 60.0, 80.0],
                "detector_label": "butterfly_like",
                "detector_score": 0.91,
                "detection_status": "detected",
            },
            {
                "source": "flickr",
                "flickr_photo_id": "photo-2",
                "image_url": "https://live.staticflickr.com/photo-2.jpg",
                "detection_id": "det-2",
                "crop_hash": None,
                "bbox_xyxyn": None,
                "bbox_xyxy": None,
                "detector_label": None,
                "detector_score": None,
                "detection_status": "no_detection",
            },
        ]
    ).write_parquet(detections)
    parser = build_parser()
    args = parser.parse_args(["dev", "vision", "crop-preview", "--detections", str(detections), "--output", str(output)])

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    html = output.read_text(encoding="utf-8")
    assert payload["preview_rows"] == 1
    assert payload["skipped_rows"] == 1
    assert payload["storage_policy"] == "remote_image_references_only"
    assert "photo-1" in html
    assert "det-1" in html
    assert "sha256:crop-1" in html
    assert "https://live.staticflickr.com/photo-1.jpg" in html
    assert "left: 10.0000%" in html
    assert "width: 50.0000%" in html
    assert not (tmp_path / "crop_preview_files").exists()


def test_bioclip_object_cli_accepts_screen_and_ablation_arguments() -> None:
    parser = build_parser()
    integrated_screen = parser.parse_args(
        [
            "vision",
            "screen",
            "--input",
            "canonical.parquet",
            "--output-dir",
            "vision_screen",
            "--species-context",
            "species_context.json",
            "--species-candidates",
            "species_candidates.parquet",
            "--vision-profile",
            "mac_m5pro_64gb",
            "--device",
            "mps",
            "--yolo-runtime-python",
            ".venv-yolo/bin/python",
            "--bioclip-runtime-python",
            ".venv-bioclip/bin/python",
            "--chunk-rows",
            "2",
            "--parquet-part-rows",
            "3",
            "--no-delete-images-after-commit",
        ]
    )
    screen = parser.parse_args(
        [
            "vision",
            "score",
            "--input",
            "filtered.parquet",
            "--detections",
            "object_detections.parquet",
            "--species-context",
            "species_context.json",
            "--geo-prior-table",
            "geo_prior.parquet",
            "--output",
            "object_bioclip_scores.parquet",
            "--ablation-mode",
            "detector_crop",
            "--parquet-batch-rows",
            "7",
            "--bioclip-batch",
            "17",
            "--text-embedding-batch-size",
            "2",
            "--candidate-text-embedding-cache",
            "candidate_text_embeddings.parquet",
            "--object-image-embedding-cache",
            "object_image_embeddings.parquet",
            "--segmenter",
            "none",
        ]
    )
    ablate = parser.parse_args(
        [
            "vision",
            "ablate",
            "--input",
            "filtered.parquet",
            "--detections",
            "object_detections.parquet",
            "--species-context",
            "species_context.json",
            "--geo-prior-table",
            "geo_prior.parquet",
            "--output-dir",
            "ablations",
            "--modes",
            "whole_image,detector_crop,detector_crop_segmentation",
            "--parquet-batch-rows",
            "11",
            "--bioclip-batch",
            "19",
            "--text-embedding-batch-size",
            "3",
            "--candidate-text-embedding-cache",
            "candidate_text_embeddings.parquet",
            "--object-image-embedding-cache",
            "object_image_embeddings.parquet",
            "--segmenter",
            "none",
        ]
    )
    vision_score = parser.parse_args(
        [
            "vision",
            "score",
            "--species-context",
            "species_context.json",
            "--input",
            "filtered.parquet",
            "--detections",
            "object_detections.parquet",
            "--geo-prior-table",
            "geo_prior.parquet",
            "--output",
            "object_bioclip_scores.parquet",
            "--parquet-batch-rows",
            "9",
            "--bioclip-batch",
            "23",
            "--text-embedding-batch-size",
            "4",
            "--candidate-text-embedding-cache",
            "candidate_text_embeddings.parquet",
            "--object-image-embedding-cache",
            "object_image_embeddings.parquet",
            "--segmenter",
            "none",
        ]
    )
    vision_ablate = parser.parse_args(
        [
            "vision",
            "ablate",
            "--species-context",
            "species_context.json",
            "--input",
            "filtered.parquet",
            "--detections",
            "object_detections.parquet",
            "--geo-prior-table",
            "geo_prior.parquet",
            "--output-dir",
            "ablations",
            "--parquet-batch-rows",
            "13",
            "--bioclip-batch",
            "29",
            "--text-embedding-batch-size",
            "5",
            "--candidate-text-embedding-cache",
            "candidate_text_embeddings.parquet",
            "--object-image-embedding-cache",
            "object_image_embeddings.parquet",
            "--segmenter",
            "none",
        ]
    )
    evidence_join = parser.parse_args(
        [
            "evidence",
            "join",
            "--species-context",
            "species_context.json",
            "--input",
            "filtered.parquet",
            "--detections",
            "object_detections.parquet",
            "--scores",
            "object_bioclip_scores.parquet",
            "--joined-output",
            "object_evidence_joined.parquet",
            "--photo-summary-output",
            "photo_evidence_summary.parquet",
        ]
    )

    assert integrated_screen.command == "vision"
    assert integrated_screen.vision_command == "screen"
    assert integrated_screen.input == "canonical.parquet"
    assert integrated_screen.output_dir == "vision_screen"
    assert integrated_screen.species_context == "species_context.json"
    assert integrated_screen.species_candidates == "species_candidates.parquet"
    assert integrated_screen.vision_profile == "mac_m5pro_64gb"
    assert integrated_screen.device == "mps"
    assert integrated_screen.yolo_runtime_python == ".venv-yolo/bin/python"
    assert integrated_screen.bioclip_runtime_python == ".venv-bioclip/bin/python"
    assert integrated_screen.chunk_rows == 2
    assert integrated_screen.parquet_part_rows == 3
    assert integrated_screen.delete_images_after_commit is False
    assert screen.command == "vision"
    assert screen.vision_command == "score"
    assert screen.ablation_mode == "detector_crop"
    assert screen.geo_prior_table == "geo_prior.parquet"
    assert screen.parquet_batch_rows == 7
    assert screen.bioclip_batch == 17
    assert screen.text_embedding_batch_size == 2
    assert screen.candidate_text_embedding_cache == "candidate_text_embeddings.parquet"
    assert screen.object_image_embedding_cache == "object_image_embeddings.parquet"
    assert screen.segmenter == "none"
    assert ablate.command == "vision"
    assert ablate.vision_command == "ablate"
    assert ablate.modes == "whole_image,detector_crop,detector_crop_segmentation"
    assert ablate.geo_prior_table == "geo_prior.parquet"
    assert ablate.parquet_batch_rows == 11
    assert ablate.bioclip_batch == 19
    assert ablate.text_embedding_batch_size == 3
    assert ablate.candidate_text_embedding_cache == "candidate_text_embeddings.parquet"
    assert ablate.object_image_embedding_cache == "object_image_embeddings.parquet"
    assert ablate.segmenter == "none"
    assert vision_score.command == "vision"
    assert vision_score.vision_command == "score"
    assert vision_score.geo_prior_table == "geo_prior.parquet"
    assert vision_score.parquet_batch_rows == 9
    assert vision_score.bioclip_batch == 23
    assert vision_score.text_embedding_batch_size == 4
    assert vision_score.candidate_text_embedding_cache == "candidate_text_embeddings.parquet"
    assert vision_score.object_image_embedding_cache == "object_image_embeddings.parquet"
    assert vision_score.segmenter == "none"
    assert vision_ablate.command == "vision"
    assert vision_ablate.vision_command == "ablate"
    assert vision_ablate.geo_prior_table == "geo_prior.parquet"
    assert vision_ablate.parquet_batch_rows == 13
    assert vision_ablate.bioclip_batch == 29
    assert vision_ablate.text_embedding_batch_size == 5
    assert vision_ablate.candidate_text_embedding_cache == "candidate_text_embeddings.parquet"
    assert vision_ablate.object_image_embedding_cache == "object_image_embeddings.parquet"
    assert vision_ablate.segmenter == "none"
    assert evidence_join.command == "evidence"
    assert evidence_join.evidence_command == "join"
    assert evidence_join.species_context == "species_context.json"
    with pytest.raises(SystemExit):
        parser.parse_args(["bioclip", "join-object-evidence"])
    with pytest.raises(SystemExit):
        parser.parse_args(["vision", "join"])
    with pytest.raises(SystemExit):
        parser.parse_args(["bioclip", "screen-objects"])
    with pytest.raises(SystemExit):
        parser.parse_args(["bioclip", "ablate-objects"])
    for removed in ("detect", "bioclip-objects", "ablate-objects", "join-object-evidence"):
        with pytest.raises(SystemExit):
            parser.parse_args(["species", removed])


def test_vision_screen_runs_integrated_detector_bioclip_parts(tmp_path, capsys, monkeypatch) -> None:
    yolo_python = tmp_path / "yolo" / "bin" / "python"
    bioclip_python = tmp_path / "bioclip" / "bin" / "python"
    yolo_python.parent.mkdir(parents=True)
    bioclip_python.parent.mkdir(parents=True)
    yolo_python.write_text("# fake yolo python", encoding="utf-8")
    bioclip_python.write_text("# fake bioclip python", encoding="utf-8")
    input_path = tmp_path / "canonical.parquet"
    pl.DataFrame(
        [
            {"source": "flickr", "flickr_photo_id": "photo-1", "image_url": "memory://photo-1"},
            {"source": "flickr", "flickr_photo_id": "photo-2", "image_url": "memory://photo-2"},
            {"source": "flickr", "flickr_photo_id": "photo-3", "image_url": "memory://photo-3"},
        ]
    ).write_parquet(input_path)
    context_path = tmp_path / "species_context.json"
    context_path.write_text(
        json.dumps(
            {
                "scientific_name": "Danaus plexippus",
                "accepted_taxon_key": "gbif:5131654",
                "canonical_name": "Danaus plexippus",
                "family": "Nymphalidae",
                "genus": "Danaus",
                "family_key": "gbif:7017",
                "genus_key": "gbif:1927164",
                "species_key": "gbif:5131654",
                "registry_version": "registry-v1",
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "vision_screen"
    calls: dict[str, object] = {"detect_chunks": [], "screen_chunks": []}

    class FakeYoloDetector:
        backend = "yoloe26"
        model_id = "yoloe26"
        model_version = "test"
        checkpoint = "yoloe-26s-seg.pt"

        def __init__(self, **kwargs):  # noqa: ANN003 - mirrors sidecar detector init.
            calls["detector_init"] = kwargs

        def close(self) -> None:
            calls["detector_closed"] = True

    class FakePersistentScorer:
        def __init__(self, **kwargs):  # noqa: ANN003 - mirrors persistent scorer init.
            calls["persistent_init"] = kwargs

        def close(self) -> None:
            calls["persistent_closed"] = True

    class FakeCropScorer:
        model_id = "bioclip2_5"
        model_version = "bioclip2_5_huge"
        model_checkpoint = "fake-checkpoint"

        def __init__(self, **kwargs):  # noqa: ANN003 - mirrors crop scorer init.
            calls["crop_scorer_init"] = kwargs

    def fake_build_candidate_set(context, **kwargs):  # noqa: ANN001, ANN003, ANN202 - mirrors build_candidate_set.
        calls["candidate_set"] = kwargs
        return SimpleNamespace(candidate_set_id="candidate-set-1")

    def fake_run_detection_pipeline(**kwargs):  # noqa: ANN003, ANN202 - mirrors run_detection_pipeline.
        records = list(kwargs["records"])
        calls["detect_chunks"].append(len(records))  # type: ignore[index,union-attr]
        calls["detection_policy"] = kwargs["detection_policy"]
        calls["run_policy"] = kwargs["run_policy"]
        frame = pl.DataFrame(
            [
                {
                    "source": record["source"],
                    "flickr_photo_id": record["flickr_photo_id"],
                    "detection_id": f"det-{record['flickr_photo_id']}",
                    "crop_hash": f"sha256:{record['flickr_photo_id']}",
                    "detector_label": "butterfly_like",
                    "detection_status": "detected",
                }
                for record in records
            ]
        )
        frame.write_parquet(kwargs["output_path"])
        return SimpleNamespace(
            frame=frame,
            output_path=Path(kwargs["output_path"]),
            records_seen=len(records),
            images_loaded=len(records),
            image_failures=0,
            detections_written=len(records),
            crops_created=len(records),
            parquet_batches_written=1,
        )

    def fake_screen_object_detections(**kwargs):  # noqa: ANN003, ANN202 - mirrors screen_object_detections.
        calls["screen_chunks"].append(kwargs["canonical_records"].height)  # type: ignore[index,union-attr]
        calls["screen_kwargs"] = kwargs
        frame = pl.DataFrame(
            [
                {
                    "source": row["source"],
                    "flickr_photo_id": row["flickr_photo_id"],
                    "detection_id": row["detection_id"],
                    "crop_hash": row["crop_hash"],
                    "target_species_score": 0.8,
                    "occurrence_bin": "gold",
                    "species_top1_scientific_name": "Danaus plexippus",
                }
                for row in kwargs["detections"].to_dicts()
            ]
        )
        frame.write_parquet(kwargs["output_path"])
        return SimpleNamespace(
            frame=frame,
            output_path=Path(kwargs["output_path"]),
            records_seen=kwargs["canonical_records"].height,
            detections_seen=kwargs["detections"].height,
            crops_scored=frame.height,
            score_batches_written=1,
            segmentation_unavailable_count=0,
            segmentation_unavailable_reason=None,
            visual_classifier="bioclip_object",
            visual_mode="detector_crop",
            visual_mode_status="available",
        )

    def fake_write_object_evidence_outputs(**kwargs):  # noqa: ANN003, ANN202 - mirrors evidence writer.
        pl.DataFrame([{"source": "flickr"}]).write_parquet(kwargs["joined_output_path"])
        pl.DataFrame([{"source": "flickr"}]).write_parquet(kwargs["photo_summary_output_path"])
        return SimpleNamespace(
            object_evidence_joined=Path(kwargs["joined_output_path"]),
            photo_evidence_summary=Path(kwargs["photo_summary_output_path"]),
        )

    monkeypatch.setattr("biominer.detection.yoloe26_detector.YoloE26SidecarObjectDetector", FakeYoloDetector)
    monkeypatch.setattr("biominer.cli.PersistentBioClipScorer", FakePersistentScorer)
    monkeypatch.setattr("biominer.cli.EphemeralCropBioClipScorer", FakeCropScorer)
    monkeypatch.setattr("biominer.cli.build_candidate_set", fake_build_candidate_set)
    monkeypatch.setattr("biominer.cli.run_detection_pipeline", fake_run_detection_pipeline)
    monkeypatch.setattr("biominer.cli.screen_object_detections", fake_screen_object_detections)
    monkeypatch.setattr("biominer.cli.write_object_evidence_outputs", fake_write_object_evidence_outputs)

    parser = build_parser()
    args = parser.parse_args(
        [
            "vision",
            "screen",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--species-context",
            str(context_path),
            "--yolo-runtime-python",
            str(yolo_python),
            "--bioclip-runtime-python",
            str(bioclip_python),
            "--device",
            "mps",
            "--chunk-rows",
            "2",
            "--parquet-part-rows",
            "2",
            "--no-delete-images-after-commit",
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    manifest = json.loads((output_dir / "vision_screen_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "vision_screen_metrics.json").read_text(encoding="utf-8"))
    assert payload["records_seen"] == 3
    assert payload["detections_written"] == 3
    assert payload["crops_scored"] == 3
    assert payload["image_cleanup_status"] == "disabled"
    assert calls["detect_chunks"] == [2, 1]
    assert calls["screen_chunks"] == [2, 1]
    assert calls["detector_init"]["device"] == "mps"
    assert calls["detector_init"]["imgsz"] == 768
    assert calls["persistent_init"]["device"] == "mps"
    assert calls["crop_scorer_init"]["crop_padding_ratio"] == 0.08
    assert calls["screen_kwargs"]["bioclip_batch_size"] == 24
    assert calls["detection_policy"].crop_padding_ratio == 0.08
    assert calls["run_policy"].detector_batch_size == 16
    assert calls["persistent_closed"] is True
    assert calls["detector_closed"] is True
    assert manifest["status"] == "complete"
    assert manifest["image_cleanup_status"] == "disabled"
    assert len(manifest["part_outputs"]) == 2
    assert metrics["records_seen"] == 3
    assert metrics["detection_parts"] == 2
    assert metrics["score_parts"] == 2
    assert (output_dir / "object_detections" / "part-000000.parquet").exists()
    assert (output_dir / "object_bioclip_scores" / "part-000001.parquet").exists()


def test_vision_screen_deletes_cached_images_after_relevant_parts_commit(tmp_path, capsys, monkeypatch) -> None:
    yolo_python, bioclip_python = _fake_runtime_pythons(tmp_path)
    input_path = tmp_path / "canonical.parquet"
    pl.DataFrame(
        [
            {"source": "flickr", "flickr_photo_id": "photo-no-detection", "image_url": "memory://photo-no-detection"},
            {"source": "flickr", "flickr_photo_id": "photo-butterfly", "image_url": "memory://photo-butterfly"},
        ]
    ).write_parquet(input_path)
    context_path = _write_screen_context(tmp_path)
    no_detection_image = tmp_path / "cache" / "no_detection.jpg"
    butterfly_image = tmp_path / "cache" / "butterfly.jpg"
    no_detection_image.parent.mkdir(parents=True)
    no_detection_image.write_bytes(b"no-detection")
    butterfly_image.write_bytes(b"butterfly")

    class FakeYoloDetector:
        backend = "yoloe26"
        model_id = "yoloe26"
        model_version = "test"
        checkpoint = "yoloe-26s-seg.pt"

        def __init__(self, **_kwargs):  # noqa: ANN003 - mirrors sidecar detector init.
            return None

        def close(self) -> None:
            return None

    class FakePersistentScorer:
        def __init__(self, **_kwargs):  # noqa: ANN003 - mirrors persistent scorer init.
            return None

        def close(self) -> None:
            return None

    class FakeCropScorer:
        model_id = "bioclip2_5"
        model_version = "bioclip2_5_huge"
        model_checkpoint = "fake-checkpoint"

        def __init__(self, **_kwargs):  # noqa: ANN003 - mirrors crop scorer init.
            return None

    def fake_load_decoded_image(record, *, cache_root):  # noqa: ANN001, ANN202 - mirrors image loader.
        path = no_detection_image if record["flickr_photo_id"] == "photo-no-detection" else butterfly_image
        return DecodedImage(width=1, height=1, mode="RGB", data=b"\x00\x00\x00", source_uri=str(path))

    def fake_run_detection_pipeline(**kwargs):  # noqa: ANN003, ANN202 - mirrors run_detection_pipeline.
        for record in kwargs["records"]:
            kwargs["image_loader"](record)
        frame = pl.DataFrame(
            [
                {
                    "source": "flickr",
                    "flickr_photo_id": "photo-no-detection",
                    "detection_id": "no-det",
                    "crop_hash": None,
                    "detector_label": "no_detection",
                    "detection_status": "no_detection",
                },
                {
                    "source": "flickr",
                    "flickr_photo_id": "photo-butterfly",
                    "detection_id": "det-butterfly",
                    "crop_hash": "sha256:butterfly",
                    "detector_label": "butterfly_like",
                    "detection_status": "detected",
                },
            ]
        )
        frame.write_parquet(kwargs["output_path"])
        return SimpleNamespace(
            frame=frame,
            output_path=Path(kwargs["output_path"]),
            records_seen=2,
            images_loaded=2,
            image_failures=0,
            detections_written=1,
            crops_created=1,
            parquet_batches_written=1,
        )

    def fake_screen_object_detections(**kwargs):  # noqa: ANN003, ANN202 - mirrors screen_object_detections.
        assert not no_detection_image.exists()
        assert butterfly_image.exists()
        frame = pl.DataFrame(
            [
                {
                    "source": "flickr",
                    "flickr_photo_id": "photo-butterfly",
                    "detection_id": "det-butterfly",
                    "crop_hash": "sha256:butterfly",
                    "target_species_score": 0.8,
                    "occurrence_bin": "gold",
                    "species_top1_scientific_name": "Danaus plexippus",
                }
            ]
        )
        frame.write_parquet(kwargs["output_path"])
        return SimpleNamespace(
            frame=frame,
            output_path=Path(kwargs["output_path"]),
            records_seen=2,
            detections_seen=2,
            crops_scored=1,
            score_batches_written=1,
            segmentation_unavailable_count=0,
            segmentation_unavailable_reason=None,
            visual_classifier="bioclip_object",
            visual_mode="detector_crop",
            visual_mode_status="available",
        )

    def fake_write_object_evidence_outputs(**kwargs):  # noqa: ANN003, ANN202 - mirrors evidence writer.
        pl.DataFrame([{"source": "flickr"}]).write_parquet(kwargs["joined_output_path"])
        pl.DataFrame([{"source": "flickr"}]).write_parquet(kwargs["photo_summary_output_path"])
        return SimpleNamespace(
            object_evidence_joined=Path(kwargs["joined_output_path"]),
            photo_evidence_summary=Path(kwargs["photo_summary_output_path"]),
        )

    monkeypatch.setattr("biominer.detection.yoloe26_detector.YoloE26SidecarObjectDetector", FakeYoloDetector)
    monkeypatch.setattr("biominer.cli.PersistentBioClipScorer", FakePersistentScorer)
    monkeypatch.setattr("biominer.cli.EphemeralCropBioClipScorer", FakeCropScorer)
    monkeypatch.setattr("biominer.cli.build_candidate_set", lambda context, **kwargs: SimpleNamespace(candidate_set_id="candidate-set-1"))
    monkeypatch.setattr("biominer.cli.load_decoded_image_from_record", fake_load_decoded_image)
    monkeypatch.setattr("biominer.cli.run_detection_pipeline", fake_run_detection_pipeline)
    monkeypatch.setattr("biominer.cli.screen_object_detections", fake_screen_object_detections)
    monkeypatch.setattr("biominer.cli.write_object_evidence_outputs", fake_write_object_evidence_outputs)

    parser = build_parser()
    args = parser.parse_args(
        [
            "vision",
            "screen",
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "vision_screen"),
            "--species-context",
            str(context_path),
            "--yolo-runtime-python",
            str(yolo_python),
            "--bioclip-runtime-python",
            str(bioclip_python),
            "--chunk-rows",
            "2",
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    metrics = json.loads((tmp_path / "vision_screen" / "vision_screen_metrics.json").read_text(encoding="utf-8"))
    assert payload["image_cleanup_status"] == "commit_aware"
    assert metrics["cached_images_deleted"] == 2
    assert not no_detection_image.exists()
    assert not butterfly_image.exists()


def test_vision_screen_keeps_butterfly_image_when_score_write_fails(tmp_path, monkeypatch) -> None:
    yolo_python, bioclip_python = _fake_runtime_pythons(tmp_path)
    input_path = tmp_path / "canonical.parquet"
    pl.DataFrame([{"source": "flickr", "flickr_photo_id": "photo-butterfly", "image_url": "memory://photo-butterfly"}]).write_parquet(
        input_path
    )
    context_path = _write_screen_context(tmp_path)
    butterfly_image = tmp_path / "cache" / "butterfly.jpg"
    butterfly_image.parent.mkdir(parents=True)
    butterfly_image.write_bytes(b"butterfly")

    class FakeYoloDetector:
        backend = "yoloe26"
        model_id = "yoloe26"
        model_version = "test"
        checkpoint = "yoloe-26s-seg.pt"

        def __init__(self, **_kwargs):  # noqa: ANN003 - mirrors sidecar detector init.
            return None

        def close(self) -> None:
            return None

    class FakePersistentScorer:
        def __init__(self, **_kwargs):  # noqa: ANN003 - mirrors persistent scorer init.
            return None

        def close(self) -> None:
            return None

    class FakeCropScorer:
        model_id = "bioclip2_5"
        model_version = "bioclip2_5_huge"
        model_checkpoint = "fake-checkpoint"

        def __init__(self, **_kwargs):  # noqa: ANN003 - mirrors crop scorer init.
            return None

    def fake_load_decoded_image(record, *, cache_root):  # noqa: ANN001, ANN202 - mirrors image loader.
        return DecodedImage(width=1, height=1, mode="RGB", data=b"\x00\x00\x00", source_uri=str(butterfly_image))

    def fake_run_detection_pipeline(**kwargs):  # noqa: ANN003, ANN202 - mirrors run_detection_pipeline.
        for record in kwargs["records"]:
            kwargs["image_loader"](record)
        frame = pl.DataFrame(
            [
                {
                    "source": "flickr",
                    "flickr_photo_id": "photo-butterfly",
                    "detection_id": "det-butterfly",
                    "crop_hash": "sha256:butterfly",
                    "detector_label": "butterfly_like",
                    "detection_status": "detected",
                }
            ]
        )
        frame.write_parquet(kwargs["output_path"])
        return SimpleNamespace(
            frame=frame,
            output_path=Path(kwargs["output_path"]),
            records_seen=1,
            images_loaded=1,
            image_failures=0,
            detections_written=1,
            crops_created=1,
            parquet_batches_written=1,
        )

    def failing_screen_object_detections(**_kwargs):  # noqa: ANN003, ANN202 - mirrors screen_object_detections.
        raise RuntimeError("score write failed")

    monkeypatch.setattr("biominer.detection.yoloe26_detector.YoloE26SidecarObjectDetector", FakeYoloDetector)
    monkeypatch.setattr("biominer.cli.PersistentBioClipScorer", FakePersistentScorer)
    monkeypatch.setattr("biominer.cli.EphemeralCropBioClipScorer", FakeCropScorer)
    monkeypatch.setattr("biominer.cli.build_candidate_set", lambda context, **kwargs: SimpleNamespace(candidate_set_id="candidate-set-1"))
    monkeypatch.setattr("biominer.cli.load_decoded_image_from_record", fake_load_decoded_image)
    monkeypatch.setattr("biominer.cli.run_detection_pipeline", fake_run_detection_pipeline)
    monkeypatch.setattr("biominer.cli.screen_object_detections", failing_screen_object_detections)

    parser = build_parser()
    args = parser.parse_args(
        [
            "vision",
            "screen",
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "vision_screen"),
            "--species-context",
            str(context_path),
            "--yolo-runtime-python",
            str(yolo_python),
            "--bioclip-runtime-python",
            str(bioclip_python),
        ]
    )

    with pytest.raises(RuntimeError, match="score write failed"):
        run(args)

    assert butterfly_image.exists()


def _fake_runtime_pythons(tmp_path: Path) -> tuple[Path, Path]:
    yolo_python = tmp_path / "yolo" / "bin" / "python"
    bioclip_python = tmp_path / "bioclip" / "bin" / "python"
    yolo_python.parent.mkdir(parents=True)
    bioclip_python.parent.mkdir(parents=True)
    yolo_python.write_text("# fake yolo python", encoding="utf-8")
    bioclip_python.write_text("# fake bioclip python", encoding="utf-8")
    return yolo_python, bioclip_python


def _write_screen_context(tmp_path: Path) -> Path:
    context_path = tmp_path / "species_context.json"
    context_path.write_text(
        json.dumps(
            {
                "scientific_name": "Danaus plexippus",
                "accepted_taxon_key": "gbif:5131654",
                "canonical_name": "Danaus plexippus",
                "family": "Nymphalidae",
                "genus": "Danaus",
                "family_key": "gbif:7017",
                "genus_key": "gbif:1927164",
                "species_key": "gbif:5131654",
                "registry_version": "registry-v1",
            }
        ),
        encoding="utf-8",
    )
    return context_path


def test_bioclip_runtime_check_uses_sidecar_python(tmp_path, capsys, monkeypatch) -> None:
    runtime_python = tmp_path / "runtime" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("# fake python", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_run(cmd, *, capture_output, check, env, text):  # noqa: ANN001 - mirrors subprocess.run.
        calls.append({"cmd": cmd, "capture_output": capture_output, "check": check, "env": env, "text": text})
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout='{"device_resolved":"mps","model_load":true,"mps_available":true,"pytorch_mps_fallback_enabled":true}\n',
            stderr="",
        )

    monkeypatch.setattr("biominer.cli.subprocess.run", fake_run)
    parser = build_parser()
    args = parser.parse_args(
        [
            "dev",
            "vision",
            "bioclip-runtime-check",
            "--runtime-python",
            str(runtime_python),
            "--hf-cache-dir",
            str(tmp_path / "hf"),
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["device_resolved"] == "mps"
    assert payload["model_load"] is True
    assert payload["pytorch_mps_fallback_enabled"] is True
    assert calls[0]["cmd"][0] == str(runtime_python)
    assert calls[0]["cmd"][-1] == "auto"
    assert "hf-hub:imageomics/bioclip-2.5-vith14" in calls[0]["cmd"][2]
    assert "create_model_and_transforms" in calls[0]["cmd"][2]
    assert calls[0]["env"]["HF_HOME"] == str((tmp_path / "hf").resolve())
    assert calls[0]["env"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_yoloe26_runtime_check_uses_sidecar_python_and_mps_fallback(tmp_path, capsys, monkeypatch) -> None:
    runtime_python = tmp_path / "YOLO26" / "venv" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("# fake python", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_run(cmd, *, capture_output, check, cwd, env, text):  # noqa: ANN001 - mirrors subprocess.run.
        calls.append({"cmd": cmd, "capture_output": capture_output, "check": check, "cwd": cwd, "env": env, "text": text})
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout='{"checkpoint_resolved":true,"device_resolved":"mps","mps_available":true,"pytorch_mps_fallback_enabled":true}\n',
            stderr="",
        )

    monkeypatch.setattr("biominer.cli.subprocess.run", fake_run)
    parser = build_parser()
    args = parser.parse_args(
        [
            "dev",
            "vision",
            "yoloe26-runtime-check",
            "--runtime-python",
            str(runtime_python),
            "--device",
            "mps",
            "--checkpoint",
            "yoloe-26s-seg.pt",
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["device_resolved"] == "mps"
    assert payload["checkpoint_resolved"] is True
    assert payload["pytorch_mps_fallback_enabled"] is True
    assert calls[0]["cmd"][0] == str(runtime_python)
    assert calls[0]["cmd"][-2:] == ["mps", "yoloe-26s-seg.pt"]
    assert "from ultralytics import YOLOE" in calls[0]["cmd"][2]
    assert calls[0]["cwd"] == str(tmp_path / "YOLO26" / "models")
    assert calls[0]["env"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    assert calls[0]["env"]["BIOMINER_YOLO26_MODEL_DIR"] == str(tmp_path / "YOLO26" / "models")


def test_bioclip_prefetch_model_uses_snapshot_download_sidecar(tmp_path, capsys, monkeypatch) -> None:
    runtime_python = tmp_path / "runtime" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("# fake python", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_run(cmd, *, capture_output, check, env, text):  # noqa: ANN001 - mirrors subprocess.run.
        calls.append({"cmd": cmd, "capture_output": capture_output, "check": check, "env": env, "text": text})
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout='{"snapshot_path":"/tmp/hf/model","model_name":"imageomics/bioclip-2.5-vith14"}\n',
            stderr="",
        )

    monkeypatch.setattr("biominer.cli.subprocess.run", fake_run)
    parser = build_parser()
    args = parser.parse_args(
        [
            "dev",
            "vision",
            "bioclip-prefetch-model",
            "--runtime-python",
            str(runtime_python),
            "--hf-cache-dir",
            str(tmp_path / "hf"),
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["model_name"] == "imageomics/bioclip-2.5-vith14"
    assert calls[0]["cmd"][0] == str(runtime_python)
    assert calls[0]["cmd"][-3] == "imageomics/bioclip-2.5-vith14"
    assert calls[0]["env"]["HUGGINGFACE_HUB_CACHE"] == str((tmp_path / "hf" / "hub").resolve())


def test_whole_image_register_bioclip_commands_are_removed_from_public_cli() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["bioclip", "screen"])
    with pytest.raises(SystemExit):
        parser.parse_args(["species", "bioclip-funnel"])


def test_bioclip_screen_objects_uses_embedding_caches_for_detector_crop_scoring(tmp_path, capsys, monkeypatch) -> None:
    runtime_python = tmp_path / "runtime" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("# fake python", encoding="utf-8")
    context_path = tmp_path / "species_context.json"
    context_path.write_text(
        json.dumps(
            {
                "scientific_name": "Danaus plexippus",
                "accepted_taxon_key": "gbif:1",
                "canonical_name": "Danaus plexippus",
                "family": "Nymphalidae",
                "genus": "Danaus",
                "family_key": "gbif:f",
                "genus_key": "gbif:g",
                "species_key": "gbif:1",
                "registry_version": "registry-v1",
            }
        ),
        encoding="utf-8",
    )
    input_path = tmp_path / "filtered.parquet"
    detections_path = tmp_path / "detections.parquet"
    geo_prior_path = tmp_path / "geo_prior.parquet"
    output_path = tmp_path / "scores.parquet"
    text_cache_path = tmp_path / "candidate_text_embeddings.parquet"
    image_cache_path = tmp_path / "object_image_embeddings.parquet"
    pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "image_url": "https://example.test/1.jpg",
                "scientific_names_detected": ["Danaus gilippus"],
            }
        ]
    ).write_parquet(input_path)
    pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:1",
                "scientific_name": "Danaus plexippus",
                "bbox": "-170.0,5.0,-50.0,75.0",
            }
        ]
    ).write_parquet(geo_prior_path)
    pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "detection_id": "det-1",
                "crop_hash": "sha256:crop",
                "bbox_xyxy": [0.0, 0.0, 1.0, 1.0],
                "detection_status": "detected",
            }
        ]
    ).write_parquet(detections_path)
    calls: dict[str, object] = {}

    class FakePersistentScorer:
        def __init__(self, *, runtime, hf_cache_dir, device):  # noqa: ANN001 - mirrors scorer init.
            calls["persistent"] = {"runtime": runtime, "hf_cache_dir": hf_cache_dir, "device": device}

        def embed_text_labels(self, labels):  # noqa: ANN001, ANN201 - mirrors scorer API.
            calls.setdefault("embedded_labels", []).append(list(labels))
            return [[float(index), float(index + 1)] for index, _label in enumerate(labels)]

        def embed_image_paths(self, image_paths):  # noqa: ANN001, ANN201 - mirrors scorer API.
            calls["embedded_image_paths"] = [str(path) for path in image_paths]
            return [[0.5, 0.5] for _path in image_paths]

        def close(self) -> None:
            calls["closed"] = True

    class FakeCropScorer:
        def __init__(self, **kwargs):  # noqa: ANN003 - mirrors crop scorer init.
            calls["crop_scorer"] = kwargs
            self.model_id = kwargs["model_id"]
            self.model_version = kwargs["model_version"]
            self.model_checkpoint = kwargs["model_checkpoint"]

        def score(self, item, labels):  # noqa: ANN001, ANN201 - mirrors object scorer.
            return {label: 0.0 for label in labels}

    def fake_screen(**kwargs):  # noqa: ANN003, ANN202 - mirrors screen_object_detections.
        calls["screen"] = kwargs
        return SimpleNamespace(
            frame=pl.DataFrame([{"occurrence_bin": "bronze"}]),
            output_path=Path(kwargs["output_path"]),
            records_seen=1,
            detections_seen=1,
            crops_scored=1,
            score_batches_written=3,
            visual_classifier="bioclip_object",
            visual_mode="detector_crop",
            visual_mode_status="available",
            segmentation_unavailable_count=0,
            segmentation_unavailable_reason=None,
        )

    def fake_build_candidate_set(context, **kwargs):  # noqa: ANN001, ANN003, ANN202 - mirrors build_candidate_set.
        from biominer.bioclip.candidate_sets import CandidateTaxon

        calls["candidate_set"] = kwargs
        return SimpleNamespace(
            candidate_set_id="candidate-set-from-records",
            family_candidates=(CandidateTaxon(scientific_name="Nymphalidae", accepted_taxon_key="gbif:7017", rank="family"),),
            genus_candidates=(CandidateTaxon(scientific_name="Danaus", accepted_taxon_key="gbif:5131645", rank="genus"),),
            species_candidates=(CandidateTaxon(scientific_name="Danaus plexippus", accepted_taxon_key="gbif:5131654", rank="species"),),
        )

    def fake_materialize(**kwargs):  # noqa: ANN003, ANN202 - mirrors materialize_detector_crop_inputs.
        calls["materialize"] = kwargs
        crop_path = tmp_path / "crop.ppm"
        crop_path.write_bytes(b"P6\n1 1\n255\nabc")
        return SimpleNamespace(
            rows=[{"source": "flickr", "flickr_photo_id": "photo-1", "detection_id": "det-1", "crop_hash": "sha256:crop"}],
            crop_path_by_hash={"sha256:crop": crop_path},
            cleanup=lambda: calls.setdefault("materialized_cleanup", True),
        )

    class FakeCachedScorer:
        def __init__(self, **kwargs):  # noqa: ANN003 - mirrors cached scorer init.
            calls["cached_scorer"] = kwargs
            self.model_id = kwargs["model_id"]
            self.model_version = kwargs["model_version"]
            self.model_checkpoint = kwargs["model_checkpoint"]

        def score(self, item, labels):  # noqa: ANN001, ANN201 - mirrors object scorer.
            return {label: 0.0 for label in labels}

    monkeypatch.setattr("biominer.cli.PersistentBioClipScorer", FakePersistentScorer)
    monkeypatch.setattr("biominer.cli.EphemeralCropBioClipScorer", FakeCropScorer)
    monkeypatch.setattr("biominer.cli.screen_object_detections", fake_screen)
    monkeypatch.setattr("biominer.cli.build_candidate_set", fake_build_candidate_set)
    monkeypatch.setattr("biominer.cli.materialize_detector_crop_inputs", fake_materialize)
    monkeypatch.setattr("biominer.cli.CachedObjectEmbeddingScorer", FakeCachedScorer)
    parser = build_parser()
    args = parser.parse_args(
        [
            "vision",
            "score",
            "--input",
            str(input_path),
            "--detections",
            str(detections_path),
            "--species-context",
            str(context_path),
            "--output",
            str(output_path),
            "--geo-prior-table",
            str(geo_prior_path),
            "--runtime-python",
            str(runtime_python),
            "--device",
            "mps",
            "--parquet-batch-rows",
            "3",
            "--bioclip-batch",
            "5",
            "--text-embedding-batch-size",
            "2",
            "--candidate-text-embedding-cache",
            str(text_cache_path),
            "--object-image-embedding-cache",
            str(image_cache_path),
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == 1
    assert payload["primary_visual_classifier"] == "bioclip_object"
    assert payload["visual_mode"] == "detector_crop"
    assert payload["visual_mode_status"] == "available"
    assert payload["scorer"] == "cached_object_embedding"
    assert payload["score_batches_written"] == 3
    assert payload["candidate_text_embedding_cache"]["embeddings_computed"] == 4
    assert payload["candidate_text_embedding_cache"]["rows_added"] == 4
    assert payload["candidate_text_embedding_cache"]["text_embedding_batch_size"] == 2
    assert payload["object_image_embedding_cache"]["embeddings_computed"] == 1
    assert payload["object_image_embedding_cache"]["rows_added"] == 1
    assert calls["closed"] is True
    assert calls["materialized_cleanup"] is True
    assert calls["embedded_labels"] == [["Nymphalidae", "Danaus"], ["Danaus plexippus", "a photo of Danaus plexippus"]]
    assert calls["embedded_image_paths"] == [str(tmp_path / "crop.ppm")]
    assert calls["persistent"]["device"] == "mps"
    assert calls["cached_scorer"]["candidate_set_id"] == "candidate-set-from-records"
    assert calls["cached_scorer"]["model_checkpoint"] == "191d741545e4c741cdef4b22c6eb69c945c1e592"
    assert calls["screen"]["scorer"].model_id == "bioclip2_5"
    assert calls["screen"]["ablation_mode"] == "detector_crop"
    assert calls["screen"]["parquet_batch_rows"] == 3
    assert calls["screen"]["bioclip_batch_size"] == 5
    assert calls["screen"]["geo_prior_table"].height == 1
    assert calls["candidate_set"]["geospatial_scope"] == str(geo_prior_path)
    assert calls["candidate_set"]["geo_prior_table"].height == 1
    assert calls["candidate_set"]["records"][0]["flickr_photo_id"] == "photo-1"
    assert calls["candidate_set"]["records"][0]["scientific_names_detected"] == ["Danaus gilippus"]
    assert pl.read_parquet(text_cache_path).height == 4
    assert pl.read_parquet(image_cache_path).height == 1


def test_bioclip_ablate_objects_forwards_parquet_batch_rows(tmp_path, capsys, monkeypatch) -> None:
    runtime_python = tmp_path / "runtime" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("# fake python", encoding="utf-8")
    context_path = tmp_path / "species_context.json"
    context_path.write_text(
        json.dumps(
            {
                "scientific_name": "Danaus plexippus",
                "accepted_taxon_key": "gbif:1",
                "canonical_name": "Danaus plexippus",
                "family": "Nymphalidae",
                "genus": "Danaus",
                "family_key": "gbif:f",
                "genus_key": "gbif:g",
                "species_key": "gbif:1",
                "registry_version": "registry-v1",
            }
        ),
        encoding="utf-8",
    )
    input_path = tmp_path / "filtered.parquet"
    detections_path = tmp_path / "detections.parquet"
    output_dir = tmp_path / "ablations"
    pl.DataFrame([{"source": "flickr", "flickr_photo_id": "photo-1", "image_url": "https://example.test/1.jpg"}]).write_parquet(input_path)
    pl.DataFrame([{"source": "flickr", "flickr_photo_id": "photo-1", "detection_id": "det-1", "detection_status": "detected"}]).write_parquet(
        detections_path
    )
    calls: dict[str, object] = {}

    class FakePersistentScorer:
        def __init__(self, **kwargs):  # noqa: ANN003 - mirrors scorer init.
            calls["persistent"] = kwargs

        def close(self) -> None:
            calls["closed"] = True

    class FakeCropScorer:
        def __init__(self, **kwargs):  # noqa: ANN003 - mirrors crop scorer init.
            calls["crop_scorer"] = kwargs

    def fake_build_candidate_set(context, **kwargs):  # noqa: ANN001, ANN003, ANN202 - mirrors build_candidate_set.
        calls["candidate_set"] = kwargs
        return SimpleNamespace(candidate_set_id="candidate-set")

    def fake_run_ablations(**kwargs):  # noqa: ANN003, ANN202 - mirrors run_object_ablations.
        calls["ablation"] = kwargs
        return SimpleNamespace(
            output_dir=Path(kwargs["output_dir"]),
            report={"score_batches_written_by_mode": {"detector_crop": 2}, "score_batches_written": 2},
        )

    monkeypatch.setattr("biominer.cli.PersistentBioClipScorer", FakePersistentScorer)
    monkeypatch.setattr("biominer.cli.EphemeralCropBioClipScorer", FakeCropScorer)
    monkeypatch.setattr("biominer.cli.build_candidate_set", fake_build_candidate_set)
    monkeypatch.setattr("biominer.cli.run_object_ablations", fake_run_ablations)
    parser = build_parser()
    args = parser.parse_args(
        [
            "vision",
            "ablate",
            "--input",
            str(input_path),
            "--detections",
            str(detections_path),
            "--species-context",
            str(context_path),
            "--output-dir",
            str(output_dir),
            "--runtime-python",
            str(runtime_python),
            "--modes",
            "detector_crop",
            "--parquet-batch-rows",
            "2",
            "--bioclip-batch",
            "6",
            "--segmenter",
            "none",
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["score_batches_written"] == 2
    assert calls["closed"] is True
    assert calls["ablation"]["parquet_batch_rows"] == 2
    assert calls["ablation"]["bioclip_batch_size"] == 6
    assert calls["ablation"]["modes"] == ("detector_crop",)
    assert calls["crop_scorer"]["segmenter"].backend == "none"
    assert calls["candidate_set"]["geo_prior_table"] is None


def test_bioclip_score_reports_missing_candidate_expansion(tmp_path, capsys) -> None:
    runtime_python, context_path, input_path, detections_path = _write_candidate_expansion_error_inputs(tmp_path)
    parser = build_parser()
    args = parser.parse_args(
        [
            "vision",
            "score",
            "--input",
            str(input_path),
            "--detections",
            str(detections_path),
            "--species-context",
            str(context_path),
            "--output",
            str(tmp_path / "scores.parquet"),
            "--runtime-python",
            str(runtime_python),
        ]
    )

    assert run(args) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "vision score"
    assert "registry-derived same-genus/same-family candidates" in payload["error"]
    assert "--species-candidates" in payload["hint"]


def test_bioclip_ablate_reports_missing_candidate_expansion(tmp_path, capsys) -> None:
    runtime_python, context_path, input_path, detections_path = _write_candidate_expansion_error_inputs(tmp_path)
    parser = build_parser()
    args = parser.parse_args(
        [
            "vision",
            "ablate",
            "--input",
            str(input_path),
            "--detections",
            str(detections_path),
            "--species-context",
            str(context_path),
            "--output-dir",
            str(tmp_path / "ablations"),
            "--runtime-python",
            str(runtime_python),
        ]
    )

    assert run(args) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "vision ablate"
    assert "registry-derived same-genus/same-family candidates" in payload["error"]
    assert "--species-candidates" in payload["hint"]


def test_evidence_join_cli_writes_join_tables(tmp_path, capsys, monkeypatch) -> None:
    context_path = tmp_path / "species_context.json"
    context_path.write_text(
        json.dumps(
            {
                "scientific_name": "Danaus plexippus",
                "accepted_taxon_key": "gbif:1",
                "canonical_name": "Danaus plexippus",
                "family": "Nymphalidae",
                "genus": "Danaus",
                "family_key": "gbif:f",
                "genus_key": "gbif:g",
                "species_key": "gbif:1",
                "registry_version": "registry-v1",
            }
        ),
        encoding="utf-8",
    )
    input_path = tmp_path / "filtered.parquet"
    detections_path = tmp_path / "object_detections.parquet"
    scores_path = tmp_path / "object_bioclip_scores.parquet"
    joined_path = tmp_path / "object_evidence_joined.parquet"
    summary_path = tmp_path / "photo_evidence_summary.parquet"
    calls: dict[str, object] = {}

    def fake_write_outputs(**kwargs):  # noqa: ANN003, ANN202 - mirrors write_object_evidence_outputs.
        calls["kwargs"] = kwargs
        return SimpleNamespace(
            object_evidence_joined=Path(kwargs["joined_output_path"]),
            photo_evidence_summary=Path(kwargs["photo_summary_output_path"]),
        )

    monkeypatch.setattr("biominer.cli.write_object_evidence_outputs", fake_write_outputs)
    parser = build_parser()
    args = parser.parse_args(
        [
            "evidence",
            "join",
            "--input",
            str(input_path),
            "--detections",
            str(detections_path),
            "--scores",
            str(scores_path),
            "--joined-output",
            str(joined_path),
            "--photo-summary-output",
            str(summary_path),
            "--species-context",
            str(context_path),
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    kwargs = calls["kwargs"]
    assert payload == {
        "object_evidence_joined": str(joined_path),
        "photo_evidence_summary": str(summary_path),
    }
    assert kwargs["canonical_records_path"] == str(input_path)
    assert kwargs["detections_path"] == str(detections_path)
    assert kwargs["scores_path"] == str(scores_path)
    assert kwargs["species_context"].scientific_name == "Danaus plexippus"


def test_detect_boxes_fake_backend_writes_crop_metadata(tmp_path, capsys) -> None:
    input_path = tmp_path / "filtered.parquet"
    output_path = tmp_path / "object_detections.parquet"
    pl.DataFrame(
        [
            {
                "source": "flickr",
                "flickr_photo_id": "photo-1",
                "image_url": "memory://photo-1",
                "image_width": 4,
                "image_height": 4,
            }
        ]
    ).write_parquet(input_path)

    parser = build_parser()
    args = parser.parse_args(
        [
            "vision",
            "detect",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--backend",
            "fake",
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    row = pl.read_parquet(output_path).to_dicts()[0]
    assert payload["rows"] == 1
    assert row["crop_hash"].startswith("sha256:")
    assert row["crop_width"] == 336
    assert row["crop_height"] == 336
    assert row["crop_storage_policy"] == "ephemeral"


def test_detect_boxes_public_backend_excludes_legacy_yolo() -> None:
    parser = build_parser()
    detection_dir = Path("src/biominer/detection")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "vision",
                "detect",
                "--input",
                "filtered.parquet",
                "--output",
                "object_detections.parquet",
                "--backend",
                "yolo",
            ]
        )
    assert importlib.util.find_spec("biominer.detection.yolo_detector") is None
    assert not (detection_dir / "yolo_detector.py").exists()
    assert not any("yolov8n.pt" in path.read_text(encoding="utf-8") for path in detection_dir.glob("*.py"))


def test_detect_boxes_yoloe26_backend_uses_sidecar_runtime(tmp_path, monkeypatch) -> None:
    runtime_python = tmp_path / "YOLO26" / "venv" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("# fake python", encoding="utf-8")
    calls: dict[str, object] = {}

    class FakeYoloE26SidecarDetector:
        backend = "yoloe26"
        model_id = "fake-yoloe26-sidecar"
        model_version = "sidecar-test"
        checkpoint = "yoloe-26s-seg.pt"

        def __init__(self, **kwargs):  # noqa: ANN003 - mirrors sidecar init.
            calls["sidecar_init"] = kwargs

        def detect_batch(self, images):  # noqa: ANN001, ANN201 - mirrors ObjectDetector protocol.
            return [[] for _image in images]

    class InProcessYoloE26Detector:
        def __init__(self, **kwargs):  # noqa: ANN003, ANN204 - should not be called.
            raise AssertionError(f"in-process YOLOE-26 should not be used when sidecar exists: {kwargs}")

    monkeypatch.setattr("biominer.detection.yoloe26_detector.YoloE26SidecarObjectDetector", FakeYoloE26SidecarDetector)
    monkeypatch.setattr("biominer.detection.yoloe26_detector.YoloE26ObjectDetector", InProcessYoloE26Detector)
    parser = build_parser()
    args = parser.parse_args(
        [
            "vision",
            "detect",
            "--input",
            str(tmp_path / "filtered.parquet"),
            "--output",
            str(tmp_path / "object_detections.parquet"),
            "--backend",
            "yoloe26",
            "--runtime-python",
            str(runtime_python),
            "--device",
            "mps",
            "--checkpoint",
            "yoloe-26s-seg.pt",
            "--imgsz",
            "640",
            "--conf",
            "0.2",
            "--iou",
            "0.5",
            "--max-det",
            "8",
        ]
    )

    detector, image_loader = _detect_boxes_backend(args, [])

    assert detector.backend == "yoloe26"
    assert image_loader is load_decoded_image_from_record
    assert calls["sidecar_init"] == {
        "runtime_python": str(runtime_python),
        "checkpoint": "yoloe-26s-seg.pt",
        "device": "mps",
        "imgsz": 640,
        "conf": 0.2,
        "iou": 0.5,
        "max_det": 8,
        "prompt_classes": (
            "butterfly",
            "moth",
            "caterpillar",
            "chrysalis",
            "pupa",
            "insect",
            "butterfly wing",
            "pinned butterfly specimen",
            "butterfly specimen",
            "lepidoptera",
            "flower",
            "leaf",
            "person",
            "hand",
            "drawing",
            "painting",
            "logo",
            "text",
            "sign",
            "museum label",
        ),
    }


def test_detect_boxes_yolo26_backend_requires_explicit_checkpoint(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "vision",
            "detect",
            "--input",
            str(tmp_path / "filtered.parquet"),
            "--output",
            str(tmp_path / "object_detections.parquet"),
            "--backend",
            "yolo26",
        ]
    )

    with pytest.raises(ValueError, match="user-provided coarse object checkpoint"):
        _detect_boxes_backend(args, [])


def test_detect_boxes_yolo26_backend_uses_sidecar_runtime(tmp_path, monkeypatch) -> None:
    runtime_python = tmp_path / "YOLO26" / "venv" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("# fake python", encoding="utf-8")
    calls: dict[str, object] = {}

    class FakeYolo26SidecarDetector:
        backend = "yolo26"
        model_id = "fake-yolo26-sidecar"
        model_version = "sidecar-test"
        checkpoint = "coarse-objects.pt"

        def __init__(self, **kwargs):  # noqa: ANN003 - mirrors sidecar init.
            calls["sidecar_init"] = kwargs

        def detect_batch(self, images):  # noqa: ANN001, ANN201 - mirrors ObjectDetector protocol.
            return [[] for _image in images]

    class InProcessYolo26Detector:
        def __init__(self, **kwargs):  # noqa: ANN003, ANN204 - should not be called.
            raise AssertionError(f"in-process YOLO26 should not be used when sidecar exists: {kwargs}")

    monkeypatch.setattr("biominer.detection.yolo26_detector.Yolo26SidecarObjectDetector", FakeYolo26SidecarDetector)
    monkeypatch.setattr("biominer.detection.yolo26_detector.Yolo26ObjectDetector", InProcessYolo26Detector)
    parser = build_parser()
    args = parser.parse_args(
        [
            "vision",
            "detect",
            "--input",
            str(tmp_path / "filtered.parquet"),
            "--output",
            str(tmp_path / "object_detections.parquet"),
            "--backend",
            "yolo26",
            "--runtime-python",
            str(runtime_python),
            "--device",
            "mps",
            "--checkpoint",
            "coarse-objects.pt",
            "--imgsz",
            "640",
            "--conf",
            "0.2",
            "--iou",
            "0.5",
            "--max-det",
            "8",
        ]
    )

    detector, image_loader = _detect_boxes_backend(args, [])

    assert detector.backend == "yolo26"
    assert image_loader is load_decoded_image_from_record
    assert calls["sidecar_init"] == {
        "runtime_python": str(runtime_python),
        "checkpoint": "coarse-objects.pt",
        "device": "mps",
        "imgsz": 640,
        "conf": 0.2,
        "iou": 0.5,
        "max_det": 8,
    }


def test_yoloe26_runtime_commands_parse_with_applications_defaults() -> None:
    parser = build_parser()

    runtime = parser.parse_args(["dev", "vision", "yoloe26-runtime-check", "--device", "mps"])
    prefetch = parser.parse_args(["dev", "vision", "yoloe26-prefetch", "--checkpoint", "yoloe-26s-seg.pt"])
    smoke = parser.parse_args(["dev", "vision", "yoloe26-smoke", "--image", "manual.jpg"])
    prototype = parser.parse_args(
        [
            "dev",
            "vision",
            "yoloe26-prototype-run",
            "--input",
            "filtered.parquet",
            "--species-context",
            "species_context.json",
            "--output-dir",
            "reports/yoloe26",
            "--limit",
            "10",
        ]
    )

    assert runtime.vision_command == "yoloe26-runtime-check"
    assert runtime.dev_command == "vision"
    assert runtime.runtime_python.endswith("/YOLO26/venv/bin/python")
    assert prefetch.checkpoint == "yoloe-26s-seg.pt"
    assert smoke.image == "manual.jpg"
    assert prototype.vision_runtime_python.endswith("/YOLO26/venv/bin/python")
    assert prototype.bioclip_runtime_python.endswith("/BioCLIP25/venv/bin/python")
    assert prototype.limit == 10


def test_public_vision_surface_excludes_debug_runtime_commands() -> None:
    parser = build_parser()

    for command in (
        "bioclip-runtime-check",
        "bioclip-prefetch-model",
        "yoloe26-runtime-check",
        "yoloe26-prefetch",
        "yoloe26-smoke",
        "yoloe26-prototype-run",
        "crop-preview",
        "eval",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(["vision", command])


def test_yoloe26_smoke_resolves_paths_before_sidecar_run(tmp_path, capsys, monkeypatch) -> None:
    runtime_python = tmp_path / "YOLO26" / "venv" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("# fake python", encoding="utf-8")
    image_path = tmp_path / "manual.jpg"
    image_path.write_bytes(b"not-a-real-image-for-this-mocked-test")
    calls: list[dict[str, object]] = []

    def fake_run(cmd, *, capture_output, check, cwd, env, text):  # noqa: ANN001 - mirrors subprocess.run.
        calls.append({"cmd": cmd, "capture_output": capture_output, "check": check, "cwd": cwd, "env": env, "text": text})
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout='{"detections":0,"synthetic_image":false}\n',
            stderr="",
        )

    monkeypatch.setattr("biominer.cli.subprocess.run", fake_run)
    monkeypatch.chdir(tmp_path)
    parser = build_parser()
    args = parser.parse_args(
        [
            "dev",
            "vision",
            "yoloe26-smoke",
            "--runtime-python",
            str(runtime_python),
            "--image",
            "manual.jpg",
            "--output-dir",
            "smoke-output",
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    command = calls[0]["cmd"]
    assert payload["synthetic_image"] is False
    assert command[0] == str(runtime_python)
    assert command[5] == str((tmp_path / "smoke-output").resolve())
    assert command[6] == str(image_path.resolve())
    assert calls[0]["cwd"] == str(tmp_path / "YOLO26" / "models")


def test_yoloe26_prototype_metrics_aggregate_tiny_frames() -> None:
    detections = pl.DataFrame(
        [
            {"detection_status": "detected", "detector_label": "butterfly_like", "detector_score": 0.8},
            {"detection_status": "detected", "detector_label": "hard_negative", "detector_score": 0.2},
            {"detection_status": "no_detection", "detector_label": None, "detector_score": 0.0},
        ]
    )
    scores = pl.DataFrame(
        [
            {
                "occurrence_bin": "silver",
                "species_top1_scientific_name": "Danaus plexippus",
                "species_top1_score": 0.7,
                "species_top1_margin": 0.3,
            },
            {
                "occurrence_bin": "bronze",
                "species_top1_scientific_name": "Danaus plexippus",
                "species_top1_score": 0.4,
                "species_top1_margin": 0.1,
            },
        ]
    )
    photo_summary = pl.DataFrame([{"photo_occurrence_bin": "silver"}, {"photo_occurrence_bin": "bin"}])

    metrics = _yoloe26_metrics(
        detection_result=SimpleNamespace(
            frame=detections,
            records_seen=3,
            images_loaded=2,
            image_failures=1,
            detections_written=2,
            crops_created=2,
        ),
        score_frame=scores,
        photo_summary=photo_summary,
        checkpoint="yoloe-26s-seg.pt",
        prompt_classes=("butterfly",),
        device="mps",
        imgsz=640,
        conf=0.2,
        iou=0.5,
    )

    assert metrics["metrics_kind"] == "heuristic_without_ground_truth"
    assert metrics["records_seen"] == 3
    assert metrics["detections_by_detector_label"] == {"butterfly_like": 1, "hard_negative": 1}
    assert metrics["hard_negative_count"] == 1
    assert metrics["occurrence_bin_counts"] == {"bronze": 1, "silver": 1}
    assert metrics["photo_occurrence_bin_counts"] == {"bin": 1, "silver": 1}
    assert metrics["mean_species_top1_score"] == 0.55
    assert metrics["top20_bioclip_top1_species"] == [{"value": "Danaus plexippus", "count": 2}]


def _fake_cli_image(record: dict[str, object]):
    from biominer.detection.detector_base import DecodedImage

    width = max(1, int(record.get("image_width") or 1))
    height = max(1, int(record.get("image_height") or 1))
    return DecodedImage(width=width, height=height, mode="RGB", data=b"\x00\x00\x00" * width * height)


def test_production_run_cli_resolves_registry_and_writes_dry_run_manifest(tmp_path, capsys) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:100",
                "scientific_name": "Papilio demoleus",
                "rank": "SPECIES",
                "family_key": "gbif:10",
                "family": "Papilionidae",
                "genus_key": "gbif:90",
                "genus": "Papilio",
                "species_key": "gbif:100",
                "species": "Papilio demoleus",
            }
        ]
    ).write_parquet(registry / "taxa.parquet")
    pl.DataFrame(
        [
            {
                "name_id": "name:1",
                "registry_version": "registry-v1",
                "accepted_taxon_key": "gbif:100",
                "verbatim_name": "Papilio demoleus",
                "display_name": "Papilio demoleus",
                "language": "la",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "accepted_scientific",
                "source": "gbif",
                "source_record_id": "gbif:100",
                "trust_tier": "T1",
                "precision_tier": "high",
                "confidence": "high",
                "enabled": True,
                "disabled_reason": "",
            }
        ]
    ).write_parquet(registry / "names.parquet")
    pl.DataFrame([]).write_parquet(registry / "name_evidence.parquet")
    pl.DataFrame([{"source": "gbif", "source_version": "fixture", "retrieved_at": "2026-01-01"}]).write_parquet(
        registry / "source_snapshots.parquet"
    )
    pl.DataFrame([]).write_parquet(registry / "flickr_query_definitions.parquet")
    (registry / "manifest.json").write_text(json.dumps({"registry_version": "registry-v1"}), encoding="utf-8")
    output = tmp_path / "species_run"
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--taxon",
            "Papilio demoleus",
            "--rank",
            "species",
            "--registry-dir",
            str(registry),
            "--output-prefix",
            str(output),
            "--storage-backend",
            "local",
            "--workstore-backend",
            "sqlite",
            "--dry-run",
        ]
    )

    assert run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    manifest = payload["manifest"]

    assert payload["request"]["taxon"] == "Papilio demoleus"
    assert payload["request"]["storage_backend"] == "local"
    assert payload["request"]["workstore_backend"] == "sqlite"
    assert payload["request"]["worker_id"] == "local"
    assert manifest["taxon_scope"]["accepted_taxon_key"] == "gbif:100"
    assert manifest["taxon_scope"]["accepted_rank"] == "species"
    assert manifest["stages"][0]["stage"] == "resolve_taxon_scope"
    assert manifest["stages"][0]["status"] == "complete"
    assert manifest["stages"][1]["status"] == "skipped"
    assert (output / "run_id=species_papilio_demoleus" / "run_manifest.json").exists()


def test_production_run_requires_cloud_config_by_default(tmp_path, capsys, monkeypatch) -> None:
    config = BioMinerConfig(
        storage=StorageConfig(
            backend="s3",
            bucket=None,
            prefix="",
            endpoint_url=None,
            access_key_id=None,
            secret_access_key=None,
            region="",
        ),
        workstore=WorkStoreConfig(backend="postgres", dsn=None),
        runtime=RuntimeConfig(worker_id=""),
    )
    monkeypatch.setattr("biominer.cli.load_biominer_config", lambda path: config)
    args = build_parser().parse_args(
        [
            "run",
            "--taxon",
            "Papilio demoleus",
            "--registry-dir",
            str(tmp_path / "registry"),
            "--output-prefix",
            "s3://biominer/runs",
            "--dry-run",
        ]
    )

    assert run(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert "BIOMINER_S3_BUCKET" in payload["error"]
    assert "BIOMINER_WORKSTORE_DSN" in payload["error"]
    assert payload["config"]["storage"]["secret_access_key"] is None


def test_production_run_dry_run_reads_cloud_registry(capsys, monkeypatch) -> None:
    fake_storage = _FakeCloudStorage()
    registry_uri = "s3://biominer/registry/current"
    fake_storage.parquet_payloads[f"{registry_uri}/taxa.parquet"] = pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:100",
                "scientific_name": "Papilio demoleus",
                "rank": "SPECIES",
                "family_key": "gbif:10",
                "family": "Papilionidae",
                "genus_key": "gbif:90",
                "genus": "Papilio",
                "species_key": "gbif:100",
                "species": "Papilio demoleus",
                "parent_key": "gbif:90",
            }
        ]
    )
    fake_storage.parquet_payloads[f"{registry_uri}/names.parquet"] = pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Papilio demoleus",
                "name_class": "accepted_scientific",
                "language": "la",
                "source": "GBIF",
                "trust_tier": "T1",
                "enabled": True,
                "disabled_reason": "",
            }
        ]
    )
    fake_storage.json_payloads[f"{registry_uri}/manifest.json"] = {"registry_version": "registry-v1"}
    config = _fake_cloud_config()
    monkeypatch.setattr("biominer.cli.load_biominer_config", lambda path: config)
    monkeypatch.setattr("biominer.cli.create_storage_backend", lambda storage_config: fake_storage)

    def fail_create_workstore(_workstore_config):  # noqa: ANN001, ANN202
        raise AssertionError("dry-run registry resolution should not open the workstore")

    monkeypatch.setattr("biominer.cli.create_workstore", fail_create_workstore)

    args = build_parser().parse_args(
        [
            "run",
            "--taxon",
            "Papilio demoleus",
            "--rank",
            "species",
            "--registry-dir",
            registry_uri,
            "--output-prefix",
            "s3://biominer/runs",
            "--dry-run",
        ]
    )

    assert run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    manifest_uri = "s3://biominer/runs/run_id=species_papilio_demoleus/run_manifest.json"
    assert payload["manifest"]["taxon_scope"]["accepted_taxon_key"] == "gbif:100"
    assert payload["manifest"]["taxon_scope"]["registry_version"] == "registry-v1"
    assert fake_storage.json_payloads[manifest_uri]["status"] == "complete"


def test_production_run_enqueue_stage_uses_configured_workstore(tmp_path, capsys, monkeypatch) -> None:
    from biominer.workstore.sqlite import SQLiteWorkStore

    registry = tmp_path / "registry"
    registry.mkdir()
    pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:100",
                "scientific_name": "Papilio demoleus",
                "rank": "SPECIES",
                "family_key": "gbif:10",
                "family": "Papilionidae",
                "genus_key": "gbif:90",
                "genus": "Papilio",
                "species_key": "gbif:100",
                "species": "Papilio demoleus",
                "parent_key": "gbif:90",
            }
        ]
    ).write_parquet(registry / "taxa.parquet")
    pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Papilio demoleus",
                "name_class": "accepted_scientific",
                "language": "la",
                "source": "GBIF",
                "trust_tier": "T1",
                "enabled": True,
                "disabled_reason": "",
            }
        ]
    ).write_parquet(registry / "names.parquet")
    pl.DataFrame(
        [
            {
                "query_definition_id": "q-1",
                "registry_version": "registry-v1",
                "accepted_taxon_key": "gbif:100",
                "accepted_scientific_name": "Papilio demoleus",
                "family_key": "gbif:10",
                "genus_key": "gbif:90",
                "species_key": "gbif:100",
                "source_term": "Papilio demoleus",
                "language": "la",
                "search_field": "text",
                "search_priority": 10,
                "bbox": "",
                "region": "",
                "name_class": "accepted_scientific",
                "confidence": "high",
                "enabled": True,
            }
        ]
    ).write_parquet(registry / "flickr_query_definitions.parquet")
    (registry / "manifest.json").write_text(json.dumps({"registry_version": "registry-v1"}), encoding="utf-8")
    workstore_path = tmp_path / "workstore.sqlite"
    config = BioMinerConfig(
        storage=StorageConfig(backend="local", prefix=str(tmp_path / "artifacts")),
        workstore=WorkStoreConfig(backend="sqlite", sqlite_path=str(workstore_path), dsn_env=None),
        runtime=RuntimeConfig(worker_id="worker-1"),
    )
    monkeypatch.setattr("biominer.cli.load_biominer_config", lambda path: config)

    args = build_parser().parse_args(
        [
            "run",
            "--taxon",
            "Papilio demoleus",
            "--rank",
            "species",
            "--registry-dir",
            str(registry),
            "--output-prefix",
            str(tmp_path / "runs"),
            "--storage-backend",
            "local",
            "--workstore-backend",
            "sqlite",
            "--stages",
            "resolve,enqueue",
            "--limit-records",
            "1",
        ]
    )

    assert run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["request"]["worker_id"] == "worker-1"
    assert payload["manifest"]["query_counts"]["enqueued_work_items"] == 1
    queued = SQLiteWorkStore(workstore_path).list_work_items(
        job_name="biominer_production_run",
        stage="poll_flickr",
        registry_version="registry-v1",
    )
    assert len(queued) == 1
    assert queued[0]["payload"]["query"]["query_definition_id"] == "q-1"


def test_registry_compile_fixture_cli_writes_registry_outputs(tmp_path, capsys) -> None:
    source = tmp_path / "registry_source.json"
    source.write_text(
        json.dumps(
            {
                "source": "fixture",
                "source_version": "2026-06-20",
                "retrieved_at": "2026-06-20T00:00:00+00:00",
                "taxa": [
                    {
                        "accepted_taxon_key": "gbif:1",
                        "scientific_name": "Papilionoidea",
                        "rank": "SUPERFAMILY",
                    },
                    {
                        "accepted_taxon_key": "gbif:100",
                        "scientific_name": "Papilio demoleus",
                        "rank": "SPECIES",
                        "family_key": "gbif:10",
                        "family": "Papilionidae",
                        "genus_key": "gbif:90",
                        "genus": "Papilio",
                        "species_key": "gbif:100",
                        "species": "Papilio demoleus",
                    }
                ],
                "names": [
                    {
                        "accepted_taxon_key": "gbif:1",
                        "verbatim_name": "Papilionoidea",
                        "display_name": "Papilionoidea",
                        "language": "la",
                        "script": "Latn",
                        "name_class": "accepted_scientific",
                        "source": "GBIF",
                        "source_record_id": "gbif:1",
                        "trust_tier": "T1",
                        "precision_tier": "high",
                        "confidence": "high",
                        "enabled": True,
                    },
                    {
                        "accepted_taxon_key": "gbif:100",
                        "verbatim_name": "Papilio demoleus",
                        "display_name": "Papilio demoleus",
                        "language": "la",
                        "script": "Latn",
                        "name_class": "accepted_scientific",
                        "source": "GBIF",
                        "source_record_id": "gbif:100",
                        "trust_tier": "T1",
                        "precision_tier": "high",
                        "confidence": "high",
                        "enabled": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "registry"
    parser = build_parser()
    args = parser.parse_args(
        [
            "dev",
            "registry",
            "compile-fixture",
            "--source-json",
            str(source),
            "--output-dir",
            str(output),
            "--registry-version",
            "test-registry",
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["registry_version"] == "test-registry"
    assert payload["query_definition_rows"] == 2
    assert (output / "manifest.json").exists()
    assert (output / "flickr_query_definitions.parquet").exists()


def test_registry_fetch_taxonomy_cli_writes_gbif_source_snapshot(tmp_path, capsys, monkeypatch) -> None:
    output = tmp_path / "gbif_source.json"

    def fake_build(client, scope, *, retrieved_at):  # noqa: ANN001 - CLI test verifies wiring, not types.
        return {
            "source": "GBIF",
            "source_version": "gbif-species-api",
            "retrieved_at": retrieved_at,
            "taxa": [{"accepted_taxon_key": "gbif:1", "scientific_name": scope.root_scientific_name}],
            "names": [],
            "source_assertions": [],
        }

    monkeypatch.setattr("biominer.cli.build_gbif_source_snapshot", fake_build)
    parser = build_parser()
    args = parser.parse_args(
        [
            "dev",
            "registry",
            "fetch-taxonomy",
            "--output-json",
            str(output),
            "--retrieved-at",
            "2026-06-20T00:00:00+00:00",
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    source = json.loads(output.read_text(encoding="utf-8"))
    assert payload == {
        "output_json": str(output),
        "source": "GBIF",
        "taxa_rows": 1,
        "name_rows": 0,
        "source_assertion_rows": 0,
    }
    assert source["taxa"][0]["scientific_name"] == "Papilionoidea"


def test_registry_build_cli_reuses_source_json_and_writes_report(tmp_path, capsys) -> None:
    scope = tmp_path / "scope.json"
    scope.write_text(
        json.dumps(
            {
                "scope_id": "test-scope",
                "root": {"scientific_name": "Papilionoidea", "rank": "SUPERFAMILY"},
                "included_families": [],
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "registry_source.json"
    source.write_text(
        json.dumps(
            {
                "source": "GBIF",
                "source_version": "gbif-species-api",
                "retrieved_at": "2026-06-20T00:00:00+00:00",
                "metrics": {"gbif_calls": 12},
                "taxa": [
                    {
                        "accepted_taxon_key": "gbif:1",
                        "scientific_name": "Papilionoidea",
                        "rank": "SUPERFAMILY",
                    }
                ],
                "names": [
                    {
                        "accepted_taxon_key": "gbif:1",
                        "verbatim_name": "Papilionoidea",
                        "display_name": "Papilionoidea",
                        "language": "la",
                        "script": "Latn",
                        "name_class": "accepted_scientific",
                        "source": "GBIF",
                        "source_record_id": "gbif:1",
                        "trust_tier": "T1",
                        "precision_tier": "high",
                        "confidence": "high",
                        "enabled": True,
                    }
                ],
                "source_assertions": [],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "registry"
    reports = tmp_path / "reports"
    parser = build_parser()
    args = parser.parse_args(
        [
            "registry",
            "build",
            "--source-json",
            str(source),
            "--reuse-source-json",
            "--output-dir",
            str(output),
            "--registry-version",
            "test-registry",
            "--scope-json",
            str(scope),
            "--report-dir",
            str(reports),
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["registry_version"] == "test-registry"
    assert payload["source_json"] == str(source)
    assert payload["manifest"]["qa_status"] == "passed"
    assert (output / "manifest.json").exists()
    assert (output / "flickr_query_definitions.parquet").exists()
    assert Path(payload["report_json"]).exists()
    assert Path(payload["report_md"]).exists()


def test_registry_seed_flickr_queries_cli_loads_query_definitions_into_state(tmp_path, capsys) -> None:
    query_definitions = tmp_path / "flickr_query_definitions.parquet"
    pl.DataFrame(
        [
            {
                "query_definition_id": "q-tags",
                "registry_version": "registry-v1",
                "accepted_taxon_key": "gbif:100",
                "accepted_scientific_name": "Papilio demoleus",
                "family_key": "gbif:10",
                "genus_key": "gbif:90",
                "species_key": "gbif:100",
                "source_term": "Papilio demoleus",
                "language": "la",
                "search_field": "tags",
                "search_priority": 10,
                "normalized_match_key": "papilio demoleus",
                "bbox": "",
                "region": "",
                "name_class": "accepted_scientific",
                "confidence": "high",
                "enabled": True,
            }
        ]
    ).write_parquet(query_definitions)
    state_db = tmp_path / "poller.sqlite"
    parser = build_parser()
    args = parser.parse_args(
        [
            "dev",
            "registry",
            "seed-flickr-queries",
            "--query-definitions",
            str(query_definitions),
            "--state-db",
            str(state_db),
        ]
    )

    assert run(args) == 0
    capsys.readouterr()
    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["query_definitions"] == str(query_definitions)
    assert payload["work_items_seen"] == 1
    assert payload["work_items_inserted"] == 0


def test_registry_audit_cli_summarizes_registry_parquet_with_duckdb(tmp_path, capsys) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    pl.DataFrame(
        [
            {"rank": "SUPERFAMILY", "family": ""},
            {"rank": "FAMILY", "family": "Papilionidae"},
            {"rank": "SPECIES", "family": "Papilionidae"},
        ]
    ).write_parquet(registry / "taxa.parquet")
    pl.DataFrame(
        [
            {"name_class": "accepted_scientific", "source": "GBIF", "language": "la", "enabled": True},
            {"name_class": "vernacular", "source": "GBIF", "language": "eng", "enabled": True},
            {"name_class": "vernacular_alias", "source": "fixture", "language": "en", "enabled": False},
        ]
    ).write_parquet(registry / "names.parquet")
    pl.DataFrame(
        [
            {"search_field": "tags", "enabled": True},
            {"search_field": "text", "enabled": True},
        ]
    ).write_parquet(registry / "flickr_query_definitions.parquet")
    pl.DataFrame(
        [
            {"severity": "warning", "code": "disabled_names_excluded_from_queries", "subject": "1"},
        ]
    ).write_parquet(registry / "qa_findings.parquet")

    parser = build_parser()
    report_dir = tmp_path / "reports"
    args = parser.parse_args(["registry", "audit", "--registry-dir", str(registry), "--report-dir", str(report_dir)])

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["registry_dir"] == str(registry)
    assert payload["taxa_by_rank"] == {"FAMILY": 1, "SPECIES": 1, "SUPERFAMILY": 1}
    assert payload["enabled_names_by_class"] == {"accepted_scientific": 1, "vernacular": 1}
    assert payload["flickr_queries_by_field"] == {"tags": 1, "text": 1}
    assert payload["qa_by_severity"] == {"warning": 1}
    assert payload["language_target_coverage_report"].startswith(str(report_dir))
    assert payload["curated_vernacular_gap_report"].startswith(str(report_dir))


def test_cli_help_does_not_describe_old_gold_silver_bronze_logic(capsys) -> None:
    parser = build_parser()

    parser.print_help()
    help_text = capsys.readouterr().out

    assert "human_verified_bioclip_positive" not in help_text
    assert "human verification" not in help_text.casefold()
    assert "bioclip_positive_without_human_verification" not in help_text


def test_removed_filter_and_apply_rules_commands_no_longer_parse() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["apply-rules", "--evidence", "evidence.parquet", "--output", "classified.parquet"])
    with pytest.raises(SystemExit):
        parser.parse_args(["filter", "--input", "evidence.parquet", "--output", "flagged.parquet"])


def test_low_level_fetch_and_comment_commands_are_dev_only() -> None:
    parser = build_parser()

    removed_commands = (
        ["fetch-comments", "--photo-id", "1"],
        ["build-comment-review-queue", "--input", "classified.parquet"],
        ["review-comments-once"],
        ["apply-comment-review-decisions", "--input", "classified.parquet", "--output", "reviewed.parquet"],
        ["poll-once"],
    )
    for command in removed_commands:
        with pytest.raises(SystemExit):
            parser.parse_args(command)


def test_legacy_local_compaction_and_gc_cache_commands_no_longer_parse() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["compact-parquet", "--input-root", "predictions", "--output", "compacted.parquet"])
    with pytest.raises(SystemExit):
        parser.parse_args(["gc-cache", "--cache-root", "data/cache", "--delete"])


def test_legacy_ad_hoc_report_commands_no_longer_parse() -> None:
    parser = build_parser()

    for command in ("qa-rate-limit", "qa-summary", "export-bucket-views", "report-name-evidence"):
        with pytest.raises(SystemExit):
            parser.parse_args([command])


def test_comments_enrichment_cli(tmp_path, capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(["dev", "comments", "fetch", "--photo-id", "1", "--state-db", str(tmp_path / "comments.sqlite"), "--dry-run"])
    assert run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["implemented"] is True
    assert payload["comment_fetch_scope"] == "selected_candidate_records_only"
    assert payload["photo_ids_requested"] == ["1"]
    assert payload["queued_comment_candidates_added"] == 1


def _write_candidate_expansion_error_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    runtime_python = tmp_path / "runtime" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("# fake python", encoding="utf-8")
    context_path = tmp_path / "species_context.json"
    context_path.write_text(
        json.dumps(
            {
                "scientific_name": "Danaus plexippus",
                "accepted_taxon_key": "gbif:1",
                "canonical_name": "Danaus plexippus",
                "family": "Nymphalidae",
                "genus": "Danaus",
                "family_key": "gbif:f",
                "genus_key": "gbif:g",
                "species_key": "gbif:1",
                "registry_version": "registry-v1",
            }
        ),
        encoding="utf-8",
    )
    input_path = tmp_path / "filtered.parquet"
    detections_path = tmp_path / "detections.parquet"
    pl.DataFrame([{"source": "flickr", "flickr_photo_id": "photo-1", "image_url": "https://example.test/1.jpg"}]).write_parquet(input_path)
    pl.DataFrame([{"source": "flickr", "flickr_photo_id": "photo-1", "detection_id": "det-1", "detection_status": "detected"}]).write_parquet(
        detections_path
    )
    return runtime_python, context_path, input_path, detections_path


def _fake_cloud_config() -> BioMinerConfig:
    return BioMinerConfig(
        storage=StorageConfig(
            backend="s3",
            bucket="biominer",
            prefix="biominer",
            endpoint_url="https://s3.us-east-005.backblazeb2.com",
            access_key_id="key-id",
            secret_access_key="secret-value",
            region="us-east-005",
        ),
        workstore=WorkStoreConfig(
            backend="postgres",
            dsn="postgresql://user:password@example.test:5432/postgres",
        ),
        runtime=RuntimeConfig(worker_id="worker-001"),
    )


class _FakeCloudStorage:
    def __init__(self) -> None:
        self.json_payloads: dict[str, dict[str, Any]] = {}
        self.parquet_payloads: dict[str, pl.DataFrame] = {}

    def write_json(self, uri: str, payload: dict[str, Any]) -> str:
        self.json_payloads[uri] = payload
        return uri

    def read_json(self, uri: str) -> dict[str, Any]:
        return self.json_payloads[uri]

    def delete(self, uri: str) -> bool:
        return self.json_payloads.pop(uri, None) is not None

    def write_parquet_shard(self, uri: str, frame: pl.DataFrame) -> str:
        self.parquet_payloads[uri] = frame
        return uri

    def read_parquet(self, uri: str) -> pl.DataFrame:
        return self.parquet_payloads[uri]

    def scan_parquet(self, uri: str) -> pl.LazyFrame:
        return self.parquet_payloads[uri].lazy()

    def exists(self, uri: str) -> bool:
        return uri in self.parquet_payloads or uri in self.json_payloads


class _FakeCloudWorkStore:
    def __init__(self, *, init_error: Exception | None = None) -> None:
        self.init_error = init_error
        self.schema_initialized = False
        self.items: dict[str, dict[str, Any]] = {}
        self.shards: list[dict[str, Any]] = []

    def init_schema(self) -> None:
        if self.init_error is not None:
            raise self.init_error
        self.schema_initialized = True

    def get_or_create_run(self, **kwargs) -> dict[str, Any]:  # noqa: ANN003
        return {"run_id": kwargs["run_id"], "status": "planned"}

    def enqueue_work(self, job_name, registry_version=None, items=None, *, stage="default") -> int:  # noqa: ANN001, ANN202
        inserted = 0
        for item in items or []:
            payload = dict(item)
            work_key = str(payload.pop("work_key"))
            if work_key in self.items:
                continue
            self.items[work_key] = {
                "work_key": work_key,
                "job_name": job_name,
                "stage": stage,
                "registry_version": registry_version,
                "status": "pending",
                "payload": payload,
            }
            inserted += 1
        return inserted

    def claim_next_batch(self, worker_id, limit=None, **filters) -> list[dict[str, Any]]:  # noqa: ANN001
        claimed: list[dict[str, Any]] = []
        for item in self.items.values():
            if item["status"] != "pending":
                continue
            item["status"] = "claimed"
            item["claimed_by"] = worker_id
            claimed.append(item)
            if len(claimed) == limit:
                break
        return claimed

    def mark_completed(self, work_key, output_uri, checksum, row_count) -> None:  # noqa: ANN001
        self.items[work_key].update(
            {
                "status": "completed",
                "output_uri": output_uri,
                "checksum": checksum,
                "row_count": row_count,
            }
        )

    def completed_keys(self, job_name, registry_version=None, *, stage=None) -> set[str]:  # noqa: ANN001
        return {
            key
            for key, item in self.items.items()
            if item["job_name"] == job_name
            and item["registry_version"] == registry_version
            and item["stage"] == stage
            and item["status"] == "completed"
        }

    def register_shard(self, **kwargs) -> None:  # noqa: ANN003
        self.shards.append(kwargs)

    def list_committed_shards(self, **kwargs) -> list[dict[str, Any]]:  # noqa: ANN003
        return list(self.shards)
