from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from biominer.config import BioMinerConfig, RuntimeConfig, StorageConfig, WorkStoreConfig
from biominer.cli import (
    _create_production_vision_runtime,
    _production_vision_settings_from_args,
    build_parser,
    run,
)
from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION, TARGET_SCOPE_OBJECT_SCREENING
from biominer.detection.detector_base import DecodedImage
from biominer.detection.policy import VisionRuntimeSettings
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
    assert "vision" not in commands
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


def test_production_vision_runtime_wires_persistent_sidecars(monkeypatch) -> None:
    created: dict[str, Any] = {}

    class FakeDetector:
        def __init__(self, **kwargs):  # noqa: ANN003
            created["detector"] = kwargs

        def close(self) -> None:
            created["detector_closed"] = True

    class FakePersistent:
        def __init__(self, **kwargs):  # noqa: ANN003
            created["persistent"] = kwargs

        def close(self) -> None:
            created["persistent_closed"] = True

    class FakeCropScorer:
        def __init__(self, **kwargs):  # noqa: ANN003
            created["crop_scorer"] = kwargs

    runtime = SimpleNamespace(
        model=SimpleNamespace(model_name="imageomics/bioclip", checkpoint="revision-1"),
        package_version="3.3.0",
    )
    monkeypatch.setattr("biominer.detection.yoloe26_detector.YoloE26SidecarObjectDetector", FakeDetector)
    monkeypatch.setattr("biominer.bioclip.bioclip.PersistentBioClipScorer", FakePersistent)
    monkeypatch.setattr("biominer.bioclip.object_runner.EphemeralCropBioClipScorer", FakeCropScorer)
    monkeypatch.setattr("biominer.cli._bioclip_runtime", lambda **_kwargs: runtime)

    detector, image_loader, scorer, resources = _create_production_vision_runtime(VisionRuntimeSettings())

    assert detector is not None
    assert callable(image_loader)
    assert scorer is not None
    assert resources == [created["crop_scorer"]["scorer"], detector]
    assert created["detector"]["runtime_python"]
    assert created["persistent"]["device"] == "auto"
    assert created["crop_scorer"]["model_checkpoint"] == "revision-1"


def test_build_text_embedding_cache_command_writes_and_closes_worker(tmp_path, monkeypatch, capsys) -> None:
    runtime_python = tmp_path / "python"
    runtime_python.touch()
    closed: list[bool] = []
    runtime = SimpleNamespace(
        model=SimpleNamespace(model_name="imageomics/bioclip", checkpoint="revision-1"),
        package_version="3.3.0",
    )

    class FakePersistent:
        def __init__(self, **_kwargs):
            self.runtime = runtime

        def embed_text_labels(self, labels):  # noqa: ANN001, ANN201
            return [[1.0, 0.0] for _label in labels]

        def close(self) -> None:
            closed.append(True)

    store = SimpleNamespace(taxonomy_fingerprint="sha256:taxonomy")
    frame = pl.DataFrame([{"embedding_cache_fingerprint": "sha256:cache"}])
    monkeypatch.setattr("biominer.bioclip.bioclip.PersistentBioClipScorer", FakePersistent)
    monkeypatch.setattr("biominer.bioclip.five_rank_store.FiveRankTaxonomyStore.read", lambda _path: store)
    monkeypatch.setattr("biominer.bioclip.five_rank_embedding_cache.build_five_rank_text_embedding_cache", lambda *_args, **_kwargs: frame)
    output = tmp_path / "text-embeddings.parquet"
    args = build_parser().parse_args(
        [
            "dev",
            "vision",
            "build-text-embedding-cache",
            "--taxonomy-candidate-table",
            str(tmp_path / "taxonomy"),
            "--output",
            str(output),
            "--runtime-python",
            str(runtime_python),
        ]
    )

    assert run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert output.exists()
    assert payload["embedding_cache_fingerprint"] == "sha256:cache"
    assert closed == [True]


def test_registry_public_cli_exposes_only_build_and_audit() -> None:
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices  # noqa: SLF001 - parser surface regression test.
    registry_choices = commands["registry"]._subparsers._group_actions[0].choices  # noqa: SLF001
    dev_choices = commands["dev"]._subparsers._group_actions[0].choices  # noqa: SLF001
    dev_registry_choices = dev_choices["registry"]._subparsers._group_actions[0].choices  # noqa: SLF001

    assert set(registry_choices) == {"build", "build-classification", "audit"}
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
    assert settings.adaptive_batching is False
    assert settings.yolo_sidecar_transport == "json_b64"

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
            "--adaptive-batching",
            "--yolo-sidecar-transport",
            "image_path",
            "--no-delete-images-after-commit",
        ]
    )

    overridden_settings = _production_vision_settings_from_args(overridden)

    assert overridden_settings.device == "cpu"
    assert overridden_settings.detector_batch_size == 7
    assert overridden_settings.bioclip_model == "hf-hub:imageomics/bioclip-2.5-vith14"
    assert overridden_settings.adaptive_batching is True
    assert overridden_settings.yolo_sidecar_transport == "image_path"
    assert overridden_settings.delete_images_after_commit is False
    assert overridden_settings.yolo_imgsz == 768
    assert overridden_settings.crop_padding_ratio == 0.08

    invalid = parser.parse_args(
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
            "--yolo-batch",
            "0",
        ]
    )
    with pytest.raises(ValueError, match="detector_batch_size"):
        _production_vision_settings_from_args(invalid)


def test_run_cli_parses_classification_mode_and_top_k_controls() -> None:
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
            "--classification-mode",
            "hierarchical",
            "--taxonomy-candidate-table",
            "s3://biominer/biominer/registry/current",
            "--family-top-k",
            "4",
            "--species-first-pass-top-k",
            "25",
            "--species-rerank-top-k",
            "7",
        ]
    )

    assert args.classification_mode == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
    assert args.taxonomy_candidate_table.endswith("registry/current")
    assert args.family_top_k == 4
    assert args.species_first_pass_top_k == 25
    assert args.species_rerank_top_k == 7

    default_args = parser.parse_args(
        [
            "run",
            "--taxon",
            "Papilio demoleus",
            "--registry-dir",
            "s3://biominer/biominer/registry/current",
            "--output-prefix",
            "s3://biominer/biominer/runs/papilio_demoleus",
        ]
    )

    assert default_args.classification_mode == TARGET_SCOPE_OBJECT_SCREENING
    assert default_args.family_top_k == 3
    assert default_args.species_first_pass_top_k == 20
    assert default_args.species_rerank_top_k == 5


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


def _parquet_column_compressions(source: Path) -> set[str]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(source)
    compressions: set[str] = set()
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            compressions.add(str(row_group.column(column_index).compression).upper())
    return compressions


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


def test_yoloe26_runtime_commands_parse_with_applications_defaults() -> None:
    parser = build_parser()

    runtime = parser.parse_args(["dev", "vision", "yoloe26-runtime-check", "--device", "mps"])
    prefetch = parser.parse_args(["dev", "vision", "yoloe26-prefetch", "--checkpoint", "yoloe-26s-seg.pt"])
    smoke = parser.parse_args(["dev", "vision", "yoloe26-smoke", "--image", "manual.jpg"])

    assert runtime.vision_command == "yoloe26-runtime-check"
    assert runtime.dev_command == "vision"
    assert runtime.runtime_python.endswith("/YOLO26/venv/bin/python")
    assert prefetch.checkpoint == "yoloe-26s-seg.pt"
    assert smoke.image == "manual.jpg"
    with pytest.raises(SystemExit):
        parser.parse_args(["dev", "vision", "yoloe26-prototype-run"])


def test_public_vision_surface_excludes_debug_runtime_commands() -> None:
    parser = build_parser()

    for command in (
        "bioclip-runtime-check",
        "bioclip-prefetch-model",
        "yoloe26-runtime-check",
        "yoloe26-prefetch",
        "yoloe26-smoke",
        "yoloe26-prototype-run",
        "benchmark-plumbing",
        "benchmark-live-m5pro",
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


def _fake_cli_image(record: dict[str, object]):

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
    assert payload["request"]["vision_worker"] == "local_dev"
    assert payload["request"]["worker_id"] == "local"
    assert manifest["taxon_scope"]["accepted_taxon_key"] == "gbif:100"
    assert manifest["taxon_scope"]["accepted_rank"] == "species"
    assert manifest["model_configs"]["vision_worker"] == "local_dev"
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
            "--skip-classification",
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


def test_registry_build_cli_skip_classification_omits_artifacts(tmp_path, capsys) -> None:
    scope = tmp_path / "scope.json"
    scope.write_text(
        json.dumps({"scope_id": "test-scope", "root": {"scientific_name": "Papilionoidea", "rank": "SUPERFAMILY"}, "included_families": []}),
        encoding="utf-8",
    )
    source = tmp_path / "registry_source.json"
    source.write_text(
        json.dumps(
            {
                "source": "GBIF",
                "source_version": "gbif-species-api",
                "retrieved_at": "2026-06-20T00:00:00+00:00",
                "taxa": [{"accepted_taxon_key": "gbif:1", "scientific_name": "Papilionoidea", "rank": "SUPERFAMILY"}],
                "names": [],
                "source_assertions": [],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "registry"
    parser = build_parser()

    assert run(
        parser.parse_args(
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
                "--skip-classification",
            ]
        )
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest"]["classification_skipped"] is True
    assert not (output / "classification_nodes.parquet").exists()
    assert not (output / "classification_edges.parquet").exists()
    assert not (output / "classification_leaf_paths.parquet").exists()


def test_registry_build_classification_cli_writes_v3_outputs(tmp_path, capsys) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    pl.DataFrame(
        [
            {
                "registry_schema_version": "registry-foundation-v2",
                "scope_id": "scope",
                "accepted_taxon_key": "gbif:1938069",
                "scientific_name": "Papilio demoleus",
                "rank": "SPECIES",
                "parent_key": "gbif:90",
                "family_key": "gbif:9417",
                "family": "Papilionidae",
                "genus_key": "gbif:90",
                "genus": "Papilio",
                "species_key": "gbif:1938069",
                "species": "Papilio demoleus",
                "taxonomic_status": "ACCEPTED",
                "in_scope": True,
            },
            {
                "registry_schema_version": "registry-foundation-v2",
                "scope_id": "scope",
                "accepted_taxon_key": "gbif:101",
                "scientific_name": "Papilio machaon",
                "rank": "SPECIES",
                "parent_key": "gbif:90",
                "family_key": "gbif:9417",
                "family": "Papilionidae",
                "genus_key": "gbif:90",
                "genus": "Papilio",
                "species_key": "gbif:101",
                "species": "Papilio machaon",
                "in_scope": True,
            },
        ]
    ).write_parquet(registry / "taxa.parquet")
    (registry / "manifest.json").write_text(json.dumps({"registry_version": "registry-v1", "qa_status": "passed"}), encoding="utf-8")
    parser = build_parser()

    assert run(parser.parse_args(["registry", "build-classification", "--registry-dir", str(registry)])) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["classification_version"] == "butterfly-classification-v3.0.0"
    assert payload["enabled_leaf_path_count"] == 1
    assert payload["prompt_label_count"] == 5
    assert payload["reviewed_rank_skip_count"] == 1
    assert (registry / "classification_nodes.parquet").exists()
    assert (registry / "classification_edges.parquet").exists()
    assert (registry / "classification_leaf_paths.parquet").exists()
    assert (registry / "classification_manifest.json").exists()


def test_registry_build_classification_cli_reports_missing_taxa(tmp_path, capsys) -> None:
    parser = build_parser()

    assert run(parser.parse_args(["registry", "build-classification", "--registry-dir", str(tmp_path)])) == 2

    payload = json.loads(capsys.readouterr().out)
    assert "taxa.parquet" in payload["error"]


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


@pytest.mark.parametrize(
    "command",
    (
        ["apply-rules", "--evidence", "evidence.parquet", "--output", "classified.parquet"],
        ["filter", "--input", "evidence.parquet", "--output", "flagged.parquet"],
        ["fetch-comments", "--photo-id", "1"],
        ["build-comment-review-queue", "--input", "classified.parquet"],
        ["review-comments-once"],
        ["apply-comment-review-decisions", "--input", "classified.parquet", "--output", "reviewed.parquet"],
        ["poll-once"],
        ["compact-parquet", "--input-root", "predictions", "--output", "compacted.parquet"],
        ["gc-cache", "--cache-root", "data/cache", "--delete"],
        ["qa-rate-limit"],
        ["qa-summary"],
        ["export-bucket-views"],
        ["report-name-evidence"],
    ),
)
def test_removed_top_level_commands_no_longer_parse(command: list[str]) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(command)


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
