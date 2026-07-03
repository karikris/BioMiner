from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sqlite3
from types import SimpleNamespace

import polars as pl

from biominer.cli import _detect_boxes_backend, build_parser, load_decoded_image_from_record, run
from biominer.detection.detector_base import DetectionCandidate
from biominer.detection.policy import DetectionPolicy, DetectionRunPolicy


def test_cli_exposes_only_lean_pipeline_commands() -> None:
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices  # noqa: SLF001 - parser surface regression test.

    assert "poll-once" in commands
    assert "bioclip" in commands
    assert "species" in commands
    assert "build-papilio-demoleus-query-plan" not in commands
    assert "fetch" not in commands
    assert "fetch-live" not in commands
    assert "benchmark-existing-payloads" not in commands

def test_poll_once_cli_accepts_bounded_cycle_arguments() -> None:
    parser = build_parser()
    args = parser.parse_args(["poll-once", "--max-api-calls", "3500"])

    assert args.command == "poll-once"
    assert args.max_api_calls == 3500


def test_poll_once_cli_accepts_cloud_storage_phase2_arguments() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "poll-once",
            "--run-id",
            "run-1",
            "--worker-id",
            "worker-001",
            "--storage-backend",
            "local",
            "--storage-prefix",
            "staging",
            "--evidence-stage",
            "poll_once",
            "--no-compact",
        ]
    )

    assert args.run_id == "run-1"
    assert args.worker_id == "worker-001"
    assert args.storage_backend == "local"
    assert args.storage_prefix == "staging"
    assert args.evidence_stage == "poll_once"
    assert args.no_compact is True


def test_detect_boxes_cli_accepts_object_detection_arguments() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "detect",
            "boxes",
            "--input",
            "filtered.parquet",
            "--output",
            "object_detections.parquet",
            "--backend",
            "yolo",
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
    species_args = parser.parse_args(
        [
            "species",
            "detect",
            "--input",
            "filtered.parquet",
            "--output",
            "object_detections.parquet",
            "--backend",
            "fake",
            "--profile",
            "mac_m5pro_64gb",
            "--parquet-batch-rows",
            "6",
            "--image-max-side-px",
            "1024",
        ]
    )

    assert args.command == "detect"
    assert args.detect_command == "boxes"
    assert args.input == "filtered.parquet"
    assert args.output == "object_detections.parquet"
    assert args.backend == "yolo"
    assert args.runtime_python == ".venv-vision-py312/bin/python"
    assert args.device == "mps"
    assert args.download_workers == 2
    assert args.max_inflight_images == 7
    assert args.detector_batch_size == 3
    assert args.parquet_batch_rows == 5
    assert args.crop_target_px == 224
    assert args.image_max_side_px == 960
    assert args.crop_padding_ratio == 0.2
    assert species_args.species_command == "detect"
    assert species_args.profile == "mac_m5pro_64gb"
    assert species_args.parquet_batch_rows == 6
    assert species_args.image_max_side_px == 1024


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
            "detect",
            "boxes",
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
            "detect",
            "boxes",
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
    assert pipeline["run_policy"].detector_batch_size == 4
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
            "detect",
            "boxes",
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
            "detect",
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


def test_bioclip_object_cli_accepts_screen_and_ablation_arguments() -> None:
    parser = build_parser()
    screen = parser.parse_args(
        [
            "bioclip",
            "screen-objects",
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
            "--text-embedding-batch-size",
            "2",
            "--candidate-text-embedding-cache",
            "candidate_text_embeddings.parquet",
            "--object-image-embedding-cache",
            "object_image_embeddings.parquet",
        ]
    )
    ablate = parser.parse_args(
        [
            "bioclip",
            "ablate-objects",
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
            "--text-embedding-batch-size",
            "3",
            "--candidate-text-embedding-cache",
            "candidate_text_embeddings.parquet",
            "--object-image-embedding-cache",
            "object_image_embeddings.parquet",
        ]
    )
    species_screen = parser.parse_args(
        [
            "species",
            "bioclip-objects",
            "--context-json",
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
            "--text-embedding-batch-size",
            "4",
            "--candidate-text-embedding-cache",
            "candidate_text_embeddings.parquet",
            "--object-image-embedding-cache",
            "object_image_embeddings.parquet",
        ]
    )
    species_ablate = parser.parse_args(
        [
            "species",
            "ablate-objects",
            "--context-json",
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
            "--text-embedding-batch-size",
            "5",
            "--candidate-text-embedding-cache",
            "candidate_text_embeddings.parquet",
            "--object-image-embedding-cache",
            "object_image_embeddings.parquet",
        ]
    )
    join = parser.parse_args(
        [
            "bioclip",
            "join-object-evidence",
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
            "--species-context",
            "species_context.json",
        ]
    )
    species_join = parser.parse_args(
        [
            "species",
            "join-object-evidence",
            "--context-json",
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

    assert screen.bioclip_command == "screen-objects"
    assert screen.ablation_mode == "detector_crop"
    assert screen.geo_prior_table == "geo_prior.parquet"
    assert screen.parquet_batch_rows == 7
    assert screen.text_embedding_batch_size == 2
    assert screen.candidate_text_embedding_cache == "candidate_text_embeddings.parquet"
    assert screen.object_image_embedding_cache == "object_image_embeddings.parquet"
    assert ablate.bioclip_command == "ablate-objects"
    assert ablate.modes == "whole_image,detector_crop,detector_crop_segmentation"
    assert ablate.geo_prior_table == "geo_prior.parquet"
    assert ablate.parquet_batch_rows == 11
    assert ablate.text_embedding_batch_size == 3
    assert ablate.candidate_text_embedding_cache == "candidate_text_embeddings.parquet"
    assert ablate.object_image_embedding_cache == "object_image_embeddings.parquet"
    assert species_screen.species_command == "bioclip-objects"
    assert species_screen.geo_prior_table == "geo_prior.parquet"
    assert species_screen.parquet_batch_rows == 9
    assert species_screen.text_embedding_batch_size == 4
    assert species_screen.candidate_text_embedding_cache == "candidate_text_embeddings.parquet"
    assert species_screen.object_image_embedding_cache == "object_image_embeddings.parquet"
    assert species_ablate.species_command == "ablate-objects"
    assert species_ablate.geo_prior_table == "geo_prior.parquet"
    assert species_ablate.parquet_batch_rows == 13
    assert species_ablate.text_embedding_batch_size == 5
    assert species_ablate.candidate_text_embedding_cache == "candidate_text_embeddings.parquet"
    assert species_ablate.object_image_embedding_cache == "object_image_embeddings.parquet"
    assert join.bioclip_command == "join-object-evidence"
    assert join.scores == "object_bioclip_scores.parquet"
    assert species_join.species_command == "join-object-evidence"
    assert species_join.context_json == "species_context.json"


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
            stdout='{"device_resolved":"mps","mps_available":true}\n',
            stderr="",
        )

    monkeypatch.setattr("biominer.cli.subprocess.run", fake_run)
    parser = build_parser()
    args = parser.parse_args(
        [
            "bioclip",
            "runtime-check",
            "--runtime-python",
            str(runtime_python),
            "--hf-cache-dir",
            str(tmp_path / "hf"),
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["device_resolved"] == "mps"
    assert calls[0]["cmd"][0] == str(runtime_python)
    assert calls[0]["cmd"][-1] == "auto"
    assert calls[0]["env"]["HF_HOME"] == str((tmp_path / "hf").resolve())


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
            "bioclip",
            "prefetch-model",
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


def test_bioclip_screen_wires_register_runner_with_sidecar_runtime(tmp_path, capsys, monkeypatch) -> None:
    runtime_python = tmp_path / "runtime" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("# fake python", encoding="utf-8")
    input_path = tmp_path / "filtered.parquet"
    candidates_path = tmp_path / "candidates.tsv"
    output_path = tmp_path / "classified.parquet"
    candidates_path.write_text("scientific_name\trank\nPapilio demoleus\tspecies\n", encoding="utf-8")
    pl.DataFrame(
        [
            {
                "source_record_id": "1",
                "flickr_photo_id": "1",
                "image_url": "https://live.staticflickr.com/1.jpg",
                "title": "Papilio demoleus",
            }
        ]
    ).write_parquet(input_path)
    calls: dict[str, object] = {}

    class FakeScorer:
        def __init__(self, *, runtime, hf_cache_dir, device):  # noqa: ANN001 - mirrors scorer init.
            calls["scorer"] = {"runtime": runtime, "hf_cache_dir": hf_cache_dir, "device": device}

        def close(self) -> None:
            calls["closed"] = True

    class FakeClassifier:
        def __init__(self, *, runtime, scorer):  # noqa: ANN001 - mirrors classifier init.
            calls["classifier"] = {"runtime": runtime, "scorer": scorer}

    def fake_process(records, **kwargs):  # noqa: ANN001 - mirrors register runner call.
        calls["records"] = records
        calls["runner_kwargs"] = kwargs
        return SimpleNamespace(
            frame=pl.DataFrame([{"classification_status": "success"}]),
            output_path=Path(kwargs["output_path"]),
            records_seen=1,
            records_classified=1,
            records_skipped_existing=0,
            download_failures=0,
            bioclip_failures=0,
            images_deleted_after_classification=1,
            max_staged_images=1,
            register_count=kwargs["register_count"],
            register_size=kwargs["register_size"],
        )

    monkeypatch.setattr("biominer.cli.PersistentBioClipScorer", FakeScorer)
    monkeypatch.setattr("biominer.cli.BioClipClassifier", FakeClassifier)
    monkeypatch.setattr("biominer.cli.process_records_with_registers", fake_process)
    parser = build_parser()
    args = parser.parse_args(
        [
            "bioclip",
            "screen",
            "--input",
            str(input_path),
            "--species-candidates",
            str(candidates_path),
            "--output",
            str(output_path),
            "--runtime-python",
            str(runtime_python),
            "--device",
            "mps",
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["records_classified"] == 1
    assert payload["register_count"] == 2
    assert payload["register_size"] == 4
    assert calls["closed"] is True
    assert calls["records"][0]["flickr_photo_id"] == "1"
    assert calls["scorer"]["device"] == "mps"
    assert calls["runner_kwargs"]["model_checkpoint"] == "191d741545e4c741cdef4b22c6eb69c945c1e592"


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
            "bioclip",
            "screen-objects",
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
    assert calls["screen"]["geo_prior_table"].height == 1
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
            "bioclip",
            "ablate-objects",
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
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["score_batches_written"] == 2
    assert calls["closed"] is True
    assert calls["ablation"]["parquet_batch_rows"] == 2
    assert calls["ablation"]["modes"] == ("detector_crop",)


def test_bioclip_join_object_evidence_cli_writes_join_tables(tmp_path, capsys, monkeypatch) -> None:
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
            "bioclip",
            "join-object-evidence",
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


def test_species_join_object_evidence_cli_writes_join_tables(tmp_path, capsys, monkeypatch) -> None:
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
            "species",
            "join-object-evidence",
            "--context-json",
            str(context_path),
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
            "detect",
            "boxes",
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


def test_detect_boxes_yolo_backend_uses_lazy_optional_adapter(tmp_path, capsys, monkeypatch) -> None:
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
    calls: dict[str, object] = {}

    class FakeYoloDetector:
        backend = "yolo"
        model_id = "fake-yolo"
        model_version = "test"
        checkpoint = "fake-yolo.pt"

        def __init__(self, *, model_path: str = "yolov8n.pt", device: str = "auto") -> None:
            calls["detector_init"] = {"model_path": model_path, "device": device}

        def detect_batch(self, images):  # noqa: ANN001, ANN201 - mirrors ObjectDetector protocol.
            calls["batch_size"] = len(images)
            return [[DetectionCandidate(label="butterfly", score=0.9, bbox_xyxy=(0, 0, 4, 4), objectness_score=0.9)]]

    monkeypatch.setattr("biominer.detection.yolo_detector.YoloObjectDetector", FakeYoloDetector)
    monkeypatch.setattr("biominer.cli.load_decoded_image_from_record", lambda record: _fake_cli_image(record))
    parser = build_parser()
    args = parser.parse_args(
        [
            "detect",
            "boxes",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--backend",
            "yolo",
            "--device",
            "mps",
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    row = pl.read_parquet(output_path).to_dicts()[0]
    assert calls["detector_init"] == {"model_path": "yolov8n.pt", "device": "mps"}
    assert calls["batch_size"] == 1
    assert payload["backend"] == "yolo"
    assert payload["rows"] == 1
    assert row["detector_backend"] == "yolo"
    assert row["detector_model_id"] == "fake-yolo"
    assert row["detection_status"] == "detected"
    assert row["crop_storage_policy"] == "ephemeral"


def test_detect_boxes_yolo_backend_uses_existing_sidecar_runtime(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "filtered.parquet"
    input_path.write_text("not read by backend selection", encoding="utf-8")
    runtime_python = tmp_path / "vision" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("# fake python", encoding="utf-8")
    calls: dict[str, object] = {}

    class FakeYoloSidecarDetector:
        backend = "yolo"
        model_id = "fake-yolo-sidecar"
        model_version = "sidecar-test"
        checkpoint = "fake-yolo.pt"

        def __init__(self, *, runtime_python: str, model_path: str = "yolov8n.pt", device: str = "auto") -> None:
            calls["sidecar_init"] = {"runtime_python": runtime_python, "model_path": model_path, "device": device}

        def detect_batch(self, images):  # noqa: ANN001, ANN201 - mirrors ObjectDetector protocol.
            return [[] for _image in images]

    class InProcessYoloDetector:
        def __init__(self, **kwargs):  # noqa: ANN003, ANN204 - should not be called.
            raise AssertionError(f"in-process YOLO should not be used when sidecar exists: {kwargs}")

    monkeypatch.setattr("biominer.detection.yolo_detector.YoloObjectDetector", InProcessYoloDetector)
    monkeypatch.setattr("biominer.detection.yolo_detector.YoloSidecarObjectDetector", FakeYoloSidecarDetector)
    parser = build_parser()
    args = parser.parse_args(
        [
            "detect",
            "boxes",
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "object_detections.parquet"),
            "--backend",
            "yolo",
            "--runtime-python",
            str(runtime_python),
            "--device",
            "mps",
        ]
    )

    detector, image_loader = _detect_boxes_backend(args, [])

    assert detector.backend == "yolo"
    assert image_loader is load_decoded_image_from_record
    assert calls["sidecar_init"] == {"runtime_python": str(runtime_python), "model_path": "yolov8n.pt", "device": "mps"}


def _fake_cli_image(record: dict[str, object]):
    from biominer.detection.detector_base import DecodedImage

    width = max(1, int(record.get("image_width") or 1))
    height = max(1, int(record.get("image_height") or 1))
    return DecodedImage(width=width, height=height, mode="RGB", data=b"\x00\x00\x00" * width * height)


def test_species_run_cli_resolves_registry_compiles_queries_and_seeds_work(tmp_path, capsys) -> None:
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
            "species",
            "run",
            "--scientific-name",
            "Papilio demoleus",
            "--registry-dir",
            str(registry),
            "--output-root",
            str(output),
        ]
    )

    assert run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scientific_name"] == "Papilio demoleus"
    assert payload["fetch_status"] == "skipped_missing_api_key"
    assert (output / "species_context.json").exists()
    assert (output / "flickr_query_definitions.parquet").exists()
    with sqlite3.connect(output / "state" / "flickr_poller.sqlite") as conn:
        assert conn.execute("SELECT count(*) FROM flickr_work_items").fetchone()[0] > 0


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
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "registry"
    parser = build_parser()
    args = parser.parse_args(
        [
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
            "registry",
            "seed-flickr-queries",
            "--query-definitions",
            str(query_definitions),
            "--state-db",
            str(state_db),
            "--start-date",
            "2026-01-01",
            "--end-date",
            "2026-01-05",
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
    args = parser.parse_args(["registry", "audit", "--registry-dir", str(registry)])

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["registry_dir"] == str(registry)
    assert payload["taxa_by_rank"] == {"FAMILY": 1, "SPECIES": 1, "SUPERFAMILY": 1}
    assert payload["enabled_names_by_class"] == {"accepted_scientific": 1, "vernacular": 1}
    assert payload["flickr_queries_by_field"] == {"tags": 1, "text": 1}
    assert payload["qa_by_severity"] == {"warning": 1}


def test_cli_help_does_not_describe_old_gold_silver_bronze_logic(capsys) -> None:
    parser = build_parser()

    parser.print_help()
    help_text = capsys.readouterr().out

    assert "human_verified_bioclip_positive" not in help_text
    assert "human verification" not in help_text.casefold()
    assert "bioclip_positive_without_human_verification" not in help_text


def test_qa_rate_limit_outputs_limiter_status_json(tmp_path, capsys) -> None:
    state = tmp_path / "poller.sqlite"
    from biominer.flickr_fetch.metadata_poller import MetadataPollState

    poll_state = MetadataPollState(state)
    poll_state.log_api_call(work_item_id="work-1", endpoint="flickr.photos.search", status="ok")
    parser = build_parser()
    args = parser.parse_args(["qa-rate-limit", "--state-db", str(state)])

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["api_calls_in_window"] == 1
    assert payload["photo_records_in_window"] == "not_instrumented"
    assert payload["soft_api_calls_per_hour"] == 3500
    assert payload["hard_api_calls_per_hour"] == 3600


def test_qa_summary_outputs_report_summary(tmp_path, capsys) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "species": "Papilio demoleus",
                "actual_unique_records": 16,
                "api_calls_made": 0,
                "step_timings_seconds": {"vision_classification": 84.9},
                "storage_artifacts": {"total_artifact_bytes": 1234},
                "memory_artifacts": {"peak_traced_bytes": 4567},
                "compute_artifacts": {"vision_model_loaded": True},
            }
        ),
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(["qa-summary", "--report", str(report_path)])

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["species"] == "Papilio demoleus"
    assert payload["actual_unique_records"] == 16
    assert payload["vision_model_loaded"] is True
    assert payload["total_artifact_bytes"] == 1234


def test_apply_rules_compact_and_gc_cache_cli(tmp_path, capsys) -> None:
    parser = build_parser()
    evidence_path = tmp_path / "evidence.parquet"
    pl.DataFrame(
        {
            "flickr_photo_id": ["1"],
            "image_url": ["https://live.staticflickr.com/large.jpg"],
            "bioclip_top1_label": ["a photo of Papilio demoleus"],
            "bioclip_top1_score": [0.9],
            "bioclip_species_agreement_status": ["exact_species_agreement"],
        }
    ).write_parquet(evidence_path)

    classified_path = tmp_path / "classified.parquet"
    args = parser.parse_args(["apply-rules", "--evidence", str(evidence_path), "--output", str(classified_path)])
    assert run(args) == 0
    rules_payload = json.loads(capsys.readouterr().out)
    assert rules_payload["rows"] == 1
    assert sum(rules_payload["publication_state_counts"].values()) == 1
    assert rules_payload["in_review_without_reason"] == 0

    predictions = tmp_path / "predictions"
    predictions.mkdir()
    pl.DataFrame({"flickr_photo_id": ["1"]}).write_parquet(predictions / "part.parquet")
    compacted_path = tmp_path / "compacted.parquet"
    args = parser.parse_args(["compact-parquet", "--input-root", str(predictions), "--output", str(compacted_path)])
    assert run(args) == 0
    compact_payload = json.loads(capsys.readouterr().out)
    assert compact_payload["input_parquet_files"] == 1
    assert compact_payload["rows"] == 1
    assert compacted_path.exists()


def test_comments_enrichment_cli(tmp_path, capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(["fetch-comments", "--photo-id", "1", "--state-db", str(tmp_path / "comments.sqlite"), "--dry-run"])
    assert run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["implemented"] is True
    assert payload["comment_fetch_scope"] == "selected_candidate_records_only"
    assert payload["photo_ids_requested"] == ["1"]
    assert payload["queued_comment_candidates_added"] == 1


def test_gc_cache_reports_deleted_files(tmp_path, capsys) -> None:
    parser = build_parser()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "image.jpg").write_bytes(b"abc")
    args = parser.parse_args(["gc-cache", "--cache-root", str(cache_root), "--delete"])
    assert run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["files_seen"] == 1
    assert payload["deleted_files"] == 1


def test_export_bucket_views_cli_writes_derived_parquet_files(tmp_path, capsys) -> None:
    input_path = tmp_path / "bucketed_records.parquet"
    output_dir = tmp_path / "views"
    pl.DataFrame(
        [
            {"flickr_photo_id": "1", "occurrence_bin": "gold"},
            {"flickr_photo_id": "2", "occurrence_bin": "silver"},
            {"flickr_photo_id": "3", "occurrence_bin": "bronze"},
            {"flickr_photo_id": "4", "occurrence_bin": "bin"},
        ]
    ).write_parquet(input_path)

    assert run(build_parser().parse_args(["export-bucket-views", "--input", str(input_path), "--output-dir", str(output_dir)])) == 0
    payload = json.loads(capsys.readouterr().out)

    assert set(payload) == {"gold", "silver", "bronze", "bin"}
    assert (output_dir / "gold_records.parquet").exists()
    assert (output_dir / "silver_records.parquet").exists()
    assert (output_dir / "bronze_records.parquet").exists()
    assert (output_dir / "bin_records.parquet").exists()
