from __future__ import annotations

import json

import polars as pl

from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION
from biominer.cli import build_parser, run
from biominer.config import BioMinerConfig, RuntimeConfig, StorageConfig, WorkStoreConfig


def test_evaluation_classify_parser_accepts_object_scores_command() -> None:
    args = build_parser().parse_args(
        [
            "evaluation",
            "classify",
            "--object-scores",
            "runs/example/object_bioclip_scores.parquet",
            "--reviewed-labels",
            "tests/fixtures/evaluation/reviewed_labels_valid.jsonl",
            "--output-dir",
            "reports/evaluation/example",
        ]
    )

    assert args.command == "evaluation"
    assert args.evaluation_command == "classify"
    assert args.object_scores == "runs/example/object_bioclip_scores.parquet"
    assert args.object_evidence is None
    assert args.storage_backend == "local"


def test_evaluation_classify_missing_input_path_fails_clearly(tmp_path, capsys) -> None:
    labels = tmp_path / "labels.parquet"
    pl.DataFrame([_label()]).write_parquet(labels)
    args = build_parser().parse_args(
        [
            "evaluation",
            "classify",
            "--object-scores",
            str(tmp_path / "missing.parquet"),
            "--reviewed-labels",
            str(labels),
            "--output-dir",
            str(tmp_path / "report"),
        ]
    )

    assert run(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert "object_scores path does not exist" in payload["error"]


def test_evaluation_classify_cloud_uri_with_local_backend_fails_clearly(tmp_path, capsys) -> None:
    labels = tmp_path / "labels.parquet"
    pl.DataFrame([_label()]).write_parquet(labels)
    args = build_parser().parse_args(
        [
            "evaluation",
            "classify",
            "--object-scores",
            "s3://biominer/evaluation/object_scores.parquet",
            "--reviewed-labels",
            str(labels),
            "--output-dir",
            str(tmp_path / "report"),
        ]
    )

    assert run(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert "use --storage-backend s3" in payload["error"]


def test_evaluation_classify_s3_without_config_fails_clearly(monkeypatch, capsys) -> None:
    for key in (
        "BIOMINER_S3_ENDPOINT_URL",
        "BIOMINER_S3_ACCESS_KEY_ID",
        "BIOMINER_S3_SECRET_ACCESS_KEY",
        "BIOMINER_S3_REGION",
        "BIOMINER_S3_BUCKET",
        "BIOMINER_S3_PREFIX",
    ):
        monkeypatch.delenv(key, raising=False)
    args = build_parser().parse_args(
        [
            "evaluation",
            "classify",
            "--object-scores",
            "s3://biominer/evaluation/object_scores.parquet",
            "--reviewed-labels",
            "s3://biominer/evaluation/reviewed_labels.parquet",
            "--output-dir",
            "s3://biominer/evaluation/report",
            "--storage-backend",
            "s3",
        ]
    )

    assert run(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert "BIOMINER_S3_BUCKET" in payload["error"]


def test_evaluation_classify_command_writes_report_from_object_evidence(tmp_path, capsys) -> None:
    object_evidence = tmp_path / "object_evidence_joined.parquet"
    labels = tmp_path / "reviewed_labels.parquet"
    output = tmp_path / "evaluation"
    pl.DataFrame([_prediction()]).write_parquet(object_evidence)
    pl.DataFrame([_label()]).write_parquet(labels)
    args = build_parser().parse_args(
        [
            "evaluation",
            "classify",
            "--object-evidence",
            str(object_evidence),
            "--reviewed-labels",
            str(labels),
            "--output-dir",
            str(output),
        ]
    )

    assert run(args) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "complete"
    assert payload["input_kind"] == "object_evidence"
    assert payload["metrics"]["species_top1_accuracy"] == 1.0
    assert payload["metrics"]["species_top20_recall"] == 1.0
    assert payload["label_validation"]["finding_count"] == 0
    assert (output / "evaluation_metrics.json").exists()
    assert (output / "evaluation_summary.md").exists()
    assert (output / "family_confusion_matrix.parquet").exists()
    assert (output / "species_confusion_matrix.parquet").exists()


def test_evaluation_classify_command_writes_report_to_s3_storage(monkeypatch, capsys) -> None:
    storage = _MemoryStorage()
    object_scores = "s3://biominer/evaluation/object_scores.parquet"
    reviewed_labels = "s3://biominer/evaluation/reviewed_labels.parquet"
    output_dir = "s3://biominer/evaluation/report"
    storage.parquet_payloads[object_scores] = pl.DataFrame([_prediction()])
    storage.parquet_payloads[reviewed_labels] = pl.DataFrame([_label()])
    monkeypatch.setattr("biominer.cli.load_biominer_config", lambda path: _fake_cloud_config())
    monkeypatch.setattr("biominer.cli.create_storage_backend", lambda storage_config: storage)
    args = build_parser().parse_args(
        [
            "evaluation",
            "classify",
            "--object-scores",
            object_scores,
            "--reviewed-labels",
            reviewed_labels,
            "--output-dir",
            output_dir,
            "--storage-backend",
            "s3",
            "--config",
            "config/production.toml",
        ]
    )

    assert run(args) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "complete"
    assert payload["storage_backend"] == "s3"
    assert payload["input_kind"] == "object_scores"
    assert payload["paths"]["metrics"] == f"{output_dir}/evaluation_metrics.json"
    assert storage.json_payloads[payload["paths"]["metrics"]]["metrics"]["species_top1_accuracy"] == 1.0
    assert "Family top1 accuracy" in storage.text_payloads[payload["paths"]["summary"]]
    assert payload["paths"]["family_confusion_matrix"] in storage.parquet_payloads


def _prediction() -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": "1",
        "detection_id": "d1",
        "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        "family_top3": ["Papilionidae", "Nymphalidae", "Pieridae"],
        "family_top3_accepted_taxon_keys": ["gbif:9417", "gbif:7017", "gbif:5481"],
        "selected_family": "Papilionidae",
        "selected_family_key": "gbif:9417",
        "species_top1_scientific_name": "Papilio demoleus",
        "species_top1_accepted_taxon_key": "gbif:100",
        "accepted_taxon_key": "gbif:100",
        "species_top1_score": 0.91,
        "species_top5": ["Papilio demoleus", "Papilio machaon"],
        "species_top5_accepted_taxon_keys": ["gbif:100", "gbif:200"],
        "species_top20": ["Papilio demoleus", "Papilio machaon"],
        "species_top20_accepted_taxon_keys": ["gbif:100", "gbif:200"],
    }


def _label() -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": "1",
        "detection_id": "d1",
        "crop_hash": "sha256:d1",
        "label_level": "species",
        "is_butterfly": True,
        "accepted_taxon_key": "gbif:100",
        "scientific_name": "Papilio demoleus",
        "family_key": "gbif:9417",
        "family": "Papilionidae",
        "genus_key": "gbif:90",
        "genus": "Papilio",
        "label_source": "fixture",
        "reviewer_id": "reviewer-a",
        "reviewed_at": "2026-07-10T00:00:00Z",
        "review_confidence": "high",
        "review_notes": "synthetic",
    }


def _fake_cloud_config() -> BioMinerConfig:
    return BioMinerConfig(
        storage=StorageConfig(
            backend="s3",
            bucket="biominer",
            prefix="evaluation",
            endpoint_url="https://s3.example.test",
            access_key_id="key-id",
            secret_access_key="secret-value",
            region="us-east-1",
        ),
        workstore=WorkStoreConfig(backend="sqlite", dsn_env=None),
        runtime=RuntimeConfig(worker_id="test"),
    )


class _MemoryStorage:
    def __init__(self) -> None:
        self.parquet_payloads: dict[str, pl.DataFrame] = {}
        self.json_payloads: dict[str, dict[str, object]] = {}
        self.text_payloads: dict[str, str] = {}

    def read_parquet(self, uri: str) -> pl.DataFrame:
        return self.parquet_payloads[uri]

    def write_parquet_shard(self, uri: str, frame: pl.DataFrame) -> str:
        self.parquet_payloads[uri] = frame
        return uri

    def read_json(self, uri: str) -> dict[str, object]:
        return self.json_payloads[uri]

    def write_json(self, uri: str, payload: dict[str, object]) -> str:
        self.json_payloads[uri] = payload
        return uri

    def read_text(self, uri: str, *, encoding: str = "utf-8") -> str:
        return self.text_payloads[uri]

    def write_text(self, uri: str, text: str, *, encoding: str = "utf-8") -> str:
        self.text_payloads[uri] = text
        return uri

    def exists(self, uri: str) -> bool:
        return uri in self.parquet_payloads or uri in self.json_payloads or uri in self.text_payloads
