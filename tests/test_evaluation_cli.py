from __future__ import annotations

import json

import polars as pl

from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION
from biominer.cli import build_parser, run


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
