from __future__ import annotations

import json

from biominer.benchmarks.vision_plumbing import (
    run_vision_plumbing_benchmark,
    write_benchmark_taxonomy_store,
)
from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION
from biominer.cli import build_parser, run


def test_vision_plumbing_benchmark_runs_model_free_pipeline(tmp_path) -> None:
    taxonomy = tmp_path / "taxonomy_store"
    write_benchmark_taxonomy_store(taxonomy)
    output = tmp_path / "benchmark"

    result = run_vision_plumbing_benchmark(
        records=12,
        butterfly_rate=0.25,
        detections_per_butterfly=2,
        classification_mode=HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        taxonomy_candidate_table=taxonomy,
        output_dir=output,
    )

    metrics = result.metrics
    assert metrics["benchmark_kind"] == "vision_plumbing_model_free"
    assert metrics["records_generated"] == 12
    assert metrics["images_loaded"] == 12
    assert metrics["detection_rows_written"] == 15
    assert metrics["eligible_butterfly_like_detections"] == 6
    assert metrics["non_butterfly_detections"] == 9
    assert metrics["crops_materialised"] == 6
    assert metrics["crops_scored"] == 6
    assert metrics["score_rows_written"] == 6
    assert metrics["joined_evidence_rows"] == 15
    assert metrics["photo_summary_rows"] == 12
    assert metrics["classification_mode"] == HIERARCHICAL_BUTTERFLY_CLASSIFICATION
    assert metrics["peak_tracemalloc_bytes"] > 0
    assert "detect_objects" in metrics["elapsed_seconds_by_stage"]
    assert "score_crops" in metrics["elapsed_seconds_by_stage"]
    assert result.metrics_path.exists()
    assert result.summary_path.exists()
    assert (output / "object_detections.parquet").exists()
    assert (output / "object_bioclip_scores.parquet").exists()
    assert (output / "object_evidence_joined.parquet").exists()
    assert (output / "photo_evidence_summary.parquet").exists()


def test_dev_vision_benchmark_plumbing_cli_writes_metrics(tmp_path, capsys) -> None:
    output = tmp_path / "plumbing"
    taxonomy = tmp_path / "generated_taxonomy_store"
    parser = build_parser()
    args = parser.parse_args(
        [
            "dev",
            "vision",
            "benchmark-plumbing",
            "--records",
            "8",
            "--butterfly-rate",
            "0.25",
            "--detections-per-butterfly",
            "1",
            "--classification-mode",
            "hierarchical_butterfly_classification",
            "--taxonomy-candidate-table",
            str(taxonomy),
            "--output-dir",
            str(output),
        ]
    )

    assert run(args) == 0

    cli_payload = json.loads(capsys.readouterr().out)
    metrics_path = output / "benchmark_metrics.json"
    summary_path = output / "benchmark_summary.md"
    assert cli_payload["benchmark_metrics"] == str(metrics_path)
    assert cli_payload["benchmark_summary"] == str(summary_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["taxonomy_fixture_created"] is True
    assert metrics["records_generated"] == 8
    assert metrics["eligible_butterfly_like_detections"] == 2
    assert metrics["crops_scored"] == 2
    assert summary_path.exists()
