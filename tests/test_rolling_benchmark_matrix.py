from __future__ import annotations

import json

from biominer.benchmarks.vision_plumbing import rolling_worker_benchmark_variants, run_rolling_worker_benchmark_matrix
from biominer.cli import build_parser, run


def test_rolling_worker_benchmark_variants_cover_required_phase5_matrix() -> None:
    variants = rolling_worker_benchmark_variants()

    assert len(variants) == 72
    assert {variant["yolo_sidecar_transport"] for variant in variants} == {"json_b64", "image_path"}
    assert {variant["accelerator_concurrency"] for variant in variants} == {1, 2}
    assert {variant["bioclip_preprocess_workers"] for variant in variants} == {1, 2, 4}
    assert {variant["bioclip_gate_mode"] for variant in variants} == {"butterfly_like_only", "exclude_hard_negative"}
    assert {variant["vision_batch_rows"] for variant in variants} == {250, 500, 1000}


def test_rolling_worker_benchmark_matrix_writes_metrics(tmp_path) -> None:
    result = run_rolling_worker_benchmark_matrix(records=3, output_dir=tmp_path / "matrix")

    assert result.metrics["benchmark_kind"] == "rolling_vision_worker_model_free_matrix"
    assert result.metrics["variant_count"] == 72
    assert result.metrics["records"] == 3
    assert result.metrics_path.exists()
    assert result.summary_path.exists()
    assert json.loads(result.metrics_path.read_text(encoding="utf-8"))["variant_count"] == 72
    assert result.metrics["variants"][0]["detection_rows_per_image"] == 1.0


def test_dev_vision_benchmark_rolling_matrix_cli_writes_reports(tmp_path, capsys) -> None:
    args = build_parser().parse_args(
        [
            "dev",
            "vision",
            "benchmark-rolling-matrix",
            "--records",
            "2",
            "--output-dir",
            str(tmp_path / "matrix"),
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["variant_count"] == 72
    assert payload["records"] == 2
    assert (tmp_path / "matrix" / "rolling_benchmark_matrix_metrics.json").exists()
    assert (tmp_path / "matrix" / "rolling_benchmark_matrix_summary.md").exists()
