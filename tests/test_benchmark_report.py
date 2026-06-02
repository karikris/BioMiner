from __future__ import annotations

import json

from flickr_bio_occurrence.benchmark.mock_run import run_mock_1000_record_benchmark


def test_mock_1000_record_benchmark_writes_metrics_artifact(tmp_path) -> None:
    report_path = run_mock_1000_record_benchmark(output_dir=tmp_path, species="Papilio demoleus", region="Australia")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["species"] == "Papilio demoleus"
    assert report["region"] == "Australia"
    assert report["record_count"] == 1000
    assert set(report["step_timings_seconds"]) == {
        "mock_metadata_generation",
        "bronze_flattening",
        "silver_candidate_build",
        "dwc_mapping",
        "artifact_write",
    }
    assert report["storage_artifacts"]["metrics_json"].endswith("mock_1000_record_benchmark.json")
    assert report["storage_artifacts"]["bronze_parquet_files"] >= 1
    assert report["storage_artifacts"]["silver_parquet_files"] >= 1
    assert report["storage_artifacts"]["gold_parquet_files"] >= 1
    assert report["storage_artifacts"]["duckdb_path"].endswith("mock_1000_record_benchmark.duckdb")
    assert report["memory_artifacts"]["peak_traced_bytes"] > 0
    assert report["compute_artifacts"]["worker_count"] >= 1
