from __future__ import annotations

import json

import polars as pl

from biominer.bioclip.benchmark import build_benchmark_report
from biominer.bioclip.diagnostics import mps_memory_metrics


def test_benchmark_report_schema(tmp_path) -> None:
    input_path = tmp_path / "filtered.parquet"
    species_path = tmp_path / "taxa.parquet"
    output_path = tmp_path / "benchmark.json"
    pl.DataFrame([{"flickr_photo_id": "1"}]).write_parquet(input_path)
    pl.DataFrame([{"scientific_name": "Danaus plexippus", "rank": "species"}]).write_parquet(species_path)

    report = build_benchmark_report(
        input_path=input_path,
        species_candidates_path=species_path,
        output_path=output_path,
        device="mps",
        register_sizes=[8],
        register_counts=[2],
        candidate_limits=[500],
        classification_modes=[],
        candidate_strategy="all",
        download_workers=4,
    )

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["runs"] == report["runs"]


def test_benchmark_report_records_configuration_rows(tmp_path) -> None:
    input_path = tmp_path / "filtered.parquet"
    species_path = tmp_path / "taxa.parquet"
    output_path = tmp_path / "benchmark.json"
    pl.DataFrame([{"flickr_photo_id": "1"}, {"flickr_photo_id": "2"}]).write_parquet(input_path)
    pl.DataFrame([{"scientific_name": "Danaus plexippus", "rank": "species"}]).write_parquet(species_path)

    report = build_benchmark_report(
        input_path=input_path,
        species_candidates_path=species_path,
        output_path=output_path,
        device="mps",
        register_sizes=[8, 16],
        register_counts=[2],
        candidate_limits=[500],
        classification_modes=[__import__("biominer.bioclip.candidate_sets", fromlist=["CandidateMode"]).CandidateMode.TRIAGE],
        candidate_strategy="geo",
        download_workers=4,
    )

    row = report["runs"][0]
    assert len(report["runs"]) == 2
    assert row["rows_in"] == 2
    assert row["classification_mode"] == "triage"
    assert row["candidate_strategy"] == "geo"
    assert row["images_per_second"] == "not_instrumented"


def test_mps_metrics_fallback_when_unavailable() -> None:
    metrics = mps_memory_metrics(FakeTorch(mps_available=False))

    assert metrics["mps_available"] is False
    assert metrics["mps_current_allocated_memory_bytes"] is None
    assert metrics["mps_driver_allocated_memory_bytes"] is None
    assert metrics["mps_recommended_max_memory_bytes"] is None


class FakeTorch:
    def __init__(self, *, mps_available: bool) -> None:
        self.backends = FakeBackends(mps_available)


class FakeBackends:
    def __init__(self, available: bool) -> None:
        self.mps = FakeMps(available)


class FakeMps:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available
