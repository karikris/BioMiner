from __future__ import annotations

import json
from types import SimpleNamespace

import polars as pl

from biominer.bioclip.benchmark import build_benchmark_report
from biominer.bioclip.candidate_sets import CandidateMode
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
    assert output_path.with_suffix(output_path.suffix + ".runs.parquet").exists()


def test_benchmark_real_run_records_throughput_and_distributions(tmp_path) -> None:
    input_path = tmp_path / "filtered.parquet"
    species_path = tmp_path / "taxa.parquet"
    output_path = tmp_path / "benchmark.json"
    pl.DataFrame([{"flickr_photo_id": "1"}, {"flickr_photo_id": "2"}]).write_parquet(input_path)
    pl.DataFrame([{"scientific_name": "Danaus plexippus", "rank": "species"}]).write_parquet(species_path)

    def fake_screen_runner(config):  # noqa: ANN001 - benchmark callback shape.
        return SimpleNamespace(
            frame=pl.DataFrame(
                [
                    {
                        "occurrence_bin": "gold",
                        "image_category": "adult_butterfly",
                        "life_stage": "adult_butterfly",
                        "species_top1_score": 0.91,
                        "species_top1_top2_margin": 0.2,
                        "species_entropy": 0.1,
                    },
                    {
                        "occurrence_bin": "in_review",
                        "image_category": "unknown",
                        "life_stage": "unknown",
                        "species_top1_score": 0.42,
                        "species_top1_top2_margin": 0.03,
                        "species_entropy": 1.1,
                    },
                ]
            ),
            records_classified=2,
            download_failures=1,
            bioclip_failures=0,
        )

    report = build_benchmark_report(
        input_path=input_path,
        species_candidates_path=species_path,
        output_path=output_path,
        device="cpu",
        register_sizes=[4],
        register_counts=[1],
        candidate_limits=[10],
        classification_modes=[CandidateMode.HYBRID],
        candidate_strategy="all",
        download_workers=1,
        dry_run=False,
        screen_runner=fake_screen_runner,
    )

    row = report["runs"][0]
    assert row["rows_out"] == 2
    assert row["images_classified"] == 2
    assert isinstance(row["images_per_second"], float)
    assert row["download_failure_count"] == 1
    assert row["bucket_counts"] == {"gold": 1, "in_review": 1}
    assert row["species_top1_score_distribution"]["count"] == 2
    assert output_path.with_suffix(output_path.suffix + ".runs.parquet").exists()


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
