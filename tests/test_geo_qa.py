from __future__ import annotations

import json

import polars as pl

from biominer.cli import build_parser, run
from biominer.geo.qa import build_geo_qa_report


def test_geo_qa_report_reads_parquet_with_duckdb_and_writes_reports(tmp_path) -> None:
    classified = tmp_path / "classified.parquet"
    geo_index = tmp_path / "geo_species_index.parquet"
    benchmark = tmp_path / "benchmark.json"
    pl.DataFrame(
        [
            {
                "source_record_id": "1",
                "candidate_set_signature": "c1",
                "occurrence_bin": "gold",
                "latitude": -27.0,
                "longitude": 153.0,
                "species_candidate_count": 2,
                "geo_candidate_cell_id": "cell-a",
                "geo_candidate_grid_level": "G4_5deg",
                "geo_candidate_fallback_level": None,
            },
            {
                "source_record_id": "2",
                "candidate_set_signature": "c1",
                "occurrence_bin": "bronze",
                "latitude": -27.1,
                "longitude": 153.1,
                "species_candidate_count": 0,
                "geo_candidate_cell_id": "cell-a",
                "geo_candidate_grid_level": "G4_5deg",
                "geo_candidate_fallback_level": "local_cell_below_min_species",
            },
            {
                "source_record_id": "3",
                "candidate_set_signature": "c2",
                "occurrence_bin": "in_review",
                "latitude": None,
                "longitude": None,
                "species_candidate_count": None,
                "geo_candidate_cell_id": None,
                "geo_candidate_grid_level": None,
                "geo_candidate_fallback_level": None,
            },
        ]
    ).write_parquet(classified)
    pl.DataFrame(
        [
            {"grid_level": "G4_5deg", "geocell_id": "cell-a", "species_key": "1", "scientific_name": "Danaus plexippus"},
            {"grid_level": "G4_5deg", "geocell_id": "cell-a", "species_key": "2", "scientific_name": "Papilio machaon"},
            {"grid_level": "G0_world", "geocell_id": "G0_world:global", "species_key": "1", "scientific_name": "Danaus plexippus"},
        ]
    ).write_parquet(geo_index)
    benchmark.write_text(json.dumps({"run_id": "bench-1", "runs": [{"run_id": "bench-1"}]}), encoding="utf-8")

    report = build_geo_qa_report(
        classified_path=classified,
        geo_candidates_path=geo_index,
        benchmark_json=benchmark,
        output_dir=tmp_path / "reports",
        report_name="geo-test",
    )

    assert report["classified_rows"] == 3
    assert report["geo_candidate_coverage"]["rows_with_geo"] == 2
    assert report["candidate_set_count"] == 2
    assert report["avg_records_per_candidate_set"] == 1.5
    assert report["max_records_per_candidate_set"] == 2
    assert report["candidate_set_count_distribution"]["max"] == 2
    assert report["empty_suspicious_geo_cells"][0]["geocell_id"] == "cell-a"
    assert report["benchmark_summary"]["run_id"] == "bench-1"
    assert (tmp_path / "reports" / "geo-test.json").exists()
    assert (tmp_path / "reports" / "geo-test.md").read_text(encoding="utf-8").startswith("# Geo QA Report")


def test_geo_qa_cli_writes_report(tmp_path, capsys) -> None:
    classified = tmp_path / "classified.parquet"
    geo_index = tmp_path / "geo_species_index.parquet"
    pl.DataFrame([{"candidate_set_signature": "c1", "occurrence_bin": "gold"}]).write_parquet(classified)
    pl.DataFrame([{"grid_level": "G0_world", "geocell_id": "G0_world:global", "species_key": "1"}]).write_parquet(geo_index)
    parser = build_parser()
    args = parser.parse_args(
        [
            "geo",
            "qa",
            "--classified",
            str(classified),
            "--geo-candidates",
            str(geo_index),
            "--output-dir",
            str(tmp_path / "reports"),
            "--report-name",
            "geo-cli",
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["classified_rows"] == 1
    assert payload["geo_candidate_index_rows"] == 1
    assert (tmp_path / "reports" / "geo-cli.json").exists()
