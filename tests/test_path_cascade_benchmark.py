from __future__ import annotations

import json

from biominer.benchmarks.path_cascade import (
    BENCHMARK_FIXTURE_NOTICE,
    BENCHMARK_KIND,
    BENCHMARK_SELECTED_GENUS_NODE_IDS,
    BENCHMARK_VERSION,
    run_path_cascade_benchmark,
)
from biominer.cli import build_parser, run


def test_path_cascade_benchmark_emits_global_beam_telemetry(tmp_path) -> None:
    result = run_path_cascade_benchmark(output_dir=tmp_path / "cascade")
    metrics = result.metrics

    assert metrics["benchmark_kind"] == BENCHMARK_KIND
    assert metrics["benchmark_version"] == BENCHMARK_VERSION
    assert metrics["status"] == "ok"
    assert metrics["benchmark_fixture"] is True
    assert metrics["authoritative_taxonomy"] is False
    assert metrics["gbif_authority"] is False
    assert metrics["fixture_notice"] == BENCHMARK_FIXTURE_NOTICE
    assert metrics["beam_strategy"] == "global_rank_top_k"
    assert metrics["rank_beam_width"] == 3
    assert metrics["family_candidate_count"] == 7

    steps = metrics["rank_steps"]
    assert [step["rank"] for step in steps] == [
        "FAMILY",
        "SUBFAMILY",
        "TRIBE",
        "SUBTRIBE",
        "GENUS",
        "SPECIES",
    ]
    assert all(step["candidate_count"] >= step["retained_node_count"] for step in steps)
    assert all(step["labels_scored"] == step["unique_labels_scored"] for step in steps)
    assert all(
        step["retained_node_count"] <= 3
        for step in steps
        if step["rank"] != "SPECIES"
    )
    assert metrics["species_candidates_beneath_genus_top3"] == 75
    assert metrics["genus_top3_node_ids"] == list(BENCHMARK_SELECTED_GENUS_NODE_IDS)
    assert len(metrics["species_candidate_node_ids_beneath_genus_top3"]) == 75
    assert set(metrics["species_candidate_counts_by_genus"].values()) == {25}
    assert metrics["species_first_pass_candidate_count"] == 75
    assert metrics["species_first_pass_retained_count"] == 20
    assert metrics["species_rerank_candidate_count"] == 20
    assert metrics["species_rerank_retained_count"] == 5
    assert metrics["reported_species_count"] == 3
    assert all(
        set(step["retained_node_ids"]).isdisjoint(step["pruned_node_ids"])
        for step in steps
    )
    assert metrics["species_rerank_step"]["labels_scored"] == 40
    assert metrics["unique_labels_scored"] == sum(
        step["unique_labels_scored"] for step in steps
    ) + metrics["species_rerank_step"]["unique_labels_scored"]
    assert metrics["elapsed_seconds"] >= 0
    assert all(value >= 0 for value in metrics["elapsed_seconds_by_stage"].values())

    assert result.metrics_path.exists()
    assert result.summary_path.exists()
    assert json.loads(result.metrics_path.read_text(encoding="utf-8")) == metrics
    summary = result.summary_path.read_text(encoding="utf-8")
    assert BENCHMARK_FIXTURE_NOTICE in summary
    assert "| FAMILY / rank_screen | 7 | 3 |" in summary


def test_path_cascade_benchmark_uses_distinct_deterministic_species_rerank(tmp_path) -> None:
    first = run_path_cascade_benchmark(output_dir=tmp_path / "first").cascade_result
    second = run_path_cascade_benchmark(output_dir=tmp_path / "second").cascade_result

    assert [score.node_id for score in first.species_top20] == [
        score.node_id for score in second.species_top20
    ]
    assert [score.node_id for score in first.species_reranked_top20] == [
        score.node_id for score in second.species_reranked_top20
    ]
    assert first.species_top20[0].node_id.endswith(":01")
    assert first.species_reranked_top20[0].node_id.endswith(":20")


def test_dev_vision_benchmark_cascade_cli_writes_reports(tmp_path, capsys) -> None:
    output = tmp_path / "cascade"
    args = build_parser().parse_args(
        [
            "dev",
            "vision",
            "benchmark-cascade",
            "--output-dir",
            str(output),
        ]
    )

    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["family_candidate_count"] == 7
    assert payload["species_candidates_beneath_genus_top3"] == 75
    assert payload["benchmark_metrics"] == str(output / "benchmark_metrics.json")
    assert payload["benchmark_summary"] == str(output / "benchmark_summary.md")
    assert payload["elapsed_seconds"] >= 0
    assert (output / "benchmark_metrics.json").exists()
    assert (output / "benchmark_summary.md").exists()
