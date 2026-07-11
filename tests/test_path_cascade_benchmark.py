from __future__ import annotations

from dataclasses import replace
import json

import pytest

import biominer.benchmarks.path_cascade as path_cascade_benchmark
from biominer.benchmarks.path_cascade import (
    BENCHMARK_FIXTURE_NOTICE,
    BENCHMARK_KIND,
    BENCHMARK_SELECTED_GENUS_NODE_IDS,
    BENCHMARK_VERSION,
    build_seven_family_path_cascade_fixture,
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
        -1.0 <= score <= 1.0
        for step in steps
        for score in step["candidate_raw_similarities"]
    )
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


def test_comparative_benchmark_proves_current_rank_and_cumulative_beams_diverge(
    tmp_path,
) -> None:
    result = run_path_cascade_benchmark(output_dir=tmp_path / "comparison")
    comparison = result.metrics["subfamily_selection_comparison"]

    assert comparison["fixture_only"] is True
    assert comparison["rank"] == "SUBFAMILY"
    assert comparison["beam_width"] == 3
    assert comparison["production_beam_strategy"] == "global_rank_top_k"
    assert comparison["production_score_basis"] == "current_rank_raw_similarity_only"
    assert comparison["historical_score_basis"] == (
        "mean_family_and_subfamily_raw_similarity"
    )
    assert comparison["production_selected_node_ids"] == [
        "fixture:subfamily:01:01",
        "fixture:subfamily:02:01",
        "fixture:subfamily:02:02",
    ]
    assert comparison["historical_selected_node_ids"] == [
        "fixture:subfamily:02:01",
        "fixture:subfamily:02:02",
        "fixture:subfamily:03:01",
    ]
    assert comparison["historical_candidates"][0][
        "cumulative_path_raw_similarity"
    ] == pytest.approx(0.895)
    assert comparison["historical_candidates"][1][
        "cumulative_path_raw_similarity"
    ] == pytest.approx(0.89)
    assert comparison["historical_candidates"][2][
        "cumulative_path_raw_similarity"
    ] == pytest.approx(0.79)
    assert all(
        -1.0 <= candidate["current_rank_raw_similarity"] <= 1.0
        for candidate in comparison["production_candidates"]
    )
    assert comparison["selections_differ"] is True

    summary = result.summary_path.read_text(encoding="utf-8")
    assert "Production global current-rank top 3" in summary
    assert "Historical cumulative-path top 3" in summary
    assert "fixture:subfamily:01:01" in summary
    assert "fixture:subfamily:03:01" in summary


def test_historical_comparison_helper_is_private_and_fixture_only() -> None:
    helper = path_cascade_benchmark._historical_cumulative_subfamily_selection

    assert helper.__name__.startswith("_")
    assert helper.__module__ == path_cascade_benchmark.__name__
    assert not hasattr(
        path_cascade_benchmark,
        "historical_cumulative_subfamily_selection",
    )

    fixture = build_seven_family_path_cascade_fixture()
    scorer = path_cascade_benchmark.DeterministicRawSimilarityScorer(
        fixture.taxonomy_store
    )
    cascade_result = path_cascade_benchmark.classify_path_cascade(
        item={"benchmark_item_id": "private-helper-scope"},
        scorer=scorer,
        taxonomy_store=fixture.taxonomy_store,
    )
    family_step = next(
        step for step in cascade_result.rank_steps if step.rank == "FAMILY"
    )
    subfamily_step = next(
        step for step in cascade_result.rank_steps if step.rank == "SUBFAMILY"
    )
    non_fixture = replace(fixture, manifest={"benchmark_fixture": False})

    with pytest.raises(ValueError, match="restricted to the synthetic benchmark fixture"):
        helper(
            fixture=non_fixture,
            family_step=family_step,
            subfamily_step=subfamily_step,
        )


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
