from __future__ import annotations

import json

import pytest

from biominer.benchmarks.reference_workflow_baseline import (
    REFERENCE_WORKFLOW_PERFORMANCE_BASELINE_SCHEMA_VERSION,
    BaselineMetric,
    build_reference_workflow_performance_baseline,
    main,
    reference_workflow_performance_markdown,
    validate_reference_workflow_performance_baseline,
)


def _phase14() -> dict[str, object]:
    return {
        "schema_version": "papilio-demoleus-build-week-prototype-report-v1.0.0",
        "reference_bank": {
            "selected_media": 93,
            "prototype_support": 81,
            "human_verified": 0,
        },
        "staged_flickr_inference": {
            "classified": 13_496,
            "candidate_score_rows": 634_312,
            "performance": {
                "rss_peak_memory_bytes": 1_765_261_312,
                "records_per_second": 2.274524,
            },
        },
    }


def _phase15() -> dict[str, object]:
    return {
        "schema_version": "biominer-phase15-prototype-final-verification-v1.0.0",
        "resume_and_cache": {
            "completed_resume_without_stage_work": True,
            "support_embedding_resume_recomputed": 0,
            "support_embedding_resume_reused": 81,
            "bioclip_persistent_model_loads": 1,
            "bioclip_model_cache_hits": 6,
        },
    }


def test_baseline_preserves_measured_derived_and_unavailable_metrics() -> None:
    baseline = build_reference_workflow_performance_baseline(
        _phase14(),
        _phase15(),
        source_git_sha="abc123",
        generated_at="2026-07-17T10:19:41Z",
    )

    report = baseline.report
    assert (
        report["schema_version"]
        == REFERENCE_WORKFLOW_PERFORMANCE_BASELINE_SCHEMA_VERSION
    )
    assert report["strict_live_run_executed"] is False
    metrics = report["metrics"]
    assert metrics["reference_media_selected"]["value"] == 93
    assert metrics["references_awaiting_human_review"] == {
        "status": "derived",
        "value": 81,
        "unit": "rows",
        "source": (
            "phase14.reference_bank.prototype_support - "
            "phase14.reference_bank.human_verified"
        ),
        "reason": None,
    }
    assert metrics["manual_reference_reviews_completed"]["value"] == 0
    assert metrics["reference_embedding_cache_hits"]["value"] == 81
    assert metrics["bioclip_persistent_model_loads"]["value"] == 1
    assert metrics["peak_rss_memory"]["value"] == 1_765_261_312
    assert metrics["strict_time_to_first_flickr_score"]["status"] == "unavailable"
    assert metrics["strict_time_to_first_flickr_score"]["value"] is None
    assert metrics["full_rerun_work_avoided"]["status"] == "not_instrumented"
    validate_reference_workflow_performance_baseline(report)


def test_baseline_fingerprint_is_deterministic_and_tamper_evident() -> None:
    first = build_reference_workflow_performance_baseline(
        _phase14(),
        _phase15(),
        source_git_sha="abc123",
        generated_at="2026-07-17T10:19:41Z",
    ).report
    second = build_reference_workflow_performance_baseline(
        _phase14(),
        _phase15(),
        source_git_sha="abc123",
        generated_at="2026-07-17T10:19:41Z",
    ).report
    assert first == second

    first["metrics"]["reference_media_selected"]["value"] = 94
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_reference_workflow_performance_baseline(first)


def test_baseline_rejects_false_measured_or_unavailable_values() -> None:
    with pytest.raises(ValueError, match="nonnegative value"):
        BaselineMetric(status="measured", value=None, unit="seconds", source="x")
    with pytest.raises(ValueError, match="must not have a value"):
        BaselineMetric(
            status="unavailable",
            value=0,
            unit="seconds",
            reason="not measured",
        )

    phase14 = _phase14()
    phase14["reference_bank"]["human_verified"] = 82
    with pytest.raises(ValueError, match="exceeds support count"):
        build_reference_workflow_performance_baseline(
            phase14,
            _phase15(),
            source_git_sha="abc123",
            generated_at="2026-07-17T10:19:41Z",
        )


def test_markdown_labels_proxy_and_unavailable_values() -> None:
    report = build_reference_workflow_performance_baseline(
        _phase14(),
        _phase15(),
        source_git_sha="abc123",
        generated_at="2026-07-17T10:19:41Z",
    ).report

    markdown = reference_workflow_performance_markdown(report)

    assert "prototype-only proxy" in markdown
    assert "strict_time_to_first_flickr_score" in markdown
    assert "unavailable" in markdown
    assert "Unavailable values are not zero" in markdown


def test_module_cli_writes_reproducible_json_and_markdown(tmp_path, capsys) -> None:
    phase14 = tmp_path / "phase14.json"
    phase15 = tmp_path / "phase15.json"
    phase14.write_text(json.dumps(_phase14()), encoding="utf-8")
    phase15.write_text(json.dumps(_phase15()), encoding="utf-8")
    output = tmp_path / "baseline"

    assert (
        main(
            [
                "--phase14-report",
                str(phase14),
                "--phase15-verification",
                str(phase15),
                "--source-git-sha",
                "abc123",
                "--generated-at",
                "2026-07-17T10:19:41Z",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    report = json.loads(
        (output / "strict_workflow_performance_baseline.json").read_text()
    )
    assert result["json"] == str(output / "strict_workflow_performance_baseline.json")
    assert report["source_git_sha"] == "abc123"
    assert report["metrics"]["reference_candidates_acquired"]["status"] == "unavailable"
    assert (output / "strict_workflow_performance_baseline.md").exists()
