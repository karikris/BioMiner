"""Tests for the public dynamic-pooling CLI surface."""

from __future__ import annotations

import json

from biominer.cli import build_parser, run
from biominer.run.dynamic_pool_cli import DYNAMIC_POOL_OPERATION_SPECS
from biominer.run.dynamic_pool_config import (
    DynamicPoolingSettings,
    write_dynamic_pooling_settings,
)


def test_parser_exposes_all_dynamic_pooling_operations() -> None:
    parser = build_parser()
    dynamic_parser = parser._subparsers._group_actions[0].choices["dynamic-pooling"]
    choices = dynamic_parser._subparsers._group_actions[0].choices

    assert set(choices) == set(DYNAMIC_POOL_OPERATION_SPECS)


def test_build_reference_geography_index_dry_run_writes_bound_plan(
    tmp_path, capsys
) -> None:
    settings_path = write_dynamic_pooling_settings(
        DynamicPoolingSettings(),
        tmp_path / "settings.json",
    )
    plan_path = tmp_path / "plans" / "reference-index.json"
    args = build_parser().parse_args(
        [
            "dynamic-pooling",
            "build-reference-geography-index",
            "--settings",
            str(settings_path),
            "--input",
            "reference_embeddings=s3://evidence/reference_embeddings.parquet",
            "--input",
            "normalized_reference_geography=s3://evidence/reference_geo.parquet",
            "--input",
            "reference_support_manifest=s3://evidence/reference_support.json",
            "--output-root",
            "s3://plans/run-17",
            "--plan-output",
            str(plan_path),
            "--dry-run",
        ]
    )

    assert run(args) == 0
    printed = json.loads(capsys.readouterr().out)
    persisted = json.loads(plan_path.read_text(encoding="utf-8"))
    assert printed["operation"] == "build-reference-geography-index"
    assert printed["adapter_status"] == "not_connected"
    assert printed["scientific_authority"] == {
        "calibration": False,
        "human_verification": False,
        "occurrence_release": False,
        "statistical_support": False,
    }
    assert printed["settings_selection_status"] == "unselected"
    assert printed["output_bindings"]["reference_geography_index.jsonl"] == (
        "s3://plans/run-17/reference_geography_index.jsonl"
    )
    assert persisted["plan_fingerprint"] == printed["plan_fingerprint"]
