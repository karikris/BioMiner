"""Tests for the public dynamic-pooling CLI surface."""

from __future__ import annotations

import json

import pytest

from biominer.cli import build_parser, run
from biominer.run.dynamic_pool_cli import (
    DYNAMIC_POOL_OPERATION_SPECS,
    load_dynamic_pool_command_plan,
)
from biominer.run.dynamic_pool_config import (
    DynamicPoolingSettings,
    write_dynamic_pooling_settings,
)


def _sha(character: str) -> str:
    return f"sha256:{character * 64}"


def _operation_args(
    operation: str,
    settings_path,
    *,
    dry_run: bool = True,
    plan_output=None,
) -> list[str]:
    values = [
        "dynamic-pooling",
        operation,
        "--settings",
        str(settings_path),
    ]
    for name in DYNAMIC_POOL_OPERATION_SPECS[operation].required_inputs:
        values.extend(("--input", f"{name}=s3://inputs/run-17/{name}"))
    values.extend(("--output-root", "s3://outputs/run-17"))
    if plan_output is not None:
        values.extend(("--plan-output", str(plan_output)))
    if dry_run:
        values.append("--dry-run")
    return values


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
    assert load_dynamic_pool_command_plan(plan_path).to_dict() == persisted


@pytest.mark.parametrize(
    ("operation", "expected_issues"),
    [
        ("build-reference-geography-index", []),
        ("plan-pools", ["candidate_strategy is unselected"]),
        ("score-pools", ["fusion_method is unselected"]),
        ("build-review-sample", []),
        ("audit-quality", []),
        ("selective-rerun", []),
        (
            "export-handoff",
            ["candidate_strategy is unselected", "fusion_method is unselected"],
        ),
    ],
)
def test_every_operation_builds_a_structurally_valid_dry_run_plan(
    operation, expected_issues, tmp_path, capsys
) -> None:
    settings_path = write_dynamic_pooling_settings(
        DynamicPoolingSettings(),
        tmp_path / "settings.json",
    )

    assert (
        run(build_parser().parse_args(_operation_args(operation, settings_path))) == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["operation"] == operation
    assert payload["structural_validation_status"] == "valid"
    assert payload["selection_validation_issues"] == expected_issues
    assert payload["selection_requirements_satisfied"] is not bool(expected_issues)
    assert set(payload["input_bindings"]) == set(
        DYNAMIC_POOL_OPERATION_SPECS[operation].required_inputs
    )
    assert set(payload["output_bindings"]) == set(
        DYNAMIC_POOL_OPERATION_SPECS[operation].intended_outputs
    )


@pytest.mark.parametrize("operation", ["plan-pools", "score-pools", "export-handoff"])
def test_evidence_bound_selections_satisfy_operation_readiness(
    operation, tmp_path, capsys
) -> None:
    settings_path = write_dynamic_pooling_settings(
        DynamicPoolingSettings(
            candidate_strategy="parallel_family_geography_union",
            candidate_strategy_selection_fingerprint=_sha("a"),
            fusion_method="validation_fitted_linear",
            fusion_selection_fingerprint=_sha("b"),
        ),
        tmp_path / "selected-settings.json",
    )

    assert (
        run(build_parser().parse_args(_operation_args(operation, settings_path))) == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["selection_requirements_satisfied"] is True
    assert payload["selection_validation_issues"] == []
    assert payload["adapter_status"] == "not_connected"
    assert payload["scientific_authority"]["occurrence_release"] is False


@pytest.mark.parametrize(
    ("input_values", "message"),
    [
        ([], "missing="),
        (
            [
                "normalized_reference_geography=s3://inputs/reference_geo",
                "reference_embeddings=s3://inputs/embeddings",
                "reference_support_manifest=s3://inputs/support",
                "unexpected=s3://inputs/unexpected",
            ],
            "extra=",
        ),
        (
            [
                "normalized_reference_geography=s3://inputs/reference_geo",
                "reference_embeddings=s3://inputs/embeddings",
                "reference_embeddings=s3://inputs/other_embeddings",
                "reference_support_manifest=s3://inputs/support",
            ],
            "duplicate input binding",
        ),
        (
            [
                "normalized_reference_geography=https://example.test/reference_geo",
                "reference_embeddings=s3://inputs/embeddings",
                "reference_support_manifest=s3://inputs/support",
            ],
            "local, file, or s3 artifact URI",
        ),
    ],
)
def test_dry_run_rejects_invalid_bindings_before_writing_a_plan(
    input_values, message, tmp_path, capsys
) -> None:
    settings_path = write_dynamic_pooling_settings(
        DynamicPoolingSettings(),
        tmp_path / "settings.json",
    )
    plan_path = tmp_path / "should-not-exist.json"
    values = [
        "dynamic-pooling",
        "build-reference-geography-index",
        "--settings",
        str(settings_path),
    ]
    for value in input_values:
        values.extend(("--input", value))
    values.extend(
        (
            "--output-root",
            "s3://outputs/run-17",
            "--plan-output",
            str(plan_path),
            "--dry-run",
        )
    )

    assert run(build_parser().parse_args(values)) == 2
    assert message in json.loads(capsys.readouterr().out)["error"]
    assert not plan_path.exists()


def test_non_dry_run_fails_closed_without_creating_a_plan(tmp_path, capsys) -> None:
    settings_path = write_dynamic_pooling_settings(
        DynamicPoolingSettings(),
        tmp_path / "settings.json",
    )
    plan_path = tmp_path / "should-not-exist.json"
    values = _operation_args(
        "build-reference-geography-index",
        settings_path,
        dry_run=False,
        plan_output=plan_path,
    )

    assert run(build_parser().parse_args(values)) == 2
    assert (
        "production adapters are not connected"
        in json.loads(capsys.readouterr().out)["error"]
    )
    assert not plan_path.exists()


def test_persisted_plan_rejects_fingerprint_and_output_tampering(tmp_path) -> None:
    settings_path = write_dynamic_pooling_settings(
        DynamicPoolingSettings(),
        tmp_path / "settings.json",
    )
    plan_path = tmp_path / "plan.json"
    args = build_parser().parse_args(
        _operation_args(
            "build-reference-geography-index",
            settings_path,
            plan_output=plan_path,
        )
    )
    assert run(args) == 0

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["plan_fingerprint"] = _sha("f")
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_dynamic_pool_command_plan(plan_path)

    payload["output_bindings"]["reference_geography_index.jsonl"] = (
        "s3://outputs/a-different-run/reference_geography_index.jsonl"
    )
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="outputs do not match"):
        load_dynamic_pool_command_plan(plan_path)
