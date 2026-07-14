from __future__ import annotations

from dataclasses import replace
import hashlib
from math import isfinite

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from biominer.evaluation.leakage import (
    EVALUATION_IDENTITY_COMPONENT_SCHEMA,
    NATURAL_STREAM_PARTITION,
    build_evaluation_identity_components,
    build_evaluation_leakage_register,
)
from biominer.evaluation.uncertainty import (
    TARGET_METRIC_CONFIDENCE_INTERVAL_SCHEMA,
    GroupedBootstrapConfig,
    build_grouped_metric_confidence_intervals,
    validate_grouped_bootstrap_result,
)
from test_evaluation_leakage import _identity


def test_identity_components_are_transitive_and_deterministic() -> None:
    first = _identity("a", partition=NATURAL_STREAM_PARTITION)
    second = replace(
        _identity("b", partition=NATURAL_STREAM_PARTITION),
        duplicate_group_id=first.duplicate_group_id,
    )
    third = replace(
        _identity("c", partition=NATURAL_STREAM_PARTITION),
        photographer_id=second.photographer_id,
    )
    fourth = _identity("d", partition=NATURAL_STREAM_PARTITION)
    register = build_evaluation_leakage_register(
        (first, second, third, fourth),
        register_version="bootstrap-components-v1",
    )

    components = build_evaluation_identity_components(register)
    repeated = build_evaluation_identity_components(register.reverse())

    assert components.schema == EVALUATION_IDENTITY_COMPONENT_SCHEMA
    assert_frame_equal(components, repeated)
    by_item = {row["item_id"]: row for row in components.iter_rows(named=True)}
    assert {
        by_item[item_id]["bootstrap_component_id"] for item_id in ("a", "b", "c")
    } == {by_item["a"]["bootstrap_component_id"]}
    assert by_item["a"]["component_size"] == 3
    assert by_item["d"]["component_size"] == 1
    assert (
        by_item["d"]["bootstrap_component_id"] != by_item["a"]["bootstrap_component_id"]
    )


def test_grouped_bootstrap_resamples_whole_components_and_is_deterministic() -> None:
    frame, components = _bootstrap_fixture()
    observed_samples: list[set[str]] = []

    def evaluate(sample: pl.DataFrame) -> dict[str, float | None]:
        present = set(sample["evaluation_item_id"].to_list())
        observed_samples.append(present)
        transitive_group = {"a", "b", "c"}
        assert not (present & transitive_group) or transitive_group <= present
        if transitive_group <= present:
            group_weights = sample.filter(
                pl.col("evaluation_item_id").is_in(transitive_group)
            )["sampling_weight"].to_list()
            assert len(set(group_weights)) == 1
        total = float(sample["sampling_weight"].sum())
        target = float(sample.filter(pl.col("target_present"))["sampling_weight"].sum())
        return {"target_rate": target / total}

    config = GroupedBootstrapConfig(
        replicate_count=256,
        confidence_level=0.95,
        random_seed=71,
        minimum_valid_fraction=0.80,
    )
    first = build_grouped_metric_confidence_intervals(
        frame,
        components,
        metric_names=("target_rate",),
        point_estimates={
            (NATURAL_STREAM_PARTITION, "target_rate"): 0.75,
        },
        metric_evaluator=evaluate,
        input_fingerprint=_sha("input"),
        metric_configuration_fingerprint=_sha("metric-config"),
        config=config,
    )
    second = build_grouped_metric_confidence_intervals(
        frame.reverse(),
        components.reverse(),
        metric_names=("target_rate",),
        point_estimates={
            (NATURAL_STREAM_PARTITION, "target_rate"): 0.75,
        },
        metric_evaluator=evaluate,
        input_fingerprint=_sha("input"),
        metric_configuration_fingerprint=_sha("metric-config"),
        config=config,
    )

    validate_grouped_bootstrap_result(first)
    assert first.intervals.schema == TARGET_METRIC_CONFIDENCE_INTERVAL_SCHEMA
    assert_frame_equal(first.intervals, second.intervals)
    assert first.uncertainty_fingerprint == second.uncertainty_fingerprint
    row = first.intervals.row(0, named=True)
    assert row["point_estimate"] == pytest.approx(0.75)
    assert row["confidence_interval_lower"] == pytest.approx(0.0)
    assert row["confidence_interval_upper"] == pytest.approx(1.0)
    assert row["valid_replicates"] == 256
    assert row["undefined_replicates"] == 0
    assert row["independent_component_count"] == 2
    assert row["interval_status"] == "complete"
    assert observed_samples

    inconsistent = replace(
        first,
        intervals=first.intervals.with_columns(
            (pl.col("undefined_replicates") + 1).alias("undefined_replicates")
        ),
    )
    with pytest.raises(ValueError, match="replicate accounting"):
        validate_grouped_bootstrap_result(inconsistent)


def test_grouped_bootstrap_counts_undefined_replicates() -> None:
    frame, components = _bootstrap_fixture()

    def evaluate(sample: pl.DataFrame) -> dict[str, float | None]:
        classes = set(sample["target_present"].to_list())
        if classes != {False, True}:
            return {"two_class_metric": None}
        value = float(sample.filter(pl.col("target_present"))["sampling_weight"].sum())
        return {"two_class_metric": value / float(sample["sampling_weight"].sum())}

    result = build_grouped_metric_confidence_intervals(
        frame,
        components,
        metric_names=("two_class_metric",),
        point_estimates={
            (NATURAL_STREAM_PARTITION, "two_class_metric"): 0.75,
        },
        metric_evaluator=evaluate,
        input_fingerprint=_sha("input"),
        metric_configuration_fingerprint=_sha("metric-config"),
        config=GroupedBootstrapConfig(
            replicate_count=256,
            random_seed=37,
            minimum_valid_fraction=0.40,
        ),
    )

    row = result.intervals.row(0, named=True)
    assert 0 < row["valid_replicates"] < 256
    assert row["undefined_replicates"] == 256 - row["valid_replicates"]
    assert row["interval_status"] == "complete"
    assert isfinite(row["confidence_interval_lower"])
    assert isfinite(row["confidence_interval_upper"])


def test_grouped_bootstrap_rejects_incomplete_or_single_component_assignments() -> None:
    frame, components = _bootstrap_fixture()
    incomplete = components.filter(pl.col("item_id") != "d")
    with pytest.raises(ValueError, match="coverage mismatch"):
        build_grouped_metric_confidence_intervals(
            frame,
            incomplete,
            metric_names=("metric",),
            point_estimates={(NATURAL_STREAM_PARTITION, "metric"): 0.5},
            metric_evaluator=lambda _frame: {"metric": 0.5},
            input_fingerprint=_sha("input"),
            metric_configuration_fingerprint=_sha("metric-config"),
            config=GroupedBootstrapConfig(replicate_count=32),
        )

    shared = _identity("a", partition=NATURAL_STREAM_PARTITION)
    one_component_register = build_evaluation_leakage_register(
        (
            shared,
            *(
                replace(
                    _identity(item_id, partition=NATURAL_STREAM_PARTITION),
                    duplicate_group_id=shared.duplicate_group_id,
                )
                for item_id in ("b", "c", "d")
            ),
        ),
        register_version="one-bootstrap-component-v1",
    )
    one_component = build_evaluation_identity_components(one_component_register)
    with pytest.raises(ValueError, match="at least 2 independent components"):
        build_grouped_metric_confidence_intervals(
            frame,
            one_component,
            metric_names=("metric",),
            point_estimates={(NATURAL_STREAM_PARTITION, "metric"): 0.5},
            metric_evaluator=lambda _frame: {"metric": 0.5},
            input_fingerprint=_sha("input"),
            metric_configuration_fingerprint=_sha("metric-config"),
            config=GroupedBootstrapConfig(replicate_count=32),
        )


def _bootstrap_fixture() -> tuple[pl.DataFrame, pl.DataFrame]:
    first = _identity("a", partition=NATURAL_STREAM_PARTITION)
    second = replace(
        _identity("b", partition=NATURAL_STREAM_PARTITION),
        duplicate_group_id=first.duplicate_group_id,
    )
    third = replace(
        _identity("c", partition=NATURAL_STREAM_PARTITION),
        photographer_id=second.photographer_id,
    )
    fourth = _identity("d", partition=NATURAL_STREAM_PARTITION)
    register = build_evaluation_leakage_register(
        (first, second, third, fourth),
        register_version="bootstrap-fixture-v1",
    )
    components = build_evaluation_identity_components(register)
    frame = pl.DataFrame(
        {
            "evaluation_item_id": ["a", "b", "c", "d"],
            "evaluation_set": [NATURAL_STREAM_PARTITION] * 4,
            "sampling_weight": [1.0, 1.0, 1.0, 1.0],
            "target_present": [True, True, True, False],
        }
    )
    return frame, components


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
