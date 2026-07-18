"""Tests for leakage-safe reviewed Flickr evaluation splits."""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from biominer.evaluation.dynamic_pool_splits import (
    DYNAMIC_POOL_EVALUATION_SPLIT_SCHEMA,
    DYNAMIC_POOL_EVALUATION_SPLITS,
    REVIEWED_FLICKR_COMPONENT_SCHEMA,
    DynamicPoolEvaluationSplitPolicy,
    ReviewedFlickrSplitItem,
    build_reviewed_flickr_components,
    build_dynamic_pool_evaluation_splits,
    validate_dynamic_pool_evaluation_splits,
    validate_reviewed_flickr_components,
)


def _sha(character: str) -> str:
    return f"sha256:{character * 64}"


def _item(index: int, **changes: object) -> ReviewedFlickrSplitItem:
    character = format(index % 16, "x")
    values: dict[str, object] = {
        "item_id": f"item-{index}",
        "source_record_hash": _sha(character),
        "source_artifact_fingerprint": _sha(format((index + 1) % 16, "x")),
        "review_decision_fingerprint": _sha(format((index + 2) % 16, "x")),
        "flickr_photo_id": f"photo-{index}",
        "owner_group_id": f"owner-{index}",
        "duplicate_group_id": f"duplicate-{index}",
        "observation_group_id": f"observation-{index}",
        "geographic_cluster_id": f"geo-{index}",
        "no_geo": False,
        "source_mirror_group_id": f"mirror-{index}",
        "stratum_id": f"stratum-{index % 2}",
        "candidate_species_key": f"species-{index % 2}",
        "human_supported": index % 2 == 0,
        "sampling_weight": 1.0,
    }
    values.update(changes)
    return ReviewedFlickrSplitItem(**values)


def test_all_source_independence_identifiers_form_transitive_components() -> None:
    items = [_item(index) for index in range(8)]
    items[1] = replace(items[1], flickr_photo_id=items[0].flickr_photo_id)
    items[2] = replace(items[2], owner_group_id=items[1].owner_group_id)
    items[3] = replace(items[3], duplicate_group_id=items[2].duplicate_group_id)
    items[4] = replace(items[4], observation_group_id=items[3].observation_group_id)
    items[5] = replace(items[5], geographic_cluster_id=items[4].geographic_cluster_id)
    items[6] = replace(items[6], source_mirror_group_id=items[5].source_mirror_group_id)

    build = build_reviewed_flickr_components(items)
    chain = build.register.filter(
        pl.col("item_id").is_in([f"item-{i}" for i in range(7)])
    )

    assert build.register.schema == REVIEWED_FLICKR_COMPONENT_SCHEMA
    assert chain["independence_component_id"].n_unique() == 1
    assert chain["independence_component_size"].unique().to_list() == [7]
    assert build.component_count == 2
    assert build.maximum_component_size == 7


def test_no_geo_rows_do_not_share_a_synthetic_geographic_identity() -> None:
    items = [
        _item(index, geographic_cluster_id=None, no_geo=True) for index in range(6)
    ]

    build = build_reviewed_flickr_components(items)

    assert build.component_count == len(items)
    assert build.register["independence_component_id"].n_unique() == len(items)


def test_component_register_is_order_independent_and_fingerprinted() -> None:
    items = [_item(index) for index in range(8)]
    first = build_reviewed_flickr_components(items)
    second = build_reviewed_flickr_components(list(reversed(items)))

    assert first.register.to_dicts() == second.register.to_dicts()
    assert first.register_fingerprint == second.register_fingerprint
    assert first.register["register_fingerprint"].n_unique() == 1


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"flickr_photo_id": ""}, "flickr_photo_id"),
        ({"owner_group_id": ""}, "owner_group_id"),
        ({"duplicate_group_id": ""}, "duplicate_group_id"),
        ({"observation_group_id": ""}, "observation_group_id"),
        ({"source_mirror_group_id": ""}, "source_mirror_group_id"),
        ({"geographic_cluster_id": None, "no_geo": False}, "require"),
        ({"geographic_cluster_id": "geo", "no_geo": True}, "cannot claim"),
    ],
)
def test_split_items_fail_closed_on_missing_source_independence(
    changes: dict[str, object], match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _item(1, **changes)


def test_component_validator_rejects_tampering() -> None:
    register = build_reviewed_flickr_components(
        [_item(index) for index in range(8)]
    ).register
    tampered = register.with_columns(
        pl.when(pl.col("item_id") == "item-0")
        .then(pl.lit(register["independence_component_id"].item(1)))
        .otherwise(pl.col("independence_component_id"))
        .alias("independence_component_id")
    )

    with pytest.raises(ValueError, match="component mismatch"):
        validate_reviewed_flickr_components(tampered)


def _independent_component_register(count: int = 18) -> pl.DataFrame:
    return build_reviewed_flickr_components(
        [_item(index) for index in range(count)]
    ).register


def test_frozen_split_is_deterministic_component_atomic_and_outcome_complete() -> None:
    register = _independent_component_register()
    policy = DynamicPoolEvaluationSplitPolicy(
        split_version="dynamic-pool-reviewed-v1", random_seed=73
    )
    first = build_dynamic_pool_evaluation_splits(register, policy)
    reversed_register = build_reviewed_flickr_components(
        [_item(index) for index in reversed(range(18))]
    ).register
    second = build_dynamic_pool_evaluation_splits(reversed_register, policy)

    assert first.manifest.schema == DYNAMIC_POOL_EVALUATION_SPLIT_SCHEMA
    assert first.manifest.to_dicts() == second.manifest.to_dicts()
    assert set(first.manifest["evaluation_split"].to_list()) == set(
        DYNAMIC_POOL_EVALUATION_SPLITS
    )
    assert first.manifest["split_fingerprint"].n_unique() == 1
    weight_by_split = dict(policy.weights)
    count_by_split = dict(first.split_item_counts)
    for split in DYNAMIC_POOL_EVALUATION_SPLITS:
        target = register.height * weight_by_split[split] / policy.total_weight
        assert abs(count_by_split[split] - target) <= 1.0
    for split in DYNAMIC_POOL_EVALUATION_SPLITS:
        outcomes = set(
            first.manifest.filter(pl.col("evaluation_split") == split)[
                "human_supported"
            ].to_list()
        )
        assert outcomes == {False, True}


def test_shared_component_never_crosses_frozen_partitions() -> None:
    items = [_item(index) for index in range(18)]
    items[1] = replace(items[1], duplicate_group_id=items[0].duplicate_group_id)
    register = build_reviewed_flickr_components(items).register
    manifest = build_dynamic_pool_evaluation_splits(
        register,
        DynamicPoolEvaluationSplitPolicy(split_version="atomic-v1"),
    ).manifest
    shared = manifest.filter(
        pl.col("independence_component_id")
        == manifest.filter(pl.col("item_id") == "item-0")[
            "independence_component_id"
        ].item()
    )

    assert shared.height == 2
    assert shared["evaluation_split"].n_unique() == 1


def test_split_fails_when_outcomes_lack_three_independent_components() -> None:
    items = [_item(index, human_supported=True) for index in range(8)]
    items.extend(_item(20 + index, human_supported=False) for index in range(2))
    register = build_reviewed_flickr_components(items).register

    with pytest.raises(ValueError, match="outcome components cannot cover"):
        build_dynamic_pool_evaluation_splits(
            register,
            DynamicPoolEvaluationSplitPolicy(split_version="sparse-outcome-v1"),
        )


def test_split_validator_rejects_manual_component_movement() -> None:
    build = build_dynamic_pool_evaluation_splits(
        _independent_component_register(),
        DynamicPoolEvaluationSplitPolicy(split_version="tamper-split-v1"),
    )
    first = build.manifest.row(0, named=True)
    replacement = next(
        split
        for split in DYNAMIC_POOL_EVALUATION_SPLITS
        if split != first["evaluation_split"]
    )
    tampered = build.manifest.with_columns(
        pl.when(pl.col("item_id") == first["item_id"])
        .then(pl.lit(replacement))
        .otherwise(pl.col("evaluation_split"))
        .alias("evaluation_split")
    )

    with pytest.raises(ValueError, match="assignment mismatch"):
        validate_dynamic_pool_evaluation_splits(tampered)
