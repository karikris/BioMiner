"""Tests for frozen, leakage-audited dynamic-pool model features."""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from biominer.evaluation.dynamic_pool_splits import (
    DynamicPoolEvaluationSplitPolicy,
    ReviewedFlickrSplitItem,
    build_dynamic_pool_evaluation_splits,
    build_reviewed_flickr_components,
)
from biominer.ml.dynamic_pool_features import (
    DYNAMIC_POOL_FEATURE_SCHEMA,
    DYNAMIC_POOL_MODEL_FEATURE_NAMES,
    DynamicPoolFeatureInput,
    build_dynamic_pool_feature_table,
    load_dynamic_pool_feature_table,
    validate_dynamic_pool_feature_table,
    write_dynamic_pool_feature_table,
)


def _sha(index: int) -> str:
    return f"sha256:{format(index % 16, 'x') * 64}"


def _split_item(index: int) -> ReviewedFlickrSplitItem:
    no_geo = index % 4 == 0
    return ReviewedFlickrSplitItem(
        item_id=f"item-{index:02d}",
        source_record_hash=_sha(index),
        source_artifact_fingerprint=_sha(index + 1),
        review_decision_fingerprint=_sha(index + 2),
        flickr_photo_id=f"photo-{index}",
        owner_group_id=f"owner-{index}",
        duplicate_group_id=f"duplicate-{index}",
        observation_group_id=f"observation-{index}",
        geographic_cluster_id=None if no_geo else f"geo-{index}",
        no_geo=no_geo,
        source_mirror_group_id=f"mirror-{index}",
        stratum_id=f"stratum-{index % 3}",
        candidate_species_key=f"species-{index % 2}",
        human_supported=index % 2 == 0,
        sampling_weight=1.0 + (index % 3) / 10,
    )


def _manifest() -> pl.DataFrame:
    components = build_reviewed_flickr_components(
        [_split_item(index) for index in range(18)]
    ).register
    return build_dynamic_pool_evaluation_splits(
        components,
        DynamicPoolEvaluationSplitPolicy(
            split_version="dynamic-pool-features-fixture-v1",
            random_seed=73,
        ),
    ).manifest


def _feature_input(index: int, **changes: object) -> DynamicPoolFeatureInput:
    no_geo = index % 4 == 0
    local_available = not no_geo and index % 3 != 0
    family_available = index % 5 != 0
    values: dict[str, object] = {
        "item_id": f"item-{index:02d}",
        "candidate_species_key": f"species-{index % 2}",
        "score_component_fingerprint": _sha(index + 3),
        "model_fingerprint": _sha(14),
        "reference_evidence_fingerprint": _sha(13),
        "query_fingerprint": _sha(index + 4),
        "global_prototype_similarity": 0.72,
        "global_nearest_reference_similarity": 0.81,
        "global_top_k_mean_similarity": 0.76,
        "raw_competitor_margin": 0.08 if index % 2 == 0 else -0.03,
        "local_evidence_available": local_available,
        "local_evidence_unavailable_reason": (
            None if local_available else "no_eligible_local_support"
        ),
        "geographic_cluster_id": None if no_geo else f"geo-{index}",
        "no_geo": no_geo,
        "route": "adult_field",
        "visual_domain": "live_field",
        "route_compatible": True,
        "quality_flag_count": index % 3,
        "global_support_coverage_fraction": 0.8,
        "global_top_k_coverage_fraction": 1.0,
        "global_observation_independence_fraction": 1.0,
        "global_reference_count": 4,
        "global_configured_reference_count": 5,
        "global_independent_observation_count": 4,
        "global_reference_shortfall_count": 1,
        "local_reference_count": 3 if local_available else 0,
        "local_configured_reference_count": 4,
        "local_independent_observation_count": 3 if local_available else 0,
        "local_reference_shortfall_count": 1 if local_available else 4,
        "primary_query_tier": f"T{index % 5 + 1}",
        "query_hit_count": index + 1,
        "family_similarity": 0.68 if family_available else None,
        "family_rank": 1 if family_available else None,
        "family_margin_to_next_raw": 0.12 if family_available else None,
        "local_prototype_similarity": 0.78 if local_available else None,
        "local_nearest_reference_similarity": 0.84 if local_available else None,
        "local_top_k_mean_similarity": 0.80 if local_available else None,
        "prototype_absolute_disagreement": 0.06 if local_available else None,
        "nearest_absolute_disagreement": 0.03 if local_available else None,
        "top_k_absolute_disagreement": 0.04 if local_available else None,
        "prototype_rank_movement": -1 if local_available else None,
        "nearest_rank_movement": 0 if local_available else None,
        "top_k_rank_movement": 1 if local_available else None,
        "local_support_coverage_fraction": 0.75 if local_available else None,
        "local_top_k_coverage_fraction": 1.0 if local_available else None,
        "local_observation_independence_fraction": (1.0 if local_available else None),
        "subject_area_ratio": None if index % 6 == 0 else 0.22,
        "query_text_similarity": None if index % 7 == 0 else 0.55,
        "query_text_margin": None if index % 7 == 0 else 0.08,
    }
    values.update(changes)
    return DynamicPoolFeatureInput(**values)


def _build():
    manifest = _manifest()
    inputs = [_feature_input(index) for index in range(18)]
    return manifest, build_dynamic_pool_feature_table(inputs, manifest)


def test_feature_table_retains_raw_evidence_and_frozen_split_contract() -> None:
    manifest, build = _build()

    assert build.table.schema == DYNAMIC_POOL_FEATURE_SCHEMA
    assert build.row_count == manifest.height == 18
    assert build.feature_count == len(DYNAMIC_POOL_MODEL_FEATURE_NAMES)
    assert {split for split, count in build.split_row_counts if count} == {
        "calibration",
        "validation",
        "final_test",
    }
    assert build.table["feature_table_fingerprint"].n_unique() == 1
    assert set(build.table["score_semantics"].to_list()) == {
        "raw_model_evidence_not_probability"
    }
    assert not any(build.table["probability_available"].to_list())


def test_label_is_retained_but_never_enters_model_feature_vector() -> None:
    manifest = _manifest()
    first = _feature_input(2)
    second = replace(
        first,
        item_id="item-03",
        candidate_species_key="species-1",
        geographic_cluster_id="geo-3",
    )
    inputs = [
        _feature_input(0),
        _feature_input(1),
        first,
        second,
        *[_feature_input(index) for index in range(4, 18)],
    ]
    table = build_dynamic_pool_feature_table(inputs, manifest).table

    assert "human_supported" not in DYNAMIC_POOL_MODEL_FEATURE_NAMES
    positive = table.filter(pl.col("item_id") == "item-02").row(0, named=True)
    negative = table.filter(pl.col("item_id") == "item-03").row(0, named=True)
    assert positive["human_supported"] is True
    assert negative["human_supported"] is False
    assert positive["feature_vector"] == negative["feature_vector"]


def test_missing_values_remain_raw_nulls_with_explicit_vector_indicators() -> None:
    _, build = _build()
    row = build.table.filter(pl.col("item_id") == "item-00").row(0, named=True)
    vector = dict(zip(row["feature_names"], row["feature_vector"], strict=True))

    assert row["local_prototype_similarity"] is None
    assert vector["local_prototype_similarity"] == 0.0
    assert vector["local_prototype_similarity_available"] == 0.0
    assert row["subject_area_ratio"] is None
    assert vector["subject_area_ratio"] == 0.0
    assert vector["subject_area_ratio_available"] == 0.0
    assert vector["no_geo"] == 1.0


def test_feature_table_is_deterministic_for_reversed_inputs() -> None:
    manifest = _manifest()
    inputs = [_feature_input(index) for index in range(18)]
    first = build_dynamic_pool_feature_table(inputs, manifest)
    second = build_dynamic_pool_feature_table(list(reversed(inputs)), manifest)

    assert first.feature_table_fingerprint == second.feature_table_fingerprint
    assert first.table.equals(second.table)


def test_feature_builder_requires_complete_exact_split_membership() -> None:
    manifest = _manifest()

    with pytest.raises(ValueError, match="exactly cover"):
        build_dynamic_pool_feature_table(
            [_feature_input(index) for index in range(17)], manifest
        )
    bad = [_feature_input(index) for index in range(18)]
    bad[0] = replace(bad[0], candidate_species_key="another-species")
    with pytest.raises(ValueError, match="candidate key"):
        build_dynamic_pool_feature_table(bad, manifest)


def test_feature_input_rejects_implicit_local_imputation_and_bad_shortfall() -> None:
    with pytest.raises(ValueError, match="null local values"):
        _feature_input(0, local_prototype_similarity=0.5)
    with pytest.raises(ValueError, match="shortfall"):
        _feature_input(1, global_reference_shortfall_count=2)


def test_feature_validator_detects_raw_or_component_split_tampering() -> None:
    _, build = _build()
    raw_tampered = build.table.with_columns(
        pl.when(pl.col("item_id") == "item-00")
        .then(pl.lit(0.11))
        .otherwise(pl.col("global_prototype_similarity"))
        .alias("global_prototype_similarity")
    )
    with pytest.raises(ValueError, match="input fingerprint"):
        validate_dynamic_pool_feature_table(raw_tampered)

    source = build.table.row(0, named=True)
    crossing = build.table.with_columns(
        pl.when(pl.col("item_id") == "item-00")
        .then(
            pl.lit(
                "validation"
                if source["evaluation_split"] != "validation"
                else "calibration"
            )
        )
        .otherwise(pl.col("evaluation_split"))
        .alias("evaluation_split")
    )
    with pytest.raises(ValueError, match="component crosses|row fingerprint"):
        validate_dynamic_pool_feature_table(crossing)


def test_feature_parquet_round_trip_preserves_nullable_raw_evidence(tmp_path) -> None:
    _, build = _build()

    path = write_dynamic_pool_feature_table(build.table, tmp_path)
    loaded = load_dynamic_pool_feature_table(path)

    assert loaded.equals(build.table)
    assert (
        loaded.filter(pl.col("item_id") == "item-00")[
            "local_prototype_similarity"
        ].item()
        is None
    )
