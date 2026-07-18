"""Tests for selective dynamic-pool revision impact analysis."""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from biominer.run.dynamic_pool_revision_impact import (
    DynamicPoolDependency,
    DynamicReferenceChange,
    DynamicReferenceRevision,
    identify_affected_reference_pools,
    validate_dynamic_pool_revision_impact,
)


def _sha(index: int) -> str:
    return f"sha256:{format(index % 16, 'x') * 64}"


def _revision() -> DynamicReferenceRevision:
    return DynamicReferenceRevision(
        old_reference_bank_fingerprint=_sha(10),
        new_reference_bank_fingerprint=_sha(11),
        old_reference_geography_index_fingerprint=_sha(12),
        new_reference_geography_index_fingerprint=_sha(13),
        changes=(
            DynamicReferenceChange(
                reference_media_id="reference-member",
                change_type="removed",
                old_taxon_key="species-a",
                old_route="adult_field",
                old_geo_cluster_id="geo-a",
                old_global_anchor_eligible=True,
                old_local_anchor_eligible=True,
            ),
            DynamicReferenceChange(
                reference_media_id="reference-new-local",
                change_type="added",
                new_taxon_key="species-b",
                new_route="adult_field",
                new_geo_cluster_id="geo-b",
                new_local_anchor_eligible=True,
            ),
            DynamicReferenceChange(
                reference_media_id="reference-irrelevant",
                change_type="added",
                new_taxon_key="species-z",
                new_route="larval",
                new_geo_cluster_id="geo-z",
                new_global_anchor_eligible=True,
            ),
        ),
    )


def _pool(index: int, *, taxa: tuple[str, ...], geo: str) -> DynamicPoolDependency:
    return DynamicPoolDependency(
        plan_id=f"plan-{index}",
        plan_fingerprint=_sha(index),
        query_route="adult_field",
        query_geo_cluster_id=geo,
        candidate_taxon_keys=taxa,
        member_reference_media_ids=(
            "reference-member" if index == 0 else f"reference-stable-{index}",
        ),
        member_fingerprints=(_sha(index + 1),),
        reference_bank_fingerprint=_sha(10),
        reference_geography_index_fingerprint=_sha(12),
    )


def test_member_and_newly_eligible_references_affect_only_declared_pools() -> None:
    projection = identify_affected_reference_pools(
        _revision(),
        [
            _pool(0, taxa=("species-a",), geo="geo-a"),
            _pool(1, taxa=("species-b",), geo="geo-b"),
            _pool(2, taxa=("species-c",), geo="geo-c"),
        ],
    )

    by_id = {row["plan_id"]: row for row in projection.table.iter_rows(named=True)}
    assert projection.affected_plan_ids == ("plan-0", "plan-1")
    assert projection.reusable_plan_ids == ("plan-2",)
    assert by_id["plan-0"]["changed_member_reference_media_ids"] == ["reference-member"]
    assert by_id["plan-0"]["direct_member_change"] is True
    assert by_id["plan-1"]["changed_eligible_reference_media_ids"] == [
        "reference-new-local"
    ]
    assert by_id["plan-2"]["expected_action"] == "reuse_pool_without_rebuild"
    assert projection.changed_reference_ids_irrelevant_to_declared_pools == (
        "reference-irrelevant",
    )


def test_local_eligibility_does_not_invalidate_another_geography() -> None:
    projection = identify_affected_reference_pools(
        _revision(),
        [_pool(1, taxa=("species-b",), geo="geo-other")],
    )

    assert projection.affected_plan_ids == ()
    assert projection.reusable_plan_ids == ("plan-1",)
    assert "reference-new-local" in (
        projection.changed_reference_ids_irrelevant_to_declared_pools
    )


def test_new_global_eligibility_affects_every_matching_taxon_route_pool() -> None:
    revision = replace(
        _revision(),
        changes=(
            DynamicReferenceChange(
                reference_media_id="reference-new-global",
                change_type="added",
                new_taxon_key="species-b",
                new_route="adult_field",
                new_geo_cluster_id="geo-unrelated",
                new_global_anchor_eligible=True,
            ),
        ),
    )

    projection = identify_affected_reference_pools(
        revision,
        [
            _pool(1, taxa=("species-b",), geo="geo-a"),
            _pool(2, taxa=("species-b",), geo="geo-b"),
        ],
    )

    assert projection.affected_plan_ids == ("plan-1", "plan-2")


def test_pool_impact_is_deterministic_and_tamper_evident() -> None:
    pools = [
        _pool(0, taxa=("species-a",), geo="geo-a"),
        _pool(1, taxa=("species-b",), geo="geo-b"),
    ]
    first = identify_affected_reference_pools(_revision(), pools)
    second = identify_affected_reference_pools(_revision(), list(reversed(pools)))

    assert first.projection_fingerprint == second.projection_fingerprint
    assert first.table.equals(second.table)
    tampered = first.table.with_columns(
        pl.lit("reuse_pool_without_rebuild").alias("expected_action")
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_dynamic_pool_revision_impact(tampered)


def test_pool_dependency_must_bind_the_prior_revision_artifacts() -> None:
    stale = replace(
        _pool(0, taxa=("species-a",), geo="geo-a"),
        reference_bank_fingerprint=_sha(9),
    )

    with pytest.raises(ValueError, match="stale reference-bank"):
        identify_affected_reference_pools(_revision(), [stale])
