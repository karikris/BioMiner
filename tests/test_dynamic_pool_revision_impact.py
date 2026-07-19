"""Tests for selective dynamic-pool revision impact analysis."""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from biominer.run.dynamic_pool_revision_impact import (
    DynamicMatrixDependency,
    DynamicPoolDependency,
    DynamicReferenceChange,
    DynamicReferenceRevision,
    DynamicScoringRecordDependency,
    identify_affected_candidate_matrices,
    identify_affected_reference_pools,
    identify_affected_scoring_records,
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


def _matrix(
    index: int,
    *,
    kind: str,
    subjects: tuple[str, ...],
    references: tuple[str, ...],
    plans: tuple[str, ...] = (),
) -> DynamicMatrixDependency:
    return DynamicMatrixDependency(
        matrix_id=f"matrix-{index}",
        matrix_kind=kind,
        matrix_signature=_sha(index),
        source_fingerprint=_sha(index + 1),
        model_fingerprint=_sha(15),
        route="adult_field",
        subject_keys=subjects,
        reference_media_ids=references,
        upstream_plan_ids=plans,
    )


def test_matrix_impact_combines_reference_rows_and_upstream_pools() -> None:
    revision = _revision()
    pool_impacts = identify_affected_reference_pools(
        revision,
        [
            _pool(0, taxa=("species-a",), geo="geo-a"),
            _pool(2, taxa=("species-c",), geo="geo-c"),
        ],
    ).table
    matrices = [
        _matrix(
            0,
            kind="family_prototype",
            subjects=("family-a",),
            references=("reference-member",),
        ),
        _matrix(
            1,
            kind="candidate_prototype",
            subjects=("species-c",),
            references=("reference-stable",),
        ),
        _matrix(
            2,
            kind="dynamic_pool_reference",
            subjects=("species-a",),
            references=("reference-member",),
            plans=("plan-0",),
        ),
        _matrix(
            3,
            kind="dynamic_pool_reference",
            subjects=("species-c",),
            references=("reference-stable",),
            plans=("plan-2",),
        ),
    ]

    impacts = identify_affected_candidate_matrices(revision, pool_impacts, matrices)
    by_id = {row["matrix_id"]: row for row in impacts.iter_rows(named=True)}

    assert by_id["matrix-0"]["impact_status"] == "affected"
    assert by_id["matrix-1"]["impact_status"] == "reusable_as_is"
    assert by_id["matrix-2"]["affected_plan_ids"] == ["plan-0"]
    assert by_id["matrix-3"]["expected_action"] == (
        "reuse_matrix_without_materialization"
    )


def test_newly_eligible_reference_affects_matching_candidate_matrix() -> None:
    revision = _revision()
    pool_impacts = identify_affected_reference_pools(
        revision,
        [_pool(2, taxa=("species-c",), geo="geo-c")],
    ).table
    matrix = _matrix(
        1,
        kind="candidate_prototype",
        subjects=("species-b",),
        references=("reference-stable",),
    )

    impact = identify_affected_candidate_matrices(revision, pool_impacts, [matrix]).row(
        0, named=True
    )

    assert impact["impact_status"] == "affected"
    assert impact["affected_reference_media_ids"] == ["reference-new-local"]


def test_matrix_impact_rejects_unknown_pool_dependency() -> None:
    revision = _revision()
    pool_impacts = identify_affected_reference_pools(
        revision,
        [_pool(2, taxa=("species-c",), geo="geo-c")],
    ).table
    matrix = _matrix(
        2,
        kind="dynamic_pool_reference",
        subjects=("species-c",),
        references=("reference-stable",),
        plans=("plan-missing",),
    )

    with pytest.raises(ValueError, match="unknown pool plans"):
        identify_affected_candidate_matrices(revision, pool_impacts, [matrix])


def _record(
    index: int,
    *,
    plan: str,
    matrix: str,
) -> DynamicScoringRecordDependency:
    return DynamicScoringRecordDependency(
        scoring_record_id=f"score-{index}",
        source_record_id=f"flickr:{index}",
        flickr_photo_id=f"photo-{index}",
        organism_unit_id=f"organism-{index}",
        source_image_sha256=_sha(index),
        flickr_embedding_fingerprint=_sha(index + 1),
        score_partition_id="partition-a",
        score_partition_fingerprint=_sha(14),
        upstream_plan_ids=(plan,),
        upstream_matrix_ids=(matrix,),
    )


def _impact_chain() -> tuple[
    DynamicReferenceRevision,
    pl.DataFrame,
    pl.DataFrame,
]:
    revision = _revision()
    pool_impacts = identify_affected_reference_pools(
        revision,
        [
            _pool(0, taxa=("species-a",), geo="geo-a"),
            _pool(2, taxa=("species-c",), geo="geo-c"),
        ],
    ).table
    matrix_impacts = identify_affected_candidate_matrices(
        revision,
        pool_impacts,
        [
            _matrix(
                0,
                kind="dynamic_pool_reference",
                subjects=("species-a",),
                references=("reference-member",),
                plans=("plan-0",),
            ),
            _matrix(
                2,
                kind="dynamic_pool_reference",
                subjects=("species-c",),
                references=("reference-stable",),
                plans=("plan-2",),
            ),
        ],
    )
    return revision, pool_impacts, matrix_impacts


def test_scoring_impact_selects_only_records_with_affected_dependencies() -> None:
    revision, pools, matrices = _impact_chain()

    impacts = identify_affected_scoring_records(
        revision,
        pools,
        matrices,
        [
            _record(0, plan="plan-0", matrix="matrix-0"),
            _record(1, plan="plan-2", matrix="matrix-2"),
        ],
    )
    by_id = {row["scoring_record_id"]: row for row in impacts.iter_rows(named=True)}

    assert by_id["score-0"]["impact_status"] == "affected"
    assert by_id["score-0"]["affected_plan_ids"] == ["plan-0"]
    assert by_id["score-0"]["affected_matrix_ids"] == ["matrix-0"]
    assert by_id["score-0"]["expected_action"] == (
        "rescore_record_from_reused_flickr_embedding"
    )
    assert by_id["score-1"]["impact_status"] == "reusable_as_is"
    assert by_id["score-1"]["flickr_embedding_reusable"] is True


def test_scoring_impact_is_record_granular_within_one_partition() -> None:
    revision, pools, matrices = _impact_chain()
    impacts = identify_affected_scoring_records(
        revision,
        pools,
        matrices,
        [
            _record(0, plan="plan-0", matrix="matrix-0"),
            _record(1, plan="plan-2", matrix="matrix-2"),
        ],
    )

    assert impacts["score_partition_id"].unique().to_list() == ["partition-a"]
    assert impacts.filter(pl.col("impact_status") == "affected").height == 1
    assert impacts.filter(pl.col("impact_status") == "reusable_as_is").height == 1


def test_scoring_impact_rejects_unknown_dependencies() -> None:
    revision, pools, matrices = _impact_chain()

    with pytest.raises(ValueError, match="unknown identities"):
        identify_affected_scoring_records(
            revision,
            pools,
            matrices,
            [_record(0, plan="plan-missing", matrix="matrix-0")],
        )
