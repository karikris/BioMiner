"""Tests for bounded reuse and execution after dynamic-pool revisions."""

from __future__ import annotations

import polars as pl
import pytest

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.run.dynamic_pool_revision_impact import (
    DynamicMatrixDependency,
    DynamicPoolDependency,
    DynamicReferenceChange,
    DynamicReferenceRevision,
    DynamicScoringRecordDependency,
    identify_affected_candidate_matrices,
    identify_affected_reference_pools,
    identify_affected_scoring_records,
)
from biominer.run.dynamic_pool_selective_rerun import (
    plan_dynamic_flickr_embedding_reuse,
    plan_dynamic_reference_embedding_reuse,
    validate_dynamic_flickr_embedding_reuse,
    validate_dynamic_reference_embedding_reuse,
)
from biominer.run.incremental_feature_reuse import (
    FLICKR_EMBEDDING_SCOPE,
    REFERENCE_EMBEDDING_SCOPE,
    feature_cache_entry_frame,
    feature_reuse_request_frame,
)


def _fp(value: str) -> str:
    return canonical_semantic_fingerprint({"fixture": value})


def _revision() -> DynamicReferenceRevision:
    return DynamicReferenceRevision(
        old_reference_bank_fingerprint=_fp("old-bank"),
        new_reference_bank_fingerprint=_fp("new-bank"),
        old_reference_geography_index_fingerprint=_fp("old-geo"),
        new_reference_geography_index_fingerprint=_fp("new-geo"),
        changes=(
            DynamicReferenceChange(
                reference_media_id="reference-added",
                change_type="added",
                new_taxon_key="species-a",
                new_route="adult_field",
                new_global_anchor_eligible=True,
            ),
            DynamicReferenceChange(
                reference_media_id="reference-metadata-modified",
                change_type="modified",
                old_taxon_key="species-b",
                new_taxon_key="species-b",
                old_route="adult_field",
                new_route="adult_field",
                old_geo_cluster_id="geo-a",
                new_geo_cluster_id="geo-b",
                old_global_anchor_eligible=True,
                new_global_anchor_eligible=True,
            ),
            DynamicReferenceChange(
                reference_media_id="reference-content-modified",
                change_type="modified",
                old_taxon_key="species-c",
                new_taxon_key="species-c",
                old_route="adult_field",
                new_route="adult_field",
                old_global_anchor_eligible=True,
                new_global_anchor_eligible=True,
            ),
            DynamicReferenceChange(
                reference_media_id="reference-removed",
                change_type="removed",
                old_taxon_key="species-d",
                old_route="adult_field",
                old_global_anchor_eligible=True,
            ),
        ),
    )


def _request(
    reference_media_id: str,
    content: str,
    *,
    required: bool = True,
    newly_admitted: bool = False,
) -> dict[str, object]:
    return {
        "feature_scope": REFERENCE_EMBEDDING_SCOPE,
        "item_id": reference_media_id,
        "input_content_fingerprint": _fp(content),
        "producer_fingerprint": _fp("bioclip-model"),
        "preprocessing_fingerprint": _fp("full-frame-preprocessing"),
        "required": required,
        "newly_admitted": newly_admitted,
    }


def _cache(cache_id: str, content: str) -> dict[str, object]:
    return {
        "cache_entry_id": cache_id,
        "feature_scope": REFERENCE_EMBEDDING_SCOPE,
        "input_content_fingerprint": _fp(content),
        "producer_fingerprint": _fp("bioclip-model"),
        "preprocessing_fingerprint": _fp("full-frame-preprocessing"),
        "artifact_id": f"embedding:{cache_id}",
        "artifact_fingerprint": _fp(f"embedding:{cache_id}"),
    }


def _requests() -> pl.DataFrame:
    return feature_reuse_request_frame(
        [
            _request("reference-stable", "stable"),
            _request("reference-added", "added", newly_admitted=True),
            _request("reference-metadata-modified", "metadata-same-content"),
            _request("reference-content-modified", "changed-content"),
            _request("reference-removed", "removed", required=False),
        ]
    )


def _cache_entries() -> pl.DataFrame:
    return feature_cache_entry_frame(
        [
            _cache("stable", "stable"),
            _cache("metadata", "metadata-same-content"),
            _cache("old-content", "old-content"),
            _cache("removed", "removed"),
        ]
    )


def test_reference_vectors_reuse_only_exact_semantic_cache_identities() -> None:
    projection = plan_dynamic_reference_embedding_reuse(
        _revision(),
        _requests(),
        _cache_entries(),
    )
    actions = {
        row["reference_media_id"]: row["action"]
        for row in projection.table.iter_rows(named=True)
    }

    assert actions == {
        "reference-added": "embed_new_reference_image",
        "reference-content-modified": "reembed_changed_reference_image",
        "reference-metadata-modified": "reuse_reference_embedding",
        "reference-removed": "filter_excluded_reference",
        "reference-stable": "reuse_reference_embedding",
    }
    assert projection.reused_reference_media_ids == (
        "reference-metadata-modified",
        "reference-stable",
    )
    assert projection.reference_media_ids_to_embed == (
        "reference-added",
        "reference-content-modified",
    )
    assert projection.excluded_reference_media_ids == ("reference-removed",)
    assert (
        projection.table.filter(
            pl.col("reference_media_id") == "reference-metadata-modified"
        ).item(0, "revision_change_type")
        == "modified"
    )


def test_reference_reuse_requires_complete_and_consistent_revision_inventory() -> None:
    incomplete = _requests().filter(pl.col("item_id") != "reference-added")
    with pytest.raises(ValueError, match="missing revision changes"):
        plan_dynamic_reference_embedding_reuse(
            _revision(), incomplete, _cache_entries()
        )

    bad_rows = _requests().drop("request_fingerprint").to_dicts()
    for row in bad_rows:
        if row["item_id"] == "reference-removed":
            row["required"] = True
    bad_flags = feature_reuse_request_frame(bad_rows)
    with pytest.raises(ValueError, match="flags do not match"):
        plan_dynamic_reference_embedding_reuse(_revision(), bad_flags, _cache_entries())


def test_reference_reuse_decisions_are_tamper_evident() -> None:
    projection = plan_dynamic_reference_embedding_reuse(
        _revision(), _requests(), _cache_entries()
    )
    tampered = projection.table.with_columns(
        pl.when(pl.col("reference_media_id") == "reference-stable")
        .then(pl.lit("reembed_changed_reference_image"))
        .otherwise(pl.col("action"))
        .alias("action")
    )

    with pytest.raises(ValueError, match="action mismatch|fingerprint mismatch"):
        validate_dynamic_reference_embedding_reuse(
            tampered,
            revision=_revision(),
        )


def _scoring_impacts() -> tuple[DynamicReferenceRevision, pl.DataFrame]:
    revision = _revision()
    pools = identify_affected_reference_pools(
        revision,
        [
            DynamicPoolDependency(
                plan_id="plan-affected",
                plan_fingerprint=_fp("plan-affected"),
                query_route="adult_field",
                query_geo_cluster_id="geo-a",
                candidate_taxon_keys=("species-d",),
                member_reference_media_ids=("reference-removed",),
                member_fingerprints=(_fp("member-removed"),),
                reference_bank_fingerprint=revision.old_reference_bank_fingerprint,
                reference_geography_index_fingerprint=(
                    revision.old_reference_geography_index_fingerprint
                ),
            ),
            DynamicPoolDependency(
                plan_id="plan-reusable",
                plan_fingerprint=_fp("plan-reusable"),
                query_route="adult_field",
                query_geo_cluster_id="geo-z",
                candidate_taxon_keys=("species-z",),
                member_reference_media_ids=("reference-stable",),
                member_fingerprints=(_fp("member-stable"),),
                reference_bank_fingerprint=revision.old_reference_bank_fingerprint,
                reference_geography_index_fingerprint=(
                    revision.old_reference_geography_index_fingerprint
                ),
            ),
        ],
    ).table
    matrices = identify_affected_candidate_matrices(
        revision,
        pools,
        [
            DynamicMatrixDependency(
                matrix_id="matrix-affected",
                matrix_kind="dynamic_pool_reference",
                matrix_signature=_fp("matrix-affected"),
                source_fingerprint=_fp("source-affected"),
                model_fingerprint=_fp("bioclip-model"),
                route="adult_field",
                subject_keys=("species-d",),
                reference_media_ids=("reference-removed",),
                upstream_plan_ids=("plan-affected",),
            ),
            DynamicMatrixDependency(
                matrix_id="matrix-reusable",
                matrix_kind="dynamic_pool_reference",
                matrix_signature=_fp("matrix-reusable"),
                source_fingerprint=_fp("source-reusable"),
                model_fingerprint=_fp("bioclip-model"),
                route="adult_field",
                subject_keys=("species-z",),
                reference_media_ids=("reference-stable",),
                upstream_plan_ids=("plan-reusable",),
            ),
        ],
    )
    shared_embedding = _fp("flickr-embedding-shared")
    shared_source = _fp("flickr-source-shared")
    missing_embedding = _fp("flickr-embedding-missing")
    scoring = identify_affected_scoring_records(
        revision,
        pools,
        matrices,
        [
            _scoring_dependency(
                "score-affected-shared",
                plan_id="plan-affected",
                matrix_id="matrix-affected",
                photo_id="photo-shared",
                source_hash=shared_source,
                embedding_fingerprint=shared_embedding,
            ),
            _scoring_dependency(
                "score-reusable-shared",
                plan_id="plan-reusable",
                matrix_id="matrix-reusable",
                photo_id="photo-shared",
                source_hash=shared_source,
                embedding_fingerprint=shared_embedding,
            ),
            _scoring_dependency(
                "score-affected-missing",
                plan_id="plan-affected",
                matrix_id="matrix-affected",
                photo_id="photo-missing",
                source_hash=_fp("flickr-source-missing"),
                embedding_fingerprint=missing_embedding,
            ),
        ],
    )
    return revision, scoring


def _scoring_dependency(
    scoring_record_id: str,
    *,
    plan_id: str,
    matrix_id: str,
    photo_id: str,
    source_hash: str,
    embedding_fingerprint: str,
) -> DynamicScoringRecordDependency:
    return DynamicScoringRecordDependency(
        scoring_record_id=scoring_record_id,
        source_record_id=f"flickr:{scoring_record_id}",
        flickr_photo_id=photo_id,
        organism_unit_id=f"organism:{scoring_record_id}",
        source_image_sha256=source_hash,
        flickr_embedding_fingerprint=embedding_fingerprint,
        score_partition_id="partition-a",
        score_partition_fingerprint=_fp("partition-a"),
        upstream_plan_ids=(plan_id,),
        upstream_matrix_ids=(matrix_id,),
    )


def _flickr_request(
    embedding_fingerprint: str,
    source_hash: str,
) -> dict[str, object]:
    return {
        "feature_scope": FLICKR_EMBEDDING_SCOPE,
        "item_id": embedding_fingerprint,
        "input_content_fingerprint": source_hash,
        "producer_fingerprint": _fp("bioclip-model"),
        "preprocessing_fingerprint": _fp("full-frame-preprocessing"),
        "required": True,
        "newly_admitted": False,
    }


def _flickr_cache(
    embedding_fingerprint: str,
    source_hash: str,
    *,
    artifact_fingerprint: str | None = None,
) -> dict[str, object]:
    return {
        "cache_entry_id": f"cache:{embedding_fingerprint}",
        "feature_scope": FLICKR_EMBEDDING_SCOPE,
        "input_content_fingerprint": source_hash,
        "producer_fingerprint": _fp("bioclip-model"),
        "preprocessing_fingerprint": _fp("full-frame-preprocessing"),
        "artifact_id": f"embedding:{embedding_fingerprint}",
        "artifact_fingerprint": artifact_fingerprint or embedding_fingerprint,
    }


def _flickr_requests_and_cache(
    scoring_impacts: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    unique = scoring_impacts.select(
        "flickr_embedding_fingerprint", "source_image_sha256"
    ).unique()
    requests = feature_reuse_request_frame(
        [_flickr_request(row[0], row[1]) for row in unique.iter_rows()]
    )
    shared = scoring_impacts.filter(pl.col("flickr_photo_id") == "photo-shared").row(
        0, named=True
    )
    cache = feature_cache_entry_frame(
        [
            _flickr_cache(
                shared["flickr_embedding_fingerprint"],
                shared["source_image_sha256"],
            )
        ]
    )
    return requests, cache


def test_flickr_embedding_reuse_is_unique_and_independent_of_score_impact() -> None:
    revision, scoring_impacts = _scoring_impacts()
    requests, cache = _flickr_requests_and_cache(scoring_impacts)

    projection = plan_dynamic_flickr_embedding_reuse(
        revision,
        scoring_impacts,
        requests,
        cache,
    )
    shared = projection.table.filter(pl.col("action") == "reuse_flickr_embedding").row(
        0, named=True
    )
    missing = projection.table.filter(pl.col("action") == "embed_flickr_image").row(
        0, named=True
    )

    assert projection.table.height == 2
    assert shared["affected_scoring_record_ids"] == ["score-affected-shared"]
    assert shared["reusable_scoring_record_ids"] == ["score-reusable-shared"]
    assert projection.reused_embedding_fingerprints == (
        shared["flickr_embedding_fingerprint"],
    )
    assert projection.embedding_fingerprints_to_materialize == (
        missing["flickr_embedding_fingerprint"],
    )


def test_flickr_reuse_rejects_missing_requests_and_wrong_cached_artifacts() -> None:
    revision, scoring_impacts = _scoring_impacts()
    requests, cache = _flickr_requests_and_cache(scoring_impacts)
    incomplete = requests.head(1)
    with pytest.raises(ValueError, match="request inventory mismatch"):
        plan_dynamic_flickr_embedding_reuse(
            revision,
            scoring_impacts,
            incomplete,
            cache,
        )

    cached = cache.row(0, named=True)
    wrong_cache = feature_cache_entry_frame(
        [
            _flickr_cache(
                cached["artifact_fingerprint"],
                cached["input_content_fingerprint"],
                artifact_fingerprint=_fp("wrong-artifact"),
            )
        ]
    )
    with pytest.raises(ValueError, match="artifact fingerprint mismatch"):
        plan_dynamic_flickr_embedding_reuse(
            revision,
            scoring_impacts,
            requests,
            wrong_cache,
        )


def test_flickr_reuse_decisions_are_tamper_evident() -> None:
    revision, scoring_impacts = _scoring_impacts()
    requests, cache = _flickr_requests_and_cache(scoring_impacts)
    projection = plan_dynamic_flickr_embedding_reuse(
        revision,
        scoring_impacts,
        requests,
        cache,
    )
    tampered = projection.table.with_columns(
        pl.lit([], dtype=pl.List(pl.String)).alias("scoring_record_ids")
    )

    with pytest.raises(ValueError, match="scoring evidence mismatch"):
        validate_dynamic_flickr_embedding_reuse(
            tampered,
            revision=revision,
            scoring_impacts=scoring_impacts,
        )
