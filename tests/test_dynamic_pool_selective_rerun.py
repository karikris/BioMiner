"""Tests for bounded reuse and execution after dynamic-pool revisions."""

from __future__ import annotations

import polars as pl
import pytest

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.run.dynamic_pool_revision_impact import (
    DynamicReferenceChange,
    DynamicReferenceRevision,
)
from biominer.run.dynamic_pool_selective_rerun import (
    plan_dynamic_reference_embedding_reuse,
    validate_dynamic_reference_embedding_reuse,
)
from biominer.run.incremental_feature_reuse import (
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
