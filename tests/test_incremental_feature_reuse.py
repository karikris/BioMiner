from __future__ import annotations

import polars as pl
import pytest

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.run.incremental_feature_reuse import (
    DETECTOR_SCOPE,
    FLICKR_EMBEDDING_SCOPE,
    INCREMENTAL_FEATURE_REUSE_PLAN_FILE,
    REFERENCE_EMBEDDING_SCOPE,
    calculate_incremental_feature_reuse,
    feature_cache_entry_frame,
    feature_item_ids_to_compute,
    feature_reuse_request_frame,
    incremental_feature_reuse_metrics,
    validate_incremental_feature_reuse_plan,
    write_incremental_feature_reuse_plan,
)


def _fp(value: str) -> str:
    return canonical_semantic_fingerprint({"fixture": value})


def _request(
    scope: str,
    item_id: str,
    content: str,
    *,
    producer: str = "producer-v1",
    preprocessing: str = "preprocessing-v1",
    required: bool = True,
    newly_admitted: bool = False,
) -> dict[str, object]:
    return {
        "feature_scope": scope,
        "item_id": item_id,
        "input_content_fingerprint": _fp(content),
        "producer_fingerprint": _fp(producer),
        "preprocessing_fingerprint": _fp(preprocessing),
        "required": required,
        "newly_admitted": newly_admitted,
    }


def _cache(
    scope: str,
    cache_id: str,
    content: str,
    *,
    producer: str = "producer-v1",
    preprocessing: str = "preprocessing-v1",
) -> dict[str, object]:
    return {
        "cache_entry_id": cache_id,
        "feature_scope": scope,
        "input_content_fingerprint": _fp(content),
        "producer_fingerprint": _fp(producer),
        "preprocessing_fingerprint": _fp(preprocessing),
        "artifact_id": f"artifact:{cache_id}",
        "artifact_fingerprint": _fp(f"artifact:{cache_id}"),
    }


def test_content_addressed_plan_reuses_only_exact_feature_identities(
    tmp_path,
) -> None:
    requests = feature_reuse_request_frame(
        [
            _request(DETECTOR_SCOPE, "flickr:detector-hit", "photo-a"),
            _request(
                DETECTOR_SCOPE,
                "flickr:detector-changed",
                "photo-b",
                producer="producer-v2",
            ),
            _request(FLICKR_EMBEDDING_SCOPE, "flickr:embedding-hit", "photo-a"),
            _request(
                FLICKR_EMBEDDING_SCOPE,
                "flickr:preprocessing-changed",
                "photo-c",
                preprocessing="preprocessing-v2",
            ),
            _request(
                REFERENCE_EMBEDDING_SCOPE,
                "reference:retained",
                "reference-a",
            ),
            _request(
                REFERENCE_EMBEDDING_SCOPE,
                "reference:excluded",
                "reference-b",
                required=False,
            ),
            _request(
                REFERENCE_EMBEDDING_SCOPE,
                "reference:new",
                "reference-c",
                newly_admitted=True,
            ),
        ]
    )
    cache = feature_cache_entry_frame(
        [
            _cache(DETECTOR_SCOPE, "detector-a", "photo-a"),
            _cache(DETECTOR_SCOPE, "detector-b-old", "photo-b"),
            _cache(FLICKR_EMBEDDING_SCOPE, "flickr-a", "photo-a"),
            _cache(FLICKR_EMBEDDING_SCOPE, "flickr-c-old", "photo-c"),
            _cache(REFERENCE_EMBEDDING_SCOPE, "reference-a", "reference-a"),
            _cache(REFERENCE_EMBEDDING_SCOPE, "reference-b", "reference-b"),
        ]
    )

    plan = calculate_incremental_feature_reuse(requests, cache)
    actions = {
        str(row["item_id"]): row["action"]
        for row in plan.iter_rows(named=True)
    }

    assert actions == {
        "flickr:detector-hit": "reuse_yoloe_detection",
        "flickr:detector-changed": "run_yoloe",
        "flickr:embedding-hit": "reuse_flickr_embedding",
        "flickr:preprocessing-changed": "embed_flickr_image",
        "reference:retained": "reuse_reference_embedding",
        "reference:excluded": "filter_excluded_reference",
        "reference:new": "embed_new_reference_image",
    }
    assert feature_item_ids_to_compute(plan, feature_scope=DETECTOR_SCOPE) == (
        "flickr:detector-changed",
    )
    assert feature_item_ids_to_compute(
        plan,
        feature_scope=FLICKR_EMBEDDING_SCOPE,
    ) == ("flickr:preprocessing-changed",)
    assert feature_item_ids_to_compute(
        plan,
        feature_scope=REFERENCE_EMBEDDING_SCOPE,
    ) == ("reference:new",)
    excluded = plan.filter(pl.col("item_id") == "reference:excluded").row(
        0,
        named=True,
    )
    assert excluded["cache_hit"] is True
    assert excluded["reusable_artifact_id"] == "artifact:reference-b"
    metrics = incremental_feature_reuse_metrics(plan)
    assert metrics["item_count"].sum() == plan.height
    assert metrics["cache_hit_count"].sum() == 4
    path = write_incremental_feature_reuse_plan(plan, tmp_path)
    assert path.name == INCREMENTAL_FEATURE_REUSE_PLAN_FILE
    assert pl.read_parquet(path).equals(plan)


def test_plan_rejects_tampered_cache_decisions() -> None:
    requests = feature_reuse_request_frame(
        [_request(DETECTOR_SCOPE, "flickr:1", "photo-a")]
    )
    cache = feature_cache_entry_frame(
        [_cache(DETECTOR_SCOPE, "detector-a", "photo-a")]
    )
    plan = calculate_incremental_feature_reuse(requests, cache)
    tampered = plan.with_columns(pl.lit("run_yoloe").alias("action"))

    with pytest.raises(ValueError, match="action mismatch|fingerprint mismatch"):
        validate_incremental_feature_reuse_plan(tampered)


def test_cache_rejects_duplicate_content_identity() -> None:
    rows = [
        _cache(FLICKR_EMBEDDING_SCOPE, "cache-a", "photo-a"),
        _cache(FLICKR_EMBEDDING_SCOPE, "cache-b", "photo-a"),
    ]

    with pytest.raises(ValueError, match="repeats a content identity"):
        feature_cache_entry_frame(rows)


def test_only_required_references_can_be_newly_admitted() -> None:
    with pytest.raises(ValueError, match="newly admitted"):
        feature_reuse_request_frame(
            [
                _request(
                    REFERENCE_EMBEDDING_SCOPE,
                    "reference:excluded-new",
                    "reference-a",
                    required=False,
                    newly_admitted=True,
                )
            ]
        )
