"""Tests for measured dynamic-pooling work-reuse reporting."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from biominer.bioclip.dynamic_pool_expansion import (
    DYNAMIC_POOL_EXPANSION_CACHE_REUSE_SCHEMA_VERSION,
    dynamic_pool_expansion_cache_reuse_schema,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.reports.dynamic_pool_efficiency import (
    EMBEDDING_REUSE_METRICS_VERSION,
    measure_embedding_reuse,
    validate_embedding_reuse_metrics,
)
from biominer.vision.flickr_embeddings import (
    FlickrEmbeddingArtifacts,
    FlickrEmbeddingPersistenceResult,
)
from biominer.vision.target_full_frame import EmbeddedTargetFullFramePlan


def test_embedding_reuse_counts_observed_flickr_and_reference_work() -> None:
    metrics = measure_embedding_reuse(
        (
            _flickr_result(hits=0, misses=2, loads_before=0, loads_after=1),
            _flickr_result(hits=2, misses=0, loads_before=1, loads_after=1),
        ),
        _reference_reuse_frame(),
    )

    assert metrics.schema_version == EMBEDDING_REUSE_METRICS_VERSION
    assert metrics.flickr_runs == 2
    assert metrics.flickr_embedding_requests == 4
    assert metrics.flickr_embedding_cache_hits == 2
    assert metrics.flickr_embedding_cache_misses == 2
    assert metrics.flickr_embeddings_materialized == 2
    assert metrics.flickr_encoder_calls == 1
    assert metrics.flickr_cache_hit_rate == pytest.approx(0.5)
    assert metrics.flickr_cache_hit_rate_status == "measured"
    assert metrics.reference_expansion_rows == 1
    assert metrics.reference_embedding_requests == 3
    assert metrics.reference_embedding_reuse_events == 3
    assert metrics.unique_reference_embeddings_reused == 3
    assert metrics.reference_embeddings_materialized == 0
    assert metrics.reference_encoder_invocations == 0
    assert metrics.total_embedding_requests == 7
    assert metrics.total_embedding_reuse_events == 5
    assert metrics.total_embeddings_materialized == 2
    assert metrics.avoided_embedding_bytes is None
    assert metrics.avoided_embedding_bytes_status == "not_instrumented"
    assert metrics.avoided_encoder_seconds is None
    assert metrics.avoided_encoder_seconds_status == "not_instrumented"
    assert len(metrics.source_fingerprints) == 3
    assert metrics.metrics_fingerprint.startswith("sha256:")
    validate_embedding_reuse_metrics(metrics)


def test_embedding_reuse_is_deterministic_and_rejects_metric_drift() -> None:
    flickr = (_flickr_result(hits=1, misses=1, loads_before=0, loads_after=1),)
    references = _reference_reuse_frame()

    first = measure_embedding_reuse(flickr, references)
    second = measure_embedding_reuse(flickr, references)

    assert first == second
    with pytest.raises(ValueError, match="total embedding reuse count mismatch"):
        validate_embedding_reuse_metrics(
            replace(first, total_embedding_reuse_events=999)
        )
    with pytest.raises(ValueError, match="must remain not_instrumented"):
        validate_embedding_reuse_metrics(
            replace(
                first,
                avoided_encoder_seconds=1.25,
                avoided_encoder_seconds_status="estimated",
            )
        )


def test_embedding_reuse_rejects_incomplete_sources_and_empty_observation() -> None:
    inconsistent = replace(
        _flickr_result(hits=0, misses=1, loads_before=0, loads_after=1),
        images_encoded=0,
    )
    with pytest.raises(ValueError, match="encoded-image count"):
        measure_embedding_reuse((inconsistent,), _empty_reference_reuse_frame())

    with pytest.raises(ValueError, match="at least one observation"):
        measure_embedding_reuse((), _empty_reference_reuse_frame())


def test_reference_only_observation_does_not_report_a_zero_flickr_rate() -> None:
    metrics = measure_embedding_reuse((), _reference_reuse_frame())

    assert metrics.flickr_embedding_requests == 0
    assert metrics.flickr_cache_hit_rate is None
    assert metrics.flickr_cache_hit_rate_status == "unavailable"


def _flickr_result(
    *,
    hits: int,
    misses: int,
    loads_before: int,
    loads_after: int,
) -> FlickrEmbeddingPersistenceResult:
    return FlickrEmbeddingPersistenceResult(
        embedded_plan=EmbeddedTargetFullFramePlan(
            embeddings=(),
            scoring_unit_references=(),
        ),
        artifacts=FlickrEmbeddingArtifacts(
            embeddings=pl.DataFrame(),
            photo_bindings=pl.DataFrame(),
        ),
        embeddings_path=Path("fixture/flickr_full_frame_embeddings.parquet"),
        photo_bindings_path=Path("fixture/flickr_embedding_bindings.parquet"),
        embedding_cache_fingerprint=_sha("a"),
        binding_set_fingerprint=_sha("b"),
        visual_inputs_total=hits + misses,
        photo_bindings_total=hits + misses,
        cache_hits=hits,
        cache_misses=misses,
        encoder_calls=int(misses > 0),
        images_encoded=misses,
        encoder_model_load_count_before=loads_before,
        encoder_model_load_count_after=loads_after,
        encoder_model_load_count_delta=loads_after - loads_before,
    )


def _reference_reuse_frame() -> pl.DataFrame:
    prior = (_sha("1"), _sha("2"))
    added = (_sha("3"),)
    expanded = tuple(sorted((*prior, *added)))
    row: dict[str, object] = {
        "schema_version": DYNAMIC_POOL_EXPANSION_CACHE_REUSE_SCHEMA_VERSION,
        "run_id": "fixture-run",
        "prior_plan_id": "dynamic-pool-plan:" + "a" * 64,
        "prior_plan_fingerprint": _sha("4"),
        "expanded_plan_id": "dynamic-pool-plan:" + "b" * 64,
        "expanded_plan_fingerprint": _sha("5"),
        "expansion_evidence_fingerprint": _sha("6"),
        "selection_policy_fingerprint": _sha("7"),
        "model_fingerprint": _sha("8"),
        "query_embedding_fingerprint": _sha("9"),
        "expansion_round": 1,
        "retained_reference_count": 2,
        "added_reference_count": 1,
        "dropped_reference_count": 0,
        "prior_reference_embedding_fingerprints": prior,
        "added_reference_embedding_fingerprints": added,
        "expanded_reference_embedding_fingerprints": expanded,
        "query_embedding_reused": True,
        "reference_embeddings_reused": True,
        "encoder_invocations": 0,
        "embedding_vectors_materialized": False,
        "expanded_membership_fingerprint": _sha("c"),
    }
    row["reuse_fingerprint"] = canonical_semantic_fingerprint(row)
    return pl.DataFrame(
        [row],
        schema=dynamic_pool_expansion_cache_reuse_schema(),
        orient="row",
        strict=True,
    )


def _empty_reference_reuse_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=dynamic_pool_expansion_cache_reuse_schema())


def _sha(character: str) -> str:
    return "sha256:" + character * 64
