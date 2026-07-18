"""Measured work-reuse reporting for geography-conditioned dynamic pooling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from math import isclose, isfinite
import re

import polars as pl

from biominer.bioclip.dynamic_pool_expansion import (
    validate_dynamic_pool_expansion_cache_reuse,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.vision.flickr_embeddings import FlickrEmbeddingPersistenceResult


EMBEDDING_REUSE_METRICS_VERSION = "dynamic-embedding-reuse-metrics-v1"
NOT_INSTRUMENTED = "not_instrumented"
MEASURED = "measured"
UNAVAILABLE = "unavailable"

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class EmbeddingReuseMetrics:
    """Observed Flickr and reference embedding requests and materializations."""

    schema_version: str
    flickr_runs: int
    flickr_embedding_requests: int
    flickr_embedding_cache_hits: int
    flickr_embedding_cache_misses: int
    flickr_embeddings_materialized: int
    flickr_encoder_calls: int
    flickr_cache_hit_rate: float | None
    flickr_cache_hit_rate_status: str
    reference_expansion_rows: int
    reference_embedding_requests: int
    reference_embedding_reuse_events: int
    unique_reference_embeddings_reused: int
    reference_embeddings_materialized: int
    reference_encoder_invocations: int
    total_embedding_requests: int
    total_embedding_reuse_events: int
    total_embeddings_materialized: int
    avoided_embedding_bytes: int | None
    avoided_embedding_bytes_status: str
    avoided_encoder_seconds: float | None
    avoided_encoder_seconds_status: str
    source_fingerprints: tuple[str, ...]
    metrics_fingerprint: str


def measure_embedding_reuse(
    flickr_results: Sequence[FlickrEmbeddingPersistenceResult],
    reference_cache_reuse: pl.DataFrame,
) -> EmbeddingReuseMetrics:
    """Aggregate observed cache activity without estimating bytes or time saved."""

    flickr_runs = _flickr_results(flickr_results)
    validate_dynamic_pool_expansion_cache_reuse(reference_cache_reuse)
    if not flickr_runs and reference_cache_reuse.is_empty():
        raise ValueError("embedding reuse metrics require at least one observation")

    flickr_requests = sum(item.visual_inputs_total for item in flickr_runs)
    flickr_hits = sum(item.cache_hits for item in flickr_runs)
    flickr_misses = sum(item.cache_misses for item in flickr_runs)
    flickr_materialized = sum(item.images_encoded for item in flickr_runs)
    flickr_encoder_calls = sum(item.encoder_calls for item in flickr_runs)

    reference_requests = sum(
        len(row["expanded_reference_embedding_fingerprints"])
        for row in reference_cache_reuse.iter_rows(named=True)
    )
    unique_reference_fingerprints = {
        str(fingerprint)
        for row in reference_cache_reuse.iter_rows(named=True)
        for fingerprint in row["expanded_reference_embedding_fingerprints"]
    }
    reference_materializations = sum(
        int(bool(row["embedding_vectors_materialized"]))
        for row in reference_cache_reuse.iter_rows(named=True)
    )
    reference_encoder_invocations = sum(
        int(row["encoder_invocations"])
        for row in reference_cache_reuse.iter_rows(named=True)
    )
    source_fingerprints = tuple(
        sorted(
            {
                *(
                    _flickr_source_fingerprint(index, item)
                    for index, item in enumerate(flickr_runs)
                ),
                *(
                    str(value)
                    for value in reference_cache_reuse["reuse_fingerprint"].to_list()
                ),
            }
        )
    )
    values: dict[str, object] = {
        "schema_version": EMBEDDING_REUSE_METRICS_VERSION,
        "flickr_runs": len(flickr_runs),
        "flickr_embedding_requests": flickr_requests,
        "flickr_embedding_cache_hits": flickr_hits,
        "flickr_embedding_cache_misses": flickr_misses,
        "flickr_embeddings_materialized": flickr_materialized,
        "flickr_encoder_calls": flickr_encoder_calls,
        "flickr_cache_hit_rate": (
            flickr_hits / flickr_requests if flickr_requests else None
        ),
        "flickr_cache_hit_rate_status": (MEASURED if flickr_requests else UNAVAILABLE),
        "reference_expansion_rows": reference_cache_reuse.height,
        "reference_embedding_requests": reference_requests,
        "reference_embedding_reuse_events": reference_requests,
        "unique_reference_embeddings_reused": len(unique_reference_fingerprints),
        "reference_embeddings_materialized": reference_materializations,
        "reference_encoder_invocations": reference_encoder_invocations,
        "total_embedding_requests": flickr_requests + reference_requests,
        "total_embedding_reuse_events": flickr_hits + reference_requests,
        "total_embeddings_materialized": (
            flickr_materialized + reference_materializations
        ),
        "avoided_embedding_bytes": None,
        "avoided_embedding_bytes_status": NOT_INSTRUMENTED,
        "avoided_encoder_seconds": None,
        "avoided_encoder_seconds_status": NOT_INSTRUMENTED,
        "source_fingerprints": source_fingerprints,
    }
    result = EmbeddingReuseMetrics(
        **values,
        metrics_fingerprint=canonical_semantic_fingerprint(values),
    )
    validate_embedding_reuse_metrics(result)
    return result


def validate_embedding_reuse_metrics(metrics: EmbeddingReuseMetrics) -> None:
    """Reject inconsistent arithmetic, unsupported savings and fingerprint drift."""

    if not isinstance(metrics, EmbeddingReuseMetrics):
        raise TypeError("metrics must be EmbeddingReuseMetrics")
    if metrics.schema_version != EMBEDDING_REUSE_METRICS_VERSION:
        raise ValueError("unsupported embedding reuse metrics version")
    integer_fields = (
        "flickr_runs",
        "flickr_embedding_requests",
        "flickr_embedding_cache_hits",
        "flickr_embedding_cache_misses",
        "flickr_embeddings_materialized",
        "flickr_encoder_calls",
        "reference_expansion_rows",
        "reference_embedding_requests",
        "reference_embedding_reuse_events",
        "unique_reference_embeddings_reused",
        "reference_embeddings_materialized",
        "reference_encoder_invocations",
        "total_embedding_requests",
        "total_embedding_reuse_events",
        "total_embeddings_materialized",
    )
    for field in integer_fields:
        _nonnegative_int(getattr(metrics, field), field=field)
    if metrics.flickr_embedding_requests != (
        metrics.flickr_embedding_cache_hits + metrics.flickr_embedding_cache_misses
    ):
        raise ValueError("Flickr embedding request accounting is incomplete")
    if metrics.flickr_embeddings_materialized != metrics.flickr_embedding_cache_misses:
        raise ValueError("Flickr materializations must equal observed cache misses")
    if metrics.flickr_embedding_requests:
        expected_rate = (
            metrics.flickr_embedding_cache_hits / metrics.flickr_embedding_requests
        )
        rate = _bounded_rate(
            metrics.flickr_cache_hit_rate,
            field="flickr_cache_hit_rate",
        )
        if metrics.flickr_cache_hit_rate_status != MEASURED or not isclose(
            rate,
            expected_rate,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("Flickr embedding cache-hit rate mismatch")
    elif (
        metrics.flickr_cache_hit_rate is not None
        or metrics.flickr_cache_hit_rate_status != UNAVAILABLE
    ):
        raise ValueError("unobserved Flickr cache-hit rate must remain unavailable")
    if metrics.reference_embedding_reuse_events != metrics.reference_embedding_requests:
        raise ValueError("reference reuse events must cover every observed request")
    if metrics.unique_reference_embeddings_reused > (
        metrics.reference_embedding_reuse_events
    ):
        raise ValueError("unique reference reuse cannot exceed reuse events")
    if (
        metrics.reference_embeddings_materialized != 0
        or metrics.reference_encoder_invocations != 0
    ):
        raise ValueError("reference expansion reuse metrics contain recomputation")
    if metrics.total_embedding_requests != (
        metrics.flickr_embedding_requests + metrics.reference_embedding_requests
    ):
        raise ValueError("total embedding request count mismatch")
    if metrics.total_embedding_reuse_events != (
        metrics.flickr_embedding_cache_hits + metrics.reference_embedding_reuse_events
    ):
        raise ValueError("total embedding reuse count mismatch")
    if metrics.total_embeddings_materialized != (
        metrics.flickr_embeddings_materialized
        + metrics.reference_embeddings_materialized
    ):
        raise ValueError("total embedding materialization count mismatch")
    if (
        metrics.avoided_embedding_bytes is not None
        or metrics.avoided_embedding_bytes_status != NOT_INSTRUMENTED
        or metrics.avoided_encoder_seconds is not None
        or metrics.avoided_encoder_seconds_status != NOT_INSTRUMENTED
    ):
        raise ValueError("unmeasured embedding savings must remain not_instrumented")
    if not metrics.source_fingerprints:
        raise ValueError("embedding reuse metrics require source fingerprints")
    if tuple(sorted(set(metrics.source_fingerprints))) != metrics.source_fingerprints:
        raise ValueError("embedding reuse source fingerprints are not canonical")
    for fingerprint in metrics.source_fingerprints:
        _sha256(fingerprint, field="source_fingerprint")
    if metrics.metrics_fingerprint != canonical_semantic_fingerprint(
        _embedding_reuse_base(metrics)
    ):
        raise ValueError("embedding reuse metrics fingerprint mismatch")


def _flickr_results(
    values: object,
) -> tuple[FlickrEmbeddingPersistenceResult, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError("flickr_results must be a sequence")
    items = tuple(values)
    if any(not isinstance(item, FlickrEmbeddingPersistenceResult) for item in items):
        raise TypeError("flickr_results contains an invalid result")
    for item in items:
        for field in (
            "visual_inputs_total",
            "cache_hits",
            "cache_misses",
            "encoder_calls",
            "images_encoded",
            "encoder_model_load_count_before",
            "encoder_model_load_count_after",
            "encoder_model_load_count_delta",
        ):
            _nonnegative_int(getattr(item, field), field=field)
        if item.cache_hits + item.cache_misses != item.visual_inputs_total:
            raise ValueError("Flickr result cache accounting is incomplete")
        if item.images_encoded != item.cache_misses:
            raise ValueError("Flickr encoded-image count differs from cache misses")
        if item.encoder_calls != int(item.cache_misses > 0):
            raise ValueError("Flickr encoder-call count differs from cache misses")
        if (
            item.encoder_model_load_count_after - item.encoder_model_load_count_before
            != (item.encoder_model_load_count_delta)
        ):
            raise ValueError("Flickr encoder model-load accounting is inconsistent")
        _sha256(
            item.embedding_cache_fingerprint,
            field="embedding_cache_fingerprint",
        )
        _sha256(item.binding_set_fingerprint, field="binding_set_fingerprint")
    return items


def _flickr_source_fingerprint(
    run_index: int,
    result: FlickrEmbeddingPersistenceResult,
) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema_version": "flickr-embedding-reuse-observation-v1",
            "run_index": run_index,
            "embedding_cache_fingerprint": result.embedding_cache_fingerprint,
            "binding_set_fingerprint": result.binding_set_fingerprint,
            "visual_inputs_total": result.visual_inputs_total,
            "cache_hits": result.cache_hits,
            "cache_misses": result.cache_misses,
            "encoder_calls": result.encoder_calls,
            "images_encoded": result.images_encoded,
            "encoder_model_load_count_before": (result.encoder_model_load_count_before),
            "encoder_model_load_count_after": result.encoder_model_load_count_after,
            "encoder_model_load_count_delta": result.encoder_model_load_count_delta,
        }
    )


def _embedding_reuse_base(metrics: EmbeddingReuseMetrics) -> dict[str, object]:
    return {
        field.name: getattr(metrics, field.name)
        for field in fields(metrics)
        if field.name != "metrics_fingerprint"
    }


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _bounded_rate(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    number = float(value)
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return number


def _sha256(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a canonical sha256 fingerprint")
    return text


__all__ = [
    "EMBEDDING_REUSE_METRICS_VERSION",
    "EmbeddingReuseMetrics",
    "measure_embedding_reuse",
    "validate_embedding_reuse_metrics",
]
