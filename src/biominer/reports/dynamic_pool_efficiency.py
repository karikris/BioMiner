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
from biominer.bioclip.dynamic_pool_compute import (
    POOL_MATRIX_BATCH_METRICS_VERSION,
    PoolMatrixBatchMetrics,
)
from biominer.bioclip.matrix_cache import (
    DynamicPoolMatrixCacheMetrics,
    MatrixCacheMetrics,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.vision.flickr_embeddings import FlickrEmbeddingPersistenceResult


EMBEDDING_REUSE_METRICS_VERSION = "dynamic-embedding-reuse-metrics-v1"
MATRIX_CACHE_ROLE_METRICS_VERSION = "dynamic-matrix-cache-role-metrics-v1"
MATRIX_REUSE_METRICS_VERSION = "dynamic-matrix-reuse-metrics-v1"
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


@dataclass(frozen=True, slots=True)
class MatrixCacheRoleMetrics:
    """Terminal worker-cache observations for one semantic matrix role."""

    schema_version: str
    matrix_role: str
    worker_snapshots: int
    requests: int
    hits: int
    misses: int
    materializations: int
    active_entries: int
    rows_materialized: int
    bytes_materialized: int
    evictions: int
    hit_rate: float | None
    hit_rate_status: str
    metrics_fingerprint: str


@dataclass(frozen=True, slots=True)
class MatrixReuseMetrics:
    """Observed worker-cache reuse and separate pool-batch sharing metrics."""

    schema_version: str
    family: MatrixCacheRoleMetrics
    candidate: MatrixCacheRoleMetrics
    pool: MatrixCacheRoleMetrics
    worker_cache_requests: int
    worker_cache_hits: int
    worker_cache_misses: int
    worker_cache_materializations: int
    worker_cache_rows_materialized: int
    worker_cache_bytes_materialized: int
    worker_cache_evictions: int
    worker_cache_hit_rate: float | None
    worker_cache_hit_rate_status: str
    pool_matrix_batch_runs: int
    pool_matrix_batch_work_items: int
    pool_matrix_execution_batches: int
    pool_matrix_references: int
    unique_pool_matrix_observations: int
    unique_pool_matrix_row_observations: int
    unique_pool_matrix_byte_observations: int
    within_batch_matrix_reuses: int
    cross_batch_matrix_reloads: int
    maximum_batch_work_items: int
    maximum_batch_unique_pool_matrices: int
    maximum_batch_pool_matrix_bytes: int
    batch_encoder_invocations: int
    batch_image_materializations: int
    observed_matrix_reuse_events: int
    avoided_matrix_bytes: int | None
    avoided_matrix_bytes_status: str
    avoided_matrix_seconds: float | None
    avoided_matrix_seconds_status: str
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


def measure_matrix_reuse(
    family_snapshots: Sequence[MatrixCacheMetrics],
    dynamic_snapshots: Sequence[DynamicPoolMatrixCacheMetrics],
    pool_batch_metrics: Sequence[PoolMatrixBatchMetrics],
) -> MatrixReuseMetrics:
    """Aggregate terminal cache snapshots and pool-batch sharing observations."""

    family_items = _typed_sequence(
        family_snapshots,
        expected_type=MatrixCacheMetrics,
        field="family_snapshots",
    )
    dynamic_items = _typed_sequence(
        dynamic_snapshots,
        expected_type=DynamicPoolMatrixCacheMetrics,
        field="dynamic_snapshots",
    )
    batch_items = _typed_sequence(
        pool_batch_metrics,
        expected_type=PoolMatrixBatchMetrics,
        field="pool_batch_metrics",
    )
    if not family_items and not dynamic_items and not batch_items:
        raise ValueError("matrix reuse metrics require at least one observation")
    for item in batch_items:
        _validate_pool_matrix_batch_metrics(item)

    family = _matrix_cache_role_metrics("family", family_items)
    candidate = _matrix_cache_role_metrics(
        "candidate",
        tuple(item.candidate for item in dynamic_items),
    )
    pool = _matrix_cache_role_metrics(
        "pool",
        tuple(item.pool for item in dynamic_items),
    )
    roles = (family, candidate, pool)
    worker_requests = sum(item.requests for item in roles)
    worker_hits = sum(item.hits for item in roles)
    worker_misses = sum(item.misses for item in roles)
    source_fingerprints = tuple(
        sorted(
            {
                *(
                    _matrix_cache_source_fingerprint("family", index, item)
                    for index, item in enumerate(family_items)
                ),
                *(
                    _matrix_cache_source_fingerprint("candidate", index, item.candidate)
                    for index, item in enumerate(dynamic_items)
                ),
                *(
                    _matrix_cache_source_fingerprint("pool", index, item.pool)
                    for index, item in enumerate(dynamic_items)
                ),
                *(item.metrics_fingerprint for item in batch_items),
            }
        )
    )
    within_batch_reuses = sum(item.within_batch_matrix_reuses for item in batch_items)
    values: dict[str, object] = {
        "schema_version": MATRIX_REUSE_METRICS_VERSION,
        "family": family,
        "candidate": candidate,
        "pool": pool,
        "worker_cache_requests": worker_requests,
        "worker_cache_hits": worker_hits,
        "worker_cache_misses": worker_misses,
        "worker_cache_materializations": sum(item.materializations for item in roles),
        "worker_cache_rows_materialized": sum(item.rows_materialized for item in roles),
        "worker_cache_bytes_materialized": sum(
            item.bytes_materialized for item in roles
        ),
        "worker_cache_evictions": sum(item.evictions for item in roles),
        "worker_cache_hit_rate": (
            worker_hits / worker_requests if worker_requests else None
        ),
        "worker_cache_hit_rate_status": (MEASURED if worker_requests else UNAVAILABLE),
        "pool_matrix_batch_runs": len(batch_items),
        "pool_matrix_batch_work_items": sum(item.work_items for item in batch_items),
        "pool_matrix_execution_batches": sum(
            item.execution_batches for item in batch_items
        ),
        "pool_matrix_references": sum(
            item.pool_matrix_references for item in batch_items
        ),
        "unique_pool_matrix_observations": sum(
            item.unique_pool_matrices for item in batch_items
        ),
        "unique_pool_matrix_row_observations": sum(
            item.unique_pool_matrix_rows for item in batch_items
        ),
        "unique_pool_matrix_byte_observations": sum(
            item.unique_pool_matrix_bytes for item in batch_items
        ),
        "within_batch_matrix_reuses": within_batch_reuses,
        "cross_batch_matrix_reloads": sum(
            item.cross_batch_matrix_reloads for item in batch_items
        ),
        "maximum_batch_work_items": max(
            (item.maximum_batch_work_items for item in batch_items),
            default=0,
        ),
        "maximum_batch_unique_pool_matrices": max(
            (item.maximum_batch_unique_pool_matrices for item in batch_items),
            default=0,
        ),
        "maximum_batch_pool_matrix_bytes": max(
            (item.maximum_batch_pool_matrix_bytes for item in batch_items),
            default=0,
        ),
        "batch_encoder_invocations": sum(
            item.encoder_invocations for item in batch_items
        ),
        "batch_image_materializations": sum(
            item.image_materializations for item in batch_items
        ),
        "observed_matrix_reuse_events": worker_hits + within_batch_reuses,
        "avoided_matrix_bytes": None,
        "avoided_matrix_bytes_status": NOT_INSTRUMENTED,
        "avoided_matrix_seconds": None,
        "avoided_matrix_seconds_status": NOT_INSTRUMENTED,
        "source_fingerprints": source_fingerprints,
    }
    result = MatrixReuseMetrics(
        **values,
        metrics_fingerprint=canonical_semantic_fingerprint(
            _canonical_metric_values(values)
        ),
    )
    validate_matrix_reuse_metrics(result)
    return result


def validate_matrix_reuse_metrics(metrics: MatrixReuseMetrics) -> None:
    """Reject cache/batch conflation, incomplete arithmetic and guessed savings."""

    if not isinstance(metrics, MatrixReuseMetrics):
        raise TypeError("metrics must be MatrixReuseMetrics")
    if metrics.schema_version != MATRIX_REUSE_METRICS_VERSION:
        raise ValueError("unsupported matrix reuse metrics version")
    roles = (metrics.family, metrics.candidate, metrics.pool)
    expected_roles = ("family", "candidate", "pool")
    for role, expected_role in zip(roles, expected_roles, strict=True):
        _validate_matrix_cache_role_metrics(role, expected_role=expected_role)
    integer_fields = (
        "worker_cache_requests",
        "worker_cache_hits",
        "worker_cache_misses",
        "worker_cache_materializations",
        "worker_cache_rows_materialized",
        "worker_cache_bytes_materialized",
        "worker_cache_evictions",
        "pool_matrix_batch_runs",
        "pool_matrix_batch_work_items",
        "pool_matrix_execution_batches",
        "pool_matrix_references",
        "unique_pool_matrix_observations",
        "unique_pool_matrix_row_observations",
        "unique_pool_matrix_byte_observations",
        "within_batch_matrix_reuses",
        "cross_batch_matrix_reloads",
        "maximum_batch_work_items",
        "maximum_batch_unique_pool_matrices",
        "maximum_batch_pool_matrix_bytes",
        "batch_encoder_invocations",
        "batch_image_materializations",
        "observed_matrix_reuse_events",
    )
    for field in integer_fields:
        _nonnegative_int(getattr(metrics, field), field=field)
    _validate_sum(metrics.worker_cache_requests, roles, "requests")
    _validate_sum(metrics.worker_cache_hits, roles, "hits")
    _validate_sum(metrics.worker_cache_misses, roles, "misses")
    _validate_sum(metrics.worker_cache_materializations, roles, "materializations")
    _validate_sum(metrics.worker_cache_rows_materialized, roles, "rows_materialized")
    _validate_sum(metrics.worker_cache_bytes_materialized, roles, "bytes_materialized")
    _validate_sum(metrics.worker_cache_evictions, roles, "evictions")
    if metrics.worker_cache_requests != (
        metrics.worker_cache_hits + metrics.worker_cache_misses
    ):
        raise ValueError("worker matrix cache request accounting is incomplete")
    _validate_optional_rate(
        value=metrics.worker_cache_hit_rate,
        status=metrics.worker_cache_hit_rate_status,
        numerator=metrics.worker_cache_hits,
        denominator=metrics.worker_cache_requests,
        label="worker matrix cache-hit rate",
    )
    if metrics.pool_matrix_batch_runs == 0 and any(
        getattr(metrics, field)
        for field in (
            "pool_matrix_batch_work_items",
            "pool_matrix_execution_batches",
            "pool_matrix_references",
            "unique_pool_matrix_observations",
            "unique_pool_matrix_row_observations",
            "unique_pool_matrix_byte_observations",
            "within_batch_matrix_reuses",
            "cross_batch_matrix_reloads",
            "maximum_batch_work_items",
            "maximum_batch_unique_pool_matrices",
            "maximum_batch_pool_matrix_bytes",
            "batch_encoder_invocations",
            "batch_image_materializations",
        )
    ):
        raise ValueError("matrix batch metrics exist without a batch observation")
    if metrics.observed_matrix_reuse_events != (
        metrics.worker_cache_hits + metrics.within_batch_matrix_reuses
    ):
        raise ValueError("observed matrix reuse event count mismatch")
    if metrics.batch_encoder_invocations or metrics.batch_image_materializations:
        raise ValueError("matrix batch metrics crossed the encoder-free boundary")
    if (
        metrics.avoided_matrix_bytes is not None
        or metrics.avoided_matrix_bytes_status != NOT_INSTRUMENTED
        or metrics.avoided_matrix_seconds is not None
        or metrics.avoided_matrix_seconds_status != NOT_INSTRUMENTED
    ):
        raise ValueError("unmeasured matrix savings must remain not_instrumented")
    _validate_source_fingerprints(metrics.source_fingerprints, label="matrix reuse")
    expected_fingerprint = canonical_semantic_fingerprint(
        _canonical_metric_values(_metrics_base(metrics))
    )
    if metrics.metrics_fingerprint != expected_fingerprint:
        raise ValueError("matrix reuse metrics fingerprint mismatch")


def _validate_pool_matrix_batch_metrics(metrics: PoolMatrixBatchMetrics) -> None:
    if not isinstance(metrics, PoolMatrixBatchMetrics):
        raise TypeError("pool batch metrics must be PoolMatrixBatchMetrics")
    if metrics.schema_version != POOL_MATRIX_BATCH_METRICS_VERSION:
        raise ValueError("unsupported pool matrix batch metrics version")
    integer_fields = (
        "work_items",
        "execution_batches",
        "pool_matrix_references",
        "unique_pool_matrices",
        "unique_pool_matrix_rows",
        "unique_pool_matrix_bytes",
        "within_batch_matrix_reuses",
        "cross_batch_matrix_reloads",
        "maximum_batch_work_items",
        "maximum_batch_unique_pool_matrices",
        "maximum_batch_pool_matrix_bytes",
        "encoder_invocations",
        "image_materializations",
    )
    for field in integer_fields:
        _nonnegative_int(getattr(metrics, field), field=field)
    if not metrics.work_items or not metrics.execution_batches:
        raise ValueError("pool matrix batch metrics must describe executed work")
    if metrics.unique_pool_matrices > metrics.pool_matrix_references:
        raise ValueError("unique pool matrices exceed observed references")
    if metrics.maximum_batch_work_items > metrics.work_items:
        raise ValueError("maximum batch work exceeds total work")
    if metrics.maximum_batch_unique_pool_matrices > (
        metrics.unique_pool_matrices + metrics.cross_batch_matrix_reloads
    ):
        raise ValueError("maximum batch matrix count exceeds observed matrices")
    if metrics.maximum_batch_pool_matrix_bytes > (metrics.unique_pool_matrix_bytes):
        raise ValueError("maximum batch bytes exceed observed unique matrix bytes")
    if metrics.encoder_invocations or metrics.image_materializations:
        raise ValueError("pool matrix batch metrics crossed the encoder-free boundary")
    expected = canonical_semantic_fingerprint(_metrics_base(metrics))
    if metrics.metrics_fingerprint != expected:
        raise ValueError("pool matrix batch metrics fingerprint mismatch")


def _matrix_cache_role_metrics(
    role: str,
    snapshots: Sequence[MatrixCacheMetrics],
) -> MatrixCacheRoleMetrics:
    for snapshot in snapshots:
        _validate_matrix_cache_snapshot(snapshot)
    requests = sum(item.requests for item in snapshots)
    hits = sum(item.hits for item in snapshots)
    values: dict[str, object] = {
        "schema_version": MATRIX_CACHE_ROLE_METRICS_VERSION,
        "matrix_role": role,
        "worker_snapshots": len(snapshots),
        "requests": requests,
        "hits": hits,
        "misses": sum(item.misses for item in snapshots),
        "materializations": sum(item.materializations for item in snapshots),
        "active_entries": sum(item.entries for item in snapshots),
        "rows_materialized": sum(item.rows_materialized for item in snapshots),
        "bytes_materialized": sum(item.bytes_materialized for item in snapshots),
        "evictions": sum(item.evictions for item in snapshots),
        "hit_rate": hits / requests if requests else None,
        "hit_rate_status": MEASURED if requests else UNAVAILABLE,
    }
    result = MatrixCacheRoleMetrics(
        **values,
        metrics_fingerprint=canonical_semantic_fingerprint(values),
    )
    _validate_matrix_cache_role_metrics(result, expected_role=role)
    return result


def _validate_matrix_cache_snapshot(metrics: MatrixCacheMetrics) -> None:
    if not isinstance(metrics, MatrixCacheMetrics):
        raise TypeError("matrix cache snapshot must be MatrixCacheMetrics")
    for field in (
        "requests",
        "hits",
        "misses",
        "materializations",
        "entries",
        "rows_materialized",
        "bytes_materialized",
        "evictions",
    ):
        _nonnegative_int(getattr(metrics, field), field=field)
    if metrics.requests != metrics.hits + metrics.misses:
        raise ValueError("matrix cache snapshot request accounting is incomplete")
    if metrics.misses != metrics.materializations:
        raise ValueError("matrix cache misses must equal materializations")
    if metrics.entries != metrics.materializations - metrics.evictions:
        raise ValueError("matrix cache active-entry accounting is inconsistent")
    expected_rate = metrics.hits / metrics.requests if metrics.requests else None
    if metrics.hit_rate != expected_rate:
        raise ValueError("matrix cache snapshot hit rate mismatch")


def _validate_matrix_cache_role_metrics(
    metrics: MatrixCacheRoleMetrics,
    *,
    expected_role: str,
) -> None:
    if not isinstance(metrics, MatrixCacheRoleMetrics):
        raise TypeError("matrix role metrics must be MatrixCacheRoleMetrics")
    if metrics.schema_version != MATRIX_CACHE_ROLE_METRICS_VERSION:
        raise ValueError("unsupported matrix cache role metrics version")
    if metrics.matrix_role != expected_role:
        raise ValueError("matrix cache role mismatch")
    for field in (
        "worker_snapshots",
        "requests",
        "hits",
        "misses",
        "materializations",
        "active_entries",
        "rows_materialized",
        "bytes_materialized",
        "evictions",
    ):
        _nonnegative_int(getattr(metrics, field), field=field)
    if metrics.requests != metrics.hits + metrics.misses:
        raise ValueError("matrix role request accounting is incomplete")
    if metrics.misses != metrics.materializations:
        raise ValueError("matrix role misses must equal materializations")
    if metrics.active_entries != metrics.materializations - metrics.evictions:
        raise ValueError("matrix role active-entry accounting is inconsistent")
    _validate_optional_rate(
        value=metrics.hit_rate,
        status=metrics.hit_rate_status,
        numerator=metrics.hits,
        denominator=metrics.requests,
        label=f"{expected_role} matrix cache-hit rate",
    )
    expected = canonical_semantic_fingerprint(_metrics_base(metrics))
    if metrics.metrics_fingerprint != expected:
        raise ValueError("matrix cache role metrics fingerprint mismatch")


def _matrix_cache_source_fingerprint(
    role: str,
    snapshot_index: int,
    metrics: MatrixCacheMetrics,
) -> str:
    _validate_matrix_cache_snapshot(metrics)
    return canonical_semantic_fingerprint(
        {
            "schema_version": "matrix-cache-observation-v1",
            "matrix_role": role,
            "snapshot_index": snapshot_index,
            "requests": metrics.requests,
            "hits": metrics.hits,
            "misses": metrics.misses,
            "materializations": metrics.materializations,
            "entries": metrics.entries,
            "rows_materialized": metrics.rows_materialized,
            "bytes_materialized": metrics.bytes_materialized,
            "evictions": metrics.evictions,
        }
    )


def _typed_sequence(
    values: object,
    *,
    expected_type: type,
    field: str,
) -> tuple:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError(f"{field} must be a sequence")
    items = tuple(values)
    if any(not isinstance(item, expected_type) for item in items):
        raise TypeError(f"{field} contains an invalid item")
    return items


def _validate_sum(
    observed: int,
    roles: Sequence[MatrixCacheRoleMetrics],
    field: str,
) -> None:
    if observed != sum(int(getattr(role, field)) for role in roles):
        raise ValueError(f"worker matrix cache {field} total mismatch")


def _validate_optional_rate(
    *,
    value: float | None,
    status: str,
    numerator: int,
    denominator: int,
    label: str,
) -> None:
    if denominator:
        expected = numerator / denominator
        rate = _bounded_rate(value, field=label)
        if status != MEASURED or not isclose(
            rate,
            expected,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"{label} mismatch")
    elif value is not None or status != UNAVAILABLE:
        raise ValueError(f"unobserved {label} must remain unavailable")


def _validate_source_fingerprints(
    values: tuple[str, ...],
    *,
    label: str,
) -> None:
    if not values:
        raise ValueError(f"{label} metrics require source fingerprints")
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{label} source fingerprints are not canonical")
    for fingerprint in values:
        _sha256(fingerprint, field="source_fingerprint")


def _canonical_metric_values(values: dict[str, object]) -> dict[str, object]:
    return {key: _canonical_metric_value(value) for key, value in values.items()}


def _canonical_metric_value(value: object) -> object:
    if isinstance(value, MatrixCacheRoleMetrics):
        return {
            field.name: _canonical_metric_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_canonical_metric_value(item) for item in value]
    return value


def _metrics_base(metrics: object) -> dict[str, object]:
    return {
        field.name: getattr(metrics, field.name)
        for field in fields(metrics)
        if field.name != "metrics_fingerprint"
    }


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
    return _metrics_base(metrics)


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
    "MATRIX_CACHE_ROLE_METRICS_VERSION",
    "MATRIX_REUSE_METRICS_VERSION",
    "MatrixCacheRoleMetrics",
    "MatrixReuseMetrics",
    "measure_embedding_reuse",
    "measure_matrix_reuse",
    "validate_embedding_reuse_metrics",
    "validate_matrix_reuse_metrics",
]
