"""Deterministic dense-vector caches for dynamic BioCLIP scoring."""

from __future__ import annotations

from array import array
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import fsum, isfinite, sqrt
import re
from threading import Lock

from biominer.bioclip.dynamic_pool_contracts import DYNAMIC_POOL_GEOGRAPHIC_SCOPES
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.readiness import REFERENCE_ROUTES
from biominer.vision.full_frame_attention import (
    FOCUSED_FULL_FRAME_KIND,
    MASKED_FULL_FRAME_KIND,
    MULTI_OBJECT_FULL_FRAME_KIND,
    RAW_FULL_IMAGE_KIND,
)


FAMILY_MATRIX_SIGNATURE_VERSION = "family-prototype-matrix-signature-v1"
CANDIDATE_MATRIX_SIGNATURE_VERSION = "candidate-prototype-matrix-signature-v1"
POOL_MATRIX_SIGNATURE_VERSION = "dynamic-pool-reference-matrix-signature-v1"
CANDIDATE_POOL_SIGNATURE_VERSION = "candidate-pool-signature-v1"
CACHED_VECTOR_MATRIX_VERSION = "cached-vector-matrix-v1"

_VISUAL_INPUT_KINDS = frozenset(
    {
        RAW_FULL_IMAGE_KIND,
        FOCUSED_FULL_FRAME_KIND,
        MASKED_FULL_FRAME_KIND,
        MULTI_OBJECT_FULL_FRAME_KIND,
    }
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_UNIT_NORM_TOLERANCE = 1e-5


@dataclass(frozen=True, slots=True)
class FamilyPrototypeVector:
    """One family prototype bound to its durable evidence identity."""

    family_key: str
    family_name: str
    prototype_fingerprint: str
    embedding: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family_key",
            _required_text(self.family_key, field="family_key"),
        )
        object.__setattr__(
            self,
            "family_name",
            _required_text(self.family_name, field="family_name"),
        )
        object.__setattr__(
            self,
            "prototype_fingerprint",
            _sha256(self.prototype_fingerprint, field="prototype_fingerprint"),
        )
        object.__setattr__(
            self,
            "embedding",
            _unit_float32_vector(self.embedding, field="family prototype embedding"),
        )


@dataclass(frozen=True, slots=True)
class CandidatePrototypeVector:
    """One candidate prototype already derived from frozen reference evidence."""

    accepted_taxon_key: str
    scientific_name: str
    prototype_fingerprint: str
    embedding: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "accepted_taxon_key",
            _required_text(self.accepted_taxon_key, field="accepted_taxon_key"),
        )
        object.__setattr__(
            self,
            "scientific_name",
            _required_text(self.scientific_name, field="scientific_name"),
        )
        object.__setattr__(
            self,
            "prototype_fingerprint",
            _sha256(self.prototype_fingerprint, field="prototype_fingerprint"),
        )
        object.__setattr__(
            self,
            "embedding",
            _unit_float32_vector(
                self.embedding,
                field="candidate prototype embedding",
            ),
        )


@dataclass(frozen=True, slots=True)
class PoolReferenceVector:
    """One cached reference vector selected into an immutable dynamic pool."""

    reference_media_id: str
    reference_observation_id: str
    member_fingerprint: str
    reference_embedding_fingerprint: str
    embedding: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_media_id",
            _required_text(self.reference_media_id, field="reference_media_id"),
        )
        object.__setattr__(
            self,
            "reference_observation_id",
            _required_text(
                self.reference_observation_id,
                field="reference_observation_id",
            ),
        )
        object.__setattr__(
            self,
            "member_fingerprint",
            _sha256(self.member_fingerprint, field="member_fingerprint"),
        )
        object.__setattr__(
            self,
            "reference_embedding_fingerprint",
            _sha256(
                self.reference_embedding_fingerprint,
                field="reference_embedding_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "embedding",
            _unit_float32_vector(self.embedding, field="pool reference embedding"),
        )


@dataclass(frozen=True, slots=True)
class CachedVectorMatrix:
    """Immutable row-major Float32 matrix with canonical row identities."""

    matrix_kind: str
    matrix_signature: str
    route: str
    visual_input_kind: str
    partition: str
    source_fingerprint: str
    model_fingerprint: str
    row_ids: tuple[str, ...]
    row_names: tuple[str, ...]
    row_fingerprints: tuple[str, ...]
    embedding_dimension: int
    _float32_bytes: bytes
    subject_id: str | None = None

    @property
    def row_count(self) -> int:
        return len(self.row_ids)

    @property
    def byte_count(self) -> int:
        return len(self._float32_bytes)

    @property
    def float32_buffer(self) -> memoryview:
        """Return a read-only, contiguous native-Float32 view."""

        return memoryview(self._float32_bytes).cast("f")

    def vector(self, row_index: int) -> tuple[float, ...]:
        """Return one matrix row without exposing mutable cache storage."""

        if isinstance(row_index, bool) or not isinstance(row_index, int):
            raise TypeError("matrix row_index must be an integer")
        if not 0 <= row_index < self.row_count:
            raise IndexError("matrix row_index is out of range")
        start = row_index * self.embedding_dimension
        values = self.float32_buffer[start : start + self.embedding_dimension]
        return tuple(float(value) for value in values)


@dataclass(frozen=True, slots=True)
class MatrixCacheMetrics:
    """Measured family-matrix cache activity for one worker lifetime."""

    requests: int
    hits: int
    misses: int
    materializations: int
    entries: int
    rows_materialized: int
    bytes_materialized: int
    evictions: int

    @property
    def hit_rate(self) -> float | None:
        return self.hits / self.requests if self.requests else None

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "family_matrix_requests": self.requests,
            "family_matrix_cache_hits": self.hits,
            "family_matrix_cache_misses": self.misses,
            "family_matrix_materializations": self.materializations,
            "family_matrix_cache_entries": self.entries,
            "family_matrix_rows_materialized": self.rows_materialized,
            "family_matrix_bytes_materialized": self.bytes_materialized,
            "family_matrix_cache_evictions": self.evictions,
            "family_matrix_cache_hit_rate": self.hit_rate,
        }


@dataclass(frozen=True, slots=True)
class DynamicPoolMatrixCacheMetrics:
    """Separate candidate and reference-pool cache measurements."""

    candidate: MatrixCacheMetrics
    pool: MatrixCacheMetrics

    @property
    def requests(self) -> int:
        return self.candidate.requests + self.pool.requests

    @property
    def hits(self) -> int:
        return self.candidate.hits + self.pool.hits

    @property
    def misses(self) -> int:
        return self.candidate.misses + self.pool.misses

    @property
    def hit_rate(self) -> float | None:
        return self.hits / self.requests if self.requests else None

    def as_dict(self) -> dict[str, int | float | None]:
        values: dict[str, int | float | None] = {}
        values.update(_cache_metric_values("candidate_matrix", self.candidate))
        values.update(_cache_metric_values("pool_matrix", self.pool))
        values.update(
            {
                "dynamic_matrix_requests": self.requests,
                "dynamic_matrix_cache_hits": self.hits,
                "dynamic_matrix_cache_misses": self.misses,
                "dynamic_matrix_cache_hit_rate": self.hit_rate,
                "dynamic_matrix_materializations": (
                    self.candidate.materializations + self.pool.materializations
                ),
                "dynamic_matrix_cache_entries": (
                    self.candidate.entries + self.pool.entries
                ),
                "dynamic_matrix_rows_materialized": (
                    self.candidate.rows_materialized + self.pool.rows_materialized
                ),
                "dynamic_matrix_bytes_materialized": (
                    self.candidate.bytes_materialized + self.pool.bytes_materialized
                ),
                "dynamic_matrix_cache_evictions": (
                    self.candidate.evictions + self.pool.evictions
                ),
            }
        )
        return values


class _MatrixStore:
    """Thread-safe bounded storage shared by semantic matrix indexes."""

    def __init__(self, maximum_entries: int) -> None:
        self.maximum_entries = maximum_entries
        self.entries: OrderedDict[str, CachedVectorMatrix] = OrderedDict()
        self.lock = Lock()
        self.requests = 0
        self.hits = 0
        self.misses = 0
        self.materializations = 0
        self.rows_materialized = 0
        self.bytes_materialized = 0
        self.evictions = 0

    def get_or_build(
        self,
        signature: str,
        builder: Callable[[], CachedVectorMatrix],
    ) -> CachedVectorMatrix:
        with self.lock:
            self.requests += 1
            cached = self.entries.get(signature)
            if cached is not None:
                self.hits += 1
                self.entries.move_to_end(signature)
                return cached
            self.misses += 1
            matrix = builder()
            if not isinstance(matrix, CachedVectorMatrix):
                raise TypeError("matrix builder must return CachedVectorMatrix")
            if matrix.matrix_signature != signature:
                raise ValueError("matrix builder returned a conflicting signature")
            self.entries[signature] = matrix
            self.materializations += 1
            self.rows_materialized += matrix.row_count
            self.bytes_materialized += matrix.byte_count
            if len(self.entries) > self.maximum_entries:
                self.entries.popitem(last=False)
                self.evictions += 1
            return matrix

    def metrics(self) -> MatrixCacheMetrics:
        with self.lock:
            return MatrixCacheMetrics(
                requests=self.requests,
                hits=self.hits,
                misses=self.misses,
                materializations=self.materializations,
                entries=len(self.entries),
                rows_materialized=self.rows_materialized,
                bytes_materialized=self.bytes_materialized,
                evictions=self.evictions,
            )


class FamilyPrototypeMatrixCache:
    """Bounded worker-local cache of canonical family prototype matrices."""

    def __init__(self, *, maximum_entries: int = 64) -> None:
        self.maximum_entries = _positive_integer(
            maximum_entries,
            field="maximum_entries",
        )
        self._store = _MatrixStore(self.maximum_entries)

    def get_or_build(
        self,
        *,
        route: str,
        visual_input_kind: str,
        family_partition: str,
        model_fingerprint: str,
        family_prototype_set_fingerprint: str,
        prototypes: Sequence[FamilyPrototypeVector],
    ) -> CachedVectorMatrix:
        """Return the one matrix for a complete family scoring identity."""

        request = _family_matrix_request(
            route=route,
            visual_input_kind=visual_input_kind,
            family_partition=family_partition,
            model_fingerprint=model_fingerprint,
            family_prototype_set_fingerprint=family_prototype_set_fingerprint,
            prototypes=prototypes,
        )
        signature = str(request["matrix_signature"])
        return self._store.get_or_build(
            signature,
            lambda: _materialize_family_matrix(request),
        )

    def cache_metrics(self) -> MatrixCacheMetrics:
        """Snapshot measured cache counters without resetting them."""

        return self._store.metrics()


class DynamicPoolMatrixCache:
    """Worker-local candidate and pool matrix indexes over cached vectors."""

    def __init__(
        self,
        *,
        maximum_candidate_entries: int = 256,
        maximum_pool_entries: int = 2048,
    ) -> None:
        self.maximum_candidate_entries = _positive_integer(
            maximum_candidate_entries,
            field="maximum_candidate_entries",
        )
        self.maximum_pool_entries = _positive_integer(
            maximum_pool_entries,
            field="maximum_pool_entries",
        )
        self._candidate_store = _MatrixStore(self.maximum_candidate_entries)
        self._pool_store = _MatrixStore(self.maximum_pool_entries)

    def get_candidate_matrix(
        self,
        *,
        route: str,
        visual_input_kind: str,
        family_partition: str,
        model_fingerprint: str,
        candidate_set_fingerprint: str,
        reference_prototype_artifact_fingerprint: str,
        candidates: Sequence[CandidatePrototypeVector],
    ) -> CachedVectorMatrix:
        """Return a matrix for the complete candidate prototype signature."""

        request = _candidate_matrix_request(
            route=route,
            visual_input_kind=visual_input_kind,
            family_partition=family_partition,
            model_fingerprint=model_fingerprint,
            candidate_set_fingerprint=candidate_set_fingerprint,
            reference_prototype_artifact_fingerprint=(
                reference_prototype_artifact_fingerprint
            ),
            candidates=candidates,
        )
        signature = str(request["matrix_signature"])
        return self._candidate_store.get_or_build(
            signature,
            lambda: _materialize_candidate_matrix(request),
        )

    def get_pool_matrix(
        self,
        *,
        route: str,
        visual_input_kind: str,
        geographic_scope: str,
        candidate_accepted_taxon_key: str,
        model_fingerprint: str,
        reference_embedding_artifact_fingerprint: str,
        pool_membership_fingerprint: str,
        pool_ids: Sequence[str],
        references: Sequence[PoolReferenceVector],
    ) -> CachedVectorMatrix:
        """Return a matrix for exact global/local dynamic-pool membership."""

        request = _pool_matrix_request(
            route=route,
            visual_input_kind=visual_input_kind,
            geographic_scope=geographic_scope,
            candidate_accepted_taxon_key=candidate_accepted_taxon_key,
            model_fingerprint=model_fingerprint,
            reference_embedding_artifact_fingerprint=(
                reference_embedding_artifact_fingerprint
            ),
            pool_membership_fingerprint=pool_membership_fingerprint,
            pool_ids=pool_ids,
            references=references,
        )
        signature = str(request["matrix_signature"])
        return self._pool_store.get_or_build(
            signature,
            lambda: _materialize_pool_matrix(request),
        )

    def cache_metrics(self) -> DynamicPoolMatrixCacheMetrics:
        """Snapshot candidate and pool cache counters independently."""

        return DynamicPoolMatrixCacheMetrics(
            candidate=self._candidate_store.metrics(),
            pool=self._pool_store.metrics(),
        )


def family_matrix_signature(
    *,
    route: str,
    visual_input_kind: str,
    family_partition: str,
    model_fingerprint: str,
    family_prototype_set_fingerprint: str,
    prototypes: Sequence[FamilyPrototypeVector],
) -> str:
    """Derive the reusable matrix signature without populating a cache."""

    return str(
        _family_matrix_request(
            route=route,
            visual_input_kind=visual_input_kind,
            family_partition=family_partition,
            model_fingerprint=model_fingerprint,
            family_prototype_set_fingerprint=family_prototype_set_fingerprint,
            prototypes=prototypes,
        )["matrix_signature"]
    )


def candidate_matrix_signature(
    *,
    route: str,
    visual_input_kind: str,
    family_partition: str,
    model_fingerprint: str,
    candidate_set_fingerprint: str,
    reference_prototype_artifact_fingerprint: str,
    candidates: Sequence[CandidatePrototypeVector],
) -> str:
    """Derive a candidate-prototype matrix signature without caching it."""

    return str(
        _candidate_matrix_request(
            route=route,
            visual_input_kind=visual_input_kind,
            family_partition=family_partition,
            model_fingerprint=model_fingerprint,
            candidate_set_fingerprint=candidate_set_fingerprint,
            reference_prototype_artifact_fingerprint=(
                reference_prototype_artifact_fingerprint
            ),
            candidates=candidates,
        )["matrix_signature"]
    )


def pool_matrix_signature(
    *,
    route: str,
    visual_input_kind: str,
    geographic_scope: str,
    candidate_accepted_taxon_key: str,
    model_fingerprint: str,
    reference_embedding_artifact_fingerprint: str,
    pool_membership_fingerprint: str,
    pool_ids: Sequence[str],
    references: Sequence[PoolReferenceVector],
) -> str:
    """Derive an exact reference-pool matrix signature without caching it."""

    return str(
        _pool_matrix_request(
            route=route,
            visual_input_kind=visual_input_kind,
            geographic_scope=geographic_scope,
            candidate_accepted_taxon_key=candidate_accepted_taxon_key,
            model_fingerprint=model_fingerprint,
            reference_embedding_artifact_fingerprint=(
                reference_embedding_artifact_fingerprint
            ),
            pool_membership_fingerprint=pool_membership_fingerprint,
            pool_ids=pool_ids,
            references=references,
        )["matrix_signature"]
    )


def candidate_pool_signature(
    candidate_signature: str,
    pool_signatures: Sequence[str],
) -> str:
    """Bind one candidate matrix to its exact set of pool matrices."""

    candidate = _sha256(candidate_signature, field="candidate_signature")
    pools = _canonical_nonempty_sha256_values(
        pool_signatures,
        field="pool_signatures",
    )
    return canonical_semantic_fingerprint(
        {
            "schema_version": CANDIDATE_POOL_SIGNATURE_VERSION,
            "candidate_matrix_signature": candidate,
            "pool_matrix_signatures": list(pools),
        }
    )


def _candidate_matrix_request(
    *,
    route: str,
    visual_input_kind: str,
    family_partition: str,
    model_fingerprint: str,
    candidate_set_fingerprint: str,
    reference_prototype_artifact_fingerprint: str,
    candidates: Sequence[CandidatePrototypeVector],
) -> dict[str, object]:
    route_value = _route(route)
    input_kind = _visual_input_kind(visual_input_kind)
    partition = _required_text(family_partition, field="family_partition")
    model = _sha256(model_fingerprint, field="model_fingerprint")
    candidate_set = _sha256(
        candidate_set_fingerprint,
        field="candidate_set_fingerprint",
    )
    prototype_artifact = _sha256(
        reference_prototype_artifact_fingerprint,
        field="reference_prototype_artifact_fingerprint",
    )
    ordered = _canonical_vector_rows(
        candidates,
        expected_type=CandidatePrototypeVector,
        label="candidate prototypes",
        row_key=lambda row: (row.accepted_taxon_key, row.scientific_name),
        unique_key=lambda row: row.accepted_taxon_key,
        duplicate_message="candidate matrix contains duplicate accepted taxon keys",
    )
    dimensions = {len(row.embedding) for row in ordered}
    if len(dimensions) != 1:
        raise ValueError("candidate matrix contains mixed embedding dimensions")
    semantic_rows = [
        {
            "accepted_taxon_key": row.accepted_taxon_key,
            "scientific_name": row.scientific_name,
            "prototype_fingerprint": row.prototype_fingerprint,
            "embedding": list(row.embedding),
        }
        for row in ordered
    ]
    payload = {
        "schema_version": CANDIDATE_MATRIX_SIGNATURE_VERSION,
        "route": route_value,
        "visual_input_kind": input_kind,
        "family_partition": partition,
        "model_fingerprint": model,
        "candidate_set_fingerprint": candidate_set,
        "reference_prototype_artifact_fingerprint": prototype_artifact,
        "embedding_dimension": next(iter(dimensions)),
        "rows": semantic_rows,
    }
    return {
        **payload,
        "matrix_signature": canonical_semantic_fingerprint(payload),
        "candidates": ordered,
    }


def _pool_matrix_request(
    *,
    route: str,
    visual_input_kind: str,
    geographic_scope: str,
    candidate_accepted_taxon_key: str,
    model_fingerprint: str,
    reference_embedding_artifact_fingerprint: str,
    pool_membership_fingerprint: str,
    pool_ids: Sequence[str],
    references: Sequence[PoolReferenceVector],
) -> dict[str, object]:
    route_value = _route(route)
    input_kind = _visual_input_kind(visual_input_kind)
    scope = _geographic_scope(geographic_scope)
    candidate_key = _required_text(
        candidate_accepted_taxon_key,
        field="candidate_accepted_taxon_key",
    )
    model = _sha256(model_fingerprint, field="model_fingerprint")
    reference_artifact = _sha256(
        reference_embedding_artifact_fingerprint,
        field="reference_embedding_artifact_fingerprint",
    )
    membership = _sha256(
        pool_membership_fingerprint,
        field="pool_membership_fingerprint",
    )
    pools = _canonical_nonempty_text_values(pool_ids, field="pool_ids")
    ordered = _canonical_vector_rows(
        references,
        expected_type=PoolReferenceVector,
        label="pool references",
        row_key=lambda row: (
            row.reference_media_id,
            row.reference_observation_id,
            row.member_fingerprint,
        ),
        unique_key=lambda row: row.reference_media_id,
        duplicate_message="pool matrix contains duplicate reference media IDs",
    )
    dimensions = {len(row.embedding) for row in ordered}
    if len(dimensions) != 1:
        raise ValueError("pool matrix contains mixed embedding dimensions")
    semantic_rows = [
        {
            "reference_media_id": row.reference_media_id,
            "reference_observation_id": row.reference_observation_id,
            "member_fingerprint": row.member_fingerprint,
            "reference_embedding_fingerprint": (row.reference_embedding_fingerprint),
            "embedding": list(row.embedding),
        }
        for row in ordered
    ]
    payload = {
        "schema_version": POOL_MATRIX_SIGNATURE_VERSION,
        "route": route_value,
        "visual_input_kind": input_kind,
        "geographic_scope": scope,
        "candidate_accepted_taxon_key": candidate_key,
        "model_fingerprint": model,
        "reference_embedding_artifact_fingerprint": reference_artifact,
        "pool_membership_fingerprint": membership,
        "pool_ids": list(pools),
        "embedding_dimension": next(iter(dimensions)),
        "rows": semantic_rows,
    }
    return {
        **payload,
        "matrix_signature": canonical_semantic_fingerprint(payload),
        "references": ordered,
    }


def _family_matrix_request(
    *,
    route: str,
    visual_input_kind: str,
    family_partition: str,
    model_fingerprint: str,
    family_prototype_set_fingerprint: str,
    prototypes: Sequence[FamilyPrototypeVector],
) -> dict[str, object]:
    route_value = _route(route)
    input_kind = _visual_input_kind(visual_input_kind)
    partition = _required_text(family_partition, field="family_partition")
    model = _sha256(model_fingerprint, field="model_fingerprint")
    source = _sha256(
        family_prototype_set_fingerprint,
        field="family_prototype_set_fingerprint",
    )
    if isinstance(prototypes, str | bytes) or not isinstance(prototypes, Sequence):
        raise TypeError("family prototypes must be a sequence")
    rows = tuple(prototypes)
    if not rows:
        raise ValueError("family prototype matrix requires at least one row")
    if any(not isinstance(row, FamilyPrototypeVector) for row in rows):
        raise TypeError("family prototypes must contain FamilyPrototypeVector values")
    ordered = tuple(sorted(rows, key=lambda row: (row.family_key, row.family_name)))
    keys = [row.family_key for row in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("family prototype matrix contains duplicate family keys")
    dimensions = {len(row.embedding) for row in ordered}
    if len(dimensions) != 1:
        raise ValueError("family prototype matrix contains mixed embedding dimensions")
    semantic_rows = [
        {
            "family_key": row.family_key,
            "family_name": row.family_name,
            "prototype_fingerprint": row.prototype_fingerprint,
            "embedding": list(row.embedding),
        }
        for row in ordered
    ]
    payload = {
        "schema_version": FAMILY_MATRIX_SIGNATURE_VERSION,
        "route": route_value,
        "visual_input_kind": input_kind,
        "family_partition": partition,
        "model_fingerprint": model,
        "family_prototype_set_fingerprint": source,
        "embedding_dimension": next(iter(dimensions)),
        "rows": semantic_rows,
    }
    return {
        **payload,
        "matrix_signature": canonical_semantic_fingerprint(payload),
        "prototypes": ordered,
    }


def _materialize_family_matrix(request: dict[str, object]) -> CachedVectorMatrix:
    rows = request["prototypes"]
    if not isinstance(rows, tuple):
        raise AssertionError("normalized family matrix rows must be a tuple")
    vectors = array("f")
    for row in rows:
        if not isinstance(row, FamilyPrototypeVector):
            raise AssertionError("normalized family matrix row has an invalid type")
        vectors.extend(row.embedding)
    return CachedVectorMatrix(
        matrix_kind="family_prototype",
        matrix_signature=str(request["matrix_signature"]),
        route=str(request["route"]),
        visual_input_kind=str(request["visual_input_kind"]),
        partition=str(request["family_partition"]),
        source_fingerprint=str(request["family_prototype_set_fingerprint"]),
        model_fingerprint=str(request["model_fingerprint"]),
        row_ids=tuple(row.family_key for row in rows),
        row_names=tuple(row.family_name for row in rows),
        row_fingerprints=tuple(row.prototype_fingerprint for row in rows),
        embedding_dimension=int(request["embedding_dimension"]),
        _float32_bytes=vectors.tobytes(),
    )


def _materialize_candidate_matrix(request: dict[str, object]) -> CachedVectorMatrix:
    rows = request["candidates"]
    if not isinstance(rows, tuple):
        raise AssertionError("normalized candidate matrix rows must be a tuple")
    vectors = array("f")
    for row in rows:
        if not isinstance(row, CandidatePrototypeVector):
            raise AssertionError("normalized candidate matrix row has an invalid type")
        vectors.extend(row.embedding)
    return CachedVectorMatrix(
        matrix_kind="candidate_prototype",
        matrix_signature=str(request["matrix_signature"]),
        route=str(request["route"]),
        visual_input_kind=str(request["visual_input_kind"]),
        partition=str(request["family_partition"]),
        source_fingerprint=str(request["candidate_set_fingerprint"]),
        model_fingerprint=str(request["model_fingerprint"]),
        row_ids=tuple(row.accepted_taxon_key for row in rows),
        row_names=tuple(row.scientific_name for row in rows),
        row_fingerprints=tuple(row.prototype_fingerprint for row in rows),
        embedding_dimension=int(request["embedding_dimension"]),
        _float32_bytes=vectors.tobytes(),
    )


def _materialize_pool_matrix(request: dict[str, object]) -> CachedVectorMatrix:
    rows = request["references"]
    if not isinstance(rows, tuple):
        raise AssertionError("normalized pool matrix rows must be a tuple")
    vectors = array("f")
    for row in rows:
        if not isinstance(row, PoolReferenceVector):
            raise AssertionError("normalized pool matrix row has an invalid type")
        vectors.extend(row.embedding)
    return CachedVectorMatrix(
        matrix_kind="dynamic_reference_pool",
        matrix_signature=str(request["matrix_signature"]),
        route=str(request["route"]),
        visual_input_kind=str(request["visual_input_kind"]),
        partition=str(request["geographic_scope"]),
        source_fingerprint=str(request["pool_membership_fingerprint"]),
        model_fingerprint=str(request["model_fingerprint"]),
        row_ids=tuple(row.reference_media_id for row in rows),
        row_names=tuple(row.reference_observation_id for row in rows),
        row_fingerprints=tuple(row.reference_embedding_fingerprint for row in rows),
        embedding_dimension=int(request["embedding_dimension"]),
        _float32_bytes=vectors.tobytes(),
        subject_id=str(request["candidate_accepted_taxon_key"]),
    )


def _canonical_vector_rows(
    values: object,
    *,
    expected_type: type,
    label: str,
    row_key: object,
    unique_key: object,
    duplicate_message: str,
) -> tuple[object, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence")
    rows = tuple(values)
    if not rows:
        raise ValueError(f"{label} must contain at least one row")
    if any(not isinstance(row, expected_type) for row in rows):
        raise TypeError(f"{label} contain invalid row types")
    if not callable(row_key) or not callable(unique_key):
        raise TypeError("matrix row key functions must be callable")
    ordered = tuple(sorted(rows, key=row_key))
    keys = [unique_key(row) for row in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError(duplicate_message)
    return ordered


def _canonical_nonempty_text_values(
    values: Sequence[str],
    *,
    field: str,
) -> tuple[str, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError(f"{field} must be a sequence")
    normalized = tuple(_required_text(value, field=field) for value in values)
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} must contain unique values")
    return tuple(sorted(normalized))


def _canonical_nonempty_sha256_values(
    values: Sequence[str],
    *,
    field: str,
) -> tuple[str, ...]:
    normalized = _canonical_nonempty_text_values(values, field=field)
    return tuple(_sha256(value, field=field) for value in normalized)


def _cache_metric_values(
    prefix: str,
    metrics: MatrixCacheMetrics,
) -> dict[str, int | float | None]:
    return {
        f"{prefix}_requests": metrics.requests,
        f"{prefix}_cache_hits": metrics.hits,
        f"{prefix}_cache_misses": metrics.misses,
        f"{prefix}_materializations": metrics.materializations,
        f"{prefix}_cache_entries": metrics.entries,
        f"{prefix}_rows_materialized": metrics.rows_materialized,
        f"{prefix}_bytes_materialized": metrics.bytes_materialized,
        f"{prefix}_cache_evictions": metrics.evictions,
        f"{prefix}_cache_hit_rate": metrics.hit_rate,
    }


def _unit_float32_vector(values: Sequence[float], *, field: str) -> tuple[float, ...]:
    if isinstance(values, str | bytes):
        raise TypeError(f"{field} must be a numeric sequence")
    vector = array("f")
    for raw in values:
        if isinstance(raw, bool):
            raise ValueError(f"{field} must contain finite numeric values")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must contain finite numeric values") from exc
        if not isfinite(value):
            raise ValueError(f"{field} must contain finite numeric values")
        vector.append(value)
    if not vector:
        raise ValueError(f"{field} must not be empty")
    result = tuple(float(value) for value in vector)
    norm = sqrt(fsum(value * value for value in result))
    if abs(norm - 1.0) > _UNIT_NORM_TOLERANCE:
        raise ValueError(f"{field} must be unit-normalized")
    return result


def _route(value: object) -> str:
    route = _required_text(value, field="route")
    if route not in REFERENCE_ROUTES:
        raise ValueError(f"unsupported matrix route: {route}")
    return route


def _visual_input_kind(value: object) -> str:
    kind = _required_text(value, field="visual_input_kind")
    if kind not in _VISUAL_INPUT_KINDS:
        raise ValueError(f"unsupported matrix visual_input_kind: {kind}")
    return kind


def _geographic_scope(value: object) -> str:
    scope = _required_text(value, field="geographic_scope")
    if scope not in DYNAMIC_POOL_GEOGRAPHIC_SCOPES:
        raise ValueError(f"unsupported matrix geographic_scope: {scope}")
    return scope


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank")
    return text


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a canonical sha256 fingerprint")
    return text


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


__all__ = [
    "CANDIDATE_MATRIX_SIGNATURE_VERSION",
    "CANDIDATE_POOL_SIGNATURE_VERSION",
    "CACHED_VECTOR_MATRIX_VERSION",
    "FAMILY_MATRIX_SIGNATURE_VERSION",
    "POOL_MATRIX_SIGNATURE_VERSION",
    "CachedVectorMatrix",
    "CandidatePrototypeVector",
    "DynamicPoolMatrixCache",
    "DynamicPoolMatrixCacheMetrics",
    "FamilyPrototypeMatrixCache",
    "FamilyPrototypeVector",
    "MatrixCacheMetrics",
    "PoolReferenceVector",
    "candidate_matrix_signature",
    "candidate_pool_signature",
    "family_matrix_signature",
    "pool_matrix_signature",
]
