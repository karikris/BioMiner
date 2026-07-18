"""Deterministic dense-vector caches for dynamic BioCLIP scoring."""

from __future__ import annotations

from array import array
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isfinite, sqrt
import re
from threading import Lock

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.readiness import REFERENCE_ROUTES
from biominer.vision.full_frame_attention import (
    FOCUSED_FULL_FRAME_KIND,
    MASKED_FULL_FRAME_KIND,
    MULTI_OBJECT_FULL_FRAME_KIND,
    RAW_FULL_IMAGE_KIND,
)


FAMILY_MATRIX_SIGNATURE_VERSION = "family-prototype-matrix-signature-v1"
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


class FamilyPrototypeMatrixCache:
    """Bounded worker-local cache of canonical family prototype matrices."""

    def __init__(self, *, maximum_entries: int = 64) -> None:
        self.maximum_entries = _positive_integer(
            maximum_entries,
            field="maximum_entries",
        )
        self._entries: OrderedDict[str, CachedVectorMatrix] = OrderedDict()
        self._lock = Lock()
        self._requests = 0
        self._hits = 0
        self._misses = 0
        self._materializations = 0
        self._rows_materialized = 0
        self._bytes_materialized = 0
        self._evictions = 0

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
        with self._lock:
            self._requests += 1
            cached = self._entries.get(signature)
            if cached is not None:
                self._hits += 1
                self._entries.move_to_end(signature)
                return cached
            self._misses += 1
            matrix = _materialize_family_matrix(request)
            self._entries[signature] = matrix
            self._materializations += 1
            self._rows_materialized += matrix.row_count
            self._bytes_materialized += matrix.byte_count
            if len(self._entries) > self.maximum_entries:
                self._entries.popitem(last=False)
                self._evictions += 1
            return matrix

    def cache_metrics(self) -> MatrixCacheMetrics:
        """Snapshot measured cache counters without resetting them."""

        with self._lock:
            return MatrixCacheMetrics(
                requests=self._requests,
                hits=self._hits,
                misses=self._misses,
                materializations=self._materializations,
                entries=len(self._entries),
                rows_materialized=self._rows_materialized,
                bytes_materialized=self._bytes_materialized,
                evictions=self._evictions,
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
    "CACHED_VECTOR_MATRIX_VERSION",
    "FAMILY_MATRIX_SIGNATURE_VERSION",
    "CachedVectorMatrix",
    "FamilyPrototypeMatrixCache",
    "FamilyPrototypeVector",
    "MatrixCacheMetrics",
    "family_matrix_signature",
]
