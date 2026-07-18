"""Raw, non-probabilistic component scoring over cached BioCLIP matrices."""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isfinite, sqrt
import re

from biominer.bioclip.matrix_cache import CachedVectorMatrix
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.readiness import REFERENCE_ROUTES
from biominer.vision.full_frame_attention import (
    FOCUSED_FULL_FRAME_KIND,
    MASKED_FULL_FRAME_KIND,
    MULTI_OBJECT_FULL_FRAME_KIND,
    RAW_FULL_IMAGE_KIND,
)


RAW_SCORING_QUERY_VERSION = "raw-dynamic-pool-query-v1"
RAW_FAMILY_EVIDENCE_VERSION = "raw-family-evidence-v1"
RAW_FAMILY_EVIDENCE_SET_VERSION = "raw-family-evidence-set-v1"

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
class RawScoringQuery:
    """One cached unit query vector and its complete scoring contract."""

    query_id: str
    query_embedding_fingerprint: str
    route: str
    visual_input_kind: str
    model_fingerprint: str
    embedding: tuple[float, ...]
    query_fingerprint: str | None = None

    def __post_init__(self) -> None:
        query_id = _required_text(self.query_id, field="query_id")
        embedding_fingerprint = _sha256(
            self.query_embedding_fingerprint,
            field="query_embedding_fingerprint",
        )
        route = _route(self.route)
        input_kind = _visual_input_kind(self.visual_input_kind)
        model = _sha256(self.model_fingerprint, field="model_fingerprint")
        embedding = _unit_float32_vector(self.embedding, field="query embedding")
        expected = canonical_semantic_fingerprint(
            {
                "schema_version": RAW_SCORING_QUERY_VERSION,
                "query_id": query_id,
                "query_embedding_fingerprint": embedding_fingerprint,
                "route": route,
                "visual_input_kind": input_kind,
                "model_fingerprint": model,
                "embedding": list(embedding),
            }
        )
        if (
            self.query_fingerprint is not None
            and _sha256(
                self.query_fingerprint,
                field="query_fingerprint",
            )
            != expected
        ):
            raise ValueError("query_fingerprint does not match raw scoring query")
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(
            self,
            "query_embedding_fingerprint",
            embedding_fingerprint,
        )
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "visual_input_kind", input_kind)
        object.__setattr__(self, "model_fingerprint", model)
        object.__setattr__(self, "embedding", embedding)
        object.__setattr__(self, "query_fingerprint", expected)


@dataclass(frozen=True, slots=True)
class RawFamilyEvidence:
    """One family prototype cosine and deterministic raw rank."""

    schema_version: str
    query_id: str
    family_matrix_signature: str
    family_partition: str
    family_key: str
    family_name: str
    family_prototype_fingerprint: str
    raw_similarity: float
    family_rank: int
    margin_to_next_raw: float | None
    score_fingerprint: str


@dataclass(frozen=True, slots=True)
class RawFamilyEvidenceSet:
    """Complete family matrix result; no row is pruned by its score."""

    schema_version: str
    query_id: str
    query_fingerprint: str
    family_matrix_signature: str
    family_partition: str
    scores: tuple[RawFamilyEvidence, ...]
    score_set_fingerprint: str


def score_family_evidence(
    query: RawScoringQuery,
    family_matrix: CachedVectorMatrix,
) -> RawFamilyEvidenceSet:
    """Score every family row as raw cosine evidence without candidate pruning."""

    _validate_query_matrix(
        query,
        family_matrix,
        expected_kind="family_prototype",
    )
    scored = sorted(
        zip(
            _matrix_raw_cosines(query.embedding, family_matrix),
            family_matrix.row_ids,
            family_matrix.row_names,
            family_matrix.row_fingerprints,
            strict=True,
        ),
        key=lambda item: (-item[0], item[1]),
    )
    scores: list[RawFamilyEvidence] = []
    for index, (similarity, family_key, family_name, row_fingerprint) in enumerate(
        scored
    ):
        margin = similarity - scored[index + 1][0] if index + 1 < len(scored) else None
        base = {
            "schema_version": RAW_FAMILY_EVIDENCE_VERSION,
            "query_id": query.query_id,
            "query_fingerprint": query.query_fingerprint,
            "family_matrix_signature": family_matrix.matrix_signature,
            "family_partition": family_matrix.partition,
            "family_key": family_key,
            "family_name": family_name,
            "family_prototype_fingerprint": row_fingerprint,
            "raw_similarity": similarity,
            "family_rank": index + 1,
            "margin_to_next_raw": margin,
        }
        scores.append(
            RawFamilyEvidence(
                schema_version=RAW_FAMILY_EVIDENCE_VERSION,
                query_id=query.query_id,
                family_matrix_signature=family_matrix.matrix_signature,
                family_partition=family_matrix.partition,
                family_key=family_key,
                family_name=family_name,
                family_prototype_fingerprint=row_fingerprint,
                raw_similarity=similarity,
                family_rank=index + 1,
                margin_to_next_raw=margin,
                score_fingerprint=canonical_semantic_fingerprint(base),
            )
        )
    score_set_base = {
        "schema_version": RAW_FAMILY_EVIDENCE_SET_VERSION,
        "query_id": query.query_id,
        "query_fingerprint": query.query_fingerprint,
        "family_matrix_signature": family_matrix.matrix_signature,
        "family_partition": family_matrix.partition,
        "score_fingerprints": [score.score_fingerprint for score in scores],
    }
    return RawFamilyEvidenceSet(
        schema_version=RAW_FAMILY_EVIDENCE_SET_VERSION,
        query_id=query.query_id,
        query_fingerprint=_required_sha256(query.query_fingerprint),
        family_matrix_signature=family_matrix.matrix_signature,
        family_partition=family_matrix.partition,
        scores=tuple(scores),
        score_set_fingerprint=canonical_semantic_fingerprint(score_set_base),
    )


def _validate_query_matrix(
    query: RawScoringQuery,
    matrix: CachedVectorMatrix,
    *,
    expected_kind: str,
) -> None:
    if not isinstance(query, RawScoringQuery):
        raise TypeError("query must be a RawScoringQuery")
    if not isinstance(matrix, CachedVectorMatrix):
        raise TypeError("scoring matrix must be a CachedVectorMatrix")
    if matrix.matrix_kind != expected_kind:
        raise ValueError(f"scoring matrix must have kind {expected_kind}")
    _sha256(matrix.matrix_signature, field="matrix_signature")
    if matrix.route != query.route:
        raise ValueError("query and scoring matrix routes differ")
    if matrix.visual_input_kind != query.visual_input_kind:
        raise ValueError("query and scoring matrix visual-input kinds differ")
    if matrix.model_fingerprint != query.model_fingerprint:
        raise ValueError("query and scoring matrix model fingerprints differ")
    if matrix.embedding_dimension != len(query.embedding):
        raise ValueError("query and scoring matrix embedding dimensions differ")
    if matrix.row_count <= 0 or matrix.embedding_dimension <= 0:
        raise ValueError("scoring matrix must be non-empty")
    if not (
        len(matrix.row_ids)
        == len(matrix.row_names)
        == len(matrix.row_fingerprints)
        == matrix.row_count
    ):
        raise ValueError("scoring matrix row metadata lengths differ")
    if len(set(matrix.row_ids)) != matrix.row_count:
        raise ValueError("scoring matrix row IDs must be unique")
    if matrix.byte_count != matrix.row_count * matrix.embedding_dimension * 4:
        raise ValueError("scoring matrix byte length is invalid")
    for row_id, row_name, fingerprint in zip(
        matrix.row_ids,
        matrix.row_names,
        matrix.row_fingerprints,
        strict=True,
    ):
        _required_text(row_id, field="matrix row ID")
        _required_text(row_name, field="matrix row name")
        _sha256(fingerprint, field="matrix row fingerprint")
    for index in range(matrix.row_count):
        _require_unit_vector(matrix.vector(index), field="scoring matrix row")


def _matrix_raw_cosines(
    query: tuple[float, ...],
    matrix: CachedVectorMatrix,
) -> tuple[float, ...]:
    values = matrix.float32_buffer
    dimension = matrix.embedding_dimension
    return tuple(
        min(
            1.0,
            max(
                -1.0,
                fsum(
                    query[column] * float(values[row * dimension + column])
                    for column in range(dimension)
                ),
            ),
        )
        for row in range(matrix.row_count)
    )


def _unit_float32_vector(
    values: Sequence[float],
    *,
    field: str,
) -> tuple[float, ...]:
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
    result = tuple(float(value) for value in vector)
    if not result:
        raise ValueError(f"{field} must not be empty")
    _require_unit_vector(result, field=field)
    return result


def _require_unit_vector(values: tuple[float, ...], *, field: str) -> None:
    if not values or any(not isfinite(value) for value in values):
        raise ValueError(f"{field} must contain finite values")
    norm = sqrt(fsum(value * value for value in values))
    if abs(norm - 1.0) > _UNIT_NORM_TOLERANCE:
        raise ValueError(f"{field} must be unit-normalized")


def _route(value: object) -> str:
    route = _required_text(value, field="route")
    if route not in REFERENCE_ROUTES:
        raise ValueError(f"unsupported raw scoring route: {route}")
    return route


def _visual_input_kind(value: object) -> str:
    kind = _required_text(value, field="visual_input_kind")
    if kind not in _VISUAL_INPUT_KINDS:
        raise ValueError(f"unsupported raw scoring visual_input_kind: {kind}")
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


def _required_sha256(value: str | None) -> str:
    if value is None:
        raise AssertionError("validated fingerprint is unexpectedly absent")
    return value


__all__ = [
    "RAW_FAMILY_EVIDENCE_SET_VERSION",
    "RAW_FAMILY_EVIDENCE_VERSION",
    "RAW_SCORING_QUERY_VERSION",
    "RawFamilyEvidence",
    "RawFamilyEvidenceSet",
    "RawScoringQuery",
    "score_family_evidence",
]
