"""Raw, non-probabilistic component scoring over cached BioCLIP matrices."""

from __future__ import annotations

from array import array
from collections import defaultdict
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
GLOBAL_REFERENCE_POOL_INPUT_VERSION = "global-reference-pool-input-v1"
RAW_GLOBAL_REFERENCE_EVIDENCE_VERSION = "raw-global-reference-evidence-v1"
RAW_GLOBAL_REFERENCE_EVIDENCE_SET_VERSION = "raw-global-reference-evidence-set-v1"

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


@dataclass(frozen=True, slots=True)
class GlobalReferencePoolInput:
    """One candidate-bound global pool and its configured support opportunity."""

    candidate_accepted_taxon_key: str
    candidate_scientific_name: str
    pool_matrix: CachedVectorMatrix
    configured_reference_count: int
    configured_top_k: int
    input_fingerprint: str | None = None

    def __post_init__(self) -> None:
        candidate_key = _required_text(
            self.candidate_accepted_taxon_key,
            field="candidate_accepted_taxon_key",
        )
        scientific_name = _required_text(
            self.candidate_scientific_name,
            field="candidate_scientific_name",
        )
        if not isinstance(self.pool_matrix, CachedVectorMatrix):
            raise TypeError("global pool_matrix must be a CachedVectorMatrix")
        if self.pool_matrix.matrix_kind != "dynamic_reference_pool":
            raise ValueError("global pool matrix has the wrong matrix kind")
        if self.pool_matrix.partition != "global":
            raise ValueError("global pool matrix must have global scope")
        if self.pool_matrix.subject_id != candidate_key:
            raise ValueError("global pool matrix is bound to another candidate")
        configured_count = _positive_integer(
            self.configured_reference_count,
            field="configured_reference_count",
        )
        configured_top_k = _positive_integer(
            self.configured_top_k,
            field="configured_top_k",
        )
        base = {
            "schema_version": GLOBAL_REFERENCE_POOL_INPUT_VERSION,
            "candidate_accepted_taxon_key": candidate_key,
            "candidate_scientific_name": scientific_name,
            "pool_matrix_signature": self.pool_matrix.matrix_signature,
            "pool_membership_fingerprint": self.pool_matrix.source_fingerprint,
            "configured_reference_count": configured_count,
            "configured_top_k": configured_top_k,
        }
        fingerprint = canonical_semantic_fingerprint(base)
        if (
            self.input_fingerprint is not None
            and _sha256(
                self.input_fingerprint,
                field="input_fingerprint",
            )
            != fingerprint
        ):
            raise ValueError("input_fingerprint does not match global pool input")
        object.__setattr__(self, "candidate_accepted_taxon_key", candidate_key)
        object.__setattr__(self, "candidate_scientific_name", scientific_name)
        object.__setattr__(self, "configured_reference_count", configured_count)
        object.__setattr__(self, "configured_top_k", configured_top_k)
        object.__setattr__(self, "input_fingerprint", fingerprint)


@dataclass(frozen=True, slots=True)
class RawGlobalReferenceEvidence:
    """Separate raw global prototype and observation-level similarities."""

    schema_version: str
    query_id: str
    candidate_accepted_taxon_key: str
    candidate_scientific_name: str
    candidate_matrix_signature: str
    candidate_prototype_fingerprint: str
    pool_matrix_signature: str
    pool_membership_fingerprint: str
    score_status: str
    prototype_similarity: float
    nearest_reference_similarity: float
    nearest_reference_observation_id: str
    top_k_mean_similarity: float
    configured_k: int
    effective_k: int
    configured_reference_count: int
    reference_count: int
    independent_observation_count: int
    reference_shortfall_count: int
    ranked_reference_observation_ids: tuple[str, ...]
    top_k_reference_observation_ids: tuple[str, ...]
    score_fingerprint: str


@dataclass(frozen=True, slots=True)
class RawGlobalReferenceEvidenceSet:
    """Complete global evidence for every candidate prototype row."""

    schema_version: str
    query_id: str
    query_fingerprint: str
    candidate_matrix_signature: str
    candidate_set_fingerprint: str
    scores: tuple[RawGlobalReferenceEvidence, ...]
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


def score_global_reference_evidence(
    query: RawScoringQuery,
    candidate_matrix: CachedVectorMatrix,
    pools: Sequence[GlobalReferencePoolInput],
) -> RawGlobalReferenceEvidenceSet:
    """Score global prototype and observation evidence for every candidate."""

    _validate_query_matrix(
        query,
        candidate_matrix,
        expected_kind="candidate_prototype",
    )
    pool_items = _global_pool_inputs(pools)
    candidate_keys = tuple(candidate_matrix.row_ids)
    if set(pool_items) != set(candidate_keys):
        raise ValueError(
            "global pool inputs must match the complete candidate matrix membership"
        )
    candidate_rows = {
        candidate_key: (
            index,
            candidate_matrix.row_names[index],
            candidate_matrix.row_fingerprints[index],
        )
        for index, candidate_key in enumerate(candidate_keys)
    }
    output: list[RawGlobalReferenceEvidence] = []
    for candidate_key in candidate_keys:
        index, scientific_name, prototype_fingerprint = candidate_rows[candidate_key]
        pool = pool_items[candidate_key]
        if pool.candidate_scientific_name != scientific_name:
            raise ValueError(
                "global pool scientific name differs from candidate prototype"
            )
        _validate_query_matrix(
            query,
            pool.pool_matrix,
            expected_kind="dynamic_reference_pool",
        )
        if pool.pool_matrix.partition != "global":
            raise ValueError("global reference evidence requires global pool scope")
        if pool.pool_matrix.subject_id != candidate_key:
            raise ValueError("global reference pool is bound to another candidate")
        observations = _independent_observation_vectors(pool.pool_matrix)
        ranked = sorted(
            (
                (_raw_cosine(query.embedding, vector), observation_id)
                for observation_id, vector in observations
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if not ranked:
            raise ValueError("global reference pool has no independent observations")
        effective_k = min(pool.configured_top_k, len(ranked))
        top_k = tuple(ranked[:effective_k])
        prototype_similarity = _raw_cosine(
            query.embedding,
            candidate_matrix.vector(index),
        )
        top_k_mean = fsum(similarity for similarity, _ in top_k) / effective_k
        base = {
            "schema_version": RAW_GLOBAL_REFERENCE_EVIDENCE_VERSION,
            "query_id": query.query_id,
            "query_fingerprint": query.query_fingerprint,
            "candidate_accepted_taxon_key": candidate_key,
            "candidate_scientific_name": scientific_name,
            "candidate_matrix_signature": candidate_matrix.matrix_signature,
            "candidate_prototype_fingerprint": prototype_fingerprint,
            "pool_matrix_signature": pool.pool_matrix.matrix_signature,
            "pool_membership_fingerprint": pool.pool_matrix.source_fingerprint,
            "pool_input_fingerprint": pool.input_fingerprint,
            "score_status": "available",
            "prototype_similarity": prototype_similarity,
            "nearest_reference_similarity": ranked[0][0],
            "nearest_reference_observation_id": ranked[0][1],
            "top_k_mean_similarity": top_k_mean,
            "configured_k": pool.configured_top_k,
            "effective_k": effective_k,
            "configured_reference_count": pool.configured_reference_count,
            "reference_count": pool.pool_matrix.row_count,
            "independent_observation_count": len(observations),
            "reference_shortfall_count": max(
                0,
                pool.configured_reference_count - len(observations),
            ),
            "ranked_reference_observation_ids": [item[1] for item in ranked],
            "top_k_reference_observation_ids": [item[1] for item in top_k],
        }
        output.append(
            RawGlobalReferenceEvidence(
                schema_version=RAW_GLOBAL_REFERENCE_EVIDENCE_VERSION,
                query_id=query.query_id,
                candidate_accepted_taxon_key=candidate_key,
                candidate_scientific_name=scientific_name,
                candidate_matrix_signature=candidate_matrix.matrix_signature,
                candidate_prototype_fingerprint=prototype_fingerprint,
                pool_matrix_signature=pool.pool_matrix.matrix_signature,
                pool_membership_fingerprint=pool.pool_matrix.source_fingerprint,
                score_status="available",
                prototype_similarity=prototype_similarity,
                nearest_reference_similarity=ranked[0][0],
                nearest_reference_observation_id=ranked[0][1],
                top_k_mean_similarity=top_k_mean,
                configured_k=pool.configured_top_k,
                effective_k=effective_k,
                configured_reference_count=pool.configured_reference_count,
                reference_count=pool.pool_matrix.row_count,
                independent_observation_count=len(observations),
                reference_shortfall_count=max(
                    0,
                    pool.configured_reference_count - len(observations),
                ),
                ranked_reference_observation_ids=tuple(item[1] for item in ranked),
                top_k_reference_observation_ids=tuple(item[1] for item in top_k),
                score_fingerprint=canonical_semantic_fingerprint(base),
            )
        )
    score_set_base = {
        "schema_version": RAW_GLOBAL_REFERENCE_EVIDENCE_SET_VERSION,
        "query_id": query.query_id,
        "query_fingerprint": query.query_fingerprint,
        "candidate_matrix_signature": candidate_matrix.matrix_signature,
        "candidate_set_fingerprint": candidate_matrix.source_fingerprint,
        "score_fingerprints": [score.score_fingerprint for score in output],
    }
    return RawGlobalReferenceEvidenceSet(
        schema_version=RAW_GLOBAL_REFERENCE_EVIDENCE_SET_VERSION,
        query_id=query.query_id,
        query_fingerprint=_required_sha256(query.query_fingerprint),
        candidate_matrix_signature=candidate_matrix.matrix_signature,
        candidate_set_fingerprint=candidate_matrix.source_fingerprint,
        scores=tuple(output),
        score_set_fingerprint=canonical_semantic_fingerprint(score_set_base),
    )


def _global_pool_inputs(
    pools: Sequence[GlobalReferencePoolInput],
) -> dict[str, GlobalReferencePoolInput]:
    if isinstance(pools, str | bytes) or not isinstance(pools, Sequence):
        raise TypeError("global pool inputs must be a sequence")
    items = tuple(pools)
    if not items:
        raise ValueError("global pool inputs must not be empty")
    if any(not isinstance(item, GlobalReferencePoolInput) for item in items):
        raise TypeError("global pool inputs contain invalid row types")
    result: dict[str, GlobalReferencePoolInput] = {}
    for item in items:
        if item.candidate_accepted_taxon_key in result:
            raise ValueError("global pool inputs repeat a candidate taxon key")
        result[item.candidate_accepted_taxon_key] = item
    return result


def _independent_observation_vectors(
    matrix: CachedVectorMatrix,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    grouped: dict[str, list[tuple[float, ...]]] = defaultdict(list)
    for index, observation_id in enumerate(matrix.row_names):
        grouped[observation_id].append(matrix.vector(index))
    return tuple(
        (observation_id, _normalized_mean(grouped[observation_id]))
        for observation_id in sorted(grouped)
    )


def _normalized_mean(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors:
        raise ValueError("observation aggregation requires at least one vector")
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("observation aggregation has mixed dimensions")
    mean = tuple(
        fsum(vector[index] for vector in vectors) / len(vectors)
        for index in range(dimension)
    )
    norm = sqrt(fsum(value * value for value in mean))
    if not isfinite(norm) or norm <= 0.0:
        raise ValueError("observation aggregation has no scoring direction")
    stored = array("f", (value / norm for value in mean))
    result = tuple(float(value) for value in stored)
    _require_unit_vector(result, field="observation aggregate")
    return result


def _raw_cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("raw cosine vectors have different dimensions")
    return min(
        1.0,
        max(-1.0, fsum(a * b for a, b in zip(left, right, strict=True))),
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


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


__all__ = [
    "GLOBAL_REFERENCE_POOL_INPUT_VERSION",
    "RAW_FAMILY_EVIDENCE_SET_VERSION",
    "RAW_FAMILY_EVIDENCE_VERSION",
    "RAW_GLOBAL_REFERENCE_EVIDENCE_SET_VERSION",
    "RAW_GLOBAL_REFERENCE_EVIDENCE_VERSION",
    "RAW_SCORING_QUERY_VERSION",
    "GlobalReferencePoolInput",
    "RawFamilyEvidence",
    "RawFamilyEvidenceSet",
    "RawGlobalReferenceEvidence",
    "RawGlobalReferenceEvidenceSet",
    "RawScoringQuery",
    "score_family_evidence",
    "score_global_reference_evidence",
]
