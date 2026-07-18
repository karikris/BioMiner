"""Encoder-free execution boundary for cached dynamic-pool vector scoring."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot, isclose, isfinite
import re

from biominer.bioclip.dynamic_pool_fusion import (
    DynamicScoreComponentSet,
    RawFusionRankingSet,
    RawFusionScoreSet,
    ValidationLinearFusionParameters,
    evaluate_raw_fusion_methods,
    preserve_dynamic_score_components,
    rank_raw_fusion_candidates,
)
from biominer.bioclip.dynamic_pool_scoring import (
    GlobalReferencePoolInput,
    LocalReferencePoolInput,
    RawDisagreementCoverageSet,
    RawFamilyEvidenceSet,
    RawGlobalReferenceEvidenceSet,
    RawLocalReferenceEvidenceSet,
    RawScoringQuery,
    calculate_dynamic_pool_disagreement_coverage,
    score_family_evidence,
    score_global_reference_evidence,
    score_local_reference_evidence,
)
from biominer.bioclip.matrix_cache import CachedVectorMatrix
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.vision.target_full_frame import RawFullFrameEmbedding


CACHED_QUERY_NORMALIZATION_VERSION = "cached-query-l2-normalization-v1"
DYNAMIC_VECTOR_SCORING_WORK_VERSION = "dynamic-vector-scoring-work-v1"
DYNAMIC_VECTOR_SCORING_RESULT_VERSION = "dynamic-vector-scoring-result-v1"

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class DynamicVectorScoringWork:
    """One scoring job containing vectors and matrices but no image or encoder."""

    schema_version: str
    source_embedding: RawFullFrameEmbedding
    query: RawScoringQuery
    family_matrix: CachedVectorMatrix
    candidate_matrix: CachedVectorMatrix
    global_pools: tuple[GlobalReferencePoolInput, ...]
    local_pools: tuple[LocalReferencePoolInput, ...]
    linear_parameters: ValidationLinearFusionParameters
    work_fingerprint: str


@dataclass(frozen=True, slots=True)
class DynamicVectorScoringResult:
    """Complete vector-only scoring result and explicit stage-separation metrics."""

    schema_version: str
    work: DynamicVectorScoringWork
    work_fingerprint: str
    source_embedding_id: str
    source_embedding_fingerprint: str
    encoder_invocations: int
    image_materializations: int
    cached_query_vectors_consumed: int
    family_evidence: RawFamilyEvidenceSet
    global_evidence: RawGlobalReferenceEvidenceSet
    local_evidence: RawLocalReferenceEvidenceSet
    disagreement_coverage: RawDisagreementCoverageSet
    components: DynamicScoreComponentSet
    fusion_scores: RawFusionScoreSet
    rankings: RawFusionRankingSet
    result_fingerprint: str


def build_dynamic_vector_scoring_work(
    source_embedding: RawFullFrameEmbedding,
    *,
    query_id: str,
    route: str,
    family_matrix: CachedVectorMatrix,
    candidate_matrix: CachedVectorMatrix,
    global_pools: Sequence[GlobalReferencePoolInput],
    local_pools: Sequence[LocalReferencePoolInput],
    linear_parameters: ValidationLinearFusionParameters,
) -> DynamicVectorScoringWork:
    """Build vector-only work from one persisted embedding identity."""

    query = _query_from_cached_embedding(
        source_embedding,
        query_id=query_id,
        route=route,
    )
    global_items = _canonical_pool_tuple(
        global_pools,
        expected_type=GlobalReferencePoolInput,
        field="global_pools",
    )
    local_items = _canonical_pool_tuple(
        local_pools,
        expected_type=LocalReferencePoolInput,
        field="local_pools",
    )
    base = _work_base(
        source_embedding=source_embedding,
        query=query,
        family_matrix=family_matrix,
        candidate_matrix=candidate_matrix,
        global_pools=global_items,
        local_pools=local_items,
        linear_parameters=linear_parameters,
    )
    result = DynamicVectorScoringWork(
        schema_version=DYNAMIC_VECTOR_SCORING_WORK_VERSION,
        source_embedding=source_embedding,
        query=query,
        family_matrix=family_matrix,
        candidate_matrix=candidate_matrix,
        global_pools=global_items,
        local_pools=local_items,
        linear_parameters=linear_parameters,
        work_fingerprint=canonical_semantic_fingerprint(base),
    )
    validate_dynamic_vector_scoring_work(result)
    return result


def validate_dynamic_vector_scoring_work(work: DynamicVectorScoringWork) -> None:
    """Validate the cached-vector boundary without touching model runtime state."""

    if not isinstance(work, DynamicVectorScoringWork):
        raise TypeError("work must be DynamicVectorScoringWork")
    if work.schema_version != DYNAMIC_VECTOR_SCORING_WORK_VERSION:
        raise ValueError("unsupported dynamic vector scoring work version")
    expected_query = _query_from_cached_embedding(
        work.source_embedding,
        query_id=work.query.query_id,
        route=work.query.route,
    )
    if work.query != expected_query:
        raise ValueError("vector scoring query differs from cached embedding")
    if not isinstance(work.family_matrix, CachedVectorMatrix):
        raise TypeError("family_matrix must be CachedVectorMatrix")
    if work.family_matrix.matrix_kind != "family_prototype":
        raise ValueError("vector scoring family matrix has the wrong kind")
    if not isinstance(work.candidate_matrix, CachedVectorMatrix):
        raise TypeError("candidate_matrix must be CachedVectorMatrix")
    if work.candidate_matrix.matrix_kind != "candidate_prototype":
        raise ValueError("vector scoring candidate matrix has the wrong kind")
    global_items = _canonical_pool_tuple(
        work.global_pools,
        expected_type=GlobalReferencePoolInput,
        field="global_pools",
    )
    local_items = _canonical_pool_tuple(
        work.local_pools,
        expected_type=LocalReferencePoolInput,
        field="local_pools",
    )
    if not isinstance(work.linear_parameters, ValidationLinearFusionParameters):
        raise TypeError("linear_parameters must be ValidationLinearFusionParameters")
    if work.global_pools != global_items or work.local_pools != local_items:
        raise ValueError("dynamic vector scoring pools are not canonically ordered")
    base = _work_base(
        source_embedding=work.source_embedding,
        query=work.query,
        family_matrix=work.family_matrix,
        candidate_matrix=work.candidate_matrix,
        global_pools=global_items,
        local_pools=local_items,
        linear_parameters=work.linear_parameters,
    )
    if work.work_fingerprint != canonical_semantic_fingerprint(base):
        raise ValueError("dynamic vector scoring work fingerprint mismatch")


def execute_dynamic_vector_scoring(
    work: DynamicVectorScoringWork,
) -> DynamicVectorScoringResult:
    """Run cached-vector scoring; this API has no encoder or image input."""

    validate_dynamic_vector_scoring_work(work)
    return _execute_dynamic_vector_scoring(work)


def validate_dynamic_vector_scoring_result(
    result: DynamicVectorScoringResult,
) -> None:
    """Audit by recomputing the encoder-free result and rejecting output drift."""

    if not isinstance(result, DynamicVectorScoringResult):
        raise TypeError("result must be DynamicVectorScoringResult")
    if result.schema_version != DYNAMIC_VECTOR_SCORING_RESULT_VERSION:
        raise ValueError("unsupported dynamic vector scoring result version")
    validate_dynamic_vector_scoring_work(result.work)
    expected = _execute_dynamic_vector_scoring(result.work)
    if result != expected:
        raise ValueError("dynamic vector scoring result does not match its work")


def _execute_dynamic_vector_scoring(
    work: DynamicVectorScoringWork,
) -> DynamicVectorScoringResult:
    family_evidence = score_family_evidence(work.query, work.family_matrix)
    global_evidence = score_global_reference_evidence(
        work.query,
        work.candidate_matrix,
        work.global_pools,
    )
    local_evidence = score_local_reference_evidence(
        work.query,
        work.candidate_matrix,
        work.local_pools,
    )
    disagreement = calculate_dynamic_pool_disagreement_coverage(
        global_evidence,
        local_evidence,
    )
    components = preserve_dynamic_score_components(
        family_evidence,
        global_evidence,
        local_evidence,
        disagreement,
    )
    fusion_scores = evaluate_raw_fusion_methods(
        components,
        work.linear_parameters,
    )
    rankings = rank_raw_fusion_candidates(fusion_scores)
    base = {
        "schema_version": DYNAMIC_VECTOR_SCORING_RESULT_VERSION,
        "work_fingerprint": work.work_fingerprint,
        "source_embedding_id": work.source_embedding.embedding_id,
        "source_embedding_fingerprint": work.source_embedding.embedding_fingerprint,
        "encoder_invocations": 0,
        "image_materializations": 0,
        "cached_query_vectors_consumed": 1,
        "family_evidence_set_fingerprint": family_evidence.score_set_fingerprint,
        "global_evidence_set_fingerprint": global_evidence.score_set_fingerprint,
        "local_evidence_set_fingerprint": local_evidence.score_set_fingerprint,
        "disagreement_coverage_set_fingerprint": (
            disagreement.evidence_set_fingerprint
        ),
        "component_set_fingerprint": components.component_set_fingerprint,
        "fusion_score_set_fingerprint": fusion_scores.score_set_fingerprint,
        "ranking_set_fingerprint": rankings.ranking_set_fingerprint,
    }
    return DynamicVectorScoringResult(
        schema_version=DYNAMIC_VECTOR_SCORING_RESULT_VERSION,
        work=work,
        work_fingerprint=work.work_fingerprint,
        source_embedding_id=work.source_embedding.embedding_id,
        source_embedding_fingerprint=work.source_embedding.embedding_fingerprint,
        encoder_invocations=0,
        image_materializations=0,
        cached_query_vectors_consumed=1,
        family_evidence=family_evidence,
        global_evidence=global_evidence,
        local_evidence=local_evidence,
        disagreement_coverage=disagreement,
        components=components,
        fusion_scores=fusion_scores,
        rankings=rankings,
        result_fingerprint=canonical_semantic_fingerprint(base),
    )


def _query_from_cached_embedding(
    source_embedding: RawFullFrameEmbedding,
    *,
    query_id: str,
    route: str,
) -> RawScoringQuery:
    vector, norm = _validated_cached_embedding(source_embedding)
    unit_vector = tuple(value / norm for value in vector)
    query_embedding_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": CACHED_QUERY_NORMALIZATION_VERSION,
            "source_embedding_id": source_embedding.embedding_id,
            "source_embedding_fingerprint": source_embedding.embedding_fingerprint,
            "normalization": "l2",
            "unit_embedding": list(unit_vector),
        }
    )
    return RawScoringQuery(
        query_id=str(query_id or "").strip(),
        query_embedding_fingerprint=query_embedding_fingerprint,
        route=route,
        visual_input_kind=source_embedding.visual_input_kind,
        model_fingerprint=source_embedding.model_fingerprint,
        embedding=unit_vector,
    )


def _validated_cached_embedding(
    embedding: RawFullFrameEmbedding,
) -> tuple[tuple[float, ...], float]:
    if not isinstance(embedding, RawFullFrameEmbedding):
        raise TypeError("source_embedding must be RawFullFrameEmbedding")
    for field in (
        "embedding_id",
        "embedding_fingerprint",
        "visual_input_id",
        "raw_image_content_hash",
        "transformation_fingerprint",
        "model_fingerprint",
        "preprocessing_contract_fingerprint",
        "preprocessing_fingerprint",
    ):
        _sha256(getattr(embedding, field), field=field)
    if not str(embedding.embedding_version or "").strip():
        raise ValueError("embedding_version must be nonblank")
    if not str(embedding.image_resize_mode or "").strip():
        raise ValueError("image_resize_mode must be nonblank")
    vector = tuple(
        _finite_number(value, field="embedding") for value in embedding.embedding
    )
    if not vector:
        raise ValueError("cached embedding must not be empty")
    if (
        isinstance(embedding.embedding_dimension, bool)
        or not isinstance(embedding.embedding_dimension, int)
        or embedding.embedding_dimension != len(vector)
    ):
        raise ValueError("cached embedding dimension mismatch")
    norm = hypot(*vector)
    if norm == 0.0 or not isfinite(norm):
        raise ValueError("cached embedding norm must be finite and nonzero")
    recorded_norm = _finite_number(embedding.embedding_norm, field="embedding_norm")
    if not isclose(norm, recorded_norm, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("cached embedding norm mismatch")
    expected_fingerprint = canonical_semantic_fingerprint(
        {
            "embedding": vector,
            "embedding_id": embedding.embedding_id,
            "embedding_version": embedding.embedding_version,
        }
    )
    if embedding.embedding_fingerprint != expected_fingerprint:
        raise ValueError("cached embedding fingerprint mismatch")
    return vector, norm


def _work_base(
    *,
    source_embedding: RawFullFrameEmbedding,
    query: RawScoringQuery,
    family_matrix: CachedVectorMatrix,
    candidate_matrix: CachedVectorMatrix,
    global_pools: tuple[GlobalReferencePoolInput, ...],
    local_pools: tuple[LocalReferencePoolInput, ...],
    linear_parameters: ValidationLinearFusionParameters,
) -> dict[str, object]:
    if not isinstance(family_matrix, CachedVectorMatrix):
        raise TypeError("family_matrix must be CachedVectorMatrix")
    if not isinstance(candidate_matrix, CachedVectorMatrix):
        raise TypeError("candidate_matrix must be CachedVectorMatrix")
    if not isinstance(linear_parameters, ValidationLinearFusionParameters):
        raise TypeError("linear_parameters must be ValidationLinearFusionParameters")
    return {
        "schema_version": DYNAMIC_VECTOR_SCORING_WORK_VERSION,
        "source_embedding_id": source_embedding.embedding_id,
        "source_embedding_fingerprint": source_embedding.embedding_fingerprint,
        "query_fingerprint": query.query_fingerprint,
        "family_matrix_signature": family_matrix.matrix_signature,
        "candidate_matrix_signature": candidate_matrix.matrix_signature,
        "global_pool_input_fingerprints": [
            pool.input_fingerprint for pool in global_pools
        ],
        "local_pool_input_fingerprints": [
            pool.input_fingerprint for pool in local_pools
        ],
        "linear_parameters_fingerprint": linear_parameters.parameters_fingerprint,
        "stage_boundary": "cached_vectors_only_no_encoder_or_image_payload",
    }


def _typed_tuple(
    values: object,
    *,
    expected_type: type,
    field: str,
) -> tuple:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError(f"{field} must be a sequence")
    items = tuple(values)
    if not items:
        raise ValueError(f"{field} must not be empty")
    if any(not isinstance(item, expected_type) for item in items):
        raise TypeError(f"{field} contains an invalid item")
    return items


def _canonical_pool_tuple(
    values: object,
    *,
    expected_type: type,
    field: str,
) -> tuple:
    return tuple(
        sorted(
            _typed_tuple(
                values,
                expected_type=expected_type,
                field=field,
            ),
            key=lambda item: item.candidate_accepted_taxon_key,
        )
    )


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must contain numeric values")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field} must contain finite values")
    return number


def _sha256(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a canonical sha256 fingerprint")
    return text


__all__ = [
    "CACHED_QUERY_NORMALIZATION_VERSION",
    "DYNAMIC_VECTOR_SCORING_RESULT_VERSION",
    "DYNAMIC_VECTOR_SCORING_WORK_VERSION",
    "DynamicVectorScoringResult",
    "DynamicVectorScoringWork",
    "build_dynamic_vector_scoring_work",
    "execute_dynamic_vector_scoring",
    "validate_dynamic_vector_scoring_result",
    "validate_dynamic_vector_scoring_work",
]
