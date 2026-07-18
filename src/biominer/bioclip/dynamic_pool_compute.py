"""Encoder-free execution boundary for cached dynamic-pool vector scoring."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
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
    validate_raw_fusion_rankings,
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
POOL_MATRIX_BATCH_POLICY_VERSION = "pool-matrix-batch-policy-v1"
POOL_MATRIX_BATCH_VERSION = "pool-matrix-batch-v1"
POOL_MATRIX_BATCH_METRICS_VERSION = "pool-matrix-batch-metrics-v1"
POOL_MATRIX_BATCH_RESULT_VERSION = "pool-matrix-batch-result-v1"

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


@dataclass(frozen=True, slots=True)
class PoolMatrixBatchPolicy:
    """Hard work-count, unique-matrix and Float32-byte batch ceilings."""

    maximum_work_items_per_batch: int = 64
    maximum_unique_pool_matrices_per_batch: int = 256
    maximum_pool_matrix_bytes_per_batch: int = 512 * 1024 * 1024
    schema_version: str = POOL_MATRIX_BATCH_POLICY_VERSION
    policy_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != POOL_MATRIX_BATCH_POLICY_VERSION:
            raise ValueError("unsupported pool matrix batch policy version")
        work_items = _positive_int(
            self.maximum_work_items_per_batch,
            field="maximum_work_items_per_batch",
        )
        matrices = _positive_int(
            self.maximum_unique_pool_matrices_per_batch,
            field="maximum_unique_pool_matrices_per_batch",
        )
        byte_limit = _positive_int(
            self.maximum_pool_matrix_bytes_per_batch,
            field="maximum_pool_matrix_bytes_per_batch",
        )
        base = {
            "schema_version": POOL_MATRIX_BATCH_POLICY_VERSION,
            "maximum_work_items_per_batch": work_items,
            "maximum_unique_pool_matrices_per_batch": matrices,
            "maximum_pool_matrix_bytes_per_batch": byte_limit,
            "matrix_byte_semantics": "unique_immutable_float32_pool_buffers",
        }
        fingerprint = canonical_semantic_fingerprint(base)
        if self.policy_fingerprint not in {None, fingerprint}:
            raise ValueError("pool matrix batch policy fingerprint mismatch")
        object.__setattr__(self, "maximum_work_items_per_batch", work_items)
        object.__setattr__(
            self,
            "maximum_unique_pool_matrices_per_batch",
            matrices,
        )
        object.__setattr__(
            self,
            "maximum_pool_matrix_bytes_per_batch",
            byte_limit,
        )
        object.__setattr__(self, "policy_fingerprint", fingerprint)


@dataclass(frozen=True, slots=True)
class PoolMatrixBatch:
    """One execution batch over a bounded shared pool-matrix working set."""

    schema_version: str
    batch_index: int
    work_fingerprints: tuple[str, ...]
    pool_matrix_signatures: tuple[str, ...]
    work_item_count: int
    pool_matrix_reference_count: int
    unique_pool_matrix_count: int
    pool_matrix_row_count: int
    pool_matrix_bytes: int
    batch_fingerprint: str


@dataclass(frozen=True, slots=True)
class PoolMatrixBatchMetrics:
    """Matrix memory, reuse and encoder-free totals across all batches."""

    schema_version: str
    work_items: int
    execution_batches: int
    pool_matrix_references: int
    unique_pool_matrices: int
    unique_pool_matrix_rows: int
    unique_pool_matrix_bytes: int
    within_batch_matrix_reuses: int
    cross_batch_matrix_reloads: int
    maximum_batch_work_items: int
    maximum_batch_unique_pool_matrices: int
    maximum_batch_pool_matrix_bytes: int
    encoder_invocations: int
    image_materializations: int
    metrics_fingerprint: str


@dataclass(frozen=True, slots=True)
class PoolMatrixBatchResult:
    """Locality-ordered execution plus independent canonical result order."""

    schema_version: str
    policy: PoolMatrixBatchPolicy
    policy_fingerprint: str
    batches: tuple[PoolMatrixBatch, ...]
    execution_results: tuple[DynamicVectorScoringResult, ...]
    canonical_results: tuple[DynamicVectorScoringResult, ...]
    metrics: PoolMatrixBatchMetrics
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


def execute_dynamic_vector_scoring_batches(
    works: Sequence[DynamicVectorScoringWork],
    *,
    policy: PoolMatrixBatchPolicy | None = None,
) -> PoolMatrixBatchResult:
    """Execute vector work against bounded, locality-ordered pool-matrix batches."""

    selected_policy = policy or PoolMatrixBatchPolicy()
    if not isinstance(selected_policy, PoolMatrixBatchPolicy):
        raise TypeError("policy must be PoolMatrixBatchPolicy")
    work_items = _validated_work_items(works)
    batches, work_by_fingerprint = _plan_pool_matrix_batches(
        work_items,
        selected_policy,
    )
    execution_results = tuple(
        _execute_dynamic_vector_scoring(work_by_fingerprint[work_fingerprint])
        for batch in batches
        for work_fingerprint in batch.work_fingerprints
    )
    result = _assemble_pool_matrix_batch_result(
        selected_policy,
        batches,
        execution_results,
    )
    validate_pool_matrix_batch_result(result)
    return result


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


def validate_pool_matrix_batch_result(result: PoolMatrixBatchResult) -> None:
    """Validate batch bounds and fingerprints without repeating matrix scoring."""

    if not isinstance(result, PoolMatrixBatchResult):
        raise TypeError("result must be PoolMatrixBatchResult")
    if result.schema_version != POOL_MATRIX_BATCH_RESULT_VERSION:
        raise ValueError("unsupported pool matrix batch result version")
    if not isinstance(result.policy, PoolMatrixBatchPolicy):
        raise TypeError("batch result policy must be PoolMatrixBatchPolicy")
    if result.policy_fingerprint != result.policy.policy_fingerprint:
        raise ValueError("pool matrix batch policy fingerprint mismatch")
    if not result.execution_results:
        raise ValueError("pool matrix batch result must not be empty")
    for execution_result in result.execution_results:
        _validate_dynamic_vector_scoring_result_structure(execution_result)
    works = tuple(item.work for item in result.execution_results)
    expected_batches, _ = _plan_pool_matrix_batches(works, result.policy)
    if result.batches != expected_batches:
        raise ValueError("pool matrix batches do not match their work or policy")
    expected_execution_order = tuple(
        work_fingerprint
        for batch in expected_batches
        for work_fingerprint in batch.work_fingerprints
    )
    observed_execution_order = tuple(
        item.work_fingerprint for item in result.execution_results
    )
    if observed_execution_order != expected_execution_order:
        raise ValueError("dynamic vector results are not in batch execution order")
    expected = _assemble_pool_matrix_batch_result(
        result.policy,
        expected_batches,
        result.execution_results,
    )
    if result != expected:
        raise ValueError("pool matrix batch result fingerprint or metrics mismatch")


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
    result = DynamicVectorScoringResult(
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
        result_fingerprint="",
    )
    return _with_result_fingerprint(result)


def _with_result_fingerprint(
    result: DynamicVectorScoringResult,
) -> DynamicVectorScoringResult:
    """Bind a vector result to all stage fingerprints without re-running stages."""

    return replace(
        result,
        result_fingerprint=canonical_semantic_fingerprint(
            _dynamic_vector_scoring_result_base(result)
        ),
    )


def _validate_dynamic_vector_scoring_result_structure(
    result: DynamicVectorScoringResult,
) -> None:
    if not isinstance(result, DynamicVectorScoringResult):
        raise TypeError("batch execution result must be DynamicVectorScoringResult")
    if result.schema_version != DYNAMIC_VECTOR_SCORING_RESULT_VERSION:
        raise ValueError("unsupported dynamic vector scoring result version")
    validate_dynamic_vector_scoring_work(result.work)
    if result.work_fingerprint != result.work.work_fingerprint:
        raise ValueError("dynamic vector result work fingerprint mismatch")
    if result.source_embedding_id != result.work.source_embedding.embedding_id:
        raise ValueError("dynamic vector result source embedding id mismatch")
    if result.source_embedding_fingerprint != (
        result.work.source_embedding.embedding_fingerprint
    ):
        raise ValueError("dynamic vector result source embedding fingerprint mismatch")
    if result.encoder_invocations != 0 or result.image_materializations != 0:
        raise ValueError("dynamic vector result crossed the encoder-free boundary")
    if result.cached_query_vectors_consumed != 1:
        raise ValueError("dynamic vector result must consume one cached query vector")
    if result.components.family_evidence != result.family_evidence:
        raise ValueError("dynamic vector family evidence differs from components")
    if result.components.global_evidence_set_fingerprint != (
        result.global_evidence.score_set_fingerprint
    ):
        raise ValueError("dynamic vector global evidence differs from components")
    if result.components.local_evidence_set_fingerprint != (
        result.local_evidence.score_set_fingerprint
    ):
        raise ValueError("dynamic vector local evidence differs from components")
    if result.components.disagreement_coverage != result.disagreement_coverage:
        raise ValueError("dynamic vector disagreement differs from components")
    if result.fusion_scores.component_set != result.components:
        raise ValueError("dynamic vector fusion scores differ from components")
    if result.rankings.fusion_scores != result.fusion_scores:
        raise ValueError("dynamic vector rankings differ from fusion scores")
    validate_raw_fusion_rankings(result.rankings)
    expected_fingerprint = canonical_semantic_fingerprint(
        _dynamic_vector_scoring_result_base(result)
    )
    if result.result_fingerprint != expected_fingerprint:
        raise ValueError("dynamic vector scoring result fingerprint mismatch")


def _dynamic_vector_scoring_result_base(
    result: DynamicVectorScoringResult,
) -> dict[str, object]:
    return {
        "schema_version": DYNAMIC_VECTOR_SCORING_RESULT_VERSION,
        "work_fingerprint": result.work_fingerprint,
        "source_embedding_id": result.source_embedding_id,
        "source_embedding_fingerprint": result.source_embedding_fingerprint,
        "encoder_invocations": result.encoder_invocations,
        "image_materializations": result.image_materializations,
        "cached_query_vectors_consumed": result.cached_query_vectors_consumed,
        "family_evidence_set_fingerprint": (
            result.family_evidence.score_set_fingerprint
        ),
        "global_evidence_set_fingerprint": (
            result.global_evidence.score_set_fingerprint
        ),
        "local_evidence_set_fingerprint": result.local_evidence.score_set_fingerprint,
        "disagreement_coverage_set_fingerprint": (
            result.disagreement_coverage.evidence_set_fingerprint
        ),
        "component_set_fingerprint": result.components.component_set_fingerprint,
        "fusion_score_set_fingerprint": result.fusion_scores.score_set_fingerprint,
        "ranking_set_fingerprint": result.rankings.ranking_set_fingerprint,
    }


def _validated_work_items(
    works: object,
) -> tuple[DynamicVectorScoringWork, ...]:
    items = _typed_tuple(
        works,
        expected_type=DynamicVectorScoringWork,
        field="works",
    )
    seen: set[str] = set()
    for work in items:
        validate_dynamic_vector_scoring_work(work)
        if work.work_fingerprint in seen:
            raise ValueError("works must not repeat a work fingerprint")
        seen.add(work.work_fingerprint)
    return items


def _plan_pool_matrix_batches(
    works: Sequence[DynamicVectorScoringWork],
    policy: PoolMatrixBatchPolicy,
) -> tuple[tuple[PoolMatrixBatch, ...], dict[str, DynamicVectorScoringWork]]:
    ordered_works = tuple(sorted(works, key=_pool_matrix_locality_key))
    work_by_fingerprint = {work.work_fingerprint: work for work in ordered_works}
    batches: list[PoolMatrixBatch] = []
    pending: list[DynamicVectorScoringWork] = []
    pending_matrices: dict[str, CachedVectorMatrix] = {}
    for work in ordered_works:
        work_matrices, _ = _pool_matrix_inventory((work,))
        _reject_oversized_pool_matrix_work(work, work_matrices, policy)
        combined = _merge_pool_matrix_inventory(pending_matrices, work_matrices)
        exceeds_policy = (
            len(pending) + 1 > policy.maximum_work_items_per_batch
            or len(combined) > policy.maximum_unique_pool_matrices_per_batch
            or _pool_matrix_bytes(combined) > policy.maximum_pool_matrix_bytes_per_batch
        )
        if pending and exceeds_policy:
            batches.append(
                _make_pool_matrix_batch(
                    len(batches),
                    tuple(pending),
                    pending_matrices,
                    policy,
                )
            )
            pending = []
            pending_matrices = {}
            combined = work_matrices
        pending.append(work)
        pending_matrices = combined
    if pending:
        batches.append(
            _make_pool_matrix_batch(
                len(batches),
                tuple(pending),
                pending_matrices,
                policy,
            )
        )
    return tuple(batches), work_by_fingerprint


def _reject_oversized_pool_matrix_work(
    work: DynamicVectorScoringWork,
    matrices: dict[str, CachedVectorMatrix],
    policy: PoolMatrixBatchPolicy,
) -> None:
    if len(matrices) > policy.maximum_unique_pool_matrices_per_batch:
        raise ValueError(
            f"work {work.work_fingerprint} exceeds the unique pool-matrix limit"
        )
    if _pool_matrix_bytes(matrices) > policy.maximum_pool_matrix_bytes_per_batch:
        raise ValueError(
            f"work {work.work_fingerprint} exceeds the pool-matrix byte limit"
        )


def _make_pool_matrix_batch(
    batch_index: int,
    works: tuple[DynamicVectorScoringWork, ...],
    matrices: dict[str, CachedVectorMatrix],
    policy: PoolMatrixBatchPolicy,
) -> PoolMatrixBatch:
    signatures = tuple(sorted(matrices))
    reference_count = sum(len(_pool_matrix_references(work)) for work in works)
    row_count = sum(matrices[signature].row_count for signature in signatures)
    byte_count = _pool_matrix_bytes(matrices)
    work_fingerprints = tuple(work.work_fingerprint for work in works)
    base = {
        "schema_version": POOL_MATRIX_BATCH_VERSION,
        "policy_fingerprint": policy.policy_fingerprint,
        "batch_index": batch_index,
        "work_fingerprints": list(work_fingerprints),
        "pool_matrix_signatures": list(signatures),
        "work_item_count": len(works),
        "pool_matrix_reference_count": reference_count,
        "unique_pool_matrix_count": len(signatures),
        "pool_matrix_row_count": row_count,
        "pool_matrix_bytes": byte_count,
    }
    return PoolMatrixBatch(
        schema_version=POOL_MATRIX_BATCH_VERSION,
        batch_index=batch_index,
        work_fingerprints=work_fingerprints,
        pool_matrix_signatures=signatures,
        work_item_count=len(works),
        pool_matrix_reference_count=reference_count,
        unique_pool_matrix_count=len(signatures),
        pool_matrix_row_count=row_count,
        pool_matrix_bytes=byte_count,
        batch_fingerprint=canonical_semantic_fingerprint(base),
    )


def _pool_matrix_locality_key(work: DynamicVectorScoringWork) -> tuple[object, ...]:
    matrices, _ = _pool_matrix_inventory((work,))
    return (
        work.query.route,
        work.query.visual_input_kind,
        work.family_matrix.partition,
        work.family_matrix.matrix_signature,
        work.candidate_matrix.matrix_signature,
        tuple(sorted(matrices)),
        work.query.query_id,
        work.work_fingerprint,
    )


def _pool_matrix_references(
    work: DynamicVectorScoringWork,
) -> tuple[CachedVectorMatrix, ...]:
    return (
        *(pool.pool_matrix for pool in work.global_pools),
        *(
            pool.pool_matrix
            for pool in work.local_pools
            if pool.pool_matrix is not None
        ),
    )


def _pool_matrix_inventory(
    works: Sequence[DynamicVectorScoringWork],
) -> tuple[dict[str, CachedVectorMatrix], int]:
    matrices: dict[str, CachedVectorMatrix] = {}
    references = 0
    for work in works:
        for matrix in _pool_matrix_references(work):
            references += 1
            existing = matrices.get(matrix.matrix_signature)
            if existing is not None and existing != matrix:
                raise ValueError("one pool matrix signature identifies different data")
            matrices[matrix.matrix_signature] = matrix
    return matrices, references


def _merge_pool_matrix_inventory(
    first: dict[str, CachedVectorMatrix],
    second: dict[str, CachedVectorMatrix],
) -> dict[str, CachedVectorMatrix]:
    merged = dict(first)
    for signature, matrix in second.items():
        existing = merged.get(signature)
        if existing is not None and existing != matrix:
            raise ValueError("one pool matrix signature identifies different data")
        merged[signature] = matrix
    return merged


def _pool_matrix_bytes(matrices: dict[str, CachedVectorMatrix]) -> int:
    return sum(matrix.byte_count for matrix in matrices.values())


def _assemble_pool_matrix_batch_result(
    policy: PoolMatrixBatchPolicy,
    batches: tuple[PoolMatrixBatch, ...],
    execution_results: tuple[DynamicVectorScoringResult, ...],
) -> PoolMatrixBatchResult:
    canonical_results = tuple(
        sorted(
            execution_results,
            key=lambda item: (item.work.query.query_id, item.work_fingerprint),
        )
    )
    metrics = _pool_matrix_batch_metrics(batches, execution_results)
    base = {
        "schema_version": POOL_MATRIX_BATCH_RESULT_VERSION,
        "policy_fingerprint": policy.policy_fingerprint,
        "batch_fingerprints": [batch.batch_fingerprint for batch in batches],
        "execution_result_fingerprints": [
            item.result_fingerprint for item in execution_results
        ],
        "canonical_result_fingerprints": [
            item.result_fingerprint for item in canonical_results
        ],
        "metrics_fingerprint": metrics.metrics_fingerprint,
    }
    return PoolMatrixBatchResult(
        schema_version=POOL_MATRIX_BATCH_RESULT_VERSION,
        policy=policy,
        policy_fingerprint=policy.policy_fingerprint or "",
        batches=batches,
        execution_results=execution_results,
        canonical_results=canonical_results,
        metrics=metrics,
        result_fingerprint=canonical_semantic_fingerprint(base),
    )


def _pool_matrix_batch_metrics(
    batches: tuple[PoolMatrixBatch, ...],
    execution_results: tuple[DynamicVectorScoringResult, ...],
) -> PoolMatrixBatchMetrics:
    all_matrices, reference_count = _pool_matrix_inventory(
        tuple(item.work for item in execution_results)
    )
    unique_rows = sum(matrix.row_count for matrix in all_matrices.values())
    unique_bytes = _pool_matrix_bytes(all_matrices)
    within_batch_reuses = sum(
        batch.pool_matrix_reference_count - batch.unique_pool_matrix_count
        for batch in batches
    )
    cross_batch_reloads = sum(
        batch.unique_pool_matrix_count for batch in batches
    ) - len(all_matrices)
    values = {
        "schema_version": POOL_MATRIX_BATCH_METRICS_VERSION,
        "work_items": len(execution_results),
        "execution_batches": len(batches),
        "pool_matrix_references": reference_count,
        "unique_pool_matrices": len(all_matrices),
        "unique_pool_matrix_rows": unique_rows,
        "unique_pool_matrix_bytes": unique_bytes,
        "within_batch_matrix_reuses": within_batch_reuses,
        "cross_batch_matrix_reloads": cross_batch_reloads,
        "maximum_batch_work_items": max(
            (batch.work_item_count for batch in batches),
            default=0,
        ),
        "maximum_batch_unique_pool_matrices": max(
            (batch.unique_pool_matrix_count for batch in batches),
            default=0,
        ),
        "maximum_batch_pool_matrix_bytes": max(
            (batch.pool_matrix_bytes for batch in batches),
            default=0,
        ),
        "encoder_invocations": sum(
            item.encoder_invocations for item in execution_results
        ),
        "image_materializations": sum(
            item.image_materializations for item in execution_results
        ),
    }
    return PoolMatrixBatchMetrics(
        **values,
        metrics_fingerprint=canonical_semantic_fingerprint(values),
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


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


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
    "POOL_MATRIX_BATCH_METRICS_VERSION",
    "POOL_MATRIX_BATCH_POLICY_VERSION",
    "POOL_MATRIX_BATCH_RESULT_VERSION",
    "POOL_MATRIX_BATCH_VERSION",
    "PoolMatrixBatch",
    "PoolMatrixBatchMetrics",
    "PoolMatrixBatchPolicy",
    "PoolMatrixBatchResult",
    "build_dynamic_vector_scoring_work",
    "execute_dynamic_vector_scoring",
    "execute_dynamic_vector_scoring_batches",
    "validate_pool_matrix_batch_result",
    "validate_dynamic_vector_scoring_result",
    "validate_dynamic_vector_scoring_work",
]
