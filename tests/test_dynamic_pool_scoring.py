"""Numeric tests for raw dynamic-pool component scoring."""

from __future__ import annotations

from dataclasses import replace
import inspect

import pytest

from biominer.bioclip import dynamic_pool_compute
from biominer.bioclip.dynamic_pool_compute import (
    build_dynamic_vector_scoring_work,
    execute_dynamic_vector_scoring,
    validate_dynamic_vector_scoring_result,
    validate_dynamic_vector_scoring_work,
)
from biominer.bioclip.dynamic_pool_fusion import (
    DYNAMIC_SCORE_COMPONENT_SET_VERSION,
    DYNAMIC_SCORE_COMPONENT_VERSION,
    FUSION_COMPONENTS,
    GLOBAL_FUSION_COMPONENTS,
    MAXIMUM_SCOPE_EVIDENCE,
    RAW_FUSION_METHODS,
    ROBUST_RANK_AGGREGATION,
    UNWEIGHTED_COMPONENT_MEAN,
    VALIDATION_FITTED_LINEAR,
    ValidationLinearFusionParameters,
    evaluate_raw_fusion_methods,
    preserve_dynamic_score_components,
    rank_raw_fusion_candidates,
    validate_dynamic_score_components,
    validate_raw_fusion_rankings,
    validate_raw_fusion_scores,
)
from biominer.bioclip.dynamic_pool_scoring import (
    GlobalReferencePoolInput,
    LocalReferencePoolInput,
    RawScoringQuery,
    calculate_dynamic_pool_disagreement_coverage,
    score_family_evidence,
    score_global_reference_evidence,
    score_local_reference_evidence,
)
from biominer.bioclip.matrix_cache import (
    CandidatePrototypeVector,
    DynamicPoolMatrixCache,
    FamilyPrototypeMatrixCache,
    FamilyPrototypeVector,
    PoolReferenceVector,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.vision.target_full_frame import RawFullFrameEmbedding


_MODEL_FINGERPRINT = "sha256:" + "a" * 64
_PROTOTYPE_SET_FINGERPRINT = "sha256:" + "b" * 64
_CANDIDATE_SET_FINGERPRINT = "sha256:" + "c" * 64
_REFERENCE_PROTOTYPE_FINGERPRINT = "sha256:" + "d" * 64
_REFERENCE_EMBEDDING_FINGERPRINT = "sha256:" + "e" * 64


def test_family_evidence_scores_every_row_as_raw_cosine() -> None:
    result = score_family_evidence(_query(), _family_matrix())

    assert result.family_partition == "all-families"
    assert [score.family_key for score in result.scores] == [
        "gbif:9417",
        "gbif:7017",
    ]
    assert [score.raw_similarity for score in result.scores] == pytest.approx(
        [1.0, 0.0]
    )
    assert [score.family_rank for score in result.scores] == [1, 2]
    assert result.scores[0].margin_to_next_raw == pytest.approx(1.0)
    assert result.scores[1].margin_to_next_raw is None
    assert result.score_set_fingerprint.startswith("sha256:")
    assert all(score.score_fingerprint.startswith("sha256:") for score in result.scores)


def test_family_matrix_order_does_not_change_scores_or_fingerprints() -> None:
    rows = _family_rows()
    first = score_family_evidence(_query(), _family_matrix(rows))
    second = score_family_evidence(_query(), _family_matrix(tuple(reversed(rows))))

    assert first == second


def test_adding_family_preserves_existing_raw_cosines_without_pruning() -> None:
    baseline = score_family_evidence(_query(), _family_matrix())
    expanded = score_family_evidence(
        _query(),
        _family_matrix(
            (
                *_family_rows(),
                FamilyPrototypeVector(
                    family_key="gbif:other",
                    family_name="Otheridae",
                    prototype_fingerprint="sha256:" + "3" * 64,
                    embedding=(-1.0, 0.0),
                ),
            ),
            source_fingerprint="sha256:" + "c" * 64,
        ),
    )

    baseline_raw = {score.family_key: score.raw_similarity for score in baseline.scores}
    expanded_raw = {score.family_key: score.raw_similarity for score in expanded.scores}
    assert len(expanded.scores) == 3
    assert {key: expanded_raw[key] for key in baseline_raw} == baseline_raw
    assert expanded_raw["gbif:other"] == pytest.approx(-1.0)


def test_family_scoring_rejects_query_or_matrix_contract_drift() -> None:
    query = _query()
    matrix = _family_matrix()

    with pytest.raises(ValueError, match="routes differ"):
        score_family_evidence(
            replace(query, route="larval", query_fingerprint=None), matrix
        )
    with pytest.raises(ValueError, match="visual-input kinds differ"):
        score_family_evidence(
            replace(
                query,
                visual_input_kind="raw_full_image",
                query_fingerprint=None,
            ),
            matrix,
        )
    with pytest.raises(ValueError, match="model fingerprints differ"):
        score_family_evidence(
            replace(
                query,
                model_fingerprint="sha256:" + "d" * 64,
                query_fingerprint=None,
            ),
            matrix,
        )
    with pytest.raises(ValueError, match="kind family_prototype"):
        score_family_evidence(query, replace(matrix, matrix_kind="candidate_prototype"))


def test_family_scoring_rejects_non_unit_query_and_tampered_matrix() -> None:
    with pytest.raises(ValueError, match="unit-normalized"):
        RawScoringQuery(
            query_id="query:bad",
            query_embedding_fingerprint="sha256:" + "e" * 64,
            route="adult_field",
            visual_input_kind="focused_full_frame",
            model_fingerprint=_MODEL_FINGERPRINT,
            embedding=(2.0, 0.0),
        )

    matrix = _family_matrix()
    with pytest.raises(ValueError, match="byte length is invalid"):
        score_family_evidence(_query(), replace(matrix, _float32_bytes=b"bad"))


def test_global_evidence_preserves_prototype_nearest_top_k_and_support() -> None:
    candidate_matrix, pools = _global_inputs()

    result = score_global_reference_evidence(_query(), candidate_matrix, pools)

    assert [score.candidate_accepted_taxon_key for score in result.scores] == [
        "gbif:100",
        "gbif:200",
    ]
    first, second = result.scores
    assert first.score_status == "available"
    assert first.prototype_similarity == pytest.approx(1.0)
    assert first.nearest_reference_similarity == pytest.approx(1.0)
    assert first.nearest_reference_observation_id == "reference-observation:1"
    assert first.top_k_mean_similarity == pytest.approx((1.0 + 2**-0.5) / 2)
    assert first.configured_k == 5
    assert first.effective_k == 2
    assert first.configured_reference_count == 3
    assert first.reference_count == 3
    assert first.independent_observation_count == 2
    assert first.reference_shortfall_count == 1
    assert first.ranked_reference_observation_ids == (
        "reference-observation:1",
        "reference-observation:2",
    )
    assert (
        first.top_k_reference_observation_ids == first.ranked_reference_observation_ids
    )
    assert second.prototype_similarity == pytest.approx(0.0)
    assert second.nearest_reference_similarity == pytest.approx(0.0)
    assert second.top_k_mean_similarity == pytest.approx(-0.5)
    assert result.score_set_fingerprint.startswith("sha256:")


def test_global_evidence_aggregates_duplicate_media_per_observation() -> None:
    candidate_matrix, pools = _global_inputs()
    first = score_global_reference_evidence(_query(), candidate_matrix, pools).scores[0]

    assert first.reference_count == 3
    assert first.independent_observation_count == 2
    assert first.effective_k == 2
    assert len(first.ranked_reference_observation_ids) == 2


def test_global_scoring_is_independent_of_pool_input_order() -> None:
    candidate_matrix, pools = _global_inputs()

    first = score_global_reference_evidence(_query(), candidate_matrix, pools)
    second = score_global_reference_evidence(
        _query(),
        candidate_matrix,
        tuple(reversed(pools)),
    )

    assert first == second


def test_global_scoring_requires_one_bound_pool_for_every_candidate() -> None:
    candidate_matrix, pools = _global_inputs()

    with pytest.raises(ValueError, match="complete candidate matrix membership"):
        score_global_reference_evidence(_query(), candidate_matrix, pools[:1])
    with pytest.raises(ValueError, match="repeat a candidate"):
        score_global_reference_evidence(
            _query(),
            candidate_matrix,
            (pools[0], pools[0], pools[1]),
        )
    with pytest.raises(ValueError, match="scientific name differs"):
        score_global_reference_evidence(
            _query(),
            candidate_matrix,
            (
                replace(
                    pools[0],
                    candidate_scientific_name="Conflicting name",
                    input_fingerprint=None,
                ),
                pools[1],
            ),
        )


def test_global_pool_input_rejects_candidate_and_scope_drift() -> None:
    cache = DynamicPoolMatrixCache()
    rows = _reference_rows("one")
    foreign = _pool_matrix(
        cache,
        candidate_key="gbif:200",
        geographic_scope="global",
        rows=rows,
        fingerprint_digit="8",
    )
    with pytest.raises(ValueError, match="another candidate"):
        GlobalReferencePoolInput(
            candidate_accepted_taxon_key="gbif:100",
            candidate_scientific_name="Papilio demoleus",
            pool_matrix=foreign,
            configured_reference_count=3,
            configured_top_k=5,
        )

    local = _pool_matrix(
        cache,
        candidate_key="gbif:100",
        geographic_scope="exact_local_cell",
        rows=rows,
        fingerprint_digit="9",
    )
    with pytest.raises(ValueError, match="global scope"):
        GlobalReferencePoolInput(
            candidate_accepted_taxon_key="gbif:100",
            candidate_scientific_name="Papilio demoleus",
            pool_matrix=local,
            configured_reference_count=3,
            configured_top_k=5,
        )


def test_local_evidence_scores_geographic_pool_and_preserves_unavailable_state() -> (
    None
):
    candidate_matrix, _ = _global_inputs()
    pools = _local_inputs()

    result = score_local_reference_evidence(_query(), candidate_matrix, pools)

    available, unavailable = result.scores
    assert available.candidate_accepted_taxon_key == "gbif:100"
    assert available.score_status == "available"
    assert available.score_unavailable_reason is None
    assert available.geographic_scope == "exact_local_cell"
    assert available.prototype_similarity == pytest.approx(3 / 10**0.5)
    assert available.nearest_reference_similarity == pytest.approx(1.0)
    assert available.top_k_mean_similarity == pytest.approx(0.9)
    assert available.configured_k == 3
    assert available.effective_k == 2
    assert available.reference_count == 2
    assert available.independent_observation_count == 2
    assert available.reference_shortfall_count == 1

    assert unavailable.candidate_accepted_taxon_key == "gbif:200"
    assert unavailable.score_status == "unavailable"
    assert unavailable.score_unavailable_reason == "no_local_geographic_evidence"
    assert unavailable.pool_matrix_signature is None
    assert unavailable.pool_membership_fingerprint is None
    assert unavailable.geographic_scope is None
    assert unavailable.prototype_similarity is None
    assert unavailable.nearest_reference_similarity is None
    assert unavailable.top_k_mean_similarity is None
    assert unavailable.effective_k == 0
    assert unavailable.reference_count == 0
    assert unavailable.independent_observation_count == 0
    assert unavailable.reference_shortfall_count == 2
    assert unavailable.ranked_reference_observation_ids == ()


def test_local_scoring_is_order_independent_and_requires_complete_candidates() -> None:
    candidate_matrix, _ = _global_inputs()
    pools = _local_inputs()

    first = score_local_reference_evidence(_query(), candidate_matrix, pools)
    second = score_local_reference_evidence(
        _query(),
        candidate_matrix,
        tuple(reversed(pools)),
    )
    assert first == second

    with pytest.raises(ValueError, match="complete candidate matrix membership"):
        score_local_reference_evidence(_query(), candidate_matrix, pools[:1])
    with pytest.raises(ValueError, match="repeat a candidate"):
        score_local_reference_evidence(
            _query(),
            candidate_matrix,
            (pools[0], pools[0], pools[1]),
        )


def test_local_pool_input_requires_exact_available_or_unavailable_contract() -> None:
    cache = DynamicPoolMatrixCache()
    local = _pool_matrix(
        cache,
        candidate_key="gbif:100",
        geographic_scope="exact_local_cell",
        rows=_reference_rows("local"),
        fingerprint_digit="d",
    )
    global_pool = _pool_matrix(
        cache,
        candidate_key="gbif:100",
        geographic_scope="global",
        rows=_reference_rows("global"),
        fingerprint_digit="e",
    )

    with pytest.raises(ValueError, match="geographic scope"):
        _local_input(pool_matrix=global_pool)
    with pytest.raises(ValueError, match="cannot have an unavailable reason"):
        _local_input(
            pool_matrix=local,
            unavailable_reason="conflicting_reason",
        )
    with pytest.raises(ValueError, match="no matrix and an exact reason"):
        _local_input(
            status="unavailable",
            pool_matrix=local,
            unavailable_reason="no_local_geographic_evidence",
        )
    with pytest.raises(ValueError, match="no matrix and an exact reason"):
        _local_input(
            status="unavailable",
            pool_matrix=None,
            unavailable_reason=None,
        )
    with pytest.raises(ValueError, match="positive configured_reference_count"):
        replace(
            _local_input(pool_matrix=local),
            configured_reference_count=0,
            input_fingerprint=None,
        )


def test_unavailable_local_evidence_never_substitutes_global_scores() -> None:
    candidate_matrix, global_pools = _global_inputs()
    global_scores = score_global_reference_evidence(
        _query(),
        candidate_matrix,
        global_pools,
    )
    local_scores = score_local_reference_evidence(
        _query(),
        candidate_matrix,
        _local_inputs(),
    )
    global_by_key = {
        score.candidate_accepted_taxon_key: score for score in global_scores.scores
    }
    local_by_key = {
        score.candidate_accepted_taxon_key: score for score in local_scores.scores
    }

    assert global_by_key["gbif:200"].prototype_similarity is not None
    assert local_by_key["gbif:200"].prototype_similarity is None
    assert local_by_key["gbif:200"].nearest_reference_similarity is None
    assert local_by_key["gbif:200"].top_k_mean_similarity is None


def test_dynamic_pool_disagreement_exposes_raw_deltas_ranks_and_coverage() -> None:
    candidate_matrix, global_pools = _global_inputs()
    global_scores = score_global_reference_evidence(
        _query(),
        candidate_matrix,
        global_pools,
    )
    local_scores = score_local_reference_evidence(
        _query(),
        candidate_matrix,
        _local_inputs(),
    )

    result = calculate_dynamic_pool_disagreement_coverage(
        global_scores,
        local_scores,
    )

    available, unavailable = result.scores
    assert available.candidate_accepted_taxon_key == "gbif:100"
    assert available.disagreement_status == "available"
    assert available.disagreement_unavailable_reason is None
    assert available.prototype_signed_difference == pytest.approx(3 / 10**0.5 - 1)
    assert available.prototype_absolute_disagreement == pytest.approx(1 - 3 / 10**0.5)
    assert available.nearest_signed_difference == pytest.approx(0.0)
    assert available.nearest_absolute_disagreement == pytest.approx(0.0)
    assert available.top_k_signed_difference == pytest.approx(0.9 - (1 + 2**-0.5) / 2)
    assert available.global_prototype_rank == 1
    assert available.local_prototype_rank == 1
    assert available.prototype_rank_movement == 0
    assert available.global_nearest_rank == 1
    assert available.local_nearest_rank == 1
    assert available.nearest_rank_movement == 0
    assert available.global_top_k_rank == 1
    assert available.local_top_k_rank == 1
    assert available.top_k_rank_movement == 0
    assert available.global_coverage_status == "shortfall"
    assert available.local_coverage_status == "shortfall"
    assert available.global_support_coverage_fraction == pytest.approx(2 / 3)
    assert available.local_support_coverage_fraction == pytest.approx(2 / 3)
    assert available.global_top_k_coverage_fraction == pytest.approx(2 / 5)
    assert available.local_top_k_coverage_fraction == pytest.approx(2 / 3)
    assert available.global_observation_independence_fraction == pytest.approx(2 / 3)
    assert available.local_observation_independence_fraction == pytest.approx(1.0)
    assert available.global_support_complete is False
    assert available.local_support_complete is False
    assert available.evidence_fingerprint.startswith("sha256:")

    assert unavailable.candidate_accepted_taxon_key == "gbif:200"
    assert unavailable.disagreement_status == "unavailable"
    assert unavailable.disagreement_unavailable_reason == "no_local_geographic_evidence"
    assert unavailable.global_prototype_similarity == pytest.approx(0.0)
    assert unavailable.local_prototype_similarity is None
    assert unavailable.prototype_signed_difference is None
    assert unavailable.prototype_absolute_disagreement is None
    assert unavailable.local_prototype_rank is None
    assert unavailable.prototype_rank_movement is None
    assert unavailable.local_coverage_status == "unavailable"
    assert unavailable.local_support_coverage_fraction is None
    assert unavailable.local_top_k_coverage_fraction is None
    assert unavailable.local_observation_independence_fraction is None
    assert unavailable.local_support_complete is None
    assert result.global_prototype_top_candidate == "gbif:100"
    assert result.local_prototype_top_candidate == "gbif:100"
    assert result.prototype_top1_agreement is True
    assert result.nearest_top1_agreement is True
    assert result.top_k_top1_agreement is True
    assert result.evidence_set_fingerprint.startswith("sha256:")


def test_dynamic_pool_disagreement_exposes_rank_reversal() -> None:
    candidate_matrix, global_pools = _global_inputs()
    global_scores = score_global_reference_evidence(
        _query(),
        candidate_matrix,
        global_pools,
    )
    local_scores = score_local_reference_evidence(
        _query(),
        candidate_matrix,
        _all_local_inputs(),
    )

    result = calculate_dynamic_pool_disagreement_coverage(
        global_scores,
        local_scores,
    )
    first, second = result.scores

    assert first.global_prototype_rank == 1
    assert first.local_prototype_rank == 2
    assert first.prototype_rank_movement == 1
    assert first.nearest_rank_movement == 1
    assert first.top_k_rank_movement == 1
    assert second.global_prototype_rank == 2
    assert second.local_prototype_rank == 1
    assert second.prototype_rank_movement == -1
    assert second.nearest_rank_movement == -1
    assert second.top_k_rank_movement == -1
    assert result.global_prototype_top_candidate == "gbif:100"
    assert result.local_prototype_top_candidate == "gbif:200"
    assert result.prototype_top1_agreement is False
    assert result.nearest_top1_agreement is False
    assert result.top_k_top1_agreement is False


def test_dynamic_pool_disagreement_retains_all_local_unavailable_state() -> None:
    candidate_matrix, global_pools = _global_inputs()
    global_scores = score_global_reference_evidence(
        _query(),
        candidate_matrix,
        global_pools,
    )
    local_scores = score_local_reference_evidence(
        _query(),
        candidate_matrix,
        _all_local_unavailable_inputs(),
    )

    result = calculate_dynamic_pool_disagreement_coverage(
        global_scores,
        local_scores,
    )

    assert result.local_prototype_top_candidate is None
    assert result.local_nearest_top_candidate is None
    assert result.local_top_k_top_candidate is None
    assert result.prototype_top1_agreement is None
    assert result.nearest_top1_agreement is None
    assert result.top_k_top1_agreement is None
    assert all(score.disagreement_status == "unavailable" for score in result.scores)


def test_dynamic_pool_disagreement_rejects_identity_or_membership_drift() -> None:
    candidate_matrix, global_pools = _global_inputs()
    global_scores = score_global_reference_evidence(
        _query(),
        candidate_matrix,
        global_pools,
    )
    local_scores = score_local_reference_evidence(
        _query(),
        candidate_matrix,
        _local_inputs(),
    )

    with pytest.raises(ValueError, match="differ on query_fingerprint"):
        calculate_dynamic_pool_disagreement_coverage(
            global_scores,
            replace(local_scores, query_fingerprint="sha256:" + "7" * 64),
        )
    with pytest.raises(ValueError, match="candidate memberships differ"):
        calculate_dynamic_pool_disagreement_coverage(
            global_scores,
            replace(local_scores, scores=local_scores.scores[:1]),
        )
    with pytest.raises(ValueError, match="not canonically ordered"):
        calculate_dynamic_pool_disagreement_coverage(
            global_scores,
            replace(local_scores, scores=tuple(reversed(local_scores.scores))),
        )


def test_preserved_dynamic_components_retain_every_source_row_without_fusion() -> None:
    family, global_scores, local_scores, disagreement = _component_inputs()

    result = preserve_dynamic_score_components(
        family,
        global_scores,
        local_scores,
        disagreement,
    )

    assert result.schema_version == DYNAMIC_SCORE_COMPONENT_SET_VERSION
    assert result.family_evidence == family
    assert result.global_evidence_set_fingerprint == global_scores.score_set_fingerprint
    assert result.local_evidence_set_fingerprint == local_scores.score_set_fingerprint
    assert result.disagreement_coverage == disagreement
    assert result.disagreement_coverage_set_fingerprint == (
        disagreement.evidence_set_fingerprint
    )
    assert [candidate.schema_version for candidate in result.candidates] == [
        DYNAMIC_SCORE_COMPONENT_VERSION,
        DYNAMIC_SCORE_COMPONENT_VERSION,
    ]
    for candidate, global_score, local_score, difference in zip(
        result.candidates,
        global_scores.scores,
        local_scores.scores,
        disagreement.scores,
        strict=True,
    ):
        assert candidate.global_evidence == global_score
        assert candidate.local_evidence == local_score
        assert candidate.disagreement_coverage == difference
        assert candidate.component_fingerprint.startswith("sha256:")
        assert not hasattr(candidate, "fused_raw_score")
    assert result.candidates[1].local_evidence.prototype_similarity is None
    assert result.candidates[1].disagreement_coverage.local_prototype_rank is None
    assert result.component_set_fingerprint.startswith("sha256:")


def test_preserved_dynamic_components_are_deterministic_and_validate() -> None:
    evidence = _component_inputs()

    first = preserve_dynamic_score_components(*evidence)
    second = preserve_dynamic_score_components(*evidence)

    assert first == second
    validate_dynamic_score_components(first)


def test_preserved_dynamic_components_reject_source_or_fingerprint_drift() -> None:
    family, global_scores, local_scores, disagreement = _component_inputs()

    with pytest.raises(ValueError, match="query fingerprints differ"):
        preserve_dynamic_score_components(
            replace(family, query_fingerprint="sha256:" + "7" * 64),
            global_scores,
            local_scores,
            disagreement,
        )
    with pytest.raises(ValueError, match="does not match global/local evidence"):
        preserve_dynamic_score_components(
            family,
            global_scores,
            local_scores,
            replace(
                disagreement,
                global_evidence_set_fingerprint="sha256:" + "8" * 64,
            ),
        )
    tampered_family = replace(
        family,
        scores=(
            replace(family.scores[0], family_name="Tampered family"),
            *family.scores[1:],
        ),
    )
    with pytest.raises(ValueError, match="score fingerprint mismatch"):
        preserve_dynamic_score_components(
            tampered_family,
            global_scores,
            local_scores,
            disagreement,
        )
    tampered_global = replace(
        global_scores,
        scores=(
            replace(global_scores.scores[0], query_id="another-query"),
            *global_scores.scores[1:],
        ),
    )
    with pytest.raises(ValueError, match="row query identities differ"):
        preserve_dynamic_score_components(
            family,
            tampered_global,
            local_scores,
            disagreement,
        )


def test_preserved_dynamic_component_validator_rejects_row_tampering() -> None:
    result = preserve_dynamic_score_components(*_component_inputs())
    first = replace(
        result.candidates[0],
        component_fingerprint="sha256:" + "9" * 64,
    )

    with pytest.raises(ValueError, match="component row or fingerprint mismatch"):
        validate_dynamic_score_components(
            replace(result, candidates=(first, *result.candidates[1:])),
        )


def test_raw_fusion_methods_have_exact_numeric_and_missing_local_semantics() -> None:
    components = preserve_dynamic_score_components(*_component_inputs())

    result = evaluate_raw_fusion_methods(components, _linear_parameters())
    scores = {
        (score.method, score.candidate_accepted_taxon_key): score
        for score in result.scores
    }

    assert result.component_set == components
    assert result.methods == RAW_FUSION_METHODS
    assert len(result.scores) == len(RAW_FUSION_METHODS) * 2
    assert not hasattr(result, "selected_method")
    first_values = (1.0, 1.0, (1 + 2**-0.5) / 2, 3 / 10**0.5, 1.0, 0.9)
    assert scores[
        (UNWEIGHTED_COMPONENT_MEAN, "gbif:100")
    ].raw_fusion_score == pytest.approx(sum(first_values) / len(first_values))
    assert scores[
        (VALIDATION_FITTED_LINEAR, "gbif:100")
    ].raw_fusion_score == pytest.approx((1 + 3 / 10**0.5) / 2)
    assert scores[
        (MAXIMUM_SCOPE_EVIDENCE, "gbif:100")
    ].raw_fusion_score == pytest.approx((1 + 1 + (1 + 2**-0.5) / 2) / 3)
    assert scores[
        (ROBUST_RANK_AGGREGATION, "gbif:100")
    ].raw_fusion_score == pytest.approx(1.0)

    for method in RAW_FUSION_METHODS:
        unavailable = scores[(method, "gbif:200")]
        assert unavailable.local_evidence_status == "unavailable"
        assert unavailable.component_names_used == GLOBAL_FUSION_COMPONENTS
    assert scores[
        (UNWEIGHTED_COMPONENT_MEAN, "gbif:200")
    ].raw_fusion_score == pytest.approx(-1 / 6)
    assert scores[
        (VALIDATION_FITTED_LINEAR, "gbif:200")
    ].raw_fusion_score == pytest.approx(0.0)
    assert scores[
        (MAXIMUM_SCOPE_EVIDENCE, "gbif:200")
    ].raw_fusion_score == pytest.approx(-1 / 6)
    assert scores[
        (ROBUST_RANK_AGGREGATION, "gbif:200")
    ].raw_fusion_score == pytest.approx(0.0)
    assert all(score.score_fingerprint.startswith("sha256:") for score in result.scores)
    assert result.score_set_fingerprint.startswith("sha256:")


def test_robust_rank_aggregation_is_tie_aware_across_inverted_components() -> None:
    components = preserve_dynamic_score_components(
        *_component_inputs(local_inputs=_all_local_inputs())
    )

    result = evaluate_raw_fusion_methods(components, _linear_parameters())
    robust = {
        score.candidate_accepted_taxon_key: score
        for score in result.scores
        if score.method == ROBUST_RANK_AGGREGATION
    }

    assert robust["gbif:100"].raw_fusion_score == pytest.approx(0.5)
    assert robust["gbif:200"].raw_fusion_score == pytest.approx(0.5)
    assert robust["gbif:100"].component_names_used == FUSION_COMPONENTS
    assert robust["gbif:100"].method_component_values == pytest.approx(
        (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    )
    assert robust["gbif:200"].method_component_values == pytest.approx(
        (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    )
    assert robust["gbif:100"].raw_component_values == pytest.approx(
        (1.0, 1.0, (1 + 2**-0.5) / 2, 0.0, 0.0, 0.0)
    )


def test_validation_linear_parameters_are_explicit_and_fingerprint_bound() -> None:
    parameters = _linear_parameters()

    assert parameters.full_weights == pytest.approx((0.5, 0, 0, 0.5, 0, 0))
    assert parameters.global_only_weights == pytest.approx((1, 0, 0))
    assert parameters.parameters_fingerprint.startswith("sha256:")
    with pytest.raises(ValueError, match="exactly 6"):
        ValidationLinearFusionParameters(
            validation_artifact_fingerprint="sha256:" + "6" * 64,
            full_weights=(1.0,),
            global_only_weights=(1.0, 0.0, 0.0),
        )
    with pytest.raises(ValueError, match="at least one nonzero"):
        ValidationLinearFusionParameters(
            validation_artifact_fingerprint="sha256:" + "6" * 64,
            full_weights=(0.0,) * 6,
            global_only_weights=(1.0, 0.0, 0.0),
        )
    with pytest.raises(ValueError, match="parameter fingerprint mismatch"):
        replace(parameters, parameters_fingerprint="sha256:" + "7" * 64)


def test_raw_fusion_validator_rejects_score_or_policy_drift() -> None:
    components = preserve_dynamic_score_components(*_component_inputs())
    result = evaluate_raw_fusion_methods(components, _linear_parameters())
    tampered_score = replace(result.scores[0], raw_fusion_score=-0.25)

    with pytest.raises(ValueError, match="does not match components and policy"):
        validate_raw_fusion_scores(
            replace(result, scores=(tampered_score, *result.scores[1:])),
        )

    changed_parameters = ValidationLinearFusionParameters(
        validation_artifact_fingerprint="sha256:" + "8" * 64,
        full_weights=(1 / 6,) * 6,
        global_only_weights=(1 / 3,) * 3,
    )
    changed = evaluate_raw_fusion_methods(components, changed_parameters)
    assert changed.linear_parameters.parameters_fingerprint != (
        result.linear_parameters.parameters_fingerprint
    )
    assert changed.score_set_fingerprint != result.score_set_fingerprint


def test_candidate_rankings_retain_complete_alternatives_for_every_method() -> None:
    fusion_scores = _fusion_scores()

    result = rank_raw_fusion_candidates(fusion_scores)

    assert result.fusion_scores == fusion_scores
    assert result.method_selection_status == "not_selected"
    assert result.cross_method_top1_agreement is True
    assert result.agreed_top_candidate_key == "gbif:100"
    assert not hasattr(result, "selected_method")
    assert [ranking.method for ranking in result.method_rankings] == list(
        RAW_FUSION_METHODS
    )
    for ranking in result.method_rankings:
        assert ranking.candidate_count == 2
        assert ranking.complete_candidate_set is True
        assert ranking.top_candidate_accepted_taxon_key == "gbif:100"
        assert ranking.alternative_candidate_keys == ("gbif:200",)
        assert [candidate.candidate_rank for candidate in ranking.candidates] == [1, 2]
        assert [
            candidate.candidate_accepted_taxon_key for candidate in ranking.candidates
        ] == ["gbif:100", "gbif:200"]
        assert ranking.candidates[0].margin_to_next_raw == pytest.approx(
            ranking.candidates[0].raw_fusion_score
            - ranking.candidates[1].raw_fusion_score
        )
        assert ranking.candidates[1].margin_to_next_raw is None
        assert ranking.ranking_fingerprint.startswith("sha256:")
    assert result.ranking_set_fingerprint.startswith("sha256:")


def test_candidate_rankings_expose_method_disagreement_and_score_ties() -> None:
    fusion_scores = _fusion_scores(local_inputs=_all_local_inputs())

    result = rank_raw_fusion_candidates(fusion_scores)
    rankings = {ranking.method: ranking for ranking in result.method_rankings}

    assert result.cross_method_top1_agreement is False
    assert result.agreed_top_candidate_key is None
    assert rankings[UNWEIGHTED_COMPONENT_MEAN].top_candidate_accepted_taxon_key == (
        "gbif:100"
    )
    assert rankings[VALIDATION_FITTED_LINEAR].top_candidate_accepted_taxon_key == (
        "gbif:100"
    )
    assert rankings[MAXIMUM_SCOPE_EVIDENCE].top_candidate_accepted_taxon_key == (
        "gbif:200"
    )
    assert rankings[ROBUST_RANK_AGGREGATION].top_candidate_accepted_taxon_key == (
        "gbif:100"
    )
    maximum = rankings[MAXIMUM_SCOPE_EVIDENCE]
    assert maximum.alternative_candidate_keys == ("gbif:100",)
    linear = rankings[VALIDATION_FITTED_LINEAR]
    assert linear.top_margin_raw == pytest.approx(0.0)
    assert linear.candidates[0].score_tied_with_previous is False
    assert linear.candidates[0].score_tied_with_next is True
    assert linear.candidates[1].score_tied_with_previous is True
    assert linear.candidates[1].score_tied_with_next is False


def test_candidate_ranking_is_deterministic_and_validator_rejects_drift() -> None:
    fusion_scores = _fusion_scores()

    first = rank_raw_fusion_candidates(fusion_scores)
    second = rank_raw_fusion_candidates(fusion_scores)

    assert first == second
    validate_raw_fusion_rankings(first)
    first_method = first.method_rankings[0]
    tampered_candidate = replace(first_method.candidates[0], candidate_rank=2)
    tampered_method = replace(
        first_method,
        candidates=(tampered_candidate, *first_method.candidates[1:]),
    )
    with pytest.raises(ValueError, match="do not match the fusion score set"):
        validate_raw_fusion_rankings(
            replace(
                first, method_rankings=(tampered_method, *first.method_rankings[1:])
            ),
        )


def test_vector_scoring_stage_consumes_cached_vectors_without_encoder_or_images() -> (
    None
):
    work = _vector_work()

    first = execute_dynamic_vector_scoring(work)
    second = execute_dynamic_vector_scoring(work)

    assert first == second
    assert tuple(inspect.signature(execute_dynamic_vector_scoring).parameters) == (
        "work",
    )
    assert not hasattr(work, "encoder")
    assert not hasattr(work, "image")
    assert first.encoder_invocations == 0
    assert first.image_materializations == 0
    assert first.cached_query_vectors_consumed == 1
    assert first.source_embedding_id == work.source_embedding.embedding_id
    assert first.family_evidence.scores[0].raw_similarity == pytest.approx(1.0)
    assert first.global_evidence.scores[0].prototype_similarity == pytest.approx(1.0)
    assert len(first.fusion_scores.scores) == 8
    assert len(first.rankings.method_rankings) == 4
    assert first.result_fingerprint.startswith("sha256:")


def test_vector_scoring_execution_invokes_each_matrix_scorer_once(monkeypatch) -> None:
    calls = {"family": 0, "global": 0, "local": 0}
    original_family = dynamic_pool_compute.score_family_evidence
    original_global = dynamic_pool_compute.score_global_reference_evidence
    original_local = dynamic_pool_compute.score_local_reference_evidence

    def counted_family(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - test spy.
        calls["family"] += 1
        return original_family(*args, **kwargs)

    def counted_global(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - test spy.
        calls["global"] += 1
        return original_global(*args, **kwargs)

    def counted_local(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - test spy.
        calls["local"] += 1
        return original_local(*args, **kwargs)

    monkeypatch.setattr(dynamic_pool_compute, "score_family_evidence", counted_family)
    monkeypatch.setattr(
        dynamic_pool_compute,
        "score_global_reference_evidence",
        counted_global,
    )
    monkeypatch.setattr(
        dynamic_pool_compute,
        "score_local_reference_evidence",
        counted_local,
    )

    execute_dynamic_vector_scoring(_vector_work())

    assert calls == {"family": 1, "global": 1, "local": 1}


def test_vector_scoring_work_normalizes_cache_vector_and_pool_order() -> None:
    candidate_matrix, global_pools = _global_inputs()
    local_pools = _local_inputs()

    first = build_dynamic_vector_scoring_work(
        _cached_embedding(embedding=(3.0, 4.0)),
        query_id="cached-query:1",
        route="adult_field",
        family_matrix=_family_matrix(),
        candidate_matrix=candidate_matrix,
        global_pools=global_pools,
        local_pools=local_pools,
        linear_parameters=_linear_parameters(),
    )
    second = build_dynamic_vector_scoring_work(
        _cached_embedding(embedding=(3.0, 4.0)),
        query_id="cached-query:1",
        route="adult_field",
        family_matrix=_family_matrix(),
        candidate_matrix=candidate_matrix,
        global_pools=tuple(reversed(global_pools)),
        local_pools=tuple(reversed(local_pools)),
        linear_parameters=_linear_parameters(),
    )

    assert first == second
    assert first.query.embedding == pytest.approx((0.6, 0.8))
    assert first.query.query_embedding_fingerprint != (
        first.source_embedding.embedding_fingerprint
    )
    assert first.work_fingerprint.startswith("sha256:")
    validate_dynamic_vector_scoring_work(first)


def test_vector_scoring_boundary_rejects_source_query_and_result_drift() -> None:
    work = _vector_work()

    with pytest.raises(ValueError, match="cached embedding fingerprint mismatch"):
        build_dynamic_vector_scoring_work(
            replace(
                work.source_embedding,
                embedding_fingerprint="sha256:" + "f" * 64,
            ),
            query_id=work.query.query_id,
            route=work.query.route,
            family_matrix=work.family_matrix,
            candidate_matrix=work.candidate_matrix,
            global_pools=work.global_pools,
            local_pools=work.local_pools,
            linear_parameters=work.linear_parameters,
        )
    drifted_query = replace(
        work.query,
        embedding=(0.0, 1.0),
        query_fingerprint=None,
    )
    with pytest.raises(ValueError, match="query differs from cached embedding"):
        validate_dynamic_vector_scoring_work(replace(work, query=drifted_query))

    result = execute_dynamic_vector_scoring(work)
    with pytest.raises(ValueError, match="result does not match its work"):
        validate_dynamic_vector_scoring_result(
            replace(result, encoder_invocations=1),
        )


def _query() -> RawScoringQuery:
    return RawScoringQuery(
        query_id="flickr-embedding:query",
        query_embedding_fingerprint="sha256:" + "f" * 64,
        route="adult_field",
        visual_input_kind="focused_full_frame",
        model_fingerprint=_MODEL_FINGERPRINT,
        embedding=(1.0, 0.0),
    )


def _component_inputs(
    *,
    local_inputs: tuple[LocalReferencePoolInput, ...] | None = None,
):
    candidate_matrix, global_pools = _global_inputs()
    family = score_family_evidence(_query(), _family_matrix())
    global_scores = score_global_reference_evidence(
        _query(),
        candidate_matrix,
        global_pools,
    )
    local_scores = score_local_reference_evidence(
        _query(),
        candidate_matrix,
        local_inputs or _local_inputs(),
    )
    disagreement = calculate_dynamic_pool_disagreement_coverage(
        global_scores,
        local_scores,
    )
    return family, global_scores, local_scores, disagreement


def _linear_parameters() -> ValidationLinearFusionParameters:
    return ValidationLinearFusionParameters(
        validation_artifact_fingerprint="sha256:" + "6" * 64,
        full_weights=(0.5, 0.0, 0.0, 0.5, 0.0, 0.0),
        global_only_weights=(1.0, 0.0, 0.0),
    )


def _fusion_scores(
    *,
    local_inputs: tuple[LocalReferencePoolInput, ...] | None = None,
):
    components = preserve_dynamic_score_components(
        *_component_inputs(local_inputs=local_inputs)
    )
    return evaluate_raw_fusion_methods(components, _linear_parameters())


def _vector_work():
    candidate_matrix, global_pools = _global_inputs()
    return build_dynamic_vector_scoring_work(
        _cached_embedding(),
        query_id="cached-query:1",
        route="adult_field",
        family_matrix=_family_matrix(),
        candidate_matrix=candidate_matrix,
        global_pools=global_pools,
        local_pools=_local_inputs(),
        linear_parameters=_linear_parameters(),
    )


def _cached_embedding(
    *,
    embedding: tuple[float, ...] = (1.0, 0.0),
) -> RawFullFrameEmbedding:
    embedding_id = "sha256:" + "1" * 64
    embedding_version = "target-full-frame-embedding-v3"
    embedding_fingerprint = canonical_semantic_fingerprint(
        {
            "embedding": embedding,
            "embedding_id": embedding_id,
            "embedding_version": embedding_version,
        }
    )
    return RawFullFrameEmbedding(
        embedding_id=embedding_id,
        embedding_version=embedding_version,
        embedding_fingerprint=embedding_fingerprint,
        visual_input_id="sha256:" + "2" * 64,
        visual_input_kind="focused_full_frame",
        raw_image_content_hash="sha256:" + "3" * 64,
        transformation_fingerprint="sha256:" + "4" * 64,
        model_fingerprint=_MODEL_FINGERPRINT,
        image_resize_mode="longest",
        preprocessing_contract_fingerprint="sha256:" + "5" * 64,
        preprocessing_fingerprint="sha256:" + "6" * 64,
        embedding_dimension=len(embedding),
        embedding=embedding,
        embedding_norm=sum(value * value for value in embedding) ** 0.5,
    )


def _family_matrix(
    rows: tuple[FamilyPrototypeVector, ...] | None = None,
    *,
    source_fingerprint: str = _PROTOTYPE_SET_FINGERPRINT,
):
    return FamilyPrototypeMatrixCache().get_or_build(
        route="adult_field",
        visual_input_kind="focused_full_frame",
        family_partition="all-families",
        model_fingerprint=_MODEL_FINGERPRINT,
        family_prototype_set_fingerprint=source_fingerprint,
        prototypes=rows or _family_rows(),
    )


def _family_rows() -> tuple[FamilyPrototypeVector, ...]:
    return (
        FamilyPrototypeVector(
            family_key="gbif:7017",
            family_name="Nymphalidae",
            prototype_fingerprint="sha256:" + "1" * 64,
            embedding=(0.0, 1.0),
        ),
        FamilyPrototypeVector(
            family_key="gbif:9417",
            family_name="Papilionidae",
            prototype_fingerprint="sha256:" + "2" * 64,
            embedding=(1.0, 0.0),
        ),
    )


def _global_inputs():
    cache = DynamicPoolMatrixCache()
    candidates = cache.get_candidate_matrix(
        route="adult_field",
        visual_input_kind="focused_full_frame",
        family_partition="all-families",
        model_fingerprint=_MODEL_FINGERPRINT,
        candidate_set_fingerprint=_CANDIDATE_SET_FINGERPRINT,
        reference_prototype_artifact_fingerprint=_REFERENCE_PROTOTYPE_FINGERPRINT,
        candidates=(
            CandidatePrototypeVector(
                accepted_taxon_key="gbif:200",
                scientific_name="Papilio machaon",
                prototype_fingerprint="sha256:" + "4" * 64,
                embedding=(0.0, 1.0),
            ),
            CandidatePrototypeVector(
                accepted_taxon_key="gbif:100",
                scientific_name="Papilio demoleus",
                prototype_fingerprint="sha256:" + "5" * 64,
                embedding=(1.0, 0.0),
            ),
        ),
    )
    pools = (
        GlobalReferencePoolInput(
            candidate_accepted_taxon_key="gbif:100",
            candidate_scientific_name="Papilio demoleus",
            pool_matrix=_pool_matrix(
                cache,
                candidate_key="gbif:100",
                geographic_scope="global",
                rows=(
                    PoolReferenceVector(
                        reference_media_id="reference-media:1",
                        reference_observation_id="reference-observation:1",
                        member_fingerprint="sha256:" + "1" * 64,
                        reference_embedding_fingerprint="sha256:" + "2" * 64,
                        embedding=(1.0, 0.0),
                    ),
                    PoolReferenceVector(
                        reference_media_id="reference-media:2a",
                        reference_observation_id="reference-observation:2",
                        member_fingerprint="sha256:" + "3" * 64,
                        reference_embedding_fingerprint="sha256:" + "4" * 64,
                        embedding=(0.8, 0.6),
                    ),
                    PoolReferenceVector(
                        reference_media_id="reference-media:2b",
                        reference_observation_id="reference-observation:2",
                        member_fingerprint="sha256:" + "5" * 64,
                        reference_embedding_fingerprint="sha256:" + "6" * 64,
                        embedding=(0.6, 0.8),
                    ),
                ),
                fingerprint_digit="6",
            ),
            configured_reference_count=3,
            configured_top_k=5,
        ),
        GlobalReferencePoolInput(
            candidate_accepted_taxon_key="gbif:200",
            candidate_scientific_name="Papilio machaon",
            pool_matrix=_pool_matrix(
                cache,
                candidate_key="gbif:200",
                geographic_scope="global",
                rows=(
                    PoolReferenceVector(
                        reference_media_id="reference-media:3",
                        reference_observation_id="reference-observation:3",
                        member_fingerprint="sha256:" + "7" * 64,
                        reference_embedding_fingerprint="sha256:" + "8" * 64,
                        embedding=(0.0, 1.0),
                    ),
                    PoolReferenceVector(
                        reference_media_id="reference-media:4",
                        reference_observation_id="reference-observation:4",
                        member_fingerprint="sha256:" + "9" * 64,
                        reference_embedding_fingerprint="sha256:" + "a" * 64,
                        embedding=(-1.0, 0.0),
                    ),
                ),
                fingerprint_digit="7",
            ),
            configured_reference_count=3,
            configured_top_k=5,
        ),
    )
    return candidates, pools


def _pool_matrix(
    cache: DynamicPoolMatrixCache,
    *,
    candidate_key: str,
    geographic_scope: str,
    rows: tuple[PoolReferenceVector, ...],
    fingerprint_digit: str,
):
    return cache.get_pool_matrix(
        route="adult_field",
        visual_input_kind="focused_full_frame",
        geographic_scope=geographic_scope,
        candidate_accepted_taxon_key=candidate_key,
        model_fingerprint=_MODEL_FINGERPRINT,
        reference_embedding_artifact_fingerprint=_REFERENCE_EMBEDDING_FINGERPRINT,
        pool_membership_fingerprint="sha256:" + fingerprint_digit * 64,
        pool_ids=(f"dynamic-reference-pool:{candidate_key}:{geographic_scope}",),
        references=rows,
    )


def _reference_rows(prefix: str) -> tuple[PoolReferenceVector, ...]:
    return (
        PoolReferenceVector(
            reference_media_id=f"reference-media:{prefix}",
            reference_observation_id=f"reference-observation:{prefix}",
            member_fingerprint="sha256:" + "b" * 64,
            reference_embedding_fingerprint="sha256:" + "c" * 64,
            embedding=(1.0, 0.0),
        ),
    )


def _local_inputs() -> tuple[LocalReferencePoolInput, ...]:
    cache = DynamicPoolMatrixCache()
    return (
        LocalReferencePoolInput(
            candidate_accepted_taxon_key="gbif:100",
            candidate_scientific_name="Papilio demoleus",
            local_pool_status="available",
            local_pool_unavailable_reason=None,
            pool_matrix=_pool_matrix(
                cache,
                candidate_key="gbif:100",
                geographic_scope="exact_local_cell",
                rows=(
                    PoolReferenceVector(
                        reference_media_id="reference-media:local-1",
                        reference_observation_id="reference-observation:local-1",
                        member_fingerprint="sha256:" + "d" * 64,
                        reference_embedding_fingerprint="sha256:" + "e" * 64,
                        embedding=(1.0, 0.0),
                    ),
                    PoolReferenceVector(
                        reference_media_id="reference-media:local-2",
                        reference_observation_id="reference-observation:local-2",
                        member_fingerprint="sha256:" + "f" * 64,
                        reference_embedding_fingerprint="sha256:" + "0" * 64,
                        embedding=(0.8, 0.6),
                    ),
                ),
                fingerprint_digit="1",
            ),
            configured_reference_count=3,
            configured_top_k=3,
        ),
        LocalReferencePoolInput(
            candidate_accepted_taxon_key="gbif:200",
            candidate_scientific_name="Papilio machaon",
            local_pool_status="unavailable",
            local_pool_unavailable_reason="no_local_geographic_evidence",
            pool_matrix=None,
            configured_reference_count=2,
            configured_top_k=3,
        ),
    )


def _all_local_inputs() -> tuple[LocalReferencePoolInput, ...]:
    cache = DynamicPoolMatrixCache()
    return (
        LocalReferencePoolInput(
            candidate_accepted_taxon_key="gbif:100",
            candidate_scientific_name="Papilio demoleus",
            local_pool_status="available",
            local_pool_unavailable_reason=None,
            pool_matrix=_pool_matrix(
                cache,
                candidate_key="gbif:100",
                geographic_scope="exact_local_cell",
                rows=(
                    PoolReferenceVector(
                        reference_media_id="reference-media:local-100",
                        reference_observation_id="reference-observation:local-100",
                        member_fingerprint="sha256:" + "1" * 64,
                        reference_embedding_fingerprint="sha256:" + "2" * 64,
                        embedding=(0.0, 1.0),
                    ),
                ),
                fingerprint_digit="3",
            ),
            configured_reference_count=1,
            configured_top_k=1,
        ),
        LocalReferencePoolInput(
            candidate_accepted_taxon_key="gbif:200",
            candidate_scientific_name="Papilio machaon",
            local_pool_status="available",
            local_pool_unavailable_reason=None,
            pool_matrix=_pool_matrix(
                cache,
                candidate_key="gbif:200",
                geographic_scope="exact_local_cell",
                rows=(
                    PoolReferenceVector(
                        reference_media_id="reference-media:local-200",
                        reference_observation_id="reference-observation:local-200",
                        member_fingerprint="sha256:" + "4" * 64,
                        reference_embedding_fingerprint="sha256:" + "5" * 64,
                        embedding=(1.0, 0.0),
                    ),
                ),
                fingerprint_digit="6",
            ),
            configured_reference_count=1,
            configured_top_k=1,
        ),
    )


def _all_local_unavailable_inputs() -> tuple[LocalReferencePoolInput, ...]:
    return tuple(
        LocalReferencePoolInput(
            candidate_accepted_taxon_key=candidate_key,
            candidate_scientific_name=scientific_name,
            local_pool_status="unavailable",
            local_pool_unavailable_reason="no_local_geographic_evidence",
            pool_matrix=None,
            configured_reference_count=2,
            configured_top_k=3,
        )
        for candidate_key, scientific_name in (
            ("gbif:100", "Papilio demoleus"),
            ("gbif:200", "Papilio machaon"),
        )
    )


def _local_input(
    *,
    status: str = "available",
    pool_matrix=None,
    unavailable_reason: str | None = None,
) -> LocalReferencePoolInput:
    return LocalReferencePoolInput(
        candidate_accepted_taxon_key="gbif:100",
        candidate_scientific_name="Papilio demoleus",
        local_pool_status=status,
        local_pool_unavailable_reason=unavailable_reason,
        pool_matrix=pool_matrix,
        configured_reference_count=3,
        configured_top_k=3,
    )
