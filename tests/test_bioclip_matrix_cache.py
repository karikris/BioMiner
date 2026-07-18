"""Tests for deterministic BioCLIP vector-matrix caches."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from biominer.bioclip.matrix_cache import (
    CandidatePrototypeVector,
    DynamicPoolMatrixCache,
    FamilyPrototypeMatrixCache,
    FamilyPrototypeVector,
    PoolReferenceVector,
    candidate_matrix_signature,
    candidate_pool_signature,
    family_matrix_signature,
    pool_matrix_signature,
)


_MODEL_FINGERPRINT = "sha256:" + "a" * 64
_PROTOTYPE_SET_FINGERPRINT = "sha256:" + "b" * 64
_CANDIDATE_SET_FINGERPRINT = "sha256:" + "c" * 64
_REFERENCE_PROTOTYPE_FINGERPRINT = "sha256:" + "d" * 64
_REFERENCE_EMBEDDING_FINGERPRINT = "sha256:" + "e" * 64
_POOL_MEMBERSHIP_FINGERPRINT = "sha256:" + "f" * 64


def test_family_matrix_reuses_canonical_content_and_reports_cache_metrics() -> None:
    cache = FamilyPrototypeMatrixCache()
    rows = _family_rows()

    first = _family_matrix(cache, rows)
    second = _family_matrix(cache, tuple(reversed(rows)))

    assert second is first
    assert first.row_ids == ("gbif:7017", "gbif:9417")
    assert first.row_names == ("Nymphalidae", "Papilionidae")
    assert first.row_count == 2
    assert first.embedding_dimension == 2
    assert first.byte_count == 16
    assert first.float32_buffer.readonly
    assert first.vector(0) == pytest.approx((0.0, 1.0))
    assert first.vector(1) == pytest.approx((1.0, 0.0))
    assert first.matrix_signature == family_matrix_signature(
        route="adult_field",
        visual_input_kind="focused_full_frame",
        family_partition="all-configured-families",
        model_fingerprint=_MODEL_FINGERPRINT,
        family_prototype_set_fingerprint=_PROTOTYPE_SET_FINGERPRINT,
        prototypes=tuple(reversed(rows)),
    )

    metrics = cache.cache_metrics()
    assert metrics.requests == 2
    assert metrics.hits == 1
    assert metrics.misses == 1
    assert metrics.materializations == 1
    assert metrics.entries == 1
    assert metrics.rows_materialized == 2
    assert metrics.bytes_materialized == 16
    assert metrics.evictions == 0
    assert metrics.hit_rate == 0.5
    assert metrics.as_dict()["family_matrix_cache_hit_rate"] == 0.5


def test_family_matrix_signature_binds_context_and_float32_content() -> None:
    rows = _family_rows()
    baseline = _signature(rows)

    assert _signature(rows, family_partition="papilionoidea") != baseline
    assert _signature(rows, visual_input_kind="raw_full_image") != baseline
    assert (
        _signature(
            (
                rows[0],
                FamilyPrototypeVector(
                    family_key=rows[1].family_key,
                    family_name=rows[1].family_name,
                    prototype_fingerprint="sha256:" + "c" * 64,
                    embedding=rows[1].embedding,
                ),
            )
        )
        != baseline
    )


def test_family_matrix_cache_is_bounded_and_evicts_least_recently_used() -> None:
    cache = FamilyPrototypeMatrixCache(maximum_entries=2)
    first = _family_matrix(cache, _family_rows(), family_partition="first")
    second = _family_matrix(cache, _family_rows(), family_partition="second")
    assert _family_matrix(cache, _family_rows(), family_partition="first") is first

    _family_matrix(cache, _family_rows(), family_partition="third")
    rebuilt_second = _family_matrix(cache, _family_rows(), family_partition="second")

    assert rebuilt_second is not second
    metrics = cache.cache_metrics()
    assert metrics.entries == 2
    assert metrics.requests == 5
    assert metrics.hits == 1
    assert metrics.misses == 4
    assert metrics.evictions == 2


def test_concurrent_family_matrix_requests_materialize_once() -> None:
    cache = FamilyPrototypeMatrixCache()

    with ThreadPoolExecutor(max_workers=8) as executor:
        matrices = tuple(
            executor.map(
                lambda _index: _family_matrix(cache, _family_rows()),
                range(32),
            )
        )

    assert len({id(matrix) for matrix in matrices}) == 1
    metrics = cache.cache_metrics()
    assert metrics.requests == 32
    assert metrics.hits == 31
    assert metrics.misses == 1
    assert metrics.materializations == 1


def test_family_matrix_rejects_ambiguous_or_invalid_vectors() -> None:
    cache = FamilyPrototypeMatrixCache()
    rows = _family_rows()
    duplicate = (
        rows[0],
        FamilyPrototypeVector(
            family_key=rows[0].family_key,
            family_name="Conflicting name",
            prototype_fingerprint="sha256:" + "d" * 64,
            embedding=(1.0, 0.0),
        ),
    )
    with pytest.raises(ValueError, match="duplicate family keys"):
        _family_matrix(cache, duplicate)
    with pytest.raises(ValueError, match="mixed embedding dimensions"):
        _family_matrix(
            cache,
            (
                rows[0],
                FamilyPrototypeVector(
                    family_key="gbif:other",
                    family_name="Otheridae",
                    prototype_fingerprint="sha256:" + "e" * 64,
                    embedding=(1.0, 0.0, 0.0),
                ),
            ),
        )
    with pytest.raises(ValueError, match="unit-normalized"):
        FamilyPrototypeVector(
            family_key="gbif:bad",
            family_name="Badidae",
            prototype_fingerprint="sha256:" + "f" * 64,
            embedding=(2.0, 0.0),
        )
    with pytest.raises(ValueError, match="canonical sha256"):
        _family_matrix(cache, rows, model_fingerprint="not-a-fingerprint")


def test_dynamic_cache_reuses_candidate_and_pool_signatures_independently() -> None:
    cache = DynamicPoolMatrixCache()
    candidates = _candidate_rows()
    references = _pool_rows()

    first_candidates = _candidate_matrix(cache, candidates)
    second_candidates = _candidate_matrix(cache, tuple(reversed(candidates)))
    first_pool = _pool_matrix(cache, references)
    second_pool = _pool_matrix(
        cache,
        tuple(reversed(references)),
        pool_ids=("dynamic-reference-pool:local", "dynamic-reference-pool:global"),
    )

    assert second_candidates is first_candidates
    assert second_pool is first_pool
    assert first_candidates.matrix_kind == "candidate_prototype"
    assert first_candidates.row_ids == ("gbif:100", "gbif:200")
    assert first_pool.matrix_kind == "dynamic_reference_pool"
    assert first_pool.row_ids == ("reference-media:1", "reference-media:2")
    assert first_pool.row_names == (
        "reference-observation:1",
        "reference-observation:2",
    )

    metrics = cache.cache_metrics()
    assert metrics.candidate.requests == 2
    assert metrics.candidate.hits == 1
    assert metrics.candidate.misses == 1
    assert metrics.pool.requests == 2
    assert metrics.pool.hits == 1
    assert metrics.pool.misses == 1
    assert metrics.requests == 4
    assert metrics.hits == 2
    assert metrics.misses == 2
    assert metrics.hit_rate == 0.5
    assert metrics.as_dict() == {
        "candidate_matrix_requests": 2,
        "candidate_matrix_cache_hits": 1,
        "candidate_matrix_cache_misses": 1,
        "candidate_matrix_materializations": 1,
        "candidate_matrix_cache_entries": 1,
        "candidate_matrix_rows_materialized": 2,
        "candidate_matrix_bytes_materialized": 16,
        "candidate_matrix_cache_evictions": 0,
        "candidate_matrix_cache_hit_rate": 0.5,
        "pool_matrix_requests": 2,
        "pool_matrix_cache_hits": 1,
        "pool_matrix_cache_misses": 1,
        "pool_matrix_materializations": 1,
        "pool_matrix_cache_entries": 1,
        "pool_matrix_rows_materialized": 2,
        "pool_matrix_bytes_materialized": 16,
        "pool_matrix_cache_evictions": 0,
        "pool_matrix_cache_hit_rate": 0.5,
        "dynamic_matrix_requests": 4,
        "dynamic_matrix_cache_hits": 2,
        "dynamic_matrix_cache_misses": 2,
        "dynamic_matrix_cache_hit_rate": 0.5,
        "dynamic_matrix_materializations": 2,
        "dynamic_matrix_cache_entries": 2,
        "dynamic_matrix_rows_materialized": 4,
        "dynamic_matrix_bytes_materialized": 32,
        "dynamic_matrix_cache_evictions": 0,
    }


def test_candidate_and_pool_signatures_bind_all_semantic_inputs() -> None:
    candidate = _candidate_signature(_candidate_rows())
    pool = _pool_signature(_pool_rows())

    assert (
        _candidate_signature(
            _candidate_rows(),
            candidate_set_fingerprint="sha256:" + "0" * 64,
        )
        != candidate
    )
    assert (
        _pool_signature(
            _pool_rows(),
            geographic_scope="exact_local_cell",
        )
        != pool
    )
    assert (
        _pool_signature(
            _pool_rows(),
            pool_membership_fingerprint="sha256:" + "9" * 64,
        )
        != pool
    )
    assert candidate_pool_signature(candidate, (pool,)) == candidate_pool_signature(
        candidate,
        tuple(reversed((pool,))),
    )
    with pytest.raises(ValueError, match="unique values"):
        candidate_pool_signature(candidate, (pool, pool))


def test_dynamic_candidate_and_pool_capacities_are_independent() -> None:
    cache = DynamicPoolMatrixCache(
        maximum_candidate_entries=1,
        maximum_pool_entries=1,
    )
    first_candidate = _candidate_matrix(
        cache, _candidate_rows(), family_partition="one"
    )
    _candidate_matrix(cache, _candidate_rows(), family_partition="two")
    rebuilt_candidate = _candidate_matrix(
        cache,
        _candidate_rows(),
        family_partition="one",
    )
    first_pool = _pool_matrix(cache, _pool_rows(), geographic_scope="global")
    _pool_matrix(cache, _pool_rows(), geographic_scope="exact_local_cell")
    rebuilt_pool = _pool_matrix(cache, _pool_rows(), geographic_scope="global")

    assert rebuilt_candidate is not first_candidate
    assert rebuilt_pool is not first_pool
    metrics = cache.cache_metrics()
    assert metrics.candidate.entries == 1
    assert metrics.candidate.evictions == 2
    assert metrics.pool.entries == 1
    assert metrics.pool.evictions == 2


def test_dynamic_matrices_reject_duplicate_and_mixed_dimension_rows() -> None:
    cache = DynamicPoolMatrixCache()
    candidates = _candidate_rows()
    duplicate_candidate = (
        candidates[0],
        CandidatePrototypeVector(
            accepted_taxon_key=candidates[0].accepted_taxon_key,
            scientific_name="Conflicting species",
            prototype_fingerprint="sha256:" + "3" * 64,
            embedding=(0.0, 1.0),
        ),
    )
    with pytest.raises(ValueError, match="duplicate accepted taxon keys"):
        _candidate_matrix(cache, duplicate_candidate)

    references = _pool_rows()
    duplicate_reference = (
        references[0],
        PoolReferenceVector(
            reference_media_id=references[0].reference_media_id,
            reference_observation_id="reference-observation:other",
            member_fingerprint="sha256:" + "4" * 64,
            reference_embedding_fingerprint="sha256:" + "5" * 64,
            embedding=(0.0, 1.0),
        ),
    )
    with pytest.raises(ValueError, match="duplicate reference media IDs"):
        _pool_matrix(cache, duplicate_reference)
    with pytest.raises(ValueError, match="mixed embedding dimensions"):
        _pool_matrix(
            cache,
            (
                references[0],
                PoolReferenceVector(
                    reference_media_id="reference-media:3",
                    reference_observation_id="reference-observation:3",
                    member_fingerprint="sha256:" + "6" * 64,
                    reference_embedding_fingerprint="sha256:" + "7" * 64,
                    embedding=(1.0, 0.0, 0.0),
                ),
            ),
        )


def _family_rows() -> tuple[FamilyPrototypeVector, ...]:
    return (
        FamilyPrototypeVector(
            family_key="gbif:9417",
            family_name="Papilionidae",
            prototype_fingerprint="sha256:" + "1" * 64,
            embedding=(1.0, 0.0),
        ),
        FamilyPrototypeVector(
            family_key="gbif:7017",
            family_name="Nymphalidae",
            prototype_fingerprint="sha256:" + "2" * 64,
            embedding=(0.0, 1.0),
        ),
    )


def _candidate_rows() -> tuple[CandidatePrototypeVector, ...]:
    return (
        CandidatePrototypeVector(
            accepted_taxon_key="gbif:200",
            scientific_name="Papilio machaon",
            prototype_fingerprint="sha256:" + "8" * 64,
            embedding=(0.0, 1.0),
        ),
        CandidatePrototypeVector(
            accepted_taxon_key="gbif:100",
            scientific_name="Papilio demoleus",
            prototype_fingerprint="sha256:" + "7" * 64,
            embedding=(1.0, 0.0),
        ),
    )


def _pool_rows() -> tuple[PoolReferenceVector, ...]:
    return (
        PoolReferenceVector(
            reference_media_id="reference-media:2",
            reference_observation_id="reference-observation:2",
            member_fingerprint="sha256:" + "6" * 64,
            reference_embedding_fingerprint="sha256:" + "5" * 64,
            embedding=(0.0, 1.0),
        ),
        PoolReferenceVector(
            reference_media_id="reference-media:1",
            reference_observation_id="reference-observation:1",
            member_fingerprint="sha256:" + "4" * 64,
            reference_embedding_fingerprint="sha256:" + "3" * 64,
            embedding=(1.0, 0.0),
        ),
    )


def _family_matrix(
    cache: FamilyPrototypeMatrixCache,
    rows: tuple[FamilyPrototypeVector, ...],
    *,
    family_partition: str = "all-configured-families",
    visual_input_kind: str = "focused_full_frame",
    model_fingerprint: str = _MODEL_FINGERPRINT,
):
    return cache.get_or_build(
        route="adult_field",
        visual_input_kind=visual_input_kind,
        family_partition=family_partition,
        model_fingerprint=model_fingerprint,
        family_prototype_set_fingerprint=_PROTOTYPE_SET_FINGERPRINT,
        prototypes=rows,
    )


def _signature(
    rows: tuple[FamilyPrototypeVector, ...],
    *,
    family_partition: str = "all-configured-families",
    visual_input_kind: str = "focused_full_frame",
) -> str:
    return family_matrix_signature(
        route="adult_field",
        visual_input_kind=visual_input_kind,
        family_partition=family_partition,
        model_fingerprint=_MODEL_FINGERPRINT,
        family_prototype_set_fingerprint=_PROTOTYPE_SET_FINGERPRINT,
        prototypes=rows,
    )


def _candidate_matrix(
    cache: DynamicPoolMatrixCache,
    rows: tuple[CandidatePrototypeVector, ...],
    *,
    family_partition: str = "papilionoidea",
):
    return cache.get_candidate_matrix(
        route="adult_field",
        visual_input_kind="focused_full_frame",
        family_partition=family_partition,
        model_fingerprint=_MODEL_FINGERPRINT,
        candidate_set_fingerprint=_CANDIDATE_SET_FINGERPRINT,
        reference_prototype_artifact_fingerprint=_REFERENCE_PROTOTYPE_FINGERPRINT,
        candidates=rows,
    )


def _pool_matrix(
    cache: DynamicPoolMatrixCache,
    rows: tuple[PoolReferenceVector, ...],
    *,
    geographic_scope: str = "global",
    pool_ids: tuple[str, ...] = (
        "dynamic-reference-pool:global",
        "dynamic-reference-pool:local",
    ),
):
    return cache.get_pool_matrix(
        route="adult_field",
        visual_input_kind="focused_full_frame",
        geographic_scope=geographic_scope,
        candidate_accepted_taxon_key="gbif:100",
        model_fingerprint=_MODEL_FINGERPRINT,
        reference_embedding_artifact_fingerprint=_REFERENCE_EMBEDDING_FINGERPRINT,
        pool_membership_fingerprint=_POOL_MEMBERSHIP_FINGERPRINT,
        pool_ids=pool_ids,
        references=rows,
    )


def _candidate_signature(
    rows: tuple[CandidatePrototypeVector, ...],
    *,
    candidate_set_fingerprint: str = _CANDIDATE_SET_FINGERPRINT,
) -> str:
    return candidate_matrix_signature(
        route="adult_field",
        visual_input_kind="focused_full_frame",
        family_partition="papilionoidea",
        model_fingerprint=_MODEL_FINGERPRINT,
        candidate_set_fingerprint=candidate_set_fingerprint,
        reference_prototype_artifact_fingerprint=_REFERENCE_PROTOTYPE_FINGERPRINT,
        candidates=rows,
    )


def _pool_signature(
    rows: tuple[PoolReferenceVector, ...],
    *,
    geographic_scope: str = "global",
    pool_membership_fingerprint: str = _POOL_MEMBERSHIP_FINGERPRINT,
) -> str:
    return pool_matrix_signature(
        route="adult_field",
        visual_input_kind="focused_full_frame",
        geographic_scope=geographic_scope,
        candidate_accepted_taxon_key="gbif:100",
        model_fingerprint=_MODEL_FINGERPRINT,
        reference_embedding_artifact_fingerprint=_REFERENCE_EMBEDDING_FINGERPRINT,
        pool_membership_fingerprint=pool_membership_fingerprint,
        pool_ids=("dynamic-reference-pool:global",),
        references=rows,
    )
