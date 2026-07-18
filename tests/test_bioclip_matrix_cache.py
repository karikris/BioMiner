"""Tests for deterministic BioCLIP vector-matrix caches."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from biominer.bioclip.matrix_cache import (
    FamilyPrototypeMatrixCache,
    FamilyPrototypeVector,
    family_matrix_signature,
)


_MODEL_FINGERPRINT = "sha256:" + "a" * 64
_PROTOTYPE_SET_FINGERPRINT = "sha256:" + "b" * 64


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
