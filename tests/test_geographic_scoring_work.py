"""Tests for deterministic cache-local geographic scoring work ordering."""

from __future__ import annotations

from dataclasses import replace

import pytest

from biominer.bioclip.matrix_cache import candidate_pool_signature
from biominer.run.geographic_scoring_work import (
    GeographicScoringWorkItem,
    sort_geographic_scoring_work,
)


def test_work_is_sorted_by_required_locality_fields_deterministically() -> None:
    items = (
        _item("six", route="larval", result_order_key=("06",)),
        _item("five", visual_input_kind="raw_full_image", result_order_key=("05",)),
        _item("four", family_partition="b-family", result_order_key=("04",)),
        _item(
            "three",
            family_partition="b-family",
            geographic_scope="global",
            candidate_digit="3",
            pool_digit="4",
            result_order_key=("03",),
        ),
        _item(
            "two",
            family_partition="b-family",
            geographic_scope="exact_local_cell",
            result_order_key=("02",),
        ),
        _item("one", family_partition="a-family", result_order_key=("01",)),
    )

    first = sort_geographic_scoring_work(items)
    second = sort_geographic_scoring_work(tuple(reversed(items)))

    assert tuple(item.execution_sort_key for item in first.items) == tuple(
        sorted(item.execution_sort_key for item in items)
    )
    assert [item.work_id for item in first.items] == [
        item.work_id for item in second.items
    ]
    assert first.ordering_fingerprint == second.ordering_fingerprint
    assert [item.work_id for item in first.canonical_result_order()] == [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
    ]
    assert first.metrics.execution_order_changed
    assert second.metrics.execution_order_changed == (
        tuple(item.work_id for item in reversed(items))
        != tuple(item.work_id for item in second.items)
    )


def test_ordering_groups_equal_signatures_and_measures_reuse() -> None:
    repeated = (
        _item("repeat-3", result_order_key=("03",)),
        _item("other", candidate_digit="3", pool_digit="4", result_order_key=("04",)),
        _item("repeat-1", result_order_key=("01",)),
        _item("repeat-2", result_order_key=("02",)),
    )

    order = sort_geographic_scoring_work(repeated)
    metrics = order.metrics
    repeated_positions = [
        index
        for index, item in enumerate(order.items)
        if item.candidate_matrix_signature == _sha("1")
    ]

    assert repeated_positions == list(
        range(repeated_positions[0], repeated_positions[0] + 3)
    )
    assert metrics.work_item_count == 4
    assert metrics.unique_route_count == 1
    assert metrics.unique_visual_input_kind_count == 1
    assert metrics.unique_family_partition_count == 1
    assert metrics.unique_geographic_scope_count == 1
    assert metrics.unique_candidate_matrix_signature_count == 2
    assert metrics.unique_pool_matrix_signature_count == 2
    assert metrics.unique_candidate_pool_signature_count == 2
    assert metrics.cache_locality_run_count == 2
    assert metrics.candidate_matrix_run_count == 2
    assert metrics.pool_matrix_run_count == 2
    assert metrics.candidate_pool_signature_run_count == 2
    assert metrics.candidate_matrix_reuse_opportunity_count == 2
    assert metrics.pool_matrix_reuse_opportunity_count == 2
    assert metrics.adjacent_candidate_matrix_reuse_count == 2
    assert metrics.adjacent_pool_matrix_reuse_count == 2
    assert metrics.as_dict()["cache_locality_run_count"] == 2


def test_empty_order_is_valid_and_fingerprinted() -> None:
    first = sort_geographic_scoring_work(())
    second = sort_geographic_scoring_work(())

    assert first.items == ()
    assert first.ordering_fingerprint == second.ordering_fingerprint
    assert first.metrics.work_item_count == 0
    assert first.metrics.cache_locality_run_count == 0
    assert first.metrics.adjacent_candidate_matrix_reuse_count == 0
    assert not first.metrics.execution_order_changed


def test_work_identity_rejects_drift_and_invalid_contract_fields() -> None:
    item = _item("valid")
    with pytest.raises(ValueError, match="candidate_pool_signature does not match"):
        replace(item, candidate_pool_signature=_sha("9"))
    with pytest.raises(ValueError, match="work_fingerprint does not match"):
        replace(item, work_fingerprint=_sha("9"))
    with pytest.raises(ValueError, match="unsupported scoring work route"):
        replace(item, route="unknown-route", work_fingerprint=None)
    with pytest.raises(ValueError, match="unsupported scoring work visual_input_kind"):
        replace(item, visual_input_kind="crop", work_fingerprint=None)
    with pytest.raises(ValueError, match="unsupported scoring work geographic_scope"):
        replace(item, geographic_scope="invented", work_fingerprint=None)
    with pytest.raises(ValueError, match="result_order_key must not be empty"):
        replace(item, result_order_key=(), work_fingerprint=None)


def test_order_rejects_duplicate_work_identity_and_wrong_row_types() -> None:
    first = _item("duplicate")
    second = replace(first, payload_fingerprint=_sha("8"), work_fingerprint=None)

    with pytest.raises(ValueError, match="work IDs must be unique"):
        sort_geographic_scoring_work((first, second))
    with pytest.raises(TypeError, match="GeographicScoringWorkItem"):
        sort_geographic_scoring_work((first, object()))  # type: ignore[arg-type]


def _item(
    work_id: str,
    *,
    route: str = "adult_field",
    visual_input_kind: str = "focused_full_frame",
    family_partition: str = "a-family",
    geographic_scope: str = "global",
    candidate_digit: str = "1",
    pool_digit: str = "2",
    result_order_key: tuple[str, ...] = ("01",),
) -> GeographicScoringWorkItem:
    candidate_signature = _sha(candidate_digit)
    pool_signature = _sha(pool_digit)
    return GeographicScoringWorkItem(
        work_id=work_id,
        route=route,
        visual_input_kind=visual_input_kind,
        family_partition=family_partition,
        geographic_scope=geographic_scope,
        candidate_matrix_signature=candidate_signature,
        pool_matrix_signature=pool_signature,
        candidate_pool_signature=candidate_pool_signature(
            candidate_signature,
            (pool_signature,),
        ),
        payload_fingerprint=_sha("a"),
        result_order_key=result_order_key,
    )


def _sha(character: str) -> str:
    return "sha256:" + character * 64
