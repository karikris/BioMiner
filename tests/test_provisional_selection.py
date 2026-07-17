from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from biominer.references.provisional_selection import (
    PROVISIONAL_SELECTION_CANDIDATE_SCHEMA,
    PROVISIONAL_SELECTION_DECISION_SCHEMA,
    ProvisionalSelectionPolicy,
    select_independent_provisional_support,
)


def test_selects_one_observation_duplicate_and_observer_before_reuse(
    tmp_path: Path,
) -> None:
    candidates = _frame(
        [
            _candidate(1, taxon="a", observation="o1", observer="p1", quality=0.9),
            _candidate(2, taxon="a", observation="o1", observer="p1", quality=0.8),
            _candidate(3, taxon="a", observation="o2", observer="p2", quality=0.7),
            _candidate(4, taxon="a", observation="o3", observer="p1", quality=0.6),
        ]
    )

    result = select_independent_provisional_support(
        candidates,
        output_dir=tmp_path,
        policy=ProvisionalSelectionPolicy(quota_per_species=3),
    )

    selected = result.selections.to_dicts()
    assert [row["reference_media_id"] for row in selected] == [
        _media_id(1),
        _media_id(3),
        _media_id(4),
    ]
    assert [row["observer_selection_ordinal"] for row in selected] == [1, 1, 2]
    assert [row["observer_reuse_justified"] for row in selected] == [
        False,
        False,
        True,
    ]
    skipped = result.decisions.filter(pl.col("decision") == "skipped").row(
        0, named=True
    )
    assert skipped["decision_reason"] == (
        "observation_already_selected_without_distinct_view"
    )


def test_selects_only_canonical_member_of_each_duplicate_group(
    tmp_path: Path,
) -> None:
    canonical = _candidate(1, taxon="a", duplicate_group="g1")
    mirror = _candidate(
        2,
        taxon="a",
        observation="o2",
        observer="p2",
        duplicate_group="g1",
        canonical_media_id=_media_id(1),
    )

    result = select_independent_provisional_support(
        _frame([mirror, canonical]),
        output_dir=tmp_path,
        policy=ProvisionalSelectionPolicy(quota_per_species=2),
    )

    assert result.selections["reference_media_id"].to_list() == [_media_id(1)]
    skipped = result.decisions.filter(
        pl.col("reference_media_id") == _media_id(2)
    ).row(0, named=True)
    assert skipped["decision_reason"] == "noncanonical_duplicate_member"


def test_additional_observation_image_requires_distinct_documented_view(
    tmp_path: Path,
) -> None:
    rows = [
        _candidate(
            1,
            taxon="a",
            observation="o1",
            observer="p1",
            view="dorsal",
            view_evidence="provider_documented",
        ),
        _candidate(
            2,
            taxon="a",
            observation="o1",
            observer="p1",
            duplicate_group="g2",
            view="lateral",
            view_evidence="embedding_distinct",
            quality=0.8,
        ),
        _candidate(
            3,
            taxon="a",
            observation="o1",
            observer="p1",
            duplicate_group="g3",
            view="dorsal",
            view_evidence="provider_documented",
            quality=0.7,
        ),
    ]

    result = select_independent_provisional_support(
        _frame(rows),
        output_dir=tmp_path,
        policy=ProvisionalSelectionPolicy(quota_per_species=3),
    )

    assert result.selections["reference_media_id"].to_list() == [
        _media_id(1),
        _media_id(2),
    ]
    second = result.selections.row(1, named=True)
    assert second["selection_round"] == "distinct_additional_view"
    assert second["observation_selection_ordinal"] == 2
    assert second["distinct_additional_view_justified"] is True
    third = result.decisions.filter(
        pl.col("reference_media_id") == _media_id(3)
    ).row(0, named=True)
    assert third["decision_reason"] == "documented_view_already_selected"


def test_balances_species_round_robin_and_reports_fixed_quota_shortfalls(
    tmp_path: Path,
) -> None:
    rows = [
        _candidate(index, taxon="a", observation=f"a{index}", observer=f"pa{index}")
        for index in range(1, 5)
    ] + [
        _candidate(
            index,
            taxon="b",
            observation=f"b{index}",
            observer=f"pb{index}",
        )
        for index in range(5, 7)
    ]

    result = select_independent_provisional_support(
        _frame(rows),
        output_dir=tmp_path,
        policy=ProvisionalSelectionPolicy(quota_per_species=3),
    )

    assert result.selections["accepted_taxon_key"].to_list() == [
        "a",
        "b",
        "a",
        "b",
        "a",
    ]
    assert result.selected_per_species == (("a", 3), ("b", 2))
    assert result.shortfall_per_species == (("a", 0), ("b", 1))
    extra = result.decisions.filter(
        (pl.col("accepted_taxon_key") == "a")
        & (pl.col("decision") == "skipped")
    ).row(0, named=True)
    assert extra["decision_reason"] == "species_quota_reached"


def test_persists_complete_deterministic_selected_and_skipped_ledger(
    tmp_path: Path,
) -> None:
    candidates = _frame(
        [
            _candidate(1, taxon="a"),
            _candidate(
                2,
                taxon="a",
                observation="o2",
                observer="p2",
                admission="review_required",
            ),
            _candidate(
                3,
                taxon="a",
                observation="o3",
                observer=None,
            ),
        ]
    )
    before = candidates.clone()
    policy = ProvisionalSelectionPolicy(quota_per_species=2, selection_seed=42)

    first = select_independent_provisional_support(
        candidates, output_dir=tmp_path / "first", policy=policy
    )
    second = select_independent_provisional_support(
        candidates, output_dir=tmp_path / "second", policy=policy
    )

    assert candidates.equals(before)
    assert first.decisions.schema == PROVISIONAL_SELECTION_DECISION_SCHEMA
    assert first.decisions.height == candidates.height
    assert first.decisions.equals(second.decisions)
    assert first.decisions_path.exists()
    assert first.selections_path.exists()
    reasons = dict(
        first.decisions.select("reference_media_id", "decision_reason").iter_rows()
    )
    assert reasons[_media_id(2)] == "admission_review_required"
    assert reasons[_media_id(3)] == "observer_identity_missing"
    assert first.policy_fingerprint == policy.fingerprint


def test_policy_rejects_unbounded_or_inconsistent_independence() -> None:
    policy = ProvisionalSelectionPolicy()

    with pytest.raises(ValueError, match="one image per observation"):
        replace(policy, maximum_images_per_observation=2)
    with pytest.raises(ValueError, match="disabled additional views"):
        replace(policy, allow_distinct_additional_views=False)
    with pytest.raises(ValueError, match="one image per observer"):
        replace(policy, maximum_images_per_observer_before_reuse=2)
    with pytest.raises(ValueError, match="bounded second view"):
        replace(policy, maximum_distinct_views_per_observation=3)


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=PROVISIONAL_SELECTION_CANDIDATE_SCHEMA)


def _candidate(
    index: int,
    *,
    taxon: str,
    observation: str = "o1",
    observer: str | None = "p1",
    duplicate_group: str | None = None,
    canonical_media_id: str | None = None,
    view: str = "unknown",
    view_evidence: str = "none",
    quality: float = 0.9,
    admission: str = "admitted",
) -> dict[str, object]:
    media_id = _media_id(index)
    return {
        "reference_media_id": media_id,
        "accepted_taxon_key": taxon,
        "reference_observation_id": observation,
        "observer_id": observer,
        "duplicate_group_id": duplicate_group or f"duplicate-group:{index}",
        "canonical_reference_media_id": canonical_media_id or media_id,
        "route": "adult_field",
        "documented_view": view,
        "distinct_view_evidence": view_evidence,
        "quality_score": quality,
        "admission_decision": admission,
    }


def _media_id(index: int) -> str:
    return "reference-media:" + f"{index:x}" * 64
