"""Numeric tests for raw dynamic-pool component scoring."""

from __future__ import annotations

from dataclasses import replace

import pytest

from biominer.bioclip.dynamic_pool_scoring import (
    RawScoringQuery,
    score_family_evidence,
)
from biominer.bioclip.matrix_cache import (
    FamilyPrototypeMatrixCache,
    FamilyPrototypeVector,
)


_MODEL_FINGERPRINT = "sha256:" + "a" * 64
_PROTOTYPE_SET_FINGERPRINT = "sha256:" + "b" * 64


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


def _query() -> RawScoringQuery:
    return RawScoringQuery(
        query_id="flickr-embedding:query",
        query_embedding_fingerprint="sha256:" + "f" * 64,
        route="adult_field",
        visual_input_kind="focused_full_frame",
        model_fingerprint=_MODEL_FINGERPRINT,
        embedding=(1.0, 0.0),
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
