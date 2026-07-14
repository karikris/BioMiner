from __future__ import annotations

from math import isclose, sqrt
from pathlib import Path

import polars as pl
import pytest

from biominer.bioclip.reference_prototypes import (
    PROTOTYPE_METHOD_NORMALIZED_MEAN,
    PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED,
    build_reference_prototypes,
)
from biominer.bioclip.reference_scoring import (
    ReferenceCandidate,
    ReferenceEvidenceIndex,
    ReferenceQuery,
)
from biominer.vision.full_frame_attention import RAW_FULL_IMAGE_KIND
from test_reference_prototypes import _embedding_artifact, _spec, _unit


TARGET = "gbif:1938069"
COMPETITOR = "gbif:1938070"
MISSING = "gbif:9999999"


def test_scores_fixed_top_k_centroid_and_local_global_prototypes(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec("a1", "oa1", TARGET, "Papilio demoleus", "cluster-a", (1, 0, 0)),
            _spec(
                "a2",
                "oa2",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (0.8, 0.6, 0),
            ),
            _spec(
                "a3",
                "oa3",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (0.6, 0.8, 0),
            ),
            _spec("a4", "oa4", TARGET, "Papilio demoleus", "cluster-b", (0, 1, 0)),
            _spec("a5", "oa5", TARGET, "Papilio demoleus", "cluster-b", (-1, 0, 0)),
        ),
    )
    prototypes = build_reference_prototypes(embeddings)
    index = ReferenceEvidenceIndex(
        embeddings,
        prototypes,
        balanced_reference_count=5,
        prototype_method=PROTOTYPE_METHOD_NORMALIZED_MEAN,
    )

    score = index.score(
        _query(embeddings, vector=(1, 0, 0), geo_cluster_id="cluster-a"),
        (ReferenceCandidate(TARGET, "Papilio demoleus"),),
    )[0]

    assert score.support_count == 5
    assert score.local_support_count == 3
    assert score.selected_support_count == 5
    assert score.selected_local_support_count == 3
    assert score.nearest_reference_observation_id == "oa1"
    assert score.nearest_support_similarity == pytest.approx(1.0)
    assert score.mean_top_three_similarity == pytest.approx(0.8)
    assert score.mean_top_five_similarity == pytest.approx(0.28)
    assert score.centroid_similarity == pytest.approx(1.4 / sqrt(1.4**2 + 2.4**2))
    assert score.local_cluster_prototype_similarity == pytest.approx(
        2.4 / sqrt(2.4**2 + 1.4**2)
    )
    assert score.global_prototype_similarity == pytest.approx(score.centroid_similarity)
    assert score.distance_to_nearest_independent_observation == pytest.approx(0.0)
    assert score.insufficient_support is False
    assert score.insufficient_support_reasons == ()


def test_caps_every_candidate_at_same_fixed_reference_pool(tmp_path: Path) -> None:
    specs = [
        _spec(
            f"target-{index}",
            f"target-observation-{index}",
            TARGET,
            "Papilio demoleus",
            "cluster-a",
            (1, index + 1, 0),
        )
        for index in range(5)
    ]
    specs.extend(
        _spec(
            f"competitor-{index}",
            f"competitor-observation-{index}",
            COMPETITOR,
            "Papilio polytes",
            "cluster-a",
            (index + 1, 1, 0),
        )
        for index in range(20)
    )
    embeddings = _embedding_artifact(tmp_path, tuple(specs))
    index = ReferenceEvidenceIndex(
        embeddings,
        build_reference_prototypes(embeddings),
        balanced_reference_count=5,
    )

    target, competitor = index.score(
        _query(embeddings),
        (
            ReferenceCandidate(TARGET, "Papilio demoleus"),
            ReferenceCandidate(COMPETITOR, "Papilio polytes"),
        ),
    )

    assert target.support_count == 5
    assert competitor.support_count == 20
    assert target.selected_support_count == competitor.selected_support_count == 5
    assert len(target.selected_reference_observation_ids) == 5
    assert len(competitor.selected_reference_observation_ids) == 5
    assert competitor.insufficient_support is False


def test_balanced_selection_identity_does_not_depend_on_embedding_values(
    tmp_path: Path,
) -> None:
    def artifact(root: Path, *, reverse: bool):
        return _embedding_artifact(
            root,
            tuple(
                _spec(
                    f"media-{index}",
                    f"observation-{index}",
                    TARGET,
                    "Papilio demoleus",
                    "cluster-a",
                    (1, 20 - index if reverse else index + 1, 0),
                )
                for index in range(10)
            ),
        )

    first = artifact(tmp_path / "first", reverse=False)
    second = artifact(tmp_path / "second", reverse=True)
    first_index = ReferenceEvidenceIndex(first, build_reference_prototypes(first))
    second_index = ReferenceEvidenceIndex(second, build_reference_prototypes(second))

    first_score = first_index.score(
        _query(first),
        (ReferenceCandidate(TARGET, "Papilio demoleus"),),
    )[0]
    second_score = second_index.score(
        _query(second),
        (ReferenceCandidate(TARGET, "Papilio demoleus"),),
    )[0]

    assert (
        first_score.selected_reference_observation_ids
        == second_score.selected_reference_observation_ids
    )
    assert (
        first_score.nearest_support_similarity
        != second_score.nearest_support_similarity
    )


def test_local_observations_are_selected_before_global_fill(tmp_path: Path) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        tuple(
            _spec(
                f"media-{index}",
                f"observation-{index}",
                TARGET,
                "Papilio demoleus",
                "cluster-a" if index < 4 else "cluster-b",
                (1, index + 1, 0),
            )
            for index in range(10)
        ),
    )
    index = ReferenceEvidenceIndex(embeddings, build_reference_prototypes(embeddings))

    score = index.score(
        _query(embeddings, geo_cluster_id="cluster-a"),
        (ReferenceCandidate(TARGET, "Papilio demoleus"),),
    )[0]

    assert score.local_support_count == 4
    assert score.selected_local_support_count == 4
    assert len(score.selected_reference_observation_ids) == 5
    assert {
        "observation-0",
        "observation-1",
        "observation-2",
        "observation-3",
    }.issubset(score.selected_reference_observation_ids)


def test_reports_insufficient_support_without_averaging_short_top_k(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec("a1", "oa1", TARGET, "Papilio demoleus", "cluster-a", (1, 0, 0)),
            _spec("a2", "oa2", TARGET, "Papilio demoleus", "cluster-a", (0, 1, 0)),
        ),
    )
    index = ReferenceEvidenceIndex(embeddings, build_reference_prototypes(embeddings))

    score = index.score(
        _query(embeddings),
        (ReferenceCandidate(TARGET, "Papilio demoleus"),),
    )[0]

    assert score.support_count == score.selected_support_count == 2
    assert score.nearest_support_similarity == pytest.approx(1.0)
    assert score.centroid_similarity == pytest.approx(1 / sqrt(2))
    assert score.mean_top_three_similarity is None
    assert score.mean_top_five_similarity is None
    assert score.insufficient_support is True
    assert score.insufficient_support_reasons == (
        "fewer_than_balanced_reference_count",
        "fewer_than_three_independent_observations",
        "fewer_than_five_independent_observations",
    )


def test_candidate_without_support_returns_explicit_null_evidence(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (_spec("a1", "oa1", TARGET, "Papilio demoleus", "cluster-a", (1, 0, 0)),),
    )
    index = ReferenceEvidenceIndex(embeddings, build_reference_prototypes(embeddings))

    score = index.score(
        _query(embeddings),
        (ReferenceCandidate(MISSING, "Papilio missing"),),
    )[0]

    assert score.support_count == score.local_support_count == 0
    assert score.selected_support_count == 0
    assert score.nearest_support_similarity is None
    assert score.mean_top_three_similarity is None
    assert score.mean_top_five_similarity is None
    assert score.centroid_similarity is None
    assert score.local_cluster_prototype_similarity is None
    assert score.global_prototype_similarity is None
    assert score.distance_to_nearest_independent_observation is None
    assert score.insufficient_support is True
    assert score.insufficient_support_reasons[0] == "no_route_support"


def test_multiple_media_are_collapsed_to_one_independent_observation(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec(
                "a1",
                "shared-observation",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (1, 0, 0),
            ),
            _spec(
                "a2",
                "shared-observation",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (0, 1, 0),
            ),
        ),
    )
    index = ReferenceEvidenceIndex(embeddings, build_reference_prototypes(embeddings))

    score = index.score(
        _query(embeddings),
        (ReferenceCandidate(TARGET, "Papilio demoleus"),),
    )[0]

    assert score.support_count == 1
    assert score.selected_support_count == 1
    assert score.nearest_support_similarity == pytest.approx(1 / sqrt(2))
    assert score.selected_reference_observation_ids == ("shared-observation",)


def test_query_observation_is_excluded_from_its_own_reference_pool(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (_spec("a1", "oa1", TARGET, "Papilio demoleus", "cluster-a", (1, 0, 0)),),
    )
    index = ReferenceEvidenceIndex(embeddings, build_reference_prototypes(embeddings))
    query = _query(
        embeddings,
        excluded_reference_observation_ids=("oa1",),
    )

    score = index.score(
        query,
        (ReferenceCandidate(TARGET, "Papilio demoleus"),),
    )[0]

    assert score.support_count == 0
    assert score.nearest_support_similarity is None
    assert "no_route_support" in score.insufficient_support_reasons


def test_no_geo_uses_global_prototype_without_fabricating_local_evidence(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        tuple(
            _spec(
                f"media-{index}",
                f"observation-{index}",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (1, index + 1, 0),
            )
            for index in range(5)
        ),
    )
    index = ReferenceEvidenceIndex(embeddings, build_reference_prototypes(embeddings))

    score = index.score(
        _query(embeddings, geo_cluster_id="no_geo"),
        (ReferenceCandidate(TARGET, "Papilio demoleus"),),
    )[0]

    assert score.local_support_count == 0
    assert score.selected_local_support_count == 0
    assert score.local_cluster_prototype_similarity is None
    assert score.global_prototype_similarity is not None


def test_route_separation_prevents_adult_queries_using_larval_support(
    tmp_path: Path,
) -> None:
    adult = tuple(
        _spec(
            f"adult-{index}",
            f"adult-observation-{index}",
            TARGET,
            "Papilio demoleus",
            "cluster-a",
            (1, index + 1, 0),
        )
        for index in range(5)
    )
    larval = tuple(
        _spec(
            f"larva-{index}",
            f"larva-observation-{index}",
            TARGET,
            "Papilio demoleus",
            "cluster-a",
            (-1, index + 1, 0),
            life_stage="larva",
            route="larval",
        )
        for index in range(5)
    )
    embeddings = _embedding_artifact(tmp_path, (*adult, *larval))
    index = ReferenceEvidenceIndex(embeddings, build_reference_prototypes(embeddings))

    adult_score = index.score(
        _query(embeddings, route="adult_field"),
        (ReferenceCandidate(TARGET, "Papilio demoleus"),),
    )[0]
    larval_score = index.score(
        _query(embeddings, route="larval"),
        (ReferenceCandidate(TARGET, "Papilio demoleus"),),
    )[0]

    assert adult_score.support_count == larval_score.support_count == 5
    assert adult_score.nearest_support_similarity > 0
    assert larval_score.nearest_support_similarity < 0
    assert all(
        observation_id.startswith("adult-")
        for observation_id in adult_score.selected_reference_observation_ids
    )
    assert all(
        observation_id.startswith("larva-")
        for observation_id in larval_score.selected_reference_observation_ids
    )


def test_negative_cosines_remain_similarities_and_distance_reaches_two(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        tuple(
            _spec(
                f"media-{index}",
                f"observation-{index}",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (-1, 0, 0),
            )
            for index in range(5)
        ),
    )
    index = ReferenceEvidenceIndex(embeddings, build_reference_prototypes(embeddings))

    score = index.score(
        _query(embeddings),
        (ReferenceCandidate(TARGET, "Papilio demoleus"),),
    )[0]

    assert score.nearest_support_similarity == pytest.approx(-1.0)
    assert score.mean_top_three_similarity == pytest.approx(-1.0)
    assert score.mean_top_five_similarity == pytest.approx(-1.0)
    assert score.centroid_similarity == pytest.approx(-1.0)
    assert score.global_prototype_similarity == pytest.approx(-1.0)
    assert score.distance_to_nearest_independent_observation == pytest.approx(2.0)


def test_mean_centered_scoring_requires_matching_seeded_prototypes(
    tmp_path: Path,
) -> None:
    specs = tuple(
        _spec(
            f"target-{index}",
            f"target-observation-{index}",
            TARGET,
            "Papilio demoleus",
            "cluster-a",
            (1, index + 1, 0),
        )
        for index in range(5)
    ) + tuple(
        _spec(
            f"competitor-{index}",
            f"competitor-observation-{index}",
            COMPETITOR,
            "Papilio polytes",
            "cluster-a",
            (-1, index + 1, 0),
        )
        for index in range(5)
    )
    embeddings = _embedding_artifact(tmp_path, specs)
    prototypes = build_reference_prototypes(embeddings, balanced_sampling_seed=17)
    index = ReferenceEvidenceIndex(
        embeddings,
        prototypes,
        balanced_sampling_seed=17,
        prototype_method=PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED,
    )

    scores = index.score(
        _query(embeddings),
        (
            ReferenceCandidate(TARGET, "Papilio demoleus"),
            ReferenceCandidate(COMPETITOR, "Papilio polytes"),
        ),
    )

    assert all(score.global_prototype_similarity is not None for score in scores)
    assert all(score.centering_fingerprint is not None for score in scores)
    assert all(
        isclose(score.query_embedding_norm, 1.0, abs_tol=1e-5) for score in scores
    )

    with pytest.raises(ValueError, match="balanced sampling seed"):
        ReferenceEvidenceIndex(
            embeddings,
            prototypes,
            balanced_sampling_seed=18,
            prototype_method=PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED,
        )


def test_rejects_query_model_and_reference_prototype_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    specs = tuple(
        _spec(
            f"media-{index}",
            f"observation-{index}",
            TARGET,
            "Papilio demoleus",
            "cluster-a",
            (1, index + 1, 0),
        )
        for index in range(5)
    )
    first = _embedding_artifact(tmp_path / "first", specs)
    first_prototypes = build_reference_prototypes(first)
    index = ReferenceEvidenceIndex(first, first_prototypes)

    with pytest.raises(ValueError, match="query model fingerprint"):
        index.score(
            ReferenceQuery(
                query_id="query-1",
                embedding=(1.0, 0.0, 0.0),
                route="adult_field",
                visual_input_kind=RAW_FULL_IMAGE_KIND,
                geo_cluster_id="cluster-a",
                model_fingerprint="sha256:" + "0" * 64,
            ),
            (ReferenceCandidate(TARGET, "Papilio demoleus"),),
        )

    changed_specs = tuple(
        _spec(
            spec.media_id,
            spec.observation_id,
            spec.taxon_key,
            spec.scientific_name,
            spec.geo_cluster_id,
            (0, index + 1, 1),
        )
        for index, spec in enumerate(specs)
    )
    changed = _embedding_artifact(tmp_path / "changed", changed_specs)
    with pytest.raises(ValueError, match="reference_embedding_fingerprint"):
        ReferenceEvidenceIndex(changed, first_prototypes)


def test_rejects_structurally_valid_but_incomplete_raw_prototype_coverage(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        tuple(
            _spec(
                f"media-{index}",
                f"observation-{index}",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (1, index + 1, 0),
            )
            for index in range(5)
        ),
    )
    prototypes = build_reference_prototypes(embeddings).filter(
        pl.col("cluster_scope_type") == "global"
    )

    with pytest.raises(ValueError, match="prototype coverage"):
        ReferenceEvidenceIndex(embeddings, prototypes)


def _query(
    embeddings,
    *,
    vector: tuple[float, float, float] = (1, 0, 0),
    route: str = "adult_field",
    geo_cluster_id: str = "cluster-a",
    excluded_reference_observation_ids: tuple[str, ...] = (),
) -> ReferenceQuery:
    normalized = _unit(vector)
    return ReferenceQuery(
        query_id="query-1",
        embedding=normalized,
        route=route,
        visual_input_kind=RAW_FULL_IMAGE_KIND,
        geo_cluster_id=geo_cluster_id,
        model_fingerprint=str(embeddings["model_fingerprint"][0]),
        excluded_reference_observation_ids=excluded_reference_observation_ids,
    )
