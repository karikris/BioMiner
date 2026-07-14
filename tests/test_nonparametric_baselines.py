from __future__ import annotations

from math import fsum
from pathlib import Path

import pytest

from biominer.bioclip.reference_prototypes import (
    PROTOTYPE_KIND_AGGREGATE,
    PROTOTYPE_METHOD_NORMALIZED_MEAN,
    PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED,
    PROTOTYPE_SCOPE_GLOBAL,
    MultiPrototypeConfig,
    build_multi_reference_prototypes,
    build_reference_centering_contexts,
    build_reference_prototypes,
    mean_center_query_embedding,
)
from biominer.bioclip.reference_scoring import ReferenceCandidate, ReferenceQuery
from biominer.ml.nonparametric import (
    MEAN_CENTERED_NEAREST_CENTROID_METHOD,
    MULTI_PROTOTYPE_NEAREST_CLASS_METHOD,
    NEAREST_CENTROID_METHOD,
    SCORE_KIND_COSINE_SIMILARITY,
    SCORE_KIND_NEIGHBOR_VOTE_FRACTION,
    TOP_K_NEAREST_NEIGHBORS_METHOD,
    NonparametricBaselineIndex,
)
from test_reference_prototypes import _embedding_artifact, _spec, _unit


TARGET = "gbif:1938069"
COMPETITOR = "gbif:1938070"
MISSING = "gbif:9999999"
TARGET_NAME = "Papilio demoleus"
COMPETITOR_NAME = "Papilio polytes"


def test_nearest_centroid_scores_persisted_global_species_prototypes(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec(
                "target-a", "target-observation-a", TARGET, TARGET_NAME, "a", (1, 0, 0)
            ),
            _spec(
                "target-b", "target-observation-b", TARGET, TARGET_NAME, "b", (1, 1, 0)
            ),
            _spec(
                "competitor-a",
                "competitor-observation-a",
                COMPETITOR,
                COMPETITOR_NAME,
                "a",
                (0, 1, 0),
            ),
            _spec(
                "competitor-b",
                "competitor-observation-b",
                COMPETITOR,
                COMPETITOR_NAME,
                "b",
                (0, 1, 1),
            ),
        ),
    )
    prototypes = build_reference_prototypes(embeddings)
    index = NonparametricBaselineIndex(embeddings, prototypes)
    query = _query(embeddings, (1, 0, 0))

    prediction = index.predict_nearest_centroid(
        query,
        _candidates(reverse=True),
    )

    expected_row = prototypes.filter(
        (prototypes["accepted_taxon_key"] == TARGET)
        & (prototypes["cluster_scope_type"] == PROTOTYPE_SCOPE_GLOBAL)
        & (prototypes["prototype_kind"] == PROTOTYPE_KIND_AGGREGATE)
        & (prototypes["prototype_method"] == PROTOTYPE_METHOD_NORMALIZED_MEAN)
    ).row(0, named=True)
    target_score = _score_for(prediction, TARGET)
    assert prediction.method == NEAREST_CENTROID_METHOD
    assert prediction.predicted_taxon_key == TARGET
    assert prediction.abstained is False
    assert target_score.score_kind == SCORE_KIND_COSINE_SIMILARITY
    assert target_score.raw_score == pytest.approx(
        _dot(query.embedding, expected_row["embedding"])
    )
    assert target_score.winning_evidence_id == expected_row["prototype_id"]
    assert prediction.reference_embedding_fingerprint == str(
        expected_row["reference_embedding_fingerprint"]
    )
    assert prediction.reference_prototype_fingerprint.startswith("sha256:")
    assert not hasattr(index, "fit")


def test_mean_centered_nearest_centroid_reuses_attested_simpleshot_context(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec("target-a", "target-a", TARGET, TARGET_NAME, "a", (1, 0, 1)),
            _spec("target-b", "target-b", TARGET, TARGET_NAME, "b", (1, 1, 0)),
            _spec(
                "competitor-a",
                "competitor-a",
                COMPETITOR,
                COMPETITOR_NAME,
                "a",
                (0, 1, 1),
            ),
            _spec(
                "competitor-b",
                "competitor-b",
                COMPETITOR,
                COMPETITOR_NAME,
                "b",
                (0, 1, 0),
            ),
        ),
    )
    prototypes = build_reference_prototypes(
        embeddings,
        balanced_sampling_seed=17,
    )
    index = NonparametricBaselineIndex(
        embeddings,
        prototypes,
        balanced_sampling_seed=17,
    )
    query = _query(embeddings, (1, 0, 0))

    prediction = index.predict_mean_centered_nearest_centroid(
        query,
        _candidates(),
    )

    context = next(
        item
        for item in build_reference_centering_contexts(
            embeddings,
            balanced_sampling_seed=17,
        )
        if item.route == query.route
        and item.visual_input_kind == query.visual_input_kind
    )
    centered_query = mean_center_query_embedding(query.embedding, context)
    row = prototypes.filter(
        (prototypes["accepted_taxon_key"] == TARGET)
        & (prototypes["cluster_scope_type"] == PROTOTYPE_SCOPE_GLOBAL)
        & (prototypes["prototype_kind"] == PROTOTYPE_KIND_AGGREGATE)
        & (prototypes["prototype_method"] == PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED)
    ).row(0, named=True)
    assert prediction.method == MEAN_CENTERED_NEAREST_CENTROID_METHOD
    assert prediction.centering_fingerprint == context.centering_fingerprint
    assert prediction.centering_fingerprint == row["centering_fingerprint"]
    assert _score_for(prediction, TARGET).raw_score == pytest.approx(
        _dot(centered_query, row["embedding"])
    )


def test_top_k_neighbours_use_exact_unweighted_votes_and_stable_ties(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec("target-a", "target-a", TARGET, TARGET_NAME, "a", (-1, 0.2, 0)),
            _spec("target-b", "target-b", TARGET, TARGET_NAME, "a", (-1, -0.2, 0)),
            _spec(
                "competitor-a",
                "competitor-a",
                COMPETITOR,
                COMPETITOR_NAME,
                "a",
                (1, 0, 0),
            ),
        ),
    )
    index = NonparametricBaselineIndex(
        embeddings,
        build_reference_prototypes(embeddings),
    )

    first = index.predict_top_k_nearest_neighbors(
        _query(embeddings, (1, 0, 0)),
        _candidates(reverse=True),
        k=3,
    )
    second = index.predict_top_k_nearest_neighbors(
        _query(embeddings, (1, 0, 0)),
        _candidates(),
        k=3,
    )

    assert first.method == TOP_K_NEAREST_NEIGHBORS_METHOD
    assert first.predicted_taxon_key == TARGET
    assert first.predicted_taxon_key == second.predicted_taxon_key
    assert first.neighbors == second.neighbors
    assert [item.reference_observation_id for item in first.neighbors] == [
        "competitor-a",
        "target-a",
        "target-b",
    ]
    target_score = _score_for(first, TARGET)
    competitor_score = _score_for(first, COMPETITOR)
    assert target_score.score_kind == SCORE_KIND_NEIGHBOR_VOTE_FRACTION
    assert target_score.vote_count == 2
    assert target_score.raw_score == pytest.approx(2 / 3)
    assert competitor_score.vote_count == 1
    assert competitor_score.raw_score == pytest.approx(1 / 3)
    assert target_score.vote_similarity_sum < competitor_score.vote_similarity_sum
    assert first.raw_margin == pytest.approx(1 / 3)


def test_top_k_neighbours_abstain_instead_of_shortening_k_or_deleting_candidate(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec("target-a", "target-a", TARGET, TARGET_NAME, "a", (1, 0, 0)),
            _spec(
                "competitor-a",
                "competitor-a",
                COMPETITOR,
                COMPETITOR_NAME,
                "a",
                (0, 1, 0),
            ),
        ),
    )
    index = NonparametricBaselineIndex(
        embeddings,
        build_reference_prototypes(embeddings),
    )

    prediction = index.predict_top_k_nearest_neighbors(
        _query(embeddings),
        (*_candidates(), ReferenceCandidate(MISSING, "Missing species")),
        k=5,
    )

    assert prediction.abstained is True
    assert prediction.predicted_taxon_key is None
    assert prediction.neighbors == ()
    assert prediction.abstention_reasons == (
        "insufficient_support_for_fixed_k:2/5",
        f"missing_candidate_support:{MISSING}",
    )
    missing = _score_for(prediction, MISSING)
    assert missing.raw_score is None
    assert missing.unavailable_reason == "no_matching_support_observation"


def test_multi_prototype_nearest_class_preserves_within_species_visual_modes(
    tmp_path: Path,
) -> None:
    specs = [
        _spec("target-x-1", "target-x-1", TARGET, TARGET_NAME, "a", (1, 0.02, 0)),
        _spec("target-x-2", "target-x-2", TARGET, TARGET_NAME, "a", (1, -0.02, 0)),
        _spec("target-y-1", "target-y-1", TARGET, TARGET_NAME, "a", (0.02, 1, 0)),
        _spec("target-y-2", "target-y-2", TARGET, TARGET_NAME, "a", (-0.02, 1, 0)),
    ]
    specs.extend(
        _spec(
            f"competitor-{index}",
            f"competitor-{index}",
            COMPETITOR,
            COMPETITOR_NAME,
            "a",
            (0.8, 0.6 + (index - 1.5) * 0.01, 0),
        )
        for index in range(4)
    )
    embeddings = _embedding_artifact(tmp_path, tuple(specs))
    prototypes = build_multi_reference_prototypes(
        embeddings,
        config=MultiPrototypeConfig(
            minimum_metadata_observation_count=2,
            minimum_clustering_observation_count=4,
            minimum_embedding_cluster_size=2,
            maximum_embedding_cluster_count=2,
            maximum_clustering_observation_count=16,
            cosine_distance_threshold=0.2,
        ),
    )
    index = NonparametricBaselineIndex(embeddings, prototypes)
    query = _query(embeddings, (1, 0, 0))

    centroid = index.predict_nearest_centroid(query, _candidates())
    multi = index.predict_multi_prototype_nearest_class(query, _candidates())

    assert centroid.predicted_taxon_key == COMPETITOR
    assert multi.method == MULTI_PROTOTYPE_NEAREST_CLASS_METHOD
    assert multi.predicted_taxon_key == TARGET
    target_score = _score_for(multi, TARGET)
    assert target_score.prototype_count == 2
    assert target_score.raw_score is not None
    assert target_score.raw_score > 0.99
    assert target_score.winning_evidence_id is not None
    assert target_score.winning_evidence_id.startswith("reference-prototype:")

    excluded = index.predict_multi_prototype_nearest_class(
        _query(
            embeddings,
            (1, 0, 0),
            excluded_reference_observation_ids=("target-x-1",),
        ),
        _candidates(),
    )
    assert _score_for(excluded, TARGET).prototype_count == 1


def test_baselines_never_cross_route_or_model_contracts(tmp_path: Path) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec("target-adult", "target-adult", TARGET, TARGET_NAME, "a", (-1, 0, 0)),
            _spec(
                "target-larva",
                "target-larva",
                TARGET,
                TARGET_NAME,
                "a",
                (1, 0, 0),
                life_stage="larva",
                route="larval",
            ),
            _spec(
                "competitor-adult",
                "competitor-adult",
                COMPETITOR,
                COMPETITOR_NAME,
                "a",
                (1, 0, 0),
            ),
        ),
    )
    index = NonparametricBaselineIndex(
        embeddings,
        build_reference_prototypes(embeddings),
    )

    prediction = index.predict_nearest_centroid(
        _query(embeddings, (1, 0, 0), route="adult_field"),
        _candidates(),
    )
    assert prediction.predicted_taxon_key == COMPETITOR
    assert _score_for(prediction, TARGET).support_count == 1

    bad_query = ReferenceQuery(
        query_id="query:wrong-model",
        embedding=_unit((1, 0, 0)),
        route="adult_field",
        visual_input_kind="raw_full_image",
        geo_cluster_id="a",
        model_fingerprint="sha256:" + "0" * 64,
    )
    with pytest.raises(ValueError, match="model fingerprint"):
        index.predict_nearest_centroid(bad_query, _candidates())


def test_observation_and_duplicate_exclusions_remove_knn_leakage(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec(
                "target-a",
                "target-a",
                TARGET,
                TARGET_NAME,
                "a",
                (1, 0, 0),
                duplicate_group_id="duplicate:shared",
            ),
            _spec(
                "target-b",
                "target-b",
                TARGET,
                TARGET_NAME,
                "a",
                (0.9, 0.1, 0),
                duplicate_group_id="duplicate:shared",
            ),
            _spec(
                "competitor-a",
                "competitor-a",
                COMPETITOR,
                COMPETITOR_NAME,
                "a",
                (0, 1, 0),
            ),
            _spec(
                "competitor-b",
                "competitor-b",
                COMPETITOR,
                COMPETITOR_NAME,
                "a",
                (0, 0.9, 0.1),
            ),
        ),
    )
    index = NonparametricBaselineIndex(
        embeddings,
        build_reference_prototypes(embeddings),
    )
    query = _query(
        embeddings,
        excluded_reference_observation_ids=("competitor-a",),
        excluded_duplicate_group_ids=("duplicate:shared",),
    )

    prediction = index.predict_top_k_nearest_neighbors(
        query,
        _candidates(),
        k=1,
    )

    assert [item.reference_observation_id for item in prediction.neighbors] == [
        "competitor-b"
    ]
    assert _score_for(prediction, TARGET).unavailable_reason == (
        "no_matching_support_observation_after_exclusions"
    )
    assert prediction.abstention_reasons == (f"missing_candidate_support:{TARGET}",)


def test_rejects_centering_seed_and_candidate_name_mismatches(tmp_path: Path) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec("target-a", "target-a", TARGET, TARGET_NAME, "a", (1, 0, 0)),
            _spec(
                "competitor-a",
                "competitor-a",
                COMPETITOR,
                COMPETITOR_NAME,
                "a",
                (0, 1, 0),
            ),
        ),
    )
    prototypes = build_reference_prototypes(
        embeddings,
        balanced_sampling_seed=17,
    )

    with pytest.raises(ValueError, match="balanced sampling seed"):
        NonparametricBaselineIndex(embeddings, prototypes)

    index = NonparametricBaselineIndex(
        embeddings,
        prototypes,
        balanced_sampling_seed=17,
    )
    with pytest.raises(ValueError, match="scientific name"):
        index.predict_nearest_centroid(
            _query(embeddings),
            (
                ReferenceCandidate(TARGET, "Papilio incorrectus"),
                ReferenceCandidate(COMPETITOR, COMPETITOR_NAME),
            ),
        )


def test_exact_score_ties_use_taxon_identity_not_candidate_order(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec("target-a", "target-a", TARGET, TARGET_NAME, "a", (1, 0, 0)),
            _spec(
                "competitor-a",
                "competitor-a",
                COMPETITOR,
                COMPETITOR_NAME,
                "a",
                (1, 0, 0),
            ),
        ),
    )
    index = NonparametricBaselineIndex(
        embeddings,
        build_reference_prototypes(embeddings),
    )

    forward = index.predict_nearest_centroid(_query(embeddings), _candidates())
    reverse = index.predict_nearest_centroid(
        _query(embeddings),
        _candidates(reverse=True),
    )

    assert forward.predicted_taxon_key == TARGET
    assert reverse.predicted_taxon_key == TARGET
    assert forward.class_scores == reverse.class_scores
    assert forward.raw_margin == pytest.approx(0.0)


def _query(
    embeddings,
    vector: tuple[float, float, float] = (1, 0, 0),
    *,
    route: str = "adult_field",
    excluded_reference_observation_ids: tuple[str, ...] = (),
    excluded_duplicate_group_ids: tuple[str, ...] = (),
) -> ReferenceQuery:
    return ReferenceQuery(
        query_id="query:fixture",
        embedding=_unit(vector),
        route=route,
        visual_input_kind="raw_full_image",
        geo_cluster_id="a",
        model_fingerprint=str(embeddings["model_fingerprint"][0]),
        excluded_reference_observation_ids=excluded_reference_observation_ids,
        excluded_duplicate_group_ids=excluded_duplicate_group_ids,
    )


def _candidates(*, reverse: bool = False) -> tuple[ReferenceCandidate, ...]:
    result = (
        ReferenceCandidate(TARGET, TARGET_NAME),
        ReferenceCandidate(COMPETITOR, COMPETITOR_NAME),
    )
    return tuple(reversed(result)) if reverse else result


def _score_for(prediction, taxon_key: str):
    return next(
        score
        for score in prediction.class_scores
        if score.accepted_taxon_key == taxon_key
    )


def _dot(left, right) -> float:
    return fsum(a * b for a, b in zip(left, right, strict=True))
