"""Auditable nonparametric baselines over frozen BioCLIP reference artifacts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from biominer.bioclip.reference_embeddings import (
    load_reference_embeddings,
    reference_embeddings_artifact_fingerprint,
    validate_reference_embeddings,
)
from biominer.bioclip.reference_prototypes import (
    DEFAULT_BALANCED_SAMPLING_SEED,
    PROTOTYPE_GLOBAL_GEO_CLUSTER_ID,
    PROTOTYPE_KIND_AGGREGATE,
    PROTOTYPE_KIND_EMBEDDING_CLUSTER,
    PROTOTYPE_KIND_METADATA,
    PROTOTYPE_METHOD_NORMALIZED_MEAN,
    PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED,
    PROTOTYPE_SCOPE_GLOBAL,
    REFERENCE_PROTOTYPE_METHODS,
    ReferenceCenteringContext,
    ReferenceObservationEmbedding,
    aggregate_reference_observation_embeddings,
    build_reference_centering_contexts,
    load_reference_prototypes,
    mean_center_query_embedding,
    reference_prototypes_artifact_fingerprint,
    validate_reference_prototypes,
)
from biominer.bioclip.reference_scoring import ReferenceCandidate, ReferenceQuery


NONPARAMETRIC_SCORING_VERSION = "few-shot-nonparametric-v1.0.0"

NEAREST_CENTROID_METHOD = "nearest_centroid"
MEAN_CENTERED_NEAREST_CENTROID_METHOD = "mean_centered_nearest_centroid"
TOP_K_NEAREST_NEIGHBORS_METHOD = "top_k_nearest_neighbors"
MULTI_PROTOTYPE_NEAREST_CLASS_METHOD = "multi_prototype_nearest_class"
NONPARAMETRIC_METHODS = frozenset(
    {
        NEAREST_CENTROID_METHOD,
        MEAN_CENTERED_NEAREST_CENTROID_METHOD,
        TOP_K_NEAREST_NEIGHBORS_METHOD,
        MULTI_PROTOTYPE_NEAREST_CLASS_METHOD,
    }
)

SCORE_KIND_COSINE_SIMILARITY = "cosine_similarity"
SCORE_KIND_NEIGHBOR_VOTE_FRACTION = "neighbor_vote_fraction"
NONPARAMETRIC_SCORE_KINDS = frozenset(
    {
        SCORE_KIND_COSINE_SIMILARITY,
        SCORE_KIND_NEIGHBOR_VOTE_FRACTION,
    }
)

_UNIT_NORM_TOLERANCE = 1e-5
_ZERO_NORM_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class NonparametricNeighbor:
    """One independent support observation in an exact-k neighborhood."""

    rank: int
    reference_observation_id: str
    accepted_taxon_key: str
    scientific_name: str
    cosine_similarity: float
    duplicate_group_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NonparametricClassScore:
    """Raw, non-probabilistic evidence for one supplied candidate species."""

    accepted_taxon_key: str
    scientific_name: str
    score_kind: str
    raw_score: float | None
    support_count: int
    prototype_count: int
    vote_count: int
    vote_similarity_sum: float | None
    nearest_similarity: float | None
    winning_evidence_id: str | None
    evidence_ids: tuple[str, ...]
    unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class NonparametricPrediction:
    """One deterministic baseline ranking with complete artifact provenance."""

    scoring_version: str
    method: str
    query_id: str
    route: str
    visual_input_kind: str
    geo_cluster_id: str
    top_k: int | None
    prototype_method: str | None
    predicted_taxon_key: str | None
    predicted_scientific_name: str | None
    winner_raw_score: float | None
    runner_up_taxon_key: str | None
    runner_up_scientific_name: str | None
    runner_up_raw_score: float | None
    raw_margin: float | None
    class_scores: tuple[NonparametricClassScore, ...]
    neighbors: tuple[NonparametricNeighbor, ...]
    abstained: bool
    abstention_reasons: tuple[str, ...]
    centering_fingerprint: str | None
    balanced_sampling_seed: int
    model_fingerprint: str
    reference_embedding_fingerprint: str
    reference_prototype_fingerprint: str
    support_manifest_fingerprint: str


class NonparametricBaselineIndex:
    """Validated reference index exposing four estimator-free classifiers."""

    def __init__(
        self,
        reference_embeddings: pl.DataFrame | str | Path,
        reference_prototypes: pl.DataFrame | str | Path,
        *,
        balanced_sampling_seed: int = DEFAULT_BALANCED_SAMPLING_SEED,
    ) -> None:
        self.balanced_sampling_seed = _sampling_seed(balanced_sampling_seed)

        embeddings = _reference_embedding_frame(reference_embeddings)
        support = embeddings.filter(pl.col("support_split") == "support_train")
        if support.is_empty():
            raise ValueError("nonparametric baselines require support_train embeddings")
        self.embedding_dimension = int(support["embedding_dimension"][0])
        self.model_fingerprint = _single_value(support, "model_fingerprint")
        self.support_manifest_fingerprint = _single_value(
            support,
            "support_manifest_fingerprint",
        )
        self.reference_embedding_fingerprint = (
            reference_embeddings_artifact_fingerprint(support)
        )

        prototypes = _reference_prototype_frame(
            reference_prototypes,
            expected_model_fingerprint=self.model_fingerprint,
            expected_reference_embedding_fingerprint=(
                self.reference_embedding_fingerprint
            ),
            expected_support_manifest_fingerprint=self.support_manifest_fingerprint,
        )
        self.reference_prototype_fingerprint = (
            reference_prototypes_artifact_fingerprint(prototypes)
        )
        self._prototype_rows = tuple(prototypes.iter_rows(named=True))

        observations = aggregate_reference_observation_embeddings(embeddings)
        grouped: dict[
            tuple[str, str, str],
            list[ReferenceObservationEmbedding],
        ] = defaultdict(list)
        names_by_key: dict[str, str] = {}
        for observation in observations:
            _record_name(
                names_by_key,
                observation.accepted_taxon_key,
                observation.scientific_name,
            )
            grouped[
                (
                    observation.route,
                    observation.visual_input_kind,
                    observation.accepted_taxon_key,
                )
            ].append(observation)
        for row in self._prototype_rows:
            _record_name(
                names_by_key,
                str(row["accepted_taxon_key"]),
                str(row["species"]),
            )
        self._names_by_key = names_by_key
        self._observations_by_contract = {
            key: tuple(sorted(values, key=_observation_sort_key))
            for key, values in grouped.items()
        }

        centered_rows = tuple(
            row
            for row in self._prototype_rows
            if row["prototype_method"] == PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED
        )
        if centered_rows:
            seeds = {int(row["balanced_sampling_seed"]) for row in centered_rows}
            if seeds != {self.balanced_sampling_seed}:
                raise ValueError(
                    "reference prototype balanced sampling seed does not match "
                    "nonparametric index"
                )
        contexts = build_reference_centering_contexts(
            embeddings,
            balanced_sampling_seed=self.balanced_sampling_seed,
        )
        self._centering_contexts = {
            (context.route, context.visual_input_kind): context for context in contexts
        }
        for row in centered_rows:
            context = self._centering_contexts.get(
                (str(row["route"]), str(row["visual_input_kind"]))
            )
            if (
                context is None
                or row["centering_fingerprint"] != context.centering_fingerprint
            ):
                raise ValueError(
                    "reference prototype centering fingerprint does not match "
                    "nonparametric index"
                )

    def predict_nearest_centroid(
        self,
        query: ReferenceQuery,
        candidates: Sequence[ReferenceCandidate],
    ) -> NonparametricPrediction:
        """Rank candidates by cosine to the raw global species centroid."""

        return self._predict_centroid(
            query,
            candidates,
            method=NEAREST_CENTROID_METHOD,
            prototype_method=PROTOTYPE_METHOD_NORMALIZED_MEAN,
        )

    def predict_mean_centered_nearest_centroid(
        self,
        query: ReferenceQuery,
        candidates: Sequence[ReferenceCandidate],
    ) -> NonparametricPrediction:
        """Rank candidates with the persisted SimpleShot centering context."""

        return self._predict_centroid(
            query,
            candidates,
            method=MEAN_CENTERED_NEAREST_CENTROID_METHOD,
            prototype_method=PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED,
        )

    def predict_top_k_nearest_neighbors(
        self,
        query: ReferenceQuery,
        candidates: Sequence[ReferenceCandidate],
        *,
        k: int,
    ) -> NonparametricPrediction:
        """Use an exact-k, unweighted vote over independent observations."""

        query, candidate_items = self._validated_request(query, candidates)
        fixed_k = _positive_integer(k, field="k")
        retained_by_key: dict[str, tuple[ReferenceObservationEmbedding, ...]] = {}
        missing_reasons: list[str] = []
        unavailable_by_key: dict[str, str | None] = {}
        all_retained: list[ReferenceObservationEmbedding] = []
        for candidate in candidate_items:
            original, retained = self._candidate_observations(query, candidate)
            retained_by_key[candidate.accepted_taxon_key] = retained
            unavailable = _support_unavailable_reason(original, retained)
            unavailable_by_key[candidate.accepted_taxon_key] = unavailable
            if unavailable is not None:
                missing_reasons.append(
                    f"missing_candidate_support:{candidate.accepted_taxon_key}"
                )
            all_retained.extend(retained)

        if len(all_retained) < fixed_k:
            missing_reasons.append(
                f"insufficient_support_for_fixed_k:{len(all_retained)}/{fixed_k}"
            )
            scores = tuple(
                NonparametricClassScore(
                    accepted_taxon_key=candidate.accepted_taxon_key,
                    scientific_name=candidate.scientific_name,
                    score_kind=SCORE_KIND_NEIGHBOR_VOTE_FRACTION,
                    raw_score=None,
                    support_count=len(retained_by_key[candidate.accepted_taxon_key]),
                    prototype_count=0,
                    vote_count=0,
                    vote_similarity_sum=None,
                    nearest_similarity=None,
                    winning_evidence_id=None,
                    evidence_ids=(),
                    unavailable_reason=unavailable_by_key[candidate.accepted_taxon_key],
                )
                for candidate in candidate_items
            )
            return self._prediction(
                query=query,
                method=TOP_K_NEAREST_NEIGHBORS_METHOD,
                scores=scores,
                abstention_reasons=missing_reasons,
                top_k=fixed_k,
            )

        ranked_observations = sorted(
            (
                (_cosine_similarity(query.embedding, item.embedding), item)
                for item in all_retained
            ),
            key=lambda item: (
                -item[0],
                item[1].accepted_taxon_key,
                item[1].reference_observation_id,
            ),
        )[:fixed_k]
        neighbors = tuple(
            NonparametricNeighbor(
                rank=rank,
                reference_observation_id=observation.reference_observation_id,
                accepted_taxon_key=observation.accepted_taxon_key,
                scientific_name=observation.scientific_name,
                cosine_similarity=similarity,
                duplicate_group_ids=observation.duplicate_group_ids,
            )
            for rank, (similarity, observation) in enumerate(
                ranked_observations,
                start=1,
            )
        )
        votes_by_key: dict[str, list[NonparametricNeighbor]] = defaultdict(list)
        for neighbor in neighbors:
            votes_by_key[neighbor.accepted_taxon_key].append(neighbor)

        scores = []
        for candidate in candidate_items:
            key = candidate.accepted_taxon_key
            votes = tuple(votes_by_key.get(key, ()))
            unavailable = unavailable_by_key[key]
            scores.append(
                NonparametricClassScore(
                    accepted_taxon_key=key,
                    scientific_name=candidate.scientific_name,
                    score_kind=SCORE_KIND_NEIGHBOR_VOTE_FRACTION,
                    raw_score=(len(votes) / fixed_k if unavailable is None else None),
                    support_count=len(retained_by_key[key]),
                    prototype_count=0,
                    vote_count=len(votes),
                    vote_similarity_sum=(
                        fsum(item.cosine_similarity for item in votes)
                        if votes
                        else (0.0 if unavailable is None else None)
                    ),
                    nearest_similarity=(votes[0].cosine_similarity if votes else None),
                    winning_evidence_id=(
                        votes[0].reference_observation_id if votes else None
                    ),
                    evidence_ids=tuple(item.reference_observation_id for item in votes),
                    unavailable_reason=unavailable,
                )
            )
        return self._prediction(
            query=query,
            method=TOP_K_NEAREST_NEIGHBORS_METHOD,
            scores=tuple(scores),
            abstention_reasons=missing_reasons,
            neighbors=neighbors,
            top_k=fixed_k,
        )

    def predict_multi_prototype_nearest_class(
        self,
        query: ReferenceQuery,
        candidates: Sequence[ReferenceCandidate],
        *,
        prototype_method: str = PROTOTYPE_METHOD_NORMALIZED_MEAN,
    ) -> NonparametricPrediction:
        """Rank each class by its closest finest-grained global prototype."""

        if prototype_method not in REFERENCE_PROTOTYPE_METHODS:
            raise ValueError(
                f"unsupported reference prototype method: {prototype_method}"
            )
        query, candidate_items = self._validated_request(query, candidates)
        query_embedding, context = self._query_embedding(query, prototype_method)
        scores: list[NonparametricClassScore] = []
        abstention_reasons: list[str] = []
        for candidate in candidate_items:
            original, retained = self._candidate_observations(query, candidate)
            unavailable = _support_unavailable_reason(original, retained)
            if unavailable is not None:
                abstention_reasons.append(
                    f"missing_candidate_support:{candidate.accepted_taxon_key}"
                )
                scores.append(
                    _unavailable_cosine_score(
                        candidate,
                        support_count=len(retained),
                        unavailable_reason=unavailable,
                    )
                )
                continue
            selected = self._multi_prototype_rows(
                query=query,
                candidate=candidate,
                prototype_method=prototype_method,
                retained=retained,
            )
            if not selected:
                if unavailable is None:
                    unavailable = "no_usable_matching_prototype"
                abstention_reasons.append(
                    f"missing_candidate_prototype:{candidate.accepted_taxon_key}"
                )
                scores.append(
                    _unavailable_cosine_score(
                        candidate,
                        support_count=len(retained),
                        unavailable_reason=unavailable,
                    )
                )
                continue
            ranked = sorted(
                (
                    (
                        _cosine_similarity(query_embedding, row["embedding"]),
                        str(row["prototype_id"]),
                    )
                    for row in selected
                ),
                key=lambda item: (-item[0], item[1]),
            )
            scores.append(
                NonparametricClassScore(
                    accepted_taxon_key=candidate.accepted_taxon_key,
                    scientific_name=candidate.scientific_name,
                    score_kind=SCORE_KIND_COSINE_SIMILARITY,
                    raw_score=ranked[0][0],
                    support_count=len(retained),
                    prototype_count=len(selected),
                    vote_count=0,
                    vote_similarity_sum=None,
                    nearest_similarity=ranked[0][0],
                    winning_evidence_id=ranked[0][1],
                    evidence_ids=tuple(sorted(item[1] for item in ranked)),
                    unavailable_reason=None,
                )
            )
        return self._prediction(
            query=query,
            method=MULTI_PROTOTYPE_NEAREST_CLASS_METHOD,
            scores=tuple(scores),
            abstention_reasons=abstention_reasons,
            prototype_method=prototype_method,
            centering_context=context,
        )

    def _predict_centroid(
        self,
        query: ReferenceQuery,
        candidates: Sequence[ReferenceCandidate],
        *,
        method: str,
        prototype_method: str,
    ) -> NonparametricPrediction:
        query, candidate_items = self._validated_request(query, candidates)
        query_embedding, context = self._query_embedding(query, prototype_method)
        scores: list[NonparametricClassScore] = []
        abstention_reasons: list[str] = []
        for candidate in candidate_items:
            original, retained = self._candidate_observations(query, candidate)
            unavailable = _support_unavailable_reason(original, retained)
            if unavailable is not None:
                abstention_reasons.append(
                    f"missing_candidate_support:{candidate.accepted_taxon_key}"
                )
                scores.append(
                    _unavailable_cosine_score(
                        candidate,
                        support_count=len(retained),
                        unavailable_reason=unavailable,
                    )
                )
                continue
            rows = self._matching_prototype_rows(
                query=query,
                candidate=candidate,
                prototype_method=prototype_method,
                prototype_kind=PROTOTYPE_KIND_AGGREGATE,
            )
            if len(rows) != 1:
                if unavailable is None:
                    unavailable = "missing_global_aggregate_prototype"
                abstention_reasons.append(
                    f"missing_candidate_prototype:{candidate.accepted_taxon_key}"
                )
                scores.append(
                    _unavailable_cosine_score(
                        candidate,
                        support_count=len(retained),
                        unavailable_reason=unavailable,
                    )
                )
                continue

            prototype = rows[0]
            affected_by_exclusions = len(original) != len(retained)
            if affected_by_exclusions:
                vectors = tuple(
                    _observation_scoring_embedding(
                        observation,
                        prototype_method=prototype_method,
                        centering_context=context,
                    )
                    for observation in retained
                )
                vector = _normalized_mean(
                    tuple(item for item in vectors if item is not None)
                )
                evidence_id = None
                evidence_ids = tuple(item.reference_observation_id for item in retained)
            else:
                vector = _finite_vector(
                    prototype["embedding"],
                    dimension=self.embedding_dimension,
                    field="prototype embedding",
                )
                evidence_id = str(prototype["prototype_id"])
                evidence_ids = (evidence_id,)
            if vector is None:
                unavailable = "no_usable_support_direction_after_exclusions"
                abstention_reasons.append(
                    f"missing_candidate_prototype:{candidate.accepted_taxon_key}"
                )
                scores.append(
                    _unavailable_cosine_score(
                        candidate,
                        support_count=len(retained),
                        unavailable_reason=unavailable,
                    )
                )
                continue
            raw_score = _cosine_similarity(query_embedding, vector)
            scores.append(
                NonparametricClassScore(
                    accepted_taxon_key=candidate.accepted_taxon_key,
                    scientific_name=candidate.scientific_name,
                    score_kind=SCORE_KIND_COSINE_SIMILARITY,
                    raw_score=raw_score,
                    support_count=len(retained),
                    prototype_count=1,
                    vote_count=0,
                    vote_similarity_sum=None,
                    nearest_similarity=raw_score,
                    winning_evidence_id=evidence_id,
                    evidence_ids=evidence_ids,
                    unavailable_reason=None,
                )
            )
        return self._prediction(
            query=query,
            method=method,
            scores=tuple(scores),
            abstention_reasons=abstention_reasons,
            prototype_method=prototype_method,
            centering_context=context,
        )

    def _validated_request(
        self,
        query: ReferenceQuery,
        candidates: Sequence[ReferenceCandidate],
    ) -> tuple[ReferenceQuery, tuple[ReferenceCandidate, ...]]:
        if not isinstance(query, ReferenceQuery):
            raise TypeError("query must be a ReferenceQuery")
        if query.model_fingerprint != self.model_fingerprint:
            raise ValueError(
                "query model fingerprint does not match nonparametric index"
            )
        if len(query.embedding) != self.embedding_dimension:
            raise ValueError(
                "query embedding dimension does not match nonparametric index"
            )
        candidate_items = tuple(candidates)
        if not candidate_items:
            raise ValueError("nonparametric prediction requires candidate species")
        if any(not isinstance(item, ReferenceCandidate) for item in candidate_items):
            raise TypeError("candidates must contain ReferenceCandidate values")
        keys = [item.accepted_taxon_key for item in candidate_items]
        if len(keys) != len(set(keys)):
            raise ValueError("nonparametric candidate taxon keys must be unique")
        for candidate in candidate_items:
            expected_name = self._names_by_key.get(candidate.accepted_taxon_key)
            if expected_name is not None and expected_name != candidate.scientific_name:
                raise ValueError(
                    "candidate scientific name does not match reference artifacts"
                )
        return query, tuple(
            sorted(
                candidate_items,
                key=lambda item: (item.accepted_taxon_key, item.scientific_name),
            )
        )

    def _candidate_observations(
        self,
        query: ReferenceQuery,
        candidate: ReferenceCandidate,
    ) -> tuple[
        tuple[ReferenceObservationEmbedding, ...],
        tuple[ReferenceObservationEmbedding, ...],
    ]:
        original = self._observations_by_contract.get(
            (
                query.route,
                query.visual_input_kind,
                candidate.accepted_taxon_key,
            ),
            (),
        )
        excluded_ids = set(query.excluded_reference_observation_ids)
        excluded_groups = set(query.excluded_duplicate_group_ids)
        retained = tuple(
            item
            for item in original
            if item.reference_observation_id not in excluded_ids
            and not excluded_groups.intersection(item.duplicate_group_ids)
        )
        return original, retained

    def _query_embedding(
        self,
        query: ReferenceQuery,
        prototype_method: str,
    ) -> tuple[tuple[float, ...], ReferenceCenteringContext | None]:
        if prototype_method == PROTOTYPE_METHOD_NORMALIZED_MEAN:
            return query.embedding, None
        if prototype_method != PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED:
            raise ValueError(
                f"unsupported reference prototype method: {prototype_method}"
            )
        context = self._centering_contexts.get((query.route, query.visual_input_kind))
        if context is None:
            raise ValueError(
                "query route/input contract has no attested mean-centering context"
            )
        return mean_center_query_embedding(query.embedding, context), context

    def _matching_prototype_rows(
        self,
        *,
        query: ReferenceQuery,
        candidate: ReferenceCandidate,
        prototype_method: str,
        prototype_kind: str | None = None,
    ) -> tuple[Mapping[str, object], ...]:
        return tuple(
            row
            for row in self._prototype_rows
            if row["accepted_taxon_key"] == candidate.accepted_taxon_key
            and row["species"] == candidate.scientific_name
            and row["route"] == query.route
            and row["visual_input_kind"] == query.visual_input_kind
            and row["prototype_method"] == prototype_method
            and row["cluster_scope_type"] == PROTOTYPE_SCOPE_GLOBAL
            and row["geo_cluster_id"] == PROTOTYPE_GLOBAL_GEO_CLUSTER_ID
            and (prototype_kind is None or row["prototype_kind"] == prototype_kind)
        )

    def _multi_prototype_rows(
        self,
        *,
        query: ReferenceQuery,
        candidate: ReferenceCandidate,
        prototype_method: str,
        retained: Sequence[ReferenceObservationEmbedding],
    ) -> tuple[Mapping[str, object], ...]:
        rows = self._matching_prototype_rows(
            query=query,
            candidate=candidate,
            prototype_method=prototype_method,
        )
        metadata = sorted(
            (row for row in rows if row["prototype_kind"] == PROTOTYPE_KIND_METADATA),
            key=lambda row: str(row["prototype_id"]),
        )
        clusters_by_parent: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            if row["prototype_kind"] == PROTOTYPE_KIND_EMBEDDING_CLUSTER:
                clusters_by_parent[str(row["metadata_group_id"])].append(row)

        selected: list[Mapping[str, object]] = []
        for row in metadata:
            children = clusters_by_parent.get(str(row["metadata_group_id"]), ())
            selected.extend(children or (row,))
        if not selected:
            selected.extend(
                row for row in rows if row["prototype_kind"] == PROTOTYPE_KIND_AGGREGATE
            )

        retained_ids = {item.reference_observation_id for item in retained}
        uncontaminated = tuple(
            row
            for row in selected
            if set(str(item) for item in row["member_observation_ids"]) <= retained_ids
        )
        return tuple(sorted(uncontaminated, key=lambda row: str(row["prototype_id"])))

    def _prediction(
        self,
        *,
        query: ReferenceQuery,
        method: str,
        scores: Sequence[NonparametricClassScore],
        abstention_reasons: Sequence[str],
        neighbors: tuple[NonparametricNeighbor, ...] = (),
        top_k: int | None = None,
        prototype_method: str | None = None,
        centering_context: ReferenceCenteringContext | None = None,
    ) -> NonparametricPrediction:
        reasons = tuple(sorted(set(abstention_reasons)))
        if method == TOP_K_NEAREST_NEIGHBORS_METHOD:
            ranked = tuple(sorted(scores, key=_knn_score_sort_key))
        else:
            ranked = tuple(sorted(scores, key=_cosine_score_sort_key))
        available = tuple(item for item in ranked if item.raw_score is not None)
        if reasons or not available:
            predicted_key = None
            predicted_name = None
            winner_score = None
            runner_up_key = None
            runner_up_name = None
            runner_up_score = None
            margin = None
            if not reasons:
                reasons = ("no_scoreable_candidate",)
        else:
            winner = available[0]
            runner_up = available[1] if len(available) > 1 else None
            predicted_key = winner.accepted_taxon_key
            predicted_name = winner.scientific_name
            winner_score = winner.raw_score
            runner_up_key = runner_up.accepted_taxon_key if runner_up else None
            runner_up_name = runner_up.scientific_name if runner_up else None
            runner_up_score = runner_up.raw_score if runner_up else None
            margin = (
                winner.raw_score - runner_up.raw_score
                if runner_up is not None
                and winner.raw_score is not None
                and runner_up.raw_score is not None
                else None
            )
        return NonparametricPrediction(
            scoring_version=NONPARAMETRIC_SCORING_VERSION,
            method=method,
            query_id=query.query_id,
            route=query.route,
            visual_input_kind=query.visual_input_kind,
            geo_cluster_id=query.geo_cluster_id,
            top_k=top_k,
            prototype_method=prototype_method,
            predicted_taxon_key=predicted_key,
            predicted_scientific_name=predicted_name,
            winner_raw_score=winner_score,
            runner_up_taxon_key=runner_up_key,
            runner_up_scientific_name=runner_up_name,
            runner_up_raw_score=runner_up_score,
            raw_margin=margin,
            class_scores=ranked,
            neighbors=neighbors,
            abstained=bool(reasons),
            abstention_reasons=reasons,
            centering_fingerprint=(
                centering_context.centering_fingerprint
                if centering_context is not None
                else None
            ),
            balanced_sampling_seed=self.balanced_sampling_seed,
            model_fingerprint=self.model_fingerprint,
            reference_embedding_fingerprint=self.reference_embedding_fingerprint,
            reference_prototype_fingerprint=self.reference_prototype_fingerprint,
            support_manifest_fingerprint=self.support_manifest_fingerprint,
        )


def _reference_embedding_frame(
    source: pl.DataFrame | str | Path,
) -> pl.DataFrame:
    if isinstance(source, pl.DataFrame):
        validate_reference_embeddings(source)
        return source
    return load_reference_embeddings(source)


def _reference_prototype_frame(
    source: pl.DataFrame | str | Path,
    *,
    expected_model_fingerprint: str,
    expected_reference_embedding_fingerprint: str,
    expected_support_manifest_fingerprint: str,
) -> pl.DataFrame:
    if isinstance(source, pl.DataFrame):
        validate_reference_prototypes(
            source,
            expected_model_fingerprint=expected_model_fingerprint,
            expected_reference_embedding_fingerprint=(
                expected_reference_embedding_fingerprint
            ),
            expected_support_manifest_fingerprint=(
                expected_support_manifest_fingerprint
            ),
        )
        return source
    return load_reference_prototypes(
        source,
        expected_model_fingerprint=expected_model_fingerprint,
        expected_reference_embedding_fingerprint=(
            expected_reference_embedding_fingerprint
        ),
        expected_support_manifest_fingerprint=expected_support_manifest_fingerprint,
    )


def _support_unavailable_reason(
    original: Sequence[ReferenceObservationEmbedding],
    retained: Sequence[ReferenceObservationEmbedding],
) -> str | None:
    if retained:
        return None
    if original:
        return "no_matching_support_observation_after_exclusions"
    return "no_matching_support_observation"


def _unavailable_cosine_score(
    candidate: ReferenceCandidate,
    *,
    support_count: int,
    unavailable_reason: str,
) -> NonparametricClassScore:
    return NonparametricClassScore(
        accepted_taxon_key=candidate.accepted_taxon_key,
        scientific_name=candidate.scientific_name,
        score_kind=SCORE_KIND_COSINE_SIMILARITY,
        raw_score=None,
        support_count=support_count,
        prototype_count=0,
        vote_count=0,
        vote_similarity_sum=None,
        nearest_similarity=None,
        winning_evidence_id=None,
        evidence_ids=(),
        unavailable_reason=unavailable_reason,
    )


def _observation_scoring_embedding(
    observation: ReferenceObservationEmbedding,
    *,
    prototype_method: str,
    centering_context: ReferenceCenteringContext | None,
) -> tuple[float, ...] | None:
    if prototype_method == PROTOTYPE_METHOD_NORMALIZED_MEAN:
        return observation.embedding
    if prototype_method != PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED:
        raise ValueError(f"unsupported reference prototype method: {prototype_method}")
    if centering_context is None:
        raise ValueError("mean-centered observation scoring requires a context")
    centered = tuple(
        value - mean
        for value, mean in zip(
            observation.embedding,
            centering_context.mean_embedding,
            strict=True,
        )
    )
    norm = _vector_norm(centered)
    if norm <= _ZERO_NORM_EPSILON:
        return None
    return tuple(value / norm for value in centered)


def _normalized_mean(
    vectors: Sequence[Sequence[float]],
) -> tuple[float, ...] | None:
    if not vectors:
        return None
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("vectors must have one embedding dimension")
    mean = tuple(
        fsum(vector[index] for vector in vectors) / len(vectors)
        for index in range(dimension)
    )
    norm = _vector_norm(mean)
    if norm <= _ZERO_NORM_EPSILON:
        return None
    return tuple(value / norm for value in mean)


def _cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("cosine vectors must have one embedding dimension")
    value = fsum(a * b for a, b in zip(left, right, strict=True))
    if not isfinite(value):
        raise ValueError("cosine similarity must be finite")
    return max(-1.0, min(1.0, value))


def _finite_vector(
    values: object,
    *,
    dimension: int,
    field: str,
) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be a numeric sequence")
    result = tuple(float(value) for value in values)
    if len(result) != dimension:
        raise ValueError(f"{field} must have dimension {dimension}")
    if any(not isfinite(value) for value in result):
        raise ValueError(f"{field} must contain only finite values")
    norm = _vector_norm(result)
    if abs(norm - 1.0) > _UNIT_NORM_TOLERANCE:
        raise ValueError(f"{field} must be unit-normalized")
    return result


def _vector_norm(values: Sequence[float]) -> float:
    return sqrt(fsum(value * value for value in values))


def _cosine_score_sort_key(
    score: NonparametricClassScore,
) -> tuple[object, ...]:
    return (
        score.raw_score is None,
        -(score.raw_score if score.raw_score is not None else -2.0),
        score.accepted_taxon_key,
        score.scientific_name,
    )


def _knn_score_sort_key(
    score: NonparametricClassScore,
) -> tuple[object, ...]:
    return (
        score.raw_score is None,
        -score.vote_count,
        -(
            score.vote_similarity_sum
            if score.vote_similarity_sum is not None
            else -float("inf")
        ),
        -(
            score.nearest_similarity
            if score.nearest_similarity is not None
            else -float("inf")
        ),
        score.accepted_taxon_key,
        score.scientific_name,
    )


def _record_name(names: dict[str, str], key: str, scientific_name: str) -> None:
    previous = names.setdefault(key, scientific_name)
    if previous != scientific_name:
        raise ValueError("reference artifacts map one taxon key to multiple names")


def _observation_sort_key(
    observation: ReferenceObservationEmbedding,
) -> tuple[str, ...]:
    return (
        observation.accepted_taxon_key,
        observation.reference_observation_id,
        observation.geo_cluster_id,
        observation.life_stage,
        observation.visual_domain,
        observation.view,
    )


def _single_value(frame: pl.DataFrame, field: str) -> str:
    values = frame[field].unique().to_list()
    if len(values) != 1 or not isinstance(values[0], str) or not values[0]:
        raise ValueError(f"reference embeddings require one {field}")
    return values[0]


def _sampling_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("balanced_sampling_seed must be a non-negative integer")
    if value > 2**64 - 1:
        raise ValueError("balanced_sampling_seed exceeds UInt64")
    return value


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


__all__ = [
    "MEAN_CENTERED_NEAREST_CENTROID_METHOD",
    "MULTI_PROTOTYPE_NEAREST_CLASS_METHOD",
    "NEAREST_CENTROID_METHOD",
    "NONPARAMETRIC_METHODS",
    "NONPARAMETRIC_SCORE_KINDS",
    "NONPARAMETRIC_SCORING_VERSION",
    "SCORE_KIND_COSINE_SIMILARITY",
    "SCORE_KIND_NEIGHBOR_VOTE_FRACTION",
    "TOP_K_NEAREST_NEIGHBORS_METHOD",
    "NonparametricBaselineIndex",
    "NonparametricClassScore",
    "NonparametricNeighbor",
    "NonparametricPrediction",
]
