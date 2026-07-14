"""Class-balanced nearest-reference evidence over frozen BioCLIP embeddings."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from pathlib import Path
import re

import polars as pl

from biominer.bioclip.reference_embeddings import (
    load_reference_embeddings,
    reference_embeddings_artifact_fingerprint,
    validate_reference_embeddings,
)
from biominer.bioclip.reference_prototypes import (
    DEFAULT_BALANCED_SAMPLING_SEED,
    PROTOTYPE_GLOBAL_GEO_CLUSTER_ID,
    PROTOTYPE_METHOD_NORMALIZED_MEAN,
    PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED,
    PROTOTYPE_SCOPE_GLOBAL,
    PROTOTYPE_SCOPE_REGIONAL,
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
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.flickr_fetch.geographic_clustering import NO_GEO_CLUSTER_ID
from biominer.references.readiness import REFERENCE_ROUTES
from biominer.vision.full_frame_attention import (
    FOCUSED_FULL_FRAME_KIND,
    MASKED_FULL_FRAME_KIND,
    MULTI_OBJECT_FULL_FRAME_KIND,
    RAW_FULL_IMAGE_KIND,
)


DEFAULT_BALANCED_REFERENCE_COUNT = 5
REFERENCE_EVIDENCE_SCORING_VERSION = "reference-evidence-scoring-v1.0.0"

_VISUAL_INPUT_KINDS = frozenset(
    {
        RAW_FULL_IMAGE_KIND,
        FOCUSED_FULL_FRAME_KIND,
        MASKED_FULL_FRAME_KIND,
        MULTI_OBJECT_FULL_FRAME_KIND,
    }
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_UNIT_NORM_TOLERANCE = 1e-5
_ZERO_NORM_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class ReferenceCandidate:
    """One accepted species that must receive reference evidence."""

    accepted_taxon_key: str
    scientific_name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "accepted_taxon_key",
            _required_text(self.accepted_taxon_key, field="accepted_taxon_key"),
        )
        object.__setattr__(
            self,
            "scientific_name",
            _required_text(self.scientific_name, field="scientific_name"),
        )


@dataclass(frozen=True, slots=True)
class ReferenceQuery:
    """A unit query embedding plus the route and geographic scoring context."""

    query_id: str
    embedding: tuple[float, ...]
    route: str
    visual_input_kind: str
    geo_cluster_id: str
    model_fingerprint: str
    excluded_reference_observation_ids: tuple[str, ...] = ()
    excluded_duplicate_group_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "query_id", _required_text(self.query_id, field="query_id")
        )
        if self.route not in REFERENCE_ROUTES:
            raise ValueError(f"unsupported reference query route: {self.route}")
        if self.visual_input_kind not in _VISUAL_INPUT_KINDS:
            raise ValueError(
                "unsupported reference query visual input kind: "
                f"{self.visual_input_kind}"
            )
        object.__setattr__(
            self,
            "geo_cluster_id",
            _required_text(self.geo_cluster_id, field="geo_cluster_id"),
        )
        object.__setattr__(
            self,
            "model_fingerprint",
            _sha256(self.model_fingerprint, field="model_fingerprint"),
        )
        vector = _finite_vector(self.embedding, field="query embedding")
        norm = _vector_norm(vector)
        if abs(norm - 1.0) > _UNIT_NORM_TOLERANCE:
            raise ValueError("reference query embedding must be unit-normalized")
        object.__setattr__(self, "embedding", vector)
        object.__setattr__(
            self,
            "excluded_reference_observation_ids",
            _unique_text_tuple(
                self.excluded_reference_observation_ids,
                field="excluded_reference_observation_ids",
            ),
        )
        object.__setattr__(
            self,
            "excluded_duplicate_group_ids",
            _unique_text_tuple(
                self.excluded_duplicate_group_ids,
                field="excluded_duplicate_group_ids",
            ),
        )


@dataclass(frozen=True, slots=True)
class CandidateReferenceEvidence:
    """Auditable similarities for one query/candidate species pair."""

    scoring_version: str
    query_id: str
    accepted_taxon_key: str
    scientific_name: str
    route: str
    visual_input_kind: str
    geo_cluster_id: str
    prototype_method: str
    balanced_sampling_seed: int
    fixed_reference_count: int
    support_count: int
    usable_support_count: int
    local_support_count: int
    selected_support_count: int
    selected_local_support_count: int
    selected_reference_observation_ids: tuple[str, ...]
    nearest_reference_observation_id: str | None
    nearest_support_similarity: float | None
    mean_top_three_similarity: float | None
    mean_top_five_similarity: float | None
    centroid_similarity: float | None
    local_cluster_prototype_similarity: float | None
    global_prototype_similarity: float | None
    distance_to_nearest_independent_observation: float | None
    insufficient_support: bool
    insufficient_support_reasons: tuple[str, ...]
    local_support_available: bool
    local_prototype_available: bool
    global_prototype_available: bool
    query_embedding_norm: float
    centering_fingerprint: str | None
    model_fingerprint: str
    reference_embedding_fingerprint: str
    reference_prototype_fingerprint: str
    support_manifest_fingerprint: str


class ReferenceEvidenceIndex:
    """Validated reusable index for class-balanced reference scoring."""

    def __init__(
        self,
        reference_embeddings: pl.DataFrame | str | Path,
        reference_prototypes: pl.DataFrame | str | Path,
        *,
        balanced_reference_count: int = DEFAULT_BALANCED_REFERENCE_COUNT,
        balanced_sampling_seed: int = DEFAULT_BALANCED_SAMPLING_SEED,
        prototype_method: str = PROTOTYPE_METHOD_NORMALIZED_MEAN,
    ) -> None:
        self.balanced_reference_count = _fixed_reference_count(balanced_reference_count)
        self.balanced_sampling_seed = _sampling_seed(balanced_sampling_seed)
        if prototype_method not in REFERENCE_PROTOTYPE_METHODS:
            raise ValueError(
                f"unsupported reference prototype method: {prototype_method}"
            )
        self.prototype_method = prototype_method

        embeddings = _reference_embedding_frame(reference_embeddings)
        support = embeddings.filter(pl.col("support_split") == "support_train")
        if support.is_empty():
            raise ValueError("reference scoring requires support_train embeddings")
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
        self._prototype_rows = _prototype_index(prototypes)
        self._observations = aggregate_reference_observation_embeddings(embeddings)
        grouped: dict[tuple[str, str, str], list[ReferenceObservationEmbedding]] = (
            defaultdict(list)
        )
        for observation in self._observations:
            grouped[
                (
                    observation.route,
                    observation.visual_input_kind,
                    observation.accepted_taxon_key,
                )
            ].append(observation)
        self._observations_by_contract = {
            key: tuple(values) for key, values in grouped.items()
        }
        _validate_prototype_coverage(
            prototype_method=prototype_method,
            observations=self._observations,
            prototype_rows=self._prototype_rows,
        )

        self._centering_contexts: dict[tuple[str, str], ReferenceCenteringContext] = {}
        if prototype_method == PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED:
            method_rows = prototypes.filter(
                pl.col("prototype_method") == prototype_method
            )
            if method_rows.is_empty():
                raise ValueError(
                    "reference prototypes contain no mean-centered scoring method"
                )
            seeds = method_rows["balanced_sampling_seed"].unique().to_list()
            if seeds != [self.balanced_sampling_seed]:
                raise ValueError(
                    "reference prototype balanced sampling seed does not match scorer"
                )
            contexts = build_reference_centering_contexts(
                embeddings,
                balanced_sampling_seed=self.balanced_sampling_seed,
            )
            self._centering_contexts = {
                (context.route, context.visual_input_kind): context
                for context in contexts
            }
            for row in method_rows.iter_rows(named=True):
                context = self._centering_contexts.get(
                    (str(row["route"]), str(row["visual_input_kind"]))
                )
                if (
                    context is None
                    or row["centering_fingerprint"] != context.centering_fingerprint
                ):
                    raise ValueError(
                        "reference prototype centering fingerprint does not match scorer"
                    )

    def score(
        self,
        query: ReferenceQuery,
        candidates: Sequence[ReferenceCandidate],
    ) -> tuple[CandidateReferenceEvidence, ...]:
        """Score every supplied species without hierarchy-based candidate deletion."""

        if not isinstance(query, ReferenceQuery):
            raise TypeError("query must be a ReferenceQuery")
        if query.model_fingerprint != self.model_fingerprint:
            raise ValueError("query model fingerprint does not match reference index")
        if len(query.embedding) != self.embedding_dimension:
            raise ValueError("query embedding dimension does not match reference index")
        candidate_items = tuple(candidates)
        if not candidate_items:
            raise ValueError(
                "reference scoring requires at least one candidate species"
            )
        if any(not isinstance(item, ReferenceCandidate) for item in candidate_items):
            raise TypeError("candidates must contain ReferenceCandidate values")
        keys = [item.accepted_taxon_key for item in candidate_items]
        if len(keys) != len(set(keys)):
            raise ValueError("reference scoring candidate taxon keys must be unique")

        context = self._centering_contexts.get((query.route, query.visual_input_kind))
        if self.prototype_method == PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED:
            if context is None:
                raise ValueError(
                    "reference query contract has no mean-centering context"
                )
            query_embedding = mean_center_query_embedding(query.embedding, context)
            centering_fingerprint = context.centering_fingerprint
        else:
            query_embedding = query.embedding
            centering_fingerprint = None

        return tuple(
            self._score_candidate(
                query=query,
                query_embedding=query_embedding,
                centering_context=context,
                centering_fingerprint=centering_fingerprint,
                candidate=candidate,
            )
            for candidate in candidate_items
        )

    def _score_candidate(
        self,
        *,
        query: ReferenceQuery,
        query_embedding: tuple[float, ...],
        centering_context: ReferenceCenteringContext | None,
        centering_fingerprint: str | None,
        candidate: ReferenceCandidate,
    ) -> CandidateReferenceEvidence:
        observations = self._observations_by_contract.get(
            (
                query.route,
                query.visual_input_kind,
                candidate.accepted_taxon_key,
            ),
            (),
        )
        _validate_candidate_name(candidate, observations)
        excluded_observation_ids = set(query.excluded_reference_observation_ids)
        excluded_duplicate_groups = set(query.excluded_duplicate_group_ids)
        retained = tuple(
            observation
            for observation in observations
            if observation.reference_observation_id not in excluded_observation_ids
            and not excluded_duplicate_groups.intersection(
                observation.duplicate_group_ids
            )
        )
        has_exclusions = len(retained) != len(observations)
        local = tuple(
            observation
            for observation in retained
            if query.geo_cluster_id != NO_GEO_CLUSTER_ID
            and observation.geo_cluster_id == query.geo_cluster_id
        )

        vectors: dict[str, tuple[float, ...] | None] = {}
        for observation in retained:
            vectors[observation.reference_observation_id] = (
                _observation_scoring_embedding(
                    observation,
                    prototype_method=self.prototype_method,
                    centering_context=centering_context,
                )
            )
        usable = tuple(
            observation
            for observation in retained
            if vectors[observation.reference_observation_id] is not None
        )
        local_ids = {observation.reference_observation_id for observation in local}
        ranked_local = sorted(
            (
                observation
                for observation in usable
                if observation.reference_observation_id in local_ids
            ),
            key=lambda item: _balanced_observation_rank(
                item,
                balanced_sampling_seed=self.balanced_sampling_seed,
            ),
        )
        ranked_global = sorted(
            (
                observation
                for observation in usable
                if observation.reference_observation_id not in local_ids
            ),
            key=lambda item: _balanced_observation_rank(
                item,
                balanced_sampling_seed=self.balanced_sampling_seed,
            ),
        )
        selected = tuple(
            (ranked_local + ranked_global)[: self.balanced_reference_count]
        )
        selected_vectors = tuple(
            _required_scoring_vector(vectors[observation.reference_observation_id])
            for observation in selected
        )
        similarities = sorted(
            (
                (
                    _cosine_similarity(query_embedding, vector),
                    observation.reference_observation_id,
                )
                for observation, vector in zip(
                    selected,
                    selected_vectors,
                    strict=True,
                )
            ),
            key=lambda item: (-item[0], item[1]),
        )
        nearest_similarity = similarities[0][0] if similarities else None
        nearest_observation_id = similarities[0][1] if similarities else None
        top_three = _fixed_top_mean(similarities, 3)
        top_five = _fixed_top_mean(similarities, 5)
        centroid = _normalized_mean(selected_vectors)
        centroid_similarity = (
            _cosine_similarity(query_embedding, centroid)
            if centroid is not None
            else None
        )

        global_prototype = self._prototype_embedding(
            candidate=candidate,
            query=query,
            scope_type=PROTOTYPE_SCOPE_GLOBAL,
            geo_cluster_id=PROTOTYPE_GLOBAL_GEO_CLUSTER_ID,
            retained=retained,
            vectors=vectors,
            has_exclusions=has_exclusions,
        )
        local_prototype = None
        if query.geo_cluster_id != NO_GEO_CLUSTER_ID and local:
            local_prototype = self._prototype_embedding(
                candidate=candidate,
                query=query,
                scope_type=PROTOTYPE_SCOPE_REGIONAL,
                geo_cluster_id=query.geo_cluster_id,
                retained=local,
                vectors=vectors,
                has_exclusions=has_exclusions,
            )
        global_similarity = (
            _cosine_similarity(query_embedding, global_prototype)
            if global_prototype is not None
            else None
        )
        local_similarity = (
            _cosine_similarity(query_embedding, local_prototype)
            if local_prototype is not None
            else None
        )
        reasons = _insufficient_support_reasons(
            support_count=len(retained),
            usable_support_count=len(usable),
            selected_support_count=len(selected),
            balanced_reference_count=self.balanced_reference_count,
        )
        return CandidateReferenceEvidence(
            scoring_version=REFERENCE_EVIDENCE_SCORING_VERSION,
            query_id=query.query_id,
            accepted_taxon_key=candidate.accepted_taxon_key,
            scientific_name=candidate.scientific_name,
            route=query.route,
            visual_input_kind=query.visual_input_kind,
            geo_cluster_id=query.geo_cluster_id,
            prototype_method=self.prototype_method,
            balanced_sampling_seed=self.balanced_sampling_seed,
            fixed_reference_count=self.balanced_reference_count,
            support_count=len(retained),
            usable_support_count=len(usable),
            local_support_count=len(local),
            selected_support_count=len(selected),
            selected_local_support_count=sum(
                observation.reference_observation_id in local_ids
                for observation in selected
            ),
            selected_reference_observation_ids=tuple(
                observation.reference_observation_id for observation in selected
            ),
            nearest_reference_observation_id=nearest_observation_id,
            nearest_support_similarity=nearest_similarity,
            mean_top_three_similarity=top_three,
            mean_top_five_similarity=top_five,
            centroid_similarity=centroid_similarity,
            local_cluster_prototype_similarity=local_similarity,
            global_prototype_similarity=global_similarity,
            distance_to_nearest_independent_observation=(
                1.0 - nearest_similarity if nearest_similarity is not None else None
            ),
            insufficient_support=bool(reasons),
            insufficient_support_reasons=reasons,
            local_support_available=bool(local),
            local_prototype_available=local_prototype is not None,
            global_prototype_available=global_prototype is not None,
            query_embedding_norm=_vector_norm(query_embedding),
            centering_fingerprint=centering_fingerprint,
            model_fingerprint=self.model_fingerprint,
            reference_embedding_fingerprint=self.reference_embedding_fingerprint,
            reference_prototype_fingerprint=self.reference_prototype_fingerprint,
            support_manifest_fingerprint=self.support_manifest_fingerprint,
        )

    def _prototype_embedding(
        self,
        *,
        candidate: ReferenceCandidate,
        query: ReferenceQuery,
        scope_type: str,
        geo_cluster_id: str,
        retained: Sequence[ReferenceObservationEmbedding],
        vectors: dict[str, tuple[float, ...] | None],
        has_exclusions: bool,
    ) -> tuple[float, ...] | None:
        key = (
            candidate.accepted_taxon_key,
            query.route,
            query.visual_input_kind,
            self.prototype_method,
            scope_type,
            geo_cluster_id,
        )
        row = self._prototype_rows.get(key)
        if row is not None and str(row["species"]) != candidate.scientific_name:
            raise ValueError(
                "candidate scientific name does not match reference prototype"
            )
        if not has_exclusions:
            if row is None:
                return None
            return tuple(float(value) for value in row["embedding"])

        dynamic_vectors = tuple(
            vectors[observation.reference_observation_id]
            or tuple(0.0 for _ in range(self.embedding_dimension))
            for observation in retained
        )
        return _normalized_mean(dynamic_vectors)


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


def _prototype_index(
    prototypes: pl.DataFrame,
) -> dict[tuple[str, str, str, str, str, str], dict[str, object]]:
    result: dict[tuple[str, str, str, str, str, str], dict[str, object]] = {}
    for row in prototypes.iter_rows(named=True):
        key = (
            str(row["accepted_taxon_key"]),
            str(row["route"]),
            str(row["visual_input_kind"]),
            str(row["prototype_method"]),
            str(row["cluster_scope_type"]),
            str(row["geo_cluster_id"]),
        )
        if key in result:
            raise ValueError("reference prototypes repeat a scoring identity")
        result[key] = row
    return result


def _validate_prototype_coverage(
    *,
    prototype_method: str,
    observations: Sequence[ReferenceObservationEmbedding],
    prototype_rows: dict[tuple[str, str, str, str, str, str], dict[str, object]],
) -> None:
    if prototype_method != PROTOTYPE_METHOD_NORMALIZED_MEAN:
        return
    expected: set[tuple[str, str, str, str, str, str]] = set()
    expected_names: dict[tuple[str, str, str, str, str, str], str] = {}
    for observation in observations:
        base = (
            observation.accepted_taxon_key,
            observation.route,
            observation.visual_input_kind,
            prototype_method,
        )
        global_key = (
            *base,
            PROTOTYPE_SCOPE_GLOBAL,
            PROTOTYPE_GLOBAL_GEO_CLUSTER_ID,
        )
        regional_key = (
            *base,
            PROTOTYPE_SCOPE_REGIONAL,
            observation.geo_cluster_id,
        )
        for key in (global_key, regional_key):
            expected.add(key)
            previous = expected_names.setdefault(key, observation.scientific_name)
            if previous != observation.scientific_name:
                raise ValueError(
                    "reference support maps one prototype identity to multiple names"
                )
    actual = {
        key for key in prototype_rows if key[3] == PROTOTYPE_METHOD_NORMALIZED_MEAN
    }
    if actual != expected:
        raise ValueError(
            "raw reference prototype coverage does not match support observations"
        )
    for key, scientific_name in expected_names.items():
        if str(prototype_rows[key]["species"]) != scientific_name:
            raise ValueError(
                "raw reference prototype name does not match support observations"
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
        raise ValueError("mean-centered reference scoring requires a context")
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


def _balanced_observation_rank(
    observation: ReferenceObservationEmbedding,
    *,
    balanced_sampling_seed: int,
) -> str:
    return canonical_semantic_fingerprint(
        {
            "balanced_sampling_seed": balanced_sampling_seed,
            "accepted_taxon_key": observation.accepted_taxon_key,
            "reference_observation_id": observation.reference_observation_id,
            "route": observation.route,
            "visual_input_kind": observation.visual_input_kind,
        }
    )


def _fixed_top_mean(
    similarities: Sequence[tuple[float, str]],
    k: int,
) -> float | None:
    if len(similarities) < k:
        return None
    return fsum(value for value, _ in similarities[:k]) / k


def _normalized_mean(
    vectors: Sequence[Sequence[float]],
) -> tuple[float, ...] | None:
    if not vectors:
        return None
    dimension = len(vectors[0])
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
    value = fsum(a * b for a, b in zip(left, right, strict=True))
    return min(1.0, max(-1.0, value))


def _insufficient_support_reasons(
    *,
    support_count: int,
    usable_support_count: int,
    selected_support_count: int,
    balanced_reference_count: int,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if support_count == 0:
        reasons.append("no_route_support")
    elif usable_support_count == 0:
        reasons.append("no_usable_support")
    if selected_support_count < balanced_reference_count:
        reasons.append("fewer_than_balanced_reference_count")
    if selected_support_count < 3:
        reasons.append("fewer_than_three_independent_observations")
    if selected_support_count < 5:
        reasons.append("fewer_than_five_independent_observations")
    return tuple(reasons)


def _validate_candidate_name(
    candidate: ReferenceCandidate,
    observations: Sequence[ReferenceObservationEmbedding],
) -> None:
    names = {observation.scientific_name for observation in observations}
    if names and names != {candidate.scientific_name}:
        raise ValueError("candidate scientific name does not match reference support")


def _required_scoring_vector(
    vector: tuple[float, ...] | None,
) -> tuple[float, ...]:
    if vector is None:
        raise AssertionError("selected reference observation has no scoring direction")
    return vector


def _finite_vector(values: Sequence[float], *, field: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be a numeric sequence")
    result: list[float] = []
    for raw in values:
        if isinstance(raw, bool):
            raise ValueError(f"{field} must contain finite numeric values")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must contain finite numeric values") from exc
        if not isfinite(value):
            raise ValueError(f"{field} must contain finite numeric values")
        result.append(value)
    if not result:
        raise ValueError(f"{field} must not be empty")
    return tuple(result)


def _vector_norm(values: Sequence[float]) -> float:
    return sqrt(fsum(value * value for value in values))


def _single_value(frame: pl.DataFrame, field: str) -> str:
    values = frame[field].unique().to_list()
    if len(values) != 1:
        raise ValueError(f"reference embeddings have mixed {field} values")
    return _sha256(values[0], field=field)


def _fixed_reference_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("balanced_reference_count must be an integer")
    if value < 5:
        raise ValueError("balanced_reference_count must be at least five")
    return value


def _sampling_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("balanced_sampling_seed must be an integer")
    if not 0 <= value <= 18_446_744_073_709_551_615:
        raise ValueError("balanced_sampling_seed must fit an unsigned 64-bit integer")
    return value


def _unique_text_tuple(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be a string sequence")
    result = tuple(sorted(_required_text(value, field=field) for value in values))
    if len(result) != len(set(result)):
        raise ValueError(f"{field} must contain unique values")
    return result


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a sha256 fingerprint")
    return value


__all__ = [
    "DEFAULT_BALANCED_REFERENCE_COUNT",
    "REFERENCE_EVIDENCE_SCORING_VERSION",
    "CandidateReferenceEvidence",
    "ReferenceCandidate",
    "ReferenceEvidenceIndex",
    "ReferenceQuery",
]
