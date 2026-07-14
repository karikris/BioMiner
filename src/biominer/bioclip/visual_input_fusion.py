"""Versioned full-frame embedding reuse and visual-input score fusion."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import hypot, isclose, isfinite
import re

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.readiness import REFERENCE_ROUTES
from biominer.vision.full_frame_attention import (
    FOCUSED_FULL_FRAME_KIND,
    FULL_FRAME_VISUAL_INPUT_VERSION,
    MASKED_FULL_FRAME_KIND,
    MULTI_OBJECT_FULL_FRAME_KIND,
    RAW_FULL_IMAGE_KIND,
    TARGET_FULL_FRAME_IMAGE_RESIZE_MODE,
    TARGET_FULL_FRAME_PREPROCESSING,
    FullFrameAttentionResult,
    FullFrameAttentionVariant,
    FullFrameVariantEvidence,
    UnavailableAttentionVariant,
    decoded_image_content_hash,
    raw_full_frame_visual_input,
)
from biominer.vision.target_full_frame import (
    TARGET_FULL_FRAME_EMBEDDING_VERSION,
    FullFrameImageEncoder,
    RawFullFrameEmbedding,
    full_frame_embedding_id,
)


VISUAL_INPUT_EMBEDDING_SET_VERSION = "target-aware-full-frame-embedding-set-v1.0.0"
VISUAL_INPUT_SCORE_VERSION = "target-aware-visual-input-score-v1.0.0"
VISUAL_INPUT_FUSION_POLICY_SCHEMA_VERSION = (
    "target-aware-visual-input-fusion-policy-v1.0.0"
)
VISUAL_INPUT_FUSION_POLICY_VERSION = "full-frame-linear-fusion-v1.0.0"
VISUAL_INPUT_FUSION_RESULT_VERSION = "full-frame-visual-evidence-fusion-v1.0.0"
VISUAL_INPUT_DISAGREEMENT_VERSION = "regional-competitor-margin-range-v1.0.0"
VISUAL_INPUT_WITHIN_KIND_REDUCTION = (
    "arithmetic-mean-with-explicit-input-weights-v1.0.0"
)

RAW_ONLY_VISUAL_INPUT_FUSION = "raw_only"
LEARNED_LINEAR_VISUAL_INPUT_FUSION = "validation_learned_linear"
VISUAL_INPUT_FUSION_STRATEGIES = frozenset(
    {RAW_ONLY_VISUAL_INPUT_FUSION, LEARNED_LINEAR_VISUAL_INPUT_FUSION}
)
STRONG_VISUAL_INPUT_DISAGREEMENT = "strong_visual_input_disagreement"

_VISUAL_INPUT_KINDS = (
    RAW_FULL_IMAGE_KIND,
    FOCUSED_FULL_FRAME_KIND,
    MASKED_FULL_FRAME_KIND,
    MULTI_OBJECT_FULL_FRAME_KIND,
)
_VISUAL_INPUT_KIND_ORDER = {
    value: index for index, value in enumerate(_VISUAL_INPUT_KINDS)
}
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class FullFrameVisualInputEmbedding:
    """One immutable BioCLIP embedding for a complete-canvas visual input."""

    embedding_id: str
    embedding_version: str
    embedding_fingerprint: str
    visual_input_id: str
    visual_input_kind: str
    visual_input_version: str
    raw_image_content_hash: str
    visual_content_hash: str
    transformation_fingerprint: str
    model_fingerprint: str
    image_resize_mode: str
    preprocessing_contract_fingerprint: str
    preprocessing_fingerprint: str
    embedding_dimension: int
    embedding: tuple[float, ...]
    embedding_norm: float


@dataclass(frozen=True, slots=True)
class EmbeddedFullFrameVisualInputs:
    """Available embeddings plus retained unavailable-variant evidence."""

    embedding_set_version: str
    visual_input_version: str
    raw_image_content_hash: str
    route: str
    model_fingerprint: str
    preprocessing_fingerprint: str
    embeddings: tuple[FullFrameVisualInputEmbedding, ...]
    attention_evidence: tuple[FullFrameVariantEvidence, ...]
    unavailable_variants: tuple[UnavailableAttentionVariant, ...]
    embedding_set_fingerprint: str

    def embedding(self, visual_input_kind: str) -> FullFrameVisualInputEmbedding:
        kind = _required_choice(
            visual_input_kind,
            field="visual_input_kind",
            allowed=frozenset(_VISUAL_INPUT_KINDS),
        )
        matches = tuple(
            item for item in self.embeddings if item.visual_input_kind == kind
        )
        if len(matches) != 1:
            raise KeyError(kind)
        return matches[0]


@dataclass(frozen=True, slots=True)
class FullFrameCandidateScore:
    """Comparable regional and reference scores for one candidate and input."""

    accepted_taxon_key: str
    scientific_name: str
    regional_classifier_decision_score: float
    reference_centroid_similarity: float | None

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
        object.__setattr__(
            self,
            "regional_classifier_decision_score",
            _finite_float(
                self.regional_classifier_decision_score,
                field="regional_classifier_decision_score",
            ),
        )
        object.__setattr__(
            self,
            "reference_centroid_similarity",
            _optional_bounded_float(
                self.reference_centroid_similarity,
                field="reference_centroid_similarity",
                minimum=-1.0,
                maximum=1.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class FullFrameVisualInputScore:
    """Complete raw score evidence produced from one visual-input embedding."""

    score_version: str
    visual_input_id: str
    visual_input_kind: str
    transformation_fingerprint: str
    embedding_id: str
    embedding_fingerprint: str
    model_fingerprint: str
    preprocessing_fingerprint: str
    route: str
    target_accepted_taxon_key: str
    candidate_set_fingerprint: str
    target_classifier_fingerprint: str
    regional_classifier_fingerprint: str
    target_classifier_decision_score: float
    candidate_scores: tuple[FullFrameCandidateScore, ...]
    target_regional_classifier_decision_score: float
    best_competitor_accepted_taxon_key: str
    best_competitor_scientific_name: str
    best_competitor_regional_classifier_decision_score: float
    regional_competitor_margin: float
    target_reference_centroid_similarity: float | None
    best_competitor_reference_centroid_similarity: float | None
    reference_competitor_margin: float | None
    source_fusion_fingerprint: str
    score_contract_fingerprint: str
    score_fingerprint: str


@dataclass(frozen=True, slots=True)
class VisualInputFusionPolicy:
    """Validation-bound coefficients and disagreement threshold."""

    schema_version: str
    policy_version: str
    strategy: str
    route: str
    visual_input_version: str
    score_version: str
    within_kind_reduction: str
    visual_input_kind_weights: tuple[tuple[str, float], ...]
    disagreement_version: str
    strong_disagreement_threshold: float
    model_fingerprint: str
    preprocessing_fingerprint: str
    target_classifier_fingerprint: str
    regional_classifier_fingerprint: str
    split_fingerprint: str
    training_data_fingerprint: str
    fit_partition: str
    policy_fingerprint: str


@dataclass(frozen=True, slots=True)
class AggregatedFullFrameCandidateScore:
    """One candidate's raw scores after visual-input linear pooling."""

    accepted_taxon_key: str
    scientific_name: str
    regional_classifier_decision_score: float
    reference_centroid_similarity: float | None


@dataclass(frozen=True, slots=True)
class FullFrameVisualInputEvidence:
    """Auditable per-input evidence and its exact aggregation contribution."""

    visual_input_id: str
    visual_input_kind: str
    transformation_fingerprint: str
    embedding_id: str
    embedding_fingerprint: str
    source_fusion_fingerprint: str
    target_classifier_decision_score: float
    target_regional_classifier_decision_score: float
    best_competitor_accepted_taxon_key: str
    best_competitor_scientific_name: str
    best_competitor_regional_classifier_decision_score: float
    regional_competitor_margin: float
    target_reference_centroid_similarity: float | None
    best_competitor_reference_centroid_similarity: float | None
    reference_competitor_margin: float | None
    aggregation_weight: float
    input_score_fingerprint: str


@dataclass(frozen=True, slots=True)
class VisualInputFusionResult:
    """Raw fused evidence with an explicit strong-disagreement review gate."""

    fusion_version: str
    source: str
    photo_id: str
    scoring_unit_id: str
    route: str
    target_accepted_taxon_key: str
    candidate_set_fingerprint: str
    model_fingerprint: str
    preprocessing_fingerprint: str
    target_classifier_fingerprint: str
    regional_classifier_fingerprint: str
    embedding_set_fingerprint: str
    policy_fingerprint: str
    policy_strategy: str
    target_classifier_decision_score: float
    candidate_scores: tuple[AggregatedFullFrameCandidateScore, ...]
    target_regional_classifier_decision_score: float
    best_competitor_accepted_taxon_key: str
    best_competitor_scientific_name: str
    best_competitor_regional_classifier_decision_score: float
    regional_competitor_margin: float
    target_reference_centroid_similarity: float | None
    best_competitor_reference_centroid_similarity: float | None
    reference_competitor_margin: float | None
    disagreement_version: str
    visual_input_disagreement: float
    strong_visual_input_disagreement: bool
    routing_action: str
    review_reason: str | None
    visual_input_evidence: tuple[FullFrameVisualInputEvidence, ...]
    unavailable_variants: tuple[UnavailableAttentionVariant, ...]
    fusion_fingerprint: str


def embed_full_frame_visual_inputs(
    attention_result: FullFrameAttentionResult,
    *,
    encoder: FullFrameImageEncoder,
    model_fingerprint: str,
    preprocessing_fingerprint: str,
    raw_embedding: RawFullFrameEmbedding | None = None,
) -> EmbeddedFullFrameVisualInputs:
    """Embed every available attention input, reusing a cached raw embedding."""

    variants, evidence, unavailable, route = _validated_attention_result(
        attention_result
    )
    model = _sha256(model_fingerprint, field="model_fingerprint")
    preprocessing = _sha256(
        preprocessing_fingerprint,
        field="preprocessing_fingerprint",
    )
    _validate_encoder(
        encoder,
        preprocessing_fingerprint=preprocessing,
    )
    raw_variant = next(
        item for item in variants if item.visual_input_kind == RAW_FULL_IMAGE_KIND
    )
    reused_raw = None
    variants_to_encode = variants
    if raw_embedding is not None:
        reused_raw = _embedding_from_cached_raw(
            raw_embedding,
            raw_variant=raw_variant,
            model_fingerprint=model,
            preprocessing_fingerprint=preprocessing,
        )
        variants_to_encode = tuple(
            item for item in variants if item.visual_input_kind != RAW_FULL_IMAGE_KIND
        )

    raw_vectors = tuple(
        encoder.encode_images(tuple(item.image for item in variants_to_encode))
    )
    if len(raw_vectors) != len(variants_to_encode):
        raise ValueError(
            "full-frame encoder returned "
            f"{len(raw_vectors)} vectors for {len(variants_to_encode)} visual inputs"
        )
    encoded = tuple(
        _embedding_from_vector(
            variant,
            raw_vector,
            model_fingerprint=model,
            preprocessing_fingerprint=preprocessing,
        )
        for variant, raw_vector in zip(variants_to_encode, raw_vectors, strict=True)
    )
    embeddings = tuple(
        sorted(
            ((*encoded, reused_raw) if reused_raw is not None else encoded),
            key=_embedding_sort_key,
        )
    )
    dimensions = {item.embedding_dimension for item in embeddings}
    if len(dimensions) != 1:
        raise ValueError("full-frame encoder returned mixed embedding dimensions")
    values: dict[str, object] = {
        "embedding_set_version": VISUAL_INPUT_EMBEDDING_SET_VERSION,
        "visual_input_version": FULL_FRAME_VISUAL_INPUT_VERSION,
        "raw_image_content_hash": raw_variant.raw_image_content_hash,
        "route": route,
        "model_fingerprint": model,
        "preprocessing_fingerprint": preprocessing,
        "embeddings": embeddings,
        "attention_evidence": evidence,
        "unavailable_variants": unavailable,
    }
    fingerprint = canonical_semantic_fingerprint(_embedding_set_semantics(values))
    return EmbeddedFullFrameVisualInputs(
        **values,
        embedding_set_fingerprint=fingerprint,
    )


def build_full_frame_visual_input_score(
    *,
    embedding: FullFrameVisualInputEmbedding,
    route: str,
    target_accepted_taxon_key: str,
    candidate_set_fingerprint: str,
    target_classifier_fingerprint: str,
    regional_classifier_fingerprint: str,
    target_classifier_decision_score: float,
    candidate_scores: Sequence[FullFrameCandidateScore],
    source_fusion_fingerprint: str,
) -> FullFrameVisualInputScore:
    """Bind complete raw species scores to one immutable visual input."""

    _validate_embedding(embedding)
    route_value = _required_choice(
        route,
        field="route",
        allowed=REFERENCE_ROUTES,
    )
    target_key = _required_text(
        target_accepted_taxon_key,
        field="target_accepted_taxon_key",
    )
    candidates = _validated_candidate_scores(candidate_scores, target_key=target_key)
    candidate_set = _sha256(
        candidate_set_fingerprint,
        field="candidate_set_fingerprint",
    )
    target_classifier = _sha256(
        target_classifier_fingerprint,
        field="target_classifier_fingerprint",
    )
    regional_classifier = _sha256(
        regional_classifier_fingerprint,
        field="regional_classifier_fingerprint",
    )
    source_fusion = _sha256(
        source_fusion_fingerprint,
        field="source_fusion_fingerprint",
    )
    target_score = next(
        item for item in candidates if item.accepted_taxon_key == target_key
    )
    best_competitor = _best_regional_competitor(candidates, target_key=target_key)
    reference_margin = _optional_margin(
        target_score.reference_centroid_similarity,
        best_competitor.reference_centroid_similarity,
    )
    score_contract = _score_contract_fingerprint(
        route=route_value,
        target_key=target_key,
        candidate_set_fingerprint=candidate_set,
        model_fingerprint=embedding.model_fingerprint,
        preprocessing_fingerprint=embedding.preprocessing_fingerprint,
        target_classifier_fingerprint=target_classifier,
        regional_classifier_fingerprint=regional_classifier,
        candidate_keys=tuple(item.accepted_taxon_key for item in candidates),
    )
    values: dict[str, object] = {
        "score_version": VISUAL_INPUT_SCORE_VERSION,
        "visual_input_id": embedding.visual_input_id,
        "visual_input_kind": embedding.visual_input_kind,
        "transformation_fingerprint": embedding.transformation_fingerprint,
        "embedding_id": embedding.embedding_id,
        "embedding_fingerprint": embedding.embedding_fingerprint,
        "model_fingerprint": embedding.model_fingerprint,
        "preprocessing_fingerprint": embedding.preprocessing_fingerprint,
        "route": route_value,
        "target_accepted_taxon_key": target_key,
        "candidate_set_fingerprint": candidate_set,
        "target_classifier_fingerprint": target_classifier,
        "regional_classifier_fingerprint": regional_classifier,
        "target_classifier_decision_score": _finite_float(
            target_classifier_decision_score,
            field="target_classifier_decision_score",
        ),
        "candidate_scores": candidates,
        "target_regional_classifier_decision_score": (
            target_score.regional_classifier_decision_score
        ),
        "best_competitor_accepted_taxon_key": (best_competitor.accepted_taxon_key),
        "best_competitor_scientific_name": best_competitor.scientific_name,
        "best_competitor_regional_classifier_decision_score": (
            best_competitor.regional_classifier_decision_score
        ),
        "regional_competitor_margin": (
            target_score.regional_classifier_decision_score
            - best_competitor.regional_classifier_decision_score
        ),
        "target_reference_centroid_similarity": (
            target_score.reference_centroid_similarity
        ),
        "best_competitor_reference_centroid_similarity": (
            best_competitor.reference_centroid_similarity
        ),
        "reference_competitor_margin": reference_margin,
        "source_fusion_fingerprint": source_fusion,
        "score_contract_fingerprint": score_contract,
    }
    fingerprint = canonical_semantic_fingerprint(_input_score_semantics(values))
    return FullFrameVisualInputScore(**values, score_fingerprint=fingerprint)


def build_visual_input_fusion_policy(
    *,
    strategy: str,
    route: str,
    visual_input_kind_weights: Mapping[str, float],
    strong_disagreement_threshold: float,
    model_fingerprint: str,
    preprocessing_fingerprint: str,
    target_classifier_fingerprint: str,
    regional_classifier_fingerprint: str,
    split_fingerprint: str,
    training_data_fingerprint: str,
    fit_partition: str,
) -> VisualInputFusionPolicy:
    """Bind linear pooling and review threshold to model-selection evidence."""

    strategy_value = _required_choice(
        strategy,
        field="strategy",
        allowed=VISUAL_INPUT_FUSION_STRATEGIES,
    )
    route_value = _required_choice(
        route,
        field="route",
        allowed=REFERENCE_ROUTES,
    )
    partition = _required_text(fit_partition, field="fit_partition")
    if partition != "model_selection":
        raise ValueError(
            "visual-input fusion weights and disagreement threshold must use "
            "model_selection data only"
        )
    if set(visual_input_kind_weights) != set(_VISUAL_INPUT_KINDS):
        raise ValueError("visual-input fusion weight kind set is incomplete")
    raw_weights = {
        kind: _nonnegative_float(
            visual_input_kind_weights[kind],
            field=f"visual_input_kind_weights[{kind}]",
        )
        for kind in _VISUAL_INPUT_KINDS
    }
    total = sum(raw_weights.values())
    if total <= 0.0:
        raise ValueError("visual-input fusion weights must have positive total")
    normalized = tuple(
        (kind, raw_weights[kind] / total) for kind in _VISUAL_INPUT_KINDS
    )
    if dict(normalized)[RAW_FULL_IMAGE_KIND] <= 0.0:
        raise ValueError("visual-input fusion must retain positive raw-image weight")
    if strategy_value == RAW_ONLY_VISUAL_INPUT_FUSION and dict(normalized) != {
        RAW_FULL_IMAGE_KIND: 1.0,
        FOCUSED_FULL_FRAME_KIND: 0.0,
        MASKED_FULL_FRAME_KIND: 0.0,
        MULTI_OBJECT_FULL_FRAME_KIND: 0.0,
    }:
        raise ValueError("raw_only fusion requires only raw-image weight")
    threshold = _finite_float(
        strong_disagreement_threshold,
        field="strong_disagreement_threshold",
    )
    if threshold <= 0.0:
        raise ValueError("strong_disagreement_threshold must be greater than zero")
    values: dict[str, object] = {
        "schema_version": VISUAL_INPUT_FUSION_POLICY_SCHEMA_VERSION,
        "policy_version": VISUAL_INPUT_FUSION_POLICY_VERSION,
        "strategy": strategy_value,
        "route": route_value,
        "visual_input_version": FULL_FRAME_VISUAL_INPUT_VERSION,
        "score_version": VISUAL_INPUT_SCORE_VERSION,
        "within_kind_reduction": VISUAL_INPUT_WITHIN_KIND_REDUCTION,
        "visual_input_kind_weights": normalized,
        "disagreement_version": VISUAL_INPUT_DISAGREEMENT_VERSION,
        "strong_disagreement_threshold": threshold,
        "model_fingerprint": _sha256(
            model_fingerprint,
            field="model_fingerprint",
        ),
        "preprocessing_fingerprint": _sha256(
            preprocessing_fingerprint,
            field="preprocessing_fingerprint",
        ),
        "target_classifier_fingerprint": _sha256(
            target_classifier_fingerprint,
            field="target_classifier_fingerprint",
        ),
        "regional_classifier_fingerprint": _sha256(
            regional_classifier_fingerprint,
            field="regional_classifier_fingerprint",
        ),
        "split_fingerprint": _sha256(
            split_fingerprint,
            field="split_fingerprint",
        ),
        "training_data_fingerprint": _sha256(
            training_data_fingerprint,
            field="training_data_fingerprint",
        ),
        "fit_partition": partition,
    }
    fingerprint = canonical_semantic_fingerprint(_policy_semantics(values))
    return VisualInputFusionPolicy(**values, policy_fingerprint=fingerprint)


def fuse_full_frame_visual_evidence(
    *,
    source: str,
    photo_id: str,
    scoring_unit_id: str,
    embedded_inputs: EmbeddedFullFrameVisualInputs,
    visual_input_scores: Sequence[FullFrameVisualInputScore],
    policy: VisualInputFusionPolicy,
) -> VisualInputFusionResult:
    """Linearly pool raw scores while retaining every input and disagreement."""

    _validate_embedded_inputs(embedded_inputs)
    _validate_policy(policy)
    scores = tuple(visual_input_scores)
    if any(not isinstance(item, FullFrameVisualInputScore) for item in scores):
        raise TypeError(
            "visual_input_scores must contain FullFrameVisualInputScore values"
        )
    score_by_id: dict[str, FullFrameVisualInputScore] = {}
    for item in scores:
        if item.visual_input_id in score_by_id:
            raise ValueError("visual_input_scores contain duplicate visual_input_id")
        score_by_id[item.visual_input_id] = item
    embedding_by_id = {
        item.visual_input_id: item for item in embedded_inputs.embeddings
    }
    if set(score_by_id) != set(embedding_by_id):
        raise ValueError(
            "visual-input score coverage must exactly match available embeddings"
        )
    ordered_scores = tuple(
        score_by_id[item.visual_input_id]
        for item in sorted(embedding_by_id.values(), key=_embedding_sort_key)
    )
    first = ordered_scores[0]
    for item in ordered_scores:
        if item.candidate_set_fingerprint != first.candidate_set_fingerprint:
            raise ValueError("visual-input candidate-set identity differs")
        if item.target_accepted_taxon_key != first.target_accepted_taxon_key:
            raise ValueError("visual-input target identity differs")
        if item.route != first.route:
            raise ValueError("visual-input route identity differs")
        if item.target_classifier_fingerprint != first.target_classifier_fingerprint:
            raise ValueError("visual-input target-classifier identity differs")
        if (
            item.regional_classifier_fingerprint
            != first.regional_classifier_fingerprint
        ):
            raise ValueError("visual-input regional-classifier identity differs")
    if first.route != embedded_inputs.route or first.route != policy.route:
        raise ValueError("visual-input fusion route identity is incompatible")
    for field_name in (
        "model_fingerprint",
        "preprocessing_fingerprint",
        "target_classifier_fingerprint",
        "regional_classifier_fingerprint",
    ):
        if getattr(first, field_name) != getattr(policy, field_name):
            raise ValueError(f"visual-input fusion {field_name} is incompatible")
    if first.model_fingerprint != embedded_inputs.model_fingerprint:
        raise ValueError("visual-input embedding model identity is incompatible")
    if first.preprocessing_fingerprint != embedded_inputs.preprocessing_fingerprint:
        raise ValueError(
            "visual-input embedding preprocessing identity is incompatible"
        )
    for item in ordered_scores:
        _validate_input_score(item, embedding=embedding_by_id[item.visual_input_id])
    candidate_keys = tuple(
        candidate.accepted_taxon_key for candidate in first.candidate_scores
    )
    candidate_names = {
        candidate.accepted_taxon_key: candidate.scientific_name
        for candidate in first.candidate_scores
    }
    for item in ordered_scores[1:]:
        if (
            tuple(candidate.accepted_taxon_key for candidate in item.candidate_scores)
            != candidate_keys
        ):
            raise ValueError("visual-input candidate score coverage differs")
        if {
            candidate.accepted_taxon_key: candidate.scientific_name
            for candidate in item.candidate_scores
        } != candidate_names:
            raise ValueError("visual-input candidate scientific names differ")

    effective_weights = _effective_input_weights(ordered_scores, policy=policy)
    scores_by_candidate = {
        item.accepted_taxon_key: item for item in first.candidate_scores
    }
    aggregated_candidates = []
    for candidate_key in candidate_keys:
        rows = tuple(
            next(
                candidate
                for candidate in item.candidate_scores
                if candidate.accepted_taxon_key == candidate_key
            )
            for item in ordered_scores
        )
        aggregated_candidates.append(
            AggregatedFullFrameCandidateScore(
                accepted_taxon_key=candidate_key,
                scientific_name=scores_by_candidate[candidate_key].scientific_name,
                regional_classifier_decision_score=sum(
                    row.regional_classifier_decision_score
                    * effective_weights[item.visual_input_id]
                    for row, item in zip(rows, ordered_scores, strict=True)
                ),
                reference_centroid_similarity=_weighted_optional_score(
                    tuple(row.reference_centroid_similarity for row in rows),
                    tuple(
                        effective_weights[item.visual_input_id]
                        for item in ordered_scores
                    ),
                ),
            )
        )
    aggregated = tuple(aggregated_candidates)
    target = next(
        item
        for item in aggregated
        if item.accepted_taxon_key == first.target_accepted_taxon_key
    )
    best_competitor = _best_aggregated_regional_competitor(
        aggregated,
        target_key=first.target_accepted_taxon_key,
    )
    target_reference = target.reference_centroid_similarity
    competitor_reference = best_competitor.reference_centroid_similarity
    margins = tuple(item.regional_competitor_margin for item in ordered_scores)
    disagreement = max(margins) - min(margins)
    strong_disagreement = disagreement >= policy.strong_disagreement_threshold
    input_evidence = tuple(
        FullFrameVisualInputEvidence(
            visual_input_id=item.visual_input_id,
            visual_input_kind=item.visual_input_kind,
            transformation_fingerprint=item.transformation_fingerprint,
            embedding_id=item.embedding_id,
            embedding_fingerprint=item.embedding_fingerprint,
            source_fusion_fingerprint=item.source_fusion_fingerprint,
            target_classifier_decision_score=(item.target_classifier_decision_score),
            target_regional_classifier_decision_score=(
                item.target_regional_classifier_decision_score
            ),
            best_competitor_accepted_taxon_key=(
                item.best_competitor_accepted_taxon_key
            ),
            best_competitor_scientific_name=(item.best_competitor_scientific_name),
            best_competitor_regional_classifier_decision_score=(
                item.best_competitor_regional_classifier_decision_score
            ),
            regional_competitor_margin=item.regional_competitor_margin,
            target_reference_centroid_similarity=(
                item.target_reference_centroid_similarity
            ),
            best_competitor_reference_centroid_similarity=(
                item.best_competitor_reference_centroid_similarity
            ),
            reference_competitor_margin=item.reference_competitor_margin,
            aggregation_weight=effective_weights[item.visual_input_id],
            input_score_fingerprint=item.score_fingerprint,
        )
        for item in ordered_scores
    )
    values: dict[str, object] = {
        "fusion_version": VISUAL_INPUT_FUSION_RESULT_VERSION,
        "source": _required_text(source, field="source"),
        "photo_id": _required_text(photo_id, field="photo_id"),
        "scoring_unit_id": _sha256(scoring_unit_id, field="scoring_unit_id"),
        "route": first.route,
        "target_accepted_taxon_key": first.target_accepted_taxon_key,
        "candidate_set_fingerprint": first.candidate_set_fingerprint,
        "model_fingerprint": first.model_fingerprint,
        "preprocessing_fingerprint": first.preprocessing_fingerprint,
        "target_classifier_fingerprint": first.target_classifier_fingerprint,
        "regional_classifier_fingerprint": first.regional_classifier_fingerprint,
        "embedding_set_fingerprint": embedded_inputs.embedding_set_fingerprint,
        "policy_fingerprint": policy.policy_fingerprint,
        "policy_strategy": policy.strategy,
        "target_classifier_decision_score": sum(
            item.target_classifier_decision_score
            * effective_weights[item.visual_input_id]
            for item in ordered_scores
        ),
        "candidate_scores": aggregated,
        "target_regional_classifier_decision_score": (
            target.regional_classifier_decision_score
        ),
        "best_competitor_accepted_taxon_key": (best_competitor.accepted_taxon_key),
        "best_competitor_scientific_name": best_competitor.scientific_name,
        "best_competitor_regional_classifier_decision_score": (
            best_competitor.regional_classifier_decision_score
        ),
        "regional_competitor_margin": (
            target.regional_classifier_decision_score
            - best_competitor.regional_classifier_decision_score
        ),
        "target_reference_centroid_similarity": target_reference,
        "best_competitor_reference_centroid_similarity": competitor_reference,
        "reference_competitor_margin": _optional_margin(
            target_reference,
            competitor_reference,
        ),
        "disagreement_version": VISUAL_INPUT_DISAGREEMENT_VERSION,
        "visual_input_disagreement": disagreement,
        "strong_visual_input_disagreement": strong_disagreement,
        "routing_action": "review" if strong_disagreement else "score",
        "review_reason": (
            STRONG_VISUAL_INPUT_DISAGREEMENT if strong_disagreement else None
        ),
        "visual_input_evidence": input_evidence,
        "unavailable_variants": embedded_inputs.unavailable_variants,
    }
    fingerprint = canonical_semantic_fingerprint(_fusion_result_semantics(values))
    return VisualInputFusionResult(**values, fusion_fingerprint=fingerprint)


def visual_input_fusion_result_payload(
    result: VisualInputFusionResult,
) -> dict[str, object]:
    """Validate and return a canonical JSON-like fusion result payload."""

    if not isinstance(result, VisualInputFusionResult):
        raise TypeError("result must be a VisualInputFusionResult")
    values = {
        name: getattr(result, name)
        for name in VisualInputFusionResult.__dataclass_fields__
        if name != "fusion_fingerprint"
    }
    fingerprint = _sha256(
        result.fusion_fingerprint,
        field="fusion_fingerprint",
    )
    semantics = _fusion_result_semantics(values)
    if canonical_semantic_fingerprint(semantics) != fingerprint:
        raise ValueError("visual-input fusion result fingerprint is inconsistent")
    return {**semantics, "fusion_fingerprint": fingerprint}


def _validated_attention_result(
    result: FullFrameAttentionResult,
) -> tuple[
    tuple[FullFrameAttentionVariant, ...],
    tuple[FullFrameVariantEvidence, ...],
    tuple[UnavailableAttentionVariant, ...],
    str,
]:
    if not isinstance(result, FullFrameAttentionResult):
        raise TypeError("attention_result must be a FullFrameAttentionResult")
    variants = tuple(result.variants)
    if not variants:
        raise ValueError("full-frame attention result has no available variants")
    if any(not isinstance(item, FullFrameAttentionVariant) for item in variants):
        raise TypeError("attention variants have an incompatible type")
    ids = [item.visual_input_id for item in variants]
    if len(ids) != len(set(ids)):
        raise ValueError("attention result contains duplicate visual_input_id")
    raw_variants = tuple(
        item for item in variants if item.visual_input_kind == RAW_FULL_IMAGE_KIND
    )
    if len(raw_variants) != 1:
        raise ValueError("attention result must contain exactly one raw full image")
    raw = raw_variants[0]
    expected_raw = raw_full_frame_visual_input(raw.image)
    if raw != expected_raw:
        raise ValueError("attention result contains a stale raw visual-input identity")
    for variant in variants:
        _validate_attention_variant(variant, raw=raw)
    evidence = tuple(result.evidence)
    if not evidence or any(
        not isinstance(item, FullFrameVariantEvidence) for item in evidence
    ):
        raise ValueError("attention result requires typed variant evidence")
    known = {item.visual_input_id: item for item in variants}
    evidence_ids = set()
    routes = set()
    for item in evidence:
        variant = known.get(item.visual_input_id)
        if variant is None:
            raise ValueError("attention evidence references an unknown visual input")
        if (
            item.visual_input_kind != variant.visual_input_kind
            or item.transformation_fingerprint != variant.transformation_fingerprint
        ):
            raise ValueError("attention evidence visual-input identity differs")
        _sha256(item.evidence_id, field="attention evidence_id")
        routes.add(
            _required_choice(item.route, field="route", allowed=REFERENCE_ROUTES)
        )
        evidence_ids.add(item.visual_input_id)
    if evidence_ids != set(known):
        raise ValueError("attention evidence coverage is incomplete")
    unavailable = tuple(result.unavailable_variants)
    if any(not isinstance(item, UnavailableAttentionVariant) for item in unavailable):
        raise TypeError("unavailable attention variants have an incompatible type")
    for item in unavailable:
        if item.visual_input_kind not in _VISUAL_INPUT_KINDS[1:]:
            raise ValueError("unavailable attention variant has an invalid kind")
        _sha256(item.unavailable_id, field="unavailable_id")
        routes.add(
            _required_choice(item.route, field="route", allowed=REFERENCE_ROUTES)
        )
    if len(routes) != 1:
        raise ValueError("attention result mixes comparison routes")
    return (
        tuple(sorted(variants, key=_variant_sort_key)),
        tuple(sorted(evidence, key=_evidence_sort_key)),
        tuple(sorted(unavailable, key=_unavailable_sort_key)),
        next(iter(routes)),
    )


def _validate_attention_variant(
    variant: FullFrameAttentionVariant,
    *,
    raw: FullFrameAttentionVariant,
) -> None:
    _sha256(variant.visual_input_id, field="visual_input_id")
    _sha256(variant.raw_image_content_hash, field="raw_image_content_hash")
    _sha256(variant.visual_content_hash, field="visual_content_hash")
    _sha256(
        variant.transformation_fingerprint,
        field="transformation_fingerprint",
    )
    if variant.visual_input_kind not in _VISUAL_INPUT_KINDS:
        raise ValueError("attention variant has an unsupported visual_input_kind")
    if variant.visual_input_version != FULL_FRAME_VISUAL_INPUT_VERSION:
        raise ValueError("attention variant visual-input version is incompatible")
    if variant.raw_image_content_hash != raw.raw_image_content_hash:
        raise ValueError("attention variants do not share one raw image")
    if (variant.width, variant.height, variant.mode) != (
        raw.width,
        raw.height,
        raw.mode,
    ):
        raise ValueError("attention variant changed the complete image canvas")
    if (
        variant.image.width,
        variant.image.height,
        variant.image.mode,
    ) != (variant.width, variant.height, variant.mode):
        raise ValueError("attention variant image metadata is inconsistent")
    if decoded_image_content_hash(variant.image) != variant.visual_content_hash:
        raise ValueError("attention variant visual content hash is stale")
    if variant.visual_input_kind == RAW_FULL_IMAGE_KIND:
        if variant.transformation_applied:
            raise ValueError("raw visual input cannot report a transformation")
        return
    if not variant.transformation_applied:
        raise ValueError("attention variant must report its transformation")
    expected_id = canonical_semantic_fingerprint(
        {
            "raw_image_content_hash": variant.raw_image_content_hash,
            "transformation_fingerprint": variant.transformation_fingerprint,
            "visual_content_hash": variant.visual_content_hash,
            "visual_input_kind": variant.visual_input_kind,
            "visual_input_version": variant.visual_input_version,
        }
    )
    if expected_id != variant.visual_input_id:
        raise ValueError("attention variant visual-input identity is stale")


def _validate_encoder(
    encoder: FullFrameImageEncoder,
    *,
    preprocessing_fingerprint: str,
) -> None:
    if (
        getattr(encoder, "image_resize_mode", None)
        != TARGET_FULL_FRAME_IMAGE_RESIZE_MODE
    ):
        raise ValueError(
            "full-frame visual-input encoder requires image_resize_mode "
            f"{TARGET_FULL_FRAME_IMAGE_RESIZE_MODE!r}"
        )
    if (
        getattr(encoder, "preprocessing_contract_fingerprint", None)
        != TARGET_FULL_FRAME_PREPROCESSING.fingerprint
    ):
        raise ValueError("full-frame visual-input preprocessing contract mismatch")
    if getattr(encoder, "preprocessing_fingerprint", None) != (
        preprocessing_fingerprint
    ):
        raise ValueError("full-frame visual-input preprocessing fingerprint mismatch")


def _embedding_from_cached_raw(
    value: RawFullFrameEmbedding,
    *,
    raw_variant: FullFrameAttentionVariant,
    model_fingerprint: str,
    preprocessing_fingerprint: str,
) -> FullFrameVisualInputEmbedding:
    if not isinstance(value, RawFullFrameEmbedding):
        raise TypeError("raw_embedding must be a RawFullFrameEmbedding")
    if (
        value.visual_input_id != raw_variant.visual_input_id
        or value.visual_input_kind != RAW_FULL_IMAGE_KIND
        or value.raw_image_content_hash != raw_variant.raw_image_content_hash
        or value.transformation_fingerprint != raw_variant.transformation_fingerprint
    ):
        raise ValueError("cached raw embedding visual-input identity is incompatible")
    if (
        value.model_fingerprint != model_fingerprint
        or value.preprocessing_fingerprint != preprocessing_fingerprint
        or value.preprocessing_contract_fingerprint
        != TARGET_FULL_FRAME_PREPROCESSING.fingerprint
        or value.image_resize_mode != TARGET_FULL_FRAME_IMAGE_RESIZE_MODE
    ):
        raise ValueError("cached raw embedding model or preprocessing identity differs")
    if value.embedding_version != TARGET_FULL_FRAME_EMBEDDING_VERSION:
        raise ValueError("cached raw embedding version is incompatible")
    embedding = FullFrameVisualInputEmbedding(
        embedding_id=value.embedding_id,
        embedding_version=value.embedding_version,
        embedding_fingerprint=value.embedding_fingerprint,
        visual_input_id=value.visual_input_id,
        visual_input_kind=value.visual_input_kind,
        visual_input_version=raw_variant.visual_input_version,
        raw_image_content_hash=value.raw_image_content_hash,
        visual_content_hash=raw_variant.visual_content_hash,
        transformation_fingerprint=value.transformation_fingerprint,
        model_fingerprint=value.model_fingerprint,
        image_resize_mode=value.image_resize_mode,
        preprocessing_contract_fingerprint=(value.preprocessing_contract_fingerprint),
        preprocessing_fingerprint=value.preprocessing_fingerprint,
        embedding_dimension=value.embedding_dimension,
        embedding=tuple(value.embedding),
        embedding_norm=value.embedding_norm,
    )
    _validate_embedding(embedding)
    return embedding


def _embedding_from_vector(
    variant: FullFrameAttentionVariant,
    raw_vector: object,
    *,
    model_fingerprint: str,
    preprocessing_fingerprint: str,
) -> FullFrameVisualInputEmbedding:
    vector = _validated_vector(raw_vector, field="full-frame embedding")
    embedding_id = full_frame_embedding_id(
        visual_input_id=variant.visual_input_id,
        model_fingerprint=model_fingerprint,
        preprocessing_fingerprint=preprocessing_fingerprint,
    )
    embedding_fingerprint = canonical_semantic_fingerprint(
        {
            "embedding": vector,
            "embedding_id": embedding_id,
            "embedding_version": TARGET_FULL_FRAME_EMBEDDING_VERSION,
        }
    )
    return FullFrameVisualInputEmbedding(
        embedding_id=embedding_id,
        embedding_version=TARGET_FULL_FRAME_EMBEDDING_VERSION,
        embedding_fingerprint=embedding_fingerprint,
        visual_input_id=variant.visual_input_id,
        visual_input_kind=variant.visual_input_kind,
        visual_input_version=variant.visual_input_version,
        raw_image_content_hash=variant.raw_image_content_hash,
        visual_content_hash=variant.visual_content_hash,
        transformation_fingerprint=variant.transformation_fingerprint,
        model_fingerprint=model_fingerprint,
        image_resize_mode=TARGET_FULL_FRAME_IMAGE_RESIZE_MODE,
        preprocessing_contract_fingerprint=TARGET_FULL_FRAME_PREPROCESSING.fingerprint,
        preprocessing_fingerprint=preprocessing_fingerprint,
        embedding_dimension=len(vector),
        embedding=vector,
        embedding_norm=hypot(*vector),
    )


def _validate_embedded_inputs(value: EmbeddedFullFrameVisualInputs) -> None:
    if not isinstance(value, EmbeddedFullFrameVisualInputs):
        raise TypeError("embedded_inputs must be EmbeddedFullFrameVisualInputs")
    if value.embedding_set_version != VISUAL_INPUT_EMBEDDING_SET_VERSION:
        raise ValueError("full-frame embedding-set version is incompatible")
    if value.visual_input_version != FULL_FRAME_VISUAL_INPUT_VERSION:
        raise ValueError("full-frame visual-input version is incompatible")
    raw_image_content_hash = _sha256(
        value.raw_image_content_hash,
        field="raw_image_content_hash",
    )
    route = _required_choice(value.route, field="route", allowed=REFERENCE_ROUTES)
    model = _sha256(value.model_fingerprint, field="model_fingerprint")
    preprocessing = _sha256(
        value.preprocessing_fingerprint,
        field="preprocessing_fingerprint",
    )
    embeddings = tuple(value.embeddings)
    if not embeddings:
        raise ValueError("embedded inputs must not be empty")
    if any(not isinstance(item, FullFrameVisualInputEmbedding) for item in embeddings):
        raise TypeError("embedded inputs contain an incompatible embedding type")
    ids = [item.visual_input_id for item in embeddings]
    if len(ids) != len(set(ids)):
        raise ValueError("embedded inputs contain duplicate visual_input_id")
    if sum(item.visual_input_kind == RAW_FULL_IMAGE_KIND for item in embeddings) != 1:
        raise ValueError("embedded inputs require exactly one raw full image")
    if any(
        item.raw_image_content_hash != raw_image_content_hash for item in embeddings
    ):
        raise ValueError("embedded inputs do not share one raw image")
    for item in embeddings:
        _validate_embedding(item)
        if (
            item.model_fingerprint != model
            or item.preprocessing_fingerprint != preprocessing
        ):
            raise ValueError("embedded input model or preprocessing identity differs")
    if len({item.embedding_dimension for item in embeddings}) != 1:
        raise ValueError("embedded inputs contain mixed embedding dimensions")

    known = {item.visual_input_id: item for item in embeddings}
    evidence = tuple(value.attention_evidence)
    if not evidence or any(
        not isinstance(item, FullFrameVariantEvidence) for item in evidence
    ):
        raise ValueError("embedded inputs require typed attention evidence")
    evidence_coverage = set()
    for item in evidence:
        embedding = known.get(item.visual_input_id)
        if embedding is None:
            raise ValueError("attention evidence references an unknown embedding")
        if (
            item.visual_input_kind != embedding.visual_input_kind
            or item.transformation_fingerprint != embedding.transformation_fingerprint
            or item.route != route
        ):
            raise ValueError("attention evidence embedding identity differs")
        _sha256(item.evidence_id, field="attention evidence_id")
        evidence_coverage.add(item.visual_input_id)
    if evidence_coverage != set(known):
        raise ValueError("attention evidence coverage is incomplete")

    unavailable = tuple(value.unavailable_variants)
    if any(not isinstance(item, UnavailableAttentionVariant) for item in unavailable):
        raise TypeError("unavailable attention variants have an incompatible type")
    for item in unavailable:
        if item.visual_input_kind not in _VISUAL_INPUT_KINDS[1:]:
            raise ValueError("unavailable attention variant has an invalid kind")
        if item.route != route:
            raise ValueError("unavailable attention variant route differs")
        _sha256(item.unavailable_id, field="unavailable_id")
    fingerprint = _sha256(
        value.embedding_set_fingerprint,
        field="embedding_set_fingerprint",
    )
    values = {
        name: getattr(value, name)
        for name in EmbeddedFullFrameVisualInputs.__dataclass_fields__
        if name != "embedding_set_fingerprint"
    }
    if canonical_semantic_fingerprint(_embedding_set_semantics(values)) != fingerprint:
        raise ValueError("full-frame embedding-set fingerprint is inconsistent")


def _validate_embedding(value: FullFrameVisualInputEmbedding) -> None:
    if not isinstance(value, FullFrameVisualInputEmbedding):
        raise TypeError("embedding must be a FullFrameVisualInputEmbedding")
    if value.embedding_version != TARGET_FULL_FRAME_EMBEDDING_VERSION:
        raise ValueError("full-frame embedding version is incompatible")
    if value.visual_input_version != FULL_FRAME_VISUAL_INPUT_VERSION:
        raise ValueError("full-frame visual-input version is incompatible")
    if value.visual_input_kind not in _VISUAL_INPUT_KINDS:
        raise ValueError("full-frame embedding visual-input kind is unsupported")
    _sha256(value.visual_input_id, field="visual_input_id")
    _sha256(value.raw_image_content_hash, field="raw_image_content_hash")
    _sha256(value.visual_content_hash, field="visual_content_hash")
    _sha256(
        value.transformation_fingerprint,
        field="transformation_fingerprint",
    )
    model = _sha256(value.model_fingerprint, field="model_fingerprint")
    preprocessing = _sha256(
        value.preprocessing_fingerprint,
        field="preprocessing_fingerprint",
    )
    expected_id = full_frame_embedding_id(
        visual_input_id=value.visual_input_id,
        model_fingerprint=model,
        preprocessing_fingerprint=preprocessing,
    )
    if value.embedding_id != expected_id:
        raise ValueError("full-frame embedding ID is inconsistent")
    vector = _validated_vector(value.embedding, field="embedding")
    dimension = _positive_integer(
        value.embedding_dimension,
        field="embedding_dimension",
    )
    if dimension != len(vector):
        raise ValueError("full-frame embedding dimension is inconsistent")
    norm = hypot(*vector)
    stored_norm = _finite_float(value.embedding_norm, field="embedding_norm")
    if stored_norm <= 0.0 or not isclose(
        stored_norm,
        norm,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("full-frame embedding norm is inconsistent")
    expected_fingerprint = canonical_semantic_fingerprint(
        {
            "embedding": vector,
            "embedding_id": value.embedding_id,
            "embedding_version": value.embedding_version,
        }
    )
    if value.embedding_fingerprint != expected_fingerprint:
        raise ValueError("full-frame embedding fingerprint is inconsistent")
    if (
        value.image_resize_mode != TARGET_FULL_FRAME_IMAGE_RESIZE_MODE
        or value.preprocessing_contract_fingerprint
        != TARGET_FULL_FRAME_PREPROCESSING.fingerprint
    ):
        raise ValueError("full-frame embedding preprocessing contract differs")


def _validate_policy(value: VisualInputFusionPolicy) -> None:
    if not isinstance(value, VisualInputFusionPolicy):
        raise TypeError("policy must be a VisualInputFusionPolicy")
    if value.schema_version != VISUAL_INPUT_FUSION_POLICY_SCHEMA_VERSION:
        raise ValueError("visual-input fusion policy schema is incompatible")
    if value.policy_version != VISUAL_INPUT_FUSION_POLICY_VERSION:
        raise ValueError("visual-input fusion policy version is incompatible")
    strategy = _required_choice(
        value.strategy,
        field="strategy",
        allowed=VISUAL_INPUT_FUSION_STRATEGIES,
    )
    _required_choice(value.route, field="route", allowed=REFERENCE_ROUTES)
    if value.visual_input_version != FULL_FRAME_VISUAL_INPUT_VERSION:
        raise ValueError("visual-input fusion policy input version is incompatible")
    if value.score_version != VISUAL_INPUT_SCORE_VERSION:
        raise ValueError("visual-input fusion policy score version is incompatible")
    if value.within_kind_reduction != VISUAL_INPUT_WITHIN_KIND_REDUCTION:
        raise ValueError("visual-input within-kind reduction is incompatible")
    if value.disagreement_version != VISUAL_INPUT_DISAGREEMENT_VERSION:
        raise ValueError("visual-input disagreement policy is incompatible")
    weights = tuple(value.visual_input_kind_weights)
    if tuple(kind for kind, _weight in weights) != _VISUAL_INPUT_KINDS:
        raise ValueError("visual-input fusion weight kind order is incompatible")
    normalized_weights = tuple(
        (kind, _nonnegative_float(weight, field=f"weight[{kind}]"))
        for kind, weight in weights
    )
    if not isclose(
        sum(weight for _kind, weight in normalized_weights),
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("visual-input fusion policy weights must sum to one")
    if dict(normalized_weights)[RAW_FULL_IMAGE_KIND] <= 0.0:
        raise ValueError("visual-input fusion policy must retain the raw image")
    if strategy == RAW_ONLY_VISUAL_INPUT_FUSION and dict(normalized_weights) != {
        RAW_FULL_IMAGE_KIND: 1.0,
        FOCUSED_FULL_FRAME_KIND: 0.0,
        MASKED_FULL_FRAME_KIND: 0.0,
        MULTI_OBJECT_FULL_FRAME_KIND: 0.0,
    }:
        raise ValueError("raw_only visual-input policy has incompatible weights")
    threshold = _finite_float(
        value.strong_disagreement_threshold,
        field="strong_disagreement_threshold",
    )
    if threshold <= 0.0:
        raise ValueError("strong_disagreement_threshold must be greater than zero")
    for field_name in (
        "model_fingerprint",
        "preprocessing_fingerprint",
        "target_classifier_fingerprint",
        "regional_classifier_fingerprint",
        "split_fingerprint",
        "training_data_fingerprint",
    ):
        _sha256(getattr(value, field_name), field=field_name)
    if value.fit_partition != "model_selection":
        raise ValueError("visual-input fusion policy is not validation-bound")
    values = {
        name: getattr(value, name)
        for name in VisualInputFusionPolicy.__dataclass_fields__
        if name != "policy_fingerprint"
    }
    fingerprint = _sha256(value.policy_fingerprint, field="policy_fingerprint")
    if canonical_semantic_fingerprint(_policy_semantics(values)) != fingerprint:
        raise ValueError("visual-input fusion policy fingerprint is inconsistent")


def _validate_input_score(
    value: FullFrameVisualInputScore,
    *,
    embedding: FullFrameVisualInputEmbedding,
) -> None:
    if (
        value.visual_input_id != embedding.visual_input_id
        or value.visual_input_kind != embedding.visual_input_kind
        or value.transformation_fingerprint != embedding.transformation_fingerprint
        or value.embedding_id != embedding.embedding_id
        or value.embedding_fingerprint != embedding.embedding_fingerprint
    ):
        raise ValueError("visual-input score embedding identity differs")
    if value.score_version != VISUAL_INPUT_SCORE_VERSION:
        raise ValueError("visual-input score version is incompatible")
    _required_choice(value.route, field="route", allowed=REFERENCE_ROUTES)
    target_key = _required_text(
        value.target_accepted_taxon_key,
        field="target_accepted_taxon_key",
    )
    for field_name in (
        "candidate_set_fingerprint",
        "target_classifier_fingerprint",
        "regional_classifier_fingerprint",
        "source_fusion_fingerprint",
    ):
        _sha256(getattr(value, field_name), field=field_name)
    if (
        value.model_fingerprint != embedding.model_fingerprint
        or value.preprocessing_fingerprint != embedding.preprocessing_fingerprint
    ):
        raise ValueError("visual-input score model or preprocessing identity differs")
    candidates = _validated_candidate_scores(
        value.candidate_scores,
        target_key=target_key,
    )
    if candidates != value.candidate_scores:
        raise ValueError("visual-input candidate scores are not canonically ordered")
    target = next(item for item in candidates if item.accepted_taxon_key == target_key)
    competitor = _best_regional_competitor(candidates, target_key=target_key)
    expected_derived = {
        "target_regional_classifier_decision_score": (
            target.regional_classifier_decision_score
        ),
        "best_competitor_accepted_taxon_key": competitor.accepted_taxon_key,
        "best_competitor_scientific_name": competitor.scientific_name,
        "best_competitor_regional_classifier_decision_score": (
            competitor.regional_classifier_decision_score
        ),
        "regional_competitor_margin": (
            target.regional_classifier_decision_score
            - competitor.regional_classifier_decision_score
        ),
        "target_reference_centroid_similarity": (target.reference_centroid_similarity),
        "best_competitor_reference_centroid_similarity": (
            competitor.reference_centroid_similarity
        ),
        "reference_competitor_margin": _optional_margin(
            target.reference_centroid_similarity,
            competitor.reference_centroid_similarity,
        ),
    }
    if any(
        getattr(value, name) != expected for name, expected in expected_derived.items()
    ):
        raise ValueError("visual-input score derived competitor evidence differs")
    _finite_float(
        value.target_classifier_decision_score,
        field="target_classifier_decision_score",
    )
    expected_contract = _score_contract_fingerprint(
        route=value.route,
        target_key=target_key,
        candidate_set_fingerprint=value.candidate_set_fingerprint,
        model_fingerprint=value.model_fingerprint,
        preprocessing_fingerprint=value.preprocessing_fingerprint,
        target_classifier_fingerprint=value.target_classifier_fingerprint,
        regional_classifier_fingerprint=value.regional_classifier_fingerprint,
        candidate_keys=tuple(item.accepted_taxon_key for item in candidates),
    )
    if value.score_contract_fingerprint != expected_contract:
        raise ValueError("visual-input score contract fingerprint is inconsistent")
    values = {
        name: getattr(value, name)
        for name in FullFrameVisualInputScore.__dataclass_fields__
        if name != "score_fingerprint"
    }
    fingerprint = _sha256(value.score_fingerprint, field="score_fingerprint")
    if canonical_semantic_fingerprint(_input_score_semantics(values)) != fingerprint:
        raise ValueError("visual-input score fingerprint is inconsistent")


def _validated_candidate_scores(
    values: Sequence[FullFrameCandidateScore],
    *,
    target_key: str,
) -> tuple[FullFrameCandidateScore, ...]:
    rows = tuple(values)
    if len(rows) < 2:
        raise ValueError("visual-input scoring requires a target and competitor")
    if any(not isinstance(item, FullFrameCandidateScore) for item in rows):
        raise TypeError("candidate_scores must contain FullFrameCandidateScore values")
    keys = [item.accepted_taxon_key for item in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("candidate_scores contain duplicate accepted taxon keys")
    if keys.count(target_key) != 1:
        raise ValueError("candidate_scores must contain the target exactly once")
    return tuple(sorted(rows, key=lambda item: item.accepted_taxon_key))


def _effective_input_weights(
    scores: Sequence[FullFrameVisualInputScore],
    *,
    policy: VisualInputFusionPolicy,
) -> dict[str, float]:
    counts = Counter(item.visual_input_kind for item in scores)
    kind_weights = dict(policy.visual_input_kind_weights)
    active_total = sum(kind_weights[kind] for kind in counts)
    if active_total <= 0.0:
        raise ValueError("available visual inputs have no positive fusion weight")
    result = {
        item.visual_input_id: (
            kind_weights[item.visual_input_kind]
            / active_total
            / counts[item.visual_input_kind]
        )
        for item in scores
    }
    if not isclose(sum(result.values()), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("effective visual-input weights do not sum to one")
    return result


def _weighted_optional_score(
    scores: tuple[float | None, ...],
    weights: tuple[float, ...],
) -> float | None:
    contributing = tuple(
        (score, weight)
        for score, weight in zip(scores, weights, strict=True)
        if weight > 0.0
    )
    if any(score is None for score, _weight in contributing):
        return None
    return sum(float(score) * weight for score, weight in contributing)


def _best_regional_competitor(
    values: Sequence[FullFrameCandidateScore],
    *,
    target_key: str,
) -> FullFrameCandidateScore:
    return min(
        (item for item in values if item.accepted_taxon_key != target_key),
        key=lambda item: (
            -item.regional_classifier_decision_score,
            item.accepted_taxon_key,
        ),
    )


def _best_aggregated_regional_competitor(
    values: Sequence[AggregatedFullFrameCandidateScore],
    *,
    target_key: str,
) -> AggregatedFullFrameCandidateScore:
    return min(
        (item for item in values if item.accepted_taxon_key != target_key),
        key=lambda item: (
            -item.regional_classifier_decision_score,
            item.accepted_taxon_key,
        ),
    )


def _optional_margin(target: float | None, competitor: float | None) -> float | None:
    return None if target is None or competitor is None else target - competitor


def _score_contract_fingerprint(
    *,
    route: str,
    target_key: str,
    candidate_set_fingerprint: str,
    model_fingerprint: str,
    preprocessing_fingerprint: str,
    target_classifier_fingerprint: str,
    regional_classifier_fingerprint: str,
    candidate_keys: tuple[str, ...],
) -> str:
    return canonical_semantic_fingerprint(
        {
            "score_version": VISUAL_INPUT_SCORE_VERSION,
            "score_space": {
                "regional_classifier": "uncalibrated_decision_score",
                "target_classifier": "uncalibrated_decision_score",
                "reference": "cosine_similarity",
            },
            "route": route,
            "target_accepted_taxon_key": target_key,
            "candidate_set_fingerprint": candidate_set_fingerprint,
            "model_fingerprint": model_fingerprint,
            "preprocessing_fingerprint": preprocessing_fingerprint,
            "target_classifier_fingerprint": target_classifier_fingerprint,
            "regional_classifier_fingerprint": regional_classifier_fingerprint,
            "candidate_keys": list(candidate_keys),
        }
    )


def _embedding_set_semantics(values: Mapping[str, object]) -> dict[str, object]:
    embeddings = tuple(values["embeddings"])
    evidence = tuple(values["attention_evidence"])
    unavailable = tuple(values["unavailable_variants"])
    return {
        "embedding_set_version": values["embedding_set_version"],
        "visual_input_version": values["visual_input_version"],
        "raw_image_content_hash": values["raw_image_content_hash"],
        "route": values["route"],
        "model_fingerprint": values["model_fingerprint"],
        "preprocessing_fingerprint": values["preprocessing_fingerprint"],
        "embeddings": [
            asdict(item) for item in sorted(embeddings, key=_embedding_sort_key)
        ],
        "attention_evidence": [
            asdict(item) for item in sorted(evidence, key=_evidence_sort_key)
        ],
        "unavailable_variants": [
            asdict(item) for item in sorted(unavailable, key=_unavailable_sort_key)
        ],
    }


def _input_score_semantics(values: Mapping[str, object]) -> dict[str, object]:
    result = dict(values)
    result["candidate_scores"] = [
        asdict(item) for item in tuple(values["candidate_scores"])
    ]
    return result


def _policy_semantics(values: Mapping[str, object]) -> dict[str, object]:
    result = dict(values)
    result["visual_input_kind_weights"] = [
        [kind, weight] for kind, weight in tuple(values["visual_input_kind_weights"])
    ]
    return result


def _fusion_result_semantics(values: Mapping[str, object]) -> dict[str, object]:
    result = dict(values)
    result["candidate_scores"] = [
        asdict(item) for item in tuple(values["candidate_scores"])
    ]
    result["visual_input_evidence"] = [
        asdict(item) for item in tuple(values["visual_input_evidence"])
    ]
    result["unavailable_variants"] = [
        asdict(item)
        for item in sorted(
            tuple(values["unavailable_variants"]),
            key=_unavailable_sort_key,
        )
    ]
    return result


def _validated_vector(value: object, *, field: str) -> tuple[float, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a numeric vector")
    vector = tuple(_finite_float(item, field=field) for item in value)
    if not vector:
        raise ValueError(f"{field} must not be empty")
    norm = hypot(*vector)
    if not isfinite(norm):
        raise ValueError(f"{field} has a non-finite norm")
    if norm == 0.0:
        raise ValueError(f"{field} has zero norm")
    return vector


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _required_choice(value: object, *, field: str, allowed: frozenset[str]) -> str:
    result = _required_text(value, field=field)
    if result not in allowed:
        raise ValueError(f"unsupported {field}: {result}")
    return result


def _sha256(value: object, *, field: str) -> str:
    result = _required_text(value, field=field)
    if not _SHA256_PATTERN.fullmatch(result):
        raise ValueError(f"{field} must be a full sha256 fingerprint")
    return result


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _nonnegative_float(value: object, *, field: str) -> float:
    result = _finite_float(value, field=field)
    if result < 0.0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _optional_bounded_float(
    value: object,
    *,
    field: str,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None
    result = _finite_float(value, field=field)
    if not minimum <= result <= maximum:
        raise ValueError(f"{field} must be in [{minimum}, {maximum}]")
    return result


def _variant_sort_key(value: FullFrameAttentionVariant) -> tuple[int, str]:
    return _VISUAL_INPUT_KIND_ORDER[value.visual_input_kind], value.visual_input_id


def _embedding_sort_key(value: FullFrameVisualInputEmbedding) -> tuple[int, str]:
    return _VISUAL_INPUT_KIND_ORDER[value.visual_input_kind], value.visual_input_id


def _evidence_sort_key(value: FullFrameVariantEvidence) -> tuple[int, str, str]:
    return (
        _VISUAL_INPUT_KIND_ORDER[value.visual_input_kind],
        value.visual_input_id,
        value.evidence_id,
    )


def _unavailable_sort_key(
    value: UnavailableAttentionVariant,
) -> tuple[int, tuple[str, ...], str]:
    return (
        _VISUAL_INPUT_KIND_ORDER[value.visual_input_kind],
        value.source_detection_ids,
        value.unavailable_id,
    )


__all__ = [
    "LEARNED_LINEAR_VISUAL_INPUT_FUSION",
    "RAW_ONLY_VISUAL_INPUT_FUSION",
    "STRONG_VISUAL_INPUT_DISAGREEMENT",
    "VISUAL_INPUT_DISAGREEMENT_VERSION",
    "VISUAL_INPUT_EMBEDDING_SET_VERSION",
    "VISUAL_INPUT_FUSION_POLICY_VERSION",
    "VISUAL_INPUT_FUSION_RESULT_VERSION",
    "VISUAL_INPUT_SCORE_VERSION",
    "AggregatedFullFrameCandidateScore",
    "EmbeddedFullFrameVisualInputs",
    "FullFrameCandidateScore",
    "FullFrameVisualInputEmbedding",
    "FullFrameVisualInputEvidence",
    "FullFrameVisualInputScore",
    "VisualInputFusionPolicy",
    "VisualInputFusionResult",
    "build_full_frame_visual_input_score",
    "build_visual_input_fusion_policy",
    "embed_full_frame_visual_inputs",
    "fuse_full_frame_visual_evidence",
    "visual_input_fusion_result_payload",
]
