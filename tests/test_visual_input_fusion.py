from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import pytest

from biominer.bioclip import visual_input_fusion as fusion_module
from biominer.bioclip.visual_input_fusion import (
    LEARNED_LINEAR_VISUAL_INPUT_FUSION,
    RAW_ONLY_VISUAL_INPUT_FUSION,
    FullFrameCandidateScore,
    build_full_frame_visual_input_score,
    build_visual_input_fusion_policy,
    embed_full_frame_visual_inputs,
    fuse_full_frame_visual_evidence,
    visual_input_fusion_result_payload,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.detection.detector_base import DecodedImage
from biominer.vision.full_frame_attention import (
    FOCUSED_FULL_FRAME_KIND,
    MASKED_FULL_FRAME_KIND,
    MULTI_OBJECT_FULL_FRAME_KIND,
    RAW_FULL_IMAGE_KIND,
    TARGET_FULL_FRAME_IMAGE_RESIZE_MODE,
    TARGET_FULL_FRAME_PREPROCESSING,
    AttentionRegion,
    generate_full_frame_attention_variants,
)
from biominer.vision.target_full_frame import (
    TARGET_FULL_FRAME_EMBEDDING_VERSION,
    RawFullFrameEmbedding,
    full_frame_embedding_id,
)


MODEL_FINGERPRINT = "sha256:" + "1" * 64
PREPROCESSING_FINGERPRINT = "sha256:" + "2" * 64
TARGET_CLASSIFIER_FINGERPRINT = "sha256:" + "3" * 64
REGIONAL_CLASSIFIER_FINGERPRINT = "sha256:" + "4" * 64
CANDIDATE_SET_FINGERPRINT = "sha256:" + "5" * 64
SPLIT_FINGERPRINT = "sha256:" + "6" * 64
TRAINING_DATA_FINGERPRINT = "sha256:" + "7" * 64
TARGET_KEY = "gbif:target"
COMPETITOR_KEY = "gbif:competitor"


class RecordingEncoder:
    image_resize_mode = TARGET_FULL_FRAME_IMAGE_RESIZE_MODE
    preprocessing_contract_fingerprint = TARGET_FULL_FRAME_PREPROCESSING.fingerprint
    preprocessing_fingerprint = PREPROCESSING_FINGERPRINT

    def __init__(self) -> None:
        self.batches: list[tuple[DecodedImage, ...]] = []

    def encode_images(
        self,
        images: Sequence[DecodedImage],
    ) -> list[list[float]]:
        batch = tuple(images)
        self.batches.append(batch)
        return [
            [float(index + 1), float((image.data[0] % 5) + 1), 1.0]
            for index, image in enumerate(batch)
        ]


def _sha(value: str) -> str:
    return canonical_semantic_fingerprint({"fixture": value})


def _image() -> DecodedImage:
    return DecodedImage(
        width=4,
        height=4,
        mode="RGB",
        data=bytes(range(48)),
        source_uri="memory://photo-1",
    )


def _attention_result(*, include_mask: bool = True, multiple: bool = False):
    regions = [
        AttentionRegion(
            source_detection_id="det-a",
            route="adult_field",
            bbox_xyxyn=(0.0, 0.0, 0.5, 0.5),
            mask_polygon_xyn=(
                ((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5))
                if include_mask
                else None
            ),
            detector_score=0.9,
        )
    ]
    if multiple:
        regions.append(
            AttentionRegion(
                source_detection_id="det-b",
                route="adult_field",
                bbox_xyxyn=(0.5, 0.5, 1.0, 1.0),
                mask_polygon_xyn=(
                    (0.5, 0.5),
                    (1.0, 0.5),
                    (1.0, 1.0),
                    (0.5, 1.0),
                ),
                detector_score=0.8,
            )
        )
    return generate_full_frame_attention_variants(
        _image(),
        regions,
        source_type="flickr",
        source_record_id=_sha("source-row"),
    )


def _raw_embedding(result) -> RawFullFrameEmbedding:
    raw = next(
        item
        for item in result.variants
        if item.visual_input_kind == RAW_FULL_IMAGE_KIND
    )
    vector = (1.0, 0.0, 0.0)
    embedding_id = full_frame_embedding_id(
        visual_input_id=raw.visual_input_id,
        model_fingerprint=MODEL_FINGERPRINT,
        preprocessing_fingerprint=PREPROCESSING_FINGERPRINT,
    )
    embedding_fingerprint = canonical_semantic_fingerprint(
        {
            "embedding": vector,
            "embedding_id": embedding_id,
            "embedding_version": TARGET_FULL_FRAME_EMBEDDING_VERSION,
        }
    )
    return RawFullFrameEmbedding(
        embedding_id=embedding_id,
        embedding_version=TARGET_FULL_FRAME_EMBEDDING_VERSION,
        embedding_fingerprint=embedding_fingerprint,
        visual_input_id=raw.visual_input_id,
        visual_input_kind=RAW_FULL_IMAGE_KIND,
        raw_image_content_hash=raw.raw_image_content_hash,
        transformation_fingerprint=raw.transformation_fingerprint,
        model_fingerprint=MODEL_FINGERPRINT,
        image_resize_mode=TARGET_FULL_FRAME_IMAGE_RESIZE_MODE,
        preprocessing_contract_fingerprint=TARGET_FULL_FRAME_PREPROCESSING.fingerprint,
        preprocessing_fingerprint=PREPROCESSING_FINGERPRINT,
        embedding_dimension=len(vector),
        embedding=vector,
        embedding_norm=1.0,
    )


def _embedded(*, include_mask: bool = True, multiple: bool = False):
    attention = _attention_result(include_mask=include_mask, multiple=multiple)
    encoder = RecordingEncoder()
    embedded = embed_full_frame_visual_inputs(
        attention,
        encoder=encoder,
        model_fingerprint=MODEL_FINGERPRINT,
        preprocessing_fingerprint=PREPROCESSING_FINGERPRINT,
        raw_embedding=_raw_embedding(attention),
    )
    return attention, encoder, embedded


def _candidate_scores(
    *,
    target: float,
    competitor: float,
    target_reference: float | None = 0.7,
    competitor_reference: float | None = 0.3,
) -> tuple[FullFrameCandidateScore, ...]:
    return (
        FullFrameCandidateScore(
            accepted_taxon_key=TARGET_KEY,
            scientific_name="Target species",
            regional_classifier_decision_score=target,
            reference_centroid_similarity=target_reference,
        ),
        FullFrameCandidateScore(
            accepted_taxon_key=COMPETITOR_KEY,
            scientific_name="Competitor species",
            regional_classifier_decision_score=competitor,
            reference_centroid_similarity=competitor_reference,
        ),
    )


def _scores(embedded, margins: dict[str, float] | None = None):
    margins = margins or {}
    rows = []
    for index, embedding in enumerate(embedded.embeddings):
        margin = margins.get(embedding.visual_input_kind, 0.5 - (index * 0.05))
        rows.append(
            build_full_frame_visual_input_score(
                embedding=embedding,
                route="adult_field",
                target_accepted_taxon_key=TARGET_KEY,
                candidate_set_fingerprint=CANDIDATE_SET_FINGERPRINT,
                target_classifier_fingerprint=TARGET_CLASSIFIER_FINGERPRINT,
                regional_classifier_fingerprint=REGIONAL_CLASSIFIER_FINGERPRINT,
                target_classifier_decision_score=margin + 0.2,
                candidate_scores=_candidate_scores(target=margin, competitor=0.0),
                source_fusion_fingerprint=_sha(f"fusion-{embedding.visual_input_id}"),
            )
        )
    return tuple(rows)


def _policy(
    *,
    strategy: str = LEARNED_LINEAR_VISUAL_INPUT_FUSION,
    threshold: float = 0.4,
):
    weights = {
        RAW_FULL_IMAGE_KIND: 0.5,
        FOCUSED_FULL_FRAME_KIND: 0.3,
        MASKED_FULL_FRAME_KIND: 0.2,
        MULTI_OBJECT_FULL_FRAME_KIND: 0.0,
    }
    if strategy == RAW_ONLY_VISUAL_INPUT_FUSION:
        weights = {
            RAW_FULL_IMAGE_KIND: 1.0,
            FOCUSED_FULL_FRAME_KIND: 0.0,
            MASKED_FULL_FRAME_KIND: 0.0,
            MULTI_OBJECT_FULL_FRAME_KIND: 0.0,
        }
    return build_visual_input_fusion_policy(
        strategy=strategy,
        route="adult_field",
        visual_input_kind_weights=weights,
        strong_disagreement_threshold=threshold,
        model_fingerprint=MODEL_FINGERPRINT,
        preprocessing_fingerprint=PREPROCESSING_FINGERPRINT,
        target_classifier_fingerprint=TARGET_CLASSIFIER_FINGERPRINT,
        regional_classifier_fingerprint=REGIONAL_CLASSIFIER_FINGERPRINT,
        split_fingerprint=SPLIT_FINGERPRINT,
        training_data_fingerprint=TRAINING_DATA_FINGERPRINT,
        fit_partition="model_selection",
    )


def test_raw_embedding_is_reused_and_each_attention_variant_is_embedded_once() -> None:
    attention, encoder, embedded = _embedded()
    raw_id = _raw_embedding(attention).embedding_id

    assert len(encoder.batches) == 1
    assert len(encoder.batches[0]) == len(attention.variants) - 1
    assert all(image != _image() for image in encoder.batches[0])
    assert len(embedded.embeddings) == len(attention.variants)
    assert embedded.embedding(RAW_FULL_IMAGE_KIND).embedding_id == raw_id
    assert {item.visual_input_id for item in embedded.embeddings} == {
        item.visual_input_id for item in attention.variants
    }
    assert embedded.unavailable_variants == ()


def test_embedding_without_cached_raw_encodes_every_available_input_once() -> None:
    attention = _attention_result()
    encoder = RecordingEncoder()

    embedded = embed_full_frame_visual_inputs(
        attention,
        encoder=encoder,
        model_fingerprint=MODEL_FINGERPRINT,
        preprocessing_fingerprint=PREPROCESSING_FINGERPRINT,
    )

    assert len(encoder.batches) == 1
    assert len(encoder.batches[0]) == len(attention.variants)
    assert (
        sum(
            item.visual_input_kind == RAW_FULL_IMAGE_KIND
            for item in embedded.embeddings
        )
        == 1
    )


def test_unavailable_attention_variant_is_retained_but_not_embedded_or_scored() -> None:
    attention, encoder, embedded = _embedded(include_mask=False)

    assert {item.visual_input_kind for item in embedded.unavailable_variants} == {
        MASKED_FULL_FRAME_KIND
    }
    assert len(encoder.batches[0]) == len(attention.variants) - 1
    assert MASKED_FULL_FRAME_KIND not in {
        item.visual_input_kind for item in embedded.embeddings
    }

    result = fuse_full_frame_visual_evidence(
        source="flickr",
        photo_id="photo-1",
        scoring_unit_id=_sha("scoring-unit"),
        embedded_inputs=embedded,
        visual_input_scores=_scores(embedded),
        policy=_policy(),
    )
    assert result.unavailable_variants == embedded.unavailable_variants
    weights = {
        item.visual_input_kind: item.aggregation_weight
        for item in result.visual_input_evidence
    }
    assert weights[RAW_FULL_IMAGE_KIND] == pytest.approx(0.625)
    assert weights[FOCUSED_FULL_FRAME_KIND] == pytest.approx(0.375)


def test_learned_linear_fusion_preserves_inputs_weights_and_is_order_independent() -> (
    None
):
    _attention, _encoder, embedded = _embedded(multiple=True)
    scores = _scores(embedded)
    policy = _policy(threshold=1.0)

    first = fuse_full_frame_visual_evidence(
        source="flickr",
        photo_id="photo-1",
        scoring_unit_id=_sha("scoring-unit"),
        embedded_inputs=embedded,
        visual_input_scores=scores,
        policy=policy,
    )
    second = fuse_full_frame_visual_evidence(
        source="flickr",
        photo_id="photo-1",
        scoring_unit_id=_sha("scoring-unit"),
        embedded_inputs=replace(
            embedded,
            embeddings=tuple(reversed(embedded.embeddings)),
        ),
        visual_input_scores=tuple(reversed(scores)),
        policy=policy,
    )

    assert first.fusion_fingerprint == second.fusion_fingerprint
    assert len(first.visual_input_evidence) == len(embedded.embeddings)
    assert sum(item.aggregation_weight for item in first.visual_input_evidence) == (
        pytest.approx(1.0)
    )
    kind_weight = {
        kind: sum(
            item.aggregation_weight
            for item in first.visual_input_evidence
            if item.visual_input_kind == kind
        )
        for kind in {item.visual_input_kind for item in first.visual_input_evidence}
    }
    assert kind_weight[RAW_FULL_IMAGE_KIND] == pytest.approx(0.5)
    assert kind_weight[FOCUSED_FULL_FRAME_KIND] == pytest.approx(0.3)
    assert kind_weight[MASKED_FULL_FRAME_KIND] == pytest.approx(0.2)
    assert kind_weight[MULTI_OBJECT_FULL_FRAME_KIND] == pytest.approx(0.0)
    assert {item.accepted_taxon_key for item in first.candidate_scores} == {
        TARGET_KEY,
        COMPETITOR_KEY,
    }
    assert visual_input_fusion_result_payload(first)["fusion_fingerprint"] == (
        first.fusion_fingerprint
    )


def test_strong_margin_disagreement_routes_to_review_without_erasing_scores() -> None:
    _attention, _encoder, embedded = _embedded()
    scores = _scores(
        embedded,
        margins={
            RAW_FULL_IMAGE_KIND: 0.8,
            FOCUSED_FULL_FRAME_KIND: -0.5,
            MASKED_FULL_FRAME_KIND: 0.1,
        },
    )

    result = fuse_full_frame_visual_evidence(
        source="flickr",
        photo_id="photo-1",
        scoring_unit_id=_sha("scoring-unit"),
        embedded_inputs=embedded,
        visual_input_scores=scores,
        policy=_policy(threshold=0.4),
    )

    assert result.visual_input_disagreement == pytest.approx(1.3)
    assert result.strong_visual_input_disagreement is True
    assert result.routing_action == "review"
    assert result.review_reason == "strong_visual_input_disagreement"
    assert len(result.visual_input_evidence) == len(scores)
    assert {
        item.regional_competitor_margin for item in result.visual_input_evidence
    } == {
        -0.5,
        0.1,
        0.8,
    }


def test_low_disagreement_remains_scoreable_and_raw_only_keeps_diagnostics() -> None:
    _attention, _encoder, embedded = _embedded()
    scores = _scores(embedded)

    result = fuse_full_frame_visual_evidence(
        source="flickr",
        photo_id="photo-1",
        scoring_unit_id=_sha("scoring-unit"),
        embedded_inputs=embedded,
        visual_input_scores=scores,
        policy=_policy(strategy=RAW_ONLY_VISUAL_INPUT_FUSION, threshold=1.0),
    )

    assert result.routing_action == "score"
    assert result.review_reason is None
    assert len(result.visual_input_evidence) == len(scores)
    assert next(
        item.aggregation_weight
        for item in result.visual_input_evidence
        if item.visual_input_kind == RAW_FULL_IMAGE_KIND
    ) == pytest.approx(1.0)
    assert sum(
        item.aggregation_weight
        for item in result.visual_input_evidence
        if item.visual_input_kind != RAW_FULL_IMAGE_KIND
    ) == pytest.approx(0.0)


def test_competitor_reference_score_belongs_to_the_regional_winner() -> None:
    _attention, _encoder, embedded = _embedded()
    embedding = embedded.embeddings[0]

    score = build_full_frame_visual_input_score(
        embedding=embedding,
        route="adult_field",
        target_accepted_taxon_key=TARGET_KEY,
        candidate_set_fingerprint=CANDIDATE_SET_FINGERPRINT,
        target_classifier_fingerprint=TARGET_CLASSIFIER_FINGERPRINT,
        regional_classifier_fingerprint=REGIONAL_CLASSIFIER_FINGERPRINT,
        target_classifier_decision_score=0.7,
        candidate_scores=(
            *_candidate_scores(
                target=0.8,
                competitor=0.7,
                target_reference=0.6,
                competitor_reference=0.1,
            ),
            FullFrameCandidateScore(
                accepted_taxon_key="gbif:reference-winner",
                scientific_name="Reference winner",
                regional_classifier_decision_score=0.2,
                reference_centroid_similarity=0.9,
            ),
        ),
        source_fusion_fingerprint=_sha("three-candidate-fusion"),
    )

    assert score.best_competitor_accepted_taxon_key == COMPETITOR_KEY
    assert score.best_competitor_reference_centroid_similarity == pytest.approx(0.1)
    assert score.reference_competitor_margin == pytest.approx(0.5)


def test_missing_contributing_reference_score_fails_closed_to_null() -> None:
    _attention, _encoder, embedded = _embedded()
    scores = list(_scores(embedded))
    focused_index = next(
        index
        for index, item in enumerate(scores)
        if item.visual_input_kind == FOCUSED_FULL_FRAME_KIND
    )
    focused_embedding = next(
        item
        for item in embedded.embeddings
        if item.visual_input_kind == FOCUSED_FULL_FRAME_KIND
    )
    scores[focused_index] = build_full_frame_visual_input_score(
        embedding=focused_embedding,
        route="adult_field",
        target_accepted_taxon_key=TARGET_KEY,
        candidate_set_fingerprint=CANDIDATE_SET_FINGERPRINT,
        target_classifier_fingerprint=TARGET_CLASSIFIER_FINGERPRINT,
        regional_classifier_fingerprint=REGIONAL_CLASSIFIER_FINGERPRINT,
        target_classifier_decision_score=0.7,
        candidate_scores=_candidate_scores(
            target=0.5,
            competitor=0.0,
            target_reference=None,
        ),
        source_fusion_fingerprint=_sha("missing-reference-fusion"),
    )

    result = fuse_full_frame_visual_evidence(
        source="flickr",
        photo_id="photo-1",
        scoring_unit_id=_sha("scoring-unit"),
        embedded_inputs=embedded,
        visual_input_scores=scores,
        policy=_policy(threshold=1.0),
    )

    target = next(
        item
        for item in result.candidate_scores
        if item.accepted_taxon_key == TARGET_KEY
    )
    assert target.reference_centroid_similarity is None
    assert result.target_reference_centroid_similarity is None
    assert result.reference_competitor_margin is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"fit_partition": "final_test"}, "model_selection"),
        ({"visual_input_kind_weights": {RAW_FULL_IMAGE_KIND: 1.0}}, "kind set"),
        (
            {
                "visual_input_kind_weights": {
                    RAW_FULL_IMAGE_KIND: 1.1,
                    FOCUSED_FULL_FRAME_KIND: -0.1,
                    MASKED_FULL_FRAME_KIND: 0.0,
                    MULTI_OBJECT_FULL_FRAME_KIND: 0.0,
                }
            },
            "nonnegative",
        ),
    ],
)
def test_policy_rejects_unvalidated_or_incomplete_linear_weights(
    changes: dict[str, Any],
    message: str,
) -> None:
    kwargs: dict[str, Any] = {
        "strategy": LEARNED_LINEAR_VISUAL_INPUT_FUSION,
        "route": "adult_field",
        "visual_input_kind_weights": {
            RAW_FULL_IMAGE_KIND: 0.5,
            FOCUSED_FULL_FRAME_KIND: 0.3,
            MASKED_FULL_FRAME_KIND: 0.2,
            MULTI_OBJECT_FULL_FRAME_KIND: 0.0,
        },
        "strong_disagreement_threshold": 0.4,
        "model_fingerprint": MODEL_FINGERPRINT,
        "preprocessing_fingerprint": PREPROCESSING_FINGERPRINT,
        "target_classifier_fingerprint": TARGET_CLASSIFIER_FINGERPRINT,
        "regional_classifier_fingerprint": REGIONAL_CLASSIFIER_FINGERPRINT,
        "split_fingerprint": SPLIT_FINGERPRINT,
        "training_data_fingerprint": TRAINING_DATA_FINGERPRINT,
        "fit_partition": "model_selection",
    }
    kwargs.update(changes)

    with pytest.raises(ValueError, match=message):
        build_visual_input_fusion_policy(**kwargs)


def test_fusion_rejects_score_coverage_and_identity_mismatches() -> None:
    _attention, _encoder, embedded = _embedded()
    scores = _scores(embedded)
    common = {
        "source": "flickr",
        "photo_id": "photo-1",
        "scoring_unit_id": _sha("scoring-unit"),
        "embedded_inputs": embedded,
        "policy": _policy(),
    }

    with pytest.raises(ValueError, match="score coverage"):
        fuse_full_frame_visual_evidence(
            **common,
            visual_input_scores=scores[:-1],
        )

    mismatched = replace(
        scores[0],
        candidate_set_fingerprint=_sha("another-candidate-set"),
    )
    with pytest.raises(ValueError, match="candidate-set identity"):
        fuse_full_frame_visual_evidence(
            **common,
            visual_input_scores=(mismatched, *scores[1:]),
        )

    duplicate_raw = replace(
        embedded,
        embeddings=(embedded.embeddings[0], embedded.embeddings[0]),
    )
    with pytest.raises(ValueError, match="duplicate visual_input_id"):
        fuse_full_frame_visual_evidence(
            **common | {"embedded_inputs": duplicate_raw},
            visual_input_scores=scores,
        )


def test_fusion_rejects_semantically_invalid_policy_with_recomputed_hash() -> None:
    _attention, _encoder, embedded = _embedded()
    policy = replace(
        _policy(),
        visual_input_kind_weights=(
            (RAW_FULL_IMAGE_KIND, 0.5),
            (FOCUSED_FULL_FRAME_KIND, 0.5),
            (MASKED_FULL_FRAME_KIND, 0.5),
            (MULTI_OBJECT_FULL_FRAME_KIND, 0.0),
        ),
    )
    values = {
        name: getattr(policy, name)
        for name in policy.__dataclass_fields__
        if name != "policy_fingerprint"
    }
    forged = replace(
        policy,
        policy_fingerprint=canonical_semantic_fingerprint(
            fusion_module._policy_semantics(values)  # noqa: SLF001
        ),
    )

    with pytest.raises(ValueError, match="weights must sum to one"):
        fuse_full_frame_visual_evidence(
            source="flickr",
            photo_id="photo-1",
            scoring_unit_id=_sha("scoring-unit"),
            embedded_inputs=embedded,
            visual_input_scores=_scores(embedded),
            policy=forged,
        )


def test_result_payload_detects_tampering() -> None:
    _attention, _encoder, embedded = _embedded()
    result = fuse_full_frame_visual_evidence(
        source="flickr",
        photo_id="photo-1",
        scoring_unit_id=_sha("scoring-unit"),
        embedded_inputs=embedded,
        visual_input_scores=_scores(embedded),
        policy=_policy(),
    )

    with pytest.raises(ValueError, match="fingerprint is inconsistent"):
        visual_input_fusion_result_payload(
            replace(result, visual_input_disagreement=0.0)
        )
