from __future__ import annotations

from dataclasses import replace

import pytest

from biominer.detection.detector_base import DecodedImage
from biominer.vision.full_frame_attention import (
    FOCUSED_FULL_FRAME_KIND,
    MASKED_FULL_FRAME_KIND,
    MASK_COVERAGE_BASIS_RASTERIZED_POLYGON,
    MULTI_OBJECT_FULL_FRAME_KIND,
    RAW_FULL_IMAGE_KIND,
    TARGET_FULL_FRAME_PREPROCESSING,
    AttentionQualityPolicy,
    AttentionRegion,
    FullFrameTransformPolicy,
    TargetPreprocessingContract,
    decoded_image_content_hash,
    generate_full_frame_attention_variants,
    raw_full_frame_visual_input,
)


def _image(
    *,
    width: int = 4,
    height: int = 4,
    value: int = 100,
    source_uri: str | None = "test://image",
) -> DecodedImage:
    return DecodedImage(
        width=width,
        height=height,
        mode="RGB",
        data=bytes([value, value + 1, value + 2]) * (width * height),
        source_uri=source_uri,
    )


def _region(
    detection_id: str = "det-1",
    *,
    route: str = "adult_field",
    bbox: tuple[float, float, float, float] = (0.25, 0.25, 0.75, 0.75),
    polygon: tuple[tuple[float, float], ...] | None = None,
    score: float | None = 0.8,
) -> AttentionRegion:
    return AttentionRegion(
        source_detection_id=detection_id,
        route=route,
        bbox_xyxyn=bbox,
        mask_polygon_xyn=polygon,
        detector_score=score,
    )


def _generate(
    image: DecodedImage,
    regions: tuple[AttentionRegion, ...],
    **kwargs: object,
):
    return generate_full_frame_attention_variants(
        image,
        regions,
        source_type=str(kwargs.pop("source_type", "flickr")),
        source_record_id=str(kwargs.pop("source_record_id", "photo-1")),
        **kwargs,
    )


def _variants(result: object, kind: str):
    return [
        item
        for item in result.variants  # type: ignore[attr-defined]
        if item.visual_input_kind == kind
    ]


def _evidence(result: object, kind: str):
    return [
        item
        for item in result.evidence  # type: ignore[attr-defined]
        if item.visual_input_kind == kind
    ]


def test_raw_visual_input_is_authoritative_byte_identical_identity() -> None:
    image = _image(width=3, height=2)

    direct = raw_full_frame_visual_input(image)
    generated = _generate(image, (_region(),))
    raw = _variants(generated, RAW_FULL_IMAGE_KIND)[0]

    assert raw == direct
    assert raw.image is image
    assert raw.image.data == image.data
    assert (raw.width, raw.height, raw.mode) == (3, 2, "RGB")
    assert raw.raw_image_content_hash == decoded_image_content_hash(image)
    assert raw.visual_content_hash == raw.raw_image_content_hash
    assert raw.transformation_applied is False
    assert raw.visual_input_version == "target-full-frame-visual-input-v2"
    assert raw.transformation_version == "raw-full-image-transform-v2"


def test_content_hash_and_raw_identity_ignore_source_uri() -> None:
    first = _image(source_uri="reference://one")
    second = replace(first, source_uri="flickr://two")

    assert decoded_image_content_hash(first) == decoded_image_content_hash(second)
    assert (
        raw_full_frame_visual_input(first).visual_input_id
        == raw_full_frame_visual_input(second).visual_input_id
    )


def test_focused_variant_preserves_canvas_and_uses_exact_integer_attenuation() -> None:
    image = _image(width=4, height=4, value=100)
    region = _region(bbox=(0.25, 0.25, 0.5, 0.5))

    result = _generate(image, (region,))
    focused = _variants(result, FOCUSED_FULL_FRAME_KIND)[0]
    focused_evidence = _evidence(result, FOCUSED_FULL_FRAME_KIND)[0]
    pixels = [
        focused.image.data[offset : offset + 3]
        for offset in range(0, len(focused.image.data), 3)
    ]

    assert (focused.width, focused.height) == (image.width, image.height)
    assert len(focused.image.data) == len(image.data)
    assert pixels[5] == bytes((100, 101, 102))
    assert all(
        pixel == bytes(((100 + 2) // 4, (101 + 2) // 4, (102 + 2) // 4))
        for index, pixel in enumerate(pixels)
        if index != 5
    )
    assert focused.transformation_applied is True
    assert focused.transformation_version == "full-frame-attention-transform-v2"
    assert focused_evidence.evidence_version == "full-frame-attention-evidence-v2"


def test_bbox_projection_uses_floor_start_and_ceil_exclusive_end() -> None:
    image = _image(width=4, height=4, value=80)
    result = _generate(
        image,
        (_region(bbox=(0.24, 0.24, 0.26, 0.26)),),
    )
    focused = _variants(result, FOCUSED_FULL_FRAME_KIND)[0]
    original = bytes((80, 81, 82))
    retained = {
        index
        for index in range(16)
        if focused.image.data[index * 3 : index * 3 + 3] == original
    }

    assert retained == {0, 1, 4, 5}


def test_mask_polygon_is_half_up_rasterized_with_explicit_coverage_basis() -> None:
    image = _image(width=4, height=4)
    polygon = ((0.125, 0.125), (0.375, 0.125), (0.125, 0.375))
    result = _generate(image, (_region(polygon=polygon),))
    masked = _variants(result, MASKED_FULL_FRAME_KIND)[0]
    evidence = _evidence(result, MASKED_FULL_FRAME_KIND)[0]
    retained_pixels = sum(
        masked.image.data[offset : offset + 3] == image.data[offset : offset + 3]
        for offset in range(0, len(image.data), 3)
    )

    assert retained_pixels == 3
    assert evidence.mask_coverage == pytest.approx(3 / 16)
    assert evidence.mask_coverage_basis == MASK_COVERAGE_BASIS_RASTERIZED_POLYGON
    assert "mask_polygon_approximation" in evidence.visual_input_quality_flags


def test_polygon_on_bbox_boundary_is_not_reported_outside_bbox() -> None:
    bbox = (0.25, 0.25, 0.75, 0.75)
    polygon = ((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75))

    result = _generate(_image(), (_region(bbox=bbox, polygon=polygon),))
    evidence = _evidence(result, MASKED_FULL_FRAME_KIND)[0]

    assert evidence.mask_coverage == pytest.approx(9 / 16)
    assert "mask_outside_bbox" not in evidence.visual_input_quality_flags
    assert "mask_polygon_approximation" in evidence.visual_input_quality_flags


@pytest.mark.parametrize(
    ("polygon", "reason"),
    [
        (None, "mask_missing"),
        (((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)), "mask_degenerate"),
    ],
)
def test_unavailable_mask_is_explicit_without_bbox_fallback(
    polygon: tuple[tuple[float, float], ...] | None,
    reason: str,
) -> None:
    result = _generate(_image(), (_region(polygon=polygon),))

    assert _variants(result, MASKED_FULL_FRAME_KIND) == []
    assert len(result.unavailable_variants) == 1
    unavailable = result.unavailable_variants[0]
    assert unavailable.visual_input_kind == MASKED_FULL_FRAME_KIND
    assert unavailable.unavailable_version == "full-frame-attention-unavailable-v2"
    assert unavailable.reason == reason
    assert reason in unavailable.visual_input_quality_flags


def test_polygon_identity_is_invariant_to_cyclic_start_and_reversal() -> None:
    image = _image()
    polygon = ((0.0, 0.0), (0.75, 0.0), (0.5, 0.75), (0.0, 0.5))
    rotated = polygon[2:] + polygon[:2]
    reversed_polygon = tuple(reversed(polygon))

    variants = [
        _variants(_generate(image, (_region(polygon=value),)), MASKED_FULL_FRAME_KIND)[
            0
        ]
        for value in (polygon, rotated, reversed_polygon)
    ]

    assert len({variant.visual_input_id for variant in variants}) == 1
    assert len({variant.transformation_fingerprint for variant in variants}) == 1
    assert len({variant.image.data for variant in variants}) == 1


def test_multi_object_union_area_does_not_double_count_overlap() -> None:
    image = _image()
    regions = (
        _region("det-a", bbox=(0.0, 0.0, 0.75, 0.75)),
        _region("det-b", bbox=(0.25, 0.25, 1.0, 1.0)),
    )
    result = _generate(image, regions)
    raw_evidence = _evidence(result, RAW_FULL_IMAGE_KIND)[0]
    multi_evidence = _evidence(result, MULTI_OBJECT_FULL_FRAME_KIND)[0]

    assert len(_variants(result, MULTI_OBJECT_FULL_FRAME_KIND)) == 1
    assert raw_evidence.subject_area_ratio == pytest.approx(0.875)
    assert multi_evidence.subject_area_ratio == pytest.approx(0.875)
    assert multi_evidence.relevant_detection_count == 2
    assert multi_evidence.source_detection_count == 2
    assert multi_evidence.detector_score == 0.8
    assert multi_evidence.detector_score_aggregation == "max"
    assert "multiple_relevant_detections" in multi_evidence.visual_input_quality_flags


def test_multi_object_variant_is_not_generated_for_one_region() -> None:
    result = _generate(_image(), (_region(),))

    assert _variants(result, MULTI_OBJECT_FULL_FRAME_KIND) == []


def test_generation_rejects_mixed_routes() -> None:
    regions = (
        _region("adult", route="adult_field"),
        _region("larva", route="larval"),
    )

    with pytest.raises(ValueError, match="route-compatible"):
        _generate(_image(), regions)


def test_region_order_does_not_affect_variants_or_evidence() -> None:
    image = _image()
    regions = (
        _region("det-b", bbox=(0.5, 0.5, 1.0, 1.0), score=0.7),
        _region("det-a", bbox=(0.0, 0.0, 0.5, 0.5), score=0.9),
    )

    forward = _generate(image, regions)
    reverse = _generate(image, tuple(reversed(regions)))

    assert forward == reverse
    assert all(
        item.source_detection_ids == tuple(sorted(item.source_detection_ids))
        for item in forward.evidence
    )


def test_detection_score_id_and_route_do_not_change_visual_identity() -> None:
    image = _image()
    first = _generate(
        image,
        (_region("reference-det", route="adult_field", score=0.4),),
    )
    second = _generate(
        image,
        (_region("flickr-det", route="adult_specimen", score=0.99),),
    )

    assert [item.visual_input_id for item in first.variants] == [
        item.visual_input_id for item in second.variants
    ]
    assert [item.visual_content_hash for item in first.variants] == [
        item.visual_content_hash for item in second.variants
    ]
    assert [item.evidence_id for item in first.evidence] != [
        item.evidence_id for item in second.evidence
    ]


def test_detector_score_alone_invalidates_evidence_not_visual_identity() -> None:
    image = _image()
    low_score = _generate(image, (_region(score=0.4),))
    high_score = _generate(image, (_region(score=0.9),))

    assert [item.visual_input_id for item in low_score.variants] == [
        item.visual_input_id for item in high_score.variants
    ]
    assert [item.evidence_id for item in low_score.evidence] != [
        item.evidence_id for item in high_score.evidence
    ]


def test_partial_mask_set_is_explicit_and_coverage_remains_unknown() -> None:
    polygon = ((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5))
    result = _generate(
        _image(),
        (
            _region("det-a", bbox=(0.0, 0.0, 0.5, 0.5), polygon=polygon),
            _region("det-b", bbox=(0.5, 0.5, 1.0, 1.0)),
        ),
    )

    for kind in (RAW_FULL_IMAGE_KIND, MULTI_OBJECT_FULL_FRAME_KIND):
        item = _evidence(result, kind)[0]
        assert item.mask_coverage is None
        assert item.mask_coverage_basis is None
        assert {"mask_missing", "partial_mask_set"}.issubset(
            item.visual_input_quality_flags
        )


def test_reference_and_flickr_use_identical_transform_pixels_and_ids() -> None:
    reference_image = _image(source_uri="reference://image")
    flickr_image = replace(reference_image, source_uri="flickr://image")
    regions = (
        _region(polygon=((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75))),
    )

    reference = _generate(
        reference_image,
        regions,
        source_type="reference",
        source_record_id="ref-1",
    )
    flickr = _generate(
        flickr_image,
        regions,
        source_type="flickr",
        source_record_id="photo-1",
    )

    assert [item.image.data for item in reference.variants] == [
        item.image.data for item in flickr.variants
    ]
    assert [item.visual_content_hash for item in reference.variants] == [
        item.visual_content_hash for item in flickr.variants
    ]
    assert [item.visual_input_id for item in reference.variants] == [
        item.visual_input_id for item in flickr.variants
    ]
    assert [item.evidence_id for item in reference.evidence] != [
        item.evidence_id for item in flickr.evidence
    ]


def test_applied_transform_remains_visible_when_full_canvas_is_pixel_noop() -> None:
    image = _image()
    result = _generate(image, (_region(bbox=(0.0, 0.0, 1.0, 1.0)),))
    raw = _variants(result, RAW_FULL_IMAGE_KIND)[0]
    focused = _variants(result, FOCUSED_FULL_FRAME_KIND)[0]
    evidence = _evidence(result, FOCUSED_FULL_FRAME_KIND)[0]

    assert focused.image.data == raw.image.data
    assert focused.visual_content_hash == raw.visual_content_hash
    assert focused.visual_input_id != raw.visual_input_id
    assert focused.transformation_applied is True
    assert focused.transformation_fingerprint != raw.transformation_fingerprint
    assert evidence.transformation_applied is True
    assert "attention_transform_noop" in evidence.visual_input_quality_flags


def test_transform_and_quality_policy_versions_have_separate_invalidation() -> None:
    image = _image()
    regions = (_region(),)
    baseline_transform = FullFrameTransformPolicy()
    changed_transform = replace(
        baseline_transform,
        version="full-frame-attention-transform-v3",
    )
    changed_dependency = replace(baseline_transform, pillow_version="future")
    baseline_quality = AttentionQualityPolicy()
    changed_quality = replace(
        baseline_quality,
        version="full-frame-attention-quality-v3",
        small_subject_area_ratio_threshold=0.5,
    )

    baseline = _generate(
        image,
        regions,
        transform_policy=baseline_transform,
        quality_policy=baseline_quality,
    )
    transformed = _generate(image, regions, transform_policy=changed_transform)
    dependency = _generate(image, regions, transform_policy=changed_dependency)
    quality = _generate(image, regions, quality_policy=changed_quality)

    baseline_focused = _variants(baseline, FOCUSED_FULL_FRAME_KIND)[0]
    assert baseline_transform.version == "full-frame-attention-transform-v2"
    assert baseline_quality.version == "full-frame-attention-quality-v2"
    assert baseline_transform.fingerprint != changed_transform.fingerprint
    assert baseline_transform.fingerprint != changed_dependency.fingerprint
    assert (
        baseline_focused.visual_input_id
        != _variants(transformed, FOCUSED_FULL_FRAME_KIND)[0].visual_input_id
    )
    assert (
        baseline_focused.visual_input_id
        != _variants(dependency, FOCUSED_FULL_FRAME_KIND)[0].visual_input_id
    )
    assert baseline_quality.fingerprint != changed_quality.fingerprint
    assert [item.visual_input_id for item in baseline.variants] == [
        item.visual_input_id for item in quality.variants
    ]
    assert [item.evidence_id for item in baseline.evidence] != [
        item.evidence_id for item in quality.evidence
    ]


def test_quality_evidence_records_small_low_resolution_edge_and_padding() -> None:
    image = _image(width=20, height=10)
    result = _generate(
        image,
        (_region(bbox=(0.0, 0.0, 0.05, 0.05), score=None),),
    )
    evidence = _evidence(result, FOCUSED_FULL_FRAME_KIND)[0]

    assert evidence.subject_area_ratio == pytest.approx(0.0025)
    assert evidence.detector_score is None
    assert evidence.detector_score_aggregation is None
    assert {
        "small_subject",
        "low_source_resolution",
        "preprocess_padding_required",
        "subject_touches_canvas_edge",
        "detector_score_missing",
    }.issubset(evidence.visual_input_quality_flags)


def test_evidence_keeps_complete_sorted_source_region_records() -> None:
    polygon = ((0.5, 0.5), (1.0, 0.5), (1.0, 1.0))
    regions = (
        _region("det-b", bbox=(0.5, 0.5, 1.0, 1.0), polygon=polygon, score=0.6),
        _region("det-a", bbox=(0.0, 0.0, 0.5, 0.5), score=0.9),
    )
    result = _generate(_image(), regions)
    evidence = _evidence(result, MULTI_OBJECT_FULL_FRAME_KIND)[0]

    assert evidence.source_detection_ids == ("det-a", "det-b")
    assert evidence.source_regions == tuple(
        sorted(regions, key=lambda item: item.source_detection_id)
    )
    assert evidence.source_regions[1].mask_polygon_xyn == polygon
    assert evidence.source_regions[1].detector_score == 0.6


def test_equivalent_duplicate_geometry_coalesces_visual_content_but_keeps_evidence() -> (
    None
):
    image = _image()
    regions = (
        _region("det-a", score=0.7),
        _region("det-b", score=0.9),
    )

    result = _generate(image, regions)
    focused = _variants(result, FOCUSED_FULL_FRAME_KIND)

    assert len(focused) == 1
    evidence = _evidence(result, FOCUSED_FULL_FRAME_KIND)
    assert [item.source_detection_ids for item in evidence] == [
        ("det-a",),
        ("det-b",),
    ]
    assert {item.visual_input_id for item in evidence} == {focused[0].visual_input_id}
    assert len({item.evidence_id for item in evidence}) == 2


def test_target_preprocessing_contract_is_explicit_and_fingerprinted() -> None:
    contract = TARGET_FULL_FRAME_PREPROCESSING

    assert contract.version == "target-full-frame-openclip-preprocess-v2"
    assert contract.image_size_px == 224
    assert contract.patch_size_px == 14
    assert contract.interpolation == "bicubic"
    assert contract.resize_mode == "longest"
    assert contract.padding_position == "center"
    assert contract.padding_fill == 0
    assert contract.normalization_mean == (0.48145466, 0.4578275, 0.40821073)
    assert contract.normalization_std == (0.26862954, 0.26130258, 0.27577711)
    assert contract.resize_geometry(width=400, height=200) == (
        224,
        112,
        (0, 56, 0, 56),
    )
    assert contract.fingerprint.startswith("sha256:")
    assert (
        replace(contract, version="target-preprocess-v3").fingerprint
        != contract.fingerprint
    )


def test_preprocessing_fingerprint_preserves_float64_signed_zero() -> None:
    positive = TargetPreprocessingContract(
        normalization_mean=(0.0, 0.2, 0.3),
    )
    negative = TargetPreprocessingContract(
        normalization_mean=(-0.0, 0.2, 0.3),
    )

    assert positive.fingerprint != negative.fingerprint


def test_transformation_fingerprint_binds_exact_normalized_geometry() -> None:
    image = _image(width=4, height=4)
    first = _variants(
        _generate(image, (_region(bbox=(0.25, 0.25, 0.75, 0.75)),)),
        FOCUSED_FULL_FRAME_KIND,
    )[0]
    second = _variants(
        _generate(image, (_region(bbox=(0.250001, 0.25, 0.75, 0.75)),)),
        FOCUSED_FULL_FRAME_KIND,
    )[0]

    assert first.image.data == second.image.data
    assert first.visual_content_hash == second.visual_content_hash
    assert first.transformation_fingerprint != second.transformation_fingerprint
    assert first.visual_input_id != second.visual_input_id


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_size_px": 225},
        {"patch_size_px": 7},
        {"interpolation": "nearest"},
        {"resize_mode": "shortest"},
        {"padding_position": "top-left"},
        {"padding_fill": 128},
        {"normalization_mean": (0.0, 0.0)},
        {"normalization_mean": (True, 0.0, 0.0)},
        {"normalization_mean": None},
        {"normalization_std": (1.0, 0.0, 1.0)},
    ],
)
def test_target_preprocessing_contract_rejects_canvas_losing_configuration(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        TargetPreprocessingContract(**kwargs)


def test_attention_region_validation_rejects_invalid_geometry_and_scores() -> None:
    with pytest.raises(ValueError, match="positive width"):
        _region(bbox=(0.5, 0.0, 0.5, 1.0))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _region(bbox=(-0.1, 0.0, 0.5, 1.0))
    with pytest.raises(ValueError, match="finite"):
        _region(score=float("nan"))
    with pytest.raises(ValueError, match="at least three points"):
        _region(polygon=((0.0, 0.0), (1.0, 1.0)))


def test_generation_fails_before_unbounded_region_or_pixel_materialization() -> None:
    regions = tuple(
        _region(
            f"det-{index}",
            bbox=(index / 10, 0.0, (index + 1) / 10, 0.5),
        )
        for index in range(9)
    )
    with pytest.raises(ValueError, match="region count exceeds"):
        _generate(_image(width=10, height=10), regions)

    constrained = replace(
        AttentionQualityPolicy(),
        max_materialization_bytes=100,
    )
    with pytest.raises(ValueError, match="materialization budget exceeded"):
        _generate(
            _image(width=10, height=10),
            (_region(),),
            quality_policy=constrained,
        )
