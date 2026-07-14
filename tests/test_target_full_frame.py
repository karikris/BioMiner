from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, Sequence

import pytest

from biominer.vision import target_full_frame
from biominer.detection.detector_base import DecodedImage
from biominer.detection.policy import DetectionRunPolicy
from biominer.vision.target_full_frame import (
    RAW_FULL_IMAGE_KIND,
    TARGET_AWARE_VISUAL_MODE,
    TARGET_FULL_FRAME_EMBEDDING_VERSION,
    RawFullFrameEmbedding,
    RawFullFrameVisualInput,
    TargetFullFrameScoringUnit,
    build_target_full_frame_plan,
    encode_target_full_frame_plan,
    full_frame_embedding_id,
    generate_target_full_frame_attention_variants,
    target_full_frame_detection_run_policy,
)
from biominer.vision.full_frame_attention import (
    FOCUSED_FULL_FRAME_KIND,
    MASKED_FULL_FRAME_KIND,
    TARGET_FULL_FRAME_IMAGE_RESIZE_MODE,
    TARGET_FULL_FRAME_PREPROCESSING,
)


_ROUTING_POLICY_FINGERPRINT = "sha256:" + "a" * 64
_MODEL_FINGERPRINT = "sha256:" + "b" * 64
_PREPROCESSING_FINGERPRINT = "sha256:" + "c" * 64


class RecordingEncoder:
    image_resize_mode = TARGET_FULL_FRAME_IMAGE_RESIZE_MODE
    preprocessing_contract_fingerprint = TARGET_FULL_FRAME_PREPROCESSING.fingerprint

    def __init__(
        self,
        *,
        preprocessing_fingerprint: str = _PREPROCESSING_FINGERPRINT,
    ) -> None:
        self.preprocessing_fingerprint = preprocessing_fingerprint
        self.batches: list[tuple[DecodedImage, ...]] = []

    def encode_images(self, images: Sequence[DecodedImage]) -> list[list[float]]:
        batch = tuple(images)
        self.batches.append(batch)
        return [
            [float(image.data[0]), float(image.width), float(image.height)]
            for image in batch
        ]


def test_full_frame_plan_encodes_adult_and_larval_routes_once() -> None:
    image = _image(bytes(range(12)), width=2, height=2)
    loaded_photo_ids: list[str] = []

    def load(row: dict[str, Any]) -> DecodedImage:
        loaded_photo_ids.append(str(row["flickr_photo_id"]))
        return image

    rows = [
        _detection_row(
            "photo-1",
            "adult-1",
            route="adult_field",
            detection_route="adult_butterfly_field",
            bbox=(0.0, 0.0, 1.0, 2.0),
            mask_polygon_xyn=((0.0, 0.0), (0.5, 0.0), (0.5, 1.0)),
        ),
        _detection_row(
            "photo-1",
            "larva-1",
            route="larval",
            detection_route="caterpillar_field",
            bbox=(1.0, 0.0, 2.0, 2.0),
        ),
        _detection_row(
            "photo-excluded",
            "none-1",
            route=None,
            detection_route="no_relevant_organism",
            action="exclude",
            status="no_detection",
        ),
    ]

    plan = build_target_full_frame_plan(
        detection_rows=rows,
        image_loader=load,
    )

    assert TARGET_AWARE_VISUAL_MODE == "whole_image_reference_ensemble"
    assert RAW_FULL_IMAGE_KIND == "raw_full_image"
    assert plan.visual_mode == TARGET_AWARE_VISUAL_MODE
    assert loaded_photo_ids == ["photo-1"]
    assert len(plan.visual_inputs) == 1
    assert [unit.route for unit in plan.scoring_units] == ["adult_field", "larval"]
    adult, larval = plan.scoring_units
    assert adult.detection_ids == ("adult-1",)
    assert adult.detector_boxes_xyxy == ((0.0, 0.0, 1.0, 2.0),)
    assert adult.boxes_xyxyn == ((0.0, 0.0, 0.5, 1.0),)
    assert adult.full_frame_boxes_xyxy == ((0.0, 0.0, 1.0, 2.0),)
    assert adult.mask_polygons_xyn == (((0.0, 0.0), (0.5, 0.0), (0.5, 1.0)),)
    assert larval.detection_ids == ("larva-1",)
    assert adult.raw_visual_input_id == larval.raw_visual_input_id
    assert adult.scoring_unit_id != larval.scoring_unit_id
    assert adult.visual_mode == TARGET_AWARE_VISUAL_MODE
    encoder = RecordingEncoder()
    embedded = encode_target_full_frame_plan(
        plan,
        encoder=encoder,
        model_fingerprint=_MODEL_FINGERPRINT,
        preprocessing_fingerprint=_PREPROCESSING_FINGERPRINT,
    )

    assert encoder.batches == [(image,)]
    assert len(embedded.embeddings) == 1
    embedding = embedded.embeddings[0]
    assert embedding.embedding_version == TARGET_FULL_FRAME_EMBEDDING_VERSION
    assert embedding.embedding_version == "target-full-frame-embedding-v2"
    assert embedding.image_resize_mode == TARGET_FULL_FRAME_IMAGE_RESIZE_MODE
    assert (
        embedding.preprocessing_contract_fingerprint
        == TARGET_FULL_FRAME_PREPROCESSING.fingerprint
    )
    assert len(embedded.scoring_unit_references) == 2
    assert {reference.route for reference in embedded.scoring_unit_references} == {
        "adult_field",
        "larval",
    }
    assert (
        len({reference.embedding_id for reference in embedded.scoring_unit_references})
        == 1
    )


def test_embedding_identity_binds_target_preprocessing_contract(monkeypatch) -> None:
    kwargs = {
        "visual_input_id": "sha256:" + "a" * 64,
        "model_fingerprint": _MODEL_FINGERPRINT,
        "preprocessing_fingerprint": _PREPROCESSING_FINGERPRINT,
    }
    baseline = full_frame_embedding_id(**kwargs)

    monkeypatch.setattr(
        target_full_frame,
        "TARGET_FULL_FRAME_IMAGE_RESIZE_MODE",
        "shortest",
    )
    assert full_frame_embedding_id(**kwargs) != baseline

    monkeypatch.setattr(
        target_full_frame,
        "TARGET_FULL_FRAME_IMAGE_RESIZE_MODE",
        TARGET_FULL_FRAME_IMAGE_RESIZE_MODE,
    )
    monkeypatch.setattr(
        target_full_frame,
        "TARGET_FULL_FRAME_PREPROCESSING",
        replace(TARGET_FULL_FRAME_PREPROCESSING, version="changed-contract-v2"),
    )
    assert full_frame_embedding_id(**kwargs) != baseline


def test_visual_identity_ignores_detection_metadata_and_input_order() -> None:
    image = _image(b"\x07" * 12, width=2, height=2)
    original = [
        _detection_row(
            "photo-1", "det-b", bbox=(1.0, 0.0, 2.0, 2.0), crop_hash="legacy-b"
        ),
        _detection_row(
            "photo-1", "det-a", bbox=(0.0, 0.0, 1.0, 2.0), crop_hash="legacy-a"
        ),
    ]
    reordered = list(reversed(original))

    first = build_target_full_frame_plan(
        detection_rows=original, image_loader=lambda _row: image
    )
    second = build_target_full_frame_plan(
        detection_rows=reordered, image_loader=lambda _row: image
    )

    assert (
        first.visual_inputs[0].visual_input_id
        == second.visual_inputs[0].visual_input_id
    )
    assert (
        first.scoring_units[0].scoring_unit_id
        == second.scoring_units[0].scoring_unit_id
    )
    assert first.scoring_units[0].detection_ids == ("det-a", "det-b")

    changed = build_target_full_frame_plan(
        detection_rows=[original[0], original[1] | {"bbox_xyxy": [0.0, 0.0, 2.0, 2.0]}],
        image_loader=lambda _row: image,
    )
    assert (
        first.visual_inputs[0].visual_input_id
        == changed.visual_inputs[0].visual_input_id
    )
    assert (
        first.scoring_units[0].scoring_unit_id
        != changed.scoring_units[0].scoring_unit_id
    )


def test_content_and_embedding_identities_have_separate_invalidation() -> None:
    shared_bytes = b"\x03" * 12
    images = {
        "photo-a": _image(shared_bytes, width=2, height=2),
        "photo-b": _image(shared_bytes, width=2, height=2),
        "photo-c": _image(b"\x04" + shared_bytes[1:], width=2, height=2),
        "photo-d": _image(shared_bytes, width=1, height=4),
    }
    plan = build_target_full_frame_plan(
        detection_rows=[
            _detection_row(
                photo_id,
                f"det-{photo_id}",
                bbox=(0.0, 0.0, float(image.width), float(image.height)),
                bbox_canvas=(image.width, image.height),
            )
            for photo_id, image in images.items()
        ],
        image_loader=lambda row: images[str(row["flickr_photo_id"])],
    )

    assert len(plan.visual_inputs) == 3
    assert len(plan.scoring_units) == 4
    visual_ids = {
        unit.flickr_photo_id: unit.raw_visual_input_id for unit in plan.scoring_units
    }
    assert visual_ids["photo-a"] == visual_ids["photo-b"]
    assert visual_ids["photo-a"] != visual_ids["photo-c"]
    assert visual_ids["photo-a"] != visual_ids["photo-d"]

    first_encoder = RecordingEncoder()
    first = encode_target_full_frame_plan(
        plan,
        encoder=first_encoder,
        model_fingerprint=_MODEL_FINGERPRINT,
        preprocessing_fingerprint=_PREPROCESSING_FINGERPRINT,
    )
    second = encode_target_full_frame_plan(
        plan,
        encoder=RecordingEncoder(
            preprocessing_fingerprint=_PREPROCESSING_FINGERPRINT,
        ),
        model_fingerprint="sha256:" + "d" * 64,
        preprocessing_fingerprint=_PREPROCESSING_FINGERPRINT,
    )
    third = encode_target_full_frame_plan(
        plan,
        encoder=RecordingEncoder(
            preprocessing_fingerprint="sha256:" + "e" * 64,
        ),
        model_fingerprint=_MODEL_FINGERPRINT,
        preprocessing_fingerprint="sha256:" + "e" * 64,
    )

    assert len(first_encoder.batches) == 1
    assert len(first_encoder.batches[0]) == 3
    assert [item.visual_input_id for item in first.embeddings] == [
        item.visual_input_id for item in second.embeddings
    ]
    assert {item.embedding_id for item in first.embeddings}.isdisjoint(
        item.embedding_id for item in second.embeddings
    )
    assert {item.embedding_id for item in first.embeddings}.isdisjoint(
        item.embedding_id for item in third.embeddings
    )


def test_target_full_frame_plan_never_materializes_spatial_crops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_crop(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("target-aware full-frame mode must not crop")

    monkeypatch.setattr("biominer.detection.cropper.crop_with_padding", fail_crop)
    image = _image(b"\x05" * 12, width=2, height=2)
    plan = build_target_full_frame_plan(
        detection_rows=[
            _detection_row("photo-1", "det-b", bbox=(1.0, 0.0, 2.0, 2.0)),
            _detection_row("photo-1", "det-a", bbox=(0.0, 0.0, 1.0, 2.0)),
        ],
        image_loader=lambda _row: image,
    )

    assert len(plan.visual_inputs) == 1
    for output_type in (
        RawFullFrameEmbedding,
        RawFullFrameVisualInput,
        TargetFullFrameScoringUnit,
    ):
        output_fields = {field.name for field in fields(output_type)}
        assert not any("crop" in name for name in output_fields)
        assert not any("path" in name for name in output_fields)


def test_target_attention_adapter_preserves_raw_identity_and_detection_evidence() -> (
    None
):
    image = _image(b"\x05" * (4 * 4 * 3), width=4, height=4)
    plan = build_target_full_frame_plan(
        detection_rows=[
            _detection_row(
                "photo-1",
                "det-a",
                bbox=(0.0, 0.0, 1.0, 1.0),
                bbox_canvas=(2, 2),
                mask_polygon_xyn=(
                    (0.0, 0.0),
                    (0.5, 0.0),
                    (0.5, 0.5),
                    (0.0, 0.5),
                ),
            )
        ],
        image_loader=lambda _row: image,
    )
    unit = plan.scoring_units[0]

    result = generate_target_full_frame_attention_variants(
        plan,
        scoring_unit_id=unit.scoring_unit_id,
    )

    raw = next(
        variant
        for variant in result.variants
        if variant.visual_input_kind == RAW_FULL_IMAGE_KIND
    )
    assert raw.visual_input_id == unit.raw_visual_input_id
    assert {variant.visual_input_kind for variant in result.variants} == {
        RAW_FULL_IMAGE_KIND,
        FOCUSED_FULL_FRAME_KIND,
        MASKED_FULL_FRAME_KIND,
    }
    assert {evidence.source_detection_ids for evidence in result.evidence} == {
        ("det-a",)
    }
    assert all(evidence.route == "adult_field" for evidence in result.evidence)


def test_normalized_detector_box_projects_to_original_full_frame() -> None:
    image = _image(b"\x05" * (4 * 4 * 3), width=4, height=4)
    row = _detection_row(
        "photo-1",
        "det-a",
        bbox=(0.0, 0.0, 2.0, 2.0),
        bbox_canvas=(2, 2),
    )

    plan = build_target_full_frame_plan(
        detection_rows=[row],
        image_loader=lambda _row: image,
    )

    evidence = plan.scoring_units[0].detections[0]
    assert evidence.detector_bbox_xyxy == (0.0, 0.0, 2.0, 2.0)
    assert evidence.bbox_xyxyn == (0.0, 0.0, 1.0, 1.0)
    assert evidence.full_frame_bbox_xyxy == (0.0, 0.0, 4.0, 4.0)
    assert plan.visual_inputs[0].image == image


def test_empty_or_non_score_routes_do_not_load_or_encode() -> None:
    def fail_load(_row: dict[str, Any]) -> DecodedImage:
        raise AssertionError("excluded detections must not load an image")

    plan = build_target_full_frame_plan(
        detection_rows=[
            _detection_row(
                "photo-none",
                "none-1",
                route=None,
                detection_route="no_relevant_organism",
                action="exclude",
                status="no_detection",
            ),
            _detection_row(
                "photo-review",
                "review-1",
                route="adult_field",
                detection_route="ambiguous_visual_domain",
                action="review",
            ),
        ],
        image_loader=fail_load,
    )
    encoder = RecordingEncoder()
    embedded = encode_target_full_frame_plan(
        plan,
        encoder=encoder,
        model_fingerprint=_MODEL_FINGERPRINT,
        preprocessing_fingerprint=_PREPROCESSING_FINGERPRINT,
    )

    assert plan.visual_inputs == ()
    assert plan.scoring_units == ()
    assert encoder.batches == []
    assert embedded.embeddings == ()
    assert embedded.scoring_unit_references == ()


def test_target_full_frame_plan_rejects_source_and_route_identity_conflicts() -> None:
    image = _image(b"\x06" * 12, width=2, height=2)
    source_conflict = [
        _detection_row("photo-1", "det-a"),
        _detection_row("photo-1", "det-b")
        | {"source_record_hash": "sha256:" + "1" * 64},
    ]
    with pytest.raises(ValueError, match="conflicting per-photo source identity"):
        build_target_full_frame_plan(
            detection_rows=source_conflict,
            image_loader=lambda _row: image,
        )

    route_mismatch = _detection_row(
        "photo-1",
        "det-a",
        route="larval",
        detection_route="adult_butterfly_field",
    )
    with pytest.raises(ValueError, match="route mismatch"):
        build_target_full_frame_plan(
            detection_rows=[route_mismatch],
            image_loader=lambda _row: image,
        )

    invalid_policy = _detection_row("photo-1", "det-a") | {
        "routing_policy_fingerprint": "not-a-fingerprint"
    }
    with pytest.raises(ValueError, match="routing_policy_fingerprint"):
        build_target_full_frame_plan(
            detection_rows=[invalid_policy],
            image_loader=lambda _row: image,
        )

    mixed_routing_policy = [
        _detection_row("photo-1", "det-a"),
        _detection_row("photo-1", "det-b")
        | {"routing_policy_fingerprint": "sha256:" + "2" * 64},
    ]
    with pytest.raises(ValueError, match="routing_policy_fingerprint"):
        build_target_full_frame_plan(
            detection_rows=mixed_routing_policy,
            image_loader=lambda _row: image,
        )

    missing_source_hash = _detection_row("photo-1", "det-a")
    missing_source_hash.pop("source_record_hash")
    with pytest.raises(ValueError, match="source_record_hash"):
        build_target_full_frame_plan(
            detection_rows=[missing_source_hash],
            image_loader=lambda _row: image,
        )


@pytest.mark.parametrize("schema_version", [None, "object-detection-v1"])
def test_target_full_frame_plan_rejects_unsupported_detection_schema(
    schema_version: str | None,
) -> None:
    row = _detection_row("photo-1", "det-a")
    if schema_version is None:
        row.pop("schema_version")
    else:
        row["schema_version"] = schema_version

    with pytest.raises(ValueError, match="object-detection-v2"):
        build_target_full_frame_plan(
            detection_rows=[row],
            image_loader=lambda _row: _image(b"\x06" * 12, width=2, height=2),
        )


def test_target_full_frame_plan_rejects_mixed_detection_schema_versions() -> None:
    rows = [
        _detection_row("photo-1", "det-a"),
        _detection_row("photo-1", "det-b") | {"schema_version": "object-detection-v1"},
    ]

    with pytest.raises(ValueError, match="object-detection-v2"):
        build_target_full_frame_plan(
            detection_rows=rows,
            image_loader=lambda _row: _image(b"\x06" * 12, width=2, height=2),
        )

    mixed_with_review = [
        _detection_row("photo-1", "det-a"),
        _detection_row(
            "photo-1",
            "det-review",
            detection_route="ambiguous_visual_domain",
            route="adult_field",
            action="review",
        )
        | {"schema_version": "object-detection-v1"},
    ]
    with pytest.raises(ValueError, match="object-detection-v2"):
        build_target_full_frame_plan(
            detection_rows=mixed_with_review,
            image_loader=lambda _row: _image(b"\x06" * 12, width=2, height=2),
        )


@pytest.mark.parametrize(
    ("detection_route", "route"),
    [
        ("adult_butterfly_field", "adult_field"),
        ("caterpillar_field", "larval"),
        ("pinned_specimen", "pinned_specimen"),
    ],
)
def test_target_full_frame_plan_accepts_every_closed_score_route(
    detection_route: str,
    route: str,
) -> None:
    plan = build_target_full_frame_plan(
        detection_rows=[
            _detection_row(
                "photo-1",
                "det-a",
                detection_route=detection_route,
                route=route,
            )
        ],
        image_loader=lambda _row: _image(b"\x06" * 12, width=2, height=2),
    )

    assert plan.scoring_units[0].route == route


def test_target_full_frame_plan_requires_exact_visual_mode() -> None:
    image = _image(b"\x08" * 12, width=2, height=2)
    row = _detection_row("photo-1", "det-a")

    with pytest.raises(ValueError, match="visual_mode"):
        build_target_full_frame_plan(
            detection_rows=[row],
            image_loader=lambda _row: image,
            visual_mode="detector_crop",
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"routing_action": ""}, "unsupported routing_action"),
        (
            {"routing_action": "score", "detection_status": "no_detection"},
            "requires detection_status",
        ),
        (
            {
                "routing_action": "score",
                "detection_route": "no_relevant_organism",
            },
            "score-compatible detection_route",
        ),
        ({"routing_action": "score", "bioclip_route": None}, "requires bioclip_route"),
    ],
)
def test_target_full_frame_plan_rejects_malformed_score_rows(
    updates: dict[str, object],
    message: str,
) -> None:
    row = _detection_row("photo-1", "det-a") | updates

    with pytest.raises(ValueError, match=message):
        build_target_full_frame_plan(
            detection_rows=[row],
            image_loader=lambda _row: _image(b"\x08" * 12, width=2, height=2),
        )


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([], "returned 0 vectors"),
        ([[], [1.0]], "empty embedding"),
        ([[float("nan")], [1.0]], "non-finite embedding"),
        ([[0.0], [1.0]], "zero-norm embedding"),
        (
            [[1e308, 1e308, 1e308, 1e308], [1.0, 1.0, 1.0, 1.0]],
            "non-finite embedding norm",
        ),
        ([[1.0], [1.0, 2.0]], "mixed embedding dimensions"),
    ],
)
def test_full_frame_encoder_rejects_invalid_outputs(
    vectors: list[list[float]],
    message: str,
) -> None:
    images = {
        "photo-a": _image(b"\x01" * 12, width=2, height=2),
        "photo-b": _image(b"\x02" * 12, width=2, height=2),
    }
    plan = build_target_full_frame_plan(
        detection_rows=[
            _detection_row(photo_id, f"det-{photo_id}") for photo_id in images
        ],
        image_loader=lambda row: images[str(row["flickr_photo_id"])],
    )

    class StaticEncoder:
        image_resize_mode = TARGET_FULL_FRAME_IMAGE_RESIZE_MODE
        preprocessing_contract_fingerprint = TARGET_FULL_FRAME_PREPROCESSING.fingerprint
        preprocessing_fingerprint = _PREPROCESSING_FINGERPRINT

        def encode_images(
            self,
            _images: Sequence[DecodedImage],
        ) -> list[list[float]]:
            return vectors

    with pytest.raises(ValueError, match=message):
        encode_target_full_frame_plan(
            plan,
            encoder=StaticEncoder(),
            model_fingerprint=_MODEL_FINGERPRINT,
            preprocessing_fingerprint=_PREPROCESSING_FINGERPRINT,
        )


def test_full_frame_encoder_rejects_invalid_identity_fingerprints() -> None:
    image = _image(b"\x09" * 12, width=2, height=2)
    plan = build_target_full_frame_plan(
        detection_rows=[_detection_row("photo-1", "det-a")],
        image_loader=lambda _row: image,
    )

    with pytest.raises(ValueError, match="model_fingerprint"):
        encode_target_full_frame_plan(
            plan,
            encoder=RecordingEncoder(),
            model_fingerprint="model-v1",
            preprocessing_fingerprint=_PREPROCESSING_FINGERPRINT,
        )
    with pytest.raises(ValueError, match="preprocessing_fingerprint"):
        encode_target_full_frame_plan(
            plan,
            encoder=RecordingEncoder(),
            model_fingerprint=_MODEL_FINGERPRINT,
            preprocessing_fingerprint="preprocess-v1",
        )


def test_full_frame_encoder_requires_bound_longest_side_preprocessing() -> None:
    plan = build_target_full_frame_plan(
        detection_rows=[_detection_row("photo-1", "det-a")],
        image_loader=lambda _row: _image(b"\x09" * 12, width=2, height=2),
    )

    wrong_mode = RecordingEncoder()
    wrong_mode.image_resize_mode = "shortest"
    with pytest.raises(ValueError, match="image_resize_mode"):
        encode_target_full_frame_plan(
            plan,
            encoder=wrong_mode,
            model_fingerprint=_MODEL_FINGERPRINT,
            preprocessing_fingerprint=_PREPROCESSING_FINGERPRINT,
        )
    assert wrong_mode.batches == []

    wrong_contract = RecordingEncoder()
    wrong_contract.preprocessing_contract_fingerprint = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="contract fingerprint mismatch"):
        encode_target_full_frame_plan(
            plan,
            encoder=wrong_contract,
            model_fingerprint=_MODEL_FINGERPRINT,
            preprocessing_fingerprint=_PREPROCESSING_FINGERPRINT,
        )
    assert wrong_contract.batches == []

    wrong_runtime_fingerprint = RecordingEncoder()
    with pytest.raises(ValueError, match="preprocessing fingerprint mismatch"):
        encode_target_full_frame_plan(
            plan,
            encoder=wrong_runtime_fingerprint,
            model_fingerprint=_MODEL_FINGERPRINT,
            preprocessing_fingerprint="sha256:" + "1" * 64,
        )
    assert wrong_runtime_fingerprint.batches == []


def test_full_frame_encoder_rejects_stale_or_duplicate_plan_identity() -> None:
    image = _image(b"\x09" * 12, width=2, height=2)
    plan = build_target_full_frame_plan(
        detection_rows=[_detection_row("photo-1", "det-a")],
        image_loader=lambda _row: image,
    )
    visual_input = plan.visual_inputs[0]
    stale_plan = replace(
        plan,
        visual_inputs=(
            replace(
                visual_input,
                image=_image(b"\x0a" * 12, width=2, height=2),
            ),
        ),
    )
    encoder = RecordingEncoder()

    with pytest.raises(ValueError, match="stale raw visual-input identity"):
        encode_target_full_frame_plan(
            stale_plan,
            encoder=encoder,
            model_fingerprint=_MODEL_FINGERPRINT,
            preprocessing_fingerprint=_PREPROCESSING_FINGERPRINT,
        )
    assert encoder.batches == []

    duplicate_plan = replace(
        plan,
        visual_inputs=(visual_input, visual_input),
    )
    with pytest.raises(ValueError, match="duplicate visual_input_id"):
        encode_target_full_frame_plan(
            duplicate_plan,
            encoder=RecordingEncoder(),
            model_fingerprint=_MODEL_FINGERPRINT,
            preprocessing_fingerprint=_PREPROCESSING_FINGERPRINT,
        )


def test_full_frame_encoder_rejects_stale_scoring_identity() -> None:
    image = _image(b"\x0b" * 12, width=2, height=2)
    plan = build_target_full_frame_plan(
        detection_rows=[
            _detection_row("photo-1", "det-a", bbox=(0.0, 0.0, 1.0, 2.0)),
            _detection_row("photo-1", "det-b", bbox=(1.0, 0.0, 2.0, 2.0)),
        ],
        image_loader=lambda _row: image,
    )
    unit = plan.scoring_units[0]

    stale_route = replace(plan, scoring_units=(replace(unit, route="larval"),))
    with pytest.raises(ValueError, match="detection from another route"):
        encode_target_full_frame_plan(
            stale_route,
            encoder=RecordingEncoder(),
            model_fingerprint=_MODEL_FINGERPRINT,
            preprocessing_fingerprint=_PREPROCESSING_FINGERPRINT,
        )


def test_target_mode_disables_crop_metadata_without_changing_other_run_policy() -> None:
    base = DetectionRunPolicy(
        download_workers=7,
        decode_workers=6,
        detector_workers=2,
        max_inflight_images=19,
        max_inflight_crops=23,
        detector_batch_size=5,
        crop_batch_size=11,
        parquet_batch_rows=101,
        adaptive_batching=True,
        min_detector_batch_size=2,
        create_crop_metadata=True,
    )

    policy = target_full_frame_detection_run_policy(base)

    assert policy.create_crop_metadata is False
    assert policy == DetectionRunPolicy(
        download_workers=7,
        decode_workers=6,
        detector_workers=2,
        max_inflight_images=19,
        max_inflight_crops=23,
        detector_batch_size=5,
        crop_batch_size=11,
        parquet_batch_rows=101,
        adaptive_batching=True,
        min_detector_batch_size=2,
        create_crop_metadata=False,
    )


def _image(data: bytes, *, width: int, height: int) -> DecodedImage:
    return DecodedImage(width=width, height=height, mode="RGB", data=data)


def _detection_row(
    photo_id: str,
    detection_id: str,
    *,
    route: str | None = "adult_field",
    detection_route: str = "adult_butterfly_field",
    action: str = "score",
    status: str = "detected",
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 2.0, 2.0),
    bbox_canvas: tuple[int, int] = (2, 2),
    mask_polygon_xyn: tuple[tuple[float, float], ...] | None = None,
    crop_hash: str | None = None,
) -> dict[str, Any]:
    return {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "source_record_hash": "sha256:" + "f" * 64,
        "detection_id": detection_id,
        "detection_status": status,
        "detector_score": 0.9,
        "detector_label": "butterfly_like",
        "bbox_xyxy": list(bbox),
        "bbox_xyxyn": [
            bbox[0] / bbox_canvas[0],
            bbox[1] / bbox_canvas[1],
            bbox[2] / bbox_canvas[0],
            bbox[3] / bbox_canvas[1],
        ],
        "mask_polygon_xyn": mask_polygon_xyn,
        "detection_route": detection_route,
        "routing_action": action,
        "bioclip_route": route,
        "routing_policy_version": "detection-routing-policy-v1",
        "routing_policy_fingerprint": _ROUTING_POLICY_FINGERPRINT,
        "crop_hash": crop_hash,
        "crop_path": "/legacy/must/not/be/read.ppm",
        "schema_version": "object-detection-v2",
    }
