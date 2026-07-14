from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from typing import Any, Protocol

from biominer.detection.detector_base import (
    DecodedImage,
    normalize_mask_polygon_xyn,
)
from biominer.detection.policy import DetectionRunPolicy
from biominer.detection.schema import DETECTION_SCHEMA_VERSION
from biominer.vision.gates import (
    COMPARISON_ROUTE_BY_DETECTION_ROUTE,
    SUPPORTED_COMPARISON_ROUTES,
)
from biominer.vision.full_frame_attention import (
    FULL_FRAME_VISUAL_INPUT_VERSION,
    RAW_FULL_IMAGE_KIND,
    RAW_FULL_IMAGE_TRANSFORMATION_FINGERPRINT,
    TARGET_FULL_FRAME_IMAGE_RESIZE_MODE,
    TARGET_FULL_FRAME_PREPROCESSING,
    AttentionQualityPolicy,
    AttentionRegion,
    FullFrameAttentionResult,
    FullFrameTransformPolicy,
    generate_full_frame_attention_variants,
    raw_full_frame_visual_input,
)


TARGET_AWARE_VISUAL_MODE = "whole_image_reference_ensemble"
TARGET_FULL_FRAME_VISUAL_INPUT_VERSION = FULL_FRAME_VISUAL_INPUT_VERSION
TARGET_FULL_FRAME_SCORING_UNIT_VERSION = "target-full-frame-scoring-unit-v2"
TARGET_FULL_FRAME_EMBEDDING_VERSION = "target-full-frame-embedding-v2"
TARGET_FULL_FRAME_REQUIRES_CROP_METADATA = False
TARGET_FULL_FRAME_MATERIALIZES_CROP_FILES = False

_SHA256_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}\Z")


class FullFrameImageEncoder(Protocol):
    image_resize_mode: str
    preprocessing_contract_fingerprint: str
    preprocessing_fingerprint: str

    def encode_images(
        self,
        images: Sequence[DecodedImage],
    ) -> Sequence[Sequence[float]]:
        """Encode every supplied full image exactly once."""


@dataclass(frozen=True)
class RoutedDetectionEvidence:
    detection_id: str
    detection_route: str
    route: str
    detector_bbox_xyxy: tuple[float, float, float, float]
    bbox_xyxyn: tuple[float, float, float, float]
    full_frame_bbox_xyxy: tuple[float, float, float, float]
    mask_polygon_xyn: tuple[tuple[float, float], ...] | None
    detector_score: float | None
    detector_prompt: str | None
    detector_class_id: int | None
    detector_prompt_set_fingerprint: str | None
    routing_policy_version: str
    routing_policy_fingerprint: str


@dataclass(frozen=True)
class RawFullFrameVisualInput:
    visual_input_id: str
    visual_input_kind: str
    visual_input_version: str
    raw_image_content_hash: str
    transformation_fingerprint: str
    width: int
    height: int
    mode: str
    image: DecodedImage


@dataclass(frozen=True)
class TargetFullFrameScoringUnit:
    scoring_unit_id: str
    scoring_unit_version: str
    visual_mode: str
    source: str
    flickr_photo_id: str
    source_record_hash: str
    route: str
    raw_visual_input_id: str
    detections: tuple[RoutedDetectionEvidence, ...]

    @property
    def detection_ids(self) -> tuple[str, ...]:
        return tuple(detection.detection_id for detection in self.detections)

    @property
    def detector_boxes_xyxy(
        self,
    ) -> tuple[tuple[float, float, float, float], ...]:
        return tuple(detection.detector_bbox_xyxy for detection in self.detections)

    @property
    def boxes_xyxyn(self) -> tuple[tuple[float, float, float, float], ...]:
        return tuple(detection.bbox_xyxyn for detection in self.detections)

    @property
    def full_frame_boxes_xyxy(
        self,
    ) -> tuple[tuple[float, float, float, float], ...]:
        return tuple(detection.full_frame_bbox_xyxy for detection in self.detections)

    @property
    def mask_polygons_xyn(
        self,
    ) -> tuple[tuple[tuple[float, float], ...] | None, ...]:
        return tuple(detection.mask_polygon_xyn for detection in self.detections)


@dataclass(frozen=True)
class TargetFullFramePlan:
    visual_mode: str
    visual_inputs: tuple[RawFullFrameVisualInput, ...]
    scoring_units: tuple[TargetFullFrameScoringUnit, ...]


@dataclass(frozen=True)
class RawFullFrameEmbedding:
    embedding_id: str
    embedding_version: str
    embedding_fingerprint: str
    visual_input_id: str
    visual_input_kind: str
    raw_image_content_hash: str
    transformation_fingerprint: str
    model_fingerprint: str
    image_resize_mode: str
    preprocessing_contract_fingerprint: str
    preprocessing_fingerprint: str
    embedding_dimension: int
    embedding: tuple[float, ...]
    embedding_norm: float


@dataclass(frozen=True)
class ScoringUnitEmbeddingReference:
    scoring_unit_id: str
    source: str
    flickr_photo_id: str
    route: str
    raw_visual_input_id: str
    embedding_id: str


@dataclass(frozen=True)
class EmbeddedTargetFullFramePlan:
    embeddings: tuple[RawFullFrameEmbedding, ...]
    scoring_unit_references: tuple[ScoringUnitEmbeddingReference, ...]


def target_full_frame_detection_run_policy(
    base: DetectionRunPolicy | None = None,
) -> DetectionRunPolicy:
    """Bind target full-frame detection to the no-crop-metadata policy."""

    return replace(base or DetectionRunPolicy(), create_crop_metadata=False)


def build_target_full_frame_plan(
    *,
    detection_rows: Iterable[Mapping[str, Any]],
    image_loader: Callable[[dict[str, Any]], DecodedImage],
    visual_mode: str = TARGET_AWARE_VISUAL_MODE,
) -> TargetFullFramePlan:
    """Build photo/route scoring units without constructing spatial crops."""

    _require_target_full_frame_mode(visual_mode)
    rows_by_photo: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for raw_row in detection_rows:
        row = dict(raw_row)
        if str(row.get("schema_version") or "") != DETECTION_SCHEMA_VERSION:
            raise ValueError(
                "target full-frame scoring requires detection schema "
                f"{DETECTION_SCHEMA_VERSION!r}"
            )
        if not _is_score_eligible(row):
            continue
        source = _required_string(row.get("source"), "source")
        photo_id = _required_string(
            row.get("flickr_photo_id"),
            "flickr_photo_id",
        )
        rows_by_photo.setdefault((source, photo_id), []).append(row)

    visual_inputs_by_id: dict[str, RawFullFrameVisualInput] = {}
    scoring_units: list[TargetFullFrameScoringUnit] = []
    for (source, photo_id), photo_rows in sorted(rows_by_photo.items()):
        source_record_hash = _consistent_required_string(
            photo_rows,
            field_name="source_record_hash",
            source=source,
            photo_id=photo_id,
        )
        _require_sha256(source_record_hash, "source_record_hash")
        _consistent_required_string(
            photo_rows,
            field_name="routing_policy_version",
            source=source,
            photo_id=photo_id,
        )
        routing_policy_fingerprint = _consistent_required_string(
            photo_rows,
            field_name="routing_policy_fingerprint",
            source=source,
            photo_id=photo_id,
        )
        _require_sha256(
            routing_policy_fingerprint,
            "routing_policy_fingerprint",
        )
        _consistent_optional_string(
            photo_rows,
            field_name="image_url",
            source=source,
            photo_id=photo_id,
        )
        representative = min(
            photo_rows,
            key=lambda row: (
                str(row.get("detection_id") or ""),
                str(row.get("bioclip_route") or ""),
            ),
        )
        image = image_loader(dict(representative))
        if not isinstance(image, DecodedImage):
            raise TypeError("image_loader must return a DecodedImage")
        visual_input = _raw_full_frame_visual_input(image)
        existing_visual_input = visual_inputs_by_id.get(visual_input.visual_input_id)
        if existing_visual_input is None:
            visual_inputs_by_id[visual_input.visual_input_id] = visual_input
        elif not _same_decoded_content(existing_visual_input, visual_input):
            raise RuntimeError(
                "raw full-image visual-input identity collision for "
                f"{visual_input.visual_input_id}"
            )

        evidence_by_id: dict[str, RoutedDetectionEvidence] = {}
        for row in photo_rows:
            evidence = _routed_detection_evidence(
                row,
                image=image,
            )
            existing = evidence_by_id.get(evidence.detection_id)
            if existing is not None and existing != evidence:
                raise ValueError(
                    "conflicting routed detection rows for "
                    f"source={source!r}, flickr_photo_id={photo_id!r}, "
                    f"detection_id={evidence.detection_id!r}"
                )
            evidence_by_id[evidence.detection_id] = evidence
        photo_evidence = tuple(
            sorted(evidence_by_id.values(), key=lambda item: item.detection_id)
        )
        route_groups: dict[str, list[RoutedDetectionEvidence]] = {}
        for evidence in photo_evidence:
            route_groups.setdefault(evidence.route, []).append(evidence)
        for route, route_evidence_values in sorted(route_groups.items()):
            route_evidence = tuple(route_evidence_values)
            scoring_unit_id = _scoring_unit_id(
                source=source,
                photo_id=photo_id,
                source_record_hash=source_record_hash,
                route=route,
                raw_visual_input_id=visual_input.visual_input_id,
                detections=route_evidence,
            )
            scoring_units.append(
                TargetFullFrameScoringUnit(
                    scoring_unit_id=scoring_unit_id,
                    scoring_unit_version=TARGET_FULL_FRAME_SCORING_UNIT_VERSION,
                    visual_mode=visual_mode,
                    source=source,
                    flickr_photo_id=photo_id,
                    source_record_hash=source_record_hash,
                    route=route,
                    raw_visual_input_id=visual_input.visual_input_id,
                    detections=route_evidence,
                )
            )

    return TargetFullFramePlan(
        visual_mode=visual_mode,
        visual_inputs=tuple(
            sorted(
                visual_inputs_by_id.values(),
                key=lambda item: item.visual_input_id,
            )
        ),
        scoring_units=tuple(
            sorted(
                scoring_units,
                key=lambda item: (
                    item.source,
                    item.flickr_photo_id,
                    item.route,
                    item.scoring_unit_id,
                ),
            )
        ),
    )


def generate_target_full_frame_attention_variants(
    plan: TargetFullFramePlan,
    *,
    scoring_unit_id: str,
    transform_policy: FullFrameTransformPolicy | None = None,
    quality_policy: AttentionQualityPolicy | None = None,
) -> FullFrameAttentionResult:
    """Materialize one route unit with the shared reference/query transformer."""

    _validate_target_full_frame_plan(plan)
    scoring_unit_id = _required_string(scoring_unit_id, "scoring_unit_id")
    matching_units = tuple(
        unit for unit in plan.scoring_units if unit.scoring_unit_id == scoring_unit_id
    )
    if len(matching_units) != 1:
        raise KeyError(scoring_unit_id)
    unit = matching_units[0]
    visual_input_by_id = {
        visual_input.visual_input_id: visual_input
        for visual_input in plan.visual_inputs
    }
    raw_visual_input = visual_input_by_id[unit.raw_visual_input_id]
    result = generate_full_frame_attention_variants(
        raw_visual_input.image,
        tuple(
            AttentionRegion(
                source_detection_id=detection.detection_id,
                route=detection.route,
                bbox_xyxyn=detection.bbox_xyxyn,
                mask_polygon_xyn=detection.mask_polygon_xyn,
                detector_score=detection.detector_score,
            )
            for detection in unit.detections
        ),
        source_type=unit.source,
        source_record_id=unit.source_record_hash,
        transform_policy=transform_policy,
        quality_policy=quality_policy,
    )
    raw_variants = tuple(
        variant
        for variant in result.variants
        if variant.visual_input_kind == RAW_FULL_IMAGE_KIND
    )
    if (
        len(raw_variants) != 1
        or raw_variants[0].visual_input_id != unit.raw_visual_input_id
    ):
        raise RuntimeError(
            "full-frame attention generation did not preserve raw visual identity"
        )
    return result


def encode_target_full_frame_plan(
    plan: TargetFullFramePlan,
    *,
    encoder: FullFrameImageEncoder,
    model_fingerprint: str,
    preprocessing_fingerprint: str,
) -> EmbeddedTargetFullFramePlan:
    """Encode each unique raw input once and fan references out to route units."""

    _validate_target_full_frame_plan(plan)
    _require_sha256(model_fingerprint, "model_fingerprint")
    _require_sha256(
        preprocessing_fingerprint,
        "preprocessing_fingerprint",
    )
    if (
        getattr(encoder, "image_resize_mode", None)
        != TARGET_FULL_FRAME_IMAGE_RESIZE_MODE
    ):
        raise ValueError(
            "target full-frame encoder requires image_resize_mode "
            f"{TARGET_FULL_FRAME_IMAGE_RESIZE_MODE!r}"
        )
    if (
        getattr(encoder, "preprocessing_contract_fingerprint", None)
        != TARGET_FULL_FRAME_PREPROCESSING.fingerprint
    ):
        raise ValueError(
            "target full-frame encoder preprocessing contract fingerprint mismatch"
        )
    if getattr(encoder, "preprocessing_fingerprint", None) != preprocessing_fingerprint:
        raise ValueError("target full-frame encoder preprocessing fingerprint mismatch")
    if not plan.visual_inputs:
        if plan.scoring_units:
            raise ValueError(
                "target full-frame plan has scoring units but no visual inputs"
            )
        return EmbeddedTargetFullFramePlan(embeddings=(), scoring_unit_references=())

    raw_vectors = tuple(
        encoder.encode_images(
            tuple(visual_input.image for visual_input in plan.visual_inputs)
        )
    )
    if len(raw_vectors) != len(plan.visual_inputs):
        raise ValueError(
            "full-frame encoder returned "
            f"{len(raw_vectors)} vectors for {len(plan.visual_inputs)} images"
        )

    embeddings: list[RawFullFrameEmbedding] = []
    expected_dimension: int | None = None
    embedding_id_by_visual_input: dict[str, str] = {}
    for visual_input, raw_vector in zip(
        plan.visual_inputs,
        raw_vectors,
        strict=True,
    ):
        vector = _validated_embedding(raw_vector)
        if expected_dimension is None:
            expected_dimension = len(vector)
        elif len(vector) != expected_dimension:
            raise ValueError(
                "full-frame encoder returned mixed embedding dimensions: "
                f"expected {expected_dimension}, got {len(vector)}"
            )
        embedding_id = full_frame_embedding_id(
            visual_input_id=visual_input.visual_input_id,
            model_fingerprint=model_fingerprint,
            preprocessing_fingerprint=preprocessing_fingerprint,
        )
        embedding_id_by_visual_input[visual_input.visual_input_id] = embedding_id
        embedding_norm = math.hypot(*vector)
        if not math.isfinite(embedding_norm):
            raise ValueError("full-frame encoder returned a non-finite embedding norm")
        if embedding_norm == 0.0:
            raise ValueError("full-frame encoder returned a zero-norm embedding")
        embedding_fingerprint = _identity(
            {
                "embedding": vector,
                "embedding_id": embedding_id,
                "embedding_version": TARGET_FULL_FRAME_EMBEDDING_VERSION,
            }
        )
        embeddings.append(
            RawFullFrameEmbedding(
                embedding_id=embedding_id,
                embedding_version=TARGET_FULL_FRAME_EMBEDDING_VERSION,
                embedding_fingerprint=embedding_fingerprint,
                visual_input_id=visual_input.visual_input_id,
                visual_input_kind=RAW_FULL_IMAGE_KIND,
                raw_image_content_hash=visual_input.raw_image_content_hash,
                transformation_fingerprint=visual_input.transformation_fingerprint,
                model_fingerprint=model_fingerprint,
                image_resize_mode=TARGET_FULL_FRAME_IMAGE_RESIZE_MODE,
                preprocessing_contract_fingerprint=(
                    TARGET_FULL_FRAME_PREPROCESSING.fingerprint
                ),
                preprocessing_fingerprint=preprocessing_fingerprint,
                embedding_dimension=len(vector),
                embedding=vector,
                embedding_norm=embedding_norm,
            )
        )

    references: list[ScoringUnitEmbeddingReference] = []
    for unit in plan.scoring_units:
        try:
            embedding_id = embedding_id_by_visual_input[unit.raw_visual_input_id]
        except KeyError as exc:
            raise ValueError(
                "target full-frame scoring unit references an unknown visual input: "
                f"{unit.raw_visual_input_id}"
            ) from exc
        references.append(
            ScoringUnitEmbeddingReference(
                scoring_unit_id=unit.scoring_unit_id,
                source=unit.source,
                flickr_photo_id=unit.flickr_photo_id,
                route=unit.route,
                raw_visual_input_id=unit.raw_visual_input_id,
                embedding_id=embedding_id,
            )
        )
    return EmbeddedTargetFullFramePlan(
        embeddings=tuple(embeddings),
        scoring_unit_references=tuple(references),
    )


def _validate_target_full_frame_plan(plan: TargetFullFramePlan) -> None:
    _require_target_full_frame_mode(plan.visual_mode)
    visual_input_by_id: dict[str, RawFullFrameVisualInput] = {}
    for visual_input in plan.visual_inputs:
        if visual_input.visual_input_id in visual_input_by_id:
            raise ValueError(
                "target full-frame plan contains duplicate visual_input_id "
                f"{visual_input.visual_input_id!r}"
            )
        expected = _raw_full_frame_visual_input(visual_input.image)
        if not _same_visual_input_identity(visual_input, expected):
            raise ValueError(
                "target full-frame plan contains stale raw visual-input identity "
                f"{visual_input.visual_input_id!r}"
            )
        visual_input_by_id[visual_input.visual_input_id] = visual_input

    scoring_unit_ids: set[str] = set()
    photo_routes: set[tuple[str, str, str]] = set()
    photo_identity: dict[tuple[str, str], tuple[str, str]] = {}
    for unit in plan.scoring_units:
        if unit.scoring_unit_id in scoring_unit_ids:
            raise ValueError(
                "target full-frame plan contains duplicate scoring_unit_id "
                f"{unit.scoring_unit_id!r}"
            )
        scoring_unit_ids.add(unit.scoring_unit_id)
        photo_route = (unit.source, unit.flickr_photo_id, unit.route)
        if photo_route in photo_routes:
            raise ValueError(
                "target full-frame plan contains duplicate photo/route scoring unit"
            )
        photo_routes.add(photo_route)
        if unit.visual_mode != plan.visual_mode:
            raise ValueError("target full-frame scoring unit visual_mode mismatch")
        if unit.scoring_unit_version != TARGET_FULL_FRAME_SCORING_UNIT_VERSION:
            raise ValueError("unsupported target full-frame scoring_unit_version")
        if unit.raw_visual_input_id not in visual_input_by_id:
            raise ValueError(
                "target full-frame scoring unit references an unknown visual input: "
                f"{unit.raw_visual_input_id}"
            )
        if not unit.detections:
            raise ValueError("target full-frame scoring unit must retain detections")
        detection_ids = tuple(detection.detection_id for detection in unit.detections)
        if detection_ids != tuple(sorted(set(detection_ids))):
            raise ValueError(
                "target full-frame scoring unit detections must have unique, "
                "sorted detection IDs"
            )
        if any(detection.route != unit.route for detection in unit.detections):
            raise ValueError(
                "target full-frame scoring unit contains a detection from another route"
            )
        expected_id = _scoring_unit_id(
            source=unit.source,
            photo_id=unit.flickr_photo_id,
            source_record_hash=unit.source_record_hash,
            route=unit.route,
            raw_visual_input_id=unit.raw_visual_input_id,
            detections=unit.detections,
        )
        if unit.scoring_unit_id != expected_id:
            raise ValueError(
                "target full-frame plan contains stale scoring-unit identity "
                f"{unit.scoring_unit_id!r}"
            )
        key = (unit.source, unit.flickr_photo_id)
        identity = (unit.source_record_hash, unit.raw_visual_input_id)
        previous_identity = photo_identity.setdefault(key, identity)
        if previous_identity != identity:
            raise ValueError(
                "target full-frame plan contains conflicting per-photo identity"
            )


def _same_visual_input_identity(
    actual: RawFullFrameVisualInput,
    expected: RawFullFrameVisualInput,
) -> bool:
    return (
        actual.visual_input_id == expected.visual_input_id
        and actual.visual_input_kind == expected.visual_input_kind
        and actual.visual_input_version == expected.visual_input_version
        and actual.raw_image_content_hash == expected.raw_image_content_hash
        and actual.transformation_fingerprint == expected.transformation_fingerprint
        and actual.width == expected.width
        and actual.height == expected.height
        and actual.mode == expected.mode
        and actual.image.width == expected.image.width
        and actual.image.height == expected.image.height
        and actual.image.mode == expected.image.mode
        and actual.image.data == expected.image.data
    )


def full_frame_embedding_id(
    *,
    visual_input_id: str,
    model_fingerprint: str,
    preprocessing_fingerprint: str,
) -> str:
    _require_sha256(visual_input_id, "visual_input_id")
    _require_sha256(model_fingerprint, "model_fingerprint")
    _require_sha256(
        preprocessing_fingerprint,
        "preprocessing_fingerprint",
    )
    return _identity(
        {
            "embedding_version": TARGET_FULL_FRAME_EMBEDDING_VERSION,
            "image_resize_mode": TARGET_FULL_FRAME_IMAGE_RESIZE_MODE,
            "model_fingerprint": model_fingerprint,
            "preprocessing_contract_fingerprint": (
                TARGET_FULL_FRAME_PREPROCESSING.fingerprint
            ),
            "preprocessing_fingerprint": preprocessing_fingerprint,
            "visual_input_id": visual_input_id,
        }
    )


def _is_score_eligible(row: Mapping[str, Any]) -> bool:
    action = str(row.get("routing_action") or "").strip()
    if action in {"review", "exclude"}:
        return False
    if action != "score":
        raise ValueError(f"unsupported routing_action {action!r}")
    status = str(row.get("detection_status") or "").strip()
    if status != "detected":
        raise ValueError("routing_action='score' requires detection_status='detected'")
    detection_route = str(row.get("detection_route") or "").strip()
    if not detection_route or detection_route == "no_relevant_organism":
        raise ValueError(
            "routing_action='score' requires a score-compatible detection_route"
        )
    if not str(row.get("bioclip_route") or "").strip():
        raise ValueError("routing_action='score' requires bioclip_route")
    return True


def _routed_detection_evidence(
    row: Mapping[str, Any],
    *,
    image: DecodedImage,
) -> RoutedDetectionEvidence:
    detection_id = _required_string(row.get("detection_id"), "detection_id")
    detection_route = _required_string(
        row.get("detection_route"),
        "detection_route",
    )
    route = _required_string(row.get("bioclip_route"), "bioclip_route")
    if route not in SUPPORTED_COMPARISON_ROUTES:
        raise ValueError(f"unsupported BioCLIP route {route!r}")
    expected_route = COMPARISON_ROUTE_BY_DETECTION_ROUTE.get(detection_route)
    if expected_route != route:
        raise ValueError(
            "detection/comparison route mismatch: "
            f"{detection_route!r} requires {expected_route!r}, got {route!r}"
        )

    detector_bbox_xyxy = _positive_bbox(
        row.get("bbox_xyxy"),
        name="bbox_xyxy",
    )
    raw_xyxyn = row.get("bbox_xyxyn")
    if raw_xyxyn is None or raw_xyxyn == [] or raw_xyxyn == ():
        raise ValueError(
            "target full-frame scoring requires normalized bbox_xyxyn from "
            f"{DETECTION_SCHEMA_VERSION}"
        )
    bbox_xyxyn = _normalized_bbox(raw_xyxyn, name="bbox_xyxyn")
    full_frame_bbox_xyxy = (
        bbox_xyxyn[0] * image.width,
        bbox_xyxyn[1] * image.height,
        bbox_xyxyn[2] * image.width,
        bbox_xyxyn[3] * image.height,
    )

    detector_score = _optional_finite_float(
        row.get("detector_score"),
        "detector_score",
    )
    detector_class_id = _optional_non_negative_int(
        row.get("detector_class_id"),
        "detector_class_id",
    )
    prompt_set_fingerprint = _optional_string(
        row.get("detector_prompt_set_fingerprint")
    )
    if prompt_set_fingerprint is not None:
        _require_sha256(
            prompt_set_fingerprint,
            "detector_prompt_set_fingerprint",
        )
    routing_policy_fingerprint = _required_string(
        row.get("routing_policy_fingerprint"),
        "routing_policy_fingerprint",
    )
    _require_sha256(
        routing_policy_fingerprint,
        "routing_policy_fingerprint",
    )
    return RoutedDetectionEvidence(
        detection_id=detection_id,
        detection_route=detection_route,
        route=route,
        detector_bbox_xyxy=detector_bbox_xyxy,
        bbox_xyxyn=bbox_xyxyn,
        full_frame_bbox_xyxy=full_frame_bbox_xyxy,
        mask_polygon_xyn=normalize_mask_polygon_xyn(row.get("mask_polygon_xyn")),
        detector_score=detector_score,
        detector_prompt=_optional_string(row.get("detector_prompt")),
        detector_class_id=detector_class_id,
        detector_prompt_set_fingerprint=prompt_set_fingerprint,
        routing_policy_version=_required_string(
            row.get("routing_policy_version"),
            "routing_policy_version",
        ),
        routing_policy_fingerprint=routing_policy_fingerprint,
    )


def _raw_full_frame_visual_input(
    image: DecodedImage,
) -> RawFullFrameVisualInput:
    authoritative = raw_full_frame_visual_input(image)
    return RawFullFrameVisualInput(
        visual_input_id=authoritative.visual_input_id,
        visual_input_kind=authoritative.visual_input_kind,
        visual_input_version=authoritative.visual_input_version,
        raw_image_content_hash=authoritative.raw_image_content_hash,
        transformation_fingerprint=authoritative.transformation_fingerprint,
        width=image.width,
        height=image.height,
        mode=image.mode,
        image=image,
    )


def _scoring_unit_id(
    *,
    source: str,
    photo_id: str,
    source_record_hash: str,
    route: str,
    raw_visual_input_id: str,
    detections: tuple[RoutedDetectionEvidence, ...],
) -> str:
    return _identity(
        {
            "detections": [
                {
                    "detector_bbox_xyxy": detection.detector_bbox_xyxy,
                    "bbox_xyxyn": detection.bbox_xyxyn,
                    "full_frame_bbox_xyxy": detection.full_frame_bbox_xyxy,
                    "detection_id": detection.detection_id,
                    "detection_route": detection.detection_route,
                    "detector_class_id": detection.detector_class_id,
                    "detector_prompt": detection.detector_prompt,
                    "detector_prompt_set_fingerprint": detection.detector_prompt_set_fingerprint,
                    "detector_score": detection.detector_score,
                    "mask_polygon_xyn": detection.mask_polygon_xyn,
                    "route": detection.route,
                    "routing_policy_fingerprint": detection.routing_policy_fingerprint,
                    "routing_policy_version": detection.routing_policy_version,
                }
                for detection in detections
            ],
            "flickr_photo_id": photo_id,
            "raw_visual_input_id": raw_visual_input_id,
            "route": route,
            "scoring_unit_version": TARGET_FULL_FRAME_SCORING_UNIT_VERSION,
            "source": source,
            "source_record_hash": source_record_hash,
            "visual_mode": TARGET_AWARE_VISUAL_MODE,
        }
    )


def _same_decoded_content(
    left: RawFullFrameVisualInput,
    right: RawFullFrameVisualInput,
) -> bool:
    return (
        left.raw_image_content_hash == right.raw_image_content_hash
        and left.width == right.width
        and left.height == right.height
        and left.mode == right.mode
        and left.image.data == right.image.data
    )


def _positive_bbox(
    value: object,
    *,
    name: str,
) -> tuple[float, float, float, float]:
    bbox = _four_finite_floats(value, name=name)
    x1, y1, x2, y2 = bbox
    if x1 < 0.0 or y1 < 0.0:
        raise ValueError(f"{name} coordinates must be non-negative")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name} must have positive width and height")
    return bbox


def _normalized_bbox(
    value: object,
    *,
    name: str,
) -> tuple[float, float, float, float]:
    bbox = _four_finite_floats(value, name=name)
    if any(coordinate < 0.0 or coordinate > 1.0 for coordinate in bbox):
        raise ValueError(f"{name} coordinates must be in [0, 1]")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"{name} must have positive width and height")
    return bbox


def _four_finite_floats(
    value: object,
    *,
    name: str,
) -> tuple[float, float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError(f"{name} must contain four coordinates")
    coordinates = tuple(float(coordinate) for coordinate in value)
    if not all(math.isfinite(coordinate) for coordinate in coordinates):
        raise ValueError(f"{name} coordinates must be finite")
    return coordinates  # type: ignore[return-value]


def _consistent_optional_string(
    rows: Sequence[Mapping[str, Any]],
    *,
    field_name: str,
    source: str,
    photo_id: str,
) -> str | None:
    values = {
        value
        for row in rows
        if (value := _optional_string(row.get(field_name))) is not None
    }
    if len(values) > 1:
        raise ValueError(
            "conflicting per-photo source identity field "
            f"{field_name!r} for source={source!r}, "
            f"flickr_photo_id={photo_id!r}"
        )
    return next(iter(values), None)


def _consistent_required_string(
    rows: Sequence[Mapping[str, Any]],
    *,
    field_name: str,
    source: str,
    photo_id: str,
) -> str:
    values = [_required_string(row.get(field_name), field_name) for row in rows]
    unique_values = set(values)
    if len(unique_values) != 1:
        raise ValueError(
            "conflicting per-photo source identity field "
            f"{field_name!r} for source={source!r}, "
            f"flickr_photo_id={photo_id!r}"
        )
    return values[0]


def _validated_embedding(value: object) -> tuple[float, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Iterable):
        raise ValueError("full-frame encoder output must be a numeric vector")
    vector = tuple(float(item) for item in value)
    if not vector:
        raise ValueError("full-frame encoder returned an empty embedding")
    if not all(math.isfinite(item) for item in vector):
        raise ValueError("full-frame encoder returned a non-finite embedding")
    return vector


def _optional_finite_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_non_negative_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _required_string(value: object, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _optional_string(value: object) -> str | None:
    result = str(value or "").strip()
    return result or None


def _require_target_full_frame_mode(visual_mode: object) -> None:
    if str(visual_mode) != TARGET_AWARE_VISUAL_MODE:
        raise ValueError(
            "target full-frame plan requires visual_mode "
            f"{TARGET_AWARE_VISUAL_MODE!r}, got {visual_mode!r}"
        )


def _require_sha256(value: object, name: str) -> None:
    if _SHA256_FINGERPRINT.fullmatch(str(value)) is None:
        raise ValueError(f"{name} must be a sha256 fingerprint")


def _identity(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "RAW_FULL_IMAGE_KIND",
    "RAW_FULL_IMAGE_TRANSFORMATION_FINGERPRINT",
    "TARGET_AWARE_VISUAL_MODE",
    "TARGET_FULL_FRAME_EMBEDDING_VERSION",
    "TARGET_FULL_FRAME_MATERIALIZES_CROP_FILES",
    "TARGET_FULL_FRAME_REQUIRES_CROP_METADATA",
    "TARGET_FULL_FRAME_SCORING_UNIT_VERSION",
    "TARGET_FULL_FRAME_VISUAL_INPUT_VERSION",
    "EmbeddedTargetFullFramePlan",
    "FullFrameImageEncoder",
    "RawFullFrameEmbedding",
    "RawFullFrameVisualInput",
    "RoutedDetectionEvidence",
    "ScoringUnitEmbeddingReference",
    "TargetFullFramePlan",
    "TargetFullFrameScoringUnit",
    "build_target_full_frame_plan",
    "encode_target_full_frame_plan",
    "full_frame_embedding_id",
    "generate_target_full_frame_attention_variants",
    "target_full_frame_detection_run_policy",
]
