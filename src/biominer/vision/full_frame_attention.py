from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math

import PIL
from PIL import Image, ImageChops, ImageDraw

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.detection.detector_base import DecodedImage


RAW_FULL_IMAGE_KIND = "raw_full_image"
FOCUSED_FULL_FRAME_KIND = "focused_full_frame"
MASKED_FULL_FRAME_KIND = "masked_full_frame"
MULTI_OBJECT_FULL_FRAME_KIND = "multi_object_full_frame"
FULL_FRAME_VISUAL_INPUT_VERSION = "target-full-frame-visual-input-v2"
FULL_FRAME_ATTENTION_TRANSFORMATION_VERSION = "full-frame-attention-transform-v2"
FULL_FRAME_ATTENTION_EVIDENCE_VERSION = "full-frame-attention-evidence-v2"
FULL_FRAME_ATTENTION_UNAVAILABLE_VERSION = "full-frame-attention-unavailable-v2"
MASK_COVERAGE_BASIS_RASTERIZED_POLYGON = "rasterized_polygon_xyn"
TARGET_FULL_FRAME_IMAGE_RESIZE_MODE = "longest"

_RAW_FULL_IMAGE_TRANSFORMATION = {
    "canvas": "complete_original_decoded_canvas",
    "colour_mode": "RGB",
    "operation": "identity",
    "version": "raw-full-image-transform-v2",
}
RAW_FULL_IMAGE_TRANSFORMATION_FINGERPRINT = canonical_semantic_fingerprint(
    _RAW_FULL_IMAGE_TRANSFORMATION
)

_VARIANT_ORDER = {
    RAW_FULL_IMAGE_KIND: 0,
    FOCUSED_FULL_FRAME_KIND: 1,
    MASKED_FULL_FRAME_KIND: 2,
    MULTI_OBJECT_FULL_FRAME_KIND: 3,
}


@dataclass(frozen=True)
class TargetPreprocessingContract:
    """Canvas-preserving OpenCLIP inference preprocessing for target mode."""

    version: str = "target-full-frame-openclip-preprocess-v2"
    image_size_px: int = 224
    patch_size_px: int = 14
    interpolation: str = "bicubic"
    resize_mode: str = TARGET_FULL_FRAME_IMAGE_RESIZE_MODE
    padding_position: str = "center"
    padding_fill: int = 0
    implementation: str = "open_clip.ResizeKeepRatio+CenterCropOrPad"
    normalization_mean: tuple[float, float, float] = (
        0.48145466,
        0.4578275,
        0.40821073,
    )
    normalization_std: tuple[float, float, float] = (
        0.26862954,
        0.26130258,
        0.27577711,
    )

    def __post_init__(self) -> None:
        _require_non_empty(self.version, "preprocessing version")
        if (
            isinstance(self.image_size_px, bool)
            or not isinstance(self.image_size_px, int)
            or isinstance(self.patch_size_px, bool)
            or not isinstance(self.patch_size_px, int)
        ):
            raise ValueError("preprocessing image and patch sizes must be integers")
        if self.image_size_px <= 0 or self.patch_size_px <= 0:
            raise ValueError("preprocessing image and patch sizes must be positive")
        if self.patch_size_px != 14:
            raise ValueError("BioCLIP 2.5 ViT-H/14 requires patch_size_px=14")
        if self.image_size_px % self.patch_size_px:
            raise ValueError("preprocessing image size must be divisible by patch size")
        if self.interpolation != "bicubic":
            raise ValueError(
                "target full-frame preprocessing requires bicubic interpolation"
            )
        if self.resize_mode != "longest":
            raise ValueError(
                "target full-frame preprocessing requires longest-side resize"
            )
        if self.padding_position != "center" or self.padding_fill != 0:
            raise ValueError(
                "target full-frame preprocessing requires centered zero padding"
            )
        if self.implementation != "open_clip.ResizeKeepRatio+CenterCropOrPad":
            raise ValueError(
                "unsupported target full-frame preprocessing implementation"
            )
        for field_name in ("normalization_mean", "normalization_std"):
            raw_values = getattr(self, field_name)
            if (
                not isinstance(raw_values, (tuple, list))
                or len(raw_values) != 3
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in raw_values
                )
            ):
                raise ValueError(f"{field_name} must contain three finite values")
            values = tuple(float(value) for value in raw_values)
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"{field_name} must contain three finite values")
            if field_name == "normalization_std" and any(
                value <= 0 for value in values
            ):
                raise ValueError("normalization_std values must be positive")
            object.__setattr__(self, field_name, values)

    @property
    def fingerprint(self) -> str:
        return _identity(
            {
                "image_size_px": self.image_size_px,
                "implementation": self.implementation,
                "interpolation": self.interpolation,
                "padding_fill": self.padding_fill,
                "padding_position": self.padding_position,
                "patch_size_px": self.patch_size_px,
                "normalization_mean": self.normalization_mean,
                "normalization_std": self.normalization_std,
                "resize_mode": self.resize_mode,
                "version": self.version,
            }
        )

    def resize_geometry(
        self,
        *,
        width: int,
        height: int,
    ) -> tuple[int, int, tuple[int, int, int, int]]:
        """Return OpenCLIP-longest resized dimensions and centered LTRB padding."""

        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
        ):
            raise ValueError("source dimensions must be positive")
        scale = self.image_size_px / max(width, height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        horizontal = self.image_size_px - resized_width
        vertical = self.image_size_px - resized_height
        padding = (
            horizontal // 2,
            vertical // 2,
            (horizontal + 1) // 2,
            (vertical + 1) // 2,
        )
        return resized_width, resized_height, padding


@dataclass(frozen=True)
class FullFrameTransformPolicy:
    version: str = FULL_FRAME_ATTENTION_TRANSFORMATION_VERSION
    outside_dim_divisor: int = 4
    outside_dim_rounding_bias: int = 2
    bbox_projection: str = "floor-left-top_ceil-right-bottom_exclusive-v1"
    polygon_projection: str = "half-up-clamped-to-canvas-v1"
    mask_rasterizer: str = "pillow-imagedraw-binary-l-v1"
    pillow_version: str = PIL.__version__

    def __post_init__(self) -> None:
        _require_non_empty(self.version, "transformation version")
        if (
            isinstance(self.outside_dim_divisor, bool)
            or not isinstance(self.outside_dim_divisor, int)
            or isinstance(self.outside_dim_rounding_bias, bool)
            or not isinstance(self.outside_dim_rounding_bias, int)
        ):
            raise ValueError("background attenuation parameters must be integers")
        if self.outside_dim_divisor <= 1:
            raise ValueError("outside_dim_divisor must be greater than one")
        if not 0 <= self.outside_dim_rounding_bias < self.outside_dim_divisor:
            raise ValueError(
                "outside_dim_rounding_bias must be in [0, outside_dim_divisor)"
            )
        _require_non_empty(self.pillow_version, "Pillow version")
        if self.bbox_projection != "floor-left-top_ceil-right-bottom_exclusive-v1":
            raise ValueError("unsupported bbox projection policy")
        if self.polygon_projection != "half-up-clamped-to-canvas-v1":
            raise ValueError("unsupported polygon projection policy")
        if self.mask_rasterizer != "pillow-imagedraw-binary-l-v1":
            raise ValueError("unsupported mask rasterizer policy")

    @property
    def fingerprint(self) -> str:
        return _identity(
            {
                "bbox_projection": self.bbox_projection,
                "mask_rasterizer": self.mask_rasterizer,
                "outside_dim_divisor": self.outside_dim_divisor,
                "outside_dim_rounding_bias": self.outside_dim_rounding_bias,
                "pillow_version": self.pillow_version,
                "polygon_projection": self.polygon_projection,
                "version": self.version,
            }
        )


@dataclass(frozen=True)
class AttentionQualityPolicy:
    version: str = "full-frame-attention-quality-v2"
    small_subject_area_ratio_threshold: float = 0.01
    low_resolution_min_side_px: int = 224
    max_relevant_detections: int = 8
    max_materialization_bytes: int = 512 * 1024 * 1024
    preprocessing: TargetPreprocessingContract = field(
        default_factory=TargetPreprocessingContract
    )

    def __post_init__(self) -> None:
        _require_non_empty(self.version, "quality-policy version")
        if isinstance(self.small_subject_area_ratio_threshold, bool):
            raise ValueError("small-subject threshold must be finite and in [0, 1]")
        threshold = float(self.small_subject_area_ratio_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("small-subject threshold must be finite and in [0, 1]")
        object.__setattr__(self, "small_subject_area_ratio_threshold", threshold)
        if (
            isinstance(self.low_resolution_min_side_px, bool)
            or not isinstance(self.low_resolution_min_side_px, int)
            or self.low_resolution_min_side_px <= 0
        ):
            raise ValueError("low-resolution minimum side must be positive")
        for value, name in (
            (self.max_relevant_detections, "max_relevant_detections"),
            (self.max_materialization_bytes, "max_materialization_bytes"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def fingerprint(self) -> str:
        return _identity(
            {
                "low_resolution_min_side_px": self.low_resolution_min_side_px,
                "max_materialization_bytes": self.max_materialization_bytes,
                "max_relevant_detections": self.max_relevant_detections,
                "preprocessing_fingerprint": self.preprocessing.fingerprint,
                "small_subject_area_ratio_threshold": self.small_subject_area_ratio_threshold,
                "version": self.version,
            }
        )


@dataclass(frozen=True)
class AttentionRegion:
    source_detection_id: str
    route: str
    bbox_xyxyn: tuple[float, float, float, float]
    mask_polygon_xyn: tuple[tuple[float, float], ...] | None = None
    detector_score: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_detection_id",
            _require_non_empty(self.source_detection_id, "source_detection_id"),
        )
        object.__setattr__(self, "route", _require_non_empty(self.route, "route"))
        object.__setattr__(
            self,
            "bbox_xyxyn",
            _normalized_bbox(self.bbox_xyxyn, name="bbox_xyxyn"),
        )
        if self.mask_polygon_xyn is not None:
            object.__setattr__(
                self,
                "mask_polygon_xyn",
                _normalized_polygon(self.mask_polygon_xyn),
            )
        if self.detector_score is not None:
            if isinstance(self.detector_score, bool):
                raise ValueError("detector_score must be finite")
            score = float(self.detector_score)
            if not math.isfinite(score):
                raise ValueError("detector_score must be finite")
            object.__setattr__(self, "detector_score", score)


@dataclass(frozen=True)
class FullFrameAttentionVariant:
    visual_input_id: str
    visual_input_kind: str
    visual_input_version: str
    raw_image_content_hash: str
    visual_content_hash: str
    transformation_applied: bool
    transformation_version: str
    transformation_policy_fingerprint: str
    transformation_fingerprint: str
    width: int
    height: int
    mode: str
    image: DecodedImage


@dataclass(frozen=True)
class FullFrameVariantEvidence:
    evidence_id: str
    evidence_version: str
    visual_input_id: str
    visual_input_kind: str
    source_type: str
    source_record_id: str | None
    route: str
    source_detection_ids: tuple[str, ...]
    source_regions: tuple[AttentionRegion, ...]
    subject_area_ratio: float
    detector_score: float | None
    detector_score_aggregation: str | None
    relevant_detection_count: int
    source_detection_count: int
    mask_coverage: float | None
    mask_coverage_basis: str | None
    quality_policy_version: str
    quality_policy_fingerprint: str
    visual_input_quality_flags: tuple[str, ...]
    transformation_applied: bool
    transformation_version: str
    transformation_fingerprint: str


@dataclass(frozen=True)
class UnavailableAttentionVariant:
    unavailable_id: str
    unavailable_version: str
    visual_input_kind: str
    source_type: str
    source_record_id: str | None
    route: str
    source_detection_ids: tuple[str, ...]
    reason: str
    transformation_version: str
    transformation_policy_fingerprint: str
    visual_input_quality_flags: tuple[str, ...]


@dataclass(frozen=True)
class FullFrameAttentionResult:
    variants: tuple[FullFrameAttentionVariant, ...]
    evidence: tuple[FullFrameVariantEvidence, ...]
    unavailable_variants: tuple[UnavailableAttentionVariant, ...]

    def variant(self, visual_input_id: str) -> FullFrameAttentionVariant:
        matches = [
            variant
            for variant in self.variants
            if variant.visual_input_id == visual_input_id
        ]
        if len(matches) != 1:
            raise KeyError(visual_input_id)
        return matches[0]


@dataclass(frozen=True)
class _TransformContext:
    source: Image.Image
    dimmed: Image.Image
    raw_variant: FullFrameAttentionVariant
    policy: FullFrameTransformPolicy


def raw_full_frame_visual_input(image: DecodedImage) -> FullFrameAttentionVariant:
    """Build the authoritative byte-identical raw full-frame visual input."""

    _require_decoded_rgb(image)
    content_hash = decoded_image_content_hash(image)
    visual_input_id = _identity(
        {
            "raw_image_content_hash": content_hash,
            "transformation_fingerprint": RAW_FULL_IMAGE_TRANSFORMATION_FINGERPRINT,
            "visual_input_kind": RAW_FULL_IMAGE_KIND,
            "visual_input_version": FULL_FRAME_VISUAL_INPUT_VERSION,
        }
    )
    return FullFrameAttentionVariant(
        visual_input_id=visual_input_id,
        visual_input_kind=RAW_FULL_IMAGE_KIND,
        visual_input_version=FULL_FRAME_VISUAL_INPUT_VERSION,
        raw_image_content_hash=content_hash,
        visual_content_hash=content_hash,
        transformation_applied=False,
        transformation_version=str(_RAW_FULL_IMAGE_TRANSFORMATION["version"]),
        transformation_policy_fingerprint=RAW_FULL_IMAGE_TRANSFORMATION_FINGERPRINT,
        transformation_fingerprint=RAW_FULL_IMAGE_TRANSFORMATION_FINGERPRINT,
        width=image.width,
        height=image.height,
        mode=image.mode,
        image=image,
    )


def decoded_image_content_hash(image: DecodedImage) -> str:
    _require_decoded_rgb(image)
    header = json.dumps(
        {
            "height": image.height,
            "mode": image.mode,
            "version": "decoded-image-content-v1",
            "width": image.width,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(image.data)
    return "sha256:" + digest.hexdigest()


def generate_full_frame_attention_variants(
    image: DecodedImage,
    regions: Sequence[AttentionRegion],
    *,
    source_type: str,
    source_record_id: str | None = None,
    transform_policy: FullFrameTransformPolicy | None = None,
    quality_policy: AttentionQualityPolicy | None = None,
) -> FullFrameAttentionResult:
    """Generate route-local, full-canvas visual inputs through one shared kernel."""

    _require_decoded_rgb(image)
    source_type = _require_non_empty(source_type, "source_type")
    source_record_id = _optional_non_empty(source_record_id, "source_record_id")
    transform_policy = transform_policy or FullFrameTransformPolicy()
    quality_policy = quality_policy or AttentionQualityPolicy()
    ordered_regions = _validated_regions(
        regions,
        max_relevant_detections=quality_policy.max_relevant_detections,
    )
    _validate_materialization_budget(
        image=image,
        regions=ordered_regions,
        max_materialization_bytes=quality_policy.max_materialization_bytes,
    )
    route = ordered_regions[0].route
    raw_variant = raw_full_frame_visual_input(image)
    transform_context = _transform_context(
        image=image,
        raw_variant=raw_variant,
        policy=transform_policy,
    )
    variant_by_transformation: dict[str, FullFrameAttentionVariant] = {}

    def materialize(
        *,
        kind: str,
        attention_mask: Image.Image,
        geometry: Mapping[str, object],
    ) -> FullFrameAttentionVariant:
        fingerprint = _attention_transformation_fingerprint(
            image=image,
            kind=kind,
            geometry=geometry,
            policy=transform_policy,
        )
        cached = variant_by_transformation.get(fingerprint)
        if cached is not None:
            return cached
        variant = _transformed_variant(
            context=transform_context,
            kind=kind,
            attention_mask=attention_mask,
            transformation_fingerprint=fingerprint,
        )
        variant_by_transformation[fingerprint] = variant
        return variant

    variants: list[FullFrameAttentionVariant] = [raw_variant]
    evidence: list[FullFrameVariantEvidence] = []
    unavailable: list[UnavailableAttentionVariant] = []
    all_bbox_area = _rectangle_union_area(
        tuple(region.bbox_xyxyn for region in ordered_regions)
    )
    polygon_masks: dict[
        tuple[tuple[float, float], ...],
        Image.Image | None,
    ] = {}
    valid_masks: dict[str, Image.Image | None] = {}
    for region in ordered_regions:
        if region.mask_polygon_xyn is None:
            continue
        canonical_polygon = _canonical_polygon(region.mask_polygon_xyn)
        if canonical_polygon not in polygon_masks:
            polygon_masks[canonical_polygon] = _polygon_mask(
                image,
                canonical_polygon,
            )
        valid_masks[region.source_detection_id] = polygon_masks[canonical_polygon]
    valid_mask_count = sum(mask is not None for mask in valid_masks.values())
    missing_mask_count = len(ordered_regions) - len(valid_masks)
    degenerate_mask_count = len(valid_masks) - valid_mask_count
    complete_masks = valid_mask_count == len(ordered_regions)
    raw_mask = (
        _union_masks(tuple(mask for mask in valid_masks.values() if mask is not None))
        if complete_masks
        else None
    )
    raw_flags = _quality_flags(
        image=image,
        selected_regions=ordered_regions,
        all_regions=ordered_regions,
        subject_area_ratio=all_bbox_area,
        quality_policy=quality_policy,
    )
    if raw_mask is not None:
        raw_flags = _with_flags(raw_flags, "mask_polygon_approximation")
    if missing_mask_count:
        raw_flags = _with_flags(raw_flags, "mask_missing")
    if degenerate_mask_count:
        raw_flags = _with_flags(raw_flags, "mask_degenerate")
    if valid_mask_count and not complete_masks:
        raw_flags = _with_flags(raw_flags, "partial_mask_set")
    evidence.append(
        _variant_evidence(
            variant=raw_variant,
            source_type=source_type,
            source_record_id=source_record_id,
            route=route,
            source_regions=ordered_regions,
            relevant_detection_count=len(ordered_regions),
            subject_area_ratio=all_bbox_area,
            mask_coverage=_mask_coverage(raw_mask),
            mask_coverage_basis=(
                MASK_COVERAGE_BASIS_RASTERIZED_POLYGON if raw_mask is not None else None
            ),
            quality_policy=quality_policy,
            quality_flags=raw_flags,
        )
    )

    bbox_masks: dict[tuple[float, float, float, float], Image.Image] = {}
    for region in ordered_regions:
        bbox_mask = bbox_masks.get(region.bbox_xyxyn)
        if bbox_mask is None:
            bbox_mask = _bbox_mask(image, region.bbox_xyxyn)
            bbox_masks[region.bbox_xyxyn] = bbox_mask
        focused = materialize(
            kind=FOCUSED_FULL_FRAME_KIND,
            attention_mask=bbox_mask,
            geometry={"bbox_xyxyn": region.bbox_xyxyn},
        )
        variants.append(focused)
        focused_flags = _quality_flags(
            image=image,
            selected_regions=(region,),
            all_regions=ordered_regions,
            subject_area_ratio=_bbox_area(region.bbox_xyxyn),
            quality_policy=quality_policy,
        )
        if focused.visual_content_hash == raw_variant.visual_content_hash:
            focused_flags = _with_flags(focused_flags, "attention_transform_noop")
        evidence.append(
            _variant_evidence(
                variant=focused,
                source_type=source_type,
                source_record_id=source_record_id,
                route=route,
                source_regions=(region,),
                relevant_detection_count=len(ordered_regions),
                subject_area_ratio=_bbox_area(region.bbox_xyxyn),
                mask_coverage=None,
                mask_coverage_basis=None,
                quality_policy=quality_policy,
                quality_flags=focused_flags,
            )
        )

        polygon_mask = valid_masks.get(region.source_detection_id)
        if region.mask_polygon_xyn is None:
            unavailable.append(
                _unavailable_variant(
                    kind=MASKED_FULL_FRAME_KIND,
                    source_type=source_type,
                    source_record_id=source_record_id,
                    route=route,
                    source_regions=(region,),
                    reason="mask_missing",
                    transform_policy=transform_policy,
                    quality_flags=_with_flags(focused_flags, "mask_missing"),
                )
            )
            continue
        if polygon_mask is None:
            unavailable.append(
                _unavailable_variant(
                    kind=MASKED_FULL_FRAME_KIND,
                    source_type=source_type,
                    source_record_id=source_record_id,
                    route=route,
                    source_regions=(region,),
                    reason="mask_degenerate",
                    transform_policy=transform_policy,
                    quality_flags=_with_flags(focused_flags, "mask_degenerate"),
                )
            )
            continue
        masked = materialize(
            kind=MASKED_FULL_FRAME_KIND,
            attention_mask=polygon_mask,
            geometry={"mask_polygon_xyn": _canonical_polygon(region.mask_polygon_xyn)},
        )
        variants.append(masked)
        masked_flags = _with_flags(focused_flags, "mask_polygon_approximation")
        if _polygon_extends_outside_bbox(
            region.mask_polygon_xyn,
            region.bbox_xyxyn,
        ):
            masked_flags = _with_flags(masked_flags, "mask_outside_bbox")
        if masked.visual_content_hash == raw_variant.visual_content_hash:
            masked_flags = _with_flags(masked_flags, "attention_transform_noop")
        evidence.append(
            _variant_evidence(
                variant=masked,
                source_type=source_type,
                source_record_id=source_record_id,
                route=route,
                source_regions=(region,),
                relevant_detection_count=len(ordered_regions),
                subject_area_ratio=_bbox_area(region.bbox_xyxyn),
                mask_coverage=_mask_coverage(polygon_mask),
                mask_coverage_basis=MASK_COVERAGE_BASIS_RASTERIZED_POLYGON,
                quality_policy=quality_policy,
                quality_flags=masked_flags,
            )
        )

    if len(ordered_regions) > 1:
        union_bbox_mask = _union_masks(
            tuple(bbox_masks[region.bbox_xyxyn] for region in ordered_regions)
        )
        multi = materialize(
            kind=MULTI_OBJECT_FULL_FRAME_KIND,
            attention_mask=union_bbox_mask,
            geometry={
                "bboxes_xyxyn": tuple(
                    sorted(region.bbox_xyxyn for region in ordered_regions)
                )
            },
        )
        variants.append(multi)
        multi_flags = _quality_flags(
            image=image,
            selected_regions=ordered_regions,
            all_regions=ordered_regions,
            subject_area_ratio=all_bbox_area,
            quality_policy=quality_policy,
        )
        if raw_mask is not None:
            multi_flags = _with_flags(
                multi_flags,
                "mask_polygon_approximation",
            )
        if missing_mask_count:
            multi_flags = _with_flags(multi_flags, "mask_missing")
        if degenerate_mask_count:
            multi_flags = _with_flags(multi_flags, "mask_degenerate")
        if valid_mask_count and not complete_masks:
            multi_flags = _with_flags(multi_flags, "partial_mask_set")
        if multi.visual_content_hash == raw_variant.visual_content_hash:
            multi_flags = _with_flags(multi_flags, "attention_transform_noop")
        evidence.append(
            _variant_evidence(
                variant=multi,
                source_type=source_type,
                source_record_id=source_record_id,
                route=route,
                source_regions=ordered_regions,
                relevant_detection_count=len(ordered_regions),
                subject_area_ratio=all_bbox_area,
                mask_coverage=_mask_coverage(raw_mask),
                mask_coverage_basis=(
                    MASK_COVERAGE_BASIS_RASTERIZED_POLYGON
                    if raw_mask is not None
                    else None
                ),
                quality_policy=quality_policy,
                quality_flags=multi_flags,
            )
        )

    variants_by_id: dict[str, FullFrameAttentionVariant] = {}
    for variant in variants:
        existing = variants_by_id.setdefault(variant.visual_input_id, variant)
        if existing != variant:
            raise RuntimeError(
                "attention generation produced a visual-input identity collision"
            )
    ordered_variants = tuple(sorted(variants_by_id.values(), key=_variant_sort_key))
    known_visual_ids = set(variants_by_id)
    if any(item.visual_input_id not in known_visual_ids for item in evidence):
        raise RuntimeError("attention evidence references an unknown visual input")
    return FullFrameAttentionResult(
        variants=ordered_variants,
        evidence=tuple(
            sorted(
                evidence,
                key=lambda item: (
                    _VARIANT_ORDER[item.visual_input_kind],
                    item.source_detection_ids,
                    item.evidence_id,
                ),
            )
        ),
        unavailable_variants=tuple(
            sorted(
                unavailable,
                key=lambda item: (
                    _VARIANT_ORDER[item.visual_input_kind],
                    item.source_detection_ids,
                    item.unavailable_id,
                ),
            )
        ),
    )


def _transformed_variant(
    *,
    context: _TransformContext,
    kind: str,
    attention_mask: Image.Image,
    transformation_fingerprint: str,
) -> FullFrameAttentionVariant:
    if kind not in {
        FOCUSED_FULL_FRAME_KIND,
        MASKED_FULL_FRAME_KIND,
        MULTI_OBJECT_FULL_FRAME_KIND,
    }:
        raise ValueError(f"unsupported attention variant kind {kind!r}")
    image = context.raw_variant.image
    policy = context.policy
    if attention_mask.mode != "L" or attention_mask.size != (image.width, image.height):
        raise ValueError("attention mask must be an L image on the original canvas")
    transformed = Image.composite(context.source, context.dimmed, attention_mask)
    output = DecodedImage(
        width=image.width,
        height=image.height,
        mode="RGB",
        data=transformed.tobytes(),
        source_uri=image.source_uri,
    )
    raw_hash = context.raw_variant.raw_image_content_hash
    output_hash = decoded_image_content_hash(output)
    visual_input_id = _identity(
        {
            "raw_image_content_hash": raw_hash,
            "transformation_fingerprint": transformation_fingerprint,
            "visual_content_hash": output_hash,
            "visual_input_kind": kind,
            "visual_input_version": FULL_FRAME_VISUAL_INPUT_VERSION,
        }
    )
    return FullFrameAttentionVariant(
        visual_input_id=visual_input_id,
        visual_input_kind=kind,
        visual_input_version=FULL_FRAME_VISUAL_INPUT_VERSION,
        raw_image_content_hash=raw_hash,
        visual_content_hash=output_hash,
        transformation_applied=True,
        transformation_version=policy.version,
        transformation_policy_fingerprint=policy.fingerprint,
        transformation_fingerprint=transformation_fingerprint,
        width=image.width,
        height=image.height,
        mode="RGB",
        image=output,
    )


def _transform_context(
    *,
    image: DecodedImage,
    raw_variant: FullFrameAttentionVariant,
    policy: FullFrameTransformPolicy,
) -> _TransformContext:
    source = Image.frombytes("RGB", (image.width, image.height), image.data)
    lut = [
        (value + policy.outside_dim_rounding_bias) // policy.outside_dim_divisor
        for value in range(256)
    ]
    return _TransformContext(
        source=source,
        dimmed=source.point(lut * 3),
        raw_variant=raw_variant,
        policy=policy,
    )


def _attention_transformation_fingerprint(
    *,
    image: DecodedImage,
    kind: str,
    geometry: Mapping[str, object],
    policy: FullFrameTransformPolicy,
) -> str:
    return _identity(
        {
            "canvas": {"height": image.height, "width": image.width},
            "geometry": geometry,
            "policy_fingerprint": policy.fingerprint,
            "visual_input_kind": kind,
        }
    )


def _variant_evidence(
    *,
    variant: FullFrameAttentionVariant,
    source_type: str,
    source_record_id: str | None,
    route: str,
    source_regions: tuple[AttentionRegion, ...],
    relevant_detection_count: int,
    subject_area_ratio: float,
    mask_coverage: float | None,
    mask_coverage_basis: str | None,
    quality_policy: AttentionQualityPolicy,
    quality_flags: tuple[str, ...],
) -> FullFrameVariantEvidence:
    scores = tuple(
        region.detector_score
        for region in source_regions
        if region.detector_score is not None
    )
    detector_score = max(scores) if scores else None
    aggregation = "single" if len(source_regions) == 1 else "max"
    if detector_score is None:
        aggregation = None
    source_regions = tuple(
        sorted(source_regions, key=lambda region: region.source_detection_id)
    )
    source_detection_ids = tuple(
        region.source_detection_id for region in source_regions
    )
    flags = tuple(sorted(set(quality_flags)))
    payload = {
        "detector_score": detector_score,
        "detector_score_aggregation": aggregation,
        "evidence_version": FULL_FRAME_ATTENTION_EVIDENCE_VERSION,
        "mask_coverage": mask_coverage,
        "mask_coverage_basis": mask_coverage_basis,
        "quality_flags": flags,
        "quality_policy_fingerprint": quality_policy.fingerprint,
        "relevant_detection_count": relevant_detection_count,
        "route": route,
        "source_record_id": source_record_id,
        "source_regions": [_region_payload(region) for region in source_regions],
        "source_type": source_type,
        "subject_area_ratio": subject_area_ratio,
        "transformation_applied": variant.transformation_applied,
        "transformation_fingerprint": variant.transformation_fingerprint,
        "visual_input_id": variant.visual_input_id,
        "visual_input_kind": variant.visual_input_kind,
    }
    return FullFrameVariantEvidence(
        evidence_id=_identity(payload),
        evidence_version=FULL_FRAME_ATTENTION_EVIDENCE_VERSION,
        visual_input_id=variant.visual_input_id,
        visual_input_kind=variant.visual_input_kind,
        source_type=source_type,
        source_record_id=source_record_id,
        route=route,
        source_detection_ids=source_detection_ids,
        source_regions=source_regions,
        subject_area_ratio=subject_area_ratio,
        detector_score=detector_score,
        detector_score_aggregation=aggregation,
        relevant_detection_count=relevant_detection_count,
        source_detection_count=len(source_regions),
        mask_coverage=mask_coverage,
        mask_coverage_basis=mask_coverage_basis,
        quality_policy_version=quality_policy.version,
        quality_policy_fingerprint=quality_policy.fingerprint,
        visual_input_quality_flags=flags,
        transformation_applied=variant.transformation_applied,
        transformation_version=variant.transformation_version,
        transformation_fingerprint=variant.transformation_fingerprint,
    )


def _unavailable_variant(
    *,
    kind: str,
    source_type: str,
    source_record_id: str | None,
    route: str,
    source_regions: tuple[AttentionRegion, ...],
    reason: str,
    transform_policy: FullFrameTransformPolicy,
    quality_flags: tuple[str, ...],
) -> UnavailableAttentionVariant:
    source_detection_ids = tuple(
        sorted(region.source_detection_id for region in source_regions)
    )
    flags = tuple(sorted(set(quality_flags)))
    payload = {
        "reason": reason,
        "route": route,
        "source_detection_ids": source_detection_ids,
        "source_record_id": source_record_id,
        "source_type": source_type,
        "transformation_policy_fingerprint": transform_policy.fingerprint,
        "unavailable_version": FULL_FRAME_ATTENTION_UNAVAILABLE_VERSION,
        "visual_input_kind": kind,
        "visual_input_quality_flags": flags,
    }
    return UnavailableAttentionVariant(
        unavailable_id=_identity(payload),
        unavailable_version=FULL_FRAME_ATTENTION_UNAVAILABLE_VERSION,
        visual_input_kind=kind,
        source_type=source_type,
        source_record_id=source_record_id,
        route=route,
        source_detection_ids=source_detection_ids,
        reason=reason,
        transformation_version=transform_policy.version,
        transformation_policy_fingerprint=transform_policy.fingerprint,
        visual_input_quality_flags=flags,
    )


def _quality_flags(
    *,
    image: DecodedImage,
    selected_regions: tuple[AttentionRegion, ...],
    all_regions: tuple[AttentionRegion, ...],
    subject_area_ratio: float,
    quality_policy: AttentionQualityPolicy,
) -> tuple[str, ...]:
    flags: set[str] = set()
    if subject_area_ratio < quality_policy.small_subject_area_ratio_threshold:
        flags.add("small_subject")
    if min(image.width, image.height) < quality_policy.low_resolution_min_side_px:
        flags.add("low_source_resolution")
    if image.width != image.height:
        flags.add("preprocess_padding_required")
    if len(all_regions) > 1:
        flags.add("multiple_relevant_detections")
    if any(region.detector_score is None for region in selected_regions):
        flags.add("detector_score_missing")
    if any(_bbox_touches_edge(region.bbox_xyxyn) for region in selected_regions):
        flags.add("subject_touches_canvas_edge")
    return tuple(sorted(flags))


def _validated_regions(
    regions: Sequence[AttentionRegion],
    *,
    max_relevant_detections: int,
) -> tuple[AttentionRegion, ...]:
    if isinstance(regions, str | bytes) or not isinstance(regions, Sequence):
        raise TypeError("regions must be a sequence of AttentionRegion values")
    if not regions:
        raise ValueError("at least one attention region is required")
    if len(regions) > max_relevant_detections:
        raise ValueError(
            "attention region count exceeds the materialization bound: "
            f"{len(regions)} > {max_relevant_detections}"
        )
    if any(not isinstance(region, AttentionRegion) for region in regions):
        raise TypeError("regions must contain AttentionRegion values")
    ordered = tuple(sorted(regions, key=lambda region: region.source_detection_id))
    identifiers = tuple(region.source_detection_id for region in ordered)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("attention regions require unique source_detection_id values")
    routes = {region.route for region in ordered}
    if len(routes) != 1:
        raise ValueError("attention regions must be route-compatible")
    return ordered


def _validate_materialization_budget(
    *,
    image: DecodedImage,
    regions: tuple[AttentionRegion, ...],
    max_materialization_bytes: int,
) -> None:
    unique_bboxes = {region.bbox_xyxyn for region in regions}
    unique_polygons = {
        _canonical_polygon(region.mask_polygon_xyn)
        for region in regions
        if region.mask_polygon_xyn is not None
    }
    transformed_rgb_buffers = (
        len(unique_bboxes) + len(unique_polygons) + (len(regions) > 1)
    )
    transient_rgb_buffers = 2
    mask_buffers = len(unique_bboxes) + len(unique_polygons) + 2
    estimated_bytes = (transformed_rgb_buffers + transient_rgb_buffers) * len(
        image.data
    ) + mask_buffers * image.width * image.height
    if estimated_bytes > max_materialization_bytes:
        raise ValueError(
            "full-frame attention materialization budget exceeded: "
            f"estimated_bytes={estimated_bytes}, "
            f"max_materialization_bytes={max_materialization_bytes}"
        )


def _bbox_mask(
    image: DecodedImage,
    bbox: tuple[float, float, float, float],
) -> Image.Image:
    left = max(0, min(image.width - 1, math.floor(bbox[0] * image.width)))
    top = max(0, min(image.height - 1, math.floor(bbox[1] * image.height)))
    right = max(left + 1, min(image.width, math.ceil(bbox[2] * image.width)))
    bottom = max(top + 1, min(image.height, math.ceil(bbox[3] * image.height)))
    mask = Image.new("L", (image.width, image.height), 0)
    ImageDraw.Draw(mask).rectangle((left, top, right - 1, bottom - 1), fill=255)
    return mask


def _polygon_mask(
    image: DecodedImage,
    polygon: tuple[tuple[float, float], ...] | None,
) -> Image.Image | None:
    if polygon is None or _polygon_area(polygon) == 0.0:
        return None
    points = tuple(
        (
            _half_up_pixel(x, image.width),
            _half_up_pixel(y, image.height),
        )
        for x, y in polygon
    )
    if len(set(points)) < 3 or _polygon_area(points) == 0.0:
        return None
    mask = Image.new("L", (image.width, image.height), 0)
    ImageDraw.Draw(mask).polygon(points, fill=255)
    return mask if mask.getbbox() is not None else None


def _half_up_pixel(value: float, dimension: int) -> int:
    # Ultralytics xyn values are pixel-index contours divided by the full
    # original dimension, so the inverse uses the dimension, not dimension - 1.
    projected = int(
        (Decimal(str(value)) * Decimal(dimension)).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )
    return max(0, min(dimension - 1, projected))


def _union_masks(masks: tuple[Image.Image, ...]) -> Image.Image:
    if not masks:
        raise ValueError("cannot union an empty mask collection")
    size = masks[0].size
    if any(mask.mode != "L" or mask.size != size for mask in masks):
        raise ValueError("union masks must share one L canvas")
    output = Image.new("L", size, 0)
    for mask in masks:
        output = ImageChops.lighter(output, mask)
    return output


def _mask_coverage(mask: Image.Image | None) -> float | None:
    if mask is None:
        return None
    return mask.histogram()[255] / (mask.width * mask.height)


def _polygon_extends_outside_bbox(
    polygon: tuple[tuple[float, float], ...],
    bbox: tuple[float, float, float, float],
) -> bool:
    left, top, right, bottom = bbox
    return any(x < left or x > right or y < top or y > bottom for x, y in polygon)


def _rectangle_union_area(
    rectangles: tuple[tuple[float, float, float, float], ...],
) -> float:
    x_values = sorted(
        {
            coordinate
            for rectangle in rectangles
            for coordinate in (rectangle[0], rectangle[2])
        }
    )
    area = 0.0
    for left, right in zip(x_values, x_values[1:], strict=False):
        if right <= left:
            continue
        intervals = sorted(
            (rectangle[1], rectangle[3])
            for rectangle in rectangles
            if rectangle[0] < right and rectangle[2] > left
        )
        covered = 0.0
        if intervals:
            start, end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start > end:
                    covered += end - start
                    start, end = next_start, next_end
                else:
                    end = max(end, next_end)
            covered += end - start
        area += (right - left) * covered
    return area


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def _bbox_touches_edge(bbox: tuple[float, float, float, float]) -> bool:
    return bbox[0] == 0.0 or bbox[1] == 0.0 or bbox[2] == 1.0 or bbox[3] == 1.0


def _normalized_bbox(
    value: object,
    *,
    name: str,
) -> tuple[float, float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError(f"{name} must contain four coordinates")
    coordinates = tuple(_finite_unit_float(item, name=name) for item in value)
    if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
        raise ValueError(f"{name} must have positive width and height")
    return coordinates  # type: ignore[return-value]


def _normalized_polygon(
    value: object,
) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list | tuple) or len(value) < 3:
        raise ValueError("mask_polygon_xyn must contain at least three points")
    points: list[tuple[float, float]] = []
    for point in value:
        if not isinstance(point, list | tuple) or len(point) != 2:
            raise ValueError("mask_polygon_xyn points must contain two coordinates")
        normalized = (
            _finite_unit_float(point[0], name="mask_polygon_xyn"),
            _finite_unit_float(point[1], name="mask_polygon_xyn"),
        )
        if not points or points[-1] != normalized:
            points.append(normalized)
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3:
        raise ValueError(
            "mask_polygon_xyn must contain three distinct consecutive points"
        )
    return tuple(points)


def _canonical_polygon(
    polygon: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    points = list(polygon)
    candidates: list[tuple[tuple[float, float], ...]] = []
    for values in (points, list(reversed(points))):
        candidates.extend(
            tuple(values[index:] + values[:index]) for index in range(len(values))
        )
    return min(candidates)


def _polygon_area(points: Sequence[tuple[float | int, float | int]]) -> float:
    return (
        abs(
            sum(
                float(x1) * float(y2) - float(x2) * float(y1)
                for (x1, y1), (x2, y2) in zip(
                    points, (*points[1:], points[0]), strict=True
                )
            )
        )
        / 2.0
    )


def _finite_unit_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} coordinates must be finite numbers")
    try:
        result = round(float(value), 6)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} coordinates must be finite numbers") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} coordinates must be finite and in [0, 1]")
    return 0.0 if result == 0.0 else result


def _region_payload(region: AttentionRegion) -> dict[str, object]:
    return {
        "bbox_xyxyn": region.bbox_xyxyn,
        "detector_score": region.detector_score,
        "mask_polygon_xyn": region.mask_polygon_xyn,
        "route": region.route,
        "source_detection_id": region.source_detection_id,
    }


def _variant_sort_key(
    variant: FullFrameAttentionVariant,
) -> tuple[int, str]:
    return _VARIANT_ORDER[variant.visual_input_kind], variant.visual_input_id


def _with_flags(flags: tuple[str, ...], *values: str) -> tuple[str, ...]:
    return tuple(sorted({*flags, *values}))


def _require_decoded_rgb(image: DecodedImage) -> None:
    if not isinstance(image, DecodedImage):
        raise TypeError("image must be a DecodedImage")
    if image.mode != "RGB":
        raise ValueError("full-frame attention requires an RGB image")


def _require_non_empty(value: object, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _optional_non_empty(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty(value, name)


def _identity(payload: Mapping[str, object]) -> str:
    return canonical_semantic_fingerprint(payload)


TARGET_FULL_FRAME_PREPROCESSING = TargetPreprocessingContract()


__all__ = [
    "FOCUSED_FULL_FRAME_KIND",
    "FULL_FRAME_ATTENTION_EVIDENCE_VERSION",
    "FULL_FRAME_ATTENTION_TRANSFORMATION_VERSION",
    "FULL_FRAME_ATTENTION_UNAVAILABLE_VERSION",
    "FULL_FRAME_VISUAL_INPUT_VERSION",
    "MASKED_FULL_FRAME_KIND",
    "MASK_COVERAGE_BASIS_RASTERIZED_POLYGON",
    "MULTI_OBJECT_FULL_FRAME_KIND",
    "RAW_FULL_IMAGE_KIND",
    "RAW_FULL_IMAGE_TRANSFORMATION_FINGERPRINT",
    "TARGET_FULL_FRAME_PREPROCESSING",
    "TARGET_FULL_FRAME_IMAGE_RESIZE_MODE",
    "AttentionQualityPolicy",
    "AttentionRegion",
    "FullFrameAttentionResult",
    "FullFrameAttentionVariant",
    "FullFrameTransformPolicy",
    "FullFrameVariantEvidence",
    "TargetPreprocessingContract",
    "UnavailableAttentionVariant",
    "decoded_image_content_hash",
    "generate_full_frame_attention_variants",
    "raw_full_frame_visual_input",
]
