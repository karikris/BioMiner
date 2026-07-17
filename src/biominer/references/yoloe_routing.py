"""YOLOE quality and life-stage routing for provisional references."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.detection.detector_base import DecodedImage, ObjectDetector
from biominer.detection.pipeline import DetectionPipelineResult, run_detection_pipeline
from biominer.detection.policy import DetectionPolicy, DetectionRunPolicy
from biominer.storage.parquet import write_parquet
from biominer.vision.full_frame_attention import (
    RAW_FULL_IMAGE_KIND,
    AttentionQualityPolicy,
    AttentionRegion,
    FullFrameTransformPolicy,
    generate_full_frame_attention_variants,
    raw_full_frame_visual_input,
)
from biominer.vision.target_full_frame import target_full_frame_detection_run_policy


REFERENCE_YOLOE_ROUTING_SCHEMA_VERSION = "reference-yoloe-routing-v1.0.0"
REFERENCE_YOLOE_DETECTIONS_FILE = "reference_yoloe_detections.parquet"
REFERENCE_YOLOE_ROUTES_FILE = "reference_yoloe_routes.parquet"

REFERENCE_YOLOE_ROUTE_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "reference_media_id": pl.String,
    "source": pl.String,
    "source_record_hash": pl.String,
    "route": pl.String,
    "routing_action": pl.String,
    "routing_reason": pl.String,
    "provisional_life_stage": pl.String,
    "provisional_visual_domain": pl.String,
    "subject_present": pl.Boolean,
    "subject_area_ratio": pl.Float64,
    "domain_flags": pl.List(pl.String),
    "detection_ids": pl.List(pl.String),
    "detection_count": pl.UInt32,
    "detector_backend": pl.String,
    "detector_model_id": pl.String,
    "detector_model_version": pl.String,
    "detector_checkpoint": pl.String,
    "detector_prompt_set_fingerprint": pl.String,
    "detected_at": pl.String,
    "routing_policy_version": pl.String,
    "routing_policy_fingerprint": pl.String,
    "raw_visual_input_id": pl.String,
    "raw_visual_content_hash": pl.String,
    "raw_transformation_fingerprint": pl.String,
    "attention_transform_policy_fingerprint": pl.String,
    "attention_quality_policy_fingerprint": pl.String,
    "full_frame_input_generation_succeeded": pl.Boolean,
    "species_identity_decision": pl.String,
    "run_detector_batch_retries": pl.UInt32,
    "route_evidence_fingerprint": pl.String,
}

_REFERENCE_MEDIA_ID = re.compile(r"reference-media:[0-9a-f]{64}\Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SUPPORTED_ROUTES = ("adult_field", "larval", "pinned_specimen")
_EXCLUDED_DOMAIN_ROUTES = frozenset(
    {
        "artwork_logo_tattoo_or_other_artifact",
        "no_relevant_organism",
        "possible_moth_or_other_insect",
        "pupa_or_chrysalis",
    }
)


@dataclass(frozen=True, slots=True)
class ReferenceYOLOERoutingResult:
    """Durable detections and one aggregate evidence row per reference route."""

    detections: pl.DataFrame
    routes: pl.DataFrame
    detections_path: Path
    routes_path: Path
    records_seen: int
    images_loaded: int
    image_failures: int
    detector_batch_retries: int


def run_reference_yoloe_routing(
    *,
    records: Iterable[Mapping[str, object]],
    detector: ObjectDetector,
    output_dir: str | Path,
    image_loader: Callable[[dict[str, object]], DecodedImage],
    detection_policy: DetectionPolicy | None = None,
    run_policy: DetectionRunPolicy | None = None,
) -> ReferenceYOLOERoutingResult:
    """Run one detector pass per reference and persist route evidence."""

    source_rows = _validated_source_rows(records)
    source_by_media_id = {
        str(row["reference_media_id"]): row for row in source_rows
    }
    image_by_media_id: dict[str, DecodedImage] = {}

    def load(adapted: dict[str, Any]) -> DecodedImage:
        media_id = str(adapted["flickr_photo_id"])
        image = image_loader(dict(source_by_media_id[media_id]))
        if not isinstance(image, DecodedImage):
            raise TypeError("image_loader must return a DecodedImage")
        image_by_media_id[media_id] = image
        return image

    adapted = [_detection_record(row) for row in source_rows]
    output = Path(output_dir)
    detections_path = output / REFERENCE_YOLOE_DETECTIONS_FILE
    routes_path = output / REFERENCE_YOLOE_ROUTES_FILE
    detector_result = run_detection_pipeline(
        records=adapted,
        detector=detector,
        output_path=detections_path,
        image_loader=load,
        detection_policy=detection_policy,
        run_policy=target_full_frame_detection_run_policy(run_policy),
    )
    routes = compile_reference_yoloe_routes(
        source_rows=source_rows,
        detection_rows=detector_result.frame.iter_rows(named=True),
        image_by_media_id=image_by_media_id,
        detector_batch_retries=detector_result.detector_batch_retries,
    )
    write_parquet(routes, routes_path)
    return ReferenceYOLOERoutingResult(
        detections=detector_result.frame,
        routes=routes,
        detections_path=detector_result.output_path,
        routes_path=routes_path,
        records_seen=detector_result.records_seen,
        images_loaded=detector_result.images_loaded,
        image_failures=detector_result.image_failures,
        detector_batch_retries=detector_result.detector_batch_retries,
    )


def compile_reference_yoloe_routes(
    *,
    source_rows: Iterable[Mapping[str, object]],
    detection_rows: Iterable[Mapping[str, object]],
    image_by_media_id: Mapping[str, DecodedImage],
    detector_batch_retries: int = 0,
) -> pl.DataFrame:
    """Aggregate detection rows without making any taxonomic decision."""

    if (
        isinstance(detector_batch_retries, bool)
        or not isinstance(detector_batch_retries, int)
        or detector_batch_retries < 0
    ):
        raise ValueError("detector_batch_retries must be a non-negative integer")
    sources = _validated_source_rows(source_rows)
    known_media_ids = {str(item["reference_media_id"]) for item in sources}
    detections_by_media: dict[str, list[dict[str, object]]] = {}
    for raw in detection_rows:
        row = dict(raw)
        media_id = str(row.get("flickr_photo_id") or "")
        if media_id not in known_media_ids:
            raise ValueError(f"detection references unknown media: {media_id!r}")
        detections_by_media.setdefault(media_id, []).append(row)

    rows: list[dict[str, object]] = []
    for source in sources:
        media_id = str(source["reference_media_id"])
        detections = detections_by_media.get(media_id)
        if not detections:
            raise ValueError(f"reference has no persisted detection row: {media_id}")
        rows.extend(
            _route_rows(
                source,
                detections,
                image=image_by_media_id.get(media_id),
                detector_batch_retries=detector_batch_retries,
            )
        )
    frame = pl.DataFrame(rows, schema=REFERENCE_YOLOE_ROUTE_SCHEMA)
    return frame.sort(["reference_media_id", "route"], nulls_last=True)


def _validated_source_rows(
    records: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    media_ids: set[str] = set()
    for raw in records:
        row = dict(raw)
        media_id = _required_text(row.get("reference_media_id"), "reference_media_id")
        if _REFERENCE_MEDIA_ID.fullmatch(media_id) is None:
            raise ValueError(
                "reference_media_id must be a canonical reference-media ID"
            )
        if media_id in media_ids:
            raise ValueError(f"duplicate reference_media_id: {media_id}")
        media_ids.add(media_id)
        source = _required_text(row.get("source"), "source").casefold()
        if source != "gbif":
            raise ValueError("provisional YOLOE routing permits only GBIF references")
        source_hash = _required_text(
            row.get("source_record_hash"), "source_record_hash"
        )
        if _SHA256.fullmatch(source_hash) is None:
            raise ValueError("source_record_hash must be a sha256 fingerprint")
        image_url = _required_text(
            row.get("image_url") or row.get("source_object_uri"),
            "image_url",
        )
        rows.append(
            {
                **row,
                "reference_media_id": media_id,
                "source": source,
                "source_record_hash": source_hash,
                "image_url": image_url,
            }
        )
    return sorted(rows, key=lambda row: str(row["reference_media_id"]))


def _detection_record(source: Mapping[str, object]) -> dict[str, object]:
    return {
        "source": source["source"],
        "flickr_photo_id": source["reference_media_id"],
        "source_record_hash": source["source_record_hash"],
        "image_url": source["image_url"],
        "photo_page_url": source.get("source_record_url"),
    }


def _route_rows(
    source: Mapping[str, object],
    detections: list[dict[str, object]],
    *,
    image: DecodedImage | None,
    detector_batch_retries: int,
) -> list[dict[str, object]]:
    _validate_detection_identity(detections)
    route_values = {
        str(row["bioclip_route"])
        for row in detections
        if row.get("bioclip_route") in _SUPPORTED_ROUTES
        and row.get("routing_action") in {"score", "review"}
    }
    routes: tuple[str | None, ...] = (
        tuple(sorted(route_values)) if route_values else (None,)
    )
    domain_flags = _domain_flags(detections, route_count=len(route_values))
    output: list[dict[str, object]] = []
    for route in routes:
        selected = (
            [row for row in detections if row.get("bioclip_route") == route]
            if route is not None
            else detections
        )
        action, reason = _aggregate_action(selected, domain_flags, route=route)
        full_frame = _full_frame_evidence(
            selected,
            image=image,
            media_id=str(source["reference_media_id"]),
            route=route,
        )
        flags = tuple(sorted(set(domain_flags) | set(full_frame["quality_flags"])))
        visual_domain = _visual_domain(
            detections,
            route=route,
            action=action,
            domain_flags=flags,
        )
        row: dict[str, object] = {
            "schema_version": REFERENCE_YOLOE_ROUTING_SCHEMA_VERSION,
            "reference_media_id": source["reference_media_id"],
            "source": source["source"],
            "source_record_hash": source["source_record_hash"],
            "route": route,
            "routing_action": action,
            "routing_reason": reason,
            "provisional_life_stage": _life_stage(route, detections),
            "provisional_visual_domain": visual_domain,
            "subject_present": _subject_present(detections),
            "subject_area_ratio": full_frame["subject_area_ratio"],
            "domain_flags": list(flags),
            "detection_ids": sorted(str(item["detection_id"]) for item in selected),
            "detection_count": len(selected),
            "detector_backend": detections[0]["detector_backend"],
            "detector_model_id": detections[0]["detector_model_id"],
            "detector_model_version": detections[0]["detector_model_version"],
            "detector_checkpoint": detections[0]["detector_checkpoint"],
            "detector_prompt_set_fingerprint": _single_optional(
                detections, "detector_prompt_set_fingerprint"
            ),
            "detected_at": _single_required(detections, "detected_at"),
            "routing_policy_version": _single_required(
                detections, "routing_policy_version"
            ),
            "routing_policy_fingerprint": _single_required(
                detections, "routing_policy_fingerprint"
            ),
            "raw_visual_input_id": full_frame["raw_visual_input_id"],
            "raw_visual_content_hash": full_frame["raw_visual_content_hash"],
            "raw_transformation_fingerprint": full_frame[
                "raw_transformation_fingerprint"
            ],
            "attention_transform_policy_fingerprint": (
                FullFrameTransformPolicy().fingerprint
            ),
            "attention_quality_policy_fingerprint": (
                AttentionQualityPolicy().fingerprint
            ),
            "full_frame_input_generation_succeeded": full_frame["succeeded"],
            "species_identity_decision": "not_assessed_by_yoloe",
            "run_detector_batch_retries": detector_batch_retries,
        }
        row["route_evidence_fingerprint"] = canonical_semantic_fingerprint(row)
        output.append(row)
    return output


def _domain_flags(
    detections: list[dict[str, object]], *, route_count: int
) -> tuple[str, ...]:
    flags: set[str] = set()
    routes = {str(row.get("detection_route") or "") for row in detections}
    statuses = {str(row.get("detection_status") or "") for row in detections}
    if "artwork_logo_tattoo_or_other_artifact" in routes:
        flags.add("artifact_detected")
    if "no_relevant_organism" in routes or "no_detection" in statuses:
        flags.add("no_relevant_organism_detected")
    if "ambiguous_visual_domain" in routes:
        flags.add("ambiguous_visual_domain")
    if "possible_moth_or_other_insect" in routes:
        flags.add("other_insect_detected")
    if "pupa_or_chrysalis" in routes:
        flags.add("pupa_route_not_supported")
    if route_count > 1:
        flags.add("multiple_biological_routes")
    if "failed_image_load" in statuses:
        flags.add("image_load_failed")
    if any(route in _EXCLUDED_DOMAIN_ROUTES for route in routes) and route_count:
        flags.add("mixed_biological_and_excluded_domain_evidence")
    return tuple(sorted(flags))


def _aggregate_action(
    selected: list[dict[str, object]],
    domain_flags: tuple[str, ...],
    *,
    route: str | None,
) -> tuple[str, str]:
    actions = {str(row.get("routing_action") or "") for row in selected}
    if route is None:
        return "exclude", _exclusion_reason(domain_flags)
    if "mixed_biological_and_excluded_domain_evidence" in domain_flags:
        return "review", "mixed_biological_and_excluded_domain_evidence"
    if "multiple_biological_routes" in domain_flags:
        return "review", "multiple_biological_routes"
    if "review" in actions or "ambiguous_visual_domain" in domain_flags:
        return "review", "ambiguous_domain_evidence"
    return "score", "route_supported"


def _exclusion_reason(domain_flags: tuple[str, ...]) -> str:
    for reason in (
        "image_load_failed",
        "artifact_detected",
        "no_relevant_organism_detected",
        "other_insect_detected",
        "pupa_route_not_supported",
        "ambiguous_visual_domain",
    ):
        if reason in domain_flags:
            return reason
    return "no_supported_reference_route"


def _full_frame_evidence(
    detections: list[dict[str, object]],
    *,
    image: DecodedImage | None,
    media_id: str,
    route: str | None,
) -> dict[str, object]:
    if image is None:
        return {
            "succeeded": False,
            "raw_visual_input_id": None,
            "raw_visual_content_hash": None,
            "raw_transformation_fingerprint": None,
            "subject_area_ratio": None,
            "quality_flags": ("full_frame_input_unavailable",),
        }
    raw = raw_full_frame_visual_input(image)
    if route is None:
        return {
            "succeeded": True,
            "raw_visual_input_id": raw.visual_input_id,
            "raw_visual_content_hash": raw.visual_content_hash,
            "raw_transformation_fingerprint": raw.transformation_fingerprint,
            "subject_area_ratio": None,
            "quality_flags": (),
        }
    regions = tuple(_attention_region(row, route) for row in detections)
    try:
        result = generate_full_frame_attention_variants(
            image,
            regions,
            source_type="reference",
            source_record_id=media_id,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return {
            "succeeded": False,
            "raw_visual_input_id": raw.visual_input_id,
            "raw_visual_content_hash": raw.visual_content_hash,
            "raw_transformation_fingerprint": raw.transformation_fingerprint,
            "subject_area_ratio": None,
            "quality_flags": ("full_frame_input_generation_failed",),
        }
    raw_evidence = next(
        item
        for item in result.evidence
        if item.visual_input_kind == RAW_FULL_IMAGE_KIND
    )
    return {
        "succeeded": True,
        "raw_visual_input_id": raw.visual_input_id,
        "raw_visual_content_hash": raw.visual_content_hash,
        "raw_transformation_fingerprint": raw.transformation_fingerprint,
        "subject_area_ratio": raw_evidence.subject_area_ratio,
        "quality_flags": raw_evidence.visual_input_quality_flags,
    }


def _attention_region(row: Mapping[str, object], route: str) -> AttentionRegion:
    bbox = row.get("bbox_xyxyn")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("routed detection requires bbox_xyxyn")
    polygon = _polygon(row.get("mask_polygon_xyn"))
    return AttentionRegion(
        source_detection_id=_required_text(row.get("detection_id"), "detection_id"),
        route=route,
        bbox_xyxyn=tuple(float(value) for value in bbox),  # type: ignore[arg-type]
        mask_polygon_xyn=polygon,
        detector_score=(
            None
            if row.get("detector_score") is None
            else float(row["detector_score"])
        ),
    )


def _visual_domain(
    detections: list[dict[str, object]],
    *,
    route: str | None,
    action: str,
    domain_flags: tuple[str, ...],
) -> str:
    if action == "review" or "ambiguous_visual_domain" in domain_flags:
        return "ambiguous"
    if route == "pinned_specimen":
        return "pinned_specimen"
    if route in {"adult_field", "larval"}:
        return "live_field"
    if "pupa_route_not_supported" in domain_flags:
        return "live_field"
    prompts = {str(row.get("detector_prompt") or "") for row in detections}
    if any("tattoo" in prompt for prompt in prompts):
        return "tattoo"
    if any("logo" in prompt for prompt in prompts):
        return "logo"
    if "artifact_detected" in domain_flags:
        return "artwork"
    if "no_relevant_organism_detected" in domain_flags:
        return "unsuitable"
    return "ambiguous"


def _life_stage(
    route: str | None, detections: list[dict[str, object]]
) -> str:
    if route == "adult_field":
        return "adult"
    if route == "larval":
        return "larva"
    if route == "pinned_specimen":
        return "unknown"
    if any(row.get("detection_route") == "pupa_or_chrysalis" for row in detections):
        return "pupa"
    return "unknown"


def _subject_present(detections: list[dict[str, object]]) -> bool:
    biological_routes = {
        "adult_butterfly_field",
        "caterpillar_field",
        "pupa_or_chrysalis",
        "pinned_specimen",
        "possible_moth_or_other_insect",
        "ambiguous_visual_domain",
    }
    return any(row.get("detection_route") in biological_routes for row in detections)


def _polygon(
    value: object,
) -> tuple[tuple[float, float], ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("mask_polygon_xyn must be a point list")
    points: list[tuple[float, float]] = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("mask_polygon_xyn points must contain two values")
        points.append((float(point[0]), float(point[1])))
    return tuple(points)


def _validate_detection_identity(detections: list[dict[str, object]]) -> None:
    for field in (
        "source",
        "source_record_hash",
        "detector_backend",
        "detector_model_id",
        "detector_model_version",
        "detector_checkpoint",
        "detected_at",
        "routing_policy_version",
        "routing_policy_fingerprint",
    ):
        _single_required(detections, field)
    if _single_required(detections, "source") != "gbif":
        raise ValueError("reference detection source must be GBIF")


def _single_required(rows: list[dict[str, object]], field: str) -> str:
    values = {_required_text(row.get(field), field) for row in rows}
    if len(values) != 1:
        raise ValueError(f"inconsistent {field} across reference detections")
    return next(iter(values))


def _single_optional(
    rows: list[dict[str, object]], field: str
) -> str | None:
    values = {
        str(row[field]).strip() for row in rows if row.get(field) is not None
    }
    if len(values) > 1:
        raise ValueError(f"inconsistent {field} across reference detections")
    return next(iter(values), None)


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonblank text")
    return value.strip()


__all__ = [
    "REFERENCE_YOLOE_DETECTIONS_FILE",
    "REFERENCE_YOLOE_ROUTES_FILE",
    "REFERENCE_YOLOE_ROUTE_SCHEMA",
    "REFERENCE_YOLOE_ROUTING_SCHEMA_VERSION",
    "ReferenceYOLOERoutingResult",
    "compile_reference_yoloe_routes",
    "run_reference_yoloe_routing",
]
