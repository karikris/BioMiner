"""Read-only codecs for historical YOLOE26 evidence; no model execution.

New visual routing uses biominer.detection.embeddinggemma2. Historical names
and prompt fingerprints remain unchanged so old evidence is never relabelled.
"""
from __future__ import annotations

import hashlib
import json
from typing import Sequence

from biominer.detection.detector_base import (DetectionCandidate, detector_label_is_taxon_like, normalize_detector_prompt, normalize_mask_polygon_xyn)


DEFAULT_YOLOE26_CHECKPOINT = "yoloe-26s-seg.pt"

ALLOWED_YOLOE26_CHECKPOINTS = (
    "yoloe-26n-seg.pt",
    "yoloe-26s-seg.pt",
    "yoloe-26m-seg.pt",
    "yoloe-26l-seg.pt",
    "yoloe-26x-seg.pt",
)

DEFAULT_YOLOE26_PROMPTS = (
    "butterfly",
    "moth",
    "caterpillar",
    "chrysalis",
    "pupa",
    "insect",
    "butterfly wing",
    "pinned butterfly specimen",
    "butterfly specimen",
    "lepidoptera",
    "flower",
    "leaf",
    "person",
    "hand",
    "drawing",
    "painting",
    "logo",
    "text",
    "sign",
    "museum label",
)

YOLOE26_PROMPT_LABEL_MAP = {
    "ant": "insect_like",
    "bee": "insect_like",
    "beetle": "insect_like",
    "butterfly": "adult_butterfly",
    "butterfly wing": "possible_adult_butterfly",
    "caddisfly": "insect_like",
    "cicada": "insect_like",
    "cockroach": "insect_like",
    "cricket": "insect_like",
    "damselfly": "insect_like",
    "dragonfly": "insect_like",
    "earwig": "insect_like",
    "fly": "insect_like",
    "grasshopper": "insect_like",
    "lacewing": "insect_like",
    "mantis": "insect_like",
    "mayfly": "insect_like",
    "mosquito": "insect_like",
    "pinned insect specimen": "pinned_specimen",
    "pinned butterfly specimen": "pinned_specimen",
    "butterfly specimen": "pinned_specimen",
    "stick insect": "insect_like",
    "stonefly": "insect_like",
    "termite": "insect_like",
    "true bug": "insect_like",
    "wasp": "insect_like",
    "lepidoptera": "possible_adult_butterfly",
    "moth": "moth_like",
    "caterpillar": "caterpillar",
    "chrysalis": "pupa",
    "pupa": "pupa",
    "insect": "insect_like",
    "flower": "no_relevant_organism",
    "leaf": "no_relevant_organism",
    "person": "no_relevant_organism",
    "hand": "no_relevant_organism",
    "drawing": "artifact",
    "painting": "artifact",
    "logo": "artifact",
    "text": "artifact",
    "sign": "artifact",
    "museum label": "artifact",
}

def default_yoloe26_prompts(*, include_hard_negative_prompts: bool = True) -> tuple[str, ...]:
    if include_hard_negative_prompts:
        return DEFAULT_YOLOE26_PROMPTS
    return tuple(
        prompt
        for prompt in DEFAULT_YOLOE26_PROMPTS
        if yoloe26_coarse_label(prompt) not in {"artifact", "no_relevant_organism", "hard_negative"}
    )

def yoloe26_coarse_label(prompt: str) -> str:
    if detector_label_is_taxon_like(prompt):
        raise ValueError(f"YOLOE-26 prompts must be object proposals, not taxon labels: {prompt!r}")
    normalized = normalize_detector_prompt(prompt)
    mapped = YOLOE26_PROMPT_LABEL_MAP.get(normalized)
    if mapped:
        return mapped
    raise ValueError(f"unsupported YOLOE-26 object prompt: {prompt!r}")

def yoloe26_prompt_set_fingerprint(prompt_classes: Sequence[str]) -> str:
    prompts = list(_normalized_prompt_classes(prompt_classes))
    payload = json.dumps(
        {"prompts": prompts, "schema_version": "yoloe26-prompt-set-v1"},
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

def detections_from_yoloe_result(result: object, *, prompt_classes: Sequence[str] | None = None) -> list[DetectionCandidate]:
    ordered_prompts = _validated_result_prompts(getattr(result, "names", None))
    if prompt_classes is not None:
        expected_prompts = _normalized_prompt_classes(prompt_classes)
        if sorted(expected_prompts) != sorted(ordered_prompts):
            raise ValueError("YOLOE result.names does not match the configured prompt set")
    prompt_set_fingerprint = yoloe26_prompt_set_fingerprint(ordered_prompts)
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    box_rows = list(boxes)
    mask_polygons = _result_mask_polygons_xyn(result, expected_count=len(box_rows))
    rows: list[DetectionCandidate] = []
    for index, box in enumerate(box_rows):
        xyxy = _first_vector(getattr(box, "xyxy", ()))
        if len(xyxy) != 4:
            continue
        cls_id = _result_class_id(getattr(box, "cls", -1))
        if cls_id >= len(ordered_prompts):
            raise ValueError(f"YOLOE result class id {cls_id} is outside result.names")
        prompt = ordered_prompts[cls_id]
        score = float(_first_scalar(getattr(box, "conf", 0.0), default=0.0))
        rows.append(
            DetectionCandidate(
                label=yoloe26_coarse_label(prompt),
                score=score,
                bbox_xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                objectness_score=score,
                detector_prompt=prompt,
                detector_class_id=cls_id,
                detector_prompt_set_fingerprint=prompt_set_fingerprint,
                mask_polygon_xyn=mask_polygons[index],
            )
        )
    return rows

def _result_mask_polygons_xyn(
    result: object,
    *,
    expected_count: int,
) -> list[tuple[tuple[float, float], ...] | None]:
    masks = getattr(result, "masks", None)
    if masks is None:
        return [None] * expected_count
    polygons = getattr(masks, "xyn", None)
    if not isinstance(polygons, list | tuple):
        raise ValueError("YOLOE result masks.xyn must be an ordered sequence")
    if len(polygons) != expected_count:
        raise ValueError("YOLOE result masks must align one-to-one with boxes")
    return [normalize_mask_polygon_xyn(polygon) for polygon in polygons]

def _validate_prompt_classes(prompt_classes: Sequence[str]) -> None:
    raw_prompts = [str(prompt) for prompt in prompt_classes]
    if not raw_prompts:
        raise ValueError("YOLOE-26 prompt classes must not be empty")
    for prompt in raw_prompts:
        yoloe26_coarse_label(prompt)
    normalized = [normalize_detector_prompt(prompt) for prompt in raw_prompts]
    if len(set(normalized)) != len(normalized):
        raise ValueError("YOLOE-26 prompt classes must not contain duplicates")

def _normalized_prompt_classes(prompt_classes: Sequence[str]) -> tuple[str, ...]:
    _validate_prompt_classes(prompt_classes)
    return tuple(normalize_detector_prompt(prompt) for prompt in prompt_classes)

def _validated_result_prompts(names: object) -> tuple[str, ...]:
    if isinstance(names, list | tuple):
        mapped_names = dict(enumerate(names))
    elif isinstance(names, dict):
        mapped_names: dict[int, object] = {}
        for raw_key, value in names.items():
            if isinstance(raw_key, bool):
                raise ValueError("YOLOE result.names class ids must be contiguous non-negative integers")
            try:
                key = int(raw_key)
            except (TypeError, ValueError) as exc:
                raise ValueError("YOLOE result.names class ids must be contiguous non-negative integers") from exc
            if str(raw_key).strip() != str(key):
                raise ValueError("YOLOE result.names class ids must be contiguous non-negative integers")
            if key in mapped_names:
                raise ValueError("YOLOE result.names contains duplicate class ids")
            mapped_names[key] = value
    else:
        raise ValueError("YOLOE result.names must be an ordered list or class-id mapping")
    if sorted(mapped_names) != list(range(len(mapped_names))):
        raise ValueError("YOLOE result.names class ids must be contiguous from zero")
    if not mapped_names:
        raise ValueError("YOLOE result.names must contain at least one prompt")
    raw_prompts = tuple(str(mapped_names[index]) for index in range(len(mapped_names)))
    for prompt in raw_prompts:
        yoloe26_coarse_label(prompt)
    prompts = tuple(normalize_detector_prompt(prompt) for prompt in raw_prompts)
    if len(set(prompts)) != len(prompts):
        raise ValueError("YOLOE result.names must not contain duplicate prompts")
    return prompts

def _result_class_id(value: object) -> int:
    raw = _first_scalar(value, default=-1)
    if not float(raw).is_integer() or raw < 0:
        raise ValueError(f"YOLOE result class id must be a non-negative integer, got {raw!r}")
    return int(raw)

def _first_vector(value: object) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list | tuple) and value and isinstance(value[0], list | tuple):
        value = value[0]
    if not isinstance(value, list | tuple):
        return []
    return [float(item) for item in value]

def _first_scalar(value: object, *, default: float) -> float:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list | tuple):
        if not value:
            return default
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _candidate_to_payload(candidate: DetectionCandidate) -> dict[str, object]:
    return {
        "label": candidate.label,
        "score": candidate.score,
        "bbox_xyxy": list(candidate.bbox_xyxy),
        "objectness_score": candidate.objectness_score,
        "detector_prompt": candidate.detector_prompt,
        "detector_class_id": candidate.detector_class_id,
        "detector_prompt_set_fingerprint": candidate.detector_prompt_set_fingerprint,
        "mask_polygon_xyn": None
        if candidate.mask_polygon_xyn is None
        else [list(point) for point in candidate.mask_polygon_xyn],
    }

def _detections_from_payload(payload: dict[str, object]) -> list[list[DetectionCandidate]]:
    return [
        [
            DetectionCandidate(
                label=str(candidate.get("label") or ""),
                score=float(candidate.get("score") or 0.0),
                bbox_xyxy=tuple(float(value) for value in candidate.get("bbox_xyxy", ())),  # type: ignore[arg-type]
                objectness_score=None
                if candidate.get("objectness_score") is None
                else float(candidate.get("objectness_score")),
                detector_prompt=None
                if candidate.get("detector_prompt") is None
                else str(candidate.get("detector_prompt")),
                detector_class_id=None
                if candidate.get("detector_class_id") is None
                else int(candidate.get("detector_class_id")),
                detector_prompt_set_fingerprint=None
                if candidate.get("detector_prompt_set_fingerprint") is None
                else str(candidate.get("detector_prompt_set_fingerprint")),
                mask_polygon_xyn=normalize_mask_polygon_xyn(candidate.get("mask_polygon_xyn")),
            )
            for candidate in batch
        ]
        for batch in payload.get("detections", [])
    ]
