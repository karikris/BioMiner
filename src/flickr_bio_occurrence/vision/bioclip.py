from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from flickr_bio_occurrence.vision.model_registry import BioClipRuntime


DEFAULT_BIOCLIP_LABELS = (
    "a photo of Papilio demoleus",
    "a photo of lime butterfly",
    "a photo of chequered swallowtail",
    "a photo of citrus swallowtail",
    "a photo of a swallowtail butterfly",
    "a photo of a butterfly",
    "a photo of a moth",
    "a photo of a caterpillar",
    "a photo of a pupa or chrysalis",
    "a photo of a pinned museum specimen",
    "a photo of artwork or illustration",
)


PAPILIO_DEMOLEUS_VISUAL_LABELS = {
    "a photo of Papilio demoleus",
    "a photo of lime butterfly",
    "a photo of chequered swallowtail",
    "a photo of citrus swallowtail",
}

SWALLOWTAIL_VISUAL_LABELS = {
    *PAPILIO_DEMOLEUS_VISUAL_LABELS,
    "a photo of a swallowtail butterfly",
}

BUTTERFLY_VISUAL_LABELS = {
    *SWALLOWTAIL_VISUAL_LABELS,
    "a photo of a butterfly",
}

NON_WILD_OR_CONFLICT_LABELS = {
    "a photo of a moth",
    "a photo of a caterpillar",
    "a photo of a pupa or chrysalis",
    "a photo of a pinned museum specimen",
    "a photo of artwork or illustration",
}


def classify_species_agreement(
    *,
    resolved_scientific_name: str,
    topk_labels: list[str],
    text_evidence_present: bool,
) -> str:
    if not topk_labels:
        return "text_only" if text_evidence_present else "uncertain"

    normalized_labels = {_normalize_label(label) for label in topk_labels}
    species_label = _normalize_label(f"a photo of {resolved_scientific_name}")
    if species_label in normalized_labels:
        return "exact_species_agreement" if text_evidence_present else "vision_only"

    if resolved_scientific_name == "Papilio demoleus":
        if normalized_labels & {_normalize_label(label) for label in PAPILIO_DEMOLEUS_VISUAL_LABELS}:
            return "exact_species_agreement" if text_evidence_present else "vision_only"
        if normalized_labels & {_normalize_label(label) for label in SWALLOWTAIL_VISUAL_LABELS}:
            return "same_family_agreement" if text_evidence_present else "vision_only"

    if normalized_labels & {_normalize_label(label) for label in BUTTERFLY_VISUAL_LABELS}:
        return "same_family_agreement" if text_evidence_present else "vision_only"

    if normalized_labels & {_normalize_label(label) for label in NON_WILD_OR_CONFLICT_LABELS}:
        return "text_vision_conflict" if text_evidence_present else "non_butterfly"

    return "text_vision_conflict" if text_evidence_present else "uncertain"


def build_vision_prediction_record(
    *,
    flickr_photo_id: str,
    runtime: BioClipRuntime,
    image_hash: str,
    image_url_used: str,
    resolved_scientific_name: str,
    text_evidence_present: bool,
    topk: list[tuple[str, float]],
) -> dict[str, Any]:
    topk_json = [{"label": label, "score": float(score)} for label, score in topk]
    topk_labels = [item["label"] for item in topk_json]
    agreement_status = classify_species_agreement(
        resolved_scientific_name=resolved_scientific_name,
        topk_labels=topk_labels,
        text_evidence_present=text_evidence_present,
    )
    return {
        "flickr_photo_id": flickr_photo_id,
        "model_family": "bioclip",
        "model_name": runtime.model.model_name,
        "model_version": runtime.model.model_id,
        "model_checkpoint": runtime.model.checkpoint,
        "model_hash": runtime.model.model_hash,
        "runtime_package_version": runtime.package_version,
        "image_hash": image_hash,
        "image_url_used": image_url_used,
        "top1_label": topk_json[0]["label"] if topk_json else None,
        "top1_score": topk_json[0]["score"] if topk_json else None,
        "topk_json": topk_json,
        "species_agreement_status": agreement_status,
        "vision_review_required": _vision_review_required(agreement_status),
        "created_at": datetime.now(UTC).isoformat(),
    }


def _vision_review_required(agreement_status: str) -> bool:
    return agreement_status not in {"exact_species_agreement", "same_genus_agreement"}


def _normalize_label(value: str) -> str:
    return " ".join(value.casefold().split())
