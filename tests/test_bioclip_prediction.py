from __future__ import annotations

from pathlib import Path

from flickr_bio_occurrence.vision.bioclip import (
    DEFAULT_BIOCLIP_LABELS,
    build_vision_prediction_record,
    classify_species_agreement,
)
from flickr_bio_occurrence.vision.model_registry import BioClipRuntime, ModelConfig


def _runtime() -> BioClipRuntime:
    model = ModelConfig(
        model_id="bioclip2_5_huge",
        display_name="BioCLIP 2.5 Huge",
        role="preferred",
        status="use_if_available",
        task="biology image-text classification and embedding",
        model_name="imageomics/bioclip-2",
        checkpoint="BioCLIP 2.5 Huge OpenCLIP ViT-H/14 checkpoint",
        package_name="open_clip_torch",
        package_version="pin_when_installed",
        model_hash="sha256:test",
    )
    return BioClipRuntime(
        model=model,
        home=Path("/home/toffe/bioclip25"),
        venv_python=Path("/home/toffe/bioclip25/.venv/bin/python"),
        package_version="3.3.0",
        available=True,
    )


def test_default_bioclip_labels_include_required_papilio_prompts() -> None:
    assert "a photo of Papilio demoleus" in DEFAULT_BIOCLIP_LABELS
    assert "a photo of lime butterfly" in DEFAULT_BIOCLIP_LABELS
    assert "a photo of chequered swallowtail" in DEFAULT_BIOCLIP_LABELS
    assert "a photo of citrus swallowtail" in DEFAULT_BIOCLIP_LABELS
    assert "a photo of a pinned museum specimen" in DEFAULT_BIOCLIP_LABELS
    assert "a photo of artwork or illustration" in DEFAULT_BIOCLIP_LABELS


def test_classify_species_agreement_detects_exact_species_agreement() -> None:
    status = classify_species_agreement(
        resolved_scientific_name="Papilio demoleus",
        topk_labels=["a photo of Papilio demoleus", "a photo of lime butterfly"],
        text_evidence_present=True,
    )

    assert status == "exact_species_agreement"


def test_classify_species_agreement_routes_conflicting_text_and_vision_to_review() -> None:
    status = classify_species_agreement(
        resolved_scientific_name="Papilio demoleus",
        topk_labels=["a photo of a moth", "a photo of artwork or illustration"],
        text_evidence_present=True,
    )

    assert status == "text_vision_conflict"


def test_build_vision_prediction_record_preserves_model_and_topk_metadata() -> None:
    record = build_vision_prediction_record(
        flickr_photo_id="123",
        runtime=_runtime(),
        image_hash="sha256:image",
        image_url_used="https://live.staticflickr.com/example.jpg",
        resolved_scientific_name="Papilio demoleus",
        text_evidence_present=True,
        topk=[("a photo of Papilio demoleus", 0.91), ("a photo of a butterfly", 0.08)],
    )

    assert record["flickr_photo_id"] == "123"
    assert record["model_family"] == "bioclip"
    assert record["model_name"] == "imageomics/bioclip-2"
    assert record["model_version"] == "bioclip2_5_huge"
    assert record["model_checkpoint"] == "BioCLIP 2.5 Huge OpenCLIP ViT-H/14 checkpoint"
    assert record["model_hash"] == "sha256:test"
    assert record["runtime_package_version"] == "3.3.0"
    assert record["top1_label"] == "a photo of Papilio demoleus"
    assert record["top1_score"] == 0.91
    assert record["species_agreement_status"] == "exact_species_agreement"
    assert record["vision_review_required"] is False
    assert record["topk_json"][0]["label"] == "a photo of Papilio demoleus"
