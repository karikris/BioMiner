from __future__ import annotations

from pathlib import Path
import subprocess

from flickr_bio_occurrence.vision.bioclip import (
    BioClipClassifier,
    DEFAULT_BIOCLIP_LABELS,
    ExternalBioClipScorer,
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


def test_bioclip_classifier_builds_prediction_with_injected_scorer() -> None:
    classifier = BioClipClassifier(
        runtime=_runtime(),
        scorer=lambda image_path, labels: {
            "a photo of a moth": 0.12,
            "a photo of Papilio demoleus": 0.95,
        },
    )

    record = classifier.classify_image(
        flickr_photo_id="123",
        image_path=Path("/tmp/image.jpg"),
        image_hash="sha256:image",
        image_url_used="https://live.staticflickr.com/example.jpg",
        resolved_scientific_name="Papilio demoleus",
        text_evidence_present=True,
        labels=["a photo of a moth", "a photo of Papilio demoleus"],
        top_k=2,
    )

    assert record["top1_label"] == "a photo of Papilio demoleus"
    assert record["top1_score"] == 0.95
    assert record["species_agreement_status"] == "exact_species_agreement"


def test_bioclip_classifier_fails_when_runtime_unavailable() -> None:
    unavailable = BioClipRuntime(
        model=_runtime().model,
        home=Path("/missing"),
        venv_python=Path("/missing/.venv/bin/python"),
        package_version="",
        available=False,
        unavailable_reason="missing runtime",
    )

    classifier = BioClipClassifier(runtime=unavailable, scorer=lambda image_path, labels: {})

    try:
        classifier.classify_image(
            flickr_photo_id="123",
            image_path=Path("/tmp/image.jpg"),
            image_hash="sha256:image",
            image_url_used="https://live.staticflickr.com/example.jpg",
            resolved_scientific_name="Papilio demoleus",
            text_evidence_present=True,
        )
    except RuntimeError as exc:
        assert "BioCLIP runtime is not available" in str(exc)
    else:
        raise AssertionError("expected unavailable runtime to raise")


def test_external_bioclip_scorer_invokes_runtime_python_with_json() -> None:
    calls: list[dict[str, object]] = []

    def fake_run(cmd, *, input, capture_output, check, text):  # noqa: ANN001 - mirrors subprocess.run signature.
        calls.append(
            {
                "cmd": cmd,
                "input": input,
                "capture_output": capture_output,
                "check": check,
                "text": text,
            }
        )
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout='{"scores":{"a photo of Papilio demoleus":0.97,"a photo of a moth":0.03}}',
            stderr="",
        )

    scorer = ExternalBioClipScorer(runtime=_runtime(), runner=fake_run)
    scores = scorer(Path("/tmp/image.jpg"), ["a photo of Papilio demoleus", "a photo of a moth"])

    assert scores["a photo of Papilio demoleus"] == 0.97
    assert calls[0]["cmd"][0] == "/home/toffe/bioclip25/.venv/bin/python"
    assert "imageomics/bioclip-2" in calls[0]["input"]
    assert "data/cache/huggingface" in calls[0]["input"]


def test_external_bioclip_scorer_raises_with_worker_stderr() -> None:
    def fake_run(cmd, *, input, capture_output, check, text):  # noqa: ANN001 - mirrors subprocess.run signature.
        return subprocess.CompletedProcess(args=cmd, returncode=2, stdout="", stderr="model missing")

    scorer = ExternalBioClipScorer(runtime=_runtime(), runner=fake_run)

    try:
        scorer(Path("/tmp/image.jpg"), ["a photo of Papilio demoleus"])
    except RuntimeError as exc:
        assert "BioCLIP worker failed" in str(exc)
        assert "model missing" in str(exc)
    else:
        raise AssertionError("expected worker failure to raise")
