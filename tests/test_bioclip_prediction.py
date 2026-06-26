from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from biominer.bioclip.bioclip import (
    BioClipClassifier,
    DEFAULT_BIOCLIP_LABELS,
    DEFAULT_TRIAGE_LABELS,
    ExternalBioClipScorer,
    PersistentBioClipScorer,
    build_vision_prediction_record,
    classify_species_agreement,
)
from biominer.bioclip.model_registry import BioClipRuntime, ModelConfig


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
    assert "image_paths" in calls[0]["input"]
    assert '"device": "auto"' in calls[0]["input"]
    assert "imageomics/bioclip-2" in calls[0]["input"]
    assert "data/cache/huggingface" in calls[0]["input"]


def test_external_bioclip_scorer_formats_batch_request() -> None:
    calls: list[dict[str, object]] = []

    def fake_run(cmd, *, input, capture_output, check, text):  # noqa: ANN001 - mirrors subprocess.run signature.
        calls.append({"cmd": cmd, "input": input})
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=(
                '{"scores_by_image":['
                '{"a photo of Papilio demoleus":0.9,"a photo of a moth":0.1},'
                '{"a photo of Papilio demoleus":0.2,"a photo of a moth":0.8}'
                "]}"
            ),
            stderr="",
        )

    scorer = ExternalBioClipScorer(runtime=_runtime(), runner=fake_run)
    scores = scorer.score_batch(
        [Path("/tmp/1.jpg"), Path("/tmp/2.jpg")],
        ["a photo of Papilio demoleus", "a photo of a moth"],
    )

    assert scores[0]["a photo of Papilio demoleus"] == 0.9
    assert scores[1]["a photo of a moth"] == 0.8
    assert '"image_paths": ["/tmp/1.jpg", "/tmp/2.jpg"]' in calls[0]["input"]
    assert '"device": "auto"' in calls[0]["input"]


def test_external_bioclip_scorer_formats_label_set_request() -> None:
    calls: list[dict[str, object]] = []

    def fake_run(cmd, *, input, capture_output, check, text):  # noqa: ANN001 - mirrors subprocess.run signature.
        calls.append({"cmd": cmd, "input": input})
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=(
                '{"scores_by_image_by_label_set":{'
                '"species":[{"a photo of Papilio demoleus":0.8}],'
                '"triage":[{"a photo of an adult butterfly":0.9}]'
                "}}"
            ),
            stderr="",
        )

    scorer = ExternalBioClipScorer(runtime=_runtime(), runner=fake_run)
    scores = scorer.score_label_sets_batch(
        [Path("/tmp/1.jpg")],
        {
            "species": ["a photo of Papilio demoleus"],
            "triage": ["a photo of an adult butterfly"],
        },
    )

    assert scores["species"][0]["a photo of Papilio demoleus"] == 0.8
    assert scores["triage"][0]["a photo of an adult butterfly"] == 0.9
    assert '"label_sets": {"species": ["a photo of Papilio demoleus"], "triage": ["a photo of an adult butterfly"]}' in calls[0]["input"]


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


def test_persistent_bioclip_scorer_reuses_worker_for_batches_and_uses_auto_device() -> None:
    writes: list[str] = []

    class FakeStdin:
        def write(self, value: str) -> None:
            writes.append(value)

        def flush(self) -> None:
            return None

    class FakeStdout:
        def __init__(self) -> None:
            self.lines = iter(
                [
                    '{"ready":true,"device":"cuda","gpu_name":"NVIDIA GeForce RTX 3060"}\n',
                    '{"scores_by_image":[{"a photo of Papilio demoleus":0.97},{"a photo of Papilio demoleus":0.98}]}\n',
                    '{"scores_by_image":[{"a photo of Papilio demoleus":0.96}]}\n',
                ]
            )

        def readline(self) -> str:
            return next(self.lines)

    class FakeProcess:
        def __init__(self, cmd) -> None:  # noqa: ANN001 - mirrors subprocess.Popen.
            self.cmd = cmd
            self.stdin = FakeStdin()
            self.stdout = FakeStdout()
            self.stderr = None
            self.returncode = None
            self.terminated = False
            self.waited = False

        def poll(self):  # noqa: ANN202 - mirrors subprocess.Popen.
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout=None):  # noqa: ANN001, ANN202 - mirrors subprocess.Popen.
            self.waited = True
            self.returncode = 0
            return 0

    processes: list[FakeProcess] = []

    def fake_popen(cmd, **kwargs):  # noqa: ANN001, ANN202 - mirrors subprocess.Popen.
        process = FakeProcess(cmd)
        processes.append(process)
        return process

    scorer = PersistentBioClipScorer(runtime=_runtime(), popen=fake_popen)
    try:
        first = scorer.score_batch(
            [Path("/tmp/1.jpg"), Path("/tmp/2.jpg")],
            ["a photo of Papilio demoleus"],
        )
        second = scorer(Path("/tmp/3.jpg"), ["a photo of Papilio demoleus"])
    finally:
        scorer.close()

    assert first[0]["a photo of Papilio demoleus"] == 0.97
    assert first[1]["a photo of Papilio demoleus"] == 0.98
    assert second["a photo of Papilio demoleus"] == 0.96
    assert len(processes) == 1
    assert "--persistent" in processes[0].cmd
    assert '"image_paths": ["/tmp/1.jpg", "/tmp/2.jpg"]' in writes[0]
    assert '"device": "auto"' in writes[0]
    assert '"device": "auto"' in writes[1]
    assert '"shutdown": true' in writes[-1]
    assert processes[0].waited is True


def test_persistent_bioclip_scorer_reuses_worker_for_label_sets() -> None:
    writes: list[str] = []

    class FakeStdin:
        def write(self, value: str) -> None:
            writes.append(value)

        def flush(self) -> None:
            return None

    class FakeStdout:
        def __init__(self) -> None:
            self.lines = iter(
                [
                    '{"ready":true,"device":"cuda","gpu_name":"NVIDIA GeForce RTX 3060"}\n',
                    '{"scores_by_image_by_label_set":{"species":[{"a photo of Papilio demoleus":0.97}],"triage":[{"a photo of an adult butterfly":0.98}]}}\n',
                ]
            )

        def readline(self) -> str:
            return next(self.lines)

    class FakeProcess:
        def __init__(self, cmd) -> None:  # noqa: ANN001 - mirrors subprocess.Popen.
            self.cmd = cmd
            self.stdin = FakeStdin()
            self.stdout = FakeStdout()
            self.stderr = None
            self.returncode = None

        def poll(self):  # noqa: ANN202 - mirrors subprocess.Popen.
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0

        def wait(self, timeout=None):  # noqa: ANN001, ANN202 - mirrors subprocess.Popen.
            self.returncode = 0
            return 0

    processes: list[FakeProcess] = []

    def fake_popen(cmd, **kwargs):  # noqa: ANN001, ANN202 - mirrors subprocess.Popen.
        process = FakeProcess(cmd)
        processes.append(process)
        return process

    scorer = PersistentBioClipScorer(runtime=_runtime(), popen=fake_popen)
    try:
        scores = scorer.score_label_sets_batch(
            [Path("/tmp/1.jpg")],
            {
                "species": ["a photo of Papilio demoleus"],
                "triage": ["a photo of an adult butterfly"],
            },
        )
    finally:
        scorer.close()

    assert scores["species"][0]["a photo of Papilio demoleus"] == 0.97
    assert scores["triage"][0]["a photo of an adult butterfly"] == 0.98
    assert len(processes) == 1
    assert '"label_sets": {"species": ["a photo of Papilio demoleus"], "triage": ["a photo of an adult butterfly"]}' in writes[0]


def test_bioclip_classifier_builds_species_and_triage_prediction_with_label_sets() -> None:
    class FakeScorer:
        def score_label_sets_batch(self, image_paths, label_sets):  # noqa: ANN001 - test fake.
            assert label_sets["triage"] == ["a photo of an adult butterfly"]
            return {
                "species": [{"a photo of Papilio demoleus": 0.91, "a photo of Papilio machaon": 0.09}],
                "triage": [{"a photo of an adult butterfly": 0.88}],
            }

    classifier = BioClipClassifier(runtime=_runtime(), scorer=FakeScorer())
    records = classifier.classify_images_with_label_sets(
        [
            {
                "flickr_photo_id": "1",
                "image_path": "/tmp/1.jpg",
                "image_hash": "sha256:image",
                "image_url_used": "https://live.staticflickr.com/1.jpg",
                "resolved_scientific_name": "Papilio demoleus",
                "text_evidence_present": True,
            }
        ],
        label_sets={
            "species": ["a photo of Papilio demoleus", "a photo of Papilio machaon"],
            "triage": ["a photo of an adult butterfly"],
        },
    )

    record = records[0]
    assert record["species_top1_label"] == "a photo of Papilio demoleus"
    assert record["species_top1_score"] == 0.91
    assert record["triage_top1_label"] == "a photo of an adult butterfly"
    assert record["triage_top1_score"] == 0.88
    assert record["species_top1_top2_margin"] == pytest.approx(0.82)
    assert record["triage_top1_top2_margin"] is None
    assert record["species_topk_entropy"] > 0
    assert record["triage_topk_entropy"] == 0
    assert record["triage_group_top"] == "adult_butterfly"
    assert record["triage_group_scores"]["adult_butterfly"] == pytest.approx(0.88)
    assert "a photo of an adult butterfly" in DEFAULT_TRIAGE_LABELS


def test_bioclip_classifier_aggregates_species_prompt_variants() -> None:
    from biominer.bioclip.prompt_templates import PromptVariant

    class FakeScorer:
        def score_label_sets_batch(self, image_paths, label_sets):  # noqa: ANN001 - test fake.
            return {
                "species": [
                    {
                        "a photo of Papilio demoleus": 0.62,
                        "a photo of lime butterfly": 0.84,
                        "a photo of Papilio machaon": 0.21,
                    }
                ],
                "triage": [{"a photo of an adult butterfly": 0.91}],
            }

    classifier = BioClipClassifier(runtime=_runtime(), scorer=FakeScorer())
    records = classifier.classify_images_with_label_sets(
        [
            {
                "flickr_photo_id": "1",
                "image_path": "/tmp/1.jpg",
                "image_hash": "sha256:image",
                "image_url_used": "https://live.staticflickr.com/1.jpg",
                "resolved_scientific_name": "Papilio demoleus",
                "text_evidence_present": True,
            }
        ],
        label_sets={
            "species": [
                "a photo of Papilio demoleus",
                "a photo of lime butterfly",
                "a photo of Papilio machaon",
            ],
            "triage": ["a photo of an adult butterfly"],
        },
        species_prompt_variants=[
            PromptVariant("a photo of Papilio demoleus", "Papilio demoleus", "scientific"),
            PromptVariant("a photo of lime butterfly", "Papilio demoleus", "common"),
            PromptVariant("a photo of Papilio machaon", "Papilio machaon", "scientific"),
        ],
    )

    record = records[0]
    assert record["species_top1_scientific_name"] == "Papilio demoleus"
    assert record["species_top1_score"] == 0.84
    assert record["species_top1_label"] == "a photo of lime butterfly"
    assert record["species_prompt_topk_json"][0]["prompt_scores"]["a photo of Papilio demoleus"] == 0.62
