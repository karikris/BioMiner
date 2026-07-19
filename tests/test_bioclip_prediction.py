from __future__ import annotations

import json
from collections.abc import Mapping
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
from biominer.bioclip.bioclip_worker import (
    OPENCLIP_PREPROCESSING_ATTESTATION_VERSION,
    preprocessing_attestation_fingerprint,
)
from biominer.bioclip.model_registry import BioClipRuntime, ModelConfig
from biominer.bioclip.prompt_templates import PromptVariant


REVISION = "191d741545e4c741cdef4b22c6eb69c945c1e592"
WEIGHTS_SHA256 = "sha256:" + "a" * 64
OPEN_CLIP_CONFIG_SHA256 = "sha256:" + "b" * 64
IMAGE_CONTENT_HASH_1 = "sha256:" + "d" * 64
IMAGE_CONTENT_HASH_2 = "sha256:" + "e" * 64
PREPROCESSING_CONFIG: dict[str, object] = {
    "fill_color": 0,
    "interpolation": "bicubic",
    "mean": [0.48145466, 0.4578275, 0.40821073],
    "mode": "RGB",
    "resize_mode": "longest",
    "size": [224, 224],
    "std": [0.26862954, 0.26130258, 0.27577711],
}
_MISSING = object()


def _prompt_variant(label: str, taxon_key: str, prompt_kind: str) -> PromptVariant:
    return PromptVariant(
        label=label,
        taxon_key=taxon_key,
        prompt_kind=prompt_kind,
        prompt_version="test-prompt-v1",
        template_id=f"test-{prompt_kind}-v1",
        evidence_kind="test_fixture",
    )


def _image_embedding_worker_metadata(
    *,
    weights_sha256: object = WEIGHTS_SHA256,
    preprocessing_config: dict[str, object] | None = None,
    open_clip_config_sha256: str | None = OPEN_CLIP_CONFIG_SHA256,
    open_clip_version: str = "3.3.0",
) -> dict[str, object]:
    config = dict(preprocessing_config or PREPROCESSING_CONFIG)
    metadata: dict[str, object] = {
        "device": "cuda",
        "gpu_name": "test-gpu",
        "image_resize_mode": "longest",
        "model_id": "imageomics/bioclip-2",
        "model_revision": REVISION,
        "open_clip_version": open_clip_version,
        "open_clip_config_sha256": open_clip_config_sha256,
        "preprocessing_version": OPENCLIP_PREPROCESSING_ATTESTATION_VERSION,
        "preprocessing_config": config,
        "preprocessing_fingerprint": preprocessing_attestation_fingerprint(
            open_clip_config_sha256=open_clip_config_sha256,
            open_clip_version=open_clip_version,
            preprocessing_config=config,
            preprocessing_version=OPENCLIP_PREPROCESSING_ATTESTATION_VERSION,
        ),
    }
    if weights_sha256 is not _MISSING:
        metadata["model_weights_sha256"] = weights_sha256
    return metadata


class _ProtocolStdin:
    def __init__(self) -> None:
        self.writes: list[str] = []
        self.closed = False

    def write(self, value: str) -> None:
        self.writes.append(value)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _ProtocolStdout:
    def __init__(self, lines: list[str]) -> None:
        self.lines = iter(lines)
        self.closed = False

    def readline(self) -> str:
        return next(self.lines, "")

    def close(self) -> None:
        self.closed = True


class _ProtocolProcess:
    def __init__(self, lines: list[str]) -> None:
        self.stdin = _ProtocolStdin()
        self.stdout = _ProtocolStdout(lines)
        self.stderr = _ProtocolStdout([])
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = 0

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout=None) -> int:  # noqa: ANN001 - mirrors subprocess.Popen.
        self.returncode = self.returncode or 0
        return self.returncode


def _json_lines(*payloads: Mapping[str, object]) -> list[str]:
    return [json.dumps(payload) + "\n" for payload in payloads]


def _runtime() -> BioClipRuntime:
    model = ModelConfig(
        model_id="bioclip2_5_huge",
        display_name="BioCLIP 2.5 Huge",
        role="preferred",
        status="use_if_available",
        task="biology image-text classification and embedding",
        model_name="imageomics/bioclip-2",
        checkpoint=REVISION,
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


def _generic_runtime() -> BioClipRuntime:
    model = ModelConfig(
        model_id="generic-openclip",
        display_name="Generic OpenCLIP",
        role="test",
        status="use_if_available",
        task="image-text embedding",
        model_name="ViT-H-14",
        checkpoint="generic-checkpoint",
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


def test_default_bioclip_labels_include_required_generic_triage_prompts() -> None:
    assert "a photo of an adult butterfly" in DEFAULT_BIOCLIP_LABELS
    assert "a photo of a butterfly" in DEFAULT_BIOCLIP_LABELS
    assert "a photo of a moth" in DEFAULT_BIOCLIP_LABELS
    assert "a photo of a pinned museum specimen" in DEFAULT_BIOCLIP_LABELS
    assert "a photo of artwork or illustration" in DEFAULT_BIOCLIP_LABELS


def test_classify_species_agreement_detects_exact_species_agreement() -> None:
    status = classify_species_agreement(
        resolved_scientific_name="Papilio demoleus",
        topk_labels=["a photo of Papilio demoleus", "a photo of lime butterfly"],
        text_evidence_present=True,
    )

    assert status == "exact_species_agreement"


def test_classify_species_agreement_routes_conflicting_text_and_vision_to_review() -> (
    None
):
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
    assert record["model_checkpoint"] == REVISION
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

    classifier = BioClipClassifier(
        runtime=unavailable, scorer=lambda image_path, labels: {}
    )

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
    scores = scorer(
        Path("/tmp/image.jpg"), ["a photo of Papilio demoleus", "a photo of a moth"]
    )

    assert scores["a photo of Papilio demoleus"] == 0.97
    assert calls[0]["cmd"][0] == "/home/toffe/bioclip25/.venv/bin/python"
    assert "image_paths" in calls[0]["input"]
    assert '"device": "auto"' in calls[0]["input"]
    assert "imageomics/bioclip-2" in calls[0]["input"]
    assert "data/cache/huggingface" in calls[0]["input"]
    assert "image_resize_mode" not in calls[0]["input"]


def test_bioclip_scorers_reject_invalid_image_resize_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported BioCLIP image resize mode"):
        ExternalBioClipScorer(runtime=_runtime(), image_resize_mode="center_crop")
    with pytest.raises(ValueError, match="Unsupported BioCLIP image resize mode"):
        PersistentBioClipScorer(runtime=_runtime(), image_resize_mode="center_crop")


@pytest.mark.parametrize(
    ("worker_response", "error_match"),
    [
        ('{"scores":{"butterfly":1.0}}', "did not report"),
        (
            '{"scores":{"butterfly":1.0},"image_resize_mode":"shortest"}',
            "resize mode mismatch",
        ),
    ],
)
def test_external_bioclip_scorer_requires_requested_resize_mode_acknowledgement(
    worker_response: str,
    error_match: str,
) -> None:
    def fake_run(cmd, *, input, capture_output, check, text):  # noqa: ANN001 - mirrors subprocess.run signature.
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=worker_response,
            stderr="",
        )

    scorer = ExternalBioClipScorer(
        runtime=_runtime(), runner=fake_run, image_resize_mode="longest"
    )

    with pytest.raises(RuntimeError, match=error_match):
        scorer(Path("/tmp/image.jpg"), ["butterfly"])


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
    assert (
        '"label_sets": {"species": ["a photo of Papilio demoleus"], "triage": ["a photo of an adult butterfly"]}'
        in calls[0]["input"]
    )


def test_external_bioclip_scorer_raises_with_worker_stderr() -> None:
    def fake_run(cmd, *, input, capture_output, check, text):  # noqa: ANN001 - mirrors subprocess.run signature.
        return subprocess.CompletedProcess(
            args=cmd, returncode=2, stdout="", stderr="model missing"
        )

    scorer = ExternalBioClipScorer(runtime=_runtime(), runner=fake_run)

    try:
        scorer(Path("/tmp/image.jpg"), ["a photo of Papilio demoleus"])
    except RuntimeError as exc:
        assert "BioCLIP worker failed" in str(exc)
        assert "model missing" in str(exc)
    else:
        raise AssertionError("expected worker failure to raise")


def test_persistent_bioclip_scorer_reuses_worker_for_batches_and_uses_auto_device() -> (
    None
):
    writes: list[str] = []

    class FakeStdin:
        def write(self, value: str) -> None:
            writes.append(value)

        def flush(self) -> None:
            return None

    class FakeStdout:
        def __init__(self) -> None:
            metadata = _image_embedding_worker_metadata()
            first_progress = {
                "worker_request_count": 1,
                "model_load_count": 1,
                "model_cache_hit_count": 0,
                "model_refresh_count": 0,
                "model_cache_hit": False,
            }
            second_progress = {
                "worker_request_count": 2,
                "model_load_count": 1,
                "model_cache_hit_count": 1,
                "model_refresh_count": 0,
                "model_cache_hit": True,
            }
            self.lines = iter(
                [
                    json.dumps({"ready": True, **metadata, **first_progress})
                    + "\n",
                    json.dumps(
                        {
                            "scores_by_image": [
                                {"a photo of Papilio demoleus": 0.97},
                                {"a photo of Papilio demoleus": 0.98},
                            ],
                            **metadata,
                            **first_progress,
                        }
                    )
                    + "\n",
                    json.dumps(
                        {
                            "scores_by_image": [
                                {"a photo of Papilio demoleus": 0.96}
                            ],
                            **metadata,
                            **second_progress,
                        }
                    )
                    + "\n",
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

    scorer = PersistentBioClipScorer(
        runtime=_runtime(), popen=fake_popen, preprocess_workers=4
    )
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
    assert '"preprocess_workers": 4' in writes[0]
    assert "image_resize_mode" not in writes[0]
    assert '"device": "auto"' in writes[1]
    assert '"preprocess_workers": 4' in writes[1]
    assert '"shutdown": true' in writes[-1]
    assert processes[0].waited is True
    assert scorer.cache_metrics == {
        "bioclip_worker_process_starts": 1,
        "bioclip_worker_requests": 2,
        "bioclip_model_loads": 1,
        "bioclip_model_cache_hits": 1,
        "bioclip_model_refreshes": 0,
        "bioclip_model_cache_hit_rate": 0.5,
        "bioclip_last_request_cache_hit": True,
    }


def test_persistent_bioclip_scorer_reuses_worker_for_label_sets() -> None:
    writes: list[str] = []

    class FakeStdin:
        def write(self, value: str) -> None:
            writes.append(value)

        def flush(self) -> None:
            return None

    class FakeStdout:
        def __init__(self) -> None:
            metadata = _image_embedding_worker_metadata()
            self.lines = iter(
                [
                    json.dumps({"ready": True, **metadata}) + "\n",
                    json.dumps(
                        {
                            "scores_by_image_by_label_set": {
                                "species": [
                                    {"a photo of Papilio demoleus": 0.97}
                                ],
                                "triage": [
                                    {"a photo of an adult butterfly": 0.98}
                                ],
                            },
                            **metadata,
                        }
                    )
                    + "\n",
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
    assert (
        '"label_sets": {"species": ["a photo of Papilio demoleus"], "triage": ["a photo of an adult butterfly"]}'
        in writes[0]
    )


def test_persistent_bioclip_scorer_can_embed_text_labels_for_cache() -> None:
    writes: list[str] = []

    class FakeStdin:
        def write(self, value: str) -> None:
            writes.append(value)

        def flush(self) -> None:
            return None

    class FakeStdout:
        def __init__(self) -> None:
            metadata = _image_embedding_worker_metadata()
            self.lines = iter(
                [
                    json.dumps({"ready": True, **metadata}) + "\n",
                    json.dumps(
                        {
                            "text_embeddings": [[1.0, 0.0], [0.0, 1.0]],
                            "embedding_dim": 2,
                            **metadata,
                        }
                    )
                    + "\n",
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

    def fake_popen(cmd, **kwargs):  # noqa: ANN001, ANN202 - mirrors subprocess.Popen.
        return FakeProcess(cmd)

    scorer = PersistentBioClipScorer(runtime=_runtime(), popen=fake_popen)
    try:
        embeddings = scorer.embed_text_labels(["Nymphalidae", "Danaus"])
    finally:
        scorer.close()

    assert embeddings == [[1.0, 0.0], [0.0, 1.0]]
    assert '"text_labels": ["Nymphalidae", "Danaus"]' in writes[0]
    assert '"device": "auto"' in writes[0]


def test_persistent_bioclip_scorer_can_embed_image_paths_for_cache() -> None:
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
                    json.dumps(
                        {
                            "ready": True,
                            **_image_embedding_worker_metadata(),
                        }
                    )
                    + "\n",
                    json.dumps(
                        {
                            "image_embeddings": [[0.1, 0.9], [0.8, 0.2]],
                            "embedding_dim": 2,
                            "image_content_hashes": [
                                IMAGE_CONTENT_HASH_1,
                                IMAGE_CONTENT_HASH_2,
                            ],
                            **_image_embedding_worker_metadata(),
                        }
                    )
                    + "\n",
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

    def fake_popen(cmd, **kwargs):  # noqa: ANN001, ANN202 - mirrors subprocess.Popen.
        return FakeProcess(cmd)

    scorer = PersistentBioClipScorer(
        runtime=_runtime(), popen=fake_popen, image_resize_mode="longest"
    )
    try:
        embeddings = scorer.embed_image_paths(
            [Path("/tmp/crop-1.ppm"), Path("/tmp/crop-2.ppm")]
        )
    finally:
        scorer.close()

    assert embeddings == [[0.1, 0.9], [0.8, 0.2]]
    assert (
        '"image_embedding_paths": ["/tmp/crop-1.ppm", "/tmp/crop-2.ppm"]' in writes[0]
    )
    assert '"device": "auto"' in writes[0]
    assert '"image_resize_mode": "longest"' in writes[0]
    assert scorer.effective_image_resize_mode == "longest"
    assert scorer.model_id == "imageomics/bioclip-2"
    assert scorer.model_revision == REVISION
    assert scorer.model_weights_sha256 == WEIGHTS_SHA256
    assert scorer.last_image_content_hashes == [
        IMAGE_CONTENT_HASH_1,
        IMAGE_CONTENT_HASH_2,
    ]
    assert scorer.open_clip_version == "3.3.0"
    assert scorer.open_clip_config_sha256 == OPEN_CLIP_CONFIG_SHA256
    assert scorer.preprocessing_config == PREPROCESSING_CONFIG
    assert scorer.preprocessing_version == OPENCLIP_PREPROCESSING_ATTESTATION_VERSION
    assert scorer.preprocessing_fingerprint == preprocessing_attestation_fingerprint(
        open_clip_config_sha256=OPEN_CLIP_CONFIG_SHA256,
        open_clip_version="3.3.0",
        preprocessing_config=PREPROCESSING_CONFIG,
        preprocessing_version=OPENCLIP_PREPROCESSING_ATTESTATION_VERSION,
    )


def test_persistent_bioclip_scorer_probes_frozen_model_without_scoring() -> None:
    metadata = _image_embedding_worker_metadata()
    process = _ProtocolProcess(
        [
            json.dumps({"ready": True, **metadata}) + "\n",
            json.dumps({"probed": True, **metadata}) + "\n",
        ]
    )
    scorer = PersistentBioClipScorer(
        runtime=_runtime(),
        popen=lambda *_args, **_kwargs: process,
        image_resize_mode="longest",
    )
    try:
        scorer.ensure_model_attestation()
    finally:
        scorer.close()

    request = json.loads(process.stdin.writes[0])
    assert request["probe"] is True
    assert "image_paths" not in request
    assert scorer.model_weights_sha256 == WEIGHTS_SHA256
    assert scorer.preprocessing_fingerprint == metadata["preprocessing_fingerprint"]


def test_persistent_bioclip_scorer_rechecks_pinned_identity_after_restart() -> None:
    first_metadata = _image_embedding_worker_metadata()
    changed_metadata = _image_embedding_worker_metadata(
        weights_sha256="sha256:" + "c" * 64
    )
    processes = [
        _ProtocolProcess(
            _json_lines(
                {"ready": True, **first_metadata},
                {"probed": True, **first_metadata},
                {"error": "fixture worker failure"},
            )
        ),
        _ProtocolProcess(
            _json_lines(
                {"ready": True, **changed_metadata},
                {
                    **changed_metadata,
                    "scores_by_image": [{"butterfly": 1.0}],
                },
            )
        ),
    ]
    started: list[_ProtocolProcess] = []

    def fake_popen(_cmd, **_kwargs):  # noqa: ANN202 - fake process factory.
        process = processes[len(started)]
        started.append(process)
        return process

    scorer = PersistentBioClipScorer(
        runtime=_runtime(),
        popen=fake_popen,
        image_resize_mode="longest",
    )
    scorer.ensure_model_attestation()
    scorer.pin_reference_model_identity(
        model_weights_sha256=WEIGHTS_SHA256,
        open_clip_version="3.3.0",
        open_clip_config_sha256=OPEN_CLIP_CONFIG_SHA256,
        preprocessing_fingerprint=str(first_metadata["preprocessing_fingerprint"]),
        image_resize_mode="longest",
    )

    with pytest.raises(RuntimeError, match="fixture worker failure"):
        scorer.score_batch([Path("/tmp/first.jpg")], ["butterfly"])
    with pytest.raises(RuntimeError, match="pinned reference model identity"):
        scorer.score_batch([Path("/tmp/second.jpg")], ["butterfly"])

    assert len(started) == 2
    assert started[0].terminate_calls == 1
    assert started[1].terminate_calls == 1


@pytest.mark.parametrize(
    "weights_sha256",
    [_MISSING, None],
    ids=["absent", "null"],
)
def test_persistent_image_embeddings_allow_unattested_generic_weights(
    weights_sha256: object,
) -> None:
    metadata = _image_embedding_worker_metadata(weights_sha256=weights_sha256)
    metadata.update(
        model_id="ViT-H-14",
        model_revision="generic-checkpoint",
    )
    process = _ProtocolProcess(
        _json_lines(
            {"ready": True, **metadata},
            {
                **metadata,
                "image_embeddings": [[1.0, 0.0]],
                "embedding_dim": 2,
                "image_content_hashes": [IMAGE_CONTENT_HASH_1],
            },
        )
    )
    scorer = PersistentBioClipScorer(
        runtime=_generic_runtime(),
        popen=lambda _cmd, **_kwargs: process,
        image_resize_mode="longest",
    )

    try:
        assert scorer.embed_image_paths([Path("/tmp/1.jpg")]) == [[1.0, 0.0]]
    finally:
        scorer.close()

    assert scorer.model_weights_sha256 is None


def test_persistent_image_embeddings_replace_hashes_for_each_response() -> None:
    metadata = _image_embedding_worker_metadata()
    process = _ProtocolProcess(
        _json_lines(
            {"ready": True, **metadata},
            {
                **metadata,
                "image_embeddings": [[1.0, 0.0]],
                "embedding_dim": 2,
                "image_content_hashes": [IMAGE_CONTENT_HASH_1],
            },
            {
                **metadata,
                "image_embeddings": [[0.0, 1.0]],
                "embedding_dim": 2,
                "image_content_hashes": [IMAGE_CONTENT_HASH_2],
            },
        )
    )
    scorer = PersistentBioClipScorer(
        runtime=_runtime(),
        popen=lambda _cmd, **_kwargs: process,
        image_resize_mode="longest",
    )

    assert scorer.embed_image_paths([Path("/tmp/1.jpg")]) == [[1.0, 0.0]]
    assert scorer.last_image_content_hashes == [IMAGE_CONTENT_HASH_1]
    assert scorer.embed_image_paths([Path("/tmp/2.jpg")]) == [[0.0, 1.0]]
    assert scorer.last_image_content_hashes == [IMAGE_CONTENT_HASH_2]
    scorer.close()


def test_persistent_image_embeddings_reject_incomplete_attestation() -> None:
    metadata = _image_embedding_worker_metadata()
    metadata.pop("preprocessing_fingerprint")
    process = _ProtocolProcess(_json_lines({"ready": True, **metadata}))
    scorer = PersistentBioClipScorer(
        runtime=_runtime(),
        popen=lambda _cmd, **_kwargs: process,
        image_resize_mode="longest",
    )

    with pytest.raises(RuntimeError, match="complete preprocessing attestation"):
        scorer.embed_image_paths([Path("/tmp/1.jpg")])

    assert process.terminate_calls == 1


def test_persistent_image_embeddings_reject_unpinned_openclip_version() -> None:
    metadata = _image_embedding_worker_metadata(open_clip_version="3.4.0")
    process = _ProtocolProcess(_json_lines({"ready": True, **metadata}))
    scorer = PersistentBioClipScorer(
        runtime=_runtime(),
        popen=lambda _cmd, **_kwargs: process,
        image_resize_mode="longest",
    )

    with pytest.raises(RuntimeError, match="OpenCLIP version mismatch"):
        scorer.embed_image_paths([Path("/tmp/1.jpg")])

    assert process.terminate_calls == 1


def test_persistent_image_embeddings_reject_wrong_attestation_fingerprint() -> None:
    metadata = {
        **_image_embedding_worker_metadata(),
        "preprocessing_fingerprint": "sha256:" + "c" * 64,
    }
    process = _ProtocolProcess(_json_lines({"ready": True, **metadata}))
    scorer = PersistentBioClipScorer(
        runtime=_runtime(),
        popen=lambda _cmd, **_kwargs: process,
        image_resize_mode="longest",
    )

    with pytest.raises(RuntimeError, match="preprocessing fingerprint mismatch"):
        scorer.embed_image_paths([Path("/tmp/1.jpg")])


def test_persistent_image_embeddings_reject_preprocessing_attestation_drift() -> None:
    ready_metadata = _image_embedding_worker_metadata()
    changed_config = {**PREPROCESSING_CONFIG, "interpolation": "bilinear"}
    result_metadata = _image_embedding_worker_metadata(
        preprocessing_config=changed_config
    )
    process = _ProtocolProcess(
        _json_lines(
            {"ready": True, **ready_metadata},
            {
                **result_metadata,
                "image_embeddings": [[1.0, 0.0]],
                "embedding_dim": 2,
                "image_content_hashes": [IMAGE_CONTENT_HASH_1],
            },
        )
    )
    scorer = PersistentBioClipScorer(
        runtime=_runtime(),
        popen=lambda _cmd, **_kwargs: process,
        image_resize_mode="longest",
    )

    with pytest.raises(RuntimeError, match="preprocessing config changed"):
        scorer.embed_image_paths([Path("/tmp/1.jpg")])


@pytest.mark.parametrize(
    ("ready_override", "result_override", "message"),
    [
        ({"model_id": "other/model"}, {}, "model ID mismatch"),
        ({"model_revision": "0" * 40}, {}, "model revision mismatch"),
        ({}, {"model_revision": "0" * 40}, "model revision mismatch"),
        ({}, {"image_embeddings": [[1.0, 0.0]]}, "returned 1 rows for 2 images"),
        (
            {},
            {"image_embeddings": [[1.0, 0.0], [1.0, 0.0, 0.0]]},
            "embedding dimension mismatch",
        ),
        (
            {},
            {"image_embeddings": [[float("nan"), 0.0], [1.0, 0.0]]},
            "finite values",
        ),
        ({}, {"embedding_dim": 3}, "reported embedding dimension"),
        ({}, {"image_content_hashes": None}, "content hashes must be a list"),
        (
            {},
            {"image_content_hashes": [IMAGE_CONTENT_HASH_1]},
            "1 image content hashes for 2 images",
        ),
        (
            {},
            {"image_content_hashes": [IMAGE_CONTENT_HASH_1, "sha256:INVALID"]},
            "valid SHA-256 values",
        ),
    ],
)
def test_persistent_image_embedding_boundary_fails_closed(
    ready_override: dict[str, object],
    result_override: dict[str, object],
    message: str,
) -> None:
    metadata = _image_embedding_worker_metadata()
    ready = {"ready": True, **metadata, **ready_override}
    result = {
        **metadata,
        "image_embeddings": [[1.0, 0.0], [0.0, 1.0]],
        "embedding_dim": 2,
        "image_content_hashes": [IMAGE_CONTENT_HASH_1, IMAGE_CONTENT_HASH_2],
        **result_override,
    }

    class FakeStdin:
        def write(self, _value: str) -> None:
            return None

        def flush(self) -> None:
            return None

    class FakeStdout:
        def __init__(self) -> None:
            self.lines = iter([json.dumps(ready) + "\n", json.dumps(result) + "\n"])

        def readline(self) -> str:
            return next(self.lines)

    class FakeProcess:
        def __init__(self) -> None:
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

    scorer = PersistentBioClipScorer(
        runtime=_runtime(),
        popen=lambda _cmd, **_kwargs: FakeProcess(),
        image_resize_mode="longest",
    )
    try:
        with pytest.raises(RuntimeError, match=message):
            scorer.embed_image_paths([Path("/tmp/1.jpg"), Path("/tmp/2.jpg")])
    finally:
        scorer.close()


def test_persistent_image_embedding_protocol_error_discards_stale_worker() -> None:
    processes: list[object] = []

    class FakeStdin:
        def write(self, _value: str) -> None:
            return None

        def flush(self) -> None:
            return None

    class FakeStdout:
        def __init__(self, lines: list[dict[str, object]]) -> None:
            self.lines = iter(json.dumps(line) + "\n" for line in lines)

        def readline(self) -> str:
            return next(self.lines)

    class FakeProcess:
        def __init__(self, lines: list[dict[str, object]]) -> None:
            self.stdin = FakeStdin()
            self.stdout = FakeStdout(lines)
            self.stderr = None
            self.returncode = None
            self.terminate_calls = 0

        def poll(self):  # noqa: ANN202 - mirrors subprocess.Popen.
            return self.returncode

        def terminate(self) -> None:
            self.terminate_calls += 1
            self.returncode = 0

        def wait(self, timeout=None):  # noqa: ANN001, ANN202 - mirrors subprocess.Popen.
            self.returncode = 0
            return 0

    metadata = _image_embedding_worker_metadata()
    responses = [
        [
            {"ready": True, **metadata, "model_revision": "0" * 40},
            {
                **metadata,
                "image_embeddings": [[1.0, 0.0]],
                "embedding_dim": 2,
                "image_content_hashes": [IMAGE_CONTENT_HASH_1],
            },
        ],
        [
            {"ready": True, **metadata},
            {
                **metadata,
                "image_embeddings": [[0.0, 1.0]],
                "embedding_dim": 2,
                "image_content_hashes": [IMAGE_CONTENT_HASH_2],
            },
        ],
    ]

    def fake_popen(_cmd, **_kwargs):  # noqa: ANN202 - fake process factory.
        process = FakeProcess(responses[len(processes)])
        processes.append(process)
        return process

    scorer = PersistentBioClipScorer(
        runtime=_runtime(),
        popen=fake_popen,
        image_resize_mode="longest",
    )
    with pytest.raises(RuntimeError, match="model revision mismatch"):
        scorer.embed_image_paths([Path("/tmp/first.jpg")])

    assert scorer.embed_image_paths([Path("/tmp/second.jpg")]) == [[0.0, 1.0]]
    assert scorer.last_image_content_hashes == [IMAGE_CONTENT_HASH_2]
    assert len(processes) == 2
    assert processes[0].terminate_calls == 1
    scorer.close()


@pytest.mark.parametrize(
    "operation",
    ["scores", "label_sets", "text_embeddings", "image_embeddings"],
)
def test_persistent_protocol_failures_discard_worker_for_every_operation(
    operation: str,
) -> None:
    process = _ProtocolProcess(["not-json\n"])
    scorer = PersistentBioClipScorer(
        runtime=_runtime(),
        popen=lambda _cmd, **_kwargs: process,
        image_resize_mode="longest" if operation == "image_embeddings" else None,
    )

    with pytest.raises(json.JSONDecodeError):
        if operation == "scores":
            scorer.score_batch([Path("/tmp/1.jpg")], ["butterfly"])
        elif operation == "label_sets":
            scorer.score_label_sets_batch(
                [Path("/tmp/1.jpg")],
                {"triage": ["butterfly"]},
            )
        elif operation == "text_embeddings":
            scorer.embed_text_labels(["butterfly"])
        else:
            scorer.embed_image_paths([Path("/tmp/1.jpg")])

    assert process.terminate_calls == 1
    assert scorer._process is None  # noqa: SLF001 - protocol poison contract.
    assert process.stdin.closed is True
    assert process.stdout.closed is True
    assert process.stderr.closed is True


def test_persistent_protocol_cleanup_kills_stubborn_worker_and_closes_pipes() -> None:
    class StubbornProcess(_ProtocolProcess):
        def __init__(self) -> None:
            super().__init__(["not-json\n"])
            self.wait_calls = 0

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1
            self.returncode = -9

        def wait(self, timeout=None) -> int:  # noqa: ANN001 - mirrors subprocess.Popen.
            self.wait_calls += 1
            if self.kill_calls == 0:
                raise subprocess.TimeoutExpired("bioclip-worker", timeout)
            return -9

    process = StubbornProcess()
    scorer = PersistentBioClipScorer(
        runtime=_runtime(),
        popen=lambda _cmd, **_kwargs: process,
    )

    with pytest.raises(json.JSONDecodeError):
        scorer.score_batch([Path("/tmp/1.jpg")], ["butterfly"])

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_calls == 2
    assert process.stdin.closed is True
    assert process.stdout.closed is True
    assert process.stderr.closed is True


def test_bioclip_classifier_builds_species_and_triage_prediction_with_label_sets() -> (
    None
):
    class FakeScorer:
        def score_label_sets_batch(self, image_paths, label_sets):  # noqa: ANN001 - test fake.
            assert label_sets["triage"] == ["a photo of an adult butterfly"]
            return {
                "species": [
                    {
                        "a photo of Papilio demoleus": 0.91,
                        "a photo of Papilio machaon": 0.09,
                    }
                ],
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


def test_bioclip_classifier_ranks_species_prompt_variants_by_best_two_mean() -> None:
    from biominer.bioclip.prompt_templates import (
        SPECIES_PROMPT_AGGREGATION_DEFAULT,
    )

    assert SPECIES_PROMPT_AGGREGATION_DEFAULT == "mean_best_two"

    class FakeScorer:
        def score_label_sets_batch(self, image_paths, label_sets):  # noqa: ANN001 - test fake.
            return {
                "species": [
                    {
                        "a photo of Papilio demoleus": 0.95,
                        "a field photo of Papilio demoleus adult butterfly": 0.05,
                        "a photo of Papilio machaon": 0.58,
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
                "a field photo of Papilio demoleus adult butterfly",
                "a photo of Papilio machaon",
            ],
            "triage": ["a photo of an adult butterfly"],
        },
        species_prompt_variants=[
            _prompt_variant(
                "a photo of Papilio demoleus", "Papilio demoleus", "scientific"
            ),
            _prompt_variant(
                "a field photo of Papilio demoleus adult butterfly",
                "Papilio demoleus",
                "field_adult",
            ),
            _prompt_variant(
                "a photo of Papilio machaon", "Papilio machaon", "scientific"
            ),
        ],
    )

    record = records[0]
    assert record["species_top1_scientific_name"] == "Papilio machaon"
    assert record["species_top1_score"] == 0.58
    assert record["species_top1_label"] == "a photo of Papilio machaon"
    assert record["species_prompt_topk_json"][1]["taxon_key"] == "Papilio demoleus"
    assert record["species_prompt_topk_json"][1]["score"] == pytest.approx(0.50)
    assert (
        record["species_prompt_topk_json"][1]["best_label"]
        == "a photo of Papilio demoleus"
    )
