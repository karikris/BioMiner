from __future__ import annotations

import copy
import importlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from biominer.detection.detector_base import DecodedImage
from biominer.vision.full_frame_attention import decoded_image_content_hash


def _module():
    return importlib.import_module("biominer.detection.embeddinggemma2")


def _vector(index: int = 0, value: float = 1.0) -> list[float]:
    result = [0.0] * 768
    result[index] = value
    return result


def _image() -> DecodedImage:
    return DecodedImage(width=3, height=2, mode="RGB", data=bytes(range(18)))


class FakeModel:
    def __init__(self, image_vectors=None, text_vectors=None):
        self.image_vectors = image_vectors
        self.text_vectors = text_vectors
        self.calls = []
        self.frozen = False
        self.evaluation = False

    def requires_grad_(self, value):
        self.frozen = value is False
        return self

    def eval(self):
        self.evaluation = True
        return self

    def encode(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        if isinstance(inputs[0], str):
            return self.text_vectors if self.text_vectors is not None else [_vector(i) for i in range(len(inputs))]
        return self.image_vectors if self.image_vectors is not None else [_vector() for _ in inputs]


def _router(model=None, **config):
    module = _module()
    fake = model or FakeModel()
    return module.EmbeddingGemma2Router(
        config=module.EmbeddingGemma2Config(**config), model_factory=lambda **kwargs: fake
    ), fake


def test_backend_module_does_not_import_ml_dependencies(monkeypatch) -> None:
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "sentence_transformers"}:
            raise AssertionError("heavy model dependency imported during module import")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    module = importlib.reload(_module())

    assert module.DEFAULT_EMBEDDINGGEMMA2_MODEL == "google/embeddinggemma-2"


def test_default_config_binds_immutable_model_and_fixed_dimension() -> None:
    config = _module().EmbeddingGemma2Config()

    assert config.revision == "f9972ac22e696a105c482202646979e33e4b5a1e"
    assert config.embedding_dimension == 768
    assert config.dtype == "float32"
    assert config.as_dict()["revision"] == config.revision
    assert config.fingerprint.startswith("sha256:")


@pytest.mark.parametrize("kwargs", [
    {"revision": "main"}, {"revision": ""}, {"revision": "f" * 39},
    {"dtype": "float16"}, {"device": ""}, {"model_id": ""},
    {"min_similarity": float("nan")}, {"min_similarity": 1.1},
    {"min_margin": -0.01}, {"min_margin": float("inf")}, {"min_margin": True},
])
def test_config_rejects_unbound_or_unsafe_settings(kwargs) -> None:
    with pytest.raises(ValueError):
        _module().EmbeddingGemma2Config(**kwargs)


def test_config_rejects_custom_prompt_subsets() -> None:
    with pytest.raises(TypeError):
        _module().EmbeddingGemma2Config(prompts=("butterfly",))


def test_text_vectors_are_cached_and_images_keep_complete_rgb_canvas() -> None:
    router, model = _router()
    image = _image()

    first = router.route_batch([image, image])
    second = router.route_batch([image])

    assert len(first) == 2 and len(second) == 1
    assert len(model.calls) == 3
    assert model.calls[0][1]["prompt_name"] == "Classification"
    assert model.calls[0][1]["normalize_embeddings"] is True
    assert len(model.calls[1][0]) == 2
    assert len(model.calls[2][0]) == 1
    actual = model.calls[1][0][0]["image"]
    assert actual.size == (3, 2)
    assert actual.mode == "RGB"
    assert actual.tobytes() == image.data
    assert model.calls[1][1]["prompt"] == ""
    assert model.frozen and model.evaluation


def test_empty_batch_does_not_load_model() -> None:
    def forbidden(**kwargs):
        raise AssertionError("empty batch loaded weights")

    router = _module().EmbeddingGemma2Router(model_factory=forbidden)

    assert router.route_batch([]) == []


def test_image_evidence_never_invents_object_measurements() -> None:
    router, _ = _router()

    row = router.route_batch([_image()])[0]

    assert row["proposed_domain"] == "adult_butterfly_field"
    assert row["bioclip_route"] == "adult_field"
    assert row["routing_action"] == "review"
    assert row["routing_reason"] == "provisional_embedding_route"
    assert row["evidence_kind"] == "full_frame_embedding"
    assert row["localization_status"] == "unavailable"
    for field in ("subject_present", "subject_area_ratio", "detection_count", "bbox_xyxy", "mask_polygon_xyn", "detector_score"):
        assert row[field] is None
    assert row["image_content_hash"] == decoded_image_content_hash(_image())
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("domain,bioclip", [
    ("adult_butterfly_field", "adult_field"), ("caterpillar_field", "larval"),
    ("pinned_specimen", "pinned_specimen"), ("pupa_or_chrysalis", None),
    ("possible_moth_or_other_insect", None), ("artwork_logo_tattoo_or_other_artifact", None),
    ("no_relevant_organism", None),
])
def test_domain_prompts_remain_separate_review_evidence(domain, bioclip) -> None:
    module = _module()
    prompt_index = next(i for i, (_, item_domain) in enumerate(module.DEFAULT_EMBEDDINGGEMMA2_PROMPTS) if item_domain == domain)
    router, _ = _router(FakeModel(image_vectors=[_vector(prompt_index)]))

    row = router.route_batch([_image()])[0]

    assert row["proposed_domain"] == domain
    assert row["bioclip_route"] == bioclip
    assert row["routing_action"] == "review"


def test_margin_compares_runner_up_from_different_domain() -> None:
    module = _module()
    prompts = module.DEFAULT_EMBEDDINGGEMMA2_PROMPTS
    texts = []
    for _, domain in prompts:
        vector = _vector(1)
        if domain == "adult_butterfly_field":
            vector = _vector()
        texts.append(vector)
    router, _ = _router(FakeModel(text_vectors=texts), min_margin=0.5)

    row = router.route_batch([_image()])[0]

    assert row["proposed_domain"] == "adult_butterfly_field"
    assert row["similarity_margin"] == pytest.approx(1.0)


@pytest.mark.parametrize("similarity,margin,reason", [(1.0, 0.05, "ambiguous_margin"), (0.99, 0.0, "ambiguous_similarity")])
def test_ambiguous_evidence_cannot_propose_a_bioclip_route(similarity, margin, reason) -> None:
    model = FakeModel(image_vectors=[[_value for _value in ([1.0] * 768)]])
    if reason == "ambiguous_margin":
        model.image_vectors = [[1.0, 0.0, 1.0] + [0.0] * 765]
        similarity = 0.25
    router, _ = _router(model, min_similarity=similarity, min_margin=margin)

    row = router.route_batch([_image()])[0]

    assert row["proposed_domain"] == "ambiguous_visual_domain"
    assert row["bioclip_route"] is None
    assert row["routing_reason"] == reason
    assert row["routing_action"] == "review"


def test_signed_cosine_is_preserved_without_probability_conversion() -> None:
    router, _ = _router(FakeModel(image_vectors=[_vector(value=-2.0)]), min_similarity=-1.0, min_margin=0.0)

    row = router.route_batch([_image()])[0]
    similarities = json.loads(row["prompt_similarities_json"])

    assert similarities[0]["score"] == pytest.approx(-1.0)
    assert row["embedding"][0] == pytest.approx(-1.0)
    assert math.fsum(value * value for value in row["embedding"]) == pytest.approx(1.0)


@pytest.mark.parametrize("vectors", [[], [[0.0] * 768], [[1.0] * 767], [[float("nan")] * 768], [[float("inf")] * 768], [[True] * 768], [_vector(), _vector()]])
def test_invalid_image_embedding_batches_fail_closed(vectors) -> None:
    router, _ = _router(FakeModel(image_vectors=vectors))

    with pytest.raises(RuntimeError, match="embedding|cardinality|dimension|finite|zero"):
        router.route_batch([_image()])


def test_invalid_text_embedding_batch_fails_before_image_encoding() -> None:
    router, model = _router(FakeModel(text_vectors=[_vector()]))

    with pytest.raises(RuntimeError, match="cardinality"):
        router.route_batch([_image()])

    assert len(model.calls) == 1


def test_fingerprint_binds_policy_and_runtime_versions(monkeypatch) -> None:
    module = _module()
    router, _ = _router()
    changed, _ = _router(min_similarity=0.4)
    original = router.fingerprint

    monkeypatch.setattr(module, "runtime_identity", lambda: {"python": "different", "packages": {"torch": "different"}})
    runtime_changed, _ = _router()

    assert original != changed.fingerprint
    assert original != runtime_changed.fingerprint
    assert router.prompt_set_fingerprint == changed.prompt_set_fingerprint


def test_default_loader_pins_model_revision_disables_audio_and_avoids_float16(monkeypatch) -> None:
    module = _module()
    model = FakeModel()
    captured = []
    torch_dtype = object()

    def factory(*args, **kwargs):
        captured.append((args, kwargs))
        return model

    real_version = module.importlib.metadata.version
    versions = {"sentence-transformers": "6.1.0", "transformers": "5.18.0"}
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: versions[name] if name in versions else real_version(name))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32=torch_dtype))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=factory))
    router = module.EmbeddingGemma2Router()

    router.route_batch([_image()])

    args, kwargs = captured[0]
    assert args == (module.DEFAULT_EMBEDDINGGEMMA2_MODEL,)
    assert kwargs["revision"] == module.DEFAULT_EMBEDDINGGEMMA2_REVISION
    assert kwargs["trust_remote_code"] is False
    assert kwargs["config_kwargs"] == {"audio_config": None}
    assert kwargs["model_kwargs"] == {"torch_dtype": torch_dtype}
    assert kwargs["device"] is None


def test_default_loader_rejects_unsupported_sentence_transformers(monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "6.0.0")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32=object()))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda **kwargs: pytest.fail("unsupported runtime loaded model")))
    router = module.EmbeddingGemma2Router()

    with pytest.raises(RuntimeError, match=">=6.1.0"):
        router.load()


def test_default_loader_rejects_unsupported_transformers(monkeypatch) -> None:
    module = _module()
    versions = {"sentence-transformers": "6.1.0", "transformers": "5.17.0"}
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: versions.get(name, "1.0.0"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32=object()))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda *args, **kwargs: FakeModel()))
    router = module.EmbeddingGemma2Router()

    with pytest.raises(RuntimeError, match="transformers >=5.18"):
        router.load()


def test_closed_router_does_not_reload_weights() -> None:
    router, _ = _router()
    router.close()

    with pytest.raises(RuntimeError, match="closed"):
        router.route_batch([_image()])


@pytest.mark.parametrize("sidecar", [False, True])
def test_router_configuration_cannot_change_under_an_existing_fingerprint(sidecar) -> None:
    module = _module()
    router, _ = _router()
    if sidecar:
        router = module.EmbeddingGemma2SidecarRouter(runtime_python=sys.executable)

    with pytest.raises(AttributeError):
        router.config = module.EmbeddingGemma2Config(min_similarity=0.75)


class FakeProcess:
    def __init__(self, responses):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO("".join(json.dumps(item) + "\n" for item in responses))
        self.stderr = io.StringIO()
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def _sidecar(responses=None, alter_metadata=None, runtime_python=sys.executable):
    module = _module()
    router, _ = _router()
    metadata = router.metadata()
    if alter_metadata:
        metadata.update(alter_metadata)
    messages = [{"protocol": module.EMBEDDINGGEMMA2_PROTOCOL_VERSION, "request_id": 1, "metadata": metadata}]
    messages += responses or []
    process = FakeProcess(messages)
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return process

    sidecar = module.EmbeddingGemma2SidecarRouter(runtime_python=runtime_python, popen=popen)
    return sidecar, process, calls, router


def test_sidecar_uses_prefetch_runtime_caches_and_mps_fallback(monkeypatch, tmp_path) -> None:
    module = _module()
    for name in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME", "PYTORCH_ENABLE_MPS_FALLBACK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTHONPATH", "existing/source")
    runtime_root = tmp_path / "EmbeddingGemma2"
    sidecar, _, calls, _ = _sidecar(runtime_python=runtime_root / "venv/bin/python")

    with sidecar:
        sidecar.metadata()
    env = calls[0][1]["env"]

    assert env["HF_HOME"] == str(runtime_root / "cache/huggingface")
    assert env["HUGGINGFACE_HUB_CACHE"] == str(runtime_root / "cache/huggingface/hub")
    assert env["TORCH_HOME"] == str(runtime_root / "cache/torch")
    assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    assert env["PYTHONPATH"] == str(Path(module.__file__).resolve().parents[2]) + os.pathsep + "existing/source"


@pytest.mark.parametrize("explicit_hub", [False, True])
def test_sidecar_preserves_explicit_huggingface_cache_configuration(monkeypatch, tmp_path, explicit_hub) -> None:
    hf_home = tmp_path / "user_huggingface"
    hub_cache = tmp_path / "user_hub" if explicit_hub else hf_home / "hub"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    if explicit_hub:
        monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub_cache))
    else:
        monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    sidecar, _, calls, _ = _sidecar(runtime_python=tmp_path / "EmbeddingGemma2/venv/bin/python")

    with sidecar:
        sidecar.metadata()
    env = calls[0][1]["env"]

    assert env["HF_HOME"] == str(hf_home)
    assert env["HUGGINGFACE_HUB_CACHE"] == str(hub_cache)


def test_sidecar_persists_worker_and_validates_fullframe_batch() -> None:
    module = _module()
    native, _ = _router()
    row = native.route_batch([_image()])[0]
    responses = [
        {"protocol": module.EMBEDDINGGEMMA2_PROTOCOL_VERSION, "request_id": i, "metadata": native.metadata(), "results": [row]}
        for i in (2, 3)
    ]
    sidecar, process, calls, router = _sidecar(responses)

    assert sidecar.route_batch([_image()]) == [row]
    assert sidecar.route_batch([_image()]) == [row]
    assert sidecar.fingerprint == router.fingerprint
    assert len(calls) == 1
    assert "biominer.detection.embeddinggemma2_worker" in calls[0][0]
    payloads = [json.loads(line) for line in process.stdin.getvalue().splitlines()]
    assert payloads[0]["operation"] == "hello"
    assert payloads[1]["images"][0]["width"] == 3
    assert payloads[1]["images"][0]["height"] == 2
    sidecar.close()
    assert process.poll() == 0


@pytest.mark.parametrize("field,value", [("revision", "a" * 40), ("prompt_set_fingerprint", "sha256:" + "0" * 64), ("fingerprint", "sha256:" + "0" * 64), ("model_id", "other/model")])
def test_sidecar_rejects_changed_handshake_identity(field, value) -> None:
    sidecar, process, _, _ = _sidecar(alter_metadata={field: value})

    with pytest.raises(RuntimeError, match="metadata|identity|fingerprint|revision|model"):
        sidecar.route_batch([_image()])

    assert process.poll() is not None


def test_sidecar_rejects_response_cardinality() -> None:
    module = _module()
    native, _ = _router()
    sidecar, _, _, _ = _sidecar([
        {"protocol": module.EMBEDDINGGEMMA2_PROTOCOL_VERSION, "request_id": 2, "metadata": native.metadata(), "results": []}
    ])

    with pytest.raises(RuntimeError, match="cardinality"):
        sidecar.route_batch([_image()])


def test_sidecar_rejects_fabricated_localization() -> None:
    module = _module()
    native, _ = _router()
    row = copy.deepcopy(native.route_batch([_image()])[0])
    row["bbox_xyxy"] = [0.0, 0.0, 3.0, 2.0]
    sidecar, _, _, _ = _sidecar([
        {"protocol": module.EMBEDDINGGEMMA2_PROTOCOL_VERSION, "request_id": 2, "metadata": native.metadata(), "results": [row]}
    ])

    with pytest.raises(RuntimeError, match="localization|bbox"):
        sidecar.route_batch([_image()])


def test_sidecar_model_errors_are_visible() -> None:
    module = _module()
    sidecar, _, _, _ = _sidecar([
        {"protocol": module.EMBEDDINGGEMMA2_PROTOCOL_VERSION, "request_id": 2, "error": "model unavailable"}
    ])

    with pytest.raises(RuntimeError, match="model unavailable"):
        sidecar.route_batch([_image()])


@pytest.mark.parametrize("changes", [
    {"cosine_similarity": -0.25}, {"similarity_margin": 0.01},
    {"proposed_domain": "pupa_or_chrysalis", "bioclip_route": None},
    {"routing_reason": "ambiguous_margin"},
])
def test_sidecar_rejects_inconsistent_prompt_score_evidence(changes) -> None:
    module = _module()
    native, _ = _router()
    row = native.route_batch([_image()])[0]
    row.update(changes)
    sidecar, _, _, _ = _sidecar([
        {"protocol": module.EMBEDDINGGEMMA2_PROTOCOL_VERSION, "request_id": 2, "metadata": native.metadata(), "results": [row]}
    ])

    with pytest.raises(RuntimeError, match="score|evidence|domain|reason"):
        sidecar.route_batch([_image()])


def test_sidecar_protocol_mismatch_terminates_owned_worker() -> None:
    sidecar, process, _, _ = _sidecar([{"protocol": "unexpected", "request_id": 2}])

    with pytest.raises(RuntimeError, match="protocol"):
        sidecar.route_batch([_image()])

    assert process.terminated


@pytest.mark.parametrize("request_id", [True, 1.0])
def test_sidecar_rejects_noninteger_response_request_id(request_id) -> None:
    module = _module()
    native, _ = _router()
    process = FakeProcess([{"protocol": module.EMBEDDINGGEMMA2_PROTOCOL_VERSION, "request_id": request_id, "metadata": native.metadata()}])
    sidecar = module.EmbeddingGemma2SidecarRouter(runtime_python=sys.executable, popen=lambda *args, **kwargs: process)

    with pytest.raises(RuntimeError, match="protocol|identity|request"):
        sidecar.metadata()

    assert process.terminated


def test_sidecar_rejects_duplicate_response_json_keys() -> None:
    module = _module()
    native, _ = _router()
    process = FakeProcess([])
    process.stdout = io.StringIO('{"protocol":' + json.dumps(module.EMBEDDINGGEMMA2_PROTOCOL_VERSION) + ',"request_id":999,"request_id":1,"metadata":' + json.dumps(native.metadata()) + '}\n')
    sidecar = module.EmbeddingGemma2SidecarRouter(runtime_python=sys.executable, popen=lambda *args, **kwargs: process)

    with pytest.raises(RuntimeError, match="duplicate"):
        sidecar.metadata()


def test_sidecar_rejects_duplicate_prompt_score_json_keys() -> None:
    module = _module()
    native, _ = _router()
    row = native.route_batch([_image()])[0]
    row["prompt_similarities_json"] = row["prompt_similarities_json"].replace('"score":1.0', '"score":0.0,"score":1.0', 1)
    sidecar, _, _, _ = _sidecar([
        {"protocol": module.EMBEDDINGGEMMA2_PROTOCOL_VERSION, "request_id": 2, "metadata": native.metadata(), "results": [row]}
    ])

    with pytest.raises(RuntimeError, match="duplicate"):
        sidecar.route_batch([_image()])


class BlockingOutput:
    def __init__(self):
        self.closed = Event()

    def __iter__(self):
        return self

    def __next__(self):
        self.closed.wait()
        raise StopIteration

    def close(self):
        self.closed.set()


class BlockingInput:
    def __init__(self):
        self.closed = Event()

    def write(self, data):
        self.closed.wait(timeout=1.0)
        raise BrokenPipeError("owned worker was closed")

    def close(self):
        self.closed.set()


class StubbornProcess(FakeProcess):
    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("worker", timeout)
        return self.returncode


def test_sidecar_response_timeout_escalates_to_kill_owned_worker() -> None:
    module = _module()
    process = StubbornProcess([])
    process.stdout = BlockingOutput()
    sidecar = module.EmbeddingGemma2SidecarRouter(runtime_python=sys.executable, popen=lambda *args, **kwargs: process, timeout_seconds=0.01)

    with pytest.raises(RuntimeError, match="timed out"):
        sidecar.metadata()

    assert process.terminated
    assert process.returncode == -9


def test_sidecar_write_timeout_terminates_owned_worker() -> None:
    module = _module()
    process = FakeProcess([])
    process.stdin = BlockingInput()
    process.stdout = BlockingOutput()
    sidecar = module.EmbeddingGemma2SidecarRouter(runtime_python=sys.executable, popen=lambda *args, **kwargs: process, timeout_seconds=0.01)

    with pytest.raises(RuntimeError, match="write timed out"):
        sidecar.metadata()

    assert process.terminated


def test_sidecar_eof_is_explicit_worker_failure() -> None:
    sidecar, process, _, _ = _sidecar()

    with pytest.raises(RuntimeError, match="closed stdout"):
        sidecar.route_batch([_image()])

    assert process.terminated


def test_real_worker_handshake_does_not_load_model_weights() -> None:
    module = _module()

    with module.EmbeddingGemma2SidecarRouter(runtime_python=sys.executable, timeout_seconds=10) as sidecar:
        metadata = sidecar.metadata()

    assert metadata["model_id"] == module.DEFAULT_EMBEDDINGGEMMA2_MODEL
    assert metadata["revision"] == module.DEFAULT_EMBEDDINGGEMMA2_REVISION
    assert metadata["runtime"]["python"]
    assert sidecar.worker_process_starts == 1


def test_worker_round_trip_uses_same_config_and_cached_router() -> None:
    worker = importlib.import_module("biominer.detection.embeddinggemma2_worker")
    module = _module()
    native, _ = _router()
    image = _image()
    import base64

    request = {"protocol": module.EMBEDDINGGEMMA2_PROTOCOL_VERSION, "request_id": 1, "operation": "hello", "config": native.config.as_dict()}
    route = {"protocol": module.EMBEDDINGGEMMA2_PROTOCOL_VERSION, "request_id": 2, "operation": "route", "images": [{"width": image.width, "height": image.height, "mode": "RGB", "data_b64": base64.b64encode(image.data).decode("ascii")} ]}
    source = io.StringIO(json.dumps(request) + "\n" + json.dumps(route) + "\n")
    output = io.StringIO()

    worker.serve(source, output, router_factory=lambda **kwargs: native)
    messages = [json.loads(line) for line in output.getvalue().splitlines()]

    assert messages[0]["metadata"]["fingerprint"] == native.fingerprint
    assert messages[1]["request_id"] == 2
    assert len(messages[1]["results"]) == 1
    assert messages[1]["results"][0]["image_content_hash"] == decoded_image_content_hash(image)


@pytest.mark.parametrize("payload", [
    {"width": True, "height": 2, "mode": "RGB", "data_b64": ""},
    {"width": 3, "height": 2, "mode": "RGBA", "data_b64": ""},
    {"width": 3, "height": 2, "mode": "RGB", "data_b64": "!invalid!"},
    {"width": 3, "height": 2, "mode": "RGB", "data_b64": ""},
])
def test_worker_reports_invalid_rgb_payloads(payload) -> None:
    worker = importlib.import_module("biominer.detection.embeddinggemma2_worker")
    module = _module()
    native, _ = _router()
    hello = {"protocol": module.EMBEDDINGGEMMA2_PROTOCOL_VERSION, "request_id": 1, "operation": "hello", "config": native.config.as_dict()}
    route = {"protocol": module.EMBEDDINGGEMMA2_PROTOCOL_VERSION, "request_id": 2, "operation": "route", "images": [payload]}
    output = io.StringIO()

    worker.serve(io.StringIO(json.dumps(hello) + "\n" + json.dumps(route) + "\n"), output, router_factory=lambda **kwargs: native)
    response = json.loads(output.getvalue().splitlines()[1])

    assert "error" in response
    assert "results" not in response


def test_worker_rejects_duplicate_request_json_keys() -> None:
    worker = importlib.import_module("biominer.detection.embeddinggemma2_worker")
    module = _module()
    request = '{"protocol":' + json.dumps(module.EMBEDDINGGEMMA2_PROTOCOL_VERSION) + ',"request_id":1,"operation":"hello","operation":"load"}\n'
    output = io.StringIO()

    worker.serve(io.StringIO(request), output)
    response = json.loads(output.getvalue())

    assert "duplicate" in response["error"]
