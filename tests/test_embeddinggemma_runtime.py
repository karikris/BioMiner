import numpy as np
import polars as pl
import pytest


def test_official_adapter_uses_float32_no_audio_and_explicit_budget(
    tmp_path, monkeypatch
):
    import sys
    from types import SimpleNamespace

    from PIL import Image

    from biominer.embeddinggemma.runtime import RuntimeConfig, SentenceEncoder

    calls = {}
    dtype = object()

    class Model:
        def __init__(self, model_id, **kwargs):
            calls["load"] = kwargs

        def eval(self):
            pass

        def float(self):
            calls["force_float32"] = True

        def parameters(self):
            return []

        def __getitem__(self, key):
            return SimpleNamespace(
                auto_model=SimpleNamespace(config=SimpleNamespace(audio_config=None))
            )

        def encode(self, inputs, **kwargs):
            calls["encode"] = kwargs
            return np.ones((len(inputs), 768), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32=dtype))
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model)
    )
    path = tmp_path / "synthetic.png"
    Image.new("RGB", (20, 20)).save(path)
    encoder = SentenceEncoder(RuntimeConfig(device="cpu"))
    result = encoder.encode_images([path], 1120)
    assert result.shape == (1, 768)
    assert calls["load"]["config_kwargs"] == {"audio_config": None}
    assert calls["load"]["model_kwargs"]["torch_dtype"] is dtype
    assert calls["force_float32"]
    assert calls["encode"]["prompt"] == ""
    assert calls["encode"]["processing_kwargs"] == {"image": {"max_soft_tokens": 1120}}


def test_failed_runtime_recheck_invalidates_previous_pass(tmp_path, monkeypatch):
    import json

    from biominer.embeddinggemma.common import atomic_json
    from biominer.embeddinggemma.workflow import runtime_check

    atomic_json(tmp_path / "receipt.json", {"status": "passed"})

    def fail(*args, **kwargs):
        raise ValueError("invalid vector")

    monkeypatch.setattr("biominer.embeddinggemma.workflow._runtime_check", fail)
    with pytest.raises(ValueError):
        runtime_check(None, tmp_path)
    assert json.loads((tmp_path / "receipt.json").read_text())["status"] == "failed"


def test_float16_is_rejected_before_model_import():
    from biominer.embeddinggemma.runtime import RuntimeConfig

    with pytest.raises(ValueError, match="float32"):
        RuntimeConfig(dtype="float16")


def test_truncated_vectors_are_normalized():
    from biominer.embeddinggemma.runtime import normalize_vectors

    vectors = np.ones((2, 768), dtype=np.float32)
    compact = normalize_vectors(vectors, 256)
    assert compact.shape == (2, 256)
    np.testing.assert_allclose(np.linalg.norm(compact, axis=1), 1, atol=1e-6)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0])
def test_invalid_vectors_fail_closed(value):
    from biominer.embeddinggemma.runtime import normalize_vectors

    with pytest.raises(ValueError):
        normalize_vectors(np.full((1, 768), value), 256)


def test_checkpoint_resume_reuses_content_across_items(tmp_path):
    from biominer.embeddinggemma.encode import encode_media, read_embeddings
    from biominer.embeddinggemma.runtime import RuntimeConfig

    class Encoder:
        config = RuntimeConfig(device="cpu")
        calls = 0

        def encode_images(self, paths, budget):
            assert budget == 1120
            self.calls += len(paths)
            return np.ones((len(paths), 768), dtype=np.float32)

        def metrics(self):
            return {"device": "cpu"}

    image = tmp_path / "image"
    image.write_bytes(b"fixture")
    from biominer.embeddinggemma.common import file_sha256

    media = pl.DataFrame(
        {
            "item_id": ["a", "b"],
            "status": ["ok", "ok"],
            "image_hash": [file_sha256(image)] * 2,
            "local_path": [str(image)] * 2,
        }
    )
    encoder = Encoder()
    out = tmp_path / "embeddings"
    first = encode_media(media, out, encoder=encoder, budget=1120, batch_size=2)
    second = encode_media(media, out, encoder=encoder, budget=1120)
    assert first["images_encoded"] == 1
    assert second["images_encoded"] == 0
    assert encoder.calls == 1
    assert read_embeddings(out).height == 1


def test_changed_model_cannot_resume_checkpoint(tmp_path):
    from biominer.embeddinggemma.encode import encode_media
    from biominer.embeddinggemma.runtime import RuntimeConfig

    class Encoder:
        config = RuntimeConfig(device="cpu")

        def metrics(self):
            return {}

    empty = pl.DataFrame(
        schema={"status": pl.String, "image_hash": pl.String, "local_path": pl.String}
    )
    encoder = Encoder()
    encode_media(empty, tmp_path, encoder=encoder, budget=280)
    encoder.config = RuntimeConfig(device="cpu", revision="a" * 40)
    with pytest.raises(ValueError, match="fingerprint"):
        encode_media(empty, tmp_path, encoder=encoder, budget=280)


def test_invalid_inference_leaves_explicit_failed_checkpoint(tmp_path):
    import json

    from biominer.embeddinggemma.common import file_sha256
    from biominer.embeddinggemma.encode import encode_media
    from biominer.embeddinggemma.runtime import RuntimeConfig

    class Encoder:
        config = RuntimeConfig(device="cpu")

        def encode_images(self, paths, budget):
            return np.zeros((len(paths), 768), dtype=np.float32)

        def metrics(self):
            return {}

    image = tmp_path / "image"
    image.write_bytes(b"image")
    media = pl.DataFrame(
        {
            "item_id": ["a"],
            "status": ["ok"],
            "image_hash": [file_sha256(image)],
            "local_path": [str(image)],
        }
    )
    with pytest.raises(ValueError):
        encode_media(media, tmp_path / "output", encoder=Encoder(), budget=280)
    manifest = json.loads((tmp_path / "output" / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "ValueError"
