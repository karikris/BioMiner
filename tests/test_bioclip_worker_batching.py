from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import types

from biominer.bioclip import bioclip_worker


def test_loaded_model_encodes_image_batch_once_for_label_sets(monkeypatch) -> None:  # noqa: ANN001 - pytest fixture.
    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = FakeImageModule()
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)

    fake_torch = FakeTorch()
    fake_model = FakeModel()
    loaded = bioclip_worker._LoadedBioClipModel(  # noqa: SLF001 - worker batching contract.
        model=fake_model,
        preprocess=lambda image: FakeTensor(f"preprocessed:{image.path}"),
        tokenizer=FakeTokenizer(),
        torch=fake_torch,
        device="cuda",
        gpu_name="test-gpu",
    )

    scores = loaded.score_image_label_sets(
        [Path("/tmp/1.jpg"), Path("/tmp/2.jpg")],
        {
            "species": ["a photo of Papilio demoleus"],
            "triage": ["a photo of an adult butterfly"],
        },
    )

    assert fake_torch.stacked_lengths == [2]
    assert fake_model.encoded_image_batch_lengths == [2]
    assert len(scores["species"]) == 2
    assert len(scores["triage"]) == 2


def test_loaded_model_parallel_preprocess_preserves_image_order(monkeypatch) -> None:  # noqa: ANN001 - pytest fixture.
    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = FakeImageModule()
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)

    fake_torch = FakeTorch()
    loaded = bioclip_worker._LoadedBioClipModel(  # noqa: SLF001 - worker batching contract.
        model=FakeModel(),
        preprocess=lambda image: FakeTensor(f"preprocessed:{image.path}"),
        tokenizer=FakeTokenizer(),
        torch=fake_torch,
        device="cuda",
        gpu_name="test-gpu",
    )

    loaded.score_images(
        [Path("/tmp/1.jpg"), Path("/tmp/2.jpg"), Path("/tmp/3.jpg")],
        ["a photo of a butterfly"],
        preprocess_workers=2,
    )

    assert fake_torch.stacked_values == [
        "preprocessed:/tmp/1.jpg",
        "preprocessed:/tmp/2.jpg",
        "preprocessed:/tmp/3.jpg",
    ]


def test_loaded_model_chunks_large_text_feature_sets(monkeypatch) -> None:  # noqa: ANN001 - pytest fixture.
    monkeypatch.setenv("BIOMINER_BIOCLIP_TEXT_FEATURE_BATCH_SIZE", "2")
    fake_torch = FakeTorch()
    fake_model = FakeModel()
    loaded = bioclip_worker._LoadedBioClipModel(  # noqa: SLF001 - worker text-batching contract.
        model=fake_model,
        preprocess=lambda image: FakeTensor(f"preprocessed:{image.path}"),
        tokenizer=FakeTokenizer(),
        torch=fake_torch,
        device="mps",
        gpu_name="Apple MPS",
    )
    labels = ["label-1", "label-2", "label-3", "label-4", "label-5"]

    features = loaded._text_features(labels)  # noqa: SLF001 - focused worker contract test.
    cached = loaded._text_features(labels)  # noqa: SLF001 - focused worker contract test.

    assert features is cached
    assert features.label_count == 5
    assert fake_model.encoded_text_batch_lengths == [2, 2, 1]
    assert fake_torch.cat_lengths == [3]


def test_loaded_model_does_not_cache_one_off_text_embedding_batches(
    monkeypatch,
) -> None:  # noqa: ANN001 - pytest fixture.
    monkeypatch.setenv("BIOMINER_BIOCLIP_TEXT_FEATURE_BATCH_SIZE", "2")
    fake_model = FakeModel()
    loaded = bioclip_worker._LoadedBioClipModel(  # noqa: SLF001 - worker text-batching contract.
        model=fake_model,
        preprocess=lambda image: FakeTensor(f"preprocessed:{image.path}"),
        tokenizer=FakeTokenizer(),
        torch=FakeTorch(),
        device="mps",
        gpu_name="Apple MPS",
    )

    embeddings = loaded.text_embeddings(["label-1", "label-2", "label-3"])

    assert embeddings == [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    assert loaded._text_features_by_labels == {}  # noqa: SLF001 - focused worker cache contract.
    assert fake_model.encoded_text_batch_lengths == [2, 1]


def test_loaded_model_bounds_text_feature_cache_with_lru_eviction(monkeypatch) -> None:  # noqa: ANN001 - pytest fixture.
    monkeypatch.setenv("BIOMINER_BIOCLIP_TEXT_FEATURE_CACHE_ENTRIES", "2")
    fake_model = FakeModel()
    loaded = bioclip_worker._LoadedBioClipModel(  # noqa: SLF001 - worker cache contract.
        model=fake_model,
        preprocess=lambda image: FakeTensor(f"preprocessed:{image.path}"),
        tokenizer=FakeTokenizer(),
        torch=FakeTorch(),
        device="mps",
        gpu_name="Apple MPS",
    )

    loaded._text_features(["one"])  # noqa: SLF001 - focused worker cache contract.
    loaded._text_features(["two"])  # noqa: SLF001 - focused worker cache contract.
    loaded._text_features(["one"])  # noqa: SLF001 - mark the first entry most-recently-used.
    loaded._text_features(["three"])  # noqa: SLF001 - evicts the second entry.

    assert list(loaded._text_features_by_labels) == [("one",), ("three",)]  # noqa: SLF001 - focused worker cache contract.


def test_persistent_worker_clears_model_caches_on_replacement_and_shutdown(
    monkeypatch,
) -> None:  # noqa: ANN001 - pytest fixture.
    loaded_models: list[FakePersistentLoadedModel] = []

    def fake_load(
        *, model_name: str, checkpoint: str, device: str, image_resize_mode: str | None
    ) -> FakePersistentLoadedModel:
        loaded = FakePersistentLoadedModel(
            f"{model_name}:{checkpoint}:{device}:{image_resize_mode}"
        )
        loaded_models.append(loaded)
        return loaded

    requests = (
        {
            "model_name": "model-a",
            "checkpoint": "one",
            "device": "cpu",
            "text_labels": [],
        },
        {
            "model_name": "model-b",
            "checkpoint": "two",
            "device": "cpu",
            "text_labels": [],
        },
        {"shutdown": True},
    )
    monkeypatch.setattr(
        bioclip_worker._LoadedBioClipModel, "load", staticmethod(fake_load)
    )  # noqa: SLF001
    monkeypatch.setattr(bioclip_worker, "configure_hf_cache_env", lambda _path: None)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO("".join(json.dumps(request) + "\n" for request in requests)),
    )
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    bioclip_worker.run_persistent_worker()

    assert [loaded.key for loaded in loaded_models] == [
        "model-a:one:cpu:None",
        "model-b:two:cpu:None",
    ]
    assert [loaded.close_calls for loaded in loaded_models] == [1, 1]


def test_persistent_worker_keys_text_only_model_by_resize_mode_and_reports_it(
    monkeypatch,
) -> None:  # noqa: ANN001 - pytest fixture.
    loaded_models: list[FakePersistentLoadedModel] = []

    def fake_load(
        *, model_name: str, checkpoint: str, device: str, image_resize_mode: str | None
    ) -> FakePersistentLoadedModel:
        loaded = FakePersistentLoadedModel(
            f"{model_name}:{checkpoint}:{device}:{image_resize_mode}",
            image_resize_mode=image_resize_mode,
        )
        loaded_models.append(loaded)
        return loaded

    requests = (
        {
            "model_name": "model-a",
            "checkpoint": "one",
            "device": "cpu",
            "image_resize_mode": "shortest",
            "text_labels": [],
        },
        {
            "model_name": "model-a",
            "checkpoint": "one",
            "device": "cpu",
            "image_resize_mode": "shortest",
            "text_labels": [],
        },
        {
            "model_name": "model-a",
            "checkpoint": "one",
            "device": "cpu",
            "image_resize_mode": "longest",
            "text_labels": [],
        },
        {"shutdown": True},
    )
    stdout = io.StringIO()
    monkeypatch.setattr(
        bioclip_worker._LoadedBioClipModel, "load", staticmethod(fake_load)
    )  # noqa: SLF001
    monkeypatch.setattr(bioclip_worker, "configure_hf_cache_env", lambda _path: None)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO("".join(json.dumps(request) + "\n" for request in requests)),
    )
    monkeypatch.setattr(sys, "stdout", stdout)

    bioclip_worker.run_persistent_worker()

    assert [loaded.key for loaded in loaded_models] == [
        "model-a:one:cpu:shortest",
        "model-a:one:cpu:longest",
    ]
    assert [loaded.close_calls for loaded in loaded_models] == [1, 1]
    payloads = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [
        payload["image_resize_mode"] for payload in payloads if payload.get("ready")
    ] == ["shortest", "longest"]
    assert [
        payload["image_resize_mode"]
        for payload in payloads
        if "text_embeddings" in payload
    ] == [
        "shortest",
        "shortest",
        "longest",
    ]


class FakePersistentLoadedModel:
    device = "cpu"
    gpu_name = "test-cpu"

    def __init__(self, key: str, *, image_resize_mode: str | None = None) -> None:
        self.key = key
        self.image_resize_mode = image_resize_mode
        self.close_calls = 0

    def text_embeddings(self, _labels):  # noqa: ANN001, ANN201 - fake worker protocol.
        return []

    def close(self) -> None:
        self.close_calls += 1


class FakeTorch:
    def __init__(self) -> None:
        self.stacked_lengths: list[int] = []
        self.stacked_values: list[str] = []
        self.cat_lengths: list[int] = []

    def no_grad(self):  # noqa: ANN201 - mirrors torch context manager.
        return self

    def __enter__(self):  # noqa: ANN204 - mirrors torch context manager.
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001 - mirrors context manager.
        return None

    def stack(self, tensors):  # noqa: ANN001, ANN201 - test fake.
        self.stacked_lengths.append(len(tensors))
        self.stacked_values = [tensor.value for tensor in tensors]
        return FakeBatch(tensors)

    def cat(self, tensors, dim=0):  # noqa: ANN001, ANN201 - mirrors torch.cat.
        assert dim == 0
        parts = list(tensors)
        self.cat_lengths.append(len(parts))
        return FakeTextFeatures(sum(part.label_count for part in parts))


class FakeModel:
    def __init__(self) -> None:
        self.encoded_image_batch_lengths: list[int] = []
        self.encoded_text_batch_lengths: list[int] = []

    def encode_image(self, batch):  # noqa: ANN001, ANN201 - test fake.
        self.encoded_image_batch_lengths.append(len(batch.items))
        return FakeFeatureBatch(len(batch.items))

    def encode_text(self, text):  # noqa: ANN001, ANN201 - test fake.
        self.encoded_text_batch_lengths.append(len(text.labels))
        return FakeTextFeatures(len(text.labels))


class FakeTokenizer:
    def __call__(self, labels):  # noqa: ANN001, ANN204 - test fake.
        return FakeTokenized(labels)


class FakeImageModule:
    @staticmethod
    def open(path):  # noqa: ANN001, ANN205 - mirrors PIL.Image.open.
        return FakeImage(path)


class FakeImage:
    def __init__(self, path) -> None:  # noqa: ANN001 - test fake.
        self.path = path

    def convert(self, mode):  # noqa: ANN001, ANN201 - mirrors PIL image.
        return self


class FakeTokenized:
    def __init__(self, labels) -> None:  # noqa: ANN001 - test fake.
        self.labels = labels

    def to(self, device):  # noqa: ANN001, ANN201 - test fake.
        return self


class FakeTensor:
    def __init__(self, value: str) -> None:
        self.value = value


class FakeBatch:
    def __init__(self, items) -> None:  # noqa: ANN001 - test fake.
        self.items = items

    def to(self, device):  # noqa: ANN001, ANN201 - test fake.
        return self


class FakeFeatureBatch:
    def __init__(self, count: int) -> None:
        self.count = count

    def norm(self, dim, keepdim):  # noqa: ANN001, ANN201 - mirrors torch tensor.
        return self

    def __truediv__(self, other):  # noqa: ANN001, ANN204 - mirrors torch tensor.
        return self

    def __rmul__(self, other):  # noqa: ANN001, ANN204 - mirrors torch tensor.
        return self

    def __matmul__(self, other):  # noqa: ANN001, ANN204 - mirrors torch tensor.
        return FakeLogits(self.count)


class FakeTextFeatures:
    def __init__(self, label_count: int = 1) -> None:
        self.label_count = label_count

    @property
    def T(self):  # noqa: N802 - mirrors torch tensor API.
        return self

    def __iter__(self):  # noqa: ANN204 - mirrors tensor row iteration.
        return iter(FakeEmbeddingRow() for _index in range(self.label_count))

    def norm(self, dim, keepdim):  # noqa: ANN001, ANN201 - mirrors torch tensor.
        return self

    def __truediv__(self, other):  # noqa: ANN001, ANN204 - mirrors torch tensor.
        return self


class FakeEmbeddingRow:
    def detach(self):  # noqa: ANN201 - mirrors torch tensor.
        return self

    def cpu(self):  # noqa: ANN201 - mirrors torch tensor.
        return self

    def tolist(self) -> list[float]:
        return [1.0, 0.0]


class FakeLogits:
    def __init__(self, count: int) -> None:
        self.count = count

    def __rmul__(self, other):  # noqa: ANN001, ANN204 - mirrors torch tensor.
        return self

    def softmax(self, dim):  # noqa: ANN001, ANN201 - mirrors torch tensor.
        return [FakeProbabilities() for _ in range(self.count)]


class FakeProbabilities:
    def __getitem__(self, index):  # noqa: ANN001, ANN204 - mirrors torch tensor.
        return FakeProbability(1.0)


class FakeProbability:
    def __init__(self, value: float) -> None:
        self.value = value

    def detach(self):  # noqa: ANN201 - mirrors torch tensor.
        return self

    def cpu(self):  # noqa: ANN201 - mirrors torch tensor.
        return self

    def __float__(self) -> float:
        return self.value
