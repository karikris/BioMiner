import json
from argparse import Namespace

import httpx
import numpy as np
import polars as pl

from biominer.embeddinggemma.runtime import RuntimeConfig


def test_text_cache_resume_preserves_exact_float32_values(tmp_path):
    from biominer.embeddinggemma.runtime import normalize_vectors
    from biominer.embeddinggemma.workflow import text_embeddings

    class Encoder:
        config = RuntimeConfig(device="cpu")

        def encode_text(self, sentences):
            values = (
                np.random.default_rng(1)
                .normal(size=(len(sentences), 768))
                .astype(np.float32)
            )
            return normalize_vectors(values)

    encoder = Encoder()
    labels = [
        {"label_id": f"label_{i}", "text": f"A subject {i}", "group": "Other Animals"}
        for i in range(392)
    ]
    first = text_embeddings(encoder, labels, tmp_path)
    second = text_embeddings(encoder, labels, tmp_path)
    np.testing.assert_array_equal(first, second)


def test_early_preparation_failure_has_run_receipt(tmp_path):
    import pytest

    from biominer.embeddinggemma.workflow import run_experiment

    args = Namespace(
        output_dir=tmp_path / "experiment", label_bank=tmp_path / "missing.md"
    )
    with pytest.raises(FileNotFoundError):
        run_experiment(args)
    receipt = json.loads((args.output_dir / "run.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["current_stage"] == "prepare"


def test_standalone_inputs_must_match_prepared_manifest(tmp_path):
    import pytest

    from biominer.embeddinggemma.common import atomic_json, file_sha256
    from biominer.embeddinggemma.prepare import SCHEMA_VERSION
    from biominer.embeddinggemma.workflow import read_prepared

    pl.DataFrame({"item_id": ["original"]}).write_parquet(tmp_path / "items.parquet")
    pl.DataFrame({"item_id": ["original"]}).write_parquet(
        tmp_path / "associations.parquet"
    )
    atomic_json(
        tmp_path / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "artifacts": {
                name: {"sha256": file_sha256(tmp_path / name)}
                for name in ("items.parquet", "associations.parquet")
            },
        },
    )
    pl.DataFrame({"item_id": ["changed"]}).write_parquet(tmp_path / "items.parquet")
    with pytest.raises(ValueError, match="checksum"):
        read_prepared(tmp_path)


def test_concurrent_prepare_publication_is_rejected(tmp_path):
    import pytest

    from biominer.embeddinggemma.common import run_lock
    from biominer.embeddinggemma.prepare import prepare

    with run_lock(tmp_path), pytest.raises(RuntimeError, match="worker"):
        prepare([], tmp_path)


def test_concurrent_score_publication_is_rejected(tmp_path):
    import pytest

    from biominer.embeddinggemma.common import run_lock
    from biominer.embeddinggemma.scoring import score_configuration

    empty = pl.DataFrame()
    with run_lock(tmp_path), pytest.raises(RuntimeError, match="worker"):
        score_configuration(
            empty,
            empty,
            empty,
            [],
            [],
            tmp_path,
            budget=280,
            dimension=256,
            provenance={},
        )


def test_cohort_media_snapshot_freezes_failures_for_resume(tmp_path, monkeypatch):
    from biominer.embeddinggemma import media as media_module
    from biominer.embeddinggemma.workflow import cohort_media

    items = pl.DataFrame(
        {"item_id": ["missing"], "image_url": ["https://example.test/missing"]}
    )
    failed = pl.DataFrame(
        {
            "item_id": ["missing"],
            "status": ["unavailable"],
            "image_hash": [None],
            "local_path": [None],
        }
    )
    calls = []

    def cache(*args, **kwargs):
        calls.append(1)
        return failed

    monkeypatch.setattr(media_module, "cache_items", cache)
    first = cohort_media(items, tmp_path / "cache", tmp_path, "cache", workers=1)
    second = cohort_media(items, tmp_path / "cache", tmp_path, "cache", workers=1)
    assert len(calls) == 1
    assert first.equals(second)


def test_cache_audit_excludes_cross_group_duplicate_bytes(tmp_path):
    from biominer.embeddinggemma.workflow import audit_cache

    items = pl.DataFrame(
        {
            "item_id": ["a", "b", "c"],
            "cache_group": ["butterfly", "non_biological", "butterfly"],
        }
    )
    media = pl.DataFrame(
        {
            "item_id": ["a", "b", "c"],
            "status": ["ok"] * 3,
            "image_hash": ["same", "same", "different"],
            "bytes_downloaded": [10] * 3,
        }
    )
    selected, audited = audit_cache(items, media, tmp_path, requested=2)
    assert selected["item_id"].to_list() == ["c"]
    assert audited["item_id"].to_list() == ["c"]
    receipt = json.loads((tmp_path / "cache-audit.json").read_text())
    assert receipt["cross_group_content_conflicts"] == 1
    assert receipt["groups"]["butterfly"]["shortfall"] == 1


def test_run_stages_reuse_embeddings_and_never_encode_four_times(tmp_path, monkeypatch):
    from io import BytesIO

    from PIL import Image

    from biominer.embeddinggemma import media as media_module
    from biominer.embeddinggemma.workflow import run_experiment

    bank = tmp_path / "labels.md"
    bank.write_text(
        "## Butterfly Subject (1)\n- `real_butterfly_tiny` — A small real butterfly\n"
        "## Butterfly Depictions (1)\n- `butterfly_cake` — A butterfly cake\n"
        "## Other Animals (390)\n"
        + "".join(f"- `animal_{i}` — An animal {i}\n" for i in range(390))
    )
    inputs = tmp_path / "input.parquet"
    pl.DataFrame(
        {
            "source": ["flickr", "flickr"],
            "flickr_photo_id": ["one", "two"],
            "image_url": ["https://example.test/one", "https://example.test/two"],
            "first_round_review_count": [1, 0],
            "first_round_reviews_json": ['[{"result":"adult_butterfly"}]', "[]"],
        }
    ).write_parquet(inputs)

    def transport(request):
        buffer = BytesIO()
        Image.new(
            "RGB", (20, 20), "green" if request.url.path == "/one" else "blue"
        ).save(buffer, format="PNG")
        return httpx.Response(
            200, content=buffer.getvalue(), headers={"content-type": "image/png"}
        )

    client = httpx.Client(transport=httpx.MockTransport(transport))
    original_cache = media_module.cache_items
    monkeypatch.setattr(
        media_module,
        "cache_items",
        lambda items, root, **kwargs: original_cache(
            items, root, http_client=client, **kwargs
        ),
    )
    monkeypatch.setattr(
        "biominer.embeddinggemma.workflow.runtime_check",
        lambda *a, **kw: {"status": "fixture_only"},
    )

    class Encoder:
        config = RuntimeConfig(device="cpu")
        text_calls = 0

        def __init__(self):
            self.calls = {280: 0, 1120: 0}

        def encode_images(self, paths, budget):
            self.calls[budget] += len(paths)
            return np.ones((len(paths), 768), dtype=np.float32)

        def encode_text(self, sentences):
            self.text_calls += 1
            return np.ones((len(sentences), 768), dtype=np.float32)

        def metrics(self):
            return {"device": "cpu", "dtype": "float32"}

    encoder = Encoder()
    args = Namespace(
        input=[inputs],
        output_dir=tmp_path / "experiment",
        label_bank=bank,
        cache_dir=tmp_path / "cache",
        seed="fixture",
        sample_per_group=1,
        revision=encoder.config.revision,
        device="cpu",
        dtype="float32",
        model_cache=tmp_path / "model",
        cohort="all",
        limit=None,
        batch_size=4,
        download_workers=1,
    )
    first = run_experiment(args, encoder=encoder)
    second = run_experiment(args, encoder=encoder)
    assert first["status"] == second["status"] == "complete"
    assert encoder.calls == {280: 2, 1120: 2}
    assert encoder.text_calls == 1
    assert len(list((args.output_dir / "scores").glob("*/*/manifest.json"))) == 12
    assert (
        json.loads((args.output_dir / "run.json").read_text())["last_complete_cohort"]
        == "all"
    )
    client.close()
