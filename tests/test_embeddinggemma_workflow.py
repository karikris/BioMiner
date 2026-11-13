import json
from argparse import Namespace

import httpx
import numpy as np
import polars as pl
import pytest

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


def test_cache_audit_reports_explicit_sample_group_denominators(tmp_path):
    from biominer.embeddinggemma.workflow import audit_cache

    items = pl.DataFrame(
        {
            "item_id": ["recent", "earlier", "other"],
            "cache_group": ["butterfly", "butterfly", "non_biological"],
        }
    )
    media = pl.DataFrame(
        {
            "item_id": ["recent", "earlier", "other"],
            "status": ["ok", "unavailable", "ok"],
            "image_hash": ["one", None, "two"],
            "bytes_downloaded": [10, 0, 10],
        }
    )
    audit_cache(
        items,
        media,
        tmp_path,
        requested=500,
        group_requests={"butterfly": 2, "non_biological": 1},
    )
    groups = json.loads((tmp_path / "cache-audit.json").read_text())["groups"]
    assert groups["butterfly"] == {"selected": 1, "requested": 2, "shortfall": 1}
    assert groups["non_biological"] == {"selected": 1, "requested": 1, "shortfall": 0}
    assert groups["other_biological"] == {"selected": 0, "requested": 0, "shortfall": 0}


@pytest.mark.parametrize(
    "configurations,expected_budgets",
    [
        (None, [280, 1120]),
        (["1120-256", "560-256", "1120-768", "560-768"], [1120, 560]),
        (["1120-768", "560-768", "1120-256", "560-256"], [1120, 560]),
    ],
)
def test_run_stages_reuse_embeddings_and_never_encode_four_times(
    tmp_path, monkeypatch, configurations, expected_budgets
):
    from io import BytesIO

    from PIL import Image

    from biominer.embeddinggemma import media as media_module
    from biominer.embeddinggemma import workflow
    from biominer.embeddinggemma.workflow import run_experiment

    score_order = []
    original_score = workflow.score_cohort

    def score(*args, **kwargs):
        score_order.append((args[6], args[7]))
        return original_score(*args, **kwargs)

    monkeypatch.setattr(workflow, "score_cohort", score)

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
            self.calls = {budget: 0 for budget in expected_budgets}

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
        configuration=configurations,
        binary=configurations is not None,
    )
    first = run_experiment(args, encoder=encoder)
    second = run_experiment(args, encoder=encoder)
    assert first["status"] == second["status"] == "complete"
    assert encoder.calls == {budget: 2 for budget in expected_budgets}
    expected_order = workflow.run_configurations(configurations)
    assert score_order == expected_order * 6  # Three cohorts, then a full resume.
    assert encoder.text_calls == 1
    assert len(list((args.output_dir / "scores").glob("*/*/manifest.json"))) == 12
    if configurations is not None:
        assert (
            len(
                list(
                    (args.output_dir / "scores").glob(
                        "*/*/butterfly_classification.parquet"
                    )
                )
            )
            == 12
        )
    assert (
        json.loads((args.output_dir / "run.json").read_text())["last_complete_cohort"]
        == "all"
    )
    client.close()


@pytest.mark.parametrize(
    "values", [["560-256", "560-256"], ["999-256"], ["560-128"], ["bad"]]
)
def test_run_rejects_invalid_configuration_plan(values):
    from biominer.embeddinggemma.workflow import run_configurations

    with pytest.raises(ValueError):
        run_configurations(values)


def test_changed_configuration_plan_cannot_resume_existing_run(tmp_path):
    from biominer.embeddinggemma.common import atomic_json
    from biominer.embeddinggemma.workflow import run_experiment

    plan = {"configurations": [[280, 256], [1120, 256]], "binary": False}
    atomic_json(tmp_path / "configuration-plan.json", plan)
    args = Namespace(output_dir=tmp_path, configuration=["1120-256", "560-256"])
    with pytest.raises(ValueError, match="configuration plan changed"):
        run_experiment(args)
    assert json.loads((tmp_path / "configuration-plan.json").read_text()) == plan


def test_standalone_score_binary_flag_publishes_decision_artifact(
    tmp_path, monkeypatch
):
    from biominer.cli import build_parser
    from biominer.embeddinggemma import labels as label_module
    from biominer.embeddinggemma import media as media_module
    from biominer.embeddinggemma import runtime, workflow
    from biominer.embeddinggemma.common import atomic_json, file_sha256
    from biominer.embeddinggemma.encode import encode_media

    class Encoder:
        config = RuntimeConfig(device="cpu")

        def encode_images(self, paths, budget):
            vectors = np.zeros((len(paths), 768), dtype=np.float32)
            vectors[:, 0] = 1
            return vectors

        def encode_text(self, sentences):
            vectors = np.zeros((len(sentences), 768), dtype=np.float32)
            vectors[:, : len(sentences)] = np.eye(len(sentences))
            return vectors

        def metrics(self):
            return {"device": "cpu"}

    encoder = Encoder()
    image = tmp_path / "synthetic-image"
    image.write_bytes(b"fixture")
    media = pl.DataFrame(
        {
            "item_id": ["image"],
            "status": ["ok"],
            "local_path": [str(image)],
            "image_hash": [file_sha256(image)],
        }
    )
    embedding_dir = tmp_path / "embeddings"
    encode_media(media, embedding_dir, encoder=encoder, budget=560)
    prepared = tmp_path / "prepared"
    atomic_json(prepared / "manifest.json", {"fixture_only": True})
    bank = tmp_path / "labels.md"
    bank.write_text(
        "## Butterfly Subject (1)\n- `real_butterfly_closeup` — A butterfly\n"
        "## Plants Fungi (1)\n- `flower_closeup` — A flower\n"
    )
    original_load = label_module.load_labels
    monkeypatch.setattr(
        label_module, "load_labels", lambda path: original_load(path, expected_count=2)
    )
    monkeypatch.setattr(runtime, "SentenceEncoder", lambda *args, **kw: encoder)
    monkeypatch.setattr(
        workflow, "read_prepared", lambda path: pl.DataFrame({"item_id": ["image"]})
    )
    monkeypatch.setattr(media_module, "read_cached_items", lambda *args: media)
    output = tmp_path / "score"
    args = build_parser().parse_args(
        [
            "vision",
            "embeddinggemma",
            "score",
            "--prepared-dir",
            str(prepared),
            "--embedding-dir",
            str(embedding_dir),
            "--label-bank",
            str(bank),
            "--output-dir",
            str(output),
            "--budget",
            "560",
            "--dimension",
            "256",
            "--device",
            "cpu",
            "--binary",
        ]
    )
    workflow.execute(args)
    result = pl.read_parquet(output / "butterfly_classification.parquet")
    assert result["prediction"].item() == "butterfly"
