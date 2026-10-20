"""Run identical synthetic fixtures against baseline and candidate environments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import httpx
import numpy as np
from PIL import Image
import polars as pl

from biominer.ml.classifiers import (
    ClassifierTrainingConfig,
    materialize_classifier_feature_matrix,
    train_frozen_embedding_classifiers,
)
from biominer.ml.persistence import load_frozen_classifier, write_frozen_classifier
from biominer.references.deduplication import compute_reference_perceptual_hash
from biominer.references.downloader import download_reference_media
from biominer.storage.parquet import (
    ParquetRowSource,
    iter_parquet_batches,
    write_parquet,
    write_parquet_batches,
)

# Only deterministic fixture constructors are reused. Every operation under test
# goes through the public production interface.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_reference_downloader import _MemoryStorage, _NOW, _config, _frames
from test_frozen_embedding_classifiers import _training_frame
from test_few_shot_training_features import TARGET, COMPETITOR


def digest(data):
    return hashlib.sha256(data).hexdigest()


def caught(operation):
    try:
        result = operation()
    except Exception as exc:
        return {"exception": type(exc).__name__}
    raise AssertionError(f"Expected rejection; received {result!r}")


def create_fixtures(root):
    root.mkdir(parents=True, exist_ok=True)
    pixels = bytes(
        component
        for y in range(21)
        for x in range(29)
        for component in ((x * 11 + y * 7) % 256, (x * x + y * 19) % 256, (x * 3 + y * 29) % 256)
    )
    with Image.frombytes("RGB", (29, 21), pixels) as image:
        for fmt, suffix in (("JPEG", "jpg"), ("PNG", "png"), ("TIFF", "tif"), ("WEBP", "webp")):
            path = root / f"pattern.{suffix}"
            assert not path.exists(), path
            image.save(path, format=fmt)
    (root / "malformed.png").write_bytes(b"\x89PNG\r\n\x1a\ninvalid")


def image_cases(fixtures, work):
    rows = {}
    for suffix, content_type in (
        ("jpg", "image/jpeg"), ("png", "image/png"),
        ("tif", "image/tiff"), ("webp", "image/webp"),
    ):
        path = fixtures / f"pattern.{suffix}"
        data = path.read_bytes()
        with Image.open(path) as image:
            image.load()
            assert getattr(image, "n_frames", 1) == 1
            decoded = {
                "format": image.format, "mode": image.mode, "size": list(image.size),
                "rgb_sha256": digest(image.convert("RGB").tobytes()),
                "perceptual_hash": compute_reference_perceptual_hash(image),
            }
        candidates, selections = _frames()
        storage = _MemoryStorage()
        def handler(request):
            return httpx.Response(200, headers={"Content-Type": content_type}, content=data)
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            result = download_reference_media(
                selections, candidates, storage=storage,
                output_prefix="s3://paired-fixture/bank",
                config=_config(temporary_directory=work),
                http_client=client, run_id=f"paired-{suffix}", now=lambda: _NOW,
            )
        row = result.media_objects.to_dicts()[0]
        assert row["decode_status"] == "valid", row
        assert storage.files[row["source_object_uri"]] == data
        assert row["perceptual_hash"] == decoded["perceptual_hash"]
        rows[suffix] = {
            "source_sha256": digest(data), "decoded": decoded,
            "downloader": {key: row[key] for key in (
                "decode_status", "content_type", "source_byte_count",
                "decoded_width", "decoded_height", "sha256", "perceptual_hash",
                "source_object_uri", "quarantine_reason",
            )},
            "storage_operations": storage.operations,
            "stored_files": {key: digest(value) for key, value in storage.files.items()},
        }
    malformed = fixtures / "malformed.png"
    missing = fixtures / "missing.png"
    rows["rejections"] = {
        "malformed": caught(lambda: Image.open(malformed)),
        "missing": caught(lambda: Image.open(missing)),
        "null": caught(lambda: compute_reference_perceptual_hash(None)),
    }
    candidates, selections = _frames()
    storage = _MemoryStorage()
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(
            200, headers={"Content-Type": "image/png"}, content=malformed.read_bytes()
        )
    )) as client:
        rejected = download_reference_media(
            selections, candidates, storage=storage,
            output_prefix="s3://paired-fixture/rejected",
            config=_config(temporary_directory=work), http_client=client,
            run_id="paired-malformed", now=lambda: _NOW,
        )
    row = rejected.media_objects.to_dicts()[0]
    assert row["decode_status"] != "valid"
    assert not storage.files
    rows["malformed_downloader"] = {
        key: row[key] for key in ("decode_status", "quarantine_reason", "source_object_uri")
    }
    rows["malformed_downloader"]["storage_operations"] = storage.operations
    rows["temporary_files_remaining"] = sorted(str(p.relative_to(work)) for p in work.rglob("*") if p.is_file())
    assert not rows["temporary_files_remaining"]
    return rows


def parquet_cases(work):
    schema = {"id": pl.Int64, "name": pl.String, "score": pl.Float64, "tags": pl.List(pl.String)}
    frame = pl.DataFrame(
        {"id": [1, 2, None], "name": ["a", "", None], "score": [1.5, None, -2.0], "tags": [["x", None], [], None]},
        schema=schema,
    )
    result = {}
    for name, batches in (("single", None), ("batches", [frame[:1], frame[1:]])):
        path = work / f"{name}.parquet"
        if batches is None:
            write_parquet(frame, path)
        else:
            write_parquet_batches(batches, path, schema=schema)
        restored = pl.read_parquet(path)
        bounded = list(ParquetRowSource(path, batch_size=1))
        assert frame.equals(restored)
        assert bounded == frame.to_dicts()
        result[name] = {
            "schema": {key: str(value) for key, value in restored.schema.items()},
            "rows": bounded, "physical_sha256": digest(path.read_bytes()),
            "overwrite_rejection": caught(lambda: write_parquet(frame, path, overwrite=False)),
        }
    path = work / "empty.parquet"
    write_parquet_batches([], path, schema=schema)
    empty = pl.read_parquet(path)
    assert empty.schema == frame.schema and empty.height == 0
    result["empty"] = {"schema": {k: str(v) for k, v in empty.schema.items()}, "rows": empty.to_dicts()}
    malformed = work / "malformed.parquet"
    malformed.write_bytes(b"not a parquet file")
    result["rejections"] = {
        "malformed": caught(lambda: list(iter_parquet_batches(malformed))),
        "missing": caught(lambda: list(iter_parquet_batches(work / "missing.parquet"))),
        "null": caught(lambda: list(iter_parquet_batches(None))),
        "invalid_batch_size": caught(lambda: list(iter_parquet_batches(path, batch_size=0))),
    }
    result["temporary_files_remaining"] = sorted(p.name for p in work.glob("*.tmp"))
    assert not result["temporary_files_remaining"]
    return result


def classifier_cases(work, source_sha):
    frame = _training_frame(
        task="binary_target_verifier", classes=(TARGET, COMPETITOR),
        fit_groups_per_class=4, model_selection_groups_per_class=2,
    )
    run = train_frozen_embedding_classifiers(
        frame, ClassifierTrainingConfig(
            target_task="binary_target_verifier", target_accepted_taxon_key=TARGET,
            route="adult_field", n_splits=2, random_seed=17,
        ),
    )
    result = {"selected_model_name": run.selected_model_name, "models": {}}
    for candidate in run.candidates:
        paths = write_frozen_classifier(
            run, work / candidate.model_name, model_name=candidate.model_name,
            preprocessing_fingerprint="sha256:" + "1" * 64,
            reference_bank_version="paired-fixture-v1",
            reference_bank_fingerprint="sha256:" + "2" * 64,
            git_sha=source_sha,
            created_at=datetime(2026, 10, 20, tzinfo=timezone.utc),
        )
        loaded = load_frozen_classifier(paths.directory, expected_classifier_fingerprint=paths.classifier_fingerprint)
        raw = materialize_classifier_feature_matrix(frame, candidate.feature_layout)
        margins = loaded.decision_function(raw)
        np.testing.assert_allclose(margins, candidate.pipeline.decision_function(raw), rtol=1e-12, atol=1e-12)
        assert loaded.predict(raw) == tuple(str(v) for v in candidate.pipeline.predict(raw))
        assert loaded.probability_calibrated is False
        assert not hasattr(loaded, "predict_proba")
        result["models"][candidate.model_name] = {
            "predictions": loaded.predict(raw), "margins": margins.tolist(),
            "class_labels": loaded.class_labels, "probability_calibrated": loaded.probability_calibrated,
            "classifier_fingerprint": paths.classifier_fingerprint,
            "manifest_sha256": digest(paths.manifest_path.read_bytes()),
            "arrays_sha256": digest(paths.arrays_path.read_bytes()),
            "null_features": caught(lambda: loaded.decision_function(None)),
            "wrong_width": caught(lambda: loaded.decision_function(np.zeros((2, raw.shape[1] + 1)))),
        }
    result["rejections"] = {
        "missing": caught(lambda: load_frozen_classifier(work / "missing")),
        "null": caught(lambda: load_frozen_classifier(None)),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--create-fixtures", action="store_true")
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    if len(args.source_sha) != 40 or any(c not in "0123456789abcdef" for c in args.source_sha):
        parser.error("source-sha must be an exact lowercase Git commit")
    if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in args.label):
        parser.error("label must contain lowercase letters, digits, underscores or hyphens")
    (args.root / "receipts").mkdir(parents=True, exist_ok=True)
    fixtures = args.root / "fixtures"
    if args.create_fixtures:
        create_fixtures(fixtures)
    work = args.root / f"{args.label}-behavior-work"
    work.mkdir()
    result = {"label": args.label, "versions": {name: version(name) for name in (
        "pillow", "pydantic", "numpy", "polars", "pyarrow", "duckdb", "scikit-learn", "skops", "psycopg",
    )}}
    for name, operation in (("images", lambda folder: image_cases(fixtures, folder)), ("parquet", parquet_cases), ("classifiers", lambda folder: classifier_cases(folder, args.source_sha))):
        folder = work / name
        folder.mkdir()
        result[name] = operation(folder)
    output = args.root / "receipts" / f"{args.label}-behavior.json"
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "status": "passed", "versions": result["versions"]}))


if __name__ == "__main__":
    main()
