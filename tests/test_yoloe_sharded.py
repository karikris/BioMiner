from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl

from biominer.detection.detector_base import DecodedImage, DetectionCandidate
from biominer.detection.yoloe26_detector import yoloe26_prompt_set_fingerprint
from biominer.flickr_fetch import yoloe_pilot, yoloe_sharded
from biominer.flickr_fetch.yoloe_sharded import (
    ShardedYoloeConfig,
    prepare_yoloe_shards,
    run_sharded_yoloe,
)


class _FakeSidecar:
    model_id = "fake-yoloe"
    model_version = "test"
    checkpoint = "fake.pt"

    def __init__(self, **kwargs) -> None:
        self.prompt_classes = tuple(kwargs["prompt_classes"])
        self.prompt_set_fingerprint = yoloe26_prompt_set_fingerprint(self.prompt_classes)
        self.closed = False
        self.batch_calls = 0

    def detect_batch(self, images: list[DecodedImage]) -> list[list[DetectionCandidate]]:
        self.batch_calls += 1
        return [[DetectionCandidate("insect_like", 0.9, (0, 0, 1, 1))] for _ in images]

    def close(self) -> None:
        self.closed = True


class _FailingSidecar(_FakeSidecar):
    def detect_batch(self, images: list[DecodedImage]) -> list[list[DetectionCandidate]]:
        self.batch_calls += 1
        raise RuntimeError("sidecar exited")


def _source_db(path: Path, count: int) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """CREATE TABLE source_records (
              source TEXT, flickr_photo_id TEXT, image_url TEXT, image_url_kind TEXT,
              source_record_hash TEXT, query_term TEXT, query_language TEXT,
              query_field TEXT, created_at TEXT)"""
        )
        conn.executemany(
            "INSERT INTO source_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "flickr",
                    str(index),
                    f"https://example.test/{index}.jpg",
                    "large",
                    f"hash-{index}",
                    "butterfly",
                    "en",
                    "text",
                    "2026-07-22",
                )
                for index in range(count)
            ],
        )


def test_shard_registers_are_balanced_disjoint_and_immutable(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    _source_db(source, 9)
    config = ShardedYoloeConfig(
        state_db=source,
        output_dir=tmp_path / "output",
        reports_dir=tmp_path / "reports",
        expected_images=9,
        shard_count=4,
        sample_seed="test-seed",
    )

    prepared = prepare_yoloe_shards(config)

    assert prepared.shard_sizes == (3, 2, 2, 2)
    identities: list[set[str]] = []
    for shard_index, path in enumerate(prepared.shard_paths):
        frame = pl.read_parquet(path / "sample_register.parquet")
        identities.append(set(frame["flickr_photo_id"].to_list()))
        assert frame["shard_index"].unique().to_list() == [shard_index]
    assert set().union(*identities) == {str(index) for index in range(9)}
    assert all(not left.intersection(right) for index, left in enumerate(identities) for right in identities[index + 1 :])
    first = prepared.sample_path.read_bytes()

    resumed = prepare_yoloe_shards(config)

    assert resumed.sample_path.read_bytes() == first
    assert resumed.source_snapshot_fingerprint == prepared.source_snapshot_fingerprint


def test_sharded_run_uses_four_exact_prompt_models_and_merges_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.sqlite"
    _source_db(source, 12)
    models: list[_FakeSidecar] = []

    def detector_factory(**kwargs) -> _FakeSidecar:
        detector = _FakeSidecar(**kwargs)
        models.append(detector)
        return detector

    monkeypatch.setattr(yoloe_sharded, "YoloE26SidecarObjectDetector", detector_factory)
    monkeypatch.setattr(
        yoloe_pilot,
        "_download_decode",
        lambda row, config: DecodedImage(1, 1, "RGB", b"\0\0\0"),
    )
    config = ShardedYoloeConfig(
        state_db=source,
        output_dir=tmp_path / "output",
        reports_dir=tmp_path / "reports",
        expected_images=12,
        shard_count=4,
        checkpoint="fake.pt",
        prompt_classes=("insect",),
    )

    result = run_sharded_yoloe(config)

    merged = pl.read_parquet(result.results_path)
    assert result.report["status"] == "complete"
    assert result.report["counts"]["classified"] == 12
    assert result.report["runtime"]["persistent_models"] == 4
    assert result.report["runtime"]["total_download_workers"] == 16
    assert result.report["runtime"]["prompt_classes"] == ["insect"]
    assert merged.height == 12
    assert merged["flickr_photo_id"].n_unique() == 12
    assert len(models) == 4
    assert all(model.prompt_classes == ("insect",) and model.closed for model in models)
    assert all((config.output_dir / f"shard_{index:03d}" / "pilot_state.sqlite").exists() for index in range(4))

    first_pass_calls = sum(model.batch_calls for model in models)
    resumed = run_sharded_yoloe(config)

    assert pl.read_parquet(resumed.results_path).height == 12
    assert sum(model.batch_calls for model in models) == first_pass_calls


def test_sharded_run_restarts_a_failed_sidecar_without_publishing_negatives(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.sqlite"
    _source_db(source, 3)
    models: list[_FakeSidecar] = []

    def detector_factory(**kwargs) -> _FakeSidecar:
        detector = _FailingSidecar(**kwargs) if not models else _FakeSidecar(**kwargs)
        models.append(detector)
        return detector

    monkeypatch.setattr(yoloe_sharded, "YoloE26SidecarObjectDetector", detector_factory)
    monkeypatch.setattr(
        yoloe_pilot,
        "_download_decode",
        lambda row, config: DecodedImage(1, 1, "RGB", b"\0\0\0"),
    )
    config = ShardedYoloeConfig(
        state_db=source,
        output_dir=tmp_path / "output",
        reports_dir=tmp_path / "reports",
        expected_images=3,
        shard_count=1,
        checkpoint="fake.pt",
        max_attempts=3,
    )

    result = run_sharded_yoloe(config)

    assert len(models) == 2
    assert all(model.closed for model in models)
    assert result.report["counts"]["classified"] == 3
    assert result.report["counts"]["terminal_operational_failures"] == 0
